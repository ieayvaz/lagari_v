#include "lagari/capture/isaac_shm_capture.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <chrono>

namespace lagari {

// ============================================================================
// Constructor / Destructor
// ============================================================================

IsaacShmCapture::IsaacShmCapture(const CaptureConfig& config)
    : config_(config)
{
    // Default SHM config
    shm_config_.shm_name = "lagari_camera";
}

IsaacShmCapture::IsaacShmCapture(const CaptureConfig& config, const ShmConfig& shm_config)
    : config_(config)
    , shm_config_(shm_config)
{
}

IsaacShmCapture::~IsaacShmCapture() {
    stop();
    detach();
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool IsaacShmCapture::initialize(const Config& config) {
    // Parse SHM-specific config
    shm_config_.shm_name = config.get_string(
        "capture.isaac.shm_name", "lagari_camera");
    shm_config_.poll_interval_us = config.get_int(
        "capture.isaac.poll_interval_us", 100);
    shm_config_.auto_reconnect = config.get_bool(
        "capture.isaac.reconnect", true);
    shm_config_.reconnect_delay_ms = config.get_int(
        "capture.isaac.reconnect_delay_ms", 1000);

    return attach();
}

void IsaacShmCapture::start() {
    if (running_.load(std::memory_order_acquire)) {
        return;
    }

    if (!attached_.load(std::memory_order_acquire)) {
        LOG_ERROR("IsaacShmCapture: Not attached to shared memory");
        return;
    }

    should_stop_.store(false, std::memory_order_release);
    start_time_ = Clock::now();
    last_frame_time_ = start_time_;

    read_thread_ = std::thread(&IsaacShmCapture::read_loop, this);
    running_.store(true, std::memory_order_release);

    LOG_INFO("IsaacShmCapture started, reading from /dev/shm/{}", shm_config_.shm_name);
}

void IsaacShmCapture::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    should_stop_.store(true, std::memory_order_release);
    frame_cv_.notify_all();

    if (read_thread_.joinable()) {
        read_thread_.join();
    }

    running_.store(false, std::memory_order_release);
    LOG_INFO("IsaacShmCapture stopped");
}

bool IsaacShmCapture::is_running() const {
    return running_.load(std::memory_order_acquire);
}

// ============================================================================
// ICapture Implementation
// ============================================================================

FramePtr IsaacShmCapture::get_latest_frame() {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

FramePtr IsaacShmCapture::wait_for_frame(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(frame_mutex_);

    if (latest_frame_ && latest_frame_->metadata.frame_id > last_frame_id_) {
        last_frame_id_ = latest_frame_->metadata.frame_id;
        return latest_frame_;
    }

    auto current_id = latest_frame_ ? latest_frame_->metadata.frame_id : 0;
    
    frame_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, current_id]() {
        return (latest_frame_ && latest_frame_->metadata.frame_id > current_id) ||
               should_stop_.load(std::memory_order_acquire);
    });

    if (latest_frame_) {
        last_frame_id_ = latest_frame_->metadata.frame_id;
    }
    return latest_frame_;
}

void IsaacShmCapture::set_frame_callback(FrameCallback callback) {
    frame_callback_ = std::move(callback);
}

CaptureStats IsaacShmCapture::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool IsaacShmCapture::is_open() const {
    return attached_.load(std::memory_order_acquire);
}

bool IsaacShmCapture::set_resolution(uint32_t width, uint32_t height) {
    LOG_WARN("IsaacShmCapture: Resolution is controlled by Isaac Sim");
    config_.width = width;
    config_.height = height;
    return false;
}

bool IsaacShmCapture::set_framerate(uint32_t fps) {
    LOG_WARN("IsaacShmCapture: Framerate is controlled by Isaac Sim");
    config_.fps = fps;
    return false;
}

bool IsaacShmCapture::set_exposure(bool /* auto_exp */, float /* exposure_time */) {
    LOG_WARN("IsaacShmCapture: Exposure is controlled by Isaac Sim");
    return false;
}

// ============================================================================
// SHM-specific Methods
// ============================================================================

bool IsaacShmCapture::attach(const std::string& shm_name) {
    if (attached_.load(std::memory_order_acquire)) {
        detach();
    }

    std::string name = shm_name.empty() ? shm_config_.shm_name : shm_name;
    std::string shm_path = "/" + name;

    // Open shared memory
    shm_fd_ = shm_open(shm_path.c_str(), O_RDWR, 0666);
    if (shm_fd_ < 0) {
        LOG_ERROR("IsaacShmCapture: Failed to open /dev/shm/{}: {}", 
                  name, strerror(errno));
        return false;
    }

    // Get size
    struct stat sb;
    if (fstat(shm_fd_, &sb) < 0) {
        LOG_ERROR("IsaacShmCapture: Failed to stat shared memory: {}", strerror(errno));
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }
    shm_size_ = sb.st_size;

    // Memory map
    shm_ptr_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
        LOG_ERROR("IsaacShmCapture: Failed to mmap: {}", strerror(errno));
        close(shm_fd_);
        shm_fd_ = -1;
        shm_ptr_ = nullptr;
        return false;
    }

    // Validate header
    const auto* header = static_cast<const ShmHeader*>(shm_ptr_);
    if (header->magic != MAGIC) {
        LOG_ERROR("IsaacShmCapture: Invalid magic: 0x{:08X} (expected 0x{:08X})",
                  header->magic, MAGIC);
        detach();
        return false;
    }

    if (header->version != VERSION) {
        LOG_ERROR("IsaacShmCapture: Version mismatch: {} (expected {})",
                  header->version, VERSION);
        detach();
        return false;
    }

    // Update config from header
    config_.width = header->width;
    config_.height = header->height;
    
    shm_config_.shm_name = name;
    attached_.store(true, std::memory_order_release);

    LOG_INFO("IsaacShmCapture: Attached to /dev/shm/{} ({}x{}, {} bytes)",
             name, header->width, header->height, shm_size_);

    return true;
}

void IsaacShmCapture::detach() {
    attached_.store(false, std::memory_order_release);

    if (shm_ptr_ && shm_ptr_ != MAP_FAILED) {
        munmap(shm_ptr_, shm_size_);
        shm_ptr_ = nullptr;
    }

    if (shm_fd_ >= 0) {
        close(shm_fd_);
        shm_fd_ = -1;
    }

    shm_size_ = 0;
    LOG_INFO("IsaacShmCapture: Detached");
}

bool IsaacShmCapture::is_attached() const {
    return attached_.load(std::memory_order_acquire);
}

Duration IsaacShmCapture::get_frame_latency() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!latest_frame_) {
        return Duration::zero();
    }
    return Clock::now() - latest_frame_->metadata.timestamp;
}

// ============================================================================
// Read Loop
// ============================================================================

void IsaacShmCapture::read_loop() {
    LOG_DEBUG("IsaacShmCapture: Read thread started");

    while (!should_stop_.load(std::memory_order_acquire)) {
        if (!attached_.load(std::memory_order_acquire)) {
            // Try to reconnect
            if (shm_config_.auto_reconnect) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(shm_config_.reconnect_delay_ms));
                if (attach()) {
                    LOG_INFO("IsaacShmCapture: Reconnected");
                }
            } else {
                break;
            }
            continue;
        }

        if (!try_read_frame()) {
            // No new frame, poll again
            std::this_thread::sleep_for(
                std::chrono::microseconds(shm_config_.poll_interval_us));
        }
    }

    LOG_DEBUG("IsaacShmCapture: Read thread exiting");
}

bool IsaacShmCapture::try_read_frame() {
    if (!shm_ptr_) {
        return false;
    }

    // Read header atomically
    ShmHeader header;
    std::memcpy(&header, shm_ptr_, sizeof(header));

    // Check if write is complete (even sequence number)
    if (header.write_seq & 1) {
        return false;  // Write in progress
    }

    // Check if this is a new frame
    if (header.write_seq <= last_write_seq_) {
        return false;  // Already processed
    }

    // Validate header
    if (header.magic != MAGIC) {
        LOG_WARN("IsaacShmCapture: Invalid magic, detaching");
        detach();
        return false;
    }

    // Calculate expected size
    size_t frame_size = header.width * header.height * 3;  // BGR24
    size_t expected_size = HEADER_SIZE + frame_size;
    
    if (shm_size_ < expected_size) {
        LOG_WARN("IsaacShmCapture: Size mismatch ({} < {})", shm_size_, expected_size);
        return false;
    }

    // Read frame data
    const uint8_t* frame_data = static_cast<const uint8_t*>(shm_ptr_) + HEADER_SIZE;

    // Re-check sequence (ensure no write started during our read)
    ShmHeader header_after;
    std::memcpy(&header_after, shm_ptr_, sizeof(header_after));
    
    if (header_after.write_seq != header.write_seq) {
        // Write happened during our read, discard
        frames_skipped_++;
        return false;
    }

    // Process the frame
    FramePtr frame = process_shm_data(header, frame_data);
    
    if (frame) {
        // Update our read sequence
        last_write_seq_ = header.write_seq;
        
        // Update the shared memory read_seq (optional, for producer feedback)
        auto* mutable_header = static_cast<ShmHeader*>(shm_ptr_);
        mutable_header->read_seq = header.write_seq;

        // Store latest frame
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_ = frame;
        }
        frame_cv_.notify_all();

        // Call callback
        if (frame_callback_) {
            frame_callback_(frame);
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.frames_captured++;

            auto now = Clock::now();
            auto elapsed = std::chrono::duration<float>(now - start_time_).count();
            if (elapsed > 0) {
                stats_.average_fps = stats_.frames_captured / elapsed;
            }

            auto frame_time = std::chrono::duration<float>(now - last_frame_time_).count();
            if (frame_time > 0) {
                stats_.current_fps = 1.0f / frame_time;
            }
            last_frame_time_ = now;

            // Calculate latency from Isaac Sim timestamp
            auto isaac_time = std::chrono::nanoseconds(header.timestamp_ns);
            auto system_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch());
            // Note: This assumes clocks are synchronized
            stats_.average_latency = std::chrono::duration_cast<Duration>(
                system_time - isaac_time);
        }

        return true;
    }

    return false;
}

FramePtr IsaacShmCapture::process_shm_data(const ShmHeader& header, const uint8_t* data) {
    // Create frame
    auto frame = std::make_shared<Frame>(header.width, header.height, PixelFormat::BGR24);
    frame->metadata.frame_id = header.frame_id;
    
    // Convert Isaac timestamp (ns since epoch) to our steady_clock
    // Note: For precise timing, both sides should use steady_clock
    // Here we use the current time as approximation
    frame->metadata.timestamp = Clock::now();

    // Copy frame data (could optimize with zero-copy using shared_ptr aliasing)
    size_t frame_size = header.width * header.height * 3;
    std::memcpy(frame->ptr(), data, frame_size);

    // Update config with actual resolution
    config_.width = header.width;
    config_.height = header.height;

    return frame;
}

}  // namespace lagari
