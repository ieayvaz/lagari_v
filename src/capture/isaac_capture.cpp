#include "lagari/capture/isaac_capture.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#include <zmq.hpp>
#include <cstring>

namespace lagari {

// ============================================================================
// Constructor / Destructor
// ============================================================================

IsaacCapture::IsaacCapture(const CaptureConfig& config)
    : config_(config)
{
    // Default Isaac config
    isaac_config_.zmq_endpoint = "tcp://localhost:5555";
}

IsaacCapture::IsaacCapture(const CaptureConfig& config, const IsaacConfig& isaac_config)
    : config_(config)
    , isaac_config_(isaac_config)
{
}

IsaacCapture::~IsaacCapture() {
    stop();
    disconnect();
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool IsaacCapture::initialize(const Config& config) {
    // Parse Isaac-specific config
    isaac_config_.zmq_endpoint = config.get_string(
        "capture.isaac.endpoint", "tcp://localhost:5555");
    isaac_config_.zmq_recv_timeout_ms = config.get_int(
        "capture.isaac.timeout_ms", 100);
    isaac_config_.zmq_hwm = config.get_int(
        "capture.isaac.hwm", 2);
    isaac_config_.reconnect_on_timeout = config.get_bool(
        "capture.isaac.reconnect", true);

    return connect();
}

void IsaacCapture::start() {
    if (running_.load(std::memory_order_acquire)) {
        return;
    }

    if (!connected_.load(std::memory_order_acquire)) {
        LOG_ERROR("IsaacCapture: Not connected to Isaac Sim");
        return;
    }

    should_stop_.store(false, std::memory_order_release);
    start_time_ = Clock::now();
    last_frame_time_ = start_time_;

    receive_thread_ = std::thread(&IsaacCapture::receive_loop, this);
    running_.store(true, std::memory_order_release);

    LOG_INFO("IsaacCapture started, receiving from {}", isaac_config_.zmq_endpoint);
}

void IsaacCapture::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    should_stop_.store(true, std::memory_order_release);
    frame_cv_.notify_all();

    if (receive_thread_.joinable()) {
        receive_thread_.join();
    }

    running_.store(false, std::memory_order_release);
    LOG_INFO("IsaacCapture stopped");
}

bool IsaacCapture::is_running() const {
    return running_.load(std::memory_order_acquire);
}

// ============================================================================
// ICapture Implementation
// ============================================================================

FramePtr IsaacCapture::get_latest_frame() {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

FramePtr IsaacCapture::wait_for_frame(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(frame_mutex_);

    if (latest_frame_) {
        return latest_frame_;
    }

    frame_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() {
        return latest_frame_ != nullptr || should_stop_.load(std::memory_order_acquire);
    });

    return latest_frame_;
}

void IsaacCapture::set_frame_callback(FrameCallback callback) {
    frame_callback_ = std::move(callback);
}

CaptureStats IsaacCapture::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool IsaacCapture::is_open() const {
    return connected_.load(std::memory_order_acquire);
}

bool IsaacCapture::set_resolution(uint32_t width, uint32_t height) {
    // Resolution is controlled by Isaac Sim side
    LOG_WARN("IsaacCapture: Resolution is controlled by Isaac Sim");
    config_.width = width;
    config_.height = height;
    return false;
}

bool IsaacCapture::set_framerate(uint32_t fps) {
    // Framerate is controlled by Isaac Sim side
    LOG_WARN("IsaacCapture: Framerate is controlled by Isaac Sim");
    config_.fps = fps;
    return false;
}

bool IsaacCapture::set_exposure(bool /* auto_exp */, float /* exposure_time */) {
    // Exposure is controlled by Isaac Sim side
    LOG_WARN("IsaacCapture: Exposure is controlled by Isaac Sim");
    return false;
}

// ============================================================================
// Isaac-specific Methods
// ============================================================================

bool IsaacCapture::connect(const std::string& endpoint) {
    if (connected_.load(std::memory_order_acquire)) {
        disconnect();
    }

    std::string actual_endpoint = endpoint.empty() ? 
        isaac_config_.zmq_endpoint : endpoint;

    try {
        // Create ZMQ context and socket
        zmq_context_ = std::make_unique<zmq::context_t>(1);
        zmq_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, zmq::socket_type::sub);

        // Configure socket
        zmq_socket_->set(zmq::sockopt::rcvhwm, isaac_config_.zmq_hwm);
        zmq_socket_->set(zmq::sockopt::rcvtimeo, isaac_config_.zmq_recv_timeout_ms);
        zmq_socket_->set(zmq::sockopt::linger, 0);

        // Subscribe to "frame" topic
        zmq_socket_->set(zmq::sockopt::subscribe, "frame");

        // Connect
        zmq_socket_->connect(actual_endpoint);

        isaac_config_.zmq_endpoint = actual_endpoint;
        connected_.store(true, std::memory_order_release);

        LOG_INFO("IsaacCapture: Connected to {}", actual_endpoint);
        return true;

    } catch (const zmq::error_t& e) {
        LOG_ERROR("IsaacCapture: ZMQ connection failed: {}", e.what());
        zmq_socket_.reset();
        zmq_context_.reset();
        return false;
    }
}

void IsaacCapture::disconnect() {
    connected_.store(false, std::memory_order_release);

    if (zmq_socket_) {
        try {
            zmq_socket_->close();
        } catch (...) {}
        zmq_socket_.reset();
    }

    if (zmq_context_) {
        try {
            zmq_context_->close();
        } catch (...) {}
        zmq_context_.reset();
    }

    LOG_INFO("IsaacCapture: Disconnected");
}

bool IsaacCapture::is_connected() const {
    return connected_.load(std::memory_order_acquire);
}

// ============================================================================
// Receive Loop
// ============================================================================

void IsaacCapture::receive_loop() {
    LOG_DEBUG("IsaacCapture: Receive thread started");

    while (!should_stop_.load(std::memory_order_acquire)) {
        if (!receive_frame()) {
            // Timeout or error
            timeout_count_++;
            
            if (timeout_count_ > 50 && isaac_config_.reconnect_on_timeout) {
                LOG_WARN("IsaacCapture: Too many timeouts, reconnecting...");
                disconnect();
                if (!connect()) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                timeout_count_ = 0;
            }
        } else {
            timeout_count_ = 0;
        }
    }

    LOG_DEBUG("IsaacCapture: Receive thread exiting");
}

bool IsaacCapture::receive_frame() {
    if (!zmq_socket_) {
        return false;
    }

    try {
        // Receive multipart message: [topic, header, data]
        std::vector<zmq::message_t> messages;
        zmq::recv_result_t result;

        // Receive topic
        zmq::message_t topic_msg;
        result = zmq_socket_->recv(topic_msg, zmq::recv_flags::none);
        if (!result) {
            return false;  // Timeout
        }

        // Check for more parts
        int more = zmq_socket_->get(zmq::sockopt::rcvmore);
        if (!more) {
            LOG_WARN("IsaacCapture: Incomplete message (no header)");
            return false;
        }

        // Receive header
        zmq::message_t header_msg;
        result = zmq_socket_->recv(header_msg, zmq::recv_flags::none);
        if (!result) {
            return false;
        }

        more = zmq_socket_->get(zmq::sockopt::rcvmore);
        if (!more) {
            LOG_WARN("IsaacCapture: Incomplete message (no data)");
            return false;
        }

        // Receive frame data
        zmq::message_t data_msg;
        result = zmq_socket_->recv(data_msg, zmq::recv_flags::none);
        if (!result) {
            return false;
        }

        // Process the frame
        FramePtr frame = process_message(
            header_msg.data(), header_msg.size(),
            data_msg.data(), data_msg.size()
        );

        if (frame) {
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
            }

            return true;
        }

    } catch (const zmq::error_t& e) {
        if (e.num() != EAGAIN) {
            LOG_ERROR("IsaacCapture: ZMQ receive error: {}", e.what());
        }
    }

    return false;
}

FramePtr IsaacCapture::process_message(
    const void* header_data, size_t header_size,
    const void* frame_data, size_t frame_size)
{
    // Validate header size
    if (header_size != sizeof(FrameHeader)) {
        LOG_WARN("IsaacCapture: Invalid header size: {} (expected {})",
                 header_size, sizeof(FrameHeader));
        return nullptr;
    }

    // Parse header
    FrameHeader header;
    std::memcpy(&header, header_data, sizeof(header));

    // Validate frame data size
    size_t expected_size = header.width * header.height * 3;  // BGR24
    if (frame_size != expected_size) {
        LOG_WARN("IsaacCapture: Frame size mismatch: {} (expected {})",
                 frame_size, expected_size);
        return nullptr;
    }

    // Create frame
    auto frame = std::make_shared<Frame>(header.width, header.height, PixelFormat::BGR24);
    frame->metadata.frame_id = header.frame_id;
    frame->metadata.timestamp = Clock::now();  // Use local timestamp

    // Copy frame data
    std::memcpy(frame->ptr(), frame_data, frame_size);

    // Update config with actual resolution
    config_.width = header.width;
    config_.height = header.height;

    return frame;
}

}  // namespace lagari
