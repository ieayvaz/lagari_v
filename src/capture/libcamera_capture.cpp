#include "lagari/capture/libcamera_capture.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#ifdef HAS_LIBCAMERA

#include <libcamera/libcamera.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>
#include <libcamera/controls.h>
#include <libcamera/control_ids.h>
#include <libcamera/property_ids.h>

#include <sys/mman.h>
#include <cstring>
#include <algorithm>
#include <map>

namespace lagari {

// ============================================================================
// Libcamera State (PIMPL)
// ============================================================================

struct LibcameraCapture::LibcameraState {
    std::unique_ptr<libcamera::CameraManager> camera_manager;
    std::shared_ptr<libcamera::Camera> camera;
    std::unique_ptr<libcamera::CameraConfiguration> config;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator;
    libcamera::Stream* stream = nullptr;
    
    std::vector<std::unique_ptr<libcamera::Request>> requests;
    std::map<libcamera::FrameBuffer*, void*> mapped_buffers;
    
    libcamera::ControlList controls;
    
    bool initialized = false;
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

LibcameraCapture::LibcameraCapture(const CaptureConfig& config)
    : config_(config)
    , lc_(std::make_unique<LibcameraState>())
{
}

LibcameraCapture::LibcameraCapture(const CaptureConfig& config, const LibcameraConfig& lc_config)
    : config_(config)
    , lc_config_(lc_config)
    , lc_(std::make_unique<LibcameraState>())
{
}

LibcameraCapture::~LibcameraCapture() {
    stop();
    
    if (lc_) {
        // Unmap buffers
        for (auto& [buffer, ptr] : lc_->mapped_buffers) {
            if (ptr) {
                munmap(ptr, buffer->planes()[0].length);
            }
        }
        lc_->mapped_buffers.clear();
        
        // Release allocator
        if (lc_->allocator && lc_->stream) {
            lc_->allocator->free(lc_->stream);
        }
        lc_->allocator.reset();
        
        // Release camera
        if (lc_->camera) {
            lc_->camera->release();
            lc_->camera.reset();
        }
        
        // Stop camera manager
        if (lc_->camera_manager) {
            lc_->camera_manager->stop();
        }
    }
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool LibcameraCapture::initialize(const Config& config) {
    // Parse libcamera-specific config
    lc_config_.ae_enable = config.get_bool("capture.libcamera.ae_enable", true);
    lc_config_.exposure_time_us = config.get_int("capture.libcamera.exposure_us", 0);
    lc_config_.analogue_gain = config.get_float("capture.libcamera.gain", 1.0f);
    lc_config_.awb_enable = config.get_bool("capture.libcamera.awb", true);
    lc_config_.brightness = config.get_float("capture.libcamera.brightness", 0.0f);
    lc_config_.contrast = config.get_float("capture.libcamera.contrast", 1.0f);
    lc_config_.saturation = config.get_float("capture.libcamera.saturation", 1.0f);
    lc_config_.sharpness = config.get_float("capture.libcamera.sharpness", 1.0f);
    lc_config_.denoise_mode = config.get_int("capture.libcamera.denoise", 2);
    lc_config_.hflip = config.get_bool("capture.flip_horizontal", false);
    lc_config_.vflip = config.get_bool("capture.flip_vertical", false);

    // Create camera manager
    lc_->camera_manager = std::make_unique<libcamera::CameraManager>();
    int ret = lc_->camera_manager->start();
    if (ret) {
        LOG_ERROR("LibcameraCapture: Failed to start camera manager: {}", ret);
        return false;
    }

    // Open camera
    if (!open_camera(config_.camera_id)) {
        LOG_ERROR("LibcameraCapture: Failed to open camera {}", config_.camera_id);
        return false;
    }

    // Configure camera
    if (!configure_camera()) {
        LOG_ERROR("LibcameraCapture: Failed to configure camera");
        return false;
    }

    // Create buffers
    if (!create_buffers()) {
        LOG_ERROR("LibcameraCapture: Failed to create buffers");
        return false;
    }

    lc_->initialized = true;
    LOG_INFO("LibcameraCapture: Initialized {} at {}x{} @ {} fps",
             get_camera_model(), config_.width, config_.height, config_.fps);

    return true;
}

void LibcameraCapture::start() {
    if (running_.load(std::memory_order_acquire)) {
        return;
    }

    if (!lc_->initialized) {
        LOG_ERROR("LibcameraCapture: Not initialized");
        return;
    }

    // Apply initial controls
    apply_controls();

    // Start camera
    int ret = lc_->camera->start(&lc_->controls);
    if (ret) {
        LOG_ERROR("LibcameraCapture: Failed to start camera: {}", ret);
        return;
    }

    // Queue all requests
    for (auto& request : lc_->requests) {
        ret = lc_->camera->queueRequest(request.get());
        if (ret) {
            LOG_ERROR("LibcameraCapture: Failed to queue request: {}", ret);
            lc_->camera->stop();
            return;
        }
    }

    should_stop_.store(false, std::memory_order_release);
    start_time_ = Clock::now();
    last_frame_time_ = start_time_;
    running_.store(true, std::memory_order_release);

    LOG_INFO("LibcameraCapture: Started");
}

void LibcameraCapture::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    should_stop_.store(true, std::memory_order_release);
    frame_cv_.notify_all();

    if (lc_->camera) {
        lc_->camera->stop();
    }

    running_.store(false, std::memory_order_release);
    LOG_INFO("LibcameraCapture: Stopped");
}

bool LibcameraCapture::is_running() const {
    return running_.load(std::memory_order_acquire);
}

// ============================================================================
// ICapture Implementation
// ============================================================================

FramePtr LibcameraCapture::get_latest_frame() {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

FramePtr LibcameraCapture::wait_for_frame(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(frame_mutex_);

    auto current_id = latest_frame_ ? latest_frame_->metadata.frame_id : 0;
    
    frame_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, current_id]() {
        return (latest_frame_ && latest_frame_->metadata.frame_id > current_id) ||
               should_stop_.load(std::memory_order_acquire);
    });

    return latest_frame_;
}

void LibcameraCapture::set_frame_callback(FrameCallback callback) {
    frame_callback_ = std::move(callback);
}

CaptureStats LibcameraCapture::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool LibcameraCapture::is_open() const {
    return lc_ && lc_->initialized;
}

bool LibcameraCapture::set_resolution(uint32_t width, uint32_t height) {
    // Resolution changes require reconfiguration
    LOG_WARN("LibcameraCapture: Runtime resolution change not supported");
    config_.width = width;
    config_.height = height;
    return false;
}

bool LibcameraCapture::set_framerate(uint32_t fps) {
    config_.fps = fps;
    
    // Set frame duration control
    int64_t frame_duration = 1000000 / fps;  // microseconds
    lc_->controls.set(libcamera::controls::FrameDurationLimits,
                      libcamera::Span<const int64_t, 2>({frame_duration, frame_duration}));
    
    return true;
}

bool LibcameraCapture::set_exposure(bool auto_exp, float exposure_time) {
    lc_config_.ae_enable = auto_exp;
    
    if (auto_exp) {
        lc_->controls.set(libcamera::controls::AeEnable, true);
    } else {
        lc_->controls.set(libcamera::controls::AeEnable, false);
        int64_t exposure_us = static_cast<int64_t>(exposure_time * 1e6f);
        lc_->controls.set(libcamera::controls::ExposureTime, exposure_us);
        lc_config_.exposure_time_us = exposure_us;
    }
    
    config_.auto_exposure = auto_exp;
    return true;
}

// ============================================================================
// Libcamera-specific Methods
// ============================================================================

int LibcameraCapture::get_camera_count() const {
    if (!lc_->camera_manager) {
        return 0;
    }
    return static_cast<int>(lc_->camera_manager->cameras().size());
}

std::string LibcameraCapture::get_camera_model() const {
    if (!lc_->camera) {
        return "Unknown";
    }
    
    const libcamera::ControlList& props = lc_->camera->properties();
    auto model = props.get(libcamera::properties::Model);
    if (model) {
        return *model;
    }
    return lc_->camera->id();
}

bool LibcameraCapture::set_gain(float gain) {
    lc_config_.analogue_gain = gain;
    lc_->controls.set(libcamera::controls::AnalogueGain, gain);
    return true;
}

bool LibcameraCapture::set_awb_mode(bool enable) {
    lc_config_.awb_enable = enable;
    lc_->controls.set(libcamera::controls::AwbEnable, enable);
    return true;
}

bool LibcameraCapture::set_colour_gains(float red_gain, float blue_gain) {
    lc_config_.colour_gains[0] = red_gain;
    lc_config_.colour_gains[1] = blue_gain;
    lc_->controls.set(libcamera::controls::ColourGains,
                      libcamera::Span<const float, 2>({red_gain, blue_gain}));
    return true;
}

bool LibcameraCapture::set_brightness(float brightness) {
    lc_config_.brightness = std::clamp(brightness, -1.0f, 1.0f);
    lc_->controls.set(libcamera::controls::Brightness, lc_config_.brightness);
    return true;
}

bool LibcameraCapture::set_contrast(float contrast) {
    lc_config_.contrast = std::clamp(contrast, 0.0f, 2.0f);
    lc_->controls.set(libcamera::controls::Contrast, lc_config_.contrast);
    return true;
}

bool LibcameraCapture::set_saturation(float saturation) {
    lc_config_.saturation = std::clamp(saturation, 0.0f, 2.0f);
    lc_->controls.set(libcamera::controls::Saturation, lc_config_.saturation);
    return true;
}

bool LibcameraCapture::set_sharpness(float sharpness) {
    lc_config_.sharpness = std::clamp(sharpness, 0.0f, 2.0f);
    lc_->controls.set(libcamera::controls::Sharpness, lc_config_.sharpness);
    return true;
}

// ============================================================================
// Private Methods
// ============================================================================

bool LibcameraCapture::open_camera(int camera_id) {
    auto cameras = lc_->camera_manager->cameras();
    
    if (cameras.empty()) {
        LOG_ERROR("LibcameraCapture: No cameras available");
        return false;
    }

    if (camera_id >= static_cast<int>(cameras.size())) {
        LOG_ERROR("LibcameraCapture: Camera {} not found ({} available)",
                  camera_id, cameras.size());
        return false;
    }

    lc_->camera = cameras[camera_id];
    
    int ret = lc_->camera->acquire();
    if (ret) {
        LOG_ERROR("LibcameraCapture: Failed to acquire camera: {}", ret);
        lc_->camera.reset();
        return false;
    }

    LOG_INFO("LibcameraCapture: Opened camera: {}", lc_->camera->id());
    return true;
}

bool LibcameraCapture::configure_camera() {
    // Generate configuration for video capture
    lc_->config = lc_->camera->generateConfiguration(
        {libcamera::StreamRole::VideoRecording});
    
    if (!lc_->config || lc_->config->empty()) {
        LOG_ERROR("LibcameraCapture: Failed to generate configuration");
        return false;
    }

    // Configure stream
    libcamera::StreamConfiguration& stream_config = lc_->config->at(0);
    stream_config.size.width = config_.width;
    stream_config.size.height = config_.height;
    stream_config.pixelFormat = libcamera::formats::BGR888;
    stream_config.bufferCount = config_.buffer_count;

    // Validate configuration
    libcamera::CameraConfiguration::Status status = lc_->config->validate();
    if (status == libcamera::CameraConfiguration::Invalid) {
        LOG_ERROR("LibcameraCapture: Invalid configuration");
        return false;
    }
    
    if (status == libcamera::CameraConfiguration::Adjusted) {
        LOG_WARN("LibcameraCapture: Configuration adjusted to {}x{}",
                 stream_config.size.width, stream_config.size.height);
        config_.width = stream_config.size.width;
        config_.height = stream_config.size.height;
    }

    // Apply configuration
    int ret = lc_->camera->configure(lc_->config.get());
    if (ret) {
        LOG_ERROR("LibcameraCapture: Failed to configure camera: {}", ret);
        return false;
    }

    lc_->stream = stream_config.stream();
    
    LOG_INFO("LibcameraCapture: Configured {}x{} {}",
             stream_config.size.width, stream_config.size.height,
             stream_config.pixelFormat.toString());

    return true;
}

bool LibcameraCapture::create_buffers() {
    lc_->allocator = std::make_unique<libcamera::FrameBufferAllocator>(lc_->camera);
    
    int ret = lc_->allocator->allocate(lc_->stream);
    if (ret < 0) {
        LOG_ERROR("LibcameraCapture: Failed to allocate buffers: {}", ret);
        return false;
    }

    LOG_INFO("LibcameraCapture: Allocated {} buffers", ret);

    // Create requests and mmap buffers
    const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& buffers =
        lc_->allocator->buffers(lc_->stream);

    for (const auto& buffer : buffers) {
        std::unique_ptr<libcamera::Request> request = lc_->camera->createRequest();
        if (!request) {
            LOG_ERROR("LibcameraCapture: Failed to create request");
            return false;
        }

        ret = request->addBuffer(lc_->stream, buffer.get());
        if (ret) {
            LOG_ERROR("LibcameraCapture: Failed to add buffer to request: {}", ret);
            return false;
        }

        // Memory map the buffer
        const libcamera::FrameBuffer::Plane& plane = buffer->planes()[0];
        void* ptr = mmap(nullptr, plane.length, PROT_READ, MAP_SHARED,
                         plane.fd.get(), plane.offset);
        if (ptr == MAP_FAILED) {
            LOG_ERROR("LibcameraCapture: Failed to mmap buffer: {}", strerror(errno));
            return false;
        }
        
        lc_->mapped_buffers[buffer.get()] = ptr;
        lc_->requests.push_back(std::move(request));
    }

    // Connect request completed signal
    lc_->camera->requestCompleted.connect(this, &LibcameraCapture::request_complete);

    return true;
}

void LibcameraCapture::request_complete(libcamera::Request* request) {
    if (request->status() == libcamera::Request::RequestCancelled) {
        return;
    }

    if (should_stop_.load(std::memory_order_acquire)) {
        return;
    }

    // Get completed buffer
    const libcamera::Request::BufferMap& buffers = request->buffers();
    auto it = buffers.find(lc_->stream);
    if (it == buffers.end()) {
        LOG_WARN("LibcameraCapture: No buffer in request");
        return;
    }

    libcamera::FrameBuffer* buffer = it->second;

    // Process frame
    FramePtr frame = process_buffer(buffer);
    
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
    }

    // Re-queue the request
    request->reuse(libcamera::Request::ReuseBuffers);
    lc_->camera->queueRequest(request);
}

FramePtr LibcameraCapture::process_buffer(libcamera::FrameBuffer* buffer) {
    auto it = lc_->mapped_buffers.find(buffer);
    if (it == lc_->mapped_buffers.end()) {
        LOG_WARN("LibcameraCapture: Unmapped buffer");
        return nullptr;
    }

    const uint8_t* src = static_cast<const uint8_t*>(it->second);
    
    // Create frame
    auto frame = std::make_shared<Frame>(config_.width, config_.height, PixelFormat::BGR24);
    frame->metadata.frame_id = ++frame_counter_;
    frame->metadata.timestamp = Clock::now();

    // Copy data (BGR888 is already in our desired format)
    size_t data_size = config_.width * config_.height * 3;
    std::memcpy(frame->ptr(), src, data_size);

    return frame;
}

void LibcameraCapture::apply_controls() {
    // Auto exposure
    lc_->controls.set(libcamera::controls::AeEnable, lc_config_.ae_enable);
    if (!lc_config_.ae_enable && lc_config_.exposure_time_us > 0) {
        lc_->controls.set(libcamera::controls::ExposureTime, lc_config_.exposure_time_us);
    }

    // Gain
    if (lc_config_.analogue_gain > 0) {
        lc_->controls.set(libcamera::controls::AnalogueGain, lc_config_.analogue_gain);
    }

    // Auto white balance
    lc_->controls.set(libcamera::controls::AwbEnable, lc_config_.awb_enable);
    if (!lc_config_.awb_enable && (lc_config_.colour_gains[0] > 0 || lc_config_.colour_gains[1] > 0)) {
        lc_->controls.set(libcamera::controls::ColourGains,
                          libcamera::Span<const float, 2>(lc_config_.colour_gains));
    }

    // Image tuning
    lc_->controls.set(libcamera::controls::Brightness, lc_config_.brightness);
    lc_->controls.set(libcamera::controls::Contrast, lc_config_.contrast);
    lc_->controls.set(libcamera::controls::Saturation, lc_config_.saturation);
    lc_->controls.set(libcamera::controls::Sharpness, lc_config_.sharpness);

    // Frame rate
    if (config_.fps > 0) {
        int64_t frame_duration = 1000000 / config_.fps;
        lc_->controls.set(libcamera::controls::FrameDurationLimits,
                          libcamera::Span<const int64_t, 2>({frame_duration, frame_duration}));
    }
}

}  // namespace lagari

#else  // !HAS_LIBCAMERA

namespace lagari {

// Stub implementation when libcamera is not available
struct LibcameraCapture::LibcameraState {};

LibcameraCapture::LibcameraCapture(const CaptureConfig& config) : config_(config) {}
LibcameraCapture::LibcameraCapture(const CaptureConfig& config, const LibcameraConfig&) : config_(config) {}
LibcameraCapture::~LibcameraCapture() = default;

bool LibcameraCapture::initialize(const Config&) {
    LOG_ERROR("LibcameraCapture: Not available (compile with HAS_LIBCAMERA)");
    return false;
}

void LibcameraCapture::start() {}
void LibcameraCapture::stop() {}
bool LibcameraCapture::is_running() const { return false; }
FramePtr LibcameraCapture::get_latest_frame() { return nullptr; }
FramePtr LibcameraCapture::wait_for_frame(uint32_t) { return nullptr; }
void LibcameraCapture::set_frame_callback(FrameCallback) {}
CaptureStats LibcameraCapture::get_stats() const { return {}; }
bool LibcameraCapture::is_open() const { return false; }
bool LibcameraCapture::set_resolution(uint32_t, uint32_t) { return false; }
bool LibcameraCapture::set_framerate(uint32_t) { return false; }
bool LibcameraCapture::set_exposure(bool, float) { return false; }
int LibcameraCapture::get_camera_count() const { return 0; }
std::string LibcameraCapture::get_camera_model() const { return ""; }
bool LibcameraCapture::set_gain(float) { return false; }
bool LibcameraCapture::set_awb_mode(bool) { return false; }
bool LibcameraCapture::set_colour_gains(float, float) { return false; }
bool LibcameraCapture::set_brightness(float) { return false; }
bool LibcameraCapture::set_contrast(float) { return false; }
bool LibcameraCapture::set_saturation(float) { return false; }
bool LibcameraCapture::set_sharpness(float) { return false; }
bool LibcameraCapture::open_camera(int) { return false; }
bool LibcameraCapture::configure_camera() { return false; }
bool LibcameraCapture::create_buffers() { return false; }
void LibcameraCapture::request_complete(libcamera::Request*) {}
FramePtr LibcameraCapture::process_buffer(libcamera::FrameBuffer*) { return nullptr; }
void LibcameraCapture::apply_controls() {}

}  // namespace lagari

#endif  // HAS_LIBCAMERA
