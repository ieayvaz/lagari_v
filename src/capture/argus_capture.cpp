#include "lagari/capture/argus_capture.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#ifdef HAS_ARGUS

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/NV/ImageNativeBuffer.h>
#include <nvbuf_utils.h>
#include <NvLogging.h>

#include <cstring>
#include <algorithm>

namespace lagari {

// ============================================================================
// Argus State (PIMPL)
// ============================================================================

struct ArgusCapture::ArgusState {
    // Argus objects
    Argus::UniqueObj<Argus::CameraProvider> camera_provider;
    Argus::CameraDevice* camera_device = nullptr;
    Argus::UniqueObj<Argus::CaptureSession> capture_session;
    Argus::UniqueObj<Argus::OutputStreamSettings> stream_settings;
    Argus::UniqueObj<Argus::OutputStream> output_stream;
    Argus::UniqueObj<Argus::Request> request;
    
    // EGLStream consumer
    Argus::UniqueObj<EGLStream::FrameConsumer> consumer;
    
    // Interfaces
    Argus::ICaptureSession* i_session = nullptr;
    Argus::IEventProvider* i_event_provider = nullptr;
    
    // Sensor info
    std::vector<Argus::SensorMode*> sensor_modes;
    Argus::SensorMode* current_mode = nullptr;
    
    // State
    bool initialized = false;
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

ArgusCapture::ArgusCapture(const CaptureConfig& config)
    : config_(config)
    , argus_(std::make_unique<ArgusState>())
{
}

ArgusCapture::ArgusCapture(const CaptureConfig& config, const ArgusConfig& argus_config)
    : config_(config)
    , argus_config_(argus_config)
    , argus_(std::make_unique<ArgusState>())
{
}

ArgusCapture::~ArgusCapture() {
    stop();
    
    // Clean up Argus objects in reverse order
    if (argus_) {
        argus_->request.reset();
        argus_->consumer.reset();
        argus_->output_stream.reset();
        argus_->stream_settings.reset();
        argus_->capture_session.reset();
        argus_->camera_provider.reset();
    }
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool ArgusCapture::initialize(const Config& config) {
    // Parse Argus-specific config
    argus_config_.sensor_mode = config.get_int("capture.argus.sensor_mode", 0);
    argus_config_.denoise_enable = config.get_bool("capture.argus.denoise", true);
    argus_config_.denoise_strength = config.get_float("capture.argus.denoise_strength", 0.5f);
    argus_config_.edge_enhance_enable = config.get_bool("capture.argus.edge_enhance", true);
    argus_config_.edge_enhance_strength = config.get_float("capture.argus.edge_enhance_strength", 0.5f);
    argus_config_.awb_enable = config.get_bool("capture.argus.awb", true);

    // Create camera provider
    if (!create_camera_provider()) {
        LOG_ERROR("ArgusCapture: Failed to create camera provider");
        return false;
    }

    // Open camera
    if (!open_camera(config_.camera_id)) {
        LOG_ERROR("ArgusCapture: Failed to open camera {}", config_.camera_id);
        return false;
    }

    // Create capture session
    if (!create_capture_session()) {
        LOG_ERROR("ArgusCapture: Failed to create capture session");
        return false;
    }

    // Create output stream
    if (!create_output_stream()) {
        LOG_ERROR("ArgusCapture: Failed to create output stream");
        return false;
    }

    // Create and configure request
    if (!create_request() || !configure_request()) {
        LOG_ERROR("ArgusCapture: Failed to create/configure request");
        return false;
    }

    argus_->initialized = true;
    LOG_INFO("ArgusCapture: Initialized camera {} at {}x{} @ {} fps",
             config_.camera_id, config_.width, config_.height, config_.fps);

    return true;
}

void ArgusCapture::start() {
    if (running_.load(std::memory_order_acquire)) {
        return;
    }

    if (!argus_->initialized) {
        LOG_ERROR("ArgusCapture: Not initialized");
        return;
    }

    // Start repeat capture
    Argus::Status status = argus_->i_session->repeat(argus_->request.get());
    if (status != Argus::STATUS_OK) {
        LOG_ERROR("ArgusCapture: Failed to start capture: {}", static_cast<int>(status));
        return;
    }

    should_stop_.store(false, std::memory_order_release);
    start_time_ = Clock::now();
    last_frame_time_ = start_time_;

    capture_thread_ = std::thread(&ArgusCapture::capture_loop, this);
    running_.store(true, std::memory_order_release);

    LOG_INFO("ArgusCapture: Started");
}

void ArgusCapture::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    should_stop_.store(true, std::memory_order_release);
    frame_cv_.notify_all();

    // Stop repeat capture
    if (argus_->i_session) {
        argus_->i_session->stopRepeat();
        argus_->i_session->waitForIdle();
    }

    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }

    running_.store(false, std::memory_order_release);
    LOG_INFO("ArgusCapture: Stopped");
}

bool ArgusCapture::is_running() const {
    return running_.load(std::memory_order_acquire);
}

// ============================================================================
// ICapture Implementation
// ============================================================================

FramePtr ArgusCapture::get_latest_frame() {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

FramePtr ArgusCapture::wait_for_frame(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(frame_mutex_);

    auto current_id = latest_frame_ ? latest_frame_->metadata.frame_id : 0;
    
    frame_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, current_id]() {
        return (latest_frame_ && latest_frame_->metadata.frame_id > current_id) ||
               should_stop_.load(std::memory_order_acquire);
    });

    return latest_frame_;
}

void ArgusCapture::set_frame_callback(FrameCallback callback) {
    frame_callback_ = std::move(callback);
}

CaptureStats ArgusCapture::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool ArgusCapture::is_open() const {
    return argus_ && argus_->initialized;
}

bool ArgusCapture::set_resolution(uint32_t width, uint32_t height) {
    // Resolution is determined by sensor mode
    LOG_WARN("ArgusCapture: Use set_sensor_mode() to change resolution");
    config_.width = width;
    config_.height = height;
    return false;
}

bool ArgusCapture::set_framerate(uint32_t fps) {
    if (!argus_->request) {
        return false;
    }

    config_.fps = fps;
    argus_config_.frame_duration_ns = 1e9f / fps;

    // Update the request
    Argus::ISourceSettings* source_settings = 
        Argus::interface_cast<Argus::ISourceSettings>(argus_->request);
    if (source_settings) {
        source_settings->setFrameDurationRange(
            Argus::Range<uint64_t>(argus_config_.frame_duration_ns));
        return true;
    }
    return false;
}

bool ArgusCapture::set_exposure(bool auto_exp, float exposure_time) {
    if (!argus_->request) {
        return false;
    }

    Argus::ISourceSettings* source_settings = 
        Argus::interface_cast<Argus::ISourceSettings>(argus_->request);
    if (!source_settings) {
        return false;
    }

    if (auto_exp) {
        // Enable auto exposure
        Argus::IAutoControlSettings* ac_settings = 
            Argus::interface_cast<Argus::IAutoControlSettings>(argus_->request);
        if (ac_settings) {
            ac_settings->setExposureCompensation(0.0f);
        }
        source_settings->setExposureTimeRange(
            Argus::Range<uint64_t>(argus_config_.exposure_time_min_ns,
                                   argus_config_.exposure_time_max_ns));
    } else {
        // Set manual exposure
        uint64_t exposure_ns = static_cast<uint64_t>(exposure_time * 1e9f);
        source_settings->setExposureTimeRange(
            Argus::Range<uint64_t>(exposure_ns, exposure_ns));
    }

    config_.auto_exposure = auto_exp;
    return true;
}

// ============================================================================
// Argus-specific Methods
// ============================================================================

int ArgusCapture::get_camera_count() const {
    if (!argus_->camera_provider) {
        return 0;
    }
    
    Argus::ICameraProvider* i_provider = 
        Argus::interface_cast<Argus::ICameraProvider>(argus_->camera_provider);
    if (!i_provider) {
        return 0;
    }

    std::vector<Argus::CameraDevice*> devices;
    i_provider->getCameraDevices(&devices);
    return static_cast<int>(devices.size());
}

std::vector<std::tuple<uint32_t, uint32_t, float>> ArgusCapture::get_sensor_modes() const {
    std::vector<std::tuple<uint32_t, uint32_t, float>> modes;
    
    for (auto* mode : argus_->sensor_modes) {
        Argus::ISensorMode* i_mode = Argus::interface_cast<Argus::ISensorMode>(mode);
        if (i_mode) {
            auto resolution = i_mode->getResolution();
            auto frame_duration = i_mode->getFrameDurationRange().min();
            float fps = frame_duration > 0 ? 1e9f / frame_duration : 0;
            modes.emplace_back(resolution.width(), resolution.height(), fps);
        }
    }
    
    return modes;
}

bool ArgusCapture::set_sensor_mode(int mode_index) {
    if (mode_index < 0 || mode_index >= static_cast<int>(argus_->sensor_modes.size())) {
        return false;
    }

    argus_->current_mode = argus_->sensor_modes[mode_index];
    argus_config_.sensor_mode = mode_index;

    Argus::ISensorMode* i_mode = 
        Argus::interface_cast<Argus::ISensorMode>(argus_->current_mode);
    if (i_mode) {
        auto resolution = i_mode->getResolution();
        config_.width = resolution.width();
        config_.height = resolution.height();
    }

    return true;
}

bool ArgusCapture::set_gain(float gain) {
    if (!argus_->request) {
        return false;
    }

    Argus::ISourceSettings* source_settings = 
        Argus::interface_cast<Argus::ISourceSettings>(argus_->request);
    if (source_settings) {
        source_settings->setGainRange(Argus::Range<float>(gain, gain));
        return true;
    }
    return false;
}

bool ArgusCapture::set_awb_mode(bool enable) {
    if (!argus_->request) {
        return false;
    }

    Argus::IAutoControlSettings* ac_settings = 
        Argus::interface_cast<Argus::IAutoControlSettings>(argus_->request);
    if (ac_settings) {
        ac_settings->setAwbMode(enable ? 
            Argus::AWB_MODE_AUTO : Argus::AWB_MODE_OFF);
        argus_config_.awb_enable = enable;
        return true;
    }
    return false;
}

bool ArgusCapture::set_denoise(bool enable, float strength) {
    if (!argus_->request) {
        return false;
    }

    Argus::IDenoiseSettings* denoise_settings = 
        Argus::interface_cast<Argus::IDenoiseSettings>(argus_->request);
    if (denoise_settings) {
        denoise_settings->setDenoiseMode(enable ? 
            Argus::DENOISE_MODE_FAST : Argus::DENOISE_MODE_OFF);
        denoise_settings->setDenoiseStrength(strength);
        argus_config_.denoise_enable = enable;
        argus_config_.denoise_strength = strength;
        return true;
    }
    return false;
}

// ============================================================================
// Private Methods
// ============================================================================

bool ArgusCapture::create_camera_provider() {
    argus_->camera_provider = Argus::UniqueObj<Argus::CameraProvider>(
        Argus::CameraProvider::create());
    
    if (!argus_->camera_provider) {
        LOG_ERROR("ArgusCapture: Failed to create CameraProvider");
        return false;
    }

    Argus::ICameraProvider* i_provider = 
        Argus::interface_cast<Argus::ICameraProvider>(argus_->camera_provider);
    if (!i_provider) {
        LOG_ERROR("ArgusCapture: Failed to get ICameraProvider interface");
        return false;
    }

    LOG_INFO("ArgusCapture: Argus version: {}", i_provider->getVersion().c_str());
    return true;
}

bool ArgusCapture::open_camera(int camera_id) {
    Argus::ICameraProvider* i_provider = 
        Argus::interface_cast<Argus::ICameraProvider>(argus_->camera_provider);
    
    std::vector<Argus::CameraDevice*> devices;
    i_provider->getCameraDevices(&devices);

    if (camera_id >= static_cast<int>(devices.size())) {
        LOG_ERROR("ArgusCapture: Camera {} not found ({} available)",
                  camera_id, devices.size());
        return false;
    }

    argus_->camera_device = devices[camera_id];

    // Get sensor modes
    Argus::ICameraProperties* i_props = 
        Argus::interface_cast<Argus::ICameraProperties>(argus_->camera_device);
    if (i_props) {
        i_props->getAllSensorModes(&argus_->sensor_modes);
        LOG_INFO("ArgusCapture: Found {} sensor modes", argus_->sensor_modes.size());
    }

    // Select sensor mode
    if (!argus_->sensor_modes.empty()) {
        int mode_idx = std::min(argus_config_.sensor_mode,
                                static_cast<int>(argus_->sensor_modes.size()) - 1);
        argus_->current_mode = argus_->sensor_modes[mode_idx];

        // Get resolution from mode
        Argus::ISensorMode* i_mode = 
            Argus::interface_cast<Argus::ISensorMode>(argus_->current_mode);
        if (i_mode) {
            auto resolution = i_mode->getResolution();
            config_.width = resolution.width();
            config_.height = resolution.height();
            LOG_INFO("ArgusCapture: Sensor mode {}: {}x{}", 
                     mode_idx, config_.width, config_.height);
        }
    }

    return true;
}

bool ArgusCapture::create_capture_session() {
    Argus::ICameraProvider* i_provider = 
        Argus::interface_cast<Argus::ICameraProvider>(argus_->camera_provider);

    argus_->capture_session = Argus::UniqueObj<Argus::CaptureSession>(
        i_provider->createCaptureSession(argus_->camera_device));
    
    if (!argus_->capture_session) {
        LOG_ERROR("ArgusCapture: Failed to create CaptureSession");
        return false;
    }

    argus_->i_session = 
        Argus::interface_cast<Argus::ICaptureSession>(argus_->capture_session);
    
    return argus_->i_session != nullptr;
}

bool ArgusCapture::create_output_stream() {
    // Create stream settings
    argus_->stream_settings = Argus::UniqueObj<Argus::OutputStreamSettings>(
        argus_->i_session->createOutputStreamSettings(Argus::STREAM_TYPE_EGL));
    
    Argus::IEGLOutputStreamSettings* i_stream_settings = 
        Argus::interface_cast<Argus::IEGLOutputStreamSettings>(argus_->stream_settings);
    
    if (!i_stream_settings) {
        LOG_ERROR("ArgusCapture: Failed to create stream settings");
        return false;
    }

    // Configure stream
    i_stream_settings->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888);
    i_stream_settings->setResolution(Argus::Size2D<uint32_t>(config_.width, config_.height));
    i_stream_settings->setMode(Argus::EGL_STREAM_MODE_FIFO);
    i_stream_settings->setFifoLength(config_.buffer_count);

    // Create stream
    argus_->output_stream = Argus::UniqueObj<Argus::OutputStream>(
        argus_->i_session->createOutputStream(argus_->stream_settings.get()));
    
    if (!argus_->output_stream) {
        LOG_ERROR("ArgusCapture: Failed to create output stream");
        return false;
    }

    // Create EGLStream consumer
    argus_->consumer = Argus::UniqueObj<EGLStream::FrameConsumer>(
        EGLStream::FrameConsumer::create(argus_->output_stream.get()));
    
    if (!argus_->consumer) {
        LOG_ERROR("ArgusCapture: Failed to create frame consumer");
        return false;
    }

    return true;
}

bool ArgusCapture::create_request() {
    argus_->request = Argus::UniqueObj<Argus::Request>(
        argus_->i_session->createRequest(Argus::CAPTURE_INTENT_VIDEO_RECORD));
    
    if (!argus_->request) {
        LOG_ERROR("ArgusCapture: Failed to create request");
        return false;
    }

    // Connect output stream to request
    Argus::IRequest* i_request = 
        Argus::interface_cast<Argus::IRequest>(argus_->request);
    
    if (!i_request) {
        return false;
    }

    i_request->enableOutputStream(argus_->output_stream.get());
    return true;
}

bool ArgusCapture::configure_request() {
    Argus::ISourceSettings* source_settings = 
        Argus::interface_cast<Argus::ISourceSettings>(argus_->request);
    
    if (!source_settings) {
        return false;
    }

    // Set sensor mode
    source_settings->setSensorMode(argus_->current_mode);

    // Set frame rate
    if (argus_config_.frame_duration_ns > 0) {
        source_settings->setFrameDurationRange(
            Argus::Range<uint64_t>(argus_config_.frame_duration_ns));
    } else {
        uint64_t frame_duration = static_cast<uint64_t>(1e9 / config_.fps);
        source_settings->setFrameDurationRange(
            Argus::Range<uint64_t>(frame_duration));
    }

    // Set gain range
    source_settings->setGainRange(
        Argus::Range<float>(argus_config_.gain_range_min, 
                           argus_config_.gain_range_max));

    // Configure auto control
    Argus::IAutoControlSettings* ac_settings = 
        Argus::interface_cast<Argus::IAutoControlSettings>(argus_->request);
    
    if (ac_settings) {
        ac_settings->setAwbMode(argus_config_.awb_enable ? 
            Argus::AWB_MODE_AUTO : Argus::AWB_MODE_OFF);
    }

    // Configure denoise
    Argus::IDenoiseSettings* denoise_settings = 
        Argus::interface_cast<Argus::IDenoiseSettings>(argus_->request);
    
    if (denoise_settings) {
        denoise_settings->setDenoiseMode(argus_config_.denoise_enable ? 
            Argus::DENOISE_MODE_FAST : Argus::DENOISE_MODE_OFF);
        denoise_settings->setDenoiseStrength(argus_config_.denoise_strength);
    }

    // Configure edge enhancement
    Argus::IEdgeEnhanceSettings* edge_settings = 
        Argus::interface_cast<Argus::IEdgeEnhanceSettings>(argus_->request);
    
    if (edge_settings) {
        edge_settings->setEdgeEnhanceMode(argus_config_.edge_enhance_enable ? 
            Argus::EDGE_ENHANCE_MODE_FAST : Argus::EDGE_ENHANCE_MODE_OFF);
        edge_settings->setEdgeEnhanceStrength(argus_config_.edge_enhance_strength);
    }

    return true;
}

void ArgusCapture::capture_loop() {
    LOG_DEBUG("ArgusCapture: Capture thread started");

    EGLStream::IFrameConsumer* i_consumer = 
        Argus::interface_cast<EGLStream::IFrameConsumer>(argus_->consumer);
    
    if (!i_consumer) {
        LOG_ERROR("ArgusCapture: Failed to get frame consumer interface");
        return;
    }

    while (!should_stop_.load(std::memory_order_acquire)) {
        // Acquire frame with timeout
        Argus::UniqueObj<EGLStream::Frame> frame(
            i_consumer->acquireFrame(100000000)); // 100ms timeout

        if (!frame) {
            continue;  // Timeout, try again
        }

        // Process frame
        FramePtr lagari_frame = process_frame(frame.get());
        
        if (lagari_frame) {
            // Store latest frame
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                latest_frame_ = lagari_frame;
            }
            frame_cv_.notify_all();

            // Call callback
            if (frame_callback_) {
                frame_callback_(lagari_frame);
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
    }

    LOG_DEBUG("ArgusCapture: Capture thread exiting");
}

FramePtr ArgusCapture::process_frame(EGLStream::Frame* frame) {
    EGLStream::IFrame* i_frame = Argus::interface_cast<EGLStream::IFrame>(frame);
    if (!i_frame) {
        return nullptr;
    }

    // Get native buffer
    EGLStream::NV::IImageNativeBuffer* i_native = 
        Argus::interface_cast<EGLStream::NV::IImageNativeBuffer>(i_frame->getImage());
    
    if (!i_native) {
        LOG_WARN("ArgusCapture: Failed to get native buffer");
        return nullptr;
    }

    // Create NvBuffer for format conversion
    int dmabuf_fd = i_native->createNvBuffer(
        Argus::Size2D<uint32_t>(config_.width, config_.height),
        NvBufferColorFormat_ABGR32,
        NvBufferLayout_Pitch);

    if (dmabuf_fd < 0) {
        LOG_WARN("ArgusCapture: Failed to create NvBuffer");
        return nullptr;
    }

    // Map buffer
    NvBufferParams params;
    NvBufferGetParams(dmabuf_fd, &params);

    void* data = nullptr;
    NvBufferMemMap(dmabuf_fd, 0, NvBufferMem_Read, &data);
    NvBufferMemSyncForCpu(dmabuf_fd, 0, &data);

    // Create frame and copy data (convert RGBA to BGR)
    auto lagari_frame = std::make_shared<Frame>(config_.width, config_.height, PixelFormat::BGR24);
    lagari_frame->metadata.frame_id = ++frame_counter_;
    lagari_frame->metadata.timestamp = Clock::now();

    // Convert RGBA32 to BGR24
    const uint8_t* src = static_cast<const uint8_t*>(data);
    uint8_t* dst = lagari_frame->ptr();
    
    for (uint32_t y = 0; y < config_.height; ++y) {
        const uint8_t* src_row = src + y * params.pitch[0];
        uint8_t* dst_row = dst + y * config_.width * 3;
        
        for (uint32_t x = 0; x < config_.width; ++x) {
            dst_row[x * 3 + 0] = src_row[x * 4 + 2];  // B
            dst_row[x * 3 + 1] = src_row[x * 4 + 1];  // G
            dst_row[x * 3 + 2] = src_row[x * 4 + 0];  // R
        }
    }

    // Cleanup
    NvBufferMemUnMap(dmabuf_fd, 0, &data);
    NvBufferDestroy(dmabuf_fd);

    return lagari_frame;
}

}  // namespace lagari

#else  // !HAS_ARGUS

namespace lagari {

// Stub implementation when Argus is not available
struct ArgusCapture::ArgusState {};

ArgusCapture::ArgusCapture(const CaptureConfig& config) : config_(config) {}
ArgusCapture::ArgusCapture(const CaptureConfig& config, const ArgusConfig&) : config_(config) {}
ArgusCapture::~ArgusCapture() = default;

bool ArgusCapture::initialize(const Config&) {
    LOG_ERROR("ArgusCapture: Not available (compile with HAS_ARGUS)");
    return false;
}

void ArgusCapture::start() {}
void ArgusCapture::stop() {}
bool ArgusCapture::is_running() const { return false; }
FramePtr ArgusCapture::get_latest_frame() { return nullptr; }
FramePtr ArgusCapture::wait_for_frame(uint32_t) { return nullptr; }
void ArgusCapture::set_frame_callback(FrameCallback) {}
CaptureStats ArgusCapture::get_stats() const { return {}; }
bool ArgusCapture::is_open() const { return false; }
bool ArgusCapture::set_resolution(uint32_t, uint32_t) { return false; }
bool ArgusCapture::set_framerate(uint32_t) { return false; }
bool ArgusCapture::set_exposure(bool, float) { return false; }
int ArgusCapture::get_camera_count() const { return 0; }
std::vector<std::tuple<uint32_t, uint32_t, float>> ArgusCapture::get_sensor_modes() const { return {}; }
bool ArgusCapture::set_sensor_mode(int) { return false; }
bool ArgusCapture::set_gain(float) { return false; }
bool ArgusCapture::set_awb_mode(bool) { return false; }
bool ArgusCapture::set_denoise(bool, float) { return false; }
bool ArgusCapture::create_camera_provider() { return false; }
bool ArgusCapture::open_camera(int) { return false; }
bool ArgusCapture::create_capture_session() { return false; }
bool ArgusCapture::create_output_stream() { return false; }
bool ArgusCapture::create_request() { return false; }
bool ArgusCapture::configure_request() { return false; }
void ArgusCapture::capture_loop() {}
FramePtr ArgusCapture::process_frame(EGLStream::Frame*) { return nullptr; }

}  // namespace lagari

#endif  // HAS_ARGUS
