/**
 * @file gstreamer_display.cpp
 * @brief GStreamer-based display implementation using appsrc
 */

#include "lagari/display/gstreamer_display.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#include <opencv2/imgproc.hpp>
#include <cstring>
#include <sstream>

#ifdef HAS_GSTREAMER

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>

namespace lagari {

// ============================================================================
// GStreamer State (PIMPL)
// ============================================================================

struct GstreamerDisplay::GstState {
    GstElement* pipeline = nullptr;
    GstElement* appsrc = nullptr;
    GstBus* bus = nullptr;
    
    std::string pipeline_str;
    bool initialized = false;
    
    // Format info
    uint32_t width = 0;
    uint32_t height = 0;
    bool caps_set = false;
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

GstreamerDisplay::GstreamerDisplay(const DisplayConfig& config)
    : config_(config)
    , gst_(std::make_unique<GstState>())
    , overlay_renderer_(config.overlay)
    , min_frame_interval_(std::chrono::milliseconds(1000 / std::max(1u, config.max_fps)))
{
}

GstreamerDisplay::~GstreamerDisplay() {
    stop();
    destroy_pipeline();
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool GstreamerDisplay::initialize(const Config& config) {
    // Parse configuration
    config_.enabled = config.get_bool("display.enabled", false);
    config_.target = config.get_string("display.target", "window");
    config_.window_name = config.get_string("display.window_name", "Lagari Vision");
    config_.fullscreen = config.get_bool("display.fullscreen", false);
    config_.host = config.get_string("display.host", "0.0.0.0");
    config_.port = static_cast<uint16_t>(config.get_int("display.port", 5600));
    config_.bitrate_kbps = config.get_int("display.bitrate_kbps", 2000);
    config_.pipeline = config.get_string("display.pipeline", "");
    config_.max_fps = config.get_int("display.max_fps", 30);
    config_.drop_frames = config.get_bool("display.drop_frames", true);
    config_.codec = config.get_string("display.codec", "h264");
    config_.hw_encode = config.get_bool("display.hw_encode", true);
    
    // Overlay config
    config_.overlay.enabled = config.get_bool("display.overlay.enabled", true);
    config_.overlay.timestamp = config.get_bool("display.overlay.timestamp", true);
    config_.overlay.bounding_boxes = config.get_bool("display.overlay.bounding_boxes", true);
    config_.overlay.state = config.get_bool("display.overlay.state", true);
    config_.overlay.latency = config.get_bool("display.overlay.latency", true);
    
    overlay_renderer_.set_config(config_.overlay);
    min_frame_interval_ = std::chrono::milliseconds(1000 / std::max(1u, config_.max_fps));

    if (!config_.enabled) {
        LOG_INFO("GstreamerDisplay: Disabled by configuration");
        return true;  // Not an error, just disabled
    }

    // Initialize GStreamer
    GError* error = nullptr;
    if (!gst_init_check(nullptr, nullptr, &error)) {
        LOG_ERROR("GstreamerDisplay: Failed to initialize GStreamer: {}",
                  error ? error->message : "unknown");
        if (error) g_error_free(error);
        return false;
    }

    gst_->initialized = true;
    enabled_.store(true, std::memory_order_release);
    
    LOG_INFO("GstreamerDisplay: Initialized with target '{}'", config_.target);
    return true;
}

void GstreamerDisplay::start() {
    if (!config_.enabled || !gst_->initialized) {
        return;
    }

    if (running_.load(std::memory_order_acquire)) {
        return;
    }

    // Pipeline is created lazily on first frame to know dimensions
    start_time_ = Clock::now();
    last_frame_time_ = start_time_;
    last_push_time_ = start_time_;
    
    running_.store(true, std::memory_order_release);
    LOG_INFO("GstreamerDisplay: Started");
}

void GstreamerDisplay::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    destroy_pipeline();
    running_.store(false, std::memory_order_release);
    LOG_INFO("GstreamerDisplay: Stopped");
}

bool GstreamerDisplay::is_running() const {
    return running_.load(std::memory_order_acquire);
}

// ============================================================================
// IDisplay Implementation
// ============================================================================

void GstreamerDisplay::push_frame(const Frame& frame,
                                  const DetectionResult* detections,
                                  SystemState state,
                                  Duration latency)
{
    if (!enabled_.load(std::memory_order_acquire) || !running_.load(std::memory_order_acquire)) {
        return;
    }

    // Frame rate limiting
    if (!should_process_frame()) {
        if (config_.drop_frames) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.frames_dropped++;
        }
        return;
    }

    // Create pipeline on first frame if needed
    if (!gst_->pipeline) {
        gst_->width = frame.metadata.width;
        gst_->height = frame.metadata.height;
        
        if (!create_pipeline()) {
            LOG_ERROR("GstreamerDisplay: Failed to create pipeline");
            enabled_.store(false, std::memory_order_release);
            return;
        }
    }

    // Check if dimensions changed
    if (frame.metadata.width != gst_->width || frame.metadata.height != gst_->height) {
        LOG_WARN("GstreamerDisplay: Frame dimensions changed, recreating pipeline");
        destroy_pipeline();
        gst_->width = frame.metadata.width;
        gst_->height = frame.metadata.height;
        if (!create_pipeline()) {
            enabled_.store(false, std::memory_order_release);
            return;
        }
    }

    // Apply overlay if enabled
    cv::Mat output_mat;
    if (config_.overlay.enabled) {
        cv::Mat input_mat = frame_to_mat(frame);
        if (input_mat.empty()) {
            return;
        }
        output_mat = input_mat.clone();
        overlay_renderer_.render_inplace(output_mat, detections, state, latency);
    } else {
        output_mat = frame_to_mat(frame);
        if (output_mat.empty()) {
            return;
        }
        // Clone if not continuous to ensure we have contiguous data
        if (!output_mat.isContinuous()) {
            output_mat = output_mat.clone();
        }
    }

    // Push to GStreamer
    if (push_buffer(output_mat.data, output_mat.total() * output_mat.elemSize(),
                   static_cast<uint32_t>(output_mat.cols),
                   static_cast<uint32_t>(output_mat.rows))) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.frames_displayed++;
        
        auto now = Clock::now();
        auto elapsed = std::chrono::duration<float>(now - last_frame_time_).count();
        if (elapsed > 0) {
            stats_.current_fps = 1.0f / elapsed;
        }
        last_frame_time_ = now;
    }
    
    last_push_time_ = Clock::now();
}

bool GstreamerDisplay::is_enabled() const {
    return enabled_.load(std::memory_order_acquire);
}

void GstreamerDisplay::set_enabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_release);
}

GstreamerDisplay::DisplayStats GstreamerDisplay::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// ============================================================================
// Pipeline Building
// ============================================================================

std::string GstreamerDisplay::build_pipeline() const {
    if (!config_.pipeline.empty()) {
        return config_.pipeline;
    }

    std::ostringstream oss;
    oss << build_source_element() << " ! ";
    
    // Add encoding for streaming targets
    if (config_.target != "window") {
        oss << build_encode_element() << " ! ";
    }
    
    oss << build_sink_element();
    
    return oss.str();
}

std::string GstreamerDisplay::build_source_element() const {
    std::ostringstream oss;
    
    oss << "appsrc name=src "
        << "is-live=true "
        << "format=time "
        << "block=false ";
        
    // Caps will be set dynamically when we know the frame size
    
    return oss.str();
}

std::string GstreamerDisplay::build_encode_element() const {
    std::ostringstream oss;
    
    oss << "videoconvert ! ";
    
    if (config_.codec == "mjpeg") {
        oss << "jpegenc quality=85";
    } else {
        // H.264 encoding
        if (config_.hw_encode) {
#ifdef __aarch64__
            // NVIDIA Jetson
            oss << "nvvidconv ! nvv4l2h264enc bitrate=" 
                << (config_.bitrate_kbps * 1000);
#else
            // Try VAAPI on x86, fallback to software
            oss << "vaapih264enc bitrate=" << config_.bitrate_kbps;
#endif
        } else {
            oss << "x264enc tune=zerolatency bitrate=" << config_.bitrate_kbps
                << " speed-preset=ultrafast";
        }
    }
    
    return oss.str();
}

std::string GstreamerDisplay::build_sink_element() const {
    std::ostringstream oss;
    
    if (config_.target == "window") {
        oss << "videoconvert ! autovideosink sync=false";
    } else if (config_.target == "udp") {
        if (config_.codec == "mjpeg") {
            oss << "rtpjpegpay ! udpsink host=" << config_.host 
                << " port=" << config_.port << " sync=false";
        } else {
            oss << "h264parse ! rtph264pay config-interval=1 pt=96 ! "
                << "udpsink host=" << config_.host 
                << " port=" << config_.port << " sync=false";
        }
    } else if (config_.target == "rtsp") {
        // For RTSP, typically use rtsp-simple-server or similar
        // Here we output to UDP which can be picked up by an RTSP server
        oss << "h264parse ! rtph264pay config-interval=1 pt=96 ! "
            << "udpsink host=127.0.0.1 port=" << config_.port << " sync=false";
    } else {
        // Fallback to window
        oss << "videoconvert ! autovideosink sync=false";
    }
    
    return oss.str();
}

// ============================================================================
// Pipeline Management
// ============================================================================

bool GstreamerDisplay::create_pipeline() {
    gst_->pipeline_str = build_pipeline();
    LOG_DEBUG("GstreamerDisplay: Creating pipeline: {}", gst_->pipeline_str);

    GError* error = nullptr;
    gst_->pipeline = gst_parse_launch(gst_->pipeline_str.c_str(), &error);
    
    if (!gst_->pipeline || error) {
        LOG_ERROR("GstreamerDisplay: Failed to create pipeline: {}",
                  error ? error->message : "unknown");
        if (error) g_error_free(error);
        return false;
    }

    // Get appsrc element
    gst_->appsrc = gst_bin_get_by_name(GST_BIN(gst_->pipeline), "src");
    if (!gst_->appsrc) {
        LOG_ERROR("GstreamerDisplay: Failed to get appsrc element");
        destroy_pipeline();
        return false;
    }

    // Set caps on appsrc
    std::ostringstream caps_str;
    caps_str << "video/x-raw,format=BGR,width=" << gst_->width
             << ",height=" << gst_->height
             << ",framerate=" << config_.max_fps << "/1";
    
    GstCaps* caps = gst_caps_from_string(caps_str.str().c_str());
    if (caps) {
        gst_app_src_set_caps(GST_APP_SRC(gst_->appsrc), caps);
        gst_caps_unref(caps);
        gst_->caps_set = true;
    }

    // Configure appsrc
    g_object_set(G_OBJECT(gst_->appsrc),
                 "stream-type", GST_APP_STREAM_TYPE_STREAM,
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 nullptr);

    // Get bus for error handling
    gst_->bus = gst_element_get_bus(gst_->pipeline);

    // Start pipeline
    GstStateChangeReturn ret = gst_element_set_state(gst_->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("GstreamerDisplay: Failed to start pipeline");
        destroy_pipeline();
        return false;
    }

    LOG_INFO("GstreamerDisplay: Pipeline created successfully");
    return true;
}

void GstreamerDisplay::destroy_pipeline() {
    if (gst_->bus) {
        gst_object_unref(gst_->bus);
        gst_->bus = nullptr;
    }

    if (gst_->appsrc) {
        // Send EOS
        gst_app_src_end_of_stream(GST_APP_SRC(gst_->appsrc));
        gst_object_unref(gst_->appsrc);
        gst_->appsrc = nullptr;
    }

    if (gst_->pipeline) {
        gst_element_set_state(gst_->pipeline, GST_STATE_NULL);
        gst_object_unref(gst_->pipeline);
        gst_->pipeline = nullptr;
    }

    gst_->caps_set = false;
}

bool GstreamerDisplay::push_buffer(const uint8_t* data, size_t size,
                                   uint32_t width, uint32_t height)
{
    if (!gst_->appsrc) {
        return false;
    }

    // Allocate buffer
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    if (!buffer) {
        LOG_WARN("GstreamerDisplay: Failed to allocate buffer");
        return false;
    }

    // Copy frame data
    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        std::memcpy(map.data, data, size);
        gst_buffer_unmap(buffer, &map);
    } else {
        gst_buffer_unref(buffer);
        return false;
    }

    // Set buffer timestamps
    static uint64_t frame_count = 0;
    GstClockTime duration = GST_SECOND / config_.max_fps;
    GST_BUFFER_PTS(buffer) = frame_count * duration;
    GST_BUFFER_DURATION(buffer) = duration;
    frame_count++;

    // Push buffer
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(gst_->appsrc), buffer);
    
    if (ret != GST_FLOW_OK) {
        LOG_WARN("GstreamerDisplay: Failed to push buffer: {}", static_cast<int>(ret));
        return false;
    }

    return true;
}

bool GstreamerDisplay::should_process_frame() {
    auto now = Clock::now();
    auto elapsed = now - last_push_time_;
    return elapsed >= min_frame_interval_;
}

}  // namespace lagari

#else  // !HAS_GSTREAMER

namespace lagari {

// Stub implementation when GStreamer is not available
struct GstreamerDisplay::GstState {};

GstreamerDisplay::GstreamerDisplay(const DisplayConfig& config)
    : config_(config), overlay_renderer_(config.overlay) {}
GstreamerDisplay::~GstreamerDisplay() = default;

bool GstreamerDisplay::initialize(const Config&) {
    LOG_WARN("GstreamerDisplay: Not available (compile with HAS_GSTREAMER)");
    return false;
}

void GstreamerDisplay::start() {}
void GstreamerDisplay::stop() {}
bool GstreamerDisplay::is_running() const { return false; }

void GstreamerDisplay::push_frame(const Frame&, const DetectionResult*,
                                  SystemState, Duration) {}
bool GstreamerDisplay::is_enabled() const { return false; }
void GstreamerDisplay::set_enabled(bool) {}
GstreamerDisplay::DisplayStats GstreamerDisplay::get_stats() const { return {}; }

std::string GstreamerDisplay::build_pipeline() const { return ""; }
std::string GstreamerDisplay::build_source_element() const { return ""; }
std::string GstreamerDisplay::build_encode_element() const { return ""; }
std::string GstreamerDisplay::build_sink_element() const { return ""; }
bool GstreamerDisplay::create_pipeline() { return false; }
void GstreamerDisplay::destroy_pipeline() {}
bool GstreamerDisplay::push_buffer(const uint8_t*, size_t, uint32_t, uint32_t) { return false; }
bool GstreamerDisplay::should_process_frame() { return false; }

}  // namespace lagari

#endif  // HAS_GSTREAMER
