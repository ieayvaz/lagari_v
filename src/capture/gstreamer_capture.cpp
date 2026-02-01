#include "lagari/capture/gstreamer_capture.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#ifdef HAS_GSTREAMER

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include <cstring>
#include <sstream>

namespace lagari {

// ============================================================================
// GStreamer State (PIMPL)
// ============================================================================

struct GstreamerCapture::GstState {
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* decode = nullptr;
    GstElement* convert = nullptr;
    GstElement* capsfilter = nullptr;
    GstElement* appsink = nullptr;
    
    GstBus* bus = nullptr;
    GMainLoop* loop = nullptr;
    
    std::string pipeline_str;
    bool initialized = false;
};

// ============================================================================
// Helper function to convert sample (internal use only)
// ============================================================================

static FramePtr convert_gst_sample(GstSample* sample, CaptureConfig& config, uint64_t& frame_counter) {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        return nullptr;
    }

    GstCaps* caps = gst_sample_get_caps(sample);
    if (!caps) {
        return nullptr;
    }

    // Get video info
    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, caps)) {
        return nullptr;
    }

    // Update config with actual dimensions
    config.width = GST_VIDEO_INFO_WIDTH(&info);
    config.height = GST_VIDEO_INFO_HEIGHT(&info);

    // Map buffer
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        return nullptr;
    }

    // Create frame
    auto frame = std::make_shared<Frame>(config.width, config.height, PixelFormat::BGR24);
    frame->metadata.frame_id = ++frame_counter;
    frame->metadata.timestamp = Clock::now();

    // Copy data
    size_t expected_size = config.width * config.height * 3;
    size_t copy_size = std::min(static_cast<size_t>(map.size), expected_size);
    std::memcpy(frame->ptr(), map.data, copy_size);

    gst_buffer_unmap(buffer, &map);

    return frame;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

GstreamerCapture::GstreamerCapture(const CaptureConfig& config)
    : config_(config)
    , gst_(std::make_unique<GstState>())
{
}

GstreamerCapture::GstreamerCapture(const CaptureConfig& config, const GstConfig& gst_config)
    : config_(config)
    , gst_config_(gst_config)
    , gst_(std::make_unique<GstState>())
{
}

GstreamerCapture::~GstreamerCapture() {
    stop();
    destroy_pipeline();
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool GstreamerCapture::initialize(const Config& config) {
    // Initialize GStreamer
    GError* error = nullptr;
    if (!gst_init_check(nullptr, nullptr, &error)) {
        LOG_ERROR("GstreamerCapture: Failed to initialize GStreamer: {}", 
                  error ? error->message : "unknown");
        if (error) g_error_free(error);
        return false;
    }

    // Parse GStreamer-specific config
    std::string source_type_str = config.get_string("capture.gstreamer.source_type", "auto");
    if (source_type_str == "v4l2") gst_config_.source_type = GstConfig::SourceType::V4L2;
    else if (source_type_str == "rtsp") gst_config_.source_type = GstConfig::SourceType::RTSP;
    else if (source_type_str == "file") gst_config_.source_type = GstConfig::SourceType::FILE;
    else if (source_type_str == "http") gst_config_.source_type = GstConfig::SourceType::HTTP;
    else if (source_type_str == "test") gst_config_.source_type = GstConfig::SourceType::TEST;
    else if (source_type_str == "uri") gst_config_.source_type = GstConfig::SourceType::URI;
    else if (source_type_str == "argus") gst_config_.source_type = GstConfig::SourceType::ARGUS;
    else if (source_type_str == "libcamera") gst_config_.source_type = GstConfig::SourceType::LIBCAMERA;
    else gst_config_.source_type = GstConfig::SourceType::AUTO;

    gst_config_.pipeline = config.get_string("capture.gstreamer.pipeline", "");
    gst_config_.uri = config.get_string("capture.gstreamer.uri", config_.file_path);
    gst_config_.device = config.get_string("capture.device", "/dev/video0");
    gst_config_.latency_ms = config.get_int("capture.gstreamer.latency_ms", 200);
    gst_config_.tcp_timeout = config.get_int("capture.gstreamer.tcp_timeout", 5000000);
    gst_config_.use_tcp = config.get_bool("capture.gstreamer.use_tcp", true);
    gst_config_.drop_on_latency = config.get_bool("capture.gstreamer.drop_on_latency", true);
    gst_config_.hw_decode = config.get_bool("capture.gstreamer.hw_decode", true);
    gst_config_.decoder = config.get_string("capture.gstreamer.decoder", "");
    gst_config_.sync = config.get_bool("capture.gstreamer.sync", false);
    gst_config_.queue_size = config.get_int("capture.gstreamer.queue_size", 2);
    gst_config_.loop = config.get_bool("capture.loop_file", true);

    // Auto-detect source type from URI/config
    if (gst_config_.source_type == GstConfig::SourceType::AUTO) {
        if (!gst_config_.uri.empty()) {
            if (gst_config_.uri.find("rtsp://") == 0) {
                gst_config_.source_type = GstConfig::SourceType::RTSP;
            } else if (gst_config_.uri.find("http://") == 0 ||
                       gst_config_.uri.find("https://") == 0) {
                gst_config_.source_type = GstConfig::SourceType::HTTP;
            } else {
                gst_config_.source_type = GstConfig::SourceType::FILE;
            }
        } else if (!gst_config_.device.empty()) {
            gst_config_.source_type = GstConfig::SourceType::V4L2;
        } else {
            gst_config_.source_type = GstConfig::SourceType::TEST;
        }
    }

    // Create pipeline
    if (!create_pipeline()) {
        LOG_ERROR("GstreamerCapture: Failed to create pipeline");
        return false;
    }

    gst_->initialized = true;
    LOG_INFO("GstreamerCapture: Initialized with pipeline: {}", gst_->pipeline_str);

    return true;
}

void GstreamerCapture::start() {
    if (running_.load(std::memory_order_acquire)) {
        return;
    }

    if (!gst_->initialized) {
        LOG_ERROR("GstreamerCapture: Not initialized");
        return;
    }

    // Set pipeline to playing
    GstStateChangeReturn ret = gst_element_set_state(gst_->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("GstreamerCapture: Failed to start pipeline");
        return;
    }

    should_stop_.store(false, std::memory_order_release);
    eos_.store(false, std::memory_order_release);
    start_time_ = Clock::now();
    last_frame_time_ = start_time_;

    process_thread_ = std::thread(&GstreamerCapture::process_loop, this);
    running_.store(true, std::memory_order_release);

    LOG_INFO("GstreamerCapture: Started");
}

void GstreamerCapture::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    should_stop_.store(true, std::memory_order_release);
    frame_cv_.notify_all();

    if (gst_->pipeline) {
        gst_element_set_state(gst_->pipeline, GST_STATE_NULL);
    }

    if (process_thread_.joinable()) {
        process_thread_.join();
    }

    running_.store(false, std::memory_order_release);
    LOG_INFO("GstreamerCapture: Stopped");
}

bool GstreamerCapture::is_running() const {
    return running_.load(std::memory_order_acquire);
}

// ============================================================================
// ICapture Implementation
// ============================================================================

FramePtr GstreamerCapture::get_latest_frame() {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

FramePtr GstreamerCapture::wait_for_frame(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(frame_mutex_);

    auto current_id = latest_frame_ ? latest_frame_->metadata.frame_id : 0;
    
    frame_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, current_id]() {
        return (latest_frame_ && latest_frame_->metadata.frame_id > current_id) ||
               should_stop_.load(std::memory_order_acquire);
    });

    return latest_frame_;
}

void GstreamerCapture::set_frame_callback(FrameCallback callback) {
    frame_callback_ = std::move(callback);
}

CaptureStats GstreamerCapture::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool GstreamerCapture::is_open() const {
    return gst_ && gst_->initialized;
}

bool GstreamerCapture::set_resolution(uint32_t width, uint32_t height) {
    LOG_WARN("GstreamerCapture: Runtime resolution change not supported");
    config_.width = width;
    config_.height = height;
    return false;
}

bool GstreamerCapture::set_framerate(uint32_t fps) {
    LOG_WARN("GstreamerCapture: Runtime framerate change not supported");
    config_.fps = fps;
    return false;
}

bool GstreamerCapture::set_exposure(bool /* auto_exp */, float /* exposure_time */) {
    LOG_WARN("GstreamerCapture: Exposure control not supported");
    return false;
}

// ============================================================================
// GStreamer-specific Methods
// ============================================================================

bool GstreamerCapture::set_pipeline(const std::string& pipeline) {
    if (running_.load(std::memory_order_acquire)) {
        LOG_WARN("GstreamerCapture: Cannot change pipeline while running");
        return false;
    }

    gst_config_.pipeline = pipeline;
    destroy_pipeline();
    return create_pipeline();
}

std::string GstreamerCapture::get_pipeline() const {
    return gst_->pipeline_str;
}

bool GstreamerCapture::seek(int64_t position_ns) {
    if (!gst_->pipeline) {
        return false;
    }

    return gst_element_seek_simple(
        gst_->pipeline,
        GST_FORMAT_TIME,
        static_cast<GstSeekFlags>(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT),
        position_ns);
}

int64_t GstreamerCapture::get_position() const {
    if (!gst_->pipeline) {
        return -1;
    }

    gint64 position = 0;
    if (!gst_element_query_position(gst_->pipeline, GST_FORMAT_TIME, &position)) {
        return -1;
    }
    return position;
}

int64_t GstreamerCapture::get_duration() const {
    if (!gst_->pipeline) {
        return -1;
    }

    gint64 duration = 0;
    if (!gst_element_query_duration(gst_->pipeline, GST_FORMAT_TIME, &duration)) {
        return -1;
    }
    return duration;
}

bool GstreamerCapture::is_eos() const {
    return eos_.load(std::memory_order_acquire);
}

// ============================================================================
// Pipeline Building
// ============================================================================

std::string GstreamerCapture::build_pipeline() const {
    // Use custom pipeline if provided
    if (!gst_config_.pipeline.empty()) {
        return gst_config_.pipeline;
    }

    std::ostringstream oss;
    oss << build_source_element() << " ! ";
    oss << build_decode_element() << " ! ";
    oss << build_convert_element() << " ! ";
    oss << build_sink_element();
    
    return oss.str();
}

std::string GstreamerCapture::build_source_element() const {
    std::ostringstream oss;

    switch (gst_config_.source_type) {
        case GstConfig::SourceType::V4L2:
            oss << "v4l2src device=" << gst_config_.device;
            break;

        case GstConfig::SourceType::RTSP:
            oss << "rtspsrc location=\"" << gst_config_.uri << "\" "
                << "latency=" << gst_config_.latency_ms << " "
                << "tcp-timeout=" << gst_config_.tcp_timeout << " "
                << "protocols=" << (gst_config_.use_tcp ? "tcp" : "udp") << " "
                << "drop-on-latency=" << (gst_config_.drop_on_latency ? "true" : "false")
                << " ! rtph264depay ! h264parse";
            break;

        case GstConfig::SourceType::FILE:
            oss << "filesrc location=\"" << gst_config_.uri << "\"";
            break;

        case GstConfig::SourceType::HTTP:
            oss << "souphttpsrc location=\"" << gst_config_.uri << "\"";
            break;

        case GstConfig::SourceType::URI:
            oss << "uridecodebin uri=\"" << gst_config_.uri << "\"";
            break;

        case GstConfig::SourceType::ARGUS:
            oss << "nvarguscamerasrc sensor-id=" << config_.camera_id << " "
                << "! video/x-raw(memory:NVMM),width=" << config_.width 
                << ",height=" << config_.height 
                << ",framerate=" << config_.fps << "/1";
            break;

        case GstConfig::SourceType::LIBCAMERA:
            oss << "libcamerasrc camera-name=" << config_.camera_id << " "
                << "! video/x-raw,width=" << config_.width 
                << ",height=" << config_.height;
            break;

        case GstConfig::SourceType::TEST:
        default:
            oss << "videotestsrc pattern=ball "
                << "! video/x-raw,width=" << config_.width 
                << ",height=" << config_.height 
                << ",framerate=" << config_.fps << "/1";
            break;
    }

    return oss.str();
}

std::string GstreamerCapture::build_decode_element() const {
    std::ostringstream oss;

    // Some sources already decoded or need specific decoders
    switch (gst_config_.source_type) {
        case GstConfig::SourceType::V4L2:
        case GstConfig::SourceType::TEST:
        case GstConfig::SourceType::ARGUS:
        case GstConfig::SourceType::LIBCAMERA:
            // Already raw video
            oss << "queue";
            break;

        case GstConfig::SourceType::RTSP:
            // H264 decode
            if (!gst_config_.decoder.empty()) {
                oss << gst_config_.decoder;
            } else if (gst_config_.hw_decode) {
#ifdef __aarch64__
                // Try NVIDIA decoder on Jetson
                oss << "nvv4l2decoder";
#else
                // Try VAAPI on x86
                oss << "vaapih264dec";
#endif
            } else {
                oss << "avdec_h264";
            }
            break;

        case GstConfig::SourceType::FILE:
        case GstConfig::SourceType::HTTP:
        case GstConfig::SourceType::URI:
        default:
            // Generic decoding
            oss << "decodebin";
            break;
    }

    return oss.str();
}

std::string GstreamerCapture::build_convert_element() const {
    std::ostringstream oss;

#ifdef __aarch64__
    // Use NVIDIA converter on Jetson if NVMM memory
    if (gst_config_.source_type == GstConfig::SourceType::ARGUS ||
        (gst_config_.source_type == GstConfig::SourceType::RTSP && gst_config_.hw_decode)) {
        oss << "nvvidconv ! video/x-raw,format=BGRx ! videoconvert";
    } else {
        oss << "videoconvert";
    }
#else
    oss << "videoconvert";
#endif

    oss << " ! video/x-raw,format=BGR";

    return oss.str();
}

std::string GstreamerCapture::build_sink_element() const {
    std::ostringstream oss;

    oss << "appsink name=sink "
        << "max-buffers=" << gst_config_.queue_size << " "
        << "drop=true "
        << "sync=" << (gst_config_.sync ? "true" : "false") << " "
        << "emit-signals=false";

    return oss.str();
}

// ============================================================================
// GStreamer Callbacks (using internal function names for header-safe declarations)
// ============================================================================

int GstreamerCapture::on_new_sample_cb(void* sink_ptr, void* user_data) {
    auto* sink = static_cast<GstAppSink*>(sink_ptr);
    auto* capture = static_cast<GstreamerCapture*>(user_data);
    
    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (!sample) {
        return GST_FLOW_ERROR;
    }

    FramePtr frame = convert_gst_sample(sample, capture->config_, capture->frame_counter_);
    gst_sample_unref(sample);

    if (frame) {
        // Store latest frame
        {
            std::lock_guard<std::mutex> lock(capture->frame_mutex_);
            capture->latest_frame_ = frame;
        }
        capture->frame_cv_.notify_all();

        // Call callback
        if (capture->frame_callback_) {
            capture->frame_callback_(frame);
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(capture->stats_mutex_);
            capture->stats_.frames_captured++;

            auto now = Clock::now();
            auto elapsed = std::chrono::duration<float>(now - capture->start_time_).count();
            if (elapsed > 0) {
                capture->stats_.average_fps = capture->stats_.frames_captured / elapsed;
            }

            auto frame_time = std::chrono::duration<float>(now - capture->last_frame_time_).count();
            if (frame_time > 0) {
                capture->stats_.current_fps = 1.0f / frame_time;
            }
            capture->last_frame_time_ = now;
        }
    }

    return GST_FLOW_OK;
}

int GstreamerCapture::on_bus_message_cb(void* /* bus_ptr */, void* message_ptr, void* user_data) {
    auto* capture = static_cast<GstreamerCapture*>(user_data);
    GstMessage* msg = static_cast<GstMessage*>(message_ptr);

    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError* err = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_error(msg, &err, &debug);
            LOG_ERROR("GstreamerCapture: Pipeline error: {}", err ? err->message : "unknown");
            if (debug) {
                LOG_DEBUG("GstreamerCapture: Debug: {}", debug);
            }
            if (err) g_error_free(err);
            if (debug) g_free(debug);
            break;
        }

        case GST_MESSAGE_WARNING: {
            GError* err = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_warning(msg, &err, &debug);
            LOG_WARN("GstreamerCapture: Pipeline warning: {}", err ? err->message : "unknown");
            if (err) g_error_free(err);
            if (debug) g_free(debug);
            break;
        }

        case GST_MESSAGE_EOS:
            LOG_INFO("GstreamerCapture: End of stream");
            capture->eos_.store(true, std::memory_order_release);
            
            // Loop if configured
            if (capture->gst_config_.loop && 
                capture->gst_config_.source_type == GstConfig::SourceType::FILE) {
                capture->seek(0);
                capture->eos_.store(false, std::memory_order_release);
            }
            break;

        case GST_MESSAGE_STATE_CHANGED:
            // Ignore state change messages
            break;

        default:
            break;
    }

    return TRUE;
}

void GstreamerCapture::on_pad_added_cb(void* /* element */, void* /* pad */, void* /* user_data */) {
    // Dynamic pad linking for decodebin - implement if needed
}

// ============================================================================
// Pipeline Management
// ============================================================================

bool GstreamerCapture::create_pipeline() {
    // Build pipeline string
    gst_->pipeline_str = build_pipeline();
    LOG_DEBUG("GstreamerCapture: Creating pipeline: {}", gst_->pipeline_str);

    // Parse pipeline
    GError* error = nullptr;
    gst_->pipeline = gst_parse_launch(gst_->pipeline_str.c_str(), &error);
    
    if (!gst_->pipeline || error) {
        LOG_ERROR("GstreamerCapture: Failed to create pipeline: {}", 
                  error ? error->message : "unknown");
        if (error) g_error_free(error);
        return false;
    }

    // Get appsink
    gst_->appsink = gst_bin_get_by_name(GST_BIN(gst_->pipeline), "sink");
    if (!gst_->appsink) {
        LOG_ERROR("GstreamerCapture: Failed to get appsink");
        destroy_pipeline();
        return false;
    }

    // Configure appsink callbacks
    GstAppSinkCallbacks callbacks = {};
    callbacks.new_sample = reinterpret_cast<GstFlowReturn(*)(GstAppSink*, gpointer)>(on_new_sample_cb);
    gst_app_sink_set_callbacks(GST_APP_SINK(gst_->appsink), &callbacks, this, nullptr);

    // Set up bus watch
    gst_->bus = gst_element_get_bus(gst_->pipeline);
    gst_bus_add_watch(gst_->bus, reinterpret_cast<GstBusFunc>(on_bus_message_cb), this);

    return true;
}

bool GstreamerCapture::link_pipeline() {
    // Not used when using gst_parse_launch, but kept for completeness
    return true;
}

void GstreamerCapture::destroy_pipeline() {
    if (gst_->bus) {
        gst_bus_remove_watch(gst_->bus);
        gst_object_unref(gst_->bus);
        gst_->bus = nullptr;
    }

    if (gst_->appsink) {
        gst_object_unref(gst_->appsink);
        gst_->appsink = nullptr;
    }

    if (gst_->pipeline) {
        gst_element_set_state(gst_->pipeline, GST_STATE_NULL);
        gst_object_unref(gst_->pipeline);
        gst_->pipeline = nullptr;
    }

    gst_->initialized = false;
}

// ============================================================================
// Frame Processing
// ============================================================================

void GstreamerCapture::process_loop() {
    LOG_DEBUG("GstreamerCapture: Process thread started");

    // The main loop is handled by GStreamer internally
    // We just wait for stop signal
    while (!should_stop_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    LOG_DEBUG("GstreamerCapture: Process thread exiting");
}

FramePtr GstreamerCapture::convert_sample(void* sample_ptr) {
    GstSample* sample = static_cast<GstSample*>(sample_ptr);
    return convert_gst_sample(sample, config_, frame_counter_);
}

}  // namespace lagari

#else  // !HAS_GSTREAMER

namespace lagari {

// Stub implementation when GStreamer is not available
struct GstreamerCapture::GstState {};

GstreamerCapture::GstreamerCapture(const CaptureConfig& config) : config_(config) {}
GstreamerCapture::GstreamerCapture(const CaptureConfig& config, const GstConfig&) : config_(config) {}
GstreamerCapture::~GstreamerCapture() = default;

bool GstreamerCapture::initialize(const Config&) {
    LOG_ERROR("GstreamerCapture: Not available (compile with HAS_GSTREAMER)");
    return false;
}

void GstreamerCapture::start() {}
void GstreamerCapture::stop() {}
bool GstreamerCapture::is_running() const { return false; }
FramePtr GstreamerCapture::get_latest_frame() { return nullptr; }
FramePtr GstreamerCapture::wait_for_frame(uint32_t) { return nullptr; }
void GstreamerCapture::set_frame_callback(FrameCallback) {}
CaptureStats GstreamerCapture::get_stats() const { return {}; }
bool GstreamerCapture::is_open() const { return false; }
bool GstreamerCapture::set_resolution(uint32_t, uint32_t) { return false; }
bool GstreamerCapture::set_framerate(uint32_t) { return false; }
bool GstreamerCapture::set_exposure(bool, float) { return false; }
bool GstreamerCapture::set_pipeline(const std::string&) { return false; }
std::string GstreamerCapture::get_pipeline() const { return ""; }
bool GstreamerCapture::seek(int64_t) { return false; }
int64_t GstreamerCapture::get_position() const { return -1; }
int64_t GstreamerCapture::get_duration() const { return -1; }
bool GstreamerCapture::is_eos() const { return false; }
std::string GstreamerCapture::build_pipeline() const { return ""; }
std::string GstreamerCapture::build_source_element() const { return ""; }
std::string GstreamerCapture::build_decode_element() const { return ""; }
std::string GstreamerCapture::build_convert_element() const { return ""; }
std::string GstreamerCapture::build_sink_element() const { return ""; }
int GstreamerCapture::on_new_sample_cb(void*, void*) { return 0; }
void GstreamerCapture::on_pad_added_cb(void*, void*, void*) {}
int GstreamerCapture::on_bus_message_cb(void*, void*, void*) { return 0; }
bool GstreamerCapture::create_pipeline() { return false; }
bool GstreamerCapture::link_pipeline() { return false; }
void GstreamerCapture::destroy_pipeline() {}
void GstreamerCapture::process_loop() {}
FramePtr GstreamerCapture::convert_sample(void*) { return nullptr; }

}  // namespace lagari

#endif  // HAS_GSTREAMER
