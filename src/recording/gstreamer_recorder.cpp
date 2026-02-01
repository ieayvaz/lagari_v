/**
 * @file gstreamer_recorder.cpp
 * @brief GStreamer-based video recorder implementation
 */

#include "lagari/recording/gstreamer_recorder.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace fs = std::filesystem;

#ifdef HAS_GSTREAMER

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>

namespace lagari {

// ============================================================================
// GStreamer State (PIMPL)
// ============================================================================

struct GstreamerRecorder::GstState {
    GstElement* pipeline = nullptr;
    GstElement* appsrc = nullptr;
    GstBus* bus = nullptr;
    
    std::string pipeline_str;
    bool initialized = false;
    
    // Format info (set on first frame)
    uint32_t width = 0;
    uint32_t height = 0;
    bool caps_set = false;
    
    // Frame counter for timestamps
    uint64_t frame_count = 0;
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

GstreamerRecorder::GstreamerRecorder(const RecordingConfig& config)
    : config_(config)
    , gst_(std::make_unique<GstState>())
    , overlay_renderer_(config.overlay)
{
}

GstreamerRecorder::~GstreamerRecorder() {
    stop_recording();
    stop();
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool GstreamerRecorder::initialize(const Config& config) {
    // Parse configuration
    config_.enabled = config.get_bool("recording.enabled", false);
    config_.output_dir = config.get_string("recording.output_dir", "/var/lagari/recordings");
    config_.codec = config.get_string("recording.codec", "h264");
    config_.bitrate_kbps = config.get_int("recording.bitrate_kbps", 8000);
    config_.fps = config.get_int("recording.fps", 30);
    config_.hw_encode = config.get_bool("recording.hw_encode", true);
    config_.container = config.get_string("recording.container", "mp4");
    config_.segment_duration_s = config.get_int("recording.segment_duration_s", 0);
    config_.max_storage_bytes = static_cast<uint64_t>(
        config.get_double("recording.max_storage_gb", 10.0) * 1024 * 1024 * 1024);
    config_.delete_oldest = config.get_bool("recording.delete_oldest", true);
    
    // Overlay config
    config_.overlay.enabled = config.get_bool("recording.overlay.enabled", true);
    config_.overlay.timestamp = config.get_bool("recording.overlay.timestamp", true);
    config_.overlay.bounding_boxes = config.get_bool("recording.overlay.bounding_boxes", true);
    config_.overlay.state = config.get_bool("recording.overlay.state", true);
    config_.overlay.latency = config.get_bool("recording.overlay.latency", true);
    
    overlay_renderer_.set_config(config_.overlay);

    if (!config_.enabled) {
        LOG_INFO("GstreamerRecorder: Disabled by configuration");
        return true;
    }

    // Create output directory
    try {
        fs::create_directories(config_.output_dir);
    } catch (const std::exception& e) {
        LOG_ERROR("GstreamerRecorder: Failed to create output directory '{}': {}",
                  config_.output_dir, e.what());
        return false;
    }

    // Initialize GStreamer
    GError* error = nullptr;
    if (!gst_init_check(nullptr, nullptr, &error)) {
        LOG_ERROR("GstreamerRecorder: Failed to initialize GStreamer: {}",
                  error ? error->message : "unknown");
        if (error) g_error_free(error);
        return false;
    }

    gst_->initialized = true;
    initialized_.store(true, std::memory_order_release);
    
    LOG_INFO("GstreamerRecorder: Initialized, output dir: {}", config_.output_dir);
    return true;
}

void GstreamerRecorder::start() {
    if (!initialized_.load(std::memory_order_acquire)) {
        return;
    }
    running_.store(true, std::memory_order_release);
    LOG_INFO("GstreamerRecorder: Started (ready to record)");
}

void GstreamerRecorder::stop() {
    stop_recording();
    running_.store(false, std::memory_order_release);
    LOG_INFO("GstreamerRecorder: Stopped");
}

bool GstreamerRecorder::is_running() const {
    return running_.load(std::memory_order_acquire);
}

// ============================================================================
// IRecorder Implementation
// ============================================================================

bool GstreamerRecorder::start_recording(const std::string& filename) {
    if (!initialized_.load(std::memory_order_acquire)) {
        LOG_ERROR("GstreamerRecorder: Not initialized");
        return false;
    }

    if (recording_.load(std::memory_order_acquire)) {
        LOG_WARN("GstreamerRecorder: Already recording");
        return false;
    }

    // Check storage before starting
    check_storage();

    // Generate filename if not provided
    std::string output_file = filename.empty() ? generate_filename() : filename;
    
    // Ensure full path
    if (!fs::path(output_file).is_absolute()) {
        output_file = (fs::path(config_.output_dir) / output_file).string();
    }

    {
        std::lock_guard<std::mutex> lock(info_mutex_);
        current_filename_ = output_file;
        bytes_written_ = 0;
        frames_recorded_ = 0;
    }

    // Pipeline will be created on first frame when we know dimensions
    recording_start_ = Clock::now();
    recording_.store(true, std::memory_order_release);
    
    LOG_INFO("GstreamerRecorder: Started recording to {}", output_file);
    return true;
}

void GstreamerRecorder::stop_recording() {
    if (!recording_.load(std::memory_order_acquire)) {
        return;
    }

    recording_.store(false, std::memory_order_release);
    
    destroy_pipeline();

    std::lock_guard<std::mutex> lock(info_mutex_);
    auto duration = recording_duration();
    LOG_INFO("GstreamerRecorder: Stopped recording. Duration: {:.1f}s, Frames: {}, Size: {} bytes",
             duration, frames_recorded_, bytes_written_);
}

bool GstreamerRecorder::is_recording() const {
    return recording_.load(std::memory_order_acquire);
}

void GstreamerRecorder::add_frame(const Frame& frame, const DetectionResult* detections) {
    if (!recording_.load(std::memory_order_acquire)) {
        return;
    }

    // Create pipeline on first frame if needed
    if (!gst_->pipeline) {
        gst_->width = frame.metadata.width;
        gst_->height = frame.metadata.height;
        
        std::string filename;
        {
            std::lock_guard<std::mutex> lock(info_mutex_);
            filename = current_filename_;
        }
        
        if (!create_pipeline(filename)) {
            LOG_ERROR("GstreamerRecorder: Failed to create pipeline");
            recording_.store(false, std::memory_order_release);
            return;
        }
    }

    // Check if dimensions changed (shouldn't happen, but handle gracefully)
    if (frame.metadata.width != gst_->width || frame.metadata.height != gst_->height) {
        LOG_ERROR("GstreamerRecorder: Frame dimensions changed during recording");
        return;
    }

    // Apply overlay if enabled
    cv::Mat output_mat;
    if (config_.overlay.enabled) {
        cv::Mat input_mat = frame_to_mat(frame);
        if (input_mat.empty()) {
            return;
        }
        output_mat = input_mat.clone();
        overlay_renderer_.render_inplace(output_mat, detections, current_state_, current_latency_);
    } else {
        output_mat = frame_to_mat(frame);
        if (output_mat.empty()) {
            return;
        }
        if (!output_mat.isContinuous()) {
            output_mat = output_mat.clone();
        }
    }

    // Push to GStreamer
    if (push_buffer(output_mat.data, output_mat.total() * output_mat.elemSize(),
                   static_cast<uint32_t>(output_mat.cols),
                   static_cast<uint32_t>(output_mat.rows))) {
        std::lock_guard<std::mutex> lock(info_mutex_);
        frames_recorded_++;
        // Note: bytes_written is updated via pad probes or estimated
        bytes_written_ += output_mat.total() * output_mat.elemSize();
    }
}

void GstreamerRecorder::set_overlay_enabled(bool enabled) {
    config_.overlay.enabled = enabled;
    overlay_renderer_.set_config(config_.overlay);
}

std::string GstreamerRecorder::current_filename() const {
    std::lock_guard<std::mutex> lock(info_mutex_);
    return current_filename_;
}

double GstreamerRecorder::recording_duration() const {
    if (!recording_.load(std::memory_order_acquire)) {
        return 0.0;
    }
    return std::chrono::duration<double>(Clock::now() - recording_start_).count();
}

uint64_t GstreamerRecorder::bytes_written() const {
    std::lock_guard<std::mutex> lock(info_mutex_);
    return bytes_written_;
}

// ============================================================================
// Pipeline Building
// ============================================================================

std::string GstreamerRecorder::build_pipeline(const std::string& filename) const {
    std::ostringstream oss;
    
    oss << build_source_element() << " ! ";
    oss << "videoconvert ! ";
    oss << build_encode_element() << " ! ";
    oss << build_mux_element() << " ! ";
    oss << build_sink_element(filename);
    
    return oss.str();
}

std::string GstreamerRecorder::build_source_element() const {
    std::ostringstream oss;
    
    oss << "appsrc name=src "
        << "is-live=true "
        << "format=time "
        << "block=true ";  // Block if pipeline is slow
    
    return oss.str();
}

std::string GstreamerRecorder::build_encode_element() const {
    std::ostringstream oss;
    
    if (config_.codec == "h265" || config_.codec == "hevc") {
        if (config_.hw_encode) {
#ifdef __aarch64__
            oss << "nvv4l2h265enc bitrate=" << (config_.bitrate_kbps * 1000);
#else
            oss << "x265enc bitrate=" << config_.bitrate_kbps
                << " speed-preset=ultrafast tune=zerolatency";
#endif
        } else {
            oss << "x265enc bitrate=" << config_.bitrate_kbps
                << " speed-preset=ultrafast tune=zerolatency";
        }
        oss << " ! h265parse";
    } else {
        // Default to H.264
        if (config_.hw_encode) {
#ifdef __aarch64__
            oss << "nvv4l2h264enc bitrate=" << (config_.bitrate_kbps * 1000);
#else
            // Try VAAPI, but most systems will need x264enc
            oss << "x264enc bitrate=" << config_.bitrate_kbps
                << " speed-preset=ultrafast tune=zerolatency";
#endif
        } else {
            oss << "x264enc bitrate=" << config_.bitrate_kbps
                << " speed-preset=ultrafast tune=zerolatency";
        }
        oss << " ! h264parse";
    }
    
    return oss.str();
}

std::string GstreamerRecorder::build_mux_element() const {
    if (config_.container == "mkv") {
        return "matroskamux";
    } else if (config_.container == "ts") {
        return "mpegtsmux";
    } else {
        // Default to MP4
        return "mp4mux faststart=true";
    }
}

std::string GstreamerRecorder::build_sink_element(const std::string& filename) const {
    return "filesink location=\"" + filename + "\"";
}

// ============================================================================
// Pipeline Management
// ============================================================================

bool GstreamerRecorder::create_pipeline(const std::string& filename) {
    gst_->pipeline_str = build_pipeline(filename);
    LOG_DEBUG("GstreamerRecorder: Creating pipeline: {}", gst_->pipeline_str);

    GError* error = nullptr;
    gst_->pipeline = gst_parse_launch(gst_->pipeline_str.c_str(), &error);
    
    if (!gst_->pipeline || error) {
        LOG_ERROR("GstreamerRecorder: Failed to create pipeline: {}",
                  error ? error->message : "unknown");
        if (error) g_error_free(error);
        return false;
    }

    // Get appsrc element
    gst_->appsrc = gst_bin_get_by_name(GST_BIN(gst_->pipeline), "src");
    if (!gst_->appsrc) {
        LOG_ERROR("GstreamerRecorder: Failed to get appsrc element");
        destroy_pipeline();
        return false;
    }

    // Set caps on appsrc
    std::ostringstream caps_str;
    caps_str << "video/x-raw,format=BGR,width=" << gst_->width
             << ",height=" << gst_->height
             << ",framerate=" << config_.fps << "/1";
    
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

    // Reset frame counter
    gst_->frame_count = 0;

    // Start pipeline
    GstStateChangeReturn ret = gst_element_set_state(gst_->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("GstreamerRecorder: Failed to start pipeline");
        destroy_pipeline();
        return false;
    }

    LOG_INFO("GstreamerRecorder: Pipeline created for {}", filename);
    return true;
}

void GstreamerRecorder::destroy_pipeline() {
    if (gst_->bus) {
        gst_object_unref(gst_->bus);
        gst_->bus = nullptr;
    }

    if (gst_->appsrc) {
        // Send EOS and wait for it to propagate
        gst_app_src_end_of_stream(GST_APP_SRC(gst_->appsrc));
        
        // Wait for EOS to reach the sink (important for proper file closing)
        if (gst_->pipeline) {
            GstBus* bus = gst_element_get_bus(gst_->pipeline);
            if (bus) {
                GstMessage* msg = gst_bus_timed_pop_filtered(
                    bus, 5 * GST_SECOND,
                    static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
                if (msg) {
                    gst_message_unref(msg);
                }
                gst_object_unref(bus);
            }
        }
        
        gst_object_unref(gst_->appsrc);
        gst_->appsrc = nullptr;
    }

    if (gst_->pipeline) {
        gst_element_set_state(gst_->pipeline, GST_STATE_NULL);
        gst_object_unref(gst_->pipeline);
        gst_->pipeline = nullptr;
    }

    gst_->caps_set = false;
    gst_->frame_count = 0;
}

bool GstreamerRecorder::push_buffer(const uint8_t* data, size_t size,
                                    uint32_t width, uint32_t height) {
    if (!gst_->appsrc) {
        return false;
    }

    // Allocate buffer
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    if (!buffer) {
        LOG_WARN("GstreamerRecorder: Failed to allocate buffer");
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
    GstClockTime duration = GST_SECOND / config_.fps;
    GST_BUFFER_PTS(buffer) = gst_->frame_count * duration;
    GST_BUFFER_DURATION(buffer) = duration;
    gst_->frame_count++;

    // Push buffer
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(gst_->appsrc), buffer);
    
    if (ret != GST_FLOW_OK) {
        LOG_WARN("GstreamerRecorder: Failed to push buffer: {}", static_cast<int>(ret));
        return false;
    }

    return true;
}

// ============================================================================
// File Management
// ============================================================================

std::string GstreamerRecorder::generate_filename() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << "recording_";
    oss << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    
    if (config_.container == "mkv") {
        oss << ".mkv";
    } else if (config_.container == "ts") {
        oss << ".ts";
    } else {
        oss << ".mp4";
    }
    
    return oss.str();
}

void GstreamerRecorder::check_storage() {
    if (!config_.delete_oldest) {
        return;
    }

    try {
        uint64_t total_size = 0;
        std::vector<fs::directory_entry> recordings;

        for (const auto& entry : fs::directory_iterator(config_.output_dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".mp4" || ext == ".mkv" || ext == ".ts") {
                    total_size += entry.file_size();
                    recordings.push_back(entry);
                }
            }
        }

        // Sort by modification time (oldest first)
        std::sort(recordings.begin(), recordings.end(),
            [](const fs::directory_entry& a, const fs::directory_entry& b) {
                return fs::last_write_time(a) < fs::last_write_time(b);
            });

        // Delete oldest files until under limit
        while (total_size > config_.max_storage_bytes && !recordings.empty()) {
            const auto& oldest = recordings.front();
            uint64_t file_size = oldest.file_size();
            
            LOG_INFO("GstreamerRecorder: Deleting old recording: {}", oldest.path().string());
            fs::remove(oldest.path());
            
            total_size -= file_size;
            recordings.erase(recordings.begin());
        }
    } catch (const std::exception& e) {
        LOG_WARN("GstreamerRecorder: Failed to check storage: {}", e.what());
    }
}

void GstreamerRecorder::delete_oldest_recording() {
    // Called when we need to free up space immediately
    check_storage();
}

}  // namespace lagari

#else  // !HAS_GSTREAMER

namespace lagari {

// Stub implementation when GStreamer is not available
struct GstreamerRecorder::GstState {};

GstreamerRecorder::GstreamerRecorder(const RecordingConfig& config)
    : config_(config), overlay_renderer_(config.overlay) {}
GstreamerRecorder::~GstreamerRecorder() = default;

bool GstreamerRecorder::initialize(const Config&) {
    LOG_WARN("GstreamerRecorder: Not available (compile with HAS_GSTREAMER)");
    return false;
}

void GstreamerRecorder::start() {}
void GstreamerRecorder::stop() {}
bool GstreamerRecorder::is_running() const { return false; }
bool GstreamerRecorder::start_recording(const std::string&) { return false; }
void GstreamerRecorder::stop_recording() {}
bool GstreamerRecorder::is_recording() const { return false; }
void GstreamerRecorder::add_frame(const Frame&, const DetectionResult*) {}
void GstreamerRecorder::set_overlay_enabled(bool) {}
std::string GstreamerRecorder::current_filename() const { return ""; }
double GstreamerRecorder::recording_duration() const { return 0.0; }
uint64_t GstreamerRecorder::bytes_written() const { return 0; }

std::string GstreamerRecorder::build_pipeline(const std::string&) const { return ""; }
std::string GstreamerRecorder::build_source_element() const { return ""; }
std::string GstreamerRecorder::build_encode_element() const { return ""; }
std::string GstreamerRecorder::build_mux_element() const { return ""; }
std::string GstreamerRecorder::build_sink_element(const std::string&) const { return ""; }
bool GstreamerRecorder::create_pipeline(const std::string&) { return false; }
void GstreamerRecorder::destroy_pipeline() {}
bool GstreamerRecorder::push_buffer(const uint8_t*, size_t, uint32_t, uint32_t) { return false; }
std::string GstreamerRecorder::generate_filename() const { return ""; }
void GstreamerRecorder::check_storage() {}
void GstreamerRecorder::delete_oldest_recording() {}

}  // namespace lagari

#endif  // HAS_GSTREAMER
