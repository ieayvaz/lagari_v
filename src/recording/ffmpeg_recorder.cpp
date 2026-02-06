#include "lagari/recording/ffmpeg_recorder.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstring>

namespace fs = std::filesystem;

namespace lagari {

// ============================================================================
// Constructor / Destructor
// ============================================================================

FFmpegRecorder::FFmpegRecorder(const RecordingConfig& config)
    : config_(config)
    , overlay_renderer_(config.overlay) {
}

FFmpegRecorder::~FFmpegRecorder() {
    stop();
}

// ============================================================================
// IModule Interface
// ============================================================================

bool FFmpegRecorder::initialize(const Config& config) {
    // Read config overrides
    config_.enabled = config.get_bool("recording.enabled", config_.enabled);
    config_.output_dir = config.get_string("recording.output_dir", config_.output_dir);
    config_.codec = config.get_string("recording.codec", config_.codec);
    config_.bitrate_kbps = config.get_int("recording.bitrate_kbps", config_.bitrate_kbps);
    config_.fps = config.get_int("recording.fps", config_.fps);
    config_.hw_encode = config.get_bool("recording.hw_encode", config_.hw_encode);
    config_.container = config.get_string("recording.container", config_.container);
    
    // Overlay config
    config_.overlay.enabled = config.get_bool("recording.overlay.enabled", config_.overlay.enabled);
    config_.overlay.timestamp = config.get_bool("recording.overlay.timestamp", config_.overlay.timestamp);
    config_.overlay.bounding_boxes = config.get_bool("recording.overlay.bounding_boxes", config_.overlay.bounding_boxes);
    config_.overlay.state = config.get_bool("recording.overlay.state", config_.overlay.state);
    config_.overlay.latency = config.get_bool("recording.overlay.latency", config_.overlay.latency);
    config_.overlay.fps = config.get_bool("recording.overlay.fps", config_.overlay.fps);
    
    // File management
    config_.segment_duration_s = config.get_int("recording.segment_duration_s", config_.segment_duration_s);
    config_.max_storage_bytes = config.get_int("recording.max_storage_gb", 10) * 1024ULL * 1024ULL * 1024ULL;
    config_.delete_oldest = config.get_bool("recording.delete_oldest", config_.delete_oldest);

    // Create output directory
    try {
        // Expand ~ to home directory
        std::string output_dir = config_.output_dir;
        if (!output_dir.empty() && output_dir[0] == '~') {
            const char* home = std::getenv("HOME");
            if (home) {
                output_dir = std::string(home) + output_dir.substr(1);
            }
        }
        config_.output_dir = output_dir;
        
        fs::create_directories(config_.output_dir);
    } catch (const std::exception& e) {
        LOG_ERROR("FFmpegRecorder: Failed to create output directory: {}", e.what());
        return false;
    }

    // Reinitialize overlay renderer with updated config
    overlay_renderer_ = OverlayRenderer(config_.overlay);

    LOG_INFO("FFmpegRecorder: Initialized, output dir: {}", config_.output_dir);
    return true;
}

void FFmpegRecorder::start() {
    running_ = true;
    LOG_INFO("FFmpegRecorder: Started (ready to record)");
}

void FFmpegRecorder::stop() {
    stop_recording();
    running_ = false;
    LOG_INFO("FFmpegRecorder: Stopped");
}

bool FFmpegRecorder::is_running() const {
    return running_;
}

// ============================================================================
// IRecorder Interface
// ============================================================================

bool FFmpegRecorder::start_recording(const std::string& filename) {
    std::lock_guard<std::mutex> lock(pipe_mutex_);
    
    if (recording_) {
        LOG_WARN("FFmpegRecorder: Already recording");
        return false;
    }

    // Check storage
    check_storage();

    // Generate filename if not provided
    current_filename_ = filename.empty() ? generate_filename() : filename;
    
    LOG_INFO("FFmpegRecorder: Starting recording to {}", current_filename_);
    
    // We'll start FFmpeg on first frame when we know the size
    frame_count_ = 0;
    frame_width_ = 0;
    frame_height_ = 0;
    recording_start_ = Clock::now();
    recording_ = true;
    
    return true;
}

void FFmpegRecorder::stop_recording() {
    std::lock_guard<std::mutex> lock(pipe_mutex_);
    
    if (!recording_) {
        return;
    }
    
    recording_ = false;
    
    stop_ffmpeg_process();
    
    auto duration = std::chrono::duration<double>(Clock::now() - recording_start_).count();
    LOG_INFO("FFmpegRecorder: Stopped recording. Duration: {:.1f}s, Frames: {}", 
             duration, frame_count_);
    
    current_filename_.clear();
}

bool FFmpegRecorder::is_recording() const {
    return recording_;
}

void FFmpegRecorder::add_frame(const Frame& frame, 
                                const DetectionResult* detections,
                                SystemState state,
                                Duration latency) {
    if (!recording_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pipe_mutex_);
    
    if (!recording_) {
        return;
    }
    
    // Convert frame to BGR cv::Mat
    cv::Mat output_mat;
    
    uint32_t width = frame.metadata.width;
    uint32_t height = frame.metadata.height;
    const uint8_t* data_ptr = frame.data.get();
    
    if (frame.metadata.format == PixelFormat::BGR24) {
        output_mat = cv::Mat(height, width, CV_8UC3, 
                            const_cast<uint8_t*>(data_ptr));
    } else if (frame.metadata.format == PixelFormat::BGRA32) {
        cv::Mat bgra(height, width, CV_8UC4,
                     const_cast<uint8_t*>(data_ptr));
        cv::cvtColor(bgra, output_mat, cv::COLOR_BGRA2BGR);
    } else if (frame.metadata.format == PixelFormat::RGB24) {
        cv::Mat rgb(height, width, CV_8UC3,
                    const_cast<uint8_t*>(data_ptr));
        cv::cvtColor(rgb, output_mat, cv::COLOR_RGB2BGR);
    } else if (frame.metadata.format == PixelFormat::RGBA32) {
        cv::Mat rgba(height, width, CV_8UC4,
                     const_cast<uint8_t*>(data_ptr));
        cv::cvtColor(rgba, output_mat, cv::COLOR_RGBA2BGR);
    } else if (frame.metadata.format == PixelFormat::GRAY8) {
        cv::Mat gray(height, width, CV_8UC1,
                     const_cast<uint8_t*>(data_ptr));
        cv::cvtColor(gray, output_mat, cv::COLOR_GRAY2BGR);
    } else if (frame.metadata.format == PixelFormat::YUYV) {
        cv::Mat yuyv(height, width, CV_8UC2,
                     const_cast<uint8_t*>(data_ptr));
        cv::cvtColor(yuyv, output_mat, cv::COLOR_YUV2BGR_YUYV);
    } else if (frame.metadata.format == PixelFormat::NV12) {
        cv::Mat nv12(height * 3 / 2, width, CV_8UC1,
                     const_cast<uint8_t*>(data_ptr));
        cv::cvtColor(nv12, output_mat, cv::COLOR_YUV2BGR_NV12);
    } else {
        LOG_WARN("FFmpegRecorder: Unsupported pixel format");
        return;
    }
    
    // Apply overlay if enabled
    if (config_.overlay.enabled) {
        // Clone if we wrapped the original data
        if (output_mat.data == data_ptr) {
            output_mat = output_mat.clone();
        }
        overlay_renderer_.render_inplace(output_mat, detections, state, latency);
    }
    
    // Start FFmpeg on first frame (now we know the size)
    if (ffmpeg_pipe_ == nullptr) {
        if (!start_ffmpeg_process(output_mat.cols, output_mat.rows)) {
            LOG_ERROR("FFmpegRecorder: Failed to start FFmpeg process");
            recording_ = false;
            return;
        }
        frame_width_ = output_mat.cols;
        frame_height_ = output_mat.rows;
    }
    
    // Ensure the frame is contiguous (no padding between rows)
    if (!output_mat.isContinuous()) {
        output_mat = output_mat.clone();
    }
    
    // Write frame data to FFmpeg pipe
    size_t frame_size = output_mat.total() * output_mat.elemSize();
    size_t written = fwrite(output_mat.data, 1, frame_size, ffmpeg_pipe_);
    
    if (written != frame_size) {
        LOG_ERROR("FFmpegRecorder: Failed to write frame (wrote {} of {} bytes)", 
                  written, frame_size);
        // Don't stop recording, FFmpeg might recover
    }
    
    frame_count_++;
}

void FFmpegRecorder::set_overlay_enabled(bool enabled) {
    config_.overlay.enabled = enabled;
}

std::string FFmpegRecorder::current_filename() const {
    return current_filename_;
}

double FFmpegRecorder::recording_duration() const {
    if (!recording_) {
        return 0.0;
    }
    return std::chrono::duration<double>(Clock::now() - recording_start_).count();
}

uint64_t FFmpegRecorder::bytes_written() const {
    if (current_filename_.empty()) {
        return 0;
    }
    try {
        return fs::file_size(current_filename_);
    } catch (...) {
        return 0;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

std::string FFmpegRecorder::generate_filename() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << config_.output_dir << "/recording_"
        << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    
    // Extension based on container
    if (config_.container == "mkv") {
        oss << ".mkv";
    } else if (config_.container == "ts") {
        oss << ".ts";
    } else {
        oss << ".mp4";
    }
    
    return oss.str();
}

std::string FFmpegRecorder::build_ffmpeg_command(uint32_t width, uint32_t height) const {
    std::ostringstream cmd;
    
    // FFmpeg command to read raw BGR frames from stdin
    cmd << "ffmpeg -y ";  // Overwrite output
    
    // Input format: raw video from pipe
    cmd << "-f rawvideo ";
    cmd << "-pix_fmt bgr24 ";
    cmd << "-s " << width << "x" << height << " ";
    
    // Use wallclock timestamps for accurate VFR timing
    // This timestamps each frame as it arrives, giving us true VFR
    cmd << "-use_wallclock_as_timestamps 1 ";
    
    // Read from stdin
    cmd << "-i pipe:0 ";
    
    // Video codec selection
    if (config_.hw_encode) {
        // Try NVIDIA NVENC first
        if (config_.codec == "h265" || config_.codec == "hevc") {
            cmd << "-c:v hevc_nvenc ";
            cmd << "-preset p4 ";  // Fast preset
            cmd << "-b:v " << config_.bitrate_kbps << "k ";
        } else {
            cmd << "-c:v h264_nvenc ";
            cmd << "-preset p4 ";
            cmd << "-b:v " << config_.bitrate_kbps << "k ";
        }
    } else {
        // Software encoding with libx264
        if (config_.codec == "h265" || config_.codec == "hevc") {
            cmd << "-c:v libx265 ";
            cmd << "-preset ultrafast ";
            cmd << "-crf 23 ";
        } else {
            cmd << "-c:v libx264 ";
            cmd << "-preset ultrafast ";
            cmd << "-tune zerolatency ";
            cmd << "-crf 23 ";
        }
    }
    
    // Output format settings
    if (config_.container == "mp4") {
        // For MP4, we need to use a specific muxer for streaming
        cmd << "-movflags +faststart ";
    }
    
    // Suppress FFmpeg banner but keep errors
    cmd << "-loglevel warning ";
    
    // Output file
    cmd << "\"" << current_filename_ << "\"";
    
    return cmd.str();
}

bool FFmpegRecorder::start_ffmpeg_process(uint32_t width, uint32_t height) {
    std::string cmd = build_ffmpeg_command(width, height);
    
    LOG_DEBUG("FFmpegRecorder: Starting FFmpeg with command: {}", cmd);
    
    // Open pipe to FFmpeg
    ffmpeg_pipe_ = popen(cmd.c_str(), "w");
    
    if (ffmpeg_pipe_ == nullptr) {
        LOG_ERROR("FFmpegRecorder: Failed to start FFmpeg process: {}", strerror(errno));
        return false;
    }
    
    LOG_INFO("FFmpegRecorder: FFmpeg started for {}x{} @ VFR, encoder: {}", 
             width, height, config_.hw_encode ? "hardware" : "software");
    
    return true;
}

void FFmpegRecorder::stop_ffmpeg_process() {
    if (ffmpeg_pipe_ != nullptr) {
        // Close the pipe, which sends EOF to FFmpeg
        // FFmpeg will then finalize the file
        int status = pclose(ffmpeg_pipe_);
        ffmpeg_pipe_ = nullptr;
        
        if (status != 0) {
            LOG_WARN("FFmpegRecorder: FFmpeg exited with status {}", status);
        } else {
            LOG_DEBUG("FFmpegRecorder: FFmpeg finished successfully");
        }
    }
}

void FFmpegRecorder::check_storage() {
    if (!config_.delete_oldest) {
        return;
    }

    try {
        uint64_t total_size = 0;
        std::vector<fs::directory_entry> recordings;

        for (const auto& entry : fs::directory_iterator(config_.output_dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".mp4" || ext == ".mkv" || ext == ".ts" || ext == ".avi") {
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
            
            LOG_INFO("FFmpegRecorder: Deleting old recording: {}", oldest.path().string());
            fs::remove(oldest.path());
            
            total_size -= file_size;
            recordings.erase(recordings.begin());
        }
    } catch (const std::exception& e) {
        LOG_WARN("FFmpegRecorder: Failed to check storage: {}", e.what());
    }
}

}  // namespace lagari
