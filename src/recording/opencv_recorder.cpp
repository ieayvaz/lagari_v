#include "lagari/recording/opencv_recorder.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

namespace lagari {

// ============================================================================
// Constructor / Destructor
// ============================================================================

OpenCVRecorder::OpenCVRecorder(const RecordingConfig& config)
    : config_(config)
    , overlay_renderer_(config.overlay) {
}

OpenCVRecorder::~OpenCVRecorder() {
    stop();
}

// ============================================================================
// IModule Interface
// ============================================================================

bool OpenCVRecorder::initialize(const Config& config) {
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
        LOG_ERROR("OpenCVRecorder: Failed to create output directory: {}", e.what());
        return false;
    }

    // Reinitialize overlay renderer with updated config
    overlay_renderer_ = OverlayRenderer(config_.overlay);

    LOG_INFO("OpenCVRecorder: Initialized, output dir: {}", config_.output_dir);
    return true;
}

void OpenCVRecorder::start() {
    running_ = true;
    LOG_INFO("OpenCVRecorder: Started (ready to record)");
}

void OpenCVRecorder::stop() {
    stop_recording();
    running_ = false;
    LOG_INFO("OpenCVRecorder: Stopped");
}

bool OpenCVRecorder::is_running() const {
    return running_;
}

// ============================================================================
// IRecorder Interface
// ============================================================================

bool OpenCVRecorder::start_recording(const std::string& filename) {
    std::lock_guard<std::mutex> lock(writer_mutex_);
    
    if (recording_) {
        LOG_WARN("OpenCVRecorder: Already recording");
        return false;
    }

    // Check storage
    check_storage();

    // Generate filename if not provided
    current_filename_ = filename.empty() ? generate_filename() : filename;
    
    LOG_INFO("OpenCVRecorder: Starting recording to {}", current_filename_);
    
    // We'll open the writer on first frame when we know the size
    frame_count_ = 0;
    recording_start_ = Clock::now();
    recording_ = true;
    
    return true;
}

void OpenCVRecorder::stop_recording() {
    std::lock_guard<std::mutex> lock(writer_mutex_);
    
    if (!recording_) {
        return;
    }
    
    recording_ = false;
    
    if (writer_.isOpened()) {
        writer_.release();
        
        auto duration = std::chrono::duration<double>(Clock::now() - recording_start_).count();
        LOG_INFO("OpenCVRecorder: Stopped recording. Duration: {:.1f}s, Frames: {}", 
                 duration, frame_count_);
    }
    
    current_filename_.clear();
}

bool OpenCVRecorder::is_recording() const {
    return recording_;
}

void OpenCVRecorder::add_frame(const Frame& frame, 
                                const DetectionResult* detections,
                                SystemState state,
                                Duration latency) {
    if (!recording_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(writer_mutex_);
    
    if (!recording_) {
        return;
    }
    
    // Track frame timing for FPS calculation
    TimePoint frame_time = frame.metadata.timestamp;
    if (frame_count_ == 0) {
        first_frame_time_ = frame_time;
    }
    last_frame_time_ = frame_time;
    
    // Convert frame to cv::Mat
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
        LOG_WARN("OpenCVRecorder: Unsupported pixel format");
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
    
    // Open writer after a few frames to calculate actual FPS
    if (!writer_.isOpened()) {
        // Wait for at least 5 frames to get a stable FPS measurement
        if (frame_count_ < 5) {
            frame_count_++;
            // Buffer frames would require more code, so we skip these initial frames
            return;
        }
        
        // Calculate actual FPS from frame timestamps
        actual_fps_ = calculate_actual_fps();
        if (actual_fps_ <= 0.0 || actual_fps_ > 120.0) {
            // Fallback to config FPS if calculation fails
            actual_fps_ = config_.fps;
        }
        
        int fourcc = get_fourcc();
        cv::Size size(output_mat.cols, output_mat.rows);
        
        if (!writer_.open(current_filename_, fourcc, actual_fps_, size, true)) {
            LOG_ERROR("OpenCVRecorder: Failed to open video writer for {}", current_filename_);
            recording_ = false;
            return;
        }
        
        LOG_INFO("OpenCVRecorder: Writer opened: {}x{} @ {:.1f} fps (actual), codec: {}", 
                 size.width, size.height, actual_fps_, config_.codec);
    }
    
    // Write frame
    writer_.write(output_mat);
    frame_count_++;
}

void OpenCVRecorder::set_overlay_enabled(bool enabled) {
    config_.overlay.enabled = enabled;
}

std::string OpenCVRecorder::current_filename() const {
    return current_filename_;
}

double OpenCVRecorder::recording_duration() const {
    if (!recording_) {
        return 0.0;
    }
    return std::chrono::duration<double>(Clock::now() - recording_start_).count();
}

uint64_t OpenCVRecorder::bytes_written() const {
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

std::string OpenCVRecorder::generate_filename() const {
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

int OpenCVRecorder::get_fourcc() const {
    // Try to use hardware encoding if available
    if (config_.hw_encode) {
        // NVIDIA NVENC via FFmpeg backend
        if (config_.codec == "h265" || config_.codec == "hevc") {
            return cv::VideoWriter::fourcc('h', 'v', 'c', '1');
        } else {
            // H.264 - use avc1 which typically triggers hardware encoding
            return cv::VideoWriter::fourcc('a', 'v', 'c', '1');
        }
    } else {
        // Software encoding
        if (config_.codec == "h265" || config_.codec == "hevc") {
            return cv::VideoWriter::fourcc('h', 'e', 'v', '1');
        } else {
            // X264 software encoder
            return cv::VideoWriter::fourcc('X', '2', '6', '4');
        }
    }
}

void OpenCVRecorder::check_storage() {
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
            
            LOG_INFO("OpenCVRecorder: Deleting old recording: {}", oldest.path().string());
            fs::remove(oldest.path());
            
            total_size -= file_size;
            recordings.erase(recordings.begin());
        }
    } catch (const std::exception& e) {
        LOG_WARN("OpenCVRecorder: Failed to check storage: {}", e.what());
    }
}

double OpenCVRecorder::calculate_actual_fps() const {
    if (frame_count_ < 2) {
        return config_.fps;  // Not enough frames, use config
    }
    
    auto duration = std::chrono::duration<double>(last_frame_time_ - first_frame_time_);
    double elapsed_seconds = duration.count();
    
    if (elapsed_seconds <= 0.0) {
        return config_.fps;  // Invalid timing, use config
    }
    
    // FPS = (frame_count - 1) / elapsed_time
    // -1 because first frame is at time 0
    double fps = static_cast<double>(frame_count_ - 1) / elapsed_seconds;
    
    LOG_DEBUG("OpenCVRecorder: Calculated actual FPS: {:.1f} from {} frames over {:.2f}s",
              fps, frame_count_, elapsed_seconds);
    
    return fps;
}

}  // namespace lagari

