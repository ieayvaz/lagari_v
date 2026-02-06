#pragma once

#include "lagari/recording/recorder.hpp"
#include "lagari/recording/overlay_renderer.hpp"

#include <opencv2/videoio.hpp>
#include <atomic>
#include <mutex>
#include <memory>
#include <filesystem>

namespace lagari {

/**
 * @brief OpenCV-based video recorder
 * 
 * Simple and reliable video recording using OpenCV's VideoWriter.
 * Handles encoding internally with proper codec negotiation.
 */
class OpenCVRecorder : public IRecorder {
public:
    /**
     * @brief Construct recorder with configuration
     */
    explicit OpenCVRecorder(const RecordingConfig& config);
    
    ~OpenCVRecorder() override;

    // Prevent copying
    OpenCVRecorder(const OpenCVRecorder&) = delete;
    OpenCVRecorder& operator=(const OpenCVRecorder&) = delete;

    // ========================================================================
    // IModule interface
    // ========================================================================
    
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override;
    std::string name() const override { return "OpenCVRecorder"; }

    // ========================================================================
    // IRecorder interface
    // ========================================================================
    
    bool start_recording(const std::string& filename = "") override;
    void stop_recording() override;
    bool is_recording() const override;
    
    void add_frame(const Frame& frame, 
                  const DetectionResult* detections,
                  SystemState state,
                  Duration latency) override;
    
    void set_overlay_enabled(bool enabled) override;
    std::string current_filename() const override;
    double recording_duration() const override;
    uint64_t bytes_written() const override;

private:
    // Configuration
    RecordingConfig config_;
    
    // State
    std::atomic<bool> running_{false};
    std::atomic<bool> recording_{false};
    
    // Recording
    cv::VideoWriter writer_;
    std::string current_filename_;
    TimePoint recording_start_;
    uint64_t frame_count_{0};
    mutable std::mutex writer_mutex_;
    
    // FPS tracking - calculate actual FPS from frame timestamps
    TimePoint first_frame_time_;
    TimePoint last_frame_time_;
    double actual_fps_{0.0};
    
    // Overlay
    OverlayRenderer overlay_renderer_;
    
    // Helpers
    std::string generate_filename() const;
    int get_fourcc() const;
    void check_storage();
    double calculate_actual_fps() const;
};

}  // namespace lagari
