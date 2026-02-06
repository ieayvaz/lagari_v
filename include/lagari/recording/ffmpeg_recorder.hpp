#pragma once

#include "lagari/recording/recorder.hpp"
#include "lagari/recording/overlay_renderer.hpp"

#include <opencv2/core.hpp>
#include <atomic>
#include <mutex>
#include <memory>
#include <filesystem>
#include <cstdio>

namespace lagari {

/**
 * @brief FFmpeg-based video recorder with accurate timestamps
 * 
 * Pipes raw frames to an FFmpeg subprocess for encoding.
 * Supports variable frame rate (VFR) with per-frame timestamps
 * for accurate playback timing regardless of capture rate variations.
 * 
 * FFmpeg uses wall-clock timestamps for each frame, ensuring that
 * the recorded video plays back at the correct speed even if
 * frames arrive at irregular intervals.
 */
class FFmpegRecorder : public IRecorder {
public:
    /**
     * @brief Construct recorder with configuration
     */
    explicit FFmpegRecorder(const RecordingConfig& config);
    
    ~FFmpegRecorder() override;

    // Prevent copying
    FFmpegRecorder(const FFmpegRecorder&) = delete;
    FFmpegRecorder& operator=(const FFmpegRecorder&) = delete;

    // ========================================================================
    // IModule interface
    // ========================================================================
    
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override;
    std::string name() const override { return "FFmpegRecorder"; }

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
    
    // FFmpeg process
    FILE* ffmpeg_pipe_{nullptr};
    std::string current_filename_;
    TimePoint recording_start_;
    uint64_t frame_count_{0};
    uint32_t frame_width_{0};
    uint32_t frame_height_{0};
    mutable std::mutex pipe_mutex_;
    
    // Overlay
    OverlayRenderer overlay_renderer_;
    
    // Helpers
    std::string generate_filename() const;
    std::string build_ffmpeg_command(uint32_t width, uint32_t height) const;
    void check_storage();
    bool start_ffmpeg_process(uint32_t width, uint32_t height);
    void stop_ffmpeg_process();
};

}  // namespace lagari
