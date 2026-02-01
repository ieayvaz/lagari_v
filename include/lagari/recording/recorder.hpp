#pragma once

#include "lagari/core/module.hpp"
#include "lagari/core/types.hpp"
#include "lagari/recording/overlay_renderer.hpp"
#include <memory>
#include <string>

namespace lagari {

class Config;

/**
 * @brief Recording configuration
 */
struct RecordingConfig {
    bool enabled = false;           // Master toggle
    std::string output_dir = "/var/lagari/recordings";
    
    // Encoding settings
    std::string codec = "h264";     // h264, h265
    uint32_t bitrate_kbps = 8000;
    uint32_t fps = 30;
    bool hw_encode = true;          // Use hardware encoding if available
    
    // Container format
    std::string container = "mp4";  // mp4, mkv, ts
    
    // Overlay configuration
    OverlayConfig overlay;
    
    // File management
    uint32_t segment_duration_s = 0;  // 0 = single file, >0 = split into segments
    uint64_t max_storage_bytes = 10ULL * 1024 * 1024 * 1024;  // 10 GB
    bool delete_oldest = true;
};

/**
 * @brief Recording interface
 */
class IRecorder : public IModule {
public:
    virtual ~IRecorder() = default;

    /**
     * @brief Start recording
     * 
     * @param filename Output filename (auto-generated if empty)
     * @return true if recording started
     */
    virtual bool start_recording(const std::string& filename = "") = 0;

    /**
     * @brief Stop recording
     */
    virtual void stop_recording() = 0;

    /**
     * @brief Check if currently recording
     */
    virtual bool is_recording() const = 0;

    /**
     * @brief Add frame to recording
     * 
     * @param frame Frame to record
     * @param detections Optional detections for overlay
     */
    virtual void add_frame(const Frame& frame, 
                          const DetectionResult* detections = nullptr) = 0;

    /**
     * @brief Set overlay enabled
     */
    virtual void set_overlay_enabled(bool enabled) = 0;

    /**
     * @brief Get current recording filename
     */
    virtual std::string current_filename() const = 0;

    /**
     * @brief Get recording duration in seconds
     */
    virtual double recording_duration() const = 0;

    /**
     * @brief Get bytes written
     */
    virtual uint64_t bytes_written() const = 0;
};

/**
 * @brief Create recorder instance
 */
std::unique_ptr<IRecorder> create_recorder(const Config& config);
std::unique_ptr<IRecorder> create_recorder(const RecordingConfig& config);

}  // namespace lagari
