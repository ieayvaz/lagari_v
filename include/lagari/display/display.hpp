#pragma once

#include "lagari/core/module.hpp"
#include "lagari/core/types.hpp"
#include "lagari/recording/overlay_renderer.hpp"
#include <memory>
#include <string>

namespace lagari {

class Config;

/**
 * @brief Display configuration
 */
struct DisplayConfig {
    bool enabled = false;               // Master toggle
    std::string target = "window";      // window, rtsp, udp, custom
    
    // Window settings
    std::string window_name = "Lagari Vision";
    bool fullscreen = false;
    
    // Streaming settings (for rtsp/udp targets)
    std::string host = "0.0.0.0";
    uint16_t port = 5600;
    uint32_t bitrate_kbps = 2000;
    
    // Custom pipeline (overrides target if not empty)
    std::string pipeline = "";
    
    // Overlay settings
    OverlayConfig overlay;
    
    // Performance settings
    uint32_t max_fps = 30;              // Limit display framerate
    bool drop_frames = true;            // Drop frames if pipeline is behind
    
    // Encoding (for streaming targets)
    std::string codec = "h264";         // h264, mjpeg
    bool hw_encode = true;              // Use hardware encoding if available
};

/**
 * @brief Display interface for showing frames with overlay
 * 
 * The display module takes frames pushed to it and renders them
 * via a GStreamer pipeline. Frames are processed with optional
 * overlay before being pushed to the display sink.
 */
class IDisplay : public IModule {
public:
    virtual ~IDisplay() = default;

    /**
     * @brief Push a frame to display
     * 
     * This method should be called from the main processing loop.
     * If display is disabled, this is a no-op.
     * 
     * @param frame Frame to display
     * @param detections Optional detection results for overlay
     * @param state Current system state for overlay
     * @param latency Processing latency for overlay
     */
    virtual void push_frame(const Frame& frame,
                           const DetectionResult* detections = nullptr,
                           SystemState state = SystemState::IDLE,
                           Duration latency = Duration::zero()) = 0;

    /**
     * @brief Check if display is enabled
     */
    virtual bool is_enabled() const = 0;

    /**
     * @brief Enable or disable display at runtime
     */
    virtual void set_enabled(bool enabled) = 0;

    /**
     * @brief Get current display statistics
     */
    struct DisplayStats {
        uint64_t frames_displayed = 0;
        uint64_t frames_dropped = 0;
        float current_fps = 0.0f;
    };
    virtual DisplayStats get_stats() const = 0;
};

/**
 * @brief Create display instance from configuration
 * 
 * Returns nullptr if display is disabled or GStreamer is not available.
 */
std::unique_ptr<IDisplay> create_display(const Config& config);
std::unique_ptr<IDisplay> create_display(const DisplayConfig& config);

}  // namespace lagari
