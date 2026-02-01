#pragma once

#include "lagari/display/display.hpp"
#include "lagari/recording/overlay_renderer.hpp"

#include <atomic>
#include <mutex>
#include <memory>
#include <thread>

namespace lagari {

/**
 * @brief GStreamer-based display implementation using appsrc
 * 
 * Pushes frames into a GStreamer pipeline for display. The pipeline
 * is built based on the target configuration:
 * - window: Displays in a window using autovideosink
 * - rtsp: Streams via RTP/UDP (requires external RTSP server)
 * - udp: Raw UDP streaming
 * - custom: User-provided pipeline string
 */
class GstreamerDisplay : public IDisplay {
public:
    /**
     * @brief Construct display with configuration
     */
    explicit GstreamerDisplay(const DisplayConfig& config);
    
    ~GstreamerDisplay() override;

    // Prevent copying
    GstreamerDisplay(const GstreamerDisplay&) = delete;
    GstreamerDisplay& operator=(const GstreamerDisplay&) = delete;

    // ========================================================================
    // IModule interface
    // ========================================================================
    
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override;
    std::string name() const override { return "GstreamerDisplay"; }

    // ========================================================================
    // IDisplay interface
    // ========================================================================
    
    void push_frame(const Frame& frame,
                   const DetectionResult* detections = nullptr,
                   SystemState state = SystemState::IDLE,
                   Duration latency = Duration::zero()) override;

    bool is_enabled() const override;
    void set_enabled(bool enabled) override;
    DisplayStats get_stats() const override;

private:
    // Pipeline building
    std::string build_pipeline() const;
    std::string build_source_element() const;
    std::string build_encode_element() const;
    std::string build_sink_element() const;

    // Pipeline management
    bool create_pipeline();
    void destroy_pipeline();
    bool push_buffer(const uint8_t* data, size_t size, uint32_t width, uint32_t height);

    // Frame rate limiting
    bool should_process_frame();

    // GStreamer state (PIMPL to avoid GStreamer includes in header)
    struct GstState;
    std::unique_ptr<GstState> gst_;

    // Configuration
    DisplayConfig config_;
    
    // Overlay renderer
    OverlayRenderer overlay_renderer_;

    // State
    std::atomic<bool> enabled_{false};
    std::atomic<bool> running_{false};
    
    // Statistics
    mutable std::mutex stats_mutex_;
    DisplayStats stats_;
    TimePoint last_frame_time_;
    TimePoint start_time_;
    
    // Frame rate limiting
    Duration min_frame_interval_;
    TimePoint last_push_time_;
};

}  // namespace lagari
