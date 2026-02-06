#pragma once

#include "lagari/recording/recorder.hpp"
#include "lagari/recording/overlay_renderer.hpp"

#include <atomic>
#include <mutex>
#include <memory>

namespace lagari {

/**
 * @brief GStreamer-based video recorder using appsrc
 * 
 * Records video to file using a GStreamer pipeline. Frames are pushed
 * via appsrc, encoded, and written to a container file.
 * 
 * Pipeline structure:
 *   appsrc -> videoconvert -> encoder -> parser -> muxer -> filesink
 */
class GstreamerRecorder : public IRecorder {
public:
    /**
     * @brief Construct recorder with configuration
     */
    explicit GstreamerRecorder(const RecordingConfig& config);
    
    ~GstreamerRecorder() override;

    // Prevent copying
    GstreamerRecorder(const GstreamerRecorder&) = delete;
    GstreamerRecorder& operator=(const GstreamerRecorder&) = delete;

    // ========================================================================
    // IModule interface
    // ========================================================================
    
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override;
    std::string name() const override { return "GstreamerRecorder"; }

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
    // Pipeline building
    std::string build_pipeline(const std::string& filename) const;
    std::string build_source_element() const;
    std::string build_encode_element() const;
    std::string build_mux_element() const;
    std::string build_sink_element(const std::string& filename) const;

    // Pipeline management
    bool create_pipeline(const std::string& filename);
    void destroy_pipeline();
    bool push_buffer(const uint8_t* data, size_t size, uint32_t width, uint32_t height,
                     TimePoint frame_timestamp);

    // File management
    std::string generate_filename() const;
    void check_storage();
    void delete_oldest_recording();

    // GStreamer state (PIMPL)
    struct GstState;
    std::unique_ptr<GstState> gst_;

    // Configuration
    RecordingConfig config_;
    
    // Overlay renderer
    OverlayRenderer overlay_renderer_;

    // State
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> recording_{false};
    
    // Recording info
    mutable std::mutex info_mutex_;
    std::string current_filename_;
    TimePoint recording_start_;
    uint64_t bytes_written_{0};
    uint64_t frames_recorded_{0};
    
    // System state for overlay (updated via add_frame context)
    SystemState current_state_{SystemState::IDLE};
    Duration current_latency_{Duration::zero()};
};

}  // namespace lagari
