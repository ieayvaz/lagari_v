#pragma once

#include "lagari/capture/capture.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <string>

// Forward declarations for GStreamer types
typedef struct _GstElement GstElement;
typedef struct _GstPipeline GstPipeline;
typedef struct _GstBus GstBus;
typedef struct _GstSample GstSample;
typedef struct _GstAppSink GstAppSink;
typedef struct _GMainLoop GMainLoop;

namespace lagari {

/**
 * @brief GStreamer capture backend
 * 
 * Versatile capture backend using GStreamer for:
 * - RTSP streams (IP cameras)
 * - Video files (MP4, AVI, MKV, etc.)
 * - HTTP/HLS streams
 * - Test patterns
 * - USB cameras (via v4l2src)
 * - CSI cameras (via nvarguscamerasrc on Jetson)
 * 
 * Features:
 * - Hardware-accelerated decoding (NVDEC, VAAPI, V4L2M2M)
 * - Automatic format conversion
 * - Network resilience for RTSP
 * - Seeking support for files
 * 
 * Requires:
 * - GStreamer 1.x with gst-plugins-base, gst-plugins-good
 * - Optional: gst-plugins-bad (for hardware acceleration)
 */
class GstreamerCapture : public ICapture {
public:
    /**
     * @brief GStreamer-specific configuration
     */
    struct GstConfig {
        // Pipeline configuration
        std::string pipeline;          // Custom pipeline (overrides other settings)
        
        // Source type
        enum class SourceType {
            AUTO,       // Auto-detect from URI
            V4L2,       // v4l2src
            RTSP,       // rtspsrc
            FILE,       // filesrc + decodebin
            HTTP,       // souphttpsrc
            TEST,       // videotestsrc
            URI,        // uridecodebin (generic)
            ARGUS,      // nvarguscamerasrc (Jetson)
            LIBCAMERA   // libcamerasrc (RPi)
        };
        SourceType source_type = SourceType::AUTO;
        
        // Source settings
        std::string uri;               // URI for RTSP/HTTP/file
        std::string device;            // Device path for V4L2
        
        // RTSP settings
        int latency_ms = 200;          // RTSP latency/buffer
        int tcp_timeout = 5000000;     // Connection timeout (µs)
        bool use_tcp = true;           // Use TCP for RTP (more reliable)
        bool drop_on_latency = true;   // Drop frames if behind
        
        // Decoding settings
        bool hw_decode = true;         // Use hardware decoding if available
        std::string decoder;           // Specific decoder element (empty = auto)
        
        // Video settings
        bool sync = false;             // Sync to clock (false for real-time)
        int queue_size = 2;            // AppSink queue size
        
        // File playback
        bool loop = true;              // Loop file playback
        int64_t seek_position_ns = 0;  // Start position for seeking
    };

    explicit GstreamerCapture(const CaptureConfig& config);
    GstreamerCapture(const CaptureConfig& config, const GstConfig& gst_config);
    ~GstreamerCapture() override;

    // Non-copyable
    GstreamerCapture(const GstreamerCapture&) = delete;
    GstreamerCapture& operator=(const GstreamerCapture&) = delete;

    // IModule interface
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override;
    std::string name() const override { return "GstreamerCapture"; }

    // ICapture interface
    FramePtr get_latest_frame() override;
    FramePtr wait_for_frame(uint32_t timeout_ms) override;
    void set_frame_callback(FrameCallback callback) override;
    CaptureStats get_stats() const override;
    bool is_open() const override;
    const CaptureConfig& config() const override { return config_; }
    bool set_resolution(uint32_t width, uint32_t height) override;
    bool set_framerate(uint32_t fps) override;
    bool set_exposure(bool auto_exp, float exposure_time) override;

    // GStreamer-specific methods
    
    /**
     * @brief Set custom pipeline string
     * @param pipeline GStreamer pipeline description
     * @return true if pipeline is valid
     */
    bool set_pipeline(const std::string& pipeline);

    /**
     * @brief Get current pipeline string
     */
    std::string get_pipeline() const;

    /**
     * @brief Seek to position (for file sources)
     * @param position_ns Position in nanoseconds
     * @return true if seek successful
     */
    bool seek(int64_t position_ns);

    /**
     * @brief Get current position (for file sources)
     * @return Position in nanoseconds, -1 if not available
     */
    int64_t get_position() const;

    /**
     * @brief Get duration (for file sources)
     * @return Duration in nanoseconds, -1 if not available
     */
    int64_t get_duration() const;

    /**
     * @brief Check if end of stream reached
     */
    bool is_eos() const;

    /**
     * @brief Get GStreamer configuration
     */
    const GstConfig& gst_config() const { return gst_config_; }

private:
    // Pipeline building
    std::string build_pipeline() const;
    std::string build_source_element() const;
    std::string build_decode_element() const;
    std::string build_convert_element() const;
    std::string build_sink_element() const;

    // GStreamer callbacks (defined in cpp with proper GST types)
    static int on_new_sample_cb(void* sink, void* user_data);
    static void on_pad_added_cb(void* element, void* pad, void* user_data);
    static int on_bus_message_cb(void* bus, void* message, void* user_data);

    // Pipeline management
    bool create_pipeline();
    bool link_pipeline();
    void destroy_pipeline();

    // Frame processing
    void process_loop();
    FramePtr convert_sample(void* sample);

    // Configuration
    CaptureConfig config_;
    GstConfig gst_config_;

    // GStreamer state (using void* to avoid header dependency)
    struct GstState;
    std::unique_ptr<GstState> gst_;

    // Threading
    std::thread process_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> eos_{false};

    // Frame storage
    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    FramePtr latest_frame_;
    uint64_t frame_counter_ = 0;

    // Callback
    FrameCallback frame_callback_;

    // Statistics
    mutable std::mutex stats_mutex_;
    CaptureStats stats_;
    TimePoint start_time_;
    TimePoint last_frame_time_;
};

}  // namespace lagari
