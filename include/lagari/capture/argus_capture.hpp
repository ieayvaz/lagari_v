#pragma once

#include "lagari/capture/capture.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>

// Forward declarations for Argus types
namespace Argus {
    class CameraProvider;
    class CameraDevice;
    class CaptureSession;
    class OutputStreamSettings;
    class OutputStream;
    class Request;
    class ICaptureSession;
    class IEventProvider;
}

namespace EGLStream {
    class FrameConsumer;
    class Frame;
}

namespace lagari {

/**
 * @brief NVIDIA Argus capture backend for Jetson platforms
 * 
 * Uses the NVIDIA Argus camera API for high-performance CSI camera
 * capture on Jetson devices (Nano, TX2, Xavier, Orin).
 * 
 * Features:
 * - Direct CSI camera access
 * - Hardware ISP processing
 * - Zero-copy with EGLStream
 * - Low latency capture
 * 
 * Requires:
 * - NVIDIA Jetson platform
 * - L4T (Linux for Tegra) with Argus libraries
 * - Compatible CSI camera (IMX219, IMX477, etc.)
 */
class ArgusCapture : public ICapture {
public:
    /**
     * @brief Argus-specific configuration
     */
    struct ArgusConfig {
        int sensor_mode = 0;           // Sensor mode index
        float frame_duration_ns = 0;   // 0 = auto (based on fps)
        float gain_range_min = 1.0f;
        float gain_range_max = 16.0f;
        float exposure_time_min_ns = 13000;     // 13µs
        float exposure_time_max_ns = 683709000; // 683ms
        bool denoise_enable = true;
        float denoise_strength = 0.5f;
        bool edge_enhance_enable = true;
        float edge_enhance_strength = 0.5f;
        bool awb_enable = true;       // Auto white balance
    };

    explicit ArgusCapture(const CaptureConfig& config);
    ArgusCapture(const CaptureConfig& config, const ArgusConfig& argus_config);
    ~ArgusCapture() override;

    // Non-copyable
    ArgusCapture(const ArgusCapture&) = delete;
    ArgusCapture& operator=(const ArgusCapture&) = delete;

    // IModule interface
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override;
    std::string name() const override { return "ArgusCapture"; }

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

    // Argus-specific methods
    
    /**
     * @brief Get available camera devices
     * @return Number of available cameras
     */
    int get_camera_count() const;

    /**
     * @brief Get sensor modes for current camera
     * @return Vector of (width, height, fps) tuples
     */
    std::vector<std::tuple<uint32_t, uint32_t, float>> get_sensor_modes() const;

    /**
     * @brief Set sensor mode by index
     */
    bool set_sensor_mode(int mode_index);

    /**
     * @brief Set ISP digital gain
     */
    bool set_gain(float gain);

    /**
     * @brief Set AWB mode
     */
    bool set_awb_mode(bool enable);

    /**
     * @brief Set denoise settings
     */
    bool set_denoise(bool enable, float strength);

    /**
     * @brief Get Argus configuration
     */
    const ArgusConfig& argus_config() const { return argus_config_; }

private:
    // Argus initialization
    bool create_camera_provider();
    bool open_camera(int camera_id);
    bool create_capture_session();
    bool create_output_stream();
    bool create_request();
    bool configure_request();

    // Capture thread
    void capture_loop();
    FramePtr process_frame(EGLStream::Frame* frame);

    // Configuration
    CaptureConfig config_;
    ArgusConfig argus_config_;

    // Argus objects (opaque pointers to avoid header dependency)
    struct ArgusState;
    std::unique_ptr<ArgusState> argus_;

    // Threading
    std::thread capture_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};

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
