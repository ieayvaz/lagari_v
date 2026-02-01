#pragma once

#include "lagari/capture/capture.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>

// Forward declarations for libcamera types
namespace libcamera {
    class CameraManager;
    class Camera;
    class CameraConfiguration;
    class FrameBufferAllocator;
    class FrameBuffer;
    class Request;
    class Stream;
    class ControlList;
}

namespace lagari {

/**
 * @brief libcamera capture backend for Raspberry Pi
 * 
 * Uses the libcamera library for camera capture on Raspberry Pi
 * with support for official cameras (IMX219, IMX477, IMX708, etc.)
 * and USB cameras.
 * 
 * Features:
 * - Support for Pi Camera v2 (IMX219) and HQ Camera (IMX477)
 * - Support for Camera Module 3 (IMX708)
 * - Auto/manual exposure control
 * - Auto/manual white balance
 * - Hardware-accelerated format conversion
 * 
 * Requires:
 * - Raspberry Pi OS with libcamera
 * - libcamera-dev package
 */
class LibcameraCapture : public ICapture {
public:
    /**
     * @brief libcamera-specific configuration
     */
    struct LibcameraConfig {
        // Exposure
        bool ae_enable = true;          // Auto exposure
        int64_t exposure_time_us = 0;   // Manual exposure (0 = auto)
        float analogue_gain = 1.0f;     // Manual gain
        
        // White balance
        bool awb_enable = true;
        float colour_gains[2] = {0, 0}; // Red/Blue gains (0 = auto)
        
        // Image tuning
        float brightness = 0.0f;        // -1.0 to 1.0
        float contrast = 1.0f;          // 0.0 to 2.0
        float saturation = 1.0f;        // 0.0 to 2.0
        float sharpness = 1.0f;         // 0.0 to 2.0
        
        // Denoise
        int denoise_mode = 2;           // 0=off, 1=fast, 2=high quality
        
        // Transform
        bool hflip = false;
        bool vflip = false;
        int rotation = 0;               // 0, 180
    };

    explicit LibcameraCapture(const CaptureConfig& config);
    LibcameraCapture(const CaptureConfig& config, const LibcameraConfig& lc_config);
    ~LibcameraCapture() override;

    // Non-copyable
    LibcameraCapture(const LibcameraCapture&) = delete;
    LibcameraCapture& operator=(const LibcameraCapture&) = delete;

    // IModule interface
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override;
    std::string name() const override { return "LibcameraCapture"; }

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

    // Libcamera-specific methods
    
    /**
     * @brief Get available cameras
     * @return Number of available cameras
     */
    int get_camera_count() const;

    /**
     * @brief Get camera model name
     */
    std::string get_camera_model() const;

    /**
     * @brief Set analogue gain
     */
    bool set_gain(float gain);

    /**
     * @brief Set white balance mode
     */
    bool set_awb_mode(bool enable);

    /**
     * @brief Set colour gains (manual white balance)
     */
    bool set_colour_gains(float red_gain, float blue_gain);

    /**
     * @brief Set image brightness
     */
    bool set_brightness(float brightness);

    /**
     * @brief Set image contrast
     */
    bool set_contrast(float contrast);

    /**
     * @brief Set image saturation
     */
    bool set_saturation(float saturation);

    /**
     * @brief Set image sharpness
     */
    bool set_sharpness(float sharpness);

    /**
     * @brief Get libcamera configuration
     */
    const LibcameraConfig& libcamera_config() const { return lc_config_; }

private:
    // libcamera initialization
    bool open_camera(int camera_id);
    bool configure_camera();
    bool create_buffers();
    bool queue_request(libcamera::Request* request);

    // Request completed callback
    void request_complete(libcamera::Request* request);

    // Frame processing
    FramePtr process_buffer(libcamera::FrameBuffer* buffer);

    // Apply control settings
    void apply_controls();

    // Configuration
    CaptureConfig config_;
    LibcameraConfig lc_config_;

    // libcamera objects (using PIMPL to avoid header dependency)
    struct LibcameraState;
    std::unique_ptr<LibcameraState> lc_;

    // Threading
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
