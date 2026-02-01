#pragma once

#include "lagari/core/module.hpp"
#include "lagari/core/types.hpp"
#include <memory>
#include <functional>

namespace lagari {

class Config;

/**
 * @brief Capture source type
 */
enum class CaptureSource : uint8_t {
    AUTO = 0,       // Auto-detect based on platform
    CSI,            // CSI camera (Jetson Argus / RPi libcamera)
    USB,            // USB camera (V4L2)
    FILE,           // Video file or image sequence
    RTSP,           // RTSP stream
    GSTREAMER,      // GStreamer pipeline (generic)
    SIMULATION,     // Simulated frames for testing
    ISAAC_SIM       // Isaac Sim via shared memory
};

inline const char* to_string(CaptureSource source) {
    switch (source) {
        case CaptureSource::AUTO: return "AUTO";
        case CaptureSource::CSI: return "CSI";
        case CaptureSource::USB: return "USB";
        case CaptureSource::FILE: return "FILE";
        case CaptureSource::RTSP: return "RTSP";
        case CaptureSource::GSTREAMER: return "GSTREAMER";
        case CaptureSource::SIMULATION: return "SIMULATION";
        case CaptureSource::ISAAC_SIM: return "ISAAC_SIM";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Capture configuration
 */
struct CaptureConfig {
    CaptureSource source = CaptureSource::AUTO;
    
    uint32_t width = 1280;
    uint32_t height = 720;
    uint32_t fps = 30;
    PixelFormat format = PixelFormat::BGR24;
    
    // Device selection
    std::string device;          // Device path or ID (e.g., "/dev/video0", "0")
    int camera_id = 0;           // Camera index for multi-camera systems
    
    // File/RTSP source
    std::string file_path;       // Path to video file or RTSP URL
    bool loop_file = true;       // Loop video file playback
    
    // Buffer settings
    size_t buffer_count = 4;     // Number of buffers in ring buffer
    bool drop_frames = true;     // Drop old frames if consumers are slow
    
    // Auto-exposure / gain (if supported)
    bool auto_exposure = true;
    float exposure_time = 0.0f;  // Manual exposure (if auto_exposure false)
    float gain = 1.0f;           // Manual gain
    
    // Flip/rotate
    bool flip_horizontal = false;
    bool flip_vertical = false;
    int rotation = 0;            // 0, 90, 180, 270
};

/**
 * @brief Frame callback type
 */
using FrameCallback = std::function<void(FramePtr frame)>;

/**
 * @brief Capture interface
 * 
 * Abstract interface for camera/video capture. Platform-specific backends
 * implement this interface for different capture sources.
 */
class ICapture : public IModule {
public:
    virtual ~ICapture() = default;

    /**
     * @brief Get the latest captured frame
     * 
     * Non-blocking. Returns nullptr if no frame available.
     * The returned frame is a shared pointer and can be safely
     * held by multiple consumers.
     * 
     * @return Latest frame or nullptr
     */
    virtual FramePtr get_latest_frame() = 0;

    /**
     * @brief Wait for and get next frame
     * 
     * Blocking until a new frame is available or timeout.
     * 
     * @param timeout_ms Maximum wait time in milliseconds
     * @return Next frame or nullptr on timeout
     */
    virtual FramePtr wait_for_frame(uint32_t timeout_ms = 100) = 0;

    /**
     * @brief Register callback for new frames
     * 
     * Callback is invoked from capture thread. Should be fast
     * to avoid blocking capture.
     * 
     * @param callback Function to call on new frame
     */
    virtual void set_frame_callback(FrameCallback callback) = 0;

    /**
     * @brief Get current capture statistics
     */
    virtual CaptureStats get_stats() const = 0;

    /**
     * @brief Check if capture source is open and working
     */
    virtual bool is_open() const = 0;

    /**
     * @brief Get current configuration
     */
    virtual const CaptureConfig& config() const = 0;

    /**
     * @brief Set resolution (if supported by source)
     * 
     * @return true if resolution was changed
     */
    virtual bool set_resolution(uint32_t width, uint32_t height) = 0;

    /**
     * @brief Set framerate (if supported by source)
     * 
     * @return true if framerate was changed
     */
    virtual bool set_framerate(uint32_t fps) = 0;

    /**
     * @brief Set exposure (if supported by source)
     * 
     * @param auto_exp Enable auto-exposure
     * @param exposure_time Manual exposure time (if auto_exp false)
     * @return true if setting was changed
     */
    virtual bool set_exposure(bool auto_exp, float exposure_time = 0.0f) = 0;
};

/**
 * @brief Create capture instance based on configuration
 * 
 * Factory function that creates the appropriate capture backend
 * based on platform and configuration.
 * 
 * @param config Full system configuration
 * @return Capture instance or nullptr on failure
 */
std::unique_ptr<ICapture> create_capture(const Config& config);

/**
 * @brief Create capture instance with explicit configuration
 * 
 * @param capture_config Capture-specific configuration
 * @return Capture instance or nullptr on failure
 */
std::unique_ptr<ICapture> create_capture(const CaptureConfig& capture_config);

}  // namespace lagari
