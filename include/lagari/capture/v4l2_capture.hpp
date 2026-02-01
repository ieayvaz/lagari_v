#pragma once

#include "lagari/capture/capture.hpp"
#include "lagari/core/ring_buffer.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <string>

namespace lagari {

/**
 * @brief V4L2-based video capture for USB cameras and webcams
 * 
 * Uses Video4Linux2 API with memory-mapped buffers for efficient
 * frame capture on Linux systems.
 */
class V4L2Capture : public ICapture {
public:
    explicit V4L2Capture(const CaptureConfig& config);
    ~V4L2Capture() override;

    // Non-copyable
    V4L2Capture(const V4L2Capture&) = delete;
    V4L2Capture& operator=(const V4L2Capture&) = delete;

    // IModule interface
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override;
    std::string name() const override { return "V4L2Capture"; }

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

private:
    // V4L2 buffer structure
    struct Buffer {
        void* start = nullptr;
        size_t length = 0;
    };

    // Open and configure device
    bool open_device();
    void close_device();
    bool init_device();
    bool init_mmap();
    bool start_capturing();
    bool stop_capturing();

    // Frame processing
    bool read_frame();
    void process_frame(const void* data, size_t size);
    FramePtr convert_frame(const void* data, size_t size);

    // Capture thread
    void capture_loop();

    // Configuration
    CaptureConfig config_;
    std::string device_path_;

    // V4L2 state
    int fd_ = -1;
    std::vector<Buffer> buffers_;
    uint32_t actual_width_ = 0;
    uint32_t actual_height_ = 0;
    uint32_t actual_fps_ = 0;
    uint32_t pixel_format_ = 0;

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
