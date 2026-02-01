#pragma once

#include "lagari/capture/capture.hpp"
#include "lagari/core/ring_buffer.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <random>

namespace lagari {

/**
 * @brief Simulation capture for testing without real camera
 * 
 * Generates synthetic frames with optional patterns for testing
 * the vision pipeline.
 */
class SimCapture : public ICapture {
public:
    enum class Pattern {
        SOLID,          // Solid color
        GRADIENT,       // Gradient pattern
        CHECKERBOARD,   // Checkerboard pattern
        MOVING_BOX,     // Moving box for tracking tests
        NOISE           // Random noise
    };

    explicit SimCapture(const CaptureConfig& config);
    ~SimCapture() override;

    // IModule interface
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override;
    std::string name() const override { return "SimCapture"; }

    // ICapture interface
    FramePtr get_latest_frame() override;
    FramePtr wait_for_frame(uint32_t timeout_ms) override;
    void set_frame_callback(FrameCallback callback) override;
    CaptureStats get_stats() const override;
    bool is_open() const override { return true; }
    const CaptureConfig& config() const override { return config_; }
    bool set_resolution(uint32_t width, uint32_t height) override;
    bool set_framerate(uint32_t fps) override;
    bool set_exposure(bool auto_exp, float exposure_time) override;

    // Simulation-specific
    void set_pattern(Pattern pattern);
    void set_box_position(float x, float y);  // For MOVING_BOX pattern

private:
    void generate_loop();
    FramePtr generate_frame();
    void draw_solid(uint8_t* data, uint8_t r, uint8_t g, uint8_t b);
    void draw_gradient(uint8_t* data);
    void draw_checkerboard(uint8_t* data, int cell_size);
    void draw_moving_box(uint8_t* data);
    void draw_noise(uint8_t* data);

    CaptureConfig config_;
    Pattern pattern_ = Pattern::MOVING_BOX;
    
    // Threading
    std::thread gen_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    
    // Frame storage
    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    FramePtr latest_frame_;
    uint64_t frame_counter_ = 0;
    
    // Callback
    FrameCallback frame_callback_;
    
    // Moving box state
    float box_x_ = 0.5f;
    float box_y_ = 0.5f;
    float box_vx_ = 0.01f;
    float box_vy_ = 0.008f;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    CaptureStats stats_;
    TimePoint start_time_;
    
    // RNG for noise
    std::mt19937 rng_{std::random_device{}()};
};

}  // namespace lagari
