#include "lagari/capture/sim_capture.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#include <cstring>
#include <algorithm>

namespace lagari {

SimCapture::SimCapture(const CaptureConfig& config)
    : config_(config)
{
}

SimCapture::~SimCapture() {
    stop();
}

bool SimCapture::initialize(const Config& /* config */) {
    LOG_INFO("SimCapture initialized: {}x{} @ {} fps",
             config_.width, config_.height, config_.fps);
    return true;
}

void SimCapture::start() {
    if (running_.load(std::memory_order_acquire)) {
        return;
    }

    should_stop_.store(false, std::memory_order_release);
    start_time_ = Clock::now();
    
    gen_thread_ = std::thread(&SimCapture::generate_loop, this);
    running_.store(true, std::memory_order_release);
    
    LOG_INFO("SimCapture started");
}

void SimCapture::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    should_stop_.store(true, std::memory_order_release);
    frame_cv_.notify_all();

    if (gen_thread_.joinable()) {
        gen_thread_.join();
    }

    running_.store(false, std::memory_order_release);
    LOG_INFO("SimCapture stopped");
}

bool SimCapture::is_running() const {
    return running_.load(std::memory_order_acquire);
}

FramePtr SimCapture::get_latest_frame() {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

FramePtr SimCapture::wait_for_frame(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(frame_mutex_);
    
    if (latest_frame_) {
        return latest_frame_;
    }

    frame_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() {
        return latest_frame_ != nullptr || should_stop_.load(std::memory_order_acquire);
    });

    return latest_frame_;
}

void SimCapture::set_frame_callback(FrameCallback callback) {
    frame_callback_ = std::move(callback);
}

CaptureStats SimCapture::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool SimCapture::set_resolution(uint32_t width, uint32_t height) {
    config_.width = width;
    config_.height = height;
    return true;
}

bool SimCapture::set_framerate(uint32_t fps) {
    config_.fps = fps;
    return true;
}

bool SimCapture::set_exposure(bool /* auto_exp */, float /* exposure_time */) {
    // No-op for simulation
    return true;
}

void SimCapture::set_pattern(Pattern pattern) {
    pattern_ = pattern;
}

void SimCapture::set_box_position(float x, float y) {
    box_x_ = x;
    box_y_ = y;
}

void SimCapture::generate_loop() {
    LOG_DEBUG("Simulation generate thread started");

    auto frame_duration = std::chrono::microseconds(1000000 / config_.fps);
    auto next_frame_time = Clock::now();

    while (!should_stop_.load(std::memory_order_acquire)) {
        auto now = Clock::now();
        
        if (now < next_frame_time) {
            std::this_thread::sleep_until(next_frame_time);
        }
        next_frame_time = Clock::now() + frame_duration;

        // Generate frame
        FramePtr frame = generate_frame();
        frame->metadata.frame_id = ++frame_counter_;
        frame->metadata.timestamp = Clock::now();

        // Store latest frame
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_ = frame;
        }
        frame_cv_.notify_all();

        // Call callback
        if (frame_callback_) {
            frame_callback_(frame);
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.frames_captured++;
            auto elapsed = std::chrono::duration<float>(Clock::now() - start_time_).count();
            if (elapsed > 0) {
                stats_.average_fps = stats_.frames_captured / elapsed;
                stats_.current_fps = static_cast<float>(config_.fps);
            }
        }
    }

    LOG_DEBUG("Simulation generate thread exiting");
}

FramePtr SimCapture::generate_frame() {
    auto frame = std::make_shared<Frame>(config_.width, config_.height, PixelFormat::BGR24);

    switch (pattern_) {
        case Pattern::SOLID:
            draw_solid(frame->ptr(), 64, 64, 64);
            break;
        case Pattern::GRADIENT:
            draw_gradient(frame->ptr());
            break;
        case Pattern::CHECKERBOARD:
            draw_checkerboard(frame->ptr(), 32);
            break;
        case Pattern::MOVING_BOX:
            draw_moving_box(frame->ptr());
            break;
        case Pattern::NOISE:
            draw_noise(frame->ptr());
            break;
    }

    return frame;
}

void SimCapture::draw_solid(uint8_t* data, uint8_t r, uint8_t g, uint8_t b) {
    for (uint32_t i = 0; i < config_.width * config_.height; ++i) {
        data[i * 3 + 0] = b;  // BGR format
        data[i * 3 + 1] = g;
        data[i * 3 + 2] = r;
    }
}

void SimCapture::draw_gradient(uint8_t* data) {
    for (uint32_t y = 0; y < config_.height; ++y) {
        for (uint32_t x = 0; x < config_.width; ++x) {
            size_t idx = (y * config_.width + x) * 3;
            data[idx + 0] = static_cast<uint8_t>(255 * x / config_.width);      // B
            data[idx + 1] = static_cast<uint8_t>(255 * y / config_.height);     // G
            data[idx + 2] = static_cast<uint8_t>(128);                          // R
        }
    }
}

void SimCapture::draw_checkerboard(uint8_t* data, int cell_size) {
    for (uint32_t y = 0; y < config_.height; ++y) {
        for (uint32_t x = 0; x < config_.width; ++x) {
            bool white = ((x / cell_size) + (y / cell_size)) % 2 == 0;
            uint8_t val = white ? 220 : 35;
            size_t idx = (y * config_.width + x) * 3;
            data[idx + 0] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
}

void SimCapture::draw_moving_box(uint8_t* data) {
    // Dark background
    draw_solid(data, 30, 30, 50);

    // Update box position
    box_x_ += box_vx_;
    box_y_ += box_vy_;

    // Bounce off edges
    if (box_x_ < 0.1f || box_x_ > 0.9f) box_vx_ = -box_vx_;
    if (box_y_ < 0.1f || box_y_ > 0.9f) box_vy_ = -box_vy_;

    // Draw box
    int box_w = static_cast<int>(config_.width * 0.1f);
    int box_h = static_cast<int>(config_.height * 0.1f);
    int box_cx = static_cast<int>(box_x_ * config_.width);
    int box_cy = static_cast<int>(box_y_ * config_.height);
    int x0 = std::max(0, box_cx - box_w / 2);
    int y0 = std::max(0, box_cy - box_h / 2);
    int x1 = std::min(static_cast<int>(config_.width) - 1, box_cx + box_w / 2);
    int y1 = std::min(static_cast<int>(config_.height) - 1, box_cy + box_h / 2);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            size_t idx = (y * config_.width + x) * 3;
            // Bright green box
            data[idx + 0] = 50;   // B
            data[idx + 1] = 220;  // G
            data[idx + 2] = 50;   // R
        }
    }

    // Draw center crosshair
    int cx = config_.width / 2;
    int cy = config_.height / 2;
    for (int i = -20; i <= 20; ++i) {
        if (cx + i >= 0 && cx + i < static_cast<int>(config_.width)) {
            size_t idx = (cy * config_.width + cx + i) * 3;
            data[idx + 0] = 0; data[idx + 1] = 0; data[idx + 2] = 255;
        }
        if (cy + i >= 0 && cy + i < static_cast<int>(config_.height)) {
            size_t idx = ((cy + i) * config_.width + cx) * 3;
            data[idx + 0] = 0; data[idx + 1] = 0; data[idx + 2] = 255;
        }
    }
}

void SimCapture::draw_noise(uint8_t* data) {
    std::uniform_int_distribution<int> dist(0, 255);
    for (uint32_t i = 0; i < config_.width * config_.height * 3; ++i) {
        data[i] = static_cast<uint8_t>(dist(rng_));
    }
}

}  // namespace lagari
