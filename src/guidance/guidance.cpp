#include "lagari/guidance/guidance.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#include <algorithm>
#include <cmath>

namespace lagari {

// ============================================================================
// PIDController Implementation
// ============================================================================

PIDController::PIDController(float kp, float ki, float kd)
    : kp_(kp), ki_(ki), kd_(kd) {}

float PIDController::compute(float error, float dt) {
    // Proportional term
    float p_term = kp_ * error;
    
    // Integral term
    integral_ += error * dt;
    integral_ = std::clamp(integral_, -integral_limit_, integral_limit_);
    float i_term = ki_ * integral_;
    
    // Derivative term
    float d_term = 0.0f;
    if (!first_run_) {
        float derivative = (error - prev_error_) / dt;
        d_term = kd_ * derivative;
    }
    first_run_ = false;
    prev_error_ = error;
    
    // Sum and clamp
    float output = p_term + i_term + d_term;
    return std::clamp(output, min_output_, max_output_);
}

void PIDController::reset() {
    integral_ = 0.0f;
    prev_error_ = 0.0f;
    first_run_ = true;
}

void PIDController::set_limits(float min_output, float max_output) {
    min_output_ = min_output;
    max_output_ = max_output;
}

void PIDController::set_gains(float kp, float ki, float kd) {
    kp_ = kp;
    ki_ = ki;
    kd_ = kd;
}

void PIDController::get_gains(float& kp, float& ki, float& kd) const {
    kp = kp_;
    ki = ki_;
    kd = kd_;
}

void PIDController::set_integral_limit(float limit) {
    integral_limit_ = limit;
}

// ============================================================================
// PIDGuidance Implementation
// ============================================================================

class PIDGuidance : public IGuidance {
public:
    explicit PIDGuidance(const GuidanceConfig& config)
        : config_(config)
        , roll_pid_(config.roll_kp, config.roll_ki, config.roll_kd)
        , pitch_pid_(config.pitch_kp, config.pitch_ki, config.pitch_kd)
        , yaw_pid_(config.yaw_kp, config.yaw_ki, config.yaw_kd)
    {
        roll_pid_.set_limits(-config.max_roll_rad, config.max_roll_rad);
        pitch_pid_.set_limits(-config.max_pitch_rad, config.max_pitch_rad);
        yaw_pid_.set_limits(-config.max_yaw_rate_rad, config.max_yaw_rate_rad);
    }

    bool initialize(const Config& config) override {
        (void)config;
        LOG_INFO("PID Guidance initialized");
        return true;
    }

    void start() override {
        running_ = true;
        LOG_INFO("PID Guidance started");
    }

    void stop() override {
        running_ = false;
        LOG_INFO("PID Guidance stopped");
    }

    bool is_running() const override {
        return running_;
    }

    std::string name() const override {
        return "PIDGuidance";
    }

    GuidanceCommand compute(const DetectionResult& result,
                           const VehicleState& state) override {
        GuidanceCommand cmd;
        cmd.timestamp = Clock::now();
        cmd.valid = false;
        
        // Find target detection
        auto target = result.find_by_class(config_.target_class_id);
        if (!target) {
            tracking_ = false;
            return cmd;
        }
        
        // Compute error from target position
        error_x_ = target->bbox.x - config_.target_x;
        error_y_ = target->bbox.y - config_.target_y;
        
        // Time delta
        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - last_update_).count();
        if (dt <= 0.0f || dt > 1.0f) {
            dt = 1.0f / config_.update_rate_hz;
        }
        last_update_ = now;
        
        // Compute control outputs
        // Note: Positive X error (target right of center) -> negative roll to move right
        // Note: Positive Y error (target below center) -> negative pitch to move forward/down
        cmd.roll = -roll_pid_.compute(error_x_, dt);
        cmd.pitch = -pitch_pid_.compute(error_y_, dt);
        cmd.yaw_rate = 0.0f;  // Could add yaw control based on heading
        cmd.thrust = 0.5f;    // Neutral thrust, could be altitude controlled
        
        cmd.valid = true;
        tracking_ = true;
        
        (void)state;  // Could use for more advanced control
        
        return cmd;
    }

    void set_target_class(int class_id) override {
        config_.target_class_id = class_id;
    }

    void set_target_position(float x, float y) override {
        config_.target_x = x;
        config_.target_y = y;
    }

    void reset() override {
        roll_pid_.reset();
        pitch_pid_.reset();
        yaw_pid_.reset();
        tracking_ = false;
        error_x_ = 0.0f;
        error_y_ = 0.0f;
    }

    void set_roll_gains(float kp, float ki, float kd) override {
        roll_pid_.set_gains(kp, ki, kd);
    }

    void set_pitch_gains(float kp, float ki, float kd) override {
        pitch_pid_.set_gains(kp, ki, kd);
    }

    void set_yaw_gains(float kp, float ki, float kd) override {
        yaw_pid_.set_gains(kp, ki, kd);
    }

    bool is_tracking() const override {
        return tracking_;
    }

    void get_error(float& error_x, float& error_y) const override {
        error_x = error_x_;
        error_y = error_y_;
    }

private:
    GuidanceConfig config_;
    PIDController roll_pid_;
    PIDController pitch_pid_;
    PIDController yaw_pid_;
    
    bool running_ = false;
    bool tracking_ = false;
    float error_x_ = 0.0f;
    float error_y_ = 0.0f;
    TimePoint last_update_ = Clock::now();
};

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<IGuidance> create_guidance(const Config& config) {
    GuidanceConfig gc;
    gc.target_class_id = config.get_int("guidance.target_class_id", 0);
    
    gc.roll_kp = config.get_float("guidance.pid.roll.kp", 0.5f);
    gc.roll_ki = config.get_float("guidance.pid.roll.ki", 0.01f);
    gc.roll_kd = config.get_float("guidance.pid.roll.kd", 0.1f);
    
    gc.pitch_kp = config.get_float("guidance.pid.pitch.kp", 0.5f);
    gc.pitch_ki = config.get_float("guidance.pid.pitch.ki", 0.01f);
    gc.pitch_kd = config.get_float("guidance.pid.pitch.kd", 0.1f);
    
    gc.yaw_kp = config.get_float("guidance.pid.yaw.kp", 0.3f);
    gc.yaw_ki = config.get_float("guidance.pid.yaw.ki", 0.0f);
    gc.yaw_kd = config.get_float("guidance.pid.yaw.kd", 0.05f);
    
    float max_roll_deg = config.get_float("guidance.max_roll_deg", 15.0f);
    float max_pitch_deg = config.get_float("guidance.max_pitch_deg", 15.0f);
    float max_yaw_rate_dps = config.get_float("guidance.max_yaw_rate_dps", 30.0f);
    
    gc.max_roll_rad = max_roll_deg * M_PI / 180.0f;
    gc.max_pitch_rad = max_pitch_deg * M_PI / 180.0f;
    gc.max_yaw_rate_rad = max_yaw_rate_dps * M_PI / 180.0f;
    
    gc.update_rate_hz = config.get_float("guidance.update_rate_hz", 50.0f);
    
    return create_guidance(gc);
}

std::unique_ptr<IGuidance> create_guidance(const GuidanceConfig& config) {
    return std::make_unique<PIDGuidance>(config);
}

}  // namespace lagari
