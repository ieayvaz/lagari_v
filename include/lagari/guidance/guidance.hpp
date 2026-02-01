#pragma once

#include "lagari/core/module.hpp"
#include "lagari/core/types.hpp"
#include <memory>

namespace lagari {

class Config;

/**
 * @brief PID Controller
 */
class PIDController {
public:
    PIDController(float kp = 0.0f, float ki = 0.0f, float kd = 0.0f);

    /**
     * @brief Compute control output
     * 
     * @param error Current error (target - actual)
     * @param dt Time step in seconds
     * @return Control output
     */
    float compute(float error, float dt);

    /**
     * @brief Reset controller state
     */
    void reset();

    /**
     * @brief Set output limits
     */
    void set_limits(float min_output, float max_output);

    /**
     * @brief Set gains
     */
    void set_gains(float kp, float ki, float kd);

    /**
     * @brief Get current gains
     */
    void get_gains(float& kp, float& ki, float& kd) const;

    /**
     * @brief Set integral windup limit
     */
    void set_integral_limit(float limit);

private:
    float kp_, ki_, kd_;
    float integral_ = 0.0f;
    float prev_error_ = 0.0f;
    float min_output_ = -1.0f;
    float max_output_ = 1.0f;
    float integral_limit_ = 1.0f;
    bool first_run_ = true;
};

/**
 * @brief Guidance configuration
 */
struct GuidanceConfig {
    int target_class_id = 0;
    
    // PID gains for roll
    float roll_kp = 0.5f;
    float roll_ki = 0.01f;
    float roll_kd = 0.1f;
    
    // PID gains for pitch
    float pitch_kp = 0.5f;
    float pitch_ki = 0.01f;
    float pitch_kd = 0.1f;
    
    // PID gains for yaw
    float yaw_kp = 0.3f;
    float yaw_ki = 0.0f;
    float yaw_kd = 0.05f;
    
    // Safety limits
    float max_roll_rad = 0.26f;      // ~15 deg
    float max_pitch_rad = 0.26f;     // ~15 deg
    float max_yaw_rate_rad = 0.52f;  // ~30 deg/s
    
    // Target position (normalized image coordinates)
    float target_x = 0.5f;  // Center of image
    float target_y = 0.5f;  // Center of image
    
    // Update rate
    float update_rate_hz = 50.0f;
};

/**
 * @brief Guidance interface
 */
class IGuidance : public IModule {
public:
    virtual ~IGuidance() = default;

    /**
     * @brief Compute guidance command from detection
     * 
     * @param result Detection result
     * @param state Current vehicle state
     * @return Guidance command
     */
    virtual GuidanceCommand compute(const DetectionResult& result,
                                   const VehicleState& state) = 0;

    /**
     * @brief Set target class to track
     */
    virtual void set_target_class(int class_id) = 0;

    /**
     * @brief Set target position in image
     * 
     * @param x Normalized X position (0-1)
     * @param y Normalized Y position (0-1)
     */
    virtual void set_target_position(float x, float y) = 0;

    /**
     * @brief Reset controller state
     */
    virtual void reset() = 0;

    /**
     * @brief Update PID gains at runtime
     */
    virtual void set_roll_gains(float kp, float ki, float kd) = 0;
    virtual void set_pitch_gains(float kp, float ki, float kd) = 0;
    virtual void set_yaw_gains(float kp, float ki, float kd) = 0;

    /**
     * @brief Check if target is being tracked
     */
    virtual bool is_tracking() const = 0;

    /**
     * @brief Get current tracking error (if tracking)
     */
    virtual void get_error(float& error_x, float& error_y) const = 0;
};

/**
 * @brief Create guidance instance
 */
std::unique_ptr<IGuidance> create_guidance(const Config& config);
std::unique_ptr<IGuidance> create_guidance(const GuidanceConfig& config);

}  // namespace lagari
