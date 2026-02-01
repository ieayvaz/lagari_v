#include <gtest/gtest.h>

#include "lagari/guidance/guidance.hpp"

using namespace lagari;

class PIDControllerTest : public ::testing::Test {
protected:
    PIDController pid{1.0f, 0.1f, 0.01f};
};

TEST_F(PIDControllerTest, ZeroError) {
    float output = pid.compute(0.0f, 0.01f);
    EXPECT_FLOAT_EQ(output, 0.0f);
}

TEST_F(PIDControllerTest, ProportionalResponse) {
    PIDController p_only{1.0f, 0.0f, 0.0f};
    
    float output = p_only.compute(0.5f, 0.01f);
    EXPECT_FLOAT_EQ(output, 0.5f);
    
    output = p_only.compute(-0.3f, 0.01f);
    EXPECT_FLOAT_EQ(output, -0.3f);
}

TEST_F(PIDControllerTest, IntegralAccumulation) {
    PIDController i_only{0.0f, 1.0f, 0.0f};
    
    // Constant error should accumulate
    float output1 = i_only.compute(1.0f, 0.1f);
    float output2 = i_only.compute(1.0f, 0.1f);
    float output3 = i_only.compute(1.0f, 0.1f);
    
    EXPECT_GT(output2, output1);
    EXPECT_GT(output3, output2);
}

TEST_F(PIDControllerTest, DerivativeResponse) {
    PIDController d_only{0.0f, 0.0f, 1.0f};
    
    // First call establishes baseline
    d_only.compute(0.0f, 0.01f);
    
    // Increasing error should give positive output
    float output = d_only.compute(1.0f, 0.01f);
    EXPECT_GT(output, 0.0f);
    
    // Decreasing error should give negative output
    output = d_only.compute(0.5f, 0.01f);
    EXPECT_LT(output, 0.0f);
}

TEST_F(PIDControllerTest, OutputLimits) {
    pid.set_limits(-0.5f, 0.5f);
    
    // Large error should be clamped
    float output = pid.compute(100.0f, 0.01f);
    EXPECT_LE(output, 0.5f);
    
    output = pid.compute(-100.0f, 0.01f);
    EXPECT_GE(output, -0.5f);
}

TEST_F(PIDControllerTest, Reset) {
    // Accumulate some integral
    pid.compute(1.0f, 0.1f);
    pid.compute(1.0f, 0.1f);
    pid.compute(1.0f, 0.1f);
    
    pid.reset();
    
    // After reset, P-only response to new error
    PIDController p_only{1.0f, 0.0f, 0.0f};
    float output_reset = pid.compute(0.5f, 0.1f);
    float output_fresh = p_only.compute(0.5f, 0.1f);
    
    // Should be similar (small integral from first step after reset)
    EXPECT_NEAR(output_reset, output_fresh, 0.1f);
}

TEST_F(PIDControllerTest, SetGains) {
    pid.set_gains(2.0f, 0.0f, 0.0f);
    
    float output = pid.compute(0.5f, 0.01f);
    EXPECT_FLOAT_EQ(output, 1.0f);  // 2.0 * 0.5
}

TEST_F(PIDControllerTest, GetGains) {
    float kp, ki, kd;
    pid.get_gains(kp, ki, kd);
    
    EXPECT_FLOAT_EQ(kp, 1.0f);
    EXPECT_FLOAT_EQ(ki, 0.1f);
    EXPECT_FLOAT_EQ(kd, 0.01f);
}

TEST_F(PIDControllerTest, IntegralWindup) {
    PIDController i_heavy{0.0f, 10.0f, 0.0f};
    i_heavy.set_integral_limit(0.5f);
    i_heavy.set_limits(-1.0f, 1.0f);
    
    // Large constant error for many iterations
    for (int i = 0; i < 100; ++i) {
        i_heavy.compute(1.0f, 0.1f);
    }
    
    // Output should be limited
    float output = i_heavy.compute(1.0f, 0.1f);
    EXPECT_LE(output, 1.0f);
}
