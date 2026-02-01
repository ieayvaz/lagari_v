#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "lagari/capture/capture.hpp"
#include "lagari/core/config.hpp"

// These tests check capture interface and configuration
// V4L2-specific tests require actual hardware

using namespace lagari;

TEST(CaptureConfigTest, DefaultValues) {
    CaptureConfig config;
    
    EXPECT_EQ(config.source, CaptureSource::AUTO);
    EXPECT_EQ(config.width, 1280);
    EXPECT_EQ(config.height, 720);
    EXPECT_EQ(config.fps, 30);
    EXPECT_EQ(config.format, PixelFormat::BGR24);
    EXPECT_EQ(config.camera_id, 0);
    EXPECT_TRUE(config.drop_frames);
    EXPECT_TRUE(config.auto_exposure);
}

TEST(CaptureSourceTest, ToString) {
    EXPECT_STREQ(to_string(CaptureSource::AUTO), "AUTO");
    EXPECT_STREQ(to_string(CaptureSource::CSI), "CSI");
    EXPECT_STREQ(to_string(CaptureSource::USB), "USB");
    EXPECT_STREQ(to_string(CaptureSource::FILE), "FILE");
    EXPECT_STREQ(to_string(CaptureSource::RTSP), "RTSP");
    EXPECT_STREQ(to_string(CaptureSource::SIMULATION), "SIMULATION");
}

#ifdef HAS_V4L2
#include "lagari/capture/v4l2_capture.hpp"
#include <filesystem>

namespace fs = std::filesystem;

class V4L2CaptureTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if a video device exists
        has_camera_ = fs::exists("/dev/video0");
    }
    
    bool has_camera_ = false;
};

TEST_F(V4L2CaptureTest, CreateWithConfig) {
    CaptureConfig config;
    config.source = CaptureSource::USB;
    config.width = 640;
    config.height = 480;
    config.fps = 30;
    
    V4L2Capture capture(config);
    EXPECT_FALSE(capture.is_running());
    EXPECT_FALSE(capture.is_open());
}

TEST_F(V4L2CaptureTest, DISABLED_InitializeWithCamera) {
    // This test requires actual camera hardware
    // Run manually when hardware is available
    
    if (!has_camera_) {
        GTEST_SKIP() << "No camera available at /dev/video0";
    }
    
    CaptureConfig config;
    config.source = CaptureSource::USB;
    config.camera_id = 0;
    config.width = 640;
    config.height = 480;
    config.fps = 30;
    
    V4L2Capture capture(config);
    
    // Initialize with empty config (uses member config)
    Config dummy_config;
    EXPECT_TRUE(capture.initialize(dummy_config));
    EXPECT_TRUE(capture.is_open());
    
    // Start capture
    capture.start();
    EXPECT_TRUE(capture.is_running());
    
    // Wait for frame
    auto frame = capture.wait_for_frame(1000);
    EXPECT_NE(frame, nullptr);
    if (frame) {
        EXPECT_TRUE(frame->valid());
        EXPECT_GT(frame->metadata.width, 0u);
        EXPECT_GT(frame->metadata.height, 0u);
    }
    
    // Stop
    capture.stop();
    EXPECT_FALSE(capture.is_running());
}

#endif  // HAS_V4L2

// Simulation capture tests (don't require hardware)
#include "lagari/capture/sim_capture.hpp"

class SimCaptureTest : public ::testing::Test {
protected:
    CaptureConfig config;
    
    void SetUp() override {
        config.source = CaptureSource::SIMULATION;
        config.width = 640;
        config.height = 480;
        config.fps = 30;
    }
};

TEST_F(SimCaptureTest, Initialize) {
    SimCapture capture(config);
    Config dummy;
    
    EXPECT_TRUE(capture.initialize(dummy));
    EXPECT_TRUE(capture.is_open());
}

TEST_F(SimCaptureTest, StartStop) {
    SimCapture capture(config);
    Config dummy;
    capture.initialize(dummy);
    
    capture.start();
    EXPECT_TRUE(capture.is_running());
    
    capture.stop();
    EXPECT_FALSE(capture.is_running());
}

TEST_F(SimCaptureTest, GeneratesFrames) {
    SimCapture capture(config);
    Config dummy;
    capture.initialize(dummy);
    capture.start();
    
    // Wait for a frame
    auto frame = capture.wait_for_frame(500);
    
    ASSERT_NE(frame, nullptr);
    EXPECT_TRUE(frame->valid());
    EXPECT_EQ(frame->metadata.width, config.width);
    EXPECT_EQ(frame->metadata.height, config.height);
    EXPECT_EQ(frame->metadata.format, PixelFormat::BGR24);
    EXPECT_GT(frame->metadata.frame_id, 0u);
    
    capture.stop();
}

TEST_F(SimCaptureTest, FrameCallback) {
    SimCapture capture(config);
    Config dummy;
    capture.initialize(dummy);
    
    std::atomic<int> callback_count{0};
    capture.set_frame_callback([&callback_count](FramePtr frame) {
        if (frame && frame->valid()) {
            callback_count.fetch_add(1);
        }
    });
    
    capture.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    capture.stop();
    
    // Should have received several frames at 30fps in 200ms
    EXPECT_GE(callback_count.load(), 3);
}

TEST_F(SimCaptureTest, Statistics) {
    SimCapture capture(config);
    Config dummy;
    capture.initialize(dummy);
    capture.start();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    auto stats = capture.get_stats();
    EXPECT_GT(stats.frames_captured, 0u);
    EXPECT_GT(stats.average_fps, 0.0f);
    
    capture.stop();
}
