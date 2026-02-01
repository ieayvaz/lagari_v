#include <gtest/gtest.h>

#include "lagari/core/types.hpp"
#include "lagari/core/config.hpp"

using namespace lagari;

TEST(TypesTest, PixelFormatBytesPerPixel) {
    EXPECT_EQ(bytes_per_pixel(PixelFormat::RGB24), 3);
    EXPECT_EQ(bytes_per_pixel(PixelFormat::BGR24), 3);
    EXPECT_EQ(bytes_per_pixel(PixelFormat::RGBA32), 4);
    EXPECT_EQ(bytes_per_pixel(PixelFormat::GRAY8), 1);
    EXPECT_EQ(bytes_per_pixel(PixelFormat::GRAY16), 2);
    EXPECT_EQ(bytes_per_pixel(PixelFormat::UNKNOWN), 0);
}

TEST(TypesTest, FrameMetadataDataSize) {
    FrameMetadata meta;
    meta.width = 1280;
    meta.height = 720;
    meta.format = PixelFormat::BGR24;
    meta.stride = 0;
    
    EXPECT_EQ(meta.data_size(), 1280 * 720 * 3);
    
    // With stride (padded rows)
    meta.stride = 1280 * 3 + 64;  // 64 bytes padding per row
    EXPECT_EQ(meta.data_size(), (1280 * 3 + 64) * 720);
}

TEST(TypesTest, FrameConstruction) {
    Frame frame(640, 480, PixelFormat::RGB24);
    
    EXPECT_TRUE(frame.valid());
    EXPECT_EQ(frame.metadata.width, 640);
    EXPECT_EQ(frame.metadata.height, 480);
    EXPECT_EQ(frame.metadata.format, PixelFormat::RGB24);
    EXPECT_EQ(frame.size(), 640 * 480 * 3);
    EXPECT_NE(frame.ptr(), nullptr);
}

TEST(TypesTest, FrameInvalid) {
    Frame frame;
    EXPECT_FALSE(frame.valid());
}

TEST(TypesTest, BoundingBoxToPixels) {
    BoundingBox bbox;
    bbox.x = 0.5f;  // Center
    bbox.y = 0.5f;
    bbox.width = 0.2f;
    bbox.height = 0.1f;
    
    int x, y, w, h;
    bbox.to_pixels(1000, 1000, x, y, w, h);
    
    EXPECT_EQ(w, 200);
    EXPECT_EQ(h, 100);
    EXPECT_EQ(x, 400);  // 500 - 200/2
    EXPECT_EQ(y, 450);  // 500 - 100/2
}

TEST(TypesTest, BoundingBoxIoU) {
    BoundingBox box1{0.5f, 0.5f, 0.4f, 0.4f};
    BoundingBox box2{0.5f, 0.5f, 0.4f, 0.4f};
    
    // Identical boxes
    EXPECT_FLOAT_EQ(box1.iou(box2), 1.0f);
    
    // Non-overlapping boxes
    BoundingBox box3{0.1f, 0.1f, 0.1f, 0.1f};
    EXPECT_FLOAT_EQ(box1.iou(box3), 0.0f);
    
    // Partial overlap
    BoundingBox box4{0.6f, 0.6f, 0.4f, 0.4f};
    float iou = box1.iou(box4);
    EXPECT_GT(iou, 0.0f);
    EXPECT_LT(iou, 1.0f);
}

TEST(TypesTest, DetectionResultFindByClass) {
    DetectionResult result;
    result.detections.push_back(Detection{{0.5f, 0.5f, 0.1f, 0.1f}, 0.9f, 0, "person",-1});
    result.detections.push_back(Detection{{0.3f, 0.3f, 0.2f, 0.2f}, 0.8f, 1, "car",-1});
    result.detections.push_back(Detection{{0.7f, 0.7f, 0.1f, 0.1f}, 0.7f, 2, "dog",-1});
    
    auto person = result.find_by_class(0);
    ASSERT_TRUE(person.has_value());
    EXPECT_EQ(person->class_name, "person");
    
    auto cat = result.find_by_class(99);
    EXPECT_FALSE(cat.has_value());
}

TEST(TypesTest, DetectionResultBest) {
    DetectionResult result;
    result.detections.push_back(Detection{{0.5f, 0.5f, 0.1f, 0.1f}, 0.7f, 0, "a",-1});
    result.detections.push_back(Detection{{0.3f, 0.3f, 0.2f, 0.2f}, 0.95f, 1, "b",-1});
    result.detections.push_back(Detection{{0.7f, 0.7f, 0.1f, 0.1f}, 0.8f, 2, "c",-1});
    
    auto best = result.best();
    ASSERT_TRUE(best.has_value());
    EXPECT_EQ(best->class_name, "b");
    EXPECT_FLOAT_EQ(best->confidence, 0.95f);
}

TEST(TypesTest, GuidanceCommandClamp) {
    GuidanceCommand cmd;
    cmd.roll = 1.0f;
    cmd.pitch = -1.0f;
    cmd.yaw_rate = 2.0f;
    cmd.thrust = 1.5f;
    
    cmd.clamp(0.5f, 1.0f);
    
    EXPECT_FLOAT_EQ(cmd.roll, 0.5f);
    EXPECT_FLOAT_EQ(cmd.pitch, -0.5f);
    EXPECT_FLOAT_EQ(cmd.yaw_rate, 1.0f);
    EXPECT_FLOAT_EQ(cmd.thrust, 1.0f);
}

TEST(TypesTest, SystemStateString) {
    EXPECT_STREQ(to_string(SystemState::INIT), "INIT");
    EXPECT_STREQ(to_string(SystemState::IDLE), "IDLE");
    EXPECT_STREQ(to_string(SystemState::SEARCHING), "SEARCHING");
    EXPECT_STREQ(to_string(SystemState::DETECTED), "DETECTED");
    EXPECT_STREQ(to_string(SystemState::TRACKING), "TRACKING");
    EXPECT_STREQ(to_string(SystemState::ERROR), "ERROR");
    EXPECT_STREQ(to_string(SystemState::SHUTDOWN), "SHUTDOWN");
}
