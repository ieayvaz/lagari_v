#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <optional>

namespace lagari {

// ============================================================================
// Time Types
// ============================================================================
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Duration = Clock::duration;
using Milliseconds = std::chrono::milliseconds;
using Microseconds = std::chrono::microseconds;

// ============================================================================
// Pixel Format
// ============================================================================
enum class PixelFormat : uint8_t {
    UNKNOWN = 0,
    RGB24,
    BGR24,
    RGBA32,
    BGRA32,
    NV12,       // YUV 4:2:0
    NV21,
    YUYV,
    GRAY8,
    GRAY16
};

inline constexpr size_t bytes_per_pixel(PixelFormat format) {
    switch (format) {
        case PixelFormat::RGB24:
        case PixelFormat::BGR24:
            return 3;
        case PixelFormat::RGBA32:
        case PixelFormat::BGRA32:
            return 4;
        case PixelFormat::GRAY8:
            return 1;
        case PixelFormat::GRAY16:
            return 2;
        case PixelFormat::NV12:
        case PixelFormat::NV21:
            return 1;  // Per-plane calculation needed
        case PixelFormat::YUYV:
            return 2;
        default:
            return 0;
    }
}

// ============================================================================
// Frame Metadata
// ============================================================================
struct FrameMetadata {
    uint64_t frame_id = 0;
    TimePoint timestamp = Clock::now();
    uint32_t width = 0;
    uint32_t height = 0;
    PixelFormat format = PixelFormat::UNKNOWN;
    uint32_t stride = 0;  // Bytes per row (may include padding)
    
    size_t data_size() const {
        if (stride > 0) {
            return stride * height;
        }
        return width * height * bytes_per_pixel(format);
    }
};

// ============================================================================
// Frame
// ============================================================================
struct Frame {
    FrameMetadata metadata;
    std::shared_ptr<uint8_t[]> data;
    
    Frame() = default;
    
    Frame(uint32_t width, uint32_t height, PixelFormat format)
        : metadata{0, Clock::now(), width, height, format, 0}
    {
        size_t size = metadata.data_size();
        data = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
    }
    
    bool valid() const {
        return data != nullptr && metadata.width > 0 && metadata.height > 0;
    }
    
    size_t size() const { return metadata.data_size(); }
    uint8_t* ptr() { return data.get(); }
    const uint8_t* ptr() const { return data.get(); }
};

using FramePtr = std::shared_ptr<Frame>;

// ============================================================================
// Bounding Box
// ============================================================================
struct BoundingBox {
    float x = 0.0f;       // Center X (normalized 0-1)
    float y = 0.0f;       // Center Y (normalized 0-1)
    float width = 0.0f;   // Width (normalized 0-1)
    float height = 0.0f;  // Height (normalized 0-1)
    
    // Convert to pixel coordinates
    void to_pixels(uint32_t img_width, uint32_t img_height,
                   int& out_x, int& out_y, int& out_w, int& out_h) const {
        out_w = static_cast<int>(width * img_width);
        out_h = static_cast<int>(height * img_height);
        out_x = static_cast<int>(x * img_width) - out_w / 2;
        out_y = static_cast<int>(y * img_height) - out_h / 2;
    }
    
    // Intersection over Union
    float iou(const BoundingBox& other) const {
        float x1 = std::max(x - width/2, other.x - other.width/2);
        float y1 = std::max(y - height/2, other.y - other.height/2);
        float x2 = std::min(x + width/2, other.x + other.width/2);
        float y2 = std::min(y + height/2, other.y + other.height/2);
        
        float inter_w = std::max(0.0f, x2 - x1);
        float inter_h = std::max(0.0f, y2 - y1);
        float inter_area = inter_w * inter_h;
        
        float area1 = width * height;
        float area2 = other.width * other.height;
        float union_area = area1 + area2 - inter_area;
        
        return union_area > 0 ? inter_area / union_area : 0.0f;
    }
};

// ============================================================================
// Detection
// ============================================================================
struct Detection {
    BoundingBox bbox;
    float confidence = 0.0f;
    int class_id = -1;
    std::string class_name;
    
    // Optional tracking ID (set by tracker)
    std::optional<uint64_t> track_id;
};

// ============================================================================
// Detection Result
// ============================================================================
struct DetectionResult {
    uint64_t frame_id = 0;
    TimePoint timestamp;
    TimePoint detection_time;  // When detection completed
    std::vector<Detection> detections;
    
    Duration latency() const {
        return detection_time - timestamp;
    }
    
    // Find detection by class
    std::optional<Detection> find_by_class(int class_id) const {
        for (const auto& det : detections) {
            if (det.class_id == class_id) {
                return det;
            }
        }
        return std::nullopt;
    }
    
    // Find detection with highest confidence
    std::optional<Detection> best() const {
        if (detections.empty()) return std::nullopt;
        return *std::max_element(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) {
                return a.confidence < b.confidence;
            });
    }
};

// ============================================================================
// QR Result
// ============================================================================
struct QRResult {
    bool success = false;
    std::string data;
    std::array<float, 8> corners = {};  // 4 corner points (x,y pairs)
    TimePoint timestamp;
    uint64_t frame_id = 0;
};

// ============================================================================
// Vehicle State (from autopilot)
// ============================================================================
struct VehicleState {
    TimePoint timestamp;
    
    // Position (local NED or GPS)
    float pos_x = 0.0f;
    float pos_y = 0.0f;
    float pos_z = 0.0f;  // Altitude (positive down in NED)
    
    // Velocity
    float vel_x = 0.0f;
    float vel_y = 0.0f;
    float vel_z = 0.0f;
    
    // Attitude (radians)
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    
    // Angular rates
    float roll_rate = 0.0f;
    float pitch_rate = 0.0f;
    float yaw_rate = 0.0f;
    
    bool armed = false;
    bool in_air = false;
};

// ============================================================================
// Guidance Command
// ============================================================================
struct GuidanceCommand {
    TimePoint timestamp;
    
    float roll = 0.0f;       // Roll angle command (radians)
    float pitch = 0.0f;      // Pitch angle command (radians)
    float yaw_rate = 0.0f;   // Yaw rate command (rad/s)
    float thrust = 0.5f;     // Normalized thrust (0-1)
    
    bool valid = false;      // Command validity
    
    // Apply safety limits
    void clamp(float max_angle, float max_yaw_rate) {
        roll = std::clamp(roll, -max_angle, max_angle);
        pitch = std::clamp(pitch, -max_angle, max_angle);
        yaw_rate = std::clamp(yaw_rate, -max_yaw_rate, max_yaw_rate);
        thrust = std::clamp(thrust, 0.0f, 1.0f);
    }
};

// ============================================================================
// System State
// ============================================================================
enum class SystemState : uint8_t {
    INIT = 0,       // System initializing
    IDLE,           // Ready but not processing
    SEARCHING,      // Looking for targets
    DETECTED,       // Target found
    TRACKING,       // Actively tracking
    LANDING,        // Landing sequence
    ERROR,          // Error state
    SHUTDOWN        // Shutting down
};

inline const char* to_string(SystemState state) {
    switch (state) {
        case SystemState::INIT: return "INIT";
        case SystemState::IDLE: return "IDLE";
        case SystemState::SEARCHING: return "SEARCHING";
        case SystemState::DETECTED: return "DETECTED";
        case SystemState::TRACKING: return "TRACKING";
        case SystemState::LANDING: return "LANDING";
        case SystemState::ERROR: return "ERROR";
        case SystemState::SHUTDOWN: return "SHUTDOWN";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Capture Statistics
// ============================================================================
struct CaptureStats {
    uint64_t frames_captured = 0;
    uint64_t frames_dropped = 0;
    float current_fps = 0.0f;
    float average_fps = 0.0f;
    Duration average_latency{0};
};

// ============================================================================
// Detection Statistics
// ============================================================================
struct DetectionStats {
    uint64_t frames_processed = 0;
    float inference_fps = 0.0f;
    Duration average_inference_time{0};
    Duration last_inference_time{0};
};

}  // namespace lagari
