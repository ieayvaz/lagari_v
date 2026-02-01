#pragma once

#include "lagari/core/types.hpp"
#include <memory>
#include <string>
#include <opencv2/core.hpp>

namespace lagari {

/**
 * @brief Overlay rendering configuration
 */
struct OverlayConfig {
    bool enabled = false;           // Master toggle for overlay
    bool timestamp = true;          // Show timestamp
    bool bounding_boxes = true;     // Draw detection bounding boxes
    bool state = true;              // Show system state
    bool latency = true;            // Show processing latency
    
    // Style configuration
    float font_scale = 0.6f;
    int line_thickness = 2;
    bool draw_class_name = true;
    bool draw_confidence = true;
    
    // Colors (BGR format)
    struct Colors {
        cv::Scalar box_color{0, 255, 0};      // Green for boxes
        cv::Scalar text_color{255, 255, 255}; // White for text
        cv::Scalar bg_color{0, 0, 0};         // Black background
        cv::Scalar state_idle{128, 128, 128}; // Gray for IDLE
        cv::Scalar state_active{0, 255, 0};   // Green for active states
        cv::Scalar state_error{0, 0, 255};    // Red for ERROR
    } colors;
};

/**
 * @brief Overlay renderer for drawing detection info on frames
 * 
 * Renders bounding boxes, timestamps, system state, and latency
 * information onto video frames. Used by both display and recording modules.
 */
class OverlayRenderer {
public:
    /**
     * @brief Construct overlay renderer with configuration
     */
    explicit OverlayRenderer(const OverlayConfig& config = OverlayConfig{});
    
    ~OverlayRenderer() = default;

    /**
     * @brief Render overlay onto a frame copy
     * 
     * Creates a copy of the input frame and renders overlay onto it.
     * Original frame is not modified.
     * 
     * @param frame Source frame to render overlay on
     * @param detections Optional detection results for bounding boxes
     * @param state Current system state
     * @param latency Processing latency to display
     * @return New frame with overlay rendered, or nullptr on failure
     */
    FramePtr render(const Frame& frame,
                   const DetectionResult* detections = nullptr,
                   SystemState state = SystemState::IDLE,
                   Duration latency = Duration::zero());

    /**
     * @brief Render overlay in-place on cv::Mat
     * 
     * Modifies the input image directly. More efficient when a copy
     * is not needed.
     * 
     * @param image Image to render overlay on (modified in place)
     * @param detections Optional detection results for bounding boxes
     * @param state Current system state
     * @param latency Processing latency to display
     */
    void render_inplace(cv::Mat& image,
                       const DetectionResult* detections = nullptr,
                       SystemState state = SystemState::IDLE,
                       Duration latency = Duration::zero());

    /**
     * @brief Update overlay configuration
     */
    void set_config(const OverlayConfig& config) { config_ = config; }
    
    /**
     * @brief Get current configuration
     */
    const OverlayConfig& config() const { return config_; }

    /**
     * @brief Check if overlay is enabled
     */
    bool is_enabled() const { return config_.enabled; }

private:
    // Drawing helpers
    void draw_bounding_boxes(cv::Mat& image, const DetectionResult& detections);
    void draw_timestamp(cv::Mat& image);
    void draw_state(cv::Mat& image, SystemState state);
    void draw_latency(cv::Mat& image, Duration latency);
    void draw_text_with_background(cv::Mat& image, const std::string& text,
                                   cv::Point position, cv::Scalar text_color,
                                   cv::Scalar bg_color);

    OverlayConfig config_;
};

/**
 * @brief Convert Frame to cv::Mat (shallow copy, shares data)
 */
cv::Mat frame_to_mat(const Frame& frame);

/**
 * @brief Convert cv::Mat to Frame (deep copy)
 */
FramePtr mat_to_frame(const cv::Mat& mat, uint64_t frame_id = 0);

}  // namespace lagari
