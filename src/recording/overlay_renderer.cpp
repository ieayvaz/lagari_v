/**
 * @file overlay_renderer.cpp
 * @brief Overlay rendering implementation using OpenCV
 */

#include "lagari/recording/overlay_renderer.hpp"
#include "lagari/core/logger.hpp"

#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>

namespace lagari {

// ============================================================================
// Utility Functions
// ============================================================================

cv::Mat frame_to_mat(const Frame& frame) {
    if (!frame.valid()) {
        return cv::Mat();
    }

    int cv_type = CV_8UC3;  // Default BGR24
    switch (frame.metadata.format) {
        case PixelFormat::BGR24:
        case PixelFormat::RGB24:
            cv_type = CV_8UC3;
            break;
        case PixelFormat::BGRA32:
        case PixelFormat::RGBA32:
            cv_type = CV_8UC4;
            break;
        case PixelFormat::GRAY8:
            cv_type = CV_8UC1;
            break;
        case PixelFormat::GRAY16:
            cv_type = CV_16UC1;
            break;
        default:
            cv_type = CV_8UC3;
            break;
    }

    // Create Mat that shares Frame's data (no copy)
    return cv::Mat(
        static_cast<int>(frame.metadata.height),
        static_cast<int>(frame.metadata.width),
        cv_type,
        const_cast<uint8_t*>(frame.ptr()),
        frame.metadata.stride > 0 ? frame.metadata.stride : cv::Mat::AUTO_STEP
    );
}

FramePtr mat_to_frame(const cv::Mat& mat, uint64_t frame_id) {
    if (mat.empty()) {
        return nullptr;
    }

    PixelFormat format = PixelFormat::BGR24;
    switch (mat.type()) {
        case CV_8UC3:
            format = PixelFormat::BGR24;
            break;
        case CV_8UC4:
            format = PixelFormat::BGRA32;
            break;
        case CV_8UC1:
            format = PixelFormat::GRAY8;
            break;
        case CV_16UC1:
            format = PixelFormat::GRAY16;
            break;
        default:
            LOG_WARN("OverlayRenderer: Unknown cv::Mat type {}", mat.type());
            format = PixelFormat::BGR24;
            break;
    }

    auto frame = std::make_shared<Frame>(
        static_cast<uint32_t>(mat.cols),
        static_cast<uint32_t>(mat.rows),
        format
    );
    frame->metadata.frame_id = frame_id;
    frame->metadata.timestamp = Clock::now();

    // Deep copy the data
    if (mat.isContinuous()) {
        std::memcpy(frame->ptr(), mat.data, frame->size());
    } else {
        size_t row_bytes = mat.cols * mat.elemSize();
        for (int row = 0; row < mat.rows; ++row) {
            std::memcpy(
                frame->ptr() + row * row_bytes,
                mat.ptr(row),
                row_bytes
            );
        }
    }

    return frame;
}

// ============================================================================
// OverlayRenderer Implementation
// ============================================================================

OverlayRenderer::OverlayRenderer(const OverlayConfig& config)
    : config_(config)
{
}

FramePtr OverlayRenderer::render(const Frame& frame,
                                 const DetectionResult* detections,
                                 SystemState state,
                                 Duration latency)
{
    if (!config_.enabled) {
        // Return a copy of the frame without overlay
        cv::Mat mat = frame_to_mat(frame);
        if (mat.empty()) {
            return nullptr;
        }
        return mat_to_frame(mat.clone(), frame.metadata.frame_id);
    }

    // Create a copy of the frame data
    cv::Mat mat = frame_to_mat(frame).clone();
    if (mat.empty()) {
        return nullptr;
    }

    // Render overlay onto the copy
    render_inplace(mat, detections, state, latency);

    return mat_to_frame(mat, frame.metadata.frame_id);
}

void OverlayRenderer::render_inplace(cv::Mat& image,
                                     const DetectionResult* detections,
                                     SystemState state,
                                     Duration latency)
{
    if (!config_.enabled || image.empty()) {
        return;
    }

    // Convert to BGR if needed for drawing
    if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    }

    // Draw overlay elements
    if (config_.bounding_boxes && detections) {
        draw_bounding_boxes(image, *detections);
    }

    if (config_.timestamp) {
        draw_timestamp(image);
    }

    if (config_.state) {
        draw_state(image, state);
    }

    if (config_.latency) {
        draw_latency(image, latency);
    }
}

void OverlayRenderer::draw_bounding_boxes(cv::Mat& image, const DetectionResult& detections)
{
    const int img_width = image.cols;
    const int img_height = image.rows;

    for (const auto& det : detections.detections) {
        // Convert normalized coordinates to pixels
        int x, y, w, h;
        det.bbox.to_pixels(img_width, img_height, x, y, w, h);

        // Clamp to image bounds
        x = std::max(0, std::min(x, img_width - 1));
        y = std::max(0, std::min(y, img_height - 1));
        w = std::min(w, img_width - x);
        h = std::min(h, img_height - y);

        // Draw bounding box
        cv::rectangle(image, cv::Rect(x, y, w, h), 
                     config_.colors.box_color, config_.line_thickness);

        // Build label string
        std::ostringstream label;
        if (config_.draw_class_name && !det.class_name.empty()) {
            label << det.class_name;
        } else if (config_.draw_class_name) {
            label << "class_" << det.class_id;
        }
        
        if (config_.draw_confidence) {
            if (label.tellp() > 0) label << " ";
            label << std::fixed << std::setprecision(2) << det.confidence;
        }

        if (det.track_id.has_value()) {
            label << " [" << det.track_id.value() << "]";
        }

        // Draw label with background
        std::string label_str = label.str();
        if (!label_str.empty()) {
            draw_text_with_background(
                image, label_str,
                cv::Point(x, y - 5),
                config_.colors.text_color,
                config_.colors.box_color
            );
        }
    }
}

void OverlayRenderer::draw_timestamp(cv::Mat& image)
{
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();

    // Draw in top-left corner
    draw_text_with_background(
        image, oss.str(),
        cv::Point(10, 25),
        config_.colors.text_color,
        config_.colors.bg_color
    );
}

void OverlayRenderer::draw_state(cv::Mat& image, SystemState state)
{
    std::string state_str = to_string(state);

    // Choose color based on state
    cv::Scalar color;
    switch (state) {
        case SystemState::ERROR:
        case SystemState::SHUTDOWN:
            color = config_.colors.state_error;
            break;
        case SystemState::IDLE:
        case SystemState::INIT:
            color = config_.colors.state_idle;
            break;
        default:
            color = config_.colors.state_active;
            break;
    }

    // Draw in top-right area
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX,
                                          config_.font_scale, 1, &baseline);
    int x = image.cols - text_size.width - 20;

    draw_text_with_background(
        image, state_str,
        cv::Point(x, 25),
        config_.colors.text_color,
        color
    );
}

void OverlayRenderer::draw_latency(cv::Mat& image, Duration latency)
{
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(latency).count();
    
    std::ostringstream oss;
    oss << "Latency: " << ms << " ms";

    // Color based on latency (green < 50ms, yellow < 100ms, red >= 100ms)
    cv::Scalar color;
    if (ms < 50) {
        color = cv::Scalar(0, 255, 0);  // Green
    } else if (ms < 100) {
        color = cv::Scalar(0, 255, 255);  // Yellow
    } else {
        color = cv::Scalar(0, 0, 255);  // Red
    }

    // Draw below timestamp
    draw_text_with_background(
        image, oss.str(),
        cv::Point(10, 50),
        config_.colors.text_color,
        color
    );
}

void OverlayRenderer::draw_text_with_background(cv::Mat& image,
                                                 const std::string& text,
                                                 cv::Point position,
                                                 cv::Scalar text_color,
                                                 cv::Scalar bg_color)
{
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(
        text, cv::FONT_HERSHEY_SIMPLEX, config_.font_scale, 1, &baseline);

    // Adjust position if it would go off screen
    position.x = std::max(0, std::min(position.x, image.cols - text_size.width - 4));
    position.y = std::max(text_size.height + 4, std::min(position.y, image.rows - 4));

    // Draw background rectangle
    cv::rectangle(
        image,
        cv::Point(position.x - 2, position.y - text_size.height - 2),
        cv::Point(position.x + text_size.width + 2, position.y + baseline + 2),
        bg_color,
        cv::FILLED
    );

    // Draw text
    cv::putText(
        image, text, position,
        cv::FONT_HERSHEY_SIMPLEX, config_.font_scale,
        text_color, 1, cv::LINE_AA
    );
}

}  // namespace lagari
