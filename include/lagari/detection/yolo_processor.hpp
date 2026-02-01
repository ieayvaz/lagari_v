#pragma once

#include "lagari/detection/detector.hpp"
#include "lagari/core/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <fstream>

namespace lagari {

/**
 * @brief YOLO preprocessing and postprocessing utilities
 * 
 * Common functionality shared across all YOLO backends.
 * Handles input preprocessing, output parsing, and NMS.
 */
class YOLOProcessor {
public:
    explicit YOLOProcessor(const DetectionConfig& config);
    virtual ~YOLOProcessor() = default;

    /**
     * @brief Preprocess frame for YOLO inference
     * 
     * Performs letterbox resize, normalization, and channel ordering.
     * 
     * @param frame Input frame (BGR)
     * @param output Output buffer (CHW format, normalized)
     * @return Scale and padding info for coordinate restoration
     */
    struct PreprocessInfo {
        float scale;
        int pad_x;
        int pad_y;
        int orig_width;
        int orig_height;
    };
    
    PreprocessInfo preprocess(const Frame& frame, float* output);
    PreprocessInfo preprocess(const cv::Mat& frame, float* output);

    /**
     * @brief Parse YOLO output and apply NMS
     * 
     * Handles different YOLO output formats (v5/v7/v8/v10/v11).
     * 
     * @param output Raw inference output
     * @param output_shape Output tensor shape
     * @param preproc_info Preprocessing info for coordinate restoration
     * @return Detection result with boxes
     */
    DetectionResult postprocess(
        const float* output,
        const std::vector<int>& output_shape,
        const PreprocessInfo& preproc_info,
        uint64_t frame_id = 0);

    /**
     * @brief Load class names from file
     */
    bool load_labels(const std::string& labels_path);

    /**
     * @brief Get loaded class names
     */
    const std::vector<std::string>& class_names() const { return class_names_; }

    /**
     * @brief Get number of classes
     */
    int num_classes() const { return static_cast<int>(class_names_.size()); }

    /**
     * @brief Set confidence threshold
     */
    void set_confidence_threshold(float threshold) { conf_threshold_ = threshold; }

    /**
     * @brief Set NMS threshold
     */
    void set_nms_threshold(float threshold) { nms_threshold_ = threshold; }

    /**
     * @brief Get input size in bytes (for buffer allocation)
     */
    size_t input_size_bytes() const {
        return input_width_ * input_height_ * 3 * sizeof(float);
    }

protected:
    // Output parsing for different YOLO versions
    std::vector<Detection> parse_yolov5_output(
        const float* output, int num_boxes, int num_classes,
        const PreprocessInfo& info);
    
    std::vector<Detection> parse_yolov8_output(
        const float* output, int num_boxes, int num_classes,
        const PreprocessInfo& info);
    
    std::vector<Detection> parse_yolov10_output(
        const float* output, int num_boxes, int num_classes,
        const PreprocessInfo& info);

    // NMS implementation
    std::vector<Detection> apply_nms(
        std::vector<Detection>& detections,
        float nms_threshold);

    // Coordinate transformation
    void scale_coords(Detection& det, const PreprocessInfo& info);

    // Configuration
    YOLOVersion yolo_version_;
    uint32_t input_width_;
    uint32_t input_height_;
    float conf_threshold_;
    float nms_threshold_;
    int max_detections_;
    bool normalize_;
    bool swap_rb_;
    std::vector<int> target_classes_;

    // Class names
    std::vector<std::string> class_names_;
};

/**
 * @brief Base class for YOLO detector implementations
 * 
 * Provides common functionality for all YOLO backends.
 */
class YOLODetectorBase : public IDetector {
public:
    explicit YOLODetectorBase(const DetectionConfig& config);
    ~YOLODetectorBase() override = default;

    // IDetector interface
    DetectionResult detect(const Frame& frame) override;
    bool detect_async(FramePtr frame) override;
    DetectionResult get_latest_result() override;
    void set_confidence_threshold(float threshold) override;
    void set_nms_threshold(float threshold) override;
    const DetectionConfig& config() const override { return config_; }
    DetectionStats get_stats() const override;
    const std::vector<std::string>& class_names() const override;

protected:
    /**
     * @brief Run inference on preprocessed input
     * 
     * Backend-specific implementation.
     * 
     * @param input Preprocessed input (CHW, normalized)
     * @param output Output buffer
     * @return true if inference successful
     */
    virtual bool infer(const float* input, float* output) = 0;

    /**
     * @brief Get output tensor shape
     */
    virtual std::vector<int> get_output_shape() const = 0;

    /**
     * @brief Allocate buffers for inference
     */
    virtual bool allocate_buffers() = 0;

    DetectionConfig config_;
    YOLOProcessor processor_;
    
    // Buffers
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;

    // Latest result (for async mode)
    mutable std::mutex result_mutex_;
    DetectionResult latest_result_;

    // Statistics
    mutable std::mutex stats_mutex_;
    DetectionStats stats_;
    uint64_t inference_count_ = 0;
    double total_inference_time_ = 0.0;
};

}  // namespace lagari
