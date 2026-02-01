#pragma once

#include "lagari/core/module.hpp"
#include "lagari/core/types.hpp"
#include <memory>
#include <string>
#include <vector>

namespace lagari {

class Config;

/**
 * @brief Inference backend type
 */
enum class InferenceBackend : uint8_t {
    AUTO = 0,       // Auto-detect best available
    TENSORRT,       // NVIDIA TensorRT (Jetson, x86 GPU)
    HAILO,          // HailoRT (RPi with Hailo Hat)
    NCNN,           // NCNN (RPi CPU)
    OPENVINO,       // OpenVINO (x86 CPU)
    ONNXRUNTIME     // ONNX Runtime (fallback)
};

inline const char* to_string(InferenceBackend backend) {
    switch (backend) {
        case InferenceBackend::AUTO: return "AUTO";
        case InferenceBackend::TENSORRT: return "TENSORRT";
        case InferenceBackend::HAILO: return "HAILO";
        case InferenceBackend::NCNN: return "NCNN";
        case InferenceBackend::OPENVINO: return "OPENVINO";
        case InferenceBackend::ONNXRUNTIME: return "ONNXRUNTIME";
        default: return "UNKNOWN";
    }
}

/**
 * @brief YOLO model version
 */
enum class YOLOVersion : uint8_t {
    UNKNOWN = 0,
    YOLOv5,
    YOLOv7,
    YOLOv8,
    YOLOv9,
    YOLOv10,
    YOLO11
};

/**
 * @brief Detection configuration
 */
struct DetectionConfig {
    InferenceBackend backend = InferenceBackend::AUTO;
    YOLOVersion yolo_version = YOLOVersion::YOLOv8;
    
    std::string model_path;          // Path to model file
    std::string labels_path;         // Path to class labels file
    
    // Input dimensions
    uint32_t input_width = 640;
    uint32_t input_height = 640;
    
    // Detection parameters
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.45f;
    int max_detections = 100;
    
    // Target classes (empty = all classes)
    std::vector<int> target_classes;
    
    // Performance tuning
    bool fp16 = true;                // Use FP16 inference (if supported)
    bool int8 = false;               // Use INT8 quantization (if supported)
    int batch_size = 1;              // Batch size
    int dla_core = -1;               // DLA core (-1 = disabled, Jetson only)
    
    // Preprocessing
    bool normalize = true;           // Normalize to 0-1
    bool swap_rb = true;             // Swap R and B channels
};

/**
 * @brief Detector interface
 * 
 * Abstract interface for object detection. Different backends implement
 * this interface for various inference runtimes.
 */
class IDetector : public IModule {
public:
    virtual ~IDetector() = default;

    /**
     * @brief Run detection on a frame
     * 
     * @param frame Input frame
     * @return Detection result with bounding boxes
     */
    virtual DetectionResult detect(const Frame& frame) = 0;

    /**
     * @brief Run detection on a frame (async version)
     * 
     * Queues frame for processing and returns immediately.
     * Use get_latest_result() to retrieve results.
     * 
     * @param frame Input frame
     * @return true if frame was queued
     */
    virtual bool detect_async(FramePtr frame) = 0;

    /**
     * @brief Get latest detection result (for async mode)
     * 
     * @return Latest result or empty result if none available
     */
    virtual DetectionResult get_latest_result() = 0;

    /**
     * @brief Load detection model
     * 
     * @param model_path Path to model file
     * @return true if model loaded successfully
     */
    virtual bool load_model(const std::string& model_path) = 0;

    /**
     * @brief Set confidence threshold
     */
    virtual void set_confidence_threshold(float threshold) = 0;

    /**
     * @brief Set NMS threshold
     */
    virtual void set_nms_threshold(float threshold) = 0;

    /**
     * @brief Get current configuration
     */
    virtual const DetectionConfig& config() const = 0;

    /**
     * @brief Get detection statistics
     */
    virtual DetectionStats get_stats() const = 0;

    /**
     * @brief Get class names
     */
    virtual const std::vector<std::string>& class_names() const = 0;

    /**
     * @brief Get inference backend in use
     */
    virtual InferenceBackend backend() const = 0;
};

/**
 * @brief Create detector instance based on configuration
 * 
 * Factory function that creates the appropriate detection backend
 * based on platform and configuration.
 * 
 * @param config Full system configuration
 * @return Detector instance or nullptr on failure
 */
std::unique_ptr<IDetector> create_detector(const Config& config);

/**
 * @brief Create detector instance with explicit configuration
 * 
 * @param detection_config Detection-specific configuration
 * @return Detector instance or nullptr on failure
 */
std::unique_ptr<IDetector> create_detector(const DetectionConfig& detection_config);

/**
 * @brief Check which backends are available on this platform
 * 
 * @return Vector of available backends
 */
std::vector<InferenceBackend> available_backends();

}  // namespace lagari
