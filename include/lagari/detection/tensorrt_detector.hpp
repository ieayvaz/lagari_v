#pragma once

#include "lagari/detection/yolo_processor.hpp"

#include <memory>
#include <mutex>

namespace lagari {

/**
 * @brief TensorRT-based YOLO detector
 * 
 * High-performance inference using NVIDIA TensorRT.
 * Supports FP16 and INT8 precision, and DLA acceleration on Jetson.
 * 
 * Model formats:
 * - .engine: Pre-built TensorRT engine (fastest startup)
 * - .onnx: Will be converted to engine on first run
 * 
 * Requirements:
 * - CUDA Toolkit
 * - TensorRT 8.x+
 */
class TensorRTDetector : public YOLODetectorBase {
public:
    explicit TensorRTDetector(const DetectionConfig& config);
    ~TensorRTDetector() override;

    // Non-copyable
    TensorRTDetector(const TensorRTDetector&) = delete;
    TensorRTDetector& operator=(const TensorRTDetector&) = delete;

    // IModule interface
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override { return running_; }
    std::string name() const override { return "TensorRTDetector"; }

    // IDetector interface
    bool load_model(const std::string& model_path) override;
    InferenceBackend backend() const override { return InferenceBackend::TENSORRT; }

protected:
    // YOLODetectorBase interface
    bool infer(const float* input, float* output) override;
    std::vector<int> get_output_shape() const override;
    bool allocate_buffers() override;

private:
    // TensorRT engine building
    bool build_engine_from_onnx(const std::string& onnx_path);
    bool load_engine(const std::string& engine_path);
    bool save_engine(const std::string& engine_path);

    // TensorRT state (PIMPL to avoid TensorRT headers in interface)
    struct TRTState;
    std::unique_ptr<TRTState> trt_;

    bool running_ = false;
    std::string model_path_;

    // Output shape cache
    std::vector<int> output_shape_;
};

}  // namespace lagari
