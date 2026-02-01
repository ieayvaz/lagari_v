#pragma once

#include "lagari/detection/yolo_processor.hpp"

#include <memory>

namespace lagari {

/**
 * @brief ONNX Runtime-based YOLO detector
 * 
 * Cross-platform inference using Microsoft ONNX Runtime.
 * Works on any platform with CPU, and supports GPU acceleration
 * via CUDA, TensorRT, DirectML, or CoreML execution providers.
 * 
 * Model formats:
 * - .onnx: ONNX model format
 * 
 * Features:
 * - Cross-platform compatibility
 * - Multiple execution providers (CPU, CUDA, TensorRT, etc.)
 * - Easy deployment
 * 
 * Requirements:
 * - ONNX Runtime library (https://onnxruntime.ai)
 */
class ONNXRuntimeDetector : public YOLODetectorBase {
public:
    /**
     * @brief Execution provider type
     */
    enum class ExecutionProvider {
        CPU,        // Default CPU provider
        CUDA,       // NVIDIA CUDA
        TENSORRT,   // NVIDIA TensorRT (via ONNX Runtime)
        DIRECTML,   // Windows DirectX ML
        COREML,     // Apple CoreML
        OPENVINO    // Intel OpenVINO (via ONNX Runtime)
    };

    explicit ONNXRuntimeDetector(const DetectionConfig& config);
    ~ONNXRuntimeDetector() override;

    // Non-copyable
    ONNXRuntimeDetector(const ONNXRuntimeDetector&) = delete;
    ONNXRuntimeDetector& operator=(const ONNXRuntimeDetector&) = delete;

    // IModule interface
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override { return running_; }
    std::string name() const override { return "ONNXRuntimeDetector"; }

    // IDetector interface
    bool load_model(const std::string& model_path) override;
    InferenceBackend backend() const override { return InferenceBackend::ONNXRUNTIME; }

    /**
     * @brief Set execution provider
     * @param provider Execution provider to use
     * @param device_id Device ID (for CUDA/TensorRT)
     * @return true if provider was set successfully
     */
    bool set_execution_provider(ExecutionProvider provider, int device_id = 0);

    /**
     * @brief Get available execution providers
     */
    static std::vector<ExecutionProvider> available_providers();

protected:
    // YOLODetectorBase interface
    bool infer(const float* input, float* output) override;
    std::vector<int> get_output_shape() const override;
    bool allocate_buffers() override;

private:
    // ONNX Runtime state (PIMPL to avoid ORT headers in interface)
    struct ORTState;
    std::unique_ptr<ORTState> ort_;

    bool running_ = false;
    ExecutionProvider provider_ = ExecutionProvider::CPU;
    int device_id_ = 0;
    std::vector<int> output_shape_;
};

}  // namespace lagari
