#pragma once

#include "lagari/detection/yolo_processor.hpp"

#include <memory>

namespace lagari {

/**
 * @brief OpenVINO-based YOLO detector
 * 
 * CPU-optimized inference using Intel OpenVINO toolkit.
 * Supports x86/x64 CPUs with AVX/AVX2/AVX512 acceleration.
 * 
 * Model formats:
 * - .onnx: ONNX model (converted automatically)
 * - .xml: OpenVINO IR format (with .bin weights)
 * 
 * Requirements:
 * - OpenVINO Runtime 2022.1+
 */
class OpenVINODetector : public YOLODetectorBase {
public:
    explicit OpenVINODetector(const DetectionConfig& config);
    ~OpenVINODetector() override;

    // Non-copyable
    OpenVINODetector(const OpenVINODetector&) = delete;
    OpenVINODetector& operator=(const OpenVINODetector&) = delete;

    // IModule interface
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override { return running_; }
    std::string name() const override { return "OpenVINODetector"; }

    // IDetector interface
    bool load_model(const std::string& model_path) override;
    InferenceBackend backend() const override { return InferenceBackend::OPENVINO; }

    /**
     * @brief Get available devices
     */
    static std::vector<std::string> available_devices();

    /**
     * @brief Set device for inference
     * @param device Device name (CPU, GPU, AUTO, etc.)
     */
    void set_device(const std::string& device);

protected:
    // YOLODetectorBase interface
    bool infer(const float* input, float* output) override;
    std::vector<int> get_output_shape() const override;
    bool allocate_buffers() override;

private:
    // OpenVINO state (PIMPL to avoid OpenVINO headers in interface)
    struct OVState;
    std::unique_ptr<OVState> ov_;

    bool running_ = false;
    std::string device_ = "CPU";
    std::vector<int> output_shape_;
};

}  // namespace lagari
