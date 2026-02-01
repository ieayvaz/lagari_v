#pragma once

#include "lagari/detection/yolo_processor.hpp"

#include <memory>

namespace lagari {

/**
 * @brief NCNN-based YOLO detector
 * 
 * Lightweight inference using Tencent NCNN framework.
 * Optimized for ARM CPUs (Raspberry Pi, mobile devices).
 * 
 * Model formats:
 * - .param + .bin: NCNN native format
 * - .onnx: Can be converted using ncnn tools
 * 
 * Features:
 * - ARM NEON optimization
 * - Vulkan GPU acceleration (optional)
 * - Low memory footprint
 * 
 * Requirements:
 * - NCNN library (https://github.com/Tencent/ncnn)
 */
class NCNNDetector : public YOLODetectorBase {
public:
    explicit NCNNDetector(const DetectionConfig& config);
    ~NCNNDetector() override;

    // Non-copyable
    NCNNDetector(const NCNNDetector&) = delete;
    NCNNDetector& operator=(const NCNNDetector&) = delete;

    // IModule interface
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override { return running_; }
    std::string name() const override { return "NCNNDetector"; }

    // IDetector interface
    bool load_model(const std::string& model_path) override;
    InferenceBackend backend() const override { return InferenceBackend::NCNN; }

    /**
     * @brief Enable/disable Vulkan GPU acceleration
     * @param enable true to enable GPU
     * @return true if GPU was enabled successfully
     */
    bool set_vulkan(bool enable);

    /**
     * @brief Set number of threads
     */
    void set_num_threads(int threads);

protected:
    // YOLODetectorBase interface
    bool infer(const float* input, float* output) override;
    std::vector<int> get_output_shape() const override;
    bool allocate_buffers() override;

private:
    // NCNN state (PIMPL to avoid NCNN headers in interface)
    struct NCNNState;
    std::unique_ptr<NCNNState> ncnn_;

    bool running_ = false;
    int num_threads_ = 4;
    bool use_vulkan_ = false;
    std::vector<int> output_shape_;
};

}  // namespace lagari
