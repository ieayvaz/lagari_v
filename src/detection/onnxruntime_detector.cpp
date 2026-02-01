#include "lagari/detection/onnxruntime_detector.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#ifdef HAS_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>

#include <cstring>

namespace lagari {

// ============================================================================
// ONNX Runtime State (PIMPL)
// ============================================================================

struct ONNXRuntimeDetector::ORTState {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "LagariDetector"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    
    bool loaded = false;
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

ONNXRuntimeDetector::ONNXRuntimeDetector(const DetectionConfig& config)
    : YOLODetectorBase(config)
    , ort_(std::make_unique<ORTState>())
{
}

ONNXRuntimeDetector::~ONNXRuntimeDetector() {
    stop();
}

// ============================================================================
// Static Methods
// ============================================================================

std::vector<ONNXRuntimeDetector::ExecutionProvider> ONNXRuntimeDetector::available_providers() {
    std::vector<ExecutionProvider> providers;
    providers.push_back(ExecutionProvider::CPU);  // Always available
    
    auto available = Ort::GetAvailableProviders();
    for (const auto& p : available) {
        if (p == "CUDAExecutionProvider") {
            providers.push_back(ExecutionProvider::CUDA);
        } else if (p == "TensorrtExecutionProvider") {
            providers.push_back(ExecutionProvider::TENSORRT);
        } else if (p == "DmlExecutionProvider") {
            providers.push_back(ExecutionProvider::DIRECTML);
        } else if (p == "CoreMLExecutionProvider") {
            providers.push_back(ExecutionProvider::COREML);
        } else if (p == "OpenVINOExecutionProvider") {
            providers.push_back(ExecutionProvider::OPENVINO);
        }
    }
    
    return providers;
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool ONNXRuntimeDetector::initialize(const Config& config) {
    // Get execution provider
    std::string provider_str = config.get_string("detection.onnxruntime.provider", "cpu");
    if (provider_str == "cuda") provider_ = ExecutionProvider::CUDA;
    else if (provider_str == "tensorrt") provider_ = ExecutionProvider::TENSORRT;
    else if (provider_str == "directml") provider_ = ExecutionProvider::DIRECTML;
    else if (provider_str == "coreml") provider_ = ExecutionProvider::COREML;
    else if (provider_str == "openvino") provider_ = ExecutionProvider::OPENVINO;
    else provider_ = ExecutionProvider::CPU;
    
    device_id_ = config.get_int("detection.onnxruntime.device_id", 0);

    // Load labels
    std::string labels_path = config.get_string("detection.labels_path", "");
    if (!labels_path.empty()) {
        processor_.load_labels(labels_path);
    }

    // Load model
    std::string model_path = config.get_string("detection.model_path", "");
    if (model_path.empty()) {
        LOG_ERROR("ONNXRuntimeDetector: No model path specified");
        return false;
    }

    if (!load_model(model_path)) {
        return false;
    }

    // Allocate buffers
    if (!allocate_buffers()) {
        return false;
    }

    LOG_INFO("ONNXRuntimeDetector: Initialized with model: {}", model_path);
    return true;
}

void ONNXRuntimeDetector::start() {
    running_ = true;
}

void ONNXRuntimeDetector::stop() {
    running_ = false;
}

bool ONNXRuntimeDetector::set_execution_provider(ExecutionProvider provider, int device_id) {
    provider_ = provider;
    device_id_ = device_id;
    return true;
}

// ============================================================================
// Model Loading
// ============================================================================

bool ONNXRuntimeDetector::load_model(const std::string& model_path) {
    try {
        // Configure session options
        ort_->session_options.SetIntraOpNumThreads(4);
        ort_->session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Set execution provider
        switch (provider_) {
            case ExecutionProvider::CUDA:
#ifdef USE_CUDA
                {
                    OrtCUDAProviderOptions cuda_options;
                    cuda_options.device_id = device_id_;
                    ort_->session_options.AppendExecutionProvider_CUDA(cuda_options);
                    LOG_INFO("ONNXRuntimeDetector: Using CUDA execution provider");
                }
#else
                LOG_WARN("ONNXRuntimeDetector: CUDA not available, falling back to CPU");
#endif
                break;
                
            case ExecutionProvider::TENSORRT:
#ifdef USE_TENSORRT
                {
                    OrtTensorRTProviderOptions trt_options;
                    trt_options.device_id = device_id_;
                    trt_options.trt_fp16_enable = config_.fp16 ? 1 : 0;
                    ort_->session_options.AppendExecutionProvider_TensorRT(trt_options);
                    LOG_INFO("ONNXRuntimeDetector: Using TensorRT execution provider");
                }
#else
                LOG_WARN("ONNXRuntimeDetector: TensorRT not available, falling back to CPU");
#endif
                break;
                
            case ExecutionProvider::CPU:
            default:
                LOG_INFO("ONNXRuntimeDetector: Using CPU execution provider");
                break;
        }

        // Load model
        LOG_INFO("ONNXRuntimeDetector: Loading model: {}", model_path);
        ort_->session = std::make_unique<Ort::Session>(
            ort_->env, model_path.c_str(), ort_->session_options);

        // Get input info
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = ort_->session->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = ort_->session->GetInputNameAllocated(i, allocator);
            ort_->input_names_str.push_back(name.get());
            
            auto type_info = ort_->session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            if (i == 0) {
                ort_->input_shape = tensor_info.GetShape();
            }
        }

        // Get output info
        size_t num_outputs = ort_->session->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = ort_->session->GetOutputNameAllocated(i, allocator);
            ort_->output_names_str.push_back(name.get());
            
            auto type_info = ort_->session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            if (i == 0) {
                ort_->output_shape = tensor_info.GetShape();
            }
        }

        // Build name pointers
        for (const auto& name : ort_->input_names_str) {
            ort_->input_names.push_back(name.c_str());
        }
        for (const auto& name : ort_->output_names_str) {
            ort_->output_names.push_back(name.c_str());
        }

        // Store output shape for postprocessing
        output_shape_.clear();
        for (auto dim : ort_->output_shape) {
            output_shape_.push_back(static_cast<int>(dim));
        }

        LOG_DEBUG("ONNXRuntimeDetector: Input shape: [{}, {}, {}, {}]",
                  ort_->input_shape.size() > 0 ? ort_->input_shape[0] : 0,
                  ort_->input_shape.size() > 1 ? ort_->input_shape[1] : 0,
                  ort_->input_shape.size() > 2 ? ort_->input_shape[2] : 0,
                  ort_->input_shape.size() > 3 ? ort_->input_shape[3] : 0);

        LOG_DEBUG("ONNXRuntimeDetector: Output shape: [{}, {}, {}]",
                  output_shape_.size() > 0 ? output_shape_[0] : 0,
                  output_shape_.size() > 1 ? output_shape_[1] : 0,
                  output_shape_.size() > 2 ? output_shape_[2] : 0);

        ort_->loaded = true;
        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNXRuntimeDetector: Failed to load model: {}", e.what());
        return false;
    }
}

// ============================================================================
// Buffer Allocation
// ============================================================================

bool ONNXRuntimeDetector::allocate_buffers() {
    if (!ort_->loaded) {
        return false;
    }

    try {
        // Calculate input size
        size_t input_size = 1;
        for (auto dim : ort_->input_shape) {
            if (dim > 0) input_size *= dim;
        }
        // Handle dynamic batch dimension
        if (ort_->input_shape.size() > 0 && ort_->input_shape[0] <= 0) {
            ort_->input_shape[0] = 1;  // Set batch to 1
        }

        // Calculate output size
        size_t output_size = 1;
        for (auto dim : ort_->output_shape) {
            if (dim > 0) output_size *= dim;
        }
        if (ort_->output_shape.size() > 0 && ort_->output_shape[0] <= 0) {
            ort_->output_shape[0] = 1;
        }

        input_buffer_.resize(input_size);
        output_buffer_.resize(output_size > 0 ? output_size : 1000000);  // Fallback size

        LOG_DEBUG("ONNXRuntimeDetector: Allocated buffers - input: {}, output: {}",
                  input_buffer_.size(), output_buffer_.size());

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("ONNXRuntimeDetector: Failed to allocate buffers: {}", e.what());
        return false;
    }
}

// ============================================================================
// Inference
// ============================================================================

bool ONNXRuntimeDetector::infer(const float* input, float* output) {
    if (!ort_->loaded || !ort_->session) {
        return false;
    }

    try {
        // Create input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            ort_->memory_info,
            const_cast<float*>(input),
            input_buffer_.size(),
            ort_->input_shape.data(),
            ort_->input_shape.size());

        // Run inference
        auto output_tensors = ort_->session->Run(
            Ort::RunOptions{nullptr},
            ort_->input_names.data(),
            &input_tensor,
            1,
            ort_->output_names.data(),
            ort_->output_names.size());

        // Copy output
        if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
            const float* output_data = output_tensors[0].GetTensorData<float>();
            auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            size_t output_size = tensor_info.GetElementCount();
            
            std::memcpy(output, output_data, output_size * sizeof(float));
            
            // Update output shape if dynamic
            auto shape = tensor_info.GetShape();
            output_shape_.clear();
            for (auto dim : shape) {
                output_shape_.push_back(static_cast<int>(dim));
            }
        }

        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNXRuntimeDetector: Inference failed: {}", e.what());
        return false;
    }
}

std::vector<int> ONNXRuntimeDetector::get_output_shape() const {
    return output_shape_;
}

}  // namespace lagari

#else  // !HAS_ONNXRUNTIME

namespace lagari {

// Stub implementation
struct ONNXRuntimeDetector::ORTState {};

ONNXRuntimeDetector::ONNXRuntimeDetector(const DetectionConfig& config) 
    : YOLODetectorBase(config) {}
ONNXRuntimeDetector::~ONNXRuntimeDetector() = default;

std::vector<ONNXRuntimeDetector::ExecutionProvider> ONNXRuntimeDetector::available_providers() {
    return {};
}

bool ONNXRuntimeDetector::initialize(const Config&) {
    LOG_ERROR("ONNXRuntimeDetector: Not available (compile with HAS_ONNXRUNTIME)");
    return false;
}

void ONNXRuntimeDetector::start() {}
void ONNXRuntimeDetector::stop() {}
bool ONNXRuntimeDetector::set_execution_provider(ExecutionProvider, int) { return false; }
bool ONNXRuntimeDetector::load_model(const std::string&) { return false; }
bool ONNXRuntimeDetector::infer(const float*, float*) { return false; }
std::vector<int> ONNXRuntimeDetector::get_output_shape() const { return {}; }
bool ONNXRuntimeDetector::allocate_buffers() { return false; }

}  // namespace lagari

#endif  // HAS_ONNXRUNTIME
