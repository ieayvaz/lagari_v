#include "lagari/detection/openvino_detector.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#ifdef HAS_OPENVINO

#include <openvino/openvino.hpp>

namespace lagari {

// ============================================================================
// OpenVINO State (PIMPL)
// ============================================================================

struct OpenVINODetector::OVState {
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    
    std::string input_name;
    std::string output_name;
    
    ov::Shape input_shape;
    ov::Shape output_shape;
    
    bool loaded = false;
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

OpenVINODetector::OpenVINODetector(const DetectionConfig& config)
    : YOLODetectorBase(config)
    , ov_(std::make_unique<OVState>())
{
}

OpenVINODetector::~OpenVINODetector() {
    stop();
}

// ============================================================================
// Static Methods
// ============================================================================

std::vector<std::string> OpenVINODetector::available_devices() {
    try {
        ov::Core core;
        return core.get_available_devices();
    } catch (const std::exception& e) {
        LOG_ERROR("OpenVINODetector: Failed to get devices: {}", e.what());
        return {};
    }
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool OpenVINODetector::initialize(const Config& config) {
    // Get device
    device_ = config.get_string("detection.openvino.device", "CPU");
    
    // Load labels
    std::string labels_path = config.get_string("detection.labels_path", "");
    if (!labels_path.empty()) {
        processor_.load_labels(labels_path);
    }

    // Load model
    std::string model_path = config.get_string("detection.model_path", "");
    if (model_path.empty()) {
        LOG_ERROR("OpenVINODetector: No model path specified");
        return false;
    }

    if (!load_model(model_path)) {
        return false;
    }

    // Allocate buffers
    if (!allocate_buffers()) {
        return false;
    }

    LOG_INFO("OpenVINODetector: Initialized with model: {}", model_path);
    return true;
}

void OpenVINODetector::start() {
    running_ = true;
}

void OpenVINODetector::stop() {
    running_ = false;
}

void OpenVINODetector::set_device(const std::string& device) {
    device_ = device;
}

// ============================================================================
// Model Loading
// ============================================================================

bool OpenVINODetector::load_model(const std::string& model_path) {
    try {
        // Read model
        LOG_INFO("OpenVINODetector: Loading model: {}", model_path);
        ov_->model = ov_->core.read_model(model_path);
        
        if (!ov_->model) {
            LOG_ERROR("OpenVINODetector: Failed to read model");
            return false;
        }

        // Get input/output info
        auto inputs = ov_->model->inputs();
        auto outputs = ov_->model->outputs();
        
        if (inputs.empty() || outputs.empty()) {
            LOG_ERROR("OpenVINODetector: Model has no inputs/outputs");
            return false;
        }

        ov_->input_name = inputs[0].get_any_name();
        ov_->output_name = outputs[0].get_any_name();
        ov_->input_shape = inputs[0].get_shape();
        ov_->output_shape = outputs[0].get_shape();

        LOG_DEBUG("OpenVINODetector: Input: {} [{}, {}, {}, {}]",
                  ov_->input_name,
                  ov_->input_shape.size() > 0 ? ov_->input_shape[0] : 0,
                  ov_->input_shape.size() > 1 ? ov_->input_shape[1] : 0,
                  ov_->input_shape.size() > 2 ? ov_->input_shape[2] : 0,
                  ov_->input_shape.size() > 3 ? ov_->input_shape[3] : 0);

        LOG_DEBUG("OpenVINODetector: Output: {} [{}, {}, {}]",
                  ov_->output_name,
                  ov_->output_shape.size() > 0 ? ov_->output_shape[0] : 0,
                  ov_->output_shape.size() > 1 ? ov_->output_shape[1] : 0,
                  ov_->output_shape.size() > 2 ? ov_->output_shape[2] : 0);

        // Compile model
        LOG_INFO("OpenVINODetector: Compiling model for device: {}", device_);
        
        ov::AnyMap config_map;
        
        // Performance hints
        config_map[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        config_map[ov::hint::num_requests.name()] = 1;
        
        // CPU-specific optimizations
        if (device_ == "CPU") {
            config_map[ov::inference_num_threads.name()] = 0;  // Auto
            config_map[ov::enable_profiling.name()] = false;
        }

        ov_->compiled_model = ov_->core.compile_model(ov_->model, device_, config_map);
        ov_->infer_request = ov_->compiled_model.create_infer_request();

        ov_->loaded = true;

        // Store output shape for postprocessing
        output_shape_.clear();
        for (auto dim : ov_->output_shape) {
            output_shape_.push_back(static_cast<int>(dim));
        }

        LOG_INFO("OpenVINODetector: Model compiled successfully");
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("OpenVINODetector: Failed to load model: {}", e.what());
        return false;
    }
}

// ============================================================================
// Buffer Allocation
// ============================================================================

bool OpenVINODetector::allocate_buffers() {
    if (!ov_->loaded) {
        return false;
    }

    try {
        // Calculate sizes
        size_t input_size = 1;
        for (auto dim : ov_->input_shape) {
            input_size *= dim;
        }

        size_t output_size = 1;
        for (auto dim : ov_->output_shape) {
            output_size *= dim;
        }

        input_buffer_.resize(input_size);
        output_buffer_.resize(output_size);

        LOG_DEBUG("OpenVINODetector: Allocated buffers - input: {}, output: {}",
                  input_size, output_size);

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("OpenVINODetector: Failed to allocate buffers: {}", e.what());
        return false;
    }
}

// ============================================================================
// Inference
// ============================================================================

bool OpenVINODetector::infer(const float* input, float* output) {
    if (!ov_->loaded) {
        return false;
    }

    try {
        // Create input tensor from data
        ov::Tensor input_tensor(ov::element::f32, ov_->input_shape, 
                                const_cast<float*>(input));
        ov_->infer_request.set_input_tensor(input_tensor);

        // Run inference
        ov_->infer_request.infer();

        // Get output
        ov::Tensor output_tensor = ov_->infer_request.get_output_tensor();
        const float* output_data = output_tensor.data<float>();
        
        size_t output_size = output_tensor.get_size();
        std::memcpy(output, output_data, output_size * sizeof(float));

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("OpenVINODetector: Inference failed: {}", e.what());
        return false;
    }
}

std::vector<int> OpenVINODetector::get_output_shape() const {
    return output_shape_;
}

}  // namespace lagari

#else  // !HAS_OPENVINO

namespace lagari {

// Stub implementation
struct OpenVINODetector::OVState {};

OpenVINODetector::OpenVINODetector(const DetectionConfig& config) : YOLODetectorBase(config) {}
OpenVINODetector::~OpenVINODetector() = default;

std::vector<std::string> OpenVINODetector::available_devices() { return {}; }

bool OpenVINODetector::initialize(const Config&) {
    LOG_ERROR("OpenVINODetector: Not available (compile with HAS_OPENVINO)");
    return false;
}

void OpenVINODetector::start() {}
void OpenVINODetector::stop() {}
void OpenVINODetector::set_device(const std::string&) {}
bool OpenVINODetector::load_model(const std::string&) { return false; }
bool OpenVINODetector::infer(const float*, float*) { return false; }
std::vector<int> OpenVINODetector::get_output_shape() const { return {}; }
bool OpenVINODetector::allocate_buffers() { return false; }

}  // namespace lagari

#endif  // HAS_OPENVINO
