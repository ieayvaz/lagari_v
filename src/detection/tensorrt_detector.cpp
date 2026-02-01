#include "lagari/detection/tensorrt_detector.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#ifdef HAS_TENSORRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <filesystem>

namespace lagari {

// ============================================================================
// TensorRT Logger
// ============================================================================

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                LOG_ERROR("TensorRT: {}", msg);
                break;
            case Severity::kWARNING:
                LOG_WARN("TensorRT: {}", msg);
                break;
            case Severity::kINFO:
                LOG_DEBUG("TensorRT: {}", msg);
                break;
            default:
                break;
        }
    }
};

// ============================================================================
// TensorRT State (PIMPL)
// ============================================================================

struct TensorRTDetector::TRTState {
    TRTLogger logger;
    
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    // Device buffers
    void* input_device = nullptr;
    void* output_device = nullptr;
    size_t input_size = 0;
    size_t output_size = 0;
    
    // Binding info
    int input_binding = -1;
    int output_binding = -1;
    
    cudaStream_t stream = nullptr;
    
    ~TRTState() {
        if (stream) cudaStreamDestroy(stream);
        if (input_device) cudaFree(input_device);
        if (output_device) cudaFree(output_device);
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
    }
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

TensorRTDetector::TensorRTDetector(const DetectionConfig& config)
    : YOLODetectorBase(config)
    , trt_(std::make_unique<TRTState>())
{
}

TensorRTDetector::~TensorRTDetector() {
    stop();
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool TensorRTDetector::initialize(const Config& config) {
    // Load labels
    std::string labels_path = config.get_string("detection.labels_path", "");
    if (!labels_path.empty()) {
        processor_.load_labels(labels_path);
    }

    // Load model
    std::string model_path = config.get_string("detection.model_path", "");
    if (model_path.empty()) {
        LOG_ERROR("TensorRTDetector: No model path specified");
        return false;
    }

    if (!load_model(model_path)) {
        return false;
    }

    // Allocate buffers
    if (!allocate_buffers()) {
        return false;
    }

    LOG_INFO("TensorRTDetector: Initialized with model: {}", model_path);
    return true;
}

void TensorRTDetector::start() {
    running_ = true;
}

void TensorRTDetector::stop() {
    running_ = false;
}

// ============================================================================
// Model Loading
// ============================================================================

bool TensorRTDetector::load_model(const std::string& model_path) {
    model_path_ = model_path;

    std::filesystem::path path(model_path);
    std::string ext = path.extension().string();

    bool success = false;
    
    if (ext == ".engine" || ext == ".trt") {
        // Load pre-built engine
        success = load_engine(model_path);
    } else if (ext == ".onnx") {
        // Build engine from ONNX
        std::string engine_path = path.replace_extension(".engine").string();
        
        // Check if cached engine exists
        if (std::filesystem::exists(engine_path)) {
            LOG_INFO("TensorRTDetector: Found cached engine: {}", engine_path);
            success = load_engine(engine_path);
        }
        
        if (!success) {
            LOG_INFO("TensorRTDetector: Building engine from ONNX: {}", model_path);
            success = build_engine_from_onnx(model_path);
            
            if (success) {
                save_engine(engine_path);
            }
        }
    } else {
        LOG_ERROR("TensorRTDetector: Unsupported model format: {}", ext);
        return false;
    }

    if (!success) {
        return false;
    }

    // Get binding info
    trt_->input_binding = trt_->engine->getBindingIndex("images");
    if (trt_->input_binding < 0) {
        // Try alternative names
        trt_->input_binding = trt_->engine->getBindingIndex("input");
    }
    if (trt_->input_binding < 0) {
        trt_->input_binding = 0;  // Assume first binding is input
    }

    trt_->output_binding = trt_->engine->getBindingIndex("output0");
    if (trt_->output_binding < 0) {
        trt_->output_binding = trt_->engine->getBindingIndex("output");
    }
    if (trt_->output_binding < 0) {
        trt_->output_binding = 1;  // Assume second binding is output
    }

    // Get output shape
    auto output_dims = trt_->engine->getBindingDimensions(trt_->output_binding);
    output_shape_.clear();
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_shape_.push_back(output_dims.d[i]);
    }

    // Create execution context
    trt_->context = trt_->engine->createExecutionContext();
    if (!trt_->context) {
        LOG_ERROR("TensorRTDetector: Failed to create execution context");
        return false;
    }

    // Create CUDA stream
    cudaStreamCreate(&trt_->stream);

    LOG_INFO("TensorRTDetector: Model loaded, output shape: [{}, {}, {}]",
             output_shape_.size() > 0 ? output_shape_[0] : 0,
             output_shape_.size() > 1 ? output_shape_[1] : 0,
             output_shape_.size() > 2 ? output_shape_[2] : 0);

    return true;
}

bool TensorRTDetector::build_engine_from_onnx(const std::string& onnx_path) {
    // Create builder
    auto builder = nvinfer1::createInferBuilder(trt_->logger);
    if (!builder) {
        LOG_ERROR("TensorRTDetector: Failed to create builder");
        return false;
    }

    // Create network with explicit batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    if (!network) {
        LOG_ERROR("TensorRTDetector: Failed to create network");
        delete builder;
        return false;
    }

    // Create ONNX parser
    auto parser = nvonnxparser::createParser(*network, trt_->logger);
    if (!parser) {
        LOG_ERROR("TensorRTDetector: Failed to create ONNX parser");
        delete network;
        delete builder;
        return false;
    }

    // Parse ONNX model
    if (!parser->parseFromFile(onnx_path.c_str(), 
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        LOG_ERROR("TensorRTDetector: Failed to parse ONNX file");
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Create builder config
    auto config = builder->createBuilderConfig();
    if (!config) {
        LOG_ERROR("TensorRTDetector: Failed to create builder config");
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Set memory pool limit (using newer API)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 1GB

    // Set precision
    if (config_.fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        LOG_INFO("TensorRTDetector: Using FP16 precision");
    }
    
    if (config_.int8 && builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        LOG_INFO("TensorRTDetector: Using INT8 precision");
    }

    // DLA support (Jetson)
    if (config_.dla_core >= 0) {
        if (builder->getNbDLACores() > config_.dla_core) {
            config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            config->setDLACore(config_.dla_core);
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
            LOG_INFO("TensorRTDetector: Using DLA core {}", config_.dla_core);
        } else {
            LOG_WARN("TensorRTDetector: DLA core {} not available", config_.dla_core);
        }
    }

    // Build serialized network
    auto serialized = builder->buildSerializedNetwork(*network, *config);
    if (!serialized) {
        LOG_ERROR("TensorRTDetector: Failed to build serialized network");
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Create runtime
    trt_->runtime = nvinfer1::createInferRuntime(trt_->logger);
    if (!trt_->runtime) {
        LOG_ERROR("TensorRTDetector: Failed to create runtime");
        delete serialized;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Deserialize engine
    trt_->engine = trt_->runtime->deserializeCudaEngine(
        serialized->data(), serialized->size());
    
    delete serialized;
    delete config;
    delete parser;
    delete network;
    delete builder;

    if (!trt_->engine) {
        LOG_ERROR("TensorRTDetector: Failed to deserialize engine");
        return false;
    }

    return true;
}

bool TensorRTDetector::load_engine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        LOG_ERROR("TensorRTDetector: Failed to open engine file: {}", engine_path);
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    // Create runtime
    trt_->runtime = nvinfer1::createInferRuntime(trt_->logger);
    if (!trt_->runtime) {
        LOG_ERROR("TensorRTDetector: Failed to create runtime");
        return false;
    }

    // Deserialize engine
    trt_->engine = trt_->runtime->deserializeCudaEngine(buffer.data(), size);
    if (!trt_->engine) {
        LOG_ERROR("TensorRTDetector: Failed to deserialize engine");
        return false;
    }

    return true;
}

bool TensorRTDetector::save_engine(const std::string& engine_path) {
    auto serialized = trt_->engine->serialize();
    if (!serialized) {
        LOG_WARN("TensorRTDetector: Failed to serialize engine");
        return false;
    }

    std::ofstream file(engine_path, std::ios::binary);
    if (!file) {
        LOG_WARN("TensorRTDetector: Failed to save engine: {}", engine_path);
        delete serialized;
        return false;
    }

    file.write(static_cast<const char*>(serialized->data()), serialized->size());
    delete serialized;

    LOG_INFO("TensorRTDetector: Saved engine to: {}", engine_path);
    return true;
}

// ============================================================================
// Buffer Allocation
// ============================================================================

bool TensorRTDetector::allocate_buffers() {
    if (!trt_->engine) {
        return false;
    }

    // Get input size
    auto input_dims = trt_->engine->getBindingDimensions(trt_->input_binding);
    trt_->input_size = 1;
    for (int i = 0; i < input_dims.nbDims; ++i) {
        trt_->input_size *= input_dims.d[i];
    }
    trt_->input_size *= sizeof(float);

    // Get output size
    auto output_dims = trt_->engine->getBindingDimensions(trt_->output_binding);
    trt_->output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        trt_->output_size *= output_dims.d[i];
    }
    trt_->output_size *= sizeof(float);

    // Allocate device memory
    cudaMalloc(&trt_->input_device, trt_->input_size);
    cudaMalloc(&trt_->output_device, trt_->output_size);

    // Allocate host buffers
    input_buffer_.resize(trt_->input_size / sizeof(float));
    output_buffer_.resize(trt_->output_size / sizeof(float));

    LOG_DEBUG("TensorRTDetector: Allocated buffers - input: {} bytes, output: {} bytes",
              trt_->input_size, trt_->output_size);

    return true;
}

// ============================================================================
// Inference
// ============================================================================

bool TensorRTDetector::infer(const float* input, float* output) {
    if (!trt_->context) {
        return false;
    }

    // Copy input to device
    cudaMemcpyAsync(trt_->input_device, input, trt_->input_size,
                    cudaMemcpyHostToDevice, trt_->stream);

    // Run inference
    void* bindings[] = {trt_->input_device, trt_->output_device};
    bool success = trt_->context->enqueueV2(bindings, trt_->stream, nullptr);

    if (!success) {
        LOG_ERROR("TensorRTDetector: Inference failed");
        return false;
    }

    // Copy output to host
    cudaMemcpyAsync(output, trt_->output_device, trt_->output_size,
                    cudaMemcpyDeviceToHost, trt_->stream);

    // Synchronize
    cudaStreamSynchronize(trt_->stream);

    return true;
}

std::vector<int> TensorRTDetector::get_output_shape() const {
    return output_shape_;
}

}  // namespace lagari

#else  // !HAS_TENSORRT

namespace lagari {

// Stub implementation
struct TensorRTDetector::TRTState {};

TensorRTDetector::TensorRTDetector(const DetectionConfig& config) : YOLODetectorBase(config) {}
TensorRTDetector::~TensorRTDetector() = default;

bool TensorRTDetector::initialize(const Config&) {
    LOG_ERROR("TensorRTDetector: Not available (compile with HAS_TENSORRT)");
    return false;
}

void TensorRTDetector::start() {}
void TensorRTDetector::stop() {}
bool TensorRTDetector::load_model(const std::string&) { return false; }
bool TensorRTDetector::infer(const float*, float*) { return false; }
std::vector<int> TensorRTDetector::get_output_shape() const { return {}; }
bool TensorRTDetector::allocate_buffers() { return false; }
bool TensorRTDetector::build_engine_from_onnx(const std::string&) { return false; }
bool TensorRTDetector::load_engine(const std::string&) { return false; }
bool TensorRTDetector::save_engine(const std::string&) { return false; }

}  // namespace lagari

#endif  // HAS_TENSORRT
