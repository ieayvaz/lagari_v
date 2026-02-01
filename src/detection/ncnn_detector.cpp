#include "lagari/detection/ncnn_detector.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#ifdef HAS_NCNN

#include <ncnn/net.h>
#include <ncnn/layer.h>

#include <filesystem>

namespace lagari {

// ============================================================================
// NCNN State (PIMPL)
// ============================================================================

struct NCNNDetector::NCNNState {
    ncnn::Net net;
    std::string input_name = "images";
    std::string output_name = "output0";
    
    bool loaded = false;
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

NCNNDetector::NCNNDetector(const DetectionConfig& config)
    : YOLODetectorBase(config)
    , ncnn_(std::make_unique<NCNNState>())
{
}

NCNNDetector::~NCNNDetector() {
    stop();
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool NCNNDetector::initialize(const Config& config) {
    // Get settings
    num_threads_ = config.get_int("detection.ncnn.threads", 4);
    use_vulkan_ = config.get_bool("detection.ncnn.vulkan", false);

    // Load labels
    std::string labels_path = config.get_string("detection.labels_path", "");
    if (!labels_path.empty()) {
        processor_.load_labels(labels_path);
    }

    // Load model
    std::string model_path = config.get_string("detection.model_path", "");
    if (model_path.empty()) {
        LOG_ERROR("NCNNDetector: No model path specified");
        return false;
    }

    if (!load_model(model_path)) {
        return false;
    }

    // Allocate buffers
    if (!allocate_buffers()) {
        return false;
    }

    LOG_INFO("NCNNDetector: Initialized with model: {}", model_path);
    return true;
}

void NCNNDetector::start() {
    running_ = true;
}

void NCNNDetector::stop() {
    running_ = false;
}

bool NCNNDetector::set_vulkan(bool enable) {
#if NCNN_VULKAN
    use_vulkan_ = enable;
    if (ncnn_->loaded) {
        ncnn_->net.opt.use_vulkan_compute = enable;
    }
    return true;
#else
    if (enable) {
        LOG_WARN("NCNNDetector: Vulkan not available in this build");
    }
    return false;
#endif
}

void NCNNDetector::set_num_threads(int threads) {
    num_threads_ = threads;
    if (ncnn_->loaded) {
        ncnn_->net.opt.num_threads = threads;
    }
}

// ============================================================================
// Model Loading
// ============================================================================

bool NCNNDetector::load_model(const std::string& model_path) {
    try {
        std::filesystem::path path(model_path);
        std::string ext = path.extension().string();

        // Configure NCNN options
        ncnn_->net.opt.num_threads = num_threads_;
        ncnn_->net.opt.use_fp16_storage = config_.fp16;
        ncnn_->net.opt.use_fp16_packed = config_.fp16;
        ncnn_->net.opt.use_fp16_arithmetic = config_.fp16;
        
#if NCNN_VULKAN
        ncnn_->net.opt.use_vulkan_compute = use_vulkan_;
        if (use_vulkan_) {
            LOG_INFO("NCNNDetector: Using Vulkan GPU acceleration");
        }
#endif

        // Load model files
        std::string param_path, bin_path;
        
        if (ext == ".param") {
            param_path = model_path;
            bin_path = path.replace_extension(".bin").string();
        } else if (ext == ".bin") {
            bin_path = model_path;
            param_path = path.replace_extension(".param").string();
        } else {
            LOG_ERROR("NCNNDetector: Unsupported format: {}", ext);
            LOG_ERROR("NCNNDetector: Expected .param and .bin files");
            return false;
        }

        LOG_INFO("NCNNDetector: Loading param: {}", param_path);
        LOG_INFO("NCNNDetector: Loading bin: {}", bin_path);

        if (ncnn_->net.load_param(param_path.c_str()) != 0) {
            LOG_ERROR("NCNNDetector: Failed to load param file");
            return false;
        }

        if (ncnn_->net.load_model(bin_path.c_str()) != 0) {
            LOG_ERROR("NCNNDetector: Failed to load bin file");
            return false;
        }

        // Get input/output layer names
        const std::vector<const char*>& input_names = ncnn_->net.input_names();
        const std::vector<const char*>& output_names = ncnn_->net.output_names();

        if (!input_names.empty()) {
            ncnn_->input_name = input_names[0];
        }
        if (!output_names.empty()) {
            ncnn_->output_name = output_names[0];
        }

        LOG_DEBUG("NCNNDetector: Input: {}, Output: {}", 
                  ncnn_->input_name, ncnn_->output_name);

        ncnn_->loaded = true;

        // Determine output shape by doing a dummy inference
        ncnn::Mat dummy_input(config_.input_width, config_.input_height, 3);
        ncnn::Extractor ex = ncnn_->net.create_extractor();
        ex.set_num_threads(num_threads_);
        ex.input(ncnn_->input_name.c_str(), dummy_input);
        
        ncnn::Mat dummy_output;
        ex.extract(ncnn_->output_name.c_str(), dummy_output);

        output_shape_.clear();
        output_shape_.push_back(1);  // batch
        output_shape_.push_back(dummy_output.c);
        output_shape_.push_back(dummy_output.h * dummy_output.w);

        LOG_INFO("NCNNDetector: Output shape: [1, {}, {}]", 
                 dummy_output.c, dummy_output.h * dummy_output.w);

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("NCNNDetector: Failed to load model: {}", e.what());
        return false;
    }
}

// ============================================================================
// Buffer Allocation
// ============================================================================

bool NCNNDetector::allocate_buffers() {
    if (!ncnn_->loaded) {
        return false;
    }

    // NCNN handles its own buffers, but we still need host buffers for I/O
    size_t input_size = config_.input_width * config_.input_height * 3;
    input_buffer_.resize(input_size);

    // Estimate output size based on YOLO output format
    // For YOLOv8: [84, 8400] or similar for 80 classes
    size_t output_size = 1;
    for (int dim : output_shape_) {
        output_size *= dim;
    }
    output_buffer_.resize(output_size);

    LOG_DEBUG("NCNNDetector: Allocated buffers - input: {}, output: {}",
              input_size, output_size);

    return true;
}

// ============================================================================
// Inference
// ============================================================================

bool NCNNDetector::infer(const float* input, float* output) {
    if (!ncnn_->loaded) {
        return false;
    }

    try {
        // Create NCNN Mat from input (CHW format)
        ncnn::Mat in(config_.input_width, config_.input_height, 3);
        
        // Copy input data (already in CHW format from preprocessing)
        const size_t channel_size = config_.input_width * config_.input_height;
        for (int c = 0; c < 3; ++c) {
            float* channel_data = in.channel(c);
            std::memcpy(channel_data, input + c * channel_size, 
                        channel_size * sizeof(float));
        }

        // Create extractor
        ncnn::Extractor ex = ncnn_->net.create_extractor();
        ex.set_num_threads(num_threads_);

#if NCNN_VULKAN
        if (use_vulkan_) {
            ex.set_vulkan_compute(true);
        }
#endif

        // Run inference
        ex.input(ncnn_->input_name.c_str(), in);
        
        ncnn::Mat out;
        if (ex.extract(ncnn_->output_name.c_str(), out) != 0) {
            LOG_ERROR("NCNNDetector: Inference failed");
            return false;
        }

        // Copy output to flat buffer
        // NCNN Mat is usually in [C, H, W] or [C, N] format
        size_t output_size = out.total();
        std::memcpy(output, out.data, output_size * sizeof(float));

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("NCNNDetector: Inference error: {}", e.what());
        return false;
    }
}

std::vector<int> NCNNDetector::get_output_shape() const {
    return output_shape_;
}

}  // namespace lagari

#else  // !HAS_NCNN

namespace lagari {

// Stub implementation
struct NCNNDetector::NCNNState {};

NCNNDetector::NCNNDetector(const DetectionConfig& config) : YOLODetectorBase(config) {}
NCNNDetector::~NCNNDetector() = default;

bool NCNNDetector::initialize(const Config&) {
    LOG_ERROR("NCNNDetector: Not available (compile with HAS_NCNN)");
    return false;
}

void NCNNDetector::start() {}
void NCNNDetector::stop() {}
bool NCNNDetector::set_vulkan(bool) { return false; }
void NCNNDetector::set_num_threads(int) {}
bool NCNNDetector::load_model(const std::string&) { return false; }
bool NCNNDetector::infer(const float*, float*) { return false; }
std::vector<int> NCNNDetector::get_output_shape() const { return {}; }
bool NCNNDetector::allocate_buffers() { return false; }

}  // namespace lagari

#endif  // HAS_NCNN
