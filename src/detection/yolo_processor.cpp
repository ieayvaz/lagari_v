#include "lagari/detection/yolo_processor.hpp"
#include "lagari/core/logger.hpp"
#include "lagari/core/profiler.hpp"

#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace lagari {

// ============================================================================
// YOLOProcessor Implementation
// ============================================================================

YOLOProcessor::YOLOProcessor(const DetectionConfig& config)
    : yolo_version_(config.yolo_version)
    , input_width_(config.input_width)
    , input_height_(config.input_height)
    , conf_threshold_(config.confidence_threshold)
    , nms_threshold_(config.nms_threshold)
    , max_detections_(config.max_detections)
    , normalize_(config.normalize)
    , swap_rb_(config.swap_rb)
    , target_classes_(config.target_classes)
{
}

YOLOProcessor::PreprocessInfo YOLOProcessor::preprocess(const Frame& frame, float* output) {
    // Convert Frame to cv::Mat
    cv::Mat mat(frame.metadata.height, frame.metadata.width, CV_8UC3, 
                const_cast<uint8_t*>(frame.ptr()));
    return preprocess(mat, output);
}

YOLOProcessor::PreprocessInfo YOLOProcessor::preprocess(const cv::Mat& frame, float* output) {
    PreprocessInfo info;
    info.orig_width = frame.cols;
    info.orig_height = frame.rows;

    // Calculate letterbox scaling
    float scale_x = static_cast<float>(input_width_) / frame.cols;
    float scale_y = static_cast<float>(input_height_) / frame.rows;
    info.scale = std::min(scale_x, scale_y);

    int new_width = static_cast<int>(frame.cols * info.scale);
    int new_height = static_cast<int>(frame.rows * info.scale);
    info.pad_x = (input_width_ - new_width) / 2;
    info.pad_y = (input_height_ - new_height) / 2;

    // Resize with letterbox
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_width, new_height));

    // Create letterboxed image
    cv::Mat letterbox(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterbox(cv::Rect(info.pad_x, info.pad_y, new_width, new_height)));

    // Swap R/B if needed
    if (swap_rb_) {
        cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
    }

    // Convert to float and normalize (HWC -> CHW)
    const float norm_factor = normalize_ ? 1.0f / 255.0f : 1.0f;
    
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < static_cast<int>(input_height_); ++y) {
            for (int x = 0; x < static_cast<int>(input_width_); ++x) {
                int src_idx = y * letterbox.step + x * 3 + c;
                int dst_idx = c * input_height_ * input_width_ + y * input_width_ + x;
                output[dst_idx] = letterbox.data[src_idx] * norm_factor;
            }
        }
    }

    return info;
}

DetectionResult YOLOProcessor::postprocess(
    const float* output,
    const std::vector<int>& output_shape,
    const PreprocessInfo& preproc_info,
    uint64_t frame_id)
{
    DetectionResult result;
    result.frame_id = frame_id;
    result.timestamp = Clock::now();

    std::vector<Detection> detections;

    // Parse output based on YOLO version
    int num_boxes = 0;
    int num_classes = 0;

    // Determine output layout
    // YOLOv5/v7: [batch, num_boxes, 5+num_classes] - xywh + objectness + class_probs
    // YOLOv8/v11: [batch, 4+num_classes, num_boxes] - transposed, no objectness
    // YOLOv10: [batch, num_boxes, 6] - xyxy + score + class_id (no NMS needed)

    if (output_shape.size() >= 2) {
        if (yolo_version_ == YOLOVersion::YOLOv8 || 
            yolo_version_ == YOLOVersion::YOLOv9 ||
            yolo_version_ == YOLOVersion::YOLO11) {
            // YOLOv8/v9/v11 format: [4+num_classes, num_boxes]
            num_classes = output_shape[1] - 4;
            num_boxes = output_shape[2];
            detections = parse_yolov8_output(output, num_boxes, num_classes, preproc_info);
        } else if (yolo_version_ == YOLOVersion::YOLOv10) {
            // YOLOv10 format: [num_boxes, 6]
            num_boxes = output_shape[1];
            num_classes = class_names_.size() > 0 ? static_cast<int>(class_names_.size()) : 80;
            detections = parse_yolov10_output(output, num_boxes, num_classes, preproc_info);
            // YOLOv10 doesn't need NMS
        } else {
            // YOLOv5/v7 format: [num_boxes, 5+num_classes]
            num_boxes = output_shape[1];
            num_classes = output_shape[2] - 5;
            detections = parse_yolov5_output(output, num_boxes, num_classes, preproc_info);
        }
    }

    // Apply NMS (except for YOLOv10 which has built-in NMS)
    if (yolo_version_ != YOLOVersion::YOLOv10) {
        detections = apply_nms(detections, nms_threshold_);
    }

    // Limit to max detections
    if (static_cast<int>(detections.size()) > max_detections_) {
        detections.resize(max_detections_);
    }

    result.detections = std::move(detections);
    return result;
}

std::vector<Detection> YOLOProcessor::parse_yolov5_output(
    const float* output, int num_boxes, int num_classes,
    const PreprocessInfo& info)
{
    std::vector<Detection> detections;
    const int stride = 5 + num_classes;

    for (int i = 0; i < num_boxes; ++i) {
        const float* row = output + i * stride;
        
        float objectness = row[4];
        if (objectness < conf_threshold_) continue;

        // Find best class
        int best_class = 0;
        float best_score = row[5];
        for (int c = 1; c < num_classes; ++c) {
            if (row[5 + c] > best_score) {
                best_score = row[5 + c];
                best_class = c;
            }
        }

        float confidence = objectness * best_score;
        if (confidence < conf_threshold_) continue;

        // Filter by target classes
        if (!target_classes_.empty()) {
            if (std::find(target_classes_.begin(), target_classes_.end(), best_class) 
                == target_classes_.end()) {
                continue;
            }
        }

        Detection det;
        det.class_id = best_class;
        det.confidence = confidence;
        if (best_class < static_cast<int>(class_names_.size())) {
            det.class_name = class_names_[best_class];
        }

        // Convert from center format to corner format
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        det.bbox.x = cx - w / 2;
        det.bbox.y = cy - h / 2;
        det.bbox.width = w;
        det.bbox.height = h;

        scale_coords(det, info);
        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> YOLOProcessor::parse_yolov8_output(
    const float* output, int num_boxes, int num_classes,
    const PreprocessInfo& info)
{
    std::vector<Detection> detections;

    // YOLOv8 output is transposed: [4+num_classes, num_boxes]
    for (int i = 0; i < num_boxes; ++i) {
        // Find best class
        int best_class = 0;
        float best_score = output[(4 + 0) * num_boxes + i];
        for (int c = 1; c < num_classes; ++c) {
            float score = output[(4 + c) * num_boxes + i];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < conf_threshold_) continue;

        // Filter by target classes
        if (!target_classes_.empty()) {
            if (std::find(target_classes_.begin(), target_classes_.end(), best_class) 
                == target_classes_.end()) {
                continue;
            }
        }

        Detection det;
        det.class_id = best_class;
        det.confidence = best_score;
        if (best_class < static_cast<int>(class_names_.size())) {
            det.class_name = class_names_[best_class];
        }

        // Get box coordinates (center format)
        float cx = output[0 * num_boxes + i];
        float cy = output[1 * num_boxes + i];
        float w = output[2 * num_boxes + i];
        float h = output[3 * num_boxes + i];

        det.bbox.x = cx - w / 2;
        det.bbox.y = cy - h / 2;
        det.bbox.width = w;
        det.bbox.height = h;

        scale_coords(det, info);
        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> YOLOProcessor::parse_yolov10_output(
    const float* output, int num_boxes, int num_classes,
    const PreprocessInfo& info)
{
    std::vector<Detection> detections;

    // YOLOv10 output: [num_boxes, 6] - x1, y1, x2, y2, score, class_id
    for (int i = 0; i < num_boxes; ++i) {
        const float* row = output + i * 6;
        
        float score = row[4];
        if (score < conf_threshold_) continue;

        int class_id = static_cast<int>(row[5]);
        
        // Filter by target classes
        if (!target_classes_.empty()) {
            if (std::find(target_classes_.begin(), target_classes_.end(), class_id) 
                == target_classes_.end()) {
                continue;
            }
        }

        Detection det;
        det.class_id = class_id;
        det.confidence = score;
        if (class_id < static_cast<int>(class_names_.size())) {
            det.class_name = class_names_[class_id];
        }

        // Coordinates are already in xyxy format
        float x1 = row[0];
        float y1 = row[1];
        float x2 = row[2];
        float y2 = row[3];

        det.bbox.x = x1;
        det.bbox.y = y1;
        det.bbox.width = x2 - x1;
        det.bbox.height = y2 - y1;

        scale_coords(det, info);
        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> YOLOProcessor::apply_nms(
    std::vector<Detection>& detections,
    float nms_threshold)
{
    if (detections.empty()) return {};

    // Convert to OpenCV format for NMS
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (const auto& det : detections) {
        boxes.emplace_back(
            static_cast<int>(det.bbox.x),
            static_cast<int>(det.bbox.y),
            static_cast<int>(det.bbox.width),
            static_cast<int>(det.bbox.height));
        scores.push_back(det.confidence);
        class_ids.push_back(det.class_id);
    }

    // Per-class NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold, indices);

    std::vector<Detection> result;
    for (int idx : indices) {
        result.push_back(detections[idx]);
    }

    // Sort by confidence
    std::sort(result.begin(), result.end(), 
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    return result;
}

void YOLOProcessor::scale_coords(Detection& det, const PreprocessInfo& info) {
    // Remove letterbox padding
    det.bbox.x = (det.bbox.x - info.pad_x) / info.scale;
    det.bbox.y = (det.bbox.y - info.pad_y) / info.scale;
    det.bbox.width = det.bbox.width / info.scale;
    det.bbox.height = det.bbox.height / info.scale;

    // Clip to image bounds
    det.bbox.x = std::max(0.0f, det.bbox.x);
    det.bbox.y = std::max(0.0f, det.bbox.y);
    det.bbox.width = std::min(det.bbox.width, static_cast<float>(info.orig_width) - det.bbox.x);
    det.bbox.height = std::min(det.bbox.height, static_cast<float>(info.orig_height) - det.bbox.y);
}

bool YOLOProcessor::load_labels(const std::string& labels_path) {
    std::ifstream file(labels_path);
    if (!file.is_open()) {
        LOG_ERROR("YOLOProcessor: Failed to open labels file: {}", labels_path);
        return false;
    }

    class_names_.clear();
    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty()) {
            class_names_.push_back(line);
        }
    }

    LOG_INFO("YOLOProcessor: Loaded {} class labels", class_names_.size());
    return !class_names_.empty();
}

// ============================================================================
// YOLODetectorBase Implementation
// ============================================================================

YOLODetectorBase::YOLODetectorBase(const DetectionConfig& config)
    : config_(config)
    , processor_(config)
{
}

DetectionResult YOLODetectorBase::detect(const Frame& frame) {
    auto start = Clock::now();

    // Preprocess
    YOLOProcessor::PreprocessInfo preproc_info;
    {
        PERF_SCOPE("detection.preprocess");
        if (input_buffer_.empty()) {
            input_buffer_.resize(config_.input_width * config_.input_height * 3);
        }
        preproc_info = processor_.preprocess(frame, input_buffer_.data());
    }

    // Infer
    {
        PERF_SCOPE("detection.inference");
        if (!infer(input_buffer_.data(), output_buffer_.data())) {
            LOG_ERROR("YOLODetectorBase: Inference failed");
            return {};
        }
    }

    // Postprocess
    DetectionResult result;
    {
        PERF_SCOPE("detection.postprocess");
        result = processor_.postprocess(
            output_buffer_.data(), 
            get_output_shape(), 
            preproc_info,
            frame.metadata.frame_id);
    }

    // Update stats
    auto end = Clock::now();
    double inference_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        total_inference_time_ += inference_ms;
        inference_count_++;
        stats_.frames_processed = inference_count_;
        stats_.average_inference_time = std::chrono::duration_cast<Duration>(
            std::chrono::duration<double, std::milli>(total_inference_time_ / inference_count_));
        stats_.last_inference_time = std::chrono::duration_cast<Duration>(
            std::chrono::duration<double, std::milli>(inference_ms));
        stats_.inference_fps = inference_count_ > 0 ? static_cast<float>(1000.0 / (total_inference_time_ / inference_count_)) : 0.0f;
    }

    return result;
}

bool YOLODetectorBase::detect_async(FramePtr frame) {
    // Simple implementation - run synchronously and store result
    if (!frame) return false;
    
    auto result = detect(*frame);
    
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        latest_result_ = std::move(result);
    }
    
    return true;
}

DetectionResult YOLODetectorBase::get_latest_result() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    return latest_result_;
}

void YOLODetectorBase::set_confidence_threshold(float threshold) {
    config_.confidence_threshold = threshold;
    processor_.set_confidence_threshold(threshold);
}

void YOLODetectorBase::set_nms_threshold(float threshold) {
    config_.nms_threshold = threshold;
    processor_.set_nms_threshold(threshold);
}

DetectionStats YOLODetectorBase::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

const std::vector<std::string>& YOLODetectorBase::class_names() const {
    return processor_.class_names();
}

}  // namespace lagari
