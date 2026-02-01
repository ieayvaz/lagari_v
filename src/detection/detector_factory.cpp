#include "lagari/detection/detector.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

namespace lagari {

std::unique_ptr<IDetector> create_detector(const Config& config) {
    DetectionConfig dc;
    
    std::string backend_str = config.get_string("detection.backend", "auto");
    if (backend_str == "tensorrt") dc.backend = InferenceBackend::TENSORRT;
    else if (backend_str == "hailo") dc.backend = InferenceBackend::HAILO;
    else if (backend_str == "ncnn") dc.backend = InferenceBackend::NCNN;
    else if (backend_str == "openvino") dc.backend = InferenceBackend::OPENVINO;
    else dc.backend = InferenceBackend::AUTO;
    
    std::string yolo_str = config.get_string("detection.yolo_version", "yolov8");
    if (yolo_str == "yolov5") dc.yolo_version = YOLOVersion::YOLOv5;
    else if (yolo_str == "yolov7") dc.yolo_version = YOLOVersion::YOLOv7;
    else if (yolo_str == "yolov8") dc.yolo_version = YOLOVersion::YOLOv8;
    else if (yolo_str == "yolov9") dc.yolo_version = YOLOVersion::YOLOv9;
    else if (yolo_str == "yolov10") dc.yolo_version = YOLOVersion::YOLOv10;
    else if (yolo_str == "yolo11") dc.yolo_version = YOLOVersion::YOLO11;
    
    dc.model_path = config.get_string("detection.model_path", "");
    dc.labels_path = config.get_string("detection.labels_path", "");
    dc.input_width = config.get_uint("detection.input_width", 640);
    dc.input_height = config.get_uint("detection.input_height", 640);
    dc.confidence_threshold = config.get_float("detection.confidence_threshold", 0.5f);
    dc.nms_threshold = config.get_float("detection.nms_threshold", 0.45f);
    dc.max_detections = config.get_int("detection.max_detections", 100);
    dc.fp16 = config.get_bool("detection.fp16", true);
    dc.int8 = config.get_bool("detection.int8", false);
    dc.batch_size = config.get_int("detection.batch_size", 1);
    dc.dla_core = config.get_int("detection.dla_core", -1);
    
    dc.target_classes = config.get_int_list("detection.target_classes");
    
    return create_detector(dc);
}

std::unique_ptr<IDetector> create_detector(const DetectionConfig& config) {
    LOG_INFO("Creating detector with backend: {}", to_string(config.backend));
    
    InferenceBackend backend = config.backend;
    
    if (backend == InferenceBackend::AUTO) {
        // Auto-select best available backend
#if defined(HAS_TENSORRT)
        backend = InferenceBackend::TENSORRT;
#elif defined(HAS_HAILO)
        backend = InferenceBackend::HAILO;
#elif defined(HAS_NCNN)
        backend = InferenceBackend::NCNN;
#elif defined(HAS_OPENVINO)
        backend = InferenceBackend::OPENVINO;
#else
        LOG_ERROR("No inference backend available");
        return nullptr;
#endif
    }
    
    switch (backend) {
        case InferenceBackend::TENSORRT:
#ifdef HAS_TENSORRT
            // return std::make_unique<TensorRTDetector>(config);
#endif
            LOG_WARN("TensorRT not available");
            break;
            
        case InferenceBackend::HAILO:
#ifdef HAS_HAILO
            // return std::make_unique<HailoDetector>(config);
#endif
            LOG_WARN("HailoRT not available");
            break;
            
        case InferenceBackend::NCNN:
#ifdef HAS_NCNN
            // return std::make_unique<NCNNDetector>(config);
#endif
            LOG_WARN("NCNN not available");
            break;
            
        case InferenceBackend::OPENVINO:
#ifdef HAS_OPENVINO
            // return std::make_unique<OpenVINODetector>(config);
#endif
            LOG_WARN("OpenVINO not available");
            break;
            
        default:
            LOG_ERROR("Unknown inference backend");
            break;
    }
    
    return nullptr;
}

std::vector<InferenceBackend> available_backends() {
    std::vector<InferenceBackend> backends;
    
#ifdef HAS_TENSORRT
    backends.push_back(InferenceBackend::TENSORRT);
#endif
#ifdef HAS_HAILO
    backends.push_back(InferenceBackend::HAILO);
#endif
#ifdef HAS_NCNN
    backends.push_back(InferenceBackend::NCNN);
#endif
#ifdef HAS_OPENVINO
    backends.push_back(InferenceBackend::OPENVINO);
#endif
    
    return backends;
}

}  // namespace lagari
