#include "lagari/capture/capture.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

namespace lagari {

std::unique_ptr<ICapture> create_capture(const Config& config) {
    CaptureConfig cc;
    
    std::string source_str = config.get_string("capture.source", "auto");
    if (source_str == "csi") cc.source = CaptureSource::CSI;
    else if (source_str == "usb") cc.source = CaptureSource::USB;
    else if (source_str == "file") cc.source = CaptureSource::FILE;
    else if (source_str == "rtsp") cc.source = CaptureSource::RTSP;
    else if (source_str == "simulation") cc.source = CaptureSource::SIMULATION;
    else cc.source = CaptureSource::AUTO;
    
    cc.width = config.get_uint("capture.width", 1280);
    cc.height = config.get_uint("capture.height", 720);
    cc.fps = config.get_uint("capture.fps", 30);
    cc.device = config.get_string("capture.device", "");
    cc.camera_id = config.get_int("capture.camera_id", 0);
    cc.file_path = config.get_string("capture.file_path", "");
    cc.loop_file = config.get_bool("capture.loop_file", true);
    cc.buffer_count = config.get_uint("capture.buffer_count", 4);
    cc.drop_frames = config.get_bool("capture.drop_frames", true);
    cc.auto_exposure = config.get_bool("capture.auto_exposure", true);
    cc.flip_horizontal = config.get_bool("capture.flip_horizontal", false);
    cc.flip_vertical = config.get_bool("capture.flip_vertical", false);
    cc.rotation = config.get_int("capture.rotation", 0);
    
    return create_capture(cc);
}

std::unique_ptr<ICapture> create_capture(const CaptureConfig& config) {
    // TODO: Implement capture factory based on source and platform
    
    LOG_INFO("Creating capture with source: {}", to_string(config.source));
    
    CaptureSource source = config.source;
    
    if (source == CaptureSource::AUTO) {
        // Auto-detect based on platform
#if defined(PLATFORM_JETSON) && defined(HAS_ARGUS)
        source = CaptureSource::CSI;
#elif defined(PLATFORM_RPI) && defined(HAS_LIBCAMERA)
        source = CaptureSource::CSI;
#elif defined(HAS_V4L2)
        source = CaptureSource::USB;
#else
        source = CaptureSource::SIMULATION;
#endif
    }
    
    switch (source) {
        case CaptureSource::CSI:
#if defined(PLATFORM_JETSON) && defined(HAS_ARGUS)
            // return std::make_unique<ArgusCapture>(config);
#elif defined(PLATFORM_RPI) && defined(HAS_LIBCAMERA)
            // return std::make_unique<LibcameraCapture>(config);
#endif
            LOG_WARN("CSI capture not available, falling back to simulation");
            [[fallthrough]];
            
        case CaptureSource::USB:
#ifdef HAS_V4L2
            // return std::make_unique<V4L2Capture>(config);
#endif
            LOG_WARN("V4L2 capture not available, falling back to simulation");
            [[fallthrough]];
            
        case CaptureSource::FILE:
        case CaptureSource::RTSP:
            // return std::make_unique<FileCapture>(config);
            LOG_WARN("File/RTSP capture not implemented");
            [[fallthrough]];
            
        case CaptureSource::SIMULATION:
        default:
            // return std::make_unique<SimCapture>(config);
            LOG_WARN("Capture implementations not yet available");
            return nullptr;
    }
}

}  // namespace lagari
