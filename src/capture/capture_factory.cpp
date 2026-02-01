#include "lagari/capture/capture.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#ifdef HAS_V4L2
#include "lagari/capture/v4l2_capture.hpp"
#endif

#ifdef HAS_ARGUS
#include "lagari/capture/argus_capture.hpp"
#endif

#ifdef HAS_LIBCAMERA
#include "lagari/capture/libcamera_capture.hpp"
#endif

#ifdef HAS_GSTREAMER
#include "lagari/capture/gstreamer_capture.hpp"
#endif

#include "lagari/capture/sim_capture.hpp"
#include "lagari/capture/isaac_shm_capture.hpp"

namespace lagari {

std::unique_ptr<ICapture> create_capture(const Config& config) {
    CaptureConfig cc;
    
    std::string source_str = config.get_string("capture.source", "auto");
    if (source_str == "csi") cc.source = CaptureSource::CSI;
    else if (source_str == "usb") cc.source = CaptureSource::USB;
    else if (source_str == "file") cc.source = CaptureSource::FILE;
    else if (source_str == "rtsp") cc.source = CaptureSource::RTSP;
    else if (source_str == "gstreamer") cc.source = CaptureSource::GSTREAMER;
    else if (source_str == "simulation") cc.source = CaptureSource::SIMULATION;
    else if (source_str == "isaac" || source_str == "isaac_sim") cc.source = CaptureSource::ISAAC_SIM;
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
    
    // For platform-dependent source, we need to create the right type
    auto capture = create_capture(cc);
    
    // Initialize with full config (for backend-specific settings)
    if (capture) {
        if (!capture->initialize(config)) {
            LOG_ERROR("Failed to initialize capture");
            return nullptr;
        }
    }
    
    return capture;
}

std::unique_ptr<ICapture> create_capture(const CaptureConfig& config) {
    LOG_INFO("Creating capture with source: {}", to_string(config.source));
    
    CaptureSource source = config.source;
    
    if (source == CaptureSource::AUTO) {
        // Auto-detect based on platform
#if defined(PLATFORM_JETSON) && defined(HAS_ARGUS)
        source = CaptureSource::CSI;
        LOG_INFO("Auto-detected: Jetson with Argus");
#elif defined(PLATFORM_RPI) && defined(HAS_LIBCAMERA)
        source = CaptureSource::CSI;
        LOG_INFO("Auto-detected: Raspberry Pi with libcamera");
#elif defined(HAS_V4L2)
        source = CaptureSource::USB;
        LOG_INFO("Auto-detected: V4L2 for USB camera");
#else
        source = CaptureSource::SIMULATION;
        LOG_INFO("Auto-detected: Simulation (no hardware capture available)");
#endif
    }

    switch (source) {
        case CaptureSource::CSI:
#if defined(PLATFORM_JETSON) && defined(HAS_ARGUS)
            return std::make_unique<ArgusCapture>(config);
#elif defined(PLATFORM_RPI) && defined(HAS_LIBCAMERA)
            return std::make_unique<LibcameraCapture>(config);
#else
            LOG_WARN("CSI capture not available on this platform");
            break;
#endif

        case CaptureSource::USB:
#ifdef HAS_V4L2
            return std::make_unique<V4L2Capture>(config);
#else
            LOG_WARN("V4L2 not available");
            break;
#endif

        case CaptureSource::FILE:
        case CaptureSource::RTSP:
#ifdef HAS_GSTREAMER
            // Use GStreamer for file/RTSP sources
            return std::make_unique<GstreamerCapture>(config);
#else
            LOG_WARN("File/RTSP capture requires GStreamer (HAS_GSTREAMER not defined)");
            break;
#endif

        case CaptureSource::GSTREAMER:
#ifdef HAS_GSTREAMER
            return std::make_unique<GstreamerCapture>(config);
#else
            LOG_WARN("GStreamer not available");
            break;
#endif

        case CaptureSource::ISAAC_SIM:
            // Isaac Sim via shared memory (high performance)
            return std::make_unique<IsaacShmCapture>(config);

        case CaptureSource::SIMULATION:
            return std::make_unique<SimCapture>(config);

        case CaptureSource::AUTO:
        default:
            LOG_ERROR("Could not determine capture source");
            break;
    }

    return nullptr;
}

}  // namespace lagari
