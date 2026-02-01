/**
 * @file recorder.cpp
 * @brief Factory function for creating recorder instances
 */

#include "lagari/recording/recorder.hpp"
#include "lagari/recording/gstreamer_recorder.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

namespace lagari {

std::unique_ptr<IRecorder> create_recorder(const Config& config) {
    bool enabled = config.get_bool("recording.enabled", false);
    
    if (!enabled) {
        LOG_DEBUG("Recording module disabled by configuration");
        return nullptr;
    }

#ifdef HAS_GSTREAMER
    RecordingConfig rec_config;
    rec_config.enabled = true;
    rec_config.output_dir = config.get_string("recording.output_dir", "/var/lagari/recordings");
    rec_config.codec = config.get_string("recording.codec", "h264");
    rec_config.bitrate_kbps = config.get_int("recording.bitrate_kbps", 8000);
    rec_config.fps = config.get_int("recording.fps", 30);
    rec_config.hw_encode = config.get_bool("recording.hw_encode", true);
    rec_config.container = config.get_string("recording.container", "mp4");
    rec_config.segment_duration_s = config.get_int("recording.segment_duration_s", 0);
    rec_config.max_storage_bytes = static_cast<uint64_t>(
        config.get_double("recording.max_storage_gb", 10.0) * 1024 * 1024 * 1024);
    rec_config.delete_oldest = config.get_bool("recording.delete_oldest", true);
    
    // Overlay configuration
    rec_config.overlay.enabled = config.get_bool("recording.overlay.enabled", true);
    rec_config.overlay.timestamp = config.get_bool("recording.overlay.timestamp", true);
    rec_config.overlay.bounding_boxes = config.get_bool("recording.overlay.bounding_boxes", true);
    rec_config.overlay.state = config.get_bool("recording.overlay.state", true);
    rec_config.overlay.latency = config.get_bool("recording.overlay.latency", true);

    return std::make_unique<GstreamerRecorder>(rec_config);
#else
    LOG_WARN("Recording requires GStreamer, which is not available");
    return nullptr;
#endif
}

std::unique_ptr<IRecorder> create_recorder(const RecordingConfig& config) {
    if (!config.enabled) {
        return nullptr;
    }

#ifdef HAS_GSTREAMER
    return std::make_unique<GstreamerRecorder>(config);
#else
    LOG_WARN("Recording requires GStreamer, which is not available");
    return nullptr;
#endif
}

}  // namespace lagari
