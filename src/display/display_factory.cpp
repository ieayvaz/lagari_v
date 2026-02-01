/**
 * @file display_factory.cpp
 * @brief Factory function for creating display instances
 */

#include "lagari/display/display.hpp"
#include "lagari/display/gstreamer_display.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

namespace lagari {

std::unique_ptr<IDisplay> create_display(const Config& config) {
    bool enabled = config.get_bool("display.enabled", false);
    
    if (!enabled) {
        LOG_DEBUG("Display module disabled by configuration");
        return nullptr;
    }

#ifdef HAS_GSTREAMER
    DisplayConfig display_config;
    display_config.enabled = true;
    display_config.target = config.get_string("display.target", "window");
    display_config.window_name = config.get_string("display.window_name", "Lagari Vision");
    display_config.fullscreen = config.get_bool("display.fullscreen", false);
    display_config.host = config.get_string("display.host", "0.0.0.0");
    display_config.port = static_cast<uint16_t>(config.get_int("display.port", 5600));
    display_config.bitrate_kbps = config.get_int("display.bitrate_kbps", 2000);
    display_config.pipeline = config.get_string("display.pipeline", "");
    display_config.max_fps = config.get_int("display.max_fps", 30);
    display_config.drop_frames = config.get_bool("display.drop_frames", true);
    display_config.codec = config.get_string("display.codec", "h264");
    display_config.hw_encode = config.get_bool("display.hw_encode", true);
    
    // Overlay
    display_config.overlay.enabled = config.get_bool("display.overlay.enabled", true);
    display_config.overlay.timestamp = config.get_bool("display.overlay.timestamp", true);
    display_config.overlay.bounding_boxes = config.get_bool("display.overlay.bounding_boxes", true);
    display_config.overlay.state = config.get_bool("display.overlay.state", true);
    display_config.overlay.latency = config.get_bool("display.overlay.latency", true);

    return std::make_unique<GstreamerDisplay>(display_config);
#else
    LOG_WARN("Display requires GStreamer, which is not available");
    return nullptr;
#endif
}

std::unique_ptr<IDisplay> create_display(const DisplayConfig& config) {
    if (!config.enabled) {
        return nullptr;
    }

#ifdef HAS_GSTREAMER
    return std::make_unique<GstreamerDisplay>(config);
#else
    LOG_WARN("Display requires GStreamer, which is not available");
    return nullptr;
#endif
}

}  // namespace lagari
