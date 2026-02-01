#pragma once

#include "lagari/core/module.hpp"
#include "lagari/core/types.hpp"
#include <memory>
#include <string>

namespace lagari {

class Config;

/**
 * @brief Telemetry configuration
 */
struct TelemetryConfig {
    std::string host = "0.0.0.0";
    uint16_t data_port = 5601;
    uint16_t video_port = 5602;
    
    float data_rate_hz = 10.0f;
    
    // Video settings
    bool video_enabled = true;
    std::string video_codec = "h264";
    uint32_t video_bitrate_kbps = 2000;
};

/**
 * @brief Telemetry interface
 */
class ITelemetry : public IModule {
public:
    virtual ~ITelemetry() = default;

    /**
     * @brief Send detection data to GCS
     */
    virtual void send_detection_data(const DetectionResult& result) = 0;

    /**
     * @brief Send QR data to GCS
     */
    virtual void send_qr_data(const QRResult& result) = 0;

    /**
     * @brief Send system status to GCS
     */
    virtual void send_status(SystemState state, const GuidanceCommand& cmd) = 0;

    /**
     * @brief Send frame for video streaming
     */
    virtual void send_video_frame(const Frame& frame) = 0;

    /**
     * @brief Check if GCS is connected
     */
    virtual bool is_gcs_connected() const = 0;

    /**
     * @brief Get number of connected clients
     */
    virtual int connected_clients() const = 0;
};

/**
 * @brief Create telemetry instance
 */
std::unique_ptr<ITelemetry> create_telemetry(const Config& config);
std::unique_ptr<ITelemetry> create_telemetry(const TelemetryConfig& config);

}  // namespace lagari
