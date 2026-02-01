// Telemetry to GCS
// TODO: Implement UDP telemetry

#include "lagari/comms/telemetry.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

namespace lagari {

std::unique_ptr<ITelemetry> create_telemetry(const Config& config) {
    (void)config;
    LOG_INFO("Telemetry not yet implemented");
    return nullptr;
}

std::unique_ptr<ITelemetry> create_telemetry(const TelemetryConfig& config) {
    (void)config;
    return nullptr;
}

}  // namespace lagari
