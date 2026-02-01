// MAVLink interface
// TODO: Implement MAVLink 2.0 communication

#include "lagari/comms/mavlink_interface.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

namespace lagari {

std::unique_ptr<IMAVLink> create_mavlink(const Config& config) {
    (void)config;
    LOG_INFO("MAVLink interface not yet implemented");
    return nullptr;
}

std::unique_ptr<IMAVLink> create_mavlink(const MAVLinkConfig& config) {
    (void)config;
    return nullptr;
}

}  // namespace lagari
