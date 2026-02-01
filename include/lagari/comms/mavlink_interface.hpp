#pragma once

#include "lagari/core/module.hpp"
#include "lagari/core/types.hpp"
#include <memory>
#include <string>
#include <functional>

namespace lagari {

class Config;

/**
 * @brief MAVLink message callback type
 */
using MAVLinkMessageCallback = std::function<void(uint32_t msg_id, const void* msg)>;

/**
 * @brief MAVLink configuration
 */
struct MAVLinkConfig {
    std::string connection;      // Connection string (e.g., "udp://:14540")
    uint8_t system_id = 1;
    uint8_t component_id = 196;  // MAV_COMP_ID_USER1
    
    uint32_t heartbeat_interval_ms = 1000;
    uint32_t timeout_ms = 5000;
    
    bool offboard_control = true;
};

/**
 * @brief MAVLink interface
 */
class IMAVLink : public IModule {
public:
    virtual ~IMAVLink() = default;

    /**
     * @brief Connect to autopilot
     * 
     * @param connection Connection string (overrides config if provided)
     * @return true if connected
     */
    virtual bool connect(const std::string& connection = "") = 0;

    /**
     * @brief Disconnect from autopilot
     */
    virtual void disconnect() = 0;

    /**
     * @brief Check if connected
     */
    virtual bool is_connected() const = 0;

    /**
     * @brief Send guidance command
     * 
     * @param cmd Guidance command
     * @return true if sent successfully
     */
    virtual bool send_guidance_command(const GuidanceCommand& cmd) = 0;

    /**
     * @brief Send heartbeat
     * 
     * @return true if sent successfully
     */
    virtual bool send_heartbeat() = 0;

    /**
     * @brief Get current vehicle state
     * 
     * @return Latest vehicle state
     */
    virtual VehicleState get_vehicle_state() const = 0;

    /**
     * @brief Register callback for specific message type
     * 
     * @param msg_id MAVLink message ID
     * @param callback Function to call when message received
     */
    virtual void register_message_callback(uint32_t msg_id, MAVLinkMessageCallback callback) = 0;

    /**
     * @brief Request offboard control
     * 
     * @return true if request sent
     */
    virtual bool request_offboard_control() = 0;

    /**
     * @brief Release offboard control
     */
    virtual void release_offboard_control() = 0;

    /**
     * @brief Check if in offboard mode
     */
    virtual bool is_offboard() const = 0;
};

/**
 * @brief Create MAVLink interface
 */
std::unique_ptr<IMAVLink> create_mavlink(const Config& config);
std::unique_ptr<IMAVLink> create_mavlink(const MAVLinkConfig& config);

}  // namespace lagari
