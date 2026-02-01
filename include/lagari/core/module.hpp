#pragma once

#include <memory>
#include <string>

namespace lagari {

// Forward declarations
class Config;

/**
 * @brief Base interface for all modules
 * 
 * Provides common lifecycle management for system components.
 * All modules follow the same initialization -> start -> stop pattern.
 */
class IModule {
public:
    virtual ~IModule() = default;

    /**
     * @brief Initialize module with configuration
     * 
     * Called once during system startup. Module should validate
     * configuration and allocate resources.
     * 
     * @param config System configuration
     * @return true if initialization successful
     */
    virtual bool initialize(const Config& config) = 0;

    /**
     * @brief Start module operation
     * 
     * Called after successful initialization. Module should start
     * any worker threads and begin processing.
     */
    virtual void start() = 0;

    /**
     * @brief Stop module operation
     * 
     * Called during shutdown. Module should stop worker threads
     * and release resources. Must be safe to call multiple times.
     */
    virtual void stop() = 0;

    /**
     * @brief Check if module is currently running
     * 
     * @return true if module is running
     */
    virtual bool is_running() const = 0;

    /**
     * @brief Get module name for logging/debugging
     * 
     * @return Module name string
     */
    virtual std::string name() const = 0;
};

/**
 * @brief Unique pointer type for modules
 */
using ModulePtr = std::unique_ptr<IModule>;

}  // namespace lagari
