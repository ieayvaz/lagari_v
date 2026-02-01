#pragma once

#include <memory>
#include <string>
#include <string_view>

#include <spdlog/spdlog.h>
// Forward declaration for spdlog
namespace spdlog {
class logger;
}

namespace lagari {

/**
 * @brief Log severity levels
 */
enum class LogLevel : int {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    CRITICAL = 5,
    OFF = 6
};

/**
 * @brief Logger class wrapping spdlog
 * 
 * Provides:
 * - Async logging with ring buffer
 * - Multiple sinks (console, file)
 * - Module-tagged messages
 * - Performance-oriented design
 */
class Logger {
public:
    /**
     * @brief Initialize the logging system
     * 
     * @param log_file Path to log file (optional)
     * @param console_level Minimum level for console output
     * @param file_level Minimum level for file output
     * @return true if initialization successful
     */
    static bool init(const std::string& log_file = "",
                     LogLevel console_level = LogLevel::INFO,
                     LogLevel file_level = LogLevel::DEBUG);

    /**
     * @brief Shutdown the logging system
     * 
     * Flushes all pending messages.
     */
    static void shutdown();

    /**
     * @brief Set global log level
     */
    static void set_level(LogLevel level);

    /**
     * @brief Set log level for specific module
     */
    static void set_level(const std::string& module, LogLevel level);

    /**
     * @brief Flush all pending log messages
     */
    static void flush();

    /**
     * @brief Get or create a logger for a module
     * 
     * @param module Module name
     * @return Shared pointer to logger
     */
    static std::shared_ptr<spdlog::logger> get(const std::string& module);

    /**
     * @brief Get the default logger
     */
    static std::shared_ptr<spdlog::logger> get();
};

// ============================================================================
// Logging Macros
// ============================================================================

#define LAGARI_LOG_TRACE(module, ...) \
    if (auto _log = ::lagari::Logger::get(module)) _log->trace(__VA_ARGS__)

#define LAGARI_LOG_DEBUG(module, ...) \
    if (auto _log = ::lagari::Logger::get(module)) _log->debug(__VA_ARGS__)

#define LAGARI_LOG_INFO(module, ...) \
    if (auto _log = ::lagari::Logger::get(module)) _log->info(__VA_ARGS__)

#define LAGARI_LOG_WARN(module, ...) \
    if (auto _log = ::lagari::Logger::get(module)) _log->warn(__VA_ARGS__)

#define LAGARI_LOG_ERROR(module, ...) \
    if (auto _log = ::lagari::Logger::get(module)) _log->error(__VA_ARGS__)

#define LAGARI_LOG_CRITICAL(module, ...) \
    if (auto _log = ::lagari::Logger::get(module)) _log->critical(__VA_ARGS__)

// Shorthand with default module
#define LOG_TRACE(...) LAGARI_LOG_TRACE("lagari", __VA_ARGS__)
#define LOG_DEBUG(...) LAGARI_LOG_DEBUG("lagari", __VA_ARGS__)
#define LOG_INFO(...)  LAGARI_LOG_INFO("lagari", __VA_ARGS__)
#define LOG_WARN(...)  LAGARI_LOG_WARN("lagari", __VA_ARGS__)
#define LOG_ERROR(...) LAGARI_LOG_ERROR("lagari", __VA_ARGS__)
#define LOG_CRITICAL(...) LAGARI_LOG_CRITICAL("lagari", __VA_ARGS__)

}  // namespace lagari
