#include "lagari/core/logger.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <mutex>
#include <unordered_map>

namespace lagari {

namespace {

// Convert our level to spdlog level
spdlog::level::level_enum to_spdlog_level(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return spdlog::level::trace;
        case LogLevel::DEBUG: return spdlog::level::debug;
        case LogLevel::INFO: return spdlog::level::info;
        case LogLevel::WARN: return spdlog::level::warn;
        case LogLevel::ERROR: return spdlog::level::err;
        case LogLevel::CRITICAL: return spdlog::level::critical;
        case LogLevel::OFF: return spdlog::level::off;
        default: return spdlog::level::info;
    }
}

// Global state
std::once_flag init_flag;
std::mutex logger_mutex;
std::unordered_map<std::string, std::shared_ptr<spdlog::logger>> loggers;
std::vector<spdlog::sink_ptr> sinks;

}  // namespace

bool Logger::init(const std::string& log_file,
                  LogLevel console_level,
                  LogLevel file_level) {
    std::call_once(init_flag, [&]() {
        try {
            // Initialize async logging thread pool
            spdlog::init_thread_pool(8192, 1);  // Queue size, thread count
            
            // Console sink with colors
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(to_spdlog_level(console_level));
            console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
            sinks.push_back(console_sink);
            
            // File sink (rotating, 10MB max, 3 files)
            if (!log_file.empty()) {
                auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                    log_file, 10 * 1024 * 1024, 3);
                file_sink->set_level(to_spdlog_level(file_level));
                file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] [%t] %v");
                sinks.push_back(file_sink);
            }
            
            // Create default logger
            auto default_logger = std::make_shared<spdlog::async_logger>(
                "lagari", sinks.begin(), sinks.end(),
                spdlog::thread_pool(),
                spdlog::async_overflow_policy::overrun_oldest);
            
            default_logger->set_level(spdlog::level::trace);  // Actual filtering per sink
            spdlog::register_logger(default_logger);
            spdlog::set_default_logger(default_logger);
            
            loggers["lagari"] = default_logger;
            
            spdlog::info("Logging system initialized");
            
        } catch (const spdlog::spdlog_ex& e) {
            // Fallback to stderr
            fprintf(stderr, "Logger initialization failed: %s\n", e.what());
        }
    });
    
    return true;
}

void Logger::shutdown() {
    spdlog::shutdown();
}

void Logger::set_level(LogLevel level) {
    spdlog::set_level(to_spdlog_level(level));
}

void Logger::set_level(const std::string& module, LogLevel level) {
    std::lock_guard<std::mutex> lock(logger_mutex);
    
    auto it = loggers.find(module);
    if (it != loggers.end()) {
        it->second->set_level(to_spdlog_level(level));
    }
}

void Logger::flush() {
    spdlog::default_logger()->flush();
    
    std::lock_guard<std::mutex> lock(logger_mutex);
    for (auto& [name, logger] : loggers) {
        logger->flush();
    }
}

std::shared_ptr<spdlog::logger> Logger::get(const std::string& module) {
    // Check if logger exists
    {
        std::lock_guard<std::mutex> lock(logger_mutex);
        auto it = loggers.find(module);
        if (it != loggers.end()) {
            return it->second;
        }
    }
    
    // Create new logger for module
    std::lock_guard<std::mutex> lock(logger_mutex);
    
    // Double-check after acquiring lock
    auto it = loggers.find(module);
    if (it != loggers.end()) {
        return it->second;
    }
    
    try {
        auto logger = std::make_shared<spdlog::async_logger>(
            module, sinks.begin(), sinks.end(),
            spdlog::thread_pool(),
            spdlog::async_overflow_policy::overrun_oldest);
        
        logger->set_level(spdlog::level::trace);
        spdlog::register_logger(logger);
        loggers[module] = logger;
        
        return logger;
        
    } catch (const spdlog::spdlog_ex& e) {
        // Return default logger on failure
        return spdlog::default_logger();
    }
}

std::shared_ptr<spdlog::logger> Logger::get() {
    return spdlog::default_logger();
}

}  // namespace lagari
