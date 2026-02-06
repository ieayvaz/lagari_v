#pragma once

/**
 * @file profiler.hpp
 * @brief Zero-overhead performance profiling system
 * 
 * All profiling macros expand to nothing when ENABLE_PROFILING is not defined,
 * resulting in zero runtime overhead in Release builds.
 * 
 * Usage:
 *   PERF_SCOPE("module.operation");  // Scoped timer, auto start/stop
 *   
 *   PERF_START("module.operation");
 *   // ... do work ...
 *   PERF_STOP("module.operation");
 *   
 *   PERF_LOG_SUMMARY();  // Log all accumulated stats
 */

#include "lagari/core/types.hpp"
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace lagari {

#ifdef ENABLE_PROFILING

/**
 * @brief Performance statistics for a single metric
 */
struct PerfStats {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> min_ns{UINT64_MAX};
    std::atomic<uint64_t> max_ns{0};
    
    void record(uint64_t duration_ns) {
        count.fetch_add(1, std::memory_order_relaxed);
        total_ns.fetch_add(duration_ns, std::memory_order_relaxed);
        
        // Update min/max with compare-exchange
        uint64_t current_min = min_ns.load(std::memory_order_relaxed);
        while (duration_ns < current_min && 
               !min_ns.compare_exchange_weak(current_min, duration_ns,
                                              std::memory_order_relaxed));
        
        uint64_t current_max = max_ns.load(std::memory_order_relaxed);
        while (duration_ns > current_max && 
               !max_ns.compare_exchange_weak(current_max, duration_ns,
                                              std::memory_order_relaxed));
    }
    
    double avg_ms() const {
        uint64_t c = count.load(std::memory_order_relaxed);
        if (c == 0) return 0.0;
        return static_cast<double>(total_ns.load(std::memory_order_relaxed)) / c / 1e6;
    }
    
    double min_ms() const {
        uint64_t m = min_ns.load(std::memory_order_relaxed);
        return m == UINT64_MAX ? 0.0 : static_cast<double>(m) / 1e6;
    }
    
    double max_ms() const {
        return static_cast<double>(max_ns.load(std::memory_order_relaxed)) / 1e6;
    }
    
    double total_ms() const {
        return static_cast<double>(total_ns.load(std::memory_order_relaxed)) / 1e6;
    }
};

/**
 * @brief Singleton profiler for collecting performance metrics
 */
class Profiler {
public:
    static Profiler& instance() {
        static Profiler instance;
        return instance;
    }
    
    void start(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        active_timers_[name] = Clock::now();
    }
    
    void stop(const std::string& name) {
        auto end = Clock::now();
        
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = active_timers_.find(name);
        if (it != active_timers_.end()) {
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - it->second).count();
            stats_[name].record(duration);
            active_timers_.erase(it);
        }
    }
    
    void record(const std::string& name, uint64_t duration_ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_[name].record(duration_ns);
    }
    
    const std::unordered_map<std::string, PerfStats>& stats() const {
        return stats_;
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.clear();
        active_timers_.clear();
    }
    
    void log_summary() const;
    void log_frame_breakdown() const;

private:
    Profiler() = default;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, PerfStats> stats_;
    std::unordered_map<std::string, TimePoint> active_timers_;
};

/**
 * @brief RAII scoped timer for automatic start/stop
 */
class ScopedTimer {
public:
    explicit ScopedTimer(const char* name) 
        : name_(name)
        , start_(Clock::now()) 
    {}
    
    ~ScopedTimer() {
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start_).count();
        Profiler::instance().record(name_, duration);
    }
    
    // Prevent copying
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    const char* name_;
    TimePoint start_;
};

// ============================================================================
// Profiling Macros (enabled)
// ============================================================================

#define PERF_SCOPE(name) \
    ::lagari::ScopedTimer _perf_timer_##__LINE__(name)

#define PERF_START(name) \
    ::lagari::Profiler::instance().start(name)

#define PERF_STOP(name) \
    ::lagari::Profiler::instance().stop(name)

#define PERF_LOG_SUMMARY() \
    ::lagari::Profiler::instance().log_summary()

#define PERF_LOG_FRAME() \
    ::lagari::Profiler::instance().log_frame_breakdown()

#define PERF_RESET() \
    ::lagari::Profiler::instance().reset()

#else  // ENABLE_PROFILING not defined

// ============================================================================
// Profiling Macros (disabled - zero overhead)
// ============================================================================

#define PERF_SCOPE(name)      ((void)0)
#define PERF_START(name)      ((void)0)
#define PERF_STOP(name)       ((void)0)
#define PERF_LOG_SUMMARY()    ((void)0)
#define PERF_LOG_FRAME()      ((void)0)
#define PERF_RESET()          ((void)0)

#endif  // ENABLE_PROFILING

}  // namespace lagari
