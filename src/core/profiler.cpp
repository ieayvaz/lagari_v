/**
 * @file profiler.cpp
 * @brief Zero-overhead profiler implementation
 */

#include "lagari/core/profiler.hpp"

#ifdef ENABLE_PROFILING

#include "lagari/core/logger.hpp"
#include <vector>
#include <algorithm>

namespace lagari {

void Profiler::log_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (stats_.empty()) {
        LOG_INFO("Profiler: No performance data collected");
        return;
    }
    
    // Sort entries by total time (descending)
    std::vector<std::pair<std::string, const PerfStats*>> sorted;
    sorted.reserve(stats_.size());
    for (const auto& [name, stats] : stats_) {
        sorted.emplace_back(name, &stats);
    }
    std::sort(sorted.begin(), sorted.end(), 
        [](const auto& a, const auto& b) {
            return a.second->total_ns.load() > b.second->total_ns.load();
        });
    
    LOG_INFO("=== Performance Summary ===");
    LOG_INFO("{:<40} {:>8} {:>10} {:>10} {:>10} {:>12}",
             "Metric", "Count", "Avg(ms)", "Min(ms)", "Max(ms)", "Total(ms)");
    LOG_INFO("{}", std::string(92, '-'));
    
    for (const auto& [name, stats] : sorted) {
        LOG_INFO("{:<40} {:>8} {:>10.2f} {:>10.2f} {:>10.2f} {:>12.1f}",
                 name,
                 stats->count.load(),
                 stats->avg_ms(),
                 stats->min_ms(),
                 stats->max_ms(),
                 stats->total_ms());
    }
    
    LOG_INFO("{}", std::string(92, '='));
}

void Profiler::log_frame_breakdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (stats_.empty()) {
        return;
    }
    
    // Group stats by module prefix (before the '.')
    std::unordered_map<std::string, double> module_totals;
    double total_time = 0.0;
    
    for (const auto& [name, stats] : stats_) {
        double avg = stats.avg_ms();
        total_time += avg;
        
        // Extract module prefix
        size_t dot_pos = name.find('.');
        std::string module = (dot_pos != std::string::npos) 
                           ? name.substr(0, dot_pos) 
                           : name;
        module_totals[module] += avg;
    }
    
    // Log breakdown
    std::ostringstream oss;
    oss << "Latency breakdown: ";
    
    // Sort by time
    std::vector<std::pair<std::string, double>> sorted(
        module_totals.begin(), module_totals.end());
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    bool first = true;
    for (const auto& [module, time] : sorted) {
        if (!first) oss << ", ";
        first = false;
        
        double pct = (total_time > 0) ? (time / total_time * 100.0) : 0.0;
        oss << module << "=" << std::fixed << std::setprecision(1) 
            << time << "ms (" << std::setprecision(0) << pct << "%)";
    }
    
    LOG_DEBUG("{}", oss.str());
}

}  // namespace lagari

#endif  // ENABLE_PROFILING
