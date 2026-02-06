/**
 * @file main.cpp
 * @brief Lagari Vision System main entry point
 * 
 * This implements a threaded architecture with:
 * - Capture thread: Grabs frames from camera
 * - Processing thread: Runs detection, display, and recording
 * - Main thread: Monitors system health and handles signals
 */

#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"
#include "lagari/core/system_state.hpp"
#include "lagari/core/types.hpp"
#include "lagari/core/ring_buffer.hpp"
#include "lagari/core/spsc_queue.hpp"
#include "lagari/core/profiler.hpp"

#include "lagari/capture/capture.hpp"
#include "lagari/detection/detector.hpp"
#include "lagari/display/display.hpp"
#include "lagari/recording/recorder.hpp"

// Future modules
// #include "lagari/qr/qr_decoder.hpp"
// #include "lagari/guidance/guidance.hpp"
// #include "lagari/comms/mavlink_interface.hpp"
// #include "lagari/comms/telemetry.hpp"

#include <csignal>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace {

std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        LOG_INFO("Shutdown signal received");
        g_shutdown_requested.store(true, std::memory_order_release);
    }
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n\n"
              << "Options:\n"
              << "  --config <path>    Path to configuration file (default: config/default.yaml)\n"
              << "  --help             Show this help message\n"
              << "  --version          Show version information\n"
              << "\n"
              << "Configuration can also be overridden via command line:\n"
              << "  --capture.source=gstreamer\n"
              << "  --detection.enabled=true\n"
              << "  --display.enabled=true\n"
              << "  --recording.enabled=true\n"
              << std::endl;
}

void print_version() {
    std::cout << "Lagari Vision System v1.0.0\n"
              << "Built for: "
#if defined(PLATFORM_JETSON)
              << "NVIDIA Jetson"
#elif defined(PLATFORM_RPI)
              << "Raspberry Pi"
#else
              << "x86/x64"
#endif
              << "\n"
              << "Build type: "
#ifdef NDEBUG
              << "Release"
#else
              << "Debug"
#endif
              << std::endl;
}

}  // namespace

// ============================================================================
// Processing Thread
// ============================================================================

/**
 * @brief Processing thread function
 * 
 * This thread receives frames from the capture module, runs detection,
 * and pushes results to display and recording modules.
 */
void processing_thread(
    lagari::ICapture* capture,
    lagari::IDetector* detector,
    lagari::IDisplay* display,
    lagari::IRecorder* recorder,
    lagari::SystemStateMachine& state,
    std::atomic<bool>& shutdown)
{
    using namespace lagari;
    
    LOG_INFO("Processing thread started");
    
    const auto target_fps = 30;
    const auto frame_interval = std::chrono::milliseconds(1000 / target_fps);
    
    uint64_t frames_processed = 0;
    uint64_t detections_count = 0;
    auto last_stats_time = Clock::now();
    
    // Detection result storage
    DetectionResult latest_detections;
    
    while (!shutdown.load(std::memory_order_acquire)) {
        auto loop_start = Clock::now();
        
        // Get latest frame from capture
        FramePtr frame = capture->wait_for_frame(100);  // 100ms timeout
        
        if (!frame || !frame->valid()) {
            // No frame available, check if capture is still running
            if (!capture->is_running()) {
                LOG_WARN("Capture stopped, processing thread exiting");
                break;
            }
            continue;
        }
        
        // Calculate capture latency
        Duration capture_latency = Clock::now() - frame->metadata.timestamp;
        SystemState current_state = state.state();
        
        // ====================================================================
        // Detection
        // ====================================================================
        const DetectionResult* detections_ptr = nullptr;
        
        if (detector && detector->is_running()) {
            PERF_SCOPE("detection.total");
            auto detect_start = Clock::now();
            
            // Run synchronous detection
            latest_detections = detector->detect(*frame);
            
            auto detect_time = Clock::now() - detect_start;
            
            if (!latest_detections.detections.empty()) {
                detections_count += latest_detections.detections.size();
                
                // Update state based on detections
                if (current_state == SystemState::SEARCHING) {
                    state.set_state(SystemState::TRACKING);
                }
            } else {
                // No detections - might want to transition back to SEARCHING
                if (current_state == SystemState::TRACKING) {
                    // Could add a timeout here before going back to SEARCHING
                }
            }
            
            detections_ptr = &latest_detections;
            
            // Log detection details at debug level
            if (!latest_detections.detections.empty()) {
                LOG_DEBUG("Detected {} objects, inference={:.1f}ms",
                          latest_detections.detections.size(),
                          std::chrono::duration<float, std::milli>(detect_time).count());
            }
        }
        
        // Total latency including detection
        Duration total_latency = Clock::now() - frame->metadata.timestamp;
        
        // ====================================================================
        // Display
        // ====================================================================
        if (display && display->is_enabled()) {
            PERF_SCOPE("display.push");
            display->push_frame(*frame, detections_ptr, current_state, total_latency);
        }
        
        // ====================================================================
        // Recording
        // ====================================================================
        if (recorder && recorder->is_recording()) {
            PERF_SCOPE("recording.add_frame");
            recorder->add_frame(*frame, detections_ptr, current_state, total_latency);
        }
        
        frames_processed++;
        
        // ====================================================================
        // Statistics (every 10 seconds)
        // ====================================================================
        auto now = Clock::now();
        if (now - last_stats_time > std::chrono::seconds(10)) {
            auto elapsed = std::chrono::duration<float>(now - last_stats_time).count();
            float fps = frames_processed / elapsed;
            
            LOG_INFO("Processing: fps={:.1f}, latency={:.1f}ms, detections={}",
                     fps,
                     std::chrono::duration<float, std::milli>(total_latency).count(),
                     detections_count);
            
            // Capture stats
            auto capture_stats = capture->get_stats();
            LOG_DEBUG("Capture: frames={}, dropped={}, fps={:.1f}",
                      capture_stats.frames_captured, capture_stats.frames_dropped,
                      capture_stats.current_fps);
            
            // Detection stats
            if (detector && detector->is_running()) {
                auto det_stats = detector->get_stats();
                LOG_DEBUG("Detection: frames={}, avg_time={:.1f}ms, fps={:.1f}",
                          det_stats.frames_processed,
                          std::chrono::duration<float, std::milli>(det_stats.average_inference_time).count(),
                          det_stats.inference_fps);
            }
            
            // Display stats
            if (display && display->is_enabled()) {
                auto display_stats = display->get_stats();
                LOG_DEBUG("Display: frames={}, dropped={}, fps={:.1f}",
                          display_stats.frames_displayed, display_stats.frames_dropped,
                          display_stats.current_fps);
            }
            
            // Recording stats
            if (recorder && recorder->is_recording()) {
                LOG_DEBUG("Recording: duration={:.1f}s, bytes={}",
                          recorder->recording_duration(), recorder->bytes_written());
            }
            
            frames_processed = 0;
            detections_count = 0;
            last_stats_time = now;
            
            // Log profiler summary periodically
            PERF_LOG_SUMMARY();
            PERF_RESET();
        }
        
        // Rate limiting (if processing is faster than capture)
        auto loop_duration = Clock::now() - loop_start;
        if (loop_duration < frame_interval) {
            std::this_thread::sleep_for(frame_interval - loop_duration);
        }
    }
    
    LOG_INFO("Processing thread exiting");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    using namespace lagari;

    // Parse command line for --help and --version first
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--version" || arg == "-v") {
            print_version();
            return 0;
        }
    }

    // Initialize logging
    if (!Logger::init("", LogLevel::DEBUG, LogLevel::DEBUG)) {
        std::cerr << "Failed to initialize logging" << std::endl;
        return 1;
    }

    LOG_INFO("=== Lagari Vision System Starting ===");
    print_version();

    // Install signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Load configuration
    Config& config = global_config();
    
    std::string config_path = "config/default.yaml";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    if (!config.load(config_path)) {
        LOG_ERROR("Failed to load configuration from: {}", config_path);
        return 1;
    }

    // Apply command-line overrides
    config.parse_args(argc, argv);

    LOG_INFO("Configuration loaded from: {}", config_path);

    // Log available inference backends
    auto backends = available_backends();
    if (backends.empty()) {
        LOG_WARN("No inference backends available - detection disabled");
    } else {
        LOG_INFO("Available inference backends: {}", backends.size());
        for (auto& backend : backends) {
            LOG_INFO("  - {}", to_string(backend));
        }
    }

    // Initialize state machine
    SystemStateMachine& state = global_state();
    state.on_state_change([](SystemState old_state, SystemState new_state) {
        LOG_INFO("State transition: {} -> {}", to_string(old_state), to_string(new_state));
    });

    // ========================================================================
    // Create Modules
    // ========================================================================
    
    LOG_INFO("Creating modules...");

    // Capture module (required)
    auto capture = create_capture(config);
    if (!capture) {
        LOG_ERROR("Failed to create capture module");
        return 1;
    }

    // Detection module (optional, depends on model path and backends)
    std::unique_ptr<IDetector> detector;
    bool detection_enabled = config.get_bool("detection.enabled", true);
    std::string model_path = config.get_string("detection.model_path", "");
    
    if (detection_enabled && !model_path.empty() && !backends.empty()) {
        detector = create_detector(config);
        if (!detector) {
            LOG_WARN("Failed to create detector - continuing without detection");
        }
    } else if (detection_enabled && model_path.empty()) {
        LOG_WARN("Detection enabled but no model_path specified - skipping detection");
    }

    // Display module (optional)
    auto display = create_display(config);
    
    // Recorder module (optional)
    auto recorder = create_recorder(config);

    // ========================================================================
    // Initialize Modules
    // ========================================================================
    
    LOG_INFO("Initializing modules...");

    if (!capture->initialize(config)) {
        LOG_ERROR("Failed to initialize capture module");
        return 1;
    }
    LOG_INFO("Capture module initialized: {}", capture->name());

    if (detector) {
        // Detector is initialized in factory, just log
        LOG_INFO("Detector module initialized: backend={}", to_string(detector->backend()));
    }

    if (display) {
        if (!display->initialize(config)) {
            LOG_WARN("Failed to initialize display module, continuing without display");
            display.reset();
        } else {
            LOG_INFO("Display module initialized: {}", display->name());
        }
    }

    if (recorder) {
        if (!recorder->initialize(config)) {
            LOG_WARN("Failed to initialize recorder module, continuing without recording");
            recorder.reset();
        } else {
            LOG_INFO("Recorder module initialized: {}", recorder->name());
        }
    }

    // ========================================================================
    // Start Modules
    // ========================================================================
    
    LOG_INFO("Starting modules...");
    
    state.set_state(SystemState::IDLE);

    // Start capture
    capture->start();
    if (!capture->is_running()) {
        LOG_ERROR("Capture module failed to start");
        return 1;
    }

    // Start detector
    if (detector) {
        detector->start();
        LOG_INFO("Detector started");
    }

    // Start display
    if (display) {
        display->start();
    }

    // Start recorder
    if (recorder) {
        recorder->start();
        
        // Auto-start recording if enabled
        if (config.get_bool("recording.enabled", false)) {
            if (recorder->start_recording()) {
                LOG_INFO("Recording started automatically");
            }
        }
    }

    // ========================================================================
    // Start Processing Thread
    // ========================================================================
    
    LOG_INFO("Starting processing thread...");
    
    std::thread proc_thread(
        processing_thread,
        capture.get(),
        detector.get(),
        display.get(),
        recorder.get(),
        std::ref(state),
        std::ref(g_shutdown_requested)
    );

    state.set_state(SystemState::SEARCHING);

    // ========================================================================
    // Main Loop (Monitoring)
    // ========================================================================
    
    LOG_INFO("Entering main loop");
    LOG_INFO("Press Ctrl+C to stop");

    while (!g_shutdown_requested.load(std::memory_order_acquire)) {
        // Main loop runs at low frequency for monitoring
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Check module health
        if (!capture->is_running()) {
            LOG_ERROR("Capture module stopped unexpectedly");
            g_shutdown_requested.store(true, std::memory_order_release);
            break;
        }
        
        // Periodic status log
        static auto last_log = Clock::now();
        auto now = Clock::now();
        if (now - last_log > std::chrono::seconds(30)) {
            auto capture_stats = capture->get_stats();
            LOG_INFO("System status: state={}, capture_fps={:.1f}",
                     state.state_string(), capture_stats.current_fps);
            
            if (detector && detector->is_running()) {
                auto det_stats = detector->get_stats();
                LOG_INFO("Detection status: inference_fps={:.1f}, frames={}",
                         det_stats.inference_fps, det_stats.frames_processed);
            }
            
            last_log = now;
        }
    }

    // ========================================================================
    // Shutdown
    // ========================================================================
    
    LOG_INFO("Shutting down...");
    state.set_state(SystemState::SHUTDOWN);

    // Signal processing thread to stop
    g_shutdown_requested.store(true, std::memory_order_release);
    
    // Wait for processing thread
    if (proc_thread.joinable()) {
        proc_thread.join();
    }
    LOG_INFO("Processing thread stopped");

    // Stop modules in reverse order
    if (recorder) {
        recorder->stop_recording();
        recorder->stop();
        LOG_INFO("Recorder stopped");
    }

    if (display) {
        display->stop();
        LOG_INFO("Display stopped");
    }

    if (detector) {
        detector->stop();
        LOG_INFO("Detector stopped");
    }

    capture->stop();
    LOG_INFO("Capture stopped");

    Logger::flush();
    Logger::shutdown();

    std::cout << "Lagari Vision System stopped." << std::endl;
    return 0;
}
