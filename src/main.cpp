#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"
#include "lagari/core/system_state.hpp"
#include "lagari/core/types.hpp"
#include "lagari/core/ring_buffer.hpp"
#include "lagari/core/spsc_queue.hpp"

#include "lagari/capture/capture.hpp"
#include "lagari/detection/detector.hpp"
#include "lagari/qr/qr_decoder.hpp"
#include "lagari/guidance/guidance.hpp"
#include "lagari/comms/mavlink_interface.hpp"
#include "lagari/comms/telemetry.hpp"
#include "lagari/recording/recorder.hpp"

#include <csignal>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

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
              << "  --capture.source=usb\n"
              << "  --detection.confidence_threshold=0.6\n"
              << "  --guidance.enabled=false\n"
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

    // Initialize state machine
    SystemStateMachine& state = global_state();
    state.on_state_change([](SystemState old_state, SystemState new_state) {
        LOG_INFO("State transition: {} -> {}", to_string(old_state), to_string(new_state));
    });

    // ========================================================================
    // Create data structures for inter-module communication
    // ========================================================================
    
    // Frame buffer: camera -> consumers (detection, recording, telemetry)
    SPMCRingBuffer<FramePtr, 4> frame_buffer;
    
    // Detection queue: detector -> guidance, QR decoder
    SPSCQueue<DetectionResult, 16> detection_queue;
    
    // QR queue: decoder -> telemetry
    SPSCQueue<QRResult, 8> qr_queue;
    
    // Command queue: guidance -> MAVLink
    SPSCQueue<GuidanceCommand, 32> command_queue;

    // ========================================================================
    // Initialize modules
    // ========================================================================
    
    LOG_INFO("Initializing modules...");

    // TODO: Create and initialize actual module implementations
    // For now, we just demonstrate the structure
    
    /*
    auto capture = create_capture(config);
    auto detector = create_detector(config);
    auto qr_decoder = create_qr_decoder(config);
    auto guidance = create_guidance(config);
    auto mavlink = create_mavlink(config);
    auto telemetry = create_telemetry(config);
    auto recorder = create_recorder(config);

    if (!capture || !capture->initialize(config)) {
        LOG_ERROR("Failed to initialize capture module");
        return 1;
    }

    if (!detector || !detector->initialize(config)) {
        LOG_ERROR("Failed to initialize detection module");
        return 1;
    }

    // ... initialize other modules
    */

    // ========================================================================
    // Start modules
    // ========================================================================
    
    LOG_INFO("Starting modules...");
    
    state.set_state(SystemState::IDLE);

    // TODO: Start module threads
    
    /*
    capture->start();
    detector->start();
    guidance->start();
    mavlink->start();
    telemetry->start();
    recorder->start();
    */

    // ========================================================================
    // Main loop
    // ========================================================================
    
    LOG_INFO("Entering main loop");
    
    if (config.get_bool("system.state_machine.auto_start", false)) {
        state.set_state(SystemState::SEARCHING);
    }

    while (!g_shutdown_requested.load(std::memory_order_acquire)) {
        // Main loop runs at low frequency for monitoring
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // TODO: 
        // - Monitor module health
        // - Handle state transitions
        // - Log performance metrics
        // - Check for configuration updates
        
        // Example: Log periodic status
        static auto last_log = Clock::now();
        auto now = Clock::now();
        if (now - last_log > std::chrono::seconds(10)) {
            LOG_INFO("System status: state={}, frame_buffer_size={}",
                     state.state_string(), frame_buffer.size());
            last_log = now;
        }
    }

    // ========================================================================
    // Shutdown
    // ========================================================================
    
    LOG_INFO("Shutting down...");
    state.set_state(SystemState::SHUTDOWN);

    // TODO: Stop modules in reverse order
    /*
    recorder->stop();
    telemetry->stop();
    mavlink->stop();
    guidance->stop();
    detector->stop();
    capture->stop();
    */

    Logger::flush();
    Logger::shutdown();

    std::cout << "Lagari Vision System stopped." << std::endl;
    return 0;
}
