#pragma once

#include "lagari/capture/capture.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <string>

// Forward declare ZMQ types to avoid header dependency
namespace zmq {
    class context_t;
    class socket_t;
}

namespace lagari {

/**
 * @brief Isaac Sim camera capture via ZeroMQ
 * 
 * Receives camera frames streamed from Isaac Sim via ZeroMQ.
 * The Isaac Sim side runs a Python script that publishes frames.
 * 
 * Frame protocol:
 * - Topic: "frame" (for ZMQ subscription filtering)
 * - Header: frame_id(8) + timestamp_ns(8) + width(4) + height(4) + format(4)
 * - Data: Raw pixel data (BGR24)
 */
class IsaacCapture : public ICapture {
public:
    /**
     * @brief Isaac Sim specific configuration
     */
    struct IsaacConfig {
        std::string zmq_endpoint = "tcp://localhost:5555";
        int zmq_recv_timeout_ms = 100;
        int zmq_hwm = 2;  // High water mark (buffer size)
        bool reconnect_on_timeout = true;
    };

    explicit IsaacCapture(const CaptureConfig& config);
    IsaacCapture(const CaptureConfig& config, const IsaacConfig& isaac_config);
    ~IsaacCapture() override;

    // Non-copyable
    IsaacCapture(const IsaacCapture&) = delete;
    IsaacCapture& operator=(const IsaacCapture&) = delete;

    // IModule interface
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override;
    std::string name() const override { return "IsaacCapture"; }

    // ICapture interface
    FramePtr get_latest_frame() override;
    FramePtr wait_for_frame(uint32_t timeout_ms) override;
    void set_frame_callback(FrameCallback callback) override;
    CaptureStats get_stats() const override;
    bool is_open() const override;
    const CaptureConfig& config() const override { return config_; }
    bool set_resolution(uint32_t width, uint32_t height) override;
    bool set_framerate(uint32_t fps) override;
    bool set_exposure(bool auto_exp, float exposure_time) override;

    // Isaac-specific methods
    
    /**
     * @brief Connect to Isaac Sim ZMQ publisher
     * 
     * @param endpoint ZMQ endpoint (e.g., "tcp://localhost:5555")
     * @return true if connected
     */
    bool connect(const std::string& endpoint = "");

    /**
     * @brief Disconnect from Isaac Sim
     */
    void disconnect();

    /**
     * @brief Check if connected to Isaac Sim
     */
    bool is_connected() const;

    /**
     * @brief Get Isaac-specific configuration
     */
    const IsaacConfig& isaac_config() const { return isaac_config_; }

private:
    // Frame header structure (must match Python side)
#pragma pack(push, 1)
    struct FrameHeader {
        uint64_t frame_id;
        uint64_t timestamp_ns;
        uint32_t width;
        uint32_t height;
        int32_t format;
    };
#pragma pack(pop)

    static_assert(sizeof(FrameHeader) == 28, "FrameHeader must be 28 bytes");

    // ZMQ receive loop
    void receive_loop();
    bool receive_frame();
    FramePtr process_message(const void* header_data, size_t header_size,
                            const void* frame_data, size_t frame_size);

    // Configuration
    CaptureConfig config_;
    IsaacConfig isaac_config_;

    // ZMQ state
    std::unique_ptr<zmq::context_t> zmq_context_;
    std::unique_ptr<zmq::socket_t> zmq_socket_;
    std::atomic<bool> connected_{false};

    // Threading
    std::thread receive_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};

    // Frame storage
    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    FramePtr latest_frame_;
    uint64_t frame_counter_ = 0;

    // Callback
    FrameCallback frame_callback_;

    // Statistics
    mutable std::mutex stats_mutex_;
    CaptureStats stats_;
    TimePoint start_time_;
    TimePoint last_frame_time_;
    uint64_t timeout_count_ = 0;
};

}  // namespace lagari
