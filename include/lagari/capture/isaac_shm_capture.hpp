#pragma once

#include "lagari/capture/capture.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <string>

namespace lagari {

/**
 * @brief Isaac Sim camera capture via POSIX shared memory
 * 
 * Ultra-low-latency frame capture from Isaac Sim using shared memory
 * for zero-copy data transfer. This is the highest performance option
 * for local simulation.
 * 
 * Shared memory layout (64-byte header + frame data):
 * 
 *   Offset  Size  Field
 *   ------  ----  -----
 *   0       4     Magic ("LGRV")
 *   4       4     Version (1)
 *   8       4     Width
 *   12      4     Height
 *   16      4     Format (PixelFormat enum)
 *   20      8     Frame ID
 *   28      8     Timestamp (nanoseconds)
 *   36      8     Write sequence (atomic)
 *   44      8     Read sequence (atomic)
 *   52      12    Reserved
 *   64      W*H*3 BGR24 pixel data
 * 
 * Lock-free protocol using sequence numbers:
 * - Writer increments write_seq to odd before writing
 * - Writer increments write_seq to even after writing
 * - Reader only reads when write_seq is even and > read_seq
 * - Reader sets read_seq = write_seq after reading
 */
class IsaacShmCapture : public ICapture {
public:
    /**
     * @brief Shared memory specific configuration
     */
    struct ShmConfig {
        std::string shm_name = "lagari_camera";
        int poll_interval_us = 100;    // Microseconds between polls
        bool auto_reconnect = true;
        int reconnect_delay_ms = 1000;
    };

    explicit IsaacShmCapture(const CaptureConfig& config);
    IsaacShmCapture(const CaptureConfig& config, const ShmConfig& shm_config);
    ~IsaacShmCapture() override;

    // Non-copyable
    IsaacShmCapture(const IsaacShmCapture&) = delete;
    IsaacShmCapture& operator=(const IsaacShmCapture&) = delete;

    // IModule interface
    bool initialize(const Config& config) override;
    void start() override;
    void stop() override;
    bool is_running() const override;
    std::string name() const override { return "IsaacShmCapture"; }

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

    // SHM-specific methods
    
    /**
     * @brief Attach to shared memory segment
     * @param shm_name Name of shared memory (e.g., "lagari_camera")
     * @return true if attached successfully
     */
    bool attach(const std::string& shm_name = "");

    /**
     * @brief Detach from shared memory
     */
    void detach();

    /**
     * @brief Check if attached to shared memory
     */
    bool is_attached() const;

    /**
     * @brief Get shared memory configuration
     */
    const ShmConfig& shm_config() const { return shm_config_; }

    /**
     * @brief Get current latency (time since frame was captured)
     */
    Duration get_frame_latency() const;

private:
    // Shared memory header structure (must match Python side)
    static constexpr uint32_t MAGIC = 0x5652474C;  // "LGRV" little-endian
    static constexpr uint32_t VERSION = 1;
    static constexpr size_t HEADER_SIZE = 64;

#pragma pack(push, 1)
    struct ShmHeader {
        uint32_t magic;
        uint32_t version;
        uint32_t width;
        uint32_t height;
        int32_t format;
        uint64_t frame_id;
        uint64_t timestamp_ns;
        uint64_t write_seq;
        uint64_t read_seq;
        uint8_t reserved[12];
    };
#pragma pack(pop)

    static_assert(sizeof(ShmHeader) == HEADER_SIZE, "ShmHeader must be 64 bytes");

    // Read loop
    void read_loop();
    bool try_read_frame();
    FramePtr process_shm_data(const ShmHeader& header, const uint8_t* data);

    // Configuration
    CaptureConfig config_;
    ShmConfig shm_config_;

    // Shared memory state
    int shm_fd_ = -1;
    void* shm_ptr_ = nullptr;
    size_t shm_size_ = 0;
    std::atomic<bool> attached_{false};

    // Threading
    std::thread read_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};

    // Frame storage
    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    FramePtr latest_frame_;
    uint64_t last_frame_id_ = 0;
    uint64_t last_write_seq_ = 0;

    // Callback
    FrameCallback frame_callback_;

    // Statistics
    mutable std::mutex stats_mutex_;
    CaptureStats stats_;
    TimePoint start_time_;
    TimePoint last_frame_time_;
    uint64_t frames_skipped_ = 0;
};

}  // namespace lagari
