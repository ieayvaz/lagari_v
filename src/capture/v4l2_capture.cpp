#include "lagari/capture/v4l2_capture.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"
#include "lagari/core/profiler.hpp"

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <algorithm>

namespace lagari {

namespace {

// Helper for ioctl with retry on EINTR
int xioctl(int fd, unsigned long request, void* arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

// Convert V4L2 pixel format to our format
PixelFormat v4l2_to_pixel_format(uint32_t v4l2_fmt) {
    switch (v4l2_fmt) {
        case V4L2_PIX_FMT_YUYV:
            return PixelFormat::YUYV;
        case V4L2_PIX_FMT_MJPEG:
        case V4L2_PIX_FMT_BGR24:
            return PixelFormat::BGR24;
        case V4L2_PIX_FMT_RGB24:
            return PixelFormat::RGB24;
        case V4L2_PIX_FMT_NV12:
            return PixelFormat::NV12;
        case V4L2_PIX_FMT_GREY:
            return PixelFormat::GRAY8;
        default:
            return PixelFormat::UNKNOWN;
    }
}

// Get preferred V4L2 format
uint32_t get_preferred_format(int fd) {
    // Preference order: YUYV, MJPEG, RGB24, NV12
    const uint32_t preferred[] = {
        V4L2_PIX_FMT_YUYV,
        V4L2_PIX_FMT_MJPEG,
        V4L2_PIX_FMT_RGB24,
        V4L2_PIX_FMT_BGR24,
        V4L2_PIX_FMT_NV12
    };

    struct v4l2_fmtdesc fmtdesc{};
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    
    std::vector<uint32_t> supported;
    while (xioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
        supported.push_back(fmtdesc.pixelformat);
        fmtdesc.index++;
    }

    for (uint32_t pref : preferred) {
        if (std::find(supported.begin(), supported.end(), pref) != supported.end()) {
            return pref;
        }
    }

    return supported.empty() ? V4L2_PIX_FMT_YUYV : supported[0];
}

// YUYV to BGR24 conversion
void yuyv_to_bgr(const uint8_t* yuyv, uint8_t* bgr, int width, int height) {
    for (int i = 0; i < width * height / 2; i++) {
        int y0 = yuyv[0];
        int u  = yuyv[1];
        int y1 = yuyv[2];
        int v  = yuyv[3];
        yuyv += 4;

        // Clamp helper
        auto clamp = [](int val) -> uint8_t {
            return static_cast<uint8_t>(std::max(0, std::min(255, val)));
        };

        // YUV to BGR
        int c0 = y0 - 16;
        int c1 = y1 - 16;
        int d = u - 128;
        int e = v - 128;

        // First pixel
        bgr[0] = clamp((298 * c0 + 516 * d + 128) >> 8);           // B
        bgr[1] = clamp((298 * c0 - 100 * d - 208 * e + 128) >> 8); // G
        bgr[2] = clamp((298 * c0 + 409 * e + 128) >> 8);           // R

        // Second pixel
        bgr[3] = clamp((298 * c1 + 516 * d + 128) >> 8);           // B
        bgr[4] = clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8); // G
        bgr[5] = clamp((298 * c1 + 409 * e + 128) >> 8);           // R

        bgr += 6;
    }
}

}  // namespace

// ============================================================================
// Constructor / Destructor
// ============================================================================

V4L2Capture::V4L2Capture(const CaptureConfig& config)
    : config_(config)
{
    // Determine device path
    if (!config_.device.empty()) {
        device_path_ = config_.device;
    } else {
        device_path_ = "/dev/video" + std::to_string(config_.camera_id);
    }
}

V4L2Capture::~V4L2Capture() {
    stop();
    close_device();
}

// ============================================================================
// IModule Implementation
// ============================================================================

bool V4L2Capture::initialize(const Config& /* config */) {
    if (!open_device()) {
        return false;
    }

    if (!init_device()) {
        close_device();
        return false;
    }

    if (!init_mmap()) {
        close_device();
        return false;
    }

    LOG_INFO("V4L2Capture initialized: {} ({}x{} @ {} fps)",
             device_path_, actual_width_, actual_height_, actual_fps_);

    return true;
}

void V4L2Capture::start() {
    if (running_.load(std::memory_order_acquire)) {
        return;
    }

    should_stop_.store(false, std::memory_order_release);

    if (!start_capturing()) {
        LOG_ERROR("Failed to start V4L2 streaming");
        return;
    }

    start_time_ = Clock::now();
    last_frame_time_ = start_time_;

    capture_thread_ = std::thread(&V4L2Capture::capture_loop, this);
    running_.store(true, std::memory_order_release);

    LOG_INFO("V4L2Capture started");
}

void V4L2Capture::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    should_stop_.store(true, std::memory_order_release);
    frame_cv_.notify_all();

    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }

    stop_capturing();
    running_.store(false, std::memory_order_release);

    LOG_INFO("V4L2Capture stopped");
}

bool V4L2Capture::is_running() const {
    return running_.load(std::memory_order_acquire);
}

// ============================================================================
// ICapture Implementation
// ============================================================================

FramePtr V4L2Capture::get_latest_frame() {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

FramePtr V4L2Capture::wait_for_frame(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(frame_mutex_);
    
    if (latest_frame_) {
        return latest_frame_;
    }

    frame_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() {
        return latest_frame_ != nullptr || should_stop_.load(std::memory_order_acquire);
    });

    return latest_frame_;
}

void V4L2Capture::set_frame_callback(FrameCallback callback) {
    frame_callback_ = std::move(callback);
}

CaptureStats V4L2Capture::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool V4L2Capture::is_open() const {
    return fd_ >= 0;
}

bool V4L2Capture::set_resolution(uint32_t width, uint32_t height) {
    // Can only change resolution when stopped
    if (running_.load(std::memory_order_acquire)) {
        LOG_WARN("Cannot change resolution while running");
        return false;
    }

    config_.width = width;
    config_.height = height;

    // Reinitialize device
    close_device();
    if (!open_device() || !init_device() || !init_mmap()) {
        LOG_ERROR("Failed to reinitialize with new resolution");
        return false;
    }

    return true;
}

bool V4L2Capture::set_framerate(uint32_t fps) {
    if (fd_ < 0) return false;

    struct v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps;

    if (xioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
        LOG_WARN("Failed to set framerate: {}", strerror(errno));
        return false;
    }

    actual_fps_ = parm.parm.capture.timeperframe.denominator /
                  parm.parm.capture.timeperframe.numerator;

    LOG_INFO("Framerate set to {} fps", actual_fps_);
    return true;
}

bool V4L2Capture::set_exposure(bool auto_exp, float exposure_time) {
    if (fd_ < 0) return false;

    struct v4l2_control ctrl{};
    
    // Set auto exposure mode
    ctrl.id = V4L2_CID_EXPOSURE_AUTO;
    ctrl.value = auto_exp ? V4L2_EXPOSURE_AUTO : V4L2_EXPOSURE_MANUAL;
    
    if (xioctl(fd_, VIDIOC_S_CTRL, &ctrl) < 0) {
        LOG_WARN("Failed to set auto exposure: {}", strerror(errno));
        return false;
    }

    if (!auto_exp && exposure_time > 0) {
        ctrl.id = V4L2_CID_EXPOSURE_ABSOLUTE;
        ctrl.value = static_cast<int>(exposure_time * 10000);  // Convert to 100µs units
        
        if (xioctl(fd_, VIDIOC_S_CTRL, &ctrl) < 0) {
            LOG_WARN("Failed to set exposure time: {}", strerror(errno));
            return false;
        }
    }

    return true;
}

// ============================================================================
// Device Management
// ============================================================================

bool V4L2Capture::open_device() {
    struct stat st;
    if (stat(device_path_.c_str(), &st) < 0) {
        LOG_ERROR("Cannot identify '{}': {}", device_path_, strerror(errno));
        return false;
    }

    if (!S_ISCHR(st.st_mode)) {
        LOG_ERROR("'{}' is not a device", device_path_);
        return false;
    }

    fd_ = open(device_path_.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd_ < 0) {
        LOG_ERROR("Cannot open '{}': {}", device_path_, strerror(errno));
        return false;
    }

    return true;
}

void V4L2Capture::close_device() {
    // Unmap buffers
    for (auto& buffer : buffers_) {
        if (buffer.start != MAP_FAILED && buffer.start != nullptr) {
            munmap(buffer.start, buffer.length);
        }
    }
    buffers_.clear();

    // Close file descriptor
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

bool V4L2Capture::init_device() {
    // Query capabilities
    struct v4l2_capability cap{};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        LOG_ERROR("VIDIOC_QUERYCAP failed: {}", strerror(errno));
        return false;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        LOG_ERROR("Device does not support video capture");
        return false;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        LOG_ERROR("Device does not support streaming");
        return false;
    }

    LOG_DEBUG("Device: {} (driver: {}, bus: {})",
              reinterpret_cast<char*>(cap.card),
              reinterpret_cast<char*>(cap.driver),
              reinterpret_cast<char*>(cap.bus_info));

    // Set format
    struct v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = config_.width;
    fmt.fmt.pix.height = config_.height;
    fmt.fmt.pix.pixelformat = get_preferred_format(fd_);
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        LOG_ERROR("VIDIOC_S_FMT failed: {}", strerror(errno));
        return false;
    }

    // Store actual values (may differ from requested)
    actual_width_ = fmt.fmt.pix.width;
    actual_height_ = fmt.fmt.pix.height;
    pixel_format_ = fmt.fmt.pix.pixelformat;

    char fourcc[5] = {0};
    memcpy(fourcc, &pixel_format_, 4);
    LOG_DEBUG("Format: {}x{}, fourcc: {}", actual_width_, actual_height_, fourcc);

    // Set framerate
    struct v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = config_.fps;

    if (xioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
        LOG_WARN("VIDIOC_S_PARM failed: {}", strerror(errno));
    }

    actual_fps_ = parm.parm.capture.timeperframe.denominator /
                  std::max(1u, parm.parm.capture.timeperframe.numerator);

    return true;
}

bool V4L2Capture::init_mmap() {
    struct v4l2_requestbuffers req{};
    req.count = config_.buffer_count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        LOG_ERROR("VIDIOC_REQBUFS failed: {}", strerror(errno));
        return false;
    }

    if (req.count < 2) {
        LOG_ERROR("Insufficient buffer memory");
        return false;
    }

    buffers_.resize(req.count);

    for (size_t i = 0; i < buffers_.size(); ++i) {
        struct v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = static_cast<uint32_t>(i);

        if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            LOG_ERROR("VIDIOC_QUERYBUF failed: {}", strerror(errno));
            return false;
        }

        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(
            nullptr, buf.length,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd_, buf.m.offset
        );

        if (buffers_[i].start == MAP_FAILED) {
            LOG_ERROR("mmap failed: {}", strerror(errno));
            return false;
        }
    }

    LOG_DEBUG("Allocated {} buffers", buffers_.size());
    return true;
}

bool V4L2Capture::start_capturing() {
    // Queue all buffers
    for (size_t i = 0; i < buffers_.size(); ++i) {
        struct v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = static_cast<uint32_t>(i);

        if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            LOG_ERROR("VIDIOC_QBUF failed: {}", strerror(errno));
            return false;
        }
    }

    // Start streaming
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        LOG_ERROR("VIDIOC_STREAMON failed: {}", strerror(errno));
        return false;
    }

    return true;
}

bool V4L2Capture::stop_capturing() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMOFF, &type) < 0) {
        LOG_WARN("VIDIOC_STREAMOFF failed: {}", strerror(errno));
        return false;
    }
    return true;
}

// ============================================================================
// Capture Loop
// ============================================================================

void V4L2Capture::capture_loop() {
    LOG_DEBUG("Capture thread started");

    while (!should_stop_.load(std::memory_order_acquire)) {
        // Use select for timeout
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);

        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000;  // 100ms timeout

        int r = select(fd_ + 1, &fds, nullptr, nullptr, &tv);

        if (r < 0) {
            if (errno == EINTR) continue;
            LOG_ERROR("select error: {}", strerror(errno));
            break;
        }

        if (r == 0) {
            // Timeout, check if we should stop
            continue;
        }

        if (!read_frame()) {
            // Non-fatal read error, continue
            continue;
        }
    }

    LOG_DEBUG("Capture thread exiting");
}

bool V4L2Capture::read_frame() {
    struct v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) {
            return false;  // No frame available yet
        }
        LOG_ERROR("VIDIOC_DQBUF failed: {}", strerror(errno));
        return false;
    }

    // Process the frame
    process_frame(buffers_[buf.index].start, buf.bytesused);

    // Re-queue the buffer
    if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        LOG_ERROR("VIDIOC_QBUF failed: {}", strerror(errno));
        return false;
    }

    return true;
}

void V4L2Capture::process_frame(const void* data, size_t size) {
    PERF_SCOPE("capture.process");
    auto now = Clock::now();
    
    // Convert to our Frame format
    FramePtr frame = convert_frame(data, size);
    if (!frame) {
        return;
    }

    // Update frame metadata
    frame->metadata.frame_id = ++frame_counter_;
    frame->metadata.timestamp = now;

    // Store latest frame
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = frame;
    }
    frame_cv_.notify_all();

    // Call callback if set
    if (frame_callback_) {
        frame_callback_(frame);
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.frames_captured++;
        
        auto elapsed = std::chrono::duration<float>(now - start_time_).count();
        if (elapsed > 0) {
            stats_.average_fps = stats_.frames_captured / elapsed;
        }

        auto frame_time = std::chrono::duration<float>(now - last_frame_time_).count();
        if (frame_time > 0) {
            stats_.current_fps = 1.0f / frame_time;
        }

        stats_.average_latency = std::chrono::duration_cast<Duration>(
            std::chrono::microseconds(static_cast<int64_t>(1000000.0f / stats_.current_fps))
        );
    }

    last_frame_time_ = now;
}

FramePtr V4L2Capture::convert_frame(const void* data, size_t size) {
    PERF_SCOPE("capture.convert");
    // Create output frame in BGR24 format
    auto frame = std::make_shared<Frame>(actual_width_, actual_height_, PixelFormat::BGR24);

    switch (pixel_format_) {
        case V4L2_PIX_FMT_YUYV: {
            // Convert YUYV to BGR24
            yuyv_to_bgr(
                static_cast<const uint8_t*>(data),
                frame->ptr(),
                actual_width_,
                actual_height_
            );
            break;
        }

        case V4L2_PIX_FMT_BGR24: {
            // Direct copy
            size_t expected = actual_width_ * actual_height_ * 3;
            if (size >= expected) {
                memcpy(frame->ptr(), data, expected);
            }
            break;
        }

        case V4L2_PIX_FMT_RGB24: {
            // Swap R and B channels
            const uint8_t* src = static_cast<const uint8_t*>(data);
            uint8_t* dst = frame->ptr();
            for (uint32_t i = 0; i < actual_width_ * actual_height_; ++i) {
                dst[0] = src[2];  // B
                dst[1] = src[1];  // G
                dst[2] = src[0];  // R
                src += 3;
                dst += 3;
            }
            break;
        }

        case V4L2_PIX_FMT_MJPEG: {
            // TODO: Decode MJPEG using libjpeg or OpenCV
            LOG_WARN("MJPEG decoding not implemented, using placeholder");
            memset(frame->ptr(), 128, frame->size());
            break;
        }

        default: {
            LOG_WARN("Unsupported pixel format: 0x{:08X}", pixel_format_);
            return nullptr;
        }
    }

    return frame;
}

}  // namespace lagari
