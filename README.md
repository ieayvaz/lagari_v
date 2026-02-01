# Lagari Vision System

Multi-platform UAV vision system for real-time object detection and tracking, built in C++17 for NVIDIA Jetson, Raspberry Pi, and x86 platforms.

## Features

- **Real-time Object Detection**: YOLO-based detection with multiple backend support
- **QR Code Decoding**: ZBar-based QR code extraction and decoding
- **Vision-based Guidance**: PID-controlled target tracking
- **MAVLink Communication**: Autopilot integration with MAVLink 2.0
- **Telemetry Streaming**: Detection data and video to ground station
- **Video Recording**: Hardware-accelerated recording with overlays

## Supported Platforms

| Platform | Capture | Inference | Status |
|----------|---------|-----------|--------|
| NVIDIA Jetson | Argus (CSI) | TensorRT | ✅ Implemented |
| Raspberry Pi | libcamera (CSI) | HailoRT / NCNN | ✅ Implemented |
| x86/x64 | V4L2 (USB) / GStreamer | TensorRT / OpenVINO | ✅ Implemented |
| Isaac Sim | Shared Memory | N/A | ✅ Implemented |

## Capture Backends

The vision system supports multiple capture backends for different platforms and use cases:

| Backend | Platform | Sources | Features |
|---------|----------|---------|----------|
| **V4L2** | Linux | USB cameras | Direct V4L2 API, low latency |
| **Argus** | Jetson | CSI cameras | Hardware ISP, zero-copy, AWB, denoise |
| **libcamera** | RPi | Pi cameras | IMX219/477/708, AE, AWB, image tuning |
| **GStreamer** | All | RTSP, file, HTTP | HW decode, VAAPI/NVDEC, seeking |
| **Isaac Sim** | Linux | Isaac Sim | POSIX shared memory, lock-free |
| **Simulation** | All | Test patterns | Synthetic frames for testing |

### GStreamer Capture

The GStreamer backend enables capture from network streams and video files:

```yaml
capture:
  source: rtsp  # or file, gstreamer
  gstreamer:
    uri: "rtsp://192.168.1.100:554/stream"
    latency_ms: 200
    use_tcp: true
    hw_decode: true
```

### Isaac Sim Integration

For simulation with NVIDIA Isaac Sim, use the shared memory streamer:

```bash
# In Isaac Sim, run the streamer script
cd scripts/isaac_sim
python lagari_shm_streamer.py --width 1280 --height 720 --fps 30
```

See [scripts/isaac_sim/README.md](scripts/isaac_sim/README.md) for details.

## Requirements

### All Platforms
- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+)
- OpenCV 4.x
- yaml-cpp
- spdlog
- ZBar

### Optional Dependencies
- **GStreamer**: `libgstreamer1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad`
- **Google Test**: For testing

### Platform-Specific
- **Jetson**: CUDA Toolkit, TensorRT, Jetson Multimedia API, Argus
- **Raspberry Pi**: libcamera-dev
- **x86**: V4L2, TensorRT (GPU) or OpenVINO (CPU)

## Building

### Quick Start (x86 Development)

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt install \
    build-essential cmake ninja-build \
    libopencv-dev libyaml-cpp-dev libspdlog-dev \
    libzbar-dev libgtest-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

# Configure with x86 release preset
cmake --preset x86-release

# Build
cmake --build build/x86-release

# Run tests
ctest --test-dir build/x86-release --output-on-failure
```

### Using CMake Presets

Available presets:
- `x86-debug` / `x86-release`
- `jetson-debug` / `jetson-release`
- `rpi-debug` / `rpi-release`

```bash
# Configure
cmake --preset <preset-name>

# Build
cmake --build build/<preset-name>
```

### Build Configuration Output

The build system automatically detects available backends:

```
=== Lagari Vision Build Configuration ===
Platform: X86
Build type: Release
C++ Standard: 17

Capture backends:
  Argus (Jetson): FALSE
  libcamera (RPi): FALSE
  V4L2: TRUE
  GStreamer: TRUE
  Isaac Sim (SHM): TRUE

Inference backends:
  TensorRT: FALSE
  HailoRT: FALSE
  NCNN: FALSE
  OpenVINO: FALSE
==========================================
```

## Usage

```bash
# Run with default configuration
./lagari_vision

# Run with custom configuration
./lagari_vision --config /path/to/config.yaml

# Override specific settings
./lagari_vision --capture.source=usb --detection.confidence_threshold=0.6

# Show help
./lagari_vision --help
```

## Configuration

Configuration is managed via YAML files. See `config/default.yaml` for all options.

Key sections:
- `capture`: Camera source and settings (with backend-specific options)
- `detection`: Model and inference settings
- `qr`: QR decoding options
- `guidance`: PID gains and limits
- `autopilot`: MAVLink connection
- `telemetry`: GCS streaming
- `recording`: Video capture

### Capture Configuration Example

```yaml
capture:
  source: auto       # auto, csi, usb, file, rtsp, gstreamer, simulation, isaac
  width: 1280
  height: 720
  fps: 30
  
  # Backend-specific settings
  argus:
    sensor_mode: 0
    denoise: true
    awb: true
    
  libcamera:
    ae_enable: true
    brightness: 0.0
    contrast: 1.0
    
  gstreamer:
    uri: "rtsp://camera/stream"
    latency_ms: 200
    hw_decode: true
    
  isaac:
    shm_name: lagari_camera
    poll_interval_us: 100
```

## Project Structure

```
lagari_v/
├── cmake/                  # CMake modules
├── config/                 # Configuration files
├── include/lagari/         # Public headers
│   ├── core/              # Core utilities
│   ├── capture/           # Camera capture backends
│   ├── detection/         # Object detection
│   ├── qr/                # QR decoding
│   ├── guidance/          # Control algorithms
│   ├── comms/             # Communication
│   └── recording/         # Video recording
├── src/                    # Source files
├── tests/                  # Unit and integration tests
├── models/                 # AI model files
└── scripts/               # Utility scripts
    └── isaac_sim/         # Isaac Sim integration
```

## Architecture

```
Camera → Frame Buffer → Detection → Guidance → Autopilot
           ↓               ↓           ↓
       Recording      QR Decode    Telemetry
```

### Data Flow
- **SPMC Ring Buffer**: Lock-free buffer for frame distribution
- **SPSC Queues**: Lock-free queues for messages between modules

### Threading Model
- High priority: Capture, Detection, Guidance, MAVLink TX
- Medium priority: QR Decode, Recording, Telemetry
- Low priority: Logging, Monitoring

## Testing

```bash
# Run all tests
ctest --test-dir build/x86-release --output-on-failure

# Run specific test
./build/x86-release/tests/test_ring_buffer
./build/x86-release/tests/test_pid_controller
```

## License

## Contributing
