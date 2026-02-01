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
| NVIDIA Jetson | Argus (CSI) | TensorRT | 🚧 Planned |
| Raspberry Pi | libcamera (CSI) | HailoRT / NCNN | 🚧 Planned |
| x86/x64 | V4L2 (USB) | TensorRT / OpenVINO | 🔧 In Progress |

## Requirements

### All Platforms
- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+)
- OpenCV 4.x
- yaml-cpp
- spdlog
- ZBar
- Google Test (for testing)

### Platform-Specific
- **Jetson**: CUDA Toolkit, TensorRT, Jetson Multimedia API
- **Raspberry Pi**: libcamera, HailoRT SDK (optional), NCNN
- **x86**: V4L2, TensorRT (GPU) or OpenVINO (CPU)

## Building

### Quick Start (x86 Development)

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt install \
    build-essential cmake ninja-build \
    libopencv-dev libyaml-cpp-dev libspdlog-dev \
    libzbar-dev libgtest-dev

# Configure with x86 debug preset
cmake --preset x86-debug

# Build
cmake --build build/x86-debug

# Run tests
ctest --test-dir build/x86-debug --output-on-failure
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

### Manual Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)
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
- `capture`: Camera source and settings
- `detection`: Model and inference settings
- `qr`: QR decoding options
- `guidance`: PID gains and limits
- `autopilot`: MAVLink connection
- `telemetry`: GCS streaming
- `recording`: Video capture

## Project Structure

```
lagari_v/
├── cmake/                  # CMake modules
├── config/                 # Configuration files
├── include/lagari/         # Public headers
│   ├── core/              # Core utilities
│   ├── capture/           # Camera capture
│   ├── detection/         # Object detection
│   ├── qr/                # QR decoding
│   ├── guidance/          # Control algorithms
│   ├── comms/             # Communication
│   └── recording/         # Video recording
├── src/                    # Source files
├── tests/                  # Unit and integration tests
├── models/                 # AI model files
└── scripts/               # Utility scripts
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
ctest --test-dir build/x86-debug --output-on-failure

# Run specific test
./build/x86-debug/tests/test_ring_buffer
./build/x86-debug/tests/test_pid_controller
```

## License

[Your License Here]

## Contributing

[Contribution Guidelines]
