# Isaac Sim Integration for Lagari Vision System

This directory contains scripts for integrating the Lagari vision system with
NVIDIA Isaac Sim for simulation-based testing and development.

## Overview

The integration uses **POSIX shared memory** for ultra-low-latency, zero-copy
frame transfer from Isaac Sim to the Lagari C++ vision pipeline. This is
significantly faster than network-based solutions like ZeroMQ.

**Performance**: ~100µs latency for 1280x720 BGR frames (vs ~5-10ms with ZMQ)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Isaac Sim                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  USD Scene (World, UAV, Camera, Target)                   │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │ Camera Sensor                        │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │  lagari_shm_streamer.py                                   │   │
│  │  - Captures RGBA frames from camera                       │   │
│  │  - Converts to BGR24                                      │   │
│  │  - Writes to shared memory with lock-free protocol        │   │
│  └────────────────────────┬─────────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────────┘
                            │ POSIX Shared Memory
                            │ /dev/shm/lagari_camera
                            │ Zero-copy, lock-free
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Lagari Vision System                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  IsaacShmCapture (C++)                                     │ │
│  │  - Attaches to shared memory segment                       │ │
│  │  - Lock-free frame reading via sequence numbers            │ │
│  │  - Provides frames to vision pipeline                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Detection → QR Decoding → Guidance                        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `lagari_shm_streamer.py` | **Primary** - Shared memory camera streamer |
| `lagari_camera_extension.py` | Isaac Sim extension (ZMQ fallback) |
| `lagari_camera_standalone.py` | Standalone ZMQ streamer (fallback) |
| `create_uav_scene.py` | Sample UAV scene generator |

## Quick Start

### 1. Start Isaac Sim with Shared Memory Streaming

```bash
cd /path/to/isaac-sim

# Run with the shared memory streamer
./isaac_sim.sh --allow-root \
    --exec /path/to/lagari_v/scripts/isaac_sim/lagari_shm_streamer.py -- \
    --usd /path/to/your/scene.usd \
    --camera /World/UAV/Camera \
    --shm-name lagari_camera \
    --width 1280 \
    --height 720 \
    --fps 30
```

### 2. Verify Shared Memory Segment

```bash
# Check that the shared memory was created
ls -la /dev/shm/lagari_camera

# Monitor the segment (optional)
watch -n 0.1 'stat /dev/shm/lagari_camera'
```

### 3. Run Lagari Vision

```bash
cd /path/to/lagari_v

# Build
cmake --preset x86-release
cmake --build build/x86-release

# Run with Isaac Sim config
./build/x86-release/lagari_vision --config config/isaac_sim.yaml
```

## Shared Memory Protocol

The shared memory segment has a fixed layout for lock-free access:

### Memory Layout

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `magic` | "LGRV" (0x5652474C) |
| 4 | 4 | `version` | Protocol version (1) |
| 8 | 4 | `width` | Frame width in pixels |
| 12 | 4 | `height` | Frame height in pixels |
| 16 | 4 | `format` | Pixel format (2 = BGR24) |
| 20 | 8 | `frame_id` | Monotonic frame counter |
| 28 | 8 | `timestamp_ns` | Capture time (nanoseconds) |
| 36 | 8 | `write_seq` | Writer sequence number |
| 44 | 8 | `read_seq` | Reader sequence number |
| 52 | 12 | reserved | Future use |
| 64 | W×H×3 | data | BGR24 pixel data |

### Lock-Free Protocol

The protocol uses sequence numbers for synchronization without locks:

**Writer (Isaac Sim Python):**
1. Increment `write_seq` to odd (signals "write in progress")
2. Update header fields (frame_id, timestamp)
3. Write frame data
4. Increment `write_seq` to even (signals "write complete")

**Reader (Lagari C++):**
1. Read header, check if `write_seq` is even (complete frame)
2. Check if `write_seq > last_seen_write_seq` (new frame)
3. Copy frame data
4. Re-check `write_seq` (detect torn read)
5. Update `read_seq = write_seq`

This ensures:
- No locks or mutexes needed
- Readers never block writers
- Torn reads are detected and discarded

## Configuration

In your Lagari config YAML:

```yaml
capture:
  source: isaac                    # Use Isaac Sim capture
  width: 1280
  height: 720
  fps: 30
  
  isaac:
    shm_name: lagari_camera        # Shared memory segment name
    poll_interval_us: 100          # Polling interval (microseconds)
    reconnect: true                # Auto-reconnect if segment disappears
    reconnect_delay_ms: 1000       # Delay before reconnect attempt
```

## Performance Tuning

### Minimize Latency

1. **Lower poll interval** (at cost of CPU):
   ```yaml
   capture:
     isaac:
       poll_interval_us: 50  # Poll every 50µs
   ```

2. **Pin processes to CPUs**:
   ```bash
   # Isaac Sim on cores 0-7
   taskset -c 0-7 ./isaac_sim.sh ...
   
   # Lagari on cores 8-15
   taskset -c 8-15 ./build/x86-release/lagari_vision
   ```

3. **Use real-time scheduling** (requires root):
   ```bash
   sudo chrt -f 50 ./build/x86-release/lagari_vision
   ```

### Maximize Throughput

1. **Match frame rates**: Set Isaac Sim camera frequency to match Lagari's consumption rate

2. **Use appropriate resolution**: Higher resolution means larger shared memory and longer copy times

## Troubleshooting

### No frames received

1. **Check shared memory exists**:
   ```bash
   ls -la /dev/shm/lagari_camera
   ```

2. **Verify Isaac Sim is writing**:
   ```bash
   # Watch for size/timestamp changes
   watch -n 0.1 'stat /dev/shm/lagari_camera'
   ```

3. **Check magic header**:
   ```bash
   xxd -l 16 /dev/shm/lagari_camera
   # Should show: 4c 47 52 56 (LGRV) at start
   ```

### Permission denied

```bash
# Check permissions
ls -la /dev/shm/

# Isaac Sim and Lagari must run as same user, or:
chmod 666 /dev/shm/lagari_camera
```

### Stale shared memory

If Isaac Sim crashes, the shared memory may persist:
```bash
rm /dev/shm/lagari_camera
```

### High latency

1. Check poll interval isn't too high
2. Ensure both processes are on the same machine
3. Verify no disk swapping is occurring

## Comparison: Shared Memory vs ZeroMQ

| Aspect | Shared Memory | ZeroMQ |
|--------|---------------|--------|
| Latency | ~100µs | ~5-10ms |
| Throughput | Limited by memory bandwidth | Limited by network stack |
| CPU usage | Low (polling-based) | Higher (serialization) |
| Complexity | Simple protocol | More robust |
| Cross-machine | ❌ No | ✅ Yes |
| Error handling | Manual | Built-in |

**Recommendation**: Use shared memory for local simulation (maximum performance),
ZeroMQ for remote/distributed setups or when reliability is critical.
