#!/usr/bin/env python3
"""
Isaac Sim Shared Memory Camera Streamer

High-performance frame streaming using POSIX shared memory for
zero-copy transfer to the Lagari C++ vision system.

The shared memory layout:
    Header (64 bytes):
        - magic (4 bytes): "LGRV"
        - version (4 bytes): 1
        - width (4 bytes)
        - height (4 bytes)
        - format (4 bytes): pixel format enum
        - frame_id (8 bytes): monotonic frame counter
        - timestamp_ns (8 bytes): nanosecond timestamp
        - write_seq (8 bytes): writer sequence number (atomic)
        - read_seq (8 bytes): reader sequence number (atomic)
        - reserved (12 bytes)
    Data (width * height * 3 bytes for BGR24)
    
The protocol uses double-buffering with sequence numbers for
lock-free synchronization.
"""

import mmap
import os
import struct
import time
import signal
import argparse
from typing import Optional
import numpy as np

# Try to import POSIX shared memory
try:
    from multiprocessing import shared_memory
    HAS_MULTIPROCESSING_SHM = True
except ImportError:
    HAS_MULTIPROCESSING_SHM = False

# Constants
MAGIC = b'LGRV'
VERSION = 1
HEADER_SIZE = 64
FORMAT_BGR24 = 2

# Header struct format (little-endian)
# 4s = magic, I = version, I = width, I = height, i = format,
# Q = frame_id, Q = timestamp_ns, Q = write_seq, Q = read_seq, 12x = reserved
HEADER_FORMAT = '<4sIIIiQQQQ12x'
assert struct.calcsize(HEADER_FORMAT) == HEADER_SIZE


class SharedMemoryFrameBuffer:
    """
    Shared memory buffer for frame streaming.
    
    Uses a simple lock-free protocol with sequence numbers:
    1. Writer increments write_seq to odd (writing in progress)
    2. Writer writes frame data
    3. Writer increments write_seq to even (write complete)
    4. Reader checks if write_seq is even and > read_seq
    5. Reader reads frame data
    6. Reader sets read_seq = write_seq
    """
    
    def __init__(
        self,
        name: str = "lagari_camera",
        width: int = 1280,
        height: int = 720,
        create: bool = True
    ):
        self.name = name
        self.width = width
        self.height = height
        self.frame_size = width * height * 3  # BGR24
        self.total_size = HEADER_SIZE + self.frame_size
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.frame_id = 0
        
        if create:
            self._create()
        else:
            self._attach()
    
    def _create(self):
        """Create new shared memory segment"""
        # Remove existing if present
        try:
            existing = shared_memory.SharedMemory(name=self.name)
            existing.close()
            existing.unlink()
        except FileNotFoundError:
            pass
        
        # Create new segment
        self.shm = shared_memory.SharedMemory(
            name=self.name,
            create=True,
            size=self.total_size
        )
        
        # Initialize header
        self._write_header(0, 0, 0)
        print(f"[Lagari SHM] Created shared memory: {self.name} ({self.total_size} bytes)")
    
    def _attach(self):
        """Attach to existing shared memory segment"""
        self.shm = shared_memory.SharedMemory(name=self.name)
        print(f"[Lagari SHM] Attached to shared memory: {self.name}")
    
    def _write_header(self, frame_id: int, timestamp_ns: int, write_seq: int, read_seq: int = 0):
        """Write header to shared memory"""
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            VERSION,
            self.width,
            self.height,
            FORMAT_BGR24,
            frame_id,
            timestamp_ns,
            write_seq,
            read_seq
        )
        self.shm.buf[:HEADER_SIZE] = header
    
    def write_frame(self, bgr_data: np.ndarray) -> bool:
        """
        Write a frame to shared memory using lock-free protocol.
        
        Args:
            bgr_data: BGR24 numpy array (height, width, 3)
            
        Returns:
            True if frame was written successfully
        """
        if self.shm is None:
            return False
        
        if bgr_data.shape != (self.height, self.width, 3):
            print(f"[Lagari SHM] Frame size mismatch: {bgr_data.shape}")
            return False
        
        # Get current write sequence
        current_header = struct.unpack(HEADER_FORMAT, bytes(self.shm.buf[:HEADER_SIZE]))
        write_seq = current_header[7]
        read_seq = current_header[8]
        
        # Increment to odd (writing in progress)
        new_write_seq = (write_seq | 1) + 1  # Next odd number
        
        # Update header with odd sequence (indicates write in progress)
        self.frame_id += 1
        timestamp_ns = int(time.time() * 1e9)
        self._write_header(self.frame_id, timestamp_ns, new_write_seq, read_seq)
        
        # Write frame data
        frame_bytes = bgr_data.tobytes()
        self.shm.buf[HEADER_SIZE:HEADER_SIZE + len(frame_bytes)] = frame_bytes
        
        # Increment to even (write complete)
        new_write_seq += 1
        self._write_header(self.frame_id, timestamp_ns, new_write_seq, read_seq)
        
        return True
    
    def close(self):
        """Close shared memory (don't unlink)"""
        if self.shm:
            self.shm.close()
            self.shm = None
    
    def unlink(self):
        """Unlink (delete) the shared memory segment"""
        if self.shm:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


class IsaacShmStreamer:
    """
    Isaac Sim camera streamer using shared memory.
    """
    
    def __init__(
        self,
        camera_prim_path: str = "/World/Camera",
        shm_name: str = "lagari_camera",
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0
    ):
        self.camera_prim_path = camera_prim_path
        self.shm_name = shm_name
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        self._camera = None
        self._shm_buffer: Optional[SharedMemoryFrameBuffer] = None
        self._running = False
        self._last_frame_time = 0.0
        self._frames_sent = 0
        self._start_time = 0.0
    
    def initialize(self) -> bool:
        """Initialize camera and shared memory"""
        try:
            from omni.isaac.sensor import Camera
            
            # Create camera sensor
            self._camera = Camera(
                prim_path=self.camera_prim_path,
                frequency=self.fps,
                resolution=(self.width, self.height)
            )
            self._camera.initialize()
            
            # Create shared memory buffer
            self._shm_buffer = SharedMemoryFrameBuffer(
                name=self.shm_name,
                width=self.width,
                height=self.height,
                create=True
            )
            
            print(f"[Lagari SHM] Initialized: {self.camera_prim_path}")
            print(f"[Lagari SHM] Resolution: {self.width}x{self.height} @ {self.fps} FPS")
            print(f"[Lagari SHM] Shared memory: /dev/shm/{self.shm_name}")
            return True
            
        except Exception as e:
            print(f"[Lagari SHM] Initialization failed: {e}")
            return False
    
    def start(self):
        """Start streaming"""
        self._running = True
        self._start_time = time.time()
        self._frames_sent = 0
        print("[Lagari SHM] Streaming started")
    
    def stop(self):
        """Stop streaming"""
        self._running = False
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        fps = self._frames_sent / elapsed if elapsed > 0 else 0
        print(f"[Lagari SHM] Stopped. Frames: {self._frames_sent}, FPS: {fps:.1f}")
    
    def shutdown(self):
        """Clean up resources"""
        self.stop()
        if self._shm_buffer:
            self._shm_buffer.unlink()
            self._shm_buffer.close()
            self._shm_buffer = None
        self._camera = None
    
    def on_physics_step(self, dt: float):
        """Called on each physics step"""
        if not self._running or not self._camera or not self._shm_buffer:
            return
        
        # Rate limiting
        current_time = time.time()
        if current_time - self._last_frame_time < self.frame_interval:
            return
        self._last_frame_time = current_time
        
        try:
            # Get camera frame (RGBA)
            rgba = self._camera.get_rgba()
            if rgba is None:
                return
            
            # Convert RGBA to BGR (drop alpha, reverse RGB)
            bgr = rgba[:, :, :3][:, :, ::-1].copy()
            
            # Write to shared memory
            if self._shm_buffer.write_frame(bgr):
                self._frames_sent += 1
                
        except Exception as e:
            print(f"[Lagari SHM] Error: {e}")


def stream_camera_shm(
    scene_path: str,
    camera_prim_path: str = "/World/Camera",
    shm_name: str = "lagari_camera",
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    duration: float = 0.0,
    headless: bool = True
):
    """
    Standalone function to stream camera from Isaac Sim via shared memory.
    """
    from omni.isaac.kit import SimulationApp
    import omni
    from omni.isaac.core import World
    from omni.isaac.sensor import Camera
    
    # Create simulation
    config = {
        "width": width,
        "height": height,
        "headless": headless,
        "renderer": "RayTracedLighting",
    }
    sim = SimulationApp(config)
    
    try:
        # Load scene
        omni.usd.get_context().open_stage(scene_path)
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            sim.update()
        
        # Create world
        world = World(stage_units_in_meters=1.0)
        world.reset()
        
        # Create camera
        camera = Camera(
            prim_path=camera_prim_path,
            frequency=fps,
            resolution=(width, height)
        )
        camera.initialize()
        
        # Create shared memory buffer
        shm_buffer = SharedMemoryFrameBuffer(
            name=shm_name,
            width=width,
            height=height,
            create=True
        )
        
        print(f"[Lagari SHM] Streaming from {camera_prim_path}")
        print(f"[Lagari SHM] Shared memory: /dev/shm/{shm_name}")
        print(f"[Lagari SHM] Resolution: {width}x{height} @ {fps} FPS")
        print("[Lagari SHM] Press Ctrl+C to stop")
        
        # Streaming state
        frame_interval = 1.0 / fps
        last_frame_time = 0.0
        start_time = time.time()
        frames_sent = 0
        running = True
        
        def signal_handler(sig, frame):
            nonlocal running
            print("\n[Lagari SHM] Stopping...")
            running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Main loop
        while running:
            world.step(render=True)
            
            # Check duration
            current_time = time.time()
            if duration > 0 and (current_time - start_time) >= duration:
                break
            
            # Rate limit
            if current_time - last_frame_time < frame_interval:
                continue
            last_frame_time = current_time
            
            # Get and write frame
            rgba = camera.get_rgba()
            if rgba is not None:
                bgr = rgba[:, :, :3][:, :, ::-1].copy()
                if shm_buffer.write_frame(bgr):
                    frames_sent += 1
        
        # Stats
        elapsed = time.time() - start_time
        print(f"[Lagari SHM] Sent {frames_sent} frames in {elapsed:.1f}s ({frames_sent/elapsed:.1f} fps)")
        
        # Cleanup
        shm_buffer.unlink()
        shm_buffer.close()
        
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description="Isaac Sim Shared Memory Camera Streamer")
    parser.add_argument("--usd", required=True, help="Path to USD scene file")
    parser.add_argument("--camera", default="/World/Camera", help="Camera prim path")
    parser.add_argument("--shm-name", default="lagari_camera", help="Shared memory name")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration (0=infinite)")
    parser.add_argument("--gui", action="store_true", help="Show GUI")
    
    args = parser.parse_args()
    
    stream_camera_shm(
        scene_path=args.usd,
        camera_prim_path=args.camera,
        shm_name=args.shm_name,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        headless=not args.gui
    )


if __name__ == "__main__":
    main()
