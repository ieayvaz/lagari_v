#!/usr/bin/env python3
"""
Standalone Isaac Sim Camera Streamer

This script can be run from Isaac Sim's Python environment to stream
camera frames without needing to install the extension.

Usage:
    ./isaac_sim.sh --allow-root --/app/streaming/enabled=true \
        --exec scripts/lagari_camera_standalone.py -- \
        --usd /path/to/scene.usd \
        --camera /World/Camera \
        --port 5555

Or from Python:
    from scripts.lagari_camera_standalone import stream_camera
    stream_camera(scene_path, camera_path, port)
"""

import argparse
import signal
import sys
import time
from typing import Optional

# Isaac Sim imports (available when running from Isaac Sim python)
try:
    from omni.isaac.kit import SimulationApp
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("Warning: Isaac Sim not available. Run from Isaac Sim Python environment.")


def create_simulation(headless: bool = True) -> Optional["SimulationApp"]:
    """Create Isaac Sim simulation app"""
    if not ISAAC_AVAILABLE:
        return None
        
    config = {
        "width": 1280,
        "height": 720,
        "headless": headless,
        "renderer": "RayTracedLighting",
        "anti_aliasing": 0,
    }
    return SimulationApp(config)


def stream_camera(
    scene_path: str,
    camera_prim_path: str = "/World/Camera",
    zmq_port: int = 5555,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    duration: float = 0.0,  # 0 = infinite
    headless: bool = True
):
    """
    Stream camera frames from an Isaac Sim scene.
    
    Args:
        scene_path: Path to USD scene file
        camera_prim_path: USD path to camera prim
        zmq_port: ZeroMQ publisher port
        width: Output width
        height: Output height
        fps: Target framerate
        duration: Duration in seconds (0 = run until interrupted)
        headless: Run without GUI
    """
    import omni
    from omni.isaac.core import World
    from omni.isaac.sensor import Camera
    import numpy as np
    import zmq
    import struct
    
    # Create simulation
    sim = create_simulation(headless)
    if sim is None:
        return
    
    try:
        # Load scene
        omni.usd.get_context().open_stage(scene_path)
        
        # Wait for stage to load
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
        
        # Initialize ZeroMQ
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 2)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://*:{zmq_port}")
        
        print(f"[Lagari] Streaming from {camera_prim_path}")
        print(f"[Lagari] ZMQ Publisher on tcp://*:{zmq_port}")
        print(f"[Lagari] Resolution: {width}x{height} @ {fps} FPS")
        print("[Lagari] Press Ctrl+C to stop")
        
        # Header format
        HEADER_FORMAT = "<QQIIi"
        FORMAT_BGR24 = 2
        
        # Streaming loop
        frame_id = 0
        frame_interval = 1.0 / fps
        last_frame_time = 0.0
        start_time = time.time()
        frames_sent = 0
        running = True
        
        def signal_handler(sig, frame):
            nonlocal running
            print("\n[Lagari] Stopping...")
            running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while running:
            # Step simulation
            world.step(render=True)
            
            # Check duration
            current_time = time.time()
            if duration > 0 and (current_time - start_time) >= duration:
                break
            
            # Rate limiting
            if current_time - last_frame_time < frame_interval:
                continue
            last_frame_time = current_time
            
            # Get frame
            rgba = camera.get_rgba()
            if rgba is None:
                continue
            
            # Convert RGBA to BGR
            bgr = rgba[:, :, :3][:, :, ::-1].copy()
            
            # Build header
            frame_id += 1
            timestamp_ns = int(current_time * 1e9)
            header = struct.pack(
                HEADER_FORMAT,
                frame_id,
                timestamp_ns,
                width,
                height,
                FORMAT_BGR24
            )
            
            # Send
            try:
                socket.send_multipart([
                    b"frame",
                    header,
                    bgr.tobytes()
                ], zmq.NOBLOCK)
                frames_sent += 1
            except zmq.Again:
                pass
        
        # Stats
        elapsed = time.time() - start_time
        print(f"[Lagari] Sent {frames_sent} frames in {elapsed:.1f}s ({frames_sent/elapsed:.1f} fps)")
        
        # Cleanup
        socket.close()
        context.term()
        
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description="Isaac Sim Camera Streamer for Lagari")
    parser.add_argument("--usd", required=True, help="Path to USD scene file")
    parser.add_argument("--camera", default="/World/Camera", help="Camera prim path")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration (0=infinite)")
    parser.add_argument("--gui", action="store_true", help="Show GUI")
    
    args = parser.parse_args()
    
    stream_camera(
        scene_path=args.usd,
        camera_prim_path=args.camera,
        zmq_port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        headless=not args.gui
    )


if __name__ == "__main__":
    main()
