"""
Isaac Sim Camera Streamer Extension for Lagari Vision System

This extension captures camera frames from Isaac Sim and streams them
to the Lagari vision system via ZeroMQ.

Installation:
1. Copy this folder to ~/.local/share/ov/pkg/isaac-sim-*/exts/
   Or create a symlink
2. Enable the extension in Isaac Sim: Window > Extensions > lagari.camera

Usage:
- The extension automatically starts streaming when enabled
- Configure camera path and ZMQ port in the extension settings
- Frames are sent as raw BGR24 data with metadata header
"""

import omni.ext
import omni.kit.app
import omni.usd
from omni.isaac.sensor import Camera
from pxr import UsdGeom, Gf

import numpy as np
import zmq
import struct
import threading
import time
from typing import Optional


class LagariCameraStreamer:
    """Captures frames from Isaac Sim camera and streams via ZeroMQ"""
    
    # Frame header format: frame_id(8) + timestamp_ns(8) + width(4) + height(4) + format(4)
    HEADER_FORMAT = "<QQIIi"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    # Pixel format constants (must match C++ PixelFormat enum)
    FORMAT_BGR24 = 2
    FORMAT_RGB24 = 1
    FORMAT_RGBA32 = 3
    
    def __init__(
        self,
        camera_prim_path: str = "/World/Camera",
        zmq_port: int = 5555,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0
    ):
        """
        Initialize the camera streamer.
        
        Args:
            camera_prim_path: USD path to the camera prim
            zmq_port: ZeroMQ publisher port
            width: Output frame width
            height: Output frame height
            fps: Target framerate
        """
        self.camera_prim_path = camera_prim_path
        self.zmq_port = zmq_port
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        self._camera: Optional[Camera] = None
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._running = False
        self._frame_id = 0
        self._last_frame_time = 0.0
        
        # Statistics
        self._frames_sent = 0
        self._start_time = 0.0
    
    def initialize(self) -> bool:
        """Initialize camera and ZMQ socket"""
        try:
            # Create camera sensor
            self._camera = Camera(
                prim_path=self.camera_prim_path,
                frequency=self.fps,
                resolution=(self.width, self.height)
            )
            self._camera.initialize()
            
            # Initialize ZeroMQ publisher
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.PUB)
            self._socket.setsockopt(zmq.SNDHWM, 2)  # High water mark: drop old frames
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.bind(f"tcp://*:{self.zmq_port}")
            
            print(f"[Lagari] Camera streamer initialized on port {self.zmq_port}")
            print(f"[Lagari] Camera: {self.camera_prim_path} @ {self.width}x{self.height}")
            return True
            
        except Exception as e:
            print(f"[Lagari] Failed to initialize: {e}")
            return False
    
    def start(self):
        """Start the camera streaming"""
        if self._running:
            return
            
        self._running = True
        self._start_time = time.time()
        self._frames_sent = 0
        print("[Lagari] Camera streaming started")
    
    def stop(self):
        """Stop the camera streaming"""
        self._running = False
        print(f"[Lagari] Streaming stopped. Frames sent: {self._frames_sent}")
    
    def shutdown(self):
        """Clean up resources"""
        self.stop()
        
        if self._socket:
            self._socket.close()
            self._socket = None
            
        if self._context:
            self._context.term()
            self._context = None
            
        self._camera = None
    
    def on_physics_step(self, dt: float):
        """Called on each physics step to capture and send frame"""
        if not self._running or not self._camera:
            return
        
        # Rate limiting
        current_time = time.time()
        if current_time - self._last_frame_time < self.frame_interval:
            return
        self._last_frame_time = current_time
        
        try:
            # Get camera frame
            frame = self._camera.get_rgba()
            if frame is None:
                return
            
            # Convert RGBA to BGR for OpenCV compatibility
            bgr = frame[:, :, :3][:, :, ::-1].copy()
            
            # Build header
            self._frame_id += 1
            timestamp_ns = int(current_time * 1e9)
            header = struct.pack(
                self.HEADER_FORMAT,
                self._frame_id,
                timestamp_ns,
                self.width,
                self.height,
                self.FORMAT_BGR24
            )
            
            # Send frame (topic + header + data)
            self._socket.send_multipart([
                b"frame",           # Topic for filtering
                header,             # Metadata
                bgr.tobytes()       # Raw pixel data
            ], zmq.NOBLOCK)
            
            self._frames_sent += 1
            
        except zmq.Again:
            # Socket would block, skip this frame
            pass
        except Exception as e:
            print(f"[Lagari] Error sending frame: {e}")
    
    def get_stats(self) -> dict:
        """Get streaming statistics"""
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        return {
            "frames_sent": self._frames_sent,
            "elapsed_time": elapsed,
            "average_fps": self._frames_sent / elapsed if elapsed > 0 else 0,
            "running": self._running
        }


class LagariCameraExtension(omni.ext.IExt):
    """Isaac Sim Extension for Lagari Camera Streaming"""
    
    def on_startup(self, ext_id):
        print("[Lagari] Extension starting up")
        
        self._streamer: Optional[LagariCameraStreamer] = None
        self._physics_sub = None
        
        # Default settings (can be configured via extension settings)
        self._camera_path = "/World/Camera"
        self._zmq_port = 5555
        self._width = 1280
        self._height = 720
        self._fps = 30.0
        
        # Create streamer
        self._streamer = LagariCameraStreamer(
            camera_prim_path=self._camera_path,
            zmq_port=self._zmq_port,
            width=self._width,
            height=self._height,
            fps=self._fps
        )
        
        # Subscribe to physics updates
        physics_interface = omni.physx.get_physx_interface()
        self._physics_sub = physics_interface.subscribe_physics_step_events(
            self._on_physics_step
        )
        
    def on_shutdown(self):
        print("[Lagari] Extension shutting down")
        
        if self._physics_sub:
            self._physics_sub.unsubscribe()
            self._physics_sub = None
            
        if self._streamer:
            self._streamer.shutdown()
            self._streamer = None
    
    def _on_physics_step(self, dt: float):
        """Physics step callback"""
        if self._streamer:
            self._streamer.on_physics_step(dt)
    
    def start_streaming(self):
        """Start camera streaming (can be called from UI or script)"""
        if self._streamer and not self._streamer._running:
            if self._streamer.initialize():
                self._streamer.start()
    
    def stop_streaming(self):
        """Stop camera streaming"""
        if self._streamer:
            self._streamer.stop()
    
    def configure(
        self,
        camera_path: str = None,
        zmq_port: int = None,
        width: int = None,
        height: int = None,
        fps: float = None
    ):
        """Reconfigure streamer settings (requires restart)"""
        if camera_path:
            self._camera_path = camera_path
        if zmq_port:
            self._zmq_port = zmq_port
        if width:
            self._width = width
        if height:
            self._height = height
        if fps:
            self._fps = fps
        
        # Recreate streamer with new settings
        if self._streamer:
            self._streamer.shutdown()
            
        self._streamer = LagariCameraStreamer(
            camera_prim_path=self._camera_path,
            zmq_port=self._zmq_port,
            width=self._width,
            height=self._height,
            fps=self._fps
        )
