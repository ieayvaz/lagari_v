#!/usr/bin/env python3
"""
Download and convert pre-trained YOLO models for Lagari Vision.

Downloads the specified YOLO model from Ultralytics and exports it
to the requested format for deployment.

Usage:
    python download_models.py yolo11n tensorrt   # For Jetson/GPU
    python download_models.py yolo11n openvino   # For Intel CPU
    python download_models.py yolo11n ncnn       # For ARM CPU
"""

import argparse
import os
import sys
from pathlib import Path

# Model sizes available
YOLO_MODELS = {
    # YOLO11 (latest)
    'yolo11n': 'yolo11n.pt',
    'yolo11s': 'yolo11s.pt',
    'yolo11m': 'yolo11m.pt',
    'yolo11l': 'yolo11l.pt',
    'yolo11x': 'yolo11x.pt',
    # YOLOv10
    'yolov10n': 'yolov10n.pt',
    'yolov10s': 'yolov10s.pt',
    'yolov10m': 'yolov10m.pt',
    'yolov10l': 'yolov10l.pt',
    # YOLOv8
    'yolov8n': 'yolov8n.pt',
    'yolov8s': 'yolov8s.pt',
    'yolov8m': 'yolov8m.pt',
    'yolov8l': 'yolov8l.pt',
    'yolov8x': 'yolov8x.pt',
    # YOLOv5
    'yolov5n': 'yolov5n.pt',
    'yolov5s': 'yolov5s.pt',
    'yolov5m': 'yolov5m.pt',
    'yolov5l': 'yolov5l.pt',
}

def main():
    parser = argparse.ArgumentParser(
        description='Download and convert YOLO models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  YOLO11: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x (recommended)
  YOLOv10: yolov10n, yolov10s, yolov10m, yolov10l
  YOLOv8: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  YOLOv5: yolov5n, yolov5s, yolov5m, yolov5l

Recommended for UAV:
  - yolo11n: Fastest, good for real-time on embedded
  - yolo11s: Balanced speed/accuracy
        """
    )
    
    parser.add_argument('model', type=str, choices=list(YOLO_MODELS.keys()),
                        help='Model name to download')
    parser.add_argument('format', type=str, 
                        choices=['onnx', 'tensorrt', 'openvino', 'ncnn'],
                        help='Export format')
    parser.add_argument('--output', '-o', type=str, default='models',
                        help='Output directory')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision')
    
    args = parser.parse_args()
    
    # Install ultralytics if needed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and load model
    model_name = YOLO_MODELS[args.model]
    print(f"\n→ Loading {model_name}...")
    model = YOLO(model_name)
    
    # Export
    print(f"→ Exporting to {args.format}...")
    
    export_args = {
        'format': 'engine' if args.format == 'tensorrt' else args.format,
        'imgsz': args.imgsz,
    }
    
    if args.format == 'tensorrt':
        export_args['half'] = args.fp16
        export_args['simplify'] = True
    elif args.format in ['openvino', 'ncnn']:
        export_args['half'] = args.fp16
    
    output_path = model.export(**export_args)
    
    print(f"\n✓ Model exported to: {output_path}")
    print(f"\nTo use in Lagari:")
    print(f"  detection:")
    print(f"    model_path: {output_path}")
    print(f"    backend: {args.format}")
    print(f"    input_width: {args.imgsz}")
    print(f"    input_height: {args.imgsz}")

if __name__ == '__main__':
    main()
