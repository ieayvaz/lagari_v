#!/usr/bin/env python3
"""
YOLO Model Converter for Lagari Vision System

Converts Ultralytics YOLO models to various inference formats:
- TensorRT (.engine) for NVIDIA GPUs
- OpenVINO IR (.xml/.bin) for Intel CPUs
- NCNN (.param/.bin) for ARM/mobile CPUs
- ONNX for cross-platform compatibility

Usage:
    python convert_model.py --model yolo11n.pt --format all
    python convert_model.py --model yolo11n.pt --format tensorrt --fp16
    python convert_model.py --model yolo11s.pt --format openvino --int8 --data coco.yaml
"""

import argparse
import os
import sys
from pathlib import Path

def install_dependencies():
    """Install required packages if not present."""
    try:
        import ultralytics
    except ImportError:
        print("Installing ultralytics...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        
def convert_to_onnx(model_path: str, output_dir: str, **kwargs) -> str:
    """Export model to ONNX format."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    # Export to ONNX
    output_path = model.export(
        format='onnx',
        imgsz=kwargs.get('imgsz', 640),
        opset=kwargs.get('opset', 12),
        simplify=kwargs.get('simplify', True),
        dynamic=kwargs.get('dynamic', False),
    )
    
    print(f"✓ ONNX model exported to: {output_path}")
    return output_path

def convert_to_tensorrt(model_path: str, output_dir: str, **kwargs) -> str:
    """Export model to TensorRT engine."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    # Export to TensorRT
    output_path = model.export(
        format='engine',
        imgsz=kwargs.get('imgsz', 640),
        half=kwargs.get('fp16', True),
        int8=kwargs.get('int8', False),
        workspace=kwargs.get('workspace', 4),  # GB
        batch=kwargs.get('batch', 1),
        simplify=True,
        device=kwargs.get('device', 0),
    )
    
    print(f"✓ TensorRT engine exported to: {output_path}")
    return output_path

def convert_to_openvino(model_path: str, output_dir: str, **kwargs) -> str:
    """Export model to OpenVINO IR format."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    # Export to OpenVINO
    output_path = model.export(
        format='openvino',
        imgsz=kwargs.get('imgsz', 640),
        half=kwargs.get('fp16', False),
        int8=kwargs.get('int8', False),
        data=kwargs.get('data', None),  # Required for INT8 calibration
    )
    
    print(f"✓ OpenVINO model exported to: {output_path}")
    return output_path

def convert_to_ncnn(model_path: str, output_dir: str, **kwargs) -> str:
    """Export model to NCNN format."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    # Export to NCNN
    output_path = model.export(
        format='ncnn',
        imgsz=kwargs.get('imgsz', 640),
        half=kwargs.get('fp16', False),
    )
    
    print(f"✓ NCNN model exported to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO models to various inference formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Convert to all formats:
    python convert_model.py --model yolo11n.pt --format all
    
  TensorRT with FP16:
    python convert_model.py --model yolo11n.pt --format tensorrt --fp16
    
  OpenVINO with INT8 quantization:
    python convert_model.py --model yolo11n.pt --format openvino --int8 --data coco.yaml
    
  NCNN for ARM devices:
    python convert_model.py --model yolo11n.pt --format ncnn
        """
    )
    
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to YOLO model (.pt file) or model name (e.g., yolo11n)')
    parser.add_argument('--format', '-f', type=str, default='all',
                        choices=['onnx', 'tensorrt', 'openvino', 'ncnn', 'all'],
                        help='Output format (default: all)')
    parser.add_argument('--output', '-o', type=str, default='models',
                        help='Output directory (default: models)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference (default: 640)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 half-precision')
    parser.add_argument('--int8', action='store_true',
                        help='Use INT8 quantization (requires calibration data)')
    parser.add_argument('--data', type=str, default=None,
                        help='Dataset config for INT8 calibration (e.g., coco.yaml)')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size for TensorRT engine')
    parser.add_argument('--workspace', type=int, default=4,
                        help='TensorRT workspace size in GB')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device for TensorRT export')
    
    args = parser.parse_args()
    
    # Install dependencies
    install_dependencies()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert model
    model_path = args.model
    kwargs = {
        'imgsz': args.imgsz,
        'fp16': args.fp16,
        'int8': args.int8,
        'data': args.data,
        'batch': args.batch,
        'workspace': args.workspace,
        'device': args.device,
    }
    
    formats = ['onnx', 'tensorrt', 'openvino', 'ncnn'] if args.format == 'all' else [args.format]
    
    print(f"\n{'='*60}")
    print(f"YOLO Model Converter")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Formats: {', '.join(formats)}")
    print(f"Image size: {args.imgsz}")
    print(f"FP16: {args.fp16}, INT8: {args.int8}")
    print(f"{'='*60}\n")
    
    results = {}
    
    for fmt in formats:
        try:
            print(f"\n→ Converting to {fmt.upper()}...")
            
            if fmt == 'onnx':
                results['onnx'] = convert_to_onnx(model_path, str(output_dir), **kwargs)
            elif fmt == 'tensorrt':
                results['tensorrt'] = convert_to_tensorrt(model_path, str(output_dir), **kwargs)
            elif fmt == 'openvino':
                results['openvino'] = convert_to_openvino(model_path, str(output_dir), **kwargs)
            elif fmt == 'ncnn':
                results['ncnn'] = convert_to_ncnn(model_path, str(output_dir), **kwargs)
                
        except Exception as e:
            print(f"✗ Failed to convert to {fmt}: {e}")
            results[fmt] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("Conversion Summary")
    print(f"{'='*60}")
    for fmt, path in results.items():
        status = "✓" if path else "✗"
        print(f"  {status} {fmt.upper()}: {path or 'FAILED'}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
