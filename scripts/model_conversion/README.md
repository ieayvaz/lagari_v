# YOLO Model Conversion Scripts

Scripts for converting Ultralytics YOLO models to various inference formats.

## Prerequisites

```bash
pip install ultralytics
```

## Quick Start

### Download and Convert Pre-trained Models

```bash
# For NVIDIA GPU (Jetson/x86)
python download_models.py yolo11n tensorrt --fp16

# For Intel CPU
python download_models.py yolo11n openvino

# For ARM CPU (Raspberry Pi)
python download_models.py yolo11n ncnn
```

### Convert Custom Models

```bash
# Convert to all formats
python convert_model.py --model path/to/custom.pt --format all

# Convert with specific settings
python convert_model.py --model best.pt --format tensorrt --fp16 --imgsz 640
```

## Available Models

| Model | Speed | mAP | Recommended For |
|-------|-------|-----|-----------------|
| yolo11n | ⚡⚡⚡ | 38.5 | Real-time embedded |
| yolo11s | ⚡⚡ | 46.3 | Balanced |
| yolo11m | ⚡ | 51.4 | Higher accuracy |
| yolov10n | ⚡⚡⚡ | 38.5 | No NMS overhead |
| yolov8n | ⚡⚡⚡ | 37.3 | Well-tested |

## Usage in Lagari

After conversion, update your config:

```yaml
detection:
  backend: tensorrt  # or openvino, ncnn
  model_path: models/yolo11n.engine
  labels_path: models/coco.names
  input_width: 640
  input_height: 640
  confidence_threshold: 0.5
  fp16: true
```

## Platform Recommendations

| Platform | Backend | Command |
|----------|---------|---------|
| Jetson Orin | TensorRT | `download_models.py yolo11n tensorrt --fp16` |
| x86 + NVIDIA GPU | TensorRT | `download_models.py yolo11n tensorrt --fp16` |
| x86 Intel CPU | OpenVINO | `download_models.py yolo11n openvino` |
| Raspberry Pi 5 | NCNN | `download_models.py yolo11n ncnn` |

## INT8 Quantization

For INT8 quantization (best performance but requires calibration):

```bash
python convert_model.py \
    --model yolo11n.pt \
    --format openvino \
    --int8 \
    --data coco.yaml
```

## Troubleshooting

### TensorRT Export Fails
- Ensure CUDA and TensorRT are installed
- Check GPU memory (engine building requires significant VRAM)
- Try reducing workspace size: `--workspace 2`

### OpenVINO Export Fails
- Install OpenVINO: `pip install openvino`
- For INT8, ensure calibration dataset is accessible

### NCNN Export Fails
- NCNN export requires additional dependencies
- Install: `pip install ncnn`
