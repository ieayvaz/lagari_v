// YOLO preprocessing and postprocessing
// TODO: Implement YOLO-specific pre/post processing

#include "lagari/detection/detector.hpp"

namespace lagari {

// Common YOLO preprocessing:
// 1. Resize to input size (letterbox for aspect ratio preservation)
// 2. Convert BGR to RGB (if needed)
// 3. Normalize to 0-1
// 4. HWC to CHW

// Common YOLO postprocessing:
// 1. Parse output tensor(s) based on YOLO version
// 2. Apply confidence filtering
// 3. Apply NMS
// 4. Convert coordinates back to original image space

}  // namespace lagari
