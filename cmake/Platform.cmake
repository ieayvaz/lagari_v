# ============================================================================
# Platform Detection
# ============================================================================

# Detect platform
if(EXISTS "/etc/nv_tegra_release" OR EXISTS "/proc/device-tree/compatible")
    file(READ "/proc/device-tree/compatible" DEVICE_TREE_COMPAT LIMIT 256)
    if(DEVICE_TREE_COMPAT MATCHES "nvidia")
        set(PLATFORM_JETSON TRUE)
        set(LAGARI_PLATFORM "JETSON")
        message(STATUS "Detected platform: NVIDIA Jetson")
    endif()
endif()

if(NOT PLATFORM_JETSON)
    execute_process(
        COMMAND cat /proc/cpuinfo
        OUTPUT_VARIABLE CPUINFO
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CPUINFO MATCHES "Raspberry Pi" OR CPUINFO MATCHES "BCM")
        set(PLATFORM_RPI TRUE)
        set(LAGARI_PLATFORM "RASPBERRY_PI")
        message(STATUS "Detected platform: Raspberry Pi")
    endif()
endif()

if(NOT PLATFORM_JETSON AND NOT PLATFORM_RPI)
    set(PLATFORM_X86 TRUE)
    set(LAGARI_PLATFORM "X86")
    message(STATUS "Detected platform: x86/x64")
endif()

# Set compile definitions for platform
if(PLATFORM_JETSON)
    add_compile_definitions(PLATFORM_JETSON)
elseif(PLATFORM_RPI)
    add_compile_definitions(PLATFORM_RPI)
else()
    add_compile_definitions(PLATFORM_X86)
endif()

# ============================================================================
# Capture Backend Detection
# ============================================================================

# Argus (Jetson only)
set(HAS_ARGUS FALSE)
if(PLATFORM_JETSON)
    find_path(ARGUS_INCLUDE_DIR Argus/Argus.h
        PATHS /usr/src/jetson_multimedia_api/argus/include
    )
    find_library(ARGUS_LIBRARY nvargus
        PATHS /usr/lib/aarch64-linux-gnu/tegra
    )
    if(ARGUS_INCLUDE_DIR AND ARGUS_LIBRARY)
        set(HAS_ARGUS TRUE)
        set(ARGUS_LIBRARIES ${ARGUS_LIBRARY})
        include_directories(${ARGUS_INCLUDE_DIR})
        message(STATUS "Found Argus: ${ARGUS_LIBRARY}")
    endif()
endif()

# libcamera (RPi)
set(HAS_LIBCAMERA FALSE)
if(PLATFORM_RPI)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(LIBCAMERA libcamera)
        if(LIBCAMERA_FOUND)
            set(HAS_LIBCAMERA TRUE)
            set(LIBCAMERA_LIBRARIES ${LIBCAMERA_LIBRARIES})
            include_directories(${LIBCAMERA_INCLUDE_DIRS})
            message(STATUS "Found libcamera: ${LIBCAMERA_VERSION}")
        endif()
    endif()
endif()

# V4L2 (always check, used for USB cameras)
set(HAS_V4L2 FALSE)
find_path(V4L2_INCLUDE_DIR linux/videodev2.h)
if(V4L2_INCLUDE_DIR)
    set(HAS_V4L2 TRUE)
    message(STATUS "Found V4L2 headers")
endif()

# ============================================================================
# Inference Backend Detection
# ============================================================================

# TensorRT (Jetson + x86 with NVIDIA GPU)
set(HAS_TENSORRT FALSE)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS 
            /usr/include/x86_64-linux-gnu
            /usr/include/aarch64-linux-gnu
            /usr/local/cuda/include
            ${CUDA_TOOLKIT_ROOT_DIR}/include
    )
    find_library(TENSORRT_LIBRARY nvinfer
        PATHS
            /usr/lib/x86_64-linux-gnu
            /usr/lib/aarch64-linux-gnu
            /usr/local/cuda/lib64
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    )
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        set(HAS_TENSORRT TRUE)
        set(TENSORRT_LIBRARIES 
            ${TENSORRT_LIBRARY}
            nvinfer_plugin
            nvonnxparser
        )
        set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
        include_directories(${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
        message(STATUS "Found TensorRT: ${TENSORRT_LIBRARY}")
    endif()
endif()

# HailoRT (RPi with Hailo AI Hat)
set(HAS_HAILO FALSE)
if(PLATFORM_RPI)
    find_path(HAILO_INCLUDE_DIR hailo/hailort.h
        PATHS /usr/include
    )
    find_library(HAILO_LIBRARY hailort
        PATHS /usr/lib
    )
    if(HAILO_INCLUDE_DIR AND HAILO_LIBRARY)
        set(HAS_HAILO TRUE)
        set(HAILO_LIBRARIES ${HAILO_LIBRARY})
        include_directories(${HAILO_INCLUDE_DIR})
        message(STATUS "Found HailoRT: ${HAILO_LIBRARY}")
    endif()
endif()

# NCNN (RPi CPU fallback)
set(HAS_NCNN FALSE)
find_package(ncnn QUIET)
if(ncnn_FOUND)
    set(HAS_NCNN TRUE)
    message(STATUS "Found NCNN")
endif()

# OpenVINO (x86 CPU)
set(HAS_OPENVINO FALSE)
if(PLATFORM_X86)
    find_package(OpenVINO QUIET COMPONENTS Runtime)
    if(OpenVINO_FOUND)
        set(HAS_OPENVINO TRUE)
        message(STATUS "Found OpenVINO: ${OpenVINO_VERSION}")
    endif()
endif()
