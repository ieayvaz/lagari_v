cmake_minimum_required(VERSION 3.18)
project(lagari_v 
    VERSION 1.0.0 
    LANGUAGES CXX
    DESCRIPTION "Multi-platform UAV Vision System"
)

# ============================================================================
# Build Options
# ============================================================================
option(BUILD_TESTS "Build unit and integration tests" ON)
option(BUILD_EXAMPLES "Build example applications" OFF)
option(ENABLE_SANITIZERS "Enable address/undefined sanitizers" OFF)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Platform Detection and Configuration
# ============================================================================
include(cmake/Platform.cmake)
include(cmake/Dependencies.cmake)

# ============================================================================
# Compiler Flags
# ============================================================================
add_compile_options(-Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG)
    if(NOT PLATFORM_RPI)
        add_compile_options(-march=native)
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
endif()

if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

# ============================================================================
# Include Directories
# ============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# Core Library
# ============================================================================
add_library(lagari_core STATIC
    src/core/config.cpp
    src/core/logger.cpp
    src/core/system_state.cpp
)
target_include_directories(lagari_core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(lagari_core PUBLIC
    yaml-cpp
    spdlog::spdlog
    Threads::Threads
)

# ============================================================================
# Capture Module
# ============================================================================
set(CAPTURE_SOURCES
    src/capture/capture_factory.cpp
    src/capture/sim_capture.cpp
    src/capture/isaac_shm_capture.cpp  # Shared memory for Isaac Sim
)

if(PLATFORM_JETSON AND HAS_ARGUS)
    list(APPEND CAPTURE_SOURCES src/capture/argus_capture.cpp)
endif()

if(PLATFORM_RPI AND HAS_LIBCAMERA)
    list(APPEND CAPTURE_SOURCES src/capture/libcamera_capture.cpp)
endif()

if(HAS_V4L2)
    list(APPEND CAPTURE_SOURCES src/capture/v4l2_capture.cpp)
endif()

if(HAS_GSTREAMER)
    list(APPEND CAPTURE_SOURCES src/capture/gstreamer_capture.cpp)
endif()

add_library(lagari_capture STATIC ${CAPTURE_SOURCES})
target_link_libraries(lagari_capture PUBLIC
    lagari_core
    ${OpenCV_LIBS}
    rt  # For POSIX shared memory (shm_open, etc.)
)

if(PLATFORM_JETSON AND HAS_ARGUS)
    target_link_libraries(lagari_capture PRIVATE ${ARGUS_LIBRARIES})
    target_compile_definitions(lagari_capture PRIVATE HAS_ARGUS)
endif()

if(PLATFORM_RPI AND HAS_LIBCAMERA)
    target_link_libraries(lagari_capture PRIVATE ${LIBCAMERA_LIBRARIES})
    target_compile_definitions(lagari_capture PRIVATE HAS_LIBCAMERA)
endif()

if(HAS_V4L2)
    target_compile_definitions(lagari_capture PRIVATE HAS_V4L2)
endif()

if(HAS_GSTREAMER)
    target_link_libraries(lagari_capture PRIVATE ${GSTREAMER_LIBRARIES})
    target_compile_definitions(lagari_capture PRIVATE HAS_GSTREAMER)
endif()

# ============================================================================
# Detection Module
# ============================================================================
set(DETECTION_SOURCES
    src/detection/detector_factory.cpp
    src/detection/yolo_processor.cpp
)

if(HAS_TENSORRT)
    list(APPEND DETECTION_SOURCES src/detection/tensorrt_detector.cpp)
endif()

if(HAS_HAILO)
    list(APPEND DETECTION_SOURCES src/detection/hailo_detector.cpp)
endif()

if(HAS_NCNN)
    list(APPEND DETECTION_SOURCES src/detection/ncnn_detector.cpp)
endif()

if(HAS_OPENVINO)
    list(APPEND DETECTION_SOURCES src/detection/openvino_detector.cpp)
endif()

add_library(lagari_detection STATIC ${DETECTION_SOURCES})
target_link_libraries(lagari_detection PUBLIC
    lagari_core
    ${OpenCV_LIBS}
)

if(HAS_TENSORRT)
    target_link_libraries(lagari_detection PRIVATE ${TENSORRT_LIBRARIES} ${CUDA_LIBRARIES})
    target_compile_definitions(lagari_detection PRIVATE HAS_TENSORRT)
endif()

if(HAS_HAILO)
    target_link_libraries(lagari_detection PRIVATE ${HAILO_LIBRARIES})
    target_compile_definitions(lagari_detection PRIVATE HAS_HAILO)
endif()

if(HAS_NCNN)
    target_link_libraries(lagari_detection PRIVATE ncnn)
    target_compile_definitions(lagari_detection PRIVATE HAS_NCNN)
endif()

if(HAS_OPENVINO)
    target_link_libraries(lagari_detection PRIVATE openvino::runtime)
    target_compile_definitions(lagari_detection PRIVATE HAS_OPENVINO)
endif()

# ============================================================================
# QR Module
# ============================================================================
add_library(lagari_qr STATIC
    src/qr/qr_decoder.cpp
)
target_link_libraries(lagari_qr PUBLIC
    lagari_core
    ${OpenCV_LIBS}
    ${ZBAR_LIBRARIES}
)

# ============================================================================
# Guidance Module
# ============================================================================
add_library(lagari_guidance STATIC
    src/guidance/guidance.cpp
    src/guidance/tracker.cpp
)
target_link_libraries(lagari_guidance PUBLIC
    lagari_core
)

# ============================================================================
# Communications Module
# ============================================================================
add_library(lagari_comms STATIC
    src/comms/mavlink_interface.cpp
    src/comms/telemetry.cpp
    src/comms/udp_transport.cpp
    src/comms/serial_transport.cpp
)
target_link_libraries(lagari_comms PUBLIC
    lagari_core
)
target_include_directories(lagari_comms PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/mavlink/common
)

# ============================================================================
# Recording Module
# ============================================================================
add_library(lagari_recording STATIC
    src/recording/recorder.cpp
    src/recording/overlay_renderer.cpp
)
target_link_libraries(lagari_recording PUBLIC
    lagari_core
    ${OpenCV_LIBS}
)

# ============================================================================
# Main Executable
# ============================================================================
add_executable(lagari_vision src/main.cpp)
target_link_libraries(lagari_vision PRIVATE
    lagari_core
    lagari_capture
    lagari_detection
    lagari_qr
    lagari_guidance
    lagari_comms
    lagari_recording
)

# ============================================================================
# Tests
# ============================================================================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS lagari_vision
    RUNTIME DESTINATION bin
)
install(DIRECTORY config/
    DESTINATION etc/lagari
)
install(DIRECTORY models/
    DESTINATION share/lagari/models
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== Lagari Vision Build Configuration ===")
message(STATUS "Platform: ${LAGARI_PLATFORM}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Capture backends:")
message(STATUS "  Argus (Jetson): ${HAS_ARGUS}")
message(STATUS "  libcamera (RPi): ${HAS_LIBCAMERA}")
message(STATUS "  V4L2: ${HAS_V4L2}")
message(STATUS "  GStreamer: ${HAS_GSTREAMER}")
message(STATUS "  Isaac Sim (SHM): TRUE")  # Always available on Linux
message(STATUS "")
message(STATUS "Inference backends:")
message(STATUS "  TensorRT: ${HAS_TENSORRT}")
message(STATUS "  HailoRT: ${HAS_HAILO}")
message(STATUS "  NCNN: ${HAS_NCNN}")
message(STATUS "  OpenVINO: ${HAS_OPENVINO}")
message(STATUS "")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "==========================================")
message(STATUS "")
