# ============================================================================
# Required Dependencies
# ============================================================================

# Threads
find_package(Threads REQUIRED)

# OpenCV
find_package(OpenCV 4.0 REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    videoio
    highgui
)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# yaml-cpp
find_package(yaml-cpp REQUIRED)
message(STATUS "Found yaml-cpp: ${yaml-cpp_VERSION}")

# spdlog
find_package(spdlog REQUIRED)
message(STATUS "Found spdlog: ${spdlog_VERSION}")

# ZBar for QR decoding
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZBAR REQUIRED zbar)
include_directories(${ZBAR_INCLUDE_DIRS})
message(STATUS "Found ZBar: ${ZBAR_VERSION}")

# ============================================================================
# Testing Dependencies (optional)
# ============================================================================
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    message(STATUS "Found GTest: ${GTest_VERSION}")
endif()
