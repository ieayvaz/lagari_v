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
    dnn
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

# ZeroMQ for Isaac Sim capture (optional)
set(HAS_ZMQ FALSE)
pkg_check_modules(ZMQ libzmq)
if(ZMQ_FOUND)
    set(HAS_ZMQ TRUE)
    include_directories(${ZMQ_INCLUDE_DIRS})
    message(STATUS "Found ZeroMQ: ${ZMQ_VERSION}")
else()
    # Try finding cppzmq header
    find_path(CPPZMQ_INCLUDE_DIR zmq.hpp
        PATHS /usr/include /usr/local/include
    )
    find_library(ZMQ_LIBRARY zmq
        PATHS /usr/lib /usr/local/lib
    )
    if(CPPZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
        set(HAS_ZMQ TRUE)
        set(ZMQ_LIBRARIES ${ZMQ_LIBRARY})
        include_directories(${CPPZMQ_INCLUDE_DIR})
        message(STATUS "Found ZeroMQ (manual): ${ZMQ_LIBRARY}")
    else()
        message(STATUS "ZeroMQ not found (Isaac Sim capture disabled)")
    endif()
endif()

# ============================================================================
# Testing Dependencies (optional)
# ============================================================================
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    message(STATUS "Found GTest: ${GTest_VERSION}")
endif()
