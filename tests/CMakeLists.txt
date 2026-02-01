# ============================================================================
# Tests CMakeLists.txt
# ============================================================================

# GTest
find_package(GTest REQUIRED)
include(GoogleTest)

# ============================================================================
# Unit Tests
# ============================================================================

# Core tests
add_executable(test_ring_buffer
    unit/test_ring_buffer.cpp
)
target_link_libraries(test_ring_buffer
    GTest::gtest_main
    GTest::gmock
    Threads::Threads
)
gtest_discover_tests(test_ring_buffer)

add_executable(test_spsc_queue
    unit/test_spsc_queue.cpp
)
target_link_libraries(test_spsc_queue
    GTest::gtest_main
    GTest::gmock
    Threads::Threads
)
gtest_discover_tests(test_spsc_queue)

add_executable(test_types
    unit/test_types.cpp
)
target_link_libraries(test_types
    GTest::gtest_main
)
gtest_discover_tests(test_types)

add_executable(test_config
    unit/test_config.cpp
)
target_link_libraries(test_config
    lagari_core
    GTest::gtest_main
)
gtest_discover_tests(test_config)

# Guidance tests
add_executable(test_pid_controller
    unit/test_pid_controller.cpp
)
target_link_libraries(test_pid_controller
    lagari_guidance
    GTest::gtest_main
)
gtest_discover_tests(test_pid_controller)

# ============================================================================
# Integration Tests
# ============================================================================

# TODO: Add integration tests as modules are implemented
