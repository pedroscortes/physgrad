# Test configuration
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    # Download and build Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50f33f9142a332e68d86e69d0dace0.zip
    )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Test sources
set(TEST_SOURCES
    test_physics_engine.cpp
    test_memory_manager.cpp
)

if(WITH_CUDA)
    list(APPEND TEST_SOURCES
        test_cuda_kernels.cu
        test_gpu_memory.cu
    )
endif()

# Create test executable
add_executable(physgrad_tests ${TEST_SOURCES})

# Link with PhysGrad and GTest
target_link_libraries(physgrad_tests
    physgrad_core
    gtest_main
    gtest
    pthread
)

if(WITH_CUDA)
    set_target_properties(physgrad_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Discover tests
include(GoogleTest)
gtest_discover_tests(physgrad_tests)

# Individual test executables for better isolation
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME}
        physgrad_core
        gtest_main
        gtest
        pthread
    )

    if(WITH_CUDA AND ${TEST_SOURCE} MATCHES ".*\\.cu$")
        set_target_properties(${TEST_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    endif()

    gtest_discover_tests(${TEST_NAME})
endforeach()

# Performance benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()