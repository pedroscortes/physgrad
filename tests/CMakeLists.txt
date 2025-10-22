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
    test_contact_mechanics.cpp
    test_fluid_dynamics.cpp
    test_memory_optimization.cpp
    test_concepts.cpp
    test_adjoint_integrators.cpp
    test_adjoint_standalone.cpp
    test_differentiable_contact.cpp
    test_gradient_verification.cpp
    test_symplectic_energy_conservation.cpp
    test_mpm_materials.cpp
    test_gradient_flow_validation.cpp
)

if(WITH_PYTORCH AND TORCH_FOUND)
    list(APPEND TEST_SOURCES test_pytorch_integration.cpp)
endif()

if(WITH_CUDA)
    list(APPEND TEST_SOURCES
        test_cuda_kernels.cu
        test_mpm_kernels.cu
        test_adjoint_kernels.cu
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
        target_link_libraries(${TEST_NAME} physgrad_kernels)
        set_target_properties(${TEST_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    endif()

    # Link PyTorch for PyTorch integration tests
    if(WITH_PYTORCH AND TORCH_FOUND AND TEST_NAME STREQUAL "test_pytorch_integration")
        # Use PyTorch's bundled CUDA runtime (has cudaGetDriverEntryPointByVersion symbol)
        set(PYTORCH_CUDA_RUNTIME_DIR "$ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib")

        # Add PyTorch's CUDA runtime directory
        target_link_directories(${TEST_NAME} BEFORE PRIVATE ${PYTORCH_CUDA_RUNTIME_DIR})

        # Link libraries: physgrad_kernels, then PyTorch
        target_link_libraries(${TEST_NAME}
            physgrad_kernels
            ${TORCH_LIBRARIES}
        )

        # Set RPATH to find PyTorch's CUDA runtime at runtime
        set_target_properties(${TEST_NAME} PROPERTIES
            BUILD_RPATH "${PYTORCH_CUDA_RUNTIME_DIR}:${CMAKE_CURRENT_BINARY_DIR}/.."
            INSTALL_RPATH "${PYTORCH_CUDA_RUNTIME_DIR}"
            BUILD_WITH_INSTALL_RPATH OFF
        )

        message(STATUS "PyTorch test will use CUDA runtime from: ${PYTORCH_CUDA_RUNTIME_DIR}")
    endif()

    gtest_discover_tests(${TEST_NAME})
endforeach()

# Performance benchmarks
if(BUILD_BENCHMARKS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
        add_subdirectory(benchmarks)
    endif()
endif()