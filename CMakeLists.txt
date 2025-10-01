cmake_minimum_required(VERSION 3.18)
project(PhysGrad VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard (upgraded to C++20 for concepts)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enhanced project options
option(PHYSGRAD_ENABLE_CUDA "Enable CUDA support" ON)
option(PHYSGRAD_ENABLE_PYTORCH "Enable PyTorch integration" ON)
option(PHYSGRAD_ENABLE_VISUALIZATION "Enable OpenGL/ImGui visualization" ON)
option(PHYSGRAD_ENABLE_PYTHON "Build Python bindings" ON)
option(PHYSGRAD_ENABLE_TESTS "Build comprehensive test suite" ON)
option(PHYSGRAD_ENABLE_DEMOS "Build demonstration programs" ON)
option(PHYSGRAD_ENABLE_BENCHMARKS "Build performance benchmarks" ON)
option(PHYSGRAD_ENABLE_DOCUMENTATION "Build documentation" OFF)
option(PHYSGRAD_ENABLE_SANITIZERS "Enable address/thread sanitizers" OFF)
option(PHYSGRAD_AUTO_DETECT_CUDA_ARCH "Auto-detect CUDA architecture" ON)

# Legacy compatibility aliases
option(WITH_CUDA "Enable CUDA support" ON)
option(WITH_PYTORCH "Enable PyTorch integration" ON)
option(WITH_VISUALIZATION "Enable OpenGL/ImGui visualization" ON)
option(WITH_PYTHON "Build Python bindings" ON)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_DEMOS "Build demonstration programs" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# CUDA flags
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")

# CUDA architecture detection
if(PHYSGRAD_AUTO_DETECT_CUDA_ARCH)
    include(cmake/DetectCUDAArchitecture.cmake)
    detect_cuda_architecture(DETECTED_CUDA_ARCHITECTURES)
    if(DETECTED_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_CUDA_ARCHITECTURES})
        message(STATUS "Auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        # Fallback to sensible defaults based on CUDA version
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "12.0")
            set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")  # Volta, Turing, Ampere, Ada, Hopper
        elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.0")
            set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")  # Volta, Turing, Ampere
        else()
            set(CMAKE_CUDA_ARCHITECTURES "60;70;75")  # Pascal, Volta, Turing
        endif()
        message(STATUS "Using fallback CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
else()
    # Manual override - can be set via environment variable or cmake option
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "89")  # Default for RTX 2000 Ada
    endif()
    message(STATUS "Using manual CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Find CUDA libraries
    find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})
    if(CUDNN_LIBRARY)
        set(HAVE_CUDNN TRUE)
        message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
    else()
        set(HAVE_CUDNN FALSE)
        message(WARNING "cuDNN not found - some neural network features will be disabled")
    endif()
endif()

if(WITH_VISUALIZATION)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GL REQUIRED gl)
    pkg_check_modules(GLFW REQUIRED glfw3)
    pkg_check_modules(GLEW REQUIRED glew)
endif()

if(WITH_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)
endif()

# PyTorch detection
if(WITH_PYTORCH)
    # Get PyTorch CMake prefix path from Python installation
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE TORCH_PYTHON_RESULT
    )

    if(TORCH_PYTHON_RESULT EQUAL 0 AND TORCH_CMAKE_PREFIX_PATH)
        # Use PyTorch's proper CMake configuration
        set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}")
        find_package(Torch QUIET)

        if(Torch_FOUND)
            set(TORCH_FOUND TRUE)
            message(STATUS "Found PyTorch via Python CMake path")

            # Check if CUDA support is available
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.cuda.is_available())"
                OUTPUT_VARIABLE PYTORCH_CUDA_AVAILABLE
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )

            if(PYTORCH_CUDA_AVAILABLE STREQUAL "True")
                set(TORCH_CUDA_FOUND TRUE)
                message(STATUS "  PyTorch CUDA support: YES")
            else()
                set(TORCH_CUDA_FOUND FALSE)
                message(STATUS "  PyTorch CUDA support: NO")
            endif()
        endif()
    endif()

    if(NOT TORCH_FOUND)
        message(WARNING "PyTorch not found - ML features will be disabled")
        set(WITH_PYTORCH OFF)
    else()
        add_definitions(-DHAVE_PYTORCH)
        if(TORCH_CUDA_FOUND)
            add_definitions(-DHAVE_PYTORCH_CUDA)
        endif()
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
)

if(WITH_VISUALIZATION)
    include_directories(
        ${CMAKE_SOURCE_DIR}/external/imgui
        ${CMAKE_SOURCE_DIR}/external/imgui/backends
        ${GL_INCLUDE_DIRS}
        ${GLFW_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
    )
endif()

# Core PhysGrad library sources
set(PHYSGRAD_SOURCES
    # Original contact mechanics
    src/variational_contact.cpp
    src/differentiable_contact.cpp
    src/rigid_body.cpp
    src/symplectic_integrators.cpp
    src/constraints.cpp
    src/collision_detection.cpp

    # New comprehensive physics systems
    src/physics_engine.cpp
    src/contact_mechanics.cpp
    src/fluid_dynamics.cpp
    # src/soft_body_dynamics.cpp  # Temporarily disabled due to thrust usage
    src/electromagnetic_fields.cpp
    src/multi_scale_physics.cpp
    src/gpu_memory_manager.cpp
    # src/mpi_physics.cpp  # Temporarily disabled - requires MPI
    # src/physics_streaming.cpp  # Temporarily disabled - requires websocketpp and nlohmann::json
    # src/neural_fluid_dynamics.cpp  # Temporarily disabled - has compilation issues
    # src/symbolic_physics_ai.cpp  # Temporarily disabled - incomplete type issues
    # src/physics_generative_models.cpp  # Temporarily disabled - CUDA kernel launch in cpp file
    # src/quantum_classical_coupling.cpp  # Temporarily disabled - incomplete type issues
)

# CUDA kernel sources
set(PHYSGRAD_CUDA_SOURCES
    # Original CUDA sources
    src/variational_contact_gpu.cu
    src/variational_contact_kernels.cu
    src/simulation.cu
    src/stability_improvements.cu

    # New CUDA kernels
    src/physics_kernels.cu
    src/contact_kernels.cu
    src/fluid_kernels.cu
    src/softbody_kernels.cu
    src/electromagnetic_kernels.cu
    src/electromagnetic_fields_gpu.cu
    # src/soft_body_dynamics_gpu.cu  # Temporarily disabled
    src/memory_kernels.cu
    src/neural_kernels.cu
    src/symbolic_kernels.cu
    src/generative_kernels.cu
    src/quantum_kernels.cu
)

# Demonstration programs
set(DEMO_SOURCES
    demo_contact_mechanics.cu
    demo_fluid_dynamics.cu
    demo_soft_body_dynamics.cu
    demo_electromagnetic_fields.cu
    demo_multi_scale_physics.cu
    demo_gpu_memory_management.cu
    demo_mpi_physics.cu
    demo_physics_streaming.cu
    demo_neural_fluid_dynamics.cu
    demo_symbolic_physics_ai.cu
    demo_physics_generative_models.cu
    demo_quantum_classical_coupling.cu
)

if(WITH_VISUALIZATION)
    list(APPEND PHYSGRAD_SOURCES
        src/visualization.cpp
        external/imgui/imgui.cpp
        external/imgui/imgui_draw.cpp
        external/imgui/imgui_tables.cpp
        external/imgui/imgui_widgets.cpp
        external/imgui/backends/imgui_impl_glfw.cpp
        external/imgui/backends/imgui_impl_opengl3.cpp
    )
endif()

# Define preprocessor macros
add_definitions(-DHAVE_EIGEN)
if(WITH_CUDA)
    add_definitions(-DHAVE_CUDA)
    add_definitions(-DHAVE_CUBLAS)
    add_definitions(-DHAVE_CUFFT)
    add_definitions(-DHAVE_CURAND)
endif()
if(HAVE_CUDNN)
    add_definitions(-DHAVE_CUDNN)
endif()

# Create the main PhysGrad library
add_library(physgrad_core ${PHYSGRAD_SOURCES})

# Create CUDA kernels library
if(WITH_CUDA)
    add_library(physgrad_kernels STATIC ${PHYSGRAD_CUDA_SOURCES})

    # Set CUDA properties for kernel library
    set_target_properties(physgrad_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Set target properties
target_compile_definitions(physgrad_core PRIVATE
    $<$<BOOL:${WITH_CUDA}>:WITH_CUDA>
    $<$<BOOL:${WITH_VISUALIZATION}>:IMGUI_IMPL_OPENGL_LOADER_GLEW>
)

# Link libraries
target_link_libraries(physgrad_core
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${MPI_LIBRARIES}
    $<$<BOOL:${WITH_VISUALIZATION}>:${GL_LIBRARIES}>
    $<$<BOOL:${WITH_VISUALIZATION}>:${GLFW_LIBRARIES}>
    $<$<BOOL:${WITH_VISUALIZATION}>:${GLEW_LIBRARIES}>
)

# Link PyTorch if available (using proper target-based linking)
if(WITH_PYTORCH AND TORCH_FOUND)
    target_link_libraries(physgrad_core ${TORCH_LIBRARIES})
    if(TORCH_CUDA_FOUND)
        # Add CUDA runtime for proper linking
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            target_link_libraries(physgrad_core CUDA::cudart)
        endif()
    endif()
endif()

if(WITH_CUDA)
    target_link_libraries(physgrad_core
        physgrad_kernels
        CUDA::cudart
        CUDA::cublas
        CUDA::cufft
        CUDA::curand
    )

    if(HAVE_CUDNN)
        target_link_libraries(physgrad_core ${CUDNN_LIBRARY})
    endif()
endif()

if(WITH_VISUALIZATION)
    target_compile_options(physgrad_core PRIVATE ${GL_CFLAGS_OTHER} ${GLFW_CFLAGS_OTHER} ${GLEW_CFLAGS_OTHER})
endif()

# Python bindings
if(WITH_PYTHON)
    pybind11_add_module(physgrad_cpp
        python/src/physgrad_binding_simple.cpp
    )

    target_link_libraries(physgrad_cpp PRIVATE physgrad_core)

    target_compile_definitions(physgrad_cpp PRIVATE
        VERSION_INFO="${PROJECT_VERSION}"
        $<$<BOOL:${WITH_CUDA}>:WITH_CUDA>
        $<$<BOOL:${WITH_VISUALIZATION}>:WITH_VISUALIZATION>
    )
endif()

# Create demo executables
if(BUILD_DEMOS)
    foreach(DEMO_SOURCE ${DEMO_SOURCES})
        get_filename_component(DEMO_NAME ${DEMO_SOURCE} NAME_WE)
        add_executable(${DEMO_NAME} ${DEMO_SOURCE})
        target_link_libraries(${DEMO_NAME} physgrad_core)

        if(WITH_CUDA)
            set_target_properties(${DEMO_NAME} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
            )
        endif()
    endforeach()
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
set(INSTALL_TARGETS physgrad_core)
if(WITH_CUDA)
    list(APPEND INSTALL_TARGETS physgrad_kernels)
endif()

install(TARGETS ${INSTALL_TARGETS}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY src/
    DESTINATION include/physgrad
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# Install demo programs
if(BUILD_DEMOS)
    foreach(DEMO_SOURCE ${DEMO_SOURCES})
        get_filename_component(DEMO_NAME ${DEMO_SOURCE} NAME_WE)
        install(TARGETS ${DEMO_NAME} DESTINATION bin)
    endforeach()
endif()

if(WITH_PYTHON)
    install(TARGETS physgrad_cpp
        LIBRARY DESTINATION ${Python3_SITEARCH}
    )
endif()

# Print configuration summary
message(STATUS "PhysGrad Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  CUDA support: ${WITH_CUDA}")
message(STATUS "  PyTorch support: ${WITH_PYTORCH}")
message(STATUS "  Visualization: ${WITH_VISUALIZATION}")
message(STATUS "  Python bindings: ${WITH_PYTHON}")
message(STATUS "  Build tests: ${BUILD_TESTS}")

if(WITH_VISUALIZATION)
    message(STATUS "  OpenGL: ${GL_VERSION}")
    message(STATUS "  GLFW: ${GLFW_VERSION}")
    message(STATUS "  GLEW: ${GLEW_VERSION}")
endif()

if(WITH_PYTHON)
    message(STATUS "  Python: ${Python3_VERSION}")
    message(STATUS "  pybind11: ${pybind11_VERSION}")
endif()