cmake_minimum_required(VERSION 3.18)
project(physgrad LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# CUDA architecture (modify based on your GPU)
# 70 = V100, 75 = RTX 2000, 80 = A100, 86 = RTX 3000, 89 = RTX 4000
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/simulation.cu
)

# Create executable
add_executable(physgrad ${SOURCES})

# Compiler flags
target_compile_options(physgrad PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
        --extended-lambda
    >
)

# Link libraries
target_link_libraries(physgrad
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    glfw
    GL
    ${CMAKE_DL_LIBS}
)

# Copy shaders to build directory
configure_file(shaders/particle.vert ${CMAKE_CURRENT_BINARY_DIR}/shaders/particle.vert COPYONLY)
configure_file(shaders/particle.frag ${CMAKE_CURRENT_BINARY_DIR}/shaders/particle.frag COPYONLY)

# Enable verbose CUDA compilation for debugging
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")

# Add tests subdirectory if it exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
endif()