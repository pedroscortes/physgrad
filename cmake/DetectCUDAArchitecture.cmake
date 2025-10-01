# DetectCUDAArchitecture.cmake
# Automatically detect CUDA GPU architecture or provide flexible configuration options

# Function to detect local GPU compute capability
function(detect_cuda_architecture output_var)
    set(DETECT_CUDA_CODE "
#include <cuda_runtime.h>
#include <iostream>
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << \"No CUDA devices found\" << std::endl;
        return 1;
    }

    // Get the most capable device
    int maxDevice = 0;
    int maxMajor = 0, maxMinor = 0;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        if (prop.major > maxMajor || (prop.major == maxMajor && prop.minor > maxMinor)) {
            maxMajor = prop.major;
            maxMinor = prop.minor;
            maxDevice = dev;
        }
    }

    std::cout << maxMajor << maxMinor << std::endl;
    return 0;
}")

    # Write detection program
    set(DETECT_FILE "${CMAKE_BINARY_DIR}/detect_cuda_arch.cu")
    file(WRITE "${DETECT_FILE}" "${DETECT_CUDA_CODE}")

    # Try to compile and run detection program
    try_run(RUN_RESULT COMPILE_RESULT
        "${CMAKE_BINARY_DIR}"
        "${DETECT_FILE}"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        LINK_LIBRARIES "${CMAKE_CUDA_RUNTIME_LIBRARY}"
        RUN_OUTPUT_VARIABLE DETECTED_ARCH
        COMPILE_OUTPUT_VARIABLE COMPILE_OUT)

    if(COMPILE_RESULT AND RUN_RESULT EQUAL 0)
        string(STRIP "${DETECTED_ARCH}" DETECTED_ARCH)
        set(${output_var} "${DETECTED_ARCH}" PARENT_SCOPE)
        message(STATUS "Auto-detected CUDA architecture: ${DETECTED_ARCH}")
    else()
        set(${output_var} "" PARENT_SCOPE)
        message(WARNING "Failed to auto-detect CUDA architecture")
        if(NOT COMPILE_RESULT)
            message(WARNING "Compilation failed: ${COMPILE_OUT}")
        endif()
    endif()
endfunction()

# Function to configure CUDA architectures with fallbacks
function(configure_cuda_architectures)
    # Option 1: User explicitly specified architectures
    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        message(STATUS "Using user-specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        return()
    endif()

    # Option 2: Environment variable
    if(DEFINED ENV{CUDA_ARCHITECTURES} AND NOT "$ENV{CUDA_ARCHITECTURES}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDA_ARCHITECTURES}" PARENT_SCOPE)
        message(STATUS "Using CUDA_ARCHITECTURES from environment: $ENV{CUDA_ARCHITECTURES}")
        return()
    endif()

    # Option 3: Auto-detection
    detect_cuda_architecture(DETECTED_ARCH)
    if(NOT DETECTED_ARCH STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "${DETECTED_ARCH}" PARENT_SCOPE)
        message(STATUS "Using auto-detected CUDA architecture: ${DETECTED_ARCH}")
        return()
    endif()

    # Option 4: Reasonable defaults based on CUDA version
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
        # CUDA 12.x supports up to compute capability 9.0
        set(DEFAULT_ARCHITECTURES "60;70;75;80;86;89;90")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
        # CUDA 11.x supports up to compute capability 8.6
        set(DEFAULT_ARCHITECTURES "60;70;75;80;86")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "10.0")
        # CUDA 10.x supports up to compute capability 7.5
        set(DEFAULT_ARCHITECTURES "60;70;75")
    else()
        # Older CUDA versions
        set(DEFAULT_ARCHITECTURES "60;70")
    endif()

    set(CMAKE_CUDA_ARCHITECTURES "${DEFAULT_ARCHITECTURES}" PARENT_SCOPE)
    message(STATUS "Using default CUDA architectures for CUDA ${CMAKE_CUDA_COMPILER_VERSION}: ${DEFAULT_ARCHITECTURES}")
endfunction()

# Function to get architecture-specific compile flags
function(get_cuda_arch_flags arch_list output_var)
    set(FLAGS "")
    foreach(ARCH ${arch_list})
        set(FLAGS "${FLAGS} -gencode arch=compute_${ARCH},code=sm_${ARCH}")
    endforeach()
    # Add PTX for the highest architecture for forward compatibility
    list(GET arch_list -1 HIGHEST_ARCH)
    set(FLAGS "${FLAGS} -gencode arch=compute_${HIGHEST_ARCH},code=compute_${HIGHEST_ARCH}")
    set(${output_var} "${FLAGS}" PARENT_SCOPE)
endfunction()

# Main configuration function
function(setup_cuda_architectures)
    configure_cuda_architectures()

    # Convert semicolon list to space-separated for display
    string(REPLACE ";" " " ARCH_DISPLAY "${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "Final CUDA architectures: ${ARCH_DISPLAY}")

    # Validate architectures
    foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
        if(NOT ARCH MATCHES "^[0-9]+$")
            message(FATAL_ERROR "Invalid CUDA architecture: ${ARCH}")
        endif()
    endforeach()

    # Set the architectures in parent scope
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" PARENT_SCOPE)
endfunction()