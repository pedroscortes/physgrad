cmake_minimum_required(VERSION 3.18)
project(PhysGrad VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_VISUALIZATION "Enable OpenGL/ImGui visualization" ON)
option(WITH_PYTHON "Build Python bindings" ON)
option(BUILD_TESTS "Build test suite" ON)

# Find packages
find_package(Eigen3 REQUIRED)

if(WITH_VISUALIZATION)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GL REQUIRED gl)
    pkg_check_modules(GLFW REQUIRED glfw3)
    pkg_check_modules(GLEW REQUIRED glew)
endif()

if(WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)
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

# Source files
set(PHYSGRAD_SOURCES
    src/variational_contact.cpp
    src/differentiable_contact.cpp
    src/rigid_body.cpp
    src/symplectic_integrators.cpp
    src/constraints.cpp
    src/collision_detection.cpp
)

if(WITH_CUDA)
    list(APPEND PHYSGRAD_SOURCES
        src/variational_contact_gpu.cu
        src/variational_contact_kernels.cu
        src/simulation.cu
        src/stability_improvements.cu
    )
endif()

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

# Create main library
add_library(physgrad_core ${PHYSGRAD_SOURCES})

# Set target properties
target_compile_definitions(physgrad_core PRIVATE
    $<$<BOOL:${WITH_CUDA}>:WITH_CUDA>
    $<$<BOOL:${WITH_VISUALIZATION}>:IMGUI_IMPL_OPENGL_LOADER_GLEW>
)

target_link_libraries(physgrad_core
    Eigen3::Eigen
    $<$<BOOL:${WITH_VISUALIZATION}>:${GL_LIBRARIES}>
    $<$<BOOL:${WITH_VISUALIZATION}>:${GLFW_LIBRARIES}>
    $<$<BOOL:${WITH_VISUALIZATION}>:${GLEW_LIBRARIES}>
    $<$<BOOL:${WITH_CUDA}>:CUDA::cudart>
    $<$<BOOL:${WITH_CUDA}>:CUDA::cublas>
)

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

# Tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS physgrad_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if(WITH_PYTHON)
    install(TARGETS physgrad_cpp
        LIBRARY DESTINATION ${Python3_SITEARCH}
    )
endif()

# Print configuration summary
message(STATUS "PhysGrad Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  CUDA support: ${WITH_CUDA}")
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