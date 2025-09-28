import os
import sys
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

from setuptools import setup, Extension, find_packages

def check_dependencies():
    """Check if required system dependencies are available."""
    missing_deps = []

    # Check for OpenGL libraries
    try:
        subprocess.check_output(['pkg-config', '--exists', 'gl'])
    except subprocess.CalledProcessError:
        missing_deps.append('OpenGL development libraries (libgl1-mesa-dev)')

    # Check for GLFW
    try:
        subprocess.check_output(['pkg-config', '--exists', 'glfw3'])
    except subprocess.CalledProcessError:
        missing_deps.append('GLFW3 development libraries (libglfw3-dev)')

    # Check for GLEW
    try:
        subprocess.check_output(['pkg-config', '--exists', 'glew'])
    except subprocess.CalledProcessError:
        missing_deps.append('GLEW development libraries (libglew-dev)')

    if missing_deps:
        print("ERROR: Missing required system dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with: sudo apt-get install libgl1-mesa-dev libglfw3-dev libglew-dev")
        sys.exit(1)


def get_extensions():
    """Build the pybind11 extension with all necessary sources."""

    # Check system dependencies
    check_dependencies()

    # Get CUDA paths
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

    # Core C++ sources
    cpp_sources = [
        'src/physgrad_binding_simple.cpp',  # Our main binding file
        '../src/variational_contact.cpp',
        '../src/differentiable_contact.cpp',
        '../src/rigid_body.cpp',
        '../src/symplectic_integrators.cpp',
        '../src/constraints.cpp',
        '../src/collision_detection.cpp',
        '../src/visualization.cpp',  # Visualization system
    ]

    # CUDA sources
    cuda_sources = [
        '../src/variational_contact_gpu.cu',
        '../src/variational_contact_kernels.cu',
        '../src/simulation.cu',
        '../src/stability_improvements.cu',
    ]

    # ImGui sources
    imgui_sources = [
        '../external/imgui/imgui.cpp',
        '../external/imgui/imgui_draw.cpp',
        '../external/imgui/imgui_tables.cpp',
        '../external/imgui/imgui_widgets.cpp',
        '../external/imgui/backends/imgui_impl_glfw.cpp',
        '../external/imgui/backends/imgui_impl_opengl3.cpp',
    ]

    # Check if CUDA is available
    cuda_available = os.path.exists(cuda_home) and os.path.exists(f'{cuda_home}/include/cuda.h')

    if cuda_available:
        all_sources = cpp_sources + cuda_sources + imgui_sources
    else:
        all_sources = cpp_sources + imgui_sources
        print("WARNING: CUDA not found, building CPU-only version")

    # Include directories
    include_dirs = [
        pybind11.get_include(),
        '../src',
        '../external/imgui',
        '../external/imgui/backends',
        '/usr/include/eigen3',
        '/usr/include/GL',
    ]

    if cuda_available:
        include_dirs.append(f'{cuda_home}/include')

    # Get system library flags
    gl_flags = subprocess.check_output(['pkg-config', '--cflags', 'gl']).decode().strip().split()
    glfw_flags = subprocess.check_output(['pkg-config', '--cflags', 'glfw3']).decode().strip().split()
    glew_flags = subprocess.check_output(['pkg-config', '--cflags', 'glew']).decode().strip().split()

    gl_libs = subprocess.check_output(['pkg-config', '--libs', 'gl']).decode().strip().split()
    glfw_libs = subprocess.check_output(['pkg-config', '--libs', 'glfw3']).decode().strip().split()
    glew_libs = subprocess.check_output(['pkg-config', '--libs', 'glew']).decode().strip().split()

    # Compiler flags
    cxx_flags = [
        '-std=c++17',
        '-O3',
        '-fPIC',
        '-Wall',
        '-Wno-unused-variable',
        '-DIMGUI_IMPL_OPENGL_LOADER_GLEW',
    ]

    if cuda_available:
        cxx_flags.extend(['-DWITH_CUDA', '-DCUDA_AVAILABLE'])

    # Add system flags
    cxx_flags.extend(gl_flags + glfw_flags + glew_flags)

    # Linker flags
    link_args = gl_libs + glfw_libs + glew_libs

    if cuda_available:
        link_args.extend([f'-L{cuda_home}/lib64', '-lcudart', '-lcublas'])

    # Create extension
    if cuda_available:
        # Use CUDA extension for full GPU support
        try:
            import torch
            from torch.utils import cpp_extension

            nvcc_flags = [
                '-std=c++17',
                '--use_fast_math',
                '-lineinfo',
                '--extended-lambda',
                '-arch=sm_75',  # Adjust based on target architecture
                '-DWITH_CUDA',
                '-DCUDA_AVAILABLE',
            ]

            ext = cpp_extension.CUDAExtension(
                name='physgrad_cpp',
                sources=all_sources,
                include_dirs=include_dirs,
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': nvcc_flags
                },
                extra_link_args=link_args,
                language='c++'
            )
        except ImportError:
            print("WARNING: PyTorch not available, building CPU-only version")
            cuda_available = False

    if not cuda_available:
        # CPU-only version using pybind11
        ext = Pybind11Extension(
            name='physgrad_cpp',
            sources=cpp_sources + imgui_sources,
            include_dirs=include_dirs,
            cxx_std=17,
            extra_compile_args=cxx_flags,
            extra_link_args=link_args,
        )

    return [ext]


# Read requirements
def read_requirements():
    """Read requirements from requirements.txt if it exists."""
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        return req_file.read_text().strip().split('\n')
    else:
        return [
            "numpy>=1.20.0",
            "scipy>=1.8.0",
            "matplotlib>=3.5.0",
        ]


# Read long description
this_directory = Path(__file__).parent
readme_file = this_directory / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text()
else:
    long_description = "Differentiable Physics Simulation with GPU Acceleration and Real-time Visualization"

# Determine build command for setup
cmdclass = {}
try:
    import torch
    from torch.utils import cpp_extension
    cmdclass['build_ext'] = cpp_extension.BuildExtension.with_options(use_ninja=False)
except ImportError:
    cmdclass['build_ext'] = build_ext

setup(
    name="physgrad",
    version="0.1.0",
    author="PhysGrad Team",
    author_email="contact@physgrad.ai",
    description="Differentiable Physics Simulation with GPU Acceleration and Real-time Visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/physgrad/physgrad",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gpu": [
            "torch>=1.12.0",
        ],
        "jax": [
            "jax>=0.3.0",
            "jaxlib>=0.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="physics simulation gpu cuda pytorch jax differentiable visualization opengl",
    project_urls={
        "Bug Reports": "https://github.com/physgrad/physgrad/issues",
        "Source": "https://github.com/physgrad/physgrad",
        "Documentation": "https://physgrad.readthedocs.io",
    },
)