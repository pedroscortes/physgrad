import os
import sys
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

from setuptools import setup, Extension, find_packages
import torch
from torch.utils import cpp_extension

# Define the extension module
def get_extensions():
    """Build the pybind11 extension."""

    # Get CUDA paths
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

    # Source files
    sources = [
        'src/physgrad_binding.cpp',
        'src/tensor_interop.cpp',
        'src/torch_integration.cpp',
        'src/jax_integration.cpp',
        '../src/simulation.cu',
        '../src/multi_gpu.cu',
        '../src/domain_decomposition.cu',
        '../src/gpu_communication.cu',
        '../src/load_balancer.cu',
        '../src/differentiable_contact.cpp',
        '../src/rigid_body.cpp',
        '../src/symplectic_integrators.cpp',
        '../src/constraints.cpp',
        '../src/collision_detection.cpp',
    ]

    # Include directories
    include_dirs = [
        pybind11.get_include(),
        torch.utils.cpp_extension.include_paths(),
        f'{cuda_home}/include',
        '../src',
        '/usr/include/eigen3',  # For linear algebra
    ]

    # Library directories
    library_dirs = [
        f'{cuda_home}/lib64',
        torch.utils.cpp_extension.library_paths(),
    ]

    # Libraries
    libraries = [
        'cudart',
        'cublas',
        'curand',
        'cusparse',
        'nccl',
    ] + torch.utils.cpp_extension.libraries()

    # Compiler flags
    cxx_flags = [
        '-std=c++17',
        '-O3',
        '-fPIC',
        '-DWITH_CUDA',
        '-DWITH_PYTORCH',
        '-DWITH_JAX',
    ]

    nvcc_flags = [
        '-std=c++17',
        '--use_fast_math',
        '-lineinfo',
        '--extended-lambda',
        '-arch=sm_75',  # Adjust based on target architecture
        '-DWITH_CUDA',
        '-DWITH_PYTORCH',
        '-DWITH_JAX',
    ]

    # Create extension
    ext = cpp_extension.CUDAExtension(
        name='physgrad_cpp',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': nvcc_flags
        },
        language='c++'
    )

    return [ext]

# Read long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="physgrad",
    version="0.1.0",
    author="PhysGrad Team",
    author_email="contact@physgrad.ai",
    description="Differentiable Physics Simulation with GPU Acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/physgrad/physgrad",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "opencv-python>=4.5.0",
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
    keywords="physics simulation gpu cuda pytorch jax differentiable",
    project_urls={
        "Bug Reports": "https://github.com/physgrad/physgrad/issues",
        "Source": "https://github.com/physgrad/physgrad",
        "Documentation": "https://physgrad.readthedocs.io",
    },
)