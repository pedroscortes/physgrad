#!/bin/bash
#
# Build script for PhysGrad Adjoint PyTorch Extension
#
# This script builds the C++ extension that provides Python bindings
# for the unified adjoint integrator.
#
# Usage:
#   ./build_adjoint_extension.sh
#
# Requirements:
#   - PyTorch with C++ extensions support
#   - pybind11
#   - C++17 compiler
#

set -e  # Exit on error

echo "========================================================================"
echo "Building PhysGrad Adjoint PyTorch Extension"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "ERROR: setup.py not found. Run this script from the python/ directory."
    exit 1
fi

# Check for PyTorch
echo "Checking for PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "ERROR: PyTorch not found. Install with: pip install torch"
    exit 1
}

# Check for pybind11
echo "Checking for pybind11..."
python3 -c "import pybind11; print(f'pybind11 version: {pybind11.__version__}')" || {
    echo "ERROR: pybind11 not found. Install with: pip install pybind11"
    exit 1
}

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info
rm -f physgrad/adjoint_verlet_cpp*.so

# Build extension
echo ""
echo "Building C++ extension..."
python3 setup.py build_ext --inplace

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ Build successful!"
    echo "========================================================================"
    echo ""
    echo "Extension installed in: physgrad/"
    ls -lh physgrad/adjoint_verlet_cpp*.so 2>/dev/null || echo "Note: Extension file not found in expected location"
    echo ""
    echo "Test with:"
    echo "  python3 -c 'from physgrad.adjoint import SpringMassSystem; print(\"Success!\")'"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "✗ Build failed!"
    echo "========================================================================"
    exit 1
fi
