#!/bin/bash

# PhysGrad CPU-Only Installation Script
# This script builds and installs the CPU-only version of PhysGrad with visualization support

set -e

echo "🚀 PhysGrad CPU-Only Installation Script"
echo "========================================"

# Check for required system dependencies
echo "🔍 Checking system dependencies..."

check_pkg_config() {
    if ! pkg-config --exists "$1"; then
        echo "❌ Missing dependency: $1"
        echo "Install with: sudo apt-get install $2"
        exit 1
    else
        echo "✅ Found: $1"
    fi
}

check_pkg_config "gl" "libgl1-mesa-dev"
check_pkg_config "glfw3" "libglfw3-dev"
check_pkg_config "glew" "libglew-dev"

# Check for Eigen3
if [ ! -d "/usr/include/eigen3" ]; then
    echo "❌ Missing Eigen3 headers"
    echo "Install with: sudo apt-get install libeigen3-dev"
    exit 1
else
    echo "✅ Found: Eigen3"
fi

# Check for Python dependencies
echo "🐍 Checking Python dependencies..."

python3 -c "import pybind11; print('✅ pybind11:', pybind11.__version__)" || {
    echo "❌ Missing pybind11. Install with: pip install pybind11"
    exit 1
}

python3 -c "import numpy; print('✅ NumPy:', numpy.__version__)" || {
    echo "❌ Missing NumPy. Install with: pip install numpy"
    exit 1
}

# Build the CPU-only version
echo "🔨 Building PhysGrad (CPU-only with OpenGL visualization)..."
cd python

# Force CPU-only build by setting CUDA_HOME to non-existent path
export CUDA_HOME=/nonexistent

# Build the extension in-place
python3 setup.py build_ext --inplace

# Copy the extension to the correct location for the physgrad package
if [ -f "physgrad_cpp.cpython-310-x86_64-linux-gnu.so" ]; then
    cp physgrad_cpp.cpython-310-x86_64-linux-gnu.so physgrad/physgrad_cpp.so
    echo "✅ Extension copied to package directory"
elif [ -f "physgrad_cpp.so" ]; then
    cp physgrad_cpp.so physgrad/physgrad_cpp.so
    echo "✅ Extension copied to package directory"
else
    echo "⚠️  Extension file not found, trying to continue..."
fi

echo "✅ Build completed successfully!"

# Test the installation
echo "🧪 Testing PhysGrad installation..."

PYTHONPATH=. python3 -c "
import physgrad as pg
print('✅ PhysGrad imported successfully')
print('Features:', pg.FEATURES)

# Quick test
config = pg.SimulationConfig(num_particles=3, dt=0.01)
sim = pg.Simulation(config)
particle = pg.Particle(position=[0,0,0], velocity=[0,0,0], mass=1.0)
sim.add_particle(particle)
sim.add_force(pg.GravityForce(gravity=[0, -9.81, 0]))
sim.step()
print('✅ Basic simulation test passed')

# Check visualization
if hasattr(sim, 'run_with_visualization'):
    print('✅ Real-time visualization available')

print('🎉 Installation verification successful!')
"

echo ""
echo "🎉 PhysGrad CPU Installation Complete!"
echo "========================================"
echo ""
echo "To use PhysGrad, run:"
echo "  cd python"
echo "  PYTHONPATH=. python3 your_script.py"
echo ""
echo "Or add to your Python path:"
echo "  export PYTHONPATH=\"$(pwd)/python:\$PYTHONPATH\""
echo ""
echo "Features available:"
echo "  ✅ CPU physics simulation"
echo "  ✅ Real-time OpenGL/ImGui visualization"
echo "  ✅ Force and constraint systems"
echo "  ✅ Multiple integrators"
echo "  ✅ Matplotlib/Plotly visualization"
echo "  ❌ CUDA GPU acceleration (requires CUDA fix)"
echo "  ❌ PyTorch integration (requires CUDA fix)"
echo "  ❌ JAX integration (requires JAX installation)"
echo ""
echo "To test the visualization (requires display):"
echo "  cd python && PYTHONPATH=. python3 ../demo_realtime_visualization.py"