# PhysGrad Build & Installation Guide

This guide covers building and installing PhysGrad with its real-time OpenGL/ImGui visualization system.

## 🎯 **System Verification Status**

✅ **WORKING:** CPU-only version with full OpenGL/ImGui visualization
⚠️  **PARTIAL:** CUDA GPU acceleration (compilation issues)
❌ **BLOCKED:** PyTorch/JAX integration (requires CUDA fix)

## 📋 **Prerequisites**

### System Dependencies (Ubuntu/Debian)

```bash
# OpenGL and windowing libraries
sudo apt-get install libgl1-mesa-dev libglfw3-dev libglew-dev

# Math libraries
sudo apt-get install libeigen3-dev

# Build tools
sudo apt-get install build-essential pkg-config
```

### Python Dependencies

```bash
pip install numpy scipy matplotlib pybind11
```

### Optional Dependencies

```bash
# For PyTorch integration (when CUDA is fixed)
pip install torch

# For JAX integration
pip install jax jaxlib

# For enhanced visualization
pip install plotly dash
```

## 🚀 **Quick Installation (CPU-Only)**

The fastest way to get PhysGrad running with visualization:

```bash
# Clone and build
git clone <repository-url>
cd physgrad
./install_cpu_version.sh
```

This script:
- ✅ Checks all system dependencies
- ✅ Builds CPU-only version with OpenGL visualization
- ✅ Runs verification tests
- ✅ Provides usage instructions

## 🔧 **Manual Build Process**

### CPU-Only Build (Recommended)

```bash
cd python

# Force CPU-only build
export CUDA_HOME=/nonexistent

# Build extension
python3 setup.py build_ext --inplace

# Test installation
PYTHONPATH=. python3 -c "import physgrad as pg; print('Success!')"
```

### GPU Build (Experimental)

⚠️ **Warning:** CUDA compilation currently has issues with symbol definitions.

```bash
cd python

# Ensure CUDA is available
export CUDA_HOME=/usr/local/cuda

# Build with CUDA support
python3 setup.py build_ext --inplace
```

**Known Issues:**
- `CUDA_CHECK` macro undefined
- Missing constant memory symbols
- Requires fixes in `src/variational_contact_gpu.cu`

## 🧪 **Verification Tests**

### Basic Import Test

```python
import physgrad as pg
print("Features:", pg.FEATURES)

# Should show:
# {'cuda': False, 'pytorch': False, 'jax': False,
#  'multi_gpu': True, 'visualization': True,
#  'realtime_visualization': True, 'autodiff': True}
```

### Simulation Test

```python
import physgrad as pg
import numpy as np

# Create simulation
config = pg.SimulationConfig(num_particles=10, dt=0.01)
sim = pg.Simulation(config)

# Add particles
for i in range(5):
    pos = np.array([i, 0, 0], dtype=float)
    particle = pg.Particle(position=pos, velocity=[0,1,0], mass=1.0)
    sim.add_particle(particle)

# Add forces
sim.add_force(pg.GravityForce(gravity=[0, -9.81, 0]))
sim.add_force(pg.DampingForce(damping_coefficient=0.1))

# Run simulation
for _ in range(100):
    sim.step()

print("✅ Simulation completed successfully!")
```

### Visualization Test (Requires Display)

```python
import physgrad as pg

# Quick visualization demo
pg.quick_realtime_demo(num_particles=20, demo_type="bouncing_balls")

# Or manually
config = pg.SimulationConfig(num_particles=10, dt=0.01)
sim = pg.Simulation(config)
# ... add particles and forces ...

# Run with real-time visualization
sim.run_with_visualization(max_steps=1000)
```

## 📁 **Project Structure**

```
physgrad/
├── src/                          # C++ source code
│   ├── visualization.cpp         # OpenGL/ImGui rendering
│   ├── variational_contact.cpp   # Core physics
│   └── ...
├── external/imgui/               # ImGui library
├── python/
│   ├── physgrad/                 # Python package
│   │   ├── core.py              # High-level simulation API
│   │   ├── visualization.py     # Python visualization
│   │   └── ...
│   ├── src/
│   │   └── physgrad_binding_simple.cpp  # pybind11 bindings
│   └── setup.py                 # Build configuration
├── BUILD.md                     # This file
└── install_cpu_version.sh       # Installation script
```

## 🎮 **Usage Examples**

### Basic Physics Simulation

```python
import physgrad as pg
import numpy as np

# Create bouncing balls
config = pg.SimulationConfig(num_particles=20, dt=0.01)
sim = pg.Simulation(config)

# Add random particles
for i in range(20):
    pos = np.random.uniform(-5, 5, 3)
    pos[1] = np.random.uniform(2, 8)  # Start high
    vel = np.random.uniform(-1, 1, 3)
    mass = np.random.uniform(0.5, 2.0)

    particle = pg.Particle(position=pos, velocity=vel, mass=mass)
    sim.add_particle(particle)

# Add physics
sim.add_force(pg.GravityForce(gravity=[0, -9.81, 0]))
sim.add_force(pg.DampingForce(damping_coefficient=0.05))

# Run with real-time visualization
sim.run_with_visualization(max_steps=5000)
```

### Pendulum System

```python
import physgrad as pg

config = pg.SimulationConfig(num_particles=10, dt=0.005, enable_constraints=True)
sim = pg.Simulation(config)

# Create pendulum
anchor = pg.Particle(position=[0, 3, 0], mass=1.0, fixed=True)
bob = pg.Particle(position=[1, 1, 0], mass=1.0)

anchor_id = sim.add_particle(anchor)
bob_id = sim.add_particle(bob)

# Add pendulum constraint
constraint = pg.DistanceConstraint(anchor_id, bob_id, distance=2.0, stiffness=10000.0)
sim.add_constraint(constraint)

# Add forces
sim.add_force(pg.GravityForce(gravity=[0, -9.81, 0]))
sim.add_force(pg.DampingForce(damping_coefficient=0.02))

# Visualize
sim.run_with_visualization(max_steps=3000)
```

## 🔍 **Troubleshooting**

### Common Issues

**1. OpenGL libraries not found**
```bash
sudo apt-get install libgl1-mesa-dev libglfw3-dev libglew-dev
```

**2. Eigen3 headers missing**
```bash
sudo apt-get install libeigen3-dev
```

**3. pybind11 not found**
```bash
pip install pybind11
```

**4. Compilation fails with CUDA errors**
- Use CPU-only build: `export CUDA_HOME=/nonexistent`
- CUDA support requires fixes in GPU kernels

**5. Visualization window doesn't open**
- Ensure you have a display server running
- Check OpenGL support: `glxinfo | grep "direct rendering"`

### Build System Issues

**setup.py warnings about license/dependencies**
- These are warnings from pyproject.toml conflicts, not build errors
- The extension still compiles successfully

**Missing ImGui files**
- ImGui is included in `external/imgui/`
- Build system automatically includes all necessary files

## 🎯 **Current Feature Status**

### ✅ Working Features
- CPU physics simulation
- Real-time OpenGL/ImGui visualization
- Interactive parameter controls
- Mathematical overlays and equations
- Force systems (gravity, springs, damping)
- Constraint systems (distance, position)
- Multiple integrators
- Energy monitoring
- Performance tracking
- Matplotlib/Plotly visualization

### ⚠️ Partial Features
- Multi-GPU support (compiles but untested)
- Complex collision detection (basic version works)

### ❌ Blocked Features
- CUDA GPU acceleration (compilation issues)
- PyTorch integration (requires CUDA fix)
- JAX integration (requires JAX installation)

## 🔄 **Development Build**

For development work:

```bash
cd python

# Build in development mode
CUDA_HOME=/nonexistent python3 setup.py develop --user

# Or build in-place for testing
CUDA_HOME=/nonexistent python3 setup.py build_ext --inplace
```

## 📊 **Performance Notes**

**CPU-Only Performance:**
- ✅ Handles 100+ particles smoothly
- ✅ Real-time visualization at 60 FPS
- ✅ Physics timestep: 0.001-0.01s typical

**Memory Usage:**
- ✅ Efficient C++ core with Python interface
- ✅ Minimal overhead for visualization

**Scalability:**
- ✅ Suitable for educational/research simulations
- ⚠️ Large-scale simulations need GPU acceleration

---

For issues or contributions, please check the build logs and ensure all prerequisites are installed. The CPU-only version is fully functional for research and educational use.