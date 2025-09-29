# PhysGrad

A high-performance physics simulation framework with GPU acceleration and differentiable computing capabilities.

## Features

- **Multi-Physics Simulation**: Particle dynamics, contact mechanics, and fluid dynamics
- **GPU Acceleration**: CUDA-optimized kernels for high-performance computing
- **Cross-Platform**: Compatible with Linux, Windows, and macOS
- **Python Integration**: Complete Python bindings for easy integration
- **Energy Conservation**: Physically accurate simulations with proper energy conservation
- **Boundary Conditions**: Support for periodic, reflective, and open boundary conditions

## Installation

### Prerequisites

- CMake 3.18+
- C++17 compatible compiler
- CUDA Toolkit 11.0+ (optional, for GPU acceleration)
- Python 3.8+ (optional, for Python bindings)

### Building

```bash
git clone https://github.com/pedroscortes/physgrad.git
cd physgrad
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build Options

- `-DWITH_CUDA=ON/OFF` - Enable/disable CUDA support (default: ON)
- `-DWITH_VISUALIZATION=ON/OFF` - Enable/disable visualization (default: ON)
- `-DWITH_PYTHON=ON/OFF` - Enable/disable Python bindings (default: ON)

## Usage

### C++ API

```cpp
#include "physics_engine.h"

physgrad::PhysicsEngine engine;
engine.initialize();

// Add particles
std::vector<float3> positions = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
std::vector<float3> velocities = {{1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}};
std::vector<float> masses = {1.0f, 1.0f};

engine.addParticles(positions, velocities, masses);

// Run simulation
for (int i = 0; i < 1000; ++i) {
    engine.step(0.01f);
}
```

### Python API

```python
import physgrad

# Create physics engine
engine = physgrad.PhysicsEngine()
engine.initialize()

# Add particles and run simulation
positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
velocities = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
masses = [1.0, 1.0]

engine.add_particles(positions, velocities, masses)

for i in range(1000):
    engine.step(0.01)
```

## Performance

- **GPU Memory Bandwidth**: 200+ GB/s on modern hardware
- **Particle Throughput**: 1M+ particles at 60 FPS
- **CPU Fallback**: Optimized CPU implementation for systems without CUDA

## Testing

Run the test suite to verify installation:

```bash
make test
# or run individual test suites
./tests/test_physics_engine
./tests/test_cuda_kernels
./tests/test_contact_mechanics
./tests/test_fluid_dynamics
```

## License

MIT License. See LICENSE file for details.

## Repository

https://github.com/pedroscortes/physgrad