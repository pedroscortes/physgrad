# PhysGrad

A high-performance, differentiable physics simulation library with GPU acceleration for computational physics research and development.

## Features

- **GPU-Accelerated Physics**: CUDA kernels for contact mechanics, fluid dynamics, and electromagnetic fields
- **Cross-Platform Compatibility**: Unified C++/CUDA type system works across compilation contexts
- **Memory Management**: Optimized GPU memory allocation and bandwidth utilization
- **Real-Time Visualization**: OpenGL/ImGui interface for interactive simulation monitoring
- **Python Integration**: pybind11 bindings for scientific computing workflows
- **Modular Architecture**: Extensible framework for advanced physics research

## Quick Start

### Prerequisites

**System Dependencies (Ubuntu/Debian):**
```bash
sudo apt-get install libgl1-mesa-dev libglfw3-dev libglew-dev libeigen3-dev
```

**CUDA Requirements:**
- CUDA Toolkit 11.0+
- Compatible NVIDIA GPU
- Driver version 450.36.06+

### Build Instructions

```bash
# Clone and build
git clone <repository-url>
cd physgrad
mkdir build && cd build
cmake .. -DWITH_CUDA=ON -DWITH_VISUALIZATION=ON -DWITH_PYTHON=ON
make -j$(nproc)

# Run tests
make test

# Run demo
./demo_contact_mechanics
```

### Python Usage

```bash
# Install Python package
cd python && pip install -e .

# Basic simulation
import physgrad as pg
engine = pg.PhysicsEngine()
engine.initialize()
engine.add_particles(positions, velocities, masses)
engine.step(dt=0.001)
```

## Project Structure

```
physgrad/
├── src/                    # Core C++/CUDA implementation
│   ├── physics_engine.*    # Main physics simulation engine
│   ├── physics_kernels.cu  # GPU acceleration kernels
│   ├── memory_manager.*    # GPU memory management
│   └── common_types.h      # Cross-platform type definitions
├── tests/                  # Unit testing framework
├── python/                 # Python bindings and API
├── external/imgui/         # Third-party visualization
└── CMakeLists.txt         # Modern CMake build system
```

## Performance

- **1000 particles**: 4ms simulation step
- **Memory bandwidth**: 22 GB/s (optimization ongoing)
- **Test coverage**: Core systems verified (memory management: 13/13 tests passing)

## Development Status

**Working Components:**
- Core physics engine framework
- GPU memory management system
- Cross-platform compilation
- Real-time visualization
- Python bindings

**Current Focus:**
- CUDA kernel computational logic optimization
- Physics algorithm accuracy improvements
- Performance optimization

See `TECH_DEBT.md` for detailed development roadmap.

## Configuration Options

```bash
# Build options
cmake .. -DWITH_CUDA=ON          # Enable GPU acceleration
         -DWITH_VISUALIZATION=ON  # Enable OpenGL/ImGui
         -DWITH_PYTHON=ON         # Build Python bindings
         -DBUILD_TESTS=ON         # Build test suite
         -DBUILD_DEMOS=ON         # Build demonstration programs
```

## Documentation

- **Tech Debt**: `TECH_DEBT.md` - Development roadmap and known issues
- **Strategy**: `STRATEGY.md` - Implementation strategy for core fixes
- **Python API**: `python/src/` - Detailed Python binding documentation

## Contributing

1. Focus on fixing critical CUDA kernel issues first
2. Ensure all tests pass before adding features
3. Follow existing code patterns and type definitions
4. Update relevant documentation

## License

MIT License - See LICENSE file for details.

---
*PhysGrad: Advancing computational physics through GPU-accelerated simulation*