# PhysGrad

A high-performance, differentiable physics simulation library with GPU acceleration.

## Features

- **CUDA-accelerated physics simulation** with multi-GPU support
- **PyTorch and JAX integration** with automatic differentiation
- **Symplectic integrators** for energy conservation
- **Constraint-based physics** (joints, springs, rigid connections)
- **Collision detection and response**
- **Rigid body dynamics** with rotational motion
- **Real-time visualization** with interactive controls

## Quick Start

### C++ Library

```bash
# Build the library
make

# Run tests
./run_tests
```

### Python Package

```bash
# Install Python package
cd python
pip install -e .

# Basic usage
import physgrad as pg

sim = pg.quick_simulation(num_particles=1000)
sim.run(1000)
```

## Project Structure

```
physgrad/
├── src/                    # Core C++/CUDA implementation
├── python/                 # Python bindings and API
├── tests/                  # Unit tests
├── external/               # External dependencies
└── Makefile               # Build system
```

## Requirements

- CUDA 11.0+
- Python 3.8+ (for Python bindings)
- PyTorch 1.12+ (optional)
- JAX 0.3+ (optional)

## Documentation

See `python/README.md` for detailed Python API documentation.

## License

MIT License - see LICENSE file.