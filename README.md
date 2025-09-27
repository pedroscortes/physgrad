# PhysGrad

GPU-accelerated N-body gravitational simulation with differentiable programming capabilities.

## Features

- Massively parallel CUDA implementation
- Real-time simulation of up to 16K+ particles
- 1.9+ TFLOPS sustained performance on RTX GPUs
- Energy-conserving numerical integration
- OpenGL visualization with CUDA interoperability
- Extensible architecture for differentiable physics

## Requirements

- CUDA-capable GPU (Compute Capability 7.5+)
- NVIDIA CUDA Toolkit 12.1+
- C++17 compatible compiler
- GLFW (for visualization)
- OpenGL 3.3+

## Building

```bash
make console    # Console-only version (no OpenGL)
make physgrad   # Full version with visualization
```

## Usage

```bash
./physgrad_console 4096    # Run simulation with 4096 particles
./physgrad                 # Interactive visualization
```

## Performance

| Particles | Time/Step | GFLOPS |
|-----------|-----------|---------|
| 1,024     | 0.09ms    | 247     |
| 4,096     | 0.33ms    | 1,012   |
| 16,384    | 3.28ms    | 1,930   |

## Architecture

- `src/simulation.h` - Core data structures and interfaces
- `src/simulation.cu` - CUDA kernels and physics implementation
- `src/main.cpp` - OpenGL visualization frontend
- `src/console.cpp` - Console testing and benchmarking
- `tests/` - Validation and performance tests

## License

MIT