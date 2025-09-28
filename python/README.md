# PhysGrad

A high-performance physics simulation library with automatic differentiation support, designed for scientific computing, robotics, and machine learning applications.

## Features

- **CUDA-accelerated physics simulation** with multi-GPU support
- **PyTorch and JAX integration** with automatic differentiation
- **Symplectic integrators** for energy conservation
- **Constraint-based physics** (joints, springs, rigid connections)
- **Collision detection and response**
- **Rigid body dynamics** with rotational motion
- **Interactive visualization** and real-time parameter tuning

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0 or higher (for GPU acceleration)
- PyTorch 1.12.0 or higher (optional, for PyTorch integration)
- JAX 0.3.0 or higher (optional, for JAX integration)

### Install from pip

```bash
pip install physgrad
```

### Install from source

```bash
git clone https://github.com/physgrad/physgrad.git
cd physgrad/python
pip install -e .
```

## Quick Start

```python
import physgrad as pg
import numpy as np

# Create a simple particle simulation
config = pg.SimulationConfig(
    num_particles=1000,
    dt=0.01,
    enable_gpu=True
)

sim = pg.Simulation(config)

# Add particles
particles = []
for i in range(100):
    pos = np.random.uniform(-5, 5, 3)
    vel = np.random.uniform(-1, 1, 3)
    particle = pg.Particle(position=pos, velocity=vel, mass=1.0)
    particles.append(particle)

particle_ids = sim.add_particles(particles)

# Add gravity
gravity = pg.GravityForce(gravity=[0, -9.81, 0])
sim.add_force(gravity)

# Run simulation
for step in range(1000):
    sim.step()

    if step % 100 == 0:
        energy = sim.get_total_energy()
        print(f"Step {step}: Total energy = {energy:.3f}")
```

## PyTorch Integration

```python
import physgrad as pg
import torch

# Create simulation with PyTorch tensors
positions = torch.randn(100, 3, requires_grad=True, device='cuda')
velocities = torch.randn(100, 3, requires_grad=True, device='cuda')
masses = torch.ones(100, requires_grad=True, device='cuda')
forces = torch.zeros(100, 3, device='cuda')

# Run differentiable physics step
new_pos, new_vel = pg.TorchPhysicsFunction.apply(
    positions, velocities, masses, forces, 0.01
)

# Compute loss and backpropagate
target_positions = torch.randn(100, 3, device='cuda')
loss = torch.nn.functional.mse_loss(new_pos, target_positions)
loss.backward()

print(f"Gradients w.r.t. masses: {masses.grad}")
```

## JAX Integration

```python
import physgrad as pg
import jax.numpy as jnp
from jax import grad, jit

# Register JAX operations
pg.register_jax_ops()

@jit
def physics_loss(params, initial_state, target_trajectory):
    positions, velocities = initial_state
    total_loss = 0.0

    for target_pos in target_trajectory:
        # Simulate one step
        positions, velocities = pg.simulation_step_jax(
            positions, velocities, params['masses'],
            params['forces'], dt=0.01
        )

        # Compute loss
        loss = jnp.sum((positions - target_pos)**2)
        total_loss += loss

    return total_loss

# Compute gradients
grad_fn = grad(physics_loss)
gradients = grad_fn(params, initial_state, target_trajectory)
```

## Multi-GPU Simulation

```python
import physgrad as pg

# Configure multi-GPU setup
config = pg.MultiGPUConfig(
    device_ids=[0, 1, 2, 3],
    partitioning=pg.PartitioningStrategy.SPATIAL_GRID,
    communication=pg.CommunicationPattern.NCCL_COLLECTIVE
)

# Create multi-GPU simulation
sim = pg.MultiGPUSimulation(config)

# Large-scale simulation with 1M particles
sim.setup_large_simulation(num_particles=1_000_000)
sim.run(num_steps=10000)

# Print performance statistics
stats = sim.get_stats()
print(f"Particles per second: {stats.particles_per_second:.0f}")
print(f"Load balance factor: {stats.load_balance_factor:.3f}")
```

## Interactive Visualization

```python
import physgrad as pg

# Create simulation with visualization
config = pg.SimulationConfig(enable_visualization=True)
sim = pg.Simulation(config)

# Setup interactive controls
controls = pg.InteractiveControls()
controls.add_slider("gravity", -20, 0, -9.81)
controls.add_slider("damping", 0, 1, 0.1)

# Add mathematical overlay
overlay = pg.MathematicalOverlay()
overlay.add_energy_plot()
overlay.add_equation_display()

# Run with real-time visualization
visualizer = pg.RealTimeVisualizer(sim, controls, overlay)
visualizer.run()
```

## Documentation

Full documentation is available at [https://physgrad.readthedocs.io](https://physgrad.readthedocs.io)

## Examples

See the `examples/` directory for complete examples including:

- Pendulum simulation
- Cloth dynamics
- Rigid body collisions
- Parameter optimization
- Multi-GPU benchmarks

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PhysGrad in your research, please cite:

```bibtex
@software{physgrad,
  title={PhysGrad: Differentiable Physics Simulation with GPU Acceleration},
  author={PhysGrad Team},
  year={2024},
  url={https://github.com/physgrad/physgrad}
}
```