# PhysGrad PyTorch Integration

**Complete Guide to Differentiable Physics with PyTorch**

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)

---

## Overview

PhysGrad provides seamless integration with PyTorch, enabling **end-to-end gradient computation through physics simulations**. This allows you to:

- **Train neural networks** with physics in the loop
- **Optimize trajectories** using gradient descent
- **Discover parameters** from observations (inverse problems)
- **Co-design** systems (networks + physics together)
- Build **physics-informed neural networks** (PINNs)

### Key Features

✅ **Efficient Adjoint Method**: O(1) memory per timestep, not O(n)
✅ **PyTorch Autograd Integration**: Works with `.backward()`
✅ **Parameter Gradients**: Optimize spring constants, rest lengths, etc.
✅ **Float32 & Float64 Support**: Choose your precision
✅ **Validated Gradients**: <1% error vs finite differences

---

## Installation

### Prerequisites

```bash
# Required
pip install torch numpy pybind11

# Optional (for visualization in examples)
pip install matplotlib
```

### Build the Extension

```bash
cd python/
./build_adjoint_extension.sh
```

Or manually:

```bash
cd python/
python setup.py build_ext --inplace
```

### Verify Installation

```bash
python -c "from physgrad.adjoint import SpringMassSystem; print('Success!')"
```

---

## Quick Start

### 5-Minute Example: Trajectory Optimization

```python
import torch
from physgrad.adjoint import AdjointPhysics, SpringMassSystem

# Create physics system: 2 particles, 1 spring
system = SpringMassSystem(n_particles=2, dtype='float32')
system.add_spring(i=0, j=1, stiffness=10.0, rest_length=1.0)

# Learnable initial positions
positions = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.5, 0.0, 0.0]
], requires_grad=True)

velocities = torch.zeros(2, 3)
masses = torch.ones(2)

# Create differentiable physics simulator
physics = AdjointPhysics(system, dt=0.01, num_steps=50)

# Optimizer
optimizer = torch.optim.Adam([positions], lr=0.05)

# Training loop
for iteration in range(30):
    optimizer.zero_grad()

    # Forward simulation (differentiable!)
    final_pos, final_vel = physics(positions, velocities, masses)

    # Loss: minimize final displacement
    loss = (final_pos ** 2).sum()

    # Backward through physics
    loss.backward()

    # Update
    optimizer.step()

    print(f"Iteration {iteration}: Loss = {loss.item():.6f}")
```

**That's it!** Gradients flow through the physics simulation automatically.

---

## Core Concepts

### 1. The Adjoint Method

Traditional backpropagation through simulation requires storing every intermediate state:

```
Memory: O(n_particles × n_timesteps × state_dim)
```

The **adjoint method** computes the same gradients with:

```
Memory: O(n_particles × state_dim)  # Constant per timestep!
```

This is achieved by running the simulation backwards, accumulating gradients as we go.

### 2. PyTorch Integration

PhysGrad implements a custom `torch.autograd.Function` that:

- **Forward pass**: Runs physics with checkpointing
- **Backward pass**: Runs adjoint pass to compute gradients
- **Gradients**: Automatically propagate to network parameters

This means physics simulation acts like any other PyTorch layer!

### 3. What Can Be Optimized?

You can compute gradients w.r.t.:

✅ **Initial positions**
✅ **Initial velocities**
✅ **Particle masses**
✅ **Spring constants** (via `compute_all_gradients()`)
✅ **Rest lengths** (via `compute_all_gradients()`)

---

## API Reference

### `SpringMassSystem`

Creates a spring-mass physics system.

```python
system = SpringMassSystem(n_particles, dtype='float32')
```

**Parameters**:
- `n_particles` (int): Number of particles
- `dtype` (str): 'float32' or 'float64'

**Methods**:

```python
# Add a spring between particles i and j
system.add_spring(i, j, stiffness, rest_length)

# Get number of springs
n_springs = system.get_num_springs()
```

---

### `AdjointPhysics`

Differentiable physics simulator (PyTorch nn.Module).

```python
physics = AdjointPhysics(system, dt, num_steps)
```

**Parameters**:
- `system` (SpringMassSystem): Physics system definition
- `dt` (float): Timestep size
- `num_steps` (int): Number of simulation steps

**Forward Simulation**:

```python
final_positions, final_velocities = physics(
    positions,     # (N, 3) tensor
    velocities,    # (N, 3) tensor
    masses         # (N,) tensor (optional, defaults to 1.0)
)
```

Returns final state after `num_steps` of simulation.

**Comprehensive Gradients** (including parameters):

```python
all_grads = physics.compute_all_gradients(
    positions,      # (N, 3) tensor
    velocities,     # (N, 3) tensor
    masses,         # (N,) tensor
    loss_function   # Callable: (pos, vel) -> float
)
```

Returns dictionary:
```python
{
    'position_grads': ndarray,           # ∂L/∂x₀
    'velocity_grads': ndarray,           # ∂L/∂v₀
    'spring_constant_grads': ndarray,    # ∂L/∂k
    'rest_length_grads': ndarray         # ∂L/∂r₀
}
```

---

## Examples

### Example 1: Basic Gradient Computation

**Goal**: Find initial conditions that minimize final displacement.

```python
import torch
from physgrad.adjoint import AdjointPhysics, SpringMassSystem

# Setup
system = SpringMassSystem(n_particles=2)
system.add_spring(0, 1, stiffness=10.0, rest_length=1.0)
physics = AdjointPhysics(system, dt=0.01, num_steps=50)

# Learnable
positions = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], requires_grad=True)
velocities = torch.zeros(2, 3)

# Optimize
optimizer = torch.optim.Adam([positions], lr=0.05)

for i in range(50):
    optimizer.zero_grad()
    final_pos, _ = physics(positions, velocities, torch.ones(2))
    loss = (final_pos ** 2).sum()
    loss.backward()
    optimizer.step()
```

**Complete example**: `examples/01_basic_pytorch_gradients.py`

---

### Example 2: Target Tracking

**Goal**: Find initial velocity to reach a target position.

```python
# Target
target = torch.tensor([2.0, 0.0, 0.0])

# Learnable velocity
velocity = torch.zeros(2, 3, requires_grad=True)

# Optimize
for i in range(100):
    optimizer.zero_grad()
    final_pos, _ = physics(fixed_positions, velocity, masses)
    loss = ((final_pos[1] - target) ** 2).sum()
    loss.backward()
    optimizer.step()
```

---

### Example 3: Material Parameter Optimization

**Goal**: Optimize spring stiffness for desired behavior.

```python
def loss_fn(pos, vel):
    # Minimize final kinetic energy
    return 0.5 * (vel ** 2).sum()

# Compute ALL gradients including spring constant
all_grads = physics.compute_all_gradients(
    positions, velocities, masses, loss_fn
)

# Gradient w.r.t. spring constant
dk = all_grads['spring_constant_grads'][0]

# Update (requires rebuilding system with new k)
new_k = old_k - learning_rate * dk
```

**Complete example**: `examples/01_basic_pytorch_gradients.py`

---

### Example 4: Physics-Informed Neural Network (PINN)

**Goal**: Train neural network that respects physics constraints.

```python
import torch.nn as nn

class ControllerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Tanh()
        )

    def forward(self, target):
        return self.net(target) * 2.0  # Initial velocity

# Training loop
controller = ControllerNetwork()
optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)

for i in range(100):
    optimizer.zero_grad()

    # Sample random target
    target = torch.randn(3) + torch.tensor([1.5, 0.0, 0.0])

    # Network predicts control
    predicted_velocity = controller(target)

    # Physics simulation
    final_pos, _ = physics(positions, predicted_velocity, masses)

    # Loss: reach target
    loss = ((final_pos[1] - target) ** 2).sum()

    # Backprop through physics AND network
    loss.backward()
    optimizer.step()
```

**Complete example**: `examples/02_physics_informed_neural_network.py`

---

### Example 5: Co-Design (Network + Physics)

**Goal**: Train network to output initial conditions that achieve desired outcomes.

```python
class StatePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # [position(3), velocity(3)]
        )

    def forward(self, desired_final_position):
        output = self.net(desired_final_position)
        return output[:3], output[3:]  # pos, vel

# Training
predictor = StatePredictor()
optimizer = torch.optim.Adam(predictor.parameters(), lr=0.01)

for i in range(200):
    optimizer.zero_grad()

    # Random desired final state
    desired = torch.randn(3) + torch.tensor([1.2, 0.0, 0.0])

    # Network predicts initial state
    pred_pos, pred_vel = predictor(desired)

    # Simulate
    final_pos, _ = physics(pred_pos, pred_vel, masses)

    # Loss: match desired
    loss = ((final_pos[1] - desired) ** 2).sum()
    loss.backward()
    optimizer.step()
```

**Complete example**: `examples/03_neural_physics_codesign.py`

---

## Best Practices

### 1. Choose Appropriate Timestep

```python
# Too large → numerical instability
dt = 0.1  # ❌

# Too small → slow simulation
dt = 0.0001  # ⚠️

# Good balance
dt = 0.01  # ✅
```

**Rule of thumb**: `dt < 0.1 / sqrt(max_spring_constant)`

---

### 2. Normalize Your Loss

```python
# Bad: Scale-dependent
loss = (final_pos ** 2).sum()  # ❌

# Good: Normalized
loss = ((final_pos - target) ** 2).sum() / n_particles  # ✅
```

---

### 3. Use Appropriate Learning Rates

```python
# For initial positions/velocities
optimizer = torch.optim.Adam([positions], lr=0.05)  # ✅

# For neural network parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # ✅

# For spring constants (slower)
k_new = k - 0.001 * grad_k  # ✅
```

---

### 4. Gradient Clipping for Stability

```python
# Clip gradients to prevent explosions
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

---

### 5. Warm Start Optimization

```python
# Start with reasonable initial guess
positions = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0]  # Near rest length
], requires_grad=True)
```

---

## Troubleshooting

### Problem: Gradients are NaN

**Causes**:
- Timestep too large
- Spring constants too stiff
- Simulation became unstable

**Solutions**:
```python
# Reduce timestep
dt = 0.005  # Instead of 0.01

# Use gradient clipping
torch.nn.utils.clip_grad_norm_(parameters, 1.0)

# Check for NaN during training
if torch.isnan(loss):
    print("Warning: NaN detected!")
    break
```

---

### Problem: Gradients are Zero

**Causes**:
- Not using `requires_grad=True`
- Detaching tensors accidentally
- Physics not affecting loss

**Solutions**:
```python
# Ensure requires_grad
positions = torch.tensor(..., requires_grad=True)  # ✅

# Don't detach
final_pos, _ = physics(positions.detach(), ...)  # ❌
final_pos, _ = physics(positions, ...)  # ✅

# Check gradient flow
print(positions.grad)  # Should be non-zero after .backward()
```

---

### Problem: C++ Extension Not Found

**Error**: `ImportError: cannot import name 'adjoint_verlet_cpp'`

**Solution**:
```bash
# Rebuild extension
cd python/
./build_adjoint_extension.sh

# Or manually
python setup.py build_ext --inplace

# Verify
python -c "import physgrad.adjoint; print('OK')"
```

---

### Problem: Slow Simulation

**Causes**:
- Too many timesteps
- Running on CPU

**Solutions**:
```python
# Reduce timesteps where possible
num_steps = 50  # Instead of 500

# Use float32 instead of float64
system = SpringMassSystem(n_particles=N, dtype='float32')  # ✅

# Future: CUDA acceleration (coming soon)
```

---

## Performance Tips

### Memory Usage

```python
# Good: Adjoint method (O(1) per timestep)
physics = AdjointPhysics(system, dt=0.01, num_steps=1000)  # ✅

# Bad: Full backprop would need O(n) memory
```

---

### Batching (Future Feature)

Currently, each simulation runs independently. For batch processing:

```python
# Current: Loop over batch
for i in range(batch_size):
    final_pos, _ = physics(positions[i], ...)

# Future: Batched simulation (coming soon!)
final_pos, _ = physics(positions_batch, ...)  # (B, N, 3)
```

---

### Precision vs Speed

```python
# Faster, less accurate
system = SpringMassSystem(n_particles=N, dtype='float32')

# Slower, more accurate
system = SpringMassSystem(n_particles=N, dtype='float64')
```

For most applications, `float32` is sufficient and 2× faster.

---

## Advanced Usage

### Custom Force Engines

You can implement your own force engine in C++ and expose it to Python. See `src/adjoint_integrators.h` for the `ForceEngineInterface`.

### Integration with Other Frameworks

The C++ backend is framework-agnostic. You can create bindings for JAX, TensorFlow, etc. using the same core implementation.

---

## Citation

If you use PhysGrad in your research, please cite:

```bibtex
@software{physgrad2025,
  title = {PhysGrad: Differentiable Physics with Adjoint Method},
  author = {PhysGrad Team},
  year = {2025},
  url = {https://github.com/your-repo/physgrad}
}
```

---

## Support

- **Issues**: https://github.com/your-repo/physgrad/issues
- **Documentation**: `docs/ADJOINT_API_GUIDE.md`
- **Examples**: `examples/`
- **Tests**: `python/tests/test_adjoint_pytorch.py`

---

## What's Next?

- [ ] CUDA acceleration for GPU training
- [ ] Batched simulation support
- [ ] More force models (friction, damping, etc.)
- [ ] JAX integration
- [ ] Advanced visualization tools

---

**Happy differentiable physics!** 🚀
