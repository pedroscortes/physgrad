"""
PhysGrad Adjoint Physics Integration

This module provides a clean Python API for differentiable physics simulation
using the adjoint method. Integrates seamlessly with PyTorch for end-to-end
gradient-based optimization.

Features:
- Automatic gradient computation through physics simulation
- PyTorch autograd integration
- Support for both state and parameter gradients
- Efficient adjoint method (O(1) memory per timestep)

Example:
    >>> import torch
    >>> from physgrad.adjoint import AdjointPhysics, SpringMassSystem
    >>>
    >>> # Create physics system
    >>> system = SpringMassSystem(n_particles=2)
    >>> system.add_spring(0, 1, stiffness=10.0, rest_length=1.0)
    >>>
    >>> # Learnable initial conditions
    >>> positions = torch.tensor([[0.0, 0.0, 0.0],
    ...                           [1.5, 0.0, 0.0]], requires_grad=True)
    >>> velocities = torch.zeros(2, 3)
    >>>
    >>> # Simulate
    >>> physics = AdjointPhysics(system, dt=0.01, num_steps=100)
    >>> final_pos, final_vel = physics(positions, velocities)
    >>>
    >>> # Compute loss and backpropagate
    >>> loss = (final_pos ** 2).sum()
    >>> loss.backward()
    >>>
    >>> print(positions.grad)  # Gradients w.r.t. initial positions!
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict
import warnings

try:
    import torch
    import torch.nn as nn
    from torch.autograd import Function
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

try:
    # Import C++ extension (built from adjoint_verlet_bindings.cpp)
    from . import adjoint_verlet_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    warnings.warn(
        "C++ adjoint extension not available. "
        "Build with: python setup.py build_ext --inplace"
    )


class SpringMassSystem:
    """
    Spring-mass physics system for adjoint simulation.

    This is a simple force engine that computes spring forces between particles.
    Supports both forward simulation and backward gradient computation.

    Attributes:
        n_particles (int): Number of particles in the system
        dtype (str): Precision ('float32' or 'float64')
        springs (list): List of (i, j, k, r0) tuples for each spring
    """

    def __init__(self, n_particles: int, dtype: str = 'float32'):
        """
        Initialize spring-mass system.

        Args:
            n_particles: Number of particles in the system
            dtype: Precision ('float32' or 'float64')
        """
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ extension not available. Please build the extension first.")

        self.n_particles = n_particles
        self.dtype = dtype
        self.springs = []

        # Create C++ force engine
        if dtype == 'float32':
            self._engine = adjoint_verlet_cpp.SimpleForceEngineFloat()
        elif dtype == 'float64':
            self._engine = adjoint_verlet_cpp.SimpleForceEngineDouble()
        else:
            raise ValueError(f"Invalid dtype: {dtype}. Must be 'float32' or 'float64'")

    def add_spring(self, i: int, j: int, stiffness: float, rest_length: float):
        """
        Add a spring between two particles.

        Args:
            i: Index of first particle
            j: Index of second particle
            stiffness: Spring constant (k)
            rest_length: Rest length (r0)
        """
        if i >= self.n_particles or j >= self.n_particles:
            raise ValueError(f"Particle indices must be < {self.n_particles}")

        self.springs.append((i, j, stiffness, rest_length))
        self._engine.add_spring(i, j, stiffness, rest_length)

    def get_num_springs(self) -> int:
        """Get number of springs in the system."""
        return self._engine.get_num_springs()

    def __repr__(self) -> str:
        return (f"SpringMassSystem(n_particles={self.n_particles}, "
                f"n_springs={self.get_num_springs()}, dtype={self.dtype})")


class AdjointPhysicsFunction(Function):
    """
    PyTorch autograd Function for differentiable physics simulation.

    This wraps the C++ adjoint integrator to provide automatic differentiation
    through PyTorch. Gradients are computed using the adjoint method, which is
    much more efficient than backpropagating through the entire simulation.

    Forward pass: Runs physics simulation with checkpointing
    Backward pass: Computes gradients using adjoint method

    Note: This is an internal class. Users should use AdjointPhysics instead.
    """

    @staticmethod
    def forward(ctx, positions, velocities, masses, system, dt, num_steps, loss_fn):
        """
        Forward pass: Run physics simulation.

        Args:
            ctx: PyTorch context for saving tensors
            positions: Initial positions (N, 3) tensor
            velocities: Initial velocities (N, 3) tensor
            masses: Particle masses (N,) tensor
            system: SpringMassSystem instance
            dt: Timestep size
            num_steps: Number of simulation steps
            loss_fn: Optional loss function for custom gradients

        Returns:
            final_positions, final_velocities: Final state tensors
        """
        # Save for backward pass
        ctx.save_for_backward(positions, velocities, masses)
        ctx.system = system
        ctx.dt = dt
        ctx.num_steps = num_steps
        ctx.loss_fn = loss_fn

        # Convert to numpy for C++ backend
        pos_np = positions.detach().cpu().numpy().astype(
            np.float32 if system.dtype == 'float32' else np.float64
        )
        vel_np = velocities.detach().cpu().numpy().astype(
            np.float32 if system.dtype == 'float32' else np.float64
        )
        mass_np = masses.detach().cpu().numpy().astype(
            np.float32 if system.dtype == 'float32' else np.float64
        )

        # Create C++ simulation
        if system.dtype == 'float32':
            sim = adjoint_verlet_cpp.AdjointSimulationFloat(system._engine)
        else:
            sim = adjoint_verlet_cpp.AdjointSimulationDouble(system._engine)

        # Run forward simulation
        final_pos_np, final_vel_np = sim.run_forward(
            pos_np, vel_np, mass_np, dt, num_steps
        )

        # Store simulation for backward pass
        ctx.simulation = sim

        # Convert back to PyTorch tensors
        device = positions.device
        dtype = positions.dtype
        final_positions = torch.from_numpy(final_pos_np).to(device=device, dtype=dtype)
        final_velocities = torch.from_numpy(final_vel_np).to(device=device, dtype=dtype)

        return final_positions, final_velocities

    @staticmethod
    def backward(ctx, grad_positions, grad_velocities):
        """
        Backward pass: Compute gradients using adjoint method.

        Args:
            ctx: PyTorch context with saved tensors
            grad_positions: Gradient of loss w.r.t. final positions
            grad_velocities: Gradient of loss w.r.t. final velocities

        Returns:
            Gradients w.r.t. inputs: (pos_grad, vel_grad, mass_grad, None, None, None, None)
        """
        positions, velocities, masses = ctx.saved_tensors
        system = ctx.system

        # Convert gradients to numpy
        grad_pos_np = grad_positions.detach().cpu().numpy().astype(
            np.float32 if system.dtype == 'float32' else np.float64
        )
        grad_vel_np = grad_velocities.detach().cpu().numpy().astype(
            np.float32 if system.dtype == 'float32' else np.float64
        )
        mass_np = masses.detach().cpu().numpy().astype(
            np.float32 if system.dtype == 'float32' else np.float64
        )

        # Run adjoint backward pass
        init_pos_grad_np, init_vel_grad_np, mass_grad_np = ctx.simulation.run_backward(
            grad_pos_np, grad_vel_np, mass_np
        )

        # Convert back to PyTorch tensors
        device = positions.device
        dtype = positions.dtype
        init_pos_grad = torch.from_numpy(init_pos_grad_np).to(device=device, dtype=dtype)
        init_vel_grad = torch.from_numpy(init_vel_grad_np).to(device=device, dtype=dtype)
        mass_grad = torch.from_numpy(mass_grad_np).to(device=device, dtype=dtype)

        # Return gradients for all inputs
        # Order must match forward() signature
        return init_pos_grad, init_vel_grad, mass_grad, None, None, None, None


class AdjointPhysics(nn.Module):
    """
    High-level interface for differentiable physics simulation with PyTorch.

    This module provides a PyTorch-native interface to the adjoint physics
    simulator. It can be used like any other PyTorch module in your models.

    Features:
    - Automatic gradient computation through simulation
    - Efficient O(1) memory scaling per timestep
    - Support for custom loss functions
    - Compatible with PyTorch optimizers

    Example:
        >>> system = SpringMassSystem(n_particles=3)
        >>> system.add_spring(0, 1, stiffness=10.0, rest_length=1.0)
        >>> system.add_spring(1, 2, stiffness=10.0, rest_length=1.0)
        >>>
        >>> physics = AdjointPhysics(system, dt=0.01, num_steps=100)
        >>>
        >>> # Learnable initial state
        >>> positions = torch.randn(3, 3, requires_grad=True)
        >>> velocities = torch.zeros(3, 3)
        >>> masses = torch.ones(3)
        >>>
        >>> # Forward simulation
        >>> final_pos, final_vel = physics(positions, velocities, masses)
        >>>
        >>> # Compute loss and gradients
        >>> loss = (final_pos ** 2).sum()
        >>> loss.backward()
        >>> print(positions.grad)
    """

    def __init__(self, system: SpringMassSystem, dt: float, num_steps: int):
        """
        Initialize adjoint physics simulator.

        Args:
            system: SpringMassSystem instance defining the physics
            dt: Timestep size for simulation
            num_steps: Number of simulation steps
        """
        super().__init__()

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for AdjointPhysics. Install with: pip install torch")

        if not CPP_AVAILABLE:
            raise RuntimeError("C++ extension not available. Please build the extension first.")

        self.system = system
        self.dt = dt
        self.num_steps = num_steps

        # Register as buffer (not a learnable parameter)
        self.register_buffer('_dt', torch.tensor(dt))
        self.register_buffer('_num_steps', torch.tensor(num_steps))

    def forward(self,
                positions: torch.Tensor,
                velocities: torch.Tensor,
                masses: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run differentiable physics simulation.

        Args:
            positions: Initial positions (N, 3) tensor
            velocities: Initial velocities (N, 3) tensor
            masses: Particle masses (N,) tensor. If None, uses unit masses.

        Returns:
            final_positions: Final positions (N, 3) tensor
            final_velocities: Final velocities (N, 3) tensor

        Note:
            Gradients are automatically computed when you call .backward()
            on a loss computed from the outputs.
        """
        if masses is None:
            masses = torch.ones(self.system.n_particles,
                              dtype=positions.dtype,
                              device=positions.device)

        # Validate inputs
        if positions.shape != (self.system.n_particles, 3):
            raise ValueError(f"Expected positions shape ({self.system.n_particles}, 3), "
                           f"got {positions.shape}")
        if velocities.shape != (self.system.n_particles, 3):
            raise ValueError(f"Expected velocities shape ({self.system.n_particles}, 3), "
                           f"got {velocities.shape}")
        if masses.shape != (self.system.n_particles,):
            raise ValueError(f"Expected masses shape ({self.system.n_particles},), "
                           f"got {masses.shape}")

        # Use custom autograd function
        return AdjointPhysicsFunction.apply(
            positions, velocities, masses,
            self.system, self.dt, self.num_steps, None
        )

    def compute_all_gradients(self,
                             positions: torch.Tensor,
                             velocities: torch.Tensor,
                             masses: torch.Tensor,
                             loss_fn: Callable) -> Dict[str, np.ndarray]:
        """
        Compute comprehensive gradients including parameter gradients.

        This method computes gradients w.r.t. not only initial state (positions,
        velocities) but also force parameters (spring constants, rest lengths).

        Args:
            positions: Initial positions (N, 3) tensor
            velocities: Initial velocities (N, 3) tensor
            masses: Particle masses (N,) tensor
            loss_fn: Loss function taking (positions, velocities) and returning scalar

        Returns:
            Dictionary with keys:
            - 'position_grads': Gradients w.r.t. initial positions
            - 'velocity_grads': Gradients w.r.t. initial velocities
            - 'spring_constant_grads': Gradients w.r.t. spring constants
            - 'rest_length_grads': Gradients w.r.t. rest lengths

        Example:
            >>> def loss_fn(pos, vel):
            ...     return (pos ** 2).sum()
            >>>
            >>> all_grads = physics.compute_all_gradients(
            ...     positions, velocities, masses, loss_fn
            ... )
            >>>
            >>> # Optimize spring constants
            >>> spring_k -= learning_rate * all_grads['spring_constant_grads']
        """
        # Convert to numpy
        pos_np = positions.detach().cpu().numpy().astype(
            np.float32 if self.system.dtype == 'float32' else np.float64
        )
        vel_np = velocities.detach().cpu().numpy().astype(
            np.float32 if self.system.dtype == 'float32' else np.float64
        )
        mass_np = masses.detach().cpu().numpy().astype(
            np.float32 if self.system.dtype == 'float32' else np.float64
        )

        # Wrap loss function
        def cpp_loss_fn(pos, vel):
            return float(loss_fn(
                torch.from_numpy(pos),
                torch.from_numpy(vel)
            ))

        # Create simulation and compute all gradients
        if self.system.dtype == 'float32':
            sim = adjoint_verlet_cpp.AdjointSimulationFloat(self.system._engine)
        else:
            sim = adjoint_verlet_cpp.AdjointSimulationDouble(self.system._engine)

        all_grads = sim.compute_all_gradients(
            pos_np, vel_np, mass_np, self.dt, self.num_steps, cpp_loss_fn
        )

        return all_grads

    def __repr__(self) -> str:
        return (f"AdjointPhysics(system={self.system}, "
                f"dt={self.dt}, num_steps={self.num_steps})")
