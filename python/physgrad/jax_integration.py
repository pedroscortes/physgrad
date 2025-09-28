"""
JAX integration for PhysGrad with automatic differentiation and XLA compilation.

This module provides JAX-compatible functions that enable differentiable
physics simulations with just-in-time (JIT) compilation.
"""

import warnings
from typing import Optional, Tuple, Dict, Any
import numpy as np
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_vjp, jit, grad
    from jax import lax
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False
    warnings.warn("JAX not available. Install with: pip install jax jaxlib")

try:
    try:
        from . import physgrad_cpp
    except ImportError:
        import physgrad_cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    warnings.warn("C++ backend not available. JAX integration requires compiled bindings.")


if _JAX_AVAILABLE and _CPP_AVAILABLE:

    def _compute_contact_forces_host(positions, velocities, masses, radii, material_ids,
                                   barrier_stiffness=1e6, barrier_threshold=1e-4,
                                   friction_regularization=1e-6):
        """Host callback for contact force computation."""
        # Create solver
        params = physgrad_cpp.VariationalContactParams()
        params.barrier_stiffness = float(barrier_stiffness)
        params.barrier_threshold = float(barrier_threshold)
        params.friction_regularization = float(friction_regularization)

        solver = physgrad_cpp.VariationalContactSolver(params)

        # Compute forces
        forces = solver.compute_contact_forces(
            positions, velocities, masses, radii, material_ids
        )
        return forces

    @custom_vjp
    def variational_contact_forces_jax(positions, velocities, masses, radii, material_ids,
                                     barrier_stiffness=1e6, barrier_threshold=1e-4,
                                     friction_regularization=1e-6):
        """
        JAX-compatible contact force computation with automatic differentiation.

        Args:
            positions: Body positions, shape (N, 3)
            velocities: Body velocities, shape (N, 3)
            masses: Body masses, shape (N,)
            radii: Body radii, shape (N,)
            material_ids: Material type IDs, shape (N,)
            barrier_stiffness: Barrier function stiffness
            barrier_threshold: Distance threshold for barrier activation
            friction_regularization: Friction regularization parameter

        Returns:
            Contact forces, shape (N, 3)
        """
        # Use JAX host callback for CPU computation
        def host_callback(arrays):
            pos, vel, m, r, mat = arrays
            return _compute_contact_forces_host(
                pos, vel, m, r, mat, barrier_stiffness, barrier_threshold, friction_regularization
            )

        # Convert to numpy for host callback
        pos_np = np.asarray(positions, dtype=np.float64)
        vel_np = np.asarray(velocities, dtype=np.float64)
        masses_np = np.asarray(masses, dtype=np.float64)
        radii_np = np.asarray(radii, dtype=np.float64)
        materials_np = np.asarray(material_ids, dtype=np.int32)

        # Call host function
        forces_np = host_callback((pos_np, vel_np, masses_np, radii_np, materials_np))

        return jnp.array(forces_np)

    def _contact_forces_fwd(positions, velocities, masses, radii, material_ids,
                           barrier_stiffness, barrier_threshold, friction_regularization):
        """Forward pass for custom VJP."""
        forces = variational_contact_forces_jax(
            positions, velocities, masses, radii, material_ids,
            barrier_stiffness, barrier_threshold, friction_regularization
        )
        # Save inputs for backward pass
        residuals = (positions, velocities, masses, radii, material_ids,
                    barrier_stiffness, barrier_threshold, friction_regularization)
        return forces, residuals

    def _contact_forces_bwd(residuals, force_cotangents):
        """Backward pass using finite differences."""
        (positions, velocities, masses, radii, material_ids,
         barrier_stiffness, barrier_threshold, friction_regularization) = residuals

        # Convert to numpy for gradient computation
        pos_np = np.asarray(positions, dtype=np.float64)
        vel_np = np.asarray(velocities, dtype=np.float64)
        masses_np = np.asarray(masses, dtype=np.float64)
        radii_np = np.asarray(radii, dtype=np.float64)
        materials_np = np.asarray(material_ids, dtype=np.int32)
        cotang_np = np.asarray(force_cotangents, dtype=np.float64)

        # Compute gradients using finite differences
        epsilon = 1e-7
        n_bodies = pos_np.shape[0]
        grad_positions = np.zeros_like(pos_np)

        for i in range(n_bodies):
            for j in range(3):
                # Positive perturbation
                pos_plus = pos_np.copy()
                pos_plus[i, j] += epsilon
                forces_plus = _compute_contact_forces_host(
                    pos_plus, vel_np, masses_np, radii_np, materials_np,
                    barrier_stiffness, barrier_threshold, friction_regularization
                )

                # Negative perturbation
                pos_minus = pos_np.copy()
                pos_minus[i, j] -= epsilon
                forces_minus = _compute_contact_forces_host(
                    pos_minus, vel_np, masses_np, radii_np, materials_np,
                    barrier_stiffness, barrier_threshold, friction_regularization
                )

                # Finite difference jacobian
                jacobian = (forces_plus - forces_minus) / (2 * epsilon)

                # Chain rule
                grad_positions[i, j] = np.sum(cotang_np * jacobian)

        # Return gradients (None for non-differentiable inputs)
        return (
            jnp.array(grad_positions),  # positions
            None,  # velocities
            None,  # masses
            None,  # radii
            None,  # material_ids
            None,  # barrier_stiffness
            None,  # barrier_threshold
            None   # friction_regularization
        )

    # Set up custom VJP
    variational_contact_forces_jax.defvjp(_contact_forces_fwd, _contact_forces_bwd)

    class JAXPhysicsIntegrator:
        """
        JAX-compatible physics integrator with JIT compilation.
        """

        def __init__(self, dt=0.01, gravity=(0.0, -9.81, 0.0)):
            """
            Initialize integrator.

            Args:
                dt: Time step size
                gravity: Gravity vector
            """
            self.dt = dt
            self.gravity = jnp.array(gravity)

        def step(self, positions, velocities, masses, radii, material_ids, external_forces=None):
            """
            Perform one physics simulation step.

            Args:
                positions: Current positions, shape (N, 3)
                velocities: Current velocities, shape (N, 3)
                masses: Body masses, shape (N,)
                radii: Body radii, shape (N,)
                material_ids: Material types, shape (N,)
                external_forces: Additional forces, shape (N, 3)

            Returns:
                Tuple of (new_positions, new_velocities)
            """
            # Initialize total forces
            total_forces = jnp.zeros_like(positions)

            # Add gravity
            total_forces += self.gravity[None, :] * masses[:, None]

            # Add external forces
            if external_forces is not None:
                total_forces += external_forces

            # Add contact forces using custom VJP function (differentiable)
            contact_forces = variational_contact_forces_jax(
                positions, velocities, masses, radii, material_ids
            )
            total_forces += contact_forces

            # Compute accelerations
            accelerations = total_forces / masses[:, None]

            # Symplectic Euler integration
            new_velocities = velocities + accelerations * self.dt
            new_positions = positions + new_velocities * self.dt

            return new_positions, new_velocities

    class JAXSimulation:
        """
        High-level JAX simulation class.
        """

        def __init__(self, n_bodies, dt=0.01):
            """
            Initialize simulation.

            Args:
                n_bodies: Number of bodies
                dt: Time step size
            """
            self.n_bodies = n_bodies
            self.dt = dt
            self.integrator = JAXPhysicsIntegrator(dt=dt)

            # Initialize state
            self.positions = jnp.zeros((n_bodies, 3))
            self.velocities = jnp.zeros((n_bodies, 3))
            self.masses = jnp.ones(n_bodies)
            self.radii = jnp.ones(n_bodies) * 0.5
            self.material_ids = jnp.zeros(n_bodies, dtype=jnp.int32)

        def set_state(self, positions=None, velocities=None, masses=None, radii=None):
            """Set simulation state."""
            if positions is not None:
                self.positions = jnp.array(positions)
            if velocities is not None:
                self.velocities = jnp.array(velocities)
            if masses is not None:
                self.masses = jnp.array(masses)
            if radii is not None:
                self.radii = jnp.array(radii)

        def simulate(self, n_steps, return_trajectory=True):
            """
            Run simulation for multiple steps.

            Args:
                n_steps: Number of steps
                return_trajectory: Whether to save trajectory

            Returns:
                Dictionary with simulation results
            """
            positions = self.positions
            velocities = self.velocities

            if return_trajectory:
                trajectory_positions = []
                trajectory_velocities = []

                for _ in range(n_steps):
                    positions, velocities = self.integrator.step(
                        positions, velocities, self.masses, self.radii, self.material_ids
                    )
                    trajectory_positions.append(positions)
                    trajectory_velocities.append(velocities)

                return {
                    'final_positions': positions,
                    'final_velocities': velocities,
                    'trajectory_positions': jnp.stack(trajectory_positions),
                    'trajectory_velocities': jnp.stack(trajectory_velocities)
                }
            else:
                for _ in range(n_steps):
                    positions, velocities = self.integrator.step(
                        positions, velocities, self.masses, self.radii, self.material_ids
                    )

                return {
                    'final_positions': positions,
                    'final_velocities': velocities
                }

    def simulate_and_loss(initial_positions, initial_velocities, target_positions,
                         masses, radii, material_ids, n_steps=10):
        """
        Simulate physics and compute loss for optimization.

        Args:
            initial_positions: Starting positions
            initial_velocities: Starting velocities
            target_positions: Target final positions
            masses: Body masses
            radii: Body radii
            material_ids: Material types
            n_steps: Number of simulation steps

        Returns:
            Loss value
        """
        integrator = JAXPhysicsIntegrator()
        positions = initial_positions
        velocities = initial_velocities

        # Run simulation
        for _ in range(n_steps):
            positions, velocities = integrator.step(
                positions, velocities, masses, radii, material_ids
            )

        # Compute loss as distance to target
        loss = jnp.sum((positions - target_positions) ** 2)
        return loss

    # Utility functions
    def array_to_physgrad(array):
        """Convert JAX array to PhysGrad-compatible numpy array."""
        return np.asarray(array, dtype=np.float64)

    def physgrad_to_array(array):
        """Convert numpy array to JAX array."""
        return jnp.array(array)

    def register_jax_ops():
        """Register JAX operations for PhysGrad.

        This function registers custom JAX operations and primitives
        for use in PhysGrad simulations.
        """
        # This is a placeholder for registering custom JAX operations
        # In a full implementation, this would register custom primitives
        # with JAX for specialized physics operations
        pass

else:
    # Define None for all exports when dependencies aren't available
    variational_contact_forces_jax = None
    JAXPhysicsIntegrator = None
    JAXSimulation = None
    simulate_and_loss = None
    array_to_physgrad = None
    physgrad_to_array = None
    register_jax_ops = None

    if not _JAX_AVAILABLE:
        warnings.warn("JAX not available. Install with: pip install jax jaxlib")
    if not _CPP_AVAILABLE:
        warnings.warn("PhysGrad C++ backend not available. Build with: python build_bindings.py")