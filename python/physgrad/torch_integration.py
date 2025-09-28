"""
PyTorch integration for PhysGrad with automatic differentiation support.

This module provides PyTorch autograd functions that enable differentiable
physics simulations with GPU acceleration.
"""

import warnings
from typing import Optional, Tuple, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.autograd import Function
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

try:
    try:
        from . import physgrad_cpp
    except ImportError:
        import physgrad_cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    warnings.warn("C++ backend not available. PyTorch integration requires compiled bindings.")


if _TORCH_AVAILABLE and _CPP_AVAILABLE:

    class VariationalContactFunction(Function):
        """
        PyTorch autograd function for differentiable contact mechanics.

        This function enables automatic differentiation through the variational
        contact solver, allowing physics simulations to be integrated into
        neural network training pipelines.
        """

        @staticmethod
        def forward(ctx, positions, velocities, masses, radii, material_ids, params_dict):
            """
            Forward pass: compute contact forces.

            Args:
                positions: Tensor of shape (N, 3) with body positions
                velocities: Tensor of shape (N, 3) with body velocities
                masses: Tensor of shape (N,) with body masses
                radii: Tensor of shape (N,) with body radii
                material_ids: Tensor of shape (N,) with material type IDs
                params_dict: Dictionary with solver parameters

            Returns:
                forces: Tensor of shape (N, 3) with contact forces
            """
            # Convert tensors to numpy arrays for C++ backend
            pos_np = positions.detach().cpu().numpy().astype(np.float64)
            vel_np = velocities.detach().cpu().numpy().astype(np.float64)
            masses_np = masses.detach().cpu().numpy().astype(np.float64)
            radii_np = radii.detach().cpu().numpy().astype(np.float64)
            materials_np = material_ids.detach().cpu().numpy().astype(np.int32)

            # Create solver with parameters
            cpp_params = physgrad_cpp.VariationalContactParams()
            cpp_params.barrier_stiffness = params_dict.get('barrier_stiffness', 1e6)
            cpp_params.barrier_threshold = params_dict.get('barrier_threshold', 1e-4)
            cpp_params.friction_regularization = params_dict.get('friction_regularization', 1e-6)
            cpp_params.max_newton_iterations = params_dict.get('max_newton_iterations', 50)
            cpp_params.newton_tolerance = params_dict.get('newton_tolerance', 1e-10)

            # Use GPU solver if available and tensors are on CUDA
            use_gpu = positions.is_cuda and physgrad_cpp.cuda_available

            if use_gpu:
                solver = physgrad_cpp.VariationalContactSolverGPU(cpp_params)
            else:
                solver = physgrad_cpp.VariationalContactSolver(cpp_params)

            # Compute forces
            forces_np = solver.compute_contact_forces(
                pos_np, vel_np, masses_np, radii_np, materials_np
            )

            # Convert back to tensor
            forces = torch.from_numpy(forces_np).to(
                dtype=positions.dtype,
                device=positions.device
            )

            # Save for backward pass
            ctx.save_for_backward(positions, velocities, masses, radii, material_ids)
            ctx.solver = solver
            ctx.params_dict = params_dict

            return forces

        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass: compute gradients through contact forces.

            Uses the chain rule with analytically computed contact gradients.
            """
            positions, velocities, masses, radii, material_ids = ctx.saved_tensors
            solver = ctx.solver

            # Convert to numpy
            pos_np = positions.detach().cpu().numpy().astype(np.float64)
            vel_np = velocities.detach().cpu().numpy().astype(np.float64)
            masses_np = masses.detach().cpu().numpy().astype(np.float64)
            radii_np = radii.detach().cpu().numpy().astype(np.float64)
            materials_np = material_ids.detach().cpu().numpy().astype(np.int32)
            grad_out_np = grad_output.detach().cpu().numpy().astype(np.float64)

            # We need to compute gradients with respect to positions
            # For now, we'll use finite differences (TODO: implement analytical gradients)
            epsilon = 1e-7
            n_bodies = pos_np.shape[0]
            grad_positions = np.zeros_like(pos_np)

            for i in range(n_bodies):
                for j in range(3):
                    # Positive perturbation
                    pos_plus = pos_np.copy()
                    pos_plus[i, j] += epsilon
                    forces_plus = solver.compute_contact_forces(
                        pos_plus, vel_np, masses_np, radii_np, materials_np
                    )

                    # Negative perturbation
                    pos_minus = pos_np.copy()
                    pos_minus[i, j] -= epsilon
                    forces_minus = solver.compute_contact_forces(
                        pos_minus, vel_np, masses_np, radii_np, materials_np
                    )

                    # Finite difference gradient
                    jacobian = (forces_plus - forces_minus) / (2 * epsilon)

                    # Chain rule: grad_positions[i,j] = sum_k,l grad_out[k,l] * jacobian[k,l]
                    grad_positions[i, j] = np.sum(grad_out_np * jacobian)

            # Convert gradients back to tensors
            grad_positions_tensor = torch.from_numpy(grad_positions).to(
                dtype=positions.dtype,
                device=positions.device
            )

            # Gradients for other inputs (None for those we don't differentiate)
            return grad_positions_tensor, None, None, None, None, None


    def variational_contact_forces(
        positions: torch.Tensor,
        velocities: torch.Tensor,
        masses: torch.Tensor,
        radii: torch.Tensor,
        material_ids: Optional[torch.Tensor] = None,
        barrier_stiffness: float = 1e6,
        barrier_threshold: float = 1e-4,
        friction_regularization: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute differentiable contact forces using variational mechanics.

        Args:
            positions: Body positions, shape (N, 3)
            velocities: Body velocities, shape (N, 3)
            masses: Body masses, shape (N,)
            radii: Body radii, shape (N,)
            material_ids: Material type IDs, shape (N,). If None, all bodies have material 0.
            barrier_stiffness: Barrier function stiffness parameter
            barrier_threshold: Distance threshold for barrier activation
            friction_regularization: Huber regularization for smooth friction

        Returns:
            Contact forces, shape (N, 3)
        """
        if material_ids is None:
            material_ids = torch.zeros(len(positions), dtype=torch.int32, device=positions.device)

        params_dict = {
            'barrier_stiffness': barrier_stiffness,
            'barrier_threshold': barrier_threshold,
            'friction_regularization': friction_regularization,
            'max_newton_iterations': 50,
            'newton_tolerance': 1e-10
        }

        return VariationalContactFunction.apply(
            positions, velocities, masses, radii, material_ids, params_dict
        )


    class DifferentiableContactLayer(nn.Module):
        """
        PyTorch module for differentiable contact mechanics.

        This layer can be integrated into neural networks to learn physics-aware
        representations or to optimize control policies through contact dynamics.
        """

        def __init__(
            self,
            barrier_stiffness: float = 1e6,
            barrier_threshold: float = 1e-4,
            friction_regularization: float = 1e-6,
            learnable_params: bool = False
        ):
            """
            Initialize the differentiable contact layer.

            Args:
                barrier_stiffness: Initial barrier stiffness
                barrier_threshold: Initial barrier threshold
                friction_regularization: Initial friction regularization
                learnable_params: Whether to make contact parameters learnable
            """
            super().__init__()

            if learnable_params:
                self.log_barrier_stiffness = nn.Parameter(
                    torch.tensor(np.log(barrier_stiffness))
                )
                self.log_barrier_threshold = nn.Parameter(
                    torch.tensor(np.log(barrier_threshold))
                )
                self.log_friction_reg = nn.Parameter(
                    torch.tensor(np.log(friction_regularization))
                )
            else:
                self.register_buffer('log_barrier_stiffness', torch.tensor(np.log(barrier_stiffness)))
                self.register_buffer('log_barrier_threshold', torch.tensor(np.log(barrier_threshold)))
                self.register_buffer('log_friction_reg', torch.tensor(np.log(friction_regularization)))

        def forward(
            self,
            positions: torch.Tensor,
            velocities: torch.Tensor,
            masses: torch.Tensor,
            radii: torch.Tensor,
            material_ids: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Compute contact forces.

            Args:
                positions: Body positions, shape (N, 3)
                velocities: Body velocities, shape (N, 3)
                masses: Body masses, shape (N,)
                radii: Body radii, shape (N,)
                material_ids: Material type IDs, shape (N,)

            Returns:
                Contact forces, shape (N, 3)
            """
            # Use exp to ensure positive parameters
            barrier_stiffness = torch.exp(self.log_barrier_stiffness).item()
            barrier_threshold = torch.exp(self.log_barrier_threshold).item()
            friction_reg = torch.exp(self.log_friction_reg).item()

            return variational_contact_forces(
                positions, velocities, masses, radii, material_ids,
                barrier_stiffness, barrier_threshold, friction_reg
            )


    class PhysicsIntegrator(nn.Module):
        """
        Differentiable physics integrator with contact handling.

        This module performs a physics simulation step that can be differentiated
        through, enabling learning of control policies or system identification.
        """

        def __init__(
            self,
            dt: float = 0.01,
            gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0),
            contact_params: Optional[dict] = None
        ):
            """
            Initialize the physics integrator.

            Args:
                dt: Time step size
                gravity: Gravity vector
                contact_params: Parameters for contact mechanics
            """
            super().__init__()

            self.dt = dt
            self.register_buffer('gravity', torch.tensor(gravity))

            # Initialize contact layer
            if contact_params is None:
                contact_params = {}

            self.contact_layer = DifferentiableContactLayer(
                barrier_stiffness=contact_params.get('barrier_stiffness', 1e6),
                barrier_threshold=contact_params.get('barrier_threshold', 1e-4),
                friction_regularization=contact_params.get('friction_regularization', 1e-6),
                learnable_params=contact_params.get('learnable_params', False)
            )

        def forward(
            self,
            positions: torch.Tensor,
            velocities: torch.Tensor,
            masses: torch.Tensor,
            radii: torch.Tensor,
            material_ids: Optional[torch.Tensor] = None,
            external_forces: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Perform one physics simulation step.

            Args:
                positions: Current positions, shape (N, 3)
                velocities: Current velocities, shape (N, 3)
                masses: Body masses, shape (N,)
                radii: Body radii, shape (N,)
                material_ids: Material types, shape (N,)
                external_forces: Additional external forces, shape (N, 3)

            Returns:
                Tuple of (new_positions, new_velocities)
            """
            N = positions.shape[0]

            # Compute total forces
            total_forces = torch.zeros_like(positions)

            # Add gravity
            total_forces += self.gravity.unsqueeze(0) * masses.unsqueeze(1)

            # Add external forces if provided
            if external_forces is not None:
                total_forces += external_forces

            # Add contact forces
            contact_forces = self.contact_layer(
                positions, velocities, masses, radii, material_ids
            )
            total_forces += contact_forces

            # Compute accelerations
            accelerations = total_forces / masses.unsqueeze(1)

            # Symplectic Euler integration
            new_velocities = velocities + accelerations * self.dt
            new_positions = positions + new_velocities * self.dt

            return new_positions, new_velocities


    def tensor_to_physgrad(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to PhysGrad-compatible numpy array.

        Args:
            tensor: PyTorch tensor

        Returns:
            NumPy array with appropriate dtype for PhysGrad
        """
        return tensor.detach().cpu().numpy().astype(np.float64)


    def physgrad_to_tensor(
        array: np.ndarray,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Convert PhysGrad numpy array to PyTorch tensor.

        Args:
            array: NumPy array from PhysGrad
            dtype: Target tensor dtype
            device: Target device
            requires_grad: Whether to track gradients

        Returns:
            PyTorch tensor
        """
        tensor = torch.from_numpy(array)

        if dtype is not None:
            tensor = tensor.to(dtype)

        if device is not None:
            tensor = tensor.to(device)

        if requires_grad:
            tensor = tensor.requires_grad_(True)

        return tensor


    # Convenience class for full simulations
    class TorchSimulation:
        """
        PyTorch-compatible physics simulation with differentiable contact.
        """

        def __init__(
            self,
            n_bodies: int,
            dt: float = 0.01,
            device: Optional[torch.device] = None,
            contact_params: Optional[dict] = None
        ):
            """
            Initialize a differentiable simulation.

            Args:
                n_bodies: Number of bodies in the simulation
                dt: Time step size
                device: Torch device (cpu/cuda)
                contact_params: Contact mechanics parameters
            """
            self.n_bodies = n_bodies
            self.dt = dt
            self.device = device if device else torch.device('cpu')

            # Initialize integrator
            self.integrator = PhysicsIntegrator(
                dt=dt,
                contact_params=contact_params
            ).to(self.device)

            # Initialize state tensors
            self.positions = torch.zeros(n_bodies, 3, device=self.device)
            self.velocities = torch.zeros(n_bodies, 3, device=self.device)
            self.masses = torch.ones(n_bodies, device=self.device)
            self.radii = torch.ones(n_bodies, device=self.device) * 0.5
            self.material_ids = torch.zeros(n_bodies, dtype=torch.int32, device=self.device)

        def set_state(
            self,
            positions: Optional[torch.Tensor] = None,
            velocities: Optional[torch.Tensor] = None,
            masses: Optional[torch.Tensor] = None,
            radii: Optional[torch.Tensor] = None
        ):
            """Set simulation state."""
            if positions is not None:
                self.positions = positions.to(self.device)
            if velocities is not None:
                self.velocities = velocities.to(self.device)
            if masses is not None:
                self.masses = masses.to(self.device)
            if radii is not None:
                self.radii = radii.to(self.device)

        def step(
            self,
            external_forces: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Perform one simulation step.

            Args:
                external_forces: Additional forces to apply

            Returns:
                Tuple of (positions, velocities) after the step
            """
            self.positions, self.velocities = self.integrator(
                self.positions,
                self.velocities,
                self.masses,
                self.radii,
                self.material_ids,
                external_forces
            )

            return self.positions, self.velocities

        def simulate(
            self,
            n_steps: int,
            external_forces: Optional[torch.Tensor] = None,
            return_trajectory: bool = True
        ) -> dict:
            """
            Run simulation for multiple steps.

            Args:
                n_steps: Number of simulation steps
                external_forces: Forces to apply at each step
                return_trajectory: Whether to return full trajectory

            Returns:
                Dictionary with simulation results
            """
            trajectory = [] if return_trajectory else None

            for step in range(n_steps):
                forces = external_forces[step] if external_forces is not None else None
                pos, vel = self.step(forces)

                if return_trajectory:
                    trajectory.append({
                        'positions': pos.clone(),
                        'velocities': vel.clone()
                    })

            return {
                'final_positions': self.positions,
                'final_velocities': self.velocities,
                'trajectory': trajectory
            }

    # Placeholder for backward compatibility
    TorchPhysicsFunction = VariationalContactFunction

else:
    # Define None for all exports when dependencies aren't available
    VariationalContactFunction = None
    variational_contact_forces = None
    DifferentiableContactLayer = None
    PhysicsIntegrator = None
    TorchSimulation = None
    TorchPhysicsFunction = None
    tensor_to_physgrad = None
    physgrad_to_tensor = None

    if not _TORCH_AVAILABLE:
        warnings.warn("PyTorch not available. Install with: pip install torch")
    if not _CPP_AVAILABLE:
        warnings.warn("PhysGrad C++ backend not available. Build with: python build_bindings.py")