"""
PyTorch integration for PhysGrad.
"""

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    try:
        from . import physgrad_cpp

        class TorchSimulation:
            """PyTorch-based simulation wrapper."""
            def __init__(self, config):
                self.config = config

        class TorchPhysicsFunction:
            """PyTorch autograd function for physics operations."""
            @staticmethod
            def apply(positions, velocities, masses, forces, dt):
                if hasattr(physgrad_cpp, 'simulation_step_torch'):
                    return physgrad_cpp.simulation_step_torch(
                        positions, velocities, masses, forces, dt
                    )
                else:
                    raise RuntimeError("PyTorch integration not available")

        def tensor_to_physgrad(tensor):
            """Convert PyTorch tensor to PhysGrad format."""
            return tensor.detach().cpu().numpy()

        def physgrad_to_tensor(array, device='cpu', requires_grad=False):
            """Convert PhysGrad array to PyTorch tensor."""
            return torch.from_numpy(array).to(device).requires_grad_(requires_grad)

    except ImportError:
        TorchSimulation = None
        TorchPhysicsFunction = None
        tensor_to_physgrad = None
        physgrad_to_tensor = None
else:
    TorchSimulation = None
    TorchPhysicsFunction = None
    tensor_to_physgrad = None
    physgrad_to_tensor = None