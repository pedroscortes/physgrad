"""
PhysGrad: Differentiable Physics Simulation with GPU Acceleration
"""

import warnings
from typing import Optional, Union, List, Tuple, Dict, Any

# Check for optional dependencies
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. PyTorch integration disabled.")

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False
    warnings.warn("JAX not available. JAX integration disabled.")

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    raise ImportError("NumPy is required for PhysGrad")

# Import the compiled C++ extension
try:
    from . import physgrad_cpp
    _CPP_AVAILABLE = True
except ImportError as e:
    _CPP_AVAILABLE = False
    warnings.warn(f"PhysGrad C++ extension not available: {e}")

# Core simulation classes
from .core import (
    Simulation,
    SimulationConfig,
    Particle,
    RigidBody,
    Material,
    Environment
)

# Physics components
from .physics import (
    Force,
    GravityForce,
    SpringForce,
    DampingForce,
    CustomForce,
    Constraint,
    DistanceConstraint,
    SpringConstraint,
    PositionConstraint,
    Integrator,
    SymplecticEuler,
    VelocityVerlet,
    ForestRuth,
    Yoshida4
)

# Multi-GPU support
from .multigpu import (
    MultiGPUSimulation,
    GPUManager,
    PartitioningStrategy,
    CommunicationPattern
)

# Visualization
from .visualization import (
    Visualizer,
    RealTimeVisualizer,
    InteractiveControls,
    MathematicalOverlay
)

# Automatic differentiation
from .autodiff import (
    Variable,
    AutoDiffEngine,
    Optimizer,
    SGD,
    Adam,
    PhysicsOptimizer
)

# Framework integrations
if _TORCH_AVAILABLE:
    from .torch_integration import (
        TorchSimulation,
        TorchPhysicsFunction,
        tensor_to_physgrad,
        physgrad_to_tensor
    )

if _JAX_AVAILABLE:
    from .jax_integration import (
        JAXSimulation,
        register_jax_ops,
        array_to_physgrad,
        physgrad_to_array
    )

# Utilities and examples
from .utils import (
    create_particle_grid,
    create_rope,
    create_cloth,
    create_rigid_body,
    load_scene,
    save_scene,
    benchmark_performance
)

from .examples import (
    pendulum_demo,
    cloth_simulation,
    rigid_body_dynamics,
    multi_gpu_benchmark,
    optimization_example
)

# Version and capability information
__version__ = "0.1.0"
__author__ = "PhysGrad Team"
__email__ = "contact@physgrad.ai"

# Feature flags
FEATURES = {
    "cuda": _CPP_AVAILABLE and physgrad_cpp.cuda_available if _CPP_AVAILABLE else False,
    "pytorch": _TORCH_AVAILABLE and (_CPP_AVAILABLE and physgrad_cpp.pytorch_available if _CPP_AVAILABLE else False),
    "jax": _JAX_AVAILABLE and (_CPP_AVAILABLE and physgrad_cpp.jax_available if _CPP_AVAILABLE else False),
    "multi_gpu": _CPP_AVAILABLE,
    "visualization": True,  # Pure Python implementation
    "autodiff": True       # Pure Python implementation
}

def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    info = {
        "cpu_available": True,
        "cuda_available": FEATURES["cuda"],
        "gpu_count": 0,
        "gpu_names": [],
        "memory_info": {}
    }

    if FEATURES["cuda"] and _CPP_AVAILABLE:
        try:
            info["gpu_count"] = physgrad_cpp.get_available_gpus()
            # Additional GPU info would be retrieved from CUDA
        except Exception as e:
            warnings.warn(f"Could not retrieve GPU information: {e}")

    return info

def set_default_device(device: str) -> None:
    """Set the default compute device for simulations.

    Args:
        device: Device string ('cpu', 'cuda', 'cuda:0', etc.)
    """
    if device.startswith('cuda') and not FEATURES["cuda"]:
        raise RuntimeError("CUDA not available")

    # Implementation would set global device state
    pass

def enable_interactive_mode() -> None:
    """Enable interactive mode for real-time visualization and control."""
    # This would initialize GUI backends if available
    pass

def configure_logging(level: str = "INFO") -> None:
    """Configure PhysGrad logging.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    import logging
    logging.getLogger('physgrad').setLevel(getattr(logging, level.upper()))

# Module-level convenience functions
def quick_simulation(
    num_particles: int = 100,
    domain_size: Tuple[float, float, float] = (10.0, 10.0, 10.0),
    dt: float = 0.01,
    enable_gpu: bool = True,
    enable_visualization: bool = False
) -> 'Simulation':
    """Create a quick simulation setup with sensible defaults.

    Args:
        num_particles: Number of particles in the simulation
        domain_size: Size of the simulation domain (x, y, z)
        dt: Time step size
        enable_gpu: Whether to use GPU acceleration
        enable_visualization: Whether to enable real-time visualization

    Returns:
        Configured simulation instance
    """
    config = SimulationConfig(
        num_particles=num_particles,
        domain_size=domain_size,
        dt=dt,
        enable_gpu=enable_gpu and FEATURES["cuda"],
        enable_visualization=enable_visualization
    )

    return Simulation(config)

def benchmark(
    num_particles: List[int] = [100, 1000, 10000],
    num_steps: int = 1000,
    enable_multi_gpu: bool = False
) -> Dict[str, Any]:
    """Run performance benchmarks.

    Args:
        num_particles: List of particle counts to benchmark
        num_steps: Number of simulation steps per benchmark
        enable_multi_gpu: Whether to include multi-GPU benchmarks

    Returns:
        Dictionary containing benchmark results
    """
    results = {}

    for n in num_particles:
        sim = quick_simulation(num_particles=n, enable_visualization=False)

        import time
        start_time = time.time()

        for _ in range(num_steps):
            sim.step()

        elapsed = time.time() - start_time
        results[f"particles_{n}"] = {
            "time_per_step": elapsed / num_steps,
            "particles_per_second": n * num_steps / elapsed
        }

    return results

# Cleanup function
def cleanup() -> None:
    """Clean up PhysGrad resources and GPU memory."""
    if _CPP_AVAILABLE and FEATURES["cuda"]:
        # Would call C++ cleanup functions
        pass

# Register cleanup to be called on module unload
import atexit
atexit.register(cleanup)

# Export main classes and functions
__all__ = [
    # Core classes
    "Simulation", "SimulationConfig", "Particle", "RigidBody", "Material", "Environment",

    # Physics components
    "Force", "GravityForce", "SpringForce", "DampingForce", "CustomForce",
    "Constraint", "DistanceConstraint", "SpringConstraint", "PositionConstraint",
    "Integrator", "SymplecticEuler", "VelocityVerlet", "ForestRuth", "Yoshida4",

    # Multi-GPU
    "MultiGPUSimulation", "GPUManager", "PartitioningStrategy", "CommunicationPattern",

    # Visualization
    "Visualizer", "RealTimeVisualizer", "InteractiveControls", "MathematicalOverlay",

    # Automatic differentiation
    "Variable", "AutoDiffEngine", "Optimizer", "SGD", "Adam", "PhysicsOptimizer",

    # Framework integrations (conditionally available)
    *([
        "TorchSimulation", "TorchPhysicsFunction", "tensor_to_physgrad", "physgrad_to_tensor"
    ] if _TORCH_AVAILABLE else []),

    *([
        "JAXSimulation", "register_jax_ops", "array_to_physgrad", "physgrad_to_array"
    ] if _JAX_AVAILABLE else []),

    # Utilities
    "create_particle_grid", "create_rope", "create_cloth", "create_rigid_body",
    "load_scene", "save_scene", "benchmark_performance",

    # Examples
    "pendulum_demo", "cloth_simulation", "rigid_body_dynamics",
    "multi_gpu_benchmark", "optimization_example",

    # Module functions
    "quick_simulation", "benchmark", "get_device_info", "set_default_device",
    "enable_interactive_mode", "configure_logging", "cleanup",

    # Constants
    "FEATURES", "__version__"
]