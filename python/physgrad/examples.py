"""
Example simulations and demos for PhysGrad.
"""

import numpy as np
from .core import Simulation, SimulationConfig, Particle
from .physics import GravityForce, SpringConstraint, DistanceConstraint
from .utils import create_particle_grid, create_rope, create_cloth

def pendulum_demo():
    """Simple pendulum simulation."""
    config = SimulationConfig(
        num_particles=2,
        dt=0.01,
        enable_visualization=True
    )

    sim = Simulation(config)

    anchor = Particle(
        position=np.array([0, 5, 0]),
        mass=1.0,
        fixed=True
    )

    bob = Particle(
        position=np.array([2, 0, 0]),
        mass=1.0
    )

    sim.add_particle(anchor)
    sim.add_particle(bob)

    return sim

def cloth_simulation():
    """Cloth physics simulation."""
    config = SimulationConfig(
        num_particles=100,
        dt=0.01,
        enable_collisions=True
    )

    sim = Simulation(config)
    particles = create_cloth(resolution=(10, 10))
    sim.add_particles(particles)

    return sim

def rigid_body_dynamics():
    """Rigid body collision demo."""
    config = SimulationConfig(
        num_particles=50,
        dt=0.01,
        enable_collisions=True
    )

    sim = Simulation(config)
    return sim

def multi_gpu_benchmark():
    """Multi-GPU performance benchmark."""
    try:
        from .multigpu import MultiGPUSimulation, MultiGPUConfig
        config = MultiGPUConfig(device_ids=[0, 1])
        sim = MultiGPUSimulation(config)
        return sim
    except ImportError:
        print("Multi-GPU support not available")
        return None

def optimization_example():
    """Parameter optimization example."""
    try:
        from .autodiff import PhysicsOptimizer
        optimizer = PhysicsOptimizer()
        return optimizer
    except ImportError:
        print("Autodiff support not available")
        return None