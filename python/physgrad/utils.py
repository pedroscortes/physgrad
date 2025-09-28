"""
Utility functions for PhysGrad.
"""

import numpy as np
from typing import List, Tuple, Optional
from .core import Particle, RigidBody, Material

def create_particle_grid(
    grid_size: Tuple[int, int, int] = (10, 10, 10),
    spacing: float = 1.0,
    mass: float = 1.0,
    material: Optional[Material] = None
) -> List[Particle]:
    """Create a grid of particles."""
    particles = []
    if material is None:
        material = Material()

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(grid_size[2]):
                pos = np.array([i * spacing, j * spacing, k * spacing], dtype=np.float32)
                particle = Particle(position=pos, mass=mass, material=material)
                particles.append(particle)

    return particles

def create_rope(
    num_segments: int = 10,
    segment_length: float = 0.5,
    mass_per_segment: float = 0.1,
    start_position: np.ndarray = np.array([0, 5, 0])
) -> List[Particle]:
    """Create a rope as a chain of particles."""
    particles = []

    for i in range(num_segments):
        pos = start_position + np.array([0, -i * segment_length, 0])
        particle = Particle(position=pos, mass=mass_per_segment)
        particles.append(particle)

    return particles

def create_cloth(
    resolution: Tuple[int, int] = (10, 10),
    size: Tuple[float, float] = (2.0, 2.0),
    mass_per_particle: float = 0.01,
    position: np.ndarray = np.array([0, 5, 0])
) -> List[Particle]:
    """Create a cloth as a grid of particles."""
    particles = []
    spacing_x = size[0] / resolution[0]
    spacing_y = size[1] / resolution[1]

    for i in range(resolution[0]):
        for j in range(resolution[1]):
            pos = position + np.array([
                i * spacing_x - size[0]/2,
                0,
                j * spacing_y - size[1]/2
            ])
            particle = Particle(position=pos, mass=mass_per_particle)
            particles.append(particle)

    return particles

def create_rigid_body(
    shape: str = "sphere",
    dimensions: np.ndarray = np.array([1.0, 1.0, 1.0]),
    position: np.ndarray = np.array([0, 0, 0]),
    mass: float = 1.0,
    material: Optional[Material] = None
) -> RigidBody:
    """Create a rigid body with specified shape."""
    if material is None:
        material = Material()

    return RigidBody(
        center_of_mass=position,
        mass=mass,
        shape=shape,
        dimensions=dimensions,
        material=material
    )

def load_scene(filename: str):
    """Load a scene from file."""
    import json
    with open(filename, 'r') as f:
        scene_data = json.load(f)
    return scene_data

def save_scene(filename: str, scene_data):
    """Save a scene to file."""
    import json
    with open(filename, 'w') as f:
        json.dump(scene_data, f, indent=2)

def benchmark_performance(simulation, num_steps: int = 1000):
    """Benchmark simulation performance."""
    import time

    start_time = time.time()
    for _ in range(num_steps):
        simulation.step()
    end_time = time.time()

    total_time = end_time - start_time
    steps_per_second = num_steps / total_time

    return {
        "total_time": total_time,
        "steps_per_second": steps_per_second,
        "time_per_step": total_time / num_steps
    }