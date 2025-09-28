"""
Core simulation classes and configuration for PhysGrad.

This module provides the main simulation interface and configuration classes
that users interact with for setting up and running physics simulations.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Import C++ backend if available
try:
    from . import physgrad_cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


class IntegratorType(Enum):
    """Available numerical integration schemes."""
    SYMPLECTIC_EULER = "symplectic_euler"
    VELOCITY_VERLET = "velocity_verlet"
    FOREST_RUTH = "forest_ruth"
    YOSHIDA4 = "yoshida4"
    BLANES_MOAN8 = "blanes_moan8"


class DeviceType(Enum):
    """Available compute devices."""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


@dataclass
class SimulationConfig:
    """Configuration for physics simulation.

    This class contains all the parameters needed to configure a physics simulation,
    including numerical integration settings, physical parameters, and computational options.
    """
    # Simulation parameters
    num_particles: int = 1000
    dt: float = 0.01
    max_steps: int = -1  # -1 means no limit
    domain_size: Tuple[float, float, float] = (10.0, 10.0, 10.0)

    # Physical parameters
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    damping: float = 0.01
    enable_collisions: bool = True
    enable_constraints: bool = True

    # Numerical integration
    integrator: IntegratorType = IntegratorType.VELOCITY_VERLET
    enable_energy_monitoring: bool = True
    energy_tolerance: float = 1e-6

    # Computational settings
    device: DeviceType = DeviceType.AUTO
    enable_gpu: bool = True
    enable_multi_gpu: bool = False
    num_threads: int = -1  # -1 means auto-detect

    # Visualization
    enable_visualization: bool = False
    visualization_fps: int = 60
    save_trajectory: bool = False

    # Output and logging
    output_frequency: int = 10
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if any(d <= 0 for d in self.domain_size):
            raise ValueError("domain_size dimensions must be positive")

        # Auto-detect device if needed
        if self.device == DeviceType.AUTO:
            if _CPP_AVAILABLE and physgrad_cpp.cuda_available and self.enable_gpu:
                self.device = DeviceType.CUDA
            else:
                self.device = DeviceType.CPU


@dataclass
class Material:
    """Material properties for particles and rigid bodies."""
    density: float = 1000.0  # kg/m³
    restitution: float = 0.5  # Coefficient of restitution
    friction: float = 0.3     # Coefficient of friction
    viscosity: float = 0.0    # Fluid viscosity
    thermal_conductivity: float = 0.0
    name: str = "default"

    def __post_init__(self):
        """Validate material properties."""
        if self.density <= 0:
            raise ValueError("density must be positive")
        if not 0 <= self.restitution <= 1:
            raise ValueError("restitution must be between 0 and 1")
        if self.friction < 0:
            raise ValueError("friction must be non-negative")


@dataclass
class Particle:
    """Individual particle in the simulation."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    radius: float = 0.1
    material: Material = field(default_factory=Material)
    fixed: bool = False
    id: int = -1

    def __post_init__(self):
        """Validate particle properties."""
        self.position = np.asarray(self.position, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)
        self.acceleration = np.asarray(self.acceleration, dtype=np.float32)

        if self.position.shape != (3,):
            raise ValueError("position must be a 3D vector")
        if self.velocity.shape != (3,):
            raise ValueError("velocity must be a 3D vector")
        if self.acceleration.shape != (3,):
            raise ValueError("acceleration must be a 3D vector")
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.radius <= 0:
            raise ValueError("radius must be positive")

    def kinetic_energy(self) -> float:
        """Calculate kinetic energy of the particle."""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def momentum(self) -> np.ndarray:
        """Calculate momentum of the particle."""
        return self.mass * self.velocity


@dataclass
class RigidBody:
    """Rigid body composed of multiple particles or with continuous mass distribution."""
    center_of_mass: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # quaternion
    mass: float = 1.0
    inertia_tensor: np.ndarray = field(default_factory=lambda: np.eye(3))
    material: Material = field(default_factory=Material)
    shape: str = "sphere"
    dimensions: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    fixed: bool = False
    id: int = -1

    def __post_init__(self):
        """Validate rigid body properties."""
        self.center_of_mass = np.asarray(self.center_of_mass, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float32)
        self.orientation = np.asarray(self.orientation, dtype=np.float32)
        self.inertia_tensor = np.asarray(self.inertia_tensor, dtype=np.float32)
        self.dimensions = np.asarray(self.dimensions, dtype=np.float32)

        # Normalize quaternion
        self.orientation = self.orientation / np.linalg.norm(self.orientation)

        if self.mass <= 0:
            raise ValueError("mass must be positive")

    def total_energy(self) -> float:
        """Calculate total kinetic energy (translational + rotational)."""
        translational = 0.5 * self.mass * np.dot(self.velocity, self.velocity)
        rotational = 0.5 * np.dot(self.angular_velocity,
                                 np.dot(self.inertia_tensor, self.angular_velocity))
        return translational + rotational


@dataclass
class Environment:
    """Global environment settings for the simulation."""
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, -9.81, 0]))
    air_density: float = 1.225  # kg/m³
    temperature: float = 293.15  # K
    pressure: float = 101325.0   # Pa
    wind_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    boundaries: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate environment parameters."""
        self.gravity = np.asarray(self.gravity, dtype=np.float32)
        self.wind_velocity = np.asarray(self.wind_velocity, dtype=np.float32)

        if self.air_density < 0:
            raise ValueError("air_density must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.pressure < 0:
            raise ValueError("pressure must be non-negative")


class Simulation:
    """Main physics simulation class.

    This is the primary interface for running physics simulations. It manages
    particles, rigid bodies, forces, constraints, and the numerical integration.
    """

    def __init__(self, config: SimulationConfig):
        """Initialize simulation with given configuration.

        Args:
            config: Simulation configuration parameters
        """
        self.config = config
        self.time = 0.0
        self.step_count = 0

        # Initialize containers
        self.particles: List[Particle] = []
        self.rigid_bodies: List[RigidBody] = []
        self.environment = Environment()

        # Initialize backend
        self._initialize_backend()

        # Storage for trajectory data
        self.trajectory_data = [] if config.save_trajectory else None

        # Performance tracking
        self.performance_stats = {
            "total_time": 0.0,
            "average_step_time": 0.0,
            "steps_per_second": 0.0,
            "last_step_time": 0.0
        }

    def _initialize_backend(self):
        """Initialize the computational backend."""
        if _CPP_AVAILABLE and self.config.device != DeviceType.CPU:
            # Initialize C++ backend
            params = physgrad_cpp.SimulationParams()
            params.dt = self.config.dt
            params.num_particles = self.config.num_particles
            params.gravity_x, params.gravity_y, params.gravity_z = self.config.gravity
            params.damping = self.config.damping
            params.enable_collisions = self.config.enable_collisions
            params.enable_constraints = self.config.enable_constraints

            self._cpp_sim = physgrad_cpp.PhysicsSimulation(params)
            self._use_cpp = True
        else:
            self._cpp_sim = None
            self._use_cpp = False
            if self.config.device == DeviceType.CUDA:
                warnings.warn("CUDA requested but not available, falling back to CPU")

    def add_particle(self, particle: Particle) -> int:
        """Add a particle to the simulation.

        Args:
            particle: Particle to add

        Returns:
            Particle ID
        """
        particle.id = len(self.particles)
        self.particles.append(particle)

        if self._use_cpp and len(self.particles) == 1:
            # Initialize C++ simulation with first particle
            self._cpp_sim.initialize()

        return particle.id

    def add_particles(self, particles: List[Particle]) -> List[int]:
        """Add multiple particles to the simulation.

        Args:
            particles: List of particles to add

        Returns:
            List of particle IDs
        """
        ids = []
        for particle in particles:
            ids.append(self.add_particle(particle))
        return ids

    def add_rigid_body(self, rigid_body: RigidBody) -> int:
        """Add a rigid body to the simulation.

        Args:
            rigid_body: Rigid body to add

        Returns:
            Rigid body ID
        """
        rigid_body.id = len(self.rigid_bodies)
        self.rigid_bodies.append(rigid_body)
        return rigid_body.id

    def step(self) -> None:
        """Advance simulation by one time step."""
        import time
        start_time = time.time()

        if self._use_cpp:
            self._step_cpp()
        else:
            self._step_python()

        self.time += self.config.dt
        self.step_count += 1

        # Update performance statistics
        step_time = time.time() - start_time
        self.performance_stats["last_step_time"] = step_time
        self.performance_stats["total_time"] += step_time
        self.performance_stats["average_step_time"] = (
            self.performance_stats["total_time"] / self.step_count
        )
        self.performance_stats["steps_per_second"] = 1.0 / step_time

        # Save trajectory data if enabled
        if self.trajectory_data is not None:
            self._save_trajectory_step()

    def _step_cpp(self) -> None:
        """Perform simulation step using C++ backend."""
        # Update particle data in C++ simulation
        if self.particles:
            pos_x = [p.position[0] for p in self.particles]
            pos_y = [p.position[1] for p in self.particles]
            pos_z = [p.position[2] for p in self.particles]
            self._cpp_sim.set_positions(
                np.array(pos_x, dtype=np.float32),
                np.array(pos_y, dtype=np.float32),
                np.array(pos_z, dtype=np.float32)
            )

            vel_x = [p.velocity[0] for p in self.particles]
            vel_y = [p.velocity[1] for p in self.particles]
            vel_z = [p.velocity[2] for p in self.particles]
            self._cpp_sim.set_velocities(
                np.array(vel_x, dtype=np.float32),
                np.array(vel_y, dtype=np.float32),
                np.array(vel_z, dtype=np.float32)
            )

            masses = [p.mass for p in self.particles]
            self._cpp_sim.set_masses(np.array(masses, dtype=np.float32))

        # Perform simulation step
        self._cpp_sim.step()

        # Retrieve updated data
        if self.particles:
            new_pos = self._cpp_sim.get_positions()
            new_vel = self._cpp_sim.get_velocities()

            for i, particle in enumerate(self.particles):
                particle.position[0] = new_pos[0][i]
                particle.position[1] = new_pos[1][i]
                particle.position[2] = new_pos[2][i]
                particle.velocity[0] = new_vel[0][i]
                particle.velocity[1] = new_vel[1][i]
                particle.velocity[2] = new_vel[2][i]

    def _step_python(self) -> None:
        """Perform simulation step using pure Python backend."""
        # Simple Euler integration for Python fallback
        for particle in self.particles:
            if not particle.fixed:
                # Apply gravity
                particle.acceleration = np.array(self.environment.gravity)

                # Simple damping
                particle.acceleration -= self.config.damping * particle.velocity

                # Update velocity and position
                particle.velocity += particle.acceleration * self.config.dt
                particle.position += particle.velocity * self.config.dt

    def _save_trajectory_step(self) -> None:
        """Save current state to trajectory data."""
        step_data = {
            "time": self.time,
            "particles": [
                {
                    "position": p.position.copy(),
                    "velocity": p.velocity.copy(),
                    "acceleration": p.acceleration.copy()
                }
                for p in self.particles
            ],
            "rigid_bodies": [
                {
                    "center_of_mass": rb.center_of_mass.copy(),
                    "velocity": rb.velocity.copy(),
                    "angular_velocity": rb.angular_velocity.copy(),
                    "orientation": rb.orientation.copy()
                }
                for rb in self.rigid_bodies
            ]
        }
        self.trajectory_data.append(step_data)

    def run(self, num_steps: int) -> None:
        """Run simulation for specified number of steps.

        Args:
            num_steps: Number of simulation steps to execute
        """
        for _ in range(num_steps):
            self.step()

            if self.config.max_steps > 0 and self.step_count >= self.config.max_steps:
                break

    def run_until(self, end_time: float) -> None:
        """Run simulation until specified time.

        Args:
            end_time: Target simulation time
        """
        while self.time < end_time:
            self.step()

            if self.config.max_steps > 0 and self.step_count >= self.config.max_steps:
                break

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.time = 0.0
        self.step_count = 0

        if self._use_cpp:
            self._cpp_sim.reset()

        # Clear trajectory data
        if self.trajectory_data is not None:
            self.trajectory_data.clear()

        # Reset performance stats
        self.performance_stats = {
            "total_time": 0.0,
            "average_step_time": 0.0,
            "steps_per_second": 0.0,
            "last_step_time": 0.0
        }

    def get_kinetic_energy(self) -> float:
        """Calculate total kinetic energy of the system."""
        if self._use_cpp:
            return self._cpp_sim.get_kinetic_energy()
        else:
            total_ke = sum(p.kinetic_energy() for p in self.particles)
            total_ke += sum(rb.total_energy() for rb in self.rigid_bodies)
            return total_ke

    def get_potential_energy(self) -> float:
        """Calculate total potential energy of the system."""
        if self._use_cpp:
            return self._cpp_sim.get_potential_energy()
        else:
            # Simple gravitational potential energy calculation
            total_pe = 0.0
            for particle in self.particles:
                height = particle.position[1]  # Assuming y is up
                total_pe += particle.mass * abs(self.environment.gravity[1]) * height
            return total_pe

    def get_total_energy(self) -> float:
        """Calculate total energy of the system."""
        return self.get_kinetic_energy() + self.get_potential_energy()

    def apply_force(self, particle_id: int, force: np.ndarray) -> None:
        """Apply force to a specific particle.

        Args:
            particle_id: ID of the particle
            force: Force vector to apply
        """
        if self._use_cpp:
            self._cpp_sim.apply_force(particle_id, force[0], force[1], force[2])
        else:
            if 0 <= particle_id < len(self.particles):
                self.particles[particle_id].acceleration += force / self.particles[particle_id].mass

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the simulation."""
        return self.performance_stats.copy()

    def save_state(self, filename: str) -> None:
        """Save current simulation state to file.

        Args:
            filename: File path to save state
        """
        import pickle
        state = {
            "config": self.config,
            "time": self.time,
            "step_count": self.step_count,
            "particles": self.particles,
            "rigid_bodies": self.rigid_bodies,
            "environment": self.environment
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filename: str) -> None:
        """Load simulation state from file.

        Args:
            filename: File path to load state from
        """
        import pickle
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        self.config = state["config"]
        self.time = state["time"]
        self.step_count = state["step_count"]
        self.particles = state["particles"]
        self.rigid_bodies = state["rigid_bodies"]
        self.environment = state["environment"]

        # Reinitialize backend with loaded config
        self._initialize_backend()