"""Core simulation classes and configuration for PhysGrad."""

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
    SYMPLECTIC_EULER = "symplectic_euler"
    VELOCITY_VERLET = "velocity_verlet"
    FOREST_RUTH = "forest_ruth"
    YOSHIDA4 = "yoshida4"
    BLANES_MOAN8 = "blanes_moan8"


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


@dataclass
class SimulationConfig:
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
        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if any(d <= 0 for d in self.domain_size):
            raise ValueError("domain_size dimensions must be positive")

        if self.device == DeviceType.AUTO:
            if _CPP_AVAILABLE and physgrad_cpp.cuda_available and self.enable_gpu:
                self.device = DeviceType.CUDA
            else:
                self.device = DeviceType.CPU


@dataclass
class Material:
    density: float = 1000.0  # kg/m³
    restitution: float = 0.5  # Coefficient of restitution
    friction: float = 0.3     # Coefficient of friction
    viscosity: float = 0.0
    thermal_conductivity: float = 0.0
    name: str = "default"

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError("density must be positive")
        if not 0 <= self.restitution <= 1:
            raise ValueError("restitution must be between 0 and 1")
        if self.friction < 0:
            raise ValueError("friction must be non-negative")


@dataclass
class Particle:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    radius: float = 0.1
    material: Material = field(default_factory=Material)
    fixed: bool = False
    id: int = -1

    def __post_init__(self):
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
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def momentum(self) -> np.ndarray:
        return self.mass * self.velocity


@dataclass
class RigidBody:
    center_of_mass: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    mass: float = 1.0
    inertia_tensor: np.ndarray = field(default_factory=lambda: np.eye(3))
    material: Material = field(default_factory=Material)
    shape: str = "sphere"
    dimensions: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    fixed: bool = False
    id: int = -1

    def __post_init__(self):
        self.center_of_mass = np.asarray(self.center_of_mass, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float32)
        self.orientation = np.asarray(self.orientation, dtype=np.float32)
        self.inertia_tensor = np.asarray(self.inertia_tensor, dtype=np.float32)
        self.dimensions = np.asarray(self.dimensions, dtype=np.float32)

        self.orientation = self.orientation / np.linalg.norm(self.orientation)

        if self.mass <= 0:
            raise ValueError("mass must be positive")

    def total_energy(self) -> float:
        translational = 0.5 * self.mass * np.dot(self.velocity, self.velocity)
        rotational = 0.5 * np.dot(self.angular_velocity,
                                 np.dot(self.inertia_tensor, self.angular_velocity))
        return translational + rotational


@dataclass
class Environment:
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, -9.81, 0]))
    air_density: float = 1.225
    temperature: float = 293.15
    pressure: float = 101325.0
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

        self.particles: List[Particle] = []
        self.rigid_bodies: List[RigidBody] = []
        self.environment = Environment()

        self.forces: List['Force'] = []
        self.constraints: List['Constraint'] = []

        self._initialize_backend()

        self.trajectory_data = [] if config.save_trajectory else None

        self.performance_stats = {
            "total_time": 0.0,
            "average_step_time": 0.0,
            "steps_per_second": 0.0,
            "last_step_time": 0.0
        }

        self._visualizer = None
        self._visualization_enabled = False

    def _initialize_backend(self):
        if _CPP_AVAILABLE and self.config.device != DeviceType.CPU:
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
        particle.id = len(self.particles)
        self.particles.append(particle)

        if self._use_cpp and len(self.particles) == 1:
            self._cpp_sim.initialize()

        return particle.id

    def add_particles(self, particles: List[Particle]) -> List[int]:
        ids = []
        for particle in particles:
            ids.append(self.add_particle(particle))
        return ids

    def add_rigid_body(self, rigid_body: RigidBody) -> int:
        rigid_body.id = len(self.rigid_bodies)
        self.rigid_bodies.append(rigid_body)
        return rigid_body.id

    def add_force(self, force: 'Force') -> None:
        self.forces.append(force)

    def remove_force(self, force: 'Force') -> bool:
        try:
            self.forces.remove(force)
            return True
        except ValueError:
            return False

    def add_constraint(self, constraint: 'Constraint') -> None:
        self.constraints.append(constraint)

    def remove_constraint(self, constraint: 'Constraint') -> bool:
        try:
            self.constraints.remove(constraint)
            return True
        except ValueError:
            return False

    def step(self) -> None:
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
        # Collect all forces
        total_forces = {}

        # Apply all forces in the system
        for force in self.forces:
            if force.enabled:
                force_dict = force.compute_force(self.particles, self.rigid_bodies, self.time)
                for particle_id, force_vec in force_dict.items():
                    if particle_id in total_forces:
                        total_forces[particle_id] += force_vec
                    else:
                        total_forces[particle_id] = force_vec.copy()

        # Apply all constraints
        for constraint in self.constraints:
            if constraint.enabled:
                constraint_forces = constraint.compute_constraint_force(
                    self.particles, self.rigid_bodies, self.config.dt
                )
                for particle_id, force_vec in constraint_forces.items():
                    if particle_id in total_forces:
                        total_forces[particle_id] += force_vec
                    else:
                        total_forces[particle_id] = force_vec.copy()

        # Store forces for visualization
        if self.particles:
            forces_array = np.zeros((len(self.particles), 3))
            for i, particle in enumerate(self.particles):
                if particle.id in total_forces:
                    forces_array[i] = total_forces[particle.id]
            self._last_forces = forces_array

        # Integrate particles using symplectic Euler
        from .physics import SymplecticEuler
        integrator = SymplecticEuler()
        integrator.integrate(self.particles, self.rigid_bodies, total_forces, self.config.dt)

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

    def enable_visualization(self, width: int = 1280, height: int = 720) -> bool:
        """Enable real-time OpenGL/ImGui visualization.

        Args:
            width: Window width in pixels
            height: Window height in pixels

        Returns:
            True if visualization was successfully initialized
        """
        if not _CPP_AVAILABLE:
            warnings.warn("C++ backend not available, visualization requires compiled bindings")
            return False

        try:
            self._visualizer = physgrad_cpp.VisualizationManager()
            if self._visualizer.initialize(width, height):
                self._visualization_enabled = True
                return True
            else:
                warnings.warn("Failed to initialize OpenGL/ImGui visualization")
                return False
        except Exception as e:
            warnings.warn(f"Visualization initialization failed: {e}")
            return False

    def disable_visualization(self) -> None:
        """Disable real-time visualization."""
        if self._visualizer:
            self._visualizer.shutdown()
            self._visualizer = None
        self._visualization_enabled = False

    def update_visualization(self) -> None:
        """Update visualization with current simulation state."""
        if not self._visualization_enabled or not self._visualizer:
            return

        if not self.particles:
            return

        # Extract particle data
        positions = np.array([p.position for p in self.particles])
        velocities = np.array([p.velocity for p in self.particles])
        masses = np.array([p.mass for p in self.particles])

        # Update visualizer
        self._visualizer.update_from_simulation(positions, velocities, masses)

        # Update energy information
        kinetic = self.get_kinetic_energy()
        potential = self.get_potential_energy()
        self._visualizer.update_energy(kinetic, potential)

        # Update forces if available
        if hasattr(self, '_last_forces') and self._last_forces is not None:
            self._visualizer.update_forces(self._last_forces)

    def render_visualization(self) -> None:
        """Render one frame of visualization."""
        if self._visualization_enabled and self._visualizer:
            self._visualizer.render()

    def should_close_visualization(self) -> bool:
        """Check if the visualization window should close."""
        if self._visualization_enabled and self._visualizer:
            return self._visualizer.should_close()
        return False

    def get_visualization_params(self) -> Optional[Dict[str, Any]]:
        """Get interactive parameters from visualization interface.

        Returns:
            Dictionary of interactive parameters or None if visualization disabled
        """
        if self._visualization_enabled and self._visualizer:
            return self._visualizer.get_interactive_params()
        return None

    def run_with_visualization(self, max_steps: Optional[int] = None) -> None:
        """Run simulation with real-time visualization.

        Args:
            max_steps: Maximum number of steps to run (None for infinite)
        """
        if not self._visualization_enabled:
            if not self.enable_visualization():
                warnings.warn("Failed to enable visualization, falling back to regular run")
                if max_steps:
                    self.run(max_steps)
                return

        step = 0
        while not self.should_close_visualization():
            if max_steps is not None and step >= max_steps:
                break

            # Check if simulation should be running from UI
            if self._visualizer.is_simulation_running() or self._visualizer.should_single_step():
                self.step()
                if self._visualizer.should_single_step():
                    self._visualizer.reset_single_step()

            # Update and render visualization
            self.update_visualization()
            self.render_visualization()

            step += 1

        print(f"Simulation completed after {step} steps")
        self.disable_visualization()