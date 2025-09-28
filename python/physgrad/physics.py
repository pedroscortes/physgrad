"""
Physics components including forces, constraints, and integrators.

This module provides classes for different types of forces, constraints,
and numerical integration schemes used in physics simulations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .core import Particle, RigidBody


class ForceType(Enum):
    """Types of forces available in the simulation."""
    GRAVITY = "gravity"
    SPRING = "spring"
    DAMPING = "damping"
    CUSTOM = "custom"
    MAGNETIC = "magnetic"
    ELECTRIC = "electric"
    FLUID_DRAG = "fluid_drag"


class ConstraintType(Enum):
    """Types of constraints available in the simulation."""
    DISTANCE = "distance"
    SPRING = "spring"
    POSITION = "position"
    ANGLE = "angle"
    HINGE = "hinge"
    BALL_SOCKET = "ball_socket"


class Force(ABC):
    """Abstract base class for all forces."""

    def __init__(self, name: str = ""):
        self.name = name
        self.enabled = True

    @abstractmethod
    def compute_force(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                     time: float) -> Dict[int, np.ndarray]:
        """Compute forces on particles and rigid bodies.

        Args:
            particles: List of particles in the simulation
            rigid_bodies: List of rigid bodies in the simulation
            time: Current simulation time

        Returns:
            Dictionary mapping particle/body IDs to force vectors
        """
        pass

    def enable(self):
        """Enable this force."""
        self.enabled = True

    def disable(self):
        """Disable this force."""
        self.enabled = False


class GravityForce(Force):
    """Gravitational force affecting all particles and rigid bodies."""

    def __init__(self, gravity: np.ndarray = np.array([0, -9.81, 0]), name: str = "gravity"):
        super().__init__(name)
        self.gravity = np.asarray(gravity, dtype=np.float32)

    def compute_force(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                     time: float) -> Dict[int, np.ndarray]:
        forces = {}

        if not self.enabled:
            return forces

        # Apply gravity to particles
        for particle in particles:
            if not particle.fixed:
                forces[particle.id] = particle.mass * self.gravity

        # Apply gravity to rigid bodies
        for rigid_body in rigid_bodies:
            if not rigid_body.fixed:
                forces[f"rb_{rigid_body.id}"] = rigid_body.mass * self.gravity

        return forces


class SpringForce(Force):
    """Spring force between two particles or between particle and fixed point."""

    def __init__(self, particle1_id: int, particle2_id: Optional[int] = None,
                 fixed_point: Optional[np.ndarray] = None,
                 spring_constant: float = 100.0, rest_length: float = 1.0,
                 damping: float = 0.1, name: str = "spring"):
        super().__init__(name)
        self.particle1_id = particle1_id
        self.particle2_id = particle2_id
        self.fixed_point = np.asarray(fixed_point) if fixed_point is not None else None
        self.spring_constant = spring_constant
        self.rest_length = rest_length
        self.damping = damping

        if particle2_id is None and fixed_point is None:
            raise ValueError("Must specify either particle2_id or fixed_point")

    def compute_force(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                     time: float) -> Dict[int, np.ndarray]:
        forces = {}

        if not self.enabled:
            return forces

        # Find particle1
        particle1 = None
        for p in particles:
            if p.id == self.particle1_id:
                particle1 = p
                break

        if particle1 is None:
            return forces

        # Determine second position
        if self.particle2_id is not None:
            # Spring between two particles
            particle2 = None
            for p in particles:
                if p.id == self.particle2_id:
                    particle2 = p
                    break

            if particle2 is None:
                return forces

            pos2 = particle2.position
            vel2 = particle2.velocity
        else:
            # Spring to fixed point
            pos2 = self.fixed_point
            vel2 = np.zeros(3)

        # Calculate spring force
        displacement = particle1.position - pos2
        distance = np.linalg.norm(displacement)

        if distance > 1e-8:  # Avoid division by zero
            direction = displacement / distance
            spring_force = -self.spring_constant * (distance - self.rest_length) * direction

            # Add damping force
            relative_velocity = particle1.velocity - vel2
            damping_force = -self.damping * relative_velocity

            total_force = spring_force + damping_force

            forces[self.particle1_id] = total_force

            # Apply reaction force to particle2 if it exists
            if self.particle2_id is not None and not particle2.fixed:
                forces[self.particle2_id] = -total_force

        return forces


class DampingForce(Force):
    """Velocity-proportional damping force."""

    def __init__(self, damping_coefficient: float = 0.1, name: str = "damping"):
        super().__init__(name)
        self.damping_coefficient = damping_coefficient

    def compute_force(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                     time: float) -> Dict[int, np.ndarray]:
        forces = {}

        if not self.enabled:
            return forces

        # Apply damping to particles
        for particle in particles:
            if not particle.fixed:
                forces[particle.id] = -self.damping_coefficient * particle.velocity

        # Apply damping to rigid bodies
        for rigid_body in rigid_bodies:
            if not rigid_body.fixed:
                forces[f"rb_{rigid_body.id}"] = -self.damping_coefficient * rigid_body.velocity

        return forces


class CustomForce(Force):
    """User-defined custom force."""

    def __init__(self, force_function: Callable[[List[Particle], List[RigidBody], float], Dict[int, np.ndarray]],
                 name: str = "custom"):
        super().__init__(name)
        self.force_function = force_function

    def compute_force(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                     time: float) -> Dict[int, np.ndarray]:
        if not self.enabled:
            return {}

        return self.force_function(particles, rigid_bodies, time)


class FluidDragForce(Force):
    """Fluid drag force proportional to velocity squared."""

    def __init__(self, drag_coefficient: float = 0.01, fluid_density: float = 1.0,
                 reference_area: float = 0.01, name: str = "fluid_drag"):
        super().__init__(name)
        self.drag_coefficient = drag_coefficient
        self.fluid_density = fluid_density
        self.reference_area = reference_area

    def compute_force(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                     time: float) -> Dict[int, np.ndarray]:
        forces = {}

        if not self.enabled:
            return forces

        for particle in particles:
            if not particle.fixed:
                velocity_magnitude = np.linalg.norm(particle.velocity)
                if velocity_magnitude > 1e-8:
                    drag_magnitude = 0.5 * self.drag_coefficient * self.fluid_density * \
                                   self.reference_area * velocity_magnitude**2
                    drag_direction = -particle.velocity / velocity_magnitude
                    forces[particle.id] = drag_magnitude * drag_direction

        return forces


class Constraint(ABC):
    """Abstract base class for all constraints."""

    def __init__(self, name: str = ""):
        self.name = name
        self.enabled = True

    @abstractmethod
    def compute_constraint_force(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                               dt: float) -> Dict[int, np.ndarray]:
        """Compute constraint forces.

        Args:
            particles: List of particles in the simulation
            rigid_bodies: List of rigid bodies in the simulation
            dt: Time step size

        Returns:
            Dictionary mapping particle/body IDs to constraint force vectors
        """
        pass

    @abstractmethod
    def get_constraint_error(self, particles: List[Particle], rigid_bodies: List[RigidBody]) -> float:
        """Get constraint violation error.

        Args:
            particles: List of particles in the simulation
            rigid_bodies: List of rigid bodies in the simulation

        Returns:
            Constraint error magnitude
        """
        pass

    def enable(self):
        """Enable this constraint."""
        self.enabled = True

    def disable(self):
        """Disable this constraint."""
        self.enabled = False


class DistanceConstraint(Constraint):
    """Constraint maintaining fixed distance between two particles."""

    def __init__(self, particle1_id: int, particle2_id: int, distance: float,
                 stiffness: float = 1000.0, name: str = "distance"):
        super().__init__(name)
        self.particle1_id = particle1_id
        self.particle2_id = particle2_id
        self.distance = distance
        self.stiffness = stiffness

    def compute_constraint_force(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                               dt: float) -> Dict[int, np.ndarray]:
        forces = {}

        if not self.enabled:
            return forces

        # Find particles
        particle1 = particle2 = None
        for p in particles:
            if p.id == self.particle1_id:
                particle1 = p
            elif p.id == self.particle2_id:
                particle2 = p

        if particle1 is None or particle2 is None:
            return forces

        # Calculate constraint force
        displacement = particle1.position - particle2.position
        current_distance = np.linalg.norm(displacement)

        if current_distance > 1e-8:
            direction = displacement / current_distance
            error = current_distance - self.distance
            force_magnitude = -self.stiffness * error

            constraint_force = force_magnitude * direction

            if not particle1.fixed:
                forces[self.particle1_id] = constraint_force
            if not particle2.fixed:
                forces[self.particle2_id] = -constraint_force

        return forces

    def get_constraint_error(self, particles: List[Particle], rigid_bodies: List[RigidBody]) -> float:
        # Find particles
        particle1 = particle2 = None
        for p in particles:
            if p.id == self.particle1_id:
                particle1 = p
            elif p.id == self.particle2_id:
                particle2 = p

        if particle1 is None or particle2 is None:
            return 0.0

        displacement = particle1.position - particle2.position
        current_distance = np.linalg.norm(displacement)
        return abs(current_distance - self.distance)


class SpringConstraint(Constraint):
    """Spring-like constraint between two particles."""

    def __init__(self, particle1_id: int, particle2_id: int, rest_length: float,
                 spring_constant: float = 100.0, damping: float = 0.1, name: str = "spring_constraint"):
        super().__init__(name)
        self.particle1_id = particle1_id
        self.particle2_id = particle2_id
        self.rest_length = rest_length
        self.spring_constant = spring_constant
        self.damping = damping

    def compute_constraint_force(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                               dt: float) -> Dict[int, np.ndarray]:
        forces = {}

        if not self.enabled:
            return forces

        # Find particles
        particle1 = particle2 = None
        for p in particles:
            if p.id == self.particle1_id:
                particle1 = p
            elif p.id == self.particle2_id:
                particle2 = p

        if particle1 is None or particle2 is None:
            return forces

        # Calculate spring constraint force
        displacement = particle1.position - particle2.position
        distance = np.linalg.norm(displacement)

        if distance > 1e-8:
            direction = displacement / distance
            spring_force = -self.spring_constant * (distance - self.rest_length) * direction

            # Add damping
            relative_velocity = particle1.velocity - particle2.velocity
            damping_force = -self.damping * relative_velocity

            total_force = spring_force + damping_force

            if not particle1.fixed:
                forces[self.particle1_id] = total_force
            if not particle2.fixed:
                forces[self.particle2_id] = -total_force

        return forces

    def get_constraint_error(self, particles: List[Particle], rigid_bodies: List[RigidBody]) -> float:
        # For spring constraints, error is the deviation from rest length
        particle1 = particle2 = None
        for p in particles:
            if p.id == self.particle1_id:
                particle1 = p
            elif p.id == self.particle2_id:
                particle2 = p

        if particle1 is None or particle2 is None:
            return 0.0

        displacement = particle1.position - particle2.position
        current_distance = np.linalg.norm(displacement)
        return abs(current_distance - self.rest_length)


class PositionConstraint(Constraint):
    """Constraint fixing a particle to a specific position."""

    def __init__(self, particle_id: int, position: np.ndarray, stiffness: float = 1000.0,
                 name: str = "position"):
        super().__init__(name)
        self.particle_id = particle_id
        self.position = np.asarray(position, dtype=np.float32)
        self.stiffness = stiffness

    def compute_constraint_force(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                               dt: float) -> Dict[int, np.ndarray]:
        forces = {}

        if not self.enabled:
            return forces

        # Find particle
        particle = None
        for p in particles:
            if p.id == self.particle_id:
                particle = p
                break

        if particle is None or particle.fixed:
            return forces

        # Calculate restoring force
        displacement = particle.position - self.position
        constraint_force = -self.stiffness * displacement

        forces[self.particle_id] = constraint_force

        return forces

    def get_constraint_error(self, particles: List[Particle], rigid_bodies: List[RigidBody]) -> float:
        # Find particle
        particle = None
        for p in particles:
            if p.id == self.particle_id:
                particle = p
                break

        if particle is None:
            return 0.0

        displacement = particle.position - self.position
        return np.linalg.norm(displacement)


class Integrator(ABC):
    """Abstract base class for numerical integrators."""

    def __init__(self, name: str = ""):
        self.name = name

    @abstractmethod
    def integrate(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                 forces: Dict[int, np.ndarray], dt: float) -> None:
        """Perform one integration step.

        Args:
            particles: List of particles to integrate
            rigid_bodies: List of rigid bodies to integrate
            forces: Dictionary of forces acting on particles/bodies
            dt: Time step size
        """
        pass


class SymplecticEuler(Integrator):
    """Symplectic Euler integrator (semi-implicit Euler)."""

    def __init__(self):
        super().__init__("symplectic_euler")

    def integrate(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                 forces: Dict[int, np.ndarray], dt: float) -> None:
        # Update particles
        for particle in particles:
            if not particle.fixed:
                # Get total force on particle
                total_force = forces.get(particle.id, np.zeros(3))

                # Update velocity first (symplectic)
                particle.acceleration = total_force / particle.mass
                particle.velocity += particle.acceleration * dt

                # Then update position using new velocity
                particle.position += particle.velocity * dt


class VelocityVerlet(Integrator):
    """Velocity Verlet integrator."""

    def __init__(self):
        super().__init__("velocity_verlet")
        self.previous_accelerations = {}

    def integrate(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                 forces: Dict[int, np.ndarray], dt: float) -> None:
        # Update particles
        for particle in particles:
            if not particle.fixed:
                # Get total force on particle
                total_force = forces.get(particle.id, np.zeros(3))
                new_acceleration = total_force / particle.mass

                # Get previous acceleration
                prev_accel = self.previous_accelerations.get(particle.id, new_acceleration)

                # Update position using current velocity and acceleration
                particle.position += particle.velocity * dt + 0.5 * prev_accel * dt**2

                # Update velocity using average of old and new acceleration
                particle.velocity += 0.5 * (prev_accel + new_acceleration) * dt

                # Store acceleration for next step
                particle.acceleration = new_acceleration
                self.previous_accelerations[particle.id] = new_acceleration


class ForestRuth(Integrator):
    """Forest-Ruth symplectic integrator (4th order)."""

    def __init__(self):
        super().__init__("forest_ruth")
        self.theta = 1.0 / (2.0 - 2**(1.0/3.0))
        self.coeffs = [
            self.theta / 2.0,
            (1.0 - self.theta) / 2.0,
            (1.0 - self.theta) / 2.0,
            self.theta / 2.0
        ]

    def integrate(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                 forces: Dict[int, np.ndarray], dt: float) -> None:
        # Forest-Ruth scheme with 4 substeps
        for i, coeff in enumerate(self.coeffs):
            sub_dt = coeff * dt

            for particle in particles:
                if not particle.fixed:
                    total_force = forces.get(particle.id, np.zeros(3))
                    acceleration = total_force / particle.mass

                    if i % 2 == 0:  # Position update steps
                        particle.position += particle.velocity * sub_dt
                    else:  # Velocity update steps
                        particle.velocity += acceleration * sub_dt
                        particle.acceleration = acceleration


class Yoshida4(Integrator):
    """4th order Yoshida symplectic integrator."""

    def __init__(self):
        super().__init__("yoshida4")
        w1 = 1.0 / (2.0 - 2**(1.0/3.0))
        w0 = 1.0 - 2.0 * w1
        self.c = [w1/2.0, (w0+w1)/2.0, (w0+w1)/2.0, w1/2.0]
        self.d = [w1, w0, w1, 0.0]

    def integrate(self, particles: List[Particle], rigid_bodies: List[RigidBody],
                 forces: Dict[int, np.ndarray], dt: float) -> None:
        for i in range(4):
            # Position update
            for particle in particles:
                if not particle.fixed:
                    particle.position += self.c[i] * particle.velocity * dt

            # Force calculation would be done here in a full implementation
            # For now, using the provided forces

            # Velocity update
            if i < 3:  # Skip last velocity update
                for particle in particles:
                    if not particle.fixed:
                        total_force = forces.get(particle.id, np.zeros(3))
                        acceleration = total_force / particle.mass
                        particle.velocity += self.d[i] * acceleration * dt
                        particle.acceleration = acceleration