#!/usr/bin/env python3
"""
Demo script showing the documented examples working with the high-level API.
"""

import sys
sys.path.insert(0, './python')

import physgrad as pg
import numpy as np

def documented_example_1():
    """Example 1: Basic particle simulation (from README)."""
    print("🚀 Running Documented Example 1: Basic Particle Simulation")
    print("=" * 60)

    # Create a simple particle simulation
    config = pg.SimulationConfig(
        num_particles=1000,
        dt=0.01,
        enable_gpu=True
    )

    sim = pg.Simulation(config)

    # Add particles
    particles = []
    for i in range(100):
        pos = np.random.uniform(-5, 5, 3)
        vel = np.random.uniform(-1, 1, 3)
        particle = pg.Particle(position=pos, velocity=vel, mass=1.0)
        particles.append(particle)

    particle_ids = sim.add_particles(particles)

    # Add gravity
    gravity = pg.GravityForce(gravity=[0, -9.81, 0])
    sim.add_force(gravity)

    # Run simulation
    for step in range(1000):
        sim.step()

        if step % 100 == 0:
            energy = sim.get_total_energy()
            print(f"Step {step}: Total energy = {energy:.3f}")

    print("✅ Example 1 completed successfully!")


def documented_example_2():
    """Example 2: Quick simulation function."""
    print("\n🚀 Running Documented Example 2: Quick Simulation")
    print("=" * 60)

    # Using the quick_simulation function
    sim = pg.quick_simulation(num_particles=100)

    # Run simulation
    sim.run(1000)

    # Get performance stats
    stats = sim.get_performance_stats()
    print(f"✅ Quick simulation: {stats['steps_per_second']:.0f} steps/second")


def documented_example_3():
    """Example 3: Force and constraint system."""
    print("\n🚀 Running Documented Example 3: Forces and Constraints")
    print("=" * 60)

    # Create simulation
    config = pg.SimulationConfig(
        num_particles=10,
        dt=0.01,
        enable_gpu=False  # Use CPU for this example
    )
    sim = pg.Simulation(config)

    # Create a pendulum using constraints
    anchor = pg.Particle(position=[0, 5, 0], mass=1.0, fixed=True)
    bob = pg.Particle(position=[0, 3, 0], mass=1.0)

    anchor_id = sim.add_particle(anchor)
    bob_id = sim.add_particle(bob)

    # Add gravity
    gravity = pg.GravityForce(gravity=[0, -9.81, 0])
    sim.add_force(gravity)

    # Add pendulum constraint (distance constraint)
    pendulum_constraint = pg.DistanceConstraint(
        anchor_id, bob_id, distance=2.0, stiffness=10000.0
    )
    sim.add_constraint(pendulum_constraint)

    # Add damping
    damping = pg.DampingForce(damping_coefficient=0.1)
    sim.add_force(damping)

    print("Initial pendulum bob position:", sim.particles[bob_id].position)

    # Run simulation
    for step in range(500):
        sim.step()

        if step % 100 == 0:
            pos = sim.particles[bob_id].position
            print(f"Step {step}: Bob position = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    print("✅ Example 3 completed successfully!")


def documented_example_4():
    """Example 4: Device information and features."""
    print("\n🚀 Running Documented Example 4: Device Information")
    print("=" * 60)

    # Get device information
    device_info = pg.get_device_info()
    print("Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")

    print("\nAvailable Features:")
    for feature, available in pg.FEATURES.items():
        status = "✅" if available else "❌"
        print(f"  {status} {feature}: {available}")

    print("✅ Example 4 completed successfully!")


def main():
    """Run all documented examples."""
    print("🎉 PhysGrad Documented Examples Demo")
    print("=" * 60)
    print("This script demonstrates that the documented examples")
    print("from the README actually work with our implementation!")
    print()

    try:
        documented_example_1()
        documented_example_2()
        documented_example_3()
        documented_example_4()

        print("\n" + "=" * 60)
        print("🎉 ALL DOCUMENTED EXAMPLES WORKING CORRECTLY!")
        print("PhysGrad high-level API is ready for users!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())