#!/usr/bin/env python3
"""
Real-time OpenGL/ImGui visualization demo for PhysGrad.

This demo showcases the real-time visualization capabilities with:
- 3D particle rendering with OpenGL shaders
- Interactive ImGui controls for physics parameters
- Real-time energy monitoring
- Mathematical overlays and physics debugging
- Force vector visualization
- Particle trails and velocity-based coloring
"""

import sys
import numpy as np
import time

# Add PhysGrad to path
sys.path.insert(0, './python')

def demo_bouncing_balls():
    """Demo: Bouncing balls with gravity and collision."""
    print("🎬 Real-time Visualization Demo: Bouncing Balls")
    print("=" * 60)

    try:
        import physgrad as pg

        # Create simulation with moderate number of particles
        config = pg.SimulationConfig(
            num_particles=20,
            dt=0.01,
            enable_gpu=True,
            domain_size=[10.0, 10.0, 10.0],
            enable_collisions=True
        )

        sim = pg.Simulation(config)

        # Add bouncing balls at different heights
        particles = []
        for i in range(20):
            # Random positions in upper half of domain
            pos = np.array([
                np.random.uniform(-4, 4),
                np.random.uniform(2, 8),
                np.random.uniform(-4, 4)
            ])

            # Small random velocities
            vel = np.array([
                np.random.uniform(-1, 1),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-1, 1)
            ])

            # Varying masses for interesting dynamics
            mass = np.random.uniform(0.5, 2.0)

            particle = pg.Particle(position=pos, velocity=vel, mass=mass)
            particles.append(particle)

        particle_ids = sim.add_particles(particles)
        print(f"✅ Added {len(particle_ids)} particles")

        # Add gravity
        gravity = pg.GravityForce(gravity=[0, -9.81, 0])
        sim.add_force(gravity)

        # Add air damping to prevent infinite bouncing
        damping = pg.DampingForce(damping_coefficient=0.05)
        sim.add_force(damping)

        print("✅ Added physics forces")

        # Enable real-time visualization
        print("🖥️  Initializing real-time OpenGL/ImGui visualization...")
        if sim.enable_visualization(1440, 900):
            print("✅ Visualization initialized successfully!")
            print("\n" + "=" * 60)
            print("🎮 CONTROLS:")
            print("  • Mouse drag: Rotate camera")
            print("  • Mouse wheel: Zoom in/out")
            print("  • WASD/QE: Navigate camera")
            print("  • ImGui panels: Adjust physics parameters")
            print("  • Close window or ESC: Exit")
            print("=" * 60)

            # Run simulation with real-time visualization
            sim.run_with_visualization(max_steps=10000)

        else:
            print("❌ Failed to initialize visualization")
            print("   Running simulation without visualization...")
            sim.run(1000)

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def demo_pendulum_system():
    """Demo: Pendulum system with constraints."""
    print("\n🎬 Real-time Visualization Demo: Pendulum System")
    print("=" * 60)

    try:
        import physgrad as pg

        # Create simulation for pendulum
        config = pg.SimulationConfig(
            num_particles=10,
            dt=0.005,
            enable_gpu=False,  # Use CPU for better constraint stability
            domain_size=[6.0, 6.0, 6.0],
            enable_constraints=True
        )

        sim = pg.Simulation(config)

        # Create multiple pendulums
        pendulum_length = 2.0
        num_pendulums = 5

        for i in range(num_pendulums):
            x_offset = (i - 2) * 1.0  # Spread pendulums horizontally

            # Fixed anchor point
            anchor = pg.Particle(
                position=[x_offset, 3.0, 0.0],
                mass=1.0,
                fixed=True
            )

            # Pendulum bob with initial displacement
            bob = pg.Particle(
                position=[x_offset + 1.0, 1.0, 0.0],  # Start displaced
                velocity=[0.0, 0.0, 0.0],
                mass=1.0 + i * 0.2  # Varying masses
            )

            anchor_id = sim.add_particle(anchor)
            bob_id = sim.add_particle(bob)

            # Distance constraint (pendulum rod)
            constraint = pg.DistanceConstraint(
                anchor_id, bob_id,
                distance=pendulum_length,
                stiffness=10000.0
            )
            sim.add_constraint(constraint)

        print(f"✅ Created {num_pendulums} pendulums")

        # Add gravity
        gravity = pg.GravityForce(gravity=[0, -9.81, 0])
        sim.add_force(gravity)

        # Add light damping for realism
        damping = pg.DampingForce(damping_coefficient=0.02)
        sim.add_force(damping)

        print("✅ Added physics forces and constraints")

        # Enable visualization
        if sim.enable_visualization(1440, 900):
            print("✅ Visualization initialized!")
            print("\n🔗 Watch the pendulum motion and energy conservation!")

            # Run with visualization
            sim.run_with_visualization(max_steps=5000)
        else:
            print("❌ Visualization failed, running without graphics")
            sim.run(1000)

    except Exception as e:
        print(f"❌ Pendulum demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def demo_particle_fountain():
    """Demo: Particle fountain with continuous spawning."""
    print("\n🎬 Real-time Visualization Demo: Particle Fountain")
    print("=" * 60)

    try:
        import physgrad as pg

        # Create simulation
        config = pg.SimulationConfig(
            num_particles=100,
            dt=0.01,
            enable_gpu=True,
            domain_size=[8.0, 10.0, 8.0]
        )

        sim = pg.Simulation(config)

        # Start with initial fountain particles
        fountain_particles = []
        for i in range(30):
            # Spawn from center bottom
            pos = np.array([
                np.random.uniform(-0.5, 0.5),
                0.1,
                np.random.uniform(-0.5, 0.5)
            ])

            # Upward velocities with spread
            vel = np.array([
                np.random.uniform(-2, 2),
                np.random.uniform(8, 12),
                np.random.uniform(-2, 2)
            ])

            mass = np.random.uniform(0.3, 1.0)
            particle = pg.Particle(position=pos, velocity=vel, mass=mass)
            fountain_particles.append(particle)

        sim.add_particles(fountain_particles)

        # Add gravity
        gravity = pg.GravityForce(gravity=[0, -9.81, 0])
        sim.add_force(gravity)

        # Add air resistance
        damping = pg.DampingForce(damping_coefficient=0.1)
        sim.add_force(damping)

        print("✅ Created particle fountain")

        # Enable visualization
        if sim.enable_visualization(1440, 900):
            print("✅ Visualization ready!")
            print("\n⛲ Watch the particle fountain dynamics!")
            print("   Use ImGui controls to adjust gravity and damping!")

            # Custom loop for particle spawning
            step = 0
            spawn_interval = 10  # Spawn new particles every 10 steps

            while not sim.should_close_visualization():
                # Spawn new particles periodically
                if step % spawn_interval == 0 and len(sim.particles) < 80:
                    new_pos = np.array([
                        np.random.uniform(-0.3, 0.3),
                        0.1,
                        np.random.uniform(-0.3, 0.3)
                    ])

                    new_vel = np.array([
                        np.random.uniform(-1.5, 1.5),
                        np.random.uniform(10, 14),
                        np.random.uniform(-1.5, 1.5)
                    ])

                    new_particle = pg.Particle(
                        position=new_pos,
                        velocity=new_vel,
                        mass=np.random.uniform(0.5, 1.2)
                    )
                    sim.add_particle(new_particle)

                # Remove particles that fall too low
                particles_to_remove = []
                for i, particle in enumerate(sim.particles):
                    if particle.position[1] < -5.0:
                        particles_to_remove.append(i)

                # Remove from back to front to maintain indices
                for i in reversed(particles_to_remove):
                    if i < len(sim.particles):
                        sim.particles.pop(i)

                # Step simulation
                sim.step()
                sim.update_visualization()
                sim.render_visualization()

                step += 1

            sim.disable_visualization()

        else:
            print("❌ Visualization failed")
            return False

    except Exception as e:
        print(f"❌ Fountain demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Run all real-time visualization demos."""
    print("🎉 PhysGrad Real-time Visualization Demo Suite")
    print("=" * 60)
    print("This demo showcases OpenGL/ImGui real-time visualization:")
    print("• 3D particle rendering with velocity-based colors")
    print("• Interactive physics parameter controls")
    print("• Real-time energy monitoring and mathematical info")
    print("• Force vector visualization")
    print("• Particle trails and collision detection")
    print("• Professional-grade physics debugging tools")
    print()

    demos = [
        ("Bouncing Balls", demo_bouncing_balls),
        ("Pendulum System", demo_pendulum_system),
        ("Particle Fountain", demo_particle_fountain),
    ]

    for demo_name, demo_func in demos:
        print(f"\n🚀 Starting {demo_name} demo...")
        try:
            success = demo_func()
            if success:
                print(f"✅ {demo_name} demo completed successfully!")
            else:
                print(f"⚠️  {demo_name} demo encountered issues")
        except KeyboardInterrupt:
            print(f"\n⏹️  {demo_name} demo interrupted by user")
            break
        except Exception as e:
            print(f"❌ {demo_name} demo crashed: {e}")

        # Small delay between demos
        time.sleep(1)

    print("\n" + "=" * 60)
    print("🎉 All real-time visualization demos completed!")
    print("PhysGrad real-time visualization system is ready for research!")


if __name__ == "__main__":
    main()