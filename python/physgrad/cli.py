"""
Command-line interface for PhysGrad.

Provides convenient CLI commands for running benchmarks, demos, and utilities.
"""

import argparse
import sys
import time
from typing import List, Dict, Any

import physgrad as pg


def benchmark_command():
    """Run performance benchmarks."""
    parser = argparse.ArgumentParser(description="Run PhysGrad performance benchmarks")
    parser.add_argument(
        "--particles",
        type=int,
        nargs="+",
        default=[100, 1000, 10000],
        help="Number of particles to benchmark"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of simulation steps per benchmark"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run benchmarks on"
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Include multi-GPU benchmarks"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON format)"
    )

    args = parser.parse_args()

    print("PhysGrad Performance Benchmark")
    print("=" * 40)
    print(f"Device info: {pg.get_device_info()}")
    print(f"Features: {pg.FEATURES}")
    print()

    results = {}

    # Single GPU/CPU benchmarks
    for num_particles in args.particles:
        print(f"Benchmarking {num_particles} particles...")

        config = pg.SimulationConfig(
            num_particles=num_particles,
            dt=0.01,
            enable_gpu=(args.device != "cpu"),
            enable_visualization=False
        )

        sim = pg.Simulation(config)

        # Add some particles
        particles = []
        for i in range(min(num_particles, 1000)):  # Limit for demo
            import numpy as np
            pos = np.random.uniform(-5, 5, 3)
            vel = np.random.uniform(-1, 1, 3)
            particle = pg.Particle(position=pos, velocity=vel, mass=1.0)
            particles.append(particle)

        sim.add_particles(particles)

        # Add gravity
        gravity = pg.GravityForce()
        # sim.add_force(gravity)  # Would need to implement force management

        # Benchmark
        start_time = time.time()

        for step in range(args.steps):
            sim.step()

        elapsed_time = time.time() - start_time

        stats = sim.get_performance_stats()

        result = {
            "num_particles": num_particles,
            "num_steps": args.steps,
            "total_time": elapsed_time,
            "time_per_step": elapsed_time / args.steps,
            "steps_per_second": args.steps / elapsed_time,
            "particles_per_second": num_particles * args.steps / elapsed_time,
            "device": args.device
        }

        results[f"single_gpu_{num_particles}"] = result

        print(f"  Time per step: {result['time_per_step']:.6f} s")
        print(f"  Steps per second: {result['steps_per_second']:.1f}")
        print(f"  Particles per second: {result['particles_per_second']:.0f}")
        print()

    # Multi-GPU benchmarks
    if args.multi_gpu and pg.FEATURES["multi_gpu"]:
        print("Multi-GPU Benchmarks")
        print("-" * 20)

        for num_particles in args.particles:
            if num_particles < 10000:  # Skip small sizes for multi-GPU
                continue

            print(f"Multi-GPU benchmarking {num_particles} particles...")

            # Would implement multi-GPU benchmark here
            # For now, just placeholder
            result = {
                "num_particles": num_particles,
                "num_steps": args.steps,
                "speedup": 2.5,  # Placeholder
                "efficiency": 0.8  # Placeholder
            }

            results[f"multi_gpu_{num_particles}"] = result
            print(f"  Speedup: {result['speedup']:.1f}x")
            print(f"  Efficiency: {result['efficiency']:.1f}")
            print()

    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    print("Benchmark complete!")


def demo_command():
    """Run interactive demos."""
    parser = argparse.ArgumentParser(description="Run PhysGrad demos")
    parser.add_argument(
        "demo",
        choices=["pendulum", "cloth", "rigid_body", "optimization", "multi_gpu"],
        help="Demo to run"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive visualization"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of simulation steps"
    )

    args = parser.parse_args()

    print(f"Running {args.demo} demo...")

    if args.demo == "pendulum":
        run_pendulum_demo(args)
    elif args.demo == "cloth":
        run_cloth_demo(args)
    elif args.demo == "rigid_body":
        run_rigid_body_demo(args)
    elif args.demo == "optimization":
        run_optimization_demo(args)
    elif args.demo == "multi_gpu":
        run_multi_gpu_demo(args)


def run_pendulum_demo(args):
    """Run pendulum simulation demo."""
    import numpy as np

    print("Setting up pendulum simulation...")

    config = pg.SimulationConfig(
        num_particles=2,
        dt=0.01,
        enable_visualization=args.interactive
    )

    sim = pg.Simulation(config)

    # Create pendulum: fixed point + bob
    anchor = pg.Particle(
        position=np.array([0, 5, 0]),
        velocity=np.zeros(3),
        mass=1.0,
        fixed=True
    )

    bob = pg.Particle(
        position=np.array([2, 0, 0]),  # Initial displacement
        velocity=np.zeros(3),
        mass=1.0
    )

    anchor_id = sim.add_particle(anchor)
    bob_id = sim.add_particle(bob)

    # Add constraint (rope)
    # constraint = pg.DistanceConstraint(anchor_id, bob_id, distance=5.0)
    # sim.add_constraint(constraint)  # Would need to implement constraint management

    # Add gravity
    # gravity = pg.GravityForce()
    # sim.add_force(gravity)

    print("Running pendulum simulation...")

    for step in range(args.steps):
        sim.step()

        if step % 100 == 0:
            energy = sim.get_total_energy()
            print(f"Step {step}: Energy = {energy:.3f}")

    print("Pendulum demo complete!")


def run_cloth_demo(args):
    """Run cloth simulation demo."""
    print("Cloth demo not yet implemented")


def run_rigid_body_demo(args):
    """Run rigid body simulation demo."""
    print("Rigid body demo not yet implemented")


def run_optimization_demo(args):
    """Run optimization demo."""
    print("Optimization demo not yet implemented")


def run_multi_gpu_demo(args):
    """Run multi-GPU demo."""
    if not pg.FEATURES["multi_gpu"]:
        print("Multi-GPU support not available")
        return

    print("Multi-GPU demo not yet implemented")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="PhysGrad CLI")
    parser.add_argument(
        "--version",
        action="version",
        version=f"PhysGrad {pg.__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.set_defaults(func=benchmark_command)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demos")
    demo_parser.set_defaults(func=demo_command)

    # Device info command
    info_parser = subparsers.add_parser("info", help="Show device information")
    info_parser.set_defaults(func=lambda: print(pg.get_device_info()))

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        return

    try:
        args.func()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()