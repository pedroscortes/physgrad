"""
Example 1: Basic PyTorch Gradient Computation

This example demonstrates the fundamental capability of PhysGrad: computing
gradients through physics simulation using PyTorch.

Learning objectives:
1. Create a simple spring-mass system
2. Run differentiable physics simulation
3. Compute loss from final state
4. Backpropagate through simulation to get gradients
5. Optimize initial conditions using gradient descent

Use case:
- Trajectory optimization
- Inverse kinematics
- Finding initial conditions that achieve a target
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from physgrad.adjoint import AdjointPhysics, SpringMassSystem


def example_1_simple_optimization():
    """
    Example 1.1: Simple Trajectory Optimization

    Goal: Find initial positions that minimize final displacement.
    """
    print("=" * 70)
    print("Example 1.1: Simple Trajectory Optimization")
    print("=" * 70)
    print("\nGoal: Minimize final displacement from origin\n")

    # Create physics system: 2 particles connected by a spring
    system = SpringMassSystem(n_particles=2, dtype='float32')
    system.add_spring(
        i=0, j=1,
        stiffness=10.0,
        rest_length=1.0
    )

    print(f"System: {system}")

    # Learnable parameters: initial positions
    initial_positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0]  # Stretched beyond rest length
    ], dtype=torch.float32, requires_grad=True)

    # Fixed parameters
    initial_velocities = torch.zeros(2, 3, dtype=torch.float32)
    masses = torch.ones(2, dtype=torch.float32)

    # Create differentiable physics simulator
    physics = AdjointPhysics(system, dt=0.01, num_steps=50)

    # Optimizer
    optimizer = torch.optim.Adam([initial_positions], lr=0.05)

    # Training loop
    n_iterations = 30
    losses = []

    print(f"\nRunning {n_iterations} optimization iterations...")
    print(f"Initial positions: {initial_positions.detach().numpy()}\n")

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Forward simulation (differentiable!)
        final_pos, final_vel = physics(
            initial_positions,
            initial_velocities,
            masses
        )

        # Loss: minimize final displacement
        loss = (final_pos ** 2).sum()

        # Backward pass through physics
        loss.backward()

        # Gradient descent step
        optimizer.step()

        losses.append(loss.item())

        if iteration % 5 == 0:
            grad_norm = torch.norm(initial_positions.grad).item()
            print(f"  Iteration {iteration:2d}: Loss = {loss.item():.6f}, "
                  f"Grad norm = {grad_norm:.6f}")

    print(f"\nOptimization complete!")
    print(f"Final positions: {initial_positions.detach().numpy()}")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Improvement: {100 * (losses[0] - losses[-1]) / losses[0]:.1f}%")

    return losses


def example_2_target_tracking():
    """
    Example 1.2: Target Tracking

    Goal: Find initial velocities that make particle reach a target.
    """
    print("\n" + "=" * 70)
    print("Example 1.2: Target Tracking")
    print("=" * 70)
    print("\nGoal: Find initial velocities to reach target position\n")

    # System setup
    system = SpringMassSystem(n_particles=2, dtype='float32')
    system.add_spring(0, 1, stiffness=5.0, rest_length=1.0)

    # Fixed initial positions
    initial_positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=torch.float32)

    # Learnable: initial velocities
    initial_velocities = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=torch.float32, requires_grad=True)

    masses = torch.ones(2, dtype=torch.float32)

    # Target: particle 1 at position [2.0, 0, 0]
    target_position = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)

    # Physics simulator
    physics = AdjointPhysics(system, dt=0.01, num_steps=50)

    # Optimizer
    optimizer = torch.optim.Adam([initial_velocities], lr=0.1)

    # Training
    n_iterations = 50
    losses = []

    print(f"Target: {target_position.numpy()}")
    print(f"Running {n_iterations} optimization iterations...\n")

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Simulate
        final_pos, _ = physics(initial_positions, initial_velocities, masses)

        # Loss: distance from target
        loss = ((final_pos[1] - target_position) ** 2).sum()

        # Backprop
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iteration % 10 == 0:
            final_pos_np = final_pos[1].detach().numpy()
            print(f"  Iteration {iteration:2d}: Loss = {loss.item():.6f}, "
                  f"Position = [{final_pos_np[0]:.3f}, {final_pos_np[1]:.3f}, {final_pos_np[2]:.3f}]")

    # Final result
    with torch.no_grad():
        final_pos, _ = physics(initial_positions, initial_velocities, masses)
        error = torch.norm(final_pos[1] - target_position).item()

    print(f"\nOptimization complete!")
    print(f"Optimized velocities: {initial_velocities.detach().numpy()}")
    print(f"Final position: {final_pos[1].detach().numpy()}")
    print(f"Target position: {target_position.numpy()}")
    print(f"Error: {error:.6f}")

    return losses


def example_3_parameter_optimization():
    """
    Example 1.3: Material Parameter Optimization

    Goal: Optimize spring constants to achieve desired behavior.
    """
    print("\n" + "=" * 70)
    print("Example 1.3: Material Parameter Optimization")
    print("=" * 70)
    print("\nGoal: Optimize spring stiffness for desired behavior\n")

    # Create system with learnable spring constant
    system = SpringMassSystem(n_particles=2, dtype='float32')

    # Initial spring constant (will be optimized)
    initial_k = 5.0
    system.add_spring(0, 1, stiffness=initial_k, rest_length=1.0)

    # Fixed simulation setup
    initial_positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0]
    ], dtype=torch.float32)

    initial_velocities = torch.zeros(2, 3, dtype=torch.float32)
    masses = torch.ones(2, dtype=torch.float32)

    # Target: minimize final kinetic energy (want oscillation to dampen)
    def loss_fn(pos, vel):
        """Loss: final kinetic energy (want it minimized)"""
        ke = 0.5 * (vel ** 2).sum()
        return ke

    physics = AdjointPhysics(system, dt=0.01, num_steps=100)

    # Use compute_all_gradients to get parameter gradients
    print("Computing gradients w.r.t. spring constant...")

    all_grads = physics.compute_all_gradients(
        initial_positions,
        initial_velocities,
        masses,
        loss_fn
    )

    print(f"\nGradient w.r.t. spring constant: {all_grads['spring_constant_grads'][0]:.6f}")
    print(f"Gradient w.r.t. rest length: {all_grads['rest_length_grads'][0]:.6f}")
    print(f"Gradient w.r.t. initial position[0]: {all_grads['position_grads'][0]}")
    print(f"Gradient w.r.t. initial position[1]: {all_grads['position_grads'][1]}")

    # Demonstrate optimization
    learning_rate = 0.01
    optimized_k = initial_k - learning_rate * all_grads['spring_constant_grads'][0]

    print(f"\nOptimization step:")
    print(f"  Initial k = {initial_k:.3f}")
    print(f"  Optimized k = {optimized_k:.3f}")
    print(f"  Change = {optimized_k - initial_k:.3f}")

    print("\nNote: This demonstrates parameter gradient computation.")
    print("For full optimization loop, rebuild system with new parameters each iteration.")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PhysGrad PyTorch Integration - Basic Examples")
    print("=" * 70)
    print("\nThese examples demonstrate gradient computation through physics")
    print("using the adjoint method integrated with PyTorch.\n")

    try:
        # Run examples
        losses1 = example_1_simple_optimization()
        losses2 = example_2_target_tracking()
        example_3_parameter_optimization()

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(losses1, 'b-', linewidth=2, label='Simple Optimization')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Example 1.1: Minimize Displacement')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(losses2, 'r-', linewidth=2, label='Target Tracking')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Example 1.2: Reach Target')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_yscale('log')

        plt.tight_layout()
        output_path = '/tmp/pytorch_basic_examples.png'
        plt.savefig(output_path, dpi=150)
        print(f"\n{'='*70}")
        print(f"Plot saved to: {output_path}")
        print(f"{'='*70}\n")

        print("✓ All examples completed successfully!")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
