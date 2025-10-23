"""
Example: Training with PyTorch Adjoint Integration

This example demonstrates how to use the adjoint-based differentiable
physics simulator with PyTorch for end-to-end training.

Use cases:
1. Trajectory optimization: Find initial conditions that achieve a target
2. Inverse kinematics: Optimize parameters to reach a goal
3. Neural network + physics co-training: Learn control policies
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("ERROR: PyTorch not available. Install with: pip install torch")
    exit(1)

try:
    import sys
    import os
    # Add parent directory to path to import physgrad
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from physgrad.adjoint_integration import AdjointPhysicsSimulator
except ImportError as e:
    print(f"ERROR: Could not import physgrad: {e}")
    print("Make sure you have built and installed the Python package")
    exit(1)


def example_1_trajectory_optimization():
    """
    Example 1: Trajectory Optimization

    Goal: Find initial positions that minimize final displacement
    for a simple spring-mass system.
    """
    print("=" * 70)
    print("Example 1: Trajectory Optimization")
    print("=" * 70)
    print("\nGoal: Optimize initial positions to minimize final displacement\n")

    # Setup: 3-particle chain connected by springs
    n_particles = 3

    # Learnable parameters: initial positions
    initial_positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ], dtype=torch.float32, requires_grad=True)

    # Fixed parameters
    initial_velocities = torch.zeros(n_particles, 3, dtype=torch.float32)
    masses = torch.ones(n_particles, dtype=torch.float32)

    # Spring connectivity: 0-1, 1-2
    spring_pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    spring_constants = torch.tensor([10.0, 10.0], dtype=torch.float32)
    rest_lengths = torch.tensor([1.0, 1.0], dtype=torch.float32)

    # Simulation parameters
    dt = 0.01
    num_steps = 20

    # Create simulator
    simulator = AdjointPhysicsSimulator(dt=dt, num_steps=num_steps)

    # Optimizer
    optimizer = optim.Adam([initial_positions], lr=0.05)

    # Training loop
    n_iterations = 50
    losses = []

    print(f"Running {n_iterations} optimization iterations...")
    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Forward simulation
        final_positions, final_velocities = simulator(
            initial_positions, initial_velocities, masses,
            spring_pairs, spring_constants, rest_lengths
        )

        # Loss: minimize final displacement from origin
        loss = (final_positions ** 2).sum()

        # Backward pass
        loss.backward()

        # Optimization step
        optimizer.step()

        losses.append(loss.item())

        if iteration % 10 == 0:
            print(f"  Iteration {iteration:3d}: Loss = {loss.item():.6f}, "
                  f"Grad norm = {torch.norm(initial_positions.grad).item():.6f}")

    print(f"\nOptimization complete!")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Improvement: {100 * (losses[0] - losses[-1]) / losses[0]:.2f}%")

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Progress (Log Scale)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('/tmp/trajectory_optimization.png', dpi=150)
    print(f"\nPlot saved to: /tmp/trajectory_optimization.png")

    return initial_positions.detach(), losses


def example_2_inverse_design():
    """
    Example 2: Inverse Design

    Goal: Find spring constants that achieve a target final configuration.
    """
    print("\n" + "=" * 70)
    print("Example 2: Inverse Design (Spring Constants)")
    print("=" * 70)
    print("\nGoal: Optimize spring constants to reach target configuration\n")

    # Setup: 2-particle system
    initial_positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=torch.float32)

    initial_velocities = torch.zeros(2, 3, dtype=torch.float32)
    masses = torch.ones(2, dtype=torch.float32)

    # Learnable parameters: spring constants
    # Note: Current implementation doesn't support gradients w.r.t. spring constants
    # This is documented in TECHNICAL_DEBT.md as issue #2
    # For now, we optimize initial positions instead

    # Target: particle 1 at position [2.0, 0, 0]
    target_position = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)

    # Learnable initial position
    initial_pos_learnable = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=torch.float32, requires_grad=True)

    spring_pairs = torch.tensor([[0, 1]], dtype=torch.int32)
    spring_constants = torch.tensor([5.0], dtype=torch.float32)
    rest_lengths = torch.tensor([1.0], dtype=torch.float32)

    simulator = AdjointPhysicsSimulator(dt=0.01, num_steps=20)
    optimizer = optim.Adam([initial_pos_learnable], lr=0.02)

    n_iterations = 30
    losses = []

    print(f"Target: particle 1 at {target_position.numpy()}")
    print(f"Running {n_iterations} optimization iterations...\n")

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        final_pos, _ = simulator(
            initial_pos_learnable, initial_velocities, masses,
            spring_pairs, spring_constants, rest_lengths
        )

        # Loss: distance from target
        loss = ((final_pos[1] - target_position) ** 2).sum()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iteration % 5 == 0:
            print(f"  Iteration {iteration:3d}: Loss = {loss.item():.6f}, "
                  f"Final pos = [{final_pos[1, 0].item():.3f}, "
                  f"{final_pos[1, 1].item():.3f}, "
                  f"{final_pos[1, 2].item():.3f}]")

    print(f"\nOptimization complete!")
    print(f"  Target distance: 0.000")
    print(f"  Final distance: {np.sqrt(losses[-1]):.6f}")

    return losses


def example_3_neural_network_control():
    """
    Example 3: Neural Network Control Policy

    Goal: Train a neural network to output initial velocities that
    achieve a target final state.
    """
    print("\n" + "=" * 70)
    print("Example 3: Neural Network Control Policy")
    print("=" * 70)
    print("\nGoal: Train NN to predict initial velocities for target states\n")

    class ControlPolicy(nn.Module):
        """Simple MLP that predicts initial velocities from target positions."""
        def __init__(self, input_dim=3, hidden_dim=16, output_dim=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh()  # Bounded velocities
            )

        def forward(self, target):
            return self.net(target) * 2.0  # Scale to [-2, 2] range

    # Setup
    policy = ControlPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    # Fixed system
    initial_positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=torch.float32)

    masses = torch.ones(2, dtype=torch.float32)
    spring_pairs = torch.tensor([[0, 1]], dtype=torch.int32)
    spring_constants = torch.tensor([10.0], dtype=torch.float32)
    rest_lengths = torch.tensor([1.0], dtype=torch.float32)

    simulator = AdjointPhysicsSimulator(dt=0.01, num_steps=15)

    n_iterations = 50
    losses = []

    print(f"Training neural network control policy...")
    print(f"Network parameters: {sum(p.numel() for p in policy.parameters())}\n")

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Sample random target for particle 1
        target = torch.randn(3) * 0.5 + torch.tensor([1.5, 0.0, 0.0])

        # Predict initial velocity using policy
        predicted_velocity = policy(target)

        # Setup initial velocities (particle 0 stays fixed)
        initial_vels = torch.zeros(2, 3, dtype=torch.float32)
        initial_vels[1] = predicted_velocity

        # Simulate
        final_pos, _ = simulator(
            initial_positions, initial_vels, masses,
            spring_pairs, spring_constants, rest_lengths
        )

        # Loss: reach target
        loss = ((final_pos[1] - target) ** 2).sum()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iteration % 10 == 0:
            print(f"  Iteration {iteration:3d}: Loss = {loss.item():.6f}")

    print(f"\nTraining complete!")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")

    # Test the policy
    print(f"\nTesting trained policy:")
    test_target = torch.tensor([2.0, 0.0, 0.0])
    with torch.no_grad():
        predicted_vel = policy(test_target)
        initial_vels_test = torch.zeros(2, 3)
        initial_vels_test[1] = predicted_vel

        final_test, _ = simulator(
            initial_positions, initial_vels_test, masses,
            spring_pairs, spring_constants, rest_lengths
        )

        error = torch.norm(final_test[1] - test_target).item()
        print(f"  Target: {test_target.numpy()}")
        print(f"  Achieved: {final_test[1].numpy()}")
        print(f"  Error: {error:.6f}")

    return policy, losses


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PyTorch Adjoint Integration Examples")
    print("=" * 70)
    print("\nThese examples demonstrate differentiable physics with PyTorch")
    print("using the adjoint method for gradient computation.")
    print("\n")

    # Run examples
    try:
        initial_pos, losses1 = example_1_trajectory_optimization()
        losses2 = example_2_inverse_design()
        policy, losses3 = example_3_neural_network_control()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
