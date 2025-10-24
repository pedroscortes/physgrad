"""
Example 3: Neural Network + Physics Co-Design

This example demonstrates end-to-end training where a neural network and
physics simulation work together to achieve a goal.

Concept:
- Neural network generates control signals or initial conditions
- Physics simulator runs forward to produce final state
- Loss computed on final state
- Gradients backpropagate through physics to train the network

Use cases:
- Robot control policy learning
- Trajectory planning with learned controllers
- Material design (network suggests materials, physics evaluates)
- Inverse design problems
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from physgrad.adjoint import AdjointPhysics, SpringMassSystem


class ControllerNetwork(nn.Module):
    """
    Neural network that learns to control a physical system.

    Input: Target state (where we want to go)
    Output: Control signal (initial velocity)

    The network learns the mapping: target → initial_velocity
    by training end-to-end through the physics simulator.
    """

    def __init__(self, input_dim=3, hidden_dim=32, output_dim=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Bounded output
        )

        self.velocity_scale = 2.0  # Max velocity magnitude

    def forward(self, target_position):
        """
        Predict initial velocity to reach target.

        Args:
            target_position: (batch_size, 3) or (3,)

        Returns:
            initial_velocity: (batch_size, 3) or (3,)
        """
        return self.net(target_position) * self.velocity_scale


class StatePredictor(nn.Module):
    """
    Neural network that predicts initial conditions for a desired outcome.

    Input: Desired final state
    Output: Initial positions and velocities

    This is trained through the physics simulator using inverse design.
    """

    def __init__(self, output_dim=6, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),   # Final position desired
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  # [pos(3), vel(3)]
        )

    def forward(self, desired_final_position):
        """
        Predict initial state that leads to desired final position.

        Args:
            desired_final_position: (3,) target for particle 1

        Returns:
            initial_position: (3,)
            initial_velocity: (3,)
        """
        output = self.net(desired_final_position)
        initial_position = output[:3]
        initial_velocity = output[3:]
        return initial_position, initial_velocity


def example_1_control_policy():
    """
    Example 3.1: Learn Control Policy

    Train a neural network to control a spring-mass system.
    Network learns to output velocities that achieve target positions.
    """
    print("=" * 70)
    print("Example 3.1: Neural Control Policy Learning")
    print("=" * 70)
    print("\nGoal: Train NN to control spring-mass system\n")

    # Physics system
    system = SpringMassSystem(n_particles=2, dtype='float32')
    system.add_spring(0, 1, stiffness=10.0, rest_length=1.0)

    physics = AdjointPhysics(system, dt=0.01, num_steps=30)

    # Controller network
    controller = ControllerNetwork(input_dim=3, hidden_dim=32, output_dim=3)

    print(f"Controller network: {sum(p.numel() for p in controller.parameters())} parameters")
    print(f"Physics system: {system}\n")

    # Optimizer
    optimizer = optim.Adam(controller.parameters(), lr=0.01)

    # Fixed initial configuration
    fixed_positions = torch.tensor([
        [0.0, 0.0, 0.0],  # Particle 0 stays fixed
        [1.0, 0.0, 0.0]   # Particle 1 will be controlled
    ], dtype=torch.float32)

    masses = torch.ones(2, dtype=torch.float32)

    # Training loop
    n_iterations = 100
    losses = []
    errors = []

    print(f"Training for {n_iterations} iterations...")

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Sample random target
        target = torch.randn(3) * 0.3 + torch.tensor([1.5, 0.0, 0.0])

        # Network predicts initial velocity
        predicted_velocity = controller(target)

        # Setup initial velocities (particle 0 is fixed)
        initial_velocities = torch.zeros(2, 3, dtype=torch.float32)
        initial_velocities[1] = predicted_velocity

        # Simulate with predicted velocity
        final_pos, _ = physics(fixed_positions, initial_velocities, masses)

        # Loss: reach target
        loss = ((final_pos[1] - target) ** 2).sum()

        # Backpropagate through physics
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Track error
        with torch.no_grad():
            error = torch.norm(final_pos[1] - target).item()
            errors.append(error)

        if iteration % 20 == 0:
            print(f"  Iteration {iteration:3d}: Loss = {loss.item():.6f}, "
                  f"Error = {error:.6f}")

    print(f"\nTraining complete!")
    print(f"Initial error: {errors[0]:.6f}")
    print(f"Final error: {errors[-1]:.6f}")
    print(f"Improvement: {100 * (errors[0] - errors[-1]) / errors[0]:.1f}%")

    # Test the trained controller
    print(f"\n{'='*70}")
    print("Testing trained controller:")
    print(f"{'='*70}\n")

    test_targets = [
        torch.tensor([1.5, 0.0, 0.0]),
        torch.tensor([2.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.2, 0.0]),
    ]

    with torch.no_grad():
        for i, target in enumerate(test_targets):
            # Network predicts control
            vel = controller(target)

            # Simulate
            initial_vels = torch.zeros(2, 3)
            initial_vels[1] = vel

            final_pos, _ = physics(fixed_positions, initial_vels, masses)

            error = torch.norm(final_pos[1] - target).item()

            print(f"Test {i+1}:")
            print(f"  Target:   {target.numpy()}")
            print(f"  Achieved: {final_pos[1].numpy()}")
            print(f"  Error:    {error:.6f}")
            print()

    return losses, errors


def example_2_inverse_design():
    """
    Example 3.2: Inverse Design

    Given desired final state, learn to predict initial conditions.
    """
    print("=" * 70)
    print("Example 3.2: Inverse Design with Neural Network")
    print("=" * 70)
    print("\nGoal: Predict initial state that leads to desired outcome\n")

    # Physics
    system = SpringMassSystem(n_particles=2, dtype='float32')
    system.add_spring(0, 1, stiffness=8.0, rest_length=1.0)

    physics = AdjointPhysics(system, dt=0.01, num_steps=40)

    # Predictor network
    predictor = StatePredictor(output_dim=6, hidden_dim=64)

    print(f"Predictor network: {sum(p.numel() for p in predictor.parameters())} parameters\n")

    # Optimizer
    optimizer = optim.Adam(predictor.parameters(), lr=0.01)

    # Training
    n_iterations = 150
    losses = []

    print(f"Training for {n_iterations} iterations...")

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Random desired final position
        desired_final = torch.randn(3) * 0.4 + torch.tensor([1.2, 0.0, 0.0])

        # Network predicts initial state
        pred_init_pos, pred_init_vel = predictor(desired_final)

        # Build full initial state
        init_positions = torch.stack([
            torch.tensor([0.0, 0.0, 0.0]),
            pred_init_pos
        ])

        init_velocities = torch.stack([
            torch.zeros(3),
            pred_init_vel
        ])

        masses = torch.ones(2)

        # Simulate
        final_pos, _ = physics(init_positions, init_velocities, masses)

        # Loss: match desired final position
        loss = ((final_pos[1] - desired_final) ** 2).sum()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iteration % 30 == 0:
            print(f"  Iteration {iteration:3d}: Loss = {loss.item():.6f}")

    print(f"\nTraining complete!")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")

    # Test
    print(f"\n{'='*70}")
    print("Testing inverse design:")
    print(f"{'='*70}\n")

    with torch.no_grad():
        test_target = torch.tensor([1.8, 0.0, 0.0])

        pred_pos, pred_vel = predictor(test_target)

        init_pos = torch.stack([torch.zeros(3), pred_pos])
        init_vel = torch.stack([torch.zeros(3), pred_vel])

        final_pos, _ = physics(init_pos, init_vel, torch.ones(2))

        print(f"Desired final position: {test_target.numpy()}")
        print(f"Predicted initial position: {pred_pos.numpy()}")
        print(f"Predicted initial velocity: {pred_vel.numpy()}")
        print(f"Achieved final position: {final_pos[1].numpy()}")
        print(f"Error: {torch.norm(final_pos[1] - test_target).item():.6f}")

    return losses


def example_3_material_designer():
    """
    Example 3.3: Material Property Designer

    Network suggests spring properties, physics evaluates performance.
    This demonstrates co-design of materials and structures.
    """
    print("\n" + "=" * 70)
    print("Example 3.3: Material Property Designer")
    print("=" * 70)
    print("\nGoal: Design spring properties for desired behavior\n")

    class MaterialDesigner(nn.Module):
        """Network that designs material properties."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(6, 32),   # Input: init state
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2),   # Output: [stiffness, rest_length]
                nn.Softplus()       # Ensure positive
            )

        def forward(self, initial_state):
            """Predict material properties."""
            return self.net(initial_state) + 0.1  # Minimum value

    designer = MaterialDesigner()

    print(f"Material designer: {sum(p.numel() for p in designer.parameters())} parameters")
    print("\nConcept: Network learns to design spring properties")
    print("         that achieve target behavior under different conditions")

    print("\n✓ This demonstrates co-design:")
    print("  → Network: Generates material parameters")
    print("  → Physics: Evaluates performance")
    print("  → Gradients: Flow back to improve design")

    print("\nNote: Full implementation would rebuild physics system")
    print("      with predicted parameters at each iteration.")


def main():
    """Run all co-design examples."""
    print("\n" + "=" * 70)
    print("PhysGrad - Neural Network + Physics Co-Design")
    print("=" * 70)
    print("\nThese examples show end-to-end training through physics simulation\n")

    try:
        losses1, errors1 = example_1_control_policy()
        losses2 = example_2_inverse_design()
        example_3_material_designer()

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(losses1, 'b-', linewidth=2, alpha=0.7, label='Loss')
        axes[0].plot(errors1, 'r--', linewidth=2, alpha=0.7, label='Error')
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('Value', fontsize=12)
        axes[0].set_title('Example 3.1: Control Policy Learning', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        axes[1].plot(losses2, 'g-', linewidth=2)
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Example 3.2: Inverse Design', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

        plt.tight_layout()
        output_path = '/tmp/pytorch_codesign_examples.png'
        plt.savefig(output_path, dpi=150)

        print(f"\n{'='*70}")
        print(f"Plot saved to: {output_path}")
        print(f"{'='*70}\n")

        print("✓ All co-design examples completed successfully!")

        print("\nKey insights:")
        print("  1. Networks can learn control policies through physics")
        print("  2. Inverse design: predict causes from desired effects")
        print("  3. End-to-end differentiability enables co-optimization")
        print("  4. Physics provides inductive bias for faster learning")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
