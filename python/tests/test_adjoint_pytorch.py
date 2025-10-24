"""
Unit tests for PhysGrad PyTorch Integration

Tests cover:
1. Basic gradient computation through physics
2. PyTorch autograd integration
3. Parameter gradient computation
4. Numerical gradient validation
5. Edge cases and error handling
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import torch

from physgrad.adjoint import SpringMassSystem, AdjointPhysics


class TestSpringMassSystem(unittest.TestCase):
    """Test SpringMassSystem class."""

    def test_create_system(self):
        """Test creating a spring-mass system."""
        system = SpringMassSystem(n_particles=3, dtype='float32')
        self.assertEqual(system.n_particles, 3)
        self.assertEqual(system.dtype, 'float32')
        self.assertEqual(system.get_num_springs(), 0)

    def test_add_springs(self):
        """Test adding springs to the system."""
        system = SpringMassSystem(n_particles=3, dtype='float32')
        system.add_spring(0, 1, stiffness=10.0, rest_length=1.0)
        system.add_spring(1, 2, stiffness=5.0, rest_length=1.5)

        self.assertEqual(system.get_num_springs(), 2)

    def test_invalid_spring_indices(self):
        """Test that invalid spring indices raise errors."""
        system = SpringMassSystem(n_particles=2, dtype='float32')

        with self.assertRaises(ValueError):
            system.add_spring(0, 5, stiffness=10.0, rest_length=1.0)  # Index too large

    def test_double_precision(self):
        """Test creating system with double precision."""
        system = SpringMassSystem(n_particles=2, dtype='float64')
        self.assertEqual(system.dtype, 'float64')


class TestAdjointPhysics(unittest.TestCase):
    """Test AdjointPhysics simulation and gradients."""

    def setUp(self):
        """Create a simple test system."""
        self.system = SpringMassSystem(n_particles=2, dtype='float32')
        self.system.add_spring(0, 1, stiffness=10.0, rest_length=1.0)
        self.physics = AdjointPhysics(self.system, dt=0.01, num_steps=10)

    def test_forward_simulation(self):
        """Test basic forward simulation."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]
        ], dtype=torch.float32)

        velocities = torch.zeros(2, 3, dtype=torch.float32)
        masses = torch.ones(2, dtype=torch.float32)

        final_pos, final_vel = self.physics(positions, velocities, masses)

        self.assertEqual(final_pos.shape, (2, 3))
        self.assertEqual(final_vel.shape, (2, 3))

        # System should oscillate (spring is stretched)
        # Particles should move toward each other
        self.assertLess(final_pos[1, 0].item(), 1.5)  # Particle 1 moves left

    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]
        ], dtype=torch.float32, requires_grad=True)

        velocities = torch.zeros(2, 3, dtype=torch.float32)
        masses = torch.ones(2, dtype=torch.float32)

        final_pos, final_vel = self.physics(positions, velocities, masses)

        # Compute loss
        loss = (final_pos ** 2).sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(positions.grad)
        self.assertGreater(torch.abs(positions.grad).sum().item(), 0.0)

    def test_gradient_accuracy_finite_diff(self):
        """Validate gradients against finite differences."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0]
        ], dtype=torch.float32, requires_grad=True)

        velocities = torch.zeros(2, 3, dtype=torch.float32)
        masses = torch.ones(2, dtype=torch.float32)

        # Compute adjoint gradient
        final_pos, _ = self.physics(positions, velocities, masses)
        loss = (final_pos[1, 0] ** 2)
        loss.backward()

        adjoint_grad = positions.grad[1, 0].item()

        # Compute finite difference gradient
        epsilon = 1e-4
        positions_plus = positions.detach().clone()
        positions_plus[1, 0] += epsilon

        with torch.no_grad():
            final_pos_plus, _ = self.physics(positions_plus, velocities, masses)
            loss_plus = (final_pos_plus[1, 0] ** 2).item()

            positions_minus = positions.detach().clone()
            positions_minus[1, 0] -= epsilon
            final_pos_minus, _ = self.physics(positions_minus, velocities, masses)
            loss_minus = (final_pos_minus[1, 0] ** 2).item()

        finite_diff_grad = (loss_plus - loss_minus) / (2 * epsilon)

        # Check relative error
        rel_error = abs(adjoint_grad - finite_diff_grad) / (abs(finite_diff_grad) + 1e-8)

        print(f"\nGradient validation:")
        print(f"  Adjoint: {adjoint_grad:.6f}")
        print(f"  Finite diff: {finite_diff_grad:.6f}")
        print(f"  Relative error: {rel_error:.2%}")

        self.assertLess(rel_error, 0.05)  # 5% tolerance

    def test_velocity_gradients(self):
        """Test gradients w.r.t. initial velocities."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=torch.float32)

        velocities = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0]
        ], dtype=torch.float32, requires_grad=True)

        masses = torch.ones(2, dtype=torch.float32)

        final_pos, final_vel = self.physics(positions, velocities, masses)

        # Kinetic energy loss
        loss = 0.5 * (final_vel ** 2).sum()
        loss.backward()

        self.assertIsNotNone(velocities.grad)
        self.assertGreater(torch.abs(velocities.grad).sum().item(), 0.0)

    def test_parameter_gradients(self):
        """Test computation of parameter gradients."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.3, 0.0, 0.0]
        ], dtype=torch.float32)

        velocities = torch.zeros(2, 3, dtype=torch.float32)
        masses = torch.ones(2, dtype=torch.float32)

        def loss_fn(pos, vel):
            """Simple loss function."""
            return float((pos[1, 0] ** 2).sum())

        all_grads = self.physics.compute_all_gradients(
            positions, velocities, masses, loss_fn
        )

        # Check that we got all gradient types
        self.assertIn('position_grads', all_grads)
        self.assertIn('velocity_grads', all_grads)
        self.assertIn('spring_constant_grads', all_grads)
        self.assertIn('rest_length_grads', all_grads)

        # Check shapes
        self.assertEqual(all_grads['position_grads'].shape, (2, 3))
        self.assertEqual(all_grads['velocity_grads'].shape, (2, 3))
        self.assertEqual(len(all_grads['spring_constant_grads']), 1)
        self.assertEqual(len(all_grads['rest_length_grads']), 1)

        # Parameter gradients should be non-zero
        self.assertNotEqual(all_grads['spring_constant_grads'][0], 0.0)

    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # Wrong shape
        velocities = torch.zeros(2, 3, dtype=torch.float32)
        masses = torch.ones(2, dtype=torch.float32)

        with self.assertRaises(ValueError):
            self.physics(positions, velocities, masses)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability under various conditions."""

    def test_long_simulation(self):
        """Test stability of long simulation."""
        system = SpringMassSystem(n_particles=2, dtype='float32')
        system.add_spring(0, 1, stiffness=5.0, rest_length=1.0)

        # Long simulation
        physics = AdjointPhysics(system, dt=0.01, num_steps=200)

        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0]
        ], dtype=torch.float32, requires_grad=True)

        velocities = torch.zeros(2, 3, dtype=torch.float32)
        masses = torch.ones(2, dtype=torch.float32)

        final_pos, _ = physics(positions, velocities, masses)

        # Loss and gradients
        loss = (final_pos ** 2).sum()
        loss.backward()

        # Check for NaN or Inf
        self.assertFalse(torch.isnan(final_pos).any())
        self.assertFalse(torch.isinf(final_pos).any())
        self.assertFalse(torch.isnan(positions.grad).any())
        self.assertFalse(torch.isinf(positions.grad).any())

    def test_zero_initial_conditions(self):
        """Test with zero initial conditions."""
        system = SpringMassSystem(n_particles=2, dtype='float32')
        system.add_spring(0, 1, stiffness=10.0, rest_length=1.0)

        physics = AdjointPhysics(system, dt=0.01, num_steps=50)

        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]  # At rest length
        ], dtype=torch.float32, requires_grad=True)

        velocities = torch.zeros(2, 3, dtype=torch.float32)
        masses = torch.ones(2, dtype=torch.float32)

        final_pos, _ = physics(positions, velocities, masses)

        # At rest, system shouldn't move much
        displacement = torch.norm(final_pos - positions.detach()).item()
        self.assertLess(displacement, 0.1)


class TestMultipleSpringConfigurations(unittest.TestCase):
    """Test various spring configurations."""

    def test_chain_of_springs(self):
        """Test chain of 3 particles with 2 springs."""
        system = SpringMassSystem(n_particles=3, dtype='float32')
        system.add_spring(0, 1, stiffness=10.0, rest_length=1.0)
        system.add_spring(1, 2, stiffness=10.0, rest_length=1.0)

        physics = AdjointPhysics(system, dt=0.01, num_steps=20)

        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ], dtype=torch.float32, requires_grad=True)

        velocities = torch.zeros(3, 3, dtype=torch.float32)
        masses = torch.ones(3, dtype=torch.float32)

        final_pos, _ = physics(positions, velocities, masses)
        loss = (final_pos ** 2).sum()
        loss.backward()

        # All particles should have gradients
        for i in range(3):
            self.assertGreater(torch.abs(positions.grad[i]).sum().item(), 0.0)

    def test_different_spring_constants(self):
        """Test with varying spring stiffness."""
        system = SpringMassSystem(n_particles=3, dtype='float32')
        system.add_spring(0, 1, stiffness=5.0, rest_length=1.0)  # Soft
        system.add_spring(1, 2, stiffness=20.0, rest_length=1.0)  # Stiff

        physics = AdjointPhysics(system, dt=0.01, num_steps=20)

        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0]
        ], dtype=torch.float32)

        velocities = torch.zeros(3, 3, dtype=torch.float32)
        masses = torch.ones(3, dtype=torch.float32)

        final_pos, _ = physics(positions, velocities, masses)

        # System should be well-behaved
        self.assertFalse(torch.isnan(final_pos).any())


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSpringMassSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestAdjointPhysics))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalStability))
    suite.addTests(loader.loadTestsFromTestCase(TestMultipleSpringConfigurations))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit(run_tests())
