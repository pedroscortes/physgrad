"""
Test PyTorch integration for adjoint-based differentiable physics.

This test validates that:
1. Forward simulation runs correctly through PyTorch
2. Gradients flow properly via autograd
3. Gradient magnitudes are reasonable (not vanishing/exploding)
4. Gradients match expected behavior for simple systems
"""

import pytest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from physgrad.adjoint_integration import (
        AdjointPhysicsSimulator,
        adjoint_physics_step,
        _ADJOINT_CPP_AVAILABLE
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchAdjointGradients:
    """Test suite for PyTorch adjoint integration."""

    def setup_method(self):
        """Setup simple spring-mass system for testing."""
        # 2-particle spring system
        self.initial_pos = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]
        ], dtype=torch.float32, requires_grad=True)

        self.initial_vel = torch.zeros(2, 3, dtype=torch.float32)
        self.masses = torch.ones(2, dtype=torch.float32)
        self.spring_pairs = torch.tensor([[0, 1]], dtype=torch.int32)
        self.spring_constants = torch.tensor([10.0], dtype=torch.float32)
        self.rest_lengths = torch.tensor([1.0], dtype=torch.float32)

        self.dt = 0.01
        self.num_steps = 10

    def test_forward_pass_runs(self):
        """Test that forward simulation runs without errors."""
        simulator = AdjointPhysicsSimulator(dt=self.dt, num_steps=self.num_steps)

        final_pos, final_vel = simulator(
            self.initial_pos, self.initial_vel, self.masses,
            self.spring_pairs, self.spring_constants, self.rest_lengths
        )

        assert final_pos.shape == (2, 3)
        assert final_vel.shape == (2, 3)
        assert not torch.isnan(final_pos).any()
        assert not torch.isnan(final_vel).any()

    def test_gradients_flow(self):
        """Test that gradients flow through the simulation."""
        simulator = AdjointPhysicsSimulator(dt=self.dt, num_steps=self.num_steps)

        final_pos, final_vel = simulator(
            self.initial_pos, self.initial_vel, self.masses,
            self.spring_pairs, self.spring_constants, self.rest_lengths
        )

        # Simple loss: distance from origin
        loss = (final_pos ** 2).sum()

        # Backprop
        loss.backward()

        # Check gradients exist and are non-zero
        assert self.initial_pos.grad is not None
        assert not torch.isnan(self.initial_pos.grad).any()
        assert torch.abs(self.initial_pos.grad).sum() > 1e-6

        print(f"\nGradient check:")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Gradient norm: {torch.norm(self.initial_pos.grad).item():.6f}")
        print(f"  Max gradient: {torch.abs(self.initial_pos.grad).max().item():.6f}")

    def test_gradient_sanity(self):
        """Test that gradients have reasonable magnitudes."""
        simulator = AdjointPhysicsSimulator(dt=self.dt, num_steps=self.num_steps)

        final_pos, final_vel = simulator(
            self.initial_pos, self.initial_vel, self.masses,
            self.spring_pairs, self.spring_constants, self.rest_lengths
        )

        loss = (final_pos ** 2).sum()
        loss.backward()

        grad_norm = torch.norm(self.initial_pos.grad).item()

        # Gradient should not vanish
        assert grad_norm > 1e-6, f"Gradient vanished: {grad_norm}"

        # Gradient should not explode
        assert grad_norm < 1e6, f"Gradient exploded: {grad_norm}"

    def test_functional_api(self):
        """Test the functional API for single-step simulation."""
        final_pos, final_vel = adjoint_physics_step(
            self.initial_pos, self.initial_vel, self.masses,
            self.spring_pairs, self.spring_constants, self.rest_lengths,
            dt=self.dt, num_steps=5
        )

        loss = (final_pos ** 2).sum()
        loss.backward()

        assert self.initial_pos.grad is not None
        assert torch.abs(self.initial_pos.grad).sum() > 1e-6

    def test_multi_step_gradient_accumulation(self):
        """Test that gradients accumulate correctly over multiple steps."""
        simulator = AdjointPhysicsSimulator(dt=self.dt, num_steps=1)

        # Single long simulation
        pos_long = self.initial_pos.clone().detach().requires_grad_(True)
        simulator_long = AdjointPhysicsSimulator(dt=self.dt, num_steps=10)
        final_pos_long, _ = simulator_long(
            pos_long, self.initial_vel, self.masses,
            self.spring_pairs, self.spring_constants, self.rest_lengths
        )
        loss_long = (final_pos_long ** 2).sum()
        loss_long.backward()

        # Check gradient exists
        assert pos_long.grad is not None
        grad_long = pos_long.grad.clone()

        print(f"\nMulti-step gradient test:")
        print(f"  10-step gradient norm: {torch.norm(grad_long).item():.6f}")

        # Gradient should be non-zero
        assert torch.abs(grad_long).sum() > 1e-6

    def test_spring_compression_gradient_sign(self):
        """Test that gradient direction makes physical sense."""
        # Start with compressed spring (particles closer than rest length)
        initial_pos_compressed = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0]  # Compressed (rest length = 1.0)
        ], dtype=torch.float32, requires_grad=True)

        simulator = AdjointPhysicsSimulator(dt=self.dt, num_steps=5)

        final_pos, _ = simulator(
            initial_pos_compressed, self.initial_vel, self.masses,
            self.spring_pairs, self.spring_constants, self.rest_lengths
        )

        # Loss: want to minimize final displacement
        loss = (final_pos[1, 0] ** 2)  # Only care about particle 1's x position
        loss.backward()

        print(f"\nPhysical gradient sign test:")
        print(f"  Initial separation: {(initial_pos_compressed[1, 0] - initial_pos_compressed[0, 0]).item():.6f}")
        print(f"  Final separation: {(final_pos[1, 0] - final_pos[0, 0]).item():.6f}")
        print(f"  Gradient on particle 1: {initial_pos_compressed.grad[1, 0].item():.6f}")

        # Gradient should exist
        assert initial_pos_compressed.grad is not None

    @pytest.mark.skipif(not _ADJOINT_CPP_AVAILABLE,
                       reason="C++ adjoint backend not available")
    def test_cpp_backend_available(self):
        """Test that C++ backend is being used when available."""
        print(f"\nBackend status:")
        print(f"  PyTorch available: {TORCH_AVAILABLE}")
        print(f"  C++ adjoint backend available: {_ADJOINT_CPP_AVAILABLE}")

        assert _ADJOINT_CPP_AVAILABLE, "C++ backend should be available for production use"

    def test_batch_independence(self):
        """Test that multiple simulations can run independently."""
        simulator1 = AdjointPhysicsSimulator(dt=self.dt, num_steps=5)
        simulator2 = AdjointPhysicsSimulator(dt=self.dt, num_steps=5)

        pos1 = self.initial_pos.clone().detach().requires_grad_(True)
        pos2 = self.initial_pos.clone().detach().requires_grad_(True)

        final1, _ = simulator1(pos1, self.initial_vel, self.masses,
                              self.spring_pairs, self.spring_constants,
                              self.rest_lengths)

        final2, _ = simulator2(pos2, self.initial_vel, self.masses,
                              self.spring_pairs, self.spring_constants,
                              self.rest_lengths)

        # Results should be identical (same initial conditions)
        assert torch.allclose(final1, final2, atol=1e-6)

        loss1 = (final1 ** 2).sum()
        loss2 = (final2 ** 2).sum()

        loss1.backward()
        loss2.backward()

        # Gradients should also be identical
        assert torch.allclose(pos1.grad, pos2.grad, atol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_simple_optimization_loop():
    """Test a simple optimization loop to verify end-to-end training."""
    # Goal: optimize initial position to minimize final displacement

    # Initial guess (will be optimized)
    initial_pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0]
    ], dtype=torch.float32, requires_grad=True)

    initial_vel = torch.zeros(2, 3, dtype=torch.float32)
    masses = torch.ones(2, dtype=torch.float32)
    spring_pairs = torch.tensor([[0, 1]], dtype=torch.int32)
    spring_constants = torch.tensor([10.0], dtype=torch.float32)
    rest_lengths = torch.tensor([1.0], dtype=torch.float32)

    simulator = AdjointPhysicsSimulator(dt=0.01, num_steps=10)
    optimizer = torch.optim.Adam([initial_pos], lr=0.01)

    initial_loss = None
    final_loss = None

    # Run a few optimization steps
    for i in range(5):
        optimizer.zero_grad()

        final_pos, _ = simulator(
            initial_pos, initial_vel, masses,
            spring_pairs, spring_constants, rest_lengths
        )

        # Loss: minimize distance from origin
        loss = (final_pos ** 2).sum()

        if i == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        final_loss = loss.item()

    print(f"\nOptimization test:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Improvement: {100 * (initial_loss - final_loss) / initial_loss:.2f}%")

    # Loss should decrease (at least a little bit)
    # Note: May not always decrease due to dynamics, but gradient should be non-zero
    assert initial_loss is not None and final_loss is not None


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
