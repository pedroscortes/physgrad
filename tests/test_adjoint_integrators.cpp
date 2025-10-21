/**
 * PhysGrad Adjoint Integrators Unit Tests
 *
 * Comprehensive tests for automatic differentiation of integration schemes.
 * Validates gradient computation accuracy using finite differences.
 */

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include <random>

#include "adjoint_integrators.h"
#include "differentiable_forces.h"
#include "common_types.h"

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "concepts/forward_declarations.h"
#endif

using namespace physgrad;

class AdjointIntegratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random number generator for reproducible tests
        rng_.seed(12345);

        // Create force engine with harmonic oscillator
        force_engine_ = std::make_shared<DifferentiableForceEngine<float>>();

        // Add harmonic spring force (simple case for testing)
        spring_constant_ = 10.0f;
        force_engine_->addHarmonicSpring(0, 1, spring_constant_, 1.0f); // rest length = 1.0

        // Create test particles
        setupTestSystem();
    }

    void setupTestSystem() {
        // Two particles connected by spring
        positions_ = {
            ConceptVector3D<float>{0.0f, 0.0f, 0.0f},
            ConceptVector3D<float>{1.5f, 0.0f, 0.0f}  // stretched spring
        };

        velocities_ = {
            ConceptVector3D<float>{0.1f, 0.0f, 0.0f},
            ConceptVector3D<float>{-0.1f, 0.0f, 0.0f}
        };

        masses_ = {1.0f, 1.0f};
    }

    // Finite difference gradient computation for validation
    float computeFiniteDifferenceGradient(
        std::function<float(const std::vector<ConceptVector3D<float>>&)> func,
        const std::vector<ConceptVector3D<float>>& state,
        size_t particle_idx,
        size_t component_idx,
        float h = 1e-5f) {

        auto state_plus = state;
        auto state_minus = state;

        state_plus[particle_idx][component_idx] += h;
        state_minus[particle_idx][component_idx] -= h;

        float f_plus = func(state_plus);
        float f_minus = func(state_minus);

        return (f_plus - f_minus) / (2.0f * h);
    }

    std::mt19937 rng_;
    std::shared_ptr<DifferentiableForceEngine<float>> force_engine_;
    std::vector<ConceptVector3D<float>> positions_;
    std::vector<ConceptVector3D<float>> velocities_;
    std::vector<float> masses_;
    float spring_constant_;
};

// =============================================================================
// ADJOINT STATE MANAGER TESTS
// =============================================================================

TEST_F(AdjointIntegratorTest, StateManagerBasics) {
    AdjointStateManager<float> manager;

    EXPECT_EQ(manager.getNumCheckpoints(), 0);

    // Create and push checkpoint
    AdjointCheckpoint<float> checkpoint(positions_, velocities_,
                                       positions_, masses_, 0.01f, 0);
    manager.pushCheckpoint(checkpoint);

    EXPECT_EQ(manager.getNumCheckpoints(), 1);

    // Pop checkpoint
    auto retrieved = manager.popCheckpoint();
    EXPECT_EQ(manager.getNumCheckpoints(), 0);
    EXPECT_EQ(retrieved.step_index, 0);
    EXPECT_FLOAT_EQ(retrieved.timestep, 0.01f);
}

TEST_F(AdjointIntegratorTest, AdjointStateInitialization) {
    AdjointStateManager<float> manager;

    std::vector<ConceptVector3D<float>> pos_grads = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}
    };

    std::vector<ConceptVector3D<float>> vel_grads = {
        {0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f}
    };

    manager.initializeAdjointState(pos_grads, vel_grads);

    auto retrieved_pos = manager.getPositionAdjoints();
    auto retrieved_vel = manager.getVelocityAdjoints();

    ASSERT_EQ(retrieved_pos.size(), 2);
    ASSERT_EQ(retrieved_vel.size(), 2);

    EXPECT_FLOAT_EQ(retrieved_pos[0][0], 1.0f);
    EXPECT_FLOAT_EQ(retrieved_vel[1][2], 1.0f);
}

// =============================================================================
// ADJOINT VERLET INTEGRATOR TESTS
// =============================================================================

TEST_F(AdjointIntegratorTest, VerletForwardStep) {
    AdjointVerletIntegrator<float> integrator(force_engine_);

    auto initial_positions = positions_;
    auto initial_velocities = velocities_;

    // Take one forward step
    integrator.forwardStep(positions_, velocities_, masses_, 0.01f);

    // Verify that state changed
    bool positions_changed = false;
    bool velocities_changed = false;

    for (size_t i = 0; i < positions_.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (std::abs(positions_[i][j] - initial_positions[i][j]) > 1e-6f) {
                positions_changed = true;
            }
            if (std::abs(velocities_[i][j] - initial_velocities[i][j]) > 1e-6f) {
                velocities_changed = true;
            }
        }
    }

    EXPECT_TRUE(positions_changed);
    EXPECT_TRUE(velocities_changed);
    EXPECT_EQ(integrator.getNumCheckpoints(), 1);
}

TEST_F(AdjointIntegratorTest, VerletEnergyConservation) {
    AdjointVerletIntegrator<float> integrator(force_engine_);

    // Compute initial energy
    float initial_kinetic = 0.0f;
    for (size_t i = 0; i < positions_.size(); ++i) {
        float v_sq = velocities_[i][0]*velocities_[i][0] +
                     velocities_[i][1]*velocities_[i][1] +
                     velocities_[i][2]*velocities_[i][2];
        initial_kinetic += 0.5f * masses_[i] * v_sq;
    }

    // Compute initial potential energy (harmonic spring)
    float dx = positions_[1][0] - positions_[0][0];
    float initial_potential = 0.5f * spring_constant_ * (dx - 1.0f) * (dx - 1.0f);
    float initial_energy = initial_kinetic + initial_potential;

    // Run simulation for several steps
    for (int step = 0; step < 100; ++step) {
        integrator.forwardStep(positions_, velocities_, masses_, 0.001f);
    }

    // Compute final energy
    float final_kinetic = 0.0f;
    for (size_t i = 0; i < positions_.size(); ++i) {
        float v_sq = velocities_[i][0]*velocities_[i][0] +
                     velocities_[i][1]*velocities_[i][1] +
                     velocities_[i][2]*velocities_[i][2];
        final_kinetic += 0.5f * masses_[i] * v_sq;
    }

    dx = positions_[1][0] - positions_[0][0];
    float final_potential = 0.5f * spring_constant_ * (dx - 1.0f) * (dx - 1.0f);
    float final_energy = final_kinetic + final_potential;

    // Energy should be approximately conserved
    float energy_error = std::abs(final_energy - initial_energy) / initial_energy;
    EXPECT_LT(energy_error, 0.01f); // Less than 1% energy drift
}

TEST_F(AdjointIntegratorTest, VerletGradientAccuracy) {
    AdjointVerletIntegrator<float> integrator(force_engine_);

    auto initial_positions = positions_;
    auto initial_velocities = velocities_;

    // Forward pass
    integrator.forwardStep(positions_, velocities_, masses_, 0.01f);

    // Define simple loss function: L = |x_1|²
    auto loss_function = [](const std::vector<ConceptVector3D<float>>& pos) {
        return pos[1][0] * pos[1][0];
    };

    // Compute analytical gradient using adjoint method
    std::vector<ConceptVector3D<float>> loss_grad_pos(2);
    loss_grad_pos[1][0] = 2.0f * positions_[1][0]; // ∂L/∂x₁

    std::vector<ConceptVector3D<float>> loss_grad_vel(2); // zero gradients

    integrator.initializeBackward(loss_grad_pos, loss_grad_vel);

    std::vector<ConceptVector3D<float>> pos_grads, vel_grads;
    std::vector<float> mass_grads;
    integrator.backwardStep(pos_grads, vel_grads, mass_grads);

    // Compute finite difference gradient for comparison
    auto fd_func = [&](const std::vector<ConceptVector3D<float>>& init_pos) {
        auto pos = init_pos;
        auto vel = initial_velocities;
        AdjointVerletIntegrator<float> temp_integrator(force_engine_);
        temp_integrator.forwardStep(pos, vel, masses_, 0.01f);
        return loss_function(pos);
    };

    float fd_grad = computeFiniteDifferenceGradient(fd_func, initial_positions, 1, 0);

    // Compare analytical and finite difference gradients
    float gradient_error = std::abs(pos_grads[1][0] - fd_grad) / std::max(std::abs(fd_grad), 1e-6f);
    EXPECT_LT(gradient_error, 0.01f); // Less than 1% error
}

// =============================================================================
// ADJOINT SIMULATION FRAMEWORK TESTS
// =============================================================================

TEST_F(AdjointIntegratorTest, SimulationFrameworkBasics) {
    AdjointSimulation<float> simulation(force_engine_);

    auto initial_positions = positions_;
    auto initial_velocities = velocities_;

    // Run forward simulation
    simulation.runForward(positions_, velocities_, masses_, 0.01f, 10);

    // Verify state changed
    bool changed = false;
    for (size_t i = 0; i < positions_.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (std::abs(positions_[i][j] - initial_positions[i][j]) > 1e-6f) {
                changed = true;
                break;
            }
        }
    }
    EXPECT_TRUE(changed);
}

TEST_F(AdjointIntegratorTest, EndToEndGradientComputation) {
    AdjointSimulation<float> simulation(force_engine_);

    auto initial_positions = positions_;
    auto initial_velocities = velocities_;

    // Define loss function: sum of squared positions
    auto loss_function = [](const std::vector<ConceptVector3D<float>>& pos,
                           const std::vector<ConceptVector3D<float>>& vel) {
        float loss = 0.0f;
        for (const auto& p : pos) {
            loss += p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
        }
        return loss;
    };

    // Compute gradients using adjoint method
    auto [pos_grads, vel_grads] = simulation.computeGradients(
        initial_positions, initial_velocities, masses_, 0.01f, 5, loss_function);

    // Verify gradients are non-zero (indicating computation worked)
    bool has_nonzero_grad = false;
    for (const auto& grad : pos_grads) {
        for (size_t j = 0; j < 3; ++j) {
            if (std::abs(grad[j]) > 1e-6f) {
                has_nonzero_grad = true;
                break;
            }
        }
    }
    EXPECT_TRUE(has_nonzero_grad);
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

TEST_F(AdjointIntegratorTest, MemoryEfficiency) {
    AdjointVerletIntegrator<float> integrator(force_engine_);

    size_t initial_checkpoints = integrator.getNumCheckpoints();

    // Run many forward steps
    for (int step = 0; step < 1000; ++step) {
        integrator.forwardStep(positions_, velocities_, masses_, 0.001f);
    }

    size_t final_checkpoints = integrator.getNumCheckpoints();

    // Should have stored exactly the number of steps taken
    EXPECT_EQ(final_checkpoints - initial_checkpoints, 1000);

    // Reset should clear memory
    integrator.reset();
    EXPECT_EQ(integrator.getNumCheckpoints(), 0);
}

TEST_F(AdjointIntegratorTest, NumericalStability) {
    AdjointVerletIntegrator<float> integrator(force_engine_);

    // Test with various timesteps
    std::vector<float> timesteps = {0.1f, 0.01f, 0.001f, 0.0001f};

    for (float dt : timesteps) {
        auto test_positions = positions_;
        auto test_velocities = velocities_;

        // Run forward
        for (int step = 0; step < 10; ++step) {
            integrator.forwardStep(test_positions, test_velocities, masses_, dt);
        }

        // Check for NaN or infinite values
        for (const auto& pos : test_positions) {
            for (size_t j = 0; j < 3; ++j) {
                EXPECT_TRUE(std::isfinite(pos[j])) << "Non-finite position at dt=" << dt;
            }
        }

        for (const auto& vel : test_velocities) {
            for (size_t j = 0; j < 3; ++j) {
                EXPECT_TRUE(std::isfinite(vel[j])) << "Non-finite velocity at dt=" << dt;
            }
        }

        integrator.reset();
    }
}

// =============================================================================
// CONCEPT COMPLIANCE TESTS
// =============================================================================

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE

TEST_F(AdjointIntegratorTest, ConceptCompliance) {
    // Test that our types satisfy physics concepts
    EXPECT_TRUE((concepts::PhysicsScalar<float>));
    EXPECT_TRUE((concepts::PhysicsScalar<double>));
    EXPECT_TRUE((concepts::Vector3D<ConceptVector3D<float>>));
}

#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}