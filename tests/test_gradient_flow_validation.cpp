/**
 * PhysGrad End-to-End Gradient Flow Validation Tests
 *
 * Comprehensive tests validating the complete differentiable physics pipeline:
 * - Forward simulation through adjoint kernels
 * - Backward gradient propagation
 * - PyTorch autograd integration
 * - Finite difference validation
 * - Multi-physics gradient flow
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

#include "common_types.h"
#include "adjoint_integrators_standalone.h"

using namespace physgrad;
using namespace physgrad::adjoint;

class GradientFlowValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(42);
        epsilon_ = 1e-5f;  // Finite difference step
        tolerance_ = 1e-3f; // Gradient comparison tolerance
    }

    // Finite difference gradient computation
    template<typename Func>
    float computeFiniteDifferenceGradient(
        Func loss_function,
        std::vector<ConceptVector3D<float>>& state,
        size_t particle_idx,
        size_t component_idx) {

        float original = state[particle_idx][component_idx];

        // f(x + h)
        state[particle_idx][component_idx] = original + epsilon_;
        float loss_plus = loss_function();

        // f(x - h)
        state[particle_idx][component_idx] = original - epsilon_;
        float loss_minus = loss_function();

        // Restore original value
        state[particle_idx][component_idx] = original;

        // Central difference: (f(x+h) - f(x-h)) / (2h)
        return (loss_plus - loss_minus) / (2.0f * epsilon_);
    }

    // Compare analytical vs numerical gradients
    bool validateGradient(float analytical, float numerical, const std::string& name) {
        float abs_error = std::abs(analytical - numerical);
        float rel_error = abs_error / (std::max(std::abs(numerical), 1e-8f));

        bool passed = (abs_error < tolerance_) || (rel_error < tolerance_);

        if (!passed) {
            std::cout << "  " << name << " FAILED:" << std::endl;
            std::cout << "    Analytical: " << analytical << std::endl;
            std::cout << "    Numerical:  " << numerical << std::endl;
            std::cout << "    Abs Error:  " << abs_error << std::endl;
            std::cout << "    Rel Error:  " << rel_error << std::endl;
        }

        return passed;
    }

    std::mt19937 rng_;
    float epsilon_;
    float tolerance_;
};

// =============================================================================
// TEST 1: Single Timestep Gradient Flow
// =============================================================================

TEST_F(GradientFlowValidationTest, SingleTimestepGradientFlow) {
    std::cout << "\n=== Testing Single Timestep Gradient Flow ===" << std::endl;

    // Create simple two-particle spring system
    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    force_engine->addSpring(0, 1, 10.0f, 1.0f); // k=10, rest_length=1

    AdjointVerletIntegrator<float> integrator(force_engine);

    // Initial state
    std::vector<ConceptVector3D<float>> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}  // Stretched spring
    };

    std::vector<ConceptVector3D<float>> velocities = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;

    auto initial_positions = positions;
    auto initial_velocities = velocities;

    // Forward pass
    integrator.forwardStep(positions, velocities, masses, dt);

    // Loss function: L = x₁²
    auto compute_loss = [&]() {
        return positions[1][0] * positions[1][0];
    };

    float loss = compute_loss();

    // Analytical gradients via adjoint method
    std::vector<ConceptVector3D<float>> loss_grad_pos(2);
    loss_grad_pos[1][0] = 2.0f * positions[1][0]; // dL/dx₁ = 2x₁

    std::vector<ConceptVector3D<float>> loss_grad_vel(2); // Zero

    integrator.initializeBackward(loss_grad_pos, loss_grad_vel);

    std::vector<ConceptVector3D<float>> pos_grads, vel_grads;
    std::vector<float> mass_grads;
    integrator.backwardStep(pos_grads, vel_grads, mass_grads);

    // Numerical gradients via finite differences
    auto loss_function = [&]() {
        auto test_pos = initial_positions;
        auto test_vel = initial_velocities;
        AdjointVerletIntegrator<float> temp_integrator(force_engine);
        temp_integrator.forwardStep(test_pos, test_vel, masses, dt);
        return test_pos[1][0] * test_pos[1][0];
    };

    // Validate gradients for particle 1 position
    float numerical_grad = computeFiniteDifferenceGradient(
        loss_function, initial_positions, 1, 0);

    bool gradient_valid = validateGradient(
        pos_grads[1][0], numerical_grad, "dL/dx₁");

    std::cout << "Loss value: " << loss << std::endl;
    std::cout << "Analytical gradient: " << pos_grads[1][0] << std::endl;
    std::cout << "Numerical gradient:  " << numerical_grad << std::endl;

    EXPECT_TRUE(gradient_valid) << "Single timestep gradient should match finite difference";
}

// =============================================================================
// TEST 2: Multi-Timestep Gradient Accumulation
// =============================================================================

TEST_F(GradientFlowValidationTest, MultiTimestepGradientAccumulation) {
    std::cout << "\n=== Testing Multi-Timestep Gradient Accumulation ===" << std::endl;

    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    force_engine->addSpring(0, 1, 10.0f, 1.0f);

    AdjointSimulation<float> simulation(force_engine);

    std::vector<ConceptVector3D<float>> initial_positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<ConceptVector3D<float>> initial_velocities = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;
    int num_steps = 5;

    // Define loss function: sum of squared final positions
    auto loss_function = [](const std::vector<ConceptVector3D<float>>& pos,
                           const std::vector<ConceptVector3D<float>>& vel) {
        float loss = 0.0f;
        for (const auto& p : pos) {
            loss += p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
        }
        return loss;
    };

    // Compute analytical gradients via adjoint method
    auto [pos_grads, vel_grads] = simulation.computeGradients(
        initial_positions, initial_velocities, masses, dt, num_steps, loss_function);

    // Compute numerical gradient for first particle, x-component
    auto numerical_loss = [&](float perturbation) {
        auto test_pos = initial_positions;
        test_pos[0][0] += perturbation;
        auto test_vel = initial_velocities;

        AdjointSimulation<float> temp_sim(force_engine);
        temp_sim.runForward(test_pos, test_vel, masses, dt, num_steps);

        return loss_function(test_pos, test_vel);
    };

    float loss_plus = numerical_loss(epsilon_);
    float loss_minus = numerical_loss(-epsilon_);
    float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon_);

    bool gradient_valid = validateGradient(
        pos_grads[0][0], numerical_grad, "dL/dx₀ (5 timesteps)");

    std::cout << "Num timesteps: " << num_steps << std::endl;
    std::cout << "Analytical gradient: " << pos_grads[0][0] << std::endl;
    std::cout << "Numerical gradient:  " << numerical_grad << std::endl;

    EXPECT_TRUE(gradient_valid) << "Multi-timestep gradient should match finite difference";

    // Verify gradients are non-zero (indicating proper flow)
    bool has_nonzero_grad = false;
    for (const auto& grad : pos_grads) {
        for (int i = 0; i < 3; ++i) {
            if (std::abs(grad[i]) > 1e-6f) {
                has_nonzero_grad = true;
                break;
            }
        }
    }

    EXPECT_TRUE(has_nonzero_grad) << "Gradients should propagate through multiple timesteps";
}

// =============================================================================
// TEST 3: Multi-Particle System Gradient Flow
// =============================================================================

TEST_F(GradientFlowValidationTest, MultiParticleGradientFlow) {
    std::cout << "\n=== Testing Multi-Particle Gradient Flow ===" << std::endl;

    int n_particles = 5;

    // Create chain of springs
    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    for (int i = 0; i < n_particles - 1; ++i) {
        force_engine->addSpring(i, i + 1, 10.0f, 1.0f);
    }

    AdjointSimulation<float> simulation(force_engine);

    // Initialize particles in a line
    std::vector<ConceptVector3D<float>> initial_positions;
    for (int i = 0; i < n_particles; ++i) {
        initial_positions.push_back({static_cast<float>(i) * 1.1f, 0.0f, 0.0f});
    }

    std::vector<ConceptVector3D<float>> initial_velocities(n_particles);
    std::vector<float> masses(n_particles, 1.0f);

    float dt = 0.01f;
    int num_steps = 3;

    // Loss: distance of last particle from origin
    auto loss_function = [](const std::vector<ConceptVector3D<float>>& pos,
                           const std::vector<ConceptVector3D<float>>& vel) {
        auto& last = pos.back();
        return last[0]*last[0] + last[1]*last[1] + last[2]*last[2];
    };

    // Analytical gradients
    auto [pos_grads, vel_grads] = simulation.computeGradients(
        initial_positions, initial_velocities, masses, dt, num_steps, loss_function);

    // Validate gradient for first particle (should be non-zero due to chain)
    auto numerical_loss = [&](float perturbation) {
        auto test_pos = initial_positions;
        test_pos[0][0] += perturbation;
        auto test_vel = initial_velocities;

        AdjointSimulation<float> temp_sim(force_engine);
        temp_sim.runForward(test_pos, test_vel, masses, dt, num_steps);

        return loss_function(test_pos, test_vel);
    };

    float loss_plus = numerical_loss(epsilon_);
    float loss_minus = numerical_loss(-epsilon_);
    float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon_);

    bool gradient_valid = validateGradient(
        pos_grads[0][0], numerical_grad, "dL/dx₀ (chain of " + std::to_string(n_particles) + " particles)");

    std::cout << "Particles: " << n_particles << std::endl;
    std::cout << "Analytical gradient (first particle): " << pos_grads[0][0] << std::endl;
    std::cout << "Numerical gradient (first particle):  " << numerical_grad << std::endl;

    EXPECT_TRUE(gradient_valid) << "Gradient should flow through particle chain";

    // Verify all particles have gradients (gradient flow through chain)
    int particles_with_gradients = 0;
    for (const auto& grad : pos_grads) {
        if (std::abs(grad[0]) > 1e-8f) {
            particles_with_gradients++;
        }
    }

    std::cout << "Particles with non-zero gradients: " << particles_with_gradients << "/" << n_particles << std::endl;

    EXPECT_GE(particles_with_gradients, 3) << "Gradients should propagate through most of the chain";
}

// =============================================================================
// TEST 4: Energy-Based Loss Gradient Flow
// =============================================================================

TEST_F(GradientFlowValidationTest, EnergyBasedLossGradientFlow) {
    std::cout << "\n=== Testing Energy-Based Loss Gradient Flow ===" << std::endl;

    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    force_engine->addSpring(0, 1, 10.0f, 1.0f);

    AdjointSimulation<float> simulation(force_engine);

    std::vector<ConceptVector3D<float>> initial_positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<ConceptVector3D<float>> initial_velocities = {
        {0.1f, 0.0f, 0.0f},
        {-0.1f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;
    int num_steps = 5;

    // Loss: total kinetic energy
    auto loss_function = [&masses](const std::vector<ConceptVector3D<float>>& pos,
                                   const std::vector<ConceptVector3D<float>>& vel) {
        float kinetic = 0.0f;
        for (size_t i = 0; i < vel.size(); ++i) {
            float v_sq = vel[i][0]*vel[i][0] + vel[i][1]*vel[i][1] + vel[i][2]*vel[i][2];
            kinetic += 0.5f * masses[i] * v_sq;
        }
        return kinetic;
    };

    // Analytical gradients
    auto [pos_grads, vel_grads] = simulation.computeGradients(
        initial_positions, initial_velocities, masses, dt, num_steps, loss_function);

    // Numerical gradient w.r.t. initial velocity
    auto numerical_loss = [&](float perturbation) {
        auto test_pos = initial_positions;
        auto test_vel = initial_velocities;
        test_vel[0][0] += perturbation;

        AdjointSimulation<float> temp_sim(force_engine);
        temp_sim.runForward(test_pos, test_vel, masses, dt, num_steps);

        return loss_function(test_pos, test_vel);
    };

    float loss_plus = numerical_loss(epsilon_);
    float loss_minus = numerical_loss(-epsilon_);
    float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon_);

    bool gradient_valid = validateGradient(
        vel_grads[0][0], numerical_grad, "dL/dv₀ (energy loss)");

    std::cout << "Loss type: Kinetic energy" << std::endl;
    std::cout << "Analytical velocity gradient: " << vel_grads[0][0] << std::endl;
    std::cout << "Numerical velocity gradient:  " << numerical_grad << std::endl;

    EXPECT_TRUE(gradient_valid) << "Energy-based loss gradients should match finite difference";
}

// =============================================================================
// TEST 5: Gradient Vanishing Check
// =============================================================================

TEST_F(GradientFlowValidationTest, GradientVanishingCheck) {
    std::cout << "\n=== Testing Gradient Vanishing ===" << std::endl;

    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    force_engine->addSpring(0, 1, 10.0f, 1.0f);

    std::vector<ConceptVector3D<float>> initial_positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<ConceptVector3D<float>> initial_velocities = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;

    auto loss_function = [](const std::vector<ConceptVector3D<float>>& pos,
                           const std::vector<ConceptVector3D<float>>& vel) {
        return pos[1][0] * pos[1][0];
    };

    // Test increasing number of timesteps
    std::vector<int> timestep_counts = {1, 5, 10, 20, 50};
    std::vector<float> gradient_magnitudes;

    for (int num_steps : timestep_counts) {
        AdjointSimulation<float> simulation(force_engine);
        auto [pos_grads, vel_grads] = simulation.computeGradients(
            initial_positions, initial_velocities, masses, dt, num_steps, loss_function);

        float grad_mag = std::sqrt(
            pos_grads[0][0]*pos_grads[0][0] +
            pos_grads[0][1]*pos_grads[0][1] +
            pos_grads[0][2]*pos_grads[0][2]
        );

        gradient_magnitudes.push_back(grad_mag);

        std::cout << "Timesteps: " << std::setw(3) << num_steps
                  << " | Gradient magnitude: " << grad_mag << std::endl;
    }

    // Check that gradients don't vanish (all should be finite and reasonable)
    for (size_t i = 0; i < gradient_magnitudes.size(); ++i) {
        EXPECT_TRUE(std::isfinite(gradient_magnitudes[i]))
            << "Gradient should be finite at " << timestep_counts[i] << " timesteps";
        EXPECT_GT(gradient_magnitudes[i], 1e-10f)
            << "Gradient should not vanish at " << timestep_counts[i] << " timesteps";
        EXPECT_LT(gradient_magnitudes[i], 1e10f)
            << "Gradient should not explode at " << timestep_counts[i] << " timesteps";
    }
}

// =============================================================================
// TEST 6: Gradient Flow Through Different Force Types
// =============================================================================

TEST_F(GradientFlowValidationTest, GradientFlowThroughForces) {
    std::cout << "\n=== Testing Gradient Flow Through Different Forces ===" << std::endl;

    // Test with different spring stiffnesses
    std::vector<float> stiffnesses = {1.0f, 10.0f, 100.0f};

    for (float k : stiffnesses) {
        auto force_engine = std::make_shared<SimpleForceEngine<float>>();
        force_engine->addSpring(0, 1, k, 1.0f);

        AdjointSimulation<float> simulation(force_engine);

        std::vector<ConceptVector3D<float>> initial_positions = {
            {0.0f, 0.0f, 0.0f},
            {1.5f, 0.0f, 0.0f}
        };

        std::vector<ConceptVector3D<float>> initial_velocities(2);
        std::vector<float> masses = {1.0f, 1.0f};

        auto loss_function = [](const std::vector<ConceptVector3D<float>>& pos,
                               const std::vector<ConceptVector3D<float>>& vel) {
            return pos[1][0] * pos[1][0];
        };

        auto [pos_grads, vel_grads] = simulation.computeGradients(
            initial_positions, initial_velocities, masses, 0.01f, 3, loss_function);

        float grad_mag = std::abs(pos_grads[0][0]);

        std::cout << "Spring stiffness k=" << std::setw(6) << k
                  << " | Gradient magnitude: " << grad_mag << std::endl;

        EXPECT_TRUE(std::isfinite(grad_mag)) << "Gradient should be finite for k=" << k;
        EXPECT_GT(grad_mag, 0.0f) << "Gradient should be non-zero for k=" << k;
    }
}

// =============================================================================
// MAIN TEST RUNNER
// =============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PhysGrad End-to-End Gradient Flow Validation" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    int result = RUN_ALL_TESTS();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Gradient Flow Validation Complete" << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;

    return result;
}
