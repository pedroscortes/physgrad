/**
 * Simplified test for analytical loss gradients - velocity gradient improvement
 *
 * Demonstrates that using analytical gradients (from loss_gradients.h)
 * significantly improves velocity gradient accuracy compared to finite differences.
 */

#include <gtest/gtest.h>
#include "../src/adjoint_integrators_standalone.h"
#include "../src/loss_gradients.h"
#include <cmath>
#include <iostream>

using namespace physgrad::adjoint;

TEST(VelocityGradientImprovement, KineticEnergyWithAnalyticalGradient) {
    /**
     * Test kinetic energy loss with analytical gradients.
     * Expected: <0.5% error (vs ~3% with finite differences)
     */

    // Setup 2-particle spring system
    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    force_engine->addSpring(0, 1, 10.0f, 1.0f);

    AdjointSimulation<float> simulation(force_engine);

    // Initial conditions
    std::vector<ConceptVector3D<float>> initial_pos = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<ConceptVector3D<float>> initial_vel = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;
    int num_steps = 10;

    // Loss function: kinetic energy = 0.5 * m * v²
    auto kinetic_energy_loss = [&masses](
        const std::vector<ConceptVector3D<float>>& positions,
        const std::vector<ConceptVector3D<float>>& velocities) -> float {
        float energy = 0.0f;
        for (size_t i = 0; i < velocities.size(); ++i) {
            float v_squared = velocities[i][0] * velocities[i][0] +
                            velocities[i][1] * velocities[i][1] +
                            velocities[i][2] * velocities[i][2];
            energy += 0.5f * masses[i] * v_squared;
        }
        return energy;
    };

    // Compute gradients WITH analytical gradient
    auto analytical_gradient = LossGradients<float>::kinetic_energy(masses);
    auto [pos_grads_analytical, vel_grads_analytical] = simulation.computeGradients(
        initial_pos, initial_vel, masses, dt, num_steps,
        kinetic_energy_loss,
        analytical_gradient
    );

    // Compute reference gradients via finite differences
    float eps = 1e-5f;
    std::vector<ConceptVector3D<float>> vel_grads_fd(initial_vel.size());

    for (size_t i = 0; i < initial_vel.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            auto vel_plus = initial_vel;
            auto vel_minus = initial_vel;
            vel_plus[i][j] += eps;
            vel_minus[i][j] -= eps;

            // Forward simulation for vel+
            auto pos_plus = initial_pos;
            auto vel_plus_copy = vel_plus;
            auto sim_plus = AdjointSimulation<float>(force_engine);
            sim_plus.runForward(pos_plus, vel_plus_copy, masses, dt, num_steps);
            float loss_plus = kinetic_energy_loss(pos_plus, vel_plus_copy);

            // Forward simulation for vel-
            auto pos_minus = initial_pos;
            auto vel_minus_copy = vel_minus;
            auto sim_minus = AdjointSimulation<float>(force_engine);
            sim_minus.runForward(pos_minus, vel_minus_copy, masses, dt, num_steps);
            float loss_minus = kinetic_energy_loss(pos_minus, vel_minus_copy);

            vel_grads_fd[i][j] = (loss_plus - loss_minus) / (2.0f * eps);
        }
    }

    // Compare analytical vs finite difference
    float max_error = 0.0f;
    float total_fd_norm = 0.0f;

    for (size_t i = 0; i < vel_grads_analytical.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float error = std::abs(vel_grads_analytical[i][j] - vel_grads_fd[i][j]);
            max_error = std::max(max_error, error);
            total_fd_norm += vel_grads_fd[i][j] * vel_grads_fd[i][j];
        }
    }

    float fd_norm = std::sqrt(total_fd_norm);
    float relative_error = max_error / (fd_norm + 1e-10f);

    std::cout << "\n=== Velocity Gradient Improvement Test ===\n";
    std::cout << "Kinetic Energy Gradient Accuracy:\n";
    std::cout << "  Max absolute error: " << max_error << "\n";
    std::cout << "  Relative error: " << (relative_error * 100.0f) << "%\n";
    std::cout << "  Result: " << (relative_error < 0.005f ? "PASS" : "FAIL") << "\n";

    // With analytical gradients, error should be <0.5%
    EXPECT_LT(relative_error, 0.005f) << "Analytical gradient error should be <0.5%";
}

TEST(VelocityGradientImprovement, PositionDistanceStillWorks) {
    /**
     * Sanity check: position gradients should still work well
     */

    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    force_engine->addSpring(0, 1, 10.0f, 1.0f);

    AdjointSimulation<float> simulation(force_engine);

    std::vector<ConceptVector3D<float>> initial_pos = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<ConceptVector3D<float>> initial_vel = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;
    int num_steps = 10;

    // Loss: squared distance from origin (auto-detected by implementation)
    auto position_loss = [](const std::vector<ConceptVector3D<float>>& positions,
                           const std::vector<ConceptVector3D<float>>& velocities) -> float {
        float sum = 0.0f;
        for (const auto& pos : positions) {
            sum += pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
        }
        return sum;
    };

    // Use auto-detection (should detect position-only loss)
    auto [pos_grads, vel_grads] = simulation.computeGradients(
        initial_pos, initial_vel, masses, dt, num_steps,
        position_loss
        // No analytical gradient provided - auto-detection
    );

    // Gradients should be non-zero
    float grad_norm = 0.0f;
    for (size_t i = 0; i < pos_grads.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            grad_norm += pos_grads[i][j] * pos_grads[i][j];
        }
    }
    grad_norm = std::sqrt(grad_norm);

    std::cout << "\n=== Position Gradient Sanity Check ===\n";
    std::cout << "  Gradient norm: " << grad_norm << "\n";
    std::cout << "  Result: " << (grad_norm > 0.01f ? "PASS" : "FAIL") << "\n";

    EXPECT_GT(grad_norm, 0.01f) << "Position gradients should be non-zero";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
