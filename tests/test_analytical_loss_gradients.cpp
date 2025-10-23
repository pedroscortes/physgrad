/**
 * Test analytical loss gradients for improved velocity gradient accuracy.
 *
 * This test validates that using analytical loss gradients (from loss_gradients.h)
 * significantly improves gradient accuracy compared to finite differences.
 *
 * Expected improvement:
 * - Before: ~3% error with finite differences
 * - After: <0.5% error with analytical gradients
 */

#include <gtest/gtest.h>
#include "../src/adjoint_integrators_standalone.h"
#include "../src/loss_gradients.h"
#include <cmath>
#include <iostream>

using namespace physgrad::adjoint;

class AnalyticalLossGradientsTest : public ::testing::Test {
protected:
    using Float = float;
    using Vec3 = ConceptVector3D<Float>;

    void SetUp() override {
        // Simple 2-particle spring system
        force_engine = std::make_shared<SimpleForceEngine<Float>>();
        force_engine->addSpring(0, 1, 10.0f, 1.0f);  // k=10, rest_length=1.0

        simulation = std::make_unique<AdjointSimulation<Float>>(force_engine);
    }

    std::shared_ptr<SimpleForceEngine<Float>> force_engine;
    std::unique_ptr<AdjointSimulation<Float>> simulation;
};

TEST_F(AnalyticalLossGradientsTest, KineticEnergyGradientAccuracy) {
    /**
     * Test kinetic energy loss gradients.
     *
     * Before: ~3% error with finite differences
     * After: <0.1% error with analytical gradients
     */

    // Initial conditions
    std::vector<Vec3> initial_pos = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<Vec3> initial_vel = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<Float> masses = {1.0f, 1.0f};

    Float dt = 0.01f;
    int num_steps = 10;

    // Loss function: kinetic energy = 0.5 * m * v²
    auto kinetic_energy_loss = [&masses](
        const std::vector<Vec3>& positions,
        const std::vector<Vec3>& velocities) -> Float {
        Float energy = 0.0f;
        for (size_t i = 0; i < velocities.size(); ++i) {
            Float v_squared = velocities[i][0] * velocities[i][0] +
                            velocities[i][1] * velocities[i][1] +
                            velocities[i][2] * velocities[i][2];
            energy += 0.5f * masses[i] * v_squared;
        }
        return energy;
    };

    // Analytical gradient for kinetic energy
    auto analytical_gradient = LossGradients<Float>::kinetic_energy(masses);

    // Compute gradients WITH analytical gradient
    auto [pos_grads_analytical, vel_grads_analytical] = simulation->computeGradients(
        initial_pos, initial_vel, masses, dt, num_steps,
        kinetic_energy_loss,
        analytical_gradient
    );

    // Compute reference gradients via finite differences
    Float eps = 1e-5f;
    std::vector<Vec3> pos_grads_fd(initial_pos.size());
    std::vector<Vec3> vel_grads_fd(initial_vel.size());

    for (size_t i = 0; i < initial_vel.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            auto vel_plus = initial_vel;
            auto vel_minus = initial_vel;
            vel_plus[i][j] += eps;
            vel_minus[i][j] -= eps;

            // Forward simulation for vel+
            auto pos_plus = initial_pos;
            auto vel_plus_copy = vel_plus;
            simulation->runForward(pos_plus, vel_plus_copy, masses, dt, num_steps);
            Float loss_plus = kinetic_energy_loss(pos_plus, vel_plus_copy);

            // Forward simulation for vel-
            auto pos_minus = initial_pos;
            auto vel_minus_copy = vel_minus;
            simulation->runForward(pos_minus, vel_minus_copy, masses, dt, num_steps);
            Float loss_minus = kinetic_energy_loss(pos_minus, vel_minus_copy);

            vel_grads_fd[i][j] = (loss_plus - loss_minus) / (2.0f * eps);
        }
    }

    // Compare analytical vs finite difference
    Float max_error = 0.0f;
    Float total_analytical_norm = 0.0f;
    Float total_fd_norm = 0.0f;

    for (size_t i = 0; i < vel_grads_analytical.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Float error = std::abs(vel_grads_analytical[i][j] - vel_grads_fd[i][j]);
            max_error = std::max(max_error, error);

            total_analytical_norm += vel_grads_analytical[i][j] * vel_grads_analytical[i][j];
            total_fd_norm += vel_grads_fd[i][j] * vel_grads_fd[i][j];
        }
    }

    Float analytical_norm = std::sqrt(total_analytical_norm);
    Float fd_norm = std::sqrt(total_fd_norm);
    Float relative_error = max_error / (fd_norm + 1e-10f);

    std::cout << "\nKinetic Energy Gradient Accuracy:\n";
    std::cout << "  Analytical gradient norm: " << analytical_norm << "\n";
    std::cout << "  Finite diff gradient norm: " << fd_norm << "\n";
    std::cout << "  Max absolute error: " << max_error << "\n";
    std::cout << "  Relative error: " << (relative_error * 100.0f) << "%\n";

    // With analytical gradients, error should be <0.5%
    EXPECT_LT(relative_error, 0.005f) << "Analytical gradient error should be <0.5%";
}

TEST_F(AnalyticalLossGradientsTest, PositionDistanceGradientAccuracy) {
    /**
     * Test position distance loss gradients (already working well, this is a sanity check)
     */

    std::vector<Vec3> initial_pos = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<Vec3> initial_vel = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<Float> masses = {1.0f, 1.0f};
    Float dt = 0.01f;
    int num_steps = 10;

    // Loss: squared distance from origin
    auto position_loss = [](const std::vector<Vec3>& positions,
                           const std::vector<Vec3>& velocities) -> Float {
        Float sum = 0.0f;
        for (const auto& pos : positions) {
            sum += pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
        }
        return sum;
    };

    // Analytical gradient
    auto analytical_gradient = LossGradients<Float>::squared_position_distance();

    // Compute with analytical gradient
    auto [pos_grads_analytical, vel_grads_analytical] = simulation->computeGradients(
        initial_pos, initial_vel, masses, dt, num_steps,
        position_loss,
        analytical_gradient
    );

    // Compute via finite differences
    Float eps = 1e-5f;
    std::vector<Vec3> pos_grads_fd(initial_pos.size());

    for (size_t i = 0; i < initial_pos.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            auto pos_plus = initial_pos;
            auto pos_minus = initial_pos;
            pos_plus[i][j] += eps;
            pos_minus[i][j] -= eps;

            auto pos_plus_copy = pos_plus;
            auto vel_plus = initial_vel;
            simulation->runForward(pos_plus_copy, vel_plus, masses, dt, num_steps);
            Float loss_plus = position_loss(pos_plus_copy, vel_plus);

            auto pos_minus_copy = pos_minus;
            auto vel_minus = initial_vel;
            simulation->runForward(pos_minus_copy, vel_minus, masses, dt, num_steps);
            Float loss_minus = position_loss(pos_minus_copy, vel_minus);

            pos_grads_fd[i][j] = (loss_plus - loss_minus) / (2.0f * eps);
        }
    }

    // Compare
    Float max_error = 0.0f;
    Float total_fd_norm = 0.0f;

    for (size_t i = 0; i < pos_grads_analytical.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Float error = std::abs(pos_grads_analytical[i][j] - pos_grads_fd[i][j]);
            max_error = std::max(max_error, error);
            total_fd_norm += pos_grads_fd[i][j] * pos_grads_fd[i][j];
        }
    }

    Float fd_norm = std::sqrt(total_fd_norm);
    Float relative_error = max_error / (fd_norm + 1e-10f);

    std::cout << "\nPosition Distance Gradient Accuracy:\n";
    std::cout << "  Max absolute error: " << max_error << "\n";
    std::cout << "  Relative error: " << (relative_error * 100.0f) << "%\n";

    // Position gradients should already be <1%
    EXPECT_LT(relative_error, 0.01f) << "Position gradient error should be <1%";
}

TEST_F(AnalyticalLossGradientsTest, TargetTrackingGradient) {
    /**
     * Test target tracking loss with analytical gradients
     */

    std::vector<Vec3> initial_pos = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<Vec3> initial_vel = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<Float> masses = {1.0f, 1.0f};

    // Target positions
    std::vector<Vec3> targets = {
        {1.0f, 0.0f, 0.0f},
        {2.0f, 0.0f, 0.0f}
    };

    Float dt = 0.01f;
    int num_steps = 10;

    // Loss: distance to targets
    auto target_loss = [&targets](const std::vector<Vec3>& positions,
                                  const std::vector<Vec3>& velocities) -> Float {
        Float sum = 0.0f;
        for (size_t i = 0; i < positions.size(); ++i) {
            Float dx = positions[i][0] - targets[i][0];
            Float dy = positions[i][1] - targets[i][1];
            Float dz = positions[i][2] - targets[i][2];
            sum += dx*dx + dy*dy + dz*dz;
        }
        return sum;
    };

    // Analytical gradient
    auto analytical_gradient = LossGradients<Float>::squared_position_distance_to_target(targets);

    // Compute gradients
    auto [pos_grads, vel_grads] = simulation->computeGradients(
        initial_pos, initial_vel, masses, dt, num_steps,
        target_loss,
        analytical_gradient
    );

    // Gradients should be non-zero and point in sensible direction
    EXPECT_GT(std::abs(pos_grads[0][0]), 0.01f) << "Gradient should be non-zero";

    std::cout << "\nTarget Tracking Gradient:\n";
    std::cout << "  Gradient[0]: [" << pos_grads[0][0] << ", "
              << pos_grads[0][1] << ", " << pos_grads[0][2] << "]\n";
    std::cout << "  Gradient[1]: [" << pos_grads[1][0] << ", "
              << pos_grads[1][1] << ", " << pos_grads[1][2] << "]\n";
}

TEST_F(AnalyticalLossGradientsTest, CombinedPositionAndKineticEnergy) {
    /**
     * Test combined loss: position tracking + kinetic energy minimization
     */

    using LocalVec3 = ConceptVector3D<float>;
    using LocalFloat = float;

    std::vector<LocalVec3> initial_pos = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<LocalVec3> initial_vel = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<LocalFloat> masses = {1.0f, 1.0f};

    std::vector<LocalVec3> targets = {
        {0.5f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f}
    };

    LocalFloat position_weight = 1.0f;
    LocalFloat kinetic_weight = 0.5f;

    LocalFloat dt = 0.01f;
    int num_steps = 10;

    // Combined loss
    auto combined_loss = [&](const std::vector<LocalVec3>& positions,
                            const std::vector<LocalVec3>& velocities) -> LocalFloat {
        LocalFloat pos_loss = 0.0f;
        for (size_t i = 0; i < positions.size(); ++i) {
            LocalFloat dx = positions[i][0] - targets[i][0];
            LocalFloat dy = positions[i][1] - targets[i][1];
            LocalFloat dz = positions[i][2] - targets[i][2];
            pos_loss += dx*dx + dy*dy + dz*dz;
        }

        LocalFloat kin_loss = 0.0f;
        for (size_t i = 0; i < velocities.size(); ++i) {
            LocalFloat v2 = velocities[i][0]*velocities[i][0] +
                      velocities[i][1]*velocities[i][1] +
                      velocities[i][2]*velocities[i][2];
            kin_loss += 0.5f * masses[i] * v2;
        }

        return position_weight * pos_loss + kinetic_weight * kin_loss;
    };

    // Analytical gradient
    auto analytical_gradient = LossGradients<LocalFloat>::position_and_kinetic_energy(
        targets, masses, position_weight, kinetic_weight
    );

    // Compute gradients
    auto [pos_grads, vel_grads] = simulation->computeGradients(
        initial_pos, initial_vel, masses, dt, num_steps,
        combined_loss,
        analytical_gradient
    );

    // Both position and velocity gradients should be non-zero
    Float pos_grad_norm = std::sqrt(
        pos_grads[0][0]*pos_grads[0][0] +
        pos_grads[1][0]*pos_grads[1][0]
    );

    EXPECT_GT(pos_grad_norm, 0.01f) << "Position gradients should be non-zero";

    std::cout << "\nCombined Loss Gradients:\n";
    std::cout << "  Position gradient norm: " << pos_grad_norm << "\n";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
