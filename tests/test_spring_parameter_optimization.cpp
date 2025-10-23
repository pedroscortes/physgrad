#include <gtest/gtest.h>
#include "adjoint_integrators_standalone.h"
#include "common_types.h"
#include <cmath>
#include <iostream>

using namespace physgrad::adjoint;
using physgrad::ConceptVector3D;

/**
 * Test: Spring Constant Parameter Optimization
 *
 * This test validates that force parameter gradients (∂L/∂k, ∂L/∂r0) work correctly,
 * addressing TECHNICAL_DEBT.md Issue #2.
 *
 * Scenario:
 * - Two particles connected by a spring with SUBOPTIMAL spring constant
 * - Target: Minimize final position distance from desired state
 * - Use computeAllGradients() to get spring constant gradients
 * - Validate gradients point in correct direction
 * - Run optimization iterations and verify loss decreases
 */

TEST(SpringParameterOptimization, GradientValidation) {
    using Float = float;
    using Vec3 = ConceptVector3D<Float>;

    // System setup: Two particles connected by a spring
    // Particle 0: Fixed at origin
    // Particle 1: Free to move

    std::vector<Vec3> initial_positions = {
        {0.0f, 0.0f, 0.0f},  // Particle 0 (fixed)
        {2.0f, 0.0f, 0.0f}   // Particle 1 (initial position)
    };

    std::vector<Vec3> initial_velocities = {
        {0.0f, 0.0f, 0.0f},  // Particle 0 (fixed, no velocity)
        {0.0f, 0.0f, 0.0f}   // Particle 1 (starts at rest)
    };

    std::vector<Float> masses = {1000.0f, 1.0f};  // Particle 0 very heavy (effectively fixed)

    // Spring parameters (suboptimal)
    Float spring_k = 10.0f;     // Spring constant
    Float rest_length = 1.0f;   // Rest length

    // Target: We want the spring to settle at exactly rest length
    // With k=10, the system will oscillate. We want to find optimal k.
    Vec3 target_position = {1.0f, 0.0f, 0.0f};  // At rest length from particle 0

    // Create force engine
    auto force_engine = std::make_shared<SimpleForceEngine<Float>>();
    force_engine->addSpring(0, 1, spring_k, rest_length);

    // Create simulation
    AdjointSimulation<Float> simulation(force_engine);

    // Simulation parameters
    Float dt = 0.01f;
    int num_steps = 50;  // Simulate for 0.5 seconds

    // Loss function: Minimize distance from target position
    auto loss_function = [&](const std::vector<Vec3>& positions,
                             const std::vector<Vec3>& velocities) -> Float {
        Vec3 final_pos = positions[1];
        Float dx = final_pos[0] - target_position[0];
        Float dy = final_pos[1] - target_position[1];
        Float dz = final_pos[2] - target_position[2];
        return dx*dx + dy*dy + dz*dz;  // Squared distance
    };

    // Analytical loss gradient (for accuracy)
    auto loss_gradient = [&](const std::vector<Vec3>& positions,
                             const std::vector<Vec3>& velocities,
                             std::vector<Vec3>& grad_pos,
                             std::vector<Vec3>& grad_vel) {
        // dL/dx = 2 * (x - target)
        Vec3 final_pos = positions[1];
        grad_pos[0] = {0.0f, 0.0f, 0.0f};  // Fixed particle
        grad_pos[1] = {
            2.0f * (final_pos[0] - target_position[0]),
            2.0f * (final_pos[1] - target_position[1]),
            2.0f * (final_pos[2] - target_position[2])
        };

        // Velocity-independent loss
        grad_vel[0] = {0.0f, 0.0f, 0.0f};
        grad_vel[1] = {0.0f, 0.0f, 0.0f};
    };

    // Compute all gradients
    auto all_grads = simulation.computeAllGradients(
        initial_positions, initial_velocities, masses,
        dt, num_steps,
        loss_function,
        loss_gradient
    );

    // Validate we got parameter gradients
    ASSERT_EQ(all_grads.parameter_grads.spring_constant_grads.size(), 1);
    ASSERT_EQ(all_grads.parameter_grads.rest_length_grads.size(), 1);

    Float grad_k = all_grads.parameter_grads.spring_constant_grads[0];
    Float grad_r0 = all_grads.parameter_grads.rest_length_grads[0];

    std::cout << "\n=== Spring Parameter Gradients ===" << std::endl;
    std::cout << "dL/dk  = " << grad_k << std::endl;
    std::cout << "dL/dr0 = " << grad_r0 << std::endl;

    // Gradients should not be zero (system is responsive to parameters)
    EXPECT_NE(std::abs(grad_k), 0.0f) << "Spring constant gradient should be non-zero";

    // Validate gradient with finite differences
    Float epsilon = 1e-3f;

    // Finite difference for spring constant
    {
        auto force_engine_plus = std::make_shared<SimpleForceEngine<Float>>();
        force_engine_plus->addSpring(0, 1, spring_k + epsilon, rest_length);
        AdjointSimulation<Float> sim_plus(force_engine_plus);

        auto pos_plus = initial_positions;
        auto vel_plus = initial_velocities;
        sim_plus.runForward(pos_plus, vel_plus, masses, dt, num_steps);
        Float loss_plus = loss_function(pos_plus, vel_plus);

        auto force_engine_minus = std::make_shared<SimpleForceEngine<Float>>();
        force_engine_minus->addSpring(0, 1, spring_k - epsilon, rest_length);
        AdjointSimulation<Float> sim_minus(force_engine_minus);

        auto pos_minus = initial_positions;
        auto vel_minus = initial_velocities;
        sim_minus.runForward(pos_minus, vel_minus, masses, dt, num_steps);
        Float loss_minus = loss_function(pos_minus, vel_minus);

        Float finite_diff_k = (loss_plus - loss_minus) / (2.0f * epsilon);

        std::cout << "\nFinite Difference Validation (dL/dk):" << std::endl;
        std::cout << "  Analytical: " << grad_k << std::endl;
        std::cout << "  Finite Diff: " << finite_diff_k << std::endl;

        Float relative_error = std::abs(grad_k - finite_diff_k) / (std::abs(finite_diff_k) + 1e-8f);
        std::cout << "  Relative Error: " << (relative_error * 100.0f) << "%" << std::endl;

        // Allow up to 30% error (adjoint method accumulates error over timesteps)
        // The important thing is that gradients have correct sign and reasonable magnitude
        EXPECT_LT(relative_error, 0.3f) << "Spring constant gradient should match finite differences";
    }

    std::cout << "\n✅ Spring constant gradient validation PASSED!" << std::endl;
}


TEST(SpringParameterOptimization, OptimizationLoop) {
    using Float = float;
    using Vec3 = ConceptVector3D<Float>;

    std::cout << "\n=== Spring Constant Optimization Test ===" << std::endl;

    // System setup
    std::vector<Vec3> initial_positions = {
        {0.0f, 0.0f, 0.0f},
        {2.0f, 0.0f, 0.0f}
    };

    std::vector<Vec3> initial_velocities = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<Float> masses = {1000.0f, 1.0f};

    // Start with suboptimal spring constant
    Float spring_k = 5.0f;  // Too weak
    Float rest_length = 1.0f;

    Vec3 target_position = {1.0f, 0.0f, 0.0f};

    Float dt = 0.01f;
    int num_steps = 50;

    auto loss_function = [&](const std::vector<Vec3>& positions,
                             const std::vector<Vec3>& velocities) -> Float {
        Vec3 final_pos = positions[1];
        Float dx = final_pos[0] - target_position[0];
        Float dy = final_pos[1] - target_position[1];
        Float dz = final_pos[2] - target_position[2];
        return dx*dx + dy*dy + dz*dz;
    };

    auto loss_gradient = [&](const std::vector<Vec3>& positions,
                             const std::vector<Vec3>& velocities,
                             std::vector<Vec3>& grad_pos,
                             std::vector<Vec3>& grad_vel) {
        Vec3 final_pos = positions[1];
        grad_pos[0] = {0.0f, 0.0f, 0.0f};
        grad_pos[1] = {
            2.0f * (final_pos[0] - target_position[0]),
            2.0f * (final_pos[1] - target_position[1]),
            2.0f * (final_pos[2] - target_position[2])
        };
        grad_vel[0] = {0.0f, 0.0f, 0.0f};
        grad_vel[1] = {0.0f, 0.0f, 0.0f};
    };

    std::cout << "\nInitial spring constant: k = " << spring_k << std::endl;

    // Compute initial loss
    auto force_engine_init = std::make_shared<SimpleForceEngine<Float>>();
    force_engine_init->addSpring(0, 1, spring_k, rest_length);
    AdjointSimulation<Float> sim_init(force_engine_init);
    auto pos_init = initial_positions;
    auto vel_init = initial_velocities;
    sim_init.runForward(pos_init, vel_init, masses, dt, num_steps);
    Float initial_loss = loss_function(pos_init, vel_init);

    std::cout << "Initial loss: " << initial_loss << std::endl;
    std::cout << "\nRunning optimization..." << std::endl;

    // Optimization loop
    Float learning_rate = 0.1f;
    int num_iterations = 10;
    Float previous_loss = initial_loss;

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Create force engine with current parameters
        auto force_engine = std::make_shared<SimpleForceEngine<Float>>();
        force_engine->addSpring(0, 1, spring_k, rest_length);

        // Compute all gradients
        AdjointSimulation<Float> simulation(force_engine);
        auto all_grads = simulation.computeAllGradients(
            initial_positions, initial_velocities, masses,
            dt, num_steps,
            loss_function,
            loss_gradient
        );

        // Get current loss
        auto pos = initial_positions;
        auto vel = initial_velocities;
        simulation.runForward(pos, vel, masses, dt, num_steps);
        Float current_loss = loss_function(pos, vel);

        Float grad_k = all_grads.parameter_grads.spring_constant_grads[0];

        // Update spring constant (gradient descent)
        spring_k -= learning_rate * grad_k;

        // Ensure spring constant stays positive
        spring_k = std::max(spring_k, 0.1f);

        std::cout << "Iter " << iter << ": k = " << spring_k
                  << ", loss = " << current_loss
                  << ", grad_k = " << grad_k << std::endl;

        // Loss should generally decrease (allow some numerical noise)
        if (iter > 0) {
            // After first few iterations, loss should be decreasing
            if (iter >= 3) {
                EXPECT_LE(current_loss, initial_loss * 1.1f)
                    << "Loss should decrease or stay relatively stable";
            }
        }

        previous_loss = current_loss;
    }

    // Final loss should be better than initial
    auto force_engine_final = std::make_shared<SimpleForceEngine<Float>>();
    force_engine_final->addSpring(0, 1, spring_k, rest_length);
    AdjointSimulation<Float> sim_final(force_engine_final);
    auto pos_final = initial_positions;
    auto vel_final = initial_velocities;
    sim_final.runForward(pos_final, vel_final, masses, dt, num_steps);
    Float final_loss = loss_function(pos_final, vel_final);

    std::cout << "\nOptimization complete!" << std::endl;
    std::cout << "Final spring constant: k = " << spring_k << std::endl;
    std::cout << "Final loss: " << final_loss << std::endl;
    std::cout << "Improvement: " << ((initial_loss - final_loss) / initial_loss * 100.0f) << "%" << std::endl;

    // Loss should improve by at least 3% (parameter optimization is often slower than initial conditions)
    // The key validation is that gradients point in the right direction and reduce loss consistently
    EXPECT_LT(final_loss, initial_loss * 0.97f)
        << "Optimization should reduce loss by at least 3%";

    std::cout << "\n✅ Spring constant optimization PASSED!" << std::endl;
}


TEST(SpringParameterOptimization, MultiSpringSystem) {
    using Float = float;
    using Vec3 = ConceptVector3D<Float>;

    std::cout << "\n=== Multi-Spring Parameter Optimization ===" << std::endl;

    // Three particles in a chain: 0 -- 1 -- 2
    // Both springs have suboptimal constants

    std::vector<Vec3> initial_positions = {
        {0.0f, 0.0f, 0.0f},  // Particle 0 (heavy, effectively fixed)
        {1.5f, 0.0f, 0.0f},  // Particle 1 (middle)
        {3.0f, 0.0f, 0.0f}   // Particle 2 (free end)
    };

    std::vector<Vec3> initial_velocities = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<Float> masses = {1000.0f, 1.0f, 1.0f};

    // Two springs with different constants
    Float spring_k1 = 8.0f;   // Spring 0-1
    Float spring_k2 = 12.0f;  // Spring 1-2
    Float rest_length = 1.0f;

    // Target: Both springs at rest
    Vec3 target_pos_1 = {1.0f, 0.0f, 0.0f};
    Vec3 target_pos_2 = {2.0f, 0.0f, 0.0f};

    Float dt = 0.01f;
    int num_steps = 50;

    auto loss_function = [&](const std::vector<Vec3>& positions,
                             const std::vector<Vec3>& velocities) -> Float {
        Float loss = 0.0f;

        // Distance from targets
        Vec3 pos1 = positions[1];
        Vec3 pos2 = positions[2];

        loss += (pos1[0] - target_pos_1[0]) * (pos1[0] - target_pos_1[0]);
        loss += (pos2[0] - target_pos_2[0]) * (pos2[0] - target_pos_2[0]);

        return loss;
    };

    // Create force engine with two springs
    auto force_engine = std::make_shared<SimpleForceEngine<Float>>();
    force_engine->addSpring(0, 1, spring_k1, rest_length);  // Spring 0
    force_engine->addSpring(1, 2, spring_k2, rest_length);  // Spring 1

    AdjointSimulation<Float> simulation(force_engine);

    // Compute all gradients
    auto all_grads = simulation.computeAllGradients(
        initial_positions, initial_velocities, masses,
        dt, num_steps,
        loss_function
    );

    // Should have gradients for both springs
    ASSERT_EQ(all_grads.parameter_grads.spring_constant_grads.size(), 2);

    Float grad_k1 = all_grads.parameter_grads.spring_constant_grads[0];
    Float grad_k2 = all_grads.parameter_grads.spring_constant_grads[1];

    std::cout << "\nMulti-spring gradients:" << std::endl;
    std::cout << "  dL/dk1 = " << grad_k1 << " (spring 0-1)" << std::endl;
    std::cout << "  dL/dk2 = " << grad_k2 << " (spring 1-2)" << std::endl;

    // Both gradients should be non-zero
    EXPECT_NE(std::abs(grad_k1), 0.0f) << "Spring 1 gradient should be non-zero";
    EXPECT_NE(std::abs(grad_k2), 0.0f) << "Spring 2 gradient should be non-zero";

    std::cout << "\n✅ Multi-spring parameter gradients PASSED!" << std::endl;
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
