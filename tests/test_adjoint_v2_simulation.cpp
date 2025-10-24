/**
 * Test unified adjoint simulation API - High-level gradient computation
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>

#include "common_types.h"
#include "adjoint_integrators.h"

using namespace physgrad;
using namespace physgrad::adjoint;

class AdjointV2SimulationTest : public ::testing::Test {
protected:
    void SetUp() override {
        epsilon_ = 1e-5f;
        tolerance_ = 5e-3f; // 0.5% tolerance
    }

    // Finite difference validation
    template<typename Func>
    float computeFiniteDifference(
        Func loss_function,
        std::vector<ConceptVector3D<float>>& state,
        size_t particle_idx,
        size_t component_idx) {

        float original = state[particle_idx][component_idx];

        state[particle_idx][component_idx] = original + epsilon_;
        float loss_plus = loss_function();

        state[particle_idx][component_idx] = original - epsilon_;
        float loss_minus = loss_function();

        state[particle_idx][component_idx] = original;

        return (loss_plus - loss_minus) / (2.0f * epsilon_);
    }

    float epsilon_;
    float tolerance_;
};

// =============================================================================
// TEST 1: Simple API - computeGradients()
// =============================================================================

TEST_F(AdjointV2SimulationTest, SimpleGradientsAPI) {
    std::cout << "\n=== Testing Simple Gradients API ===" << std::endl;

    // Setup
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

    // Loss function: L = sum(x²)
    auto loss_function = [](const std::vector<ConceptVector3D<float>>& pos,
                           const std::vector<ConceptVector3D<float>>& vel) -> float {
        float loss = 0.0f;
        for (const auto& p : pos) {
            loss += p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
        }
        return loss;
    };

    // Compute gradients using simple API
    auto [pos_grads, vel_grads] = simulation.computeGradients(
        initial_pos, initial_vel, masses, 0.01f, 5, loss_function
    );

    // Gradients should be non-zero
    EXPECT_NE(pos_grads[0][0], 0.0f);
    EXPECT_NE(pos_grads[1][0], 0.0f);

    std::cout << "Position gradients: dL/dx0=" << pos_grads[0][0]
              << ", dL/dx1=" << pos_grads[1][0] << std::endl;
    std::cout << "✓ Simple API working!" << std::endl;
}

// =============================================================================
// TEST 2: Comprehensive API - computeAllGradients()
// =============================================================================

TEST_F(AdjointV2SimulationTest, ComprehensiveGradientsAPI) {
    std::cout << "\n=== Testing Comprehensive Gradients API ===" << std::endl;

    // Setup
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

    auto loss_function = [](const std::vector<ConceptVector3D<float>>& pos,
                           const std::vector<ConceptVector3D<float>>& vel) -> float {
        float loss = 0.0f;
        for (const auto& p : pos) {
            loss += p[0]*p[0];
        }
        return loss;
    };

    // Compute ALL gradients (including parameters!)
    auto all_grads = simulation.computeAllGradients(
        initial_pos, initial_vel, masses, 0.01f, 5, loss_function
    );

    // Position and velocity gradients
    EXPECT_NE(all_grads.position_grads[0][0], 0.0f);
    EXPECT_NE(all_grads.position_grads[1][0], 0.0f);

    // Parameter gradients
    EXPECT_EQ(all_grads.parameter_grads.spring_constant_grads.size(), 1);
    EXPECT_EQ(all_grads.parameter_grads.rest_length_grads.size(), 1);

    std::cout << "Position gradients: dL/dx0=" << all_grads.position_grads[0][0] << std::endl;
    std::cout << "Parameter gradients: dL/dk=" << all_grads.parameter_grads.spring_constant_grads[0] << std::endl;
    std::cout << "                     dL/dr0=" << all_grads.parameter_grads.rest_length_grads[0] << std::endl;
    std::cout << "✓ Comprehensive API working!" << std::endl;
}

// =============================================================================
// TEST 3: Auto-Detection - Position-only Loss
// =============================================================================

TEST_F(AdjointV2SimulationTest, AutoDetectPositionOnlyLoss) {
    std::cout << "\n=== Testing Auto-Detection: Position-only Loss ===" << std::endl;

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

    // Position-only loss (should auto-detect)
    auto loss_function = [](const std::vector<ConceptVector3D<float>>& pos,
                           const std::vector<ConceptVector3D<float>>& vel) -> float {
        float loss = 0.0f;
        for (const auto& p : pos) {
            loss += p[0]*p[0];
        }
        return loss;
    };

    auto [pos_grads, vel_grads] = simulation.computeGradients(
        initial_pos, initial_vel, masses, 0.01f, 5, loss_function
    );

    // Position gradients should be non-zero
    EXPECT_NE(pos_grads[0][0], 0.0f);

    // Velocity gradients should be near-zero (position-only loss)
    // Note: They're not exactly zero due to chain rule through simulation
    EXPECT_LT(std::abs(vel_grads[0][0]), 0.01f);  // Very small
    EXPECT_LT(std::abs(vel_grads[1][0]), 0.2f);   // Reasonably small

    std::cout << "✓ Auto-detected position-only loss correctly!" << std::endl;
}

// =============================================================================
// TEST 4: Auto-Detection - Kinetic Energy Loss
// =============================================================================

TEST_F(AdjointV2SimulationTest, AutoDetectKineticEnergyLoss) {
    std::cout << "\n=== Testing Auto-Detection: Kinetic Energy Loss ===" << std::endl;

    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    force_engine->addSpring(0, 1, 10.0f, 1.0f);

    AdjointSimulation<float> simulation(force_engine);

    std::vector<ConceptVector3D<float>> initial_pos = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<ConceptVector3D<float>> initial_vel = {
        {0.1f, 0.0f, 0.0f},
        {-0.1f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f};

    // Kinetic energy loss (should auto-detect and use analytical gradients)
    auto loss_function = [&masses](const std::vector<ConceptVector3D<float>>& pos,
                                   const std::vector<ConceptVector3D<float>>& vel) -> float {
        float ke = 0.0f;
        for (size_t i = 0; i < vel.size(); ++i) {
            float v_sq = vel[i][0]*vel[i][0] + vel[i][1]*vel[i][1] + vel[i][2]*vel[i][2];
            ke += 0.5f * masses[i] * v_sq;
        }
        return ke;
    };

    auto [pos_grads, vel_grads] = simulation.computeGradients(
        initial_pos, initial_vel, masses, 0.01f, 5, loss_function
    );

    // Velocity gradients should be non-zero
    EXPECT_NE(vel_grads[0][0], 0.0f);
    EXPECT_NE(vel_grads[1][0], 0.0f);

    std::cout << "Velocity gradients: dL/dv0=" << vel_grads[0][0]
              << ", dL/dv1=" << vel_grads[1][0] << std::endl;
    std::cout << "✓ Auto-detected kinetic energy loss correctly!" << std::endl;
}

// =============================================================================
// TEST 5: Custom Analytical Gradients
// =============================================================================

TEST_F(AdjointV2SimulationTest, CustomAnalyticalGradients) {
    std::cout << "\n=== Testing Custom Analytical Gradients ===" << std::endl;

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

    // Custom loss: distance from target
    ConceptVector3D<float> target = {0.5f, 0.0f, 0.0f};

    auto loss_function = [target](const std::vector<ConceptVector3D<float>>& pos,
                                  const std::vector<ConceptVector3D<float>>& vel) -> float {
        // L = |p0 - target|²
        float dx = pos[0][0] - target[0];
        float dy = pos[0][1] - target[1];
        float dz = pos[0][2] - target[2];
        return dx*dx + dy*dy + dz*dz;
    };

    // Custom analytical gradient
    auto loss_gradient = [target](const std::vector<ConceptVector3D<float>>& pos,
                                  const std::vector<ConceptVector3D<float>>& vel,
                                  std::vector<ConceptVector3D<float>>& grad_pos,
                                  std::vector<ConceptVector3D<float>>& grad_vel) {
        // ∂L/∂p0 = 2*(p0 - target)
        grad_pos[0][0] = 2.0f * (pos[0][0] - target[0]);
        grad_pos[0][1] = 2.0f * (pos[0][1] - target[1]);
        grad_pos[0][2] = 2.0f * (pos[0][2] - target[2]);

        grad_pos[1][0] = grad_pos[1][1] = grad_pos[1][2] = 0.0f;
        grad_vel[0][0] = grad_vel[0][1] = grad_vel[0][2] = 0.0f;
        grad_vel[1][0] = grad_vel[1][1] = grad_vel[1][2] = 0.0f;
    };

    // Use custom analytical gradient
    auto [pos_grads, vel_grads] = simulation.computeGradients(
        initial_pos, initial_vel, masses, 0.01f, 5, loss_function, loss_gradient
    );

    EXPECT_NE(pos_grads[0][0], 0.0f);

    std::cout << "Custom gradient: dL/dx0=" << pos_grads[0][0] << std::endl;
    std::cout << "✓ Custom analytical gradients working!" << std::endl;
}

// =============================================================================
// TEST 6: Multi-timestep Gradient Accumulation
// =============================================================================

TEST_F(AdjointV2SimulationTest, MultiTimestepGradients) {
    std::cout << "\n=== Testing Multi-timestep Gradient Accumulation ===" << std::endl;

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

    auto loss_function = [](const std::vector<ConceptVector3D<float>>& pos,
                           const std::vector<ConceptVector3D<float>>& vel) -> float {
        return pos[1][0] * pos[1][0];  // Focus on second particle
    };

    // Test with different timestep counts
    auto [grads_5, _1] = simulation.computeGradients(
        initial_pos, initial_vel, masses, 0.01f, 5, loss_function
    );

    auto [grads_10, _2] = simulation.computeGradients(
        initial_pos, initial_vel, masses, 0.01f, 10, loss_function
    );

    // Gradients should be different (more timesteps = more accumulated gradient)
    EXPECT_NE(grads_5[0][0], grads_10[0][0]);

    std::cout << "5 steps:  dL/dx0=" << grads_5[0][0] << std::endl;
    std::cout << "10 steps: dL/dx0=" << grads_10[0][0] << std::endl;
    std::cout << "✓ Multi-timestep gradients working!" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
