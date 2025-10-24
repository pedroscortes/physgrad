/**
 * Test unified adjoint integrator v2 - Core functionality
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>

#include "common_types.h"
#include "adjoint_integrators.h"

using namespace physgrad;
using namespace physgrad::adjoint;

class AdjointV2CoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        epsilon_ = 1e-5f;
        tolerance_ = 5e-3f; // 0.5% tolerance
    }

    float epsilon_;
    float tolerance_;
};

// =============================================================================
// TEST 1: Simple Spring Forward Pass
// =============================================================================

TEST_F(AdjointV2CoreTest, SimpleSpringForwardPass) {
    std::cout << "\n=== Testing Simple Spring Forward Pass ===" << std::endl;

    // Create spring force engine
    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    force_engine->addSpring(0, 1, 10.0f, 1.0f); // k=10, r0=1

    // Create integrator
    AdjointVerletIntegrator<float> integrator(force_engine);

    // Initial state: stretched spring
    std::vector<ConceptVector3D<float>> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}  // stretched by 0.5
    };

    std::vector<ConceptVector3D<float>> velocities = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f};

    // Run forward simulation
    float dt = 0.01f;
    int num_steps = 10;

    for (int step = 0; step < num_steps; ++step) {
        integrator.forwardStep(positions, velocities, masses, dt);
    }

    // Particles should have moved (spring pulls them together)
    EXPECT_LT(positions[1][0], 1.5f);  // Particle 1 moved left
    EXPECT_GT(positions[0][0], 0.0f);  // Particle 0 moved right

    std::cout << "Final positions: p0=" << positions[0][0] << ", p1=" << positions[1][0] << std::endl;
    std::cout << "✓ Forward pass working!" << std::endl;
}

// =============================================================================
// TEST 2: Backward Pass - Position Gradients
// =============================================================================

TEST_F(AdjointV2CoreTest, BackwardPassPositionGradients) {
    std::cout << "\n=== Testing Backward Pass - Position Gradients ===" << std::endl;

    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    force_engine->addSpring(0, 1, 10.0f, 1.0f);

    AdjointVerletIntegrator<float> integrator(force_engine);

    // Initial state
    std::vector<ConceptVector3D<float>> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<ConceptVector3D<float>> velocities = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f};

    // Forward pass
    float dt = 0.01f;
    int num_steps = 5;

    std::vector<ConceptVector3D<float>> final_pos, final_vel;

    for (int step = 0; step < num_steps; ++step) {
        integrator.forwardStep(positions, velocities, masses, dt);
    }

    final_pos = positions;
    final_vel = velocities;

    // Backward pass - simple loss: L = sum(x²)
    std::vector<ConceptVector3D<float>> loss_grad_pos(2);
    loss_grad_pos[0] = {2.0f * final_pos[0][0], 0.0f, 0.0f};
    loss_grad_pos[1] = {2.0f * final_pos[1][0], 0.0f, 0.0f};

    std::vector<ConceptVector3D<float>> loss_grad_vel(2);
    loss_grad_vel[0] = {0.0f, 0.0f, 0.0f};
    loss_grad_vel[1] = {0.0f, 0.0f, 0.0f};

    std::vector<ConceptVector3D<float>> pos_grads = loss_grad_pos;
    std::vector<ConceptVector3D<float>> vel_grads = loss_grad_vel;
    std::vector<float> mass_grads(2);

    // Run backward pass
    for (int step = 0; step < num_steps; ++step) {
        integrator.backwardStep(pos_grads, vel_grads, mass_grads);
    }

    // Gradients should be non-zero
    EXPECT_NE(pos_grads[0][0], 0.0f);
    EXPECT_NE(pos_grads[1][0], 0.0f);

    std::cout << "Position gradients: dL/dx0=" << pos_grads[0][0] << ", dL/dx1=" << pos_grads[1][0] << std::endl;
    std::cout << "✓ Backward pass working!" << std::endl;
}

// =============================================================================
// TEST 3: Parameter Gradients
// =============================================================================

TEST_F(AdjointV2CoreTest, ParameterGradients) {
    std::cout << "\n=== Testing Parameter Gradients ===" << std::endl;

    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    force_engine->addSpring(0, 1, 10.0f, 1.0f);

    AdjointVerletIntegrator<float> integrator(force_engine);

    // Initial state
    std::vector<ConceptVector3D<float>> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<ConceptVector3D<float>> velocities = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f};

    // Forward pass
    float dt = 0.01f;
    int num_steps = 5;

    for (int step = 0; step < num_steps; ++step) {
        integrator.forwardStep(positions, velocities, masses, dt);
    }

    // Backward pass WITH parameter gradients
    std::vector<ConceptVector3D<float>> loss_grad_pos(2);
    loss_grad_pos[0] = {2.0f * positions[0][0], 0.0f, 0.0f};
    loss_grad_pos[1] = {2.0f * positions[1][0], 0.0f, 0.0f};

    std::vector<ConceptVector3D<float>> loss_grad_vel(2);
    loss_grad_vel[0] = {0.0f, 0.0f, 0.0f};
    loss_grad_vel[1] = {0.0f, 0.0f, 0.0f};

    std::vector<ConceptVector3D<float>> pos_grads = loss_grad_pos;
    std::vector<ConceptVector3D<float>> vel_grads = loss_grad_vel;
    std::vector<float> mass_grads(2);
    ParameterGradients<float> param_grads;

    for (int step = 0; step < num_steps; ++step) {
        integrator.backwardStep(pos_grads, vel_grads, mass_grads, &param_grads);
    }

    // Parameter gradients should be computed
    EXPECT_EQ(param_grads.spring_constant_grads.size(), 1);
    EXPECT_EQ(param_grads.rest_length_grads.size(), 1);

    std::cout << "dL/dk = " << param_grads.spring_constant_grads[0] << std::endl;
    std::cout << "dL/dr0 = " << param_grads.rest_length_grads[0] << std::endl;
    std::cout << "✓ Parameter gradients working!" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
