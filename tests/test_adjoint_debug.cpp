/**
 * Debug adjoint backward pass
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "common_types.h"
#include "adjoint_integrators_standalone.h"

using namespace physgrad;
using namespace physgrad::adjoint;

TEST(AdjointDebug, TwoTimesteps) {
    // Simple 2-particle system with spring between them
    auto force_engine = std::make_shared<SimpleForceEngine<float>>();
    force_engine->addSpring(0, 1, 10.0f, 1.0f);  // Spring between particles 0 and 1

    AdjointSimulation<float> simulation(force_engine);

    std::vector<ConceptVector3D<float>> initial_positions = {
        {0.0f, 0.0f, 0.0f},  // Particle 0 at origin (fixed below)
        {1.5f, 0.0f, 0.0f}   // Particle 1 stretched
    };

    std::vector<ConceptVector3D<float>> initial_velocities = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f};  // Both particles have equal mass
    float dt = 0.01f;
    int num_steps = 1;  // Just 1 step for simple debugging

    // Loss function: sum of squared final positions
    auto loss_function = [](const std::vector<ConceptVector3D<float>>& pos,
                           const std::vector<ConceptVector3D<float>>& vel) {
        float loss = 0.0f;
        for (const auto& p : pos) {
            loss += p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
        }
        return loss;
    };

    // Manual forward pass to see intermediate states
    auto pos = initial_positions;
    auto vel = initial_velocities;

    std::cout << "\n=== FORWARD PASS ===" << std::endl;
    std::cout << "Initial: pos=(" << pos[0][0] << "), vel=(" << vel[0][0] << ")" << std::endl;

    for (int step = 0; step < num_steps; ++step) {
        auto [forces, _] = force_engine->computeForcesAndGradients(pos);
        ConceptVector3D<float> a = {forces[0][0] / masses[0], 0.0f, 0.0f};

        std::cout << "\nStep " << step << ":" << std::endl;
        std::cout << "  Force: " << forces[0][0] << std::endl;
        std::cout << "  Accel: " << a[0] << std::endl;

        // Verlet position update
        ConceptVector3D<float> new_pos;
        new_pos[0] = pos[0][0] + vel[0][0]*dt + 0.5f*a[0]*dt*dt;
        new_pos[1] = 0.0f;
        new_pos[2] = 0.0f;

        // Compute new acceleration
        auto [new_forces, __] = force_engine->computeForcesAndGradients(std::vector{new_pos});
        ConceptVector3D<float> new_a = {new_forces[0][0] / masses[0], 0.0f, 0.0f};

        // Verlet velocity update
        ConceptVector3D<float> new_vel;
        new_vel[0] = vel[0][0] + 0.5f*(a[0] + new_a[0])*dt;
        new_vel[1] = 0.0f;
        new_vel[2] = 0.0f;

        pos[0] = new_pos;
        vel[0] = new_vel;

        std::cout << "  After: pos=(" << pos[0][0] << "), vel=(" << vel[0][0] << ")" << std::endl;
    }

    float final_loss = loss_function(pos, vel);
    std::cout << "\nFinal loss: " << final_loss << std::endl;
    std::cout << "Final pos[0]: (" << pos[0][0] << ", " << pos[0][1] << ", " << pos[0][2] << ")" << std::endl;
    std::cout << "Final pos[1]: (" << pos[1][0] << ", " << pos[1][1] << ", " << pos[1][2] << ")" << std::endl;
    std::cout << "dL/dpos_final[0]: " << 2.0f * pos[0][0] << std::endl;
    std::cout << "dL/dpos_final[1]: " << 2.0f * pos[1][0] << std::endl;

    // Now compute analytical gradients using DIRECT adjoint (like passing test)
    std::cout << "\n=== ANALYTICAL GRADIENTS (DIRECT) ===" << std::endl;

    // Run forward pass
    auto test_pos = initial_positions;
    auto test_vel = initial_velocities;
    AdjointVerletIntegrator<float> integrator(force_engine);
    integrator.forwardStep(test_pos, test_vel, masses, dt);

    // Compute loss gradients analytically
    std::vector<ConceptVector3D<float>> loss_grad_pos(2);
    loss_grad_pos[0][0] = 2.0f * test_pos[0][0];
    loss_grad_pos[1][0] = 2.0f * test_pos[1][0];

    std::vector<ConceptVector3D<float>> loss_grad_vel(2); // Zero

    std::cout << "Final pos after forward: [0]=" << test_pos[0][0] << ", [1]=" << test_pos[1][0] << std::endl;
    std::cout << "Loss gradients: [0]=" << loss_grad_pos[0][0] << ", [1]=" << loss_grad_pos[1][0] << std::endl;

    // Run backward pass
    integrator.initializeBackward(loss_grad_pos, loss_grad_vel);
    std::vector<ConceptVector3D<float>> pos_grads, vel_grads;
    std::vector<float> mass_grads;
    integrator.backwardStep(pos_grads, vel_grads, mass_grads);

    std::cout << "dL/dpos_initial[1] (direct): " << pos_grads[1][0] << std::endl;

    // Also test via AdjointSimulation
    std::cout << "\n=== ANALYTICAL GRADIENTS (VIA SIMULATION) ===" << std::endl;
    auto [pos_grads2, vel_grads2] = simulation.computeGradients(
        initial_positions, initial_velocities, masses, dt, num_steps, loss_function);

    std::cout << "dL/dpos_initial[1] (via simulation): " << pos_grads2[1][0] << std::endl;

    // Compute numerical gradient for particle 1
    std::cout << "\n=== NUMERICAL GRADIENT ===" << std::endl;
    float epsilon = 1e-5f;

    auto compute_loss = [&](float perturbation) {
        auto test_pos = initial_positions;
        test_pos[1][0] += perturbation;  // Perturb particle 1
        auto test_vel = initial_velocities;

        AdjointSimulation<float> temp_sim(force_engine);
        temp_sim.runForward(test_pos, test_vel, masses, dt, num_steps);

        return loss_function(test_pos, test_vel);
    };

    float loss_plus = compute_loss(epsilon);
    float loss_minus = compute_loss(-epsilon);
    float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

    std::cout << "dL/dpos_initial[1] (numerical): " << numerical_grad << std::endl;

    float error_direct = std::abs(pos_grads[1][0] - numerical_grad) / std::abs(numerical_grad);
    float error_simulation = std::abs(pos_grads2[1][0] - numerical_grad) / std::abs(numerical_grad);

    std::cout << "\nRelative error (direct): " << (error_direct * 100.0f) << "%" << std::endl;
    std::cout << "Relative error (simulation): " << (error_simulation * 100.0f) << "%" << std::endl;

    EXPECT_LT(error_direct, 0.01f) << "Direct adjoint gradient should match finite difference";
    EXPECT_LT(error_simulation, 0.01f) << "Simulation gradient should match finite difference";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
