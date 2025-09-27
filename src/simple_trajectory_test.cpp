#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace physgrad;

int main() {
    std::cout << "Simple Trajectory Optimization Test\n";
    std::cout << "==================================\n\n";

    // Very simple case: single body with small perturbation
    SimParams params;
    params.num_bodies = 1;  // Just one body, no forces
    params.time_step = 0.1f;
    params.G = 0.0f;  // No gravity - simple ballistic motion
    params.epsilon = 0.001f;

    auto sim = std::make_unique<Simulation>(params);
    sim->enableGradients();
    BodySystem* bodies = sim->getBodies();

    // Initial state
    std::vector<float> pos_x = {0.0f};
    std::vector<float> pos_y = {0.0f};
    std::vector<float> pos_z = {0.0f};
    std::vector<float> vel_x = {1.0f};  // We'll optimize this
    std::vector<float> vel_y = {1.0f};  // And this
    std::vector<float> vel_z = {0.0f};
    std::vector<float> masses = {1.0f};

    // Target: after 10 steps, we want to be at (5, 5)
    // With dt=0.1 and 10 steps = 1.0 time units
    // So we need velocity (5, 5) for straight line motion
    int target_steps = 10;
    std::vector<float> target_pos_x = {5.0f};
    std::vector<float> target_pos_y = {5.0f};
    std::vector<float> target_pos_z = {0.0f};

    std::cout << "Problem: Find velocity to reach (5,5) after " << target_steps << " steps\n";
    std::cout << "Expected solution: velocity = (5,5)\n";
    std::cout << "Initial guess: velocity = (1,1)\n\n";

    // Optimization loop
    float learning_rate = 0.1f;
    for (int iter = 0; iter < 20; iter++) {
        // Reset simulation
        sim = std::make_unique<Simulation>(params);
        sim->enableGradients();
        bodies = sim->getBodies();
        size_t size = bodies->n * sizeof(float);

        // Set initial state
        cudaMemcpy(bodies->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        // Forward simulation
        for (int step = 0; step < target_steps; step++) {
            sim->step();
        }

        // Get final position
        std::vector<float> final_pos_x(1), final_pos_y(1), final_pos_z(1);
        bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

        // Compute loss
        float dx = final_pos_x[0] - target_pos_x[0];
        float dy = final_pos_y[0] - target_pos_y[0];
        float loss = 0.5f * (dx*dx + dy*dy);

        // Compute gradients
        float grad_loss = sim->computeGradients(target_pos_x, target_pos_y, target_pos_z);

        // Get velocity gradients
        std::vector<float> grad_vel_x(1), grad_vel_y(1);
        cudaMemcpy(grad_vel_x.data(), bodies->d_grad_vel_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_vel_y.data(), bodies->d_grad_vel_y, size, cudaMemcpyDeviceToHost);

        // Update velocities
        vel_x[0] -= learning_rate * grad_vel_x[0];
        vel_y[0] -= learning_rate * grad_vel_y[0];

        // Print progress
        if (iter % 5 == 0 || iter == 19) {
            std::cout << "Iter " << iter << ": loss=" << std::fixed << std::setprecision(4) << loss;
            std::cout << " | pos=(" << final_pos_x[0] << "," << final_pos_y[0] << ")";
            std::cout << " | vel=(" << vel_x[0] << "," << vel_y[0] << ")";
            std::cout << " | grad=(" << grad_vel_x[0] << "," << grad_vel_y[0] << ")\n";
        }

        if (loss < 1e-6f) {
            std::cout << "\n✓ Converged!\n";
            break;
        }
    }

    std::cout << "\nFinal velocity: (" << vel_x[0] << ", " << vel_y[0] << ")\n";
    std::cout << "Expected: (5, 5)\n";

    return 0;
}