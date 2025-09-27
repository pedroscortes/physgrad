#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

using namespace physgrad;

int main() {
    std::cout << "PhysGrad Inverse Problem Demonstration\n";
    std::cout << "=====================================\n\n";

    // Create a simple 2-body system
    SimParams params;
    params.num_bodies = 2;
    params.time_step = 0.01f;
    params.G = 1.0f;

    auto simulation = std::make_unique<Simulation>(params);

    // Manually set initial conditions for a simple binary system
    BodySystem* bodies = simulation->getBodies();

    // Set up a simple binary system manually on GPU
    std::vector<float> init_pos_x = {-0.5f, 0.5f};
    std::vector<float> init_pos_y = {0.0f, 0.0f};
    std::vector<float> init_pos_z = {0.0f, 0.0f};
    std::vector<float> init_vel_x = {0.0f, 0.0f};
    std::vector<float> init_vel_y = {0.5f, -0.5f};
    std::vector<float> init_vel_z = {0.0f, 0.0f};
    std::vector<float> masses = {1.0f, 1.0f};

    // Copy initial conditions to GPU
    size_t size = bodies->n * sizeof(float);
    cudaMemcpy(bodies->d_pos_x, init_pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_y, init_pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_z, init_pos_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_x, init_vel_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_y, init_vel_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_z, init_vel_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

    std::cout << "1. Forward Simulation (10 steps)\n";
    std::cout << "Initial energy: " << bodies->computeEnergy(params) << "\n";

    // Forward simulation to create target trajectory
    simulation->enableGradients();

    for (int step = 0; step < 10; step++) {
        simulation->step();
        if (step % 2 == 0) {
            float energy = bodies->computeEnergy(params);
            std::cout << "Step " << step << " | Energy: " << std::fixed
                      << std::setprecision(6) << energy << "\n";
        }
    }

    // Get final positions as "target"
    std::vector<float> target_pos_x(bodies->n), target_pos_y(bodies->n), target_pos_z(bodies->n);
    bodies->getPositions(target_pos_x, target_pos_y, target_pos_z);

    std::cout << "\nTarget final positions:\n";
    for (int i = 0; i < bodies->n; i++) {
        std::cout << "Body " << i << ": (" << target_pos_x[i] << ", "
                  << target_pos_y[i] << ", " << target_pos_z[i] << ")\n";
    }

    std::cout << "\n2. Inverse Problem: Finding Initial Conditions\n";

    // Reset to perturbed initial conditions
    std::vector<float> perturbed_pos_x = {-0.4f, 0.6f};  // Slightly different
    std::vector<float> perturbed_pos_y = {0.1f, -0.1f};
    std::vector<float> perturbed_pos_z = {0.0f, 0.0f};
    std::vector<float> perturbed_vel_x = {0.05f, -0.05f};
    std::vector<float> perturbed_vel_y = {0.45f, -0.55f};

    cudaMemcpy(bodies->d_pos_x, perturbed_pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_y, perturbed_pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_x, perturbed_vel_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_y, perturbed_vel_y.data(), size, cudaMemcpyHostToDevice);

    std::cout << "Perturbed initial conditions set.\n";

    // Compute gradients
    simulation->clearTape();
    simulation->enableGradients();

    // Forward pass with gradient recording
    for (int step = 0; step < 10; step++) {
        simulation->step();
    }

    // Compute gradients with respect to target
    float loss = simulation->computeGradients(target_pos_x, target_pos_y, target_pos_z);

    std::cout << "\nLoss (MSE): " << loss << "\n";

    // Get computed gradients
    std::vector<float> grad_pos_x(bodies->n), grad_pos_y(bodies->n), grad_pos_z(bodies->n);
    bodies->getGradients(grad_pos_x, grad_pos_y, grad_pos_z);

    std::cout << "\nGradients w.r.t. initial positions:\n";
    for (int i = 0; i < bodies->n; i++) {
        std::cout << "Body " << i << ": (" << grad_pos_x[i] << ", "
                  << grad_pos_y[i] << ", " << grad_pos_z[i] << ")\n";
    }

    std::cout << "\n3. Gradient-Based Optimization (simplified)\n";

    // Simple gradient descent step
    float learning_rate = 0.1f;
    for (int i = 0; i < bodies->n; i++) {
        perturbed_pos_x[i] -= learning_rate * grad_pos_x[i];
        perturbed_pos_y[i] -= learning_rate * grad_pos_y[i];
        perturbed_pos_z[i] -= learning_rate * grad_pos_z[i];
    }

    // Test improved initial conditions
    cudaMemcpy(bodies->d_pos_x, perturbed_pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_y, perturbed_pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_z, perturbed_pos_z.data(), size, cudaMemcpyHostToDevice);

    simulation->clearTape();
    simulation->enableGradients();

    for (int step = 0; step < 10; step++) {
        simulation->step();
    }

    float improved_loss = simulation->computeGradients(target_pos_x, target_pos_y, target_pos_z);

    std::cout << "Improved loss after one gradient step: " << improved_loss << "\n";
    std::cout << "Loss reduction: " << (loss - improved_loss) << "\n";

    if (improved_loss < loss) {
        std::cout << "✓ Gradient descent is working! Loss decreased.\n";
    } else {
        std::cout << "⚠ Gradient descent step increased loss (may need smaller learning rate).\n";
    }

    std::cout << "\nDemonstration complete.\n";
    std::cout << "The differentiable physics engine can compute gradients\n";
    std::cout << "that enable solving inverse problems and parameter optimization.\n";

    return 0;
}