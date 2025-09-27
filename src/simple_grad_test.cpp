#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace physgrad;

// Simple test with 1 step to debug gradient computation
int main() {
    std::cout << "Simple Gradient Debug Test\n";
    std::cout << "==========================\n\n";

    // Very simple 2-body system
    SimParams params;
    params.num_bodies = 2;
    params.time_step = 0.01f;
    params.G = 1.0f;
    params.epsilon = 0.001f;

    auto sim = std::make_unique<Simulation>(params);
    BodySystem* bodies = sim->getBodies();

    // Simple initial conditions
    std::vector<float> init_pos_x = {-0.5f, 0.5f};
    std::vector<float> init_pos_y = {0.0f, 0.0f};
    std::vector<float> init_pos_z = {0.0f, 0.0f};
    std::vector<float> init_vel_x = {0.0f, 0.0f};
    std::vector<float> init_vel_y = {0.5f, -0.5f};
    std::vector<float> init_vel_z = {0.0f, 0.0f};
    std::vector<float> masses = {1.0f, 1.0f};

    size_t size = bodies->n * sizeof(float);

    // Set initial conditions
    cudaMemcpy(bodies->d_pos_x, init_pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_y, init_pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_z, init_pos_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_x, init_vel_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_y, init_vel_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_z, init_vel_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

    std::cout << "Initial positions: (" << init_pos_x[0] << ", " << init_pos_y[0] << "), ("
              << init_pos_x[1] << ", " << init_pos_y[1] << ")\n";

    // Run 1 simulation step with gradient recording
    sim->enableGradients();
    sim->step();

    // Get final positions
    std::vector<float> final_pos_x(2), final_pos_y(2), final_pos_z(2);
    bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

    std::cout << "Final positions: (" << final_pos_x[0] << ", " << final_pos_y[0] << "), ("
              << final_pos_x[1] << ", " << final_pos_y[1] << ")\n";

    // Create simple target (slightly perturbed final positions)
    std::vector<float> target_pos_x = {final_pos_x[0] + 0.01f, final_pos_x[1] - 0.01f};
    std::vector<float> target_pos_y = {final_pos_y[0] + 0.01f, final_pos_y[1] - 0.01f};
    std::vector<float> target_pos_z = {0.0f, 0.0f};

    std::cout << "Target positions: (" << target_pos_x[0] << ", " << target_pos_y[0] << "), ("
              << target_pos_x[1] << ", " << target_pos_y[1] << ")\n\n";

    // Compute adjoint gradients
    float loss = sim->computeGradients(target_pos_x, target_pos_y, target_pos_z);
    std::cout << "Loss: " << loss << "\n";

    std::vector<float> adj_grad_x(2), adj_grad_y(2), adj_grad_z(2);
    bodies->getGradients(adj_grad_x, adj_grad_y, adj_grad_z);

    std::cout << "Adjoint gradients w.r.t. initial pos_x: [" << adj_grad_x[0] << ", " << adj_grad_x[1] << "]\n";
    std::cout << "Adjoint gradients w.r.t. initial pos_y: [" << adj_grad_y[0] << ", " << adj_grad_y[1] << "]\n";

    // Compute finite difference gradients
    std::cout << "\nComputing finite difference gradients...\n";

    float eps = 1e-5f;
    std::vector<float> fd_grad_x(2), fd_grad_y(2);

    for (int i = 0; i < 2; i++) {
        // Gradient w.r.t. pos_x[i]
        auto sim_plus = std::make_unique<Simulation>(params);
        auto sim_minus = std::make_unique<Simulation>(params);

        std::vector<float> pos_x_plus = init_pos_x;
        std::vector<float> pos_x_minus = init_pos_x;
        pos_x_plus[i] += eps;
        pos_x_minus[i] -= eps;

        // Forward perturbation
        BodySystem* bodies_plus = sim_plus->getBodies();
        cudaMemcpy(bodies_plus->d_pos_x, pos_x_plus.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_pos_y, init_pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_pos_z, init_pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_vel_x, init_vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_vel_y, init_vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_vel_z, init_vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        sim_plus->step();
        std::vector<float> final_plus_x(2), final_plus_y(2), final_plus_z(2);
        bodies_plus->getPositions(final_plus_x, final_plus_y, final_plus_z);

        float loss_plus = 0.0f;
        for (int j = 0; j < 2; j++) {
            float dx = final_plus_x[j] - target_pos_x[j];
            float dy = final_plus_y[j] - target_pos_y[j];
            loss_plus += dx*dx + dy*dy;
        }
        loss_plus /= (2.0f * 2);

        // Backward perturbation
        BodySystem* bodies_minus = sim_minus->getBodies();
        cudaMemcpy(bodies_minus->d_pos_x, pos_x_minus.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_pos_y, init_pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_pos_z, init_pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_vel_x, init_vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_vel_y, init_vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_vel_z, init_vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        sim_minus->step();
        std::vector<float> final_minus_x(2), final_minus_y(2), final_minus_z(2);
        bodies_minus->getPositions(final_minus_x, final_minus_y, final_minus_z);

        float loss_minus = 0.0f;
        for (int j = 0; j < 2; j++) {
            float dx = final_minus_x[j] - target_pos_x[j];
            float dy = final_minus_y[j] - target_pos_y[j];
            loss_minus += dx*dx + dy*dy;
        }
        loss_minus /= (2.0f * 2);

        fd_grad_x[i] = (loss_plus - loss_minus) / (2.0f * eps);

        // Gradient w.r.t. pos_y[i]
        auto sim_y_plus = std::make_unique<Simulation>(params);
        auto sim_y_minus = std::make_unique<Simulation>(params);

        std::vector<float> pos_y_plus = init_pos_y;
        std::vector<float> pos_y_minus = init_pos_y;
        pos_y_plus[i] += eps;
        pos_y_minus[i] -= eps;

        // Forward perturbation in y
        BodySystem* bodies_y_plus = sim_y_plus->getBodies();
        cudaMemcpy(bodies_y_plus->d_pos_x, init_pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_plus->d_pos_y, pos_y_plus.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_plus->d_pos_z, init_pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_plus->d_vel_x, init_vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_plus->d_vel_y, init_vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_plus->d_vel_z, init_vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_plus->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        sim_y_plus->step();
        std::vector<float> final_y_plus_x(2), final_y_plus_y(2), final_y_plus_z(2);
        bodies_y_plus->getPositions(final_y_plus_x, final_y_plus_y, final_y_plus_z);

        float loss_y_plus = 0.0f;
        for (int j = 0; j < 2; j++) {
            float dx = final_y_plus_x[j] - target_pos_x[j];
            float dy = final_y_plus_y[j] - target_pos_y[j];
            loss_y_plus += dx*dx + dy*dy;
        }
        loss_y_plus /= (2.0f * 2);

        // Backward perturbation in y
        BodySystem* bodies_y_minus = sim_y_minus->getBodies();
        cudaMemcpy(bodies_y_minus->d_pos_x, init_pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_minus->d_pos_y, pos_y_minus.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_minus->d_pos_z, init_pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_minus->d_vel_x, init_vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_minus->d_vel_y, init_vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_minus->d_vel_z, init_vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_y_minus->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        sim_y_minus->step();
        std::vector<float> final_y_minus_x(2), final_y_minus_y(2), final_y_minus_z(2);
        bodies_y_minus->getPositions(final_y_minus_x, final_y_minus_y, final_y_minus_z);

        float loss_y_minus = 0.0f;
        for (int j = 0; j < 2; j++) {
            float dx = final_y_minus_x[j] - target_pos_x[j];
            float dy = final_y_minus_y[j] - target_pos_y[j];
            loss_y_minus += dx*dx + dy*dy;
        }
        loss_y_minus /= (2.0f * 2);

        fd_grad_y[i] = (loss_y_plus - loss_y_minus) / (2.0f * eps);
    }

    std::cout << "Finite diff gradients w.r.t. initial pos_x: [" << fd_grad_x[0] << ", " << fd_grad_x[1] << "]\n";
    std::cout << "Finite diff gradients w.r.t. initial pos_y: [" << fd_grad_y[0] << ", " << fd_grad_y[1] << "]\n\n";

    std::cout << "Ratios (adjoint/finite_diff):\n";
    std::cout << "pos_x: [" << adj_grad_x[0]/fd_grad_x[0] << ", " << adj_grad_x[1]/fd_grad_x[1] << "]\n";
    std::cout << "pos_y: [" << adj_grad_y[0]/fd_grad_y[0] << ", " << adj_grad_y[1]/fd_grad_y[1] << "]\n";

    return 0;
}