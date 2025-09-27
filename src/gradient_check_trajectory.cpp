#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace physgrad;

float evaluateTrajectory(float vel_x, float vel_y) {
    SimParams params;
    params.num_bodies = 2;
    params.time_step = 0.01f;
    params.G = 1.0f;
    params.epsilon = 0.001f;

    auto sim = std::make_unique<Simulation>(params);
    BodySystem* bodies = sim->getBodies();

    // Initial state
    std::vector<float> pos_x = {0.0f, 1.0f};
    std::vector<float> pos_y = {0.0f, 0.0f};
    std::vector<float> pos_z = {0.0f, 0.0f};
    std::vector<float> vx = {0.0f, vel_x};
    std::vector<float> vy = {0.0f, vel_y};
    std::vector<float> vz = {0.0f, 0.0f};
    std::vector<float> masses = {10.0f, 0.1f};

    size_t size = bodies->n * sizeof(float);
    cudaMemcpy(bodies->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_x, vx.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_y, vy.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_z, vz.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

    // Run for fewer steps to isolate issues
    for (int i = 0; i < 10; i++) {
        sim->step();
    }

    // Get final position
    std::vector<float> final_pos_x(2), final_pos_y(2), final_pos_z(2);
    bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

    // Target (0, 1) - match computeGradients loss calculation
    float loss = 0.0f;
    std::vector<float> target_x = {0.0f, 0.0f};
    std::vector<float> target_y = {0.0f, 1.0f};

    for (int i = 0; i < 2; i++) {
        float dx = final_pos_x[i] - target_x[i];
        float dy = final_pos_y[i] - target_y[i];
        loss += dx*dx + dy*dy;
    }
    return loss / (2.0f * 2);  // Match MSE/2 with n=2
}

int main() {
    std::cout << "Trajectory Gradient Check\n";
    std::cout << "========================\n\n";

    // Test point from our optimization - let's try a simpler case
    float vel_x = 0.0f;
    float vel_y = 1.0f;  // Much smaller velocity

    std::cout << "Testing gradients at velocity: (" << vel_x << ", " << vel_y << ")\n\n";

    // Compute analytic gradients
    SimParams params;
    params.num_bodies = 2;
    params.time_step = 0.01f;
    params.G = 1.0f;
    params.epsilon = 0.001f;

    auto sim = std::make_unique<Simulation>(params);
    sim->enableGradients();
    BodySystem* bodies = sim->getBodies();

    std::vector<float> pos_x = {0.0f, 1.0f};
    std::vector<float> pos_y = {0.0f, 0.0f};
    std::vector<float> pos_z = {0.0f, 0.0f};
    std::vector<float> vx = {0.0f, vel_x};
    std::vector<float> vy = {0.0f, vel_y};
    std::vector<float> vz = {0.0f, 0.0f};
    std::vector<float> masses = {10.0f, 0.1f};
    std::vector<float> target_pos_x = {0.0f, 0.0f};
    std::vector<float> target_pos_y = {0.0f, 1.0f};
    std::vector<float> target_pos_z = {0.0f, 0.0f};

    size_t size = bodies->n * sizeof(float);
    cudaMemcpy(bodies->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_x, vx.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_y, vy.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_z, vz.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

    for (int i = 0; i < 10; i++) {
        sim->step();
    }

    float analytic_loss = sim->computeGradients(target_pos_x, target_pos_y, target_pos_z);

    std::vector<float> grad_vel_x(2), grad_vel_y(2);
    cudaMemcpy(grad_vel_x.data(), bodies->d_grad_vel_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_vel_y.data(), bodies->d_grad_vel_y, size, cudaMemcpyDeviceToHost);

    std::cout << "Analytic loss: " << analytic_loss << "\n";
    std::cout << "Analytic gradients: (" << grad_vel_x[1] << ", " << grad_vel_y[1] << ")\n\n";

    // Finite difference check
    float eps = 0.001f;

    float loss_base = evaluateTrajectory(vel_x, vel_y);
    float loss_x_plus = evaluateTrajectory(vel_x + eps, vel_y);
    float loss_x_minus = evaluateTrajectory(vel_x - eps, vel_y);
    float loss_y_plus = evaluateTrajectory(vel_x, vel_y + eps);
    float loss_y_minus = evaluateTrajectory(vel_x, vel_y - eps);

    float finite_grad_x = (loss_x_plus - loss_x_minus) / (2.0f * eps);
    float finite_grad_y = (loss_y_plus - loss_y_minus) / (2.0f * eps);

    std::cout << "Finite difference loss: " << loss_base << "\n";
    std::cout << "Finite difference gradients: (" << finite_grad_x << ", " << finite_grad_y << ")\n\n";

    // Compare
    float error_x = std::abs(grad_vel_x[1] - finite_grad_x);
    float error_y = std::abs(grad_vel_y[1] - finite_grad_y);
    float rel_error_x = error_x / std::max(std::abs(finite_grad_x), 1e-10f) * 100;
    float rel_error_y = error_y / std::max(std::abs(finite_grad_y), 1e-10f) * 100;

    std::cout << "Gradient errors:\n";
    std::cout << "  X: absolute=" << error_x << ", relative=" << rel_error_x << "%\n";
    std::cout << "  Y: absolute=" << error_y << ", relative=" << rel_error_y << "%\n";

    if (rel_error_x < 5.0f && rel_error_y < 5.0f) {
        std::cout << "\n✓ Gradients match within 5%\n";
    } else {
        std::cout << "\n✗ Gradient mismatch > 5%\n";
    }

    return 0;
}