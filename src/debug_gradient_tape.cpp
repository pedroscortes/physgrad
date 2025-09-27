#include "simulation.h"
#include <iostream>

using namespace physgrad;

int main() {
    std::cout << "Debugging Gradient Tape\n";
    std::cout << "======================\n\n";

    // Minimal case: 1 body, 1 step, no forces
    SimParams params;
    params.num_bodies = 1;
    params.time_step = 0.1f;
    params.G = 0.0f;  // No forces
    params.epsilon = 0.001f;

    auto sim = std::make_unique<Simulation>(params);
    sim->enableGradients();
    BodySystem* bodies = sim->getBodies();

    // Initial state: position (0), velocity (1)
    std::vector<float> pos_x = {0.0f};
    std::vector<float> pos_y = {0.0f};
    std::vector<float> pos_z = {0.0f};
    std::vector<float> vel_x = {1.0f};  // This is what we'll differentiate w.r.t.
    std::vector<float> vel_y = {1.0f};
    std::vector<float> vel_z = {0.0f};
    std::vector<float> masses = {1.0f};

    size_t size = bodies->n * sizeof(float);
    cudaMemcpy(bodies->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

    std::cout << "Initial state: pos=(0,0), vel=(1,1)\n";

    // One step: pos_new = pos + vel * dt = 0 + 1 * 0.1 = 0.1
    sim->step();

    std::vector<float> final_pos_x(1), final_pos_y(1), final_pos_z(1);
    bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

    std::cout << "After 1 step: pos=(" << final_pos_x[0] << "," << final_pos_y[0] << ")\n";
    std::cout << "Expected: pos=(0.1, 0.1)\n\n";

    // Target at (0.2, 0.2) - loss should be (0.1-0.2)^2 + (0.1-0.2)^2 = 0.02
    std::vector<float> target_x = {0.2f};
    std::vector<float> target_y = {0.2f};
    std::vector<float> target_z = {0.0f};

    std::cout << "Target: (0.2, 0.2)\n";
    std::cout << "Expected loss: 0.5 * ((0.1-0.2)^2 + (0.1-0.2)^2) = 0.01\n\n";

    float loss = sim->computeGradients(target_x, target_y, target_z);

    std::vector<float> grad_vel_x(1), grad_vel_y(1);
    cudaMemcpy(grad_vel_x.data(), bodies->d_grad_vel_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_vel_y.data(), bodies->d_grad_vel_y, size, cudaMemcpyDeviceToHost);

    std::cout << "Computed loss: " << loss << "\n";
    std::cout << "Gradients w.r.t. initial velocity: (" << grad_vel_x[0] << ", " << grad_vel_y[0] << ")\n";

    // For ballistic motion: pos_final = pos_initial + vel_initial * dt
    // Loss = 0.5 * (pos_final - target)^2 = 0.5 * (pos_initial + vel_initial * dt - target)^2
    // d(Loss)/d(vel_initial) = (pos_initial + vel_initial * dt - target) * dt
    // = (0 + 1 * 0.1 - 0.2) * 0.1 = (-0.1) * 0.1 = -0.01
    std::cout << "Expected gradient: (-0.01, -0.01)\n";

    float expected_grad = (0.1f - 0.2f) * 0.1f;
    std::cout << "Expected: " << expected_grad << ", Got: " << grad_vel_x[0] << "\n";

    if (std::abs(grad_vel_x[0] - expected_grad) < 1e-6f &&
        std::abs(grad_vel_y[0] - expected_grad) < 1e-6f) {
        std::cout << "✓ Gradients correct!\n";
    } else {
        std::cout << "✗ Gradient mismatch\n";
    }

    return 0;
}