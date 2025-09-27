#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace physgrad;

int main() {
    std::cout << "Simple Time Step Gradient Debug\n";
    std::cout << "===============================\n\n";

    // Ultra-simple 1-body test
    SimParams params;
    params.num_bodies = 1;
    params.time_step = 0.01f;
    params.G = 1.0f;
    params.epsilon = 0.001f;

    auto sim = std::make_unique<Simulation>(params);
    BodySystem* bodies = sim->getBodies();

    std::cout << "Time step: " << params.time_step << "\n\n";

    // Simple initial conditions - one body at origin with constant acceleration
    std::vector<float> pos_x = {0.0f};
    std::vector<float> pos_y = {0.0f};
    std::vector<float> pos_z = {0.0f};
    std::vector<float> vel_x = {1.0f};  // Simple velocity
    std::vector<float> vel_y = {0.0f};
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

    std::cout << "Initial state:\n";
    std::cout << "pos_x: " << pos_x[0] << ", vel_x: " << vel_x[0] << "\n\n";

    try {
        // Enable gradients
        sim->enableGradients();
        sim->enableParameterGradients(true);

        // Run exactly 1 simulation step
        sim->step();
        std::cout << "Completed 1 simulation step\n";

        // Get final positions and velocities
        std::vector<float> final_pos_x(1), final_pos_y(1), final_pos_z(1);
        bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

        // Also get velocities (need to add this method if it doesn't exist)
        std::vector<float> final_vel_x(1), final_vel_y(1), final_vel_z(1);
        cudaMemcpy(final_vel_x.data(), bodies->d_vel_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(final_vel_y.data(), bodies->d_vel_y, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(final_vel_z.data(), bodies->d_vel_z, size, cudaMemcpyDeviceToHost);

        std::cout << "Final state:\n";
        std::cout << "pos_x: " << final_pos_x[0] << ", vel_x: " << final_vel_x[0] << "\n\n";

        // Manual calculation of what should happen
        float dt = params.time_step;
        float acc_x = 0.0f; // No external forces in this simple case
        float expected_vel_x = vel_x[0] + acc_x * dt;
        float expected_pos_x = pos_x[0] + expected_vel_x * dt;

        std::cout << "Expected (manual calculation):\n";
        std::cout << "acc_x: " << acc_x << "\n";
        std::cout << "expected_vel_x: " << expected_vel_x << "\n";
        std::cout << "expected_pos_x: " << expected_pos_x << "\n\n";

        // Simple target: slightly offset from final position
        std::vector<float> target_pos_x = {final_pos_x[0] + 0.001f};
        std::vector<float> target_pos_y = {0.0f};
        std::vector<float> target_pos_z = {0.0f};

        std::cout << "Target: pos_x = " << target_pos_x[0] << "\n\n";

        // Test time step gradients
        std::vector<float> grad_mass;
        float grad_G, grad_epsilon, grad_dt;

        float loss = sim->computeParameterGradientsWithTime(target_pos_x, target_pos_y, target_pos_z,
                                                           grad_mass, grad_G, grad_epsilon, grad_dt);

        std::cout << "Analytical Results:\n";
        std::cout << "Loss: " << std::fixed << std::setprecision(8) << loss << "\n";
        std::cout << "grad_dt: " << grad_dt << "\n\n";

        // Manual finite difference with larger epsilon for clearer signal
        float dt_eps = 0.0001f;  // Larger epsilon

        std::cout << "Finite Difference Check:\n";
        std::cout << "dt_eps: " << dt_eps << "\n";

        // Forward perturbation: dt + eps
        params.time_step = dt + dt_eps;
        auto sim_plus = std::make_unique<Simulation>(params);
        BodySystem* bodies_plus = sim_plus->getBodies();

        cudaMemcpy(bodies_plus->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_plus->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        sim_plus->step();

        std::vector<float> pos_plus_x(1), pos_plus_y(1), pos_plus_z(1);
        bodies_plus->getPositions(pos_plus_x, pos_plus_y, pos_plus_z);

        // Backward perturbation: dt - eps
        params.time_step = dt - dt_eps;
        auto sim_minus = std::make_unique<Simulation>(params);
        BodySystem* bodies_minus = sim_minus->getBodies();

        cudaMemcpy(bodies_minus->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies_minus->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        sim_minus->step();

        std::vector<float> pos_minus_x(1), pos_minus_y(1), pos_minus_z(1);
        bodies_minus->getPositions(pos_minus_x, pos_minus_y, pos_minus_z);

        std::cout << "pos_plus_x: " << pos_plus_x[0] << "\n";
        std::cout << "pos_minus_x: " << pos_minus_x[0] << "\n";

        // Compute finite difference loss
        float loss_plus = 0.5f * pow(pos_plus_x[0] - target_pos_x[0], 2);
        float loss_minus = 0.5f * pow(pos_minus_x[0] - target_pos_x[0], 2);

        std::cout << "loss_plus: " << loss_plus << "\n";
        std::cout << "loss_minus: " << loss_minus << "\n";

        float finite_diff_grad = (loss_plus - loss_minus) / (2.0f * dt_eps);

        std::cout << "\nComparison:\n";
        std::cout << "Analytical grad_dt: " << grad_dt << "\n";
        std::cout << "Finite diff grad_dt: " << finite_diff_grad << "\n";

        if (std::abs(finite_diff_grad) > 1e-10f) {
            float rel_error = std::abs(grad_dt - finite_diff_grad) / std::abs(finite_diff_grad) * 100;
            std::cout << "Relative error: " << rel_error << "%\n";

            if (rel_error < 5.0f) {
                std::cout << "✓ PASSED (< 5% error)\n";
            } else {
                std::cout << "✗ FAILED (>= 5% error)\n";
            }
        } else {
            std::cout << "Warning: finite difference too small to compare\n";
        }

    } catch (const std::exception& e) {
        std::cout << "✗ Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}