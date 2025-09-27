#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace physgrad;

int main() {
    std::cout << "Time Step Gradient with Forces Debug\n";
    std::cout << "====================================\n\n";

    // 2-body test with forces, but only 1 step
    SimParams params;
    params.num_bodies = 2;
    params.time_step = 0.01f;
    params.G = 1.0f;
    params.epsilon = 0.001f;

    auto sim = std::make_unique<Simulation>(params);
    BodySystem* bodies = sim->getBodies();

    std::cout << "Time step: " << params.time_step << "\n";
    std::cout << "G: " << params.G << ", epsilon: " << params.epsilon << "\n\n";

    // Simple 2-body setup
    std::vector<float> pos_x = {-0.1f, 0.1f};
    std::vector<float> pos_y = {0.0f, 0.0f};
    std::vector<float> pos_z = {0.0f, 0.0f};
    std::vector<float> vel_x = {0.0f, 0.0f};
    std::vector<float> vel_y = {0.1f, -0.1f};
    std::vector<float> vel_z = {0.0f, 0.0f};
    std::vector<float> masses = {1.0f, 1.0f};

    size_t size = bodies->n * sizeof(float);
    cudaMemcpy(bodies->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

    std::cout << "Initial state:\n";
    std::cout << "Body 0: pos=(" << pos_x[0] << ", " << pos_y[0] << "), vel=(" << vel_x[0] << ", " << vel_y[0] << ")\n";
    std::cout << "Body 1: pos=(" << pos_x[1] << ", " << pos_y[1] << "), vel=(" << vel_x[1] << ", " << vel_y[1] << ")\n\n";

    try {
        // Enable gradients
        sim->enableGradients();
        sim->enableParameterGradients(true);

        // Run exactly 1 simulation step
        sim->step();
        std::cout << "Completed 1 simulation step\n";

        // Get final positions
        std::vector<float> final_pos_x(2), final_pos_y(2), final_pos_z(2);
        bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

        std::cout << "Final state:\n";
        std::cout << "Body 0: pos=(" << final_pos_x[0] << ", " << final_pos_y[0] << ")\n";
        std::cout << "Body 1: pos=(" << final_pos_x[1] << ", " << final_pos_y[1] << ")\n\n";

        // Create targets
        std::vector<float> target_pos_x = {final_pos_x[0] + 0.001f, final_pos_x[1] - 0.001f};
        std::vector<float> target_pos_y = {final_pos_y[0] + 0.001f, final_pos_y[1] - 0.001f};
        std::vector<float> target_pos_z = {0.0f, 0.0f};

        std::cout << "Targets:\n";
        std::cout << "Body 0: target=(" << target_pos_x[0] << ", " << target_pos_y[0] << ")\n";
        std::cout << "Body 1: target=(" << target_pos_x[1] << ", " << target_pos_y[1] << ")\n\n";

        // Test time step gradients
        std::vector<float> grad_mass;
        float grad_G, grad_epsilon, grad_dt;

        float loss = sim->computeParameterGradientsWithTime(target_pos_x, target_pos_y, target_pos_z,
                                                           grad_mass, grad_G, grad_epsilon, grad_dt);

        std::cout << "Analytical Results:\n";
        std::cout << "Loss: " << std::fixed << std::setprecision(8) << loss << "\n";
        std::cout << "grad_dt: " << grad_dt << "\n";
        std::cout << "grad_G: " << grad_G << "\n";
        std::cout << "grad_epsilon: " << grad_epsilon << "\n\n";

        // Finite difference for dt
        float dt_eps = 0.001f;  // Larger perturbation for better numerical precision
        float original_dt = params.time_step;

        std::cout << "Finite Difference Check (dt):\n";

        // Plus perturbation
        params.time_step = original_dt + dt_eps;
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

        std::vector<float> pos_plus_x(2), pos_plus_y(2), pos_plus_z(2);
        bodies_plus->getPositions(pos_plus_x, pos_plus_y, pos_plus_z);

        // Minus perturbation
        params.time_step = original_dt - dt_eps;
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

        std::vector<float> pos_minus_x(2), pos_minus_y(2), pos_minus_z(2);
        bodies_minus->getPositions(pos_minus_x, pos_minus_y, pos_minus_z);

        // Compute losses
        float loss_plus = 0.0f, loss_minus = 0.0f;
        for (int i = 0; i < 2; i++) {
            float dx_plus = pos_plus_x[i] - target_pos_x[i];
            float dy_plus = pos_plus_y[i] - target_pos_y[i];
            loss_plus += 0.5f * (dx_plus*dx_plus + dy_plus*dy_plus);

            float dx_minus = pos_minus_x[i] - target_pos_x[i];
            float dy_minus = pos_minus_y[i] - target_pos_y[i];
            loss_minus += 0.5f * (dx_minus*dx_minus + dy_minus*dy_minus);
        }

        float finite_diff_grad = (loss_plus - loss_minus) / (2.0f * dt_eps);

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