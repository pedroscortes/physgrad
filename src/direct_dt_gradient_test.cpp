#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace physgrad;

int main() {
    std::cout << "Direct Time Step Gradient Test (No Tape)\n";
    std::cout << "========================================\n\n";

    // 2-body test with forces, but compute gradient directly
    SimParams params;
    params.num_bodies = 2;
    params.time_step = 0.01f;
    params.G = 1.0f;
    params.epsilon = 0.001f;

    auto sim = std::make_unique<Simulation>(params);
    BodySystem* bodies = sim->getBodies();

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

    std::cout << "Computing time step gradient manually (no tape)...\n\n";

    try {
        // Enable gradients but NOT parameter gradients (we'll do it manually)
        sim->enableGradients();

        // Use public API - just run one simulation step
        std::cout << "1. Running one simulation step...\n";

        // Store initial velocities before the step
        std::vector<float> initial_vel_x(2), initial_vel_y(2), initial_vel_z(2);
        cudaMemcpy(initial_vel_x.data(), bodies->d_vel_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(initial_vel_y.data(), bodies->d_vel_y, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(initial_vel_z.data(), bodies->d_vel_z, size, cudaMemcpyDeviceToHost);

        sim->step();

        // Get acceleration values - these were computed during the step
        std::vector<float> acc_x(2), acc_y(2), acc_z(2);
        cudaMemcpy(acc_x.data(), bodies->d_acc_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(acc_y.data(), bodies->d_acc_y, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(acc_z.data(), bodies->d_acc_z, size, cudaMemcpyDeviceToHost);

        std::cout << "Accelerations used:\n";
        std::cout << "Body 0: acc=(" << acc_x[0] << ", " << acc_y[0] << ")\n";
        std::cout << "Body 1: acc=(" << acc_x[1] << ", " << acc_y[1] << ")\n\n";

        // Get final positions and velocities
        std::vector<float> final_pos_x(2), final_pos_y(2), final_pos_z(2);
        std::vector<float> final_vel_x(2), final_vel_y(2), final_vel_z(2);

        cudaMemcpy(final_pos_x.data(), bodies->d_pos_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(final_pos_y.data(), bodies->d_pos_y, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(final_pos_z.data(), bodies->d_pos_z, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(final_vel_x.data(), bodies->d_vel_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(final_vel_y.data(), bodies->d_vel_y, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(final_vel_z.data(), bodies->d_vel_z, size, cudaMemcpyDeviceToHost);

        std::cout << "Final state:\n";
        std::cout << "Body 0: pos=(" << final_pos_x[0] << ", " << final_pos_y[0] << "), vel=(" << final_vel_x[0] << ", " << final_vel_y[0] << ")\n";
        std::cout << "Body 1: pos=(" << final_pos_x[1] << ", " << final_pos_y[1] << "), vel=(" << final_vel_x[1] << ", " << final_vel_y[1] << ")\n\n";

        // Create simple targets
        std::vector<float> target_pos_x = {final_pos_x[0] + 0.001f, final_pos_x[1] - 0.001f};
        std::vector<float> target_pos_y = {final_pos_y[0] + 0.001f, final_pos_y[1] - 0.001f};

        // Compute loss and gradients w.r.t. final positions
        float loss = 0.0f;
        std::vector<float> grad_pos_x(2), grad_pos_y(2);

        for (int i = 0; i < 2; i++) {
            float dx = final_pos_x[i] - target_pos_x[i];
            float dy = final_pos_y[i] - target_pos_y[i];
            loss += 0.5f * (dx*dx + dy*dy);

            grad_pos_x[i] = dx;
            grad_pos_y[i] = dy;
        }

        std::cout << "Loss: " << loss << "\n";
        std::cout << "grad_pos: [(" << grad_pos_x[0] << ", " << grad_pos_y[0] << "), (" << grad_pos_x[1] << ", " << grad_pos_y[1] << ")]\n\n";

        // Manual time step gradient computation
        std::cout << "3. Computing time step gradient manually...\n";

        float grad_dt_manual = 0.0f;

        for (int i = 0; i < 2; i++) {
            // From velocity update: ∂vel_new/∂dt = acc
            // Contribution: grad_vel_final * acc (but we don't have grad_vel_final directly)

            // From position update: ∂pos_new/∂dt = vel_new + acc * dt
            // Contribution: grad_pos_final * (vel_new + acc * dt)
            grad_dt_manual += grad_pos_x[i] * (final_vel_x[i] + acc_x[i] * params.time_step);
            grad_dt_manual += grad_pos_y[i] * (final_vel_y[i] + acc_y[i] * params.time_step);
        }

        std::cout << "Manual grad_dt: " << grad_dt_manual << "\n\n";

        // Finite difference verification
        std::cout << "4. Finite difference verification...\n";

        float dt_eps = 0.001f;

        // Plus perturbation
        params.time_step += dt_eps;
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

        std::vector<float> pos_plus_x(2), pos_plus_y(2);
        cudaMemcpy(pos_plus_x.data(), bodies_plus->d_pos_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(pos_plus_y.data(), bodies_plus->d_pos_y, size, cudaMemcpyDeviceToHost);

        // Minus perturbation
        params.time_step -= 2.0f * dt_eps;
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

        std::vector<float> pos_minus_x(2), pos_minus_y(2);
        cudaMemcpy(pos_minus_x.data(), bodies_minus->d_pos_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(pos_minus_y.data(), bodies_minus->d_pos_y, size, cudaMemcpyDeviceToHost);

        // Compute finite difference losses
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

        std::cout << "Manual grad_dt: " << grad_dt_manual << "\n";
        std::cout << "Finite diff grad_dt: " << finite_diff_grad << "\n";

        if (std::abs(finite_diff_grad) > 1e-10f) {
            float rel_error = std::abs(grad_dt_manual - finite_diff_grad) / std::abs(finite_diff_grad) * 100;
            std::cout << "Relative error: " << rel_error << "%\n";

            if (rel_error < 5.0f) {
                std::cout << "✓ PASSED (< 5% error)\n";
            } else {
                std::cout << "✗ FAILED (>= 5% error)\n";
            }
        }

    } catch (const std::exception& e) {
        std::cout << "✗ Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}