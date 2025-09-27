#include "simulation.h"
#include <iostream>
#include <iomanip>

using namespace physgrad;

int main() {
    std::cout << "Time Step Gradient Test\n";
    std::cout << "======================\n\n";

    // Simple 2-body test
    SimParams params;
    params.num_bodies = 2;
    params.time_step = 0.01f;
    params.G = 1.0f;
    params.epsilon = 0.001f;

    auto sim = std::make_unique<Simulation>(params);
    BodySystem* bodies = sim->getBodies();

    std::cout << "1. Created simulation with time step: " << params.time_step << "\n";

    // Simple initial conditions
    std::vector<float> pos_x = {-0.2f, 0.2f};
    std::vector<float> pos_y = {0.0f, 0.0f};
    std::vector<float> pos_z = {0.0f, 0.0f};
    std::vector<float> vel_x = {0.0f, 0.0f};
    std::vector<float> vel_y = {0.2f, -0.2f};
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

    std::cout << "2. Set initial conditions\n";

    try {
        // Enable gradients
        sim->enableGradients();
        sim->enableParameterGradients(true);
        std::cout << "3. Enabled gradients and parameter gradients\n";

        // Run a few simulation steps
        const int num_steps = 3;
        for (int i = 0; i < num_steps; i++) {
            sim->step();
        }
        std::cout << "4. Completed " << num_steps << " simulation steps\n";

        // Get final positions
        std::vector<float> final_pos_x(2), final_pos_y(2), final_pos_z(2);
        bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

        // Create targets slightly offset from current positions
        std::vector<float> target_pos_x = {final_pos_x[0] + 0.005f, final_pos_x[1] - 0.005f};
        std::vector<float> target_pos_y = {final_pos_y[0] + 0.005f, final_pos_y[1] - 0.005f};
        std::vector<float> target_pos_z = {0.0f, 0.0f};

        std::cout << "5. Set targets\n";

        // Test time step gradients
        std::vector<float> grad_mass;
        float grad_G, grad_epsilon, grad_dt;

        std::cout << "6. Computing parameter gradients with time step...\n";

        float loss = sim->computeParameterGradientsWithTime(target_pos_x, target_pos_y, target_pos_z,
                                                           grad_mass, grad_G, grad_epsilon, grad_dt);

        std::cout << "7. Time step gradients computed!\n\n";

        std::cout << "Results:\n";
        std::cout << "========\n";
        std::cout << "Loss: " << std::fixed << std::setprecision(8) << loss << "\n";
        std::cout << "grad_G: " << grad_G << "\n";
        std::cout << "grad_epsilon: " << grad_epsilon << "\n";
        std::cout << "grad_dt: " << grad_dt << "\n";

        if (grad_mass.size() >= 2) {
            std::cout << "grad_mass[0]: " << grad_mass[0] << "\n";
            std::cout << "grad_mass[1]: " << grad_mass[1] << "\n";
        }

        // Finite difference verification for time step gradient
        std::cout << "\nFinite Difference Verification:\n";
        std::cout << "===============================\n";

        float dt_eps = 1e-6f;

        // Forward perturbation
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

        for (int i = 0; i < num_steps; i++) {
            sim_plus->step();
        }

        std::vector<float> pos_plus_x(2), pos_plus_y(2), pos_plus_z(2);
        bodies_plus->getPositions(pos_plus_x, pos_plus_y, pos_plus_z);

        // Backward perturbation
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

        for (int i = 0; i < num_steps; i++) {
            sim_minus->step();
        }

        std::vector<float> pos_minus_x(2), pos_minus_y(2), pos_minus_z(2);
        bodies_minus->getPositions(pos_minus_x, pos_minus_y, pos_minus_z);

        // Compute finite difference loss
        float loss_plus = 0.0f, loss_minus = 0.0f;
        for (int i = 0; i < 2; i++) {
            float dx_plus = pos_plus_x[i] - target_pos_x[i];
            float dy_plus = pos_plus_y[i] - target_pos_y[i];
            float dz_plus = pos_plus_z[i] - target_pos_z[i];
            loss_plus += 0.5f * (dx_plus*dx_plus + dy_plus*dy_plus + dz_plus*dz_plus);

            float dx_minus = pos_minus_x[i] - target_pos_x[i];
            float dy_minus = pos_minus_y[i] - target_pos_y[i];
            float dz_minus = pos_minus_z[i] - target_pos_z[i];
            loss_minus += 0.5f * (dx_minus*dx_minus + dy_minus*dy_minus + dz_minus*dz_minus);
        }

        float finite_diff_grad = (loss_plus - loss_minus) / (2.0f * dt_eps);

        std::cout << "Analytical grad_dt: " << grad_dt << "\n";
        std::cout << "Finite diff grad_dt: " << finite_diff_grad << "\n";
        std::cout << "Relative error: " << std::abs(grad_dt - finite_diff_grad) / std::max(std::abs(grad_dt), std::abs(finite_diff_grad)) * 100 << "%\n";

        if (std::abs(grad_dt - finite_diff_grad) / std::max(std::abs(grad_dt), std::abs(finite_diff_grad)) < 0.01f) {
            std::cout << "✓ Time step gradient verification PASSED!\n";
        } else {
            std::cout << "✗ Time step gradient verification FAILED!\n";
        }

    } catch (const std::exception& e) {
        std::cout << "✗ Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}