#include "simulation.h"
#include <iostream>

using namespace physgrad;

int main() {
    std::cout << "Simple Parameter Gradient Test\n";
    std::cout << "==============================\n";

    // Very simple test
    SimParams params;
    params.num_bodies = 2;
    params.time_step = 0.01f;
    params.G = 1.0f;
    params.epsilon = 0.001f;

    auto sim = std::make_unique<Simulation>(params);
    BodySystem* bodies = sim->getBodies();

    std::cout << "1. Created simulation with " << params.num_bodies << " bodies\n";

    // Simple initial conditions
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

    std::cout << "2. Set initial conditions\n";

    // Enable gradients
    try {
        sim->enableGradients();
        std::cout << "3. Enabled gradients\n";

        sim->enableParameterGradients(true);
        std::cout << "4. Enabled parameter gradients\n";

        // Run just 1 step
        sim->step();
        std::cout << "5. Completed 1 simulation step\n";

        // Simple target (current position)
        std::vector<float> final_pos_x(2), final_pos_y(2), final_pos_z(2);
        bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

        std::vector<float> target_pos_x = {final_pos_x[0] + 0.01f, final_pos_x[1] - 0.01f};
        std::vector<float> target_pos_y = {final_pos_y[0] + 0.01f, final_pos_y[1] - 0.01f};
        std::vector<float> target_pos_z = {0.0f, 0.0f};

        std::cout << "6. Set targets\n";

        // Test parameter gradients
        std::vector<float> grad_mass;
        float grad_G, grad_epsilon;

        std::cout << "7. About to compute parameter gradients...\n";

        float loss = sim->computeParameterGradients(target_pos_x, target_pos_y, target_pos_z,
                                                   grad_mass, grad_G, grad_epsilon);

        std::cout << "8. Parameter gradients computed!\n";
        std::cout << "Loss: " << loss << "\n";
        std::cout << "grad_G: " << grad_G << "\n";
        std::cout << "grad_epsilon: " << grad_epsilon << "\n";

        if (grad_mass.size() >= 2) {
            std::cout << "grad_mass[0]: " << grad_mass[0] << "\n";
            std::cout << "grad_mass[1]: " << grad_mass[1] << "\n";
        }

        std::cout << "✓ Test completed successfully!\n";

    } catch (const std::exception& e) {
        std::cout << "✗ Exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cout << "✗ Unknown exception occurred\n";
        return 1;
    }

    return 0;
}