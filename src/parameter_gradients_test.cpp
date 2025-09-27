#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace physgrad;

int main() {
    std::cout << "PhysGrad Parameter Gradients Test\n";
    std::cout << "=================================\n\n";

    // Simple 2-body system for testing
    SimParams params;
    params.num_bodies = 2;
    params.time_step = 0.01f;
    params.G = 1.0f;
    params.epsilon = 0.001f;

    auto sim = std::make_unique<Simulation>(params);
    BodySystem* bodies = sim->getBodies();

    // Set up simple binary system
    std::vector<float> init_pos_x = {-0.5f, 0.5f};
    std::vector<float> init_pos_y = {0.0f, 0.0f};
    std::vector<float> init_pos_z = {0.0f, 0.0f};
    std::vector<float> init_vel_x = {0.0f, 0.0f};
    std::vector<float> init_vel_y = {0.3f, -0.3f};
    std::vector<float> init_vel_z = {0.0f, 0.0f};
    std::vector<float> masses = {1.0f, 1.0f};

    size_t size = bodies->n * sizeof(float);
    cudaMemcpy(bodies->d_pos_x, init_pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_y, init_pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_pos_z, init_pos_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_x, init_vel_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_y, init_vel_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_vel_z, init_vel_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

    std::cout << "Initial setup:\n";
    std::cout << "- Bodies: " << params.num_bodies << "\n";
    std::cout << "- G: " << params.G << "\n";
    std::cout << "- Epsilon: " << params.epsilon << "\n";
    std::cout << "- Time step: " << params.time_step << "\n\n";

    // Enable gradients
    sim->enableGradients();
    sim->enableParameterGradients(true);

    // Run simulation for a few steps
    const int num_steps = 5;
    for (int step = 0; step < num_steps; step++) {
        sim->step();
    }

    // Get final positions
    std::vector<float> final_pos_x(params.num_bodies), final_pos_y(params.num_bodies), final_pos_z(params.num_bodies);
    bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

    std::cout << "Final positions after " << num_steps << " steps:\n";
    for (int i = 0; i < params.num_bodies; i++) {
        std::cout << "Body " << i << ": (" << final_pos_x[i] << ", " << final_pos_y[i] << ", " << final_pos_z[i] << ")\n";
    }
    std::cout << "\n";

    // Create target positions (slightly different from final)
    std::vector<float> target_pos_x = final_pos_x;
    std::vector<float> target_pos_y = final_pos_y;
    std::vector<float> target_pos_z = final_pos_z;

    // Perturb targets slightly
    for (int i = 0; i < params.num_bodies; i++) {
        target_pos_x[i] += 0.01f * (i % 2 == 0 ? 1 : -1);
        target_pos_y[i] += 0.01f * (i % 2 == 0 ? 1 : -1);
    }

    std::cout << "Target positions:\n";
    for (int i = 0; i < params.num_bodies; i++) {
        std::cout << "Body " << i << ": (" << target_pos_x[i] << ", " << target_pos_y[i] << ", " << target_pos_z[i] << ")\n";
    }
    std::cout << "\n";

    // Compute parameter gradients
    std::vector<float> grad_mass;
    float grad_G, grad_epsilon;

    float loss = sim->computeParameterGradients(target_pos_x, target_pos_y, target_pos_z,
                                               grad_mass, grad_G, grad_epsilon);

    std::cout << "Parameter Gradient Results:\n";
    std::cout << "===========================\n";
    std::cout << "Loss: " << loss << "\n\n";

    std::cout << "Gradients w.r.t. masses:\n";
    for (int i = 0; i < params.num_bodies; i++) {
        std::cout << "  grad_mass[" << i << "] = " << grad_mass[i] << "\n";
    }
    std::cout << "\n";

    std::cout << "Gradient w.r.t. G: " << grad_G << "\n";
    std::cout << "Gradient w.r.t. epsilon: " << grad_epsilon << "\n\n";

    // Validate gradients are non-zero and finite
    bool gradients_valid = true;

    if (std::isnan(grad_G) || std::isinf(grad_G)) {
        std::cout << "⚠ WARNING: grad_G is not finite\n";
        gradients_valid = false;
    }

    if (std::isnan(grad_epsilon) || std::isinf(grad_epsilon)) {
        std::cout << "⚠ WARNING: grad_epsilon is not finite\n";
        gradients_valid = false;
    }

    for (int i = 0; i < params.num_bodies; i++) {
        if (std::isnan(grad_mass[i]) || std::isinf(grad_mass[i])) {
            std::cout << "⚠ WARNING: grad_mass[" << i << "] is not finite\n";
            gradients_valid = false;
        }
    }

    // Check if gradients have reasonable magnitudes
    float total_grad_norm = grad_G * grad_G + grad_epsilon * grad_epsilon;
    for (float g : grad_mass) {
        total_grad_norm += g * g;
    }
    total_grad_norm = std::sqrt(total_grad_norm);

    std::cout << "Total gradient norm: " << total_grad_norm << "\n";

    if (total_grad_norm > 1e-10f && total_grad_norm < 1e6f) {
        std::cout << "✓ Gradient magnitudes are reasonable\n";
    } else if (total_grad_norm < 1e-10f) {
        std::cout << "⚠ WARNING: Gradients are very small (may be zero)\n";
        gradients_valid = false;
    } else {
        std::cout << "⚠ WARNING: Gradients are very large\n";
        gradients_valid = false;
    }

    std::cout << "\n";

    if (gradients_valid) {
        std::cout << "✓ Parameter gradients computed successfully!\n";
        std::cout << "The system can now differentiate through simulation parameters.\n";
        std::cout << "This enables:\n";
        std::cout << "- Learning physical constants from observations\n";
        std::cout << "- Parameter estimation and system identification\n";
        std::cout << "- Physics-informed optimization\n";
    } else {
        std::cout << "✗ Parameter gradient computation has issues\n";
        std::cout << "Check the adjoint kernel implementation\n";
    }

    // Simple finite difference check for G
    std::cout << "\nFinite Difference Validation for G:\n";
    std::cout << "===================================\n";

    float eps = 1e-5f;

    // Forward perturbation
    auto sim_plus = std::make_unique<Simulation>(params);
    sim_plus->getBodies();
    SimParams params_plus = params;
    params_plus.G = params.G + eps;
    sim_plus = std::make_unique<Simulation>(params_plus);

    size = sim_plus->getBodies()->n * sizeof(float);
    cudaMemcpy(sim_plus->getBodies()->d_pos_x, init_pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_plus->getBodies()->d_pos_y, init_pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_plus->getBodies()->d_pos_z, init_pos_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_plus->getBodies()->d_vel_x, init_vel_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_plus->getBodies()->d_vel_y, init_vel_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_plus->getBodies()->d_vel_z, init_vel_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_plus->getBodies()->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

    sim_plus->enableGradients();
    for (int step = 0; step < num_steps; step++) {
        sim_plus->step();
    }

    float loss_plus = sim_plus->computeGradients(target_pos_x, target_pos_y, target_pos_z);

    // Backward perturbation
    auto sim_minus = std::make_unique<Simulation>(params);
    SimParams params_minus = params;
    params_minus.G = params.G - eps;
    sim_minus = std::make_unique<Simulation>(params_minus);

    cudaMemcpy(sim_minus->getBodies()->d_pos_x, init_pos_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_minus->getBodies()->d_pos_y, init_pos_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_minus->getBodies()->d_pos_z, init_pos_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_minus->getBodies()->d_vel_x, init_vel_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_minus->getBodies()->d_vel_y, init_vel_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_minus->getBodies()->d_vel_z, init_vel_z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(sim_minus->getBodies()->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

    sim_minus->enableGradients();
    for (int step = 0; step < num_steps; step++) {
        sim_minus->step();
    }

    float loss_minus = sim_minus->computeGradients(target_pos_x, target_pos_y, target_pos_z);

    float finite_diff_grad_G = (loss_plus - loss_minus) / (2.0f * eps);

    std::cout << "Finite difference grad_G: " << finite_diff_grad_G << "\n";
    std::cout << "Adjoint grad_G: " << grad_G << "\n";

    if (std::abs(finite_diff_grad_G) > 1e-10f) {
        float relative_error = std::abs(grad_G - finite_diff_grad_G) / std::abs(finite_diff_grad_G);
        std::cout << "Relative error: " << relative_error << "\n";

        if (relative_error < 0.1f) {
            std::cout << "✓ Excellent agreement with finite differences!\n";
        } else if (relative_error < 0.5f) {
            std::cout << "✓ Good agreement with finite differences\n";
        } else {
            std::cout << "⚠ Poor agreement with finite differences\n";
        }
    } else {
        std::cout << "⚠ Finite difference gradient too small for comparison\n";
    }

    return 0;
}