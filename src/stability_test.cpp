#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

using namespace physgrad;

int main() {
    std::cout << "PhysGrad Numerical Stability Test\n";
    std::cout << "=================================\n\n";

    // Create a challenging scenario with close bodies
    SimParams params;
    params.num_bodies = 4;
    params.time_step = 0.005f;  // Smaller time step for stability
    params.G = 1.0f;
    params.epsilon = 0.0001f;   // Small epsilon to challenge stability
    params.max_force = 50.0f;   // Force clamping

    // Stability parameters
    params.stability.gradient_clipping_threshold = 5.0f;
    params.stability.use_gradient_clipping = true;
    params.stability.use_loss_regularization = true;

    auto sim_normal = std::make_unique<Simulation>(params);
    auto sim_stable = std::make_unique<Simulation>(params);

    // Set up a challenging configuration with very close bodies
    std::vector<float> init_pos_x = {-0.01f, 0.01f, -0.02f, 0.02f};
    std::vector<float> init_pos_y = {-0.01f, 0.01f, 0.02f, -0.02f};
    std::vector<float> init_pos_z = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> init_vel_x = {0.1f, -0.1f, 0.05f, -0.05f};
    std::vector<float> init_vel_y = {0.1f, -0.1f, -0.05f, 0.05f};
    std::vector<float> init_vel_z = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> masses = {1.0f, 1.0f, 0.5f, 0.5f};

    // Setup both simulations with same initial conditions
    auto setupSim = [&](std::unique_ptr<Simulation>& sim) {
        BodySystem* bodies = sim->getBodies();
        size_t size = bodies->n * sizeof(float);

        cudaMemcpy(bodies->d_pos_x, init_pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, init_pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, init_pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_x, init_vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, init_vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, init_vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);
    };

    setupSim(sim_normal);
    setupSim(sim_stable);

    // Enable stable forces for one simulation
    sim_stable->enableStableForces(true);

    std::cout << "Configuration:\n";
    std::cout << "- Bodies: " << params.num_bodies << " (close proximity)\n";
    std::cout << "- Time step: " << params.time_step << "\n";
    std::cout << "- Epsilon: " << params.epsilon << " (very small)\n";
    std::cout << "- Max force (stable): " << params.max_force << "\n\n";

    std::cout << "Comparing normal vs stabilized force computation:\n";
    std::cout << std::setw(6) << "Step"
              << std::setw(15) << "Normal Energy"
              << std::setw(15) << "Stable Energy"
              << std::setw(15) << "Energy Ratio" << "\n";
    std::cout << std::string(60, '-') << "\n";

    const int num_steps = 50;
    bool normal_diverged = false;
    bool stable_diverged = false;

    for (int step = 0; step < num_steps; step++) {
        // Step both simulations
        if (!normal_diverged) {
            try {
                sim_normal->step();
            } catch (...) {
                normal_diverged = true;
            }
        }

        if (!stable_diverged) {
            try {
                sim_stable->step();
            } catch (...) {
                stable_diverged = true;
            }
        }

        if (step % 5 == 0) {
            float normal_energy = std::numeric_limits<float>::infinity();
            float stable_energy = std::numeric_limits<float>::infinity();

            if (!normal_diverged) {
                normal_energy = sim_normal->getBodies()->computeEnergy(params);
                // Check for NaN or extremely large values
                if (std::isnan(normal_energy) || std::abs(normal_energy) > 1e6f) {
                    normal_diverged = true;
                    normal_energy = std::numeric_limits<float>::infinity();
                }
            }

            if (!stable_diverged) {
                stable_energy = sim_stable->getBodies()->computeEnergy(params);
                if (std::isnan(stable_energy) || std::abs(stable_energy) > 1e6f) {
                    stable_diverged = true;
                    stable_energy = std::numeric_limits<float>::infinity();
                }
            }

            float ratio = 1.0f;
            if (!normal_diverged && !stable_diverged && std::abs(normal_energy) > 1e-10f) {
                ratio = stable_energy / normal_energy;
            }

            std::cout << std::setw(6) << step
                      << std::setw(15) << std::fixed << std::setprecision(6);

            if (normal_diverged) {
                std::cout << "DIVERGED";
            } else {
                std::cout << normal_energy;
            }

            std::cout << std::setw(15);
            if (stable_diverged) {
                std::cout << "DIVERGED";
            } else {
                std::cout << stable_energy;
            }

            std::cout << std::setw(15);
            if (normal_diverged || stable_diverged) {
                std::cout << "N/A";
            } else {
                std::cout << ratio;
            }
            std::cout << "\n";
        }

        if (normal_diverged && stable_diverged) {
            std::cout << "Both simulations diverged at step " << step << "\n";
            break;
        }
    }

    std::cout << "\n";

    if (normal_diverged && !stable_diverged) {
        std::cout << "✓ SUCCESS: Stable forces prevented divergence while normal forces failed\n";
    } else if (!normal_diverged && !stable_diverged) {
        std::cout << "✓ Both simulations remained stable (increase challenge for better test)\n";
    } else if (stable_diverged && !normal_diverged) {
        std::cout << "⚠ WARNING: Stable forces diverged while normal forces didn't\n";
    } else {
        std::cout << "⚠ Both simulations diverged (may need better parameters)\n";
    }

    // Test gradient stability
    std::cout << "\nTesting gradient stability:\n";
    std::cout << "============================\n";

    // Reset simulations for gradient test
    setupSim(sim_normal);
    setupSim(sim_stable);

    sim_normal->enableGradients();
    sim_stable->enableGradients();
    sim_stable->enableStableForces(true);

    // Run a few steps
    for (int step = 0; step < 5; step++) {
        sim_normal->step();
        sim_stable->step();
    }

    // Create target positions
    std::vector<float> target_pos_x(params.num_bodies, 0.0f);
    std::vector<float> target_pos_y(params.num_bodies, 0.0f);
    std::vector<float> target_pos_z(params.num_bodies, 0.0f);

    // Compute gradients
    float loss_normal = sim_normal->computeGradients(target_pos_x, target_pos_y, target_pos_z);
    float loss_stable = sim_stable->computeGradients(target_pos_x, target_pos_y, target_pos_z);

    // Check gradient magnitudes
    std::vector<float> grad_normal_x(params.num_bodies), grad_normal_y(params.num_bodies), grad_normal_z(params.num_bodies);
    std::vector<float> grad_stable_x(params.num_bodies), grad_stable_y(params.num_bodies), grad_stable_z(params.num_bodies);

    sim_normal->getBodies()->getGradients(grad_normal_x, grad_normal_y, grad_normal_z);
    sim_stable->getBodies()->getGradients(grad_stable_x, grad_stable_y, grad_stable_z);

    auto computeGradNorm = [](const std::vector<float>& gx, const std::vector<float>& gy, const std::vector<float>& gz) {
        float norm = 0.0f;
        for (size_t i = 0; i < gx.size(); i++) {
            norm += gx[i]*gx[i] + gy[i]*gy[i] + gz[i]*gz[i];
        }
        return std::sqrt(norm);
    };

    float normal_grad_norm = computeGradNorm(grad_normal_x, grad_normal_y, grad_normal_z);
    float stable_grad_norm = computeGradNorm(grad_stable_x, grad_stable_y, grad_stable_z);

    std::cout << "Loss (normal): " << loss_normal << "\n";
    std::cout << "Loss (stable): " << loss_stable << "\n";
    std::cout << "Gradient norm (normal): " << normal_grad_norm << "\n";
    std::cout << "Gradient norm (stable): " << stable_grad_norm << "\n";

    if (std::isnan(normal_grad_norm) || normal_grad_norm > 1e6f) {
        std::cout << "✓ Normal gradients are unstable (NaN or very large)\n";
    }
    if (!std::isnan(stable_grad_norm) && stable_grad_norm < 1e6f) {
        std::cout << "✓ Stable gradients remain finite\n";
    }

    std::cout << "\nStability improvements test completed.\n";

    return 0;
}