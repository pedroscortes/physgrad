#include "simulation.h"
#include "optimizers.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>
#include <chrono>

using namespace physgrad;

class OptimizedTrajectoryOptimizer {
private:
    SimParams params;
    std::unique_ptr<Simulation> sim;

    // Optimization state
    std::vector<float> initial_pos_x, initial_pos_y, initial_pos_z;
    std::vector<float> initial_vel_x, initial_vel_y, initial_vel_z;
    std::vector<float> masses;

    // Target state
    std::vector<float> target_pos_x, target_pos_y, target_pos_z;
    int target_time_steps;

    // Optimizers for comparison
    std::unique_ptr<Optimizer> current_optimizer;

public:
    OptimizedTrajectoryOptimizer() {
        // Setup simulation parameters
        params.num_bodies = 2;  // Two-body problem (e.g., spacecraft and planet)
        params.time_step = 0.01f;
        params.G = 1.0f;
        params.epsilon = 0.001f;

        sim = std::make_unique<Simulation>(params);
    }

    void setupOrbitalTransfer() {
        std::cout << "Setting up orbital transfer problem...\n";
        std::cout << "Goal: Find initial velocity to reach target position\n\n";

        // Body 0: Central body (planet)
        // Body 1: Spacecraft
        initial_pos_x = {0.0f, 1.0f};   // Planet at origin, spacecraft at distance 1
        initial_pos_y = {0.0f, 0.0f};
        initial_pos_z = {0.0f, 0.0f};

        // Initial guess for velocities
        initial_vel_x = {0.0f, 0.0f};
        initial_vel_y = {0.0f, 1.0f};  // Simple initial guess
        initial_vel_z = {0.0f, 0.0f};

        masses = {10.0f, 0.1f};  // Heavy planet, light spacecraft

        // Target: reachable with simpler dynamics
        target_time_steps = 20;  // Much shorter time = easier and more stable
        target_pos_x = {0.0f, 0.8f};  // Spacecraft close to opposite side
        target_pos_y = {0.0f, 0.6f};  // Reachable target
        target_pos_z = {0.0f, 0.0f};

        std::cout << "Initial spacecraft position: (" << initial_pos_x[1] << ", " << initial_pos_y[1] << ")\n";
        std::cout << "Target spacecraft position: (" << target_pos_x[1] << ", " << target_pos_y[1] << ")\n";
        std::cout << "Time steps: " << target_time_steps << "\n";
        std::cout << "Initial velocity guess: (" << initial_vel_x[1] << ", " << initial_vel_y[1] << ")\n\n";
    }

    float forward(const std::vector<float>& vel_x, const std::vector<float>& vel_y) {
        // Reset simulation state for reuse
        sim->resetState();
        sim->enableGradients();

        BodySystem* bodies = sim->getBodies();

        // Set initial state using optimized batched transfers
        bodies->setStateFromHost(initial_pos_x, initial_pos_y, initial_pos_z,
                                vel_x, vel_y, initial_vel_z, masses);

        // Run simulation forward
        for (int i = 0; i < target_time_steps; i++) {
            sim->step();
        }

        // Get final positions
        std::vector<float> final_pos_x(params.num_bodies);
        std::vector<float> final_pos_y(params.num_bodies);
        std::vector<float> final_pos_z(params.num_bodies);
        bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

        // Compute loss (distance to target)
        float loss = 0.0f;
        for (int i = 0; i < params.num_bodies; i++) {
            float dx = final_pos_x[i] - target_pos_x[i];
            float dy = final_pos_y[i] - target_pos_y[i];
            float dz = final_pos_z[i] - target_pos_z[i];
            loss += 0.5f * (dx*dx + dy*dy + dz*dz);
        }

        return loss;
    }

    void computeGradients(const std::vector<float>& vel_x, const std::vector<float>& vel_y,
                         std::vector<float>& grad_vel_x, std::vector<float>& grad_vel_y) {
        // Reset simulation state for reuse
        sim->resetState();
        sim->enableGradients();

        BodySystem* bodies = sim->getBodies();

        // Set initial state using optimized batched transfers
        bodies->setStateFromHost(initial_pos_x, initial_pos_y, initial_pos_z,
                                vel_x, vel_y, initial_vel_z, masses);

        // Forward pass
        for (int i = 0; i < target_time_steps; i++) {
            sim->step();
        }

        // Compute gradients w.r.t. final positions
        float loss = sim->computeGradients(target_pos_x, target_pos_y, target_pos_z);

        // Get gradients w.r.t. initial velocities
        grad_vel_x.resize(params.num_bodies);
        grad_vel_y.resize(params.num_bodies);

        size_t size = params.num_bodies * sizeof(float);
        cudaMemcpy(grad_vel_x.data(), bodies->d_grad_vel_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_vel_y.data(), bodies->d_grad_vel_y, size, cudaMemcpyDeviceToHost);
    }

    void optimizeWithAlgorithm(const std::string& optimizer_name, int max_iterations = 50) {
        std::cout << "Optimizing with " << optimizer_name << " optimizer\n";
        std::cout << std::string(50, '=') << "\n";

        // Create optimizer based on name
        if (optimizer_name == "Momentum") {
            current_optimizer = std::make_unique<MomentumOptimizer>(0.5f, 0.9f);
        } else if (optimizer_name == "Adam") {
            current_optimizer = std::make_unique<AdamOptimizer>(0.01f, 0.9f, 0.999f, 1e-8f);
        } else if (optimizer_name == "AdamW") {
            current_optimizer = std::make_unique<AdamWOptimizer>(0.01f, 0.9f, 0.999f, 1e-8f, 0.001f);
        } else if (optimizer_name == "L-BFGS") {
            current_optimizer = std::make_unique<LBFGSOptimizer>(1.0f, 10);
        } else {
            std::cout << "Unknown optimizer: " << optimizer_name << "\n";
            return;
        }

        // Current parameters (only optimizing spacecraft velocity)
        std::vector<float> params_to_optimize = {initial_vel_x[1], initial_vel_y[1]};
        std::vector<float> best_params = params_to_optimize;
        float best_loss = 1e10f;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < max_iterations; iter++) {
            // Reconstruct full velocity vectors
            std::vector<float> vel_x = initial_vel_x;
            std::vector<float> vel_y = initial_vel_y;
            vel_x[1] = params_to_optimize[0];
            vel_y[1] = params_to_optimize[1];

            // Forward pass
            float loss = forward(vel_x, vel_y);

            // Track best solution
            if (loss < best_loss) {
                best_loss = loss;
                best_params = params_to_optimize;
            }

            // Compute gradients
            std::vector<float> grad_vel_x, grad_vel_y;
            computeGradients(vel_x, vel_y, grad_vel_x, grad_vel_y);

            // Extract gradients for spacecraft velocity only
            std::vector<float> gradients = {grad_vel_x[1], grad_vel_y[1]};

            // Update parameters using optimizer
            current_optimizer->step(params_to_optimize, gradients);

            // Print progress
            if (iter % 10 == 0 || iter == max_iterations - 1) {
                // Get current final position for debugging
                std::vector<float> debug_pos_x(2), debug_pos_y(2), debug_pos_z(2);

                // Quick debug forward pass
                sim->resetState();
                BodySystem* bodies = sim->getBodies();
                bodies->setStateFromHost(initial_pos_x, initial_pos_y, initial_pos_z,
                                        vel_x, vel_y, initial_vel_z, masses);

                for (int i = 0; i < target_time_steps; i++) {
                    sim->step();
                }
                bodies->getPositions(debug_pos_x, debug_pos_y, debug_pos_z);

                float error_distance = std::sqrt(
                    std::pow(debug_pos_x[1] - target_pos_x[1], 2) +
                    std::pow(debug_pos_y[1] - target_pos_y[1], 2)
                );

                std::cout << "Iteration " << iter << ":\n";
                std::cout << "  Loss: " << std::fixed << std::setprecision(6) << loss;
                std::cout << " | Final pos: (" << debug_pos_x[1] << ", " << debug_pos_y[1] << ")";
                std::cout << " | Target: (" << target_pos_x[1] << ", " << target_pos_y[1] << ")";
                std::cout << " | Error: " << error_distance << "\n";
                std::cout << "  Velocity: (" << vel_x[1] << ", " << vel_y[1] << ")";
                std::cout << " | Gradient: (" << gradients[0] << ", " << gradients[1] << ")\n";

                // Check convergence
                if (loss < 0.01f) {  // Error < ~0.1 units distance
                    std::cout << "\n✓ Converged! Found optimal trajectory.\n";
                    break;
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "\nOptimization completed in " << duration.count() << " ms\n";
        std::cout << "Final optimized velocity: (" << best_params[0] << ", " << best_params[1] << ")\n";
        std::cout << "Best loss achieved: " << best_loss << "\n\n";

        // Reset optimizer for next run
        current_optimizer->reset();
    }

    void compareOptimizers() {
        std::cout << "PhysGrad Advanced Trajectory Optimization\n";
        std::cout << "========================================\n\n";

        setupOrbitalTransfer();

        std::cout << "Comparing optimization algorithms:\n\n";

        // Test different optimizers
        std::vector<std::string> optimizers = {"Momentum", "Adam", "AdamW", "L-BFGS"};

        for (const auto& opt_name : optimizers) {
            // Reset to initial conditions for fair comparison
            initial_vel_x = {0.0f, 0.0f};
            initial_vel_y = {0.0f, 1.0f};

            optimizeWithAlgorithm(opt_name, 50);
        }

        std::cout << "Optimization comparison completed!\n";
        std::cout << "Applications include:\n";
        std::cout << "- Spacecraft mission planning\n";
        std::cout << "- Asteroid deflection scenarios\n";
        std::cout << "- Satellite constellation deployment\n";
        std::cout << "- Interplanetary transfers\n";
    }
};

int main() {
    try {
        OptimizedTrajectoryOptimizer optimizer;
        optimizer.compareOptimizers();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}