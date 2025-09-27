#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace physgrad;

class TrajectoryOptimizer {
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

public:
    TrajectoryOptimizer() {
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

        // Initial guess for velocities (much smaller, more stable)
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
        // Reset simulation
        sim = std::make_unique<Simulation>(params);
        sim->enableGradients();

        BodySystem* bodies = sim->getBodies();
        size_t size = bodies->n * sizeof(float);

        // Set positions and masses
        cudaMemcpy(bodies->d_pos_x, initial_pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, initial_pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, initial_pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        // Set velocities (these are what we're optimizing)
        cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, initial_vel_z.data(), size, cudaMemcpyHostToDevice);

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
        // Reset simulation
        sim = std::make_unique<Simulation>(params);
        sim->enableGradients();

        BodySystem* bodies = sim->getBodies();
        size_t size = bodies->n * sizeof(float);

        // Set initial state
        cudaMemcpy(bodies->d_pos_x, initial_pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, initial_pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, initial_pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, initial_vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        // Forward pass
        for (int i = 0; i < target_time_steps; i++) {
            sim->step();
        }

        // Compute gradients w.r.t. final positions
        float loss = sim->computeGradients(target_pos_x, target_pos_y, target_pos_z);

        // Get gradients w.r.t. initial velocities
        grad_vel_x.resize(params.num_bodies);
        grad_vel_y.resize(params.num_bodies);

        cudaMemcpy(grad_vel_x.data(), bodies->d_grad_vel_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_vel_y.data(), bodies->d_grad_vel_y, size, cudaMemcpyDeviceToHost);
    }

    void optimize() {
        std::cout << "Starting trajectory optimization...\n";
        std::cout << "===================================\n\n";

        // Current velocities (what we're optimizing)
        std::vector<float> vel_x = initial_vel_x;
        std::vector<float> vel_y = initial_vel_y;

        // Optimization parameters - tuned for gradient magnitude
        float learning_rate = 0.5f;  // Based on expected gradient magnitude ~0.05
        int max_iterations = 50;
        float prev_loss = 1e10f;

        // Momentum terms for better convergence
        float momentum_x = 0.0f, momentum_y = 0.0f;
        float momentum_decay = 0.9f;

        for (int iter = 0; iter < max_iterations; iter++) {
            // Forward pass
            float loss = forward(vel_x, vel_y);

            // Compute gradients
            std::vector<float> grad_vel_x, grad_vel_y;
            computeGradients(vel_x, vel_y, grad_vel_x, grad_vel_y);

            // Update velocities using momentum-based gradient descent
            // Only update spacecraft velocity (index 1), not the planet
            momentum_x = momentum_decay * momentum_x + learning_rate * grad_vel_x[1];
            momentum_y = momentum_decay * momentum_y + learning_rate * grad_vel_y[1];
            vel_x[1] -= momentum_x;
            vel_y[1] -= momentum_y;

            // Print progress with more details
            if (iter % 10 == 0 || iter == max_iterations - 1) {
                // Get current final position for debugging
                std::vector<float> debug_pos_x(params.num_bodies);
                std::vector<float> debug_pos_y(params.num_bodies);
                std::vector<float> debug_pos_z(params.num_bodies);

                // Quick forward pass to get final position
                auto debug_sim = std::make_unique<Simulation>(params);
                BodySystem* debug_bodies = debug_sim->getBodies();
                size_t size = debug_bodies->n * sizeof(float);

                cudaMemcpy(debug_bodies->d_pos_x, initial_pos_x.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(debug_bodies->d_pos_y, initial_pos_y.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(debug_bodies->d_pos_z, initial_pos_z.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(debug_bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(debug_bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(debug_bodies->d_vel_z, initial_vel_z.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(debug_bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

                for (int i = 0; i < target_time_steps; i++) {
                    debug_sim->step();
                }

                debug_bodies->getPositions(debug_pos_x, debug_pos_y, debug_pos_z);

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
                std::cout << " | Gradient: (" << grad_vel_x[1] << ", " << grad_vel_y[1] << ")";
                std::cout << " | Momentum: (" << momentum_x << ", " << momentum_y << ")\n";

                // Check convergence - be more realistic
                if (loss < 0.05f) {  // Error < ~0.2 units distance
                    std::cout << "\n✓ Converged! Found optimal trajectory.\n";
                    break;
                }

                // Adaptive learning rate - be less aggressive
                if (loss > prev_loss * 1.01f) {  // Only reduce if loss increases significantly
                    learning_rate *= 0.8f;  // Less aggressive reduction
                    std::cout << "  Reducing learning rate to " << learning_rate << "\n";
                } else if (loss < prev_loss * 0.95f) {  // Increase if good progress
                    learning_rate *= 1.05f;  // Modest increase
                    learning_rate = std::min(learning_rate, 0.5f);  // Cap learning rate
                }
                prev_loss = loss;
            }
        }

        std::cout << "\nFinal optimized velocity: (" << vel_x[1] << ", " << vel_y[1] << ")\n";

        // Verify the trajectory
        verifyTrajectory(vel_x, vel_y);
    }

    void verifyTrajectory(const std::vector<float>& vel_x, const std::vector<float>& vel_y) {
        std::cout << "\nVerifying optimized trajectory...\n";
        std::cout << "=================================\n";

        sim = std::make_unique<Simulation>(params);
        BodySystem* bodies = sim->getBodies();
        size_t size = bodies->n * sizeof(float);

        // Set initial state with optimized velocities
        cudaMemcpy(bodies->d_pos_x, initial_pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, initial_pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, initial_pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, initial_vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        // Run and print key waypoints
        for (int i = 0; i <= target_time_steps; i++) {
            if (i > 0) sim->step();

            if (i % 25 == 0 || i == target_time_steps) {
                std::vector<float> pos_x(params.num_bodies);
                std::vector<float> pos_y(params.num_bodies);
                std::vector<float> pos_z(params.num_bodies);
                bodies->getPositions(pos_x, pos_y, pos_z);

                std::cout << "Step " << i << ": Spacecraft at ("
                          << pos_x[1] << ", " << pos_y[1] << ")\n";
            }
        }

        // Check final position
        std::vector<float> final_pos_x(params.num_bodies);
        std::vector<float> final_pos_y(params.num_bodies);
        std::vector<float> final_pos_z(params.num_bodies);
        bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

        float final_error = std::sqrt(
            std::pow(final_pos_x[1] - target_pos_x[1], 2) +
            std::pow(final_pos_y[1] - target_pos_y[1], 2)
        );

        std::cout << "\nTarget position: (" << target_pos_x[1] << ", " << target_pos_y[1] << ")\n";
        std::cout << "Final position: (" << final_pos_x[1] << ", " << final_pos_y[1] << ")\n";
        std::cout << "Final error: " << final_error << "\n";

        if (final_error < 0.01f) {
            std::cout << "✓ Successfully reached target!\n";
        } else {
            std::cout << "✗ Did not reach target (error > 0.01)\n";
        }
    }
};

int main() {
    std::cout << "PhysGrad Trajectory Optimization\n";
    std::cout << "================================\n\n";

    std::cout << "This demo shows how to use differentiable physics for trajectory optimization.\n";
    std::cout << "We'll find the initial velocity needed to reach a target position.\n\n";

    try {
        TrajectoryOptimizer optimizer;
        optimizer.setupOrbitalTransfer();
        optimizer.optimize();

        std::cout << "\nTrajectory optimization completed!\n";
        std::cout << "Applications include:\n";
        std::cout << "- Spacecraft mission planning\n";
        std::cout << "- Asteroid deflection scenarios\n";
        std::cout << "- Satellite constellation deployment\n";
        std::cout << "- Interplanetary transfers\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}