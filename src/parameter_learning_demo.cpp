#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>

using namespace physgrad;

class ParameterLearner {
private:
    SimParams true_params;
    SimParams learned_params;
    std::unique_ptr<Simulation> sim;

public:
    ParameterLearner() {
        // Set up true parameters (the "ground truth" we want to learn)
        true_params.num_bodies = 3;
        true_params.time_step = 0.01f;
        true_params.G = 1.5f;           // True gravitational constant
        true_params.epsilon = 0.002f;   // True softening parameter

        // Initial guess for parameters (what we're trying to learn)
        learned_params = true_params;
        learned_params.G = 1.0f;        // Wrong initial guess
        learned_params.epsilon = 0.001f; // Wrong initial guess

        sim = std::make_unique<Simulation>(learned_params);
    }

    std::vector<std::vector<float>> generateTrajectoryData() {
        std::cout << "Generating ground truth trajectory data...\n";

        // Create simulation with true parameters
        auto true_sim = std::make_unique<Simulation>(true_params);
        BodySystem* bodies = true_sim->getBodies();

        // Set up interesting 3-body configuration
        std::vector<float> pos_x = {-0.5f, 0.5f, 0.0f};
        std::vector<float> pos_y = {0.0f, 0.0f, 0.8f};
        std::vector<float> pos_z = {0.0f, 0.0f, 0.0f};
        std::vector<float> vel_x = {0.0f, 0.0f, 0.1f};
        std::vector<float> vel_y = {0.4f, -0.4f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f, 0.8f};

        size_t size = bodies->n * sizeof(float);
        cudaMemcpy(bodies->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        // Store initial conditions for learning simulation
        initial_pos_x = pos_x;
        initial_pos_y = pos_y;
        initial_pos_z = pos_z;
        initial_vel_x = vel_x;
        initial_vel_y = vel_y;
        initial_vel_z = vel_z;
        initial_masses = masses;

        // Generate trajectory
        const int num_steps = 20;
        std::vector<std::vector<float>> trajectory;

        for (int step = 0; step <= num_steps; step++) {
            if (step > 0) true_sim->step();

            std::vector<float> state_x, state_y, state_z;
            bodies->getPositions(state_x, state_y, state_z);

            std::vector<float> combined_state;
            for (int i = 0; i < true_params.num_bodies; i++) {
                combined_state.push_back(state_x[i]);
                combined_state.push_back(state_y[i]);
                combined_state.push_back(state_z[i]);
            }
            trajectory.push_back(combined_state);
        }

        std::cout << "Generated trajectory with " << trajectory.size() << " time points\n";
        std::cout << "True parameters: G=" << true_params.G << ", epsilon=" << true_params.epsilon << "\n\n";

        return trajectory;
    }

    void learnParameters(const std::vector<std::vector<float>>& trajectory) {
        std::cout << "Learning parameters from trajectory data...\n";
        std::cout << "Initial guess: G=" << learned_params.G << ", epsilon=" << learned_params.epsilon << "\n\n";

        const float learning_rate = 0.1f;
        const int num_iterations = 10;

        for (int iter = 0; iter < num_iterations; iter++) {
            std::cout << "Iteration " << iter + 1 << ":\n";

            // Reset simulation with current parameter estimates
            sim = std::make_unique<Simulation>(learned_params);
            BodySystem* bodies = sim->getBodies();

            // Set initial conditions
            size_t size = bodies->n * sizeof(float);
            cudaMemcpy(bodies->d_pos_x, initial_pos_x.data(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(bodies->d_pos_y, initial_pos_y.data(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(bodies->d_pos_z, initial_pos_z.data(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(bodies->d_vel_x, initial_vel_x.data(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(bodies->d_vel_y, initial_vel_y.data(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(bodies->d_vel_z, initial_vel_z.data(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(bodies->d_mass, initial_masses.data(), size, cudaMemcpyHostToDevice);

            // Enable gradients
            sim->enableGradients();
            sim->enableParameterGradients(true);

            // Compute loss against multiple time points
            float total_loss = 0.0f;
            std::vector<float> total_grad_mass(learned_params.num_bodies, 0.0f);
            float total_grad_G = 0.0f;
            float total_grad_epsilon = 0.0f;

            // Use every 4th time point to avoid overfitting to noise
            for (size_t t = 4; t < trajectory.size(); t += 4) {
                // Simulate to time point t
                sim->clearTape();
                for (size_t step = 0; step < t; step++) {
                    sim->step();
                }

                // Extract target positions from trajectory
                const auto& target_state = trajectory[t];
                std::vector<float> target_pos_x(learned_params.num_bodies);
                std::vector<float> target_pos_y(learned_params.num_bodies);
                std::vector<float> target_pos_z(learned_params.num_bodies);

                for (int i = 0; i < learned_params.num_bodies; i++) {
                    target_pos_x[i] = target_state[i * 3 + 0];
                    target_pos_y[i] = target_state[i * 3 + 1];
                    target_pos_z[i] = target_state[i * 3 + 2];
                }

                // Compute gradients for this time point
                std::vector<float> grad_mass;
                float grad_G, grad_epsilon;

                float loss = sim->computeParameterGradients(target_pos_x, target_pos_y, target_pos_z,
                                                           grad_mass, grad_G, grad_epsilon);

                total_loss += loss;
                total_grad_G += grad_G;
                total_grad_epsilon += grad_epsilon;
                for (int i = 0; i < learned_params.num_bodies; i++) {
                    total_grad_mass[i] += grad_mass[i];
                }

                // Reset for next time point
                sim = std::make_unique<Simulation>(learned_params);
                bodies = sim->getBodies();
                cudaMemcpy(bodies->d_pos_x, initial_pos_x.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(bodies->d_pos_y, initial_pos_y.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(bodies->d_pos_z, initial_pos_z.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(bodies->d_vel_x, initial_vel_x.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(bodies->d_vel_y, initial_vel_y.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(bodies->d_vel_z, initial_vel_z.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpy(bodies->d_mass, initial_masses.data(), size, cudaMemcpyHostToDevice);
                sim->enableGradients();
                sim->enableParameterGradients(true);
            }

            // Update parameters using gradient descent
            learned_params.G -= learning_rate * total_grad_G;
            learned_params.epsilon -= learning_rate * total_grad_epsilon;

            // Clamp parameters to reasonable ranges
            learned_params.G = std::max(0.1f, std::min(3.0f, learned_params.G));
            learned_params.epsilon = std::max(0.0001f, std::min(0.01f, learned_params.epsilon));

            std::cout << "  Loss: " << std::fixed << std::setprecision(6) << total_loss;
            std::cout << " | G: " << learned_params.G << " (true: " << true_params.G << ")";
            std::cout << " | ε: " << learned_params.epsilon << " (true: " << true_params.epsilon << ")\n";

            // Check convergence
            float G_error = std::abs(learned_params.G - true_params.G) / true_params.G;
            float eps_error = std::abs(learned_params.epsilon - true_params.epsilon) / true_params.epsilon;

            if (G_error < 0.05f && eps_error < 0.05f) {
                std::cout << "✓ Converged! Parameter errors < 5%\n";
                break;
            }
        }

        std::cout << "\nFinal Results:\n";
        std::cout << "==============\n";
        std::cout << "True G: " << true_params.G << " | Learned G: " << learned_params.G;
        std::cout << " | Error: " << std::abs(learned_params.G - true_params.G) / true_params.G * 100 << "%\n";
        std::cout << "True ε: " << true_params.epsilon << " | Learned ε: " << learned_params.epsilon;
        std::cout << " | Error: " << std::abs(learned_params.epsilon - true_params.epsilon) / true_params.epsilon * 100 << "%\n";
    }

private:
    std::vector<float> initial_pos_x, initial_pos_y, initial_pos_z;
    std::vector<float> initial_vel_x, initial_vel_y, initial_vel_z;
    std::vector<float> initial_masses;
};

int main() {
    std::cout << "PhysGrad Parameter Learning Demonstration\n";
    std::cout << "=========================================\n\n";

    std::cout << "This demo shows how PhysGrad can learn physical parameters\n";
    std::cout << "from observed trajectory data using differentiable physics.\n\n";

    try {
        ParameterLearner learner;

        // Generate synthetic trajectory data with known parameters
        auto trajectory = learner.generateTrajectoryData();

        // Learn parameters from the trajectory
        learner.learnParameters(trajectory);

        std::cout << "\nDemonstration completed!\n";
        std::cout << "PhysGrad successfully learned physical parameters from trajectory data.\n";
        std::cout << "This capability enables:\n";
        std::cout << "- Estimating masses from orbital observations\n";
        std::cout << "- Learning gravitational constants from planetary motion\n";
        std::cout << "- Calibrating simulation parameters from real-world data\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}