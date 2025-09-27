#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>

using namespace physgrad;

class ImprovedParameterLearner {
private:
    SimParams true_params;
    SimParams learned_params;
    std::unique_ptr<Simulation> sim;

    // Optimizer state for Adam optimizer
    float m_G = 0.0f, v_G = 0.0f;
    float m_epsilon = 0.0f, v_epsilon = 0.0f;
    std::vector<float> m_mass, v_mass;

    // Optimizer hyperparameters
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps_adam = 1e-8f;
    float learning_rate = 0.01f;

public:
    ImprovedParameterLearner() {
        // Set up true parameters
        true_params.num_bodies = 3;
        true_params.time_step = 0.01f;
        true_params.G = 1.5f;
        true_params.epsilon = 0.002f;

        // Initial guess with some noise
        learned_params = true_params;
        learned_params.G = 1.0f;
        learned_params.epsilon = 0.001f;

        // Initialize optimizer state
        m_mass.resize(true_params.num_bodies, 0.0f);
        v_mass.resize(true_params.num_bodies, 0.0f);

        sim = std::make_unique<Simulation>(learned_params);
    }

    std::vector<std::vector<float>> generateTrajectoryData() {
        std::cout << "Generating ground truth trajectory data...\n";

        auto true_sim = std::make_unique<Simulation>(true_params);
        BodySystem* bodies = true_sim->getBodies();

        // More interesting 3-body configuration
        std::vector<float> pos_x = {-0.7f, 0.7f, 0.0f};
        std::vector<float> pos_y = {0.0f, 0.0f, 1.0f};
        std::vector<float> pos_z = {0.0f, 0.0f, 0.0f};
        std::vector<float> vel_x = {0.1f, -0.1f, 0.0f};
        std::vector<float> vel_y = {0.5f, -0.5f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f, 0.0f};
        std::vector<float> masses = {1.2f, 1.0f, 0.8f};

        size_t size = bodies->n * sizeof(float);
        cudaMemcpy(bodies->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        // Store initial conditions
        initial_pos_x = pos_x;
        initial_pos_y = pos_y;
        initial_pos_z = pos_z;
        initial_vel_x = vel_x;
        initial_vel_y = vel_y;
        initial_vel_z = vel_z;
        initial_masses = masses;

        // Generate longer trajectory
        const int num_steps = 30;
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

    void adamUpdate(float& param, float grad, float& m, float& v, int t) {
        // Adam optimizer update
        m = beta1 * m + (1.0f - beta1) * grad;
        v = beta2 * v + (1.0f - beta2) * grad * grad;

        float m_hat = m / (1.0f - std::pow(beta1, t));
        float v_hat = v / (1.0f - std::pow(beta2, t));

        param -= learning_rate * m_hat / (std::sqrt(v_hat) + eps_adam);
    }

    void learnParameters(const std::vector<std::vector<float>>& trajectory) {
        std::cout << "Learning parameters using Adam optimizer...\n";
        std::cout << "Initial guess: G=" << learned_params.G << ", epsilon=" << learned_params.epsilon << "\n\n";

        const int num_iterations = 50;
        float prev_loss = 1e6f;
        int patience = 0;
        const int max_patience = 10;

        for (int iter = 0; iter < num_iterations; iter++) {
            // Reset simulation
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

            sim->enableGradients();
            sim->enableParameterGradients(true);

            // Accumulate gradients from multiple time points
            float total_loss = 0.0f;
            std::vector<float> total_grad_mass(learned_params.num_bodies, 0.0f);
            float total_grad_G = 0.0f;
            float total_grad_epsilon = 0.0f;
            int num_points = 0;

            // Use more frequent time points for better gradient signal
            for (size_t t = 3; t < trajectory.size(); t += 2) {
                sim->clearTape();

                // Simulate to time point t
                for (size_t step = 0; step < t; step++) {
                    sim->step();
                }

                // Extract target positions
                const auto& target_state = trajectory[t];
                std::vector<float> target_pos_x(learned_params.num_bodies);
                std::vector<float> target_pos_y(learned_params.num_bodies);
                std::vector<float> target_pos_z(learned_params.num_bodies);

                for (int i = 0; i < learned_params.num_bodies; i++) {
                    target_pos_x[i] = target_state[i * 3 + 0];
                    target_pos_y[i] = target_state[i * 3 + 1];
                    target_pos_z[i] = target_state[i * 3 + 2];
                }

                // Compute gradients
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
                num_points++;

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

            // Average gradients
            total_loss /= num_points;
            total_grad_G /= num_points;
            total_grad_epsilon /= num_points;
            for (int i = 0; i < learned_params.num_bodies; i++) {
                total_grad_mass[i] /= num_points;
            }

            // Gradient clipping for stability
            float grad_norm = std::sqrt(total_grad_G * total_grad_G + total_grad_epsilon * total_grad_epsilon);
            if (grad_norm > 1.0f) {
                total_grad_G /= grad_norm;
                total_grad_epsilon /= grad_norm;
            }

            // Adam optimizer updates
            adamUpdate(learned_params.G, total_grad_G, m_G, v_G, iter + 1);
            adamUpdate(learned_params.epsilon, total_grad_epsilon, m_epsilon, v_epsilon, iter + 1);

            // Clamp parameters to reasonable ranges
            learned_params.G = std::max(0.1f, std::min(5.0f, learned_params.G));
            learned_params.epsilon = std::max(0.0001f, std::min(0.01f, learned_params.epsilon));

            // Progress output
            if (iter % 5 == 0 || iter < 10) {
                std::cout << "Iteration " << iter + 1 << ":\n";
                std::cout << "  Loss: " << std::fixed << std::setprecision(6) << total_loss;
                std::cout << " | G: " << learned_params.G << " (true: " << true_params.G << ")";
                std::cout << " | ε: " << learned_params.epsilon << " (true: " << true_params.epsilon << ")\n";
            }

            // Adaptive learning rate and early stopping
            if (total_loss < prev_loss) {
                patience = 0;
            } else {
                patience++;
                if (patience > 3) {
                    learning_rate *= 0.8f;  // Reduce learning rate
                    patience = 0;
                }
            }
            prev_loss = total_loss;

            // Early stopping
            if (patience >= max_patience) {
                std::cout << "Early stopping: no improvement\n";
                break;
            }

            // Check convergence
            float G_error = std::abs(learned_params.G - true_params.G) / true_params.G;
            float eps_error = std::abs(learned_params.epsilon - true_params.epsilon) / true_params.epsilon;

            if (G_error < 0.02f && eps_error < 0.02f) {
                std::cout << "✓ Converged! Parameter errors < 2%\n";
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
    std::cout << "PhysGrad Improved Parameter Learning\n";
    std::cout << "===================================\n\n";

    std::cout << "This demo shows improved parameter learning with:\n";
    std::cout << "- Adam optimizer with adaptive learning rate\n";
    std::cout << "- Gradient clipping for numerical stability\n";
    std::cout << "- Early stopping and convergence detection\n";
    std::cout << "- More trajectory points for better gradient signal\n\n";

    try {
        ImprovedParameterLearner learner;

        auto trajectory = learner.generateTrajectoryData();
        learner.learnParameters(trajectory);

        std::cout << "\nImproved parameter learning completed!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}