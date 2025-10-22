/**
 * PhysGrad Manipulation Demo 1: Gradient-Based Pushing
 *
 * Demonstrates end-to-end differentiable physics for robotic manipulation.
 * Task: Push a box to a goal position using gradient descent on trajectory.
 *
 * Key Features:
 * - Differentiable contact mechanics
 * - Gradient-based trajectory optimization
 * - PyTorch integration for learning
 * - Visualization of optimization process
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>

#include "common_types.h"
#include "differentiable_contact.h"
#include "adjoint_integrators_standalone.h"

#ifdef HAVE_PYTORCH
#include <torch/torch.h>
#endif

using namespace physgrad;
using namespace physgrad::contact;

// =============================================================================
// PUSHING TASK CONFIGURATION
// =============================================================================

struct PushingTaskConfig {
    // Object properties
    float box_mass = 1.0f;
    float box_size = 0.2f;
    ConceptVector3D<float> box_initial_pos{0.0f, 0.1f, 0.0f};
    ConceptVector3D<float> box_goal_pos{0.5f, 0.1f, 0.0f};

    // Pusher properties
    float pusher_mass = 1000.0f;  // Very large (kinematic/infinite mass approximation)
    float pusher_radius = 0.05f;

    // Physics parameters
    float dt = 0.01f;
    int num_timesteps = 50;
    float friction_coeff = 0.8f;  // Increased from 0.5 for better grip

    // Optimization parameters
    int num_optimization_iters = 50;
    float learning_rate = 0.0005f;  // Very small for stable convergence

    // Visualization
    bool visualize = true;
    int print_every = 5;
};

// =============================================================================
// PUSHING SIMULATION
// =============================================================================

class PushingSimulation {
public:
    PushingSimulation(const PushingTaskConfig& config) : config_(config) {
        // Initialize contact solver
        typename DifferentiableContactSolver<float>::SolverParams solver_params;
        solver_params.max_iterations = 100;  // Increased for better convergence
        solver_params.tolerance = 1e-4f;     // Relaxed from 1e-6
        solver_params.contact_stiffness = 0.001f;  // Very low for gentle pushing
        solver_params.relaxation = 0.8f;
        solver_params.use_friction = true;

        contact_solver_ = std::make_unique<DifferentiableContactSolver<float>>(solver_params);

        // Initialize contact detector with radii
        std::vector<float> radii = {
            config_.box_size / 2.0f,  // Box (approximated as sphere)
            config_.pusher_radius      // Pusher
        };
        contact_detector_ = std::make_unique<SphereContactDetector<float>>(radii);
    }

    /**
     * Run forward simulation given pusher trajectory
     * Returns final box position
     */
    ConceptVector3D<float> simulate(
        const std::vector<ConceptVector3D<float>>& pusher_trajectory,
        std::vector<ConceptVector3D<float>>* box_trajectory = nullptr) {

        // Initialize state
        std::vector<ConceptVector3D<float>> positions = {
            config_.box_initial_pos,
            pusher_trajectory[0]  // Pusher starts at first waypoint
        };

        std::vector<ConceptVector3D<float>> velocities(2);
        std::vector<float> masses = {config_.box_mass, config_.pusher_mass};

        if (box_trajectory) {
            box_trajectory->clear();
            box_trajectory->push_back(positions[0]);
        }

        // Simulate each timestep
        for (int t = 0; t < config_.num_timesteps; ++t) {
            // Apply gravity BEFORE contact resolution
            velocities[0][1] -= 9.8f * config_.dt;  // Box only

            // Detect contacts
            auto contacts = contact_detector_->detectContacts(positions);

            // Set friction coefficients
            for (auto& contact : contacts) {
                contact.friction_coefficient = config_.friction_coeff;
            }

            // Solve contacts
            auto solution = contact_solver_->solveContacts(
                contacts, velocities, masses, config_.dt);

            // Apply contact impulses to velocities (even if not fully converged)
            if (!contacts.empty() && !solution.normal_impulses.empty()) {
                for (size_t c = 0; c < contacts.size(); ++c) {
                    int body_a = contacts[c].body_a_id;
                    int body_b = contacts[c].body_b_id;

                    // Normal impulse
                    float impulse_n = solution.normal_impulses[c];
                    velocities[body_a] = velocities[body_a] -
                        contacts[c].normal * (impulse_n / masses[body_a]);
                    if (body_b != SIZE_MAX) {
                        velocities[body_b] = velocities[body_b] +
                            contacts[c].normal * (impulse_n / masses[body_b]);
                    }

                    // Friction impulses - apply tangential impulses from solver
                    if (solution.friction_impulses_u.size() > c && solution.friction_impulses_v.size() > c) {
                        // Get tangent directions from contact
                        ConceptVector3D<float> tangent1{1.0f, 0.0f, 0.0f};
                        ConceptVector3D<float> tangent2{0.0f, 0.0f, 1.0f};

                        // Make tangents orthogonal to normal
                        auto n = contacts[c].normal;
                        tangent1 = tangent1 - n * (tangent1[0]*n[0] + tangent1[1]*n[1] + tangent1[2]*n[2]);
                        float len1 = std::sqrt(tangent1[0]*tangent1[0] + tangent1[1]*tangent1[1] + tangent1[2]*tangent1[2]);
                        if (len1 > 1e-6f) {
                            tangent1 = tangent1 * (1.0f / len1);

                            // Tangent 2 orthogonal to both normal and tangent1
                            tangent2 = ConceptVector3D<float>{
                                n[1]*tangent1[2] - n[2]*tangent1[1],
                                n[2]*tangent1[0] - n[0]*tangent1[2],
                                n[0]*tangent1[1] - n[1]*tangent1[0]
                            };

                            // Apply friction impulses in both tangent directions
                            float impulse_u = solution.friction_impulses_u[c];
                            float impulse_v = solution.friction_impulses_v[c];

                            velocities[body_a] = velocities[body_a] - tangent1 * (impulse_u / masses[body_a])
                                                                     - tangent2 * (impulse_v / masses[body_a]);
                            if (body_b != SIZE_MAX) {
                                velocities[body_b] = velocities[body_b] + tangent1 * (impulse_u / masses[body_b])
                                                                         + tangent2 * (impulse_v / masses[body_b]);
                            }
                        }
                    }
                }
            }

            // Apply velocity damping to prevent explosions
            const float damping = 0.98f;
            velocities[0] = velocities[0] * damping;

            // Velocity clamping as safety net
            const float max_vel = 5.0f;
            for (int d = 0; d < 3; ++d) {
                velocities[0][d] = std::max(-max_vel, std::min(max_vel, velocities[0][d]));
            }

            // Integrate box position using its velocity (simple Euler)
            positions[0] = positions[0] + velocities[0] * config_.dt;

            // Ground collision (simple)
            if (positions[0][1] < config_.box_size / 2.0f) {
                positions[0][1] = config_.box_size / 2.0f;
                velocities[0][1] = std::max(0.0f, velocities[0][1]);
            }

            // Update pusher position to follow trajectory (kinematic control)
            if (t < static_cast<int>(pusher_trajectory.size()) - 1) {
                // Compute pusher velocity from trajectory for NEXT timestep
                velocities[1] = (pusher_trajectory[t+1] - pusher_trajectory[t]) * (1.0f / config_.dt);
                positions[1] = pusher_trajectory[t+1];
            }

            if (box_trajectory) {
                box_trajectory->push_back(positions[0]);
            }
        }

        return positions[0];  // Final box position
    }

    /**
     * Compute loss: distance from goal
     */
    float computeLoss(const ConceptVector3D<float>& final_box_pos) {
        auto diff = final_box_pos - config_.box_goal_pos;
        return diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
    }

private:
    PushingTaskConfig config_;
    std::unique_ptr<DifferentiableContactSolver<float>> contact_solver_;
    std::unique_ptr<SphereContactDetector<float>> contact_detector_;
};

// =============================================================================
// GRADIENT-BASED OPTIMIZATION
// =============================================================================

class PushingOptimizer {
public:
    PushingOptimizer(const PushingTaskConfig& config)
        : config_(config), simulation_(config) {

        // Initialize pusher trajectory (very conservative - undershoot initially)
        trajectory_.resize(config_.num_timesteps);
        for (int t = 0; t < config_.num_timesteps; ++t) {
            float progress = static_cast<float>(t) / config_.num_timesteps;
            // Start far back, barely reach box to avoid overshooting
            trajectory_[t] = ConceptVector3D<float>{
                -0.4f + progress * 0.45f,  // Move from -0.4 to 0.05 (very gentle)
                0.1f,                       // Fixed height
                0.0f                        // No z-motion
            };
        }
    }

    /**
     * Optimize trajectory using gradient descent
     */
    void optimize() {
        std::cout << "\n=== Gradient-Based Pushing Optimization ===" << std::endl;
        std::cout << "Goal position: (" << config_.box_goal_pos[0] << ", "
                  << config_.box_goal_pos[1] << ", " << config_.box_goal_pos[2] << ")" << std::endl;
        std::cout << "\nIteration | Loss | Final Box Position" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        for (int iter = 0; iter < config_.num_optimization_iters; ++iter) {
            // Forward simulation
            std::vector<ConceptVector3D<float>> box_trajectory;
            auto final_pos = simulation_.simulate(trajectory_, &box_trajectory);

            // Compute loss
            float loss = simulation_.computeLoss(final_pos);

            // Compute gradients (finite differences for now)
            // In full implementation, would use adjoint method
            std::vector<ConceptVector3D<float>> gradients(config_.num_timesteps);

            const float epsilon = 1e-4f;
            for (int t = 0; t < config_.num_timesteps; ++t) {
                for (int dim = 0; dim < 3; ++dim) {
                    // Perturb trajectory
                    auto original = trajectory_[t][dim];

                    trajectory_[t][dim] = original + epsilon;
                    auto final_plus = simulation_.simulate(trajectory_);
                    float loss_plus = simulation_.computeLoss(final_plus);

                    trajectory_[t][dim] = original - epsilon;
                    auto final_minus = simulation_.simulate(trajectory_);
                    float loss_minus = simulation_.computeLoss(final_minus);

                    trajectory_[t][dim] = original;

                    // Gradient via central difference
                    gradients[t][dim] = (loss_plus - loss_minus) / (2.0f * epsilon);
                }
            }

            // Gradient descent update
            for (int t = 0; t < config_.num_timesteps; ++t) {
                trajectory_[t] = trajectory_[t] - gradients[t] * config_.learning_rate;
            }

            // Print progress
            if (iter % config_.print_every == 0) {
                std::cout << std::setw(9) << iter << " | "
                          << std::setw(10) << std::fixed << std::setprecision(6) << loss << " | "
                          << "(" << final_pos[0] << ", " << final_pos[1] << ", " << final_pos[2] << ")"
                          << std::endl;
            }

            // Early stopping
            if (loss < 1e-4f) {
                std::cout << "\nConverged at iteration " << iter << "!" << std::endl;
                break;
            }
        }

        // Final simulation
        std::cout << "\n=== Final Result ===" << std::endl;
        std::vector<ConceptVector3D<float>> final_trajectory;
        auto final_pos = simulation_.simulate(trajectory_, &final_trajectory);
        float final_loss = simulation_.computeLoss(final_pos);

        std::cout << "Final loss: " << final_loss << std::endl;
        std::cout << "Final box position: (" << final_pos[0] << ", "
                  << final_pos[1] << ", " << final_pos[2] << ")" << std::endl;
        std::cout << "Goal position: (" << config_.box_goal_pos[0] << ", "
                  << config_.box_goal_pos[1] << ", " << config_.box_goal_pos[2] << ")" << std::endl;

        float distance_to_goal = std::sqrt(final_loss);
        std::cout << "Distance to goal: " << distance_to_goal << " meters" << std::endl;
    }

    const std::vector<ConceptVector3D<float>>& getTrajectory() const {
        return trajectory_;
    }

private:
    PushingTaskConfig config_;
    PushingSimulation simulation_;
    std::vector<ConceptVector3D<float>> trajectory_;
};

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PhysGrad Manipulation Demo 1: Gradient-Based Pushing" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Configure task
    PushingTaskConfig config;
    config.num_optimization_iters = 100;
    config.learning_rate = 0.001f;
    config.print_every = 10;

    // Run optimization
    PushingOptimizer optimizer(config);
    optimizer.optimize();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Demo Complete!" << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;

    return 0;
}
