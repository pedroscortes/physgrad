/**
 * PhysGrad Manipulation Demo 3: Object Stacking
 *
 * Demonstrates differentiable physics for sequential manipulation.
 * Task: Stack multiple blocks to build a stable tower.
 *
 * Key Features:
 * - Sequential manipulation planning
 * - Stability analysis through simulation
 * - Contact-rich multi-object dynamics
 * - Long-horizon optimization
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>
#include <algorithm>

#include "common_types.h"
#include "differentiable_contact.h"

using namespace physgrad;
using namespace physgrad::contact;

// =============================================================================
// STACKING TASK CONFIGURATION
// =============================================================================

struct StackingTaskConfig {
    // Block properties
    int num_blocks = 3;
    float block_mass = 0.5f;
    float block_size = 0.1f;  // Cube side length

    // Initial block positions (scattered)
    std::vector<ConceptVector3D<float>> initial_positions = {
        {-0.2f, 0.05f, 0.0f},
        {0.0f, 0.05f, 0.2f},
        {0.2f, 0.05f, -0.1f}
    };

    // Target tower position
    ConceptVector3D<float> tower_base{0.0f, 0.05f, 0.0f};

    // Gripper properties
    float gripper_mass = 0.3f;
    float gripper_radius = 0.03f;

    // Physics parameters
    float dt = 0.01f;
    int timesteps_per_placement = 30;
    float friction_coeff = 0.7f;
    float gravity = 9.8f;

    // Stability check
    int stability_check_timesteps = 50;
    float stability_threshold = 0.01f;  // meters

    // Loss function weights
    float height_weight = 2.0f;
    float stability_weight = 1.0f;
    float alignment_weight = 0.5f;
    float collision_penalty_weight = 10.0f;

    // Optimization parameters
    int num_optimization_iters = 50;
    float learning_rate = 0.005f;
    int print_every = 5;
};

// =============================================================================
// STACKING SIMULATION
// =============================================================================

class StackingSimulation {
public:
    StackingSimulation(const StackingTaskConfig& config) : config_(config) {
        // Initialize contact solver
        typename DifferentiableContactSolver<float>::SolverParams solver_params;
        solver_params.max_iterations = 50;
        solver_params.tolerance = 1e-6f;
        solver_params.contact_stiffness = 0.3f;
        solver_params.relaxation = 0.8f;
        solver_params.use_friction = true;

        contact_solver_ = std::make_unique<DifferentiableContactSolver<float>>(solver_params);
    }

    /**
     * Simulate stacking sequence
     * placement_order: order in which to stack blocks (permutation of [0,1,2,...])
     * placement_positions: where to place each block
     */
    float simulate(
        const std::vector<int>& placement_order,
        const std::vector<ConceptVector3D<float>>& placement_positions,
        std::vector<ConceptVector3D<float>>* final_positions = nullptr) {

        // Initialize state: all blocks at initial positions
        std::vector<ConceptVector3D<float>> positions = config_.initial_positions;
        std::vector<ConceptVector3D<float>> velocities(config_.num_blocks);
        std::vector<float> masses(config_.num_blocks, config_.block_mass);
        std::vector<float> radii(config_.num_blocks, config_.block_size / 2.0f);

        SphereContactDetector<float> detector(radii);

        // Place each block sequentially
        for (size_t step = 0; step < placement_order.size(); ++step) {
            int block_to_place = placement_order[step];
            ConceptVector3D<float> target_pos = placement_positions[step];

            // Move block to target position (simplified - instant teleport)
            positions[block_to_place] = target_pos;
            velocities[block_to_place] = ConceptVector3D<float>{0.0f, 0.0f, 0.0f};

            // Simulate settling
            for (int t = 0; t < config_.timesteps_per_placement; ++t) {
                // Detect contacts
                auto contacts = detector.detectContacts(positions);

                // Set friction
                for (auto& contact : contacts) {
                    contact.friction_coefficient = config_.friction_coeff;
                }

                // Solve contacts
                auto solution = contact_solver_->solveContacts(
                    contacts, velocities, masses, config_.dt);

                // Apply contact impulses
                if (solution.converged && !contacts.empty()) {
                    for (size_t c = 0; c < contacts.size(); ++c) {
                        int body_a = contacts[c].body_a_id;
                        int body_b = contacts[c].body_b_id;

                        float impulse_n = solution.normal_impulses[c];
                        velocities[body_a] = velocities[body_a] -
                            contacts[c].normal * (impulse_n / masses[body_a]);
                        if (body_b != SIZE_MAX) {
                            velocities[body_b] = velocities[body_b] +
                                contacts[c].normal * (impulse_n / masses[body_b]);
                        }
                    }
                }

                // Apply gravity
                for (int i = 0; i < config_.num_blocks; ++i) {
                    velocities[i][1] -= config_.gravity * config_.dt;
                }

                // Integrate
                for (int i = 0; i < config_.num_blocks; ++i) {
                    positions[i] = positions[i] + velocities[i] * config_.dt;
                }

                // Ground collision
                for (int i = 0; i < config_.num_blocks; ++i) {
                    if (positions[i][1] < config_.block_size / 2.0f) {
                        positions[i][1] = config_.block_size / 2.0f;
                        velocities[i][1] = std::max(0.0f, velocities[i][1]);
                    }
                }
            }
        }

        // Stability check - simulate further to see if tower falls
        float max_displacement = 0.0f;
        auto positions_before_stability = positions;

        for (int t = 0; t < config_.stability_check_timesteps; ++t) {
            auto contacts = detector.detectContacts(positions);

            for (auto& contact : contacts) {
                contact.friction_coefficient = config_.friction_coeff;
            }

            auto solution = contact_solver_->solveContacts(
                contacts, velocities, masses, config_.dt);

            if (solution.converged && !contacts.empty()) {
                for (size_t c = 0; c < contacts.size(); ++c) {
                    int body_a = contacts[c].body_a_id;
                    int body_b = contacts[c].body_b_id;

                    float impulse_n = solution.normal_impulses[c];
                    velocities[body_a] = velocities[body_a] -
                        contacts[c].normal * (impulse_n / masses[body_a]);
                    if (body_b != SIZE_MAX) {
                        velocities[body_b] = velocities[body_b] +
                            contacts[c].normal * (impulse_n / masses[body_b]);
                    }
                }
            }

            for (int i = 0; i < config_.num_blocks; ++i) {
                velocities[i][1] -= config_.gravity * config_.dt;
            }

            for (int i = 0; i < config_.num_blocks; ++i) {
                positions[i] = positions[i] + velocities[i] * config_.dt;
            }

            for (int i = 0; i < config_.num_blocks; ++i) {
                if (positions[i][1] < config_.block_size / 2.0f) {
                    positions[i][1] = config_.block_size / 2.0f;
                    velocities[i][1] = std::max(0.0f, velocities[i][1]);
                }
            }

            // Track displacement
            for (int i = 0; i < config_.num_blocks; ++i) {
                auto disp = positions[i] - positions_before_stability[i];
                float dist = std::sqrt(disp[0]*disp[0] + disp[1]*disp[1] + disp[2]*disp[2]);
                max_displacement = std::max(max_displacement, dist);
            }
        }

        if (final_positions) {
            *final_positions = positions;
        }

        return max_displacement;  // Stability metric (lower is better)
    }

    /**
     * Compute stacking quality
     */
    float computeStackingLoss(
        const std::vector<int>& placement_order,
        const std::vector<ConceptVector3D<float>>& placement_positions) {

        std::vector<ConceptVector3D<float>> final_positions;
        float stability_metric = simulate(placement_order, placement_positions, &final_positions);

        // Tower height (maximize)
        float max_height = 0.0f;
        for (const auto& pos : final_positions) {
            max_height = std::max(max_height, pos[1]);
        }

        // Alignment (blocks should be vertically aligned)
        float alignment_error = 0.0f;
        for (const auto& pos : final_positions) {
            float x_diff = pos[0] - config_.tower_base[0];
            float z_diff = pos[2] - config_.tower_base[2];
            alignment_error += std::sqrt(x_diff*x_diff + z_diff*z_diff);
        }

        // Combined loss (minimize)
        float loss = -config_.height_weight * max_height
                    + config_.stability_weight * stability_metric
                    + config_.alignment_weight * alignment_error;

        return loss;
    }

private:
    StackingTaskConfig config_;
    std::unique_ptr<DifferentiableContactSolver<float>> contact_solver_;
};

// =============================================================================
// STACKING OPTIMIZER
// =============================================================================

class StackingOptimizer {
public:
    StackingOptimizer(const StackingTaskConfig& config)
        : config_(config), simulation_(config) {

        // Initialize placement order (just sequential for now)
        for (int i = 0; i < config_.num_blocks; ++i) {
            placement_order_.push_back(i);
        }

        // Initialize placement positions (stack vertically at base)
        for (int i = 0; i < config_.num_blocks; ++i) {
            float height = config_.block_size / 2.0f + i * config_.block_size;
            placement_positions_.push_back(ConceptVector3D<float>{
                config_.tower_base[0],
                height,
                config_.tower_base[2]
            });
        }
    }

    void optimize() {
        std::cout << "\n=== Object Stacking Optimization ===" << std::endl;
        std::cout << "Number of blocks: " << config_.num_blocks << std::endl;
        std::cout << "Tower base: (" << config_.tower_base[0] << ", "
                  << config_.tower_base[1] << ", " << config_.tower_base[2] << ")" << std::endl;
        std::cout << "\nIteration | Loss | Tower Height | Stability" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        for (int iter = 0; iter < config_.num_optimization_iters; ++iter) {
            // Compute current loss
            float loss = simulation_.computeStackingLoss(placement_order_, placement_positions_);

            // Compute gradients via finite differences
            std::vector<ConceptVector3D<float>> gradients(config_.num_blocks);

            const float epsilon = 1e-4f;
            for (int b = 0; b < config_.num_blocks; ++b) {
                for (int dim = 0; dim < 3; ++dim) {
                    auto original = placement_positions_[b][dim];

                    placement_positions_[b][dim] = original + epsilon;
                    float loss_plus = simulation_.computeStackingLoss(
                        placement_order_, placement_positions_);

                    placement_positions_[b][dim] = original - epsilon;
                    float loss_minus = simulation_.computeStackingLoss(
                        placement_order_, placement_positions_);

                    placement_positions_[b][dim] = original;

                    gradients[b][dim] = (loss_plus - loss_minus) / (2.0f * epsilon);
                }
            }

            // Gradient descent update
            for (int b = 0; b < config_.num_blocks; ++b) {
                placement_positions_[b] = placement_positions_[b] - gradients[b] * config_.learning_rate;

                // Constrain to reasonable placement region
                placement_positions_[b][1] = std::max(config_.block_size / 2.0f,
                                                     placement_positions_[b][1]);
            }

            // Compute metrics for display
            std::vector<ConceptVector3D<float>> final_positions;
            float stability = simulation_.simulate(placement_order_, placement_positions_,
                                                   &final_positions);
            float max_height = 0.0f;
            for (const auto& pos : final_positions) {
                max_height = std::max(max_height, pos[1]);
            }

            if (iter % config_.print_every == 0) {
                std::cout << std::setw(9) << iter << " | "
                          << std::setw(10) << std::fixed << std::setprecision(6) << loss << " | "
                          << std::setw(12) << max_height << " | "
                          << std::setw(10) << stability << std::endl;
            }
        }

        // Final result
        std::cout << "\n=== Final Stacking Configuration ===" << std::endl;
        std::vector<ConceptVector3D<float>> final_positions;
        float final_stability = simulation_.simulate(placement_order_, placement_positions_,
                                                     &final_positions);

        float final_loss = simulation_.computeStackingLoss(placement_order_, placement_positions_);
        std::cout << "Final loss: " << final_loss << std::endl;
        std::cout << "Stability metric: " << final_stability << " meters" << std::endl;

        float max_height = 0.0f;
        for (const auto& pos : final_positions) {
            max_height = std::max(max_height, pos[1]);
        }
        std::cout << "Tower height: " << max_height << " meters" << std::endl;

        std::cout << "\nFinal block positions:" << std::endl;
        for (int i = 0; i < config_.num_blocks; ++i) {
            std::cout << "Block " << i << ": ("
                      << final_positions[i][0] << ", "
                      << final_positions[i][1] << ", "
                      << final_positions[i][2] << ")" << std::endl;
        }

        if (final_stability < config_.stability_threshold) {
            std::cout << "\n✓ Stable tower achieved!" << std::endl;
        } else {
            std::cout << "\n⚠ Tower may be unstable (threshold: "
                      << config_.stability_threshold << ")" << std::endl;
        }
    }

private:
    StackingTaskConfig config_;
    StackingSimulation simulation_;
    std::vector<int> placement_order_;
    std::vector<ConceptVector3D<float>> placement_positions_;
};

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PhysGrad Manipulation Demo 3: Object Stacking" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Configure task
    StackingTaskConfig config;
    config.num_blocks = 3;
    config.num_optimization_iters = 30;
    config.learning_rate = 0.003f;
    config.print_every = 5;

    // Run optimization
    StackingOptimizer optimizer(config);
    optimizer.optimize();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Demo Complete!" << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;

    return 0;
}
