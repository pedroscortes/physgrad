/**
 * PhysGrad Manipulation Demo 2: Grasping Optimization
 *
 * Demonstrates differentiable physics for grasp optimization.
 * Task: Optimize gripper finger positions for stable grasp of an object.
 *
 * Key Features:
 * - Multi-contact grasp quality metrics
 * - Force closure analysis
 * - Differentiable contact normals
 * - Gripper configuration optimization
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
// GRASPING TASK CONFIGURATION
// =============================================================================

struct GraspingTaskConfig {
    // Object properties
    float object_mass = 1.0f;
    float object_radius = 0.1f;
    ConceptVector3D<float> object_pos{0.0f, 0.2f, 0.0f};

    // Gripper properties
    int num_fingers = 3;  // Parallel jaw gripper (2 fingers) or multi-finger (3+)
    float finger_mass = 0.1f;
    float finger_radius = 0.02f;

    // Physics parameters
    float dt = 0.01f;
    int num_timesteps = 10;  // Reduced from 20 to limit fall distance if contact lost
    float friction_coeff = 0.6f;
    float gravity = 9.8f;

    // Grasp quality weights
    float force_closure_weight = 1.0f;
    float contact_normal_weight = 0.5f;
    float penetration_penalty_weight = 5.0f;  // Balanced to prevent both explosion and instability

    // Optimization parameters
    int num_optimization_iters = 100;
    float learning_rate = 0.01f;
    int print_every = 10;
};

// =============================================================================
// GRASP QUALITY METRICS
// =============================================================================

class GraspQualityMetrics {
public:
    /**
     * Compute force closure quality
     * Higher is better - measures ability to resist arbitrary wrenches
     */
    static float computeForceClosureQuality(
        const std::vector<ConceptVector3D<float>>& contact_positions,
        const std::vector<ConceptVector3D<float>>& contact_normals,
        const ConceptVector3D<float>& object_center) {

        if (contact_positions.size() < 2) return 0.0f;

        // Simple force closure metric: sum of force contributions in all directions
        float quality = 0.0f;

        // Check coverage in different directions
        std::vector<ConceptVector3D<float>> directions = {
            {1.0f, 0.0f, 0.0f},
            {-1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, -1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, -1.0f}
        };

        for (const auto& dir : directions) {
            float max_projection = 0.0f;
            for (const auto& normal : contact_normals) {
                float proj = normal[0] * dir[0] + normal[1] * dir[1] + normal[2] * dir[2];
                max_projection = std::max(max_projection, proj);
            }
            quality += max_projection;
        }

        // Normalize by number of directions
        quality /= directions.size();

        return quality;
    }

    /**
     * Compute contact normal quality
     * Prefer normals pointing toward object center
     */
    static float computeContactNormalQuality(
        const std::vector<ConceptVector3D<float>>& contact_positions,
        const std::vector<ConceptVector3D<float>>& contact_normals,
        const ConceptVector3D<float>& object_center) {

        if (contact_normals.empty()) return 0.0f;

        float total_quality = 0.0f;

        for (size_t i = 0; i < contact_positions.size(); ++i) {
            // Vector from contact to object center
            auto to_center = object_center - contact_positions[i];
            float dist = std::sqrt(to_center[0]*to_center[0] +
                                  to_center[1]*to_center[1] +
                                  to_center[2]*to_center[2]);

            if (dist > 1e-6f) {
                // Normalize
                to_center = ConceptVector3D<float>{
                    to_center[0] / dist,
                    to_center[1] / dist,
                    to_center[2] / dist
                };

                // Dot product with contact normal
                float alignment = contact_normals[i][0] * to_center[0] +
                                contact_normals[i][1] * to_center[1] +
                                contact_normals[i][2] * to_center[2];

                total_quality += alignment;
            }
        }

        return total_quality / contact_positions.size();
    }

    /**
     * Compute grasp stability metric
     * Measures resistance to gravity
     */
    static float computeGraspStability(
        const std::vector<ConceptVector3D<float>>& contact_normals,
        float object_mass,
        float gravity) {

        if (contact_normals.empty()) return 0.0f;

        // Sum vertical components of contact normals
        float vertical_support = 0.0f;
        for (const auto& normal : contact_normals) {
            vertical_support += std::max(0.0f, normal[1]);  // Upward component
        }

        // Normalize by weight
        float weight = object_mass * gravity;
        return std::min(1.0f, vertical_support / weight);
    }
};

// =============================================================================
// GRASPING SIMULATION
// =============================================================================

class GraspingSimulation {
public:
    GraspingSimulation(const GraspingTaskConfig& config) : config_(config) {
        // Initialize contact solver
        typename DifferentiableContactSolver<float>::SolverParams solver_params;
        solver_params.max_iterations = 100;  // Increased for better convergence
        solver_params.tolerance = 1e-4f;     // Relaxed from 1e-6
        solver_params.contact_stiffness = 0.001f;  // Very low for gentle contact forces
        solver_params.relaxation = 0.8f;
        solver_params.use_friction = true;

        contact_solver_ = std::make_unique<DifferentiableContactSolver<float>>(solver_params);
    }

    /**
     * Simulate grasp with given finger positions
     * Returns grasp quality metrics
     */
    float simulate(
        const std::vector<ConceptVector3D<float>>& finger_positions,
        std::vector<ConceptVector3D<float>>* final_positions = nullptr,
        std::vector<ConceptVector3D<float>>* contact_normals_out = nullptr) {

        // Initialize state: object + fingers
        std::vector<ConceptVector3D<float>> positions;
        positions.push_back(config_.object_pos);  // Object at index 0
        for (const auto& finger_pos : finger_positions) {
            positions.push_back(finger_pos);
        }

        std::vector<ConceptVector3D<float>> velocities(positions.size());

        std::vector<float> masses;
        masses.push_back(config_.object_mass);
        for (size_t i = 0; i < finger_positions.size(); ++i) {
            // Use very high mass for kinematic fingers (infinite mass approximation)
            masses.push_back(1000.0f);
        }

        std::vector<float> radii;
        radii.push_back(config_.object_radius);
        for (size_t i = 0; i < finger_positions.size(); ++i) {
            radii.push_back(config_.finger_radius);
        }

        SphereContactDetector<float> detector(radii);

        // Collect contact information
        std::vector<ConceptVector3D<float>> contact_positions;
        std::vector<ConceptVector3D<float>> contact_normals;

        // Simulate
        for (int t = 0; t < config_.num_timesteps; ++t) {
            // Detect contacts
            auto contacts = detector.detectContacts(positions);

            // Store contact info
            if (!contacts.empty()) {
                contact_positions.clear();
                contact_normals.clear();
                for (const auto& contact : contacts) {
                    contact_positions.push_back(contact.position);
                    contact_normals.push_back(contact.normal);
                }
            }

            // Set friction
            for (auto& contact : contacts) {
                contact.friction_coefficient = config_.friction_coeff;
            }

            // Solve contacts
            auto solution = contact_solver_->solveContacts(
                contacts, velocities, masses, config_.dt);

            // Apply impulses (even if not fully converged)
            if (!contacts.empty() && !solution.normal_impulses.empty()) {
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

            // Apply gravity to object
            velocities[0][1] -= config_.gravity * config_.dt;

            // Apply velocity damping to prevent numerical instability
            const float damping = 0.95f;
            for (size_t i = 0; i < velocities.size(); ++i) {
                velocities[i] = velocities[i] * damping;
            }

            // Integrate
            for (size_t i = 0; i < positions.size(); ++i) {
                positions[i] = positions[i] + velocities[i] * config_.dt;
            }

            // Fingers stay fixed (gripper doesn't move)
            for (size_t i = 1; i < positions.size(); ++i) {
                positions[i] = finger_positions[i-1];
                velocities[i] = ConceptVector3D<float>{0.0f, 0.0f, 0.0f};
            }
        }

        if (final_positions) {
            *final_positions = positions;
        }

        if (contact_normals_out) {
            *contact_normals_out = contact_normals;
        }

        return 0.0f;  // Quality computed externally
    }

    /**
     * Compute grasp quality for given finger configuration
     */
    float computeGraspQuality(const std::vector<ConceptVector3D<float>>& finger_positions) {
        std::vector<ConceptVector3D<float>> final_positions;
        std::vector<ConceptVector3D<float>> contact_normals;

        simulate(finger_positions, &final_positions, &contact_normals);

        if (contact_normals.empty()) {
            return 1000.0f;  // High penalty for no contact
        }

        // Get contact positions
        std::vector<ConceptVector3D<float>> contact_positions;
        for (size_t i = 1; i < final_positions.size(); ++i) {
            contact_positions.push_back(final_positions[i]);
        }

        // Compute quality metrics
        float force_closure = GraspQualityMetrics::computeForceClosureQuality(
            contact_positions, contact_normals, config_.object_pos);

        float normal_quality = GraspQualityMetrics::computeContactNormalQuality(
            contact_positions, contact_normals, config_.object_pos);

        // Object displacement penalty (should stay in place)
        auto obj_displacement = final_positions[0] - config_.object_pos;
        float displacement_penalty = obj_displacement[0]*obj_displacement[0] +
                                    obj_displacement[1]*obj_displacement[1] +
                                    obj_displacement[2]*obj_displacement[2];

        // Combined loss (minimize)
        float loss = -config_.force_closure_weight * force_closure
                    -config_.contact_normal_weight * normal_quality
                    + config_.penetration_penalty_weight * displacement_penalty;

        return loss;
    }

private:
    GraspingTaskConfig config_;
    std::unique_ptr<DifferentiableContactSolver<float>> contact_solver_;
};

// =============================================================================
// GRASPING OPTIMIZER
// =============================================================================

class GraspingOptimizer {
public:
    GraspingOptimizer(const GraspingTaskConfig& config)
        : config_(config), simulation_(config) {

        // Initialize finger positions (circular arrangement around object)
        finger_positions_.resize(config_.num_fingers);
        for (int i = 0; i < config_.num_fingers; ++i) {
            float angle = 2.0f * M_PI * i / config_.num_fingers;
            // Start in contact: radius < object_radius + finger_radius
            float radius = config_.object_radius + config_.finger_radius * 0.8f;
            finger_positions_[i] = ConceptVector3D<float>{
                config_.object_pos[0] + radius * std::cos(angle),
                config_.object_pos[1],
                config_.object_pos[2] + radius * std::sin(angle)
            };
        }
    }

    void optimize() {
        std::cout << "\n=== Grasping Optimization ===" << std::endl;
        std::cout << "Number of fingers: " << config_.num_fingers << std::endl;
        std::cout << "Object position: (" << config_.object_pos[0] << ", "
                  << config_.object_pos[1] << ", " << config_.object_pos[2] << ")" << std::endl;
        std::cout << "\nIteration | Loss | Finger Spread" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        for (int iter = 0; iter < config_.num_optimization_iters; ++iter) {
            // Compute current loss
            float loss = simulation_.computeGraspQuality(finger_positions_);

            // Compute gradients via finite differences
            std::vector<ConceptVector3D<float>> gradients(config_.num_fingers);

            const float epsilon = 1e-4f;
            for (int f = 0; f < config_.num_fingers; ++f) {
                for (int dim = 0; dim < 3; ++dim) {
                    auto original = finger_positions_[f][dim];

                    finger_positions_[f][dim] = original + epsilon;
                    float loss_plus = simulation_.computeGraspQuality(finger_positions_);

                    finger_positions_[f][dim] = original - epsilon;
                    float loss_minus = simulation_.computeGraspQuality(finger_positions_);

                    finger_positions_[f][dim] = original;

                    gradients[f][dim] = (loss_plus - loss_minus) / (2.0f * epsilon);
                }
            }

            // Clip gradients to prevent explosion
            const float max_gradient = 1.0f;  // Reduced from 10.0 for more stability
            for (int f = 0; f < config_.num_fingers; ++f) {
                for (int dim = 0; dim < 3; ++dim) {
                    if (gradients[f][dim] > max_gradient) gradients[f][dim] = max_gradient;
                    if (gradients[f][dim] < -max_gradient) gradients[f][dim] = -max_gradient;
                }
            }

            // Gradient descent update
            for (int f = 0; f < config_.num_fingers; ++f) {
                finger_positions_[f] = finger_positions_[f] - gradients[f] * config_.learning_rate;

                // Constrain fingers to stay within reasonable distance from object
                auto to_finger = finger_positions_[f] - config_.object_pos;
                float dist = std::sqrt(to_finger[0]*to_finger[0] + to_finger[1]*to_finger[1] + to_finger[2]*to_finger[2]);
                if (dist > 0.3f) {  // Max 30cm from object
                    finger_positions_[f] = config_.object_pos + to_finger * (0.3f / dist);
                }
                if (dist < 0.08f) {  // Min 8cm (allow some contact but not too deep)
                    finger_positions_[f] = config_.object_pos + to_finger * (0.08f / dist);
                }
            }

            // Compute finger spread (avg distance from object)
            float avg_distance = 0.0f;
            for (const auto& pos : finger_positions_) {
                auto diff = pos - config_.object_pos;
                avg_distance += std::sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
            }
            avg_distance /= config_.num_fingers;

            if (iter % config_.print_every == 0) {
                std::cout << std::setw(9) << iter << " | "
                          << std::setw(10) << std::fixed << std::setprecision(6) << loss << " | "
                          << std::setw(10) << avg_distance << std::endl;
            }

            // Early stopping
            if (loss < -0.9f) {  // Good grasp quality
                std::cout << "\nGood grasp found at iteration " << iter << "!" << std::endl;
                break;
            }
        }

        // Final result
        std::cout << "\n=== Final Grasp Configuration ===" << std::endl;
        float final_loss = simulation_.computeGraspQuality(finger_positions_);
        std::cout << "Final loss: " << final_loss << std::endl;

        for (int i = 0; i < config_.num_fingers; ++i) {
            std::cout << "Finger " << i << ": ("
                      << finger_positions_[i][0] << ", "
                      << finger_positions_[i][1] << ", "
                      << finger_positions_[i][2] << ")" << std::endl;
        }
    }

private:
    GraspingTaskConfig config_;
    GraspingSimulation simulation_;
    std::vector<ConceptVector3D<float>> finger_positions_;
};

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PhysGrad Manipulation Demo 2: Grasping Optimization" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Configure task
    GraspingTaskConfig config;
    config.num_fingers = 3;
    config.num_optimization_iters = 100;
    config.learning_rate = 0.0005f;  // Very small for stability with contact dynamics
    config.print_every = 10;

    // Run optimization
    GraspingOptimizer optimizer(config);
    optimizer.optimize();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Demo Complete!" << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;

    return 0;
}
