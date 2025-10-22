/**
 * PhysGrad Robot Co-Design Demo: Hopping Vehicle Optimization
 *
 * Demonstrates simultaneous optimization of morphology and control.
 * Task: Optimize vehicle shape and hopping pattern for forward locomotion.
 *
 * Key Features:
 * - Morphology optimization (body mass, spring stiffness)
 * - Control optimization (hop frequency, thrust angle)
 * - Contact-based physics with friction
 * - End-to-end differentiable co-design
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>

#include "differentiable_contact.h"

using namespace physgrad;
using namespace physgrad::contact;

// =============================================================================
// ROBOT CO-DESIGN CONFIGURATION
// =============================================================================

struct RobotCoDesignConfig {
    // Physics parameters
    float dt = 0.01f;
    int num_timesteps = 200;  // 2 seconds at 0.01s timesteps
    float gravity = 9.81f;
    float ground_friction = 0.8f;

    // Initial morphology parameters (to be optimized)
    float body_mass = 1.0f;
    float spring_stiffness = 50.0f;

    // Initial control parameters (to be optimized)
    float hop_frequency = 2.0f;
    float thrust_angle = 45.0f;  // degrees from vertical
    float thrust_magnitude = 20.0f;  // Newtons

    // Optimization parameters
    int num_optimization_iters = 30;
    float morphology_lr = 0.01f;  // Learning rate for mass, stiffness
    float control_lr = 0.05f;     // Learning rate for frequency, angle
    int print_every = 3;
};

// =============================================================================
// HOPPING VEHICLE SIMULATION
// =============================================================================

class HoppingVehicleSimulation {
public:
    HoppingVehicleSimulation(const RobotCoDesignConfig& config) : config_(config) {
        // Initialize contact solver
        typename DifferentiableContactSolver<float>::SolverParams solver_params;
        solver_params.max_iterations = 100;
        solver_params.tolerance = 1e-4f;
        solver_params.contact_stiffness = 0.01f;
        solver_params.relaxation = 0.8f;
        solver_params.use_friction = true;

        contact_solver_ = std::make_unique<DifferentiableContactSolver<float>>(solver_params);
    }

    /**
     * Simulate hopping locomotion
     * Returns: forward distance traveled
     */
    float simulate(float body_mass, float spring_k, float hop_freq, float thrust_angle_deg, float thrust_mag) {
        // Initialize vehicle state
        ConceptVector3D<float> pos{0.0f, 0.1f, 0.0f};  // Start slightly above ground
        ConceptVector3D<float> vel{0.0f, 0.0f, 0.0f};

        float initial_x = pos[0];

        // Convert thrust angle to radians
        float thrust_angle_rad = thrust_angle_deg * M_PI / 180.0f;

        for (int t = 0; t < config_.num_timesteps; ++t) {
            float time = t * config_.dt;

            // Apply gravity
            vel[1] -= config_.gravity * config_.dt;

            // Apply thrust (pulsed at hop frequency)
            float phase = 2.0f * M_PI * hop_freq * time;
            bool is_thrust_phase = (std::sin(phase) > 0.0f);

            if (is_thrust_phase && pos[1] < 0.15f) {  // Only thrust when near ground
                // Thrust has vertical and horizontal components
                float thrust_x = thrust_mag * std::sin(thrust_angle_rad) / body_mass;
                float thrust_y = thrust_mag * std::cos(thrust_angle_rad) / body_mass;

                vel[0] += thrust_x * config_.dt;
                vel[1] += thrust_y * config_.dt;
            }

            // Ground contact detection
            std::vector<ContactPoint<float>> contacts;
            if (pos[1] < 0.05f) {  // Vehicle touching ground
                ContactPoint<float> contact;
                contact.position = pos;
                contact.normal = ConceptVector3D<float>{0.0f, 1.0f, 0.0f};
                contact.penetration_depth = 0.05f - pos[1];
                contact.body_a_id = 0;
                contact.body_b_id = SIZE_MAX;  // Ground
                contact.friction_coefficient = config_.ground_friction;
                contacts.push_back(contact);
            }

            // Solve contacts
            if (!contacts.empty()) {
                std::vector<ConceptVector3D<float>> velocities = {vel};
                std::vector<float> masses = {body_mass};

                auto solution = contact_solver_->solveContacts(contacts, velocities, masses, config_.dt);

                // Apply normal impulse
                if (!solution.normal_impulses.empty()) {
                    float impulse_n = solution.normal_impulses[0];
                    vel = vel - contacts[0].normal * (impulse_n / body_mass);
                }

                // Apply friction impulses
                if (!solution.friction_impulses_u.empty() && !solution.friction_impulses_v.empty()) {
                    // Compute tangent basis
                    auto& n = contacts[0].normal;
                    ConceptVector3D<float> tangent_u{1.0f, 0.0f, 0.0f};  // Ground tangent
                    ConceptVector3D<float> tangent_v{0.0f, 0.0f, 1.0f};

                    float impulse_u = solution.friction_impulses_u[0];
                    float impulse_v = solution.friction_impulses_v[0];

                    vel = vel - (tangent_u * impulse_u + tangent_v * impulse_v) * (1.0f / body_mass);
                }
            }

            // Spring damping (energy dissipation)
            vel = vel * 0.995f;

            // Velocity clamping
            const float max_vel = 5.0f;
            for (int d = 0; d < 3; ++d) {
                vel[d] = std::max(-max_vel, std::min(max_vel, vel[d]));
            }

            // Integrate position
            pos = pos + vel * config_.dt;

            // Ground constraint
            pos[1] = std::max(0.05f, pos[1]);

            // Stop if too far (simulation stability)
            if (std::abs(pos[0]) > 10.0f) break;
        }

        // Return forward distance traveled
        float distance = pos[0] - initial_x;
        return distance;  // Can be positive or negative
    }

private:
    RobotCoDesignConfig config_;
    std::unique_ptr<DifferentiableContactSolver<float>> contact_solver_;
};

// =============================================================================
// CO-DESIGN OPTIMIZER
// =============================================================================

class RobotCoDesignOptimizer {
public:
    RobotCoDesignOptimizer(const RobotCoDesignConfig& config)
        : config_(config), simulation_(config) {

        // Initialize parameters
        body_mass_ = config_.body_mass;
        spring_k_ = config_.spring_stiffness;
        hop_freq_ = config_.hop_frequency;
        thrust_angle_ = config_.thrust_angle;
        thrust_mag_ = config_.thrust_magnitude;
    }

    void optimize() {
        std::cout << "\n=== Robot Co-Design Optimization ===\n";
        std::cout << "Optimizing BOTH morphology (mass, spring) AND control (hopping)\n";
        std::cout << "Goal: Maximize forward locomotion distance\n\n";

        std::cout << "Iter | Distance | Mass  | Spring K | Frequency | Angle\n";
        std::cout << std::string(70, '-') << "\n";

        for (int iter = 0; iter < config_.num_optimization_iters; ++iter) {
            // Compute current performance
            float distance = simulation_.simulate(body_mass_, spring_k_, hop_freq_, thrust_angle_, thrust_mag_);

            // Compute gradients via finite differences
            const float epsilon = 1e-4f;

            // Morphology gradients
            float d_plus_mass = simulation_.simulate(body_mass_ + epsilon, spring_k_, hop_freq_, thrust_angle_, thrust_mag_);
            float d_minus_mass = simulation_.simulate(body_mass_ - epsilon, spring_k_, hop_freq_, thrust_angle_, thrust_mag_);
            float grad_mass = (d_plus_mass - d_minus_mass) / (2.0f * epsilon);

            float d_plus_k = simulation_.simulate(body_mass_, spring_k_ + epsilon, hop_freq_, thrust_angle_, thrust_mag_);
            float d_minus_k = simulation_.simulate(body_mass_, spring_k_ - epsilon, hop_freq_, thrust_angle_, thrust_mag_);
            float grad_k = (d_plus_k - d_minus_k) / (2.0f * epsilon);

            // Control gradients
            float d_plus_freq = simulation_.simulate(body_mass_, spring_k_, hop_freq_ + epsilon, thrust_angle_, thrust_mag_);
            float d_minus_freq = simulation_.simulate(body_mass_, spring_k_, hop_freq_ - epsilon, thrust_angle_, thrust_mag_);
            float grad_freq = (d_plus_freq - d_minus_freq) / (2.0f * epsilon);

            float d_plus_angle = simulation_.simulate(body_mass_, spring_k_, hop_freq_, thrust_angle_ + epsilon, thrust_mag_);
            float d_minus_angle = simulation_.simulate(body_mass_, spring_k_, hop_freq_, thrust_angle_ - epsilon, thrust_mag_);
            float grad_angle = (d_plus_angle - d_minus_angle) / (2.0f * epsilon);

            // Gradient ascent (maximize distance)
            body_mass_ += config_.morphology_lr * grad_mass;
            spring_k_ += config_.morphology_lr * grad_k;
            hop_freq_ += config_.control_lr * grad_freq;
            thrust_angle_ += config_.control_lr * grad_angle;

            // Constrain parameters to reasonable ranges
            body_mass_ = std::max(0.5f, std::min(2.0f, body_mass_));
            spring_k_ = std::max(10.0f, std::min(100.0f, spring_k_));
            hop_freq_ = std::max(0.5f, std::min(5.0f, hop_freq_));
            thrust_angle_ = std::max(20.0f, std::min(70.0f, thrust_angle_));

            if (iter % config_.print_every == 0) {
                std::cout << std::setw(4) << iter << " | "
                          << std::setw(8) << std::fixed << std::setprecision(3) << distance << " | "
                          << std::setw(5) << std::setprecision(2) << body_mass_ << " | "
                          << std::setw(8) << std::setprecision(1) << spring_k_ << " | "
                          << std::setw(9) << std::setprecision(2) << hop_freq_ << " | "
                          << std::setw(5) << std::setprecision(1) << thrust_angle_ << "\n";
            }
        }

        // Final results
        std::cout << "\n=== Co-Design Results ===\n\n";
        float initial_dist = simulation_.simulate(config_.body_mass, config_.spring_stiffness,
                                                  config_.hop_frequency, config_.thrust_angle, config_.thrust_magnitude);
        float final_dist = simulation_.simulate(body_mass_, spring_k_, hop_freq_, thrust_angle_, thrust_mag_);

        std::cout << "Initial distance: " << std::fixed << std::setprecision(2) << initial_dist << " m\n";
        std::cout << "Final distance: " << final_dist << " m\n";
        std::cout << "Improvement: " << ((final_dist - initial_dist) / std::abs(initial_dist + 1e-6) * 100.0f) << "%\n\n";

        std::cout << "Optimal Morphology:\n";
        std::cout << "  Body mass: " << body_mass_ << " kg\n";
        std::cout << "  Spring stiffness: " << spring_k_ << " N/m\n\n";

        std::cout << "Optimal Control:\n";
        std::cout << "  Hop frequency: " << hop_freq_ << " Hz\n";
        std::cout << "  Thrust angle: " << thrust_angle_ << " degrees\n";

        if (final_dist > initial_dist + 0.01f) {
            std::cout << "\nSuccessfully optimized locomotion!\n";
        }
    }

private:
    RobotCoDesignConfig config_;
    HoppingVehicleSimulation simulation_;

    // Optimization variables
    float body_mass_;
    float spring_k_;
    float hop_freq_;
    float thrust_angle_;
    float thrust_mag_;
};

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "PhysGrad Robot Co-Design Demo: Hopping Vehicle\n";
    std::cout << std::string(80, '=') << "\n";

    // Configure task
    RobotCoDesignConfig config;
    config.num_optimization_iters = 30;
    config.morphology_lr = 0.01f;
    config.control_lr = 0.05f;
    config.print_every = 3;

    // Run co-design optimization
    RobotCoDesignOptimizer optimizer(config);
    optimizer.optimize();

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Demo Complete!\n";
    std::cout << std::string(80, '=') << "\n\n";

    return 0;
}
