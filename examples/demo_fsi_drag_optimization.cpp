/**
 * PhysGrad FSI Demo: Cylinder Drag Optimization
 *
 * Demonstrates fluid-structure interaction with gradient-based optimization.
 * Task: Optimize cylinder shape to minimize drag force in fluid flow.
 *
 * Key Features:
 * - Fluid-structure coupling via immersed boundary method
 * - Drag force computation from FSI
 * - Shape parameterization (radius, aspect ratio)
 * - Gradient-based optimization
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>

#include "fsi_coupling.h"
#include "mpm_data_structures.h"

using namespace physgrad;
using namespace physgrad::fsi;

// =============================================================================
// FSI DRAG OPTIMIZATION CONFIGURATION
// =============================================================================

struct FSIDragConfig {
    // Fluid domain
    int fluid_grid_x = 40;
    int fluid_grid_y = 20;
    float fluid_spacing = 0.1f;
    float inlet_velocity = 2.0f;

    // Cylinder (to be optimized)
    float cylinder_center_x = 1.0f;
    float cylinder_center_y = 1.0f;
    float cylinder_radius = 0.2f;  // Initial radius
    float cylinder_aspect_ratio = 1.0f;  // Height/width ratio
    int cylinder_points = 32;

    // Physics parameters
    float dt = 0.001f;
    int num_timesteps = 100;
    float fluid_density = 1000.0f;
    float fluid_viscosity = 0.001f;

    // FSI coupling
    float support_radius = 0.15f;

    // Optimization parameters
    int num_optimization_iters = 30;
    float learning_rate = 0.01f;
    int print_every = 5;
};

// =============================================================================
// FSI DRAG SIMULATION
// =============================================================================

class FSIDragSimulation {
public:
    FSIDragSimulation(const FSIDragConfig& config) : config_(config) {
        // Create FSI coupling method (support_radius, coupling_strength)
        coupling_method_ = std::make_unique<ImmersedBoundaryMethod<float>>(
            config_.support_radius, 1.0f);
    }

    /**
     * Simulate flow around cylinder and compute drag force
     */
    float simulate(float radius, float aspect_ratio) {
        // Create fluid particles (grid)
        mpm::ParticleAoSoA<float> fluid_particles(config_.fluid_grid_x * config_.fluid_grid_y);

        size_t particle_idx = 0;
        for (int i = 0; i < config_.fluid_grid_x; ++i) {
            for (int j = 0; j < config_.fluid_grid_y; ++j) {
                float x = i * config_.fluid_spacing;
                float y = j * config_.fluid_spacing;

                fluid_particles.setPosition(particle_idx, x, y, 0.0);
                fluid_particles.setVelocity(particle_idx,
                                          config_.inlet_velocity, 0.0, 0.0);
                fluid_particles.setMass(particle_idx, config_.fluid_density *
                                       config_.fluid_spacing * config_.fluid_spacing);

                particle_idx++;
            }
        }

        // Create cylinder particles (boundary)
        mpm::ParticleAoSoA<float> cylinder_particles(config_.cylinder_points);

        for (int i = 0; i < config_.cylinder_points; ++i) {
            float angle = 2.0f * M_PI * i / config_.cylinder_points;
            float x = config_.cylinder_center_x + radius * std::cos(angle);
            float y = config_.cylinder_center_y + radius * aspect_ratio * std::sin(angle);

            cylinder_particles.setPosition(i, x, y, 0.0);
            cylinder_particles.setVelocity(i, 0.0, 0.0, 0.0);
            cylinder_particles.setMass(i, 1.0);  // Fixed boundary
        }

        // Simulate
        float total_drag_force = 0.0f;
        int force_samples = 0;

        for (int t = 0; t < config_.num_timesteps; ++t) {
            // Perform FSI coupling
            coupling_method_->couple(fluid_particles, cylinder_particles,
                                    config_.dt, t * config_.dt);

            // Compute drag force (x-direction force on cylinder)
            auto solid_forces = coupling_method_->computeSolidForces(
                fluid_particles, cylinder_particles);

            // Accumulate drag force (only after initial transient)
            if (t > config_.num_timesteps / 2) {
                float drag_x = 0.0f;
                for (size_t i = 0; i < config_.cylinder_points; ++i) {
                    // Forces are stored as [fx0, fy0, fz0, fx1, fy1, fz1, ...]
                    drag_x += solid_forces[i * 3];
                }
                total_drag_force += std::abs(drag_x);
                force_samples++;
            }

            // Update fluid velocities (simple advection)
            for (size_t i = 0; i < fluid_particles.size(); ++i) {
                float vx, vy, vz;
                fluid_particles.getVelocity(i, vx, vy, vz);

                // Maintain inlet velocity on left side
                float x, y, z;
                fluid_particles.getPosition(i, x, y, z);
                if (x < config_.fluid_spacing) {
                    fluid_particles.setVelocity(i, config_.inlet_velocity, 0.0f, 0.0f);
                }
            }
        }

        // Return average drag force
        return (force_samples > 0) ? (total_drag_force / force_samples) : 1e6;
    }

private:
    FSIDragConfig config_;
    std::unique_ptr<ImmersedBoundaryMethod<float>> coupling_method_;
};

// =============================================================================
// FSI DRAG OPTIMIZER
// =============================================================================

class FSIDragOptimizer {
public:
    FSIDragOptimizer(const FSIDragConfig& config)
        : config_(config), simulation_(config) {

        // Initialize with default cylinder shape
        current_radius_ = config_.cylinder_radius;
        current_aspect_ratio_ = config_.cylinder_aspect_ratio;
    }

    void optimize() {
        std::cout << "\n=== FSI Cylinder Drag Optimization ===\n";
        std::cout << "Goal: Minimize drag force on cylinder in flow\n";
        std::cout << "Inlet velocity: " << config_.inlet_velocity << " m/s\n";
        std::cout << "Initial radius: " << current_radius_ << " m\n";
        std::cout << "Initial aspect ratio: " << current_aspect_ratio_ << "\n\n";

        std::cout << "Iteration | Drag Force | Radius | Aspect Ratio\n";
        std::cout << std::string(60, '-') << "\n";

        for (int iter = 0; iter < config_.num_optimization_iters; ++iter) {
            // Compute current drag
            float drag = simulation_.simulate(current_radius_, current_aspect_ratio_);

            // Compute gradients via finite differences
            const float epsilon = 1e-4f;

            // Gradient w.r.t. radius
            float drag_plus_r = simulation_.simulate(current_radius_ + epsilon, current_aspect_ratio_);
            float drag_minus_r = simulation_.simulate(current_radius_ - epsilon, current_aspect_ratio_);
            float grad_radius = (drag_plus_r - drag_minus_r) / (2.0f * epsilon);

            // Gradient w.r.t. aspect ratio
            float drag_plus_a = simulation_.simulate(current_radius_, current_aspect_ratio_ + epsilon);
            float drag_minus_a = simulation_.simulate(current_radius_, current_aspect_ratio_ - epsilon);
            float grad_aspect = (drag_plus_a - drag_minus_a) / (2.0f * epsilon);

            // Gradient descent update
            current_radius_ -= config_.learning_rate * grad_radius;
            current_aspect_ratio_ -= config_.learning_rate * grad_aspect;

            // Constrain parameters to reasonable ranges
            current_radius_ = std::max(0.1f, std::min(0.5f, current_radius_));
            current_aspect_ratio_ = std::max(0.5f, std::min(2.0f, current_aspect_ratio_));

            if (iter % config_.print_every == 0) {
                std::cout << std::setw(9) << iter << " | "
                          << std::setw(10) << std::fixed << std::setprecision(4) << drag << " | "
                          << std::setw(6) << std::setprecision(3) << current_radius_ << " | "
                          << std::setw(12) << std::setprecision(3) << current_aspect_ratio_ << "\n";
            }
        }

        // Final result
        std::cout << "\n=== Optimization Complete ===\n";
        float final_drag = simulation_.simulate(current_radius_, current_aspect_ratio_);
        float initial_drag = simulation_.simulate(config_.cylinder_radius, config_.cylinder_aspect_ratio);

        std::cout << "Initial drag: " << initial_drag << " N\n";
        std::cout << "Final drag: " << final_drag << " N\n";
        std::cout << "Drag reduction: " << (initial_drag - final_drag) / initial_drag * 100.0f << "%\n";
        std::cout << "\nOptimal cylinder shape:\n";
        std::cout << "  Radius: " << current_radius_ << " m\n";
        std::cout << "  Aspect ratio: " << current_aspect_ratio_ << "\n";

        if (final_drag < initial_drag) {
            std::cout << "\n✓ Successfully optimized drag!\n";
        } else {
            std::cout << "\n⚠ Optimization did not reduce drag (may need more iterations)\n";
        }
    }

private:
    FSIDragConfig config_;
    FSIDragSimulation simulation_;
    float current_radius_;
    float current_aspect_ratio_;
};

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "PhysGrad FSI Demo: Cylinder Drag Optimization\n";
    std::cout << std::string(80, '=') << "\n";

    // Configure task
    FSIDragConfig config;
    config.num_optimization_iters = 20;
    config.learning_rate = 0.005f;
    config.print_every = 2;
    config.num_timesteps = 50;  // Shorter for faster demo

    // Run optimization
    FSIDragOptimizer optimizer(config);
    optimizer.optimize();

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Demo Complete!\n";
    std::cout << std::string(80, '=') << "\n\n";

    return 0;
}
