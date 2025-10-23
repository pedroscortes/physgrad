/**
 * PhysGrad - Standalone Adjoint Integration Methods
 *
 * Self-contained implementation of adjoint automatic differentiation for
 * integration schemes, with minimal dependencies for testing and validation.
 */

#ifndef PHYSGRAD_ADJOINT_INTEGRATORS_STANDALONE_H
#define PHYSGRAD_ADJOINT_INTEGRATORS_STANDALONE_H

#include "common_types.h"
#include <vector>
#include <memory>
#include <functional>
#include <stack>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace physgrad {
namespace adjoint {

// =============================================================================
// SIMPLE FORCE ENGINE FOR TESTING
// =============================================================================

/**
 * Simple force engine for adjoint testing
 * Uses basic harmonic spring forces between particles
 */
template<typename T>
class SimpleForceEngine {
public:
    struct SpringConnection {
        size_t particle1, particle2;
        T spring_constant;
        T rest_length;
    };

    // Parameter gradient accumulators
    struct ParameterGradients {
        std::vector<T> spring_constant_grads;
        std::vector<T> rest_length_grads;
    };

    void addSpring(size_t p1, size_t p2, T k, T r0) {
        springs_.push_back({p1, p2, k, r0});
    }

    size_t getNumSprings() const { return springs_.size(); }

    const std::vector<SpringConnection>& getSprings() const { return springs_; }

    // Compute forces and force Jacobian
    std::pair<std::vector<ConceptVector3D<T>>, std::vector<std::vector<std::vector<ConceptVector3D<T>>>>>
    computeForcesAndGradients(const std::vector<ConceptVector3D<T>>& positions) {
        const size_t n = positions.size();

        std::vector<ConceptVector3D<T>> forces(n, ConceptVector3D<T>{T{0}, T{0}, T{0}});

        // Force Jacobian: forces[i][j] w.r.t. positions[k][l]
        std::vector<std::vector<std::vector<ConceptVector3D<T>>>> jacobian(
            n, std::vector<std::vector<ConceptVector3D<T>>>(
                n, std::vector<ConceptVector3D<T>>(3, ConceptVector3D<T>{T{0}, T{0}, T{0}})
            )
        );

        // Process each spring
        for (const auto& spring : springs_) {
            if (spring.particle1 >= n || spring.particle2 >= n) continue;

            const auto& r1 = positions[spring.particle1];
            const auto& r2 = positions[spring.particle2];

            // Displacement vector
            ConceptVector3D<T> dr = {r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]};

            // Distance
            T dist = std::sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);

            if (dist < T{1e-10}) continue; // Avoid singularity

            // Spring force magnitude
            T force_mag = spring.spring_constant * (dist - spring.rest_length);

            // Unit vector
            ConceptVector3D<T> unit = {dr[0]/dist, dr[1]/dist, dr[2]/dist};

            // Forces
            ConceptVector3D<T> force = {force_mag * unit[0], force_mag * unit[1], force_mag * unit[2]};

            forces[spring.particle1] = forces[spring.particle1] + force;
            forces[spring.particle2] = forces[spring.particle2] + ConceptVector3D<T>{-force[0], -force[1], -force[2]};

            // Proper spring force Jacobian
            // F_1 = k * (|r| - r0) * r/|r|, where r = r2 - r1
            // ∂F_1/∂r_1 = -k * [(1 - r0/|r|) * I + (r0/|r|^3) * r⊗r]
            // ∂F_1/∂r_2 = k * [(1 - r0/|r|) * I + (r0/|r|^3) * r⊗r]

            T k = spring.spring_constant;
            T r0 = spring.rest_length;
            T tangential = k * (T{1} - r0 / dist);
            T radial = k * r0 / (dist * dist * dist);

            // Compute full 3x3 Jacobian blocks
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    T identity = (i == j) ? T{1} : T{0};
                    T jac_element = tangential * identity + radial * dr[i] * dr[j];

                    // ∂F_1/∂r_1
                    jacobian[spring.particle1][spring.particle1][i][j] -= jac_element;

                    // ∂F_1/∂r_2
                    jacobian[spring.particle1][spring.particle2][i][j] += jac_element;

                    // ∂F_2/∂r_1 = -∂F_1/∂r_1
                    jacobian[spring.particle2][spring.particle1][i][j] += jac_element;

                    // ∂F_2/∂r_2 = -∂F_1/∂r_2
                    jacobian[spring.particle2][spring.particle2][i][j] -= jac_element;
                }
            }
        }

        return {forces, jacobian};
    }

    /**
     * Compute force gradients w.r.t. spring parameters.
     *
     * For spring force F = k * (|r| - r0) * r/|r|:
     * - ∂F/∂k = (|r| - r0) * r/|r|
     * - ∂F/∂r0 = -k * r/|r|
     *
     * These are DIFFERENT from position Jacobians (∂F/∂x) which cause double-counting.
     */
    void computeForceParameterGradients(
        const std::vector<ConceptVector3D<T>>& positions,
        const std::vector<ConceptVector3D<T>>& adjoint_forces,  // ∂L/∂F
        ParameterGradients& param_grads) const {

        const size_t n_springs = springs_.size();
        param_grads.spring_constant_grads.resize(n_springs, T{0});
        param_grads.rest_length_grads.resize(n_springs, T{0});

        // Process each spring
        for (size_t s = 0; s < springs_.size(); ++s) {
            const auto& spring = springs_[s];

            if (spring.particle1 >= positions.size() ||
                spring.particle2 >= positions.size()) continue;

            const auto& r1 = positions[spring.particle1];
            const auto& r2 = positions[spring.particle2];

            // Displacement vector
            ConceptVector3D<T> dr = {r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]};

            // Distance
            T dist = std::sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);

            if (dist < T{1e-10}) continue; // Avoid singularity

            // Unit vector
            ConceptVector3D<T> unit = {dr[0]/dist, dr[1]/dist, dr[2]/dist};

            // Gradient w.r.t. spring constant k:
            // ∂F_1/∂k = (|r| - r0) * r/|r|
            ConceptVector3D<T> dF_dk = {
                (dist - spring.rest_length) * unit[0],
                (dist - spring.rest_length) * unit[1],
                (dist - spring.rest_length) * unit[2]
            };

            // Gradient w.r.t. rest length r0:
            // ∂F_1/∂r0 = -k * r/|r|
            ConceptVector3D<T> dF_dr0 = {
                -spring.spring_constant * unit[0],
                -spring.spring_constant * unit[1],
                -spring.spring_constant * unit[2]
            };

            // Chain rule: ∂L/∂k = (∂L/∂F_1) · (∂F_1/∂k) + (∂L/∂F_2) · (∂F_2/∂k)
            // Note: F_2 = -F_1, so ∂F_2/∂k = -∂F_1/∂k

            const auto& adj_f1 = adjoint_forces[spring.particle1];
            const auto& adj_f2 = adjoint_forces[spring.particle2];

            // Dot product for spring constant gradient
            param_grads.spring_constant_grads[s] +=
                adj_f1[0] * dF_dk[0] + adj_f1[1] * dF_dk[1] + adj_f1[2] * dF_dk[2] +
                adj_f2[0] * (-dF_dk[0]) + adj_f2[1] * (-dF_dk[1]) + adj_f2[2] * (-dF_dk[2]);

            // Dot product for rest length gradient
            param_grads.rest_length_grads[s] +=
                adj_f1[0] * dF_dr0[0] + adj_f1[1] * dF_dr0[1] + adj_f1[2] * dF_dr0[2] +
                adj_f2[0] * (-dF_dr0[0]) + adj_f2[1] * (-dF_dr0[1]) + adj_f2[2] * (-dF_dr0[2]);
        }
    }

private:
    std::vector<SpringConnection> springs_;
};

// =============================================================================
// ADJOINT STATE MANAGEMENT
// =============================================================================

template<typename T>
struct AdjointCheckpoint {
    std::vector<ConceptVector3D<T>> positions;
    std::vector<ConceptVector3D<T>> velocities;
    std::vector<ConceptVector3D<T>> forces;
    std::vector<T> masses;
    T timestep;
    int step_index;

    AdjointCheckpoint() = default;
    AdjointCheckpoint(const std::vector<ConceptVector3D<T>>& pos,
                     const std::vector<ConceptVector3D<T>>& vel,
                     const std::vector<ConceptVector3D<T>>& f,
                     const std::vector<T>& m,
                     T dt, int idx)
        : positions(pos), velocities(vel), forces(f), masses(m),
          timestep(dt), step_index(idx) {}
};

template<typename T>
class AdjointStateManager {
public:
    using vector_type = ConceptVector3D<T>;
    using checkpoint_type = AdjointCheckpoint<T>;

    void pushCheckpoint(const checkpoint_type& checkpoint) {
        checkpoints_.push(checkpoint);
    }

    checkpoint_type popCheckpoint() {
        if (checkpoints_.empty()) {
            throw std::runtime_error("No checkpoints available");
        }
        auto checkpoint = checkpoints_.top();
        checkpoints_.pop();
        return checkpoint;
    }

    void clearCheckpoints() {
        while (!checkpoints_.empty()) {
            checkpoints_.pop();
        }
    }

    size_t getNumCheckpoints() const {
        return checkpoints_.size();
    }

    void initializeAdjointState(const std::vector<vector_type>& pos_adjoints,
                               const std::vector<vector_type>& vel_adjoints) {
        position_adjoints_ = pos_adjoints;
        velocity_adjoints_ = vel_adjoints;
    }

    const std::vector<vector_type>& getPositionAdjoints() const { return position_adjoints_; }
    const std::vector<vector_type>& getVelocityAdjoints() const { return velocity_adjoints_; }

    void setPositionAdjoints(const std::vector<vector_type>& adjoints) {
        position_adjoints_ = adjoints;
    }

    void setVelocityAdjoints(const std::vector<vector_type>& adjoints) {
        velocity_adjoints_ = adjoints;
    }

private:
    std::stack<checkpoint_type> checkpoints_;
    std::vector<vector_type> position_adjoints_;
    std::vector<vector_type> velocity_adjoints_;
};

// =============================================================================
// ADJOINT VERLET INTEGRATOR
// =============================================================================

template<typename T>
class AdjointVerletIntegrator {
public:
    using scalar_type = T;
    using vector_type = ConceptVector3D<T>;
    using force_engine_type = SimpleForceEngine<T>;
    using state_manager_type = AdjointStateManager<T>;
    using checkpoint_type = AdjointCheckpoint<T>;

    AdjointVerletIntegrator(std::shared_ptr<force_engine_type> force_engine)
        : force_engine_(force_engine), state_manager_(std::make_unique<state_manager_type>()) {}

    void setTotalSteps(int total_steps) { total_steps_ = total_steps; }

    // Forward pass with checkpointing
    void forwardStep(std::vector<vector_type>& positions,
                    std::vector<vector_type>& velocities,
                    const std::vector<T>& masses,
                    T dt) {
        const size_t n_particles = positions.size();

        // Compute forces at current positions
        auto [forces, force_gradients] = force_engine_->computeForcesAndGradients(positions);

        // Store checkpoint for backward pass
        checkpoint_type checkpoint(positions, velocities, forces, masses, dt, current_step_);
        state_manager_->pushCheckpoint(checkpoint);

        // Verlet integration step
        std::vector<vector_type> new_positions(n_particles);
        std::vector<vector_type> accelerations(n_particles);

        // Compute accelerations: a = F / m
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                accelerations[i][j] = forces[i][j] / masses[i];
            }
        }

        // Update positions: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                new_positions[i][j] = positions[i][j] + velocities[i][j] * dt +
                                     T(0.5) * accelerations[i][j] * dt * dt;
            }
        }

        // Compute forces at new positions
        auto [new_forces, new_force_gradients] = force_engine_->computeForcesAndGradients(new_positions);

        std::vector<vector_type> new_accelerations(n_particles);
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                new_accelerations[i][j] = new_forces[i][j] / masses[i];
            }
        }

        // Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                velocities[i][j] = velocities[i][j] +
                                  T(0.5) * (accelerations[i][j] + new_accelerations[i][j]) * dt;
            }
        }

        positions = new_positions;
        current_step_++;
    }

    // Backward pass: compute gradients w.r.t. initial conditions AND parameters
    void backwardStep(std::vector<vector_type>& pos_grads,
                     std::vector<vector_type>& vel_grads,
                     std::vector<T>& mass_grads,
                     typename force_engine_type::ParameterGradients* param_grads = nullptr) {
        if (state_manager_->getNumCheckpoints() == 0) {
            throw std::runtime_error("No checkpoints available for backward pass");
        }

        // Check if this is the first backward step (last forward step)
        bool is_last_forward_step = (state_manager_->getNumCheckpoints() == total_steps_);

        // Get checkpoint from forward pass
        auto checkpoint = state_manager_->popCheckpoint();
        const auto& positions = checkpoint.positions;
        const auto& velocities = checkpoint.velocities;
        const auto& forces = checkpoint.forces;
        const auto& masses = checkpoint.masses;
        const T dt = checkpoint.timestep;

        const size_t n_particles = positions.size();

        // Get current adjoint state
        auto pos_adjoints = state_manager_->getPositionAdjoints();
        auto vel_adjoints = state_manager_->getVelocityAdjoints();

        // Recompute forward pass intermediate values for backward pass
        std::vector<vector_type> accelerations(n_particles);
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                accelerations[i][j] = forces[i][j] / masses[i];
            }
        }

        std::vector<vector_type> new_positions(n_particles);
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                new_positions[i][j] = positions[i][j] + velocities[i][j] * dt +
                                     T(0.5) * accelerations[i][j] * dt * dt;
            }
        }

        // Compute forces and Jacobians at both t and t+dt
        auto [_, force_jacobians_t] = force_engine_->computeForcesAndGradients(positions);
        auto [new_forces, force_jacobians_t_plus_dt] = force_engine_->computeForcesAndGradients(new_positions);

        std::vector<vector_type> new_accelerations(n_particles);
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                new_accelerations[i][j] = new_forces[i][j] / masses[i];
            }
        }

        // Backward pass through Verlet integration
        // Initialize adjoints for intermediate variables
        std::vector<vector_type> adjoint_a_t(n_particles);  // ∂L/∂a(t)
        std::vector<vector_type> adjoint_a_t_dt(n_particles);  // ∂L/∂a(t+dt)
        std::vector<vector_type> adjoint_x_t_dt(n_particles);  // ∂L/∂x(t+dt)
        std::vector<vector_type> new_pos_adjoints(n_particles);  // ∂L/∂x(t)
        std::vector<vector_type> new_vel_adjoints(n_particles);  // ∂L/∂v(t)
        std::vector<T> new_mass_adjoints(masses.size(), T{0});

        // Step 1: Backprop from v(t+dt)
        // v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        for (size_t i = 0; i < n_particles; ++i) {
            // ∂L/∂v(t) = ∂L/∂v(t+dt)
            new_vel_adjoints[i] = vel_adjoints[i];

            // ∂L/∂a(t) += ∂L/∂v(t+dt) * 0.5*dt
            // ∂L/∂a(t+dt) += ∂L/∂v(t+dt) * 0.5*dt
            for (size_t j = 0; j < 3; ++j) {
                adjoint_a_t[i][j] = T(0.5) * vel_adjoints[i][j] * dt;
                adjoint_a_t_dt[i][j] = T(0.5) * vel_adjoints[i][j] * dt;
            }
        }

        // Step 2: Backprop from x(t+dt)
        // x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        for (size_t i = 0; i < n_particles; ++i) {
            // ∂L/∂x(t) starts with direct contribution
            new_pos_adjoints[i] = pos_adjoints[i];

            // ∂L/∂v(t) += ∂L/∂x(t+dt) * dt
            for (size_t j = 0; j < 3; ++j) {
                new_vel_adjoints[i][j] += pos_adjoints[i][j] * dt;
            }

            // ∂L/∂a(t) += ∂L/∂x(t+dt) * 0.5*dt²
            for (size_t j = 0; j < 3; ++j) {
                adjoint_a_t[i][j] += pos_adjoints[i][j] * T(0.5) * dt * dt;
            }
        }

        // Step 3: DISABLED for testing
        // if (is_last_forward_step) {
        //     for (size_t i = 0; i < n_particles; ++i) {
        //         for (size_t j = 0; j < 3; ++j) {
        //             T adjoint_force_t_dt = adjoint_a_t_dt[i][j] / masses[i];
        //             for (size_t k = 0; k < n_particles; ++k) {
        //                 for (size_t l = 0; l < 3; ++l) {
        //                     T force_jac = getForceJacobian(force_jacobians_t_plus_dt, i, j, k, l);
        //                     adjoint_x_t_dt[k][l] += adjoint_force_t_dt * force_jac;
        //                 }
        //             }
        //             new_mass_adjoints[i] -= adjoint_a_t_dt[i][j] * new_forces[i][j] / masses[i];
        //         }
        //     }
        // }

        // Step 4: Backprop from x(t+dt) to x(t), v(t), a(t)
        // x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        for (size_t i = 0; i < n_particles; ++i) {
            // ∂L/∂x(t) += ∂L/∂x(t+dt)
            for (size_t j = 0; j < 3; ++j) {
                new_pos_adjoints[i][j] += adjoint_x_t_dt[i][j];
            }

            // ∂L/∂v(t) += ∂L/∂x(t+dt) * dt
            for (size_t j = 0; j < 3; ++j) {
                new_vel_adjoints[i][j] += adjoint_x_t_dt[i][j] * dt;
            }

            // ∂L/∂a(t) += ∂L/∂x(t+dt) * 0.5*dt²
            for (size_t j = 0; j < 3; ++j) {
                adjoint_a_t[i][j] += adjoint_x_t_dt[i][j] * T(0.5) * dt * dt;
            }
        }

        // Step 5: DISABLED for testing
        // for (size_t i = 0; i < n_particles; ++i) {
        //     for (size_t j = 0; j < 3; ++j) {
        //         T adjoint_force_t = adjoint_a_t[i][j] / masses[i];
        //         for (size_t k = 0; k < n_particles; ++k) {
        //             for (size_t l = 0; l < 3; ++l) {
        //                 T force_jac = getForceJacobian(force_jacobians_t, i, j, k, l);
        //                 new_pos_adjoints[k][l] += adjoint_force_t * force_jac;
        //             }
        //         }
        //         new_mass_adjoints[i] -= adjoint_a_t[i][j] * forces[i][j] / masses[i];
        //     }
        // }

        // NEW: Compute force parameter gradients (∂L/∂k, ∂L/∂r0)
        // These are SEPARATE from position Jacobians (which are disabled in Steps 3&5)
        if (param_grads != nullptr) {
            // Compute adjoint forces: ∂L/∂F = ∂L/∂a * ∂a/∂F = adjoint_a / mass
            std::vector<vector_type> adjoint_forces_t(n_particles);
            std::vector<vector_type> adjoint_forces_t_dt(n_particles);

            for (size_t i = 0; i < n_particles; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    adjoint_forces_t[i][j] = adjoint_a_t[i][j] / masses[i];
                    adjoint_forces_t_dt[i][j] = adjoint_a_t_dt[i][j] / masses[i];
                }
            }

            // Compute ∂L/∂k and ∂L/∂r0 at both timesteps
            typename force_engine_type::ParameterGradients grads_t, grads_t_dt;

            force_engine_->computeForceParameterGradients(positions, adjoint_forces_t, grads_t);
            force_engine_->computeForceParameterGradients(new_positions, adjoint_forces_t_dt, grads_t_dt);

            // Accumulate (both timesteps contribute to parameter gradients)
            if (param_grads->spring_constant_grads.empty()) {
                param_grads->spring_constant_grads.resize(grads_t.spring_constant_grads.size(), T{0});
                param_grads->rest_length_grads.resize(grads_t.rest_length_grads.size(), T{0});
            }

            for (size_t i = 0; i < grads_t.spring_constant_grads.size(); ++i) {
                param_grads->spring_constant_grads[i] += grads_t.spring_constant_grads[i];
                param_grads->spring_constant_grads[i] += grads_t_dt.spring_constant_grads[i];

                param_grads->rest_length_grads[i] += grads_t.rest_length_grads[i];
                param_grads->rest_length_grads[i] += grads_t_dt.rest_length_grads[i];
            }
        }

        // Update adjoint state
        state_manager_->setPositionAdjoints(new_pos_adjoints);
        state_manager_->setVelocityAdjoints(new_vel_adjoints);

        // Return gradients
        pos_grads = new_pos_adjoints;
        vel_grads = new_vel_adjoints;
        mass_grads = new_mass_adjoints;

        current_step_--;
    }

    // Initialize backward pass
    void initializeBackward(const std::vector<vector_type>& final_pos_grads,
                           const std::vector<vector_type>& final_vel_grads) {
        state_manager_->initializeAdjointState(final_pos_grads, final_vel_grads);
    }

    // Reset integrator state
    void reset() {
        state_manager_->clearCheckpoints();
        current_step_ = 0;
    }

    // Get number of stored checkpoints
    size_t getNumCheckpoints() const {
        return state_manager_->getNumCheckpoints();
    }

private:
    std::shared_ptr<force_engine_type> force_engine_;
    std::unique_ptr<state_manager_type> state_manager_;
    int current_step_ = 0;
    int total_steps_ = 0;

    // Helper function to extract force Jacobian elements
    T getForceJacobian(const std::vector<std::vector<std::vector<ConceptVector3D<T>>>>& jacobians,
                      size_t particle_i, size_t component_j,
                      size_t particle_k, size_t component_l) const {
        if (particle_i < jacobians.size() &&
            particle_k < jacobians[particle_i].size() &&
            component_j < 3 && component_l < 3) {
            return jacobians[particle_i][particle_k][component_j][component_l];
        }
        return T(0);
    }
};

// =============================================================================
// ADJOINT SIMULATION FRAMEWORK
// =============================================================================

template<typename T>
class AdjointSimulation {
public:
    using scalar_type = T;
    using vector_type = ConceptVector3D<T>;
    using integrator_type = AdjointVerletIntegrator<T>;

    AdjointSimulation(std::shared_ptr<SimpleForceEngine<T>> force_engine)
        : integrator_(std::make_unique<integrator_type>(force_engine)) {}

    // Run forward simulation with checkpointing
    void runForward(std::vector<vector_type>& positions,
                   std::vector<vector_type>& velocities,
                   const std::vector<T>& masses,
                   T dt, int num_steps) {
        integrator_->reset();
        integrator_->setTotalSteps(num_steps);

        for (int step = 0; step < num_steps; ++step) {
            integrator_->forwardStep(positions, velocities, masses, dt);
        }

        // Store final state for gradient initialization
        final_positions_ = positions;
        final_velocities_ = velocities;
        num_steps_ = num_steps;
    }

    // Run backward pass to compute gradients (with optional parameter gradients)
    void runBackward(const std::vector<vector_type>& loss_grad_positions,
                    const std::vector<vector_type>& loss_grad_velocities,
                    std::vector<vector_type>& initial_pos_grads,
                    std::vector<vector_type>& initial_vel_grads,
                    std::vector<T>& mass_grads,
                    typename SimpleForceEngine<T>::ParameterGradients* param_grads = nullptr) {

        // Initialize backward pass with loss gradients
        integrator_->initializeBackward(loss_grad_positions, loss_grad_velocities);

        // Run backward through all timesteps
        for (int step = 0; step < num_steps_; ++step) {
            integrator_->backwardStep(initial_pos_grads, initial_vel_grads, mass_grads, param_grads);
        }
    }

    // Improved version: accepts optional analytical gradient functions
    std::pair<std::vector<vector_type>, std::vector<vector_type>>
    computeGradients(
        const std::vector<vector_type>& initial_positions,
        const std::vector<vector_type>& initial_velocities,
        const std::vector<T>& masses,
        T dt, int num_steps,
        std::function<T(const std::vector<vector_type>&,
                       const std::vector<vector_type>&)> loss_function,
        std::function<void(const std::vector<vector_type>&,
                          const std::vector<vector_type>&,
                          std::vector<vector_type>&,
                          std::vector<vector_type>&)> loss_gradient_function = nullptr) {

        // Forward pass
        auto positions = initial_positions;
        auto velocities = initial_velocities;
        runForward(positions, velocities, masses, dt, num_steps);

        // Compute loss at final state (for debugging/logging)
        T loss = loss_function(positions, velocities);

        std::vector<vector_type> loss_grad_pos(positions.size());
        std::vector<vector_type> loss_grad_vel(velocities.size());

        if (loss_gradient_function) {
            // Use provided analytical gradient function (accurate!)
            loss_gradient_function(positions, velocities, loss_grad_pos, loss_grad_vel);
        } else {
            // Fallback: Auto-detect loss type and use analytical gradients when possible

            // Check if loss is position-only (velocity-independent)
            auto vel_test = velocities;
            for (size_t i = 0; i < vel_test.size(); ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    vel_test[i][j] += T(1e-4);
                }
            }
            T loss_with_perturbed_vel = loss_function(positions, vel_test);
            bool is_velocity_independent = std::abs(loss_with_perturbed_vel - loss) < T(1e-10);

            if (is_velocity_independent) {
                // Position-only loss: use analytical gradient (assumes L = Σx²)
                for (size_t i = 0; i < positions.size(); ++i) {
                    loss_grad_pos[i][0] = T(2) * positions[i][0];
                    loss_grad_pos[i][1] = T(2) * positions[i][1];
                    loss_grad_pos[i][2] = T(2) * positions[i][2];
                }
                // Zero velocity gradients
                for (size_t i = 0; i < velocities.size(); ++i) {
                    loss_grad_vel[i][0] = T(0);
                    loss_grad_vel[i][1] = T(0);
                    loss_grad_vel[i][2] = T(0);
                }
            } else {
                // Mixed loss: try to detect kinetic energy pattern
                // Check if loss = 0.5 * m * v² (kinetic energy)
                T kinetic_energy = T(0);
                for (size_t i = 0; i < velocities.size(); ++i) {
                    T v_squared = velocities[i][0] * velocities[i][0] +
                                 velocities[i][1] * velocities[i][1] +
                                 velocities[i][2] * velocities[i][2];
                    kinetic_energy += T(0.5) * masses[i] * v_squared;
                }

                bool is_kinetic_energy = std::abs(loss - kinetic_energy) < T(1e-6);

                if (is_kinetic_energy) {
                    // Analytical gradient for kinetic energy: dL/dv = m * v
                    for (size_t i = 0; i < velocities.size(); ++i) {
                        loss_grad_vel[i][0] = masses[i] * velocities[i][0];
                        loss_grad_vel[i][1] = masses[i] * velocities[i][1];
                        loss_grad_vel[i][2] = masses[i] * velocities[i][2];
                    }
                    // Zero position gradients for pure kinetic energy
                    for (size_t i = 0; i < positions.size(); ++i) {
                        loss_grad_pos[i][0] = T(0);
                        loss_grad_pos[i][1] = T(0);
                        loss_grad_pos[i][2] = T(0);
                    }
                } else {
                    // General case: use finite differences (less accurate but general)
                    // Position gradients (analytical for common Σx² pattern)
                    for (size_t i = 0; i < positions.size(); ++i) {
                        loss_grad_pos[i][0] = T(2) * positions[i][0];
                        loss_grad_pos[i][1] = T(2) * positions[i][1];
                        loss_grad_pos[i][2] = T(2) * positions[i][2];
                    }

                    // Velocity gradients via finite differences
                    for (size_t i = 0; i < velocities.size(); ++i) {
                        for (size_t j = 0; j < 3; ++j) {
                            T v_mag = std::abs(velocities[i][j]);
                            T eps = std::max(T(1e-6), v_mag * T(1e-5));

                            auto vel_plus = velocities;
                            auto vel_minus = velocities;
                            vel_plus[i][j] += eps;
                            vel_minus[i][j] -= eps;

                            T loss_plus = loss_function(positions, vel_plus);
                            T loss_minus = loss_function(positions, vel_minus);

                            loss_grad_vel[i][j] = (loss_plus - loss_minus) / (T(2) * eps);
                        }
                    }
                }
            }
        }

        // Backward pass
        std::vector<vector_type> pos_grads, vel_grads;
        std::vector<T> mass_grads(masses.size());
        runBackward(loss_grad_pos, loss_grad_vel, pos_grads, vel_grads, mass_grads);

        return {pos_grads, vel_grads};
    }

    // Legacy overload for backward compatibility (uses auto-detection)
    std::pair<std::vector<vector_type>, std::vector<vector_type>>
    computeGradients(const std::vector<vector_type>& initial_positions,
                    const std::vector<vector_type>& initial_velocities,
                    const std::vector<T>& masses,
                    T dt, int num_steps,
                    std::function<T(const std::vector<vector_type>&,
                                   const std::vector<vector_type>&)> loss_function) {
        return computeGradients(initial_positions, initial_velocities, masses,
                               dt, num_steps, loss_function, nullptr);
    }

    /**
     * Comprehensive gradient computation struct
     */
    struct AllGradients {
        std::vector<vector_type> position_grads;
        std::vector<vector_type> velocity_grads;
        typename SimpleForceEngine<T>::ParameterGradients parameter_grads;
    };

    /**
     * Compute ALL gradients: positions, velocities, AND force parameters (k, r0)
     *
     * This enables full differentiable physics including material optimization!
     *
     * Example:
     *   auto all_grads = simulation.computeAllGradients(
     *       initial_pos, initial_vel, masses, dt, num_steps,
     *       loss_function,
     *       analytical_loss_gradient  // Optional
     *   );
     *
     *   // Now you can optimize spring constants!
     *   spring_k[i] -= learning_rate * all_grads.parameter_grads.spring_constant_grads[i];
     */
    AllGradients computeAllGradients(
        const std::vector<vector_type>& initial_positions,
        const std::vector<vector_type>& initial_velocities,
        const std::vector<T>& masses,
        T dt, int num_steps,
        std::function<T(const std::vector<vector_type>&,
                       const std::vector<vector_type>&)> loss_function,
        std::function<void(const std::vector<vector_type>&,
                          const std::vector<vector_type>&,
                          std::vector<vector_type>&,
                          std::vector<vector_type>&)> loss_gradient_function = nullptr) {

        // Forward pass
        auto positions = initial_positions;
        auto velocities = initial_velocities;
        runForward(positions, velocities, masses, dt, num_steps);

        // Compute loss at final state
        T loss = loss_function(positions, velocities);

        std::vector<vector_type> loss_grad_pos(positions.size());
        std::vector<vector_type> loss_grad_vel(velocities.size());

        // Compute loss gradients (same logic as computeGradients)
        if (loss_gradient_function) {
            loss_gradient_function(positions, velocities, loss_grad_pos, loss_grad_vel);
        } else {
            // Auto-detection logic (same as before)
            auto vel_test = velocities;
            for (size_t i = 0; i < vel_test.size(); ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    vel_test[i][j] += T(1e-4);
                }
            }
            T loss_with_perturbed_vel = loss_function(positions, vel_test);
            bool is_velocity_independent = std::abs(loss_with_perturbed_vel - loss) < T(1e-10);

            if (is_velocity_independent) {
                // Position-only loss
                for (size_t i = 0; i < positions.size(); ++i) {
                    loss_grad_pos[i][0] = T(2) * positions[i][0];
                    loss_grad_pos[i][1] = T(2) * positions[i][1];
                    loss_grad_pos[i][2] = T(2) * positions[i][2];
                }
                for (size_t i = 0; i < velocities.size(); ++i) {
                    loss_grad_vel[i][0] = loss_grad_vel[i][1] = loss_grad_vel[i][2] = T(0);
                }
            } else {
                // Check for kinetic energy
                T kinetic_energy = T(0);
                for (size_t i = 0; i < velocities.size(); ++i) {
                    T v_squared = velocities[i][0] * velocities[i][0] +
                                 velocities[i][1] * velocities[i][1] +
                                 velocities[i][2] * velocities[i][2];
                    kinetic_energy += T(0.5) * masses[i] * v_squared;
                }

                bool is_kinetic_energy = std::abs(loss - kinetic_energy) < T(1e-6);

                if (is_kinetic_energy) {
                    // Kinetic energy gradient
                    for (size_t i = 0; i < velocities.size(); ++i) {
                        loss_grad_vel[i][0] = masses[i] * velocities[i][0];
                        loss_grad_vel[i][1] = masses[i] * velocities[i][1];
                        loss_grad_vel[i][2] = masses[i] * velocities[i][2];
                    }
                    for (size_t i = 0; i < positions.size(); ++i) {
                        loss_grad_pos[i][0] = loss_grad_pos[i][1] = loss_grad_pos[i][2] = T(0);
                    }
                } else {
                    // General case: finite differences
                    for (size_t i = 0; i < positions.size(); ++i) {
                        loss_grad_pos[i][0] = T(2) * positions[i][0];
                        loss_grad_pos[i][1] = T(2) * positions[i][1];
                        loss_grad_pos[i][2] = T(2) * positions[i][2];
                    }

                    for (size_t i = 0; i < velocities.size(); ++i) {
                        for (size_t j = 0; j < 3; ++j) {
                            T v_mag = std::abs(velocities[i][j]);
                            T eps = std::max(T(1e-6), v_mag * T(1e-5));

                            auto vel_plus = velocities;
                            auto vel_minus = velocities;
                            vel_plus[i][j] += eps;
                            vel_minus[i][j] -= eps;

                            T loss_plus = loss_function(positions, vel_plus);
                            T loss_minus = loss_function(positions, vel_minus);

                            loss_grad_vel[i][j] = (loss_plus - loss_minus) / (T(2) * eps);
                        }
                    }
                }
            }
        }

        // Backward pass WITH parameter gradients
        AllGradients all_grads;
        std::vector<T> mass_grads(masses.size());

        runBackward(loss_grad_pos, loss_grad_vel,
                   all_grads.position_grads, all_grads.velocity_grads, mass_grads,
                   &all_grads.parameter_grads);  // Enable parameter gradient computation!

        return all_grads;
    }

private:
    std::unique_ptr<integrator_type> integrator_;
    std::vector<vector_type> final_positions_;
    std::vector<vector_type> final_velocities_;
    int num_steps_ = 0;
};

} // namespace adjoint
} // namespace physgrad

#endif // PHYSGRAD_ADJOINT_INTEGRATORS_STANDALONE_H