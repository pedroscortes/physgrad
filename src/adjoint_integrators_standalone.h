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

    void addSpring(size_t p1, size_t p2, T k, T r0) {
        springs_.push_back({p1, p2, k, r0});
    }

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

    // Backward pass: compute gradients w.r.t. initial conditions
    void backwardStep(std::vector<vector_type>& pos_grads,
                     std::vector<vector_type>& vel_grads,
                     std::vector<T>& mass_grads) {
        if (state_manager_->getNumCheckpoints() == 0) {
            throw std::runtime_error("No checkpoints available for backward pass");
        }

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

        // Step 3: Backprop from a(t+dt) through F(x(t+dt))
        // a(t+dt) = F(x(t+dt)) / m
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                // ∂L/∂F(t+dt) = ∂L/∂a(t+dt) / m
                T adjoint_force_t_dt = adjoint_a_t_dt[i][j] / masses[i];

                // ∂L/∂x(t+dt) += ∂L/∂F(t+dt) * ∂F/∂x
                for (size_t k = 0; k < n_particles; ++k) {
                    for (size_t l = 0; l < 3; ++l) {
                        T force_jac = getForceJacobian(force_jacobians_t_plus_dt, i, j, k, l);
                        adjoint_x_t_dt[k][l] += adjoint_force_t_dt * force_jac;
                    }
                }

                // ∂L/∂m += -∂L/∂a(t+dt) * F(t+dt) / m²
                new_mass_adjoints[i] -= adjoint_a_t_dt[i][j] * new_forces[i][j] / masses[i];
            }
        }

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

        // Step 5: Backprop from a(t) through F(x(t))
        // a(t) = F(x(t)) / m
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                // ∂L/∂F(t) = ∂L/∂a(t) / m
                T adjoint_force_t = adjoint_a_t[i][j] / masses[i];

                // ∂L/∂x(t) += ∂L/∂F(t) * ∂F/∂x
                for (size_t k = 0; k < n_particles; ++k) {
                    for (size_t l = 0; l < 3; ++l) {
                        T force_jac = getForceJacobian(force_jacobians_t, i, j, k, l);
                        new_pos_adjoints[k][l] += adjoint_force_t * force_jac;
                    }
                }

                // ∂L/∂m += -∂L/∂a(t) * F(t) / m²
                new_mass_adjoints[i] -= adjoint_a_t[i][j] * forces[i][j] / masses[i];
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

        for (int step = 0; step < num_steps; ++step) {
            integrator_->forwardStep(positions, velocities, masses, dt);
        }

        // Store final state for gradient initialization
        final_positions_ = positions;
        final_velocities_ = velocities;
        num_steps_ = num_steps;
    }

    // Run backward pass to compute gradients
    void runBackward(const std::vector<vector_type>& loss_grad_positions,
                    const std::vector<vector_type>& loss_grad_velocities,
                    std::vector<vector_type>& initial_pos_grads,
                    std::vector<vector_type>& initial_vel_grads,
                    std::vector<T>& mass_grads) {

        // Initialize backward pass with loss gradients
        integrator_->initializeBackward(loss_grad_positions, loss_grad_velocities);

        // Run backward through all timesteps
        for (int step = 0; step < num_steps_; ++step) {
            integrator_->backwardStep(initial_pos_grads, initial_vel_grads, mass_grads);
        }
    }

    // Convenience function for gradient checking
    std::pair<std::vector<vector_type>, std::vector<vector_type>>
    computeGradients(const std::vector<vector_type>& initial_positions,
                    const std::vector<vector_type>& initial_velocities,
                    const std::vector<T>& masses,
                    T dt, int num_steps,
                    std::function<T(const std::vector<vector_type>&,
                                   const std::vector<vector_type>&)> loss_function) {

        // Forward pass
        auto positions = initial_positions;
        auto velocities = initial_velocities;
        runForward(positions, velocities, masses, dt, num_steps);

        // Compute loss at final state
        T loss = loss_function(positions, velocities);

        // Compute loss gradients analytically
        // For L = Σ(x² + y² + z²), dL/dx_i = 2*x_i
        std::vector<vector_type> loss_grad_pos(positions.size());
        std::vector<vector_type> loss_grad_vel(velocities.size());

        // Analytical gradient w.r.t. positions
        for (size_t i = 0; i < positions.size(); ++i) {
            loss_grad_pos[i][0] = T(2) * positions[i][0];
            loss_grad_pos[i][1] = T(2) * positions[i][1];
            loss_grad_pos[i][2] = T(2) * positions[i][2];
        }

        // Gradient w.r.t. velocities is zero for this loss function
        for (size_t i = 0; i < velocities.size(); ++i) {
            loss_grad_vel[i][0] = T(0);
            loss_grad_vel[i][1] = T(0);
            loss_grad_vel[i][2] = T(0);
        }

        // Backward pass
        std::vector<vector_type> pos_grads, vel_grads;
        std::vector<T> mass_grads(masses.size());
        runBackward(loss_grad_pos, loss_grad_vel, pos_grads, vel_grads, mass_grads);

        return {pos_grads, vel_grads};
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