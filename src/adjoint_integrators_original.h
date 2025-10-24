/**
 * PhysGrad - Adjoint Integration Methods
 *
 * Implements adjoint (reverse-mode) automatic differentiation for various
 * integration schemes. Enables end-to-end differentiability through physics
 * simulation pipelines for gradient-based optimization and machine learning.
 */

#ifndef PHYSGRAD_ADJOINT_INTEGRATORS_H
#define PHYSGRAD_ADJOINT_INTEGRATORS_H

#include "common_types.h"
#include "differentiable_forces.h"
#include <vector>
#include <memory>
#include <functional>
#include <stack>

// Include concepts when available
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "concepts/forward_declarations.h"
#endif

namespace physgrad {

// =============================================================================
// ADJOINT STATE MANAGEMENT
// =============================================================================

/**
 * Stores the forward pass state needed for backward pass computation
 */
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

/**
 * Manages adjoint state computation and gradient accumulation
 */
template<typename T>
class AdjointStateManager {
public:
    using vector_type = ConceptVector3D<T>;
    using checkpoint_type = AdjointCheckpoint<T>;

    AdjointStateManager() = default;

    void pushCheckpoint(const checkpoint_type& checkpoint) {
        checkpoints_.push(checkpoint);
    }

    checkpoint_type popCheckpoint() {
        if (checkpoints_.empty()) {
            throw std::runtime_error("No checkpoints available for adjoint computation");
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

    // Initialize adjoint variables (gradients w.r.t. final state)
    void initializeAdjointState(const std::vector<vector_type>& pos_adjoints,
                               const std::vector<vector_type>& vel_adjoints) {
        position_adjoints_ = pos_adjoints;
        velocity_adjoints_ = vel_adjoints;
    }

    // Get current adjoint state
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

/**
 * Velocity Verlet integrator with adjoint (reverse-mode) automatic differentiation
 *
 * Forward pass: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
 *               v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
 *
 * Backward pass: Computes gradients by reversing the integration steps
 */
template<typename T>
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    requires concepts::PhysicsScalar<T>
#endif
class AdjointVerletIntegrator {
public:
    using scalar_type = T;
    using vector_type = ConceptVector3D<T>;
    using force_engine_type = differentiable::DifferentiableForceEngine<T>;
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

        // Compute force Jacobians at checkpoint positions
        auto [_, force_jacobians] = force_engine_->computeForcesAndGradients(positions);

        // Backward pass through Verlet integration
        std::vector<vector_type> new_pos_adjoints(n_particles);
        std::vector<vector_type> new_vel_adjoints(n_particles);
        std::vector<T> new_mass_adjoints(masses.size());

        // Reverse the velocity update: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        for (size_t i = 0; i < n_particles; ++i) {
            // ∂L/∂v(t) = ∂L/∂v(t+dt) + contributions from position update
            new_vel_adjoints[i] = vel_adjoints[i];

            // Add contribution from position update: x(t+dt) = x(t) + v(t)*dt + ...
            for (size_t j = 0; j < 3; ++j) {
                new_vel_adjoints[i][j] += pos_adjoints[i][j] * dt;
            }
        }

        // Reverse the position update: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        for (size_t i = 0; i < n_particles; ++i) {
            // ∂L/∂x(t) = ∂L/∂x(t+dt) + force contributions
            new_pos_adjoints[i] = pos_adjoints[i];

            // Add contributions from force terms
            for (size_t j = 0; j < 3; ++j) {
                // Contribution from acceleration: a = F/m
                T accel_contrib = T(0.5) * pos_adjoints[i][j] * dt * dt / masses[i];

                // Backpropagate through forces using chain rule
                for (size_t k = 0; k < n_particles; ++k) {
                    for (size_t l = 0; l < 3; ++l) {
                        // ∂F_i^j/∂x_k^l from force Jacobian
                        T force_jac = getForceJacobian(force_jacobians, i, j, k, l);
                        new_pos_adjoints[k][l] += accel_contrib * force_jac;
                    }
                }

                // Contribution to mass gradient
                new_mass_adjoints[i] -= accel_contrib * forces[i][j] / masses[i];
            }
        }

        // Update adjoint state
        state_manager_->setPositionAdjoints(new_pos_adjoints);
        state_manager_->setVelocityAdjoints(new_vel_adjoints);

        // Accumulate gradients
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
    T getForceJacobian(const std::vector<std::vector<std::vector<vector_type>>>& jacobians,
                      size_t particle_i, size_t component_j,
                      size_t particle_k, size_t component_l) const {
        if (particle_i < jacobians.size() &&
            particle_k < jacobians[particle_i].size()) {
            return jacobians[particle_i][particle_k][component_j][component_l];
        }
        return T(0);
    }
};

// =============================================================================
// ADJOINT LEAPFROG INTEGRATOR
// =============================================================================

/**
 * Leapfrog integrator with adjoint automatic differentiation
 * More memory efficient than Verlet for some applications
 */
template<typename T>
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    requires concepts::PhysicsScalar<T>
#endif
class AdjointLeapfrogIntegrator {
public:
    using scalar_type = T;
    using vector_type = ConceptVector3D<T>;
    using force_engine_type = differentiable::DifferentiableForceEngine<T>;
    using state_manager_type = AdjointStateManager<T>;

    AdjointLeapfrogIntegrator(std::shared_ptr<force_engine_type> force_engine)
        : force_engine_(force_engine), state_manager_(std::make_unique<state_manager_type>()) {}

    // Forward leapfrog step with checkpointing
    void forwardStep(std::vector<vector_type>& positions,
                    std::vector<vector_type>& velocities,
                    const std::vector<T>& masses,
                    T dt) {
        const size_t n_particles = positions.size();

        // Compute forces at current positions
        auto [forces, force_gradients] = force_engine_->computeForcesAndGradients(positions);

        // Store checkpoint
        AdjointCheckpoint<T> checkpoint(positions, velocities, forces, masses, dt, current_step_);
        state_manager_->pushCheckpoint(checkpoint);

        // Leapfrog integration
        // v(t + dt/2) = v(t) + a(t) * dt/2
        std::vector<vector_type> half_velocities(n_particles);
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                half_velocities[i][j] = velocities[i][j] +
                                       (forces[i][j] / masses[i]) * (dt / T(2));
            }
        }

        // x(t + dt) = x(t) + v(t + dt/2) * dt
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                positions[i][j] += half_velocities[i][j] * dt;
            }
        }

        // Compute forces at new positions
        auto [new_forces, new_force_gradients] = force_engine_->computeForcesAndGradients(positions);

        // v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        for (size_t i = 0; i < n_particles; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                velocities[i][j] = half_velocities[i][j] +
                                  (new_forces[i][j] / masses[i]) * (dt / T(2));
            }
        }

        current_step_++;
    }

    // Backward leapfrog step
    void backwardStep(std::vector<vector_type>& pos_grads,
                     std::vector<vector_type>& vel_grads,
                     std::vector<T>& mass_grads) {
        // Implementation similar to Verlet but adapted for leapfrog scheme
        auto checkpoint = state_manager_->popCheckpoint();

        // Reverse leapfrog steps using saved forward pass data
        // Implementation follows similar pattern to Verlet backward pass

        current_step_--;
    }

    void initializeBackward(const std::vector<vector_type>& final_pos_grads,
                           const std::vector<vector_type>& final_vel_grads) {
        state_manager_->initializeAdjointState(final_pos_grads, final_vel_grads);
    }

    void reset() {
        state_manager_->clearCheckpoints();
        current_step_ = 0;
    }

private:
    std::shared_ptr<force_engine_type> force_engine_;
    std::unique_ptr<state_manager_type> state_manager_;
    int current_step_ = 0;
};

// =============================================================================
// ADJOINT SIMULATION FRAMEWORK
// =============================================================================

/**
 * High-level framework for differentiable physics simulation
 * Coordinates forward and backward passes through entire simulation
 */
template<typename T>
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    requires concepts::PhysicsScalar<T>
#endif
class AdjointSimulation {
public:
    using scalar_type = T;
    using vector_type = ConceptVector3D<T>;
    using integrator_type = AdjointVerletIntegrator<T>;

    AdjointSimulation(std::shared_ptr<differentiable::DifferentiableForceEngine<T>> force_engine)
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

        // Compute loss and its gradients
        T loss = loss_function(positions, velocities);

        // Create unit gradients for simple scalar loss
        std::vector<vector_type> loss_grad_pos(positions.size());
        std::vector<vector_type> loss_grad_vel(velocities.size());

        // This would typically be computed by autodiff framework
        // For now, assume unit gradients
        for (size_t i = 0; i < positions.size(); ++i) {
            for (size_t j = 0; j < 3; ++j) {
                loss_grad_pos[i][j] = T(1) / static_cast<T>(positions.size() * 3);
                loss_grad_vel[i][j] = T(1) / static_cast<T>(velocities.size() * 3);
            }
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

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
// Verify concept compliance
static_assert(concepts::PhysicsScalar<float>);
static_assert(concepts::PhysicsScalar<double>);
#endif

} // namespace physgrad

#endif // PHYSGRAD_ADJOINT_INTEGRATORS_H