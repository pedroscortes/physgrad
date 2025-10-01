/**
 * PhysGrad - Adjoint Integration Methods - Simple Implementation
 *
 * Simplified implementation for testing and validation.
 * Provides concrete implementations without heavy dependencies.
 */

#include "adjoint_integrators.h"
#include <iostream>
#include <cassert>

namespace physgrad {

// =============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// =============================================================================

// AdjointCheckpoint
template struct AdjointCheckpoint<float>;
template struct AdjointCheckpoint<double>;

// AdjointStateManager
template class AdjointStateManager<float>;
template class AdjointStateManager<double>;

// AdjointVerletIntegrator
template class AdjointVerletIntegrator<float>;
template class AdjointVerletIntegrator<double>;

// AdjointLeapfrogIntegrator
template class AdjointLeapfrogIntegrator<float>;
template class AdjointLeapfrogIntegrator<double>;

// AdjointSimulation
template class AdjointSimulation<float>;
template class AdjointSimulation<double>;

// =============================================================================
// SIMPLIFIED IMPLEMENTATIONS FOR TESTING
// =============================================================================

/**
 * Simple differentiable harmonic oscillator for testing
 */
template<typename T>
class SimpleHarmonicForce {
public:
    SimpleHarmonicForce(T spring_constant, T rest_length)
        : k_(spring_constant), rest_length_(rest_length) {}

    // Compute force and gradient for two particles connected by spring
    std::pair<std::vector<ConceptVector3D<T>>, std::vector<std::vector<std::vector<ConceptVector3D<T>>>>>
    computeForcesAndGradients(const std::vector<ConceptVector3D<T>>& positions) {
        std::vector<ConceptVector3D<T>> forces(positions.size());
        std::vector<std::vector<std::vector<ConceptVector3D<T>>>> gradients(
            positions.size(),
            std::vector<std::vector<ConceptVector3D<T>>>(
                positions.size(),
                std::vector<ConceptVector3D<T>>(3)
            )
        );

        if (positions.size() >= 2) {
            // Spring force between particles 0 and 1
            T dx = positions[1][0] - positions[0][0];
            T dy = positions[1][1] - positions[0][1];
            T dz = positions[1][2] - positions[0][2];

            T distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            T force_magnitude = k_ * (distance - rest_length_);

            if (distance > 1e-10) {
                T fx = force_magnitude * dx / distance;
                T fy = force_magnitude * dy / distance;
                T fz = force_magnitude * dz / distance;

                // Forces on particles
                forces[0] = ConceptVector3D<T>{fx, fy, fz};
                forces[1] = ConceptVector3D<T>{-fx, -fy, -fz};

                // Force gradients (simplified - diagonal terms only)
                gradients[0][0][0] = ConceptVector3D<T>{-k_, 0, 0};
                gradients[0][1][0] = ConceptVector3D<T>{k_, 0, 0};
                gradients[1][0][0] = ConceptVector3D<T>{k_, 0, 0};
                gradients[1][1][0] = ConceptVector3D<T>{-k_, 0, 0};
            }
        }

        return {forces, gradients};
    }

private:
    T k_;
    T rest_length_;
};

// =============================================================================
// EXAMPLE USAGE FUNCTIONS
// =============================================================================

/**
 * Example: Simple pendulum with adjoint gradients
 */
template<typename T>
void exampleSimplePendulum() {
    std::cout << "Running simple pendulum adjoint example...\n";

    // Create simple force engine
    auto force_engine = std::make_shared<differentiable::DifferentiableForceEngine<T>>();

    // Create adjoint integrator
    AdjointVerletIntegrator<T> integrator(force_engine);

    // Initial conditions
    std::vector<ConceptVector3D<T>> positions = {
        {T(0), T(0), T(0)},      // Fixed point
        {T(1), T(0), T(0)}       // Pendulum bob
    };

    std::vector<ConceptVector3D<T>> velocities = {
        {T(0), T(0), T(0)},      // Fixed
        {T(0), T(1), T(0)}       // Initial velocity
    };

    std::vector<T> masses = {T(1e6), T(1)}; // Heavy anchor, light bob

    T dt = T(0.01);
    int num_steps = 100;

    // Forward simulation
    std::cout << "Forward simulation...\n";
    for (int step = 0; step < num_steps; ++step) {
        integrator.forwardStep(positions, velocities, masses, dt);

        if (step % 20 == 0) {
            std::cout << "Step " << step << ": bob position = ("
                      << positions[1][0] << ", " << positions[1][1] << ", " << positions[1][2] << ")\n";
        }
    }

    // Example backward pass
    std::cout << "Backward pass...\n";
    std::vector<ConceptVector3D<T>> loss_grad_pos(2);
    std::vector<ConceptVector3D<T>> loss_grad_vel(2);

    // Simple loss: distance from origin
    loss_grad_pos[1] = positions[1]; // ∇(|x|²) = 2x ≈ x for unit gradients

    integrator.initializeBackward(loss_grad_pos, loss_grad_vel);

    std::vector<ConceptVector3D<T>> pos_grads, vel_grads;
    std::vector<T> mass_grads;

    // Run a few backward steps
    for (int step = 0; step < std::min(5, num_steps); ++step) {
        integrator.backwardStep(pos_grads, vel_grads, mass_grads);

        if (step == 0) {
            std::cout << "Initial position gradient for bob: ("
                      << pos_grads[1][0] << ", " << pos_grads[1][1] << ", " << pos_grads[1][2] << ")\n";
        }
    }

    std::cout << "Adjoint pendulum example completed.\n\n";
}

/**
 * Example: Gradient accuracy validation
 */
template<typename T>
void exampleGradientValidation() {
    std::cout << "Running gradient validation example...\n";

    // Create simulation framework
    auto force_engine = std::make_shared<differentiable::DifferentiableForceEngine<T>>();
    AdjointSimulation<T> simulation(force_engine);

    // Simple two-particle system
    std::vector<ConceptVector3D<T>> initial_positions = {
        {T(0), T(0), T(0)},
        {T(1), T(0), T(0)}
    };

    std::vector<ConceptVector3D<T>> initial_velocities = {
        {T(0), T(0), T(0)},
        {T(0), T(0), T(0)}
    };

    std::vector<T> masses = {T(1), T(1)};

    // Simple quadratic loss function
    auto loss_function = [](const std::vector<ConceptVector3D<T>>& pos,
                           const std::vector<ConceptVector3D<T>>& vel) -> T {
        return pos[1][0] * pos[1][0]; // |x₁|²
    };

    try {
        // Compute gradients
        auto [pos_grads, vel_grads] = simulation.computeGradients(
            initial_positions, initial_velocities, masses,
            T(0.01), 5, loss_function
        );

        std::cout << "Computed gradients successfully:\n";
        std::cout << "Position gradient for particle 1: ("
                  << pos_grads[1][0] << ", " << pos_grads[1][1] << ", " << pos_grads[1][2] << ")\n";

        // Finite difference validation
        T h = T(1e-5);
        auto positions_plus = initial_positions;
        auto positions_minus = initial_positions;
        auto velocities_copy = initial_velocities;

        positions_plus[1][0] += h;
        positions_minus[1][0] -= h;

        AdjointSimulation<T> sim_plus(force_engine);
        AdjointSimulation<T> sim_minus(force_engine);

        auto vel_copy_plus = initial_velocities;
        sim_plus.runForward(positions_plus, vel_copy_plus, masses, T(0.01), 5);
        T loss_plus = loss_function(positions_plus, vel_copy_plus);

        auto vel_copy_minus = initial_velocities;
        sim_minus.runForward(positions_minus, vel_copy_minus, masses, T(0.01), 5);
        T loss_minus = loss_function(positions_minus, vel_copy_minus);

        T fd_gradient = (loss_plus - loss_minus) / (T(2) * h);

        std::cout << "Finite difference gradient: " << fd_gradient << "\n";
        std::cout << "Adjoint gradient: " << pos_grads[1][0] << "\n";

        T relative_error = std::abs(pos_grads[1][0] - fd_gradient) / std::max(std::abs(fd_gradient), T(1e-10));
        std::cout << "Relative error: " << relative_error << "\n";

        if (relative_error < T(0.1)) {
            std::cout << "✓ Gradient validation PASSED\n";
        } else {
            std::cout << "✗ Gradient validation FAILED\n";
        }

    } catch (const std::exception& e) {
        std::cout << "Error in gradient computation: " << e.what() << "\n";
    }

    std::cout << "Gradient validation example completed.\n\n";
}

} // namespace physgrad

// =============================================================================
// MAIN FUNCTION FOR STANDALONE TESTING
// =============================================================================

int main() {
    std::cout << "PhysGrad Adjoint Integrators - Examples and Tests\n";
    std::cout << "=================================================\n\n";

    try {
        // Run examples with different precision
        std::cout << "--- Float precision examples ---\n";
        physgrad::exampleSimplePendulum<float>();
        physgrad::exampleGradientValidation<float>();

        std::cout << "--- Double precision examples ---\n";
        physgrad::exampleSimplePendulum<double>();
        physgrad::exampleGradientValidation<double>();

        std::cout << "All examples completed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}