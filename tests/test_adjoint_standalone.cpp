/**
 * PhysGrad Standalone Adjoint Integrators Test
 *
 * Tests for the self-contained adjoint automatic differentiation implementation.
 * Validates gradient computation accuracy using finite differences.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>

#include "adjoint_integrators_standalone.h"

using namespace physgrad;
using namespace physgrad::adjoint;

/**
 * Finite difference gradient computation for validation
 */
template<typename T>
T computeFiniteDifferenceGradient(
    std::function<T(const std::vector<ConceptVector3D<T>>&)> func,
    const std::vector<ConceptVector3D<T>>& state,
    size_t particle_idx,
    size_t component_idx,
    T h = T{1e-5}) {

    auto state_plus = state;
    auto state_minus = state;

    state_plus[particle_idx][component_idx] += h;
    state_minus[particle_idx][component_idx] -= h;

    T f_plus = func(state_plus);
    T f_minus = func(state_minus);

    return (f_plus - f_minus) / (T{2} * h);
}

/**
 * Test basic forward integration
 */
template<typename T>
bool testForwardIntegration() {
    std::cout << "Testing forward integration...\n";

    // Create force engine with harmonic spring
    auto force_engine = std::make_shared<SimpleForceEngine<T>>();
    force_engine->addSpring(0, 1, T{10}, T{1}); // k=10, rest length=1

    // Create integrator
    AdjointVerletIntegrator<T> integrator(force_engine);

    // Initial conditions: two particles with stretched spring
    std::vector<ConceptVector3D<T>> positions = {
        {T{0}, T{0}, T{0}},      // Particle 0 at origin
        {T{1.5}, T{0}, T{0}}     // Particle 1 stretched spring
    };

    std::vector<ConceptVector3D<T>> velocities = {
        {T{0}, T{0}, T{0}},      // Initially at rest
        {T{0}, T{0}, T{0}}
    };

    std::vector<T> masses = {T{1}, T{1}};

    auto initial_positions = positions;

    // Take forward step
    integrator.forwardStep(positions, velocities, masses, T{0.01});

    // Check that particles moved (spring should pull them together)
    T displacement = std::abs(positions[1][0] - initial_positions[1][0]);
    bool moved = displacement > T{1e-6};

    std::cout << "Initial position[1]: " << initial_positions[1][0] << "\n";
    std::cout << "Final position[1]: " << positions[1][0] << "\n";
    std::cout << "Displacement: " << displacement << "\n";

    return moved;
}

/**
 * Test adjoint gradient computation
 */
template<typename T>
bool testAdjointGradients() {
    std::cout << "Testing adjoint gradients...\n";

    // Create force engine
    auto force_engine = std::make_shared<SimpleForceEngine<T>>();
    force_engine->addSpring(0, 1, T{10}, T{1});

    // Create simulation
    AdjointSimulation<T> simulation(force_engine);

    // Initial conditions
    std::vector<ConceptVector3D<T>> initial_positions = {
        {T{0}, T{0}, T{0}},
        {T{1.2}, T{0}, T{0}}  // Slightly stretched spring
    };

    std::vector<ConceptVector3D<T>> initial_velocities = {
        {T{0}, T{0}, T{0}},
        {T{0}, T{0}, T{0}}
    };

    std::vector<T> masses = {T{1}, T{1}};

    // Simple quadratic loss function: L = x₁²
    auto loss_function = [](const std::vector<ConceptVector3D<T>>& pos,
                           const std::vector<ConceptVector3D<T>>& vel) -> T {
        return pos[1][0] * pos[1][0];
    };

    try {
        // Compute gradients using adjoint method
        auto [pos_grads, vel_grads] = simulation.computeGradients(
            initial_positions, initial_velocities, masses,
            T{0.01}, 3, loss_function
        );

        std::cout << "Adjoint gradient for position[1][0]: " << pos_grads[1][0] << "\n";

        // Compute finite difference gradient for comparison
        auto fd_func = [&](const std::vector<ConceptVector3D<T>>& init_pos) -> T {
            auto pos = init_pos;
            auto vel = initial_velocities;
            AdjointSimulation<T> temp_sim(force_engine);
            temp_sim.runForward(pos, vel, masses, T{0.01}, 3);
            return loss_function(pos, vel);
        };

        std::function<T(const std::vector<ConceptVector3D<T>>&)> fd_func_std = fd_func;
        T fd_gradient = computeFiniteDifferenceGradient(fd_func_std, initial_positions, 1, 0);
        std::cout << "Finite difference gradient: " << fd_gradient << "\n";

        // Compare gradients
        T relative_error = std::abs(pos_grads[1][0] - fd_gradient) /
                          std::max(std::abs(fd_gradient), T{1e-10});
        std::cout << "Relative error: " << relative_error << "\n";

        return relative_error < T{0.1}; // Allow 10% error for simplified implementation

    } catch (const std::exception& e) {
        std::cout << "Error in gradient computation: " << e.what() << "\n";
        return false;
    }
}

/**
 * Test energy conservation
 */
template<typename T>
bool testEnergyConservation() {
    std::cout << "Testing energy conservation...\n";

    auto force_engine = std::make_shared<SimpleForceEngine<T>>();
    force_engine->addSpring(0, 1, T{10}, T{1});

    AdjointVerletIntegrator<T> integrator(force_engine);

    // Initial conditions with kinetic energy
    std::vector<ConceptVector3D<T>> positions = {
        {T{0}, T{0}, T{0}},
        {T{1.1}, T{0}, T{0}}  // Slightly compressed spring
    };

    std::vector<ConceptVector3D<T>> velocities = {
        {T{0.1}, T{0}, T{0}},   // Some initial velocity
        {T{-0.1}, T{0}, T{0}}
    };

    std::vector<T> masses = {T{1}, T{1}};

    // Compute initial energy
    T initial_kinetic = T{0};
    for (size_t i = 0; i < velocities.size(); ++i) {
        T v_sq = velocities[i][0]*velocities[i][0] +
                 velocities[i][1]*velocities[i][1] +
                 velocities[i][2]*velocities[i][2];
        initial_kinetic += T{0.5} * masses[i] * v_sq;
    }

    T dx = positions[1][0] - positions[0][0];
    T initial_potential = T{0.5} * T{10} * (dx - T{1}) * (dx - T{1}); // k*(x-x0)²/2
    T initial_energy = initial_kinetic + initial_potential;

    std::cout << "Initial energy: " << initial_energy << "\n";

    // Run simulation
    for (int step = 0; step < 50; ++step) {
        integrator.forwardStep(positions, velocities, masses, T{0.01});
    }

    // Compute final energy
    T final_kinetic = T{0};
    for (size_t i = 0; i < velocities.size(); ++i) {
        T v_sq = velocities[i][0]*velocities[i][0] +
                 velocities[i][1]*velocities[i][1] +
                 velocities[i][2]*velocities[i][2];
        final_kinetic += T{0.5} * masses[i] * v_sq;
    }

    dx = positions[1][0] - positions[0][0];
    T final_potential = T{0.5} * T{10} * (dx - T{1}) * (dx - T{1});
    T final_energy = final_kinetic + final_potential;

    std::cout << "Final energy: " << final_energy << "\n";

    T energy_error = std::abs(final_energy - initial_energy) / initial_energy;
    std::cout << "Energy error: " << energy_error << "\n";

    return energy_error < T{0.05}; // Allow 5% energy drift
}

/**
 * Test checkpointing mechanism
 */
template<typename T>
bool testCheckpointing() {
    std::cout << "Testing checkpointing mechanism...\n";

    auto force_engine = std::make_shared<SimpleForceEngine<T>>();
    force_engine->addSpring(0, 1, T{5}, T{1});

    AdjointVerletIntegrator<T> integrator(force_engine);

    std::vector<ConceptVector3D<T>> positions = {
        {T{0}, T{0}, T{0}},
        {T{1.5}, T{0}, T{0}}
    };

    std::vector<ConceptVector3D<T>> velocities = {
        {T{0}, T{0}, T{0}},
        {T{0}, T{0}, T{0}}
    };

    std::vector<T> masses = {T{1}, T{1}};

    // Initially no checkpoints
    if (integrator.getNumCheckpoints() != 0) {
        std::cout << "Initial checkpoint count should be 0\n";
        return false;
    }

    // Take several forward steps
    int num_steps = 5;
    for (int step = 0; step < num_steps; ++step) {
        integrator.forwardStep(positions, velocities, masses, T{0.01});
    }

    // Should have stored checkpoints
    if (integrator.getNumCheckpoints() != num_steps) {
        std::cout << "Expected " << num_steps << " checkpoints, got "
                  << integrator.getNumCheckpoints() << "\n";
        return false;
    }

    // Test backward pass
    std::vector<ConceptVector3D<T>> loss_grad_pos(2);
    std::vector<ConceptVector3D<T>> loss_grad_vel(2);
    loss_grad_pos[1][0] = T{1}; // Simple gradient

    integrator.initializeBackward(loss_grad_pos, loss_grad_vel);

    std::vector<ConceptVector3D<T>> pos_grads, vel_grads;
    std::vector<T> mass_grads;

    // Take one backward step
    integrator.backwardStep(pos_grads, vel_grads, mass_grads);

    // Should have one less checkpoint
    if (integrator.getNumCheckpoints() != num_steps - 1) {
        std::cout << "Expected " << (num_steps - 1) << " checkpoints after backward step, got "
                  << integrator.getNumCheckpoints() << "\n";
        return false;
    }

    // Reset should clear all checkpoints
    integrator.reset();
    if (integrator.getNumCheckpoints() != 0) {
        std::cout << "Reset should clear all checkpoints\n";
        return false;
    }

    return true;
}

/**
 * Main test function
 */
int main() {
    std::cout << "PhysGrad Adjoint Integrators - Standalone Tests\n";
    std::cout << "===============================================\n\n";

    std::cout << std::fixed << std::setprecision(6);

    bool all_passed = true;

    // Test with float precision
    std::cout << "--- Float precision tests ---\n";
    all_passed &= testForwardIntegration<float>();
    std::cout << "\n";

    all_passed &= testEnergyConservation<float>();
    std::cout << "\n";

    all_passed &= testCheckpointing<float>();
    std::cout << "\n";

    all_passed &= testAdjointGradients<float>();
    std::cout << "\n";

    // Test with double precision
    std::cout << "--- Double precision tests ---\n";
    all_passed &= testForwardIntegration<double>();
    std::cout << "\n";

    all_passed &= testAdjointGradients<double>();
    std::cout << "\n";

    if (all_passed) {
        std::cout << "✓ All tests PASSED!\n";
        return 0;
    } else {
        std::cout << "✗ Some tests FAILED!\n";
        return 1;
    }
}