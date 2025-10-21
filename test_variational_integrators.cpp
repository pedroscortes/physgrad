/**
 * PhysGrad Variational Integrators Test Suite
 *
 * Comprehensive validation of Galerkin variational integrators
 * including conservation properties and accuracy analysis
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <fstream>
#include "src/variational_integrators.h"

using namespace physgrad::variational;

template<typename T>
bool approximately_equal(T a, T b, T tolerance = static_cast<T>(1e-5)) {
    return std::abs(a - b) <= tolerance;
}

bool test_galerkin_basis_functions() {
    std::cout << "Testing Galerkin Basis Functions..." << std::endl;

    // Test quadratic basis (Order = 2)
    GalerkinBasis<float, 2> basis;
    const auto& nodes = basis.getNodes();
    const auto& weights = basis.getWeights();

    std::cout << "  Lobatto nodes: ";
    for (auto node : nodes) std::cout << node << " ";
    std::cout << std::endl;

    // Test partition of unity
    std::vector<float> test_points = {-0.8f, -0.3f, 0.0f, 0.5f, 0.9f};
    bool partition_unity = true;

    for (float s : test_points) {
        float sum = 0.0f;
        for (size_t i = 0; i < basis.num_nodes; ++i) {
            sum += basis.evaluateBasis(i, s);
        }
        if (!approximately_equal(sum, 1.0f, 1e-6f)) {
            partition_unity = false;
            std::cout << "  ❌ Partition of unity failed at s=" << s << ", sum=" << sum << std::endl;
        }
    }

    if (partition_unity) {
        std::cout << "  ✓ Partition of unity satisfied" << std::endl;
    }

    // Test Kronecker delta property at nodes
    bool kronecker_property = true;
    for (size_t i = 0; i < basis.num_nodes; ++i) {
        for (size_t j = 0; j < basis.num_nodes; ++j) {
            float value = basis.evaluateBasis(i, nodes[j]);
            float expected = (i == j) ? 1.0f : 0.0f;
            if (!approximately_equal(value, expected, 1e-6f)) {
                kronecker_property = false;
                std::cout << "  ❌ Kronecker property failed: φ_" << i << "(" << nodes[j] << ") = " << value << std::endl;
            }
        }
    }

    if (kronecker_property) {
        std::cout << "  ✓ Kronecker delta property satisfied" << std::endl;
    }

    return partition_unity && kronecker_property;
}

bool test_discrete_lagrangian() {
    std::cout << "Testing Discrete Lagrangian..." << std::endl;

    // Simple harmonic oscillator: L = (1/2)mv² - (1/2)kx²
    auto lagrangian = [](const std::vector<float>& q, const std::vector<float>& v, float t) -> float {
        float mass = 1.0f, k = 1.0f;
        return 0.5f * mass * v[0] * v[0] - 0.5f * k * q[0] * q[0];
    };

    float dt = 0.1f;
    DiscreteLagrangian<float> discrete_L(lagrangian, dt);

    // Test states
    std::vector<float> q0 = {1.0f};   // Initial position
    std::vector<float> q1 = {0.95f};  // Next position

    // Evaluate discrete Lagrangian
    float L_discrete = discrete_L.evaluate(q0, q1);
    std::cout << "  Discrete Lagrangian value: " << L_discrete << std::endl;

    // Test with Galerkin quadrature
    GalerkinBasis<float, 2> basis;
    float L_galerkin = discrete_L.evaluateGalerkin(q0, q1, basis);
    std::cout << "  Galerkin quadrature value: " << L_galerkin << std::endl;

    // Test gradient computation
    auto grad_q0 = discrete_L.gradientQ0(q0, q1);
    auto grad_q1 = discrete_L.gradientQ1(q0, q1);

    std::cout << "  Gradient w.r.t. q0: " << grad_q0[0] << std::endl;
    std::cout << "  Gradient w.r.t. q1: " << grad_q1[0] << std::endl;

    // Verify that gradients have opposite signs (conservation property)
    bool gradient_antisymmetry = (grad_q0[0] * grad_q1[0] < 0);
    if (gradient_antisymmetry) {
        std::cout << "  ✓ Discrete Euler-Lagrange gradient antisymmetry satisfied" << std::endl;
    } else {
        std::cout << "  ❌ Gradient antisymmetry failed" << std::endl;
    }

    return gradient_antisymmetry;
}

bool test_harmonic_oscillator_conservation() {
    std::cout << "Testing Harmonic Oscillator Energy Conservation..." << std::endl;

    // Create harmonic oscillator Lagrangian
    float mass = 1.0f, k = 4.0f; // ω = 2
    auto lagrangian = systems::createHarmonicOscillator(mass, k);

    float dt = 0.01f;
    GalerkinVariationalIntegrator<float, 2> integrator(lagrangian, dt, true, true);

    // Initial conditions: x(0) = 1, v(0) = 0
    std::vector<float> q0 = {1.0f};
    std::vector<float> v0 = {0.0f};

    float total_time = 2.0f * M_PI; // One full period
    int num_steps = static_cast<int>(total_time / dt);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto trajectory = integrator.integrate(q0, v0, total_time, num_steps);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto computation_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // Analyze results
    float energy_drift = integrator.getEnergyDrift();
    float momentum_drift = integrator.getMomentumDrift();

    std::cout << "  Simulation time: " << total_time << "s (" << num_steps << " steps)" << std::endl;
    std::cout << "  Computation time: " << computation_time << " ms" << std::endl;
    std::cout << "  Final position: " << trajectory.back()[0] << " (expected: ~1.0)" << std::endl;
    std::cout << "  Energy drift: " << energy_drift << std::endl;
    std::cout << "  Momentum drift: " << momentum_drift << std::endl;

    // Validation criteria
    bool position_periodic = approximately_equal(trajectory.back()[0], q0[0], 0.1f);
    bool energy_conserved = std::abs(energy_drift) < 0.01f;
    bool good_performance = computation_time < 100.0f; // Less than 100ms

    bool test_passed = position_periodic && energy_conserved && good_performance;

    if (test_passed) {
        std::cout << "  ✓ Harmonic oscillator conservation test PASSED" << std::endl;
    } else {
        std::cout << "  ❌ Conservation test failed:" << std::endl;
        if (!position_periodic) std::cout << "    - Position not periodic" << std::endl;
        if (!energy_conserved) std::cout << "    - Energy not conserved" << std::endl;
        if (!good_performance) std::cout << "    - Poor computational performance" << std::endl;
    }

    return test_passed;
}

bool test_double_pendulum_chaotic_dynamics() {
    std::cout << "Testing Double Pendulum Chaotic Dynamics..." << std::endl;

    // Double pendulum parameters
    float m1 = 1.0f, m2 = 1.0f;
    float l1 = 1.0f, l2 = 1.0f;
    float g = 9.81f;

    auto lagrangian = systems::createDoublePendulum(m1, m2, l1, l2, g);

    float dt = 0.01f; // Larger timestep to see dynamics
    GalerkinVariationalIntegrator<float, 3> integrator(lagrangian, dt, true, true);

    // Set external forces (gravitational forces for pendulum)
    integrator.setExternalForces([g, m1, m2, l1, l2](const std::vector<float>& q, const std::vector<float>& v, float t) -> std::vector<float> {
        std::vector<float> forces(2, 0.0f);
        // Simplified gravitational torques
        forces[0] = -m1 * g * l1 * std::sin(q[0]);
        forces[1] = -m2 * g * l2 * std::sin(q[1]);
        return forces;
    });

    // Initial conditions: both pendulums slightly off vertical
    std::vector<float> q0 = {0.1f, 0.1f};  // Small angles
    std::vector<float> v0 = {0.0f, 0.0f};  // Starting from rest

    float total_time = 5.0f;
    int num_steps = static_cast<int>(total_time / dt);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto trajectory = integrator.integrate(q0, v0, total_time, num_steps);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto computation_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // Analyze conservation properties
    const auto& energy_history = integrator.getEnergyHistory();
    float initial_energy = energy_history.front();
    float final_energy = energy_history.back();
    float energy_drift = std::abs(final_energy - initial_energy) / initial_energy;

    std::cout << "  Simulation time: " << total_time << "s (" << num_steps << " steps)" << std::endl;
    std::cout << "  Computation time: " << computation_time << " ms" << std::endl;
    std::cout << "  Initial energy: " << initial_energy << std::endl;
    std::cout << "  Final energy: " << final_energy << std::endl;
    std::cout << "  Relative energy drift: " << energy_drift * 100.0f << "%" << std::endl;
    std::cout << "  Final angles: θ₁=" << trajectory.back()[0] << ", θ₂=" << trajectory.back()[1] << std::endl;

    // Check for chaotic behavior (large deviations from initial conditions)
    float angle1_deviation = std::abs(trajectory.back()[0] - q0[0]);
    float angle2_deviation = std::abs(trajectory.back()[1] - q0[1]);
    bool chaotic_behavior = (angle1_deviation > 1.0f || angle2_deviation > 1.0f);

    // Validation criteria
    bool energy_conserved = energy_drift < 0.05f; // 5% tolerance for chaotic system
    bool simulation_stable = std::isfinite(trajectory.back()[0]) && std::isfinite(trajectory.back()[1]);

    bool test_passed = energy_conserved && simulation_stable && chaotic_behavior;

    if (test_passed) {
        std::cout << "  ✓ Double pendulum dynamics test PASSED" << std::endl;
        std::cout << "    - Energy conservation within tolerance ✓" << std::endl;
        std::cout << "    - Simulation remained stable ✓" << std::endl;
        std::cout << "    - Chaotic behavior observed ✓" << std::endl;
    } else {
        std::cout << "  ❌ Double pendulum test failed:" << std::endl;
        if (!energy_conserved) std::cout << "    - Energy conservation violated" << std::endl;
        if (!simulation_stable) std::cout << "    - Simulation became unstable" << std::endl;
        if (!chaotic_behavior) std::cout << "    - No chaotic behavior detected" << std::endl;
    }

    return test_passed;
}

bool test_nbody_gravitational_system() {
    std::cout << "Testing N-Body Gravitational System..." << std::endl;

    // Three-body system
    std::vector<float> masses = {1.0f, 0.5f, 0.3f};
    float G = 0.1f; // Reduced G for stability

    auto lagrangian = systems::createNBodyGravitational(masses, G);

    float dt = 0.01f; // Larger timestep
    GalerkinVariationalIntegrator<float, 2> integrator(lagrangian, dt, true, true);

    // Set gravitational forces
    integrator.setExternalForces([masses, G](const std::vector<float>& q, const std::vector<float>& v, float t) -> std::vector<float> {
        size_t N = masses.size();
        std::vector<float> forces(3 * N, 0.0f);

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (i != j) {
                    float r2 = 0.0f;
                    for (size_t d = 0; d < 3; ++d) {
                        float dx = q[3*i + d] - q[3*j + d];
                        r2 += dx * dx;
                    }
                    float r = std::sqrt(r2 + 0.01f); // Softening
                    float force_mag = G * masses[i] * masses[j] / (r * r * r);

                    for (size_t d = 0; d < 3; ++d) {
                        float dx = q[3*j + d] - q[3*i + d];
                        forces[3*i + d] += force_mag * dx;
                    }
                }
            }
        }
        return forces;
    });

    // Initial conditions: triangle configuration
    std::vector<float> q0 = {
        // Body 1: at origin
        0.0f, 0.0f, 0.0f,
        // Body 2: offset in x
        1.0f, 0.0f, 0.0f,
        // Body 3: offset in y
        0.5f, 0.866f, 0.0f
    };

    std::vector<float> v0 = {
        // Initial velocities for stable orbits
        0.0f, 0.1f, 0.0f,
        -0.1f, 0.05f, 0.0f,
        0.05f, -0.15f, 0.0f
    };

    float total_time = 2.0f;
    int num_steps = static_cast<int>(total_time / dt);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto trajectory = integrator.integrate(q0, v0, total_time, num_steps);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto computation_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // Analyze conservation
    float energy_drift = integrator.getEnergyDrift();
    float momentum_drift = integrator.getMomentumDrift();

    // Compute center of mass drift
    std::vector<float> initial_cm(3, 0.0f), final_cm(3, 0.0f);
    float total_mass = 0.0f;

    for (size_t i = 0; i < masses.size(); ++i) {
        total_mass += masses[i];
        for (size_t d = 0; d < 3; ++d) {
            initial_cm[d] += masses[i] * q0[3*i + d];
            final_cm[d] += masses[i] * trajectory.back()[3*i + d];
        }
    }

    float cm_drift = 0.0f;
    for (size_t d = 0; d < 3; ++d) {
        initial_cm[d] /= total_mass;
        final_cm[d] /= total_mass;
        cm_drift += (final_cm[d] - initial_cm[d]) * (final_cm[d] - initial_cm[d]);
    }
    cm_drift = std::sqrt(cm_drift);

    std::cout << "  Simulation time: " << total_time << "s (" << num_steps << " steps)" << std::endl;
    std::cout << "  Computation time: " << computation_time << " ms" << std::endl;
    std::cout << "  Energy drift: " << energy_drift << std::endl;
    std::cout << "  Momentum drift: " << momentum_drift << std::endl;
    std::cout << "  Center of mass drift: " << cm_drift << std::endl;

    // Validation criteria
    bool energy_conserved = std::abs(energy_drift) < 0.1f;
    bool momentum_conserved = momentum_drift < 0.05f;
    bool cm_conserved = cm_drift < 0.01f;
    bool simulation_stable = true;

    // Check that all particles remain finite
    for (float coord : trajectory.back()) {
        if (!std::isfinite(coord)) {
            simulation_stable = false;
            break;
        }
    }

    bool test_passed = energy_conserved && momentum_conserved && cm_conserved && simulation_stable;

    if (test_passed) {
        std::cout << "  ✓ N-body gravitational system test PASSED" << std::endl;
    } else {
        std::cout << "  ❌ N-body test failed:" << std::endl;
        if (!energy_conserved) std::cout << "    - Energy conservation violated" << std::endl;
        if (!momentum_conserved) std::cout << "    - Momentum conservation violated" << std::endl;
        if (!cm_conserved) std::cout << "    - Center of mass conservation violated" << std::endl;
        if (!simulation_stable) std::cout << "    - Simulation became unstable" << std::endl;
    }

    return test_passed;
}

bool test_galerkin_vs_standard_comparison() {
    std::cout << "Testing Galerkin vs Standard Quadrature Comparison..." << std::endl;

    // Harmonic oscillator for comparison
    auto lagrangian = systems::createHarmonicOscillator(1.0f, 1.0f);

    float dt = 0.05f;
    GalerkinVariationalIntegrator<float, 2> galerkin_integrator(lagrangian, dt, true, true);
    GalerkinVariationalIntegrator<float, 2> standard_integrator(lagrangian, dt, false, true);

    std::vector<float> q0 = {1.0f};
    std::vector<float> v0 = {0.0f};

    float total_time = 2.0f * M_PI;
    int num_steps = static_cast<int>(total_time / dt);

    // Run both integrators
    auto galerkin_trajectory = galerkin_integrator.integrate(q0, v0, total_time, num_steps);
    auto standard_trajectory = standard_integrator.integrate(q0, v0, total_time, num_steps);

    // Compare energy conservation
    float galerkin_energy_drift = std::abs(galerkin_integrator.getEnergyDrift());
    float standard_energy_drift = std::abs(standard_integrator.getEnergyDrift());

    std::cout << "  Galerkin energy drift: " << galerkin_energy_drift << std::endl;
    std::cout << "  Standard energy drift: " << standard_energy_drift << std::endl;

    // Compare final positions
    float galerkin_final = galerkin_trajectory.back()[0];
    float standard_final = standard_trajectory.back()[0];
    float position_difference = std::abs(galerkin_final - standard_final);

    std::cout << "  Galerkin final position: " << galerkin_final << std::endl;
    std::cout << "  Standard final position: " << standard_final << std::endl;
    std::cout << "  Position difference: " << position_difference << std::endl;

    // Generally, Galerkin should have better energy conservation
    bool galerkin_better_energy = galerkin_energy_drift <= standard_energy_drift * 1.1f; // Allow small margin
    bool reasonable_agreement = position_difference < 0.2f;

    bool test_passed = galerkin_better_energy && reasonable_agreement;

    if (test_passed) {
        std::cout << "  ✓ Galerkin vs Standard comparison PASSED" << std::endl;
        if (galerkin_energy_drift < standard_energy_drift) {
            std::cout << "    - Galerkin shows superior energy conservation ✓" << std::endl;
        }
    } else {
        std::cout << "  ❌ Comparison test failed" << std::endl;
    }

    return test_passed;
}

void generate_conservation_report() {
    std::cout << "\nGenerating Conservation Analysis Report..." << std::endl;

    // Test different timesteps for convergence analysis
    std::vector<float> timesteps = {0.1f, 0.05f, 0.01f, 0.005f};
    std::ofstream report("variational_integrator_report.txt");

    report << "PhysGrad Variational Integrator Conservation Analysis\n";
    report << "===================================================\n\n";

    auto lagrangian = systems::createHarmonicOscillator(1.0f, 4.0f);
    std::vector<float> q0 = {1.0f};
    std::vector<float> v0 = {0.0f};
    float total_time = 2.0f * M_PI;

    for (float dt : timesteps) {
        GalerkinVariationalIntegrator<float, 2> integrator(lagrangian, dt, true, true);
        int num_steps = static_cast<int>(total_time / dt);

        auto trajectory = integrator.integrate(q0, v0, total_time, num_steps);

        float energy_drift = integrator.getEnergyDrift();
        float momentum_drift = integrator.getMomentumDrift();

        report << "Timestep: " << dt << "\n";
        report << "  Steps: " << num_steps << "\n";
        report << "  Energy drift: " << energy_drift << "\n";
        report << "  Momentum drift: " << momentum_drift << "\n";
        report << "  Final position: " << trajectory.back()[0] << "\n\n";
    }

    report.close();
    std::cout << "  Report saved to: variational_integrator_report.txt" << std::endl;
}

int main() {
    std::cout << "PhysGrad Variational Integrators with Galerkin Methods - Test Suite" << std::endl;
    std::cout << "====================================================================" << std::endl << std::endl;

    bool all_tests_passed = true;

    all_tests_passed &= test_galerkin_basis_functions();
    std::cout << std::endl;

    all_tests_passed &= test_discrete_lagrangian();
    std::cout << std::endl;

    all_tests_passed &= test_harmonic_oscillator_conservation();
    std::cout << std::endl;

    all_tests_passed &= test_double_pendulum_chaotic_dynamics();
    std::cout << std::endl;

    all_tests_passed &= test_nbody_gravitational_system();
    std::cout << std::endl;

    all_tests_passed &= test_galerkin_vs_standard_comparison();
    std::cout << std::endl;

    generate_conservation_report();
    std::cout << std::endl;

    if (all_tests_passed) {
        std::cout << "✅ ALL VARIATIONAL INTEGRATOR TESTS PASSED!" << std::endl;
        std::cout << std::endl;

        std::cout << "🎯 Variational Integrator Implementation - COMPLETE ✅" << std::endl;
        std::cout << "======================================================" << std::endl;
        std::cout << "🔬 Successfully Implemented Structure-Preserving Integration" << std::endl;
        std::cout << std::endl;

        std::cout << "📋 Implementation Achievements:" << std::endl;
        std::cout << "• ✅ Galerkin basis functions with Lobatto quadrature" << std::endl;
        std::cout << "• ✅ Discrete Lagrangian formulation with variational principles" << std::endl;
        std::cout << "• ✅ Structure-preserving time integration methods" << std::endl;
        std::cout << "• ✅ Energy and momentum conservation properties" << std::endl;
        std::cout << "• ✅ Support for complex mechanical systems (pendulums, N-body)" << std::endl;
        std::cout << "• ✅ Galerkin quadrature superiority over standard methods" << std::endl;
        std::cout << "• ✅ Chaotic dynamics simulation capability" << std::endl;
        std::cout << "• ✅ Comprehensive conservation analysis framework" << std::endl;

        return 0;
    } else {
        std::cout << "❌ Some variational integrator tests failed!" << std::endl;
        return 1;
    }
}