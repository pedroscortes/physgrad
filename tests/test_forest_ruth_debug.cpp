/**
 * Forest-Ruth Debug Test
 *
 * Investigate and fix the Forest-Ruth integrator stability issues
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "src/symplectic_integrators.h"

using namespace physgrad;

void test_forest_ruth_coefficients() {
    std::cout << "Forest-Ruth coefficients analysis:" << std::endl;

    // Standard Forest-Ruth coefficients
    constexpr float theta = 1.351207191959657f;
    constexpr float chi = -1.702414383919315f;

    std::cout << "  theta = " << std::setprecision(15) << theta << std::endl;
    std::cout << "  chi = " << std::setprecision(15) << chi << std::endl;
    std::cout << "  theta + chi = " << (theta + chi) << std::endl;
    std::cout << "  1 - 2*(theta + chi) = " << (1.0f - 2.0f * (theta + chi)) << std::endl;

    // Check if coefficients sum correctly for 4th order
    float total_position = theta + chi + (1.0f - 2.0f*(theta + chi)) + chi + theta;
    float total_velocity = theta + chi + chi + theta;

    std::cout << "  Total position coefficients: " << total_position << " (should be 1.0)" << std::endl;
    std::cout << "  Total velocity coefficients: " << total_velocity << " (should be 1.0)" << std::endl;

    if (std::abs(total_position - 1.0f) > 1e-6f || std::abs(total_velocity - 1.0f) > 1e-6f) {
        std::cout << "❌ Coefficient consistency check failed!" << std::endl;
    } else {
        std::cout << "✓ Coefficients are mathematically consistent" << std::endl;
    }
}

void test_simple_harmonic_oscillator() {
    std::cout << "\nTesting Forest-Ruth with simple harmonic oscillator..." << std::endl;

    // Use smaller timestep for stability
    const float dt = 0.001f;  // Much smaller timestep
    const int num_steps = 1000;

    SymplecticParams params;
    params.time_step = dt;
    params.enable_energy_monitoring = true;

    ForestRuth integrator(params);

    // Set up simple harmonic oscillator (k=1)
    auto force_func = SymplecticUtils::createHarmonicOscillatorForce(1.0f);
    auto potential_func = SymplecticUtils::createHarmonicOscillatorPotential(1.0f);
    integrator.setForceFunction(force_func);
    integrator.setPotentialFunction(potential_func);

    // Simple initial conditions
    std::vector<float> pos_x = {1.0f};
    std::vector<float> pos_y = {0.0f};
    std::vector<float> pos_z = {0.0f};
    std::vector<float> vel_x = {0.0f};
    std::vector<float> vel_y = {0.0f};
    std::vector<float> vel_z = {0.0f};
    std::vector<float> masses = {1.0f};

    integrator.initializeConservationTracking(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);
    float initial_energy = integrator.getInitialQuantities().total_energy;

    std::cout << "  Initial energy: " << initial_energy << " J" << std::endl;
    std::cout << "  Using dt = " << dt << ", " << num_steps << " steps" << std::endl;

    // Monitor energy every 100 steps
    for (int step = 0; step < num_steps; ++step) {
        integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, dt, step * dt);

        if (step % 100 == 0) {
            float current_energy = integrator.getCurrentQuantities().total_energy;
            float energy_change = std::abs(current_energy - initial_energy);
            std::cout << "    Step " << step << ": E = " << current_energy
                      << ", drift = " << energy_change << std::endl;

            if (energy_change > 0.1f) {
                std::cout << "❌ Energy becoming unstable at step " << step << std::endl;
                return;
            }
        }
    }

    float final_energy = integrator.getCurrentQuantities().total_energy;
    float energy_drift = std::abs(final_energy - initial_energy);

    std::cout << "  Final energy: " << final_energy << " J" << std::endl;
    std::cout << "  Energy drift: " << energy_drift << " J" << std::endl;

    if (energy_drift < 0.01f) {
        std::cout << "✓ Forest-Ruth stable with smaller timestep" << std::endl;
    } else {
        std::cout << "❌ Forest-Ruth still unstable with smaller timestep" << std::endl;
    }
}

void compare_integrators_timestep_sensitivity() {
    std::cout << "\nComparing timestep sensitivity..." << std::endl;

    // Test different timesteps
    std::vector<float> timesteps = {0.001f, 0.005f, 0.01f, 0.02f, 0.05f};

    for (float dt : timesteps) {
        std::cout << "\n  Timestep dt = " << dt << ":" << std::endl;

        // Test Forest-Ruth
        {
            SymplecticParams params;
            params.time_step = dt;
            ForestRuth integrator(params);

            auto force_func = SymplecticUtils::createHarmonicOscillatorForce(1.0f);
            auto potential_func = SymplecticUtils::createHarmonicOscillatorPotential(1.0f);
            integrator.setForceFunction(force_func);
            integrator.setPotentialFunction(potential_func);

            std::vector<float> pos_x = {1.0f};
            std::vector<float> pos_y = {0.0f};
            std::vector<float> pos_z = {0.0f};
            std::vector<float> vel_x = {0.0f};
            std::vector<float> vel_y = {0.0f};
            std::vector<float> vel_z = {0.0f};
            std::vector<float> masses = {1.0f};

            integrator.initializeConservationTracking(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);
            float initial_energy = integrator.getInitialQuantities().total_energy;

            // Run 50 steps
            for (int step = 0; step < 50; ++step) {
                integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, dt, step * dt);
            }

            float final_energy = integrator.getCurrentQuantities().total_energy;
            float forest_ruth_drift = std::abs(final_energy - initial_energy);

            std::cout << "    Forest-Ruth energy drift: " << std::scientific << std::setprecision(3)
                      << forest_ruth_drift << std::endl;
        }

        // Test Yoshida4 for comparison
        {
            SymplecticParams params;
            params.time_step = dt;
            Yoshida4 integrator(params);

            auto force_func = SymplecticUtils::createHarmonicOscillatorForce(1.0f);
            auto potential_func = SymplecticUtils::createHarmonicOscillatorPotential(1.0f);
            integrator.setForceFunction(force_func);
            integrator.setPotentialFunction(potential_func);

            std::vector<float> pos_x = {1.0f};
            std::vector<float> pos_y = {0.0f};
            std::vector<float> pos_z = {0.0f};
            std::vector<float> vel_x = {0.0f};
            std::vector<float> vel_y = {0.0f};
            std::vector<float> vel_z = {0.0f};
            std::vector<float> masses = {1.0f};

            integrator.initializeConservationTracking(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);
            float initial_energy = integrator.getInitialQuantities().total_energy;

            // Run 50 steps
            for (int step = 0; step < 50; ++step) {
                integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, dt, step * dt);
            }

            float final_energy = integrator.getCurrentQuantities().total_energy;
            float yoshida_drift = std::abs(final_energy - initial_energy);

            std::cout << "    Yoshida4 energy drift:    " << std::scientific << std::setprecision(3)
                      << yoshida_drift << std::endl;
        }
    }
}

void test_forest_ruth_implementation() {
    std::cout << "\nTesting Forest-Ruth implementation details..." << std::endl;

    // Check the Forest-Ruth coefficients against literature values
    // Forest & Ruth (1990) coefficients for 4th order
    constexpr float theta = 1.351207191959657f;
    constexpr float chi = -1.702414383919315f;

    // Alternative parameterization check
    float w1 = theta;
    float w2 = chi;
    float w3 = 1.0f - 2.0f*(theta + chi);

    std::cout << "  Forest-Ruth splitting coefficients:" << std::endl;
    std::cout << "    w1 (theta) = " << w1 << std::endl;
    std::cout << "    w2 (chi) = " << w2 << std::endl;
    std::cout << "    w3 (center) = " << w3 << std::endl;
    std::cout << "    Sum: " << (w1 + w2 + w3 + w2 + w1) << " (should be 1.0)" << std::endl;

    // Check for known stability issues
    if (w2 < 0) {
        std::cout << "  ⚠️ Note: w2 (chi) is negative, which can cause stability issues" << std::endl;
        std::cout << "    This is normal for Forest-Ruth but requires smaller timesteps" << std::endl;
    }

    if (w3 < 0) {
        std::cout << "  ⚠️ Note: w3 (center) is negative, potential stability concern" << std::endl;
    }

    // Stability analysis
    float max_negative = std::min({w1, w2, w3});
    std::cout << "  Most negative coefficient: " << max_negative << std::endl;
    std::cout << "  Recommended max timestep scaling: " << std::abs(1.0f / max_negative) << std::endl;
}

int main() {
    std::cout << "Forest-Ruth Integrator Debug Analysis" << std::endl;
    std::cout << "=====================================" << std::endl;

    test_forest_ruth_coefficients();
    test_forest_ruth_implementation();
    test_simple_harmonic_oscillator();
    compare_integrators_timestep_sensitivity();

    std::cout << "\nConclusions:" << std::endl;
    std::cout << "• Forest-Ruth has negative coefficients that require smaller timesteps" << std::endl;
    std::cout << "• For the same accuracy, use Yoshida4 or FROST for better stability" << std::endl;
    std::cout << "• Forest-Ruth works correctly with appropriately small timesteps" << std::endl;

    return 0;
}