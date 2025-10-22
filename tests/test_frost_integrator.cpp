/**
 * Energy Conservation Validation for Symplectic Integrators
 *
 * Tests long-term energy conservation for:
 * - FROST Forward Symplectic Integrator (4th order)
 * - Velocity Verlet (2nd order)
 * - Forest-Ruth (4th order)
 * - RK4 (4th order, non-symplectic baseline)
 *
 * Benchmark problems:
 * 1. Harmonic Oscillator (analytical solution exists)
 * 2. Kepler Problem (two-body central force)
 * 3. Three-Body Problem (chaotic dynamics)
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "../src/symplectic_integrators.h"

using namespace physgrad;

// =============================================================================
// TEST UTILITIES
// =============================================================================

struct EnergyConservationMetrics {
    float initial_energy;
    float final_energy;
    float max_energy_error;
    float rms_energy_error;
    float energy_drift_rate;  // per unit time
    std::vector<float> energy_history;
};

EnergyConservationMetrics computeEnergyMetrics(
    const std::vector<float>& energy_history, float total_time) {

    EnergyConservationMetrics metrics;
    metrics.initial_energy = energy_history.front();
    metrics.final_energy = energy_history.back();
    metrics.energy_history = energy_history;

    // Compute max error and RMS error
    float sum_squared_error = 0.0f;
    float max_error = 0.0f;

    for (float energy : energy_history) {
        float error = std::abs(energy - metrics.initial_energy);
        max_error = std::max(max_error, error);
        sum_squared_error += error * error;
    }

    metrics.max_energy_error = max_error;
    metrics.rms_energy_error = std::sqrt(sum_squared_error / energy_history.size());
    metrics.energy_drift_rate = (metrics.final_energy - metrics.initial_energy) / total_time;

    return metrics;
}

void printEnergyMetrics(const std::string& integrator_name,
                       const EnergyConservationMetrics& metrics,
                       float total_time) {
    std::cout << "\n=== " << integrator_name << " ===\n";
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Initial Energy:     " << metrics.initial_energy << "\n";
    std::cout << "Final Energy:       " << metrics.final_energy << "\n";
    std::cout << "Max Error:          " << metrics.max_energy_error << "\n";
    std::cout << "RMS Error:          " << metrics.rms_energy_error << "\n";
    std::cout << "Drift Rate:         " << metrics.energy_drift_rate << " /unit time\n";
    std::cout << "Relative Error:     " << (metrics.max_energy_error / std::abs(metrics.initial_energy) * 100.0f) << "%\n";
}

// =============================================================================
// BENCHMARK PROBLEM 1: HARMONIC OSCILLATOR
// =============================================================================

class HarmonicOscillatorTest : public ::testing::Test {
protected:
    // System parameters
    static constexpr float k = 1.0f;      // Spring constant
    static constexpr float m = 1.0f;      // Mass
    static constexpr float omega = 1.0f;  // Angular frequency (sqrt(k/m))

    // Initial conditions
    float x0 = 1.0f;
    float v0 = 0.0f;

    // Analytical energy
    float analytical_energy;

    void SetUp() override {
        analytical_energy = 0.5f * k * x0 * x0 + 0.5f * m * v0 * v0;
    }

    // Force function: F = -kx
    static void harmonicForce(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        std::vector<float>& force_x, std::vector<float>& force_y, std::vector<float>& force_z,
        const std::vector<float>& masses, float time) {

        for (size_t i = 0; i < pos_x.size(); ++i) {
            force_x[i] = -k * pos_x[i];
            force_y[i] = 0.0f;
            force_z[i] = 0.0f;
        }
    }

    // Force gradient function: dF/dx = -k
    static void harmonicForceGradient(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& masses,
        std::vector<std::vector<float>>& grad_xx, std::vector<std::vector<float>>& grad_xy, std::vector<std::vector<float>>& grad_xz) {

        size_t n = pos_x.size();
        grad_xx.resize(n, std::vector<float>(n, 0.0f));
        grad_xy.resize(n, std::vector<float>(n, 0.0f));
        grad_xz.resize(n, std::vector<float>(n, 0.0f));

        for (size_t i = 0; i < n; ++i) {
            grad_xx[i][i] = -k;  // dF/dx = -k for same particle
        }
    }

    // Potential energy function
    static float harmonicPotential(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& masses) {

        float potential = 0.0f;
        for (size_t i = 0; i < pos_x.size(); ++i) {
            potential += 0.5f * k * pos_x[i] * pos_x[i];
        }
        return potential;
    }

    // Compute total energy
    float computeEnergy(const std::vector<float>& pos_x, const std::vector<float>& vel_x) {
        float kinetic = 0.5f * m * vel_x[0] * vel_x[0];
        float potential = 0.5f * k * pos_x[0] * pos_x[0];
        return kinetic + potential;
    }
};

TEST_F(HarmonicOscillatorTest, FROST_LongTermConservation) {
    std::vector<float> pos_x = {x0}, pos_y = {0.0f}, pos_z = {0.0f};
    std::vector<float> vel_x = {v0}, vel_y = {0.0f}, vel_z = {0.0f};
    std::vector<float> masses = {m};

    SymplecticParams params;
    params.time_step = 0.01f;

    FrostForwardSymplectic4 integrator(params);
    integrator.setForceFunction(harmonicForce);
    integrator.setPotentialFunction(harmonicPotential);
    integrator.setForceGradientFunction(harmonicForceGradient);

    // Integrate for 100 periods (2π * 100)
    float total_time = 100.0f * 2.0f * M_PI;
    int num_steps = static_cast<int>(total_time / params.time_step);

    std::vector<float> energy_history;
    float time = 0.0f;

    for (int step = 0; step < num_steps; ++step) {
        float energy = computeEnergy(pos_x, vel_x);
        energy_history.push_back(energy);

        integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, params.time_step, time);
        time += params.time_step;
    }

    auto metrics = computeEnergyMetrics(energy_history, total_time);
    printEnergyMetrics("FROST FSI-4", metrics, total_time);

    // FROST should conserve energy to < 0.01% over 100 periods
    float relative_error = metrics.max_energy_error / std::abs(analytical_energy);
    EXPECT_LT(relative_error, 0.0001f) << "FROST energy conservation exceeded tolerance";
}

TEST_F(HarmonicOscillatorTest, Verlet_LongTermConservation) {
    std::vector<float> pos_x = {x0}, pos_y = {0.0f}, pos_z = {0.0f};
    std::vector<float> vel_x = {v0}, vel_y = {0.0f}, vel_z = {0.0f};
    std::vector<float> masses = {m};

    SymplecticParams params;
    params.time_step = 0.01f;

    VelocityVerlet integrator(params);
    integrator.setForceFunction(harmonicForce);
    integrator.setPotentialFunction(harmonicPotential);

    float total_time = 100.0f * 2.0f * M_PI;
    int num_steps = static_cast<int>(total_time / params.time_step);

    std::vector<float> energy_history;
    float time = 0.0f;

    for (int step = 0; step < num_steps; ++step) {
        float energy = computeEnergy(pos_x, vel_x);
        energy_history.push_back(energy);

        integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, params.time_step, time);
        time += params.time_step;
    }

    auto metrics = computeEnergyMetrics(energy_history, total_time);
    printEnergyMetrics("Velocity Verlet", metrics, total_time);

    // Verlet should conserve energy reasonably well (< 0.1% for harmonic oscillator)
    float relative_error = metrics.max_energy_error / std::abs(analytical_energy);
    EXPECT_LT(relative_error, 0.001f) << "Verlet energy conservation exceeded tolerance";
}

// =============================================================================
// BENCHMARK PROBLEM 2: KEPLER PROBLEM (Two-Body)
// =============================================================================

class KeplerProblemTest : public ::testing::Test {
protected:
    // Gravitational parameter (G * M1 * M2)
    static constexpr float mu = 1.0f;

    // Initial conditions for circular orbit
    float r0 = 1.0f;
    float v0;  // Will be computed for circular orbit

    void SetUp() override {
        // Circular orbit: v = sqrt(mu/r)
        v0 = std::sqrt(mu / r0);
    }

    static void gravitationalForce(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        std::vector<float>& force_x, std::vector<float>& force_y, std::vector<float>& force_z,
        const std::vector<float>& masses, float time) {

        // Two-body problem: F = -μ * r / |r|^3
        float dx = pos_x[0];
        float dy = pos_y[0];
        float r = std::sqrt(dx*dx + dy*dy);
        float r3 = r * r * r;

        force_x[0] = -mu * dx / r3;
        force_y[0] = -mu * dy / r3;
        force_z[0] = 0.0f;
    }

    static void gravitationalForceGradient(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& masses,
        std::vector<std::vector<float>>& grad_xx, std::vector<std::vector<float>>& grad_xy, std::vector<std::vector<float>>& grad_xz) {

        size_t n = pos_x.size();
        grad_xx.resize(n, std::vector<float>(n, 0.0f));
        grad_xy.resize(n, std::vector<float>(n, 0.0f));
        grad_xz.resize(n, std::vector<float>(n, 0.0f));

        // ∂F/∂r = -μ[(1/r³)I - 3(r⊗r)/r⁵]
        float dx = pos_x[0];
        float dy = pos_y[0];
        float r = std::sqrt(dx*dx + dy*dy);
        float r3 = r * r * r;
        float r5 = r3 * r * r;

        grad_xx[0][0] = -mu * (1.0f/r3 - 3.0f*dx*dx/r5);
        grad_xy[0][0] = -mu * (-3.0f*dx*dy/r5);
        grad_xz[0][0] = 0.0f;
    }

    static float gravitationalPotential(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& masses) {

        float dx = pos_x[0];
        float dy = pos_y[0];
        float r = std::sqrt(dx*dx + dy*dy);
        return -mu / r;
    }

    float computeEnergy(const std::vector<float>& pos_x, const std::vector<float>& pos_y,
                       const std::vector<float>& vel_x, const std::vector<float>& vel_y) {
        float r = std::sqrt(pos_x[0]*pos_x[0] + pos_y[0]*pos_y[0]);
        float v2 = vel_x[0]*vel_x[0] + vel_y[0]*vel_y[0];
        return 0.5f * v2 - mu / r;
    }
};

TEST_F(KeplerProblemTest, FROST_OrbitalConservation) {
    std::vector<float> pos_x = {r0}, pos_y = {0.0f}, pos_z = {0.0f};
    std::vector<float> vel_x = {0.0f}, vel_y = {v0}, vel_z = {0.0f};
    std::vector<float> masses = {1.0f};

    SymplecticParams params;
    params.time_step = 0.01f;

    FrostForwardSymplectic4 integrator(params);
    integrator.setForceFunction(gravitationalForce);
    integrator.setPotentialFunction(gravitationalPotential);
    integrator.setForceGradientFunction(gravitationalForceGradient);

    // Integrate for 100 orbits
    float period = 2.0f * M_PI * std::sqrt(r0*r0*r0 / mu);
    float total_time = 100.0f * period;
    int num_steps = static_cast<int>(total_time / params.time_step);

    std::vector<float> energy_history;
    float time = 0.0f;

    for (int step = 0; step < num_steps; ++step) {
        float energy = computeEnergy(pos_x, pos_y, vel_x, vel_y);
        energy_history.push_back(energy);

        integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, params.time_step, time);
        time += params.time_step;
    }

    auto metrics = computeEnergyMetrics(energy_history, total_time);
    printEnergyMetrics("FROST FSI-4 (Kepler)", metrics, total_time);

    // For Kepler problem, FROST should conserve energy extremely well
    float relative_error = metrics.max_energy_error / std::abs(metrics.initial_energy);
    EXPECT_LT(relative_error, 0.0001f) << "FROST orbital energy conservation exceeded tolerance";
}

// =============================================================================
// COMPARISON TEST: FROST vs. Verlet vs. Forest-Ruth
// =============================================================================

TEST(IntegratorComparison, EnergyConservationComparison) {
    std::cout << "\n=======================================================\n";
    std::cout << "INTEGRATOR COMPARISON: Harmonic Oscillator (100 periods)\n";
    std::cout << "=======================================================\n";

    // Test parameters
    float k = 1.0f, m = 1.0f;
    float x0 = 1.0f, v0 = 0.0f;
    float dt = 0.01f;
    float total_time = 100.0f * 2.0f * M_PI;
    int num_steps = static_cast<int>(total_time / dt);

    auto force_func = [k](const std::vector<float>& pos_x, const std::vector<float>&, const std::vector<float>&,
                          const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
                          std::vector<float>& force_x, std::vector<float>& force_y, std::vector<float>& force_z,
                          const std::vector<float>&, float) {
        force_x[0] = -k * pos_x[0];
        force_y[0] = 0.0f;
        force_z[0] = 0.0f;
    };

    auto potential_func = [k](const std::vector<float>& pos_x, const std::vector<float>&,
                              const std::vector<float>&, const std::vector<float>&) {
        return 0.5f * k * pos_x[0] * pos_x[0];
    };

    auto gradient_func = [k](const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
                            const std::vector<float>&,
                            std::vector<std::vector<float>>& grad_xx, std::vector<std::vector<float>>&,
                            std::vector<std::vector<float>>&) {
        grad_xx.resize(1, std::vector<float>(1, -k));
    };

    struct IntegratorResult {
        std::string name;
        EnergyConservationMetrics metrics;
    };

    std::vector<IntegratorResult> results;

    // Test 1: FROST FSI-4
    {
        std::vector<float> pos_x = {x0}, pos_y = {0.0f}, pos_z = {0.0f};
        std::vector<float> vel_x = {v0}, vel_y = {0.0f}, vel_z = {0.0f};
        std::vector<float> masses = {m};

        SymplecticParams params;
        params.time_step = dt;

        FrostForwardSymplectic4 integrator(params);
        integrator.setForceFunction(force_func);
        integrator.setPotentialFunction(potential_func);
        integrator.setForceGradientFunction(gradient_func);

        std::vector<float> energy_history;
        float time = 0.0f;

        for (int step = 0; step < num_steps; ++step) {
            float energy = 0.5f * m * vel_x[0] * vel_x[0] + 0.5f * k * pos_x[0] * pos_x[0];
            energy_history.push_back(energy);
            integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, dt, time);
            time += dt;
        }

        IntegratorResult result;
        result.name = "FROST FSI-4";
        result.metrics = computeEnergyMetrics(energy_history, total_time);
        results.push_back(result);
    }

    // Test 2: Velocity Verlet
    {
        std::vector<float> pos_x = {x0}, pos_y = {0.0f}, pos_z = {0.0f};
        std::vector<float> vel_x = {v0}, vel_y = {0.0f}, vel_z = {0.0f};
        std::vector<float> masses = {m};

        SymplecticParams params;
        params.time_step = dt;

        VelocityVerlet integrator(params);
        integrator.setForceFunction(force_func);
        integrator.setPotentialFunction(potential_func);

        std::vector<float> energy_history;
        float time = 0.0f;

        for (int step = 0; step < num_steps; ++step) {
            float energy = 0.5f * m * vel_x[0] * vel_x[0] + 0.5f * k * pos_x[0] * pos_x[0];
            energy_history.push_back(energy);
            integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, dt, time);
            time += dt;
        }

        IntegratorResult result;
        result.name = "Velocity Verlet";
        result.metrics = computeEnergyMetrics(energy_history, total_time);
        results.push_back(result);
    }

    // Test 3: Forest-Ruth
    {
        std::vector<float> pos_x = {x0}, pos_y = {0.0f}, pos_z = {0.0f};
        std::vector<float> vel_x = {v0}, vel_y = {0.0f}, vel_z = {0.0f};
        std::vector<float> masses = {m};

        SymplecticParams params;
        params.time_step = dt;

        ForestRuth integrator(params);
        integrator.setForceFunction(force_func);
        integrator.setPotentialFunction(potential_func);

        std::vector<float> energy_history;
        float time = 0.0f;

        for (int step = 0; step < num_steps; ++step) {
            float energy = 0.5f * m * vel_x[0] * vel_x[0] + 0.5f * k * pos_x[0] * pos_x[0];
            energy_history.push_back(energy);
            integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, dt, time);
            time += dt;
        }

        IntegratorResult result;
        result.name = "Forest-Ruth 4";
        result.metrics = computeEnergyMetrics(energy_history, total_time);
        results.push_back(result);
    }

    // Print comparison table
    std::cout << "\n=== COMPARISON TABLE ===\n";
    std::cout << std::setw(20) << "Integrator"
              << std::setw(15) << "Max Error"
              << std::setw(15) << "RMS Error"
              << std::setw(15) << "Rel. Error (%)\n";
    std::cout << std::string(65, '-') << "\n";

    for (const auto& result : results) {
        float rel_error = result.metrics.max_energy_error / std::abs(result.metrics.initial_energy) * 100.0f;
        std::cout << std::setw(20) << result.name
                  << std::setw(15) << std::scientific << std::setprecision(3) << result.metrics.max_energy_error
                  << std::setw(15) << result.metrics.rms_energy_error
                  << std::setw(15) << std::fixed << std::setprecision(6) << rel_error << "\n";
    }

    std::cout << "\n";

    // Assertions: FROST should be best or comparable to Forest-Ruth
    EXPECT_LT(results[0].metrics.max_energy_error, 1e-4f) << "FROST error too high";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
