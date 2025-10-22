/**
 * Diagnostic test to understand FROST integrator behavior
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "../src/symplectic_integrators.h"

using namespace physgrad;

// Simple harmonic oscillator for debugging
class FROSTDiagnosticTest : public ::testing::Test {
protected:
    static constexpr float k = 1.0f;
    static constexpr float m = 1.0f;

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

    static void harmonicForceGradient(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& masses,
        std::vector<std::vector<float>>& grad_xx, std::vector<std::vector<float>>& grad_xy, std::vector<std::vector<float>>& grad_xz) {

        size_t n = pos_x.size();
        grad_xx.resize(n, std::vector<float>(n, 0.0f));
        grad_xy.resize(n, std::vector<float>(n, 0.0f));
        grad_xz.resize(n, std::vector<float>(n, 0.0f));

        for (size_t i = 0; i < n; ++i) {
            grad_xx[i][i] = -k;  // dF/dx = -k
        }

        std::cout << "Force gradient called: grad_xx[0][0] = " << grad_xx[0][0] << "\n";
    }

    static float harmonicPotential(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& masses) {

        float potential = 0.0f;
        for (size_t i = 0; i < pos_x.size(); ++i) {
            potential += 0.5f * k * pos_x[i] * pos_x[i];
        }
        return potential;
    }
};

TEST_F(FROSTDiagnosticTest, SingleStepAnalysis) {
    std::vector<float> pos_x = {1.0f}, pos_y = {0.0f}, pos_z = {0.0f};
    std::vector<float> vel_x = {0.0f}, vel_y = {0.0f}, vel_z = {0.0f};
    std::vector<float> masses = {m};

    SymplecticParams params;
    params.time_step = 0.1f;
    float dt = params.time_step;

    FrostForwardSymplectic4 integrator(params);
    integrator.setForceFunction(harmonicForce);
    integrator.setPotentialFunction(harmonicPotential);
    integrator.setForceGradientFunction(harmonicForceGradient);

    std::cout << "\n=== Initial State ===\n";
    std::cout << "Position: " << pos_x[0] << "\n";
    std::cout << "Velocity: " << vel_x[0] << "\n";

    float E0 = 0.5f * k * pos_x[0] * pos_x[0] + 0.5f * m * vel_x[0] * vel_x[0];
    std::cout << "Energy: " << E0 << "\n";

    // Analytical solution for single step
    float omega = std::sqrt(k / m);
    float pos_analytical = pos_x[0] * std::cos(omega * dt) + vel_x[0] * std::sin(omega * dt) / omega;
    float vel_analytical = -pos_x[0] * omega * std::sin(omega * dt) + vel_x[0] * std::cos(omega * dt);

    std::cout << "\n=== Analytical Solution (dt=" << dt << ") ===\n";
    std::cout << "Position: " << pos_analytical << "\n";
    std::cout << "Velocity: " << vel_analytical << "\n";
    float E_analytical = 0.5f * k * pos_analytical * pos_analytical + 0.5f * m * vel_analytical * vel_analytical;
    std::cout << "Energy: " << E_analytical << "\n";

    // FROST integration
    std::cout << "\n=== FROST Integration ===\n";
    float time = 0.0f;
    integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, dt, time);

    std::cout << "Position: " << pos_x[0] << "\n";
    std::cout << "Velocity: " << vel_x[0] << "\n";
    float E1 = 0.5f * k * pos_x[0] * pos_x[0] + 0.5f * m * vel_x[0] * vel_x[0];
    std::cout << "Energy: " << E1 << "\n";

    std::cout << "\n=== Errors ===\n";
    std::cout << "Position error: " << (pos_x[0] - pos_analytical) << "\n";
    std::cout << "Velocity error: " << (vel_x[0] - vel_analytical) << "\n";
    std::cout << "Energy error: " << (E1 - E0) << " (" << (E1 - E0) / E0 * 100.0f << "%)\n";
}

TEST_F(FROSTDiagnosticTest, CompareWithVerlet) {
    // FROST
    std::vector<float> pos_x_frost = {1.0f}, pos_y = {0.0f}, pos_z = {0.0f};
    std::vector<float> vel_x_frost = {0.0f}, vel_y = {0.0f}, vel_z = {0.0f};
    std::vector<float> masses = {m};

    SymplecticParams params;
    params.time_step = 0.01f;

    FrostForwardSymplectic4 frost(params);
    frost.setForceFunction(harmonicForce);
    frost.setPotentialFunction(harmonicPotential);
    frost.setForceGradientFunction(harmonicForceGradient);

    // Verlet
    std::vector<float> pos_x_verlet = pos_x_frost;
    std::vector<float> vel_x_verlet = vel_x_frost;

    VelocityVerlet verlet(params);
    verlet.setForceFunction(harmonicForce);
    verlet.setPotentialFunction(harmonicPotential);

    float E0 = 0.5f * k * pos_x_frost[0] * pos_x_frost[0];

    std::cout << "\n=== Energy Conservation Comparison (10 periods) ===\n";
    std::cout << "Initial Energy: " << E0 << "\n\n";

    int num_steps = static_cast<int>(10.0f * 2.0f * M_PI / params.time_step);
    float time = 0.0f;

    float max_frost_error = 0.0f;
    float max_verlet_error = 0.0f;

    for (int step = 0; step < num_steps; ++step) {
        frost.integrateStep(pos_x_frost, pos_y, pos_z, vel_x_frost, vel_y, vel_z, masses, params.time_step, time);
        verlet.integrateStep(pos_x_verlet, pos_y, pos_z, vel_x_verlet, vel_y, vel_z, masses, params.time_step, time);
        time += params.time_step;

        float E_frost = 0.5f * k * pos_x_frost[0] * pos_x_frost[0] + 0.5f * m * vel_x_frost[0] * vel_x_frost[0];
        float E_verlet = 0.5f * k * pos_x_verlet[0] * pos_x_verlet[0] + 0.5f * m * vel_x_verlet[0] * vel_x_verlet[0];

        max_frost_error = std::max(max_frost_error, std::abs(E_frost - E0));
        max_verlet_error = std::max(max_verlet_error, std::abs(E_verlet - E0));
    }

    std::cout << "FROST max energy error:  " << max_frost_error << " (" << max_frost_error / E0 * 100.0f << "%)\n";
    std::cout << "Verlet max energy error: " << max_verlet_error << " (" << max_verlet_error / E0 * 100.0f << "%)\n";

    // FROST should be more accurate (4th order vs 2nd order)
    // But currently it's worse due to bugs
    std::cout << "\nNote: FROST (4th order) should be MORE accurate than Verlet (2nd order)\n";
    std::cout << "If FROST is worse, there's a bug in the implementation.\n";
}
