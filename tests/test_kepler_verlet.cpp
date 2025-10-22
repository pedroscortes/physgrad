/**
 * Test Kepler problem with Verlet to verify the force function
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "../src/symplectic_integrators.h"

using namespace physgrad;

TEST(KeplerVerletTest, OrbitalConservation) {
    static constexpr float mu = 1.0f;
    float r0 = 1.0f;
    float v0 = std::sqrt(mu / r0);  // Circular orbit velocity

    std::vector<float> pos_x = {r0}, pos_y = {0.0f}, pos_z = {0.0f};
    std::vector<float> vel_x = {0.0f}, vel_y = {v0}, vel_z = {0.0f};
    std::vector<float> masses = {1.0f};

    auto gravitationalForce = [](const std::vector<float>& px, const std::vector<float>& py, const std::vector<float>& pz,
                                  const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
                                  std::vector<float>& fx, std::vector<float>& fy, std::vector<float>& fz,
                                  const std::vector<float>&, float) {
        float dx = px[0];
        float dy = py[0];
        float r = std::sqrt(dx*dx + dy*dy);
        float r3 = r * r * r;
        fx[0] = -mu * dx / r3;
        fy[0] = -mu * dy / r3;
        fz[0] = 0.0f;
    };

    auto gravitationalPotential = [](const std::vector<float>& px, const std::vector<float>& py,
                                     const std::vector<float>&, const std::vector<float>&) {
        float r = std::sqrt(px[0]*px[0] + py[0]*py[0]);
        return -mu / r;
    };

    SymplecticParams params;
    params.time_step = 0.01f;

    VelocityVerlet integrator(params);
    integrator.setForceFunction(gravitationalForce);
    integrator.setPotentialFunction(gravitationalPotential);

    // Integrate for 10 orbits
    float period = 2.0f * M_PI * std::sqrt(r0*r0*r0 / mu);
    float total_time = 10.0f * period;
    int num_steps = static_cast<int>(total_time / params.time_step);

    float initial_energy = 0.5f * (vel_x[0]*vel_x[0] + vel_y[0]*vel_y[0]) - mu / r0;
    std::cout << "\n=== Verlet on Kepler (10 orbits) ===\n";
    std::cout << "Initial energy: " << initial_energy << "\n";

    float time = 0.0f;
    float max_error = 0.0f;

    for (int step = 0; step < num_steps; ++step) {
        float r = std::sqrt(pos_x[0]*pos_x[0] + pos_y[0]*pos_y[0]);
        float v2 = vel_x[0]*vel_x[0] + vel_y[0]*vel_y[0];
        float energy = 0.5f * v2 - mu / r;
        float error = std::abs(energy - initial_energy);
        max_error = std::max(max_error, error);

        integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, params.time_step, time);
        time += params.time_step;
    }

    float r_final = std::sqrt(pos_x[0]*pos_x[0] + pos_y[0]*pos_y[0]);
    float v2_final = vel_x[0]*vel_x[0] + vel_y[0]*vel_y[0];
    float final_energy = 0.5f * v2_final - mu / r_final;

    std::cout << "Final energy: " << final_energy << "\n";
    std::cout << "Max error: " << max_error << "\n";
    std::cout << "Relative error: " << (max_error / std::abs(initial_energy) * 100.0f) << "%\n";

    EXPECT_LT(max_error / std::abs(initial_energy), 0.01f) << "Verlet should conserve Kepler energy reasonably";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
