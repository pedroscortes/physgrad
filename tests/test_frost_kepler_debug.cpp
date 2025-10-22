/**
 * Debug FROST on Kepler problem
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "../src/symplectic_integrators.h"

using namespace physgrad;

TEST(FROSTKeplerDebug, SingleStep) {
    static constexpr float mu = 1.0f;
    float r0 = 1.0f;
    float v0 = std::sqrt(mu / r0);

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

    auto gravitationalForceGradient = [](const std::vector<float>& px, const std::vector<float>& py, const std::vector<float>&,
                                         const std::vector<float>&,
                                         std::vector<std::vector<float>>& gxx, std::vector<std::vector<float>>& gxy,
                                         std::vector<std::vector<float>>& gxz) {
        float dx = px[0];
        float dy = py[0];
        float r = std::sqrt(dx*dx + dy*dy);
        float r3 = r * r * r;
        float r5 = r3 * r * r;

        gxx.resize(1, std::vector<float>(1));
        gxy.resize(1, std::vector<float>(1));
        gxz.resize(1, std::vector<float>(1));

        gxx[0][0] = -mu * (1.0f/r3 - 3.0f*dx*dx/r5);  // ∂Fx/∂x
        gxy[0][0] = -mu * (-3.0f*dx*dy/r5);            // ∂Fx/∂y
        gxz[0][0] = -mu * (1.0f/r3 - 3.0f*dy*dy/r5);   // ∂Fy/∂y (hack)

        std::cout << "Gradients at (x=" << dx << ", y=" << dy << "):\n";
        std::cout << "  ∂Fx/∂x = " << gxx[0][0] << "\n";
        std::cout << "  ∂Fx/∂y = " << gxy[0][0] << "\n";
        std::cout << "  ∂Fy/∂y = " << gxz[0][0] << " (in gxz)\n";
    };

    SymplecticParams params;
    params.time_step = 0.01f;

    FrostForwardSymplectic4 integrator(params);
    integrator.setForceFunction(gravitationalForce);
    integrator.setPotentialFunction(gravitationalPotential);
    integrator.setForceGradientFunction(gravitationalForceGradient);

    float E0 = 0.5f * v0*v0 - mu / r0;
    std::cout << "\n=== FROST Kepler Single Step ===\n";
    std::cout << "Initial: pos=(" << pos_x[0] << ", " << pos_y[0] << "), vel=(" << vel_x[0] << ", " << vel_y[0] << ")\n";
    std::cout << "Initial energy: " << E0 << "\n\n";

    float time = 0.0f;
    integrator.integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, params.time_step, time);

    float r = std::sqrt(pos_x[0]*pos_x[0] + pos_y[0]*pos_y[0]);
    float v2 = vel_x[0]*vel_x[0] + vel_y[0]*vel_y[0];
    float E1 = 0.5f * v2 - mu / r;

    std::cout << "\nFinal: pos=(" << pos_x[0] << ", " << pos_y[0] << "), vel=(" << vel_x[0] << ", " << vel_y[0] << ")\n";
    std::cout << "Final energy: " << E1 << "\n";
    std::cout << "Energy change: " << (E1 - E0) << " (" << ((E1-E0)/E0 * 100.0f) << "%)\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
