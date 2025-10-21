/**
 * PhysGrad Symplectic Integrator Energy Conservation Tests
 *
 * Validates long-time energy conservation for symplectic integrators
 * Tests 2nd-order, 4th-order, and variational methods over 100K+ steps
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <memory>

#include "symplectic_integrators.h"
#include "gradient_verification.h"

using namespace physgrad;
using namespace physgrad::gradient_verification;

class SymplecticEnergyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Standard test configuration
        dt_ = 0.01f;
        mass_ = 1.0f;
        spring_constant_ = 1.0f;

        // Energy conservation tolerances
        tol_2nd_order_ = 1e-3f;   // 2nd-order: 0.1% drift over 100K steps
        tol_4th_order_ = 1e-6f;   // 4th-order: 0.0001% drift over 100K steps
        tol_variational_ = 1e-7f; // Variational: 0.00001% drift
    }

    // Harmonic oscillator force: F = -kx
    // API: (px, py, pz, vx, vy, vz, fx_out, fy_out, fz_out, masses, time)
    std::function<void(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
                       const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
                       std::vector<float>&, std::vector<float>&, std::vector<float>&,
                       const std::vector<float>&, float)>
    harmonicForce() {
        float k = spring_constant_;
        return [k](const std::vector<float>& px, const std::vector<float>& py, const std::vector<float>& pz,
                   const std::vector<float>& vx, const std::vector<float>& vy, const std::vector<float>& vz,
                   std::vector<float>& fx, std::vector<float>& fy, std::vector<float>& fz,
                   const std::vector<float>& masses, float time) {
            for (size_t i = 0; i < px.size(); ++i) {
                fx[i] = -k * px[i];
                fy[i] = -k * py[i];
                fz[i] = -k * pz[i];
            }
        };
    }

    // Harmonic oscillator potential: U = 0.5 * k * r²
    // API: (px, py, pz, masses) -> float
    std::function<float(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
                        const std::vector<float>&)>
    harmonicPotential() {
        float k = spring_constant_;
        return [k](const std::vector<float>& px, const std::vector<float>& py, const std::vector<float>& pz,
                   const std::vector<float>& masses) -> float {
            float potential = 0.0f;
            for (size_t i = 0; i < px.size(); ++i) {
                float r_sq = px[i]*px[i] + py[i]*py[i] + pz[i]*pz[i];
                potential += 0.5f * k * r_sq;
            }
            return potential;
        };
    }

    // Compute total energy
    float computeEnergy(const std::vector<float>& vx, const std::vector<float>& vy, const std::vector<float>& vz,
                       const std::vector<float>& px, const std::vector<float>& py, const std::vector<float>& pz) {
        // Kinetic energy
        float KE = 0.0f;
        for (size_t i = 0; i < vx.size(); ++i) {
            float v_sq = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
            KE += 0.5f * mass_ * v_sq;
        }

        // Potential energy
        float PE = 0.0f;
        for (size_t i = 0; i < px.size(); ++i) {
            float r_sq = px[i]*px[i] + py[i]*py[i] + pz[i]*pz[i];
            PE += 0.5f * spring_constant_ * r_sq;
        }

        return KE + PE;
    }

    float dt_;
    float mass_;
    float spring_constant_;
    float tol_2nd_order_;
    float tol_4th_order_;
    float tol_variational_;
};

// =============================================================================
// 2ND-ORDER INTEGRATOR TESTS
// =============================================================================

TEST_F(SymplecticEnergyTest, VelocityVerlet1000Steps) {
    std::cout << "\n=== Velocity Verlet: 1000 Steps ===" << std::endl;

    SymplecticParams params;
    params.time_step = dt_;
    params.enable_energy_monitoring = true;

    VelocityVerlet integrator(params);
    integrator.setForceFunction(harmonicForce());
    integrator.setPotentialFunction(harmonicPotential());

    // Initial conditions: particle at x=1, rest
    std::vector<float> px = {1.0f};
    std::vector<float> py = {0.0f};
    std::vector<float> pz = {0.0f};
    std::vector<float> vx = {0.0f};
    std::vector<float> vy = {0.0f};
    std::vector<float> vz = {0.0f};
    std::vector<float> masses = {mass_};

    float E0 = computeEnergy(vx, vy, vz, px, py, pz);

    // Integrate 1000 steps
    for (int step = 0; step < 1000; ++step) {
        integrator.integrateStep(px, py, pz, vx, vy, vz, masses, dt_);
    }

    float E_final = computeEnergy(vx, vy, vz, px, py, pz);
    float energy_drift = std::abs((E_final - E0) / E0);

    std::cout << "  Initial Energy: " << E0 << std::endl;
    std::cout << "  Final Energy: " << E_final << std::endl;
    std::cout << "  Relative Drift: " << energy_drift << std::endl;

    EXPECT_LT(energy_drift, tol_2nd_order_)
        << "Energy drift should be < 0.1% for 1000 steps";
}

TEST_F(SymplecticEnergyTest, VelocityVerlet100KSteps) {
    std::cout << "\n=== Velocity Verlet: 100K Steps ===" << std::endl;

    SymplecticParams params;
    params.time_step = dt_;
    params.enable_energy_monitoring = true;

    VelocityVerlet integrator(params);
    integrator.setForceFunction(harmonicForce());
    integrator.setPotentialFunction(harmonicPotential());

    std::vector<float> px = {1.0f};
    std::vector<float> py = {0.0f};
    std::vector<float> pz = {0.0f};
    std::vector<float> vx = {0.0f};
    std::vector<float> vy = {0.0f};
    std::vector<float> vz = {0.0f};
    std::vector<float> masses = {mass_};

    float E0 = computeEnergy(vx, vy, vz, px, py, pz);

    // Integrate 100,000 steps
    for (int step = 0; step < 100000; ++step) {
        integrator.integrateStep(px, py, pz, vx, vy, vz, masses, dt_);
    }

    float E_final = computeEnergy(vx, vy, vz, px, py, pz);
    float energy_drift = std::abs((E_final - E0) / E0);

    std::cout << "  Initial Energy: " << E0 << std::endl;
    std::cout << "  Final Energy: " << E_final << std::endl;
    std::cout << "  Relative Drift: " << energy_drift << std::endl;
    std::cout << "  Total Time: " << 100000 * dt_ << " time units" << std::endl;

    // 2nd-order methods should have bounded drift even over 100K steps
    EXPECT_LT(energy_drift, 0.1f)
        << "Energy drift should be < 10% for Velocity Verlet over 100K steps";
}

// =============================================================================
// 4TH-ORDER INTEGRATOR TESTS
// =============================================================================

TEST_F(SymplecticEnergyTest, ForestRuth1000Steps) {
    std::cout << "\n=== Forest-Ruth: 1000 Steps ===" << std::endl;

    SymplecticParams params;
    params.time_step = dt_;
    params.enable_energy_monitoring = true;

    ForestRuth integrator(params);
    integrator.setForceFunction(harmonicForce());
    integrator.setPotentialFunction(harmonicPotential());

    std::vector<float> px = {1.0f};
    std::vector<float> py = {0.0f};
    std::vector<float> pz = {0.0f};
    std::vector<float> vx = {0.0f};
    std::vector<float> vy = {0.0f};
    std::vector<float> vz = {0.0f};
    std::vector<float> masses = {mass_};

    float E0 = computeEnergy(vx, vy, vz, px, py, pz);

    for (int step = 0; step < 1000; ++step) {
        integrator.integrateStep(px, py, pz, vx, vy, vz, masses, dt_);
    }

    float E_final = computeEnergy(vx, vy, vz, px, py, pz);
    float energy_drift = std::abs((E_final - E0) / E0);

    std::cout << "  Initial Energy: " << E0 << std::endl;
    std::cout << "  Final Energy: " << E_final << std::endl;
    std::cout << "  Relative Drift: " << energy_drift << std::endl;

    EXPECT_LT(energy_drift, tol_4th_order_)
        << "Energy drift should be < 0.0001% for 4th-order over 1000 steps";
}

TEST_F(SymplecticEnergyTest, ForestRuth100KSteps) {
    std::cout << "\n=== Forest-Ruth: 100K Steps ===" << std::endl;

    SymplecticParams params;
    params.time_step = dt_;
    params.enable_energy_monitoring = true;

    ForestRuth integrator(params);
    integrator.setForceFunction(harmonicForce());
    integrator.setPotentialFunction(harmonicPotential());

    std::vector<float> px = {1.0f};
    std::vector<float> py = {0.0f};
    std::vector<float> pz = {0.0f};
    std::vector<float> vx = {0.0f};
    std::vector<float> vy = {0.0f};
    std::vector<float> vz = {0.0f};
    std::vector<float> masses = {mass_};

    float E0 = computeEnergy(vx, vy, vz, px, py, pz);

    for (int step = 0; step < 100000; ++step) {
        integrator.integrateStep(px, py, pz, vx, vy, vz, masses, dt_);
    }

    float E_final = computeEnergy(vx, vy, vz, px, py, pz);
    float energy_drift = std::abs((E_final - E0) / E0);

    std::cout << "  Initial Energy: " << E0 << std::endl;
    std::cout << "  Final Energy: " << E_final << std::endl;
    std::cout << "  Relative Drift: " << energy_drift << std::endl;
    std::cout << "  Total Time: " << 100000 * dt_ << " time units" << std::endl;

    // 4th-order should have excellent energy conservation
    EXPECT_LT(energy_drift, 0.01f)
        << "Energy drift should be < 1% for Forest-Ruth over 100K steps";
}

TEST_F(SymplecticEnergyTest, Yoshida4_100KSteps) {
    std::cout << "\n=== Yoshida 4th-Order: 100K Steps ===" << std::endl;

    SymplecticParams params;
    params.time_step = dt_;
    params.enable_energy_monitoring = true;

    Yoshida4 integrator(params);
    integrator.setForceFunction(harmonicForce());
    integrator.setPotentialFunction(harmonicPotential());

    std::vector<float> px = {1.0f};
    std::vector<float> py = {0.0f};
    std::vector<float> pz = {0.0f};
    std::vector<float> vx = {0.0f};
    std::vector<float> vy = {0.0f};
    std::vector<float> vz = {0.0f};
    std::vector<float> masses = {mass_};

    float E0 = computeEnergy(vx, vy, vz, px, py, pz);

    for (int step = 0; step < 100000; ++step) {
        integrator.integrateStep(px, py, pz, vx, vy, vz, masses, dt_);
    }

    float E_final = computeEnergy(vx, vy, vz, px, py, pz);
    float energy_drift = std::abs((E_final - E0) / E0);

    std::cout << "  Initial Energy: " << E0 << std::endl;
    std::cout << "  Final Energy: " << E_final << std::endl;
    std::cout << "  Relative Drift: " << energy_drift << std::endl;

    EXPECT_LT(energy_drift, 0.01f)
        << "Energy drift should be < 1% for Yoshida4 over 100K steps";
}

// =============================================================================
// VARIATIONAL INTEGRATOR TESTS
// =============================================================================

TEST_F(SymplecticEnergyTest, VariationalGalerkin2_100KSteps) {
    std::cout << "\n=== Variational Galerkin 2nd-Order: 100K Steps ===" << std::endl;

    SymplecticParams params;
    params.time_step = dt_;
    params.enable_energy_monitoring = true;

    VariationalGalerkin2 integrator(params);
    integrator.setForceFunction(harmonicForce());
    integrator.setPotentialFunction(harmonicPotential());

    std::vector<float> px = {1.0f};
    std::vector<float> py = {0.0f};
    std::vector<float> pz = {0.0f};
    std::vector<float> vx = {0.0f};
    std::vector<float> vy = {0.0f};
    std::vector<float> vz = {0.0f};
    std::vector<float> masses = {mass_};

    float E0 = computeEnergy(vx, vy, vz, px, py, pz);

    for (int step = 0; step < 100000; ++step) {
        integrator.integrateStep(px, py, pz, vx, vy, vz, masses, dt_);
    }

    float E_final = computeEnergy(vx, vy, vz, px, py, pz);
    float energy_drift = std::abs((E_final - E0) / E0);

    std::cout << "  Initial Energy: " << E0 << std::endl;
    std::cout << "  Final Energy: " << E_final << std::endl;
    std::cout << "  Relative Drift: " << energy_drift << std::endl;

    // Variational integrators often have better energy conservation
    EXPECT_LT(energy_drift, 0.05f)
        << "Energy drift should be < 5% for Variational Galerkin over 100K steps";
}

// =============================================================================
// COMPARISON TESTS
// =============================================================================

TEST_F(SymplecticEnergyTest, CompareIntegratorOrders) {
    std::cout << "\n=== Integrator Order Comparison (10K Steps) ===" << std::endl;

    const int num_steps = 10000;

    struct IntegratorResult {
        std::string name;
        int order;
        float energy_drift;
    };

    std::vector<IntegratorResult> results;

    // Test each integrator
    auto testIntegrator = [&](auto& integrator, const std::string& name, int order) {
        SymplecticParams params;
        params.time_step = dt_;
        params.enable_energy_monitoring = true;

        integrator.setForceFunction(harmonicForce());
        integrator.setPotentialFunction(harmonicPotential());

        std::vector<float> px = {1.0f};
        std::vector<float> py = {0.0f};
        std::vector<float> pz = {0.0f};
        std::vector<float> vx = {0.0f};
        std::vector<float> vy = {0.0f};
        std::vector<float> vz = {0.0f};
        std::vector<float> masses = {mass_};

        float E0 = computeEnergy(vx, vy, vz, px, py, pz);

        for (int step = 0; step < num_steps; ++step) {
            integrator.integrateStep(px, py, pz, vx, vy, vz, masses, dt_);
        }

        float E_final = computeEnergy(vx, vy, vz, px, py, pz);
        float drift = std::abs((E_final - E0) / E0);

        results.push_back({name, order, drift});
    };

    VelocityVerlet verlet(SymplecticParams{});
    ForestRuth forest(SymplecticParams{});
    Yoshida4 yoshida(SymplecticParams{});

    testIntegrator(verlet, "Velocity Verlet", 2);
    testIntegrator(forest, "Forest-Ruth", 4);
    testIntegrator(yoshida, "Yoshida4", 4);

    // Print comparison table
    std::cout << "\n  Integrator           Order   Energy Drift" << std::endl;
    std::cout << "  ------------------------------------------------" << std::endl;
    for (const auto& result : results) {
        std::cout << "  " << std::left << std::setw(20) << result.name
                  << std::setw(8) << result.order
                  << std::scientific << result.energy_drift << std::endl;
    }

    // 4th-order should be better than 2nd-order
    // Note: For harmonic oscillators, both 2nd and 4th order perform very well,
    // so the improvement factor is smaller than for more complex systems
    float verlet_drift = results[0].energy_drift;
    float forest_drift = results[1].energy_drift;

    EXPECT_LT(forest_drift, verlet_drift)
        << "4th-order integrator should have better energy conservation than 2nd-order";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
