#include "test_framework.h"
#include "../src/symplectic_integrators.h"
#include <random>
#include <map>

using namespace physgrad;
using namespace physgrad_tests;

void testSymplecticIntegrators(TestFramework& framework) {
    framework.setCategory("Symplectic Integrators");

    // Test harmonic oscillator conservation
    framework.runTest("Harmonic Oscillator Energy Conservation", []() {
        SymplecticParams params;
        params.time_step = 0.01f;
        params.enable_energy_monitoring = true;

        auto integrator = SymplecticIntegratorFactory::create(SymplecticScheme::VELOCITY_VERLET, params);

        std::vector<float> pos_x = {1.0f};
        std::vector<float> pos_y = {0.0f};
        std::vector<float> pos_z = {0.0f};
        std::vector<float> vel_x = {0.0f};
        std::vector<float> vel_y = {1.0f};
        std::vector<float> vel_z = {0.0f};
        std::vector<float> masses = {1.0f};

        float k = 1.0f;
        auto force_func = SymplecticUtils::createHarmonicOscillatorForce(k);
        auto potential_func = SymplecticUtils::createHarmonicOscillatorPotential(k);

        integrator->setForceFunction(force_func);
        integrator->setPotentialFunction(potential_func);
        integrator->initializeConservationTracking(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);

        float initial_energy = integrator->getInitialQuantities().total_energy;

        // Simulate for 10 periods
        float period = 2.0f * M_PI / std::sqrt(k);
        float simulation_time = 10.0f * period;
        float time = 0.0f;

        while (time < simulation_time) {
            integrator->integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, params.time_step, time);
            time += params.time_step;
        }

        float final_energy = integrator->getCurrentQuantities().total_energy;
        float energy_drift = std::abs(final_energy - initial_energy) / initial_energy;

        ASSERT_LT(energy_drift, 1e-3f); // Energy should be conserved to within 0.1%
    });

    // Test different integrator orders
    framework.runTest("Integrator Order Verification", []() {
        std::vector<SymplecticScheme> schemes = {
            SymplecticScheme::SYMPLECTIC_EULER,
            SymplecticScheme::VELOCITY_VERLET,
            SymplecticScheme::YOSHIDA4,
            SymplecticScheme::BLANES_MOAN8
        };

        std::vector<int> expected_orders = {1, 2, 4, 8};

        for (size_t i = 0; i < schemes.size(); ++i) {
            int actual_order = SymplecticIntegratorFactory::getSchemeOrder(schemes[i]);
            ASSERT_EQ(expected_orders[i], actual_order);
        }
    });

    // Test momentum conservation for isolated system
    framework.runTest("Linear Momentum Conservation", []() {
        SymplecticParams params;
        params.time_step = 0.005f;
        params.enable_momentum_conservation = true;

        auto integrator = SymplecticIntegratorFactory::create(SymplecticScheme::VELOCITY_VERLET, params);

        // Two-body system with equal masses and opposite velocities
        std::vector<float> pos_x = {-0.5f, 0.5f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> vel_x = {1.0f, -1.0f};
        std::vector<float> vel_y = {0.0f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f};

        // No external forces - should conserve momentum perfectly
        auto force_func = [](const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
                           const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
                           std::vector<float>& fx, std::vector<float>& fy, std::vector<float>& fz,
                           const std::vector<float>&, float) {
            std::fill(fx.begin(), fx.end(), 0.0f);
            std::fill(fy.begin(), fy.end(), 0.0f);
            std::fill(fz.begin(), fz.end(), 0.0f);
        };

        integrator->setForceFunction(force_func);
        integrator->initializeConservationTracking(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);

        float initial_momentum = PhysicsTestUtils::computeLinearMomentumMagnitude(vel_x, vel_y, vel_z, masses);
        ASSERT_NEAR(initial_momentum, 0.0f, 1e-6f); // Should start with zero momentum

        // Simulate for some time
        for (int step = 0; step < 1000; ++step) {
            integrator->integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, params.time_step);
        }

        float final_momentum = PhysicsTestUtils::computeLinearMomentumMagnitude(vel_x, vel_y, vel_z, masses);
        ASSERT_NEAR(final_momentum, 0.0f, 1e-5f); // Momentum should remain zero
    });

    // Test gravitational two-body problem
    framework.runTest("Gravitational Two-Body Problem", []() {
        SymplecticParams params;
        params.time_step = 0.01f;
        params.enable_energy_monitoring = true;

        auto integrator = SymplecticIntegratorFactory::create(SymplecticScheme::VELOCITY_VERLET, params);

        // Circular orbit setup
        std::vector<float> pos_x = {-0.5f, 0.5f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> vel_x = {0.0f, 0.0f};
        std::vector<float> vel_y = {0.5f, -0.5f}; // Circular orbit velocities
        std::vector<float> vel_z = {0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f};

        float G = 1.0f;
        auto force_func = SymplecticUtils::createGravitationalForce(G, 0.01f);
        auto potential_func = SymplecticUtils::createGravitationalPotential(G, 0.01f);

        integrator->setForceFunction(force_func);
        integrator->setPotentialFunction(potential_func);
        integrator->initializeConservationTracking(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);

        float initial_energy = integrator->getInitialQuantities().total_energy;

        // Simulate for several orbits
        for (int step = 0; step < 2000; ++step) {
            integrator->integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, params.time_step);
        }

        float final_energy = integrator->getCurrentQuantities().total_energy;
        float energy_drift = std::abs(final_energy - initial_energy) / std::abs(initial_energy);

        ASSERT_LT(energy_drift, 1e-2f); // Energy should be well conserved
        ASSERT_TRUE(PhysicsTestUtils::isSystemStable(pos_x, vel_x)); // System should remain stable
        ASSERT_TRUE(PhysicsTestUtils::isSystemStable(pos_y, vel_y));
    });

    // Test different integrator accuracy
    framework.runTest("Integrator Accuracy Comparison", []() {
        std::vector<SymplecticScheme> schemes = {
            SymplecticScheme::SYMPLECTIC_EULER,
            SymplecticScheme::VELOCITY_VERLET,
            SymplecticScheme::YOSHIDA4
        };

        std::vector<float> expected_accuracy = {1e-1f, 1e-3f, 1e-5f}; // Higher order = better accuracy

        for (size_t i = 0; i < schemes.size(); ++i) {
            SymplecticParams params;
            params.time_step = 0.01f;
            params.enable_energy_monitoring = true;

            auto integrator = SymplecticIntegratorFactory::create(schemes[i], params);

            // Simple harmonic oscillator
            std::vector<float> pos_x = {1.0f};
            std::vector<float> pos_y = {0.0f};
            std::vector<float> pos_z = {0.0f};
            std::vector<float> vel_x = {0.0f};
            std::vector<float> vel_y = {0.0f};
            std::vector<float> vel_z = {0.0f};
            std::vector<float> masses = {1.0f};

            auto force_func = SymplecticUtils::createHarmonicOscillatorForce(1.0f);
            auto potential_func = SymplecticUtils::createHarmonicOscillatorPotential(1.0f);

            integrator->setForceFunction(force_func);
            integrator->setPotentialFunction(potential_func);
            integrator->initializeConservationTracking(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);

            float initial_energy = integrator->getInitialQuantities().total_energy;

            // Simulate for one period
            float period = 2.0f * M_PI;
            float time = 0.0f;
            while (time < period) {
                integrator->integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, params.time_step, time);
                time += params.time_step;
            }

            float final_energy = integrator->getCurrentQuantities().total_energy;
            float energy_error = std::abs(final_energy - initial_energy) / initial_energy;

            // Higher order integrators should have better energy conservation
            ASSERT_LT(energy_error, expected_accuracy[i]);
        }
    });

    // Test factory pattern
    framework.runTest("Integrator Factory", []() {
        std::vector<SymplecticScheme> all_schemes = {
            SymplecticScheme::SYMPLECTIC_EULER,
            SymplecticScheme::VELOCITY_VERLET,
            SymplecticScheme::FOREST_RUTH,
            SymplecticScheme::YOSHIDA4,
            SymplecticScheme::BLANES_MOAN8
        };

        for (auto scheme : all_schemes) {
            auto integrator = SymplecticIntegratorFactory::create(scheme);
            ASSERT_TRUE(integrator != nullptr);

            std::string description = SymplecticIntegratorFactory::getSchemeDescription(scheme);
            ASSERT_TRUE(!description.empty());

            int order = SymplecticIntegratorFactory::getSchemeOrder(scheme);
            ASSERT_GT(order, 0);
        }
    });

    // Test parameter validation
    framework.runTest("Parameter Validation", []() {
        SymplecticParams params;
        params.time_step = -0.01f; // Invalid negative time step
        params.energy_tolerance = -1e-6f; // Invalid negative tolerance

        // Should handle invalid parameters gracefully
        ASSERT_NO_THROW({
            auto integrator = SymplecticIntegratorFactory::create(SymplecticScheme::VELOCITY_VERLET, params);
            integrator->setParameters(params);
        });
    });
}