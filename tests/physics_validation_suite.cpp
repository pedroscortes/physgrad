/**
 * PhysGrad Physics Validation Suite
 *
 * Comprehensive validation of physics correctness, conservation laws,
 * numerical stability, and performance characteristics.
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <memory>

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "src/concepts/physics_concepts.h"
    #include "src/concepts/type_traits.h"
    #include "src/concepts/concept_demo.h"
#endif

#include "src/common_types.h"
#include "src/physics_engine.h"

using namespace physgrad;

// =============================================================================
// VALIDATION FRAMEWORK
// =============================================================================

struct ValidationResult {
    std::string test_name;
    bool passed;
    double error;
    double performance_metric;
    std::string details;
};

class PhysicsValidator {
public:
    PhysicsValidator() {
        generator_.seed(42); // Reproducible results
    }

    void run_all_tests(const std::string& suite = "all", int duration_seconds = 60) {
        std::cout << "=== PhysGrad Physics Validation Suite ===\n";
        std::cout << "Test suite: " << suite << "\n";
        std::cout << "Duration: " << duration_seconds << " seconds\n\n";

        auto start_time = std::chrono::steady_clock::now();

        if (suite == "all" || suite == "conservation") {
            run_conservation_tests();
        }

        if (suite == "all" || suite == "stability") {
            run_stability_tests();
        }

        if (suite == "all" || suite == "accuracy") {
            run_accuracy_tests();
        }

        if (suite == "all" || suite == "performance") {
            run_performance_tests();
        }

        if (suite == "all" || suite == "integration") {
            run_integration_tests();
        }

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        generate_report(elapsed.count());
    }

private:
    std::vector<ValidationResult> results_;
    std::mt19937 generator_;

    // =============================================================================
    // CONSERVATION LAW TESTS
    // =============================================================================

    void run_conservation_tests() {
        std::cout << "Running conservation law tests...\n";

        test_energy_conservation();
        test_momentum_conservation();
        test_angular_momentum_conservation();
        test_symplectic_properties();
    }

    void test_energy_conservation() {
        std::cout << "  Testing energy conservation...\n";

        PhysicsEngine engine;
        if (!engine.initialize()) {
            record_result("Energy Conservation", false, 1.0, 0.0, "Engine initialization failed");
            return;
        }

        // Create oscillator system
        std::vector<float3> positions = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f}
        };
        std::vector<float3> velocities = {
            {0.0f, 1.0f, 0.0f},
            {0.0f, -1.0f, 0.0f}
        };
        std::vector<float> masses = {1.0f, 1.0f};

        engine.addParticles(positions, velocities, masses);

        float initial_energy = engine.calculateTotalEnergy();

        // Run simulation
        const int steps = 10000;
        const float dt = 0.001f;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < steps; ++i) {
            engine.step(dt);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        float final_energy = engine.calculateTotalEnergy();
        double energy_error = std::abs(final_energy - initial_energy) / initial_energy;
        double performance = static_cast<double>(steps) / (duration.count() * 1e-6);

        bool passed = energy_error < 1e-6;

        record_result("Energy Conservation", passed, energy_error, performance,
                     "Initial: " + std::to_string(initial_energy) +
                     ", Final: " + std::to_string(final_energy));
    }

    void test_momentum_conservation() {
        std::cout << "  Testing momentum conservation...\n";

        PhysicsEngine engine;
        engine.initialize();

        // Create collision system
        std::vector<float3> positions = {
            {-1.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f}
        };
        std::vector<float3> velocities = {
            {1.0f, 0.0f, 0.0f},
            {-1.0f, 0.0f, 0.0f}
        };
        std::vector<float> masses = {1.0f, 1.0f};

        engine.addParticles(positions, velocities, masses);

        // Calculate initial momentum
        auto initial_positions = engine.getPositions();
        auto initial_velocities = engine.getVelocities();

        float3 initial_momentum = {0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < masses.size(); ++i) {
            initial_momentum.x += masses[i] * initial_velocities[i].x;
            initial_momentum.y += masses[i] * initial_velocities[i].y;
            initial_momentum.z += masses[i] * initial_velocities[i].z;
        }

        // Run simulation
        const int steps = 5000;
        const float dt = 0.001f;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < steps; ++i) {
            engine.step(dt);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Calculate final momentum
        auto final_velocities = engine.getVelocities();

        float3 final_momentum = {0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < masses.size(); ++i) {
            final_momentum.x += masses[i] * final_velocities[i].x;
            final_momentum.y += masses[i] * final_velocities[i].y;
            final_momentum.z += masses[i] * final_velocities[i].z;
        }

        double momentum_error = std::sqrt(
            (final_momentum.x - initial_momentum.x) * (final_momentum.x - initial_momentum.x) +
            (final_momentum.y - initial_momentum.y) * (final_momentum.y - initial_momentum.y) +
            (final_momentum.z - initial_momentum.z) * (final_momentum.z - initial_momentum.z)
        );

        double performance = static_cast<double>(steps) / (duration.count() * 1e-6);
        bool passed = momentum_error < 1e-10;

        record_result("Momentum Conservation", passed, momentum_error, performance,
                     "Momentum drift: " + std::to_string(momentum_error));
    }

    void test_angular_momentum_conservation() {
        std::cout << "  Testing angular momentum conservation...\n";

        PhysicsEngine engine;
        engine.initialize();

        // Create rotating system
        std::vector<float3> positions = {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {-1.0f, 0.0f, 0.0f},
            {0.0f, -1.0f, 0.0f}
        };
        std::vector<float3> velocities = {
            {0.0f, 1.0f, 0.0f},
            {-1.0f, 0.0f, 0.0f},
            {0.0f, -1.0f, 0.0f},
            {1.0f, 0.0f, 0.0f}
        };
        std::vector<float> masses = {1.0f, 1.0f, 1.0f, 1.0f};

        engine.addParticles(positions, velocities, masses);

        // Calculate initial angular momentum
        auto initial_positions = engine.getPositions();
        auto initial_velocities = engine.getVelocities();

        float3 initial_L = calculate_angular_momentum(initial_positions, initial_velocities, masses);

        // Run simulation
        const int steps = 3000;
        const float dt = 0.001f;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < steps; ++i) {
            engine.step(dt);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Calculate final angular momentum
        auto final_positions = engine.getPositions();
        auto final_velocities = engine.getVelocities();

        float3 final_L = calculate_angular_momentum(final_positions, final_velocities, masses);

        double L_error = magnitude(make_float3(
            final_L.x - initial_L.x,
            final_L.y - initial_L.y,
            final_L.z - initial_L.z
        ));

        double performance = static_cast<double>(steps) / (duration.count() * 1e-6);
        bool passed = L_error < 1e-8;

        record_result("Angular Momentum Conservation", passed, L_error, performance,
                     "Angular momentum drift: " + std::to_string(L_error));
    }

    void test_symplectic_properties() {
        std::cout << "  Testing symplectic integrator properties...\n";

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
        // Test using concept-aware integrator
        concept_demo::ConceptVerletIntegrator<float> integrator;
        concept_demo::ConceptParticle<float> particle(
            concept_demo::SimpleVector3D<float>{1.0f, 0.0f, 0.0f},
            concept_demo::SimpleVector3D<float>{0.0f, 1.0f, 0.0f},
            1.0f
        );

        auto start = std::chrono::high_resolution_clock::now();

        // Test long-term stability (10K steps)
        auto original_particle = particle;
        for (int i = 0; i < 10000; ++i) {
            particle = integrator.step(particle, 0.001f);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Reverse integration to test time reversibility
        for (int i = 0; i < 10000; ++i) {
            particle = integrator.step(particle, -0.001f);
        }

        double reversibility_error = std::sqrt(
            (particle.position()[0] - original_particle.position()[0]) *
            (particle.position()[0] - original_particle.position()[0]) +
            (particle.position()[1] - original_particle.position()[1]) *
            (particle.position()[1] - original_particle.position()[1]) +
            (particle.position()[2] - original_particle.position()[2]) *
            (particle.position()[2] - original_particle.position()[2])
        );

        double performance = 20000.0 / (duration.count() * 1e-6);
        bool passed = reversibility_error < 1e-10;

        record_result("Symplectic Time Reversibility", passed, reversibility_error, performance,
                     "Reversibility error: " + std::to_string(reversibility_error));
#else
        record_result("Symplectic Time Reversibility", false, 1.0, 0.0, "Concepts not available");
#endif
    }

    // =============================================================================
    // NUMERICAL STABILITY TESTS
    // =============================================================================

    void run_stability_tests() {
        std::cout << "Running numerical stability tests...\n";

        test_long_term_stability();
        test_large_timestep_stability();
        test_extreme_mass_ratios();
        test_stiff_system_stability();
    }

    void test_long_term_stability() {
        std::cout << "  Testing long-term numerical stability...\n";

        PhysicsEngine engine;
        engine.initialize();

        // Create planetary-like system
        std::vector<float3> positions = {
            {0.0f, 0.0f, 0.0f},    // Central mass
            {1.0f, 0.0f, 0.0f}     // Orbiting mass
        };
        std::vector<float3> velocities = {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}     // Circular orbit velocity
        };
        std::vector<float> masses = {1000.0f, 1.0f};

        engine.addParticles(positions, velocities, masses);

        float initial_energy = engine.calculateTotalEnergy();

        // Long simulation (1M steps)
        const int steps = 1000000;
        const float dt = 0.0001f;

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<float> energy_history;
        energy_history.reserve(steps / 1000);

        for (int i = 0; i < steps; ++i) {
            engine.step(dt);

            if (i % 1000 == 0) {
                energy_history.push_back(engine.calculateTotalEnergy());
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        // Analyze energy drift
        float max_energy_deviation = 0.0f;
        for (float energy : energy_history) {
            float deviation = std::abs(energy - initial_energy) / initial_energy;
            max_energy_deviation = std::max(max_energy_deviation, deviation);
        }

        double performance = static_cast<double>(steps) / duration.count();
        bool passed = max_energy_deviation < 1e-4; // Allow small drift for long simulations

        record_result("Long-term Stability", passed, max_energy_deviation, performance,
                     "Max energy deviation: " + std::to_string(max_energy_deviation) +
                     " over " + std::to_string(steps) + " steps");
    }

    void test_large_timestep_stability() {
        std::cout << "  Testing large timestep stability...\n";

        PhysicsEngine engine;
        engine.initialize();

        std::vector<float3> positions = {{0.0f, 0.0f, 0.0f}};
        std::vector<float3> velocities = {{1.0f, 0.0f, 0.0f}};
        std::vector<float> masses = {1.0f};

        engine.addParticles(positions, velocities, masses);

        // Test various timestep sizes
        std::vector<float> timesteps = {0.1f, 0.05f, 0.01f, 0.005f};
        std::vector<double> errors;

        for (float dt : timesteps) {
            PhysicsEngine test_engine;
            test_engine.initialize();
            test_engine.addParticles(positions, velocities, masses);

            float initial_energy = test_engine.calculateTotalEnergy();

            // Run for fixed simulation time
            int steps = static_cast<int>(1.0f / dt); // 1 second simulation time

            for (int i = 0; i < steps; ++i) {
                test_engine.step(dt);
            }

            float final_energy = test_engine.calculateTotalEnergy();
            double error = std::abs(final_energy - initial_energy) / initial_energy;
            errors.push_back(error);
        }

        // Check convergence order
        double convergence_rate = 0.0;
        for (size_t i = 1; i < errors.size(); ++i) {
            double ratio = errors[i-1] / errors[i];
            double step_ratio = timesteps[i-1] / timesteps[i];
            convergence_rate += std::log(ratio) / std::log(step_ratio);
        }
        convergence_rate /= (errors.size() - 1);

        bool passed = convergence_rate > 1.5; // Should be close to 2 for Verlet

        record_result("Timestep Stability", passed, errors.back(), convergence_rate,
                     "Convergence rate: " + std::to_string(convergence_rate));
    }

    void test_extreme_mass_ratios() {
        std::cout << "  Testing extreme mass ratios...\n";

        PhysicsEngine engine;
        engine.initialize();

        // Light particle orbiting heavy particle
        std::vector<float3> positions = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f}
        };
        std::vector<float3> velocities = {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 100.0f, 0.0f}  // High velocity for stability
        };
        std::vector<float> masses = {1e6f, 1e-6f}; // Mass ratio of 10^12

        engine.addParticles(positions, velocities, masses);

        float initial_energy = engine.calculateTotalEnergy();

        auto start = std::chrono::high_resolution_clock::now();

        const int steps = 10000;
        const float dt = 1e-5f; // Small timestep for stability

        bool numerical_overflow = false;
        for (int i = 0; i < steps; ++i) {
            engine.step(dt);

            // Check for numerical overflow
            auto positions = engine.getPositions();
            for (const auto& pos : positions) {
                if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z)) {
                    numerical_overflow = true;
                    break;
                }
            }

            if (numerical_overflow) break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        float final_energy = engine.calculateTotalEnergy();
        double energy_error = std::abs(final_energy - initial_energy) / std::abs(initial_energy);
        double performance = static_cast<double>(steps) / (duration.count() * 1e-6);

        bool passed = !numerical_overflow && energy_error < 1e-3;

        record_result("Extreme Mass Ratios", passed, energy_error, performance,
                     "Overflow: " + std::string(numerical_overflow ? "YES" : "NO") +
                     ", Energy error: " + std::to_string(energy_error));
    }

    void test_stiff_system_stability() {
        std::cout << "  Testing stiff system stability...\n";

        // Test with high-frequency oscillator
        PhysicsEngine engine;
        engine.initialize();

        std::vector<float3> positions = {{1.0f, 0.0f, 0.0f}};
        std::vector<float3> velocities = {{0.0f, 0.0f, 0.0f}};
        std::vector<float> masses = {1.0f};

        engine.addParticles(positions, velocities, masses);

        float initial_energy = engine.calculateTotalEnergy();

        // Use small timestep for stiff system
        const int steps = 50000;
        const float dt = 1e-5f;

        auto start = std::chrono::high_resolution_clock::now();

        bool stable = true;
        for (int i = 0; i < steps; ++i) {
            engine.step(dt);

            // Check for instability (energy blowup)
            float current_energy = engine.calculateTotalEnergy();
            if (current_energy > 10.0f * std::abs(initial_energy) || !std::isfinite(current_energy)) {
                stable = false;
                break;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        float final_energy = engine.calculateTotalEnergy();
        double energy_error = std::abs(final_energy - initial_energy) / std::abs(initial_energy);
        double performance = static_cast<double>(steps) / (duration.count() * 1e-6);

        bool passed = stable && energy_error < 1e-2;

        record_result("Stiff System Stability", passed, energy_error, performance,
                     "Stable: " + std::string(stable ? "YES" : "NO"));
    }

    // =============================================================================
    // ACCURACY TESTS
    // =============================================================================

    void run_accuracy_tests() {
        std::cout << "Running accuracy tests...\n";

        test_harmonic_oscillator_accuracy();
        test_planetary_motion_accuracy();
        test_collision_accuracy();
    }

    void test_harmonic_oscillator_accuracy() {
        std::cout << "  Testing harmonic oscillator accuracy...\n";

        PhysicsEngine engine;
        engine.initialize();

        // Simple harmonic oscillator
        std::vector<float3> positions = {{1.0f, 0.0f, 0.0f}};
        std::vector<float3> velocities = {{0.0f, 0.0f, 0.0f}};
        std::vector<float> masses = {1.0f};

        engine.addParticles(positions, velocities, masses);

        const float omega = 1.0f; // Angular frequency
        const float dt = 0.01f;
        const int steps = static_cast<int>(2.0f * M_PI / (omega * dt)); // One period

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < steps; ++i) {
            engine.step(dt);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        auto final_positions = engine.getPositions();
        auto final_velocities = engine.getVelocities();

        // Analytical solution after one period should return to initial state
        double position_error = std::abs(final_positions[0].x - 1.0f);
        double velocity_error = std::abs(final_velocities[0].x);

        double total_error = std::sqrt(position_error * position_error + velocity_error * velocity_error);
        double performance = static_cast<double>(steps) / (duration.count() * 1e-6);

        bool passed = total_error < 1e-3;

        record_result("Harmonic Oscillator Accuracy", passed, total_error, performance,
                     "Position error: " + std::to_string(position_error) +
                     ", Velocity error: " + std::to_string(velocity_error));
    }

    void test_planetary_motion_accuracy() {
        std::cout << "  Testing planetary motion accuracy...\n";

        PhysicsEngine engine;
        engine.initialize();

        // Earth-Sun system (simplified)
        std::vector<float3> positions = {
            {0.0f, 0.0f, 0.0f},     // Sun
            {149.6e6f, 0.0f, 0.0f}  // Earth (km)
        };
        std::vector<float3> velocities = {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 29780.0f, 0.0f}  // Earth orbital velocity (m/s)
        };
        std::vector<float> masses = {1.989e30f, 5.972e24f}; // kg

        engine.addParticles(positions, velocities, masses);

        // Simulate one year
        const float dt = 3600.0f; // 1 hour timestep
        const int steps = 8760;   // Hours in a year

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<float3> orbit_positions;
        orbit_positions.reserve(steps / 24); // Daily positions

        for (int i = 0; i < steps; ++i) {
            engine.step(dt);

            if (i % 24 == 0) { // Record daily
                auto pos = engine.getPositions();
                orbit_positions.push_back(pos[1]);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

        // Check orbital eccentricity (should be close to Earth's ~0.017)
        float min_distance = 1e20f, max_distance = 0.0f;
        for (const auto& pos : orbit_positions) {
            float distance = magnitude(pos);
            min_distance = std::min(min_distance, distance);
            max_distance = std::max(max_distance, distance);
        }

        float eccentricity = (max_distance - min_distance) / (max_distance + min_distance);
        double eccentricity_error = std::abs(eccentricity - 0.017f);

        double performance = static_cast<double>(steps) / duration.count();
        bool passed = eccentricity_error < 0.01; // Within 1% of actual eccentricity

        record_result("Planetary Motion Accuracy", passed, eccentricity_error, performance,
                     "Computed eccentricity: " + std::to_string(eccentricity) +
                     ", Expected: 0.017");
    }

    void test_collision_accuracy() {
        std::cout << "  Testing collision accuracy...\n";

        PhysicsEngine engine;
        engine.initialize();

        // Head-on elastic collision
        std::vector<float3> positions = {
            {-1.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f}
        };
        std::vector<float3> velocities = {
            {1.0f, 0.0f, 0.0f},
            {-1.0f, 0.0f, 0.0f}
        };
        std::vector<float> masses = {1.0f, 1.0f};

        engine.addParticles(positions, velocities, masses);

        const int steps = 2000;
        const float dt = 0.001f;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < steps; ++i) {
            engine.step(dt);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        auto final_velocities = engine.getVelocities();

        // After elastic collision, velocities should be exchanged
        double velocity_error = std::sqrt(
            (final_velocities[0].x - (-1.0f)) * (final_velocities[0].x - (-1.0f)) +
            (final_velocities[1].x - 1.0f) * (final_velocities[1].x - 1.0f)
        );

        double performance = static_cast<double>(steps) / (duration.count() * 1e-6);
        bool passed = velocity_error < 0.1;

        record_result("Collision Accuracy", passed, velocity_error, performance,
                     "Velocity exchange error: " + std::to_string(velocity_error));
    }

    // =============================================================================
    // PERFORMANCE TESTS
    // =============================================================================

    void run_performance_tests() {
        std::cout << "Running performance tests...\n";

        test_scaling_performance();
        test_memory_bandwidth();
        test_force_computation_performance();
    }

    void test_scaling_performance() {
        std::cout << "  Testing scaling performance...\n";

        std::vector<int> particle_counts = {1000, 5000, 10000, 50000};
        std::vector<double> performance_metrics;

        for (int count : particle_counts) {
            PhysicsEngine engine;
            engine.initialize();

            // Generate random particles
            std::vector<float3> positions, velocities;
            std::vector<float> masses;

            positions.reserve(count);
            velocities.reserve(count);
            masses.reserve(count);

            std::uniform_real_distribution<float> pos_dist(-10.0f, 10.0f);
            std::uniform_real_distribution<float> vel_dist(-1.0f, 1.0f);
            std::uniform_real_distribution<float> mass_dist(0.5f, 2.0f);

            for (int i = 0; i < count; ++i) {
                positions.push_back({pos_dist(generator_), pos_dist(generator_), pos_dist(generator_)});
                velocities.push_back({vel_dist(generator_), vel_dist(generator_), vel_dist(generator_)});
                masses.push_back(mass_dist(generator_));
            }

            engine.addParticles(positions, velocities, masses);

            const int steps = 100;
            const float dt = 0.001f;

            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < steps; ++i) {
                engine.step(dt);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            double particles_per_second = static_cast<double>(count * steps) / (duration.count() * 1e-6);
            performance_metrics.push_back(particles_per_second);

            std::cout << "    " << count << " particles: "
                      << std::fixed << std::setprecision(0) << particles_per_second
                      << " particles/second\n";
        }

        // Check for reasonable scaling (not necessarily linear due to overhead)
        double scaling_efficiency = performance_metrics.back() / performance_metrics.front();
        double ideal_scaling = static_cast<double>(particle_counts.back()) / particle_counts.front();
        double efficiency_ratio = scaling_efficiency / ideal_scaling;

        bool passed = efficiency_ratio > 0.1; // Should scale somewhat reasonably

        record_result("Scaling Performance", passed, 1.0 - efficiency_ratio,
                     performance_metrics.back(),
                     "Efficiency ratio: " + std::to_string(efficiency_ratio));
    }

    void test_memory_bandwidth() {
        std::cout << "  Testing memory bandwidth utilization...\n";

        PhysicsEngine engine;
        engine.initialize();

        const int particle_count = 100000;

        // Generate particles
        std::vector<float3> positions, velocities;
        std::vector<float> masses;

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < particle_count; ++i) {
            positions.push_back({dist(generator_), dist(generator_), dist(generator_)});
            velocities.push_back({dist(generator_), dist(generator_), dist(generator_)});
            masses.push_back(1.0f);
        }

        engine.addParticles(positions, velocities, masses);

        const int steps = 10;
        const float dt = 0.001f;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < steps; ++i) {
            engine.step(dt);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Estimate memory bandwidth
        size_t bytes_per_particle = sizeof(float3) * 2 + sizeof(float); // position, velocity, mass
        size_t total_bytes = bytes_per_particle * particle_count * steps * 2; // Read + write
        double bandwidth_gbps = static_cast<double>(total_bytes) / (duration.count() * 1e-3); // GB/s

        double performance = static_cast<double>(particle_count * steps) / (duration.count() * 1e-6);
        bool passed = bandwidth_gbps > 1.0; // Should achieve at least 1 GB/s

        record_result("Memory Bandwidth", passed, 0.0, bandwidth_gbps,
                     "Bandwidth: " + std::to_string(bandwidth_gbps) + " GB/s");
    }

    void test_force_computation_performance() {
        std::cout << "  Testing force computation performance...\n";

        PhysicsEngine engine;
        engine.initialize();

        const int particle_count = 10000;

        std::vector<float3> positions, velocities;
        std::vector<float> masses;

        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

        for (int i = 0; i < particle_count; ++i) {
            positions.push_back({dist(generator_), dist(generator_), dist(generator_)});
            velocities.push_back({0.0f, 0.0f, 0.0f});
            masses.push_back(1.0f);
        }

        engine.addParticles(positions, velocities, masses);

        const int force_computations = 1000;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < force_computations; ++i) {
            engine.updateForces();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double interactions_per_second = static_cast<double>(particle_count * particle_count * force_computations) / (duration.count() * 1e-6);
        bool passed = interactions_per_second > 1e8; // Should handle 100M interactions/second

        record_result("Force Computation Performance", passed, 0.0, interactions_per_second,
                     "Interactions/sec: " + std::to_string(interactions_per_second));
    }

    // =============================================================================
    // INTEGRATION TESTS
    // =============================================================================

    void run_integration_tests() {
        std::cout << "Running integration tests...\n";

        test_multi_physics_coupling();
        test_boundary_conditions();
        test_restart_capability();
    }

    void test_multi_physics_coupling() {
        std::cout << "  Testing multi-physics coupling...\n";

        // Create system with multiple physics domains
        PhysicsEngine engine;
        engine.initialize();

        // Particles with different physics (gravitational + electromagnetic)
        std::vector<float3> positions = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}
        };
        std::vector<float3> velocities = {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.1f, 0.0f},
            {-0.1f, 0.0f, 0.0f}
        };
        std::vector<float> masses = {10.0f, 1.0f, 1.0f};

        engine.addParticles(positions, velocities, masses);

        // Set charges for electromagnetic interaction
        std::vector<float> charges = {0.0f, 1.0f, -1.0f};
        engine.setCharges(charges);

        float initial_energy = engine.calculateTotalEnergy();

        auto start = std::chrono::high_resolution_clock::now();

        const int steps = 5000;
        const float dt = 0.001f;

        for (int i = 0; i < steps; ++i) {
            engine.step(dt);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        float final_energy = engine.calculateTotalEnergy();
        double energy_error = std::abs(final_energy - initial_energy) / std::abs(initial_energy);
        double performance = static_cast<double>(steps) / (duration.count() * 1e-6);

        bool passed = energy_error < 1e-2; // Allow some drift due to complex interactions

        record_result("Multi-physics Coupling", passed, energy_error, performance,
                     "Energy conservation in coupled system");
    }

    void test_boundary_conditions() {
        std::cout << "  Testing boundary conditions...\n";

        PhysicsEngine engine;
        engine.initialize();

        // Set periodic boundary conditions
        engine.setBoundaryConditions(BoundaryType::PERIODIC, {10.0f, 10.0f, 10.0f});

        // Particle near boundary
        std::vector<float3> positions = {{9.9f, 5.0f, 5.0f}};
        std::vector<float3> velocities = {{1.0f, 0.0f, 0.0f}};
        std::vector<float> masses = {1.0f};

        engine.addParticles(positions, velocities, masses);

        auto start = std::chrono::high_resolution_clock::now();

        const int steps = 200;
        const float dt = 0.01f;

        for (int i = 0; i < steps; ++i) {
            engine.step(dt);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        auto final_positions = engine.getPositions();

        // Particle should have wrapped around to other side
        bool wrapped = final_positions[0].x < 5.0f;
        double performance = static_cast<double>(steps) / (duration.count() * 1e-6);

        record_result("Boundary Conditions", wrapped, 0.0, performance,
                     "Periodic boundary wrapping: " + std::string(wrapped ? "YES" : "NO"));
    }

    void test_restart_capability() {
        std::cout << "  Testing restart capability...\n";

        // First simulation
        PhysicsEngine engine1;
        engine1.initialize();

        std::vector<float3> positions = {{1.0f, 0.0f, 0.0f}};
        std::vector<float3> velocities = {{0.0f, 1.0f, 0.0f}};
        std::vector<float> masses = {1.0f};

        engine1.addParticles(positions, velocities, masses);

        const int steps1 = 1000;
        const float dt = 0.001f;

        for (int i = 0; i < steps1; ++i) {
            engine1.step(dt);
        }

        auto mid_positions = engine1.getPositions();
        auto mid_velocities = engine1.getVelocities();

        // Second simulation (restart from midpoint)
        PhysicsEngine engine2;
        engine2.initialize();
        engine2.addParticles(mid_positions, mid_velocities, masses);

        const int steps2 = 1000;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < steps2; ++i) {
            engine2.step(dt);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Continuous simulation for comparison
        for (int i = 0; i < steps2; ++i) {
            engine1.step(dt);
        }

        auto final_pos1 = engine1.getPositions();
        auto final_pos2 = engine2.getPositions();

        double restart_error = magnitude(make_float3(
            final_pos1[0].x - final_pos2[0].x,
            final_pos1[0].y - final_pos2[0].y,
            final_pos1[0].z - final_pos2[0].z
        ));

        double performance = static_cast<double>(steps2) / (duration.count() * 1e-6);
        bool passed = restart_error < 1e-10;

        record_result("Restart Capability", passed, restart_error, performance,
                     "Restart error: " + std::to_string(restart_error));
    }

    // =============================================================================
    // UTILITY FUNCTIONS
    // =============================================================================

    float3 calculate_angular_momentum(const std::vector<float3>& positions,
                                     const std::vector<float3>& velocities,
                                     const std::vector<float>& masses) {
        float3 L = {0.0f, 0.0f, 0.0f};

        for (size_t i = 0; i < positions.size(); ++i) {
            float3 r = positions[i];
            float3 v = velocities[i];
            float m = masses[i];

            // L = r × (mv)
            float3 momentum = {m * v.x, m * v.y, m * v.z};
            float3 angular_momentum = cross(r, momentum);

            L.x += angular_momentum.x;
            L.y += angular_momentum.y;
            L.z += angular_momentum.z;
        }

        return L;
    }

    void record_result(const std::string& name, bool passed, double error,
                      double performance, const std::string& details) {
        results_.push_back({name, passed, error, performance, details});

        std::cout << "    " << name << ": "
                  << (passed ? "PASS" : "FAIL")
                  << " (error: " << std::scientific << error
                  << ", perf: " << std::fixed << performance << ")\n";
    }

    void generate_report(int elapsed_seconds) {
        std::cout << "\n=== VALIDATION SUMMARY ===\n";

        int passed = 0, failed = 0;
        double total_error = 0.0;

        for (const auto& result : results_) {
            if (result.passed) {
                passed++;
            } else {
                failed++;
            }
            total_error += result.error;
        }

        std::cout << "Total tests: " << results_.size() << "\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";
        std::cout << "Success rate: " << std::fixed << std::setprecision(1)
                  << (100.0 * passed / results_.size()) << "%\n";
        std::cout << "Total runtime: " << elapsed_seconds << " seconds\n\n";

        // Generate JSON report
        std::ofstream json_file("validation_results.json");
        json_file << "{\n";
        json_file << "  \"summary\": {\n";
        json_file << "    \"total_tests\": " << results_.size() << ",\n";
        json_file << "    \"passed\": " << passed << ",\n";
        json_file << "    \"failed\": " << failed << ",\n";
        json_file << "    \"success_rate\": " << (100.0 * passed / results_.size()) << ",\n";
        json_file << "    \"runtime_seconds\": " << elapsed_seconds << "\n";
        json_file << "  },\n";
        json_file << "  \"tests\": [\n";

        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& result = results_[i];
            json_file << "    {\n";
            json_file << "      \"name\": \"" << result.test_name << "\",\n";
            json_file << "      \"passed\": " << (result.passed ? "true" : "false") << ",\n";
            json_file << "      \"error\": " << result.error << ",\n";
            json_file << "      \"performance\": " << result.performance_metric << ",\n";
            json_file << "      \"details\": \"" << result.details << "\"\n";
            json_file << "    }" << (i < results_.size() - 1 ? "," : "") << "\n";
        }

        json_file << "  ]\n";
        json_file << "}\n";
        json_file.close();

        std::cout << "Detailed results saved to validation_results.json\n";
    }
};

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main(int argc, char** argv) {
    std::string test_suite = "all";
    int duration = 60;
    int timesteps = 1000000;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test-suite" && i + 1 < argc) {
            test_suite = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            duration = std::stoi(argv[++i]);
        } else if (arg == "--timesteps" && i + 1 < argc) {
            timesteps = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --test-suite <suite>  Run specific test suite (all, conservation, stability, accuracy, performance)\n";
            std::cout << "  --duration <seconds>  Maximum runtime for tests\n";
            std::cout << "  --timesteps <n>       Number of timesteps for stability tests\n";
            std::cout << "  --help                Show this help message\n";
            return 0;
        }
    }

    PhysicsValidator validator;
    validator.run_all_tests(test_suite, duration);

    return 0;
}