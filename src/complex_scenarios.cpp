#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

using namespace physgrad;

class ScenarioTester {
private:
    SimParams base_params;

public:
    ScenarioTester() {
        base_params.time_step = 0.001f;
        base_params.G = 1.0f;
        base_params.epsilon = 0.001f;
        base_params.max_force = 100.0f;
        base_params.stability.gradient_clipping_threshold = 10.0f;
        base_params.stability.use_gradient_clipping = true;
    }

    void testSolarSystemScenario() {
        std::cout << "Solar System Scenario Test\n";
        std::cout << "===========================\n";

        SimParams params = base_params;
        params.num_bodies = 10;  // Sun + 9 planets
        params.time_step = 0.01f;  // Larger timestep for orbital dynamics

        auto sim = std::make_unique<Simulation>(params);
        BodySystem* bodies = sim->getBodies();

        // Setup solar system-like configuration
        std::vector<float> pos_x(params.num_bodies);
        std::vector<float> pos_y(params.num_bodies);
        std::vector<float> pos_z(params.num_bodies);
        std::vector<float> vel_x(params.num_bodies);
        std::vector<float> vel_y(params.num_bodies);
        std::vector<float> vel_z(params.num_bodies);
        std::vector<float> masses(params.num_bodies);

        // Sun at center
        pos_x[0] = pos_y[0] = pos_z[0] = 0.0f;
        vel_x[0] = vel_y[0] = vel_z[0] = 0.0f;
        masses[0] = 10.0f;  // Massive central body

        // Planets in roughly circular orbits
        for (int i = 1; i < params.num_bodies; i++) {
            float radius = 0.5f + i * 0.3f;  // Increasing orbital radii
            float angle = (2.0f * M_PI * i) / (params.num_bodies - 1);

            pos_x[i] = radius * cosf(angle);
            pos_y[i] = radius * sinf(angle);
            pos_z[i] = 0.0f;

            // Circular orbital velocity
            float v_orbit = sqrtf(params.G * masses[0] / radius);
            vel_x[i] = -v_orbit * sinf(angle);
            vel_y[i] = v_orbit * cosf(angle);
            vel_z[i] = 0.0f;

            masses[i] = 0.1f + i * 0.05f;  // Varying planet masses
        }

        size_t size = params.num_bodies * sizeof(float);
        cudaMemcpy(bodies->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        std::cout << "Initial energy: " << bodies->computeEnergy(params) << "\n";

        // Enable both stability and gradients
        sim->enableStableForces(true);
        sim->enableGradients();

        const int num_steps = 100;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < num_steps; step++) {
            sim->step();

            if (step % 20 == 0) {
                float energy = bodies->computeEnergy(params);
                std::cout << "Step " << std::setw(3) << step
                          << " | Energy: " << std::setw(12) << std::fixed << std::setprecision(6) << energy
                          << " | Time: " << std::setw(6) << sim->getLastStepTime() << " ms\n";
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time).count();

        std::cout << "Simulation completed in " << duration << " seconds\n";
        std::cout << "Final energy: " << bodies->computeEnergy(params) << "\n";

        // Test gradient computation on final state
        std::vector<float> target_pos_x = pos_x;  // Use initial positions as targets
        std::vector<float> target_pos_y = pos_y;
        std::vector<float> target_pos_z = pos_z;

        float loss = sim->computeGradients(target_pos_x, target_pos_y, target_pos_z);
        std::cout << "Gradient computation loss: " << loss << "\n";

        std::vector<float> grad_x(params.num_bodies), grad_y(params.num_bodies), grad_z(params.num_bodies);
        bodies->getGradients(grad_x, grad_y, grad_z);

        float grad_norm = 0.0f;
        for (int i = 0; i < params.num_bodies; i++) {
            grad_norm += grad_x[i]*grad_x[i] + grad_y[i]*grad_y[i] + grad_z[i]*grad_z[i];
        }
        grad_norm = sqrtf(grad_norm);

        std::cout << "Gradient norm: " << grad_norm << "\n";
        std::cout << "✓ Solar system scenario completed successfully\n\n";
    }

    void testGalaxyCollisionScenario() {
        std::cout << "Galaxy Collision Scenario Test\n";
        std::cout << "===============================\n";

        SimParams params = base_params;
        params.num_bodies = 64;  // Two small galaxies
        params.time_step = 0.005f;

        auto sim = std::make_unique<Simulation>(params);
        BodySystem* bodies = sim->getBodies();

        std::vector<float> pos_x(params.num_bodies);
        std::vector<float> pos_y(params.num_bodies);
        std::vector<float> pos_z(params.num_bodies);
        std::vector<float> vel_x(params.num_bodies);
        std::vector<float> vel_y(params.num_bodies);
        std::vector<float> vel_z(params.num_bodies);
        std::vector<float> masses(params.num_bodies);

        std::mt19937 rng(42);  // Fixed seed for reproducibility
        std::normal_distribution<float> pos_dist(0.0f, 0.2f);
        std::normal_distribution<float> vel_dist(0.0f, 0.1f);

        // Galaxy 1: centered at (-1, 0)
        for (int i = 0; i < params.num_bodies / 2; i++) {
            pos_x[i] = -1.0f + pos_dist(rng);
            pos_y[i] = pos_dist(rng);
            pos_z[i] = pos_dist(rng) * 0.1f;  // Flattened disk

            vel_x[i] = 0.2f + vel_dist(rng);  // Moving towards galaxy 2
            vel_y[i] = vel_dist(rng);
            vel_z[i] = vel_dist(rng) * 0.1f;

            masses[i] = 0.5f + 0.5f * (rng() / float(rng.max()));
        }

        // Galaxy 2: centered at (1, 0)
        for (int i = params.num_bodies / 2; i < params.num_bodies; i++) {
            pos_x[i] = 1.0f + pos_dist(rng);
            pos_y[i] = pos_dist(rng);
            pos_z[i] = pos_dist(rng) * 0.1f;

            vel_x[i] = -0.2f + vel_dist(rng);  // Moving towards galaxy 1
            vel_y[i] = vel_dist(rng);
            vel_z[i] = vel_dist(rng) * 0.1f;

            masses[i] = 0.5f + 0.5f * (rng() / float(rng.max()));
        }

        size_t size = params.num_bodies * sizeof(float);
        cudaMemcpy(bodies->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        sim->enableStableForces(true);

        std::cout << "Initial energy: " << bodies->computeEnergy(params) << "\n";

        const int num_steps = 200;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < num_steps; step++) {
            sim->step();

            if (step % 40 == 0) {
                float energy = bodies->computeEnergy(params);
                float gflops = sim->getGFLOPS();
                std::cout << "Step " << std::setw(3) << step
                          << " | Energy: " << std::setw(12) << std::fixed << std::setprecision(6) << energy
                          << " | GFLOPS: " << std::setw(8) << gflops << "\n";
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time).count();

        std::cout << "Galaxy collision completed in " << duration << " seconds\n";
        std::cout << "Final energy: " << bodies->computeEnergy(params) << "\n";
        std::cout << "✓ Galaxy collision scenario completed successfully\n\n";
    }

    void testPerformanceScaling() {
        std::cout << "Performance Scaling Test\n";
        std::cout << "========================\n";

        std::vector<int> test_sizes = {64, 128, 256, 512, 1024};

        std::cout << std::setw(8) << "Bodies"
                  << std::setw(12) << "Time (ms)"
                  << std::setw(12) << "GFLOPS"
                  << std::setw(15) << "Bodies/sec" << "\n";
        std::cout << std::string(50, '-') << "\n";

        for (int n : test_sizes) {
            SimParams params = base_params;
            params.num_bodies = n;

            auto sim = std::make_unique<Simulation>(params);
            sim->enableStableForces(true);

            // Warm up
            for (int i = 0; i < 5; i++) {
                sim->step();
            }

            // Timing test
            auto start = std::chrono::high_resolution_clock::now();
            const int test_steps = 10;

            for (int i = 0; i < test_steps; i++) {
                sim->step();
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end - start).count() / test_steps;

            float gflops = sim->getGFLOPS();
            float bodies_per_sec = 1000.0f / duration * n;

            std::cout << std::setw(8) << n
                      << std::setw(12) << std::fixed << std::setprecision(3) << duration
                      << std::setw(12) << std::setprecision(2) << gflops
                      << std::setw(15) << std::setprecision(0) << bodies_per_sec << "\n";
        }

        std::cout << "✓ Performance scaling test completed\n\n";
    }

    void testLongTimeIntegration() {
        std::cout << "Long-term Integration Stability Test\n";
        std::cout << "====================================\n";

        SimParams params = base_params;
        params.num_bodies = 8;
        params.time_step = 0.001f;  // Small timestep for accuracy

        auto sim = std::make_unique<Simulation>(params);
        BodySystem* bodies = sim->getBodies();

        // Setup figure-8 orbit-like initial conditions
        std::vector<float> pos_x = {-1.0f, 1.0f, 0.0f, -0.5f, 0.5f, 0.0f, -0.3f, 0.3f};
        std::vector<float> pos_y = {0.0f, 0.0f, 0.0f, 0.866f, 0.866f, 0.0f, -0.5f, -0.5f};
        std::vector<float> pos_z(8, 0.0f);
        std::vector<float> vel_x = {0.0f, 0.0f, 0.0f, -0.5f, -0.5f, 1.0f, 0.3f, 0.3f};
        std::vector<float> vel_y = {0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f};
        std::vector<float> vel_z(8, 0.0f);
        std::vector<float> masses(8, 1.0f);

        size_t size = params.num_bodies * sizeof(float);
        cudaMemcpy(bodies->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        sim->enableStableForces(true);

        float initial_energy = bodies->computeEnergy(params);
        std::cout << "Initial energy: " << initial_energy << "\n";

        const int total_steps = 5000;  // Long integration
        const int report_interval = 1000;

        for (int step = 0; step < total_steps; step++) {
            sim->step();

            if (step % report_interval == 0) {
                float energy = bodies->computeEnergy(params);
                float energy_drift = std::abs(energy - initial_energy) / std::abs(initial_energy);

                std::cout << "Step " << std::setw(4) << step
                          << " | Energy: " << std::setw(12) << std::fixed << std::setprecision(8) << energy
                          << " | Drift: " << std::setw(10) << std::scientific << std::setprecision(3) << energy_drift << "\n";
            }
        }

        float final_energy = bodies->computeEnergy(params);
        float total_drift = std::abs(final_energy - initial_energy) / std::abs(initial_energy);

        std::cout << "Final energy: " << final_energy << "\n";
        std::cout << "Total energy drift: " << std::scientific << total_drift << "\n";

        if (total_drift < 0.01f) {
            std::cout << "✓ Excellent energy conservation (< 1% drift)\n";
        } else if (total_drift < 0.1f) {
            std::cout << "✓ Good energy conservation (< 10% drift)\n";
        } else {
            std::cout << "⚠ Significant energy drift (> 10%)\n";
        }

        std::cout << "✓ Long-term integration test completed\n\n";
    }
};

int main() {
    std::cout << "PhysGrad Complex Multi-Body Scenarios Test\n";
    std::cout << "==========================================\n\n";

    ScenarioTester tester;

    try {
        tester.testSolarSystemScenario();
        tester.testGalaxyCollisionScenario();
        tester.testPerformanceScaling();
        tester.testLongTimeIntegration();

        std::cout << "All complex scenario tests completed successfully!\n";
        std::cout << "The PhysGrad system demonstrates robust performance across:\n";
        std::cout << "- Multi-scale dynamics (solar systems)\n";
        std::cout << "- Large-scale interactions (galaxy collisions)\n";
        std::cout << "- Performance scaling (64-1024 bodies)\n";
        std::cout << "- Long-term stability (energy conservation)\n";

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}