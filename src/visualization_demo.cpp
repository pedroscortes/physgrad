#include "simulation.h"
#include "visualization.h"
#include "optimizers.h"
#include "adaptive_integration.h"
#include "config_system.h"
#include "logging_system.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>

using namespace physgrad;

void computeGravitationalForces(
    const float* pos_x, const float* pos_y, const float* pos_z,
    float* force_x, float* force_y, float* force_z,
    const float* masses, int n, float G, float epsilon)
{
    for (int i = 0; i < n; ++i) {
        force_x[i] = force_y[i] = force_z[i] = 0.0f;

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            float dx = pos_x[j] - pos_x[i];
            float dy = pos_y[j] - pos_y[i];
            float dz = pos_z[j] - pos_z[i];
            float r_sq = dx*dx + dy*dy + dz*dz + epsilon*epsilon;
            float r = std::sqrt(r_sq);
            float force_mag = G * masses[i] * masses[j] / (r_sq * r);

            force_x[i] += force_mag * dx;
            force_y[i] += force_mag * dy;
            force_z[i] += force_mag * dz;
        }
    }
}

int main() {
    Logger::getInstance().info("demo", "Starting PhysGrad 3D Visualization Demo");

    ConfigManager config;
    config.loadFromFile("config.json");

    auto sim_params = config.getSimulationParams();

    VisualizationManager viz_manager;
    if (!viz_manager.initialize(1280, 720)) {
        Logger::getInstance().error("demo", "Failed to initialize visualization");
        return -1;
    }

    const int n_particles = 3;
    std::vector<float> pos_x = {0.0f, 1.0f, -1.0f};
    std::vector<float> pos_y = {0.0f, 0.0f, 0.0f};
    std::vector<float> pos_z = {0.0f, 0.0f, 0.0f};
    std::vector<float> vel_x = {0.0f, 0.0f, 0.0f};
    std::vector<float> vel_y = {0.0f, 0.5f, -0.5f};
    std::vector<float> vel_z = {0.0f, 0.0f, 0.0f};
    std::vector<float> masses = {1.0f, 0.5f, 0.5f};

    AdaptiveIntegrator integrator(IntegrationScheme::ADAPTIVE_RK45);

    std::vector<float> force_x(n_particles), force_y(n_particles), force_z(n_particles);

    Logger::getInstance().info("demo", "Starting main visualization loop");

    auto last_time = std::chrono::high_resolution_clock::now();
    float total_time = 0.0f;

    while (!viz_manager.shouldClose()) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto delta_time = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;

        if (viz_manager.isSimulationRunning()) {
            computeGravitationalForces(
                pos_x.data(), pos_y.data(), pos_z.data(),
                force_x.data(), force_y.data(), force_z.data(),
                masses.data(), n_particles, sim_params.G, sim_params.epsilon
            );

            float actual_dt = integrator.integrateStep(
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses,
                sim_params.time_step * viz_manager.getTimeScale(),
                sim_params.G, sim_params.epsilon
            );

            total_time += actual_dt;

            viz_manager.updateFromSimulation(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);
            viz_manager.updateForces(force_x, force_y, force_z);

            float kinetic_energy = 0.0f;
            for (int i = 0; i < n_particles; ++i) {
                float v_squared = vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i] + vel_z[i] * vel_z[i];
                kinetic_energy += 0.5f * masses[i] * v_squared;
            }

            float potential_energy = 0.0f;
            for (int i = 0; i < n_particles; ++i) {
                for (int j = i + 1; j < n_particles; ++j) {
                    float dx = pos_x[i] - pos_x[j];
                    float dy = pos_y[i] - pos_y[j];
                    float dz = pos_z[i] - pos_z[j];
                    float r = std::sqrt(dx*dx + dy*dy + dz*dz + sim_params.epsilon);
                    potential_energy -= sim_params.G * masses[i] * masses[j] / r;
                }
            }

            viz_manager.updateEnergy(kinetic_energy, potential_energy);
        }

        if (viz_manager.shouldSingleStep()) {
            computeGravitationalForces(
                pos_x.data(), pos_y.data(), pos_z.data(),
                force_x.data(), force_y.data(), force_z.data(),
                masses.data(), n_particles, sim_params.G, sim_params.epsilon
            );

            float actual_dt = integrator.integrateStep(
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses,
                sim_params.time_step, sim_params.G, sim_params.epsilon
            );

            viz_manager.updateFromSimulation(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);
            viz_manager.updateForces(force_x, force_y, force_z);
            viz_manager.resetSingleStep();
        }

        viz_manager.render();

        if (delta_time < 0.016f) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16 - (int)(delta_time * 1000)));
        }
    }

    viz_manager.shutdown();
    Logger::getInstance().info("demo", "Visualization demo completed");
    return 0;
}