#include "collision_detection.h"
#include "visualization.h"
#include "adaptive_integration.h"
#include "config_system.h"
#include "logging_system.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <random>

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
    Logger::getInstance().info("demo", "Starting PhysGrad Collision Detection Demo");

    // Load configuration from proper config file
    ConfigManager config;
    if (!config.loadFromFile("config.conf")) {
        Logger::getInstance().warning("demo", "Could not load config.conf, using defaults");
    }
    auto sim_params = config.getSimulationParams();

    VisualizationManager viz_manager;
    if (!viz_manager.initialize(1280, 720)) {
        Logger::getInstance().error("demo", "Failed to initialize visualization");
        return -1;
    }

    // Setup collision detection
    CollisionParams collision_params;
    collision_params.contact_threshold = 0.05f;
    collision_params.contact_stiffness = 500.0f;
    collision_params.contact_damping = 5.0f;
    collision_params.global_restitution = 0.7f;
    collision_params.global_friction = 0.2f;

    CollisionDetector collision_detector(collision_params);

    // Create scenario: bouncing balls in a box
    const int n_particles = 8;
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> vel_x, vel_y, vel_z;
    std::vector<float> masses;

    // Random number generator for initial conditions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> vel_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> mass_dist(0.5f, 2.0f);

    // Initialize particles
    for (int i = 0; i < n_particles; ++i) {
        pos_x.push_back(pos_dist(gen));
        pos_y.push_back(pos_dist(gen) + 3.0f);  // Start above ground
        pos_z.push_back(pos_dist(gen));

        vel_x.push_back(vel_dist(gen));
        vel_y.push_back(vel_dist(gen));
        vel_z.push_back(vel_dist(gen));

        masses.push_back(mass_dist(gen));
    }

    // Add ground plane particles (large masses, zero velocity)
    const float ground_y = -3.0f;
    const int ground_particles = 25;
    for (int x = -2; x <= 2; ++x) {
        for (int z = -2; z <= 2; ++z) {
            pos_x.push_back(static_cast<float>(x));
            pos_y.push_back(ground_y);
            pos_z.push_back(static_cast<float>(z));

            vel_x.push_back(0.0f);
            vel_y.push_back(0.0f);
            vel_z.push_back(0.0f);

            masses.push_back(10.0f);  // Heavy ground particles
        }
    }

    const int total_particles = static_cast<int>(pos_x.size());

    // Update collision detector with particle radii
    collision_detector.updateBodyRadiiFromMasses(masses, 1.0f);

    AdaptiveIntegrator integrator(IntegrationScheme::ADAPTIVE_RK45);

    std::vector<float> force_x(total_particles), force_y(total_particles), force_z(total_particles);

    Logger::getInstance().info("demo", "Starting collision simulation with " + std::to_string(total_particles) + " particles");

    auto last_time = std::chrono::high_resolution_clock::now();
    float total_time = 0.0f;

    while (!viz_manager.shouldClose()) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto delta_time = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;

        if (viz_manager.isSimulationRunning()) {
            // Reset forces
            std::fill(force_x.begin(), force_x.end(), 0.0f);
            std::fill(force_y.begin(), force_y.end(), 0.0f);
            std::fill(force_z.begin(), force_z.end(), 0.0f);

            // Apply gravity
            for (int i = 0; i < n_particles; ++i) {  // Only to non-ground particles
                force_y[i] -= masses[i] * 9.8f;  // Gravity
            }

            // Apply gravitational forces between particles (optional, weaker)
            /*
            computeGravitationalForces(
                pos_x.data(), pos_y.data(), pos_z.data(),
                force_x.data(), force_y.data(), force_z.data(),
                masses.data(), total_particles, 0.1f, 0.01f  // Weak gravity
            );
            */

            // Apply collision forces
            collision_detector.applyCollisionForces(
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
                force_x, force_y, force_z, masses
            );

            // Integrate motion
            float actual_dt = integrator.integrateStep(
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses,
                sim_params.time_step * viz_manager.getTimeScale() * 0.1f,  // Slower for better visualization
                0.0f, 0.001f  // No gravitational constant for integration
            );

            // Resolve any remaining collisions with impulse-based method
            collision_detector.resolveCollisions(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);

            // Add some damping to non-ground particles to prevent infinite bouncing
            for (int i = 0; i < n_particles; ++i) {
                vel_x[i] *= 0.999f;
                vel_y[i] *= 0.999f;
                vel_z[i] *= 0.999f;
            }

            total_time += actual_dt;

            // Calculate energy for only the moving particles
            float kinetic_energy = 0.0f;
            for (int i = 0; i < n_particles; ++i) {
                float v_squared = vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i] + vel_z[i] * vel_z[i];
                kinetic_energy += 0.5f * masses[i] * v_squared;
            }

            // Update visualization
            viz_manager.updateFromSimulation(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);
            viz_manager.updateForces(force_x, force_y, force_z);
            viz_manager.updateEnergy(kinetic_energy, 0.0f);
            viz_manager.setCollisionStats(
                collision_detector.getBroadPhasePairs(),
                collision_detector.getNarrowPhaseTests(),
                collision_detector.getActualContacts()
            );
        }

        if (viz_manager.shouldSingleStep()) {
            // Single step logic similar to above
            std::fill(force_x.begin(), force_x.end(), 0.0f);
            std::fill(force_y.begin(), force_y.end(), 0.0f);
            std::fill(force_z.begin(), force_z.end(), 0.0f);

            for (int i = 0; i < n_particles; ++i) {
                force_y[i] -= masses[i] * 9.8f;
            }

            collision_detector.applyCollisionForces(
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
                force_x, force_y, force_z, masses
            );

            integrator.integrateStep(
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses,
                sim_params.time_step, 0.0f, 0.001f
            );

            collision_detector.resolveCollisions(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);

            viz_manager.updateFromSimulation(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);
            viz_manager.updateForces(force_x, force_y, force_z);
            viz_manager.resetSingleStep();
        }

        // Render everything (this includes the ImGui frame setup)
        viz_manager.render();

        // Limit frame rate
        if (delta_time < 0.016f) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16 - static_cast<int>(delta_time * 1000)));
        }
    }

    viz_manager.shutdown();
    Logger::getInstance().info("demo", "Collision detection demo completed");
    return 0;
}