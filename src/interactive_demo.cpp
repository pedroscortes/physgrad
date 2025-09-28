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

struct ParticleSystem {
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> vel_x, vel_y, vel_z;
    std::vector<float> masses;
    std::vector<float> force_x, force_y, force_z;

    int n_moving_particles = 0;
    int n_ground_particles = 0;

    void clear() {
        pos_x.clear(); pos_y.clear(); pos_z.clear();
        vel_x.clear(); vel_y.clear(); vel_z.clear();
        masses.clear();
        force_x.clear(); force_y.clear(); force_z.clear();
        n_moving_particles = 0;
        n_ground_particles = 0;
    }

    void resize(int total_particles) {
        pos_x.resize(total_particles);
        pos_y.resize(total_particles);
        pos_z.resize(total_particles);
        vel_x.resize(total_particles);
        vel_y.resize(total_particles);
        vel_z.resize(total_particles);
        masses.resize(total_particles);
        force_x.resize(total_particles);
        force_y.resize(total_particles);
        force_z.resize(total_particles);
    }

    void addRandomParticle(float mass_scale = 1.0f) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> pos_dist(-2.0f, 2.0f);
        static std::uniform_real_distribution<float> vel_dist(-1.0f, 1.0f);
        static std::uniform_real_distribution<float> mass_dist(0.5f, 2.0f);

        pos_x.push_back(pos_dist(gen));
        pos_y.push_back(pos_dist(gen) + 3.0f);
        pos_z.push_back(pos_dist(gen));

        vel_x.push_back(vel_dist(gen));
        vel_y.push_back(vel_dist(gen));
        vel_z.push_back(vel_dist(gen));

        masses.push_back(mass_dist(gen) * mass_scale);

        force_x.push_back(0.0f);
        force_y.push_back(0.0f);
        force_z.push_back(0.0f);

        n_moving_particles++;
    }

    void createGroundPlane() {
        const float ground_y = -3.0f;

        // Create 5x5 grid of ground particles
        for (int x = -2; x <= 2; ++x) {
            for (int z = -2; z <= 2; ++z) {
                pos_x.push_back(static_cast<float>(x));
                pos_y.push_back(ground_y);
                pos_z.push_back(static_cast<float>(z));

                vel_x.push_back(0.0f);
                vel_y.push_back(0.0f);
                vel_z.push_back(0.0f);

                masses.push_back(10.0f);

                force_x.push_back(0.0f);
                force_y.push_back(0.0f);
                force_z.push_back(0.0f);

                n_ground_particles++;
            }
        }
    }

    int getTotalParticles() const {
        return n_moving_particles + n_ground_particles;
    }
};

int main() {
    Logger::getInstance().info("demo", "Starting PhysGrad Interactive Physics Demo");

    // Load configuration
    ConfigManager config;
    if (!config.loadFromFile("config.conf")) {
        Logger::getInstance().warning("demo", "Could not load config.conf, using defaults");
    }
    auto sim_params = config.getSimulationParams();

    // Initialize visualization
    VisualizationManager viz_manager;
    if (!viz_manager.initialize(1280, 720)) {
        Logger::getInstance().error("demo", "Failed to initialize visualization");
        return -1;
    }

    // Setup collision detection
    CollisionParams collision_params;
    CollisionDetector collision_detector(collision_params);

    // Initialize particle system
    ParticleSystem particles;
    particles.createGroundPlane();

    // Add initial particles
    for (int i = 0; i < 6; ++i) {
        particles.addRandomParticle();
    }

    particles.resize(particles.getTotalParticles());
    collision_detector.updateBodyRadiiFromMasses(particles.masses, 1.0f);

    AdaptiveIntegrator integrator(IntegrationScheme::ADAPTIVE_RK45);

    Logger::getInstance().info("demo", "Starting interactive simulation with " +
                              std::to_string(particles.getTotalParticles()) + " particles");

    auto last_time = std::chrono::high_resolution_clock::now();

    while (!viz_manager.shouldClose()) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto delta_time = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;

        // Get interactive parameters
        auto& params = viz_manager.getInteractiveParams();

        if (viz_manager.isSimulationRunning()) {
            // Reset forces
            std::fill(particles.force_x.begin(), particles.force_x.end(), 0.0f);
            std::fill(particles.force_y.begin(), particles.force_y.end(), 0.0f);
            std::fill(particles.force_z.begin(), particles.force_z.end(), 0.0f);

            // Apply gravity to moving particles only
            if (params.enable_gravity) {
                for (int i = 0; i < particles.n_moving_particles; ++i) {
                    particles.force_y[i] -= particles.masses[i] * params.gravity_strength;
                }
            }

            // Apply collision forces
            if (params.enable_collisions) {
                // Update collision parameters in real-time
                collision_params.contact_stiffness = params.contact_stiffness;
                collision_params.contact_damping = params.contact_damping;
                collision_params.global_restitution = params.restitution;
                collision_params.global_friction = params.friction;
                collision_detector.setParameters(collision_params);

                collision_detector.applyCollisionForces(
                    particles.pos_x, particles.pos_y, particles.pos_z,
                    particles.vel_x, particles.vel_y, particles.vel_z,
                    particles.force_x, particles.force_y, particles.force_z,
                    particles.masses
                );
            }

            // Integrate motion
            float actual_dt = integrator.integrateStep(
                particles.pos_x, particles.pos_y, particles.pos_z,
                particles.vel_x, particles.vel_y, particles.vel_z,
                particles.masses,
                sim_params.time_step * viz_manager.getTimeScale() * 0.1f,
                0.0f, 0.001f
            );

            // Resolve collisions
            if (params.enable_collisions) {
                collision_detector.resolveCollisions(
                    particles.pos_x, particles.pos_y, particles.pos_z,
                    particles.vel_x, particles.vel_y, particles.vel_z,
                    particles.masses
                );
            }

            // Apply air damping to moving particles
            for (int i = 0; i < particles.n_moving_particles; ++i) {
                particles.vel_x[i] *= params.air_damping;
                particles.vel_y[i] *= params.air_damping;
                particles.vel_z[i] *= params.air_damping;
            }

            // Calculate energy for moving particles only
            float kinetic_energy = 0.0f;
            for (int i = 0; i < particles.n_moving_particles; ++i) {
                float v_squared = particles.vel_x[i] * particles.vel_x[i] +
                                 particles.vel_y[i] * particles.vel_y[i] +
                                 particles.vel_z[i] * particles.vel_z[i];
                kinetic_energy += 0.5f * particles.masses[i] * v_squared;
            }

            // Calculate potential energy (gravitational)
            float potential_energy = 0.0f;
            if (params.enable_gravity) {
                for (int i = 0; i < particles.n_moving_particles; ++i) {
                    // Potential energy = mgh (taking ground at y=-3 as reference)
                    potential_energy += particles.masses[i] * params.gravity_strength *
                                       (particles.pos_y[i] + 3.0f);
                }
            }

            // Update visualization
            viz_manager.updateFromSimulation(particles.pos_x, particles.pos_y, particles.pos_z,
                                           particles.vel_x, particles.vel_y, particles.vel_z,
                                           particles.masses);

            if (params.show_force_vectors) {
                viz_manager.updateForces(particles.force_x, particles.force_y, particles.force_z);
            }

            viz_manager.updateEnergy(kinetic_energy, potential_energy);
            viz_manager.setCollisionStats(
                collision_detector.getBroadPhasePairs(),
                collision_detector.getNarrowPhaseTests(),
                collision_detector.getActualContacts()
            );

            // Check for pause on collision
            if (params.pause_on_collision && collision_detector.getActualContacts() > 0) {
                viz_manager.getInteractiveParams().pause_on_collision = false; // Auto-resume
                // Could add a pause mechanism here
            }
        }

        if (viz_manager.shouldSingleStep()) {
            // Similar logic for single step
            viz_manager.resetSingleStep();
        }

        // Handle UI button presses (simplified for now)
        // In a real implementation, you'd need to add button state tracking

        viz_manager.render();

        // Limit frame rate
        if (delta_time < 0.016f) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16 - static_cast<int>(delta_time * 1000)));
        }
    }

    viz_manager.shutdown();
    Logger::getInstance().info("demo", "Interactive physics demo completed");
    return 0;
}