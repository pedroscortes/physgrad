/**
 * PhysGrad - Fluid Dynamics Unit Tests
 */

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include "common_types.h"

// Mock fluid dynamics interface
class FluidDynamics {
public:
    struct FluidParticle {
        float3 position;
        float3 velocity;
        float density;
        float pressure;
        float mass;
    };

    bool initialize() { return true; }
    void cleanup() {}

    void calculateDensity(
        std::vector<FluidParticle>& particles,
        float smoothing_length,
        float rest_density
    ) {
        for (size_t i = 0; i < particles.size(); ++i) {
            float density = 0.0f;
            for (size_t j = 0; j < particles.size(); ++j) {
                float3 r_ij = {
                    particles[i].position.x - particles[j].position.x,
                    particles[i].position.y - particles[j].position.y,
                    particles[i].position.z - particles[j].position.z
                };
                float r = std::sqrt(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);

                if (r < smoothing_length) {
                    float q = r / smoothing_length;
                    float kernel = poly6Kernel(q);
                    density += particles[j].mass * kernel;
                }
            }
            particles[i].density = std::max(density, 0.001f * rest_density);
        }
    }

    void calculatePressure(
        std::vector<FluidParticle>& particles,
        float rest_density,
        float gas_constant
    ) {
        for (auto& particle : particles) {
            particle.pressure = std::max(0.0f, gas_constant * (particle.density - rest_density));
        }
    }

    void calculateForces(
        std::vector<FluidParticle>& particles,
        std::vector<float3>& forces,
        float smoothing_length,
        float viscosity
    ) {
        forces.assign(particles.size(), {0.0f, 0.0f, 0.0f});

        for (size_t i = 0; i < particles.size(); ++i) {
            float3 pressure_force = {0.0f, 0.0f, 0.0f};
            float3 viscosity_force = {0.0f, 0.0f, 0.0f};

            for (size_t j = 0; j < particles.size(); ++j) {
                if (i == j) continue;

                float3 r_ij = {
                    particles[i].position.x - particles[j].position.x,
                    particles[i].position.y - particles[j].position.y,
                    particles[i].position.z - particles[j].position.z
                };
                float r = std::sqrt(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);

                if (r < smoothing_length && r > 1e-6f) {
                    float3 r_unit = {r_ij.x / r, r_ij.y / r, r_ij.z / r};

                    // Pressure force
                    float pressure_term = (particles[i].pressure + particles[j].pressure) /
                                        (2.0f * particles[j].density);
                    float3 gradient = spikyGradient(r, smoothing_length, r_unit);

                    pressure_force.x -= particles[j].mass * pressure_term * gradient.x;
                    pressure_force.y -= particles[j].mass * pressure_term * gradient.y;
                    pressure_force.z -= particles[j].mass * pressure_term * gradient.z;

                    // Viscosity force
                    float3 vel_diff = {
                        particles[j].velocity.x - particles[i].velocity.x,
                        particles[j].velocity.y - particles[i].velocity.y,
                        particles[j].velocity.z - particles[i].velocity.z
                    };

                    float laplacian = viscosityLaplacian(r, smoothing_length);
                    float viscosity_term = viscosity * particles[j].mass / particles[j].density * laplacian;

                    viscosity_force.x += viscosity_term * vel_diff.x;
                    viscosity_force.y += viscosity_term * vel_diff.y;
                    viscosity_force.z += viscosity_term * vel_diff.z;
                }
            }

            forces[i].x = pressure_force.x + viscosity_force.x;
            forces[i].y = pressure_force.y + viscosity_force.y;
            forces[i].z = pressure_force.z + viscosity_force.z;
        }
    }

    void integrate(
        std::vector<FluidParticle>& particles,
        const std::vector<float3>& forces,
        float dt
    ) {
        for (size_t i = 0; i < particles.size(); ++i) {
            float3 acceleration = {
                forces[i].x / particles[i].mass,
                forces[i].y / particles[i].mass,
                forces[i].z / particles[i].mass
            };

            particles[i].velocity.x += acceleration.x * dt;
            particles[i].velocity.y += acceleration.y * dt;
            particles[i].velocity.z += acceleration.z * dt;

            particles[i].position.x += particles[i].velocity.x * dt;
            particles[i].position.y += particles[i].velocity.y * dt;
            particles[i].position.z += particles[i].velocity.z * dt;
        }
    }

private:
    float poly6Kernel(float q) {
        if (q >= 1.0f) return 0.0f;
        float q2 = q * q;
        return 315.0f / (64.0f * M_PI) * (1 - q2) * (1 - q2) * (1 - q2);
    }

    float3 spikyGradient(float r, float h, float3 r_unit) {
        if (r >= h) return {0.0f, 0.0f, 0.0f};
        float factor = -45.0f / (M_PI * std::pow(h, 6)) * std::pow(h - r, 2);
        return {factor * r_unit.x, factor * r_unit.y, factor * r_unit.z};
    }

    float viscosityLaplacian(float r, float h) {
        if (r >= h) return 0.0f;
        return 45.0f / (M_PI * std::pow(h, 6)) * (h - r);
    }
};

class FluidDynamicsTest : public ::testing::Test {
protected:
    void SetUp() override {
        fluid_system_ = std::make_unique<FluidDynamics>();
        ASSERT_TRUE(fluid_system_->initialize());
    }

    void TearDown() override {
        if (fluid_system_) {
            fluid_system_->cleanup();
        }
    }

    std::unique_ptr<FluidDynamics> fluid_system_;
};

TEST_F(FluidDynamicsTest, InitializationAndCleanup) {
    EXPECT_TRUE(fluid_system_ != nullptr);
}

TEST_F(FluidDynamicsTest, DensityCalculation) {
    std::vector<FluidDynamics::FluidParticle> particles = {
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f, 1.0f},
        {{0.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f, 1.0f},
        {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f, 1.0f}
    };

    float smoothing_length = 1.0f;
    float rest_density = 1000.0f;

    fluid_system_->calculateDensity(particles, smoothing_length, rest_density);

    // All particles should have positive density
    for (const auto& particle : particles) {
        EXPECT_GT(particle.density, 0.0f);
    }

    // Center particle should have higher density due to neighbors
    EXPECT_GT(particles[1].density, particles[0].density);
    EXPECT_GT(particles[1].density, particles[2].density);
}

TEST_F(FluidDynamicsTest, PressureCalculation) {
    std::vector<FluidDynamics::FluidParticle> particles = {
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 1200.0f, 0.0f, 1.0f}, // High density
        {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 800.0f, 0.0f, 1.0f}   // Low density
    };

    float rest_density = 1000.0f;
    float gas_constant = 1000.0f;

    fluid_system_->calculatePressure(particles, rest_density, gas_constant);

    // High density particle should have positive pressure
    EXPECT_GT(particles[0].pressure, 0.0f);

    // Low density particle should have zero pressure (clamped)
    EXPECT_EQ(particles[1].pressure, 0.0f);

    // Higher density should result in higher pressure
    EXPECT_GT(particles[0].pressure, particles[1].pressure);
}

TEST_F(FluidDynamicsTest, ForceCalculation) {
    std::vector<FluidDynamics::FluidParticle> particles = {
        {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1200.0f, 200.0f, 1.0f},
        {{0.5f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, 1200.0f, 200.0f, 1.0f}
    };

    std::vector<float3> forces;
    float smoothing_length = 1.0f;
    float viscosity = 0.01f;

    fluid_system_->calculateForces(particles, forces, smoothing_length, viscosity);

    EXPECT_EQ(forces.size(), particles.size());

    // Forces should be opposite (Newton's 3rd law)
    EXPECT_NEAR(forces[0].x, -forces[1].x, 1e-6f);
    EXPECT_NEAR(forces[0].y, -forces[1].y, 1e-6f);
    EXPECT_NEAR(forces[0].z, -forces[1].z, 1e-6f);

    // Forces should be non-zero due to pressure and viscosity differences
    EXPECT_NE(forces[0].x, 0.0f);
}

TEST_F(FluidDynamicsTest, TimeIntegration) {
    std::vector<FluidDynamics::FluidParticle> particles = {
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 1000.0f, 0.0f, 1.0f}
    };

    std::vector<float3> forces = {{10.0f, 0.0f, 0.0f}};
    float dt = 0.01f;

    float3 initial_position = particles[0].position;
    float3 initial_velocity = particles[0].velocity;

    fluid_system_->integrate(particles, forces, dt);

    // Velocity should increase due to force
    EXPECT_GT(particles[0].velocity.x, initial_velocity.x);

    // Position should change due to velocity
    EXPECT_GT(particles[0].position.x, initial_position.x);
}

TEST_F(FluidDynamicsTest, MassConservation) {
    std::vector<FluidDynamics::FluidParticle> particles = {
        {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1000.0f, 0.0f, 1.0f},
        {{1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, 1000.0f, 0.0f, 1.0f}
    };

    float total_mass = 0.0f;
    for (const auto& particle : particles) {
        total_mass += particle.mass;
    }

    // Run simulation steps
    for (int step = 0; step < 10; ++step) {
        fluid_system_->calculateDensity(particles, 1.0f, 1000.0f);
        fluid_system_->calculatePressure(particles, 1000.0f, 1000.0f);

        std::vector<float3> forces;
        fluid_system_->calculateForces(particles, forces, 1.0f, 0.01f);
        fluid_system_->integrate(particles, forces, 0.001f);
    }

    // Total mass should be conserved
    float final_mass = 0.0f;
    for (const auto& particle : particles) {
        final_mass += particle.mass;
    }

    EXPECT_NEAR(total_mass, final_mass, 1e-6f);
}

TEST_F(FluidDynamicsTest, StabilityTest) {
    // Create a regular grid of particles
    std::vector<FluidDynamics::FluidParticle> particles;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            FluidDynamics::FluidParticle particle;
            particle.position = {static_cast<float>(i) * 0.5f, static_cast<float>(j) * 0.5f, 0.0f};
            particle.velocity = {0.0f, 0.0f, 0.0f};
            particle.density = 1000.0f;
            particle.pressure = 0.0f;
            particle.mass = 1.0f;
            particles.push_back(particle);
        }
    }

    // Run simulation and check for stability
    bool simulation_stable = true;
    for (int step = 0; step < 100; ++step) {
        fluid_system_->calculateDensity(particles, 1.0f, 1000.0f);
        fluid_system_->calculatePressure(particles, 1000.0f, 100.0f);

        std::vector<float3> forces;
        fluid_system_->calculateForces(particles, forces, 1.0f, 0.01f);
        fluid_system_->integrate(particles, forces, 0.001f);

        // Check for NaN or infinite values
        for (const auto& particle : particles) {
            if (std::isnan(particle.position.x) || std::isinf(particle.position.x) ||
                std::isnan(particle.velocity.x) || std::isinf(particle.velocity.x)) {
                simulation_stable = false;
                break;
            }
        }

        if (!simulation_stable) break;
    }

    EXPECT_TRUE(simulation_stable);
}

TEST_F(FluidDynamicsTest, ViscosityEffect) {
    std::vector<FluidDynamics::FluidParticle> particles = {
        {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1000.0f, 0.0f, 1.0f},
        {{0.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 1000.0f, 0.0f, 1.0f}
    };

    // Test with different viscosity values
    std::vector<float> viscosities = {0.0f, 0.01f, 0.1f};
    std::vector<float> velocity_changes;

    for (float viscosity : viscosities) {
        auto test_particles = particles;
        std::vector<float3> forces;

        fluid_system_->calculateForces(test_particles, forces, 1.0f, viscosity);

        float velocity_change = std::abs(forces[0].x);
        velocity_changes.push_back(velocity_change);
    }

    // Higher viscosity should result in larger forces between particles with different velocities
    EXPECT_LT(velocity_changes[0], velocity_changes[1]); // No viscosity < low viscosity
    EXPECT_LT(velocity_changes[1], velocity_changes[2]); // Low viscosity < high viscosity
}

TEST_F(FluidDynamicsTest, PressureGradient) {
    // Create particles with pressure gradient
    std::vector<FluidDynamics::FluidParticle> particles = {
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 1000.0f, 1000.0f, 1.0f}, // High pressure
        {{0.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 1000.0f, 0.0f, 1.0f}     // Low pressure
    };

    std::vector<float3> forces;
    fluid_system_->calculateForces(particles, forces, 1.0f, 0.0f);

    // Pressure forces should be equal and opposite (Newton's 3rd law)
    EXPECT_LT(forces[0].x, 0.0f); // Force from pressure gradient
    EXPECT_GT(forces[1].x, 0.0f); // Equal and opposite force
    EXPECT_NEAR(forces[0].x, -forces[1].x, 1e-6); // Newton's 3rd law
}