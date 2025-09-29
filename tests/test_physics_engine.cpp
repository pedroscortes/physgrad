/**
 * PhysGrad - Physics Engine Unit Tests
 */

#include <gtest/gtest.h>
#include "physics_engine.h"
#include <memory>
#include <vector>
#include <chrono>

using namespace physgrad;

class PhysicsEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_unique<PhysicsEngine>();
        ASSERT_TRUE(engine_->initialize());
    }

    void TearDown() override {
        if (engine_) {
            engine_->cleanup();
        }
    }

    std::unique_ptr<PhysicsEngine> engine_;
};

TEST_F(PhysicsEngineTest, InitializationAndCleanup) {
    EXPECT_TRUE(engine_ != nullptr);
    // Engine should be properly initialized in SetUp
}

TEST_F(PhysicsEngineTest, AddRemoveParticles) {
    // Test adding particles
    std::vector<float3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}
    };

    std::vector<float3> velocities = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };

    std::vector<float> masses = {1.0f, 1.0f, 1.0f};

    engine_->addParticles(positions, velocities, masses);

    EXPECT_EQ(engine_->getNumParticles(), 3);

    // Test removing particles
    engine_->removeParticle(1);
    EXPECT_EQ(engine_->getNumParticles(), 2);
}

TEST_F(PhysicsEngineTest, TimeStepIntegration) {
    // Add a simple particle system
    std::vector<float3> positions = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    std::vector<float3> velocities = {{1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}};
    std::vector<float> masses = {1.0f, 1.0f};

    engine_->addParticles(positions, velocities, masses);

    float dt = 0.01f;
    float initial_energy = engine_->calculateTotalEnergy();

    // Step forward in time
    engine_->step(dt);

    // Check that particles moved
    auto new_positions = engine_->getPositions();
    EXPECT_NE(new_positions[0].x, 0.0f);
    EXPECT_NE(new_positions[1].x, 1.0f);

    // Energy should be conserved (approximately)
    // Note: Single time step with dt=0.01 will have some numerical error
    float final_energy = engine_->calculateTotalEnergy();
    EXPECT_NEAR(initial_energy, final_energy, 1e-3);
}

TEST_F(PhysicsEngineTest, ForceCalculation) {
    // Test electrostatic force calculation
    std::vector<float3> positions = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    std::vector<float3> velocities = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    std::vector<float> masses = {1.0f, 1.0f};
    std::vector<float> charges = {1.0f, -1.0f};

    engine_->addParticles(positions, velocities, masses);
    engine_->setCharges(charges);

    engine_->updateForces();
    auto forces = engine_->getForces();

    // Force should be attractive (negative x-direction for first particle)
    EXPECT_LT(forces[0].x, 0.0f);
    EXPECT_GT(forces[1].x, 0.0f);

    // Forces should be equal and opposite (Newton's 3rd law)
    EXPECT_NEAR(forces[0].x, -forces[1].x, 1e-6);
}

TEST_F(PhysicsEngineTest, EnergyConservation) {
    // Create a simple two-body system
    std::vector<float3> positions = {{-0.5f, 0.0f, 0.0f}, {0.5f, 0.0f, 0.0f}};
    std::vector<float3> velocities = {{0.1f, 0.0f, 0.0f}, {-0.1f, 0.0f, 0.0f}};
    std::vector<float> masses = {1.0f, 1.0f};

    engine_->addParticles(positions, velocities, masses);

    float initial_energy = engine_->calculateTotalEnergy();

    // Run simulation for multiple steps
    float dt = 0.001f;
    for (int i = 0; i < 100; ++i) {
        engine_->step(dt);
    }

    float final_energy = engine_->calculateTotalEnergy();

    // Energy should be conserved within numerical precision
    EXPECT_NEAR(initial_energy, final_energy, 1e-4);
}

TEST_F(PhysicsEngineTest, PerformanceBaseline) {
    // Test with larger number of particles for performance baseline
    int num_particles = 1000;
    std::vector<float3> positions, velocities;
    std::vector<float> masses;

    // Create random particle distribution
    for (int i = 0; i < num_particles; ++i) {
        positions.push_back({
            static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f,
            static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f,
            static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f
        });
        velocities.push_back({0.0f, 0.0f, 0.0f});
        masses.push_back(1.0f);
    }

    engine_->addParticles(positions, velocities, masses);

    // Time a single simulation step
    auto start = std::chrono::high_resolution_clock::now();
    engine_->step(0.001f);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should complete within reasonable time (< 100ms for 1000 particles)
    EXPECT_LT(duration.count(), 100000);

    std::cout << "Performance: " << num_particles << " particles in "
              << duration.count() << " microseconds" << std::endl;
}

// Test boundary conditions
TEST_F(PhysicsEngineTest, BoundaryConditions) {
    engine_->setBoundaryConditions(BoundaryType::PERIODIC, {10.0f, 10.0f, 10.0f});

    std::vector<float3> positions = {{9.5f, 0.0f, 0.0f}};
    std::vector<float3> velocities = {{1.0f, 0.0f, 0.0f}};
    std::vector<float> masses = {1.0f};

    engine_->addParticles(positions, velocities, masses);

    // Step forward - should wrap around due to periodic boundaries
    float dt = 1.0f;
    engine_->step(dt);

    auto new_positions = engine_->getPositions();

    // Position should wrap around
    EXPECT_LT(new_positions[0].x, 5.0f);  // Should be on the other side
}

// Integration methods test
TEST_F(PhysicsEngineTest, IntegrationMethods) {
    std::vector<float3> positions = {{0.0f, 0.0f, 0.0f}};
    std::vector<float3> velocities = {{1.0f, 0.0f, 0.0f}};
    std::vector<float> masses = {1.0f};

    engine_->addParticles(positions, velocities, masses);

    // Test different integration methods
    std::vector<IntegrationMethod> methods = {
        IntegrationMethod::VERLET,
        IntegrationMethod::RUNGE_KUTTA_4,
        IntegrationMethod::LEAPFROG
    };

    for (auto method : methods) {
        engine_->setIntegrationMethod(method);

        // Reset positions
        engine_->setPositions({{0.0f, 0.0f, 0.0f}});
        engine_->setVelocities({{1.0f, 0.0f, 0.0f}});

        float dt = 0.1f;
        engine_->step(dt);

        auto final_positions = engine_->getPositions();

        // Should have moved approximately dt * velocity
        EXPECT_NEAR(final_positions[0].x, dt, 0.01f);
    }
}