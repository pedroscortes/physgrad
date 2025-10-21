/**
 * PhysGrad Physics Engine Unit Tests
 *
 * Comprehensive unit tests for the physics engine core functionality.
 */

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>

#include "physics_engine.h"
#include "common_types.h"

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "concepts/forward_declarations.h"
#endif

using namespace physgrad;

class PhysicsEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_unique<PhysicsEngine>();
        ASSERT_TRUE(engine_->initialize()) << "Physics engine initialization failed";
    }

    void TearDown() override {
        engine_->cleanup();
        engine_.reset();
    }

    std::unique_ptr<PhysicsEngine> engine_;
};

// =============================================================================
// BASIC FUNCTIONALITY TESTS
// =============================================================================

TEST_F(PhysicsEngineTest, InitializationAndCleanup) {
    // Test is implicit in SetUp/TearDown
    EXPECT_EQ(engine_->getNumParticles(), 0);
}

TEST_F(PhysicsEngineTest, ParticleAddition) {
    std::vector<float3> positions = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    std::vector<float3> velocities = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f}
    };
    std::vector<float> masses = {1.0f, 2.0f};

    engine_->addParticles(positions, velocities, masses);

    EXPECT_EQ(engine_->getNumParticles(), 2);

    auto retrieved_positions = engine_->getPositions();
    auto retrieved_velocities = engine_->getVelocities();

    ASSERT_EQ(retrieved_positions.size(), 2);
    ASSERT_EQ(retrieved_velocities.size(), 2);

    // Check positions
    EXPECT_FLOAT_EQ(retrieved_positions[0].x, 1.0f);
    EXPECT_FLOAT_EQ(retrieved_positions[0].y, 2.0f);
    EXPECT_FLOAT_EQ(retrieved_positions[0].z, 3.0f);

    EXPECT_FLOAT_EQ(retrieved_positions[1].x, 4.0f);
    EXPECT_FLOAT_EQ(retrieved_positions[1].y, 5.0f);
    EXPECT_FLOAT_EQ(retrieved_positions[1].z, 6.0f);

    // Check velocities
    EXPECT_FLOAT_EQ(retrieved_velocities[0].x, 0.1f);
    EXPECT_FLOAT_EQ(retrieved_velocities[0].y, 0.2f);
    EXPECT_FLOAT_EQ(retrieved_velocities[0].z, 0.3f);
}

// Simplified test for limited implementation
TEST_F(PhysicsEngineTest, BasicSimulation) {
    std::vector<float3> positions = {{0.0f, 0.0f, 0.0f}};
    std::vector<float3> velocities = {{1.0f, 0.0f, 0.0f}};
    std::vector<float> masses = {1.0f};

    engine_->addParticles(positions, velocities, masses);

    // Single step
    engine_->step(0.1f);

    auto final_positions = engine_->getPositions();

    // Position should have advanced
    EXPECT_GT(final_positions[0].x, 0.0f);
}

// =============================================================================
// CONCEPT-BASED TESTS (if available)
// =============================================================================

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE

TEST_F(PhysicsEngineTest, ConceptValidation) {
    // Test that our types satisfy physics concepts
    EXPECT_TRUE((concepts::PhysicsScalar<float>));
    EXPECT_TRUE((concepts::PhysicsScalar<double>));
    EXPECT_TRUE((concepts::HighPrecisionScalar<double>));

    using TestVector = ConceptVector3D<float>;
    EXPECT_TRUE((concepts::Vector3D<TestVector>));

    using TestParticle = ConceptParticleData<float>;
    EXPECT_TRUE((concepts::DynamicParticle<TestParticle>));
    EXPECT_TRUE((concepts::GPUCompatible<TestParticle>));
}

TEST_F(PhysicsEngineTest, TypeOptimization) {
    // Test automatic type optimization
    using optimal_32 = type_traits::optimal_scalar_t<32>;
    using optimal_64 = type_traits::optimal_scalar_t<64>;

    EXPECT_TRUE((std::same_as<optimal_32, float>));
    EXPECT_TRUE((std::same_as<optimal_64, double>));

    // Test GPU layout optimization
    constexpr bool float_optimal = type_traits::gpu_layout_type<float>::is_optimal;
    constexpr int block_size = type_traits::cuda_block_size_v<float>;

    EXPECT_GE(block_size, 32);  // Should be reasonable block size
    EXPECT_LE(block_size, 1024); // Should not exceed GPU limits
}

#endif

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}