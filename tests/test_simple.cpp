/**
 * PhysGrad Simple Test
 *
 * Basic test to verify core functionality without complex dependencies.
 */

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include <chrono>
#include <iostream>

#include "src/common_types.h"

using namespace physgrad;

class SimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Basic setup
    }

    void TearDown() override {
        // Basic cleanup
    }
};

// =============================================================================
// BASIC TYPE TESTS
// =============================================================================

TEST_F(SimpleTest, CommonTypesWork) {
    float3 pos = make_float3(1.0f, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(pos.x, 1.0f);
    EXPECT_FLOAT_EQ(pos.y, 2.0f);
    EXPECT_FLOAT_EQ(pos.z, 3.0f);

    float3 vel = make_float3(0.1f, 0.2f, 0.3f);
    float3 sum = pos + vel;
    EXPECT_FLOAT_EQ(sum.x, 1.1f);
    EXPECT_FLOAT_EQ(sum.y, 2.2f);
    EXPECT_FLOAT_EQ(sum.z, 3.3f);
}

TEST_F(SimpleTest, VectorOperations) {
    float3 a = make_float3(1.0f, 0.0f, 0.0f);
    float3 b = make_float3(0.0f, 1.0f, 0.0f);

    float mag_a = magnitude(a);
    EXPECT_NEAR(mag_a, 1.0f, 1e-6);

    float3 norm_a = normalize(a);
    EXPECT_NEAR(norm_a.x, 1.0f, 1e-6);
    EXPECT_NEAR(norm_a.y, 0.0f, 1e-6);
    EXPECT_NEAR(norm_a.z, 0.0f, 1e-6);

    float dot_ab = dot(a, b);
    EXPECT_NEAR(dot_ab, 0.0f, 1e-6);

    float3 cross_ab = cross(a, b);
    EXPECT_NEAR(cross_ab.x, 0.0f, 1e-6);
    EXPECT_NEAR(cross_ab.y, 0.0f, 1e-6);
    EXPECT_NEAR(cross_ab.z, 1.0f, 1e-6);
}

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE

TEST_F(SimpleTest, ConceptsCompile) {
    // Test that basic concepts work
    static_assert(concepts::PhysicsScalar<float>);
    static_assert(concepts::PhysicsScalar<double>);
    static_assert(concepts::HighPrecisionScalar<double>);
    static_assert(concepts::GPUCompatible<float>);

    using TestVector = ConceptVector3D<float>;
    TestVector v{1.0f, 2.0f, 3.0f};
    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
    EXPECT_FLOAT_EQ(v[2], 3.0f);
    EXPECT_EQ(v.size(), 3);
}

TEST_F(SimpleTest, ConceptParticle) {
    using TestParticle = ConceptParticleData<float>;
    ConceptVector3D<float> pos{1.0f, 2.0f, 3.0f};
    ConceptVector3D<float> vel{0.1f, 0.2f, 0.3f};

    TestParticle particle{pos, vel, 1.5f};

    auto p = particle.position();
    auto v = particle.velocity();
    auto m = particle.mass();

    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(v[0], 0.1f);
    EXPECT_FLOAT_EQ(m, 1.5f);
}

#endif

// =============================================================================
// PHYSICS CONSTANTS TESTS
// =============================================================================

TEST_F(SimpleTest, PhysicsConstants) {
    // Test that physics constants are reasonable
    EXPECT_GT(COULOMB_CONSTANT, 1e9);
    EXPECT_LT(COULOMB_CONSTANT, 1e10);

    EXPECT_GT(SPEED_OF_LIGHT, 1e8);
    EXPECT_LT(SPEED_OF_LIGHT, 1e9);

    EXPECT_GT(PLANCK_CONSTANT, 1e-35);
    EXPECT_LT(PLANCK_CONSTANT, 1e-33);
}

// =============================================================================
// BOUNDARY CONDITIONS TESTS
// =============================================================================

TEST_F(SimpleTest, BoundaryTypes) {
    // Test that boundary condition enum works
    BoundaryType open = BoundaryType::OPEN;
    BoundaryType periodic = BoundaryType::PERIODIC;
    BoundaryType reflective = BoundaryType::REFLECTIVE;

    EXPECT_NE(open, periodic);
    EXPECT_NE(periodic, reflective);
    EXPECT_NE(open, reflective);
}

TEST_F(SimpleTest, IntegrationMethods) {
    // Test that integration method enum works
    IntegrationMethod euler = IntegrationMethod::EULER;
    IntegrationMethod verlet = IntegrationMethod::VERLET;
    IntegrationMethod rk4 = IntegrationMethod::RUNGE_KUTTA_4;
    IntegrationMethod leapfrog = IntegrationMethod::LEAPFROG;

    EXPECT_NE(euler, verlet);
    EXPECT_NE(verlet, rk4);
    EXPECT_NE(rk4, leapfrog);
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

TEST_F(SimpleTest, VectorPerformance) {
    const int num_operations = 100000;

    std::vector<float3> vectors;
    vectors.reserve(num_operations);

    // Generate test vectors
    for (int i = 0; i < num_operations; ++i) {
        vectors.push_back(make_float3(
            static_cast<float>(i),
            static_cast<float>(i * 2),
            static_cast<float>(i * 3)
        ));
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Perform vector operations
    float total_magnitude = 0.0f;
    for (const auto& vec : vectors) {
        total_magnitude += magnitude(vec);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should complete in reasonable time
    EXPECT_LT(duration.count(), 100000); // Less than 100ms

    // Result should be reasonable
    EXPECT_GT(total_magnitude, 0.0f);
}

// =============================================================================
// COMPILATION TESTS
// =============================================================================

TEST_F(SimpleTest, CompilationFeatures) {
    // Test that we have the expected C++ features
    EXPECT_GE(__cplusplus, 202002L); // C++20 or later

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    std::cout << "C++20 concepts are available\n";
#else
    std::cout << "C++20 concepts are not available\n";
#endif

#ifdef __CUDACC__
    std::cout << "CUDA compilation enabled\n";
#else
    std::cout << "CPU-only compilation\n";
#endif
}

// Let gtest_main handle the main function