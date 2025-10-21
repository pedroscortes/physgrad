/**
 * PhysGrad C++20 Concepts Validation Test
 *
 * Comprehensive tests for physics concepts type system
 */

#include <gtest/gtest.h>
#include <type_traits>
#include <vector>
#include <array>

#define PHYSGRAD_CONCEPTS_AVAILABLE
#include "common_types.h"
#include "concepts/physics_concepts.h"
#include "concepts/type_traits.h"

using namespace physgrad;
using namespace physgrad::concepts;
using namespace physgrad::type_traits;

// =============================================================================
// BASIC CONCEPT VALIDATION
// =============================================================================

TEST(ConceptsTest, PhysicsScalarConcept) {
    // Valid physics scalars
    EXPECT_TRUE(PhysicsScalar<float>);
    EXPECT_TRUE(PhysicsScalar<double>);

    // Invalid types
    EXPECT_FALSE(PhysicsScalar<int>);
    EXPECT_FALSE(PhysicsScalar<char>);
    EXPECT_FALSE(PhysicsScalar<std::string>);
}

TEST(ConceptsTest, HighPrecisionScalarConcept) {
    // Double is high precision
    EXPECT_TRUE(HighPrecisionScalar<double>);
    EXPECT_TRUE(HighPrecisionScalar<long double>);

    // Float is not high precision
    EXPECT_FALSE(HighPrecisionScalar<float>);
}

TEST(ConceptsTest, Vector3DConcept) {
    // ConceptVector3D should satisfy Vector3D
    EXPECT_TRUE((Vector3D<ConceptVector3D<float>>));
    EXPECT_TRUE((Vector3D<ConceptVector3D<double>>));

    // std::array<T, 3> also satisfies Vector3D concept (has operator[] and size())
    constexpr bool array_is_vector = Vector3D<std::array<float, 3>>;
    EXPECT_TRUE(array_is_vector);
}

TEST(ConceptsTest, GPUCompatibleConcept) {
    // ConceptParticleData should be GPU compatible
    EXPECT_TRUE((GPUCompatible<ConceptParticleData<float>>));
    EXPECT_TRUE((GPUCompatible<ConceptParticleData<double>>));

    // Basic types should be GPU compatible
    EXPECT_TRUE((GPUCompatible<float>));
    EXPECT_TRUE((GPUCompatible<double>));
    EXPECT_TRUE((GPUCompatible<int>));

    // std::vector is not GPU compatible (not trivially copyable)
    constexpr bool vector_is_gpu = GPUCompatible<std::vector<float>>;
    EXPECT_FALSE(vector_is_gpu);
}

// =============================================================================
// CONCEPT VECTOR3D FUNCTIONALITY
// =============================================================================

TEST(ConceptsTest, ConceptVector3DConstruction) {
    ConceptVector3D<float> v1;
    EXPECT_FLOAT_EQ(v1.x, 0.0f);
    EXPECT_FLOAT_EQ(v1.y, 0.0f);
    EXPECT_FLOAT_EQ(v1.z, 0.0f);

    ConceptVector3D<float> v2(1.0f, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(v2.x, 1.0f);
    EXPECT_FLOAT_EQ(v2.y, 2.0f);
    EXPECT_FLOAT_EQ(v2.z, 3.0f);
}

TEST(ConceptsTest, ConceptVector3DOperations) {
    ConceptVector3D<float> v1(1.0f, 2.0f, 3.0f);
    ConceptVector3D<float> v2(4.0f, 5.0f, 6.0f);

    // Addition
    auto v3 = v1 + v2;
    EXPECT_FLOAT_EQ(v3.x, 5.0f);
    EXPECT_FLOAT_EQ(v3.y, 7.0f);
    EXPECT_FLOAT_EQ(v3.z, 9.0f);

    // Subtraction
    auto v4 = v2 - v1;
    EXPECT_FLOAT_EQ(v4.x, 3.0f);
    EXPECT_FLOAT_EQ(v4.y, 3.0f);
    EXPECT_FLOAT_EQ(v4.z, 3.0f);

    // Scalar multiplication
    auto v5 = v1 * 2.0f;
    EXPECT_FLOAT_EQ(v5.x, 2.0f);
    EXPECT_FLOAT_EQ(v5.y, 4.0f);
    EXPECT_FLOAT_EQ(v5.z, 6.0f);
}

TEST(ConceptsTest, ConceptVector3DArrayAccess) {
    ConceptVector3D<float> v(1.0f, 2.0f, 3.0f);

    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
    EXPECT_FLOAT_EQ(v[2], 3.0f);

    EXPECT_EQ(v.size(), 3);
}

// =============================================================================
// CONCEPT PARTICLE DATA
// =============================================================================

TEST(ConceptsTest, ConceptParticleDataConstruction) {
    ConceptParticleData<float> particle;

    EXPECT_FLOAT_EQ(particle.mass(), 1.0f);

    ConceptVector3D<float> pos(1.0f, 2.0f, 3.0f);
    ConceptVector3D<float> vel(0.1f, 0.2f, 0.3f);

    ConceptParticleData<float> particle2(pos, vel, 2.5f);
    EXPECT_FLOAT_EQ(particle2.mass(), 2.5f);
    EXPECT_FLOAT_EQ(particle2.position().x, 1.0f);
    EXPECT_FLOAT_EQ(particle2.velocity().x, 0.1f);
}

TEST(ConceptsTest, ConceptParticleDataModification) {
    ConceptParticleData<float> particle;

    ConceptVector3D<float> new_pos(5.0f, 6.0f, 7.0f);
    particle.set_position(new_pos);

    auto retrieved_pos = particle.position();
    EXPECT_FLOAT_EQ(retrieved_pos.x, 5.0f);
    EXPECT_FLOAT_EQ(retrieved_pos.y, 6.0f);
    EXPECT_FLOAT_EQ(retrieved_pos.z, 7.0f);
}

// =============================================================================
// TYPE TRAITS VALIDATION
// =============================================================================

TEST(ConceptsTest, ScalarPrecisionTraits) {
    EXPECT_EQ(scalar_precision_v<float>, 32);
    EXPECT_EQ(scalar_precision_v<double>, 64);
    EXPECT_EQ(scalar_precision_v<int>, 0);  // Not a physics scalar
}

TEST(ConceptsTest, OptimalScalarSelection) {
    using optimal_32 = optimal_scalar_t<32>;
    using optimal_64 = optimal_scalar_t<64>;

    EXPECT_TRUE((std::is_same_v<optimal_32, float>));
    EXPECT_TRUE((std::is_same_v<optimal_64, double>));
}

TEST(ConceptsTest, CUDABlockSizeCalculation) {
    // Small types should use more threads
    EXPECT_EQ(cuda_block_size_v<float>, 256);
    EXPECT_EQ(cuda_block_size_v<double>, 256);

    // Large types should use fewer threads
    struct LargeType { char data[128]; };
    EXPECT_EQ(cuda_block_size_v<LargeType>, 64);
}

TEST(ConceptsTest, PhysicsDataFactory) {
    using factory = physics_data_factory<float, 3>;
    using vec_type = factory::vector_type;

    EXPECT_EQ(sizeof(vec_type), sizeof(float) * 3);
}

// =============================================================================
// COMPATIBILITY TESTS
// =============================================================================

TEST(ConceptsTest, LegacyTypeConversion) {
    // Test conversion between legacy and concept types
    float3 legacy{1.0f, 2.0f, 3.0f};

    ConceptVector3D<float> concept_vec(legacy);
    EXPECT_FLOAT_EQ(concept_vec.x, 1.0f);
    EXPECT_FLOAT_EQ(concept_vec.y, 2.0f);
    EXPECT_FLOAT_EQ(concept_vec.z, 3.0f);

    float3 back_to_legacy = concept_vec.to_float3();
    EXPECT_FLOAT_EQ(back_to_legacy.x, 1.0f);
    EXPECT_FLOAT_EQ(back_to_legacy.y, 2.0f);
    EXPECT_FLOAT_EQ(back_to_legacy.z, 3.0f);
}

// =============================================================================
// COMPILE-TIME VALIDATION
// =============================================================================

TEST(ConceptsTest, CompileTimeValidation) {
    // These should compile without errors
    static_assert(PhysicsScalar<float>);
    static_assert(PhysicsScalar<double>);
    static_assert(Vector3D<ConceptVector3D<float>>);
    static_assert(GPUCompatible<ConceptParticleData<float>>);

    // Verify type sizes are reasonable
    static_assert(sizeof(ConceptVector3D<float>) <= 16);  // Should be compact
    static_assert(sizeof(ConceptParticleData<float>) <= 64);  // Should be cache-friendly

    SUCCEED() << "All compile-time assertions passed";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
