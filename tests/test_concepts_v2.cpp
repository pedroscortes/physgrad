/**
 * PhysGrad C++20 Concepts Validation Test (Version 2)
 */

#include <iostream>
#include <vector>
#include <array>
#include <complex>
#include <concepts>
#include <type_traits>
#include <iomanip>

#include "src/concepts/physics_concepts_v2.h"

using namespace physgrad::concepts::v2;

template<typename T>
struct TestVector3D {
    using value_type = T;
    T data[3];

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    TestVector3D operator+(const TestVector3D& other) const {
        return {data[0] + other.data[0], data[1] + other.data[1], data[2] + other.data[2]};
    }

    TestVector3D operator-(const TestVector3D& other) const {
        return {data[0] - other.data[0], data[1] - other.data[1], data[2] - other.data[2]};
    }
};

template<typename T>
struct TestParticle {
    using scalar_type = T;
    using vector_type = TestVector3D<T>;

    vector_type pos;
    vector_type vel;
    T m;

    vector_type position() const { return pos; }
    vector_type velocity() const { return vel; }
    T mass() const { return m; }
};

template<typename T>
struct TestChargedParticle : TestParticle<T> {
    T q;
    T charge() const { return q; }
};

template<typename T>
struct TestTensor {
    using scalar_type = T;
    T data[3][3];

    size_t rank() const { return 2; }
    size_t size() const { return 9; }
};

struct GPUStruct {
    float x, y, z, w;
};

struct NonGPUStruct {
    float x, y, z;
    virtual ~NonGPUStruct() = default;
};

bool testScalarConcepts() {
    std::cout << "Testing scalar concepts..." << std::endl;

    static_assert(PhysicsScalar<float>);
    static_assert(PhysicsScalar<double>);
    static_assert(!PhysicsScalar<int>);

    static_assert(!HighPrecisionScalar<float>);
    static_assert(HighPrecisionScalar<double>);

    static_assert(ComplexNumber<std::complex<float>>);
    static_assert(ComplexNumber<std::complex<double>>);

    std::cout << "✓ Scalar concepts validated" << std::endl;
    return true;
}

bool testVectorTensorConcepts() {
    std::cout << "Testing vector and tensor concepts..." << std::endl;

    static_assert(Vector3D<TestVector3D<float>>);
    static_assert(Vector3D<TestVector3D<double>>);
    static_assert(!Vector3D<float>);

    static_assert(Tensor<TestTensor<float>>);
    static_assert(!Tensor<float>);

    std::cout << "✓ Vector and tensor concepts validated" << std::endl;
    return true;
}

bool testParticleConcepts() {
    std::cout << "Testing particle concepts..." << std::endl;

    static_assert(Particle<TestParticle<float>>);
    static_assert(Particle<TestChargedParticle<float>>);

    static_assert(ChargedParticle<TestChargedParticle<float>>);
    static_assert(!ChargedParticle<TestParticle<float>>);

    std::cout << "✓ Particle concepts validated" << std::endl;
    return true;
}

bool testGPUConcepts() {
    std::cout << "Testing GPU concepts..." << std::endl;

    static_assert(GPUCompatible<GPUStruct>);
    static_assert(!GPUCompatible<NonGPUStruct>);
    static_assert(GPUCompatible<float>);
    static_assert(GPUCompatible<std::array<float, 4>>);

    struct Vec16 { float data[4]; };
    static_assert(VectorizedCompatible<Vec16>);
    static_assert(!VectorizedCompatible<float>);

    std::cout << "✓ GPU concepts validated" << std::endl;
    return true;
}

bool testConceptConstraints() {
    std::cout << "Testing concept-constrained functions..." << std::endl;

    auto physics_compute = []<PhysicsScalar T>(T value) {
        return std::sin(value) * std::cos(value);
    };

    float result1 = physics_compute(1.0f);
    double result2 = physics_compute(1.0);

    auto particle_energy = []<Particle P>(const P& p) {
        auto vel = p.velocity();
        return 0.5f * p.mass() * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    };

    TestParticle<float> particle{{1,2,3}, {4,5,6}, 2.0f};
    float energy = particle_energy(particle);

    std::cout << "✓ Concept constraints working" << std::endl;
    return true;
}

bool testHelperTemplates() {
    std::cout << "Testing helper templates..." << std::endl;

    static_assert(is_physics_scalar_v<float>);
    static_assert(is_physics_scalar_v<double>);
    static_assert(!is_physics_scalar_v<int>);

    static_assert(is_gpu_compatible_v<GPUStruct>);
    static_assert(!is_gpu_compatible_v<NonGPUStruct>);

    static_assert(std::same_as<optimal_scalar_t<double>, double>);
    static_assert(std::same_as<optimal_scalar_t<float>, float>);
    static_assert(std::same_as<optimal_scalar_t<int>, float>);

    std::cout << "✓ Helper templates validated" << std::endl;
    return true;
}

int main() {
    std::cout << "PhysGrad C++20 Concepts Validation (Version 2)" << std::endl;
    std::cout << "==============================================" << std::endl << std::endl;

    std::cout << "C++ Standard: " << __cplusplus << std::endl;
#ifdef __cpp_concepts
    std::cout << "C++20 Concepts: Supported (version " << __cpp_concepts << ")" << std::endl << std::endl;
#else
    std::cout << "C++20 Concepts: Not supported" << std::endl << std::endl;
    return 1;
#endif

    bool all_passed = true;

    all_passed &= testScalarConcepts();
    std::cout << std::endl;

    all_passed &= testVectorTensorConcepts();
    std::cout << std::endl;

    all_passed &= testParticleConcepts();
    std::cout << std::endl;

    all_passed &= testGPUConcepts();
    std::cout << std::endl;

    all_passed &= testConceptConstraints();
    std::cout << std::endl;

    all_passed &= testHelperTemplates();
    std::cout << std::endl;

    if (all_passed) {
        std::cout << "✓ All C++20 concepts tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Some C++20 concepts tests FAILED!" << std::endl;
        return 1;
    }
}