/**
 * PhysGrad - Common Types Header
 *
 * Provides common type definitions that work in both CUDA and C++ compilation contexts.
 * Enhanced with C++20 concepts for type safety and automatic optimization.
 */

#ifndef PHYSGRAD_COMMON_TYPES_H
#define PHYSGRAD_COMMON_TYPES_H

#include <cmath>
#include <array>
#include <type_traits>

// Include C++20 concepts when available
#if __cplusplus >= 202002L
    #include "concepts/forward_declarations.h"
    #define PHYSGRAD_CONCEPTS_AVAILABLE
#endif

#ifdef __CUDACC__
    // CUDA compilation - use native CUDA types
    #include <cuda_runtime.h>
    #include <vector_types.h>
#else
    // C++ compilation - define compatible types
    struct float2 {
        float x, y;
        float2() : x(0.0f), y(0.0f) {}
        float2(float x_, float y_) : x(x_), y(y_) {}
    };

    struct float3 {
        float x, y, z;
        float3() : x(0.0f), y(0.0f), z(0.0f) {}
        float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    };

    struct float4 {
        float x, y, z, w;
        float4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
        float4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    };

    struct int2 {
        int x, y;
        int2() : x(0), y(0) {}
        int2(int x_, int y_) : x(x_), y(y_) {}
    };

    struct int3 {
        int x, y, z;
        int3() : x(0), y(0), z(0) {}
        int3(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    };

    struct int4 {
        int x, y, z, w;
        int4() : x(0), y(0), z(0), w(0) {}
        int4(int x_, int y_, int z_, int w_) : x(x_), y(y_), z(z_), w(w_) {}
    };

    // Helper functions for C++ compilation
    // Note: Don't define these if CUDA runtime headers might be included
    #ifndef __VECTOR_FUNCTIONS_H__
    inline float3 make_float3(float x, float y, float z) {
        return float3(x, y, z);
    }

    inline float2 make_float2(float x, float y) {
        return float2(x, y);
    }

    inline float4 make_float4(float x, float y, float z, float w) {
        return float4(x, y, z, w);
    }

    inline int3 make_int3(int x, int y, int z) {
        return int3(x, y, z);
    }
    #endif

    // Basic math operations for float3
    inline float3 operator+(const float3& a, const float3& b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline float3 operator-(const float3& a, const float3& b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    inline float3 operator*(const float3& a, float s) {
        return make_float3(a.x * s, a.y * s, a.z * s);
    }

    inline float3 operator*(float s, const float3& a) {
        return make_float3(a.x * s, a.y * s, a.z * s);
    }
#endif

// Common constants and enums that work in both contexts
namespace physgrad {

    // Boundary condition types
    enum class BoundaryType {
        OPEN,
        PERIODIC,
        REFLECTIVE
    };

    // Integration methods
    enum class IntegrationMethod {
        EULER,
        VERLET,
        RUNGE_KUTTA_4,
        LEAPFROG
    };

    // Physics constants
    constexpr float COULOMB_CONSTANT = 8.9875517923e9f;  // N⋅m²/C²
    constexpr float EPSILON_0 = 8.8541878128e-12f;       // Permittivity of free space (F/m)
    constexpr float MU_0 = 1.25663706212e-6f;            // Permeability of free space (H/m)
    constexpr float SPEED_OF_LIGHT = 299792458.0f;       // Speed of light (m/s)
    constexpr float PLANCK_CONSTANT = 6.62607015e-34f;   // Planck constant (J⋅s)
    constexpr float BOLTZMANN_CONSTANT = 1.380649e-23f;  // Boltzmann constant (J/K)

    // Utility functions
    inline float magnitude(const float3& v) {
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    inline float3 normalize(const float3& v) {
        float mag = magnitude(v);
        if (mag > 1e-10f) {
            return make_float3(v.x / mag, v.y / mag, v.z / mag);
        }
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    inline float dot(const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline float3 cross(const float3& a, const float3& b) {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    // =============================================================================
    // CONCEPT-COMPLIANT VECTOR TYPES
    // =============================================================================

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    // Concept-compliant 3D vector type that satisfies Vector3D concept
    template<concepts::PhysicsScalar T>
    class ConceptVector3D {
    public:
        using value_type = T;

        T x, y, z;

        constexpr ConceptVector3D() : x(T{0}), y(T{0}), z(T{0}) {}
        constexpr ConceptVector3D(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

        // Array-like access for concept compliance
        constexpr T& operator[](size_t i) {
            return (&x)[i];
        }

        constexpr const T& operator[](size_t i) const {
            return (&x)[i];
        }

        static constexpr size_t size() { return 3; }

        // Vector operations
        ConceptVector3D operator+(const ConceptVector3D& other) const {
            return ConceptVector3D(x + other.x, y + other.y, z + other.z);
        }

        ConceptVector3D operator-(const ConceptVector3D& other) const {
            return ConceptVector3D(x - other.x, y - other.y, z - other.z);
        }

        ConceptVector3D operator*(T scalar) const {
            return ConceptVector3D(x * scalar, y * scalar, z * scalar);
        }

        // Convert to/from float3 for compatibility
        ConceptVector3D(const float3& f3) : x(static_cast<T>(f3.x)),
                                           y(static_cast<T>(f3.y)),
                                           z(static_cast<T>(f3.z)) {}

        float3 to_float3() const {
            return make_float3(static_cast<float>(x),
                              static_cast<float>(y),
                              static_cast<float>(z));
        }
    };

    // Verify that our ConceptVector3D satisfies the Vector3D concept
    static_assert(concepts::Vector3D<ConceptVector3D<float>>);
    static_assert(concepts::Vector3D<ConceptVector3D<double>>);

    // GPU-compatible particle type that satisfies Particle concept
    template<concepts::PhysicsScalar T>
    struct ConceptParticleData {
        using scalar_type = T;
        using vector_type = ConceptVector3D<T>;

        ConceptVector3D<T> position_;
        ConceptVector3D<T> velocity_;
        T mass_;

        ConceptParticleData() : mass_(T{1}) {}
        ConceptParticleData(const ConceptVector3D<T>& pos,
                           const ConceptVector3D<T>& vel,
                           T mass)
            : position_(pos), velocity_(vel), mass_(mass) {}

        auto position() const -> ConceptVector3D<T> { return position_; }
        auto velocity() const -> ConceptVector3D<T> { return velocity_; }
        auto mass() const -> T { return mass_; }

        void set_position(const ConceptVector3D<T>& pos) { position_ = pos; }
        void set_velocity(const ConceptVector3D<T>& vel) { velocity_ = vel; }
    };

    // Verify Particle concept compliance
    static_assert(concepts::DynamicParticle<ConceptParticleData<float>>);
    static_assert(concepts::GPUCompatible<ConceptParticleData<float>>);

    // =============================================================================
    // AUTOMATIC TYPE OPTIMIZATION UTILITIES
    // =============================================================================

    // Type aliases for optimization (simplified without full type_traits)
    template<int precision_bits>
    using optimal_physics_scalar = std::conditional_t<precision_bits >= 64, double, float>;

    // Simple vector type alias
    template<concepts::PhysicsScalar T>
    using optimal_physics_vector = ConceptVector3D<T>;

#else // !PHYSGRAD_CONCEPTS_AVAILABLE
    // Fallback vector type when concepts are not available
    template<typename T>
    class ConceptVector3D {
    public:
        using value_type = T;

        T x, y, z;

        constexpr ConceptVector3D() : x(T{0}), y(T{0}), z(T{0}) {}
        constexpr ConceptVector3D(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

        // Array-like access
        constexpr T& operator[](size_t i) {
            return (&x)[i];
        }

        constexpr const T& operator[](size_t i) const {
            return (&x)[i];
        }

        static constexpr size_t size() { return 3; }

        // Vector operations
        ConceptVector3D operator+(const ConceptVector3D& other) const {
            return ConceptVector3D(x + other.x, y + other.y, z + other.z);
        }

        ConceptVector3D operator-(const ConceptVector3D& other) const {
            return ConceptVector3D(x - other.x, y - other.y, z - other.z);
        }

        ConceptVector3D operator*(T scalar) const {
            return ConceptVector3D(x * scalar, y * scalar, z * scalar);
        }

        // Convert to/from float3 for compatibility
        ConceptVector3D(const float3& f3) : x(static_cast<T>(f3.x)),
                                           y(static_cast<T>(f3.y)),
                                           z(static_cast<T>(f3.z)) {}

        float3 to_float3() const {
            return make_float3(static_cast<float>(x),
                              static_cast<float>(y),
                              static_cast<float>(z));
        }
    };

    // Fallback particle type when concepts are not available
    template<typename T>
    struct ConceptParticleData {
        using scalar_type = T;
        using vector_type = ConceptVector3D<T>;

        ConceptVector3D<T> position_;
        ConceptVector3D<T> velocity_;
        T mass_;

        ConceptParticleData() : mass_(T{1}) {}
        ConceptParticleData(const ConceptVector3D<T>& pos,
                           const ConceptVector3D<T>& vel,
                           T mass)
            : position_(pos), velocity_(vel), mass_(mass) {}

        auto position() const -> ConceptVector3D<T> { return position_; }
        auto velocity() const -> ConceptVector3D<T> { return velocity_; }
        auto mass() const -> T { return mass_; }

        void set_position(const ConceptVector3D<T>& pos) { position_ = pos; }
        void set_velocity(const ConceptVector3D<T>& vel) { velocity_ = vel; }
    };

    // Type aliases for optimization (simplified without full type_traits)
    template<int precision_bits>
    using optimal_physics_scalar = std::conditional_t<precision_bits >= 64, double, float>;

    // Simple vector type alias
    template<typename T>
    using optimal_physics_vector = ConceptVector3D<T>;

#endif // PHYSGRAD_CONCEPTS_AVAILABLE

    // =============================================================================
    // LEGACY COMPATIBILITY FUNCTIONS
    // =============================================================================

    // Convert between legacy and concept-aware types
    template<typename T>
    float3 to_legacy_float3(const T& vec) {
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
        if constexpr (concepts::Vector3D<T>) {
            return make_float3(static_cast<float>(vec[0]),
                              static_cast<float>(vec[1]),
                              static_cast<float>(vec[2]));
        } else {
#endif
            return make_float3(vec.x, vec.y, vec.z);
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
        }
#endif
    }

    template<typename T>
    T from_legacy_float3(const float3& f3) {
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
        if constexpr (concepts::Vector3D<T>) {
            return T{static_cast<typename T::value_type>(f3.x),
                    static_cast<typename T::value_type>(f3.y),
                    static_cast<typename T::value_type>(f3.z)};
        } else {
#endif
            return T{f3.x, f3.y, f3.z};
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
        }
#endif
    }
}

#endif // PHYSGRAD_COMMON_TYPES_H