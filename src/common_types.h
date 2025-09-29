/**
 * PhysGrad - Common Types Header
 *
 * Provides common type definitions that work in both CUDA and C++ compilation contexts.
 */

#ifndef PHYSGRAD_COMMON_TYPES_H
#define PHYSGRAD_COMMON_TYPES_H

#include <cmath>

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
}

#endif // PHYSGRAD_COMMON_TYPES_H