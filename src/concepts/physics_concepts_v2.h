/**
 * PhysGrad - C++20 Physics Concepts (Version 2)
 *
 * Refined concept definitions that compile correctly with C++20
 */

#pragma once

#include <concepts>
#include <type_traits>
#include <cmath>
#include <array>
#include <complex>

namespace physgrad::concepts::v2 {

// =============================================================================
// NUMERIC CONCEPTS
// =============================================================================

template<typename T>
concept PhysicsScalar = std::floating_point<T>;

template<typename T>
concept HighPrecisionScalar = PhysicsScalar<T> && sizeof(T) >= 8;

template<typename T>
concept ComplexNumber = requires(T z) {
    typename T::value_type;
    { z.real() } -> std::convertible_to<typename T::value_type>;
    { z.imag() } -> std::convertible_to<typename T::value_type>;
};

// =============================================================================
// VECTOR CONCEPTS
// =============================================================================

template<typename V>
concept Vector3D = requires(V v, size_t i) {
    typename V::value_type;
    { v[i] } -> std::convertible_to<typename V::value_type>;
    { v + v } -> std::convertible_to<V>;
    { v - v } -> std::convertible_to<V>;
};

template<typename T>
concept Tensor = requires(T t) {
    typename T::scalar_type;
    { t.rank() } -> std::convertible_to<size_t>;
    { t.size() } -> std::convertible_to<size_t>;
};

// =============================================================================
// PARTICLE CONCEPTS
// =============================================================================

template<typename P>
concept Particle = requires(P p) {
    typename P::scalar_type;
    typename P::vector_type;
    { p.position() } -> std::convertible_to<typename P::vector_type>;
    { p.velocity() } -> std::convertible_to<typename P::vector_type>;
    { p.mass() } -> std::convertible_to<typename P::scalar_type>;
};

template<typename P>
concept ChargedParticle = Particle<P> && requires(P p) {
    { p.charge() } -> std::convertible_to<typename P::scalar_type>;
};

// =============================================================================
// FIELD CONCEPTS
// =============================================================================

template<typename F>
concept Field = requires(F f) {
    typename F::value_type;
    typename F::position_type;
    { f.dimension() } -> std::convertible_to<size_t>;
};

// =============================================================================
// INTEGRATOR CONCEPTS
// =============================================================================

template<typename I>
concept Integrator = requires(I integrator) {
    typename I::scalar_type;
    { I::order } -> std::convertible_to<int>;
};

template<typename I>
concept SymplecticIntegrator = Integrator<I> && requires {
    { I::preserves_energy } -> std::convertible_to<bool>;
};

// =============================================================================
// GPU CONCEPTS
// =============================================================================

template<typename T>
concept GPUCompatible = std::is_trivially_copyable_v<T> &&
                        std::is_standard_layout_v<T>;

template<typename T>
concept VectorizedCompatible = GPUCompatible<T> &&
                              (sizeof(T) == 16 || sizeof(T) == 32 || sizeof(T) == 64);

// =============================================================================
// MATERIAL CONCEPTS
// =============================================================================

template<typename M>
concept MaterialModel = requires(M material) {
    typename M::scalar_type;
    typename M::tensor_type;
    { material.density() } -> std::convertible_to<typename M::scalar_type>;
};

template<typename M>
concept ElasticMaterial = MaterialModel<M> && requires(M material) {
    { material.young_modulus() } -> std::convertible_to<typename M::scalar_type>;
    { material.poisson_ratio() } -> std::convertible_to<typename M::scalar_type>;
};

// =============================================================================
// SYSTEM CONCEPTS
// =============================================================================

template<typename S>
concept PhysicalSystem = requires(S system) {
    typename S::state_type;
    typename S::scalar_type;
    { system.time() } -> std::convertible_to<typename S::scalar_type>;
};

template<typename S>
concept ConservativeSystem = PhysicalSystem<S> && requires(S system) {
    { system.total_energy() } -> std::convertible_to<typename S::scalar_type>;
    { system.kinetic_energy() } -> std::convertible_to<typename S::scalar_type>;
    { system.potential_energy() } -> std::convertible_to<typename S::scalar_type>;
};

// =============================================================================
// VALIDATION CONCEPTS
// =============================================================================

template<typename T>
concept PhysicsValidatable = requires(T t) {
    { t.is_valid() } -> std::convertible_to<bool>;
};

// =============================================================================
// HELPER TEMPLATES
// =============================================================================

template<typename T>
struct is_physics_scalar : std::bool_constant<PhysicsScalar<T>> {};

template<typename T>
inline constexpr bool is_physics_scalar_v = is_physics_scalar<T>::value;

template<typename T>
struct is_gpu_compatible : std::bool_constant<GPUCompatible<T>> {};

template<typename T>
inline constexpr bool is_gpu_compatible_v = is_gpu_compatible<T>::value;

// Type selection based on concepts
template<typename T>
using optimal_scalar_t = std::conditional_t<
    HighPrecisionScalar<T>, T,
    std::conditional_t<PhysicsScalar<T>, T, float>
>;

} // namespace physgrad::concepts::v2