/**
 * PhysGrad - Advanced C++20 Physics Concepts
 *
 * Comprehensive concept definitions for type-safe physics simulations
 * with compile-time validation and optimization hints.
 */

#pragma once

#include <concepts>
#include <type_traits>
#include <ranges>
#include <numbers>
#include <array>
#include <vector>
#include <memory>
#include <functional>

namespace physgrad::concepts {

// =============================================================================
// FOUNDATIONAL NUMERIC CONCEPTS
// =============================================================================

/**
 * Concept for types that can represent physics scalars with required precision
 */
template<typename T>
concept PhysicsScalar = std::floating_point<T> &&
    requires(T a, T b) {
        { a + b } -> std::convertible_to<T>;
        { a - b } -> std::convertible_to<T>;
        { a * b } -> std::convertible_to<T>;
        { a / b } -> std::convertible_to<T>;
        { std::abs(a) } -> std::convertible_to<T>;
        { std::sqrt(a) } -> std::convertible_to<T>;
        { std::sin(a) } -> std::convertible_to<T>;
        { std::cos(a) } -> std::convertible_to<T>;
    };

/**
 * High-precision scalar for critical calculations
 */
template<typename T>
concept HighPrecisionScalar = PhysicsScalar<T> &&
    (sizeof(T) >= 8) &&
    std::numeric_limits<T>::digits >= 53;

/**
 * Concept for complex numbers in physics
 */
template<typename T>
concept ComplexNumber = requires(T z) {
    typename T::value_type;
    requires PhysicsScalar<typename T::value_type>;
    { z.real() } -> std::convertible_to<typename T::value_type>;
    { z.imag() } -> std::convertible_to<typename T::value_type>;
    { std::conj(z) } -> std::same_as<T>;
    { std::abs(z) } -> std::convertible_to<typename T::value_type>;
};

// =============================================================================
// VECTOR AND TENSOR CONCEPTS
// =============================================================================

/**
 * Concept for 3D vectors in physics simulations
 */
template<typename V>
concept Vector3D = requires(V v, typename V::value_type s, size_t i) {
    typename V::value_type;
    requires PhysicsScalar<typename V::value_type>;
    { v[i] } -> std::convertible_to<typename V::value_type>;
    { v + v } -> std::convertible_to<V>;
    { v - v } -> std::convertible_to<V>;
    { v * s } -> std::convertible_to<V>;
    { v / s } -> std::convertible_to<V>;
    { V{} } -> std::same_as<V>;
};

/**
 * Concept for mathematical tensors
 */
template<typename T>
concept Tensor = requires(T t) {
    typename T::scalar_type;
    typename T::shape_type;
    requires PhysicsScalar<typename T::scalar_type>;
    { t.rank() } -> std::convertible_to<size_t>;
    { t.shape() } -> std::convertible_to<typename T::shape_type>;
    { t.size() } -> std::convertible_to<size_t>;
};

/**
 * Concept for symmetric tensors (stress, strain, etc.)
 */
template<typename T>
concept SymmetricTensor = Tensor<T> && requires(T t, size_t i, size_t j) {
    { t(i, j) } -> std::same_as<decltype(t(j, i))>;
};

// =============================================================================
// PARTICLE PHYSICS CONCEPTS
// =============================================================================

/**
 * Basic particle concept
 */
template<typename P>
concept Particle = requires(P p) {
    typename P::scalar_type;
    typename P::vector_type;
    requires PhysicsScalar<typename P::scalar_type>;
    requires Vector3D<typename P::vector_type>;
    { p.position() } -> std::convertible_to<typename P::vector_type>;
    { p.velocity() } -> std::convertible_to<typename P::vector_type>;
    { p.mass() } -> std::convertible_to<typename P::scalar_type>;
};

/**
 * Particle with electromagnetic properties
 */
template<typename P>
concept ChargedParticle = Particle<P> && requires(P p) {
    { p.charge() } -> std::convertible_to<typename P::scalar_type>;
};

/**
 * Quantum mechanical particle
 */
template<typename P>
concept QuantumParticle = Particle<P> && requires(P p) {
    typename P::wavefunction_type;
    { p.wavefunction() } -> std::convertible_to<typename P::wavefunction_type>;
    { p.spin() } -> std::convertible_to<typename P::scalar_type>;
};

/**
 * Deformable particle for continuum mechanics
 */
template<typename P>
concept DeformableParticle = Particle<P> && requires(P p) {
    typename P::tensor_type;
    { p.deformation_gradient() } -> std::convertible_to<typename P::tensor_type>;
    { p.stress() } -> std::convertible_to<typename P::tensor_type>;
};

// =============================================================================
// FIELD CONCEPTS
// =============================================================================

/**
 * Concept for physical fields (scalar, vector, tensor)
 */
template<typename F>
concept Field = requires(F f, typename F::position_type x) {
    typename F::value_type;
    typename F::position_type;
    requires Vector3D<typename F::position_type>;
    { f(x) } -> std::convertible_to<typename F::value_type>;
    { f.dimension() } -> std::convertible_to<size_t>;
};

/**
 * Differentiable field concept
 */
template<typename F>
concept DifferentiableField = Field<F> && requires(F f, typename F::position_type x) {
    typename F::gradient_type;
    { f.gradient(x) } -> std::convertible_to<typename F::gradient_type>;
    { f.laplacian(x) } -> std::convertible_to<typename F::value_type>;
};

/**
 * Time-varying field
 */
template<typename F>
concept TimeVaryingField = Field<F> && requires(F f, typename F::position_type x,
                                                typename F::scalar_type t) {
    typename F::scalar_type;
    requires PhysicsScalar<typename F::scalar_type>;
    { f(x, t) } -> std::convertible_to<typename F::value_type>;
    { f.time_derivative(x, t) } -> std::convertible_to<typename F::value_type>;
};

// =============================================================================
// FORCE AND INTERACTION CONCEPTS
// =============================================================================

/**
 * Concept for force fields
 */
template<typename F>
concept ForceField = requires(F f, Particle<auto> p) {
    typename F::vector_type;
    requires Vector3D<typename F::vector_type>;
    { f.force_on(p) } -> std::convertible_to<typename F::vector_type>;
    { f.is_conservative() } -> std::convertible_to<bool>;
};

/**
 * Pairwise interaction between particles
 */
template<typename I>
concept PairwiseInteraction = requires(I interaction, Particle<auto> p1, Particle<auto> p2) {
    typename I::scalar_type;
    typename I::vector_type;
    requires PhysicsScalar<typename I::scalar_type>;
    requires Vector3D<typename I::vector_type>;
    { interaction.force(p1, p2) } -> std::convertible_to<typename I::vector_type>;
    { interaction.energy(p1, p2) } -> std::convertible_to<typename I::scalar_type>;
    { interaction.cutoff_distance() } -> std::convertible_to<typename I::scalar_type>;
};

// =============================================================================
// INTEGRATOR CONCEPTS
// =============================================================================

/**
 * Basic time integrator
 */
template<typename I>
concept Integrator = requires(I integrator) {
    typename I::scalar_type;
    requires PhysicsScalar<typename I::scalar_type>;
    { integrator.order } -> std::convertible_to<int>;
    { integrator.is_explicit() } -> std::convertible_to<bool>;
    { integrator.is_implicit() } -> std::convertible_to<bool>;
};

/**
 * Symplectic integrator for Hamiltonian systems
 */
template<typename I>
concept SymplecticIntegrator = Integrator<I> && requires(I integrator) {
    { integrator.preserves_energy() } -> std::convertible_to<bool>;
    { integrator.preserves_phase_space() } -> std::convertible_to<bool>;
};

/**
 * Adaptive timestep integrator
 */
template<typename I>
concept AdaptiveIntegrator = Integrator<I> &&
    requires(I integrator, typename I::scalar_type error) {
    { integrator.estimate_error() } -> std::convertible_to<typename I::scalar_type>;
    { integrator.adjust_timestep(error) } -> std::convertible_to<typename I::scalar_type>;
};

// =============================================================================
// CONSTRAINT CONCEPTS
// =============================================================================

/**
 * Geometric constraint
 */
template<typename C>
concept Constraint = requires(C constraint, Particle<auto> p) {
    typename C::scalar_type;
    requires PhysicsScalar<typename C::scalar_type>;
    { constraint.evaluate(p) } -> std::convertible_to<typename C::scalar_type>;
    { constraint.gradient(p) } -> std::convertible_to<typename C::scalar_type>;
    { constraint.is_satisfied(p) } -> std::convertible_to<bool>;
};

/**
 * Bilateral (equality) constraint
 */
template<typename C>
concept BilateralConstraint = Constraint<C> && requires(C c) {
    { c.target_value() } -> std::convertible_to<typename C::scalar_type>;
};

/**
 * Unilateral (inequality) constraint
 */
template<typename C>
concept UnilateralConstraint = Constraint<C> && requires(C c) {
    { c.is_active() } -> std::convertible_to<bool>;
    { c.normal_impulse() } -> std::convertible_to<typename C::scalar_type>;
};

// =============================================================================
// MATERIAL MODELS
// =============================================================================

/**
 * Constitutive material model
 */
template<typename M>
concept MaterialModel = requires(M material) {
    typename M::scalar_type;
    typename M::tensor_type;
    requires PhysicsScalar<typename M::scalar_type>;
    requires Tensor<typename M::tensor_type>;
    { material.density() } -> std::convertible_to<typename M::scalar_type>;
    { material.compute_stress(typename M::tensor_type{}) } ->
        std::convertible_to<typename M::tensor_type>;
};

/**
 * Elastic material
 */
template<typename M>
concept ElasticMaterial = MaterialModel<M> && requires(M material) {
    { material.young_modulus() } -> std::convertible_to<typename M::scalar_type>;
    { material.poisson_ratio() } -> std::convertible_to<typename M::scalar_type>;
    { material.shear_modulus() } -> std::convertible_to<typename M::scalar_type>;
    { material.bulk_modulus() } -> std::convertible_to<typename M::scalar_type>;
};

/**
 * Plastic material with yield criterion
 */
template<typename M>
concept PlasticMaterial = MaterialModel<M> && requires(M material, typename M::tensor_type stress) {
    { material.yield_stress() } -> std::convertible_to<typename M::scalar_type>;
    { material.is_yielding(stress) } -> std::convertible_to<bool>;
    { material.plastic_flow(stress) } -> std::convertible_to<typename M::tensor_type>;
};

// =============================================================================
// SOLVER CONCEPTS
// =============================================================================

/**
 * Linear system solver
 */
template<typename S>
concept LinearSolver = requires(S solver) {
    typename S::matrix_type;
    typename S::vector_type;
    typename S::scalar_type;
    { solver.solve(typename S::matrix_type{}, typename S::vector_type{}) } ->
        std::convertible_to<typename S::vector_type>;
    { solver.tolerance() } -> std::convertible_to<typename S::scalar_type>;
    { solver.max_iterations() } -> std::convertible_to<size_t>;
};

/**
 * Iterative solver with convergence monitoring
 */
template<typename S>
concept IterativeSolver = LinearSolver<S> && requires(S solver) {
    { solver.residual() } -> std::convertible_to<typename S::scalar_type>;
    { solver.iteration_count() } -> std::convertible_to<size_t>;
    { solver.has_converged() } -> std::convertible_to<bool>;
};

// =============================================================================
// SYSTEM CONCEPTS
// =============================================================================

/**
 * Physical system that can be simulated
 */
template<typename S>
concept PhysicalSystem = requires(S system) {
    typename S::state_type;
    typename S::scalar_type;
    requires PhysicsScalar<typename S::scalar_type>;
    { system.state() } -> std::convertible_to<typename S::state_type>;
    { system.time() } -> std::convertible_to<typename S::scalar_type>;
    { system.advance(typename S::scalar_type{}) } -> std::same_as<void>;
};

/**
 * System with energy conservation
 */
template<typename S>
concept ConservativeSystem = PhysicalSystem<S> && requires(S system) {
    { system.total_energy() } -> std::convertible_to<typename S::scalar_type>;
    { system.kinetic_energy() } -> std::convertible_to<typename S::scalar_type>;
    { system.potential_energy() } -> std::convertible_to<typename S::scalar_type>;
};

/**
 * Coupled multi-physics system
 */
template<typename S>
concept CoupledSystem = PhysicalSystem<S> && requires(S system) {
    typename S::subsystem_type;
    { system.num_subsystems() } -> std::convertible_to<size_t>;
    { system.subsystem(size_t{}) } -> std::convertible_to<typename S::subsystem_type>;
    { system.coupling_strength() } -> std::convertible_to<typename S::scalar_type>;
};

// =============================================================================
// GPU/PARALLEL CONCEPTS
// =============================================================================

/**
 * Type that can be efficiently used on GPU
 */
template<typename T>
concept GPUCompatible = std::is_trivially_copyable_v<T> &&
                        std::is_standard_layout_v<T> &&
                        (alignof(T) % 4 == 0);

/**
 * Type that supports vectorized operations
 */
template<typename T>
concept VectorizedCompatible = GPUCompatible<T> &&
                               (sizeof(T) % 16 == 0) &&
                               (alignof(T) >= 16);

/**
 * Thread-safe type for parallel execution
 */
template<typename T>
concept ThreadSafe = requires(T t) {
    { t.lock() } -> std::same_as<void>;
    { t.unlock() } -> std::same_as<void>;
    { t.try_lock() } -> std::convertible_to<bool>;
} || std::atomic<T>::is_always_lock_free;

// =============================================================================
// DIFFERENTIABILITY CONCEPTS
// =============================================================================

/**
 * Type that supports automatic differentiation
 */
template<typename T>
concept Differentiable = requires(T t) {
    typename T::gradient_type;
    typename T::scalar_type;
    { t.value() } -> std::convertible_to<typename T::scalar_type>;
    { t.gradient() } -> std::convertible_to<typename T::gradient_type>;
    { t.requires_grad() } -> std::convertible_to<bool>;
};

/**
 * Function that can be differentiated
 */
template<typename F, typename Input>
concept DifferentiableFunction = requires(F f, Input x) {
    typename F::output_type;
    typename F::jacobian_type;
    { f(x) } -> std::convertible_to<typename F::output_type>;
    { f.jacobian(x) } -> std::convertible_to<typename F::jacobian_type>;
};

// =============================================================================
// VALIDATION CONCEPTS
// =============================================================================

/**
 * Type that can validate its physical correctness
 */
template<typename T>
concept PhysicsValidatable = requires(T t) {
    { t.is_valid() } -> std::convertible_to<bool>;
    { t.check_conservation_laws() } -> std::convertible_to<bool>;
    { t.validate_bounds() } -> std::convertible_to<bool>;
};

/**
 * Simulation component with error estimation
 */
template<typename T>
concept ErrorEstimatable = requires(T t) {
    typename T::scalar_type;
    { t.estimate_error() } -> std::convertible_to<typename T::scalar_type>;
    { t.error_tolerance() } -> std::convertible_to<typename T::scalar_type>;
    { t.refine() } -> std::same_as<void>;
};

// =============================================================================
// CONCEPT VALIDATION HELPERS
// =============================================================================

/**
 * Compile-time validation of concept requirements
 */
template<template<typename> typename Concept, typename T>
constexpr bool satisfies = requires { requires Concept<T>; };

/**
 * Helper to check multiple concepts
 */
template<typename T, template<typename> typename... Concepts>
constexpr bool satisfies_all = (satisfies<Concepts, T> && ...);

/**
 * Helper to check at least one concept
 */
template<typename T, template<typename> typename... Concepts>
constexpr bool satisfies_any = (satisfies<Concepts, T> || ...);

} // namespace physgrad::concepts