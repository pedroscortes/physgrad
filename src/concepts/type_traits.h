#pragma once

#include "physics_concepts.h"
#include <type_traits>
#include <utility>
#include <tuple>
#include <array>
#include <vector>

namespace physgrad::type_traits {

// =============================================================================
// CONCEPT-BASED TYPE DETECTION
// =============================================================================

// Detect scalar precision level
template<typename T>
struct scalar_precision {
    static constexpr int value =
        concepts::HighPrecisionScalar<T> ? 64 :
        concepts::PhysicsScalar<T> ? 32 : 0;
};

template<typename T>
constexpr int scalar_precision_v = scalar_precision<T>::value;

// Detect optimal GPU memory layout
template<typename T>
struct gpu_layout_type {
    using type = std::conditional_t<
        concepts::VectorizedCompatible<T>, T,
        std::conditional_t<concepts::GPUCompatible<T>, T, void>
    >;
    static constexpr bool is_optimal = concepts::VectorizedCompatible<T>;
};

template<typename T>
using gpu_layout_type_t = typename gpu_layout_type<T>::type;

// Detect physics system classification
template<typename T>
struct physics_system_traits {
    static constexpr bool is_conservative = requires {
        typename T::is_conservative;
        T::is_conservative::value;
    };

    static constexpr bool is_symplectic = requires {
        typename T::is_symplectic;
        T::is_symplectic::value;
    };

    static constexpr bool is_variational = requires {
        typename T::is_variational;
        T::is_variational::value;
    };

    static constexpr bool supports_quantum = concepts::QuantumParticle<T>;
    static constexpr bool supports_coupling = concepts::CoupledSystem<T>;
};

// =============================================================================
// AUTOMATIC CONCEPT VALIDATION
// =============================================================================

// Validate physics correctness at compile time
template<concepts::PhysicsScalar T>
struct physics_validator {
    // Check numerical precision requirements
    static_assert(std::numeric_limits<T>::is_iec559,
                 "Physics scalar must follow IEEE 754 standard");

    static_assert(std::numeric_limits<T>::digits >= 23,
                 "Insufficient precision for physics calculations");

    static_assert(!std::numeric_limits<T>::is_integer,
                 "Physics calculations require floating-point arithmetic");

    // Ensure proper infinity/NaN handling
    static_assert(std::numeric_limits<T>::has_infinity,
                 "Physics scalar must support infinity representation");

    static_assert(std::numeric_limits<T>::has_quiet_NaN,
                 "Physics scalar must support NaN for error handling");

    using type = T;
    static constexpr bool is_valid = true;
};

// Validate integrator properties
template<concepts::Integrator T>
struct integrator_validator {
    static_assert(T::order >= 1, "Integrator must have positive order");
    static_assert(T::order <= 8, "Very high-order integrators may be unstable");

    // Symplectic integrators should have even order for optimal performance
    static constexpr bool symplectic_order_warning =
        concepts::SymplecticIntegrator<T> && (T::order % 2 != 0);

    using type = T;
    static constexpr bool is_valid = true;
};

// Validate memory layout for GPU performance
template<concepts::GPUCompatible T>
struct gpu_memory_validator {
    static_assert(std::is_trivially_copyable_v<T>,
                 "GPU types must be trivially copyable");

    static_assert(std::is_standard_layout_v<T>,
                 "GPU types must have standard layout");

    static_assert(alignof(T) >= 4,
                 "GPU types should be at least 4-byte aligned");

    // Warn about suboptimal alignment
    static constexpr bool alignment_warning = (alignof(T) % 16 != 0);

    using type = T;
    static constexpr bool is_valid = true;
};

// =============================================================================
// PERFORMANCE OPTIMIZATION TRAITS
// =============================================================================

// Determine optimal CUDA block size based on type
template<typename T>
struct cuda_block_size {
    static constexpr int value =
        sizeof(T) <= 16 ? 256 :  // Small types: more threads
        sizeof(T) <= 64 ? 128 :  // Medium types: balanced
        64;                      // Large types: fewer threads
};

template<typename T>
constexpr int cuda_block_size_v = cuda_block_size<T>::value;

// Determine vectorization factor
template<typename T>
struct vectorization_factor {
    static constexpr int value =
        concepts::VectorizedCompatible<T> ? (64 / sizeof(T)) : 1;
};

template<typename T>
constexpr int vectorization_factor_v = vectorization_factor<T>::value;

// Memory coalescing optimization
template<typename T>
struct memory_coalescing {
    static constexpr bool is_coalescable =
        concepts::GPUCompatible<T> && (sizeof(T) % 4 == 0);

    static constexpr int optimal_stride =
        is_coalescable ? (128 / sizeof(T)) : 1; // 128-byte cache line
};

// =============================================================================
// AUTOMATIC TYPE ADAPTATION
// =============================================================================

// Automatically choose best scalar type for given precision requirements
template<int required_precision>
struct optimal_scalar {
    using type = std::conditional_t<
        required_precision >= 64, double,
        std::conditional_t<required_precision >= 32, float, void>
    >;
};

template<int precision>
using optimal_scalar_t = typename optimal_scalar<precision>::type;

// Choose best vector type for 3D physics
template<concepts::PhysicsScalar T>
struct optimal_vector3d {
    using type = std::conditional_t<
        concepts::VectorizedCompatible<std::array<T, 4>>,
        std::array<T, 4>,  // Use 4-component for better alignment
        std::array<T, 3>   // Fallback to 3-component
    >;
};

template<concepts::PhysicsScalar T>
using optimal_vector3d_t = typename optimal_vector3d<T>::type;

// Choose best container for particle data
template<concepts::Particle ParticleT, size_t expected_count>
struct optimal_particle_container {
    using type = std::conditional_t<
        (expected_count > 10000) && concepts::VectorizedCompatible<ParticleT>,
        std::vector<ParticleT>,  // Large datasets: standard vector
        std::array<ParticleT, expected_count> // Small datasets: stack allocation
    >;
};

// =============================================================================
// PHYSICS-SPECIFIC TYPE UTILITIES
// =============================================================================

// Extract physics properties from types
template<typename T>
struct physics_properties {
    using scalar_type = std::conditional_t<
        requires { typename T::scalar_type; },
        typename T::scalar_type,
        void
    >;

    using vector_type = std::conditional_t<
        requires { typename T::vector_type; },
        typename T::vector_type,
        void
    >;

    static constexpr bool has_mass = requires(T t) { t.mass(); };
    static constexpr bool has_charge = requires(T t) { t.charge(); };
    static constexpr bool has_position = requires(T t) { t.position(); };
    static constexpr bool has_velocity = requires(T t) { t.velocity(); };
    static constexpr bool has_force = requires(T t) { t.force(); };
};

// Determine coupling compatibility between physics systems
template<typename System1, typename System2>
struct coupling_compatibility {
    static constexpr bool compatible =
        concepts::CoupledSystem<System1> &&
        concepts::CoupledSystem<System2> &&
        std::convertible_to<
            typename System1::boundary_type,
            typename System2::boundary_type
        >;

    using common_boundary_type = std::conditional_t<
        compatible,
        std::common_type_t<
            typename System1::boundary_type,
            typename System2::boundary_type
        >,
        void
    >;
};

// =============================================================================
// DIFFERENTIABILITY UTILITIES
// =============================================================================

// Check if a computation chain preserves differentiability
template<typename... Transforms>
struct differentiability_chain {
    static constexpr bool preserves_gradients =
        (concepts::DifferentiableFunction<Transforms, void> && ...);

    // Compose gradient types through the chain
    template<typename InputT>
    using gradient_type = typename std::tuple_element_t<
        sizeof...(Transforms) - 1,
        std::tuple<typename Transforms::template gradient_type<InputT>...>
    >;
};

// Automatic gradient type deduction
template<concepts::Differentiable T>
struct gradient_traits {
    using gradient_type = typename T::gradient_type;
    using scalar_type = typename physics_properties<T>::scalar_type;

    static constexpr bool supports_higher_order = requires(T t) {
        typename T::hessian_type;
        t.hessian();
    };
};

// =============================================================================
// COMPILE-TIME PHYSICS VALIDATION
// =============================================================================

// Validate conservation laws can be checked
template<typename SystemT>
struct conservation_validator {
    static constexpr bool can_check_energy =
        concepts::PhysicsValidatable<SystemT> &&
        requires(SystemT s) { s.check_energy_conservation(); };

    static constexpr bool can_check_momentum =
        concepts::PhysicsValidatable<SystemT> &&
        requires(SystemT s) { s.check_momentum_conservation(); };

    static constexpr bool can_check_angular_momentum =
        requires(SystemT s) { s.check_angular_momentum_conservation(); };

    // Ensure at least one conservation law can be verified
    static_assert(can_check_energy || can_check_momentum,
                 "System must support at least one conservation law check");
};

// Validate numerical stability requirements
template<concepts::Integrator IntegratorT, concepts::PhysicsScalar ScalarT>
struct stability_validator {
    static constexpr ScalarT max_timestep = IntegratorT{}.max_stable_timestep();

    // CFL-like conditions for stability
    static_assert(max_timestep > ScalarT{0},
                 "Integrator must have positive stable timestep");

    // Symplectic integrators should be unconditionally stable for Hamiltonian systems
    static constexpr bool unconditionally_stable =
        concepts::SymplecticIntegrator<IntegratorT>;
};

// =============================================================================
// METAPROGRAMMING UTILITIES
// =============================================================================

// Generate optimal data structures based on physics requirements
template<concepts::PhysicsScalar ScalarT, size_t Dimensions>
struct physics_data_factory {
    using vector_type = std::array<ScalarT, Dimensions>;
    using tensor_type = std::array<vector_type, Dimensions>;

    // Choose optimal memory layout
    using optimized_vector = std::conditional_t<
        (Dimensions == 3) && concepts::VectorizedCompatible<std::array<ScalarT, 4>>,
        std::array<ScalarT, 4>,  // Pad to 4 for better alignment
        vector_type
    >;
};

// Automatic kernel launch parameter calculation
template<concepts::GPUCompatible T, size_t dataset_size>
struct kernel_launch_params {
    static constexpr int block_size = cuda_block_size_v<T>;
    static constexpr int grid_size = (dataset_size + block_size - 1) / block_size;
    static constexpr int shared_memory = block_size * sizeof(T);

    // Validate parameters are within GPU limits
    static_assert(block_size <= 1024, "Block size exceeds GPU limits");
    static_assert(shared_memory <= 48 * 1024, "Shared memory exceeds GPU limits");
};

// Physics system composition utilities
template<typename... Systems>
struct system_composer {
    static constexpr bool all_coupleable =
        (concepts::CoupledSystem<Systems> && ...);

    static constexpr bool all_differentiable =
        (concepts::Differentiable<Systems> && ...);

    using common_scalar_type = std::common_type_t<
        typename physics_properties<Systems>::scalar_type...
    >;

    // Ensure all systems use compatible scalar types
    static_assert(
        (std::same_as<
            typename physics_properties<Systems>::scalar_type,
            common_scalar_type
        > && ...),
        "All systems in composition must use compatible scalar types"
    );
};

} // namespace physgrad::type_traits