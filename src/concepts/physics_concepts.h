#pragma once

#include <concepts>
#include <type_traits>
#include <complex>
#include <array>
#include <vector>
#include <limits>

namespace physgrad::concepts {

// =============================================================================
// FUNDAMENTAL PHYSICS TYPE CONCEPTS
// =============================================================================

// Basic scalar types suitable for physics computations
template<typename T>
concept PhysicsScalar = std::floating_point<T> && requires {
    // Must support basic arithmetic
    std::declval<T>() + std::declval<T>();
    std::declval<T>() - std::declval<T>();
    std::declval<T>() * std::declval<T>();
    std::declval<T>() / std::declval<T>();

    // Must have reasonable precision for physics
    requires std::numeric_limits<T>::digits >= 23; // At least float precision

    // Must support mathematical functions
    requires requires(T x) {
        std::sqrt(x);
        std::sin(x);
        std::cos(x);
        std::exp(x);
        std::log(x);
    };
};

// Enhanced scalar with additional physics requirements
template<typename T>
concept HighPrecisionScalar = PhysicsScalar<T> && requires {
    requires std::numeric_limits<T>::digits >= 52; // At least double precision
    requires std::numeric_limits<T>::has_infinity;
    requires std::numeric_limits<T>::has_quiet_NaN;
};

// Complex number support for quantum mechanics
template<typename T>
concept PhysicsComplex = requires {
    typename T::value_type;
    requires PhysicsScalar<typename T::value_type>;
    requires std::same_as<T, std::complex<typename T::value_type>>;
};

// =============================================================================
// VECTOR AND TENSOR CONCEPTS
// =============================================================================

// 3D vector types for spatial physics
template<typename T>
concept Vector3D = requires(T v) {
    typename T::value_type;

    // Must have 3 components accessible as [0], [1], [2]
    { v[0] } -> std::convertible_to<typename T::value_type>;
    { v[1] } -> std::convertible_to<typename T::value_type>;
    { v[2] } -> std::convertible_to<typename T::value_type>;

    // Scalar type must be physics-compatible
    requires PhysicsScalar<typename T::value_type>;

    // Must have size information that evaluates to 3
    requires requires { { v.size() } -> std::convertible_to<size_t>; };
    requires v.size() == 3;
};

// Homogeneous coordinates for transformations
template<typename T>
concept Vector4D = requires(T v) {
    { v[0] } -> std::convertible_to<typename T::value_type>;
    { v[1] } -> std::convertible_to<typename T::value_type>;
    { v[2] } -> std::convertible_to<typename T::value_type>;
    { v[3] } -> std::convertible_to<typename T::value_type>;

    requires PhysicsScalar<typename T::value_type>;
    requires T{}.size() == 4;
};

// General tensor concept for neural network integration
template<typename T>
concept PhysicsTensor = requires(T tensor) {
    typename T::value_type;
    requires PhysicsScalar<typename T::value_type>;

    // Must have shape/dimension information
    { tensor.size() } -> std::convertible_to<size_t>;
    { tensor.data() } -> std::convertible_to<typename T::value_type*>;

    // Must support element access
    { tensor[0] } -> std::convertible_to<typename T::value_type>;
};

// =============================================================================
// MEMORY LAYOUT AND PERFORMANCE CONCEPTS
// =============================================================================

// Types suitable for GPU memory operations
template<typename T>
concept GPUCompatible = std::is_trivially_copyable_v<T> &&
                       std::is_standard_layout_v<T> &&
                       requires {
    // Must be aligned for efficient GPU access
    requires (sizeof(T) % 4 == 0) || sizeof(T) <= 4;

    // Must not contain pointers or virtual functions
    requires !std::is_pointer_v<T>;
    requires std::is_trivially_destructible_v<T>;
};

// Memory-aligned types for vectorized operations
template<typename T>
concept VectorizedCompatible = GPUCompatible<T> && requires {
    // Must be aligned to SIMD boundaries
    requires alignof(T) >= 16; // 128-bit alignment for SSE/NEON

    // Size must be a power of 2 for efficient vectorization
    requires (sizeof(T) & (sizeof(T) - 1)) == 0;
};

// Array-of-Structures-of-Arrays (AoSoA) layout concept
template<typename T>
concept AoSoACompatible = requires {
    // Must have a compile-time known structure
    requires std::is_aggregate_v<T>;
    requires std::is_trivially_copyable_v<T>;

    // Must define chunk size for optimal memory layout
    requires requires { T::chunk_size; };
    requires std::same_as<decltype(T::chunk_size), const size_t>;
};

// Cache-friendly data layout
template<typename T>
concept CacheOptimized = GPUCompatible<T> && requires {
    // Must fit within common cache line sizes
    requires sizeof(T) <= 64; // Typical L1 cache line size

    // No padding waste
    requires sizeof(T) == sizeof(typename T::value_type) * T{}.size();
};

// =============================================================================
// PHYSICS SIMULATION CONCEPTS
// =============================================================================

// Particle types for N-body simulations
template<typename T>
concept Particle = requires(T p) {
    // Must have position (essential for all physics)
    { p.position() } -> Vector3D;

    // Must have mass (for dynamics)
    { p.mass() } -> PhysicsScalar;

    // Must be GPU-compatible for performance
    requires GPUCompatible<T>;

    // Must support state copying/assignment
    requires std::copyable<T>;
};

// Enhanced particle with velocity for dynamics
template<typename T>
concept DynamicParticle = Particle<T> && requires(T p) {
    { p.velocity() } -> Vector3D;

    // Must support Newton's laws
    requires std::same_as<
        typename decltype(p.position())::value_type,
        typename decltype(p.velocity())::value_type
    >;
};

// Charged particle for electromagnetic simulations
template<typename T>
concept ChargedParticle = DynamicParticle<T> && requires(T p) {
    { p.charge() } -> PhysicsScalar;
};

// Quantum particle with wave function
template<typename T>
concept QuantumParticle = Particle<T> && requires(T p) {
    { p.wavefunction() } -> PhysicsComplex;

    // Must support quantum state operations
    requires requires(T p1, T p2) {
        p1.inner_product(p2); // Quantum inner product
        p1.normalize();       // Wave function normalization
    };
};

// =============================================================================
// INTEGRATION SCHEME CONCEPTS
// =============================================================================

// Basic numerical integrator requirements
template<typename T>
concept Integrator = requires(T integrator) {
    typename T::scalar_type;
    typename T::state_type;

    requires PhysicsScalar<typename T::scalar_type>;

    // Must implement time stepping
    requires requires(
        typename T::state_type state,
        typename T::scalar_type dt
    ) {
        { integrator.step(state, dt) } -> std::same_as<typename T::state_type>;
    };

    // Must provide stability information
    { integrator.max_stable_timestep() } -> std::convertible_to<typename T::scalar_type>;
};

// Symplectic integrator for Hamiltonian systems
template<typename T>
concept SymplecticIntegrator = Integrator<T> && requires {
    // Must preserve symplectic structure
    requires requires { typename T::is_symplectic; };
    requires T::is_symplectic::value;

    // Must provide order of accuracy
    requires requires { T::order; };
    requires std::integral<decltype(T::order)>;
    requires T::order >= 2;
};

// Variational integrator with discrete Lagrangian
template<typename T>
concept VariationalIntegrator = SymplecticIntegrator<T> && requires {
    // Must implement discrete Euler-Lagrange equations
    requires requires { typename T::is_variational; };
    requires T::is_variational::value;

    // Must conserve momentum when Lagrangian is translation-invariant
    requires requires { typename T::conserves_momentum; };
};

// Adaptive integrator with error control
template<typename T>
concept AdaptiveIntegrator = Integrator<T> && requires(T integrator) {
    // Must provide error estimation
    requires requires(
        typename T::state_type state,
        typename T::scalar_type dt,
        typename T::scalar_type tolerance
    ) {
        { integrator.adaptive_step(state, dt, tolerance) } ->
            std::same_as<std::pair<typename T::state_type, typename T::scalar_type>>;
    };

    // Must support error tolerance setting
    { integrator.set_tolerance(std::declval<typename T::scalar_type>()) } -> std::same_as<void>;
};

// =============================================================================
// FORCE COMPUTATION CONCEPTS
// =============================================================================

// Basic force field interface
template<typename T>
concept ForceField = requires(T field) {
    typename T::scalar_type;
    typename T::vector_type;

    requires PhysicsScalar<typename T::scalar_type>;
    requires Vector3D<typename T::vector_type>;

    // Must compute force from position
    requires requires(typename T::vector_type position) {
        { field.force_at(position) } -> std::same_as<typename T::vector_type>;
    };

    // Must provide potential energy (for conservative forces)
    requires requires(typename T::vector_type position) {
        { field.potential_at(position) } -> std::same_as<typename T::scalar_type>;
    };
};

// Conservative force field with energy conservation
template<typename T>
concept ConservativeForceField = ForceField<T> && requires {
    // Force must be negative gradient of potential
    requires requires { typename T::is_conservative; };
    requires T::is_conservative::value;

    // Must support gradient computation for symplectic integrators
    requires requires(T field, typename T::vector_type position) {
        { field.force_gradient_at(position) } -> std::convertible_to<std::array<typename T::vector_type, 3>>;
    };
};

// Pairwise force for N-body simulations
template<typename T>
concept PairwiseForce = requires(T force) {
    typename T::scalar_type;
    typename T::vector_type;

    requires PhysicsScalar<typename T::scalar_type>;
    requires Vector3D<typename T::vector_type>;

    // Must compute force between two particles
    requires requires(
        typename T::vector_type r1,
        typename T::vector_type r2,
        typename T::scalar_type m1,
        typename T::scalar_type m2
    ) {
        { force.compute_force(r1, r2, m1, m2) } ->
            std::same_as<std::pair<typename T::vector_type, typename T::vector_type>>;
    };

    // Must provide interaction range for optimization
    { force.cutoff_radius() } -> std::convertible_to<typename T::scalar_type>;
};

// =============================================================================
// DIFFERENTIABILITY CONCEPTS
// =============================================================================

// Types that support automatic differentiation
template<typename T>
concept Differentiable = requires(T value) {
    // Must support gradient computation
    requires requires { typename T::gradient_type; };

    // Must provide backward pass interface
    requires requires(typename T::gradient_type grad) {
        value.backward(grad);
    };

    // Must support gradient accumulation
    requires requires {
        { value.gradient() } -> std::same_as<typename T::gradient_type>;
        value.zero_gradient();
    };
};

// Differentiable tensor compatible with ML frameworks
template<typename T>
concept DifferentiableTensor = PhysicsTensor<T> && Differentiable<T> && requires(T tensor) {
    // Must integrate with PyTorch/JAX autograd
    requires requires { tensor.requires_grad(true); };

    // Must support device placement
    requires requires(int device_id) {
        { tensor.to_device(device_id) } -> std::same_as<T>;
    };

    // Must support zero-copy with physics data structures
    requires requires(typename T::value_type* data, size_t size) {
        { T::from_blob(data, size) } -> std::same_as<T>;
    };
};

// Function that preserves differentiability
template<typename F, typename T>
concept DifferentiableFunction = requires(F func, T input) {
    requires Differentiable<T>;
    requires Differentiable<std::invoke_result_t<F, T>>;

    // Must provide both forward and backward pass
    { func.forward(input) } -> Differentiable;
    requires requires(
        std::invoke_result_t<F, T> output,
        typename std::invoke_result_t<F, T>::gradient_type grad_output
    ) {
        { func.backward(input, output, grad_output) } ->
            std::same_as<typename T::gradient_type>;
    };
};

// =============================================================================
// MULTI-PHYSICS COUPLING CONCEPTS
// =============================================================================

// System that can be coupled with other physics domains
template<typename T>
concept CoupledSystem = requires(T system) {
    typename T::state_type;
    typename T::boundary_type;

    // Must provide boundary conditions for coupling
    requires requires(typename T::boundary_type boundary) {
        system.set_boundary_conditions(boundary);
        { system.get_boundary_state() } -> std::same_as<typename T::boundary_type>;
    };

    // Must support time synchronization
    requires requires(typename T::scalar_type time) {
        system.synchronize_to_time(time);
        { system.current_time() } -> std::same_as<typename T::scalar_type>;
    };
};

// Fluid-structure interaction coupling
template<typename FluidT, typename SolidT>
concept FSICoupled = requires {
    requires CoupledSystem<FluidT>;
    requires CoupledSystem<SolidT>;

    // Boundary types must be compatible
    requires std::convertible_to<
        typename FluidT::boundary_type,
        typename SolidT::boundary_type
    >;

    // Must support force/displacement exchange
    requires requires(FluidT fluid, SolidT solid) {
        fluid.apply_solid_displacement(solid.get_boundary_state());
        solid.apply_fluid_forces(fluid.get_boundary_state());
    };
};

// =============================================================================
// QUANTUM-CLASSICAL HYBRID CONCEPTS
// =============================================================================

// System that bridges quantum and classical physics
template<typename T>
concept QuantumClassicalHybrid = requires(T system) {
    typename T::quantum_state_type;
    typename T::classical_state_type;

    requires requires(T system) {
        // Must support state decomposition
        { system.quantum_subsystem() } -> std::same_as<typename T::quantum_state_type>;
        { system.classical_subsystem() } -> std::same_as<typename T::classical_state_type>;

        // Must handle decoherence
        requires requires(typename T::scalar_type decoherence_time) {
            system.set_decoherence_time(decoherence_time);
        };

        // Must support measurement-induced collapse
        system.perform_measurement();
    };
};

// =============================================================================
// VALIDATION AND TESTING CONCEPTS
// =============================================================================

// Type that can validate physics correctness
template<typename T>
concept PhysicsValidatable = requires(T obj) {
    // Must check conservation laws
    { obj.check_energy_conservation() } -> std::convertible_to<bool>;
    { obj.check_momentum_conservation() } -> std::convertible_to<bool>;

    // Must validate numerical stability
    { obj.is_numerically_stable() } -> std::convertible_to<bool>;

    // Must provide error metrics
    { obj.relative_error() } -> PhysicsScalar;
};

// Benchmarkable simulation component
template<typename T>
concept Benchmarkable = requires(T component) {
    // Must provide performance metrics
    { component.particles_per_second() } -> std::convertible_to<double>;
    { component.memory_bandwidth_utilization() } -> std::convertible_to<double>;

    // Must support profiling
    requires requires {
        component.start_profiling();
        component.stop_profiling();
        { component.get_profile_results() } -> std::convertible_to<std::string>;
    };
};

} // namespace physgrad::concepts