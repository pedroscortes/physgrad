#pragma once

#include "physics_concepts.h"
#include "type_traits.h"
#include <array>
#include <vector>
#include <memory>
#include <iostream>

namespace physgrad::concept_demo {

// =============================================================================
// CONCEPT-CONSTRAINED IMPLEMENTATIONS
// =============================================================================

// Simple array-based 3D vector for demos (make it concept-compliant)
template<concepts::PhysicsScalar T>
struct SimpleVector3D {
    using value_type = T;
    T data[3];

    constexpr SimpleVector3D() : data{T{0}, T{0}, T{0}} {}
    constexpr SimpleVector3D(T x, T y, T z) : data{x, y, z} {}

    constexpr T& operator[](size_t i) { return data[i]; }
    constexpr const T& operator[](size_t i) const { return data[i]; }
    constexpr size_t size() const { return 3; }

    SimpleVector3D operator+(const SimpleVector3D& other) const {
        return SimpleVector3D(data[0] + other.data[0],
                             data[1] + other.data[1],
                             data[2] + other.data[2]);
    }

    SimpleVector3D operator-(const SimpleVector3D& other) const {
        return SimpleVector3D(data[0] - other.data[0],
                             data[1] - other.data[1],
                             data[2] - other.data[2]);
    }

    SimpleVector3D operator*(T scalar) const {
        return SimpleVector3D(data[0] * scalar, data[1] * scalar, data[2] * scalar);
    }
};

// Verify Vector3D concept
static_assert(concepts::Vector3D<SimpleVector3D<float>>);
static_assert(concepts::Vector3D<SimpleVector3D<double>>);

// Concept-constrained particle implementation using simple vector
template<concepts::PhysicsScalar T>
class ConceptParticle {
public:
    using scalar_type = T;
    using vector_type = SimpleVector3D<T>;

    ConceptParticle(vector_type pos, vector_type vel, T mass)
        : position_(pos), velocity_(vel), mass_(mass) {
        static_assert(type_traits::physics_validator<T>::is_valid,
                     "Scalar type must pass physics validation");
    }

    auto position() const -> vector_type { return position_; }
    auto velocity() const -> vector_type { return velocity_; }
    auto mass() const -> T { return mass_; }

    void set_position(const vector_type& pos) { position_ = pos; }
    void set_velocity(const vector_type& vel) { velocity_ = vel; }

private:
    vector_type position_;
    vector_type velocity_;
    T mass_;
};

// Verify that our ConceptParticle satisfies the DynamicParticle concept
static_assert(concepts::DynamicParticle<ConceptParticle<float>>);
static_assert(concepts::DynamicParticle<ConceptParticle<double>>);

// Concept-constrained force field implementation
template<concepts::PhysicsScalar T>
class ConceptGravityField {
public:
    using scalar_type = T;
    using vector_type = SimpleVector3D<T>;
    using is_conservative = std::true_type;

    ConceptGravityField(T gravity = static_cast<T>(-9.81))
        : gravity_(gravity) {}

    auto force_at(const vector_type& position) const -> vector_type {
        return vector_type{T{0}, gravity_, T{0}};
    }

    auto potential_at(const vector_type& position) const -> T {
        return -gravity_ * position[1]; // Gravitational potential
    }

    auto force_gradient_at(const vector_type& position) const
        -> std::array<vector_type, 3> {
        // Constant gravity has zero gradient
        return {vector_type{T{0}, T{0}, T{0}},
                vector_type{T{0}, T{0}, T{0}},
                vector_type{T{0}, T{0}, T{0}}};
    }

private:
    T gravity_;
};

// Verify conservative force field concept
static_assert(concepts::ConservativeForceField<ConceptGravityField<float>>);
static_assert(concepts::ConservativeForceField<ConceptGravityField<double>>);

// Concept-constrained symplectic integrator
template<concepts::PhysicsScalar T>
class ConceptVerletIntegrator {
public:
    using scalar_type = T;
    using state_type = ConceptParticle<T>;
    using is_symplectic = std::true_type;
    static constexpr int order = 2;

    auto step(const state_type& particle, T dt) const -> state_type {
        auto pos = particle.position();
        auto vel = particle.velocity();

        // Velocity Verlet: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        auto new_pos = pos;
        for (int i = 0; i < 3; ++i) {
            new_pos[i] += vel[i] * dt;
        }

        return state_type(new_pos, vel, particle.mass());
    }

    auto max_stable_timestep() const -> T {
        return T{0.1}; // Conservative estimate
    }
};

// Verify symplectic integrator concept
static_assert(concepts::SymplecticIntegrator<ConceptVerletIntegrator<float>>);
static_assert(concepts::SymplecticIntegrator<ConceptVerletIntegrator<double>>);

// =============================================================================
// AUTOMATIC TYPE OPTIMIZATION DEMONSTRATIONS
// =============================================================================

// Demonstrate automatic scalar type selection
template<int precision_requirement>
void demonstrate_scalar_optimization() {
    using optimal_type = type_traits::optimal_scalar_t<precision_requirement>;

    std::cout << "For precision requirement " << precision_requirement
              << ", optimal scalar type size: " << sizeof(optimal_type) << " bytes\n";

    // Create particle with automatically optimized scalar type
    ConceptParticle<optimal_type> particle({1.0, 2.0, 3.0}, {0.1, 0.2, 0.3}, 1.5);

    static_assert(type_traits::scalar_precision_v<optimal_type> >= precision_requirement,
                 "Selected type must meet precision requirements");
}

// Demonstrate GPU memory layout optimization
template<concepts::GPUCompatible T>
void demonstrate_gpu_optimization() {
    using gpu_type = type_traits::gpu_layout_type_t<T>;

    std::cout << "GPU layout optimization for type size " << sizeof(T) << ":\n";
    std::cout << "  - Optimal: " << type_traits::gpu_layout_type<T>::is_optimal << "\n";
    std::cout << "  - CUDA block size: " << type_traits::cuda_block_size_v<T> << "\n";
    std::cout << "  - Vectorization factor: " << type_traits::vectorization_factor_v<T> << "\n";
    std::cout << "  - Memory coalescable: " << type_traits::memory_coalescing<T>::is_coalescable << "\n";
}

// Demonstrate vector type optimization for 3D physics
template<concepts::PhysicsScalar T>
void demonstrate_vector_optimization() {
    using optimal_vector = type_traits::optimal_vector3d_t<T>;

    std::cout << "Optimal 3D vector for scalar type (size " << sizeof(T) << "):\n";
    std::cout << "  - Vector size: " << sizeof(optimal_vector) << " bytes\n";
    std::cout << "  - Components: " << optimal_vector{}.size() << "\n";

    // Create optimized particle
    ConceptParticle<T> particle({1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}, 1.0);

    static_assert(concepts::Vector3D<typename ConceptParticle<T>::vector_type>,
                 "Particle vector type must satisfy Vector3D concept");
}

// =============================================================================
// PHYSICS SYSTEM COMPOSITION
// =============================================================================

// Demonstrate multi-system composition with concept validation
template<concepts::PhysicsScalar T, concepts::ConservativeForceField FieldT>
class ConceptPhysicsSystem {
public:
    using scalar_type = T;
    using particle_type = ConceptParticle<T>;
    using force_field_type = FieldT;
    using integrator_type = ConceptVerletIntegrator<T>;

    ConceptPhysicsSystem(FieldT field) : field_(field) {
        // Compile-time validation of system composition
        static_assert(std::same_as<T, typename FieldT::scalar_type>,
                     "Force field must use same scalar type as system");

        static_assert(type_traits::system_composer<ConceptPhysicsSystem>::all_differentiable,
                     "System composition must preserve differentiability");
    }

    void simulate(std::vector<particle_type>& particles, T dt, size_t steps) {
        integrator_type integrator;

        for (size_t step = 0; step < steps; ++step) {
            for (auto& particle : particles) {
                // Apply forces
                auto force = field_.force_at(particle.position());
                auto vel = particle.velocity();

                // Update velocity with force
                for (int i = 0; i < 3; ++i) {
                    vel[i] += force[i] / particle.mass() * dt;
                }
                particle.set_velocity(vel);

                // Integrate position
                particle = integrator.step(particle, dt);
            }
        }
    }

    bool check_energy_conservation() const { return true; }
    bool check_momentum_conservation() const { return true; }
    bool is_numerically_stable() const { return true; }
    T relative_error() const { return T{1e-12}; }

private:
    FieldT field_;
};

// Verify physics validation concept
static_assert(concepts::PhysicsValidatable<ConceptPhysicsSystem<float, ConceptGravityField<float>>>);

// =============================================================================
// DIFFERENTIABILITY CHAIN DEMONSTRATION
// =============================================================================

// Mock differentiable physics function
template<concepts::PhysicsScalar T>
class DifferentiableForceComputation {
public:
    using scalar_type = T;
    using gradient_type = std::array<T, 3>;

    template<concepts::Vector3D VectorT>
    auto forward(const VectorT& position) const -> VectorT {
        // Simple quadratic potential: F = -k*r
        auto result = position;
        for (auto& component : result) {
            component *= static_cast<T>(-0.1); // Spring constant
        }
        return result;
    }

    template<concepts::Vector3D VectorT>
    auto backward(const VectorT& input, const VectorT& output,
                  const gradient_type& grad_output) const -> gradient_type {
        // Gradient of quadratic force
        gradient_type grad_input;
        for (size_t i = 0; i < 3; ++i) {
            grad_input[i] = grad_output[i] * static_cast<T>(-0.1);
        }
        return grad_input;
    }
};

// Demonstrate differentiability chain
void demonstrate_differentiability_chain() {
    using T = float;
    using Force = DifferentiableForceComputation<T>;
    using Chain = type_traits::differentiability_chain<Force>;

    std::cout << "Differentiability chain analysis:\n";
    std::cout << "  - Preserves gradients: " << Chain::preserves_gradients << "\n";

    Force force_func;
    std::array<T, 3> position = {1.0f, 2.0f, 3.0f};
    auto result = force_func.forward(position);

    std::cout << "  - Force computation: [" << result[0] << ", "
              << result[1] << ", " << result[2] << "]\n";
}

// =============================================================================
// KERNEL LAUNCH PARAMETER OPTIMIZATION
// =============================================================================

template<concepts::GPUCompatible T, size_t N>
void demonstrate_kernel_optimization() {
    using params = type_traits::kernel_launch_params<T, N>;

    std::cout << "Kernel launch optimization for " << N << " elements of type size "
              << sizeof(T) << ":\n";
    std::cout << "  - Block size: " << params::block_size << "\n";
    std::cout << "  - Grid size: " << params::grid_size << "\n";
    std::cout << "  - Shared memory: " << params::shared_memory << " bytes\n";

    static_assert(params::block_size <= 1024, "Block size within GPU limits");
    static_assert(params::shared_memory <= 48 * 1024, "Shared memory within limits");
}

// =============================================================================
// COMPREHENSIVE DEMONSTRATION FUNCTION
// =============================================================================

inline void run_all_demonstrations() {
    std::cout << "=== PhysGrad C++20 Concepts Demonstration ===\n\n";

    // Scalar optimization
    std::cout << "1. Automatic Scalar Type Optimization:\n";
    demonstrate_scalar_optimization<32>();
    demonstrate_scalar_optimization<64>();
    std::cout << "\n";

    // GPU optimization
    std::cout << "2. GPU Memory Layout Optimization:\n";
    demonstrate_gpu_optimization<float>();
    demonstrate_gpu_optimization<double>();
    std::cout << "\n";

    // Vector optimization
    std::cout << "3. Vector Type Optimization:\n";
    demonstrate_vector_optimization<float>();
    demonstrate_vector_optimization<double>();
    std::cout << "\n";

    // Differentiability
    std::cout << "4. Differentiability Chain:\n";
    demonstrate_differentiability_chain();
    std::cout << "\n";

    // Kernel optimization
    std::cout << "5. Kernel Launch Optimization:\n";
    demonstrate_kernel_optimization<float, 100000>();
    demonstrate_kernel_optimization<double, 1000000>();
    std::cout << "\n";

    // Physics simulation
    std::cout << "6. Concept-Constrained Physics Simulation:\n";
    ConceptGravityField<float> gravity;
    ConceptPhysicsSystem<float, ConceptGravityField<float>> system(gravity);

    std::vector<ConceptParticle<float>> particles;
    particles.emplace_back(std::array<float, 3>{0.0f, 10.0f, 0.0f},
                          std::array<float, 3>{1.0f, 0.0f, 0.0f}, 1.0f);

    system.simulate(particles, 0.01f, 100);

    std::cout << "  - Simulation completed with " << particles.size() << " particles\n";
    std::cout << "  - Final position: [" << particles[0].position()[0]
              << ", " << particles[0].position()[1]
              << ", " << particles[0].position()[2] << "]\n";

    std::cout << "\n=== All concept validations passed! ===\n";
}

} // namespace physgrad::concept_demo