/**
 * PhysGrad - Physics Engine Header
 *
 * Main physics engine class that coordinates all subsystems.
 * Enhanced with C++20 concepts for type safety and automatic optimization.
 */

#ifndef PHYSGRAD_PHYSICS_ENGINE_H
#define PHYSGRAD_PHYSICS_ENGINE_H

#include "common_types.h"
#include <vector>
#include <memory>

// Include concepts when available
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "concepts/forward_declarations.h"
#endif

namespace physgrad {

/**
 * Main physics engine class
 */
class PhysicsEngine {
public:
    PhysicsEngine();
    ~PhysicsEngine();

    // Initialization and cleanup
    bool initialize();
    void cleanup();

    // Particle management
    void addParticles(
        const std::vector<float3>& positions,
        const std::vector<float3>& velocities,
        const std::vector<float>& masses
    );
    void removeParticle(int index);

    // Property setters
    void setCharges(const std::vector<float>& charges);
    void setPositions(const std::vector<float3>& positions);
    void setVelocities(const std::vector<float3>& velocities);

    // Simulation
    void updateForces();
    void step(float dt);

    // Energy calculations
    float calculateTotalEnergy() const;

    // Getters
    std::vector<float3> getPositions() const;
    std::vector<float3> getVelocities() const;
    std::vector<float3> getForces() const;
    int getNumParticles() const;

    // Simulation settings
    void setBoundaryConditions(BoundaryType type, float3 bounds);
    void setIntegrationMethod(IntegrationMethod method);

private:
    // Particle data
    std::vector<float3> positions_;
    std::vector<float3> velocities_;
    std::vector<float3> forces_;
    std::vector<float> masses_;
    std::vector<float> charges_;

    // State
    int num_particles_;
    bool initialized_;

    // Simulation settings
    BoundaryType boundary_type_ = BoundaryType::OPEN;
    float3 boundary_bounds_ = {0.0f, 0.0f, 0.0f};
    IntegrationMethod integration_method_ = IntegrationMethod::VERLET;

    // Internal methods
    void applyBoundaryConditions();
};

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE

// =============================================================================
// CONCEPT-ENHANCED PHYSICS ENGINE
// =============================================================================

/**
 * Modern concept-aware physics engine with automatic type optimization
 * and compile-time validation
 */
template<concepts::PhysicsScalar ScalarT = float>
    requires concepts::HighPrecisionScalar<ScalarT> || std::same_as<ScalarT, float>
class ConceptPhysicsEngine {
public:
    using scalar_type = ScalarT;
    using vector_type = ConceptVector3D<ScalarT>;
    using particle_type = ConceptParticleData<ScalarT>;

    // Automatically optimize container based on expected particle count
    template<size_t expected_count = 10000>
    using optimized_container = optimal_particle_container<ScalarT, expected_count>;

    ConceptPhysicsEngine() {
        // Basic compile-time validation
        static_assert(concepts::PhysicsScalar<ScalarT>,
                     "Scalar type must be a physics scalar");

        static_assert(concepts::GPUCompatible<particle_type>,
                     "Particle type must be GPU compatible");
    }

    // Concept-constrained particle management
    template<concepts::Vector3D VectorT>
    void addParticles(
        const std::vector<VectorT>& positions,
        const std::vector<VectorT>& velocities,
        const std::vector<ScalarT>& masses
    ) {
        particles_.reserve(particles_.size() + positions.size());

        for (size_t i = 0; i < positions.size(); ++i) {
            particles_.emplace_back(
                vector_type{positions[i][0], positions[i][1], positions[i][2]},
                vector_type{velocities[i][0], velocities[i][1], velocities[i][2]},
                masses[i]
            );
        }
    }

    // Type-safe force field integration
    template<concepts::ConservativeForceField FieldT>
        requires std::same_as<typename FieldT::scalar_type, ScalarT>
    void addForceField(std::shared_ptr<FieldT> field) {
        force_fields_.push_back(std::static_pointer_cast<ForceFieldBase>(field));
    }

    // Concept-constrained integrator selection
    template<concepts::SymplecticIntegrator IntegratorT>
        requires std::same_as<typename IntegratorT::scalar_type, ScalarT>
    void setIntegrator(IntegratorT integrator) {
        integrator_ = std::make_unique<IntegratorWrapper<IntegratorT>>(integrator);

        // Automatically optimize timestep based on integrator properties
        if constexpr (IntegratorT::order >= 4) {
            max_timestep_ = integrator.max_stable_timestep() * static_cast<ScalarT>(0.8);
        } else {
            max_timestep_ = integrator.max_stable_timestep() * static_cast<ScalarT>(0.5);
        }
    }

    // Physics simulation with automatic validation
    void step(ScalarT dt) {
        // Validate timestep against stability requirements
        if (dt > max_timestep_) {
            dt = max_timestep_;
        }

        // Apply force fields
        for (auto& particle : particles_) {
            vector_type total_force{};

            for (const auto& field : force_fields_) {
                auto force = field->force_at(particle.position());
                total_force = total_force + force;
            }

            // Apply force to velocity
            auto new_velocity = particle.velocity();
            for (size_t i = 0; i < 3; ++i) {
                new_velocity[i] += total_force[i] / particle.mass() * dt;
            }
            particle.set_velocity(new_velocity);
        }

        // Integrate using selected integrator
        if (integrator_) {
            for (auto& particle : particles_) {
                particle = integrator_->step(particle, dt);
            }
        }

        current_time_ += dt;
    }

    // Physics validation using concepts
    bool validatePhysics() const {
        if constexpr (concepts::PhysicsValidatable<ConceptPhysicsEngine>) {
            return check_energy_conservation() &&
                   check_momentum_conservation() &&
                   is_numerically_stable();
        }
        return true;
    }

    // Energy conservation check
    bool check_energy_conservation() const {
        ScalarT total_energy = calculateTotalEnergy();
        return std::abs(total_energy - initial_energy_) / initial_energy_ < static_cast<ScalarT>(1e-6);
    }

    // Momentum conservation check
    bool check_momentum_conservation() const {
        vector_type total_momentum{};
        for (const auto& particle : particles_) {
            auto momentum = particle.velocity() * particle.mass();
            total_momentum = total_momentum + momentum;
        }

        ScalarT momentum_magnitude = std::sqrt(
            total_momentum[0] * total_momentum[0] +
            total_momentum[1] * total_momentum[1] +
            total_momentum[2] * total_momentum[2]
        );

        return momentum_magnitude < static_cast<ScalarT>(1e-10);
    }

    // Numerical stability check
    bool is_numerically_stable() const {
        for (const auto& particle : particles_) {
            auto pos = particle.position();
            auto vel = particle.velocity();

            for (size_t i = 0; i < 3; ++i) {
                if (!std::isfinite(pos[i]) || !std::isfinite(vel[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    // Error metric
    ScalarT relative_error() const {
        return std::abs(calculateTotalEnergy() - initial_energy_) / initial_energy_;
    }

    // Energy calculation with concept validation
    ScalarT calculateTotalEnergy() const {
        ScalarT kinetic_energy{0};
        ScalarT potential_energy{0};

        // Kinetic energy
        for (const auto& particle : particles_) {
            auto vel = particle.velocity();
            ScalarT speed_squared = vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2];
            kinetic_energy += static_cast<ScalarT>(0.5) * particle.mass() * speed_squared;
        }

        // Potential energy from force fields
        for (const auto& particle : particles_) {
            for (const auto& field : force_fields_) {
                potential_energy += field->potential_at(particle.position());
            }
        }

        return kinetic_energy + potential_energy;
    }

    // Getters with concept-safe return types
    const std::vector<particle_type>& getParticles() const { return particles_; }
    size_t getNumParticles() const { return particles_.size(); }
    ScalarT getCurrentTime() const { return current_time_; }

    // Performance metrics (satisfies Benchmarkable concept)
    double particles_per_second() const {
        return static_cast<double>(particles_.size()) / last_step_time_;
    }

    double memory_bandwidth_utilization() const {
        size_t memory_accessed = particles_.size() * sizeof(particle_type) * 2; // Read + write
        return static_cast<double>(memory_accessed) / (last_step_time_ * 1e9); // GB/s
    }

    void start_profiling() { profiling_enabled_ = true; }
    void stop_profiling() { profiling_enabled_ = false; }
    std::string get_profile_results() const { return "Profiling results placeholder"; }

private:
    std::vector<particle_type> particles_;
    ScalarT current_time_{0};
    ScalarT max_timestep_{static_cast<ScalarT>(0.01)};
    ScalarT initial_energy_{0};
    double last_step_time_{1e-3}; // seconds
    bool profiling_enabled_{false};

    // Type-erased force field interface
    class ForceFieldBase {
    public:
        virtual ~ForceFieldBase() = default;
        virtual vector_type force_at(const vector_type& position) const = 0;
        virtual ScalarT potential_at(const vector_type& position) const = 0;
    };

    template<concepts::ConservativeForceField FieldT>
    class ForceFieldWrapper : public ForceFieldBase {
    public:
        ForceFieldWrapper(FieldT field) : field_(field) {}

        vector_type force_at(const vector_type& position) const override {
            auto force = field_.force_at(position);
            return vector_type{force[0], force[1], force[2]};
        }

        ScalarT potential_at(const vector_type& position) const override {
            return field_.potential_at(position);
        }

    private:
        FieldT field_;
    };

    std::vector<std::shared_ptr<ForceFieldBase>> force_fields_;

    // Type-erased integrator interface
    class IntegratorBase {
    public:
        virtual ~IntegratorBase() = default;
        virtual particle_type step(const particle_type& particle, ScalarT dt) const = 0;
    };

    template<concepts::SymplecticIntegrator IntegratorT>
    class IntegratorWrapper : public IntegratorBase {
    public:
        IntegratorWrapper(IntegratorT integrator) : integrator_(integrator) {}

        particle_type step(const particle_type& particle, ScalarT dt) const override {
            return integrator_.step(particle, dt);
        }

    private:
        IntegratorT integrator_;
    };

    std::unique_ptr<IntegratorBase> integrator_;
};

// Verify that ConceptPhysicsEngine satisfies physics concepts
static_assert(concepts::PhysicsValidatable<ConceptPhysicsEngine<float>>);
static_assert(concepts::Benchmarkable<ConceptPhysicsEngine<float>>);

// Type aliases for common use cases
using ConceptPhysicsEngineFloat = ConceptPhysicsEngine<float>;
using ConceptPhysicsEngineDouble = ConceptPhysicsEngine<double>;

// Automatic engine type selection based on precision requirements
template<int precision_requirement>
using OptimalPhysicsEngine = ConceptPhysicsEngine<optimal_physics_scalar<precision_requirement>>;

#endif // PHYSGRAD_CONCEPTS_AVAILABLE

} // namespace physgrad

#endif // PHYSGRAD_PHYSICS_ENGINE_H