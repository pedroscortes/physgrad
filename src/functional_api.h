#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <type_traits>
#include <concepts>
#include <array>
#include <vector>
#include <map>
#include <tuple>
#include <cmath>
#include <random>
#include <limits>
#include "memory_optimization.h"

namespace physgrad::functional {

// =============================================================================
// PURE FUNCTION CONCEPTS AND TYPE SYSTEM
// =============================================================================

// Concept for pure computational functions (no side effects)
template<typename F, typename... Args>
concept PureFunction = std::invocable<F, Args...> &&
    requires {
        typename std::invoke_result_t<F, Args...>;
    };

// Concept for immutable data structures
template<typename T>
concept ImmutableData = std::is_const_v<T> ||
    (std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>);

// Concept for GPU-compatible data types
template<typename T>
concept GPUCompatible = std::is_trivially_copyable_v<T> &&
    (sizeof(T) % 4 == 0) && // Aligned to 4-byte boundaries
    (std::is_arithmetic_v<T> || std::is_pod_v<T>);

// =============================================================================
// IMMUTABLE STATE REPRESENTATION
// =============================================================================

// Immutable particle state with zero-copy semantics
template<typename Scalar = float>
struct ImmutableParticleState {
    using Vector3 = std::array<Scalar, 3>;
    using Vector4 = std::array<Scalar, 4>;

    const std::vector<Vector4> positions;  // w = mass
    const std::vector<Vector3> velocities;
    const std::vector<Scalar> charges;
    const std::vector<Scalar> masses;

    // Physics metadata
    const Scalar total_energy;
    const Scalar kinetic_energy;
    const Scalar potential_energy;
    const Scalar temperature;
    const size_t timestamp;

    // Factory methods for creating new states
    static ImmutableParticleState create(
        std::vector<Vector4> pos,
        std::vector<Vector3> vel,
        std::vector<Scalar> charges,
        std::vector<Scalar> masses
    );

    // Functional transformation methods
    ImmutableParticleState withUpdatedPositions(const std::vector<Vector4>& new_pos) const;
    ImmutableParticleState withUpdatedVelocities(const std::vector<Vector3>& new_vel) const;
    ImmutableParticleState withTimestamp(size_t ts) const;

    // Validation and consistency checks
    bool isValid() const noexcept;
    size_t particleCount() const noexcept { return positions.size(); }

    // Zero-copy GPU memory views
    struct GPUView {
        const Scalar* positions_ptr;
        const Scalar* velocities_ptr;
        const Scalar* charges_ptr;
        const Scalar* masses_ptr;
        size_t count;
    };

    GPUView getGPUView() const noexcept;
};

// =============================================================================
// PURE COMPUTATION PRIMITIVES
// =============================================================================

namespace pure {

// Pure force computation function
template<typename Scalar>
PureFunction auto computeElectrostaticForces = [](
    const ImmutableParticleState<Scalar>& state,
    Scalar coulomb_constant = 8.99e9f
) -> std::vector<std::array<Scalar, 3>> {
    const size_t n = state.particleCount();
    std::vector<std::array<Scalar, 3>> forces(n, {0, 0, 0});

    // Pure O(n²) force computation with no side effects
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const auto& ri = state.positions[i];
            const auto& rj = state.positions[j];

            // Compute distance vector
            std::array<Scalar, 3> dr = {
                ri[0] - rj[0],
                ri[1] - rj[1],
                ri[2] - rj[2]
            };

            Scalar r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
            Scalar r = std::sqrt(r2);

            if (r < std::numeric_limits<Scalar>::epsilon()) continue;

            // Coulomb force magnitude
            Scalar force_mag = coulomb_constant * state.charges[i] * state.charges[j] / (r2 * r);

            // Apply Newton's third law
            for (int k = 0; k < 3; ++k) {
                forces[i][k] += force_mag * dr[k];
                forces[j][k] -= force_mag * dr[k];
            }
        }
    }

    return forces;
};

// Pure Verlet integration step
template<typename Scalar>
PureFunction auto verletIntegrationStep = [](
    const ImmutableParticleState<Scalar>& current_state,
    const std::vector<std::array<Scalar, 3>>& forces,
    Scalar dt
) -> ImmutableParticleState<Scalar> {
    const size_t n = current_state.particleCount();

    std::vector<std::array<Scalar, 4>> new_positions(n);
    std::vector<std::array<Scalar, 3>> new_velocities(n);

    for (size_t i = 0; i < n; ++i) {
        const auto& pos = current_state.positions[i];
        const auto& vel = current_state.velocities[i];
        const auto& force = forces[i];
        Scalar mass = current_state.masses[i];

        // Verlet integration: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        for (int k = 0; k < 3; ++k) {
            Scalar acceleration = force[k] / mass;
            new_positions[i][k] = pos[k] + vel[k] * dt + 0.5f * acceleration * dt * dt;
            new_velocities[i][k] = vel[k] + acceleration * dt;
        }
        new_positions[i][3] = pos[3]; // preserve mass
    }

    return ImmutableParticleState<Scalar>::create(
        std::move(new_positions),
        std::move(new_velocities),
        current_state.charges,
        current_state.masses
    ).withTimestamp(current_state.timestamp + 1);
};

// Pure energy computation
template<typename Scalar>
PureFunction auto computeTotalEnergy = [](
    const ImmutableParticleState<Scalar>& state,
    Scalar coulomb_constant = 8.99e9f
) -> std::tuple<Scalar, Scalar, Scalar> {
    const size_t n = state.particleCount();
    Scalar kinetic = 0;
    Scalar potential = 0;

    // Kinetic energy
    for (size_t i = 0; i < n; ++i) {
        const auto& vel = state.velocities[i];
        Scalar v2 = vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2];
        kinetic += 0.5f * state.masses[i] * v2;
    }

    // Potential energy
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const auto& ri = state.positions[i];
            const auto& rj = state.positions[j];

            Scalar dx = ri[0] - rj[0];
            Scalar dy = ri[1] - rj[1];
            Scalar dz = ri[2] - rj[2];
            Scalar r = std::sqrt(dx*dx + dy*dy + dz*dz);

            if (r > std::numeric_limits<Scalar>::epsilon()) {
                potential += coulomb_constant * state.charges[i] * state.charges[j] / r;
            }
        }
    }

    return std::make_tuple(kinetic, potential, kinetic + potential);
};

// Pure spatial binning for fast neighbor finding
template<typename Scalar>
PureFunction auto spatialBinning = [](
    const ImmutableParticleState<Scalar>& state,
    Scalar bin_size
) -> std::map<std::tuple<int, int, int>, std::vector<size_t>> {
    std::map<std::tuple<int, int, int>, std::vector<size_t>> bins;

    for (size_t i = 0; i < state.particleCount(); ++i) {
        const auto& pos = state.positions[i];
        int bx = static_cast<int>(std::floor(pos[0] / bin_size));
        int by = static_cast<int>(std::floor(pos[1] / bin_size));
        int bz = static_cast<int>(std::floor(pos[2] / bin_size));

        bins[{bx, by, bz}].push_back(i);
    }

    return bins;
};

} // namespace pure

// =============================================================================
// FUNCTIONAL COMPOSITION AND PIPELINES
// =============================================================================

// Function composition utility
template<typename F1, typename F2>
auto compose(F1&& f1, F2&& f2) {
    return [f1 = std::forward<F1>(f1), f2 = std::forward<F2>(f2)](auto&& x) {
        return f1(f2(std::forward<decltype(x)>(x)));
    };
}

// Pipeline builder for chaining pure computations
template<typename T>
class FunctionalPipeline {
private:
    T current_value_;

public:
    explicit FunctionalPipeline(T value) : current_value_(std::move(value)) {}

    template<typename F>
    auto then(F&& func) -> FunctionalPipeline<std::invoke_result_t<F, T>> {
        return FunctionalPipeline<std::invoke_result_t<F, T>>(
            std::forward<F>(func)(current_value_)
        );
    }

    template<typename F>
    auto map(F&& func) -> FunctionalPipeline<std::invoke_result_t<F, T>> {
        return then(std::forward<F>(func));
    }

    const T& get() const { return current_value_; }
    T extract() && { return std::move(current_value_); }
};

// Pipeline factory
template<typename T>
auto pipeline(T&& value) {
    return FunctionalPipeline<std::decay_t<T>>(std::forward<T>(value));
}

// =============================================================================
// SIMULATION ORCHESTRATION WITH PURE FUNCTIONS
// =============================================================================

template<typename Scalar = float>
class FunctionalSimulation {
public:
    using State = ImmutableParticleState<Scalar>;
    using ForceFunction = std::function<std::vector<std::array<Scalar, 3>>(const State&)>;
    using IntegrationFunction = std::function<State(const State&, const std::vector<std::array<Scalar, 3>>&, Scalar)>;

private:
    ForceFunction force_function_;
    IntegrationFunction integration_function_;
    Scalar dt_;

public:
    FunctionalSimulation(
        ForceFunction force_func,
        IntegrationFunction integration_func,
        Scalar timestep
    ) : force_function_(std::move(force_func)),
        integration_function_(std::move(integration_func)),
        dt_(timestep) {}

    // Pure simulation step
    State step(const State& current_state) const {
        return pipeline(current_state)
            .then([this](const State& state) { return force_function_(state); })
            .then([this, &current_state](const auto& forces) {
                return integration_function_(current_state, forces, dt_);
            })
            .get();
    }

    // Multi-step evolution (pure, no side effects)
    std::vector<State> evolve(const State& initial_state, size_t num_steps) const {
        std::vector<State> trajectory;
        trajectory.reserve(num_steps + 1);
        trajectory.push_back(initial_state);

        State current = initial_state;
        for (size_t i = 0; i < num_steps; ++i) {
            current = step(current);
            trajectory.push_back(current);
        }

        return trajectory;
    }

    // Energy conservation analysis
    std::vector<Scalar> analyzeEnergyConservation(
        const State& initial_state,
        size_t num_steps
    ) const {
        std::vector<Scalar> energies;
        energies.reserve(num_steps + 1);

        auto trajectory = evolve(initial_state, num_steps);
        for (const auto& state : trajectory) {
            auto [kinetic, potential, total] = pure::computeTotalEnergy<Scalar>(state);
            energies.push_back(total);
        }

        return energies;
    }
};

// =============================================================================
// FACTORY FUNCTIONS FOR COMMON SETUPS
// =============================================================================

// Factory for electrostatic simulation
template<typename Scalar = float>
auto createElectrostaticSimulation(Scalar dt, Scalar coulomb_constant = 8.99e9f) {
    auto force_func = [coulomb_constant](const auto& state) {
        return pure::computeElectrostaticForces<Scalar>(state, coulomb_constant);
    };

    auto integration_func = [](const auto& state, const auto& forces, Scalar timestep) {
        return pure::verletIntegrationStep<Scalar>(state, forces, timestep);
    };

    return FunctionalSimulation<Scalar>(force_func, integration_func, dt);
}

// Factory for creating test particle states
template<typename Scalar = float>
auto createRandomParticleState(size_t num_particles, Scalar domain_size = 1.0f) {
    std::vector<std::array<Scalar, 4>> positions(num_particles);
    std::vector<std::array<Scalar, 3>> velocities(num_particles);
    std::vector<Scalar> charges(num_particles);
    std::vector<Scalar> masses(num_particles);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> pos_dist(-domain_size, domain_size);
    std::uniform_real_distribution<Scalar> vel_dist(-0.1f, 0.1f);
    std::uniform_real_distribution<Scalar> charge_dist(-1.0f, 1.0f);

    for (size_t i = 0; i < num_particles; ++i) {
        positions[i] = {pos_dist(gen), pos_dist(gen), pos_dist(gen), 1.0f}; // mass = 1
        velocities[i] = {vel_dist(gen), vel_dist(gen), vel_dist(gen)};
        charges[i] = charge_dist(gen);
        masses[i] = 1.0f;
    }

    return ImmutableParticleState<Scalar>::create(
        std::move(positions), std::move(velocities),
        std::move(charges), std::move(masses)
    );
}

} // namespace physgrad::functional