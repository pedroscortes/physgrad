#pragma once

#include <functional>
#include <memory>
#include <array>
#include <vector>
#include <map>
#include <tuple>
#include <cmath>
#include <random>
#include <limits>
#include <type_traits>

namespace physgrad::functional {

// =============================================================================
// SIMPLIFIED PURE FUNCTION INTERFACE
// =============================================================================

// Immutable particle state with zero-copy semantics
template<typename Scalar = float>
struct ImmutableParticleState {
    using Vector3 = std::array<Scalar, 3>;
    using Vector4 = std::array<Scalar, 4>;

    std::vector<Vector4> positions;  // w = mass
    std::vector<Vector3> velocities;
    std::vector<Scalar> charges;
    std::vector<Scalar> masses;

    // Physics metadata (immutable by convention, not const to allow move semantics)
    Scalar total_energy;
    Scalar kinetic_energy;
    Scalar potential_energy;
    Scalar temperature;
    size_t timestamp;

    // Factory method for creating states
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

    // Validation
    bool isValid() const noexcept;
    size_t particleCount() const noexcept { return positions.size(); }
};

// =============================================================================
// PURE COMPUTATION FUNCTIONS
// =============================================================================

// Pure force computation function
template<typename Scalar>
std::vector<std::array<Scalar, 3>> computeElectrostaticForces(
    const ImmutableParticleState<Scalar>& state,
    Scalar coulomb_constant = 8.99e9f
) {
    const size_t n = state.particleCount();
    std::vector<std::array<Scalar, 3>> forces(n, {0, 0, 0});

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const auto& ri = state.positions[i];
            const auto& rj = state.positions[j];

            std::array<Scalar, 3> dr = {
                ri[0] - rj[0],
                ri[1] - rj[1],
                ri[2] - rj[2]
            };

            Scalar r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
            Scalar r = std::sqrt(r2);

            if (r < std::numeric_limits<Scalar>::epsilon()) continue;

            Scalar force_mag = coulomb_constant * state.charges[i] * state.charges[j] / (r2 * r);

            for (int k = 0; k < 3; ++k) {
                forces[i][k] += force_mag * dr[k];
                forces[j][k] -= force_mag * dr[k];
            }
        }
    }

    return forces;
}

// Pure Verlet integration step
template<typename Scalar>
ImmutableParticleState<Scalar> verletIntegrationStep(
    const ImmutableParticleState<Scalar>& current_state,
    const std::vector<std::array<Scalar, 3>>& forces,
    Scalar dt
) {
    const size_t n = current_state.particleCount();

    std::vector<std::array<Scalar, 4>> new_positions(n);
    std::vector<std::array<Scalar, 3>> new_velocities(n);

    for (size_t i = 0; i < n; ++i) {
        const auto& pos = current_state.positions[i];
        const auto& vel = current_state.velocities[i];
        const auto& force = forces[i];
        Scalar mass = current_state.masses[i];

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
}

// Pure energy computation
template<typename Scalar>
std::tuple<Scalar, Scalar, Scalar> computeTotalEnergy(
    const ImmutableParticleState<Scalar>& state,
    Scalar coulomb_constant = 8.99e9f
) {
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
}

// =============================================================================
// FUNCTIONAL SIMULATION ORCHESTRATION
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
        auto forces = force_function_(current_state);
        return integration_function_(current_state, forces, dt_);
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
            auto [kinetic, potential, total] = computeTotalEnergy<Scalar>(state);
            energies.push_back(total);
        }

        return energies;
    }
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// Compute center of mass (pure function)
template<typename Scalar>
std::array<Scalar, 3> computeCenterOfMass(const ImmutableParticleState<Scalar>& state) {
    std::array<Scalar, 3> com = {0, 0, 0};
    Scalar total_mass = 0;

    for (size_t i = 0; i < state.particleCount(); ++i) {
        const auto& pos = state.positions[i];
        Scalar mass = state.masses[i];

        com[0] += mass * pos[0];
        com[1] += mass * pos[1];
        com[2] += mass * pos[2];
        total_mass += mass;
    }

    if (total_mass > 0) {
        com[0] /= total_mass;
        com[1] /= total_mass;
        com[2] /= total_mass;
    }

    return com;
}

// Compute angular momentum (pure function)
template<typename Scalar>
std::array<Scalar, 3> computeAngularMomentum(const ImmutableParticleState<Scalar>& state) {
    std::array<Scalar, 3> L = {0, 0, 0};

    for (size_t i = 0; i < state.particleCount(); ++i) {
        const auto& pos = state.positions[i];
        const auto& vel = state.velocities[i];
        Scalar mass = state.masses[i];

        // L = r × p = r × (m * v)
        L[0] += mass * (pos[1] * vel[2] - pos[2] * vel[1]);
        L[1] += mass * (pos[2] * vel[0] - pos[0] * vel[2]);
        L[2] += mass * (pos[0] * vel[1] - pos[1] * vel[0]);
    }

    return L;
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

// Factory for electrostatic simulation
template<typename Scalar = float>
FunctionalSimulation<Scalar> createElectrostaticSimulation(
    Scalar dt,
    Scalar coulomb_constant = 8.99e9f
) {
    auto force_func = [coulomb_constant](const ImmutableParticleState<Scalar>& state) {
        return computeElectrostaticForces<Scalar>(state, coulomb_constant);
    };

    auto integration_func = [](const ImmutableParticleState<Scalar>& state,
                              const std::vector<std::array<Scalar, 3>>& forces,
                              Scalar timestep) {
        return verletIntegrationStep<Scalar>(state, forces, timestep);
    };

    return FunctionalSimulation<Scalar>(force_func, integration_func, dt);
}

// Factory for creating test particle states
template<typename Scalar = float>
ImmutableParticleState<Scalar> createRandomParticleState(
    size_t num_particles,
    Scalar domain_size = 1.0f
) {
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