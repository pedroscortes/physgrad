#include "functional_api.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace physgrad::functional {

// =============================================================================
// IMMUTABLE PARTICLE STATE IMPLEMENTATION
// =============================================================================

template<typename Scalar>
ImmutableParticleState<Scalar> ImmutableParticleState<Scalar>::create(
    std::vector<Vector4> pos,
    std::vector<Vector3> vel,
    std::vector<Scalar> charges,
    std::vector<Scalar> masses
) {
    // Compute energies
    Scalar kinetic = 0;
    for (size_t i = 0; i < vel.size(); ++i) {
        const auto& v = vel[i];
        Scalar v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        kinetic += 0.5f * masses[i] * v2;
    }

    Scalar potential = 0;
    const Scalar coulomb_k = 8.99e9f;
    for (size_t i = 0; i < pos.size(); ++i) {
        for (size_t j = i + 1; j < pos.size(); ++j) {
            Scalar dx = pos[i][0] - pos[j][0];
            Scalar dy = pos[i][1] - pos[j][1];
            Scalar dz = pos[i][2] - pos[j][2];
            Scalar r = std::sqrt(dx*dx + dy*dy + dz*dz);

            if (r > std::numeric_limits<Scalar>::epsilon()) {
                potential += coulomb_k * charges[i] * charges[j] / r;
            }
        }
    }

    Scalar total = kinetic + potential;

    // Temperature from kinetic energy (3/2 * k_B * T = <K.E.>)
    Scalar temperature = (2.0f / 3.0f) * kinetic / (1.38e-23f * pos.size());

    return ImmutableParticleState<Scalar>{
        .positions = std::move(pos),
        .velocities = std::move(vel),
        .charges = std::move(charges),
        .masses = std::move(masses),
        .total_energy = total,
        .kinetic_energy = kinetic,
        .potential_energy = potential,
        .temperature = temperature,
        .timestamp = 0
    };
}

template<typename Scalar>
ImmutableParticleState<Scalar> ImmutableParticleState<Scalar>::withUpdatedPositions(
    const std::vector<Vector4>& new_pos
) const {
    return create(new_pos, velocities, charges, masses).withTimestamp(timestamp);
}

template<typename Scalar>
ImmutableParticleState<Scalar> ImmutableParticleState<Scalar>::withUpdatedVelocities(
    const std::vector<Vector3>& new_vel
) const {
    return create(positions, new_vel, charges, masses).withTimestamp(timestamp);
}

template<typename Scalar>
ImmutableParticleState<Scalar> ImmutableParticleState<Scalar>::withTimestamp(size_t ts) const {
    auto result = *this;
    const_cast<size_t&>(result.timestamp) = ts;
    return result;
}

template<typename Scalar>
bool ImmutableParticleState<Scalar>::isValid() const noexcept {
    const size_t n = positions.size();

    // Check size consistency
    if (velocities.size() != n || charges.size() != n || masses.size() != n) {
        return false;
    }

    // Check for NaN/Inf values
    for (size_t i = 0; i < n; ++i) {
        for (int k = 0; k < 4; ++k) {
            if (!std::isfinite(positions[i][k])) return false;
        }
        for (int k = 0; k < 3; ++k) {
            if (!std::isfinite(velocities[i][k])) return false;
        }
        if (!std::isfinite(charges[i]) || !std::isfinite(masses[i])) return false;
        if (masses[i] <= 0) return false; // Mass must be positive
    }

    // Check energy values
    if (!std::isfinite(total_energy) || !std::isfinite(kinetic_energy) ||
        !std::isfinite(potential_energy) || !std::isfinite(temperature)) {
        return false;
    }

    return true;
}

template<typename Scalar>
typename ImmutableParticleState<Scalar>::GPUView
ImmutableParticleState<Scalar>::getGPUView() const noexcept {
    return GPUView{
        .positions_ptr = reinterpret_cast<const Scalar*>(positions.data()),
        .velocities_ptr = reinterpret_cast<const Scalar*>(velocities.data()),
        .charges_ptr = charges.data(),
        .masses_ptr = masses.data(),
        .count = particleCount()
    };
}

// Explicit instantiations
template class ImmutableParticleState<float>;
template class ImmutableParticleState<double>;

// =============================================================================
// FUNCTIONAL UTILITIES AND HELPERS
// =============================================================================

namespace utils {

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

// Compute virial (for virial theorem validation)
template<typename Scalar>
Scalar computeVirial(
    const ImmutableParticleState<Scalar>& state,
    const std::vector<std::array<Scalar, 3>>& forces
) {
    Scalar virial = 0;

    for (size_t i = 0; i < state.particleCount(); ++i) {
        const auto& pos = state.positions[i];
        const auto& force = forces[i];

        virial += pos[0] * force[0] + pos[1] * force[1] + pos[2] * force[2];
    }

    return virial;
}

// Statistical analysis of particle distribution
template<typename Scalar>
struct StatisticalAnalysis {
    std::array<Scalar, 3> mean_position;
    std::array<Scalar, 3> mean_velocity;
    std::array<Scalar, 3> std_position;
    std::array<Scalar, 3> std_velocity;
    Scalar mean_kinetic_energy;
    Scalar std_kinetic_energy;
};

template<typename Scalar>
StatisticalAnalysis<Scalar> analyzeDistribution(const ImmutableParticleState<Scalar>& state) {
    const size_t n = state.particleCount();
    if (n == 0) return {};

    StatisticalAnalysis<Scalar> analysis = {};

    // Compute means
    for (size_t i = 0; i < n; ++i) {
        const auto& pos = state.positions[i];
        const auto& vel = state.velocities[i];

        for (int k = 0; k < 3; ++k) {
            analysis.mean_position[k] += pos[k];
            analysis.mean_velocity[k] += vel[k];
        }

        Scalar ke = 0.5f * state.masses[i] *
                   (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
        analysis.mean_kinetic_energy += ke;
    }

    for (int k = 0; k < 3; ++k) {
        analysis.mean_position[k] /= n;
        analysis.mean_velocity[k] /= n;
    }
    analysis.mean_kinetic_energy /= n;

    // Compute standard deviations
    for (size_t i = 0; i < n; ++i) {
        const auto& pos = state.positions[i];
        const auto& vel = state.velocities[i];

        for (int k = 0; k < 3; ++k) {
            Scalar pos_dev = pos[k] - analysis.mean_position[k];
            Scalar vel_dev = vel[k] - analysis.mean_velocity[k];
            analysis.std_position[k] += pos_dev * pos_dev;
            analysis.std_velocity[k] += vel_dev * vel_dev;
        }

        Scalar ke = 0.5f * state.masses[i] *
                   (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
        Scalar ke_dev = ke - analysis.mean_kinetic_energy;
        analysis.std_kinetic_energy += ke_dev * ke_dev;
    }

    for (int k = 0; k < 3; ++k) {
        analysis.std_position[k] = std::sqrt(analysis.std_position[k] / n);
        analysis.std_velocity[k] = std::sqrt(analysis.std_velocity[k] / n);
    }
    analysis.std_kinetic_energy = std::sqrt(analysis.std_kinetic_energy / n);

    return analysis;
}

// Explicit instantiations
template std::array<float, 3> computeCenterOfMass(const ImmutableParticleState<float>&);
template std::array<double, 3> computeCenterOfMass(const ImmutableParticleState<double>&);
template std::array<float, 3> computeAngularMomentum(const ImmutableParticleState<float>&);
template std::array<double, 3> computeAngularMomentum(const ImmutableParticleState<double>&);
template float computeVirial(const ImmutableParticleState<float>&, const std::vector<std::array<float, 3>>&);
template double computeVirial(const ImmutableParticleState<double>&, const std::vector<std::array<double, 3>>&);
template StatisticalAnalysis<float> analyzeDistribution(const ImmutableParticleState<float>&);
template StatisticalAnalysis<double> analyzeDistribution(const ImmutableParticleState<double>&);

} // namespace utils

// =============================================================================
// OPTIMIZED FORCE COMPUTATION WITH FUNCTIONAL INTERFACE
// =============================================================================

namespace optimized {

// Fast multipole method approximation (pure function)
template<typename Scalar>
std::vector<std::array<Scalar, 3>> computeFastMultipoleForces(
    const ImmutableParticleState<Scalar>& state,
    Scalar theta = 0.5f,
    Scalar coulomb_constant = 8.99e9f
) {
    const size_t n = state.particleCount();
    std::vector<std::array<Scalar, 3>> forces(n, {0, 0, 0});

    // For now, implement direct summation as fallback
    // In a full implementation, this would use octree structure
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

// Optimized short-range cutoff forces (pure function)
template<typename Scalar>
std::vector<std::array<Scalar, 3>> computeCutoffForces(
    const ImmutableParticleState<Scalar>& state,
    Scalar cutoff_radius,
    Scalar coulomb_constant = 8.99e9f
) {
    const size_t n = state.particleCount();
    std::vector<std::array<Scalar, 3>> forces(n, {0, 0, 0});
    const Scalar cutoff2 = cutoff_radius * cutoff_radius;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const auto& ri = state.positions[i];
            const auto& rj = state.positions[j];

            Scalar dx = ri[0] - rj[0];
            Scalar dy = ri[1] - rj[1];
            Scalar dz = ri[2] - rj[2];
            Scalar r2 = dx*dx + dy*dy + dz*dz;

            if (r2 > cutoff2) continue;

            Scalar r = std::sqrt(r2);
            if (r < std::numeric_limits<Scalar>::epsilon()) continue;

            Scalar force_mag = coulomb_constant * state.charges[i] * state.charges[j] / (r2 * r);

            forces[i][0] += force_mag * dx;
            forces[i][1] += force_mag * dy;
            forces[i][2] += force_mag * dz;
            forces[j][0] -= force_mag * dx;
            forces[j][1] -= force_mag * dy;
            forces[j][2] -= force_mag * dz;
        }
    }

    return forces;
}

// Explicit instantiations
template std::vector<std::array<float, 3>> computeFastMultipoleForces(
    const ImmutableParticleState<float>&, float, float);
template std::vector<std::array<double, 3>> computeFastMultipoleForces(
    const ImmutableParticleState<double>&, double, double);
template std::vector<std::array<float, 3>> computeCutoffForces(
    const ImmutableParticleState<float>&, float, float);
template std::vector<std::array<double, 3>> computeCutoffForces(
    const ImmutableParticleState<double>&, double, double);

} // namespace optimized

} // namespace physgrad::functional