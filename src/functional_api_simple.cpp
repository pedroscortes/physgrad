#include "functional_api_simple.h"
#include <algorithm>
#include <numeric>
#include <cmath>

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
    Scalar temperature = (2.0f / 3.0f) * kinetic / (1.38e-23f * pos.size());

    ImmutableParticleState<Scalar> result;
    result.positions = std::move(pos);
    result.velocities = std::move(vel);
    result.charges = std::move(charges);
    result.masses = std::move(masses);
    result.total_energy = total;
    result.kinetic_energy = kinetic;
    result.potential_energy = potential;
    result.temperature = temperature;
    result.timestamp = 0;
    return result;
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
    result.timestamp = ts;
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

// Explicit instantiations
template class ImmutableParticleState<float>;
template class ImmutableParticleState<double>;

// Explicit instantiations for template functions
template std::vector<std::array<float, 3>> computeElectrostaticForces(const ImmutableParticleState<float>&, float);
template std::vector<std::array<double, 3>> computeElectrostaticForces(const ImmutableParticleState<double>&, double);

template ImmutableParticleState<float> verletIntegrationStep(const ImmutableParticleState<float>&, const std::vector<std::array<float, 3>>&, float);
template ImmutableParticleState<double> verletIntegrationStep(const ImmutableParticleState<double>&, const std::vector<std::array<double, 3>>&, double);

template std::tuple<float, float, float> computeTotalEnergy(const ImmutableParticleState<float>&, float);
template std::tuple<double, double, double> computeTotalEnergy(const ImmutableParticleState<double>&, double);

template std::array<float, 3> computeCenterOfMass(const ImmutableParticleState<float>&);
template std::array<double, 3> computeCenterOfMass(const ImmutableParticleState<double>&);

template std::array<float, 3> computeAngularMomentum(const ImmutableParticleState<float>&);
template std::array<double, 3> computeAngularMomentum(const ImmutableParticleState<double>&);

template FunctionalSimulation<float> createElectrostaticSimulation(float, float);
template FunctionalSimulation<double> createElectrostaticSimulation(double, double);

template ImmutableParticleState<float> createRandomParticleState(size_t, float);
template ImmutableParticleState<double> createRandomParticleState(size_t, double);

} // namespace physgrad::functional