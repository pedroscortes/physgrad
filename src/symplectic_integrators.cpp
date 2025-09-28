#include "symplectic_integrators.h"
#include "logging_system.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>

namespace physgrad {

SymplecticIntegratorBase::SymplecticIntegratorBase(const SymplecticParams& p) : params(p) {
    energy_history.reserve(10000);
    momentum_history.reserve(10000);
}

void SymplecticIntegratorBase::computeConservationQuantities(
    const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
    const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
    const std::vector<float>& masses, float time) {

    current_quantities.kinetic_energy = SymplecticUtils::computeKineticEnergy(vel_x, vel_y, vel_z, masses);

    if (potential_function) {
        current_quantities.potential_energy = potential_function(pos_x, pos_y, pos_z, masses);
    } else {
        current_quantities.potential_energy = 0.0f;
    }

    current_quantities.total_energy = current_quantities.kinetic_energy + current_quantities.potential_energy;

    SymplecticUtils::computeLinearMomentum(vel_x, vel_y, vel_z, masses, current_quantities.linear_momentum);
    SymplecticUtils::computeAngularMomentum(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, current_quantities.angular_momentum);

    if (total_steps > 0) {
        current_quantities.energy_drift = std::abs(current_quantities.total_energy - initial_quantities.total_energy);

        float momentum_magnitude = std::sqrt(
            current_quantities.linear_momentum[0] * current_quantities.linear_momentum[0] +
            current_quantities.linear_momentum[1] * current_quantities.linear_momentum[1] +
            current_quantities.linear_momentum[2] * current_quantities.linear_momentum[2]
        );

        float initial_momentum_magnitude = std::sqrt(
            initial_quantities.linear_momentum[0] * initial_quantities.linear_momentum[0] +
            initial_quantities.linear_momentum[1] * initial_quantities.linear_momentum[1] +
            initial_quantities.linear_momentum[2] * initial_quantities.linear_momentum[2]
        );

        current_quantities.momentum_drift = std::abs(momentum_magnitude - initial_momentum_magnitude);

        current_quantities.conservation_violated =
            (current_quantities.energy_drift > params.energy_tolerance) ||
            (params.enable_momentum_conservation && current_quantities.momentum_drift > params.energy_tolerance);

        if (params.enable_energy_monitoring) {
            energy_history.push_back(current_quantities.total_energy);
            momentum_history.push_back(momentum_magnitude);
        }
    }
}

void SymplecticIntegratorBase::initializeConservationTracking(
    const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
    const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
    const std::vector<float>& masses) {

    computeConservationQuantities(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);
    initial_quantities = current_quantities;
    energy_history.clear();
    momentum_history.clear();
    total_steps = 0;
    rejected_steps = 0;
    average_step_size = 0.0f;
}

void SymplecticIntegratorBase::updateStatistics(float actual_dt) {
    total_steps++;
    average_step_size = (average_step_size * (total_steps - 1) + actual_dt) / total_steps;
}

bool SymplecticIntegratorBase::checkConservation(float energy_change, float momentum_change) {
    return (energy_change <= params.energy_tolerance) &&
           (!params.enable_momentum_conservation || momentum_change <= params.energy_tolerance);
}

float SymplecticIntegratorBase::adaptiveStepSize(float current_dt, float error_estimate) {
    if (!params.adaptive_time_stepping) return current_dt;

    float new_dt = current_dt * params.safety_factor * std::pow(params.energy_tolerance / error_estimate, 0.2f);
    return std::clamp(new_dt, params.min_time_step, params.max_time_step);
}

void SymplecticIntegratorBase::velocityKick(
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
    const std::vector<float>& force_x, const std::vector<float>& force_y, const std::vector<float>& force_z,
    const std::vector<float>& masses, float dt) {

    for (size_t i = 0; i < vel_x.size(); ++i) {
        float inv_mass = 1.0f / masses[i];
        vel_x[i] += force_x[i] * inv_mass * dt;
        vel_y[i] += force_y[i] * inv_mass * dt;
        vel_z[i] += force_z[i] * inv_mass * dt;
    }
}

void SymplecticIntegratorBase::positionDrift(
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
    float dt) {

    for (size_t i = 0; i < pos_x.size(); ++i) {
        pos_x[i] += vel_x[i] * dt;
        pos_y[i] += vel_y[i] * dt;
        pos_z[i] += vel_z[i] * dt;
    }
}

SymplecticEuler::SymplecticEuler(const SymplecticParams& params) : SymplecticIntegratorBase(params) {}

float SymplecticEuler::integrateStep(
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
    const std::vector<float>& masses,
    float dt, float time) {

    if (!force_function) return dt;

    std::vector<float> force_x(pos_x.size()), force_y(pos_x.size()), force_z(pos_x.size());

    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);

    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, dt);
    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, dt);

    updateStatistics(dt);
    if (params.enable_energy_monitoring) {
        computeConservationQuantities(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, time + dt);
    }

    return dt;
}

VelocityVerlet::VelocityVerlet(const SymplecticParams& params) : SymplecticIntegratorBase(params) {}

float VelocityVerlet::integrateStep(
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
    const std::vector<float>& masses,
    float dt, float time) {

    if (!force_function) return dt;

    size_t n = pos_x.size();
    std::vector<float> force_x_new(n), force_y_new(n), force_z_new(n);

    if (force_x_old.empty()) {
        force_x_old.resize(n);
        force_y_old.resize(n);
        force_z_old.resize(n);
        force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x_old, force_y_old, force_z_old, masses, time);
    }

    for (size_t i = 0; i < n; ++i) {
        float inv_mass = 1.0f / masses[i];

        pos_x[i] += vel_x[i] * dt + 0.5f * force_x_old[i] * inv_mass * dt * dt;
        pos_y[i] += vel_y[i] * dt + 0.5f * force_y_old[i] * inv_mass * dt * dt;
        pos_z[i] += vel_z[i] * dt + 0.5f * force_z_old[i] * inv_mass * dt * dt;
    }

    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x_new, force_y_new, force_z_new, masses, time + dt);

    for (size_t i = 0; i < n; ++i) {
        float inv_mass = 1.0f / masses[i];

        vel_x[i] += 0.5f * (force_x_old[i] + force_x_new[i]) * inv_mass * dt;
        vel_y[i] += 0.5f * (force_y_old[i] + force_y_new[i]) * inv_mass * dt;
        vel_z[i] += 0.5f * (force_z_old[i] + force_z_new[i]) * inv_mass * dt;
    }

    force_x_old = force_x_new;
    force_y_old = force_y_new;
    force_z_old = force_z_new;

    updateStatistics(dt);
    if (params.enable_energy_monitoring) {
        computeConservationQuantities(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, time + dt);
    }

    return dt;
}

ForestRuth::ForestRuth(const SymplecticParams& params) : SymplecticIntegratorBase(params) {}

float ForestRuth::integrateStep(
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
    const std::vector<float>& masses,
    float dt, float time) {

    if (!force_function) return dt;

    std::vector<float> force_x(pos_x.size()), force_y(pos_x.size()), force_z(pos_x.size());

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, theta * dt / 2.0f);

    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, theta * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, chi * dt / 2.0f);

    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, chi * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, (1.0f - 2.0f * (chi + theta)) * dt / 2.0f);

    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, chi * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, chi * dt / 2.0f);

    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, theta * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, theta * dt / 2.0f);

    updateStatistics(dt);
    if (params.enable_energy_monitoring) {
        computeConservationQuantities(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, time + dt);
    }

    return dt;
}

Yoshida4::Yoshida4(const SymplecticParams& params) : SymplecticIntegratorBase(params) {}

float Yoshida4::integrateStep(
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
    const std::vector<float>& masses,
    float dt, float time) {

    if (!force_function) return dt;

    std::vector<float> force_x(pos_x.size()), force_y(pos_x.size()), force_z(pos_x.size());

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, c1 * dt);

    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, d1 * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, c2 * dt);

    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, d2 * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, c3 * dt);

    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, d3 * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, c4 * dt);

    updateStatistics(dt);
    if (params.enable_energy_monitoring) {
        computeConservationQuantities(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, time + dt);
    }

    return dt;
}

BlanesMoan8::BlanesMoan8(const SymplecticParams& params) : SymplecticIntegratorBase(params) {}

float BlanesMoan8::integrateStep(
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
    const std::vector<float>& masses,
    float dt, float time) {

    if (!force_function) return dt;

    std::vector<float> force_x(pos_x.size()), force_y(pos_x.size()), force_z(pos_x.size());

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, a1 * dt);
    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, b1 * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, a2 * dt);
    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, b2 * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, a3 * dt);
    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, b3 * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, a4 * dt);
    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, b4 * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, a5 * dt);
    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, b5 * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, a6 * dt);
    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, b6 * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, a7 * dt);
    force_function(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, time);
    velocityKick(vel_x, vel_y, vel_z, force_x, force_y, force_z, masses, b7 * dt);

    positionDrift(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, a8 * dt);

    updateStatistics(dt);
    if (params.enable_energy_monitoring) {
        computeConservationQuantities(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, time + dt);
    }

    return dt;
}

std::unique_ptr<SymplecticIntegratorBase> SymplecticIntegratorFactory::create(
    SymplecticScheme scheme, const SymplecticParams& params) {

    switch (scheme) {
        case SymplecticScheme::SYMPLECTIC_EULER:
            return std::make_unique<SymplecticEuler>(params);
        case SymplecticScheme::VELOCITY_VERLET:
            return std::make_unique<VelocityVerlet>(params);
        case SymplecticScheme::FOREST_RUTH:
            return std::make_unique<ForestRuth>(params);
        case SymplecticScheme::YOSHIDA4:
            return std::make_unique<Yoshida4>(params);
        case SymplecticScheme::BLANES_MOAN8:
            return std::make_unique<BlanesMoan8>(params);
        default:
            return std::make_unique<VelocityVerlet>(params);
    }
}

std::string SymplecticIntegratorFactory::getSchemeDescription(SymplecticScheme scheme) {
    switch (scheme) {
        case SymplecticScheme::SYMPLECTIC_EULER: return "Symplectic Euler (1st order)";
        case SymplecticScheme::VELOCITY_VERLET: return "Velocity Verlet (2nd order)";
        case SymplecticScheme::FOREST_RUTH: return "Forest-Ruth (4th order)";
        case SymplecticScheme::YOSHIDA4: return "Yoshida (4th order)";
        case SymplecticScheme::BLANES_MOAN8: return "Blanes-Moan (8th order)";
        default: return "Unknown scheme";
    }
}

int SymplecticIntegratorFactory::getSchemeOrder(SymplecticScheme scheme) {
    switch (scheme) {
        case SymplecticScheme::SYMPLECTIC_EULER: return 1;
        case SymplecticScheme::VELOCITY_VERLET: return 2;
        case SymplecticScheme::FOREST_RUTH: return 4;
        case SymplecticScheme::YOSHIDA4: return 4;
        case SymplecticScheme::BLANES_MOAN8: return 8;
        default: return 2;
    }
}

namespace SymplecticUtils {

ForceFunction createGravitationalForce(float G, float softening) {
    return [G, softening](
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
        std::vector<float>& force_x, std::vector<float>& force_y, std::vector<float>& force_z,
        const std::vector<float>& masses, float) {

        int n = static_cast<int>(pos_x.size());
        std::fill(force_x.begin(), force_x.end(), 0.0f);
        std::fill(force_y.begin(), force_y.end(), 0.0f);
        std::fill(force_z.begin(), force_z.end(), 0.0f);

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                float dx = pos_x[j] - pos_x[i];
                float dy = pos_y[j] - pos_y[i];
                float dz = pos_z[j] - pos_z[i];
                float r_sq = dx*dx + dy*dy + dz*dz + softening*softening;
                float r = std::sqrt(r_sq);
                float force_mag = G * masses[i] * masses[j] / (r_sq * r);

                force_x[i] += force_mag * dx;
                force_y[i] += force_mag * dy;
                force_z[i] += force_mag * dz;

                force_x[j] -= force_mag * dx;
                force_y[j] -= force_mag * dy;
                force_z[j] -= force_mag * dz;
            }
        }
    };
}

ForceFunction createHarmonicOscillatorForce(float k, const float center[3]) {
    float center_x = center ? center[0] : 0.0f;
    float center_y = center ? center[1] : 0.0f;
    float center_z = center ? center[2] : 0.0f;

    return [k, center_x, center_y, center_z](
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
        std::vector<float>& force_x, std::vector<float>& force_y, std::vector<float>& force_z,
        const std::vector<float>&, float) {

        for (size_t i = 0; i < pos_x.size(); ++i) {
            force_x[i] = -k * (pos_x[i] - center_x);
            force_y[i] = -k * (pos_y[i] - center_y);
            force_z[i] = -k * (pos_z[i] - center_z);
        }
    };
}

PotentialFunction createGravitationalPotential(float G, float softening) {
    return [G, softening](
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& masses) -> float {

        float potential = 0.0f;
        int n = static_cast<int>(pos_x.size());

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                float dx = pos_x[j] - pos_x[i];
                float dy = pos_y[j] - pos_y[i];
                float dz = pos_z[j] - pos_z[i];
                float r = std::sqrt(dx*dx + dy*dy + dz*dz + softening*softening);
                potential -= G * masses[i] * masses[j] / r;
            }
        }

        return potential;
    };
}

PotentialFunction createHarmonicOscillatorPotential(float k, const float center[3]) {
    float center_x = center ? center[0] : 0.0f;
    float center_y = center ? center[1] : 0.0f;
    float center_z = center ? center[2] : 0.0f;

    return [k, center_x, center_y, center_z](
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>&) -> float {

        float potential = 0.0f;
        for (size_t i = 0; i < pos_x.size(); ++i) {
            float dx = pos_x[i] - center_x;
            float dy = pos_y[i] - center_y;
            float dz = pos_z[i] - center_z;
            potential += 0.5f * k * (dx*dx + dy*dy + dz*dz);
        }

        return potential;
    };
}

float computeKineticEnergy(
    const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
    const std::vector<float>& masses) {

    float kinetic_energy = 0.0f;
    for (size_t i = 0; i < vel_x.size(); ++i) {
        float v_squared = vel_x[i]*vel_x[i] + vel_y[i]*vel_y[i] + vel_z[i]*vel_z[i];
        kinetic_energy += 0.5f * masses[i] * v_squared;
    }
    return kinetic_energy;
}

void computeLinearMomentum(
    const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
    const std::vector<float>& masses, float momentum[3]) {

    momentum[0] = momentum[1] = momentum[2] = 0.0f;
    for (size_t i = 0; i < vel_x.size(); ++i) {
        momentum[0] += masses[i] * vel_x[i];
        momentum[1] += masses[i] * vel_y[i];
        momentum[2] += masses[i] * vel_z[i];
    }
}

void computeAngularMomentum(
    const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
    const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
    const std::vector<float>& masses, float angular_momentum[3], const float center[3]) {

    float center_x = center ? center[0] : 0.0f;
    float center_y = center ? center[1] : 0.0f;
    float center_z = center ? center[2] : 0.0f;

    angular_momentum[0] = angular_momentum[1] = angular_momentum[2] = 0.0f;

    for (size_t i = 0; i < pos_x.size(); ++i) {
        float r_x = pos_x[i] - center_x;
        float r_y = pos_y[i] - center_y;
        float r_z = pos_z[i] - center_z;

        float p_x = masses[i] * vel_x[i];
        float p_y = masses[i] * vel_y[i];
        float p_z = masses[i] * vel_z[i];

        angular_momentum[0] += r_y * p_z - r_z * p_y;
        angular_momentum[1] += r_z * p_x - r_x * p_z;
        angular_momentum[2] += r_x * p_y - r_y * p_x;
    }
}

void runConvergenceTest(
    SymplecticScheme scheme,
    const std::vector<float>& initial_positions_x,
    const std::vector<float>& initial_positions_y,
    const std::vector<float>& initial_positions_z,
    const std::vector<float>& initial_velocities_x,
    const std::vector<float>& initial_velocities_y,
    const std::vector<float>& initial_velocities_z,
    const std::vector<float>& masses,
    ForceFunction force_func,
    PotentialFunction potential_func,
    float simulation_time,
    int num_step_sizes) {

    Logger::getInstance().info("symplectic", "Running convergence test for " +
                              SymplecticIntegratorFactory::getSchemeDescription(scheme));

    std::vector<float> step_sizes;
    std::vector<float> energy_errors;

    float base_dt = simulation_time / 100.0f;

    for (int i = 0; i < num_step_sizes; ++i) {
        float dt = base_dt / std::pow(2.0f, i);
        step_sizes.push_back(dt);

        SymplecticParams params;
        params.time_step = dt;
        params.enable_energy_monitoring = true;

        auto integrator = SymplecticIntegratorFactory::create(scheme, params);
        integrator->setForceFunction(force_func);
        integrator->setPotentialFunction(potential_func);

        auto pos_x = initial_positions_x;
        auto pos_y = initial_positions_y;
        auto pos_z = initial_positions_z;
        auto vel_x = initial_velocities_x;
        auto vel_y = initial_velocities_y;
        auto vel_z = initial_velocities_z;

        integrator->initializeConservationTracking(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);

        float time = 0.0f;
        while (time < simulation_time) {
            integrator->integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, dt, time);
            time += dt;
        }

        float energy_error = integrator->getCurrentQuantities().energy_drift;
        energy_errors.push_back(energy_error);

        Logger::getInstance().info("symplectic",
            "Step size: " + std::to_string(dt) +
            ", Energy error: " + std::to_string(energy_error) +
            ", Steps: " + std::to_string(integrator->getTotalSteps()));
    }
}

void compareIntegrators(
    const std::vector<SymplecticScheme>& schemes,
    const std::vector<float>& initial_positions_x,
    const std::vector<float>& initial_positions_y,
    const std::vector<float>& initial_positions_z,
    const std::vector<float>& initial_velocities_x,
    const std::vector<float>& initial_velocities_y,
    const std::vector<float>& initial_velocities_z,
    const std::vector<float>& masses,
    ForceFunction force_func,
    PotentialFunction potential_func,
    float simulation_time,
    float time_step) {

    Logger::getInstance().info("symplectic", "Comparing " + std::to_string(schemes.size()) + " integrators");

    for (auto scheme : schemes) {
        SymplecticParams params;
        params.time_step = time_step;
        params.enable_energy_monitoring = true;

        auto integrator = SymplecticIntegratorFactory::create(scheme, params);
        integrator->setForceFunction(force_func);
        integrator->setPotentialFunction(potential_func);

        auto pos_x = initial_positions_x;
        auto pos_y = initial_positions_y;
        auto pos_z = initial_positions_z;
        auto vel_x = initial_velocities_x;
        auto vel_y = initial_velocities_y;
        auto vel_z = initial_velocities_z;

        integrator->initializeConservationTracking(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);

        auto start_time = std::chrono::high_resolution_clock::now();

        float time = 0.0f;
        while (time < simulation_time) {
            integrator->integrateStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, time_step, time);
            time += time_step;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<float>(end_time - start_time).count();

        Logger::getInstance().info("symplectic",
            SymplecticIntegratorFactory::getSchemeDescription(scheme) + ":" +
            " Energy error: " + std::to_string(integrator->getCurrentQuantities().energy_drift) +
            ", Runtime: " + std::to_string(duration) + "s" +
            ", Steps: " + std::to_string(integrator->getTotalSteps()));
    }
}

} // namespace SymplecticUtils

} // namespace physgrad