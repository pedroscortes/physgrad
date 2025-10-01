/**
 * PhysGrad Fluid-Structure Interaction (FSI) Coupling
 *
 * Advanced coupling framework for fluid-solid interactions using
 * immersed boundary methods, partitioned schemes, and adaptive algorithms
 */

#pragma once

#include <vector>
#include <memory>
#include <array>
#include <functional>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <chrono>

#include "mpm_data_structures.h"
#include "sparse_data_structures.h"

// Simple Vec3 template for testing
template<typename T>
struct Vec3 {
    T x, y, z;

    Vec3(T x_ = T{0}, T y_ = T{0}, T z_ = T{0}) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& other) const {
        return Vec3{x + other.x, y + other.y, z + other.z};
    }

    Vec3 operator-(const Vec3& other) const {
        return Vec3{x - other.x, y - other.y, z - other.z};
    }

    Vec3 operator*(T scalar) const {
        return Vec3{x * scalar, y * scalar, z * scalar};
    }

    T dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    T magnitude() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    Vec3 normalized() const {
        T mag = magnitude();
        return (mag > T{0}) ? (*this) * (T{1} / mag) : Vec3{};
    }
};

namespace physgrad::fsi {

// =============================================================================
// FSI COUPLING INTERFACE AND BASE CLASSES
// =============================================================================

/**
 * Abstract base class for FSI coupling methods
 * Defines the interface for different coupling strategies
 */
template<typename T>
class FSICouplingMethod {
public:
    using StateVector = std::vector<T>;
    using ForceVector = std::vector<T>;
    using ParticleSet = mpm::ParticleAoSoA<T>;
    using SpatialHash = sparse::SpatialHashGrid<T, uint32_t>;

    virtual ~FSICouplingMethod() = default;

    // Core coupling interface
    virtual void couple(ParticleSet& fluid_particles, ParticleSet& solid_particles,
                       T time_step, T current_time) = 0;

    // Force transfer methods
    virtual ForceVector computeFluidForces(const ParticleSet& fluid_particles,
                                         const ParticleSet& solid_particles) = 0;
    virtual ForceVector computeSolidForces(const ParticleSet& fluid_particles,
                                         const ParticleSet& solid_particles) = 0;

    // Constraint enforcement
    virtual void enforceNoSlipBoundary(ParticleSet& fluid_particles,
                                     const ParticleSet& solid_particles) = 0;
    virtual void enforceContinuityConstraint(ParticleSet& fluid_particles,
                                           ParticleSet& solid_particles) = 0;

    // Configuration and parameters
    virtual void setParameters(const std::unordered_map<std::string, T>& params) = 0;
    virtual std::unordered_map<std::string, T> getParameters() const = 0;
};

// =============================================================================
// IMMERSED BOUNDARY METHOD (IBM) IMPLEMENTATION
// =============================================================================

/**
 * Immersed Boundary Method for FSI coupling
 * Uses distributed Lagrange multipliers and smooth delta functions
 */
template<typename T>
class ImmersedBoundaryMethod : public FSICouplingMethod<T> {
private:
    using Base = FSICouplingMethod<T>;
    using typename Base::StateVector;
    using typename Base::ForceVector;
    using typename Base::ParticleSet;
    using typename Base::SpatialHash;

    // IBM parameters
    T support_radius_;        // Delta function support radius
    T coupling_strength_;     // Force coupling coefficient
    T damping_factor_;        // Velocity damping at interface
    bool use_adaptive_radius_; // Adaptive support radius based on local density

    // Spatial acceleration
    std::unique_ptr<SpatialHash> spatial_hash_;
    T hash_cell_size_;

    // Force distribution data
    std::vector<T> force_distribution_weights_;
    std::vector<uint32_t> force_distribution_indices_;

public:
    ImmersedBoundaryMethod(T support_radius = 2.0f, T coupling_strength = 1.0f)
        : support_radius_(support_radius), coupling_strength_(coupling_strength),
          damping_factor_(0.1f), use_adaptive_radius_(false) {

        hash_cell_size_ = support_radius_ * 0.5f;
    }

    void couple(ParticleSet& fluid_particles, ParticleSet& solid_particles,
                T time_step, T current_time) override {

        // Update spatial acceleration structures
        updateSpatialHash(fluid_particles, solid_particles);

        // Compute interaction forces
        auto fluid_forces = computeFluidForces(fluid_particles, solid_particles);
        auto solid_forces = computeSolidForces(fluid_particles, solid_particles);

        // Apply forces to particles
        applyForcesToParticles(fluid_particles, fluid_forces);
        applyForcesToParticles(solid_particles, solid_forces);

        // Enforce constraints
        enforceNoSlipBoundary(fluid_particles, solid_particles);
        enforceContinuityConstraint(fluid_particles, solid_particles);
    }

    ForceVector computeFluidForces(const ParticleSet& fluid_particles,
                                  const ParticleSet& solid_particles) override {

        size_t num_fluid = fluid_particles.size();
        ForceVector forces(num_fluid * 3, T{0});

        // For each fluid particle, compute forces from nearby solid particles
        for (size_t i = 0; i < num_fluid; ++i) {
            T fx, fy, fz;
            fluid_particles.getPosition(i, fx, fy, fz);

            auto solid_neighbors = getSolidNeighbors(fx, fy, fz, solid_particles);

            T total_force_x = 0, total_force_y = 0, total_force_z = 0;

            for (uint32_t j : solid_neighbors) {
                T sx, sy, sz, svx, svy, svz;
                solid_particles.getPosition(j, sx, sy, sz);
                solid_particles.getVelocity(j, svx, svy, svz);

                // Distance and delta function
                T dx = fx - sx, dy = fy - sy, dz = fz - sz;
                T distance = std::sqrt(dx*dx + dy*dy + dz*dz);

                if (distance < support_radius_) {
                    T delta_weight = deltaFunction(distance);
                    T solid_mass = solid_particles.getMass(j);

                    // Fluid velocity
                    T fvx, fvy, fvz;
                    fluid_particles.getVelocity(i, fvx, fvy, fvz);

                    // Velocity difference
                    T dvx = svx - fvx, dvy = svy - fvy, dvz = svz - fvz;

                    // Force based on velocity difference and solid mass
                    T force_magnitude = coupling_strength_ * solid_mass * delta_weight;

                    total_force_x += force_magnitude * dvx;
                    total_force_y += force_magnitude * dvy;
                    total_force_z += force_magnitude * dvz;
                }
            }

            forces[i * 3] = total_force_x;
            forces[i * 3 + 1] = total_force_y;
            forces[i * 3 + 2] = total_force_z;
        }

        return forces;
    }

    ForceVector computeSolidForces(const ParticleSet& fluid_particles,
                                  const ParticleSet& solid_particles) override {

        size_t num_solid = solid_particles.size();
        ForceVector forces(num_solid * 3, T{0});

        // For each solid particle, compute forces from nearby fluid particles
        for (size_t i = 0; i < num_solid; ++i) {
            T sx, sy, sz;
            solid_particles.getPosition(i, sx, sy, sz);

            auto fluid_neighbors = getFluidNeighbors(sx, sy, sz, fluid_particles);

            T total_force_x = 0, total_force_y = 0, total_force_z = 0;

            for (uint32_t j : fluid_neighbors) {
                T fx, fy, fz, fvx, fvy, fvz;
                fluid_particles.getPosition(j, fx, fy, fz);
                fluid_particles.getVelocity(j, fvx, fvy, fvz);

                // Distance and delta function
                T dx = sx - fx, dy = sy - fy, dz = sz - fz;
                T distance = std::sqrt(dx*dx + dy*dy + dz*dz);

                if (distance < support_radius_) {
                    T delta_weight = deltaFunction(distance);
                    T fluid_mass = fluid_particles.getMass(j);

                    // Solid velocity
                    T svx, svy, svz;
                    solid_particles.getVelocity(i, svx, svy, svz);

                    // Velocity difference (Newton's third law)
                    T dvx = fvx - svx, dvy = fvy - svy, dvz = fvz - svz;

                    // Force based on fluid pressure and velocity
                    T force_magnitude = coupling_strength_ * fluid_mass * delta_weight;

                    total_force_x += force_magnitude * dvx;
                    total_force_y += force_magnitude * dvy;
                    total_force_z += force_magnitude * dvz;
                }
            }

            forces[i * 3] = total_force_x;
            forces[i * 3 + 1] = total_force_y;
            forces[i * 3 + 2] = total_force_z;
        }

        return forces;
    }

    void enforceNoSlipBoundary(ParticleSet& fluid_particles,
                              const ParticleSet& solid_particles) override {

        size_t num_fluid = fluid_particles.size();

        for (size_t i = 0; i < num_fluid; ++i) {
            T fx, fy, fz;
            fluid_particles.getPosition(i, fx, fy, fz);

            auto solid_neighbors = getSolidNeighbors(fx, fy, fz, solid_particles);

            if (!solid_neighbors.empty()) {
                T weighted_velocity_x = 0, weighted_velocity_y = 0, weighted_velocity_z = 0;
                T total_weight = 0;

                // Compute weighted average of solid velocities
                for (uint32_t j : solid_neighbors) {
                    T sx, sy, sz, svx, svy, svz;
                    solid_particles.getPosition(j, sx, sy, sz);
                    solid_particles.getVelocity(j, svx, svy, svz);

                    T dx = fx - sx, dy = fy - sy, dz = fz - sz;
                    T distance = std::sqrt(dx*dx + dy*dy + dz*dz);

                    if (distance < support_radius_) {
                        T weight = deltaFunction(distance);
                        total_weight += weight;

                        weighted_velocity_x += weight * svx;
                        weighted_velocity_y += weight * svy;
                        weighted_velocity_z += weight * svz;
                    }
                }

                if (total_weight > 1e-8) {
                    T target_vx = weighted_velocity_x / total_weight;
                    T target_vy = weighted_velocity_y / total_weight;
                    T target_vz = weighted_velocity_z / total_weight;

                    // Current fluid velocity
                    T fvx, fvy, fvz;
                    fluid_particles.getVelocity(i, fvx, fvy, fvz);

                    // Apply damped correction towards solid velocity
                    T new_vx = fvx + damping_factor_ * (target_vx - fvx);
                    T new_vy = fvy + damping_factor_ * (target_vy - fvy);
                    T new_vz = fvz + damping_factor_ * (target_vz - fvz);

                    fluid_particles.setVelocity(i, new_vx, new_vy, new_vz);
                }
            }
        }
    }

    void enforceContinuityConstraint(ParticleSet& fluid_particles,
                                   ParticleSet& solid_particles) override {

        // Simple continuity enforcement through density correction
        size_t num_fluid = fluid_particles.size();

        for (size_t i = 0; i < num_fluid; ++i) {
            T fx, fy, fz;
            fluid_particles.getPosition(i, fx, fy, fz);

            // Count fluid neighbors for density estimation
            auto fluid_neighbors = getFluidNeighbors(fx, fy, fz, fluid_particles);
            auto solid_neighbors = getSolidNeighbors(fx, fy, fz, solid_particles);

            // Adjust fluid mass based on local solid volume fraction
            T solid_volume_fraction = calculateSolidVolumeFraction(
                fx, fy, fz, solid_particles, support_radius_);

            if (solid_volume_fraction > 0.1) {
                // Reduce effective fluid density in solid regions
                T current_mass = fluid_particles.getMass(i);
                T adjusted_mass = current_mass * (1.0 - solid_volume_fraction * 0.5);
                fluid_particles.setMass(i, adjusted_mass);
            }
        }
    }

    void setParameters(const std::unordered_map<std::string, T>& params) override {
        auto it = params.find("support_radius");
        if (it != params.end()) support_radius_ = it->second;

        it = params.find("coupling_strength");
        if (it != params.end()) coupling_strength_ = it->second;

        it = params.find("damping_factor");
        if (it != params.end()) damping_factor_ = it->second;

        it = params.find("use_adaptive_radius");
        if (it != params.end()) use_adaptive_radius_ = (it->second > 0.5);
    }

    std::unordered_map<std::string, T> getParameters() const override {
        return {
            {"support_radius", support_radius_},
            {"coupling_strength", coupling_strength_},
            {"damping_factor", damping_factor_},
            {"use_adaptive_radius", use_adaptive_radius_ ? T{1} : T{0}}
        };
    }

private:
    // Smooth delta function for force distribution
    T deltaFunction(T distance) const {
        if (distance >= support_radius_) return 0.0;

        T r = distance / support_radius_;

        // Cosine-based delta function
        if (r < 1.0) {
            return (1.0 + std::cos(M_PI * r)) / (2.0 * support_radius_);
        }
        return 0.0;
    }

    void updateSpatialHash(const ParticleSet& fluid_particles,
                          const ParticleSet& solid_particles) {

        // Combine all particles for spatial hashing
        size_t total_particles = fluid_particles.size() + solid_particles.size();
        std::vector<T> all_positions(total_particles * 3);

        // Copy fluid positions
        for (size_t i = 0; i < fluid_particles.size(); ++i) {
            T x, y, z;
            fluid_particles.getPosition(i, x, y, z);
            all_positions[i * 3] = x;
            all_positions[i * 3 + 1] = y;
            all_positions[i * 3 + 2] = z;
        }

        // Copy solid positions
        for (size_t i = 0; i < solid_particles.size(); ++i) {
            size_t idx = fluid_particles.size() + i;
            T x, y, z;
            solid_particles.getPosition(i, x, y, z);
            all_positions[idx * 3] = x;
            all_positions[idx * 3 + 1] = y;
            all_positions[idx * 3 + 2] = z;
        }

        // Determine domain bounds
        std::array<T, 3> domain_min = {1e6, 1e6, 1e6};
        std::array<T, 3> domain_max = {-1e6, -1e6, -1e6};

        for (size_t i = 0; i < total_particles; ++i) {
            for (int d = 0; d < 3; ++d) {
                domain_min[d] = std::min(domain_min[d], all_positions[i * 3 + d]);
                domain_max[d] = std::max(domain_max[d], all_positions[i * 3 + d]);
            }
        }

        // Add margin
        for (int d = 0; d < 3; ++d) {
            domain_min[d] -= support_radius_;
            domain_max[d] += support_radius_;
        }

        // Create spatial hash
        spatial_hash_ = std::make_unique<SpatialHash>(
            hash_cell_size_, domain_min, domain_max, total_particles);
        spatial_hash_->buildHashTable(all_positions, total_particles);
    }

    std::vector<uint32_t> getFluidNeighbors(T x, T y, T z,
                                           const ParticleSet& fluid_particles) const {
        if (!spatial_hash_) return {};

        auto all_neighbors = spatial_hash_->getNeighbors(x, y, z, support_radius_);
        std::vector<uint32_t> fluid_neighbors;

        for (uint32_t idx : all_neighbors) {
            if (idx < fluid_particles.size()) {
                fluid_neighbors.push_back(idx);
            }
        }

        return fluid_neighbors;
    }

    std::vector<uint32_t> getSolidNeighbors(T x, T y, T z,
                                           const ParticleSet& solid_particles) const {
        if (!spatial_hash_) return {};

        auto all_neighbors = spatial_hash_->getNeighbors(x, y, z, support_radius_);
        std::vector<uint32_t> solid_neighbors;
        size_t fluid_offset = spatial_hash_->getParticleIndices().size() - solid_particles.size();

        for (uint32_t idx : all_neighbors) {
            if (idx >= fluid_offset) {
                solid_neighbors.push_back(idx - fluid_offset);
            }
        }

        return solid_neighbors;
    }

    T calculateSolidVolumeFraction(T x, T y, T z, const ParticleSet& solid_particles,
                                  T radius) const {
        auto solid_neighbors = getSolidNeighbors(x, y, z, solid_particles);

        T total_volume = 0.0;
        T search_volume = (4.0/3.0) * M_PI * radius * radius * radius;

        for (uint32_t idx : solid_neighbors) {
            T sx, sy, sz;
            solid_particles.getPosition(idx, sx, sy, sz);

            T dx = x - sx, dy = y - sy, dz = z - sz;
            T distance = std::sqrt(dx*dx + dy*dy + dz*dz);

            if (distance < radius) {
                T particle_volume = solid_particles.getVolume(idx);
                total_volume += particle_volume;
            }
        }

        return std::min(T{1}, total_volume / search_volume);
    }

    void applyForcesToParticles(ParticleSet& particles, const ForceVector& forces) {
        size_t num_particles = particles.size();

        for (size_t i = 0; i < num_particles; ++i) {
            T vx, vy, vz;
            particles.getVelocity(i, vx, vy, vz);

            T mass = particles.getMass(i);
            T dt = 0.001; // This should be passed as parameter

            // Apply forces (F = ma, so a = F/m, v_new = v_old + a*dt)
            T ax = forces[i * 3] / mass;
            T ay = forces[i * 3 + 1] / mass;
            T az = forces[i * 3 + 2] / mass;

            particles.setVelocity(i, vx + ax * dt, vy + ay * dt, vz + az * dt);
        }
    }
};

// =============================================================================
// PARTITIONED COUPLING SCHEME
// =============================================================================

/**
 * Partitioned coupling scheme for FSI
 * Separates fluid and solid solvers with interface exchange
 */
template<typename T>
class PartitionedCouplingScheme : public FSICouplingMethod<T> {
private:
    using Base = FSICouplingMethod<T>;
    using typename Base::StateVector;
    using typename Base::ForceVector;
    using typename Base::ParticleSet;

    // Coupling parameters
    T relaxation_factor_;      // Under-relaxation for stability
    uint32_t max_iterations_;  // Maximum coupling iterations
    T convergence_tolerance_;  // Convergence criterion
    bool use_aitken_acceleration_; // Aitken acceleration for convergence

    // Interface data
    std::vector<T> interface_forces_;
    std::vector<T> interface_velocities_;
    std::vector<T> interface_displacements_;

public:
    PartitionedCouplingScheme(T relaxation_factor = 0.7, uint32_t max_iterations = 10)
        : relaxation_factor_(relaxation_factor), max_iterations_(max_iterations),
          convergence_tolerance_(1e-6), use_aitken_acceleration_(true) {}

    void couple(ParticleSet& fluid_particles, ParticleSet& solid_particles,
                T time_step, T current_time) override {

        // Fixed-point iteration for strong coupling
        std::vector<T> fluid_velocities_old, solid_displacements_old;

        for (uint32_t iter = 0; iter < max_iterations_; ++iter) {
            // Store previous iteration values
            if (iter == 0) {
                fluid_velocities_old = extractVelocities(fluid_particles);
                solid_displacements_old = extractPositions(solid_particles);
            }

            // Solve fluid with current solid interface conditions
            auto fluid_forces = computeFluidForces(fluid_particles, solid_particles);
            updateFluidVelocities(fluid_particles, fluid_forces, time_step);

            // Solve solid with current fluid interface conditions
            auto solid_forces = computeSolidForces(fluid_particles, solid_particles);
            updateSolidPositions(solid_particles, solid_forces, time_step);

            // Check convergence
            auto fluid_velocities_new = extractVelocities(fluid_particles);
            auto solid_displacements_new = extractPositions(solid_particles);

            T fluid_residual = computeResidual(fluid_velocities_old, fluid_velocities_new);
            T solid_residual = computeResidual(solid_displacements_old, solid_displacements_new);

            if (fluid_residual < convergence_tolerance_ && solid_residual < convergence_tolerance_) {
                break; // Converged
            }

            // Apply relaxation
            if (iter > 0) {
                applyRelaxation(fluid_particles, fluid_velocities_old, relaxation_factor_);
                applyRelaxation(solid_particles, solid_displacements_old, relaxation_factor_);
            }

            fluid_velocities_old = fluid_velocities_new;
            solid_displacements_old = solid_displacements_new;
        }

        // Enforce final constraints
        enforceNoSlipBoundary(fluid_particles, solid_particles);
        enforceContinuityConstraint(fluid_particles, solid_particles);
    }

    ForceVector computeFluidForces(const ParticleSet& fluid_particles,
                                  const ParticleSet& solid_particles) override {
        // Simplified implementation - would use actual fluid solver
        size_t num_fluid = fluid_particles.size();
        ForceVector forces(num_fluid * 3, T{0});

        // Apply pressure forces and viscous forces from solid interface
        // This is a placeholder for actual FSI force computation
        for (size_t i = 0; i < num_fluid; ++i) {
            forces[i * 3] = 0.0;     // X force
            forces[i * 3 + 1] = -9.81; // Gravity in Y
            forces[i * 3 + 2] = 0.0;   // Z force
        }

        return forces;
    }

    ForceVector computeSolidForces(const ParticleSet& fluid_particles,
                                  const ParticleSet& solid_particles) override {
        // Simplified implementation - would use actual solid solver
        size_t num_solid = solid_particles.size();
        ForceVector forces(num_solid * 3, T{0});

        // Apply fluid pressure and viscous forces to solid interface
        for (size_t i = 0; i < num_solid; ++i) {
            forces[i * 3] = 0.0;     // X force
            forces[i * 3 + 1] = -9.81; // Gravity in Y
            forces[i * 3 + 2] = 0.0;   // Z force
        }

        return forces;
    }

    void enforceNoSlipBoundary(ParticleSet& fluid_particles,
                              const ParticleSet& solid_particles) override {
        // Match fluid velocity to solid velocity at interface
        // Simplified implementation
    }

    void enforceContinuityConstraint(ParticleSet& fluid_particles,
                                   ParticleSet& solid_particles) override {
        // Enforce mass conservation at interface
        // Simplified implementation
    }

    void setParameters(const std::unordered_map<std::string, T>& params) override {
        auto it = params.find("relaxation_factor");
        if (it != params.end()) relaxation_factor_ = it->second;

        it = params.find("max_iterations");
        if (it != params.end()) max_iterations_ = static_cast<uint32_t>(it->second);

        it = params.find("convergence_tolerance");
        if (it != params.end()) convergence_tolerance_ = it->second;
    }

    std::unordered_map<std::string, T> getParameters() const override {
        return {
            {"relaxation_factor", relaxation_factor_},
            {"max_iterations", static_cast<T>(max_iterations_)},
            {"convergence_tolerance", convergence_tolerance_}
        };
    }

private:
    std::vector<T> extractVelocities(const ParticleSet& particles) const {
        size_t num_particles = particles.size();
        std::vector<T> velocities(num_particles * 3);

        for (size_t i = 0; i < num_particles; ++i) {
            T vx, vy, vz;
            particles.getVelocity(i, vx, vy, vz);
            velocities[i * 3] = vx;
            velocities[i * 3 + 1] = vy;
            velocities[i * 3 + 2] = vz;
        }

        return velocities;
    }

    std::vector<T> extractPositions(const ParticleSet& particles) const {
        size_t num_particles = particles.size();
        std::vector<T> positions(num_particles * 3);

        for (size_t i = 0; i < num_particles; ++i) {
            T x, y, z;
            particles.getPosition(i, x, y, z);
            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;
        }

        return positions;
    }

    T computeResidual(const std::vector<T>& old_values, const std::vector<T>& new_values) const {
        T residual = 0.0;
        for (size_t i = 0; i < old_values.size(); ++i) {
            T diff = new_values[i] - old_values[i];
            residual += diff * diff;
        }
        return std::sqrt(residual / old_values.size());
    }

    void updateFluidVelocities(ParticleSet& particles, const ForceVector& forces, T dt) {
        size_t num_particles = particles.size();

        for (size_t i = 0; i < num_particles; ++i) {
            T vx, vy, vz;
            particles.getVelocity(i, vx, vy, vz);

            T mass = particles.getMass(i);
            T ax = forces[i * 3] / mass;
            T ay = forces[i * 3 + 1] / mass;
            T az = forces[i * 3 + 2] / mass;

            particles.setVelocity(i, vx + ax * dt, vy + ay * dt, vz + az * dt);
        }
    }

    void updateSolidPositions(ParticleSet& particles, const ForceVector& forces, T dt) {
        size_t num_particles = particles.size();

        for (size_t i = 0; i < num_particles; ++i) {
            T x, y, z, vx, vy, vz;
            particles.getPosition(i, x, y, z);
            particles.getVelocity(i, vx, vy, vz);

            T mass = particles.getMass(i);
            T ax = forces[i * 3] / mass;
            T ay = forces[i * 3 + 1] / mass;
            T az = forces[i * 3 + 2] / mass;

            // Update velocity and position
            T new_vx = vx + ax * dt;
            T new_vy = vy + ay * dt;
            T new_vz = vz + az * dt;

            particles.setVelocity(i, new_vx, new_vy, new_vz);
            particles.setPosition(i, x + new_vx * dt, y + new_vy * dt, z + new_vz * dt);
        }
    }

    void applyRelaxation(ParticleSet& particles, const std::vector<T>& old_values, T factor) {
        size_t num_particles = particles.size();

        for (size_t i = 0; i < num_particles; ++i) {
            T vx, vy, vz;
            particles.getVelocity(i, vx, vy, vz);

            T relaxed_vx = factor * old_values[i * 3] + (1 - factor) * vx;
            T relaxed_vy = factor * old_values[i * 3 + 1] + (1 - factor) * vy;
            T relaxed_vz = factor * old_values[i * 3 + 2] + (1 - factor) * vz;

            particles.setVelocity(i, relaxed_vx, relaxed_vy, relaxed_vz);
        }
    }
};

// =============================================================================
// FSI COUPLING FACTORY AND UTILITIES
// =============================================================================

/**
 * Factory for creating FSI coupling methods
 */
template<typename T>
class FSICouplingFactory {
public:
    enum class CouplingType {
        IMMERSED_BOUNDARY,
        PARTITIONED_SCHEME,
        MONOLITHIC  // Future implementation
    };

    static std::unique_ptr<FSICouplingMethod<T>> create(CouplingType type,
                                                        const std::unordered_map<std::string, T>& params = {}) {
        switch (type) {
            case CouplingType::IMMERSED_BOUNDARY: {
                auto method = std::make_unique<ImmersedBoundaryMethod<T>>();
                method->setParameters(params);
                return method;
            }
            case CouplingType::PARTITIONED_SCHEME: {
                auto method = std::make_unique<PartitionedCouplingScheme<T>>();
                method->setParameters(params);
                return method;
            }
            default:
                return nullptr;
        }
    }
};

/**
 * FSI Simulation Manager
 * Coordinates multiple coupling methods and simulation components
 */
template<typename T>
class FSISimulationManager {
private:
    std::unique_ptr<FSICouplingMethod<T>> coupling_method_;
    T time_step_;
    T current_time_;
    T total_time_;

    // Performance monitoring
    std::vector<T> coupling_times_;
    std::vector<T> energy_history_;

public:
    FSISimulationManager(std::unique_ptr<FSICouplingMethod<T>> coupling_method,
                        T time_step = 0.001, T total_time = 1.0)
        : coupling_method_(std::move(coupling_method)), time_step_(time_step),
          current_time_(0.0), total_time_(total_time) {}

    void simulate(mpm::ParticleAoSoA<T>& fluid_particles,
                 mpm::ParticleAoSoA<T>& solid_particles) {

        coupling_times_.clear();
        energy_history_.clear();

        while (current_time_ < total_time_) {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Perform FSI coupling
            coupling_method_->couple(fluid_particles, solid_particles, time_step_, current_time_);

            auto end_time = std::chrono::high_resolution_clock::now();
            T coupling_time = std::chrono::duration<T, std::milli>(end_time - start_time).count();
            coupling_times_.push_back(coupling_time);

            // Compute energy for monitoring
            T total_energy = computeTotalEnergy(fluid_particles, solid_particles);
            energy_history_.push_back(total_energy);

            current_time_ += time_step_;
        }
    }

    // Performance and analysis methods
    T getAverageCouplingTime() const {
        if (coupling_times_.empty()) return 0.0;
        T sum = std::accumulate(coupling_times_.begin(), coupling_times_.end(), T{0});
        return sum / coupling_times_.size();
    }

    T getEnergyDrift() const {
        if (energy_history_.size() < 2) return 0.0;
        return energy_history_.back() - energy_history_.front();
    }

    const std::vector<T>& getEnergyHistory() const { return energy_history_; }
    const std::vector<T>& getCouplingTimes() const { return coupling_times_; }

private:
    T computeTotalEnergy(const mpm::ParticleAoSoA<T>& fluid_particles,
                        const mpm::ParticleAoSoA<T>& solid_particles) const {
        T total_energy = 0.0;

        // Fluid kinetic energy
        for (size_t i = 0; i < fluid_particles.size(); ++i) {
            T vx, vy, vz;
            fluid_particles.getVelocity(i, vx, vy, vz);
            T mass = fluid_particles.getMass(i);
            total_energy += 0.5 * mass * (vx*vx + vy*vy + vz*vz);
        }

        // Solid kinetic energy
        for (size_t i = 0; i < solid_particles.size(); ++i) {
            T vx, vy, vz;
            solid_particles.getVelocity(i, vx, vy, vz);
            T mass = solid_particles.getMass(i);
            total_energy += 0.5 * mass * (vx*vx + vy*vy + vz*vz);
        }

        return total_energy;
    }
};

} // namespace physgrad::fsi