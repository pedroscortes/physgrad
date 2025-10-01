/**
 * PhysGrad - SPH-MPM Fluid-Structure Coupling
 *
 * Implements coupling between Smoothed Particle Hydrodynamics (SPH) for fluids
 * and Material Point Method (MPM) for solid structures. Enables simulation of
 * complex fluid-structure interactions with high fidelity.
 */

#ifndef PHYSGRAD_SPH_MPM_COUPLING_H
#define PHYSGRAD_SPH_MPM_COUPLING_H

#include "material_point_method.h"
#include "common_types.h"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <functional>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define PHYSGRAD_DEVICE __device__
    #define PHYSGRAD_HOST_DEVICE __host__ __device__
    #define PHYSGRAD_GLOBAL __global__
    #define PHYSGRAD_SHARED __shared__
#else
    #define PHYSGRAD_DEVICE
    #define PHYSGRAD_HOST_DEVICE
    #define PHYSGRAD_GLOBAL
    #define PHYSGRAD_SHARED
#endif

// Include concepts when available
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "concepts/forward_declarations.h"
#endif

namespace physgrad {
namespace coupling {
namespace sph_mpm {

    // =============================================================================
    // SPH PARTICLE SYSTEM
    // =============================================================================

    /**
     * SPH particle data structure optimized for fluid simulation
     */
    template<typename T>
    struct SPHParticle {
        ConceptVector3D<T> position;
        ConceptVector3D<T> velocity;
        ConceptVector3D<T> acceleration;
        ConceptVector3D<T> force;

        T mass;
        T density;
        T pressure;
        T temperature;
        T viscosity;
        T surface_tension_coefficient;

        // Fluid properties
        T rest_density;
        T bulk_modulus;
        T gas_constant;

        // Neighbor information for efficient computation
        std::vector<size_t> neighbors;
        T smoothing_length;

        // Material identification
        int material_id;
        bool is_boundary_particle;

        SPHParticle() :
            position{T{0}, T{0}, T{0}},
            velocity{T{0}, T{0}, T{0}},
            acceleration{T{0}, T{0}, T{0}},
            force{T{0}, T{0}, T{0}},
            mass(T{1}),
            density(T{1000}), // Water density
            pressure(T{0}),
            temperature(T{293.15}), // Room temperature
            viscosity(T{0.001}), // Water viscosity
            surface_tension_coefficient(T{0.0728}), // Water surface tension
            rest_density(T{1000}),
            bulk_modulus(T{2.2e9}), // Water bulk modulus
            gas_constant(T{7}),
            smoothing_length(T{0.1}),
            material_id(0),
            is_boundary_particle(false) {}
    };

    /**
     * SPH kernel functions for smoothing operations
     */
    template<typename T>
    class SPHKernels {
    public:
        // Cubic spline kernel (Monaghan 1992)
        static PHYSGRAD_HOST_DEVICE T cubicSpline(T r, T h) {
            T q = r / h;
            T sigma = T{8} / (T{3.14159265359} * h * h * h); // 3D normalization

            if (q <= T{1}) {
                if (q <= T{0.5}) {
                    return sigma * (T{1} - T{6} * q * q + T{6} * q * q * q);
                } else {
                    T temp = T{2} - q;
                    return sigma * T{2} * temp * temp * temp;
                }
            }
            return T{0};
        }

        // Cubic spline kernel gradient
        static PHYSGRAD_HOST_DEVICE ConceptVector3D<T> cubicSplineGradient(
            const ConceptVector3D<T>& r_vec, T r, T h) {
            if (r < T{1e-6}) return ConceptVector3D<T>{T{0}, T{0}, T{0}};

            T q = r / h;
            T sigma = T{8} / (T{3.14159265359} * h * h * h);
            T factor;

            if (q <= T{1}) {
                if (q <= T{0.5}) {
                    factor = sigma * (-T{12} * q + T{18} * q * q) / (h * h);
                } else {
                    T temp = T{2} - q;
                    factor = sigma * (-T{6} * temp * temp) / (h * h);
                }
            } else {
                factor = T{0};
            }

            return r_vec * (factor / r);
        }

        // Wendland quintic kernel (better stability)
        static PHYSGRAD_HOST_DEVICE T wendlandQuintic(T r, T h) {
            T q = r / h;
            if (q >= T{2}) return T{0};

            T sigma = T{21} / (T{16} * T{3.14159265359} * h * h * h);
            T temp = T{1} - q / T{2};
            return sigma * temp * temp * temp * temp * (T{2} * q + T{1});
        }

        // Wendland quintic gradient
        static PHYSGRAD_HOST_DEVICE ConceptVector3D<T> wendlandQuinticGradient(
            const ConceptVector3D<T>& r_vec, T r, T h) {
            if (r < T{1e-6}) return ConceptVector3D<T>{T{0}, T{0}, T{0}};

            T q = r / h;
            if (q >= T{2}) return ConceptVector3D<T>{T{0}, T{0}, T{0}};

            T sigma = T{21} / (T{16} * T{3.14159265359} * h * h * h);
            T temp = T{1} - q / T{2};
            T factor = sigma * temp * temp * temp * (-T{5} * q) / (T{2} * h * h);

            return r_vec * (factor / r);
        }
    };

    /**
     * SPH system with optimized neighbor search and force computation
     */
    template<typename T>
    class SPHSystem {
    private:
        std::vector<SPHParticle<T>> particles_;
        T smoothing_length_;
        T support_radius_; // Typically 2 * smoothing_length

        // Spatial hashing for efficient neighbor search
        struct SpatialHash {
            std::unordered_map<size_t, std::vector<size_t>> hash_grid;
            T cell_size;
            int3 grid_size;
            ConceptVector3D<T> domain_min, domain_max;

            size_t getHash(int x, int y, int z) const {
                return static_cast<size_t>(x) +
                       static_cast<size_t>(y) * grid_size.x +
                       static_cast<size_t>(z) * grid_size.x * grid_size.y;
            }

            int3 getGridPos(const ConceptVector3D<T>& pos) const {
                return {
                    static_cast<int>((pos[0] - domain_min[0]) / cell_size),
                    static_cast<int>((pos[1] - domain_min[1]) / cell_size),
                    static_cast<int>((pos[2] - domain_min[2]) / cell_size)
                };
            }
        } spatial_hash_;

    public:
        SPHSystem(T smoothing_length = T{0.1})
            : smoothing_length_(smoothing_length),
              support_radius_(T{2} * smoothing_length) {
            spatial_hash_.cell_size = support_radius_;
        }

        void addParticle(const SPHParticle<T>& particle) {
            particles_.push_back(particle);
        }

        void resize(size_t num_particles) {
            particles_.resize(num_particles);
        }

        size_t size() const { return particles_.size(); }

        SPHParticle<T>& getParticle(size_t id) { return particles_[id]; }
        const SPHParticle<T>& getParticle(size_t id) const { return particles_[id]; }

        std::vector<SPHParticle<T>>& getParticles() { return particles_; }
        const std::vector<SPHParticle<T>>& getParticles() const { return particles_; }

        void updateSpatialHash() {
            // Compute domain bounds
            if (particles_.empty()) return;

            spatial_hash_.domain_min = particles_[0].position;
            spatial_hash_.domain_max = particles_[0].position;

            for (const auto& particle : particles_) {
                for (int d = 0; d < 3; ++d) {
                    spatial_hash_.domain_min[d] = std::min(spatial_hash_.domain_min[d],
                                                          particle.position[d] - support_radius_);
                    spatial_hash_.domain_max[d] = std::max(spatial_hash_.domain_max[d],
                                                          particle.position[d] + support_radius_);
                }
            }

            // Compute grid size
            ConceptVector3D<T> domain_size = spatial_hash_.domain_max - spatial_hash_.domain_min;
            spatial_hash_.grid_size = {
                static_cast<int>(domain_size[0] / spatial_hash_.cell_size) + 1,
                static_cast<int>(domain_size[1] / spatial_hash_.cell_size) + 1,
                static_cast<int>(domain_size[2] / spatial_hash_.cell_size) + 1
            };

            // Clear and rebuild hash grid
            spatial_hash_.hash_grid.clear();

            for (size_t i = 0; i < particles_.size(); ++i) {
                int3 grid_pos = spatial_hash_.getGridPos(particles_[i].position);
                size_t hash = spatial_hash_.getHash(grid_pos.x, grid_pos.y, grid_pos.z);
                spatial_hash_.hash_grid[hash].push_back(i);
            }
        }

        void findNeighbors() {
            updateSpatialHash();

            for (size_t i = 0; i < particles_.size(); ++i) {
                particles_[i].neighbors.clear();

                int3 grid_pos = spatial_hash_.getGridPos(particles_[i].position);

                // Search neighboring grid cells
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dz = -1; dz <= 1; ++dz) {
                            int gx = grid_pos.x + dx;
                            int gy = grid_pos.y + dy;
                            int gz = grid_pos.z + dz;

                            if (gx < 0 || gx >= spatial_hash_.grid_size.x ||
                                gy < 0 || gy >= spatial_hash_.grid_size.y ||
                                gz < 0 || gz >= spatial_hash_.grid_size.z) continue;

                            size_t hash = spatial_hash_.getHash(gx, gy, gz);
                            auto it = spatial_hash_.hash_grid.find(hash);
                            if (it == spatial_hash_.hash_grid.end()) continue;

                            for (size_t j : it->second) {
                                if (i == j) continue;

                                ConceptVector3D<T> r_vec = particles_[j].position - particles_[i].position;
                                T r = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);

                                if (r < support_radius_) {
                                    particles_[i].neighbors.push_back(j);
                                }
                            }
                        }
                    }
                }
            }
        }

        void computeDensityPressure() {
            // Compute density using SPH summation
            for (size_t i = 0; i < particles_.size(); ++i) {
                T density = particles_[i].mass * SPHKernels<T>::cubicSpline(T{0}, smoothing_length_);

                for (size_t j : particles_[i].neighbors) {
                    ConceptVector3D<T> r_vec = particles_[j].position - particles_[i].position;
                    T r = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);

                    density += particles_[j].mass * SPHKernels<T>::cubicSpline(r, smoothing_length_);
                }

                particles_[i].density = density;

                // Equation of state (Tait equation)
                T ratio = particles_[i].density / particles_[i].rest_density;
                particles_[i].pressure = particles_[i].gas_constant *
                    (std::pow(ratio, T{7}) - T{1}) * particles_[i].rest_density;
            }
        }

        void computeForces() {
            // Clear forces
            for (auto& particle : particles_) {
                particle.force = ConceptVector3D<T>{T{0}, T{0}, T{0}};
            }

            // Compute SPH forces
            for (size_t i = 0; i < particles_.size(); ++i) {
                ConceptVector3D<T> pressure_force{T{0}, T{0}, T{0}};
                ConceptVector3D<T> viscosity_force{T{0}, T{0}, T{0}};

                for (size_t j : particles_[i].neighbors) {
                    ConceptVector3D<T> r_vec = particles_[j].position - particles_[i].position;
                    T r = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);

                    if (r < T{1e-6}) continue;

                    ConceptVector3D<T> gradient = SPHKernels<T>::cubicSplineGradient(r_vec, r, smoothing_length_);

                    // Pressure force (symmetric formulation)
                    T pressure_term = (particles_[i].pressure + particles_[j].pressure) /
                                     (T{2} * particles_[i].density * particles_[j].density);
                    pressure_force = pressure_force - gradient * (particles_[j].mass * pressure_term);

                    // Viscosity force
                    ConceptVector3D<T> vel_diff = particles_[j].velocity - particles_[i].velocity;
                    T viscosity_term = particles_[i].viscosity * particles_[j].mass *
                                      (r_vec[0]*gradient[0] + r_vec[1]*gradient[1] + r_vec[2]*gradient[2]) /
                                      (particles_[j].density * (r*r + T{0.01}*smoothing_length_*smoothing_length_));
                    viscosity_force = viscosity_force + vel_diff * viscosity_term;
                }

                particles_[i].force = pressure_force + viscosity_force;
            }
        }

        void integrate(T dt) {
            for (auto& particle : particles_) {
                if (particle.is_boundary_particle) continue;

                // Add gravity
                particle.force = particle.force + ConceptVector3D<T>{T{0}, T{-9.81} * particle.mass, T{0}};

                // Compute acceleration
                particle.acceleration = particle.force * (T{1} / particle.mass);

                // Leapfrog integration
                particle.velocity = particle.velocity + particle.acceleration * dt;
                particle.position = particle.position + particle.velocity * dt;
            }
        }
    };

    // =============================================================================
    // SPH-MPM COUPLING INTERFACE
    // =============================================================================

    /**
     * Coupling forces between SPH fluid and MPM solid particles
     */
    template<typename T>
    struct CouplingForce {
        ConceptVector3D<T> fluid_force;  // Force on fluid particle
        ConceptVector3D<T> solid_force;  // Force on solid particle
        T contact_area;                  // Effective contact area
        T penetration_depth;             // Penetration depth for contact
        size_t fluid_particle_id;
        size_t solid_particle_id;
    };

    /**
     * SPH-MPM coupling system configuration
     */
    template<typename T>
    struct CouplingConfig {
        T coupling_stiffness = T{1e5};        // Contact stiffness
        T coupling_damping = T{1e3};          // Contact damping
        T friction_coefficient = T{0.3};      // Friction coefficient
        T coupling_radius = T{0.15};          // Coupling interaction radius
        bool enable_two_way_coupling = true;  // Enable forces on both phases
        bool enable_heat_transfer = false;    // Enable thermal coupling
        bool enable_mass_transfer = false;    // Enable mass transfer (dissolution, etc.)

        // Surface tension parameters
        bool enable_surface_tension = true;
        T surface_tension_strength = T{0.0728};

        // Adhesion parameters
        bool enable_adhesion = true;
        T adhesion_strength = T{0.01};
    };

    /**
     * Main SPH-MPM coupling system
     */
    template<typename T>
    class SPHMPMCouplingSystem {
    private:
        std::unique_ptr<SPHSystem<T>> sph_system_;
        std::unique_ptr<mpm::ParticleAoSoA<T>> mpm_particles_;
        std::unique_ptr<mpm::MPMGrid<T>> mpm_grid_;

        CouplingConfig<T> config_;
        std::vector<CouplingForce<T>> coupling_forces_;

        // Performance optimization data structures
        std::vector<std::pair<size_t, size_t>> coupling_pairs_;  // (sph_id, mpm_id)

    public:
        SPHMPMCouplingSystem(const CouplingConfig<T>& config = CouplingConfig<T>{})
            : config_(config) {
            sph_system_ = std::make_unique<SPHSystem<T>>();
        }

        void initializeSPHSystem(size_t num_fluid_particles, T smoothing_length = T{0.1}) {
            sph_system_ = std::make_unique<SPHSystem<T>>(smoothing_length);
            sph_system_->resize(num_fluid_particles);
        }

        void initializeMPMSystem(const int3& grid_dims,
                               const ConceptVector3D<T>& cell_size,
                               const ConceptVector3D<T>& origin,
                               size_t num_solid_particles) {
            mpm_grid_ = std::make_unique<mpm::MPMGrid<T>>(grid_dims, cell_size, origin);
            mpm_particles_ = std::make_unique<mpm::ParticleAoSoA<T>>();
            mpm_particles_->resize(num_solid_particles);
        }

        SPHSystem<T>& getSPHSystem() { return *sph_system_; }
        const SPHSystem<T>& getSPHSystem() const { return *sph_system_; }

        mpm::ParticleAoSoA<T>& getMPMParticles() { return *mpm_particles_; }
        const mpm::ParticleAoSoA<T>& getMPMParticles() const { return *mpm_particles_; }

        mpm::MPMGrid<T>& getMPMGrid() { return *mpm_grid_; }
        const mpm::MPMGrid<T>& getMPMGrid() const { return *mpm_grid_; }

        void findCouplingPairs() {
            coupling_pairs_.clear();
            coupling_forces_.clear();

            if (!sph_system_ || !mpm_particles_) return;

            auto& sph_particles = sph_system_->getParticles();

            for (size_t sph_id = 0; sph_id < sph_particles.size(); ++sph_id) {
                const auto& sph_pos = sph_particles[sph_id].position;

                for (size_t mpm_id = 0; mpm_id < mpm_particles_->size(); ++mpm_id) {
                    auto mpm_pos = mpm_particles_->getPosition(mpm_id);

                    ConceptVector3D<T> r_vec = mpm_pos - sph_pos;
                    T distance = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);

                    if (distance < config_.coupling_radius) {
                        coupling_pairs_.emplace_back(sph_id, mpm_id);

                        // Create coupling force structure
                        CouplingForce<T> force;
                        force.fluid_particle_id = sph_id;
                        force.solid_particle_id = mpm_id;
                        force.penetration_depth = config_.coupling_radius - distance;
                        force.contact_area = T{3.14159265359} * config_.coupling_radius * config_.coupling_radius;

                        coupling_forces_.push_back(force);
                    }
                }
            }
        }

        void computeCouplingForces() {
            if (!sph_system_ || !mpm_particles_) return;

            auto& sph_particles = sph_system_->getParticles();

            for (auto& coupling_force : coupling_forces_) {
                size_t sph_id = coupling_force.fluid_particle_id;
                size_t mpm_id = coupling_force.solid_particle_id;

                auto& sph_particle = sph_particles[sph_id];
                auto mpm_pos = mpm_particles_->getPosition(mpm_id);
                auto mpm_vel = mpm_particles_->getVelocity(mpm_id);

                ConceptVector3D<T> r_vec = mpm_pos - sph_particle.position;
                T distance = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);

                if (distance < T{1e-6}) continue;

                ConceptVector3D<T> normal = r_vec * (T{1} / distance);
                ConceptVector3D<T> vel_diff = mpm_vel - sph_particle.velocity;

                // Normal force (elastic contact)
                T normal_force_magnitude = config_.coupling_stiffness * coupling_force.penetration_depth;

                // Damping force
                T normal_vel = vel_diff[0]*normal[0] + vel_diff[1]*normal[1] + vel_diff[2]*normal[2];
                T damping_force_magnitude = config_.coupling_damping * normal_vel;

                ConceptVector3D<T> total_normal_force = normal * (normal_force_magnitude + damping_force_magnitude);

                // Friction force
                ConceptVector3D<T> tangential_vel = vel_diff - normal * normal_vel;
                T tangential_speed = std::sqrt(tangential_vel[0]*tangential_vel[0] +
                                             tangential_vel[1]*tangential_vel[1] +
                                             tangential_vel[2]*tangential_vel[2]);

                ConceptVector3D<T> friction_force{T{0}, T{0}, T{0}};
                if (tangential_speed > T{1e-6}) {
                    ConceptVector3D<T> tangential_dir = tangential_vel * (T{1} / tangential_speed);
                    T friction_magnitude = config_.friction_coefficient * normal_force_magnitude;
                    friction_force = tangential_dir * (-friction_magnitude);
                }

                // Total coupling forces
                coupling_force.solid_force = total_normal_force + friction_force;
                coupling_force.fluid_force = (total_normal_force + friction_force) * T{-1};

                // Apply forces if two-way coupling is enabled
                if (config_.enable_two_way_coupling) {
                    sph_particle.force = sph_particle.force + coupling_force.fluid_force;

                    // For MPM, forces are applied during grid update
                    // This would require integration with MPM solver
                }
            }
        }

        void simulationStep(T dt) {
            // Update SPH system
            sph_system_->findNeighbors();
            sph_system_->computeDensityPressure();
            sph_system_->computeForces();

            // Find coupling interactions
            findCouplingPairs();
            computeCouplingForces();

            // Integrate SPH particles
            sph_system_->integrate(dt);

            // MPM simulation step would be called here
            // This requires integration with the MPM solver
        }

        // Analysis and diagnostic functions
        size_t getNumCouplingPairs() const { return coupling_pairs_.size(); }

        T getTotalCouplingEnergy() const {
            T total_energy = T{0};
            for (const auto& force : coupling_forces_) {
                T force_magnitude = std::sqrt(
                    force.solid_force[0]*force.solid_force[0] +
                    force.solid_force[1]*force.solid_force[1] +
                    force.solid_force[2]*force.solid_force[2]
                );
                total_energy += T{0.5} * config_.coupling_stiffness *
                              force.penetration_depth * force.penetration_depth;
            }
            return total_energy;
        }

        std::vector<CouplingForce<T>>& getCouplingForces() { return coupling_forces_; }
        const std::vector<CouplingForce<T>>& getCouplingForces() const { return coupling_forces_; }
    };

    // =============================================================================
    // ADVANCED COUPLING FEATURES
    // =============================================================================

    /**
     * Surface reconstruction for accurate fluid-solid interface detection
     */
    template<typename T>
    class SurfaceReconstruction {
    private:
        struct MarchingCubesCell {
            ConceptVector3D<T> vertices[8];
            T values[8];
        };

    public:
        static std::vector<ConceptVector3D<T>> extractFluidSurface(
            const SPHSystem<T>& sph_system, T iso_value = T{0.5}) {

            // Simplified surface extraction using density field
            std::vector<ConceptVector3D<T>> surface_points;

            const auto& particles = sph_system.getParticles();
            for (const auto& particle : particles) {
                // Check if particle is near surface (low density)
                if (particle.density < iso_value * particle.rest_density) {
                    surface_points.push_back(particle.position);
                }
            }

            return surface_points;
        }
    };

    /**
     * Adaptive time stepping for coupled SPH-MPM system
     */
    template<typename T>
    class AdaptiveTimeStep {
    private:
        T min_dt_;
        T max_dt_;
        T cfl_factor_;
        T force_factor_;

    public:
        AdaptiveTimeStep(T min_dt = T{1e-6}, T max_dt = T{1e-3},
                        T cfl_factor = T{0.3}, T force_factor = T{0.1})
            : min_dt_(min_dt), max_dt_(max_dt), cfl_factor_(cfl_factor), force_factor_(force_factor) {}

        T computeOptimalTimeStep(const SPHMPMCouplingSystem<T>& system) {
            const auto& sph_particles = system.getSPHSystem().getParticles();

            T min_time_step = max_dt_;

            // CFL condition for SPH
            for (const auto& particle : sph_particles) {
                T speed = std::sqrt(particle.velocity[0]*particle.velocity[0] +
                                  particle.velocity[1]*particle.velocity[1] +
                                  particle.velocity[2]*particle.velocity[2]);
                if (speed > T{1e-6}) {
                    T cfl_dt = cfl_factor_ * particle.smoothing_length / speed;
                    min_time_step = std::min(min_time_step, cfl_dt);
                }

                // Force-based time step
                T accel_magnitude = std::sqrt(particle.acceleration[0]*particle.acceleration[0] +
                                            particle.acceleration[1]*particle.acceleration[1] +
                                            particle.acceleration[2]*particle.acceleration[2]);
                if (accel_magnitude > T{1e-6}) {
                    T force_dt = force_factor_ * std::sqrt(particle.smoothing_length / accel_magnitude);
                    min_time_step = std::min(min_time_step, force_dt);
                }
            }

            return std::max(min_dt_, std::min(max_dt_, min_time_step));
        }
    };

} // namespace sph_mpm
} // namespace coupling
} // namespace physgrad

#endif // PHYSGRAD_SPH_MPM_COUPLING_H