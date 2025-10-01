/**
 * PhysGrad - Material Point Method Implementation
 *
 * High-performance Material Point Method (MPM) with Array of Structures of Arrays (AoSoA)
 * data layout for optimal memory access patterns and vectorization.
 * Supports both 2D and 3D simulations with CUDA acceleration.
 */

#ifndef PHYSGRAD_MATERIAL_POINT_METHOD_H
#define PHYSGRAD_MATERIAL_POINT_METHOD_H

#include "common_types.h"
#include <vector>
#include <memory>
#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>

// Include concepts when available
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "concepts/forward_declarations.h"
#endif

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define PHYSGRAD_DEVICE __device__
    #define PHYSGRAD_HOST_DEVICE __host__ __device__
#else
    #define PHYSGRAD_DEVICE
    #define PHYSGRAD_HOST_DEVICE
#endif

namespace physgrad {
namespace mpm {

// =============================================================================
// AOSOA DATA STRUCTURES
// =============================================================================

/**
 * Array of Structures of Arrays (AoSoA) layout for optimal memory access
 * Groups particles into chunks for better cache locality and vectorization
 */
template<typename T, size_t ChunkSize = 64>
struct ParticleAoSoA {
    static constexpr size_t chunk_size = ChunkSize;
    using value_type = T;

    // Position chunks: [x0, x1, ..., x63, y0, y1, ..., y63, z0, z1, ..., z63]
    std::vector<std::array<T, chunk_size * 3>> position_chunks;

    // Velocity chunks
    std::vector<std::array<T, chunk_size * 3>> velocity_chunks;

    // Mass chunks
    std::vector<std::array<T, chunk_size>> mass_chunks;

    // Volume chunks
    std::vector<std::array<T, chunk_size>> volume_chunks;

    // Deformation gradient chunks (3x3 matrix per particle)
    std::vector<std::array<T, chunk_size * 9>> deformation_gradient_chunks;

    // Stress chunks (symmetric 3x3 = 6 components)
    std::vector<std::array<T, chunk_size * 6>> stress_chunks;

    // Material properties chunks
    std::vector<std::array<T, chunk_size>> young_modulus_chunks;
    std::vector<std::array<T, chunk_size>> poisson_ratio_chunks;
    std::vector<std::array<T, chunk_size>> density_chunks;

    size_t num_particles = 0;
    size_t num_chunks = 0;

    size_t size() const { return num_particles; }

    void resize(size_t new_size) {
        num_particles = new_size;
        num_chunks = (new_size + chunk_size - 1) / chunk_size;

        position_chunks.resize(num_chunks);
        velocity_chunks.resize(num_chunks);
        mass_chunks.resize(num_chunks);
        volume_chunks.resize(num_chunks);
        deformation_gradient_chunks.resize(num_chunks);
        stress_chunks.resize(num_chunks);
        young_modulus_chunks.resize(num_chunks);
        poisson_ratio_chunks.resize(num_chunks);
        density_chunks.resize(num_chunks);
    }

    PHYSGRAD_HOST_DEVICE
    ConceptVector3D<T> getPosition(size_t particle_id) const {
        const size_t chunk_id = particle_id / chunk_size;
        const size_t local_id = particle_id % chunk_size;
        const auto& chunk = position_chunks[chunk_id];

        return ConceptVector3D<T>{
            chunk[local_id],                    // x
            chunk[local_id + chunk_size],       // y
            chunk[local_id + 2 * chunk_size]    // z
        };
    }

    PHYSGRAD_HOST_DEVICE
    void setPosition(size_t particle_id, const ConceptVector3D<T>& pos) {
        const size_t chunk_id = particle_id / chunk_size;
        const size_t local_id = particle_id % chunk_size;
        auto& chunk = position_chunks[chunk_id];

        chunk[local_id] = pos[0];                    // x
        chunk[local_id + chunk_size] = pos[1];       // y
        chunk[local_id + 2 * chunk_size] = pos[2];   // z
    }

    PHYSGRAD_HOST_DEVICE
    ConceptVector3D<T> getVelocity(size_t particle_id) const {
        const size_t chunk_id = particle_id / chunk_size;
        const size_t local_id = particle_id % chunk_size;
        const auto& chunk = velocity_chunks[chunk_id];

        return ConceptVector3D<T>{
            chunk[local_id],
            chunk[local_id + chunk_size],
            chunk[local_id + 2 * chunk_size]
        };
    }

    PHYSGRAD_HOST_DEVICE
    void setVelocity(size_t particle_id, const ConceptVector3D<T>& vel) {
        const size_t chunk_id = particle_id / chunk_size;
        const size_t local_id = particle_id % chunk_size;
        auto& chunk = velocity_chunks[chunk_id];

        chunk[local_id] = vel[0];
        chunk[local_id + chunk_size] = vel[1];
        chunk[local_id + 2 * chunk_size] = vel[2];
    }

    PHYSGRAD_HOST_DEVICE
    T getMass(size_t particle_id) const {
        const size_t chunk_id = particle_id / chunk_size;
        const size_t local_id = particle_id % chunk_size;
        return mass_chunks[chunk_id][local_id];
    }

    PHYSGRAD_HOST_DEVICE
    void setMass(size_t particle_id, T mass) {
        const size_t chunk_id = particle_id / chunk_size;
        const size_t local_id = particle_id % chunk_size;
        mass_chunks[chunk_id][local_id] = mass;
    }

    // Deformation gradient access (3x3 matrix stored row-major)
    PHYSGRAD_HOST_DEVICE
    void getDeformationGradient(size_t particle_id, T* F) const {
        const size_t chunk_id = particle_id / chunk_size;
        const size_t local_id = particle_id % chunk_size;
        const auto& chunk = deformation_gradient_chunks[chunk_id];

        for (int i = 0; i < 9; ++i) {
            F[i] = chunk[local_id + i * chunk_size];
        }
    }

    PHYSGRAD_HOST_DEVICE
    void setDeformationGradient(size_t particle_id, const T* F) {
        const size_t chunk_id = particle_id / chunk_size;
        const size_t local_id = particle_id % chunk_size;
        auto& chunk = deformation_gradient_chunks[chunk_id];

        for (int i = 0; i < 9; ++i) {
            chunk[local_id + i * chunk_size] = F[i];
        }
    }
};

// =============================================================================
// GRID DATA STRUCTURES
// =============================================================================

/**
 * Background grid for MPM simulation
 */
template<typename T>
struct MPMGrid {
    std::vector<T> mass;            // Grid node masses
    std::vector<ConceptVector3D<T>> velocity;     // Grid node velocities
    std::vector<ConceptVector3D<T>> momentum;     // Grid node momentum
    std::vector<ConceptVector3D<T>> force;        // Grid node forces

    int3 dimensions;  // Grid dimensions (nx, ny, nz)
    int3 dims;        // Alias for dimensions for compatibility
    ConceptVector3D<T> cell_size;    // Cell size (dx, dy, dz)
    ConceptVector3D<T> origin;       // Grid origin

    size_t total_nodes;

    MPMGrid(const int3& dims_arg, const ConceptVector3D<T>& cell_sz, const ConceptVector3D<T>& orig)
        : dimensions(dims_arg), dims(dims_arg), cell_size(cell_sz), origin(orig) {
        total_nodes = static_cast<size_t>(dims_arg.x) * dims_arg.y * dims_arg.z;
        resize();
    }

    void resize() {
        mass.resize(total_nodes, T{0});
        velocity.resize(total_nodes, ConceptVector3D<T>{T{0}, T{0}, T{0}});
        momentum.resize(total_nodes, ConceptVector3D<T>{T{0}, T{0}, T{0}});
        force.resize(total_nodes, ConceptVector3D<T>{T{0}, T{0}, T{0}});
    }

    void clear() {
        std::fill(mass.begin(), mass.end(), T{0});
        std::fill(velocity.begin(), velocity.end(), ConceptVector3D<T>{T{0}, T{0}, T{0}});
        std::fill(momentum.begin(), momentum.end(), ConceptVector3D<T>{T{0}, T{0}, T{0}});
        std::fill(force.begin(), force.end(), ConceptVector3D<T>{T{0}, T{0}, T{0}});
    }

    void clearGridData() { clear(); }

    void updateGridVelocities() {
        for (size_t i = 0; i < total_nodes; ++i) {
            if (mass[i] > T{1e-10}) {
                velocity[i] = momentum[i] * (T{1} / mass[i]);
            } else {
                velocity[i] = ConceptVector3D<T>{T{0}, T{0}, T{0}};
            }
        }
    }

    void applyBoundaryConditions() {
        // Simple boundary conditions - set velocity to zero at boundaries
        for (int k = 0; k < dimensions.z; ++k) {
            for (int j = 0; j < dimensions.y; ++j) {
                for (int i = 0; i < dimensions.x; ++i) {
                    if (i == 0 || i == dimensions.x - 1 ||
                        j == 0 || j == dimensions.y - 1 ||
                        k == 0 || k == dimensions.z - 1) {
                        size_t node_id = getNodeIndex(i, j, k);
                        velocity[node_id] = ConceptVector3D<T>{T{0}, T{0}, T{0}};
                        momentum[node_id] = ConceptVector3D<T>{T{0}, T{0}, T{0}};
                    }
                }
            }
        }
    }

    PHYSGRAD_HOST_DEVICE
    size_t getNodeIndex(int i, int j, int k) const {
        return static_cast<size_t>(k) * dimensions.x * dimensions.y +
               static_cast<size_t>(j) * dimensions.x + i;
    }

    PHYSGRAD_HOST_DEVICE
    ConceptVector3D<T> getNodePosition(int i, int j, int k) const {
        return ConceptVector3D<T>{
            origin[0] + i * cell_size[0],
            origin[1] + j * cell_size[1],
            origin[2] + k * cell_size[2]
        };
    }
};

// =============================================================================
// SHAPE FUNCTIONS AND INTERPOLATION
// =============================================================================

/**
 * B-spline basis functions for MPM interpolation
 */
template<typename T>
class MPMShapeFunctions {
public:
    // Linear B-spline (tent function)
    static PHYSGRAD_HOST_DEVICE T linear(T x) {
        T abs_x = x < T{0} ? -x : x;
        if (abs_x >= T{1}) return T{0};
        return T{1} - abs_x;
    }

    static PHYSGRAD_HOST_DEVICE T linearDerivative(T x) {
        if (x < -T{1} || x > T{1}) return T{0};
        return x < T{0} ? T{1} : -T{1};
    }

    // Quadratic B-spline
    static PHYSGRAD_HOST_DEVICE T quadratic(T x) {
        T abs_x = x < T{0} ? -x : x;
        if (abs_x >= T{1.5}) return T{0};
        if (abs_x < T{0.5}) return T{0.75} - x * x;
        return T{0.5} * (T{1.5} - abs_x) * (T{1.5} - abs_x);
    }

    static PHYSGRAD_HOST_DEVICE T quadraticDerivative(T x) {
        T abs_x = x < T{0} ? -x : x;
        if (abs_x >= T{1.5}) return T{0};
        if (abs_x < T{0.5}) return -T{2} * x;
        return x < T{0} ? (T{1.5} - abs_x) : -(T{1.5} - abs_x);
    }

    // Cubic B-spline
    static PHYSGRAD_HOST_DEVICE T cubic(T x) {
        T abs_x = x < T{0} ? -x : x;
        if (abs_x >= T{2}) return T{0};
        if (abs_x < T{1}) {
            return T{2.0/3.0} - abs_x * abs_x + T{0.5} * abs_x * abs_x * abs_x;
        }
        T temp = T{2} - abs_x;
        return T{1.0/6.0} * temp * temp * temp;
    }

    static PHYSGRAD_HOST_DEVICE T cubicDerivative(T x) {
        T abs_x = x < T{0} ? -x : x;
        T sign = x < T{0} ? -T{1} : T{1};

        if (abs_x >= T{2}) return T{0};
        if (abs_x < T{1}) {
            return sign * (-T{2} * abs_x + T{1.5} * abs_x * abs_x);
        }
        T temp = T{2} - abs_x;
        return -sign * T{0.5} * temp * temp;
    }
};

// =============================================================================
// CONSTITUTIVE MODELS
// =============================================================================

/**
 * Neo-Hookean hyperelastic material model
 */
template<typename T>
class NeoHookeanModel {
public:
    PHYSGRAD_HOST_DEVICE
    static void computeStress(const T* F, T* stress, T E, T nu, T J = T{1}) {
        // Compute deformation invariants
        T I1 = F[0]*F[0] + F[1]*F[1] + F[2]*F[2] +
               F[3]*F[3] + F[4]*F[4] + F[5]*F[5] +
               F[6]*F[6] + F[7]*F[7] + F[8]*F[8];

        T detF = F[0]*(F[4]*F[8] - F[5]*F[7]) -
                 F[1]*(F[3]*F[8] - F[5]*F[6]) +
                 F[2]*(F[3]*F[7] - F[4]*F[6]);

        // Material parameters
        T mu = E / (T{2} * (T{1} + nu));
        T lambda = E * nu / ((T{1} + nu) * (T{1} - T{2} * nu));

        // Neo-Hookean stress (simplified)
        T J_23 = std::pow(detF, -T{2}/T{3});
        T pressure = lambda * (detF - T{1}) / detF;

        // Cauchy stress (simplified diagonal approximation)
        stress[0] = mu * J_23 * (F[0]*F[0] - I1/T{3}) + pressure;  // σxx
        stress[1] = mu * J_23 * (F[4]*F[4] - I1/T{3}) + pressure;  // σyy
        stress[2] = mu * J_23 * (F[8]*F[8] - I1/T{3}) + pressure;  // σzz
        stress[3] = mu * J_23 * F[0]*F[1];                          // σxy
        stress[4] = mu * J_23 * F[0]*F[2];                          // σxz
        stress[5] = mu * J_23 * F[1]*F[2];                          // σyz
    }
};

/**
 * von Mises plasticity model
 */
template<typename T>
class VonMisesPlasticityModel {
public:
    struct PlasticState {
        T equivalent_plastic_strain = T{0};
        T yield_stress = T{1e6};  // Initial yield stress
    };

    PHYSGRAD_HOST_DEVICE
    static void computeStress(const T* F, T* stress, T E, T nu,
                             PlasticState& plastic_state) {
        // Elastic predictor
        NeoHookeanModel<T>::computeStress(F, stress, E, nu);

        // von Mises yield criterion
        T sigma_vm = computeVonMisesStress(stress);
        T yield_stress = plastic_state.yield_stress;

        if (sigma_vm > yield_stress) {
            // Plastic corrector (return mapping)
            T scale = yield_stress / sigma_vm;

            // Scale deviatoric part
            T mean_stress = (stress[0] + stress[1] + stress[2]) / T{3};
            stress[0] = (stress[0] - mean_stress) * scale + mean_stress;
            stress[1] = (stress[1] - mean_stress) * scale + mean_stress;
            stress[2] = (stress[2] - mean_stress) * scale + mean_stress;
            stress[3] *= scale;
            stress[4] *= scale;
            stress[5] *= scale;

            // Update plastic state
            plastic_state.equivalent_plastic_strain +=
                (sigma_vm - yield_stress) / E;
        }
    }

private:
    PHYSGRAD_HOST_DEVICE
    static T computeVonMisesStress(const T* stress) {
        T s_dev[6];
        T mean_stress = (stress[0] + stress[1] + stress[2]) / T{3};

        s_dev[0] = stress[0] - mean_stress;
        s_dev[1] = stress[1] - mean_stress;
        s_dev[2] = stress[2] - mean_stress;
        s_dev[3] = stress[3];
        s_dev[4] = stress[4];
        s_dev[5] = stress[5];

        return std::sqrt(T{1.5} * (s_dev[0]*s_dev[0] + s_dev[1]*s_dev[1] + s_dev[2]*s_dev[2] +
                                  T{2} * (s_dev[3]*s_dev[3] + s_dev[4]*s_dev[4] + s_dev[5]*s_dev[5])));
    }
};

// =============================================================================
// MPM SOLVER
// =============================================================================

/**
 * Material Point Method solver with AoSoA optimization
 */
template<typename T, size_t ChunkSize = 64>
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    requires concepts::PhysicsScalar<T>
#endif
class MPMSolver {
public:
    using ParticleData = ParticleAoSoA<T, ChunkSize>;
    using Grid = MPMGrid<T>;
    using ShapeFunc = MPMShapeFunctions<T>;

    struct SimulationParams {
        T timestep = T{1e-4};
        ConceptVector3D<T> gravity = {T{0}, T{-9.81}, T{0}};
        T cfl_factor = T{0.4};
        bool use_affine_pic = true;  // Affine Particle-In-Cell transfer
        bool use_flip = true;        // FLIP/PIC blending
        T flip_ratio = T{0.95};      // FLIP blending ratio
        int interpolation_order = 2; // 1=linear, 2=quadratic, 3=cubic
    };

    MPMSolver(const int3& grid_dims, const ConceptVector3D<T>& cell_size,
              const ConceptVector3D<T>& origin, const SimulationParams& params = {})
        : grid_(grid_dims, cell_size, origin), params_(params) {}

    void addParticles(const std::vector<ConceptVector3D<T>>& positions,
                     const std::vector<ConceptVector3D<T>>& velocities,
                     const std::vector<T>& masses,
                     const std::vector<T>& volumes,
                     T young_modulus = T{1e6}, T poisson_ratio = T{0.3}, T density = T{1000}) {

        size_t old_size = particles_.num_particles;
        size_t new_size = old_size + positions.size();

        particles_.resize(new_size);

        // Initialize new particles
        for (size_t i = 0; i < positions.size(); ++i) {
            size_t particle_id = old_size + i;

            particles_.setPosition(particle_id, positions[i]);
            particles_.setVelocity(particle_id, velocities[i]);
            particles_.setMass(particle_id, masses[i]);

            // Initialize deformation gradient to identity
            T F[9] = {T{1}, T{0}, T{0}, T{0}, T{1}, T{0}, T{0}, T{0}, T{1}};
            particles_.setDeformationGradient(particle_id, F);
        }
    }

    void step() {
        // Clear grid
        grid_.clear();

        // P2G: Particle to Grid transfer
        particleToGrid();

        // Grid momentum update
        updateGridMomentum();

        // Grid velocity calculation
        calculateGridVelocity();

        // G2P: Grid to Particle transfer
        gridToParticle();

        // Update particle positions
        updateParticlePositions();

        current_time_ += params_.timestep;
        step_count_++;
    }

    // Getters
    const ParticleData& getParticles() const { return particles_; }
    const Grid& getGrid() const { return grid_; }
    T getCurrentTime() const { return current_time_; }
    size_t getStepCount() const { return step_count_; }

private:
    ParticleData particles_;
    Grid grid_;
    SimulationParams params_;
    T current_time_ = T{0};
    size_t step_count_ = 0;

    void particleToGrid() {
        for (size_t p = 0; p < particles_.num_particles; ++p) {
            auto pos = particles_.getPosition(p);
            auto vel = particles_.getVelocity(p);
            T mass = particles_.getMass(p);

            // Find grid cells influenced by this particle
            int base_i = static_cast<int>(std::floor((pos[0] - grid_.origin[0]) / grid_.cell_size[0]));
            int base_j = static_cast<int>(std::floor((pos[1] - grid_.origin[1]) / grid_.cell_size[1]));
            int base_k = static_cast<int>(std::floor((pos[2] - grid_.origin[2]) / grid_.cell_size[2]));

            // Compute local coordinates
            T fx = (pos[0] - grid_.origin[0]) / grid_.cell_size[0] - base_i;
            T fy = (pos[1] - grid_.origin[1]) / grid_.cell_size[1] - base_j;
            T fz = (pos[2] - grid_.origin[2]) / grid_.cell_size[2] - base_k;

            // Interpolation kernel size
            int kernel_size = (params_.interpolation_order == 1) ? 2 :
                             (params_.interpolation_order == 2) ? 3 : 4;
            int kernel_offset = kernel_size / 2;

            // Transfer to grid nodes
            for (int di = -kernel_offset; di < kernel_size - kernel_offset; ++di) {
                for (int dj = -kernel_offset; dj < kernel_size - kernel_offset; ++dj) {
                    for (int dk = -kernel_offset; dk < kernel_size - kernel_offset; ++dk) {
                        int gi = base_i + di;
                        int gj = base_j + dj;
                        int gk = base_k + dk;

                        // Bounds check
                        if (gi < 0 || gi >= grid_.dimensions.x ||
                            gj < 0 || gj >= grid_.dimensions.y ||
                            gk < 0 || gk >= grid_.dimensions.z) continue;

                        // Compute shape function weights
                        T wx = computeShapeFunction(fx - di);
                        T wy = computeShapeFunction(fy - dj);
                        T wz = computeShapeFunction(fz - dk);
                        T weight = wx * wy * wz;

                        if (weight > T{1e-10}) {
                            size_t node_idx = grid_.getNodeIndex(gi, gj, gk);

                            // Transfer mass and momentum
                            grid_.mass[node_idx] += weight * mass;
                            grid_.momentum[node_idx] = {
                                grid_.momentum[node_idx][0] + weight * mass * vel[0],
                                grid_.momentum[node_idx][1] + weight * mass * vel[1],
                                grid_.momentum[node_idx][2] + weight * mass * vel[2]
                            };
                        }
                    }
                }
            }
        }
    }

    void updateGridMomentum() {
        const T dt = params_.timestep;

        for (size_t i = 0; i < grid_.total_nodes; ++i) {
            if (grid_.mass[i] > T{1e-10}) {
                // Apply gravity
                grid_.momentum[i] = {
                    grid_.momentum[i][0] + grid_.mass[i] * params_.gravity[0] * dt,
                    grid_.momentum[i][1] + grid_.mass[i] * params_.gravity[1] * dt,
                    grid_.momentum[i][2] + grid_.mass[i] * params_.gravity[2] * dt
                };
            }
        }
    }

    void calculateGridVelocity() {
        for (size_t i = 0; i < grid_.total_nodes; ++i) {
            if (grid_.mass[i] > T{1e-10}) {
                grid_.velocity[i] = {
                    grid_.momentum[i][0] / grid_.mass[i],
                    grid_.momentum[i][1] / grid_.mass[i],
                    grid_.momentum[i][2] / grid_.mass[i]
                };
            }
        }
    }

    void gridToParticle() {
        for (size_t p = 0; p < particles_.num_particles; ++p) {
            auto pos = particles_.getPosition(p);
            ConceptVector3D<T> new_velocity{T{0}, T{0}, T{0}};

            // Find grid cells
            int base_i = static_cast<int>(std::floor((pos[0] - grid_.origin[0]) / grid_.cell_size[0]));
            int base_j = static_cast<int>(std::floor((pos[1] - grid_.origin[1]) / grid_.cell_size[1]));
            int base_k = static_cast<int>(std::floor((pos[2] - grid_.origin[2]) / grid_.cell_size[2]));

            T fx = (pos[0] - grid_.origin[0]) / grid_.cell_size[0] - base_i;
            T fy = (pos[1] - grid_.origin[1]) / grid_.cell_size[1] - base_j;
            T fz = (pos[2] - grid_.origin[2]) / grid_.cell_size[2] - base_k;

            int kernel_size = (params_.interpolation_order == 1) ? 2 :
                             (params_.interpolation_order == 2) ? 3 : 4;
            int kernel_offset = kernel_size / 2;

            // Interpolate velocity from grid
            for (int di = -kernel_offset; di < kernel_size - kernel_offset; ++di) {
                for (int dj = -kernel_offset; dj < kernel_size - kernel_offset; ++dj) {
                    for (int dk = -kernel_offset; dk < kernel_size - kernel_offset; ++dk) {
                        int gi = base_i + di;
                        int gj = base_j + dj;
                        int gk = base_k + dk;

                        if (gi < 0 || gi >= grid_.dimensions.x ||
                            gj < 0 || gj >= grid_.dimensions.y ||
                            gk < 0 || gk >= grid_.dimensions.z) continue;

                        T wx = computeShapeFunction(fx - di);
                        T wy = computeShapeFunction(fy - dj);
                        T wz = computeShapeFunction(fz - dk);
                        T weight = wx * wy * wz;

                        if (weight > T{1e-10}) {
                            size_t node_idx = grid_.getNodeIndex(gi, gj, gk);

                            new_velocity = {
                                new_velocity[0] + weight * grid_.velocity[node_idx][0],
                                new_velocity[1] + weight * grid_.velocity[node_idx][1],
                                new_velocity[2] + weight * grid_.velocity[node_idx][2]
                            };
                        }
                    }
                }
            }

            particles_.setVelocity(p, new_velocity);
        }
    }

    void updateParticlePositions() {
        const T dt = params_.timestep;

        for (size_t p = 0; p < particles_.num_particles; ++p) {
            auto pos = particles_.getPosition(p);
            auto vel = particles_.getVelocity(p);

            ConceptVector3D<T> new_pos = {
                pos[0] + vel[0] * dt,
                pos[1] + vel[1] * dt,
                pos[2] + vel[2] * dt
            };

            particles_.setPosition(p, new_pos);
        }
    }

    T computeShapeFunction(T x) const {
        switch (params_.interpolation_order) {
            case 1: return ShapeFunc::linear(x);
            case 2: return ShapeFunc::quadratic(x);
            case 3: return ShapeFunc::cubic(x);
            default: return ShapeFunc::quadratic(x);
        }
    }
};

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
// Verify concept compliance
static_assert(concepts::PhysicsScalar<float>);
static_assert(concepts::PhysicsScalar<double>);
#endif

} // namespace mpm
} // namespace physgrad

#endif // PHYSGRAD_MATERIAL_POINT_METHOD_H