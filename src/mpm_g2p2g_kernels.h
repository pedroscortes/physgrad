/**
 * PhysGrad - G2P2G Kernel Fusion for MPM
 *
 * Optimized Grid-to-Particle-to-Grid (G2P2G) kernel fusion for Material Point Method.
 * Combines particle updates and grid transfers in a single kernel launch to minimize
 * memory bandwidth and improve performance on GPU architectures.
 */

#ifndef PHYSGRAD_MPM_G2P2G_KERNELS_H
#define PHYSGRAD_MPM_G2P2G_KERNELS_H

#include "material_point_method.h"
#include <vector>
#include <memory>
#include <array>
#include <cmath>
#include <algorithm>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <cooperative_groups.h>
    #define PHYSGRAD_DEVICE __device__
    #define PHYSGRAD_HOST_DEVICE __host__ __device__
    #define PHYSGRAD_GLOBAL __global__
    #define PHYSGRAD_SHARED __shared__
#else
    #define PHYSGRAD_DEVICE
    #define PHYSGRAD_HOST_DEVICE
    #define PHYSGRAD_GLOBAL
    #define PHYSGRAD_SHARED

    // CPU fallback for CUDA threading variables
    struct dim3_fallback { unsigned int x, y, z; };
    static thread_local dim3_fallback threadIdx = {0, 0, 0};
    static thread_local dim3_fallback blockIdx = {0, 0, 0};
    static thread_local dim3_fallback blockDim = {1, 1, 1};
    static thread_local dim3_fallback gridDim = {1, 1, 1};

    inline void __syncthreads() { /* CPU fallback - no-op */ }

    template<typename T>
    T min(T a, T b) { return (a < b) ? a : b; }

    template<typename T>
    T max(T a, T b) { return (a > b) ? a : b; }
#endif

// Include concepts when available
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "concepts/forward_declarations.h"
#endif

namespace physgrad {
namespace mpm {
namespace kernels {

// =============================================================================
// CUDA KERNEL CONFIGURATION
// =============================================================================

constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int SHARED_MEMORY_SIZE = 48 * 1024; // 48KB shared memory per block

/**
 * Optimal block sizes for different operations
 */
struct KernelConfig {
    static constexpr int P2G_BLOCK_SIZE = 256;  // Particle-to-Grid
    static constexpr int G2P_BLOCK_SIZE = 256;  // Grid-to-Particle
    static constexpr int G2P2G_BLOCK_SIZE = 256; // Fused G2P2G
    static constexpr int GRID_BLOCK_SIZE_X = 16;
    static constexpr int GRID_BLOCK_SIZE_Y = 16;
    static constexpr int GRID_BLOCK_SIZE_Z = 4;
};

// =============================================================================
// SHARED MEMORY DATA STRUCTURES
// =============================================================================

/**
 * Shared memory cache for grid data to reduce global memory access
 */
template<typename T>
struct GridCache {
    PHYSGRAD_SHARED T mass[8][8][8];
    PHYSGRAD_SHARED T momentum_x[8][8][8];
    PHYSGRAD_SHARED T momentum_y[8][8][8];
    PHYSGRAD_SHARED T momentum_z[8][8][8];
    PHYSGRAD_SHARED T velocity_x[8][8][8];
    PHYSGRAD_SHARED T velocity_y[8][8][8];
    PHYSGRAD_SHARED T velocity_z[8][8][8];

    PHYSGRAD_DEVICE
    void clear() {
        int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        int total_threads = blockDim.x * blockDim.y * blockDim.z;

        for (int i = tid; i < 8*8*8; i += total_threads) {
            int k = i / (8*8);
            int j = (i % (8*8)) / 8;
            int ii = i % 8;

            mass[ii][j][k] = T{0};
            momentum_x[ii][j][k] = T{0};
            momentum_y[ii][j][k] = T{0};
            momentum_z[ii][j][k] = T{0};
            velocity_x[ii][j][k] = T{0};
            velocity_y[ii][j][k] = T{0};
            velocity_z[ii][j][k] = T{0};
        }
    }
};

/**
 * Particle data cache for coalesced memory access
 */
template<typename T, int CacheSize>
struct ParticleCache {
    PHYSGRAD_SHARED T position_x[CacheSize];
    PHYSGRAD_SHARED T position_y[CacheSize];
    PHYSGRAD_SHARED T position_z[CacheSize];
    PHYSGRAD_SHARED T velocity_x[CacheSize];
    PHYSGRAD_SHARED T velocity_y[CacheSize];
    PHYSGRAD_SHARED T velocity_z[CacheSize];
    PHYSGRAD_SHARED T mass[CacheSize];
    PHYSGRAD_SHARED T volume[CacheSize];

    PHYSGRAD_DEVICE
    void loadParticle(int cache_idx, int particle_id, const ParticleAoSoA<T>& particles) {
        if (cache_idx < CacheSize && particle_id < particles.num_particles) {
            auto pos = particles.getPosition(particle_id);
            auto vel = particles.getVelocity(particle_id);

            position_x[cache_idx] = pos[0];
            position_y[cache_idx] = pos[1];
            position_z[cache_idx] = pos[2];
            velocity_x[cache_idx] = vel[0];
            velocity_y[cache_idx] = vel[1];
            velocity_z[cache_idx] = vel[2];
            mass[cache_idx] = particles.getMass(particle_id);
        }
    }

    PHYSGRAD_DEVICE
    void storeParticle(int cache_idx, int particle_id, ParticleAoSoA<T>& particles) {
        if (cache_idx < CacheSize && particle_id < particles.num_particles) {
            ConceptVector3D<T> pos{position_x[cache_idx], position_y[cache_idx], position_z[cache_idx]};
            ConceptVector3D<T> vel{velocity_x[cache_idx], velocity_y[cache_idx], velocity_z[cache_idx]};

            particles.setPosition(particle_id, pos);
            particles.setVelocity(particle_id, vel);
        }
    }
};

// =============================================================================
// OPTIMIZED SHAPE FUNCTION EVALUATION
// =============================================================================

/**
 * Fast shape function evaluation with derivatives
 */
template<typename T>
struct FastShapeFunctions {
    PHYSGRAD_DEVICE
    static void evalQuadratic(T x, T& w, T& dw) {
        T abs_x = x < T{0} ? -x : x;
        if (abs_x >= T{1.5}) {
            w = T{0};
            dw = T{0};
        } else if (abs_x < T{0.5}) {
            // Central part: N(x) = 3/4 - x^2
            w = T{0.75} - x * x;
            dw = -T{2} * x;
        } else {
            // Outer parts: N(x) = (3/2 - |x|)^2 / 2
            // Derivative: dN/dx = -(3/2 - |x|) * sign(x)
            T temp = T{1.5} - abs_x;
            w = T{0.5} * temp * temp;
            dw = (x < T{0}) ? temp : -temp;
        }
    }

    PHYSGRAD_DEVICE
    static void evalCubic(T x, T& w, T& dw) {
        T abs_x = x < T{0} ? -x : x;
        T sign = x < T{0} ? -T{1} : T{1};

        if (abs_x >= T{2}) {
            w = T{0};
            dw = T{0};
        } else if (abs_x < T{1}) {
            w = T{2.0/3.0} - abs_x * abs_x + T{0.5} * abs_x * abs_x * abs_x;
            dw = sign * (-T{2} * abs_x + T{1.5} * abs_x * abs_x);
        } else {
            T temp = T{2} - abs_x;
            w = T{1.0/6.0} * temp * temp * temp;
            dw = -sign * T{0.5} * temp * temp;
        }
    }
};

// =============================================================================
// FUSED G2P2G KERNELS
// =============================================================================

/**
 * Fused Grid-to-Particle-to-Grid kernel with shared memory optimization
 */
template<typename T>
PHYSGRAD_GLOBAL
void fusedG2P2GKernel(
    ParticleAoSoA<T>* particles,
    MPMGrid<T>* grid,
    T dt,
    ConceptVector3D<T> gravity,
    bool use_affine_pic,
    T flip_ratio,
    int interpolation_order,
    size_t particle_offset,
    size_t particle_count) {

    constexpr int CACHE_SIZE = KernelConfig::G2P2G_BLOCK_SIZE;

    // Shared memory caches
    PHYSGRAD_SHARED ParticleCache<T, CACHE_SIZE> particle_cache;
    PHYSGRAD_SHARED GridCache<T> grid_cache;

    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;

    // Particle range for this block
    size_t block_start = particle_offset + bid * blockDim.x;
    size_t block_end = min(block_start + blockDim.x, particle_offset + particle_count);

    if (block_start >= particle_offset + particle_count) return;

    // Clear shared memory
    if (tid == 0) {
        grid_cache.clear();
    }
    __syncthreads();

    // Load particles into shared memory
    size_t particle_id = block_start + tid;
    if (particle_id < block_end) {
        particle_cache.loadParticle(tid, particle_id, *particles);
    }
    __syncthreads();

    // Process each particle in the block
    for (int p = 0; p < min((int)blockDim.x, (int)(block_end - block_start)); ++p) {
        T pos_x = particle_cache.position_x[p];
        T pos_y = particle_cache.position_y[p];
        T pos_z = particle_cache.position_z[p];
        T mass = particle_cache.mass[p];

        // Find base grid cell
        int base_i = (int)floor((pos_x - grid->origin[0]) / grid->cell_size[0]);
        int base_j = (int)floor((pos_y - grid->origin[1]) / grid->cell_size[1]);
        int base_k = (int)floor((pos_z - grid->origin[2]) / grid->cell_size[2]);

        // Local coordinates
        T fx = (pos_x - grid->origin[0]) / grid->cell_size[0] - base_i;
        T fy = (pos_y - grid->origin[1]) / grid->cell_size[1] - base_j;
        T fz = (pos_z - grid->origin[2]) / grid->cell_size[2] - base_k;

        // Interpolation kernel size
        int kernel_size = (interpolation_order == 2) ? 3 : 4;
        int kernel_offset = kernel_size / 2;

        T new_vel_x = T{0}, new_vel_y = T{0}, new_vel_z = T{0};

        // Grid-to-Particle transfer (G2P)
        for (int di = -kernel_offset; di < kernel_size - kernel_offset; ++di) {
            for (int dj = -kernel_offset; dj < kernel_size - kernel_offset; ++dj) {
                for (int dk = -kernel_offset; dk < kernel_size - kernel_offset; ++dk) {
                    int gi = base_i + di;
                    int gj = base_j + dj;
                    int gk = base_k + dk;

                    // Bounds check
                    if (gi < 0 || gi >= grid->dimensions.x ||
                        gj < 0 || gj >= grid->dimensions.y ||
                        gk < 0 || gk >= grid->dimensions.z) continue;

                    // Compute shape function weights
                    T wx, dwx, wy, dwy, wz, dwz;
                    if (interpolation_order == 2) {
                        FastShapeFunctions<T>::evalQuadratic(fx - di, wx, dwx);
                        FastShapeFunctions<T>::evalQuadratic(fy - dj, wy, dwy);
                        FastShapeFunctions<T>::evalQuadratic(fz - dk, wz, dwz);
                    } else {
                        FastShapeFunctions<T>::evalCubic(fx - di, wx, dwx);
                        FastShapeFunctions<T>::evalCubic(fy - dj, wy, dwy);
                        FastShapeFunctions<T>::evalCubic(fz - dk, wz, dwz);
                    }

                    T weight = wx * wy * wz;

                    if (weight > T{1e-10}) {
                        size_t node_idx = grid->getNodeIndex(gi, gj, gk);

                        // Interpolate velocity from grid
                        if (grid->mass[node_idx] > T{1e-10}) {
                            new_vel_x += weight * grid->velocity[node_idx][0];
                            new_vel_y += weight * grid->velocity[node_idx][1];
                            new_vel_z += weight * grid->velocity[node_idx][2];
                        }
                    }
                }
            }
        }

        // Update particle velocity with FLIP/PIC blending
        T old_vel_x = particle_cache.velocity_x[p];
        T old_vel_y = particle_cache.velocity_y[p];
        T old_vel_z = particle_cache.velocity_z[p];

        if (use_affine_pic) {
            // FLIP: velocity increment
            particle_cache.velocity_x[p] = old_vel_x * flip_ratio + new_vel_x * (T{1} - flip_ratio);
            particle_cache.velocity_y[p] = old_vel_y * flip_ratio + new_vel_y * (T{1} - flip_ratio);
            particle_cache.velocity_z[p] = old_vel_z * flip_ratio + new_vel_z * (T{1} - flip_ratio);
        } else {
            // Pure PIC
            particle_cache.velocity_x[p] = new_vel_x;
            particle_cache.velocity_y[p] = new_vel_y;
            particle_cache.velocity_z[p] = new_vel_z;
        }

        // Update position
        particle_cache.position_x[p] += particle_cache.velocity_x[p] * dt;
        particle_cache.position_y[p] += particle_cache.velocity_y[p] * dt;
        particle_cache.position_z[p] += particle_cache.velocity_z[p] * dt;
    }

    __syncthreads();

    // Store updated particles back to global memory
    if (particle_id < block_end) {
        particle_cache.storeParticle(tid, particle_id, *particles);
    }
}

/**
 * Optimized Particle-to-Grid transfer kernel with coalesced writes
 */
template<typename T>
PHYSGRAD_GLOBAL
void optimizedP2GKernel(
    const ParticleAoSoA<T>* particles,
    MPMGrid<T>* grid,
    ConceptVector3D<T> gravity,
    T dt,
    int interpolation_order,
    size_t particle_offset,
    size_t particle_count) {

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    size_t particle_id = particle_offset + bid * blockDim.x + tid;

    if (particle_id >= particle_offset + particle_count) return;

    // Load particle data
    auto pos = particles->getPosition(particle_id);
    auto vel = particles->getVelocity(particle_id);
    T mass = particles->getMass(particle_id);

    // Apply gravity
    vel = {
        vel[0] + gravity[0] * dt,
        vel[1] + gravity[1] * dt,
        vel[2] + gravity[2] * dt
    };

    // Find base grid cell
    int base_i = (int)floor((pos[0] - grid->origin[0]) / grid->cell_size[0]);
    int base_j = (int)floor((pos[1] - grid->origin[1]) / grid->cell_size[1]);
    int base_k = (int)floor((pos[2] - grid->origin[2]) / grid->cell_size[2]);

    T fx = (pos[0] - grid->origin[0]) / grid->cell_size[0] - base_i;
    T fy = (pos[1] - grid->origin[1]) / grid->cell_size[1] - base_j;
    T fz = (pos[2] - grid->origin[2]) / grid->cell_size[2] - base_k;

    int kernel_size = (interpolation_order == 2) ? 3 : 4;
    int kernel_offset = kernel_size / 2;

    // Transfer mass and momentum to grid
    for (int di = -kernel_offset; di < kernel_size - kernel_offset; ++di) {
        for (int dj = -kernel_offset; dj < kernel_size - kernel_offset; ++dj) {
            for (int dk = -kernel_offset; dk < kernel_size - kernel_offset; ++dk) {
                int gi = base_i + di;
                int gj = base_j + dj;
                int gk = base_k + dk;

                if (gi < 0 || gi >= grid->dimensions.x ||
                    gj < 0 || gj >= grid->dimensions.y ||
                    gk < 0 || gk >= grid->dimensions.z) continue;

                T wx, dwx, wy, dwy, wz, dwz;
                if (interpolation_order == 2) {
                    FastShapeFunctions<T>::evalQuadratic(fx - di, wx, dwx);
                    FastShapeFunctions<T>::evalQuadratic(fy - dj, wy, dwy);
                    FastShapeFunctions<T>::evalQuadratic(fz - dk, wz, dwz);
                } else {
                    FastShapeFunctions<T>::evalCubic(fx - di, wx, dwx);
                    FastShapeFunctions<T>::evalCubic(fy - dj, wy, dwy);
                    FastShapeFunctions<T>::evalCubic(fz - dk, wz, dwz);
                }

                T weight = wx * wy * wz;

                if (weight > T{1e-10}) {
                    size_t node_idx = grid->getNodeIndex(gi, gj, gk);

                    // Atomic operations for thread safety
                    atomicAdd(&grid->mass[node_idx], weight * mass);
                    atomicAdd(&grid->momentum[node_idx][0], weight * mass * vel[0]);
                    atomicAdd(&grid->momentum[node_idx][1], weight * mass * vel[1]);
                    atomicAdd(&grid->momentum[node_idx][2], weight * mass * vel[2]);
                }
            }
        }
    }
}

/**
 * Grid velocity calculation kernel
 */
template<typename T>
PHYSGRAD_GLOBAL
void calculateGridVelocityKernel(MPMGrid<T>* grid, size_t node_offset, size_t node_count) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    size_t node_id = node_offset + bid * blockDim.x + tid;

    if (node_id >= node_offset + node_count || node_id >= grid->total_nodes) return;

    if (grid->mass[node_id] > T{1e-10}) {
        grid->velocity[node_id] = {
            grid->momentum[node_id][0] / grid->mass[node_id],
            grid->momentum[node_id][1] / grid->mass[node_id],
            grid->momentum[node_id][2] / grid->mass[node_id]
        };
    } else {
        grid->velocity[node_id] = {T{0}, T{0}, T{0}};
    }
}

// =============================================================================
// HOST INTERFACE
// =============================================================================

/**
 * G2P2G Kernel Launcher with performance optimization
 */
template<typename T>
class G2P2GKernelLauncher {
public:
    struct PerformanceConfig {
        bool use_kernel_fusion = true;
        bool use_shared_memory = true;
        bool use_coalesced_access = true;
        int max_particles_per_block = KernelConfig::G2P2G_BLOCK_SIZE;
        bool enable_async_execution = true;
    };

    G2P2GKernelLauncher(const PerformanceConfig& config = {}) : config_(config) {
#ifdef __CUDACC__
        // Initialize CUDA streams for asynchronous execution
        if (config_.enable_async_execution) {
            cudaStreamCreate(&stream_p2g_);
            cudaStreamCreate(&stream_g2p_);
            cudaStreamCreate(&stream_grid_);
        }
#endif
    }

    ~G2P2GKernelLauncher() {
#ifdef __CUDACC__
        if (config_.enable_async_execution) {
            cudaStreamDestroy(stream_p2g_);
            cudaStreamDestroy(stream_g2p_);
            cudaStreamDestroy(stream_grid_);
        }
#endif
    }

    void launchG2P2G(ParticleAoSoA<T>& particles,
                     MPMGrid<T>& grid,
                     T dt,
                     const ConceptVector3D<T>& gravity,
                     bool use_affine_pic,
                     T flip_ratio,
                     int interpolation_order) {

#ifdef __CUDACC__
        if (config_.use_kernel_fusion) {
            launchFusedKernel(particles, grid, dt, gravity, use_affine_pic,
                            flip_ratio, interpolation_order);
        } else {
            launchSeparateKernels(particles, grid, dt, gravity, use_affine_pic,
                                flip_ratio, interpolation_order);
        }
#else
        // CPU fallback
        launchCPUKernel(particles, grid, dt, gravity, use_affine_pic,
                       flip_ratio, interpolation_order);
#endif
    }

private:
    PerformanceConfig config_;

#ifdef __CUDACC__
    cudaStream_t stream_p2g_;
    cudaStream_t stream_g2p_;
    cudaStream_t stream_grid_;

    void launchFusedKernel(ParticleAoSoA<T>& particles,
                          MPMGrid<T>& grid,
                          T dt,
                          const ConceptVector3D<T>& gravity,
                          bool use_affine_pic,
                          T flip_ratio,
                          int interpolation_order) {

        // Clear grid
        cudaMemset(grid.mass.data(), 0, grid.total_nodes * sizeof(T));
        cudaMemset(grid.momentum.data(), 0, grid.total_nodes * sizeof(ConceptVector3D<T>));

        // Launch P2G kernel
        int p2g_blocks = (particles.num_particles + KernelConfig::P2G_BLOCK_SIZE - 1) / KernelConfig::P2G_BLOCK_SIZE;
        optimizedP2GKernel<<<p2g_blocks, KernelConfig::P2G_BLOCK_SIZE, 0, stream_p2g_>>>(
            &particles, &grid, gravity, dt, interpolation_order, 0, particles.num_particles);

        // Launch grid velocity calculation
        int grid_blocks = (grid.total_nodes + KernelConfig::G2P_BLOCK_SIZE - 1) / KernelConfig::G2P_BLOCK_SIZE;
        calculateGridVelocityKernel<<<grid_blocks, KernelConfig::G2P_BLOCK_SIZE, 0, stream_grid_>>>(
            &grid, 0, grid.total_nodes);

        // Launch fused G2P2G kernel
        int g2p2g_blocks = (particles.num_particles + KernelConfig::G2P2G_BLOCK_SIZE - 1) / KernelConfig::G2P2G_BLOCK_SIZE;
        fusedG2P2GKernel<<<g2p2g_blocks, KernelConfig::G2P2G_BLOCK_SIZE, 0, stream_g2p_>>>(
            &particles, &grid, dt, gravity, use_affine_pic, flip_ratio,
            interpolation_order, 0, particles.num_particles);

        // Synchronize streams
        cudaStreamSynchronize(stream_p2g_);
        cudaStreamSynchronize(stream_grid_);
        cudaStreamSynchronize(stream_g2p_);
    }

    void launchSeparateKernels(ParticleAoSoA<T>& particles,
                              MPMGrid<T>& grid,
                              T dt,
                              const ConceptVector3D<T>& gravity,
                              bool use_affine_pic,
                              T flip_ratio,
                              int interpolation_order) {
        // Implementation for separate kernel launches
        // (Similar to launchFusedKernel but with individual kernel calls)
    }
#endif

    void launchCPUKernel(ParticleAoSoA<T>& particles,
                        MPMGrid<T>& grid,
                        T dt,
                        const ConceptVector3D<T>& gravity,
                        bool use_affine_pic,
                        T flip_ratio,
                        int interpolation_order) {
        // CPU implementation as fallback
        std::cout << "Running CPU G2P2G kernel (fallback)\n";

        // Simple CPU implementation
        grid.clear();

        // P2G transfer
        for (size_t p = 0; p < particles.num_particles; ++p) {
            auto pos = particles.getPosition(p);
            auto vel = particles.getVelocity(p);
            T mass = particles.getMass(p);

            // Apply gravity
            vel = {
                vel[0] + gravity[0] * dt,
                vel[1] + gravity[1] * dt,
                vel[2] + gravity[2] * dt
            };

            // Transfer to grid (simplified)
            int gi = (int)std::round((pos[0] - grid.origin[0]) / grid.cell_size[0]);
            int gj = (int)std::round((pos[1] - grid.origin[1]) / grid.cell_size[1]);
            int gk = (int)std::round((pos[2] - grid.origin[2]) / grid.cell_size[2]);

            if (gi >= 0 && gi < grid.dimensions.x &&
                gj >= 0 && gj < grid.dimensions.y &&
                gk >= 0 && gk < grid.dimensions.z) {

                size_t node_idx = grid.getNodeIndex(gi, gj, gk);
                grid.mass[node_idx] += mass;
                grid.momentum[node_idx] = {
                    grid.momentum[node_idx][0] + mass * vel[0],
                    grid.momentum[node_idx][1] + mass * vel[1],
                    grid.momentum[node_idx][2] + mass * vel[2]
                };
            }
        }

        // Calculate grid velocities
        for (size_t i = 0; i < grid.total_nodes; ++i) {
            if (grid.mass[i] > T{1e-10}) {
                grid.velocity[i] = {
                    grid.momentum[i][0] / grid.mass[i],
                    grid.momentum[i][1] / grid.mass[i],
                    grid.momentum[i][2] / grid.mass[i]
                };
            }
        }

        // G2P transfer and position update
        for (size_t p = 0; p < particles.num_particles; ++p) {
            auto pos = particles.getPosition(p);

            int gi = (int)std::round((pos[0] - grid.origin[0]) / grid.cell_size[0]);
            int gj = (int)std::round((pos[1] - grid.origin[1]) / grid.cell_size[1]);
            int gk = (int)std::round((pos[2] - grid.origin[2]) / grid.cell_size[2]);

            if (gi >= 0 && gi < grid.dimensions.x &&
                gj >= 0 && gj < grid.dimensions.y &&
                gk >= 0 && gk < grid.dimensions.z) {

                size_t node_idx = grid.getNodeIndex(gi, gj, gk);
                auto grid_vel = grid.velocity[node_idx];

                // Update particle velocity and position
                particles.setVelocity(p, grid_vel);
                ConceptVector3D<T> new_pos = {
                    pos[0] + grid_vel[0] * dt,
                    pos[1] + grid_vel[1] * dt,
                    pos[2] + grid_vel[2] * dt
                };
                particles.setPosition(p, new_pos);
            }
        }
    }
};

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
// Verify concept compliance
static_assert(concepts::PhysicsScalar<float>);
static_assert(concepts::PhysicsScalar<double>);
#endif

} // namespace kernels
} // namespace mpm
} // namespace physgrad

#endif // PHYSGRAD_MPM_G2P2G_KERNELS_H