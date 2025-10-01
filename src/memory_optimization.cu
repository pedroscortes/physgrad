#include "memory_optimization.h"
#include <cub/cub.cuh>
#include <cuda.h>
#include <iostream>
#include <algorithm>
#include <chrono>

namespace physgrad {

// ===============================================================
// OPTIMIZED MEMORY ACCESS PRIMITIVES
// ===============================================================

namespace optimized_access {

// Vectorized load operations for different data types
template<>
__device__ __forceinline__ void vectorized_load<float, 4>(
    float* dst, const float* __restrict__ src, int num_elements
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Use float4 for vectorized loading
    float4* dst4 = reinterpret_cast<float4*>(dst);
    const float4* src4 = reinterpret_cast<const float4*>(src);
    int vec_elements = num_elements / 4;

    for (int i = tid; i < vec_elements; i += stride) {
        dst4[i] = src4[i];
    }

    // Handle remaining elements
    int remaining_start = vec_elements * 4;
    for (int i = remaining_start + tid; i < num_elements; i += stride) {
        dst[i] = src[i];
    }
}

template<>
__device__ __forceinline__ void vectorized_store<float, 4>(
    float* __restrict__ dst, const float* src, int num_elements
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Use float4 for vectorized storing
    float4* dst4 = reinterpret_cast<float4*>(dst);
    const float4* src4 = reinterpret_cast<const float4*>(src);
    int vec_elements = num_elements / 4;

    for (int i = tid; i < vec_elements; i += stride) {
        dst4[i] = src4[i];
    }

    // Handle remaining elements
    int remaining_start = vec_elements * 4;
    for (int i = remaining_start + tid; i < num_elements; i += stride) {
        dst[i] = src[i];
    }
}

// Optimized transpose kernel with shared memory
template<int TILE_DIM, int BLOCK_ROWS>
__global__ void coalesced_transpose_kernel(
    float* __restrict__ odata,
    const float* __restrict__ idata,
    int width, int height
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced read from global memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < height && x < width) {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
        }
    }

    __syncthreads();

    // Calculate transposed coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Coalesced write to global memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < width && x < height) {
            odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Shared memory optimized load/store
template<int SHARED_SIZE>
__device__ __forceinline__ void shared_memory_load_store(
    float* __restrict__ global_out,
    const float* __restrict__ global_in,
    int thread_id, int num_threads
) {
    __shared__ float shared_data[SHARED_SIZE];

    // Coalesced load into shared memory
    for (int i = thread_id; i < SHARED_SIZE; i += num_threads) {
        shared_data[i] = global_in[i];
    }

    __syncthreads();

    // Process data in shared memory (example: simple transformation)
    for (int i = thread_id; i < SHARED_SIZE; i += num_threads) {
        shared_data[i] = shared_data[i] * 2.0f + 1.0f;
    }

    __syncthreads();

    // Coalesced store back to global memory
    for (int i = thread_id; i < SHARED_SIZE; i += num_threads) {
        global_out[i] = shared_data[i];
    }
}

} // namespace optimized_access

// ===============================================================
// AOSOA CONTAINER IMPLEMENTATION
// ===============================================================

template<typename T, int VECTOR_SIZE>
AoSoAContainer<T, VECTOR_SIZE>::AoSoAContainer(size_t initial_capacity)
    : num_elements_(0), capacity_(initial_capacity) {

    // Align capacity to chunk boundaries
    size_t aligned_capacity = ((initial_capacity + VECTORS_PER_CHUNK - 1) / VECTORS_PER_CHUNK) * VECTORS_PER_CHUNK;

    cudaError_t err = cudaMalloc(&data_, aligned_capacity * VECTOR_SIZE * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate AoSoA container memory");
    }

    capacity_ = aligned_capacity;
}

template<typename T, int VECTOR_SIZE>
AoSoAContainer<T, VECTOR_SIZE>::~AoSoAContainer() {
    if (data_) {
        cudaFree(data_);
    }
}

template<typename T, int VECTOR_SIZE>
__device__ __host__ T* AoSoAContainer<T, VECTOR_SIZE>::getVectorPtr(size_t element_id, int vector_component) {
    size_t chunk_id = element_id / VECTORS_PER_CHUNK;
    size_t local_id = element_id % VECTORS_PER_CHUNK;

    size_t offset = chunk_id * VECTORS_PER_CHUNK * VECTOR_SIZE +
                   vector_component * VECTORS_PER_CHUNK +
                   local_id;

    return &data_[offset];
}

template<typename T, int VECTOR_SIZE>
__device__ __host__ const T* AoSoAContainer<T, VECTOR_SIZE>::getVectorPtr(size_t element_id, int vector_component) const {
    size_t chunk_id = element_id / VECTORS_PER_CHUNK;
    size_t local_id = element_id % VECTORS_PER_CHUNK;

    size_t offset = chunk_id * VECTORS_PER_CHUNK * VECTOR_SIZE +
                   vector_component * VECTORS_PER_CHUNK +
                   local_id;

    return &data_[offset];
}

// ===============================================================
// MORTON ORDER OPTIMIZER
// ===============================================================

__device__ __host__ uint64_t MortonOrderOptimizer::encode3D(uint32_t x, uint32_t y, uint32_t z) {
    auto expandBits = [](uint32_t v) -> uint64_t {
        uint64_t result = v;
        result = (result | (result << 32)) & 0x1f00000000ffffULL;
        result = (result | (result << 16)) & 0x1f0000ff0000ffULL;
        result = (result | (result << 8))  & 0x100f00f00f00f00fULL;
        result = (result | (result << 4))  & 0x10c30c30c30c30c3ULL;
        result = (result | (result << 2))  & 0x1249249249249249ULL;
        return result;
    };

    return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
}

__device__ __host__ void MortonOrderOptimizer::decode3D(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z) {
    auto compactBits = [](uint64_t v) -> uint32_t {
        v &= 0x1249249249249249ULL;
        v = (v | (v >> 2))  & 0x10c30c30c30c30c3ULL;
        v = (v | (v >> 4))  & 0x100f00f00f00f00fULL;
        v = (v | (v >> 8))  & 0x1f0000ff0000ffULL;
        v = (v | (v >> 16)) & 0x1f00000000ffffULL;
        v = (v | (v >> 32)) & 0xffffffffULL;
        return static_cast<uint32_t>(v);
    };

    x = compactBits(morton);
    y = compactBits(morton >> 1);
    z = compactBits(morton >> 2);
}

// ===============================================================
// WARP-LEVEL OPTIMIZATIONS
// ===============================================================

namespace warp_optimized {

template<typename T>
__device__ __forceinline__ T warp_coalesced_load(const T* __restrict__ addr, int lane_id) {
    // Ensure coalesced access by having consecutive threads access consecutive addresses
    return addr[lane_id];
}

template<typename T>
__device__ __forceinline__ T warp_shuffle_distribute(T value, int src_lane, int width) {
    return __shfl_sync(0xffffffff, value, src_lane, width);
}

template<typename T, int WARP_SIZE>
__device__ __forceinline__ T warp_reduce_sum(T value) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template<typename T>
__device__ __forceinline__ T warp_scan_inclusive(T value) {
    int lane_id = threadIdx.x & 31;

    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        T temp = __shfl_up_sync(0xffffffff, value, offset);
        if (lane_id >= offset) {
            value += temp;
        }
    }

    return value;
}

} // namespace warp_optimized

// ===============================================================
// OPTIMIZED PHYSICS KERNELS
// ===============================================================

// High-performance particle force computation with memory optimization
__global__ void optimized_force_computation_kernel(
    const float4* __restrict__ positions,  // w component for mass
    const float* __restrict__ charges,
    float4* __restrict__ forces,           // w component for potential energy
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float4 pos_i = positions[i];
    float charge_i = charges[i];
    float4 total_force = {0.0f, 0.0f, 0.0f, 0.0f}; // xyz = force, w = potential

    // Use shared memory to cache nearby particles
    __shared__ float4 shared_positions[256];
    __shared__ float shared_charges[256];

    const int cache_size = min(blockDim.x, 256);

    for (int tile = 0; tile < (num_particles + cache_size - 1) / cache_size; ++tile) {
        int cache_idx = threadIdx.x;
        int global_idx = tile * cache_size + cache_idx;

        // Coalesced load into shared memory
        if (global_idx < num_particles && cache_idx < cache_size) {
            shared_positions[cache_idx] = positions[global_idx];
            shared_charges[cache_idx] = charges[global_idx];
        }

        __syncthreads();

        // Compute forces with cached data
        int tile_end = min(cache_size, num_particles - tile * cache_size);
        for (int j = 0; j < tile_end; ++j) {
            int global_j = tile * cache_size + j;
            if (i == global_j) continue;

            float4 pos_j = shared_positions[j];
            float charge_j = shared_charges[j];

            float3 r_ij = {
                pos_i.x - pos_j.x,
                pos_i.y - pos_j.y,
                pos_i.z - pos_j.z
            };

            float r = sqrtf(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);

            if (r > 1e-6f) {
                float inv_r = 1.0f / r;
                float inv_r2 = inv_r * inv_r;
                float force_magnitude = 8.9875517923e9f * charge_i * charge_j * inv_r2;

                total_force.x += force_magnitude * r_ij.x * inv_r;
                total_force.y += force_magnitude * r_ij.y * inv_r;
                total_force.z += force_magnitude * r_ij.z * inv_r;
                total_force.w += 8.9875517923e9f * charge_i * charge_j * inv_r; // potential
            }
        }

        __syncthreads();
    }

    forces[i] = total_force;
}

// Memory-coalesced Verlet integration with vectorized operations
__global__ void optimized_verlet_integration_kernel(
    float4* __restrict__ positions,    // xyz = position, w = mass
    float4* __restrict__ velocities,   // xyz = velocity, w = kinetic energy
    const float4* __restrict__ forces, // xyz = force, w = potential energy
    float dt,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float4 pos = positions[i];
    float4 vel = velocities[i];
    float4 force = forces[i];

    float inv_mass = 1.0f / pos.w;

    // Calculate acceleration: a = F/m
    float3 accel = {
        force.x * inv_mass,
        force.y * inv_mass,
        force.z * inv_mass
    };

    // Update position: x += v*dt + 0.5*a*dt^2
    pos.x += vel.x * dt + 0.5f * accel.x * dt * dt;
    pos.y += vel.y * dt + 0.5f * accel.y * dt * dt;
    pos.z += vel.z * dt + 0.5f * accel.z * dt * dt;

    // Update velocity: v += a*dt
    vel.x += accel.x * dt;
    vel.y += accel.y * dt;
    vel.z += accel.z * dt;

    // Calculate kinetic energy
    float vel_squared = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    vel.w = 0.5f * pos.w * vel_squared;

    // Store results with coalesced writes
    positions[i] = pos;
    velocities[i] = vel;
}

// Warp-cooperative reduction for energy summation
__global__ void optimized_energy_reduction_kernel(
    const float4* __restrict__ velocities, // w = kinetic energy
    const float4* __restrict__ forces,     // w = potential energy
    float* __restrict__ total_kinetic,
    float* __restrict__ total_potential,
    int num_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float kinetic_sum = 0.0f;
    float potential_sum = 0.0f;

    // Grid-stride loop for coalesced access
    for (int i = tid; i < num_particles; i += stride) {
        kinetic_sum += velocities[i].w;
        potential_sum += forces[i].w;
    }

    // Warp-level reduction
    kinetic_sum = warp_optimized::warp_reduce_sum<float, 32>(kinetic_sum);
    potential_sum = warp_optimized::warp_reduce_sum<float, 32>(potential_sum);

    // Block-level reduction using shared memory
    __shared__ float shared_kinetic[32];
    __shared__ float shared_potential[32];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        shared_kinetic[warp_id] = kinetic_sum;
        shared_potential[warp_id] = potential_sum;
    }

    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        kinetic_sum = (lane_id < (blockDim.x + 31) / 32) ? shared_kinetic[lane_id] : 0.0f;
        potential_sum = (lane_id < (blockDim.x + 31) / 32) ? shared_potential[lane_id] : 0.0f;

        kinetic_sum = warp_optimized::warp_reduce_sum<float, 32>(kinetic_sum);
        potential_sum = warp_optimized::warp_reduce_sum<float, 32>(potential_sum);

        if (lane_id == 0) {
            atomicAdd(total_kinetic, kinetic_sum);
            atomicAdd(total_potential, potential_sum);
        }
    }
}

// ===============================================================
// MEMORY BANDWIDTH OPTIMIZER IMPLEMENTATION
// ===============================================================

MemoryBandwidthOptimizer::BandwidthResult MemoryBandwidthOptimizer::benchmarkAccessPattern(
    size_t data_size,
    MemoryAccessPattern pattern,
    int iterations
) {
    BandwidthResult result = {};

    // Allocate test data
    float* d_data;
    float* d_output;
    cudaMalloc(&d_data, data_size * sizeof(float));
    cudaMalloc(&d_output, data_size * sizeof(float));

    // Initialize data
    cudaMemset(d_data, 0, data_size * sizeof(float));

    // Warm up GPU
    dim3 block_size(256);
    dim3 grid_size((data_size + block_size.x - 1) / block_size.x);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        switch (pattern) {
            case MemoryAccessPattern::COALESCED_SEQUENTIAL:
                optimized_access::vectorized_load<float, 4><<<grid_size, block_size>>>(
                    d_output, d_data, data_size);
                break;

            case MemoryAccessPattern::STRIDED_REGULAR:
                // Custom strided access kernel would go here
                break;

            default:
                // Default to simple copy
                cudaMemcpy(d_output, d_data, data_size * sizeof(float), cudaMemcpyDeviceToDevice);
                break;
        }
    }

    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate bandwidth
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double time_seconds = duration.count() / 1e6;
    double bytes_transferred = 2.0 * data_size * sizeof(float) * iterations; // read + write

    result.achieved_bandwidth_gb_s = (bytes_transferred / time_seconds) / 1e9;

    // Get theoretical bandwidth (approximate)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    result.theoretical_bandwidth_gb_s = (prop.memoryClockRate * 1000.0 * prop.memoryBusWidth / 8.0) / 1e9;
    result.efficiency_percentage = (result.achieved_bandwidth_gb_s / result.theoretical_bandwidth_gb_s) * 100.0f;

    cudaFree(d_data);
    cudaFree(d_output);

    return result;
}

dim3 MemoryBandwidthOptimizer::findOptimalLaunchParams(size_t data_size, size_t shared_memory_per_block) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Calculate optimal block size based on occupancy
    int min_grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       optimized_verlet_integration_kernel,
                                       shared_memory_per_block, 0);

    // Calculate grid size to cover all data
    int grid_size = (data_size + block_size - 1) / block_size;

    return dim3(grid_size, 1, 1);
}

// ===============================================================
// STREAM OPTIMIZER IMPLEMENTATION
// ===============================================================

StreamOptimizer::StreamOptimizer(int num_streams) : num_streams_(num_streams) {
    streams_.resize(num_streams_);
    for (int i = 0; i < num_streams_; ++i) {
        cudaStreamCreate(&streams_[i]);
    }
}

StreamOptimizer::~StreamOptimizer() {
    for (auto& stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

cudaError_t StreamOptimizer::overlappedMemcpyAndKernel(
    void* dst, const void* src, size_t size,
    void (*kernel)(void*), void* kernel_args,
    int stream_id
) {
    if (stream_id >= num_streams_) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = streams_[stream_id];

    // Asynchronous memory copy
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;

    // Launch kernel on same stream (will wait for memcpy to complete)
    // Note: This is a simplified interface - real implementation would need proper kernel launch

    return cudaSuccess;
}

// Explicit template instantiations
template class AoSoAContainer<float, 4>;
template class AoSoAContainer<float, 3>;

} // namespace physgrad