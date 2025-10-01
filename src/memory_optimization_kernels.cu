/**
 * GPU Memory Optimization Kernels
 *
 * Implementation of optimized memory access patterns for GPU coalescing
 */

#include "memory_optimization.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace physgrad {
namespace optimized_access {

using namespace cooperative_groups;

// =============================================================================
// VECTORIZED MEMORY OPERATIONS
// =============================================================================

template<typename T, int VecSize>
__device__ __forceinline__ void vectorized_load(
    T* dst, const T* __restrict__ src, int num_elements
) {
    static_assert(VecSize == 2 || VecSize == 4, "VecSize must be 2 or 4");

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    if constexpr (VecSize == 4 && sizeof(T) == 4) {
        // Use float4 for 4-element float vectors
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
    } else if constexpr (VecSize == 2 && sizeof(T) == 4) {
        // Use float2 for 2-element float vectors
        float2* dst2 = reinterpret_cast<float2*>(dst);
        const float2* src2 = reinterpret_cast<const float2*>(src);
        int vec_elements = num_elements / 2;

        for (int i = tid; i < vec_elements; i += stride) {
            dst2[i] = src2[i];
        }

        // Handle remaining elements
        if (num_elements % 2 == 1 && tid == 0) {
            dst[num_elements - 1] = src[num_elements - 1];
        }
    } else {
        // Fallback to scalar loads
        for (int i = tid; i < num_elements; i += stride) {
            dst[i] = src[i];
        }
    }
}

template<typename T, int VecSize>
__device__ __forceinline__ void vectorized_store(
    T* __restrict__ dst, const T* src, int num_elements
) {
    static_assert(VecSize == 2 || VecSize == 4, "VecSize must be 2 or 4");

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    if constexpr (VecSize == 4 && sizeof(T) == 4) {
        float4* dst4 = reinterpret_cast<float4*>(dst);
        const float4* src4 = reinterpret_cast<const float4*>(src);
        int vec_elements = num_elements / 4;

        for (int i = tid; i < vec_elements; i += stride) {
            dst4[i] = src4[i];
        }

        int remaining_start = vec_elements * 4;
        for (int i = remaining_start + tid; i < num_elements; i += stride) {
            dst[i] = src[i];
        }
    } else if constexpr (VecSize == 2 && sizeof(T) == 4) {
        float2* dst2 = reinterpret_cast<float2*>(dst);
        const float2* src2 = reinterpret_cast<const float2*>(src);
        int vec_elements = num_elements / 2;

        for (int i = tid; i < vec_elements; i += stride) {
            dst2[i] = src2[i];
        }

        if (num_elements % 2 == 1 && tid == 0) {
            dst[num_elements - 1] = src[num_elements - 1];
        }
    } else {
        for (int i = tid; i < num_elements; i += stride) {
            dst[i] = src[i];
        }
    }
}

// =============================================================================
// COALESCED TRANSPOSE KERNEL
// =============================================================================

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
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
        }
    }

    __syncthreads();

    // Transpose block indices for output
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Coalesced write to global memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// =============================================================================
// SHARED MEMORY OPTIMIZATION
// =============================================================================

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

    // Process data in shared memory (example: simple copy)
    // Real implementations would do computation here

    __syncthreads();

    // Coalesced store from shared memory
    for (int i = thread_id; i < SHARED_SIZE; i += num_threads) {
        global_out[i] = shared_data[i];
    }
}

// =============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// =============================================================================

template __device__ void vectorized_load<float, 4>(float*, const float*, int);
template __device__ void vectorized_load<float, 2>(float*, const float*, int);
template __device__ void vectorized_store<float, 4>(float*, const float*, int);
template __device__ void vectorized_store<float, 2>(float*, const float*, int);

template __global__ void coalesced_transpose_kernel<32, 8>(float*, const float*, int, int);
template __global__ void coalesced_transpose_kernel<16, 16>(float*, const float*, int, int);

template __device__ void shared_memory_load_store<1024>(float*, const float*, int, int);
template __device__ void shared_memory_load_store<2048>(float*, const float*, int, int);

} // namespace optimized_access

// =============================================================================
// WARP-LEVEL OPTIMIZATIONS
// =============================================================================

namespace warp_optimized {

template<typename T>
__device__ __forceinline__ T warp_coalesced_load(
    const T* __restrict__ addr, int lane_id
) {
    thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());
    return addr[warp.thread_rank()];
}

template<typename T>
__device__ __forceinline__ T warp_shuffle_distribute(
    T value, int src_lane, int width
) {
    return __shfl_sync(0xffffffff, value, src_lane, width);
}

template<typename T, int WARP_SIZE>
__device__ __forceinline__ T warp_reduce_sum(T value) {
    thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());

    #pragma unroll
    for (int delta = WARP_SIZE / 2; delta > 0; delta /= 2) {
        value += warp.shfl_down(value, delta);
    }

    return value;
}

template<typename T>
__device__ __forceinline__ T warp_scan_inclusive(T value) {
    thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());

    #pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        T temp = warp.shfl_up(value, delta);
        if (warp.thread_rank() >= delta) {
            value += temp;
        }
    }

    return value;
}

// Explicit instantiations
template __device__ float warp_coalesced_load<float>(const float*, int);
template __device__ double warp_coalesced_load<double>(const double*, int);
template __device__ float warp_shuffle_distribute<float>(float, int, int);
template __device__ double warp_shuffle_distribute<double>(double, int, int);
template __device__ float warp_reduce_sum<float, 32>(float);
template __device__ double warp_reduce_sum<double, 32>(double);
template __device__ float warp_scan_inclusive<float>(float);
template __device__ double warp_scan_inclusive<double>(double);

} // namespace warp_optimized

// =============================================================================
// MEMORY BANDWIDTH OPTIMIZATION KERNELS
// =============================================================================

// Bandwidth benchmark kernel
__global__ void bandwidth_benchmark_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    size_t num_elements,
    MemoryAccessPattern pattern
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    switch (pattern) {
        case MemoryAccessPattern::COALESCED_SEQUENTIAL:
            // Perfect coalesced access
            for (size_t i = tid; i < num_elements; i += stride) {
                output[i] = input[i] * 2.0f;
            }
            break;

        case MemoryAccessPattern::STRIDED_REGULAR:
            // Strided access with stride 8
            for (size_t i = tid * 8; i < num_elements; i += stride * 8) {
                if (i < num_elements) {
                    output[i] = input[i] * 2.0f;
                }
            }
            break;

        case MemoryAccessPattern::RANDOM_SCATTERED:
            // Pseudo-random access pattern
            for (size_t i = tid; i < num_elements; i += stride) {
                size_t random_idx = (i * 1103515245 + 12345) % num_elements;
                output[i] = input[random_idx] * 2.0f;
            }
            break;

        default:
            // Default to coalesced
            for (size_t i = tid; i < num_elements; i += stride) {
                output[i] = input[i] * 2.0f;
            }
            break;
    }
}

// Auto-tuning kernel launcher
template<typename KernelFunc>
__global__ void auto_tune_wrapper_kernel(KernelFunc kernel, void* args, size_t args_size) {
    // Simple wrapper - in practice this would be more sophisticated
    kernel(args);
}

} // namespace physgrad