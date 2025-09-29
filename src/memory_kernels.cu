/**
 * PhysGrad - GPU Memory Management CUDA Kernels
 *
 * CUDA kernels for efficient GPU memory operations and data management.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <float.h>
#include <stdint.h>

namespace physgrad {

// Memory initialization kernels
__global__ void zero_memory_kernel(float* __restrict__ data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

__global__ void set_memory_kernel(float* __restrict__ data, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

// Memory copy and transfer kernels
__global__ void copy_memory_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

__global__ void gather_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int* __restrict__ indices,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        dst[idx] = src[indices[idx]];
    }
}

__global__ void scatter_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int* __restrict__ indices,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        dst[indices[idx]] = src[idx];
    }
}

// Data compression and decompression
__global__ void compress_sparse_kernel(
    const float* __restrict__ dense_data,
    float* __restrict__ sparse_values,
    int* __restrict__ sparse_indices,
    int* __restrict__ num_nonzeros,
    float threshold,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    if (fabsf(dense_data[idx]) > threshold) {
        int sparse_idx = atomicAdd(num_nonzeros, 1);
        sparse_values[sparse_idx] = dense_data[idx];
        sparse_indices[sparse_idx] = idx;
    }
}

__global__ void decompress_sparse_kernel(
    const float* __restrict__ sparse_values,
    const int* __restrict__ sparse_indices,
    float* __restrict__ dense_data,
    int num_nonzeros,
    int dense_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // First, zero out the dense array
    if (idx < dense_size) {
        dense_data[idx] = 0.0f;
    }

    __syncthreads();

    // Then fill in the sparse values
    if (idx < num_nonzeros) {
        dense_data[sparse_indices[idx]] = sparse_values[idx];
    }
}

// Memory prefetching and caching
__global__ void prefetch_data_kernel(
    const float* __restrict__ data,
    int* __restrict__ access_pattern,
    int num_accesses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_accesses) {
        // Prefetch data into cache
        volatile float temp = data[access_pattern[idx]];
        (void)temp; // Suppress unused variable warning
    }
}

// Data layout transformation kernels
__global__ void aos_to_soa_float3_kernel(
    const float3* __restrict__ aos_data,
    float* __restrict__ soa_x,
    float* __restrict__ soa_y,
    float* __restrict__ soa_z,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        soa_x[idx] = aos_data[idx].x;
        soa_y[idx] = aos_data[idx].y;
        soa_z[idx] = aos_data[idx].z;
    }
}

__global__ void soa_to_aos_float3_kernel(
    const float* __restrict__ soa_x,
    const float* __restrict__ soa_y,
    const float* __restrict__ soa_z,
    float3* __restrict__ aos_data,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        aos_data[idx] = make_float3(soa_x[idx], soa_y[idx], soa_z[idx]);
    }
}

// Memory bandwidth testing
__global__ void memory_bandwidth_test_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int size,
    int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = idx; i < size; i += stride) {
            dst[i] = src[i] * 1.1f; // Simple operation to avoid compiler optimization
        }
    }
}

// Reduction operations
__global__ void reduce_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < size) ? input[idx] : -FLT_MAX;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_min_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < size) ? input[idx] : FLT_MAX;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Memory pattern analysis
__global__ void analyze_access_pattern_kernel(
    const int* __restrict__ access_indices,
    int* __restrict__ cache_hits,
    int* __restrict__ cache_misses,
    int cache_line_size,
    int num_accesses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_accesses - 1) return;

    int current_line = access_indices[idx] / cache_line_size;
    int next_line = access_indices[idx + 1] / cache_line_size;

    if (current_line == next_line) {
        atomicAdd(cache_hits, 1);
    } else {
        atomicAdd(cache_misses, 1);
    }
}

// Memory alignment optimization
__global__ void check_alignment_kernel(
    const void* __restrict__ ptr,
    int* __restrict__ alignment_info,
    int num_pointers
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pointers) return;

    uintptr_t addr = (uintptr_t)ptr + idx * sizeof(float);

    // Check alignment for different sizes
    alignment_info[idx * 4 + 0] = (addr % 4 == 0) ? 1 : 0;   // 4-byte aligned
    alignment_info[idx * 4 + 1] = (addr % 8 == 0) ? 1 : 0;   // 8-byte aligned
    alignment_info[idx * 4 + 2] = (addr % 16 == 0) ? 1 : 0;  // 16-byte aligned
    alignment_info[idx * 4 + 3] = (addr % 32 == 0) ? 1 : 0;  // 32-byte aligned
}

} // namespace physgrad