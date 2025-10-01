#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#else
// Mock CUDA types for CPU compilation
struct float3 { float x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int x=1, unsigned int y=1, unsigned int z=1) : x(x), y(y), z(z) {} };
enum cudaMemcpyKind { cudaMemcpyDeviceToDevice = 0 };
using cudaError_t = int;
using cudaStream_t = void*;
#endif

#include <vector>
#include <memory>

namespace physgrad {

// Memory access patterns for optimization
enum class MemoryAccessPattern {
    COALESCED_SEQUENTIAL,    // Perfect coalesced access
    STRIDED_REGULAR,         // Regular stride access
    RANDOM_SCATTERED,        // Scattered random access
    BLOCK_TILED,            // Block-wise tiled access
    WARP_COOPERATIVE,       // Warp-level cooperative access
    SHARED_MEMORY_CACHED    // Shared memory optimization
};

// Memory layout strategies
enum class MemoryLayout {
    AOS,                    // Array of Structures
    SOA,                    // Structure of Arrays
    AOSOA,                  // Array of Structures of Arrays (hybrid)
    BLOCKED,                // Block-structured layout
    MORTON_ORDERED          // Morton (Z-order) curve layout
};

// Cache optimization hints
struct CacheOptimizationHints {
    bool use_l1_cache = true;
    bool use_l2_cache = true;
    bool prefer_shared_memory = false;
    bool use_texture_cache = false;
    int prefetch_distance = 0;
    bool vectorized_loads = true;
};

// Memory coalescing analyzer
class MemoryCoalescingAnalyzer {
public:
    struct AccessAnalysis {
        float coalescing_efficiency;
        int cache_line_utilization;
        int memory_bank_conflicts;
        int warp_divergence_factor;
        size_t memory_throughput_bytes_per_sec;
    };

    static AccessAnalysis analyzeKernelAccess(
        const void* data_ptr,
        size_t element_size,
        size_t num_elements,
        MemoryAccessPattern pattern
    );
};

#ifdef __CUDACC__
// Optimized memory access primitives
namespace optimized_access {

// Vectorized memory operations
template<typename T, int VecSize>
__device__ __forceinline__ void vectorized_load(
    T* dst, const T* __restrict__ src, int num_elements
);

template<typename T, int VecSize>
__device__ __forceinline__ void vectorized_store(
    T* __restrict__ dst, const T* src, int num_elements
);

// Coalesced transpose operations
template<int TILE_DIM, int BLOCK_ROWS>
__global__ void coalesced_transpose_kernel(
    float* __restrict__ odata,
    const float* __restrict__ idata,
    int width, int height
);

// Shared memory optimization
template<int SHARED_SIZE>
__device__ __forceinline__ void shared_memory_load_store(
    float* __restrict__ global_out,
    const float* __restrict__ global_in,
    int thread_id, int num_threads
);

} // namespace optimized_access
#endif // __CUDACC__

// AoSoA (Array of Structures of Arrays) data structure
template<typename T, int VECTOR_SIZE = 4>
class AoSoAContainer {
private:
    static constexpr int CHUNK_SIZE = 64; // Cache line size
    static constexpr int VECTORS_PER_CHUNK = CHUNK_SIZE / (sizeof(T) * VECTOR_SIZE);

    T* data_;
    size_t num_elements_;
    size_t capacity_;

public:
    AoSoAContainer(size_t initial_capacity);
    ~AoSoAContainer();

    // Memory-optimized access
#ifdef __CUDACC__
    __device__ __host__ T* getVectorPtr(size_t element_id, int vector_component);
    __device__ __host__ const T* getVectorPtr(size_t element_id, int vector_component) const;
#else
    T* getVectorPtr(size_t element_id, int vector_component);
    const T* getVectorPtr(size_t element_id, int vector_component) const;
#endif

    // Bulk operations with optimal memory patterns
    void copyToDevice(const std::vector<T>& host_data);
    void copyFromDevice(std::vector<T>& host_data) const;

    // Memory layout information
    size_t getMemoryFootprint() const;
    MemoryLayout getLayout() const { return MemoryLayout::AOSOA; }
};

// Morton order (Z-order curve) spatial locality optimizer
class MortonOrderOptimizer {
public:
    // Convert 3D coordinates to Morton order index
#ifdef __CUDACC__
    __device__ __host__ static uint64_t encode3D(uint32_t x, uint32_t y, uint32_t z);
    // Convert Morton order index back to 3D coordinates
    __device__ __host__ static void decode3D(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z);
#else
    static uint64_t encode3D(uint32_t x, uint32_t y, uint32_t z);
    static void decode3D(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z);
#endif

    // Reorder particle data for better spatial locality
    template<typename ParticleData>
    static void reorderParticles(
        ParticleData* particles,
        size_t num_particles,
        float3 domain_min,
        float3 domain_max,
        int resolution_bits = 10
    );

    // Concrete implementation for float3 particles
    static void reorderParticlesFloat3(
        float3* particles,
        size_t num_particles,
        float3 domain_min,
        float3 domain_max,
        int resolution_bits = 10
    );
};

// High-performance memory operations
class OptimizedMemoryOps {
public:
    // Streaming memory copy with prefetching
    static cudaError_t streamingMemcpy(
        void* dst, const void* src, size_t size,
        cudaMemcpyKind kind = cudaMemcpyDeviceToDevice,
        cudaStream_t stream = 0
    );

    // Asynchronous memory operations with pipeline optimization
    static cudaError_t pipelinedMemcpy(
        void* dst, const void* src, size_t size,
        int pipeline_stages = 4,
        cudaStream_t* streams = nullptr
    );

    // Zero-copy memory mapping for large datasets
    static cudaError_t setupZeroCopyMapping(
        void** host_ptr, void** device_ptr, size_t size
    );
};

#ifdef __CUDACC__
// Warp-level memory optimization primitives
namespace warp_optimized {

// Warp-cooperative memory loads
template<typename T>
__device__ __forceinline__ T warp_coalesced_load(
    const T* __restrict__ addr, int lane_id
);

// Warp shuffle-based data distribution
template<typename T>
__device__ __forceinline__ T warp_shuffle_distribute(
    T value, int src_lane, int width = 32
);

// Warp-level reduction with memory optimization
template<typename T, int WARP_SIZE = 32>
__device__ __forceinline__ T warp_reduce_sum(T value);

// Warp-level scan (prefix sum) operations
template<typename T>
__device__ __forceinline__ T warp_scan_inclusive(T value);

} // namespace warp_optimized
#endif // __CUDACC__

// Memory bandwidth benchmark and optimization
class MemoryBandwidthOptimizer {
public:
    struct BandwidthResult {
        float achieved_bandwidth_gb_s;
        float theoretical_bandwidth_gb_s;
        float efficiency_percentage;
        int optimal_block_size;
        int optimal_grid_size;
    };

    // Benchmark different access patterns
    static BandwidthResult benchmarkAccessPattern(
        size_t data_size,
        MemoryAccessPattern pattern,
        int iterations = 100
    );

    // Find optimal kernel launch parameters
    static dim3 findOptimalLaunchParams(
        size_t data_size,
        size_t shared_memory_per_block = 0
    );

    // Auto-tune memory access for specific kernel
    template<typename KernelFunc>
    static void autoTuneKernel(
        KernelFunc kernel,
        size_t data_size,
        int min_block_size = 32,
        int max_block_size = 1024
    );
};

// Cache-aware data structures
template<typename T>
class CacheOptimizedArray {
private:
    T* data_;
    size_t size_;
    size_t cache_line_size_;
    bool use_padding_;

public:
    CacheOptimizedArray(size_t size, bool enable_padding = true);
    ~CacheOptimizedArray();

    // Cache-aligned access
#ifdef __CUDACC__
    __device__ __host__ T& operator[](size_t index);
    __device__ __host__ const T& operator[](size_t index) const;
    // Prefetch operations
    __device__ void prefetch(size_t index, int cache_level = 1);
#else
    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    void prefetch(size_t index, int cache_level = 1);
#endif

    // Memory pattern hints
    void setAccessPattern(MemoryAccessPattern pattern);
};

// CUDA stream optimization for memory operations
class StreamOptimizer {
private:
    std::vector<cudaStream_t> streams_;
    int num_streams_;

public:
    StreamOptimizer(int num_streams = 4);
    ~StreamOptimizer();

    // Overlapped memory transfers and computation
    cudaError_t overlappedMemcpyAndKernel(
        void* dst, const void* src, size_t size,
        void (*kernel)(void*), void* kernel_args,
        int stream_id = 0
    );

    // Multi-stream memory operations
    cudaError_t multiStreamMemcpy(
        void* dst, const void* src, size_t size,
        int num_chunks = 4
    );

    // Stream synchronization optimization
    void optimizedStreamSync();
};

} // namespace physgrad