#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <thread>
#include <cstring>

namespace physgrad {

// Forward declarations
struct MemoryBlock;
class MemoryPool;
class MemoryProfiler;

// Memory allocation strategies
enum class AllocationStrategy {
    IMMEDIATE,           // Allocate immediately when requested
    LAZY,               // Allocate only when actually used
    PREALLOC,           // Pre-allocate based on predictions
    ADAPTIVE            // Adapt based on usage patterns
};

// Memory access patterns
enum class AccessPattern {
    SEQUENTIAL,         // Sequential access (good for streaming)
    RANDOM,            // Random access (cache-friendly blocks)
    STREAMING,         // One-time streaming access
    PERSISTENT         // Long-term persistent data
};

// Memory tier for hierarchical storage
enum class MemoryTier {
    GPU_DEVICE,        // GPU device memory (fastest, most expensive)
    GPU_UNIFIED,       // CUDA unified memory (automatic migration)
    CPU_PINNED,        // CPU pinned memory (fast GPU access)
    CPU_PAGED,         // CPU pageable memory (slowest, cheapest)
    STORAGE_CACHE      // Storage-backed cache (temporary spill)
};

// Memory block descriptor
struct MemoryBlock {
    void* ptr;
    size_t size;
    size_t alignment;
    MemoryTier tier;
    AccessPattern pattern;
    int ref_count;
    bool is_free;
    std::chrono::steady_clock::time_point last_access;
    std::chrono::steady_clock::time_point allocated_time;

    // Usage statistics
    uint64_t access_count;
    uint64_t bytes_transferred;

    MemoryBlock() : ptr(nullptr), size(0), alignment(16), tier(MemoryTier::GPU_DEVICE),
                   pattern(AccessPattern::RANDOM), ref_count(0), is_free(true),
                   access_count(0), bytes_transferred(0) {}
};

// Memory statistics
struct MemoryStats {
    size_t total_allocated;
    size_t total_free;
    size_t peak_usage;
    size_t current_usage;

    // Per-tier statistics
    size_t gpu_device_usage;
    size_t gpu_unified_usage;
    size_t cpu_pinned_usage;
    size_t cpu_paged_usage;
    size_t storage_cache_usage;

    // Performance metrics
    double allocation_time_ms;
    double deallocation_time_ms;
    double migration_time_ms;
    uint64_t cache_hits;
    uint64_t cache_misses;

    // Fragmentation metrics
    double fragmentation_ratio;
    size_t largest_free_block;
    size_t num_free_blocks;
};

// Memory pool for specific allocation patterns
class MemoryPool {
private:
    MemoryTier tier_;
    size_t block_size_;
    size_t max_blocks_;
    AllocationStrategy strategy_;

    std::vector<MemoryBlock> blocks_;
    std::vector<size_t> free_blocks_;
    std::mutex pool_mutex_;

    // Statistics
    size_t total_allocated_;
    size_t peak_usage_;
    uint64_t allocation_count_;
    uint64_t deallocation_count_;

public:
    MemoryPool(MemoryTier tier, size_t block_size, size_t max_blocks,
               AllocationStrategy strategy = AllocationStrategy::IMMEDIATE);
    ~MemoryPool();

    // Pool management
    bool initialize();
    void cleanup();

    // Allocation interface
    MemoryBlock* allocate(size_t size, AccessPattern pattern = AccessPattern::RANDOM);
    bool deallocate(MemoryBlock* block);

    // Pool optimization
    void defragment();
    void preAllocate(size_t num_blocks);
    void trim(size_t target_size);

    // Statistics
    MemoryStats getStats() const;
    double getFragmentation() const;
    size_t getAvailableMemory() const;

    // Configuration
    void setAllocationStrategy(AllocationStrategy strategy);
    void setMaxBlocks(size_t max_blocks);
};

// Hierarchical memory manager for massive-scale simulations
class MassiveMemoryManager {
private:
    // Memory pools for different tiers
    std::unique_ptr<MemoryPool> gpu_device_pool_;
    std::unique_ptr<MemoryPool> gpu_unified_pool_;
    std::unique_ptr<MemoryPool> cpu_pinned_pool_;
    std::unique_ptr<MemoryPool> cpu_paged_pool_;
    std::unique_ptr<MemoryPool> storage_cache_pool_;

    // Memory allocation tracking
    std::unordered_map<void*, MemoryBlock*> active_allocations_;
    std::mutex allocation_mutex_;

    // Automatic memory migration
    bool auto_migration_enabled_;
    size_t migration_threshold_;
    std::chrono::milliseconds migration_interval_;

    // Memory pressure management
    size_t memory_pressure_threshold_;
    bool emergency_mode_;

    // Configuration
    size_t max_gpu_memory_;
    size_t max_cpu_memory_;
    size_t max_storage_cache_;
    AllocationStrategy default_strategy_;

    // Storage-backed cache configuration
    std::string storage_cache_directory_;
    bool storage_spill_enabled_;
    std::unordered_map<void*, std::string> storage_cache_files_;
    std::mutex storage_cache_mutex_;
    size_t storage_cache_file_counter_;

    // Migration thread management
    std::atomic<bool> migration_thread_running_;
    std::thread migration_thread_;

    // Performance monitoring
    std::unique_ptr<MemoryProfiler> profiler_;

    // Statistics
    mutable std::mutex stats_mutex_;
    MemoryStats global_stats_;

public:
    MassiveMemoryManager();
    ~MassiveMemoryManager();

    // Initialization
    bool initialize(size_t max_gpu_memory = 0,     // 0 = auto-detect
                   size_t max_cpu_memory = 0,      // 0 = auto-detect
                   size_t max_storage_cache = 0);  // 0 = auto-detect
    void cleanup();

    // Memory allocation interface
    void* allocate(size_t size,
                  MemoryTier preferred_tier = MemoryTier::GPU_DEVICE,
                  AccessPattern pattern = AccessPattern::RANDOM,
                  size_t alignment = 256);
    bool deallocate(void* ptr);
    bool reallocate(void** ptr, size_t old_size, size_t new_size);

    // Smart allocation based on usage patterns
    void* allocateArray(size_t count, size_t element_size,
                       AccessPattern pattern = AccessPattern::SEQUENTIAL);
    void* allocateMatrix(size_t rows, size_t cols, size_t element_size,
                        AccessPattern pattern = AccessPattern::RANDOM);
    void* allocateTemporary(size_t size,
                           std::chrono::milliseconds lifetime = std::chrono::milliseconds(1000));

    // Memory migration and optimization
    bool migrateMemory(void* ptr, MemoryTier target_tier);
    void optimizeMemoryLayout();
    void defragmentAll();

    // Massive-scale features
    bool enableMemoryOversubscription(float ratio = 2.0f);
    bool enableAutomaticMigration(bool enable = true);
    void setMemoryPressureThreshold(float threshold = 0.85f);

    // Memory hints and prefetching
    void hintSequentialAccess(void* ptr, size_t size);
    void hintRandomAccess(void* ptr, size_t size);
    void prefetchToGPU(void* ptr, size_t size);
    void prefetchToCPU(void* ptr, size_t size);

    // Statistics and monitoring
    MemoryStats getGlobalStats() const;
    MemoryStats getTierStats(MemoryTier tier) const;
    void printMemoryReport() const;

    // Configuration
    void setDefaultAllocationStrategy(AllocationStrategy strategy);

    // Emergency management
    bool handleMemoryPressure();
    void enterEmergencyMode();
    void exitEmergencyMode();

    // Advanced features for billion-particle simulations
    bool enableStorageSpill(const std::string& cache_directory);
    void optimizeForParticleSimulation(size_t particle_count, size_t particle_size);
    void configureCachePolicy(MemoryTier tier, float cache_ratio);

private:
    // Internal memory management
    MemoryBlock* allocateFromTier(size_t size, MemoryTier tier,
                                 AccessPattern pattern, size_t alignment);
    bool deallocateFromTier(MemoryBlock* block);

    // Memory migration internals
    bool migrateBlock(MemoryBlock* block, MemoryTier target_tier);
    void backgroundMigrationThread();

    // Memory pressure handling
    void monitorMemoryPressure();
    bool spillToStorage(MemoryBlock* block);
    bool reclaimFromStorage(MemoryBlock* block);

    // Statistics updates
    void updateGlobalStats();
    void recordAllocation(MemoryBlock* block);
    void recordDeallocation(MemoryBlock* block);

    // Utility functions
    size_t alignSize(size_t size, size_t alignment);
    MemoryTier selectOptimalTier(size_t size, AccessPattern pattern);
    bool isMemoryPressureHigh() const;
    void logMemoryOperation(const std::string& operation, MemoryBlock* block);

    // Pool size management
    void setPoolSizes(MemoryTier tier, size_t max_size);
};

// Memory profiler for performance optimization
class MemoryProfiler {
private:
    struct AllocationTrace {
        std::chrono::steady_clock::time_point timestamp;
        size_t size;
        MemoryTier tier;
        AccessPattern pattern;
        void* ptr;
        bool is_allocation; // true for alloc, false for dealloc
    };

    std::vector<AllocationTrace> trace_log_;
    std::mutex trace_mutex_;

    // Performance counters
    std::atomic<uint64_t> total_allocations_;
    std::atomic<uint64_t> total_deallocations_;
    std::atomic<uint64_t> failed_allocations_;
    std::atomic<uint64_t> memory_migrations_;

    // Timing statistics
    std::atomic<double> avg_allocation_time_;
    std::atomic<double> avg_deallocation_time_;
    std::atomic<double> avg_migration_time_;

    bool profiling_enabled_;

public:
    MemoryProfiler();

    // Profiling control
    void enable();
    void disable();
    void reset();

    // Event recording
    void recordAllocation(void* ptr, size_t size, MemoryTier tier,
                         AccessPattern pattern, double time_ms);
    void recordDeallocation(void* ptr, double time_ms);
    void recordMigration(void* ptr, MemoryTier from_tier, MemoryTier to_tier, double time_ms);
    void recordFailedAllocation(size_t size, MemoryTier tier);

    // Analysis
    std::vector<AllocationTrace> getTrace() const;
    void analyzeAllocationPatterns();
    void generateOptimizationReport();

    // Statistics
    uint64_t getTotalAllocations() const { return total_allocations_; }
    uint64_t getTotalDeallocations() const { return total_deallocations_; }
    uint64_t getFailedAllocations() const { return failed_allocations_; }
    double getAverageAllocationTime() const { return avg_allocation_time_; }
};

// Specialized allocators for common physics patterns
class ParticleMemoryAllocator {
private:
    MassiveMemoryManager* memory_manager_;

    // Particle-specific pools
    void* position_buffer_;
    void* velocity_buffer_;
    void* force_buffer_;
    void* property_buffer_;

    size_t max_particles_;
    size_t current_particles_;
    size_t particle_size_;

public:
    ParticleMemoryAllocator(MassiveMemoryManager* manager);
    ~ParticleMemoryAllocator();

    bool initialize(size_t max_particles, size_t particle_size);
    void cleanup();

    // Particle array management
    bool resizeParticleArrays(size_t new_count);
    void* getPositionBuffer() const { return position_buffer_; }
    void* getVelocityBuffer() const { return velocity_buffer_; }
    void* getForceBuffer() const { return force_buffer_; }
    void* getPropertyBuffer() const { return property_buffer_; }

    // Dynamic particle management
    bool addParticles(size_t count);
    bool removeParticles(size_t count);
    void compactArrays();

    size_t getCurrentParticleCount() const { return current_particles_; }
    size_t getMaxParticleCount() const { return max_particles_; }
};

// Grid-based memory allocator for spatial data structures
class SpatialMemoryAllocator {
private:
    MassiveMemoryManager* memory_manager_;

    // Grid configuration
    int3 grid_dimensions_;
    float3 cell_size_;
    float3 domain_min_;
    float3 domain_max_;

    // Memory layout
    void* cell_data_;
    void* neighbor_lists_;
    void* spatial_hash_;

    size_t total_cells_;
    size_t max_particles_per_cell_;

public:
    SpatialMemoryAllocator(MassiveMemoryManager* manager);
    ~SpatialMemoryAllocator();

    bool initialize(const int3& grid_dims, const float3& domain_min,
                   const float3& domain_max, size_t max_particles_per_cell);
    void cleanup();

    // Grid memory access
    void* getCellData(int x, int y, int z);
    void* getNeighborList(int cell_id);
    void* getSpatialHash() const { return spatial_hash_; }

    // Grid operations
    bool resizeGrid(const int3& new_dims);
    void clearGrid();
    void optimizeLayout();

    // Statistics
    size_t getTotalCells() const { return total_cells_; }
    size_t getMemoryUsage() const;
};

} // namespace physgrad