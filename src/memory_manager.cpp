#include "memory_manager.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif
#include <iostream>
#include <algorithm>
#include <numeric>
#include <thread>
#include <fstream>
#include <filesystem>

namespace physgrad {

// MemoryPool Implementation
MemoryPool::MemoryPool(MemoryTier tier, size_t block_size, size_t max_blocks,
                      AllocationStrategy strategy) :
    tier_(tier), block_size_(block_size), max_blocks_(max_blocks), strategy_(strategy),
    total_allocated_(0), peak_usage_(0), allocation_count_(0), deallocation_count_(0) {
    blocks_.reserve(max_blocks);
    free_blocks_.reserve(max_blocks);
}

MemoryPool::~MemoryPool() {
    cleanup();
}

bool MemoryPool::initialize() {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    // Pre-allocate blocks based on strategy
    if (strategy_ == AllocationStrategy::PREALLOC) {
        preAllocate(max_blocks_ / 4); // Pre-allocate 25%
        return true;
    }

    return true;
}

void MemoryPool::cleanup() {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    for (auto& block : blocks_) {
        if (block.ptr) {
            switch (tier_) {
                case MemoryTier::GPU_DEVICE:
#ifdef HAVE_CUDA
                    cudaFree(block.ptr);
#else
                    free(block.ptr);
#endif
                    break;
                case MemoryTier::GPU_UNIFIED:
#ifdef HAVE_CUDA
                    cudaFree(block.ptr);
#else
                    free(block.ptr);
#endif
                    break;
                case MemoryTier::CPU_PINNED:
#ifdef HAVE_CUDA
                    cudaFreeHost(block.ptr);
#else
                    free(block.ptr);
#endif
                    break;
                case MemoryTier::CPU_PAGED:
                    free(block.ptr);
                    break;
                case MemoryTier::STORAGE_CACHE:
                    free(block.ptr);
                    break;
            }
        }
    }

    blocks_.clear();
    free_blocks_.clear();
}

MemoryBlock* MemoryPool::allocate(size_t size, AccessPattern pattern) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    // Align size to block size
    size_t aligned_size = ((size + block_size_ - 1) / block_size_) * block_size_;

    // Look for available free block
    for (size_t i = 0; i < free_blocks_.size(); ++i) {
        size_t block_idx = free_blocks_[i];
        MemoryBlock& block = blocks_[block_idx];

        if (block.size >= aligned_size) {
            // Remove from free list
            free_blocks_.erase(free_blocks_.begin() + i);

            // Update block info
            block.is_free = false;
            block.ref_count = 1;
            block.pattern = pattern;
            block.last_access = std::chrono::steady_clock::now();
            block.allocated_time = block.last_access;

            allocation_count_++;
            total_allocated_ += block.size;
            peak_usage_ = std::max(peak_usage_, total_allocated_);

            return &block;
        }
    }

    // No suitable free block found, allocate new one
    if (blocks_.size() >= max_blocks_) {
        return nullptr; // Pool exhausted
    }

    MemoryBlock new_block;
    new_block.size = std::max(aligned_size, block_size_);
    new_block.tier = tier_;
    new_block.pattern = pattern;
    new_block.is_free = false;
    new_block.ref_count = 1;
    new_block.last_access = std::chrono::steady_clock::now();
    new_block.allocated_time = new_block.last_access;

    // Allocate memory based on tier
    bool allocation_success = false;
    switch (tier_) {
        case MemoryTier::GPU_DEVICE:
#ifdef HAVE_CUDA
            allocation_success = (cudaMalloc(&new_block.ptr, new_block.size) == cudaSuccess);
#else
            new_block.ptr = malloc(new_block.size);
            allocation_success = (new_block.ptr != nullptr);
#endif
            break;
        case MemoryTier::GPU_UNIFIED:
#ifdef HAVE_CUDA
            allocation_success = (cudaMallocManaged(&new_block.ptr, new_block.size) == cudaSuccess);
#else
            new_block.ptr = malloc(new_block.size);
            allocation_success = (new_block.ptr != nullptr);
#endif
            break;
        case MemoryTier::CPU_PINNED:
#ifdef HAVE_CUDA
            allocation_success = (cudaMallocHost(&new_block.ptr, new_block.size) == cudaSuccess);
#else
            new_block.ptr = malloc(new_block.size);
            allocation_success = (new_block.ptr != nullptr);
#endif
            break;
        case MemoryTier::CPU_PAGED:
            new_block.ptr = aligned_alloc(new_block.alignment, new_block.size);
            allocation_success = (new_block.ptr != nullptr);
            break;
        case MemoryTier::STORAGE_CACHE:
            new_block.ptr = aligned_alloc(new_block.alignment, new_block.size);
            allocation_success = (new_block.ptr != nullptr);
            break;
    }

    if (!allocation_success) {
        return nullptr;
    }

    blocks_.push_back(new_block);
    allocation_count_++;
    total_allocated_ += new_block.size;
    peak_usage_ = std::max(peak_usage_, total_allocated_);

    return &blocks_.back();
}

bool MemoryPool::deallocate(MemoryBlock* block) {
    if (!block) return false;

    std::lock_guard<std::mutex> lock(pool_mutex_);

    // Find block in our pool
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
        [block](const MemoryBlock& b) { return &b == block; });

    if (it == blocks_.end()) {
        return false; // Block not from this pool
    }

    // Mark as free
    block->is_free = true;
    block->ref_count = 0;

    // Add to free list
    size_t block_idx = std::distance(blocks_.begin(), it);
    free_blocks_.push_back(block_idx);

    deallocation_count_++;
    total_allocated_ -= block->size;

    return true;
}

void MemoryPool::defragment() {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    // Sort free blocks by size for better allocation
    std::sort(free_blocks_.begin(), free_blocks_.end(),
        [this](size_t a, size_t b) {
            return blocks_[a].size < blocks_[b].size;
        });
}

void MemoryPool::preAllocate(size_t num_blocks) {
    for (size_t i = 0; i < num_blocks && blocks_.size() < max_blocks_; ++i) {
        MemoryBlock block;
        block.size = block_size_;
        block.tier = tier_;
        block.is_free = true;
        block.ref_count = 0;

        // Allocate memory
        bool success = false;
        switch (tier_) {
            case MemoryTier::GPU_DEVICE:
#ifdef HAVE_CUDA
                success = (cudaMalloc(&block.ptr, block.size) == cudaSuccess);
#else
                block.ptr = malloc(block.size);
                success = (block.ptr != nullptr);
#endif
                break;
            case MemoryTier::GPU_UNIFIED:
#ifdef HAVE_CUDA
                success = (cudaMallocManaged(&block.ptr, block.size) == cudaSuccess);
#else
                block.ptr = malloc(block.size);
                success = (block.ptr != nullptr);
#endif
                break;
            case MemoryTier::CPU_PINNED:
#ifdef HAVE_CUDA
                success = (cudaMallocHost(&block.ptr, block.size) == cudaSuccess);
#else
                block.ptr = malloc(block.size);
                success = (block.ptr != nullptr);
#endif
                break;
            case MemoryTier::CPU_PAGED:
                block.ptr = aligned_alloc(block.alignment, block.size);
                success = (block.ptr != nullptr);
                break;
            case MemoryTier::STORAGE_CACHE:
                block.ptr = aligned_alloc(block.alignment, block.size);
                success = (block.ptr != nullptr);
                break;
        }

        if (success) {
            blocks_.push_back(block);
            free_blocks_.push_back(blocks_.size() - 1);
        } else {
            // Failed to allocate this block, continue with others
        }
    }

    // Completed pre-allocation attempt
}

MemoryStats MemoryPool::getStats() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(pool_mutex_));

    MemoryStats stats = {};
    stats.total_allocated = total_allocated_;
    stats.peak_usage = peak_usage_;
    stats.current_usage = total_allocated_;

    size_t free_memory = 0;
    for (size_t idx : free_blocks_) {
        free_memory += blocks_[idx].size;
    }
    stats.total_free = free_memory;

    // Calculate fragmentation
    if (!free_blocks_.empty()) {
        size_t largest_free = 0;
        for (size_t idx : free_blocks_) {
            largest_free = std::max(largest_free, blocks_[idx].size);
        }
        stats.largest_free_block = largest_free;
        stats.num_free_blocks = free_blocks_.size();
        stats.fragmentation_ratio = 1.0 - (double)largest_free / free_memory;
    }

    return stats;
}

void MemoryPool::setAllocationStrategy(AllocationStrategy strategy) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    strategy_ = strategy;
}

void MemoryPool::setMaxBlocks(size_t max_blocks) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    max_blocks_ = max_blocks;

    // If we have more blocks than the new limit, we should consider cleanup
    // For now, just update the limit - cleanup can be done separately
}

double MemoryPool::getFragmentation() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(pool_mutex_));

    if (free_blocks_.empty()) {
        return 0.0; // No fragmentation if no free blocks
    }

    size_t total_free = 0;
    size_t largest_free = 0;

    for (size_t idx : free_blocks_) {
        size_t block_size = blocks_[idx].size;
        total_free += block_size;
        largest_free = std::max(largest_free, block_size);
    }

    if (total_free == 0) {
        return 0.0;
    }

    return 1.0 - (double)largest_free / total_free;
}

size_t MemoryPool::getAvailableMemory() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(pool_mutex_));

    size_t available = 0;
    for (size_t idx : free_blocks_) {
        available += blocks_[idx].size;
    }

    return available;
}

void MemoryPool::trim(size_t target_size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    // Remove free blocks until we reach target size
    while (total_allocated_ > target_size && !free_blocks_.empty()) {
        size_t block_idx = free_blocks_.back();
        MemoryBlock& block = blocks_[block_idx];

        // Free the actual memory
        switch (tier_) {
            case MemoryTier::GPU_DEVICE:
            case MemoryTier::GPU_UNIFIED:
#ifdef HAVE_CUDA
                cudaFree(block.ptr);
#else
                free(block.ptr);
#endif
                break;
            case MemoryTier::CPU_PINNED:
#ifdef HAVE_CUDA
                cudaFreeHost(block.ptr);
#else
                free(block.ptr);
#endif
                break;
            case MemoryTier::CPU_PAGED:
            case MemoryTier::STORAGE_CACHE:
                free(block.ptr);
                break;
        }

        total_allocated_ -= block.size;

        // Remove from blocks and free list
        free_blocks_.pop_back();
        blocks_.erase(blocks_.begin() + block_idx);

        // Update free block indices
        for (size_t& idx : free_blocks_) {
            if (idx > block_idx) {
                idx--;
            }
        }
    }
}

// MassiveMemoryManager Implementation
MassiveMemoryManager::MassiveMemoryManager() :
    auto_migration_enabled_(false), migration_threshold_(0),
    migration_interval_(std::chrono::milliseconds(1000)),
    memory_pressure_threshold_(0), emergency_mode_(false),
    max_gpu_memory_(0), max_cpu_memory_(0), max_storage_cache_(0),
    default_strategy_(AllocationStrategy::ADAPTIVE),
    storage_spill_enabled_(false), storage_cache_file_counter_(0),
    migration_thread_running_(false) {

    // Initialize memory pools
    gpu_device_pool_ = std::make_unique<MemoryPool>(
        MemoryTier::GPU_DEVICE, 1024 * 1024, 1000); // 1MB blocks
    gpu_unified_pool_ = std::make_unique<MemoryPool>(
        MemoryTier::GPU_UNIFIED, 1024 * 1024, 1000);
    cpu_pinned_pool_ = std::make_unique<MemoryPool>(
        MemoryTier::CPU_PINNED, 1024 * 1024, 2000);
    cpu_paged_pool_ = std::make_unique<MemoryPool>(
        MemoryTier::CPU_PAGED, 1024 * 1024, 5000);
    storage_cache_pool_ = std::make_unique<MemoryPool>(
        MemoryTier::STORAGE_CACHE, 1024 * 1024, 10000);

    profiler_ = std::make_unique<MemoryProfiler>();
}

MassiveMemoryManager::~MassiveMemoryManager() {
    cleanup();
}

bool MassiveMemoryManager::initialize(size_t max_gpu_memory, size_t max_cpu_memory, size_t max_storage_cache) {
    // Auto-detect memory sizes if not specified
    if (max_gpu_memory == 0) {
        size_t free_mem, total_mem;
#ifdef HAVE_CUDA
        cudaMemGetInfo(&free_mem, &total_mem);
#else
        // Estimate based on system memory
        total_mem = 4ULL * 1024 * 1024 * 1024; // Default 4GB
        free_mem = total_mem / 2;
#endif
        max_gpu_memory_ = total_mem * 0.9; // Use 90% of GPU memory
    } else {
        max_gpu_memory_ = max_gpu_memory;
    }

    if (max_cpu_memory == 0) {
        // Rough estimate of system memory
        max_cpu_memory_ = 32ULL * 1024 * 1024 * 1024; // 32GB default
    } else {
        max_cpu_memory_ = max_cpu_memory;
    }

    if (max_storage_cache == 0) {
        max_storage_cache_ = 100ULL * 1024 * 1024 * 1024; // 100GB default
    } else {
        max_storage_cache_ = max_storage_cache;
    }

    // Initialize all pools
    bool success = true;
    success &= gpu_device_pool_->initialize();
    success &= gpu_unified_pool_->initialize();
    success &= cpu_pinned_pool_->initialize();
    success &= cpu_paged_pool_->initialize();
    success &= storage_cache_pool_->initialize();

    if (success) {
        profiler_->enable();
        memory_pressure_threshold_ = max_gpu_memory_ * 0.85; // 85% threshold
    }

    return success;
}

void MassiveMemoryManager::cleanup() {
    // Stop migration thread
    migration_thread_running_ = false;
    if (migration_thread_.joinable()) {
        migration_thread_.join();
    }

    // Clean up storage cache files
    {
        std::lock_guard<std::mutex> lock(storage_cache_mutex_);
        for (const auto& file_entry : storage_cache_files_) {
            std::filesystem::remove(file_entry.second);
        }
        storage_cache_files_.clear();
    }

    if (gpu_device_pool_) gpu_device_pool_->cleanup();
    if (gpu_unified_pool_) gpu_unified_pool_->cleanup();
    if (cpu_pinned_pool_) cpu_pinned_pool_->cleanup();
    if (cpu_paged_pool_) cpu_paged_pool_->cleanup();
    if (storage_cache_pool_) storage_cache_pool_->cleanup();

    std::lock_guard<std::mutex> lock(allocation_mutex_);
    active_allocations_.clear();
}

void* MassiveMemoryManager::allocate(size_t size, MemoryTier preferred_tier,
                                   AccessPattern pattern, size_t alignment) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Select optimal tier if adaptive strategy
    MemoryTier target_tier = preferred_tier;
    if (default_strategy_ == AllocationStrategy::ADAPTIVE) {
        target_tier = selectOptimalTier(size, pattern);
    }

    // Try to allocate from preferred tier first
    MemoryBlock* block = allocateFromTier(size, target_tier, pattern, alignment);

    // If failed, try fallback tiers
    if (!block) {
        std::vector<MemoryTier> fallback_tiers = {
            MemoryTier::GPU_UNIFIED,
            MemoryTier::CPU_PINNED,
            MemoryTier::CPU_PAGED,
            MemoryTier::STORAGE_CACHE
        };

        for (MemoryTier tier : fallback_tiers) {
            if (tier != target_tier) {
                block = allocateFromTier(size, tier, pattern, alignment);
                if (block) break;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double alloc_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (block) {
        // Record allocation
        {
            std::lock_guard<std::mutex> lock(allocation_mutex_);
            active_allocations_[block->ptr] = block;
        }

        profiler_->recordAllocation(block->ptr, size, block->tier, pattern, alloc_time);
        recordAllocation(block);

        return block->ptr;
    } else {
        profiler_->recordFailedAllocation(size, target_tier);
        return nullptr;
    }
}

bool MassiveMemoryManager::deallocate(void* ptr) {
    if (!ptr) return false;

    auto start_time = std::chrono::high_resolution_clock::now();

    MemoryBlock* block = nullptr;
    {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        auto it = active_allocations_.find(ptr);
        if (it != active_allocations_.end()) {
            block = it->second;
            active_allocations_.erase(it);
        }
    }

    if (!block) return false;

    bool success = deallocateFromTier(block);

    auto end_time = std::chrono::high_resolution_clock::now();
    double dealloc_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    profiler_->recordDeallocation(ptr, dealloc_time);
    recordDeallocation(block);

    return success;
}

void* MassiveMemoryManager::allocateArray(size_t count, size_t element_size, AccessPattern pattern) {
    size_t total_size = count * element_size;

    // For large arrays, prefer unified memory or CPU pinned
    MemoryTier preferred_tier = MemoryTier::GPU_DEVICE;
    if (total_size > 1024 * 1024 * 1024) { // > 1GB
        preferred_tier = MemoryTier::GPU_UNIFIED;
    }

    return allocate(total_size, preferred_tier, pattern, element_size);
}

void* MassiveMemoryManager::allocateMatrix(size_t rows, size_t cols, size_t element_size, AccessPattern pattern) {
    size_t total_size = rows * cols * element_size;

    // Ensure proper alignment for matrix operations
    size_t alignment = std::max(element_size, size_t(256));

    return allocate(total_size, MemoryTier::GPU_DEVICE, pattern, alignment);
}

void* MassiveMemoryManager::allocateTemporary(size_t size, std::chrono::milliseconds lifetime) {
    // Temporary allocations prefer GPU unified memory for easy CPU/GPU access
    void* ptr = allocate(size, MemoryTier::GPU_UNIFIED, AccessPattern::STREAMING);

    // TODO: Implement automatic cleanup after lifetime expires
    // This would require a background thread managing temporary allocations

    return ptr;
}

bool MassiveMemoryManager::migrateMemory(void* ptr, MemoryTier target_tier) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    auto it = active_allocations_.find(ptr);
    if (it == active_allocations_.end()) {
        return false;
    }

    MemoryBlock* block = it->second;
    return migrateBlock(block, target_tier);
}

void MassiveMemoryManager::optimizeForParticleSimulation(size_t particle_count, size_t particle_size) {
    size_t total_memory = particle_count * particle_size;

    // Configure pool sizes based on particle simulation requirements
    if (total_memory > max_gpu_memory_) {
        // Enable memory oversubscription
        enableMemoryOversubscription(2.0f);
        enableAutomaticMigration(true);
    }

    // Pre-allocate particle buffers
    size_t position_buffer_size = particle_count * 3 * sizeof(float);
    size_t velocity_buffer_size = particle_count * 3 * sizeof(float);
    size_t force_buffer_size = particle_count * 3 * sizeof(float);

    // Allocate on GPU device memory for maximum performance
    allocate(position_buffer_size, MemoryTier::GPU_DEVICE, AccessPattern::SEQUENTIAL);
    allocate(velocity_buffer_size, MemoryTier::GPU_DEVICE, AccessPattern::SEQUENTIAL);
    allocate(force_buffer_size, MemoryTier::GPU_DEVICE, AccessPattern::SEQUENTIAL);
}

bool MassiveMemoryManager::enableMemoryOversubscription(float ratio) {
    // Allow memory pools to exceed physical memory by the specified ratio
    max_gpu_memory_ *= ratio;

    // Enable automatic migration to handle oversubscription
    enableAutomaticMigration(true);

    return true;
}

bool MassiveMemoryManager::enableAutomaticMigration(bool enable) {
    auto_migration_enabled_ = enable;

    if (enable && !migration_thread_running_) {
        // Start background migration thread
        migration_thread_running_ = true;
        migration_thread_ = std::thread(&MassiveMemoryManager::backgroundMigrationThread, this);
    } else if (!enable && migration_thread_running_) {
        // Stop background migration thread
        migration_thread_running_ = false;
        if (migration_thread_.joinable()) {
            migration_thread_.join();
        }
    }

    return true;
}

MemoryStats MassiveMemoryManager::getGlobalStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    MemoryStats stats = global_stats_;

    // Aggregate stats from all pools
    auto gpu_stats = gpu_device_pool_->getStats();
    auto unified_stats = gpu_unified_pool_->getStats();
    auto pinned_stats = cpu_pinned_pool_->getStats();
    auto paged_stats = cpu_paged_pool_->getStats();
    auto storage_stats = storage_cache_pool_->getStats();

    stats.gpu_device_usage = gpu_stats.current_usage;
    stats.gpu_unified_usage = unified_stats.current_usage;
    stats.cpu_pinned_usage = pinned_stats.current_usage;
    stats.cpu_paged_usage = paged_stats.current_usage;
    stats.storage_cache_usage = storage_stats.current_usage;

    stats.total_allocated = stats.gpu_device_usage + stats.gpu_unified_usage +
                           stats.cpu_pinned_usage + stats.cpu_paged_usage +
                           stats.storage_cache_usage;

    return stats;
}

void MassiveMemoryManager::printMemoryReport() const {
    auto stats = getGlobalStats();

    std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              PhysGrad Memory Management Report          ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║ Memory Usage by Tier:                                   ║" << std::endl;
    std::cout << "║   GPU Device:    " << std::setw(10) << (stats.gpu_device_usage / (1024*1024)) << " MB ║" << std::endl;
    std::cout << "║   GPU Unified:   " << std::setw(10) << (stats.gpu_unified_usage / (1024*1024)) << " MB ║" << std::endl;
    std::cout << "║   CPU Pinned:    " << std::setw(10) << (stats.cpu_pinned_usage / (1024*1024)) << " MB ║" << std::endl;
    std::cout << "║   CPU Paged:     " << std::setw(10) << (stats.cpu_paged_usage / (1024*1024)) << " MB ║" << std::endl;
    std::cout << "║   Storage Cache: " << std::setw(10) << (stats.storage_cache_usage / (1024*1024)) << " MB ║" << std::endl;
    std::cout << "║                                                          ║" << std::endl;
    std::cout << "║ Performance Metrics:                                    ║" << std::endl;
    std::cout << "║   Total Allocated: " << std::setw(8) << (stats.total_allocated / (1024*1024)) << " MB       ║" << std::endl;
    std::cout << "║   Peak Usage:      " << std::setw(8) << (stats.peak_usage / (1024*1024)) << " MB       ║" << std::endl;
    std::cout << "║   Cache Hit Rate:  " << std::setw(8) << std::fixed << std::setprecision(1)
              << (100.0 * stats.cache_hits / (stats.cache_hits + stats.cache_misses)) << "%        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
}

// Private methods
MemoryBlock* MassiveMemoryManager::allocateFromTier(size_t size, MemoryTier tier,
                                                   AccessPattern pattern, size_t alignment) {
    switch (tier) {
        case MemoryTier::GPU_DEVICE:
            return gpu_device_pool_->allocate(size, pattern);
        case MemoryTier::GPU_UNIFIED:
            return gpu_unified_pool_->allocate(size, pattern);
        case MemoryTier::CPU_PINNED:
            return cpu_pinned_pool_->allocate(size, pattern);
        case MemoryTier::CPU_PAGED:
            return cpu_paged_pool_->allocate(size, pattern);
        case MemoryTier::STORAGE_CACHE:
            return storage_cache_pool_->allocate(size, pattern);
        default:
            return nullptr;
    }
}

bool MassiveMemoryManager::deallocateFromTier(MemoryBlock* block) {
    switch (block->tier) {
        case MemoryTier::GPU_DEVICE:
            return gpu_device_pool_->deallocate(block);
        case MemoryTier::GPU_UNIFIED:
            return gpu_unified_pool_->deallocate(block);
        case MemoryTier::CPU_PINNED:
            return cpu_pinned_pool_->deallocate(block);
        case MemoryTier::CPU_PAGED:
            return cpu_paged_pool_->deallocate(block);
        case MemoryTier::STORAGE_CACHE:
            return storage_cache_pool_->deallocate(block);
        default:
            return false;
    }
}

MemoryTier MassiveMemoryManager::selectOptimalTier(size_t size, AccessPattern pattern) {
    // Simple heuristics for tier selection
    if (size < 1024 * 1024) { // < 1MB
        return MemoryTier::GPU_DEVICE;
    } else if (size < 100 * 1024 * 1024) { // < 100MB
        if (pattern == AccessPattern::SEQUENTIAL || pattern == AccessPattern::STREAMING) {
            return MemoryTier::GPU_UNIFIED;
        } else {
            return MemoryTier::GPU_DEVICE;
        }
    } else { // > 100MB
        return MemoryTier::CPU_PINNED;
    }
}

void MassiveMemoryManager::backgroundMigrationThread() {
    while (migration_thread_running_) {
        std::this_thread::sleep_for(migration_interval_);

        // Check memory pressure and migrate if needed
        if (auto_migration_enabled_ && isMemoryPressureHigh()) {
            handleMemoryPressure();
        }
    }
}

bool MassiveMemoryManager::isMemoryPressureHigh() const {
    auto stats = getGlobalStats();
    return stats.gpu_device_usage > memory_pressure_threshold_;
}

// MemoryProfiler Implementation
MemoryProfiler::MemoryProfiler() :
    total_allocations_(0), total_deallocations_(0), failed_allocations_(0),
    memory_migrations_(0), avg_allocation_time_(0.0), avg_deallocation_time_(0.0),
    avg_migration_time_(0.0), profiling_enabled_(false) {}

void MemoryProfiler::enable() {
    profiling_enabled_ = true;
    reset();
}

void MemoryProfiler::disable() {
    profiling_enabled_ = false;
}

void MemoryProfiler::reset() {
    std::lock_guard<std::mutex> lock(trace_mutex_);
    trace_log_.clear();
    total_allocations_ = 0;
    total_deallocations_ = 0;
    failed_allocations_ = 0;
    memory_migrations_ = 0;
}

void MemoryProfiler::recordAllocation(void* ptr, size_t size, MemoryTier tier,
                                    AccessPattern pattern, double time_ms) {
    if (!profiling_enabled_) return;

    total_allocations_++;

    // Update average allocation time
    double old_avg = avg_allocation_time_.load();
    double new_avg = (old_avg * (total_allocations_ - 1) + time_ms) / total_allocations_;
    avg_allocation_time_.store(new_avg);

    // Record trace
    std::lock_guard<std::mutex> lock(trace_mutex_);
    AllocationTrace trace;
    trace.timestamp = std::chrono::steady_clock::now();
    trace.size = size;
    trace.tier = tier;
    trace.pattern = pattern;
    trace.ptr = ptr;
    trace.is_allocation = true;

    trace_log_.push_back(trace);

    // Limit trace log size
    if (trace_log_.size() > 10000) {
        trace_log_.erase(trace_log_.begin());
    }
}

void MemoryProfiler::recordDeallocation(void* ptr, double time_ms) {
    if (!profiling_enabled_) return;

    total_deallocations_++;

    // Update average deallocation time
    double old_avg = avg_deallocation_time_.load();
    double new_avg = (old_avg * (total_deallocations_ - 1) + time_ms) / total_deallocations_;
    avg_deallocation_time_.store(new_avg);

    // Record trace
    std::lock_guard<std::mutex> lock(trace_mutex_);
    AllocationTrace trace;
    trace.timestamp = std::chrono::steady_clock::now();
    trace.size = 0; // Size not relevant for deallocation
    trace.tier = MemoryTier::GPU_DEVICE; // Default, not used for deallocation
    trace.pattern = AccessPattern::RANDOM; // Default, not used for deallocation
    trace.ptr = ptr;
    trace.is_allocation = false;

    trace_log_.push_back(trace);

    // Limit trace log size
    if (trace_log_.size() > 10000) {
        trace_log_.erase(trace_log_.begin());
    }
}

void MemoryProfiler::recordMigration(void* ptr, MemoryTier from_tier, MemoryTier to_tier, double time_ms) {
    if (!profiling_enabled_) return;

    memory_migrations_++;

    // Update average migration time
    double old_avg = avg_migration_time_.load();
    double new_avg = (old_avg * (memory_migrations_ - 1) + time_ms) / memory_migrations_;
    avg_migration_time_.store(new_avg);

    // Could record migration trace if needed for detailed analysis
}

void MemoryProfiler::recordFailedAllocation(size_t size, MemoryTier tier) {
    if (!profiling_enabled_) return;

    failed_allocations_++;
}

void MassiveMemoryManager::recordAllocation(MemoryBlock* block) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    global_stats_.current_usage += block->size;
    global_stats_.peak_usage = std::max(global_stats_.peak_usage, global_stats_.current_usage);
}

void MassiveMemoryManager::recordDeallocation(MemoryBlock* block) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    global_stats_.current_usage -= block->size;
}

// ParticleMemoryAllocator Implementation
ParticleMemoryAllocator::ParticleMemoryAllocator(MassiveMemoryManager* manager) :
    memory_manager_(manager), position_buffer_(nullptr), velocity_buffer_(nullptr),
    force_buffer_(nullptr), property_buffer_(nullptr), max_particles_(0),
    current_particles_(0), particle_size_(0) {}

ParticleMemoryAllocator::~ParticleMemoryAllocator() {
    cleanup();
}

bool ParticleMemoryAllocator::initialize(size_t max_particles, size_t particle_size) {
    max_particles_ = max_particles;
    particle_size_ = particle_size;

    // Allocate particle arrays
    size_t position_size = max_particles * 3 * sizeof(float);
    size_t velocity_size = max_particles * 3 * sizeof(float);
    size_t force_size = max_particles * 3 * sizeof(float);
    size_t property_size = max_particles * particle_size;

    position_buffer_ = memory_manager_->allocate(position_size, MemoryTier::GPU_DEVICE,
                                               AccessPattern::SEQUENTIAL, 256);
    velocity_buffer_ = memory_manager_->allocate(velocity_size, MemoryTier::GPU_DEVICE,
                                               AccessPattern::SEQUENTIAL, 256);
    force_buffer_ = memory_manager_->allocate(force_size, MemoryTier::GPU_DEVICE,
                                            AccessPattern::SEQUENTIAL, 256);
    property_buffer_ = memory_manager_->allocate(property_size, MemoryTier::GPU_DEVICE,
                                               AccessPattern::RANDOM, 256);

    return position_buffer_ && velocity_buffer_ && force_buffer_ && property_buffer_;
}

void ParticleMemoryAllocator::cleanup() {
    if (position_buffer_) memory_manager_->deallocate(position_buffer_);
    if (velocity_buffer_) memory_manager_->deallocate(velocity_buffer_);
    if (force_buffer_) memory_manager_->deallocate(force_buffer_);
    if (property_buffer_) memory_manager_->deallocate(property_buffer_);

    position_buffer_ = velocity_buffer_ = force_buffer_ = property_buffer_ = nullptr;
}

bool ParticleMemoryAllocator::resizeParticleArrays(size_t new_count) {
    if (new_count <= max_particles_) {
        current_particles_ = new_count;
        return true;
    }

    // Need to reallocate larger arrays
    cleanup();
    return initialize(new_count * 1.5, particle_size_); // 50% growth
}

// Storage-backed cache implementation
bool MassiveMemoryManager::enableStorageSpill(const std::string& cache_directory) {
    storage_cache_directory_ = cache_directory;
    storage_spill_enabled_ = true;

    // Create cache directory if it doesn't exist
    try {
        std::filesystem::create_directories(cache_directory);
        return true;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Failed to create storage cache directory: " << e.what() << std::endl;
        storage_spill_enabled_ = false;
        return false;
    }
}

bool MassiveMemoryManager::spillToStorage(MemoryBlock* block) {
    if (!storage_spill_enabled_ || !block || !block->ptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(storage_cache_mutex_);

    // Generate unique filename
    std::string filename = storage_cache_directory_ + "/cache_" +
                          std::to_string(storage_cache_file_counter_++) + ".bin";

    // Write memory block to file
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write(static_cast<const char*>(block->ptr), block->size);
    file.close();

    if (file.bad()) {
        std::filesystem::remove(filename);
        return false;
    }

    // Store mapping from memory pointer to file
    storage_cache_files_[block->ptr] = filename;

    // Update statistics
    global_stats_.storage_cache_usage += block->size;

    logMemoryOperation("SPILL_TO_STORAGE", block);
    return true;
}

bool MassiveMemoryManager::reclaimFromStorage(MemoryBlock* block) {
    if (!storage_spill_enabled_ || !block || !block->ptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(storage_cache_mutex_);

    auto it = storage_cache_files_.find(block->ptr);
    if (it == storage_cache_files_.end()) {
        return false;
    }

    const std::string& filename = it->second;

    // Read data back from file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.read(static_cast<char*>(block->ptr), block->size);
    file.close();

    if (file.bad()) {
        return false;
    }

    // Remove cache file and mapping
    std::filesystem::remove(filename);
    storage_cache_files_.erase(it);

    // Update statistics
    global_stats_.storage_cache_usage -= block->size;

    logMemoryOperation("RECLAIM_FROM_STORAGE", block);
    return true;
}

bool MassiveMemoryManager::migrateBlock(MemoryBlock* block, MemoryTier target_tier) {
    if (!block || block->tier == target_tier) {
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // For storage cache migrations, use spill/reclaim
    if (target_tier == MemoryTier::STORAGE_CACHE) {
        bool success = spillToStorage(block);
        if (success) {
            // Free the original memory but keep the block metadata
            deallocateFromTier(block);
            block->tier = MemoryTier::STORAGE_CACHE;
        }
        return success;
    } else if (block->tier == MemoryTier::STORAGE_CACHE) {
        // Allocate new memory in target tier
        MemoryBlock* new_block = allocateFromTier(block->size, target_tier,
                                                 block->pattern, block->alignment);
        if (!new_block) {
            return false;
        }

        // Reclaim data from storage
        void* temp_ptr = block->ptr;
        block->ptr = new_block->ptr;
        bool success = reclaimFromStorage(block);

        if (success) {
            // Update block metadata
            block->tier = target_tier;
            // Copy other metadata from new_block
            block->ptr = new_block->ptr;
        } else {
            // Restore original state
            block->ptr = temp_ptr;
            deallocateFromTier(new_block);
            return false;
        }

        return success;
    } else {
        // Standard memory-to-memory migration
        MemoryBlock* new_block = allocateFromTier(block->size, target_tier,
                                                 block->pattern, block->alignment);
        if (!new_block) {
            return false;
        }

        // Copy data between memory tiers
        #ifdef HAVE_CUDA
        cudaError_t result = cudaMemcpy(new_block->ptr, block->ptr, block->size, cudaMemcpyDefault);
        if (result != cudaSuccess) {
            deallocateFromTier(new_block);
            return false;
        }
        #else
        memcpy(new_block->ptr, block->ptr, block->size);
        #endif

        // Free old memory and update block
        deallocateFromTier(block);
        *block = *new_block;

        auto end_time = std::chrono::high_resolution_clock::now();
        double migration_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        profiler_->recordMigration(block->ptr, block->tier, target_tier, migration_time);
        logMemoryOperation("MIGRATE", block);

        return true;
    }
}

bool MassiveMemoryManager::handleMemoryPressure() {
    auto stats = getGlobalStats();

    if (stats.gpu_device_usage <= memory_pressure_threshold_) {
        return true; // No action needed
    }

    // Find blocks to migrate from GPU to lower tiers
    std::vector<MemoryBlock*> migration_candidates;

    {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        for (const auto& allocation : active_allocations_) {
            MemoryBlock* block = allocation.second;
            if (block->tier == MemoryTier::GPU_DEVICE &&
                block->pattern != AccessPattern::PERSISTENT) {

                // Prioritize by last access time (LRU)
                auto time_since_access = std::chrono::steady_clock::now() - block->last_access;
                if (time_since_access > std::chrono::seconds(10)) {
                    migration_candidates.push_back(block);
                }
            }
        }
    }

    // Sort by last access time (oldest first)
    std::sort(migration_candidates.begin(), migration_candidates.end(),
        [](const MemoryBlock* a, const MemoryBlock* b) {
            return a->last_access < b->last_access;
        });

    // Migrate oldest blocks to relieve pressure
    size_t bytes_to_migrate = stats.gpu_device_usage - memory_pressure_threshold_;
    size_t bytes_migrated = 0;

    for (MemoryBlock* block : migration_candidates) {
        if (bytes_migrated >= bytes_to_migrate) {
            break;
        }

        // Try migrating to unified memory first, then CPU pinned
        MemoryTier target_tier = MemoryTier::GPU_UNIFIED;
        if (!migrateBlock(block, target_tier)) {
            target_tier = MemoryTier::CPU_PINNED;
            if (!migrateBlock(block, target_tier)) {
                target_tier = MemoryTier::STORAGE_CACHE;
                migrateBlock(block, target_tier);
            }
        }

        bytes_migrated += block->size;
    }

    return bytes_migrated > 0;
}

void MassiveMemoryManager::configureCachePolicy(MemoryTier tier, float cache_ratio) {
    // Configure cache policies for different memory tiers
    switch (tier) {
        case MemoryTier::GPU_DEVICE:
            // Aggressive caching for GPU memory
            setPoolSizes(tier, max_gpu_memory_ * cache_ratio);
            break;
        case MemoryTier::GPU_UNIFIED:
            setPoolSizes(tier, max_gpu_memory_ * cache_ratio * 0.5f);
            break;
        case MemoryTier::CPU_PINNED:
            setPoolSizes(tier, max_cpu_memory_ * cache_ratio * 0.3f);
            break;
        case MemoryTier::CPU_PAGED:
            setPoolSizes(tier, max_cpu_memory_ * cache_ratio * 0.6f);
            break;
        case MemoryTier::STORAGE_CACHE:
            setPoolSizes(tier, max_storage_cache_ * cache_ratio);
            break;
    }
}

void MassiveMemoryManager::setPoolSizes(MemoryTier tier, size_t max_size) {
    // Update pool size limits
    switch (tier) {
        case MemoryTier::GPU_DEVICE:
            if (gpu_device_pool_) {
                gpu_device_pool_->setMaxBlocks(max_size / (1024 * 1024)); // Convert to MB blocks
            }
            break;
        case MemoryTier::GPU_UNIFIED:
            if (gpu_unified_pool_) {
                gpu_unified_pool_->setMaxBlocks(max_size / (1024 * 1024));
            }
            break;
        case MemoryTier::CPU_PINNED:
            if (cpu_pinned_pool_) {
                cpu_pinned_pool_->setMaxBlocks(max_size / (1024 * 1024));
            }
            break;
        case MemoryTier::CPU_PAGED:
            if (cpu_paged_pool_) {
                cpu_paged_pool_->setMaxBlocks(max_size / (1024 * 1024));
            }
            break;
        case MemoryTier::STORAGE_CACHE:
            if (storage_cache_pool_) {
                storage_cache_pool_->setMaxBlocks(max_size / (1024 * 1024));
            }
            break;
    }
}

size_t MassiveMemoryManager::alignSize(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

void MassiveMemoryManager::logMemoryOperation(const std::string& operation, MemoryBlock* block) {
    // Optional logging for debugging
    #ifdef DEBUG_MEMORY_OPERATIONS
    std::cout << "[MEMORY] " << operation << " - Tier: " << static_cast<int>(block->tier)
              << ", Size: " << block->size << " bytes, Ptr: " << block->ptr << std::endl;
    #endif
}

void MassiveMemoryManager::enterEmergencyMode() {
    emergency_mode_ = true;

    // Aggressive memory reclamation
    handleMemoryPressure();

    // Enable storage spill if not already enabled
    if (!storage_spill_enabled_) {
        enableStorageSpill("/tmp/physgrad_emergency_cache");
    }

    std::cout << "WARNING: PhysGrad Memory Manager entered emergency mode due to memory pressure!" << std::endl;
}

void MassiveMemoryManager::exitEmergencyMode() {
    emergency_mode_ = false;
    std::cout << "PhysGrad Memory Manager exited emergency mode." << std::endl;
}

void MassiveMemoryManager::setMemoryPressureThreshold(float threshold) {
    memory_pressure_threshold_ = max_gpu_memory_ * threshold;
}

} // namespace physgrad