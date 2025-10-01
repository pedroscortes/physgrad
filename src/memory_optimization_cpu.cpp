#include "memory_optimization.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

namespace physgrad {

// ===============================================================
// MEMORY COALESCING ANALYZER IMPLEMENTATION
// ===============================================================

MemoryCoalescingAnalyzer::AccessAnalysis MemoryCoalescingAnalyzer::analyzeKernelAccess(
    const void* data_ptr,
    size_t element_size,
    size_t num_elements,
    MemoryAccessPattern pattern
) {
    AccessAnalysis analysis = {};

    // Simulate analysis based on access pattern
    switch (pattern) {
        case MemoryAccessPattern::COALESCED_SEQUENTIAL:
            analysis.coalescing_efficiency = 0.95f;
            analysis.cache_line_utilization = 95;
            analysis.memory_bank_conflicts = 0;
            analysis.warp_divergence_factor = 1;
            analysis.memory_throughput_bytes_per_sec = 800ULL * 1024 * 1024 * 1024; // 800 GB/s
            break;

        case MemoryAccessPattern::STRIDED_REGULAR:
            analysis.coalescing_efficiency = 0.75f;
            analysis.cache_line_utilization = 75;
            analysis.memory_bank_conflicts = 2;
            analysis.warp_divergence_factor = 1;
            analysis.memory_throughput_bytes_per_sec = 600ULL * 1024 * 1024 * 1024; // 600 GB/s
            break;

        case MemoryAccessPattern::RANDOM_SCATTERED:
            analysis.coalescing_efficiency = 0.25f;
            analysis.cache_line_utilization = 25;
            analysis.memory_bank_conflicts = 8;
            analysis.warp_divergence_factor = 4;
            analysis.memory_throughput_bytes_per_sec = 200ULL * 1024 * 1024 * 1024; // 200 GB/s
            break;

        default:
            analysis.coalescing_efficiency = 0.5f;
            analysis.cache_line_utilization = 50;
            analysis.memory_bank_conflicts = 4;
            analysis.warp_divergence_factor = 2;
            analysis.memory_throughput_bytes_per_sec = 400ULL * 1024 * 1024 * 1024; // 400 GB/s
            break;
    }

    return analysis;
}

// ===============================================================
// CACHE-OPTIMIZED ARRAY IMPLEMENTATION
// ===============================================================

template<typename T>
CacheOptimizedArray<T>::CacheOptimizedArray(size_t size, bool enable_padding)
    : size_(size), cache_line_size_(64), use_padding_(enable_padding) {

    size_t aligned_size = size_;
    if (use_padding_) {
        // Align to cache line boundaries
        size_t elements_per_line = cache_line_size_ / sizeof(T);
        aligned_size = ((size_ + elements_per_line - 1) / elements_per_line) * elements_per_line;
    }

    data_ = new T[aligned_size];
    if (!data_) {
        throw std::bad_alloc();
    }

    // Initialize to zero
    std::fill(data_, data_ + aligned_size, T{});
}

template<typename T>
CacheOptimizedArray<T>::~CacheOptimizedArray() {
    delete[] data_;
}

template<typename T>
T& CacheOptimizedArray<T>::operator[](size_t index) {
    return data_[index];
}

template<typename T>
const T& CacheOptimizedArray<T>::operator[](size_t index) const {
    return data_[index];
}

template<typename T>
void CacheOptimizedArray<T>::prefetch(size_t index, int cache_level) {
    // Platform-specific prefetch hints would go here
    // For now, this is a no-op on the host side
    (void)index;
    (void)cache_level;
}

template<typename T>
void CacheOptimizedArray<T>::setAccessPattern(MemoryAccessPattern pattern) {
    // Store the access pattern for optimization hints
    // Could be used to adjust prefetch strategies
    (void)pattern;
}

// Explicit instantiations
template class CacheOptimizedArray<float>;
template class CacheOptimizedArray<double>;
template class CacheOptimizedArray<int>;

// ===============================================================
// MORTON ORDER OPTIMIZER HOST IMPLEMENTATIONS
// ===============================================================

uint64_t MortonOrderOptimizer::encode3D(uint32_t x, uint32_t y, uint32_t z) {
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

void MortonOrderOptimizer::decode3D(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z) {
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

// Simplified version for compilation - template implementation would be in header
void MortonOrderOptimizer::reorderParticlesFloat3(
    float3* particles,
    size_t num_particles,
    float3 domain_min,
    float3 domain_max,
    int resolution_bits
) {
    if (!particles || num_particles == 0) return;

    struct MortonParticle {
        uint64_t morton_code;
        float3 data;
    };

    std::vector<MortonParticle> morton_particles(num_particles);

    // Calculate Morton codes for all particles
    float3 domain_size = {
        domain_max.x - domain_min.x,
        domain_max.y - domain_min.y,
        domain_max.z - domain_min.z
    };

    uint32_t max_coord = (1U << resolution_bits) - 1;

    for (size_t i = 0; i < num_particles; ++i) {
        // Normalize particle position to [0, 1] and then to integer grid
        float3 normalized = {
            (particles[i].x - domain_min.x) / domain_size.x,
            (particles[i].y - domain_min.y) / domain_size.y,
            (particles[i].z - domain_min.z) / domain_size.z
        };

        // Clamp to [0, 1] and convert to integer coordinates
        uint32_t x = std::min(max_coord, static_cast<uint32_t>(std::max(0.0f, std::min(1.0f, normalized.x)) * max_coord));
        uint32_t y = std::min(max_coord, static_cast<uint32_t>(std::max(0.0f, std::min(1.0f, normalized.y)) * max_coord));
        uint32_t z = std::min(max_coord, static_cast<uint32_t>(std::max(0.0f, std::min(1.0f, normalized.z)) * max_coord));

        morton_particles[i].morton_code = encode3D(x, y, z);
        morton_particles[i].data = particles[i];
    }

    // Sort particles by Morton code
    std::sort(morton_particles.begin(), morton_particles.end(),
        [](const MortonParticle& a, const MortonParticle& b) {
            return a.morton_code < b.morton_code;
        });

    // Copy back to original array
    for (size_t i = 0; i < num_particles; ++i) {
        particles[i] = morton_particles[i].data;
    }
}

} // namespace physgrad