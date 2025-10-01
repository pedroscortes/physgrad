/**
 * PhysGrad Sparse Data Structures for 10M+ Particle Scaling
 *
 * High-performance sparse data structures optimized for massive particle simulations
 * with spatial hashing, compressed sparse row formats, and GPU-accelerated operations
 */

#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOST_DEVICE
#define CUDA_DEVICE
#define CUDA_HOST
#endif

namespace physgrad::sparse {

// =============================================================================
// SPATIAL HASHING FOR PARTICLE NEIGHBORHOODS
// =============================================================================

/**
 * High-performance spatial hash table for 10M+ particles
 * Uses Morton Z-curve encoding and GPU-optimized data layout
 */
template<typename T, typename IndexType = uint32_t>
class SpatialHashGrid {
public:
    using index_type = IndexType;
    using float_type = T;

private:
    // Grid parameters
    T cell_size_;
    std::array<T, 3> domain_min_;
    std::array<T, 3> domain_max_;
    std::array<IndexType, 3> grid_resolution_;

    // Hash table storage (GPU-optimized)
    std::vector<IndexType> cell_particle_counts_;  // Number of particles per cell
    std::vector<IndexType> cell_start_indices_;    // Starting index for each cell
    std::vector<IndexType> particle_indices_;      // Sorted particle indices
    std::vector<IndexType> particle_cell_ids_;     // Cell ID for each particle

    // GPU memory pointers
#ifdef __CUDACC__
    IndexType* d_cell_particle_counts_;
    IndexType* d_cell_start_indices_;
    IndexType* d_particle_indices_;
    IndexType* d_particle_cell_ids_;
    bool gpu_memory_allocated_;
#endif

    IndexType total_cells_;
    IndexType max_particles_;

public:
    SpatialHashGrid(T cell_size, const std::array<T, 3>& domain_min,
                    const std::array<T, 3>& domain_max, IndexType max_particles = 10000000)
        : cell_size_(cell_size), domain_min_(domain_min), domain_max_(domain_max), max_particles_(max_particles) {

        // Calculate grid resolution
        for (int i = 0; i < 3; ++i) {
            grid_resolution_[i] = static_cast<IndexType>(std::ceil((domain_max_[i] - domain_min_[i]) / cell_size_));
        }

        total_cells_ = grid_resolution_[0] * grid_resolution_[1] * grid_resolution_[2];

        // Allocate CPU memory
        cell_particle_counts_.resize(total_cells_, 0);
        cell_start_indices_.resize(total_cells_, 0);
        particle_indices_.resize(max_particles_);
        particle_cell_ids_.resize(max_particles_);

#ifdef __CUDACC__
        gpu_memory_allocated_ = false;
        allocateGPUMemory();
#endif
    }

    ~SpatialHashGrid() {
#ifdef __CUDACC__
        if (gpu_memory_allocated_) {
            cudaFree(d_cell_particle_counts_);
            cudaFree(d_cell_start_indices_);
            cudaFree(d_particle_indices_);
            cudaFree(d_particle_cell_ids_);
        }
#endif
    }

#ifdef __CUDACC__
    void allocateGPUMemory() {
        cudaMalloc(&d_cell_particle_counts_, total_cells_ * sizeof(IndexType));
        cudaMalloc(&d_cell_start_indices_, total_cells_ * sizeof(IndexType));
        cudaMalloc(&d_particle_indices_, max_particles_ * sizeof(IndexType));
        cudaMalloc(&d_particle_cell_ids_, max_particles_ * sizeof(IndexType));
        gpu_memory_allocated_ = true;
    }
#endif

    // Morton Z-curve encoding for spatial coherence
    CUDA_HOST_DEVICE IndexType mortonEncode3D(IndexType x, IndexType y, IndexType z) const {
        auto expandBits = [](IndexType v) -> IndexType {
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        };

        return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
    }

    // Get cell coordinates from world position
    CUDA_HOST_DEVICE std::array<IndexType, 3> getGridCoords(T x, T y, T z) const {
        return {
            static_cast<IndexType>((x - domain_min_[0]) / cell_size_),
            static_cast<IndexType>((y - domain_min_[1]) / cell_size_),
            static_cast<IndexType>((z - domain_min_[2]) / cell_size_)
        };
    }

    // Get cell ID from world position
    CUDA_HOST_DEVICE IndexType getCellID(T x, T y, T z) const {
        auto coords = getGridCoords(x, y, z);

        // Clamp to grid bounds
        coords[0] = std::min(coords[0], grid_resolution_[0] - 1);
        coords[1] = std::min(coords[1], grid_resolution_[1] - 1);
        coords[2] = std::min(coords[2], grid_resolution_[2] - 1);

        return mortonEncode3D(coords[0], coords[1], coords[2]) % total_cells_;
    }

    // Build hash table from particle positions
    void buildHashTable(const std::vector<T>& positions, IndexType num_particles) {
        assert(num_particles <= max_particles_);
        assert(positions.size() >= num_particles * 3);

        // Reset counters
        std::fill(cell_particle_counts_.begin(), cell_particle_counts_.end(), 0);

        // Count particles per cell
        for (IndexType i = 0; i < num_particles; ++i) {
            T x = positions[3 * i];
            T y = positions[3 * i + 1];
            T z = positions[3 * i + 2];

            IndexType cell_id = getCellID(x, y, z);
            particle_cell_ids_[i] = cell_id;
            cell_particle_counts_[cell_id]++;
        }

        // Compute start indices using prefix sum
        cell_start_indices_[0] = 0;
        for (IndexType i = 1; i < total_cells_; ++i) {
            cell_start_indices_[i] = cell_start_indices_[i-1] + cell_particle_counts_[i-1];
        }

        // Sort particles by cell ID for cache coherence
        std::vector<IndexType> temp_counters = cell_start_indices_;
        for (IndexType i = 0; i < num_particles; ++i) {
            IndexType cell_id = particle_cell_ids_[i];
            particle_indices_[temp_counters[cell_id]++] = i;
        }

#ifdef __CUDACC__
        // Copy to GPU
        copyToGPU(num_particles);
#endif
    }

#ifdef __CUDACC__
    void copyToGPU(IndexType num_particles) {
        cudaMemcpy(d_cell_particle_counts_, cell_particle_counts_.data(),
                   total_cells_ * sizeof(IndexType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cell_start_indices_, cell_start_indices_.data(),
                   total_cells_ * sizeof(IndexType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_particle_indices_, particle_indices_.data(),
                   num_particles * sizeof(IndexType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_particle_cell_ids_, particle_cell_ids_.data(),
                   num_particles * sizeof(IndexType), cudaMemcpyHostToDevice);
    }
#endif

    // Get neighboring particles within radius
    std::vector<IndexType> getNeighbors(T x, T y, T z, T radius) const {
        std::vector<IndexType> neighbors;

        // Determine search radius in cells
        IndexType search_radius_cells = static_cast<IndexType>(std::ceil(radius / cell_size_));
        auto center_coords = getGridCoords(x, y, z);

        T radius_sq = radius * radius;

        // Search neighboring cells
        for (IndexType dx = 0; dx <= 2 * search_radius_cells; ++dx) {
            for (IndexType dy = 0; dy <= 2 * search_radius_cells; ++dy) {
                for (IndexType dz = 0; dz <= 2 * search_radius_cells; ++dz) {
                    IndexType gx = center_coords[0] + dx - search_radius_cells;
                    IndexType gy = center_coords[1] + dy - search_radius_cells;
                    IndexType gz = center_coords[2] + dz - search_radius_cells;

                    // Check bounds
                    if (gx >= grid_resolution_[0] || gy >= grid_resolution_[1] || gz >= grid_resolution_[2])
                        continue;

                    IndexType cell_id = mortonEncode3D(gx, gy, gz) % total_cells_;
                    IndexType start_idx = cell_start_indices_[cell_id];
                    IndexType count = cell_particle_counts_[cell_id];

                    // Check particles in this cell
                    for (IndexType i = 0; i < count; ++i) {
                        neighbors.push_back(particle_indices_[start_idx + i]);
                    }
                }
            }
        }

        return neighbors;
    }

    // Performance metrics
    struct HashTableStats {
        T average_particles_per_cell;
        T load_factor;
        IndexType max_particles_per_cell;
        IndexType occupied_cells;
        T memory_usage_mb;
    };

    HashTableStats getStats() const {
        HashTableStats stats;

        IndexType occupied_cells = 0;
        IndexType max_particles = 0;
        IndexType total_particles = 0;

        for (IndexType count : cell_particle_counts_) {
            if (count > 0) {
                occupied_cells++;
                max_particles = std::max(max_particles, count);
                total_particles += count;
            }
        }

        stats.occupied_cells = occupied_cells;
        stats.max_particles_per_cell = max_particles;
        stats.average_particles_per_cell = static_cast<T>(total_particles) / std::max(IndexType(1), occupied_cells);
        stats.load_factor = static_cast<T>(occupied_cells) / total_cells_;

        // Memory usage calculation
        size_t memory_bytes = cell_particle_counts_.size() * sizeof(IndexType) +
                             cell_start_indices_.size() * sizeof(IndexType) +
                             particle_indices_.size() * sizeof(IndexType) +
                             particle_cell_ids_.size() * sizeof(IndexType);
        stats.memory_usage_mb = static_cast<T>(memory_bytes) / (1024.0f * 1024.0f);

        return stats;
    }

    // Accessors
    T getCellSize() const { return cell_size_; }
    IndexType getTotalCells() const { return total_cells_; }
    const std::vector<IndexType>& getParticleIndices() const { return particle_indices_; }
    const std::vector<IndexType>& getCellStartIndices() const { return cell_start_indices_; }
    const std::vector<IndexType>& getCellParticleCounts() const { return cell_particle_counts_; }

#ifdef __CUDACC__
    // GPU accessors
    IndexType* getDeviceCellStartIndices() const { return d_cell_start_indices_; }
    IndexType* getDeviceParticleIndices() const { return d_particle_indices_; }
    IndexType* getDeviceCellParticleCounts() const { return d_cell_particle_counts_; }
#endif
};

// =============================================================================
// COMPRESSED SPARSE ROW (CSR) MATRIX STORAGE
// =============================================================================

/**
 * GPU-optimized CSR matrix for sparse force computations
 * Scales to 10M x 10M matrices with minimal memory footprint
 */
template<typename T, typename IndexType = uint32_t>
class SparseMatrix {
private:
    std::vector<T> values_;              // Non-zero values
    std::vector<IndexType> column_indices_;  // Column indices for each value
    std::vector<IndexType> row_pointers_;    // Starting index for each row

    IndexType num_rows_;
    IndexType num_cols_;
    IndexType nnz_;  // Number of non-zeros

#ifdef __CUDACC__
    T* d_values_;
    IndexType* d_column_indices_;
    IndexType* d_row_pointers_;
    bool gpu_memory_allocated_;
#endif

public:
    SparseMatrix(IndexType num_rows, IndexType num_cols, IndexType estimated_nnz = 0)
        : num_rows_(num_rows), num_cols_(num_cols), nnz_(0) {

        if (estimated_nnz > 0) {
            values_.reserve(estimated_nnz);
            column_indices_.reserve(estimated_nnz);
        }
        row_pointers_.resize(num_rows + 1, 0);

#ifdef __CUDACC__
        gpu_memory_allocated_ = false;
#endif
    }

    ~SparseMatrix() {
#ifdef __CUDACC__
        if (gpu_memory_allocated_) {
            cudaFree(d_values_);
            cudaFree(d_column_indices_);
            cudaFree(d_row_pointers_);
        }
#endif
    }

    // Add non-zero element (assumes row-major insertion order)
    void addElement(IndexType row, IndexType col, T value) {
        assert(row < num_rows_ && col < num_cols_);
        values_.push_back(value);
        column_indices_.push_back(col);
        nnz_++;
    }

    // Finalize matrix construction (compute row pointers)
    void finalize() {
        // This assumes elements were added in row-major order
        // In practice, you'd sort by (row, col) first

        IndexType current_row = 0;
        for (IndexType i = 0; i < nnz_; ++i) {
            while (current_row < num_rows_ && row_pointers_[current_row + 1] <= i) {
                current_row++;
            }
            if (current_row < num_rows_) {
                row_pointers_[current_row + 1] = i + 1;
            }
        }

        // Forward fill row pointers
        for (IndexType i = 1; i <= num_rows_; ++i) {
            if (row_pointers_[i] == 0) {
                row_pointers_[i] = row_pointers_[i - 1];
            }
        }

#ifdef __CUDACC__
        copyToGPU();
#endif
    }

#ifdef __CUDACC__
    void copyToGPU() {
        cudaMalloc(&d_values_, nnz_ * sizeof(T));
        cudaMalloc(&d_column_indices_, nnz_ * sizeof(IndexType));
        cudaMalloc(&d_row_pointers_, (num_rows_ + 1) * sizeof(IndexType));

        cudaMemcpy(d_values_, values_.data(), nnz_ * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_column_indices_, column_indices_.data(), nnz_ * sizeof(IndexType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_pointers_, row_pointers_.data(), (num_rows_ + 1) * sizeof(IndexType), cudaMemcpyHostToDevice);

        gpu_memory_allocated_ = true;
    }
#endif

    // Sparse matrix-vector multiplication
    std::vector<T> multiply(const std::vector<T>& x) const {
        assert(x.size() == num_cols_);
        std::vector<T> y(num_rows_, T{0});

        for (IndexType row = 0; row < num_rows_; ++row) {
            T sum = T{0};
            IndexType start = row_pointers_[row];
            IndexType end = row_pointers_[row + 1];

            for (IndexType idx = start; idx < end; ++idx) {
                sum += values_[idx] * x[column_indices_[idx]];
            }
            y[row] = sum;
        }

        return y;
    }

    // Memory usage analysis
    struct MemoryStats {
        T storage_efficiency;  // nnz / (num_rows * num_cols)
        T memory_usage_mb;
        T compression_ratio;   // vs dense storage
    };

    MemoryStats getMemoryStats() const {
        MemoryStats stats;

        size_t sparse_bytes = nnz_ * (sizeof(T) + sizeof(IndexType)) +
                             (num_rows_ + 1) * sizeof(IndexType);
        size_t dense_bytes = static_cast<size_t>(num_rows_) * num_cols_ * sizeof(T);

        stats.memory_usage_mb = static_cast<T>(sparse_bytes) / (1024.0f * 1024.0f);
        stats.compression_ratio = static_cast<T>(dense_bytes) / sparse_bytes;
        stats.storage_efficiency = static_cast<T>(nnz_) / (static_cast<T>(num_rows_) * num_cols_);

        return stats;
    }

    // Accessors
    IndexType getNumRows() const { return num_rows_; }
    IndexType getNumCols() const { return num_cols_; }
    IndexType getNNZ() const { return nnz_; }
    const std::vector<T>& getValues() const { return values_; }
    const std::vector<IndexType>& getColumnIndices() const { return column_indices_; }
    const std::vector<IndexType>& getRowPointers() const { return row_pointers_; }
};

// =============================================================================
// HIERARCHICAL DATA STRUCTURES FOR MASSIVE SCALING
// =============================================================================

/**
 * Octree-based hierarchical spatial partitioning
 * Enables O(N log N) scaling for 10M+ particles
 */
template<typename T, typename IndexType = uint32_t>
class AdaptiveOctree {
private:
    struct OctreeNode {
        std::array<T, 3> center;
        T half_width;
        std::vector<IndexType> particle_indices;
        std::array<std::unique_ptr<OctreeNode>, 8> children;
        bool is_leaf;

        OctreeNode(const std::array<T, 3>& center, T half_width)
            : center(center), half_width(half_width), is_leaf(true) {}
    };

    std::unique_ptr<OctreeNode> root_;
    IndexType max_particles_per_node_;
    IndexType max_depth_;
    IndexType total_particles_;

public:
    AdaptiveOctree(const std::array<T, 3>& domain_center, T domain_half_width,
                   IndexType max_particles_per_node = 64, IndexType max_depth = 20)
        : max_particles_per_node_(max_particles_per_node), max_depth_(max_depth), total_particles_(0) {
        root_ = std::make_unique<OctreeNode>(domain_center, domain_half_width);
    }

    void insert(IndexType particle_id, const std::array<T, 3>& position) {
        insertRecursive(root_.get(), particle_id, position, 0);
        total_particles_++;
    }

    void build(const std::vector<T>& positions, IndexType num_particles) {
        total_particles_ = 0;
        root_->particle_indices.clear();

        for (IndexType i = 0; i < num_particles; ++i) {
            std::array<T, 3> pos = {positions[3*i], positions[3*i+1], positions[3*i+2]};
            insert(i, pos);
        }
    }

    std::vector<IndexType> query(const std::array<T, 3>& position, T radius) const {
        std::vector<IndexType> result;
        queryRecursive(root_.get(), position, radius, result);
        return result;
    }

    struct TreeStats {
        IndexType total_nodes;
        IndexType leaf_nodes;
        IndexType max_depth_reached;
        T average_particles_per_leaf;
        T memory_usage_mb;
    };

    TreeStats getStats() const {
        TreeStats stats = {0, 0, 0, 0.0f, 0.0f};
        computeStatsRecursive(root_.get(), 0, stats);

        if (stats.leaf_nodes > 0) {
            stats.average_particles_per_leaf = static_cast<T>(total_particles_) / stats.leaf_nodes;
        }

        // Rough memory estimate
        stats.memory_usage_mb = static_cast<T>(stats.total_nodes * sizeof(OctreeNode)) / (1024.0f * 1024.0f);

        return stats;
    }

private:
    void insertRecursive(OctreeNode* node, IndexType particle_id,
                        const std::array<T, 3>& position, IndexType depth) {

        if (node->is_leaf) {
            node->particle_indices.push_back(particle_id);

            // Subdivide if necessary
            if (node->particle_indices.size() > max_particles_per_node_ && depth < max_depth_) {
                subdivide(node, depth);
            }
        } else {
            // Find appropriate child
            IndexType child_idx = getChildIndex(node, position);
            insertRecursive(node->children[child_idx].get(), particle_id, position, depth + 1);
        }
    }

    void subdivide(OctreeNode* node, IndexType depth) {
        node->is_leaf = false;
        T quarter_width = node->half_width * 0.5f;

        // Create 8 children
        for (IndexType i = 0; i < 8; ++i) {
            std::array<T, 3> child_center = node->center;
            child_center[0] += ((i & 1) ? quarter_width : -quarter_width);
            child_center[1] += ((i & 2) ? quarter_width : -quarter_width);
            child_center[2] += ((i & 4) ? quarter_width : -quarter_width);

            node->children[i] = std::make_unique<OctreeNode>(child_center, quarter_width);
        }

        // Redistribute particles
        std::vector<IndexType> particles = std::move(node->particle_indices);
        node->particle_indices.clear();

        for (IndexType particle_id : particles) {
            // This requires storing particle positions, simplified for now
            // In practice, you'd look up the position and redistribute
        }
    }

    IndexType getChildIndex(OctreeNode* node, const std::array<T, 3>& position) const {
        IndexType index = 0;
        if (position[0] > node->center[0]) index |= 1;
        if (position[1] > node->center[1]) index |= 2;
        if (position[2] > node->center[2]) index |= 4;
        return index;
    }

    void queryRecursive(OctreeNode* node, const std::array<T, 3>& position, T radius,
                       std::vector<IndexType>& result) const {

        // Check if sphere intersects node
        T dx = std::max(T{0}, std::max(node->center[0] - node->half_width - position[0],
                                      position[0] - (node->center[0] + node->half_width)));
        T dy = std::max(T{0}, std::max(node->center[1] - node->half_width - position[1],
                                      position[1] - (node->center[1] + node->half_width)));
        T dz = std::max(T{0}, std::max(node->center[2] - node->half_width - position[2],
                                      position[2] - (node->center[2] + node->half_width)));

        if (dx*dx + dy*dy + dz*dz > radius*radius) return;

        if (node->is_leaf) {
            result.insert(result.end(), node->particle_indices.begin(), node->particle_indices.end());
        } else {
            for (auto& child : node->children) {
                if (child) {
                    queryRecursive(child.get(), position, radius, result);
                }
            }
        }
    }

    void computeStatsRecursive(OctreeNode* node, IndexType depth, TreeStats& stats) const {
        stats.total_nodes++;
        stats.max_depth_reached = std::max(stats.max_depth_reached, depth);

        if (node->is_leaf) {
            stats.leaf_nodes++;
        } else {
            for (auto& child : node->children) {
                if (child) {
                    computeStatsRecursive(child.get(), depth + 1, stats);
                }
            }
        }
    }
};

} // namespace physgrad::sparse