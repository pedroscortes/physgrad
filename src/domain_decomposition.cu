#include "multi_gpu.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <queue>
#include <unordered_set>

namespace physgrad {

// Domain Decomposer class implementation
class DomainDecomposer {
private:
    MultiGPUConfig config;
    std::vector<DomainPartition> partitions;

public:
    DomainDecomposer(const MultiGPUConfig& config) : config(config) {}

    void createPartitions(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) {
        switch (config.partitioning) {
            case PartitioningStrategy::SPATIAL_GRID:
                createSpatialGridPartitions(pos_x, pos_y, pos_z);
                break;
            case PartitioningStrategy::OCTREE:
                createOctreePartitions(pos_x, pos_y, pos_z);
                break;
            case PartitioningStrategy::HILBERT_CURVE:
                createHilbertCurvePartitions(pos_x, pos_y, pos_z);
                break;
            case PartitioningStrategy::DYNAMIC_LOAD:
                createDynamicLoadPartitions(pos_x, pos_y, pos_z);
                break;
            case PartitioningStrategy::PARTICLE_COUNT:
                createParticleCountPartitions(pos_x, pos_y, pos_z);
                break;
        }

        // Identify neighboring partitions for communication
        identifyNeighborPartitions();

        // Identify ghost particles
        identifyGhostParticles(pos_x, pos_y, pos_z);
    }

    const std::vector<DomainPartition>& getPartitions() const { return partitions; }

private:
    void createSpatialGridPartitions(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) {
        partitions.clear();
        partitions.resize(config.device_ids.size());

        // Compute optimal grid divisions if not specified
        int total_divisions = config.grid_divisions[0] * config.grid_divisions[1] * config.grid_divisions[2];
        if (total_divisions < static_cast<int>(config.device_ids.size())) {
            MultiGPUUtils::computeOptimalGridDivisions(
                config.domain_max, static_cast<int>(config.device_ids.size()), config.grid_divisions
            );
        }

        float domain_size[3] = {
            config.domain_max[0] - config.domain_min[0],
            config.domain_max[1] - config.domain_min[1],
            config.domain_max[2] - config.domain_min[2]
        };

        float cell_size[3] = {
            domain_size[0] / config.grid_divisions[0],
            domain_size[1] / config.grid_divisions[1],
            domain_size[2] / config.grid_divisions[2]
        };

        // Assign each GPU to a region
        for (size_t gpu_idx = 0; gpu_idx < config.device_ids.size(); ++gpu_idx) {
            DomainPartition& partition = partitions[gpu_idx];
            partition.gpu_id = config.device_ids[gpu_idx];

            // Compute 3D grid coordinates for this GPU
            int total_cells = config.grid_divisions[0] * config.grid_divisions[1] * config.grid_divisions[2];
            int cells_per_gpu = total_cells / static_cast<int>(config.device_ids.size());
            int start_cell = static_cast<int>(gpu_idx) * cells_per_gpu;
            int end_cell = (gpu_idx == config.device_ids.size() - 1) ? total_cells : start_cell + cells_per_gpu;

            // Convert linear cell indices to 3D grid bounds
            int start_x = start_cell % config.grid_divisions[0];
            int start_y = (start_cell / config.grid_divisions[0]) % config.grid_divisions[1];
            int start_z = start_cell / (config.grid_divisions[0] * config.grid_divisions[1]);

            int end_x = (end_cell - 1) % config.grid_divisions[0];
            int end_y = ((end_cell - 1) / config.grid_divisions[0]) % config.grid_divisions[1];
            int end_z = (end_cell - 1) / (config.grid_divisions[0] * config.grid_divisions[1]);

            partition.bounds_min[0] = config.domain_min[0] + start_x * cell_size[0];
            partition.bounds_min[1] = config.domain_min[1] + start_y * cell_size[1];
            partition.bounds_min[2] = config.domain_min[2] + start_z * cell_size[2];

            partition.bounds_max[0] = config.domain_min[0] + (end_x + 1) * cell_size[0];
            partition.bounds_max[1] = config.domain_min[1] + (end_y + 1) * cell_size[1];
            partition.bounds_max[2] = config.domain_min[2] + (end_z + 1) * cell_size[2];
        }

        // Assign particles to partitions
        assignParticlesToPartitions(pos_x, pos_y, pos_z);
    }

    void createOctreePartitions(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) {
        partitions.clear();
        partitions.resize(config.device_ids.size());

        // Build octree
        struct OctreeNode {
            float bounds_min[3], bounds_max[3];
            std::vector<int> particles;
            std::vector<std::unique_ptr<OctreeNode>> children;
            bool is_leaf = true;
        };

        auto root = std::make_unique<OctreeNode>();
        for (int i = 0; i < 3; ++i) {
            root->bounds_min[i] = config.domain_min[i];
            root->bounds_max[i] = config.domain_max[i];
        }

        // Add all particles to root
        for (size_t i = 0; i < pos_x.size(); ++i) {
            root->particles.push_back(static_cast<int>(i));
        }

        // Recursively subdivide until we have enough leaf nodes
        std::function<void(OctreeNode*, int)> subdivide = [&](OctreeNode* node, int max_depth) {
            if (max_depth <= 0 || node->particles.size() < 100) {
                return;
            }

            node->is_leaf = false;
            node->children.resize(8);

            float mid[3];
            for (int i = 0; i < 3; ++i) {
                mid[i] = (node->bounds_min[i] + node->bounds_max[i]) * 0.5f;
            }

            // Create 8 child nodes
            for (int i = 0; i < 8; ++i) {
                node->children[i] = std::make_unique<OctreeNode>();
                auto& child = node->children[i];

                child->bounds_min[0] = (i & 1) ? mid[0] : node->bounds_min[0];
                child->bounds_max[0] = (i & 1) ? node->bounds_max[0] : mid[0];
                child->bounds_min[1] = (i & 2) ? mid[1] : node->bounds_min[1];
                child->bounds_max[1] = (i & 2) ? node->bounds_max[1] : mid[1];
                child->bounds_min[2] = (i & 4) ? mid[2] : node->bounds_min[2];
                child->bounds_max[2] = (i & 4) ? node->bounds_max[2] : mid[2];
            }

            // Distribute particles to children
            for (int particle_idx : node->particles) {
                float x = pos_x[particle_idx];
                float y = pos_y[particle_idx];
                float z = pos_z[particle_idx];

                int child_idx = 0;
                if (x >= mid[0]) child_idx |= 1;
                if (y >= mid[1]) child_idx |= 2;
                if (z >= mid[2]) child_idx |= 4;

                node->children[child_idx]->particles.push_back(particle_idx);
            }

            node->particles.clear();

            // Recursively subdivide children
            for (auto& child : node->children) {
                subdivide(child.get(), max_depth - 1);
            }
        };

        int max_depth = static_cast<int>(std::log2(config.device_ids.size() * 8));
        subdivide(root.get(), max_depth);

        // Collect leaf nodes
        std::vector<OctreeNode*> leaf_nodes;
        std::function<void(OctreeNode*)> collectLeaves = [&](OctreeNode* node) {
            if (node->is_leaf) {
                leaf_nodes.push_back(node);
            } else {
                for (auto& child : node->children) {
                    collectLeaves(child.get());
                }
            }
        };
        collectLeaves(root.get());

        // Assign leaf nodes to GPUs (group adjacent leaves)
        for (size_t gpu_idx = 0; gpu_idx < config.device_ids.size(); ++gpu_idx) {
            DomainPartition& partition = partitions[gpu_idx];
            partition.gpu_id = config.device_ids[gpu_idx];

            size_t leaves_per_gpu = leaf_nodes.size() / config.device_ids.size();
            size_t start_leaf = gpu_idx * leaves_per_gpu;
            size_t end_leaf = (gpu_idx == config.device_ids.size() - 1) ?
                             leaf_nodes.size() : start_leaf + leaves_per_gpu;

            // Compute bounding box for assigned leaves
            bool first = true;
            for (size_t leaf_idx = start_leaf; leaf_idx < end_leaf; ++leaf_idx) {
                OctreeNode* leaf = leaf_nodes[leaf_idx];

                if (first) {
                    for (int i = 0; i < 3; ++i) {
                        partition.bounds_min[i] = leaf->bounds_min[i];
                        partition.bounds_max[i] = leaf->bounds_max[i];
                    }
                    first = false;
                } else {
                    for (int i = 0; i < 3; ++i) {
                        partition.bounds_min[i] = std::min(partition.bounds_min[i], leaf->bounds_min[i]);
                        partition.bounds_max[i] = std::max(partition.bounds_max[i], leaf->bounds_max[i]);
                    }
                }

                // Add particles from this leaf
                for (int particle_idx : leaf->particles) {
                    partition.particle_indices.push_back(particle_idx);
                }
            }

            partition.particle_count = partition.particle_indices.size();
        }
    }

    void createHilbertCurvePartitions(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) {
        partitions.clear();
        partitions.resize(config.device_ids.size());

        // Convert 3D positions to Hilbert curve indices
        std::vector<std::pair<uint64_t, int>> hilbert_particles;

        int resolution = 16; // 16-bit resolution per dimension
        float domain_size[3] = {
            config.domain_max[0] - config.domain_min[0],
            config.domain_max[1] - config.domain_min[1],
            config.domain_max[2] - config.domain_min[2]
        };

        for (size_t i = 0; i < pos_x.size(); ++i) {
            // Normalize to [0, 2^resolution)
            uint32_t x = static_cast<uint32_t>((pos_x[i] - config.domain_min[0]) / domain_size[0] * ((1 << resolution) - 1));
            uint32_t y = static_cast<uint32_t>((pos_y[i] - config.domain_min[1]) / domain_size[1] * ((1 << resolution) - 1));
            uint32_t z = static_cast<uint32_t>((pos_z[i] - config.domain_min[2]) / domain_size[2] * ((1 << resolution) - 1));

            // Compute 3D Hilbert index (simplified Morton encoding for now)
            uint64_t hilbert_idx = 0;
            for (int bit = 0; bit < resolution; ++bit) {
                hilbert_idx |= ((uint64_t)(x & (1 << bit)) >> bit) << (3 * bit);
                hilbert_idx |= ((uint64_t)(y & (1 << bit)) >> bit) << (3 * bit + 1);
                hilbert_idx |= ((uint64_t)(z & (1 << bit)) >> bit) << (3 * bit + 2);
            }

            hilbert_particles.emplace_back(hilbert_idx, static_cast<int>(i));
        }

        // Sort by Hilbert index
        std::sort(hilbert_particles.begin(), hilbert_particles.end());

        // Distribute particles along the sorted curve
        size_t particles_per_gpu = hilbert_particles.size() / config.device_ids.size();

        for (size_t gpu_idx = 0; gpu_idx < config.device_ids.size(); ++gpu_idx) {
            DomainPartition& partition = partitions[gpu_idx];
            partition.gpu_id = config.device_ids[gpu_idx];

            size_t start_idx = gpu_idx * particles_per_gpu;
            size_t end_idx = (gpu_idx == config.device_ids.size() - 1) ?
                            hilbert_particles.size() : start_idx + particles_per_gpu;

            // Assign particles
            bool first = true;
            for (size_t i = start_idx; i < end_idx; ++i) {
                int particle_idx = hilbert_particles[i].second;
                partition.particle_indices.push_back(particle_idx);

                // Update bounding box
                if (first) {
                    partition.bounds_min[0] = partition.bounds_max[0] = pos_x[particle_idx];
                    partition.bounds_min[1] = partition.bounds_max[1] = pos_y[particle_idx];
                    partition.bounds_min[2] = partition.bounds_max[2] = pos_z[particle_idx];
                    first = false;
                } else {
                    partition.bounds_min[0] = std::min(partition.bounds_min[0], pos_x[particle_idx]);
                    partition.bounds_max[0] = std::max(partition.bounds_max[0], pos_x[particle_idx]);
                    partition.bounds_min[1] = std::min(partition.bounds_min[1], pos_y[particle_idx]);
                    partition.bounds_max[1] = std::max(partition.bounds_max[1], pos_y[particle_idx]);
                    partition.bounds_min[2] = std::min(partition.bounds_min[2], pos_z[particle_idx]);
                    partition.bounds_max[2] = std::max(partition.bounds_max[2], pos_z[particle_idx]);
                }
            }

            // Expand bounds slightly for ghost layer
            for (int i = 0; i < 3; ++i) {
                partition.bounds_min[i] -= config.ghost_layer_width;
                partition.bounds_max[i] += config.ghost_layer_width;
            }

            partition.particle_count = partition.particle_indices.size();
        }
    }

    void createDynamicLoadPartitions(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) {
        // Start with spatial grid partitioning
        createSpatialGridPartitions(pos_x, pos_y, pos_z);

        // TODO: Implement dynamic load balancing based on computational cost estimates
        // This would involve analyzing particle density, interaction counts, etc.
        // For now, we use spatial grid as the base
    }

    void createParticleCountPartitions(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) {
        partitions.clear();
        partitions.resize(config.device_ids.size());

        size_t particles_per_gpu = pos_x.size() / config.device_ids.size();

        for (size_t gpu_idx = 0; gpu_idx < config.device_ids.size(); ++gpu_idx) {
            DomainPartition& partition = partitions[gpu_idx];
            partition.gpu_id = config.device_ids[gpu_idx];

            size_t start_idx = gpu_idx * particles_per_gpu;
            size_t end_idx = (gpu_idx == config.device_ids.size() - 1) ?
                            pos_x.size() : start_idx + particles_per_gpu;

            // Assign particles
            bool first = true;
            for (size_t i = start_idx; i < end_idx; ++i) {
                partition.particle_indices.push_back(static_cast<int>(i));

                // Update bounding box
                if (first) {
                    partition.bounds_min[0] = partition.bounds_max[0] = pos_x[i];
                    partition.bounds_min[1] = partition.bounds_max[1] = pos_y[i];
                    partition.bounds_min[2] = partition.bounds_max[2] = pos_z[i];
                    first = false;
                } else {
                    partition.bounds_min[0] = std::min(partition.bounds_min[0], pos_x[i]);
                    partition.bounds_max[0] = std::max(partition.bounds_max[0], pos_x[i]);
                    partition.bounds_min[1] = std::min(partition.bounds_min[1], pos_y[i]);
                    partition.bounds_max[1] = std::max(partition.bounds_max[1], pos_y[i]);
                    partition.bounds_min[2] = std::min(partition.bounds_min[2], pos_z[i]);
                    partition.bounds_max[2] = std::max(partition.bounds_max[2], pos_z[i]);
                }
            }

            partition.particle_count = partition.particle_indices.size();
        }
    }

    void assignParticlesToPartitions(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) {
        for (auto& partition : partitions) {
            partition.particle_indices.clear();
        }

        for (size_t i = 0; i < pos_x.size(); ++i) {
            float x = pos_x[i];
            float y = pos_y[i];
            float z = pos_z[i];

            // Find which partition this particle belongs to
            for (auto& partition : partitions) {
                if (x >= partition.bounds_min[0] && x < partition.bounds_max[0] &&
                    y >= partition.bounds_min[1] && y < partition.bounds_max[1] &&
                    z >= partition.bounds_min[2] && z < partition.bounds_max[2]) {

                    partition.particle_indices.push_back(static_cast<int>(i));
                    break;
                }
            }
        }

        // Update particle counts
        for (auto& partition : partitions) {
            partition.particle_count = partition.particle_indices.size();
        }
    }

    void identifyNeighborPartitions() {
        for (size_t i = 0; i < partitions.size(); ++i) {
            partitions[i].neighbor_partitions.clear();

            for (size_t j = 0; j < partitions.size(); ++j) {
                if (i == j) continue;

                // Check if partitions are adjacent (within ghost layer distance)
                bool adjacent = true;
                for (int dim = 0; dim < 3; ++dim) {
                    float min_dist = std::max(
                        partitions[i].bounds_min[dim] - partitions[j].bounds_max[dim],
                        partitions[j].bounds_min[dim] - partitions[i].bounds_max[dim]
                    );
                    if (min_dist > config.ghost_layer_width) {
                        adjacent = false;
                        break;
                    }
                }

                if (adjacent) {
                    partitions[i].neighbor_partitions.push_back(static_cast<int>(j));
                }
            }
        }
    }

    void identifyGhostParticles(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) {
        for (auto& partition : partitions) {
            partition.ghost_indices.clear();

            // Expand bounds by ghost layer width
            float expanded_min[3], expanded_max[3];
            for (int i = 0; i < 3; ++i) {
                expanded_min[i] = partition.bounds_min[i] - config.ghost_layer_width;
                expanded_max[i] = partition.bounds_max[i] + config.ghost_layer_width;
            }

            // Find particles in neighboring partitions that fall within expanded bounds
            for (int neighbor_idx : partition.neighbor_partitions) {
                const auto& neighbor = partitions[neighbor_idx];

                for (int particle_idx : neighbor.particle_indices) {
                    float x = pos_x[particle_idx];
                    float y = pos_y[particle_idx];
                    float z = pos_z[particle_idx];

                    if (x >= expanded_min[0] && x < expanded_max[0] &&
                        y >= expanded_min[1] && y < expanded_max[1] &&
                        z >= expanded_min[2] && z < expanded_max[2]) {

                        // Check if it's not already owned by this partition
                        if (std::find(partition.particle_indices.begin(),
                                     partition.particle_indices.end(),
                                     particle_idx) == partition.particle_indices.end()) {
                            partition.ghost_indices.push_back(particle_idx);
                        }
                    }
                }
            }

            partition.ghost_count = partition.ghost_indices.size();
        }
    }
};

// Add domain decomposition methods to MultiGPUManager
void MultiGPUManager::partitionDomain(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z
) {
    if (!decomposer) {
        decomposer = std::make_unique<DomainDecomposer>(config);
    }

    decomposer->createPartitions(pos_x, pos_y, pos_z);
    partitions = decomposer->getPartitions();

    std::cout << "Domain partitioned into " << partitions.size() << " regions:" << std::endl;
    for (size_t i = 0; i < partitions.size(); ++i) {
        const auto& partition = partitions[i];
        std::cout << "  GPU " << partition.gpu_id
                  << ": " << partition.particle_count << " particles"
                  << ", " << partition.ghost_count << " ghosts"
                  << ", bounds: [" << partition.bounds_min[0] << "," << partition.bounds_min[1] << "," << partition.bounds_min[2]
                  << "] to [" << partition.bounds_max[0] << "," << partition.bounds_max[1] << "," << partition.bounds_max[2] << "]"
                  << std::endl;
    }
}

void MultiGPUManager::repartitionDomain(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z
) {
    if (shouldRebalance()) {
        std::cout << "Repartitioning domain for load balancing..." << std::endl;
        partitionDomain(pos_x, pos_y, pos_z);
        stats.rebalance_count++;
    }
}

// Utility function implementations
namespace MultiGPUUtils {

void computeOptimalGridDivisions(
    float domain_size[3],
    int num_gpus,
    int grid_divisions[3]
) {
    // Start with a cube root distribution
    int base_div = static_cast<int>(std::cbrt(num_gpus));

    grid_divisions[0] = base_div;
    grid_divisions[1] = base_div;
    grid_divisions[2] = base_div;

    // Adjust to match the number of GPUs more closely
    int total_cells = grid_divisions[0] * grid_divisions[1] * grid_divisions[2];

    while (total_cells < num_gpus) {
        // Find the dimension with the largest domain size
        int max_dim = 0;
        for (int i = 1; i < 3; ++i) {
            if (domain_size[i] > domain_size[max_dim]) {
                max_dim = i;
            }
        }
        grid_divisions[max_dim]++;
        total_cells = grid_divisions[0] * grid_divisions[1] * grid_divisions[2];
    }
}

int getPartitionIndex(
    float x, float y, float z,
    const float* domain_min, const float* domain_max,
    const int* grid_divisions
) {
    float domain_size[3] = {
        domain_max[0] - domain_min[0],
        domain_max[1] - domain_min[1],
        domain_max[2] - domain_min[2]
    };

    int grid_x = static_cast<int>((x - domain_min[0]) / domain_size[0] * grid_divisions[0]);
    int grid_y = static_cast<int>((y - domain_min[1]) / domain_size[1] * grid_divisions[1]);
    int grid_z = static_cast<int>((z - domain_min[2]) / domain_size[2] * grid_divisions[2]);

    // Clamp to valid range
    grid_x = std::max(0, std::min(grid_x, grid_divisions[0] - 1));
    grid_y = std::max(0, std::min(grid_y, grid_divisions[1] - 1));
    grid_z = std::max(0, std::min(grid_z, grid_divisions[2] - 1));

    return grid_z * grid_divisions[0] * grid_divisions[1] + grid_y * grid_divisions[0] + grid_x;
}

} // namespace MultiGPUUtils

} // namespace physgrad