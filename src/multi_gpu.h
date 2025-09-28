#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <chrono>
#include <cuda_runtime.h>
#include <nccl.h>

namespace physgrad {

// Forward declarations
class GPUContext;
class DomainDecomposer;
class LoadBalancer;

enum class PartitioningStrategy {
    SPATIAL_GRID,      // Divide space into regular grid cells
    OCTREE,           // Hierarchical octree partitioning
    HILBERT_CURVE,    // Space-filling curve partitioning
    DYNAMIC_LOAD,     // Dynamic load-based partitioning
    PARTICLE_COUNT    // Simple particle count balancing
};

enum class CommunicationPattern {
    PEER_TO_PEER,     // Direct GPU-to-GPU communication
    HOST_STAGING,     // Through host memory staging
    NCCL_COLLECTIVE,  // NCCL collective operations
    UNIFIED_MEMORY    // CUDA Unified Memory
};

struct MultiGPUConfig {
    std::vector<int> device_ids;                    // GPU device IDs to use
    PartitioningStrategy partitioning = PartitioningStrategy::SPATIAL_GRID;
    CommunicationPattern communication = CommunicationPattern::NCCL_COLLECTIVE;

    // Spatial partitioning parameters
    float domain_min[3] = {-10.0f, -10.0f, -10.0f};
    float domain_max[3] = {10.0f, 10.0f, 10.0f};
    int grid_divisions[3] = {2, 2, 2};              // Initial grid divisions

    // Performance parameters
    float load_balance_threshold = 0.1f;            // 10% load imbalance threshold
    int rebalance_frequency = 100;                  // Rebalance every N steps
    bool enable_dynamic_balancing = true;
    bool enable_peer_access = true;

    // Communication parameters
    float ghost_layer_width = 2.0f;                // Width of ghost particle regions
    int max_particles_per_transfer = 10000;        // Batch size for transfers
    bool async_communication = true;

    // Memory parameters
    float memory_safety_factor = 0.8f;             // Use 80% of available memory
    bool enable_memory_pooling = true;
    size_t initial_pool_size = 1024 * 1024 * 1024; // 1GB initial pool
};

struct GPUDeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    bool supports_peer_access;
    float relative_performance;  // Normalized performance metric
};

struct DomainPartition {
    int gpu_id;
    float bounds_min[3];
    float bounds_max[3];
    std::vector<int> particle_indices;     // Particles owned by this partition
    std::vector<int> ghost_indices;        // Ghost particles from other partitions
    std::vector<int> neighbor_partitions;  // Adjacent partitions for communication
    size_t particle_count;
    size_t ghost_count;
    float computational_load;              // Estimated computational cost
};

struct CommunicationBuffer {
    void* d_send_buffer;
    void* d_recv_buffer;
    void* h_staging_buffer;
    size_t buffer_size;
    cudaStream_t stream;
    cudaEvent_t send_event;
    cudaEvent_t recv_event;
};

struct MultiGPUStats {
    std::vector<float> gpu_utilization;
    std::vector<size_t> particle_counts;
    std::vector<float> computation_times;
    std::vector<float> communication_times;
    std::vector<size_t> memory_usage;

    float total_simulation_time = 0.0f;
    float load_balance_factor = 1.0f;      // 1.0 = perfect balance
    float communication_overhead = 0.0f;   // Fraction of time spent on communication
    int rebalance_count = 0;

    void reset() {
        std::fill(gpu_utilization.begin(), gpu_utilization.end(), 0.0f);
        std::fill(computation_times.begin(), computation_times.end(), 0.0f);
        std::fill(communication_times.begin(), communication_times.end(), 0.0f);
        total_simulation_time = 0.0f;
        communication_overhead = 0.0f;
    }
};

class MultiGPUManager {
private:
    MultiGPUConfig config;
    std::vector<GPUDeviceInfo> gpu_info;
    std::vector<DomainPartition> partitions;
    std::vector<std::unique_ptr<GPUContext>> gpu_contexts;
    std::unique_ptr<DomainDecomposer> decomposer;
    std::unique_ptr<LoadBalancer> load_balancer;

    // Communication infrastructure
    ncclComm_t* nccl_comms;
    std::vector<CommunicationBuffer> comm_buffers;
    std::vector<cudaStream_t> computation_streams;
    std::vector<cudaStream_t> communication_streams;

    // Performance monitoring
    MultiGPUStats stats;
    std::chrono::high_resolution_clock::time_point start_time;

    // State management
    bool initialized = false;
    int current_step = 0;

public:
    MultiGPUManager(const MultiGPUConfig& config);
    ~MultiGPUManager();

    // Initialization and setup
    bool initialize();
    void shutdown();
    bool isInitialized() const { return initialized; }

    // Device management
    int getDeviceCount() const { return static_cast<int>(config.device_ids.size()); }
    const std::vector<GPUDeviceInfo>& getDeviceInfo() const { return gpu_info; }
    bool enablePeerAccess();
    void queryDeviceCapabilities();

    // Domain decomposition
    void partitionDomain(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    );

    void repartitionDomain(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    );

    // Data distribution
    void distributeParticles(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        const std::vector<float>& vel_x,
        const std::vector<float>& vel_y,
        const std::vector<float>& vel_z,
        const std::vector<float>& masses
    );

    void gatherParticles(
        std::vector<float>& pos_x,
        std::vector<float>& pos_y,
        std::vector<float>& pos_z,
        std::vector<float>& vel_x,
        std::vector<float>& vel_y,
        std::vector<float>& vel_z,
        std::vector<float>& masses
    );

    // Simulation execution
    void executeSimulationStep(float dt);
    void synchronizeGPUs();
    void exchangeGhostParticles();
    void updateLoadBalance();

    // Memory management
    void allocateBuffers(size_t max_particles_per_gpu);
    void reallocateBuffers(const std::vector<size_t>& particle_counts);
    size_t getAvailableMemory(int gpu_id) const;

    // Performance monitoring
    const MultiGPUStats& getStats() const { return stats; }
    void updateStats();
    void printStats() const;
    void resetStats() { stats.reset(); }

    // Configuration
    void setConfig(const MultiGPUConfig& new_config);
    const MultiGPUConfig& getConfig() const { return config; }
    const std::vector<DomainPartition>& getPartitions() const { return partitions; }

    // Debugging and diagnostics
    void validatePartitioning() const;
    void checkCommunicationIntegrity();
    void dumpPartitionInfo() const;

private:
    void initializeNCCL();
    void shutdownNCCL();
    void createCommunicationBuffers();
    void destroyCommunicationBuffers();

    void computePartitionBounds();
    void assignParticlesToPartitions(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    );

    void identifyGhostParticles();
    void setupPeerToPeerAccess();

    // Communication primitives
    void sendParticleData(int src_gpu, int dst_gpu, const std::vector<int>& particle_indices);
    void receiveParticleData(int gpu_id, int src_gpu, std::vector<int>& particle_indices);

    // Load balancing helpers
    float computeLoadImbalance() const;
    bool shouldRebalance() const;
    void executeLoadBalancing();

    // Timing utilities
    void startTimer(int gpu_id);
    float stopTimer(int gpu_id);
};

// Utility functions for multi-GPU operations
namespace MultiGPUUtils {
    // Device query and selection
    std::vector<int> getAvailableGPUs();
    std::vector<int> selectOptimalGPUs(int desired_count, float min_memory_gb = 2.0f);
    bool checkGPUCompatibility(const std::vector<int>& device_ids);

    // Memory utilities
    size_t estimateMemoryRequirements(size_t particle_count, bool include_constraints = true);
    std::vector<size_t> distributeParticles(size_t total_particles, const std::vector<float>& gpu_weights);

    // Spatial partitioning helpers
    void computeOptimalGridDivisions(
        float domain_size[3],
        int num_gpus,
        int grid_divisions[3]
    );

    int getPartitionIndex(
        float x, float y, float z,
        const float* domain_min, const float* domain_max,
        const int* grid_divisions
    );

    // Performance prediction
    float predictCommunicationOverhead(
        const MultiGPUConfig& config,
        size_t particle_count
    );

    float estimateSpeedup(
        int num_gpus,
        size_t particle_count,
        const MultiGPUConfig& config
    );

    // Configuration helpers
    MultiGPUConfig createOptimalConfig(
        size_t particle_count,
        const std::vector<int>& available_gpus
    );

    void validateConfig(const MultiGPUConfig& config);

    // Debugging utilities
    void printGPUTopology(const std::vector<int>& device_ids);
    void benchmarkInterGPUBandwidth(const std::vector<int>& device_ids);
    void testPeerAccess(const std::vector<int>& device_ids);
}

} // namespace physgrad