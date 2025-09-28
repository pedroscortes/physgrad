#include "multi_gpu.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace physgrad {

// GPU Context for individual device management
class GPUContext {
private:
    int device_id;
    cudaStream_t computation_stream;
    cudaStream_t communication_stream;

    // Memory pools
    void* memory_pool;
    size_t pool_size;
    size_t pool_used;

    // Device-specific data
    float* d_pos_x;
    float* d_pos_y;
    float* d_pos_z;
    float* d_vel_x;
    float* d_vel_y;
    float* d_vel_z;
    float* d_masses;
    float* d_forces_x;
    float* d_forces_y;
    float* d_forces_z;

    size_t allocated_particles;

public:
    GPUContext(int device_id) : device_id(device_id), memory_pool(nullptr),
                               pool_size(0), pool_used(0), allocated_particles(0) {
        cudaSetDevice(device_id);
        cudaStreamCreate(&computation_stream);
        cudaStreamCreate(&communication_stream);

        // Initialize device pointers
        d_pos_x = d_pos_y = d_pos_z = nullptr;
        d_vel_x = d_vel_y = d_vel_z = nullptr;
        d_masses = d_forces_x = d_forces_y = d_forces_z = nullptr;
    }

    ~GPUContext() {
        cleanup();
    }

    bool initializeMemoryPool(size_t size) {
        cudaSetDevice(device_id);
        cudaError_t err = cudaMalloc(&memory_pool, size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate memory pool on GPU " << device_id
                      << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        pool_size = size;
        pool_used = 0;
        return true;
    }

    void* allocateFromPool(size_t size) {
        if (pool_used + size > pool_size) {
            return nullptr; // Not enough space in pool
        }
        void* ptr = static_cast<char*>(memory_pool) + pool_used;
        pool_used += size;
        return ptr;
    }

    bool allocateParticleData(size_t num_particles) {
        cudaSetDevice(device_id);

        size_t bytes_per_array = num_particles * sizeof(float);
        size_t total_bytes = 8 * bytes_per_array; // 8 arrays total

        // Try to allocate from pool first
        if (memory_pool && pool_used + total_bytes <= pool_size) {
            d_pos_x = static_cast<float*>(allocateFromPool(bytes_per_array));
            d_pos_y = static_cast<float*>(allocateFromPool(bytes_per_array));
            d_pos_z = static_cast<float*>(allocateFromPool(bytes_per_array));
            d_vel_x = static_cast<float*>(allocateFromPool(bytes_per_array));
            d_vel_y = static_cast<float*>(allocateFromPool(bytes_per_array));
            d_vel_z = static_cast<float*>(allocateFromPool(bytes_per_array));
            d_masses = static_cast<float*>(allocateFromPool(bytes_per_array));
            d_forces_x = static_cast<float*>(allocateFromPool(bytes_per_array));
            d_forces_y = static_cast<float*>(allocateFromPool(bytes_per_array));
            d_forces_z = static_cast<float*>(allocateFromPool(bytes_per_array));
        } else {
            // Fallback to individual allocations
            if (cudaMalloc(&d_pos_x, bytes_per_array) != cudaSuccess ||
                cudaMalloc(&d_pos_y, bytes_per_array) != cudaSuccess ||
                cudaMalloc(&d_pos_z, bytes_per_array) != cudaSuccess ||
                cudaMalloc(&d_vel_x, bytes_per_array) != cudaSuccess ||
                cudaMalloc(&d_vel_y, bytes_per_array) != cudaSuccess ||
                cudaMalloc(&d_vel_z, bytes_per_array) != cudaSuccess ||
                cudaMalloc(&d_masses, bytes_per_array) != cudaSuccess ||
                cudaMalloc(&d_forces_x, bytes_per_array) != cudaSuccess ||
                cudaMalloc(&d_forces_y, bytes_per_array) != cudaSuccess ||
                cudaMalloc(&d_forces_z, bytes_per_array) != cudaSuccess) {

                freeParticleData();
                return false;
            }
        }

        allocated_particles = num_particles;
        return true;
    }

    void freeParticleData() {
        if (!memory_pool) {
            // Free individual allocations
            if (d_pos_x) cudaFree(d_pos_x);
            if (d_pos_y) cudaFree(d_pos_y);
            if (d_pos_z) cudaFree(d_pos_z);
            if (d_vel_x) cudaFree(d_vel_x);
            if (d_vel_y) cudaFree(d_vel_y);
            if (d_vel_z) cudaFree(d_vel_z);
            if (d_masses) cudaFree(d_masses);
            if (d_forces_x) cudaFree(d_forces_x);
            if (d_forces_y) cudaFree(d_forces_y);
            if (d_forces_z) cudaFree(d_forces_z);
        }

        d_pos_x = d_pos_y = d_pos_z = nullptr;
        d_vel_x = d_vel_y = d_vel_z = nullptr;
        d_masses = d_forces_x = d_forces_y = d_forces_z = nullptr;
        allocated_particles = 0;
    }

    void cleanup() {
        cudaSetDevice(device_id);
        freeParticleData();

        if (memory_pool) {
            cudaFree(memory_pool);
            memory_pool = nullptr;
        }

        if (computation_stream) {
            cudaStreamDestroy(computation_stream);
            computation_stream = nullptr;
        }

        if (communication_stream) {
            cudaStreamDestroy(communication_stream);
            communication_stream = nullptr;
        }
    }

    // Getters
    int getDeviceId() const { return device_id; }
    cudaStream_t getComputationStream() const { return computation_stream; }
    cudaStream_t getCommunicationStream() const { return communication_stream; }
    size_t getAllocatedParticles() const { return allocated_particles; }

    float* getPosX() const { return d_pos_x; }
    float* getPosY() const { return d_pos_y; }
    float* getPosZ() const { return d_pos_z; }
    float* getVelX() const { return d_vel_x; }
    float* getVelY() const { return d_vel_y; }
    float* getVelZ() const { return d_vel_z; }
    float* getMasses() const { return d_masses; }
    float* getForcesX() const { return d_forces_x; }
    float* getForcesY() const { return d_forces_y; }
    float* getForcesZ() const { return d_forces_z; }
};

// MultiGPUManager Implementation
MultiGPUManager::MultiGPUManager(const MultiGPUConfig& config)
    : config(config), nccl_comms(nullptr), initialized(false), current_step(0) {
}

MultiGPUManager::~MultiGPUManager() {
    shutdown();
}

bool MultiGPUManager::initialize() {
    if (initialized) {
        return true;
    }

    std::cout << "Initializing Multi-GPU Manager..." << std::endl;

    // Query device capabilities
    queryDeviceCapabilities();

    // Validate configuration
    MultiGPUUtils::validateConfig(config);

    // Create GPU contexts
    gpu_contexts.clear();
    for (int device_id : config.device_ids) {
        auto context = std::make_unique<GPUContext>(device_id);

        // Initialize memory pool
        size_t pool_size = static_cast<size_t>(
            getAvailableMemory(device_id) * config.memory_safety_factor
        );

        if (config.enable_memory_pooling && pool_size > config.initial_pool_size) {
            pool_size = std::max(pool_size, config.initial_pool_size);
            if (!context->initializeMemoryPool(pool_size)) {
                std::cerr << "Failed to initialize memory pool for GPU " << device_id << std::endl;
                return false;
            }
        }

        gpu_contexts.push_back(std::move(context));
    }

    // Set up peer-to-peer access
    if (config.enable_peer_access) {
        enablePeerAccess();
    }

    // Initialize NCCL for collective communication
    if (config.communication == CommunicationPattern::NCCL_COLLECTIVE) {
        initializeNCCL();
    }

    // Create communication buffers
    createCommunicationBuffers();

    // Initialize statistics
    stats.gpu_utilization.resize(config.device_ids.size(), 0.0f);
    stats.particle_counts.resize(config.device_ids.size(), 0);
    stats.computation_times.resize(config.device_ids.size(), 0.0f);
    stats.communication_times.resize(config.device_ids.size(), 0.0f);
    stats.memory_usage.resize(config.device_ids.size(), 0);

    initialized = true;
    std::cout << "Multi-GPU Manager initialized successfully with "
              << config.device_ids.size() << " GPUs" << std::endl;

    return true;
}

void MultiGPUManager::shutdown() {
    if (!initialized) {
        return;
    }

    std::cout << "Shutting down Multi-GPU Manager..." << std::endl;

    // Cleanup communication
    destroyCommunicationBuffers();

    if (config.communication == CommunicationPattern::NCCL_COLLECTIVE) {
        shutdownNCCL();
    }

    // Cleanup GPU contexts
    gpu_contexts.clear();

    initialized = false;
    std::cout << "Multi-GPU Manager shutdown complete" << std::endl;
}

void MultiGPUManager::queryDeviceCapabilities() {
    gpu_info.clear();

    for (int device_id : config.device_ids) {
        GPUDeviceInfo info;
        info.device_id = device_id;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        info.name = prop.name;
        info.total_memory = prop.totalGlobalMem;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        info.multiprocessor_count = prop.multiProcessorCount;
        info.max_threads_per_block = prop.maxThreadsPerBlock;

        // Check free memory
        cudaSetDevice(device_id);
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        info.free_memory = free_mem;

        // Estimate relative performance (simplified metric)
        info.relative_performance = static_cast<float>(
            info.multiprocessor_count *
            (info.compute_capability_major * 100 + info.compute_capability_minor)
        );

        gpu_info.push_back(info);

        std::cout << "GPU " << device_id << ": " << info.name
                  << " (" << info.free_memory / (1024*1024) << " MB free)" << std::endl;
    }

    // Normalize performance values
    if (!gpu_info.empty()) {
        float max_perf = *std::max_element(gpu_info.begin(), gpu_info.end(),
            [](const GPUDeviceInfo& a, const GPUDeviceInfo& b) {
                return a.relative_performance < b.relative_performance;
            }
        ).relative_performance;

        for (auto& info : gpu_info) {
            info.relative_performance /= max_perf;
        }
    }
}

bool MultiGPUManager::enablePeerAccess() {
    std::cout << "Setting up peer-to-peer access..." << std::endl;

    bool success = true;
    for (size_t i = 0; i < config.device_ids.size(); ++i) {
        cudaSetDevice(config.device_ids[i]);

        for (size_t j = 0; j < config.device_ids.size(); ++j) {
            if (i != j) {
                int can_access;
                cudaDeviceCanAccessPeer(&can_access, config.device_ids[i], config.device_ids[j]);

                if (can_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(config.device_ids[j], 0);
                    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                        std::cerr << "Failed to enable peer access from GPU "
                                  << config.device_ids[i] << " to GPU " << config.device_ids[j]
                                  << ": " << cudaGetErrorString(err) << std::endl;
                        success = false;
                    }
                } else {
                    std::cout << "Peer access not supported between GPU "
                              << config.device_ids[i] << " and GPU " << config.device_ids[j] << std::endl;
                    gpu_info[i].supports_peer_access = false;
                }
            }
        }
    }

    return success;
}

size_t MultiGPUManager::getAvailableMemory(int gpu_id) const {
    cudaSetDevice(gpu_id);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

void MultiGPUManager::initializeNCCL() {
    if (config.device_ids.empty()) {
        return;
    }

    int num_gpus = static_cast<int>(config.device_ids.size());
    nccl_comms = new ncclComm_t[num_gpus];

    // Generate unique ID for NCCL
    ncclUniqueId id;
    ncclGetUniqueId(&id);

    // Initialize NCCL communicators
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(config.device_ids[i]);
        ncclCommInitRank(&nccl_comms[i], num_gpus, id, i);
    }

    std::cout << "NCCL initialized for " << num_gpus << " GPUs" << std::endl;
}

void MultiGPUManager::shutdownNCCL() {
    if (nccl_comms) {
        for (size_t i = 0; i < config.device_ids.size(); ++i) {
            ncclCommDestroy(nccl_comms[i]);
        }
        delete[] nccl_comms;
        nccl_comms = nullptr;
    }
}

void MultiGPUManager::createCommunicationBuffers() {
    comm_buffers.clear();

    size_t buffer_size = config.max_particles_per_transfer * sizeof(float) * 6; // pos + vel

    for (size_t i = 0; i < config.device_ids.size(); ++i) {
        CommunicationBuffer buffer;

        cudaSetDevice(config.device_ids[i]);

        // Allocate device buffers
        cudaMalloc(&buffer.d_send_buffer, buffer_size);
        cudaMalloc(&buffer.d_recv_buffer, buffer_size);

        // Allocate host staging buffer if needed
        if (config.communication == CommunicationPattern::HOST_STAGING) {
            cudaMallocHost(&buffer.h_staging_buffer, buffer_size);
        } else {
            buffer.h_staging_buffer = nullptr;
        }

        buffer.buffer_size = buffer_size;

        // Create streams and events
        cudaStreamCreate(&buffer.stream);
        cudaEventCreate(&buffer.send_event);
        cudaEventCreate(&buffer.recv_event);

        comm_buffers.push_back(buffer);
    }
}

void MultiGPUManager::destroyCommunicationBuffers() {
    for (auto& buffer : comm_buffers) {
        if (buffer.d_send_buffer) cudaFree(buffer.d_send_buffer);
        if (buffer.d_recv_buffer) cudaFree(buffer.d_recv_buffer);
        if (buffer.h_staging_buffer) cudaFreeHost(buffer.h_staging_buffer);
        if (buffer.stream) cudaStreamDestroy(buffer.stream);
        if (buffer.send_event) cudaEventDestroy(buffer.send_event);
        if (buffer.recv_event) cudaEventDestroy(buffer.recv_event);
    }
    comm_buffers.clear();
}

void MultiGPUManager::updateStats() {
    auto current_time = std::chrono::high_resolution_clock::now();
    if (current_step == 0) {
        start_time = current_time;
    }

    // Update simulation time
    auto elapsed = std::chrono::duration<float>(current_time - start_time);
    stats.total_simulation_time = elapsed.count();

    // Update particle counts
    for (size_t i = 0; i < partitions.size(); ++i) {
        stats.particle_counts[i] = partitions[i].particle_count;
    }

    // Compute load balance factor
    if (!stats.particle_counts.empty()) {
        size_t max_particles = *std::max_element(stats.particle_counts.begin(), stats.particle_counts.end());
        size_t min_particles = *std::min_element(stats.particle_counts.begin(), stats.particle_counts.end());

        if (min_particles > 0) {
            stats.load_balance_factor = static_cast<float>(max_particles) / static_cast<float>(min_particles);
        }
    }

    current_step++;
}

void MultiGPUManager::printStats() const {
    std::cout << "\n=== Multi-GPU Performance Statistics ===" << std::endl;
    std::cout << "Simulation time: " << stats.total_simulation_time << " seconds" << std::endl;
    std::cout << "Load balance factor: " << stats.load_balance_factor << std::endl;
    std::cout << "Communication overhead: " << (stats.communication_overhead * 100.0f) << "%" << std::endl;
    std::cout << "Rebalance count: " << stats.rebalance_count << std::endl;

    std::cout << "\nPer-GPU Statistics:" << std::endl;
    for (size_t i = 0; i < config.device_ids.size(); ++i) {
        std::cout << "GPU " << config.device_ids[i]
                  << ": " << stats.particle_counts[i] << " particles"
                  << ", " << stats.computation_times[i] << "ms compute"
                  << ", " << stats.communication_times[i] << "ms comm"
                  << ", " << (stats.memory_usage[i] / (1024*1024)) << " MB memory" << std::endl;
    }
}

// MultiGPUUtils Implementation
namespace MultiGPUUtils {

std::vector<int> getAvailableGPUs() {
    std::vector<int> gpus;

    int device_count;
    cudaGetDeviceCount(&device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // Check if device supports compute capability 3.5 or higher
        if (prop.major >= 3 && (prop.major > 3 || prop.minor >= 5)) {
            gpus.push_back(i);
        }
    }

    return gpus;
}

std::vector<int> selectOptimalGPUs(int desired_count, float min_memory_gb) {
    std::vector<int> all_gpus = getAvailableGPUs();
    std::vector<std::pair<int, float>> gpu_scores;

    size_t min_memory_bytes = static_cast<size_t>(min_memory_gb * 1024 * 1024 * 1024);

    for (int gpu_id : all_gpus) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpu_id);

        if (prop.totalGlobalMem >= min_memory_bytes) {
            // Score based on memory and compute capability
            float score = static_cast<float>(prop.totalGlobalMem / (1024*1024*1024)) +
                         (prop.major * 100 + prop.minor) * 0.1f;
            gpu_scores.emplace_back(gpu_id, score);
        }
    }

    // Sort by score (descending)
    std::sort(gpu_scores.begin(), gpu_scores.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<int> selected;
    for (int i = 0; i < std::min(desired_count, static_cast<int>(gpu_scores.size())); ++i) {
        selected.push_back(gpu_scores[i].first);
    }

    return selected;
}

void validateConfig(const MultiGPUConfig& config) {
    if (config.device_ids.empty()) {
        throw std::runtime_error("No GPU devices specified in config");
    }

    int device_count;
    cudaGetDeviceCount(&device_count);

    for (int device_id : config.device_ids) {
        if (device_id < 0 || device_id >= device_count) {
            throw std::runtime_error("Invalid device ID: " + std::to_string(device_id));
        }
    }

    if (config.ghost_layer_width <= 0.0f) {
        throw std::runtime_error("Ghost layer width must be positive");
    }

    if (config.load_balance_threshold < 0.0f || config.load_balance_threshold > 1.0f) {
        throw std::runtime_error("Load balance threshold must be between 0 and 1");
    }
}

} // namespace MultiGPUUtils

} // namespace physgrad