#include "multi_gpu.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#include <algorithm>
#include <iostream>

namespace physgrad {

// Particle data structure for communication
struct ParticleData {
    float pos_x, pos_y, pos_z;
    float vel_x, vel_y, vel_z;
    float mass;
    int original_index;
};

// CUDA kernels for particle packing/unpacking
__global__ void packParticleData(
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* vel_x, const float* vel_y, const float* vel_z,
    const float* masses, const int* indices, int count,
    ParticleData* packed_data
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int particle_idx = indices[idx];
        ParticleData& data = packed_data[idx];

        data.pos_x = pos_x[particle_idx];
        data.pos_y = pos_y[particle_idx];
        data.pos_z = pos_z[particle_idx];
        data.vel_x = vel_x[particle_idx];
        data.vel_y = vel_y[particle_idx];
        data.vel_z = vel_z[particle_idx];
        data.mass = masses[particle_idx];
        data.original_index = particle_idx;
    }
}

__global__ void unpackParticleData(
    const ParticleData* packed_data, int count, int offset,
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    float* masses, int* indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const ParticleData& data = packed_data[idx];
        int dest_idx = offset + idx;

        pos_x[dest_idx] = data.pos_x;
        pos_y[dest_idx] = data.pos_y;
        pos_z[dest_idx] = data.pos_z;
        vel_x[dest_idx] = data.vel_x;
        vel_y[dest_idx] = data.vel_y;
        vel_z[dest_idx] = data.vel_z;
        masses[dest_idx] = data.mass;
        indices[dest_idx] = data.original_index;
    }
}

// Communication patterns implementation
void MultiGPUManager::exchangeGhostParticles() {
    if (!initialized) return;

    auto start_time = std::chrono::high_resolution_clock::now();

    switch (config.communication) {
        case CommunicationPattern::NCCL_COLLECTIVE:
            exchangeGhostParticlesNCCL();
            break;
        case CommunicationPattern::PEER_TO_PEER:
            exchangeGhostParticlesP2P();
            break;
        case CommunicationPattern::HOST_STAGING:
            exchangeGhostParticlesHostStaging();
            break;
        case CommunicationPattern::UNIFIED_MEMORY:
            exchangeGhostParticlesUnifiedMemory();
            break;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(end_time - start_time);

    // Update communication time statistics
    for (size_t i = 0; i < stats.communication_times.size(); ++i) {
        stats.communication_times[i] += duration.count() / stats.communication_times.size();
    }
}

void MultiGPUManager::exchangeGhostParticlesNCCL() {
    if (!nccl_comms) return;

    std::cout << "Exchanging ghost particles using NCCL..." << std::endl;

    // Prepare send/receive counts for each GPU
    std::vector<std::vector<int>> send_counts(config.device_ids.size(),
                                             std::vector<int>(config.device_ids.size(), 0));
    std::vector<std::vector<int>> recv_counts(config.device_ids.size(),
                                             std::vector<int>(config.device_ids.size(), 0));

    // Calculate how many particles each GPU needs to send to each other GPU
    for (size_t src_gpu = 0; src_gpu < partitions.size(); ++src_gpu) {
        for (size_t dst_gpu = 0; dst_gpu < partitions.size(); ++dst_gpu) {
            if (src_gpu == dst_gpu) continue;

            const auto& src_partition = partitions[src_gpu];
            const auto& dst_partition = partitions[dst_gpu];

            // Check if these partitions are neighbors
            bool are_neighbors = std::find(src_partition.neighbor_partitions.begin(),
                                         src_partition.neighbor_partitions.end(),
                                         static_cast<int>(dst_gpu)) != src_partition.neighbor_partitions.end();

            if (are_neighbors) {
                // Count particles in src that need to be sent to dst as ghosts
                int count = 0;
                for (int particle_idx : src_partition.particle_indices) {
                    // Check if particle is within ghost layer of dst_partition
                    // This is a simplified check - in practice, you'd get actual positions
                    count++; // Placeholder
                }
                send_counts[src_gpu][dst_gpu] = std::min(count, config.max_particles_per_transfer);
                recv_counts[dst_gpu][src_gpu] = send_counts[src_gpu][dst_gpu];
            }
        }
    }

    // Execute NCCL all-to-all communication
    for (size_t gpu_idx = 0; gpu_idx < config.device_ids.size(); ++gpu_idx) {
        cudaSetDevice(config.device_ids[gpu_idx]);

        // Prepare send buffer
        ParticleData* send_buffer = static_cast<ParticleData*>(comm_buffers[gpu_idx].d_send_buffer);
        ParticleData* recv_buffer = static_cast<ParticleData*>(comm_buffers[gpu_idx].d_recv_buffer);

        // Pack particle data for sending
        for (size_t dst_gpu = 0; dst_gpu < config.device_ids.size(); ++dst_gpu) {
            if (gpu_idx == dst_gpu) continue;

            int count = send_counts[gpu_idx][dst_gpu];
            if (count > 0) {
                // TODO: Get actual particle data from GPU context
                // For now, we'll use a placeholder implementation
            }
        }

        // Execute NCCL AlltoAll
        size_t send_size = sizeof(ParticleData) * config.max_particles_per_transfer;
        size_t recv_size = sizeof(ParticleData) * config.max_particles_per_transfer;

        ncclAllToAll(send_buffer, recv_buffer, send_size, ncclInt8,
                    nccl_comms[gpu_idx], gpu_contexts[gpu_idx]->getCommunicationStream());
    }

    // Synchronize all GPUs
    synchronizeGPUs();

    std::cout << "NCCL ghost particle exchange complete" << std::endl;
}

void MultiGPUManager::exchangeGhostParticlesP2P() {
    std::cout << "Exchanging ghost particles using P2P..." << std::endl;

    // Create events for synchronization
    std::vector<std::vector<cudaEvent_t>> send_events(config.device_ids.size());
    std::vector<std::vector<cudaEvent_t>> recv_events(config.device_ids.size());

    for (size_t i = 0; i < config.device_ids.size(); ++i) {
        send_events[i].resize(config.device_ids.size());
        recv_events[i].resize(config.device_ids.size());

        cudaSetDevice(config.device_ids[i]);
        for (size_t j = 0; j < config.device_ids.size(); ++j) {
            if (i != j) {
                cudaEventCreate(&send_events[i][j]);
                cudaEventCreate(&recv_events[i][j]);
            }
        }
    }

    // Execute peer-to-peer transfers
    for (size_t src_gpu = 0; src_gpu < config.device_ids.size(); ++src_gpu) {
        cudaSetDevice(config.device_ids[src_gpu]);

        for (size_t dst_gpu = 0; dst_gpu < config.device_ids.size(); ++dst_gpu) {
            if (src_gpu == dst_gpu) continue;

            const auto& src_partition = partitions[src_gpu];

            // Check if these partitions are neighbors
            bool are_neighbors = std::find(src_partition.neighbor_partitions.begin(),
                                         src_partition.neighbor_partitions.end(),
                                         static_cast<int>(dst_gpu)) != src_partition.neighbor_partitions.end();

            if (are_neighbors) {
                // Prepare data for transfer
                ParticleData* src_buffer = static_cast<ParticleData*>(comm_buffers[src_gpu].d_send_buffer);
                ParticleData* dst_buffer = static_cast<ParticleData*>(comm_buffers[dst_gpu].d_recv_buffer);

                // Calculate transfer size
                size_t transfer_particles = std::min(
                    static_cast<size_t>(config.max_particles_per_transfer),
                    src_partition.particle_indices.size()
                );

                size_t transfer_size = transfer_particles * sizeof(ParticleData);

                // Enable peer access if not already done
                int can_access;
                cudaDeviceCanAccessPeer(&can_access, config.device_ids[src_gpu], config.device_ids[dst_gpu]);

                if (can_access) {
                    // Direct P2P copy
                    cudaMemcpyPeerAsync(dst_buffer, config.device_ids[dst_gpu],
                                      src_buffer, config.device_ids[src_gpu],
                                      transfer_size,
                                      gpu_contexts[src_gpu]->getCommunicationStream());

                    cudaEventRecord(send_events[src_gpu][dst_gpu],
                                  gpu_contexts[src_gpu]->getCommunicationStream());
                } else {
                    // Fallback to host staging
                    void* host_buffer = comm_buffers[src_gpu].h_staging_buffer;
                    if (host_buffer) {
                        cudaMemcpyAsync(host_buffer, src_buffer, transfer_size,
                                      cudaMemcpyDeviceToHost,
                                      gpu_contexts[src_gpu]->getCommunicationStream());
                        cudaStreamSynchronize(gpu_contexts[src_gpu]->getCommunicationStream());

                        cudaSetDevice(config.device_ids[dst_gpu]);
                        cudaMemcpyAsync(dst_buffer, host_buffer, transfer_size,
                                      cudaMemcpyHostToDevice,
                                      gpu_contexts[dst_gpu]->getCommunicationStream());
                    }
                }
            }
        }
    }

    // Wait for all transfers to complete
    for (size_t i = 0; i < config.device_ids.size(); ++i) {
        cudaSetDevice(config.device_ids[i]);
        for (size_t j = 0; j < config.device_ids.size(); ++j) {
            if (i != j) {
                cudaEventSynchronize(send_events[i][j]);
            }
        }
    }

    // Cleanup events
    for (size_t i = 0; i < config.device_ids.size(); ++i) {
        cudaSetDevice(config.device_ids[i]);
        for (size_t j = 0; j < config.device_ids.size(); ++j) {
            if (i != j) {
                cudaEventDestroy(send_events[i][j]);
                cudaEventDestroy(recv_events[i][j]);
            }
        }
    }

    std::cout << "P2P ghost particle exchange complete" << std::endl;
}

void MultiGPUManager::exchangeGhostParticlesHostStaging() {
    std::cout << "Exchanging ghost particles using host staging..." << std::endl;

    // Allocate temporary host buffers for each GPU
    std::vector<std::vector<ParticleData>> host_send_data(config.device_ids.size());
    std::vector<std::vector<ParticleData>> host_recv_data(config.device_ids.size());

    // Phase 1: Copy data from GPUs to host
    for (size_t gpu_idx = 0; gpu_idx < config.device_ids.size(); ++gpu_idx) {
        cudaSetDevice(config.device_ids[gpu_idx]);

        const auto& partition = partitions[gpu_idx];
        size_t particle_count = std::min(
            static_cast<size_t>(config.max_particles_per_transfer),
            partition.particle_indices.size()
        );

        host_send_data[gpu_idx].resize(particle_count);

        if (particle_count > 0) {
            ParticleData* device_buffer = static_cast<ParticleData*>(comm_buffers[gpu_idx].d_send_buffer);

            // Pack data on GPU first (would need actual GPU data pointers)
            // Then copy to host
            cudaMemcpyAsync(host_send_data[gpu_idx].data(), device_buffer,
                          particle_count * sizeof(ParticleData),
                          cudaMemcpyDeviceToHost,
                          gpu_contexts[gpu_idx]->getCommunicationStream());
        }
    }

    // Synchronize all streams
    for (size_t gpu_idx = 0; gpu_idx < config.device_ids.size(); ++gpu_idx) {
        cudaSetDevice(config.device_ids[gpu_idx]);
        cudaStreamSynchronize(gpu_contexts[gpu_idx]->getCommunicationStream());
    }

    // Phase 2: Host-side data redistribution
    for (size_t src_gpu = 0; src_gpu < config.device_ids.size(); ++src_gpu) {
        for (size_t dst_gpu = 0; dst_gpu < config.device_ids.size(); ++dst_gpu) {
            if (src_gpu == dst_gpu) continue;

            const auto& src_partition = partitions[src_gpu];
            bool are_neighbors = std::find(src_partition.neighbor_partitions.begin(),
                                         src_partition.neighbor_partitions.end(),
                                         static_cast<int>(dst_gpu)) != src_partition.neighbor_partitions.end();

            if (are_neighbors) {
                // Filter particles that need to be sent as ghosts
                for (const auto& particle : host_send_data[src_gpu]) {
                    const auto& dst_partition = partitions[dst_gpu];

                    // Check if particle is within ghost layer of destination
                    bool in_ghost_layer =
                        particle.pos_x >= (dst_partition.bounds_min[0] - config.ghost_layer_width) &&
                        particle.pos_x <= (dst_partition.bounds_max[0] + config.ghost_layer_width) &&
                        particle.pos_y >= (dst_partition.bounds_min[1] - config.ghost_layer_width) &&
                        particle.pos_y <= (dst_partition.bounds_max[1] + config.ghost_layer_width) &&
                        particle.pos_z >= (dst_partition.bounds_min[2] - config.ghost_layer_width) &&
                        particle.pos_z <= (dst_partition.bounds_max[2] + config.ghost_layer_width);

                    if (in_ghost_layer) {
                        host_recv_data[dst_gpu].push_back(particle);
                    }
                }
            }
        }
    }

    // Phase 3: Copy redistributed data back to GPUs
    for (size_t gpu_idx = 0; gpu_idx < config.device_ids.size(); ++gpu_idx) {
        cudaSetDevice(config.device_ids[gpu_idx]);

        if (!host_recv_data[gpu_idx].empty()) {
            ParticleData* device_buffer = static_cast<ParticleData*>(comm_buffers[gpu_idx].d_recv_buffer);
            size_t recv_count = std::min(host_recv_data[gpu_idx].size(),
                                       static_cast<size_t>(config.max_particles_per_transfer));

            cudaMemcpyAsync(device_buffer, host_recv_data[gpu_idx].data(),
                          recv_count * sizeof(ParticleData),
                          cudaMemcpyHostToDevice,
                          gpu_contexts[gpu_idx]->getCommunicationStream());
        }
    }

    // Final synchronization
    synchronizeGPUs();

    std::cout << "Host staging ghost particle exchange complete" << std::endl;
}

void MultiGPUManager::exchangeGhostParticlesUnifiedMemory() {
    std::cout << "Exchanging ghost particles using unified memory..." << std::endl;

    // With unified memory, ghost particles can be accessed directly
    // without explicit transfers. We just need to ensure proper
    // memory prefetching and coherency.

    for (size_t gpu_idx = 0; gpu_idx < config.device_ids.size(); ++gpu_idx) {
        cudaSetDevice(config.device_ids[gpu_idx]);

        const auto& partition = partitions[gpu_idx];

        // Prefetch ghost particle data to this GPU
        for (int neighbor_idx : partition.neighbor_partitions) {
            const auto& neighbor_partition = partitions[neighbor_idx];

            // Calculate size to prefetch
            size_t ghost_size = neighbor_partition.ghost_count * sizeof(ParticleData);

            if (ghost_size > 0) {
                // Prefetch neighbor's particle data
                // In a real implementation, this would prefetch the actual unified memory regions
                cudaMemPrefetchAsync(comm_buffers[neighbor_idx].d_send_buffer, ghost_size,
                                   config.device_ids[gpu_idx],
                                   gpu_contexts[gpu_idx]->getCommunicationStream());
            }
        }
    }

    synchronizeGPUs();

    std::cout << "Unified memory ghost particle exchange complete" << std::endl;
}

void MultiGPUManager::synchronizeGPUs() {
    for (size_t i = 0; i < config.device_ids.size(); ++i) {
        cudaSetDevice(config.device_ids[i]);
        cudaStreamSynchronize(gpu_contexts[i]->getComputationStream());
        cudaStreamSynchronize(gpu_contexts[i]->getCommunicationStream());
    }
}

// Helper methods for communication
void MultiGPUManager::sendParticleData(int src_gpu, int dst_gpu, const std::vector<int>& particle_indices) {
    if (src_gpu == dst_gpu || particle_indices.empty()) return;

    cudaSetDevice(config.device_ids[src_gpu]);

    // Get GPU contexts
    auto& src_context = gpu_contexts[src_gpu];
    size_t transfer_count = std::min(particle_indices.size(),
                                   static_cast<size_t>(config.max_particles_per_transfer));

    // Pack particle data
    ParticleData* send_buffer = static_cast<ParticleData*>(comm_buffers[src_gpu].d_send_buffer);

    // Launch packing kernel
    int block_size = 256;
    int grid_size = (transfer_count + block_size - 1) / block_size;

    // Create device arrays for indices
    int* d_indices;
    cudaMalloc(&d_indices, transfer_count * sizeof(int));
    cudaMemcpy(d_indices, particle_indices.data(), transfer_count * sizeof(int), cudaMemcpyHostToDevice);

    packParticleData<<<grid_size, block_size, 0, src_context->getCommunicationStream()>>>(
        src_context->getPosX(), src_context->getPosY(), src_context->getPosZ(),
        src_context->getVelX(), src_context->getVelY(), src_context->getVelZ(),
        src_context->getMasses(), d_indices, static_cast<int>(transfer_count), send_buffer
    );

    cudaFree(d_indices);

    // Record event for synchronization
    cudaEventRecord(comm_buffers[src_gpu].send_event, src_context->getCommunicationStream());
}

void MultiGPUManager::receiveParticleData(int gpu_id, int src_gpu, std::vector<int>& particle_indices) {
    if (gpu_id == src_gpu) return;

    cudaSetDevice(config.device_ids[gpu_id]);

    auto& context = gpu_contexts[gpu_id];
    ParticleData* recv_buffer = static_cast<ParticleData*>(comm_buffers[gpu_id].d_recv_buffer);

    // Wait for send to complete
    cudaEventSynchronize(comm_buffers[src_gpu].send_event);

    // Unpack received data
    size_t recv_count = std::min(particle_indices.size(),
                               static_cast<size_t>(config.max_particles_per_transfer));

    int block_size = 256;
    int grid_size = (recv_count + block_size - 1) / block_size;

    // Create device array for indices
    int* d_indices;
    cudaMalloc(&d_indices, recv_count * sizeof(int));

    size_t ghost_offset = context->getAllocatedParticles(); // Append to existing particles

    unpackParticleData<<<grid_size, block_size, 0, context->getCommunicationStream()>>>(
        recv_buffer, static_cast<int>(recv_count), static_cast<int>(ghost_offset),
        context->getPosX(), context->getPosY(), context->getPosZ(),
        context->getVelX(), context->getVelY(), context->getVelZ(),
        context->getMasses(), d_indices
    );

    // Copy indices back to host
    particle_indices.resize(recv_count);
    cudaMemcpy(particle_indices.data(), d_indices, recv_count * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_indices);

    // Record completion
    cudaEventRecord(comm_buffers[gpu_id].recv_event, context->getCommunicationStream());
}

} // namespace physgrad