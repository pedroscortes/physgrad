#include "multi_gpu.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

namespace physgrad {

// Load Balancer class for heterogeneous GPU setups
class LoadBalancer {
private:
    MultiGPUConfig config;
    std::vector<GPUDeviceInfo> gpu_info;
    std::vector<float> gpu_weights;
    std::vector<float> current_loads;
    std::vector<float> target_loads;

    // Performance history for adaptive balancing
    std::vector<std::vector<float>> computation_history;
    std::vector<std::vector<float>> communication_history;
    int history_size = 10;

public:
    LoadBalancer(const MultiGPUConfig& config, const std::vector<GPUDeviceInfo>& gpu_info)
        : config(config), gpu_info(gpu_info) {

        gpu_weights.resize(gpu_info.size());
        current_loads.resize(gpu_info.size());
        target_loads.resize(gpu_info.size());

        computation_history.resize(gpu_info.size());
        communication_history.resize(gpu_info.size());

        for (auto& history : computation_history) {
            history.reserve(history_size);
        }
        for (auto& history : communication_history) {
            history.reserve(history_size);
        }

        calculateInitialWeights();
    }

    void calculateInitialWeights() {
        std::cout << "Calculating initial GPU weights for load balancing..." << std::endl;

        float total_weight = 0.0f;

        for (size_t i = 0; i < gpu_info.size(); ++i) {
            const auto& info = gpu_info[i];

            // Base weight from relative performance
            float weight = info.relative_performance;

            // Adjust for memory capacity
            float memory_gb = static_cast<float>(info.free_memory) / (1024.0f * 1024.0f * 1024.0f);
            float memory_factor = std::min(1.0f + (memory_gb - 4.0f) * 0.1f, 2.0f); // Bonus for >4GB
            weight *= memory_factor;

            // Adjust for compute capability
            float compute_factor = 1.0f;
            if (info.compute_capability_major >= 8) {
                compute_factor = 1.2f; // Ampere and newer
            } else if (info.compute_capability_major >= 7) {
                compute_factor = 1.1f; // Turing/Volta
            } else if (info.compute_capability_major >= 6) {
                compute_factor = 1.0f; // Pascal
            } else {
                compute_factor = 0.8f; // Older architectures
            }
            weight *= compute_factor;

            gpu_weights[i] = weight;
            total_weight += weight;

            std::cout << "GPU " << info.device_id << " (" << info.name << "): weight = "
                      << weight << " (perf=" << info.relative_performance
                      << ", mem=" << memory_factor << ", compute=" << compute_factor << ")" << std::endl;
        }

        // Normalize weights
        for (float& weight : gpu_weights) {
            weight /= total_weight;
        }

        std::cout << "Normalized GPU weights: ";
        for (size_t i = 0; i < gpu_weights.size(); ++i) {
            std::cout << gpu_weights[i] << " ";
        }
        std::cout << std::endl;
    }

    void updatePerformanceHistory(const std::vector<float>& computation_times,
                                const std::vector<float>& communication_times) {
        for (size_t i = 0; i < gpu_info.size(); ++i) {
            // Add new measurements
            computation_history[i].push_back(computation_times[i]);
            communication_history[i].push_back(communication_times[i]);

            // Keep only recent history
            if (computation_history[i].size() > history_size) {
                computation_history[i].erase(computation_history[i].begin());
            }
            if (communication_history[i].size() > history_size) {
                communication_history[i].erase(communication_history[i].begin());
            }
        }

        // Recalculate weights based on actual performance
        adaptWeights();
    }

    void adaptWeights() {
        if (computation_history[0].size() < 3) {
            return; // Need more history
        }

        std::vector<float> avg_computation(gpu_info.size());
        std::vector<float> avg_communication(gpu_info.size());

        // Calculate average performance over recent history
        for (size_t i = 0; i < gpu_info.size(); ++i) {
            avg_computation[i] = std::accumulate(computation_history[i].begin(),
                                               computation_history[i].end(), 0.0f) /
                               computation_history[i].size();

            avg_communication[i] = std::accumulate(communication_history[i].begin(),
                                                 communication_history[i].end(), 0.0f) /
                                 communication_history[i].size();
        }

        // Find the fastest GPU (baseline)
        float min_total_time = std::numeric_limits<float>::max();
        for (size_t i = 0; i < gpu_info.size(); ++i) {
            float total_time = avg_computation[i] + avg_communication[i];
            min_total_time = std::min(min_total_time, total_time);
        }

        // Update weights based on relative performance
        float total_new_weight = 0.0f;
        for (size_t i = 0; i < gpu_info.size(); ++i) {
            float total_time = avg_computation[i] + avg_communication[i];
            float relative_speed = min_total_time / total_time;

            // Smooth adaptation to avoid oscillation
            float adaptation_rate = 0.1f;
            gpu_weights[i] = gpu_weights[i] * (1.0f - adaptation_rate) +
                           relative_speed * adaptation_rate;

            total_new_weight += gpu_weights[i];
        }

        // Renormalize
        for (float& weight : gpu_weights) {
            weight /= total_new_weight;
        }

        std::cout << "Adapted GPU weights: ";
        for (size_t i = 0; i < gpu_weights.size(); ++i) {
            std::cout << gpu_weights[i] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<size_t> redistributeParticles(size_t total_particles,
                                            const std::vector<size_t>& current_distribution) {
        current_loads.assign(current_distribution.begin(), current_distribution.end());

        // Calculate target distribution based on weights
        std::vector<size_t> target_distribution(gpu_info.size());
        size_t assigned_particles = 0;

        for (size_t i = 0; i < gpu_info.size(); ++i) {
            if (i == gpu_info.size() - 1) {
                // Last GPU gets remaining particles
                target_distribution[i] = total_particles - assigned_particles;
            } else {
                target_distribution[i] = static_cast<size_t>(total_particles * gpu_weights[i]);
                assigned_particles += target_distribution[i];
            }
            target_loads[i] = static_cast<float>(target_distribution[i]);
        }

        return target_distribution;
    }

    bool shouldRebalance(const std::vector<size_t>& current_distribution) const {
        if (current_distribution.size() != gpu_weights.size()) {
            return false;
        }

        // Calculate current load imbalance
        float load_imbalance = calculateLoadImbalance(current_distribution);

        return load_imbalance > config.load_balance_threshold;
    }

    float calculateLoadImbalance(const std::vector<size_t>& distribution) const {
        if (distribution.empty()) return 0.0f;

        // Calculate weighted load distribution
        std::vector<float> weighted_loads(distribution.size());
        for (size_t i = 0; i < distribution.size(); ++i) {
            weighted_loads[i] = static_cast<float>(distribution[i]) / gpu_weights[i];
        }

        auto [min_it, max_it] = std::minmax_element(weighted_loads.begin(), weighted_loads.end());

        if (*min_it == 0.0f) return std::numeric_limits<float>::max();

        return (*max_it - *min_it) / *min_it;
    }

    // Migration plan for rebalancing
    struct MigrationPlan {
        struct Transfer {
            int src_gpu;
            int dst_gpu;
            size_t particle_count;
            std::vector<int> particle_indices;
        };
        std::vector<Transfer> transfers;
        float expected_improvement;
    };

    MigrationPlan createMigrationPlan(const std::vector<size_t>& current_distribution,
                                    const std::vector<size_t>& target_distribution,
                                    const std::vector<DomainPartition>& partitions) {
        MigrationPlan plan;

        // Calculate excess and deficit for each GPU
        std::vector<int> balance(gpu_info.size());
        for (size_t i = 0; i < gpu_info.size(); ++i) {
            balance[i] = static_cast<int>(target_distribution[i]) - static_cast<int>(current_distribution[i]);
        }

        // Create transfers from excess to deficit GPUs
        for (size_t src_gpu = 0; src_gpu < gpu_info.size(); ++src_gpu) {
            if (balance[src_gpu] <= 0) continue; // No excess

            for (size_t dst_gpu = 0; dst_gpu < gpu_info.size(); ++dst_gpu) {
                if (balance[dst_gpu] >= 0) continue; // No deficit
                if (src_gpu == dst_gpu) continue;

                // Calculate how many particles to transfer
                int transfer_count = std::min(balance[src_gpu], -balance[dst_gpu]);
                if (transfer_count <= 0) continue;

                // Select particles to transfer (prefer boundary particles)
                std::vector<int> particles_to_transfer;
                selectParticlesForMigration(partitions[src_gpu], partitions[dst_gpu],
                                          transfer_count, particles_to_transfer);

                if (!particles_to_transfer.empty()) {
                    MigrationPlan::Transfer transfer;
                    transfer.src_gpu = static_cast<int>(src_gpu);
                    transfer.dst_gpu = static_cast<int>(dst_gpu);
                    transfer.particle_count = particles_to_transfer.size();
                    transfer.particle_indices = std::move(particles_to_transfer);

                    plan.transfers.push_back(transfer);

                    // Update balances
                    balance[src_gpu] -= static_cast<int>(transfer.particle_count);
                    balance[dst_gpu] += static_cast<int>(transfer.particle_count);
                }
            }
        }

        // Calculate expected improvement
        float current_imbalance = calculateLoadImbalance(current_distribution);

        std::vector<size_t> projected_distribution = current_distribution;
        for (const auto& transfer : plan.transfers) {
            projected_distribution[transfer.src_gpu] -= transfer.particle_count;
            projected_distribution[transfer.dst_gpu] += transfer.particle_count;
        }

        float projected_imbalance = calculateLoadImbalance(projected_distribution);
        plan.expected_improvement = current_imbalance - projected_imbalance;

        return plan;
    }

    void selectParticlesForMigration(const DomainPartition& src_partition,
                                   const DomainPartition& dst_partition,
                                   int count,
                                   std::vector<int>& selected_particles) {
        selected_particles.clear();

        // Prefer particles near the boundary between src and dst partitions
        std::vector<std::pair<float, int>> particle_distances;

        // Calculate center points of both partitions
        float src_center[3], dst_center[3];
        for (int i = 0; i < 3; ++i) {
            src_center[i] = (src_partition.bounds_min[i] + src_partition.bounds_max[i]) * 0.5f;
            dst_center[i] = (dst_partition.bounds_min[i] + dst_partition.bounds_max[i]) * 0.5f;
        }

        // For each particle in src partition, calculate distance from line connecting centers
        for (int particle_idx : src_partition.particle_indices) {
            // Simplified: use particle index as proxy for position
            // In real implementation, would use actual particle positions
            float distance_score = static_cast<float>(particle_idx % 1000) / 1000.0f;
            particle_distances.emplace_back(distance_score, particle_idx);
        }

        // Sort by distance (closer to boundary = lower score)
        std::sort(particle_distances.begin(), particle_distances.end());

        // Select the closest particles
        int select_count = std::min(count, static_cast<int>(particle_distances.size()));
        selected_particles.reserve(select_count);

        for (int i = 0; i < select_count; ++i) {
            selected_particles.push_back(particle_distances[i].second);
        }
    }

    const std::vector<float>& getGPUWeights() const { return gpu_weights; }
    const std::vector<float>& getCurrentLoads() const { return current_loads; }
    const std::vector<float>& getTargetLoads() const { return target_loads; }
};

// MultiGPUManager load balancing methods
void MultiGPUManager::updateLoadBalance() {
    if (!config.enable_dynamic_balancing) return;
    if (current_step % config.rebalance_frequency != 0) return;

    if (!load_balancer) {
        load_balancer = std::make_unique<LoadBalancer>(config, gpu_info);
    }

    // Update performance history
    load_balancer->updatePerformanceHistory(stats.computation_times, stats.communication_times);

    // Check if rebalancing is needed
    if (shouldRebalance()) {
        executeLoadBalancing();
    }
}

bool MultiGPUManager::shouldRebalance() const {
    if (!load_balancer) return false;

    std::vector<size_t> current_distribution;
    for (const auto& partition : partitions) {
        current_distribution.push_back(partition.particle_count);
    }

    return load_balancer->shouldRebalance(current_distribution);
}

void MultiGPUManager::executeLoadBalancing() {
    if (!load_balancer) return;

    std::cout << "Executing load balancing..." << std::endl;

    // Get current distribution
    std::vector<size_t> current_distribution;
    size_t total_particles = 0;
    for (const auto& partition : partitions) {
        current_distribution.push_back(partition.particle_count);
        total_particles += partition.particle_count;
    }

    // Calculate target distribution
    auto target_distribution = load_balancer->redistributeParticles(total_particles, current_distribution);

    // Create migration plan
    auto migration_plan = load_balancer->createMigrationPlan(current_distribution, target_distribution, partitions);

    std::cout << "Migration plan: " << migration_plan.transfers.size() << " transfers, "
              << "expected improvement: " << migration_plan.expected_improvement << std::endl;

    // Execute migrations if beneficial
    if (migration_plan.expected_improvement > 0.01f) { // 1% improvement threshold
        executeMigrationPlan(migration_plan);
        stats.rebalance_count++;
    }
}

void MultiGPUManager::executeMigrationPlan(const LoadBalancer::MigrationPlan& plan) {
    for (const auto& transfer : plan.transfers) {
        std::cout << "Migrating " << transfer.particle_count << " particles from GPU "
                  << transfer.src_gpu << " to GPU " << transfer.dst_gpu << std::endl;

        // Execute the transfer
        sendParticleData(transfer.src_gpu, transfer.dst_gpu, transfer.particle_indices);

        // Update partition information
        auto& src_partition = partitions[transfer.src_gpu];
        auto& dst_partition = partitions[transfer.dst_gpu];

        // Remove particles from source
        for (int particle_idx : transfer.particle_indices) {
            auto it = std::find(src_partition.particle_indices.begin(),
                              src_partition.particle_indices.end(), particle_idx);
            if (it != src_partition.particle_indices.end()) {
                src_partition.particle_indices.erase(it);
            }
        }

        // Add particles to destination
        dst_partition.particle_indices.insert(dst_partition.particle_indices.end(),
                                             transfer.particle_indices.begin(),
                                             transfer.particle_indices.end());

        // Update counts
        src_partition.particle_count = src_partition.particle_indices.size();
        dst_partition.particle_count = dst_partition.particle_indices.size();
    }

    // Synchronize all GPUs after migration
    synchronizeGPUs();
}

float MultiGPUManager::computeLoadImbalance() const {
    if (!load_balancer) return 0.0f;

    std::vector<size_t> current_distribution;
    for (const auto& partition : partitions) {
        current_distribution.push_back(partition.particle_count);
    }

    return load_balancer->calculateLoadImbalance(current_distribution);
}

// Utility functions for load balancing
namespace MultiGPUUtils {

std::vector<size_t> distributeParticles(size_t total_particles, const std::vector<float>& gpu_weights) {
    std::vector<size_t> distribution(gpu_weights.size());
    size_t assigned = 0;

    for (size_t i = 0; i < gpu_weights.size(); ++i) {
        if (i == gpu_weights.size() - 1) {
            distribution[i] = total_particles - assigned;
        } else {
            distribution[i] = static_cast<size_t>(total_particles * gpu_weights[i]);
            assigned += distribution[i];
        }
    }

    return distribution;
}

float predictCommunicationOverhead(const MultiGPUConfig& config, size_t particle_count) {
    // Simplified model for communication overhead prediction
    float base_overhead = 0.05f; // 5% base overhead

    // Increase with number of GPUs (more communication partners)
    float gpu_factor = 1.0f + (config.device_ids.size() - 1) * 0.02f;

    // Increase with ghost layer width (more boundary particles)
    float ghost_factor = 1.0f + config.ghost_layer_width * 0.01f;

    // Decrease with larger particle counts (amortized costs)
    float scale_factor = 1.0f / (1.0f + particle_count / 100000.0f);

    return base_overhead * gpu_factor * ghost_factor * scale_factor;
}

float estimateSpeedup(int num_gpus, size_t particle_count, const MultiGPUConfig& config) {
    // Amdahl's law with communication overhead
    float communication_overhead = predictCommunicationOverhead(config, particle_count);
    float parallel_fraction = 1.0f - communication_overhead;

    float ideal_speedup = 1.0f / ((1.0f - parallel_fraction) + parallel_fraction / num_gpus);

    // Efficiency factor for non-ideal scaling
    float efficiency = 0.9f - (num_gpus - 1) * 0.05f; // Decrease with more GPUs
    efficiency = std::max(0.5f, efficiency);

    return ideal_speedup * efficiency;
}

} // namespace MultiGPUUtils

} // namespace physgrad