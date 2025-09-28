#pragma once

#include "variational_contact.h"
#include "variational_contact_gpu.cuh"
#include <memory>
#include <vector>

namespace physgrad {

// GPU-accelerated variational contact solver with identical API to CPU version
class VariationalContactSolverGPU {
private:
    VariationalContactParams params;
    std::unique_ptr<VariationalContactGPUData> gpu_data;

    // Performance tracking
    struct GPUPerformanceMetrics {
        float contact_detection_ms;
        float force_computation_ms;
        float newton_solver_ms;
        float gradient_computation_ms;
        float memory_transfer_ms;
        int cuda_cores_utilized;
        float memory_bandwidth_gbps;
        float compute_throughput_gflops;
    };

    mutable GPUPerformanceMetrics last_metrics;

    // GPU solver state
    bool gpu_initialized;
    int current_n_bodies;

    // Host-side cache for frequent CPU-GPU transfers
    std::vector<float> host_positions_cache;
    std::vector<float> host_velocities_cache;
    std::vector<float> host_forces_cache;

    // Private implementation methods
    void initializeGPU(int n_bodies);
    void ensureGPUCapacity(int n_bodies);
    void detectContactsGPU(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<double>& radii,
        const std::vector<int>& material_ids
    );

    bool solveContactConstraintsGPU(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& masses,
        std::vector<Eigen::Vector3d>& contact_forces
    );

    void computeContactGradientsGPU(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& masses,
        const std::vector<double>& radii,
        const std::vector<int>& material_ids,
        const std::vector<Eigen::Vector3d>& output_gradients,
        std::vector<Eigen::Vector3d>& position_gradients,
        std::vector<Eigen::Vector3d>& velocity_gradients
    );

public:
    VariationalContactSolverGPU(const VariationalContactParams& p = VariationalContactParams{});
    ~VariationalContactSolverGPU();

    // Disable copy construction/assignment due to GPU resources
    VariationalContactSolverGPU(const VariationalContactSolverGPU&) = delete;
    VariationalContactSolverGPU& operator=(const VariationalContactSolverGPU&) = delete;

    // Move construction/assignment
    VariationalContactSolverGPU(VariationalContactSolverGPU&& other) noexcept;
    VariationalContactSolverGPU& operator=(VariationalContactSolverGPU&& other) noexcept;

    // API identical to CPU version for drop-in replacement
    void detectContactsVariational(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<double>& radii,
        const std::vector<int>& material_ids
    );

    void computeContactForces(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& masses,
        const std::vector<double>& radii,
        const std::vector<int>& material_ids,
        std::vector<Eigen::Vector3d>& forces
    );

    void computeContactGradients(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& masses,
        const std::vector<double>& radii,
        const std::vector<int>& material_ids,
        const std::vector<Eigen::Vector3d>& output_gradients,
        std::vector<Eigen::Vector3d>& position_gradients,
        std::vector<Eigen::Vector3d>& velocity_gradients
    );

    double computeContactEnergy(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<double>& radii,
        const std::vector<int>& material_ids
    ) const;

    bool verifyGradientCorrectness(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& masses,
        const std::vector<double>& radii,
        const std::vector<int>& material_ids,
        double tolerance = 1e-8
    ) const;

    // GPU-specific performance and analysis methods
    const GPUPerformanceMetrics& getLastPerformanceMetrics() const { return last_metrics; }

    float getGPUMemoryUsageMB() const;
    float getComputeUtilizationPercent() const;

    void enableAsyncExecution(bool enable = true);
    void enableMemoryPrefetching(bool enable = true);
    void setPreferredBlockSize(int block_size);

    // GPU resource management
    void warmupGPU(int n_bodies);  // Pre-allocate and warm GPU kernels
    void synchronizeGPU();         // Wait for all GPU operations to complete
    void clearGPUCache();          // Free GPU memory cache

    // Advanced GPU features
    void enableMultiGPU(const std::vector<int>& gpu_device_ids);
    void enableGPUDirectAccess(bool enable = true);  // GPU Direct for RDMA

    // Debugging and profiling
    void enableKernelProfiling(bool enable = true);
    void dumpGPUMemoryLayout() const;
    void validateGPUResults() const;

    // Parameter management
    const VariationalContactParams& getParameters() const { return params; }
    void setParameters(const VariationalContactParams& p);

    // GPU device information
    static std::vector<std::string> getAvailableGPUDevices();
    static bool isGPUAvailable();
    static size_t getAvailableGPUMemory(int device_id = 0);
    static int getOptimalBlockSize(int device_id = 0);
};

// GPU-accelerated hybrid integrator
class VariationalContactIntegratorGPU {
private:
    std::unique_ptr<VariationalContactSolverGPU> contact_solver_gpu;

    // Integration parameters optimized for GPU
    struct GPUIntegrationParams {
        double implicit_contact_threshold = 5e-4;
        double explicit_stability_factor = 0.6;
        bool adaptive_timestep = true;
        double min_timestep = 1e-7;
        double max_timestep = 5e-3;
        double timestep_safety_factor = 0.7;
        double max_energy_growth = 0.05;
        double velocity_damping = 0.98;

        // GPU-specific parameters
        int gpu_block_size = 256;
        bool enable_async_kernels = true;
        bool enable_memory_pools = true;
        int max_concurrent_streams = 4;
    };

    GPUIntegrationParams gpu_integration_params;

    // GPU integration state
    std::unique_ptr<VariationalContactGPUData> integrator_gpu_data;
    bool gpu_integrator_initialized;

public:
    VariationalContactIntegratorGPU(const VariationalContactParams& contact_params = VariationalContactParams{});
    ~VariationalContactIntegratorGPU();

    // Disable copy, allow move
    VariationalContactIntegratorGPU(const VariationalContactIntegratorGPU&) = delete;
    VariationalContactIntegratorGPU& operator=(const VariationalContactIntegratorGPU&) = delete;
    VariationalContactIntegratorGPU(VariationalContactIntegratorGPU&&) = default;
    VariationalContactIntegratorGPU& operator=(VariationalContactIntegratorGPU&&) = default;

    // Integration step with GPU acceleration
    double integrateStep(
        std::vector<Eigen::Vector3d>& positions,
        std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& masses,
        const std::vector<double>& radii,
        const std::vector<int>& material_ids,
        double dt,
        const std::vector<Eigen::Vector3d>& external_forces = {}
    );

    // Gradient computation through integration step (GPU-accelerated)
    void computeIntegrationGradients(
        const std::vector<Eigen::Vector3d>& positions_initial,
        const std::vector<Eigen::Vector3d>& velocities_initial,
        const std::vector<Eigen::Vector3d>& positions_final,
        const std::vector<Eigen::Vector3d>& velocities_final,
        const std::vector<double>& masses,
        const std::vector<double>& radii,
        const std::vector<int>& material_ids,
        double dt,
        const std::vector<Eigen::Vector3d>& output_position_gradients,
        const std::vector<Eigen::Vector3d>& output_velocity_gradients,
        std::vector<Eigen::Vector3d>& input_position_gradients,
        std::vector<Eigen::Vector3d>& input_velocity_gradients
    );

    // GPU-specific methods
    VariationalContactSolverGPU& getContactSolver() { return *contact_solver_gpu; }
    const VariationalContactSolverGPU& getContactSolver() const { return *contact_solver_gpu; }

    void setGPUIntegrationParams(const GPUIntegrationParams& params) { gpu_integration_params = params; }
    const GPUIntegrationParams& getGPUIntegrationParams() const { return gpu_integration_params; }

    // Performance monitoring
    float getLastIntegrationTimeMs() const;
    float getGPUUtilizationPercent() const;
    size_t getPeakGPUMemoryUsage() const;
};

// Benchmarking and comparison utilities
namespace VariationalContactGPUUtils {

    // Compare GPU vs CPU performance
    struct PerformanceComparison {
        float cpu_time_ms;
        float gpu_time_ms;
        float speedup_factor;
        float memory_usage_cpu_mb;
        float memory_usage_gpu_mb;
        int max_contacts_cpu;
        int max_contacts_gpu;
        bool cpu_converged;
        bool gpu_converged;
        double energy_error_cpu;
        double energy_error_gpu;
    };

    PerformanceComparison benchmarkGPUvsCPU(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& masses,
        const std::vector<double>& radii,
        const std::vector<int>& material_ids,
        int num_trials = 10
    );

    // GPU-specific test scenarios for validation
    void setupGPUStressTest(
        std::vector<Eigen::Vector3d>& positions,
        std::vector<Eigen::Vector3d>& velocities,
        std::vector<double>& masses,
        std::vector<double>& radii,
        std::vector<int>& material_ids,
        int num_bodies = 10000  // Large scale for GPU testing
    );

    // Memory bandwidth and compute throughput analysis
    struct GPUResourceAnalysis {
        float theoretical_memory_bandwidth_gbps;
        float achieved_memory_bandwidth_gbps;
        float theoretical_compute_tflops;
        float achieved_compute_tflops;
        float memory_efficiency_percent;
        float compute_efficiency_percent;
        int occupancy_percent;
        int active_sms;
    };

    GPUResourceAnalysis analyzeGPUPerformance(
        const VariationalContactSolverGPU& solver,
        int num_bodies,
        int num_contacts
    );

    // GPU kernel optimization suggestions
    std::vector<std::string> getOptimizationSuggestions(
        const GPUResourceAnalysis& analysis
    );
}

} // namespace physgrad