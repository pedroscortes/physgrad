/**
 * PhysGrad Material Point Method (MPM) Solver
 *
 * High-performance GPU-accelerated MPM solver with multi-material support
 * and advanced constitutive models
 */

#pragma once

#include "mpm_data_structures.h"
#include "common_types.h"
#include <memory>
#include <vector>
#include <string>
#include <chrono>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#endif

namespace physgrad::mpm {

// =============================================================================
// SOLVER CONFIGURATION
// =============================================================================

struct MPMSolverConfig {
    // Grid parameters
    int3 grid_dimensions = {64, 64, 64};
    T3 grid_spacing = {0.01f, 0.01f, 0.01f};    // 1cm grid cells
    T3 domain_origin = {0.0f, 0.0f, 0.0f};

    // Time integration
    float time_step = 1e-4f;                     // 0.1ms timestep
    float simulation_time = 1.0f;                // 1 second simulation
    float cfl_factor = 0.4f;                     // CFL stability factor

    // Solver parameters
    int max_particles = 1000000;                 // 1M particles max
    float particle_radius = 0.005f;              // 5mm particles
    bool enable_gravity = true;
    T3 gravity = {0.0f, -9.81f, 0.0f};          // Standard gravity

    // Multi-material settings
    bool enable_multi_material = true;
    int max_materials = 8;

    // Performance settings
    int cuda_block_size = 256;
    bool enable_g2p2g_fusion = true;             // Use fused kernels
    bool enable_adaptive_timestep = false;
    float min_timestep = 1e-6f;
    float max_timestep = 1e-3f;

    // Output settings
    bool enable_output = true;
    float output_frequency = 0.01f;              // Output every 10ms
    std::string output_directory = "./mpm_output";

    // Debug and profiling
    bool enable_profiling = false;
    bool enable_debug_output = false;
    bool verify_conservation = false;
};

// =============================================================================
// PERFORMANCE METRICS
// =============================================================================

struct MPMPerformanceMetrics {
    double total_simulation_time = 0.0;
    double average_step_time = 0.0;
    double g2p_time = 0.0;
    double p2g_time = 0.0;
    double grid_update_time = 0.0;
    double particle_update_time = 0.0;

    size_t total_memory_usage = 0;
    size_t particle_memory = 0;
    size_t grid_memory = 0;

    int total_steps = 0;
    int rejected_steps = 0;
    float average_particles_per_second = 0.0f;

    void reset() {
        *this = MPMPerformanceMetrics{};
    }

    void print() const {
        std::cout << "MPM Performance Metrics:" << std::endl;
        std::cout << "  Total simulation time: " << total_simulation_time << " s" << std::endl;
        std::cout << "  Average step time: " << average_step_time * 1000 << " ms" << std::endl;
        std::cout << "  Particles per second: " << average_particles_per_second / 1e6 << " M" << std::endl;
        std::cout << "  Total memory usage: " << total_memory_usage / (1024*1024) << " MB" << std::endl;
        std::cout << "  Grid update time: " << grid_update_time / total_simulation_time * 100 << "%" << std::endl;
        std::cout << "  G2P2G time: " << (g2p_time + p2g_time) / total_simulation_time * 100 << "%" << std::endl;
    }
};

// =============================================================================
// MAIN MPM SOLVER CLASS
// =============================================================================

template<typename T = float>
class MPMSolver {
public:
    using scalar_type = T;
    using particle_container = ParticleAoSoA<T>;
    using grid_container = MPMGrid<T>;

private:
    // Configuration
    MPMSolverConfig config_;
    MaterialDatabase material_db_;
    MPMPerformanceMetrics metrics_;

    // Data containers
    std::unique_ptr<particle_container> particles_;
    std::unique_ptr<grid_container> grid_;

    // GPU memory management
#ifdef __CUDACC__
    T* d_particle_positions_ = nullptr;
    T* d_particle_velocities_ = nullptr;
    T* d_particle_masses_ = nullptr;
    T* d_particle_volumes_ = nullptr;
    T* d_particle_deformation_gradients_ = nullptr;
    T* d_particle_stresses_ = nullptr;
    MaterialType* d_particle_material_types_ = nullptr;
    uint32_t* d_particle_active_ = nullptr;

    T* d_grid_masses_ = nullptr;
    T* d_grid_velocities_ = nullptr;
    T* d_grid_forces_ = nullptr;
    T* d_grid_momentum_ = nullptr;
    uint32_t* d_grid_boundary_conditions_ = nullptr;

    T* d_new_grid_masses_ = nullptr;
    T* d_new_grid_velocities_ = nullptr;

    MaterialParameters* d_material_params_ = nullptr;

    cudaStream_t compute_stream_;
    cudaEvent_t start_event_, stop_event_;
#endif

    // Simulation state
    float current_time_ = 0.0f;
    int current_step_ = 0;
    bool is_initialized_ = false;

public:
    explicit MPMSolver(const MPMSolverConfig& config = MPMSolverConfig{})
        : config_(config) {

        particles_ = std::make_unique<particle_container>(config_.max_particles);
        grid_ = std::make_unique<grid_container>(
            config_.grid_dimensions,
            config_.grid_spacing,
            config_.domain_origin
        );

        initializeGPU();
    }

    ~MPMSolver() {
        cleanup();
    }

    // Initialization and setup
    void initialize() {
        if (is_initialized_) return;

        allocateGPUMemory();
        setupBoundaryConditions();
        is_initialized_ = true;

        std::cout << "MPM Solver initialized:" << std::endl;
        std::cout << "  Grid: " << config_.grid_dimensions.x << "x"
                  << config_.grid_dimensions.y << "x" << config_.grid_dimensions.z << std::endl;
        std::cout << "  Max particles: " << config_.max_particles << std::endl;
        std::cout << "  Grid spacing: " << config_.grid_spacing.x << "m" << std::endl;
        std::cout << "  Time step: " << config_.time_step << "s" << std::endl;
    }

    // Particle management
    void addParticles(const std::vector<T3>& positions,
                     const std::vector<T3>& velocities,
                     const std::vector<T>& masses,
                     MaterialType material_type = MaterialType::ELASTIC) {

        size_t num_new_particles = positions.size();
        size_t current_particles = particles_->size();

        if (current_particles + num_new_particles > config_.max_particles) {
            throw std::runtime_error("Exceeded maximum particle count");
        }

        // Add particles to container
        for (size_t i = 0; i < num_new_particles; ++i) {
            size_t particle_id = current_particles + i;

            particles_->setPosition(particle_id, positions[i].x, positions[i].y, positions[i].z);
            particles_->setVelocity(particle_id, velocities[i].x, velocities[i].y, velocities[i].z);
            particles_->setMass(particle_id, masses[i]);
            particles_->setVolume(particle_id, calculateParticleVolume(masses[i], material_type));
            particles_->setMaterialType(particle_id, material_type);
            particles_->setActive(particle_id, true);

            // Initialize deformation gradient to identity
            T F[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            particles_->setDeformationGradient(particle_id, F);

            // Initialize stress to zero
            T stress[6] = {0, 0, 0, 0, 0, 0};
            particles_->setStress(particle_id, stress);
        }

        particles_->resize(current_particles + num_new_particles);

        std::cout << "Added " << num_new_particles << " particles (total: "
                  << particles_->size() << ")" << std::endl;
    }

    // Simulation control
    void step() {
        if (!is_initialized_) {
            throw std::runtime_error("Solver not initialized");
        }

        auto step_start = std::chrono::high_resolution_clock::now();

#ifdef __CUDACC__
        cudaEventRecord(start_event_, compute_stream_);
#endif

        // Clear grid
        clearGrid();

        // Adaptive timestep (if enabled)
        float dt = config_.time_step;
        if (config_.enable_adaptive_timestep) {
            dt = computeAdaptiveTimestep();
        }

        // Main MPM algorithm
        if (config_.enable_g2p2g_fusion) {
            // Fused G2P2G kernel
            performFusedG2P2G(dt);
        } else {
            // Separate kernels
            gridToParticle(dt);
            updateParticles(dt);
            particleToGrid();
        }

        // Update grid
        updateGrid(dt);

        // Update particle positions
        updateParticlePositions(dt);

        // Apply boundary conditions
        applyBoundaryConditions();

        // Update simulation state
        current_time_ += dt;
        current_step_++;

#ifdef __CUDACC__
        cudaEventRecord(stop_event_, compute_stream_);
        cudaEventSynchronize(stop_event_);

        float gpu_time;
        cudaEventElapsedTime(&gpu_time, start_event_, stop_event_);

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_time = std::chrono::duration<double>(step_end - step_start).count();

        updateMetrics(step_time, gpu_time / 1000.0);
#endif
    }

    void run() {
        initialize();

        std::cout << "Starting MPM simulation..." << std::endl;
        std::cout << "  Total time: " << config_.simulation_time << "s" << std::endl;
        std::cout << "  Time step: " << config_.time_step << "s" << std::endl;

        auto sim_start = std::chrono::high_resolution_clock::now();

        while (current_time_ < config_.simulation_time) {
            step();

            // Output progress
            if (current_step_ % 100 == 0) {
                float progress = current_time_ / config_.simulation_time * 100.0f;
                std::cout << "Progress: " << std::fixed << std::setprecision(1)
                          << progress << "% (t=" << current_time_ << "s)" << std::endl;
            }

            // Output data (if enabled)
            if (config_.enable_output &&
                static_cast<int>(current_time_ / config_.output_frequency) >
                static_cast<int>((current_time_ - config_.time_step) / config_.output_frequency)) {
                outputData();
            }
        }

        auto sim_end = std::chrono::high_resolution_clock::now();
        metrics_.total_simulation_time = std::chrono::duration<double>(sim_end - sim_start).count();

        std::cout << "Simulation completed!" << std::endl;
        if (config_.enable_profiling) {
            metrics_.print();
        }
    }

    // Data access
    const particle_container& getParticles() const { return *particles_; }
    const grid_container& getGrid() const { return *grid_; }
    const MPMPerformanceMetrics& getMetrics() const { return metrics_; }

    float getCurrentTime() const { return current_time_; }
    int getCurrentStep() const { return current_step_; }

private:
    void initializeGPU() {
#ifdef __CUDACC__
        cudaStreamCreate(&compute_stream_);
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
#endif
    }

    void allocateGPUMemory() {
#ifdef __CUDACC__
        size_t num_particles = particles_->size();
        size_t num_nodes = grid_->getTotalNodes();

        // Allocate particle data
        cudaMalloc(&d_particle_positions_, num_particles * 3 * sizeof(T));
        cudaMalloc(&d_particle_velocities_, num_particles * 3 * sizeof(T));
        cudaMalloc(&d_particle_masses_, num_particles * sizeof(T));
        cudaMalloc(&d_particle_volumes_, num_particles * sizeof(T));
        cudaMalloc(&d_particle_deformation_gradients_, num_particles * 9 * sizeof(T));
        cudaMalloc(&d_particle_stresses_, num_particles * 6 * sizeof(T));
        cudaMalloc(&d_particle_material_types_, num_particles * sizeof(MaterialType));
        cudaMalloc(&d_particle_active_, num_particles * sizeof(uint32_t));

        // Allocate grid data
        cudaMalloc(&d_grid_masses_, num_nodes * sizeof(T));
        cudaMalloc(&d_grid_velocities_, num_nodes * 3 * sizeof(T));
        cudaMalloc(&d_grid_forces_, num_nodes * 3 * sizeof(T));
        cudaMalloc(&d_grid_momentum_, num_nodes * 3 * sizeof(T));
        cudaMalloc(&d_grid_boundary_conditions_, num_nodes * sizeof(uint32_t));

        // Temporary grid arrays
        cudaMalloc(&d_new_grid_masses_, num_nodes * sizeof(T));
        cudaMalloc(&d_new_grid_velocities_, num_nodes * 3 * sizeof(T));

        // Material parameters
        cudaMalloc(&d_material_params_, material_db_.getNumMaterials() * sizeof(MaterialParameters));

        // Calculate memory usage
        metrics_.particle_memory = num_particles * (3 + 3 + 1 + 1 + 9 + 6) * sizeof(T) +
                                  num_particles * (sizeof(MaterialType) + sizeof(uint32_t));
        metrics_.grid_memory = num_nodes * (1 + 3 + 3 + 3) * sizeof(T) + num_nodes * sizeof(uint32_t);
        metrics_.total_memory_usage = metrics_.particle_memory + metrics_.grid_memory;

        std::cout << "GPU memory allocated: " << metrics_.total_memory_usage / (1024*1024) << " MB" << std::endl;
#endif
    }

    void clearGrid() {
#ifdef __CUDACC__
        size_t num_nodes = grid_->getTotalNodes();
        cudaMemset(d_grid_masses_, 0, num_nodes * sizeof(T));
        cudaMemset(d_grid_velocities_, 0, num_nodes * 3 * sizeof(T));
        cudaMemset(d_grid_forces_, 0, num_nodes * 3 * sizeof(T));
        cudaMemset(d_new_grid_masses_, 0, num_nodes * sizeof(T));
        cudaMemset(d_new_grid_velocities_, 0, num_nodes * 3 * sizeof(T));
#endif
    }

    void performFusedG2P2G(float dt) {
#ifdef __CUDACC__
        // Launch fused G2P2G kernel
        int num_particles = static_cast<int>(particles_->size());
        int num_blocks = (num_particles + config_.cuda_block_size - 1) / config_.cuda_block_size;

        fusedG2P2GKernel<<<num_blocks, config_.cuda_block_size, 0, compute_stream_>>>(
            d_particle_positions_,
            d_particle_velocities_,
            d_particle_masses_,
            d_particle_volumes_,
            d_particle_deformation_gradients_,
            d_particle_stresses_,
            d_particle_material_types_,
            d_particle_active_,
            d_grid_masses_,
            d_grid_velocities_,
            d_grid_forces_,
            d_new_grid_masses_,
            d_new_grid_velocities_,
            config_.grid_dimensions,
            config_.grid_spacing,
            config_.domain_origin,
            dt,
            num_particles,
            d_material_params_
        );

        cudaStreamSynchronize(compute_stream_);
#endif
    }

    void updateGrid(float dt) {
#ifdef __CUDACC__
        int num_nodes = static_cast<int>(grid_->getTotalNodes());
        int num_blocks = (num_nodes + config_.cuda_block_size - 1) / config_.cuda_block_size;

        updateGridKernel<<<num_blocks, config_.cuda_block_size, 0, compute_stream_>>>(
            d_new_grid_masses_,
            d_new_grid_velocities_,
            d_grid_forces_,
            d_grid_boundary_conditions_,
            dt,
            config_.gravity,
            num_nodes
        );

        // Swap grid arrays
        std::swap(d_grid_masses_, d_new_grid_masses_);
        std::swap(d_grid_velocities_, d_new_grid_velocities_);

        cudaStreamSynchronize(compute_stream_);
#endif
    }

    void updateParticlePositions(float dt) {
#ifdef __CUDACC__
        int num_particles = static_cast<int>(particles_->size());
        int num_blocks = (num_particles + config_.cuda_block_size - 1) / config_.cuda_block_size;

        updateParticlePositionsKernel<<<num_blocks, config_.cuda_block_size, 0, compute_stream_>>>(
            d_particle_positions_,
            d_particle_velocities_,
            d_particle_active_,
            dt,
            num_particles
        );

        cudaStreamSynchronize(compute_stream_);
#endif
    }

    // Placeholder implementations for non-fused kernels
    void gridToParticle(float dt) { /* Implementation for separate G2P */ }
    void updateParticles(float dt) { /* Implementation for particle update */ }
    void particleToGrid() { /* Implementation for separate P2G */ }

    float computeAdaptiveTimestep() {
        // Implement CFL condition based adaptive timestepping
        return config_.time_step;
    }

    void setupBoundaryConditions() {
        // Set up domain boundary conditions
        // For now, just set bottom boundary to no-slip
        int3 dims = config_.grid_dimensions;
        for (int i = 0; i < dims.x; ++i) {
            for (int j = 0; j < dims.y; ++j) {
                // Bottom boundary
                size_t node_id = grid_->getLinearIndex(i, 0, j);
                grid_->setBoundaryCondition(node_id,
                    static_cast<uint32_t>(BoundaryType::DIRICHLET_Y));
            }
        }
    }

    void applyBoundaryConditions() {
        // Additional boundary condition enforcement if needed
    }

    T calculateParticleVolume(T mass, MaterialType material_type) {
        const auto& params = material_db_.getMaterial(static_cast<uint32_t>(material_type));
        return mass / params.density;
    }

    void updateMetrics(double step_time, double gpu_time) {
        metrics_.total_steps++;
        metrics_.average_step_time = (metrics_.average_step_time * (metrics_.total_steps - 1) + step_time) / metrics_.total_steps;
        metrics_.average_particles_per_second = particles_->size() / metrics_.average_step_time;
    }

    void outputData() {
        // Implement data output (e.g., to VTK files)
        std::cout << "Writing output at t=" << current_time_ << std::endl;
    }

    void cleanup() {
#ifdef __CUDACC__
        if (d_particle_positions_) cudaFree(d_particle_positions_);
        if (d_particle_velocities_) cudaFree(d_particle_velocities_);
        if (d_particle_masses_) cudaFree(d_particle_masses_);
        if (d_particle_volumes_) cudaFree(d_particle_volumes_);
        if (d_particle_deformation_gradients_) cudaFree(d_particle_deformation_gradients_);
        if (d_particle_stresses_) cudaFree(d_particle_stresses_);
        if (d_particle_material_types_) cudaFree(d_particle_material_types_);
        if (d_particle_active_) cudaFree(d_particle_active_);

        if (d_grid_masses_) cudaFree(d_grid_masses_);
        if (d_grid_velocities_) cudaFree(d_grid_velocities_);
        if (d_grid_forces_) cudaFree(d_grid_forces_);
        if (d_grid_momentum_) cudaFree(d_grid_momentum_);
        if (d_grid_boundary_conditions_) cudaFree(d_grid_boundary_conditions_);

        if (d_new_grid_masses_) cudaFree(d_new_grid_masses_);
        if (d_new_grid_velocities_) cudaFree(d_new_grid_velocities_);

        if (d_material_params_) cudaFree(d_material_params_);

        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
        cudaStreamDestroy(compute_stream_);
#endif
    }
};

} // namespace physgrad::mpm