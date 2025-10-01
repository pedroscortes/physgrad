/**
 * PhysGrad - Differentiable Material Point Method
 *
 * Implements automatic differentiation capabilities for Material Point Method
 * simulations, enabling gradient computation through particle-grid operations
 * for optimization, learning, and inverse problems.
 */

#ifndef PHYSGRAD_DIFFERENTIABLE_MPM_H
#define PHYSGRAD_DIFFERENTIABLE_MPM_H

#include "material_point_method.h"
#include "mpm_g2p2g_kernels.h"
#include "common_types.h"
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <utility>
#include <cstring>
#include <algorithm>

#ifdef __CUDACC__
    #define PHYSGRAD_DEVICE __device__
    #define PHYSGRAD_HOST_DEVICE __host__ __device__
    #define PHYSGRAD_GLOBAL __global__
#else
    #define PHYSGRAD_DEVICE
    #define PHYSGRAD_HOST_DEVICE
    #define PHYSGRAD_GLOBAL
#endif

// Include concepts when available
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "concepts/forward_declarations.h"
#endif

namespace physgrad {
namespace mpm {
namespace differentiable {

    // =============================================================================
    // FORWARD DECLARATIONS AND AUXILIARY STRUCTURES
    // =============================================================================

    template<typename T> class DifferentiableMPMSystem;
    template<typename T> class MPMGradientTape;
    template<typename T> class ParticleGradients;
    template<typename T> class GridGradients;

    /**
     * Gradient computation mode for MPM operations
     */
    enum class GradientMode {
        FORWARD,        // Forward-mode automatic differentiation
        REVERSE,        // Reverse-mode automatic differentiation (backpropagation)
        HYBRID          // Hybrid mode: forward for positions, reverse for forces
    };

    /**
     * Checkpointing strategy for memory-efficient gradient computation
     */
    enum class CheckpointStrategy {
        NONE,           // No checkpointing - store all intermediate states
        UNIFORM,        // Uniform time interval checkpointing
        ADAPTIVE,       // Adaptive checkpointing based on memory usage
        LOGARITHMIC     // Logarithmic spacing for optimal memory-time tradeoff
    };

    /**
     * Configuration for differentiable MPM computations
     */
    template<typename T>
    struct DifferentiableMPMConfig {
        GradientMode gradient_mode = GradientMode::REVERSE;
        CheckpointStrategy checkpoint_strategy = CheckpointStrategy::ADAPTIVE;

        // Gradient computation settings
        T finite_difference_eps = T{1e-6};    // For numerical gradient verification
        bool enable_second_order = false;     // Enable second-order derivatives
        bool use_checkpointing = true;        // Use memory checkpointing

        // Memory management
        size_t max_checkpoints = 100;         // Maximum stored checkpoints
        T memory_budget_gb = T{4.0};          // Memory budget for gradients

        // Numerical stability
        T gradient_clip_threshold = T{1e3};   // Gradient clipping threshold
        bool enable_gradient_accumulation = true;  // Accumulate gradients across steps

        // Performance settings
        bool parallelize_gradient_computation = true;
        size_t gradient_block_size = 256;     // Block size for gradient kernels
    };

    // =============================================================================
    // GRADIENT STORAGE AND MANAGEMENT
    // =============================================================================

    /**
     * Storage for particle-level gradients in AoSoA format
     */
    template<typename T>
    class ParticleGradients {
    public:
        using vector_type = ConceptVector3D<T>;
        static constexpr size_t chunk_size = 64; // Align with ParticleAoSoA

    private:
        size_t num_particles_;

        // Gradient storage in chunked format for cache efficiency
        std::vector<std::array<T, chunk_size * 3>> position_gradients_;
        std::vector<std::array<T, chunk_size * 3>> velocity_gradients_;
        std::vector<std::array<T, chunk_size>> mass_gradients_;
        std::vector<std::array<T, chunk_size * 9>> deformation_grad_gradients_; // F tensor

        // Constitutive model gradients
        std::vector<std::array<T, chunk_size>> volume_gradients_;
        std::vector<std::array<T, chunk_size * 6>> stress_gradients_; // Symmetric stress tensor

    public:
        ParticleGradients() : num_particles_(0) {}

        void resize(size_t num_particles) {
            num_particles_ = num_particles;
            size_t num_chunks = (num_particles + chunk_size - 1) / chunk_size;

            position_gradients_.resize(num_chunks);
            velocity_gradients_.resize(num_chunks);
            mass_gradients_.resize(num_chunks);
            deformation_grad_gradients_.resize(num_chunks);
            volume_gradients_.resize(num_chunks);
            stress_gradients_.resize(num_chunks);

            // Initialize to zero
            std::memset(position_gradients_.data(), 0,
                       position_gradients_.size() * sizeof(position_gradients_[0]));
            std::memset(velocity_gradients_.data(), 0,
                       velocity_gradients_.size() * sizeof(velocity_gradients_[0]));
            std::memset(mass_gradients_.data(), 0,
                       mass_gradients_.size() * sizeof(mass_gradients_[0]));
            std::memset(deformation_grad_gradients_.data(), 0,
                       deformation_grad_gradients_.size() * sizeof(deformation_grad_gradients_[0]));
            std::memset(volume_gradients_.data(), 0,
                       volume_gradients_.size() * sizeof(volume_gradients_[0]));
            std::memset(stress_gradients_.data(), 0,
                       stress_gradients_.size() * sizeof(stress_gradients_[0]));
        }

        PHYSGRAD_HOST_DEVICE
        vector_type getPositionGradient(size_t particle_id) const {
            size_t chunk_id = particle_id / chunk_size;
            size_t local_id = particle_id % chunk_size;

            const auto& chunk = position_gradients_[chunk_id];
            return vector_type{
                chunk[local_id],                    // x
                chunk[local_id + chunk_size],       // y
                chunk[local_id + 2 * chunk_size]    // z
            };
        }

        PHYSGRAD_HOST_DEVICE
        void setPositionGradient(size_t particle_id, const vector_type& grad) {
            size_t chunk_id = particle_id / chunk_size;
            size_t local_id = particle_id % chunk_size;

            auto& chunk = position_gradients_[chunk_id];
            chunk[local_id] = grad[0];                    // x
            chunk[local_id + chunk_size] = grad[1];       // y
            chunk[local_id + 2 * chunk_size] = grad[2];   // z
        }

        PHYSGRAD_HOST_DEVICE
        vector_type getVelocityGradient(size_t particle_id) const {
            size_t chunk_id = particle_id / chunk_size;
            size_t local_id = particle_id % chunk_size;

            const auto& chunk = velocity_gradients_[chunk_id];
            return vector_type{
                chunk[local_id],
                chunk[local_id + chunk_size],
                chunk[local_id + 2 * chunk_size]
            };
        }

        PHYSGRAD_HOST_DEVICE
        void setVelocityGradient(size_t particle_id, const vector_type& grad) {
            size_t chunk_id = particle_id / chunk_size;
            size_t local_id = particle_id % chunk_size;

            auto& chunk = velocity_gradients_[chunk_id];
            chunk[local_id] = grad[0];
            chunk[local_id + chunk_size] = grad[1];
            chunk[local_id + 2 * chunk_size] = grad[2];
        }

        PHYSGRAD_HOST_DEVICE
        T getMassGradient(size_t particle_id) const {
            size_t chunk_id = particle_id / chunk_size;
            size_t local_id = particle_id % chunk_size;
            return mass_gradients_[chunk_id][local_id];
        }

        PHYSGRAD_HOST_DEVICE
        void setMassGradient(size_t particle_id, T grad) {
            size_t chunk_id = particle_id / chunk_size;
            size_t local_id = particle_id % chunk_size;
            mass_gradients_[chunk_id][local_id] = grad;
        }

        // Accumulate gradients (for gradient accumulation across timesteps)
        PHYSGRAD_HOST_DEVICE
        void accumulatePositionGradient(size_t particle_id, const vector_type& grad) {
            auto current = getPositionGradient(particle_id);
            setPositionGradient(particle_id, current + grad);
        }

        PHYSGRAD_HOST_DEVICE
        void accumulateVelocityGradient(size_t particle_id, const vector_type& grad) {
            auto current = getVelocityGradient(particle_id);
            setVelocityGradient(particle_id, current + grad);
        }

        // Clear all gradients
        void zero() {
            for (auto& chunk : position_gradients_) {
                std::fill(chunk.begin(), chunk.end(), T{0});
            }
            for (auto& chunk : velocity_gradients_) {
                std::fill(chunk.begin(), chunk.end(), T{0});
            }
            for (auto& chunk : mass_gradients_) {
                std::fill(chunk.begin(), chunk.end(), T{0});
            }
            for (auto& chunk : deformation_grad_gradients_) {
                std::fill(chunk.begin(), chunk.end(), T{0});
            }
            for (auto& chunk : volume_gradients_) {
                std::fill(chunk.begin(), chunk.end(), T{0});
            }
            for (auto& chunk : stress_gradients_) {
                std::fill(chunk.begin(), chunk.end(), T{0});
            }
        }

        size_t size() const { return num_particles_; }

        // Memory usage estimation
        size_t getMemoryUsageBytes() const {
            return position_gradients_.size() * sizeof(position_gradients_[0]) +
                   velocity_gradients_.size() * sizeof(velocity_gradients_[0]) +
                   mass_gradients_.size() * sizeof(mass_gradients_[0]) +
                   deformation_grad_gradients_.size() * sizeof(deformation_grad_gradients_[0]) +
                   volume_gradients_.size() * sizeof(volume_gradients_[0]) +
                   stress_gradients_.size() * sizeof(stress_gradients_[0]);
        }
    };

    /**
     * Storage for grid-level gradients
     */
    template<typename T>
    class GridGradients {
    public:
        using vector_type = ConceptVector3D<T>;

    private:
        int3 grid_dims_;
        size_t total_nodes_;

        std::vector<vector_type> velocity_gradients_;    // Grid velocity gradients
        std::vector<vector_type> momentum_gradients_;    // Grid momentum gradients
        std::vector<vector_type> force_gradients_;       // Grid force gradients
        std::vector<T> mass_gradients_;                  // Grid mass gradients

    public:
        GridGradients() : total_nodes_(0) {}

        GridGradients(const int3& dims) : grid_dims_(dims) {
            total_nodes_ = static_cast<size_t>(dims.x) * dims.y * dims.z;
            resize(total_nodes_);
        }

        void resize(size_t num_nodes) {
            total_nodes_ = num_nodes;
            velocity_gradients_.resize(num_nodes, vector_type{T{0}, T{0}, T{0}});
            momentum_gradients_.resize(num_nodes, vector_type{T{0}, T{0}, T{0}});
            force_gradients_.resize(num_nodes, vector_type{T{0}, T{0}, T{0}});
            mass_gradients_.resize(num_nodes, T{0});
        }

        PHYSGRAD_HOST_DEVICE
        vector_type getVelocityGradient(size_t node_id) const {
            return velocity_gradients_[node_id];
        }

        PHYSGRAD_HOST_DEVICE
        void setVelocityGradient(size_t node_id, const vector_type& grad) {
            velocity_gradients_[node_id] = grad;
        }

        PHYSGRAD_HOST_DEVICE
        vector_type getMomentumGradient(size_t node_id) const {
            return momentum_gradients_[node_id];
        }

        PHYSGRAD_HOST_DEVICE
        void setMomentumGradient(size_t node_id, const vector_type& grad) {
            momentum_gradients_[node_id] = grad;
        }

        PHYSGRAD_HOST_DEVICE
        void accumulateVelocityGradient(size_t node_id, const vector_type& grad) {
            velocity_gradients_[node_id] = velocity_gradients_[node_id] + grad;
        }

        PHYSGRAD_HOST_DEVICE
        void accumulateMomentumGradient(size_t node_id, const vector_type& grad) {
            momentum_gradients_[node_id] = momentum_gradients_[node_id] + grad;
        }

        void zero() {
            std::fill(velocity_gradients_.begin(), velocity_gradients_.end(),
                     vector_type{T{0}, T{0}, T{0}});
            std::fill(momentum_gradients_.begin(), momentum_gradients_.end(),
                     vector_type{T{0}, T{0}, T{0}});
            std::fill(force_gradients_.begin(), force_gradients_.end(),
                     vector_type{T{0}, T{0}, T{0}});
            std::fill(mass_gradients_.begin(), mass_gradients_.end(), T{0});
        }

        size_t size() const { return total_nodes_; }

        size_t getMemoryUsageBytes() const {
            return velocity_gradients_.size() * sizeof(vector_type) * 4 +
                   mass_gradients_.size() * sizeof(T);
        }
    };

    // =============================================================================
    // GRADIENT TAPE FOR AUTOMATIC DIFFERENTIATION
    // =============================================================================

    /**
     * Computational graph node for reverse-mode automatic differentiation
     */
    template<typename T>
    struct TapeNode {
        enum class OperationType {
            P2G_TRANSFER,           // Particle-to-grid transfer
            G2P_TRANSFER,           // Grid-to-particle transfer
            GRID_UPDATE,            // Grid velocity/momentum update
            CONSTITUTIVE_UPDATE,    // Constitutive model update
            FORCE_COMPUTATION,      // Force computation
            BOUNDARY_CONDITIONS     // Boundary condition application
        };

        OperationType operation;
        size_t timestep;
        std::vector<size_t> input_indices;     // Indices of input variables
        std::vector<size_t> output_indices;    // Indices of output variables

        // Store necessary data for backward pass
        std::vector<T> intermediate_data;      // Cached intermediate computations
        std::function<void(const TapeNode<T>&, ParticleGradients<T>&, GridGradients<T>&)> backward_fn;

        TapeNode(OperationType op, size_t step) : operation(op), timestep(step) {}
    };

    /**
     * Gradient tape for recording computational graph
     */
    template<typename T>
    class MPMGradientTape {
    private:
        std::vector<TapeNode<T>> tape_;
        std::vector<size_t> checkpoint_indices_;  // Tape indices where checkpoints are stored
        bool recording_ = false;
        size_t current_timestep_ = 0;

        // Memory management
        size_t memory_usage_bytes_ = 0;
        size_t max_memory_bytes_;

    public:
        explicit MPMGradientTape(size_t max_memory_mb = 1000)
            : max_memory_bytes_(max_memory_mb * 1024 * 1024) {}

        void startRecording() {
            recording_ = true;
            tape_.clear();
            checkpoint_indices_.clear();
            memory_usage_bytes_ = 0;
            current_timestep_ = 0;
        }

        void stopRecording() {
            recording_ = false;
        }

        bool isRecording() const { return recording_; }

        void nextTimestep() {
            current_timestep_++;
        }

        // Record operation on tape
        void recordOperation(typename TapeNode<T>::OperationType op,
                           const std::vector<size_t>& inputs,
                           const std::vector<size_t>& outputs,
                           const std::vector<T>& intermediate_data,
                           std::function<void(const TapeNode<T>&, ParticleGradients<T>&, GridGradients<T>&)> backward_fn) {
            if (!recording_) return;

            TapeNode<T> node(op, current_timestep_);
            node.input_indices = inputs;
            node.output_indices = outputs;
            node.intermediate_data = intermediate_data;
            node.backward_fn = backward_fn;

            // Estimate memory usage
            size_t node_memory = sizeof(TapeNode<T>) +
                               inputs.size() * sizeof(size_t) +
                               outputs.size() * sizeof(size_t) +
                               intermediate_data.size() * sizeof(T);

            memory_usage_bytes_ += node_memory;

            // Check if we need to create a checkpoint
            if (memory_usage_bytes_ > max_memory_bytes_ / 4) {
                createCheckpoint();
            }

            tape_.push_back(std::move(node));
        }

        // Execute backward pass
        void backward(ParticleGradients<T>& particle_grads, GridGradients<T>& grid_grads) {
            // Execute tape in reverse order
            for (auto it = tape_.rbegin(); it != tape_.rend(); ++it) {
                if (it->backward_fn) {
                    it->backward_fn(*it, particle_grads, grid_grads);
                }
            }
        }

        void createCheckpoint() {
            checkpoint_indices_.push_back(tape_.size());
        }

        void clear() {
            tape_.clear();
            checkpoint_indices_.clear();
            memory_usage_bytes_ = 0;
            current_timestep_ = 0;
        }

        size_t size() const { return tape_.size(); }
        size_t getMemoryUsage() const { return memory_usage_bytes_; }

        // Get tape node for inspection/debugging
        const TapeNode<T>& getNode(size_t index) const {
            return tape_[index];
        }
    };

    // =============================================================================
    // DIFFERENTIABLE MPM KERNELS
    // =============================================================================

    /**
     * Differentiable particle-to-grid transfer kernel with gradient computation
     */
    template<typename T>
    PHYSGRAD_GLOBAL void differentiableP2GKernel(
        const ParticleAoSoA<T>* particles,
        MPMGrid<T>* grid,
        ParticleGradients<T>* particle_grads,
        GridGradients<T>* grid_grads,
        ConceptVector3D<T> gravity,
        T dt,
        bool compute_gradients,
        size_t particle_offset,
        size_t particle_count) {

#ifdef __CUDACC__
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        size_t particle_id = particle_offset + bid * blockDim.x + tid;
#else
        // CPU fallback
        for (size_t particle_id = particle_offset;
             particle_id < particle_offset + particle_count;
             ++particle_id) {
#endif

        if (particle_id >= particle_offset + particle_count) return;

        // Get particle data
        auto pos = particles->getPosition(particle_id);
        auto vel = particles->getVelocity(particle_id);
        auto mass = particles->getMass(particle_id);

        // Grid position calculation
        ConceptVector3D<T> grid_pos = {
            (pos[0] - grid->origin[0]) / grid->cell_size[0],
            (pos[1] - grid->origin[1]) / grid->cell_size[1],
            (pos[2] - grid->origin[2]) / grid->cell_size[2]
        };

        // Base grid indices
        int base_i = static_cast<int>(grid_pos[0]);
        int base_j = static_cast<int>(grid_pos[1]);
        int base_k = static_cast<int>(grid_pos[2]);

        // Particle-to-grid transfer with gradient tracking
        for (int di = 0; di < 3; ++di) {
            for (int dj = 0; dj < 3; ++dj) {
                for (int dk = 0; dk < 3; ++dk) {
                    int gi = base_i + di - 1;
                    int gj = base_j + dj - 1;
                    int gk = base_k + dk - 1;

                    // Check bounds
                    if (gi < 0 || gi >= grid->dims.x ||
                        gj < 0 || gj >= grid->dims.y ||
                        gk < 0 || gk >= grid->dims.z) continue;

                    size_t node_id = gi + gj * grid->dims.x + gk * grid->dims.x * grid->dims.y;

                    // Compute shape function weights
                    T wi = MPMShapeFunctions<T>::quadratic(grid_pos[0] - gi);
                    T wj = MPMShapeFunctions<T>::quadratic(grid_pos[1] - gj);
                    T wk = MPMShapeFunctions<T>::quadratic(grid_pos[2] - gk);
                    T weight = wi * wj * wk;

                    // Transfer mass and momentum
                    T transfer_mass = weight * mass;
                    ConceptVector3D<T> transfer_momentum = vel * transfer_mass;

                    // Atomic adds for thread safety (simplified for CPU)
#ifdef __CUDACC__
                    atomicAdd(&grid->mass[node_id], transfer_mass);
                    atomicAdd(&grid->momentum[node_id][0], transfer_momentum[0]);
                    atomicAdd(&grid->momentum[node_id][1], transfer_momentum[1]);
                    atomicAdd(&grid->momentum[node_id][2], transfer_momentum[2]);
#else
                    // CPU version (not thread-safe but for demonstration)
                    grid->mass[node_id] += transfer_mass;
                    grid->momentum[node_id] = grid->momentum[node_id] + transfer_momentum;
#endif

                    // Gradient computation if enabled
                    if (compute_gradients && particle_grads && grid_grads) {
                        // Backward pass: compute gradients w.r.t. particle properties

                        // ∂loss/∂pos computation (simplified)
                        ConceptVector3D<T> pos_grad_contrib = grid_grads->getMomentumGradient(node_id) * (weight * mass);
                        particle_grads->accumulatePositionGradient(particle_id, pos_grad_contrib);

                        // ∂loss/∂vel computation
                        ConceptVector3D<T> vel_grad_contrib = grid_grads->getMomentumGradient(node_id) * (weight * mass);
                        particle_grads->accumulateVelocityGradient(particle_id, vel_grad_contrib);

                        // ∂loss/∂mass computation
                        T mass_grad_contrib = 0;
                        for (int d = 0; d < 3; ++d) {
                            mass_grad_contrib += grid_grads->getMomentumGradient(node_id)[d] * vel[d] * weight;
                        }
                        particle_grads->setMassGradient(particle_id,
                            particle_grads->getMassGradient(particle_id) + mass_grad_contrib);
                    }
                }
            }
        }

#ifndef __CUDACC__
        } // End CPU loop
#endif
    }

    /**
     * Differentiable grid-to-particle transfer kernel with gradient computation
     */
    template<typename T>
    PHYSGRAD_GLOBAL void differentiableG2PKernel(
        ParticleAoSoA<T>* particles,
        const MPMGrid<T>* grid,
        ParticleGradients<T>* particle_grads,
        GridGradients<T>* grid_grads,
        T dt,
        bool compute_gradients,
        size_t particle_offset,
        size_t particle_count) {

#ifdef __CUDACC__
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        size_t particle_id = particle_offset + bid * blockDim.x + tid;
#else
        // CPU fallback
        for (size_t particle_id = particle_offset;
             particle_id < particle_offset + particle_count;
             ++particle_id) {
#endif

        if (particle_id >= particle_offset + particle_count) return;

        auto pos = particles->getPosition(particle_id);
        auto vel = particles->getVelocity(particle_id);

        // Grid position calculation
        ConceptVector3D<T> grid_pos = {
            (pos[0] - grid->origin[0]) / grid->cell_size[0],
            (pos[1] - grid->origin[1]) / grid->cell_size[1],
            (pos[2] - grid->origin[2]) / grid->cell_size[2]
        };

        int base_i = static_cast<int>(grid_pos[0]);
        int base_j = static_cast<int>(grid_pos[1]);
        int base_k = static_cast<int>(grid_pos[2]);

        ConceptVector3D<T> new_vel{T{0}, T{0}, T{0}};
        ConceptVector3D<T> new_pos = pos;

        // Grid-to-particle interpolation
        for (int di = 0; di < 3; ++di) {
            for (int dj = 0; dj < 3; ++dj) {
                for (int dk = 0; dk < 3; ++dk) {
                    int gi = base_i + di - 1;
                    int gj = base_j + dj - 1;
                    int gk = base_k + dk - 1;

                    if (gi < 0 || gi >= grid->dims.x ||
                        gj < 0 || gj >= grid->dims.y ||
                        gk < 0 || gk >= grid->dims.z) continue;

                    size_t node_id = gi + gj * grid->dims.x + gk * grid->dims.x * grid->dims.y;

                    T wi = MPMShapeFunctions<T>::quadratic(grid_pos[0] - gi);
                    T wj = MPMShapeFunctions<T>::quadratic(grid_pos[1] - gj);
                    T wk = MPMShapeFunctions<T>::quadratic(grid_pos[2] - gk);
                    T weight = wi * wj * wk;

                    // Interpolate velocity from grid
                    if (grid->mass[node_id] > T{1e-10}) {
                        ConceptVector3D<T> grid_vel = grid->momentum[node_id] * (T{1} / grid->mass[node_id]);
                        new_vel = new_vel + grid_vel * weight;
                    }
                }
            }
        }

        // Update particle position using new velocity
        new_pos = pos + new_vel * dt;

        // Store updated values
        particles->setVelocity(particle_id, new_vel);
        particles->setPosition(particle_id, new_pos);

        // Gradient computation if enabled
        if (compute_gradients && particle_grads && grid_grads) {
            // Compute gradients for G2P transfer
            // This is a simplified version - full implementation would include
            // gradients through the interpolation weights and grid velocities

            auto vel_grad = particle_grads->getVelocityGradient(particle_id);
            auto pos_grad = particle_grads->getPositionGradient(particle_id);

            // Accumulate position gradient through velocity update
            particle_grads->accumulatePositionGradient(particle_id, vel_grad * dt);
        }

#ifndef __CUDACC__
        } // End CPU loop
#endif
    }

    // =============================================================================
    // DIFFERENTIABLE MPM SYSTEM
    // =============================================================================

    /**
     * Main differentiable MPM system class
     */
    template<typename T>
    class DifferentiableMPMSystem {
    private:
        DifferentiableMPMConfig<T> config_;
        MPMGradientTape<T> gradient_tape_;

        // Core MPM components
        std::unique_ptr<ParticleAoSoA<T>> particles_;
        std::unique_ptr<MPMGrid<T>> grid_;
        std::unique_ptr<kernels::G2P2GKernelLauncher<T>> kernel_launcher_;

        // Gradient storage
        std::unique_ptr<ParticleGradients<T>> particle_gradients_;
        std::unique_ptr<GridGradients<T>> grid_gradients_;

        // Simulation state
        T current_time_ = T{0};
        size_t current_step_ = 0;
        bool gradient_computation_enabled_ = false;

    public:
        explicit DifferentiableMPMSystem(const DifferentiableMPMConfig<T>& config = DifferentiableMPMConfig<T>{})
            : config_(config), gradient_tape_(static_cast<size_t>(config.memory_budget_gb * 1000)) {

            // Initialize kernel launcher
            typename kernels::G2P2GKernelLauncher<T>::PerformanceConfig kernel_config;
            kernel_config.use_kernel_fusion = true;
            kernel_config.use_shared_memory = true;
            kernel_launcher_ = std::make_unique<kernels::G2P2GKernelLauncher<T>>(kernel_config);
        }

        void initializeSystem(const int3& grid_dims,
                            const ConceptVector3D<T>& cell_size,
                            const ConceptVector3D<T>& origin,
                            size_t num_particles) {

            // Initialize grid
            grid_ = std::make_unique<MPMGrid<T>>(grid_dims, cell_size, origin);

            // Initialize particles
            particles_ = std::make_unique<ParticleAoSoA<T>>();
            particles_->resize(num_particles);

            // Initialize gradient storage
            particle_gradients_ = std::make_unique<ParticleGradients<T>>();
            particle_gradients_->resize(num_particles);

            grid_gradients_ = std::make_unique<GridGradients<T>>(grid_dims);
        }

        void enableGradientComputation(bool enable = true) {
            gradient_computation_enabled_ = enable;
            if (enable) {
                gradient_tape_.startRecording();
            } else {
                gradient_tape_.stopRecording();
            }
        }

        // Main simulation step with optional gradient computation
        void simulationStep(T dt, const ConceptVector3D<T>& gravity = {T{0}, T{-9.81}, T{0}}) {
            if (!particles_ || !grid_) {
                throw std::runtime_error("System not initialized");
            }

            // Clear grid for this timestep
            grid_->clearGridData();

            if (gradient_computation_enabled_) {
                // Record timestep transition
                gradient_tape_.nextTimestep();

                // Perform differentiable simulation step
                differentiableSimulationStep(dt, gravity);
            } else {
                // Standard simulation step (no gradient tracking)
                standardSimulationStep(dt, gravity);
            }

            current_time_ += dt;
            current_step_++;
        }

        // Compute gradients through backpropagation
        void computeGradients() {
            if (!gradient_computation_enabled_) {
                throw std::runtime_error("Gradient computation not enabled");
            }

            // Clear gradients
            particle_gradients_->zero();
            grid_gradients_->zero();

            // Execute backward pass
            gradient_tape_.backward(*particle_gradients_, *grid_gradients_);
        }

        // Access functions
        ParticleAoSoA<T>& getParticles() { return *particles_; }
        const ParticleAoSoA<T>& getParticles() const { return *particles_; }

        MPMGrid<T>& getGrid() { return *grid_; }
        const MPMGrid<T>& getGrid() const { return *grid_; }

        ParticleGradients<T>& getParticleGradients() { return *particle_gradients_; }
        const ParticleGradients<T>& getParticleGradients() const { return *particle_gradients_; }

        GridGradients<T>& getGridGradients() { return *grid_gradients_; }
        const GridGradients<T>& getGridGradients() const { return *grid_gradients_; }

        // Utility functions
        T getCurrentTime() const { return current_time_; }
        size_t getCurrentStep() const { return current_step_; }

        size_t getGradientTapeSize() const { return gradient_tape_.size(); }
        size_t getGradientMemoryUsage() const {
            return gradient_tape_.getMemoryUsage() +
                   particle_gradients_->getMemoryUsageBytes() +
                   grid_gradients_->getMemoryUsageBytes();
        }

        void resetSimulation() {
            current_time_ = T{0};
            current_step_ = 0;
            gradient_tape_.clear();
            if (particle_gradients_) particle_gradients_->zero();
            if (grid_gradients_) grid_gradients_->zero();
        }

    private:
        void standardSimulationStep(T dt, const ConceptVector3D<T>& gravity) {
            // Use existing G2P2G kernel launcher for standard simulation
            kernel_launcher_->launchG2P2G(*particles_, *grid_, dt, gravity, true, T{0.95}, 2);
        }

        void differentiableSimulationStep(T dt, const ConceptVector3D<T>& gravity) {
            size_t num_particles = particles_->size();

            // Launch differentiable P2G kernel
            const size_t block_size = config_.gradient_block_size;
            const size_t num_blocks = (num_particles + block_size - 1) / block_size;

#ifdef __CUDACC__
            differentiableP2GKernel<<<num_blocks, block_size>>>(
                particles_.get(), grid_.get(),
                particle_gradients_.get(), grid_gradients_.get(),
                gravity, dt, gradient_computation_enabled_, 0, num_particles);
            cudaDeviceSynchronize();
#else
            // CPU fallback
            differentiableP2GKernel(
                particles_.get(), grid_.get(),
                particle_gradients_.get(), grid_gradients_.get(),
                gravity, dt, gradient_computation_enabled_, 0, num_particles);
#endif

            // Update grid velocities
            grid_->updateGridVelocities();

            // Apply boundary conditions (simplified)
            grid_->applyBoundaryConditions();

            // Launch differentiable G2P kernel
#ifdef __CUDACC__
            differentiableG2PKernel<<<num_blocks, block_size>>>(
                particles_.get(), grid_.get(),
                particle_gradients_.get(), grid_gradients_.get(),
                dt, gradient_computation_enabled_, 0, num_particles);
            cudaDeviceSynchronize();
#else
            // CPU fallback
            differentiableG2PKernel(
                particles_.get(), grid_.get(),
                particle_gradients_.get(), grid_gradients_.get(),
                dt, gradient_computation_enabled_, 0, num_particles);
#endif

            // Record operation on gradient tape (simplified)
            if (gradient_tape_.isRecording()) {
                std::vector<size_t> inputs, outputs;
                std::vector<T> intermediate_data;

                // This would store necessary intermediate values for backward pass
                gradient_tape_.recordOperation(
                    TapeNode<T>::OperationType::P2G_TRANSFER,
                    inputs, outputs, intermediate_data,
                    [](const TapeNode<T>& node, ParticleGradients<T>& p_grads, GridGradients<T>& g_grads) {
                        // Backward pass implementation would go here
                    });
            }
        }
    };

    // =============================================================================
    // UTILITY FUNCTIONS AND HIGH-LEVEL INTERFACES
    // =============================================================================

    /**
     * Numerical gradient verification using finite differences
     */
    template<typename T>
    bool verifyGradients(DifferentiableMPMSystem<T>& system,
                        const std::function<T()>& objective_function,
                        T epsilon = T{1e-6},
                        T tolerance = T{1e-3}) {

        auto& particles = system.getParticles();
        size_t num_particles = particles.size();

        system.enableGradientComputation(true);

        // Compute analytical gradients
        T objective_value = objective_function();
        system.computeGradients();

        auto& analytical_grads = system.getParticleGradients();

        bool all_passed = true;

        // Check first few particles to avoid excessive computation
        for (size_t p = 0; p < std::min(num_particles, size_t{10}); ++p) {
            // Check position gradients
            auto analytical_pos_grad = analytical_grads.getPositionGradient(p);

            for (int d = 0; d < 3; ++d) {
                // Finite difference computation
                auto pos = particles.getPosition(p);
                pos[d] += epsilon;
                particles.setPosition(p, pos);

                system.resetSimulation();
                T objective_plus = objective_function();

                pos[d] -= 2 * epsilon;
                particles.setPosition(p, pos);

                system.resetSimulation();
                T objective_minus = objective_function();

                // Restore original position
                pos[d] += epsilon;
                particles.setPosition(p, pos);

                T numerical_grad = (objective_plus - objective_minus) / (2 * epsilon);
                T analytical_grad = analytical_pos_grad[d];

                T relative_error = std::abs(numerical_grad - analytical_grad) /
                                 (std::abs(numerical_grad) + T{1e-8});

                if (relative_error > tolerance) {
                    all_passed = false;
                    std::cout << "Gradient verification failed for particle " << p
                             << " dimension " << d << ": numerical=" << numerical_grad
                             << " analytical=" << analytical_grad
                             << " rel_error=" << relative_error << "\n";
                }
            }
        }

        return all_passed;
    }

    /**
     * Example usage: parameter optimization using gradient descent
     */
    template<typename T>
    class MPMParameterOptimizer {
    private:
        DifferentiableMPMSystem<T> system_;
        T learning_rate_;

    public:
        MPMParameterOptimizer(const DifferentiableMPMConfig<T>& config, T lr = T{0.001})
            : system_(config), learning_rate_(lr) {}

        void optimize(const std::function<T()>& loss_function, int num_iterations) {
            system_.enableGradientComputation(true);

            for (int iter = 0; iter < num_iterations; ++iter) {
                // Forward pass
                T loss = loss_function();

                // Backward pass
                system_.computeGradients();

                // Parameter update (simplified gradient descent)
                auto& particles = system_.getParticles();
                auto& gradients = system_.getParticleGradients();

                for (size_t p = 0; p < particles.size(); ++p) {
                    auto pos = particles.getPosition(p);
                    auto pos_grad = gradients.getPositionGradient(p);

                    // Gradient descent update
                    pos = pos - pos_grad * learning_rate_;
                    particles.setPosition(p, pos);
                }

                if (iter % 10 == 0) {
                    std::cout << "Iteration " << iter << ": loss = " << loss << "\n";
                }

                // Reset for next iteration
                system_.resetSimulation();
            }
        }
    };

} // namespace differentiable
} // namespace mpm
} // namespace physgrad

#endif // PHYSGRAD_DIFFERENTIABLE_MPM_H