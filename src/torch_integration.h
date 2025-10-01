#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <functional>
#include <type_traits>

#include "functional_api_simple.h"
#include "memory_optimization.h"

namespace physgrad::torch_integration {

// =============================================================================
// TENSOR TYPE TRAITS AND CONCEPTS
// =============================================================================

template<typename T>
concept PyTorchCompatible = std::is_same_v<T, float> || std::is_same_v<T, double> ||
                           std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

// Map C++ types to PyTorch scalar types
template<typename T>
constexpr torch::ScalarType scalar_type_v = torch::kFloat32;

template<>
constexpr torch::ScalarType scalar_type_v<double> = torch::kFloat64;

template<>
constexpr torch::ScalarType scalar_type_v<int32_t> = torch::kInt32;

template<>
constexpr torch::ScalarType scalar_type_v<int64_t> = torch::kInt64;

// =============================================================================
// ZERO-COPY TENSOR WRAPPERS
// =============================================================================

// Zero-copy wrapper around PhysGrad particle state
template<typename Scalar>
class TensorParticleState {
private:
    functional::ImmutableParticleState<Scalar> state_;
    torch::Device device_;

    // Cached tensor views (created on-demand)
    mutable std::optional<torch::Tensor> positions_tensor_;
    mutable std::optional<torch::Tensor> velocities_tensor_;
    mutable std::optional<torch::Tensor> charges_tensor_;
    mutable std::optional<torch::Tensor> masses_tensor_;

public:
    explicit TensorParticleState(
        functional::ImmutableParticleState<Scalar> state,
        torch::Device device = torch::kCPU
    );

    // Zero-copy tensor access
    torch::Tensor positions() const;
    torch::Tensor velocities() const;
    torch::Tensor charges() const;
    torch::Tensor masses() const;

    // Physics metadata as tensors
    torch::Tensor energy_tensor() const;
    torch::Tensor metadata_tensor() const; // [total_energy, kinetic, potential, temperature]

    // Create new state from tensor updates (functional)
    TensorParticleState withUpdatedPositions(const torch::Tensor& new_positions) const;
    TensorParticleState withUpdatedVelocities(const torch::Tensor& new_velocities) const;

    // Access underlying state
    const functional::ImmutableParticleState<Scalar>& state() const { return state_; }
    torch::Device device() const { return device_; }
    size_t particleCount() const { return state_.particleCount(); }

    // Move to different device
    TensorParticleState to(torch::Device new_device) const;
    TensorParticleState cuda() const { return to(torch::kCUDA); }
    TensorParticleState cpu() const { return to(torch::kCPU); }
};

// =============================================================================
// TENSOR-BASED FORCE COMPUTATION
// =============================================================================

// PyTorch-accelerated force computation
class TensorForceComputer {
private:
    torch::Device device_;
    float coulomb_constant_;
    bool use_cutoff_;
    float cutoff_radius_;

public:
    TensorForceComputer(
        torch::Device device = torch::kCUDA,
        float coulomb_constant = 8.99e9f,
        bool use_cutoff = false,
        float cutoff_radius = 10.0f
    );

    // Compute forces using PyTorch tensors (GPU-accelerated)
    torch::Tensor computeForces(
        const torch::Tensor& positions,
        const torch::Tensor& charges
    ) const;

    // Batch force computation for multiple states
    torch::Tensor computeBatchForces(
        const torch::Tensor& batch_positions,  // [batch_size, num_particles, 3]
        const torch::Tensor& batch_charges     // [batch_size, num_particles]
    ) const;

    // Energy computation
    torch::Tensor computePotentialEnergy(
        const torch::Tensor& positions,
        const torch::Tensor& charges
    ) const;

    torch::Tensor computeKineticEnergy(
        const torch::Tensor& velocities,
        const torch::Tensor& masses
    ) const;

    // Set parameters
    void setCoulombConstant(float k) { coulomb_constant_ = k; }
    void setCutoff(float radius) { cutoff_radius_ = radius; use_cutoff_ = true; }
    void disableCutoff() { use_cutoff_ = false; }
};

// =============================================================================
// TENSOR-BASED INTEGRATION
// =============================================================================

class TensorIntegrator {
private:
    torch::Device device_;
    float dt_;
    std::string method_;

public:
    TensorIntegrator(
        float timestep,
        torch::Device device = torch::kCUDA,
        const std::string& method = "verlet"
    );

    // Verlet integration step
    std::tuple<torch::Tensor, torch::Tensor> verletStep(
        const torch::Tensor& positions,
        const torch::Tensor& velocities,
        const torch::Tensor& forces,
        const torch::Tensor& masses
    ) const;

    // Leapfrog integration step
    std::tuple<torch::Tensor, torch::Tensor> leapfrogStep(
        const torch::Tensor& positions,
        const torch::Tensor& velocities,
        const torch::Tensor& forces,
        const torch::Tensor& masses
    ) const;

    // Runge-Kutta 4th order step
    std::tuple<torch::Tensor, torch::Tensor> rk4Step(
        const torch::Tensor& positions,
        const torch::Tensor& velocities,
        const TensorForceComputer& force_computer,
        const torch::Tensor& charges,
        const torch::Tensor& masses
    ) const;

    // Batch integration for multiple trajectories
    std::tuple<torch::Tensor, torch::Tensor> batchIntegrationStep(
        const torch::Tensor& batch_positions,  // [batch_size, num_particles, 3]
        const torch::Tensor& batch_velocities, // [batch_size, num_particles, 3]
        const torch::Tensor& batch_forces,     // [batch_size, num_particles, 3]
        const torch::Tensor& batch_masses      // [batch_size, num_particles]
    ) const;

    // Setters
    void setTimestep(float dt) { dt_ = dt; }
    void setMethod(const std::string& method) { method_ = method; }
    float timestep() const { return dt_; }
};

// =============================================================================
// DIFFERENTIABLE SIMULATION
// =============================================================================

class DifferentiableSimulation : public torch::nn::Module {
private:
    std::shared_ptr<TensorForceComputer> force_computer_;
    std::shared_ptr<TensorIntegrator> integrator_;
    torch::Device device_;
    int num_steps_;

public:
    DifferentiableSimulation(
        std::shared_ptr<TensorForceComputer> force_computer,
        std::shared_ptr<TensorIntegrator> integrator,
        int num_steps = 1
    );

    // Forward pass: simulate num_steps
    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor initial_positions,
        torch::Tensor initial_velocities,
        torch::Tensor charges,
        torch::Tensor masses
    );

    // Batch simulation for multiple initial conditions
    std::tuple<torch::Tensor, torch::Tensor> batchForward(
        torch::Tensor batch_initial_positions,  // [batch_size, num_particles, 3]
        torch::Tensor batch_initial_velocities, // [batch_size, num_particles, 3]
        torch::Tensor batch_charges,            // [batch_size, num_particles]
        torch::Tensor batch_masses              // [batch_size, num_particles]
    );

    // Full trajectory simulation
    std::tuple<torch::Tensor, torch::Tensor> simulateTrajectory(
        torch::Tensor initial_positions,
        torch::Tensor initial_velocities,
        torch::Tensor charges,
        torch::Tensor masses,
        int trajectory_length
    );

    // Energy loss computation
    torch::Tensor energyLoss(
        torch::Tensor predicted_positions,
        torch::Tensor predicted_velocities,
        torch::Tensor target_positions,
        torch::Tensor target_velocities
    );

    // Setters
    void setNumSteps(int steps) { num_steps_ = steps; }
    int numSteps() const { return num_steps_; }
};

// =============================================================================
// CONVERSION UTILITIES
// =============================================================================

namespace conversions {

// Convert PhysGrad state to PyTorch tensors
template<typename Scalar>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
stateToTensors(
    const functional::ImmutableParticleState<Scalar>& state,
    torch::Device device = torch::kCPU
);

// Convert PyTorch tensors to PhysGrad state
template<typename Scalar>
functional::ImmutableParticleState<Scalar> tensorsToState(
    const torch::Tensor& positions,
    const torch::Tensor& velocities,
    const torch::Tensor& charges,
    const torch::Tensor& masses
);

// Zero-copy tensor creation from raw data
torch::Tensor createTensorView(
    void* data_ptr,
    const std::vector<int64_t>& sizes,
    torch::ScalarType dtype,
    torch::Device device = torch::kCPU
);

// GPU memory transfer utilities
torch::Tensor cpuToGpu(const torch::Tensor& cpu_tensor, int gpu_id = 0);
torch::Tensor gpuToCpu(const torch::Tensor& gpu_tensor);

// Batch creation utilities
torch::Tensor stackStates(const std::vector<TensorParticleState<float>>& states);
std::vector<TensorParticleState<float>> unstackBatch(const torch::Tensor& batch_tensor);

} // namespace conversions

// =============================================================================
// AUTOGRAD FUNCTIONS FOR PHYSICS OPERATIONS
// =============================================================================

// Custom autograd function for force computation
class ForceComputationFunction : public torch::autograd::Function<ForceComputationFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor positions,
        torch::Tensor charges,
        double coulomb_constant
    );

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    );
};

// Custom autograd function for Verlet integration
class VerletIntegrationFunction : public torch::autograd::Function<VerletIntegrationFunction> {
public:
    static std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor positions,
        torch::Tensor velocities,
        torch::Tensor forces,
        torch::Tensor masses,
        double dt
    );

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    );
};

// =============================================================================
// NEURAL NETWORK MODULES
// =============================================================================

// Graph Neural Network for particle interactions
class ParticleGNN : public torch::nn::Module {
private:
    torch::nn::Linear node_encoder_{nullptr};
    torch::nn::Linear edge_encoder_{nullptr};
    torch::nn::Linear node_decoder_{nullptr};
    torch::nn::ModuleList message_layers_;
    int num_message_passing_steps_;
    float cutoff_radius_;

public:
    ParticleGNN(
        int node_features = 64,
        int edge_features = 32,
        int num_layers = 3,
        float cutoff_radius = 5.0f
    );

    // Forward pass
    torch::Tensor forward(
        torch::Tensor positions,
        torch::Tensor node_features,
        torch::Tensor charges
    );

    // Build edge connectivity based on distance cutoff
    std::tuple<torch::Tensor, torch::Tensor> buildEdges(
        torch::Tensor positions
    );

private:
    torch::Tensor messagePassingStep(
        torch::Tensor node_features,
        torch::Tensor edge_features,
        torch::Tensor edge_index
    );
};

// Neural force field
class NeuralForceField : public torch::nn::Module {
private:
    std::shared_ptr<ParticleGNN> gnn_;
    torch::nn::Linear force_head_{nullptr};
    torch::nn::Linear energy_head_{nullptr};
    bool predict_forces_;
    bool predict_energy_;

public:
    NeuralForceField(
        int node_features = 64,
        int edge_features = 32,
        int num_layers = 3,
        bool predict_forces = true,
        bool predict_energy = true
    );

    // Predict forces and/or energy
    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor positions,
        torch::Tensor charges
    );

    // Energy-conserving force prediction (forces = -∇E)
    torch::Tensor energyConservingForces(
        torch::Tensor positions,
        torch::Tensor charges
    );
};

// =============================================================================
// TRAINING UTILITIES
// =============================================================================

namespace training {

// Dataset for physics simulation data
class PhysicsDataset : public torch::data::Dataset<PhysicsDataset> {
private:
    std::vector<torch::Tensor> positions_;
    std::vector<torch::Tensor> velocities_;
    std::vector<torch::Tensor> forces_;
    std::vector<torch::Tensor> energies_;
    std::vector<torch::Tensor> charges_;
    std::vector<torch::Tensor> masses_;

public:
    PhysicsDataset(
        std::vector<torch::Tensor> positions,
        std::vector<torch::Tensor> velocities,
        std::vector<torch::Tensor> forces,
        std::vector<torch::Tensor> energies,
        std::vector<torch::Tensor> charges,
        std::vector<torch::Tensor> masses
    );

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};

// Loss functions for physics
torch::Tensor forceLoss(
    const torch::Tensor& predicted_forces,
    const torch::Tensor& target_forces
);

torch::Tensor energyLoss(
    const torch::Tensor& predicted_energy,
    const torch::Tensor& target_energy
);

torch::Tensor conservationLoss(
    const torch::Tensor& initial_energy,
    const torch::Tensor& final_energy
);

// Training loop utilities
class PhysicsTrainer {
private:
    std::shared_ptr<torch::nn::Module> model_;
    torch::optim::Optimizer* optimizer_;
    torch::Device device_;

public:
    PhysicsTrainer(
        std::shared_ptr<torch::nn::Module> model,
        torch::optim::Optimizer* optimizer,
        torch::Device device = torch::kCUDA
    );

    float trainEpoch(torch::data::DataLoader<PhysicsDataset>& data_loader);
    float validate(torch::data::DataLoader<PhysicsDataset>& data_loader);
    void saveModel(const std::string& path);
    void loadModel(const std::string& path);
};

} // namespace training

} // namespace physgrad::torch_integration