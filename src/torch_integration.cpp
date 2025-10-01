#include "torch_integration.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace physgrad::torch_integration {

// =============================================================================
// TENSOR PARTICLE STATE IMPLEMENTATION
// =============================================================================

template<typename Scalar>
TensorParticleState<Scalar>::TensorParticleState(
    functional::ImmutableParticleState<Scalar> state,
    torch::Device device
) : state_(std::move(state)), device_(device) {}

template<typename Scalar>
torch::Tensor TensorParticleState<Scalar>::positions() const {
    if (!positions_tensor_) {
        const auto& pos = state_.positions;
        auto options = torch::TensorOptions()
            .dtype(scalar_type_v<Scalar>)
            .device(device_);

        // Create tensor from raw data (zero-copy on CPU)
        if (device_.is_cpu()) {
            positions_tensor_ = torch::from_blob(
                const_cast<void*>(static_cast<const void*>(pos.data())),
                {static_cast<int64_t>(pos.size()), 4},
                options
            );
        } else {
            // Copy to GPU
            auto cpu_tensor = torch::from_blob(
                const_cast<void*>(static_cast<const void*>(pos.data())),
                {static_cast<int64_t>(pos.size()), 4},
                torch::TensorOptions().dtype(scalar_type_v<Scalar>)
            );
            positions_tensor_ = cpu_tensor.to(device_);
        }
    }
    return *positions_tensor_;
}

template<typename Scalar>
torch::Tensor TensorParticleState<Scalar>::velocities() const {
    if (!velocities_tensor_) {
        const auto& vel = state_.velocities;
        auto options = torch::TensorOptions()
            .dtype(scalar_type_v<Scalar>)
            .device(device_);

        if (device_.is_cpu()) {
            velocities_tensor_ = torch::from_blob(
                const_cast<void*>(static_cast<const void*>(vel.data())),
                {static_cast<int64_t>(vel.size()), 3},
                options
            );
        } else {
            auto cpu_tensor = torch::from_blob(
                const_cast<void*>(static_cast<const void*>(vel.data())),
                {static_cast<int64_t>(vel.size()), 3},
                torch::TensorOptions().dtype(scalar_type_v<Scalar>)
            );
            velocities_tensor_ = cpu_tensor.to(device_);
        }
    }
    return *velocities_tensor_;
}

template<typename Scalar>
torch::Tensor TensorParticleState<Scalar>::charges() const {
    if (!charges_tensor_) {
        const auto& chg = state_.charges;
        auto options = torch::TensorOptions()
            .dtype(scalar_type_v<Scalar>)
            .device(device_);

        if (device_.is_cpu()) {
            charges_tensor_ = torch::from_blob(
                const_cast<void*>(static_cast<const void*>(chg.data())),
                {static_cast<int64_t>(chg.size())},
                options
            );
        } else {
            auto cpu_tensor = torch::from_blob(
                const_cast<void*>(static_cast<const void*>(chg.data())),
                {static_cast<int64_t>(chg.size())},
                torch::TensorOptions().dtype(scalar_type_v<Scalar>)
            );
            charges_tensor_ = cpu_tensor.to(device_);
        }
    }
    return *charges_tensor_;
}

template<typename Scalar>
torch::Tensor TensorParticleState<Scalar>::masses() const {
    if (!masses_tensor_) {
        const auto& mass = state_.masses;
        auto options = torch::TensorOptions()
            .dtype(scalar_type_v<Scalar>)
            .device(device_);

        if (device_.is_cpu()) {
            masses_tensor_ = torch::from_blob(
                const_cast<void*>(static_cast<const void*>(mass.data())),
                {static_cast<int64_t>(mass.size())},
                options
            );
        } else {
            auto cpu_tensor = torch::from_blob(
                const_cast<void*>(static_cast<const void*>(mass.data())),
                {static_cast<int64_t>(mass.size())},
                torch::TensorOptions().dtype(scalar_type_v<Scalar>)
            );
            masses_tensor_ = cpu_tensor.to(device_);
        }
    }
    return *masses_tensor_;
}

template<typename Scalar>
torch::Tensor TensorParticleState<Scalar>::energy_tensor() const {
    auto options = torch::TensorOptions()
        .dtype(scalar_type_v<Scalar>)
        .device(device_);

    return torch::tensor(state_.total_energy, options);
}

template<typename Scalar>
torch::Tensor TensorParticleState<Scalar>::metadata_tensor() const {
    auto options = torch::TensorOptions()
        .dtype(scalar_type_v<Scalar>)
        .device(device_);

    std::vector<Scalar> metadata = {
        state_.total_energy,
        state_.kinetic_energy,
        state_.potential_energy,
        state_.temperature
    };

    return torch::from_blob(metadata.data(), {4}, options).clone();
}

template<typename Scalar>
TensorParticleState<Scalar> TensorParticleState<Scalar>::to(torch::Device new_device) const {
    if (new_device == device_) {
        return *this;
    }
    return TensorParticleState<Scalar>(state_, new_device);
}

// Explicit instantiations
template class TensorParticleState<float>;
template class TensorParticleState<double>;

// =============================================================================
// TENSOR FORCE COMPUTER IMPLEMENTATION
// =============================================================================

TensorForceComputer::TensorForceComputer(
    torch::Device device,
    float coulomb_constant,
    bool use_cutoff,
    float cutoff_radius
) : device_(device),
    coulomb_constant_(coulomb_constant),
    use_cutoff_(use_cutoff),
    cutoff_radius_(cutoff_radius) {}

torch::Tensor TensorForceComputer::computeForces(
    const torch::Tensor& positions,
    const torch::Tensor& charges
) const {
    // Ensure tensors are on the correct device
    auto pos = positions.to(device_);
    auto chg = charges.to(device_);

    const int64_t n = pos.size(0);
    auto forces = torch::zeros({n, 3}, pos.options());

    // Vectorized force computation using PyTorch operations
    // r_ij = positions[i] - positions[j] for all pairs
    auto pos_i = pos.unsqueeze(1);  // [n, 1, 3]
    auto pos_j = pos.unsqueeze(0);  // [1, n, 3]
    auto dr = pos_i - pos_j;        // [n, n, 3]

    // Distance calculations
    auto r_squared = (dr * dr).sum(-1);  // [n, n]
    auto r = torch::sqrt(r_squared + 1e-10f);  // Add small epsilon for numerical stability

    // Force magnitude calculation
    auto q_i = chg.unsqueeze(1);  // [n, 1]
    auto q_j = chg.unsqueeze(0);  // [1, n]
    auto force_magnitude = coulomb_constant_ * q_i * q_j / (r_squared * r + 1e-10f);

    // Apply cutoff if enabled
    if (use_cutoff_) {
        auto mask = r <= cutoff_radius_;
        force_magnitude = force_magnitude * mask.to(force_magnitude.dtype());
    }

    // Remove self-interactions
    auto eye = torch::eye(n, r.options());
    force_magnitude = force_magnitude * (1.0f - eye);

    // Compute force vectors
    auto force_vectors = force_magnitude.unsqueeze(-1) * dr;  // [n, n, 3]

    // Sum forces from all other particles
    forces = force_vectors.sum(1);  // [n, 3]

    return forces;
}

torch::Tensor TensorForceComputer::computeBatchForces(
    const torch::Tensor& batch_positions,
    const torch::Tensor& batch_charges
) const {
    const int64_t batch_size = batch_positions.size(0);
    const int64_t n_particles = batch_positions.size(1);

    auto batch_forces = torch::zeros_like(batch_positions);

    for (int64_t b = 0; b < batch_size; ++b) {
        batch_forces[b] = computeForces(batch_positions[b], batch_charges[b]);
    }

    return batch_forces;
}

torch::Tensor TensorForceComputer::computePotentialEnergy(
    const torch::Tensor& positions,
    const torch::Tensor& charges
) const {
    auto pos = positions.to(device_);
    auto chg = charges.to(device_);

    const int64_t n = pos.size(0);
    auto energy = torch::zeros({}, pos.options());

    // Vectorized energy computation
    auto pos_i = pos.unsqueeze(1);  // [n, 1, 3]
    auto pos_j = pos.unsqueeze(0);  // [1, n, 3]
    auto dr = pos_i - pos_j;        // [n, n, 3]
    auto r = torch::sqrt((dr * dr).sum(-1) + 1e-10f);

    auto q_i = chg.unsqueeze(1);
    auto q_j = chg.unsqueeze(0);
    auto pair_energies = coulomb_constant_ * q_i * q_j / (r + 1e-10f);

    // Apply cutoff if enabled
    if (use_cutoff_) {
        auto mask = r <= cutoff_radius_;
        pair_energies = pair_energies * mask.to(pair_energies.dtype());
    }

    // Remove self-interactions and double counting
    auto upper_triangular = torch::triu(torch::ones({n, n}, r.options()), 1);
    energy = (pair_energies * upper_triangular).sum();

    return energy;
}

torch::Tensor TensorForceComputer::computeKineticEnergy(
    const torch::Tensor& velocities,
    const torch::Tensor& masses
) const {
    auto vel = velocities.to(device_);
    auto m = masses.to(device_);

    auto v_squared = (vel * vel).sum(-1);  // [n]
    auto kinetic = 0.5f * m * v_squared;
    return kinetic.sum();
}

// =============================================================================
// TENSOR INTEGRATOR IMPLEMENTATION
// =============================================================================

TensorIntegrator::TensorIntegrator(
    float timestep,
    torch::Device device,
    const std::string& method
) : dt_(timestep), device_(device), method_(method) {}

std::tuple<torch::Tensor, torch::Tensor> TensorIntegrator::verletStep(
    const torch::Tensor& positions,
    const torch::Tensor& velocities,
    const torch::Tensor& forces,
    const torch::Tensor& masses
) const {
    auto pos = positions.to(device_);
    auto vel = velocities.to(device_);
    auto f = forces.to(device_);
    auto m = masses.to(device_);

    // Acceleration: a = F/m
    auto acceleration = f / m.unsqueeze(-1);  // [n, 3]

    // Verlet integration: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
    auto new_positions = pos + vel * dt_ + 0.5f * acceleration * dt_ * dt_;

    // v(t+dt) = v(t) + a(t)*dt
    auto new_velocities = vel + acceleration * dt_;

    return std::make_tuple(new_positions, new_velocities);
}

std::tuple<torch::Tensor, torch::Tensor> TensorIntegrator::leapfrogStep(
    const torch::Tensor& positions,
    const torch::Tensor& velocities,
    const torch::Tensor& forces,
    const torch::Tensor& masses
) const {
    auto pos = positions.to(device_);
    auto vel = velocities.to(device_);
    auto f = forces.to(device_);
    auto m = masses.to(device_);

    auto acceleration = f / m.unsqueeze(-1);

    // Leapfrog: v(t+dt/2) = v(t) + a(t)*dt/2
    auto vel_half = vel + acceleration * (dt_ / 2.0f);

    // x(t+dt) = x(t) + v(t+dt/2)*dt
    auto new_positions = pos + vel_half * dt_;

    // v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2
    // Note: This requires computing forces at new positions
    auto new_velocities = vel_half + acceleration * (dt_ / 2.0f);

    return std::make_tuple(new_positions, new_velocities);
}

std::tuple<torch::Tensor, torch::Tensor> TensorIntegrator::batchIntegrationStep(
    const torch::Tensor& batch_positions,
    const torch::Tensor& batch_velocities,
    const torch::Tensor& batch_forces,
    const torch::Tensor& batch_masses
) const {
    if (method_ == "verlet") {
        return verletStep(batch_positions, batch_velocities, batch_forces, batch_masses);
    } else if (method_ == "leapfrog") {
        return leapfrogStep(batch_positions, batch_velocities, batch_forces, batch_masses);
    } else {
        throw std::runtime_error("Unknown integration method: " + method_);
    }
}

// =============================================================================
// DIFFERENTIABLE SIMULATION IMPLEMENTATION
// =============================================================================

DifferentiableSimulation::DifferentiableSimulation(
    std::shared_ptr<TensorForceComputer> force_computer,
    std::shared_ptr<TensorIntegrator> integrator,
    int num_steps
) : force_computer_(force_computer),
    integrator_(integrator),
    num_steps_(num_steps) {

    // Register as a PyTorch module
    register_module("force_computer", force_computer_);
    register_module("integrator", integrator_);
}

std::tuple<torch::Tensor, torch::Tensor> DifferentiableSimulation::forward(
    torch::Tensor initial_positions,
    torch::Tensor initial_velocities,
    torch::Tensor charges,
    torch::Tensor masses
) {
    auto positions = initial_positions;
    auto velocities = initial_velocities;

    for (int step = 0; step < num_steps_; ++step) {
        // Compute forces
        auto forces = force_computer_->computeForces(positions, charges);

        // Integration step
        auto [new_pos, new_vel] = integrator_->verletStep(positions, velocities, forces, masses);

        positions = new_pos;
        velocities = new_vel;
    }

    return std::make_tuple(positions, velocities);
}

std::tuple<torch::Tensor, torch::Tensor> DifferentiableSimulation::batchForward(
    torch::Tensor batch_initial_positions,
    torch::Tensor batch_initial_velocities,
    torch::Tensor batch_charges,
    torch::Tensor batch_masses
) {
    const int64_t batch_size = batch_initial_positions.size(0);
    auto batch_positions = batch_initial_positions;
    auto batch_velocities = batch_initial_velocities;

    for (int step = 0; step < num_steps_; ++step) {
        // Compute forces for entire batch
        auto batch_forces = force_computer_->computeBatchForces(batch_positions, batch_charges);

        // Integration step for entire batch
        auto [new_pos, new_vel] = integrator_->batchIntegrationStep(
            batch_positions, batch_velocities, batch_forces, batch_masses
        );

        batch_positions = new_pos;
        batch_velocities = new_vel;
    }

    return std::make_tuple(batch_positions, batch_velocities);
}

torch::Tensor DifferentiableSimulation::energyLoss(
    torch::Tensor predicted_positions,
    torch::Tensor predicted_velocities,
    torch::Tensor target_positions,
    torch::Tensor target_velocities
) {
    auto pos_loss = torch::mse_loss(predicted_positions, target_positions);
    auto vel_loss = torch::mse_loss(predicted_velocities, target_velocities);
    return pos_loss + vel_loss;
}

// =============================================================================
// CONVERSION UTILITIES IMPLEMENTATION
// =============================================================================

namespace conversions {

template<typename Scalar>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
stateToTensors(
    const functional::ImmutableParticleState<Scalar>& state,
    torch::Device device
) {
    auto options = torch::TensorOptions()
        .dtype(scalar_type_v<Scalar>)
        .device(device);

    const auto& pos = state.positions;
    const auto& vel = state.velocities;
    const auto& charges = state.charges;
    const auto& masses = state.masses;

    auto pos_tensor = torch::from_blob(
        const_cast<void*>(static_cast<const void*>(pos.data())),
        {static_cast<int64_t>(pos.size()), 4},
        torch::TensorOptions().dtype(scalar_type_v<Scalar>)
    ).to(device);

    auto vel_tensor = torch::from_blob(
        const_cast<void*>(static_cast<const void*>(vel.data())),
        {static_cast<int64_t>(vel.size()), 3},
        torch::TensorOptions().dtype(scalar_type_v<Scalar>)
    ).to(device);

    auto charges_tensor = torch::from_blob(
        const_cast<void*>(static_cast<const void*>(charges.data())),
        {static_cast<int64_t>(charges.size())},
        torch::TensorOptions().dtype(scalar_type_v<Scalar>)
    ).to(device);

    auto masses_tensor = torch::from_blob(
        const_cast<void*>(static_cast<const void*>(masses.data())),
        {static_cast<int64_t>(masses.size())},
        torch::TensorOptions().dtype(scalar_type_v<Scalar>)
    ).to(device);

    return std::make_tuple(pos_tensor, vel_tensor, charges_tensor, masses_tensor);
}

// Explicit instantiations
template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
stateToTensors<float>(const functional::ImmutableParticleState<float>&, torch::Device);

template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
stateToTensors<double>(const functional::ImmutableParticleState<double>&, torch::Device);

torch::Tensor createTensorView(
    void* data_ptr,
    const std::vector<int64_t>& sizes,
    torch::ScalarType dtype,
    torch::Device device
) {
    auto tensor = torch::from_blob(data_ptr, sizes, torch::TensorOptions().dtype(dtype));
    if (!device.is_cpu()) {
        tensor = tensor.to(device);
    }
    return tensor;
}

torch::Tensor cpuToGpu(const torch::Tensor& cpu_tensor, int gpu_id) {
    return cpu_tensor.to(torch::Device(torch::kCUDA, gpu_id));
}

torch::Tensor gpuToCpu(const torch::Tensor& gpu_tensor) {
    return gpu_tensor.to(torch::kCPU);
}

} // namespace conversions

// =============================================================================
// AUTOGRAD FUNCTIONS IMPLEMENTATION
// =============================================================================

torch::Tensor ForceComputationFunction::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor positions,
    torch::Tensor charges,
    double coulomb_constant
) {
    ctx->save_for_backward({positions, charges});
    ctx->saved_data["coulomb_constant"] = coulomb_constant;

    TensorForceComputer force_computer(positions.device(), coulomb_constant);
    return force_computer.computeForces(positions, charges);
}

torch::autograd::tensor_list ForceComputationFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    auto positions = saved[0];
    auto charges = saved[1];
    auto coulomb_constant = ctx->saved_data["coulomb_constant"].toDouble();

    auto grad_forces = grad_outputs[0];

    // Compute gradients w.r.t. positions and charges
    // This would involve computing the Hessian of the potential energy
    // For now, return None for simplicity
    return {torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

// =============================================================================
// TRAINING UTILITIES IMPLEMENTATION
// =============================================================================

namespace training {

PhysicsDataset::PhysicsDataset(
    std::vector<torch::Tensor> positions,
    std::vector<torch::Tensor> velocities,
    std::vector<torch::Tensor> forces,
    std::vector<torch::Tensor> energies,
    std::vector<torch::Tensor> charges,
    std::vector<torch::Tensor> masses
) : positions_(std::move(positions)),
    velocities_(std::move(velocities)),
    forces_(std::move(forces)),
    energies_(std::move(energies)),
    charges_(std::move(charges)),
    masses_(std::move(masses)) {}

torch::data::Example<> PhysicsDataset::get(size_t index) {
    return {
        torch::stack({positions_[index], velocities_[index], charges_[index], masses_[index]}),
        torch::stack({forces_[index], energies_[index]})
    };
}

torch::optional<size_t> PhysicsDataset::size() const {
    return positions_.size();
}

torch::Tensor forceLoss(
    const torch::Tensor& predicted_forces,
    const torch::Tensor& target_forces
) {
    return torch::mse_loss(predicted_forces, target_forces);
}

torch::Tensor energyLoss(
    const torch::Tensor& predicted_energy,
    const torch::Tensor& target_energy
) {
    return torch::mse_loss(predicted_energy, target_energy);
}

torch::Tensor conservationLoss(
    const torch::Tensor& initial_energy,
    const torch::Tensor& final_energy
) {
    return torch::mse_loss(final_energy, initial_energy);
}

PhysicsTrainer::PhysicsTrainer(
    std::shared_ptr<torch::nn::Module> model,
    torch::optim::Optimizer* optimizer,
    torch::Device device
) : model_(model), optimizer_(optimizer), device_(device) {}

float PhysicsTrainer::trainEpoch(torch::data::DataLoader<PhysicsDataset>& data_loader) {
    model_->train();
    float total_loss = 0.0f;
    size_t batch_count = 0;

    for (auto& batch : data_loader) {
        optimizer_->zero_grad();

        auto data = batch.data.to(device_);
        auto target = batch.target.to(device_);

        // Forward pass would depend on specific model
        // This is a placeholder implementation
        auto output = model_->forward({data});
        auto loss = torch::mse_loss(output.toTensor(), target);

        loss.backward();
        optimizer_->step();

        total_loss += loss.item<float>();
        batch_count++;
    }

    return total_loss / batch_count;
}

} // namespace training

} // namespace physgrad::torch_integration