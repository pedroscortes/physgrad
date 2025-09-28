#include "torch_integration.h"
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace torch_integration {

// CUDA kernel declarations
extern "C" {
    void launch_physics_step_kernel(
        float* pos_x, float* pos_y, float* pos_z,
        float* vel_x, float* vel_y, float* vel_z,
        const float* forces_x, const float* forces_y, const float* forces_z,
        const float* masses, int num_particles, float dt,
        cudaStream_t stream
    );

    void launch_force_computation_kernel(
        const float* pos_x, const float* pos_y, const float* pos_z,
        const float* vel_x, const float* vel_y, const float* vel_z,
        float* forces_x, float* forces_y, float* forces_z,
        const float* masses, int num_particles,
        cudaStream_t stream
    );

    void launch_constraint_kernel(
        float* pos_x, float* pos_y, float* pos_z,
        float* vel_x, float* vel_y, float* vel_z,
        const float* masses, int num_particles,
        const float* constraint_params, int num_constraints,
        float dt, cudaStream_t stream
    );
}

// Helper functions for tensor validation
bool validateTensorProperties(const torch::Tensor& tensor, const std::string& name) {
    if (!tensor.is_cuda()) {
        throw std::runtime_error(name + " tensor must be on CUDA device");
    }
    if (!tensor.is_contiguous()) {
        throw std::runtime_error(name + " tensor must be contiguous");
    }
    if (tensor.dtype() != torch::kFloat32) {
        throw std::runtime_error(name + " tensor must be float32");
    }
    return true;
}

std::tuple<torch::Tensor, torch::Tensor> simulationStepTorch(
    torch::Tensor positions,
    torch::Tensor velocities,
    torch::Tensor masses,
    torch::Tensor forces,
    float dt
) {
    // Validate inputs
    validateTensorProperties(positions, "positions");
    validateTensorProperties(velocities, "velocities");
    validateTensorProperties(masses, "masses");
    validateTensorProperties(forces, "forces");

    // Check dimensions
    auto num_particles = positions.size(0);
    TORCH_CHECK(positions.size(1) == 3, "positions must have shape [N, 3]");
    TORCH_CHECK(velocities.sizes() == positions.sizes(), "velocities must match positions shape");
    TORCH_CHECK(forces.sizes() == positions.sizes(), "forces must match positions shape");
    TORCH_CHECK(masses.size(0) == num_particles, "masses must have length N");

    // Get CUDA device and stream
    auto device = positions.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index());

    // Create output tensors (clone to ensure memory allocation)
    auto new_positions = positions.clone();
    auto new_velocities = velocities.clone();

    // Get data pointers
    float* pos_data = new_positions.data_ptr<float>();
    float* vel_data = new_velocities.data_ptr<float>();
    const float* force_data = forces.data_ptr<float>();
    const float* mass_data = masses.data_ptr<float>();

    // Split position and velocity data into x, y, z components
    float* pos_x = pos_data;
    float* pos_y = pos_data + num_particles;
    float* pos_z = pos_data + 2 * num_particles;

    float* vel_x = vel_data;
    float* vel_y = vel_data + num_particles;
    float* vel_z = vel_data + 2 * num_particles;

    const float* force_x = force_data;
    const float* force_y = force_data + num_particles;
    const float* force_z = force_data + 2 * num_particles;

    // Ensure tensors are properly laid out (N, 3) -> (3, N) for kernel
    auto pos_transposed = positions.transpose(0, 1).contiguous();
    auto vel_transposed = velocities.transpose(0, 1).contiguous();
    auto forces_transposed = forces.transpose(0, 1).contiguous();

    new_positions = pos_transposed.clone();
    new_velocities = vel_transposed.clone();

    // Launch CUDA kernel
    launch_physics_step_kernel(
        new_positions.data_ptr<float>(),                    // pos_x
        new_positions.data_ptr<float>() + num_particles,    // pos_y
        new_positions.data_ptr<float>() + 2*num_particles,  // pos_z
        new_velocities.data_ptr<float>(),                   // vel_x
        new_velocities.data_ptr<float>() + num_particles,   // vel_y
        new_velocities.data_ptr<float>() + 2*num_particles, // vel_z
        forces_transposed.data_ptr<float>(),                // force_x
        forces_transposed.data_ptr<float>() + num_particles,    // force_y
        forces_transposed.data_ptr<float>() + 2*num_particles,  // force_z
        masses.data_ptr<float>(),
        static_cast<int>(num_particles),
        dt,
        stream
    );

    // Transpose back to (N, 3) format
    new_positions = new_positions.transpose(0, 1).contiguous();
    new_velocities = new_velocities.transpose(0, 1).contiguous();

    return std::make_tuple(new_positions, new_velocities);
}

torch::Tensor computeForcesTorch(
    torch::Tensor positions,
    torch::Tensor velocities,
    torch::Tensor masses
) {
    // Validate inputs
    validateTensorProperties(positions, "positions");
    validateTensorProperties(velocities, "velocities");
    validateTensorProperties(masses, "masses");

    auto num_particles = positions.size(0);
    auto device = positions.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index());

    // Create output forces tensor
    auto forces = torch::zeros_like(positions);

    // Transpose for kernel (N, 3) -> (3, N)
    auto pos_transposed = positions.transpose(0, 1).contiguous();
    auto vel_transposed = velocities.transpose(0, 1).contiguous();
    auto forces_transposed = torch::zeros({3, num_particles},
                                        torch::dtype(torch::kFloat32).device(device));

    // Launch force computation kernel
    launch_force_computation_kernel(
        pos_transposed.data_ptr<float>(),                    // pos_x
        pos_transposed.data_ptr<float>() + num_particles,    // pos_y
        pos_transposed.data_ptr<float>() + 2*num_particles,  // pos_z
        vel_transposed.data_ptr<float>(),                    // vel_x
        vel_transposed.data_ptr<float>() + num_particles,    // vel_y
        vel_transposed.data_ptr<float>() + 2*num_particles,  // vel_z
        forces_transposed.data_ptr<float>(),                 // force_x
        forces_transposed.data_ptr<float>() + num_particles,     // force_y
        forces_transposed.data_ptr<float>() + 2*num_particles,   // force_z
        masses.data_ptr<float>(),
        static_cast<int>(num_particles),
        stream
    );

    // Transpose back to (N, 3)
    forces = forces_transposed.transpose(0, 1).contiguous();

    return forces;
}

std::tuple<torch::Tensor, torch::Tensor> applyConstraintsTorch(
    torch::Tensor positions,
    torch::Tensor velocities,
    torch::Tensor masses,
    torch::Tensor constraint_params
) {
    // Validate inputs
    validateTensorProperties(positions, "positions");
    validateTensorProperties(velocities, "velocities");
    validateTensorProperties(masses, "masses");
    validateTensorProperties(constraint_params, "constraint_params");

    auto num_particles = positions.size(0);
    auto num_constraints = constraint_params.size(0);
    auto device = positions.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index());

    // Create output tensors
    auto new_positions = positions.clone();
    auto new_velocities = velocities.clone();

    // Transpose for kernel
    auto pos_transposed = new_positions.transpose(0, 1).contiguous();
    auto vel_transposed = new_velocities.transpose(0, 1).contiguous();

    // Launch constraint kernel
    launch_constraint_kernel(
        pos_transposed.data_ptr<float>(),                    // pos_x
        pos_transposed.data_ptr<float>() + num_particles,    // pos_y
        pos_transposed.data_ptr<float>() + 2*num_particles,  // pos_z
        vel_transposed.data_ptr<float>(),                    // vel_x
        vel_transposed.data_ptr<float>() + num_particles,    // vel_y
        vel_transposed.data_ptr<float>() + 2*num_particles,  // vel_z
        masses.data_ptr<float>(),
        static_cast<int>(num_particles),
        constraint_params.data_ptr<float>(),
        static_cast<int>(num_constraints),
        0.01f, // dt - should be passed as parameter
        stream
    );

    // Transpose back
    new_positions = pos_transposed.transpose(0, 1).contiguous();
    new_velocities = vel_transposed.transpose(0, 1).contiguous();

    return std::make_tuple(new_positions, new_velocities);
}

// Autograd Function Implementation
torch::autograd::variable_list TorchPhysicsFunction::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor positions,
    torch::Tensor velocities,
    torch::Tensor masses,
    torch::Tensor forces,
    double dt
) {
    // Save tensors for backward pass
    ctx->save_for_backward({positions, velocities, masses, forces});
    ctx->saved_data["dt"] = dt;

    // Forward simulation step
    auto result = simulationStepTorch(positions, velocities, masses, forces, static_cast<float>(dt));

    return {std::get<0>(result), std::get<1>(result)};
}

torch::autograd::variable_list TorchPhysicsFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs
) {
    // Retrieve saved tensors
    auto saved = ctx->get_saved_variables();
    auto positions = saved[0];
    auto velocities = saved[1];
    auto masses = saved[2];
    auto forces = saved[3];
    auto dt = ctx->saved_data["dt"].toDouble();

    auto grad_positions = grad_outputs[0];
    auto grad_velocities = grad_outputs[1];

    // Compute gradients using adjoint method
    auto gradients = computePhysicsGradients(
        positions, velocities, masses, forces,
        grad_positions, grad_velocities, static_cast<float>(dt)
    );

    return {
        std::get<0>(gradients), // grad_positions
        std::get<1>(gradients), // grad_velocities
        std::get<2>(gradients), // grad_masses
        std::get<3>(gradients), // grad_forces
        torch::Tensor()         // grad_dt (no gradient for scalar)
    };
}

torch::autograd::variable_list TorchPhysicsFunction::apply(
    torch::Tensor positions,
    torch::Tensor velocities,
    torch::Tensor masses,
    torch::Tensor forces,
    double dt
) {
    return forward(nullptr, positions, velocities, masses, forces, dt);
}

void TorchPhysicsFunction::setup_context(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& inputs,
    const torch::autograd::variable_list& outputs
) {
    // This is called automatically by PyTorch
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
computePhysicsGradients(
    torch::Tensor positions,
    torch::Tensor velocities,
    torch::Tensor masses,
    torch::Tensor forces,
    torch::Tensor grad_positions,
    torch::Tensor grad_velocities,
    float dt
) {
    auto device = positions.device();
    auto num_particles = positions.size(0);

    // Initialize gradients
    auto grad_pos_input = torch::zeros_like(positions);
    auto grad_vel_input = torch::zeros_like(velocities);
    auto grad_masses_input = torch::zeros_like(masses);
    auto grad_forces_input = torch::zeros_like(forces);

    // Simplified adjoint computation
    // In practice, this would involve solving the adjoint equations

    // Gradient w.r.t. initial positions (simplified)
    grad_pos_input = grad_positions + grad_velocities * dt;

    // Gradient w.r.t. initial velocities
    grad_vel_input = grad_velocities;

    // Gradient w.r.t. forces
    grad_forces_input = grad_velocities * dt / masses.unsqueeze(1);

    // Gradient w.r.t. masses (simplified)
    auto force_over_mass_sq = forces / (masses.unsqueeze(1).pow(2));
    grad_masses_input = torch::sum(grad_velocities * force_over_mass_sq * dt, /*dim=*/1);

    return std::make_tuple(grad_pos_input, grad_vel_input, grad_masses_input, grad_forces_input);
}

// Utility functions for tensor conversion
std::vector<float> tensorToVector(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, "Tensor must be float32");

    auto cpu_tensor = tensor.cpu();
    auto data_ptr = cpu_tensor.data_ptr<float>();
    auto size = cpu_tensor.numel();

    return std::vector<float>(data_ptr, data_ptr + size);
}

torch::Tensor vectorToTensor(const std::vector<float>& vec, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto tensor = torch::from_blob(
        const_cast<float*>(vec.data()),
        {static_cast<long>(vec.size())},
        options
    );
    return tensor.clone(); // Clone to own the data
}

// Memory management utilities
class TensorMemoryPool {
private:
    std::unordered_map<size_t, std::vector<torch::Tensor>> pools;
    torch::Device device;

public:
    TensorMemoryPool(torch::Device device) : device(device) {}

    torch::Tensor getTensor(const std::vector<long>& shape) {
        size_t size = 1;
        for (auto dim : shape) {
            size *= dim;
        }

        auto& pool = pools[size];
        if (!pool.empty()) {
            auto tensor = pool.back();
            pool.pop_back();
            return tensor.reshape(shape);
        }

        // Create new tensor
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        return torch::zeros(shape, options);
    }

    void returnTensor(torch::Tensor tensor) {
        auto size = tensor.numel();
        pools[size].push_back(tensor.flatten());
    }

    void clear() {
        pools.clear();
    }
};

// Global memory pool instance
static thread_local std::unique_ptr<TensorMemoryPool> g_memory_pool = nullptr;

torch::Tensor getTensorFromPool(const std::vector<long>& shape, torch::Device device) {
    if (!g_memory_pool) {
        g_memory_pool = std::make_unique<TensorMemoryPool>(device);
    }
    return g_memory_pool->getTensor(shape);
}

void returnTensorToPool(torch::Tensor tensor) {
    if (g_memory_pool) {
        g_memory_pool->returnTensor(tensor);
    }
}

// Batch processing utilities
std::vector<torch::Tensor> batchSimulationSteps(
    const std::vector<torch::Tensor>& positions_batch,
    const std::vector<torch::Tensor>& velocities_batch,
    const std::vector<torch::Tensor>& masses_batch,
    const std::vector<torch::Tensor>& forces_batch,
    float dt
) {
    std::vector<torch::Tensor> results;
    results.reserve(positions_batch.size() * 2);

    for (size_t i = 0; i < positions_batch.size(); ++i) {
        auto result = simulationStepTorch(
            positions_batch[i], velocities_batch[i],
            masses_batch[i], forces_batch[i], dt
        );
        results.push_back(std::get<0>(result)); // positions
        results.push_back(std::get<1>(result)); // velocities
    }

    return results;
}

} // namespace torch_integration