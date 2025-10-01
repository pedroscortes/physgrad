/**
 * PhysGrad - Custom PyTorch Autograd Functions
 *
 * Implements custom autograd functions that bridge PhysGrad physics simulation
 * with PyTorch's automatic differentiation system. Enables end-to-end
 * differentiable physics for machine learning applications.
 */

#ifndef PHYSGRAD_PYTORCH_AUTOGRAD_H
#define PHYSGRAD_PYTORCH_AUTOGRAD_H

#include "common_types.h"
#include "adjoint_integrators_standalone.h"

// Conditional PyTorch integration
#ifdef PHYSGRAD_PYTORCH_AVAILABLE
    #include <torch/torch.h>
    #include <torch/extension.h>
#else
    // Mock PyTorch types for testing without PyTorch
    namespace torch {
        struct Tensor {
            std::vector<float> data;
            std::vector<int64_t> shape;

            Tensor() = default;
            Tensor(const std::vector<float>& d, const std::vector<int64_t>& s)
                : data(d), shape(s) {}

            template<typename T>
            T* data_ptr() { return reinterpret_cast<T*>(data.data()); }

            template<typename T>
            const T* data_ptr() const { return reinterpret_cast<const T*>(data.data()); }
            int64_t size(int dim) const { return shape[dim]; }
            int64_t numel() const {
                int64_t total = 1;
                for (auto s : shape) total *= s;
                return total;
            }

            Tensor clone() const { return Tensor(data, shape); }
            Tensor detach() const { return clone(); }
            void backward() const {}
            bool requires_grad() const { return true; }
            void set_requires_grad(bool) {}
        };

        namespace autograd {
            struct Function {
                virtual ~Function() = default;
            };

            template<typename T>
            struct FunctionCtx {
                std::vector<Tensor> saved_tensors;
                void save_for_backward(const std::vector<Tensor>& tensors) {
                    saved_tensors = tensors;
                }
                std::vector<Tensor> get_saved_variables() const {
                    return saved_tensors;
                }
            };
        }

        Tensor zeros(const std::vector<int64_t>& shape) {
            int64_t total = 1;
            for (auto s : shape) total *= s;
            return Tensor(std::vector<float>(total, 0.0f), shape);
        }

        Tensor ones_like(const Tensor& other) {
            return Tensor(std::vector<float>(other.data.size(), 1.0f), other.shape);
        }
    }
#endif

#include <vector>
#include <memory>
#include <functional>

namespace physgrad {
namespace pytorch {

// =============================================================================
// TENSOR CONVERSION UTILITIES
// =============================================================================

/**
 * Convert PyTorch tensor to PhysGrad vector format
 */
template<typename T>
std::vector<ConceptVector3D<T>> tensorToPositions(const torch::Tensor& tensor) {
    const T* data = tensor.data_ptr<T>();
    const int64_t n_particles = tensor.size(0);

    std::vector<ConceptVector3D<T>> positions(n_particles);
    for (int64_t i = 0; i < n_particles; ++i) {
        positions[i] = ConceptVector3D<T>{
            data[i * 3 + 0],
            data[i * 3 + 1],
            data[i * 3 + 2]
        };
    }
    return positions;
}

/**
 * Convert PhysGrad vector format to PyTorch tensor
 */
template<typename T>
torch::Tensor positionsToTensor(const std::vector<ConceptVector3D<T>>& positions) {
    const int64_t n_particles = positions.size();
    std::vector<T> data(n_particles * 3);

    for (int64_t i = 0; i < n_particles; ++i) {
        data[i * 3 + 0] = positions[i][0];
        data[i * 3 + 1] = positions[i][1];
        data[i * 3 + 2] = positions[i][2];
    }

    return torch::Tensor(std::vector<float>(data.begin(), data.end()),
                        {n_particles, 3});
}

/**
 * Convert scalar vector to PyTorch tensor
 */
template<typename T>
torch::Tensor scalarVectorToTensor(const std::vector<T>& scalars) {
    const int64_t n = scalars.size();
    std::vector<float> data(scalars.begin(), scalars.end());
    return torch::Tensor(data, {n});
}

// =============================================================================
// PHYSICS SIMULATION AUTOGRAD FUNCTION
// =============================================================================

/**
 * Custom autograd function for physics simulation
 * Integrates PhysGrad's adjoint methods with PyTorch's AD system
 */
#ifdef PHYSGRAD_PYTORCH_AVAILABLE
template<typename T>
class PhysicsSimulationFunction : public torch::autograd::Function<PhysicsSimulationFunction<T>> {
public:
    using SimulationEngine = adjoint::AdjointSimulation<T>;
    using ForceEngine = adjoint::SimpleForceEngine<T>;

    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& initial_positions,
        const torch::Tensor& initial_velocities,
        const torch::Tensor& masses,
        double timestep,
        int64_t num_steps) {

        // Convert PyTorch tensors to PhysGrad format
        auto positions = tensorToPositions<T>(initial_positions);
        auto velocities = tensorToPositions<T>(initial_velocities);
        auto mass_data = masses.data_ptr<float>();
        auto mass_vector = std::vector<T>(mass_data, mass_data + masses.numel());

        // Create force engine with springs (simplified)
        auto force_engine = std::make_shared<ForceEngine>();
        for (size_t i = 0; i < positions.size() - 1; ++i) {
            force_engine->addSpring(i, i + 1, T{10}, T{1}); // Simple chain
        }

        // Create simulation
        auto simulation = std::make_unique<SimulationEngine>(force_engine);

        // Run forward simulation
        simulation->runForward(positions, velocities, mass_vector,
                              static_cast<T>(timestep), num_steps);

        // Save context for backward pass
        ctx->save_for_backward({initial_positions, initial_velocities, masses});
        ctx->saved_data["timestep"] = timestep;
        ctx->saved_data["num_steps"] = num_steps;
        ctx->saved_data["simulation"] = simulation.release(); // Transfer ownership

        // Convert result back to tensor
        return positionsToTensor(positions);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {

        // Retrieve saved context
        auto saved = ctx->get_saved_variables();
        auto initial_positions = saved[0];
        auto initial_velocities = saved[1];
        auto masses = saved[2];

        double timestep = std::any_cast<double>(ctx->saved_data["timestep"]);
        int64_t num_steps = std::any_cast<int64_t>(ctx->saved_data["num_steps"]);
        auto simulation = static_cast<SimulationEngine*>(
            std::any_cast<void*>(ctx->saved_data["simulation"]));

        // Get gradient w.r.t. final positions
        auto grad_final_positions = tensorToPositions<T>(grad_outputs[0]);

        // Run backward pass through simulation
        std::vector<ConceptVector3D<T>> pos_grads, vel_grads;
        std::vector<T> mass_grads;
        std::vector<ConceptVector3D<T>> zero_vel_grads(grad_final_positions.size());

        simulation->runBackward(grad_final_positions, zero_vel_grads,
                               pos_grads, vel_grads, mass_grads);

        // Clean up
        delete simulation;

        // Convert gradients back to tensors
        auto grad_initial_positions = positionsToTensor(pos_grads);
        auto grad_initial_velocities = positionsToTensor(vel_grads);
        auto grad_masses = scalarVectorToTensor(mass_grads);

        // Return gradients (same order as forward inputs)
        return {grad_initial_positions, grad_initial_velocities, grad_masses,
                torch::Tensor{}, torch::Tensor{}}; // No grad for timestep, num_steps
    }
};
#else
// Mock implementation for testing without PyTorch
template<typename T>
class PhysicsSimulationFunction {
public:
    static torch::Tensor forward(
        void* ctx,
        const torch::Tensor& initial_positions,
        const torch::Tensor& initial_velocities,
        const torch::Tensor& masses,
        double timestep,
        int64_t num_steps) {

        // Simple mock: return slightly modified positions
        auto result = initial_positions.clone();
        for (size_t i = 0; i < result.data.size(); i += 3) {
            result.data[i] += static_cast<float>(timestep * num_steps * 0.01); // Simple displacement
        }
        return result;
    }

    static std::vector<torch::Tensor> backward(
        void* ctx,
        const std::vector<torch::Tensor>& grad_outputs) {

        // Simple mock gradients
        return {grad_outputs[0].clone(), grad_outputs[0].clone(),
                torch::zeros({grad_outputs[0].size(0)})};
    }
};
#endif

// =============================================================================
// HIGH-LEVEL PYTORCH INTERFACE
// =============================================================================

/**
 * High-level function to run differentiable physics simulation
 */
template<typename T = float>
torch::Tensor physics_simulation(
    const torch::Tensor& initial_positions,
    const torch::Tensor& initial_velocities,
    const torch::Tensor& masses,
    double timestep = 0.01,
    int64_t num_steps = 100) {

#ifdef PHYSGRAD_PYTORCH_AVAILABLE
    return PhysicsSimulationFunction<T>::apply(
        initial_positions, initial_velocities, masses, timestep, num_steps);
#else
    return PhysicsSimulationFunction<T>::forward(
        nullptr, initial_positions, initial_velocities, masses, timestep, num_steps);
#endif
}

/**
 * Convenience function for creating particle systems
 */
torch::Tensor create_particle_chain(int64_t n_particles, float spacing = 1.0f) {
    std::vector<float> positions;
    for (int64_t i = 0; i < n_particles; ++i) {
        positions.push_back(i * spacing); // x
        positions.push_back(0.0f);         // y
        positions.push_back(0.0f);         // z
    }
    return torch::Tensor(positions, {n_particles, 3});
}

/**
 * Create zero initial velocities
 */
torch::Tensor create_zero_velocities(int64_t n_particles) {
    std::vector<float> velocities(n_particles * 3, 0.0f);
    return torch::Tensor(velocities, {n_particles, 3});
}

/**
 * Create uniform masses
 */
torch::Tensor create_uniform_masses(int64_t n_particles, float mass = 1.0f) {
    std::vector<float> masses(n_particles, mass);
    return torch::Tensor(masses, {n_particles});
}

// =============================================================================
// LOSS FUNCTIONS FOR PHYSICS-BASED LEARNING
// =============================================================================

/**
 * Position-based loss function
 */
torch::Tensor position_loss(const torch::Tensor& final_positions,
                           const torch::Tensor& target_positions) {
    // Simple L2 loss
    float loss = 0.0f;
    for (size_t i = 0; i < final_positions.data.size(); ++i) {
        float diff = final_positions.data[i] - target_positions.data[i];
        loss += diff * diff;
    }
    return torch::Tensor({loss}, {1});
}

/**
 * Energy conservation loss
 */
torch::Tensor energy_conservation_loss(const torch::Tensor& positions,
                                      const torch::Tensor& velocities,
                                      const torch::Tensor& masses,
                                      float target_energy) {
    // Compute kinetic energy
    float kinetic = 0.0f;
    const float* vel_data = velocities.data_ptr<float>();
    const float* mass_data = masses.data_ptr<float>();

    for (int64_t i = 0; i < masses.numel(); ++i) {
        float v_sq = vel_data[i*3]*vel_data[i*3] +
                     vel_data[i*3+1]*vel_data[i*3+1] +
                     vel_data[i*3+2]*vel_data[i*3+2];
        kinetic += 0.5f * mass_data[i] * v_sq;
    }

    // Compute potential energy (simplified spring model)
    float potential = 0.0f;
    const float* pos_data = positions.data_ptr<float>();
    for (int64_t i = 0; i < masses.numel() - 1; ++i) {
        float dx = pos_data[(i+1)*3] - pos_data[i*3];
        float dy = pos_data[(i+1)*3+1] - pos_data[i*3+1];
        float dz = pos_data[(i+1)*3+2] - pos_data[i*3+2];
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        potential += 0.5f * 10.0f * (dist - 1.0f) * (dist - 1.0f); // k=10, r0=1
    }

    float total_energy = kinetic + potential;
    float energy_error = (total_energy - target_energy) * (total_energy - target_energy);

    return torch::Tensor({energy_error}, {1});
}

/**
 * Physics-informed loss combining multiple objectives
 */
torch::Tensor physics_informed_loss(const torch::Tensor& final_positions,
                                   const torch::Tensor& final_velocities,
                                   const torch::Tensor& masses,
                                   const torch::Tensor& target_positions,
                                   float target_energy,
                                   float position_weight = 1.0f,
                                   float energy_weight = 0.1f) {

    auto pos_loss = position_loss(final_positions, target_positions);
    auto energy_loss = energy_conservation_loss(final_positions, final_velocities,
                                               masses, target_energy);

    float total_loss = position_weight * pos_loss.data[0] +
                      energy_weight * energy_loss.data[0];

    return torch::Tensor({total_loss}, {1});
}

// =============================================================================
// EXAMPLE USAGE AND VALIDATION
// =============================================================================

/**
 * Example training loop for physics-based learning
 */
template<typename T = float>
class PhysicsBasedLearning {
public:
    PhysicsBasedLearning(int64_t n_particles) : n_particles_(n_particles) {
        // Initialize learnable parameters
        initial_positions_ = create_particle_chain(n_particles);
        initial_velocities_ = create_zero_velocities(n_particles);
        masses_ = create_uniform_masses(n_particles);

        // Set requires_grad for learnable parameters
        initial_positions_.set_requires_grad(true);
        masses_.set_requires_grad(true);
    }

    torch::Tensor forward(double timestep = 0.01, int64_t num_steps = 100) {
        return physics_simulation<T>(initial_positions_, initial_velocities_,
                                   masses_, timestep, num_steps);
    }

    torch::Tensor compute_loss(const torch::Tensor& target_positions,
                              float target_energy) {
        auto final_positions = forward();
        auto final_velocities = create_zero_velocities(n_particles_); // Simplified

        return physics_informed_loss(final_positions, final_velocities, masses_,
                                    target_positions, target_energy);
    }

    void optimization_step(const torch::Tensor& target_positions,
                          float target_energy, float learning_rate = 0.01f) {
        auto loss = compute_loss(target_positions, target_energy);

        // Backward pass
        loss.backward();

        // Simple SGD update (mock implementation)
        for (size_t i = 0; i < initial_positions_.data.size(); ++i) {
            initial_positions_.data[i] -= learning_rate * 0.01f; // Mock gradient
        }
        for (size_t i = 0; i < masses_.data.size(); ++i) {
            masses_.data[i] -= learning_rate * 0.001f; // Mock gradient
        }
    }

    const torch::Tensor& getPositions() const { return initial_positions_; }
    const torch::Tensor& getMasses() const { return masses_; }

private:
    int64_t n_particles_;
    torch::Tensor initial_positions_;
    torch::Tensor initial_velocities_;
    torch::Tensor masses_;
};

} // namespace pytorch
} // namespace physgrad

#endif // PHYSGRAD_PYTORCH_AUTOGRAD_H