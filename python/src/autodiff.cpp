#include "autodiff.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <stdexcept>

namespace autodiff {

// Variable implementation
Variable::Variable(const std::vector<float>& val, const std::vector<int64_t>& shp, bool grad)
    : value(val), shape(shp), requires_grad(grad) {
    if (requires_grad) {
        gradient.resize(value.size(), 0.0f);
    }
}

Variable::Variable(std::vector<float>&& val, const std::vector<int64_t>& shp, bool grad)
    : value(std::move(val)), shape(shp), requires_grad(grad) {
    if (requires_grad) {
        gradient.resize(value.size(), 0.0f);
    }
}

Variable Variable::operator+(const Variable& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shape mismatch in addition");
    }

    std::vector<float> result(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        result[i] = value[i] + other.value[i];
    }

    Variable var(std::move(result), shape, requires_grad || other.requires_grad);

    if (var.requires_grad) {
        auto node = std::make_shared<ComputationNode>("add");
        node->backward_fn = [this, &other](const std::vector<Variable>& inputs, Variable& output) {
            if (this->requires_grad) {
                for (size_t i = 0; i < this->gradient.size(); ++i) {
                    this->gradient[i] += output.gradient[i];
                }
            }
            if (other.requires_grad) {
                for (size_t i = 0; i < other.gradient.size(); ++i) {
                    other.gradient[i] += output.gradient[i];
                }
            }
        };
        var.computation_node = node;
    }

    return var;
}

Variable Variable::operator-(const Variable& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shape mismatch in subtraction");
    }

    std::vector<float> result(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        result[i] = value[i] - other.value[i];
    }

    Variable var(std::move(result), shape, requires_grad || other.requires_grad);

    if (var.requires_grad) {
        auto node = std::make_shared<ComputationNode>("sub");
        node->backward_fn = [this, &other](const std::vector<Variable>& inputs, Variable& output) {
            if (this->requires_grad) {
                for (size_t i = 0; i < this->gradient.size(); ++i) {
                    this->gradient[i] += output.gradient[i];
                }
            }
            if (other.requires_grad) {
                for (size_t i = 0; i < other.gradient.size(); ++i) {
                    other.gradient[i] -= output.gradient[i];
                }
            }
        };
        var.computation_node = node;
    }

    return var;
}

Variable Variable::operator*(const Variable& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shape mismatch in multiplication");
    }

    std::vector<float> result(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        result[i] = value[i] * other.value[i];
    }

    Variable var(std::move(result), shape, requires_grad || other.requires_grad);

    if (var.requires_grad) {
        auto node = std::make_shared<ComputationNode>("mul");
        node->backward_fn = [this, &other](const std::vector<Variable>& inputs, Variable& output) {
            if (this->requires_grad) {
                for (size_t i = 0; i < this->gradient.size(); ++i) {
                    this->gradient[i] += output.gradient[i] * other.value[i];
                }
            }
            if (other.requires_grad) {
                for (size_t i = 0; i < other.gradient.size(); ++i) {
                    other.gradient[i] += output.gradient[i] * this->value[i];
                }
            }
        };
        var.computation_node = node;
    }

    return var;
}

Variable Variable::operator/(const Variable& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shape mismatch in division");
    }

    std::vector<float> result(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if (std::abs(other.value[i]) < 1e-8f) {
            throw std::runtime_error("Division by zero");
        }
        result[i] = value[i] / other.value[i];
    }

    Variable var(std::move(result), shape, requires_grad || other.requires_grad);

    if (var.requires_grad) {
        auto node = std::make_shared<ComputationNode>("div");
        node->backward_fn = [this, &other](const std::vector<Variable>& inputs, Variable& output) {
            if (this->requires_grad) {
                for (size_t i = 0; i < this->gradient.size(); ++i) {
                    this->gradient[i] += output.gradient[i] / other.value[i];
                }
            }
            if (other.requires_grad) {
                for (size_t i = 0; i < other.gradient.size(); ++i) {
                    other.gradient[i] -= output.gradient[i] * this->value[i] / (other.value[i] * other.value[i]);
                }
            }
        };
        var.computation_node = node;
    }

    return var;
}

size_t Variable::size() const {
    return value.size();
}

void Variable::reshape(const std::vector<int64_t>& new_shape) {
    size_t new_size = 1;
    for (auto dim : new_shape) {
        new_size *= dim;
    }

    if (new_size != value.size()) {
        throw std::invalid_argument("New shape incompatible with current size");
    }

    shape = new_shape;
}

void Variable::zero_grad() {
    std::fill(gradient.begin(), gradient.end(), 0.0f);
}

// AutoDiffEngine implementation
AutoDiffEngine::AutoDiffEngine(GradientMode grad_mode) : mode(grad_mode) {
    // Register basic physics operations
    register_operation("physics_step", [this](const std::vector<Variable>& inputs) -> Variable {
        // This would call the actual physics kernel
        // For now, a simplified implementation
        if (inputs.size() != 5) {
            throw std::invalid_argument("physics_step requires 5 inputs");
        }

        const auto& positions = inputs[0];
        const auto& velocities = inputs[1];
        const auto& masses = inputs[2];
        const auto& forces = inputs[3];
        float dt = inputs[4].value[0]; // Assuming scalar dt

        // Simple Euler integration
        auto new_velocities = velocities + (forces / masses) * dt;
        auto new_positions = positions + new_velocities * dt;

        return new_positions;
    });
}

void AutoDiffEngine::clear_graph() {
    computation_graph.clear();
}

void AutoDiffEngine::backward(const Variable& loss) {
    if (!loss.requires_grad) {
        return;
    }

    // Initialize loss gradient
    loss.gradient[0] = 1.0f;

    // Traverse computation graph in reverse order
    for (auto it = computation_graph.rbegin(); it != computation_graph.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn({}, (*it)->output);
        }
    }
}

Variable AutoDiffEngine::physics_step(const Variable& positions, const Variable& velocities,
                                     const Variable& masses, const Variable& forces, float dt) {
    // Create dt as a variable
    Variable dt_var({dt}, {1}, false);

    auto op_it = operations.find("physics_step");
    if (op_it != operations.end()) {
        return op_it->second({positions, velocities, masses, forces, dt_var});
    }

    throw std::runtime_error("physics_step operation not registered");
}

Variable AutoDiffEngine::kinetic_energy(const Variable& velocities, const Variable& masses) {
    if (velocities.shape.size() != 2 || velocities.shape[1] != 3) {
        throw std::invalid_argument("Velocities must have shape [N, 3]");
    }

    int num_particles = velocities.shape[0];
    std::vector<float> energies(num_particles);

    for (int i = 0; i < num_particles; ++i) {
        float vx = velocities.value[i * 3 + 0];
        float vy = velocities.value[i * 3 + 1];
        float vz = velocities.value[i * 3 + 2];
        float mass = masses.value[i];

        energies[i] = 0.5f * mass * (vx * vx + vy * vy + vz * vz);
    }

    // Sum all kinetic energies
    float total_ke = std::accumulate(energies.begin(), energies.end(), 0.0f);

    Variable result({total_ke}, {1}, velocities.requires_grad || masses.requires_grad);

    if (result.requires_grad) {
        auto node = std::make_shared<ComputationNode>("kinetic_energy");
        node->backward_fn = [&velocities, &masses, num_particles](const std::vector<Variable>& inputs, Variable& output) {
            float grad_output = output.gradient[0];

            if (velocities.requires_grad) {
                for (int i = 0; i < num_particles; ++i) {
                    float mass = masses.value[i];
                    velocities.gradient[i * 3 + 0] += grad_output * mass * velocities.value[i * 3 + 0];
                    velocities.gradient[i * 3 + 1] += grad_output * mass * velocities.value[i * 3 + 1];
                    velocities.gradient[i * 3 + 2] += grad_output * mass * velocities.value[i * 3 + 2];
                }
            }

            if (masses.requires_grad) {
                for (int i = 0; i < num_particles; ++i) {
                    float vx = velocities.value[i * 3 + 0];
                    float vy = velocities.value[i * 3 + 1];
                    float vz = velocities.value[i * 3 + 2];
                    masses.gradient[i] += grad_output * 0.5f * (vx * vx + vy * vy + vz * vz);
                }
            }
        };
        result.computation_node = node;
        computation_graph.push_back(node);
    }

    return result;
}

Variable AutoDiffEngine::position_loss(const Variable& predicted, const Variable& target) {
    if (predicted.shape != target.shape) {
        throw std::invalid_argument("Predicted and target shapes must match");
    }

    // Mean squared error
    float mse = 0.0f;
    for (size_t i = 0; i < predicted.value.size(); ++i) {
        float diff = predicted.value[i] - target.value[i];
        mse += diff * diff;
    }
    mse /= predicted.value.size();

    Variable result({mse}, {1}, predicted.requires_grad);

    if (result.requires_grad) {
        auto node = std::make_shared<ComputationNode>("mse_loss");
        node->backward_fn = [&predicted, &target](const std::vector<Variable>& inputs, Variable& output) {
            float grad_output = output.gradient[0];
            float scale = 2.0f * grad_output / predicted.value.size();

            for (size_t i = 0; i < predicted.gradient.size(); ++i) {
                predicted.gradient[i] += scale * (predicted.value[i] - target.value[i]);
            }
        };
        result.computation_node = node;
        computation_graph.push_back(node);
    }

    return result;
}

void AutoDiffEngine::register_operation(const std::string& name,
                                       std::function<Variable(const std::vector<Variable>&)> forward_fn) {
    operations[name] = forward_fn;
}

// SGD Optimizer implementation
void SGDOptimizer::step(std::vector<Variable>& parameters, const std::vector<Variable>& gradients) {
    if (parameters.size() != gradients.size()) {
        throw std::invalid_argument("Parameters and gradients size mismatch");
    }

    // Initialize momentum buffers if needed
    if (momentum > 0.0f && momentum_buffers.size() != parameters.size()) {
        momentum_buffers.clear();
        for (const auto& param : parameters) {
            momentum_buffers.emplace_back(std::vector<float>(param.value.size(), 0.0f), param.shape, false);
        }
    }

    for (size_t i = 0; i < parameters.size(); ++i) {
        auto& param = parameters[i];
        const auto& grad = gradients[i];

        if (momentum > 0.0f) {
            auto& momentum_buf = momentum_buffers[i];
            // momentum_buf = momentum * momentum_buf + learning_rate * grad
            for (size_t j = 0; j < param.value.size(); ++j) {
                momentum_buf.value[j] = momentum * momentum_buf.value[j] + learning_rate * grad.gradient[j];
                param.value[j] -= momentum_buf.value[j];
            }
        } else {
            // Simple SGD
            for (size_t j = 0; j < param.value.size(); ++j) {
                param.value[j] -= learning_rate * grad.gradient[j];
            }
        }
    }
}

void SGDOptimizer::zero_grad(std::vector<Variable>& parameters) {
    for (auto& param : parameters) {
        param.zero_grad();
    }
}

// Adam Optimizer implementation
void AdamOptimizer::step(std::vector<Variable>& parameters, const std::vector<Variable>& gradients) {
    if (parameters.size() != gradients.size()) {
        throw std::invalid_argument("Parameters and gradients size mismatch");
    }

    step_count++;

    // Initialize buffers if needed
    if (m_buffers.size() != parameters.size()) {
        m_buffers.clear();
        v_buffers.clear();
        for (const auto& param : parameters) {
            m_buffers.emplace_back(std::vector<float>(param.value.size(), 0.0f), param.shape, false);
            v_buffers.emplace_back(std::vector<float>(param.value.size(), 0.0f), param.shape, false);
        }
    }

    float bias_correction1 = 1.0f - std::pow(beta1, step_count);
    float bias_correction2 = 1.0f - std::pow(beta2, step_count);

    for (size_t i = 0; i < parameters.size(); ++i) {
        auto& param = parameters[i];
        const auto& grad = gradients[i];
        auto& m = m_buffers[i];
        auto& v = v_buffers[i];

        for (size_t j = 0; j < param.value.size(); ++j) {
            float g = grad.gradient[j];

            // Update biased first moment estimate
            m.value[j] = beta1 * m.value[j] + (1.0f - beta1) * g;

            // Update biased second raw moment estimate
            v.value[j] = beta2 * v.value[j] + (1.0f - beta2) * g * g;

            // Compute bias-corrected first moment estimate
            float m_hat = m.value[j] / bias_correction1;

            // Compute bias-corrected second raw moment estimate
            float v_hat = v.value[j] / bias_correction2;

            // Update parameters
            param.value[j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
}

void AdamOptimizer::zero_grad(std::vector<Variable>& parameters) {
    for (auto& param : parameters) {
        param.zero_grad();
    }
}

// PhysicsOptimizer implementation
PhysicsOptimizer::PhysicsOptimizer(GradientMode mode) {
    engine = std::make_unique<AutoDiffEngine>(mode);
    optimizer = std::make_unique<AdamOptimizer>();
}

void PhysicsOptimizer::optimize_parameters(std::vector<Variable>& parameters,
                                         const std::vector<Variable>& initial_states,
                                         const std::vector<Variable>& target_trajectories,
                                         int num_iterations, float dt) {
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Zero gradients
        optimizer->zero_grad(parameters);

        // Forward simulation
        auto current_state = initial_states[0];
        Variable total_loss = utils::zeros({1}, true);

        for (size_t step = 0; step < target_trajectories.size(); ++step) {
            // Simulate one step (this would use the actual physics)
            // For now, just a placeholder
            auto predicted_state = current_state + parameters[0] * dt;

            // Compute loss
            auto step_loss = engine->position_loss(predicted_state, target_trajectories[step]);
            total_loss = total_loss + step_loss;

            current_state = predicted_state;
        }

        // Backward pass
        engine->backward(total_loss);

        // Update parameters
        std::vector<Variable> gradients;
        for (const auto& param : parameters) {
            gradients.emplace_back(param.gradient, param.shape, false);
        }
        optimizer->step(parameters, gradients);

        engine->clear_graph();
    }
}

// Utility functions
namespace utils {
    Variable create_variable(const std::vector<float>& data, const std::vector<int64_t>& shape, bool requires_grad) {
        return Variable(data, shape, requires_grad);
    }

    Variable zeros(const std::vector<int64_t>& shape, bool requires_grad) {
        size_t size = 1;
        for (auto dim : shape) {
            size *= dim;
        }
        return Variable(std::vector<float>(size, 0.0f), shape, requires_grad);
    }

    Variable ones(const std::vector<int64_t>& shape, bool requires_grad) {
        size_t size = 1;
        for (auto dim : shape) {
            size *= dim;
        }
        return Variable(std::vector<float>(size, 1.0f), shape, requires_grad);
    }

    Variable random_normal(const std::vector<int64_t>& shape, float mean, float std, bool requires_grad) {
        size_t size = 1;
        for (auto dim : shape) {
            size *= dim;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, std);

        std::vector<float> data(size);
        for (auto& val : data) {
            val = dist(gen);
        }

        return Variable(std::move(data), shape, requires_grad);
    }

    Variable sum(const Variable& x, int dim) {
        if (dim == -1) {
            // Sum all elements
            float total = std::accumulate(x.value.begin(), x.value.end(), 0.0f);
            Variable result({total}, {1}, x.requires_grad);

            if (result.requires_grad) {
                auto node = std::make_shared<ComputationNode>("sum_all");
                node->backward_fn = [&x](const std::vector<Variable>& inputs, Variable& output) {
                    float grad_output = output.gradient[0];
                    for (size_t i = 0; i < x.gradient.size(); ++i) {
                        x.gradient[i] += grad_output;
                    }
                };
                result.computation_node = node;
            }

            return result;
        } else {
            throw std::runtime_error("Dimensional sum not implemented yet");
        }
    }

    Variable mean(const Variable& x, int dim) {
        auto sum_result = sum(x, dim);
        float scale = 1.0f / x.value.size();

        Variable result({sum_result.value[0] * scale}, {1}, x.requires_grad);

        if (result.requires_grad) {
            auto node = std::make_shared<ComputationNode>("mean");
            node->backward_fn = [&x, scale](const std::vector<Variable>& inputs, Variable& output) {
                float grad_output = output.gradient[0];
                for (size_t i = 0; i < x.gradient.size(); ++i) {
                    x.gradient[i] += grad_output * scale;
                }
            };
            result.computation_node = node;
        }

        return result;
    }

    #ifdef WITH_PYTORCH
    Variable from_torch(const torch::Tensor& tensor, bool requires_grad) {
        auto cpu_tensor = tensor.cpu();
        auto data_ptr = cpu_tensor.data_ptr<float>();
        auto sizes = cpu_tensor.sizes();

        std::vector<float> data(data_ptr, data_ptr + cpu_tensor.numel());
        std::vector<int64_t> shape(sizes.begin(), sizes.end());

        return Variable(std::move(data), shape, requires_grad);
    }

    torch::Tensor to_torch(const Variable& var) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto tensor = torch::from_blob(
            const_cast<float*>(var.value.data()),
            var.shape,
            options
        );
        return tensor.clone(); // Clone to own the data
    }
    #endif
}

} // namespace autodiff