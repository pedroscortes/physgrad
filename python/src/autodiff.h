#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

#ifdef WITH_PYTORCH
#include <torch/torch.h>
#endif

#ifdef WITH_JAX
#include "jax_integration.h"
#endif

namespace autodiff {

// Forward declarations
struct ComputationNode;
using NodePtr = std::shared_ptr<ComputationNode>;

// Gradient computation modes
enum class GradientMode {
    FORWARD,          // Forward-mode AD
    REVERSE,          // Reverse-mode AD (backpropagation)
    ADJOINT_PHYSICS   // Adjoint method for physics
};

// Variable type for autodiff
struct Variable {
    std::vector<float> value;
    std::vector<float> gradient;
    std::vector<int64_t> shape;
    NodePtr computation_node;
    bool requires_grad = false;

    Variable() = default;
    Variable(const std::vector<float>& val, const std::vector<int64_t>& shp, bool grad = false);
    Variable(std::vector<float>&& val, const std::vector<int64_t>& shp, bool grad = false);

    // Operations
    Variable operator+(const Variable& other) const;
    Variable operator-(const Variable& other) const;
    Variable operator*(const Variable& other) const;
    Variable operator/(const Variable& other) const;

    // In-place operations
    Variable& operator+=(const Variable& other);
    Variable& operator-=(const Variable& other);
    Variable& operator*=(const Variable& other);
    Variable& operator/=(const Variable& other);

    // Shape utilities
    size_t size() const;
    void reshape(const std::vector<int64_t>& new_shape);
    void zero_grad();
};

// Computation graph node
struct ComputationNode {
    std::string operation_name;
    std::vector<NodePtr> inputs;
    std::function<void(const std::vector<Variable>&, Variable&)> backward_fn;
    Variable output;

    ComputationNode(const std::string& op_name) : operation_name(op_name) {}
};

// Automatic differentiation engine
class AutoDiffEngine {
private:
    GradientMode mode;
    std::vector<NodePtr> computation_graph;
    std::unordered_map<std::string, std::function<Variable(const std::vector<Variable>&)>> operations;

public:
    AutoDiffEngine(GradientMode grad_mode = GradientMode::REVERSE);

    // Core operations
    void clear_graph();
    void backward(const Variable& loss);

    // Register custom operations
    void register_operation(const std::string& name,
                           std::function<Variable(const std::vector<Variable>&)> forward_fn);

    // Physics-specific operations
    Variable physics_step(const Variable& positions, const Variable& velocities,
                         const Variable& masses, const Variable& forces, float dt);

    Variable compute_forces(const Variable& positions, const Variable& velocities,
                           const Variable& masses);

    std::pair<Variable, Variable> apply_constraints(const Variable& positions,
                                                   const Variable& velocities,
                                                   const Variable& masses,
                                                   const Variable& constraint_params);

    // Energy and loss functions
    Variable kinetic_energy(const Variable& velocities, const Variable& masses);
    Variable potential_energy(const Variable& positions, const Variable& masses);
    Variable total_energy(const Variable& positions, const Variable& velocities, const Variable& masses);

    // Loss functions for optimization
    Variable position_loss(const Variable& predicted, const Variable& target);
    Variable velocity_loss(const Variable& predicted, const Variable& target);
    Variable energy_conservation_loss(const Variable& energy_initial, const Variable& energy_final);

    // Gradient utilities
    std::vector<Variable> compute_gradients(const Variable& loss, const std::vector<Variable>& parameters);
    void update_parameters(std::vector<Variable>& parameters, const std::vector<Variable>& gradients, float learning_rate);

    // Advanced gradient methods
    Variable compute_adjoint_gradients(const Variable& positions, const Variable& velocities,
                                      const Variable& masses, const Variable& forces,
                                      const Variable& grad_output, float dt);

    // Set gradient mode
    void set_gradient_mode(GradientMode new_mode) { mode = new_mode; }
    GradientMode get_gradient_mode() const { return mode; }
};

// Optimization algorithms
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(std::vector<Variable>& parameters, const std::vector<Variable>& gradients) = 0;
    virtual void zero_grad(std::vector<Variable>& parameters) = 0;
};

class SGDOptimizer : public Optimizer {
private:
    float learning_rate;
    float momentum;
    std::vector<Variable> momentum_buffers;

public:
    SGDOptimizer(float lr, float mom = 0.0f) : learning_rate(lr), momentum(mom) {}

    void step(std::vector<Variable>& parameters, const std::vector<Variable>& gradients) override;
    void zero_grad(std::vector<Variable>& parameters) override;
};

class AdamOptimizer : public Optimizer {
private:
    float learning_rate;
    float beta1, beta2;
    float epsilon;
    std::vector<Variable> m_buffers; // First moment
    std::vector<Variable> v_buffers; // Second moment
    int step_count = 0;

public:
    AdamOptimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps) {}

    void step(std::vector<Variable>& parameters, const std::vector<Variable>& gradients) override;
    void zero_grad(std::vector<Variable>& parameters) override;
};

// Physics-specific optimization problems
class PhysicsOptimizer {
private:
    std::unique_ptr<AutoDiffEngine> engine;
    std::unique_ptr<Optimizer> optimizer;

public:
    PhysicsOptimizer(GradientMode mode = GradientMode::ADJOINT_PHYSICS);

    // Parameter estimation
    void optimize_parameters(std::vector<Variable>& parameters,
                           const std::vector<Variable>& initial_states,
                           const std::vector<Variable>& target_trajectories,
                           int num_iterations, float dt);

    // Trajectory optimization
    std::vector<Variable> optimize_trajectory(const Variable& initial_state,
                                            const Variable& target_state,
                                            int num_steps, float dt);

    // Control optimization
    std::vector<Variable> optimize_control(const Variable& initial_state,
                                         const Variable& target_trajectory,
                                         int num_steps, float dt);

    // System identification
    Variable estimate_masses(const std::vector<Variable>& position_trajectories,
                           const std::vector<Variable>& force_trajectories,
                           const Variable& initial_guess, int num_iterations, float dt);

    // Set optimizer
    void set_optimizer(std::unique_ptr<Optimizer> opt) { optimizer = std::move(opt); }
};

// Utility functions
namespace utils {
    Variable create_variable(const std::vector<float>& data, const std::vector<int64_t>& shape, bool requires_grad = false);
    Variable zeros(const std::vector<int64_t>& shape, bool requires_grad = false);
    Variable ones(const std::vector<int64_t>& shape, bool requires_grad = false);
    Variable random_normal(const std::vector<int64_t>& shape, float mean = 0.0f, float std = 1.0f, bool requires_grad = false);

    // Conversion utilities
    #ifdef WITH_PYTORCH
    Variable from_torch(const torch::Tensor& tensor, bool requires_grad = false);
    torch::Tensor to_torch(const Variable& var);
    #endif

    #ifdef WITH_JAX
    Variable from_jax(const jax_integration::JAXArray& array, bool requires_grad = false);
    jax_integration::JAXArray to_jax(const Variable& var);
    #endif

    // Mathematical operations
    Variable sin(const Variable& x);
    Variable cos(const Variable& x);
    Variable exp(const Variable& x);
    Variable log(const Variable& x);
    Variable sqrt(const Variable& x);
    Variable pow(const Variable& x, float exponent);

    // Reduction operations
    Variable sum(const Variable& x, int dim = -1);
    Variable mean(const Variable& x, int dim = -1);
    Variable norm(const Variable& x, float p = 2.0f);

    // Reshaping and indexing
    Variable reshape(const Variable& x, const std::vector<int64_t>& new_shape);
    Variable transpose(const Variable& x, int dim0, int dim1);
    Variable slice(const Variable& x, int dim, int start, int end);
}

} // namespace autodiff