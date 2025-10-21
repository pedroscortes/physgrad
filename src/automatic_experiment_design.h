/**
 * PhysGrad Automatic Experiment Design Framework
 *
 * AI-driven parameter space exploration and experiment design for physics simulations.
 * Automatically discovers optimal parameters, configurations, and behaviors using
 * advanced optimization algorithms, Bayesian optimization, and multi-objective search.
 */

#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <thread>
#include <atomic>
#include <chrono>
#include <queue>
#include <mutex>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#endif

namespace physgrad {
namespace experiment_design {

// Forward declarations
template<typename T> class ParameterSpace;
template<typename T> class ExperimentDesigner;
template<typename T> class BayesianOptimizer;
template<typename T> class MultiObjectiveOptimizer;

// =============================================================================
// PARAMETER SPACE DEFINITION
// =============================================================================

// Parameter types for different domains
enum class ParameterType {
    CONTINUOUS,     // Real-valued parameters (e.g., mass, velocity, temperature)
    DISCRETE,       // Integer parameters (e.g., particle count, grid resolution)
    CATEGORICAL,    // Categorical choices (e.g., material type, boundary condition)
    BOOLEAN,        // Binary choices (e.g., enable/disable features)
    ORDINAL        // Ordered discrete values (e.g., quality levels)
};

// Distribution types for parameter sampling
enum class DistributionType {
    UNIFORM,        // Uniform distribution
    NORMAL,         // Gaussian distribution
    LOG_UNIFORM,    // Log-uniform for scale parameters
    LOG_NORMAL,     // Log-normal distribution
    BETA,           // Beta distribution for bounded parameters
    GAMMA,          // Gamma distribution for positive parameters
    CUSTOM          // User-defined distribution
};

// Parameter constraint types
enum class ConstraintType {
    NONE,           // No constraints
    BOUNDS,         // Simple min/max bounds
    LINEAR,         // Linear inequality constraints
    NONLINEAR,      // Nonlinear constraint functions
    CONDITIONAL     // Parameter depends on other parameters
};

// Individual parameter definition
template<typename T>
struct Parameter {
    std::string name;
    ParameterType type;
    DistributionType distribution;
    ConstraintType constraint;

    // Value bounds and properties
    T min_value;
    T max_value;
    T default_value;
    std::vector<T> discrete_values;     // For discrete/categorical parameters
    std::vector<std::string> categories; // For categorical parameters

    // Distribution parameters
    T distribution_param1;  // e.g., mean for normal, alpha for beta
    T distribution_param2;  // e.g., std for normal, beta for beta

    // Constraints
    std::function<bool(T)> constraint_function;
    std::vector<std::string> dependent_parameters;

    // Metadata
    std::string description;
    std::string units;
    bool is_optimization_target;
    T optimization_weight;

    Parameter() : type(ParameterType::CONTINUOUS),
                  distribution(DistributionType::UNIFORM),
                  constraint(ConstraintType::BOUNDS),
                  min_value(0), max_value(1), default_value(0.5),
                  distribution_param1(0), distribution_param2(1),
                  is_optimization_target(false), optimization_weight(1) {}
};

// Parameter space configuration
template<typename T>
class ParameterSpace {
private:
    std::vector<Parameter<T>> parameters_;
    std::unordered_map<std::string, size_t> parameter_index_;
    std::vector<std::function<bool(const std::vector<T>&)>> global_constraints_;

    // Random number generation
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<T> uniform_dist_;
    mutable std::normal_distribution<T> normal_dist_;

public:
    explicit ParameterSpace(unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count())
        : rng_(seed), uniform_dist_(0.0, 1.0), normal_dist_(0.0, 1.0) {}

    // Parameter management
    void addParameter(const Parameter<T>& param) {
        parameter_index_[param.name] = parameters_.size();
        parameters_.push_back(param);
    }

    void addGlobalConstraint(std::function<bool(const std::vector<T>&)> constraint) {
        global_constraints_.push_back(constraint);
    }

    // Parameter access
    size_t getParameterCount() const { return parameters_.size(); }
    const Parameter<T>& getParameter(size_t index) const { return parameters_[index]; }
    const Parameter<T>& getParameter(const std::string& name) const {
        return parameters_[parameter_index_.at(name)];
    }

    size_t getParameterIndex(const std::string& name) const {
        return parameter_index_.at(name);
    }

    // Parameter sampling
    std::vector<T> sampleParameters() const {
        std::vector<T> values(parameters_.size());

        for (size_t i = 0; i < parameters_.size(); ++i) {
            values[i] = sampleParameter(i);
        }

        // Apply global constraints
        if (!satisfiesConstraints(values)) {
            // Rejection sampling (simple approach)
            for (int attempts = 0; attempts < 1000; ++attempts) {
                for (size_t i = 0; i < parameters_.size(); ++i) {
                    values[i] = sampleParameter(i);
                }
                if (satisfiesConstraints(values)) break;
            }
        }

        return values;
    }

    std::vector<T> sampleParametersUniform() const {
        std::vector<T> values(parameters_.size());

        for (size_t i = 0; i < parameters_.size(); ++i) {
            const auto& param = parameters_[i];
            switch (param.type) {
                case ParameterType::CONTINUOUS:
                    values[i] = param.min_value + uniform_dist_(rng_) * (param.max_value - param.min_value);
                    break;
                case ParameterType::DISCRETE:
                    if (!param.discrete_values.empty()) {
                        size_t idx = static_cast<size_t>(uniform_dist_(rng_) * param.discrete_values.size());
                        values[i] = param.discrete_values[idx];
                    } else {
                        values[i] = param.min_value + static_cast<T>(
                            static_cast<int>(uniform_dist_(rng_) * (param.max_value - param.min_value + 1))
                        );
                    }
                    break;
                case ParameterType::BOOLEAN:
                    values[i] = uniform_dist_(rng_) > 0.5 ? T(1) : T(0);
                    break;
                default:
                    values[i] = param.default_value;
                    break;
            }
        }

        return values;
    }

    // Latin Hypercube Sampling for better space coverage
    std::vector<std::vector<T>> latinHypercubeSampling(size_t num_samples) const {
        std::vector<std::vector<T>> samples(num_samples, std::vector<T>(parameters_.size()));

        // Generate LHS samples for each parameter
        for (size_t param_idx = 0; param_idx < parameters_.size(); ++param_idx) {
            std::vector<T> intervals(num_samples);
            for (size_t i = 0; i < num_samples; ++i) {
                intervals[i] = (T(i) + uniform_dist_(rng_)) / T(num_samples);
            }

            // Shuffle intervals
            std::shuffle(intervals.begin(), intervals.end(), rng_);

            // Map to parameter range
            const auto& param = parameters_[param_idx];
            for (size_t i = 0; i < num_samples; ++i) {
                T normalized_value = intervals[i];
                samples[i][param_idx] = mapNormalizedToParameter(param, normalized_value);
            }
        }

        return samples;
    }

    // Sobol sequence for quasi-random sampling
    std::vector<std::vector<T>> sobolSampling(size_t num_samples) const {
        // Simplified Sobol sequence implementation
        std::vector<std::vector<T>> samples;
        samples.reserve(num_samples);

        for (size_t i = 0; i < num_samples; ++i) {
            std::vector<T> sample(parameters_.size());
            for (size_t j = 0; j < parameters_.size(); ++j) {
                // Van der Corput sequence in base 2
                T value = vanDerCorputSequence(i, 2 + j);
                sample[j] = mapNormalizedToParameter(parameters_[j], value);
            }
            samples.push_back(sample);
        }

        return samples;
    }

    // Constraint validation
    bool satisfiesConstraints(const std::vector<T>& values) const {
        // Check individual parameter constraints
        for (size_t i = 0; i < parameters_.size(); ++i) {
            const auto& param = parameters_[i];
            if (param.constraint_function && !param.constraint_function(values[i])) {
                return false;
            }

            // Basic bounds checking
            if (values[i] < param.min_value || values[i] > param.max_value) {
                return false;
            }
        }

        // Check global constraints
        for (const auto& constraint : global_constraints_) {
            if (!constraint(values)) {
                return false;
            }
        }

        return true;
    }

    // Parameter space metrics
    T getSpaceVolume() const {
        T volume = T(1);
        for (const auto& param : parameters_) {
            if (param.type == ParameterType::CONTINUOUS) {
                volume *= (param.max_value - param.min_value);
            } else if (param.type == ParameterType::DISCRETE) {
                volume *= T(param.discrete_values.size());
            }
        }
        return volume;
    }

    std::vector<T> getParameterRanges() const {
        std::vector<T> ranges;
        for (const auto& param : parameters_) {
            ranges.push_back(param.max_value - param.min_value);
        }
        return ranges;
    }

private:
    T sampleParameter(size_t index) const {
        const auto& param = parameters_[index];

        switch (param.distribution) {
            case DistributionType::UNIFORM:
                return param.min_value + uniform_dist_(rng_) * (param.max_value - param.min_value);

            case DistributionType::NORMAL: {
                T value = normal_dist_(rng_) * param.distribution_param2 + param.distribution_param1;
                return std::clamp(value, param.min_value, param.max_value);
            }

            case DistributionType::LOG_UNIFORM: {
                T log_min = std::log(param.min_value);
                T log_max = std::log(param.max_value);
                return std::exp(log_min + uniform_dist_(rng_) * (log_max - log_min));
            }

            case DistributionType::LOG_NORMAL: {
                T log_value = normal_dist_(rng_) * param.distribution_param2 + param.distribution_param1;
                T value = std::exp(log_value);
                return std::clamp(value, param.min_value, param.max_value);
            }

            case DistributionType::BETA: {
                // Simplified beta distribution using rejection sampling
                T alpha = param.distribution_param1;
                T beta = param.distribution_param2;
                T u1 = uniform_dist_(rng_);
                T u2 = uniform_dist_(rng_);

                if (alpha > 1 && beta > 1) {
                    // Use simplified approximation
                    T value = u1 / (u1 + u2);
                    return param.min_value + value * (param.max_value - param.min_value);
                } else {
                    return param.min_value + u1 * (param.max_value - param.min_value);
                }
            }

            default:
                return param.default_value;
        }
    }

    T mapNormalizedToParameter(const Parameter<T>& param, T normalized_value) const {
        switch (param.type) {
            case ParameterType::CONTINUOUS:
                return param.min_value + normalized_value * (param.max_value - param.min_value);

            case ParameterType::DISCRETE:
                if (!param.discrete_values.empty()) {
                    size_t idx = static_cast<size_t>(normalized_value * param.discrete_values.size());
                    idx = std::min(idx, param.discrete_values.size() - 1);
                    return param.discrete_values[idx];
                } else {
                    T continuous_value = param.min_value + normalized_value * (param.max_value - param.min_value);
                    return static_cast<T>(static_cast<int>(continuous_value));
                }

            case ParameterType::BOOLEAN:
                return normalized_value > 0.5 ? T(1) : T(0);

            default:
                return param.default_value;
        }
    }

    T vanDerCorputSequence(size_t n, size_t base) const {
        T result = 0;
        T f = T(1) / T(base);
        size_t i = n;

        while (i > 0) {
            result += f * T(i % base);
            i /= base;
            f /= T(base);
        }

        return result;
    }
};

// =============================================================================
// EXPERIMENT OBJECTIVES AND METRICS
// =============================================================================

// Experiment objective types
enum class ObjectiveType {
    MINIMIZE,       // Minimize the objective
    MAXIMIZE,       // Maximize the objective
    TARGET,         // Target a specific value
    CONSTRAINT      // Treat as constraint
};

// Objective function definition
template<typename T>
struct Objective {
    std::string name;
    ObjectiveType type;
    T target_value;     // For TARGET type
    T weight;           // Relative importance
    T tolerance;        // Acceptable deviation for TARGET

    std::function<T(const std::vector<T>&)> evaluation_function;

    Objective() : type(ObjectiveType::MINIMIZE), target_value(0),
                  weight(1), tolerance(0.01) {}
};

// Multi-objective optimization metrics
template<typename T>
struct ObjectiveResult {
    std::vector<T> objective_values;
    std::vector<T> parameter_values;
    T fitness_score;
    T constraint_violation;
    bool feasible;

    std::chrono::steady_clock::time_point evaluation_time;
    size_t evaluation_id;

    ObjectiveResult() : fitness_score(0), constraint_violation(0),
                        feasible(true), evaluation_id(0) {}
};

// Pareto frontier for multi-objective optimization
template<typename T>
class ParetoFrontier {
private:
    std::vector<ObjectiveResult<T>> pareto_points_;
    std::vector<ObjectiveType> objective_types_;

public:
    void setObjectiveTypes(const std::vector<ObjectiveType>& types) {
        objective_types_ = types;
    }

    void addPoint(const ObjectiveResult<T>& result) {
        if (!result.feasible) return;

        // Check if point is dominated
        if (isDominated(result)) return;

        // Remove dominated points
        auto it = std::remove_if(pareto_points_.begin(), pareto_points_.end(),
            [this, &result](const ObjectiveResult<T>& existing) {
                return dominates(result, existing);
            });
        pareto_points_.erase(it, pareto_points_.end());

        // Add new point
        pareto_points_.push_back(result);
    }

    const std::vector<ObjectiveResult<T>>& getPoints() const {
        return pareto_points_;
    }

    ObjectiveResult<T> getBestPoint(const std::vector<T>& weights) const {
        if (pareto_points_.empty()) {
            return ObjectiveResult<T>{};
        }

        T best_score = std::numeric_limits<T>::lowest();
        size_t best_idx = 0;

        for (size_t i = 0; i < pareto_points_.size(); ++i) {
            T weighted_score = 0;
            for (size_t j = 0; j < pareto_points_[i].objective_values.size(); ++j) {
                T obj_contribution = pareto_points_[i].objective_values[j];

                // Normalize based on objective type
                if (objective_types_[j] == ObjectiveType::MINIMIZE) {
                    obj_contribution = -obj_contribution;
                }

                weighted_score += weights[j] * obj_contribution;
            }

            if (weighted_score > best_score) {
                best_score = weighted_score;
                best_idx = i;
            }
        }

        return pareto_points_[best_idx];
    }

    size_t size() const { return pareto_points_.size(); }
    bool empty() const { return pareto_points_.empty(); }

private:
    bool dominates(const ObjectiveResult<T>& a, const ObjectiveResult<T>& b) const {
        bool at_least_one_better = false;

        for (size_t i = 0; i < a.objective_values.size(); ++i) {
            T val_a = a.objective_values[i];
            T val_b = b.objective_values[i];

            // Adjust for objective type
            if (objective_types_[i] == ObjectiveType::MINIMIZE) {
                if (val_a > val_b) return false;  // a is worse in this objective
                if (val_a < val_b) at_least_one_better = true;
            } else if (objective_types_[i] == ObjectiveType::MAXIMIZE) {
                if (val_a < val_b) return false;  // a is worse in this objective
                if (val_a > val_b) at_least_one_better = true;
            }
        }

        return at_least_one_better;
    }

    bool isDominated(const ObjectiveResult<T>& candidate) const {
        for (const auto& existing : pareto_points_) {
            if (dominates(existing, candidate)) {
                return true;
            }
        }
        return false;
    }
};

// =============================================================================
// EXPERIMENT EXECUTION ENGINE
// =============================================================================

// Experiment execution configuration
struct ExperimentConfig {
    size_t max_evaluations = 1000;
    size_t parallel_evaluations = 4;
    double timeout_seconds = 3600.0;  // 1 hour
    bool enable_early_stopping = true;
    double convergence_tolerance = 1e-6;
    size_t convergence_patience = 50;

    // Optimization algorithm settings
    bool use_bayesian_optimization = true;
    bool use_multi_objective = false;
    bool enable_constraint_handling = true;
    bool save_intermediate_results = true;

    std::string output_directory = "./experiment_results";
    std::string experiment_name = "physics_optimization";
};

// Experiment execution status
enum class ExperimentStatus {
    NOT_STARTED,
    INITIALIZING,
    RUNNING,
    CONVERGED,
    TIMEOUT,
    ERROR,
    COMPLETED,
    CANCELLED
};

// =============================================================================
// BAYESIAN OPTIMIZATION
// =============================================================================

// Gaussian Process for Bayesian optimization
template<typename T>
class GaussianProcess {
private:
    std::vector<std::vector<T>> training_inputs_;
    std::vector<T> training_outputs_;

    // Hyperparameters
    T length_scale_;
    T signal_variance_;
    T noise_variance_;

    // Covariance matrix and its inverse
    std::vector<std::vector<T>> K_inv_;
    bool is_fitted_;

public:
    GaussianProcess() : length_scale_(1.0), signal_variance_(1.0),
                        noise_variance_(0.01), is_fitted_(false) {}

    void setHyperparameters(T length_scale, T signal_variance, T noise_variance) {
        length_scale_ = length_scale;
        signal_variance_ = signal_variance;
        noise_variance_ = noise_variance;
        is_fitted_ = false;
    }

    void fit(const std::vector<std::vector<T>>& inputs, const std::vector<T>& outputs) {
        training_inputs_ = inputs;
        training_outputs_ = outputs;

        size_t n = inputs.size();
        if (n == 0) return;

        // Build covariance matrix
        std::vector<std::vector<T>> K(n, std::vector<T>(n));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                K[i][j] = rbfKernel(inputs[i], inputs[j]);
                if (i == j) {
                    K[i][j] += noise_variance_;
                }
            }
        }

        // Compute inverse (simplified - in practice use Cholesky decomposition)
        K_inv_ = matrixInverse(K);
        is_fitted_ = true;
    }

    std::pair<T, T> predict(const std::vector<T>& test_input) const {
        if (!is_fitted_ || training_inputs_.empty()) {
            return {0, signal_variance_};
        }

        size_t n = training_inputs_.size();

        // Compute covariance between test point and training points
        std::vector<T> k_star(n);
        for (size_t i = 0; i < n; ++i) {
            k_star[i] = rbfKernel(test_input, training_inputs_[i]);
        }

        // Compute mean prediction
        T mean = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                mean += k_star[i] * K_inv_[i][j] * training_outputs_[j];
            }
        }

        // Compute variance prediction
        T k_star_star = rbfKernel(test_input, test_input);
        T variance = k_star_star;

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                variance -= k_star[i] * K_inv_[i][j] * k_star[j];
            }
        }

        variance = std::max(variance, T(1e-8));  // Ensure positive variance

        return {mean, variance};
    }

private:
    T rbfKernel(const std::vector<T>& x1, const std::vector<T>& x2) const {
        T distance_squared = 0;
        for (size_t i = 0; i < x1.size(); ++i) {
            T diff = x1[i] - x2[i];
            distance_squared += diff * diff;
        }

        return signal_variance_ * std::exp(-distance_squared / (2 * length_scale_ * length_scale_));
    }

    std::vector<std::vector<T>> matrixInverse(const std::vector<std::vector<T>>& matrix) const {
        size_t n = matrix.size();
        std::vector<std::vector<T>> result(n, std::vector<T>(n, 0));
        std::vector<std::vector<T>> augmented = matrix;

        // Create identity matrix on the right side
        for (size_t i = 0; i < n; ++i) {
            augmented[i].resize(2 * n);
            for (size_t j = n; j < 2 * n; ++j) {
                augmented[i][j] = (i == j - n) ? 1 : 0;
            }
        }

        // Gaussian elimination
        for (size_t i = 0; i < n; ++i) {
            // Find pivot
            size_t pivot = i;
            for (size_t k = i + 1; k < n; ++k) {
                if (std::abs(augmented[k][i]) > std::abs(augmented[pivot][i])) {
                    pivot = k;
                }
            }

            // Swap rows
            if (pivot != i) {
                std::swap(augmented[i], augmented[pivot]);
            }

            // Make diagonal element 1
            T diag = augmented[i][i];
            if (std::abs(diag) < 1e-12) {
                diag = 1e-12;  // Regularization for near-singular matrices
            }

            for (size_t j = 0; j < 2 * n; ++j) {
                augmented[i][j] /= diag;
            }

            // Eliminate column
            for (size_t k = 0; k < n; ++k) {
                if (k != i) {
                    T factor = augmented[k][i];
                    for (size_t j = 0; j < 2 * n; ++j) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }

        // Extract result
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result[i][j] = augmented[i][j + n];
            }
        }

        return result;
    }
};

// Acquisition functions for Bayesian optimization
template<typename T>
class AcquisitionFunction {
public:
    virtual ~AcquisitionFunction() = default;
    virtual T evaluate(T mean, T variance, T best_value) const = 0;
    virtual std::string getName() const = 0;
};

template<typename T>
class ExpectedImprovement : public AcquisitionFunction<T> {
private:
    T xi_;  // Exploration parameter

public:
    explicit ExpectedImprovement(T xi = 0.01) : xi_(xi) {}

    T evaluate(T mean, T variance, T best_value) const override {
        if (variance <= 0) return 0;

        T std_dev = std::sqrt(variance);
        T z = (mean - best_value - xi_) / std_dev;

        // Standard normal CDF and PDF approximations
        T cdf = 0.5 * (1 + std::erf(z / std::sqrt(2)));
        T pdf = std::exp(-0.5 * z * z) / std::sqrt(2 * M_PI);

        return (mean - best_value - xi_) * cdf + std_dev * pdf;
    }

    std::string getName() const override { return "Expected Improvement"; }
};

template<typename T>
class UpperConfidenceBound : public AcquisitionFunction<T> {
private:
    T kappa_;  // Exploration parameter

public:
    explicit UpperConfidenceBound(T kappa = 2.576) : kappa_(kappa) {}  // 99% confidence

    T evaluate(T mean, T variance, T best_value) const override {
        return mean + kappa_ * std::sqrt(variance);
    }

    std::string getName() const override { return "Upper Confidence Bound"; }
};

} // namespace experiment_design
} // namespace physgrad