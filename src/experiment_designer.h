/**
 * PhysGrad Experiment Designer - Main Execution Engine
 *
 * Coordinates automatic experiment design, execution, and optimization.
 * Integrates Bayesian optimization, multi-objective search, and parallel execution.
 */

#pragma once

#include "automatic_experiment_design.h"
#include <iostream>
#include <thread>
#include <future>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace physgrad {
namespace experiment_design {

// =============================================================================
// MAIN EXPERIMENT DESIGNER CLASS
// =============================================================================

template<typename T>
class ExperimentDesigner {
private:
    ParameterSpace<T> parameter_space_;
    std::vector<Objective<T>> objectives_;
    ExperimentConfig config_;

    // Optimization components
    std::unique_ptr<GaussianProcess<T>> gaussian_process_;
    std::unique_ptr<AcquisitionFunction<T>> acquisition_function_;
    ParetoFrontier<T> pareto_frontier_;

    // Execution state
    ExperimentStatus status_;
    std::atomic<size_t> evaluations_completed_;
    std::atomic<bool> should_stop_;
    std::chrono::steady_clock::time_point start_time_;

    // Results storage
    std::vector<ObjectiveResult<T>> all_results_;
    ObjectiveResult<T> best_result_;
    std::mutex results_mutex_;

    // Parallel execution
    std::vector<std::thread> worker_threads_;
    std::queue<std::vector<T>> parameter_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Convergence tracking
    std::vector<T> convergence_history_;
    size_t stagnation_counter_;

public:
    ExperimentDesigner() : status_(ExperimentStatus::NOT_STARTED),
                           evaluations_completed_(0), should_stop_(false),
                           stagnation_counter_(0) {
        gaussian_process_ = std::make_unique<GaussianProcess<T>>();
        acquisition_function_ = std::make_unique<ExpectedImprovement<T>>();
    }

    ~ExperimentDesigner() {
        stop();
    }

    // Configuration
    void setParameterSpace(const ParameterSpace<T>& space) {
        parameter_space_ = space;
    }

    void addObjective(const Objective<T>& objective) {
        objectives_.push_back(objective);
    }

    void setConfig(const ExperimentConfig& config) {
        config_ = config;
    }

    void setAcquisitionFunction(std::unique_ptr<AcquisitionFunction<T>> func) {
        acquisition_function_ = std::move(func);
    }

    // Experiment execution
    void run() {
        if (status_ != ExperimentStatus::NOT_STARTED) {
            throw std::runtime_error("Experiment already started or completed");
        }

        start_time_ = std::chrono::steady_clock::now();
        status_ = ExperimentStatus::INITIALIZING;

        std::cout << "Starting PhysGrad Automatic Experiment Design" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "Parameter space: " << parameter_space_.getParameterCount() << " dimensions" << std::endl;
        std::cout << "Objectives: " << objectives_.size() << std::endl;
        std::cout << "Max evaluations: " << config_.max_evaluations << std::endl;
        std::cout << "Parallel workers: " << config_.parallel_evaluations << std::endl;

        try {
            initializeOptimization();
            executeOptimization();
            finalizeResults();
        } catch (const std::exception& e) {
            status_ = ExperimentStatus::ERROR;
            std::cerr << "Experiment failed: " << e.what() << std::endl;
            throw;
        }
    }

    void stop() {
        should_stop_ = true;

        // Wake up all worker threads
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_cv_.notify_all();
        }

        // Wait for workers to finish
        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        worker_threads_.clear();

        if (status_ == ExperimentStatus::RUNNING) {
            status_ = ExperimentStatus::CANCELLED;
        }
    }

    // Results access
    const ObjectiveResult<T>& getBestResult() const { return best_result_; }
    const std::vector<ObjectiveResult<T>>& getAllResults() const { return all_results_; }
    const ParetoFrontier<T>& getParetoFrontier() const { return pareto_frontier_; }
    ExperimentStatus getStatus() const { return status_; }
    size_t getEvaluationsCompleted() const { return evaluations_completed_; }

    // Progress monitoring
    double getProgress() const {
        return double(evaluations_completed_) / double(config_.max_evaluations);
    }

    double getElapsedTime() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time_).count();
    }

    std::string getStatusString() const {
        switch (status_) {
            case ExperimentStatus::NOT_STARTED: return "Not Started";
            case ExperimentStatus::INITIALIZING: return "Initializing";
            case ExperimentStatus::RUNNING: return "Running";
            case ExperimentStatus::CONVERGED: return "Converged";
            case ExperimentStatus::TIMEOUT: return "Timeout";
            case ExperimentStatus::ERROR: return "Error";
            case ExperimentStatus::COMPLETED: return "Completed";
            case ExperimentStatus::CANCELLED: return "Cancelled";
            default: return "Unknown";
        }
    }

    // Real-time optimization suggestions
    std::vector<T> suggestNextParameters() {
        if (all_results_.empty()) {
            return parameter_space_.sampleParameters();
        }

        if (config_.use_bayesian_optimization && all_results_.size() >= 5) {
            return suggestBayesianOptimization();
        } else {
            return suggestExplorationSampling();
        }
    }

    // Export results
    void exportResults(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }

        // Write header
        file << "evaluation_id,";
        for (size_t i = 0; i < parameter_space_.getParameterCount(); ++i) {
            file << "param_" << i << ",";
        }
        for (size_t i = 0; i < objectives_.size(); ++i) {
            file << "objective_" << i << ",";
        }
        file << "fitness_score,feasible,evaluation_time_ms" << std::endl;

        // Write data
        for (const auto& result : all_results_) {
            file << result.evaluation_id << ",";
            for (T param : result.parameter_values) {
                file << param << ",";
            }
            for (T obj : result.objective_values) {
                file << obj << ",";
            }
            file << result.fitness_score << ","
                 << (result.feasible ? "true" : "false") << ","
                 << std::chrono::duration_cast<std::chrono::milliseconds>(
                        result.evaluation_time - start_time_).count() << std::endl;
        }

        std::cout << "Results exported to: " << filename << std::endl;
    }

    // Parameter importance analysis
    std::vector<T> analyzeParameterImportance() const {
        if (all_results_.size() < 10) {
            return std::vector<T>(parameter_space_.getParameterCount(), T(1));
        }

        std::vector<T> importance(parameter_space_.getParameterCount(), T(0));

        // Simple variance-based sensitivity analysis
        for (size_t param_idx = 0; param_idx < parameter_space_.getParameterCount(); ++param_idx) {
            std::vector<T> param_values;
            std::vector<T> objective_values;

            for (const auto& result : all_results_) {
                if (result.feasible && !result.objective_values.empty()) {
                    param_values.push_back(result.parameter_values[param_idx]);
                    objective_values.push_back(result.objective_values[0]);  // Use first objective
                }
            }

            if (param_values.size() >= 5) {
                importance[param_idx] = computeCorrelation(param_values, objective_values);
            }
        }

        // Normalize importance scores
        T max_importance = *std::max_element(importance.begin(), importance.end());
        if (max_importance > 0) {
            for (auto& imp : importance) {
                imp /= max_importance;
            }
        }

        return importance;
    }

private:
    void initializeOptimization() {
        status_ = ExperimentStatus::INITIALIZING;

        // Create output directory (simple approach)
        // Note: In production, use std::filesystem when available
        std::string cmd = "mkdir -p " + config_.output_directory;
        system(cmd.c_str());

        // Setup Pareto frontier for multi-objective optimization
        if (config_.use_multi_objective && objectives_.size() > 1) {
            std::vector<ObjectiveType> types;
            for (const auto& obj : objectives_) {
                types.push_back(obj.type);
            }
            pareto_frontier_.setObjectiveTypes(types);
        }

        // Generate initial sample points using Latin Hypercube Sampling
        size_t initial_samples = std::min(size_t(20), config_.max_evaluations / 4);
        auto initial_points = parameter_space_.latinHypercubeSampling(initial_samples);

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            for (const auto& point : initial_points) {
                parameter_queue_.push(point);
            }
        }

        std::cout << "Generated " << initial_samples << " initial sample points" << std::endl;
    }

    void executeOptimization() {
        status_ = ExperimentStatus::RUNNING;

        // Start worker threads
        for (size_t i = 0; i < config_.parallel_evaluations; ++i) {
            worker_threads_.emplace_back(&ExperimentDesigner::workerThread, this);
        }

        // Main optimization loop
        while (evaluations_completed_ < config_.max_evaluations && !should_stop_) {
            // Check timeout
            if (getElapsedTime() > config_.timeout_seconds) {
                status_ = ExperimentStatus::TIMEOUT;
                break;
            }

            // Check convergence
            if (config_.enable_early_stopping && checkConvergence()) {
                status_ = ExperimentStatus::CONVERGED;
                break;
            }

            // Generate new parameter suggestions
            if (parameter_queue_.size() < config_.parallel_evaluations * 2) {
                generateNewParameters();
            }

            // Brief sleep to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Progress reporting
            if (evaluations_completed_ % 50 == 0 && evaluations_completed_ > 0) {
                reportProgress();
            }
        }

        // Signal workers to stop and wait for completion
        should_stop_ = true;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_cv_.notify_all();
        }

        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        if (status_ == ExperimentStatus::RUNNING) {
            status_ = ExperimentStatus::COMPLETED;
        }
    }

    void workerThread() {
        while (!should_stop_) {
            std::vector<T> parameters;

            // Get parameters from queue
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return !parameter_queue_.empty() || should_stop_; });

                if (should_stop_) break;

                if (!parameter_queue_.empty()) {
                    parameters = parameter_queue_.front();
                    parameter_queue_.pop();
                } else {
                    continue;
                }
            }

            // Evaluate parameters
            try {
                auto result = evaluateParameters(parameters);
                processResult(result);
            } catch (const std::exception& e) {
                std::cerr << "Evaluation failed: " << e.what() << std::endl;
            }
        }
    }

    ObjectiveResult<T> evaluateParameters(const std::vector<T>& parameters) {
        ObjectiveResult<T> result;
        result.parameter_values = parameters;
        result.evaluation_id = evaluations_completed_.fetch_add(1);
        result.evaluation_time = std::chrono::steady_clock::now();

        // Check parameter constraints
        result.feasible = parameter_space_.satisfiesConstraints(parameters);
        if (!result.feasible) {
            result.constraint_violation = 1.0;  // Simple violation indicator
            return result;
        }

        // Evaluate all objectives
        result.objective_values.reserve(objectives_.size());
        for (const auto& objective : objectives_) {
            try {
                T value = objective.evaluation_function(parameters);
                result.objective_values.push_back(value);
            } catch (const std::exception& e) {
                result.feasible = false;
                result.constraint_violation = 1.0;
                std::cerr << "Objective evaluation failed: " << e.what() << std::endl;
                return result;
            }
        }

        // Compute fitness score (for single-objective or weighted multi-objective)
        result.fitness_score = computeFitnessScore(result.objective_values);

        return result;
    }

    void processResult(const ObjectiveResult<T>& result) {
        std::lock_guard<std::mutex> lock(results_mutex_);

        all_results_.push_back(result);

        if (result.feasible) {
            // Update best result
            if (all_results_.size() == 1 ||
                (objectives_[0].type == ObjectiveType::MINIMIZE && result.fitness_score < best_result_.fitness_score) ||
                (objectives_[0].type == ObjectiveType::MAXIMIZE && result.fitness_score > best_result_.fitness_score)) {
                best_result_ = result;
            }

            // Update Pareto frontier for multi-objective
            if (config_.use_multi_objective) {
                pareto_frontier_.addPoint(result);
            }

            // Update convergence tracking
            convergence_history_.push_back(result.fitness_score);
        }

        // Update Gaussian Process for Bayesian optimization
        if (config_.use_bayesian_optimization && all_results_.size() >= 5) {
            updateGaussianProcess();
        }
    }

    void generateNewParameters() {
        size_t batch_size = std::min(size_t(10), config_.parallel_evaluations);

        std::lock_guard<std::mutex> lock(queue_mutex_);

        for (size_t i = 0; i < batch_size; ++i) {
            auto params = suggestNextParameters();
            parameter_queue_.push(params);
        }

        queue_cv_.notify_all();
    }

    std::vector<T> suggestBayesianOptimization() {
        // Grid search over parameter space to maximize acquisition function
        size_t grid_size = 1000;
        auto candidates = parameter_space_.sobolSampling(grid_size);

        T best_acquisition = std::numeric_limits<T>::lowest();
        std::vector<T> best_params;

        T current_best = best_result_.fitness_score;
        if (objectives_[0].type == ObjectiveType::MINIMIZE) {
            current_best = -current_best;  // Convert to maximization problem
        }

        for (const auto& candidate : candidates) {
            auto [mean, variance] = gaussian_process_->predict(candidate);

            if (objectives_[0].type == ObjectiveType::MINIMIZE) {
                mean = -mean;  // Convert to maximization problem
            }

            T acquisition_value = acquisition_function_->evaluate(mean, variance, current_best);

            if (acquisition_value > best_acquisition) {
                best_acquisition = acquisition_value;
                best_params = candidate;
            }
        }

        return best_params.empty() ? parameter_space_.sampleParameters() : best_params;
    }

    std::vector<T> suggestExplorationSampling() {
        // Use different sampling strategies based on progress
        double progress = getProgress();

        if (progress < 0.3) {
            // Early exploration: Latin Hypercube Sampling
            auto samples = parameter_space_.latinHypercubeSampling(1);
            return samples.empty() ? parameter_space_.sampleParameters() : samples[0];
        } else if (progress < 0.7) {
            // Mid exploration: Sobol sequence
            auto samples = parameter_space_.sobolSampling(1);
            return samples.empty() ? parameter_space_.sampleParameters() : samples[0];
        } else {
            // Late exploration: Random sampling around best parameters
            return perturbBestParameters();
        }
    }

    std::vector<T> perturbBestParameters() {
        if (best_result_.parameter_values.empty()) {
            return parameter_space_.sampleParameters();
        }

        std::vector<T> perturbed = best_result_.parameter_values;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> noise_dist(0, 0.1);

        auto ranges = parameter_space_.getParameterRanges();

        for (size_t i = 0; i < perturbed.size(); ++i) {
            T noise = noise_dist(gen) * ranges[i];
            perturbed[i] += noise;

            // Clamp to parameter bounds
            const auto& param = parameter_space_.getParameter(i);
            perturbed[i] = std::clamp(perturbed[i], param.min_value, param.max_value);
        }

        return perturbed;
    }

    void updateGaussianProcess() {
        std::vector<std::vector<T>> inputs;
        std::vector<T> outputs;

        for (const auto& result : all_results_) {
            if (result.feasible && !result.objective_values.empty()) {
                inputs.push_back(result.parameter_values);

                T output = result.objective_values[0];  // Use first objective
                if (objectives_[0].type == ObjectiveType::MINIMIZE) {
                    output = -output;  // Convert to maximization for GP
                }
                outputs.push_back(output);
            }
        }

        if (inputs.size() >= 3) {
            gaussian_process_->fit(inputs, outputs);
        }
    }

    T computeFitnessScore(const std::vector<T>& objective_values) const {
        if (objective_values.empty()) return 0;

        T score = 0;
        T total_weight = 0;

        for (size_t i = 0; i < objective_values.size() && i < objectives_.size(); ++i) {
            T value = objective_values[i];
            T weight = objectives_[i].weight;

            // Normalize based on objective type
            switch (objectives_[i].type) {
                case ObjectiveType::MINIMIZE:
                    value = -value;  // Convert to maximization
                    break;
                case ObjectiveType::MAXIMIZE:
                    // Already in maximization form
                    break;
                case ObjectiveType::TARGET:
                    value = -std::abs(value - objectives_[i].target_value);
                    break;
                case ObjectiveType::CONSTRAINT:
                    // Handle as constraint (not in fitness)
                    continue;
            }

            score += weight * value;
            total_weight += weight;
        }

        return total_weight > 0 ? score / total_weight : score;
    }

    bool checkConvergence() const {
        if (convergence_history_.size() < config_.convergence_patience) {
            return false;
        }

        // Check if improvement has stagnated
        size_t recent_size = config_.convergence_patience;
        auto recent_start = convergence_history_.end() - recent_size;

        T recent_best = *std::max_element(recent_start, convergence_history_.end());
        T early_best = *std::max_element(convergence_history_.begin(),
                                        convergence_history_.begin() + recent_size);

        T improvement = std::abs(recent_best - early_best);
        return improvement < config_.convergence_tolerance;
    }

    void reportProgress() const {
        double progress = getProgress();
        double elapsed = getElapsedTime();
        double eta = elapsed / progress - elapsed;

        std::cout << "Progress: " << std::fixed << std::setprecision(1)
                  << progress * 100 << "% | "
                  << "Evaluations: " << evaluations_completed_ << "/" << config_.max_evaluations
                  << " | Elapsed: " << std::setprecision(0) << elapsed << "s"
                  << " | ETA: " << eta << "s";

        if (!best_result_.objective_values.empty()) {
            std::cout << " | Best: " << std::setprecision(6) << best_result_.fitness_score;
        }

        std::cout << std::endl;
    }

    void finalizeResults() {
        std::cout << "\nOptimization completed!" << std::endl;
        std::cout << "Status: " << getStatusString() << std::endl;
        std::cout << "Total evaluations: " << evaluations_completed_ << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(2)
                  << getElapsedTime() << " seconds" << std::endl;

        if (!best_result_.objective_values.empty()) {
            std::cout << "\nBest result:" << std::endl;
            std::cout << "  Fitness score: " << best_result_.fitness_score << std::endl;
            std::cout << "  Objective values: ";
            for (size_t i = 0; i < best_result_.objective_values.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << best_result_.objective_values[i];
            }
            std::cout << std::endl;

            std::cout << "  Parameters:" << std::endl;
            for (size_t i = 0; i < best_result_.parameter_values.size(); ++i) {
                std::cout << "    " << parameter_space_.getParameter(i).name
                          << " = " << best_result_.parameter_values[i] << std::endl;
            }
        }

        if (config_.use_multi_objective && !pareto_frontier_.empty()) {
            std::cout << "\nPareto frontier contains " << pareto_frontier_.size()
                      << " non-dominated solutions" << std::endl;
        }

        // Export results
        if (config_.save_intermediate_results) {
            std::string filename = config_.output_directory + "/" + config_.experiment_name + "_results.csv";
            exportResults(filename);
        }
    }

    T computeCorrelation(const std::vector<T>& x, const std::vector<T>& y) const {
        if (x.size() != y.size() || x.size() < 2) return 0;

        T mean_x = std::accumulate(x.begin(), x.end(), T(0)) / x.size();
        T mean_y = std::accumulate(y.begin(), y.end(), T(0)) / y.size();

        T numerator = 0;
        T sum_sq_x = 0;
        T sum_sq_y = 0;

        for (size_t i = 0; i < x.size(); ++i) {
            T diff_x = x[i] - mean_x;
            T diff_y = y[i] - mean_y;
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
        }

        T denominator = std::sqrt(sum_sq_x * sum_sq_y);
        return denominator > 0 ? std::abs(numerator / denominator) : 0;
    }
};

} // namespace experiment_design
} // namespace physgrad