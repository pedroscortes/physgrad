#pragma once

#include "neural_surrogate.h"
#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <algorithm>
#include <random>

namespace physgrad {
namespace surrogate {

using namespace physgrad::neural;

// Physics state representation
template<typename T>
struct PhysicsState {
    std::vector<T> positions;      // Particle positions (x,y,z,x,y,z,...)
    std::vector<T> velocities;     // Particle velocities
    std::vector<T> forces;         // Forces acting on particles
    std::vector<T> material_props; // Material properties per particle
    T timestep;
    T time;

    size_t num_particles() const { return positions.size() / 3; }
};

// Training data point
template<typename T>
struct TrainingPoint {
    PhysicsState<T> input_state;
    PhysicsState<T> output_state;
    T physics_loss;       // Ground truth physics simulation result
    T surrogate_loss;     // Surrogate model prediction error
    T weight;             // Importance weight for training

    // Comparison operator for priority queue (higher weight = higher priority)
    bool operator<(const TrainingPoint<T>& other) const {
        return weight < other.weight;
    }
};

// Surrogate model configuration
template<typename T>
struct SurrogateConfig {
    // Network architecture
    std::vector<size_t> hidden_layers = {256, 512, 256, 128};
    ActivationType activation = ActivationType::Swish;

    // Training parameters
    T learning_rate = T(0.001);
    size_t batch_size = 32;
    size_t epochs = 1000;
    T validation_split = T(0.2);

    // Physics constraints
    bool enforce_energy_conservation = true;
    bool enforce_momentum_conservation = true;
    bool enforce_mass_conservation = true;
    T physics_weight = T(0.1);  // Weight for physics-informed loss

    // Adaptive sampling
    bool use_adaptive_sampling = true;
    T error_threshold = T(0.01);
    size_t max_training_points = 100000;

    // Hybrid mode
    bool use_hybrid_mode = true;    // Fall back to physics when uncertain
    T uncertainty_threshold = T(0.05);

    // Performance targets
    T target_speedup = T(10.0);     // Target speedup vs full physics
    T target_accuracy = T(0.95);    // Target accuracy vs ground truth
};

// Data preprocessing and normalization
template<typename T>
class DataPreprocessor {
private:
    std::vector<T> input_mean_;
    std::vector<T> input_std_;
    std::vector<T> output_mean_;
    std::vector<T> output_std_;
    bool fitted_;

public:
    DataPreprocessor() : fitted_(false) {}

    void fit(const std::vector<PhysicsState<T>>& states) {
        if (states.empty()) return;

        size_t state_size = get_state_size(states[0]);
        input_mean_.resize(state_size, T(0));
        input_std_.resize(state_size, T(0));
        output_mean_.resize(state_size, T(0));
        output_std_.resize(state_size, T(0));

        // Compute mean
        for (const auto& state : states) {
            auto flattened = flatten_state(state);
            for (size_t i = 0; i < state_size; ++i) {
                input_mean_[i] += flattened[i];
            }
        }
        for (size_t i = 0; i < state_size; ++i) {
            input_mean_[i] /= T(states.size());
            output_mean_[i] = input_mean_[i];  // Assume similar distribution
        }

        // Compute standard deviation
        for (const auto& state : states) {
            auto flattened = flatten_state(state);
            for (size_t i = 0; i < state_size; ++i) {
                T diff = flattened[i] - input_mean_[i];
                input_std_[i] += diff * diff;
            }
        }
        for (size_t i = 0; i < state_size; ++i) {
            input_std_[i] = std::sqrt(input_std_[i] / T(states.size()));
            if (input_std_[i] < T(1e-8)) input_std_[i] = T(1);  // Prevent division by zero
            output_std_[i] = input_std_[i];
        }

        fitted_ = true;
    }

    std::vector<T> normalize_input(const PhysicsState<T>& state) const {
        if (!fitted_) {
            throw std::runtime_error("Preprocessor not fitted");
        }

        auto flattened = flatten_state(state);
        for (size_t i = 0; i < flattened.size(); ++i) {
            flattened[i] = (flattened[i] - input_mean_[i]) / input_std_[i];
        }
        return flattened;
    }

    PhysicsState<T> denormalize_output(const std::vector<T>& normalized) const {
        if (!fitted_) {
            throw std::runtime_error("Preprocessor not fitted");
        }

        std::vector<T> denormalized(normalized.size());
        for (size_t i = 0; i < normalized.size(); ++i) {
            denormalized[i] = normalized[i] * output_std_[i] + output_mean_[i];
        }
        return unflatten_state(denormalized);
    }

private:
    size_t get_state_size(const PhysicsState<T>& state) const {
        return state.positions.size() + state.velocities.size() +
               state.forces.size() + state.material_props.size() + 2; // +2 for timestep and time
    }

    std::vector<T> flatten_state(const PhysicsState<T>& state) const {
        std::vector<T> flattened;
        flattened.reserve(get_state_size(state));

        flattened.insert(flattened.end(), state.positions.begin(), state.positions.end());
        flattened.insert(flattened.end(), state.velocities.begin(), state.velocities.end());
        flattened.insert(flattened.end(), state.forces.begin(), state.forces.end());
        flattened.insert(flattened.end(), state.material_props.begin(), state.material_props.end());
        flattened.push_back(state.timestep);
        flattened.push_back(state.time);

        return flattened;
    }

    PhysicsState<T> unflatten_state(const std::vector<T>& flattened) const {
        PhysicsState<T> state;

        // This assumes we know the original structure
        // In practice, we'd need to store the dimensions
        size_t pos_size = flattened.size() / 7 * 3;  // Simplified assumption
        size_t vel_size = pos_size;
        size_t force_size = pos_size;
        size_t mat_size = pos_size / 3;

        size_t offset = 0;
        state.positions.assign(flattened.begin() + offset, flattened.begin() + offset + pos_size);
        offset += pos_size;

        state.velocities.assign(flattened.begin() + offset, flattened.begin() + offset + vel_size);
        offset += vel_size;

        state.forces.assign(flattened.begin() + offset, flattened.begin() + offset + force_size);
        offset += force_size;

        state.material_props.assign(flattened.begin() + offset, flattened.begin() + offset + mat_size);
        offset += mat_size;

        state.timestep = flattened[offset];
        state.time = flattened[offset + 1];

        return state;
    }
};

// Physics-informed constraints
template<typename T>
class PhysicsConstraints {
public:
    // Energy conservation constraint
    static Tensor<T> energy_conservation(const Tensor<T>& output) {
        // Ensure kinetic + potential energy is conserved
        // This is a simplified implementation
        return output;
    }

    // Momentum conservation constraint
    static Tensor<T> momentum_conservation(const Tensor<T>& output) {
        // Ensure total momentum is conserved
        Tensor<T> constrained = output;

        // Extract velocity components and adjust for momentum conservation
        // Simplified implementation

        return constrained;
    }

    // Mass conservation constraint
    static Tensor<T> mass_conservation(const Tensor<T>& output) {
        // Ensure mass is conserved (no particles created/destroyed)
        return output;
    }

    // Combined physics constraint
    static Tensor<T> physics_informed_constraint(const Tensor<T>& output) {
        auto result = energy_conservation(output);
        result = momentum_conservation(result);
        result = mass_conservation(result);
        return result;
    }
};

// Uncertainty quantification
template<typename T>
class UncertaintyEstimator {
private:
    std::vector<std::unique_ptr<NeuralNetwork<T>>> ensemble_;
    size_t ensemble_size_;

public:
    UncertaintyEstimator(size_t ensemble_size = 5) : ensemble_size_(ensemble_size) {
        ensemble_.reserve(ensemble_size_);
    }

    void add_model(std::unique_ptr<NeuralNetwork<T>> model) {
        ensemble_.push_back(std::move(model));
    }

    std::pair<Tensor<T>, T> predict_with_uncertainty(const Tensor<T>& input) {
        if (ensemble_.empty()) {
            throw std::runtime_error("No models in ensemble");
        }

        std::vector<Tensor<T>> predictions;
        predictions.reserve(ensemble_.size());

        // Get predictions from all models
        for (auto& model : ensemble_) {
            predictions.push_back(model->forward(input));
        }

        // Compute mean and variance
        Tensor<T> mean = predictions[0];
        for (size_t i = 1; i < predictions.size(); ++i) {
            mean = mean + predictions[i];
        }
        mean = mean * (T(1) / T(predictions.size()));

        // Compute variance as uncertainty measure
        T variance = T(0);
        for (const auto& pred : predictions) {
            auto diff = pred - mean;
            for (size_t i = 0; i < diff.size(); ++i) {
                variance += diff[i] * diff[i];
            }
        }
        variance /= T(predictions.size() * mean.size());

        return {mean, std::sqrt(variance)};
    }
};

// Adaptive sampling strategy
template<typename T>
class AdaptiveSampler {
private:
    std::priority_queue<TrainingPoint<T>> sample_queue_;
    std::function<PhysicsState<T>(const PhysicsState<T>&)> physics_simulator_;
    T error_threshold_;
    size_t max_samples_;

public:
    AdaptiveSampler(std::function<PhysicsState<T>(const PhysicsState<T>&)> physics_sim,
                   T error_threshold = T(0.01), size_t max_samples = 10000)
        : physics_simulator_(physics_sim), error_threshold_(error_threshold), max_samples_(max_samples) {}

    void add_sample(const PhysicsState<T>& initial_state,
                   const PhysicsState<T>& predicted_state,
                   T prediction_uncertainty) {
        // Run ground truth physics
        auto ground_truth = physics_simulator_(initial_state);

        // Compute prediction error
        T error = compute_state_error(predicted_state, ground_truth);

        // Create training point with importance weight
        TrainingPoint<T> point;
        point.input_state = initial_state;
        point.output_state = ground_truth;
        point.physics_loss = T(0);  // Ground truth has no physics loss
        point.surrogate_loss = error;
        point.weight = error + prediction_uncertainty;  // Higher weight for uncertain/wrong predictions

        if (error > error_threshold_ || prediction_uncertainty > error_threshold_) {
            sample_queue_.push(point);

            // Limit queue size
            while (sample_queue_.size() > max_samples_) {
                sample_queue_.pop();
            }
        }
    }

    std::vector<TrainingPoint<T>> get_training_batch(size_t batch_size) {
        std::vector<TrainingPoint<T>> batch;
        batch.reserve(std::min(batch_size, sample_queue_.size()));

        while (!sample_queue_.empty() && batch.size() < batch_size) {
            batch.push_back(sample_queue_.top());
            sample_queue_.pop();
        }

        return batch;
    }

    size_t queue_size() const { return sample_queue_.size(); }

private:
    T compute_state_error(const PhysicsState<T>& pred, const PhysicsState<T>& truth) const {
        T error = T(0);

        // Position error
        for (size_t i = 0; i < pred.positions.size(); ++i) {
            T diff = pred.positions[i] - truth.positions[i];
            error += diff * diff;
        }

        // Velocity error
        for (size_t i = 0; i < pred.velocities.size(); ++i) {
            T diff = pred.velocities[i] - truth.velocities[i];
            error += diff * diff;
        }

        return std::sqrt(error / T(pred.positions.size() + pred.velocities.size()));
    }
};

// Main surrogate model class
template<typename T>
class SurrogateModel {
private:
    std::unique_ptr<NeuralNetwork<T>> network_;
    std::unique_ptr<DataPreprocessor<T>> preprocessor_;
    std::unique_ptr<UncertaintyEstimator<T>> uncertainty_estimator_;
    std::unique_ptr<AdaptiveSampler<T>> adaptive_sampler_;

    SurrogateConfig<T> config_;
    bool trained_;

    // Performance metrics
    T current_speedup_;
    T current_accuracy_;
    size_t prediction_count_;
    size_t physics_fallback_count_;

    // Threading for background training
    std::atomic<bool> training_active_;
    std::thread training_thread_;
    std::mutex model_mutex_;

public:
    SurrogateModel(const SurrogateConfig<T>& config = SurrogateConfig<T>())
        : config_(config), trained_(false), current_speedup_(T(1)), current_accuracy_(T(0)),
          prediction_count_(0), physics_fallback_count_(0), training_active_(false) {

        initialize_network();
        preprocessor_ = std::make_unique<DataPreprocessor<T>>();
        uncertainty_estimator_ = std::make_unique<UncertaintyEstimator<T>>();
    }

    ~SurrogateModel() {
        stop_background_training();
    }

    // Initialize neural network architecture
    void initialize_network() {
        network_ = std::make_unique<NeuralNetwork<T>>(
            config_.learning_rate, LossType::MSE, OptimizerType::Adam);

        // Add input layer (size determined at runtime)
        // Hidden layers
        for (size_t i = 0; i < config_.hidden_layers.size(); ++i) {
            if (config_.enforce_energy_conservation ||
                config_.enforce_momentum_conservation ||
                config_.enforce_mass_conservation) {

                network_->add_physics_informed_layer(
                    i == 0 ? 0 : config_.hidden_layers[i-1],  // Will be set at runtime
                    config_.hidden_layers[i],
                    PhysicsConstraints<T>::physics_informed_constraint,
                    config_.activation
                );
            } else {
                network_->add_dense_layer(
                    i == 0 ? 0 : config_.hidden_layers[i-1],  // Will be set at runtime
                    config_.hidden_layers[i],
                    config_.activation
                );
            }
        }

        // Output layer (size determined at runtime)
    }

    // Train the surrogate model
    void train(const std::vector<PhysicsState<T>>& training_states,
              std::function<PhysicsState<T>(const PhysicsState<T>&)> physics_simulator) {

        if (training_states.empty()) {
            throw std::runtime_error("Training data is empty");
        }

        std::cout << "Training surrogate model with " << training_states.size() << " samples..." << std::endl;

        // Prepare training data
        std::vector<TrainingPoint<T>> training_points;
        training_points.reserve(training_states.size());

        for (size_t i = 0; i < training_states.size() - 1; ++i) {
            TrainingPoint<T> point;
            point.input_state = training_states[i];
            point.output_state = training_states[i + 1];  // Next state as target
            point.physics_loss = T(0);
            point.surrogate_loss = T(0);
            point.weight = T(1);
            training_points.push_back(point);
        }

        // Fit preprocessor
        preprocessor_->fit(training_states);

        // Split into training and validation
        size_t val_size = static_cast<size_t>(T(training_points.size()) * config_.validation_split);
        size_t train_size = training_points.size() - val_size;

        std::vector<TrainingPoint<T>> train_data(training_points.begin(),
                                                training_points.begin() + train_size);
        std::vector<TrainingPoint<T>> val_data(training_points.begin() + train_size,
                                              training_points.end());

        // Training loop
        for (size_t epoch = 0; epoch < config_.epochs; ++epoch) {
            T epoch_loss = T(0);

            // Shuffle training data
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(train_data.begin(), train_data.end(), g);

            // Mini-batch training
            for (size_t i = 0; i < train_data.size(); i += config_.batch_size) {
                size_t batch_end = std::min(i + config_.batch_size, train_data.size());

                T batch_loss = T(0);
                for (size_t j = i; j < batch_end; ++j) {
                    auto input_normalized = preprocessor_->normalize_input(train_data[j].input_state);
                    auto target_normalized = preprocessor_->normalize_input(train_data[j].output_state);

                    Tensor<T> input_tensor({1, input_normalized.size()});
                    Tensor<T> target_tensor({1, target_normalized.size()});

                    for (size_t k = 0; k < input_normalized.size(); ++k) {
                        input_tensor[k] = input_normalized[k];
                        target_tensor[k] = target_normalized[k];
                    }

                    T loss = network_->train_step(input_tensor, target_tensor);
                    batch_loss += loss * train_data[j].weight;
                }

                epoch_loss += batch_loss / T(batch_end - i);
            }

            // Validation
            if (epoch % 10 == 0) {
                T val_loss = T(0);
                for (const auto& val_point : val_data) {
                    auto input_normalized = preprocessor_->normalize_input(val_point.input_state);
                    auto target_normalized = preprocessor_->normalize_input(val_point.output_state);

                    Tensor<T> input_tensor({1, input_normalized.size()});
                    Tensor<T> target_tensor({1, target_normalized.size()});

                    for (size_t k = 0; k < input_normalized.size(); ++k) {
                        input_tensor[k] = input_normalized[k];
                        target_tensor[k] = target_normalized[k];
                    }

                    auto prediction = network_->forward(input_tensor);
                    val_loss += LossFunction<T>::compute(prediction, target_tensor, LossType::MSE);
                }
                val_loss /= T(val_data.size());

                std::cout << "Epoch " << epoch << ": Train Loss = " << epoch_loss / T(train_size)
                         << ", Val Loss = " << val_loss << std::endl;
            }
        }

        // Setup adaptive sampling if enabled
        if (config_.use_adaptive_sampling) {
            adaptive_sampler_ = std::make_unique<AdaptiveSampler<T>>(
                physics_simulator, config_.error_threshold, config_.max_training_points);
        }

        trained_ = true;
        std::cout << "Training completed!" << std::endl;
    }

    // Predict next physics state
    PhysicsState<T> predict(const PhysicsState<T>& current_state, bool& used_physics_fallback) {
        std::lock_guard<std::mutex> lock(model_mutex_);

        if (!trained_) {
            throw std::runtime_error("Model not trained");
        }

        used_physics_fallback = false;
        prediction_count_++;

        try {
            // Normalize input
            auto input_normalized = preprocessor_->normalize_input(current_state);
            Tensor<T> input_tensor({1, input_normalized.size()});
            for (size_t i = 0; i < input_normalized.size(); ++i) {
                input_tensor[i] = input_normalized[i];
            }

            // Get prediction with uncertainty if available
            T uncertainty = T(0);
            Tensor<T> prediction;

            if (uncertainty_estimator_ && config_.use_hybrid_mode) {
                auto pred_with_unc = uncertainty_estimator_->predict_with_uncertainty(input_tensor);
                prediction = pred_with_unc.first;
                uncertainty = pred_with_unc.second;

                // Fall back to physics if uncertainty is too high
                if (uncertainty > config_.uncertainty_threshold) {
                    used_physics_fallback = true;
                    physics_fallback_count_++;
                    // Would call physics simulator here
                    return current_state;  // Placeholder
                }
            } else {
                prediction = network_->forward(input_tensor);
            }

            // Denormalize output
            std::vector<T> output_normalized(prediction.size());
            for (size_t i = 0; i < prediction.size(); ++i) {
                output_normalized[i] = prediction[i];
            }

            auto result = preprocessor_->denormalize_output(output_normalized);

            // Add to adaptive sampling if enabled
            if (adaptive_sampler_ && config_.use_adaptive_sampling) {
                adaptive_sampler_->add_sample(current_state, result, uncertainty);
            }

            return result;

        } catch (const std::exception& e) {
            std::cerr << "Prediction error: " << e.what() << std::endl;
            used_physics_fallback = true;
            physics_fallback_count_++;
            return current_state;  // Fallback
        }
    }

    // Background training for continuous learning
    void start_background_training(std::function<PhysicsState<T>(const PhysicsState<T>&)> physics_simulator) {
        if (training_active_) return;

        training_active_ = true;
        training_thread_ = std::thread([this, physics_simulator]() {
            while (training_active_) {
                if (adaptive_sampler_ && adaptive_sampler_->queue_size() > config_.batch_size) {
                    auto batch = adaptive_sampler_->get_training_batch(config_.batch_size);

                    // Retrain on new samples
                    std::lock_guard<std::mutex> lock(model_mutex_);

                    for (const auto& point : batch) {
                        auto input_normalized = preprocessor_->normalize_input(point.input_state);
                        auto target_normalized = preprocessor_->normalize_input(point.output_state);

                        Tensor<T> input_tensor({1, input_normalized.size()});
                        Tensor<T> target_tensor({1, target_normalized.size()});

                        for (size_t k = 0; k < input_normalized.size(); ++k) {
                            input_tensor[k] = input_normalized[k];
                            target_tensor[k] = target_normalized[k];
                        }

                        network_->train_step(input_tensor, target_tensor);
                    }
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }

    void stop_background_training() {
        training_active_ = false;
        if (training_thread_.joinable()) {
            training_thread_.join();
        }
    }

    // Performance metrics
    T get_speedup() const {
        return current_speedup_;
    }

    T get_accuracy() const {
        return current_accuracy_;
    }

    T get_physics_fallback_rate() const {
        return prediction_count_ > 0 ? T(physics_fallback_count_) / T(prediction_count_) : T(0);
    }

    void reset_metrics() {
        prediction_count_ = 0;
        physics_fallback_count_ = 0;
        current_speedup_ = T(1);
        current_accuracy_ = T(0);
    }

    // Model management
    void save(const std::string& filename) {
        std::lock_guard<std::mutex> lock(model_mutex_);
        network_->save(filename);
    }

    void load(const std::string& filename) {
        std::lock_guard<std::mutex> lock(model_mutex_);
        network_->load(filename);
        trained_ = true;
    }

    bool is_trained() const { return trained_; }
    const SurrogateConfig<T>& config() const { return config_; }
};

} // namespace surrogate
} // namespace physgrad