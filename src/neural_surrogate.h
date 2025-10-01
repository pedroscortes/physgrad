#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <random>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#endif

namespace physgrad {
namespace neural {

// Forward declarations
template<typename T> class Tensor;
template<typename T> class Layer;
template<typename T> class NeuralNetwork;

// Activation functions
enum class ActivationType {
    ReLU,
    Tanh,
    Sigmoid,
    LeakyReLU,
    ELU,
    Swish,
    GELU
};

// Loss functions
enum class LossType {
    MSE,
    MAE,
    Huber,
    LogCosh
};

// Optimizer types
enum class OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop
};

// Neural network tensor implementation
template<typename T>
class Tensor {
private:
    std::vector<size_t> shape_;
    std::vector<T> data_;
    bool requires_grad_;
    std::vector<T> grad_;

#ifdef ENABLE_CUDA
    T* device_data_;
    T* device_grad_;
    bool on_device_;
#endif

public:
    Tensor() : requires_grad_(false)
#ifdef ENABLE_CUDA
        , device_data_(nullptr), device_grad_(nullptr), on_device_(false)
#endif
    {}

    Tensor(const std::vector<size_t>& shape, bool requires_grad = false)
        : shape_(shape), requires_grad_(requires_grad)
#ifdef ENABLE_CUDA
        , device_data_(nullptr), device_grad_(nullptr), on_device_(false)
#endif
    {
        size_t total_size = 1;
        for (size_t dim : shape_) {
            total_size *= dim;
        }
        data_.resize(total_size);
        if (requires_grad_) {
            grad_.resize(total_size, T(0));
        }
    }

    // Basic operations
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }
    bool requires_grad() const { return requires_grad_; }

    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const { return data_[idx]; }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    T* grad() { return grad_.data(); }
    const T* grad() const { return grad_.data(); }

    // Tensor operations
    Tensor<T> operator+(const Tensor<T>& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Shape mismatch in tensor addition");
        }

        Tensor<T> result(shape_, requires_grad_ || other.requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Tensor<T> operator-(const Tensor<T>& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Shape mismatch in tensor subtraction");
        }

        Tensor<T> result(shape_, requires_grad_ || other.requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    Tensor<T> operator*(T scalar) const {
        Tensor<T> result(shape_, requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    // Matrix multiplication
    Tensor<T> matmul(const Tensor<T>& other) const {
        if (shape_.size() != 2 || other.shape_.size() != 2) {
            throw std::runtime_error("Matrix multiplication requires 2D tensors");
        }
        if (shape_[1] != other.shape_[0]) {
            throw std::runtime_error("Matrix dimension mismatch");
        }

        size_t m = shape_[0];
        size_t n = other.shape_[1];
        size_t k = shape_[1];

        Tensor<T> result({m, n}, requires_grad_ || other.requires_grad_);

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = T(0);
                for (size_t l = 0; l < k; ++l) {
                    sum += data_[i * k + l] * other.data_[l * n + j];
                }
                result.data_[i * n + j] = sum;
            }
        }

        return result;
    }

    // Reshape tensor
    Tensor<T> reshape(const std::vector<size_t>& new_shape) const {
        size_t total_size = 1;
        for (size_t dim : new_shape) {
            total_size *= dim;
        }
        if (total_size != data_.size()) {
            throw std::runtime_error("Reshape size mismatch");
        }

        Tensor<T> result(new_shape, requires_grad_);
        result.data_ = data_;
        if (requires_grad_) {
            result.grad_ = grad_;
        }
        return result;
    }

    // Initialize with random values
    void xavier_uniform(T gain = T(1)) {
        std::random_device rd;
        std::mt19937 gen(rd());

        T fan_in = shape_.size() > 1 ? T(shape_[0]) : T(1);
        T fan_out = shape_.size() > 1 ? T(shape_[1]) : T(1);
        T std_dev = gain * std::sqrt(T(2) / (fan_in + fan_out));

        std::normal_distribution<T> dist(T(0), std_dev);

        for (auto& val : data_) {
            val = dist(gen);
        }
    }

    void he_uniform() {
        std::random_device rd;
        std::mt19937 gen(rd());

        T fan_in = shape_.size() > 1 ? T(shape_[0]) : T(1);
        T std_dev = std::sqrt(T(2) / fan_in);

        std::normal_distribution<T> dist(T(0), std_dev);

        for (auto& val : data_) {
            val = dist(gen);
        }
    }

    // Zero gradients
    void zero_grad() {
        if (requires_grad_) {
            std::fill(grad_.begin(), grad_.end(), T(0));
        }
    }

#ifdef ENABLE_CUDA
    // CUDA operations
    void to_device() {
        if (!on_device_) {
            size_t bytes = data_.size() * sizeof(T);
            cudaMalloc(&device_data_, bytes);
            cudaMemcpy(device_data_, data_.data(), bytes, cudaMemcpyHostToDevice);

            if (requires_grad_) {
                cudaMalloc(&device_grad_, bytes);
                cudaMemcpy(device_grad_, grad_.data(), bytes, cudaMemcpyHostToDevice);
            }
            on_device_ = true;
        }
    }

    void to_cpu() {
        if (on_device_) {
            size_t bytes = data_.size() * sizeof(T);
            cudaMemcpy(data_.data(), device_data_, bytes, cudaMemcpyDeviceToHost);

            if (requires_grad_) {
                cudaMemcpy(grad_.data(), device_grad_, bytes, cudaMemcpyDeviceToHost);
            }
        }
    }

    T* device_data() { return device_data_; }
    T* device_grad() { return device_grad_; }
    bool on_device() const { return on_device_; }
#endif
};

// Activation functions implementation
template<typename T>
class ActivationFunction {
public:
    static Tensor<T> apply(const Tensor<T>& input, ActivationType type) {
        Tensor<T> output(input.shape(), input.requires_grad());

        switch (type) {
            case ActivationType::ReLU:
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = std::max(T(0), input[i]);
                }
                break;

            case ActivationType::Tanh:
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = std::tanh(input[i]);
                }
                break;

            case ActivationType::Sigmoid:
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = T(1) / (T(1) + std::exp(-input[i]));
                }
                break;

            case ActivationType::LeakyReLU:
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = input[i] > T(0) ? input[i] : T(0.01) * input[i];
                }
                break;

            case ActivationType::ELU:
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = input[i] > T(0) ? input[i] : std::exp(input[i]) - T(1);
                }
                break;

            case ActivationType::Swish:
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = input[i] / (T(1) + std::exp(-input[i]));
                }
                break;

            case ActivationType::GELU:
                for (size_t i = 0; i < input.size(); ++i) {
                    T x = input[i];
                    output[i] = T(0.5) * x * (T(1) + std::tanh(std::sqrt(T(2) / M_PI) * (x + T(0.044715) * x * x * x)));
                }
                break;
        }

        return output;
    }
};

// Neural network layer
template<typename T>
class Layer {
protected:
    size_t input_size_;
    size_t output_size_;
    ActivationType activation_;

public:
    Layer(size_t input_size, size_t output_size, ActivationType activation = ActivationType::ReLU)
        : input_size_(input_size), output_size_(output_size), activation_(activation) {}

    virtual ~Layer() = default;

    virtual Tensor<T> forward(const Tensor<T>& input) = 0;
    virtual void update_weights(T learning_rate) = 0;
    virtual void zero_grad() = 0;

    size_t input_size() const { return input_size_; }
    size_t output_size() const { return output_size_; }
};

// Dense (fully connected) layer
template<typename T>
class DenseLayer : public Layer<T> {
private:
    Tensor<T> weights_;
    Tensor<T> bias_;

public:
    DenseLayer(size_t input_size, size_t output_size, ActivationType activation = ActivationType::ReLU)
        : Layer<T>(input_size, output_size, activation),
          weights_({input_size, output_size}, true),
          bias_({output_size}, true) {

        weights_.xavier_uniform();
        // Initialize bias to zero
        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_[i] = T(0);
        }
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Linear transformation: output = input * weights + bias
        auto linear_output = input.matmul(weights_);

        // Add bias
        for (size_t i = 0; i < linear_output.shape()[0]; ++i) {
            for (size_t j = 0; j < linear_output.shape()[1]; ++j) {
                linear_output[i * linear_output.shape()[1] + j] += bias_[j];
            }
        }

        // Apply activation
        return ActivationFunction<T>::apply(linear_output, this->activation_);
    }

    void update_weights(T learning_rate) override {
        // Simple SGD update
        for (size_t i = 0; i < weights_.size(); ++i) {
            weights_[i] -= learning_rate * weights_.grad()[i];
        }
        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_[i] -= learning_rate * bias_.grad()[i];
        }
    }

    void zero_grad() override {
        weights_.zero_grad();
        bias_.zero_grad();
    }

    const Tensor<T>& weights() const { return weights_; }
    const Tensor<T>& bias() const { return bias_; }
};

// Physics-informed layer for conserving physical properties
template<typename T>
class PhysicsInformedLayer : public Layer<T> {
private:
    Tensor<T> weights_;
    Tensor<T> bias_;
    std::function<Tensor<T>(const Tensor<T>&)> physics_constraint_;

public:
    PhysicsInformedLayer(size_t input_size, size_t output_size,
                        std::function<Tensor<T>(const Tensor<T>&)> constraint,
                        ActivationType activation = ActivationType::ReLU)
        : Layer<T>(input_size, output_size, activation),
          weights_({input_size, output_size}, true),
          bias_({output_size}, true),
          physics_constraint_(constraint) {

        weights_.xavier_uniform();
        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_[i] = T(0);
        }
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        auto linear_output = input.matmul(weights_);

        // Add bias
        for (size_t i = 0; i < linear_output.shape()[0]; ++i) {
            for (size_t j = 0; j < linear_output.shape()[1]; ++j) {
                linear_output[i * linear_output.shape()[1] + j] += bias_[j];
            }
        }

        // Apply activation
        auto activated = ActivationFunction<T>::apply(linear_output, this->activation_);

        // Apply physics constraint
        if (physics_constraint_) {
            return physics_constraint_(activated);
        }

        return activated;
    }

    void update_weights(T learning_rate) override {
        for (size_t i = 0; i < weights_.size(); ++i) {
            weights_[i] -= learning_rate * weights_.grad()[i];
        }
        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_[i] -= learning_rate * bias_.grad()[i];
        }
    }

    void zero_grad() override {
        weights_.zero_grad();
        bias_.zero_grad();
    }
};

// Loss functions
template<typename T>
class LossFunction {
public:
    static T compute(const Tensor<T>& predictions, const Tensor<T>& targets, LossType type) {
        if (predictions.shape() != targets.shape()) {
            throw std::runtime_error("Prediction and target shapes must match");
        }

        T loss = T(0);
        size_t n = predictions.size();

        switch (type) {
            case LossType::MSE:
                for (size_t i = 0; i < n; ++i) {
                    T diff = predictions[i] - targets[i];
                    loss += diff * diff;
                }
                loss /= T(n);
                break;

            case LossType::MAE:
                for (size_t i = 0; i < n; ++i) {
                    loss += std::abs(predictions[i] - targets[i]);
                }
                loss /= T(n);
                break;

            case LossType::Huber:
                {
                    T delta = T(1.0);
                    for (size_t i = 0; i < n; ++i) {
                        T diff = std::abs(predictions[i] - targets[i]);
                        if (diff <= delta) {
                            loss += T(0.5) * diff * diff;
                        } else {
                            loss += delta * (diff - T(0.5) * delta);
                        }
                    }
                    loss /= T(n);
                }
                break;

            case LossType::LogCosh:
                for (size_t i = 0; i < n; ++i) {
                    T diff = predictions[i] - targets[i];
                    loss += std::log(std::cosh(diff));
                }
                loss /= T(n);
                break;
        }

        return loss;
    }
};

// Neural network implementation
template<typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer<T>>> layers_;
    T learning_rate_;
    LossType loss_type_;
    OptimizerType optimizer_type_;

    // Adam optimizer state
    std::vector<Tensor<T>> m_weights_;  // First moment
    std::vector<Tensor<T>> v_weights_;  // Second moment
    std::vector<Tensor<T>> m_bias_;
    std::vector<Tensor<T>> v_bias_;
    int t_;  // Time step

public:
    NeuralNetwork(T learning_rate = T(0.001),
                  LossType loss_type = LossType::MSE,
                  OptimizerType optimizer_type = OptimizerType::Adam)
        : learning_rate_(learning_rate), loss_type_(loss_type),
          optimizer_type_(optimizer_type), t_(0) {}

    // Add layers
    void add_dense_layer(size_t input_size, size_t output_size,
                        ActivationType activation = ActivationType::ReLU) {
        layers_.push_back(std::make_unique<DenseLayer<T>>(input_size, output_size, activation));
    }

    void add_physics_informed_layer(size_t input_size, size_t output_size,
                                   std::function<Tensor<T>(const Tensor<T>&)> constraint,
                                   ActivationType activation = ActivationType::ReLU) {
        layers_.push_back(std::make_unique<PhysicsInformedLayer<T>>(
            input_size, output_size, constraint, activation));
    }

    // Forward pass
    Tensor<T> forward(const Tensor<T>& input) {
        Tensor<T> current = input;
        for (auto& layer : layers_) {
            current = layer->forward(current);
        }
        return current;
    }

    // Training step
    T train_step(const Tensor<T>& input, const Tensor<T>& target) {
        // Forward pass
        auto prediction = forward(input);

        // Compute loss
        T loss = LossFunction<T>::compute(prediction, target, loss_type_);

        // Backward pass (simplified - would need proper autograd)
        backward(prediction, target);

        // Update weights
        update_weights();

        return loss;
    }

    // Simple backward pass implementation
    void backward(const Tensor<T>& prediction [[maybe_unused]], const Tensor<T>& target [[maybe_unused]]) {
        // This is a simplified backward pass
        // In a real implementation, you'd need proper computational graph tracking

        for (auto& layer : layers_) {
            layer->zero_grad();
        }

        // Compute gradients (simplified)
        // Real implementation would use automatic differentiation
    }

    void update_weights() {
        if (optimizer_type_ == OptimizerType::SGD) {
            for (auto& layer : layers_) {
                layer->update_weights(learning_rate_);
            }
        } else if (optimizer_type_ == OptimizerType::Adam) {
            // Adam optimizer implementation
            t_++;
            [[maybe_unused]] T beta1 = T(0.9);
            [[maybe_unused]] T beta2 = T(0.999);
            [[maybe_unused]] T epsilon = T(1e-8);

            for (auto& layer : layers_) {
                layer->update_weights(learning_rate_);  // Simplified
            }
        }
    }

    // Evaluation
    T evaluate(const std::vector<Tensor<T>>& inputs, const std::vector<Tensor<T>>& targets) {
        T total_loss = T(0);
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto prediction = forward(inputs[i]);
            total_loss += LossFunction<T>::compute(prediction, targets[i], loss_type_);
        }
        return total_loss / T(inputs.size());
    }

    // Save/load model
    void save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for saving");
        }

        // Save network architecture and weights
        size_t num_layers = layers_.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        for (auto& layer : layers_) {
            // Save layer type and parameters
            // This would need more sophisticated serialization
        }

        file.close();
    }

    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for loading");
        }

        // Load network architecture and weights
        size_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

        // Reconstruct network
        // This would need more sophisticated deserialization

        file.close();
    }

    size_t num_layers() const { return layers_.size(); }
    void set_learning_rate(T lr) { learning_rate_ = lr; }
    T get_learning_rate() const { return learning_rate_; }
};

} // namespace neural
} // namespace physgrad