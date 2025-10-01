#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <iostream>
#include <numeric>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace physgrad {
namespace pinns {

template<typename T>
struct Vec3 {
    T x, y, z;

    CUDA_HOST_DEVICE Vec3() : x(0), y(0), z(0) {}
    CUDA_HOST_DEVICE Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    CUDA_HOST_DEVICE Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    CUDA_HOST_DEVICE Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    CUDA_HOST_DEVICE Vec3 operator*(T scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    CUDA_HOST_DEVICE T dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    CUDA_HOST_DEVICE T norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
};

template<typename T>
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual T evaluate(T x) const = 0;
    virtual T derivative(T x) const = 0;
    virtual std::unique_ptr<ActivationFunction<T>> clone() const = 0;
};

template<typename T>
class TanhActivation : public ActivationFunction<T> {
public:
    T evaluate(T x) const override {
        return std::tanh(x);
    }

    T derivative(T x) const override {
        T tanh_x = std::tanh(x);
        return 1.0 - tanh_x * tanh_x;
    }

    std::unique_ptr<ActivationFunction<T>> clone() const override {
        return std::make_unique<TanhActivation<T>>();
    }
};

template<typename T>
class SineActivation : public ActivationFunction<T> {
public:
    T evaluate(T x) const override {
        return std::sin(x);
    }

    T derivative(T x) const override {
        return std::cos(x);
    }

    std::unique_ptr<ActivationFunction<T>> clone() const override {
        return std::make_unique<SineActivation<T>>();
    }
};

template<typename T>
class SwishActivation : public ActivationFunction<T> {
public:
    T evaluate(T x) const override {
        return x / (1.0 + std::exp(-x));
    }

    T derivative(T x) const override {
        T sigmoid = 1.0 / (1.0 + std::exp(-x));
        return sigmoid + x * sigmoid * (1.0 - sigmoid);
    }

    std::unique_ptr<ActivationFunction<T>> clone() const override {
        return std::make_unique<SwishActivation<T>>();
    }
};

template<typename T>
class NeuralNetwork {
private:
    std::vector<size_t> layer_sizes_;
    std::vector<std::vector<std::vector<T>>> weights_;
    std::vector<std::vector<T>> biases_;
    std::unique_ptr<ActivationFunction<T>> activation_;

public:
    NeuralNetwork(const std::vector<size_t>& layer_sizes,
                  std::unique_ptr<ActivationFunction<T>> activation)
        : layer_sizes_(layer_sizes), activation_(std::move(activation)) {

        initializeWeights();
    }

    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());

        weights_.resize(layer_sizes_.size() - 1);
        biases_.resize(layer_sizes_.size() - 1);

        for (size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
            size_t input_size = layer_sizes_[i];
            size_t output_size = layer_sizes_[i + 1];

            T xavier_bound = std::sqrt(6.0 / (input_size + output_size));
            std::uniform_real_distribution<T> dist(-xavier_bound, xavier_bound);

            weights_[i].resize(output_size);
            biases_[i].resize(output_size);

            for (size_t j = 0; j < output_size; ++j) {
                weights_[i][j].resize(input_size);
                for (size_t k = 0; k < input_size; ++k) {
                    weights_[i][j][k] = dist(gen);
                }
                biases_[i][j] = 0.0;
            }
        }
    }

    std::vector<T> forward(const std::vector<T>& input) const {
        std::vector<T> current = input;

        for (size_t layer = 0; layer < weights_.size(); ++layer) {
            std::vector<T> next(layer_sizes_[layer + 1]);

            for (size_t j = 0; j < layer_sizes_[layer + 1]; ++j) {
                T sum = biases_[layer][j];
                for (size_t k = 0; k < layer_sizes_[layer]; ++k) {
                    sum += weights_[layer][j][k] * current[k];
                }

                if (layer == weights_.size() - 1) {
                    next[j] = sum;
                } else {
                    next[j] = activation_->evaluate(sum);
                }
            }

            current = std::move(next);
        }

        return current;
    }

    std::vector<std::vector<T>> computeGradients(const std::vector<T>& input) const {
        std::vector<std::vector<T>> gradients(input.size());

        T eps = 1e-5;
        std::vector<T> output_base = forward(input);

        for (size_t i = 0; i < input.size(); ++i) {
            gradients[i].resize(output_base.size());

            std::vector<T> input_plus = input;
            input_plus[i] += eps;
            std::vector<T> output_plus = forward(input_plus);

            std::vector<T> input_minus = input;
            input_minus[i] -= eps;
            std::vector<T> output_minus = forward(input_minus);

            for (size_t j = 0; j < output_base.size(); ++j) {
                gradients[i][j] = (output_plus[j] - output_minus[j]) / (2.0 * eps);
            }
        }

        return gradients;
    }

    std::vector<std::vector<std::vector<T>>>& getWeights() { return weights_; }
    std::vector<std::vector<T>>& getBiases() { return biases_; }
    const std::vector<std::vector<std::vector<T>>>& getWeights() const { return weights_; }
    const std::vector<std::vector<T>>& getBiases() const { return biases_; }
};

template<typename T>
class PhysicsLoss {
public:
    virtual ~PhysicsLoss() = default;
    virtual T evaluate(const std::vector<T>& input, const std::vector<T>& output,
                      const std::vector<std::vector<T>>& gradients) const = 0;
    virtual std::string getName() const = 0;
};

template<typename T>
class NavierStokesLoss : public PhysicsLoss<T> {
private:
    T viscosity_;
    T density_;

public:
    NavierStokesLoss(T viscosity, T density) : viscosity_(viscosity), density_(density) {}

    T evaluate(const std::vector<T>& input, const std::vector<T>& output,
              const std::vector<std::vector<T>>& gradients) const override {

        if (input.size() < 4 || output.size() < 4) return 0.0;

        T x = input[0], y = input[1], z = input[2], t = input[3];
        T u = output[0], v = output[1], w = output[2], p = output[3];

        T u_x = gradients[0][0], u_y = gradients[1][0], u_z = gradients[2][0], u_t = gradients[3][0];
        T v_x = gradients[0][1], v_y = gradients[1][1], v_z = gradients[2][1], v_t = gradients[3][1];
        T w_x = gradients[0][2], w_y = gradients[1][2], w_z = gradients[2][2], w_t = gradients[3][2];
        T p_x = gradients[0][3], p_y = gradients[1][3], p_z = gradients[2][3];

        T nu = viscosity_ / density_;

        T momentum_x = u_t + u*u_x + v*u_y + w*u_z + p_x/density_ - nu*(u_x + u_y + u_z);
        T momentum_y = v_t + u*v_x + v*v_y + w*v_z + p_y/density_ - nu*(v_x + v_y + v_z);
        T momentum_z = w_t + u*w_x + v*w_y + w*w_z + p_z/density_ - nu*(w_x + w_y + w_z);
        T continuity = u_x + v_y + w_z;

        return momentum_x*momentum_x + momentum_y*momentum_y + momentum_z*momentum_z + continuity*continuity;
    }

    std::string getName() const override {
        return "NavierStokes";
    }
};

template<typename T>
class HeatEquationLoss : public PhysicsLoss<T> {
private:
    T thermal_diffusivity_;

public:
    HeatEquationLoss(T thermal_diffusivity) : thermal_diffusivity_(thermal_diffusivity) {}

    T evaluate(const std::vector<T>& input, const std::vector<T>& output,
              const std::vector<std::vector<T>>& gradients) const override {

        if (input.size() < 4 || output.size() < 1) return 0.0;

        T x = input[0], y = input[1], z = input[2], t = input[3];
        T u = output[0];

        T u_t = gradients[3][0];
        T u_x = gradients[0][0], u_y = gradients[1][0], u_z = gradients[2][0];

        T laplacian = u_x + u_y + u_z;
        T residual = u_t - thermal_diffusivity_ * laplacian;

        return residual * residual;
    }

    std::string getName() const override {
        return "HeatEquation";
    }
};

template<typename T>
class WaveEquationLoss : public PhysicsLoss<T> {
private:
    T wave_speed_;

public:
    WaveEquationLoss(T wave_speed) : wave_speed_(wave_speed) {}

    T evaluate(const std::vector<T>& input, const std::vector<T>& output,
              const std::vector<std::vector<T>>& gradients) const override {

        if (input.size() < 4 || output.size() < 1) return 0.0;

        T x = input[0], y = input[1], z = input[2], t = input[3];
        T u = output[0];

        T u_tt = gradients[3][0];
        T u_x = gradients[0][0], u_y = gradients[1][0], u_z = gradients[2][0];

        T laplacian = u_x + u_y + u_z;
        T residual = u_tt - wave_speed_ * wave_speed_ * laplacian;

        return residual * residual;
    }

    std::string getName() const override {
        return "WaveEquation";
    }
};

template<typename T>
class TrainingDataset {
private:
    std::vector<std::vector<T>> inputs_;
    std::vector<std::vector<T>> targets_;

public:
    void addDataPoint(const std::vector<T>& input, const std::vector<T>& target) {
        inputs_.push_back(input);
        targets_.push_back(target);
    }

    void generateCollocationPoints(size_t num_points, const std::vector<std::pair<T, T>>& bounds) {
        std::random_device rd;
        std::mt19937 gen(rd());

        inputs_.clear();
        targets_.clear();

        for (size_t i = 0; i < num_points; ++i) {
            std::vector<T> point;
            for (const auto& bound : bounds) {
                std::uniform_real_distribution<T> dist(bound.first, bound.second);
                point.push_back(dist(gen));
            }
            inputs_.push_back(point);
            targets_.push_back(std::vector<T>());
        }
    }

    size_t size() const { return inputs_.size(); }
    const std::vector<T>& getInput(size_t i) const { return inputs_[i]; }
    const std::vector<T>& getTarget(size_t i) const { return targets_[i]; }

    std::vector<size_t> generateBatchIndices(size_t batch_size) const {
        std::vector<size_t> indices(size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
        indices.resize(std::min(batch_size, indices.size()));
        return indices;
    }
};

template<typename T>
class PINNsOptimizer {
private:
    T learning_rate_;
    T momentum_;
    std::vector<std::vector<std::vector<T>>> weight_momentum_;
    std::vector<std::vector<T>> bias_momentum_;

public:
    PINNsOptimizer(T learning_rate, T momentum = 0.9)
        : learning_rate_(learning_rate), momentum_(momentum) {}

    void initializeMomentum(const NeuralNetwork<T>& network) {
        const auto& weights = network.getWeights();
        const auto& biases = network.getBiases();

        weight_momentum_.resize(weights.size());
        bias_momentum_.resize(biases.size());

        for (size_t i = 0; i < weights.size(); ++i) {
            weight_momentum_[i].resize(weights[i].size());
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weight_momentum_[i][j].resize(weights[i][j].size(), 0.0);
            }
            bias_momentum_[i].resize(biases[i].size(), 0.0);
        }
    }

    void updateParameters(NeuralNetwork<T>& network,
                         const std::vector<std::vector<std::vector<T>>>& weight_gradients,
                         const std::vector<std::vector<T>>& bias_gradients) {

        auto& weights = network.getWeights();
        auto& biases = network.getBiases();

        if (weight_momentum_.empty()) {
            initializeMomentum(network);
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                for (size_t k = 0; k < weights[i][j].size(); ++k) {
                    weight_momentum_[i][j][k] = momentum_ * weight_momentum_[i][j][k] +
                                               learning_rate_ * weight_gradients[i][j][k];
                    weights[i][j][k] -= weight_momentum_[i][j][k];
                }
            }

            for (size_t j = 0; j < biases[i].size(); ++j) {
                bias_momentum_[i][j] = momentum_ * bias_momentum_[i][j] +
                                      learning_rate_ * bias_gradients[i][j];
                biases[i][j] -= bias_momentum_[i][j];
            }
        }
    }
};

template<typename T>
class PINNsFramework {
private:
    std::unique_ptr<NeuralNetwork<T>> network_;
    std::vector<std::unique_ptr<PhysicsLoss<T>>> physics_losses_;
    std::unique_ptr<PINNsOptimizer<T>> optimizer_;
    T physics_weight_;
    T data_weight_;

public:
    PINNsFramework(std::unique_ptr<NeuralNetwork<T>> network,
                   std::unique_ptr<PINNsOptimizer<T>> optimizer,
                   T physics_weight = 1.0, T data_weight = 1.0)
        : network_(std::move(network)), optimizer_(std::move(optimizer)),
          physics_weight_(physics_weight), data_weight_(data_weight) {}

    void addPhysicsLoss(std::unique_ptr<PhysicsLoss<T>> loss) {
        physics_losses_.push_back(std::move(loss));
    }

    T computeDataLoss(const TrainingDataset<T>& dataset, const std::vector<size_t>& batch_indices) const {
        T total_loss = 0.0;

        for (size_t idx : batch_indices) {
            const auto& input = dataset.getInput(idx);
            const auto& target = dataset.getTarget(idx);

            if (!target.empty()) {
                auto output = network_->forward(input);
                for (size_t i = 0; i < std::min(output.size(), target.size()); ++i) {
                    T diff = output[i] - target[i];
                    total_loss += diff * diff;
                }
            }
        }

        return total_loss / batch_indices.size();
    }

    T computePhysicsLoss(const TrainingDataset<T>& dataset, const std::vector<size_t>& batch_indices) const {
        T total_loss = 0.0;

        for (size_t idx : batch_indices) {
            const auto& input = dataset.getInput(idx);
            auto output = network_->forward(input);
            auto gradients = network_->computeGradients(input);

            for (const auto& loss : physics_losses_) {
                total_loss += loss->evaluate(input, output, gradients);
            }
        }

        return total_loss / batch_indices.size();
    }

    T trainEpoch(const TrainingDataset<T>& dataset, size_t batch_size = 32) {
        auto batch_indices = dataset.generateBatchIndices(batch_size);

        T data_loss = computeDataLoss(dataset, batch_indices);
        T physics_loss = computePhysicsLoss(dataset, batch_indices);
        T total_loss = data_weight_ * data_loss + physics_weight_ * physics_loss;

        auto weight_gradients = computeWeightGradients(dataset, batch_indices);
        auto bias_gradients = computeBiasGradients(dataset, batch_indices);

        optimizer_->updateParameters(*network_, weight_gradients, bias_gradients);

        return total_loss;
    }

    std::vector<T> predict(const std::vector<T>& input) const {
        return network_->forward(input);
    }

    void saveModel(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }

        const auto& weights = network_->getWeights();
        const auto& biases = network_->getBiases();

        size_t num_layers = weights.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        for (size_t i = 0; i < weights.size(); ++i) {
            size_t rows = weights[i].size();
            size_t cols = weights[i][0].size();
            file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

            for (size_t j = 0; j < rows; ++j) {
                file.write(reinterpret_cast<const char*>(weights[i][j].data()), cols * sizeof(T));
            }

            file.write(reinterpret_cast<const char*>(biases[i].data()), rows * sizeof(T));
        }

        file.close();
    }

private:
    std::vector<std::vector<std::vector<T>>> computeWeightGradients(
        const TrainingDataset<T>& dataset, const std::vector<size_t>& batch_indices) const {

        const auto& weights = network_->getWeights();
        std::vector<std::vector<std::vector<T>>> gradients(weights.size());

        for (size_t i = 0; i < weights.size(); ++i) {
            gradients[i].resize(weights[i].size());
            for (size_t j = 0; j < weights[i].size(); ++j) {
                gradients[i][j].resize(weights[i][j].size(), 0.0);
            }
        }

        return gradients;
    }

    std::vector<std::vector<T>> computeBiasGradients(
        const TrainingDataset<T>& dataset, const std::vector<size_t>& batch_indices) const {

        const auto& biases = network_->getBiases();
        std::vector<std::vector<T>> gradients(biases.size());

        for (size_t i = 0; i < biases.size(); ++i) {
            gradients[i].resize(biases[i].size(), 0.0);
        }

        return gradients;
    }
};

template<typename T>
class PINNsFactory {
public:
    static std::unique_ptr<PINNsFramework<T>> createNavierStokesFramework(
        const std::vector<size_t>& layer_sizes,
        T viscosity, T density,
        T learning_rate = 1e-3) {

        auto activation = std::make_unique<TanhActivation<T>>();
        auto network = std::make_unique<NeuralNetwork<T>>(layer_sizes, std::move(activation));
        auto optimizer = std::make_unique<PINNsOptimizer<T>>(learning_rate);

        auto framework = std::make_unique<PINNsFramework<T>>(std::move(network), std::move(optimizer));
        framework->addPhysicsLoss(std::make_unique<NavierStokesLoss<T>>(viscosity, density));

        return framework;
    }

    static std::unique_ptr<PINNsFramework<T>> createHeatEquationFramework(
        const std::vector<size_t>& layer_sizes,
        T thermal_diffusivity,
        T learning_rate = 1e-3) {

        auto activation = std::make_unique<SineActivation<T>>();
        auto network = std::make_unique<NeuralNetwork<T>>(layer_sizes, std::move(activation));
        auto optimizer = std::make_unique<PINNsOptimizer<T>>(learning_rate);

        auto framework = std::make_unique<PINNsFramework<T>>(std::move(network), std::move(optimizer));
        framework->addPhysicsLoss(std::make_unique<HeatEquationLoss<T>>(thermal_diffusivity));

        return framework;
    }

    static std::unique_ptr<PINNsFramework<T>> createWaveEquationFramework(
        const std::vector<size_t>& layer_sizes,
        T wave_speed,
        T learning_rate = 1e-3) {

        auto activation = std::make_unique<SwishActivation<T>>();
        auto network = std::make_unique<NeuralNetwork<T>>(layer_sizes, std::move(activation));
        auto optimizer = std::make_unique<PINNsOptimizer<T>>(learning_rate);

        auto framework = std::make_unique<PINNsFramework<T>>(std::move(network), std::move(optimizer));
        framework->addPhysicsLoss(std::make_unique<WaveEquationLoss<T>>(wave_speed));

        return framework;
    }
};

template<typename T>
class PINNsPhysicsIntegrator {
private:
    std::unordered_map<std::string, std::unique_ptr<PINNsFramework<T>>> frameworks_;

public:
    void addFramework(const std::string& name, std::unique_ptr<PINNsFramework<T>> framework) {
        frameworks_[name] = std::move(framework);
    }

    bool hasFramework(const std::string& name) const {
        return frameworks_.find(name) != frameworks_.end();
    }

    PINNsFramework<T>* getFramework(const std::string& name) {
        auto it = frameworks_.find(name);
        return (it != frameworks_.end()) ? it->second.get() : nullptr;
    }

    std::vector<T> predictPhysics(const std::string& framework_name, const std::vector<T>& input) {
        auto* framework = getFramework(framework_name);
        if (!framework) {
            throw std::runtime_error("Framework not found: " + framework_name);
        }
        return framework->predict(input);
    }

    T trainFramework(const std::string& framework_name, const TrainingDataset<T>& dataset,
                    size_t epochs = 100, size_t batch_size = 32) {
        auto* framework = getFramework(framework_name);
        if (!framework) {
            throw std::runtime_error("Framework not found: " + framework_name);
        }

        T final_loss = 0.0;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            final_loss = framework->trainEpoch(dataset, batch_size);

            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << final_loss << std::endl;
            }
        }

        return final_loss;
    }

    void saveAllModels(const std::string& directory) const {
        for (const auto& pair : frameworks_) {
            std::string filename = directory + "/" + pair.first + "_model.bin";
            pair.second->saveModel(filename);
        }
    }

    size_t getFrameworkCount() const {
        return frameworks_.size();
    }

    std::vector<std::string> getFrameworkNames() const {
        std::vector<std::string> names;
        for (const auto& pair : frameworks_) {
            names.push_back(pair.first);
        }
        return names;
    }
};

} // namespace pinns
} // namespace physgrad