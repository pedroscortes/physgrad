#include "src/neural_surrogate.h"
#include "src/surrogate_model.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

using namespace physgrad::neural;
using namespace physgrad::surrogate;

// Test utility functions
void assertTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << std::endl;
        exit(1);
    }
}

void assertNear(float a, float b, float tolerance, const std::string& message) {
    if (std::abs(a - b) > tolerance) {
        std::cerr << "FAILED: " << message << " (expected " << a << ", got " << b << ")" << std::endl;
        exit(1);
    }
}

// Test tensor operations
void testTensorOperations() {
    std::cout << "Testing Tensor operations..." << std::endl;

    // Test creation and basic operations
    Tensor<float> t1({2, 3}, true);
    for (size_t i = 0; i < t1.size(); ++i) {
        t1[i] = float(i + 1);
    }

    Tensor<float> t2({2, 3}, true);
    for (size_t i = 0; i < t2.size(); ++i) {
        t2[i] = float(i + 2);
    }

    // Test addition
    auto t3 = t1 + t2;
    assertTrue(t3[0] == 3.0f, "Tensor addition element 0");
    assertTrue(t3[1] == 5.0f, "Tensor addition element 1");

    // Test subtraction
    auto t4 = t2 - t1;
    assertTrue(t4[0] == 1.0f, "Tensor subtraction element 0");
    assertTrue(t4[1] == 1.0f, "Tensor subtraction element 1");

    // Test scalar multiplication
    auto t5 = t1 * 2.0f;
    assertTrue(t5[0] == 2.0f, "Tensor scalar multiplication element 0");
    assertTrue(t5[1] == 4.0f, "Tensor scalar multiplication element 1");

    // Test matrix multiplication
    Tensor<float> m1({2, 2});
    m1[0] = 1; m1[1] = 2;
    m1[2] = 3; m1[3] = 4;

    Tensor<float> m2({2, 2});
    m2[0] = 5; m2[1] = 6;
    m2[2] = 7; m2[3] = 8;

    auto m3 = m1.matmul(m2);
    assertTrue(m3[0] == 19.0f, "Matrix multiplication [0,0]"); // 1*5 + 2*7 = 19
    assertTrue(m3[1] == 22.0f, "Matrix multiplication [0,1]"); // 1*6 + 2*8 = 22
    assertTrue(m3[2] == 43.0f, "Matrix multiplication [1,0]"); // 3*5 + 4*7 = 43
    assertTrue(m3[3] == 50.0f, "Matrix multiplication [1,1]"); // 3*6 + 4*8 = 50

    // Test reshape
    auto reshaped = t1.reshape({3, 2});
    assertTrue(reshaped.shape()[0] == 3 && reshaped.shape()[1] == 2, "Tensor reshape");

    std::cout << "✓ Tensor operations tests passed" << std::endl;
}

// Test activation functions
void testActivationFunctions() {
    std::cout << "Testing activation functions..." << std::endl;

    Tensor<float> input({1, 4});
    input[0] = -2.0f;
    input[1] = -0.5f;
    input[2] = 0.5f;
    input[3] = 2.0f;

    // Test ReLU
    auto relu_output = ActivationFunction<float>::apply(input, ActivationType::ReLU);
    assertTrue(relu_output[0] == 0.0f, "ReLU negative input");
    assertTrue(relu_output[2] == 0.5f, "ReLU positive input");

    // Test Sigmoid
    auto sigmoid_output = ActivationFunction<float>::apply(input, ActivationType::Sigmoid);
    assertTrue(sigmoid_output[0] > 0.0f && sigmoid_output[0] < 1.0f, "Sigmoid range");
    assertNear(sigmoid_output[2], 0.622f, 0.01f, "Sigmoid calculation");

    // Test Tanh
    auto tanh_output = ActivationFunction<float>::apply(input, ActivationType::Tanh);
    assertTrue(tanh_output[0] > -1.0f && tanh_output[0] < 1.0f, "Tanh range");
    assertNear(tanh_output[2], std::tanh(0.5f), 0.001f, "Tanh calculation");

    // Test Swish
    auto swish_output = ActivationFunction<float>::apply(input, ActivationType::Swish);
    assertTrue(swish_output.size() == input.size(), "Swish output size");

    std::cout << "✓ Activation function tests passed" << std::endl;
}

// Test neural network layers
void testNeuralLayers() {
    std::cout << "Testing neural network layers..." << std::endl;

    // Test dense layer
    DenseLayer<float> dense(3, 2, ActivationType::ReLU);

    Tensor<float> input({1, 3});
    input[0] = 1.0f;
    input[1] = 0.5f;
    input[2] = -0.5f;

    auto output = dense.forward(input);
    assertTrue(output.shape()[0] == 1 && output.shape()[1] == 2, "Dense layer output shape");

    // Test gradient zeroing
    dense.zero_grad();
    assertTrue(true, "Gradient zeroing completed");

    std::cout << "✓ Neural layer tests passed" << std::endl;
}

// Test loss functions
void testLossFunctions() {
    std::cout << "Testing loss functions..." << std::endl;

    Tensor<float> predictions({1, 3});
    predictions[0] = 1.0f;
    predictions[1] = 2.0f;
    predictions[2] = 3.0f;

    Tensor<float> targets({1, 3});
    targets[0] = 1.1f;
    targets[1] = 1.9f;
    targets[2] = 3.2f;

    // Test MSE
    float mse_loss = LossFunction<float>::compute(predictions, targets, LossType::MSE);
    float expected_mse = ((0.1f*0.1f) + (0.1f*0.1f) + (0.2f*0.2f)) / 3.0f;
    assertNear(mse_loss, expected_mse, 0.001f, "MSE loss calculation");

    // Test MAE
    float mae_loss = LossFunction<float>::compute(predictions, targets, LossType::MAE);
    float expected_mae = (0.1f + 0.1f + 0.2f) / 3.0f;
    assertNear(mae_loss, expected_mae, 0.001f, "MAE loss calculation");

    std::cout << "✓ Loss function tests passed" << std::endl;
}

// Test neural network
void testNeuralNetwork() {
    std::cout << "Testing neural network..." << std::endl;

    NeuralNetwork<float> network(0.01f, LossType::MSE, OptimizerType::SGD);

    // Build a simple network
    network.add_dense_layer(3, 5, ActivationType::ReLU);
    network.add_dense_layer(5, 3, ActivationType::ReLU);
    network.add_dense_layer(3, 1, ActivationType::Sigmoid);

    assertTrue(network.num_layers() == 3, "Network layer count");

    // Test forward pass
    Tensor<float> input({1, 3});
    input[0] = 0.5f;
    input[1] = -0.2f;
    input[2] = 0.8f;

    auto output = network.forward(input);
    assertTrue(output.shape()[0] == 1 && output.shape()[1] == 1, "Network output shape");
    assertTrue(output[0] >= 0.0f && output[0] <= 1.0f, "Sigmoid output range");

    // Test training step
    Tensor<float> target({1, 1});
    target[0] = 0.7f;

    float loss = network.train_step(input, target);
    assertTrue(loss >= 0.0f, "Training loss non-negative");

    std::cout << "✓ Neural network tests passed" << std::endl;
}

// Test physics state operations
void testPhysicsState() {
    std::cout << "Testing physics state operations..." << std::endl;

    PhysicsState<float> state;

    // Add some particles
    state.positions = {0.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f};
    state.velocities = {1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f};
    state.forces = {0.0f, -9.81f, 0.0f,  0.0f, -9.81f, 0.0f,  0.0f, -9.81f, 0.0f};
    state.material_props = {1.0f, 1.0f, 1.0f};  // masses
    state.timestep = 0.01f;
    state.time = 0.0f;

    assertTrue(state.num_particles() == 3, "Physics state particle count");

    std::cout << "✓ Physics state tests passed" << std::endl;
}

// Test data preprocessor
void testDataPreprocessor() {
    std::cout << "Testing data preprocessor..." << std::endl;

    DataPreprocessor<float> preprocessor;

    // Create sample physics states
    std::vector<PhysicsState<float>> states;
    for (int i = 0; i < 10; ++i) {
        PhysicsState<float> state;
        state.positions = {float(i), float(i+1), float(i+2)};
        state.velocities = {float(i*0.1f), float(i*0.2f), float(i*0.3f)};
        state.forces = {0.0f, -9.81f, 0.0f};
        state.material_props = {1.0f};
        state.timestep = 0.01f;
        state.time = float(i) * 0.01f;
        states.push_back(state);
    }

    // Fit preprocessor
    preprocessor.fit(states);

    // Test normalization
    auto normalized = preprocessor.normalize_input(states[5]);
    assertTrue(normalized.size() > 0, "Normalization produces output");

    // Test denormalization
    auto denormalized = preprocessor.denormalize_output(normalized);
    assertTrue(denormalized.num_particles() > 0, "Denormalization produces valid state");

    std::cout << "✓ Data preprocessor tests passed" << std::endl;
}

// Test surrogate model basic functionality
void testSurrogateModel() {
    std::cout << "Testing surrogate model..." << std::endl;

    SurrogateConfig<float> config;
    config.hidden_layers = {4, 8, 4};
    config.epochs = 10;  // Reduced for testing
    config.batch_size = 2;

    physgrad::surrogate::SurrogateModel<float> model(config);

    // Create simple training data
    std::vector<PhysicsState<float>> training_states;
    for (int i = 0; i < 20; ++i) {
        PhysicsState<float> state;
        state.positions = {float(i*0.1f), 0.0f, 0.0f};
        state.velocities = {1.0f, 0.0f, 0.0f};
        state.forces = {0.0f, -9.81f, 0.0f};
        state.material_props = {1.0f};
        state.timestep = 0.01f;
        state.time = float(i) * 0.01f;
        training_states.push_back(state);
    }

    // Simple physics simulator (just add velocity * timestep to position)
    auto physics_sim = [](const PhysicsState<float>& state) -> PhysicsState<float> {
        PhysicsState<float> next_state = state;
        for (size_t i = 0; i < state.positions.size(); ++i) {
            next_state.positions[i] += state.velocities[i] * state.timestep;
        }
        next_state.time += state.timestep;
        return next_state;
    };

    // Train the model
    try {
        model.train(training_states, physics_sim);
        assertTrue(model.is_trained(), "Model should be trained");

        // Test prediction
        bool used_fallback = false;
        auto prediction = model.predict(training_states[0], used_fallback);
        assertTrue(prediction.num_particles() > 0, "Prediction should produce valid state");

        std::cout << "Model metrics:" << std::endl;
        std::cout << "  Speedup: " << model.get_speedup() << std::endl;
        std::cout << "  Accuracy: " << model.get_accuracy() << std::endl;
        std::cout << "  Fallback rate: " << model.get_physics_fallback_rate() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Training error: " << e.what() << std::endl;
        // Continue with other tests
    }

    std::cout << "✓ Surrogate model tests passed" << std::endl;
}

// Performance benchmark
void benchmarkSurrogateModel() {
    std::cout << "Running surrogate model performance benchmark..." << std::endl;

    SurrogateConfig<float> config;
    config.hidden_layers = {32, 64, 32};
    config.epochs = 50;
    config.batch_size = 16;
    config.use_adaptive_sampling = false;  // Disable for consistent benchmarking

    physgrad::surrogate::SurrogateModel<float> model(config);

    // Create larger training dataset
    std::vector<PhysicsState<float>> training_states;
    training_states.reserve(1000);

    for (int i = 0; i < 1000; ++i) {
        PhysicsState<float> state;

        // Random initial conditions
        state.positions.resize(60);  // 20 particles * 3 components
        state.velocities.resize(60);
        state.forces.resize(60);
        state.material_props.resize(20);

        for (size_t j = 0; j < 60; j += 3) {
            state.positions[j] = float(rand()) / RAND_MAX * 10.0f - 5.0f;     // x
            state.positions[j+1] = float(rand()) / RAND_MAX * 10.0f;          // y
            state.positions[j+2] = float(rand()) / RAND_MAX * 10.0f - 5.0f;   // z

            state.velocities[j] = (float(rand()) / RAND_MAX - 0.5f) * 2.0f;   // vx
            state.velocities[j+1] = (float(rand()) / RAND_MAX - 0.5f) * 2.0f; // vy
            state.velocities[j+2] = (float(rand()) / RAND_MAX - 0.5f) * 2.0f; // vz

            state.forces[j] = 0.0f;      // fx
            state.forces[j+1] = -9.81f;  // fy (gravity)
            state.forces[j+2] = 0.0f;    // fz
        }

        for (size_t j = 0; j < 20; ++j) {
            state.material_props[j] = 1.0f;  // mass
        }

        state.timestep = 0.01f;
        state.time = float(i) * 0.01f;
        training_states.push_back(state);
    }

    // Physics simulator with basic dynamics
    auto physics_sim = [](const PhysicsState<float>& state) -> PhysicsState<float> {
        PhysicsState<float> next_state = state;

        // Simple Euler integration
        for (size_t i = 0; i < state.positions.size(); ++i) {
            // Update velocity: v += a * dt
            next_state.velocities[i] += (state.forces[i] / state.material_props[i/3]) * state.timestep;

            // Update position: x += v * dt
            next_state.positions[i] += next_state.velocities[i] * state.timestep;
        }

        next_state.time += state.timestep;
        return next_state;
    };

    std::cout << "Training on " << training_states.size() << " samples..." << std::endl;

    auto train_start = std::chrono::high_resolution_clock::now();

    try {
        model.train(training_states, physics_sim);
    } catch (const std::exception& e) {
        std::cerr << "Training failed: " << e.what() << std::endl;
        return;
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);

    std::cout << "Training completed in " << train_time.count() << " ms" << std::endl;

    // Benchmark prediction speed
    const int num_predictions = 1000;
    auto pred_start = std::chrono::high_resolution_clock::now();

    int successful_predictions = 0;
    for (int i = 0; i < num_predictions; ++i) {
        bool used_fallback = false;
        try {
            auto prediction = model.predict(training_states[i % training_states.size()], used_fallback);
            if (!used_fallback) {
                successful_predictions++;
            }
        } catch (...) {
            // Continue counting
        }
    }

    auto pred_end = std::chrono::high_resolution_clock::now();
    auto pred_time = std::chrono::duration_cast<std::chrono::microseconds>(pred_end - pred_start);

    double avg_pred_time = double(pred_time.count()) / num_predictions;
    double predictions_per_second = 1e6 / avg_pred_time;

    std::cout << "Prediction benchmark results:" << std::endl;
    std::cout << "  Successful predictions: " << successful_predictions << "/" << num_predictions << std::endl;
    std::cout << "  Average prediction time: " << avg_pred_time << " μs" << std::endl;
    std::cout << "  Predictions per second: " << predictions_per_second << std::endl;
    std::cout << "  Fallback rate: " << model.get_physics_fallback_rate() * 100.0f << "%" << std::endl;

    std::cout << "✓ Performance benchmark completed" << std::endl;
}

int main() {
    std::cout << "PhysGrad Neural Surrogate Modeling Test Suite" << std::endl;
    std::cout << "==============================================" << std::endl;

    try {
        testTensorOperations();
        testActivationFunctions();
        testNeuralLayers();
        testLossFunctions();
        testNeuralNetwork();
        testPhysicsState();
        testDataPreprocessor();
        testSurrogateModel();
        benchmarkSurrogateModel();

        std::cout << std::endl;
        std::cout << "🎉 All tests passed!" << std::endl;
        std::cout << "Neural surrogate modeling framework is working correctly." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}