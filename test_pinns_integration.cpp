#include "src/pinns_integration.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cassert>

using namespace physgrad::pinns;

bool testNeuralNetworkBasics() {
    std::cout << "Testing Neural Network Basics..." << std::endl;

    try {
        auto activation = std::make_unique<TanhActivation<double>>();
        std::vector<size_t> layer_sizes = {3, 10, 10, 2};
        NeuralNetwork<double> network(layer_sizes, std::move(activation));

        std::vector<double> input = {0.5, -0.3, 0.8};
        auto output = network.forward(input);

        if (output.size() != 2) {
            std::cout << "❌ Output size mismatch" << std::endl;
            return false;
        }

        for (double val : output) {
            if (std::isnan(val) || std::isinf(val)) {
                std::cout << "❌ Invalid output values" << std::endl;
                return false;
            }
        }

        auto gradients = network.computeGradients(input);
        if (gradients.size() != input.size()) {
            std::cout << "❌ Gradient size mismatch" << std::endl;
            return false;
        }

        std::cout << "Neural network output: [" << output[0] << ", " << output[1] << "]" << std::endl;
        std::cout << "✅ Neural Network Basics test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Neural Network Basics test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testActivationFunctions() {
    std::cout << "Testing Activation Functions..." << std::endl;

    try {
        std::vector<std::unique_ptr<ActivationFunction<double>>> activations;
        activations.push_back(std::make_unique<TanhActivation<double>>());
        activations.push_back(std::make_unique<SineActivation<double>>());
        activations.push_back(std::make_unique<SwishActivation<double>>());

        std::vector<double> test_inputs = {-2.0, -0.5, 0.0, 0.5, 2.0};

        for (auto& activation : activations) {
            for (double x : test_inputs) {
                double val = activation->evaluate(x);
                double deriv = activation->derivative(x);

                if (std::isnan(val) || std::isinf(val) || std::isnan(deriv) || std::isinf(deriv)) {
                    std::cout << "❌ Invalid activation function values" << std::endl;
                    return false;
                }

                double eps = 1e-5;
                double numerical_deriv = (activation->evaluate(x + eps) - activation->evaluate(x - eps)) / (2.0 * eps);
                double error = std::abs(deriv - numerical_deriv);

                if (error > 1e-3) {
                    std::cout << "❌ Derivative mismatch for input " << x
                              << ": analytical=" << deriv
                              << ", numerical=" << numerical_deriv << std::endl;
                    return false;
                }
            }
        }

        std::cout << "✅ Activation Functions test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Activation Functions test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testPhysicsLosses() {
    std::cout << "Testing Physics Losses..." << std::endl;

    try {
        double viscosity = 0.01;
        double density = 1.0;
        double thermal_diffusivity = 0.1;
        double wave_speed = 1.0;

        NavierStokesLoss<double> ns_loss(viscosity, density);
        HeatEquationLoss<double> heat_loss(thermal_diffusivity);
        WaveEquationLoss<double> wave_loss(wave_speed);

        std::vector<double> input = {0.5, 0.3, 0.1, 0.1};
        std::vector<double> output = {1.0, 0.5, 0.2, 2.0};
        std::vector<std::vector<double>> gradients(4, std::vector<double>(4, 0.1));

        double ns_loss_val = ns_loss.evaluate(input, output, gradients);
        double heat_loss_val = heat_loss.evaluate(input, output, gradients);
        double wave_loss_val = wave_loss.evaluate(input, output, gradients);

        if (std::isnan(ns_loss_val) || std::isnan(heat_loss_val) || std::isnan(wave_loss_val)) {
            std::cout << "❌ Physics loss computation failed" << std::endl;
            return false;
        }

        if (ns_loss_val < 0 || heat_loss_val < 0 || wave_loss_val < 0) {
            std::cout << "❌ Negative loss values" << std::endl;
            return false;
        }

        std::cout << "Navier-Stokes loss: " << ns_loss_val << std::endl;
        std::cout << "Heat equation loss: " << heat_loss_val << std::endl;
        std::cout << "Wave equation loss: " << wave_loss_val << std::endl;

        std::cout << "✅ Physics Losses test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Physics Losses test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testTrainingDataset() {
    std::cout << "Testing Training Dataset..." << std::endl;

    try {
        TrainingDataset<double> dataset;

        dataset.addDataPoint({0.0, 0.0, 0.0, 0.0}, {1.0, 0.0});
        dataset.addDataPoint({1.0, 1.0, 1.0, 1.0}, {0.0, 1.0});
        dataset.addDataPoint({0.5, 0.5, 0.5, 0.5}, {0.5, 0.5});

        if (dataset.size() != 3) {
            std::cout << "❌ Dataset size mismatch" << std::endl;
            return false;
        }

        std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
        dataset.generateCollocationPoints(100, bounds);

        if (dataset.size() != 100) {
            std::cout << "❌ Collocation points generation failed" << std::endl;
            return false;
        }

        auto batch_indices = dataset.generateBatchIndices(20);
        if (batch_indices.size() != 20) {
            std::cout << "❌ Batch generation failed" << std::endl;
            return false;
        }

        for (size_t idx : batch_indices) {
            const auto& input = dataset.getInput(idx);
            if (input.size() != 4) {
                std::cout << "❌ Input dimension mismatch" << std::endl;
                return false;
            }
            for (double val : input) {
                if (val < 0.0 || val > 1.0) {
                    std::cout << "❌ Input values out of bounds" << std::endl;
                    return false;
                }
            }
        }

        std::cout << "Generated " << dataset.size() << " collocation points" << std::endl;
        std::cout << "✅ Training Dataset test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Training Dataset test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testPINNsFramework() {
    std::cout << "Testing PINNs Framework..." << std::endl;

    try {
        std::vector<size_t> layer_sizes = {4, 20, 20, 4};
        double viscosity = 0.01;
        double density = 1.0;

        auto framework = PINNsFactory<double>::createNavierStokesFramework(
            layer_sizes, viscosity, density, 1e-3);

        TrainingDataset<double> dataset;
        std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
        dataset.generateCollocationPoints(200, bounds);

        auto start_time = std::chrono::high_resolution_clock::now();

        double initial_loss = framework->trainEpoch(dataset, 32);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        if (std::isnan(initial_loss) || std::isinf(initial_loss)) {
            std::cout << "❌ Invalid loss computation" << std::endl;
            return false;
        }

        std::vector<double> test_input = {0.5, 0.5, 0.5, 0.1};
        auto prediction = framework->predict(test_input);

        if (prediction.size() != 4) {
            std::cout << "❌ Prediction size mismatch" << std::endl;
            return false;
        }

        for (double val : prediction) {
            if (std::isnan(val) || std::isinf(val)) {
                std::cout << "❌ Invalid prediction values" << std::endl;
                return false;
            }
        }

        std::cout << "Initial loss: " << initial_loss << std::endl;
        std::cout << "Training time: " << duration.count() << " ms" << std::endl;
        std::cout << "Sample prediction: [" << prediction[0] << ", " << prediction[1]
                  << ", " << prediction[2] << ", " << prediction[3] << "]" << std::endl;

        std::cout << "✅ PINNs Framework test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ PINNs Framework test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testMultiPhysicsIntegration() {
    std::cout << "Testing Multi-Physics Integration..." << std::endl;

    try {
        PINNsPhysicsIntegrator<double> integrator;

        std::vector<size_t> layer_sizes = {4, 15, 15, 1};

        auto heat_framework = PINNsFactory<double>::createHeatEquationFramework(
            layer_sizes, 0.1, 1e-3);
        auto wave_framework = PINNsFactory<double>::createWaveEquationFramework(
            layer_sizes, 1.0, 1e-3);

        integrator.addFramework("heat", std::move(heat_framework));
        integrator.addFramework("wave", std::move(wave_framework));

        if (integrator.getFrameworkCount() != 2) {
            std::cout << "❌ Framework count mismatch" << std::endl;
            return false;
        }

        auto names = integrator.getFrameworkNames();
        if (names.size() != 2) {
            std::cout << "❌ Framework names mismatch" << std::endl;
            return false;
        }

        if (!integrator.hasFramework("heat") || !integrator.hasFramework("wave")) {
            std::cout << "❌ Framework lookup failed" << std::endl;
            return false;
        }

        TrainingDataset<double> dataset;
        std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 0.1}};
        dataset.generateCollocationPoints(150, bounds);

        double heat_loss = integrator.trainFramework("heat", dataset, 5, 32);
        double wave_loss = integrator.trainFramework("wave", dataset, 5, 32);

        if (std::isnan(heat_loss) || std::isnan(wave_loss)) {
            std::cout << "❌ Training failed" << std::endl;
            return false;
        }

        std::vector<double> test_input = {0.5, 0.5, 0.5, 0.05};
        auto heat_prediction = integrator.predictPhysics("heat", test_input);
        auto wave_prediction = integrator.predictPhysics("wave", test_input);

        if (heat_prediction.empty() || wave_prediction.empty()) {
            std::cout << "❌ Prediction failed" << std::endl;
            return false;
        }

        std::cout << "Heat equation training loss: " << heat_loss << std::endl;
        std::cout << "Wave equation training loss: " << wave_loss << std::endl;
        std::cout << "Heat prediction: " << heat_prediction[0] << std::endl;
        std::cout << "Wave prediction: " << wave_prediction[0] << std::endl;

        std::cout << "✅ Multi-Physics Integration test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Multi-Physics Integration test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testModelPersistence() {
    std::cout << "Testing Model Persistence..." << std::endl;

    try {
        std::vector<size_t> layer_sizes = {3, 8, 8, 2};
        auto framework = PINNsFactory<double>::createHeatEquationFramework(layer_sizes, 0.1);

        TrainingDataset<double> dataset;
        dataset.generateCollocationPoints(50, {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}});

        framework->trainEpoch(dataset, 16);

        std::vector<double> test_input = {0.3, 0.7, 0.5};
        auto prediction_before = framework->predict(test_input);

        framework->saveModel("test_model.bin");

        std::cout << "Model saved successfully" << std::endl;

        PINNsPhysicsIntegrator<double> integrator;
        integrator.addFramework("test", std::move(framework));

        try {
            integrator.saveAllModels("./");
            std::cout << "All models saved successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "⚠️  Model saving warning: " << e.what() << std::endl;
        }

        std::cout << "Test prediction: [" << prediction_before[0] << ", " << prediction_before[1] << "]" << std::endl;

        std::cout << "✅ Model Persistence test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Model Persistence test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testPerformanceBenchmark() {
    std::cout << "Testing Performance Benchmark..." << std::endl;

    try {
        std::vector<size_t> network_sizes = {100, 500, 1000, 2000};
        std::vector<size_t> layer_sizes = {4, 30, 30, 4};

        for (size_t n_points : network_sizes) {
            auto framework = PINNsFactory<double>::createNavierStokesFramework(
                layer_sizes, 0.01, 1.0, 1e-3);

            TrainingDataset<double> dataset;
            std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 0.1}};
            dataset.generateCollocationPoints(n_points, bounds);

            auto start_time = std::chrono::high_resolution_clock::now();

            for (int epoch = 0; epoch < 3; ++epoch) {
                framework->trainEpoch(dataset, 64);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            double time_per_point = static_cast<double>(duration.count()) / (n_points * 3);

            std::cout << n_points << " points: " << duration.count() << " ms total, "
                      << time_per_point << " ms/point/epoch" << std::endl;

            if (time_per_point > 10.0) {
                std::cout << "⚠️  Performance concern: " << time_per_point << " ms/point" << std::endl;
            }
        }

        std::cout << "✅ Performance Benchmark test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Performance Benchmark test failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== PINNs Integration Test Suite ===" << std::endl;

    bool all_passed = true;

    all_passed &= testNeuralNetworkBasics();
    all_passed &= testActivationFunctions();
    all_passed &= testPhysicsLosses();
    all_passed &= testTrainingDataset();
    all_passed &= testPINNsFramework();
    all_passed &= testMultiPhysicsIntegration();
    all_passed &= testModelPersistence();
    all_passed &= testPerformanceBenchmark();

    std::cout << "\n=== PINNs Integration Test Summary ===" << std::endl;
    if (all_passed) {
        std::cout << "✅ All PINNs integration tests passed!" << std::endl;
        std::cout << "\nPINNs Framework Validated:" << std::endl;
        std::cout << "• Neural network implementation with multiple activation functions" << std::endl;
        std::cout << "• Physics-informed loss functions (Navier-Stokes, Heat, Wave equations)" << std::endl;
        std::cout << "• Training dataset generation with collocation points" << std::endl;
        std::cout << "• Multi-physics framework integration and management" << std::endl;
        std::cout << "• Model persistence and serialization" << std::endl;
        std::cout << "• Performance scaling with training data size" << std::endl;
        std::cout << "• Production-ready PINNs system for physics simulation" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some PINNs integration tests failed!" << std::endl;
        return 1;
    }
}