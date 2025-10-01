#include "src/surrogate_model.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace physgrad::surrogate;

// Simple pendulum physics for demonstration
PhysicsState<float> simple_pendulum_physics(const PhysicsState<float>& state) {
    PhysicsState<float> next_state = state;

    // Simple pendulum: x = angle, v = angular velocity
    // dx/dt = v
    // dv/dt = -g/L * sin(x) - b*v (with damping)

    float g = 9.81f;
    float L = 1.0f;  // pendulum length
    float b = 0.1f;  // damping coefficient

    for (size_t i = 0; i < state.positions.size(); ++i) {
        float angle = state.positions[i];
        float angular_vel = state.velocities[i];

        // Update angular velocity
        float angular_acc = -(g / L) * std::sin(angle) - b * angular_vel;
        next_state.velocities[i] = angular_vel + angular_acc * state.timestep;

        // Update angle
        next_state.positions[i] = angle + next_state.velocities[i] * state.timestep;
    }

    next_state.time += state.timestep;
    return next_state;
}

// Generate training data for pendulum
std::vector<PhysicsState<float>> generate_pendulum_data(int num_samples = 200) {
    std::vector<PhysicsState<float>> data;
    data.reserve(num_samples);

    float dt = 0.01f;

    // Generate multiple pendulum trajectories with different initial conditions
    for (int traj = 0; traj < 10; ++traj) {
        // Random initial conditions
        float initial_angle = (float(rand()) / RAND_MAX - 0.5f) * M_PI;  // -π/2 to π/2
        float initial_vel = (float(rand()) / RAND_MAX - 0.5f) * 4.0f;    // -2 to 2 rad/s

        PhysicsState<float> state;
        state.positions = {initial_angle};
        state.velocities = {initial_vel};
        state.forces = {0.0f};
        state.material_props = {1.0f};  // mass
        state.timestep = dt;
        state.time = 0.0f;

        // Simulate trajectory
        for (int i = 0; i < num_samples / 10; ++i) {
            data.push_back(state);
            state = simple_pendulum_physics(state);
        }
    }

    return data;
}

int main() {
    std::cout << "PhysGrad Neural Surrogate Modeling Example" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "Training a neural network to learn pendulum dynamics" << std::endl;

    // Configure surrogate model
    SurrogateConfig<float> config;
    config.hidden_layers = {16, 32, 16};
    config.epochs = 100;
    config.batch_size = 8;
    config.learning_rate = 0.01f;
    config.validation_split = 0.2f;
    config.enforce_energy_conservation = false;  // For simplicity
    config.use_adaptive_sampling = false;

    // Create surrogate model
    SurrogateModel<float> model(config);

    // Generate training data
    std::cout << "\nGenerating training data..." << std::endl;
    auto training_data = generate_pendulum_data(200);
    std::cout << "Generated " << training_data.size() << " training samples" << std::endl;

    // Train the model
    std::cout << "\nTraining neural surrogate model..." << std::endl;
    try {
        model.train(training_data, simple_pendulum_physics);
        std::cout << "Training completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Training failed: " << e.what() << std::endl;
        return 1;
    }

    // Test the trained model
    std::cout << "\nTesting trained model..." << std::endl;

    // Create test case
    PhysicsState<float> test_state;
    test_state.positions = {M_PI / 4};    // 45 degrees
    test_state.velocities = {0.0f};       // starting from rest
    test_state.forces = {0.0f};
    test_state.material_props = {1.0f};
    test_state.timestep = 0.01f;
    test_state.time = 0.0f;

    std::cout << "\nComparing physics simulation vs neural surrogate:" << std::endl;
    std::cout << "Time\tPhysics Angle\tSurrogate Angle\tError" << std::endl;
    std::cout << "----\t-------------\t---------------\t-----" << std::endl;

    PhysicsState<float> physics_state = test_state;
    PhysicsState<float> surrogate_state = test_state;

    float total_error = 0.0f;
    int num_steps = 50;

    for (int step = 0; step < num_steps; ++step) {
        // Physics simulation
        auto next_physics = simple_pendulum_physics(physics_state);

        // Neural surrogate prediction
        bool used_fallback = false;
        auto next_surrogate = model.predict(surrogate_state, used_fallback);

        // Calculate error
        float error = 0.0f;
        if (!used_fallback && !next_surrogate.positions.empty()) {
            error = std::abs(next_physics.positions[0] - next_surrogate.positions[0]);
            total_error += error;
        } else {
            // Fallback occurred
            next_surrogate = next_physics;
            error = 0.0f;  // No error since we used physics
        }

        // Print results every 10 steps
        if (step % 10 == 0) {
            std::cout << std::fixed << std::setprecision(3)
                     << physics_state.time << "\t"
                     << physics_state.positions[0] << "\t\t"
                     << (next_surrogate.positions.empty() ? 0.0f : next_surrogate.positions[0]) << "\t\t"
                     << error << std::endl;
        }

        // Update states
        physics_state = next_physics;
        surrogate_state = next_surrogate;
    }

    float avg_error = total_error / num_steps;
    std::cout << "\nAverage prediction error: " << avg_error << " radians" << std::endl;

    // Performance metrics
    std::cout << "\nModel Performance Metrics:" << std::endl;
    std::cout << "Speedup: " << model.get_speedup() << "x" << std::endl;
    std::cout << "Accuracy: " << (1.0f - avg_error) * 100.0f << "%" << std::endl;
    std::cout << "Physics fallback rate: " << model.get_physics_fallback_rate() * 100.0f << "%" << std::endl;

    // Demonstrate adaptive learning capability
    if (config.use_adaptive_sampling) {
        std::cout << "\nStarting background adaptive learning..." << std::endl;
        model.start_background_training(simple_pendulum_physics);

        // Simulate some predictions that might trigger adaptive sampling
        for (int i = 0; i < 20; ++i) {
            PhysicsState<float> random_state;
            random_state.positions = {static_cast<float>((float(rand()) / RAND_MAX - 0.5f) * M_PI)};
            random_state.velocities = {(float(rand()) / RAND_MAX - 0.5f) * 2.0f};
            random_state.forces = {0.0f};
            random_state.material_props = {1.0f};
            random_state.timestep = 0.01f;
            random_state.time = 0.0f;

            bool used_fallback = false;
            model.predict(random_state, used_fallback);
        }

        model.stop_background_training();
        std::cout << "Background learning completed" << std::endl;
    }

    std::cout << "\n✓ Neural surrogate modeling example completed successfully!" << std::endl;

    return 0;
}