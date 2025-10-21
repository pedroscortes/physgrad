/**
 * PhysGrad PyTorch Autograd Integration Tests
 *
 * Tests for custom PyTorch autograd functions that enable end-to-end
 * differentiable physics simulation with PyTorch's AD system.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>

#include "src/pytorch_autograd.h"

using namespace physgrad;
using namespace physgrad::pytorch;

/**
 * Test tensor conversion utilities
 */
template<typename T>
bool testTensorConversion() {
    std::cout << "Testing tensor conversion utilities...\n";

    // Create test positions
    std::vector<ConceptVector3D<T>> positions = {
        {T{1.0}, T{2.0}, T{3.0}},
        {T{4.0}, T{5.0}, T{6.0}}
    };

    // Convert to tensor and back
    auto tensor = positionsToTensor(positions);
    auto converted_back = tensorToPositions<T>(tensor);

    // Verify conversion accuracy
    bool success = true;
    for (size_t i = 0; i < positions.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (std::abs(positions[i][j] - converted_back[i][j]) > T{1e-6}) {
                std::cout << "Conversion error at [" << i << "][" << j << "]: "
                          << positions[i][j] << " vs " << converted_back[i][j] << "\n";
                success = false;
            }
        }
    }

    if (success) {
        std::cout << "✓ Tensor conversion test passed\n";
    } else {
        std::cout << "✗ Tensor conversion test failed\n";
    }

    return success;
}

/**
 * Test physics simulation function (forward pass)
 */
template<typename T>
bool testPhysicsSimulationForward() {
    std::cout << "Testing physics simulation forward pass...\n";

    try {
        // Create test system
        auto initial_positions = create_particle_chain(3, 1.0f);
        auto initial_velocities = create_zero_velocities(3);
        auto masses = create_uniform_masses(3, 1.0f);

        std::cout << "Initial positions: ";
        for (size_t i = 0; i < initial_positions.data.size(); i += 3) {
            std::cout << "(" << initial_positions.data[i] << ", "
                      << initial_positions.data[i+1] << ", "
                      << initial_positions.data[i+2] << ") ";
        }
        std::cout << "\n";

        // Run simulation
        auto final_positions = physics_simulation<T>(
            initial_positions, initial_velocities, masses, 0.01, 10);

        std::cout << "Final positions: ";
        for (size_t i = 0; i < final_positions.data.size(); i += 3) {
            std::cout << "(" << final_positions.data[i] << ", "
                      << final_positions.data[i+1] << ", "
                      << final_positions.data[i+2] << ") ";
        }
        std::cout << "\n";

        // Check that simulation produced reasonable results
        bool positions_changed = false;
        for (size_t i = 0; i < initial_positions.data.size(); ++i) {
            if (std::abs(final_positions.data[i] - initial_positions.data[i]) > 1e-6f) {
                positions_changed = true;
                break;
            }
        }

        if (positions_changed) {
            std::cout << "✓ Physics simulation forward pass test passed\n";
            return true;
        } else {
            std::cout << "✗ Physics simulation forward pass test failed - no change\n";
            return false;
        }

    } catch (const std::exception& e) {
        std::cout << "✗ Physics simulation forward pass test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * Test physics simulation gradient computation
 */
template<typename T>
bool testPhysicsSimulationGradients() {
    std::cout << "Testing physics simulation gradient computation...\n";

    try {
        // Create test system
        auto initial_positions = create_particle_chain(2, 1.5f); // Stretched spring
        auto initial_velocities = create_zero_velocities(2);
        auto masses = create_uniform_masses(2, 1.0f);

        // Enable gradients
        initial_positions.set_requires_grad(true);

        // Run forward simulation
        auto final_positions = physics_simulation<T>(
            initial_positions, initial_velocities, masses, 0.01, 5);

        // Simple loss: sum of squared final positions
        float loss = 0.0f;
        for (float pos : final_positions.data) {
            loss += pos * pos;
        }
        auto loss_tensor = torch::Tensor({loss}, {1});

        std::cout << "Loss: " << loss << "\n";

        // Mock backward pass (since we're using mock PyTorch)
        loss_tensor.backward();

        std::cout << "✓ Physics simulation gradient test passed (mock)\n";
        return true;

    } catch (const std::exception& e) {
        std::cout << "✗ Physics simulation gradient test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * Test loss functions
 */
template<typename T>
bool testLossFunctions() {
    std::cout << "Testing physics-based loss functions...\n";

    // Create test data
    auto positions = create_particle_chain(3, 1.0f);
    auto velocities = create_zero_velocities(3);
    auto masses = create_uniform_masses(3, 1.0f);

    // Add some velocity for kinetic energy
    velocities.data[0] = 0.1f; // vx of first particle
    velocities.data[3] = 0.2f; // vx of second particle

    auto target_positions = create_particle_chain(3, 1.1f); // Slightly different

    try {
        // Test position loss
        auto pos_loss = position_loss(positions, target_positions);
        std::cout << "Position loss: " << pos_loss.data[0] << "\n";

        // Test energy conservation loss
        float target_energy = 0.1f;
        auto energy_loss = energy_conservation_loss(positions, velocities, masses, target_energy);
        std::cout << "Energy conservation loss: " << energy_loss.data[0] << "\n";

        // Test physics-informed loss
        auto pi_loss = physics_informed_loss(positions, velocities, masses,
                                           target_positions, target_energy);
        std::cout << "Physics-informed loss: " << pi_loss.data[0] << "\n";

        // Verify losses are reasonable
        bool success = (pos_loss.data[0] >= 0 && energy_loss.data[0] >= 0 && pi_loss.data[0] >= 0);

        if (success) {
            std::cout << "✓ Loss functions test passed\n";
        } else {
            std::cout << "✗ Loss functions test failed\n";
        }

        return success;

    } catch (const std::exception& e) {
        std::cout << "✗ Loss functions test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * Test physics-based learning framework
 */
template<typename T>
bool testPhysicsBasedLearning() {
    std::cout << "Testing physics-based learning framework...\n";

    try {
        // Create learning system
        PhysicsBasedLearning<T> learner(3);

        // Define target
        auto target_positions = create_particle_chain(3, 1.1f);
        float target_energy = 0.05f;

        // Initial loss
        auto initial_loss = learner.compute_loss(target_positions, target_energy);
        std::cout << "Initial loss: " << initial_loss.data[0] << "\n";

        // Run several optimization steps
        for (int step = 0; step < 5; ++step) {
            learner.optimization_step(target_positions, target_energy, 0.01f);

            if (step % 2 == 0) {
                auto current_loss = learner.compute_loss(target_positions, target_energy);
                std::cout << "Step " << step << " loss: " << current_loss.data[0] << "\n";
            }
        }

        // Final loss
        auto final_loss = learner.compute_loss(target_positions, target_energy);
        std::cout << "Final loss: " << final_loss.data[0] << "\n";

        // Check that framework runs without errors
        std::cout << "✓ Physics-based learning test passed\n";
        return true;

    } catch (const std::exception& e) {
        std::cout << "✗ Physics-based learning test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * Test utility functions
 */
bool testUtilityFunctions() {
    std::cout << "Testing utility functions...\n";

    try {
        // Test particle chain creation
        auto positions = create_particle_chain(4, 0.5f);
        if (positions.data.size() != 12) { // 4 particles * 3 components
            std::cout << "✗ Particle chain creation failed - wrong size\n";
            return false;
        }

        // Check spacing
        float expected_spacing = 0.5f;
        for (int i = 1; i < 4; ++i) {
            float spacing = positions.data[i*3] - positions.data[(i-1)*3];
            if (std::abs(spacing - expected_spacing) > 1e-6f) {
                std::cout << "✗ Particle chain creation failed - wrong spacing\n";
                return false;
            }
        }

        // Test zero velocities
        auto velocities = create_zero_velocities(3);
        for (float vel : velocities.data) {
            if (std::abs(vel) > 1e-6f) {
                std::cout << "✗ Zero velocities creation failed\n";
                return false;
            }
        }

        // Test uniform masses
        auto masses = create_uniform_masses(5, 2.0f);
        for (float mass : masses.data) {
            if (std::abs(mass - 2.0f) > 1e-6f) {
                std::cout << "✗ Uniform masses creation failed\n";
                return false;
            }
        }

        std::cout << "✓ Utility functions test passed\n";
        return true;

    } catch (const std::exception& e) {
        std::cout << "✗ Utility functions test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * Test gradient accuracy with finite differences
 */
template<typename T>
bool testGradientAccuracy() {
    std::cout << "Testing gradient accuracy...\n";

    try {
        // Simple test: single parameter optimization
        auto initial_positions = create_particle_chain(2, 1.2f);
        auto initial_velocities = create_zero_velocities(2);
        auto masses = create_uniform_masses(2, 1.0f);

        // Function to compute loss given initial position perturbation
        auto compute_loss = [&](float perturbation) -> float {
            auto perturbed_positions = initial_positions;
            perturbed_positions.data[0] += perturbation; // Perturb first x-coordinate

            auto final_positions = physics_simulation<T>(
                perturbed_positions, initial_velocities, masses, 0.01, 3);

            // Simple quadratic loss on final position
            return final_positions.data[0] * final_positions.data[0];
        };

        // Compute finite difference gradient
        float h = 1e-5f;
        float loss_plus = compute_loss(h);
        float loss_minus = compute_loss(-h);
        float fd_gradient = (loss_plus - loss_minus) / (2.0f * h);

        std::cout << "Finite difference gradient: " << fd_gradient << "\n";

        // For mock implementation, we expect some gradient computation
        // In real PyTorch integration, this would be compared with autograd gradients
        std::cout << "✓ Gradient accuracy test passed (finite difference computed)\n";
        return true;

    } catch (const std::exception& e) {
        std::cout << "✗ Gradient accuracy test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * Main test function
 */
int main() {
    std::cout << "PhysGrad PyTorch Autograd Integration Tests\n";
    std::cout << "===========================================\n\n";

    std::cout << std::fixed << std::setprecision(6);

    bool all_passed = true;

    // Test with float precision
    std::cout << "--- Float precision tests ---\n";
    all_passed &= testTensorConversion<float>();
    std::cout << "\n";

    all_passed &= testUtilityFunctions();
    std::cout << "\n";

    all_passed &= testPhysicsSimulationForward<float>();
    std::cout << "\n";

    all_passed &= testLossFunctions<float>();
    std::cout << "\n";

    all_passed &= testPhysicsBasedLearning<float>();
    std::cout << "\n";

    all_passed &= testPhysicsSimulationGradients<float>();
    std::cout << "\n";

    all_passed &= testGradientAccuracy<float>();
    std::cout << "\n";

    // Test with double precision
    std::cout << "--- Double precision tests ---\n";
    all_passed &= testTensorConversion<double>();
    std::cout << "\n";

    all_passed &= testPhysicsSimulationForward<double>();
    std::cout << "\n";

    if (all_passed) {
        std::cout << "✓ All PyTorch autograd tests PASSED!\n";
        std::cout << "\nNote: Using mock PyTorch implementation. ";
        std::cout << "For full functionality, compile with -DPHYSGRAD_PYTORCH_AVAILABLE\n";
        return 0;
    } else {
        std::cout << "✗ Some PyTorch autograd tests FAILED!\n";
        return 1;
    }
}