/**
 * PhysGrad Gradient Verification Utilities
 *
 * Provides numerical gradient computation and verification tools
 * to validate analytical gradients from adjoint methods.
 */

#pragma once

#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>

namespace physgrad {
namespace gradient_verification {

/**
 * Numerical gradient computation using finite differences
 */
template<typename T>
class NumericalGradient {
public:
    /**
     * Compute gradient using central differences
     * f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
     *
     * More accurate than forward/backward differences (O(h²) vs O(h))
     */
    static T centralDifference(
        std::function<T(T)> func,
        T x,
        T epsilon = static_cast<T>(1e-5)
    ) {
        T f_plus = func(x + epsilon);
        T f_minus = func(x - epsilon);
        return (f_plus - f_minus) / (static_cast<T>(2) * epsilon);
    }

    /**
     * Compute gradient of vector-valued function w.r.t. scalar
     */
    static std::vector<T> vectorGradient(
        std::function<std::vector<T>(T)> func,
        T x,
        T epsilon = static_cast<T>(1e-5)
    ) {
        auto f_plus = func(x + epsilon);
        auto f_minus = func(x - epsilon);

        std::vector<T> gradient(f_plus.size());
        for (size_t i = 0; i < f_plus.size(); ++i) {
            gradient[i] = (f_plus[i] - f_minus[i]) / (static_cast<T>(2) * epsilon);
        }
        return gradient;
    }

    /**
     * Compute gradient of scalar function w.r.t. vector
     */
    static std::vector<T> scalarGradient(
        std::function<T(const std::vector<T>&)> func,
        const std::vector<T>& x,
        T epsilon = static_cast<T>(1e-5)
    ) {
        std::vector<T> gradient(x.size());

        for (size_t i = 0; i < x.size(); ++i) {
            // Perturb dimension i
            std::vector<T> x_plus = x;
            std::vector<T> x_minus = x;
            x_plus[i] += epsilon;
            x_minus[i] -= epsilon;

            T f_plus = func(x_plus);
            T f_minus = func(x_minus);

            gradient[i] = (f_plus - f_minus) / (static_cast<T>(2) * epsilon);
        }

        return gradient;
    }

    /**
     * Compute Jacobian matrix for vector-to-vector function
     * J[i,j] = ∂f_i/∂x_j
     */
    static std::vector<std::vector<T>> jacobian(
        std::function<std::vector<T>(const std::vector<T>&)> func,
        const std::vector<T>& x,
        T epsilon = static_cast<T>(1e-5)
    ) {
        size_t output_dim = func(x).size();
        size_t input_dim = x.size();

        std::vector<std::vector<T>> J(output_dim, std::vector<T>(input_dim));

        for (size_t j = 0; j < input_dim; ++j) {
            // Perturb input dimension j
            std::vector<T> x_plus = x;
            std::vector<T> x_minus = x;
            x_plus[j] += epsilon;
            x_minus[j] -= epsilon;

            auto f_plus = func(x_plus);
            auto f_minus = func(x_minus);

            // Compute partial derivatives for all outputs
            for (size_t i = 0; i < output_dim; ++i) {
                J[i][j] = (f_plus[i] - f_minus[i]) / (static_cast<T>(2) * epsilon);
            }
        }

        return J;
    }
};

/**
 * Gradient comparison and error metrics
 */
template<typename T>
class GradientComparison {
public:
    struct ComparisonResult {
        T max_absolute_error;
        T mean_absolute_error;
        T max_relative_error;
        T mean_relative_error;
        size_t num_elements;
        bool passed;
        std::string error_message;
    };

    /**
     * Compare analytical vs numerical gradients
     */
    static ComparisonResult compare(
        const std::vector<T>& analytical,
        const std::vector<T>& numerical,
        T abs_tolerance = static_cast<T>(1e-5),
        T rel_tolerance = static_cast<T>(1e-3)
    ) {
        ComparisonResult result;
        result.num_elements = analytical.size();

        if (analytical.size() != numerical.size()) {
            result.passed = false;
            result.error_message = "Gradient size mismatch: " +
                                  std::to_string(analytical.size()) + " vs " +
                                  std::to_string(numerical.size());
            return result;
        }

        result.max_absolute_error = 0;
        result.mean_absolute_error = 0;
        result.max_relative_error = 0;
        result.mean_relative_error = 0;

        for (size_t i = 0; i < analytical.size(); ++i) {
            T abs_error = std::abs(analytical[i] - numerical[i]);
            T rel_error = 0;

            // Compute relative error carefully to avoid division by zero
            T denominator = std::max(std::abs(analytical[i]), std::abs(numerical[i]));
            if (denominator > static_cast<T>(1e-10)) {
                rel_error = abs_error / denominator;
            }

            result.max_absolute_error = std::max(result.max_absolute_error, abs_error);
            result.mean_absolute_error += abs_error;
            result.max_relative_error = std::max(result.max_relative_error, rel_error);
            result.mean_relative_error += rel_error;
        }

        result.mean_absolute_error /= analytical.size();
        result.mean_relative_error /= analytical.size();

        // Check tolerances
        result.passed = (result.max_absolute_error < abs_tolerance) &&
                       (result.max_relative_error < rel_tolerance);

        if (!result.passed) {
            std::ostringstream oss;
            oss << "Gradient check failed:\n"
                << "  Max absolute error: " << result.max_absolute_error
                << " (tolerance: " << abs_tolerance << ")\n"
                << "  Max relative error: " << result.max_relative_error
                << " (tolerance: " << rel_tolerance << ")\n"
                << "  Mean absolute error: " << result.mean_absolute_error << "\n"
                << "  Mean relative error: " << result.mean_relative_error;
            result.error_message = oss.str();
        }

        return result;
    }

    /**
     * Print comparison results
     */
    static void printResults(const ComparisonResult& result, const std::string& test_name) {
        std::cout << "=== Gradient Check: " << test_name << " ===" << std::endl;
        std::cout << "  Elements: " << result.num_elements << std::endl;
        std::cout << "  Max absolute error: " << result.max_absolute_error << std::endl;
        std::cout << "  Mean absolute error: " << result.mean_absolute_error << std::endl;
        std::cout << "  Max relative error: " << result.max_relative_error << std::endl;
        std::cout << "  Mean relative error: " << result.mean_relative_error << std::endl;
        std::cout << "  Status: " << (result.passed ? "PASSED ✓" : "FAILED ✗") << std::endl;

        if (!result.passed && !result.error_message.empty()) {
            std::cout << "  " << result.error_message << std::endl;
        }
        std::cout << std::endl;
    }
};

/**
 * Physics-specific gradient verification utilities
 */
template<typename T>
class PhysicsGradientChecker {
public:
    /**
     * Verify gradients for time integration
     *
     * Forward: x_new = x + v*dt + 0.5*a*dt²
     * Backward: ∂L/∂x, ∂L/∂v, ∂L/∂a
     */
    static typename GradientComparison<T>::ComparisonResult checkTimeIntegration(
        const std::vector<T>& positions,
        const std::vector<T>& velocities,
        const std::vector<T>& accelerations,
        T dt,
        const std::vector<T>& grad_output  // ∂L/∂x_new
    ) {
        // Numerical gradient computation
        auto integrate = [&](const std::vector<T>& x) -> std::vector<T> {
            std::vector<T> x_new(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                x_new[i] = x[i] + velocities[i] * dt +
                          static_cast<T>(0.5) * accelerations[i] * dt * dt;
            }
            return x_new;
        };

        auto numerical_grad = NumericalGradient<T>::jacobian(integrate, positions);

        // Analytical gradient: ∂x_new/∂x = I (identity)
        std::vector<T> analytical_grad_flat;
        for (size_t i = 0; i < positions.size(); ++i) {
            analytical_grad_flat.push_back(grad_output[i]); // Chain rule: ∂L/∂x = ∂L/∂x_new * I
        }

        // Flatten numerical gradient
        std::vector<T> numerical_grad_flat;
        for (size_t i = 0; i < numerical_grad.size(); ++i) {
            for (size_t j = 0; j < numerical_grad[i].size(); ++j) {
                if (i == j) {
                    numerical_grad_flat.push_back(numerical_grad[i][j] * grad_output[i]);
                }
            }
        }

        return GradientComparison<T>::compare(analytical_grad_flat, numerical_grad_flat);
    }

    /**
     * Verify gradients for force computation
     *
     * F = -∇U(x) where U is potential energy
     */
    static typename GradientComparison<T>::ComparisonResult checkForceGradients(
        std::function<T(const std::vector<T>&)> potential,
        std::function<std::vector<T>(const std::vector<T>&)> force_func,
        const std::vector<T>& positions,
        T epsilon = static_cast<T>(1e-3),
        T abs_tolerance = static_cast<T>(5e-2),
        T rel_tolerance = static_cast<T>(5e-2)
    ) {
        // Numerical gradient: F = -∇U
        auto numerical_force = NumericalGradient<T>::scalarGradient(potential, positions, epsilon);
        for (auto& f : numerical_force) {
            f = -f;  // Force is negative gradient of potential
        }

        // Analytical gradient from force function
        auto analytical_force = force_func(positions);

        return GradientComparison<T>::compare(analytical_force, numerical_force, abs_tolerance, rel_tolerance);
    }

    /**
     * Verify energy conservation properties
     *
     * For symplectic integrators, energy should be bounded over long timescales
     */
    static T measureEnergyDrift(
        std::function<T(const std::vector<T>&, const std::vector<T>&)> energy_func,
        std::vector<T>& positions,
        std::vector<T>& velocities,
        std::function<void(std::vector<T>&, std::vector<T>&, T)> integrator,
        T dt,
        int num_steps
    ) {
        T initial_energy = energy_func(positions, velocities);

        for (int step = 0; step < num_steps; ++step) {
            integrator(positions, velocities, dt);
        }

        T final_energy = energy_func(positions, velocities);

        return std::abs(final_energy - initial_energy) / std::abs(initial_energy);
    }
};

} // namespace gradient_verification
} // namespace physgrad
