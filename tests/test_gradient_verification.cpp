/**
 * PhysGrad Gradient Verification Tests
 *
 * Validates analytical gradients from adjoint methods against numerical gradients
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>

#include "gradient_verification.h"
#include "common_types.h"

using namespace physgrad;
using namespace physgrad::gradient_verification;

class GradientVerificationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set tolerances for gradient checks
        // For float precision (~7 significant digits), use larger epsilon
        epsilon_ = 1e-3f;  // Optimal for float finite differences
        abs_tol_ = 5e-2f;  // Absolute tolerance (5%)
        rel_tol_ = 5e-2f;  // Relative tolerance (5%)
    }

    float epsilon_;
    float abs_tol_;
    float rel_tol_;
};

// =============================================================================
// BASIC GRADIENT VERIFICATION TESTS
// =============================================================================

TEST_F(GradientVerificationTest, CentralDifferenceSimpleFunction) {
    // Test f(x) = x²
    // f'(x) = 2x
    auto func = [](float x) { return x * x; };

    float x = 3.0f;
    float numerical_grad = NumericalGradient<float>::centralDifference(func, x, epsilon_);
    float analytical_grad = 2.0f * x;

    EXPECT_NEAR(numerical_grad, analytical_grad, abs_tol_)
        << "Numerical gradient for f(x)=x² should match analytical gradient 2x";
}

TEST_F(GradientVerificationTest, CentralDifferenceTrigFunction) {
    // Test f(x) = sin(x)
    // f'(x) = cos(x)
    auto func = [](float x) { return std::sin(x); };

    float x = 1.5f;
    float numerical_grad = NumericalGradient<float>::centralDifference(func, x, epsilon_);
    float analytical_grad = std::cos(x);

    EXPECT_NEAR(numerical_grad, analytical_grad, abs_tol_)
        << "Numerical gradient for f(x)=sin(x) should match analytical gradient cos(x)";
}

TEST_F(GradientVerificationTest, VectorGradientComputation) {
    // Test f(x) = [x², x³]
    // f'(x) = [2x, 3x²]
    auto func = [](float x) -> std::vector<float> {
        return {x * x, x * x * x};
    };

    float x = 2.0f;
    auto numerical_grad = NumericalGradient<float>::vectorGradient(func, x, epsilon_);

    std::vector<float> analytical_grad = {2.0f * x, 3.0f * x * x};

    ASSERT_EQ(numerical_grad.size(), analytical_grad.size());
    for (size_t i = 0; i < numerical_grad.size(); ++i) {
        EXPECT_NEAR(numerical_grad[i], analytical_grad[i], abs_tol_);
    }
}

TEST_F(GradientVerificationTest, ScalarGradientVector) {
    // Test f(x, y, z) = x² + 2y² + 3z²
    // ∇f = [2x, 4y, 6z]
    auto func = [](const std::vector<float>& x) {
        return x[0] * x[0] + 2.0f * x[1] * x[1] + 3.0f * x[2] * x[2];
    };

    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    auto numerical_grad = NumericalGradient<float>::scalarGradient(func, x, epsilon_);

    std::vector<float> analytical_grad = {2.0f * x[0], 4.0f * x[1], 6.0f * x[2]};

    ASSERT_EQ(numerical_grad.size(), analytical_grad.size());
    for (size_t i = 0; i < numerical_grad.size(); ++i) {
        EXPECT_NEAR(numerical_grad[i], analytical_grad[i], abs_tol_);
    }
}

TEST_F(GradientVerificationTest, JacobianComputation) {
    // Test f(x, y) = [x*y, x²+y²]
    // J = [[y, x], [2x, 2y]]
    auto func = [](const std::vector<float>& x) -> std::vector<float> {
        return {x[0] * x[1], x[0] * x[0] + x[1] * x[1]};
    };

    std::vector<float> x = {2.0f, 3.0f};
    auto J_numerical = NumericalGradient<float>::jacobian(func, x, epsilon_);

    // Analytical Jacobian
    std::vector<std::vector<float>> J_analytical = {
        {x[1], x[0]},           // ∂[x*y]/∂[x,y] = [y, x]
        {2.0f * x[0], 2.0f * x[1]}  // ∂[x²+y²]/∂[x,y] = [2x, 2y]
    };

    ASSERT_EQ(J_numerical.size(), J_analytical.size());
    for (size_t i = 0; i < J_numerical.size(); ++i) {
        ASSERT_EQ(J_numerical[i].size(), J_analytical[i].size());
        for (size_t j = 0; j < J_numerical[i].size(); ++j) {
            EXPECT_NEAR(J_numerical[i][j], J_analytical[i][j], abs_tol_);
        }
    }
}

// =============================================================================
// PHYSICS GRADIENT VERIFICATION TESTS
// =============================================================================

TEST_F(GradientVerificationTest, VerletIntegrationGradient) {
    // Test Verlet integration: x_new = x + v*dt + 0.5*a*dt²
    std::vector<float> positions = {1.0f, 2.0f, 3.0f};
    std::vector<float> velocities = {0.1f, 0.2f, 0.3f};
    std::vector<float> accelerations = {-0.01f, -0.02f, -0.03f};
    float dt = 0.01f;

    // Forward pass
    auto integrate = [&](const std::vector<float>& x) -> std::vector<float> {
        std::vector<float> x_new(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x_new[i] = x[i] + velocities[i] * dt + 0.5f * accelerations[i] * dt * dt;
        }
        return x_new;
    };

    // Compute numerical Jacobian
    auto J = NumericalGradient<float>::jacobian(integrate, positions, epsilon_);

    // Analytical Jacobian: ∂x_new/∂x = I (identity matrix)
    for (size_t i = 0; i < J.size(); ++i) {
        for (size_t j = 0; j < J[i].size(); ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(J[i][j], expected, abs_tol_)
                << "Verlet position gradient should be identity at (" << i << "," << j << ")";
        }
    }
}

TEST_F(GradientVerificationTest, GravityForceGradient) {
    // Test gravitational force: F = -G * m1 * m2 / r²
    float G = 1.0f;
    float m1 = 1.0f;
    float m2 = 2.0f;

    auto force_magnitude = [=](const std::vector<float>& pos) -> float {
        float r = std::sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
        if (r < 1e-6f) return 0.0f;
        return -G * m1 * m2 / (r * r);
    };

    std::vector<float> position = {1.0f, 0.0f, 0.0f};
    auto numerical_grad = NumericalGradient<float>::scalarGradient(force_magnitude, position, epsilon_);

    // Analytical gradient: ∂F/∂x = 2*G*m1*m2*x / r⁴
    float r = std::sqrt(position[0]*position[0] + position[1]*position[1] + position[2]*position[2]);
    float r4 = r * r * r * r;

    std::vector<float> analytical_grad = {
        2.0f * G * m1 * m2 * position[0] / r4,
        2.0f * G * m1 * m2 * position[1] / r4,
        2.0f * G * m1 * m2 * position[2] / r4
    };

    auto result = GradientComparison<float>::compare(analytical_grad, numerical_grad, abs_tol_, rel_tol_);
    GradientComparison<float>::printResults(result, "Gravity Force");
    EXPECT_TRUE(result.passed) << result.error_message;
}

TEST_F(GradientVerificationTest, KineticEnergyGradient) {
    // Test kinetic energy: KE = 0.5 * m * v²
    // ∂KE/∂v = m * v
    float mass = 2.0f;

    auto kinetic_energy = [=](const std::vector<float>& vel) -> float {
        float v_squared = 0.0f;
        for (auto v : vel) {
            v_squared += v * v;
        }
        return 0.5f * mass * v_squared;
    };

    std::vector<float> velocity = {1.0f, 2.0f, 3.0f};
    auto numerical_grad = NumericalGradient<float>::scalarGradient(kinetic_energy, velocity, epsilon_);

    // Analytical gradient: ∂KE/∂v = m * v
    std::vector<float> analytical_grad(velocity.size());
    for (size_t i = 0; i < velocity.size(); ++i) {
        analytical_grad[i] = mass * velocity[i];
    }

    auto result = GradientComparison<float>::compare(analytical_grad, numerical_grad, abs_tol_, rel_tol_);
    GradientComparison<float>::printResults(result, "Kinetic Energy");
    EXPECT_TRUE(result.passed) << result.error_message;
}

TEST_F(GradientVerificationTest, HarmonicPotentialGradient) {
    // Test harmonic oscillator: U = 0.5 * k * x²
    // F = -∂U/∂x = -k * x
    float k = 10.0f;  // Spring constant

    auto potential = [=](const std::vector<float>& x) -> float {
        float energy = 0.0f;
        for (auto xi : x) {
            energy += 0.5f * k * xi * xi;
        }
        return energy;
    };

    auto force = [=](const std::vector<float>& x) -> std::vector<float> {
        std::vector<float> f(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            f[i] = -k * x[i];
        }
        return f;
    };

    std::vector<float> position = {0.5f, -0.3f, 0.8f};

    auto result = PhysicsGradientChecker<float>::checkForceGradients(potential, force, position);
    GradientComparison<float>::printResults(result, "Harmonic Potential");
    EXPECT_TRUE(result.passed) << result.error_message;
}

TEST_F(GradientVerificationTest, ElectrostaticForceGradient) {
    // Test Coulomb force: F = k_e * q1 * q2 / r²
    float k_e = 8.9875517923e9f;  // Coulomb constant
    float q1 = 1.0e-6f;  // 1 μC
    float q2 = -1.0e-6f; // -1 μC

    auto potential = [=](const std::vector<float>& x) -> float {
        float r = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        if (r < 1e-10f) return 0.0f;
        return k_e * q1 * q2 / r;
    };

    auto force = [=](const std::vector<float>& x) -> std::vector<float> {
        float r = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        if (r < 1e-10f) return {0.0f, 0.0f, 0.0f};

        float magnitude = k_e * q1 * q2 / (r * r * r);
        return {magnitude * x[0], magnitude * x[1], magnitude * x[2]};
    };

    std::vector<float> position = {0.1f, 0.0f, 0.0f};

    auto result = PhysicsGradientChecker<float>::checkForceGradients(potential, force, position);
    GradientComparison<float>::printResults(result, "Electrostatic Force");
    EXPECT_TRUE(result.passed) << result.error_message;
}

// =============================================================================
// CHAIN RULE VERIFICATION
// =============================================================================

TEST_F(GradientVerificationTest, ChainRuleVerification) {
    // Test chain rule: f(g(x)) where f(u) = u² and g(x) = 2x + 1
    // df/dx = df/du * du/dx = 2u * 2 = 4(2x + 1)

    auto g = [](float x) { return 2.0f * x + 1.0f; };
    auto f = [](float u) { return u * u; };
    auto composed = [&](float x) { return f(g(x)); };

    float x = 3.0f;

    // Numerical gradient
    float numerical_grad = NumericalGradient<float>::centralDifference(composed, x, epsilon_);

    // Analytical gradient via chain rule
    float u = g(x);
    float df_du = 2.0f * u;  // f'(u) = 2u
    float du_dx = 2.0f;       // g'(x) = 2
    float analytical_grad = df_du * du_dx;

    EXPECT_NEAR(numerical_grad, analytical_grad, abs_tol_)
        << "Chain rule gradient should match numerical gradient";
}

TEST_F(GradientVerificationTest, VectorChainRule) {
    // Test vector chain rule: f(g(x)) where f: R² → R and g: R → R²
    // g(x) = [x, x²]
    // f([u, v]) = u² + v²
    // df/dx = 2u * 1 + 2v * 2x = 2x + 4x³

    auto g = [](float x) -> std::vector<float> {
        return {x, x * x};
    };

    auto f = [](const std::vector<float>& u) -> float {
        return u[0] * u[0] + u[1] * u[1];
    };

    auto composed = [&](float x) -> float {
        return f(g(x));
    };

    float x = 2.0f;

    // Numerical gradient
    float numerical_grad = NumericalGradient<float>::centralDifference(composed, x, epsilon_);

    // Analytical gradient
    float analytical_grad = 2.0f * x + 4.0f * x * x * x;

    EXPECT_NEAR(numerical_grad, analytical_grad, abs_tol_)
        << "Vector chain rule gradient should match numerical gradient";
}

// =============================================================================
// GRADIENT COMPARISON UTILITIES TEST
// =============================================================================

TEST_F(GradientVerificationTest, GradientComparisonPass) {
    std::vector<float> analytical = {1.0f, 2.0f, 3.0f};
    std::vector<float> numerical = {1.00001f, 2.00001f, 3.00001f};

    auto result = GradientComparison<float>::compare(analytical, numerical, 1e-4f, 1e-3f);

    EXPECT_TRUE(result.passed);
    EXPECT_LT(result.max_absolute_error, 1e-4f);
}

TEST_F(GradientVerificationTest, GradientComparisonFail) {
    std::vector<float> analytical = {1.0f, 2.0f, 3.0f};
    std::vector<float> numerical = {1.1f, 2.1f, 3.1f};  // Large error

    auto result = GradientComparison<float>::compare(analytical, numerical, 1e-5f, 1e-3f);

    EXPECT_FALSE(result.passed);
    EXPECT_GT(result.max_absolute_error, 0.09f);
}

TEST_F(GradientVerificationTest, GradientComparisonSizeMismatch) {
    std::vector<float> analytical = {1.0f, 2.0f, 3.0f};
    std::vector<float> numerical = {1.0f, 2.0f};  // Wrong size

    auto result = GradientComparison<float>::compare(analytical, numerical);

    EXPECT_FALSE(result.passed);
    EXPECT_FALSE(result.error_message.empty());
    EXPECT_NE(result.error_message.find("size mismatch"), std::string::npos);
}

// =============================================================================
// INTEGRATION WITH PHYSICS ENGINE (Conceptual Tests)
// =============================================================================

TEST_F(GradientVerificationTest, ParticleTimeIntegrationGradient) {
    // Test full particle time integration with forces
    float dt = 0.01f;
    float mass = 1.0f;

    std::vector<float> position = {1.0f, 0.0f, 0.0f};
    std::vector<float> velocity = {0.0f, 1.0f, 0.0f};
    std::vector<float> force = {-0.1f, 0.0f, 0.0f};

    // Compute acceleration
    std::vector<float> acceleration(3);
    for (size_t i = 0; i < 3; ++i) {
        acceleration[i] = force[i] / mass;
    }

    // Forward integration function
    auto integrate_forward = [&](const std::vector<float>& x0) -> std::vector<float> {
        std::vector<float> x_new(3);
        for (size_t i = 0; i < 3; ++i) {
            x_new[i] = x0[i] + velocity[i] * dt + 0.5f * acceleration[i] * dt * dt;
        }
        return x_new;
    };

    // Check gradient
    auto J = NumericalGradient<float>::jacobian(integrate_forward, position, epsilon_);

    // For position-only integration (fixed velocity and acceleration),
    // Jacobian should be identity
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(J[i][j], expected, abs_tol_);
        }
    }

    std::cout << "Particle time integration gradient verified" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
