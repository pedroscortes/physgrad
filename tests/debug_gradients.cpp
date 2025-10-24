#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "src/force_gradients.h"

using namespace physgrad::gradients;

int main() {
    std::cout << "Debugging Force Gradients" << std::endl;
    std::cout << "=========================" << std::endl;

    // Simple two-particle system
    std::vector<std::array<float, 3>> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f}
    };
    std::vector<float> masses = {1.0f, 1.0f};

    float G = 1.0f;
    float softening = 0.01f;

    auto engine = createGravitationalForceGradientEngine<float>(G, softening);

    // Compute forces and gradients
    auto [forces, gradients] = engine->computeForcesAndGradients(positions, masses);

    std::cout << "System setup:" << std::endl;
    std::cout << "  Particle 0 at (0, 0, 0)" << std::endl;
    std::cout << "  Particle 1 at (1, 0, 0)" << std::endl;
    std::cout << "  G = " << G << ", softening = " << softening << std::endl;
    std::cout << "  Distance = 1.0, r² + softening² = " << (1.0f + softening*softening) << std::endl;

    std::cout << "\nForces computed:" << std::endl;
    std::cout << "  Force on particle 0: (" << forces[0][0] << ", " << forces[0][1] << ", " << forces[0][2] << ")" << std::endl;
    std::cout << "  Force on particle 1: (" << forces[1][0] << ", " << forces[1][1] << ", " << forces[1][2] << ")" << std::endl;

    // Manual force calculation for verification
    float dx = 1.0f - 0.0f;  // positions[1][0] - positions[0][0]
    float r2 = dx*dx + softening*softening;
    float r = std::sqrt(r2);
    float r3 = r2 * r;
    float force_mag = G * masses[0] * masses[1] / r3;
    float expected_fx0 = force_mag * dx;

    std::cout << "\nManual force calculation:" << std::endl;
    std::cout << "  dx = " << dx << std::endl;
    std::cout << "  r² = " << r2 << std::endl;
    std::cout << "  r = " << r << std::endl;
    std::cout << "  Force magnitude = G*m₁*m₂/r³ = " << force_mag << std::endl;
    std::cout << "  Expected F₀ₓ = " << expected_fx0 << std::endl;

    std::cout << "\nAnalytical gradients:" << std::endl;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            std::cout << "  ∂F" << i << "ₓ/∂x" << j << " = " << gradients.grad_fx_dx[i][j] << std::endl;
        }
    }

    // Manual gradient calculation
    float r5 = r3 * r2;
    float grad_prefactor = G * masses[0] * masses[1] / r5;

    // ∂F₀ₓ/∂x₁ = G*m₀*m₁/r⁵ * (1 - 3*dx²/r²)
    float manual_grad_01 = grad_prefactor * (1.0f - 3.0f*dx*dx/r2);
    // ∂F₀ₓ/∂x₀ = -∂F₀ₓ/∂x₁
    float manual_grad_00 = -manual_grad_01;

    std::cout << "\nManual gradient calculation:" << std::endl;
    std::cout << "  grad_prefactor = G*m₀*m₁/r⁵ = " << grad_prefactor << std::endl;
    std::cout << "  1 - 3*dx²/r² = " << (1.0f - 3.0f*dx*dx/r2) << std::endl;
    std::cout << "  Manual ∂F₀ₓ/∂x₁ = " << manual_grad_01 << std::endl;
    std::cout << "  Manual ∂F₀ₓ/∂x₀ = " << manual_grad_00 << std::endl;

    // Numerical gradient calculation
    const float eps = 1e-6f;

    auto pos_plus = positions;
    auto pos_minus = positions;
    pos_plus[1][0] += eps;   // Perturb x₁
    pos_minus[1][0] -= eps;

    auto [forces_plus, _] = engine->computeForcesAndGradients(pos_plus, masses);
    auto [forces_minus, __] = engine->computeForcesAndGradients(pos_minus, masses);

    float numerical_grad_01 = (forces_plus[0][0] - forces_minus[0][0]) / (2.0f * eps);

    pos_plus = positions;
    pos_minus = positions;
    pos_plus[0][0] += eps;   // Perturb x₀
    pos_minus[0][0] -= eps;

    auto [forces_plus2, ___] = engine->computeForcesAndGradients(pos_plus, masses);
    auto [forces_minus2, ____] = engine->computeForcesAndGradients(pos_minus, masses);

    float numerical_grad_00 = (forces_plus2[0][0] - forces_minus2[0][0]) / (2.0f * eps);

    std::cout << "\nNumerical gradients (ε = " << eps << "):" << std::endl;
    std::cout << "  Numerical ∂F₀ₓ/∂x₁ = " << numerical_grad_01 << std::endl;
    std::cout << "  Numerical ∂F₀ₓ/∂x₀ = " << numerical_grad_00 << std::endl;

    std::cout << "\nComparisons:" << std::endl;
    std::cout << "  ∂F₀ₓ/∂x₁: analytical=" << gradients.grad_fx_dx[0][1] << ", manual=" << manual_grad_01 << ", numerical=" << numerical_grad_01 << std::endl;
    std::cout << "  ∂F₀ₓ/∂x₀: analytical=" << gradients.grad_fx_dx[0][0] << ", manual=" << manual_grad_00 << ", numerical=" << numerical_grad_00 << std::endl;

    return 0;
}