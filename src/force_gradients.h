/**
 * PhysGrad Force Gradient Computation
 *
 * Comprehensive force gradient computation for geometric integrators
 * providing analytical gradients for various physics systems
 */

#pragma once

#include "common_types.h"
#include "differentiable_forces.h"
#include <vector>
#include <functional>
#include <memory>
#include <array>
#include <cmath>

namespace physgrad::gradients {

// =============================================================================
// FORCE GRADIENT INTERFACES
// =============================================================================

/**
 * Force gradient computation result structure
 */
template<typename T>
struct ForceGradientResult {
    // Force Jacobian: ∂F_i/∂r_j where F_i is force on particle i, r_j is position of particle j
    std::vector<std::vector<std::array<T, 9>>> force_jacobian; // 3x3 tensor per particle pair

    // Simplified gradient storage for compatibility with existing integrators
    std::vector<std::vector<T>> grad_fx_dx, grad_fx_dy, grad_fx_dz; // ∂F_x/∂(x,y,z)
    std::vector<std::vector<T>> grad_fy_dx, grad_fy_dy, grad_fy_dz; // ∂F_y/∂(x,y,z)
    std::vector<std::vector<T>> grad_fz_dx, grad_fz_dy, grad_fz_dz; // ∂F_z/∂(x,y,z)

    // Total computational cost metrics
    size_t gradient_evaluations = 0;
    T computational_cost = T{0};

    ForceGradientResult() = default;

    void resize(size_t num_particles) {
        force_jacobian.resize(num_particles);
        for (auto& row : force_jacobian) {
            row.resize(num_particles);
        }

        grad_fx_dx.resize(num_particles); grad_fx_dy.resize(num_particles); grad_fx_dz.resize(num_particles);
        grad_fy_dx.resize(num_particles); grad_fy_dy.resize(num_particles); grad_fy_dz.resize(num_particles);
        grad_fz_dx.resize(num_particles); grad_fz_dy.resize(num_particles); grad_fz_dz.resize(num_particles);

        for (size_t i = 0; i < num_particles; ++i) {
            grad_fx_dx[i].resize(num_particles); grad_fx_dy[i].resize(num_particles); grad_fx_dz[i].resize(num_particles);
            grad_fy_dx[i].resize(num_particles); grad_fy_dy[i].resize(num_particles); grad_fy_dz[i].resize(num_particles);
            grad_fz_dx[i].resize(num_particles); grad_fz_dy[i].resize(num_particles); grad_fz_dz[i].resize(num_particles);
        }
    }

    void clear() {
        for (auto& row : force_jacobian) {
            for (auto& tensor : row) {
                std::fill(tensor.begin(), tensor.end(), T{0});
            }
        }

        auto clear_matrix = [](auto& matrix) {
            for (auto& row : matrix) {
                std::fill(row.begin(), row.end(), T{0});
            }
        };

        clear_matrix(grad_fx_dx); clear_matrix(grad_fx_dy); clear_matrix(grad_fx_dz);
        clear_matrix(grad_fy_dx); clear_matrix(grad_fy_dy); clear_matrix(grad_fy_dz);
        clear_matrix(grad_fz_dx); clear_matrix(grad_fz_dy); clear_matrix(grad_fz_dz);

        gradient_evaluations = 0;
        computational_cost = T{0};
    }
};

/**
 * Abstract base class for force gradient computations
 */
template<typename T>
class ForceGradientEngine {
public:
    using scalar_type = T;
    using vector_type = std::array<T, 3>;
    using result_type = ForceGradientResult<T>;

    virtual ~ForceGradientEngine() = default;

    /**
     * Compute force gradients for given particle configuration
     */
    virtual result_type computeForceGradients(
        const std::vector<vector_type>& positions,
        const std::vector<T>& masses
    ) = 0;

    /**
     * Compute forces and gradients simultaneously for efficiency
     */
    virtual std::pair<std::vector<vector_type>, result_type> computeForcesAndGradients(
        const std::vector<vector_type>& positions,
        const std::vector<T>& masses
    ) = 0;

    /**
     * Check if this engine supports analytical gradients
     */
    virtual bool hasAnalyticalGradients() const = 0;

    /**
     * Get computational complexity estimate
     */
    virtual T getComputationalComplexity(size_t num_particles) const = 0;
};

// =============================================================================
// ANALYTICAL FORCE GRADIENT IMPLEMENTATIONS
// =============================================================================

/**
 * Gravitational force gradients with analytical computation
 * F_ij = G * m_i * m_j * (r_j - r_i) / |r_j - r_i|^3
 * ∂F_ij/∂r_k = analytical derivative
 */
template<typename T>
class GravitationalForceGradients : public ForceGradientEngine<T> {
private:
    T G_;         // Gravitational constant
    T softening_; // Softening parameter for numerical stability

public:
    GravitationalForceGradients(T G = T{1}, T softening = T{0.01})
        : G_(G), softening_(softening) {}

    typename ForceGradientEngine<T>::result_type computeForceGradients(
        const std::vector<typename ForceGradientEngine<T>::vector_type>& positions,
        const std::vector<T>& masses) override {

        size_t n = positions.size();
        typename ForceGradientEngine<T>::result_type result;
        result.resize(n);
        result.clear();

        // Compute analytical gradients for all particle pairs
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;

                // Position difference vector
                T dx = positions[j][0] - positions[i][0];
                T dy = positions[j][1] - positions[i][1];
                T dz = positions[j][2] - positions[i][2];

                // Distance with softening
                T r2 = dx*dx + dy*dy + dz*dz + softening_*softening_;
                T r = std::sqrt(r2);
                T r3 = r2 * r;
                T r5 = r2 * r3;

                // Force prefactor
                T force_prefactor = G_ * masses[i] * masses[j] / r3;

                // Gradient prefactor for ∂F_i/∂r_j
                T grad_prefactor = G_ * masses[i] * masses[j] / r5;

                // Analytical gradient tensor components
                // For gravitational force: F_i = G * m_i * m_j * (r_j - r_i) / |r_j - r_i|^3
                // ∂F_i/∂r_j = G * m_i * m_j * [I/r^3 - 3*(r_j-r_i)⊗(r_j-r_i)/r^5]
                T grad_fx_dx_val = grad_prefactor * (1 - 3*dx*dx/r2);
                T grad_fx_dy_val = grad_prefactor * (-3*dx*dy/r2);
                T grad_fx_dz_val = grad_prefactor * (-3*dx*dz/r2);

                // ∂F_iy/∂x_j, ∂F_iy/∂y_j, ∂F_iy/∂z_j
                T grad_fy_dx_val = grad_prefactor * (-3*dy*dx/r2);
                T grad_fy_dy_val = grad_prefactor * (1 - 3*dy*dy/r2);
                T grad_fy_dz_val = grad_prefactor * (-3*dy*dz/r2);

                // ∂F_iz/∂x_j, ∂F_iz/∂y_j, ∂F_iz/∂z_j
                T grad_fz_dx_val = grad_prefactor * (-3*dz*dx/r2);
                T grad_fz_dy_val = grad_prefactor * (-3*dz*dy/r2);
                T grad_fz_dz_val = grad_prefactor * (1 - 3*dz*dz/r2);

                // Store in simplified gradient matrices
                result.grad_fx_dx[i][j] = grad_fx_dx_val;
                result.grad_fx_dy[i][j] = grad_fx_dy_val;
                result.grad_fx_dz[i][j] = grad_fx_dz_val;

                result.grad_fy_dx[i][j] = grad_fy_dx_val;
                result.grad_fy_dy[i][j] = grad_fy_dy_val;
                result.grad_fy_dz[i][j] = grad_fy_dz_val;

                result.grad_fz_dx[i][j] = grad_fz_dx_val;
                result.grad_fz_dy[i][j] = grad_fz_dy_val;
                result.grad_fz_dz[i][j] = grad_fz_dz_val;

                // Store in full Jacobian tensor
                result.force_jacobian[i][j] = {
                    grad_fx_dx_val, grad_fx_dy_val, grad_fx_dz_val,
                    grad_fy_dx_val, grad_fy_dy_val, grad_fy_dz_val,
                    grad_fz_dx_val, grad_fz_dy_val, grad_fz_dz_val
                };

                result.gradient_evaluations++;
            }
        }

        // Self-interaction terms (negative sum of all other interactions)
        for (size_t i = 0; i < n; ++i) {
            T sum_fx_dx = 0, sum_fx_dy = 0, sum_fx_dz = 0;
            T sum_fy_dx = 0, sum_fy_dy = 0, sum_fy_dz = 0;
            T sum_fz_dx = 0, sum_fz_dy = 0, sum_fz_dz = 0;

            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    sum_fx_dx += result.grad_fx_dx[i][j];
                    sum_fx_dy += result.grad_fx_dy[i][j];
                    sum_fx_dz += result.grad_fx_dz[i][j];

                    sum_fy_dx += result.grad_fy_dx[i][j];
                    sum_fy_dy += result.grad_fy_dy[i][j];
                    sum_fy_dz += result.grad_fy_dz[i][j];

                    sum_fz_dx += result.grad_fz_dx[i][j];
                    sum_fz_dy += result.grad_fz_dy[i][j];
                    sum_fz_dz += result.grad_fz_dz[i][j];
                }
            }

            // Self-interaction gradients (Newton's third law)
            result.grad_fx_dx[i][i] = -sum_fx_dx;
            result.grad_fx_dy[i][i] = -sum_fx_dy;
            result.grad_fx_dz[i][i] = -sum_fx_dz;

            result.grad_fy_dx[i][i] = -sum_fy_dx;
            result.grad_fy_dy[i][i] = -sum_fy_dy;
            result.grad_fy_dz[i][i] = -sum_fy_dz;

            result.grad_fz_dx[i][i] = -sum_fz_dx;
            result.grad_fz_dy[i][i] = -sum_fz_dy;
            result.grad_fz_dz[i][i] = -sum_fz_dz;

            result.force_jacobian[i][i] = {
                -sum_fx_dx, -sum_fx_dy, -sum_fx_dz,
                -sum_fy_dx, -sum_fy_dy, -sum_fy_dz,
                -sum_fz_dx, -sum_fz_dy, -sum_fz_dz
            };
        }

        result.computational_cost = T(n * n);
        return result;
    }

    std::pair<std::vector<typename ForceGradientEngine<T>::vector_type>,
              typename ForceGradientEngine<T>::result_type> computeForcesAndGradients(
        const std::vector<typename ForceGradientEngine<T>::vector_type>& positions,
        const std::vector<T>& masses) override {

        size_t n = positions.size();
        std::vector<typename ForceGradientEngine<T>::vector_type> forces(n, {T{0}, T{0}, T{0}});

        // Compute forces
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;

                T dx = positions[j][0] - positions[i][0];
                T dy = positions[j][1] - positions[i][1];
                T dz = positions[j][2] - positions[i][2];

                T r2 = dx*dx + dy*dy + dz*dz + softening_*softening_;
                T r = std::sqrt(r2);
                T r3 = r2 * r;

                T force_magnitude = G_ * masses[i] * masses[j] / r3;

                forces[i][0] += force_magnitude * dx;
                forces[i][1] += force_magnitude * dy;
                forces[i][2] += force_magnitude * dz;
            }
        }

        auto gradients = computeForceGradients(positions, masses);
        return {forces, gradients};
    }

    bool hasAnalyticalGradients() const override { return true; }

    T getComputationalComplexity(size_t num_particles) const override {
        return T(num_particles * num_particles); // O(N^2)
    }
};

/**
 * Harmonic oscillator force gradients
 * F_i = -k * (r_i - r_center)
 * ∂F_i/∂r_j = -k * δ_ij * I (where I is identity matrix)
 */
template<typename T>
class HarmonicOscillatorForceGradients : public ForceGradientEngine<T> {
private:
    T k_;           // Spring constant
    std::array<T, 3> center_; // Equilibrium center

public:
    HarmonicOscillatorForceGradients(T k = T{1}, const std::array<T, 3>& center = {T{0}, T{0}, T{0}})
        : k_(k), center_(center) {}

    typename ForceGradientEngine<T>::result_type computeForceGradients(
        const std::vector<typename ForceGradientEngine<T>::vector_type>& positions,
        const std::vector<T>& masses) override {

        size_t n = positions.size();
        typename ForceGradientEngine<T>::result_type result;
        result.resize(n);
        result.clear();

        // Harmonic oscillator has simple diagonal gradients
        for (size_t i = 0; i < n; ++i) {
            // ∂F_i/∂r_i = -k * I (identity matrix)
            result.grad_fx_dx[i][i] = -k_;
            result.grad_fy_dy[i][i] = -k_;
            result.grad_fz_dz[i][i] = -k_;

            // All other components are zero
            result.force_jacobian[i][i] = {
                -k_, T{0}, T{0},
                T{0}, -k_, T{0},
                T{0}, T{0}, -k_
            };
        }

        result.gradient_evaluations = n;
        result.computational_cost = T(n); // O(N)
        return result;
    }

    std::pair<std::vector<typename ForceGradientEngine<T>::vector_type>,
              typename ForceGradientEngine<T>::result_type> computeForcesAndGradients(
        const std::vector<typename ForceGradientEngine<T>::vector_type>& positions,
        const std::vector<T>& masses) override {

        size_t n = positions.size();
        std::vector<typename ForceGradientEngine<T>::vector_type> forces(n);

        // Compute harmonic forces
        for (size_t i = 0; i < n; ++i) {
            forces[i][0] = -k_ * (positions[i][0] - center_[0]);
            forces[i][1] = -k_ * (positions[i][1] - center_[1]);
            forces[i][2] = -k_ * (positions[i][2] - center_[2]);
        }

        auto gradients = computeForceGradients(positions, masses);
        return {forces, gradients};
    }

    bool hasAnalyticalGradients() const override { return true; }

    T getComputationalComplexity(size_t num_particles) const override {
        return T(num_particles); // O(N)
    }
};

/**
 * Spring system force gradients for connected particles
 * F_ij = -k * (|r_j - r_i| - L_0) * (r_j - r_i) / |r_j - r_i|
 */
template<typename T>
class SpringSystemForceGradients : public ForceGradientEngine<T> {
private:
    std::vector<std::pair<size_t, size_t>> connections_;
    std::vector<T> spring_constants_;
    std::vector<T> rest_lengths_;

public:
    SpringSystemForceGradients(
        const std::vector<std::pair<size_t, size_t>>& connections,
        const std::vector<T>& spring_constants,
        const std::vector<T>& rest_lengths)
        : connections_(connections), spring_constants_(spring_constants), rest_lengths_(rest_lengths) {}

    typename ForceGradientEngine<T>::result_type computeForceGradients(
        const std::vector<typename ForceGradientEngine<T>::vector_type>& positions,
        const std::vector<T>& masses) override {

        size_t n = positions.size();
        typename ForceGradientEngine<T>::result_type result;
        result.resize(n);
        result.clear();

        // Process each spring connection
        for (size_t spring_idx = 0; spring_idx < connections_.size(); ++spring_idx) {
            size_t i = connections_[spring_idx].first;
            size_t j = connections_[spring_idx].second;

            T k = spring_constants_[spring_idx];
            T L0 = rest_lengths_[spring_idx];

            // Spring vector
            T dx = positions[j][0] - positions[i][0];
            T dy = positions[j][1] - positions[i][1];
            T dz = positions[j][2] - positions[i][2];

            T r = std::sqrt(dx*dx + dy*dy + dz*dz);
            T r_inv = T{1} / r;
            T r3_inv = r_inv * r_inv * r_inv;

            // Spring force magnitude
            T force_mag = k * (r - L0);

            // Gradient components for spring force
            T common_factor = k * L0 * r3_inv;

            // ∂F_i/∂r_j terms (force on i due to displacement of j)
            T grad_factor_ij = k * r_inv - common_factor;
            T grad_factor_ii = -grad_factor_ij - k * r_inv;

            // Update gradients for particle i
            result.grad_fx_dx[i][j] += grad_factor_ij + common_factor * dx * dx * r_inv;
            result.grad_fx_dy[i][j] += common_factor * dx * dy * r_inv;
            result.grad_fx_dz[i][j] += common_factor * dx * dz * r_inv;

            result.grad_fy_dx[i][j] += common_factor * dy * dx * r_inv;
            result.grad_fy_dy[i][j] += grad_factor_ij + common_factor * dy * dy * r_inv;
            result.grad_fy_dz[i][j] += common_factor * dy * dz * r_inv;

            result.grad_fz_dx[i][j] += common_factor * dz * dx * r_inv;
            result.grad_fz_dy[i][j] += common_factor * dz * dy * r_inv;
            result.grad_fz_dz[i][j] += grad_factor_ij + common_factor * dz * dz * r_inv;

            // Self-interaction terms (force on i due to displacement of i)
            result.grad_fx_dx[i][i] -= grad_factor_ij + common_factor * dx * dx * r_inv;
            result.grad_fx_dy[i][i] -= common_factor * dx * dy * r_inv;
            result.grad_fx_dz[i][i] -= common_factor * dx * dz * r_inv;

            result.grad_fy_dx[i][i] -= common_factor * dy * dx * r_inv;
            result.grad_fy_dy[i][i] -= grad_factor_ij + common_factor * dy * dy * r_inv;
            result.grad_fy_dz[i][i] -= common_factor * dy * dz * r_inv;

            result.grad_fz_dx[i][i] -= common_factor * dz * dx * r_inv;
            result.grad_fz_dy[i][i] -= common_factor * dz * dy * r_inv;
            result.grad_fz_dz[i][i] -= grad_factor_ij + common_factor * dz * dz * r_inv;

            // Symmetric terms for particle j (Newton's third law)
            result.grad_fx_dx[j][i] -= grad_factor_ij + common_factor * dx * dx * r_inv;
            result.grad_fx_dy[j][i] -= common_factor * dx * dy * r_inv;
            result.grad_fx_dz[j][i] -= common_factor * dx * dz * r_inv;

            result.grad_fy_dx[j][i] -= common_factor * dy * dx * r_inv;
            result.grad_fy_dy[j][i] -= grad_factor_ij + common_factor * dy * dy * r_inv;
            result.grad_fy_dz[j][i] -= common_factor * dy * dz * r_inv;

            result.grad_fz_dx[j][i] -= common_factor * dz * dx * r_inv;
            result.grad_fz_dy[j][i] -= common_factor * dz * dy * r_inv;
            result.grad_fz_dz[j][i] -= grad_factor_ij + common_factor * dz * dz * r_inv;

            result.grad_fx_dx[j][j] += grad_factor_ij + common_factor * dx * dx * r_inv;
            result.grad_fx_dy[j][j] += common_factor * dx * dy * r_inv;
            result.grad_fx_dz[j][j] += common_factor * dx * dz * r_inv;

            result.grad_fy_dx[j][j] += common_factor * dy * dx * r_inv;
            result.grad_fy_dy[j][j] += grad_factor_ij + common_factor * dy * dy * r_inv;
            result.grad_fy_dz[j][j] += common_factor * dy * dz * r_inv;

            result.grad_fz_dx[j][j] += common_factor * dz * dx * r_inv;
            result.grad_fz_dy[j][j] += common_factor * dz * dy * r_inv;
            result.grad_fz_dz[j][j] += grad_factor_ij + common_factor * dz * dz * r_inv;

            result.gradient_evaluations++;
        }

        result.computational_cost = T(connections_.size());
        return result;
    }

    std::pair<std::vector<typename ForceGradientEngine<T>::vector_type>,
              typename ForceGradientEngine<T>::result_type> computeForcesAndGradients(
        const std::vector<typename ForceGradientEngine<T>::vector_type>& positions,
        const std::vector<T>& masses) override {

        size_t n = positions.size();
        std::vector<typename ForceGradientEngine<T>::vector_type> forces(n, {T{0}, T{0}, T{0}});

        // Compute spring forces
        for (size_t spring_idx = 0; spring_idx < connections_.size(); ++spring_idx) {
            size_t i = connections_[spring_idx].first;
            size_t j = connections_[spring_idx].second;

            T k = spring_constants_[spring_idx];
            T L0 = rest_lengths_[spring_idx];

            T dx = positions[j][0] - positions[i][0];
            T dy = positions[j][1] - positions[i][1];
            T dz = positions[j][2] - positions[i][2];

            T r = std::sqrt(dx*dx + dy*dy + dz*dz);
            T force_mag = k * (r - L0) / r;

            T fx = force_mag * dx;
            T fy = force_mag * dy;
            T fz = force_mag * dz;

            forces[i][0] += fx;
            forces[i][1] += fy;
            forces[i][2] += fz;

            forces[j][0] -= fx;
            forces[j][1] -= fy;
            forces[j][2] -= fz;
        }

        auto gradients = computeForceGradients(positions, masses);
        return {forces, gradients};
    }

    bool hasAnalyticalGradients() const override { return true; }

    T getComputationalComplexity(size_t num_particles) const override {
        return T(connections_.size()); // O(M) where M is number of springs
    }
};

// =============================================================================
// UTILITY FUNCTIONS FOR INTEGRATOR COMPATIBILITY
// =============================================================================

/**
 * Convert force gradient result to legacy format for symplectic integrators
 */
template<typename T>
std::function<void(
    const std::vector<T>&, const std::vector<T>&, const std::vector<T>&,
    const std::vector<T>&,
    std::vector<std::vector<T>>&, std::vector<std::vector<T>>&, std::vector<std::vector<T>>&
)> createLegacyForceGradientFunction(std::shared_ptr<ForceGradientEngine<T>> engine) {

    return [engine](
        const std::vector<T>& pos_x, const std::vector<T>& pos_y, const std::vector<T>& pos_z,
        const std::vector<T>& masses,
        std::vector<std::vector<T>>& grad_fx_dx, std::vector<std::vector<T>>& grad_fx_dy, std::vector<std::vector<T>>& grad_fx_dz
    ) {
        size_t n = pos_x.size();

        // Convert to modern format
        std::vector<std::array<T, 3>> positions(n);
        for (size_t i = 0; i < n; ++i) {
            positions[i] = {pos_x[i], pos_y[i], pos_z[i]};
        }

        // Compute gradients
        auto result = engine->computeForceGradients(positions, masses);

        // Convert back to legacy format
        grad_fx_dx = result.grad_fx_dx;
        grad_fx_dy = result.grad_fx_dy;
        grad_fx_dz = result.grad_fx_dz;
    };
}

/**
 * Create gravitational force gradient function
 */
template<typename T>
std::shared_ptr<ForceGradientEngine<T>> createGravitationalForceGradientEngine(T G = T{1}, T softening = T{0.01}) {
    return std::make_shared<GravitationalForceGradients<T>>(G, softening);
}

/**
 * Create harmonic oscillator force gradient function
 */
template<typename T>
std::shared_ptr<ForceGradientEngine<T>> createHarmonicOscillatorForceGradientEngine(
    T k = T{1}, const std::array<T, 3>& center = {T{0}, T{0}, T{0}}) {
    return std::make_shared<HarmonicOscillatorForceGradients<T>>(k, center);
}

/**
 * Create spring system force gradient function
 */
template<typename T>
std::shared_ptr<ForceGradientEngine<T>> createSpringSystemForceGradientEngine(
    const std::vector<std::pair<size_t, size_t>>& connections,
    const std::vector<T>& spring_constants,
    const std::vector<T>& rest_lengths) {
    return std::make_shared<SpringSystemForceGradients<T>>(connections, spring_constants, rest_lengths);
}

} // namespace physgrad::gradients