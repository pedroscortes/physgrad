/**
 * PhysGrad Differentiable Forces
 *
 * Implementation of classical force kernels with backward pass support
 * for automatic differentiation in physics simulations.
 */

#pragma once

#include "common_types.h"
#include <array>
#include <vector>
#include <functional>
#include <memory>
#include <cmath>

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "concepts/forward_declarations.h"
#endif

namespace physgrad::differentiable {

// =============================================================================
// AUTOMATIC DIFFERENTIATION FRAMEWORK
// =============================================================================

/**
 * Dual number implementation for automatic differentiation
 */
template<typename T>
struct Dual {
    T value;
    T gradient;

    constexpr Dual() : value(T{0}), gradient(T{0}) {}
    constexpr Dual(T v) : value(v), gradient(T{0}) {}
    constexpr Dual(T v, T g) : value(v), gradient(g) {}

    // Arithmetic operations with chain rule
    constexpr Dual operator+(const Dual& other) const {
        return Dual(value + other.value, gradient + other.gradient);
    }

    constexpr Dual operator-(const Dual& other) const {
        return Dual(value - other.value, gradient - other.gradient);
    }

    constexpr Dual operator*(const Dual& other) const {
        return Dual(value * other.value,
                   gradient * other.value + value * other.gradient);
    }

    constexpr Dual operator/(const Dual& other) const {
        T inv = T{1} / other.value;
        return Dual(value * inv,
                   (gradient * other.value - value * other.gradient) * inv * inv);
    }

    constexpr Dual operator*(T scalar) const {
        return Dual(value * scalar, gradient * scalar);
    }

    constexpr Dual operator+(T scalar) const {
        return Dual(value + scalar, gradient);
    }

    constexpr Dual operator-(T scalar) const {
        return Dual(value - scalar, gradient);
    }
};

// Mathematical functions for dual numbers
template<typename T>
constexpr Dual<T> sqrt(const Dual<T>& x) {
    T sqrt_val = std::sqrt(x.value);
    return Dual<T>(sqrt_val, x.gradient / (T{2} * sqrt_val));
}

template<typename T>
constexpr Dual<T> pow(const Dual<T>& x, T n) {
    T pow_val = std::pow(x.value, n);
    return Dual<T>(pow_val, n * std::pow(x.value, n - T{1}) * x.gradient);
}

template<typename T>
constexpr Dual<T> sin(const Dual<T>& x) {
    return Dual<T>(std::sin(x.value), x.gradient * std::cos(x.value));
}

template<typename T>
constexpr Dual<T> cos(const Dual<T>& x) {
    return Dual<T>(std::cos(x.value), -x.gradient * std::sin(x.value));
}

// Vector operations for dual numbers
template<typename T>
using DualVector3 = std::array<Dual<T>, 3>;

template<typename T>
constexpr DualVector3<T> operator+(const DualVector3<T>& a, const DualVector3<T>& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

template<typename T>
constexpr DualVector3<T> operator-(const DualVector3<T>& a, const DualVector3<T>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

template<typename T>
constexpr DualVector3<T> operator*(const DualVector3<T>& v, const Dual<T>& s) {
    return {v[0] * s, v[1] * s, v[2] * s};
}

template<typename T>
constexpr Dual<T> dot(const DualVector3<T>& a, const DualVector3<T>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<typename T>
constexpr Dual<T> magnitude(const DualVector3<T>& v) {
    return sqrt(dot(v, v));
}

template<typename T>
constexpr DualVector3<T> normalize(const DualVector3<T>& v) {
    Dual<T> mag = magnitude(v);
    return {v[0] / mag, v[1] / mag, v[2] / mag};
}

// =============================================================================
// DIFFERENTIABLE FORCE IMPLEMENTATIONS
// =============================================================================

/**
 * Gravitational force with automatic differentiation
 */
template<typename T>
class DifferentiableGravity {
public:
    using scalar_type = T;
    using dual_type = Dual<T>;
    using vector_type = DualVector3<T>;

    DifferentiableGravity(T G = T{6.67430e-11}) : G_(G) {}

    /**
     * Compute gravitational force between two particles
     * F = -G * m1 * m2 * r_hat / r^2
     */
    std::pair<vector_type, vector_type> force_pair(
        const vector_type& r1, const dual_type& m1,
        const vector_type& r2, const dual_type& m2) const {

        vector_type r = r2 - r1;
        dual_type r_mag = magnitude(r);
        dual_type r_mag_inv = dual_type(T{1}) / r_mag;
        dual_type r_mag_inv3 = r_mag_inv * r_mag_inv * r_mag_inv;

        // Force magnitude: G * m1 * m2 / r^2
        dual_type force_mag = dual_type(G_) * m1 * m2 * r_mag_inv * r_mag_inv;

        // Force direction: r_hat
        vector_type r_hat = {r[0] * r_mag_inv, r[1] * r_mag_inv, r[2] * r_mag_inv};

        // Force on particle 1 (attractive towards particle 2)
        vector_type f1 = r_hat * force_mag;

        // Force on particle 2 (Newton's third law)
        vector_type f2 = {dual_type(T{0}) - f1[0],
                         dual_type(T{0}) - f1[1],
                         dual_type(T{0}) - f1[2]};

        return {f1, f2};
    }

    /**
     * Compute potential energy between two particles
     * U = -G * m1 * m2 / r
     */
    dual_type potential_pair(
        const vector_type& r1, const dual_type& m1,
        const vector_type& r2, const dual_type& m2) const {

        vector_type r = r2 - r1;
        dual_type r_mag = magnitude(r);

        return dual_type(-G_) * m1 * m2 / r_mag;
    }

private:
    T G_; // Gravitational constant
};

/**
 * Lennard-Jones potential with automatic differentiation
 * U(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
 * F(r) = -dU/dr
 */
template<typename T>
class DifferentiableLennardJones {
public:
    using scalar_type = T;
    using dual_type = Dual<T>;
    using vector_type = DualVector3<T>;

    DifferentiableLennardJones(T epsilon = T{1}, T sigma = T{1})
        : epsilon_(epsilon), sigma_(sigma) {}

    std::pair<vector_type, vector_type> force_pair(
        const vector_type& r1, const vector_type& r2) const {

        vector_type r = r2 - r1;
        dual_type r_mag = magnitude(r);
        dual_type r_mag_inv = dual_type(T{1}) / r_mag;

        // sigma/r
        dual_type sr = dual_type(sigma_) * r_mag_inv;
        dual_type sr6 = pow(sr, T{6});
        dual_type sr12 = sr6 * sr6;

        // Force magnitude: 24*epsilon * (2*sr^12 - sr^6) / r
        dual_type force_mag = dual_type(T{24} * epsilon_) * (dual_type(T{2}) * sr12 - sr6) * r_mag_inv;

        // Force direction
        vector_type r_hat = {r[0] * r_mag_inv, r[1] * r_mag_inv, r[2] * r_mag_inv};

        // Force on particle 1
        vector_type f1 = {dual_type(T{0}) - r_hat[0] * force_mag,
                         dual_type(T{0}) - r_hat[1] * force_mag,
                         dual_type(T{0}) - r_hat[2] * force_mag};

        // Force on particle 2
        vector_type f2 = r_hat * force_mag;

        return {f1, f2};
    }

    dual_type potential_pair(const vector_type& r1, const vector_type& r2) const {
        vector_type r = r2 - r1;
        dual_type r_mag = magnitude(r);
        dual_type r_mag_inv = dual_type(T{1}) / r_mag;

        dual_type sr = dual_type(sigma_) * r_mag_inv;
        dual_type sr6 = pow(sr, T{6});
        dual_type sr12 = sr6 * sr6;

        return dual_type(T{4} * epsilon_) * (sr12 - sr6);
    }

private:
    T epsilon_; // Well depth
    T sigma_;   // Collision diameter
};

/**
 * Coulomb electrostatic force with automatic differentiation
 * F = k * q1 * q2 * r_hat / r^2
 */
template<typename T>
class DifferentiableCoulomb {
public:
    using scalar_type = T;
    using dual_type = Dual<T>;
    using vector_type = DualVector3<T>;

    DifferentiableCoulomb(T k = COULOMB_CONSTANT) : k_(k) {}

    std::pair<vector_type, vector_type> force_pair(
        const vector_type& r1, const dual_type& q1,
        const vector_type& r2, const dual_type& q2) const {

        vector_type r = r2 - r1;
        dual_type r_mag = magnitude(r);
        dual_type r_mag_inv = dual_type(T{1}) / r_mag;
        dual_type r_mag_inv3 = r_mag_inv * r_mag_inv * r_mag_inv;

        // Force magnitude: k * q1 * q2 / r^2
        dual_type force_mag = dual_type(k_) * q1 * q2 * r_mag_inv * r_mag_inv;

        // Force direction
        vector_type r_hat = {r[0] * r_mag_inv, r[1] * r_mag_inv, r[2] * r_mag_inv};

        // Force on particle 1 (repulsive if same charge)
        vector_type f1 = r_hat * force_mag;

        // Force on particle 2
        vector_type f2 = {dual_type(T{0}) - f1[0],
                         dual_type(T{0}) - f1[1],
                         dual_type(T{0}) - f1[2]};

        return {f1, f2};
    }

    dual_type potential_pair(
        const vector_type& r1, const dual_type& q1,
        const vector_type& r2, const dual_type& q2) const {

        vector_type r = r2 - r1;
        dual_type r_mag = magnitude(r);

        return dual_type(k_) * q1 * q2 / r_mag;
    }

private:
    T k_; // Coulomb constant
};

/**
 * Harmonic spring force with automatic differentiation
 * F = -k * (r - r0) * r_hat
 */
template<typename T>
class DifferentiableHarmonicSpring {
public:
    using scalar_type = T;
    using dual_type = Dual<T>;
    using vector_type = DualVector3<T>;

    DifferentiableHarmonicSpring(T k = T{1}, T r0 = T{1})
        : k_(k), r0_(r0) {}

    std::pair<vector_type, vector_type> force_pair(
        const vector_type& r1, const vector_type& r2) const {

        vector_type r = r2 - r1;
        dual_type r_mag = magnitude(r);
        dual_type r_mag_inv = dual_type(T{1}) / r_mag;

        // Spring force magnitude: k * (r - r0)
        dual_type force_mag = dual_type(k_) * (r_mag - dual_type(r0_));

        // Force direction
        vector_type r_hat = {r[0] * r_mag_inv, r[1] * r_mag_inv, r[2] * r_mag_inv};

        // Force on particle 1 (towards equilibrium)
        vector_type f1 = r_hat * force_mag;

        // Force on particle 2
        vector_type f2 = {dual_type(T{0}) - f1[0],
                         dual_type(T{0}) - f1[1],
                         dual_type(T{0}) - f1[2]};

        return {f1, f2};
    }

    dual_type potential_pair(const vector_type& r1, const vector_type& r2) const {
        vector_type r = r2 - r1;
        dual_type r_mag = magnitude(r);
        dual_type dr = r_mag - dual_type(r0_);

        return dual_type(T{0.5} * k_) * dr * dr;
    }

private:
    T k_;  // Spring constant
    T r0_; // Equilibrium distance
};

// =============================================================================
// FORCE COMPUTATION ENGINE
// =============================================================================

/**
 * Differentiable force computation engine that can compute both
 * forces and their gradients simultaneously
 */
template<typename T>
class DifferentiableForceEngine {
public:
    using scalar_type = T;
    using dual_type = Dual<T>;
    using vector_type = DualVector3<T>;

    // Force computation result
    struct ForceResult {
        std::vector<std::array<T, 3>> forces;        // Forces in value space
        std::vector<std::array<T, 3>> force_gradients; // Gradients w.r.t. positions
        T total_potential;                           // Total potential energy
        std::vector<T> potential_gradients;         // Gradients w.r.t. positions
    };

    DifferentiableForceEngine() = default;

    /**
     * Compute forces and gradients for gravitational interactions
     */
    ForceResult compute_gravitational_forces(
        const std::vector<std::array<T, 3>>& positions,
        const std::vector<T>& masses) const {

        size_t n = positions.size();
        ForceResult result;
        result.forces.resize(n, {T{0}, T{0}, T{0}});
        result.force_gradients.resize(n, {T{0}, T{0}, T{0}});
        result.potential_gradients.resize(n * 3, T{0});
        result.total_potential = T{0};

        DifferentiableGravity<T> gravity;

        // Compute pairwise interactions
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                // Create dual numbers for automatic differentiation
                vector_type r1 = make_dual_vector(positions[i], i, n);
                vector_type r2 = make_dual_vector(positions[j], j, n);
                dual_type m1(masses[i]);
                dual_type m2(masses[j]);

                // Compute forces
                auto [f1, f2] = gravity.force_pair(r1, m1, r2, m2);

                // Extract values and gradients
                add_force_contribution(result.forces[i], f1);
                add_force_contribution(result.forces[j], f2);

                // Accumulate force gradients
                accumulate_force_gradients(result.force_gradients[i], f1, i, j);
                accumulate_force_gradients(result.force_gradients[j], f2, i, j);

                // Compute potential energy
                dual_type potential = gravity.potential_pair(r1, m1, r2, m2);
                result.total_potential += potential.value;

                // Accumulate potential gradients
                accumulate_potential_gradients(result.potential_gradients,
                                             potential, i, j, n);
            }
        }

        return result;
    }

    /**
     * Compute forces and gradients for Lennard-Jones interactions
     */
    ForceResult compute_lennard_jones_forces(
        const std::vector<std::array<T, 3>>& positions,
        T epsilon = T{1}, T sigma = T{1}) const {

        size_t n = positions.size();
        ForceResult result;
        result.forces.resize(n, {T{0}, T{0}, T{0}});
        result.force_gradients.resize(n, {T{0}, T{0}, T{0}});
        result.potential_gradients.resize(n * 3, T{0});
        result.total_potential = T{0};

        DifferentiableLennardJones<T> lj(epsilon, sigma);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                vector_type r1 = make_dual_vector(positions[i], i, n);
                vector_type r2 = make_dual_vector(positions[j], j, n);

                auto [f1, f2] = lj.force_pair(r1, r2);

                add_force_contribution(result.forces[i], f1);
                add_force_contribution(result.forces[j], f2);

                accumulate_force_gradients(result.force_gradients[i], f1, i, j);
                accumulate_force_gradients(result.force_gradients[j], f2, i, j);

                dual_type potential = lj.potential_pair(r1, r2);
                result.total_potential += potential.value;

                accumulate_potential_gradients(result.potential_gradients,
                                             potential, i, j, n);
            }
        }

        return result;
    }

    /**
     * Compute forces and gradients for Coulomb interactions
     */
    ForceResult compute_coulomb_forces(
        const std::vector<std::array<T, 3>>& positions,
        const std::vector<T>& charges) const {

        size_t n = positions.size();
        ForceResult result;
        result.forces.resize(n, {T{0}, T{0}, T{0}});
        result.force_gradients.resize(n, {T{0}, T{0}, T{0}});
        result.potential_gradients.resize(n * 3, T{0});
        result.total_potential = T{0};

        DifferentiableCoulomb<T> coulomb;

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                vector_type r1 = make_dual_vector(positions[i], i, n);
                vector_type r2 = make_dual_vector(positions[j], j, n);
                dual_type q1(charges[i]);
                dual_type q2(charges[j]);

                auto [f1, f2] = coulomb.force_pair(r1, q1, r2, q2);

                add_force_contribution(result.forces[i], f1);
                add_force_contribution(result.forces[j], f2);

                accumulate_force_gradients(result.force_gradients[i], f1, i, j);
                accumulate_force_gradients(result.force_gradients[j], f2, i, j);

                dual_type potential = coulomb.potential_pair(r1, q1, r2, q2);
                result.total_potential += potential.value;

                accumulate_potential_gradients(result.potential_gradients,
                                             potential, i, j, n);
            }
        }

        return result;
    }

private:
    // Helper function to create dual vectors for automatic differentiation
    vector_type make_dual_vector(const std::array<T, 3>& pos, size_t particle_idx, size_t total_particles) const {
        return {
            dual_type(pos[0], particle_idx == 0 ? T{1} : T{0}),
            dual_type(pos[1], particle_idx == 0 ? T{1} : T{0}),
            dual_type(pos[2], particle_idx == 0 ? T{1} : T{0})
        };
    }

    void add_force_contribution(std::array<T, 3>& force, const vector_type& dual_force) const {
        force[0] += dual_force[0].value;
        force[1] += dual_force[1].value;
        force[2] += dual_force[2].value;
    }

    void accumulate_force_gradients(std::array<T, 3>& gradients,
                                   const vector_type& dual_force,
                                   size_t i, size_t j) const {
        // Simplified gradient accumulation - in practice would need full Jacobian
        gradients[0] += dual_force[0].gradient;
        gradients[1] += dual_force[1].gradient;
        gradients[2] += dual_force[2].gradient;
    }

    void accumulate_potential_gradients(std::vector<T>& gradients,
                                      const dual_type& potential,
                                      size_t i, size_t j, size_t n) const {
        // Simplified gradient accumulation
        gradients[i * 3] += potential.gradient;
        gradients[j * 3] += potential.gradient;
    }
};

} // namespace physgrad::differentiable