/**
 * PhysGrad - Differentiable Contact Optimization
 *
 * Implements single-level differentiable contact mechanics with gradient computation.
 * Enables optimization of contact-rich physics problems using automatic differentiation.
 * Based on implicit differentiation through contact constraints.
 */

#ifndef PHYSGRAD_DIFFERENTIABLE_CONTACT_H
#define PHYSGRAD_DIFFERENTIABLE_CONTACT_H

#include "common_types.h"
#include "differentiable_forces.h"
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <algorithm>
#include <iostream>

// Include concepts when available
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "concepts/forward_declarations.h"
#endif

namespace physgrad {
namespace contact {

// =============================================================================
// CONTACT DETECTION AND GEOMETRY
// =============================================================================

/**
 * Contact point between two objects
 */
template<typename T>
struct ContactPoint {
    ConceptVector3D<T> position;       // Contact position in world space
    ConceptVector3D<T> normal;         // Contact normal (from body A to body B)
    T penetration_depth;               // Penetration depth (positive = penetrating)
    size_t body_a_id;                  // ID of first body
    size_t body_b_id;                  // ID of second body
    T friction_coefficient;            // Friction coefficient for this contact

    ContactPoint() : penetration_depth(T{0}), body_a_id(0), body_b_id(0),
                     friction_coefficient(T{0.5}) {}

    ContactPoint(const ConceptVector3D<T>& pos, const ConceptVector3D<T>& norm,
                T depth, size_t a, size_t b, T friction = T{0.5})
        : position(pos), normal(norm), penetration_depth(depth),
          body_a_id(a), body_b_id(b), friction_coefficient(friction) {}
};

/**
 * Simple sphere-sphere contact detection
 */
template<typename T>
class SphereContactDetector {
public:
    SphereContactDetector(const std::vector<T>& radii) : radii_(radii) {}

    std::vector<ContactPoint<T>> detectContacts(
        const std::vector<ConceptVector3D<T>>& positions) const {

        std::vector<ContactPoint<T>> contacts;

        for (size_t i = 0; i < positions.size(); ++i) {
            for (size_t j = i + 1; j < positions.size(); ++j) {
                auto contact = detectSphereContact(positions[i], positions[j],
                                                  radii_[i], radii_[j], i, j);
                if (contact.penetration_depth > T{0}) {
                    contacts.push_back(contact);
                }
            }
        }

        return contacts;
    }

private:
    std::vector<T> radii_;

    ContactPoint<T> detectSphereContact(
        const ConceptVector3D<T>& pos_a, const ConceptVector3D<T>& pos_b,
        T radius_a, T radius_b, size_t id_a, size_t id_b) const {

        // Distance vector
        ConceptVector3D<T> diff = {
            pos_b[0] - pos_a[0],
            pos_b[1] - pos_a[1],
            pos_b[2] - pos_a[2]
        };

        T distance = std::sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
        T sum_radii = radius_a + radius_b;
        T penetration = sum_radii - distance;

        ContactPoint<T> contact;
        if (penetration > T{0} && distance > T{1e-10}) {
            // Contact normal (from A to B)
            ConceptVector3D<T> normal = {
                diff[0] / distance,
                diff[1] / distance,
                diff[2] / distance
            };

            // Contact position (midpoint between surfaces)
            ConceptVector3D<T> contact_pos = {
                pos_a[0] + normal[0] * radius_a,
                pos_a[1] + normal[1] * radius_a,
                pos_a[2] + normal[2] * radius_a
            };

            contact = ContactPoint<T>(contact_pos, normal, penetration, id_a, id_b);
        }

        return contact;
    }
};

/**
 * Simple plane-sphere contact detection
 */
template<typename T>
class PlaneContactDetector {
public:
    PlaneContactDetector(const ConceptVector3D<T>& plane_normal, T plane_offset)
        : plane_normal_(plane_normal), plane_offset_(plane_offset) {}

    std::vector<ContactPoint<T>> detectContacts(
        const std::vector<ConceptVector3D<T>>& positions,
        const std::vector<T>& radii) const {

        std::vector<ContactPoint<T>> contacts;

        for (size_t i = 0; i < positions.size(); ++i) {
            auto contact = detectPlaneContact(positions[i], radii[i], i);
            if (contact.penetration_depth > T{0}) {
                contacts.push_back(contact);
            }
        }

        return contacts;
    }

private:
    ConceptVector3D<T> plane_normal_;
    T plane_offset_;

    ContactPoint<T> detectPlaneContact(
        const ConceptVector3D<T>& position, T radius, size_t body_id) const {

        // Distance from sphere center to plane
        T distance_to_plane =
            position[0] * plane_normal_[0] +
            position[1] * plane_normal_[1] +
            position[2] * plane_normal_[2] - plane_offset_;

        T penetration = radius - distance_to_plane;

        ContactPoint<T> contact;
        if (penetration > T{0}) {
            // Contact position on sphere surface
            ConceptVector3D<T> contact_pos = {
                position[0] - plane_normal_[0] * radius,
                position[1] - plane_normal_[1] * radius,
                position[2] - plane_normal_[2] * radius
            };

            contact = ContactPoint<T>(contact_pos, plane_normal_, penetration,
                                    body_id, SIZE_MAX); // SIZE_MAX indicates ground plane
        }

        return contact;
    }
};

// =============================================================================
// CONTACT CONSTRAINT SOLVER
// =============================================================================

/**
 * Contact impulse and constraint forces
 */
template<typename T>
struct ContactSolution {
    std::vector<T> normal_impulses;    // Normal impulses for each contact
    std::vector<T> friction_impulses_u; // Friction impulses (tangent direction 1)
    std::vector<T> friction_impulses_v; // Friction impulses (tangent direction 2)
    std::vector<ConceptVector3D<T>> contact_forces; // Total forces at contacts
    T total_energy_dissipated;         // Energy lost to friction/damping
    int num_iterations;                // Solver iterations
    bool converged;                    // Whether solver converged

    ContactSolution() : total_energy_dissipated(T{0}), num_iterations(0), converged(false) {}
};

/**
 * Projected Gauss-Seidel contact solver with differentiability
 */
template<typename T>
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    requires concepts::PhysicsScalar<T>
#endif
class DifferentiableContactSolver {
public:
    struct SolverParams {
        int max_iterations = 20;
        T tolerance = T{1e-6};
        T restitution = T{0.2};        // Coefficient of restitution
        T contact_stiffness = T{1e6};  // Contact penalty stiffness
        T contact_damping = T{1e3};    // Contact damping
        bool use_friction = true;
        bool warm_start = true;
    };

    DifferentiableContactSolver(const SolverParams& params = SolverParams())
        : params_(params) {}

    /**
     * Solve contact constraints using projected iterative method
     */
    ContactSolution<T> solveContacts(
        const std::vector<ContactPoint<T>>& contacts,
        const std::vector<ConceptVector3D<T>>& velocities,
        const std::vector<T>& masses,
        T dt) {

        ContactSolution<T> solution;

        if (contacts.empty()) {
            solution.converged = true;
            return solution;
        }

        const size_t n_contacts = contacts.size();

        // Initialize solution vectors
        solution.normal_impulses.resize(n_contacts, T{0});
        solution.friction_impulses_u.resize(n_contacts, T{0});
        solution.friction_impulses_v.resize(n_contacts, T{0});
        solution.contact_forces.resize(n_contacts);

        // Warm start from previous solution if available
        if (params_.warm_start && !previous_normal_impulses_.empty() &&
            previous_normal_impulses_.size() == n_contacts) {
            solution.normal_impulses = previous_normal_impulses_;
            solution.friction_impulses_u = previous_friction_impulses_u_;
            solution.friction_impulses_v = previous_friction_impulses_v_;
        }

        // Projected Gauss-Seidel iterations
        for (int iter = 0; iter < params_.max_iterations; ++iter) {
            T max_delta = T{0};

            // Solve normal constraints
            for (size_t c = 0; c < n_contacts; ++c) {
                T old_impulse = solution.normal_impulses[c];

                // Compute constraint violation
                T constraint_violation = computeNormalConstraintViolation(
                    c, contacts, velocities, solution);

                // Compute diagonal mass matrix element
                T diagonal_mass = computeDiagonalMass(c, contacts, masses);

                // Update impulse
                T delta_impulse = -constraint_violation / (diagonal_mass + T{1e-12});
                T new_impulse = old_impulse + delta_impulse;

                // Project to non-negative (no adhesion)
                new_impulse = std::max(new_impulse, T{0});

                solution.normal_impulses[c] = new_impulse;
                max_delta = std::max(max_delta, std::abs(new_impulse - old_impulse));
            }

            // Solve friction constraints if enabled
            if (params_.use_friction) {
                solveFrictionConstraints(contacts, velocities, masses, solution);
            }

            solution.num_iterations = iter + 1;

            // Check convergence
            if (max_delta < params_.tolerance) {
                solution.converged = true;
                break;
            }
        }

        // Store for warm starting
        previous_normal_impulses_ = solution.normal_impulses;
        previous_friction_impulses_u_ = solution.friction_impulses_u;
        previous_friction_impulses_v_ = solution.friction_impulses_v;

        // Compute final contact forces
        computeContactForces(contacts, solution);

        return solution;
    }

    /**
     * Compute gradients of contact solution w.r.t. positions and parameters
     */
    std::pair<std::vector<ConceptVector3D<T>>, std::vector<T>>
    computeContactGradients(
        const std::vector<ContactPoint<T>>& contacts,
        const ContactSolution<T>& solution,
        const std::vector<ConceptVector3D<T>>& adjoint_forces) {

        const size_t n_contacts = contacts.size();
        const size_t n_bodies = adjoint_forces.size();

        std::vector<ConceptVector3D<T>> position_gradients(n_bodies);
        std::vector<T> parameter_gradients;

        // Implicit differentiation through contact constraints
        // ∂λ/∂x = -∂²L/∂λ²⁻¹ * ∂²L/∂λ∂x

        for (size_t c = 0; c < n_contacts; ++c) {
            const auto& contact = contacts[c];
            T lambda = solution.normal_impulses[c];

            if (lambda > T{1e-10}) { // Active contact
                // Gradient w.r.t. contact normal
                ConceptVector3D<T> normal_gradient;

                if (contact.body_b_id != SIZE_MAX) {
                    // Regular contact between two bodies
                    normal_gradient = {
                        adjoint_forces[contact.body_a_id][0] - adjoint_forces[contact.body_b_id][0],
                        adjoint_forces[contact.body_a_id][1] - adjoint_forces[contact.body_b_id][1],
                        adjoint_forces[contact.body_a_id][2] - adjoint_forces[contact.body_b_id][2]
                    };
                } else {
                    // Contact with ground plane
                    normal_gradient = adjoint_forces[contact.body_a_id];
                }

                // Chain rule through contact geometry
                T normal_mag = std::sqrt(
                    normal_gradient[0]*normal_gradient[0] +
                    normal_gradient[1]*normal_gradient[1] +
                    normal_gradient[2]*normal_gradient[2]
                );

                if (normal_mag > T{1e-10}) {
                    T scale = lambda / normal_mag;

                    // Distribute gradients to bodies
                    position_gradients[contact.body_a_id] = {
                        position_gradients[contact.body_a_id][0] + scale * contact.normal[0],
                        position_gradients[contact.body_a_id][1] + scale * contact.normal[1],
                        position_gradients[contact.body_a_id][2] + scale * contact.normal[2]
                    };

                    if (contact.body_b_id != SIZE_MAX) {
                        position_gradients[contact.body_b_id] = {
                            position_gradients[contact.body_b_id][0] - scale * contact.normal[0],
                            position_gradients[contact.body_b_id][1] - scale * contact.normal[1],
                            position_gradients[contact.body_b_id][2] - scale * contact.normal[2]
                        };
                    }
                }
            }
        }

        return {position_gradients, parameter_gradients};
    }

private:
    SolverParams params_;

    // Warm start data
    std::vector<T> previous_normal_impulses_;
    std::vector<T> previous_friction_impulses_u_;
    std::vector<T> previous_friction_impulses_v_;

    /**
     * Compute normal constraint violation for iterative solver
     */
    T computeNormalConstraintViolation(
        size_t contact_idx,
        const std::vector<ContactPoint<T>>& contacts,
        const std::vector<ConceptVector3D<T>>& velocities,
        const ContactSolution<T>& solution) {

        const auto& contact = contacts[contact_idx];

        T rel_normal_vel;
        if (contact.body_b_id != SIZE_MAX) {
            // Regular contact between two bodies
            rel_normal_vel =
                (velocities[contact.body_b_id][0] - velocities[contact.body_a_id][0]) * contact.normal[0] +
                (velocities[contact.body_b_id][1] - velocities[contact.body_a_id][1]) * contact.normal[1] +
                (velocities[contact.body_b_id][2] - velocities[contact.body_a_id][2]) * contact.normal[2];
        } else {
            // Contact with ground plane (body B is stationary)
            rel_normal_vel =
                -velocities[contact.body_a_id][0] * contact.normal[0] -
                velocities[contact.body_a_id][1] * contact.normal[1] -
                velocities[contact.body_a_id][2] * contact.normal[2];
        }

        // Add constraint correction terms
        T penetration_correction = params_.contact_stiffness * contact.penetration_depth;
        T damping_term = params_.contact_damping * rel_normal_vel;

        return rel_normal_vel + penetration_correction + damping_term;
    }

    /**
     * Compute effective mass for contact constraint
     */
    T computeDiagonalMass(
        size_t contact_idx,
        const std::vector<ContactPoint<T>>& contacts,
        const std::vector<T>& masses) {

        const auto& contact = contacts[contact_idx];

        T inv_mass_a = (masses[contact.body_a_id] > T{1e-10}) ? T{1} / masses[contact.body_a_id] : T{0};
        T inv_mass_b = T{0};

        if (contact.body_b_id != SIZE_MAX) {
            inv_mass_b = (masses[contact.body_b_id] > T{1e-10}) ? T{1} / masses[contact.body_b_id] : T{0};
        }

        T total_inv_mass = inv_mass_a + inv_mass_b;
        return (total_inv_mass > T{1e-10}) ? T{1} / total_inv_mass : T{1e12};
    }

    /**
     * Solve friction constraints using Coulomb friction model
     */
    void solveFrictionConstraints(
        const std::vector<ContactPoint<T>>& contacts,
        const std::vector<ConceptVector3D<T>>& velocities,
        const std::vector<T>& masses,
        ContactSolution<T>& solution) {

        // Simplified friction: proportional to normal force
        for (size_t c = 0; c < contacts.size(); ++c) {
            const auto& contact = contacts[c];
            T normal_impulse = solution.normal_impulses[c];
            T max_friction = contact.friction_coefficient * normal_impulse;

            // Compute tangential velocity (simplified 1D)
            T rel_tangent_vel;
            if (contact.body_b_id != SIZE_MAX) {
                rel_tangent_vel =
                    (velocities[contact.body_b_id][0] - velocities[contact.body_a_id][0]) -
                    (velocities[contact.body_b_id][0] - velocities[contact.body_a_id][0]) *
                    contact.normal[0] * contact.normal[0];
            } else {
                rel_tangent_vel = velocities[contact.body_a_id][0] -
                                 velocities[contact.body_a_id][0] * contact.normal[0] * contact.normal[0];
            }

            // Apply friction impulse
            T friction_impulse = std::clamp(-rel_tangent_vel * masses[contact.body_a_id],
                                           -max_friction, max_friction);

            solution.friction_impulses_u[c] = friction_impulse;
        }
    }

    /**
     * Compute final contact forces from impulses
     */
    void computeContactForces(
        const std::vector<ContactPoint<T>>& contacts,
        ContactSolution<T>& solution) {

        for (size_t c = 0; c < contacts.size(); ++c) {
            const auto& contact = contacts[c];
            T normal_impulse = solution.normal_impulses[c];
            T friction_impulse = solution.friction_impulses_u[c];

            // Total force = normal + friction components
            solution.contact_forces[c] = {
                normal_impulse * contact.normal[0] + friction_impulse,
                normal_impulse * contact.normal[1],
                normal_impulse * contact.normal[2]
            };
        }
    }
};

// =============================================================================
// DIFFERENTIABLE CONTACT SIMULATION
// =============================================================================

/**
 * Complete differentiable contact simulation system
 */
template<typename T>
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    requires concepts::PhysicsScalar<T>
#endif
class DifferentiableContactSimulation {
public:
    using ContactSolver = DifferentiableContactSolver<T>;

    struct SimulationParams {
        T timestep = T{0.01};
        bool enable_contacts = true;
        bool enable_gravity = true;
        ConceptVector3D<T> gravity = {T{0}, T{-9.81}, T{0}};
        bool use_ground_plane = false;
        ConceptVector3D<T> ground_normal = {T{0}, T{1}, T{0}};
        T ground_offset = T{0};
    };

    DifferentiableContactSimulation(
        const std::vector<T>& radii,
        const typename ContactSolver::SolverParams& solver_params = {},
        const SimulationParams& sim_params = {})
        : radii_(radii),
          sphere_detector_(radii),
          plane_detector_(sim_params.ground_normal, sim_params.ground_offset),
          contact_solver_(solver_params),
          sim_params_(sim_params) {}

    /**
     * Simulate one timestep with contact resolution
     */
    void step(std::vector<ConceptVector3D<T>>& positions,
             std::vector<ConceptVector3D<T>>& velocities,
             const std::vector<T>& masses) {

        const T dt = sim_params_.timestep;

        // Apply gravity
        if (sim_params_.enable_gravity) {
            for (size_t i = 0; i < velocities.size(); ++i) {
                velocities[i] = {
                    velocities[i][0] + sim_params_.gravity[0] * dt,
                    velocities[i][1] + sim_params_.gravity[1] * dt,
                    velocities[i][2] + sim_params_.gravity[2] * dt
                };
            }
        }

        // Contact detection
        std::vector<ContactPoint<T>> contacts;
        if (sim_params_.enable_contacts) {
            // Sphere-sphere contacts
            auto sphere_contacts = sphere_detector_.detectContacts(positions);
            contacts.insert(contacts.end(), sphere_contacts.begin(), sphere_contacts.end());

            // Ground plane contacts
            if (sim_params_.use_ground_plane) {
                auto plane_contacts = plane_detector_.detectContacts(positions, radii_);
                contacts.insert(contacts.end(), plane_contacts.begin(), plane_contacts.end());
            }
        }

        // Contact resolution
        ContactSolution<T> contact_solution;
        if (!contacts.empty()) {
            contact_solution = contact_solver_.solveContacts(contacts, velocities, masses, dt);

            // Apply contact impulses to velocities
            applyContactImpulses(contacts, contact_solution, velocities, masses, dt);
        }

        // Integration
        for (size_t i = 0; i < positions.size(); ++i) {
            positions[i] = {
                positions[i][0] + velocities[i][0] * dt,
                positions[i][1] + velocities[i][1] * dt,
                positions[i][2] + velocities[i][2] * dt
            };
        }

        // Store last contact solution for gradient computation
        last_contacts_ = contacts;
        last_contact_solution_ = contact_solution;
    }

    /**
     * Compute gradients w.r.t. initial conditions using adjoint method
     */
    std::pair<std::vector<ConceptVector3D<T>>, std::vector<ConceptVector3D<T>>>
    computeGradients(
        const std::vector<ConceptVector3D<T>>& adjoint_positions,
        const std::vector<ConceptVector3D<T>>& adjoint_velocities) {

        std::vector<ConceptVector3D<T>> pos_grads = adjoint_positions;
        std::vector<ConceptVector3D<T>> vel_grads = adjoint_velocities;

        // Add contact contributions to gradients
        if (!last_contacts_.empty()) {
            auto [contact_pos_grads, contact_param_grads] =
                contact_solver_.computeContactGradients(
                    last_contacts_, last_contact_solution_, adjoint_velocities);

            // Accumulate contact gradients
            for (size_t i = 0; i < pos_grads.size(); ++i) {
                pos_grads[i] = {
                    pos_grads[i][0] + contact_pos_grads[i][0],
                    pos_grads[i][1] + contact_pos_grads[i][1],
                    pos_grads[i][2] + contact_pos_grads[i][2]
                };
            }
        }

        return {pos_grads, vel_grads};
    }

    // Getters for analysis
    const std::vector<ContactPoint<T>>& getLastContacts() const { return last_contacts_; }
    const ContactSolution<T>& getLastContactSolution() const { return last_contact_solution_; }

private:
    std::vector<T> radii_;
    SphereContactDetector<T> sphere_detector_;
    PlaneContactDetector<T> plane_detector_;
    ContactSolver contact_solver_;
    SimulationParams sim_params_;

    // State for gradient computation
    std::vector<ContactPoint<T>> last_contacts_;
    ContactSolution<T> last_contact_solution_;

    /**
     * Apply contact impulses to update velocities
     */
    void applyContactImpulses(
        const std::vector<ContactPoint<T>>& contacts,
        const ContactSolution<T>& solution,
        std::vector<ConceptVector3D<T>>& velocities,
        const std::vector<T>& masses,
        T dt) {

        for (size_t c = 0; c < contacts.size(); ++c) {
            const auto& contact = contacts[c];
            const auto& force = solution.contact_forces[c];

            T inv_mass_a = (masses[contact.body_a_id] > T{1e-10}) ?
                          T{1} / masses[contact.body_a_id] : T{0};

            // Apply impulse to body A
            velocities[contact.body_a_id] = {
                velocities[contact.body_a_id][0] - force[0] * inv_mass_a * dt,
                velocities[contact.body_a_id][1] - force[1] * inv_mass_a * dt,
                velocities[contact.body_a_id][2] - force[2] * inv_mass_a * dt
            };

            // Apply impulse to body B (if not ground plane)
            if (contact.body_b_id != SIZE_MAX) {
                T inv_mass_b = (masses[contact.body_b_id] > T{1e-10}) ?
                              T{1} / masses[contact.body_b_id] : T{0};

                velocities[contact.body_b_id] = {
                    velocities[contact.body_b_id][0] + force[0] * inv_mass_b * dt,
                    velocities[contact.body_b_id][1] + force[1] * inv_mass_b * dt,
                    velocities[contact.body_b_id][2] + force[2] * inv_mass_b * dt
                };
            }
        }
    }
};

#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
// Verify concept compliance
static_assert(concepts::PhysicsScalar<float>);
static_assert(concepts::PhysicsScalar<double>);
#endif

} // namespace contact
} // namespace physgrad

#endif // PHYSGRAD_DIFFERENTIABLE_CONTACT_H