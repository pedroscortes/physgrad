#ifndef PHYSGRAD_LOSS_GRADIENTS_H
#define PHYSGRAD_LOSS_GRADIENTS_H

#include "common_types.h"
#include <vector>
#include <functional>

namespace physgrad {
namespace adjoint {

/**
 * Common analytical loss gradient functions for differentiable physics.
 *
 * These provide exact gradients for common loss functions, avoiding
 * the ~3% error from finite difference approximations.
 *
 * Usage:
 *   auto gradients = simulation.computeGradients(
 *       initial_pos, initial_vel, masses, dt, num_steps,
 *       loss_function,
 *       LossGradients::squared_position_distance()  // Analytical!
 *   );
 */
template<typename T>
class LossGradients {
public:
    using vector_type = ConceptVector3D<T>;

    /**
     * Loss: L = Σ ||x_i||²
     *
     * Gradient:
     *   dL/dx_i = 2 * x_i
     *   dL/dv_i = 0
     *
     * Common use case: Minimize final displacement from origin
     */
    static std::function<void(const std::vector<vector_type>&,
                             const std::vector<vector_type>&,
                             std::vector<vector_type>&,
                             std::vector<vector_type>&)>
    squared_position_distance() {
        return [](const std::vector<vector_type>& positions,
                 const std::vector<vector_type>& velocities,
                 std::vector<vector_type>& grad_pos,
                 std::vector<vector_type>& grad_vel) {
            // dL/dx = 2x
            for (size_t i = 0; i < positions.size(); ++i) {
                grad_pos[i][0] = T(2) * positions[i][0];
                grad_pos[i][1] = T(2) * positions[i][1];
                grad_pos[i][2] = T(2) * positions[i][2];
            }
            // dL/dv = 0
            for (size_t i = 0; i < velocities.size(); ++i) {
                grad_vel[i][0] = T(0);
                grad_vel[i][1] = T(0);
                grad_vel[i][2] = T(0);
            }
        };
    }

    /**
     * Loss: L = Σ ||x_i - target_i||²
     *
     * Gradient:
     *   dL/dx_i = 2 * (x_i - target_i)
     *   dL/dv_i = 0
     *
     * Common use case: Reach target positions
     */
    static std::function<void(const std::vector<vector_type>&,
                             const std::vector<vector_type>&,
                             std::vector<vector_type>&,
                             std::vector<vector_type>&)>
    squared_position_distance_to_target(const std::vector<vector_type>& targets) {
        return [targets](const std::vector<vector_type>& positions,
                        const std::vector<vector_type>& velocities,
                        std::vector<vector_type>& grad_pos,
                        std::vector<vector_type>& grad_vel) {
            // dL/dx = 2(x - target)
            for (size_t i = 0; i < positions.size(); ++i) {
                grad_pos[i][0] = T(2) * (positions[i][0] - targets[i][0]);
                grad_pos[i][1] = T(2) * (positions[i][1] - targets[i][1]);
                grad_pos[i][2] = T(2) * (positions[i][2] - targets[i][2]);
            }
            // dL/dv = 0
            for (size_t i = 0; i < velocities.size(); ++i) {
                grad_vel[i][0] = T(0);
                grad_vel[i][1] = T(0);
                grad_vel[i][2] = T(0);
            }
        };
    }

    /**
     * Loss: L = 0.5 * Σ m_i * ||v_i||²  (Kinetic energy)
     *
     * Gradient:
     *   dL/dx_i = 0
     *   dL/dv_i = m_i * v_i
     *
     * Common use case: Minimize kinetic energy (bring to rest)
     */
    static std::function<void(const std::vector<vector_type>&,
                             const std::vector<vector_type>&,
                             std::vector<vector_type>&,
                             std::vector<vector_type>&)>
    kinetic_energy(const std::vector<T>& masses) {
        return [masses](const std::vector<vector_type>& positions,
                       const std::vector<vector_type>& velocities,
                       std::vector<vector_type>& grad_pos,
                       std::vector<vector_type>& grad_vel) {
            // dL/dx = 0
            for (size_t i = 0; i < positions.size(); ++i) {
                grad_pos[i][0] = T(0);
                grad_pos[i][1] = T(0);
                grad_pos[i][2] = T(0);
            }
            // dL/dv = m * v
            for (size_t i = 0; i < velocities.size(); ++i) {
                grad_vel[i][0] = masses[i] * velocities[i][0];
                grad_vel[i][1] = masses[i] * velocities[i][1];
                grad_vel[i][2] = masses[i] * velocities[i][2];
            }
        };
    }

    /**
     * Loss: L = Σ ||v_i - target_v_i||²
     *
     * Gradient:
     *   dL/dx_i = 0
     *   dL/dv_i = 2 * (v_i - target_v_i)
     *
     * Common use case: Reach target velocities
     */
    static std::function<void(const std::vector<vector_type>&,
                             const std::vector<vector_type>&,
                             std::vector<vector_type>&,
                             std::vector<vector_type>&)>
    squared_velocity_distance_to_target(const std::vector<vector_type>& target_velocities) {
        return [target_velocities](const std::vector<vector_type>& positions,
                                   const std::vector<vector_type>& velocities,
                                   std::vector<vector_type>& grad_pos,
                                   std::vector<vector_type>& grad_vel) {
            // dL/dx = 0
            for (size_t i = 0; i < positions.size(); ++i) {
                grad_pos[i][0] = T(0);
                grad_pos[i][1] = T(0);
                grad_pos[i][2] = T(0);
            }
            // dL/dv = 2(v - target_v)
            for (size_t i = 0; i < velocities.size(); ++i) {
                grad_vel[i][0] = T(2) * (velocities[i][0] - target_velocities[i][0]);
                grad_vel[i][1] = T(2) * (velocities[i][1] - target_velocities[i][1]);
                grad_vel[i][2] = T(2) * (velocities[i][2] - target_velocities[i][2]);
            }
        };
    }

    /**
     * Loss: L = α * Σ||x_i - target_i||² + β * 0.5 * Σ m_i||v_i||²
     *
     * Combined position tracking + kinetic energy minimization
     *
     * Gradient:
     *   dL/dx_i = 2α * (x_i - target_i)
     *   dL/dv_i = β * m_i * v_i
     *
     * Common use case: Reach target positions while minimizing velocity
     */
    static std::function<void(const std::vector<vector_type>&,
                             const std::vector<vector_type>&,
                             std::vector<vector_type>&,
                             std::vector<vector_type>&)>
    position_and_kinetic_energy(const std::vector<vector_type>& targets,
                               const std::vector<T>& masses,
                               T position_weight = T(1),
                               T kinetic_weight = T(1)) {
        return [targets, masses, position_weight, kinetic_weight](
                   const std::vector<vector_type>& positions,
                   const std::vector<vector_type>& velocities,
                   std::vector<vector_type>& grad_pos,
                   std::vector<vector_type>& grad_vel) {
            // dL/dx = 2α(x - target)
            for (size_t i = 0; i < positions.size(); ++i) {
                grad_pos[i][0] = T(2) * position_weight * (positions[i][0] - targets[i][0]);
                grad_pos[i][1] = T(2) * position_weight * (positions[i][1] - targets[i][1]);
                grad_pos[i][2] = T(2) * position_weight * (positions[i][2] - targets[i][2]);
            }
            // dL/dv = β * m * v
            for (size_t i = 0; i < velocities.size(); ++i) {
                grad_vel[i][0] = kinetic_weight * masses[i] * velocities[i][0];
                grad_vel[i][1] = kinetic_weight * masses[i] * velocities[i][1];
                grad_vel[i][2] = kinetic_weight * masses[i] * velocities[i][2];
            }
        };
    }

    /**
     * Loss: L = ||x_i - target||²  (Single particle tracking)
     *
     * Gradient:
     *   dL/dx_j = 2(x_i - target) if j == i, else 0
     *   dL/dv_j = 0
     *
     * Common use case: Track single particle to target
     */
    static std::function<void(const std::vector<vector_type>&,
                             const std::vector<vector_type>&,
                             std::vector<vector_type>&,
                             std::vector<vector_type>&)>
    single_particle_position(size_t particle_index, const vector_type& target) {
        return [particle_index, target](const std::vector<vector_type>& positions,
                                       const std::vector<vector_type>& velocities,
                                       std::vector<vector_type>& grad_pos,
                                       std::vector<vector_type>& grad_vel) {
            // Zero all gradients
            for (size_t i = 0; i < positions.size(); ++i) {
                grad_pos[i][0] = T(0);
                grad_pos[i][1] = T(0);
                grad_pos[i][2] = T(0);
            }
            for (size_t i = 0; i < velocities.size(); ++i) {
                grad_vel[i][0] = T(0);
                grad_vel[i][1] = T(0);
                grad_vel[i][2] = T(0);
            }
            // Only particle i contributes
            if (particle_index < positions.size()) {
                grad_pos[particle_index][0] = T(2) * (positions[particle_index][0] - target[0]);
                grad_pos[particle_index][1] = T(2) * (positions[particle_index][1] - target[1]);
                grad_pos[particle_index][2] = T(2) * (positions[particle_index][2] - target[2]);
            }
        };
    }

    /**
     * Custom loss gradient: build your own!
     *
     * Example:
     *   auto my_gradient = LossGradients::custom(
     *       [](const auto& pos, const auto& vel, auto& grad_pos, auto& grad_vel) {
     *           // Your analytical gradient here
     *           grad_pos[0][0] = ...;
     *           grad_vel[0][0] = ...;
     *       }
     *   );
     */
    template<typename F>
    static std::function<void(const std::vector<vector_type>&,
                             const std::vector<vector_type>&,
                             std::vector<vector_type>&,
                             std::vector<vector_type>&)>
    custom(F&& gradient_function) {
        return std::forward<F>(gradient_function);
    }
};

} // namespace adjoint
} // namespace physgrad

#endif // PHYSGRAD_LOSS_GRADIENTS_H
