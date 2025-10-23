/**
 * PhysGrad - Differentiable Contact Mechanics Header
 *
 * C++ interface for CUDA differentiable contact kernels
 */

#ifndef PHYSGRAD_DIFFERENTIABLE_CONTACT_KERNELS_H
#define PHYSGRAD_DIFFERENTIABLE_CONTACT_KERNELS_H

#include <cuda_runtime.h>

namespace physgrad {
namespace differentiable_contact {

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/**
 * Contact point information
 */
struct ContactPoint {
    int particle1;
    int particle2;
    float3 normal;           // Contact normal (from particle1 to particle2)
    float penetration_depth; // How much the particles overlap
    float3 contact_position; // Position in world space
};

/**
 * Contact material properties
 */
struct ContactMaterialProperties {
    float stiffness;         // Contact spring stiffness
    float damping;           // Contact damping coefficient
    float friction;          // Coefficient of friction
    float restitution;       // Coefficient of restitution (for impulse-based)
};

// =============================================================================
// KERNEL LAUNCHERS
// =============================================================================

extern "C" {

/**
 * Detect sphere-sphere contacts (differentiable)
 *
 * @param positions Particle positions
 * @param radii Particle radii
 * @param velocities Particle velocities
 * @param contacts Output contact points
 * @param num_contacts Output number of contacts detected
 * @param num_particles Number of particles
 * @param contact_threshold Early contact detection threshold
 * @param stream CUDA stream (optional)
 */
void launch_detect_sphere_contacts_differentiable(
    const float3* positions,
    const float* radii,
    const float3* velocities,
    ContactPoint* contacts,
    int* num_contacts,
    int num_particles,
    float contact_threshold,
    cudaStream_t stream = 0
);

/**
 * Compute contact forces using spring-damper model (forward pass)
 *
 * @param contacts Detected contact points
 * @param velocities Particle velocities
 * @param material_props Contact material properties
 * @param forces Output forces (accumulated)
 * @param saved_contact_forces Saved contact forces for backward pass
 * @param num_contacts Number of contacts
 * @param stream CUDA stream (optional)
 */
void launch_compute_contact_forces(
    const ContactPoint* contacts,
    const float3* velocities,
    const ContactMaterialProperties* material_props,
    float3* forces,
    float3* saved_contact_forces,
    int num_contacts,
    cudaStream_t stream = 0
);

/**
 * Backward pass for contact forces
 *
 * Propagates gradients from ∂L/∂F back to ∂L/∂x, ∂L/∂v, and material properties
 *
 * @param grad_forces Gradients w.r.t. forces (∂L/∂F)
 * @param grad_positions Output gradients w.r.t. positions (∂L/∂x)
 * @param grad_velocities Output gradients w.r.t. velocities (∂L/∂v)
 * @param grad_radii Output gradients w.r.t. radii (∂L/∂r) - optional
 * @param grad_stiffness Output gradient w.r.t. stiffness (∂L/∂k)
 * @param grad_damping Output gradient w.r.t. damping (∂L/∂c)
 * @param grad_friction Output gradient w.r.t. friction (∂L/∂μ)
 * @param contacts Saved contact points from forward pass
 * @param saved_positions Saved positions from forward pass
 * @param saved_velocities Saved velocities from forward pass
 * @param saved_radii Saved radii from forward pass
 * @param material_props Contact material properties
 * @param num_contacts Number of contacts
 * @param num_particles Number of particles
 * @param stream CUDA stream (optional)
 */
void launch_contact_forces_backward(
    const float3* grad_forces,
    float3* grad_positions,
    float3* grad_velocities,
    float* grad_radii,
    float* grad_stiffness,
    float* grad_damping,
    float* grad_friction,
    const ContactPoint* contacts,
    const float3* saved_positions,
    const float3* saved_velocities,
    const float* saved_radii,
    const ContactMaterialProperties* material_props,
    int num_contacts,
    int num_particles,
    cudaStream_t stream = 0
);

} // extern "C"

} // namespace differentiable_contact
} // namespace physgrad

#endif // PHYSGRAD_DIFFERENTIABLE_CONTACT_KERNELS_H
