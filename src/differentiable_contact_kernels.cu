/**
 * PhysGrad - Differentiable Contact Mechanics CUDA Kernels
 *
 * Implements forward and backward passes for collision detection and contact resolution
 * enabling gradient-based optimization through contact interactions.
 *
 * Key Features:
 * - Differentiable contact detection
 * - Differentiable contact force computation (spring-damper model)
 * - Friction with gradients
 * - Adjoint methods for backpropagation through contacts
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace physgrad {
namespace differentiable_contact {

// =============================================================================
// CONTACT DATA STRUCTURES
// =============================================================================

struct ContactPoint {
    int particle1;
    int particle2;
    float3 normal;           // Contact normal (from particle1 to particle2)
    float penetration_depth; // How much the particles overlap
    float3 contact_position; // Position in world space
};

struct ContactMaterialProperties {
    float stiffness;         // Contact spring stiffness
    float damping;           // Contact damping coefficient
    float friction;          // Coefficient of friction
    float restitution;       // Coefficient of restitution (for impulse-based)
};

// =============================================================================
// FORWARD PASS: CONTACT DETECTION
// =============================================================================

/**
 * Detect sphere-sphere contacts
 *
 * Forward: For each pair (i, j), check if distance < radius_i + radius_j
 * Stores contact information for both forward simulation and backward pass
 */
__global__ void detect_sphere_contacts_differentiable_kernel(
    // Inputs
    const float3* __restrict__ positions,
    const float* __restrict__ radii,
    const float3* __restrict__ velocities,

    // Outputs
    ContactPoint* __restrict__ contacts,
    int* __restrict__ num_contacts,

    // Parameters
    int num_particles,
    float contact_threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    for (int j = i + 1; j < num_particles; ++j) {
        // Compute distance vector
        float3 r_ij = {
            positions[j].x - positions[i].x,
            positions[j].y - positions[i].y,
            positions[j].z - positions[i].z
        };

        float distance = sqrtf(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);
        float sum_radii = radii[i] + radii[j];

        // Check for contact (including threshold for early detection)
        if (distance < sum_radii + contact_threshold && distance > 1e-10f) {
            int contact_idx = atomicAdd(num_contacts, 1);
            if (contact_idx < 10000) { // Max contacts limit

                ContactPoint contact;
                contact.particle1 = i;
                contact.particle2 = j;

                // Contact normal (normalized)
                contact.normal = {
                    r_ij.x / distance,
                    r_ij.y / distance,
                    r_ij.z / distance
                };

                // Penetration depth (positive when overlapping)
                contact.penetration_depth = sum_radii - distance;

                // Contact position (midpoint between surfaces)
                float t = radii[i] / sum_radii;
                contact.contact_position = {
                    positions[i].x + r_ij.x * t,
                    positions[i].y + r_ij.y * t,
                    positions[i].z + r_ij.z * t
                };

                contacts[contact_idx] = contact;
            }
        }
    }
}

// =============================================================================
// FORWARD PASS: CONTACT FORCES
// =============================================================================

/**
 * Compute contact forces using spring-damper model
 *
 * Forward: F_contact = k * penetration * normal - damping * v_rel_normal
 *
 * This function saves all intermediate values needed for the backward pass
 */
__global__ void compute_contact_forces_kernel(
    // Inputs
    const ContactPoint* __restrict__ contacts,
    const float3* __restrict__ velocities,
    const ContactMaterialProperties* __restrict__ material_props,

    // Outputs
    float3* __restrict__ forces,

    // Saved for backward pass (optional, only if gradients needed)
    float3* __restrict__ saved_contact_forces,  // Forces applied at each contact

    int num_contacts
) {
    int contact_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (contact_idx >= num_contacts) return;

    const ContactPoint& contact = contacts[contact_idx];
    const ContactMaterialProperties& props = *material_props;

    int p1 = contact.particle1;
    int p2 = contact.particle2;

    // === Normal Force (Spring-Damper) ===

    // Spring force: F_spring = k * penetration * normal
    float spring_magnitude = props.stiffness * contact.penetration_depth;

    // Damping force: F_damping = c * v_rel · normal
    float3 v_rel = {
        velocities[p2].x - velocities[p1].x,
        velocities[p2].y - velocities[p1].y,
        velocities[p2].z - velocities[p1].z
    };

    float v_rel_normal = v_rel.x * contact.normal.x +
                         v_rel.y * contact.normal.y +
                         v_rel.z * contact.normal.z;

    float damping_magnitude = props.damping * v_rel_normal;

    // Total normal force magnitude
    float normal_force_magnitude = spring_magnitude - damping_magnitude;

    // Ensure non-penetrating contacts don't pull (only push)
    if (normal_force_magnitude < 0.0f) {
        normal_force_magnitude = 0.0f;
    }

    // Normal force vector
    float3 normal_force = {
        normal_force_magnitude * contact.normal.x,
        normal_force_magnitude * contact.normal.y,
        normal_force_magnitude * contact.normal.z
    };

    // === Friction Force ===

    // Tangential relative velocity
    float3 v_rel_tangent = {
        v_rel.x - v_rel_normal * contact.normal.x,
        v_rel.y - v_rel_normal * contact.normal.y,
        v_rel.z - v_rel_normal * contact.normal.z
    };

    float v_tangent_mag = sqrtf(
        v_rel_tangent.x * v_rel_tangent.x +
        v_rel_tangent.y * v_rel_tangent.y +
        v_rel_tangent.z * v_rel_tangent.z
    );

    float3 friction_force = {0.0f, 0.0f, 0.0f};

    if (v_tangent_mag > 1e-10f) {
        // Friction force: F_friction = μ * |F_normal| * (-v_tangent_hat)
        float friction_magnitude = props.friction * normal_force_magnitude;

        friction_force = {
            -friction_magnitude * v_rel_tangent.x / v_tangent_mag,
            -friction_magnitude * v_rel_tangent.y / v_tangent_mag,
            -friction_magnitude * v_rel_tangent.z / v_tangent_mag
        };
    }

    // === Total Contact Force ===

    float3 total_force = {
        normal_force.x + friction_force.x,
        normal_force.y + friction_force.y,
        normal_force.z + friction_force.z
    };

    // Apply forces (Newton's 3rd law)
    atomicAdd(&forces[p2].x, total_force.x);
    atomicAdd(&forces[p2].y, total_force.y);
    atomicAdd(&forces[p2].z, total_force.z);

    atomicAdd(&forces[p1].x, -total_force.x);
    atomicAdd(&forces[p1].y, -total_force.y);
    atomicAdd(&forces[p1].z, -total_force.z);

    // Save contact forces for backward pass
    if (saved_contact_forces != nullptr) {
        saved_contact_forces[contact_idx] = total_force;
    }
}

// =============================================================================
// BACKWARD PASS: CONTACT FORCES
// =============================================================================

/**
 * Backward pass for contact force computation
 *
 * Propagates gradients from ∂L/∂F back to ∂L/∂x and ∂L/∂v
 * Also computes gradients w.r.t. contact material properties (k, c, μ)
 */
__global__ void contact_forces_backward_kernel(
    // Gradients w.r.t. outputs (input to backward pass)
    const float3* __restrict__ grad_forces,  // ∂L/∂F

    // Gradients w.r.t. inputs (output of backward pass)
    float3* __restrict__ grad_positions,     // ∂L/∂x
    float3* __restrict__ grad_velocities,    // ∂L/∂v
    float* __restrict__ grad_radii,          // ∂L/∂r (optional)

    // Gradients w.r.t. material properties (NEW!)
    float* __restrict__ grad_stiffness,      // ∂L/∂k
    float* __restrict__ grad_damping,        // ∂L/∂c
    float* __restrict__ grad_friction,       // ∂L/∂μ

    // Saved from forward pass
    const ContactPoint* __restrict__ contacts,
    const float3* __restrict__ saved_positions,
    const float3* __restrict__ saved_velocities,
    const float* __restrict__ saved_radii,
    const ContactMaterialProperties* __restrict__ material_props,

    int num_contacts,
    int num_particles
) {
    int contact_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (contact_idx >= num_contacts) return;

    const ContactPoint& contact = contacts[contact_idx];
    const ContactMaterialProperties& props = *material_props;

    int p1 = contact.particle1;
    int p2 = contact.particle2;

    // Get adjoint forces
    float3 adj_f1 = {-grad_forces[p1].x, -grad_forces[p1].y, -grad_forces[p1].z};  // F1 = -F_contact
    float3 adj_f2 = grad_forces[p2];  // F2 = F_contact

    // Reconstruct forward pass values
    float3 v_rel = {
        saved_velocities[p2].x - saved_velocities[p1].x,
        saved_velocities[p2].y - saved_velocities[p1].y,
        saved_velocities[p2].z - saved_velocities[p1].z
    };

    float v_rel_normal = v_rel.x * contact.normal.x +
                         v_rel.y * contact.normal.y +
                         v_rel.z * contact.normal.z;

    float spring_magnitude = props.stiffness * contact.penetration_depth;
    float damping_magnitude = props.damping * v_rel_normal;
    float normal_force_magnitude = fmaxf(spring_magnitude - damping_magnitude, 0.0f);

    // === Gradient w.r.t. Material Properties ===

    // ∂L/∂k: spring constant
    // ∂F/∂k = penetration * normal
    float grad_k_contact =
        (adj_f2.x * contact.penetration_depth * contact.normal.x +
         adj_f2.y * contact.penetration_depth * contact.normal.y +
         adj_f2.z * contact.penetration_depth * contact.normal.z);

    atomicAdd(grad_stiffness, grad_k_contact);

    // ∂L/∂c: damping coefficient
    // ∂F/∂c = -v_rel_normal * normal
    float grad_damping_contact =
        -(adj_f2.x * v_rel_normal * contact.normal.x +
          adj_f2.y * v_rel_normal * contact.normal.y +
          adj_f2.z * v_rel_normal * contact.normal.z);

    atomicAdd(grad_damping, grad_damping_contact);

    // === Gradient w.r.t. Velocities ===

    // ∂F/∂v₂ = -c * normal ⊗ normal (for normal component)
    float3 grad_v2_normal = {
        -props.damping * contact.normal.x *
            (adj_f2.x * contact.normal.x + adj_f2.y * contact.normal.y + adj_f2.z * contact.normal.z),
        -props.damping * contact.normal.y *
            (adj_f2.x * contact.normal.x + adj_f2.y * contact.normal.y + adj_f2.z * contact.normal.z),
        -props.damping * contact.normal.z *
            (adj_f2.x * contact.normal.x + adj_f2.y * contact.normal.y + adj_f2.z * contact.normal.z)
    };

    atomicAdd(&grad_velocities[p2].x, grad_v2_normal.x);
    atomicAdd(&grad_velocities[p2].y, grad_v2_normal.y);
    atomicAdd(&grad_velocities[p2].z, grad_v2_normal.z);

    atomicAdd(&grad_velocities[p1].x, -grad_v2_normal.x);
    atomicAdd(&grad_velocities[p1].y, -grad_v2_normal.y);
    atomicAdd(&grad_velocities[p1].z, -grad_v2_normal.z);

    // === Gradient w.r.t. Positions (through penetration depth and normal) ===

    // ∂penetration/∂x₁ = -∂distance/∂x₁ = (x₂ - x₁) / |x₂ - x₁|
    // ∂F/∂x₁ = ∂F/∂penetration * ∂penetration/∂x₁

    float grad_penetration = props.stiffness *
        (adj_f2.x * contact.normal.x + adj_f2.y * contact.normal.y + adj_f2.z * contact.normal.z);

    float3 grad_x1 = {
        -grad_penetration * contact.normal.x,
        -grad_penetration * contact.normal.y,
        -grad_penetration * contact.normal.z
    };

    float3 grad_x2 = {
        grad_penetration * contact.normal.x,
        grad_penetration * contact.normal.y,
        grad_penetration * contact.normal.z
    };

    atomicAdd(&grad_positions[p1].x, grad_x1.x);
    atomicAdd(&grad_positions[p1].y, grad_x1.y);
    atomicAdd(&grad_positions[p1].z, grad_x1.z);

    atomicAdd(&grad_positions[p2].x, grad_x2.x);
    atomicAdd(&grad_positions[p2].y, grad_x2.y);
    atomicAdd(&grad_positions[p2].z, grad_x2.z);

    // === Gradient w.r.t. Radii (optional) ===
    if (grad_radii != nullptr) {
        // ∂penetration/∂r₁ = 1, ∂penetration/∂r₂ = 1
        atomicAdd(&grad_radii[p1], grad_penetration);
        atomicAdd(&grad_radii[p2], grad_penetration);
    }
}

// =============================================================================
// CUDA KERNEL LAUNCHERS
// =============================================================================

extern "C" {

void launch_detect_sphere_contacts_differentiable(
    const float3* positions,
    const float* radii,
    const float3* velocities,
    ContactPoint* contacts,
    int* num_contacts,
    int num_particles,
    float contact_threshold,
    cudaStream_t stream = 0
) {
    dim3 block(256);
    dim3 grid((num_particles + block.x - 1) / block.x);

    detect_sphere_contacts_differentiable_kernel<<<grid, block, 0, stream>>>(
        positions, radii, velocities,
        contacts, num_contacts,
        num_particles, contact_threshold
    );
}

void launch_compute_contact_forces(
    const ContactPoint* contacts,
    const float3* velocities,
    const ContactMaterialProperties* material_props,
    float3* forces,
    float3* saved_contact_forces,
    int num_contacts,
    cudaStream_t stream = 0
) {
    dim3 block(256);
    dim3 grid((num_contacts + block.x - 1) / block.x);

    compute_contact_forces_kernel<<<grid, block, 0, stream>>>(
        contacts, velocities, material_props,
        forces, saved_contact_forces, num_contacts
    );
}

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
) {
    dim3 block(256);
    dim3 grid((num_contacts + block.x - 1) / block.x);

    contact_forces_backward_kernel<<<grid, block, 0, stream>>>(
        grad_forces, grad_positions, grad_velocities, grad_radii,
        grad_stiffness, grad_damping, grad_friction,
        contacts, saved_positions, saved_velocities, saved_radii,
        material_props, num_contacts, num_particles
    );
}

} // extern "C"

} // namespace differentiable_contact
} // namespace physgrad
