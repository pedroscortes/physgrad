/**
 * PhysGrad - Contact Mechanics CUDA Kernels
 *
 * CUDA kernels for collision detection, contact resolution, and friction handling.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace physgrad {

// Contact detection kernels
__global__ void detect_sphere_contacts_kernel(
    const float3* __restrict__ positions,
    const float* __restrict__ radii,
    const float3* __restrict__ velocities,
    int* __restrict__ contact_pairs,
    float3* __restrict__ contact_normals,
    float* __restrict__ contact_distances,
    int* __restrict__ num_contacts,
    int num_particles,
    float contact_threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    for (int j = i + 1; j < num_particles; ++j) {
        float3 r_ij = {
            positions[i].x - positions[j].x,
            positions[i].y - positions[j].y,
            positions[i].z - positions[j].z
        };

        float distance = sqrtf(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);
        float sum_radii = radii[i] + radii[j];

        if (distance < sum_radii + contact_threshold) {
            int contact_idx = atomicAdd(num_contacts, 1);
            if (contact_idx < 10000) { // Max contacts limit
                contact_pairs[contact_idx * 2] = i;
                contact_pairs[contact_idx * 2 + 1] = j;

                if (distance > 1e-10f) {
                    contact_normals[contact_idx] = {
                        r_ij.x / distance,
                        r_ij.y / distance,
                        r_ij.z / distance
                    };
                } else {
                    contact_normals[contact_idx] = {1.0f, 0.0f, 0.0f};
                }

                contact_distances[contact_idx] = sum_radii - distance;
            }
        }
    }
}

// Contact force resolution
__global__ void resolve_contact_forces_kernel(
    const int* __restrict__ contact_pairs,
    const float3* __restrict__ contact_normals,
    const float* __restrict__ contact_distances,
    const float3* __restrict__ velocities,
    const float* __restrict__ masses,
    float3* __restrict__ forces,
    int num_contacts,
    float contact_stiffness,
    float contact_damping
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contacts) return;

    int i = contact_pairs[idx * 2];
    int j = contact_pairs[idx * 2 + 1];

    float3 normal = contact_normals[idx];
    float penetration = contact_distances[idx];

    // Spring force
    float3 spring_force = {
        normal.x * contact_stiffness * penetration,
        normal.y * contact_stiffness * penetration,
        normal.z * contact_stiffness * penetration
    };

    // Damping force
    float3 rel_velocity = {
        velocities[i].x - velocities[j].x,
        velocities[i].y - velocities[j].y,
        velocities[i].z - velocities[j].z
    };

    float normal_velocity = rel_velocity.x * normal.x +
                           rel_velocity.y * normal.y +
                           rel_velocity.z * normal.z;

    float3 damping_force = {
        normal.x * contact_damping * normal_velocity,
        normal.y * contact_damping * normal_velocity,
        normal.z * contact_damping * normal_velocity
    };

    float3 total_force = {
        spring_force.x - damping_force.x,
        spring_force.y - damping_force.y,
        spring_force.z - damping_force.z
    };

    // Apply forces (Newton's 3rd law)
    atomicAdd(&forces[i].x, total_force.x);
    atomicAdd(&forces[i].y, total_force.y);
    atomicAdd(&forces[i].z, total_force.z);

    atomicAdd(&forces[j].x, -total_force.x);
    atomicAdd(&forces[j].y, -total_force.y);
    atomicAdd(&forces[j].z, -total_force.z);
}

// Friction force calculation
__global__ void calculate_friction_forces_kernel(
    const int* __restrict__ contact_pairs,
    const float3* __restrict__ contact_normals,
    const float3* __restrict__ velocities,
    const float* __restrict__ masses,
    float3* __restrict__ forces,
    int num_contacts,
    float friction_coefficient
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contacts) return;

    int i = contact_pairs[idx * 2];
    int j = contact_pairs[idx * 2 + 1];

    float3 normal = contact_normals[idx];

    // Relative velocity
    float3 rel_velocity = {
        velocities[i].x - velocities[j].x,
        velocities[i].y - velocities[j].y,
        velocities[i].z - velocities[j].z
    };

    // Normal component of relative velocity
    float normal_velocity = rel_velocity.x * normal.x +
                           rel_velocity.y * normal.y +
                           rel_velocity.z * normal.z;

    // Tangential velocity
    float3 tangential_velocity = {
        rel_velocity.x - normal_velocity * normal.x,
        rel_velocity.y - normal_velocity * normal.y,
        rel_velocity.z - normal_velocity * normal.z
    };

    float tangential_speed = sqrtf(
        tangential_velocity.x * tangential_velocity.x +
        tangential_velocity.y * tangential_velocity.y +
        tangential_velocity.z * tangential_velocity.z
    );

    if (tangential_speed > 1e-10f) {
        float3 tangential_direction = {
            tangential_velocity.x / tangential_speed,
            tangential_velocity.y / tangential_speed,
            tangential_velocity.z / tangential_speed
        };

        // Friction force magnitude (simplified)
        float friction_magnitude = friction_coefficient * 1000.0f; // Placeholder normal force

        float3 friction_force = {
            -tangential_direction.x * friction_magnitude,
            -tangential_direction.y * friction_magnitude,
            -tangential_direction.z * friction_magnitude
        };

        // Apply friction forces
        float mass_ratio_i = masses[j] / (masses[i] + masses[j]);
        float mass_ratio_j = masses[i] / (masses[i] + masses[j]);

        atomicAdd(&forces[i].x, friction_force.x * mass_ratio_i);
        atomicAdd(&forces[i].y, friction_force.y * mass_ratio_i);
        atomicAdd(&forces[i].z, friction_force.z * mass_ratio_i);

        atomicAdd(&forces[j].x, -friction_force.x * mass_ratio_j);
        atomicAdd(&forces[j].y, -friction_force.y * mass_ratio_j);
        atomicAdd(&forces[j].z, -friction_force.z * mass_ratio_j);
    }
}

// Broad phase collision detection using spatial hashing
__global__ void spatial_hash_kernel(
    const float3* __restrict__ positions,
    const float* __restrict__ radii,
    int* __restrict__ hash_values,
    int* __restrict__ particle_indices,
    int num_particles,
    float cell_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float3 pos = positions[idx];

    // Calculate hash based on grid cell
    int x_cell = (int)floorf(pos.x / cell_size);
    int y_cell = (int)floorf(pos.y / cell_size);
    int z_cell = (int)floorf(pos.z / cell_size);

    // Simple hash function
    int hash = ((x_cell * 73856093) ^ (y_cell * 19349663) ^ (z_cell * 83492791)) % 1000000;
    if (hash < 0) hash = -hash;

    hash_values[idx] = hash;
    particle_indices[idx] = idx;
}

// Contact constraint projection
__global__ void project_contact_constraints_kernel(
    const int* __restrict__ contact_pairs,
    const float3* __restrict__ contact_normals,
    const float* __restrict__ contact_distances,
    float3* __restrict__ positions,
    const float* __restrict__ masses,
    int num_contacts,
    float constraint_compliance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contacts) return;

    int i = contact_pairs[idx * 2];
    int j = contact_pairs[idx * 2 + 1];

    float3 normal = contact_normals[idx];
    float penetration = contact_distances[idx];

    if (penetration > 0.0f) {
        float total_mass = masses[i] + masses[j];
        float correction_magnitude = penetration / (total_mass * constraint_compliance + 1e-10f);

        float3 correction_i = {
            normal.x * correction_magnitude * masses[j] / total_mass,
            normal.y * correction_magnitude * masses[j] / total_mass,
            normal.z * correction_magnitude * masses[j] / total_mass
        };

        float3 correction_j = {
            -normal.x * correction_magnitude * masses[i] / total_mass,
            -normal.y * correction_magnitude * masses[i] / total_mass,
            -normal.z * correction_magnitude * masses[i] / total_mass
        };

        atomicAdd(&positions[i].x, correction_i.x);
        atomicAdd(&positions[i].y, correction_i.y);
        atomicAdd(&positions[i].z, correction_i.z);

        atomicAdd(&positions[j].x, correction_j.x);
        atomicAdd(&positions[j].y, correction_j.y);
        atomicAdd(&positions[j].z, correction_j.z);
    }
}

} // namespace physgrad