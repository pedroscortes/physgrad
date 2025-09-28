#include "variational_contact_gpu.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

namespace physgrad {

// Device constants for optimal performance
__constant__ float c_barrier_stiffness;
__constant__ float c_barrier_threshold;
__constant__ float c_friction_regularization;
__constant__ int c_max_newton_iterations;
__constant__ float c_newton_tolerance;

// Utility device functions
__device__ __forceinline__ float3 make_float3_safe(float x, float y, float z) {
    return make_float3(
        isfinite(x) ? x : 0.0f,
        isfinite(y) ? y : 0.0f,
        isfinite(z) ? z : 0.0f
    );
}

__device__ __forceinline__ float length_safe(float3 v) {
    float len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    return len_sq > 1e-12f ? sqrtf(len_sq) : 0.0f;
}

// Optimized barrier potential computation (C∞ smooth)
__device__ __forceinline__ float computeBarrierPotential(float signed_distance) {
    if (signed_distance >= c_barrier_threshold) return 0.0f;

    if (signed_distance <= -c_barrier_threshold) {
        // Quadratic extension for deep penetration (stability)
        float excess = signed_distance + c_barrier_threshold;
        return c_barrier_stiffness * excess * excess;
    }

    // Main barrier potential: Φ(d) = -κδ²ln(d/δ + 1)
    float normalized_distance = signed_distance / c_barrier_threshold + 1.0f;
    return -c_barrier_stiffness * c_barrier_threshold * c_barrier_threshold * logf(normalized_distance);
}

__device__ __forceinline__ float computeBarrierGradient(float signed_distance) {
    if (signed_distance >= c_barrier_threshold) return 0.0f;

    if (signed_distance <= -c_barrier_threshold) {
        // Linear gradient for quadratic extension
        float excess = signed_distance + c_barrier_threshold;
        return 2.0f * c_barrier_stiffness * excess;
    }

    // Main barrier gradient: dΦ/dd = -κδ² / (d + δ)
    float denominator = signed_distance + c_barrier_threshold;
    return -c_barrier_stiffness * c_barrier_threshold * c_barrier_threshold / denominator;
}

__device__ __forceinline__ float computeBarrierHessian(float signed_distance) {
    if (signed_distance >= c_barrier_threshold) return 0.0f;

    if (signed_distance <= -c_barrier_threshold) {
        // Constant Hessian for quadratic extension
        return 2.0f * c_barrier_stiffness;
    }

    // Main barrier Hessian: d²Φ/dd² = κδ² / (d + δ)²
    float denominator = signed_distance + c_barrier_threshold;
    return c_barrier_stiffness * c_barrier_threshold * c_barrier_threshold / (denominator * denominator);
}

// Spatial hash kernel for efficient contact detection
__global__ void computeSpatialHashKernel(
    const float* d_positions,
    const float* d_radii,
    int* d_hash_keys,
    int* d_hash_values,
    int n_bodies,
    float cell_size,
    float3 world_min,
    int3 grid_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bodies) return;

    // Load body position and radius
    float3 pos = make_float3(
        d_positions[3 * idx + 0],
        d_positions[3 * idx + 1],
        d_positions[3 * idx + 2]
    );
    float radius = d_radii[idx];

    // Compute grid coordinates with safety bounds checking
    int3 grid_pos = make_int3(
        __float2int_rd((pos.x - world_min.x) / cell_size),
        __float2int_rd((pos.y - world_min.y) / cell_size),
        __float2int_rd((pos.z - world_min.z) / cell_size)
    );

    // Clamp to grid bounds
    grid_pos.x = max(0, min(grid_size.x - 1, grid_pos.x));
    grid_pos.y = max(0, min(grid_size.y - 1, grid_pos.y));
    grid_pos.z = max(0, min(grid_size.z - 1, grid_pos.z));

    // Compute hash (simple 3D to 1D mapping)
    int hash_key = grid_pos.z * (grid_size.x * grid_size.y) +
                   grid_pos.y * grid_size.x +
                   grid_pos.x;

    d_hash_keys[idx] = hash_key;
    d_hash_values[idx] = idx;
}

// Contact pair detection kernel using spatial hash
__global__ void detectContactPairsKernel(
    const float* d_positions,
    const float* d_radii,
    const int* d_material_ids,
    const int* d_cell_starts,
    const int* d_cell_ends,
    const int* d_hash_keys,
    int* d_contact_pairs,
    int* d_pair_count,
    int n_bodies,
    int max_pairs,
    float contact_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bodies) return;

    // Load body data
    float3 pos_i = make_float3(
        d_positions[3 * idx + 0],
        d_positions[3 * idx + 1],
        d_positions[3 * idx + 2]
    );
    float radius_i = d_radii[idx];
    int material_i = d_material_ids[idx];
    int hash_i = d_hash_keys[idx];

    // Check current cell and 26 neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int neighbor_hash = hash_i + dx + dy * 64 + dz * (64 * 64); // Assuming 64x64x64 grid

                if (neighbor_hash < 0) continue;

                int cell_start = d_cell_starts[neighbor_hash];
                int cell_end = d_cell_ends[neighbor_hash];

                for (int j_idx = cell_start; j_idx < cell_end; j_idx++) {
                    if (j_idx <= idx) continue; // Avoid duplicate pairs and self-interaction

                    // Load neighbor body data
                    float3 pos_j = make_float3(
                        d_positions[3 * j_idx + 0],
                        d_positions[3 * j_idx + 1],
                        d_positions[3 * j_idx + 2]
                    );
                    float radius_j = d_radii[j_idx];
                    int material_j = d_material_ids[j_idx];

                    // Check for potential contact
                    float3 delta = make_float3(pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z);
                    float distance = length_safe(delta);
                    float contact_distance = radius_i + radius_j + contact_threshold;

                    if (distance < contact_distance) {
                        // Add contact pair atomically
                        int pair_index = atomicAdd(d_pair_count, 1);
                        if (pair_index < max_pairs) {
                            d_contact_pairs[2 * pair_index + 0] = idx;
                            d_contact_pairs[2 * pair_index + 1] = j_idx;
                        }
                    }
                }
            }
        }
    }
}

// Barrier potential computation kernel with full contact constraint setup
__global__ void computeBarrierPotentialsKernel(
    const float* d_positions,
    const float* d_radii,
    const int* d_material_ids,
    const int* d_contact_pairs,
    int pair_count,
    GPUContactConstraint* d_contacts,
    int* d_contact_count,
    float barrier_stiffness,
    float barrier_threshold,
    int max_contacts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pair_count) return;

    // Load contact pair
    int body_a = d_contact_pairs[2 * idx + 0];
    int body_j = d_contact_pairs[2 * idx + 1];

    // Load body positions and radii
    float3 pos_a = make_float3(
        d_positions[3 * body_a + 0],
        d_positions[3 * body_a + 1],
        d_positions[3 * body_a + 2]
    );
    float3 pos_j = make_float3(
        d_positions[3 * body_j + 0],
        d_positions[3 * body_j + 1],
        d_positions[3 * body_j + 2]
    );
    float radius_a = d_radii[body_a];
    float radius_j = d_radii[body_j];

    // Compute contact geometry
    float3 delta = make_float3(pos_j.x - pos_a.x, pos_j.y - pos_a.y, pos_j.z - pos_a.z);
    float distance = length_safe(delta);

    // Skip if bodies are too far apart
    if (distance > radius_a + radius_j + barrier_threshold) return;

    float signed_distance = distance - (radius_a + radius_j);

    // Compute contact normal (normalized delta)
    float3 contact_normal = make_float3(0.0f, 0.0f, 1.0f); // Default normal
    if (distance > 1e-8f) {
        contact_normal = make_float3(delta.x / distance, delta.y / distance, delta.z / distance);
    }

    // Contact point (midpoint of closest approach)
    float3 contact_point = make_float3(
        pos_a.x + contact_normal.x * radius_a,
        pos_a.y + contact_normal.y * radius_a,
        pos_a.z + contact_normal.z * radius_a
    );

    // Compute barrier potential and derivatives
    float barrier_potential = computeBarrierPotential(signed_distance);
    float barrier_gradient = computeBarrierGradient(signed_distance);
    float barrier_hessian = computeBarrierHessian(signed_distance);

    // Only add to contact list if there's significant potential energy
    if (fabsf(barrier_potential) > 1e-10f || fabsf(barrier_gradient) > 1e-8f) {
        int contact_index = atomicAdd(d_contact_count, 1);
        if (contact_index < max_contacts) {
            GPUContactConstraint& contact = d_contacts[contact_index];

            // Set basic contact data
            contact.body_a = body_a;
            contact.body_b = body_j;
            contact.signed_distance = signed_distance;
            contact.barrier_potential = barrier_potential;
            contact.barrier_gradient = barrier_gradient;
            contact.barrier_hessian = barrier_hessian;

            // Set contact geometry
            contact.contact_point[0] = contact_point.x;
            contact.contact_point[1] = contact_point.y;
            contact.contact_point[2] = contact_point.z;
            contact.contact_normal[0] = contact_normal.x;
            contact.contact_normal[1] = contact_normal.y;
            contact.contact_normal[2] = contact_normal.z;

            // Compute orthonormal tangent basis for friction
            float3 tangent1, tangent2;
            if (fabsf(contact_normal.z) < 0.9f) {
                tangent1 = make_float3(-contact_normal.y, contact_normal.x, 0.0f);
            } else {
                tangent1 = make_float3(0.0f, -contact_normal.z, contact_normal.y);
            }
            float t1_len = length_safe(tangent1);
            if (t1_len > 1e-8f) {
                tangent1.x /= t1_len;
                tangent1.y /= t1_len;
                tangent1.z /= t1_len;
            }

            // Second tangent via cross product
            tangent2 = make_float3(
                contact_normal.y * tangent1.z - contact_normal.z * tangent1.y,
                contact_normal.z * tangent1.x - contact_normal.x * tangent1.z,
                contact_normal.x * tangent1.y - contact_normal.y * tangent1.x
            );

            contact.tangent_basis[0] = tangent1.x;
            contact.tangent_basis[1] = tangent1.y;
            contact.tangent_basis[2] = tangent1.z;
            contact.tangent_basis[3] = tangent2.x;
            contact.tangent_basis[4] = tangent2.y;
            contact.tangent_basis[5] = tangent2.z;

            // Initialize Lagrange multipliers
            contact.friction_multiplier[0] = 0.0f;
            contact.friction_multiplier[1] = 0.0f;
            contact.normal_multiplier = 0.0f;

            // Material properties (simple lookup for now)
            contact.combined_friction_coeff = 0.3f;
            contact.combined_restitution = 0.8f;
            contact.combined_stiffness = barrier_stiffness;
        }
    }
}

// Contact force computation kernel
__global__ void computeContactForcesKernel(
    const GPUContactConstraint* d_contacts,
    int contact_count,
    float* d_contact_forces,
    int n_bodies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= contact_count) return;

    const GPUContactConstraint& contact = d_contacts[idx];

    // Compute contact force magnitude from barrier gradient
    float force_magnitude = contact.barrier_gradient;

    // Contact force vector (along normal)
    float3 force = make_float3(
        force_magnitude * contact.contact_normal[0],
        force_magnitude * contact.contact_normal[1],
        force_magnitude * contact.contact_normal[2]
    );

    // Apply forces to both bodies (Newton's third law)
    atomicAdd(&d_contact_forces[3 * contact.body_a + 0], -force.x);
    atomicAdd(&d_contact_forces[3 * contact.body_a + 1], -force.y);
    atomicAdd(&d_contact_forces[3 * contact.body_a + 2], -force.z);

    atomicAdd(&d_contact_forces[3 * contact.body_b + 0], force.x);
    atomicAdd(&d_contact_forces[3 * contact.body_b + 1], force.y);
    atomicAdd(&d_contact_forces[3 * contact.body_b + 2], force.z);
}

// Contact energy computation kernel
__global__ void computeContactEnergyKernel(
    const GPUContactConstraint* d_contacts,
    int contact_count,
    float* d_contact_energy
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= contact_count) return;

    const GPUContactConstraint& contact = d_contacts[idx];

    // Each thread computes its contribution to total energy
    float thread_energy = contact.barrier_potential;

    // Use shared memory for efficient reduction within block
    __shared__ float shared_energy[256];
    int tid = threadIdx.x;
    shared_energy[tid] = (idx < contact_count) ? thread_energy : 0.0f;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_energy[tid] += shared_energy[tid + stride];
        }
        __syncthreads();
    }

    // First thread in block writes partial sum to global memory
    if (tid == 0) {
        atomicAdd(d_contact_energy, shared_energy[0]);
    }
}

// Newton-Raphson residual assembly kernel
__global__ void assembleNewtonResidualKernel(
    const float* d_positions,
    const float* d_velocities,
    const float* d_masses,
    const GPUContactConstraint* d_contacts,
    int contact_count,
    const float* d_positions_initial,
    const float* d_velocities_initial,
    const float* d_external_forces,
    float dt,
    float* d_newton_residual,
    int n_bodies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bodies) return;

    // Load current and initial state
    float3 pos = make_float3(
        d_positions[3 * idx + 0],
        d_positions[3 * idx + 1],
        d_positions[3 * idx + 2]
    );
    float3 vel = make_float3(
        d_velocities[3 * idx + 0],
        d_velocities[3 * idx + 1],
        d_velocities[3 * idx + 2]
    );
    float3 pos_init = make_float3(
        d_positions_initial[3 * idx + 0],
        d_positions_initial[3 * idx + 1],
        d_positions_initial[3 * idx + 2]
    );
    float3 vel_init = make_float3(
        d_velocities_initial[3 * idx + 0],
        d_velocities_initial[3 * idx + 1],
        d_velocities_initial[3 * idx + 2]
    );
    float mass = d_masses[idx];

    // External forces
    float3 f_ext = make_float3(0.0f, 0.0f, 0.0f);
    if (d_external_forces != nullptr) {
        f_ext = make_float3(
            d_external_forces[3 * idx + 0],
            d_external_forces[3 * idx + 1],
            d_external_forces[3 * idx + 2]
        );
    }

    // Compute contact forces on this body
    float3 f_contact = make_float3(0.0f, 0.0f, 0.0f);
    for (int c = 0; c < contact_count; c++) {
        const GPUContactConstraint& contact = d_contacts[c];
        if (contact.body_a == idx) {
            f_contact.x -= contact.barrier_gradient * contact.contact_normal[0];
            f_contact.y -= contact.barrier_gradient * contact.contact_normal[1];
            f_contact.z -= contact.barrier_gradient * contact.contact_normal[2];
        } else if (contact.body_b == idx) {
            f_contact.x += contact.barrier_gradient * contact.contact_normal[0];
            f_contact.y += contact.barrier_gradient * contact.contact_normal[1];
            f_contact.z += contact.barrier_gradient * contact.contact_normal[2];
        }
    }

    // Newton residual for velocity: R_v = v - v_0 - (f_ext + f_contact) / m * dt
    float3 residual_v = make_float3(
        vel.x - vel_init.x - (f_ext.x + f_contact.x) / mass * dt,
        vel.y - vel_init.y - (f_ext.y + f_contact.y) / mass * dt,
        vel.z - vel_init.z - (f_ext.z + f_contact.z) / mass * dt
    );

    // Newton residual for position: R_p = p - p_0 - v * dt
    float3 residual_p = make_float3(
        pos.x - pos_init.x - vel.x * dt,
        pos.y - pos_init.y - vel.y * dt,
        pos.z - pos_init.z - vel.z * dt
    );

    // Write residuals to global memory
    d_newton_residual[6 * idx + 0] = residual_v.x;
    d_newton_residual[6 * idx + 1] = residual_v.y;
    d_newton_residual[6 * idx + 2] = residual_v.z;
    d_newton_residual[6 * idx + 3] = residual_p.x;
    d_newton_residual[6 * idx + 4] = residual_p.y;
    d_newton_residual[6 * idx + 5] = residual_p.z;
}

// Gradient computation kernels for adjoint method
__global__ void computePositionGradientsKernel(
    const float* d_positions,
    const float* d_velocities,
    const float* d_radii,
    const int* d_material_ids,
    const GPUContactConstraint* d_contacts,
    int contact_count,
    const float* d_output_gradients,
    float* d_position_gradients,
    int n_bodies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bodies) return;

    // Initialize gradients
    float3 grad_pos = make_float3(
        d_output_gradients[3 * idx + 0],
        d_output_gradients[3 * idx + 1],
        d_output_gradients[3 * idx + 2]
    );

    // Accumulate gradients from contact constraints
    for (int c = 0; c < contact_count; c++) {
        const GPUContactConstraint& contact = d_contacts[c];

        if (contact.body_a == idx || contact.body_b == idx) {
            // Load contact geometry
            float3 normal = make_float3(
                contact.contact_normal[0],
                contact.contact_normal[1],
                contact.contact_normal[2]
            );

            // Gradient of barrier potential w.r.t. position
            float grad_potential_wrt_distance = contact.barrier_gradient;

            // Chain rule: ∂Φ/∂x = ∂Φ/∂d * ∂d/∂x
            float sign = (contact.body_a == idx) ? -1.0f : 1.0f;
            grad_pos.x += sign * grad_potential_wrt_distance * normal.x;
            grad_pos.y += sign * grad_potential_wrt_distance * normal.y;
            grad_pos.z += sign * grad_potential_wrt_distance * normal.z;
        }
    }

    // Write gradients back
    d_position_gradients[3 * idx + 0] = grad_pos.x;
    d_position_gradients[3 * idx + 1] = grad_pos.y;
    d_position_gradients[3 * idx + 2] = grad_pos.z;
}

// Velocity gradient computation kernel
__global__ void computeVelocityGradientsKernel(
    const float* d_positions,
    const float* d_velocities,
    const GPUContactConstraint* d_contacts,
    int contact_count,
    const float* d_output_gradients,
    float* d_velocity_gradients,
    int n_bodies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bodies) return;

    // For the variational contact formulation, velocity gradients are typically
    // pass-through for position-based contact models
    d_velocity_gradients[3 * idx + 0] = d_output_gradients[3 * idx + 0];
    d_velocity_gradients[3 * idx + 1] = d_output_gradients[3 * idx + 1];
    d_velocity_gradients[3 * idx + 2] = d_output_gradients[3 * idx + 2];

    // Add friction-related velocity gradients if needed
    // (This would require more complex computation for velocity-dependent friction)
}

// Utility kernels
__global__ void resetArrayKernel(float* array, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = value;
    }
}

__global__ void resetIntArrayKernel(int* array, int value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = value;
    }
}

// Performance monitoring kernels
__global__ void reduceMaxVelocityKernel(
    const float* d_velocities,
    float* d_max_velocity,
    int n_bodies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute velocity magnitude for this thread
    float vel_mag = 0.0f;
    if (idx < n_bodies) {
        float3 vel = make_float3(
            d_velocities[3 * idx + 0],
            d_velocities[3 * idx + 1],
            d_velocities[3 * idx + 2]
        );
        vel_mag = length_safe(vel);
    }

    // Block-level reduction to find maximum
    __shared__ float shared_max[256];
    int tid = threadIdx.x;
    shared_max[tid] = vel_mag;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)d_max_velocity, __float_as_int(shared_max[0]));
    }
}

__global__ void reduceTotalEnergyKernel(
    const float* d_velocities,
    const float* d_masses,
    const float* d_contact_energy,
    float* d_total_energy,
    int n_bodies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute kinetic energy for this thread
    float kinetic_energy = 0.0f;
    if (idx < n_bodies) {
        float3 vel = make_float3(
            d_velocities[3 * idx + 0],
            d_velocities[3 * idx + 1],
            d_velocities[3 * idx + 2]
        );
        float mass = d_masses[idx];
        kinetic_energy = 0.5f * mass * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
    }

    // Block-level reduction
    __shared__ float shared_energy[256];
    int tid = threadIdx.x;
    shared_energy[tid] = kinetic_energy;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_energy[tid] += shared_energy[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_total_energy, shared_energy[0]);
        // Add contact potential energy (computed separately)
        if (blockIdx.x == 0) {
            atomicAdd(d_total_energy, *d_contact_energy);
        }
    }
}

} // namespace physgrad