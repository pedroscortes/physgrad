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
    return make_float3(x, y, z);
}

__device__ __forceinline__ float length_safe(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ float dot_safe(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Fixed barrier potential functions that take parameters
__device__ __forceinline__ float computeBarrierPotentialParam(float signed_distance, float barrier_stiffness, float barrier_threshold) {
    if (signed_distance >= barrier_threshold) return 0.0f;

    if (signed_distance <= -barrier_threshold) {
        // Quadratic extension for deep penetration
        float excess = signed_distance + barrier_threshold;
        return barrier_stiffness * (excess * excess + 2.0f * barrier_threshold * excess);
    }

    // C∞ smooth barrier function in transition region
    float ratio = signed_distance / barrier_threshold;
    float barrier_fn = (1.0f - ratio) * (1.0f - ratio) * (1.0f - ratio);
    return barrier_stiffness * barrier_threshold * barrier_threshold * barrier_fn;
}

__device__ __forceinline__ float computeBarrierGradientParam(float signed_distance, float barrier_stiffness, float barrier_threshold) {
    if (signed_distance >= barrier_threshold) return 0.0f;

    if (signed_distance <= -barrier_threshold) {
        // Linear gradient for quadratic extension
        float excess = signed_distance + barrier_threshold;
        return 2.0f * barrier_stiffness * excess;
    }

    // Smooth gradient in transition region
    float ratio = signed_distance / barrier_threshold;
    float barrier_grad = -3.0f * (1.0f - ratio) * (1.0f - ratio) / barrier_threshold;
    return barrier_stiffness * barrier_threshold * barrier_threshold * barrier_grad;
}

__device__ __forceinline__ float computeBarrierHessianParam(float signed_distance, float barrier_stiffness, float barrier_threshold) {
    if (signed_distance >= barrier_threshold) return 0.0f;

    if (signed_distance <= -barrier_threshold) {
        // Constant Hessian for quadratic extension
        return 2.0f * barrier_stiffness;
    }

    // Smooth Hessian in transition region
    float ratio = signed_distance / barrier_threshold;
    float barrier_hess = 6.0f * (1.0f - ratio) / (barrier_threshold * barrier_threshold);
    return barrier_stiffness * barrier_threshold * barrier_threshold * barrier_hess;
}

// Spatial hash function
__device__ inline int computeSpatialHash(int3 cell_pos, int3 grid_size) {
    // Wrap around for periodic boundary conditions
    cell_pos.x = (cell_pos.x + grid_size.x) % grid_size.x;
    cell_pos.y = (cell_pos.y + grid_size.y) % grid_size.y;
    cell_pos.z = (cell_pos.z + grid_size.z) % grid_size.z;

    return cell_pos.z * grid_size.x * grid_size.y + cell_pos.y * grid_size.x + cell_pos.x;
}

// Compute spatial hash for each body
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

    // Load position
    float3 pos = make_float3(d_positions[3*idx], d_positions[3*idx+1], d_positions[3*idx+2]);

    // Compute cell coordinates
    int3 cell_pos;
    cell_pos.x = __float2int_rd((pos.x - world_min.x) / cell_size);
    cell_pos.y = __float2int_rd((pos.y - world_min.y) / cell_size);
    cell_pos.z = __float2int_rd((pos.z - world_min.z) / cell_size);

    // Compute hash
    d_hash_keys[idx] = computeSpatialHash(cell_pos, grid_size);
    d_hash_values[idx] = idx;
}

// Simple brute force contact detection (O(N^2) but reliable)
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;

    // Load position and radius for body i
    float3 pos_i = make_float3(d_positions[3*i], d_positions[3*i+1], d_positions[3*i+2]);
    float radius_i = d_radii[i];

    // Check against all other bodies (brute force for reliability)
    for (int j = i + 1; j < n_bodies; j++) {
        // Load position and radius for body j
        float3 pos_j = make_float3(d_positions[3*j], d_positions[3*j+1], d_positions[3*j+2]);
        float radius_j = d_radii[j];

        // Distance check
        float3 delta = make_float3(pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z);
        float dist = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
        float threshold = radius_i + radius_j + contact_threshold;

        if (dist < threshold) {
            // Add contact pair
            int pair_idx = atomicAdd(d_pair_count, 1);
            if (pair_idx < max_pairs) {
                d_contact_pairs[2 * pair_idx] = i;
                d_contact_pairs[2 * pair_idx + 1] = j;
            }
        }
    }
}

// Fixed barrier potential computation kernel
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

    // Compute contact normal
    float3 contact_normal = make_float3(0.0f, 0.0f, 1.0f);
    if (distance > 1e-8f) {
        contact_normal = make_float3(delta.x / distance, delta.y / distance, delta.z / distance);
    }

    // Contact point
    float3 contact_point = make_float3(
        pos_a.x + contact_normal.x * radius_a,
        pos_a.y + contact_normal.y * radius_a,
        pos_a.z + contact_normal.z * radius_a
    );

    // Compute barrier potential and derivatives using fixed functions
    float barrier_potential = computeBarrierPotentialParam(signed_distance, barrier_stiffness, barrier_threshold);
    float barrier_gradient = computeBarrierGradientParam(signed_distance, barrier_stiffness, barrier_threshold);
    float barrier_hessian = computeBarrierHessianParam(signed_distance, barrier_stiffness, barrier_threshold);

    // Only add to contact list if there's significant potential energy
    if (fabsf(barrier_potential) > 1e-10f || fabsf(barrier_gradient) > 1e-8f) {
        int contact_index = atomicAdd(d_contact_count, 1);
        if (contact_index < max_contacts) {
            GPUContactConstraint& contact = d_contacts[contact_index];

            contact.body_a = body_a;
            contact.body_b = body_j;
            contact.contact_point[0] = contact_point.x;
            contact.contact_point[1] = contact_point.y;
            contact.contact_point[2] = contact_point.z;
            contact.contact_normal[0] = contact_normal.x;
            contact.contact_normal[1] = contact_normal.y;
            contact.contact_normal[2] = contact_normal.z;
            contact.signed_distance = signed_distance;
            contact.barrier_potential = barrier_potential;
            contact.barrier_gradient = barrier_gradient;
            contact.barrier_hessian = barrier_hessian;

            // Material properties (simplified)
            contact.combined_friction_coeff = 0.3f;
            contact.combined_restitution = 0.5f;
            contact.combined_stiffness = barrier_stiffness;
        }
    }
}

// Compute contact forces from barrier potentials
__global__ void computeContactForcesKernel(
    const GPUContactConstraint* d_contacts,
    int contact_count,
    float* d_contact_forces,
    int n_bodies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= contact_count) return;

    const GPUContactConstraint& contact = d_contacts[idx];

    // Compute force magnitude from barrier gradient
    float force_magnitude = -contact.barrier_gradient;

    // Force direction is along contact normal
    float3 force = make_float3(
        force_magnitude * contact.contact_normal[0],
        force_magnitude * contact.contact_normal[1],
        force_magnitude * contact.contact_normal[2]
    );

    // Apply equal and opposite forces to both bodies (Newton's third law)
    int body_a = contact.body_a;
    int body_b = contact.body_b;

    // Body A gets positive force
    atomicAdd(&d_contact_forces[3 * body_a + 0], force.x);
    atomicAdd(&d_contact_forces[3 * body_a + 1], force.y);
    atomicAdd(&d_contact_forces[3 * body_a + 2], force.z);

    // Body B gets negative force
    atomicAdd(&d_contact_forces[3 * body_b + 0], -force.x);
    atomicAdd(&d_contact_forces[3 * body_b + 1], -force.y);
    atomicAdd(&d_contact_forces[3 * body_b + 2], -force.z);
}

// Compute total contact energy
__global__ void computeContactEnergyKernel(
    const GPUContactConstraint* d_contacts,
    int contact_count,
    float* d_contact_energy
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= contact_count) return;

    const GPUContactConstraint& contact = d_contacts[idx];
    atomicAdd(d_contact_energy, contact.barrier_potential);
}

// Reset array kernel
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

} // namespace physgrad