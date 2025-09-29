/**
 * PhysGrad - Core Physics CUDA Kernels
 *
 * Basic physics computation kernels shared across multiple systems.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace physgrad {

// Verlet integration kernel
__global__ void verlet_integration_kernel(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    const float3* __restrict__ forces,
    const float* __restrict__ masses,
    float dt,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float inv_mass = 1.0f / masses[idx];
    float3 acceleration = {
        forces[idx].x * inv_mass,
        forces[idx].y * inv_mass,
        forces[idx].z * inv_mass
    };

    // Verlet integration
    velocities[idx].x += acceleration.x * dt;
    velocities[idx].y += acceleration.y * dt;
    velocities[idx].z += acceleration.z * dt;

    positions[idx].x += velocities[idx].x * dt;
    positions[idx].y += velocities[idx].y * dt;
    positions[idx].z += velocities[idx].z * dt;
}

// Classical force computation (electrostatic + LJ)
__global__ void classical_force_kernel(
    const float3* __restrict__ positions,
    const float* __restrict__ charges,
    float3* __restrict__ forces,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float3 total_force = {0.0f, 0.0f, 0.0f};
    float3 pos_i = positions[i];
    float charge_i = charges[i];

    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        float3 pos_j = positions[j];
        float charge_j = charges[j];

        float3 r_ij = {
            pos_i.x - pos_j.x,
            pos_i.y - pos_j.y,
            pos_i.z - pos_j.z
        };

        float r2 = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z;
        float r = sqrtf(r2);

        if (r > 1e-10f) {  // Avoid division by zero
            // Coulomb force
            float coulomb_constant = 8.9875517923e9f;  // N⋅m²/C²
            float force_magnitude = coulomb_constant * charge_i * charge_j / r2;

            float3 force_dir = {r_ij.x / r, r_ij.y / r, r_ij.z / r};

            total_force.x += force_magnitude * force_dir.x;
            total_force.y += force_magnitude * force_dir.y;
            total_force.z += force_magnitude * force_dir.z;
        }
    }

    forces[i] = total_force;
}

// Energy calculation kernel
__global__ void calculate_energy_kernel(
    const float3* __restrict__ positions,
    const float3* __restrict__ velocities,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    float* __restrict__ kinetic_energy,
    float* __restrict__ potential_energy,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    // Kinetic energy
    float3 vel = velocities[i];
    float vel2 = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    kinetic_energy[i] = 0.5f * masses[i] * vel2;

    // Potential energy (only upper triangle to avoid double counting)
    float potential = 0.0f;
    float3 pos_i = positions[i];
    float charge_i = charges[i];

    for (int j = i + 1; j < num_particles; ++j) {
        float3 pos_j = positions[j];
        float charge_j = charges[j];

        float3 r_ij = {
            pos_i.x - pos_j.x,
            pos_i.y - pos_j.y,
            pos_i.z - pos_j.z
        };

        float r = sqrtf(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);

        if (r > 1e-10f) {
            float coulomb_constant = 8.9875517923e9f;
            potential += coulomb_constant * charge_i * charge_j / r;
        }
    }

    potential_energy[i] = potential;
}

} // namespace physgrad