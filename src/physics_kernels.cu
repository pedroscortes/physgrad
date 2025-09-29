// PhysGrad Core Physics CUDA Kernels

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

    // Calculate acceleration: a = F/m
    float inv_mass = 1.0f / masses[idx];
    float3 acceleration = {
        forces[idx].x * inv_mass,
        forces[idx].y * inv_mass,
        forces[idx].z * inv_mass
    };

    // Update position: x += v*dt + 0.5*a*dt^2 (proper Verlet)
    positions[idx].x += velocities[idx].x * dt + 0.5f * acceleration.x * dt * dt;
    positions[idx].y += velocities[idx].y * dt + 0.5f * acceleration.y * dt * dt;
    positions[idx].z += velocities[idx].z * dt + 0.5f * acceleration.z * dt * dt;

    // Update velocity: v += a*dt
    velocities[idx].x += acceleration.x * dt;
    velocities[idx].y += acceleration.y * dt;
    velocities[idx].z += acceleration.z * dt;

    // Verlet integration
    velocities[idx].x += acceleration.x * dt;
    velocities[idx].y += acceleration.y * dt;
    velocities[idx].z += acceleration.z * dt;

    positions[idx].x += velocities[idx].x * dt;
    positions[idx].y += velocities[idx].y * dt;
    positions[idx].z += velocities[idx].z * dt;
}

// Classical force computation
__global__ void classical_force_kernel(
    const float3* __restrict__ positions,
    const float* __restrict__ charges,
    float3* __restrict__ forces,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float3 total_force = {0.0f, 0.0f, 0.0f};

    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        // Direction vector from i to j (CORRECTED for proper force directions)
        float3 diff = {
            positions[j].x - positions[i].x,
            positions[j].y - positions[i].y,
            positions[j].z - positions[i].z
        };

        float distance = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

        // Avoid singularity
        if (distance < 1e-6f) continue;

        // Direction (normalized)
        float3 direction = {
            diff.x / distance,
            diff.y / distance,
            diff.z / distance
        };

        // Electrostatic force: F = k * q1 * q2 / r^2
        const float k_e = 8.9875517923e9f;
        float force_magnitude = k_e * charges[i] * charges[j] / (distance * distance);

        // Add force contribution
        total_force.x += force_magnitude * direction.x;
        total_force.y += force_magnitude * direction.y;
        total_force.z += force_magnitude * direction.z;
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

// Memory operations kernel - simple test that modifies data
__global__ void memory_operations_kernel(
    float* __restrict__ data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Simple operation: multiply each element by 2 and add index
    data[idx] = data[idx] * 2.0f + static_cast<float>(idx);
}

} // namespace physgrad