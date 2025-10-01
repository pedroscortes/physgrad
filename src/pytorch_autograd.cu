/**
 * PhysGrad - PyTorch Autograd CUDA Kernels Implementation
 *
 * CUDA kernel implementations for custom PyTorch autograd functions
 */

#include "pytorch_autograd.h"
#include "common_types.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace physgrad {
namespace pytorch {

// =============================================================================
// CUDA KERNEL IMPLEMENTATIONS
// =============================================================================

/**
 * CUDA kernel for MPM timestep forward pass
 */
__global__ void mpm_timestep_forward_kernel(
    float* positions,        // [N, 3] particle positions
    float* velocities,       // [N, 3] particle velocities
    const float* masses,     // [N] particle masses
    int num_particles,
    int grid_resolution,
    float dt,
    const float* gravity     // [3] gravity vector
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    // Grid-to-particle transfer (simplified)
    float3 old_pos = make_float3(
        positions[tid * 3 + 0],
        positions[tid * 3 + 1],
        positions[tid * 3 + 2]
    );

    float3 old_vel = make_float3(
        velocities[tid * 3 + 0],
        velocities[tid * 3 + 1],
        velocities[tid * 3 + 2]
    );

    // Apply forces (gravity)
    float3 acceleration = make_float3(
        gravity[0], gravity[1], gravity[2]
    );

    // Simple Euler integration
    float3 new_vel = make_float3(
        old_vel.x + acceleration.x * dt,
        old_vel.y + acceleration.y * dt,
        old_vel.z + acceleration.z * dt
    );

    float3 new_pos = make_float3(
        old_pos.x + new_vel.x * dt,
        old_pos.y + new_vel.y * dt,
        old_pos.z + new_vel.z * dt
    );

    // Update particle state
    positions[tid * 3 + 0] = new_pos.x;
    positions[tid * 3 + 1] = new_pos.y;
    positions[tid * 3 + 2] = new_pos.z;

    velocities[tid * 3 + 0] = new_vel.x;
    velocities[tid * 3 + 1] = new_vel.y;
    velocities[tid * 3 + 2] = new_vel.z;
}

/**
 * CUDA kernel for MPM timestep backward pass
 */
__global__ void mmp_timestep_backward_kernel(
    const float* grad_positions,    // [N, 3] gradient w.r.t output positions
    const float* grad_velocities,   // [N, 3] gradient w.r.t output velocities
    float* grad_input_positions,    // [N, 3] gradient w.r.t input positions
    float* grad_input_velocities,   // [N, 3] gradient w.r.t input velocities
    float* grad_masses,             // [N] gradient w.r.t masses
    float* grad_gravity,            // [3] gradient w.r.t gravity
    const float* saved_positions,   // [N, 3] saved input positions
    const float* saved_velocities,  // [N, 3] saved input velocities
    int num_particles,
    int grid_resolution,
    float dt
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    // Chain rule for backward propagation
    // For simplified Euler integration:
    // new_pos = old_pos + new_vel * dt
    // new_vel = old_vel + gravity * dt

    float3 grad_out_pos = make_float3(
        grad_positions[tid * 3 + 0],
        grad_positions[tid * 3 + 1],
        grad_positions[tid * 3 + 2]
    );

    float3 grad_out_vel = make_float3(
        grad_velocities[tid * 3 + 0],
        grad_velocities[tid * 3 + 1],
        grad_velocities[tid * 3 + 2]
    );

    // Gradient w.r.t input velocity: grad_new_pos * dt + grad_new_vel
    float3 grad_in_vel = make_float3(
        grad_out_pos.x * dt + grad_out_vel.x,
        grad_out_pos.y * dt + grad_out_vel.y,
        grad_out_pos.z * dt + grad_out_vel.z
    );

    // Gradient w.r.t input position: grad_new_pos
    float3 grad_in_pos = grad_out_pos;

    // Gradient w.r.t gravity: grad_new_vel * dt
    atomicAdd(&grad_gravity[0], grad_out_vel.x * dt);
    atomicAdd(&grad_gravity[1], grad_out_vel.y * dt);
    atomicAdd(&grad_gravity[2], grad_out_vel.z * dt);

    // Store gradients
    grad_input_positions[tid * 3 + 0] = grad_in_pos.x;
    grad_input_positions[tid * 3 + 1] = grad_in_pos.y;
    grad_input_positions[tid * 3 + 2] = grad_in_pos.z;

    grad_input_velocities[tid * 3 + 0] = grad_in_vel.x;
    grad_input_velocities[tid * 3 + 1] = grad_in_vel.y;
    grad_input_velocities[tid * 3 + 2] = grad_in_vel.z;

    // Mass gradient (simplified - zero for this simple example)
    grad_masses[tid] = 0.0f;
}

/**
 * CUDA kernel for G2P2G forward pass
 */
__global__ void g2p2g_forward_kernel(
    const float* particles,     // [N, feature_dim] particle data
    const float* grid,          // [Gx, Gy, Gz, feature_dim] grid data
    float* output,              // [N, feature_dim] output particle data
    int num_particles,
    int grid_size,
    int feature_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    // Simplified G2P2G transfer using trilinear interpolation
    for (int f = 0; f < feature_dim; ++f) {
        float accumulated_value = 0.0f;

        // Sample from 8 neighboring grid cells (simplified)
        for (int dx = 0; dx < 2; ++dx) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dz = 0; dz < 2; ++dz) {
                    int gx = (tid % grid_size) + dx;
                    int gy = ((tid / grid_size) % grid_size) + dy;
                    int gz = (tid / (grid_size * grid_size)) + dz;

                    if (gx < grid_size && gy < grid_size && gz < grid_size) {
                        int grid_idx = ((gz * grid_size + gy) * grid_size + gx) * feature_dim + f;
                        float weight = 0.125f; // Uniform weights for simplicity
                        accumulated_value += weight * grid[grid_idx];
                    }
                }
            }
        }

        output[tid * feature_dim + f] = accumulated_value + particles[tid * feature_dim + f];
    }
}

/**
 * CUDA kernel for G2P2G backward pass
 */
__global__ void g2p2g_backward_kernel(
    const float* grad_output,       // [N, feature_dim] gradient w.r.t output
    float* grad_particles,          // [N, feature_dim] gradient w.r.t particles
    float* grad_grid,               // [Gx, Gy, Gz, feature_dim] gradient w.r.t grid
    const float* saved_particles,   // [N, feature_dim] saved particle data
    const float* saved_grid,        // [Gx, Gy, Gz, feature_dim] saved grid data
    int num_particles,
    int grid_size,
    int feature_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    // Backward pass for G2P2G transfer
    for (int f = 0; f < feature_dim; ++f) {
        float grad_out = grad_output[tid * feature_dim + f];

        // Gradient w.r.t particles (direct contribution)
        grad_particles[tid * feature_dim + f] = grad_out;

        // Gradient w.r.t grid (scattered through interpolation weights)
        for (int dx = 0; dx < 2; ++dx) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dz = 0; dz < 2; ++dz) {
                    int gx = (tid % grid_size) + dx;
                    int gy = ((tid / grid_size) % grid_size) + dy;
                    int gz = (tid / (grid_size * grid_size)) + dz;

                    if (gx < grid_size && gy < grid_size && gz < grid_size) {
                        int grid_idx = ((gz * grid_size + gy) * grid_size + gx) * feature_dim + f;
                        float weight = 0.125f; // Uniform weights
                        atomicAdd(&grad_grid[grid_idx], weight * grad_out);
                    }
                }
            }
        }
    }
}

/**
 * CUDA kernel for particle update
 */
__global__ void particle_update_kernel(
    float* positions,               // [N, 3] particle positions
    float* velocities,              // [N, 3] particle velocities
    const float* forces,            // [N, 3] forces on particles
    const float* masses,            // [N] particle masses
    const bool* constraints,        // [N] constraint mask
    int num_particles,
    float dt
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    if (constraints[tid]) return; // Skip constrained particles

    float mass = masses[tid];
    if (mass <= 0.0f) return;

    // Update velocities using forces
    float3 acceleration = make_float3(
        forces[tid * 3 + 0] / mass,
        forces[tid * 3 + 1] / mass,
        forces[tid * 3 + 2] / mass
    );

    float3 old_vel = make_float3(
        velocities[tid * 3 + 0],
        velocities[tid * 3 + 1],
        velocities[tid * 3 + 2]
    );

    float3 new_vel = make_float3(
        old_vel.x + acceleration.x * dt,
        old_vel.y + acceleration.y * dt,
        old_vel.z + acceleration.z * dt
    );

    // Update positions using velocities
    float3 old_pos = make_float3(
        positions[tid * 3 + 0],
        positions[tid * 3 + 1],
        positions[tid * 3 + 2]
    );

    float3 new_pos = make_float3(
        old_pos.x + new_vel.x * dt,
        old_pos.y + new_vel.y * dt,
        old_pos.z + new_vel.z * dt
    );

    // Store results
    velocities[tid * 3 + 0] = new_vel.x;
    velocities[tid * 3 + 1] = new_vel.y;
    velocities[tid * 3 + 2] = new_vel.z;

    positions[tid * 3 + 0] = new_pos.x;
    positions[tid * 3 + 1] = new_pos.y;
    positions[tid * 3 + 2] = new_pos.z;
}

/**
 * CUDA kernel for energy computation
 */
__global__ void energy_compute_kernel(
    const float* positions,         // [N, 3] particle positions
    const float* velocities,        // [N, 3] particle velocities
    const float* masses,            // [N] particle masses
    const float* potential_params,  // [P] potential parameters
    float* energy_output,           // [1] total energy output
    int num_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float kinetic_energy[256];
    __shared__ float potential_energy[256];

    float local_kinetic = 0.0f;
    float local_potential = 0.0f;

    if (tid < num_particles) {
        // Compute kinetic energy for this particle
        float3 vel = make_float3(
            velocities[tid * 3 + 0],
            velocities[tid * 3 + 1],
            velocities[tid * 3 + 2]
        );

        float vel_sq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
        local_kinetic = 0.5f * masses[tid] * vel_sq;

        // Compute potential energy (simplified spring model)
        if (tid < num_particles - 1) {
            float3 pos1 = make_float3(
                positions[tid * 3 + 0],
                positions[tid * 3 + 1],
                positions[tid * 3 + 2]
            );

            float3 pos2 = make_float3(
                positions[(tid + 1) * 3 + 0],
                positions[(tid + 1) * 3 + 1],
                positions[(tid + 1) * 3 + 2]
            );

            float3 diff = make_float3(
                pos2.x - pos1.x,
                pos2.y - pos1.y,
                pos2.z - pos1.z
            );

            float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
            float spring_k = potential_params[0]; // Spring constant
            float rest_length = potential_params[1]; // Rest length

            local_potential = 0.5f * spring_k * (dist - rest_length) * (dist - rest_length);
        }
    }

    // Store in shared memory
    int local_tid = threadIdx.x;
    kinetic_energy[local_tid] = local_kinetic;
    potential_energy[local_tid] = local_potential;

    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            kinetic_energy[local_tid] += kinetic_energy[local_tid + stride];
            potential_energy[local_tid] += potential_energy[local_tid + stride];
        }
        __syncthreads();
    }

    // Block leader writes to global memory
    if (local_tid == 0) {
        atomicAdd(&energy_output[0], kinetic_energy[0] + potential_energy[0]);
    }
}

// =============================================================================
// CUDA KERNEL LAUNCHERS
// =============================================================================

extern "C" {

void launch_mpm_timestep_forward(
    float* positions, float* velocities, const float* masses,
    int num_particles, int grid_resolution, float dt,
    const float* gravity, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_particles + block.x - 1) / block.x);

    mmp_timestep_forward_kernel<<<grid, block, 0, stream>>>(
        positions, velocities, masses, num_particles, grid_resolution, dt, gravity
    );
}

void launch_mpm_timestep_backward(
    const float* grad_positions, const float* grad_velocities,
    float* grad_input_positions, float* grad_input_velocities,
    float* grad_masses, float* grad_gravity,
    const float* saved_positions, const float* saved_velocities,
    int num_particles, int grid_resolution, float dt,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_particles + block.x - 1) / block.x);

    mmp_timestep_backward_kernel<<<grid, block, 0, stream>>>(
        grad_positions, grad_velocities,
        grad_input_positions, grad_input_velocities,
        grad_masses, grad_gravity,
        saved_positions, saved_velocities,
        num_particles, grid_resolution, dt
    );
}

void launch_g2p2g_forward(
    const float* particles, const float* grid, float* output,
    int num_particles, int grid_size, int feature_dim,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_particles + block.x - 1) / block.x);

    g2p2g_forward_kernel<<<grid, block, 0, stream>>>(
        particles, grid, output, num_particles, grid_size, feature_dim
    );
}

void launch_g2p2g_backward(
    const float* grad_output,
    float* grad_particles, float* grad_grid,
    const float* saved_particles, const float* saved_grid,
    int num_particles, int grid_size, int feature_dim,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_particles + block.x - 1) / block.x);

    g2p2g_backward_kernel<<<grid, block, 0, stream>>>(
        grad_output, grad_particles, grad_grid,
        saved_particles, saved_grid,
        num_particles, grid_size, feature_dim
    );
}

void launch_particle_update(
    float* positions, float* velocities,
    const float* forces, const float* masses,
    const bool* constraints, int num_particles, float dt,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_particles + block.x - 1) / block.x);

    particle_update_kernel<<<grid, block, 0, stream>>>(
        positions, velocities, forces, masses, constraints,
        num_particles, dt
    );
}

void launch_energy_compute(
    const float* positions, const float* velocities,
    const float* masses, const float* potential_params,
    float* energy_output, int num_particles,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_particles + block.x - 1) / block.x);

    // Initialize energy output to zero
    cudaMemsetAsync(energy_output, 0, sizeof(float), stream);

    energy_compute_kernel<<<grid, block, 0, stream>>>(
        positions, velocities, masses, potential_params,
        energy_output, num_particles
    );
}

} // extern "C"

} // namespace pytorch
} // namespace physgrad