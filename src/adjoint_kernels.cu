/**
 * PhysGrad Adjoint Methods for CUDA Kernels
 *
 * Implements backward passes (adjoint methods) for all core CUDA kernels
 * to enable automatic differentiation through physics simulations.
 *
 * Key Principles:
 * 1. For each forward kernel, implement corresponding backward kernel
 * 2. Use chain rule to propagate gradients backward through computation
 * 3. Maintain numerical stability and efficiency comparable to forward pass
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace physgrad {
namespace adjoint {

// =============================================================================
// VERLET INTEGRATION ADJOINT
// =============================================================================

/**
 * Backward pass for Verlet integration
 *
 * Forward: x_new = x + v*dt + 0.5*a*dt^2, v_new = v + a*dt
 * Backward: Propagate gradients w.r.t. x_new, v_new back to x, v, forces
 */
__global__ void verlet_integration_backward_kernel(
    // Gradients w.r.t. outputs (input to backward pass)
    const float3* __restrict__ grad_positions_out,
    const float3* __restrict__ grad_velocities_out,

    // Gradients w.r.t. inputs (output of backward pass)
    float3* __restrict__ grad_positions_in,
    float3* __restrict__ grad_velocities_in,
    float3* __restrict__ grad_forces,
    float* __restrict__ grad_masses,

    // Saved values from forward pass
    const float3* __restrict__ saved_velocities,
    const float3* __restrict__ saved_forces,
    const float* __restrict__ saved_masses,

    float dt,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float inv_mass = 1.0f / saved_masses[idx];
    float dt_squared = dt * dt;

    // Chain rule for position update: x_new = x + v*dt + 0.5*a*dt^2
    // ∂L/∂x = ∂L/∂x_new * ∂x_new/∂x = ∂L/∂x_new * 1
    grad_positions_in[idx] = grad_positions_out[idx];

    // ∂L/∂v = ∂L/∂x_new * ∂x_new/∂v + ∂L/∂v_new * ∂v_new/∂v
    // ∂x_new/∂v = dt, ∂v_new/∂v = 1
    grad_velocities_in[idx].x = grad_positions_out[idx].x * dt + grad_velocities_out[idx].x;
    grad_velocities_in[idx].y = grad_positions_out[idx].y * dt + grad_velocities_out[idx].y;
    grad_velocities_in[idx].z = grad_positions_out[idx].z * dt + grad_velocities_out[idx].z;

    // ∂L/∂F = ∂L/∂x_new * ∂x_new/∂a * ∂a/∂F + ∂L/∂v_new * ∂v_new/∂a * ∂a/∂F
    // ∂x_new/∂a = 0.5*dt^2, ∂v_new/∂a = dt, ∂a/∂F = 1/m
    float grad_accel_x = grad_positions_out[idx].x * 0.5f * dt_squared + grad_velocities_out[idx].x * dt;
    float grad_accel_y = grad_positions_out[idx].y * 0.5f * dt_squared + grad_velocities_out[idx].y * dt;
    float grad_accel_z = grad_positions_out[idx].z * 0.5f * dt_squared + grad_velocities_out[idx].z * dt;

    grad_forces[idx].x = grad_accel_x * inv_mass;
    grad_forces[idx].y = grad_accel_y * inv_mass;
    grad_forces[idx].z = grad_accel_z * inv_mass;

    // ∂L/∂m = ∂L/∂a * ∂a/∂m = ∂L/∂a * (-F/m^2)
    float force_magnitude_sq = saved_forces[idx].x * saved_forces[idx].x +
                              saved_forces[idx].y * saved_forces[idx].y +
                              saved_forces[idx].z * saved_forces[idx].z;
    float grad_mass = -(grad_accel_x * saved_forces[idx].x +
                       grad_accel_y * saved_forces[idx].y +
                       grad_accel_z * saved_forces[idx].z) / (saved_masses[idx] * saved_masses[idx]);

    grad_masses[idx] = grad_mass;
}

// =============================================================================
// CLASSICAL FORCE COMPUTATION ADJOINT
// =============================================================================

/**
 * Backward pass for electrostatic force computation
 *
 * Forward: F_i = Σ_j k*q_i*q_j/r_ij^2 * (r_j - r_i)/|r_j - r_i|
 * Backward: Propagate gradients w.r.t. forces back to positions and charges
 */
__global__ void classical_force_backward_kernel(
    // Gradients w.r.t. outputs (input to backward pass)
    const float3* __restrict__ grad_forces,

    // Gradients w.r.t. inputs (output of backward pass)
    float3* __restrict__ grad_positions,
    float* __restrict__ grad_charges,

    // Saved values from forward pass
    const float3* __restrict__ saved_positions,
    const float* __restrict__ saved_charges,

    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    const float k_e = 8.9875517923e9f;
    float3 grad_pos_i = {0.0f, 0.0f, 0.0f};
    float grad_charge_i = 0.0f;

    // For each particle j that affects particle i
    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        float3 diff = {
            saved_positions[j].x - saved_positions[i].x,
            saved_positions[j].y - saved_positions[i].y,
            saved_positions[j].z - saved_positions[i].z
        };

        float distance = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        if (distance < 1e-6f) continue;

        float distance_cubed = distance * distance * distance;
        float force_coeff = k_e * saved_charges[i] * saved_charges[j];

        // Gradient w.r.t. position of particle i
        // ∂F_i/∂r_i = k*q_i*q_j * [2*(r_j-r_i)/r^3 - 3*(r_j-r_i)*(r_j-r_i)·∇r/r^4]
        float dot_product = diff.x * (-diff.x) + diff.y * (-diff.y) + diff.z * (-diff.z);
        float grad_coeff = force_coeff * (2.0f / distance_cubed - 3.0f * dot_product / (distance_cubed * distance * distance));

        grad_pos_i.x += grad_forces[i].x * grad_coeff * (-diff.x) / distance;
        grad_pos_i.y += grad_forces[i].y * grad_coeff * (-diff.y) / distance;
        grad_pos_i.z += grad_forces[i].z * grad_coeff * (-diff.z) / distance;

        // Gradient w.r.t. charge of particle i
        // ∂F_i/∂q_i = k*q_j/r^2 * (r_j - r_i)/|r_j - r_i|
        float charge_grad_coeff = k_e * saved_charges[j] / (distance * distance * distance);
        grad_charge_i += grad_forces[i].x * charge_grad_coeff * diff.x +
                        grad_forces[i].y * charge_grad_coeff * diff.y +
                        grad_forces[i].z * charge_grad_coeff * diff.z;
    }

    // Also compute gradient from forces that particle i exerts on other particles
    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        float3 diff = {
            saved_positions[i].x - saved_positions[j].x,
            saved_positions[i].y - saved_positions[j].y,
            saved_positions[i].z - saved_positions[j].z
        };

        float distance = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        if (distance < 1e-6f) continue;

        float distance_cubed = distance * distance * distance;
        float force_coeff = k_e * saved_charges[i] * saved_charges[j];

        // Gradient contribution from F_j w.r.t. r_i
        float dot_product = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        float grad_coeff = force_coeff * (2.0f / distance_cubed - 3.0f * dot_product / (distance_cubed * distance * distance));

        grad_pos_i.x += grad_forces[j].x * grad_coeff * diff.x / distance;
        grad_pos_i.y += grad_forces[j].y * grad_coeff * diff.y / distance;
        grad_pos_i.z += grad_forces[j].z * grad_coeff * diff.z / distance;

        // Gradient contribution from F_j w.r.t. q_i
        float charge_grad_coeff = k_e * saved_charges[j] / (distance * distance * distance);
        grad_charge_i += grad_forces[j].x * charge_grad_coeff * diff.x +
                        grad_forces[j].y * charge_grad_coeff * diff.y +
                        grad_forces[j].z * charge_grad_coeff * diff.z;
    }

    grad_positions[i] = grad_pos_i;
    grad_charges[i] = grad_charge_i;
}

// =============================================================================
// ENERGY CALCULATION ADJOINT
// =============================================================================

/**
 * Backward pass for energy calculation
 *
 * Forward: E = 0.5 * Σ_i m_i * v_i^2 (kinetic) + U(positions) (potential)
 * Backward: Propagate gradient of scalar energy back to velocities, masses, positions
 */
__global__ void calculate_energy_backward_kernel(
    // Gradient w.r.t. output energy (scalar)
    float grad_energy,

    // Gradients w.r.t. inputs (output of backward pass)
    float3* __restrict__ grad_velocities,
    float* __restrict__ grad_masses,
    float3* __restrict__ grad_positions,

    // Saved values from forward pass
    const float3* __restrict__ saved_velocities,
    const float* __restrict__ saved_masses,
    const float3* __restrict__ saved_positions,
    const float* __restrict__ saved_charges,

    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    // Kinetic energy gradients: E_kinetic = 0.5 * m * v^2
    // ∂E/∂v = m * v
    grad_velocities[i].x = grad_energy * saved_masses[i] * saved_velocities[i].x;
    grad_velocities[i].y = grad_energy * saved_masses[i] * saved_velocities[i].y;
    grad_velocities[i].z = grad_energy * saved_masses[i] * saved_velocities[i].z;

    // ∂E/∂m = 0.5 * v^2
    float velocity_squared = saved_velocities[i].x * saved_velocities[i].x +
                           saved_velocities[i].y * saved_velocities[i].y +
                           saved_velocities[i].z * saved_velocities[i].z;
    grad_masses[i] = grad_energy * 0.5f * velocity_squared;

    // Potential energy gradients: U = 0.5 * Σ_ij k*q_i*q_j/r_ij (i≠j)
    // ∂U/∂r_i = Σ_j≠i k*q_i*q_j * (-1/r_ij^2) * (r_i - r_j)/|r_i - r_j|
    const float k_e = 8.9875517923e9f;
    float3 grad_pos_i = {0.0f, 0.0f, 0.0f};

    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        float3 diff = {
            saved_positions[i].x - saved_positions[j].x,
            saved_positions[i].y - saved_positions[j].y,
            saved_positions[i].z - saved_positions[j].z
        };

        float distance = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        if (distance < 1e-6f) continue;

        float potential_grad_coeff = k_e * saved_charges[i] * saved_charges[j] / (distance * distance * distance);

        grad_pos_i.x += grad_energy * potential_grad_coeff * diff.x;
        grad_pos_i.y += grad_energy * potential_grad_coeff * diff.y;
        grad_pos_i.z += grad_energy * potential_grad_coeff * diff.z;
    }

    grad_positions[i] = grad_pos_i;
}

// =============================================================================
// SPH DENSITY AND PRESSURE ADJOINT
// =============================================================================

/**
 * Backward pass for SPH density computation
 * Forward: ρ_i = Σ_j m_j * W(r_i - r_j, h)
 */
__global__ void sph_density_backward_kernel(
    const float* __restrict__ grad_densities,
    float3* __restrict__ grad_positions,
    float* __restrict__ grad_masses,

    const float3* __restrict__ saved_positions,
    const float* __restrict__ saved_masses,
    float smoothing_length,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    const float h = smoothing_length;
    const float h_squared = h * h;
    float3 grad_pos_i = {0.0f, 0.0f, 0.0f};

    for (int j = 0; j < num_particles; ++j) {
        float3 diff = {
            saved_positions[i].x - saved_positions[j].x,
            saved_positions[i].y - saved_positions[j].y,
            saved_positions[i].z - saved_positions[j].z
        };

        float distance_squared = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

        if (distance_squared < h_squared && distance_squared > 1e-12f) {
            float distance = sqrtf(distance_squared);
            float q = distance / h;

            // Cubic spline kernel gradient
            float kernel_grad;
            if (q < 1.0f) {
                kernel_grad = -3.0f * q + 2.25f * q * q;
            } else if (q < 2.0f) {
                float term = 2.0f - q;
                kernel_grad = -0.75f * term * term;
            } else {
                kernel_grad = 0.0f;
            }

            float coeff = grad_densities[i] * saved_masses[j] * kernel_grad / (h * distance);
            grad_pos_i.x += coeff * diff.x;
            grad_pos_i.y += coeff * diff.y;
            grad_pos_i.z += coeff * diff.z;
        }

        // Mass gradient: ∂ρ_i/∂m_j = W(r_i - r_j, h)
        if (distance_squared < h_squared) {
            float distance = sqrtf(distance_squared);
            float q = distance / h;
            float kernel_value;

            if (q < 1.0f) {
                kernel_value = 1.0f - 1.5f * q * q + 0.75f * q * q * q;
            } else if (q < 2.0f) {
                float term = 2.0f - q;
                kernel_value = 0.25f * term * term * term;
            } else {
                kernel_value = 0.0f;
            }

            atomicAdd(&grad_masses[j], grad_densities[i] * kernel_value);
        }
    }

    grad_positions[i] = grad_pos_i;
}

// =============================================================================
// CUDA KERNEL LAUNCHERS
// =============================================================================

extern "C" {

void launch_verlet_integration_backward(
    const float3* grad_positions_out, const float3* grad_velocities_out,
    float3* grad_positions_in, float3* grad_velocities_in,
    float3* grad_forces, float* grad_masses,
    const float3* saved_velocities, const float3* saved_forces,
    const float* saved_masses, float dt, int num_particles,
    cudaStream_t stream = 0
) {
    dim3 block(256);
    dim3 grid((num_particles + block.x - 1) / block.x);

    verlet_integration_backward_kernel<<<grid, block, 0, stream>>>(
        grad_positions_out, grad_velocities_out,
        grad_positions_in, grad_velocities_in,
        grad_forces, grad_masses,
        saved_velocities, saved_forces, saved_masses,
        dt, num_particles
    );
}

void launch_classical_force_backward(
    const float3* grad_forces,
    float3* grad_positions, float* grad_charges,
    const float3* saved_positions, const float* saved_charges,
    int num_particles, cudaStream_t stream = 0
) {
    dim3 block(256);
    dim3 grid((num_particles + block.x - 1) / block.x);

    classical_force_backward_kernel<<<grid, block, 0, stream>>>(
        grad_forces, grad_positions, grad_charges,
        saved_positions, saved_charges, num_particles
    );
}

void launch_energy_backward(
    float grad_energy,
    float3* grad_velocities, float* grad_masses, float3* grad_positions,
    const float3* saved_velocities, const float* saved_masses,
    const float3* saved_positions, const float* saved_charges,
    int num_particles, cudaStream_t stream = 0
) {
    dim3 block(256);
    dim3 grid((num_particles + block.x - 1) / block.x);

    calculate_energy_backward_kernel<<<grid, block, 0, stream>>>(
        grad_energy, grad_velocities, grad_masses, grad_positions,
        saved_velocities, saved_masses, saved_positions, saved_charges,
        num_particles
    );
}

void launch_sph_density_backward(
    const float* grad_densities,
    float3* grad_positions, float* grad_masses,
    const float3* saved_positions, const float* saved_masses,
    float smoothing_length, int num_particles,
    cudaStream_t stream = 0
) {
    dim3 block(256);
    dim3 grid((num_particles + block.x - 1) / block.x);

    sph_density_backward_kernel<<<grid, block, 0, stream>>>(
        grad_densities, grad_positions, grad_masses,
        saved_positions, saved_masses,
        smoothing_length, num_particles
    );
}

} // extern "C"

} // namespace adjoint
} // namespace physgrad