/**
 * PhysGrad - Fluid Dynamics CUDA Kernels
 *
 * CUDA kernels for smoothed particle hydrodynamics (SPH) and fluid simulation.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace physgrad {

// SPH smoothing kernels
__device__ float poly6_kernel(float r, float h) {
    if (r >= h) return 0.0f;

    float h2 = h * h;
    float h9 = h2 * h2 * h2 * h2 * h;
    float q = h2 - r * r;

    return (315.0f / (64.0f * M_PI * h9)) * q * q * q;
}

__device__ float3 spiky_gradient_kernel(float3 r_vec, float r, float h) {
    if (r >= h || r < 1e-10f) return {0.0f, 0.0f, 0.0f};

    float h6 = h * h * h * h * h * h;
    float q = h - r;
    float factor = -45.0f / (M_PI * h6) * q * q / r;

    return {
        factor * r_vec.x,
        factor * r_vec.y,
        factor * r_vec.z
    };
}

__device__ float viscosity_laplacian_kernel(float r, float h) {
    if (r >= h) return 0.0f;

    float h6 = h * h * h * h * h * h;
    return 45.0f / (M_PI * h6) * (h - r);
}

// Density calculation kernel
__global__ void calculate_density_kernel(
    const float3* __restrict__ positions,
    float* __restrict__ densities,
    const float* __restrict__ masses,
    int num_particles,
    float smoothing_length
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float density = 0.0f;
    float3 pos_i = positions[i];

    for (int j = 0; j < num_particles; ++j) {
        float3 pos_j = positions[j];
        float3 r_ij = {
            pos_i.x - pos_j.x,
            pos_i.y - pos_j.y,
            pos_i.z - pos_j.z
        };

        float r = sqrtf(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);

        if (r < smoothing_length) {
            density += masses[j] * poly6_kernel(r, smoothing_length);
        }
    }

    densities[i] = fmaxf(density, 1e-10f); // Avoid division by zero
}

// Pressure calculation kernel
__global__ void calculate_pressure_kernel(
    const float* __restrict__ densities,
    float* __restrict__ pressures,
    int num_particles,
    float rest_density,
    float gas_constant
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    // Tait equation of state
    pressures[i] = gas_constant * (densities[i] - rest_density);
    pressures[i] = fmaxf(pressures[i], 0.0f); // Pressure cannot be negative
}

// Pressure force calculation kernel
__global__ void calculate_pressure_forces_kernel(
    const float3* __restrict__ positions,
    const float* __restrict__ densities,
    const float* __restrict__ pressures,
    const float* __restrict__ masses,
    float3* __restrict__ forces,
    int num_particles,
    float smoothing_length
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float3 pressure_force = {0.0f, 0.0f, 0.0f};
    float3 pos_i = positions[i];
    float density_i = densities[i];
    float pressure_i = pressures[i];

    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        float3 pos_j = positions[j];
        float3 r_ij = {
            pos_i.x - pos_j.x,
            pos_i.y - pos_j.y,
            pos_i.z - pos_j.z
        };

        float r = sqrtf(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);

        if (r < smoothing_length) {
            float density_j = densities[j];
            float pressure_j = pressures[j];

            float3 gradient = spiky_gradient_kernel(r_ij, r, smoothing_length);
            float pressure_term = (pressure_i + pressure_j) / (2.0f * density_j);

            pressure_force.x -= masses[j] * pressure_term * gradient.x;
            pressure_force.y -= masses[j] * pressure_term * gradient.y;
            pressure_force.z -= masses[j] * pressure_term * gradient.z;
        }
    }

    atomicAdd(&forces[i].x, pressure_force.x);
    atomicAdd(&forces[i].y, pressure_force.y);
    atomicAdd(&forces[i].z, pressure_force.z);
}

// Viscosity force calculation kernel
__global__ void calculate_viscosity_forces_kernel(
    const float3* __restrict__ positions,
    const float3* __restrict__ velocities,
    const float* __restrict__ densities,
    const float* __restrict__ masses,
    float3* __restrict__ forces,
    int num_particles,
    float smoothing_length,
    float viscosity_coefficient
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float3 viscosity_force = {0.0f, 0.0f, 0.0f};
    float3 pos_i = positions[i];
    float3 vel_i = velocities[i];
    float density_i = densities[i];

    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        float3 pos_j = positions[j];
        float3 vel_j = velocities[j];
        float3 r_ij = {
            pos_i.x - pos_j.x,
            pos_i.y - pos_j.y,
            pos_i.z - pos_j.z
        };

        float r = sqrtf(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);

        if (r < smoothing_length) {
            float density_j = densities[j];
            float3 vel_diff = {
                vel_j.x - vel_i.x,
                vel_j.y - vel_i.y,
                vel_j.z - vel_i.z
            };

            float laplacian = viscosity_laplacian_kernel(r, smoothing_length);
            float viscosity_term = viscosity_coefficient * masses[j] / density_j * laplacian;

            viscosity_force.x += viscosity_term * vel_diff.x;
            viscosity_force.y += viscosity_term * vel_diff.y;
            viscosity_force.z += viscosity_term * vel_diff.z;
        }
    }

    atomicAdd(&forces[i].x, viscosity_force.x);
    atomicAdd(&forces[i].y, viscosity_force.y);
    atomicAdd(&forces[i].z, viscosity_force.z);
}

// Surface tension calculation kernel
__global__ void calculate_surface_tension_kernel(
    const float3* __restrict__ positions,
    const float* __restrict__ densities,
    const float* __restrict__ masses,
    float3* __restrict__ forces,
    int num_particles,
    float smoothing_length,
    float surface_tension_coefficient
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float3 color_gradient = {0.0f, 0.0f, 0.0f};
    float color_laplacian = 0.0f;
    float3 pos_i = positions[i];
    float density_i = densities[i];

    // Calculate color field gradient and laplacian
    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        float3 pos_j = positions[j];
        float3 r_ij = {
            pos_i.x - pos_j.x,
            pos_i.y - pos_j.y,
            pos_i.z - pos_j.z
        };

        float r = sqrtf(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);

        if (r < smoothing_length) {
            float density_j = densities[j];
            float mass_over_density = masses[j] / density_j;

            float3 gradient = spiky_gradient_kernel(r_ij, r, smoothing_length);
            color_gradient.x += mass_over_density * gradient.x;
            color_gradient.y += mass_over_density * gradient.y;
            color_gradient.z += mass_over_density * gradient.z;

            color_laplacian += mass_over_density * viscosity_laplacian_kernel(r, smoothing_length);
        }
    }

    // Calculate surface tension force
    float gradient_magnitude = sqrtf(
        color_gradient.x * color_gradient.x +
        color_gradient.y * color_gradient.y +
        color_gradient.z * color_gradient.z
    );

    if (gradient_magnitude > 1e-6f) {
        float3 normal = {
            color_gradient.x / gradient_magnitude,
            color_gradient.y / gradient_magnitude,
            color_gradient.z / gradient_magnitude
        };

        float3 surface_force = {
            -surface_tension_coefficient * color_laplacian * normal.x,
            -surface_tension_coefficient * color_laplacian * normal.y,
            -surface_tension_coefficient * color_laplacian * normal.z
        };

        atomicAdd(&forces[i].x, surface_force.x);
        atomicAdd(&forces[i].y, surface_force.y);
        atomicAdd(&forces[i].z, surface_force.z);
    }
}

// Boundary handling kernel
__global__ void handle_fluid_boundaries_kernel(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    int num_particles,
    float3 boundary_min,
    float3 boundary_max,
    float damping_factor
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float3 pos = positions[i];
    float3 vel = velocities[i];

    // Handle x boundaries
    if (pos.x < boundary_min.x) {
        pos.x = boundary_min.x;
        vel.x = fmaxf(0.0f, vel.x) * damping_factor;
    } else if (pos.x > boundary_max.x) {
        pos.x = boundary_max.x;
        vel.x = fminf(0.0f, vel.x) * damping_factor;
    }

    // Handle y boundaries
    if (pos.y < boundary_min.y) {
        pos.y = boundary_min.y;
        vel.y = fmaxf(0.0f, vel.y) * damping_factor;
    } else if (pos.y > boundary_max.y) {
        pos.y = boundary_max.y;
        vel.y = fminf(0.0f, vel.y) * damping_factor;
    }

    // Handle z boundaries
    if (pos.z < boundary_min.z) {
        pos.z = boundary_min.z;
        vel.z = fmaxf(0.0f, vel.z) * damping_factor;
    } else if (pos.z > boundary_max.z) {
        pos.z = boundary_max.z;
        vel.z = fminf(0.0f, vel.z) * damping_factor;
    }

    positions[i] = pos;
    velocities[i] = vel;
}

// Advection kernel for fluid properties
__global__ void advect_fluid_properties_kernel(
    const float3* __restrict__ velocities,
    float* __restrict__ scalar_field,
    float* __restrict__ new_scalar_field,
    int num_particles,
    float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    // Simple semi-Lagrangian advection
    float3 vel = velocities[i];

    // Backward trace (simplified)
    float advected_value = scalar_field[i] - dt * (
        vel.x * 0.1f + vel.y * 0.1f + vel.z * 0.1f
    );

    new_scalar_field[i] = advected_value;
}

} // namespace physgrad