#include "memory_optimization.h"

extern "C" {

// Wrapper for optimized force computation kernel
void launch_optimized_force_computation(
    const float4* positions,
    const float* charges,
    float4* forces,
    int num_particles,
    int block_size
) {
    int grid_size = (num_particles + block_size - 1) / block_size;

    optimized_force_computation_kernel<<<grid_size, block_size>>>(
        positions, charges, forces, num_particles
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// Wrapper for optimized Verlet integration kernel
void launch_optimized_verlet_integration(
    float4* positions,
    float4* velocities,
    const float4* forces,
    float dt,
    int num_particles,
    int block_size
) {
    int grid_size = (num_particles + block_size - 1) / block_size;

    optimized_verlet_integration_kernel<<<grid_size, block_size>>>(
        positions, velocities, forces, dt, num_particles
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// Wrapper for optimized energy reduction kernel
void launch_optimized_energy_reduction(
    const float4* velocities,
    const float4* forces,
    float* total_kinetic,
    float* total_potential,
    int num_particles,
    int block_size
) {
    int grid_size = (num_particles + block_size - 1) / block_size;

    optimized_energy_reduction_kernel<<<grid_size, block_size>>>(
        velocities, forces, total_kinetic, total_potential, num_particles
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

}