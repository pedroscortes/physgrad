/**
 * PhysGrad Adjoint Methods Header
 *
 * Forward declarations for adjoint (backward pass) CUDA kernels
 */

#pragma once

#include <cuda_runtime.h>

namespace physgrad {
namespace adjoint {

// =============================================================================
// CUDA KERNEL LAUNCHERS FOR ADJOINT METHODS
// =============================================================================

extern "C" {

/**
 * Launch backward pass for Verlet integration
 */
void launch_verlet_integration_backward(
    const float3* grad_positions_out,
    const float3* grad_velocities_out,
    float3* grad_positions_in,
    float3* grad_velocities_in,
    float3* grad_forces,
    float* grad_masses,
    const float3* saved_velocities,
    const float3* saved_forces,
    const float* saved_masses,
    float dt,
    int num_particles,
    cudaStream_t stream = 0
);

/**
 * Launch backward pass for classical force computation
 */
void launch_classical_force_backward(
    const float3* grad_forces,
    float3* grad_positions,
    float* grad_charges,
    const float3* saved_positions,
    const float* saved_charges,
    int num_particles,
    cudaStream_t stream = 0
);

/**
 * Launch backward pass for energy calculation
 */
void launch_energy_backward(
    float grad_energy,
    float3* grad_velocities,
    float* grad_masses,
    float3* grad_positions,
    const float3* saved_velocities,
    const float* saved_masses,
    const float3* saved_positions,
    const float* saved_charges,
    int num_particles,
    cudaStream_t stream = 0
);

/**
 * Launch backward pass for SPH density computation
 */
void launch_sph_density_backward(
    const float* grad_densities,
    float3* grad_positions,
    float* grad_masses,
    const float3* saved_positions,
    const float* saved_masses,
    float smoothing_length,
    int num_particles,
    cudaStream_t stream = 0
);

} // extern "C"

// =============================================================================
// HIGH-LEVEL ADJOINT API
// =============================================================================

/**
 * Adjoint context for managing saved values and gradients
 */
class AdjointContext {
public:
    AdjointContext(int max_particles);
    ~AdjointContext();

    // Save forward pass values
    void saveForBackward(const float3* positions, const float3* velocities,
                        const float3* forces, const float* masses,
                        const float* charges, int num_particles);

    // Execute backward pass
    void executeBackward(const float3* grad_positions_out,
                        const float3* grad_velocities_out,
                        float3* grad_positions_in,
                        float3* grad_velocities_in,
                        float3* grad_forces,
                        float* grad_masses,
                        float* grad_charges,
                        float dt, int num_particles);

    // Clear saved values
    void clear();

private:
    // Device memory for saved values
    float3* d_saved_positions_;
    float3* d_saved_velocities_;
    float3* d_saved_forces_;
    float* d_saved_masses_;
    float* d_saved_charges_;

    int max_particles_;
    bool has_saved_data_;
};

/**
 * Finite difference gradient checker for validation
 */
class GradientChecker {
public:
    /**
     * Check gradients using finite differences
     * @param tolerance Relative tolerance for gradient check
     * @return true if gradients are correct within tolerance
     */
    static bool checkVerletGradients(
        const float3* positions, const float3* velocities,
        const float3* forces, const float* masses,
        float dt, int num_particles,
        float tolerance = 1e-5f
    );

    static bool checkForceGradients(
        const float3* positions, const float* charges,
        int num_particles, float tolerance = 1e-5f
    );

    static bool checkEnergyGradients(
        const float3* positions, const float3* velocities,
        const float* masses, const float* charges,
        int num_particles, float tolerance = 1e-5f
    );

private:
    static constexpr float FINITE_DIFF_EPS = 1e-6f;
};

} // namespace adjoint
} // namespace physgrad