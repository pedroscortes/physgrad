/**
 * PhysGrad Material Point Method (MPM) CUDA Kernels
 *
 * High-performance GPU kernels for MPM simulation with G2P2G fusion
 * and multi-material constitutive models
 */

#include "mpm_data_structures.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

namespace physgrad::mpm {

// =============================================================================
// SHAPE FUNCTIONS AND GRADIENTS
// =============================================================================

/**
 * B-spline shape functions for MPM
 * Supports linear, quadratic, and cubic B-splines
 */
template<typename T>
struct ShapeFunctions {
    // Linear B-spline (hat function)
    __device__ static T linear(T x) {
        T abs_x = fabsf(x);
        if (abs_x < 1.0f) {
            return 1.0f - abs_x;
        }
        return 0.0f;
    }

    __device__ static T linear_gradient(T x) {
        if (x < -1.0f || x > 1.0f) return 0.0f;
        return (x < 0.0f) ? 1.0f : -1.0f;
    }

    // Quadratic B-spline
    __device__ static T quadratic(T x) {
        T abs_x = fabsf(x);
        if (abs_x < 0.5f) {
            return 0.75f - x * x;
        } else if (abs_x < 1.5f) {
            T temp = 1.5f - abs_x;
            return 0.5f * temp * temp;
        }
        return 0.0f;
    }

    __device__ static T quadratic_gradient(T x) {
        T abs_x = fabsf(x);
        if (abs_x < 0.5f) {
            return -2.0f * x;
        } else if (abs_x < 1.5f) {
            return (x > 0.0f) ? (abs_x - 1.5f) : (1.5f - abs_x);
        }
        return 0.0f;
    }

    // Cubic B-spline
    __device__ static T cubic(T x) {
        T abs_x = fabsf(x);
        if (abs_x < 1.0f) {
            T x2 = x * x;
            return (2.0f/3.0f) - x2 + 0.5f * abs_x * x2;
        } else if (abs_x < 2.0f) {
            T temp = 2.0f - abs_x;
            return temp * temp * temp / 6.0f;
        }
        return 0.0f;
    }

    __device__ static T cubic_gradient(T x) {
        T abs_x = fabsf(x);
        if (abs_x < 1.0f) {
            return x * (1.5f * abs_x - 2.0f);
        } else if (abs_x < 2.0f) {
            T temp = 2.0f - abs_x;
            return (x > 0.0f) ? -0.5f * temp * temp : 0.5f * temp * temp;
        }
        return 0.0f;
    }
};

// =============================================================================
// CONSTITUTIVE MODELS
// =============================================================================

/**
 * Constitutive model implementations for different materials
 */
template<typename T>
struct ConstitutiveModels {

    // Neo-Hookean hyperelastic model
    __device__ static void neoHookean(const T F[9], const MaterialParameters& params, T stress[6]) {
        // Compute deformation invariants
        T J = F[0] * (F[4] * F[8] - F[5] * F[7]) -
              F[1] * (F[3] * F[8] - F[5] * F[6]) +
              F[2] * (F[3] * F[7] - F[4] * F[6]);

        // Left Cauchy-Green tensor B = F * F^T
        T B[6];  // Symmetric tensor: Bxx, Byy, Bzz, Bxy, Bxz, Byz
        B[0] = F[0]*F[0] + F[1]*F[1] + F[2]*F[2];  // Bxx
        B[1] = F[3]*F[3] + F[4]*F[4] + F[5]*F[5];  // Byy
        B[2] = F[6]*F[6] + F[7]*F[7] + F[8]*F[8];  // Bzz
        B[3] = F[0]*F[3] + F[1]*F[4] + F[2]*F[5];  // Bxy
        B[4] = F[0]*F[6] + F[1]*F[7] + F[2]*F[8];  // Bxz
        B[5] = F[3]*F[6] + F[4]*F[7] + F[5]*F[8];  // Byz

        // Lamé parameters
        T lambda = params.youngs_modulus * params.poisson_ratio /
                   ((1.0f + params.poisson_ratio) * (1.0f - 2.0f * params.poisson_ratio));
        T mu = params.youngs_modulus / (2.0f * (1.0f + params.poisson_ratio));

        // Cauchy stress: σ = (μ/J)(B - I) + (λ/J)ln(J)I
        T ln_J = logf(J);
        T pressure = lambda * ln_J / J;
        T mu_over_J = mu / J;

        stress[0] = mu_over_J * (B[0] - 1.0f) + pressure;  // σxx
        stress[1] = mu_over_J * (B[1] - 1.0f) + pressure;  // σyy
        stress[2] = mu_over_J * (B[2] - 1.0f) + pressure;  // σzz
        stress[3] = mu_over_J * B[3];                       // σxy
        stress[4] = mu_over_J * B[4];                       // σxz
        stress[5] = mu_over_J * B[5];                       // σyz
    }

    // Fluid model (Newtonian viscosity)
    __device__ static void fluidModel(const T velocity_gradient[9], const MaterialParameters& params, T stress[6]) {
        // Strain rate tensor: ε̇ = 0.5(∇v + ∇v^T)
        T strain_rate[6];
        strain_rate[0] = velocity_gradient[0];                                    // ε̇xx = ∂vx/∂x
        strain_rate[1] = velocity_gradient[4];                                    // ε̇yy = ∂vy/∂y
        strain_rate[2] = velocity_gradient[8];                                    // ε̇zz = ∂vz/∂z
        strain_rate[3] = 0.5f * (velocity_gradient[1] + velocity_gradient[3]);   // ε̇xy = 0.5(∂vx/∂y + ∂vy/∂x)
        strain_rate[4] = 0.5f * (velocity_gradient[2] + velocity_gradient[6]);   // ε̇xz = 0.5(∂vx/∂z + ∂vz/∂x)
        strain_rate[5] = 0.5f * (velocity_gradient[5] + velocity_gradient[7]);   // ε̇yz = 0.5(∂vy/∂z + ∂vz/∂y)

        // Viscous stress: τ = 2μ ε̇
        T two_mu = 2.0f * params.viscosity;
        stress[0] = two_mu * strain_rate[0];
        stress[1] = two_mu * strain_rate[1];
        stress[2] = two_mu * strain_rate[2];
        stress[3] = two_mu * strain_rate[3];
        stress[4] = two_mu * strain_rate[4];
        stress[5] = two_mu * strain_rate[5];
    }

    // von Mises plasticity model
    __device__ static void vonMisesPlasticity(const T F[9], T F_plastic[9], const MaterialParameters& params, T stress[6]) {
        // Elastic deformation gradient: F_e = F * F_p^(-1)
        T F_p_inv[9];
        invertMatrix3x3(F_plastic, F_p_inv);

        T F_elastic[9];
        multiplyMatrix3x3(F, F_p_inv, F_elastic);

        // Compute elastic stress using Neo-Hookean model
        neoHookean(F_elastic, params, stress);

        // Check yield condition
        T von_mises_stress = computeVonMisesStress(stress);
        if (von_mises_stress > params.yield_stress) {
            // Return mapping algorithm (simplified)
            T scale_factor = params.yield_stress / von_mises_stress;

            // Scale deviatoric stress
            T pressure = (stress[0] + stress[1] + stress[2]) / 3.0f;
            stress[0] = pressure + scale_factor * (stress[0] - pressure);
            stress[1] = pressure + scale_factor * (stress[1] - pressure);
            stress[2] = pressure + scale_factor * (stress[2] - pressure);
            stress[3] *= scale_factor;
            stress[4] *= scale_factor;
            stress[5] *= scale_factor;

            // Update plastic deformation gradient (simplified)
            T plastic_strain_increment = (1.0f - scale_factor) * von_mises_stress / params.youngs_modulus;
            updatePlasticDeformationGradient(F_plastic, plastic_strain_increment);
        }
    }

    // Drucker-Prager plasticity model (for granular materials like sand)
    __device__ static void druckerPragerPlasticity(const T F[9], T F_plastic[9], const MaterialParameters& params, T stress[6]) {
        // Elastic deformation gradient: F_e = F * F_p^(-1)
        T F_p_inv[9];
        invertMatrix3x3(F_plastic, F_p_inv);

        T F_elastic[9];
        multiplyMatrix3x3(F, F_p_inv, F_elastic);

        // Compute elastic stress using Neo-Hookean model
        neoHookean(F_elastic, params, stress);

        // Drucker-Prager yield criterion: f = ||s|| - α*p - k
        // where s = deviatoric stress, p = pressure, α = friction coefficient, k = cohesion
        T pressure = (stress[0] + stress[1] + stress[2]) / 3.0f;

        // Deviatoric stress
        T s_dev[6];
        s_dev[0] = stress[0] - pressure;
        s_dev[1] = stress[1] - pressure;
        s_dev[2] = stress[2] - pressure;
        s_dev[3] = stress[3];
        s_dev[4] = stress[4];
        s_dev[5] = stress[5];

        T s_norm = sqrtf(s_dev[0]*s_dev[0] + s_dev[1]*s_dev[1] + s_dev[2]*s_dev[2] +
                        2.0f * (s_dev[3]*s_dev[3] + s_dev[4]*s_dev[4] + s_dev[5]*s_dev[5]));

        // Friction coefficient from friction angle: α = 6*sin(φ)/(3-sin(φ))
        T friction_angle_rad = params.friction_angle * 3.14159265f / 180.0f;
        T sin_phi = sinf(friction_angle_rad);
        T alpha = 6.0f * sin_phi / (3.0f - sin_phi);

        // Cohesion parameter: k = 6*c*cos(φ)/(3-sin(φ))
        T cos_phi = cosf(friction_angle_rad);
        T k = 6.0f * params.cohesion * cos_phi / (3.0f - sin_phi);

        // Yield function
        T yield_function = s_norm - alpha * pressure - k;

        if (yield_function > 0.0f) {
            // Return mapping: project stress back to yield surface
            T lambda = yield_function / (2.0f * params.youngs_modulus / (2.0f * (1.0f + params.poisson_ratio)) + alpha * alpha * params.youngs_modulus / (3.0f * (1.0f - 2.0f * params.poisson_ratio)));

            // Update pressure
            T mu = params.youngs_modulus / (2.0f * (1.0f + params.poisson_ratio));
            T K = params.youngs_modulus / (3.0f * (1.0f - 2.0f * params.poisson_ratio));
            pressure += alpha * lambda * K;

            // Update deviatoric stress
            T scale = (s_norm - lambda * 2.0f * mu) / s_norm;
            s_dev[0] *= scale;
            s_dev[1] *= scale;
            s_dev[2] *= scale;
            s_dev[3] *= scale;
            s_dev[4] *= scale;
            s_dev[5] *= scale;

            // Reconstruct total stress
            stress[0] = s_dev[0] + pressure;
            stress[1] = s_dev[1] + pressure;
            stress[2] = s_dev[2] + pressure;
            stress[3] = s_dev[3];
            stress[4] = s_dev[4];
            stress[5] = s_dev[5];

            // Update plastic deformation gradient
            updatePlasticDeformationGradient(F_plastic, lambda);
        }
    }

    // Snow plasticity model (based on Stomakhin et al. 2013)
    __device__ static void snowPlasticity(const T F[9], T F_plastic[9], const MaterialParameters& params, T stress[6]) {
        // Singular value decomposition: F = U * Σ * V^T
        // For snow, we clamp singular values to prevent excessive compression/stretching

        // Elastic deformation gradient
        T F_p_inv[9];
        invertMatrix3x3(F_plastic, F_p_inv);

        T F_elastic[9];
        multiplyMatrix3x3(F, F_p_inv, F_elastic);

        // Compute SVD (simplified - using polar decomposition)
        T U[9], V[9], sigma[3];
        polarDecomposition(F_elastic, U, sigma, V);

        // Critical compression and stretch thresholds
        T theta_c = params.critical_compression;  // Typical: 2.5e-2
        T theta_s = params.critical_stretch;      // Typical: 7.5e-3

        // Clamp singular values
        T sigma_clamped[3];
        bool plastic_flow = false;
        for (int i = 0; i < 3; ++i) {
            if (sigma[i] < 1.0f - theta_c) {
                // Critical compression
                sigma_clamped[i] = 1.0f - theta_c;
                plastic_flow = true;
            } else if (sigma[i] > 1.0f + theta_s) {
                // Critical stretch
                sigma_clamped[i] = 1.0f + theta_s;
                plastic_flow = true;
            } else {
                sigma_clamped[i] = sigma[i];
            }
        }

        if (plastic_flow) {
            // Update plastic deformation gradient
            T F_e_new[9];
            reconstructFromSVD(U, sigma_clamped, V, F_e_new);

            // F_p_new = F * F_e_new^(-1)
            T F_e_new_inv[9];
            invertMatrix3x3(F_e_new, F_e_new_inv);
            multiplyMatrix3x3(F, F_e_new_inv, F_plastic);

            // Recompute elastic deformation with clamped values
            for (int i = 0; i < 9; ++i) {
                F_elastic[i] = F_e_new[i];
            }
        }

        // Compute stress using clamped elastic deformation
        neoHookean(F_elastic, params, stress);

        // Apply hardening: E(F_p) = E_0 * exp(ξ * (1 - J_p))
        T J_p = F_plastic[0] * (F_plastic[4] * F_plastic[8] - F_plastic[5] * F_plastic[7]) -
                F_plastic[1] * (F_plastic[3] * F_plastic[8] - F_plastic[5] * F_plastic[6]) +
                F_plastic[2] * (F_plastic[3] * F_plastic[7] - F_plastic[4] * F_plastic[6]);

        T xi = params.hardening_coefficient; // Hardening parameter (typical: 10)
        T hardening = expf(xi * (1.0f - J_p));

        // Scale stress by hardening
        for (int i = 0; i < 6; ++i) {
            stress[i] *= hardening;
        }
    }

    // Helper functions
    __device__ static void invertMatrix3x3(const T A[9], T A_inv[9]) {
        T det = A[0] * (A[4] * A[8] - A[5] * A[7]) -
                A[1] * (A[3] * A[8] - A[5] * A[6]) +
                A[2] * (A[3] * A[7] - A[4] * A[6]);

        T inv_det = 1.0f / det;

        A_inv[0] = (A[4] * A[8] - A[5] * A[7]) * inv_det;
        A_inv[1] = (A[2] * A[7] - A[1] * A[8]) * inv_det;
        A_inv[2] = (A[1] * A[5] - A[2] * A[4]) * inv_det;
        A_inv[3] = (A[5] * A[6] - A[3] * A[8]) * inv_det;
        A_inv[4] = (A[0] * A[8] - A[2] * A[6]) * inv_det;
        A_inv[5] = (A[2] * A[3] - A[0] * A[5]) * inv_det;
        A_inv[6] = (A[3] * A[7] - A[4] * A[6]) * inv_det;
        A_inv[7] = (A[1] * A[6] - A[0] * A[7]) * inv_det;
        A_inv[8] = (A[0] * A[4] - A[1] * A[3]) * inv_det;
    }

    __device__ static void multiplyMatrix3x3(const T A[9], const T B[9], T C[9]) {
        C[0] = A[0]*B[0] + A[1]*B[3] + A[2]*B[6];
        C[1] = A[0]*B[1] + A[1]*B[4] + A[2]*B[7];
        C[2] = A[0]*B[2] + A[1]*B[5] + A[2]*B[8];
        C[3] = A[3]*B[0] + A[4]*B[3] + A[5]*B[6];
        C[4] = A[3]*B[1] + A[4]*B[4] + A[5]*B[7];
        C[5] = A[3]*B[2] + A[4]*B[5] + A[5]*B[8];
        C[6] = A[6]*B[0] + A[7]*B[3] + A[8]*B[6];
        C[7] = A[6]*B[1] + A[7]*B[4] + A[8]*B[7];
        C[8] = A[6]*B[2] + A[7]*B[5] + A[8]*B[8];
    }

    __device__ static T computeVonMisesStress(const T stress[6]) {
        T s_dev[6];  // Deviatoric stress
        T pressure = (stress[0] + stress[1] + stress[2]) / 3.0f;
        s_dev[0] = stress[0] - pressure;
        s_dev[1] = stress[1] - pressure;
        s_dev[2] = stress[2] - pressure;
        s_dev[3] = stress[3];
        s_dev[4] = stress[4];
        s_dev[5] = stress[5];

        return sqrtf(1.5f * (s_dev[0]*s_dev[0] + s_dev[1]*s_dev[1] + s_dev[2]*s_dev[2] +
                            2.0f * (s_dev[3]*s_dev[3] + s_dev[4]*s_dev[4] + s_dev[5]*s_dev[5])));
    }

    __device__ static void updatePlasticDeformationGradient(T F_plastic[9], T plastic_strain) {
        // Simplified plastic flow update (isotropic)
        T factor = 1.0f + plastic_strain;
        F_plastic[0] *= factor;
        F_plastic[4] *= factor;
        F_plastic[8] *= factor;
    }

    // =========================================================================
    // SVD HELPER FUNCTIONS (for snow plasticity)
    // =========================================================================

    /**
     * Polar decomposition via SVD: F = U * Σ * V^T
     * Uses iterative Jacobi SVD for 3x3 matrices (GPU-efficient)
     *
     * @param F Input 3x3 matrix (row-major)
     * @param U Output left singular vectors (row-major)
     * @param sigma Output singular values (3 elements)
     * @param V Output right singular vectors (row-major)
     */
    __device__ static void polarDecomposition(const T F[9], T U[9], T sigma[3], T V[9]) {
        // Compute F^T * F (symmetric positive definite)
        T FtF[9];
        FtF[0] = F[0]*F[0] + F[3]*F[3] + F[6]*F[6];
        FtF[1] = F[0]*F[1] + F[3]*F[4] + F[6]*F[7];
        FtF[2] = F[0]*F[2] + F[3]*F[5] + F[6]*F[8];
        FtF[3] = FtF[1];
        FtF[4] = F[1]*F[1] + F[4]*F[4] + F[7]*F[7];
        FtF[5] = F[1]*F[2] + F[4]*F[5] + F[7]*F[8];
        FtF[6] = FtF[2];
        FtF[7] = FtF[5];
        FtF[8] = F[2]*F[2] + F[5]*F[5] + F[8]*F[8];

        // Initialize V to identity
        V[0] = 1; V[1] = 0; V[2] = 0;
        V[3] = 0; V[4] = 1; V[5] = 0;
        V[6] = 0; V[7] = 0; V[8] = 1;

        // Jacobi iteration to diagonalize F^T * F
        const int max_iterations = 20;
        const T epsilon = 1e-6f;

        for (int iter = 0; iter < max_iterations; iter++) {
            // Find largest off-diagonal element
            T max_val = 0;
            int p = 0, q = 1;

            if (fabs(FtF[1]) > max_val) { max_val = fabs(FtF[1]); p = 0; q = 1; }
            if (fabs(FtF[2]) > max_val) { max_val = fabs(FtF[2]); p = 0; q = 2; }
            if (fabs(FtF[5]) > max_val) { max_val = fabs(FtF[5]); p = 1; q = 2; }

            if (max_val < epsilon) break;

            // Compute Jacobi rotation
            T diff = FtF[q*3 + q] - FtF[p*3 + p];
            T t;
            if (fabs(FtF[p*3 + q]) < epsilon) {
                t = 0;
            } else {
                T phi = diff / (2.0f * FtF[p*3 + q]);
                t = 1.0f / (fabs(phi) + sqrt(phi*phi + 1.0f));
                if (phi < 0) t = -t;
            }

            T c = 1.0f / sqrt(t*t + 1.0f);
            T s = t * c;

            // Apply rotation to FtF
            T FtF_pp = FtF[p*3 + p];
            T FtF_qq = FtF[q*3 + q];
            T FtF_pq = FtF[p*3 + q];

            FtF[p*3 + p] = c*c*FtF_pp - 2*s*c*FtF_pq + s*s*FtF_qq;
            FtF[q*3 + q] = s*s*FtF_pp + 2*s*c*FtF_pq + c*c*FtF_qq;
            FtF[p*3 + q] = FtF[q*3 + p] = 0;

            // Update other elements
            for (int i = 0; i < 3; i++) {
                if (i != p && i != q) {
                    T FtF_ip = FtF[i*3 + p];
                    T FtF_iq = FtF[i*3 + q];
                    FtF[i*3 + p] = FtF[p*3 + i] = c*FtF_ip - s*FtF_iq;
                    FtF[i*3 + q] = FtF[q*3 + i] = s*FtF_ip + c*FtF_iq;
                }
            }

            // Update V (accumulate rotations)
            for (int i = 0; i < 3; i++) {
                T V_ip = V[i*3 + p];
                T V_iq = V[i*3 + q];
                V[i*3 + p] = c*V_ip - s*V_iq;
                V[i*3 + q] = s*V_ip + c*V_iq;
            }
        }

        // Extract singular values (sqrt of eigenvalues of F^T * F)
        sigma[0] = sqrt(fmax(0.0f, FtF[0]));
        sigma[1] = sqrt(fmax(0.0f, FtF[4]));
        sigma[2] = sqrt(fmax(0.0f, FtF[8]));

        // Compute U = F * V * Σ^(-1)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                T sum = 0;
                for (int k = 0; k < 3; k++) {
                    sum += F[i*3 + k] * V[k*3 + j];
                }
                T sigma_inv = (sigma[j] > 1e-8f) ? (1.0f / sigma[j]) : 0.0f;
                U[i*3 + j] = sum * sigma_inv;
            }
        }
    }

    /**
     * Reconstruct matrix from SVD components: F = U * Σ * V^T
     *
     * @param U Left singular vectors (row-major 3x3)
     * @param sigma Singular values (3 elements)
     * @param V Right singular vectors (row-major 3x3)
     * @param F Output reconstructed matrix (row-major 3x3)
     */
    __device__ static void reconstructFromSVD(const T U[9], const T sigma[3], const T V[9], T F[9]) {
        // F = U * Σ * V^T
        // First compute Σ * V^T
        T sigma_Vt[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                sigma_Vt[i*3 + j] = (i == 0 ? sigma[0] : (i == 1 ? sigma[1] : sigma[2])) * V[j*3 + i];
            }
        }

        // Then compute F = U * (Σ * V^T)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                F[i*3 + j] = 0;
                for (int k = 0; k < 3; k++) {
                    F[i*3 + j] += U[i*3 + k] * sigma_Vt[k*3 + j];
                }
            }
        }
    }
};

// =============================================================================
// G2P2G FUSED KERNELS
// =============================================================================

/**
 * Fused Grid-to-Particle-to-Grid (G2P2G) kernel
 * Combines interpolation, particle update, and grid update in a single kernel
 * for maximum memory bandwidth efficiency
 */
template<typename T>
__global__ void fusedG2P2GKernel(
    // Particle data (AoSoA layout)
    const T* __restrict__ particle_positions,
    T* __restrict__ particle_velocities,
    const T* __restrict__ particle_masses,
    const T* __restrict__ particle_volumes,
    T* __restrict__ particle_deformation_gradients,
    T* __restrict__ particle_stresses,
    const MaterialType* __restrict__ particle_material_types,
    const uint32_t* __restrict__ particle_active,

    // Grid data
    const T* __restrict__ grid_masses,
    const T* __restrict__ grid_velocities,
    T* __restrict__ grid_forces,
    T* __restrict__ new_grid_masses,
    T* __restrict__ new_grid_velocities,

    // Simulation parameters
    int3 grid_dims,
    T3<T> grid_spacing,
    T3<T> grid_origin,
    T dt,
    int num_particles,
    MaterialParameters* material_params
) {
    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= num_particles) return;

    // Check if particle is active
    if (!particle_active[particle_id]) return;

    // Load particle data
    T3<T> pos = {
        particle_positions[particle_id * 3 + 0],
        particle_positions[particle_id * 3 + 1],
        particle_positions[particle_id * 3 + 2]
    };

    T3<T> vel = {
        particle_velocities[particle_id * 3 + 0],
        particle_velocities[particle_id * 3 + 1],
        particle_velocities[particle_id * 3 + 2]
    };

    T mass = particle_masses[particle_id];
    T volume = particle_volumes[particle_id];
    MaterialType mat_type = particle_material_types[particle_id];

    // Load deformation gradient
    T F[9];
    for (int i = 0; i < 9; ++i) {
        F[i] = particle_deformation_gradients[particle_id * 9 + i];
    }

    // Find grid cell
    int3 grid_pos = {
        __float2int_rd((pos.x - grid_origin.x) / grid_spacing.x),
        __float2int_rd((pos.y - grid_origin.y) / grid_spacing.y),
        __float2int_rd((pos.z - grid_origin.z) / grid_spacing.z)
    };

    // Grid-to-Particle (G2P) interpolation
    T3<T> grid_velocity = {0, 0, 0};
    T velocity_gradient[9] = {0};  // ∇v

    // Loop over neighboring grid nodes (3x3x3 stencil for quadratic B-splines)
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            for (int dk = -1; dk <= 1; dk++) {
                int gi = grid_pos.x + di;
                int gj = grid_pos.y + dj;
                int gk = grid_pos.z + dk;

                // Check bounds
                if (gi < 0 || gi >= grid_dims.x ||
                    gj < 0 || gj >= grid_dims.y ||
                    gk < 0 || gk >= grid_dims.z) continue;

                size_t grid_idx = static_cast<size_t>(gk) * grid_dims.x * grid_dims.y +
                                 static_cast<size_t>(gj) * grid_dims.x + static_cast<size_t>(gi);

                // Compute shape function and gradient
                T3<T> grid_world_pos = {
                    grid_origin.x + gi * grid_spacing.x,
                    grid_origin.y + gj * grid_spacing.y,
                    grid_origin.z + gk * grid_spacing.z
                };

                T3<T> xi = {
                    (pos.x - grid_world_pos.x) / grid_spacing.x,
                    (pos.y - grid_world_pos.y) / grid_spacing.y,
                    (pos.z - grid_world_pos.z) / grid_spacing.z
                };

                // Quadratic B-spline shape functions
                T Nx = ShapeFunctions<T>::quadratic(xi.x);
                T Ny = ShapeFunctions<T>::quadratic(xi.y);
                T Nz = ShapeFunctions<T>::quadratic(xi.z);
                T N = Nx * Ny * Nz;

                // Shape function gradients
                T dNx = ShapeFunctions<T>::quadratic_gradient(xi.x) / grid_spacing.x;
                T dNy = ShapeFunctions<T>::quadratic_gradient(xi.y) / grid_spacing.y;
                T dNz = ShapeFunctions<T>::quadratic_gradient(xi.z) / grid_spacing.z;

                T3<T> grad_N = {dNx * Ny * Nz, Nx * dNy * Nz, Nx * Ny * dNz};

                // Interpolate grid velocity to particle
                T3<T> node_vel = {
                    grid_velocities[grid_idx * 3 + 0],
                    grid_velocities[grid_idx * 3 + 1],
                    grid_velocities[grid_idx * 3 + 2]
                };

                grid_velocity.x += N * node_vel.x;
                grid_velocity.y += N * node_vel.y;
                grid_velocity.z += N * node_vel.z;

                // Compute velocity gradient tensor
                velocity_gradient[0] += grad_N.x * node_vel.x;  // ∂vx/∂x
                velocity_gradient[1] += grad_N.y * node_vel.x;  // ∂vx/∂y
                velocity_gradient[2] += grad_N.z * node_vel.x;  // ∂vx/∂z
                velocity_gradient[3] += grad_N.x * node_vel.y;  // ∂vy/∂x
                velocity_gradient[4] += grad_N.y * node_vel.y;  // ∂vy/∂y
                velocity_gradient[5] += grad_N.z * node_vel.y;  // ∂vy/∂z
                velocity_gradient[6] += grad_N.x * node_vel.z;  // ∂vz/∂x
                velocity_gradient[7] += grad_N.y * node_vel.z;  // ∂vz/∂y
                velocity_gradient[8] += grad_N.z * node_vel.z;  // ∂vz/∂z
            }
        }
    }

    // Update particle velocity
    vel = grid_velocity;

    // Update deformation gradient: F^(n+1) = (I + dt * ∇v) * F^n
    T I_plus_dt_grad_v[9];
    I_plus_dt_grad_v[0] = 1.0f + dt * velocity_gradient[0];
    I_plus_dt_grad_v[1] = dt * velocity_gradient[1];
    I_plus_dt_grad_v[2] = dt * velocity_gradient[2];
    I_plus_dt_grad_v[3] = dt * velocity_gradient[3];
    I_plus_dt_grad_v[4] = 1.0f + dt * velocity_gradient[4];
    I_plus_dt_grad_v[5] = dt * velocity_gradient[5];
    I_plus_dt_grad_v[6] = dt * velocity_gradient[6];
    I_plus_dt_grad_v[7] = dt * velocity_gradient[7];
    I_plus_dt_grad_v[8] = 1.0f + dt * velocity_gradient[8];

    T new_F[9];
    ConstitutiveModels<T>::multiplyMatrix3x3(I_plus_dt_grad_v, F, new_F);

    // Copy back new deformation gradient
    for (int i = 0; i < 9; ++i) {
        F[i] = new_F[i];
    }

    // Compute stress based on material type
    T stress[6];
    MaterialParameters params = material_params[static_cast<int>(mat_type)];

    switch (mat_type) {
        case MaterialType::ELASTIC:
            ConstitutiveModels<T>::neoHookean(F, params, stress);
            break;
        case MaterialType::FLUID:
            ConstitutiveModels<T>::fluidModel(velocity_gradient, params, stress);
            break;
        case MaterialType::ELASTOPLASTIC:
            // Would need plastic deformation gradient storage
            ConstitutiveModels<T>::neoHookean(F, params, stress);
            break;
        default:
            ConstitutiveModels<T>::neoHookean(F, params, stress);
            break;
    }

    // Store updated particle data
    particle_velocities[particle_id * 3 + 0] = vel.x;
    particle_velocities[particle_id * 3 + 1] = vel.y;
    particle_velocities[particle_id * 3 + 2] = vel.z;

    for (int i = 0; i < 9; ++i) {
        particle_deformation_gradients[particle_id * 9 + i] = F[i];
    }

    for (int i = 0; i < 6; ++i) {
        particle_stresses[particle_id * 6 + i] = stress[i];
    }

    // Particle-to-Grid (P2G) transfer
    // Loop over neighboring grid nodes again
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            for (int dk = -1; dk <= 1; dk++) {
                int gi = grid_pos.x + di;
                int gj = grid_pos.y + dj;
                int gk = grid_pos.z + dk;

                if (gi < 0 || gi >= grid_dims.x ||
                    gj < 0 || gj >= grid_dims.y ||
                    gk < 0 || gk >= grid_dims.z) continue;

                size_t grid_idx = static_cast<size_t>(gk) * grid_dims.x * grid_dims.y +
                                 static_cast<size_t>(gj) * grid_dims.x + static_cast<size_t>(gi);

                // Recompute shape function
                T3<T> grid_world_pos = {
                    grid_origin.x + gi * grid_spacing.x,
                    grid_origin.y + gj * grid_spacing.y,
                    grid_origin.z + gk * grid_spacing.z
                };

                T3<T> xi = {
                    (pos.x - grid_world_pos.x) / grid_spacing.x,
                    (pos.y - grid_world_pos.y) / grid_spacing.y,
                    (pos.z - grid_world_pos.z) / grid_spacing.z
                };

                T Nx = ShapeFunctions<T>::quadratic(xi.x);
                T Ny = ShapeFunctions<T>::quadratic(xi.y);
                T Nz = ShapeFunctions<T>::quadratic(xi.z);
                T N = Nx * Ny * Nz;

                T dNx = ShapeFunctions<T>::quadratic_gradient(xi.x) / grid_spacing.x;
                T dNy = ShapeFunctions<T>::quadratic_gradient(xi.y) / grid_spacing.y;
                T dNz = ShapeFunctions<T>::quadratic_gradient(xi.z) / grid_spacing.z;

                T3<T> grad_N = {dNx * Ny * Nz, Nx * dNy * Nz, Nx * Ny * dNz};

                // Compute forces from stress divergence
                T force_x = -volume * (stress[0] * grad_N.x + stress[3] * grad_N.y + stress[4] * grad_N.z);
                T force_y = -volume * (stress[3] * grad_N.x + stress[1] * grad_N.y + stress[5] * grad_N.z);
                T force_z = -volume * (stress[4] * grad_N.x + stress[5] * grad_N.y + stress[2] * grad_N.z);

                // Atomic operations for thread-safe updates
                atomicAdd(&new_grid_masses[grid_idx], N * mass);
                atomicAdd(&new_grid_velocities[grid_idx * 3 + 0], N * mass * vel.x);
                atomicAdd(&new_grid_velocities[grid_idx * 3 + 1], N * mass * vel.y);
                atomicAdd(&new_grid_velocities[grid_idx * 3 + 2], N * mass * vel.z);
                atomicAdd(&grid_forces[grid_idx * 3 + 0], force_x);
                atomicAdd(&grid_forces[grid_idx * 3 + 1], force_y);
                atomicAdd(&grid_forces[grid_idx * 3 + 2], force_z);
            }
        }
    }
}

/**
 * Grid update kernel - applies forces and boundary conditions
 */
template<typename T>
__global__ void updateGridKernel(
    T* __restrict__ grid_masses,
    T* __restrict__ grid_velocities,
    const T* __restrict__ grid_forces,
    const uint32_t* __restrict__ boundary_conditions,
    T dt,
    T3<T> gravity,
    int total_nodes
) {
    int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= total_nodes) return;

    T mass = grid_masses[node_id];
    if (mass < 1e-10f) return;  // Skip empty nodes

    T inv_mass = 1.0f / mass;

    // Convert momentum to velocity
    T3<T> velocity = {
        grid_velocities[node_id * 3 + 0] * inv_mass,
        grid_velocities[node_id * 3 + 1] * inv_mass,
        grid_velocities[node_id * 3 + 2] * inv_mass
    };

    // Apply forces (including gravity)
    T3<T> acceleration = {
        grid_forces[node_id * 3 + 0] * inv_mass + gravity.x,
        grid_forces[node_id * 3 + 1] * inv_mass + gravity.y,
        grid_forces[node_id * 3 + 2] * inv_mass + gravity.z
    };

    // Time integration
    velocity.x += dt * acceleration.x;
    velocity.y += dt * acceleration.y;
    velocity.z += dt * acceleration.z;

    // Apply boundary conditions
    uint32_t bc = boundary_conditions[node_id];
    if (bc & static_cast<uint32_t>(BoundaryType::DIRICHLET_X)) velocity.x = 0.0f;
    if (bc & static_cast<uint32_t>(BoundaryType::DIRICHLET_Y)) velocity.y = 0.0f;
    if (bc & static_cast<uint32_t>(BoundaryType::DIRICHLET_Z)) velocity.z = 0.0f;

    // Store updated velocity
    grid_velocities[node_id * 3 + 0] = velocity.x;
    grid_velocities[node_id * 3 + 1] = velocity.y;
    grid_velocities[node_id * 3 + 2] = velocity.z;
}

/**
 * Particle position update kernel
 */
template<typename T>
__global__ void updateParticlePositionsKernel(
    T* __restrict__ particle_positions,
    const T* __restrict__ particle_velocities,
    const uint32_t* __restrict__ particle_active,
    T dt,
    int num_particles
) {
    int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= num_particles) return;

    if (!particle_active[particle_id]) return;

    // Update position using velocity
    particle_positions[particle_id * 3 + 0] += dt * particle_velocities[particle_id * 3 + 0];
    particle_positions[particle_id * 3 + 1] += dt * particle_velocities[particle_id * 3 + 1];
    particle_positions[particle_id * 3 + 2] += dt * particle_velocities[particle_id * 3 + 2];
}

} // namespace physgrad::mpm