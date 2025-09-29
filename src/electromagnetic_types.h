/**
 * PhysGrad - Electromagnetic Types Header
 *
 * Common type definitions for electromagnetic simulation
 * that work in both CUDA and C++ compilation contexts.
 */

#ifndef PHYSGRAD_ELECTROMAGNETIC_TYPES_H
#define PHYSGRAD_ELECTROMAGNETIC_TYPES_H

#include "common_types.h"

namespace physgrad {

/**
 * GPU memory layout for electromagnetic fields
 */
struct GPUEMGrid {
    // Electric field components (Ex, Ey, Ez)
    float* Ex;
    float* Ey;
    float* Ez;

    // Magnetic field components (Hx, Hy, Hz)
    float* Hx;
    float* Hy;
    float* Hz;

    // Previous timestep fields for leap-frog integration
    float* Ex_prev;
    float* Ey_prev;
    float* Ez_prev;
    float* Hx_prev;
    float* Hy_prev;
    float* Hz_prev;

    // Material properties
    float* epsilon;    // Permittivity
    float* mu;         // Permeability
    float* sigma;      // Conductivity

    // PML absorption coefficients
    float* pml_sx;
    float* pml_sy;
    float* pml_sz;
    float* pml_ax;
    float* pml_ay;
    float* pml_az;

    // Current and charge density
    float* Jx;
    float* Jy;
    float* Jz;
    float* rho;

    // Grid dimensions and parameters
    int nx, ny, nz;           // Grid dimensions
    float dx, dy, dz;         // Grid spacing
    float dt;                 // Time step
    float c0;                 // Speed of light
    float epsilon0, mu0;      // Free space parameters
};

/**
 * Charged particle for GPU simulation
 */
struct GPUChargedParticle {
    float3 position;
    float3 velocity;
    float3 acceleration;

    float charge;
    float mass;

    float3 electric_force;
    float3 magnetic_force;

    // Radiation damping parameters
    float3 prev_acceleration;
    float radiated_power;
    float radius;              // For collision detection
};

/**
 * Electromagnetic source specification
 */
struct GPUEMSource {
    float3 position;       // Source position
    float3 direction;      // Propagation direction (for plane waves)
    float3 polarization;   // Electric field polarization

    float frequency;       // Frequency (Hz)
    float amplitude;       // Field amplitude
    float phase;          // Initial phase

    int source_type;   // 0=plane_wave, 1=dipole, 2=gaussian_beam, etc.
    float beam_waist;  // For Gaussian beams

    // Grid indices for efficient source application
    int i_start, i_end;
    int j_start, j_end;
    int k_start, k_end;
};

} // namespace physgrad

#endif // PHYSGRAD_ELECTROMAGNETIC_TYPES_H