#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <Eigen/Dense>
#include <vector>

namespace physgrad {

// GPU-optimized contact constraint representation
struct GPUContactConstraint {
    int body_a, body_b;                  // Contacting body indices
    float contact_point[3];              // Contact point in global coordinates
    float contact_normal[3];             // Outward normal from body_a to body_b
    float signed_distance;               // Negative if penetrating
    float barrier_potential;             // Φ(d) - barrier potential energy
    float barrier_gradient;              // dΦ/dd - force magnitude
    float barrier_hessian;               // d²Φ/dd² - for Newton convergence

    // Friction constraint data
    float tangent_basis[6];              // Two orthonormal tangent vectors (3 + 3)
    float friction_multiplier[2];        // Lagrange multipliers λt
    float normal_multiplier;             // Normal Lagrange multiplier λn

    // Material interface properties
    float combined_friction_coeff;
    float combined_restitution;
    float combined_stiffness;
};

// GPU memory layout for variational contact solver
struct VariationalContactGPUData {
    // Body state arrays
    float* d_positions;                  // [3*n] - packed xyz positions
    float* d_velocities;                 // [3*n] - packed xyz velocities
    float* d_masses;                     // [n] - body masses
    float* d_radii;                      // [n] - body radii
    int* d_material_ids;                 // [n] - material type IDs

    // Contact constraint arrays
    GPUContactConstraint* d_contacts;    // [max_contacts] - active contact constraints
    int* d_contact_count;                // [1] - actual number of active contacts
    int* d_contact_pairs;                // [2*max_pairs] - potential contact pairs from broad phase
    int* d_pair_count;                   // [1] - number of potential pairs

    // Force computation arrays
    float* d_contact_forces;             // [3*n] - contact forces on each body
    float* d_contact_energy;             // [1] - total contact potential energy

    // Newton solver arrays
    float* d_newton_residual;            // [6*n] - residual vector for Newton solver
    float* d_newton_delta;               // [6*n] - Newton step direction
    float* d_line_search_energies;       // [16] - energies for line search

    // Gradient computation arrays
    float* d_position_gradients;         // [3*n] - gradients w.r.t. positions
    float* d_velocity_gradients;         // [3*n] - gradients w.r.t. velocities

    // Spatial hash for contact detection
    int* d_spatial_hash_keys;           // [n] - spatial hash keys for each body
    int* d_spatial_hash_values;         // [n] - body indices sorted by hash key
    int* d_cell_starts;                 // [num_cells] - start index of each cell
    int* d_cell_ends;                   // [num_cells] - end index of each cell

    // Solver parameters (constant memory candidates)
    float barrier_stiffness;
    float barrier_threshold;
    float friction_regularization;
    int max_newton_iterations;
    float newton_tolerance;

    // Memory management
    int n_bodies;
    int max_contacts;
    int max_contact_pairs;
    int num_hash_cells;
    size_t total_allocated_bytes;

    // CUDA streams for async execution
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    cudaStream_t gradient_stream;
};

// CUDA kernel function declarations
extern "C" {

// Contact detection kernels
__global__ void computeSpatialHashKernel(
    const float* d_positions,
    const float* d_radii,
    int* d_hash_keys,
    int* d_hash_values,
    int n_bodies,
    float cell_size,
    float3 world_min,
    int3 grid_size
);

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
);

// Barrier potential computation kernels
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
);

__global__ void computeContactForcesKernel(
    const GPUContactConstraint* d_contacts,
    int contact_count,
    float* d_contact_forces,
    int n_bodies
);

__global__ void computeContactEnergyKernel(
    const GPUContactConstraint* d_contacts,
    int contact_count,
    float* d_contact_energy
);

// Newton-Raphson solver kernels
__global__ void assembleNewtonResidualKernel(
    const float* d_positions,
    const float* d_velocities,
    const float* d_masses,
    const GPUContactConstraint* d_contacts,
    int contact_count,
    const float* d_positions_initial,
    const float* d_velocities_initial,
    const float* d_external_forces,
    float dt,
    float* d_newton_residual,
    int n_bodies
);

__global__ void newtonLineSearchKernel(
    const float* d_positions,
    const float* d_velocities,
    const float* d_newton_delta,
    float* d_line_search_energies,
    const float* line_search_alphas,
    int n_search_points,
    int n_bodies
);

// Gradient computation kernels (adjoint method)
__global__ void computePositionGradientsKernel(
    const float* d_positions,
    const float* d_velocities,
    const float* d_radii,
    const int* d_material_ids,
    const GPUContactConstraint* d_contacts,
    int contact_count,
    const float* d_output_gradients,
    float* d_position_gradients,
    int n_bodies
);

__global__ void computeVelocityGradientsKernel(
    const float* d_positions,
    const float* d_velocities,
    const GPUContactConstraint* d_contacts,
    int contact_count,
    const float* d_output_gradients,
    float* d_velocity_gradients,
    int n_bodies
);

// Utility kernels
__global__ void resetArrayKernel(float* array, float value, int size);
__global__ void resetIntArrayKernel(int* array, int value, int size);

// Reduction kernels for performance metrics
__global__ void reduceMaxVelocityKernel(
    const float* d_velocities,
    float* d_max_velocity,
    int n_bodies
);

__global__ void reduceTotalEnergyKernel(
    const float* d_velocities,
    const float* d_masses,
    const float* d_contact_energy,
    float* d_total_energy,
    int n_bodies
);

} // extern "C"

// GPU memory management functions
class VariationalContactGPUManager {
public:
    static VariationalContactGPUData* allocateGPUData(
        int n_bodies,
        int max_contacts = 10000,
        int max_contact_pairs = 50000,
        int num_hash_cells = 32768
    );

    static void deallocateGPUData(VariationalContactGPUData* gpu_data);

    static void copyToGPU(
        VariationalContactGPUData* gpu_data,
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& masses,
        const std::vector<double>& radii,
        const std::vector<int>& material_ids,
        cudaStream_t stream = 0
    );

    static void copyFromGPU(
        const VariationalContactGPUData* gpu_data,
        std::vector<Eigen::Vector3d>& positions,
        std::vector<Eigen::Vector3d>& velocities,
        std::vector<Eigen::Vector3d>& forces,
        cudaStream_t stream = 0
    );

    static void copyGradientsFromGPU(
        const VariationalContactGPUData* gpu_data,
        std::vector<Eigen::Vector3d>& position_gradients,
        std::vector<Eigen::Vector3d>& velocity_gradients,
        cudaStream_t stream = 0
    );

    // Performance analysis
    static size_t getMemoryUsage(const VariationalContactGPUData* gpu_data);
    static void printMemoryLayout(const VariationalContactGPUData* gpu_data);
};

// Launch parameter computation for optimal GPU utilization
struct KernelLaunchParams {
    dim3 grid_size;
    dim3 block_size;
    size_t shared_memory_bytes;

    static KernelLaunchParams computeFor1D(int num_elements, int preferred_block_size = 256);
    static KernelLaunchParams computeFor2D(int width, int height, int preferred_block_size = 16);
    static KernelLaunchParams computeForContacts(int num_contacts, int preferred_block_size = 128);
};

// Error checking macros for GPU kernels
#define CUDA_CHECK_KERNEL() do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA kernel error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK_ASYNC(stream) do { \
    cudaError_t error = cudaStreamSynchronize(stream); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA async error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

} // namespace physgrad