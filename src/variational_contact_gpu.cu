#include "variational_contact_gpu.h"
#include "variational_contact_gpu.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <iostream>
#include <chrono>

namespace physgrad {

// Note: Constant memory symbols are defined in variational_contact_kernels.cu

// Kernel launch parameter computation
KernelLaunchParams KernelLaunchParams::computeFor1D(int num_elements, int preferred_block_size) {
    KernelLaunchParams params;
    params.block_size = dim3(preferred_block_size, 1, 1);
    params.grid_size = dim3((num_elements + preferred_block_size - 1) / preferred_block_size, 1, 1);
    params.shared_memory_bytes = 0;
    return params;
}

KernelLaunchParams KernelLaunchParams::computeFor2D(int width, int height, int preferred_block_size) {
    KernelLaunchParams params;
    int block_dim = (int)sqrt(preferred_block_size);
    params.block_size = dim3(block_dim, block_dim, 1);
    params.grid_size = dim3(
        (width + block_dim - 1) / block_dim,
        (height + block_dim - 1) / block_dim,
        1
    );
    params.shared_memory_bytes = 0;
    return params;
}

KernelLaunchParams KernelLaunchParams::computeForContacts(int num_contacts, int preferred_block_size) {
    KernelLaunchParams params;
    params.block_size = dim3(preferred_block_size, 1, 1);
    params.grid_size = dim3((num_contacts + preferred_block_size - 1) / preferred_block_size, 1, 1);
    params.shared_memory_bytes = preferred_block_size * sizeof(float); // For reductions
    return params;
}

// GPU memory management implementation
VariationalContactGPUData* VariationalContactGPUManager::allocateGPUData(
    int n_bodies,
    int max_contacts,
    int max_contact_pairs,
    int num_hash_cells
) {
    VariationalContactGPUData* gpu_data = new VariationalContactGPUData{};

    gpu_data->n_bodies = n_bodies;
    gpu_data->max_contacts = max_contacts;
    gpu_data->max_contact_pairs = max_contact_pairs;
    gpu_data->num_hash_cells = num_hash_cells;

    try {
        // Allocate body state arrays
        CUDA_CHECK(cudaMalloc(&gpu_data->d_positions, 3 * n_bodies * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_velocities, 3 * n_bodies * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_masses, n_bodies * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_radii, n_bodies * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_material_ids, n_bodies * sizeof(int)));

        // Allocate contact constraint arrays
        CUDA_CHECK(cudaMalloc(&gpu_data->d_contacts, max_contacts * sizeof(GPUContactConstraint)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_contact_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_contact_pairs, 2 * max_contact_pairs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_pair_count, sizeof(int)));

        // Allocate force computation arrays
        CUDA_CHECK(cudaMalloc(&gpu_data->d_contact_forces, 3 * n_bodies * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_contact_energy, sizeof(float)));

        // Allocate Newton solver arrays
        CUDA_CHECK(cudaMalloc(&gpu_data->d_newton_residual, 6 * n_bodies * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_newton_delta, 6 * n_bodies * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_line_search_energies, 16 * sizeof(float)));

        // Allocate gradient computation arrays
        CUDA_CHECK(cudaMalloc(&gpu_data->d_position_gradients, 3 * n_bodies * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_velocity_gradients, 3 * n_bodies * sizeof(float)));

        // Allocate spatial hash arrays
        CUDA_CHECK(cudaMalloc(&gpu_data->d_spatial_hash_keys, n_bodies * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_spatial_hash_values, n_bodies * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_cell_starts, num_hash_cells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&gpu_data->d_cell_ends, num_hash_cells * sizeof(int)));

        // Create CUDA streams
        CUDA_CHECK(cudaStreamCreate(&gpu_data->compute_stream));
        CUDA_CHECK(cudaStreamCreate(&gpu_data->transfer_stream));
        CUDA_CHECK(cudaStreamCreate(&gpu_data->gradient_stream));

        // Initialize memory
        CUDA_CHECK(cudaMemset(gpu_data->d_contact_count, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(gpu_data->d_pair_count, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(gpu_data->d_contact_forces, 0, 3 * n_bodies * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpu_data->d_contact_energy, 0, sizeof(float)));

        // Calculate total allocated memory
        gpu_data->total_allocated_bytes =
            3 * n_bodies * sizeof(float) * 5 +  // positions, velocities, forces, gradients
            n_bodies * sizeof(float) * 2 +      // masses, radii
            n_bodies * sizeof(int) * 3 +        // material_ids, hash keys/values
            max_contacts * sizeof(GPUContactConstraint) +
            2 * max_contact_pairs * sizeof(int) +
            6 * n_bodies * sizeof(float) * 2 +  // Newton arrays
            num_hash_cells * sizeof(int) * 2 +  // cell starts/ends
            sizeof(int) * 2 +                   // contact/pair counts
            sizeof(float) * 17;                 // energy arrays

        std::cout << "Allocated " << (gpu_data->total_allocated_bytes / 1024.0 / 1024.0)
                  << " MB of GPU memory for " << n_bodies << " bodies" << std::endl;

    } catch (const std::exception& e) {
        deallocateGPUData(gpu_data);
        throw;
    }

    return gpu_data;
}

void VariationalContactGPUManager::deallocateGPUData(VariationalContactGPUData* gpu_data) {
    if (!gpu_data) return;

    // Free all GPU memory
    cudaFree(gpu_data->d_positions);
    cudaFree(gpu_data->d_velocities);
    cudaFree(gpu_data->d_masses);
    cudaFree(gpu_data->d_radii);
    cudaFree(gpu_data->d_material_ids);
    cudaFree(gpu_data->d_contacts);
    cudaFree(gpu_data->d_contact_count);
    cudaFree(gpu_data->d_contact_pairs);
    cudaFree(gpu_data->d_pair_count);
    cudaFree(gpu_data->d_contact_forces);
    cudaFree(gpu_data->d_contact_energy);
    cudaFree(gpu_data->d_newton_residual);
    cudaFree(gpu_data->d_newton_delta);
    cudaFree(gpu_data->d_line_search_energies);
    cudaFree(gpu_data->d_position_gradients);
    cudaFree(gpu_data->d_velocity_gradients);
    cudaFree(gpu_data->d_spatial_hash_keys);
    cudaFree(gpu_data->d_spatial_hash_values);
    cudaFree(gpu_data->d_cell_starts);
    cudaFree(gpu_data->d_cell_ends);

    // Destroy streams
    cudaStreamDestroy(gpu_data->compute_stream);
    cudaStreamDestroy(gpu_data->transfer_stream);
    cudaStreamDestroy(gpu_data->gradient_stream);

    delete gpu_data;
}

void VariationalContactGPUManager::copyToGPU(
    VariationalContactGPUData* gpu_data,
    const std::vector<Eigen::Vector3d>& positions,
    const std::vector<Eigen::Vector3d>& velocities,
    const std::vector<double>& masses,
    const std::vector<double>& radii,
    const std::vector<int>& material_ids,
    cudaStream_t stream
) {
    int n = positions.size();

    // Convert Eigen data to packed float arrays
    std::vector<float> pos_packed(3 * n), vel_packed(3 * n);
    std::vector<float> masses_f(n), radii_f(n);

    for (int i = 0; i < n; ++i) {
        pos_packed[3 * i + 0] = static_cast<float>(positions[i][0]);
        pos_packed[3 * i + 1] = static_cast<float>(positions[i][1]);
        pos_packed[3 * i + 2] = static_cast<float>(positions[i][2]);

        vel_packed[3 * i + 0] = static_cast<float>(velocities[i][0]);
        vel_packed[3 * i + 1] = static_cast<float>(velocities[i][1]);
        vel_packed[3 * i + 2] = static_cast<float>(velocities[i][2]);

        masses_f[i] = static_cast<float>(masses[i]);
        radii_f[i] = static_cast<float>(radii[i]);
    }

    // Async memory transfers
    CUDA_CHECK(cudaMemcpyAsync(gpu_data->d_positions, pos_packed.data(),
                              3 * n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu_data->d_velocities, vel_packed.data(),
                              3 * n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu_data->d_masses, masses_f.data(),
                              n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu_data->d_radii, radii_f.data(),
                              n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu_data->d_material_ids, material_ids.data(),
                              n * sizeof(int), cudaMemcpyHostToDevice, stream));
}

void VariationalContactGPUManager::copyFromGPU(
    const VariationalContactGPUData* gpu_data,
    std::vector<Eigen::Vector3d>& positions,
    std::vector<Eigen::Vector3d>& velocities,
    std::vector<Eigen::Vector3d>& forces,
    cudaStream_t stream
) {
    int n = gpu_data->n_bodies;

    // Temporary host arrays
    std::vector<float> pos_packed(3 * n), vel_packed(3 * n), force_packed(3 * n);

    // Copy from GPU
    CUDA_CHECK(cudaMemcpyAsync(pos_packed.data(), gpu_data->d_positions,
                              3 * n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(vel_packed.data(), gpu_data->d_velocities,
                              3 * n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(force_packed.data(), gpu_data->d_contact_forces,
                              3 * n * sizeof(float), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Convert back to Eigen format
    positions.resize(n);
    velocities.resize(n);
    forces.resize(n);

    for (int i = 0; i < n; ++i) {
        positions[i] = Eigen::Vector3d(pos_packed[3 * i + 0], pos_packed[3 * i + 1], pos_packed[3 * i + 2]);
        velocities[i] = Eigen::Vector3d(vel_packed[3 * i + 0], vel_packed[3 * i + 1], vel_packed[3 * i + 2]);
        forces[i] = Eigen::Vector3d(force_packed[3 * i + 0], force_packed[3 * i + 1], force_packed[3 * i + 2]);
    }
}

size_t VariationalContactGPUManager::getMemoryUsage(const VariationalContactGPUData* gpu_data) {
    return gpu_data ? gpu_data->total_allocated_bytes : 0;
}

// GPU solver implementation
VariationalContactSolverGPU::VariationalContactSolverGPU(const VariationalContactParams& p)
    : params(p), gpu_initialized(false), current_n_bodies(0) {

    // Check GPU availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("No CUDA-capable GPU found");
    }

    // Set GPU device properties
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));

    std::cout << "Using GPU: " << device_prop.name
              << " (Compute " << device_prop.major << "." << device_prop.minor << ")" << std::endl;
    std::cout << "Global memory: " << (device_prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
}

VariationalContactSolverGPU::~VariationalContactSolverGPU() {
    if (gpu_data) {
        VariationalContactGPUManager::deallocateGPUData(gpu_data.release());
    }
}

void VariationalContactSolverGPU::initializeGPU(int n_bodies) {
    if (gpu_initialized && current_n_bodies >= n_bodies) {
        return; // Already initialized with sufficient capacity
    }

    // Deallocate previous GPU data if exists
    if (gpu_data) {
        VariationalContactGPUManager::deallocateGPUData(gpu_data.release());
    }

    // Allocate new GPU data with some extra capacity
    int capacity = std::max(n_bodies, static_cast<int>(n_bodies * 1.2)); // 20% extra capacity
    int max_contacts = std::min(capacity * capacity / 2, 100000); // Reasonable upper bound
    int max_pairs = std::min(capacity * 50, 500000); // Estimate for broad phase

    gpu_data.reset(VariationalContactGPUManager::allocateGPUData(
        capacity, max_contacts, max_pairs, 32768
    ));

    // TODO: Set GPU parameters in constant memory (currently disabled due to linking issues)
    // CUDA_CHECK(cudaMemcpyToSymbol(c_barrier_stiffness, &params.barrier_stiffness, sizeof(float)));
    // CUDA_CHECK(cudaMemcpyToSymbol(c_barrier_threshold, &params.barrier_threshold, sizeof(float)));
    // CUDA_CHECK(cudaMemcpyToSymbol(c_friction_regularization, &params.friction_regularization, sizeof(float)));
    // CUDA_CHECK(cudaMemcpyToSymbol(c_max_newton_iterations, &params.max_newton_iterations, sizeof(int)));
    // CUDA_CHECK(cudaMemcpyToSymbol(c_newton_tolerance, &params.newton_tolerance, sizeof(float)));

    // For now, store parameters in GPU data structure
    gpu_data->barrier_stiffness = static_cast<float>(params.barrier_stiffness);
    gpu_data->barrier_threshold = static_cast<float>(params.barrier_threshold);
    gpu_data->friction_regularization = static_cast<float>(params.friction_regularization);
    gpu_data->max_newton_iterations = params.max_newton_iterations;
    gpu_data->newton_tolerance = static_cast<float>(params.newton_tolerance);

    current_n_bodies = capacity;
    gpu_initialized = true;

    std::cout << "GPU initialized for up to " << capacity << " bodies" << std::endl;
}

void VariationalContactSolverGPU::detectContactsGPU(
    const std::vector<Eigen::Vector3d>& positions,
    const std::vector<double>& radii,
    const std::vector<int>& material_ids
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int n_bodies = positions.size();
    ensureGPUCapacity(n_bodies);

    // Reset contact counts
    CUDA_CHECK(cudaMemsetAsync(gpu_data->d_contact_count, 0, sizeof(int), gpu_data->compute_stream));
    CUDA_CHECK(cudaMemsetAsync(gpu_data->d_pair_count, 0, sizeof(int), gpu_data->compute_stream));

    // Convert radii to float for GPU
    std::vector<float> radii_f(n_bodies);
    for (int i = 0; i < n_bodies; ++i) {
        radii_f[i] = static_cast<float>(radii[i]);
    }

    // Upload radii and material IDs
    CUDA_CHECK(cudaMemcpyAsync(gpu_data->d_radii, radii_f.data(),
                              n_bodies * sizeof(float), cudaMemcpyHostToDevice, gpu_data->transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu_data->d_material_ids, material_ids.data(),
                              n_bodies * sizeof(int), cudaMemcpyHostToDevice, gpu_data->transfer_stream));

    // Spatial hash computation
    float cell_size = 2.0f * (*std::max_element(radii_f.begin(), radii_f.end())) + static_cast<float>(params.barrier_threshold);
    float3 world_min = make_float3(-100.0f, -100.0f, -100.0f); // Reasonable world bounds
    int3 grid_size = make_int3(64, 64, 64); // 64^3 grid

    auto hash_params = KernelLaunchParams::computeFor1D(n_bodies);
    computeSpatialHashKernel<<<hash_params.grid_size, hash_params.block_size, 0, gpu_data->compute_stream>>>(
        gpu_data->d_positions,
        gpu_data->d_radii,
        gpu_data->d_spatial_hash_keys,
        gpu_data->d_spatial_hash_values,
        n_bodies,
        cell_size,
        world_min,
        grid_size
    );
    CUDA_CHECK_KERNEL();

    // Sort by hash keys for spatial coherence
    thrust::device_ptr<int> keys_ptr(gpu_data->d_spatial_hash_keys);
    thrust::device_ptr<int> values_ptr(gpu_data->d_spatial_hash_values);
    thrust::sort_by_key(keys_ptr, keys_ptr + n_bodies, values_ptr);

    // Find cell boundaries
    thrust::device_ptr<int> cell_starts_ptr(gpu_data->d_cell_starts);
    thrust::device_ptr<int> cell_ends_ptr(gpu_data->d_cell_ends);

    // Initialize cell arrays
    thrust::fill(cell_starts_ptr, cell_starts_ptr + gpu_data->num_hash_cells, -1);
    thrust::fill(cell_ends_ptr, cell_ends_ptr + gpu_data->num_hash_cells, -1);

    // Detect contact pairs using spatial hash
    auto contact_params = KernelLaunchParams::computeFor1D(n_bodies);
    detectContactPairsKernel<<<contact_params.grid_size, contact_params.block_size, 0, gpu_data->compute_stream>>>(
        gpu_data->d_positions,
        gpu_data->d_radii,
        gpu_data->d_material_ids,
        gpu_data->d_cell_starts,
        gpu_data->d_cell_ends,
        gpu_data->d_spatial_hash_keys,
        gpu_data->d_contact_pairs,
        gpu_data->d_pair_count,
        n_bodies,
        gpu_data->max_contact_pairs,
        static_cast<float>(params.barrier_threshold)
    );
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaStreamSynchronize(gpu_data->compute_stream));

    auto end_time = std::chrono::high_resolution_clock::now();
    last_metrics.contact_detection_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
}

void VariationalContactSolverGPU::computeContactForces(
    const std::vector<Eigen::Vector3d>& positions,
    const std::vector<Eigen::Vector3d>& velocities,
    const std::vector<double>& masses,
    const std::vector<double>& radii,
    const std::vector<int>& material_ids,
    std::vector<Eigen::Vector3d>& forces
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int n_bodies = positions.size();
    ensureGPUCapacity(n_bodies);

    // Upload data to GPU
    VariationalContactGPUManager::copyToGPU(gpu_data.get(), positions, velocities, masses, radii, material_ids,
                                           gpu_data->transfer_stream);

    // Detect contacts
    detectContactsGPU(positions, radii, material_ids);

    // Get number of contact pairs
    int pair_count;
    CUDA_CHECK(cudaMemcpyAsync(&pair_count, gpu_data->d_pair_count, sizeof(int),
                              cudaMemcpyDeviceToHost, gpu_data->compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(gpu_data->compute_stream));

    if (pair_count > 0) {
        // Compute barrier potentials and contact constraints
        auto potential_params = KernelLaunchParams::computeFor1D(pair_count);
        computeBarrierPotentialsKernel<<<potential_params.grid_size, potential_params.block_size, 0, gpu_data->compute_stream>>>(
            gpu_data->d_positions,
            gpu_data->d_radii,
            gpu_data->d_material_ids,
            gpu_data->d_contact_pairs,
            pair_count,
            gpu_data->d_contacts,
            gpu_data->d_contact_count,
            static_cast<float>(params.barrier_stiffness),
            static_cast<float>(params.barrier_threshold),
            gpu_data->max_contacts
        );
        CUDA_CHECK_KERNEL();

        // Get actual contact count
        int contact_count;
        CUDA_CHECK(cudaMemcpyAsync(&contact_count, gpu_data->d_contact_count, sizeof(int),
                                  cudaMemcpyDeviceToHost, gpu_data->compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu_data->compute_stream));

        if (contact_count > 0) {
            // Reset force array
            CUDA_CHECK(cudaMemsetAsync(gpu_data->d_contact_forces, 0, 3 * n_bodies * sizeof(float), gpu_data->compute_stream));

            // Compute contact forces
            auto force_params = KernelLaunchParams::computeFor1D(contact_count);
            computeContactForcesKernel<<<force_params.grid_size, force_params.block_size, 0, gpu_data->compute_stream>>>(
                gpu_data->d_contacts,
                contact_count,
                gpu_data->d_contact_forces,
                n_bodies
            );
            CUDA_CHECK_KERNEL();
        }
    }

    // Download results
    std::vector<Eigen::Vector3d> dummy_positions, dummy_velocities;
    VariationalContactGPUManager::copyFromGPU(gpu_data.get(), dummy_positions, dummy_velocities, forces,
                                             gpu_data->transfer_stream);

    auto end_time = std::chrono::high_resolution_clock::now();
    last_metrics.force_computation_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
}

double VariationalContactSolverGPU::computeContactEnergy(
    const std::vector<Eigen::Vector3d>& positions,
    const std::vector<double>& radii,
    const std::vector<int>& material_ids
) const {
    int n_bodies = positions.size();

    // Upload data and detect contacts (similar to force computation)
    VariationalContactGPUManager::copyToGPU(gpu_data.get(), positions,
                                           std::vector<Eigen::Vector3d>(n_bodies, Eigen::Vector3d::Zero()),
                                           std::vector<double>(n_bodies, 1.0), radii, material_ids,
                                           gpu_data->transfer_stream);

    const_cast<VariationalContactSolverGPU*>(this)->detectContactsGPU(positions, radii, material_ids);

    // Reset energy accumulator
    CUDA_CHECK(cudaMemsetAsync(gpu_data->d_contact_energy, 0, sizeof(float), gpu_data->compute_stream));

    // Get contact count
    int contact_count;
    CUDA_CHECK(cudaMemcpyAsync(&contact_count, gpu_data->d_contact_count, sizeof(int),
                              cudaMemcpyDeviceToHost, gpu_data->compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(gpu_data->compute_stream));

    if (contact_count > 0) {
        // Compute total contact energy
        auto energy_params = KernelLaunchParams::computeForContacts(contact_count);
        computeContactEnergyKernel<<<energy_params.grid_size, energy_params.block_size,
                                   energy_params.shared_memory_bytes, gpu_data->compute_stream>>>(
            gpu_data->d_contacts,
            contact_count,
            gpu_data->d_contact_energy
        );
        CUDA_CHECK_KERNEL();
    }

    // Download energy result
    float total_energy;
    CUDA_CHECK(cudaMemcpyAsync(&total_energy, gpu_data->d_contact_energy, sizeof(float),
                              cudaMemcpyDeviceToHost, gpu_data->compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(gpu_data->compute_stream));

    return static_cast<double>(total_energy);
}

void VariationalContactSolverGPU::ensureGPUCapacity(int n_bodies) {
    if (!gpu_initialized || current_n_bodies < n_bodies) {
        initializeGPU(n_bodies);
    }
}

float VariationalContactSolverGPU::getGPUMemoryUsageMB() const {
    if (!gpu_data) return 0.0f;
    return VariationalContactGPUManager::getMemoryUsage(gpu_data.get()) / 1024.0f / 1024.0f;
}

void VariationalContactSolverGPU::synchronizeGPU() {
    if (gpu_data) {
        CUDA_CHECK(cudaStreamSynchronize(gpu_data->compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu_data->transfer_stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu_data->gradient_stream));
    }
}

bool VariationalContactSolverGPU::isGPUAvailable() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

std::vector<std::string> VariationalContactSolverGPU::getAvailableGPUDevices() {
    std::vector<std::string> devices;
    int device_count;

    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        return devices;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            devices.push_back(std::string(prop.name) + " (Compute " +
                            std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")");
        }
    }

    return devices;
}

} // namespace physgrad