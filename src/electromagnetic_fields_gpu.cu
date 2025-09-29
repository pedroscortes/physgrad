/**
 * GPU-Accelerated Electromagnetic Field Implementation - CUDA Implementation
 *
 * CUDA-specific implementation for electromagnetic field simulation
 */

#ifdef HAVE_CUDA

#include "electromagnetic_fields.h"
#include "electromagnetic_kernels.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <iostream>
#include <vector>

namespace physgrad {

/**
 * CUDA-specific implementation of EMGPUData
 */
class EMGPUDataImpl {
public:
    GPUEMGrid gpu_grid;
    thrust::device_vector<GPUChargedParticle> gpu_particles;
    thrust::device_vector<GPUEMSource> gpu_sources;
    thrust::device_vector<float> energy_density;
    thrust::device_vector<float3> poynting_vector;

    // FFT plans for Fourier analysis
    cufftHandle fft_plan_forward;
    cufftHandle fft_plan_inverse;
    thrust::device_vector<cufftComplex> fft_buffer;

    // CUDA streams for concurrent execution
    cudaStream_t compute_stream;
    cudaStream_t memory_stream;

    bool is_allocated = false;
    size_t total_memory_bytes = 0;

    ~EMGPUDataImpl() {
        deallocate();
    }

    void allocate(int nx, int ny, int nz, int max_particles, int max_sources) {
        if (is_allocated) deallocate();

        size_t grid_size = nx * ny * nz;

        // Allocate electromagnetic field arrays
        cudaMalloc(&gpu_grid.Ex, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Ey, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Ez, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Hx, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Hy, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Hz, grid_size * sizeof(float));

        cudaMalloc(&gpu_grid.Ex_prev, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Ey_prev, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Ez_prev, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Hx_prev, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Hy_prev, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Hz_prev, grid_size * sizeof(float));

        // Material properties
        cudaMalloc(&gpu_grid.epsilon, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.mu, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.sigma, grid_size * sizeof(float));

        // PML coefficients
        cudaMalloc(&gpu_grid.pml_sx, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.pml_sy, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.pml_sz, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.pml_ax, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.pml_ay, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.pml_az, grid_size * sizeof(float));

        // Current and charge densities
        cudaMalloc(&gpu_grid.Jx, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Jy, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.Jz, grid_size * sizeof(float));
        cudaMalloc(&gpu_grid.rho, grid_size * sizeof(float));

        // Initialize all fields to zero
        cudaMemset(gpu_grid.Ex, 0, grid_size * sizeof(float));
        cudaMemset(gpu_grid.Ey, 0, grid_size * sizeof(float));
        cudaMemset(gpu_grid.Ez, 0, grid_size * sizeof(float));
        cudaMemset(gpu_grid.Hx, 0, grid_size * sizeof(float));
        cudaMemset(gpu_grid.Hy, 0, grid_size * sizeof(float));
        cudaMemset(gpu_grid.Hz, 0, grid_size * sizeof(float));

        cudaMemset(gpu_grid.Jx, 0, grid_size * sizeof(float));
        cudaMemset(gpu_grid.Jy, 0, grid_size * sizeof(float));
        cudaMemset(gpu_grid.Jz, 0, grid_size * sizeof(float));
        cudaMemset(gpu_grid.rho, 0, grid_size * sizeof(float));

        // Resize vectors
        gpu_particles.resize(max_particles);
        gpu_sources.resize(max_sources);
        energy_density.resize(grid_size);
        poynting_vector.resize(grid_size);
        fft_buffer.resize(grid_size);

        // Create FFT plans
        int n[3] = {nz, ny, nx};
        cufftPlanMany(&fft_plan_forward, 3, n, nullptr, 1, grid_size,
                      nullptr, 1, grid_size, CUFFT_R2C, 1);
        cufftPlanMany(&fft_plan_inverse, 3, n, nullptr, 1, grid_size,
                      nullptr, 1, grid_size, CUFFT_C2R, 1);

        // Create CUDA streams
        cudaStreamCreate(&compute_stream);
        cudaStreamCreate(&memory_stream);

        // Calculate total memory usage
        total_memory_bytes = grid_size * sizeof(float) * 18 +  // EM fields + materials + PML
                            max_particles * sizeof(GPUChargedParticle) +
                            max_sources * sizeof(GPUEMSource) +
                            grid_size * sizeof(float) +        // energy density
                            grid_size * sizeof(float3) +       // poynting vector
                            grid_size * sizeof(cufftComplex);  // FFT buffer

        is_allocated = true;
    }

    void deallocate() {
        if (!is_allocated) return;

        // Free EM field arrays
        cudaFree(gpu_grid.Ex);
        cudaFree(gpu_grid.Ey);
        cudaFree(gpu_grid.Ez);
        cudaFree(gpu_grid.Hx);
        cudaFree(gpu_grid.Hy);
        cudaFree(gpu_grid.Hz);

        cudaFree(gpu_grid.Ex_prev);
        cudaFree(gpu_grid.Ey_prev);
        cudaFree(gpu_grid.Ez_prev);
        cudaFree(gpu_grid.Hx_prev);
        cudaFree(gpu_grid.Hy_prev);
        cudaFree(gpu_grid.Hz_prev);

        // Free material arrays
        cudaFree(gpu_grid.epsilon);
        cudaFree(gpu_grid.mu);
        cudaFree(gpu_grid.sigma);

        // Free PML arrays
        cudaFree(gpu_grid.pml_sx);
        cudaFree(gpu_grid.pml_sy);
        cudaFree(gpu_grid.pml_sz);
        cudaFree(gpu_grid.pml_ax);
        cudaFree(gpu_grid.pml_ay);
        cudaFree(gpu_grid.pml_az);

        // Free current/charge arrays
        cudaFree(gpu_grid.Jx);
        cudaFree(gpu_grid.Jy);
        cudaFree(gpu_grid.Jz);
        cudaFree(gpu_grid.rho);

        // Destroy FFT plans
        cufftDestroy(fft_plan_forward);
        cufftDestroy(fft_plan_inverse);

        // Destroy streams
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(memory_stream);

        is_allocated = false;
        total_memory_bytes = 0;
    }

    void addSource(const GPUEMSource& gpu_source) {
        gpu_sources.push_back(gpu_source);
    }

    void addParticle(const GPUChargedParticle& gpu_particle) {
        gpu_particles.push_back(gpu_particle);
    }

    void simulateStep(double dt) {
        static double simulation_time = 0.0;

        // Launch CUDA kernels for electromagnetic field update
        dim3 block = getOptimalBlockSize();
        dim3 grid = getGridSize(gpu_grid.nx, gpu_grid.ny, gpu_grid.nz);

        // Update magnetic field first (leap-frog integration)
        updateMagneticFieldKernel<<<grid, block, 0, compute_stream>>>(gpu_grid);

        // Apply sources
        if (!gpu_sources.empty()) {
            dim3 source_grid(gpu_sources.size(),
                            (gpu_grid.nx + block.y - 1) / block.y,
                            (gpu_grid.ny + block.z - 1) / block.z);

            applySourcesKernel<<<source_grid, block, 0, compute_stream>>>(
                gpu_grid,
                thrust::raw_pointer_cast(gpu_sources.data()),
                gpu_sources.size(),
                simulation_time
            );
        }

        // Update electric field
        updateElectricFieldKernel<<<grid, block, 0, compute_stream>>>(gpu_grid);

        // Update charged particles
        if (!gpu_particles.empty()) {
            int particle_threads = 256;
            int particle_blocks = (gpu_particles.size() + particle_threads - 1) / particle_threads;

            updateChargedParticlesKernel<<<particle_blocks, particle_threads, 0, compute_stream>>>(
                thrust::raw_pointer_cast(gpu_particles.data()),
                gpu_particles.size(),
                gpu_grid,
                dt
            );
        }

        cudaStreamSynchronize(compute_stream);
        simulation_time += dt;
    }

    double getTotalElectromagneticEnergy() {
        // Compute energy density on GPU
        dim3 block = getOptimalBlockSize();
        dim3 grid = getGridSize(gpu_grid.nx, gpu_grid.ny, gpu_grid.nz);

        computeEnergyDensityKernel<<<grid, block>>>(
            gpu_grid,
            thrust::raw_pointer_cast(energy_density.data())
        );
        cudaDeviceSynchronize();

        // Sum total energy using Thrust
        return thrust::reduce(energy_density.begin(), energy_density.end(), 0.0);
    }

    void getElectricField(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez) {
        size_t grid_size = gpu_grid.nx * gpu_grid.ny * gpu_grid.nz;

        Ex.resize(grid_size);
        Ey.resize(grid_size);
        Ez.resize(grid_size);

        cudaMemcpy(Ex.data(), gpu_grid.Ex, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Ey.data(), gpu_grid.Ey, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Ez.data(), gpu_grid.Ez, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void getMagneticField(std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz) {
        size_t grid_size = gpu_grid.nx * gpu_grid.ny * gpu_grid.nz;

        Hx.resize(grid_size);
        Hy.resize(grid_size);
        Hz.resize(grid_size);

        cudaMemcpy(Hx.data(), gpu_grid.Hx, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Hy.data(), gpu_grid.Hy, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Hz.data(), gpu_grid.Hz, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void setupPML(int pml_thickness, double pml_absorption) {
        dim3 block = getOptimalBlockSize();
        dim3 grid = getGridSize(gpu_grid.nx, gpu_grid.ny, gpu_grid.nz);

        setupPMLKernel<<<grid, block>>>(gpu_grid, pml_thickness, pml_absorption);
        cudaDeviceSynchronize();
    }

    void setMaterialProperties(const std::vector<float>& epsilon_host,
                               const std::vector<float>& mu_host,
                               const std::vector<float>& sigma_host) {
        size_t grid_size = epsilon_host.size();

        cudaMemcpy(gpu_grid.epsilon, epsilon_host.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_grid.mu, mu_host.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_grid.sigma, sigma_host.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);
    }
};

// CUDA implementation functions called from C++
EMGPUDataImpl* createEMGPUData() {
    return new EMGPUDataImpl();
}

void destroyEMGPUData(EMGPUDataImpl* impl) {
    delete impl;
}

void allocateEMGPUMemory(EMGPUDataImpl* impl, int nx, int ny, int nz, int max_particles, int max_sources) {
    impl->allocate(nx, ny, nz, max_particles, max_sources);
}

void deallocateEMGPUMemory(EMGPUDataImpl* impl) {
    impl->deallocate();
}

void setEMGPUGridParams(EMGPUDataImpl* impl, int nx, int ny, int nz, double dx, double dy, double dz,
                       double dt, double c0, double epsilon0, double mu0) {
    impl->gpu_grid.nx = nx;
    impl->gpu_grid.ny = ny;
    impl->gpu_grid.nz = nz;
    impl->gpu_grid.dx = dx;
    impl->gpu_grid.dy = dy;
    impl->gpu_grid.dz = dz;
    impl->gpu_grid.dt = dt;
    impl->gpu_grid.c0 = c0;
    impl->gpu_grid.epsilon0 = epsilon0;
    impl->gpu_grid.mu0 = mu0;
}

void addEMGPUSource(EMGPUDataImpl* impl, const GPUEMSource& source) {
    impl->addSource(source);
}

void addEMGPUParticle(EMGPUDataImpl* impl, const GPUChargedParticle& particle) {
    impl->addParticle(particle);
}

void simulateEMGPUStep(EMGPUDataImpl* impl, double dt) {
    impl->simulateStep(dt);
}

double getEMGPUTotalEnergy(EMGPUDataImpl* impl) {
    return impl->getTotalElectromagneticEnergy();
}

void getEMGPUElectricField(EMGPUDataImpl* impl, std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez) {
    impl->getElectricField(Ex, Ey, Ez);
}

void getEMGPUMagneticField(EMGPUDataImpl* impl, std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz) {
    impl->getMagneticField(Hx, Hy, Hz);
}

void setupEMGPUPML(EMGPUDataImpl* impl, int thickness, double absorption) {
    impl->setupPML(thickness, absorption);
}

void setEMGPUMaterialProperties(EMGPUDataImpl* impl, const std::vector<float>& epsilon,
                               const std::vector<float>& mu, const std::vector<float>& sigma) {
    impl->setMaterialProperties(epsilon, mu, sigma);
}

size_t getEMGPUMemoryUsage(EMGPUDataImpl* impl) {
    return impl->total_memory_bytes;
}

} // namespace physgrad

#endif // HAVE_CUDA