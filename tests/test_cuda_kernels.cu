// PhysGrad CUDA Kernels Unit Tests

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>

// Include physgrad kernels
namespace physgrad {
    // Declare kernel functions
    __global__ void verlet_integration_kernel(
        float3* positions, float3* velocities, const float3* forces,
        const float* masses, float dt, int num_particles
    );

    __global__ void classical_force_kernel(
        const float3* positions, const float* charges,
        float3* forces, int num_particles
    );

    __global__ void memory_operations_kernel(
        float* data, int size
    );
}

// Include kernel headers
extern "C" {
    // Declare kernel functions
    void launch_verlet_integration_test(
        float3* positions, float3* velocities, const float3* forces,
        const float* masses, float dt, int num_particles
    );

    void launch_classical_force_test(
        const float3* positions, const float* charges,
        float3* forces, int num_particles
    );

    void launch_memory_operations_test(
        float* data, int size
    );
}

class CudaKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Set CUDA device
        cudaSetDevice(0);

        // Get device properties
        cudaGetDeviceProperties(&device_props_, 0);
        std::cout << "Testing on: " << device_props_.name << std::endl;
    }

    void TearDown() override {
        // Synchronize and reset device
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }

    cudaDeviceProp device_props_;
};

TEST_F(CudaKernelTest, VerletIntegrationKernel) {
    const int num_particles = 1000;

    // Host data
    std::vector<float3> h_positions(num_particles);
    std::vector<float3> h_velocities(num_particles);
    std::vector<float3> h_forces(num_particles);
    std::vector<float> h_masses(num_particles);

    // Initialize test data
    for (int i = 0; i < num_particles; ++i) {
        h_positions[i] = {0.0f, 0.0f, 0.0f};
        h_velocities[i] = {1.0f, 0.0f, 0.0f};
        h_forces[i] = {0.1f, 0.0f, 0.0f};
        h_masses[i] = 1.0f;
    }

    // Device data
    float3 *d_positions, *d_velocities, *d_forces;
    float *d_masses;

    cudaMalloc(&d_positions, num_particles * sizeof(float3));
    cudaMalloc(&d_velocities, num_particles * sizeof(float3));
    cudaMalloc(&d_forces, num_particles * sizeof(float3));
    cudaMalloc(&d_masses, num_particles * sizeof(float));

    // Copy to device
    cudaMemcpy(d_positions, h_positions.data(), num_particles * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, h_velocities.data(), num_particles * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_forces, h_forces.data(), num_particles * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, h_masses.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    float dt = 0.01f;
    launch_verlet_integration_test(d_positions, d_velocities, d_forces, d_masses, dt, num_particles);

    // Check for kernel errors
    cudaError_t kernel_error = cudaGetLastError();
    EXPECT_EQ(kernel_error, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(kernel_error);

    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_positions.data(), d_positions, num_particles * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_velocities.data(), d_velocities, num_particles * sizeof(float3), cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < num_particles; ++i) {
        // Velocity should have increased due to force
        EXPECT_GT(h_velocities[i].x, 1.0f);

        // Position should have changed due to velocity
        EXPECT_GT(h_positions[i].x, 0.0f);
    }

    // Cleanup
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_forces);
    cudaFree(d_masses);
}

TEST_F(CudaKernelTest, ClassicalForceKernel) {
    const int num_particles = 100;

    // Host data
    std::vector<float3> h_positions(num_particles);
    std::vector<float> h_charges(num_particles);
    std::vector<float3> h_forces(num_particles, {0.0f, 0.0f, 0.0f});

    // Create test system: alternating positive and negative charges
    for (int i = 0; i < num_particles; ++i) {
        h_positions[i] = {static_cast<float>(i) * 0.1f, 0.0f, 0.0f};
        h_charges[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    // Device data
    float3 *d_positions, *d_forces;
    float *d_charges;

    cudaMalloc(&d_positions, num_particles * sizeof(float3));
    cudaMalloc(&d_charges, num_particles * sizeof(float));
    cudaMalloc(&d_forces, num_particles * sizeof(float3));

    // Copy to device
    cudaMemcpy(d_positions, h_positions.data(), num_particles * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_charges, h_charges.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    launch_classical_force_test(d_positions, d_charges, d_forces, num_particles);

    // Check for kernel errors
    cudaError_t kernel_error = cudaGetLastError();
    EXPECT_EQ(kernel_error, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(kernel_error);

    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_forces.data(), d_forces, num_particles * sizeof(float3), cudaMemcpyDeviceToHost);

    // Verify results - forces should be non-zero for most particles
    int non_zero_forces = 0;
    for (int i = 0; i < num_particles; ++i) {
        if (h_forces[i].x != 0.0f || h_forces[i].y != 0.0f || h_forces[i].z != 0.0f) {
            non_zero_forces++;
        }
    }

    EXPECT_GT(non_zero_forces, num_particles / 2);

    // Cleanup
    cudaFree(d_positions);
    cudaFree(d_charges);
    cudaFree(d_forces);
}

TEST_F(CudaKernelTest, MemoryOperations) {
    const int size = 1024 * 1024;  // 1M elements

    // Host data
    std::vector<float> h_data(size);
    for (int i = 0; i < size; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Device data
    float *d_data;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Test memory operations
    launch_memory_operations_test(d_data, size);

    // Check for kernel errors
    cudaError_t kernel_error = cudaGetLastError();
    EXPECT_EQ(kernel_error, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(kernel_error);

    cudaDeviceSynchronize();

    // Copy results back and verify
    cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify that memory operations completed successfully
    // Our kernel does: data[i] = data[i] * 2.0f + i
    // For index 0: 0 * 2 + 0 = 0, so check index 1 instead
    EXPECT_EQ(h_data[1], 3.0f);  // Should be: 1 * 2 + 1 = 3

    cudaFree(d_data);
}

TEST_F(CudaKernelTest, DeviceProperties) {
    // Test that we're running on a capable device
    EXPECT_GT(device_props_.multiProcessorCount, 0);
    EXPECT_GT(device_props_.maxThreadsPerBlock, 0);
    EXPECT_GT(device_props_.totalGlobalMem, 0);

    std::cout << "Device properties:" << std::endl;
    std::cout << "  Multiprocessors: " << device_props_.multiProcessorCount << std::endl;
    std::cout << "  Max threads per block: " << device_props_.maxThreadsPerBlock << std::endl;
    std::cout << "  Global memory: " << device_props_.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "  Compute capability: " << device_props_.major << "." << device_props_.minor << std::endl;
}

TEST_F(CudaKernelTest, MemoryBandwidth) {
    const int size = 32 * 1024 * 1024;  // 32M floats = 128MB

    float *d_src, *d_dst;
    cudaMalloc(&d_src, size * sizeof(float));
    cudaMalloc(&d_dst, size * sizeof(float));

    // Initialize source data
    std::vector<float> h_src(size, 1.0f);
    cudaMemcpy(d_src, h_src.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Time memory copy operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Simple copy kernel
    dim3 block_size(256);
    dim3 grid_size((size + block_size.x - 1) / block_size.x);

    // We would need to implement a simple copy kernel for this test
    // For now, just use cudaMemcpy as a baseline
    cudaMemcpy(d_dst, d_src, size * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Calculate bandwidth
    float bandwidth_gb_s = (2.0f * size * sizeof(float)) / (elapsed_ms * 1e6);  // GB/s

    std::cout << "Memory bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;

    // Should achieve reasonable bandwidth (> 100 GB/s on modern GPUs)
    EXPECT_GT(bandwidth_gb_s, 50.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
}

// Wrapper functions for kernel launches (to be implemented in separate .cu file)
extern "C" void launch_verlet_integration_test(
    float3* positions, float3* velocities, const float3* forces,
    const float* masses, float dt, int num_particles) {

    dim3 block_size(256);
    dim3 grid_size((num_particles + block_size.x - 1) / block_size.x);

    physgrad::verlet_integration_kernel<<<grid_size, block_size>>>(positions, velocities, forces, masses, dt, num_particles);
}

extern "C" void launch_classical_force_test(
    const float3* positions, const float* charges,
    float3* forces, int num_particles) {

    dim3 block_size(256);
    dim3 grid_size((num_particles + block_size.x - 1) / block_size.x);

    physgrad::classical_force_kernel<<<grid_size, block_size>>>(positions, charges, forces, num_particles);
}

extern "C" void launch_memory_operations_test(float* data, int size) {
    dim3 block_size(256);
    dim3 grid_size((size + block_size.x - 1) / block_size.x);

    physgrad::memory_operations_kernel<<<grid_size, block_size>>>(data, size);
}