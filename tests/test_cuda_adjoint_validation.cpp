/**
 * CUDA Adjoint Validation Tests
 *
 * Validates GPU gradient computation against CPU reference implementation
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "adjoint_kernels.h"
#include "adjoint_integrators_standalone.h"
#include "common_types.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace physgrad::adjoint;
using physgrad::ConceptVector3D;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Helper: Copy data to GPU
template<typename T>
T* copyToDevice(const std::vector<T>& host_data) {
    T* device_ptr;
    size_t bytes = host_data.size() * sizeof(T);
    CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
    CUDA_CHECK(cudaMemcpy(device_ptr, host_data.data(), bytes, cudaMemcpyHostToDevice));
    return device_ptr;
}

// Helper: Copy data from GPU
template<typename T>
std::vector<T> copyFromDevice(T* device_ptr, size_t count) {
    std::vector<T> host_data(count);
    CUDA_CHECK(cudaMemcpy(host_data.data(), device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host_data;
}

// Helper: Convert ConceptVector3D to float3
std::vector<float3> toFloat3(const std::vector<ConceptVector3D<float>>& vec) {
    std::vector<float3> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = make_float3(vec[i][0], vec[i][1], vec[i][2]);
    }
    return result;
}

// Helper: Convert float3 to ConceptVector3D
std::vector<ConceptVector3D<float>> fromFloat3(const std::vector<float3>& vec) {
    std::vector<ConceptVector3D<float>> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = ConceptVector3D<float>{vec[i].x, vec[i].y, vec[i].z};
    }
    return result;
}

/**
 * Test 1: Validate Spring Force Parameter Gradients (GPU vs CPU)
 */
TEST(CUDAAdjointValidation, SpringParameterGradients) {
    std::cout << "\n=== Testing Spring Parameter Gradients (GPU vs CPU) ===" << std::endl;

    // Setup simple spring system: 2 particles connected by 1 spring
    const int num_particles = 2;
    const int num_springs = 1;

    std::vector<ConceptVector3D<float>> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    std::vector<float> spring_constants = {10.0f};
    std::vector<float> rest_lengths = {1.0f};

    // Adjoint forces (from backward pass)
    std::vector<ConceptVector3D<float>> adjoint_forces = {
        {1.0f, 0.5f, 0.2f},
        {-1.0f, -0.5f, -0.2f}
    };

    // === CPU Reference Computation ===
    auto cpu_force_engine = std::make_shared<SimpleForceEngine<float>>();
    cpu_force_engine->addSpring(0, 1, spring_constants[0], rest_lengths[0]);

    SimpleForceEngine<float>::ParameterGradients cpu_param_grads;
    cpu_param_grads.spring_constant_grads.resize(num_springs, 0.0f);
    cpu_param_grads.rest_length_grads.resize(num_springs, 0.0f);

    cpu_force_engine->computeForceParameterGradients(positions, adjoint_forces, cpu_param_grads);

    std::cout << "CPU Gradients:" << std::endl;
    std::cout << "  dL/dk  = " << cpu_param_grads.spring_constant_grads[0] << std::endl;
    std::cout << "  dL/dr0 = " << cpu_param_grads.rest_length_grads[0] << std::endl;

    // === GPU Computation ===
    // Convert to float3
    auto positions_f3 = toFloat3(positions);
    auto adjoint_forces_f3 = toFloat3(adjoint_forces);

    // Setup springs for GPU
    std::vector<SpringConnection> springs = {
        {0, 1, spring_constants[0], rest_lengths[0]}
    };

    // Allocate GPU memory
    float3* d_positions = copyToDevice(positions_f3);
    float3* d_adjoint_forces = copyToDevice(adjoint_forces_f3);
    SpringConnection* d_springs = copyToDevice(springs);

    float3* d_grad_positions;
    float* d_grad_spring_k;
    float* d_grad_rest_lengths;

    CUDA_CHECK(cudaMalloc(&d_grad_positions, num_particles * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_grad_spring_k, num_springs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_rest_lengths, num_springs * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_grad_positions, 0, num_particles * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_grad_spring_k, 0, num_springs * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_rest_lengths, 0, num_springs * sizeof(float)));

    // Launch GPU kernel
    launch_spring_force_backward_with_parameters(
        d_adjoint_forces,
        d_grad_positions,
        d_grad_spring_k,
        d_grad_rest_lengths,
        d_positions,
        d_springs,
        num_particles,
        num_springs,
        0  // default stream
    );

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Copy results back
    auto gpu_grad_k = copyFromDevice(d_grad_spring_k, num_springs);
    auto gpu_grad_r0 = copyFromDevice(d_grad_rest_lengths, num_springs);

    std::cout << "\nGPU Gradients:" << std::endl;
    std::cout << "  dL/dk  = " << gpu_grad_k[0] << std::endl;
    std::cout << "  dL/dr0 = " << gpu_grad_r0[0] << std::endl;

    // Compare CPU vs GPU
    float error_k = std::abs(cpu_param_grads.spring_constant_grads[0] - gpu_grad_k[0]) /
                    (std::abs(cpu_param_grads.spring_constant_grads[0]) + 1e-10f);
    float error_r0 = std::abs(cpu_param_grads.rest_length_grads[0] - gpu_grad_r0[0]) /
                     (std::abs(cpu_param_grads.rest_length_grads[0]) + 1e-10f);

    std::cout << "\nRelative Errors:" << std::endl;
    std::cout << "  dL/dk error:  " << (error_k * 100.0f) << "%" << std::endl;
    std::cout << "  dL/dr0 error: " << (error_r0 * 100.0f) << "%" << std::endl;

    // Should match exactly (same analytical formula)
    EXPECT_LT(error_k, 0.001f) << "Spring constant gradient should match CPU within 0.1%";
    EXPECT_LT(error_r0, 0.001f) << "Rest length gradient should match CPU within 0.1%";

    // Cleanup
    cudaFree(d_positions);
    cudaFree(d_adjoint_forces);
    cudaFree(d_springs);
    cudaFree(d_grad_positions);
    cudaFree(d_grad_spring_k);
    cudaFree(d_grad_rest_lengths);

    std::cout << "\n✅ Spring parameter gradients MATCH between CPU and GPU!" << std::endl;
}


/**
 * Test 2: Multi-Spring System
 */
TEST(CUDAAdjointValidation, MultiSpringSystem) {
    std::cout << "\n=== Testing Multi-Spring System (GPU vs CPU) ===" << std::endl;

    const int num_particles = 3;
    const int num_springs = 2;

    // Three particles in a chain: 0 -- 1 -- 2
    std::vector<ConceptVector3D<float>> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.2f, 0.0f, 0.0f},
        {2.5f, 0.0f, 0.0f}
    };

    std::vector<float> spring_constants = {8.0f, 12.0f};
    std::vector<float> rest_lengths = {1.0f, 1.0f};

    std::vector<ConceptVector3D<float>> adjoint_forces = {
        {0.5f, 0.1f, 0.0f},
        {0.3f, -0.2f, 0.0f},
        {-0.8f, 0.1f, 0.0f}
    };

    // === CPU ===
    auto cpu_force_engine = std::make_shared<SimpleForceEngine<float>>();
    cpu_force_engine->addSpring(0, 1, spring_constants[0], rest_lengths[0]);
    cpu_force_engine->addSpring(1, 2, spring_constants[1], rest_lengths[1]);

    SimpleForceEngine<float>::ParameterGradients cpu_param_grads;
    cpu_param_grads.spring_constant_grads.resize(num_springs, 0.0f);
    cpu_param_grads.rest_length_grads.resize(num_springs, 0.0f);

    cpu_force_engine->computeForceParameterGradients(positions, adjoint_forces, cpu_param_grads);

    std::cout << "CPU Gradients:" << std::endl;
    std::cout << "  Spring 0: dL/dk = " << cpu_param_grads.spring_constant_grads[0] << std::endl;
    std::cout << "  Spring 1: dL/dk = " << cpu_param_grads.spring_constant_grads[1] << std::endl;

    // === GPU ===
    auto positions_f3 = toFloat3(positions);
    auto adjoint_forces_f3 = toFloat3(adjoint_forces);

    std::vector<SpringConnection> springs = {
        {0, 1, spring_constants[0], rest_lengths[0]},
        {1, 2, spring_constants[1], rest_lengths[1]}
    };

    float3* d_positions = copyToDevice(positions_f3);
    float3* d_adjoint_forces = copyToDevice(adjoint_forces_f3);
    SpringConnection* d_springs = copyToDevice(springs);

    float3* d_grad_positions;
    float* d_grad_spring_k;
    float* d_grad_rest_lengths;

    cudaMalloc(&d_grad_positions, num_particles * sizeof(float3));
    cudaMalloc(&d_grad_spring_k, num_springs * sizeof(float));
    cudaMalloc(&d_grad_rest_lengths, num_springs * sizeof(float));

    cudaMemset(d_grad_positions, 0, num_particles * sizeof(float3));
    cudaMemset(d_grad_spring_k, 0, num_springs * sizeof(float));
    cudaMemset(d_grad_rest_lengths, 0, num_springs * sizeof(float));

    launch_spring_force_backward_with_parameters(
        d_adjoint_forces, d_grad_positions, d_grad_spring_k, d_grad_rest_lengths,
        d_positions, d_springs, num_particles, num_springs, 0
    );

    cudaDeviceSynchronize();

    auto gpu_grad_k = copyFromDevice(d_grad_spring_k, num_springs);

    std::cout << "\nGPU Gradients:" << std::endl;
    std::cout << "  Spring 0: dL/dk = " << gpu_grad_k[0] << std::endl;
    std::cout << "  Spring 1: dL/dk = " << gpu_grad_k[1] << std::endl;

    // Compare
    for (int i = 0; i < num_springs; ++i) {
        float error = std::abs(cpu_param_grads.spring_constant_grads[i] - gpu_grad_k[i]) /
                      (std::abs(cpu_param_grads.spring_constant_grads[i]) + 1e-10f);
        EXPECT_LT(error, 0.001f) << "Spring " << i << " gradient mismatch";
    }

    // Cleanup
    cudaFree(d_positions);
    cudaFree(d_adjoint_forces);
    cudaFree(d_springs);
    cudaFree(d_grad_positions);
    cudaFree(d_grad_spring_k);
    cudaFree(d_grad_rest_lengths);

    std::cout << "\n✅ Multi-spring gradients MATCH between CPU and GPU!" << std::endl;
}


/**
 * Test 3: Verlet Integration Backward Pass
 */
TEST(CUDAAdjointValidation, VerletBackward) {
    std::cout << "\n=== Testing Verlet Backward Pass (GPU) ===" << std::endl;

    const int num_particles = 2;
    const float dt = 0.01f;

    std::vector<float3> grad_pos_out = {
        make_float3(1.0f, 0.5f, 0.2f),
        make_float3(-0.5f, 0.3f, -0.1f)
    };

    std::vector<float3> grad_vel_out = {
        make_float3(0.1f, -0.2f, 0.3f),
        make_float3(0.2f, 0.1f, -0.2f)
    };

    std::vector<float3> saved_velocities = {
        make_float3(0.5f, 0.2f, 0.1f),
        make_float3(-0.3f, 0.4f, 0.2f)
    };

    std::vector<float3> saved_forces = {
        make_float3(2.0f, -1.0f, 0.5f),
        make_float3(-2.0f, 1.0f, -0.5f)
    };

    std::vector<float> saved_masses = {1.0f, 1.5f};

    // Allocate GPU memory
    float3* d_grad_pos_out = copyToDevice(grad_pos_out);
    float3* d_grad_vel_out = copyToDevice(grad_vel_out);
    float3* d_saved_velocities = copyToDevice(saved_velocities);
    float3* d_saved_forces = copyToDevice(saved_forces);
    float* d_saved_masses = copyToDevice(saved_masses);

    float3* d_grad_pos_in;
    float3* d_grad_vel_in;
    float3* d_grad_forces;
    float* d_grad_masses;

    cudaMalloc(&d_grad_pos_in, num_particles * sizeof(float3));
    cudaMalloc(&d_grad_vel_in, num_particles * sizeof(float3));
    cudaMalloc(&d_grad_forces, num_particles * sizeof(float3));
    cudaMalloc(&d_grad_masses, num_particles * sizeof(float));

    // Launch kernel
    launch_verlet_integration_backward(
        d_grad_pos_out, d_grad_vel_out,
        d_grad_pos_in, d_grad_vel_in,
        d_grad_forces, d_grad_masses,
        d_saved_velocities, d_saved_forces, d_saved_masses,
        dt, num_particles, 0
    );

    cudaDeviceSynchronize();

    // Copy results
    auto gpu_grad_pos_in = copyFromDevice(d_grad_pos_in, num_particles);
    auto gpu_grad_vel_in = copyFromDevice(d_grad_vel_in, num_particles);
    auto gpu_grad_forces = copyFromDevice(d_grad_forces, num_particles);

    std::cout << "GPU Verlet Backward Results:" << std::endl;
    std::cout << "  grad_pos_in[0]: (" << gpu_grad_pos_in[0].x << ", "
              << gpu_grad_pos_in[0].y << ", " << gpu_grad_pos_in[0].z << ")" << std::endl;
    std::cout << "  grad_forces[0]: (" << gpu_grad_forces[0].x << ", "
              << gpu_grad_forces[0].y << ", " << gpu_grad_forces[0].z << ")" << std::endl;

    // Basic sanity checks
    EXPECT_FALSE(std::isnan(gpu_grad_pos_in[0].x));
    EXPECT_FALSE(std::isnan(gpu_grad_forces[0].x));

    // Cleanup
    cudaFree(d_grad_pos_out);
    cudaFree(d_grad_vel_out);
    cudaFree(d_saved_velocities);
    cudaFree(d_saved_forces);
    cudaFree(d_saved_masses);
    cudaFree(d_grad_pos_in);
    cudaFree(d_grad_vel_in);
    cudaFree(d_grad_forces);
    cudaFree(d_grad_masses);

    std::cout << "\n✅ Verlet backward pass executed successfully on GPU!" << std::endl;
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Check CUDA availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA devices found!" << std::endl;
        return 1;
    }

    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    return RUN_ALL_TESTS();
}
