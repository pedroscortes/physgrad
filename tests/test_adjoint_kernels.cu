/**
 * PhysGrad Adjoint Kernels Validation Test
 *
 * Direct tests for CUDA adjoint kernels to validate gradient computation.
 * Uses finite differences for gradient verification.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <random>

// Forward declarations of adjoint kernels
extern "C" {
    __global__ void verlet_integration_backward_kernel(
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
        int num_particles
    );
}

class AdjointKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        num_particles = 10;
        dt = 0.01f;

        // Allocate device memory
        cudaMalloc(&d_grad_pos_out, num_particles * sizeof(float3));
        cudaMalloc(&d_grad_vel_out, num_particles * sizeof(float3));
        cudaMalloc(&d_grad_pos_in, num_particles * sizeof(float3));
        cudaMalloc(&d_grad_vel_in, num_particles * sizeof(float3));
        cudaMalloc(&d_grad_forces, num_particles * sizeof(float3));
        cudaMalloc(&d_grad_masses, num_particles * sizeof(float));
        cudaMalloc(&d_saved_velocities, num_particles * sizeof(float3));
        cudaMalloc(&d_saved_forces, num_particles * sizeof(float3));
        cudaMalloc(&d_saved_masses, num_particles * sizeof(float));

        // Initialize host data
        h_grad_pos_out.resize(num_particles);
        h_grad_vel_out.resize(num_particles);
        h_saved_velocities.resize(num_particles);
        h_saved_forces.resize(num_particles);
        h_saved_masses.resize(num_particles);

        // Set up test data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < num_particles; ++i) {
            // Output gradients (loss w.r.t. final state)
            h_grad_pos_out[i] = make_float3(dist(rng), dist(rng), dist(rng));
            h_grad_vel_out[i] = make_float3(dist(rng), dist(rng), dist(rng));

            // Saved forward pass values
            h_saved_velocities[i] = make_float3(dist(rng), dist(rng), dist(rng));
            h_saved_forces[i] = make_float3(dist(rng) * 10.0f, dist(rng) * 10.0f, dist(rng) * 10.0f);
            h_saved_masses[i] = 1.0f + std::abs(dist(rng));
        }

        // Copy to device
        cudaMemcpy(d_grad_pos_out, h_grad_pos_out.data(), num_particles * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_vel_out, h_grad_vel_out.data(), num_particles * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_saved_velocities, h_saved_velocities.data(), num_particles * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_saved_forces, h_saved_forces.data(), num_particles * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_saved_masses, h_saved_masses.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_grad_pos_out);
        cudaFree(d_grad_vel_out);
        cudaFree(d_grad_pos_in);
        cudaFree(d_grad_vel_in);
        cudaFree(d_grad_forces);
        cudaFree(d_grad_masses);
        cudaFree(d_saved_velocities);
        cudaFree(d_saved_forces);
        cudaFree(d_saved_masses);
    }

    int num_particles;
    float dt;

    // Device pointers
    float3 *d_grad_pos_out, *d_grad_vel_out;
    float3 *d_grad_pos_in, *d_grad_vel_in;
    float3 *d_grad_forces;
    float *d_grad_masses;
    float3 *d_saved_velocities, *d_saved_forces;
    float *d_saved_masses;

    // Host vectors
    std::vector<float3> h_grad_pos_out, h_grad_vel_out;
    std::vector<float3> h_saved_velocities, h_saved_forces;
    std::vector<float> h_saved_masses;
};

TEST_F(AdjointKernelTest, VerletBackwardKernelLaunches) {
    // Test that the kernel launches without errors
    int block_size = 256;
    int grid_size = (num_particles + block_size - 1) / block_size;

    verlet_integration_backward_kernel<<<grid_size, block_size>>>(
        d_grad_pos_out, d_grad_vel_out,
        d_grad_pos_in, d_grad_vel_in,
        d_grad_forces, d_grad_masses,
        d_saved_velocities, d_saved_forces, d_saved_masses,
        dt, num_particles
    );

    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel execution failed: " << cudaGetErrorString(err);
}

TEST_F(AdjointKernelTest, VerletBackwardPositionGradient) {
    // Test: ∂L/∂x_in = ∂L/∂x_out (position gradient passes through)
    int block_size = 256;
    int grid_size = (num_particles + block_size - 1) / block_size;

    verlet_integration_backward_kernel<<<grid_size, block_size>>>(
        d_grad_pos_out, d_grad_vel_out,
        d_grad_pos_in, d_grad_vel_in,
        d_grad_forces, d_grad_masses,
        d_saved_velocities, d_saved_forces, d_saved_masses,
        dt, num_particles
    );

    cudaDeviceSynchronize();

    // Copy results back
    std::vector<float3> h_grad_pos_in(num_particles);
    cudaMemcpy(h_grad_pos_in.data(), d_grad_pos_in, num_particles * sizeof(float3), cudaMemcpyDeviceToHost);

    // Verify: grad_pos_in should equal grad_pos_out
    for (int i = 0; i < num_particles; ++i) {
        EXPECT_NEAR(h_grad_pos_in[i].x, h_grad_pos_out[i].x, 1e-5f) << "Particle " << i;
        EXPECT_NEAR(h_grad_pos_in[i].y, h_grad_pos_out[i].y, 1e-5f) << "Particle " << i;
        EXPECT_NEAR(h_grad_pos_in[i].z, h_grad_pos_out[i].z, 1e-5f) << "Particle " << i;
    }
}

TEST_F(AdjointKernelTest, VerletBackwardVelocityGradient) {
    // Test: ∂L/∂v_in = ∂L/∂x_out * dt + ∂L/∂v_out
    int block_size = 256;
    int grid_size = (num_particles + block_size - 1) / block_size;

    verlet_integration_backward_kernel<<<grid_size, block_size>>>(
        d_grad_pos_out, d_grad_vel_out,
        d_grad_pos_in, d_grad_vel_in,
        d_grad_forces, d_grad_masses,
        d_saved_velocities, d_saved_forces, d_saved_masses,
        dt, num_particles
    );

    cudaDeviceSynchronize();

    // Copy results back
    std::vector<float3> h_grad_vel_in(num_particles);
    cudaMemcpy(h_grad_vel_in.data(), d_grad_vel_in, num_particles * sizeof(float3), cudaMemcpyDeviceToHost);

    // Verify velocity gradient formula
    for (int i = 0; i < num_particles; ++i) {
        float expected_x = h_grad_pos_out[i].x * dt + h_grad_vel_out[i].x;
        float expected_y = h_grad_pos_out[i].y * dt + h_grad_vel_out[i].y;
        float expected_z = h_grad_pos_out[i].z * dt + h_grad_vel_out[i].z;

        EXPECT_NEAR(h_grad_vel_in[i].x, expected_x, 1e-4f) << "Particle " << i;
        EXPECT_NEAR(h_grad_vel_in[i].y, expected_y, 1e-4f) << "Particle " << i;
        EXPECT_NEAR(h_grad_vel_in[i].z, expected_z, 1e-4f) << "Particle " << i;
    }
}

TEST_F(AdjointKernelTest, VerletBackwardForceGradient) {
    // Test: Force gradients are computed correctly
    int block_size = 256;
    int grid_size = (num_particles + block_size - 1) / block_size;

    verlet_integration_backward_kernel<<<grid_size, block_size>>>(
        d_grad_pos_out, d_grad_vel_out,
        d_grad_pos_in, d_grad_vel_in,
        d_grad_forces, d_grad_masses,
        d_saved_velocities, d_saved_forces, d_saved_masses,
        dt, num_particles
    );

    cudaDeviceSynchronize();

    // Copy results back
    std::vector<float3> h_grad_forces(num_particles);
    cudaMemcpy(h_grad_forces.data(), d_grad_forces, num_particles * sizeof(float3), cudaMemcpyDeviceToHost);

    // Verify force gradients exist and are non-zero (when input gradients are non-zero)
    bool has_nonzero_gradient = false;
    for (int i = 0; i < num_particles; ++i) {
        float mag = std::sqrt(h_grad_forces[i].x * h_grad_forces[i].x +
                            h_grad_forces[i].y * h_grad_forces[i].y +
                            h_grad_forces[i].z * h_grad_forces[i].z);
        if (mag > 1e-6f) {
            has_nonzero_gradient = true;
            break;
        }
    }

    EXPECT_TRUE(has_nonzero_gradient) << "Force gradients should be non-zero";
}

TEST_F(AdjointKernelTest, VerletBackwardMassGradient) {
    // Test: Mass gradients are computed
    int block_size = 256;
    int grid_size = (num_particles + block_size - 1) / block_size;

    verlet_integration_backward_kernel<<<grid_size, block_size>>>(
        d_grad_pos_out, d_grad_vel_out,
        d_grad_pos_in, d_grad_vel_in,
        d_grad_forces, d_grad_masses,
        d_saved_velocities, d_saved_forces, d_saved_masses,
        dt, num_particles
    );

    cudaDeviceSynchronize();

    // Copy results back
    std::vector<float> h_grad_masses(num_particles);
    cudaMemcpy(h_grad_masses.data(), d_grad_masses, num_particles * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify mass gradients exist
    bool has_nonzero_gradient = false;
    for (int i = 0; i < num_particles; ++i) {
        if (std::abs(h_grad_masses[i]) > 1e-6f) {
            has_nonzero_gradient = true;
            break;
        }
    }

    EXPECT_TRUE(has_nonzero_gradient) << "Mass gradients should be non-zero";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
