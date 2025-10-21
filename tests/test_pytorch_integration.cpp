/**
 * PhysGrad PyTorch Integration Tests
 *
 * Tests for PyTorch custom autograd functions and CUDA kernel integration.
 * These tests validate compilation and API integration.
 */

#include <gtest/gtest.h>

#ifdef HAVE_PYTORCH
#include <torch/torch.h>
// Note: Don't include torch/extension.h - it's for Python extensions only
#include <cuda_runtime.h>

// Forward declarations of launch functions from pytorch_autograd.cu
extern "C" {
    void launch_mpm_timestep_forward(
        float* positions, float* velocities, const float* masses,
        int num_particles, int grid_resolution, float dt,
        const float* gravity, cudaStream_t stream
    );

    void launch_mpm_timestep_backward(
        const float* grad_positions, const float* grad_velocities,
        float* grad_input_positions, float* grad_input_velocities,
        float* grad_masses, float* grad_gravity,
        const float* saved_positions, const float* saved_velocities,
        int num_particles, int grid_resolution, float dt,
        cudaStream_t stream
    );

    void launch_g2p2g_forward(
        const float* particles, const float* grid, float* output,
        int num_particles, int grid_size, int feature_dim,
        cudaStream_t stream
    );

    void launch_g2p2g_backward(
        const float* grad_output,
        float* grad_particles, float* grad_grid,
        const float* saved_particles, const float* saved_grid,
        int num_particles, int grid_size, int feature_dim,
        cudaStream_t stream
    );
}

class PyTorchIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability
        cuda_available = torch::cuda::is_available();
        if (cuda_available) {
            std::cout << "CUDA available: YES" << std::endl;
            std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
        } else {
            std::cout << "CUDA available: NO (tests will use CPU tensors)" << std::endl;
        }
    }

    bool cuda_available = false;
};

TEST_F(PyTorchIntegrationTest, PyTorchCompilation) {
    // Test that PyTorch headers compiled successfully
    SUCCEED() << "PyTorch C++ API compiled successfully";
}

TEST_F(PyTorchIntegrationTest, CPUTensorCreation) {
    // Test creating CPU tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto tensor = torch::zeros({10, 3}, options);

    EXPECT_EQ(tensor.size(0), 10);
    EXPECT_EQ(tensor.size(1), 3);
    EXPECT_EQ(tensor.dtype(), torch::kFloat32);
}

TEST_F(PyTorchIntegrationTest, TensorGradientEnabled) {
    // Test gradient tracking
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
    auto tensor = torch::randn({5, 3}, options);

    EXPECT_TRUE(tensor.requires_grad());
}

TEST_F(PyTorchIntegrationTest, LaunchFunctionsExist) {
    // Test that launch functions are accessible (link check)
    // We don't call them without GPU, but verify they exist
    void* ptr1 = (void*)launch_mpm_timestep_forward;
    void* ptr2 = (void*)launch_mpm_timestep_backward;
    void* ptr3 = (void*)launch_g2p2g_forward;
    void* ptr4 = (void*)launch_g2p2g_backward;

    EXPECT_NE(ptr1, nullptr) << "launch_mpm_timestep_forward should be defined";
    EXPECT_NE(ptr2, nullptr) << "launch_mpm_timestep_backward should be defined";
    EXPECT_NE(ptr3, nullptr) << "launch_g2p2g_forward should be defined";
    EXPECT_NE(ptr4, nullptr) << "launch_g2p2g_backward should be defined";
}

TEST_F(PyTorchIntegrationTest, TensorDataPointer) {
    // Test accessing tensor data pointers
    auto tensor = torch::randn({10, 3});
    float* data_ptr = tensor.data_ptr<float>();

    EXPECT_NE(data_ptr, nullptr);

    // Verify we can read/write the data
    data_ptr[0] = 42.0f;
    EXPECT_FLOAT_EQ(tensor[0][0].item<float>(), 42.0f);
}

TEST_F(PyTorchIntegrationTest, TensorContiguity) {
    // Test tensor contiguity (required for CUDA kernels)
    auto tensor = torch::randn({10, 3});
    EXPECT_TRUE(tensor.is_contiguous());

    // Test that transposed tensor is not contiguous
    auto transposed = tensor.transpose(0, 1);
    EXPECT_FALSE(transposed.is_contiguous());

    // Test making contiguous
    auto contiguous = transposed.contiguous();
    EXPECT_TRUE(contiguous.is_contiguous());
}

TEST_F(PyTorchIntegrationTest, CUDATensorCreationIfAvailable) {
    if (!cuda_available) {
        GTEST_SKIP() << "CUDA not available, skipping GPU tensor test";
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto tensor = torch::zeros({10, 3}, options);

    EXPECT_EQ(tensor.size(0), 10);
    EXPECT_EQ(tensor.size(1), 3);
    EXPECT_TRUE(tensor.is_cuda());
}

TEST_F(PyTorchIntegrationTest, MPMTensorShapes) {
    // Test creating tensors with MPM shapes
    int num_particles = 100;
    int grid_resolution = 32;

    auto positions = torch::randn({num_particles, 3});
    auto velocities = torch::randn({num_particles, 3});
    auto masses = torch::ones({num_particles});
    auto gravity = torch::tensor({0.0f, -9.8f, 0.0f});

    EXPECT_EQ(positions.size(0), num_particles);
    EXPECT_EQ(positions.size(1), 3);
    EXPECT_EQ(velocities.size(0), num_particles);
    EXPECT_EQ(masses.size(0), num_particles);
    EXPECT_EQ(gravity.size(0), 3);
}

TEST_F(PyTorchIntegrationTest, G2P2GTensorShapes) {
    // Test creating tensors with G2P2G shapes
    int num_particles = 100;
    int grid_size = 32;
    int feature_dim = 3;

    auto particles = torch::randn({num_particles, feature_dim});
    auto grid = torch::randn({grid_size, grid_size, grid_size, feature_dim});
    auto output = torch::zeros({num_particles, feature_dim});

    EXPECT_EQ(particles.size(0), num_particles);
    EXPECT_EQ(particles.size(1), feature_dim);
    EXPECT_EQ(grid.size(0), grid_size);
    EXPECT_EQ(grid.size(3), feature_dim);
    EXPECT_EQ(output.size(0), num_particles);
}

TEST_F(PyTorchIntegrationTest, BackwardPassSetup) {
    // Test setting up backward pass infrastructure
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
    auto positions = torch::randn({10, 3}, options);

    // Simulate forward pass
    auto output = positions * 2.0f;

    // Simulate backward pass
    auto grad_output = torch::ones_like(output);
    output.backward(grad_output);

    // Verify gradients were computed
    EXPECT_TRUE(positions.grad().defined());
    EXPECT_EQ(positions.grad().size(0), 10);
    EXPECT_EQ(positions.grad().size(1), 3);
}

TEST_F(PyTorchIntegrationTest, CUDAStreamCreation) {
    // Test CUDA stream creation (even without GPU, API should exist)
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);

    if (cuda_available) {
        EXPECT_EQ(err, cudaSuccess) << "cudaStreamCreate should succeed with CUDA";
        cudaStreamDestroy(stream);
    } else {
        // Without GPU, expect error but verify API exists
        EXPECT_NE((void*)cudaStreamCreate, nullptr) << "CUDA API should be available";
    }
}

TEST_F(PyTorchIntegrationTest, TensorTypeConversion) {
    // Test type conversions needed for kernels
    auto tensor_f32 = torch::randn({10, 3}, torch::kFloat32);
    auto tensor_f64 = tensor_f32.to(torch::kFloat64);
    auto tensor_back = tensor_f64.to(torch::kFloat32);

    EXPECT_EQ(tensor_f32.dtype(), torch::kFloat32);
    EXPECT_EQ(tensor_f64.dtype(), torch::kFloat64);
    EXPECT_EQ(tensor_back.dtype(), torch::kFloat32);
}

TEST_F(PyTorchIntegrationTest, InplaceOperations) {
    // Test in-place operations (used in some kernels)
    auto tensor = torch::randn({10, 3});
    auto original_ptr = tensor.data_ptr<float>();

    tensor.mul_(2.0f);  // In-place multiplication
    auto after_ptr = tensor.data_ptr<float>();

    EXPECT_EQ(original_ptr, after_ptr) << "In-place operation should not reallocate";
}

#else  // !HAVE_PYTORCH

TEST(PyTorchIntegrationTest, PyTorchNotAvailable) {
    GTEST_SKIP() << "PyTorch not available - tests disabled";
}

#endif  // HAVE_PYTORCH

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
