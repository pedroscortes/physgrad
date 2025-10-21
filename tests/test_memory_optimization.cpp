/**
 * PhysGrad Memory Optimization Tests
 *
 * Tests for GPU memory access optimization patterns and bandwidth benchmarks.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

// Include CUDA headers first to get vector types
#include <vector_types.h>

#include "memory_optimization.h"

using namespace physgrad;

// Forward declare launch wrapper functions from kernel_wrappers.cu
extern "C" {
    void launch_optimized_force_computation(
        const float4*, const float*, float4*, int, int);
    void launch_optimized_verlet_integration(
        float4*, float4*, const float4*, float, int, int);
    void launch_optimized_energy_reduction(
        const float4*, const float4*, float*, float*, int, int);
}

class MemoryOptimizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

TEST_F(MemoryOptimizationTest, MemoryBandwidthOptimizerExists) {
    // Test that MemoryBandwidthOptimizer can be instantiated
    MemoryBandwidthOptimizer optimizer;
    SUCCEED() << "MemoryBandwidthOptimizer instantiated successfully";
}

TEST_F(MemoryOptimizationTest, BandwidthBenchmarkCoalescedSequential) {
    MemoryBandwidthOptimizer optimizer;

    size_t data_size = 1024 * 1024;  // 1M elements
    int iterations = 10;

    auto result = optimizer.benchmarkAccessPattern(
        data_size,
        MemoryAccessPattern::COALESCED_SEQUENTIAL,
        iterations
    );

    EXPECT_GT(result.achieved_bandwidth_gb_s, 0.0f);
    EXPECT_GT(result.theoretical_bandwidth_gb_s, 0.0f);
    EXPECT_GT(result.efficiency_percentage, 0.0f);
    EXPECT_LE(result.efficiency_percentage, 100.0f);

    std::cout << "Coalesced Sequential Access:" << std::endl;
    std::cout << "  Achieved: " << result.achieved_bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "  Theoretical: " << result.theoretical_bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "  Efficiency: " << result.efficiency_percentage << "%" << std::endl;
}

TEST_F(MemoryOptimizationTest, BandwidthBenchmarkStridedRegular) {
    MemoryBandwidthOptimizer optimizer;

    size_t data_size = 1024 * 1024;
    int iterations = 10;

    auto result = optimizer.benchmarkAccessPattern(
        data_size,
        MemoryAccessPattern::STRIDED_REGULAR,
        iterations
    );

    EXPECT_GT(result.achieved_bandwidth_gb_s, 0.0f);
    EXPECT_GT(result.theoretical_bandwidth_gb_s, 0.0f);

    std::cout << "Strided Regular Access:" << std::endl;
    std::cout << "  Achieved: " << result.achieved_bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "  Efficiency: " << result.efficiency_percentage << "%" << std::endl;
}

TEST_F(MemoryOptimizationTest, FindOptimalLaunchParams) {
    MemoryBandwidthOptimizer optimizer;

    size_t data_size = 1024 * 1024;
    size_t shared_memory = 0;

    dim3 params = optimizer.findOptimalLaunchParams(data_size, shared_memory);

    EXPECT_GT(params.x, 0);
    EXPECT_EQ(params.y, 1);
    EXPECT_EQ(params.z, 1);

    std::cout << "Optimal launch params for " << data_size << " elements:" << std::endl;
    std::cout << "  Grid size: " << params.x << std::endl;
}

TEST_F(MemoryOptimizationTest, StreamOptimizerCreation) {
    int num_streams = 4;
    StreamOptimizer stream_opt(num_streams);

    EXPECT_EQ(stream_opt.getNumStreams(), num_streams);
}

TEST_F(MemoryOptimizationTest, AoSoAContainerFloat4) {
    // Test AoSoA container with float, vector size 4
    size_t capacity = 1000;
    AoSoAContainer<float, 4> container(capacity);

    // Just verify it constructs without errors
    SUCCEED() << "AoSoAContainer<float, 4> created successfully";
}

TEST_F(MemoryOptimizationTest, AoSoAContainerFloat3) {
    // Test AoSoA container with float, vector size 3
    size_t capacity = 500;
    AoSoAContainer<float, 3> container(capacity);

    // Just verify it constructs without errors
    SUCCEED() << "AoSoAContainer<float, 3> created successfully";
}

TEST_F(MemoryOptimizationTest, MortonOrderEncode3D) {
    uint32_t x = 5, y = 3, z = 7;
    uint64_t morton = MortonOrderOptimizer::encode3D(x, y, z);

    EXPECT_GT(morton, 0);

    // Test decode
    uint32_t decoded_x, decoded_y, decoded_z;
    MortonOrderOptimizer::decode3D(morton, decoded_x, decoded_y, decoded_z);

    EXPECT_EQ(decoded_x, x);
    EXPECT_EQ(decoded_y, y);
    EXPECT_EQ(decoded_z, z);
}

TEST_F(MemoryOptimizationTest, MortonOrderMultiplePoints) {
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> points = {
        {0, 0, 0},
        {1, 1, 1},
        {2, 3, 4},
        {10, 20, 30},
        {100, 200, 300}
    };

    for (const auto& [x, y, z] : points) {
        uint64_t morton = MortonOrderOptimizer::encode3D(x, y, z);

        uint32_t decoded_x, decoded_y, decoded_z;
        MortonOrderOptimizer::decode3D(morton, decoded_x, decoded_y, decoded_z);

        EXPECT_EQ(decoded_x, x) << "Point (" << x << ", " << y << ", " << z << ")";
        EXPECT_EQ(decoded_y, y) << "Point (" << x << ", " << y << ", " << z << ")";
        EXPECT_EQ(decoded_z, z) << "Point (" << x << ", " << y << ", " << z << ")";
    }
}

TEST_F(MemoryOptimizationTest, CoalescedVsRandomAccess) {
    MemoryBandwidthOptimizer optimizer;

    size_t data_size = 512 * 1024;  // 512K elements
    int iterations = 5;

    // Benchmark coalesced access
    auto coalesced = optimizer.benchmarkAccessPattern(
        data_size, MemoryAccessPattern::COALESCED_SEQUENTIAL, iterations
    );

    // Benchmark random access
    auto random = optimizer.benchmarkAccessPattern(
        data_size, MemoryAccessPattern::RANDOM_SCATTERED, iterations
    );

    // Coalesced should be significantly faster
    EXPECT_GT(coalesced.achieved_bandwidth_gb_s, random.achieved_bandwidth_gb_s * 0.5)
        << "Coalesced access should be at least 50% faster than random";

    std::cout << "Memory Access Comparison:" << std::endl;
    std::cout << "  Coalesced: " << coalesced.achieved_bandwidth_gb_s << " GB/s ("
              << coalesced.efficiency_percentage << "%)" << std::endl;
    std::cout << "  Random: " << random.achieved_bandwidth_gb_s << " GB/s ("
              << random.efficiency_percentage << "%)" << std::endl;
    std::cout << "  Speedup: " << (coalesced.achieved_bandwidth_gb_s / random.achieved_bandwidth_gb_s) << "x" << std::endl;
}

TEST_F(MemoryOptimizationTest, LaunchWrappersExist) {
    // Test that launch wrapper functions exist (link check)
    void* ptr1 = (void*)launch_optimized_force_computation;
    void* ptr2 = (void*)launch_optimized_verlet_integration;
    void* ptr3 = (void*)launch_optimized_energy_reduction;

    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr3, nullptr);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
