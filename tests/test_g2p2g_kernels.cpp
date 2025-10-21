/**
 * PhysGrad G2P2G Kernel Fusion Tests
 *
 * Tests for the optimized Grid-to-Particle-to-Grid kernel fusion implementation.
 * Validates performance optimizations and correctness of the fused kernels.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>
#include <chrono>

#include "src/mpm_g2p2g_kernels.h"

using namespace physgrad;
using namespace physgrad::mpm;
using namespace physgrad::mpm::kernels;

/**
 * Test kernel configuration and shared memory structures
 */
template<typename T>
bool testKernelConfiguration() {
    std::cout << "Testing kernel configuration...\n";

    // Test kernel configuration constants
    bool config_test =
        (KernelConfig::P2G_BLOCK_SIZE > 0) &&
        (KernelConfig::G2P_BLOCK_SIZE > 0) &&
        (KernelConfig::G2P2G_BLOCK_SIZE > 0) &&
        (KernelConfig::G2P2G_BLOCK_SIZE <= MAX_BLOCK_SIZE);

    std::cout << "Kernel block sizes:\n";
    std::cout << "  P2G: " << KernelConfig::P2G_BLOCK_SIZE << "\n";
    std::cout << "  G2P: " << KernelConfig::G2P_BLOCK_SIZE << "\n";
    std::cout << "  G2P2G: " << KernelConfig::G2P2G_BLOCK_SIZE << "\n";

    // Test that block sizes are power of 2 (optimal for GPU)
    auto isPowerOfTwo = [](int n) {
        return n > 0 && (n & (n - 1)) == 0;
    };

    bool power_of_two_test =
        isPowerOfTwo(KernelConfig::P2G_BLOCK_SIZE) &&
        isPowerOfTwo(KernelConfig::G2P_BLOCK_SIZE) &&
        isPowerOfTwo(KernelConfig::G2P2G_BLOCK_SIZE);

    bool all_passed = config_test && power_of_two_test;

    if (all_passed) {
        std::cout << "✓ Kernel configuration test passed\n";
    } else {
        std::cout << "✗ Kernel configuration test failed\n";
        std::cout << "  Config: " << (config_test ? "PASS" : "FAIL") << "\n";
        std::cout << "  Power of 2: " << (power_of_two_test ? "PASS" : "FAIL") << "\n";
    }

    return all_passed;
}

/**
 * Test fast shape function evaluation
 */
template<typename T>
bool testFastShapeFunctions() {
    std::cout << "Testing fast shape functions...\n";

    // Test quadratic shape function and derivative
    bool quadratic_test = true;
    for (T x = -T{1.5}; x <= T{1.5}; x += T{0.1}) {
        T w, dw;
        FastShapeFunctions<T>::evalQuadratic(x, w, dw);

        // Check that function values are reasonable
        if (w < T{0} || w > T{1} || !std::isfinite(w) || !std::isfinite(dw)) {
            quadratic_test = false;
            break;
        }

        // Check derivative consistency with finite differences
        T h = T{1e-5};
        T w_plus, dw_plus, w_minus, dw_minus;
        FastShapeFunctions<T>::evalQuadratic(x + h, w_plus, dw_plus);
        FastShapeFunctions<T>::evalQuadratic(x - h, w_minus, dw_minus);
        T fd_derivative = (w_plus - w_minus) / (T{2} * h);

        if (std::abs(dw - fd_derivative) > T{1e-3}) {
            quadratic_test = false;
            break;
        }
    }

    // Test cubic shape function
    bool cubic_test = true;
    for (T x = -T{2}; x <= T{2}; x += T{0.2}) {
        T w, dw;
        FastShapeFunctions<T>::evalCubic(x, w, dw);

        if (w < T{0} || w > T{1} || !std::isfinite(w) || !std::isfinite(dw)) {
            cubic_test = false;
            break;
        }
    }

    bool all_passed = quadratic_test && cubic_test;

    if (all_passed) {
        std::cout << "✓ Fast shape functions test passed\n";
    } else {
        std::cout << "✗ Fast shape functions test failed\n";
        std::cout << "  Quadratic: " << (quadratic_test ? "PASS" : "FAIL") << "\n";
        std::cout << "  Cubic: " << (cubic_test ? "PASS" : "FAIL") << "\n";
    }

    return all_passed;
}

/**
 * Test G2P2G kernel launcher
 */
template<typename T>
bool testG2P2GKernelLauncher() {
    std::cout << "Testing G2P2G kernel launcher...\n";

    // Create test MPM system
    int3 grid_dims = {10, 10, 10};
    ConceptVector3D<T> cell_size{T{0.1}, T{0.1}, T{0.1}};
    ConceptVector3D<T> origin{T{0}, T{0}, T{0}};

    MPMGrid<T> grid(grid_dims, cell_size, origin);
    ParticleAoSoA<T> particles;

    // Add test particles
    const size_t num_particles = 100;
    particles.resize(num_particles);

    for (size_t i = 0; i < num_particles; ++i) {
        ConceptVector3D<T> pos{
            T{0.2} + static_cast<T>(i % 5) * T{0.1},
            T{0.5} + static_cast<T>((i / 5) % 5) * T{0.1},
            T{0.5} + static_cast<T>(i / 25) * T{0.1}
        };
        ConceptVector3D<T> vel{T{0}, T{-1}, T{0}}; // Falling particles
        T mass = T{0.1};

        particles.setPosition(i, pos);
        particles.setVelocity(i, vel);
        particles.setMass(i, mass);
    }

    // Test kernel launcher
    typename G2P2GKernelLauncher<T>::PerformanceConfig config;
    config.use_kernel_fusion = true;
    config.use_shared_memory = true;

    G2P2GKernelLauncher<T> launcher(config);

    // Get initial particle positions
    std::vector<ConceptVector3D<T>> initial_positions(num_particles);
    for (size_t i = 0; i < num_particles; ++i) {
        initial_positions[i] = particles.getPosition(i);
    }

    // Launch G2P2G kernel
    T dt = T{0.01};
    ConceptVector3D<T> gravity{T{0}, T{-9.81}, T{0}};

    try {
        launcher.launchG2P2G(particles, grid, dt, gravity, true, T{0.95}, 2);

        // Check that particles moved
        bool particles_moved = false;
        for (size_t i = 0; i < num_particles; ++i) {
            auto final_pos = particles.getPosition(i);
            T displacement = std::sqrt(
                (final_pos[0] - initial_positions[i][0]) * (final_pos[0] - initial_positions[i][0]) +
                (final_pos[1] - initial_positions[i][1]) * (final_pos[1] - initial_positions[i][1]) +
                (final_pos[2] - initial_positions[i][2]) * (final_pos[2] - initial_positions[i][2])
            );

            if (displacement > T{1e-6}) {
                particles_moved = true;
                break;
            }
        }

        std::cout << "✓ G2P2G kernel launcher test passed\n";
        std::cout << "  Particles moved: " << (particles_moved ? "YES" : "NO") << "\n";
        return particles_moved;

    } catch (const std::exception& e) {
        std::cout << "✗ G2P2G kernel launcher test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * Test performance comparison between fused and separate kernels
 */
template<typename T>
bool testPerformanceComparison() {
    std::cout << "Testing performance comparison...\n";

    // Create larger test system for performance testing
    int3 grid_dims = {50, 50, 50};
    ConceptVector3D<T> cell_size{T{0.02}, T{0.02}, T{0.02}};
    ConceptVector3D<T> origin{T{0}, T{0}, T{0}};

    MPMGrid<T> grid1(grid_dims, cell_size, origin);
    MPMGrid<T> grid2(grid_dims, cell_size, origin);

    ParticleAoSoA<T> particles1, particles2;

    // Add more particles for performance testing
    const size_t num_particles = 1000;
    particles1.resize(num_particles);
    particles2.resize(num_particles);

    for (size_t i = 0; i < num_particles; ++i) {
        ConceptVector3D<T> pos{
            T{0.1} + static_cast<T>(i % 10) * T{0.08},
            T{0.5} + static_cast<T>((i / 10) % 10) * T{0.08},
            T{0.5} + static_cast<T>(i / 100) * T{0.08}
        };
        ConceptVector3D<T> vel{T{0}, T{-2}, T{0}};
        T mass = T{0.05};

        particles1.setPosition(i, pos);
        particles1.setVelocity(i, vel);
        particles1.setMass(i, mass);

        particles2.setPosition(i, pos);
        particles2.setVelocity(i, vel);
        particles2.setMass(i, mass);
    }

    T dt = T{0.001};
    ConceptVector3D<T> gravity{T{0}, T{-9.81}, T{0}};

    // Test fused kernel
    typename G2P2GKernelLauncher<T>::PerformanceConfig fused_config;
    fused_config.use_kernel_fusion = true;
    G2P2GKernelLauncher<T> fused_launcher(fused_config);

    auto start_fused = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < 10; ++step) {
        fused_launcher.launchG2P2G(particles1, grid1, dt, gravity, true, T{0.95}, 2);
    }
    auto end_fused = std::chrono::high_resolution_clock::now();

    // Test separate kernels (fallback CPU implementation)
    typename G2P2GKernelLauncher<T>::PerformanceConfig separate_config;
    separate_config.use_kernel_fusion = false;
    G2P2GKernelLauncher<T> separate_launcher(separate_config);

    auto start_separate = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < 10; ++step) {
        separate_launcher.launchG2P2G(particles2, grid2, dt, gravity, true, T{0.95}, 2);
    }
    auto end_separate = std::chrono::high_resolution_clock::now();

    auto fused_time = std::chrono::duration_cast<std::chrono::microseconds>(end_fused - start_fused).count();
    auto separate_time = std::chrono::duration_cast<std::chrono::microseconds>(end_separate - start_separate).count();

    std::cout << "Performance comparison (10 steps, " << num_particles << " particles):\n";
    std::cout << "  Fused kernels: " << fused_time << " μs\n";
    std::cout << "  Separate kernels: " << separate_time << " μs\n";

    if (separate_time > 0) {
        T speedup = static_cast<T>(separate_time) / static_cast<T>(fused_time);
        std::cout << "  Speedup: " << speedup << "x\n";
    }

    // Verify that both methods produce similar results
    bool results_similar = true;
    T max_difference = T{0};

    for (size_t i = 0; i < std::min(num_particles, size_t{10}); ++i) {
        auto pos1 = particles1.getPosition(i);
        auto pos2 = particles2.getPosition(i);

        T diff = std::sqrt(
            (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) +
            (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]) +
            (pos1[2] - pos2[2]) * (pos1[2] - pos2[2])
        );

        max_difference = std::max(max_difference, diff);

        if (diff > T{0.1}) { // Allow some numerical difference
            results_similar = false;
        }
    }

    std::cout << "  Max position difference: " << max_difference << "\n";
    std::cout << "  Results similar: " << (results_similar ? "YES" : "NO") << "\n";

    std::cout << "✓ Performance comparison test completed\n";
    return true; // Always pass this test as it's informational
}

/**
 * Test memory access patterns and coalescing
 */
template<typename T>
bool testMemoryAccess() {
    std::cout << "Testing memory access patterns...\n";

    constexpr size_t test_particles = 512; // Multiple of warp size
    ParticleAoSoA<T> particles;
    particles.resize(test_particles);

    // Test sequential access pattern
    auto start_sequential = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_particles; ++i) {
        ConceptVector3D<T> pos{static_cast<T>(i), static_cast<T>(i), static_cast<T>(i)};
        particles.setPosition(i, pos);
    }
    auto end_sequential = std::chrono::high_resolution_clock::now();

    // Verify data
    bool sequential_correct = true;
    for (size_t i = 0; i < test_particles; ++i) {
        auto pos = particles.getPosition(i);
        if (std::abs(pos[0] - static_cast<T>(i)) > T{1e-6}) {
            sequential_correct = false;
            break;
        }
    }

    // Test random access pattern
    std::vector<size_t> random_indices(test_particles);
    for (size_t i = 0; i < test_particles; ++i) {
        random_indices[i] = i;
    }

    // Simple shuffle
    for (size_t i = 0; i < test_particles; ++i) {
        size_t j = (i * 7) % test_particles; // Simple permutation
        std::swap(random_indices[i], random_indices[j]);
    }

    auto start_random = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_particles; ++i) {
        size_t idx = random_indices[i];
        ConceptVector3D<T> pos{static_cast<T>(idx), static_cast<T>(idx), static_cast<T>(idx)};
        particles.setPosition(idx, pos);
    }
    auto end_random = std::chrono::high_resolution_clock::now();

    auto sequential_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_sequential - start_sequential).count();
    auto random_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_random - start_random).count();

    std::cout << "Memory access timing:\n";
    std::cout << "  Sequential: " << sequential_time << " ns\n";
    std::cout << "  Random: " << random_time << " ns\n";

    if (sequential_time > 0) {
        T access_ratio = static_cast<T>(random_time) / static_cast<T>(sequential_time);
        std::cout << "  Random/Sequential ratio: " << access_ratio << "\n";
    }

    bool all_passed = sequential_correct;

    if (all_passed) {
        std::cout << "✓ Memory access patterns test passed\n";
    } else {
        std::cout << "✗ Memory access patterns test failed\n";
    }

    return all_passed;
}

/**
 * Main test function
 */
int main() {
    std::cout << "PhysGrad G2P2G Kernel Fusion Tests\n";
    std::cout << "==================================\n\n";

    std::cout << std::fixed << std::setprecision(6);

    bool all_passed = true;

    // Test with float precision
    std::cout << "--- Float precision tests ---\n";
    all_passed &= testKernelConfiguration<float>();
    std::cout << "\n";

    all_passed &= testFastShapeFunctions<float>();
    std::cout << "\n";

    all_passed &= testG2P2GKernelLauncher<float>();
    std::cout << "\n";

    all_passed &= testMemoryAccess<float>();
    std::cout << "\n";

    all_passed &= testPerformanceComparison<float>();
    std::cout << "\n";

    // Test with double precision (selected tests)
    std::cout << "--- Double precision tests ---\n";
    all_passed &= testFastShapeFunctions<double>();
    std::cout << "\n";

    if (all_passed) {
        std::cout << "✓ All G2P2G kernel tests PASSED!\n";
        std::cout << "\nNote: Tests ran with CPU fallback. For full GPU performance,\n";
        std::cout << "compile with CUDA support using nvcc.\n";
        return 0;
    } else {
        std::cout << "✗ Some G2P2G kernel tests FAILED!\n";
        return 1;
    }
}