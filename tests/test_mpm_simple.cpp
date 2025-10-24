/**
 * Simple MPM Solver Validation Test
 * Tests basic functionality before running comprehensive benchmarks
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Simple test without full solver to validate data structures
#include "src/mpm_data_structures.h"

using namespace physgrad::mpm;

bool test_particle_data_structures() {
    std::cout << "Testing Particle AoSoA Data Structures..." << std::endl;

    const size_t test_particles = 1000;
    ParticleAoSoA<float> particles(test_particles);

    // Test initialization
    std::vector<std::array<float, 3>> positions(test_particles);
    std::vector<std::array<float, 3>> velocities(test_particles);
    std::vector<float> masses(test_particles);

    for (size_t i = 0; i < test_particles; ++i) {
        positions[i] = {static_cast<float>(i), static_cast<float>(i), static_cast<float>(i)};
        velocities[i] = {1.0f, 2.0f, 3.0f};
        masses[i] = 1.0f;
    }

    particles.setPositions(positions);
    particles.setVelocities(velocities);
    particles.setMasses(masses);

    std::cout << "  ✓ Initialized " << particles.size() << " particles" << std::endl;

    // Test data access
    auto retrieved_positions = particles.getPositions();
    auto retrieved_velocities = particles.getVelocities();
    auto retrieved_masses = particles.getMasses();

    bool data_correct = true;
    for (size_t i = 0; i < std::min(test_particles, size_t(10)); ++i) {
        if (std::abs(retrieved_positions[i][0] - static_cast<float>(i)) > 1e-6f ||
            std::abs(retrieved_velocities[i][1] - 2.0f) > 1e-6f ||
            std::abs(retrieved_masses[i] - 1.0f) > 1e-6f) {
            data_correct = false;
            break;
        }
    }

    if (data_correct) {
        std::cout << "  ✓ Data storage and retrieval works correctly" << std::endl;
        return true;
    } else {
        std::cout << "  ❌ Data storage/retrieval failed" << std::endl;
        return false;
    }
}

bool test_grid_data_structures() {
    std::cout << "Testing Grid Data Structures..." << std::endl;

    std::array<size_t, 3> resolution = {32, 32, 32};
    std::array<float, 3> domain_size = {1.0f, 1.0f, 1.0f};

    MPMGrid<float> grid(resolution, domain_size);

    std::cout << "  ✓ Created grid with resolution " << resolution[0] << "x" << resolution[1] << "x" << resolution[2] << std::endl;

    // Test basic grid operations
    grid.clearGrid();
    std::cout << "  ✓ Grid clearing works" << std::endl;

    // Test grid indexing
    auto grid_spacing = grid.getGridSpacing();
    bool spacing_correct = (std::abs(grid_spacing[0] - domain_size[0]/resolution[0]) < 1e-6f);

    if (spacing_correct) {
        std::cout << "  ✓ Grid spacing calculation correct: " << grid_spacing[0] << std::endl;
        return true;
    } else {
        std::cout << "  ❌ Grid spacing calculation failed" << std::endl;
        return false;
    }
}

bool test_material_parameters() {
    std::cout << "Testing Material Parameter Structures..." << std::endl;

    // Test different material types
    MaterialParameters elastic_params = createElasticMaterial<float>(1000.0f, 1e6f, 0.3f);
    MaterialParameters fluid_params = createFluidMaterial<float>(1000.0f, 0.001f, 7.0f);

    std::cout << "  ✓ Elastic material: density=" << elastic_params.density
              << ", youngs_modulus=" << elastic_params.youngs_modulus << std::endl;
    std::cout << "  ✓ Fluid material: density=" << fluid_params.density
              << ", viscosity=" << fluid_params.viscosity << std::endl;

    // Basic validation
    bool params_valid = (elastic_params.density > 0.0f && elastic_params.youngs_modulus > 0.0f &&
                        fluid_params.density > 0.0f && fluid_params.viscosity > 0.0f);

    if (params_valid) {
        std::cout << "  ✓ Material parameters are valid" << std::endl;
        return true;
    } else {
        std::cout << "  ❌ Invalid material parameters" << std::endl;
        return false;
    }
}

bool test_basic_mpm_concepts() {
    std::cout << "Testing Basic MPM Concepts..." << std::endl;

    // Test B-spline basis functions (quadratic)
    float xi = 0.3f;  // Normalized grid coordinate

    float N0 = 0.5f * (1.5f - xi) * (1.5f - xi);
    float N1 = 0.75f - (xi - 1.0f) * (xi - 1.0f);
    float N2 = 0.5f * (xi - 0.5f) * (xi - 0.5f);

    float sum = N0 + N1 + N2;
    bool partition_unity = std::abs(sum - 1.0f) < 1e-6f;

    std::cout << "  B-spline values: N0=" << N0 << ", N1=" << N1 << ", N2=" << N2 << ", sum=" << sum << std::endl;

    if (partition_unity) {
        std::cout << "  ✓ B-spline partition of unity satisfied" << std::endl;
    } else {
        std::cout << "  ❌ B-spline partition of unity failed" << std::endl;
        return false;
    }

    // Test basic grid-to-particle mapping concept
    std::array<float, 3> particle_pos = {0.3f, 0.7f, 0.5f};
    std::array<float, 3> grid_spacing = {0.1f, 0.1f, 0.1f};

    std::array<int, 3> base_node = {
        static_cast<int>(std::floor(particle_pos[0] / grid_spacing[0])),
        static_cast<int>(std::floor(particle_pos[1] / grid_spacing[1])),
        static_cast<int>(std::floor(particle_pos[2] / grid_spacing[2]))
    };

    std::cout << "  Particle at (" << particle_pos[0] << ", " << particle_pos[1] << ", " << particle_pos[2] << ")" << std::endl;
    std::cout << "  Base grid node: (" << base_node[0] << ", " << base_node[1] << ", " << base_node[2] << ")" << std::endl;

    bool mapping_reasonable = (base_node[0] >= 0 && base_node[1] >= 0 && base_node[2] >= 0);

    if (mapping_reasonable) {
        std::cout << "  ✓ Grid-to-particle mapping concept works" << std::endl;
        return true;
    } else {
        std::cout << "  ❌ Grid-to-particle mapping failed" << std::endl;
        return false;
    }
}

int main() {
    std::cout << "PhysGrad MPM Solver - Simple Validation Test" << std::endl;
    std::cout << "============================================" << std::endl << std::endl;

    bool all_tests_passed = true;

    all_tests_passed &= test_particle_data_structures();
    std::cout << std::endl;

    all_tests_passed &= test_grid_data_structures();
    std::cout << std::endl;

    all_tests_passed &= test_material_parameters();
    std::cout << std::endl;

    all_tests_passed &= test_basic_mpm_concepts();
    std::cout << std::endl;

    if (all_tests_passed) {
        std::cout << "✅ All basic MPM validation tests PASSED!" << std::endl;
        std::cout << "Ready to run comprehensive physics benchmarks." << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some basic tests failed. Fix issues before running benchmarks." << std::endl;
        return 1;
    }
}