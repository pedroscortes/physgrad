/**
 * PhysGrad Material Point Method Tests
 *
 * Comprehensive tests for MPM implementation including AoSoA data structure,
 * shape functions, constitutive models, and particle-grid transfers.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>

#include "material_point_method.h"

using namespace physgrad;
using namespace physgrad::mpm;

/**
 * Test AoSoA data structure performance and correctness
 */
template<typename T>
bool testAoSoADataStructure() {
    std::cout << "Testing AoSoA data structure...\n";

    constexpr size_t chunk_size = 64;
    constexpr size_t num_particles = 1000;

    ParticleAoSoA<T, chunk_size> particles;
    particles.resize(num_particles);

    // Test position access
    for (size_t i = 0; i < num_particles; ++i) {
        ConceptVector3D<T> pos{static_cast<T>(i), static_cast<T>(i * 2), static_cast<T>(i * 3)};
        particles.setPosition(i, pos);
    }

    // Verify position data
    bool position_test_passed = true;
    for (size_t i = 0; i < num_particles; ++i) {
        auto pos = particles.getPosition(i);
        if (std::abs(pos[0] - static_cast<T>(i)) > T{1e-6} ||
            std::abs(pos[1] - static_cast<T>(i * 2)) > T{1e-6} ||
            std::abs(pos[2] - static_cast<T>(i * 3)) > T{1e-6}) {
            position_test_passed = false;
            break;
        }
    }

    // Test velocity access
    for (size_t i = 0; i < num_particles; ++i) {
        ConceptVector3D<T> vel{static_cast<T>(i * 0.1), static_cast<T>(i * 0.2), static_cast<T>(i * 0.3)};
        particles.setVelocity(i, vel);
    }

    bool velocity_test_passed = true;
    for (size_t i = 0; i < num_particles; ++i) {
        auto vel = particles.getVelocity(i);
        if (std::abs(vel[0] - static_cast<T>(i * 0.1)) > T{1e-6} ||
            std::abs(vel[1] - static_cast<T>(i * 0.2)) > T{1e-6} ||
            std::abs(vel[2] - static_cast<T>(i * 0.3)) > T{1e-6}) {
            velocity_test_passed = false;
            break;
        }
    }

    // Test mass access
    for (size_t i = 0; i < num_particles; ++i) {
        particles.setMass(i, static_cast<T>(i + 1));
    }

    bool mass_test_passed = true;
    for (size_t i = 0; i < num_particles; ++i) {
        T mass = particles.getMass(i);
        if (std::abs(mass - static_cast<T>(i + 1)) > T{1e-6}) {
            mass_test_passed = false;
            break;
        }
    }

    // Test deformation gradient
    bool deformation_test_passed = true;
    for (size_t i = 0; i < std::min(num_particles, size_t{100}); ++i) {
        T F[9] = {T{1}, T{0}, T{0}, T{0}, T{1}, T{0}, T{0}, T{0}, T{1}};
        F[0] += static_cast<T>(i) * T{0.01}; // Add some deformation

        particles.setDeformationGradient(i, F);

        T F_retrieved[9];
        particles.getDeformationGradient(i, F_retrieved);

        for (int j = 0; j < 9; ++j) {
            if (std::abs(F_retrieved[j] - F[j]) > T{1e-6}) {
                deformation_test_passed = false;
                break;
            }
        }
        if (!deformation_test_passed) break;
    }

    bool all_passed = position_test_passed && velocity_test_passed &&
                     mass_test_passed && deformation_test_passed;

    if (all_passed) {
        std::cout << "✓ AoSoA data structure test passed\n";
    } else {
        std::cout << "✗ AoSoA data structure test failed\n";
        std::cout << "  Position: " << (position_test_passed ? "PASS" : "FAIL") << "\n";
        std::cout << "  Velocity: " << (velocity_test_passed ? "PASS" : "FAIL") << "\n";
        std::cout << "  Mass: " << (mass_test_passed ? "PASS" : "FAIL") << "\n";
        std::cout << "  Deformation: " << (deformation_test_passed ? "PASS" : "FAIL") << "\n";
    }

    return all_passed;
}

/**
 * Test shape functions and their derivatives
 */
template<typename T>
bool testShapeFunctions() {
    std::cout << "Testing shape functions...\n";

    using ShapeFunc = MPMShapeFunctions<T>;

    // Test linear shape function
    bool linear_test = true;
    // Should be 1 at x=0, 0 at x=±1
    if (std::abs(ShapeFunc::linear(T{0}) - T{1}) > T{1e-6} ||
        std::abs(ShapeFunc::linear(T{1})) > T{1e-6} ||
        std::abs(ShapeFunc::linear(T{-1})) > T{1e-6}) {
        linear_test = false;
    }

    // Test quadratic shape function
    bool quadratic_test = true;
    // Should be 0.75 at x=0
    if (std::abs(ShapeFunc::quadratic(T{0}) - T{0.75}) > T{1e-6}) {
        quadratic_test = false;
    }

    // Test cubic shape function
    bool cubic_test = true;
    // Should be 2/3 at x=0
    if (std::abs(ShapeFunc::cubic(T{0}) - T{2.0/3.0}) > T{1e-3}) {
        cubic_test = false;
    }

    // Test derivative consistency (finite difference)
    bool derivative_test = true;
    T h = T{1e-5};
    for (T x = -T{1.5}; x <= T{1.5}; x += T{0.1}) {
        // Linear derivative
        T linear_fd = (ShapeFunc::linear(x + h) - ShapeFunc::linear(x - h)) / (T{2} * h);
        T linear_analytical = ShapeFunc::linearDerivative(x);
        if (std::abs(linear_fd - linear_analytical) > T{1e-3}) {
            derivative_test = false;
            break;
        }

        // Quadratic derivative
        T quadratic_fd = (ShapeFunc::quadratic(x + h) - ShapeFunc::quadratic(x - h)) / (T{2} * h);
        T quadratic_analytical = ShapeFunc::quadraticDerivative(x);
        if (std::abs(quadratic_fd - quadratic_analytical) > T{1e-3}) {
            derivative_test = false;
            break;
        }
    }

    bool all_passed = linear_test && quadratic_test && cubic_test && derivative_test;

    if (all_passed) {
        std::cout << "✓ Shape functions test passed\n";
    } else {
        std::cout << "✗ Shape functions test failed\n";
        std::cout << "  Linear: " << (linear_test ? "PASS" : "FAIL") << "\n";
        std::cout << "  Quadratic: " << (quadratic_test ? "PASS" : "FAIL") << "\n";
        std::cout << "  Cubic: " << (cubic_test ? "PASS" : "FAIL") << "\n";
        std::cout << "  Derivatives: " << (derivative_test ? "PASS" : "FAIL") << "\n";
    }

    return all_passed;
}

/**
 * Test constitutive models
 */
template<typename T>
bool testConstitutiveModels() {
    std::cout << "Testing constitutive models...\n";

    // Test Neo-Hookean model
    T F[9] = {T{1.1}, T{0}, T{0}, T{0}, T{1}, T{0}, T{0}, T{0}, T{1}}; // Small stretch in x
    T stress[6];
    T E = T{1e6};  // Young's modulus
    T nu = T{0.3}; // Poisson's ratio

    NeoHookeanModel<T>::computeStress(F, stress, E, nu);

    // Check that stress is reasonable
    bool neo_hookean_test = true;
    for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(stress[i])) {
            neo_hookean_test = false;
            break;
        }
    }

    // Should have positive normal stress in x-direction for stretch
    if (stress[0] <= T{0}) {
        neo_hookean_test = false;
    }

    // Test von Mises plasticity
    typename VonMisesPlasticityModel<T>::PlasticState plastic_state;
    plastic_state.yield_stress = T{1e5};

    // Apply large deformation
    T F_large[9] = {T{2}, T{0}, T{0}, T{0}, T{1}, T{0}, T{0}, T{0}, T{1}};
    T stress_plastic[6];

    VonMisesPlasticityModel<T>::computeStress(F_large, stress_plastic, E, nu, plastic_state);

    bool plasticity_test = true;
    for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(stress_plastic[i])) {
            plasticity_test = false;
            break;
        }
    }

    // Should have updated plastic strain
    if (plastic_state.equivalent_plastic_strain <= T{0}) {
        plasticity_test = false;
    }

    bool all_passed = neo_hookean_test && plasticity_test;

    if (all_passed) {
        std::cout << "✓ Constitutive models test passed\n";
    } else {
        std::cout << "✗ Constitutive models test failed\n";
        std::cout << "  Neo-Hookean: " << (neo_hookean_test ? "PASS" : "FAIL") << "\n";
        std::cout << "  Plasticity: " << (plasticity_test ? "PASS" : "FAIL") << "\n";
    }

    return all_passed;
}

/**
 * Test grid data structure
 */
template<typename T>
bool testGridStructure() {
    std::cout << "Testing grid structure...\n";

    int3 dimensions = {10, 10, 10};
    ConceptVector3D<T> cell_size{T{0.1}, T{0.1}, T{0.1}};
    ConceptVector3D<T> origin{T{0}, T{0}, T{0}};

    MPMGrid<T> grid(dimensions, cell_size, origin);

    // Test node indexing
    bool indexing_test = true;
    for (int i = 0; i < dimensions.x; ++i) {
        for (int j = 0; j < dimensions.y; ++j) {
            for (int k = 0; k < dimensions.z; ++k) {
                size_t idx = grid.getNodeIndex(i, j, k);
                if (idx >= grid.total_nodes) {
                    indexing_test = false;
                    break;
                }
            }
            if (!indexing_test) break;
        }
        if (!indexing_test) break;
    }

    // Test node position calculation
    bool position_test = true;
    auto node_pos = grid.getNodePosition(5, 5, 5);
    ConceptVector3D<T> expected_pos{T{0.5}, T{0.5}, T{0.5}};

    if (std::abs(node_pos[0] - expected_pos[0]) > T{1e-6} ||
        std::abs(node_pos[1] - expected_pos[1]) > T{1e-6} ||
        std::abs(node_pos[2] - expected_pos[2]) > T{1e-6}) {
        position_test = false;
    }

    // Test grid clearing
    grid.mass[0] = T{1};
    grid.velocity[0] = ConceptVector3D<T>{T{1}, T{1}, T{1}};
    grid.clear();

    bool clear_test = (grid.mass[0] == T{0}) &&
                     (grid.velocity[0][0] == T{0}) &&
                     (grid.velocity[0][1] == T{0}) &&
                     (grid.velocity[0][2] == T{0});

    bool all_passed = indexing_test && position_test && clear_test;

    if (all_passed) {
        std::cout << "✓ Grid structure test passed\n";
    } else {
        std::cout << "✗ Grid structure test failed\n";
        std::cout << "  Indexing: " << (indexing_test ? "PASS" : "FAIL") << "\n";
        std::cout << "  Position: " << (position_test ? "PASS" : "FAIL") << "\n";
        std::cout << "  Clear: " << (clear_test ? "PASS" : "FAIL") << "\n";
    }

    return all_passed;
}

/**
 * Test basic MPM simulation
 */
template<typename T>
bool testMPMSimulation() {
    std::cout << "Testing MPM simulation...\n";

    // Create a small grid
    int3 grid_dims = {20, 20, 20};
    ConceptVector3D<T> cell_size{T{0.05}, T{0.05}, T{0.05}};
    ConceptVector3D<T> origin{T{0}, T{0}, T{0}};

    typename MPMSolver<T>::SimulationParams params;
    params.timestep = T{1e-4};
    params.interpolation_order = 2;

    MPMSolver<T> solver(grid_dims, cell_size, origin, params);

    // Add some particles
    std::vector<ConceptVector3D<T>> positions = {
        {T{0.5}, T{0.8}, T{0.5}},  // Particle above center
        {T{0.4}, T{0.7}, T{0.4}},
        {T{0.6}, T{0.7}, T{0.6}}
    };

    std::vector<ConceptVector3D<T>> velocities = {
        {T{0}, T{0}, T{0}},
        {T{0}, T{0}, T{0}},
        {T{0}, T{0}, T{0}}
    };

    std::vector<T> masses = {T{0.1}, T{0.1}, T{0.1}};
    std::vector<T> volumes = {T{0.01}, T{0.01}, T{0.01}};

    solver.addParticles(positions, velocities, masses, volumes);

    // Get initial state
    auto initial_particles = solver.getParticles();
    auto initial_pos_0 = initial_particles.getPosition(0);

    // Run simulation for a few steps
    for (int step = 0; step < 10; ++step) {
        solver.step();
    }

    // Check that particles moved due to gravity
    auto final_particles = solver.getParticles();
    auto final_pos_0 = final_particles.getPosition(0);

    bool gravity_test = final_pos_0[1] < initial_pos_0[1]; // Should fall down

    // Check that simulation time advanced
    bool time_test = solver.getCurrentTime() > T{0};

    // Check that step count increased
    bool step_test = solver.getStepCount() == 10;

    bool all_passed = gravity_test && time_test && step_test;

    if (all_passed) {
        std::cout << "✓ MPM simulation test passed\n";
        std::cout << "  Initial Y: " << initial_pos_0[1] << "\n";
        std::cout << "  Final Y: " << final_pos_0[1] << "\n";
        std::cout << "  Time: " << solver.getCurrentTime() << "\n";
        std::cout << "  Steps: " << solver.getStepCount() << "\n";
    } else {
        std::cout << "✗ MPM simulation test failed\n";
        std::cout << "  Gravity: " << (gravity_test ? "PASS" : "FAIL") << "\n";
        std::cout << "  Time: " << (time_test ? "PASS" : "FAIL") << "\n";
        std::cout << "  Steps: " << (step_test ? "PASS" : "FAIL") << "\n";
    }

    return all_passed;
}

/**
 * Test conservation properties
 */
template<typename T>
bool testConservation() {
    std::cout << "Testing conservation properties...\n";

    int3 grid_dims = {15, 15, 15};
    ConceptVector3D<T> cell_size{T{0.1}, T{0.1}, T{0.1}};
    ConceptVector3D<T> origin{T{0}, T{0}, T{0}};

    typename MPMSolver<T>::SimulationParams params;
    params.timestep = T{1e-3};
    params.gravity = {T{0}, T{0}, T{0}}; // No gravity for conservation test

    MPMSolver<T> solver(grid_dims, cell_size, origin, params);

    // Add particles with some initial momentum
    std::vector<ConceptVector3D<T>> positions = {
        {T{0.7}, T{0.7}, T{0.7}},
        {T{0.8}, T{0.7}, T{0.7}}
    };

    std::vector<ConceptVector3D<T>> velocities = {
        {T{1}, T{0}, T{0}},
        {T{-1}, T{0}, T{0}}
    };

    std::vector<T> masses = {T{1}, T{1}};
    std::vector<T> volumes = {T{0.1}, T{0.1}};

    solver.addParticles(positions, velocities, masses, volumes);

    // Calculate initial momentum
    ConceptVector3D<T> initial_momentum{T{0}, T{0}, T{0}};
    auto particles = solver.getParticles();
    for (size_t i = 0; i < particles.num_particles; ++i) {
        auto vel = particles.getVelocity(i);
        T mass = particles.getMass(i);
        initial_momentum = {
            initial_momentum[0] + mass * vel[0],
            initial_momentum[1] + mass * vel[1],
            initial_momentum[2] + mass * vel[2]
        };
    }

    // Run simulation
    for (int step = 0; step < 5; ++step) {
        solver.step();
    }

    // Calculate final momentum
    ConceptVector3D<T> final_momentum{T{0}, T{0}, T{0}};
    auto final_particles = solver.getParticles();
    for (size_t i = 0; i < final_particles.num_particles; ++i) {
        auto vel = final_particles.getVelocity(i);
        T mass = final_particles.getMass(i);
        final_momentum = {
            final_momentum[0] + mass * vel[0],
            final_momentum[1] + mass * vel[1],
            final_momentum[2] + mass * vel[2]
        };
    }

    // Check momentum conservation
    T momentum_error = std::sqrt(
        (final_momentum[0] - initial_momentum[0]) * (final_momentum[0] - initial_momentum[0]) +
        (final_momentum[1] - initial_momentum[1]) * (final_momentum[1] - initial_momentum[1]) +
        (final_momentum[2] - initial_momentum[2]) * (final_momentum[2] - initial_momentum[2])
    );

    bool momentum_conserved = momentum_error < T{0.1}; // Allow small numerical error

    if (momentum_conserved) {
        std::cout << "✓ Conservation test passed\n";
        std::cout << "  Initial momentum: (" << initial_momentum[0] << ", "
                  << initial_momentum[1] << ", " << initial_momentum[2] << ")\n";
        std::cout << "  Final momentum: (" << final_momentum[0] << ", "
                  << final_momentum[1] << ", " << final_momentum[2] << ")\n";
        std::cout << "  Error: " << momentum_error << "\n";
    } else {
        std::cout << "✗ Conservation test failed\n";
        std::cout << "  Momentum error: " << momentum_error << "\n";
    }

    return momentum_conserved;
}

/**
 * Main test function
 */
int main() {
    std::cout << "PhysGrad Material Point Method Tests\n";
    std::cout << "====================================\n\n";

    std::cout << std::fixed << std::setprecision(6);

    bool all_passed = true;

    // Test with float precision
    std::cout << "--- Float precision tests ---\n";
    all_passed &= testAoSoADataStructure<float>();
    std::cout << "\n";

    all_passed &= testShapeFunctions<float>();
    std::cout << "\n";

    all_passed &= testConstitutiveModels<float>();
    std::cout << "\n";

    all_passed &= testGridStructure<float>();
    std::cout << "\n";

    all_passed &= testMPMSimulation<float>();
    std::cout << "\n";

    all_passed &= testConservation<float>();
    std::cout << "\n";

    // Test with double precision (selected tests)
    std::cout << "--- Double precision tests ---\n";
    all_passed &= testAoSoADataStructure<double>();
    std::cout << "\n";

    all_passed &= testShapeFunctions<double>();
    std::cout << "\n";

    if (all_passed) {
        std::cout << "✓ All MPM tests PASSED!\n";
        return 0;
    } else {
        std::cout << "✗ Some MPM tests FAILED!\n";
        return 1;
    }
}