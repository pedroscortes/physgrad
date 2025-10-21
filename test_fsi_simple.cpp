#include "src/fsi_coupling.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cassert>

using namespace physgrad;

struct CouplingResult {
    std::vector<Vec3<double>> fluid_forces;
    std::vector<Vec3<double>> structure_forces;
    bool converged;
    int iterations;
    double residual;
};

bool testVec3Operations() {
    std::cout << "Testing Vec3 operations..." << std::endl;

    Vec3<double> v1{1.0, 2.0, 3.0};
    Vec3<double> v2{4.0, 5.0, 6.0};

    auto sum = v1 + v2;
    auto diff = v2 - v1;
    auto scaled = v1 * 2.0;

    std::cout << "v1: (" << v1.x << ", " << v1.y << ", " << v1.z << ")" << std::endl;
    std::cout << "v2: (" << v2.x << ", " << v2.y << ", " << v2.z << ")" << std::endl;
    std::cout << "sum: (" << sum.x << ", " << sum.y << ", " << sum.z << ")" << std::endl;
    std::cout << "diff: (" << diff.x << ", " << diff.y << ", " << diff.z << ")" << std::endl;
    std::cout << "scaled: (" << scaled.x << ", " << scaled.y << ", " << scaled.z << ")" << std::endl;

    double dot_product = v1.dot(v2);
    double magnitude = v1.magnitude();

    std::cout << "dot product: " << dot_product << std::endl;
    std::cout << "magnitude: " << magnitude << std::endl;

    // Check expected values
    if (std::abs(sum.x - 5.0) > 1e-10 || std::abs(sum.y - 7.0) > 1e-10 || std::abs(sum.z - 9.0) > 1e-10) {
        std::cout << "❌ Vec3 addition test failed" << std::endl;
        return false;
    }

    if (std::abs(dot_product - 32.0) > 1e-10) {
        std::cout << "❌ Vec3 dot product test failed" << std::endl;
        return false;
    }

    if (std::abs(magnitude - std::sqrt(14.0)) > 1e-10) {
        std::cout << "❌ Vec3 magnitude test failed" << std::endl;
        return false;
    }

    std::cout << "✅ Vec3 operations test passed" << std::endl;
    return true;
}

bool testImmersedBoundaryMethod() {
    std::cout << "Testing Immersed Boundary Method..." << std::endl;

    fsi::ImmersedBoundaryMethod<double> ibm(0.1);

    // Test delta function
    double delta1 = ibm.deltaFunction(0.0);   // At center
    double delta2 = ibm.deltaFunction(0.05);  // Within support
    double delta3 = ibm.deltaFunction(0.15);  // Outside support

    std::cout << "Delta function values: " << delta1 << ", " << delta2 << ", " << delta3 << std::endl;

    if (delta1 <= 0 || delta2 >= delta1 || delta3 != 0.0) {
        std::cout << "❌ Delta function test failed" << std::endl;
        return false;
    }

    // Test force transfer between position vectors
    std::vector<Vec3<double>> fluid_pos = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, {0.2, 0.0, 0.0}};
    std::vector<Vec3<double>> struct_pos = {{0.05, 0.0, 0.0}};
    std::vector<Vec3<double>> struct_forces = {{10.0, 0.0, 0.0}};

    auto fluid_forces = ibm.transferForces(struct_pos, struct_forces, fluid_pos);

    std::cout << "Transferred forces: ";
    for (const auto& f : fluid_forces) {
        std::cout << "(" << f.x << ", " << f.y << ", " << f.z << ") ";
    }
    std::cout << std::endl;

    // Check that forces are transferred
    bool has_nonzero_force = false;
    for (const auto& f : fluid_forces) {
        if (std::abs(f.x) > 1e-10 || std::abs(f.y) > 1e-10 || std::abs(f.z) > 1e-10) {
            has_nonzero_force = true;
            break;
        }
    }

    if (!has_nonzero_force) {
        std::cout << "❌ Force transfer test failed" << std::endl;
        return false;
    }

    std::cout << "✅ Immersed Boundary Method test passed" << std::endl;
    return true;
}

bool testPartitionedCoupling() {
    std::cout << "Testing Partitioned Coupling Scheme..." << std::endl;

    fsi::PartitionedCouplingScheme<double> coupling(10, 1e-6);

    std::vector<Vec3<double>> fluid_pos = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Vec3<double>> fluid_vel = {{1.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    std::vector<Vec3<double>> struct_pos = {{0.5, 0.0, 0.0}};
    std::vector<Vec3<double>> struct_vel = {{0.0, 0.0, 0.0}};

    auto result = coupling.couple(fluid_pos, fluid_vel, struct_pos, struct_vel);

    std::cout << "Coupling converged: " << (result.converged ? "Yes" : "No") << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;

    if (!result.converged) {
        std::cout << "❌ Coupling convergence test failed" << std::endl;
        return false;
    }

    // Check that coupling forces are reasonable
    bool has_reasonable_forces = true;
    for (const auto& f : result.fluid_forces) {
        if (std::abs(f.x) > 1000 || std::abs(f.y) > 1000 || std::abs(f.z) > 1000) {
            has_reasonable_forces = false;
            break;
        }
    }

    if (!has_reasonable_forces) {
        std::cout << "❌ Unreasonable coupling forces" << std::endl;
        return false;
    }

    std::cout << "✅ Partitioned Coupling test passed" << std::endl;
    return true;
}

bool testFSIFactory() {
    std::cout << "Testing FSI Coupling Factory..." << std::endl;

    // Test creating immersed boundary method
    auto ibm = fsi::FSICouplingFactory<double>::create("immersed_boundary", {{"support_radius", 0.1}});
    if (!ibm) {
        std::cout << "❌ Failed to create ImmersedBoundaryMethod" << std::endl;
        return false;
    }

    // Test creating partitioned scheme
    auto partitioned = fsi::FSICouplingFactory<double>::create("partitioned_scheme", {{"max_iterations", 10.0}, {"tolerance", 1e-6}});
    if (!partitioned) {
        std::cout << "❌ Failed to create PartitionedCouplingScheme" << std::endl;
        return false;
    }

    // Test invalid method
    auto invalid = fsi::FSICouplingFactory<double>::create("invalid_method", {});
    if (invalid) {
        std::cout << "❌ Factory should return nullptr for invalid method" << std::endl;
        return false;
    }

    std::cout << "✅ FSI Factory test passed" << std::endl;
    return true;
}

bool testPerformanceScaling() {
    std::cout << "Testing FSI Performance Scaling..." << std::endl;

    std::vector<size_t> particle_counts = {50, 100, 200, 400};

    for (size_t n_particles : particle_counts) {
        fsi::ImmersedBoundaryMethod<double> ibm(0.1);

        // Create particle positions
        std::vector<Vec3<double>> fluid_pos;
        std::vector<Vec3<double>> struct_pos;
        std::vector<Vec3<double>> struct_forces;

        for (size_t i = 0; i < n_particles; ++i) {
            double x = (i % 20) * 0.1;
            double y = (i / 20) * 0.1;
            fluid_pos.push_back({x, y, 0.0});
        }

        for (size_t i = 0; i < n_particles / 10; ++i) {
            double x = i * 0.1 + 0.05;
            struct_pos.push_back({x, 0.0, 0.0});
            struct_forces.push_back({1.0, 0.0, 0.0});
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Perform force transfer
        for (int iter = 0; iter < 10; ++iter) {
            auto forces = ibm.transferForces(struct_pos, struct_forces, fluid_pos);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double time_per_particle = static_cast<double>(duration.count()) / (n_particles * 10);
        std::cout << n_particles << " particles: " << duration.count() << " μs total, "
                  << time_per_particle << " μs/particle/iteration" << std::endl;
    }

    std::cout << "✅ Performance scaling test passed" << std::endl;
    return true;
}

bool testEnergyConservation() {
    std::cout << "Testing Energy Conservation in FSI..." << std::endl;

    fsi::PartitionedCouplingScheme<double> coupling(10, 1e-6);

    // Setup simple oscillating system
    std::vector<Vec3<double>> fluid_pos = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, {0.2, 0.0, 0.0}};
    std::vector<Vec3<double>> fluid_vel = {{1.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    std::vector<Vec3<double>> struct_pos = {{0.15, 0.0, 0.0}};
    std::vector<Vec3<double>> struct_vel = {{0.0, 0.0, 0.0}};

    double initial_kinetic_energy = 0.0;
    for (const auto& vel : fluid_vel) {
        initial_kinetic_energy += 0.5 * vel.dot(vel);
    }
    for (const auto& vel : struct_vel) {
        initial_kinetic_energy += 0.5 * vel.dot(vel);
    }

    // Run coupling for several steps
    for (int step = 0; step < 50; ++step) {
        auto result = coupling.couple(fluid_pos, fluid_vel, struct_pos, struct_vel);

        if (!result.converged) {
            std::cout << "❌ Coupling failed to converge at step " << step << std::endl;
            return false;
        }

        // Simple time integration
        double dt = 0.01;
        for (size_t i = 0; i < fluid_vel.size(); ++i) {
            fluid_vel[i] = fluid_vel[i] + result.fluid_forces[i] * dt;
            fluid_pos[i] = fluid_pos[i] + fluid_vel[i] * dt;
        }
        for (size_t i = 0; i < struct_vel.size(); ++i) {
            struct_vel[i] = struct_vel[i] + result.structure_forces[i] * dt;
            struct_pos[i] = struct_pos[i] + struct_vel[i] * dt;
        }
    }

    double final_kinetic_energy = 0.0;
    for (const auto& vel : fluid_vel) {
        final_kinetic_energy += 0.5 * vel.dot(vel);
    }
    for (const auto& vel : struct_vel) {
        final_kinetic_energy += 0.5 * vel.dot(vel);
    }

    double energy_change = std::abs(final_kinetic_energy - initial_kinetic_energy);
    double relative_change = energy_change / (initial_kinetic_energy + 1e-10);

    std::cout << "Initial kinetic energy: " << initial_kinetic_energy << std::endl;
    std::cout << "Final kinetic energy: " << final_kinetic_energy << std::endl;
    std::cout << "Relative energy change: " << relative_change << std::endl;

    // Allow for some energy dissipation due to coupling
    if (relative_change > 0.5) {
        std::cout << "❌ Excessive energy dissipation" << std::endl;
        return false;
    }

    std::cout << "✅ Energy conservation test passed" << std::endl;
    return true;
}

int main() {
    std::cout << "=== FSI Coupling Simple Test Suite ===" << std::endl;

    bool all_passed = true;

    all_passed &= testVec3Operations();
    all_passed &= testImmersedBoundaryMethod();
    all_passed &= testPartitionedCoupling();
    all_passed &= testFSIFactory();
    all_passed &= testPerformanceScaling();
    all_passed &= testEnergyConservation();

    std::cout << "\n=== Test Summary ===" << std::endl;
    if (all_passed) {
        std::cout << "✅ All FSI coupling tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some FSI coupling tests failed!" << std::endl;
        return 1;
    }
}