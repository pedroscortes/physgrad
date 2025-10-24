#include "src/fsi_coupling.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cassert>

using namespace physgrad;

struct FluidParticle {
    Vec3<double> position;
    Vec3<double> velocity;
    double pressure;
    double density;

    FluidParticle(Vec3<double> pos = Vec3<double>{0, 0, 0},
                  Vec3<double> vel = Vec3<double>{0, 0, 0})
        : position(pos), velocity(vel), pressure(0.0), density(1000.0) {}
};

struct StructureNode {
    Vec3<double> position;
    Vec3<double> velocity;
    Vec3<double> force;
    double mass;

    StructureNode(Vec3<double> pos = Vec3<double>{0, 0, 0}, double m = 1.0)
        : position(pos), velocity{0, 0, 0}, force{0, 0, 0}, mass(m) {}
};

class TestFluidSolver {
public:
    std::vector<FluidParticle> particles;
    double dt;

    TestFluidSolver(double timestep = 0.001) : dt(timestep) {}

    void addParticle(const Vec3<double>& pos, const Vec3<double>& vel = {0, 0, 0}) {
        particles.emplace_back(pos, vel);
    }

    void step() {
        // Simple fluid dynamics simulation
        for (auto& p : particles) {
            // Apply gravity
            p.velocity.y -= 9.81 * dt;

            // Update position
            p.position.x += p.velocity.x * dt;
            p.position.y += p.velocity.y * dt;
            p.position.z += p.velocity.z * dt;

            // Simple pressure calculation
            p.pressure = std::max(0.0, 1000.0 * 9.81 * (10.0 - p.position.y));
        }
    }

    std::vector<Vec3<double>> getPositions() const {
        std::vector<Vec3<double>> positions;
        for (const auto& p : particles) {
            positions.push_back(p.position);
        }
        return positions;
    }

    std::vector<Vec3<double>> getVelocities() const {
        std::vector<Vec3<double>> velocities;
        for (const auto& p : particles) {
            velocities.push_back(p.velocity);
        }
        return velocities;
    }

    void applyForces(const std::vector<Vec3<double>>& forces) {
        for (size_t i = 0; i < std::min(particles.size(), forces.size()); ++i) {
            particles[i].velocity.x += forces[i].x * dt / 1000.0; // mass = 1000 kg/m³
            particles[i].velocity.y += forces[i].y * dt / 1000.0;
            particles[i].velocity.z += forces[i].z * dt / 1000.0;
        }
    }
};

class TestStructureSolver {
public:
    std::vector<StructureNode> nodes;
    double dt;

    TestStructureSolver(double timestep = 0.001) : dt(timestep) {}

    void addNode(const Vec3<double>& pos, double mass = 1.0) {
        nodes.emplace_back(pos, mass);
    }

    void step() {
        // Simple structural dynamics
        for (auto& node : nodes) {
            // Apply forces (including gravity)
            node.force.y -= node.mass * 9.81;

            // Update velocity and position
            node.velocity.x += node.force.x * dt / node.mass;
            node.velocity.y += node.force.y * dt / node.mass;
            node.velocity.z += node.force.z * dt / node.mass;

            node.position.x += node.velocity.x * dt;
            node.position.y += node.velocity.y * dt;
            node.position.z += node.velocity.z * dt;

            // Reset forces
            node.force = {0, 0, 0};
        }
    }

    std::vector<Vec3<double>> getPositions() const {
        std::vector<Vec3<double>> positions;
        for (const auto& node : nodes) {
            positions.push_back(node.position);
        }
        return positions;
    }

    std::vector<Vec3<double>> getVelocities() const {
        std::vector<Vec3<double>> velocities;
        for (const auto& node : nodes) {
            velocities.push_back(node.velocity);
        }
        return velocities;
    }

    void applyForces(const std::vector<Vec3<double>>& forces) {
        for (size_t i = 0; i < std::min(nodes.size(), forces.size()); ++i) {
            nodes[i].force.x += forces[i].x;
            nodes[i].force.y += forces[i].y;
            nodes[i].force.z += forces[i].z;
        }
    }
};

bool testImmersedBoundaryMethod() {
    std::cout << "Testing Immersed Boundary Method..." << std::endl;

    ImmersedBoundaryMethod<double> ibm(0.1); // support radius = 0.1

    // Test delta function
    double delta1 = ibm.deltaFunction(0.0);   // At center
    double delta2 = ibm.deltaFunction(0.05);  // Within support
    double delta3 = ibm.deltaFunction(0.15);  // Outside support

    std::cout << "Delta function values: " << delta1 << ", " << delta2 << ", " << delta3 << std::endl;

    // Delta function should be positive at center, decrease with distance, zero outside support
    if (delta1 <= 0 || delta2 >= delta1 || delta3 != 0.0) {
        std::cout << "❌ Delta function test failed" << std::endl;
        return false;
    }

    // Test force transfer
    std::vector<Vec3<double>> fluid_pos = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, {0.2, 0.0, 0.0}};
    std::vector<Vec3<double>> struct_pos = {{0.05, 0.0, 0.0}};
    std::vector<Vec3<double>> struct_forces = {{10.0, 0.0, 0.0}};

    auto fluid_forces = ibm.transferForces(struct_pos, struct_forces, fluid_pos);

    std::cout << "Transferred forces: ";
    for (const auto& f : fluid_forces) {
        std::cout << "(" << f.x << ", " << f.y << ", " << f.z << ") ";
    }
    std::cout << std::endl;

    // Forces should be distributed to nearby fluid particles
    bool has_nonzero_force = false;
    for (const auto& f : fluid_forces) {
        if (f.x != 0 || f.y != 0 || f.z != 0) {
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

    PartitionedCouplingScheme<double> coupling(10, 1e-6); // 10 iterations, 1e-6 tolerance

    // Test coupling iteration
    std::vector<Vec3<double>> fluid_pos = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Vec3<double>> fluid_vel = {{1.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    std::vector<Vec3<double>> struct_pos = {{0.5, 0.0, 0.0}};
    std::vector<Vec3<double>> struct_vel = {{0.0, 0.0, 0.0}};

    auto result = coupling.couple(fluid_pos, fluid_vel, struct_pos, struct_vel);

    std::cout << "Coupling converged in " << result.iterations << " iterations" << std::endl;
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
        std::cout << "❌ Coupling forces test failed" << std::endl;
        return false;
    }

    std::cout << "✅ Partitioned Coupling test passed" << std::endl;
    return true;
}

bool testFSISimulation() {
    std::cout << "Testing FSI Simulation Manager..." << std::endl;

    auto coupling_method = FSICouplingFactory<double>::create("immersed_boundary", {{"support_radius", 0.1}});
    FSISimulationManager<double> sim_manager(std::move(coupling_method));

    // Create test solvers
    TestFluidSolver fluid_solver(0.001);
    TestStructureSolver structure_solver(0.001);

    // Add particles and nodes
    for (int i = 0; i < 10; ++i) {
        fluid_solver.addParticle({i * 0.1, 0.0, 0.0}, {1.0, 0.0, 0.0});
    }
    structure_solver.addNode({0.5, 0.0, 0.0}, 1.0);

    // Run simulation steps
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < 100; ++step) {
        // Get current states
        auto fluid_pos = fluid_solver.getPositions();
        auto fluid_vel = fluid_solver.getVelocities();
        auto struct_pos = structure_solver.getPositions();
        auto struct_vel = structure_solver.getVelocities();

        // Perform FSI coupling
        auto coupling_result = sim_manager.performCoupling(fluid_pos, fluid_vel, struct_pos, struct_vel);

        if (!coupling_result.converged) {
            std::cout << "❌ FSI coupling failed to converge at step " << step << std::endl;
            return false;
        }

        // Apply coupling forces
        fluid_solver.applyForces(coupling_result.fluid_forces);
        structure_solver.applyForces(coupling_result.structure_forces);

        // Advance solvers
        fluid_solver.step();
        structure_solver.step();

        // Check for reasonable behavior
        for (const auto& pos : fluid_solver.getPositions()) {
            if (std::isnan(pos.x) || std::isnan(pos.y) || std::isnan(pos.z)) {
                std::cout << "❌ NaN detected in fluid positions at step " << step << std::endl;
                return false;
            }
        }

        for (const auto& pos : structure_solver.getPositions()) {
            if (std::isnan(pos.x) || std::isnan(pos.y) || std::isnan(pos.z)) {
                std::cout << "❌ NaN detected in structure positions at step " << step << std::endl;
                return false;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "FSI simulation completed 100 steps in " << duration.count() << " ms" << std::endl;

    // Get performance metrics
    auto metrics = sim_manager.getPerformanceMetrics();
    std::cout << "Average coupling time: " << metrics.average_coupling_time * 1000 << " ms" << std::endl;
    std::cout << "Average iterations: " << metrics.average_iterations << std::endl;

    std::cout << "✅ FSI Simulation test passed" << std::endl;
    return true;
}

bool testFlowAroundCylinder() {
    std::cout << "Testing Flow Around Cylinder..." << std::endl;

    auto coupling_method = FSICouplingFactory<double>::create("immersed_boundary", {{"support_radius", 0.05}});
    FSISimulationManager<double> sim_manager(std::move(coupling_method));

    TestFluidSolver fluid_solver(0.001);
    TestStructureSolver structure_solver(0.001);

    // Create fluid domain (10x5 grid)
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 25; ++j) {
            double x = i * 0.2;
            double y = j * 0.2;
            Vec3<double> inlet_velocity = {2.0, 0.0, 0.0}; // 2 m/s inlet
            fluid_solver.addParticle({x, y, 0.0}, inlet_velocity);
        }
    }

    // Create cylinder boundary (circle at center)
    double cx = 5.0, cy = 2.5, radius = 0.5;
    for (int theta_deg = 0; theta_deg < 360; theta_deg += 10) {
        double theta = theta_deg * M_PI / 180.0;
        double x = cx + radius * cos(theta);
        double y = cy + radius * sin(theta);
        structure_solver.addNode({x, y, 0.0}, 0.1);
    }

    std::cout << "Created " << fluid_solver.particles.size() << " fluid particles" << std::endl;
    std::cout << "Created " << structure_solver.nodes.size() << " structure nodes" << std::endl;

    // Run simulation
    double total_drag = 0.0;
    int drag_samples = 0;

    for (int step = 0; step < 200; ++step) {
        auto fluid_pos = fluid_solver.getPositions();
        auto fluid_vel = fluid_solver.getVelocities();
        auto struct_pos = structure_solver.getPositions();
        auto struct_vel = structure_solver.getVelocities();

        auto coupling_result = sim_manager.performCoupling(fluid_pos, fluid_vel, struct_pos, struct_vel);

        // Calculate drag force
        double drag_force = 0.0;
        for (const auto& force : coupling_result.structure_forces) {
            drag_force += force.x;
        }

        if (step > 50) { // Skip initial transient
            total_drag += std::abs(drag_force);
            drag_samples++;
        }

        fluid_solver.applyForces(coupling_result.fluid_forces);
        structure_solver.applyForces(coupling_result.structure_forces);

        fluid_solver.step();
        structure_solver.step();

        if (step % 50 == 0) {
            std::cout << "Step " << step << ", Drag force: " << drag_force << " N" << std::endl;
        }
    }

    double avg_drag = total_drag / drag_samples;
    std::cout << "Average drag force: " << avg_drag << " N" << std::endl;

    // Drag should be reasonable for cylinder in cross-flow
    if (avg_drag < 0.1 || avg_drag > 100.0) {
        std::cout << "❌ Unrealistic drag force" << std::endl;
        return false;
    }

    std::cout << "✅ Flow Around Cylinder test passed" << std::endl;
    return true;
}

bool testPerformanceScaling() {
    std::cout << "Testing FSI Performance Scaling..." << std::endl;

    std::vector<size_t> particle_counts = {100, 500, 1000, 2000};

    for (size_t n_particles : particle_counts) {
        auto coupling_method = FSICouplingFactory<double>::create("immersed_boundary", {{"support_radius", 0.1}});
        FSISimulationManager<double> sim_manager(std::move(coupling_method));

        TestFluidSolver fluid_solver(0.001);
        TestStructureSolver structure_solver(0.001);

        // Create particles
        for (size_t i = 0; i < n_particles; ++i) {
            double x = (i % 100) * 0.1;
            double y = (i / 100) * 0.1;
            fluid_solver.addParticle({x, y, 0.0}, {1.0, 0.0, 0.0});
        }

        // Create structure nodes (10% of fluid particles)
        for (size_t i = 0; i < n_particles / 10; ++i) {
            double x = i * 0.1 + 0.05;
            structure_solver.addNode({x, 0.0, 0.0}, 1.0);
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Run 10 coupling steps
        for (int step = 0; step < 10; ++step) {
            auto fluid_pos = fluid_solver.getPositions();
            auto fluid_vel = fluid_solver.getVelocities();
            auto struct_pos = structure_solver.getPositions();
            auto struct_vel = structure_solver.getVelocities();

            auto coupling_result = sim_manager.performCoupling(fluid_pos, fluid_vel, struct_pos, struct_vel);

            fluid_solver.applyForces(coupling_result.fluid_forces);
            structure_solver.applyForces(coupling_result.structure_forces);

            fluid_solver.step();
            structure_solver.step();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double time_per_particle = static_cast<double>(duration.count()) / n_particles;
        std::cout << n_particles << " particles: " << duration.count() << " μs total, "
                  << time_per_particle << " μs/particle" << std::endl;
    }

    std::cout << "✅ Performance scaling test passed" << std::endl;
    return true;
}

int main() {
    std::cout << "=== FSI Coupling Test Suite ===" << std::endl;

    bool all_passed = true;

    all_passed &= testImmersedBoundaryMethod();
    all_passed &= testPartitionedCoupling();
    all_passed &= testFSISimulation();
    all_passed &= testFlowAroundCylinder();
    all_passed &= testPerformanceScaling();

    std::cout << "\n=== Test Summary ===" << std::endl;
    if (all_passed) {
        std::cout << "✅ All FSI coupling tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some FSI coupling tests failed!" << std::endl;
        return 1;
    }
}