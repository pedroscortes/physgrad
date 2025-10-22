/**
 * Performance Profiling for FSI Demo
 *
 * Instruments the FSI demo to identify performance bottlenecks
 */

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <memory>
#include <cmath>

#include "fsi_coupling.h"
#include "mpm_data_structures.h"

using namespace physgrad;
using namespace physgrad::fsi;
using namespace std::chrono;

struct TimingStats {
    double total_time = 0.0;
    double min_time = 1e9;
    double max_time = 0.0;
    int count = 0;

    void record(double time_ms) {
        total_time += time_ms;
        min_time = std::min(min_time, time_ms);
        max_time = std::max(max_time, time_ms);
        count++;
    }

    double average() const { return count > 0 ? total_time / count : 0.0; }
};

class ProfilingTimer {
public:
    ProfilingTimer(const std::string& name, TimingStats& stats)
        : name_(name), stats_(stats), start_(high_resolution_clock::now()) {}

    ~ProfilingTimer() {
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start_).count() / 1000.0;
        stats_.record(duration);
    }

private:
    std::string name_;
    TimingStats& stats_;
    time_point<high_resolution_clock> start_;
};

// Global timing stats
TimingStats stats_fsi_coupling;
TimingStats stats_force_computation;
TimingStats stats_particle_update;
TimingStats stats_total_simulation;

float runProfiledSimulation(int fluid_grid_x, int fluid_grid_y,
                           float fluid_spacing, float inlet_velocity,
                           float cylinder_radius, float cylinder_aspect_ratio,
                           int num_timesteps, float dt, float support_radius) {
    ProfilingTimer timer("total_simulation", stats_total_simulation);

    // Create FSI coupling
    auto coupling_method = std::make_unique<ImmersedBoundaryMethod<float>>(
        support_radius, 1.0f);

    // Create fluid particles
    mpm::ParticleAoSoA<float> fluid_particles(fluid_grid_x * fluid_grid_y);

    size_t particle_idx = 0;
    for (int i = 0; i < fluid_grid_x; ++i) {
        for (int j = 0; j < fluid_grid_y; ++j) {
            float x = i * fluid_spacing;
            float y = j * fluid_spacing;
            fluid_particles.setPosition(particle_idx, x, y, 0.0);
            fluid_particles.setVelocity(particle_idx, inlet_velocity, 0.0, 0.0);
            fluid_particles.setMass(particle_idx, 1000.0f * fluid_spacing * fluid_spacing);
            particle_idx++;
        }
    }

    // Create cylinder particles
    int cylinder_points = 32;
    mpm::ParticleAoSoA<float> cylinder_particles(cylinder_points);

    for (int i = 0; i < cylinder_points; ++i) {
        float angle = 2.0f * M_PI * i / cylinder_points;
        float x = 1.0f + cylinder_radius * std::cos(angle);
        float y = 1.0f + cylinder_radius * cylinder_aspect_ratio * std::sin(angle);
        cylinder_particles.setPosition(i, x, y, 0.0);
        cylinder_particles.setVelocity(i, 0.0, 0.0, 0.0);
        cylinder_particles.setMass(i, 1.0);
    }

    // Simulate
    float total_drag = 0.0f;
    int force_samples = 0;

    for (int t = 0; t < num_timesteps; ++t) {
        // FSI coupling
        {
            ProfilingTimer timer("fsi_coupling", stats_fsi_coupling);
            coupling_method->couple(fluid_particles, cylinder_particles, dt, t * dt);
        }

        // Force computation
        {
            ProfilingTimer timer("force_computation", stats_force_computation);
            auto solid_forces = coupling_method->computeSolidForces(
                fluid_particles, cylinder_particles);

            if (t > num_timesteps / 2) {
                float drag_x = 0.0f;
                for (size_t i = 0; i < cylinder_points; ++i) {
                    drag_x += solid_forces[i * 3];
                }
                total_drag += std::abs(drag_x);
                force_samples++;
            }
        }

        // Particle updates
        {
            ProfilingTimer timer("particle_update", stats_particle_update);
            for (size_t i = 0; i < fluid_particles.size(); ++i) {
                float vx, vy, vz;
                fluid_particles.getVelocity(i, vx, vy, vz);

                float x, y, z;
                fluid_particles.getPosition(i, x, y, z);
                if (x < fluid_spacing) {
                    fluid_particles.setVelocity(i, inlet_velocity, 0.0f, 0.0f);
                }
            }
        }
    }

    return (force_samples > 0) ? (total_drag / force_samples) : 1e6;
}

int main() {
    std::cout << "\n=== PhysGrad FSI Performance Profiling ===\n\n";

    // Configuration
    int fluid_grid_x = 40;
    int fluid_grid_y = 20;
    float fluid_spacing = 0.1f;
    float inlet_velocity = 2.0f;
    float cylinder_radius = 0.2f;
    float cylinder_aspect_ratio = 1.0f;
    int num_timesteps = 50;
    float dt = 0.001f;
    float support_radius = 0.15f;

    std::cout << "Configuration:\n";
    std::cout << "  Fluid particles: " << (fluid_grid_x * fluid_grid_y) << "\n";
    std::cout << "  Cylinder points: 32\n";
    std::cout << "  Timesteps: " << num_timesteps << "\n";
    std::cout << "  Support radius: " << support_radius << " m\n\n";

    // Run multiple simulations to get stable timings
    const int num_runs = 10;
    std::cout << "Running " << num_runs << " simulations...\n\n";

    for (int run = 0; run < num_runs; ++run) {
        runProfiledSimulation(fluid_grid_x, fluid_grid_y, fluid_spacing, inlet_velocity,
                             cylinder_radius, cylinder_aspect_ratio,
                             num_timesteps, dt, support_radius);
    }

    // Report results
    std::cout << "=== Timing Results (averaged over " << num_runs << " runs) ===\n\n";
    std::cout << std::fixed << std::setprecision(3);

    auto print_stats = [](const std::string& name, const TimingStats& stats) {
        std::cout << name << ":\n";
        std::cout << "  Average: " << std::setw(8) << stats.average() << " ms/call\n";
        std::cout << "  Min:     " << std::setw(8) << stats.min_time << " ms\n";
        std::cout << "  Max:     " << std::setw(8) << stats.max_time << " ms\n";
        std::cout << "  Total:   " << std::setw(8) << stats.total_time << " ms\n";
        std::cout << "  Calls:   " << std::setw(8) << stats.count << "\n\n";
    };

    print_stats("Total Simulation", stats_total_simulation);
    print_stats("FSI Coupling", stats_fsi_coupling);
    print_stats("Force Computation", stats_force_computation);
    print_stats("Particle Update", stats_particle_update);

    // Breakdown by percentage
    double total = stats_total_simulation.total_time;
    std::cout << "=== Time Breakdown ===\n\n";
    std::cout << "FSI Coupling:      " << std::setw(5) << std::setprecision(1)
              << (stats_fsi_coupling.total_time / total * 100.0) << "%\n";
    std::cout << "Force Computation: " << std::setw(5)
              << (stats_force_computation.total_time / total * 100.0) << "%\n";
    std::cout << "Particle Update:   " << std::setw(5)
              << (stats_particle_update.total_time / total * 100.0) << "%\n";

    double accounted = stats_fsi_coupling.total_time +
                      stats_force_computation.total_time +
                      stats_particle_update.total_time;
    std::cout << "Other:             " << std::setw(5)
              << ((total - accounted) / total * 100.0) << "%\n\n";

    std::cout << "=== Performance Metrics ===\n\n";
    double sims_per_sec = num_runs / (stats_total_simulation.total_time / 1000.0);
    std::cout << "Simulations/second: " << std::setprecision(2) << sims_per_sec << "\n";

    int total_timesteps = num_runs * num_timesteps;
    double timesteps_per_sec = total_timesteps / (stats_total_simulation.total_time / 1000.0);
    std::cout << "Timesteps/second:   " << std::setprecision(0) << timesteps_per_sec << "\n";

    std::cout << "\n=== Optimization Opportunities ===\n\n";

    if (stats_fsi_coupling.total_time / total > 0.5) {
        std::cout << "⚠️  FSI coupling dominates runtime (>50%)\n";
        std::cout << "   → Consider spatial hashing for neighbor search\n";
        std::cout << "   → GPU acceleration for particle-particle interactions\n\n";
    }

    if (stats_force_computation.total_time / total > 0.2) {
        std::cout << "⚠️  Force computation significant (>20%)\n";
        std::cout << "   → Vectorize force summation\n";
        std::cout << "   → Cache force evaluations if possible\n\n";
    }

    std::cout << "Profile complete!\n\n";

    return 0;
}
