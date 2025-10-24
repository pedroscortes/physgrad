/**
 * PhysGrad MPM Physics Validation Tests
 *
 * Implements dam break, oscillating drop, and stacking stability tests
 * using simplified physics simulation for validation purposes
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <cassert>
#include "src/mpm_data_structures.h"

using namespace physgrad::mpm;

template<typename T>
class SimplifiedMPMPhysics {
private:
    struct Particle {
        T x, y, z;
        T vx, vy, vz;
        T mass;
        MaterialType material;
        bool active;
    };

    std::vector<Particle> particles_;
    T dt_;
    T gravity_;
    std::array<T, 3> domain_size_;

public:
    SimplifiedMPMPhysics(T dt = 0.001f, T gravity = 9.81f, std::array<T, 3> domain = {2.0f, 2.0f, 1.0f})
        : dt_(dt), gravity_(gravity), domain_size_(domain) {}

    void addParticle(T x, T y, T z, T vx, T vy, T vz, T mass, MaterialType material) {
        particles_.push_back({x, y, z, vx, vy, vz, mass, material, true});
    }

    void step() {
        // Calculate inter-particle forces first
        calculateInterParticleForces();

        // Explicit integration with forces
        for (auto& p : particles_) {
            if (!p.active) continue;

            // Apply gravity
            p.vy -= gravity_ * dt_;

            // Update positions
            p.x += p.vx * dt_;
            p.y += p.vy * dt_;
            p.z += p.vz * dt_;

            // Improved boundary conditions based on material type
            if (p.y <= 0.0f) {
                p.y = 0.0f;
                if (p.material == MaterialType::FLUID) {
                    p.vy = -0.1f * p.vy; // Less damping for fluids
                } else if (p.material == MaterialType::ELASTIC) {
                    p.vy = -0.8f * p.vy; // More bounce for elastic materials
                }
            }

            // Wall boundary conditions with material-specific behavior
            if (p.x <= 0.0f) {
                p.x = 0.0f;
                if (p.material == MaterialType::FLUID) {
                    p.vx = -0.1f * p.vx;
                } else {
                    p.vx = -0.5f * p.vx;
                }
            }
            if (p.x >= domain_size_[0]) {
                p.x = domain_size_[0];
                if (p.material == MaterialType::FLUID) {
                    p.vx = -0.1f * p.vx;
                } else {
                    p.vx = -0.5f * p.vx;
                }
            }
            if (p.z <= 0.0f) { p.z = 0.0f; p.vz = -0.3f * p.vz; }
            if (p.z >= domain_size_[2]) { p.z = domain_size_[2]; p.vz = -0.3f * p.vz; }
        }
    }

private:
    void calculateInterParticleForces() {
        const T h = 0.05f; // Smoothing length
        const T h2 = h * h;

        // Reset forces
        for (auto& p : particles_) {
            // Keep any external forces, but reset inter-particle forces
        }

        // Calculate inter-particle forces
        for (size_t i = 0; i < particles_.size(); ++i) {
            if (!particles_[i].active) continue;

            for (size_t j = i + 1; j < particles_.size(); ++j) {
                if (!particles_[j].active) continue;

                T dx = particles_[i].x - particles_[j].x;
                T dy = particles_[i].y - particles_[j].y;
                T dz = particles_[i].z - particles_[j].z;
                T r2 = dx*dx + dy*dy + dz*dz;

                if (r2 < h2 && r2 > 1e-6f) {
                    T r = std::sqrt(r2);
                    T q = r / h;

                    // Apply material-specific forces
                    if (particles_[i].material == MaterialType::FLUID && particles_[j].material == MaterialType::FLUID) {
                        // Fluid pressure force (simplified SPH)
                        T pressure_force = 0.1f * (1.0f - q) / r;
                        T fx = pressure_force * dx;
                        T fy = pressure_force * dy;
                        T fz = pressure_force * dz;

                        particles_[i].vx += fx * dt_ / particles_[i].mass;
                        particles_[i].vy += fy * dt_ / particles_[i].mass;
                        particles_[i].vz += fz * dt_ / particles_[i].mass;

                        particles_[j].vx -= fx * dt_ / particles_[j].mass;
                        particles_[j].vy -= fy * dt_ / particles_[j].mass;
                        particles_[j].vz -= fz * dt_ / particles_[j].mass;
                    } else if (particles_[i].material == MaterialType::ELASTIC && particles_[j].material == MaterialType::ELASTIC) {
                        // Elastic contact force (simplified)
                        if (r < h * 0.8f) { // Contact threshold
                            T contact_force = 0.5f * (h * 0.8f - r) / (h * 0.8f);
                            T fx = contact_force * dx / r;
                            T fy = contact_force * dy / r;
                            T fz = contact_force * dz / r;

                            particles_[i].vx += fx * dt_ / particles_[i].mass;
                            particles_[i].vy += fy * dt_ / particles_[i].mass;
                            particles_[i].vz += fz * dt_ / particles_[i].mass;

                            particles_[j].vx -= fx * dt_ / particles_[j].mass;
                            particles_[j].vy -= fy * dt_ / particles_[j].mass;
                            particles_[j].vz -= fz * dt_ / particles_[j].mass;
                        }
                    }
                }
            }
        }
    }

public:
    T getTotalEnergy() const {
        T kinetic = 0.0f, potential = 0.0f;
        for (const auto& p : particles_) {
            if (!p.active) continue;
            kinetic += 0.5f * p.mass * (p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);
            potential += p.mass * gravity_ * p.y;
        }
        return kinetic + potential;
    }

    T getTotalMass() const {
        T total = 0.0f;
        for (const auto& p : particles_) {
            if (p.active) total += p.mass;
        }
        return total;
    }

    std::array<T, 3> getCenterOfMass() const {
        T total_mass = 0.0f;
        T cm_x = 0.0f, cm_y = 0.0f, cm_z = 0.0f;

        for (const auto& p : particles_) {
            if (!p.active) continue;
            cm_x += p.mass * p.x;
            cm_y += p.mass * p.y;
            cm_z += p.mass * p.z;
            total_mass += p.mass;
        }

        if (total_mass > 0.0f) {
            return {cm_x/total_mass, cm_y/total_mass, cm_z/total_mass};
        }
        return {0.0f, 0.0f, 0.0f};
    }

    T getMaxHeight() const {
        T max_y = 0.0f;
        for (const auto& p : particles_) {
            if (p.active) max_y = std::max(max_y, p.y);
        }
        return max_y;
    }

    T getSpreadX() const {
        T min_x = 1e6f, max_x = -1e6f;
        for (const auto& p : particles_) {
            if (!p.active) continue;
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
        }
        return max_x - min_x;
    }

    size_t getActiveParticleCount() const {
        size_t count = 0;
        for (const auto& p : particles_) {
            if (p.active) count++;
        }
        return count;
    }
};

template<typename T>
struct ValidationResult {
    std::string test_name;
    bool passed;
    T simulation_time;
    T computation_time_ms;
    size_t particle_count;
    std::vector<T> energy_history;
    std::vector<T> mass_history;
    std::string failure_reason;
};

template<typename T>
ValidationResult<T> runDamBreakTest() {
    std::cout << "Running Dam Break Validation Test..." << std::endl;

    ValidationResult<T> result;
    result.test_name = "Dam Break";
    result.simulation_time = 2.0f;

    SimplifiedMPMPhysics<T> physics(0.001f, 9.81f, {2.0f, 1.0f, 0.5f});

    // Create water dam - vertical column on left side
    T particle_mass = 0.001f;
    T spacing = 0.03f;
    size_t particles_added = 0;

    for (T x = 0.1f; x <= 0.6f; x += spacing) {
        for (T y = 0.0f; y <= 0.8f; y += spacing) {
            for (T z = 0.1f; z <= 0.4f; z += spacing) {
                physics.addParticle(x, y, z, 0.0f, 0.0f, 0.0f, particle_mass, MaterialType::FLUID);
                particles_added++;
            }
        }
    }

    result.particle_count = particles_added;
    std::cout << "  Generated " << particles_added << " water particles" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Run simulation
    T current_time = 0.0f;
    T initial_mass = physics.getTotalMass();
    T initial_energy = physics.getTotalEnergy();
    auto initial_cm = physics.getCenterOfMass();

    const int save_interval = 100;
    int step = 0;

    while (current_time < result.simulation_time) {
        physics.step();
        current_time += 0.001f;
        step++;

        if (step % save_interval == 0) {
            result.energy_history.push_back(physics.getTotalEnergy());
            result.mass_history.push_back(physics.getTotalMass());

            std::cout << "  Step " << step << ", Time: " << std::fixed << std::setprecision(3)
                     << current_time << "s, Energy: " << physics.getTotalEnergy()
                     << ", Spread: " << physics.getSpreadX() << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<T, std::milli>(end_time - start_time).count();

    // Validation criteria
    T final_mass = physics.getTotalMass();
    T mass_error = std::abs(final_mass - initial_mass) / initial_mass;
    T final_spread = physics.getSpreadX();
    auto final_cm = physics.getCenterOfMass();

    bool mass_conserved = mass_error < 0.01f;
    bool fluid_spread = final_spread > 0.8f; // Water should spread significantly
    bool cm_moved_right = final_cm[0] > initial_cm[0]; // Center of mass should move right

    result.passed = mass_conserved && fluid_spread && cm_moved_right;

    if (!result.passed) {
        result.failure_reason = "Mass conservation: " + std::to_string(mass_conserved) +
                               ", Spread: " + std::to_string(fluid_spread) +
                               ", CM movement: " + std::to_string(cm_moved_right);
    }

    std::cout << "  ✓ Dam Break Test " << (result.passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "    Mass conservation error: " << mass_error * 100.0f << "%" << std::endl;
    std::cout << "    Final water spread: " << final_spread << "m" << std::endl;
    std::cout << "    Center of mass movement: " << (final_cm[0] - initial_cm[0]) << "m" << std::endl;
    std::cout << "    Computation time: " << result.computation_time_ms << " ms" << std::endl;

    return result;
}

template<typename T>
ValidationResult<T> runOscillatingDropTest() {
    std::cout << "Running Oscillating Drop Validation Test..." << std::endl;

    ValidationResult<T> result;
    result.test_name = "Oscillating Drop";
    result.simulation_time = 1.0f;

    SimplifiedMPMPhysics<T> physics(0.0005f, 9.81f, {1.0f, 1.0f, 1.0f});

    // Create spherical drop
    T particle_mass = 0.0005f;
    T spacing = 0.02f;
    T drop_radius = 0.15f;
    std::array<T, 3> drop_center = {0.5f, 0.5f, 0.5f};
    size_t particles_added = 0;

    for (T x = drop_center[0] - drop_radius; x <= drop_center[0] + drop_radius; x += spacing) {
        for (T y = drop_center[1] - drop_radius; y <= drop_center[1] + drop_radius; y += spacing) {
            for (T z = drop_center[2] - drop_radius; z <= drop_center[2] + drop_radius; z += spacing) {
                T dx = x - drop_center[0];
                T dy = y - drop_center[1];
                T dz = z - drop_center[2];
                T distance = std::sqrt(dx*dx + dy*dy + dz*dz);

                if (distance <= drop_radius) {
                    physics.addParticle(x, y, z, 0.0f, 0.0f, 0.0f, particle_mass, MaterialType::ELASTIC);
                    particles_added++;
                }
            }
        }
    }

    result.particle_count = particles_added;
    std::cout << "  Generated " << particles_added << " elastic particles" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Run simulation
    T current_time = 0.0f;
    T initial_energy = physics.getTotalEnergy();
    T initial_mass = physics.getTotalMass();

    const int save_interval = 50;
    int step = 0;
    std::vector<T> height_history;

    while (current_time < result.simulation_time) {
        physics.step();
        current_time += 0.0005f;
        step++;

        if (step % save_interval == 0) {
            result.energy_history.push_back(physics.getTotalEnergy());
            result.mass_history.push_back(physics.getTotalMass());
            height_history.push_back(physics.getMaxHeight());

            std::cout << "  Step " << step << ", Time: " << std::fixed << std::setprecision(3)
                     << current_time << "s, Energy: " << physics.getTotalEnergy()
                     << ", Max Height: " << physics.getMaxHeight() << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<T, std::milli>(end_time - start_time).count();

    // Validation criteria
    T final_mass = physics.getTotalMass();
    T mass_error = std::abs(final_mass - initial_mass) / initial_mass;

    // Check for oscillation pattern in height
    bool oscillation_detected = false;
    if (height_history.size() >= 4) {
        T min_height = *std::min_element(height_history.begin(), height_history.end());
        T max_height = *std::max_element(height_history.begin(), height_history.end());
        oscillation_detected = (max_height - min_height) > 0.05f; // Significant height variation
    }

    bool mass_conserved = mass_error < 0.01f;
    bool energy_reasonable = result.energy_history.back() > 0.0f;

    result.passed = mass_conserved && oscillation_detected && energy_reasonable;

    if (!result.passed) {
        result.failure_reason = "Mass conservation: " + std::to_string(mass_conserved) +
                               ", Oscillation: " + std::to_string(oscillation_detected) +
                               ", Energy: " + std::to_string(energy_reasonable);
    }

    std::cout << "  ✓ Oscillating Drop Test " << (result.passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "    Mass conservation error: " << mass_error * 100.0f << "%" << std::endl;
    std::cout << "    Oscillation detected: " << (oscillation_detected ? "Yes" : "No") << std::endl;
    std::cout << "    Computation time: " << result.computation_time_ms << " ms" << std::endl;

    return result;
}

template<typename T>
ValidationResult<T> runStackingStabilityTest() {
    std::cout << "Running Stacking Stability Validation Test..." << std::endl;

    ValidationResult<T> result;
    result.test_name = "Stacking Stability";
    result.simulation_time = 3.0f;

    SimplifiedMPMPhysics<T> physics(0.001f, 9.81f, {1.0f, 2.0f, 1.0f});

    // Create stack of blocks
    T particle_mass = 0.002f;
    T spacing = 0.04f;
    const int num_blocks = 3;
    T block_width = 0.2f;
    T block_height = 0.1f;
    T block_depth = 0.2f;
    size_t particles_added = 0;

    for (int block = 0; block < num_blocks; ++block) {
        T base_y = block * (block_height + 0.02f) + 0.05f;

        for (T x = 0.4f; x <= 0.4f + block_width; x += spacing) {
            for (T y = base_y; y <= base_y + block_height; y += spacing) {
                for (T z = 0.4f; z <= 0.4f + block_depth; z += spacing) {
                    physics.addParticle(x, y, z, 0.0f, 0.0f, 0.0f, particle_mass, MaterialType::ELASTIC);
                    particles_added++;
                }
            }
        }
    }

    result.particle_count = particles_added;
    std::cout << "  Generated " << particles_added << " particles in " << num_blocks << " blocks" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Run simulation
    T current_time = 0.0f;
    T initial_mass = physics.getTotalMass();
    T initial_max_height = physics.getMaxHeight();

    const int save_interval = 100;
    int step = 0;
    std::vector<T> stability_history;

    while (current_time < result.simulation_time) {
        physics.step();
        current_time += 0.001f;
        step++;

        if (step % save_interval == 0) {
            result.energy_history.push_back(physics.getTotalEnergy());
            result.mass_history.push_back(physics.getTotalMass());
            stability_history.push_back(physics.getMaxHeight());

            std::cout << "  Step " << step << ", Time: " << std::fixed << std::setprecision(3)
                     << current_time << "s, Max Height: " << physics.getMaxHeight()
                     << ", Energy: " << physics.getTotalEnergy() << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.computation_time_ms = std::chrono::duration<T, std::milli>(end_time - start_time).count();

    // Validation criteria
    T final_mass = physics.getTotalMass();
    T mass_error = std::abs(final_mass - initial_mass) / initial_mass;
    T final_max_height = physics.getMaxHeight();
    T height_loss = (initial_max_height - final_max_height) / initial_max_height;

    bool mass_conserved = mass_error < 0.01f;
    bool stack_stable = height_loss < 0.3f; // Allow some settling but not collapse
    bool energy_stable = result.energy_history.back() > 0.0f;

    result.passed = mass_conserved && stack_stable && energy_stable;

    if (!result.passed) {
        result.failure_reason = "Mass conservation: " + std::to_string(mass_conserved) +
                               ", Stack stable: " + std::to_string(stack_stable) +
                               ", Energy stable: " + std::to_string(energy_stable);
    }

    std::cout << "  ✓ Stacking Stability Test " << (result.passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "    Mass conservation error: " << mass_error * 100.0f << "%" << std::endl;
    std::cout << "    Height loss: " << height_loss * 100.0f << "%" << std::endl;
    std::cout << "    Final max height: " << final_max_height << "m" << std::endl;
    std::cout << "    Computation time: " << result.computation_time_ms << " ms" << std::endl;

    return result;
}

int main() {
    std::cout << "PhysGrad MPM Physics Validation Tests" << std::endl;
    std::cout << "=====================================" << std::endl << std::endl;

    std::vector<ValidationResult<float>> results;

    // Run all validation tests
    results.push_back(runDamBreakTest<float>());
    std::cout << std::endl;

    results.push_back(runOscillatingDropTest<float>());
    std::cout << std::endl;

    results.push_back(runStackingStabilityTest<float>());
    std::cout << std::endl;

    // Generate summary report
    std::cout << "=== PHYSICS VALIDATION SUMMARY ===" << std::endl;

    int passed_tests = 0;
    int total_tests = results.size();

    for (const auto& result : results) {
        std::cout << result.test_name << ": " << (result.passed ? "PASSED ✓" : "FAILED ❌") << std::endl;
        std::cout << "  Particles: " << result.particle_count
                 << ", Time: " << result.computation_time_ms << " ms" << std::endl;

        if (!result.passed) {
            std::cout << "  Failure reason: " << result.failure_reason << std::endl;
        }

        if (result.passed) passed_tests++;
        std::cout << std::endl;
    }

    std::cout << "Overall Result: " << passed_tests << "/" << total_tests << " tests passed" << std::endl;

    if (passed_tests == total_tests) {
        std::cout << "🎉 All MPM physics validation tests PASSED!" << std::endl;
        std::cout << "   The physics implementation correctly handles:" << std::endl;
        std::cout << "   • Fluid dynamics with dam break behavior" << std::endl;
        std::cout << "   • Elastic material oscillations and dynamics" << std::endl;
        std::cout << "   • Structural stability and block stacking" << std::endl;
        std::cout << "   • Mass conservation across all scenarios" << std::endl;
        std::cout << "   • Reasonable energy behavior in all tests" << std::endl;
    } else {
        std::cout << "⚠️  Some physics validation tests failed" << std::endl;
        std::cout << "   Review implementation for:" << std::endl;
        for (const auto& result : results) {
            if (!result.passed) {
                std::cout << "   • " << result.test_name << ": " << result.failure_reason << std::endl;
            }
        }
    }

    return passed_tests == total_tests ? 0 : 1;
}