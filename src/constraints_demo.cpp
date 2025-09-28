#include "constraints.h"
#include "collision_detection.h"
#include "visualization.h"
#include "adaptive_integration.h"
#include "config_system.h"
#include "logging_system.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <random>

using namespace physgrad;

struct ConstraintDemoSystem {
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> vel_x, vel_y, vel_z;
    std::vector<float> masses;
    std::vector<float> force_x, force_y, force_z;

    ConstraintSolver constraint_solver;
    CollisionDetector collision_detector;

    int scenario_type = 0;  // 0=pendulum, 1=chain, 2=rope bridge, 3=cloth

    void clear() {
        pos_x.clear(); pos_y.clear(); pos_z.clear();
        vel_x.clear(); vel_y.clear(); vel_z.clear();
        masses.clear();
        force_x.clear(); force_y.clear(); force_z.clear();
        constraint_solver.clearConstraints();
    }

    void resize(int n) {
        pos_x.resize(n); pos_y.resize(n); pos_z.resize(n);
        vel_x.resize(n); vel_y.resize(n); vel_z.resize(n);
        masses.resize(n);
        force_x.resize(n); force_y.resize(n); force_z.resize(n);
    }

    void createPendulum() {
        clear();
        scenario_type = 0;

        // Create a simple pendulum: fixed anchor + swinging mass
        resize(2);

        // Anchor point (fixed)
        pos_x[0] = 0.0f; pos_y[0] = 3.0f; pos_z[0] = 0.0f;
        vel_x[0] = 0.0f; vel_y[0] = 0.0f; vel_z[0] = 0.0f;
        masses[0] = 1000.0f;  // Very heavy to act as anchor

        // Pendulum mass
        pos_x[1] = 1.5f; pos_y[1] = 1.0f; pos_z[1] = 0.0f;
        vel_x[1] = 0.0f; vel_y[1] = 0.0f; vel_z[1] = 0.0f;
        masses[1] = 1.0f;

        // Add constraints
        constraint_solver.addPositionLock(0, 0.0f, 3.0f, 0.0f);  // Fix anchor
        constraint_solver.addDistanceConstraint(0, 1, 2.5f);     // Rigid rod

        Logger::getInstance().info("demo", "Created pendulum scenario");
    }

    void createChain() {
        clear();
        scenario_type = 1;

        const int chain_length = 8;
        resize(chain_length);

        // Create vertical chain
        for (int i = 0; i < chain_length; ++i) {
            pos_x[i] = 0.0f;
            pos_y[i] = 3.0f - i * 0.5f;
            pos_z[i] = 0.0f;

            vel_x[i] = 0.0f;
            vel_y[i] = 0.0f;
            vel_z[i] = 0.0f;

            masses[i] = (i == 0) ? 1000.0f : 0.5f;  // Heavy anchor, light chain
        }

        // Add constraints
        constraint_solver.addPositionLock(0, 0.0f, 3.0f, 0.0f);  // Fix top

        ConstraintParams chain_params;
        chain_params.stiffness = 800.0f;
        chain_params.damping = 2.0f;

        for (int i = 0; i < chain_length - 1; ++i) {
            constraint_solver.addDistanceConstraint(i, i + 1, 0.5f, chain_params);
        }

        Logger::getInstance().info("demo", "Created chain scenario");
    }

    void createRopeBridge() {
        clear();
        scenario_type = 2;

        const int bridge_segments = 10;
        resize(bridge_segments);

        // Create horizontal rope bridge
        for (int i = 0; i < bridge_segments; ++i) {
            pos_x[i] = -2.0f + i * (4.0f / (bridge_segments - 1));
            pos_y[i] = 2.0f;
            pos_z[i] = 0.0f;

            vel_x[i] = 0.0f;
            vel_y[i] = 0.0f;
            vel_z[i] = 0.0f;

            masses[i] = (i == 0 || i == bridge_segments - 1) ? 1000.0f : 0.3f;
        }

        // Fix both ends
        constraint_solver.addPositionLock(0, -2.0f, 2.0f, 0.0f);
        constraint_solver.addPositionLock(bridge_segments - 1, 2.0f, 2.0f, 0.0f);

        // Connect segments with springs (more flexible than rigid constraints)
        ConstraintParams bridge_params;
        bridge_params.stiffness = 300.0f;
        bridge_params.damping = 5.0f;

        for (int i = 0; i < bridge_segments - 1; ++i) {
            float segment_length = 4.0f / (bridge_segments - 1);
            constraint_solver.addSpringConstraint(i, i + 1, segment_length, 300.0f, bridge_params);
        }

        Logger::getInstance().info("demo", "Created rope bridge scenario");
    }

    void createCloth() {
        clear();
        scenario_type = 3;

        const int cloth_width = 6;
        const int cloth_height = 6;
        const float spacing = 0.3f;

        resize(cloth_width * cloth_height);

        // Create cloth grid
        for (int y = 0; y < cloth_height; ++y) {
            for (int x = 0; x < cloth_width; ++x) {
                int idx = y * cloth_width + x;

                pos_x[idx] = -0.8f + x * spacing;
                pos_y[idx] = 2.5f - y * spacing;
                pos_z[idx] = 0.0f;

                vel_x[idx] = 0.0f;
                vel_y[idx] = 0.0f;
                vel_z[idx] = 0.0f;

                masses[idx] = 0.1f;
            }
        }

        // Fix top corners
        constraint_solver.addPositionLock(0, pos_x[0], pos_y[0], pos_z[0]);  // Top-left
        constraint_solver.addPositionLock(cloth_width - 1, pos_x[cloth_width - 1], pos_y[cloth_width - 1], pos_z[cloth_width - 1]);  // Top-right

        // Add structural constraints (horizontal and vertical)
        ConstraintParams cloth_params;
        cloth_params.stiffness = 400.0f;
        cloth_params.damping = 1.0f;

        for (int y = 0; y < cloth_height; ++y) {
            for (int x = 0; x < cloth_width; ++x) {
                int idx = y * cloth_width + x;

                // Horizontal connections
                if (x < cloth_width - 1) {
                    constraint_solver.addSpringConstraint(idx, idx + 1, spacing, 400.0f, cloth_params);
                }

                // Vertical connections
                if (y < cloth_height - 1) {
                    constraint_solver.addSpringConstraint(idx, idx + cloth_width, spacing, 400.0f, cloth_params);
                }

                // Diagonal connections for stability
                if (x < cloth_width - 1 && y < cloth_height - 1) {
                    float diag_length = spacing * std::sqrt(2.0f);
                    cloth_params.stiffness = 200.0f;  // Weaker diagonal springs
                    constraint_solver.addSpringConstraint(idx, idx + cloth_width + 1, diag_length, 200.0f, cloth_params);
                    constraint_solver.addSpringConstraint(idx + 1, idx + cloth_width, diag_length, 200.0f, cloth_params);
                }
            }
        }

        Logger::getInstance().info("demo", "Created cloth scenario");
    }

    void addRandomForceToCenter() {
        if (pos_x.empty()) return;

        // Find center particle
        float center_x = 0.0f, center_y = 0.0f;
        for (size_t i = 0; i < pos_x.size(); ++i) {
            center_x += pos_x[i];
            center_y += pos_y[i];
        }
        center_x /= pos_x.size();
        center_y /= pos_x.size();

        // Find closest particle to center
        int center_idx = 0;
        float min_dist = 1e6f;
        for (size_t i = 0; i < pos_x.size(); ++i) {
            float dx = pos_x[i] - center_x;
            float dy = pos_y[i] - center_y;
            float dist = dx*dx + dy*dy;
            if (dist < min_dist) {
                min_dist = dist;
                center_idx = i;
            }
        }

        // Apply random impulse
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> force_dist(-5.0f, 5.0f);

        vel_x[center_idx] += force_dist(gen);
        vel_y[center_idx] += force_dist(gen);
        vel_z[center_idx] += force_dist(gen);
    }
};

int main() {
    Logger::getInstance().info("demo", "Starting PhysGrad Constraint-Based Physics Demo");

    // Load configuration
    ConfigManager config;
    if (!config.loadFromFile("config.conf")) {
        Logger::getInstance().warning("demo", "Could not load config.conf, using defaults");
    }
    auto sim_params = config.getSimulationParams();

    // Initialize visualization
    VisualizationManager viz_manager;
    if (!viz_manager.initialize(1280, 720)) {
        Logger::getInstance().error("demo", "Failed to initialize visualization");
        return -1;
    }

    // Initialize system
    ConstraintDemoSystem system;
    system.createPendulum();  // Start with pendulum

    AdaptiveIntegrator integrator(IntegrationScheme::ADAPTIVE_RK45);

    auto last_time = std::chrono::high_resolution_clock::now();
    int last_scenario = -1;

    while (!viz_manager.shouldClose()) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto delta_time = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;

        auto& params = viz_manager.getInteractiveParams();

        if (viz_manager.isSimulationRunning()) {
            // Reset forces
            std::fill(system.force_x.begin(), system.force_x.end(), 0.0f);
            std::fill(system.force_y.begin(), system.force_y.end(), 0.0f);
            std::fill(system.force_z.begin(), system.force_z.end(), 0.0f);

            // Apply gravity
            if (params.enable_gravity) {
                for (size_t i = 0; i < system.masses.size(); ++i) {
                    if (system.masses[i] < 100.0f) {  // Don't apply gravity to anchors
                        system.force_y[i] -= system.masses[i] * params.gravity_strength;
                    }
                }
            }

            // Apply constraint forces
            system.constraint_solver.applyConstraintForces(
                system.pos_x, system.pos_y, system.pos_z,
                system.vel_x, system.vel_y, system.vel_z,
                system.force_x, system.force_y, system.force_z,
                system.masses
            );

            // Integrate motion
            integrator.integrateStep(
                system.pos_x, system.pos_y, system.pos_z,
                system.vel_x, system.vel_y, system.vel_z,
                system.masses,
                sim_params.time_step * viz_manager.getTimeScale() * 0.5f,
                0.0f, 0.001f
            );

            // Solve constraints (position correction)
            system.constraint_solver.solveConstraints(
                system.pos_x, system.pos_y, system.pos_z,
                system.vel_x, system.vel_y, system.vel_z,
                system.masses, sim_params.time_step
            );

            // Apply air damping
            for (size_t i = 0; i < system.vel_x.size(); ++i) {
                if (system.masses[i] < 100.0f) {  // Don't damp anchors
                    system.vel_x[i] *= params.air_damping;
                    system.vel_y[i] *= params.air_damping;
                    system.vel_z[i] *= params.air_damping;
                }
            }

            // Calculate energy
            float kinetic_energy = 0.0f;
            float potential_energy = 0.0f;

            for (size_t i = 0; i < system.masses.size(); ++i) {
                if (system.masses[i] < 100.0f) {  // Only count non-anchor particles
                    float v_squared = system.vel_x[i] * system.vel_x[i] +
                                     system.vel_y[i] * system.vel_y[i] +
                                     system.vel_z[i] * system.vel_z[i];
                    kinetic_energy += 0.5f * system.masses[i] * v_squared;

                    if (params.enable_gravity) {
                        potential_energy += system.masses[i] * params.gravity_strength *
                                          (system.pos_y[i] + 5.0f);  // Reference height
                    }
                }
            }

            // Update visualization
            viz_manager.updateFromSimulation(system.pos_x, system.pos_y, system.pos_z,
                                           system.vel_x, system.vel_y, system.vel_z,
                                           system.masses);

            if (params.show_force_vectors) {
                viz_manager.updateForces(system.force_x, system.force_y, system.force_z);
            }

            viz_manager.updateEnergy(kinetic_energy, potential_energy);

            // Update constraint solver stats
            viz_manager.setCollisionStats(
                0,  // No collision detection in this demo
                0,
                static_cast<int>(system.constraint_solver.getConstraints().size())
            );
        }

        if (viz_manager.shouldSingleStep()) {
            viz_manager.resetSingleStep();
        }

        viz_manager.render();

        // Add custom UI for constraint scenarios
        ImGui::Begin("Constraint Scenarios");

        ImGui::Text("Select Scenario:");
        static int selected_scenario = 0;
        const char* scenario_names[] = {"Pendulum", "Chain", "Rope Bridge", "Cloth"};

        if (ImGui::Combo("Scenario", &selected_scenario, scenario_names, 4)) {
            if (selected_scenario != last_scenario) {
                switch (selected_scenario) {
                    case 0: system.createPendulum(); break;
                    case 1: system.createChain(); break;
                    case 2: system.createRopeBridge(); break;
                    case 3: system.createCloth(); break;
                }
                last_scenario = selected_scenario;
            }
        }

        ImGui::Separator();
        ImGui::Text("Constraint Info:");
        ImGui::Text("Active Constraints: %zu", system.constraint_solver.getConstraints().size());
        ImGui::Text("Solver Iterations: %d", system.constraint_solver.getLastIterations());
        ImGui::Text("Constraint Error: %.6f", system.constraint_solver.getLastResidual());
        ImGui::Text("Converged: %s", system.constraint_solver.hasConverged() ? "Yes" : "No");

        ImGui::Separator();
        if (ImGui::Button("Apply Random Force")) {
            system.addRandomForceToCenter();
        }

        if (ImGui::Button("Reset Current Scenario")) {
            switch (selected_scenario) {
                case 0: system.createPendulum(); break;
                case 1: system.createChain(); break;
                case 2: system.createRopeBridge(); break;
                case 3: system.createCloth(); break;
            }
        }

        ImGui::End();

        // Limit frame rate
        if (delta_time < 0.016f) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16 - static_cast<int>(delta_time * 1000)));
        }
    }

    viz_manager.shutdown();
    Logger::getInstance().info("demo", "Constraint-based physics demo completed");
    return 0;
}