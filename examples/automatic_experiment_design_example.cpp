/**
 * PhysGrad Automatic Experiment Design - Practical Physics Example
 *
 * Demonstrates AI-driven parameter optimization for real physics problems:
 * 1. Projectile motion optimization
 * 2. Oscillator damping design
 * 3. Fluid dynamics parameter tuning
 */

#include "../src/automatic_experiment_design.h"
#include "../src/experiment_designer.h"
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace physgrad::experiment_design;

// =============================================================================
// PHYSICS SIMULATION EXAMPLES
// =============================================================================

// Example 1: Projectile Motion Optimization
class ProjectileOptimizer {
private:
    static constexpr float GRAVITY = 9.81f;
    static constexpr float AIR_DENSITY = 1.225f;

public:
    // Objective: Maximize range while minimizing energy cost
    static float evaluateProjectile(const std::vector<float>& params) {
        if (params.size() < 3) return -1000;

        float initial_velocity = params[0];  // m/s
        float launch_angle = params[1];      // radians
        float projectile_mass = params[2];   // kg

        // Basic projectile motion with air resistance
        float vx = initial_velocity * std::cos(launch_angle);
        float vy = initial_velocity * std::sin(launch_angle);

        float x = 0, y = 0;
        float dt = 0.01f;
        float drag_coefficient = 0.47f;  // Sphere
        float projectile_radius = 0.05f; // 5cm radius
        float area = M_PI * projectile_radius * projectile_radius;

        while (y >= 0 && x < 10000) {  // Safety limits
            // Air resistance
            float velocity_magnitude = std::sqrt(vx*vx + vy*vy);
            float drag_force = 0.5f * AIR_DENSITY * drag_coefficient * area * velocity_magnitude * velocity_magnitude;

            float ax = -drag_force * vx / (projectile_mass * velocity_magnitude);
            float ay = -GRAVITY - drag_force * vy / (projectile_mass * velocity_magnitude);

            vx += ax * dt;
            vy += ay * dt;
            x += vx * dt;
            y += vy * dt;
        }

        // Multi-objective: maximize range, minimize energy cost
        float kinetic_energy = 0.5f * projectile_mass * initial_velocity * initial_velocity;
        float energy_efficiency = x / kinetic_energy;  // range per unit energy

        return energy_efficiency;
    }

    static void runOptimization() {
        std::cout << "=== Projectile Motion Optimization ===" << std::endl;
        std::cout << "Finding optimal launch parameters for maximum range efficiency" << std::endl;

        ParameterSpace<float> space;

        // Initial velocity parameter
        Parameter<float> velocity_param;
        velocity_param.name = "initial_velocity";
        velocity_param.type = ParameterType::CONTINUOUS;
        velocity_param.min_value = 10.0f;
        velocity_param.max_value = 100.0f;
        velocity_param.default_value = 45.0f;
        velocity_param.distribution = DistributionType::UNIFORM;
        velocity_param.description = "Initial projectile velocity";
        velocity_param.units = "m/s";
        space.addParameter(velocity_param);

        // Launch angle parameter
        Parameter<float> angle_param;
        angle_param.name = "launch_angle";
        angle_param.type = ParameterType::CONTINUOUS;
        angle_param.min_value = 0.1f;
        angle_param.max_value = 1.4f;  // ~8 to 80 degrees
        angle_param.default_value = 0.785f;  // 45 degrees
        angle_param.description = "Launch angle";
        angle_param.units = "radians";
        space.addParameter(angle_param);

        // Projectile mass parameter
        Parameter<float> mass_param;
        mass_param.name = "projectile_mass";
        mass_param.type = ParameterType::CONTINUOUS;
        mass_param.min_value = 0.1f;
        mass_param.max_value = 10.0f;
        mass_param.default_value = 1.0f;
        mass_param.distribution = DistributionType::LOG_UNIFORM;
        mass_param.description = "Projectile mass";
        mass_param.units = "kg";
        space.addParameter(mass_param);

        // Create objective
        Objective<float> objective;
        objective.name = "energy_efficiency";
        objective.type = ObjectiveType::MAXIMIZE;
        objective.weight = 1.0f;
        objective.evaluation_function = evaluateProjectile;

        // Create and configure experiment
        ExperimentDesigner<float> designer;
        designer.setParameterSpace(space);
        designer.addObjective(objective);

        ExperimentConfig config;
        config.max_evaluations = 150;
        config.parallel_evaluations = 4;
        config.use_bayesian_optimization = true;
        config.convergence_tolerance = 1e-4;
        config.experiment_name = "projectile_optimization";
        designer.setConfig(config);

        // Run optimization
        designer.run();

        // Display results
        auto best = designer.getBestResult();
        std::cout << "\nOptimal Projectile Parameters:" << std::endl;
        std::cout << "  Initial Velocity: " << std::fixed << std::setprecision(2)
                  << best.parameter_values[0] << " m/s" << std::endl;
        std::cout << "  Launch Angle: " << best.parameter_values[1] * 180 / M_PI << " degrees" << std::endl;
        std::cout << "  Mass: " << best.parameter_values[2] << " kg" << std::endl;
        std::cout << "  Energy Efficiency: " << best.objective_values[0] << " m/J" << std::endl;
        std::cout << "  Evaluations: " << designer.getEvaluationsCompleted() << std::endl;

        // Calculate actual range for the optimal parameters
        float range = calculateRange(best.parameter_values);
        std::cout << "  Predicted Range: " << range << " m" << std::endl;
    }

private:
    static float calculateRange(const std::vector<float>& params) {
        float initial_velocity = params[0];
        float launch_angle = params[1];

        // Simple ballistic formula (no air resistance)
        return initial_velocity * initial_velocity * std::sin(2 * launch_angle) / GRAVITY;
    }
};

// Example 2: Damped Oscillator Design
class OscillatorDesigner {
public:
    // Objective: Design oscillator with specific response characteristics
    static float evaluateOscillator(const std::vector<float>& params) {
        if (params.size() < 3) return -1000;

        float mass = params[0];           // kg
        float spring_constant = params[1]; // N/m
        float damping = params[2];        // Ns/m

        // Calculate oscillator characteristics
        float omega_n = std::sqrt(spring_constant / mass);  // Natural frequency
        float zeta = damping / (2 * std::sqrt(mass * spring_constant));  // Damping ratio

        // Design goals:
        // 1. Natural frequency around 10 Hz
        // 2. Damping ratio around 0.7 (slightly underdamped)
        // 3. Minimize settling time

        float freq_hz = omega_n / (2 * M_PI);
        float freq_error = std::abs(freq_hz - 10.0f);
        float zeta_error = std::abs(zeta - 0.7f);

        // Settling time for 2% criterion
        float settling_time = (zeta < 1.0f) ? 4.0f / (zeta * omega_n) : 4.0f / omega_n;

        // Composite objective (minimize)
        float total_error = freq_error + 10.0f * zeta_error + 0.1f * settling_time;

        return -total_error;  // Negative for minimization
    }

    static void runOptimization() {
        std::cout << "\n=== Oscillator Design Optimization ===" << std::endl;
        std::cout << "Designing oscillator for 10 Hz, 0.7 damping ratio" << std::endl;

        ParameterSpace<float> space;

        // Mass parameter
        Parameter<float> mass_param;
        mass_param.name = "mass";
        mass_param.type = ParameterType::CONTINUOUS;
        mass_param.min_value = 0.01f;
        mass_param.max_value = 10.0f;
        mass_param.distribution = DistributionType::LOG_UNIFORM;
        mass_param.description = "Oscillator mass";
        mass_param.units = "kg";
        space.addParameter(mass_param);

        // Spring constant parameter
        Parameter<float> spring_param;
        spring_param.name = "spring_constant";
        spring_param.type = ParameterType::CONTINUOUS;
        spring_param.min_value = 1.0f;
        spring_param.max_value = 10000.0f;
        spring_param.distribution = DistributionType::LOG_UNIFORM;
        spring_param.description = "Spring constant";
        spring_param.units = "N/m";
        space.addParameter(spring_param);

        // Damping parameter
        Parameter<float> damping_param;
        damping_param.name = "damping";
        damping_param.type = ParameterType::CONTINUOUS;
        damping_param.min_value = 0.1f;
        damping_param.max_value = 100.0f;
        damping_param.distribution = DistributionType::LOG_UNIFORM;
        damping_param.description = "Damping coefficient";
        damping_param.units = "Ns/m";
        space.addParameter(damping_param);

        // Add constraint for physical realizability
        space.addGlobalConstraint([](const std::vector<float>& params) -> bool {
            float mass = params[0];
            float spring_k = params[1];
            float damping = params[2];

            // Ensure critically damped or underdamped (zeta <= 1)
            float zeta = damping / (2 * std::sqrt(mass * spring_k));
            return zeta <= 1.5f;  // Allow slightly overdamped
        });

        // Create objective
        Objective<float> objective;
        objective.name = "design_error";
        objective.type = ObjectiveType::MAXIMIZE;  // Maximizing negative error
        objective.evaluation_function = evaluateOscillator;

        // Create experiment
        ExperimentDesigner<float> designer;
        designer.setParameterSpace(space);
        designer.addObjective(objective);

        ExperimentConfig config;
        config.max_evaluations = 100;
        config.parallel_evaluations = 3;
        config.use_bayesian_optimization = true;
        config.enable_constraint_handling = true;
        config.experiment_name = "oscillator_design";
        designer.setConfig(config);

        // Run optimization
        designer.run();

        // Display results
        auto best = designer.getBestResult();
        std::cout << "\nOptimal Oscillator Design:" << std::endl;
        std::cout << "  Mass: " << std::scientific << std::setprecision(3)
                  << best.parameter_values[0] << " kg" << std::endl;
        std::cout << "  Spring Constant: " << best.parameter_values[1] << " N/m" << std::endl;
        std::cout << "  Damping: " << best.parameter_values[2] << " Ns/m" << std::endl;

        // Calculate achieved characteristics
        float mass = best.parameter_values[0];
        float spring_k = best.parameter_values[1];
        float damping = best.parameter_values[2];

        float omega_n = std::sqrt(spring_k / mass);
        float freq_hz = omega_n / (2 * M_PI);
        float zeta = damping / (2 * std::sqrt(mass * spring_k));
        float settling_time = 4.0f / (zeta * omega_n);

        std::cout << "\nAchieved Characteristics:" << std::endl;
        std::cout << "  Natural Frequency: " << std::fixed << std::setprecision(2)
                  << freq_hz << " Hz (target: 10 Hz)" << std::endl;
        std::cout << "  Damping Ratio: " << zeta << " (target: 0.7)" << std::endl;
        std::cout << "  Settling Time: " << settling_time << " s" << std::endl;
    }
};

// Example 3: Fluid Parameter Tuning
class FluidTuner {
public:
    // Objective: Optimize fluid simulation parameters for stability and accuracy
    static float evaluateFluidSim(const std::vector<float>& params) {
        if (params.size() < 4) return -1000;

        float density = params[0];        // kg/m³
        float viscosity = params[1];      // Pa·s
        float timestep = params[2];       // s
        float smoothing_length = params[3]; // m

        // Reynolds number
        float characteristic_velocity = 1.0f;  // m/s
        float characteristic_length = 0.1f;    // m
        float reynolds = density * characteristic_velocity * characteristic_length / viscosity;

        // CFL condition for stability
        float cfl_number = characteristic_velocity * timestep / smoothing_length;

        // Viscous stability condition
        float viscous_number = viscosity * timestep / (density * smoothing_length * smoothing_length);

        // Multi-criteria evaluation
        float stability_score = 0;

        // CFL condition (should be < 1)
        if (cfl_number <= 1.0f) {
            stability_score += 100 * (1.0f - cfl_number);
        } else {
            stability_score -= 1000 * (cfl_number - 1.0f);  // Heavy penalty
        }

        // Viscous condition (should be < 0.1)
        if (viscous_number <= 0.1f) {
            stability_score += 50 * (0.1f - viscous_number);
        } else {
            stability_score -= 500 * (viscous_number - 0.1f);
        }

        // Reynolds number in reasonable range (100 - 10000)
        if (reynolds >= 100 && reynolds <= 10000) {
            stability_score += 25;
        } else {
            stability_score -= 100;
        }

        // Prefer smaller timesteps for accuracy (but not too small)
        if (timestep >= 1e-5 && timestep <= 1e-2) {
            stability_score += 10 / timestep;  // Reward smaller timesteps
        }

        return stability_score;
    }

    static void runOptimization() {
        std::cout << "\n=== Fluid Simulation Parameter Tuning ===" << std::endl;
        std::cout << "Optimizing SPH parameters for stability and accuracy" << std::endl;

        ParameterSpace<float> space;

        // Density parameter
        Parameter<float> density_param;
        density_param.name = "density";
        density_param.type = ParameterType::CONTINUOUS;
        density_param.min_value = 100.0f;
        density_param.max_value = 2000.0f;
        density_param.default_value = 1000.0f;  // Water
        density_param.description = "Fluid density";
        density_param.units = "kg/m³";
        space.addParameter(density_param);

        // Viscosity parameter
        Parameter<float> viscosity_param;
        viscosity_param.name = "viscosity";
        viscosity_param.type = ParameterType::CONTINUOUS;
        viscosity_param.min_value = 1e-6f;
        viscosity_param.max_value = 1e-1f;
        viscosity_param.distribution = DistributionType::LOG_UNIFORM;
        viscosity_param.default_value = 1e-3f;  // Water
        viscosity_param.description = "Dynamic viscosity";
        viscosity_param.units = "Pa·s";
        space.addParameter(viscosity_param);

        // Timestep parameter
        Parameter<float> timestep_param;
        timestep_param.name = "timestep";
        timestep_param.type = ParameterType::CONTINUOUS;
        timestep_param.min_value = 1e-6f;
        timestep_param.max_value = 1e-1f;
        timestep_param.distribution = DistributionType::LOG_UNIFORM;
        timestep_param.default_value = 1e-3f;
        timestep_param.description = "Integration timestep";
        timestep_param.units = "s";
        space.addParameter(timestep_param);

        // Smoothing length parameter
        Parameter<float> smoothing_param;
        smoothing_param.name = "smoothing_length";
        smoothing_param.type = ParameterType::CONTINUOUS;
        smoothing_param.min_value = 1e-3f;
        smoothing_param.max_value = 1e-1f;
        smoothing_param.distribution = DistributionType::LOG_UNIFORM;
        smoothing_param.default_value = 1e-2f;
        smoothing_param.description = "SPH smoothing length";
        smoothing_param.units = "m";
        space.addParameter(smoothing_param);

        // Create objective
        Objective<float> objective;
        objective.name = "stability_score";
        objective.type = ObjectiveType::MAXIMIZE;
        objective.evaluation_function = evaluateFluidSim;

        // Create experiment
        ExperimentDesigner<float> designer;
        designer.setParameterSpace(space);
        designer.addObjective(objective);

        ExperimentConfig config;
        config.max_evaluations = 80;
        config.parallel_evaluations = 3;
        config.use_bayesian_optimization = true;
        config.experiment_name = "fluid_tuning";
        designer.setConfig(config);

        // Run optimization
        designer.run();

        // Display results
        auto best = designer.getBestResult();
        std::cout << "\nOptimal Fluid Parameters:" << std::endl;
        std::cout << "  Density: " << std::fixed << std::setprecision(1)
                  << best.parameter_values[0] << " kg/m³" << std::endl;
        std::cout << "  Viscosity: " << std::scientific << std::setprecision(2)
                  << best.parameter_values[1] << " Pa·s" << std::endl;
        std::cout << "  Timestep: " << best.parameter_values[2] << " s" << std::endl;
        std::cout << "  Smoothing Length: " << best.parameter_values[3] << " m" << std::endl;
        std::cout << "  Stability Score: " << std::fixed << std::setprecision(1)
                  << best.objective_values[0] << std::endl;

        // Calculate key dimensionless numbers
        float density = best.parameter_values[0];
        float viscosity = best.parameter_values[1];
        float timestep = best.parameter_values[2];
        float smoothing_length = best.parameter_values[3];

        float reynolds = density * 1.0f * 0.1f / viscosity;
        float cfl = 1.0f * timestep / smoothing_length;
        float viscous_num = viscosity * timestep / (density * smoothing_length * smoothing_length);

        std::cout << "\nDimensionless Numbers:" << std::endl;
        std::cout << "  Reynolds Number: " << std::scientific << reynolds << std::endl;
        std::cout << "  CFL Number: " << cfl << " (should be < 1)" << std::endl;
        std::cout << "  Viscous Number: " << viscous_num << " (should be < 0.1)" << std::endl;

        // Parameter importance analysis
        auto importance = designer.analyzeParameterImportance();
        std::cout << "\nParameter Importance:" << std::endl;
        for (size_t i = 0; i < importance.size(); ++i) {
            std::cout << "  " << space.getParameter(i).name << ": "
                      << std::fixed << std::setprecision(3) << importance[i] << std::endl;
        }
    }
};

int main() {
    std::cout << "PhysGrad Automatic Experiment Design - Physics Examples" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Demonstrating AI-driven parameter optimization for physics problems" << std::endl << std::endl;

    try {
        // Run all physics optimization examples
        ProjectileOptimizer::runOptimization();
        OscillatorDesigner::runOptimization();
        FluidTuner::runOptimization();

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "🎯 ALL PHYSICS OPTIMIZATION EXAMPLES COMPLETED!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "\n✅ Demonstrated Capabilities:" << std::endl;
        std::cout << "• Projectile motion optimization with air resistance" << std::endl;
        std::cout << "• Oscillator design for specific response characteristics" << std::endl;
        std::cout << "• Fluid simulation parameter tuning for stability" << std::endl;
        std::cout << "• Multi-objective optimization and constraint handling" << std::endl;
        std::cout << "• Bayesian optimization for efficient parameter search" << std::endl;
        std::cout << "• Parameter importance analysis for design insights" << std::endl;
        std::cout << "\n🚀 Real-World Applications:" << std::endl;
        std::cout << "• Aerospace: Trajectory optimization, vehicle design" << std::endl;
        std::cout << "• Mechanical: Vibration control, system design" << std::endl;
        std::cout << "• CFD: Simulation parameter tuning, solver optimization" << std::endl;
        std::cout << "• Robotics: Control parameter optimization" << std::endl;
        std::cout << "• Materials: Property optimization, design exploration" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error in physics optimization examples: " << e.what() << std::endl;
        return 1;
    }
}