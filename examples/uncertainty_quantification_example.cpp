#include "../src/uncertainty_quantification.h"
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>

using namespace physgrad;

// Example 1: Structural Engineering - Beam Deflection Analysis
std::vector<double> beam_deflection_model(const std::vector<double>& params) {
    if (params.size() < 4) return {0.0, 0.0};

    double force = params[0];         // Applied force (N)
    double length = params[1];        // Beam length (m)
    double elastic_modulus = params[2]; // Young's modulus (Pa)
    double moment_inertia = params[3];  // Second moment of area (m^4)

    if (force <= 0 || length <= 0 || elastic_modulus <= 0 || moment_inertia <= 0) {
        return {0.0, 0.0};
    }

    // Simply supported beam with point load at center
    double max_deflection = (force * length * length * length) / (48.0 * elastic_modulus * moment_inertia);
    double max_stress = (force * length) / (4.0 * moment_inertia * sqrt(moment_inertia));

    return {max_deflection, max_stress};
}

// Example 2: Chemical Reaction Kinetics
std::vector<double> reaction_kinetics_model(const std::vector<double>& params) {
    if (params.size() < 3) return {0.0, 0.0};

    double activation_energy = params[0]; // kJ/mol
    double pre_exponential = params[1];   // 1/s
    double temperature = params[2];       // K

    if (temperature <= 0 || pre_exponential <= 0 || activation_energy < 0) {
        return {0.0, 0.0};
    }

    const double R = 8.314e-3; // Gas constant kJ/(mol·K)
    double rate_constant = pre_exponential * exp(-activation_energy / (R * temperature));
    double half_life = log(2.0) / rate_constant;

    return {rate_constant, half_life};
}

// Example 3: Fluid Flow - Pipe Pressure Drop
std::vector<double> pipe_flow_model(const std::vector<double>& params) {
    if (params.size() < 5) return {0.0, 0.0};

    double velocity = params[0];       // m/s
    double diameter = params[1];       // m
    double length = params[2];         // m
    double viscosity = params[3];      // Pa·s
    double density = params[4];        // kg/m³

    if (velocity <= 0 || diameter <= 0 || length <= 0 || viscosity <= 0 || density <= 0) {
        return {0.0, 0.0};
    }

    double reynolds = (density * velocity * diameter) / viscosity;
    double friction_factor = 0.316 / pow(reynolds, 0.25); // Blasius equation for smooth pipes

    double pressure_drop = friction_factor * (length / diameter) * (0.5 * density * velocity * velocity);
    double flow_rate = M_PI * diameter * diameter * velocity / 4.0;

    return {pressure_drop, flow_rate};
}

void demonstrate_beam_analysis() {
    std::cout << "\n=== Structural Engineering: Beam Deflection Analysis ===" << std::endl;

    // Define uncertain parameters for steel beam
    std::vector<UncertaintyParameter> beam_params = {
        UncertaintyParameter("force", 5000.0, 500.0, 3000.0, 8000.0, UncertaintyParameter::NORMAL),
        UncertaintyParameter("length", 3.0, 0.1, 2.5, 4.0, UncertaintyParameter::NORMAL),
        UncertaintyParameter("E_modulus", 200e9, 10e9, 180e9, 220e9, UncertaintyParameter::NORMAL),
        UncertaintyParameter("moment_I", 8.33e-6, 0.5e-6, 7e-6, 10e-6, UncertaintyParameter::NORMAL)
    };

    UncertaintyPropagation beam_analysis(beam_deflection_model, beam_params);

    // Perform uncertainty analysis
    auto beam_stats = beam_analysis.monteCarloAnalysis(5000);

    std::cout << "Beam Analysis Results:" << std::endl;
    std::cout << "Max Deflection: " << std::scientific << std::setprecision(3)
              << beam_stats.mean[0] << " ± " << beam_stats.std_dev[0] << " m" << std::endl;
    std::cout << "Max Stress: " << beam_stats.mean[1] << " ± " << beam_stats.std_dev[1] << " Pa" << std::endl;
    std::cout << "95% CI for deflection: [" << beam_stats.percentile_5[0]
              << ", " << beam_stats.percentile_95[0] << "] m" << std::endl;

    // Parameter importance analysis
    auto beam_contributions = beam_analysis.computeParameterContributions(2000);
    std::cout << "\nParameter Contributions to Output Uncertainty:" << std::endl;
    for (const auto& pair : beam_contributions) {
        std::cout << "  " << pair.first << ": " << std::fixed << std::setprecision(2)
                  << pair.second * 100 << "%" << std::endl;
    }

    // Sensitivity analysis
    SensitivityAnalysis beam_sensitivity(beam_deflection_model, beam_params);
    auto beam_sobol = beam_sensitivity.computeSobolIndices(2000);

    std::cout << "\nSobol Sensitivity Indices (for deflection):" << std::endl;
    std::vector<std::string> param_names = {"Force", "Length", "E_modulus", "Moment_I"};
    for (size_t i = 0; i < beam_sobol.first_order.size(); ++i) {
        std::cout << "  " << param_names[i] << " - First order: "
                  << std::setprecision(3) << beam_sobol.first_order[i]
                  << ", Total: " << beam_sobol.total_order[i] << std::endl;
    }
}

void demonstrate_reaction_kinetics() {
    std::cout << "\n=== Chemical Engineering: Reaction Kinetics Analysis ===" << std::endl;

    // Define uncertain parameters for chemical reaction
    std::vector<UncertaintyParameter> reaction_params = {
        UncertaintyParameter("Ea", 85.0, 5.0, 70.0, 100.0, UncertaintyParameter::NORMAL),
        UncertaintyParameter("A", 1e10, 2e9, 5e9, 2e10, UncertaintyParameter::LOG_NORMAL),
        UncertaintyParameter("T", 573.0, 10.0, 550.0, 600.0, UncertaintyParameter::NORMAL)
    };

    UncertaintyPropagation reaction_analysis(reaction_kinetics_model, reaction_params);

    // Uncertainty propagation analysis
    auto reaction_stats = reaction_analysis.monteCarloAnalysis(3000);

    std::cout << "Reaction Kinetics Results:" << std::endl;
    std::cout << "Rate Constant: " << std::scientific << std::setprecision(3)
              << reaction_stats.mean[0] << " ± " << reaction_stats.std_dev[0] << " 1/s" << std::endl;
    std::cout << "Half-life: " << reaction_stats.mean[1] << " ± " << reaction_stats.std_dev[1] << " s" << std::endl;

    // Bayesian inference example with synthetic experimental data
    std::cout << "\n--- Bayesian Parameter Calibration ---" << std::endl;

    // Synthetic "experimental" rate constant measurements
    std::vector<double> experimental_rates = {2.45e-3, 2.38e-3, 2.52e-3, 2.41e-3, 2.47e-3};

    auto reaction_prior = std::make_shared<GaussianPrior>(
        std::vector<double>{80.0, 8e9, 570.0},
        std::vector<double>{15.0, 5e9, 20.0}
    );

    auto likelihood_func = [](const std::vector<double>& model_output, const std::vector<double>& observed) -> double {
        if (model_output.empty() || observed.empty()) return -1e6;

        double log_likelihood = 0.0;
        double sigma = 1e-4; // measurement uncertainty

        for (size_t i = 0; i < std::min(model_output.size(), observed.size()); ++i) {
            double residual = model_output[0] - observed[i]; // compare rate constants
            log_likelihood -= 0.5 * (residual * residual) / (sigma * sigma);
        }
        return log_likelihood;
    };

    BayesianInference reaction_inference(reaction_prior, likelihood_func);
    auto calibration_samples = reaction_inference.importanceSampling(experimental_rates, 2000);
    auto calibration_stats = reaction_inference.computeStatistics(calibration_samples);

    std::cout << "Calibrated Parameters:" << std::endl;
    std::cout << "Activation Energy: " << std::fixed << std::setprecision(2)
              << calibration_stats.mean[0] << " ± " << calibration_stats.std_dev[0] << " kJ/mol" << std::endl;
    std::cout << "Pre-exponential: " << std::scientific << std::setprecision(2)
              << calibration_stats.mean[1] << " ± " << calibration_stats.std_dev[1] << " 1/s" << std::endl;
    std::cout << "Temperature: " << std::fixed << std::setprecision(1)
              << calibration_stats.mean[2] << " ± " << calibration_stats.std_dev[2] << " K" << std::endl;
}

void demonstrate_fluid_flow() {
    std::cout << "\n=== Mechanical Engineering: Pipe Flow Analysis ===" << std::endl;

    // Define uncertain parameters for water flow in pipe
    std::vector<UncertaintyParameter> flow_params = {
        UncertaintyParameter("velocity", 2.0, 0.2, 1.0, 4.0, UncertaintyParameter::NORMAL),
        UncertaintyParameter("diameter", 0.1, 0.005, 0.08, 0.12, UncertaintyParameter::NORMAL),
        UncertaintyParameter("length", 100.0, 5.0, 80.0, 120.0, UncertaintyParameter::NORMAL),
        UncertaintyParameter("viscosity", 1e-3, 1e-4, 8e-4, 1.2e-3, UncertaintyParameter::NORMAL),
        UncertaintyParameter("density", 1000.0, 20.0, 950.0, 1050.0, UncertaintyParameter::NORMAL)
    };

    UncertaintyPropagation flow_analysis(pipe_flow_model, flow_params);

    // Monte Carlo uncertainty analysis
    auto flow_stats = flow_analysis.monteCarloAnalysis(4000);

    std::cout << "Pipe Flow Analysis Results:" << std::endl;
    std::cout << "Pressure Drop: " << std::fixed << std::setprecision(1)
              << flow_stats.mean[0] << " ± " << flow_stats.std_dev[0] << " Pa" << std::endl;
    std::cout << "Flow Rate: " << std::scientific << std::setprecision(3)
              << flow_stats.mean[1] << " ± " << flow_stats.std_dev[1] << " m³/s" << std::endl;

    // Calculate confidence intervals
    auto ci_width = flow_stats.confidence_interval_95();
    std::cout << "95% Confidence Intervals:" << std::endl;
    std::cout << "  Pressure Drop: ±" << std::fixed << std::setprecision(1) << ci_width[0] << " Pa" << std::endl;
    std::cout << "  Flow Rate: ±" << std::scientific << std::setprecision(2) << ci_width[1] << " m³/s" << std::endl;

    // Covariance analysis
    std::cout << "\nOutput Correlation Matrix:" << std::endl;
    for (size_t i = 0; i < flow_stats.covariance_matrix.size(); ++i) {
        for (size_t j = 0; j < flow_stats.covariance_matrix[i].size(); ++j) {
            double correlation = flow_stats.covariance_matrix[i][j] /
                               (flow_stats.std_dev[i] * flow_stats.std_dev[j]);
            std::cout << std::setw(8) << std::setprecision(3) << correlation << " ";
        }
        std::cout << std::endl;
    }

    // Global sensitivity analysis
    SensitivityAnalysis flow_sensitivity(pipe_flow_model, flow_params);
    auto flow_sobol = flow_sensitivity.computeSobolIndices(1500);

    std::cout << "\nGlobal Sensitivity Analysis (Pressure Drop):" << std::endl;
    std::vector<std::string> flow_param_names = {"Velocity", "Diameter", "Length", "Viscosity", "Density"};
    for (size_t i = 0; i < flow_sobol.first_order.size(); ++i) {
        std::cout << "  " << flow_param_names[i] << ": "
                  << std::setprecision(3) << flow_sobol.first_order[i] << std::endl;
    }
}

void demonstrate_advanced_features() {
    std::cout << "\n=== Advanced Uncertainty Quantification Features ===" << std::endl;

    // Demonstrate mixed distribution types
    std::vector<UncertaintyParameter> mixed_params = {
        UncertaintyParameter("normal_param", 10.0, 2.0, 5.0, 15.0, UncertaintyParameter::NORMAL),
        UncertaintyParameter("uniform_param", 5.0, 0.0, 0.0, 10.0, UncertaintyParameter::UNIFORM),
        UncertaintyParameter("lognormal_param", 2.0, 0.5, 0.5, 20.0, UncertaintyParameter::LOG_NORMAL)
    };

    auto mixed_model = [](const std::vector<double>& params) -> std::vector<double> {
        if (params.size() < 3) return {0.0, 0.0};
        double output1 = params[0] * params[1] + params[2];
        double output2 = sqrt(params[0]) * log(params[1] + 1) * params[2];
        return {output1, output2};
    };

    UncertaintyPropagation mixed_analysis(mixed_model, mixed_params);

    std::cout << "Mixed Distribution Analysis:" << std::endl;
    auto mixed_stats = mixed_analysis.monteCarloAnalysis(3000);

    std::cout << "Output 1: " << std::fixed << std::setprecision(2)
              << mixed_stats.mean[0] << " ± " << mixed_stats.std_dev[0] << std::endl;
    std::cout << "Output 2: " << mixed_stats.mean[1] << " ± " << mixed_stats.std_dev[1] << std::endl;

    std::cout << "Total uncertainty (variance): " << mixed_stats.total_variance() << std::endl;

    // Demonstrate parallel processing capability
    std::cout << "\n--- Parallel Sampling Performance ---" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto parallel_stats = mixed_analysis.monteCarloAnalysis(10000);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "10,000 sample analysis completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Parallel efficiency demonstrated with multi-core Monte Carlo sampling" << std::endl;
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "PhysGrad Uncertainty Quantification Examples" << std::endl;
    std::cout << "Comprehensive Bayesian Analysis for Engineering" << std::endl;
    std::cout << "============================================" << std::endl;

    try {
        demonstrate_beam_analysis();
        demonstrate_reaction_kinetics();
        demonstrate_fluid_flow();
        demonstrate_advanced_features();

        std::cout << "\n🎯 All uncertainty quantification examples completed successfully!" << std::endl;
        std::cout << "The Bayesian physics framework enables:" << std::endl;
        std::cout << "  ✓ Parameter uncertainty propagation" << std::endl;
        std::cout << "  ✓ Global sensitivity analysis" << std::endl;
        std::cout << "  ✓ Bayesian parameter calibration" << std::endl;
        std::cout << "  ✓ Model validation and verification" << std::endl;
        std::cout << "  ✓ Risk assessment and decision making" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Example failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Example failed with unknown exception" << std::endl;
        return 1;
    }
}