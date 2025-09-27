#include "config_system.h"
#include <iostream>
#include <filesystem>

using namespace physgrad;

class ConfigDemo {
public:
    static void demonstrateConfigSystem() {
        std::cout << "PhysGrad Configuration System Demo\n";
        std::cout << "==================================\n\n";

        testBasicConfiguration();
        testConfigurationFiles();
        testPresetCreation();
        testSystemIntegration();
    }

private:
    static void testBasicConfiguration() {
        std::cout << "1. BASIC CONFIGURATION MANAGEMENT:\n";
        std::cout << "----------------------------------\n";

        ConfigManager config;

        // Show default configuration
        std::cout << "Default simulation parameters:\n";
        config.printConfiguration(ConfigCategory::SIMULATION);

        // Modify some parameters
        config.setValue("num_bodies", 512);
        config.setValue("time_step", 0.005f);
        config.setValue("G", 2.0f);

        std::cout << "Modified simulation parameters:\n";
        config.printConfiguration(ConfigCategory::SIMULATION);

        // Test typed getters
        std::cout << "Reading values:\n";
        std::cout << "  num_bodies: " << config.getInt("num_bodies") << "\n";
        std::cout << "  time_step: " << config.getFloat("time_step") << "\n";
        std::cout << "  G: " << config.getFloat("G") << "\n\n";

        // Test validation
        std::cout << "Testing validation:\n";
        try {
            config.setValue("num_bodies", -100);  // Should fail
            std::cout << "  ERROR: Validation should have failed!\n";
        } catch (const std::exception& e) {
            std::cout << "  ✓ Validation correctly rejected invalid value: " << e.what() << "\n";
        }

        try {
            config.setValue("time_step", 10.0f);  // Should fail (too large)
            std::cout << "  ERROR: Validation should have failed!\n";
        } catch (const std::exception& e) {
            std::cout << "  ✓ Validation correctly rejected out-of-range value: " << e.what() << "\n";
        }

        std::cout << "\n";
    }

    static void testConfigurationFiles() {
        std::cout << "2. CONFIGURATION FILE OPERATIONS:\n";
        std::cout << "---------------------------------\n";

        ConfigManager config;

        // Create a sample configuration
        config.setValue("num_bodies", 256);
        config.setValue("time_step", 0.002f);
        config.setValue("optimizer", std::string("adamw"));
        config.setValue("learning_rate", 0.001f);
        config.setValue("tape_compression", std::string("quantized"));
        config.setValue("enable_logging", true);

        // Save to file
        std::string config_file = "test_config.cfg";
        if (config.saveToFile(config_file)) {
            std::cout << "✓ Configuration saved to " << config_file << "\n";
        } else {
            std::cout << "✗ Failed to save configuration\n";
        }

        // Load from file
        ConfigManager new_config;
        if (new_config.loadFromFile(config_file)) {
            std::cout << "✓ Configuration loaded from " << config_file << "\n";

            // Verify values
            std::cout << "Loaded values:\n";
            std::cout << "  num_bodies: " << new_config.getInt("num_bodies") << "\n";
            std::cout << "  time_step: " << new_config.getFloat("time_step") << "\n";
            std::cout << "  optimizer: " << new_config.getString("optimizer") << "\n";
            std::cout << "  learning_rate: " << new_config.getFloat("learning_rate") << "\n";
            std::cout << "  tape_compression: " << new_config.getString("tape_compression") << "\n";
            std::cout << "  enable_logging: " << (new_config.getBool("enable_logging") ? "true" : "false") << "\n";
        } else {
            std::cout << "✗ Failed to load configuration\n";
        }

        // Clean up
        std::filesystem::remove(config_file);
        std::cout << "\n";
    }

    static void testPresetCreation() {
        std::cout << "3. CONFIGURATION PRESETS:\n";
        std::cout << "------------------------\n";

        // High-accuracy preset
        {
            ConfigManager config;
            config.setValue("integration_scheme", std::string("adaptive_rk45"));
            config.setValue("adaptive_tolerance", 1e-6f);
            config.setValue("optimizer", std::string("lbfgs"));
            config.setValue("tape_compression", std::string("none"));
            config.setValue("validation_checks", true);

            config.saveToFile("high_accuracy_preset.cfg");
            std::cout << "✓ High-accuracy preset created\n";
            config.printConfiguration(ConfigCategory::INTEGRATION);
        }

        // Performance preset
        {
            ConfigManager config;
            config.setValue("integration_scheme", std::string("leapfrog"));
            config.setValue("optimizer", std::string("momentum"));
            config.setValue("learning_rate", 0.1f);
            config.setValue("tape_compression", std::string("adaptive"));
            config.setValue("enable_cuda_streams", true);
            config.setValue("validation_checks", false);

            config.saveToFile("performance_preset.cfg");
            std::cout << "✓ Performance preset created\n";
            config.printConfiguration(ConfigCategory::OPTIMIZATION);
        }

        // Memory-efficient preset
        {
            ConfigManager config;
            config.setValue("tape_compression", std::string("quantized"));
            config.setValue("max_memory_mb", 128);
            config.setValue("checkpoint_interval", 5);
            config.setValue("quantization_bits", 8);

            config.saveToFile("memory_efficient_preset.cfg");
            std::cout << "✓ Memory-efficient preset created\n";
            config.printConfiguration(ConfigCategory::TAPE);
        }

        // Clean up
        std::filesystem::remove("high_accuracy_preset.cfg");
        std::filesystem::remove("performance_preset.cfg");
        std::filesystem::remove("memory_efficient_preset.cfg");
        std::cout << "\n";
    }

    static void testSystemIntegration() {
        std::cout << "4. SYSTEM INTEGRATION:\n";
        std::cout << "---------------------\n";

        ConfigManager config;

        // Configure for a specific use case
        config.setValue("num_bodies", 100);
        config.setValue("integration_scheme", std::string("adaptive_rk45"));
        config.setValue("optimizer", std::string("adam"));
        config.setValue("learning_rate", 0.01f);
        config.setValue("tape_compression", std::string("quantized"));

        // Extract configuration for different subsystems
        std::cout << "Extracting subsystem configurations:\n\n";

        // Simulation parameters
        SimParams sim_params = config.getSimulationParams();
        std::cout << "Simulation Parameters:\n";
        std::cout << "  Bodies: " << sim_params.num_bodies << "\n";
        std::cout << "  Time step: " << sim_params.time_step << "\n";
        std::cout << "  G: " << sim_params.G << "\n";
        std::cout << "  Epsilon: " << sim_params.epsilon << "\n\n";

        // Integration parameters
        AdaptiveParams int_params = config.getIntegrationParams();
        std::cout << "Integration Parameters:\n";
        std::cout << "  Tolerance: " << int_params.tolerance << "\n";
        std::cout << "  Min dt: " << int_params.min_dt << "\n";
        std::cout << "  Max dt: " << int_params.max_dt << "\n";
        std::cout << "  Max substeps: " << int_params.max_substeps << "\n\n";

        // Tape configuration
        TapeCompressionConfig tape_config = config.getTapeConfig();
        std::cout << "Tape Configuration:\n";
        std::cout << "  Compression mode: " << static_cast<int>(tape_config.mode) << "\n";
        std::cout << "  Max memory MB: " << tape_config.max_memory_mb << "\n";
        std::cout << "  Checkpoint interval: " << tape_config.checkpoint_interval << "\n";
        std::cout << "  Quantization bits: " << tape_config.quantization_bits << "\n\n";

        // Optimizer creation
        auto optimizer = config.createOptimizer();
        std::cout << "Created optimizer: " << optimizer->getName() << "\n";
        std::cout << "Learning rate: " << optimizer->getLearningRate() << "\n\n";

        std::cout << std::string(60, '=') << "\n";
        std::cout << "CONFIGURATION SYSTEM BENEFITS:\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "✅ Centralized parameter management\n";
        std::cout << "✅ Type-safe configuration with validation\n";
        std::cout << "✅ File-based configuration persistence\n";
        std::cout << "✅ Configuration presets for common use cases\n";
        std::cout << "✅ Easy integration with all subsystems\n";
        std::cout << "✅ Runtime parameter modification\n";
        std::cout << "✅ Comprehensive parameter documentation\n";
        std::cout << "\nUsage patterns:\n";
        std::cout << "- Load configuration from file at startup\n";
        std::cout << "- Use presets for common scenarios\n";
        std::cout << "- Modify parameters during optimization runs\n";
        std::cout << "- Save successful configurations for later use\n";
    }
};

int main() {
    try {
        ConfigDemo::demonstrateConfigSystem();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}