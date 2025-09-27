#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "adaptive_integration.h"
#include "tape_compression.h"
#include "optimizers.h"
#include "simulation.h"

namespace physgrad {

// Configuration value type
using ConfigValue = std::variant<int, float, double, bool, std::string>;

// Configuration categories
enum class ConfigCategory {
    SIMULATION,     // Basic simulation parameters
    INTEGRATION,    // Numerical integration settings
    OPTIMIZATION,   // Optimizer settings
    TAPE,          // Tape recording and compression
    PERFORMANCE,   // Performance tuning
    DEBUG          // Debug and logging options
};

// Configuration parameter definition
struct ConfigParam {
    std::string key;
    std::string description;
    ConfigValue default_value;
    ConfigValue min_value;
    ConfigValue max_value;
    ConfigCategory category;
    bool is_required = false;

    ConfigParam() = default;

    ConfigParam(const std::string& k, const std::string& desc, ConfigValue def_val,
               ConfigCategory cat, bool required = false)
        : key(k), description(desc), default_value(def_val), category(cat), is_required(required) {}

    ConfigParam(const std::string& k, const std::string& desc, ConfigValue def_val,
               ConfigValue min_val, ConfigValue max_val, ConfigCategory cat, bool required = false)
        : key(k), description(desc), default_value(def_val), min_value(min_val),
          max_value(max_val), category(cat), is_required(required) {}
};

// Main configuration manager
class ConfigManager {
private:
    std::unordered_map<std::string, ConfigValue> values;
    std::unordered_map<std::string, ConfigParam> param_definitions;
    std::string config_file_path;

public:
    ConfigManager();

    // Configuration file operations
    bool loadFromFile(const std::string& filepath);
    bool saveToFile(const std::string& filepath) const;
    void loadFromString(const std::string& config_str);

    // Parameter management
    void registerParameter(const ConfigParam& param);
    void setValue(const std::string& key, const ConfigValue& value);
    ConfigValue getValue(const std::string& key) const;

    // Typed getters
    int getInt(const std::string& key) const;
    float getFloat(const std::string& key) const;
    double getDouble(const std::string& key) const;
    bool getBool(const std::string& key) const;
    std::string getString(const std::string& key) const;

    // Validation and defaults
    void applyDefaults();
    bool validateConfiguration() const;
    void printConfiguration(ConfigCategory category = ConfigCategory::SIMULATION) const;
    void printAllConfigurations() const;

    // Configuration presets
    void loadPreset(const std::string& preset_name);
    void savePreset(const std::string& preset_name) const;

    // Integration with other systems
    SimParams getSimulationParams() const;
    TapeCompressionConfig getTapeConfig() const;
    AdaptiveParams getIntegrationParams() const;

    // Optimization configuration
    std::unique_ptr<Optimizer> createOptimizer() const;

    // Helper functions
    std::vector<std::string> getKeys(ConfigCategory category) const;
    bool hasValue(const std::string& key) const;

private:
    void registerDefaultParameters();
    bool isValidValue(const std::string& key, const ConfigValue& value) const;
    std::string configValueToString(const ConfigValue& value) const;
    ConfigValue stringToConfigValue(const std::string& str, const ConfigValue& type_hint) const;
    std::string getCategoryName(ConfigCategory category) const;
};

// Implementation
inline ConfigManager::ConfigManager() {
    registerDefaultParameters();
    applyDefaults();
}

inline void ConfigManager::registerDefaultParameters() {
    // Simulation parameters
    registerParameter(ConfigParam("num_bodies", "Number of bodies in simulation", 1024, 1, 1000000, ConfigCategory::SIMULATION));
    registerParameter(ConfigParam("time_step", "Integration time step", 0.001f, 1e-6f, 1.0f, ConfigCategory::SIMULATION));
    registerParameter(ConfigParam("G", "Gravitational constant", 1.0f, 0.0f, 100.0f, ConfigCategory::SIMULATION));
    registerParameter(ConfigParam("epsilon", "Softening parameter", 0.001f, 0.0f, 1.0f, ConfigCategory::SIMULATION));
    registerParameter(ConfigParam("max_force", "Maximum force magnitude", 100.0f, 0.1f, 10000.0f, ConfigCategory::SIMULATION));

    // Integration parameters
    registerParameter(ConfigParam("integration_scheme", "Integration method (leapfrog, rk4, adaptive_rk45, adaptive_heun)",
                                std::string("leapfrog"), ConfigCategory::INTEGRATION));
    registerParameter(ConfigParam("adaptive_tolerance", "Error tolerance for adaptive schemes", 1e-4f, 1e-8f, 1e-2f, ConfigCategory::INTEGRATION));
    registerParameter(ConfigParam("min_dt", "Minimum adaptive time step", 1e-6f, 1e-12f, 1e-3f, ConfigCategory::INTEGRATION));
    registerParameter(ConfigParam("max_dt", "Maximum adaptive time step", 0.1f, 1e-3f, 1.0f, ConfigCategory::INTEGRATION));
    registerParameter(ConfigParam("max_substeps", "Maximum substeps for adaptive schemes", 100, 1, 10000, ConfigCategory::INTEGRATION));

    // Optimization parameters
    registerParameter(ConfigParam("optimizer", "Optimizer type (momentum, adam, adamw, lbfgs)",
                                std::string("adam"), ConfigCategory::OPTIMIZATION));
    registerParameter(ConfigParam("learning_rate", "Learning rate", 0.01f, 1e-6f, 10.0f, ConfigCategory::OPTIMIZATION));
    registerParameter(ConfigParam("momentum_decay", "Momentum decay factor", 0.9f, 0.0f, 0.999f, ConfigCategory::OPTIMIZATION));
    registerParameter(ConfigParam("adam_beta1", "Adam beta1 parameter", 0.9f, 0.0f, 0.999f, ConfigCategory::OPTIMIZATION));
    registerParameter(ConfigParam("adam_beta2", "Adam beta2 parameter", 0.999f, 0.0f, 0.9999f, ConfigCategory::OPTIMIZATION));
    registerParameter(ConfigParam("adam_epsilon", "Adam epsilon parameter", 1e-8f, 1e-12f, 1e-4f, ConfigCategory::OPTIMIZATION));
    registerParameter(ConfigParam("weight_decay", "Weight decay (L2 regularization)", 0.01f, 0.0f, 1.0f, ConfigCategory::OPTIMIZATION));
    registerParameter(ConfigParam("lbfgs_history", "L-BFGS history size", 10, 1, 100, ConfigCategory::OPTIMIZATION));

    // Tape compression parameters
    registerParameter(ConfigParam("tape_compression", "Tape compression mode (none, sparse, adaptive, delta, quantized)",
                                std::string("sparse"), ConfigCategory::TAPE));
    registerParameter(ConfigParam("max_memory_mb", "Maximum tape memory usage (MB)", 512, 1, 32768, ConfigCategory::TAPE));
    registerParameter(ConfigParam("checkpoint_interval", "Steps between checkpoints", 10, 1, 1000, ConfigCategory::TAPE));
    registerParameter(ConfigParam("quantization_bits", "Bits for quantization", 16, 8, 32, ConfigCategory::TAPE));
    registerParameter(ConfigParam("adaptive_threshold", "Memory threshold for adaptive compression", 0.8f, 0.1f, 1.0f, ConfigCategory::TAPE));

    // Performance parameters
    registerParameter(ConfigParam("enable_cuda_streams", "Use CUDA streams for async operations", true, ConfigCategory::PERFORMANCE));
    registerParameter(ConfigParam("enable_stability_improvements", "Enable stability enhancements", true, ConfigCategory::PERFORMANCE));
    registerParameter(ConfigParam("block_size", "CUDA block size", 256, 32, 1024, ConfigCategory::PERFORMANCE));
    registerParameter(ConfigParam("enable_gradient_caching", "Cache gradients for reuse", false, ConfigCategory::PERFORMANCE));

    // Debug parameters
    registerParameter(ConfigParam("enable_logging", "Enable detailed logging", false, ConfigCategory::DEBUG));
    registerParameter(ConfigParam("log_level", "Logging level (0=error, 1=warn, 2=info, 3=debug)", 1, 0, 3, ConfigCategory::DEBUG));
    registerParameter(ConfigParam("print_progress", "Print optimization progress", true, ConfigCategory::DEBUG));
    registerParameter(ConfigParam("save_trajectory", "Save trajectory to file", false, ConfigCategory::DEBUG));
    registerParameter(ConfigParam("validation_checks", "Enable validation checks", true, ConfigCategory::DEBUG));
}

inline bool ConfigManager::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open config file: " << filepath << std::endl;
        return false;
    }

    config_file_path = filepath;
    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) {
        line_number++;

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Parse key=value pairs
        size_t equals_pos = line.find('=');
        if (equals_pos == std::string::npos) {
            std::cerr << "Invalid config line " << line_number << ": " << line << std::endl;
            continue;
        }

        std::string key = line.substr(0, equals_pos);
        std::string value_str = line.substr(equals_pos + 1);

        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value_str.erase(0, value_str.find_first_not_of(" \t"));
        value_str.erase(value_str.find_last_not_of(" \t") + 1);

        // Convert and set value
        if (param_definitions.count(key)) {
            try {
                ConfigValue value = stringToConfigValue(value_str, param_definitions[key].default_value);
                setValue(key, value);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing config line " << line_number << ": " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Unknown config parameter: " << key << std::endl;
        }
    }

    return true;
}

inline bool ConfigManager::saveToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not create config file: " << filepath << std::endl;
        return false;
    }

    file << "# PhysGrad Configuration File\n";
    file << "# Generated automatically\n\n";

    // Write by category
    for (auto category : {ConfigCategory::SIMULATION, ConfigCategory::INTEGRATION,
                         ConfigCategory::OPTIMIZATION, ConfigCategory::TAPE,
                         ConfigCategory::PERFORMANCE, ConfigCategory::DEBUG}) {

        file << "\n# " << getCategoryName(category) << " Parameters\n";

        for (const auto& [key, param] : param_definitions) {
            if (param.category == category && values.count(key)) {
                file << "# " << param.description << "\n";
                file << key << " = " << configValueToString(values.at(key)) << "\n\n";
            }
        }
    }

    return true;
}

inline void ConfigManager::registerParameter(const ConfigParam& param) {
    param_definitions[param.key] = param;
}

inline void ConfigManager::setValue(const std::string& key, const ConfigValue& value) {
    if (!isValidValue(key, value)) {
        throw std::invalid_argument("Invalid value for parameter: " + key);
    }
    values[key] = value;
}

inline ConfigValue ConfigManager::getValue(const std::string& key) const {
    auto it = values.find(key);
    if (it != values.end()) {
        return it->second;
    }

    // Return default if available
    auto param_it = param_definitions.find(key);
    if (param_it != param_definitions.end()) {
        return param_it->second.default_value;
    }

    throw std::invalid_argument("Unknown parameter: " + key);
}

inline int ConfigManager::getInt(const std::string& key) const {
    return std::get<int>(getValue(key));
}

inline float ConfigManager::getFloat(const std::string& key) const {
    return std::get<float>(getValue(key));
}

inline double ConfigManager::getDouble(const std::string& key) const {
    return std::get<double>(getValue(key));
}

inline bool ConfigManager::getBool(const std::string& key) const {
    return std::get<bool>(getValue(key));
}

inline std::string ConfigManager::getString(const std::string& key) const {
    return std::get<std::string>(getValue(key));
}

inline void ConfigManager::applyDefaults() {
    for (const auto& [key, param] : param_definitions) {
        if (!values.count(key)) {
            values[key] = param.default_value;
        }
    }
}

inline bool ConfigManager::validateConfiguration() const {
    for (const auto& [key, param] : param_definitions) {
        if (param.is_required && !values.count(key)) {
            std::cerr << "Missing required parameter: " << key << std::endl;
            return false;
        }

        if (values.count(key) && !isValidValue(key, values.at(key))) {
            std::cerr << "Invalid value for parameter: " << key << std::endl;
            return false;
        }
    }
    return true;
}

inline void ConfigManager::printConfiguration(ConfigCategory category) const {
    std::cout << "\n" << getCategoryName(category) << " Configuration:\n";
    std::cout << std::string(50, '-') << "\n";

    for (const auto& [key, param] : param_definitions) {
        if (param.category == category && values.count(key)) {
            std::cout << "  " << key << " = " << configValueToString(values.at(key))
                     << " (" << param.description << ")\n";
        }
    }
    std::cout << "\n";
}

inline void ConfigManager::printAllConfigurations() const {
    for (auto category : {ConfigCategory::SIMULATION, ConfigCategory::INTEGRATION,
                         ConfigCategory::OPTIMIZATION, ConfigCategory::TAPE,
                         ConfigCategory::PERFORMANCE, ConfigCategory::DEBUG}) {
        printConfiguration(category);
    }
}

inline SimParams ConfigManager::getSimulationParams() const {
    SimParams params;
    params.num_bodies = getInt("num_bodies");
    params.time_step = getFloat("time_step");
    params.G = getFloat("G");
    params.epsilon = getFloat("epsilon");
    params.max_force = getFloat("max_force");
    return params;
}

inline TapeCompressionConfig ConfigManager::getTapeConfig() const {
    TapeCompressionConfig config;

    std::string mode_str = getString("tape_compression");
    if (mode_str == "none") config.mode = CompressionMode::NONE;
    else if (mode_str == "sparse") config.mode = CompressionMode::SPARSE;
    else if (mode_str == "adaptive") config.mode = CompressionMode::ADAPTIVE;
    else if (mode_str == "delta") config.mode = CompressionMode::DELTA;
    else if (mode_str == "quantized") config.mode = CompressionMode::QUANTIZED;

    config.max_memory_mb = getInt("max_memory_mb");
    config.checkpoint_interval = getInt("checkpoint_interval");
    config.quantization_bits = getInt("quantization_bits");
    config.adaptive_threshold = getFloat("adaptive_threshold");

    return config;
}

inline AdaptiveParams ConfigManager::getIntegrationParams() const {
    AdaptiveParams params;
    params.tolerance = getFloat("adaptive_tolerance");
    params.min_dt = getFloat("min_dt");
    params.max_dt = getFloat("max_dt");
    params.max_substeps = getInt("max_substeps");
    return params;
}

inline std::unique_ptr<Optimizer> ConfigManager::createOptimizer() const {
    std::string optimizer_type = getString("optimizer");
    float lr = getFloat("learning_rate");

    if (optimizer_type == "momentum") {
        float decay = getFloat("momentum_decay");
        return std::make_unique<MomentumOptimizer>(lr, decay);
    } else if (optimizer_type == "adam") {
        float beta1 = getFloat("adam_beta1");
        float beta2 = getFloat("adam_beta2");
        float epsilon = getFloat("adam_epsilon");
        return std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon);
    } else if (optimizer_type == "adamw") {
        float beta1 = getFloat("adam_beta1");
        float beta2 = getFloat("adam_beta2");
        float epsilon = getFloat("adam_epsilon");
        float weight_decay = getFloat("weight_decay");
        return std::make_unique<AdamWOptimizer>(lr, beta1, beta2, epsilon, weight_decay);
    } else if (optimizer_type == "lbfgs") {
        int history = getInt("lbfgs_history");
        return std::make_unique<LBFGSOptimizer>(lr, history);
    } else {
        return std::make_unique<AdamOptimizer>(lr);
    }
}

inline bool ConfigManager::hasValue(const std::string& key) const {
    return values.count(key) > 0;
}

inline std::vector<std::string> ConfigManager::getKeys(ConfigCategory category) const {
    std::vector<std::string> keys;
    for (const auto& [key, param] : param_definitions) {
        if (param.category == category) {
            keys.push_back(key);
        }
    }
    return keys;
}

// Helper functions
inline bool ConfigManager::isValidValue(const std::string& key, const ConfigValue& value) const {
    auto param_it = param_definitions.find(key);
    if (param_it == param_definitions.end()) return false;

    const ConfigParam& param = param_it->second;

    // Type checking is handled by std::variant
    // Range checking for numeric types
    if (std::holds_alternative<int>(value) && std::holds_alternative<int>(param.min_value)) {
        int val = std::get<int>(value);
        int min_val = std::get<int>(param.min_value);
        int max_val = std::get<int>(param.max_value);
        return val >= min_val && val <= max_val;
    }

    if (std::holds_alternative<float>(value) && std::holds_alternative<float>(param.min_value)) {
        float val = std::get<float>(value);
        float min_val = std::get<float>(param.min_value);
        float max_val = std::get<float>(param.max_value);
        return val >= min_val && val <= max_val;
    }

    return true;
}

inline std::string ConfigManager::configValueToString(const ConfigValue& value) const {
    return std::visit([](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            return v;
        } else if constexpr (std::is_same_v<T, bool>) {
            return v ? "true" : "false";
        } else {
            return std::to_string(v);
        }
    }, value);
}

inline ConfigValue ConfigManager::stringToConfigValue(const std::string& str, const ConfigValue& type_hint) const {
    return std::visit([&str](const auto& hint) -> ConfigValue {
        using T = std::decay_t<decltype(hint)>;
        if constexpr (std::is_same_v<T, int>) {
            return std::stoi(str);
        } else if constexpr (std::is_same_v<T, float>) {
            return std::stof(str);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::stod(str);
        } else if constexpr (std::is_same_v<T, bool>) {
            return str == "true" || str == "1" || str == "yes";
        } else if constexpr (std::is_same_v<T, std::string>) {
            return str;
        }
        return str;
    }, type_hint);
}

inline std::string ConfigManager::getCategoryName(ConfigCategory category) const {
    switch (category) {
        case ConfigCategory::SIMULATION: return "Simulation";
        case ConfigCategory::INTEGRATION: return "Integration";
        case ConfigCategory::OPTIMIZATION: return "Optimization";
        case ConfigCategory::TAPE: return "Tape";
        case ConfigCategory::PERFORMANCE: return "Performance";
        case ConfigCategory::DEBUG: return "Debug";
        default: return "Unknown";
    }
}

} // namespace physgrad