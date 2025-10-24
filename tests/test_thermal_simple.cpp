#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <memory>

// Simple Vec3 for testing
template<typename T>
struct Vec3 {
    T x, y, z;
    Vec3(T x_ = T{0}, T y_ = T{0}, T z_ = T{0}) : x(x_), y(y_), z(z_) {}
    T operator[](size_t i) const { return (&x)[i]; }
    T& operator[](size_t i) { return (&x)[i]; }
};

// Simple int3 for testing
struct int3 {
    int x, y, z;
    int3(int x_ = 0, int y_ = 0, int z_ = 0) : x(x_), y(y_), z(z_) {}
};

namespace physgrad {
namespace thermal {

/**
 * Basic thermal material properties
 */
template<typename T>
struct ThermalMaterial {
    T thermal_conductivity;      // W/(m·K)
    T specific_heat_capacity;    // J/(kg·K)
    T density;                   // kg/m³
    T melting_point;            // K
    T boiling_point;            // K
    T thermal_expansion_coeff;   // 1/K

    ThermalMaterial()
        : thermal_conductivity(1.0), specific_heat_capacity(1000.0), density(1000.0),
          melting_point(273.15), boiling_point(373.15), thermal_expansion_coeff(1e-4) {}

    enum class PhaseState { SOLID, LIQUID, GAS };

    PhaseState getPhaseState(T temperature) const {
        if (temperature < melting_point) return PhaseState::SOLID;
        if (temperature < boiling_point) return PhaseState::LIQUID;
        return PhaseState::GAS;
    }
};

/**
 * Material library
 */
template<typename T>
class ThermalMaterialLibrary {
public:
    static ThermalMaterial<T> createSteel() {
        ThermalMaterial<T> steel;
        steel.thermal_conductivity = 50.0;
        steel.specific_heat_capacity = 500.0;
        steel.density = 7850.0;
        steel.melting_point = 1811.0;
        steel.thermal_expansion_coeff = 12e-6;
        return steel;
    }

    static ThermalMaterial<T> createWater() {
        ThermalMaterial<T> water;
        water.thermal_conductivity = 0.6;
        water.specific_heat_capacity = 4182.0;
        water.density = 1000.0;
        water.melting_point = 273.15;
        water.boiling_point = 373.15;
        water.thermal_expansion_coeff = 214e-6;
        return water;
    }

    static ThermalMaterial<T> createAluminum() {
        ThermalMaterial<T> aluminum;
        aluminum.thermal_conductivity = 205.0;
        aluminum.specific_heat_capacity = 900.0;
        aluminum.density = 2700.0;
        aluminum.melting_point = 933.0;
        aluminum.thermal_expansion_coeff = 23e-6;
        return aluminum;
    }
};

/**
 * Simple thermal field
 */
template<typename T>
class ThermalField {
private:
    int3 dimensions_;
    Vec3<T> cell_size_;
    std::vector<T> temperatures_;
    std::vector<T> heat_sources_;
    std::vector<int> material_ids_;
    std::vector<ThermalMaterial<T>> materials_;

public:
    ThermalField(const int3& dims, const Vec3<T>& cell_size)
        : dimensions_(dims), cell_size_(cell_size) {

        size_t total_nodes = dims.x * dims.y * dims.z;
        temperatures_.resize(total_nodes, 293.15);  // Room temperature
        heat_sources_.resize(total_nodes, 0.0);
        material_ids_.resize(total_nodes, 0);
    }

    size_t getNodeIndex(int i, int j, int k) const {
        return k * dimensions_.x * dimensions_.y + j * dimensions_.x + i;
    }

    T getTemperature(int i, int j, int k) const {
        return temperatures_[getNodeIndex(i, j, k)];
    }

    void setTemperature(int i, int j, int k, T temp) {
        temperatures_[getNodeIndex(i, j, k)] = temp;
    }

    void addMaterial(const ThermalMaterial<T>& material) {
        materials_.push_back(material);
    }

    void setMaterialID(size_t node_id, int material_id) {
        material_ids_[node_id] = material_id;
    }

    const ThermalMaterial<T>& getMaterial(size_t node_id) const {
        return materials_[material_ids_[node_id]];
    }

    T getMinTemperature() const {
        return *std::min_element(temperatures_.begin(), temperatures_.end());
    }

    T getMaxTemperature() const {
        return *std::max_element(temperatures_.begin(), temperatures_.end());
    }

    T getAverageTemperature() const {
        T sum = 0;
        for (T temp : temperatures_) sum += temp;
        return sum / temperatures_.size();
    }

    const int3& getDimensions() const { return dimensions_; }
    const Vec3<T>& getCellSize() const { return cell_size_; }
    std::vector<T>& getTemperatureData() { return temperatures_; }
    const std::vector<T>& getTemperatureData() const { return temperatures_; }
    std::vector<T>& getHeatSourceData() { return heat_sources_; }

    size_t getTotalNodes() const { return temperatures_.size(); }
};

/**
 * Heat conduction solver
 */
template<typename T>
class HeatConductionSolver {
private:
    ThermalField<T>* field_;

public:
    explicit HeatConductionSolver(ThermalField<T>* field) : field_(field) {}

    void solveExplicit(T dt) {
        auto dims = field_->getDimensions();
        auto cell_size = field_->getCellSize();
        auto& temp_data = field_->getTemperatureData();
        auto& source_data = field_->getHeatSourceData();

        std::vector<T> new_temperatures = temp_data;

        T dx2 = cell_size.x * cell_size.x;
        T dy2 = cell_size.y * cell_size.y;
        T dz2 = cell_size.z * cell_size.z;

        // Apply heat equation: ∂T/∂t = α∇²T
        for (int k = 1; k < dims.z - 1; ++k) {
            for (int j = 1; j < dims.y - 1; ++j) {
                for (int i = 1; i < dims.x - 1; ++i) {
                    size_t idx = field_->getNodeIndex(i, j, k);
                    const auto& material = field_->getMaterial(idx);

                    T alpha = material.thermal_conductivity /
                             (material.density * material.specific_heat_capacity);

                    T T_center = temp_data[idx];

                    // Finite differences
                    T d2T_dx2 = (temp_data[field_->getNodeIndex(i+1, j, k)] -
                                2*T_center +
                                temp_data[field_->getNodeIndex(i-1, j, k)]) / dx2;

                    T d2T_dy2 = (temp_data[field_->getNodeIndex(i, j+1, k)] -
                                2*T_center +
                                temp_data[field_->getNodeIndex(i, j-1, k)]) / dy2;

                    T d2T_dz2 = (temp_data[field_->getNodeIndex(i, j, k+1)] -
                                2*T_center +
                                temp_data[field_->getNodeIndex(i, j, k-1)]) / dz2;

                    T laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2;
                    T source_term = source_data[idx] / (material.density * material.specific_heat_capacity);

                    new_temperatures[idx] = T_center + dt * (alpha * laplacian + source_term);
                }
            }
        }

        temp_data = new_temperatures;
    }
};

/**
 * Simple thermal simulation system
 */
template<typename T>
class ThermalSimulationSystem {
private:
    std::unique_ptr<ThermalField<T>> field_;
    std::unique_ptr<HeatConductionSolver<T>> solver_;

public:
    ThermalSimulationSystem(const int3& dims, const Vec3<T>& cell_size) {
        field_ = std::make_unique<ThermalField<T>>(dims, cell_size);
        solver_ = std::make_unique<HeatConductionSolver<T>>(field_.get());
    }

    ThermalField<T>& getThermalField() { return *field_; }
    const ThermalField<T>& getThermalField() const { return *field_; }

    void simulationStep(T dt) {
        // Clear heat sources
        auto& sources = field_->getHeatSourceData();
        std::fill(sources.begin(), sources.end(), T{0});

        // Solve heat conduction
        solver_->solveExplicit(dt);
    }

    T getTotalThermalEnergy() const {
        T total = 0;
        const auto& temps = field_->getTemperatureData();
        auto cell_volume = field_->getCellSize().x * field_->getCellSize().y * field_->getCellSize().z;

        for (size_t i = 0; i < field_->getTotalNodes(); ++i) {
            const auto& material = field_->getMaterial(i);
            total += material.density * material.specific_heat_capacity * temps[i] * cell_volume;
        }
        return total;
    }
};

} // namespace thermal
} // namespace physgrad

bool testThermalMaterials() {
    std::cout << "Testing Thermal Materials..." << std::endl;

    physgrad::thermal::ThermalMaterialLibrary<double> library;

    auto steel = library.createSteel();
    auto water = library.createWater();
    auto aluminum = library.createAluminum();

    std::cout << "Steel conductivity: " << steel.thermal_conductivity << " W/(m·K)" << std::endl;
    std::cout << "Water melting point: " << water.melting_point << " K" << std::endl;
    std::cout << "Aluminum density: " << aluminum.density << " kg/m³" << std::endl;

    // Test phase states
    auto ice_phase = water.getPhaseState(250.0);
    auto liquid_phase = water.getPhaseState(300.0);
    auto steam_phase = water.getPhaseState(400.0);

    if (ice_phase != physgrad::thermal::ThermalMaterial<double>::PhaseState::SOLID ||
        liquid_phase != physgrad::thermal::ThermalMaterial<double>::PhaseState::LIQUID ||
        steam_phase != physgrad::thermal::ThermalMaterial<double>::PhaseState::GAS) {
        std::cout << "❌ Phase state determination failed" << std::endl;
        return false;
    }

    std::cout << "✅ Thermal Materials test passed" << std::endl;
    return true;
}

bool testThermalField() {
    std::cout << "Testing Thermal Field..." << std::endl;

    int3 dims{10, 10, 10};
    Vec3<double> cell_size{0.1, 0.1, 0.1};

    physgrad::thermal::ThermalField<double> field(dims, cell_size);

    // Add materials
    physgrad::thermal::ThermalMaterialLibrary<double> library;
    field.addMaterial(library.createSteel());
    field.addMaterial(library.createWater());

    // Set temperatures
    field.setTemperature(0, 0, 0, 500.0);
    field.setTemperature(9, 9, 9, 200.0);

    double temp1 = field.getTemperature(0, 0, 0);
    double temp2 = field.getTemperature(9, 9, 9);

    if (std::abs(temp1 - 500.0) > 1e-10 || std::abs(temp2 - 200.0) > 1e-10) {
        std::cout << "❌ Temperature setting failed" << std::endl;
        return false;
    }

    // Test statistics
    double min_temp = field.getMinTemperature();
    double max_temp = field.getMaxTemperature();
    double avg_temp = field.getAverageTemperature();

    std::cout << "Temperature range: " << min_temp << " - " << max_temp << " K" << std::endl;
    std::cout << "Average temperature: " << avg_temp << " K" << std::endl;

    std::cout << "✅ Thermal Field test passed" << std::endl;
    return true;
}

bool testHeatConduction() {
    std::cout << "Testing Heat Conduction..." << std::endl;

    int3 dims{20, 20, 20};
    Vec3<double> cell_size{0.05, 0.05, 0.05};

    physgrad::thermal::ThermalField<double> field(dims, cell_size);

    // Add steel material
    physgrad::thermal::ThermalMaterialLibrary<double> library;
    field.addMaterial(library.createSteel());

    // Set material for all nodes
    for (size_t i = 0; i < field.getTotalNodes(); ++i) {
        field.setMaterialID(i, 0);
    }

    // Create temperature gradient (hot center, cold edges)
    for (int k = 0; k < dims.z; ++k) {
        for (int j = 0; j < dims.y; ++j) {
            for (int i = 0; i < dims.x; ++i) {
                double distance = std::sqrt((i - dims.x/2)*(i - dims.x/2) +
                                           (j - dims.y/2)*(j - dims.y/2) +
                                           (k - dims.z/2)*(k - dims.z/2));
                double temp = 500.0 - distance * 10.0;
                temp = std::max(temp, 273.15);
                field.setTemperature(i, j, k, temp);
            }
        }
    }

    double initial_center_temp = field.getTemperature(dims.x/2, dims.y/2, dims.z/2);
    std::cout << "Initial center temperature: " << initial_center_temp << " K" << std::endl;

    // Run heat conduction
    physgrad::thermal::HeatConductionSolver<double> solver(&field);

    auto start = std::chrono::high_resolution_clock::now();

    double dt = 0.00001;  // Small timestep for stability
    for (int step = 0; step < 100; ++step) {
        solver.solveExplicit(dt);

        if (step % 25 == 0) {
            double center_temp = field.getTemperature(dims.x/2, dims.y/2, dims.z/2);
            double avg_temp = field.getAverageTemperature();
            std::cout << "Step " << step << " - Center: " << center_temp
                     << " K, Average: " << avg_temp << " K" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double final_center_temp = field.getTemperature(dims.x/2, dims.y/2, dims.z/2);
    std::cout << "Final center temperature: " << final_center_temp << " K" << std::endl;
    std::cout << "Simulation time: " << duration.count() << " ms" << std::endl;

    // Validate heat diffusion
    if (final_center_temp >= initial_center_temp) {
        std::cout << "❌ Heat should have diffused from center" << std::endl;
        return false;
    }

    std::cout << "Temperature drop: " << (initial_center_temp - final_center_temp) << " K" << std::endl;
    std::cout << "✅ Heat Conduction test passed" << std::endl;
    return true;
}

bool testThermalSimulation() {
    std::cout << "Testing Complete Thermal Simulation..." << std::endl;

    int3 dims{15, 15, 15};
    Vec3<double> cell_size{0.1, 0.1, 0.1};

    physgrad::thermal::ThermalSimulationSystem<double> sim(dims, cell_size);
    auto& field = sim.getThermalField();

    // Add materials
    physgrad::thermal::ThermalMaterialLibrary<double> library;
    field.addMaterial(library.createSteel());
    field.addMaterial(library.createWater());

    // Create hot steel block surrounded by water
    for (int k = 0; k < dims.z; ++k) {
        for (int j = 0; j < dims.y; ++j) {
            for (int i = 0; i < dims.x; ++i) {
                size_t node_id = field.getNodeIndex(i, j, k);

                bool is_steel_block = (i >= 6 && i <= 8 && j >= 6 && j <= 8 && k >= 6 && k <= 8);

                if (is_steel_block) {
                    field.setTemperature(i, j, k, 800.0);  // Hot steel
                    field.setMaterialID(node_id, 0);  // Steel
                } else {
                    field.setTemperature(i, j, k, 293.15);  // Room temperature water
                    field.setMaterialID(node_id, 1);  // Water
                }
            }
        }
    }

    double initial_energy = sim.getTotalThermalEnergy();
    double initial_steel_temp = field.getTemperature(7, 7, 7);

    std::cout << "Initial thermal energy: " << initial_energy << " J" << std::endl;
    std::cout << "Initial steel temperature: " << initial_steel_temp << " K" << std::endl;

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    double dt = 0.00001;
    for (int step = 0; step < 200; ++step) {
        sim.simulationStep(dt);

        if (step % 50 == 0) {
            double steel_temp = field.getTemperature(7, 7, 7);
            double water_temp = field.getTemperature(1, 1, 1);
            std::cout << "Step " << step << " - Steel: " << steel_temp
                     << " K, Water: " << water_temp << " K" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double final_energy = sim.getTotalThermalEnergy();
    double final_steel_temp = field.getTemperature(7, 7, 7);

    std::cout << "Final thermal energy: " << final_energy << " J" << std::endl;
    std::cout << "Final steel temperature: " << final_steel_temp << " K" << std::endl;
    std::cout << "Simulation time: " << duration.count() << " ms" << std::endl;

    // Validate results
    if (final_steel_temp >= initial_steel_temp) {
        std::cout << "❌ Steel should have cooled" << std::endl;
        return false;
    }

    double energy_change = std::abs(final_energy - initial_energy) / initial_energy;
    if (energy_change > 0.1) {
        std::cout << "❌ Excessive energy change: " << energy_change * 100 << "%" << std::endl;
        return false;
    }

    std::cout << "Energy conservation: " << (1.0 - energy_change) * 100 << "%" << std::endl;
    std::cout << "✅ Thermal Simulation test passed" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Thermal Physics Simple Test Suite ===" << std::endl;

    bool all_passed = true;

    all_passed &= testThermalMaterials();
    all_passed &= testThermalField();
    all_passed &= testHeatConduction();
    all_passed &= testThermalSimulation();

    std::cout << "\n=== Test Summary ===" << std::endl;
    if (all_passed) {
        std::cout << "✅ All thermal physics tests passed!" << std::endl;
        std::cout << "\nThermal Physics Features Validated:" << std::endl;
        std::cout << "• Material property database (steel, water, aluminum)" << std::endl;
        std::cout << "• Temperature field management and statistics" << std::endl;
        std::cout << "• Heat conduction with finite difference solver" << std::endl;
        std::cout << "• Phase state determination" << std::endl;
        std::cout << "• Complete thermal simulation system" << std::endl;
        std::cout << "• Energy conservation validation" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some thermal physics tests failed!" << std::endl;
        return 1;
    }
}