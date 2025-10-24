#include "src/thermal_physics.h"
#include "src/mpm_data_structures.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace physgrad;

bool testThermalMaterialDatabase() {
    std::cout << "Testing Thermal Material Database..." << std::endl;

    try {
        thermal::ThermalMaterialLibrary<double> library;

        // Test creating predefined materials
        auto steel = library.createSteel();
        auto aluminum = library.createAluminum();
        auto water = library.createWater();
        auto air = library.createAir();

        std::cout << "Steel thermal conductivity: " << steel.thermal_conductivity << " W/(m·K)" << std::endl;
        std::cout << "Aluminum density: " << aluminum.density << " kg/m³" << std::endl;
        std::cout << "Water melting point: " << water.melting_point << " K" << std::endl;
        std::cout << "Air thermal expansion: " << air.thermal_expansion_coefficient << " 1/K" << std::endl;

        // Test temperature-dependent properties
        double test_temp = 400.0; // K
        double steel_conductivity = steel.getEffectiveConductivity(test_temp);
        double water_capacity = water.getEffectiveHeatCapacity(test_temp);

        std::cout << "Steel conductivity at " << test_temp << " K: " << steel_conductivity << " W/(m·K)" << std::endl;
        std::cout << "Water heat capacity at " << test_temp << " K: " << water_capacity << " J/(kg·K)" << std::endl;

        // Test phase state determination
        auto steel_phase = steel.getPhaseState(300.0);  // Room temperature
        auto water_phase_cold = water.getPhaseState(250.0);  // Below freezing
        auto water_phase_hot = water.getPhaseState(400.0);   // Above boiling

        std::cout << "Steel phase at 300K: " << static_cast<int>(steel_phase) << std::endl;
        std::cout << "Water phase at 250K: " << static_cast<int>(water_phase_cold) << std::endl;
        std::cout << "Water phase at 400K: " << static_cast<int>(water_phase_hot) << std::endl;

        // Validate properties
        if (steel.thermal_conductivity < 40.0 || steel.thermal_conductivity > 60.0) {
            std::cout << "❌ Steel thermal conductivity out of expected range" << std::endl;
            return false;
        }

        if (water.melting_point < 270.0 || water.melting_point > 275.0) {
            std::cout << "❌ Water melting point out of expected range" << std::endl;
            return false;
        }

        std::cout << "✅ Thermal Material Database test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Thermal Material Database test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testThermalField() {
    std::cout << "Testing Thermal Field..." << std::endl;

    try {
        // Create a 10x10x10 thermal field
        int3 dims{10, 10, 10};
        Vec3<double> cell_size{0.1, 0.1, 0.1};
        Vec3<double> origin{0.0, 0.0, 0.0};

        thermal::ThermalField<double> field(dims, cell_size, origin);

        // Test basic operations
        field.setTemperature(0, 0, 0, 500.0);
        field.setTemperature(9, 9, 9, 200.0);

        double temp1 = field.getTemperature(0, 0, 0);
        double temp2 = field.getTemperature(9, 9, 9);

        std::cout << "Temperature at (0,0,0): " << temp1 << " K" << std::endl;
        std::cout << "Temperature at (9,9,9): " << temp2 << " K" << std::endl;

        if (std::abs(temp1 - 500.0) > 1e-10) {
            std::cout << "❌ Temperature setting/getting failed" << std::endl;
            return false;
        }

        // Test material assignment
        thermal::ThermalMaterialLibrary<double> library;
        auto steel = library.createSteel();
        auto water = library.createWater();

        field.addMaterial(steel);
        field.addMaterial(water);

        field.setMaterialID(field.getNodeIndex(5, 5, 5), 0);  // Steel
        field.setMaterialID(field.getNodeIndex(7, 7, 7), 1);  // Water

        // Test thermal property update
        field.updateThermalProperties();

        double diffusivity1 = field.getThermalDiffusivity(field.getNodeIndex(5, 5, 5));
        double diffusivity2 = field.getThermalDiffusivity(field.getNodeIndex(7, 7, 7));

        std::cout << "Steel thermal diffusivity: " << diffusivity1 << " m²/s" << std::endl;
        std::cout << "Water thermal diffusivity: " << diffusivity2 << " m²/s" << std::endl;

        // Test field statistics
        double min_temp = field.getMinTemperature();
        double max_temp = field.getMaxTemperature();
        double avg_temp = field.getAverageTemperature();

        std::cout << "Temperature range: " << min_temp << " - " << max_temp << " K" << std::endl;
        std::cout << "Average temperature: " << avg_temp << " K" << std::endl;

        if (max_temp < 400.0 || min_temp > 300.0) {
            std::cout << "❌ Temperature statistics incorrect" << std::endl;
            return false;
        }

        std::cout << "✅ Thermal Field test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Thermal Field test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testHeatConductionSolver() {
    std::cout << "Testing Heat Conduction Solver..." << std::endl;

    try {
        // Create thermal field with temperature gradient
        int3 dims{20, 20, 20};
        Vec3<double> cell_size{0.05, 0.05, 0.05};
        Vec3<double> origin{0.0, 0.0, 0.0};

        thermal::ThermalField<double> field(dims, cell_size, origin);

        // Set up initial temperature distribution (hot center, cold edges)
        for (int k = 0; k < dims.z; ++k) {
            for (int j = 0; j < dims.y; ++j) {
                for (int i = 0; i < dims.x; ++i) {
                    double distance_from_center = std::sqrt(
                        (i - dims.x/2) * (i - dims.x/2) +
                        (j - dims.y/2) * (j - dims.y/2) +
                        (k - dims.z/2) * (k - dims.z/2)
                    );

                    double temperature = 500.0 - distance_from_center * 10.0;
                    temperature = std::max(temperature, 273.15);  // Don't go below freezing

                    field.setTemperature(i, j, k, temperature);
                }
            }
        }

        // Add uniform material
        thermal::ThermalMaterialLibrary<double> library;
        auto steel = library.createSteel();
        field.addMaterial(steel);

        for (size_t node = 0; node < field.getTotalNodes(); ++node) {
            field.setMaterialID(node, 0);
        }

        field.updateThermalProperties();

        double initial_energy = 0.0;
        const auto& temp_data = field.getTemperatureData();
        for (size_t i = 0; i < field.getTotalNodes(); ++i) {
            initial_energy += temp_data[i];
        }

        std::cout << "Initial total temperature: " << initial_energy << " K" << std::endl;
        std::cout << "Initial center temperature: " << field.getTemperature(dims.x/2, dims.y/2, dims.z/2) << " K" << std::endl;

        // Create and run solver
        thermal::HeatConductionSolver<double> solver(&field);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Run multiple time steps
        double dt = 0.0001;  // Small time step for stability
        for (int step = 0; step < 100; ++step) {
            solver.solveExplicit(dt);

            if (step % 25 == 0) {
                double center_temp = field.getTemperature(dims.x/2, dims.y/2, dims.z/2);
                double avg_temp = field.getAverageTemperature();
                std::cout << "Step " << step << " - Center: " << center_temp << " K, Average: " << avg_temp << " K" << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        double final_energy = 0.0;
        for (size_t i = 0; i < field.getTotalNodes(); ++i) {
            final_energy += temp_data[i];
        }

        double energy_change = std::abs(final_energy - initial_energy) / initial_energy;
        double final_center_temp = field.getTemperature(dims.x/2, dims.y/2, dims.z/2);

        std::cout << "Final center temperature: " << final_center_temp << " K" << std::endl;
        std::cout << "Energy conservation error: " << energy_change * 100 << "%" << std::endl;
        std::cout << "Simulation time: " << duration.count() << " ms" << std::endl;

        // Validate heat diffusion
        if (final_center_temp >= field.getTemperature(dims.x/2, dims.y/2, dims.z/2)) {
            std::cout << "❌ Heat should have diffused from center" << std::endl;
            return false;
        }

        if (energy_change > 0.1) {  // 10% energy change is concerning
            std::cout << "❌ Excessive energy change during diffusion" << std::endl;
            return false;
        }

        std::cout << "✅ Heat Conduction Solver test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Heat Conduction Solver test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testConvectiveHeatTransfer() {
    std::cout << "Testing Convective Heat Transfer..." << std::endl;

    try {
        // Create thermal field
        int3 dims{10, 10, 10};
        Vec3<double> cell_size{0.1, 0.1, 0.1};
        Vec3<double> origin{0.0, 0.0, 0.0};

        thermal::ThermalField<double> field(dims, cell_size, origin);

        // Set high initial temperature
        for (int k = 0; k < dims.z; ++k) {
            for (int j = 0; j < dims.y; ++j) {
                for (int i = 0; i < dims.x; ++i) {
                    field.setTemperature(i, j, k, 400.0);  // Hot initial temperature
                }
            }
        }

        // Add material
        thermal::ThermalMaterialLibrary<double> library;
        auto steel = library.createSteel();
        field.addMaterial(steel);

        for (size_t node = 0; node < field.getTotalNodes(); ++node) {
            field.setMaterialID(node, 0);
        }

        field.updateThermalProperties();

        double initial_avg = field.getAverageTemperature();
        std::cout << "Initial average temperature: " << initial_avg << " K" << std::endl;

        // Apply convective heat transfer
        thermal::ConvectiveHeatTransfer<double> convection(&field);

        double ambient_temp = 293.15;  // Room temperature
        double heat_transfer_coeff = 50.0;  // W/(m²·K)
        double dt = 0.001;

        for (int step = 0; step < 100; ++step) {
            convection.applyConvection(dt, ambient_temp, heat_transfer_coeff);

            if (step % 25 == 0) {
                double avg_temp = field.getAverageTemperature();
                std::cout << "Step " << step << " - Average temperature: " << avg_temp << " K" << std::endl;
            }
        }

        double final_avg = field.getAverageTemperature();
        std::cout << "Final average temperature: " << final_avg << " K" << std::endl;

        // Validate cooling effect
        if (final_avg >= initial_avg) {
            std::cout << "❌ Convection should have cooled the system" << std::endl;
            return false;
        }

        double temperature_drop = initial_avg - final_avg;
        if (temperature_drop < 1.0) {
            std::cout << "❌ Insufficient cooling from convection" << std::endl;
            return false;
        }

        std::cout << "Temperature drop: " << temperature_drop << " K" << std::endl;
        std::cout << "✅ Convective Heat Transfer test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Convective Heat Transfer test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testPhaseChangeHandler() {
    std::cout << "Testing Phase Change Handler..." << std::endl;

    try {
        // Create thermal field
        int3 dims{5, 5, 5};
        Vec3<double> cell_size{0.1, 0.1, 0.1};
        Vec3<double> origin{0.0, 0.0, 0.0};

        thermal::ThermalField<double> field(dims, cell_size, origin);

        // Set up water at different temperatures
        thermal::ThermalMaterialLibrary<double> library;
        auto water = library.createWater();
        field.addMaterial(water);

        // Ice temperature
        field.setTemperature(0, 0, 0, 250.0);  // Below freezing
        field.setMaterialID(field.getNodeIndex(0, 0, 0), 0);

        // Water temperature
        field.setTemperature(2, 2, 2, 300.0);  // Above freezing, below boiling
        field.setMaterialID(field.getNodeIndex(2, 2, 2), 0);

        // Steam temperature
        field.setTemperature(4, 4, 4, 400.0);  // Above boiling
        field.setMaterialID(field.getNodeIndex(4, 4, 4), 0);

        field.updateThermalProperties();

        // Test phase state determination
        auto ice_phase = water.getPhaseState(250.0);
        auto water_phase = water.getPhaseState(300.0);
        auto steam_phase = water.getPhaseState(400.0);

        std::cout << "Ice phase (250K): " << static_cast<int>(ice_phase) << std::endl;
        std::cout << "Water phase (300K): " << static_cast<int>(water_phase) << std::endl;
        std::cout << "Steam phase (400K): " << static_cast<int>(steam_phase) << std::endl;

        // Validate phase states
        if (ice_phase != thermal::ThermalMaterial<double>::PhaseState::SOLID) {
            std::cout << "❌ Ice should be solid" << std::endl;
            return false;
        }

        if (water_phase != thermal::ThermalMaterial<double>::PhaseState::LIQUID) {
            std::cout << "❌ Water should be liquid" << std::endl;
            return false;
        }

        if (steam_phase != thermal::ThermalMaterial<double>::PhaseState::GAS) {
            std::cout << "❌ Steam should be gas" << std::endl;
            return false;
        }

        // Test phase change handler
        thermal::PhaseChangeHandler<double> phase_handler(&field);

        double dt = 0.001;
        for (int step = 0; step < 10; ++step) {
            phase_handler.applyPhaseChange(dt);
        }

        std::cout << "✅ Phase Change Handler test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Phase Change Handler test failed: " << e.what() << std::endl;
        return false;
    }
}

bool testThermalSimulationSystem() {
    std::cout << "Testing Complete Thermal Simulation System..." << std::endl;

    try {
        // Create thermal simulation
        int3 dims{15, 15, 15};
        Vec3<double> cell_size{0.1, 0.1, 0.1};
        Vec3<double> origin{0.0, 0.0, 0.0};

        thermal::ThermalSimulationSystem<double> sim(dims, cell_size, origin);

        // Set up materials
        thermal::ThermalMaterialLibrary<double> library;
        auto steel = library.createSteel();
        auto water = library.createWater();

        auto& field = sim.getThermalField();
        field.addMaterial(steel);
        field.addMaterial(water);

        // Create a hot steel block in water
        for (int k = 0; k < dims.z; ++k) {
            for (int j = 0; j < dims.y; ++j) {
                for (int i = 0; i < dims.x; ++i) {
                    bool is_steel_block = (i >= 6 && i <= 8 && j >= 6 && j <= 8 && k >= 6 && k <= 8);

                    if (is_steel_block) {
                        field.setTemperature(i, j, k, 800.0);  // Hot steel
                        field.setMaterialID(field.getNodeIndex(i, j, k), 0);  // Steel
                    } else {
                        field.setTemperature(i, j, k, 293.15);  // Room temperature water
                        field.setMaterialID(field.getNodeIndex(i, j, k), 1);  // Water
                    }
                }
            }
        }

        // Configure simulation
        sim.setAmbientTemperature(273.15);  // Cold environment
        sim.setConvectionCoefficient(25.0);
        sim.enableConvection(true);
        sim.enableRadiation(true);
        sim.enablePhaseChange(true);

        double initial_energy = sim.getTotalThermalEnergy();
        double initial_steel_temp = field.getTemperature(7, 7, 7);  // Center of steel block

        std::cout << "Initial thermal energy: " << initial_energy << " J" << std::endl;
        std::cout << "Initial steel temperature: " << initial_steel_temp << " K" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Run simulation
        double dt = 0.0001;
        for (int step = 0; step < 200; ++step) {
            sim.simulationStep(dt);

            if (step % 50 == 0) {
                double steel_temp = field.getTemperature(7, 7, 7);
                double water_temp = field.getTemperature(1, 1, 1);
                double avg_temp = field.getAverageTemperature();

                std::cout << "Step " << step << " - Steel: " << steel_temp
                         << " K, Water: " << water_temp
                         << " K, Average: " << avg_temp << " K" << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        double final_energy = sim.getTotalThermalEnergy();
        double final_steel_temp = field.getTemperature(7, 7, 7);
        double energy_loss = (initial_energy - final_energy) / initial_energy;

        std::cout << "Final thermal energy: " << final_energy << " J" << std::endl;
        std::cout << "Final steel temperature: " << final_steel_temp << " K" << std::endl;
        std::cout << "Energy loss: " << energy_loss * 100 << "%" << std::endl;
        std::cout << "Simulation time: " << duration.count() << " ms" << std::endl;

        // Validate simulation results
        if (final_steel_temp >= initial_steel_temp) {
            std::cout << "❌ Steel should have cooled down" << std::endl;
            return false;
        }

        if (energy_loss < 0.01) {
            std::cout << "❌ System should have lost energy to environment" << std::endl;
            return false;
        }

        if (energy_loss > 0.5) {
            std::cout << "❌ Excessive energy loss" << std::endl;
            return false;
        }

        std::cout << "✅ Thermal Simulation System test passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "❌ Thermal Simulation System test failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== Thermal Physics Test Suite ===" << std::endl;

    bool all_passed = true;

    all_passed &= testThermalMaterialDatabase();
    all_passed &= testThermalField();
    all_passed &= testHeatConductionSolver();
    all_passed &= testConvectiveHeatTransfer();
    all_passed &= testPhaseChangeHandler();
    all_passed &= testThermalSimulationSystem();

    std::cout << "\n=== Thermal Physics Test Summary ===" << std::endl;
    if (all_passed) {
        std::cout << "✅ All thermal physics tests passed!" << std::endl;
        std::cout << "\nThermal Physics Implementation Features:" << std::endl;
        std::cout << "• Temperature-dependent material properties" << std::endl;
        std::cout << "• Heat conduction with finite difference solvers" << std::endl;
        std::cout << "• Convective and radiative heat transfer" << std::endl;
        std::cout << "• Phase change handling (melting, boiling)" << std::endl;
        std::cout << "• Thermal-mechanical coupling capabilities" << std::endl;
        std::cout << "• Comprehensive simulation system" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some thermal physics tests failed!" << std::endl;
        return 1;
    }
}