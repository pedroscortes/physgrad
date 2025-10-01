/**
 * PhysGrad - Thermal Physics and Heat Transfer
 *
 * Implements comprehensive thermal physics simulation including heat conduction,
 * convection, radiation, and phase changes. Supports coupled thermal-mechanical
 * simulations for temperature-dependent material properties.
 */

#ifndef PHYSGRAD_THERMAL_PHYSICS_H
#define PHYSGRAD_THERMAL_PHYSICS_H

#include "common_types.h"
#include "material_point_method.h"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <functional>
#include <unordered_map>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define PHYSGRAD_DEVICE __device__
    #define PHYSGRAD_HOST_DEVICE __host__ __device__
    #define PHYSGRAD_GLOBAL __global__
#else
    #define PHYSGRAD_DEVICE
    #define PHYSGRAD_HOST_DEVICE
    #define PHYSGRAD_GLOBAL
#endif

// Include concepts when available
#ifdef PHYSGRAD_CONCEPTS_AVAILABLE
    #include "concepts/forward_declarations.h"
#endif

namespace physgrad {
namespace thermal {

    // =============================================================================
    // THERMAL MATERIAL PROPERTIES
    // =============================================================================

    /**
     * Temperature-dependent material properties
     */
    template<typename T>
    struct ThermalMaterial {
        // Basic thermal properties
        T thermal_conductivity;          // W/(m·K)
        T specific_heat_capacity;        // J/(kg·K)
        T density;                       // kg/m³
        T emissivity;                    // Surface emissivity (0-1)
        T absorptivity;                  // Surface absorptivity (0-1)

        // Temperature-dependent properties
        std::function<T(T)> conductivity_curve;      // k(T)
        std::function<T(T)> heat_capacity_curve;     // cp(T)
        std::function<T(T)> density_curve;           // ρ(T)

        // Phase change properties
        bool has_phase_change = false;
        T melting_point = T{0};          // K
        T boiling_point = T{0};          // K
        T latent_heat_fusion = T{0};     // J/kg
        T latent_heat_vaporization = T{0}; // J/kg

        // Thermal expansion
        T thermal_expansion_coefficient = T{0}; // 1/K
        ConceptVector3D<T> reference_dimensions{T{1}, T{1}, T{1}};

        ThermalMaterial() :
            thermal_conductivity(T{1}),        // Default: poor conductor
            specific_heat_capacity(T{1000}),   // Default: water-like
            density(T{1000}),                  // Default: water density
            emissivity(T{0.9}),               // Default: high emissivity
            absorptivity(T{0.9}) {            // Default: high absorptivity

            // Default temperature-independent properties
            conductivity_curve = [this](T temp) { return thermal_conductivity; };
            heat_capacity_curve = [this](T temp) { return specific_heat_capacity; };
            density_curve = [this](T temp) { return density; };
        }

        // Get effective properties at given temperature
        PHYSGRAD_HOST_DEVICE T getEffectiveConductivity(T temperature) const {
            return conductivity_curve ? conductivity_curve(temperature) : thermal_conductivity;
        }

        PHYSGRAD_HOST_DEVICE T getEffectiveHeatCapacity(T temperature) const {
            return heat_capacity_curve ? heat_capacity_curve(temperature) : specific_heat_capacity;
        }

        PHYSGRAD_HOST_DEVICE T getEffectiveDensity(T temperature) const {
            return density_curve ? density_curve(temperature) : density;
        }

        // Phase state determination
        enum class PhaseState { SOLID, LIQUID, GAS };

        PHYSGRAD_HOST_DEVICE PhaseState getPhaseState(T temperature) const {
            if (!has_phase_change) return PhaseState::SOLID;
            if (temperature < melting_point) return PhaseState::SOLID;
            if (temperature < boiling_point) return PhaseState::LIQUID;
            return PhaseState::GAS;
        }
    };

    /**
     * Predefined thermal materials library
     */
    template<typename T>
    class ThermalMaterialLibrary {
    public:
        static ThermalMaterial<T> createSteel() {
            ThermalMaterial<T> steel;
            steel.thermal_conductivity = T{50};     // W/(m·K)
            steel.specific_heat_capacity = T{500};  // J/(kg·K)
            steel.density = T{7850};                // kg/m³
            steel.emissivity = T{0.8};
            steel.melting_point = T{1811};          // K
            steel.latent_heat_fusion = T{270000};   // J/kg
            steel.has_phase_change = true;
            steel.thermal_expansion_coefficient = T{12e-6}; // 1/K
            return steel;
        }

        static ThermalMaterial<T> createAluminum() {
            ThermalMaterial<T> aluminum;
            aluminum.thermal_conductivity = T{205}; // W/(m·K)
            aluminum.specific_heat_capacity = T{900}; // J/(kg·K)
            aluminum.density = T{2700};             // kg/m³
            aluminum.emissivity = T{0.09};
            aluminum.melting_point = T{933};        // K
            aluminum.latent_heat_fusion = T{397000}; // J/kg
            aluminum.has_phase_change = true;
            aluminum.thermal_expansion_coefficient = T{23e-6}; // 1/K
            return aluminum;
        }

        static ThermalMaterial<T> createWater() {
            ThermalMaterial<T> water;
            water.thermal_conductivity = T{0.6};    // W/(m·K)
            water.specific_heat_capacity = T{4182}; // J/(kg·K)
            water.density = T{1000};                // kg/m³
            water.emissivity = T{0.96};
            water.melting_point = T{273.15};        // K
            water.boiling_point = T{373.15};        // K
            water.latent_heat_fusion = T{334000};   // J/kg
            water.latent_heat_vaporization = T{2260000}; // J/kg
            water.has_phase_change = true;
            water.thermal_expansion_coefficient = T{214e-6}; // 1/K
            return water;
        }

        static ThermalMaterial<T> createAir() {
            ThermalMaterial<T> air;
            air.thermal_conductivity = T{0.026};    // W/(m·K)
            air.specific_heat_capacity = T{1005};   // J/(kg·K)
            air.density = T{1.225};                 // kg/m³
            air.emissivity = T{0};                  // Transparent
            air.thermal_expansion_coefficient = T{3.43e-3}; // 1/K (ideal gas)
            return air;
        }
    };

    // =============================================================================
    // THERMAL FIELD REPRESENTATION
    // =============================================================================

    /**
     * 3D thermal field with temperature distribution
     */
    template<typename T>
    class ThermalField {
    private:
        int3 grid_dims_;
        ConceptVector3D<T> cell_size_;
        ConceptVector3D<T> origin_;
        size_t total_nodes_;

        std::vector<T> temperature_;         // Current temperature field
        std::vector<T> temperature_old_;     // Previous timestep temperature
        std::vector<T> heat_source_;         // Heat source term (W/m³)
        std::vector<T> thermal_diffusivity_; // α = k/(ρ·cp) (m²/s)

        // Material assignment per grid cell
        std::vector<int> material_id_;
        std::vector<ThermalMaterial<T>> materials_;

    public:
        ThermalField(const int3& dims, const ConceptVector3D<T>& cell_size, const ConceptVector3D<T>& origin)
            : grid_dims_(dims), cell_size_(cell_size), origin_(origin) {
            total_nodes_ = static_cast<size_t>(dims.x) * dims.y * dims.z;

            temperature_.resize(total_nodes_, T{293.15}); // Room temperature
            temperature_old_.resize(total_nodes_, T{293.15});
            heat_source_.resize(total_nodes_, T{0});
            thermal_diffusivity_.resize(total_nodes_, T{1e-6}); // Default diffusivity
            material_id_.resize(total_nodes_, 0);
        }

        // Grid access
        PHYSGRAD_HOST_DEVICE size_t getNodeIndex(int i, int j, int k) const {
            return static_cast<size_t>(k) * grid_dims_.x * grid_dims_.y +
                   static_cast<size_t>(j) * grid_dims_.x + i;
        }

        PHYSGRAD_HOST_DEVICE ConceptVector3D<T> getNodePosition(int i, int j, int k) const {
            return ConceptVector3D<T>{
                origin_[0] + i * cell_size_[0],
                origin_[1] + j * cell_size_[1],
                origin_[2] + k * cell_size_[2]
            };
        }

        // Temperature field access
        PHYSGRAD_HOST_DEVICE T getTemperature(size_t node_id) const {
            return temperature_[node_id];
        }

        PHYSGRAD_HOST_DEVICE void setTemperature(size_t node_id, T temp) {
            temperature_[node_id] = temp;
        }

        PHYSGRAD_HOST_DEVICE T getTemperature(int i, int j, int k) const {
            return temperature_[getNodeIndex(i, j, k)];
        }

        PHYSGRAD_HOST_DEVICE void setTemperature(int i, int j, int k, T temp) {
            temperature_[getNodeIndex(i, j, k)] = temp;
        }

        // Heat source access
        PHYSGRAD_HOST_DEVICE T getHeatSource(size_t node_id) const {
            return heat_source_[node_id];
        }

        PHYSGRAD_HOST_DEVICE void setHeatSource(size_t node_id, T source) {
            heat_source_[node_id] = source;
        }

        // Material access
        void addMaterial(const ThermalMaterial<T>& material) {
            materials_.push_back(material);
        }

        PHYSGRAD_HOST_DEVICE const ThermalMaterial<T>& getMaterial(size_t node_id) const {
            int mat_id = material_id_[node_id];
            return materials_[mat_id];
        }

        PHYSGRAD_HOST_DEVICE void setMaterialID(size_t node_id, int material_id) {
            material_id_[node_id] = material_id;
        }

        // Thermal diffusivity access
        PHYSGRAD_HOST_DEVICE T getThermalDiffusivity(size_t node_id) const {
            return thermal_diffusivity_[node_id];
        }

        PHYSGRAD_HOST_DEVICE void setThermalDiffusivity(size_t node_id, T diffusivity) {
            thermal_diffusivity_[node_id] = diffusivity;
        }

        // Update thermal properties based on current temperature
        void updateThermalProperties() {
            for (size_t i = 0; i < total_nodes_; ++i) {
                if (material_id_[i] < materials_.size()) {
                    const auto& material = materials_[material_id_[i]];
                    T temp = temperature_[i];

                    T k = material.getEffectiveConductivity(temp);
                    T rho = material.getEffectiveDensity(temp);
                    T cp = material.getEffectiveHeatCapacity(temp);

                    thermal_diffusivity_[i] = k / (rho * cp);
                }
            }
        }

        // Backup and restore temperature field
        void backupTemperatureField() {
            temperature_old_ = temperature_;
        }

        void restoreTemperatureField() {
            temperature_ = temperature_old_;
        }

        // Grid properties
        int3 getDimensions() const { return grid_dims_; }
        ConceptVector3D<T> getCellSize() const { return cell_size_; }
        ConceptVector3D<T> getOrigin() const { return origin_; }
        size_t getTotalNodes() const { return total_nodes_; }

        // Field statistics
        T getMinTemperature() const {
            return *std::min_element(temperature_.begin(), temperature_.end());
        }

        T getMaxTemperature() const {
            return *std::max_element(temperature_.begin(), temperature_.end());
        }

        T getAverageTemperature() const {
            T sum = T{0};
            for (T temp : temperature_) {
                sum += temp;
            }
            return sum / static_cast<T>(temperature_.size());
        }

        // Direct access to internal data (for efficient computation)
        std::vector<T>& getTemperatureData() { return temperature_; }
        const std::vector<T>& getTemperatureData() const { return temperature_; }
        std::vector<T>& getHeatSourceData() { return heat_source_; }
        const std::vector<T>& getHeatSourceData() const { return heat_source_; }
    };

    // =============================================================================
    // HEAT TRANSFER MECHANISMS
    // =============================================================================

    /**
     * Heat conduction solver using finite differences
     */
    template<typename T>
    class HeatConductionSolver {
    private:
        ThermalField<T>* field_;
        std::vector<T> temperature_buffer_; // For explicit schemes

    public:
        explicit HeatConductionSolver(ThermalField<T>* field) : field_(field) {
            temperature_buffer_.resize(field->getTotalNodes());
        }

        // Explicit finite difference scheme (forward Euler)
        void solveExplicit(T dt, T max_cfl = T{0.5}) {
            if (!field_) return;

            auto dims = field_->getDimensions();
            auto cell_size = field_->getCellSize();

            field_->backupTemperatureField();
            auto& temp_data = field_->getTemperatureData();
            auto& source_data = field_->getHeatSourceData();

            // Copy current temperature to buffer
            temperature_buffer_ = temp_data;

            // Stability check: dt ≤ CFL * min(dx²,dy²,dz²) / (2 * max_diffusivity)
            T min_cell_size_sq = std::min({cell_size[0]*cell_size[0],
                                          cell_size[1]*cell_size[1],
                                          cell_size[2]*cell_size[2]});

            T max_diffusivity = T{0};
            for (size_t i = 0; i < field_->getTotalNodes(); ++i) {
                max_diffusivity = std::max(max_diffusivity, field_->getThermalDiffusivity(i));
            }

            T stable_dt = max_cfl * min_cell_size_sq / (T{6} * max_diffusivity);
            if (dt > stable_dt) {
                dt = stable_dt; // Clamp to stable time step
            }

            // Apply heat conduction equation: ∂T/∂t = α∇²T + S/(ρcp)
            for (int k = 1; k < dims.z - 1; ++k) {
                for (int j = 1; j < dims.y - 1; ++j) {
                    for (int i = 1; i < dims.x - 1; ++i) {
                        size_t idx = field_->getNodeIndex(i, j, k);
                        T alpha = field_->getThermalDiffusivity(idx);

                        // Central differences for Laplacian
                        T T_center = temperature_buffer_[idx];

                        // x-direction
                        size_t idx_xp = field_->getNodeIndex(i+1, j, k);
                        size_t idx_xm = field_->getNodeIndex(i-1, j, k);
                        T d2T_dx2 = (temperature_buffer_[idx_xp] - 2*T_center + temperature_buffer_[idx_xm]) /
                                   (cell_size[0] * cell_size[0]);

                        // y-direction
                        size_t idx_yp = field_->getNodeIndex(i, j+1, k);
                        size_t idx_ym = field_->getNodeIndex(i, j-1, k);
                        T d2T_dy2 = (temperature_buffer_[idx_yp] - 2*T_center + temperature_buffer_[idx_ym]) /
                                   (cell_size[1] * cell_size[1]);

                        // z-direction
                        size_t idx_zp = field_->getNodeIndex(i, j, k+1);
                        size_t idx_zm = field_->getNodeIndex(i, j, k-1);
                        T d2T_dz2 = (temperature_buffer_[idx_zp] - 2*T_center + temperature_buffer_[idx_zm]) /
                                   (cell_size[2] * cell_size[2]);

                        // Laplacian
                        T laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2;

                        // Heat source term
                        const auto& material = field_->getMaterial(idx);
                        T source_term = source_data[idx] / (material.getEffectiveDensity(T_center) *
                                                          material.getEffectiveHeatCapacity(T_center));

                        // Update temperature
                        temp_data[idx] = T_center + dt * (alpha * laplacian + source_term);
                    }
                }
            }
        }

        // Implicit finite difference scheme (backward Euler) - simplified version
        void solveImplicit(T dt) {
            // This would require solving a large linear system
            // For now, use a simplified approach with multiple explicit steps
            int sub_steps = 10;
            T sub_dt = dt / sub_steps;

            for (int step = 0; step < sub_steps; ++step) {
                solveExplicit(sub_dt, T{0.9});
            }
        }
    };

    /**
     * Convective heat transfer (simplified)
     */
    template<typename T>
    class ConvectiveHeatTransfer {
    private:
        ThermalField<T>* field_;

    public:
        explicit ConvectiveHeatTransfer(ThermalField<T>* field) : field_(field) {}

        // Apply convective heat transfer at boundaries
        void applyConvection(T dt, T ambient_temperature, T heat_transfer_coefficient) {
            if (!field_) return;

            auto dims = field_->getDimensions();
            auto& temp_data = field_->getTemperatureData();
            auto& source_data = field_->getHeatSourceData();

            // Apply to boundary nodes
            for (int k = 0; k < dims.z; ++k) {
                for (int j = 0; j < dims.y; ++j) {
                    for (int i = 0; i < dims.x; ++i) {
                        // Check if this is a boundary node
                        bool is_boundary = (i == 0 || i == dims.x-1 ||
                                          j == 0 || j == dims.y-1 ||
                                          k == 0 || k == dims.z-1);

                        if (is_boundary) {
                            size_t idx = field_->getNodeIndex(i, j, k);
                            T surface_temp = temp_data[idx];

                            // Newton's law of cooling: q = h(T_surface - T_ambient)
                            T heat_flux = heat_transfer_coefficient * (surface_temp - ambient_temperature);

                            // Convert to volumetric heat source (W/m³)
                            // This is a simplification - proper implementation would consider surface area
                            T cell_volume = field_->getCellSize()[0] * field_->getCellSize()[1] * field_->getCellSize()[2];
                            T volumetric_source = -heat_flux / cell_volume;

                            source_data[idx] += volumetric_source;
                        }
                    }
                }
            }
        }
    };

    /**
     * Radiative heat transfer
     */
    template<typename T>
    class RadiativeHeatTransfer {
    private:
        ThermalField<T>* field_;
        static constexpr T STEFAN_BOLTZMANN = T{5.670374419e-8}; // W/(m²·K⁴)

    public:
        explicit RadiativeHeatTransfer(ThermalField<T>* field) : field_(field) {}

        // Apply Stefan-Boltzmann radiation at boundaries
        void applyRadiation(T dt, T ambient_temperature) {
            if (!field_) return;

            auto dims = field_->getDimensions();
            auto& temp_data = field_->getTemperatureData();
            auto& source_data = field_->getHeatSourceData();

            for (int k = 0; k < dims.z; ++k) {
                for (int j = 0; j < dims.y; ++j) {
                    for (int i = 0; i < dims.x; ++i) {
                        bool is_boundary = (i == 0 || i == dims.x-1 ||
                                          j == 0 || j == dims.y-1 ||
                                          k == 0 || k == dims.z-1);

                        if (is_boundary) {
                            size_t idx = field_->getNodeIndex(i, j, k);
                            const auto& material = field_->getMaterial(idx);
                            T surface_temp = temp_data[idx];

                            // Stefan-Boltzmann law: q = εσ(T⁴ - T_amb⁴)
                            T T4_surface = surface_temp * surface_temp * surface_temp * surface_temp;
                            T T4_ambient = ambient_temperature * ambient_temperature *
                                         ambient_temperature * ambient_temperature;

                            T heat_flux = material.emissivity * STEFAN_BOLTZMANN * (T4_surface - T4_ambient);

                            // Convert to volumetric heat source
                            T cell_volume = field_->getCellSize()[0] * field_->getCellSize()[1] * field_->getCellSize()[2];
                            T volumetric_source = -heat_flux / cell_volume;

                            source_data[idx] += volumetric_source;
                        }
                    }
                }
            }
        }
    };

    // =============================================================================
    // PHASE CHANGE MODELING
    // =============================================================================

    /**
     * Phase change handler for melting/solidification and boiling/condensation
     */
    template<typename T>
    class PhaseChangeHandler {
    private:
        ThermalField<T>* field_;
        std::vector<T> latent_heat_stored_; // Energy stored during phase change

    public:
        explicit PhaseChangeHandler(ThermalField<T>* field) : field_(field) {
            latent_heat_stored_.resize(field->getTotalNodes(), T{0});
        }

        // Apply phase change effects
        void applyPhaseChange(T dt) {
            if (!field_) return;

            auto& temp_data = field_->getTemperatureData();

            for (size_t i = 0; i < field_->getTotalNodes(); ++i) {
                const auto& material = field_->getMaterial(i);
                if (!material.has_phase_change) continue;

                T current_temp = temp_data[i];
                auto current_phase = material.getPhaseState(current_temp);

                // Handle melting/solidification
                if (std::abs(current_temp - material.melting_point) < T{1.0}) {
                    handleMeltingSolidification(i, material, dt);
                }

                // Handle boiling/condensation
                if (material.boiling_point > T{0} &&
                    std::abs(current_temp - material.boiling_point) < T{1.0}) {
                    handleBoilingCondensation(i, material, dt);
                }
            }
        }

    private:
        void handleMeltingSolidification(size_t node_id, const ThermalMaterial<T>& material, T dt) {
            auto& temp_data = field_->getTemperatureData();
            T temp = temp_data[node_id];

            // If temperature crosses melting point, apply latent heat
            if (temp > material.melting_point && latent_heat_stored_[node_id] < material.latent_heat_fusion) {
                // Absorb latent heat for melting
                T energy_available = material.getEffectiveDensity(temp) *
                                   material.getEffectiveHeatCapacity(temp) *
                                   (temp - material.melting_point);

                T latent_heat_needed = material.latent_heat_fusion - latent_heat_stored_[node_id];
                T energy_absorbed = std::min(energy_available, latent_heat_needed);

                latent_heat_stored_[node_id] += energy_absorbed;
                temp_data[node_id] = material.melting_point; // Clamp temperature during phase change

            } else if (temp < material.melting_point && latent_heat_stored_[node_id] > T{0}) {
                // Release latent heat for solidification
                T energy_deficit = material.getEffectiveDensity(temp) *
                                 material.getEffectiveHeatCapacity(temp) *
                                 (material.melting_point - temp);

                T energy_released = std::min(latent_heat_stored_[node_id], energy_deficit);
                latent_heat_stored_[node_id] -= energy_released;
                temp_data[node_id] = material.melting_point; // Clamp temperature during phase change
            }
        }

        void handleBoilingCondensation(size_t node_id, const ThermalMaterial<T>& material, T dt) {
            // Similar logic to melting/solidification but for liquid-gas transition
            auto& temp_data = field_->getTemperatureData();
            T temp = temp_data[node_id];

            // Simplified implementation - clamp temperature at boiling point
            if (temp > material.boiling_point) {
                temp_data[node_id] = material.boiling_point;
            }
        }
    };

    // =============================================================================
    // THERMAL-MECHANICAL COUPLING
    // =============================================================================

    /**
     * Thermal stress computation for coupled simulations
     */
    template<typename T>
    class ThermalStressSolver {
    public:
        // Compute thermal strain tensor
        static ConceptVector3D<T> computeThermalStrain(const ThermalMaterial<T>& material,
                                                      T current_temp, T reference_temp) {
            T delta_T = current_temp - reference_temp;
            T thermal_strain = material.thermal_expansion_coefficient * delta_T;
            return ConceptVector3D<T>{thermal_strain, thermal_strain, thermal_strain};
        }

        // Compute thermal stress (requires elastic moduli)
        static ConceptVector3D<T> computeThermalStress(const ThermalMaterial<T>& material,
                                                      T current_temp, T reference_temp,
                                                      T elastic_modulus, T poisson_ratio) {
            T delta_T = current_temp - reference_temp;
            T thermal_stress_magnitude = -elastic_modulus * material.thermal_expansion_coefficient * delta_T /
                                        (T{1} - T{2} * poisson_ratio);
            return ConceptVector3D<T>{thermal_stress_magnitude, thermal_stress_magnitude, thermal_stress_magnitude};
        }
    };

    // =============================================================================
    // COMPLETE THERMAL SIMULATION SYSTEM
    // =============================================================================

    /**
     * Comprehensive thermal simulation system
     */
    template<typename T>
    class ThermalSimulationSystem {
    private:
        std::unique_ptr<ThermalField<T>> thermal_field_;
        std::unique_ptr<HeatConductionSolver<T>> conduction_solver_;
        std::unique_ptr<ConvectiveHeatTransfer<T>> convection_solver_;
        std::unique_ptr<RadiativeHeatTransfer<T>> radiation_solver_;
        std::unique_ptr<PhaseChangeHandler<T>> phase_change_handler_;

        // Simulation parameters
        T ambient_temperature_ = T{293.15}; // K
        T convection_coefficient_ = T{10};  // W/(m²·K)
        bool enable_convection_ = true;
        bool enable_radiation_ = true;
        bool enable_phase_change_ = true;

        // Time integration
        T current_time_ = T{0};
        size_t current_step_ = 0;

    public:
        ThermalSimulationSystem(const int3& grid_dims,
                              const ConceptVector3D<T>& cell_size,
                              const ConceptVector3D<T>& origin) {

            thermal_field_ = std::make_unique<ThermalField<T>>(grid_dims, cell_size, origin);
            conduction_solver_ = std::make_unique<HeatConductionSolver<T>>(thermal_field_.get());
            convection_solver_ = std::make_unique<ConvectiveHeatTransfer<T>>(thermal_field_.get());
            radiation_solver_ = std::make_unique<RadiativeHeatTransfer<T>>(thermal_field_.get());
            phase_change_handler_ = std::make_unique<PhaseChangeHandler<T>>(thermal_field_.get());
        }

        // Access to thermal field
        ThermalField<T>& getThermalField() { return *thermal_field_; }
        const ThermalField<T>& getThermalField() const { return *thermal_field_; }

        // Simulation control
        void setAmbientTemperature(T temp) { ambient_temperature_ = temp; }
        void setConvectionCoefficient(T coeff) { convection_coefficient_ = coeff; }
        void enableConvection(bool enable) { enable_convection_ = enable; }
        void enableRadiation(bool enable) { enable_radiation_ = enable; }
        void enablePhaseChange(bool enable) { enable_phase_change_ = enable; }

        // Main simulation step
        void simulationStep(T dt) {
            // Clear heat sources from previous step
            auto& source_data = thermal_field_->getHeatSourceData();
            std::fill(source_data.begin(), source_data.end(), T{0});

            // Update thermal properties based on current temperature
            thermal_field_->updateThermalProperties();

            // Apply boundary conditions
            if (enable_convection_) {
                convection_solver_->applyConvection(dt, ambient_temperature_, convection_coefficient_);
            }

            if (enable_radiation_) {
                radiation_solver_->applyRadiation(dt, ambient_temperature_);
            }

            // Solve heat conduction
            conduction_solver_->solveExplicit(dt);

            // Apply phase change effects
            if (enable_phase_change_) {
                phase_change_handler_->applyPhaseChange(dt);
            }

            current_time_ += dt;
            current_step_++;
        }

        // Analysis functions
        T getTotalThermalEnergy() const {
            T total_energy = T{0};
            const auto& temp_data = thermal_field_->getTemperatureData();

            for (size_t i = 0; i < thermal_field_->getTotalNodes(); ++i) {
                const auto& material = thermal_field_->getMaterial(i);
                T temp = temp_data[i];
                T cell_volume = thermal_field_->getCellSize()[0] *
                              thermal_field_->getCellSize()[1] *
                              thermal_field_->getCellSize()[2];

                T heat_capacity = material.getEffectiveHeatCapacity(temp);
                T density = material.getEffectiveDensity(temp);
                total_energy += density * heat_capacity * temp * cell_volume;
            }

            return total_energy;
        }

        T getCurrentTime() const { return current_time_; }
        size_t getCurrentStep() const { return current_step_; }

        void resetSimulation() {
            current_time_ = T{0};
            current_step_ = 0;
        }
    };

} // namespace thermal
} // namespace physgrad

#endif // PHYSGRAD_THERMAL_PHYSICS_H