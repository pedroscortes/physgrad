/**
 * GPU-Accelerated Electromagnetic Field Simulation for PhysGrad
 *
 * Maxwell's equations solver with complete multi-physics coupling
 * Enables electromagnetic forces in contact mechanics and fluid dynamics
 */

#pragma once

#include <vector>
#include <memory>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "variational_contact.h"
#include "sph_fluid_dynamics.h"
#include "soft_body_dynamics.h"

namespace physgrad {

/**
 * Electromagnetic field simulation parameters
 */
struct EMParams {
    // Physical constants
    double permittivity = 8.854e-12;        // ε₀ (F/m) - vacuum permittivity
    double permeability = 4e-7 * M_PI;      // μ₀ (H/m) - vacuum permeability
    double speed_of_light = 2.998e8;        // c (m/s)
    double elementary_charge = 1.602e-19;   // e (C)

    // Grid discretization
    Eigen::Vector3d domain_min = Eigen::Vector3d(-1, -1, -1);
    Eigen::Vector3d domain_max = Eigen::Vector3d(1, 1, 1);
    Eigen::Vector3i grid_resolution = Eigen::Vector3i(64, 64, 64);
    double grid_spacing = 0.01;             // m

    // Time stepping
    double dt = 1e-12;                      // s (femtosecond timesteps)
    double cfl_factor = 0.5;                // Courant stability condition

    // Material properties
    double relative_permittivity = 1.0;     // εᵣ
    double relative_permeability = 1.0;     // μᵣ
    double conductivity = 0.0;              // σ (S/m)
    double magnetic_susceptibility = 0.0;   // χₘ

    // Boundary conditions
    bool pml_boundaries = true;             // Perfectly Matched Layer
    double pml_thickness = 0.1;             // m
    double pml_absorption = 1000.0;         // absorption coefficient

    // Source parameters
    double source_frequency = 1e9;          // Hz (GHz range)
    double source_amplitude = 1e6;          // V/m
    Eigen::Vector3d source_position = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d source_direction = Eigen::Vector3d(1, 0, 0);

    // Solver settings
    int max_iterations = 1000;
    double convergence_tolerance = 1e-8;
    bool use_adaptive_timestep = true;
    bool enable_nonlinear_materials = false;
};

/**
 * Electromagnetic field components at a grid point
 */
struct EMGridPoint {
    // Electric field components (V/m)
    Eigen::Vector3d E_field;
    Eigen::Vector3d E_field_prev;

    // Magnetic field components (T)
    Eigen::Vector3d H_field;
    Eigen::Vector3d H_field_prev;

    // Auxiliary fields for complex materials
    Eigen::Vector3d D_field;  // Electric displacement
    Eigen::Vector3d B_field;  // Magnetic induction

    // Current density and charge density
    Eigen::Vector3d J_current = Eigen::Vector3d::Zero();  // A/m²
    double rho_charge = 0.0;                              // C/m³

    // Material properties at this point
    double epsilon = 8.854e-12;  // Local permittivity
    double mu = 4e-7 * M_PI;     // Local permeability
    double sigma = 0.0;          // Local conductivity

    // PML absorption parameters
    double sx = 1.0, sy = 1.0, sz = 1.0;  // PML stretching factors
    double ax = 0.0, ay = 0.0, az = 0.0;  // PML attenuation factors

    // Grid position
    Eigen::Vector3d position;
    Eigen::Vector3i grid_index;
};

/**
 * Charged particle for electromagnetic interactions
 */
struct ChargedParticle {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;

    double charge;              // C
    double mass;               // kg

    // Forces
    Eigen::Vector3d electric_force;   // qE
    Eigen::Vector3d magnetic_force;   // q(v × B)
    Eigen::Vector3d lorentz_force;    // Total electromagnetic force

    // Radiation damping (for accelerating charges)
    Eigen::Vector3d radiation_force;
    double radiated_power = 0.0;      // W

    // Particle history for radiation computation
    std::vector<Eigen::Vector3d> position_history;
    std::vector<Eigen::Vector3d> velocity_history;
    std::vector<Eigen::Vector3d> acceleration_history;
};

/**
 * Electromagnetic wave source
 */
struct EMSource {
    enum class SourceType {
        PLANE_WAVE,        // Uniform plane wave
        DIPOLE,           // Electric/magnetic dipole
        GAUSSIAN_BEAM,    // Focused Gaussian beam
        CURRENT_LOOP,     // Current loop antenna
        POINT_CHARGE,     // Moving point charge
        CUSTOM            // User-defined source
    };

    SourceType type = SourceType::PLANE_WAVE;
    Eigen::Vector3d position;
    Eigen::Vector3d direction;
    Eigen::Vector3d polarization;

    double frequency;          // Hz
    double amplitude;          // V/m or A
    double phase = 0.0;        // rad
    double pulse_width = 1e-12; // s (for pulsed sources)

    // Time-dependent amplitude function
    std::function<double(double)> time_envelope;

    // Spatial profile for beam sources
    double beam_waist = 1e-3;  // m
    double focal_length = 0.1; // m
};

/**
 * GPU-accelerated electromagnetic field solver
 */
class ElectromagneticSolver {
public:
    explicit ElectromagneticSolver(const EMParams& params);
    ~ElectromagneticSolver();

    // Grid setup and initialization
    void initializeGrid();
    void setMaterialProperties(const Eigen::Vector3d& min_corner,
                             const Eigen::Vector3d& max_corner,
                             double epsilon_r, double mu_r, double sigma);

    // Sources and boundary conditions
    void addSource(const EMSource& source);
    void addChargedParticle(const ChargedParticle& particle);
    void setPMLBoundaries(double thickness, double absorption);
    void setPeriodicBoundaries(bool enable);

    // Time stepping - FDTD (Finite Difference Time Domain)
    void simulateStep(double dt);
    void simulateSteps(int num_steps);

    // Multi-physics integration
    void integrateWithContactSolver(VariationalContactSolver& contact_solver);
    void integrateWithFluidSolver(SPHFluidSolver& fluid_solver);
    void integrateWithSoftBodySolver(SoftBodySolver& soft_body_solver);

    // Electromagnetic force computation
    std::vector<Eigen::Vector3d> computeLorentzForces(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& charges) const;

    void computeMaxwellStressTensor();
    std::vector<Eigen::Vector3d> getElectromagneticPressure() const;

    // Field analysis and diagnostics
    double getTotalElectromagneticEnergy() const;
    double getElectricEnergy() const;
    double getMagneticEnergy() const;
    double getPoyntingFlux() const;

    std::vector<Eigen::Vector3d> getElectricField() const;
    std::vector<Eigen::Vector3d> getMagneticField() const;
    std::vector<double> getElectricPotential() const;
    std::vector<double> getMagneticPotential() const;

    // Wave propagation analysis
    std::vector<std::complex<double>> computeFourierTransform() const;
    std::vector<double> getWaveIntensity() const;
    double getWaveImpedance() const;

    // Material property modification
    void updateMaterialProperties(const std::vector<double>& epsilon_r,
                                const std::vector<double>& mu_r,
                                const std::vector<double>& sigma);

    // Nonlinear electromagnetic effects
    void enableKerrEffect(double chi3);          // Optical Kerr effect
    void enableFaradayRotation(double verdet);   // Magneto-optical effect
    void enablePlasmaEffects(double plasma_freq); // Plasma frequency

    // GPU memory management
    void allocateGPUMemory();
    void deallocateGPUMemory();
    size_t getGPUMemoryUsage() const;

private:
    EMParams params_;
    std::vector<EMGridPoint> grid_;
    std::vector<EMSource> sources_;
    std::vector<ChargedParticle> charged_particles_;

    // GPU computation infrastructure
    class EMGPUData;
    std::unique_ptr<EMGPUData> gpu_data_;

    // Core FDTD computation
    void updateElectricField(double dt);
    void updateMagneticField(double dt);
    void applyBoundaryConditions();
    void updateSources(double time);

    // PML implementation
    void setupPMLLayers();
    void updatePMLFields(double dt);

    // Maxwell stress tensor computation
    void computeMaxwellStressTensorAt(const Eigen::Vector3d& position,
                                    Eigen::Matrix3d& stress_tensor) const;

    // Particle dynamics with electromagnetic forces
    void updateChargedParticles(double dt);
    void computeRadiationDamping();

    // Grid operations
    Eigen::Vector3d interpolateFieldAt(const Eigen::Vector3d& position,
                                     const std::vector<Eigen::Vector3d>& field) const;
    void addCurrentDensity(const Eigen::Vector3d& position,
                          const Eigen::Vector3d& current);

    // Stability and convergence checks
    bool checkCFLCondition(double dt) const;
    double computeFieldResidual() const;
    void adaptiveTimeStep();

    // Material response models
    void updateLinearMaterials();
    void updateNonlinearMaterials();
    void updateDispersiveMaterials();
};

/**
 * Unified multi-physics solver with electromagnetic fields
 */
class CompletePhysicsSolver {
public:
    CompletePhysicsSolver(const VariationalContactParams& contact_params,
                         const SPHParams& fluid_params,
                         const SoftBodyMaterial& soft_body_material,
                         const EMParams& em_params);

    // Add objects with electromagnetic properties
    void addConductor(const std::vector<Eigen::Vector3d>& vertices,
                     const std::vector<std::vector<int>>& elements,
                     double conductivity);

    void addDielectric(const std::vector<Eigen::Vector3d>& vertices,
                      const std::vector<std::vector<int>>& elements,
                      double permittivity);

    void addMagneticMaterial(const std::vector<Eigen::Vector3d>& vertices,
                           const std::vector<std::vector<int>>& elements,
                           double permeability);

    void addChargedFluid(const Eigen::Vector3d& min_corner,
                        const Eigen::Vector3d& max_corner,
                        double particle_spacing,
                        double charge_density);

    // Complete multi-physics simulation
    void simulateStep(double dt);

    // Specialized scenarios
    void simulateElectrohydrodynamics(double dt);    // Charged fluid flow
    void simulateMagnetohydrodynamics(double dt);    // Conductive fluid in B-field
    void simulateElastodynamics(double dt);         // EM forces on elastic bodies
    void simulatePlasmaPhysics(double dt);          // Ionized gas dynamics

    // Analysis and visualization
    std::vector<Eigen::Vector3d> getAllPositions() const;
    std::vector<Eigen::Vector3d> getAllElectricFields() const;
    std::vector<Eigen::Vector3d> getAllMagneticFields() const;
    std::vector<double> getAllCharges() const;

    // Performance metrics
    double getTotalSystemEnergy() const;
    double getElectromagneticPower() const;
    size_t getTotalDOFs() const;

private:
    std::unique_ptr<VariationalContactSolver> contact_solver_;
    std::unique_ptr<SPHFluidSolver> fluid_solver_;
    std::unique_ptr<SoftBodySolver> soft_body_solver_;
    std::unique_ptr<ElectromagneticSolver> em_solver_;

    // Advanced coupling mechanisms
    void coupleAllPhysics(double dt);
    void synchronizeElectromagneticForces();
    void updateMaterialProperties();

    // Stability and accuracy control
    void adaptTimeStep();
    void checkCouplingStability();
    double computeGlobalError() const;
};

/**
 * Specialized electromagnetic applications
 */
namespace electromagnetic_applications {

/**
 * Antenna design and radiation analysis
 */
class AntennaDesigner : public ElectromagneticSolver {
public:
    explicit AntennaDesigner(const EMParams& params);

    void designDipoleAntenna(double length, double frequency);
    void designPatchAntenna(double width, double height, double frequency);
    void designHelicalAntenna(double diameter, double pitch, int turns);

    std::vector<double> computeRadiationPattern() const;
    double computeRadiationEfficiency() const;
    double computeInputImpedance() const;
    double computeGain() const;
};

/**
 * Microwave and RF circuit simulation
 */
class MicrowaveSimulator : public ElectromagneticSolver {
public:
    explicit MicrowaveSimulator(const EMParams& params);

    void addTransmissionLine(const Eigen::Vector3d& start,
                           const Eigen::Vector3d& end,
                           double impedance);

    void addCavityResonator(const Eigen::Vector3d& center,
                          const Eigen::Vector3d& dimensions);

    std::vector<std::complex<double>> computeSParameters() const;
    std::vector<double> getResonantFrequencies() const;
    double getQFactor() const;
};

/**
 * Bioelectromagnetics and medical applications
 */
class BioEMSimulator : public ElectromagneticSolver {
public:
    explicit BioEMSimulator(const EMParams& params);

    void setTissueProperties(const std::string& tissue_type,
                           double conductivity, double permittivity);

    void simulateECG(const std::vector<Eigen::Vector3d>& electrode_positions);
    void simulateEEG(const std::vector<Eigen::Vector3d>& sensor_positions);
    void simulateMRI(double field_strength, double frequency);

    std::vector<double> computeSAR() const;  // Specific Absorption Rate
    std::vector<double> getTemperatureRise() const;
};

} // namespace electromagnetic_applications

} // namespace physgrad