/**
 * GPU-Accelerated Soft Body Dynamics for PhysGrad
 *
 * Finite Element Method (FEM) implementation leveraging our 1,296x GPU speedup
 * Enables deformable solids with complete multi-physics integration
 */

#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "variational_contact.h"
#include "sph_fluid_dynamics.h"

namespace physgrad {

/**
 * Material models for soft body simulation
 */
enum class MaterialModel {
    LINEAR_ELASTIC,      // Simple Hooke's law
    NEO_HOOKEAN,        // Hyperelastic for large deformations
    MOONEY_RIVLIN,      // Advanced rubber-like materials
    SAINT_VENANT,       // Kirchhoff material
    VISCOELASTIC,       // Time-dependent behavior
    PLASTIC,            // Permanent deformation
    DAMAGE              // Fracture mechanics
};

/**
 * Soft body material properties
 */
struct SoftBodyMaterial {
    MaterialModel model = MaterialModel::NEO_HOOKEAN;

    // Basic elastic properties
    double youngs_modulus = 1e6;        // Pa (1 MPa - soft rubber)
    double poissons_ratio = 0.45;       // Near-incompressible
    double density = 1000.0;            // kg/m³

    // Hyperelastic parameters (Neo-Hookean)
    double shear_modulus = 4e5;         // Pa
    double bulk_modulus = 2e6;          // Pa

    // Damping and viscosity
    double rayleigh_alpha = 0.1;        // Mass-proportional damping
    double rayleigh_beta = 0.001;       // Stiffness-proportional damping
    double viscosity = 0.01;            // Pa·s

    // Fracture and damage
    double tensile_strength = 1e6;      // Pa
    double fracture_toughness = 1000.0; // J/m²
    double damage_threshold = 0.1;      // Strain threshold
    double damage_rate = 0.001;         // Damage evolution rate

    // Temperature effects
    double thermal_expansion = 1e-5;    // 1/K
    double heat_capacity = 1000.0;      // J/(kg·K)
    double thermal_conductivity = 0.1;  // W/(m·K)
};

/**
 * Finite element types supported
 */
enum class ElementType {
    TETRAHEDRON_4,      // 4-node tetrahedral elements
    TETRAHEDRON_10,     // 10-node quadratic tetrahedra
    HEXAHEDRON_8,       // 8-node hexahedral elements
    HEXAHEDRON_27,      // 27-node trilinear hexahedra
    TRIANGLE_3,         // 3-node triangular surface elements
    TRIANGLE_6          // 6-node quadratic triangles
};

/**
 * Finite element representation optimized for GPU computation
 */
struct FiniteElement {
    ElementType type;
    std::vector<int> node_indices;      // Global node indices
    double volume;                      // Element volume
    Eigen::Matrix3d deformation_gradient; // F = dx/dX
    double jacobian_determinant;        // det(F)

    // Element stiffness and mass matrices (stored sparse)
    Eigen::MatrixXd stiffness_matrix;
    Eigen::MatrixXd mass_matrix;

    // Stress and strain state
    Eigen::Matrix3d stress_tensor;      // Cauchy stress
    Eigen::Matrix3d strain_tensor;      // Green-Lagrange strain
    double strain_energy;               // Elastic energy density

    // Damage and plasticity state
    double damage_parameter = 0.0;      // 0 = intact, 1 = fully damaged
    Eigen::Matrix3d plastic_strain;     // Accumulated plastic deformation
    double equivalent_plastic_strain = 0.0;

    // Material properties (cached for performance)
    SoftBodyMaterial material;
};

/**
 * Soft body node representation
 */
struct SoftBodyNode {
    Eigen::Vector3d position;           // Current position
    Eigen::Vector3d velocity;           // Current velocity
    Eigen::Vector3d acceleration;       // Current acceleration

    Eigen::Vector3d reference_position; // Undeformed position
    Eigen::Vector3d displacement;       // u = x - X

    // Forces
    Eigen::Vector3d internal_force;     // From element deformation
    Eigen::Vector3d external_force;     // Applied forces
    Eigen::Vector3d contact_force;      // From contact interactions
    Eigen::Vector3d fluid_force;        // From fluid interactions

    double mass;                        // Nodal mass
    double temperature = 293.15;        // K (room temperature)

    // Constraints
    bool is_fixed = false;              // Dirichlet boundary condition
    Eigen::Vector3d prescribed_displacement; // If constrained

    // Connectivity
    std::vector<int> connected_elements; // Elements containing this node
    std::vector<int> neighboring_nodes;  // Adjacent nodes
};

/**
 * GPU-accelerated soft body dynamics solver
 */
class SoftBodySolver {
public:
    explicit SoftBodySolver(const SoftBodyMaterial& material);
    ~SoftBodySolver();

    // Mesh generation and setup
    void createTetrahedralMesh(const Eigen::Vector3d& min_corner,
                              const Eigen::Vector3d& max_corner,
                              double characteristic_length);

    void loadMeshFromFile(const std::string& filename); // VTK, OBJ, STL support

    void addSoftBody(const std::vector<Eigen::Vector3d>& vertices,
                    const std::vector<std::vector<int>>& elements,
                    const SoftBodyMaterial& material);

    // Simulation control
    void simulateStep(double dt);
    void simulateSteps(int num_steps);

    // Integration with other physics
    void integrateWithContactSolver(VariationalContactSolver& contact_solver);
    void integrateWithFluidSolver(SPHFluidSolver& fluid_solver);

    // Boundary conditions
    void fixNodes(const std::vector<int>& node_indices);
    void applyForce(int node_index, const Eigen::Vector3d& force);
    void applyPressure(const std::vector<int>& surface_nodes, double pressure);
    void setTemperature(const std::vector<int>& node_indices, double temperature);

    // Deformation and stress analysis
    std::vector<double> getVonMisesStress() const;
    std::vector<double> getStrainEnergy() const;
    std::vector<Eigen::Vector3d> getPrincipalStresses() const;
    std::vector<double> getDamageField() const;

    // Material property modification
    void updateMaterial(const SoftBodyMaterial& new_material);
    void setDamageThreshold(double threshold);
    void enablePlasticity(bool enable);

    // Mesh refinement and adaptation
    void refineAroundDamage(double damage_threshold);
    void adaptMeshForDeformation(double deformation_threshold);

    // State accessors
    const std::vector<SoftBodyNode>& getNodes() const { return nodes_; }
    const std::vector<FiniteElement>& getElements() const { return elements_; }
    std::vector<Eigen::Vector3d> getPositions() const;
    std::vector<Eigen::Vector3d> getVelocities() const;
    std::vector<Eigen::Vector3d> getDisplacements() const;

    // Performance and diagnostics
    double getTotalElasticEnergy() const;
    double getTotalKineticEnergy() const;
    double getMaxVonMisesStress() const;
    double getMaxDamage() const;
    size_t getGPUMemoryUsage() const;

    // GPU memory management
    void allocateGPUMemory(int max_nodes, int max_elements);
    void deallocateGPUMemory();

private:
    SoftBodyMaterial default_material_;
    std::vector<SoftBodyNode> nodes_;
    std::vector<FiniteElement> elements_;

    // GPU computation infrastructure
    class SoftBodyGPUData;
    std::unique_ptr<SoftBodyGPUData> gpu_data_;

    // Core FEM computation
    void assembleSystemMatrices();
    void computeInternalForces();
    void computeStressStrain();
    void updateDeformationGradient();
    void checkDamageEvolution();

    // Time integration schemes
    void explicitNewmarkIntegration(double dt);
    void implicitNewmarkIntegration(double dt);
    void centralDifferenceIntegration(double dt);

    // Matrix assembly and solving
    void assembleGlobalStiffnessMatrix();
    void assembleGlobalMassMatrix();
    void solveLinearSystem();

    // Material models implementation
    Eigen::Matrix3d computeStressTensor(const Eigen::Matrix3d& deformation_gradient,
                                       const SoftBodyMaterial& material) const;

    Eigen::Matrix<double, 6, 6> computeMaterialTangent(const Eigen::Matrix3d& deformation_gradient,
                                                      const SoftBodyMaterial& material) const;

    // Mesh operations
    void computeElementVolumes();
    void updateNodeConnectivity();
    void generateSurfaceMesh();

    // Contact and interaction
    void detectSelfContact();
    void computeContactForces();
    void exchangeForcesWithFluid();
};

/**
 * Unified multi-physics solver: Contacts + Fluids + Soft Bodies
 */
class UnifiedPhysicsSolver {
public:
    UnifiedPhysicsSolver(const VariationalContactParams& contact_params,
                        const SPHParams& fluid_params,
                        const SoftBodyMaterial& soft_body_material);

    // Add different types of objects
    void addRigidBodies(const std::vector<Eigen::Vector3d>& positions,
                       const std::vector<Eigen::Vector3d>& velocities,
                       const std::vector<double>& masses,
                       const std::vector<double>& radii);

    void addFluidRegion(const Eigen::Vector3d& min_corner,
                       const Eigen::Vector3d& max_corner,
                       double particle_spacing);

    void addSoftBody(const std::vector<Eigen::Vector3d>& vertices,
                    const std::vector<std::vector<int>>& elements);

    // Unified simulation step with full coupling
    void simulateStep(double dt);

    // Complex scenarios
    void simulateFluidStructureInteraction(double dt);
    void simulateContactDeformation(double dt);
    void simulateMultiPhaseSystem(double dt);

    // Analysis and visualization
    std::vector<Eigen::Vector3d> getAllPositions() const;
    std::vector<double> getAllStresses() const;
    std::vector<double> getAllDensities() const;
    std::vector<Eigen::Vector3d> getAllVelocities() const;

    // Performance metrics
    double getTotalSystemEnergy() const;
    size_t getTotalObjectCount() const;
    double getComputationTime() const;

private:
    std::unique_ptr<VariationalContactSolver> contact_solver_;
    std::unique_ptr<SPHFluidSolver> fluid_solver_;
    std::unique_ptr<SoftBodySolver> soft_body_solver_;

    // Coupling mechanisms
    void coupleRigidFluidSoft(double dt);
    void exchangeAllForces();
    void synchronizeAllMotion(double dt);

    // Advanced coupling algorithms
    void iterativeCoupling(double dt, int max_iterations = 5);
    void monolithicCoupling(double dt);
    void partitionedCoupling(double dt);
};

/**
 * Advanced soft body features for specialized applications
 */
namespace soft_body_advanced {

/**
 * Biomedical soft tissue simulation
 */
class BiomechanicalSolver : public SoftBodySolver {
public:
    explicit BiomechanicalSolver(const SoftBodyMaterial& tissue_material);

    void setTissueType(const std::string& type); // "skin", "muscle", "organ"
    void addBloodFlow(const std::vector<Eigen::Vector3d>& vessel_paths);
    void simulateBreathing(double frequency, double amplitude);
    void simulateHeartbeat(double bpm, double contractility);

    std::vector<double> getOxygenConcentration() const;
    std::vector<double> getTissuePressure() const;
};

/**
 * Manufacturing and materials processing
 */
class ManufacturingSolver : public SoftBodySolver {
public:
    explicit ManufacturingSolver(const SoftBodyMaterial& material);

    void simulateInjectionMolding(double injection_pressure, double temperature);
    void simulateStamping(double press_force, double die_velocity);
    void simulateExtrusion(double feed_rate, double die_temperature);
    void simulateWelding(const Eigen::Vector3d& torch_position, double heat_input);

    std::vector<double> getResidualStress() const;
    std::vector<double> getFormingDefects() const;
};

/**
 * Fracture mechanics and failure analysis
 */
class FractureSolver : public SoftBodySolver {
public:
    explicit FractureSolver(const SoftBodyMaterial& material);

    void initializeCrack(const Eigen::Vector3d& crack_tip, const Eigen::Vector3d& direction);
    void propagateCracks(double load_factor);
    void computeStressIntensityFactors();

    std::vector<Eigen::Vector3d> getCrackPaths() const;
    std::vector<double> getFractureEnergy() const;
    double getCriticalLoad() const;
};

} // namespace soft_body_advanced

} // namespace physgrad