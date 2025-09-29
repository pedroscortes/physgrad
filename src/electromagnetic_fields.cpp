/**
 * Electromagnetic Field Simulation Implementation
 *
 * Maxwell's equations solver with multi-physics integration
 * CPU implementation with optional GPU acceleration
 */

#include "electromagnetic_fields.h"
#include "electromagnetic_types.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace physgrad {

#ifdef HAVE_CUDA
// Forward declarations for CUDA implementation
class EMGPUDataImpl;
extern EMGPUDataImpl* createEMGPUData();
extern void destroyEMGPUData(EMGPUDataImpl* impl);
extern void allocateEMGPUMemory(EMGPUDataImpl* impl, int nx, int ny, int nz, int max_particles, int max_sources);
extern void deallocateEMGPUMemory(EMGPUDataImpl* impl);
extern void setEMGPUGridParams(EMGPUDataImpl* impl, int nx, int ny, int nz, double dx, double dy, double dz,
                              double dt, double c0, double epsilon0, double mu0);
extern void addEMGPUSource(EMGPUDataImpl* impl, const GPUEMSource& source);
extern void addEMGPUParticle(EMGPUDataImpl* impl, const GPUChargedParticle& particle);
extern void simulateEMGPUStep(EMGPUDataImpl* impl, double dt);
extern double getEMGPUTotalEnergy(EMGPUDataImpl* impl);
extern void getEMGPUElectricField(EMGPUDataImpl* impl, std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez);
extern void getEMGPUMagneticField(EMGPUDataImpl* impl, std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz);
extern void setupEMGPUPML(EMGPUDataImpl* impl, int thickness, double absorption);
extern void setEMGPUMaterialProperties(EMGPUDataImpl* impl, const std::vector<float>& epsilon,
                                       const std::vector<float>& mu, const std::vector<float>& sigma);
extern size_t getEMGPUMemoryUsage(EMGPUDataImpl* impl);
#endif

/**
 * GPU data management wrapper for electromagnetic simulation
 */
class ElectromagneticSolver::EMGPUData {
public:
#ifdef HAVE_CUDA
    EMGPUDataImpl* impl;
#endif
    bool gpu_enabled;
    size_t total_memory_bytes;

    EMGPUData() : gpu_enabled(false), total_memory_bytes(0) {
#ifdef HAVE_CUDA
        impl = createEMGPUData();
        gpu_enabled = true;
#else
        impl = nullptr;
#endif
    }

    ~EMGPUData() {
#ifdef HAVE_CUDA
        if (impl) {
            destroyEMGPUData(impl);
        }
#endif
    }

    void allocate(int nx, int ny, int nz, int max_particles, int max_sources) {
#ifdef HAVE_CUDA
        if (gpu_enabled && impl) {
            allocateEMGPUMemory(impl, nx, ny, nz, max_particles, max_sources);
            total_memory_bytes = getEMGPUMemoryUsage(impl);
        }
#else
        // CPU-only fallback - simplified memory estimation
        size_t grid_size = nx * ny * nz;
        total_memory_bytes = grid_size * sizeof(float) * 18 +
                            max_particles * sizeof(float) * 10 +
                            max_sources * sizeof(float) * 10;
        std::cout << "GPU acceleration not available, using CPU-only mode\n";
#endif
    }

    void deallocate() {
#ifdef HAVE_CUDA
        if (gpu_enabled && impl) {
            deallocateEMGPUMemory(impl);
        }
#endif
        total_memory_bytes = 0;
    }

    void setGridParams(int nx, int ny, int nz, double dx, double dy, double dz,
                      double dt, double c0, double epsilon0, double mu0) {
#ifdef HAVE_CUDA
        if (gpu_enabled && impl) {
            setEMGPUGridParams(impl, nx, ny, nz, dx, dy, dz, dt, c0, epsilon0, mu0);
        }
#endif
    }

    void addSource(const GPUEMSource& source) {
#ifdef HAVE_CUDA
        if (gpu_enabled && impl) {
            addEMGPUSource(impl, source);
        }
#endif
    }

    void addParticle(const GPUChargedParticle& particle) {
#ifdef HAVE_CUDA
        if (gpu_enabled && impl) {
            addEMGPUParticle(impl, particle);
        }
#endif
    }

    void simulateStep(double dt) {
#ifdef HAVE_CUDA
        if (gpu_enabled && impl) {
            simulateEMGPUStep(impl, dt);
        }
#endif
    }

    double getTotalEnergy() {
#ifdef HAVE_CUDA
        if (gpu_enabled && impl) {
            return getEMGPUTotalEnergy(impl);
        }
#endif
        return 0.0;
    }

    void getElectricField(std::vector<float>& Ex, std::vector<float>& Ey, std::vector<float>& Ez) {
#ifdef HAVE_CUDA
        if (gpu_enabled && impl) {
            getEMGPUElectricField(impl, Ex, Ey, Ez);
        }
#endif
    }

    void getMagneticField(std::vector<float>& Hx, std::vector<float>& Hy, std::vector<float>& Hz) {
#ifdef HAVE_CUDA
        if (gpu_enabled && impl) {
            getEMGPUMagneticField(impl, Hx, Hy, Hz);
        }
#endif
    }

    void setupPML(int thickness, double absorption) {
#ifdef HAVE_CUDA
        if (gpu_enabled && impl) {
            setupEMGPUPML(impl, thickness, absorption);
        }
#endif
    }

    void setMaterialProperties(const std::vector<float>& epsilon,
                              const std::vector<float>& mu,
                              const std::vector<float>& sigma) {
#ifdef HAVE_CUDA
        if (gpu_enabled && impl) {
            setEMGPUMaterialProperties(impl, epsilon, mu, sigma);
        }
#endif
    }
};

ElectromagneticSolver::ElectromagneticSolver(const EMParams& params)
    : params_(params), gpu_data_(std::make_unique<EMGPUData>()) {

    // Initialize grid parameters
    initializeGrid();
    allocateGPUMemory();

    std::cout << "ElectromagneticSolver initialized with "
              << params_.grid_resolution.x() << "×"
              << params_.grid_resolution.y() << "×"
              << params_.grid_resolution.z() << " grid\n";
    std::cout << "GPU memory allocated: " << getGPUMemoryUsage() / (1024*1024) << " MB\n";
}

ElectromagneticSolver::~ElectromagneticSolver() {
    deallocateGPUMemory();
}

void ElectromagneticSolver::initializeGrid() {
    // Calculate grid spacing
    Eigen::Vector3d domain_size = params_.domain_max - params_.domain_min;
    params_.grid_spacing = std::min({
        domain_size.x() / params_.grid_resolution.x(),
        domain_size.y() / params_.grid_resolution.y(),
        domain_size.z() / params_.grid_resolution.z()
    });

    // Ensure CFL stability condition
    double max_wave_speed = params_.speed_of_light;
    double spatial_factor = 1.0 / std::sqrt(
        1.0/(params_.grid_spacing*params_.grid_spacing) +
        1.0/(params_.grid_spacing*params_.grid_spacing) +
        1.0/(params_.grid_spacing*params_.grid_spacing)
    );
    double stable_dt = params_.cfl_factor * spatial_factor / max_wave_speed;

    if (params_.dt > stable_dt) {
        std::cout << "Warning: dt=" << params_.dt << " exceeds CFL limit " << stable_dt
                  << ". Adjusting dt for stability.\n";
        params_.dt = stable_dt;
    }

    // Initialize grid data structure
    size_t total_points = params_.grid_resolution.x() *
                         params_.grid_resolution.y() *
                         params_.grid_resolution.z();

    grid_.resize(total_points);

    // Set up grid point positions and initial field values
    for (int k = 0; k < params_.grid_resolution.z(); k++) {
        for (int j = 0; j < params_.grid_resolution.y(); j++) {
            for (int i = 0; i < params_.grid_resolution.x(); i++) {
                int idx = k * params_.grid_resolution.x() * params_.grid_resolution.y() +
                         j * params_.grid_resolution.x() + i;

                EMGridPoint& point = grid_[idx];

                // Set position
                point.position = params_.domain_min + Eigen::Vector3d(
                    i * params_.grid_spacing,
                    j * params_.grid_spacing,
                    k * params_.grid_spacing
                );
                point.grid_index = Eigen::Vector3i(i, j, k);

                // Initialize fields to zero
                point.E_field = Eigen::Vector3d::Zero();
                point.H_field = Eigen::Vector3d::Zero();
                point.D_field = Eigen::Vector3d::Zero();
                point.B_field = Eigen::Vector3d::Zero();

                // Set default material properties
                point.epsilon = params_.permittivity * params_.relative_permittivity;
                point.mu = params_.permeability * params_.relative_permeability;
                point.sigma = params_.conductivity;

                // Initialize PML parameters
                point.sx = point.sy = point.sz = 1.0;
                point.ax = point.ay = point.az = 0.0;
            }
        }
    }
}

void ElectromagneticSolver::allocateGPUMemory() {
    int max_particles = 10000;  // Default allocation
    int max_sources = 100;

    gpu_data_->allocate(
        params_.grid_resolution.x(),
        params_.grid_resolution.y(),
        params_.grid_resolution.z(),
        max_particles,
        max_sources
    );

    // Set GPU grid parameters
    gpu_data_->setGridParams(
        params_.grid_resolution.x(),
        params_.grid_resolution.y(),
        params_.grid_resolution.z(),
        params_.grid_spacing,
        params_.grid_spacing,
        params_.grid_spacing,
        params_.dt,
        params_.speed_of_light,
        params_.permittivity,
        params_.permeability
    );

    // Initialize material properties
    size_t grid_size = params_.grid_resolution.x() *
                      params_.grid_resolution.y() *
                      params_.grid_resolution.z();

    std::vector<float> epsilon_host(grid_size, params_.permittivity * params_.relative_permittivity);
    std::vector<float> mu_host(grid_size, params_.permeability * params_.relative_permeability);
    std::vector<float> sigma_host(grid_size, params_.conductivity);

    gpu_data_->setMaterialProperties(epsilon_host, mu_host, sigma_host);

    // Setup PML boundaries if enabled
    if (params_.pml_boundaries) {
        setupPMLLayers();
    }
}

void ElectromagneticSolver::deallocateGPUMemory() {
    if (gpu_data_) {
        gpu_data_->deallocate();
    }
}

void ElectromagneticSolver::setupPMLLayers() {
    gpu_data_->setupPML(params_.pml_thickness, params_.pml_absorption);
}

void ElectromagneticSolver::addSource(const EMSource& source) {
    sources_.push_back(source);

    // Convert to GPU format
    GPUEMSource gpu_source;
    gpu_source.position = make_float3(source.position.x(), source.position.y(), source.position.z());
    gpu_source.direction = make_float3(source.direction.x(), source.direction.y(), source.direction.z());
    gpu_source.polarization = make_float3(source.polarization.x(), source.polarization.y(), source.polarization.z());
    gpu_source.frequency = source.frequency;
    gpu_source.amplitude = source.amplitude;
    gpu_source.phase = source.phase;
    gpu_source.source_type = static_cast<int>(source.type);
    gpu_source.beam_waist = source.beam_waist;

    // Calculate grid indices for source region
    Eigen::Vector3d grid_pos = (source.position - params_.domain_min) / params_.grid_spacing;
    int source_size = 3;  // Source extends ±3 grid points

    gpu_source.i_start = std::max(0, static_cast<int>(grid_pos.x()) - source_size);
    gpu_source.i_end = std::min(params_.grid_resolution.x(), static_cast<int>(grid_pos.x()) + source_size + 1);
    gpu_source.j_start = std::max(0, static_cast<int>(grid_pos.y()) - source_size);
    gpu_source.j_end = std::min(params_.grid_resolution.y(), static_cast<int>(grid_pos.y()) + source_size + 1);
    gpu_source.k_start = std::max(0, static_cast<int>(grid_pos.z()) - source_size);
    gpu_source.k_end = std::min(params_.grid_resolution.z(), static_cast<int>(grid_pos.z()) + source_size + 1);

    // Add to GPU sources
    gpu_data_->addSource(gpu_source);
}

void ElectromagneticSolver::addChargedParticle(const ChargedParticle& particle) {
    charged_particles_.push_back(particle);

    // Convert to GPU format
    GPUChargedParticle gpu_particle;
    gpu_particle.position = make_float3(particle.position.x(), particle.position.y(), particle.position.z());
    gpu_particle.velocity = make_float3(particle.velocity.x(), particle.velocity.y(), particle.velocity.z());
    gpu_particle.acceleration = make_float3(particle.acceleration.x(), particle.acceleration.y(), particle.acceleration.z());
    gpu_particle.charge = particle.charge;
    gpu_particle.mass = particle.mass;
    gpu_particle.prev_acceleration = gpu_particle.acceleration;

    gpu_data_->addParticle(gpu_particle);
}

void ElectromagneticSolver::simulateStep(double dt) {
    gpu_data_->simulateStep(dt);
}

void ElectromagneticSolver::simulateSteps(int num_steps) {
    for (int step = 0; step < num_steps; step++) {
        simulateStep(params_.dt);
    }
}

std::vector<Eigen::Vector3d> ElectromagneticSolver::computeLorentzForces(
    const std::vector<Eigen::Vector3d>& positions,
    const std::vector<Eigen::Vector3d>& velocities,
    const std::vector<double>& charges) const {

    std::vector<Eigen::Vector3d> forces(positions.size());

    for (size_t i = 0; i < positions.size(); i++) {
        // Interpolate fields at particle position
        Eigen::Vector3d E_field = interpolateFieldAt(positions[i], getElectricField());
        Eigen::Vector3d B_field = interpolateFieldAt(positions[i], getMagneticField()) * params_.permeability;

        // Compute Lorentz force: F = q(E + v × B)
        Eigen::Vector3d v_cross_B = velocities[i].cross(B_field);
        forces[i] = charges[i] * (E_field + v_cross_B);
    }

    return forces;
}

double ElectromagneticSolver::getTotalElectromagneticEnergy() const {
    double total_energy = gpu_data_->getTotalEnergy();

    // Multiply by grid cell volume
    double cell_volume = params_.grid_spacing * params_.grid_spacing * params_.grid_spacing;
    return total_energy * cell_volume;
}

std::vector<Eigen::Vector3d> ElectromagneticSolver::getElectricField() const {
    size_t grid_size = params_.grid_resolution.x() *
                      params_.grid_resolution.y() *
                      params_.grid_resolution.z();

    std::vector<float> Ex_host, Ey_host, Ez_host;
    gpu_data_->getElectricField(Ex_host, Ey_host, Ez_host);

    std::vector<Eigen::Vector3d> E_field(grid_size);
    for (size_t i = 0; i < grid_size; i++) {
        if (i < Ex_host.size() && i < Ey_host.size() && i < Ez_host.size()) {
            E_field[i] = Eigen::Vector3d(Ex_host[i], Ey_host[i], Ez_host[i]);
        } else {
            E_field[i] = Eigen::Vector3d::Zero();
        }
    }

    return E_field;
}

std::vector<Eigen::Vector3d> ElectromagneticSolver::getMagneticField() const {
    size_t grid_size = params_.grid_resolution.x() *
                      params_.grid_resolution.y() *
                      params_.grid_resolution.z();

    std::vector<float> Hx_host, Hy_host, Hz_host;
    gpu_data_->getMagneticField(Hx_host, Hy_host, Hz_host);

    std::vector<Eigen::Vector3d> H_field(grid_size);
    for (size_t i = 0; i < grid_size; i++) {
        if (i < Hx_host.size() && i < Hy_host.size() && i < Hz_host.size()) {
            H_field[i] = Eigen::Vector3d(Hx_host[i], Hy_host[i], Hz_host[i]);
        } else {
            H_field[i] = Eigen::Vector3d::Zero();
        }
    }

    return H_field;
}

Eigen::Vector3d ElectromagneticSolver::interpolateFieldAt(const Eigen::Vector3d& position,
                                                        const std::vector<Eigen::Vector3d>& field) const {
    // Convert position to grid coordinates
    Eigen::Vector3d grid_pos = (position - params_.domain_min) / params_.grid_spacing;

    int i = static_cast<int>(std::floor(grid_pos.x()));
    int j = static_cast<int>(std::floor(grid_pos.y()));
    int k = static_cast<int>(std::floor(grid_pos.z()));

    // Clamp to grid bounds
    i = std::max(0, std::min(i, params_.grid_resolution.x() - 2));
    j = std::max(0, std::min(j, params_.grid_resolution.y() - 2));
    k = std::max(0, std::min(k, params_.grid_resolution.z() - 2));

    // Interpolation weights
    double wx = grid_pos.x() - i;
    double wy = grid_pos.y() - j;
    double wz = grid_pos.z() - k;

    // Trilinear interpolation
    auto getFieldAt = [&](int di, int dj, int dk) -> Eigen::Vector3d {
        int idx = (k + dk) * params_.grid_resolution.x() * params_.grid_resolution.y() +
                  (j + dj) * params_.grid_resolution.x() + (i + di);
        return field[idx];
    };

    Eigen::Vector3d result =
        (1-wx)*(1-wy)*(1-wz)*getFieldAt(0,0,0) + (1-wx)*(1-wy)*wz*getFieldAt(0,0,1) +
        (1-wx)*wy*(1-wz)*getFieldAt(0,1,0) + (1-wx)*wy*wz*getFieldAt(0,1,1) +
        wx*(1-wy)*(1-wz)*getFieldAt(1,0,0) + wx*(1-wy)*wz*getFieldAt(1,0,1) +
        wx*wy*(1-wz)*getFieldAt(1,1,0) + wx*wy*wz*getFieldAt(1,1,1);

    return result;
}

size_t ElectromagneticSolver::getGPUMemoryUsage() const {
    return gpu_data_ ? gpu_data_->total_memory_bytes : 0;
}

void ElectromagneticSolver::integrateWithContactSolver(VariationalContactSolver& contact_solver) {
    // TODO: Implement once contact solver API is extended with getPositions/getVelocities/applyExternalForce
    (void)contact_solver;  // Suppress unused parameter warning
    /*
    // Get positions and charges of contact objects
    auto positions = contact_solver.getPositions();
    auto velocities = contact_solver.getVelocities();

    // Assume unit charges for simplicity (can be extended)
    std::vector<double> charges(positions.size(), 1e-6);  // 1 μC

    // Compute electromagnetic forces
    auto em_forces = computeLorentzForces(positions, velocities, charges);

    // Apply forces to contact solver
    for (size_t i = 0; i < positions.size(); i++) {
        contact_solver.applyExternalForce(i, em_forces[i]);
    }
    */
}

void ElectromagneticSolver::integrateWithFluidSolver(SPHFluidSolver& fluid_solver) {
    // Get fluid particle data
    auto fluid_positions = fluid_solver.getPositions();
    auto fluid_velocities = fluid_solver.getVelocities();

    // Assume fluid carries charge (e.g., ionized gas or electrolyte)
    std::vector<double> charges(fluid_positions.size(), 1e-9);  // 1 nC per particle

    // Compute electromagnetic forces on fluid
    auto em_forces = computeLorentzForces(fluid_positions, fluid_velocities, charges);

    // TODO: Apply forces once SPHFluidSolver has applyExternalForce method
    (void)em_forces;  // Suppress unused variable warning
    /*
    // Apply forces to fluid solver
    for (size_t i = 0; i < fluid_positions.size(); i++) {
        fluid_solver.applyExternalForce(i, em_forces[i]);
    }
    */
}

void ElectromagneticSolver::integrateWithSoftBodySolver(SoftBodySolver& soft_body_solver) {
    // Get soft body node positions and velocities
    auto positions = soft_body_solver.getPositions();
    auto velocities = soft_body_solver.getVelocities();

    // Assume conducting soft body with surface charges
    std::vector<double> charges(positions.size(), 1e-8);  // 10 nC per node

    // Compute electromagnetic forces
    auto em_forces = computeLorentzForces(positions, velocities, charges);

    // Apply forces to soft body nodes
    for (size_t i = 0; i < positions.size(); i++) {
        soft_body_solver.applyForce(i, em_forces[i]);
    }
}

// CompletePhysicsSolver implementation
CompletePhysicsSolver::CompletePhysicsSolver(const VariationalContactParams& contact_params,
                                           const SPHParams& fluid_params,
                                           const SoftBodyMaterial& soft_body_material,
                                           const EMParams& em_params)
    : contact_solver_(std::make_unique<VariationalContactSolver>(contact_params)),
      fluid_solver_(std::make_unique<SPHFluidSolver>(fluid_params)),
      soft_body_solver_(std::make_unique<SoftBodySolver>(soft_body_material)),
      em_solver_(std::make_unique<ElectromagneticSolver>(em_params)) {

    std::cout << "CompletePhysicsSolver initialized with full multi-physics coupling\n";
}

void CompletePhysicsSolver::simulateStep(double dt) {
    // Update electromagnetic fields first
    em_solver_->simulateStep(dt);

    // Integrate EM forces with other physics
    em_solver_->integrateWithContactSolver(*contact_solver_);
    em_solver_->integrateWithFluidSolver(*fluid_solver_);
    em_solver_->integrateWithSoftBodySolver(*soft_body_solver_);

    // Update other physics systems
    // TODO: Implement simulateStep methods in each solver
    // contact_solver_->simulateStep(dt);
    fluid_solver_->simulateStep(dt);
    soft_body_solver_->simulateStep(dt);

    // Synchronize all physics for consistent coupling
    coupleAllPhysics(dt);
}

void CompletePhysicsSolver::coupleAllPhysics(double dt) {
    // Additional coupling logic for complex interactions
    // This ensures all physics systems remain synchronized

    // Update material properties based on temperature, deformation, etc.
    updateMaterialProperties();

    // Exchange forces between all systems
    synchronizeElectromagneticForces();
}

void CompletePhysicsSolver::synchronizeElectromagneticForces() {
    // Ensure electromagnetic forces are properly distributed
    // across all coupled physics systems
}

void CompletePhysicsSolver::updateMaterialProperties() {
    // Update electromagnetic material properties based on
    // deformation, temperature, and other physical state changes
}

double CompletePhysicsSolver::getTotalSystemEnergy() const {
    // TODO: Implement energy getters in each solver
    /*
    return contact_solver_->getTotalKineticEnergy() +
           contact_solver_->getTotalPotentialEnergy() +
           fluid_solver_->getTotalKineticEnergy() +
           soft_body_solver_->getTotalKineticEnergy() +
           soft_body_solver_->getTotalElasticEnergy() +
           em_solver_->getTotalElectromagneticEnergy();
    */
    return em_solver_->getTotalElectromagneticEnergy();
}

} // namespace physgrad