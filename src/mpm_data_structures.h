/**
 * PhysGrad Material Point Method (MPM) Data Structures
 *
 * High-performance Array-of-Structs-of-Arrays (AoSoA) data layout
 * optimized for GPU computation with coalesced memory access
 */

#pragma once

#include <vector>
#include <memory>
#include <array>
#include <cmath>
#include <algorithm>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOST_DEVICE
#define CUDA_DEVICE
#define CUDA_HOST
#endif

// Basic type definitions for CUDA compatibility
struct int3 {
    int x, y, z;
    int3() = default;
    int3(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
};

template<typename T>
struct vec3 {
    T x, y, z;
    vec3() = default;
    vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
};

using float3 = vec3<float>;
using double3 = vec3<double>;

template<typename T>
using T3 = vec3<T>;

namespace physgrad::mpm {

// =============================================================================
// MATERIAL TYPES AND CONSTITUTIVE MODELS
// =============================================================================

enum class MaterialType : uint32_t {
    ELASTIC = 0,
    ELASTOPLASTIC = 1,
    FLUID = 2,
    SAND = 3,
    SNOW = 4,
    JELLY = 5
};

struct MaterialParameters {
    float density = 1000.0f;           // kg/m³
    float youngs_modulus = 1e6f;       // Pa
    float poisson_ratio = 0.3f;        // dimensionless
    float yield_stress = 1e5f;         // Pa (for plasticity)
    float hardening_coefficient = 0.0f; // Pa (for plasticity)
    float viscosity = 0.001f;          // Pa·s (for fluids)
    float friction_angle = 30.0f;      // degrees (for sand)
    float cohesion = 1000.0f;          // Pa (for sand)
    float critical_compression = 2.5e-2f; // dimensionless (for snow)
    float critical_stretch = 7.5e-3f;     // dimensionless (for snow)
    MaterialType type = MaterialType::ELASTIC;
};

// Material parameter creation functions
template<typename T>
MaterialParameters createElasticMaterial(T density, T youngs_modulus, T poisson_ratio) {
    MaterialParameters params;
    params.density = density;
    params.youngs_modulus = youngs_modulus;
    params.poisson_ratio = poisson_ratio;
    params.type = MaterialType::ELASTIC;
    return params;
}

template<typename T>
MaterialParameters createFluidMaterial(T density, T viscosity, T gamma = 7.0f) {
    MaterialParameters params;
    params.density = density;
    params.viscosity = viscosity;
    params.youngs_modulus = gamma; // Using youngs_modulus field for gamma (bulk modulus)
    params.type = MaterialType::FLUID;
    return params;
}

// =============================================================================
// AoSoA DATA LAYOUT FOR PARTICLES
// =============================================================================

/**
 * Array-of-Structs-of-Arrays (AoSoA) layout for optimal GPU memory access
 * Chunks particles into groups for vectorized operations and cache efficiency
 */
template<typename T, size_t ChunkSize = 256>
class ParticleAoSoA {
public:
    static constexpr size_t chunk_size = ChunkSize;
    using scalar_type = T;

private:
    // Position and velocity (3D vectors per particle)
    std::vector<std::array<T, chunk_size * 3>> position_chunks_;
    std::vector<std::array<T, chunk_size * 3>> velocity_chunks_;

    // Mass and volume
    std::vector<std::array<T, chunk_size>> mass_chunks_;
    std::vector<std::array<T, chunk_size>> volume_chunks_;

    // Deformation gradient (3x3 matrix per particle)
    std::vector<std::array<T, chunk_size * 9>> deformation_gradient_chunks_;

    // Stress tensor (symmetric 3x3, stored as 6 components: xx, yy, zz, xy, xz, yz)
    std::vector<std::array<T, chunk_size * 6>> stress_chunks_;

    // Plastic deformation (for elastoplastic materials)
    std::vector<std::array<T, chunk_size * 9>> plastic_deformation_chunks_;

    // Material properties per particle
    std::vector<std::array<MaterialType, chunk_size>> material_type_chunks_;
    std::vector<std::array<uint32_t, chunk_size>> material_id_chunks_;

    // Simulation state
    std::vector<std::array<uint32_t, chunk_size>> active_chunks_;  // Active particle flags

    size_t num_particles_ = 0;
    size_t num_chunks_ = 0;

public:
    ParticleAoSoA() = default;

    explicit ParticleAoSoA(size_t num_particles) {
        resize(num_particles);
    }

    void resize(size_t num_particles) {
        num_particles_ = num_particles;
        num_chunks_ = (num_particles + chunk_size - 1) / chunk_size;

        position_chunks_.resize(num_chunks_);
        velocity_chunks_.resize(num_chunks_);
        mass_chunks_.resize(num_chunks_);
        volume_chunks_.resize(num_chunks_);
        deformation_gradient_chunks_.resize(num_chunks_);
        stress_chunks_.resize(num_chunks_);
        plastic_deformation_chunks_.resize(num_chunks_);
        material_type_chunks_.resize(num_chunks_);
        material_id_chunks_.resize(num_chunks_);
        active_chunks_.resize(num_chunks_);
    }

    size_t size() const { return num_particles_; }
    size_t num_chunks() const { return num_chunks_; }

    // Position access
    CUDA_HOST_DEVICE void setPosition(size_t particle_id, T x, T y, T z) {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        position_chunks_[chunk_id][local_id * 3 + 0] = x;
        position_chunks_[chunk_id][local_id * 3 + 1] = y;
        position_chunks_[chunk_id][local_id * 3 + 2] = z;
    }

    CUDA_HOST_DEVICE void getPosition(size_t particle_id, T& x, T& y, T& z) const {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        x = position_chunks_[chunk_id][local_id * 3 + 0];
        y = position_chunks_[chunk_id][local_id * 3 + 1];
        z = position_chunks_[chunk_id][local_id * 3 + 2];
    }

    // Velocity access
    CUDA_HOST_DEVICE void setVelocity(size_t particle_id, T vx, T vy, T vz) {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        velocity_chunks_[chunk_id][local_id * 3 + 0] = vx;
        velocity_chunks_[chunk_id][local_id * 3 + 1] = vy;
        velocity_chunks_[chunk_id][local_id * 3 + 2] = vz;
    }

    CUDA_HOST_DEVICE void getVelocity(size_t particle_id, T& vx, T& vy, T& vz) const {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        vx = velocity_chunks_[chunk_id][local_id * 3 + 0];
        vy = velocity_chunks_[chunk_id][local_id * 3 + 1];
        vz = velocity_chunks_[chunk_id][local_id * 3 + 2];
    }

    // Mass and volume access
    CUDA_HOST_DEVICE void setMass(size_t particle_id, T mass) {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        mass_chunks_[chunk_id][local_id] = mass;
    }

    CUDA_HOST_DEVICE T getMass(size_t particle_id) const {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        return mass_chunks_[chunk_id][local_id];
    }

    CUDA_HOST_DEVICE void setVolume(size_t particle_id, T volume) {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        volume_chunks_[chunk_id][local_id] = volume;
    }

    CUDA_HOST_DEVICE T getVolume(size_t particle_id) const {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        return volume_chunks_[chunk_id][local_id];
    }

    // Deformation gradient access (3x3 matrix stored as 9 elements row-major)
    CUDA_HOST_DEVICE void setDeformationGradient(size_t particle_id, const T F[9]) {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        for (int i = 0; i < 9; ++i) {
            deformation_gradient_chunks_[chunk_id][local_id * 9 + i] = F[i];
        }
    }

    CUDA_HOST_DEVICE void getDeformationGradient(size_t particle_id, T F[9]) const {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        for (int i = 0; i < 9; ++i) {
            F[i] = deformation_gradient_chunks_[chunk_id][local_id * 9 + i];
        }
    }

    // Stress tensor access (symmetric 3x3 stored as 6 components)
    CUDA_HOST_DEVICE void setStress(size_t particle_id, const T stress[6]) {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        for (int i = 0; i < 6; ++i) {
            stress_chunks_[chunk_id][local_id * 6 + i] = stress[i];
        }
    }

    CUDA_HOST_DEVICE void getStress(size_t particle_id, T stress[6]) const {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        for (int i = 0; i < 6; ++i) {
            stress[i] = stress_chunks_[chunk_id][local_id * 6 + i];
        }
    }

    // Material type access
    CUDA_HOST_DEVICE void setMaterialType(size_t particle_id, MaterialType type) {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        material_type_chunks_[chunk_id][local_id] = type;
    }

    CUDA_HOST_DEVICE MaterialType getMaterialType(size_t particle_id) const {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        return material_type_chunks_[chunk_id][local_id];
    }

    // Active particle flag
    CUDA_HOST_DEVICE void setActive(size_t particle_id, bool active) {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        active_chunks_[chunk_id][local_id] = active ? 1u : 0u;
    }

    CUDA_HOST_DEVICE bool isActive(size_t particle_id) const {
        size_t chunk_id = particle_id / chunk_size;
        size_t local_id = particle_id % chunk_size;
        return active_chunks_[chunk_id][local_id] != 0u;
    }

    // Bulk data operations for initialization
    void setPositions(const std::vector<std::array<T, 3>>& positions) {
        for (size_t i = 0; i < std::min(positions.size(), size()); ++i) {
            setPosition(i, positions[i][0], positions[i][1], positions[i][2]);
        }
    }

    void setVelocities(const std::vector<std::array<T, 3>>& velocities) {
        for (size_t i = 0; i < std::min(velocities.size(), size()); ++i) {
            setVelocity(i, velocities[i][0], velocities[i][1], velocities[i][2]);
        }
    }

    void setMasses(const std::vector<T>& masses) {
        for (size_t i = 0; i < std::min(masses.size(), size()); ++i) {
            setMass(i, masses[i]);
        }
    }

    std::vector<std::array<T, 3>> getPositions() const {
        std::vector<std::array<T, 3>> result(size());
        for (size_t i = 0; i < size(); ++i) {
            T x, y, z;
            getPosition(i, x, y, z);
            result[i] = {{x, y, z}};
        }
        return result;
    }

    std::vector<std::array<T, 3>> getVelocities() const {
        std::vector<std::array<T, 3>> result(size());
        for (size_t i = 0; i < size(); ++i) {
            T vx, vy, vz;
            getVelocity(i, vx, vy, vz);
            result[i] = {{vx, vy, vz}};
        }
        return result;
    }

    std::vector<T> getMasses() const {
        std::vector<T> result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = getMass(i);
        }
        return result;
    }

    // Raw data access for GPU kernels
    T* getPositionData() { return reinterpret_cast<T*>(position_chunks_.data()); }
    T* getVelocityData() { return reinterpret_cast<T*>(velocity_chunks_.data()); }
    T* getMassData() { return reinterpret_cast<T*>(mass_chunks_.data()); }
    T* getVolumeData() { return reinterpret_cast<T*>(volume_chunks_.data()); }
    T* getDeformationGradientData() { return reinterpret_cast<T*>(deformation_gradient_chunks_.data()); }
    T* getStressData() { return reinterpret_cast<T*>(stress_chunks_.data()); }
    MaterialType* getMaterialTypeData() { return reinterpret_cast<MaterialType*>(material_type_chunks_.data()); }
    uint32_t* getActiveData() { return reinterpret_cast<uint32_t*>(active_chunks_.data()); }

    const T* getPositionData() const { return reinterpret_cast<const T*>(position_chunks_.data()); }
    const T* getVelocityData() const { return reinterpret_cast<const T*>(velocity_chunks_.data()); }
    const T* getMassData() const { return reinterpret_cast<const T*>(mass_chunks_.data()); }
    const T* getVolumeData() const { return reinterpret_cast<const T*>(volume_chunks_.data()); }
    const T* getDeformationGradientData() const { return reinterpret_cast<const T*>(deformation_gradient_chunks_.data()); }
    const T* getStressData() const { return reinterpret_cast<const T*>(stress_chunks_.data()); }
    const MaterialType* getMaterialTypeData() const { return reinterpret_cast<const MaterialType*>(material_type_chunks_.data()); }
    const uint32_t* getActiveData() const { return reinterpret_cast<const uint32_t*>(active_chunks_.data()); }

    // Memory footprint
    size_t getMemoryFootprint() const {
        return num_chunks_ * (sizeof(position_chunks_[0]) + sizeof(velocity_chunks_[0]) +
                             sizeof(mass_chunks_[0]) + sizeof(volume_chunks_[0]) +
                             sizeof(deformation_gradient_chunks_[0]) + sizeof(stress_chunks_[0]) +
                             sizeof(plastic_deformation_chunks_[0]) + sizeof(material_type_chunks_[0]) +
                             sizeof(material_id_chunks_[0]) + sizeof(active_chunks_[0]));
    }
};

// =============================================================================
// GRID DATA STRUCTURES
// =============================================================================

/**
 * Structured grid for MPM background mesh
 * Optimized for regular grid operations and boundary conditions
 */
template<typename T>
class MPMGrid {
public:
    using scalar_type = T;

private:
    // Grid dimensions and spacing
    int3 dimensions_;           // Grid dimensions (nx, ny, nz)
    T3<T> grid_spacing_;          // Grid cell size (dx, dy, dz)
    T3<T> origin_;                // Grid origin (x0, y0, z0)

    // Grid data arrays (linearized indexing)
    std::vector<T> masses_;              // Nodal masses
    std::vector<T> velocities_;          // Nodal velocities (3 components per node)
    std::vector<T> forces_;              // Nodal forces (3 components per node)
    std::vector<T> momentum_;            // Nodal momentum (3 components per node)

    // Boundary condition flags
    std::vector<uint32_t> boundary_conditions_; // Per-node boundary flags

    size_t total_nodes_;

public:
    MPMGrid() = default;

    MPMGrid(int3 dimensions, T3<T> spacing, T3<T> origin = {T{0}, T{0}, T{0}})
        : dimensions_(dimensions), grid_spacing_(spacing), origin_(origin) {

        total_nodes_ = static_cast<size_t>(dimensions_.x) * dimensions_.y * dimensions_.z;

        masses_.resize(total_nodes_, T{0});
        velocities_.resize(total_nodes_ * 3, T{0});
        forces_.resize(total_nodes_ * 3, T{0});
        momentum_.resize(total_nodes_ * 3, T{0});
        boundary_conditions_.resize(total_nodes_, 0u);
    }

    // Constructor accepting std::array
    MPMGrid(const std::array<size_t, 3>& resolution, const std::array<T, 3>& domain_size)
        : MPMGrid(int3{static_cast<int>(resolution[0]), static_cast<int>(resolution[1]), static_cast<int>(resolution[2])},
                  T3<T>{domain_size[0]/resolution[0], domain_size[1]/resolution[1], domain_size[2]/resolution[2]},
                  T3<T>{T{0}, T{0}, T{0}}) {}

    void clearGrid() {
        std::fill(masses_.begin(), masses_.end(), T{0});
        std::fill(velocities_.begin(), velocities_.end(), T{0});
        std::fill(forces_.begin(), forces_.end(), T{0});
        std::fill(momentum_.begin(), momentum_.end(), T{0});
    }

    std::array<T, 3> getGridSpacing() const {
        return {{grid_spacing_.x, grid_spacing_.y, grid_spacing_.z}};
    }

    // Grid properties
    int3 getDimensions() const { return dimensions_; }
    T3<T> getSpacing() const { return grid_spacing_; }
    T3<T> getOrigin() const { return origin_; }
    size_t getTotalNodes() const { return total_nodes_; }

    // Index conversion
    CUDA_HOST_DEVICE size_t getLinearIndex(int i, int j, int k) const {
        return static_cast<size_t>(k) * dimensions_.x * dimensions_.y +
               static_cast<size_t>(j) * dimensions_.x + static_cast<size_t>(i);
    }

    CUDA_HOST_DEVICE int3 getGridIndex(size_t linear_index) const {
        int k = static_cast<int>(linear_index / (dimensions_.x * dimensions_.y));
        int remainder = static_cast<int>(linear_index % (dimensions_.x * dimensions_.y));
        int j = remainder / dimensions_.x;
        int i = remainder % dimensions_.x;
        return {i, j, k};
    }

    // World coordinate conversion
    CUDA_HOST_DEVICE T3<T> getWorldPosition(int i, int j, int k) const {
        return {
            origin_.x + i * grid_spacing_.x,
            origin_.y + j * grid_spacing_.y,
            origin_.z + k * grid_spacing_.z
        };
    }

    CUDA_HOST_DEVICE int3 getGridPosition(T x, T y, T z) const {
        return {
            static_cast<int>(std::floor((x - origin_.x) / grid_spacing_.x)),
            static_cast<int>(std::floor((y - origin_.y) / grid_spacing_.y)),
            static_cast<int>(std::floor((z - origin_.z) / grid_spacing_.z))
        };
    }

    // Grid data access
    CUDA_HOST_DEVICE void setMass(size_t node_id, T mass) {
        masses_[node_id] = mass;
    }

    CUDA_HOST_DEVICE T getMass(size_t node_id) const {
        return masses_[node_id];
    }

    CUDA_HOST_DEVICE void setVelocity(size_t node_id, T vx, T vy, T vz) {
        velocities_[node_id * 3 + 0] = vx;
        velocities_[node_id * 3 + 1] = vy;
        velocities_[node_id * 3 + 2] = vz;
    }

    CUDA_HOST_DEVICE void getVelocity(size_t node_id, T& vx, T& vy, T& vz) const {
        vx = velocities_[node_id * 3 + 0];
        vy = velocities_[node_id * 3 + 1];
        vz = velocities_[node_id * 3 + 2];
    }

    CUDA_HOST_DEVICE void addVelocity(size_t node_id, T dvx, T dvy, T dvz) {
        velocities_[node_id * 3 + 0] += dvx;
        velocities_[node_id * 3 + 1] += dvy;
        velocities_[node_id * 3 + 2] += dvz;
    }

    CUDA_HOST_DEVICE void setForce(size_t node_id, T fx, T fy, T fz) {
        forces_[node_id * 3 + 0] = fx;
        forces_[node_id * 3 + 1] = fy;
        forces_[node_id * 3 + 2] = fz;
    }

    CUDA_HOST_DEVICE void addForce(size_t node_id, T dfx, T dfy, T dfz) {
        forces_[node_id * 3 + 0] += dfx;
        forces_[node_id * 3 + 1] += dfy;
        forces_[node_id * 3 + 2] += dfz;
    }

    // Boundary conditions
    CUDA_HOST_DEVICE void setBoundaryCondition(size_t node_id, uint32_t bc_flags) {
        boundary_conditions_[node_id] = bc_flags;
    }

    CUDA_HOST_DEVICE uint32_t getBoundaryCondition(size_t node_id) const {
        return boundary_conditions_[node_id];
    }


    // Raw data access for GPU kernels
    T* getMassData() { return masses_.data(); }
    T* getVelocityData() { return velocities_.data(); }
    T* getForceData() { return forces_.data(); }
    T* getMomentumData() { return momentum_.data(); }
    uint32_t* getBoundaryConditionData() { return boundary_conditions_.data(); }

    const T* getMassData() const { return masses_.data(); }
    const T* getVelocityData() const { return velocities_.data(); }
    const T* getForceData() const { return forces_.data(); }
    const T* getMomentumData() const { return momentum_.data(); }
    const uint32_t* getBoundaryConditionData() const { return boundary_conditions_.data(); }

    // Memory footprint
    size_t getMemoryFootprint() const {
        return masses_.size() * sizeof(T) +
               velocities_.size() * sizeof(T) +
               forces_.size() * sizeof(T) +
               momentum_.size() * sizeof(T) +
               boundary_conditions_.size() * sizeof(uint32_t);
    }
};

// =============================================================================
// MATERIAL DATABASE
// =============================================================================

/**
 * Material parameter database for multi-material simulations
 */
class MaterialDatabase {
private:
    std::vector<MaterialParameters> materials_;

public:
    MaterialDatabase() {
        // Add default materials
        addMaterial(createWaterMaterial());
        addMaterial(createJelloMaterial());
        addMaterial(createSandMaterial());
        addMaterial(createSnowMaterial());
    }

    uint32_t addMaterial(const MaterialParameters& params) {
        materials_.push_back(params);
        return static_cast<uint32_t>(materials_.size() - 1);
    }

    const MaterialParameters& getMaterial(uint32_t material_id) const {
        return materials_[material_id];
    }

    MaterialParameters& getMaterial(uint32_t material_id) {
        return materials_[material_id];
    }

    size_t getNumMaterials() const {
        return materials_.size();
    }

    // Predefined material creators
    static MaterialParameters createWaterMaterial() {
        MaterialParameters params;
        params.type = MaterialType::FLUID;
        params.density = 1000.0f;
        params.viscosity = 0.001f;
        params.youngs_modulus = 2.2e9f;  // Bulk modulus approximation
        params.poisson_ratio = 0.5f;    // Incompressible
        return params;
    }

    static MaterialParameters createJelloMaterial() {
        MaterialParameters params;
        params.type = MaterialType::ELASTIC;
        params.density = 1100.0f;
        params.youngs_modulus = 1e4f;    // Very soft
        params.poisson_ratio = 0.3f;
        return params;
    }

    static MaterialParameters createSandMaterial() {
        MaterialParameters params;
        params.type = MaterialType::SAND;
        params.density = 1600.0f;
        params.youngs_modulus = 3.5e7f;
        params.poisson_ratio = 0.3f;
        params.friction_angle = 30.0f;
        params.cohesion = 0.0f;          // Cohesionless sand
        return params;
    }

    static MaterialParameters createSnowMaterial() {
        MaterialParameters params;
        params.type = MaterialType::SNOW;
        params.density = 400.0f;
        params.youngs_modulus = 1.4e5f;
        params.poisson_ratio = 0.2f;
        params.critical_compression = 2.5e-2f;
        params.critical_stretch = 7.5e-3f;
        params.hardening_coefficient = 10.0f;
        return params;
    }
};

// =============================================================================
// BOUNDARY CONDITIONS
// =============================================================================

enum class BoundaryType : uint32_t {
    FREE = 0,
    DIRICHLET_X = 1,
    DIRICHLET_Y = 2,
    DIRICHLET_Z = 4,
    NEUMANN_X = 8,
    NEUMANN_Y = 16,
    NEUMANN_Z = 32,
    STICKY = 64,         // No-slip condition
    SLIP = 128           // Free-slip condition
};

template<typename T>
struct BoundaryCondition {
    BoundaryType type;
    T3<T> value;           // Prescribed velocity or force
    T3<T> normal;          // Surface normal (for Neumann conditions)
};

} // namespace physgrad::mpm