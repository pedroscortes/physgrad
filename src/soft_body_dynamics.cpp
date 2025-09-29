/**
 * Soft Body Dynamics Implementation
 *
 * Finite Element Method with optional GPU acceleration
 * Complete integration with contact mechanics and fluid dynamics
 */

#include "soft_body_dynamics.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>

#ifdef HAVE_CUDA
#include "soft_body_kernels.cuh"
#include "variational_contact_gpu.cuh"
#include "sph_kernels.cuh"
#include <cuda_runtime.h>
#endif

namespace physgrad {

/**
 * GPU data management for soft body simulation
 */
class SoftBodySolver::SoftBodyGPUData {
public:
    // Device memory for nodes and elements
    thrust::device_vector<GPUSoftBodyNode> d_nodes;
    thrust::device_vector<GPUFiniteElement> d_elements;

    // Element connectivity data
    thrust::device_vector<int> d_element_connectivity;
    thrust::device_vector<int> d_node_element_counts;

    // Self-contact detection
    thrust::device_vector<float> d_contact_pairs;
    thrust::device_vector<int> d_contact_count;

    // Energy computation
    thrust::device_vector<float> d_kinetic_energy;
    thrust::device_vector<float> d_elastic_energy;

    // Matrix storage for implicit methods
    thrust::device_vector<float> d_stiffness_matrix;
    thrust::device_vector<float> d_mass_matrix;
    thrust::device_vector<float> d_force_vector;

    // Performance monitoring
    size_t allocated_bytes = 0;
    int max_nodes = 0;
    int max_elements = 0;
    int current_nodes = 0;
    int current_elements = 0;

    SoftBodyGPUData(int max_nodes_count, int max_elements_count)
        : max_nodes(max_nodes_count), max_elements(max_elements_count) {
        allocateMemory();
    }

    ~SoftBodyGPUData() {
        // Thrust vectors automatically deallocate
    }

private:
    void allocateMemory() {
        try {
            // Allocate node and element data
            d_nodes.resize(max_nodes);
            d_elements.resize(max_elements);

            // Element connectivity (assume 4 nodes per tetrahedral element)
            d_element_connectivity.resize(max_elements * 4);
            d_node_element_counts.resize(max_nodes);

            // Self-contact detection
            d_contact_pairs.resize(20000);  // Max contact pairs
            d_contact_count.resize(1);

            // Energy computation
            d_kinetic_energy.resize(1);
            d_elastic_energy.resize(1);

            // Matrix storage (sparse format - simplified)
            d_stiffness_matrix.resize(max_nodes * 9);  // 3x3 per node simplified
            d_mass_matrix.resize(max_nodes * 9);
            d_force_vector.resize(max_nodes * 3);

            // Calculate memory usage
            allocated_bytes = sizeof(GPUSoftBodyNode) * max_nodes +
                            sizeof(GPUFiniteElement) * max_elements +
                            sizeof(int) * max_elements * 4 +
                            sizeof(int) * max_nodes +
                            sizeof(float) * 20000 +
                            sizeof(int) * 1 +
                            sizeof(float) * 2 +
                            sizeof(float) * max_nodes * 21;  // Matrices and vectors

            std::cout << "Soft Body GPU Memory allocated: "
                      << allocated_bytes / (1024.0 * 1024.0) << " MB for "
                      << max_nodes << " nodes, " << max_elements << " elements" << std::endl;

        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to allocate Soft Body GPU memory: " + std::string(e.what()));
        }
    }
};

SoftBodySolver::SoftBodySolver(const SoftBodyMaterial& material) : default_material_(material) {
    // Initialize GPU data with reasonable default size
    int initial_nodes = 10000;    // 10k nodes
    int initial_elements = 50000; // 50k elements
    gpu_data_ = std::make_unique<SoftBodyGPUData>(initial_nodes, initial_elements);

    // Copy material parameters to GPU constant memory
    cudaMemcpyToSymbol(c_youngs_modulus, &material.youngs_modulus, sizeof(float));
    cudaMemcpyToSymbol(c_poissons_ratio, &material.poissons_ratio, sizeof(float));
    cudaMemcpyToSymbol(c_density, &material.density, sizeof(float));
    cudaMemcpyToSymbol(c_shear_modulus, &material.shear_modulus, sizeof(float));
    cudaMemcpyToSymbol(c_bulk_modulus, &material.bulk_modulus, sizeof(float));
    cudaMemcpyToSymbol(c_rayleigh_alpha, &material.rayleigh_alpha, sizeof(float));
    cudaMemcpyToSymbol(c_rayleigh_beta, &material.rayleigh_beta, sizeof(float));
    cudaMemcpyToSymbol(c_damage_threshold, &material.damage_threshold, sizeof(float));
    cudaMemcpyToSymbol(c_fracture_toughness, &material.fracture_toughness, sizeof(float));

    float3 gravity = make_float3(0.0f, -9.81f, 0.0f);
    cudaMemcpyToSymbol(c_gravity, &gravity, sizeof(float3));

    // Newmark integration parameters
    float newmark_beta = 0.25f;    // Average acceleration
    float newmark_gamma = 0.5f;    // Trapezoidal rule
    cudaMemcpyToSymbol(c_newmark_beta, &newmark_beta, sizeof(float));
    cudaMemcpyToSymbol(c_newmark_gamma, &newmark_gamma, sizeof(float));

    std::cout << "Soft Body Solver initialized with GPU acceleration" << std::endl;
    std::cout << "Material: E=" << material.youngs_modulus/1e6 << " MPa, ν=" << material.poissons_ratio
              << ", ρ=" << material.density << " kg/m³" << std::endl;
}

SoftBodySolver::~SoftBodySolver() = default;

void SoftBodySolver::createTetrahedralMesh(const Eigen::Vector3d& min_corner,
                                          const Eigen::Vector3d& max_corner,
                                          double characteristic_length) {
    std::cout << "Generating tetrahedral mesh..." << std::endl;

    // Simple regular tetrahedral mesh generation
    double spacing = characteristic_length;

    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::vector<int>> tetrahedral_elements;

    // Generate vertices in a regular grid
    int nx = static_cast<int>((max_corner.x() - min_corner.x()) / spacing) + 1;
    int ny = static_cast<int>((max_corner.y() - min_corner.y()) / spacing) + 1;
    int nz = static_cast<int>((max_corner.z() - min_corner.z()) / spacing) + 1;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                double x = min_corner.x() + i * spacing;
                double y = min_corner.y() + j * spacing;
                double z = min_corner.z() + k * spacing;
                vertices.emplace_back(x, y, z);
            }
        }
    }

    // Generate tetrahedral elements (simplified cubic decomposition)
    for (int i = 0; i < nx-1; ++i) {
        for (int j = 0; j < ny-1; ++j) {
            for (int k = 0; k < nz-1; ++k) {
                // Each cube is decomposed into 6 tetrahedra
                int base = i * ny * nz + j * nz + k;

                std::vector<int> cube_nodes = {
                    base,                        // 0
                    base + nz,                   // 1
                    base + ny*nz,               // 2
                    base + nz + ny*nz,          // 3
                    base + 1,                    // 4
                    base + nz + 1,              // 5
                    base + ny*nz + 1,           // 6
                    base + nz + ny*nz + 1       // 7
                };

                // Decompose cube into 6 tetrahedra (standard pattern)
                tetrahedral_elements.push_back({cube_nodes[0], cube_nodes[1], cube_nodes[2], cube_nodes[4]});
                tetrahedral_elements.push_back({cube_nodes[1], cube_nodes[2], cube_nodes[3], cube_nodes[5]});
                tetrahedral_elements.push_back({cube_nodes[2], cube_nodes[4], cube_nodes[5], cube_nodes[6]});
                tetrahedral_elements.push_back({cube_nodes[1], cube_nodes[4], cube_nodes[5], cube_nodes[7]});
                tetrahedral_elements.push_back({cube_nodes[2], cube_nodes[5], cube_nodes[6], cube_nodes[7]});
                tetrahedral_elements.push_back({cube_nodes[1], cube_nodes[2], cube_nodes[5], cube_nodes[7]});
            }
        }
    }

    std::cout << "Generated mesh: " << vertices.size() << " vertices, "
              << tetrahedral_elements.size() << " tetrahedra" << std::endl;

    // Add to soft body
    addSoftBody(vertices, tetrahedral_elements, default_material_);
}

void SoftBodySolver::addSoftBody(const std::vector<Eigen::Vector3d>& vertices,
                                 const std::vector<std::vector<int>>& elements,
                                 const SoftBodyMaterial& material) {
    int new_nodes = vertices.size();
    int new_elements = elements.size();
    int current_node_count = nodes_.size();
    int current_element_count = elements_.size();

    // Check if we need to reallocate GPU memory
    if (current_node_count + new_nodes > gpu_data_->max_nodes ||
        current_element_count + new_elements > gpu_data_->max_elements) {
        int new_node_capacity = std::max(gpu_data_->max_nodes * 2, current_node_count + new_nodes);
        int new_element_capacity = std::max(gpu_data_->max_elements * 2, current_element_count + new_elements);
        allocateGPUMemory(new_node_capacity, new_element_capacity);
    }

    // Add nodes
    for (size_t i = 0; i < vertices.size(); ++i) {
        SoftBodyNode node;
        node.position = vertices[i];
        node.reference_position = vertices[i];
        node.velocity = Eigen::Vector3d::Zero();
        node.acceleration = Eigen::Vector3d::Zero();
        node.displacement = Eigen::Vector3d::Zero();

        node.internal_force = Eigen::Vector3d::Zero();
        node.external_force = Eigen::Vector3d::Zero();
        node.contact_force = Eigen::Vector3d::Zero();
        node.fluid_force = Eigen::Vector3d::Zero();

        node.mass = material.density * 0.001; // Approximate mass from density
        node.temperature = 293.15;
        node.is_fixed = false;
        node.prescribed_displacement = Eigen::Vector3d::Zero();

        nodes_.push_back(node);
    }

    // Add elements
    for (size_t i = 0; i < elements.size(); ++i) {
        if (elements[i].size() != 4) {
            throw std::invalid_argument("Only tetrahedral elements (4 nodes) are currently supported");
        }

        FiniteElement element;
        element.type = ElementType::TETRAHEDRON_4;
        element.node_indices = elements[i];

        // Offset node indices by current node count
        for (int& idx : element.node_indices) {
            idx += current_node_count;
        }

        // Compute initial element volume
        const auto& v0 = nodes_[element.node_indices[0]].position;
        const auto& v1 = nodes_[element.node_indices[1]].position;
        const auto& v2 = nodes_[element.node_indices[2]].position;
        const auto& v3 = nodes_[element.node_indices[3]].position;

        Eigen::Vector3d edge1 = v1 - v0;
        Eigen::Vector3d edge2 = v2 - v0;
        Eigen::Vector3d edge3 = v3 - v0;

        element.volume = std::abs(edge1.dot(edge2.cross(edge3))) / 6.0;

        element.deformation_gradient = Eigen::Matrix3d::Identity();
        element.jacobian_determinant = 1.0;
        element.stress_tensor = Eigen::Matrix3d::Zero();
        element.strain_tensor = Eigen::Matrix3d::Zero();
        element.strain_energy = 0.0;
        element.damage_parameter = 0.0;
        element.plastic_strain = Eigen::Matrix3d::Zero();
        element.equivalent_plastic_strain = 0.0;
        element.material = material;

        elements_.push_back(element);
    }

    // Update node connectivity
    updateNodeConnectivity();

    // Convert to GPU format and upload
    uploadToGPU();

    std::cout << "Added soft body: " << new_nodes << " nodes, " << new_elements
              << " elements (total: " << nodes_.size() << " nodes, "
              << elements_.size() << " elements)" << std::endl;
}

void SoftBodySolver::simulateStep(double dt) {
    if (nodes_.empty() || elements_.empty()) return;

    int n_nodes = gpu_data_->current_nodes;
    int n_elements = gpu_data_->current_elements;

    // Compute optimal block size for our GPU
    int block_size = 256;
    int grid_size_nodes = (n_nodes + block_size - 1) / block_size;
    int grid_size_elements = (n_elements + block_size - 1) / block_size;

    // Update time step in constant memory
    float gpu_dt = static_cast<float>(dt);
    cudaMemcpyToSymbol(c_time_step, &gpu_dt, sizeof(float));

    try {
        // Step 1: Reset force arrays
        resetForceArraysKernel<<<grid_size_nodes, block_size>>>(
            thrust::raw_pointer_cast(gpu_data_->d_nodes.data()),
            n_nodes
        );

        // Step 2: Update deformation gradient
        updateDeformationGradientKernel<<<grid_size_elements, block_size>>>(
            thrust::raw_pointer_cast(gpu_data_->d_nodes.data()),
            thrust::raw_pointer_cast(gpu_data_->d_elements.data()),
            n_elements
        );

        // Step 3: Compute stress and strain
        computeStressStrainKernel<<<grid_size_elements, block_size>>>(
            thrust::raw_pointer_cast(gpu_data_->d_elements.data()),
            n_elements
        );

        // Step 4: Update damage (if enabled)
        updateDamageKernel<<<grid_size_elements, block_size>>>(
            thrust::raw_pointer_cast(gpu_data_->d_elements.data()),
            n_elements,
            gpu_dt
        );

        // Step 5: Compute internal forces
        computeInternalForcesKernel<<<grid_size_elements, block_size>>>(
            thrust::raw_pointer_cast(gpu_data_->d_nodes.data()),
            thrust::raw_pointer_cast(gpu_data_->d_elements.data()),
            n_elements
        );

        // Step 6: Apply damping
        applyDampingKernel<<<grid_size_nodes, block_size>>>(
            thrust::raw_pointer_cast(gpu_data_->d_nodes.data()),
            n_nodes
        );

        // Step 7: Time integration
        explicitTimeIntegrationKernel<<<grid_size_nodes, block_size>>>(
            thrust::raw_pointer_cast(gpu_data_->d_nodes.data()),
            n_nodes,
            gpu_dt
        );

        // Step 8: Enforce boundary conditions
        enforceBoundaryConditionsKernel<<<grid_size_nodes, block_size>>>(
            thrust::raw_pointer_cast(gpu_data_->d_nodes.data()),
            n_nodes
        );

        // Synchronize GPU
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

    } catch (const std::exception& e) {
        throw std::runtime_error("Soft body simulation step failed: " + std::string(e.what()));
    }
}

void SoftBodySolver::updateNodeConnectivity() {
    // Clear existing connectivity
    for (auto& node : nodes_) {
        node.connected_elements.clear();
    }

    // Build connectivity from elements
    for (size_t elem_idx = 0; elem_idx < elements_.size(); ++elem_idx) {
        const auto& element = elements_[elem_idx];
        for (int node_idx : element.node_indices) {
            if (node_idx >= 0 && node_idx < static_cast<int>(nodes_.size())) {
                nodes_[node_idx].connected_elements.push_back(elem_idx);
            }
        }
    }
}

void SoftBodySolver::uploadToGPU() {
    // Convert nodes to GPU format
    std::vector<GPUSoftBodyNode> gpu_nodes(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& n = nodes_[i];
        auto& gn = gpu_nodes[i];

        gn.position = make_float3(n.position.x(), n.position.y(), n.position.z());
        gn.velocity = make_float3(n.velocity.x(), n.velocity.y(), n.velocity.z());
        gn.acceleration = make_float3(n.acceleration.x(), n.acceleration.y(), n.acceleration.z());
        gn.reference_position = make_float3(n.reference_position.x(), n.reference_position.y(), n.reference_position.z());
        gn.displacement = make_float3(n.displacement.x(), n.displacement.y(), n.displacement.z());

        gn.internal_force = make_float3(n.internal_force.x(), n.internal_force.y(), n.internal_force.z());
        gn.external_force = make_float3(n.external_force.x(), n.external_force.y(), n.external_force.z());
        gn.contact_force = make_float3(n.contact_force.x(), n.contact_force.y(), n.contact_force.z());
        gn.fluid_force = make_float3(n.fluid_force.x(), n.fluid_force.y(), n.fluid_force.z());

        gn.mass = n.mass;
        gn.temperature = n.temperature;
        gn.is_fixed = n.is_fixed ? 1 : 0;
        gn.prescribed_displacement = make_float3(n.prescribed_displacement.x(), n.prescribed_displacement.y(), n.prescribed_displacement.z());

        gn.element_count = n.connected_elements.size();
        gn.element_start_index = i * 10; // Simplified connectivity storage
    }

    // Convert elements to GPU format
    std::vector<GPUFiniteElement> gpu_elements(elements_.size());
    for (size_t i = 0; i < elements_.size(); ++i) {
        const auto& e = elements_[i];
        auto& ge = gpu_elements[i];

        // Copy node indices
        for (int j = 0; j < 4; ++j) {
            ge.node_indices[j] = j < e.node_indices.size() ? e.node_indices[j] : -1;
        }

        ge.volume = e.volume;
        ge.jacobian_determinant = e.jacobian_determinant;

        // Copy deformation gradient (3x3 matrix as 9 floats)
        for (int j = 0; j < 9; ++j) {
            int row = j / 3;
            int col = j % 3;
            ge.deformation_gradient[j] = e.deformation_gradient(row, col);
        }

        // Copy stress and strain tensors (symmetric, 6 components)
        ge.stress_tensor[0] = e.stress_tensor(0,0); // xx
        ge.stress_tensor[1] = e.stress_tensor(1,1); // yy
        ge.stress_tensor[2] = e.stress_tensor(2,2); // zz
        ge.stress_tensor[3] = e.stress_tensor(0,1); // xy
        ge.stress_tensor[4] = e.stress_tensor(0,2); // xz
        ge.stress_tensor[5] = e.stress_tensor(1,2); // yz

        ge.strain_tensor[0] = e.strain_tensor(0,0);
        ge.strain_tensor[1] = e.strain_tensor(1,1);
        ge.strain_tensor[2] = e.strain_tensor(2,2);
        ge.strain_tensor[3] = e.strain_tensor(0,1);
        ge.strain_tensor[4] = e.strain_tensor(0,2);
        ge.strain_tensor[5] = e.strain_tensor(1,2);

        ge.strain_energy = e.strain_energy;
        ge.damage_parameter = e.damage_parameter;
        ge.equivalent_plastic_strain = e.equivalent_plastic_strain;

        // Material properties
        ge.youngs_modulus = e.material.youngs_modulus;
        ge.poissons_ratio = e.material.poissons_ratio;
        ge.density = e.material.density;
        ge.shear_modulus = e.material.shear_modulus;
        ge.bulk_modulus = e.material.bulk_modulus;
    }

    // Upload to GPU
    thrust::copy(gpu_nodes.begin(), gpu_nodes.end(), gpu_data_->d_nodes.begin());
    thrust::copy(gpu_elements.begin(), gpu_elements.end(), gpu_data_->d_elements.begin());

    gpu_data_->current_nodes = nodes_.size();
    gpu_data_->current_elements = elements_.size();
}

std::vector<Eigen::Vector3d> SoftBodySolver::getPositions() const {
    if (nodes_.empty()) return {};

    // Download from GPU
    std::vector<GPUSoftBodyNode> gpu_nodes(gpu_data_->current_nodes);
    thrust::copy(gpu_data_->d_nodes.begin(),
                gpu_data_->d_nodes.begin() + gpu_data_->current_nodes,
                gpu_nodes.begin());

    std::vector<Eigen::Vector3d> positions;
    positions.reserve(gpu_nodes.size());

    for (const auto& gn : gpu_nodes) {
        positions.emplace_back(gn.position.x, gn.position.y, gn.position.z);
    }

    return positions;
}

std::vector<double> SoftBodySolver::getVonMisesStress() const {
    if (elements_.empty()) return {};

    std::vector<GPUFiniteElement> gpu_elements(gpu_data_->current_elements);
    thrust::copy(gpu_data_->d_elements.begin(),
                gpu_data_->d_elements.begin() + gpu_data_->current_elements,
                gpu_elements.begin());

    std::vector<double> von_mises_stress;
    von_mises_stress.reserve(gpu_elements.size());

    for (const auto& ge : gpu_elements) {
        // Von Mises stress = sqrt(1.5 * dev(σ) : dev(σ))
        double s11 = ge.stress_tensor[0];
        double s22 = ge.stress_tensor[1];
        double s33 = ge.stress_tensor[2];
        double s12 = ge.stress_tensor[3];
        double s13 = ge.stress_tensor[4];
        double s23 = ge.stress_tensor[5];

        double mean_stress = (s11 + s22 + s33) / 3.0;
        double dev11 = s11 - mean_stress;
        double dev22 = s22 - mean_stress;
        double dev33 = s33 - mean_stress;

        double von_mises = std::sqrt(1.5 * (dev11*dev11 + dev22*dev22 + dev33*dev33 +
                                           2.0*(s12*s12 + s13*s13 + s23*s23)));

        von_mises_stress.push_back(von_mises);
    }

    return von_mises_stress;
}

double SoftBodySolver::getTotalElasticEnergy() const {
    if (elements_.empty()) return 0.0;

    int n_elements = gpu_data_->current_elements;
    int block_size = 256;
    int grid_size = (n_elements + block_size - 1) / block_size;

    // Reset energy values
    thrust::fill(gpu_data_->d_kinetic_energy.begin(), gpu_data_->d_kinetic_energy.end(), 0.0f);
    thrust::fill(gpu_data_->d_elastic_energy.begin(), gpu_data_->d_elastic_energy.end(), 0.0f);

    // Compute energy on GPU
    computeSoftBodyEnergyKernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(gpu_data_->d_nodes.data()),
        thrust::raw_pointer_cast(gpu_data_->d_elements.data()),
        gpu_data_->current_nodes,
        n_elements,
        thrust::raw_pointer_cast(gpu_data_->d_kinetic_energy.data()),
        thrust::raw_pointer_cast(gpu_data_->d_elastic_energy.data())
    );

    // Download result
    float elastic_energy = gpu_data_->d_elastic_energy[0];
    return static_cast<double>(elastic_energy);
}

void SoftBodySolver::allocateGPUMemory(int max_nodes, int max_elements) {
    if (max_nodes <= gpu_data_->max_nodes && max_elements <= gpu_data_->max_elements) return;

    std::cout << "Reallocating Soft Body GPU memory for " << max_nodes
              << " nodes, " << max_elements << " elements..." << std::endl;

    // Create new GPU data with larger capacity
    auto new_gpu_data = std::make_unique<SoftBodyGPUData>(max_nodes, max_elements);

    // Copy existing data if any
    if (gpu_data_->current_nodes > 0) {
        thrust::copy(gpu_data_->d_nodes.begin(),
                    gpu_data_->d_nodes.begin() + gpu_data_->current_nodes,
                    new_gpu_data->d_nodes.begin());
        new_gpu_data->current_nodes = gpu_data_->current_nodes;
    }

    if (gpu_data_->current_elements > 0) {
        thrust::copy(gpu_data_->d_elements.begin(),
                    gpu_data_->d_elements.begin() + gpu_data_->current_elements,
                    new_gpu_data->d_elements.begin());
        new_gpu_data->current_elements = gpu_data_->current_elements;
    }

    // Replace old data
    gpu_data_ = std::move(new_gpu_data);
}

size_t SoftBodySolver::getGPUMemoryUsage() const {
    return gpu_data_->allocated_bytes;
}

/**
 * Unified Multi-Physics Solver Implementation
 */
UnifiedPhysicsSolver::UnifiedPhysicsSolver(const VariationalContactParams& contact_params,
                                          const SPHParams& fluid_params,
                                          const SoftBodyMaterial& soft_body_material) {
    contact_solver_ = std::make_unique<VariationalContactSolver>(contact_params);
    fluid_solver_ = std::make_unique<SPHFluidSolver>(fluid_params);
    soft_body_solver_ = std::make_unique<SoftBodySolver>(soft_body_material);

    std::cout << "Unified Physics Solver initialized: Contacts + Fluids + Soft Bodies" << std::endl;
}

void UnifiedPhysicsSolver::addSoftBody(const std::vector<Eigen::Vector3d>& vertices,
                                      const std::vector<std::vector<int>>& elements) {
    soft_body_solver_->addSoftBody(vertices, elements, SoftBodyMaterial{});
    std::cout << "Added soft body with " << vertices.size() << " vertices to unified solver" << std::endl;
}

void UnifiedPhysicsSolver::simulateStep(double dt) {
    // Step 1: Simulate individual physics
    if (contact_solver_) {
        // Contact mechanics simulation would go here
        // (Integration with existing rigid bodies)
    }

    if (fluid_solver_) {
        fluid_solver_->simulateStep(dt);
    }

    if (soft_body_solver_) {
        soft_body_solver_->simulateStep(dt);
    }

    // Step 2: Couple all physics together
    coupleRigidFluidSoft(dt);
}

void UnifiedPhysicsSolver::coupleRigidFluidSoft(double dt) {
    // This is where we'd implement detailed multi-physics coupling
    // For now, placeholder - full implementation would include:
    // 1. Fluid forces on soft bodies
    // 2. Soft body deformation affecting fluid flow
    // 3. Contact forces between soft bodies and rigid bodies
    // 4. Complex three-way interactions

    // The coupling would use specialized kernels that combine
    // SPH, FEM, and contact mechanics computations
}

double UnifiedPhysicsSolver::getTotalSystemEnergy() const {
    double total_energy = 0.0;

    if (fluid_solver_) {
        total_energy += fluid_solver_->getKineticEnergy();
    }

    if (soft_body_solver_) {
        total_energy += soft_body_solver_->getTotalKineticEnergy();
        total_energy += soft_body_solver_->getTotalElasticEnergy();
    }

    return total_energy;
}

size_t UnifiedPhysicsSolver::getTotalObjectCount() const {
    size_t count = 0;

    if (fluid_solver_) {
        count += fluid_solver_->getParticles().size();
    }

    if (soft_body_solver_) {
        count += soft_body_solver_->getNodes().size();
    }

    return count;
}

} // namespace physgrad