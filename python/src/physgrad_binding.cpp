#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#ifdef WITH_PYTORCH
#include <torch/extension.h>
#include <torch/torch.h>
#endif

#ifdef WITH_JAX
#include "jax_integration.h"
#endif

#include "simulation.h"
#include "multi_gpu.h"
#include "differentiable_contact.h"
#include "rigid_body.h"
#include "symplectic_integrators.h"
#include "constraints.h"
#include "collision_detection.h"

#include "tensor_interop.h"
#include "torch_integration.h"
#include "../../src/variational_contact.h"
#ifdef WITH_CUDA
#include "../../src/variational_contact_gpu.h"
#endif

namespace py = pybind11;
using namespace physgrad;

// Forward declaration of variational contact bindings
void bind_variational_contact(py::module& m);

// Helper functions for Eigen<->NumPy conversion
std::vector<Eigen::Vector3d> numpy_to_eigen_vector3d(py::array_t<double> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Expected array shape (N, 3)");
    }

    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<Eigen::Vector3d> result;
    result.reserve(buf.shape[0]);

    for (size_t i = 0; i < buf.shape[0]; ++i) {
        result.emplace_back(ptr[i*3 + 0], ptr[i*3 + 1], ptr[i*3 + 2]);
    }
    return result;
}

py::array_t<double> eigen_vector3d_to_numpy(const std::vector<Eigen::Vector3d>& input) {
    auto result = py::array_t<double>({input.size(), 3});
    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (size_t i = 0; i < input.size(); ++i) {
        ptr[i*3 + 0] = input[i][0];
        ptr[i*3 + 1] = input[i][1];
        ptr[i*3 + 2] = input[i][2];
    }
    return result;
}

// Helper function to convert between numpy and std::vector
template<typename T>
std::vector<T> numpy_to_vector(py::array_t<T> input) {
    py::buffer_info buf_info = input.request();
    T* ptr = static_cast<T*>(buf_info.ptr);
    return std::vector<T>(ptr, ptr + buf_info.size);
}

template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& input) {
    return py::array_t<T>(
        input.size(),
        input.data(),
        py::cast(input) // This keeps the vector alive
    );
}

PYBIND11_MODULE(physgrad_cpp, m) {
    m.doc() = "PhysGrad: Differentiable Physics Simulation with GPU Acceleration";

    

    py::class_<SimulationParams>(m, "SimulationParams")
        .def(py::init<>())
        .def_readwrite("dt", &SimulationParams::dt)
        .def_readwrite("num_particles", &SimulationParams::num_particles)
        .def_readwrite("max_steps", &SimulationParams::max_steps)
        .def_readwrite("gravity_x", &SimulationParams::gravity_x)
        .def_readwrite("gravity_y", &SimulationParams::gravity_y)
        .def_readwrite("gravity_z", &SimulationParams::gravity_z)
        .def_readwrite("damping", &SimulationParams::damping)
        .def_readwrite("enable_collisions", &SimulationParams::enable_collisions)
        .def_readwrite("enable_constraints", &SimulationParams::enable_constraints);

    py::class_<PhysicsSimulation>(m, "PhysicsSimulation")
        .def(py::init<const SimulationParams&>())
        .def("initialize", &PhysicsSimulation::initialize)
        .def("step", &PhysicsSimulation::step)
        .def("reset", &PhysicsSimulation::reset)
        .def("get_particle_count", &PhysicsSimulation::getParticleCount)
        .def("get_time", &PhysicsSimulation::getTime)
        .def("set_positions", [](PhysicsSimulation& sim,
                               py::array_t<float> pos_x,
                               py::array_t<float> pos_y,
                               py::array_t<float> pos_z) {
            auto x_vec = numpy_to_vector(pos_x);
            auto y_vec = numpy_to_vector(pos_y);
            auto z_vec = numpy_to_vector(pos_z);
            sim.setPositions(x_vec, y_vec, z_vec);
        })
        .def("get_positions", [](PhysicsSimulation& sim) {
            std::vector<float> pos_x, pos_y, pos_z;
            sim.getPositions(pos_x, pos_y, pos_z);
            return py::make_tuple(
                vector_to_numpy(pos_x),
                vector_to_numpy(pos_y),
                vector_to_numpy(pos_z)
            );
        })
        .def("set_velocities", [](PhysicsSimulation& sim,
                                py::array_t<float> vel_x,
                                py::array_t<float> vel_y,
                                py::array_t<float> vel_z) {
            auto x_vec = numpy_to_vector(vel_x);
            auto y_vec = numpy_to_vector(vel_y);
            auto z_vec = numpy_to_vector(vel_z);
            sim.setVelocities(x_vec, y_vec, z_vec);
        })
        .def("get_velocities", [](PhysicsSimulation& sim) {
            std::vector<float> vel_x, vel_y, vel_z;
            sim.getVelocities(vel_x, vel_y, vel_z);
            return py::make_tuple(
                vector_to_numpy(vel_x),
                vector_to_numpy(vel_y),
                vector_to_numpy(vel_z)
            );
        })
        .def("set_masses", [](PhysicsSimulation& sim, py::array_t<float> masses) {
            auto mass_vec = numpy_to_vector(masses);
            sim.setMasses(mass_vec);
        })
        .def("get_masses", [](PhysicsSimulation& sim) {
            std::vector<float> masses;
            sim.getMasses(masses);
            return vector_to_numpy(masses);
        })
        .def("apply_force", [](PhysicsSimulation& sim, int particle_idx,
                             float fx, float fy, float fz) {
            sim.applyForce(particle_idx, fx, fy, fz);
        })
        .def("get_energy", &PhysicsSimulation::getTotalEnergy)
        .def("get_kinetic_energy", &PhysicsSimulation::getKineticEnergy)
        .def("get_potential_energy", &PhysicsSimulation::getPotentialEnergy);

    

    py::enum_<PartitioningStrategy>(m, "PartitioningStrategy")
        .value("SPATIAL_GRID", PartitioningStrategy::SPATIAL_GRID)
        .value("OCTREE", PartitioningStrategy::OCTREE)
        .value("HILBERT_CURVE", PartitioningStrategy::HILBERT_CURVE)
        .value("DYNAMIC_LOAD", PartitioningStrategy::DYNAMIC_LOAD)
        .value("PARTICLE_COUNT", PartitioningStrategy::PARTICLE_COUNT);

    py::enum_<CommunicationPattern>(m, "CommunicationPattern")
        .value("PEER_TO_PEER", CommunicationPattern::PEER_TO_PEER)
        .value("HOST_STAGING", CommunicationPattern::HOST_STAGING)
        .value("NCCL_COLLECTIVE", CommunicationPattern::NCCL_COLLECTIVE)
        .value("UNIFIED_MEMORY", CommunicationPattern::UNIFIED_MEMORY);

    py::class_<MultiGPUConfig>(m, "MultiGPUConfig")
        .def(py::init<>())
        .def_readwrite("device_ids", &MultiGPUConfig::device_ids)
        .def_readwrite("partitioning", &MultiGPUConfig::partitioning)
        .def_readwrite("communication", &MultiGPUConfig::communication)
        .def_readwrite("ghost_layer_width", &MultiGPUConfig::ghost_layer_width)
        .def_readwrite("load_balance_threshold", &MultiGPUConfig::load_balance_threshold)
        .def_readwrite("enable_dynamic_balancing", &MultiGPUConfig::enable_dynamic_balancing)
        .def_readwrite("enable_peer_access", &MultiGPUConfig::enable_peer_access)
        .def_readwrite("async_communication", &MultiGPUConfig::async_communication);

    py::class_<MultiGPUStats>(m, "MultiGPUStats")
        .def(py::init<>())
        .def_readonly("gpu_utilization", &MultiGPUStats::gpu_utilization)
        .def_readonly("particle_counts", &MultiGPUStats::particle_counts)
        .def_readonly("computation_times", &MultiGPUStats::computation_times)
        .def_readonly("communication_times", &MultiGPUStats::communication_times)
        .def_readonly("total_simulation_time", &MultiGPUStats::total_simulation_time)
        .def_readonly("load_balance_factor", &MultiGPUStats::load_balance_factor)
        .def_readonly("communication_overhead", &MultiGPUStats::communication_overhead)
        .def_readonly("rebalance_count", &MultiGPUStats::rebalance_count);

    py::class_<MultiGPUManager>(m, "MultiGPUManager")
        .def(py::init<const MultiGPUConfig&>())
        .def("initialize", &MultiGPUManager::initialize)
        .def("shutdown", &MultiGPUManager::shutdown)
        .def("get_device_count", &MultiGPUManager::getDeviceCount)
        .def("partition_domain", [](MultiGPUManager& mgr,
                                  py::array_t<float> pos_x,
                                  py::array_t<float> pos_y,
                                  py::array_t<float> pos_z) {
            auto x_vec = numpy_to_vector(pos_x);
            auto y_vec = numpy_to_vector(pos_y);
            auto z_vec = numpy_to_vector(pos_z);
            mgr.partitionDomain(x_vec, y_vec, z_vec);
        })
        .def("get_stats", &MultiGPUManager::getStats, py::return_value_policy::reference)
        .def("print_stats", &MultiGPUManager::printStats);

    

    py::class_<ConstraintParams>(m, "ConstraintParams")
        .def(py::init<>())
        .def_readwrite("compliance", &ConstraintParams::compliance)
        .def_readwrite("damping", &ConstraintParams::damping)
        .def_readwrite("breaking_force", &ConstraintParams::breaking_force)
        .def_readwrite("enabled", &ConstraintParams::enabled)
        .def_readwrite("rest_length", &ConstraintParams::rest_length)
        .def_readwrite("stiffness", &ConstraintParams::stiffness);

    py::class_<ConstraintSolver>(m, "ConstraintSolver")
        .def(py::init<>())
        .def("add_distance_constraint", &ConstraintSolver::addDistanceConstraint)
        .def("add_spring_constraint", &ConstraintSolver::addSpringConstraint)
        .def("add_position_lock", &ConstraintSolver::addPositionLock)
        .def("solve_constraints", [](ConstraintSolver& solver,
                                   py::array_t<float> pos_x, py::array_t<float> pos_y, py::array_t<float> pos_z,
                                   py::array_t<float> vel_x, py::array_t<float> vel_y, py::array_t<float> vel_z,
                                   py::array_t<float> masses, float dt) {
            auto x_vec = numpy_to_vector(pos_x);
            auto y_vec = numpy_to_vector(pos_y);
            auto z_vec = numpy_to_vector(pos_z);
            auto vx_vec = numpy_to_vector(vel_x);
            auto vy_vec = numpy_to_vector(vel_y);
            auto vz_vec = numpy_to_vector(vel_z);
            auto mass_vec = numpy_to_vector(masses);

            solver.solveConstraints(x_vec, y_vec, z_vec, vx_vec, vy_vec, vz_vec, mass_vec, dt);

            return py::make_tuple(
                vector_to_numpy(x_vec), vector_to_numpy(y_vec), vector_to_numpy(z_vec),
                vector_to_numpy(vx_vec), vector_to_numpy(vy_vec), vector_to_numpy(vz_vec)
            );
        });

    

    py::class_<CollisionParams>(m, "CollisionParams")
        .def(py::init<>())
        .def_readwrite("contact_threshold", &CollisionParams::contact_threshold)
        .def_readwrite("contact_stiffness", &CollisionParams::contact_stiffness)
        .def_readwrite("contact_damping", &CollisionParams::contact_damping)
        .def_readwrite("enable_restitution", &CollisionParams::enable_restitution)
        .def_readwrite("enable_friction", &CollisionParams::enable_friction)
        .def_readwrite("global_restitution", &CollisionParams::global_restitution)
        .def_readwrite("global_friction", &CollisionParams::global_friction);

    py::class_<ContactInfo>(m, "ContactInfo")
        .def(py::init<>())
        .def_readwrite("body_i", &ContactInfo::body_i)
        .def_readwrite("body_j", &ContactInfo::body_j)
        .def_readwrite("contact_distance", &ContactInfo::contact_distance)
        .def_readwrite("overlap", &ContactInfo::overlap)
        .def_readwrite("normal_x", &ContactInfo::normal_x)
        .def_readwrite("normal_y", &ContactInfo::normal_y)
        .def_readwrite("normal_z", &ContactInfo::normal_z)
        .def_readwrite("restitution", &ContactInfo::restitution)
        .def_readwrite("friction", &ContactInfo::friction);

    py::class_<CollisionDetector>(m, "CollisionDetector")
        .def(py::init<const CollisionParams&>())
        .def("detect_collisions", [](CollisionDetector& detector,
                                   py::array_t<float> pos_x,
                                   py::array_t<float> pos_y,
                                   py::array_t<float> pos_z) {
            auto x_vec = numpy_to_vector(pos_x);
            auto y_vec = numpy_to_vector(pos_y);
            auto z_vec = numpy_to_vector(pos_z);

            auto contacts = detector.detectCollisions(x_vec, y_vec, z_vec);
            return contacts;
        })
        .def("update_body_radii", [](CollisionDetector& detector, py::array_t<float> radii) {
            auto radii_vec = numpy_to_vector(radii);
            detector.updateBodyRadii(radii_vec);
        });

    

    py::enum_<SymplecticScheme>(m, "SymplecticScheme")
        .value("SYMPLECTIC_EULER", SymplecticScheme::SYMPLECTIC_EULER)
        .value("VELOCITY_VERLET", SymplecticScheme::VELOCITY_VERLET)
        .value("FOREST_RUTH", SymplecticScheme::FOREST_RUTH)
        .value("YOSHIDA4", SymplecticScheme::YOSHIDA4)
        .value("BLANES_MOAN8", SymplecticScheme::BLANES_MOAN8);

    py::class_<SymplecticParams>(m, "SymplecticParams")
        .def(py::init<>())
        .def_readwrite("time_step", &SymplecticParams::time_step)
        .def_readwrite("enable_energy_monitoring", &SymplecticParams::enable_energy_monitoring)
        .def_readwrite("enable_momentum_conservation", &SymplecticParams::enable_momentum_conservation)
        .def_readwrite("energy_tolerance", &SymplecticParams::energy_tolerance);

    

    m.def("get_available_gpus", &MultiGPUUtils::getAvailableGPUs,
          "Get list of available GPU device IDs");

    m.def("select_optimal_gpus", &MultiGPUUtils::selectOptimalGPUs,
          "Select optimal GPUs based on memory and performance",
          py::arg("desired_count"), py::arg("min_memory_gb") = 2.0f);

    

#ifdef WITH_PYTORCH
    m.def("simulation_step_torch", &torch_integration::simulationStepTorch,
          "Execute simulation step with PyTorch tensors",
          py::arg("positions"), py::arg("velocities"), py::arg("masses"),
          py::arg("forces"), py::arg("dt"));

    m.def("compute_forces_torch", &torch_integration::computeForcesTorch,
          "Compute forces using PyTorch tensors",
          py::arg("positions"), py::arg("velocities"), py::arg("masses"));

    m.def("apply_constraints_torch", &torch_integration::applyConstraintsTorch,
          "Apply constraints using PyTorch tensors",
          py::arg("positions"), py::arg("velocities"), py::arg("masses"),
          py::arg("constraint_params"));

    // Custom autograd functions
    py::class_<TorchPhysicsFunction>(m, "TorchPhysicsFunction")
        .def_static("apply", &TorchPhysicsFunction::apply)
        .def_static("setup_context", &TorchPhysicsFunction::setup_context)
        .def_static("forward", &TorchPhysicsFunction::forward)
        .def_static("backward", &TorchPhysicsFunction::backward);
#endif

    

#ifdef WITH_JAX
    m.def("register_jax_primitives", &jax_integration::registerJAXPrimitives,
          "Register JAX XLA primitives for physics operations");

    m.def("simulation_step_jax", &jax_integration::simulationStepJAX,
          "Execute simulation step with JAX arrays");

    m.def("compute_forces_jax", &jax_integration::computeForcesJAX,
          "Compute forces using JAX arrays");
#endif

    

    // Bind variational contact system
    bind_variational_contact(m);

    m.attr("__version__") = "0.1.0";
    m.attr("cuda_available") = true;

#ifdef WITH_PYTORCH
    m.attr("pytorch_available") = true;
#else
    m.attr("pytorch_available") = false;
#endif

#ifdef WITH_JAX
    m.attr("jax_available") = true;
#else
    m.attr("jax_available") = false;
#endif
}

// Implementation of variational contact bindings
void bind_variational_contact(py::module& m) {
    // Variational Contact Parameters
    py::class_<VariationalContactParams>(m, "VariationalContactParams",
        "Parameters for mathematically rigorous variational contact formulation")
        .def(py::init<>())
        .def_readwrite("barrier_stiffness", &VariationalContactParams::barrier_stiffness,
            "Barrier function stiffness κ (default: 1e6)")
        .def_readwrite("barrier_threshold", &VariationalContactParams::barrier_threshold,
            "Distance threshold δ for barrier activation (default: 1e-4)")
        .def_readwrite("friction_regularization", &VariationalContactParams::friction_regularization,
            "Huber regularization for friction ε (default: 1e-6)")
        .def_readwrite("max_newton_iterations", &VariationalContactParams::max_newton_iterations,
            "Maximum Newton-Raphson iterations (default: 50)")
        .def_readwrite("newton_tolerance", &VariationalContactParams::newton_tolerance,
            "Newton convergence tolerance (default: 1e-10)")
        .def_readwrite("enable_energy_conservation", &VariationalContactParams::enable_energy_conservation,
            "Enable energy conservation guarantees (default: true)")
        .def_readwrite("enable_momentum_conservation", &VariationalContactParams::enable_momentum_conservation,
            "Enable momentum conservation guarantees (default: true)");

    // Variational Contact Solver (CPU version)
    py::class_<VariationalContactSolver>(m, "VariationalContactSolver",
        "Mathematically rigorous contact solver with theoretical guarantees")
        .def(py::init<const VariationalContactParams&>(), py::arg("params") = VariationalContactParams{},
            "Initialize with contact parameters")

        .def("compute_contact_forces", [](VariationalContactSolver& self,
                py::array_t<double> positions, py::array_t<double> velocities,
                py::array_t<double> masses, py::array_t<double> radii,
                py::array_t<int> material_ids) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto vel_vec = numpy_to_eigen_vector3d(velocities);
            auto mass_vec = py::cast<std::vector<double>>(masses);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);

            std::vector<Eigen::Vector3d> forces;
            self.computeContactForces(pos_vec, vel_vec, mass_vec, radii_vec, mat_vec, forces);
            return eigen_vector3d_to_numpy(forces);
        }, "Compute contact forces with mathematical guarantees",
           py::arg("positions"), py::arg("velocities"), py::arg("masses"),
           py::arg("radii"), py::arg("material_ids"))

        .def("compute_contact_energy", [](VariationalContactSolver& self,
                py::array_t<double> positions, py::array_t<double> radii,
                py::array_t<int> material_ids) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);
            return self.computeContactEnergy(pos_vec, radii_vec, mat_vec);
        }, "Compute total contact potential energy",
           py::arg("positions"), py::arg("radii"), py::arg("material_ids"))

        .def("verify_gradient_correctness", [](VariationalContactSolver& self,
                py::array_t<double> positions, py::array_t<double> velocities,
                py::array_t<double> masses, py::array_t<double> radii,
                py::array_t<int> material_ids, double tolerance) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto vel_vec = numpy_to_eigen_vector3d(velocities);
            auto mass_vec = py::cast<std::vector<double>>(masses);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);
            return self.verifyGradientCorrectness(pos_vec, vel_vec, mass_vec, radii_vec, mat_vec, tolerance);
        }, "Verify gradient correctness against analytical derivatives",
           py::arg("positions"), py::arg("velocities"), py::arg("masses"),
           py::arg("radii"), py::arg("material_ids"), py::arg("tolerance") = 1e-8);

    // Variational Contact Integrator
    py::class_<VariationalContactIntegrator>(m, "VariationalContactIntegrator",
        "Hybrid implicit-explicit integrator with provable stability")
        .def(py::init<const VariationalContactParams&>(), py::arg("contact_params") = VariationalContactParams{},
            "Initialize with contact parameters")

        .def("integrate_step", [](VariationalContactIntegrator& self,
                py::array_t<double> positions, py::array_t<double> velocities,
                py::array_t<double> masses, py::array_t<double> radii,
                py::array_t<int> material_ids, double dt) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto vel_vec = numpy_to_eigen_vector3d(velocities);
            auto mass_vec = py::cast<std::vector<double>>(masses);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);

            std::vector<Eigen::Vector3d> ext_forces; // Empty for now

            double actual_dt = self.integrateStep(pos_vec, vel_vec, mass_vec, radii_vec,
                                                mat_vec, dt, ext_forces);

            return py::make_tuple(eigen_vector3d_to_numpy(pos_vec),
                                eigen_vector3d_to_numpy(vel_vec), actual_dt);
        }, "Integrate one time step with adaptive timestepping",
           py::arg("positions"), py::arg("velocities"), py::arg("masses"),
           py::arg("radii"), py::arg("material_ids"), py::arg("dt"));

#ifdef WITH_CUDA
    // GPU-accelerated versions
    py::class_<VariationalContactSolverGPU>(m, "VariationalContactSolverGPU",
        "GPU-accelerated variational contact solver with identical API")
        .def(py::init<const VariationalContactParams&>(), py::arg("params") = VariationalContactParams{},
            "Initialize GPU solver with contact parameters")

        .def("compute_contact_forces", [](VariationalContactSolverGPU& self,
                py::array_t<double> positions, py::array_t<double> velocities,
                py::array_t<double> masses, py::array_t<double> radii,
                py::array_t<int> material_ids) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto vel_vec = numpy_to_eigen_vector3d(velocities);
            auto mass_vec = py::cast<std::vector<double>>(masses);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);

            std::vector<Eigen::Vector3d> forces;
            self.computeContactForces(pos_vec, vel_vec, mass_vec, radii_vec, mat_vec, forces);
            return eigen_vector3d_to_numpy(forces);
        }, "GPU-accelerated contact force computation",
           py::arg("positions"), py::arg("velocities"), py::arg("masses"),
           py::arg("radii"), py::arg("material_ids"))

        .def_static("is_gpu_available", &VariationalContactSolverGPU::isGPUAvailable,
            "Check if CUDA-capable GPU is available");
#endif
}