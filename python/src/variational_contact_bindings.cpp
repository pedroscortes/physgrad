#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "../../src/variational_contact.h"
#ifdef WITH_CUDA
#include "../../src/variational_contact_gpu.h"
#endif

namespace py = pybind11;
using namespace physgrad;

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
        .def_readwrite("contact_regularization", &VariationalContactParams::contact_regularization,
            "Contact detection smoothing σ (default: 1e-8)")
        .def_readwrite("max_newton_iterations", &VariationalContactParams::max_newton_iterations,
            "Maximum Newton-Raphson iterations (default: 50)")
        .def_readwrite("newton_tolerance", &VariationalContactParams::newton_tolerance,
            "Newton convergence tolerance (default: 1e-10)")
        .def_readwrite("line_search_alpha", &VariationalContactParams::line_search_alpha,
            "Line search parameter α (default: 1e-4)")
        .def_readwrite("line_search_beta", &VariationalContactParams::line_search_beta,
            "Line search parameter β (default: 0.8)")
        .def_readwrite("enable_energy_conservation", &VariationalContactParams::enable_energy_conservation,
            "Enable energy conservation guarantees (default: true)")
        .def_readwrite("enable_momentum_conservation", &VariationalContactParams::enable_momentum_conservation,
            "Enable momentum conservation guarantees (default: true)")
        .def_readwrite("enable_gradient_consistency", &VariationalContactParams::enable_gradient_consistency,
            "Enable gradient consistency verification (default: true)")
        .def_readwrite("enable_rolling_resistance", &VariationalContactParams::enable_rolling_resistance,
            "Enable rolling resistance modeling (default: false)")
        .def_readwrite("enable_adhesion_forces", &VariationalContactParams::enable_adhesion_forces,
            "Enable adhesion force modeling (default: false)")
        .def_readwrite("adhesion_strength", &VariationalContactParams::adhesion_strength,
            "Adhesion force strength (default: 1e-3)")
        .def("__repr__", [](const VariationalContactParams& p) {
            return "<VariationalContactParams(barrier_stiffness=" + std::to_string(p.barrier_stiffness) +
                   ", barrier_threshold=" + std::to_string(p.barrier_threshold) + ")>";
        });

    // Conservation Results structure
    py::class_<VariationalContactSolver::ConservationResults>(m, "ConservationResults",
        "Results of conservation law verification")
        .def_readonly("energy_drift", &VariationalContactSolver::ConservationResults::energy_drift,
            "Energy drift magnitude")
        .def_readonly("momentum_drift", &VariationalContactSolver::ConservationResults::momentum_drift,
            "Momentum drift vector")
        .def_readonly("angular_momentum_drift", &VariationalContactSolver::ConservationResults::angular_momentum_drift,
            "Angular momentum drift magnitude")
        .def_readonly("energy_conserved", &VariationalContactSolver::ConservationResults::energy_conserved,
            "Whether energy is conserved within tolerance")
        .def_readonly("momentum_conserved", &VariationalContactSolver::ConservationResults::momentum_conserved,
            "Whether momentum is conserved within tolerance")
        .def_readonly("angular_momentum_conserved", &VariationalContactSolver::ConservationResults::angular_momentum_conserved,
            "Whether angular momentum is conserved within tolerance");

    // Theoretical Bounds structure
    py::class_<VariationalContactSolver::TheoreticalBounds>(m, "TheoreticalBounds",
        "Theoretical analysis and bounds for contact system")
        .def_readonly("max_gradient_error", &VariationalContactSolver::TheoreticalBounds::max_gradient_error,
            "Upper bound on gradient computation error")
        .def_readonly("convergence_rate", &VariationalContactSolver::TheoreticalBounds::convergence_rate,
            "Theoretical Newton convergence rate")
        .def_readonly("condition_number", &VariationalContactSolver::TheoreticalBounds::condition_number,
            "Contact system condition number")
        .def_readonly("guaranteed_iterations", &VariationalContactSolver::TheoreticalBounds::guaranteed_iterations,
            "Maximum iterations required for convergence");

    // Variational Contact Solver (CPU version)
    py::class_<VariationalContactSolver>(m, "VariationalContactSolver",
        "Mathematically rigorous contact solver with theoretical guarantees")
        .def(py::init<const VariationalContactParams&>(), py::arg("params") = VariationalContactParams{},
            "Initialize with contact parameters")

        .def("detect_contacts_variational", [](VariationalContactSolver& self,
                py::array_t<double> positions, py::array_t<double> radii, py::array_t<int> material_ids) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);
            self.detectContactsVariational(pos_vec, radii_vec, mat_vec);
        }, "Detect contacts using C∞ smooth variational formulation",
           py::arg("positions"), py::arg("radii"), py::arg("material_ids"))

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

        .def("compute_contact_gradients", [](VariationalContactSolver& self,
                py::array_t<double> positions, py::array_t<double> velocities,
                py::array_t<double> masses, py::array_t<double> radii,
                py::array_t<int> material_ids, py::array_t<double> output_gradients) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto vel_vec = numpy_to_eigen_vector3d(velocities);
            auto mass_vec = py::cast<std::vector<double>>(masses);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);
            auto out_grad_vec = numpy_to_eigen_vector3d(output_gradients);

            std::vector<Eigen::Vector3d> pos_gradients, vel_gradients;
            self.computeContactGradients(pos_vec, vel_vec, mass_vec, radii_vec, mat_vec,
                                       out_grad_vec, pos_gradients, vel_gradients);

            return py::make_tuple(eigen_vector3d_to_numpy(pos_gradients),
                                eigen_vector3d_to_numpy(vel_gradients));
        }, "Compute provably correct gradients using adjoint method",
           py::arg("positions"), py::arg("velocities"), py::arg("masses"),
           py::arg("radii"), py::arg("material_ids"), py::arg("output_gradients"))

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
           py::arg("radii"), py::arg("material_ids"), py::arg("tolerance") = 1e-8)

        .def("verify_conservation_laws", [](VariationalContactSolver& self,
                py::array_t<double> positions_before, py::array_t<double> velocities_before,
                py::array_t<double> positions_after, py::array_t<double> velocities_after,
                py::array_t<double> masses, py::array_t<double> radii,
                py::array_t<int> material_ids, double dt) {
            auto pos_before = numpy_to_eigen_vector3d(positions_before);
            auto vel_before = numpy_to_eigen_vector3d(velocities_before);
            auto pos_after = numpy_to_eigen_vector3d(positions_after);
            auto vel_after = numpy_to_eigen_vector3d(velocities_after);
            auto mass_vec = py::cast<std::vector<double>>(masses);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);

            return self.verifyConservationLaws(pos_before, vel_before, pos_after, vel_after,
                                             mass_vec, radii_vec, mat_vec, dt);
        }, "Verify conservation laws (energy, momentum, angular momentum)",
           py::arg("positions_before"), py::arg("velocities_before"),
           py::arg("positions_after"), py::arg("velocities_after"),
           py::arg("masses"), py::arg("radii"), py::arg("material_ids"), py::arg("dt"))

        .def("analyze_theoretical_properties", [](VariationalContactSolver& self,
                py::array_t<double> positions, py::array_t<double> masses, py::array_t<double> radii) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto mass_vec = py::cast<std::vector<double>>(masses);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            return self.analyzeTheoreticalProperties(pos_vec, mass_vec, radii_vec);
        }, "Analyze theoretical properties and bounds",
           py::arg("positions"), py::arg("masses"), py::arg("radii"))

        .def("enable_rolling_resistance", &VariationalContactSolver::enableRollingResistance,
            "Enable rolling resistance with specified coefficient",
            py::arg("rolling_coeff") = 0.01)

        .def("enable_adhesion_forces", &VariationalContactSolver::enableAdhesionForces,
            "Enable adhesion forces with specified strength",
            py::arg("adhesion_strength") = 1e-3)

        .def("get_parameters", &VariationalContactSolver::getParameters,
            "Get current contact parameters", py::return_value_policy::reference)

        .def("set_parameters", &VariationalContactSolver::setParameters,
            "Set new contact parameters", py::arg("params"));

    // Variational Contact Integrator
    py::class_<VariationalContactIntegrator>(m, "VariationalContactIntegrator",
        "Hybrid implicit-explicit integrator with provable stability")
        .def(py::init<const VariationalContactParams&>(), py::arg("contact_params") = VariationalContactParams{},
            "Initialize with contact parameters")

        .def("integrate_step", [](VariationalContactIntegrator& self,
                py::array_t<double> positions, py::array_t<double> velocities,
                py::array_t<double> masses, py::array_t<double> radii,
                py::array_t<int> material_ids, double dt,
                py::array_t<double> external_forces) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto vel_vec = numpy_to_eigen_vector3d(velocities);
            auto mass_vec = py::cast<std::vector<double>>(masses);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);

            std::vector<Eigen::Vector3d> ext_forces;
            if (external_forces.size() > 0) {
                ext_forces = numpy_to_eigen_vector3d(external_forces);
            }

            double actual_dt = self.integrateStep(pos_vec, vel_vec, mass_vec, radii_vec,
                                                mat_vec, dt, ext_forces);

            return py::make_tuple(eigen_vector3d_to_numpy(pos_vec),
                                eigen_vector3d_to_numpy(vel_vec), actual_dt);
        }, "Integrate one time step with adaptive timestepping",
           py::arg("positions"), py::arg("velocities"), py::arg("masses"),
           py::arg("radii"), py::arg("material_ids"), py::arg("dt"),
           py::arg("external_forces") = py::array_t<double>())

        .def("compute_integration_gradients", [](VariationalContactIntegrator& self,
                py::array_t<double> positions_initial, py::array_t<double> velocities_initial,
                py::array_t<double> positions_final, py::array_t<double> velocities_final,
                py::array_t<double> masses, py::array_t<double> radii,
                py::array_t<int> material_ids, double dt,
                py::array_t<double> output_position_gradients,
                py::array_t<double> output_velocity_gradients) {
            auto pos_init = numpy_to_eigen_vector3d(positions_initial);
            auto vel_init = numpy_to_eigen_vector3d(velocities_initial);
            auto pos_final = numpy_to_eigen_vector3d(positions_final);
            auto vel_final = numpy_to_eigen_vector3d(velocities_final);
            auto mass_vec = py::cast<std::vector<double>>(masses);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);
            auto out_pos_grad = numpy_to_eigen_vector3d(output_position_gradients);
            auto out_vel_grad = numpy_to_eigen_vector3d(output_velocity_gradients);

            std::vector<Eigen::Vector3d> input_pos_grad, input_vel_grad;
            self.computeIntegrationGradients(pos_init, vel_init, pos_final, vel_final,
                                           mass_vec, radii_vec, mat_vec, dt,
                                           out_pos_grad, out_vel_grad,
                                           input_pos_grad, input_vel_grad);

            return py::make_tuple(eigen_vector3d_to_numpy(input_pos_grad),
                                eigen_vector3d_to_numpy(input_vel_grad));
        }, "Compute gradients through integration step",
           py::arg("positions_initial"), py::arg("velocities_initial"),
           py::arg("positions_final"), py::arg("velocities_final"),
           py::arg("masses"), py::arg("radii"), py::arg("material_ids"), py::arg("dt"),
           py::arg("output_position_gradients"), py::arg("output_velocity_gradients"))

        .def("get_contact_solver", &VariationalContactIntegrator::getContactSolver,
            "Get reference to contact solver", py::return_value_policy::reference);

#ifdef WITH_CUDA
    // GPU-accelerated versions
    py::class_<VariationalContactSolverGPU>(m, "VariationalContactSolverGPU",
        "GPU-accelerated variational contact solver with identical API")
        .def(py::init<const VariationalContactParams&>(), py::arg("params") = VariationalContactParams{},
            "Initialize GPU solver with contact parameters")

        .def("detect_contacts_variational", [](VariationalContactSolverGPU& self,
                py::array_t<double> positions, py::array_t<double> radii, py::array_t<int> material_ids) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);
            self.detectContactsVariational(pos_vec, radii_vec, mat_vec);
        }, "GPU-accelerated contact detection",
           py::arg("positions"), py::arg("radii"), py::arg("material_ids"))

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

        .def("compute_contact_energy", [](VariationalContactSolverGPU& self,
                py::array_t<double> positions, py::array_t<double> radii,
                py::array_t<int> material_ids) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);
            return self.computeContactEnergy(pos_vec, radii_vec, mat_vec);
        }, "GPU-accelerated contact energy computation",
           py::arg("positions"), py::arg("radii"), py::arg("material_ids"))

        .def("get_gpu_memory_usage_mb", &VariationalContactSolverGPU::getGPUMemoryUsageMB,
            "Get GPU memory usage in megabytes")

        .def("synchronize_gpu", &VariationalContactSolverGPU::synchronizeGPU,
            "Wait for all GPU operations to complete")

        .def("warmup_gpu", &VariationalContactSolverGPU::warmupGPU,
            "Pre-allocate and warm GPU kernels", py::arg("n_bodies"))

        .def_static("is_gpu_available", &VariationalContactSolverGPU::isGPUAvailable,
            "Check if CUDA-capable GPU is available")

        .def_static("get_available_gpu_devices", &VariationalContactSolverGPU::getAvailableGPUDevices,
            "Get list of available GPU device names")

        .def_static("get_available_gpu_memory", &VariationalContactSolverGPU::getAvailableGPUMemory,
            "Get available GPU memory in bytes", py::arg("device_id") = 0);

    py::class_<VariationalContactIntegratorGPU>(m, "VariationalContactIntegratorGPU",
        "GPU-accelerated hybrid integrator")
        .def(py::init<const VariationalContactParams&>(), py::arg("contact_params") = VariationalContactParams{},
            "Initialize GPU integrator")

        .def("integrate_step", [](VariationalContactIntegratorGPU& self,
                py::array_t<double> positions, py::array_t<double> velocities,
                py::array_t<double> masses, py::array_t<double> radii,
                py::array_t<int> material_ids, double dt,
                py::array_t<double> external_forces) {
            auto pos_vec = numpy_to_eigen_vector3d(positions);
            auto vel_vec = numpy_to_eigen_vector3d(velocities);
            auto mass_vec = py::cast<std::vector<double>>(masses);
            auto radii_vec = py::cast<std::vector<double>>(radii);
            auto mat_vec = py::cast<std::vector<int>>(material_ids);

            std::vector<Eigen::Vector3d> ext_forces;
            if (external_forces.size() > 0) {
                ext_forces = numpy_to_eigen_vector3d(external_forces);
            }

            double actual_dt = self.integrateStep(pos_vec, vel_vec, mass_vec, radii_vec,
                                                mat_vec, dt, ext_forces);

            return py::make_tuple(eigen_vector3d_to_numpy(pos_vec),
                                eigen_vector3d_to_numpy(vel_vec), actual_dt);
        }, "GPU-accelerated integration step",
           py::arg("positions"), py::arg("velocities"), py::arg("masses"),
           py::arg("radii"), py::arg("material_ids"), py::arg("dt"),
           py::arg("external_forces") = py::array_t<double>())

        .def("get_contact_solver", &VariationalContactIntegratorGPU::getContactSolver,
            "Get reference to GPU contact solver", py::return_value_policy::reference);
#endif

    // Utility functions from VariationalContactUtils namespace
    m.def("setup_chasing_contact_scenario", [](int num_bodies) {
        std::vector<Eigen::Vector3d> positions, velocities;
        std::vector<double> masses, radii;
        std::vector<int> material_ids;

        VariationalContactUtils::setupChasingContactScenario(
            positions, velocities, masses, radii, material_ids, num_bodies);

        return py::make_tuple(
            eigen_vector3d_to_numpy(positions),
            eigen_vector3d_to_numpy(velocities),
            py::cast(masses),
            py::cast(radii),
            py::cast(material_ids)
        );
    }, "Generate challenging test case for verification", py::arg("num_bodies") = 10);

    m.def("setup_constrained_system_scenario", [](int chain_length) {
        std::vector<Eigen::Vector3d> positions, velocities;
        std::vector<double> masses, radii;
        std::vector<int> material_ids;

        VariationalContactUtils::setupConstrainedSystemScenario(
            positions, velocities, masses, radii, material_ids, chain_length);

        return py::make_tuple(
            eigen_vector3d_to_numpy(positions),
            eigen_vector3d_to_numpy(velocities),
            py::cast(masses),
            py::cast(radii),
            py::cast(material_ids)
        );
    }, "Generate constrained system test scenario", py::arg("chain_length") = 5);

    m.def("comprehensive_gradient_test", [](
            py::array_t<double> positions, py::array_t<double> velocities,
            py::array_t<double> masses, py::array_t<double> radii,
            py::array_t<int> material_ids, double finite_diff_epsilon, double tolerance) {
        auto pos_vec = numpy_to_eigen_vector3d(positions);
        auto vel_vec = numpy_to_eigen_vector3d(velocities);
        auto mass_vec = py::cast<std::vector<double>>(masses);
        auto radii_vec = py::cast<std::vector<double>>(radii);
        auto mat_vec = py::cast<std::vector<int>>(material_ids);

        VariationalContactSolver solver;
        return VariationalContactUtils::comprehensiveGradientTest(
            solver, pos_vec, vel_vec, mass_vec, radii_vec, mat_vec,
            finite_diff_epsilon, tolerance);
    }, "Comprehensive gradient verification against finite differences",
       py::arg("positions"), py::arg("velocities"), py::arg("masses"),
       py::arg("radii"), py::arg("material_ids"),
       py::arg("finite_diff_epsilon") = 1e-7, py::arg("tolerance") = 1e-5);
}

PYBIND11_MODULE(variational_contact_bindings, m) {
    m.doc() = "Variational Contact Mechanics with GPU Acceleration";
    bind_variational_contact(m);
}