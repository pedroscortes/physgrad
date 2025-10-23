#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "../../src/adjoint_integrators_standalone.h"
#include "../../src/concepts.h"

namespace py = pybind11;
using namespace physgrad::adjoint;

// Helper function to convert NumPy array (N, 3) to std::vector<ConceptVector3D>
template<typename T>
std::vector<ConceptVector3D<T>> numpy_to_vector3d(py::array_t<T> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Expected array shape (N, 3)");
    }

    T* ptr = static_cast<T*>(buf.ptr);
    std::vector<ConceptVector3D<T>> result;
    result.reserve(buf.shape[0]);

    for (size_t i = 0; i < buf.shape[0]; ++i) {
        ConceptVector3D<T> vec;
        vec[0] = ptr[i*3 + 0];
        vec[1] = ptr[i*3 + 1];
        vec[2] = ptr[i*3 + 2];
        result.push_back(vec);
    }
    return result;
}

// Helper function to convert std::vector<ConceptVector3D> to NumPy array (N, 3)
template<typename T>
py::array_t<T> vector3d_to_numpy(const std::vector<ConceptVector3D<T>>& input) {
    auto result = py::array_t<T>({input.size(), size_t(3)});
    py::buffer_info buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);

    for (size_t i = 0; i < input.size(); ++i) {
        ptr[i*3 + 0] = input[i][0];
        ptr[i*3 + 1] = input[i][1];
        ptr[i*3 + 2] = input[i][2];
    }
    return result;
}

// Helper function to convert NumPy array (N,) to std::vector<T>
template<typename T>
std::vector<T> numpy_to_vector(py::array_t<T> input) {
    py::buffer_info buf_info = input.request();
    T* ptr = static_cast<T*>(buf_info.ptr);
    return std::vector<T>(ptr, ptr + buf_info.size);
}

// Helper function to convert std::vector<T> to NumPy array
template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& input) {
    return py::array_t<T>(input.size(), input.data());
}

void bind_adjoint_verlet(py::module& m) {
    // Bind SimpleForceEngine (spring-mass system)
    py::class_<SimpleForceEngine<float>, std::shared_ptr<SimpleForceEngine<float>>>(
        m, "SimpleForceEngineFloat",
        "Spring-mass force engine for adjoint simulation (float32)")
        .def(py::init<>(), "Initialize empty force engine")
        .def("add_spring", &SimpleForceEngine<float>::addSpring,
             "Add a spring between two particles",
             py::arg("particle_i"), py::arg("particle_j"),
             py::arg("stiffness"), py::arg("rest_length"))
        .def("get_num_springs", &SimpleForceEngine<float>::getNumSprings,
             "Get number of springs in the system");

    // Bind AdjointSimulation (float32 version)
    py::class_<AdjointSimulation<float>>(m, "AdjointSimulationFloat",
        "Adjoint-based differentiable physics simulator (float32)")
        .def(py::init<std::shared_ptr<SimpleForceEngine<float>>>(),
             py::arg("force_engine"),
             "Initialize with a force engine")

        .def("run_forward", [](AdjointSimulation<float>& self,
                py::array_t<float> positions, py::array_t<float> velocities,
                py::array_t<float> masses, float dt, int num_steps) {
            auto pos_vec = numpy_to_vector3d<float>(positions);
            auto vel_vec = numpy_to_vector3d<float>(velocities);
            auto mass_vec = numpy_to_vector<float>(masses);

            self.runForward(pos_vec, vel_vec, mass_vec, dt, num_steps);

            return py::make_tuple(
                vector3d_to_numpy<float>(pos_vec),
                vector3d_to_numpy<float>(vel_vec)
            );
        }, "Run forward simulation with checkpointing",
           py::arg("positions"), py::arg("velocities"), py::arg("masses"),
           py::arg("dt"), py::arg("num_steps"))

        .def("run_backward", [](AdjointSimulation<float>& self,
                py::array_t<float> loss_grad_positions,
                py::array_t<float> loss_grad_velocities,
                py::array_t<float> masses) {
            auto loss_grad_pos = numpy_to_vector3d<float>(loss_grad_positions);
            auto loss_grad_vel = numpy_to_vector3d<float>(loss_grad_velocities);
            auto mass_vec = numpy_to_vector<float>(masses);

            std::vector<ConceptVector3D<float>> initial_pos_grads;
            std::vector<ConceptVector3D<float>> initial_vel_grads;
            std::vector<float> mass_grads(mass_vec.size());

            self.runBackward(loss_grad_pos, loss_grad_vel,
                           initial_pos_grads, initial_vel_grads, mass_grads);

            return py::make_tuple(
                vector3d_to_numpy<float>(initial_pos_grads),
                vector3d_to_numpy<float>(initial_vel_grads),
                vector_to_numpy<float>(mass_grads)
            );
        }, "Run backward pass to compute gradients",
           py::arg("loss_grad_positions"), py::arg("loss_grad_velocities"),
           py::arg("masses"))

        .def("compute_gradients", [](AdjointSimulation<float>& self,
                py::array_t<float> initial_positions,
                py::array_t<float> initial_velocities,
                py::array_t<float> masses,
                float dt, int num_steps,
                py::function loss_function) {
            auto init_pos = numpy_to_vector3d<float>(initial_positions);
            auto init_vel = numpy_to_vector3d<float>(initial_velocities);
            auto mass_vec = numpy_to_vector<float>(masses);

            // Wrap Python loss function for C++
            auto cpp_loss_function = [&loss_function](
                const std::vector<ConceptVector3D<float>>& positions,
                const std::vector<ConceptVector3D<float>>& velocities) -> float {
                auto pos_np = vector3d_to_numpy<float>(positions);
                auto vel_np = vector3d_to_numpy<float>(velocities);
                return loss_function(pos_np, vel_np).cast<float>();
            };

            auto [pos_grads, vel_grads] = self.computeGradients(
                init_pos, init_vel, mass_vec, dt, num_steps, cpp_loss_function);

            return py::make_tuple(
                vector3d_to_numpy<float>(pos_grads),
                vector3d_to_numpy<float>(vel_grads)
            );
        }, "Convenience function: run forward + backward passes",
           py::arg("initial_positions"), py::arg("initial_velocities"),
           py::arg("masses"), py::arg("dt"), py::arg("num_steps"),
           py::arg("loss_function"));

    // Double precision versions
    py::class_<SimpleForceEngine<double>, std::shared_ptr<SimpleForceEngine<double>>>(
        m, "SimpleForceEngineDouble",
        "Spring-mass force engine for adjoint simulation (float64)")
        .def(py::init<>(), "Initialize empty force engine")
        .def("add_spring", &SimpleForceEngine<double>::addSpring,
             "Add a spring between two particles",
             py::arg("particle_i"), py::arg("particle_j"),
             py::arg("stiffness"), py::arg("rest_length"))
        .def("get_num_springs", &SimpleForceEngine<double>::getNumSprings,
             "Get number of springs in the system");

    py::class_<AdjointSimulation<double>>(m, "AdjointSimulationDouble",
        "Adjoint-based differentiable physics simulator (float64)")
        .def(py::init<std::shared_ptr<SimpleForceEngine<double>>>(),
             py::arg("force_engine"),
             "Initialize with a force engine")

        .def("run_forward", [](AdjointSimulation<double>& self,
                py::array_t<double> positions, py::array_t<double> velocities,
                py::array_t<double> masses, double dt, int num_steps) {
            auto pos_vec = numpy_to_vector3d<double>(positions);
            auto vel_vec = numpy_to_vector3d<double>(velocities);
            auto mass_vec = numpy_to_vector<double>(masses);

            self.runForward(pos_vec, vel_vec, mass_vec, dt, num_steps);

            return py::make_tuple(
                vector3d_to_numpy<double>(pos_vec),
                vector3d_to_numpy<double>(vel_vec)
            );
        }, "Run forward simulation with checkpointing",
           py::arg("positions"), py::arg("velocities"), py::arg("masses"),
           py::arg("dt"), py::arg("num_steps"))

        .def("run_backward", [](AdjointSimulation<double>& self,
                py::array_t<double> loss_grad_positions,
                py::array_t<double> loss_grad_velocities,
                py::array_t<double> masses) {
            auto loss_grad_pos = numpy_to_vector3d<double>(loss_grad_positions);
            auto loss_grad_vel = numpy_to_vector3d<double>(loss_grad_velocities);
            auto mass_vec = numpy_to_vector<double>(masses);

            std::vector<ConceptVector3D<double>> initial_pos_grads;
            std::vector<ConceptVector3D<double>> initial_vel_grads;
            std::vector<double> mass_grads(mass_vec.size());

            self.runBackward(loss_grad_pos, loss_grad_vel,
                           initial_pos_grads, initial_vel_grads, mass_grads);

            return py::make_tuple(
                vector3d_to_numpy<double>(initial_pos_grads),
                vector3d_to_numpy<double>(initial_vel_grads),
                vector_to_numpy<double>(mass_grads)
            );
        }, "Run backward pass to compute gradients",
           py::arg("loss_grad_positions"), py::arg("loss_grad_velocities"),
           py::arg("masses"))

        .def("compute_gradients", [](AdjointSimulation<double>& self,
                py::array_t<double> initial_positions,
                py::array_t<double> initial_velocities,
                py::array_t<double> masses,
                double dt, int num_steps,
                py::function loss_function) {
            auto init_pos = numpy_to_vector3d<double>(initial_positions);
            auto init_vel = numpy_to_vector3d<double>(initial_velocities);
            auto mass_vec = numpy_to_vector<double>(masses);

            // Wrap Python loss function for C++
            auto cpp_loss_function = [&loss_function](
                const std::vector<ConceptVector3D<double>>& positions,
                const std::vector<ConceptVector3D<double>>& velocities) -> double {
                auto pos_np = vector3d_to_numpy<double>(positions);
                auto vel_np = vector3d_to_numpy<double>(velocities);
                return loss_function(pos_np, vel_np).cast<double>();
            };

            auto [pos_grads, vel_grads] = self.computeGradients(
                init_pos, init_vel, mass_vec, dt, num_steps, cpp_loss_function);

            return py::make_tuple(
                vector3d_to_numpy<double>(pos_grads),
                vector3d_to_numpy<double>(vel_grads)
            );
        }, "Convenience function: run forward + backward passes",
           py::arg("initial_positions"), py::arg("initial_velocities"),
           py::arg("masses"), py::arg("dt"), py::arg("num_steps"),
           py::arg("loss_function"));
}

PYBIND11_MODULE(adjoint_verlet_cpp, m) {
    m.doc() = "PhysGrad Adjoint Verlet Integrator - Differentiable Physics with Adjoint Method";
    bind_adjoint_verlet(m);
    m.attr("__version__") = "0.1.0";
}
