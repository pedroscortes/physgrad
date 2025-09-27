#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace physgrad;

class GradientValidator {
private:
    SimParams params;
    std::unique_ptr<Simulation> sim1, sim2;

public:
    GradientValidator(const SimParams& p) : params(p) {
        sim1 = std::make_unique<Simulation>(params);
        sim2 = std::make_unique<Simulation>(params);
    }

    // Compute finite difference gradient for initial positions
    std::vector<float> computeFiniteDifferenceGradient(
        const std::vector<float>& init_pos_x,
        const std::vector<float>& init_pos_y,
        const std::vector<float>& init_pos_z,
        const std::vector<float>& init_vel_x,
        const std::vector<float>& init_vel_y,
        const std::vector<float>& init_vel_z,
        const std::vector<float>& masses,
        const std::vector<float>& target_pos_x,
        const std::vector<float>& target_pos_y,
        const std::vector<float>& target_pos_z,
        int num_steps,
        float eps = 1e-5f) {

        int n = params.num_bodies;
        std::vector<float> fd_gradients(n * 3, 0.0f);

        // Compute central difference gradients
        for (int i = 0; i < n; i++) {
            for (int dim = 0; dim < 3; dim++) {
                // Forward perturbation
                std::vector<float> pos_x_plus = init_pos_x;
                std::vector<float> pos_y_plus = init_pos_y;
                std::vector<float> pos_z_plus = init_pos_z;

                if (dim == 0) pos_x_plus[i] += eps;
                else if (dim == 1) pos_y_plus[i] += eps;
                else pos_z_plus[i] += eps;

                float loss_plus = runSimulation(sim1.get(), pos_x_plus, pos_y_plus, pos_z_plus,
                                              init_vel_x, init_vel_y, init_vel_z, masses,
                                              target_pos_x, target_pos_y, target_pos_z, num_steps);

                // Backward perturbation
                std::vector<float> pos_x_minus = init_pos_x;
                std::vector<float> pos_y_minus = init_pos_y;
                std::vector<float> pos_z_minus = init_pos_z;

                if (dim == 0) pos_x_minus[i] -= eps;
                else if (dim == 1) pos_y_minus[i] -= eps;
                else pos_z_minus[i] -= eps;

                float loss_minus = runSimulation(sim2.get(), pos_x_minus, pos_y_minus, pos_z_minus,
                                                init_vel_x, init_vel_y, init_vel_z, masses,
                                                target_pos_x, target_pos_y, target_pos_z, num_steps);

                // Central difference
                fd_gradients[i * 3 + dim] = (loss_plus - loss_minus) / (2.0f * eps);
            }
        }

        return fd_gradients;
    }

    // Compute adjoint gradients using our implementation
    std::vector<float> computeAdjointGradient(
        const std::vector<float>& init_pos_x,
        const std::vector<float>& init_pos_y,
        const std::vector<float>& init_pos_z,
        const std::vector<float>& init_vel_x,
        const std::vector<float>& init_vel_y,
        const std::vector<float>& init_vel_z,
        const std::vector<float>& masses,
        const std::vector<float>& target_pos_x,
        const std::vector<float>& target_pos_y,
        const std::vector<float>& target_pos_z,
        int num_steps) {

        // Set up simulation with gradient recording
        BodySystem* bodies = sim1->getBodies();
        size_t size = bodies->n * sizeof(float);

        // Copy initial conditions
        cudaMemcpy(bodies->d_pos_x, init_pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, init_pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, init_pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_x, init_vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, init_vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, init_vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        // Run forward simulation with gradient recording
        sim1->clearTape();
        sim1->enableGradients();

        for (int step = 0; step < num_steps; step++) {
            sim1->step();
        }

        // Compute gradients
        sim1->computeGradients(target_pos_x, target_pos_y, target_pos_z);

        // Get adjoint gradients
        std::vector<float> grad_pos_x(bodies->n), grad_pos_y(bodies->n), grad_pos_z(bodies->n);
        bodies->getGradients(grad_pos_x, grad_pos_y, grad_pos_z);

        // Pack into single vector [x0, y0, z0, x1, y1, z1, ...]
        std::vector<float> gradients(bodies->n * 3);
        for (int i = 0; i < bodies->n; i++) {
            gradients[i * 3 + 0] = grad_pos_x[i];
            gradients[i * 3 + 1] = grad_pos_y[i];
            gradients[i * 3 + 2] = grad_pos_z[i];
        }

        return gradients;
    }

private:
    float runSimulation(Simulation* sim,
                       const std::vector<float>& pos_x,
                       const std::vector<float>& pos_y,
                       const std::vector<float>& pos_z,
                       const std::vector<float>& vel_x,
                       const std::vector<float>& vel_y,
                       const std::vector<float>& vel_z,
                       const std::vector<float>& masses,
                       const std::vector<float>& target_pos_x,
                       const std::vector<float>& target_pos_y,
                       const std::vector<float>& target_pos_z,
                       int num_steps) {

        BodySystem* bodies = sim->getBodies();
        size_t size = bodies->n * sizeof(float);

        // Set initial conditions
        cudaMemcpy(bodies->d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(bodies->d_mass, masses.data(), size, cudaMemcpyHostToDevice);

        // Run simulation
        sim->disableGradients(); // Don't record for finite differences
        for (int step = 0; step < num_steps; step++) {
            sim->step();
        }

        // Get final positions
        std::vector<float> final_pos_x(bodies->n), final_pos_y(bodies->n), final_pos_z(bodies->n);
        bodies->getPositions(final_pos_x, final_pos_y, final_pos_z);

        // Compute MSE loss
        float loss = 0.0f;
        for (int i = 0; i < bodies->n; i++) {
            float dx = final_pos_x[i] - target_pos_x[i];
            float dy = final_pos_y[i] - target_pos_y[i];
            float dz = final_pos_z[i] - target_pos_z[i];
            loss += dx*dx + dy*dy + dz*dz;
        }

        return loss / (2.0f * bodies->n);
    }
};

// Compute relative error between two vectors
float computeRelativeError(const std::vector<float>& a, const std::vector<float>& b) {
    float numerator = 0.0f;
    float denominator = 0.0f;

    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        numerator += diff * diff;
        denominator += a[i] * a[i] + b[i] * b[i];
    }

    return sqrt(numerator / (denominator + 1e-10f));
}

int main() {
    std::cout << "PhysGrad Gradient Validation Test\n";
    std::cout << "=================================\n\n";

    // Test parameters
    SimParams params;
    params.num_bodies = 3;
    params.time_step = 0.01f;
    params.G = 1.0f;
    params.epsilon = 0.001f;

    GradientValidator validator(params);

    // Set up a simple 3-body test case
    std::vector<float> init_pos_x = {-0.5f, 0.5f, 0.0f};
    std::vector<float> init_pos_y = {0.0f, 0.0f, 0.5f};
    std::vector<float> init_pos_z = {0.0f, 0.0f, 0.0f};
    std::vector<float> init_vel_x = {0.0f, 0.0f, 0.2f};
    std::vector<float> init_vel_y = {0.3f, -0.3f, 0.0f};
    std::vector<float> init_vel_z = {0.0f, 0.0f, 0.0f};
    std::vector<float> masses = {1.0f, 1.0f, 0.5f};

    // Target positions (somewhat arbitrary)
    std::vector<float> target_pos_x = {-0.48f, 0.52f, 0.02f};
    std::vector<float> target_pos_y = {0.03f, -0.03f, 0.48f};
    std::vector<float> target_pos_z = {0.0f, 0.0f, 0.0f};

    std::cout << "Test Configuration:\n";
    std::cout << "- Bodies: " << params.num_bodies << "\n";
    std::cout << "- Time steps: 5\n";
    std::cout << "- dt: " << params.time_step << "\n\n";

    // Test different numbers of simulation steps
    std::vector<int> test_steps = {2, 5, 10};

    for (int num_steps : test_steps) {
        std::cout << "Testing " << num_steps << " simulation steps:\n";
        std::cout << "-----------------------------------\n";

        // Compute finite difference gradients
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<float> fd_gradients = validator.computeFiniteDifferenceGradient(
            init_pos_x, init_pos_y, init_pos_z,
            init_vel_x, init_vel_y, init_vel_z, masses,
            target_pos_x, target_pos_y, target_pos_z, num_steps);
        auto fd_time = std::chrono::high_resolution_clock::now();

        // Compute adjoint gradients
        auto adjoint_start = std::chrono::high_resolution_clock::now();
        std::vector<float> adjoint_gradients = validator.computeAdjointGradient(
            init_pos_x, init_pos_y, init_pos_z,
            init_vel_x, init_vel_y, init_vel_z, masses,
            target_pos_x, target_pos_y, target_pos_z, num_steps);
        auto adjoint_end = std::chrono::high_resolution_clock::now();

        // Compute timing
        auto fd_duration = std::chrono::duration<double, std::milli>(fd_time - start_time).count();
        auto adjoint_duration = std::chrono::duration<double, std::milli>(adjoint_end - adjoint_start).count();

        // Compute relative error
        float relative_error = computeRelativeError(fd_gradients, adjoint_gradients);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Finite Difference Time: " << fd_duration << " ms\n";
        std::cout << "Adjoint Time: " << adjoint_duration << " ms\n";
        std::cout << "Speedup: " << fd_duration / adjoint_duration << "x\n";
        std::cout << "Relative Error: " << relative_error << "\n";

        // Show first few gradient components for comparison
        std::cout << "\nFirst 6 gradient components:\n";
        std::cout << "FD:      ";
        for (int i = 0; i < std::min(6, (int)fd_gradients.size()); i++) {
            std::cout << std::setw(10) << fd_gradients[i] << " ";
        }
        std::cout << "\nAdjoint: ";
        for (int i = 0; i < std::min(6, (int)adjoint_gradients.size()); i++) {
            std::cout << std::setw(10) << adjoint_gradients[i] << " ";
        }
        std::cout << "\n";

        // Validation result
        if (relative_error < 1e-4f) {
            std::cout << "✓ PASSED: Gradients match within tolerance\n";
        } else if (relative_error < 1e-2f) {
            std::cout << "⚠ WARNING: Gradients match but with higher error\n";
        } else {
            std::cout << "✗ FAILED: Gradients do not match\n";
        }

        std::cout << "\n";
    }

    return 0;
}