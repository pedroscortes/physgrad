#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

int main() {
    std::cout << "=================================\n";
    std::cout << "   Physics Validation Test\n";
    std::cout << "=================================\n\n";

    physgrad::SimParams params;
    params.num_bodies = 512;
    params.time_step = 0.001f;

    auto simulation = std::make_unique<physgrad::Simulation>(params);

    float initial_energy = simulation->getBodies()->computeEnergy(params);
    std::cout << "Initial energy: " << std::fixed << std::setprecision(6)
              << initial_energy << "\n\n";

    // Test energy conservation over 1000 timesteps
    std::vector<float> energies;
    const int test_steps = 1000;

    for (int step = 0; step <= test_steps; step++) {
        if (step > 0) simulation->step();

        float energy = simulation->getBodies()->computeEnergy(params);
        energies.push_back(energy);

        if (step % 100 == 0) {
            float drift = abs(energy - initial_energy) / initial_energy * 100.0f;
            std::cout << "Step " << std::setw(4) << step
                     << " | Energy: " << std::setw(10) << energy
                     << " | Drift: " << std::setw(8) << drift << "%\n";
        }
    }

    // Calculate statistics
    float min_energy = *std::min_element(energies.begin(), energies.end());
    float max_energy = *std::max_element(energies.begin(), energies.end());
    float energy_range = max_energy - min_energy;
    float max_drift = abs(max_energy - initial_energy) / initial_energy * 100.0f;

    std::cout << "\n=================================\n";
    std::cout << "Energy Conservation Analysis:\n";
    std::cout << "=================================\n";
    std::cout << "Initial energy: " << initial_energy << "\n";
    std::cout << "Final energy:   " << energies.back() << "\n";
    std::cout << "Min energy:     " << min_energy << "\n";
    std::cout << "Max energy:     " << max_energy << "\n";
    std::cout << "Energy range:   " << energy_range << "\n";
    std::cout << "Max drift:      " << max_drift << "%\n";

    // Verdict
    if (max_drift < 1.0f) {
        std::cout << "\n✅ EXCELLENT: Energy conservation < 1%\n";
    } else if (max_drift < 5.0f) {
        std::cout << "\n✅ GOOD: Energy conservation < 5%\n";
    } else {
        std::cout << "\n⚠️  WARNING: Energy drift > 5% (may need smaller timestep)\n";
    }

    // Test momentum conservation (should be zero for isolated system)
    std::vector<float> pos_x(params.num_bodies), pos_y(params.num_bodies), pos_z(params.num_bodies);
    std::vector<float> vel_x(params.num_bodies), vel_y(params.num_bodies), vel_z(params.num_bodies);
    std::vector<float> mass(params.num_bodies);

    simulation->getBodies()->getPositions(pos_x, pos_y, pos_z);

    // Get velocities (we need to add this method)
    size_t size = params.num_bodies * sizeof(float);
    cudaMemcpy(vel_x.data(), simulation->getBodies()->d_vel_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(vel_y.data(), simulation->getBodies()->d_vel_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(vel_z.data(), simulation->getBodies()->d_vel_z, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(mass.data(), simulation->getBodies()->d_mass, size, cudaMemcpyDeviceToHost);

    float momentum_x = 0.0f, momentum_y = 0.0f, momentum_z = 0.0f;
    for (int i = 0; i < params.num_bodies; i++) {
        momentum_x += mass[i] * vel_x[i];
        momentum_y += mass[i] * vel_y[i];
        momentum_z += mass[i] * vel_z[i];
    }

    float total_momentum = sqrt(momentum_x*momentum_x + momentum_y*momentum_y + momentum_z*momentum_z);

    std::cout << "\nMomentum Conservation:\n";
    std::cout << "Total momentum: " << total_momentum << "\n";

    if (total_momentum < 1e-6) {
        std::cout << "✅ EXCELLENT: Momentum conserved to machine precision\n";
    } else if (total_momentum < 1e-3) {
        std::cout << "✅ GOOD: Momentum approximately conserved\n";
    } else {
        std::cout << "⚠️  WARNING: Significant momentum drift detected\n";
    }

    return 0;
}