#include "simulation.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

int main(int argc, char** argv) {
    physgrad::SimParams params;
    if (argc > 1) {
        params.num_bodies = std::atoi(argv[1]);
    }

    std::cout << "PhysGrad N-body Simulation\n";
    std::cout << "Bodies: " << params.num_bodies << "\n\n";

    auto simulation = std::make_unique<physgrad::Simulation>(params);

    const int num_steps = 100;
    const int report_interval = 10;

    std::cout << std::fixed << std::setprecision(2);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < num_steps; step++) {
        simulation->step();

        if (step % report_interval == 0) {
            float energy = simulation->getBodies()->computeEnergy(params);
            float step_time = simulation->getLastStepTime();
            float gflops = simulation->getGFLOPS();

            std::cout << "Step " << std::setw(3) << step
                     << " | Energy: " << std::setw(10) << energy
                     << " | Time: " << std::setw(6) << step_time << " ms"
                     << " | GFLOPS: " << std::setw(8) << gflops << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\nSimulation complete. Total time: " << duration << " seconds\n";
    std::cout << "Average step time: " << (duration * 1000.0) / num_steps << " ms\n";

    if (argc == 1) {
        std::cout << "\nPerformance scaling:\n";

        int test_sizes[] = {256, 512, 1024, 2048, 4096};

        for (int size : test_sizes) {
            params.num_bodies = size;
            auto test_sim = std::make_unique<physgrad::Simulation>(params);

            for (int i = 0; i < 5; i++) {
                test_sim->step();
            }

            auto test_start = std::chrono::high_resolution_clock::now();
            const int test_steps = 10;
            for (int i = 0; i < test_steps; i++) {
                test_sim->step();
            }
            auto test_end = std::chrono::high_resolution_clock::now();

            double test_duration = std::chrono::duration<double, std::milli>(
                test_end - test_start).count() / test_steps;

            float gflops = test_sim->getGFLOPS();

            std::cout << "N = " << std::setw(5) << size
                     << " | Time: " << std::setw(8) << test_duration << " ms"
                     << " | GFLOPS: " << std::setw(8) << gflops << "\n";
        }
    }

    return 0;
}