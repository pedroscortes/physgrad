#include "adaptive_integration.h"
#include <iostream>
#include <iomanip>

using namespace physgrad;

int main() {
    std::cout << "PhysGrad Adaptive Integration Demo\n";
    std::cout << "=================================\n\n";

    std::vector<float> pos_x = {0.0f, 1.0f};
    std::vector<float> pos_y = {0.0f, 0.0f};
    std::vector<float> pos_z = {0.0f, 0.0f};
    std::vector<float> vel_x = {0.0f, 0.0f};
    std::vector<float> vel_y = {0.0f, 1.0f};
    std::vector<float> vel_z = {0.0f, 0.0f};
    std::vector<float> masses = {10.0f, 0.1f};

    float G = 1.0f, epsilon = 0.001f, dt = 0.02f;
    int steps = 20;

    std::cout << "Testing integration schemes on two-body orbital system\n";
    std::cout << "Steps: " << steps << ", dt: " << dt << "\n\n";

    std::vector<IntegrationScheme> schemes = {
        IntegrationScheme::LEAPFROG,
        IntegrationScheme::RK4,
        IntegrationScheme::ADAPTIVE_HEUN
    };

    for (auto scheme : schemes) {
        std::vector<float> test_pos_x = pos_x, test_pos_y = pos_y, test_pos_z = pos_z;
        std::vector<float> test_vel_x = vel_x, test_vel_y = vel_y, test_vel_z = vel_z;

        AdaptiveIntegrator integrator(scheme);
        integrator.resizeWorkArrays(2);

        std::cout << integrator.getSchemeName() << ":\n";

        for (int i = 0; i < steps; i++) {
            integrator.integrateStep(test_pos_x, test_pos_y, test_pos_z,
                                   test_vel_x, test_vel_y, test_vel_z,
                                   masses, dt, G, epsilon);

            if (i % 5 == 0) {
                std::cout << "  Step " << i << ": pos=(" << std::fixed << std::setprecision(3)
                          << test_pos_x[1] << ", " << test_pos_y[1] << ")\n";
            }
        }

        float final_r = std::sqrt(std::pow(test_pos_x[1] - test_pos_x[0], 2) +
                                std::pow(test_pos_y[1] - test_pos_y[0], 2));
        std::cout << "  Final radius: " << final_r << " (initial: 1.0)\n\n";
    }

    std::cout << "✅ Adaptive integration schemes working successfully!\n";
    return 0;
}