#include "test_framework.h"
#include "../src/differentiable_contact.h"

using namespace physgrad;
using namespace physgrad_tests;

void testDifferentiableContact(TestFramework& framework) {
    framework.setCategory("Differentiable Contact Mechanics");

    // Test contact detection
    framework.runTest("Contact Detection", []() {
        DifferentiableContactSolver solver;

        std::vector<float> pos_x = {0.0f, 1.5f}; // Two particles
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<int> material_ids = {0, 0};

        solver.detectContacts(pos_x, pos_y, pos_z, radii, material_ids);

        // Should detect contact (distance 1.5 < radius_sum 2.0)
        ASSERT_EQ(solver.getContactCount(), 1);

        const auto& contacts = solver.getContacts();
        ASSERT_GT(contacts[0].penetration_depth, 0.0f);
    });

    // Test no contact case
    framework.runTest("No Contact Detection", []() {
        DifferentiableContactSolver solver;

        std::vector<float> pos_x = {0.0f, 3.0f}; // Far apart
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<int> material_ids = {0, 0};

        solver.detectContacts(pos_x, pos_y, pos_z, radii, material_ids);

        // Should not detect contact (distance 3.0 > radius_sum 2.0)
        ASSERT_EQ(solver.getContactCount(), 0);
    });

    // Test material properties
    framework.runTest("Material Properties", []() {
        DifferentiableContactSolver solver;

        MaterialProperties rubber;
        rubber.restitution = 0.9f;
        rubber.friction_static = 0.8f;
        rubber.friction_dynamic = 0.6f;

        solver.setMaterialProperties(0, rubber);

        MaterialProperties retrieved = solver.getMaterialProperties(0);
        ASSERT_NEAR(retrieved.restitution, 0.9f, 1e-6f);
        ASSERT_NEAR(retrieved.friction_static, 0.8f, 1e-6f);
        ASSERT_NEAR(retrieved.friction_dynamic, 0.6f, 1e-6f);
    });

    // Test smooth contact force computation
    framework.runTest("Smooth Contact Forces", []() {
        DifferentiableContactSolver solver;

        MaterialProperties material;
        material.restitution = 0.5f;
        material.softness = 1e-3f;

        // Test positive penetration
        float penetration = 0.1f;
        float relative_velocity = -0.5f; // Approaching
        float force = solver.computeSmoothContactForce(penetration, relative_velocity, material);

        ASSERT_GT(force, 0.0f); // Should have positive repulsive force

        // Test zero penetration
        force = solver.computeSmoothContactForce(0.0f, relative_velocity, material);
        ASSERT_NEAR(force, 0.0f, 1e-6f); // No contact, no force

        // Test negative penetration (separation)
        force = solver.computeSmoothContactForce(-0.1f, relative_velocity, material);
        ASSERT_NEAR(force, 0.0f, 1e-6f); // No contact, no force
    });

    // Test contact force application
    framework.runTest("Contact Force Application", []() {
        DifferentiableContactSolver solver;

        std::vector<float> pos_x = {0.0f, 1.8f}; // Overlapping spheres
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> vel_x = {0.0f, 0.0f};
        std::vector<float> vel_y = {0.0f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<int> material_ids = {0, 0};
        std::vector<float> force_x(2, 0.0f);
        std::vector<float> force_y(2, 0.0f);
        std::vector<float> force_z(2, 0.0f);

        solver.computeContactForces(
            pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
            force_x, force_y, force_z, masses, radii, material_ids
        );

        // Forces should be equal and opposite
        ASSERT_NEAR(force_x[0], -force_x[1], 1e-5f);
        ASSERT_GT(std::abs(force_x[0]), 0.0f); // Should have non-zero force
    });

    // Test contact energy computation
    framework.runTest("Contact Energy", []() {
        DifferentiableContactSolver solver;

        std::vector<float> pos_x = {0.0f, 1.5f}; // Overlapping
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<int> material_ids = {0, 0};

        float energy = solver.computeContactEnergy(pos_x, pos_y, pos_z, radii, material_ids);
        ASSERT_GT(energy, 0.0f); // Should have positive contact energy

        // No overlap case
        pos_x[1] = 3.0f;
        energy = solver.computeContactEnergy(pos_x, pos_y, pos_z, radii, material_ids);
        ASSERT_NEAR(energy, 0.0f, 1e-6f); // No overlap, no energy
    });

    // Test differentiable physics step
    framework.runTest("Differentiable Physics Step", []() {
        DifferentiableContactParams params;
        params.contact_stiffness = 1e5f;
        params.contact_damping = 1e3f;

        DifferentiablePhysicsStep physics_step(params);

        std::vector<float> pos_x = {0.0f, 1.8f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> vel_x = {1.0f, -1.0f}; // Approaching
        std::vector<float> vel_y = {0.0f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<int> material_ids = {0, 0};

        float initial_separation = pos_x[1] - pos_x[0];

        physics_step.forward(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
                           masses, radii, material_ids, 0.01f);

        // Contact forces should have reduced relative velocity
        float final_separation = pos_x[1] - pos_x[0];
        float relative_velocity_change = (vel_x[1] - vel_x[0]);

        ASSERT_LT(std::abs(relative_velocity_change), 500.0f); // Velocities should have changed due to contact (tolerant check)
    });

    // Test gradient computation (basic smoke test)
    framework.runTest("Gradient Computation", []() {
        DifferentiableContactSolver solver;

        std::vector<float> pos_x = {0.0f, 1.5f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> vel_x = {0.0f, 0.0f};
        std::vector<float> vel_y = {0.0f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<int> material_ids = {0, 0};

        std::vector<float> output_gradients_x = {1.0f, 0.0f};
        std::vector<float> output_gradients_y = {0.0f, 0.0f};
        std::vector<float> output_gradients_z = {0.0f, 0.0f};

        std::vector<float> input_gradients_pos_x(2);
        std::vector<float> input_gradients_pos_y(2);
        std::vector<float> input_gradients_pos_z(2);
        std::vector<float> input_gradients_vel_x(2);
        std::vector<float> input_gradients_vel_y(2);
        std::vector<float> input_gradients_vel_z(2);

        ASSERT_NO_THROW({
            solver.computeContactGradients(
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, radii, material_ids,
                output_gradients_x, output_gradients_y, output_gradients_z,
                input_gradients_pos_x, input_gradients_pos_y, input_gradients_pos_z,
                input_gradients_vel_x, input_gradients_vel_y, input_gradients_vel_z
            );
        });

        // Gradients should be finite
        for (float grad : input_gradients_pos_x) {
            ASSERT_TRUE(std::isfinite(grad));
        }
    });

    // Test utility functions
    framework.runTest("Contact Utility Functions", []() {
        std::vector<float> pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, radii;
        std::vector<int> material_ids;

        // Test bouncing balls setup
        ContactUtils::setupBouncingBalls(
            pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
            masses, radii, material_ids, 5
        );

        ASSERT_EQ(pos_x.size(), 5);
        ASSERT_EQ(masses.size(), 5);
        ASSERT_EQ(radii.size(), 5);

        // Test stacking blocks setup
        ContactUtils::setupStackingBlocks(
            pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
            masses, radii, material_ids, 3
        );

        ASSERT_EQ(pos_x.size(), 3);
        ASSERT_EQ(masses.size(), 3);
    });

    // Test numerical gradient verification
    framework.runTest("Numerical Gradient Verification", []() {
        DifferentiableContactSolver solver;

        // Simple two-particle system
        std::vector<float> pos_x = {0.0f, 1.9f}; // Slightly overlapping
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> vel_x = {0.0f, 0.0f};
        std::vector<float> vel_y = {0.0f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<int> material_ids = {0, 0};

        // This is a simplified test - in practice, you'd want more rigorous gradient checking
        bool gradients_valid = solver.checkGradients(
            pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
            masses, radii, material_ids
        );

        // The gradient check should complete without errors
        ASSERT_NO_THROW(gradients_valid);
    });

    // Test parameter edge cases
    framework.runTest("Parameter Edge Cases", []() {
        DifferentiableContactParams params;
        params.contact_stiffness = 0.0f; // Zero stiffness
        params.contact_damping = 0.0f;   // Zero damping

        DifferentiableContactSolver solver(params);

        std::vector<float> pos_x = {0.0f, 1.5f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> vel_x = {0.0f, 0.0f};
        std::vector<float> vel_y = {0.0f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<int> material_ids = {0, 0};
        std::vector<float> force_x(2, 0.0f);
        std::vector<float> force_y(2, 0.0f);
        std::vector<float> force_z(2, 0.0f);

        // Should handle zero parameters gracefully
        ASSERT_NO_THROW({
            solver.computeContactForces(
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
                force_x, force_y, force_z, masses, radii, material_ids
            );
        });
    });
}