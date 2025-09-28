#include "test_framework.h"
#include "../src/constraints.h"

using namespace physgrad;
using namespace physgrad_tests;

void testConstraints(TestFramework& framework) {
    framework.setCategory("Constraint-Based Physics");

    // Test distance constraint
    framework.runTest("Distance Constraint", []() {
        ConstraintParams params;
        params.rest_length = 2.0f;
        params.stiffness = 1000.0f;
        DistanceConstraint constraint(0, 1, 2.0f, params);

        ASSERT_EQ(constraint.particle_indices[0], 0);
        ASSERT_EQ(constraint.particle_indices[1], 1);
        ASSERT_NEAR(constraint.params.rest_length, 2.0f, 1e-6f);
        ASSERT_NEAR(constraint.params.stiffness, 1000.0f, 1e-6f);

        // Test constraint violation calculation
        std::vector<float> pos_x = {0.0f, 3.0f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};

        float violation = constraint.evaluateConstraint(pos_x, pos_y, pos_z);
        ASSERT_GT(std::abs(violation), 0.0f); // Should have some violation
    });

    // Test position lock constraint
    framework.runTest("Position Lock Constraint", []() {
        ConstraintParams params;
        params.stiffness = 500.0f;
        PositionLockConstraint constraint(0, 1.0f, 2.0f, 3.0f, params);

        ASSERT_EQ(constraint.particle_indices[0], 0);
        ASSERT_NEAR(constraint.params.stiffness, 500.0f, 1e-6f);

        std::vector<float> pos_x = {2.0f};
        std::vector<float> pos_y = {3.0f};
        std::vector<float> pos_z = {4.0f};

        float violation = constraint.evaluateConstraint(pos_x, pos_y, pos_z);
        ASSERT_GT(std::abs(violation), 0.0f); // Should have some violation
    });

    // Test spring constraint
    framework.runTest("Spring Constraint", []() {
        ConstraintParams params;
        params.rest_length = 2.0f;
        params.stiffness = 100.0f;
        params.damping = 10.0f;
        SpringConstraint constraint(0, 1, 2.0f, 100.0f, params);

        ASSERT_EQ(constraint.particle_indices[0], 0);
        ASSERT_EQ(constraint.particle_indices[1], 1);
        ASSERT_NEAR(constraint.params.rest_length, 2.0f, 1e-6f);
        ASSERT_NEAR(constraint.params.stiffness, 100.0f, 1e-6f);
        ASSERT_NEAR(constraint.params.damping, 10.0f, 1e-6f);

        std::vector<float> pos_x = {0.0f, 1.5f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};

        float violation = constraint.evaluateConstraint(pos_x, pos_y, pos_z);
        ASSERT_GT(std::abs(violation), 0.0f); // Should have some violation for compressed spring
    });

    // Test constraint system
    framework.runTest("Constraint System", []() {
        ConstraintSolver system;

        ASSERT_EQ(system.getConstraints().size(), 0);

        // Add a distance constraint
        ConstraintParams params;
        params.rest_length = 1.0f;
        params.stiffness = 1000.0f;
        system.addDistanceConstraint(0, 1, 1.0f, params);

        ASSERT_EQ(system.getConstraints().size(), 1);

        // Add a position lock constraint
        system.addPositionLock(0, 0.0f, 0.0f, 0.0f, params);

        ASSERT_EQ(system.getConstraints().size(), 2);
    });

    // Test constraint solving (basic smoke test)
    framework.runTest("Constraint Solving", []() {
        ConstraintSolver solver;

        std::vector<float> pos_x = {0.0f, 3.0f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> vel_x = {0.0f, 0.0f};
        std::vector<float> vel_y = {0.0f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f};

        ConstraintParams params;
        params.rest_length = 2.0f;
        params.stiffness = 1000.0f;
        solver.addDistanceConstraint(0, 1, 2.0f, params);

        // Test that solver doesn't crash
        ASSERT_NO_THROW({
            solver.solveConstraints(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, 0.01f);
        });
    });

    // Test rope constraint
    framework.runTest("Rope Constraint", []() {
        ConstraintParams params;
        params.max_length = 2.0f;
        params.stiffness = 500.0f;
        RopeConstraint constraint(0, 1, 2.0f, params);

        ASSERT_EQ(constraint.particle_indices[0], 0);
        ASSERT_EQ(constraint.particle_indices[1], 1);
        ASSERT_NEAR(constraint.params.max_length, 2.0f, 1e-6f);

        // Test when rope is slack (distance < max_length)
        std::vector<float> pos_x = {0.0f, 1.0f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};

        float violation = constraint.evaluateConstraint(pos_x, pos_y, pos_z);
        // For slack rope, constraint should not be violated
        ASSERT_TRUE(std::isfinite(violation));

        // Test when rope is taut (distance > max_length)
        pos_x[1] = 3.0f;
        violation = constraint.evaluateConstraint(pos_x, pos_y, pos_z);
        ASSERT_TRUE(std::isfinite(violation));
    });

    // Test ball joint constraint
    framework.runTest("Ball Joint Constraint", []() {
        ConstraintParams params;
        params.stiffness = 1000.0f;
        BallJointConstraint constraint(0, 1, params);

        ASSERT_EQ(constraint.particle_indices[0], 0);
        ASSERT_EQ(constraint.particle_indices[1], 1);

        std::vector<float> pos_x = {0.0f, 1.0f};
        std::vector<float> pos_y = {0.0f, 1.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};

        float violation = constraint.evaluateConstraint(pos_x, pos_y, pos_z);
        ASSERT_TRUE(std::isfinite(violation));
    });

    // Test constraint parameters
    framework.runTest("Constraint Parameters", []() {
        ConstraintParams params;
        params.compliance = 0.1f;
        params.damping = 0.2f;
        params.breaking_force = 1000.0f;
        params.enabled = true;
        params.bilateral = false;

        DistanceConstraint constraint(0, 1, 1.0f, params);

        ASSERT_NEAR(constraint.params.compliance, 0.1f, 1e-6f);
        ASSERT_NEAR(constraint.params.damping, 0.2f, 1e-6f);
        ASSERT_NEAR(constraint.params.breaking_force, 1000.0f, 1e-6f);
        ASSERT_TRUE(constraint.params.enabled);
        ASSERT_FALSE(constraint.params.bilateral);
    });

    // Test constraint jacobian computation (smoke test)
    framework.runTest("Constraint Jacobian", []() {
        ConstraintParams params;
        DistanceConstraint constraint(0, 1, 1.0f, params);

        std::vector<float> pos_x = {0.0f, 1.5f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};

        std::vector<std::vector<float>> jacobian(2, std::vector<float>(3, 0.0f));

        ASSERT_NO_THROW({
            constraint.computeJacobian(pos_x, pos_y, pos_z, jacobian);
        });

        // Check that jacobian is finite
        for (const auto& row : jacobian) {
            for (float val : row) {
                ASSERT_TRUE(std::isfinite(val));
            }
        }
    });
}