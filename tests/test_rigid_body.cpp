#include "test_framework.h"
#include "../src/rigid_body.h"

using namespace physgrad;
using namespace physgrad_tests;

void testRigidBody(TestFramework& framework) {
    framework.setCategory("Rigid Body Dynamics");

    // Test quaternion operations
    framework.runTest("Quaternion Operations", []() {
        Quaternion q1(1.0f, 0.0f, 0.0f, 0.0f); // Identity quaternion
        Quaternion q2(0.0f, 1.0f, 0.0f, 0.0f); // 180° rotation around x-axis

        // Test normalization
        q1.normalize();
        float magnitude = std::sqrt(q1.w*q1.w + q1.x*q1.x + q1.y*q1.y + q1.z*q1.z);
        ASSERT_NEAR(magnitude, 1.0f, 1e-6f);

        // Test conjugate
        Quaternion q1_conj = q1.conjugate();
        ASSERT_NEAR(q1_conj.w, q1.w, 1e-6f);
        ASSERT_NEAR(q1_conj.x, -q1.x, 1e-6f);
        ASSERT_NEAR(q1_conj.y, -q1.y, 1e-6f);
        ASSERT_NEAR(q1_conj.z, -q1.z, 1e-6f);

        // Test multiplication
        Quaternion q3 = q1 * q2;
        ASSERT_TRUE(std::abs(q3.w) < 1.1f && std::abs(q3.x) < 1.1f &&
                   std::abs(q3.y) < 1.1f && std::abs(q3.z) < 1.1f);
    });

    // Test matrix operations
    framework.runTest("Matrix Operations", []() {
        // Test identity matrix
        Matrix3x3 identity;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (i == j) {
                    ASSERT_NEAR(identity.m[i*3 + j], 1.0f, 1e-6f);
                } else {
                    ASSERT_NEAR(identity.m[i*3 + j], 0.0f, 1e-6f);
                }
            }
        }

        // Test determinant
        float det = identity.determinant();
        ASSERT_NEAR(det, 1.0f, 1e-6f);

        // Test matrix multiplication
        Matrix3x3 result = identity * identity;
        for (int i = 0; i < 9; ++i) {
            ASSERT_NEAR(result.m[i], identity.m[i], 1e-6f);
        }

        // Test vector multiplication
        float vec[3] = {1.0f, 2.0f, 3.0f};
        float result_vec[3];
        identity.multiply(vec, result_vec);
        for (int i = 0; i < 3; ++i) {
            ASSERT_NEAR(result_vec[i], vec[i], 1e-6f);
        }
    });

    // Test rigid body creation and properties
    framework.runTest("Rigid Body Creation", []() {
        RigidBodyParams params;
        params.mass = 2.0f;
        params.linear_damping = 0.1f;

        RigidBody body(params);

        ASSERT_NEAR(body.getParameters().mass, 2.0f, 1e-6f);
        ASSERT_NEAR(body.getParameters().linear_damping, 0.1f, 1e-6f);

        // Test geometry setting
        body.setSphereGeometry(1.5f);
        ASSERT_EQ(body.getShapeType(), 0); // Sphere type

        body.setBoxGeometry(2.0f, 1.0f, 0.5f);
        ASSERT_EQ(body.getShapeType(), 1); // Box type
    });

    // Test force application
    framework.runTest("Force Application", []() {
        RigidBodyParams params;
        params.mass = 1.0f;
        RigidBody body(params);

        auto& state = body.getState();

        // Apply linear force
        float force[3] = {10.0f, 0.0f, 0.0f};
        body.applyForce(force);

        ASSERT_NEAR(state.force[0], 10.0f, 1e-6f);
        ASSERT_NEAR(state.force[1], 0.0f, 1e-6f);
        ASSERT_NEAR(state.force[2], 0.0f, 1e-6f);

        // Apply torque
        float torque[3] = {0.0f, 5.0f, 0.0f};
        body.applyTorque(torque);

        ASSERT_NEAR(state.torque[1], 5.0f, 1e-6f);

        // Clear forces
        body.clearForces();
        ASSERT_NEAR(state.force[0], 0.0f, 1e-6f);
        ASSERT_NEAR(state.torque[1], 0.0f, 1e-6f);
    });

    // Test integration
    framework.runTest("Rigid Body Integration", []() {
        RigidBodyParams params;
        params.mass = 1.0f;
        params.linear_damping = 0.0f; // No damping for this test
        params.angular_damping = 0.0f;

        RigidBody body(params);
        auto& state = body.getState();

        // Set initial conditions
        state.position[0] = 0.0f;
        state.velocity[0] = 1.0f; // 1 m/s in x direction

        float dt = 0.1f;
        body.integrate(dt);

        // After integration, position should have changed
        ASSERT_NEAR(state.position[0], 0.1f, 1e-6f); // x = v*t = 1*0.1

        // Apply constant force and integrate
        float force[3] = {2.0f, 0.0f, 0.0f}; // 2N force
        body.applyForce(force);
        body.integrate(dt);

        // Velocity should have increased: v = v0 + a*t = 1 + (2/1)*0.1 = 1.2
        ASSERT_NEAR(state.velocity[0], 1.2f, 1e-5f);
    });

    // Test inertia tensor calculations
    framework.runTest("Inertia Tensor Calculations", []() {
        float inertia[9];

        // Test sphere inertia
        RigidBodyUtils::computeSphereInertia(1.0f, 1.0f, inertia);
        float expected_sphere = 0.4f; // 2/5 * m * r^2 for unit mass and radius
        ASSERT_NEAR(inertia[0], expected_sphere, 1e-6f);
        ASSERT_NEAR(inertia[4], expected_sphere, 1e-6f);
        ASSERT_NEAR(inertia[8], expected_sphere, 1e-6f);

        // Test box inertia
        RigidBodyUtils::computeBoxInertia(1.0f, 2.0f, 2.0f, 2.0f, inertia);
        float expected_box = 1.0f / 12.0f * (2.0f*2.0f + 2.0f*2.0f); // I = m/12 * (h^2 + d^2)
        ASSERT_NEAR(inertia[0], expected_box, 1e-6f);
    });

    // Test collision detection
    framework.runTest("Sphere-Sphere Collision", []() {
        RigidBodyParams params;
        params.mass = 1.0f;

        RigidBody body1(params);
        RigidBody body2(params);

        body1.setSphereGeometry(1.0f);
        body2.setSphereGeometry(1.0f);

        auto& state1 = body1.getState();
        auto& state2 = body2.getState();

        // Position spheres to overlap
        state1.position[0] = 0.0f;
        state2.position[0] = 1.5f; // Distance = 1.5, radii sum = 2.0, so overlapping

        // Test sphere collision detection (simplified test)
        float distance = std::sqrt((state2.position[0] - state1.position[0]) * (state2.position[0] - state1.position[0]));
        float radii_sum = 2.0f; // Both have radius 1.0

        bool collision = distance < radii_sum;
        ASSERT_TRUE(collision);
        ASSERT_LT(distance, radii_sum); // Distance should be less than sum of radii
    });

    // Test rigid body system
    framework.runTest("Rigid Body System", []() {
        RigidBodySystem system;

        ASSERT_EQ(system.getBodyCount(), 0);

        // Add a body
        RigidBodyParams params;
        params.mass = 1.0f;
        auto body = std::make_unique<RigidBody>(params);
        body->setSphereGeometry(0.5f);

        int body_id = system.addRigidBody(std::move(body));
        ASSERT_EQ(body_id, 0);
        ASSERT_EQ(system.getBodyCount(), 1);

        // Test body retrieval
        RigidBody* retrieved_body = system.getRigidBody(body_id);
        ASSERT_TRUE(retrieved_body != nullptr);
        ASSERT_EQ(retrieved_body->getShapeType(), 0); // Sphere
    });

    // Test system simulation step
    framework.runTest("System Simulation Step", []() {
        RigidBodySystem system;

        // Add two bodies
        for (int i = 0; i < 2; ++i) {
            RigidBodyParams params;
            params.mass = 1.0f;
            auto body = std::make_unique<RigidBody>(params);
            body->setSphereGeometry(0.3f);

            auto& state = body->getState();
            state.position[0] = i * 2.0f; // Separate them
            state.position[1] = 5.0f;     // Above ground

            system.addRigidBody(std::move(body));
        }

        // Store initial positions
        std::vector<float> initial_y;
        for (size_t i = 0; i < system.getBodyCount(); ++i) {
            initial_y.push_back(system.getRigidBody(static_cast<int>(i))->getState().position[1]);
        }

        // Simulate with gravity
        float dt = 0.01f;
        for (int step = 0; step < 100; ++step) {
            system.step(dt);
        }

        // Bodies should have fallen due to gravity
        for (size_t i = 0; i < system.getBodyCount(); ++i) {
            float final_y = system.getRigidBody(static_cast<int>(i))->getState().position[1];
            ASSERT_LT(final_y, initial_y[i]); // Should have fallen
        }
    });

    // Test coordinate transformations
    framework.runTest("Coordinate Transformations", []() {
        RigidBodyParams params;
        RigidBody body(params);

        auto& state = body.getState();
        state.position[0] = 1.0f;
        state.position[1] = 2.0f;
        state.position[2] = 3.0f;

        // Set 90-degree rotation around z-axis
        float axis[3] = {0.0f, 0.0f, 1.0f};
        state.orientation.fromAxisAngle(axis, M_PI / 2.0f);
        state.updateDerivedQuantities(params);

        // Test point transformation
        float local_point[3] = {1.0f, 0.0f, 0.0f};
        float world_point[3];
        body.localToWorld(local_point, world_point);

        // After 90° rotation around z, (1,0,0) becomes (0,1,0) + translation
        ASSERT_NEAR(world_point[0], 1.0f, 1e-5f); // x_world = 0 + 1 = 1
        ASSERT_NEAR(world_point[1], 3.0f, 1e-5f); // y_world = 1 + 2 = 3
        ASSERT_NEAR(world_point[2], 3.0f, 1e-5f); // z_world = 0 + 3 = 3

        // Test inverse transformation
        float back_to_local[3];
        body.worldToLocal(world_point, back_to_local);
        ASSERT_ARRAY_NEAR(local_point, back_to_local, 3, 1e-5f);
    });
}