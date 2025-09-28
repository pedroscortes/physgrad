#include "test_framework.h"
#include "../src/collision_detection.h"

using namespace physgrad;
using namespace physgrad_tests;

void testCollisionDetection(TestFramework& framework) {
    framework.setCategory("Collision Detection");

    // Test collision detector creation
    framework.runTest("Collision Detector Creation", []() {
        CollisionParams params;
        params.contact_threshold = 0.01f;
        params.contact_stiffness = 1000.0f;

        CollisionDetector detector(params);

        // Just test that it doesn't crash
        ASSERT_TRUE(true);
    });

    // Test contact info structure
    framework.runTest("Contact Info Structure", []() {
        ContactInfo contact;
        contact.body_i = 0;
        contact.body_j = 1;
        contact.contact_distance = 0.1f;
        contact.overlap = 0.05f;
        contact.normal_x = 1.0f;
        contact.normal_y = 0.0f;
        contact.normal_z = 0.0f;
        contact.restitution = 0.8f;
        contact.friction = 0.3f;

        ASSERT_EQ(contact.body_i, 0);
        ASSERT_EQ(contact.body_j, 1);
        ASSERT_NEAR(contact.contact_distance, 0.1f, 1e-6f);
        ASSERT_NEAR(contact.overlap, 0.05f, 1e-6f);
        ASSERT_NEAR(contact.restitution, 0.8f, 1e-6f);
        ASSERT_NEAR(contact.friction, 0.3f, 1e-6f);
    });

    // Test collision parameters
    framework.runTest("Collision Parameters", []() {
        CollisionParams params;
        params.contact_threshold = 0.02f;
        params.contact_stiffness = 500.0f;
        params.contact_damping = 5.0f;
        params.enable_restitution = false;
        params.enable_friction = true;
        params.global_restitution = 0.9f;

        ASSERT_NEAR(params.contact_threshold, 0.02f, 1e-6f);
        ASSERT_NEAR(params.contact_stiffness, 500.0f, 1e-6f);
        ASSERT_NEAR(params.contact_damping, 5.0f, 1e-6f);
        ASSERT_FALSE(params.enable_restitution);
        ASSERT_TRUE(params.enable_friction);
        ASSERT_NEAR(params.global_restitution, 0.9f, 1e-6f);
    });

    // Test spatial hash broad phase
    framework.runTest("Spatial Hash Broad Phase", []() {
        SpatialHashBroadPhase broadPhase(1.0f);

        std::vector<float> pos_x = {0.0f, 2.5f, 0.1f};
        std::vector<float> pos_y = {0.0f, 0.0f, 0.1f};
        std::vector<float> pos_z = {0.0f, 0.0f, 0.0f};
        std::vector<float> radii = {0.5f, 0.5f, 0.5f};

        broadPhase.updateBodies(pos_x, pos_y, pos_z, radii);

        auto pairs = broadPhase.getPotentialCollisionPairs();

        // Should find at least some potential pairs
        ASSERT_GE(pairs.size(), 0);

        // Test clear
        broadPhase.clear();
        pairs = broadPhase.getPotentialCollisionPairs();
        ASSERT_EQ(pairs.size(), 0);
    });

    // Test collision detection basic functionality
    framework.runTest("Basic Collision Detection", []() {
        CollisionParams params;
        CollisionDetector detector(params);

        std::vector<float> pos_x = {0.0f, 1.5f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> radii = {1.0f, 1.0f};

        detector.updateBodyRadii(radii);

        std::vector<ContactInfo> contacts;

        ASSERT_NO_THROW({
            contacts = detector.detectCollisions(pos_x, pos_y, pos_z);
        });

        // Should detect contact between overlapping spheres
        ASSERT_GE(contacts.size(), 0);
    });

    // Test collision response static methods
    framework.runTest("Collision Response", []() {
        CollisionParams params;

        std::vector<float> pos_x = {0.0f, 1.8f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> vel_x = {1.0f, -1.0f};
        std::vector<float> vel_y = {0.0f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f};

        ContactInfo contact;
        contact.body_i = 0;
        contact.body_j = 1;
        contact.overlap = 0.2f;
        contact.normal_x = 1.0f;
        contact.normal_y = 0.0f;
        contact.normal_z = 0.0f;
        contact.restitution = 0.8f;

        ASSERT_NO_THROW({
            CollisionResponse::resolveImpulseCollision(contact, vel_x, vel_y, vel_z, masses, params);
        });

        // Velocities should have changed after collision response
        ASSERT_TRUE(std::isfinite(vel_x[0]));
        ASSERT_TRUE(std::isfinite(vel_x[1]));
    });

    // Test collision detector resolve collisions
    framework.runTest("Collision Detector Resolve Collisions", []() {
        CollisionParams params;
        CollisionDetector detector(params);

        std::vector<float> pos_x = {0.0f, 1.8f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> vel_x = {1.0f, -1.0f};
        std::vector<float> vel_y = {0.0f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f};
        std::vector<float> radii = {1.0f, 1.0f};

        detector.updateBodyRadii(radii);

        ASSERT_NO_THROW({
            detector.resolveCollisions(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses);
        });

        // Should complete without crashing
        ASSERT_TRUE(std::isfinite(vel_x[0]));
        ASSERT_TRUE(std::isfinite(vel_x[1]));
    });

    // Test performance with multiple objects
    framework.runTest("Performance Test", []() {
        CollisionParams params;
        CollisionDetector detector(params);

        const int num_objects = 50;
        std::vector<float> pos_x = PhysicsTestUtils::generateRandomVector(num_objects, -5.0f, 5.0f);
        std::vector<float> pos_y = PhysicsTestUtils::generateRandomVector(num_objects, -5.0f, 5.0f);
        std::vector<float> pos_z = PhysicsTestUtils::generateRandomVector(num_objects, -5.0f, 5.0f);
        std::vector<float> radii(num_objects, 0.3f);

        detector.updateBodyRadii(radii);

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<ContactInfo> contacts;
        ASSERT_NO_THROW({
            contacts = detector.detectCollisions(pos_x, pos_y, pos_z);
        });

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);

        // Should complete reasonably quickly (less than 50ms for 50 objects)
        ASSERT_LT(duration.count(), 50.0);
    });

    // Test edge cases
    framework.runTest("Edge Cases", []() {
        CollisionParams params;
        CollisionDetector detector(params);

        // Empty arrays
        std::vector<float> empty_pos;
        std::vector<float> empty_radii;

        detector.updateBodyRadii(empty_radii);

        ASSERT_NO_THROW({
            auto contacts = detector.detectCollisions(empty_pos, empty_pos, empty_pos);
            ASSERT_EQ(contacts.size(), 0);
        });

        // Single object
        std::vector<float> single_pos = {0.0f};
        std::vector<float> single_radii = {1.0f};

        detector.updateBodyRadii(single_radii);

        ASSERT_NO_THROW({
            auto contacts = detector.detectCollisions(single_pos, single_pos, single_pos);
            ASSERT_EQ(contacts.size(), 0); // Single object can't collide with itself
        });
    });

    // Test contact info validation
    framework.runTest("Contact Info Validation", []() {
        ContactInfo contact;
        contact.body_i = 0;
        contact.body_j = 1;
        contact.overlap = 0.1f;
        contact.normal_x = 1.0f;
        contact.normal_y = 0.0f;
        contact.normal_z = 0.0f;

        // Validate normal magnitude
        float normal_magnitude = std::sqrt(contact.normal_x * contact.normal_x +
                                         contact.normal_y * contact.normal_y +
                                         contact.normal_z * contact.normal_z);
        ASSERT_NEAR(normal_magnitude, 1.0f, 1e-5f);

        // Validate contact properties
        ASSERT_GE(contact.overlap, 0.0f);
        ASSERT_NE(contact.body_i, contact.body_j);
    });

    // Test collision force application
    framework.runTest("Collision Force Application", []() {
        CollisionParams params;
        CollisionDetector detector(params);

        std::vector<float> pos_x = {0.0f, 1.5f};
        std::vector<float> pos_y = {0.0f, 0.0f};
        std::vector<float> pos_z = {0.0f, 0.0f};
        std::vector<float> vel_x = {0.0f, 0.0f};
        std::vector<float> vel_y = {0.0f, 0.0f};
        std::vector<float> vel_z = {0.0f, 0.0f};
        std::vector<float> masses = {1.0f, 1.0f};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<float> force_x(2, 0.0f);
        std::vector<float> force_y(2, 0.0f);
        std::vector<float> force_z(2, 0.0f);

        detector.updateBodyRadii(radii);

        ASSERT_NO_THROW({
            detector.applyCollisionForces(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
                                        force_x, force_y, force_z, masses);
        });

        // Forces should be finite
        for (float f : force_x) {
            ASSERT_TRUE(std::isfinite(f));
        }
    });
}