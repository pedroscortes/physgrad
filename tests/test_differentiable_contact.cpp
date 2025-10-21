/**
 * PhysGrad Differentiable Contact Mechanics Validation Tests
 *
 * Tests for differentiable contact solver with gradient computation.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>

#define PHYSGRAD_CONCEPTS_AVAILABLE
#include "common_types.h"
#include "differentiable_contact.h"

using namespace physgrad;
using namespace physgrad::contact;

class DifferentiableContactTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create solver with default parameters
        typename DifferentiableContactSolver<float>::SolverParams params;
        params.max_iterations = 50;
        params.tolerance = 1e-6f;
        params.contact_stiffness = 1e5f;
        params.use_friction = true;

        solver_ = std::make_unique<DifferentiableContactSolver<float>>(params);
    }

    // Helper to create a simple sphere-sphere contact
    ContactPoint<float> createSphereContact(
        const ConceptVector3D<float>& pos_a,
        const ConceptVector3D<float>& pos_b,
        float radius_a, float radius_b) {

        auto delta = pos_b - pos_a;
        float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);

        ContactPoint<float> contact;
        contact.body_a_id = 0;
        contact.body_b_id = 1;
        contact.penetration_depth = (radius_a + radius_b) - dist;

        if (dist > 1e-10f) {
            contact.normal = ConceptVector3D<float>(
                delta.x / dist,
                delta.y / dist,
                delta.z / dist
            );
        } else {
            contact.normal = ConceptVector3D<float>(1.0f, 0.0f, 0.0f);
        }

        contact.position = pos_a + contact.normal * radius_a;
        contact.friction_coefficient = 0.5f;

        return contact;
    }

    std::unique_ptr<DifferentiableContactSolver<float>> solver_;
};

TEST_F(DifferentiableContactTest, SolverConstruction) {
    // Test that solver can be constructed
    EXPECT_NE(solver_, nullptr);
}

TEST_F(DifferentiableContactTest, EmptyContactList) {
    // Test with no contacts
    std::vector<ContactPoint<float>> contacts;
    std::vector<ConceptVector3D<float>> velocities(2);
    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;

    auto solution = solver_->solveContacts(contacts, velocities, masses, dt);

    EXPECT_TRUE(solution.converged);
    EXPECT_EQ(solution.normal_impulses.size(), 0);
}

TEST_F(DifferentiableContactTest, SingleContactSphere) {
    // Two spheres in contact
    ConceptVector3D<float> pos_a(0.0f, 0.0f, 0.0f);
    ConceptVector3D<float> pos_b(1.5f, 0.0f, 0.0f);  // Overlapping by 0.5
    float radius_a = 1.0f;
    float radius_b = 1.0f;

    std::vector<ContactPoint<float>> contacts;
    contacts.push_back(createSphereContact(pos_a, pos_b, radius_a, radius_b));

    EXPECT_GT(contacts[0].penetration_depth, 0.0f) << "Spheres should be penetrating";

    std::vector<ConceptVector3D<float>> velocities = {
        ConceptVector3D<float>(1.0f, 0.0f, 0.0f),   // Moving right
        ConceptVector3D<float>(-1.0f, 0.0f, 0.0f)   // Moving left
    };
    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;

    auto solution = solver_->solveContacts(contacts, velocities, masses, dt);

    EXPECT_EQ(solution.normal_impulses.size(), 1);
    EXPECT_GT(solution.normal_impulses[0], 0.0f) << "Should have positive normal impulse";
}

TEST_F(DifferentiableContactTest, ContactConvergence) {
    // Test that solver converges
    ConceptVector3D<float> pos_a(0.0f, 0.0f, 0.0f);
    ConceptVector3D<float> pos_b(1.8f, 0.0f, 0.0f);  // Small overlap

    std::vector<ContactPoint<float>> contacts;
    contacts.push_back(createSphereContact(pos_a, pos_b, 1.0f, 1.0f));

    std::vector<ConceptVector3D<float>> velocities(2);
    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;

    auto solution = solver_->solveContacts(contacts, velocities, masses, dt);

    EXPECT_TRUE(solution.converged) << "Solver should converge for simple contact";
    EXPECT_LT(solution.num_iterations, 50) << "Should converge in reasonable iterations";
}

TEST_F(DifferentiableContactTest, SphereContactDetection) {
    // Test sphere contact detector
    std::vector<float> radii = {1.0f, 0.5f, 0.75f};
    SphereContactDetector<float> detector(radii);

    std::vector<ConceptVector3D<float>> positions = {
        ConceptVector3D<float>(0.0f, 0.0f, 0.0f),
        ConceptVector3D<float>(1.3f, 0.0f, 0.0f),  // Overlapping with 0
        ConceptVector3D<float>(5.0f, 0.0f, 0.0f)   // Not overlapping
    };

    auto contacts = detector.detectContacts(positions);

    EXPECT_EQ(contacts.size(), 1) << "Should detect exactly one contact";
    EXPECT_EQ(contacts[0].body_a_id, 0);
    EXPECT_EQ(contacts[0].body_b_id, 1);
    EXPECT_GT(contacts[0].penetration_depth, 0.0f);
}

TEST_F(DifferentiableContactTest, MultipleContacts) {
    // Three spheres, multiple contacts
    std::vector<float> radii = {1.0f, 1.0f, 1.0f};
    SphereContactDetector<float> detector(radii);

    std::vector<ConceptVector3D<float>> positions = {
        ConceptVector3D<float>(0.0f, 0.0f, 0.0f),
        ConceptVector3D<float>(1.5f, 0.0f, 0.0f),  // Touching 0
        ConceptVector3D<float>(0.75f, 1.3f, 0.0f)  // Touching 0
    };

    auto contacts = detector.detectContacts(positions);

    EXPECT_GE(contacts.size(), 2) << "Should detect at least 2 contacts";
}

TEST_F(DifferentiableContactTest, FrictionForces) {
    // Test with friction enabled
    typename DifferentiableContactSolver<float>::SolverParams params;
    params.use_friction = true;
    DifferentiableContactSolver<float> friction_solver(params);

    ConceptVector3D<float> pos_a(0.0f, 0.0f, 0.0f);
    ConceptVector3D<float> pos_b(1.5f, 0.0f, 0.0f);

    std::vector<ContactPoint<float>> contacts;
    contacts.push_back(createSphereContact(pos_a, pos_b, 1.0f, 1.0f));
    contacts[0].friction_coefficient = 0.5f;

    std::vector<ConceptVector3D<float>> velocities = {
        ConceptVector3D<float>(0.0f, 1.0f, 0.0f),   // Sliding tangentially
        ConceptVector3D<float>(0.0f, -1.0f, 0.0f)
    };
    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;

    auto solution = friction_solver.solveContacts(contacts, velocities, masses, dt);

    EXPECT_GT(solution.normal_impulses[0], 0.0f);
    // Check that friction impulses exist (at least one should be non-zero)
    bool has_friction = std::abs(solution.friction_impulses_u[0]) > 1e-6f ||
                       std::abs(solution.friction_impulses_v[0]) > 1e-6f;
    EXPECT_TRUE(has_friction) << "Should have friction impulses for tangential motion";
}

TEST_F(DifferentiableContactTest, NoFrictionMode) {
    // Test with friction disabled
    typename DifferentiableContactSolver<float>::SolverParams params;
    params.use_friction = false;
    DifferentiableContactSolver<float> no_friction_solver(params);

    ConceptVector3D<float> pos_a(0.0f, 0.0f, 0.0f);
    ConceptVector3D<float> pos_b(1.5f, 0.0f, 0.0f);

    std::vector<ContactPoint<float>> contacts;
    contacts.push_back(createSphereContact(pos_a, pos_b, 1.0f, 1.0f));

    std::vector<ConceptVector3D<float>> velocities = {
        ConceptVector3D<float>(0.0f, 1.0f, 0.0f),
        ConceptVector3D<float>(0.0f, -1.0f, 0.0f)
    };
    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;

    auto solution = no_friction_solver.solveContacts(contacts, velocities, masses, dt);

    // Friction impulses should be zero when friction is disabled
    EXPECT_FLOAT_EQ(solution.friction_impulses_u[0], 0.0f);
    EXPECT_FLOAT_EQ(solution.friction_impulses_v[0], 0.0f);
}

TEST_F(DifferentiableContactTest, WarmStarting) {
    // Test warm starting improves convergence
    ConceptVector3D<float> pos_a(0.0f, 0.0f, 0.0f);
    ConceptVector3D<float> pos_b(1.5f, 0.0f, 0.0f);

    std::vector<ContactPoint<float>> contacts;
    contacts.push_back(createSphereContact(pos_a, pos_b, 1.0f, 1.0f));

    std::vector<ConceptVector3D<float>> velocities(2);
    std::vector<float> masses = {1.0f, 1.0f};
    float dt = 0.01f;

    // First solve (cold start)
    auto solution1 = solver_->solveContacts(contacts, velocities, masses, dt);
    int iterations1 = solution1.num_iterations;

    // Second solve (warm start)
    auto solution2 = solver_->solveContacts(contacts, velocities, masses, dt);
    int iterations2 = solution2.num_iterations;

    // Warm start should converge in same or fewer iterations
    EXPECT_LE(iterations2, iterations1) << "Warm start should not increase iterations";
}

TEST_F(DifferentiableContactTest, ConceptCompliance) {
    // Verify that the solver uses concepts correctly
    static_assert(concepts::PhysicsScalar<float>, "float should satisfy PhysicsScalar");
    static_assert(concepts::PhysicsScalar<double>, "double should satisfy PhysicsScalar");

    // Verify ConceptVector3D satisfies Vector3D concept
    static_assert(concepts::Vector3D<ConceptVector3D<float>>,
                  "ConceptVector3D<float> should satisfy Vector3D");

    SUCCEED() << "Concept compliance verified at compile time";
}

TEST_F(DifferentiableContactTest, NumericStability) {
    // Test with very small and very large masses
    ConceptVector3D<float> pos_a(0.0f, 0.0f, 0.0f);
    ConceptVector3D<float> pos_b(1.5f, 0.0f, 0.0f);

    std::vector<ContactPoint<float>> contacts;
    contacts.push_back(createSphereContact(pos_a, pos_b, 1.0f, 1.0f));

    std::vector<ConceptVector3D<float>> velocities(2);
    std::vector<float> masses = {0.01f, 100.0f};  // Mass ratio 1:10000
    float dt = 0.01f;

    auto solution = solver_->solveContacts(contacts, velocities, masses, dt);

    EXPECT_TRUE(solution.converged) << "Should remain stable with extreme mass ratios";
    EXPECT_GT(solution.normal_impulses[0], 0.0f);
    EXPECT_TRUE(std::isfinite(solution.normal_impulses[0])) << "Impulse should be finite";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
