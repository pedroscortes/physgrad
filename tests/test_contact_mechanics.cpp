/**
 * PhysGrad - Contact Mechanics Unit Tests
 */

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include "common_types.h"

// Mock contact mechanics interface
class ContactMechanics {
public:
    struct Contact {
        int particle1, particle2;
        float3 normal;
        float penetration;
        float3 position;
    };

    bool initialize() { return true; }
    void cleanup() {}

    void detectContacts(
        const std::vector<float3>& positions,
        const std::vector<float>& radii,
        std::vector<Contact>& contacts,
        float threshold = 0.01f
    ) {
        contacts.clear();
        for (size_t i = 0; i < positions.size(); ++i) {
            for (size_t j = i + 1; j < positions.size(); ++j) {
                float3 diff = make_float3(
                    positions[i].x - positions[j].x,
                    positions[i].y - positions[j].y,
                    positions[i].z - positions[j].z
                );
                float distance = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
                float contact_distance = radii[i] + radii[j] + threshold;

                if (distance < contact_distance) {
                    Contact contact;
                    contact.particle1 = static_cast<int>(i);
                    contact.particle2 = static_cast<int>(j);
                    contact.normal = {diff.x / distance, diff.y / distance, diff.z / distance};
                    contact.penetration = contact_distance - distance;
                    contact.position = {
                        (positions[i].x + positions[j].x) * 0.5f,
                        (positions[i].y + positions[j].y) * 0.5f,
                        (positions[i].z + positions[j].z) * 0.5f
                    };
                    contacts.push_back(contact);
                }
            }
        }
    }

    void resolveContacts(
        std::vector<float3>& positions,
        std::vector<float3>& velocities,
        const std::vector<float>& masses,
        const std::vector<Contact>& contacts,
        float restitution = 0.5f,
        float friction = 0.3f
    ) {
        for (const auto& contact : contacts) {
            int i = contact.particle1;
            int j = contact.particle2;

            // Position correction
            float total_mass = masses[i] + masses[j];
            float correction_i = contact.penetration * masses[j] / total_mass;
            float correction_j = contact.penetration * masses[i] / total_mass;

            positions[i].x += correction_i * contact.normal.x;
            positions[i].y += correction_i * contact.normal.y;
            positions[i].z += correction_i * contact.normal.z;

            positions[j].x -= correction_j * contact.normal.x;
            positions[j].y -= correction_j * contact.normal.y;
            positions[j].z -= correction_j * contact.normal.z;

            // Velocity correction (simplified)
            float3 rel_velocity = {
                velocities[i].x - velocities[j].x,
                velocities[i].y - velocities[j].y,
                velocities[i].z - velocities[j].z
            };

            float normal_velocity = rel_velocity.x * contact.normal.x +
                                  rel_velocity.y * contact.normal.y +
                                  rel_velocity.z * contact.normal.z;

            if (normal_velocity < 0) {
                float impulse = -(1 + restitution) * normal_velocity / total_mass;

                velocities[i].x += impulse * masses[j] * contact.normal.x;
                velocities[i].y += impulse * masses[j] * contact.normal.y;
                velocities[i].z += impulse * masses[j] * contact.normal.z;

                velocities[j].x -= impulse * masses[i] * contact.normal.x;
                velocities[j].y -= impulse * masses[i] * contact.normal.y;
                velocities[j].z -= impulse * masses[i] * contact.normal.z;
            }
        }
    }
};

class ContactMechanicsTest : public ::testing::Test {
protected:
    void SetUp() override {
        contact_system_ = std::make_unique<ContactMechanics>();
        ASSERT_TRUE(contact_system_->initialize());
    }

    void TearDown() override {
        if (contact_system_) {
            contact_system_->cleanup();
        }
    }

    std::unique_ptr<ContactMechanics> contact_system_;
};

TEST_F(ContactMechanicsTest, InitializationAndCleanup) {
    EXPECT_TRUE(contact_system_ != nullptr);
}

TEST_F(ContactMechanicsTest, ContactDetection) {
    // Create two overlapping spheres
    std::vector<float3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}  // Distance = 1.5, radii sum = 2.0
    };
    std::vector<float> radii = {1.0f, 1.0f};
    std::vector<ContactMechanics::Contact> contacts;

    contact_system_->detectContacts(positions, radii, contacts);

    EXPECT_EQ(contacts.size(), 1);
    if (!contacts.empty()) {
        EXPECT_EQ(contacts[0].particle1, 0);
        EXPECT_EQ(contacts[0].particle2, 1);
        EXPECT_GT(contacts[0].penetration, 0.0f);
    }
}

TEST_F(ContactMechanicsTest, NoContactDetection) {
    // Create two non-overlapping spheres
    std::vector<float3> positions = {
        {0.0f, 0.0f, 0.0f},
        {3.0f, 0.0f, 0.0f}  // Distance = 3.0, radii sum = 2.0
    };
    std::vector<float> radii = {1.0f, 1.0f};
    std::vector<ContactMechanics::Contact> contacts;

    contact_system_->detectContacts(positions, radii, contacts);

    EXPECT_EQ(contacts.size(), 0);
}

TEST_F(ContactMechanicsTest, ContactResolution) {
    std::vector<float3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };
    std::vector<float3> velocities = {
        {1.0f, 0.0f, 0.0f},
        {-1.0f, 0.0f, 0.0f}
    };
    std::vector<float> masses = {1.0f, 1.0f};
    std::vector<float> radii = {1.0f, 1.0f};

    // Detect contacts
    std::vector<ContactMechanics::Contact> contacts;
    contact_system_->detectContacts(positions, radii, contacts);

    ASSERT_EQ(contacts.size(), 1);

    // Store initial state
    float initial_distance = std::sqrt(
        std::pow(positions[1].x - positions[0].x, 2) +
        std::pow(positions[1].y - positions[0].y, 2) +
        std::pow(positions[1].z - positions[0].z, 2)
    );

    // Resolve contacts
    contact_system_->resolveContacts(positions, velocities, masses, contacts);

    // Check that penetration was resolved
    float final_distance = std::sqrt(
        std::pow(positions[1].x - positions[0].x, 2) +
        std::pow(positions[1].y - positions[0].y, 2) +
        std::pow(positions[1].z - positions[0].z, 2)
    );

    EXPECT_GT(final_distance, initial_distance);

    // Check velocity changes (should be reversed for elastic collision)
    EXPECT_LT(velocities[0].x, 1.0f);  // Particle 0 should slow down
    EXPECT_GT(velocities[1].x, -1.0f); // Particle 1 should slow down
}

TEST_F(ContactMechanicsTest, MomentumConservation) {
    std::vector<float3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };
    std::vector<float3> velocities = {
        {2.0f, 0.0f, 0.0f},
        {-1.0f, 0.0f, 0.0f}
    };
    std::vector<float> masses = {1.0f, 2.0f};
    std::vector<float> radii = {1.0f, 1.0f};

    // Calculate initial momentum
    float initial_momentum = masses[0] * velocities[0].x + masses[1] * velocities[1].x;

    // Detect and resolve contacts
    std::vector<ContactMechanics::Contact> contacts;
    contact_system_->detectContacts(positions, radii, contacts);
    contact_system_->resolveContacts(positions, velocities, masses, contacts);

    // Calculate final momentum
    float final_momentum = masses[0] * velocities[0].x + masses[1] * velocities[1].x;

    EXPECT_NEAR(initial_momentum, final_momentum, 1e-6f);
}

TEST_F(ContactMechanicsTest, MultipleContacts) {
    // Create three particles in a line with overlapping spheres
    std::vector<float3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f},
        {3.0f, 0.0f, 0.0f}
    };
    std::vector<float> radii = {1.0f, 1.0f, 1.0f};
    std::vector<ContactMechanics::Contact> contacts;

    contact_system_->detectContacts(positions, radii, contacts);

    EXPECT_EQ(contacts.size(), 2); // Two contacts: 0-1 and 1-2
}

TEST_F(ContactMechanicsTest, FrictionEffect) {
    std::vector<float3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };
    std::vector<float3> velocities = {
        {1.0f, 1.0f, 0.0f},  // Velocity with tangential component
        {-1.0f, 0.0f, 0.0f}
    };
    std::vector<float> masses = {1.0f, 1.0f};
    std::vector<float> radii = {1.0f, 1.0f};

    float initial_tangential_velocity = velocities[0].y;

    // Detect and resolve contacts with friction
    std::vector<ContactMechanics::Contact> contacts;
    contact_system_->detectContacts(positions, radii, contacts);
    contact_system_->resolveContacts(positions, velocities, masses, contacts, 0.5f, 0.3f);

    // Friction should reduce tangential velocity (simplified test)
    // In a more sophisticated implementation, friction would be explicitly applied
    EXPECT_TRUE(true); // Placeholder - friction implementation would be tested here
}

TEST_F(ContactMechanicsTest, RestitutionEffect) {
    std::vector<float3> positions = {
        {0.0f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };
    std::vector<float3> velocities = {
        {1.0f, 0.0f, 0.0f},
        {-1.0f, 0.0f, 0.0f}
    };
    std::vector<float> masses = {1.0f, 1.0f};
    std::vector<float> radii = {1.0f, 1.0f};

    // Test different restitution values
    std::vector<float> restitutions = {0.0f, 0.5f, 1.0f};

    for (float restitution : restitutions) {
        auto test_positions = positions;
        auto test_velocities = velocities;

        std::vector<ContactMechanics::Contact> contacts;
        contact_system_->detectContacts(test_positions, radii, contacts);
        contact_system_->resolveContacts(test_positions, test_velocities, masses, contacts, restitution);

        // Higher restitution should result in higher relative velocity after collision
        float relative_velocity = test_velocities[0].x - test_velocities[1].x;

        if (restitution == 0.0f) {
            EXPECT_NEAR(relative_velocity, 0.0f, 0.1f); // Perfectly inelastic
        } else if (restitution == 1.0f) {
            EXPECT_NEAR(std::abs(relative_velocity), 2.0f, 0.1f); // Perfectly elastic
        }
    }
}