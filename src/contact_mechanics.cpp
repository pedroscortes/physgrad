/**
 * PhysGrad - Contact Mechanics Implementation
 *
 * Contact detection and resolution system.
 */

#include "common_types.h"
#include <iostream>
#include <vector>
#include <cmath>

namespace physgrad {

class ContactMechanics {
public:
    struct Contact {
        int particle1, particle2;
        float3 normal;
        float penetration;
        float3 position;
    };

    bool initialize() {
        std::cout << "Contact mechanics system initialized." << std::endl;
        return true;
    }

    void cleanup() {
        std::cout << "Contact mechanics system cleaned up." << std::endl;
    }

    void detectContacts(
        const std::vector<float3>& positions,
        const std::vector<float>& radii,
        std::vector<Contact>& contacts,
        float threshold = 0.01f
    ) {
        contacts.clear();
        for (size_t i = 0; i < positions.size(); ++i) {
            for (size_t j = i + 1; j < positions.size(); ++j) {
                float3 diff = {
                    positions[i].x - positions[j].x,
                    positions[i].y - positions[j].y,
                    positions[i].z - positions[j].z
                };
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
        float restitution = 0.5f
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

            // Velocity correction
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

} // namespace physgrad