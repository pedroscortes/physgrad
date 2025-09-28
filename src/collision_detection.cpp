#include "collision_detection.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace physgrad {

// SpatialHashBroadPhase Implementation
SpatialHashBroadPhase::SpatialHashBroadPhase(float cell_size)
    : cell_size(cell_size), inv_cell_size(1.0f / cell_size) {
}

uint64_t SpatialHashBroadPhase::hashPosition(float x, float y, float z) const {
    int cx, cy, cz;
    getCellCoords(x, y, z, cx, cy, cz);

    // Simple hash function combining the three coordinates
    uint64_t hash = 0;
    hash ^= std::hash<int>{}(cx) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(cy) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(cz) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
}

void SpatialHashBroadPhase::getCellCoords(float x, float y, float z, int& cx, int& cy, int& cz) const {
    cx = static_cast<int>(std::floor(x * inv_cell_size));
    cy = static_cast<int>(std::floor(y * inv_cell_size));
    cz = static_cast<int>(std::floor(z * inv_cell_size));
}

void SpatialHashBroadPhase::updateBodies(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    const std::vector<float>& radii) {

    clear();

    for (size_t i = 0; i < pos_x.size(); ++i) {
        float radius = (i < radii.size()) ? radii[i] : 0.1f;

        // Insert body into all cells it might overlap
        int min_cx, max_cx, min_cy, max_cy, min_cz, max_cz;
        getCellCoords(pos_x[i] - radius, pos_y[i] - radius, pos_z[i] - radius, min_cx, min_cy, min_cz);
        getCellCoords(pos_x[i] + radius, pos_y[i] + radius, pos_z[i] + radius, max_cx, max_cy, max_cz);

        for (int cx = min_cx; cx <= max_cx; ++cx) {
            for (int cy = min_cy; cy <= max_cy; ++cy) {
                for (int cz = min_cz; cz <= max_cz; ++cz) {
                    uint64_t hash = hashPosition(cx * cell_size, cy * cell_size, cz * cell_size);
                    spatial_hash[hash].bodies.push_back(static_cast<int>(i));
                }
            }
        }
    }
}

std::vector<std::pair<int, int>> SpatialHashBroadPhase::getPotentialCollisionPairs() {
    std::vector<std::pair<int, int>> pairs;
    std::unordered_set<uint64_t> processed_pairs;

    for (const auto& [hash, cell] : spatial_hash) {
        const auto& bodies = cell.bodies;

        for (size_t i = 0; i < bodies.size(); ++i) {
            for (size_t j = i + 1; j < bodies.size(); ++j) {
                int body_i = bodies[i];
                int body_j = bodies[j];

                if (body_i > body_j) std::swap(body_i, body_j);

                // Create unique pair identifier
                uint64_t pair_id = (static_cast<uint64_t>(body_i) << 32) | static_cast<uint64_t>(body_j);

                if (processed_pairs.find(pair_id) == processed_pairs.end()) {
                    processed_pairs.insert(pair_id);
                    pairs.emplace_back(body_i, body_j);
                }
            }
        }
    }

    return pairs;
}

void SpatialHashBroadPhase::clear() {
    spatial_hash.clear();
}

// NarrowPhase Implementation
bool NarrowPhase::checkSphereCollision(
    float x1, float y1, float z1, float r1,
    float x2, float y2, float z2, float r2,
    ContactInfo& contact) {

    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    float distance_sq = dx*dx + dy*dy + dz*dz;
    float distance = std::sqrt(distance_sq);

    float contact_distance = r1 + r2;

    if (distance < contact_distance) {
        contact.contact_distance = contact_distance;
        contact.overlap = contact_distance - distance;

        // Contact normal (from body 1 to body 2)
        if (distance > 1e-6f) {
            float inv_distance = 1.0f / distance;
            contact.normal_x = dx * inv_distance;
            contact.normal_y = dy * inv_distance;
            contact.normal_z = dz * inv_distance;
        } else {
            // Handle degenerate case
            contact.normal_x = 1.0f;
            contact.normal_y = 0.0f;
            contact.normal_z = 0.0f;
        }

        // Contact point (on surface of first sphere)
        contact.contact_x = x1 + contact.normal_x * r1;
        contact.contact_y = y1 + contact.normal_y * r1;
        contact.contact_z = z1 + contact.normal_z * r1;

        return true;
    }

    return false;
}

bool NarrowPhase::checkSphereBoxCollision(
    float sx, float sy, float sz, float sr,
    float bx, float by, float bz,
    float bw, float bh, float bd,
    ContactInfo& contact) {

    // Find closest point on box to sphere center
    float closest_x = std::max(bx - bw*0.5f, std::min(sx, bx + bw*0.5f));
    float closest_y = std::max(by - bh*0.5f, std::min(sy, by + bh*0.5f));
    float closest_z = std::max(bz - bd*0.5f, std::min(sz, bz + bd*0.5f));

    float dx = sx - closest_x;
    float dy = sy - closest_y;
    float dz = sz - closest_z;
    float distance_sq = dx*dx + dy*dy + dz*dz;

    if (distance_sq < sr*sr) {
        float distance = std::sqrt(distance_sq);
        contact.overlap = sr - distance;

        if (distance > 1e-6f) {
            float inv_distance = 1.0f / distance;
            contact.normal_x = dx * inv_distance;
            contact.normal_y = dy * inv_distance;
            contact.normal_z = dz * inv_distance;
        } else {
            contact.normal_x = 1.0f;
            contact.normal_y = 0.0f;
            contact.normal_z = 0.0f;
        }

        contact.contact_x = closest_x;
        contact.contact_y = closest_y;
        contact.contact_z = closest_z;

        return true;
    }

    return false;
}

bool NarrowPhase::checkBoxCollision(
    float x1, float y1, float z1, float w1, float h1, float d1,
    float x2, float y2, float z2, float w2, float h2, float d2,
    ContactInfo& contact) {

    // Simple AABB collision detection
    float left1 = x1 - w1*0.5f, right1 = x1 + w1*0.5f;
    float bottom1 = y1 - h1*0.5f, top1 = y1 + h1*0.5f;
    float back1 = z1 - d1*0.5f, front1 = z1 + d1*0.5f;

    float left2 = x2 - w2*0.5f, right2 = x2 + w2*0.5f;
    float bottom2 = y2 - h2*0.5f, top2 = y2 + h2*0.5f;
    float back2 = z2 - d2*0.5f, front2 = z2 + d2*0.5f;

    if (right1 > left2 && left1 < right2 &&
        top1 > bottom2 && bottom1 < top2 &&
        front1 > back2 && back1 < front2) {

        // Calculate overlap in each axis
        float overlap_x = std::min(right1 - left2, right2 - left1);
        float overlap_y = std::min(top1 - bottom2, top2 - bottom1);
        float overlap_z = std::min(front1 - back2, front2 - back1);

        // Use smallest overlap as separation axis
        if (overlap_x <= overlap_y && overlap_x <= overlap_z) {
            contact.overlap = overlap_x;
            contact.normal_x = (x1 < x2) ? -1.0f : 1.0f;
            contact.normal_y = 0.0f;
            contact.normal_z = 0.0f;
        } else if (overlap_y <= overlap_z) {
            contact.overlap = overlap_y;
            contact.normal_x = 0.0f;
            contact.normal_y = (y1 < y2) ? -1.0f : 1.0f;
            contact.normal_z = 0.0f;
        } else {
            contact.overlap = overlap_z;
            contact.normal_x = 0.0f;
            contact.normal_y = 0.0f;
            contact.normal_z = (z1 < z2) ? -1.0f : 1.0f;
        }

        contact.contact_x = (x1 + x2) * 0.5f;
        contact.contact_y = (y1 + y2) * 0.5f;
        contact.contact_z = (z1 + z2) * 0.5f;

        return true;
    }

    return false;
}

// CollisionResponse Implementation
void CollisionResponse::applyContactForces(
    const std::vector<ContactInfo>& contacts,
    std::vector<float>& force_x,
    std::vector<float>& force_y,
    std::vector<float>& force_z,
    const std::vector<float>& vel_x,
    const std::vector<float>& vel_y,
    const std::vector<float>& vel_z,
    const std::vector<float>& masses,
    const CollisionParams& params) {

    for (const auto& contact : contacts) {
        int i = contact.body_i;
        int j = contact.body_j;

        if (i >= static_cast<int>(force_x.size()) || j >= static_cast<int>(force_x.size())) continue;

        // Spring force based on penetration
        float spring_force = params.contact_stiffness * contact.overlap;

        // Damping force based on relative velocity
        float rel_vel_normal = (vel_x[j] - vel_x[i]) * contact.normal_x +
                               (vel_y[j] - vel_y[i]) * contact.normal_y +
                               (vel_z[j] - vel_z[i]) * contact.normal_z;

        float damping_force = params.contact_damping * rel_vel_normal;

        float total_force = spring_force + damping_force;

        // Apply forces
        force_x[i] += total_force * contact.normal_x;
        force_y[i] += total_force * contact.normal_y;
        force_z[i] += total_force * contact.normal_z;

        force_x[j] -= total_force * contact.normal_x;
        force_y[j] -= total_force * contact.normal_y;
        force_z[j] -= total_force * contact.normal_z;
    }
}

void CollisionResponse::resolveImpulseCollision(
    ContactInfo& contact,
    std::vector<float>& vel_x,
    std::vector<float>& vel_y,
    std::vector<float>& vel_z,
    const std::vector<float>& masses,
    const CollisionParams& params) {

    int i = contact.body_i;
    int j = contact.body_j;

    if (i >= static_cast<int>(vel_x.size()) || j >= static_cast<int>(vel_x.size())) return;

    float mass_i = masses[i];
    float mass_j = masses[j];
    float inv_mass_i = 1.0f / mass_i;
    float inv_mass_j = 1.0f / mass_j;
    float inv_mass_sum = inv_mass_i + inv_mass_j;

    // Relative velocity
    float rel_vel_x = vel_x[j] - vel_x[i];
    float rel_vel_y = vel_y[j] - vel_y[i];
    float rel_vel_z = vel_z[j] - vel_z[i];

    // Relative velocity along contact normal
    float rel_vel_normal = rel_vel_x * contact.normal_x +
                          rel_vel_y * contact.normal_y +
                          rel_vel_z * contact.normal_z;

    // Don't resolve if objects are separating
    if (rel_vel_normal > 0) return;

    // Calculate impulse magnitude
    float restitution = params.enable_restitution ? contact.restitution : 0.0f;
    float impulse_magnitude = -(1.0f + restitution) * rel_vel_normal / inv_mass_sum;

    // Apply impulse
    float impulse_x = impulse_magnitude * contact.normal_x;
    float impulse_y = impulse_magnitude * contact.normal_y;
    float impulse_z = impulse_magnitude * contact.normal_z;

    vel_x[i] -= impulse_x * inv_mass_i;
    vel_y[i] -= impulse_y * inv_mass_i;
    vel_z[i] -= impulse_z * inv_mass_i;

    vel_x[j] += impulse_x * inv_mass_j;
    vel_y[j] += impulse_y * inv_mass_j;
    vel_z[j] += impulse_z * inv_mass_j;

    // Friction
    if (params.enable_friction) {
        // Recalculate relative velocity after normal impulse
        rel_vel_x = vel_x[j] - vel_x[i];
        rel_vel_y = vel_y[j] - vel_y[i];
        rel_vel_z = vel_z[j] - vel_z[i];

        float new_rel_vel_normal = rel_vel_x * contact.normal_x +
                                   rel_vel_y * contact.normal_y +
                                   rel_vel_z * contact.normal_z;

        // Tangential velocity
        float tangent_vel_x = rel_vel_x - new_rel_vel_normal * contact.normal_x;
        float tangent_vel_y = rel_vel_y - new_rel_vel_normal * contact.normal_y;
        float tangent_vel_z = rel_vel_z - new_rel_vel_normal * contact.normal_z;

        float tangent_speed = std::sqrt(tangent_vel_x*tangent_vel_x +
                                       tangent_vel_y*tangent_vel_y +
                                       tangent_vel_z*tangent_vel_z);

        if (tangent_speed > 1e-6f) {
            float friction_impulse = contact.friction * impulse_magnitude;
            friction_impulse = std::min(friction_impulse, tangent_speed * mass_i * mass_j / (mass_i + mass_j));

            float tangent_dir_x = tangent_vel_x / tangent_speed;
            float tangent_dir_y = tangent_vel_y / tangent_speed;
            float tangent_dir_z = tangent_vel_z / tangent_speed;

            float friction_x = friction_impulse * tangent_dir_x;
            float friction_y = friction_impulse * tangent_dir_y;
            float friction_z = friction_impulse * tangent_dir_z;

            vel_x[i] += friction_x * inv_mass_i;
            vel_y[i] += friction_y * inv_mass_i;
            vel_z[i] += friction_z * inv_mass_i;

            vel_x[j] -= friction_x * inv_mass_j;
            vel_y[j] -= friction_y * inv_mass_j;
            vel_z[j] -= friction_z * inv_mass_j;
        }
    }
}

void CollisionResponse::applySeparationImpulse(
    ContactInfo& contact,
    std::vector<float>& pos_x,
    std::vector<float>& pos_y,
    std::vector<float>& pos_z,
    const std::vector<float>& masses) {

    int i = contact.body_i;
    int j = contact.body_j;

    if (i >= static_cast<int>(pos_x.size()) || j >= static_cast<int>(pos_x.size())) return;

    float mass_i = masses[i];
    float mass_j = masses[j];
    float total_mass = mass_i + mass_j;

    float separation_i = contact.overlap * (mass_j / total_mass);
    float separation_j = contact.overlap * (mass_i / total_mass);

    pos_x[i] -= separation_i * contact.normal_x;
    pos_y[i] -= separation_i * contact.normal_y;
    pos_z[i] -= separation_i * contact.normal_z;

    pos_x[j] += separation_j * contact.normal_x;
    pos_y[j] += separation_j * contact.normal_y;
    pos_z[j] += separation_j * contact.normal_z;
}

// CollisionDetector Implementation
CollisionDetector::CollisionDetector(const CollisionParams& collision_params)
    : params(collision_params) {
    broad_phase = std::make_unique<SpatialHashBroadPhase>(1.0f);
}

void CollisionDetector::setBroadPhase(std::unique_ptr<BroadPhase> broad_phase_impl) {
    broad_phase = std::move(broad_phase_impl);
}

void CollisionDetector::setParameters(const CollisionParams& collision_params) {
    params = collision_params;
}

void CollisionDetector::updateBodyRadii(const std::vector<float>& radii) {
    body_radii = radii;
}

void CollisionDetector::updateBodyRadiiFromMasses(const std::vector<float>& masses, float density) {
    body_radii.resize(masses.size());
    for (size_t i = 0; i < masses.size(); ++i) {
        body_radii[i] = CollisionUtils::radiusFromMass(masses[i], density);
    }
}

std::vector<ContactInfo> CollisionDetector::detectCollisions(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z) {

    current_contacts.clear();

    if (!broad_phase) return current_contacts;

    // Update broad phase
    broad_phase->updateBodies(pos_x, pos_y, pos_z, body_radii);

    // Get potential collision pairs
    auto potential_pairs = broad_phase->getPotentialCollisionPairs();
    broad_phase_pairs = static_cast<int>(potential_pairs.size());

    // Narrow phase collision detection
    narrow_phase_tests = 0;
    for (const auto& [i, j] : potential_pairs) {
        ++narrow_phase_tests;

        if (i >= static_cast<int>(pos_x.size()) || j >= static_cast<int>(pos_x.size())) continue;

        ContactInfo contact;
        contact.body_i = i;
        contact.body_j = j;
        contact.restitution = params.global_restitution;
        contact.friction = params.global_friction;

        float radius_i = (i < static_cast<int>(body_radii.size())) ? body_radii[i] : 0.1f;
        float radius_j = (j < static_cast<int>(body_radii.size())) ? body_radii[j] : 0.1f;

        if (NarrowPhase::checkSphereCollision(
            pos_x[i], pos_y[i], pos_z[i], radius_i,
            pos_x[j], pos_y[j], pos_z[j], radius_j,
            contact)) {

            current_contacts.push_back(contact);
        }
    }

    actual_contacts = static_cast<int>(current_contacts.size());
    return current_contacts;
}

void CollisionDetector::applyCollisionForces(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    const std::vector<float>& vel_x,
    const std::vector<float>& vel_y,
    const std::vector<float>& vel_z,
    std::vector<float>& force_x,
    std::vector<float>& force_y,
    std::vector<float>& force_z,
    const std::vector<float>& masses) {

    auto contacts = detectCollisions(pos_x, pos_y, pos_z);
    CollisionResponse::applyContactForces(contacts, force_x, force_y, force_z,
                                         vel_x, vel_y, vel_z, masses, params);
}

void CollisionDetector::resolveCollisions(
    std::vector<float>& pos_x,
    std::vector<float>& pos_y,
    std::vector<float>& pos_z,
    std::vector<float>& vel_x,
    std::vector<float>& vel_y,
    std::vector<float>& vel_z,
    const std::vector<float>& masses) {

    auto contacts = detectCollisions(pos_x, pos_y, pos_z);

    for (auto& contact : contacts) {
        CollisionResponse::resolveImpulseCollision(contact, vel_x, vel_y, vel_z, masses, params);
        CollisionResponse::applySeparationImpulse(contact, pos_x, pos_y, pos_z, masses);
    }
}

void CollisionDetector::clearStatistics() {
    broad_phase_pairs = 0;
    narrow_phase_tests = 0;
    actual_contacts = 0;
}

// CollisionUtils Implementation
namespace CollisionUtils {

float radiusFromMass(float mass, float density) {
    // Assuming spherical particles: V = (4/3)πr³, m = ρV
    // r = ∛(3m/(4πρ))
    const float pi = 3.14159265359f;
    return std::cbrt(3.0f * mass / (4.0f * pi * density));
}

float effectiveRadius(float base_radius, float margin) {
    return base_radius + margin;
}

bool areApproaching(
    float x1, float y1, float z1, float vx1, float vy1, float vz1,
    float x2, float y2, float z2, float vx2, float vy2, float vz2) {

    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;

    float rel_vx = vx2 - vx1;
    float rel_vy = vy2 - vy1;
    float rel_vz = vz2 - vz1;

    // Dot product of relative position and relative velocity
    float dot_product = dx * rel_vx + dy * rel_vy + dz * rel_vz;

    return dot_product < 0.0f;  // Negative means approaching
}

void computeRelativeVelocity(
    const ContactInfo& contact,
    const std::vector<float>& vel_x,
    const std::vector<float>& vel_y,
    const std::vector<float>& vel_z,
    float& rel_vel_normal,
    float& rel_vel_tangent_x,
    float& rel_vel_tangent_y,
    float& rel_vel_tangent_z) {

    int i = contact.body_i;
    int j = contact.body_j;

    float rel_vel_x = vel_x[j] - vel_x[i];
    float rel_vel_y = vel_y[j] - vel_y[i];
    float rel_vel_z = vel_z[j] - vel_z[i];

    rel_vel_normal = rel_vel_x * contact.normal_x +
                     rel_vel_y * contact.normal_y +
                     rel_vel_z * contact.normal_z;

    rel_vel_tangent_x = rel_vel_x - rel_vel_normal * contact.normal_x;
    rel_vel_tangent_y = rel_vel_y - rel_vel_normal * contact.normal_y;
    rel_vel_tangent_z = rel_vel_z - rel_vel_normal * contact.normal_z;
}

} // namespace CollisionUtils

} // namespace physgrad