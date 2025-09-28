#pragma once

#include <vector>
#include <memory>
#include <unordered_set>
#include <unordered_map>

namespace physgrad {

struct ContactInfo {
    int body_i;                // Index of first body
    int body_j;                // Index of second body
    float contact_distance;    // Distance at contact
    float overlap;            // Penetration depth
    float normal_x, normal_y, normal_z;  // Contact normal (from i to j)
    float contact_x, contact_y, contact_z;  // Contact point

    // Material properties
    float restitution = 0.8f;  // Coefficient of restitution
    float friction = 0.3f;     // Coefficient of friction
};

struct CollisionParams {
    float contact_threshold = 0.01f;  // Distance threshold for contact
    float contact_stiffness = 1000.0f;  // Contact spring stiffness
    float contact_damping = 10.0f;      // Contact damping coefficient
    float min_separation_velocity = 0.01f;  // Minimum velocity for separation

    // Collision response parameters
    bool enable_restitution = true;
    bool enable_friction = true;
    float global_restitution = 0.8f;
    float global_friction = 0.3f;

    // Performance parameters
    int max_contacts_per_body = 10;
    float broad_phase_margin = 0.1f;  // Extra margin for broad phase
};

class BroadPhase {
public:
    virtual ~BroadPhase() = default;

    virtual void updateBodies(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        const std::vector<float>& radii
    ) = 0;

    virtual std::vector<std::pair<int, int>> getPotentialCollisionPairs() = 0;
    virtual void clear() = 0;
};

class SpatialHashBroadPhase : public BroadPhase {
private:
    struct Cell {
        std::vector<int> bodies;
    };

    float cell_size;
    float inv_cell_size;
    std::unordered_map<uint64_t, Cell> spatial_hash;

    uint64_t hashPosition(float x, float y, float z) const;
    void getCellCoords(float x, float y, float z, int& cx, int& cy, int& cz) const;

public:
    SpatialHashBroadPhase(float cell_size = 1.0f);

    void updateBodies(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        const std::vector<float>& radii
    ) override;

    std::vector<std::pair<int, int>> getPotentialCollisionPairs() override;
    void clear() override;
};

class NarrowPhase {
public:
    static bool checkSphereCollision(
        float x1, float y1, float z1, float r1,
        float x2, float y2, float z2, float r2,
        ContactInfo& contact
    );

    static bool checkSphereBoxCollision(
        float sx, float sy, float sz, float sr,  // Sphere
        float bx, float by, float bz,            // Box center
        float bw, float bh, float bd,            // Box dimensions
        ContactInfo& contact
    );

    static bool checkBoxCollision(
        float x1, float y1, float z1, float w1, float h1, float d1,
        float x2, float y2, float z2, float w2, float h2, float d2,
        ContactInfo& contact
    );
};

class CollisionResponse {
public:
    static void applyContactForces(
        const std::vector<ContactInfo>& contacts,
        std::vector<float>& force_x,
        std::vector<float>& force_y,
        std::vector<float>& force_z,
        const std::vector<float>& vel_x,
        const std::vector<float>& vel_y,
        const std::vector<float>& vel_z,
        const std::vector<float>& masses,
        const CollisionParams& params
    );

    static void resolveImpulseCollision(
        ContactInfo& contact,
        std::vector<float>& vel_x,
        std::vector<float>& vel_y,
        std::vector<float>& vel_z,
        const std::vector<float>& masses,
        const CollisionParams& params
    );

    static void applySeparationImpulse(
        ContactInfo& contact,
        std::vector<float>& pos_x,
        std::vector<float>& pos_y,
        std::vector<float>& pos_z,
        const std::vector<float>& masses
    );
};

class CollisionDetector {
private:
    std::unique_ptr<BroadPhase> broad_phase;
    CollisionParams params;
    std::vector<ContactInfo> current_contacts;
    std::vector<float> body_radii;

    // Performance tracking
    int broad_phase_pairs = 0;
    int narrow_phase_tests = 0;
    int actual_contacts = 0;

public:
    CollisionDetector(const CollisionParams& collision_params = CollisionParams{});
    ~CollisionDetector() = default;

    void setBroadPhase(std::unique_ptr<BroadPhase> broad_phase_impl);
    void setParameters(const CollisionParams& collision_params);
    const CollisionParams& getParameters() const { return params; }

    void updateBodyRadii(const std::vector<float>& radii);
    void updateBodyRadiiFromMasses(const std::vector<float>& masses, float density = 1.0f);

    std::vector<ContactInfo> detectCollisions(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    );

    void applyCollisionForces(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        const std::vector<float>& vel_x,
        const std::vector<float>& vel_y,
        const std::vector<float>& vel_z,
        std::vector<float>& force_x,
        std::vector<float>& force_y,
        std::vector<float>& force_z,
        const std::vector<float>& masses
    );

    void resolveCollisions(
        std::vector<float>& pos_x,
        std::vector<float>& pos_y,
        std::vector<float>& pos_z,
        std::vector<float>& vel_x,
        std::vector<float>& vel_y,
        std::vector<float>& vel_z,
        const std::vector<float>& masses
    );

    // Performance and debugging
    const std::vector<ContactInfo>& getCurrentContacts() const { return current_contacts; }
    int getBroadPhasePairs() const { return broad_phase_pairs; }
    int getNarrowPhaseTests() const { return narrow_phase_tests; }
    int getActualContacts() const { return actual_contacts; }

    void clearStatistics();

    // Differentiable collision detection (for gradients)
    void computeContactGradients(
        const std::vector<ContactInfo>& contacts,
        const std::vector<float>& grad_force_x,
        const std::vector<float>& grad_force_y,
        const std::vector<float>& grad_force_z,
        std::vector<float>& grad_pos_x,
        std::vector<float>& grad_pos_y,
        std::vector<float>& grad_pos_z,
        std::vector<float>& grad_vel_x,
        std::vector<float>& grad_vel_y,
        std::vector<float>& grad_vel_z
    );
};

// Utility functions for collision detection in simulations
namespace CollisionUtils {
    // Compute radius from mass assuming spherical particles and given density
    float radiusFromMass(float mass, float density = 1.0f);

    // Compute effective radius for collision detection (may include safety margin)
    float effectiveRadius(float base_radius, float margin = 0.05f);

    // Check if two bodies are moving towards each other
    bool areApproaching(
        float x1, float y1, float z1, float vx1, float vy1, float vz1,
        float x2, float y2, float z2, float vx2, float vy2, float vz2
    );

    // Compute relative velocity at contact point
    void computeRelativeVelocity(
        const ContactInfo& contact,
        const std::vector<float>& vel_x,
        const std::vector<float>& vel_y,
        const std::vector<float>& vel_z,
        float& rel_vel_normal,
        float& rel_vel_tangent_x,
        float& rel_vel_tangent_y,
        float& rel_vel_tangent_z
    );
}

} // namespace physgrad