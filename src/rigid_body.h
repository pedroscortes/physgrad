#pragma once

#include <vector>
#include <memory>
#include <functional>

namespace physgrad {

struct Quaternion {
    float w, x, y, z;

    Quaternion() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}
    Quaternion(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}

    Quaternion operator*(const Quaternion& q) const;
    Quaternion conjugate() const;
    void normalize();
    void toMatrix(float matrix[9]) const;
    void fromAxisAngle(const float axis[3], float angle);
};

struct Matrix3x3 {
    float m[9];

    Matrix3x3();
    explicit Matrix3x3(const float data[9]);
    Matrix3x3(const Quaternion& q);

    Matrix3x3 operator*(const Matrix3x3& other) const;
    void multiply(const float vec[3], float result[3]) const;
    void multiplyTransposed(const float vec[3], float result[3]) const;
    Matrix3x3 transpose() const;
    Matrix3x3 inverse() const;
    float determinant() const;
};

struct RigidBodyParams {
    float mass = 1.0f;
    float inertia_tensor[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float linear_damping = 0.01f;
    float angular_damping = 0.01f;
    bool kinematic = false;
    bool enable_gravity = true;
};

struct RigidBodyState {
    // Linear motion
    float position[3] = {0.0f, 0.0f, 0.0f};
    float velocity[3] = {0.0f, 0.0f, 0.0f};
    float force[3] = {0.0f, 0.0f, 0.0f};

    // Angular motion
    Quaternion orientation;
    float angular_velocity[3] = {0.0f, 0.0f, 0.0f};
    float torque[3] = {0.0f, 0.0f, 0.0f};

    // Derived quantities
    Matrix3x3 rotation_matrix;
    Matrix3x3 inertia_world;
    Matrix3x3 inertia_world_inv;

    void updateDerivedQuantities(const RigidBodyParams& params);
};

struct ContactConstraint {
    int body_a, body_b;
    float contact_point_a[3];
    float contact_point_b[3];
    float contact_normal[3];
    float penetration;
    float restitution = 0.3f;
    float friction = 0.5f;
    bool is_active = true;
};

class RigidBody {
private:
    RigidBodyParams params;
    RigidBodyState state;
    std::vector<float> collision_vertices;
    std::vector<int> collision_indices;
    int shape_type = 0; // 0=sphere, 1=box, 2=cylinder, 3=mesh

public:
    RigidBody(const RigidBodyParams& p = RigidBodyParams{});
    ~RigidBody() = default;

    void setParameters(const RigidBodyParams& p) { params = p; }
    const RigidBodyParams& getParameters() const { return params; }

    RigidBodyState& getState() { return state; }
    const RigidBodyState& getState() const { return state; }

    // Force and torque application
    void applyForce(const float force[3]);
    void applyForceAtPoint(const float force[3], const float point[3]);
    void applyTorque(const float torque[3]);
    void applyImpulse(const float impulse[3]);
    void applyImpulseAtPoint(const float impulse[3], const float point[3]);

    // Integration
    void integrate(float dt);
    void clearForces();

    // Collision geometry
    void setSphereGeometry(float radius);
    void setBoxGeometry(float width, float height, float depth);
    void setCylinderGeometry(float radius, float height);
    void setMeshGeometry(const std::vector<float>& vertices, const std::vector<int>& indices);

    // Utility functions
    void worldToLocal(const float world_point[3], float local_point[3]) const;
    void localToWorld(const float local_point[3], float world_point[3]) const;
    void worldVectorToLocal(const float world_vector[3], float local_vector[3]) const;
    void localVectorToWorld(const float local_vector[3], float world_vector[3]) const;

    // Collision queries
    bool pointInside(const float point[3]) const;
    float distanceToPoint(const float point[3]) const;
    bool rayIntersect(const float ray_origin[3], const float ray_direction[3],
                     float& t, float intersection_point[3], float normal[3]) const;

    int getShapeType() const { return shape_type; }
    const std::vector<float>& getCollisionVertices() const { return collision_vertices; }
    const std::vector<int>& getCollisionIndices() const { return collision_indices; }
};

struct RigidBodySystemParams {
    float gravity[3] = {0.0f, -9.81f, 0.0f};
    float contact_stiffness = 1e6f;
    float contact_damping = 1e3f;
    int constraint_iterations = 10;
    float constraint_tolerance = 1e-6f;
    bool enable_sleeping = true;
    float sleep_threshold = 0.01f;
    bool enable_continuous_collision = false;
};

class RigidBodySystem {
private:
    std::vector<std::unique_ptr<RigidBody>> bodies;
    std::vector<ContactConstraint> contact_constraints;
    RigidBodySystemParams system_params;

    // Constraint solving
    std::vector<float> lambda_normal;
    std::vector<float> lambda_friction;
    std::vector<std::vector<float>> jacobian_normal;
    std::vector<std::vector<float>> jacobian_friction;

    // Performance tracking
    int last_constraint_iterations = 0;
    float last_constraint_residual = 0.0f;

public:
    RigidBodySystem(const RigidBodySystemParams& params = RigidBodySystemParams{});
    ~RigidBodySystem() = default;

    // Disable copy constructor and assignment operator due to unique_ptr
    RigidBodySystem(const RigidBodySystem&) = delete;
    RigidBodySystem& operator=(const RigidBodySystem&) = delete;

    // Enable move constructor and assignment operator
    RigidBodySystem(RigidBodySystem&&) = default;
    RigidBodySystem& operator=(RigidBodySystem&&) = default;

    // Body management
    int addRigidBody(std::unique_ptr<RigidBody> body);
    void removeRigidBody(int body_id);
    RigidBody* getRigidBody(int body_id);
    const RigidBody* getRigidBody(int body_id) const;
    size_t getBodyCount() const { return bodies.size(); }

    // System parameters
    void setSystemParams(const RigidBodySystemParams& params) { system_params = params; }
    const RigidBodySystemParams& getSystemParams() const { return system_params; }

    // Simulation step
    void step(float dt);

    // Force application
    void applyGravity();
    void clearAllForces();

    // Collision detection and response
    void detectCollisions();
    void resolveConstraints(float dt);

    // Constraint management
    void addContactConstraint(const ContactConstraint& constraint);
    void clearContactConstraints();
    const std::vector<ContactConstraint>& getContactConstraints() const { return contact_constraints; }

    // Visualization helpers
    void getVisualizationData(
        std::vector<float>& positions_x,
        std::vector<float>& positions_y,
        std::vector<float>& positions_z,
        std::vector<float>& orientations_w,
        std::vector<float>& orientations_x,
        std::vector<float>& orientations_y,
        std::vector<float>& orientations_z,
        std::vector<int>& shape_types
    ) const;

    // Performance metrics
    int getLastConstraintIterations() const { return last_constraint_iterations; }
    float getLastConstraintResidual() const { return last_constraint_residual; }

private:
    void buildConstraintSystem();
    void solveConstraintSystem(float dt);
    bool checkCollisionBetweenBodies(int body_a, int body_b, ContactConstraint& contact);
    void computeContactJacobian(const ContactConstraint& contact,
                               std::vector<float>& jac_normal,
                               std::vector<float>& jac_friction);
};

// Utility functions for rigid body physics
namespace RigidBodyUtils {
    // Inertia tensor calculations
    void computeSphereInertia(float mass, float radius, float inertia[9]);
    void computeBoxInertia(float mass, float width, float height, float depth, float inertia[9]);
    void computeCylinderInertia(float mass, float radius, float height, float inertia[9]);
    void computeMeshInertia(float mass, const std::vector<float>& vertices,
                           const std::vector<int>& indices, float inertia[9]);

    // Quaternion utilities
    Quaternion quaternionFromEuler(float roll, float pitch, float yaw);
    void quaternionToEuler(const Quaternion& q, float& roll, float& pitch, float& yaw);
    Quaternion quaternionSlerp(const Quaternion& q1, const Quaternion& q2, float t);

    // Matrix utilities
    void matrixFromQuaternion(const Quaternion& q, float matrix[9]);
    Quaternion quaternionFromMatrix(const float matrix[9]);
    void transformPoint(const float matrix[9], const float point[3], float result[3]);
    void transformVector(const float matrix[9], const float vector[3], float result[3]);

    // Collision utilities
    bool sphereSphereCollision(const RigidBody& body_a, const RigidBody& body_b, ContactConstraint& contact);
    bool sphereBoxCollision(const RigidBody& sphere, const RigidBody& box, ContactConstraint& contact);
    bool boxBoxCollision(const RigidBody& body_a, const RigidBody& body_b, ContactConstraint& contact);

    // Demo setup functions
    void setupTowerOfBlocks(RigidBodySystem& system, int num_blocks = 5);
    void setupBouncingBalls(RigidBodySystem& system, int num_balls = 10);
    void setupDominoChain(RigidBodySystem& system, int num_dominoes = 20);
    void setupNewtonsCradle(RigidBodySystem& system, int num_balls = 5);
}

} // namespace physgrad