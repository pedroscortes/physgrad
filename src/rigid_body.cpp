#include "rigid_body.h"
#include "logging_system.h"
#include <cmath>
#include <algorithm>
#include <cassert>

namespace physgrad {

Quaternion Quaternion::operator*(const Quaternion& q) const {
    return Quaternion(
        w * q.w - x * q.x - y * q.y - z * q.z,
        w * q.x + x * q.w + y * q.z - z * q.y,
        w * q.y - x * q.z + y * q.w + z * q.x,
        w * q.z + x * q.y - y * q.x + z * q.w
    );
}

Quaternion Quaternion::conjugate() const {
    return Quaternion(w, -x, -y, -z);
}

void Quaternion::normalize() {
    float length = std::sqrt(w*w + x*x + y*y + z*z);
    if (length > 1e-8f) {
        float inv_length = 1.0f / length;
        w *= inv_length;
        x *= inv_length;
        y *= inv_length;
        z *= inv_length;
    }
}

void Quaternion::toMatrix(float matrix[9]) const {
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    matrix[0] = 1.0f - 2.0f * (yy + zz);
    matrix[1] = 2.0f * (xy - wz);
    matrix[2] = 2.0f * (xz + wy);
    matrix[3] = 2.0f * (xy + wz);
    matrix[4] = 1.0f - 2.0f * (xx + zz);
    matrix[5] = 2.0f * (yz - wx);
    matrix[6] = 2.0f * (xz - wy);
    matrix[7] = 2.0f * (yz + wx);
    matrix[8] = 1.0f - 2.0f * (xx + yy);
}

void Quaternion::fromAxisAngle(const float axis[3], float angle) {
    float half_angle = angle * 0.5f;
    float sin_half = std::sin(half_angle);
    w = std::cos(half_angle);
    x = axis[0] * sin_half;
    y = axis[1] * sin_half;
    z = axis[2] * sin_half;
}

Matrix3x3::Matrix3x3() {
    for (int i = 0; i < 9; ++i) m[i] = 0.0f;
    m[0] = m[4] = m[8] = 1.0f;
}

Matrix3x3::Matrix3x3(const float data[9]) {
    for (int i = 0; i < 9; ++i) m[i] = data[i];
}

Matrix3x3::Matrix3x3(const Quaternion& q) {
    q.toMatrix(m);
}

Matrix3x3 Matrix3x3::operator*(const Matrix3x3& other) const {
    Matrix3x3 result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result.m[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                result.m[i * 3 + j] += m[i * 3 + k] * other.m[k * 3 + j];
            }
        }
    }
    return result;
}

void Matrix3x3::multiply(const float vec[3], float result[3]) const {
    result[0] = m[0] * vec[0] + m[1] * vec[1] + m[2] * vec[2];
    result[1] = m[3] * vec[0] + m[4] * vec[1] + m[5] * vec[2];
    result[2] = m[6] * vec[0] + m[7] * vec[1] + m[8] * vec[2];
}

void Matrix3x3::multiplyTransposed(const float vec[3], float result[3]) const {
    result[0] = m[0] * vec[0] + m[3] * vec[1] + m[6] * vec[2];
    result[1] = m[1] * vec[0] + m[4] * vec[1] + m[7] * vec[2];
    result[2] = m[2] * vec[0] + m[5] * vec[1] + m[8] * vec[2];
}

Matrix3x3 Matrix3x3::transpose() const {
    Matrix3x3 result;
    result.m[0] = m[0]; result.m[1] = m[3]; result.m[2] = m[6];
    result.m[3] = m[1]; result.m[4] = m[4]; result.m[5] = m[7];
    result.m[6] = m[2]; result.m[7] = m[5]; result.m[8] = m[8];
    return result;
}

float Matrix3x3::determinant() const {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) -
           m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
}

Matrix3x3 Matrix3x3::inverse() const {
    Matrix3x3 result;
    float det = determinant();

    if (std::abs(det) < 1e-8f) {
        return Matrix3x3();
    }

    float inv_det = 1.0f / det;

    result.m[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    result.m[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
    result.m[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    result.m[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
    result.m[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    result.m[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
    result.m[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    result.m[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
    result.m[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;

    return result;
}

void RigidBodyState::updateDerivedQuantities(const RigidBodyParams& params) {
    orientation.normalize();
    rotation_matrix = Matrix3x3(orientation);

    Matrix3x3 inertia_local(params.inertia_tensor);
    inertia_world = rotation_matrix * inertia_local * rotation_matrix.transpose();
    inertia_world_inv = inertia_world.inverse();
}

RigidBody::RigidBody(const RigidBodyParams& p) : params(p) {
    state.updateDerivedQuantities(params);
    setSphereGeometry(1.0f);
}

void RigidBody::applyForce(const float force[3]) {
    if (params.kinematic) return;

    state.force[0] += force[0];
    state.force[1] += force[1];
    state.force[2] += force[2];
}

void RigidBody::applyForceAtPoint(const float force[3], const float point[3]) {
    if (params.kinematic) return;

    applyForce(force);

    float r[3] = {
        point[0] - state.position[0],
        point[1] - state.position[1],
        point[2] - state.position[2]
    };

    float torque[3] = {
        r[1] * force[2] - r[2] * force[1],
        r[2] * force[0] - r[0] * force[2],
        r[0] * force[1] - r[1] * force[0]
    };

    applyTorque(torque);
}

void RigidBody::applyTorque(const float torque[3]) {
    if (params.kinematic) return;

    state.torque[0] += torque[0];
    state.torque[1] += torque[1];
    state.torque[2] += torque[2];
}

void RigidBody::applyImpulse(const float impulse[3]) {
    if (params.kinematic) return;

    float inv_mass = 1.0f / params.mass;
    state.velocity[0] += impulse[0] * inv_mass;
    state.velocity[1] += impulse[1] * inv_mass;
    state.velocity[2] += impulse[2] * inv_mass;
}

void RigidBody::applyImpulseAtPoint(const float impulse[3], const float point[3]) {
    if (params.kinematic) return;

    applyImpulse(impulse);

    float r[3] = {
        point[0] - state.position[0],
        point[1] - state.position[1],
        point[2] - state.position[2]
    };

    float angular_impulse[3] = {
        r[1] * impulse[2] - r[2] * impulse[1],
        r[2] * impulse[0] - r[0] * impulse[2],
        r[0] * impulse[1] - r[1] * impulse[0]
    };

    float world_angular_impulse[3];
    state.inertia_world_inv.multiply(angular_impulse, world_angular_impulse);

    state.angular_velocity[0] += world_angular_impulse[0];
    state.angular_velocity[1] += world_angular_impulse[1];
    state.angular_velocity[2] += world_angular_impulse[2];
}

void RigidBody::integrate(float dt) {
    if (params.kinematic) return;

    float inv_mass = 1.0f / params.mass;

    state.velocity[0] += state.force[0] * inv_mass * dt;
    state.velocity[1] += state.force[1] * inv_mass * dt;
    state.velocity[2] += state.force[2] * inv_mass * dt;

    state.position[0] += state.velocity[0] * dt;
    state.position[1] += state.velocity[1] * dt;
    state.position[2] += state.velocity[2] * dt;

    float world_torque[3];
    state.inertia_world_inv.multiply(state.torque, world_torque);

    state.angular_velocity[0] += world_torque[0] * dt;
    state.angular_velocity[1] += world_torque[1] * dt;
    state.angular_velocity[2] += world_torque[2] * dt;

    float angular_magnitude = std::sqrt(
        state.angular_velocity[0] * state.angular_velocity[0] +
        state.angular_velocity[1] * state.angular_velocity[1] +
        state.angular_velocity[2] * state.angular_velocity[2]
    );

    if (angular_magnitude > 1e-8f) {
        float angle = angular_magnitude * dt;
        float axis[3] = {
            state.angular_velocity[0] / angular_magnitude,
            state.angular_velocity[1] / angular_magnitude,
            state.angular_velocity[2] / angular_magnitude
        };

        Quaternion rotation;
        rotation.fromAxisAngle(axis, angle);
        state.orientation = rotation * state.orientation;
    }

    state.velocity[0] *= (1.0f - params.linear_damping * dt);
    state.velocity[1] *= (1.0f - params.linear_damping * dt);
    state.velocity[2] *= (1.0f - params.linear_damping * dt);

    state.angular_velocity[0] *= (1.0f - params.angular_damping * dt);
    state.angular_velocity[1] *= (1.0f - params.angular_damping * dt);
    state.angular_velocity[2] *= (1.0f - params.angular_damping * dt);

    state.updateDerivedQuantities(params);
}

void RigidBody::clearForces() {
    state.force[0] = state.force[1] = state.force[2] = 0.0f;
    state.torque[0] = state.torque[1] = state.torque[2] = 0.0f;
}

void RigidBody::setSphereGeometry(float radius) {
    shape_type = 0;
    collision_vertices.clear();
    collision_indices.clear();

    collision_vertices.push_back(radius);

    RigidBodyUtils::computeSphereInertia(params.mass, radius, params.inertia_tensor);
    state.updateDerivedQuantities(params);
}

void RigidBody::setBoxGeometry(float width, float height, float depth) {
    shape_type = 1;
    collision_vertices.clear();
    collision_indices.clear();

    collision_vertices = {width, height, depth};

    RigidBodyUtils::computeBoxInertia(params.mass, width, height, depth, params.inertia_tensor);
    state.updateDerivedQuantities(params);
}

void RigidBody::setCylinderGeometry(float radius, float height) {
    shape_type = 2;
    collision_vertices.clear();
    collision_indices.clear();

    collision_vertices = {radius, height};

    RigidBodyUtils::computeCylinderInertia(params.mass, radius, height, params.inertia_tensor);
    state.updateDerivedQuantities(params);
}

void RigidBody::setMeshGeometry(const std::vector<float>& vertices, const std::vector<int>& indices) {
    shape_type = 3;
    collision_vertices = vertices;
    collision_indices = indices;

    RigidBodyUtils::computeMeshInertia(params.mass, vertices, indices, params.inertia_tensor);
    state.updateDerivedQuantities(params);
}

void RigidBody::worldToLocal(const float world_point[3], float local_point[3]) const {
    float relative[3] = {
        world_point[0] - state.position[0],
        world_point[1] - state.position[1],
        world_point[2] - state.position[2]
    };
    state.rotation_matrix.multiplyTransposed(relative, local_point);
}

void RigidBody::localToWorld(const float local_point[3], float world_point[3]) const {
    state.rotation_matrix.multiply(local_point, world_point);
    world_point[0] += state.position[0];
    world_point[1] += state.position[1];
    world_point[2] += state.position[2];
}

void RigidBody::worldVectorToLocal(const float world_vector[3], float local_vector[3]) const {
    state.rotation_matrix.multiplyTransposed(world_vector, local_vector);
}

void RigidBody::localVectorToWorld(const float local_vector[3], float world_vector[3]) const {
    state.rotation_matrix.multiply(local_vector, world_vector);
}

bool RigidBody::pointInside(const float point[3]) const {
    float local_point[3];
    worldToLocal(point, local_point);

    switch (shape_type) {
        case 0: { // Sphere
            float radius = collision_vertices[0];
            float dist_sq = local_point[0]*local_point[0] + local_point[1]*local_point[1] + local_point[2]*local_point[2];
            return dist_sq <= radius * radius;
        }
        case 1: { // Box
            float half_width = collision_vertices[0] * 0.5f;
            float half_height = collision_vertices[1] * 0.5f;
            float half_depth = collision_vertices[2] * 0.5f;
            return (std::abs(local_point[0]) <= half_width &&
                   std::abs(local_point[1]) <= half_height &&
                   std::abs(local_point[2]) <= half_depth);
        }
        default:
            return false;
    }
}

RigidBodySystem::RigidBodySystem(const RigidBodySystemParams& params) : system_params(params) {
}

int RigidBodySystem::addRigidBody(std::unique_ptr<RigidBody> body) {
    int id = static_cast<int>(bodies.size());
    bodies.push_back(std::move(body));
    return id;
}

void RigidBodySystem::removeRigidBody(int body_id) {
    if (body_id >= 0 && body_id < static_cast<int>(bodies.size())) {
        bodies.erase(bodies.begin() + body_id);
    }
}

RigidBody* RigidBodySystem::getRigidBody(int body_id) {
    if (body_id >= 0 && body_id < static_cast<int>(bodies.size())) {
        return bodies[body_id].get();
    }
    return nullptr;
}

const RigidBody* RigidBodySystem::getRigidBody(int body_id) const {
    if (body_id >= 0 && body_id < static_cast<int>(bodies.size())) {
        return bodies[body_id].get();
    }
    return nullptr;
}

void RigidBodySystem::step(float dt) {
    clearAllForces();
    applyGravity();

    detectCollisions();
    resolveConstraints(dt);

    for (auto& body : bodies) {
        body->integrate(dt);
    }
}

void RigidBodySystem::applyGravity() {
    float gravity_force[3] = {
        system_params.gravity[0],
        system_params.gravity[1],
        system_params.gravity[2]
    };

    for (auto& body : bodies) {
        if (body->getParameters().enable_gravity && !body->getParameters().kinematic) {
            float force[3] = {
                gravity_force[0] * body->getParameters().mass,
                gravity_force[1] * body->getParameters().mass,
                gravity_force[2] * body->getParameters().mass
            };
            body->applyForce(force);
        }
    }
}

void RigidBodySystem::clearAllForces() {
    for (auto& body : bodies) {
        body->clearForces();
    }
}

void RigidBodySystem::detectCollisions() {
    contact_constraints.clear();

    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            ContactConstraint contact;
            if (checkCollisionBetweenBodies(static_cast<int>(i), static_cast<int>(j), contact)) {
                contact_constraints.push_back(contact);
            }
        }
    }
}

bool RigidBodySystem::checkCollisionBetweenBodies(int body_a, int body_b, ContactConstraint& contact) {
    contact.body_a = body_a;
    contact.body_b = body_b;

    const RigidBody* rb_a = getRigidBody(body_a);
    const RigidBody* rb_b = getRigidBody(body_b);

    if (!rb_a || !rb_b) return false;

    if (rb_a->getShapeType() == 0 && rb_b->getShapeType() == 0) {
        return RigidBodyUtils::sphereSphereCollision(*rb_a, *rb_b, contact);
    } else if ((rb_a->getShapeType() == 0 && rb_b->getShapeType() == 1) ||
               (rb_a->getShapeType() == 1 && rb_b->getShapeType() == 0)) {
        if (rb_a->getShapeType() == 0) {
            return RigidBodyUtils::sphereBoxCollision(*rb_a, *rb_b, contact);
        } else {
            return RigidBodyUtils::sphereBoxCollision(*rb_b, *rb_a, contact);
        }
    }

    return false;
}

void RigidBodySystem::addContactConstraint(const ContactConstraint& constraint) {
    contact_constraints.push_back(constraint);
}

void RigidBodySystem::clearContactConstraints() {
    contact_constraints.clear();
}

void RigidBodySystem::buildConstraintSystem() {
    lambda_normal.resize(contact_constraints.size());
    lambda_friction.resize(contact_constraints.size() * 2);
    jacobian_normal.resize(contact_constraints.size());
    jacobian_friction.resize(contact_constraints.size() * 2);

    for (size_t i = 0; i < contact_constraints.size(); ++i) {
        jacobian_normal[i].resize(bodies.size() * 6);
        jacobian_friction[i * 2].resize(bodies.size() * 6);
        jacobian_friction[i * 2 + 1].resize(bodies.size() * 6);

        std::fill(jacobian_normal[i].begin(), jacobian_normal[i].end(), 0.0f);
        std::fill(jacobian_friction[i * 2].begin(), jacobian_friction[i * 2].end(), 0.0f);
        std::fill(jacobian_friction[i * 2 + 1].begin(), jacobian_friction[i * 2 + 1].end(), 0.0f);

        computeContactJacobian(contact_constraints[i], jacobian_normal[i], jacobian_friction[i * 2]);
    }
}

void RigidBodySystem::solveConstraintSystem(float dt) {
    last_constraint_iterations = 0;
    last_constraint_residual = 0.0f;

    for (int iter = 0; iter < system_params.constraint_iterations; ++iter) {
        float max_residual = 0.0f;

        for (size_t i = 0; i < contact_constraints.size(); ++i) {
            const auto& contact = contact_constraints[i];
            RigidBody* body_a = getRigidBody(contact.body_a);
            RigidBody* body_b = getRigidBody(contact.body_b);

            if (!body_a || !body_b) continue;

            float rel_vel_normal = 0.0f;
            float bias = contact.penetration / dt * system_params.contact_stiffness;

            float lambda_old = lambda_normal[i];
            lambda_normal[i] = std::max(0.0f, lambda_old - (rel_vel_normal + bias) / system_params.contact_damping);
            float delta_lambda = lambda_normal[i] - lambda_old;

            max_residual = std::max(max_residual, std::abs(delta_lambda));

            float impulse[3] = {
                contact.contact_normal[0] * delta_lambda,
                contact.contact_normal[1] * delta_lambda,
                contact.contact_normal[2] * delta_lambda
            };

            body_a->applyImpulseAtPoint(impulse, contact.contact_point_a);

            float neg_impulse[3] = {-impulse[0], -impulse[1], -impulse[2]};
            body_b->applyImpulseAtPoint(neg_impulse, contact.contact_point_b);
        }

        last_constraint_iterations = iter + 1;
        last_constraint_residual = max_residual;

        if (max_residual < system_params.constraint_tolerance) {
            break;
        }
    }
}

void RigidBodySystem::computeContactJacobian(const ContactConstraint& contact,
                                           std::vector<float>& jac_normal,
                                           std::vector<float>& jac_friction) {
    int body_a_offset = contact.body_a * 6;
    int body_b_offset = contact.body_b * 6;

    jac_normal[body_a_offset + 0] = -contact.contact_normal[0];
    jac_normal[body_a_offset + 1] = -contact.contact_normal[1];
    jac_normal[body_a_offset + 2] = -contact.contact_normal[2];

    jac_normal[body_b_offset + 0] = contact.contact_normal[0];
    jac_normal[body_b_offset + 1] = contact.contact_normal[1];
    jac_normal[body_b_offset + 2] = contact.contact_normal[2];

    RigidBody* body_a = getRigidBody(contact.body_a);
    RigidBody* body_b = getRigidBody(contact.body_b);

    if (body_a) {
        const auto& state_a = body_a->getState();
        float r_a[3] = {
            contact.contact_point_a[0] - state_a.position[0],
            contact.contact_point_a[1] - state_a.position[1],
            contact.contact_point_a[2] - state_a.position[2]
        };

        jac_normal[body_a_offset + 3] = -(r_a[1] * contact.contact_normal[2] - r_a[2] * contact.contact_normal[1]);
        jac_normal[body_a_offset + 4] = -(r_a[2] * contact.contact_normal[0] - r_a[0] * contact.contact_normal[2]);
        jac_normal[body_a_offset + 5] = -(r_a[0] * contact.contact_normal[1] - r_a[1] * contact.contact_normal[0]);
    }

    if (body_b) {
        const auto& state_b = body_b->getState();
        float r_b[3] = {
            contact.contact_point_b[0] - state_b.position[0],
            contact.contact_point_b[1] - state_b.position[1],
            contact.contact_point_b[2] - state_b.position[2]
        };

        jac_normal[body_b_offset + 3] = r_b[1] * contact.contact_normal[2] - r_b[2] * contact.contact_normal[1];
        jac_normal[body_b_offset + 4] = r_b[2] * contact.contact_normal[0] - r_b[0] * contact.contact_normal[2];
        jac_normal[body_b_offset + 5] = r_b[0] * contact.contact_normal[1] - r_b[1] * contact.contact_normal[0];
    }
}

void RigidBodySystem::resolveConstraints(float dt) {
    if (contact_constraints.empty()) return;

    buildConstraintSystem();
    solveConstraintSystem(dt);
}

void RigidBodySystem::getVisualizationData(
    std::vector<float>& positions_x,
    std::vector<float>& positions_y,
    std::vector<float>& positions_z,
    std::vector<float>& orientations_w,
    std::vector<float>& orientations_x,
    std::vector<float>& orientations_y,
    std::vector<float>& orientations_z,
    std::vector<int>& shape_types) const {

    size_t n = bodies.size();
    positions_x.resize(n);
    positions_y.resize(n);
    positions_z.resize(n);
    orientations_w.resize(n);
    orientations_x.resize(n);
    orientations_y.resize(n);
    orientations_z.resize(n);
    shape_types.resize(n);

    for (size_t i = 0; i < n; ++i) {
        const auto& state = bodies[i]->getState();
        positions_x[i] = state.position[0];
        positions_y[i] = state.position[1];
        positions_z[i] = state.position[2];

        orientations_w[i] = state.orientation.w;
        orientations_x[i] = state.orientation.x;
        orientations_y[i] = state.orientation.y;
        orientations_z[i] = state.orientation.z;

        shape_types[i] = bodies[i]->getShapeType();
    }
}

namespace RigidBodyUtils {

void computeSphereInertia(float mass, float radius, float inertia[9]) {
    for (int i = 0; i < 9; ++i) inertia[i] = 0.0f;
    float moment = 0.4f * mass * radius * radius;
    inertia[0] = inertia[4] = inertia[8] = moment;
}

void computeBoxInertia(float mass, float width, float height, float depth, float inertia[9]) {
    for (int i = 0; i < 9; ++i) inertia[i] = 0.0f;
    float mass_over_12 = mass / 12.0f;
    inertia[0] = mass_over_12 * (height * height + depth * depth);
    inertia[4] = mass_over_12 * (width * width + depth * depth);
    inertia[8] = mass_over_12 * (width * width + height * height);
}

void computeCylinderInertia(float mass, float radius, float height, float inertia[9]) {
    for (int i = 0; i < 9; ++i) inertia[i] = 0.0f;
    float r_sq = radius * radius;
    float h_sq = height * height;
    inertia[0] = inertia[8] = mass * (3.0f * r_sq + h_sq) / 12.0f;
    inertia[4] = 0.5f * mass * r_sq;
}

void computeMeshInertia(float mass, const std::vector<float>&, const std::vector<int>&, float inertia[9]) {
    for (int i = 0; i < 9; ++i) inertia[i] = 0.0f;
    float moment = mass / 6.0f;
    inertia[0] = inertia[4] = inertia[8] = moment;
}

bool sphereSphereCollision(const RigidBody& body_a, const RigidBody& body_b, ContactConstraint& contact) {
    const auto& state_a = body_a.getState();
    const auto& state_b = body_b.getState();

    float dx = state_b.position[0] - state_a.position[0];
    float dy = state_b.position[1] - state_a.position[1];
    float dz = state_b.position[2] - state_a.position[2];
    float distance = std::sqrt(dx*dx + dy*dy + dz*dz);

    float radius_a = body_a.getCollisionVertices()[0];
    float radius_b = body_b.getCollisionVertices()[0];
    float radius_sum = radius_a + radius_b;

    if (distance < radius_sum && distance > 1e-8f) {
        contact.penetration = radius_sum - distance;

        float inv_distance = 1.0f / distance;
        contact.contact_normal[0] = dx * inv_distance;
        contact.contact_normal[1] = dy * inv_distance;
        contact.contact_normal[2] = dz * inv_distance;

        contact.contact_point_a[0] = state_a.position[0] + contact.contact_normal[0] * radius_a;
        contact.contact_point_a[1] = state_a.position[1] + contact.contact_normal[1] * radius_a;
        contact.contact_point_a[2] = state_a.position[2] + contact.contact_normal[2] * radius_a;

        contact.contact_point_b[0] = state_b.position[0] - contact.contact_normal[0] * radius_b;
        contact.contact_point_b[1] = state_b.position[1] - contact.contact_normal[1] * radius_b;
        contact.contact_point_b[2] = state_b.position[2] - contact.contact_normal[2] * radius_b;

        return true;
    }

    return false;
}

bool sphereBoxCollision(const RigidBody&, const RigidBody&, ContactConstraint&) {
    return false;
}

bool boxBoxCollision(const RigidBody&, const RigidBody&, ContactConstraint&) {
    return false;
}

void setupTowerOfBlocks(RigidBodySystem& system, int num_blocks) {
    for (int i = 0; i < num_blocks; ++i) {
        RigidBodyParams params;
        params.mass = 1.0f;
        params.linear_damping = 0.05f;
        params.angular_damping = 0.05f;

        auto body = std::make_unique<RigidBody>(params);
        body->setBoxGeometry(1.0f, 0.5f, 1.0f);

        auto& state = body->getState();
        state.position[0] = 0.0f;
        state.position[1] = 0.25f + i * 0.6f;
        state.position[2] = 0.0f;

        system.addRigidBody(std::move(body));
    }

    Logger::getInstance().info("rigid_body", "Created tower of " + std::to_string(num_blocks) + " blocks");
}

void setupBouncingBalls(RigidBodySystem& system, int num_balls) {
    for (int i = 0; i < num_balls; ++i) {
        RigidBodyParams params;
        params.mass = 0.5f + i * 0.1f;
        params.linear_damping = 0.01f;
        params.angular_damping = 0.01f;

        auto body = std::make_unique<RigidBody>(params);
        body->setSphereGeometry(0.3f + i * 0.05f);

        auto& state = body->getState();
        state.position[0] = -3.0f + i * 0.8f;
        state.position[1] = 5.0f + i * 0.2f;
        state.position[2] = 0.0f;

        state.velocity[0] = 1.0f + i * 0.3f;
        state.velocity[1] = 0.0f;
        state.velocity[2] = 0.0f;

        system.addRigidBody(std::move(body));
    }

    Logger::getInstance().info("rigid_body", "Created " + std::to_string(num_balls) + " bouncing balls");
}

} // namespace RigidBodyUtils

} // namespace physgrad