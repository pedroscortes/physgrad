#include "constraints.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace physgrad {

// Base Constraint Implementation
Constraint::Constraint(ConstraintType t, const std::vector<int>& indices,
                      const ConstraintParams& p, const std::string& n)
    : type(t), params(p), particle_indices(indices), name(n) {
}

void Constraint::applyConstraintForces(
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

    if (!is_active || is_broken) return;

    // Compute constraint violation
    float C = evaluateConstraint(pos_x, pos_y, pos_z);
    current_violation = C;

    // Compute jacobian
    std::vector<std::vector<float>> jacobian(particle_indices.size(), std::vector<float>(3, 0.0f));
    computeJacobian(pos_x, pos_y, pos_z, jacobian);

    // Compute constraint force using spring-damper model
    float force_magnitude = -params.stiffness * C;

    // Add damping based on constraint velocity (J * v)
    float constraint_velocity = 0.0f;
    for (size_t i = 0; i < particle_indices.size(); ++i) {
        int idx = particle_indices[i];
        constraint_velocity += jacobian[i][0] * vel_x[idx] +
                              jacobian[i][1] * vel_y[idx] +
                              jacobian[i][2] * vel_z[idx];
    }

    force_magnitude -= params.damping * constraint_velocity;
    current_force = std::abs(force_magnitude);

    // Check breaking condition
    if (current_force > params.breaking_force) {
        is_broken = true;
        return;
    }

    // Apply forces to particles
    for (size_t i = 0; i < particle_indices.size(); ++i) {
        int idx = particle_indices[i];
        force_x[idx] += force_magnitude * jacobian[i][0];
        force_y[idx] += force_magnitude * jacobian[i][1];
        force_z[idx] += force_magnitude * jacobian[i][2];
    }
}

bool Constraint::shouldBreak() const {
    return current_force > params.breaking_force;
}

void Constraint::computeNumericalJacobian(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    std::vector<std::vector<float>>& jacobian,
    float epsilon) const {

    jacobian.resize(particle_indices.size(), std::vector<float>(3, 0.0f));

    std::vector<float> pos_x_temp = pos_x;
    std::vector<float> pos_y_temp = pos_y;
    std::vector<float> pos_z_temp = pos_z;

    for (size_t i = 0; i < particle_indices.size(); ++i) {
        int idx = particle_indices[i];

        // X derivative
        pos_x_temp[idx] += epsilon;
        float C_plus = evaluateConstraint(pos_x_temp, pos_y_temp, pos_z_temp);
        pos_x_temp[idx] -= 2.0f * epsilon;
        float C_minus = evaluateConstraint(pos_x_temp, pos_y_temp, pos_z_temp);
        jacobian[i][0] = (C_plus - C_minus) / (2.0f * epsilon);
        pos_x_temp[idx] = pos_x[idx];

        // Y derivative
        pos_y_temp[idx] += epsilon;
        C_plus = evaluateConstraint(pos_x_temp, pos_y_temp, pos_z_temp);
        pos_y_temp[idx] -= 2.0f * epsilon;
        C_minus = evaluateConstraint(pos_x_temp, pos_y_temp, pos_z_temp);
        jacobian[i][1] = (C_plus - C_minus) / (2.0f * epsilon);
        pos_y_temp[idx] = pos_y[idx];

        // Z derivative
        pos_z_temp[idx] += epsilon;
        C_plus = evaluateConstraint(pos_x_temp, pos_y_temp, pos_z_temp);
        pos_z_temp[idx] -= 2.0f * epsilon;
        C_minus = evaluateConstraint(pos_x_temp, pos_y_temp, pos_z_temp);
        jacobian[i][2] = (C_plus - C_minus) / (2.0f * epsilon);
        pos_z_temp[idx] = pos_z[idx];
    }
}

// DistanceConstraint Implementation
DistanceConstraint::DistanceConstraint(int particle_a, int particle_b, float distance,
                                     const ConstraintParams& params)
    : Constraint(ConstraintType::DISTANCE, {particle_a, particle_b}, params,
                "Distance(" + std::to_string(particle_a) + "," + std::to_string(particle_b) + ")") {
    this->params.rest_length = distance;
}

float DistanceConstraint::evaluateConstraint(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z) const {

    int a = particle_indices[0];
    int b = particle_indices[1];

    float dx = pos_x[b] - pos_x[a];
    float dy = pos_y[b] - pos_y[a];
    float dz = pos_z[b] - pos_z[a];

    float current_distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    return current_distance - params.rest_length;
}

void DistanceConstraint::computeJacobian(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    std::vector<std::vector<float>>& jacobian) const {

    jacobian.resize(2, std::vector<float>(3, 0.0f));

    int a = particle_indices[0];
    int b = particle_indices[1];

    float dx = pos_x[b] - pos_x[a];
    float dy = pos_y[b] - pos_y[a];
    float dz = pos_z[b] - pos_z[a];

    float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (distance < 1e-8f) {
        // Handle degenerate case
        jacobian[0][0] = -1.0f; jacobian[1][0] = 1.0f;
        return;
    }

    float inv_distance = 1.0f / distance;

    // Jacobian for particle a (negative direction)
    jacobian[0][0] = -dx * inv_distance;
    jacobian[0][1] = -dy * inv_distance;
    jacobian[0][2] = -dz * inv_distance;

    // Jacobian for particle b (positive direction)
    jacobian[1][0] = dx * inv_distance;
    jacobian[1][1] = dy * inv_distance;
    jacobian[1][2] = dz * inv_distance;
}

// SpringConstraint Implementation
SpringConstraint::SpringConstraint(int particle_a, int particle_b, float rest_length, float stiffness,
                                  const ConstraintParams& params)
    : Constraint(ConstraintType::SPRING, {particle_a, particle_b}, params,
                "Spring(" + std::to_string(particle_a) + "," + std::to_string(particle_b) + ")") {
    this->params.rest_length = rest_length;
    this->params.stiffness = stiffness;
}

float SpringConstraint::evaluateConstraint(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z) const {

    int a = particle_indices[0];
    int b = particle_indices[1];

    float dx = pos_x[b] - pos_x[a];
    float dy = pos_y[b] - pos_y[a];
    float dz = pos_z[b] - pos_z[a];

    float current_distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    return current_distance - params.rest_length;
}

void SpringConstraint::computeJacobian(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    std::vector<std::vector<float>>& jacobian) const {

    // Same implementation as distance constraint for position-based springs
    jacobian.resize(2, std::vector<float>(3, 0.0f));

    int a = particle_indices[0];
    int b = particle_indices[1];

    float dx = pos_x[b] - pos_x[a];
    float dy = pos_y[b] - pos_y[a];
    float dz = pos_z[b] - pos_z[a];

    float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (distance < 1e-8f) {
        jacobian[0][0] = -1.0f; jacobian[1][0] = 1.0f;
        return;
    }

    float inv_distance = 1.0f / distance;

    // Jacobian for particle a (negative direction)
    jacobian[0][0] = -dx * inv_distance;
    jacobian[0][1] = -dy * inv_distance;
    jacobian[0][2] = -dz * inv_distance;

    // Jacobian for particle b (positive direction)
    jacobian[1][0] = dx * inv_distance;
    jacobian[1][1] = dy * inv_distance;
    jacobian[1][2] = dz * inv_distance;
}

// BallJointConstraint Implementation
BallJointConstraint::BallJointConstraint(int particle_a, int particle_b,
                                       const ConstraintParams& params)
    : Constraint(ConstraintType::BALL_JOINT, {particle_a, particle_b}, params,
                "BallJoint(" + std::to_string(particle_a) + "," + std::to_string(particle_b) + ")") {
}

float BallJointConstraint::evaluateConstraint(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z) const {

    int a = particle_indices[0];
    int b = particle_indices[1];

    float dx = pos_x[b] - pos_x[a];
    float dy = pos_y[b] - pos_y[a];
    float dz = pos_z[b] - pos_z[a];

    // Constraint is satisfied when distance is zero
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

void BallJointConstraint::computeJacobian(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    std::vector<std::vector<float>>& jacobian) const {

    jacobian.resize(2, std::vector<float>(3, 0.0f));

    int a = particle_indices[0];
    int b = particle_indices[1];

    float dx = pos_x[b] - pos_x[a];
    float dy = pos_y[b] - pos_y[a];
    float dz = pos_z[b] - pos_z[a];

    float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (distance < 1e-8f) {
        jacobian[0][0] = -1.0f; jacobian[1][0] = 1.0f;
        return;
    }

    float inv_distance = 1.0f / distance;

    jacobian[0][0] = -dx * inv_distance;
    jacobian[0][1] = -dy * inv_distance;
    jacobian[0][2] = -dz * inv_distance;

    jacobian[1][0] = dx * inv_distance;
    jacobian[1][1] = dy * inv_distance;
    jacobian[1][2] = dz * inv_distance;
}

// RopeConstraint Implementation
RopeConstraint::RopeConstraint(int particle_a, int particle_b, float max_length,
                              const ConstraintParams& params)
    : Constraint(ConstraintType::ROPE, {particle_a, particle_b}, params,
                "Rope(" + std::to_string(particle_a) + "," + std::to_string(particle_b) + ")") {
    this->params.max_length = max_length;
    this->params.bilateral = false;  // Inequality constraint
}

float RopeConstraint::evaluateConstraint(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z) const {

    int a = particle_indices[0];
    int b = particle_indices[1];

    float dx = pos_x[b] - pos_x[a];
    float dy = pos_y[b] - pos_y[a];
    float dz = pos_z[b] - pos_z[a];

    float current_distance = std::sqrt(dx*dx + dy*dy + dz*dz);

    // Inequality constraint: distance <= max_length
    // Return positive if constraint is violated
    return std::max(0.0f, current_distance - params.max_length);
}

void RopeConstraint::computeJacobian(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    std::vector<std::vector<float>>& jacobian) const {

    int a = particle_indices[0];
    int b = particle_indices[1];

    float dx = pos_x[b] - pos_x[a];
    float dy = pos_y[b] - pos_y[a];
    float dz = pos_z[b] - pos_z[a];

    float distance = std::sqrt(dx*dx + dy*dy + dz*dz);

    jacobian.resize(2, std::vector<float>(3, 0.0f));

    // Only apply constraint if rope is taut
    if (distance <= params.max_length) {
        return;  // Jacobian is zero when constraint is not active
    }

    // Use already computed variables
    if (distance < 1e-8f) {
        jacobian[0][0] = -1.0f; jacobian[1][0] = 1.0f;
        return;
    }

    float inv_distance = 1.0f / distance;

    jacobian[0][0] = -dx * inv_distance;
    jacobian[0][1] = -dy * inv_distance;
    jacobian[0][2] = -dz * inv_distance;

    jacobian[1][0] = dx * inv_distance;
    jacobian[1][1] = dy * inv_distance;
    jacobian[1][2] = dz * inv_distance;
}

// PositionLockConstraint Implementation
PositionLockConstraint::PositionLockConstraint(int particle_idx, float x, float y, float z,
                                             const ConstraintParams& params)
    : Constraint(ConstraintType::POSITION_LOCK, {particle_idx}, params,
                "PositionLock(" + std::to_string(particle_idx) + ")"),
      target_x(x), target_y(y), target_z(z) {
}

float PositionLockConstraint::evaluateConstraint(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z) const {

    int idx = particle_indices[0];

    float dx = pos_x[idx] - target_x;
    float dy = pos_y[idx] - target_y;
    float dz = pos_z[idx] - target_z;

    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

void PositionLockConstraint::computeJacobian(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    std::vector<std::vector<float>>& jacobian) const {

    jacobian.resize(1, std::vector<float>(3, 0.0f));

    int idx = particle_indices[0];

    float dx = pos_x[idx] - target_x;
    float dy = pos_y[idx] - target_y;
    float dz = pos_z[idx] - target_z;

    float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (distance < 1e-8f) {
        jacobian[0][0] = 1.0f;  // Default direction
        return;
    }

    float inv_distance = 1.0f / distance;
    jacobian[0][0] = dx * inv_distance;
    jacobian[0][1] = dy * inv_distance;
    jacobian[0][2] = dz * inv_distance;
}

// MotorConstraint Implementation
MotorConstraint::MotorConstraint(int particle_a, int particle_b, float angular_velocity,
                               const ConstraintParams& params)
    : Constraint(ConstraintType::MOTOR, {particle_a, particle_b}, params,
                "Motor(" + std::to_string(particle_a) + "," + std::to_string(particle_b) + ")"),
      target_angular_velocity(angular_velocity) {
}

float MotorConstraint::evaluateConstraint(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z) const {

    // For simplicity, motor constraint maintains target angular velocity
    // This is more complex in practice and would require angular state tracking
    return 0.0f;  // Placeholder implementation
}

void MotorConstraint::computeJacobian(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    std::vector<std::vector<float>>& jacobian) const {

    jacobian.resize(2, std::vector<float>(3, 0.0f));
    // Placeholder - would need proper angular jacobian computation
}

// ConstraintSolver Implementation
ConstraintSolver::ConstraintSolver(const ConstraintSolverParams& params)
    : solver_params(params) {
}

void ConstraintSolver::addConstraint(std::unique_ptr<Constraint> constraint) {
    constraints.push_back(std::move(constraint));
}

void ConstraintSolver::removeConstraint(size_t index) {
    if (index < constraints.size()) {
        constraints.erase(constraints.begin() + index);
    }
}

void ConstraintSolver::clearConstraints() {
    constraints.clear();
}

void ConstraintSolver::addDistanceConstraint(int particle_a, int particle_b, float distance,
                                           const ConstraintParams& params) {
    addConstraint(std::make_unique<DistanceConstraint>(particle_a, particle_b, distance, params));
}

void ConstraintSolver::addSpringConstraint(int particle_a, int particle_b, float rest_length, float stiffness,
                                         const ConstraintParams& params) {
    auto spring_params = params;
    spring_params.rest_length = rest_length;
    spring_params.stiffness = stiffness;
    addConstraint(std::make_unique<SpringConstraint>(particle_a, particle_b, rest_length, stiffness, spring_params));
}

void ConstraintSolver::addBallJoint(int particle_a, int particle_b, const ConstraintParams& params) {
    addConstraint(std::make_unique<BallJointConstraint>(particle_a, particle_b, params));
}

void ConstraintSolver::addRopeConstraint(int particle_a, int particle_b, float max_length,
                                       const ConstraintParams& params) {
    auto rope_params = params;
    rope_params.max_length = max_length;
    addConstraint(std::make_unique<RopeConstraint>(particle_a, particle_b, max_length, rope_params));
}

void ConstraintSolver::addPositionLock(int particle_idx, float x, float y, float z,
                                     const ConstraintParams& params) {
    addConstraint(std::make_unique<PositionLockConstraint>(particle_idx, x, y, z, params));
}

void ConstraintSolver::solveConstraints(
    std::vector<float>& pos_x,
    std::vector<float>& pos_y,
    std::vector<float>& pos_z,
    std::vector<float>& vel_x,
    std::vector<float>& vel_y,
    std::vector<float>& vel_z,
    const std::vector<float>& masses,
    float dt) {

    if (constraints.empty()) return;

    converged = false;
    last_residual = 1e6f;

    // Iterative constraint satisfaction
    for (int iter = 0; iter < solver_params.max_iterations; ++iter) {
        float max_violation = 0.0f;

        // Process each constraint
        for (auto& constraint : constraints) {
            if (!constraint->is_active || constraint->is_broken) continue;

            float violation = std::abs(constraint->evaluateConstraint(pos_x, pos_y, pos_z));
            max_violation = std::max(max_violation, violation);

            if (violation > solver_params.tolerance) {
                // Apply position correction using Gauss-Seidel method
                if (solver_params.project_positions) {
                    projectSingleConstraint(*constraint, pos_x, pos_y, pos_z, masses);
                }
            }
        }

        last_residual = max_violation;
        last_iterations = iter + 1;

        if (max_violation < solver_params.tolerance) {
            converged = true;
            break;
        }
    }

    // Project velocities if needed
    if (solver_params.project_velocities) {
        projectVelocities(vel_x, vel_y, vel_z, masses);
    }
}

void ConstraintSolver::projectSingleConstraint(
    Constraint& constraint,
    std::vector<float>& pos_x,
    std::vector<float>& pos_y,
    std::vector<float>& pos_z,
    const std::vector<float>& masses) {

    float C = constraint.evaluateConstraint(pos_x, pos_y, pos_z);
    if (std::abs(C) < solver_params.tolerance) return;

    std::vector<std::vector<float>> jacobian;
    constraint.computeJacobian(pos_x, pos_y, pos_z, jacobian);

    // Compute correction using Gauss-Seidel
    float numerator = -C * solver_params.position_correction;
    float denominator = 0.0f;

    for (size_t i = 0; i < constraint.particle_indices.size(); ++i) {
        int idx = constraint.particle_indices[i];
        float inv_mass = 1.0f / masses[idx];
        denominator += inv_mass * (jacobian[i][0] * jacobian[i][0] +
                                  jacobian[i][1] * jacobian[i][1] +
                                  jacobian[i][2] * jacobian[i][2]);
    }

    if (denominator < 1e-8f) return;

    float lambda = numerator / denominator;

    // Apply position corrections
    for (size_t i = 0; i < constraint.particle_indices.size(); ++i) {
        int idx = constraint.particle_indices[i];
        float inv_mass = 1.0f / masses[idx];

        pos_x[idx] += lambda * inv_mass * jacobian[i][0];
        pos_y[idx] += lambda * inv_mass * jacobian[i][1];
        pos_z[idx] += lambda * inv_mass * jacobian[i][2];
    }
}

void ConstraintSolver::applyConstraintForces(
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

    for (auto& constraint : constraints) {
        constraint->applyConstraintForces(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
                                        force_x, force_y, force_z, masses);
    }
}

void ConstraintSolver::projectVelocities(
    std::vector<float>& vel_x,
    std::vector<float>& vel_y,
    std::vector<float>& vel_z,
    const std::vector<float>& masses) {

    for (auto& constraint : constraints) {
        if (!constraint->is_active || constraint->is_broken) continue;
        if (!constraint->params.bilateral) continue;  // Only for equality constraints

        // Compute constraint velocity (J * v)
        std::vector<std::vector<float>> jacobian(constraint->particle_indices.size(),
                                                std::vector<float>(3, 0.0f));

        // Use current positions to compute jacobian (approximation)
        // In practice, you'd need to store positions from constraint evaluation

        float constraint_velocity = 0.0f;
        for (size_t i = 0; i < constraint->particle_indices.size(); ++i) {
            int idx = constraint->particle_indices[i];
            constraint_velocity += jacobian[i][0] * vel_x[idx] +
                                 jacobian[i][1] * vel_y[idx] +
                                 jacobian[i][2] * vel_z[idx];
        }

        if (std::abs(constraint_velocity) < solver_params.tolerance) continue;

        // Compute velocity correction
        float numerator = -constraint_velocity * solver_params.velocity_correction;
        float denominator = 0.0f;

        for (size_t i = 0; i < constraint->particle_indices.size(); ++i) {
            int idx = constraint->particle_indices[i];
            float inv_mass = 1.0f / masses[idx];
            denominator += inv_mass * (jacobian[i][0] * jacobian[i][0] +
                                      jacobian[i][1] * jacobian[i][1] +
                                      jacobian[i][2] * jacobian[i][2]);
        }

        if (denominator < 1e-8f) continue;

        float lambda = numerator / denominator;

        // Apply velocity corrections
        for (size_t i = 0; i < constraint->particle_indices.size(); ++i) {
            int idx = constraint->particle_indices[i];
            float inv_mass = 1.0f / masses[idx];

            vel_x[idx] += lambda * inv_mass * jacobian[i][0];
            vel_y[idx] += lambda * inv_mass * jacobian[i][1];
            vel_z[idx] += lambda * inv_mass * jacobian[i][2];
        }
    }
}

void ConstraintSolver::getConstraintVisualizationData(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    std::vector<std::pair<int, int>>& connections,
    std::vector<float>& connection_forces,
    std::vector<int>& connection_types) const {

    connections.clear();
    connection_forces.clear();
    connection_types.clear();

    for (const auto& constraint : constraints) {
        if (!constraint->is_active || constraint->is_broken) continue;
        if (constraint->particle_indices.size() != 2) continue;

        connections.emplace_back(constraint->particle_indices[0], constraint->particle_indices[1]);
        connection_forces.push_back(constraint->current_force);
        connection_types.push_back(static_cast<int>(constraint->type));
    }
}

// ConstraintUtils Implementation
namespace ConstraintUtils {

std::vector<size_t> createChain(
    ConstraintSolver& solver,
    const std::vector<int>& particle_indices,
    float segment_length,
    const ConstraintParams& params) {

    std::vector<size_t> constraint_indices;

    for (size_t i = 0; i < particle_indices.size() - 1; ++i) {
        size_t constraint_index = solver.getConstraints().size();
        solver.addDistanceConstraint(particle_indices[i], particle_indices[i + 1],
                                   segment_length, params);
        constraint_indices.push_back(constraint_index);
    }

    return constraint_indices;
}

std::vector<size_t> createRope(
    ConstraintSolver& solver,
    const std::vector<int>& particle_indices,
    float max_segment_length,
    const ConstraintParams& params) {

    std::vector<size_t> constraint_indices;

    for (size_t i = 0; i < particle_indices.size() - 1; ++i) {
        size_t constraint_index = solver.getConstraints().size();
        solver.addRopeConstraint(particle_indices[i], particle_indices[i + 1],
                               max_segment_length, params);
        constraint_indices.push_back(constraint_index);
    }

    return constraint_indices;
}

float computeConstraintError(
    const ConstraintSolver& solver,
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z) {

    float total_error = 0.0f;
    int active_constraints = 0;

    for (const auto& constraint : solver.getConstraints()) {
        if (!constraint->is_active || constraint->is_broken) continue;

        float violation = std::abs(constraint->evaluateConstraint(pos_x, pos_y, pos_z));
        total_error += violation * violation;
        active_constraints++;
    }

    return active_constraints > 0 ? std::sqrt(total_error / active_constraints) : 0.0f;
}

} // namespace ConstraintUtils

} // namespace physgrad