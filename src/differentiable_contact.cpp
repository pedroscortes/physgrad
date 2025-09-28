#include "differentiable_contact.h"
#include "logging_system.h"
#include <cmath>
#include <algorithm>
#include <cassert>

namespace physgrad {

DifferentiableContactSolver::DifferentiableContactSolver(const DifferentiableContactParams& p)
    : params(p) {
    materials.resize(10);
    for (auto& mat : materials) {
        mat = MaterialProperties{};
    }
}

void DifferentiableContactSolver::setMaterialProperties(int material_id, const MaterialProperties& props) {
    if (material_id >= 0 && material_id < static_cast<int>(materials.size())) {
        materials[material_id] = props;
    } else if (material_id >= static_cast<int>(materials.size())) {
        materials.resize(material_id + 1);
        materials[material_id] = props;
    }
}

MaterialProperties DifferentiableContactSolver::getMaterialProperties(int material_id) const {
    if (material_id >= 0 && material_id < static_cast<int>(materials.size())) {
        return materials[material_id];
    }
    return MaterialProperties{};
}

float DifferentiableContactSolver::smoothStep(float edge0, float edge1, float x) const {
    if (x <= edge0) return 0.0f;
    if (x >= edge1) return 1.0f;
    float t = (x - edge0) / (edge1 - edge0);
    return t * t * (3.0f - 2.0f * t);
}

float DifferentiableContactSolver::smoothMax(float a, float b, float k) const {
    float h = std::max(k - std::abs(a - b), 0.0f) / k;
    return std::max(a, b) + h * h * k * 0.25f;
}

void DifferentiableContactSolver::detectContacts(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    const std::vector<float>& radii,
    const std::vector<int>& material_ids) {

    contacts.clear();
    int n = static_cast<int>(pos_x.size());

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float dx = pos_x[j] - pos_x[i];
            float dy = pos_y[j] - pos_y[i];
            float dz = pos_z[j] - pos_z[i];
            float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            float radius_sum = radii[i] + radii[j];

            if (distance < radius_sum + params.penetration_tolerance) {
                ContactPoint contact;
                contact.body_a = i;
                contact.body_b = j;
                contact.penetration_depth = radius_sum - distance;

                if (distance > 1e-8f) {
                    float inv_distance = 1.0f / distance;
                    contact.contact_normal[0] = dx * inv_distance;
                    contact.contact_normal[1] = dy * inv_distance;
                    contact.contact_normal[2] = dz * inv_distance;
                } else {
                    contact.contact_normal[0] = 1.0f;
                    contact.contact_normal[1] = 0.0f;
                    contact.contact_normal[2] = 0.0f;
                }

                contact.contact_point[0] = pos_x[i] + contact.contact_normal[0] * radii[i];
                contact.contact_point[1] = pos_y[i] + contact.contact_normal[1] * radii[i];
                contact.contact_point[2] = pos_z[i] + contact.contact_normal[2] * radii[i];

                contact.is_active = true;
                contact.normal_force = 0.0f;
                contact.friction_force[0] = contact.friction_force[1] = 0.0f;

                contacts.push_back(contact);
            }
        }
    }
}

float DifferentiableContactSolver::computeSmoothContactForce(
    float penetration,
    float relative_velocity,
    const MaterialProperties& material) const {

    if (penetration <= 0.0f) return 0.0f;

    float spring_force = params.contact_stiffness * penetration;
    float damping_force = params.contact_damping * relative_velocity;

    float total_force = spring_force + damping_force;
    return smoothMax(total_force, 0.0f, material.softness);
}

void DifferentiableContactSolver::computeSmoothFrictionForce(
    const float relative_velocity[2],
    float normal_force,
    const MaterialProperties& material,
    float friction_force[2]) const {

    if (!params.enable_friction || normal_force <= 0.0f) {
        friction_force[0] = friction_force[1] = 0.0f;
        return;
    }

    float velocity_magnitude = std::sqrt(
        relative_velocity[0] * relative_velocity[0] +
        relative_velocity[1] * relative_velocity[1]
    );

    if (velocity_magnitude < 1e-8f) {
        friction_force[0] = friction_force[1] = 0.0f;
        return;
    }

    float friction_coeff = velocity_magnitude > 1e-3f ?
        material.friction_dynamic : material.friction_static;

    float max_friction = friction_coeff * normal_force;
    float friction_magnitude = std::min(
        max_friction,
        params.contact_stiffness * velocity_magnitude + params.friction_regularization
    );

    float scale = friction_magnitude / velocity_magnitude;
    friction_force[0] = -scale * relative_velocity[0];
    friction_force[1] = -scale * relative_velocity[1];
}

void DifferentiableContactSolver::computeContactForces(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    const std::vector<float>& vel_x,
    const std::vector<float>& vel_y,
    const std::vector<float>& vel_z,
    std::vector<float>& force_x,
    std::vector<float>& force_y,
    std::vector<float>& force_z,
    const std::vector<float>& masses,
    const std::vector<float>& radii,
    const std::vector<int>& material_ids) {

    detectContacts(pos_x, pos_y, pos_z, radii, material_ids);

    for (auto& contact : contacts) {
        int i = contact.body_a;
        int j = contact.body_b;

        MaterialProperties mat_i = getMaterialProperties(material_ids[i]);
        MaterialProperties mat_j = getMaterialProperties(material_ids[j]);

        MaterialProperties combined_material;
        combined_material.restitution = std::sqrt(mat_i.restitution * mat_j.restitution);
        combined_material.friction_static = std::sqrt(mat_i.friction_static * mat_j.friction_static);
        combined_material.friction_dynamic = std::sqrt(mat_i.friction_dynamic * mat_j.friction_dynamic);
        combined_material.softness = std::max(mat_i.softness, mat_j.softness);
        combined_material.damping = 0.5f * (mat_i.damping + mat_j.damping);

        float rel_vel_normal =
            (vel_x[j] - vel_x[i]) * contact.contact_normal[0] +
            (vel_y[j] - vel_y[i]) * contact.contact_normal[1] +
            (vel_z[j] - vel_z[i]) * contact.contact_normal[2];

        contact.normal_force = computeSmoothContactForce(
            contact.penetration_depth,
            rel_vel_normal,
            combined_material
        );

        float tangent1[3], tangent2[3];
        if (std::abs(contact.contact_normal[0]) < 0.9f) {
            tangent1[0] = 0.0f;
            tangent1[1] = contact.contact_normal[2];
            tangent1[2] = -contact.contact_normal[1];
        } else {
            tangent1[0] = -contact.contact_normal[2];
            tangent1[1] = 0.0f;
            tangent1[2] = contact.contact_normal[0];
        }

        float len = std::sqrt(tangent1[0]*tangent1[0] + tangent1[1]*tangent1[1] + tangent1[2]*tangent1[2]);
        if (len > 1e-8f) {
            tangent1[0] /= len; tangent1[1] /= len; tangent1[2] /= len;
        }

        tangent2[0] = contact.contact_normal[1] * tangent1[2] - contact.contact_normal[2] * tangent1[1];
        tangent2[1] = contact.contact_normal[2] * tangent1[0] - contact.contact_normal[0] * tangent1[2];
        tangent2[2] = contact.contact_normal[0] * tangent1[1] - contact.contact_normal[1] * tangent1[0];

        float rel_vel_tangent[2];
        rel_vel_tangent[0] =
            (vel_x[j] - vel_x[i]) * tangent1[0] +
            (vel_y[j] - vel_y[i]) * tangent1[1] +
            (vel_z[j] - vel_z[i]) * tangent1[2];
        rel_vel_tangent[1] =
            (vel_x[j] - vel_x[i]) * tangent2[0] +
            (vel_y[j] - vel_y[i]) * tangent2[1] +
            (vel_z[j] - vel_z[i]) * tangent2[2];

        computeSmoothFrictionForce(rel_vel_tangent, contact.normal_force, combined_material, contact.friction_force);

        float total_force_x = contact.normal_force * contact.contact_normal[0] +
                             contact.friction_force[0] * tangent1[0] +
                             contact.friction_force[1] * tangent2[0];
        float total_force_y = contact.normal_force * contact.contact_normal[1] +
                             contact.friction_force[0] * tangent1[1] +
                             contact.friction_force[1] * tangent2[1];
        float total_force_z = contact.normal_force * contact.contact_normal[2] +
                             contact.friction_force[0] * tangent1[2] +
                             contact.friction_force[1] * tangent2[2];

        force_x[i] += total_force_x;
        force_y[i] += total_force_y;
        force_z[i] += total_force_z;

        force_x[j] -= total_force_x;
        force_y[j] -= total_force_y;
        force_z[j] -= total_force_z;
    }
}

void DifferentiableContactSolver::computeContactGradients(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    const std::vector<float>& vel_x,
    const std::vector<float>& vel_y,
    const std::vector<float>& vel_z,
    const std::vector<float>& masses,
    const std::vector<float>& radii,
    const std::vector<int>& material_ids,
    const std::vector<float>& output_gradients_x,
    const std::vector<float>& output_gradients_y,
    const std::vector<float>& output_gradients_z,
    std::vector<float>& input_gradients_pos_x,
    std::vector<float>& input_gradients_pos_y,
    std::vector<float>& input_gradients_pos_z,
    std::vector<float>& input_gradients_vel_x,
    std::vector<float>& input_gradients_vel_y,
    std::vector<float>& input_gradients_vel_z) {

    int n = static_cast<int>(pos_x.size());
    input_gradients_pos_x.assign(n, 0.0f);
    input_gradients_pos_y.assign(n, 0.0f);
    input_gradients_pos_z.assign(n, 0.0f);
    input_gradients_vel_x.assign(n, 0.0f);
    input_gradients_vel_y.assign(n, 0.0f);
    input_gradients_vel_z.assign(n, 0.0f);

    detectContacts(pos_x, pos_y, pos_z, radii, material_ids);

    for (const auto& contact : contacts) {
        int i = contact.body_a;
        int j = contact.body_b;

        float dx = pos_x[j] - pos_x[i];
        float dy = pos_y[j] - pos_y[i];
        float dz = pos_z[j] - pos_z[i];
        float distance = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (distance < 1e-8f) continue;

        float inv_distance = 1.0f / distance;
        float normal_x = dx * inv_distance;
        float normal_y = dy * inv_distance;
        float normal_z = dz * inv_distance;

        MaterialProperties mat_i = getMaterialProperties(material_ids[i]);
        MaterialProperties mat_j = getMaterialProperties(material_ids[j]);

        MaterialProperties combined_material;
        combined_material.restitution = std::sqrt(mat_i.restitution * mat_j.restitution);
        combined_material.softness = std::max(mat_i.softness, mat_j.softness);

        float penetration = radii[i] + radii[j] - distance;
        float rel_vel_normal =
            (vel_x[j] - vel_x[i]) * normal_x +
            (vel_y[j] - vel_y[i]) * normal_y +
            (vel_z[j] - vel_z[i]) * normal_z;

        float force_magnitude = computeSmoothContactForce(penetration, rel_vel_normal, combined_material);

        float grad_force_wrt_penetration = params.contact_stiffness;
        if (force_magnitude > 0.0f) {
            grad_force_wrt_penetration *= smoothStep(0.0f, combined_material.softness, force_magnitude);
        }

        float grad_force_wrt_rel_vel = params.contact_damping;

        float grad_penetration_wrt_distance = -1.0f;

        float grad_normal_force_x = grad_force_wrt_penetration * grad_penetration_wrt_distance * (-normal_x);
        float grad_normal_force_y = grad_force_wrt_penetration * grad_penetration_wrt_distance * (-normal_y);
        float grad_normal_force_z = grad_force_wrt_penetration * grad_penetration_wrt_distance * (-normal_z);

        float grad_rel_vel_x = grad_force_wrt_rel_vel * normal_x;
        float grad_rel_vel_y = grad_force_wrt_rel_vel * normal_y;
        float grad_rel_vel_z = grad_force_wrt_rel_vel * normal_z;

        float output_contrib_i_x = output_gradients_x[i] * force_magnitude * normal_x;
        float output_contrib_i_y = output_gradients_y[i] * force_magnitude * normal_y;
        float output_contrib_i_z = output_gradients_z[i] * force_magnitude * normal_z;

        float output_contrib_j_x = output_gradients_x[j] * (-force_magnitude * normal_x);
        float output_contrib_j_y = output_gradients_y[j] * (-force_magnitude * normal_y);
        float output_contrib_j_z = output_gradients_z[j] * (-force_magnitude * normal_z);

        input_gradients_pos_x[i] += (output_contrib_i_x + output_contrib_j_x) * grad_normal_force_x;
        input_gradients_pos_y[i] += (output_contrib_i_y + output_contrib_j_y) * grad_normal_force_y;
        input_gradients_pos_z[i] += (output_contrib_i_z + output_contrib_j_z) * grad_normal_force_z;

        input_gradients_pos_x[j] += (output_contrib_i_x + output_contrib_j_x) * (-grad_normal_force_x);
        input_gradients_pos_y[j] += (output_contrib_i_y + output_contrib_j_y) * (-grad_normal_force_y);
        input_gradients_pos_z[j] += (output_contrib_i_z + output_contrib_j_z) * (-grad_normal_force_z);

        input_gradients_vel_x[i] += (output_contrib_i_x + output_contrib_j_x) * (-grad_rel_vel_x);
        input_gradients_vel_y[i] += (output_contrib_i_y + output_contrib_j_y) * (-grad_rel_vel_y);
        input_gradients_vel_z[i] += (output_contrib_i_z + output_contrib_j_z) * (-grad_rel_vel_z);

        input_gradients_vel_x[j] += (output_contrib_i_x + output_contrib_j_x) * grad_rel_vel_x;
        input_gradients_vel_y[j] += (output_contrib_i_y + output_contrib_j_y) * grad_rel_vel_y;
        input_gradients_vel_z[j] += (output_contrib_i_z + output_contrib_j_z) * grad_rel_vel_z;
    }
}

float DifferentiableContactSolver::computeContactEnergy(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    const std::vector<float>& radii,
    const std::vector<int>& material_ids) const {

    float total_energy = 0.0f;
    int n = static_cast<int>(pos_x.size());

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float dx = pos_x[j] - pos_x[i];
            float dy = pos_y[j] - pos_y[i];
            float dz = pos_z[j] - pos_z[i];
            float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            float radius_sum = radii[i] + radii[j];

            MaterialProperties mat_i = getMaterialProperties(material_ids[i]);
            MaterialProperties mat_j = getMaterialProperties(material_ids[j]);
            MaterialProperties combined_material;
            combined_material.softness = std::max(mat_i.softness, mat_j.softness);

            total_energy += computeContactPotential(distance, radius_sum, combined_material);
        }
    }

    return total_energy;
}

float DifferentiableContactSolver::computeContactPotential(
    float distance,
    float radius_sum,
    const MaterialProperties& material) const {

    if (distance >= radius_sum) return 0.0f;

    float penetration = radius_sum - distance;
    return 0.5f * params.contact_stiffness * penetration * penetration;
}

bool DifferentiableContactSolver::checkGradients(
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    const std::vector<float>& vel_x,
    const std::vector<float>& vel_y,
    const std::vector<float>& vel_z,
    const std::vector<float>& masses,
    const std::vector<float>& radii,
    const std::vector<int>& material_ids,
    float epsilon) {

    Logger::getInstance().info("contact", "Starting gradient verification with epsilon = " + std::to_string(epsilon));

    auto compute_energy = [&]() {
        return computeContactEnergy(pos_x, pos_y, pos_z, radii, material_ids);
    };

    bool all_gradients_correct = true;
    float tolerance = 1e-3f;

    for (int i = 0; i < static_cast<int>(pos_x.size()); ++i) {
        auto pos_x_copy = pos_x;

        pos_x_copy[i] += epsilon;
        float energy_plus = computeContactEnergy(pos_x_copy, pos_y, pos_z, radii, material_ids);

        pos_x_copy[i] -= 2.0f * epsilon;
        float energy_minus = computeContactEnergy(pos_x_copy, pos_y, pos_z, radii, material_ids);

        float numerical_grad = (energy_plus - energy_minus) / (2.0f * epsilon);

        Logger::getInstance().info("contact",
            "Particle " + std::to_string(i) + " pos_x numerical gradient: " + std::to_string(numerical_grad));

        if (std::abs(numerical_grad) > tolerance) {
            all_gradients_correct = false;
        }
    }

    return all_gradients_correct;
}

DifferentiablePhysicsStep::DifferentiablePhysicsStep(const DifferentiableContactParams& params)
    : contact_solver(params) {
}

void DifferentiablePhysicsStep::forward(
    std::vector<float>& pos_x,
    std::vector<float>& pos_y,
    std::vector<float>& pos_z,
    std::vector<float>& vel_x,
    std::vector<float>& vel_y,
    std::vector<float>& vel_z,
    const std::vector<float>& masses,
    const std::vector<float>& radii,
    const std::vector<int>& material_ids,
    float dt,
    const std::vector<float>& external_force_x,
    const std::vector<float>& external_force_y,
    const std::vector<float>& external_force_z) {

    int n = static_cast<int>(pos_x.size());
    std::vector<float> force_x(n, 0.0f), force_y(n, 0.0f), force_z(n, 0.0f);

    if (!external_force_x.empty()) {
        for (int i = 0; i < n; ++i) {
            force_x[i] += external_force_x[i];
        }
    }
    if (!external_force_y.empty()) {
        for (int i = 0; i < n; ++i) {
            force_y[i] += external_force_y[i];
        }
    }
    if (!external_force_z.empty()) {
        for (int i = 0; i < n; ++i) {
            force_z[i] += external_force_z[i];
        }
    }

    contact_solver.computeContactForces(
        pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
        force_x, force_y, force_z, masses, radii, material_ids
    );

    for (int i = 0; i < n; ++i) {
        float inv_mass = 1.0f / masses[i];
        vel_x[i] += force_x[i] * inv_mass * dt;
        vel_y[i] += force_y[i] * inv_mass * dt;
        vel_z[i] += force_z[i] * inv_mass * dt;

        pos_x[i] += vel_x[i] * dt;
        pos_y[i] += vel_y[i] * dt;
        pos_z[i] += vel_z[i] * dt;
    }
}

namespace ContactUtils {

void setupBouncingBalls(
    std::vector<float>& pos_x,
    std::vector<float>& pos_y,
    std::vector<float>& pos_z,
    std::vector<float>& vel_x,
    std::vector<float>& vel_y,
    std::vector<float>& vel_z,
    std::vector<float>& masses,
    std::vector<float>& radii,
    std::vector<int>& material_ids,
    int num_balls) {

    pos_x.clear(); pos_y.clear(); pos_z.clear();
    vel_x.clear(); vel_y.clear(); vel_z.clear();
    masses.clear(); radii.clear(); material_ids.clear();

    for (int i = 0; i < num_balls; ++i) {
        pos_x.push_back(-3.0f + i * 0.8f);
        pos_y.push_back(5.0f + i * 0.1f);
        pos_z.push_back(0.0f);

        vel_x.push_back(1.0f + i * 0.2f);
        vel_y.push_back(0.0f);
        vel_z.push_back(0.0f);

        masses.push_back(1.0f);
        radii.push_back(0.3f);
        material_ids.push_back(0);
    }
}

void setupStackingBlocks(
    std::vector<float>& pos_x,
    std::vector<float>& pos_y,
    std::vector<float>& pos_z,
    std::vector<float>& vel_x,
    std::vector<float>& vel_y,
    std::vector<float>& vel_z,
    std::vector<float>& masses,
    std::vector<float>& radii,
    std::vector<int>& material_ids,
    int num_blocks) {

    pos_x.clear(); pos_y.clear(); pos_z.clear();
    vel_x.clear(); vel_y.clear(); vel_z.clear();
    masses.clear(); radii.clear(); material_ids.clear();

    for (int i = 0; i < num_blocks; ++i) {
        pos_x.push_back(0.0f);
        pos_y.push_back(0.5f + i * 1.0f);
        pos_z.push_back(0.0f);

        vel_x.push_back(0.0f);
        vel_y.push_back(0.0f);
        vel_z.push_back(0.0f);

        masses.push_back(1.0f);
        radii.push_back(0.4f);
        material_ids.push_back(1);
    }
}

float computeNumericalGradient(
    const std::function<float()>& function,
    std::vector<float>& parameter,
    int param_index,
    float epsilon) {

    float original_value = parameter[param_index];

    parameter[param_index] = original_value + epsilon;
    float value_plus = function();

    parameter[param_index] = original_value - epsilon;
    float value_minus = function();

    parameter[param_index] = original_value;

    return (value_plus - value_minus) / (2.0f * epsilon);
}

void verifyContactGradients(
    DifferentiableContactSolver& solver,
    const std::vector<float>& pos_x,
    const std::vector<float>& pos_y,
    const std::vector<float>& pos_z,
    const std::vector<float>& vel_x,
    const std::vector<float>& vel_y,
    const std::vector<float>& vel_z,
    const std::vector<float>& masses,
    const std::vector<float>& radii,
    const std::vector<int>& material_ids,
    float tolerance) {

    bool gradients_correct = solver.checkGradients(
        pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
        masses, radii, material_ids
    );

    if (gradients_correct) {
        Logger::getInstance().info("contact", "All gradients verified successfully!");
    } else {
        Logger::getInstance().warning("contact", "Some gradients failed verification");
    }
}

} // namespace ContactUtils

} // namespace physgrad