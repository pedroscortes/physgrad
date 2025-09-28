#pragma once

#include <vector>
#include <memory>
#include <functional>

namespace physgrad {

struct ContactPoint {
    int body_a, body_b;
    float contact_point[3];
    float contact_normal[3];
    float penetration_depth;
    float normal_force;
    float friction_force[2];
    bool is_active;
};

struct MaterialProperties {
    float restitution = 0.3f;
    float friction_static = 0.8f;
    float friction_dynamic = 0.6f;
    float softness = 1e-4f;
    float damping = 0.1f;
};

struct DifferentiableContactParams {
    float contact_stiffness = 1e6f;
    float contact_damping = 1e3f;
    float friction_regularization = 1e-6f;
    float penetration_tolerance = 1e-4f;
    int max_contact_iterations = 20;
    bool enable_friction = true;
    bool enable_rolling_friction = false;
    float rolling_friction_coeff = 0.01f;
};

class DifferentiableContactSolver {
private:
    std::vector<ContactPoint> contacts;
    std::vector<MaterialProperties> materials;
    DifferentiableContactParams params;

    // Gradient computation storage
    std::vector<std::vector<float>> position_gradients;
    std::vector<std::vector<float>> velocity_gradients;
    std::vector<std::vector<float>> force_gradients;

    // Contact detection and response
    std::vector<float> contact_jacobian;
    std::vector<float> contact_forces;
    std::vector<float> contact_velocities;

    // Smooth contact force computation
    float smoothStep(float edge0, float edge1, float x) const;
    float smoothMax(float a, float b, float k) const;

public:
    DifferentiableContactSolver(const DifferentiableContactParams& p = DifferentiableContactParams{});
    ~DifferentiableContactSolver() = default;

    void setParameters(const DifferentiableContactParams& p) { params = p; }
    const DifferentiableContactParams& getParameters() const { return params; }

    void setMaterialProperties(int material_id, const MaterialProperties& props);
    MaterialProperties getMaterialProperties(int material_id) const;

    // Core contact resolution
    void detectContacts(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        const std::vector<float>& radii,
        const std::vector<int>& material_ids
    );

    void computeContactForces(
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
        const std::vector<int>& material_ids
    );

    // Differentiable interface
    void computeContactGradients(
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
        std::vector<float>& input_gradients_vel_z
    );

    // Smooth contact force functions for differentiability
    float computeSmoothContactForce(
        float penetration,
        float relative_velocity,
        const MaterialProperties& material
    ) const;

    void computeSmoothFrictionForce(
        const float relative_velocity[2],
        float normal_force,
        const MaterialProperties& material,
        float friction_force[2]
    ) const;

    // Contact energy and potential functions
    float computeContactEnergy(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        const std::vector<float>& radii,
        const std::vector<int>& material_ids
    ) const;

    float computeContactPotential(float distance, float radius_sum, const MaterialProperties& material) const;

    // Utility and debugging
    const std::vector<ContactPoint>& getContacts() const { return contacts; }
    size_t getContactCount() const { return contacts.size(); }

    void clearContacts() { contacts.clear(); }

    // Gradient checking utilities
    bool checkGradients(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        const std::vector<float>& vel_x,
        const std::vector<float>& vel_y,
        const std::vector<float>& vel_z,
        const std::vector<float>& masses,
        const std::vector<float>& radii,
        const std::vector<int>& material_ids,
        float epsilon = 1e-6f
    );

private:
    void computeContactJacobian(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        const std::vector<float>& radii
    );

    void solveContactConstraints(
        const std::vector<float>& vel_x,
        const std::vector<float>& vel_y,
        const std::vector<float>& vel_z,
        const std::vector<float>& masses
    );

    void updateContactForces(
        std::vector<float>& force_x,
        std::vector<float>& force_y,
        std::vector<float>& force_z
    );
};

// High-level differentiable physics step function
class DifferentiablePhysicsStep {
private:
    DifferentiableContactSolver contact_solver;

public:
    DifferentiablePhysicsStep(const DifferentiableContactParams& params = DifferentiableContactParams{});

    // Forward pass
    void forward(
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
        const std::vector<float>& external_force_x = {},
        const std::vector<float>& external_force_y = {},
        const std::vector<float>& external_force_z = {}
    );

    // Backward pass for gradient computation
    void backward(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        const std::vector<float>& vel_x,
        const std::vector<float>& vel_y,
        const std::vector<float>& vel_z,
        const std::vector<float>& masses,
        const std::vector<float>& radii,
        const std::vector<int>& material_ids,
        float dt,
        const std::vector<float>& grad_output_pos_x,
        const std::vector<float>& grad_output_pos_y,
        const std::vector<float>& grad_output_pos_z,
        const std::vector<float>& grad_output_vel_x,
        const std::vector<float>& grad_output_vel_y,
        const std::vector<float>& grad_output_vel_z,
        std::vector<float>& grad_input_pos_x,
        std::vector<float>& grad_input_pos_y,
        std::vector<float>& grad_input_pos_z,
        std::vector<float>& grad_input_vel_x,
        std::vector<float>& grad_input_vel_y,
        std::vector<float>& grad_input_vel_z
    );

    DifferentiableContactSolver& getContactSolver() { return contact_solver; }
    const DifferentiableContactSolver& getContactSolver() const { return contact_solver; }
};

namespace ContactUtils {
    // Utilities for setting up differentiable contact scenarios
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
        int num_balls = 10
    );

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
        int num_blocks = 5
    );

    // Gradient verification helpers
    float computeNumericalGradient(
        const std::function<float()>& function,
        std::vector<float>& parameter,
        int param_index,
        float epsilon = 1e-6f
    );

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
        float tolerance = 1e-3f
    );
}

} // namespace physgrad