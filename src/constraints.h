#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace physgrad {

// Forward declarations
class ConstraintSolver;

enum class ConstraintType {
    DISTANCE,           // Fixed distance between two particles
    SPRING,            // Spring force between particles
    HINGE,             // Hinge joint (rotational constraint)
    BALL_JOINT,        // Ball joint (point constraint)
    ROPE,              // Rope constraint (max distance)
    RIGID_BODY,        // Rigid body constraint
    MOTOR,             // Motor constraint (driven rotation)
    PRISMATIC,         // Sliding joint
    ANGLE_LIMIT,       // Angular limit constraint
    POSITION_LOCK      // Lock particle to fixed position
};

struct ConstraintParams {
    float compliance = 0.0f;        // Constraint softness (0 = rigid)
    float damping = 0.1f;          // Velocity damping
    float breaking_force = 1e6f;    // Force threshold for breaking
    bool enabled = true;           // Constraint active state
    bool bilateral = true;         // True for equality, false for inequality

    // Type-specific parameters
    float rest_length = 1.0f;      // For springs and distance constraints
    float stiffness = 1000.0f;     // Spring stiffness
    float max_length = 2.0f;       // For rope constraints
    float min_angle = -180.0f;     // For angle limits (degrees)
    float max_angle = 180.0f;      // For angle limits (degrees)
    float motor_speed = 0.0f;      // For motor constraints (rad/s)
    float motor_torque = 100.0f;   // Maximum motor torque
};

class Constraint {
public:
    ConstraintType type;
    ConstraintParams params;
    std::vector<int> particle_indices;  // Particles involved in constraint
    std::string name;                   // Human-readable name

    // Constraint state
    float current_force = 0.0f;
    float current_violation = 0.0f;
    bool is_broken = false;
    bool is_active = true;

    Constraint(ConstraintType t, const std::vector<int>& indices,
               const ConstraintParams& p = ConstraintParams{},
               const std::string& n = "");

    virtual ~Constraint() = default;

    // Core constraint interface
    virtual float evaluateConstraint(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) const = 0;

    virtual void computeJacobian(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        std::vector<std::vector<float>>& jacobian
    ) const = 0;

    virtual void applyConstraintForces(
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

    // Utility methods
    bool shouldBreak() const;
    void setEnabled(bool enabled) { is_active = enabled && !is_broken; }
    void reset() { is_broken = false; current_force = 0.0f; }

protected:
    // Helper for finite difference jacobian computation
    void computeNumericalJacobian(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        std::vector<std::vector<float>>& jacobian,
        float epsilon = 1e-6f
    ) const;
};

// Distance constraint: maintains fixed distance between two particles
class DistanceConstraint : public Constraint {
public:
    DistanceConstraint(int particle_a, int particle_b, float distance,
                      const ConstraintParams& params = ConstraintParams{});

    float evaluateConstraint(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) const override;

    void computeJacobian(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        std::vector<std::vector<float>>& jacobian
    ) const override;
};

// Spring constraint: applies spring force between particles
class SpringConstraint : public Constraint {
public:
    SpringConstraint(int particle_a, int particle_b, float rest_length, float stiffness,
                    const ConstraintParams& params = ConstraintParams{});

    float evaluateConstraint(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) const override;

    void computeJacobian(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        std::vector<std::vector<float>>& jacobian
    ) const override;
};

// Ball joint: constrains a point on one particle to a point on another
class BallJointConstraint : public Constraint {
public:
    BallJointConstraint(int particle_a, int particle_b,
                       const ConstraintParams& params = ConstraintParams{});

    float evaluateConstraint(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) const override;

    void computeJacobian(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        std::vector<std::vector<float>>& jacobian
    ) const override;
};

// Rope constraint: maximum distance constraint (inequality)
class RopeConstraint : public Constraint {
public:
    RopeConstraint(int particle_a, int particle_b, float max_length,
                  const ConstraintParams& params = ConstraintParams{});

    float evaluateConstraint(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) const override;

    void computeJacobian(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        std::vector<std::vector<float>>& jacobian
    ) const override;
};

// Position lock: fixes a particle to a world position
class PositionLockConstraint : public Constraint {
private:
    float target_x, target_y, target_z;

public:
    PositionLockConstraint(int particle_idx, float x, float y, float z,
                          const ConstraintParams& params = ConstraintParams{});

    void setTargetPosition(float x, float y, float z) {
        target_x = x; target_y = y; target_z = z;
    }

    float evaluateConstraint(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) const override;

    void computeJacobian(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        std::vector<std::vector<float>>& jacobian
    ) const override;
};

// Motor constraint: applies rotational force to maintain angular velocity
class MotorConstraint : public Constraint {
private:
    float current_angle = 0.0f;
    float target_angular_velocity = 0.0f;

public:
    MotorConstraint(int particle_a, int particle_b, float angular_velocity,
                   const ConstraintParams& params = ConstraintParams{});

    void setTargetAngularVelocity(float omega) { target_angular_velocity = omega; }
    float getCurrentAngle() const { return current_angle; }

    float evaluateConstraint(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    ) const override;

    void computeJacobian(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        std::vector<std::vector<float>>& jacobian
    ) const override;
};

struct ConstraintSolverParams {
    int max_iterations = 10;           // Maximum solver iterations
    float tolerance = 1e-6f;           // Convergence tolerance
    float relaxation = 1.0f;           // SOR relaxation parameter
    bool use_warm_starting = true;     // Reuse previous solution
    bool project_positions = true;     // Position-based constraint satisfaction
    bool project_velocities = true;    // Velocity-based constraint satisfaction
    float position_correction = 0.2f;  // Baumgarte stabilization parameter
    float velocity_correction = 0.1f;  // Velocity correction parameter
};

class ConstraintSolver {
private:
    std::vector<std::unique_ptr<Constraint>> constraints;
    ConstraintSolverParams solver_params;

    // Solver state
    std::vector<float> lambda;          // Lagrange multipliers
    std::vector<float> lambda_prev;     // Previous iteration multipliers
    std::vector<std::vector<float>> jacobian_matrix;  // Constraint jacobian
    std::vector<float> constraint_values;             // Current constraint violations

    // Performance tracking
    int last_iterations = 0;
    float last_residual = 0.0f;
    bool converged = false;

public:
    ConstraintSolver(const ConstraintSolverParams& params = ConstraintSolverParams{});
    ~ConstraintSolver() = default;

    // Constraint management
    void addConstraint(std::unique_ptr<Constraint> constraint);
    void removeConstraint(size_t index);
    void clearConstraints();

    // Constraint creation helpers
    void addDistanceConstraint(int particle_a, int particle_b, float distance,
                              const ConstraintParams& params = ConstraintParams{});
    void addSpringConstraint(int particle_a, int particle_b, float rest_length, float stiffness,
                           const ConstraintParams& params = ConstraintParams{});
    void addBallJoint(int particle_a, int particle_b,
                     const ConstraintParams& params = ConstraintParams{});
    void addRopeConstraint(int particle_a, int particle_b, float max_length,
                         const ConstraintParams& params = ConstraintParams{});
    void addPositionLock(int particle_idx, float x, float y, float z,
                        const ConstraintParams& params = ConstraintParams{});

    // Solver interface
    void solveConstraints(
        std::vector<float>& pos_x,
        std::vector<float>& pos_y,
        std::vector<float>& pos_z,
        std::vector<float>& vel_x,
        std::vector<float>& vel_y,
        std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt
    );

    void applyConstraintForces(
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

    // Configuration
    void setSolverParams(const ConstraintSolverParams& params) { solver_params = params; }
    const ConstraintSolverParams& getSolverParams() const { return solver_params; }

    // Status and debugging
    const std::vector<std::unique_ptr<Constraint>>& getConstraints() const { return constraints; }
    int getLastIterations() const { return last_iterations; }
    float getLastResidual() const { return last_residual; }
    bool hasConverged() const { return converged; }

    // Visualization helpers
    void getConstraintVisualizationData(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z,
        std::vector<std::pair<int, int>>& connections,        // Particle pairs
        std::vector<float>& connection_forces,               // Force magnitudes
        std::vector<int>& connection_types                   // Constraint types
    ) const;

private:
    void updateJacobian(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    );

    void evaluateConstraints(
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    );

    void solveLCP(const std::vector<float>& masses, float dt);  // Linear Complementarity Problem solver
    void projectPositions(
        std::vector<float>& pos_x,
        std::vector<float>& pos_y,
        std::vector<float>& pos_z,
        const std::vector<float>& masses
    );

    void projectVelocities(
        std::vector<float>& vel_x,
        std::vector<float>& vel_y,
        std::vector<float>& vel_z,
        const std::vector<float>& masses
    );

    void projectSingleConstraint(
        Constraint& constraint,
        std::vector<float>& pos_x,
        std::vector<float>& pos_y,
        std::vector<float>& pos_z,
        const std::vector<float>& masses
    );
};

// Utility functions for constraint setup
namespace ConstraintUtils {
    // Create a chain of particles connected by distance constraints
    std::vector<size_t> createChain(
        ConstraintSolver& solver,
        const std::vector<int>& particle_indices,
        float segment_length,
        const ConstraintParams& params = ConstraintParams{}
    );

    // Create a rope with multiple segments
    std::vector<size_t> createRope(
        ConstraintSolver& solver,
        const std::vector<int>& particle_indices,
        float max_segment_length,
        const ConstraintParams& params = ConstraintParams{}
    );

    // Create a cloth mesh with constraints
    std::vector<size_t> createCloth(
        ConstraintSolver& solver,
        int width, int height,
        const std::vector<int>& particle_grid,  // width*height particles
        float spacing,
        const ConstraintParams& params = ConstraintParams{}
    );

    // Create a rigid body from multiple particles
    std::vector<size_t> createRigidBody(
        ConstraintSolver& solver,
        const std::vector<int>& particle_indices,
        const ConstraintParams& params = ConstraintParams{}
    );

    // Calculate constraint satisfaction error
    float computeConstraintError(
        const ConstraintSolver& solver,
        const std::vector<float>& pos_x,
        const std::vector<float>& pos_y,
        const std::vector<float>& pos_z
    );
}

} // namespace physgrad