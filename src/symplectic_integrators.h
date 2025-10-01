#pragma once

#include <vector>
#include <functional>
#include <memory>

namespace physgrad {

// Forward declarations
class SymplecticIntegratorBase;

enum class SymplecticScheme {
    SYMPLECTIC_EULER,          // First-order symplectic (position-velocity)
    VELOCITY_VERLET,           // Second-order symplectic (Störmer-Verlet)
    FOREST_RUTH,              // Fourth-order symplectic
    RUTH3,                    // Third-order symplectic
    RUTH4,                    // Fourth-order symplectic
    YOSHIDA4,                 // Fourth-order symplectic (Yoshida)
    YOSHIDA6,                 // Sixth-order symplectic (Yoshida)
    MCLACHLAN4,               // Fourth-order symplectic (McLachlan)
    CANDY_ROZMUS4,            // Fourth-order symplectic (Candy-Rozmus)
    BLANES_MOAN8,             // Eighth-order symplectic (Blanes-Moan)
    FROST_FSI4,               // Fourth-order Forward Symplectic Integrator (FROST-inspired)
    VARIATIONAL_GALERKIN2,    // Second-order variational integrator with Galerkin discretization
    VARIATIONAL_GALERKIN4,    // Fourth-order variational integrator with Galerkin discretization
    VARIATIONAL_LOBATTO3,     // Third-order Lobatto variational integrator
    VARIATIONAL_GAUSS4,       // Fourth-order Gauss variational integrator
    ADAPTIVE_VERLET,          // Adaptive Velocity Verlet with error control
    ADAPTIVE_YOSHIDA4,        // Adaptive 4th-order Yoshida with error control
    ADAPTIVE_GAUSS_LOBATTO,   // Adaptive Gauss-Lobatto with embedded error estimation
    ADAPTIVE_DORMAND_PRINCE   // Adaptive Dormand-Prince 5(4) with symplectic post-processing
};

struct SymplecticParams {
    float time_step = 0.01f;
    float energy_tolerance = 1e-6f;
    bool enable_energy_monitoring = true;
    bool enable_momentum_conservation = true;
    bool adaptive_time_stepping = false;
    float min_time_step = 1e-6f;
    float max_time_step = 0.1f;
    float safety_factor = 0.9f;
    int max_substeps = 10;

    // Advanced adaptive timestep control
    float relative_tolerance = 1e-6f;        // Relative error tolerance
    float absolute_tolerance = 1e-8f;        // Absolute error tolerance
    float step_increase_factor = 2.0f;       // Maximum step increase
    float step_decrease_factor = 0.5f;       // Step decrease on rejection
    int max_step_rejections = 5;             // Maximum consecutive rejections
    bool enable_step_size_control = true;    // Enable PI step size controller
    float proportional_gain = 0.7f;          // PI controller proportional gain
    float integral_gain = -0.4f;             // PI controller integral gain
};

struct ConservationQuantities {
    float total_energy = 0.0f;
    float kinetic_energy = 0.0f;
    float potential_energy = 0.0f;
    float linear_momentum[3] = {0.0f, 0.0f, 0.0f};
    float angular_momentum[3] = {0.0f, 0.0f, 0.0f};
    float energy_drift = 0.0f;
    float momentum_drift = 0.0f;
    bool conservation_violated = false;
};

// Force function type: (positions, velocities, masses, forces_out, time)
using ForceFunction = std::function<void(
    const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
    std::vector<float>&, std::vector<float>&, std::vector<float>&,
    const std::vector<float>&, float
)>;

// Force gradient function type: (positions, masses, force_gradients_out)
// force_gradients_out[i][j] = d(F_i)/d(r_j) - gradient of force on particle i w.r.t. position of particle j
using ForceGradientFunction = std::function<void(
    const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&,
    std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&
)>;

// Potential energy function type: (positions, masses) -> energy
using PotentialFunction = std::function<float(
    const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&
)>;

class SymplecticIntegratorBase {
protected:
    SymplecticParams params;
    ForceFunction force_function;
    PotentialFunction potential_function;

    // Conservation tracking
    ConservationQuantities initial_quantities;
    ConservationQuantities current_quantities;
    std::vector<float> energy_history;
    std::vector<float> momentum_history;

    // Performance tracking
    int total_steps = 0;
    int rejected_steps = 0;
    float average_step_size = 0.0f;

public:
    SymplecticIntegratorBase(const SymplecticParams& p = SymplecticParams{});
    virtual ~SymplecticIntegratorBase() = default;

    // Core integration interface
    virtual float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) = 0;

    // Configuration
    void setForceFunction(const ForceFunction& func) { force_function = func; }
    void setPotentialFunction(const PotentialFunction& func) { potential_function = func; }
    void setParameters(const SymplecticParams& p) { params = p; }
    const SymplecticParams& getParameters() const { return params; }

    // Conservation monitoring
    void computeConservationQuantities(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        const std::vector<float>& masses, float time = 0.0f
    );

    void initializeConservationTracking(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        const std::vector<float>& masses
    );

    // Getters for conservation data
    const ConservationQuantities& getCurrentQuantities() const { return current_quantities; }
    const ConservationQuantities& getInitialQuantities() const { return initial_quantities; }
    const std::vector<float>& getEnergyHistory() const { return energy_history; }
    const std::vector<float>& getMomentumHistory() const { return momentum_history; }

    // Performance metrics
    int getTotalSteps() const { return total_steps; }
    int getRejectedSteps() const { return rejected_steps; }
    float getAverageStepSize() const { return average_step_size; }
    float getAcceptanceRate() const {
        return total_steps > 0 ? 1.0f - static_cast<float>(rejected_steps) / total_steps : 1.0f;
    }

protected:
    // Helper methods for derived classes
    void updateStatistics(float actual_dt);
    bool checkConservation(float energy_change, float momentum_change);
    float adaptiveStepSize(float current_dt, float error_estimate);

    // Common symplectic operations
    void velocityKick(
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& force_x, const std::vector<float>& force_y, const std::vector<float>& force_z,
        const std::vector<float>& masses, float dt
    );

    void positionDrift(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        float dt
    );
};

// Symplectic Euler (1st order, exact for harmonic oscillator)
class SymplecticEuler : public SymplecticIntegratorBase {
public:
    SymplecticEuler(const SymplecticParams& params = SymplecticParams{});

    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) override;
};

// Velocity Verlet (2nd order, most commonly used)
class VelocityVerlet : public SymplecticIntegratorBase {
private:
    std::vector<float> force_x_old, force_y_old, force_z_old;

public:
    VelocityVerlet(const SymplecticParams& params = SymplecticParams{});

    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) override;
};

// Forest-Ruth (4th order)
class ForestRuth : public SymplecticIntegratorBase {
private:
    // Correct Forest-Ruth coefficients (Ruth 1983)
    static constexpr float theta = 1.351207191959657f;
    static constexpr float chi = -1.702414383919315f;
    // Note: 2*theta + 2*chi = -0.702414383919316, so velocity coefficients need normalization

public:
    ForestRuth(const SymplecticParams& params = SymplecticParams{});

    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) override;
};

// Yoshida 4th order
class Yoshida4 : public SymplecticIntegratorBase {
private:
    static constexpr float w0 = -1.702414383919315f;
    static constexpr float w1 = 1.351207191959657f;
    static constexpr float c1 = w1 / 2.0f;
    static constexpr float c2 = (w0 + w1) / 2.0f;
    static constexpr float c3 = c2;
    static constexpr float c4 = c1;
    static constexpr float d1 = w1;
    static constexpr float d2 = w0;
    static constexpr float d3 = w1;

public:
    Yoshida4(const SymplecticParams& params = SymplecticParams{});

    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) override;
};

// Blanes-Moan 8th order (very high precision)
class BlanesMoan8 : public SymplecticIntegratorBase {
private:
    static constexpr float a1 = 0.74167036435061295344822780f;
    static constexpr float a2 = -0.40910082580003159399730010f;
    static constexpr float a3 = 0.19075471029623837995387626f;
    static constexpr float a4 = -0.57386247111608226665638773f;
    static constexpr float a5 = 0.29906418130365592384446354f;
    static constexpr float a6 = 0.33462491824529818378495798f;
    static constexpr float a7 = 0.31529309239676659663205666f;
    static constexpr float a8 = -0.79688793935291635401978884f;

    static constexpr float b1 = 1.48334072870122590689645560f;
    static constexpr float b2 = -1.23228904932781835851441844f;
    static constexpr float b3 = 0.85550637535681116028279960f;
    static constexpr float b4 = -1.52474487375142515537766467f;
    static constexpr float b5 = 1.15800838465728823740949655f;
    static constexpr float b6 = 0.25963797190851120515687623f;
    static constexpr float b7 = -1.38053697411755079232251750f;

public:
    BlanesMoan8(const SymplecticParams& params = SymplecticParams{});

    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) override;
};

// FROST Forward Symplectic Integrator (4th order)
// This integrator achieves 4th-order accuracy using only positive timesteps
// by computing force gradients. It's inspired by the FROST N-body code.
class FrostForwardSymplectic4 : public SymplecticIntegratorBase {
private:
    // Force gradient storage
    std::vector<std::vector<float>> force_grad_xx, force_grad_xy, force_grad_xz;
    std::vector<std::vector<float>> force_grad_yx, force_grad_yy, force_grad_yz;
    std::vector<std::vector<float>> force_grad_zx, force_grad_zy, force_grad_zz;

    // Temporary storage for intermediate steps
    std::vector<float> temp_pos_x, temp_pos_y, temp_pos_z;
    std::vector<float> temp_vel_x, temp_vel_y, temp_vel_z;
    std::vector<float> temp_force_x, temp_force_y, temp_force_z;

    // Forest-Ruth coefficients for 4th-order accuracy
    static constexpr float theta = 1.351207191959657f;
    static constexpr float chi = -1.702414383919315f;

    ForceGradientFunction force_gradient_function;

    void resizeBuffers(size_t num_particles);
    void computeForceGradients(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& masses
    );

public:
    FrostForwardSymplectic4(const SymplecticParams& params = SymplecticParams{});

    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) override;

    // Set the force gradient function for higher-order accuracy
    void setForceGradientFunction(const ForceGradientFunction& func) {
        force_gradient_function = func;
    }

    // Check if force gradients are available
    bool hasForceGradients() const {
        return force_gradient_function != nullptr;
    }
};

// Variational Integrator with Galerkin discretization (2nd order)
// Uses discrete variational principles for structure-preserving integration
class VariationalGalerkin2 : public SymplecticIntegratorBase {
private:
    // Lagrangian function type: L(q, q_dot, t)
    using LagrangianFunction = std::function<float(
        const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,  // positions
        const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,  // velocities
        const std::vector<float>&,  // masses
        float  // time
    )>;

    // Discrete Lagrangian for Galerkin method
    LagrangianFunction lagrangian_function;

    // Temporary storage for discrete Euler-Lagrange computations
    std::vector<float> temp_pos_x, temp_pos_y, temp_pos_z;
    std::vector<float> temp_vel_x, temp_vel_y, temp_vel_z;
    std::vector<float> q_prev_x, q_prev_y, q_prev_z;

    // Discrete variational derivative computations
    void computeDiscreteEulerLagrange(
        const std::vector<float>& q_prev_x, const std::vector<float>& q_prev_y, const std::vector<float>& q_prev_z,
        const std::vector<float>& q_curr_x, const std::vector<float>& q_curr_y, const std::vector<float>& q_curr_z,
        const std::vector<float>& q_next_x, const std::vector<float>& q_next_y, const std::vector<float>& q_next_z,
        const std::vector<float>& masses, float dt, float time
    );

public:
    VariationalGalerkin2(const SymplecticParams& params = SymplecticParams{});

    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) override;

    void setLagrangianFunction(const LagrangianFunction& func) {
        lagrangian_function = func;
    }
};

// Variational Integrator with Galerkin discretization (4th order)
// Higher-order Galerkin formulation for improved accuracy
class VariationalGalerkin4 : public SymplecticIntegratorBase {
private:
    using LagrangianFunction = std::function<float(
        const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
        const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
        const std::vector<float>&, float
    )>;

    LagrangianFunction lagrangian_function;

    // Higher-order Galerkin requires multiple time points
    std::vector<float> q_minus2_x, q_minus2_y, q_minus2_z;
    std::vector<float> q_minus1_x, q_minus1_y, q_minus1_z;
    std::vector<float> q_plus1_x, q_plus1_y, q_plus1_z;
    std::vector<float> q_plus2_x, q_plus2_y, q_plus2_z;

    // Higher-order discrete Euler-Lagrange solver
    void solveHigherOrderEulerLagrange(
        std::vector<float>& q_next_x, std::vector<float>& q_next_y, std::vector<float>& q_next_z,
        const std::vector<float>& masses, float dt, float time
    );

public:
    VariationalGalerkin4(const SymplecticParams& params = SymplecticParams{});

    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) override;

    void setLagrangianFunction(const LagrangianFunction& func) {
        lagrangian_function = func;
    }
};

// Lobatto variational integrator (3rd order)
// Uses Lobatto quadrature for discrete Lagrangian
class VariationalLobatto3 : public SymplecticIntegratorBase {
private:
    using LagrangianFunction = std::function<float(
        const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
        const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
        const std::vector<float>&, float
    )>;

    LagrangianFunction lagrangian_function;

    // Lobatto quadrature points and weights
    static constexpr float lobatto_points[3] = {0.0f, 0.5f, 1.0f};
    static constexpr float lobatto_weights[3] = {1.0f/6.0f, 4.0f/6.0f, 1.0f/6.0f};

    std::vector<float> temp_q_x, temp_q_y, temp_q_z;
    std::vector<float> temp_v_x, temp_v_y, temp_v_z;

public:
    VariationalLobatto3(const SymplecticParams& params = SymplecticParams{});

    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) override;

    void setLagrangianFunction(const LagrangianFunction& func) {
        lagrangian_function = func;
    }
};

// Gauss variational integrator (4th order)
// Uses Gauss quadrature for highly accurate discrete Lagrangian
class VariationalGauss4 : public SymplecticIntegratorBase {
private:
    using LagrangianFunction = std::function<float(
        const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
        const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
        const std::vector<float>&, float
    )>;

    LagrangianFunction lagrangian_function;

    // Gauss quadrature points and weights for 4th order
    static constexpr float gauss_points[2] = {-0.577350269189626f, 0.577350269189626f};
    static constexpr float gauss_weights[2] = {1.0f, 1.0f};

    std::vector<float> temp_q_x, temp_q_y, temp_q_z;
    std::vector<float> temp_v_x, temp_v_y, temp_v_z;

public:
    VariationalGauss4(const SymplecticParams& params = SymplecticParams{});

    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) override;

    void setLagrangianFunction(const LagrangianFunction& func) {
        lagrangian_function = func;
    }
};

// Base class for adaptive timestep integrators
class AdaptiveSymplecticIntegratorBase : public SymplecticIntegratorBase {
protected:
    // Error estimation and step control
    float current_step_size;
    float previous_error;
    int consecutive_rejections;
    bool step_accepted;

    // Error estimation using embedded methods or step doubling
    virtual float estimateLocalError(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float time
    ) = 0;

    // PI step size controller
    float computeNewStepSize(float error, float dt);

    // Step acceptance criteria
    bool acceptStep(float error);

public:
    AdaptiveSymplecticIntegratorBase(const SymplecticParams& params = SymplecticParams{});

    // Override integrateStep to include adaptive logic
    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time = 0.0f
    ) override final;

    // Pure virtual method for actual integration step
    virtual float doIntegrationStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time
    ) = 0;

    // Get the order of accuracy for the integrator
    virtual int getOrder() const = 0;
};

// Adaptive Velocity Verlet with step doubling error estimation
class AdaptiveVerlet : public AdaptiveSymplecticIntegratorBase {
private:
    std::vector<float> temp_pos_x, temp_pos_y, temp_pos_z;
    std::vector<float> temp_vel_x, temp_vel_y, temp_vel_z;

protected:
    float estimateLocalError(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float time
    ) override;

public:
    AdaptiveVerlet(const SymplecticParams& params = SymplecticParams{});

    float doIntegrationStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time
    ) override;

    int getOrder() const override;
};

// Adaptive 4th-order Yoshida with embedded error estimation
class AdaptiveYoshida4 : public AdaptiveSymplecticIntegratorBase {
private:
    std::vector<float> temp_pos_x, temp_pos_y, temp_pos_z;
    std::vector<float> temp_vel_x, temp_vel_y, temp_vel_z;

    // 3rd-order embedded method for error estimation
    void yoshida3Step(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float time
    );

protected:
    float estimateLocalError(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float time
    ) override;

public:
    AdaptiveYoshida4(const SymplecticParams& params = SymplecticParams{});

    float doIntegrationStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time
    ) override;

    int getOrder() const override;
};

// Adaptive Gauss-Lobatto with embedded error estimation
class AdaptiveGaussLobatto : public AdaptiveSymplecticIntegratorBase {
private:
    std::vector<float> temp_pos_x, temp_pos_y, temp_pos_z;
    std::vector<float> temp_vel_x, temp_vel_y, temp_vel_z;

    // Lower-order embedded method
    void lobatto2Step(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float time
    );

protected:
    float estimateLocalError(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float time
    ) override;

public:
    AdaptiveGaussLobatto(const SymplecticParams& params = SymplecticParams{});

    float doIntegrationStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time
    ) override;

    int getOrder() const override;
};

// Adaptive Dormand-Prince with symplectic post-processing
class AdaptiveDormandPrince : public AdaptiveSymplecticIntegratorBase {
private:
    std::vector<float> temp_pos_x, temp_pos_y, temp_pos_z;
    std::vector<float> temp_vel_x, temp_vel_y, temp_vel_z;
    std::vector<float> k1_x, k1_y, k1_z, k2_x, k2_y, k2_z;
    std::vector<float> k3_x, k3_y, k3_z, k4_x, k4_y, k4_z;
    std::vector<float> k5_x, k5_y, k5_z, k6_x, k6_y, k6_z;
    std::vector<float> k7_x, k7_y, k7_z;

    // Symplectic post-processing to preserve structure
    void applySymplecticCorrection(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float time
    );

protected:
    float estimateLocalError(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float time
    ) override;

public:
    AdaptiveDormandPrince(const SymplecticParams& params = SymplecticParams{});

    float doIntegrationStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float dt, float time
    ) override;

    int getOrder() const override;
};

// Factory for creating integrators
class SymplecticIntegratorFactory {
public:
    static std::unique_ptr<SymplecticIntegratorBase> create(
        SymplecticScheme scheme,
        const SymplecticParams& params = SymplecticParams{}
    );

    static std::string getSchemeDescription(SymplecticScheme scheme);
    static int getSchemeOrder(SymplecticScheme scheme);
    static bool isSchemeAdaptive(SymplecticScheme scheme);
};

// Utility functions for setting up common force functions
namespace SymplecticUtils {
    // Create force function for gravitational N-body simulation
    ForceFunction createGravitationalForce(float G = 1.0f, float softening = 0.01f);

    // Create force gradient function for gravitational N-body simulation
    ForceGradientFunction createGravitationalForceGradient(float G = 1.0f, float softening = 0.01f);

    // Enhanced force gradient functions for comprehensive physics systems
    ForceGradientFunction createHarmonicOscillatorForceGradient(float k = 1.0f, const float center[3] = nullptr);
    ForceGradientFunction createSpringSystemForceGradient(
        const std::vector<std::pair<int, int>>& connections,
        const std::vector<float>& spring_constants,
        const std::vector<float>& rest_lengths
    );
    ForceGradientFunction createLennardJonesForceGradient(float epsilon = 1.0f, float sigma = 1.0f);
    ForceGradientFunction createCoulombForceGradient(float k_coulomb = 8.9875517923e9f);

    // Create force function for harmonic oscillator
    ForceFunction createHarmonicOscillatorForce(float k = 1.0f, const float center[3] = nullptr);

    // Create force function for spring system
    ForceFunction createSpringSystemForce(
        const std::vector<std::pair<int, int>>& connections,
        const std::vector<float>& spring_constants,
        const std::vector<float>& rest_lengths
    );

    // Create potential function for gravitational system
    PotentialFunction createGravitationalPotential(float G = 1.0f, float softening = 0.01f);

    // Create potential function for harmonic oscillator
    PotentialFunction createHarmonicOscillatorPotential(float k = 1.0f, const float center[3] = nullptr);

    // Energy and conservation analysis utilities
    float computeKineticEnergy(
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        const std::vector<float>& masses
    );

    void computeLinearMomentum(
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        const std::vector<float>& masses, float momentum[3]
    );

    void computeAngularMomentum(
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
        const std::vector<float>& masses, float angular_momentum[3], const float center[3] = nullptr
    );

    // Benchmark and validation functions
    void runConvergenceTest(
        SymplecticScheme scheme,
        const std::vector<float>& initial_positions_x,
        const std::vector<float>& initial_positions_y,
        const std::vector<float>& initial_positions_z,
        const std::vector<float>& initial_velocities_x,
        const std::vector<float>& initial_velocities_y,
        const std::vector<float>& initial_velocities_z,
        const std::vector<float>& masses,
        ForceFunction force_func,
        PotentialFunction potential_func,
        float simulation_time,
        int num_step_sizes = 5
    );

    void compareIntegrators(
        const std::vector<SymplecticScheme>& schemes,
        const std::vector<float>& initial_positions_x,
        const std::vector<float>& initial_positions_y,
        const std::vector<float>& initial_positions_z,
        const std::vector<float>& initial_velocities_x,
        const std::vector<float>& initial_velocities_y,
        const std::vector<float>& initial_velocities_z,
        const std::vector<float>& masses,
        ForceFunction force_func,
        PotentialFunction potential_func,
        float simulation_time,
        float time_step
    );
}

} // namespace physgrad