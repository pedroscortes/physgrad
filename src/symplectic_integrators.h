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
    BLANES_MOAN8              // Eighth-order symplectic (Blanes-Moan)
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
    static constexpr float theta = 1.351207191959657f;
    static constexpr float chi = -1.702414383919315f;

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