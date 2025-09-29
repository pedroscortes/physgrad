/**
 * PhysGrad - Multi-Scale Physics Header
 *
 * System for coupling different physical scales and theories.
 */

#ifndef PHYSGRAD_MULTI_SCALE_PHYSICS_H
#define PHYSGRAD_MULTI_SCALE_PHYSICS_H

#include "common_types.h"
#include <vector>
#include <memory>

namespace physgrad {

/**
 * Multi-scale physics coupling system
 */
class MultiScalePhysics {
public:
    // Constructor and destructor
    MultiScalePhysics();
    ~MultiScalePhysics();

    // Initialization and cleanup
    bool initialize();
    void cleanup();

    // Scale coupling methods
    void coupleQuantumClassical(
        const std::vector<float>& quantum_states,
        std::vector<float3>& classical_positions,
        std::vector<float3>& classical_forces
    );

    void coupleMolecularContinuum(
        const std::vector<float3>& molecular_positions,
        const std::vector<float3>& molecular_velocities,
        std::vector<float>& continuum_density,
        std::vector<float3>& continuum_velocity
    );

    void updateMultiScale(float dt);

    // Configuration
    void setQuantumRegion(const float3& min_bounds, const float3& max_bounds);
    void setClassicalRegion(const float3& min_bounds, const float3& max_bounds);
    void setCouplingStrength(float strength);

    // Query methods
    bool isInQuantumRegion(const float3& position) const;
    bool isInClassicalRegion(const float3& position) const;
    bool isInCouplingRegion(const float3& position) const;

private:
    // Region definitions
    float3 quantum_min_bounds_;
    float3 quantum_max_bounds_;
    float3 classical_min_bounds_;
    float3 classical_max_bounds_;

    // Coupling parameters
    float coupling_strength_;
    bool initialized_;

    // Internal methods
    void updateQuantumClassicalCoupling(float dt);
    void updateMolecularContinuumCoupling(float dt);
    void applySmoothingAtInterfaces();
};

// Utility functions for multi-scale operations
float interpolateQuantumClassical(float quantum_value, float classical_value, float weight);
float3 projectQuantumToClassical(const std::vector<float>& quantum_state, const float3& position);
std::vector<float> projectClassicalToQuantum(const std::vector<float3>& classical_positions,
                                           const std::vector<float3>& classical_velocities);

} // namespace physgrad

#endif // PHYSGRAD_MULTI_SCALE_PHYSICS_H