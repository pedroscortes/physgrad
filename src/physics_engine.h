/**
 * PhysGrad - Physics Engine Header
 *
 * Main physics engine class that coordinates all subsystems.
 */

#ifndef PHYSGRAD_PHYSICS_ENGINE_H
#define PHYSGRAD_PHYSICS_ENGINE_H

#include "common_types.h"
#include <vector>

namespace physgrad {

/**
 * Main physics engine class
 */
class PhysicsEngine {
public:
    PhysicsEngine();
    ~PhysicsEngine();

    // Initialization and cleanup
    bool initialize();
    void cleanup();

    // Particle management
    void addParticles(
        const std::vector<float3>& positions,
        const std::vector<float3>& velocities,
        const std::vector<float>& masses
    );
    void removeParticle(int index);

    // Property setters
    void setCharges(const std::vector<float>& charges);
    void setPositions(const std::vector<float3>& positions);
    void setVelocities(const std::vector<float3>& velocities);

    // Simulation
    void updateForces();
    void step(float dt);

    // Energy calculations
    float calculateTotalEnergy() const;

    // Getters
    std::vector<float3> getPositions() const;
    std::vector<float3> getVelocities() const;
    std::vector<float3> getForces() const;
    int getNumParticles() const;

    // Simulation settings
    void setBoundaryConditions(BoundaryType type, float3 bounds);
    void setIntegrationMethod(IntegrationMethod method);

private:
    // Particle data
    std::vector<float3> positions_;
    std::vector<float3> velocities_;
    std::vector<float3> forces_;
    std::vector<float> masses_;
    std::vector<float> charges_;

    // State
    int num_particles_;
    bool initialized_;

    // Simulation settings
    BoundaryType boundary_type_ = BoundaryType::OPEN;
    float3 boundary_bounds_ = {0.0f, 0.0f, 0.0f};
    IntegrationMethod integration_method_ = IntegrationMethod::VERLET;

    // Internal methods
    void applyBoundaryConditions();
};

} // namespace physgrad

#endif // PHYSGRAD_PHYSICS_ENGINE_H