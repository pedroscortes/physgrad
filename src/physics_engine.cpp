/**
 * PhysGrad - Physics Engine Implementation
 *
 * Main physics engine class that coordinates all subsystems.
 */

#include "physics_engine.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace physgrad {

PhysicsEngine::PhysicsEngine()
    : num_particles_(0)
    , initialized_(false) {
}

PhysicsEngine::~PhysicsEngine() {
    cleanup();
}

bool PhysicsEngine::initialize() {
    if (initialized_) {
        return true;
    }

    // Initialize subsystems
    std::cout << "Initializing PhysGrad Physics Engine..." << std::endl;

    // Clear all data structures
    positions_.clear();
    velocities_.clear();
    forces_.clear();
    masses_.clear();
    charges_.clear();

    num_particles_ = 0;
    initialized_ = true;

    std::cout << "Physics engine initialized successfully." << std::endl;
    return true;
}

void PhysicsEngine::cleanup() {
    if (!initialized_) {
        return;
    }

    std::cout << "Cleaning up Physics Engine..." << std::endl;

    // Clear all data
    positions_.clear();
    velocities_.clear();
    forces_.clear();
    masses_.clear();
    charges_.clear();

    num_particles_ = 0;
    initialized_ = false;

    std::cout << "Physics engine cleanup complete." << std::endl;
}

void PhysicsEngine::addParticles(
    const std::vector<float3>& positions,
    const std::vector<float3>& velocities,
    const std::vector<float>& masses
) {
    if (!initialized_) {
        std::cerr << "Error: Physics engine not initialized!" << std::endl;
        return;
    }

    if (positions.size() != velocities.size() || positions.size() != masses.size()) {
        std::cerr << "Error: Mismatched array sizes in addParticles!" << std::endl;
        return;
    }

    // Add particles to existing arrays
    positions_.insert(positions_.end(), positions.begin(), positions.end());
    velocities_.insert(velocities_.end(), velocities.begin(), velocities.end());
    masses_.insert(masses_.end(), masses.begin(), masses.end());

    // Add default charges (neutral)
    charges_.resize(positions_.size(), 0.0f);

    // Initialize forces to zero
    forces_.resize(positions_.size(), {0.0f, 0.0f, 0.0f});

    num_particles_ = static_cast<int>(positions_.size());

    std::cout << "Added " << positions.size() << " particles. Total: " << num_particles_ << std::endl;
}

void PhysicsEngine::removeParticle(int index) {
    if (!initialized_ || index < 0 || index >= num_particles_) {
        std::cerr << "Error: Invalid particle index in removeParticle!" << std::endl;
        return;
    }

    positions_.erase(positions_.begin() + index);
    velocities_.erase(velocities_.begin() + index);
    forces_.erase(forces_.begin() + index);
    masses_.erase(masses_.begin() + index);
    charges_.erase(charges_.begin() + index);

    num_particles_--;

    std::cout << "Removed particle " << index << ". Total: " << num_particles_ << std::endl;
}

void PhysicsEngine::setCharges(const std::vector<float>& charges) {
    if (!initialized_) {
        std::cerr << "Error: Physics engine not initialized!" << std::endl;
        return;
    }

    if (charges.size() != static_cast<size_t>(num_particles_)) {
        std::cerr << "Error: Charge array size mismatch!" << std::endl;
        return;
    }

    charges_ = charges;
}

void PhysicsEngine::setPositions(const std::vector<float3>& positions) {
    if (!initialized_) {
        std::cerr << "Error: Physics engine not initialized!" << std::endl;
        return;
    }

    if (positions.size() != static_cast<size_t>(num_particles_)) {
        std::cerr << "Error: Position array size mismatch!" << std::endl;
        return;
    }

    positions_ = positions;
}

void PhysicsEngine::setVelocities(const std::vector<float3>& velocities) {
    if (!initialized_) {
        std::cerr << "Error: Physics engine not initialized!" << std::endl;
        return;
    }

    if (velocities.size() != static_cast<size_t>(num_particles_)) {
        std::cerr << "Error: Velocity array size mismatch!" << std::endl;
        return;
    }

    velocities_ = velocities;
}

void PhysicsEngine::updateForces() {
    if (!initialized_ || num_particles_ == 0) {
        return;
    }

    // Clear forces
    std::fill(forces_.begin(), forces_.end(), float3{0.0f, 0.0f, 0.0f});

    // Calculate electrostatic forces
    for (int i = 0; i < num_particles_; ++i) {
        for (int j = i + 1; j < num_particles_; ++j) {
            float3 r_ij = {
                positions_[i].x - positions_[j].x,
                positions_[i].y - positions_[j].y,
                positions_[i].z - positions_[j].z
            };

            float r2 = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z;
            float r = std::sqrt(r2);

            if (r > 1e-10f) {
                // Coulomb force
                const float k_e = 8.9875517923e9f; // N⋅m²/C²
                float force_magnitude = k_e * charges_[i] * charges_[j] / r2;

                float3 force_dir = {r_ij.x / r, r_ij.y / r, r_ij.z / r};

                float3 force = {
                    force_magnitude * force_dir.x,
                    force_magnitude * force_dir.y,
                    force_magnitude * force_dir.z
                };

                // Apply forces (Newton's 3rd law)
                forces_[i].x += force.x;
                forces_[i].y += force.y;
                forces_[i].z += force.z;

                forces_[j].x -= force.x;
                forces_[j].y -= force.y;
                forces_[j].z -= force.z;
            }
        }
    }
}

void PhysicsEngine::step(float dt) {
    if (!initialized_ || num_particles_ == 0) {
        return;
    }

    // Update forces
    updateForces();

    // Integrate using simple Euler method
    for (int i = 0; i < num_particles_; ++i) {
        if (masses_[i] > 1e-10f) {
            float3 acceleration = {
                forces_[i].x / masses_[i],
                forces_[i].y / masses_[i],
                forces_[i].z / masses_[i]
            };

            // Update velocity
            velocities_[i].x += acceleration.x * dt;
            velocities_[i].y += acceleration.y * dt;
            velocities_[i].z += acceleration.z * dt;

            // Update position
            positions_[i].x += velocities_[i].x * dt;
            positions_[i].y += velocities_[i].y * dt;
            positions_[i].z += velocities_[i].z * dt;
        }
    }
}

float PhysicsEngine::calculateTotalEnergy() const {
    if (!initialized_ || num_particles_ == 0) {
        return 0.0f;
    }

    float kinetic_energy = 0.0f;
    float potential_energy = 0.0f;

    // Calculate kinetic energy
    for (int i = 0; i < num_particles_; ++i) {
        float v2 = velocities_[i].x * velocities_[i].x +
                   velocities_[i].y * velocities_[i].y +
                   velocities_[i].z * velocities_[i].z;
        kinetic_energy += 0.5f * masses_[i] * v2;
    }

    // Calculate potential energy
    const float k_e = 8.9875517923e9f;
    for (int i = 0; i < num_particles_; ++i) {
        for (int j = i + 1; j < num_particles_; ++j) {
            float3 r_ij = {
                positions_[i].x - positions_[j].x,
                positions_[i].y - positions_[j].y,
                positions_[i].z - positions_[j].z
            };

            float r = std::sqrt(r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z);

            if (r > 1e-10f) {
                potential_energy += k_e * charges_[i] * charges_[j] / r;
            }
        }
    }

    return kinetic_energy + potential_energy;
}

std::vector<float3> PhysicsEngine::getPositions() const {
    return positions_;
}

std::vector<float3> PhysicsEngine::getVelocities() const {
    return velocities_;
}

std::vector<float3> PhysicsEngine::getForces() const {
    return forces_;
}

int PhysicsEngine::getNumParticles() const {
    return num_particles_;
}

void PhysicsEngine::setBoundaryConditions(BoundaryType type, float3 bounds) {
    boundary_type_ = type;
    boundary_bounds_ = bounds;
}

void PhysicsEngine::setIntegrationMethod(IntegrationMethod method) {
    integration_method_ = method;
}

} // namespace physgrad