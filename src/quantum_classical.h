#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <iostream>
#include <numeric>
#include <limits>
#include <complex>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace physgrad {
namespace quantum {

template<typename T>
using Complex = std::complex<T>;

template<typename T>
struct Vec3 {
    T x, y, z;

    CUDA_HOST_DEVICE Vec3() : x(0), y(0), z(0) {}
    CUDA_HOST_DEVICE Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    CUDA_HOST_DEVICE Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    CUDA_HOST_DEVICE Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    CUDA_HOST_DEVICE Vec3 operator*(T scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    CUDA_HOST_DEVICE T dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    CUDA_HOST_DEVICE T norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    CUDA_HOST_DEVICE T distance(const Vec3& other) const {
        return (*this - other).norm();
    }
};

template<typename T>
class WaveFunction {
private:
    std::vector<Complex<T>> coefficients_;
    size_t n_states_;
    T normalization_;

public:
    WaveFunction(size_t n_states) : n_states_(n_states), normalization_(1.0) {
        coefficients_.resize(n_states, Complex<T>(0, 0));
        // Initialize ground state
        if (n_states > 0) {
            coefficients_[0] = Complex<T>(1, 0);
        }
    }

    void setCoefficient(size_t state, const Complex<T>& value) {
        if (state < n_states_) {
            coefficients_[state] = value;
        }
    }

    Complex<T> getCoefficient(size_t state) const {
        return (state < n_states_) ? coefficients_[state] : Complex<T>(0, 0);
    }

    void normalize() {
        T norm_squared = 0;
        for (const auto& coeff : coefficients_) {
            norm_squared += std::norm(coeff);
        }

        if (norm_squared > 1e-12) {
            T norm = std::sqrt(norm_squared);
            for (auto& coeff : coefficients_) {
                coeff /= norm;
            }
            normalization_ = norm;
        }
    }

    T getProbability(size_t state) const {
        if (state >= n_states_) return 0;
        return std::norm(coefficients_[state]);
    }

    T getTotalProbability() const {
        T total = 0;
        for (const auto& coeff : coefficients_) {
            total += std::norm(coeff);
        }
        return total;
    }

    size_t measureState() const {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(0, 1);

        T random_value = dist(gen);
        T cumulative_prob = 0;

        for (size_t i = 0; i < n_states_; ++i) {
            cumulative_prob += getProbability(i);
            if (random_value <= cumulative_prob) {
                return i;
            }
        }

        return n_states_ - 1; // Fallback to last state
    }

    void collapseToState(size_t state) {
        if (state < n_states_) {
            std::fill(coefficients_.begin(), coefficients_.end(), Complex<T>(0, 0));
            coefficients_[state] = Complex<T>(1, 0);
        }
    }

    T getExpectationValue(const std::vector<std::vector<Complex<T>>>& operator_matrix) const {
        if (operator_matrix.size() != n_states_ || operator_matrix[0].size() != n_states_) {
            return 0;
        }

        Complex<T> expectation(0, 0);
        for (size_t i = 0; i < n_states_; ++i) {
            for (size_t j = 0; j < n_states_; ++j) {
                expectation += std::conj(coefficients_[i]) * operator_matrix[i][j] * coefficients_[j];
            }
        }

        return expectation.real();
    }

    size_t getNumStates() const { return n_states_; }
    T getNormalization() const { return normalization_; }

    void applyOperator(const std::vector<std::vector<Complex<T>>>& operator_matrix) {
        if (operator_matrix.size() != n_states_ || operator_matrix[0].size() != n_states_) {
            return;
        }

        std::vector<Complex<T>> new_coefficients(n_states_, Complex<T>(0, 0));
        for (size_t i = 0; i < n_states_; ++i) {
            for (size_t j = 0; j < n_states_; ++j) {
                new_coefficients[i] += operator_matrix[i][j] * coefficients_[j];
            }
        }

        coefficients_ = new_coefficients;
    }
};

template<typename T>
class HamiltonianOperator {
private:
    std::vector<std::vector<Complex<T>>> matrix_;
    size_t n_states_;
    std::vector<T> eigenvalues_;
    std::vector<std::vector<Complex<T>>> eigenvectors_;

public:
    HamiltonianOperator(size_t n_states) : n_states_(n_states) {
        matrix_.resize(n_states, std::vector<Complex<T>>(n_states, Complex<T>(0, 0)));
    }

    void setElement(size_t i, size_t j, const Complex<T>& value) {
        if (i < n_states_ && j < n_states_) {
            matrix_[i][j] = value;
        }
    }

    Complex<T> getElement(size_t i, size_t j) const {
        if (i < n_states_ && j < n_states_) {
            return matrix_[i][j];
        }
        return Complex<T>(0, 0);
    }

    void buildHarmonicOscillator(T omega, T hbar = 1.0) {
        // Build harmonic oscillator Hamiltonian: H = ω(a†a + 1/2)
        for (size_t n = 0; n < n_states_; ++n) {
            // Energy eigenvalues: E_n = ω(n + 1/2)
            matrix_[n][n] = Complex<T>(hbar * omega * (n + 0.5), 0);
        }
    }

    void buildSpinHamiltonian(T magnetic_field, T gyromagnetic_ratio, const Vec3<T>& field_direction) {
        // Build spin-1/2 Hamiltonian in magnetic field
        if (n_states_ != 2) return;

        T omega_larmor = gyromagnetic_ratio * magnetic_field;

        // Pauli matrices scaled by magnetic field
        matrix_[0][0] = Complex<T>(omega_larmor * field_direction.z / 2.0, 0);
        matrix_[0][1] = Complex<T>(omega_larmor * field_direction.x / 2.0, -omega_larmor * field_direction.y / 2.0);
        matrix_[1][0] = Complex<T>(omega_larmor * field_direction.x / 2.0, omega_larmor * field_direction.y / 2.0);
        matrix_[1][1] = Complex<T>(-omega_larmor * field_direction.z / 2.0, 0);
    }

    void buildTunnelCoupling(T tunnel_strength) {
        // Build tight-binding model with nearest neighbor coupling
        for (size_t i = 0; i < n_states_ - 1; ++i) {
            matrix_[i][i + 1] = Complex<T>(-tunnel_strength, 0);
            matrix_[i + 1][i] = Complex<T>(-tunnel_strength, 0);
        }
    }

    void timeEvolution(WaveFunction<T>& psi, T dt, T hbar = 1.0) const {
        // Apply time evolution operator: exp(-iHt/ħ)
        // Using simple first-order approximation for small dt

        std::vector<std::vector<Complex<T>>> evolution_matrix(n_states_,
            std::vector<Complex<T>>(n_states_, Complex<T>(0, 0)));

        Complex<T> dt_factor = Complex<T>(0, -dt / hbar);

        // Build evolution matrix: I - iHdt/ħ (first order)
        for (size_t i = 0; i < n_states_; ++i) {
            evolution_matrix[i][i] = Complex<T>(1, 0); // Identity
            for (size_t j = 0; j < n_states_; ++j) {
                evolution_matrix[i][j] += dt_factor * matrix_[i][j];
            }
        }

        psi.applyOperator(evolution_matrix);
        psi.normalize(); // Maintain normalization
    }

    T getGroundStateEnergy() const {
        if (n_states_ == 0) return 0;
        return matrix_[0][0].real(); // Assumes diagonal form or approximation
    }

    const std::vector<std::vector<Complex<T>>>& getMatrix() const {
        return matrix_;
    }

    size_t getNumStates() const { return n_states_; }

    void addPerturbation(const HamiltonianOperator<T>& perturbation, T strength) {
        if (perturbation.getNumStates() != n_states_) return;

        const auto& pert_matrix = perturbation.getMatrix();
        for (size_t i = 0; i < n_states_; ++i) {
            for (size_t j = 0; j < n_states_; ++j) {
                matrix_[i][j] += strength * pert_matrix[i][j];
            }
        }
    }
};

template<typename T>
struct ClassicalParticle {
    Vec3<T> position;
    Vec3<T> velocity;
    Vec3<T> force;
    T mass;
    T charge;
    size_t quantum_state;
    bool is_quantum_coupled;

    ClassicalParticle(T m = 1.0, T q = 0.0)
        : position(Vec3<T>()), velocity(Vec3<T>()), force(Vec3<T>()),
          mass(m), charge(q), quantum_state(0), is_quantum_coupled(false) {}

    void updatePosition(T dt) {
        position = position + velocity * dt;
    }

    void updateVelocity(T dt) {
        if (mass > 1e-12) {
            velocity = velocity + force * (dt / mass);
        }
    }

    void applyForce(const Vec3<T>& f) {
        force = force + f;
    }

    void clearForces() {
        force = Vec3<T>(0, 0, 0);
    }

    T getKineticEnergy() const {
        return 0.5 * mass * velocity.dot(velocity);
    }
};

template<typename T>
class QuantumClassicalCoupling {
private:
    T coupling_strength_;
    T decoherence_rate_;
    T measurement_frequency_;
    T last_measurement_time_;

public:
    QuantumClassicalCoupling(T coupling = 1.0, T decoherence = 0.01, T meas_freq = 10.0)
        : coupling_strength_(coupling), decoherence_rate_(decoherence),
          measurement_frequency_(meas_freq), last_measurement_time_(0) {}

    void applyQuantumForce(ClassicalParticle<T>& particle, const WaveFunction<T>& psi, T time) {
        if (!particle.is_quantum_coupled) return;

        // Calculate force based on quantum state probabilities
        Vec3<T> quantum_force(0, 0, 0);

        for (size_t state = 0; state < psi.getNumStates(); ++state) {
            T probability = psi.getProbability(state);

            // Example: different quantum states create different force fields
            if (state == 0) {
                // Ground state - attractive force toward origin
                quantum_force = particle.position * (-coupling_strength_ * probability);
            } else if (state == 1) {
                // Excited state - repulsive force
                quantum_force = particle.position * (coupling_strength_ * probability);
            } else {
                // Higher states - oscillatory forces
                T phase = 2.0 * M_PI * state * time;
                quantum_force = Vec3<T>(std::cos(phase), std::sin(phase), 0) *
                               (coupling_strength_ * probability);
            }
        }

        particle.applyForce(quantum_force);
    }

    void applyClassicalBackaction(WaveFunction<T>& psi, const ClassicalParticle<T>& particle, T dt) {
        if (!particle.is_quantum_coupled) return;

        // Classical motion affects quantum state through position-dependent coupling
        T coupling_factor = coupling_strength_ * particle.position.norm() * dt;

        // Create position-dependent Hamiltonian perturbation
        HamiltonianOperator<T> perturbation(psi.getNumStates());

        for (size_t i = 0; i < psi.getNumStates(); ++i) {
            for (size_t j = 0; j < psi.getNumStates(); ++j) {
                if (i == j) {
                    // Diagonal terms - energy shift
                    perturbation.setElement(i, j, Complex<T>(coupling_factor * i, 0));
                } else {
                    // Off-diagonal terms - state mixing
                    T mixing = coupling_factor * std::exp(-std::abs(static_cast<int>(i) - static_cast<int>(j)));
                    perturbation.setElement(i, j, Complex<T>(mixing, 0));
                }
            }
        }

        psi.applyOperator(perturbation.getMatrix());
    }

    void applyDecoherence(WaveFunction<T>& psi, T dt) {
        // Simple decoherence model - reduce off-diagonal elements
        for (size_t i = 0; i < psi.getNumStates(); ++i) {
            for (size_t j = 0; j < psi.getNumStates(); ++j) {
                if (i != j) {
                    Complex<T> current_coeff = psi.getCoefficient(j);
                    T decay_factor = std::exp(-decoherence_rate_ * dt);

                    // Reduce coherence between different states
                    psi.setCoefficient(j, current_coeff * decay_factor);
                }
            }
        }
        psi.normalize();
    }

    bool shouldMeasure(T current_time) {
        if (current_time - last_measurement_time_ >= 1.0 / measurement_frequency_) {
            last_measurement_time_ = current_time;
            return true;
        }
        return false;
    }

    void performMeasurement(WaveFunction<T>& psi, ClassicalParticle<T>& particle) {
        size_t measured_state = psi.measureState();
        psi.collapseToState(measured_state);
        particle.quantum_state = measured_state;
    }

    T getCouplingStrength() const { return coupling_strength_; }
    void setCouplingStrength(T strength) { coupling_strength_ = strength; }

    T getDecoherenceRate() const { return decoherence_rate_; }
    void setDecoherenceRate(T rate) { decoherence_rate_ = rate; }
};

template<typename T>
class QuantumClassicalSystem {
private:
    std::vector<ClassicalParticle<T>> classical_particles_;
    std::vector<WaveFunction<T>> quantum_subsystems_;
    std::vector<HamiltonianOperator<T>> hamiltonians_;
    std::unique_ptr<QuantumClassicalCoupling<T>> coupling_;
    T simulation_time_;
    T time_step_;
    size_t step_count_;

public:
    QuantumClassicalSystem(T dt = 0.001)
        : simulation_time_(0), time_step_(dt), step_count_(0) {
        coupling_ = std::make_unique<QuantumClassicalCoupling<T>>();
    }

    void addClassicalParticle(const ClassicalParticle<T>& particle) {
        classical_particles_.push_back(particle);
    }

    void addQuantumSubsystem(const WaveFunction<T>& psi, const HamiltonianOperator<T>& hamiltonian) {
        quantum_subsystems_.push_back(psi);
        hamiltonians_.push_back(hamiltonian);
    }

    void setCoupling(std::unique_ptr<QuantumClassicalCoupling<T>> new_coupling) {
        coupling_ = std::move(new_coupling);
    }

    void enableQuantumCoupling(size_t particle_index, bool enable = true) {
        if (particle_index < classical_particles_.size()) {
            classical_particles_[particle_index].is_quantum_coupled = enable;
        }
    }

    void simulationStep() {
        // Clear forces
        for (auto& particle : classical_particles_) {
            particle.clearForces();
        }

        // Apply quantum forces to classical particles
        for (size_t i = 0; i < classical_particles_.size(); ++i) {
            for (size_t j = 0; j < quantum_subsystems_.size(); ++j) {
                coupling_->applyQuantumForce(classical_particles_[i], quantum_subsystems_[j], simulation_time_);
            }
        }

        // Apply classical forces (e.g., gravity, electromagnetic)
        applyClassicalForces();

        // Update classical particles
        for (auto& particle : classical_particles_) {
            particle.updateVelocity(time_step_);
            particle.updatePosition(time_step_);
        }

        // Apply classical backaction on quantum systems
        for (size_t i = 0; i < quantum_subsystems_.size(); ++i) {
            for (const auto& particle : classical_particles_) {
                coupling_->applyClassicalBackaction(quantum_subsystems_[i], particle, time_step_);
            }
        }

        // Evolve quantum systems
        for (size_t i = 0; i < quantum_subsystems_.size(); ++i) {
            hamiltonians_[i].timeEvolution(quantum_subsystems_[i], time_step_);
        }

        // Apply decoherence
        for (auto& psi : quantum_subsystems_) {
            coupling_->applyDecoherence(psi, time_step_);
        }

        // Perform measurements if needed
        if (coupling_->shouldMeasure(simulation_time_)) {
            for (size_t i = 0; i < quantum_subsystems_.size() && i < classical_particles_.size(); ++i) {
                coupling_->performMeasurement(quantum_subsystems_[i], classical_particles_[i]);
            }
        }

        simulation_time_ += time_step_;
        step_count_++;
    }

    void run(T total_time) {
        size_t num_steps = static_cast<size_t>(total_time / time_step_);

        for (size_t step = 0; step < num_steps; ++step) {
            simulationStep();
        }
    }

    // Getters
    const std::vector<ClassicalParticle<T>>& getClassicalParticles() const { return classical_particles_; }
    const std::vector<WaveFunction<T>>& getQuantumSubsystems() const { return quantum_subsystems_; }
    T getSimulationTime() const { return simulation_time_; }
    size_t getStepCount() const { return step_count_; }

    // Analysis methods
    T getTotalClassicalEnergy() const {
        T total_energy = 0;
        for (const auto& particle : classical_particles_) {
            total_energy += particle.getKineticEnergy();
        }
        return total_energy;
    }

    T getQuantumPurity(size_t subsystem_index) const {
        if (subsystem_index >= quantum_subsystems_.size()) return 0;

        const auto& psi = quantum_subsystems_[subsystem_index];
        T purity = 0;
        for (size_t i = 0; i < psi.getNumStates(); ++i) {
            T prob = psi.getProbability(i);
            purity += prob * prob;
        }
        return purity;
    }

    std::vector<T> getQuantumExpectationValues(size_t subsystem_index) const {
        std::vector<T> expectations;
        if (subsystem_index >= quantum_subsystems_.size()) return expectations;

        const auto& psi = quantum_subsystems_[subsystem_index];
        const auto& hamiltonian = hamiltonians_[subsystem_index];

        // Energy expectation value
        expectations.push_back(psi.getExpectationValue(hamiltonian.getMatrix()));

        return expectations;
    }

    void reset() {
        simulation_time_ = 0;
        step_count_ = 0;

        for (auto& particle : classical_particles_) {
            particle.position = Vec3<T>(0, 0, 0);
            particle.velocity = Vec3<T>(0, 0, 0);
            particle.force = Vec3<T>(0, 0, 0);
        }

        for (auto& psi : quantum_subsystems_) {
            for (size_t i = 0; i < psi.getNumStates(); ++i) {
                psi.setCoefficient(i, (i == 0) ? Complex<T>(1, 0) : Complex<T>(0, 0));
            }
        }
    }

    void setTimeStep(T dt) { time_step_ = dt; }
    T getTimeStep() const { return time_step_; }

    size_t getNumClassicalParticles() const { return classical_particles_.size(); }
    size_t getNumQuantumSubsystems() const { return quantum_subsystems_.size(); }

private:
    void applyClassicalForces() {
        // Apply gravity
        for (auto& particle : classical_particles_) {
            particle.applyForce(Vec3<T>(0, 0, -9.81 * particle.mass));
        }

        // Apply inter-particle forces (Coulomb, if charged)
        for (size_t i = 0; i < classical_particles_.size(); ++i) {
            for (size_t j = i + 1; j < classical_particles_.size(); ++j) {
                auto& p1 = classical_particles_[i];
                auto& p2 = classical_particles_[j];

                if (std::abs(p1.charge) > 1e-12 && std::abs(p2.charge) > 1e-12) {
                    Vec3<T> r = p2.position - p1.position;
                    T distance = r.norm();

                    if (distance > 1e-6) {
                        T k_coulomb = 8.99e9; // Coulomb constant
                        T force_magnitude = k_coulomb * p1.charge * p2.charge / (distance * distance);
                        Vec3<T> force_direction = r * (1.0 / distance);
                        Vec3<T> force = force_direction * force_magnitude;

                        p1.applyForce(force * (-1.0));
                        p2.applyForce(force);
                    }
                }
            }
        }
    }
};

template<typename T>
class QuantumClassicalFactory {
public:
    static std::unique_ptr<QuantumClassicalSystem<T>> createSpinBosonModel(
        T spin_frequency, [[maybe_unused]] T boson_frequency, T coupling_strength) {

        auto system = std::make_unique<QuantumClassicalSystem<T>>();

        // Create spin-1/2 quantum system
        WaveFunction<T> spin_system(2);
        spin_system.setCoefficient(0, Complex<T>(1, 0)); // Start in |↓⟩ state

        HamiltonianOperator<T> spin_hamiltonian(2);
        spin_hamiltonian.buildSpinHamiltonian(spin_frequency, 1.0, Vec3<T>(0, 0, 1));

        system->addQuantumSubsystem(spin_system, spin_hamiltonian);

        // Create classical harmonic oscillator (representing bosonic bath)
        ClassicalParticle<T> boson_particle(1.0);
        boson_particle.position = Vec3<T>(1.0, 0, 0);
        boson_particle.velocity = Vec3<T>(0, 0, 0);
        boson_particle.is_quantum_coupled = true;

        system->addClassicalParticle(boson_particle);
        system->enableQuantumCoupling(0, true);

        // Set up coupling
        auto coupling = std::make_unique<QuantumClassicalCoupling<T>>(coupling_strength, 0.01, 10.0);
        system->setCoupling(std::move(coupling));

        return system;
    }

    static std::unique_ptr<QuantumClassicalSystem<T>> createQuantumDotArray(
        size_t num_dots, T tunnel_coupling, T classical_coupling) {

        auto system = std::make_unique<QuantumClassicalSystem<T>>();

        // Create quantum dot chain
        WaveFunction<T> electron_system(num_dots);
        electron_system.setCoefficient(0, Complex<T>(1, 0)); // Electron starts in first dot

        HamiltonianOperator<T> quantum_dot_hamiltonian(num_dots);
        quantum_dot_hamiltonian.buildTunnelCoupling(tunnel_coupling);

        system->addQuantumSubsystem(electron_system, quantum_dot_hamiltonian);

        // Create classical charges that can gate the quantum dots
        for (size_t i = 0; i < num_dots; ++i) {
            ClassicalParticle<T> gate_charge(1.0, 1e-9); // Small charge
            gate_charge.position = Vec3<T>(i * 1e-6, 1e-6, 0); // Above quantum dots
            gate_charge.is_quantum_coupled = true;
            system->addClassicalParticle(gate_charge);
            system->enableQuantumCoupling(i, true);
        }

        auto coupling = std::make_unique<QuantumClassicalCoupling<T>>(classical_coupling, 0.005, 5.0);
        system->setCoupling(std::move(coupling));

        return system;
    }

    static std::unique_ptr<QuantumClassicalSystem<T>> createMolecularDynamicsQM(
        size_t num_classical_atoms, size_t quantum_system_size, T qm_mm_coupling) {

        auto system = std::make_unique<QuantumClassicalSystem<T>>();

        // Create quantum subsystem (e.g., active site)
        WaveFunction<T> qm_system(quantum_system_size);
        for (size_t i = 0; i < quantum_system_size; ++i) {
            T coeff_real = 1.0 / std::sqrt(static_cast<T>(quantum_system_size));
            qm_system.setCoefficient(i, Complex<T>(coeff_real, 0));
        }

        HamiltonianOperator<T> molecular_hamiltonian(quantum_system_size);
        molecular_hamiltonian.buildHarmonicOscillator(1.0, 1.0);

        system->addQuantumSubsystem(qm_system, molecular_hamiltonian);

        // Create classical atoms
        for (size_t i = 0; i < num_classical_atoms; ++i) {
            ClassicalParticle<T> atom(1.0); // 1 amu

            // Random initial positions
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> pos_dist(-1.0, 1.0);

            atom.position = Vec3<T>(pos_dist(gen), pos_dist(gen), pos_dist(gen));
            atom.velocity = Vec3<T>(pos_dist(gen) * 0.1, pos_dist(gen) * 0.1, pos_dist(gen) * 0.1);
            atom.is_quantum_coupled = (i < quantum_system_size); // Only some atoms couple to QM

            system->addClassicalParticle(atom);
            if (atom.is_quantum_coupled) {
                system->enableQuantumCoupling(i, true);
            }
        }

        auto coupling = std::make_unique<QuantumClassicalCoupling<T>>(qm_mm_coupling, 0.02, 2.0);
        system->setCoupling(std::move(coupling));

        return system;
    }
};

} // namespace quantum
} // namespace physgrad