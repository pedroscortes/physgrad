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

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace physgrad {
namespace codesign {

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

    CUDA_HOST_DEVICE Vec3 normalize() const {
        T n = norm();
        return (n > 1e-8) ? Vec3(x/n, y/n, z/n) : Vec3(0, 0, 0);
    }

    CUDA_HOST_DEVICE Vec3 cross(const Vec3& other) const {
        return Vec3(y * other.z - z * other.y,
                   z * other.x - x * other.z,
                   x * other.y - y * other.x);
    }
};

enum class ComponentType {
    RIGID_BODY,
    SOFT_BODY,
    ACTUATOR,
    SENSOR,
    JOINT,
    CONSTRAINT
};

enum class MaterialType {
    STEEL,
    ALUMINUM,
    PLASTIC,
    RUBBER,
    CARBON_FIBER,
    BIO_MATERIAL
};

template<typename T>
struct MaterialProperties {
    T density;
    T young_modulus;
    T poisson_ratio;
    T yield_strength;
    T damping_coefficient;

    MaterialProperties()
        : density(7800), young_modulus(200e9), poisson_ratio(0.3),
          yield_strength(250e6), damping_coefficient(0.01) {}

    MaterialProperties(T d, T e, T nu, T sigma_y, T damping)
        : density(d), young_modulus(e), poisson_ratio(nu),
          yield_strength(sigma_y), damping_coefficient(damping) {}
};

template<typename T>
class MaterialDatabase {
private:
    std::unordered_map<MaterialType, MaterialProperties<T>> materials_;

public:
    MaterialDatabase() {
        materials_[MaterialType::STEEL] = MaterialProperties<T>(7800, 200e9, 0.3, 250e6, 0.01);
        materials_[MaterialType::ALUMINUM] = MaterialProperties<T>(2700, 70e9, 0.33, 95e6, 0.005);
        materials_[MaterialType::PLASTIC] = MaterialProperties<T>(1200, 3e9, 0.4, 50e6, 0.1);
        materials_[MaterialType::RUBBER] = MaterialProperties<T>(1500, 1e6, 0.49, 10e6, 0.2);
        materials_[MaterialType::CARBON_FIBER] = MaterialProperties<T>(1600, 150e9, 0.3, 500e6, 0.002);
        materials_[MaterialType::BIO_MATERIAL] = MaterialProperties<T>(1000, 10e9, 0.45, 30e6, 0.05);
    }

    const MaterialProperties<T>& getMaterial(MaterialType type) const {
        auto it = materials_.find(type);
        return (it != materials_.end()) ? it->second : materials_.at(MaterialType::STEEL);
    }

    void addMaterial(MaterialType type, const MaterialProperties<T>& props) {
        materials_[type] = props;
    }
};

template<typename T>
struct GeometricPrimitive {
    enum class Type { BOX, SPHERE, CYLINDER, CAPSULE, MESH };

    Type type;
    Vec3<T> dimensions; // x,y,z for box; radius,height,unused for cylinder; radius,unused,unused for sphere
    Vec3<T> position;
    Vec3<T> rotation; // Euler angles
    T volume;

    GeometricPrimitive(Type t, const Vec3<T>& dims, const Vec3<T>& pos = Vec3<T>())
        : type(t), dimensions(dims), position(pos), rotation(Vec3<T>()) {
        computeVolume();
    }

    void computeVolume() {
        switch (type) {
            case Type::BOX:
                volume = dimensions.x * dimensions.y * dimensions.z;
                break;
            case Type::SPHERE:
                volume = (4.0/3.0) * M_PI * std::pow(dimensions.x, 3);
                break;
            case Type::CYLINDER:
                volume = M_PI * dimensions.x * dimensions.x * dimensions.y;
                break;
            case Type::CAPSULE:
                volume = M_PI * dimensions.x * dimensions.x * dimensions.y +
                        (4.0/3.0) * M_PI * std::pow(dimensions.x, 3);
                break;
            default:
                volume = 1.0;
        }
    }

    T getMass(const MaterialProperties<T>& material) const {
        return volume * material.density;
    }

    Vec3<T> getInertia(const MaterialProperties<T>& material) const {
        T mass = getMass(material);
        switch (type) {
            case Type::BOX: {
                T ix = mass * (dimensions.y*dimensions.y + dimensions.z*dimensions.z) / 12.0;
                T iy = mass * (dimensions.x*dimensions.x + dimensions.z*dimensions.z) / 12.0;
                T iz = mass * (dimensions.x*dimensions.x + dimensions.y*dimensions.y) / 12.0;
                return Vec3<T>(ix, iy, iz);
            }
            case Type::SPHERE: {
                T i = 0.4 * mass * dimensions.x * dimensions.x;
                return Vec3<T>(i, i, i);
            }
            case Type::CYLINDER: {
                T ix = mass * dimensions.x * dimensions.x / 2.0;
                T iy = mass * (3*dimensions.x*dimensions.x + dimensions.y*dimensions.y) / 12.0;
                return Vec3<T>(ix, iy, iy);
            }
            default:
                return Vec3<T>(mass, mass, mass);
        }
    }
};

template<typename T>
class RobotComponent {
protected:
    std::string name_;
    ComponentType type_;
    MaterialType material_type_;
    GeometricPrimitive<T> geometry_;
    Vec3<T> position_;
    Vec3<T> velocity_;
    Vec3<T> force_;
    bool is_active_;

public:
    RobotComponent(const std::string& name, ComponentType type, MaterialType material,
                   const GeometricPrimitive<T>& geom)
        : name_(name), type_(type), material_type_(material), geometry_(geom),
          position_(geom.position), velocity_(Vec3<T>()), force_(Vec3<T>()), is_active_(true) {}

    virtual ~RobotComponent() = default;

    const std::string& getName() const { return name_; }
    ComponentType getType() const { return type_; }
    MaterialType getMaterialType() const { return material_type_; }
    const GeometricPrimitive<T>& getGeometry() const { return geometry_; }
    const Vec3<T>& getPosition() const { return position_; }
    const Vec3<T>& getVelocity() const { return velocity_; }
    const Vec3<T>& getForce() const { return force_; }
    bool isActive() const { return is_active_; }

    void setPosition(const Vec3<T>& pos) { position_ = pos; }
    void setVelocity(const Vec3<T>& vel) { velocity_ = vel; }
    void setForce(const Vec3<T>& f) { force_ = f; }
    void setActive(bool active) { is_active_ = active; }

    virtual void update(T dt) {
        if (is_active_) {
            position_ = position_ + velocity_ * dt;
        }
    }

    virtual T computePerformanceMetric() const { return 0.0; }
    virtual std::vector<T> getDesignParameters() const { return {}; }
    virtual void setDesignParameters(const std::vector<T>& params) {}
};

template<typename T>
class RigidBodyComponent : public RobotComponent<T> {
private:
    Vec3<T> angular_velocity_;
    Vec3<T> torque_;
    T mass_;
    Vec3<T> inertia_;

public:
    RigidBodyComponent(const std::string& name, MaterialType material,
                       const GeometricPrimitive<T>& geom, const MaterialDatabase<T>& mat_db)
        : RobotComponent<T>(name, ComponentType::RIGID_BODY, material, geom),
          angular_velocity_(Vec3<T>()), torque_(Vec3<T>()) {

        const auto& props = mat_db.getMaterial(material);
        mass_ = geom.getMass(props);
        inertia_ = geom.getInertia(props);
    }

    T getMass() const { return mass_; }
    const Vec3<T>& getInertia() const { return inertia_; }
    const Vec3<T>& getAngularVelocity() const { return angular_velocity_; }
    const Vec3<T>& getTorque() const { return torque_; }

    void setAngularVelocity(const Vec3<T>& omega) { angular_velocity_ = omega; }
    void setTorque(const Vec3<T>& tau) { torque_ = tau; }

    void update(T dt) override {
        RobotComponent<T>::update(dt);
        // Simple Euler integration for angular motion
        angular_velocity_ = angular_velocity_ + Vec3<T>(torque_.x/inertia_.x,
                                                       torque_.y/inertia_.y,
                                                       torque_.z/inertia_.z) * dt;
    }

    std::vector<T> getDesignParameters() const override {
        const auto& dims = this->geometry_.dimensions;
        return {dims.x, dims.y, dims.z};
    }

    void setDesignParameters(const std::vector<T>& params) override {
        if (params.size() >= 3) {
            this->geometry_.dimensions = Vec3<T>(params[0], params[1], params[2]);
            this->geometry_.computeVolume();
            // Update mass and inertia based on new geometry
            MaterialDatabase<T> mat_db; // Use default database
            const auto& props = mat_db.getMaterial(this->material_type_);
            mass_ = this->geometry_.getMass(props);
            inertia_ = this->geometry_.getInertia(props);
        }
    }
};

template<typename T>
class ActuatorComponent : public RobotComponent<T> {
private:
    T max_force_;
    T current_force_;
    T control_signal_;
    T efficiency_;

public:
    ActuatorComponent(const std::string& name, MaterialType material,
                      const GeometricPrimitive<T>& geom, T max_force)
        : RobotComponent<T>(name, ComponentType::ACTUATOR, material, geom),
          max_force_(max_force), current_force_(0), control_signal_(0), efficiency_(0.8) {}

    T getMaxForce() const { return max_force_; }
    T getCurrentForce() const { return current_force_; }
    T getControlSignal() const { return control_signal_; }
    T getEfficiency() const { return efficiency_; }

    void setControlSignal(T signal) {
        control_signal_ = std::clamp(signal, -1.0, 1.0);
        current_force_ = control_signal_ * max_force_ * efficiency_;
        this->setForce(Vec3<T>(current_force_, 0, 0)); // Assume force along x-axis
    }

    T computePerformanceMetric() const override {
        return std::abs(current_force_) / max_force_; // Force utilization
    }

    std::vector<T> getDesignParameters() const override {
        return {max_force_, efficiency_};
    }

    void setDesignParameters(const std::vector<T>& params) override {
        if (params.size() >= 2) {
            max_force_ = std::max(params[0], static_cast<T>(0.1));
            efficiency_ = std::clamp(params[1], static_cast<T>(0.1), static_cast<T>(1.0));
        }
    }
};

template<typename T>
class RobotMorphology {
private:
    std::vector<std::unique_ptr<RobotComponent<T>>> components_;
    std::vector<std::pair<size_t, size_t>> connections_; // Component index pairs
    MaterialDatabase<T> material_db_;
    T total_mass_;
    Vec3<T> center_of_mass_;

public:
    RobotMorphology() : total_mass_(0), center_of_mass_(Vec3<T>()) {}

    void addComponent(std::unique_ptr<RobotComponent<T>> component) {
        components_.push_back(std::move(component));
        updateMassProperties();
    }

    void addConnection(size_t comp1_idx, size_t comp2_idx) {
        if (comp1_idx < components_.size() && comp2_idx < components_.size()) {
            connections_.push_back({comp1_idx, comp2_idx});
        }
    }

    size_t getComponentCount() const { return components_.size(); }
    size_t getConnectionCount() const { return connections_.size(); }

    RobotComponent<T>* getComponent(size_t idx) {
        return (idx < components_.size()) ? components_[idx].get() : nullptr;
    }

    const RobotComponent<T>* getComponent(size_t idx) const {
        return (idx < components_.size()) ? components_[idx].get() : nullptr;
    }

    T getTotalMass() const { return total_mass_; }
    const Vec3<T>& getCenterOfMass() const { return center_of_mass_; }

    void updateMassProperties() {
        total_mass_ = 0;
        Vec3<T> weighted_pos(0, 0, 0);

        for (const auto& comp : components_) {
            if (comp->getType() == ComponentType::RIGID_BODY) {
                auto* rigid_body = static_cast<const RigidBodyComponent<T>*>(comp.get());
                T mass = rigid_body->getMass();
                total_mass_ += mass;
                weighted_pos = weighted_pos + comp->getPosition() * mass;
            }
        }

        center_of_mass_ = (total_mass_ > 0) ? weighted_pos * (1.0 / total_mass_) : Vec3<T>();
    }

    void update(T dt) {
        for (auto& comp : components_) {
            comp->update(dt);
        }
        updateMassProperties();
    }

    std::vector<T> getAllDesignParameters() const {
        std::vector<T> params;
        for (const auto& comp : components_) {
            auto comp_params = comp->getDesignParameters();
            params.insert(params.end(), comp_params.begin(), comp_params.end());
        }
        return params;
    }

    void setAllDesignParameters(const std::vector<T>& params) {
        size_t param_idx = 0;
        for (auto& comp : components_) {
            auto comp_param_count = comp->getDesignParameters().size();
            if (param_idx + comp_param_count <= params.size()) {
                std::vector<T> comp_params(params.begin() + param_idx,
                                         params.begin() + param_idx + comp_param_count);
                comp->setDesignParameters(comp_params);
                param_idx += comp_param_count;
            }
        }
        updateMassProperties();
    }

    T computeTotalPerformanceMetric() const {
        T total_metric = 0;
        for (const auto& comp : components_) {
            total_metric += comp->computePerformanceMetric();
        }
        return total_metric / components_.size();
    }
};

template<typename T>
class ObjectiveFunction {
public:
    virtual ~ObjectiveFunction() = default;
    virtual T evaluate(const RobotMorphology<T>& morphology) = 0;
    virtual std::string getName() const = 0;
};

template<typename T>
class LocomotionObjective : public ObjectiveFunction<T> {
private:
    T target_velocity_;
    T time_horizon_;

public:
    LocomotionObjective(T target_vel, T time_horizon)
        : target_velocity_(target_vel), time_horizon_(time_horizon) {}

    T evaluate(const RobotMorphology<T>& morphology) override {
        // Simplified locomotion metric based on actuator performance and mass efficiency
        T performance_metric = morphology.computeTotalPerformanceMetric();
        T mass_penalty = 1.0 / (1.0 + morphology.getTotalMass() / 100.0); // Normalize mass
        T connectivity_bonus = static_cast<T>(morphology.getConnectionCount()) /
                              static_cast<T>(morphology.getComponentCount());

        return performance_metric * mass_penalty * (1.0 + connectivity_bonus);
    }

    std::string getName() const override { return "LocomotionObjective"; }
};

template<typename T>
class StabilityObjective : public ObjectiveFunction<T> {
public:
    T evaluate(const RobotMorphology<T>& morphology) override {
        // Stability based on center of mass height and base of support
        const Vec3<T>& com = morphology.getCenterOfMass();
        T stability_metric = 1.0 / (1.0 + std::abs(com.z)); // Lower center of mass is better

        // Add mass distribution consideration
        T mass_distribution = 1.0 / (1.0 + morphology.getTotalMass() / 50.0);

        return stability_metric * mass_distribution;
    }

    std::string getName() const override { return "StabilityObjective"; }
};

template<typename T>
class EfficiencyObjective : public ObjectiveFunction<T> {
public:
    T evaluate(const RobotMorphology<T>& morphology) override {
        // Energy efficiency based on mass-to-performance ratio
        T performance = morphology.computeTotalPerformanceMetric();
        T mass = morphology.getTotalMass();

        if (mass < 1e-6) return 0.0;

        T efficiency = performance / std::sqrt(mass);
        return std::tanh(efficiency); // Bounded metric
    }

    std::string getName() const override { return "EfficiencyObjective"; }
};

template<typename T>
class MultiObjectiveFunction {
private:
    std::vector<std::unique_ptr<ObjectiveFunction<T>>> objectives_;
    std::vector<T> weights_;

public:
    void addObjective(std::unique_ptr<ObjectiveFunction<T>> objective, T weight = 1.0) {
        objectives_.push_back(std::move(objective));
        weights_.push_back(weight);
    }

    T evaluate(const RobotMorphology<T>& morphology) {
        T total_objective = 0;
        T total_weight = 0;

        for (size_t i = 0; i < objectives_.size(); ++i) {
            T obj_value = objectives_[i]->evaluate(morphology);
            total_objective += weights_[i] * obj_value;
            total_weight += weights_[i];
        }

        return (total_weight > 0) ? total_objective / total_weight : 0;
    }

    std::vector<T> evaluateAll(const RobotMorphology<T>& morphology) {
        std::vector<T> results;
        for (const auto& obj : objectives_) {
            results.push_back(obj->evaluate(morphology));
        }
        return results;
    }

    size_t getObjectiveCount() const { return objectives_.size(); }

    std::string getObjectiveName(size_t idx) const {
        return (idx < objectives_.size()) ? objectives_[idx]->getName() : "Unknown";
    }
};

template<typename T>
class EvolutionaryOptimizer {
private:
    size_t population_size_;
    size_t max_generations_;
    T mutation_rate_;
    T crossover_rate_;
    std::mt19937 rng_;

    struct Individual {
        std::vector<T> parameters;
        T fitness;

        Individual(const std::vector<T>& params) : parameters(params), fitness(0) {}
    };

    std::vector<Individual> population_;

public:
    EvolutionaryOptimizer(size_t pop_size, size_t max_gen, T mutation_rate, T crossover_rate)
        : population_size_(pop_size), max_generations_(max_gen),
          mutation_rate_(mutation_rate), crossover_rate_(crossover_rate),
          rng_(std::random_device{}()) {}

    void initializePopulation(const std::vector<T>& initial_params,
                            const std::vector<std::pair<T, T>>& bounds) {
        population_.clear();
        population_.reserve(population_size_);

        // Add initial individual
        population_.emplace_back(initial_params);

        // Generate random population
        for (size_t i = 1; i < population_size_; ++i) {
            std::vector<T> params(initial_params.size());
            for (size_t j = 0; j < params.size(); ++j) {
                std::uniform_real_distribution<T> dist(bounds[j].first, bounds[j].second);
                params[j] = dist(rng_);
            }
            population_.emplace_back(params);
        }
    }

    Individual mutate(const Individual& parent, const std::vector<std::pair<T, T>>& bounds) {
        Individual child = parent;

        for (size_t i = 0; i < child.parameters.size(); ++i) {
            if (std::uniform_real_distribution<T>(0, 1)(rng_) < mutation_rate_) {
                T range = bounds[i].second - bounds[i].first;
                T mutation = std::normal_distribution<T>(0, range * 0.1)(rng_);
                child.parameters[i] = std::clamp(parent.parameters[i] + mutation,
                                               bounds[i].first, bounds[i].second);
            }
        }

        return child;
    }

    Individual crossover(const Individual& parent1, const Individual& parent2) {
        Individual child(parent1.parameters);

        for (size_t i = 0; i < child.parameters.size(); ++i) {
            if (std::uniform_real_distribution<T>(0, 1)(rng_) < crossover_rate_) {
                child.parameters[i] = parent2.parameters[i];
            }
        }

        return child;
    }

    std::vector<T> optimize(RobotMorphology<T>& morphology,
                           MultiObjectiveFunction<T>& objective,
                           const std::vector<std::pair<T, T>>& bounds) {

        std::vector<T> initial_params = morphology.getAllDesignParameters();
        initializePopulation(initial_params, bounds);

        T best_fitness = -std::numeric_limits<T>::infinity();
        std::vector<T> best_parameters = initial_params;

        for (size_t generation = 0; generation < max_generations_; ++generation) {
            // Evaluate fitness
            for (auto& individual : population_) {
                morphology.setAllDesignParameters(individual.parameters);
                individual.fitness = objective.evaluate(morphology);

                if (individual.fitness > best_fitness) {
                    best_fitness = individual.fitness;
                    best_parameters = individual.parameters;
                }
            }

            // Sort by fitness
            std::sort(population_.begin(), population_.end(),
                     [](const Individual& a, const Individual& b) {
                         return a.fitness > b.fitness;
                     });

            if (generation % 10 == 0) {
                std::cout << "Generation " << generation
                          << ": Best fitness = " << best_fitness << std::endl;
            }

            // Create next generation
            std::vector<Individual> next_generation;
            next_generation.reserve(population_size_);

            // Elitism: keep best individuals
            size_t elite_count = population_size_ / 4;
            for (size_t i = 0; i < elite_count; ++i) {
                next_generation.push_back(population_[i]);
            }

            // Generate offspring
            while (next_generation.size() < population_size_) {
                // Tournament selection
                size_t parent1_idx = tournamentSelection();
                size_t parent2_idx = tournamentSelection();

                Individual child = crossover(population_[parent1_idx], population_[parent2_idx]);
                child = mutate(child, bounds);

                next_generation.push_back(child);
            }

            population_ = std::move(next_generation);
        }

        return best_parameters;
    }

private:
    size_t tournamentSelection() {
        size_t tournament_size = 3;
        size_t best_idx = std::uniform_int_distribution<size_t>(0, population_.size() - 1)(rng_);

        for (size_t i = 1; i < tournament_size; ++i) {
            size_t candidate_idx = std::uniform_int_distribution<size_t>(0, population_.size() - 1)(rng_);
            if (population_[candidate_idx].fitness > population_[best_idx].fitness) {
                best_idx = candidate_idx;
            }
        }

        return best_idx;
    }
};

template<typename T>
class RobotCodesignFramework {
private:
    std::unique_ptr<RobotMorphology<T>> morphology_;
    std::unique_ptr<MultiObjectiveFunction<T>> objective_function_;
    std::unique_ptr<EvolutionaryOptimizer<T>> optimizer_;
    MaterialDatabase<T> material_db_;

public:
    RobotCodesignFramework()
        : morphology_(std::make_unique<RobotMorphology<T>>()),
          objective_function_(std::make_unique<MultiObjectiveFunction<T>>()),
          optimizer_(std::make_unique<EvolutionaryOptimizer<T>>(50, 100, 0.1, 0.7)) {}

    RobotMorphology<T>* getMorphology() { return morphology_.get(); }
    MultiObjectiveFunction<T>* getObjectiveFunction() { return objective_function_.get(); }

    void addObjective(std::unique_ptr<ObjectiveFunction<T>> objective, T weight = 1.0) {
        objective_function_->addObjective(std::move(objective), weight);
    }

    std::vector<T> optimizeMorphology(const std::vector<std::pair<T, T>>& parameter_bounds) {
        if (morphology_->getComponentCount() == 0) {
            throw std::runtime_error("No components in morphology to optimize");
        }

        std::cout << "Starting robot co-design optimization..." << std::endl;
        std::cout << "Population size: 50, Generations: 100" << std::endl;
        std::cout << "Components: " << morphology_->getComponentCount() << std::endl;
        std::cout << "Objectives: " << objective_function_->getObjectiveCount() << std::endl;

        return optimizer_->optimize(*morphology_, *objective_function_, parameter_bounds);
    }

    void saveMorphology(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }

        file << "# Robot Morphology Configuration\n";
        file << "components: " << morphology_->getComponentCount() << "\n";
        file << "connections: " << morphology_->getConnectionCount() << "\n";
        file << "total_mass: " << morphology_->getTotalMass() << "\n";

        const Vec3<T>& com = morphology_->getCenterOfMass();
        file << "center_of_mass: " << com.x << " " << com.y << " " << com.z << "\n";

        auto params = morphology_->getAllDesignParameters();
        file << "design_parameters: ";
        for (T param : params) {
            file << param << " ";
        }
        file << "\n";

        file.close();
    }

    T evaluateCurrentMorphology() {
        return objective_function_->evaluate(*morphology_);
    }

    std::vector<T> evaluateAllObjectives() {
        return objective_function_->evaluateAll(*morphology_);
    }

    void setOptimizerParameters(size_t pop_size, size_t max_gen, T mutation_rate, T crossover_rate) {
        optimizer_ = std::make_unique<EvolutionaryOptimizer<T>>(pop_size, max_gen, mutation_rate, crossover_rate);
    }

    const MaterialDatabase<T>& getMaterialDatabase() const { return material_db_; }
};

template<typename T>
class RobotFactory {
public:
    static std::unique_ptr<RobotMorphology<T>> createQuadrupedRobot(const MaterialDatabase<T>& mat_db) {
        auto morphology = std::make_unique<RobotMorphology<T>>();

        // Create body
        GeometricPrimitive<T> body_geom(GeometricPrimitive<T>::Type::BOX,
                                       Vec3<T>(0.4, 0.2, 0.1), Vec3<T>(0, 0, 0.1));
        auto body = std::make_unique<RigidBodyComponent<T>>("body", MaterialType::ALUMINUM, body_geom, mat_db);
        morphology->addComponent(std::move(body));

        // Create legs
        for (int i = 0; i < 4; ++i) {
            T x = (i % 2 == 0) ? 0.15 : -0.15;
            T y = (i < 2) ? 0.08 : -0.08;

            GeometricPrimitive<T> leg_geom(GeometricPrimitive<T>::Type::CYLINDER,
                                          Vec3<T>(0.02, 0.15, 0), Vec3<T>(x, y, 0.05));
            auto leg = std::make_unique<RigidBodyComponent<T>>("leg_" + std::to_string(i),
                                                              MaterialType::CARBON_FIBER, leg_geom, mat_db);
            morphology->addComponent(std::move(leg));
        }

        // Create actuators
        for (int i = 0; i < 4; ++i) {
            T x = (i % 2 == 0) ? 0.15 : -0.15;
            T y = (i < 2) ? 0.08 : -0.08;

            GeometricPrimitive<T> actuator_geom(GeometricPrimitive<T>::Type::BOX,
                                               Vec3<T>(0.05, 0.03, 0.03), Vec3<T>(x, y, 0.08));
            auto actuator = std::make_unique<ActuatorComponent<T>>("actuator_" + std::to_string(i),
                                                                  MaterialType::STEEL, actuator_geom, 100.0);
            morphology->addComponent(std::move(actuator));
        }

        // Create connections
        for (int i = 0; i < 4; ++i) {
            // Connect body to leg
            morphology->addConnection(0, 1 + i);
            // Connect leg to actuator
            morphology->addConnection(1 + i, 5 + i);
        }

        return morphology;
    }

    static std::unique_ptr<RobotMorphology<T>> createBipedRobot(const MaterialDatabase<T>& mat_db) {
        auto morphology = std::make_unique<RobotMorphology<T>>();

        // Create torso
        GeometricPrimitive<T> torso_geom(GeometricPrimitive<T>::Type::BOX,
                                        Vec3<T>(0.3, 0.15, 0.5), Vec3<T>(0, 0, 0.4));
        auto torso = std::make_unique<RigidBodyComponent<T>>("torso", MaterialType::ALUMINUM, torso_geom, mat_db);
        morphology->addComponent(std::move(torso));

        // Create legs
        for (int i = 0; i < 2; ++i) {
            T y = (i == 0) ? 0.06 : -0.06;

            // Upper leg
            GeometricPrimitive<T> upper_leg_geom(GeometricPrimitive<T>::Type::CYLINDER,
                                                Vec3<T>(0.03, 0.2, 0), Vec3<T>(0, y, 0.2));
            auto upper_leg = std::make_unique<RigidBodyComponent<T>>("upper_leg_" + std::to_string(i),
                                                                   MaterialType::CARBON_FIBER, upper_leg_geom, mat_db);
            morphology->addComponent(std::move(upper_leg));

            // Lower leg
            GeometricPrimitive<T> lower_leg_geom(GeometricPrimitive<T>::Type::CYLINDER,
                                                Vec3<T>(0.025, 0.18, 0), Vec3<T>(0, y, 0.01));
            auto lower_leg = std::make_unique<RigidBodyComponent<T>>("lower_leg_" + std::to_string(i),
                                                                    MaterialType::CARBON_FIBER, lower_leg_geom, mat_db);
            morphology->addComponent(std::move(lower_leg));

            // Hip actuator
            GeometricPrimitive<T> hip_actuator_geom(GeometricPrimitive<T>::Type::BOX,
                                                   Vec3<T>(0.06, 0.04, 0.04), Vec3<T>(0, y, 0.3));
            auto hip_actuator = std::make_unique<ActuatorComponent<T>>("hip_actuator_" + std::to_string(i),
                                                                      MaterialType::STEEL, hip_actuator_geom, 150.0);
            morphology->addComponent(std::move(hip_actuator));

            // Knee actuator
            GeometricPrimitive<T> knee_actuator_geom(GeometricPrimitive<T>::Type::BOX,
                                                    Vec3<T>(0.05, 0.03, 0.03), Vec3<T>(0, y, 0.1));
            auto knee_actuator = std::make_unique<ActuatorComponent<T>>("knee_actuator_" + std::to_string(i),
                                                                       MaterialType::STEEL, knee_actuator_geom, 100.0);
            morphology->addComponent(std::move(knee_actuator));

            // Connections
            morphology->addConnection(0, 1 + i * 4);     // torso to upper leg
            morphology->addConnection(1 + i * 4, 2 + i * 4); // upper leg to lower leg
            morphology->addConnection(0, 3 + i * 4);     // torso to hip actuator
            morphology->addConnection(1 + i * 4, 4 + i * 4); // upper leg to knee actuator
        }

        return morphology;
    }
};

} // namespace codesign
} // namespace physgrad