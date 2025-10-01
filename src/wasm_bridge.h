#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <cmath>
#include <chrono>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#endif

namespace physgrad {
namespace wasm {

// WebAssembly-compatible vector type
template<typename T>
struct WasmVec3 {
    T x, y, z;

    WasmVec3() : x(0), y(0), z(0) {}
    WasmVec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    WasmVec3 operator+(const WasmVec3& other) const {
        return WasmVec3(x + other.x, y + other.y, z + other.z);
    }

    WasmVec3 operator-(const WasmVec3& other) const {
        return WasmVec3(x - other.x, y - other.y, z - other.z);
    }

    WasmVec3 operator*(T scalar) const {
        return WasmVec3(x * scalar, y * scalar, z * scalar);
    }

    T dot(const WasmVec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    T norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    WasmVec3 normalized() const {
        T n = norm();
        if (n > 1e-12) {
            return *this * (1.0 / n);
        }
        return *this;
    }
};

// Particle data structure for WebAssembly
template<typename T>
struct WasmParticle {
    WasmVec3<T> position;
    WasmVec3<T> velocity;
    WasmVec3<T> force;
    T mass;
    T radius;
    int material_id;
    bool active;

    WasmParticle() : mass(1.0), radius(0.1), material_id(0), active(true) {}
};

// Material properties
template<typename T>
struct WasmMaterial {
    T density;
    T youngs_modulus;
    T poisson_ratio;
    T yield_stress;
    T viscosity;
    T thermal_conductivity;
    int material_type; // 0: elastic, 1: plastic, 2: fluid

    WasmMaterial() :
        density(1000.0),
        youngs_modulus(1e6),
        poisson_ratio(0.3),
        yield_stress(1e4),
        viscosity(0.001),
        thermal_conductivity(0.5),
        material_type(0) {}
};

// Grid cell for spatial partitioning
template<typename T>
struct WasmGridCell {
    std::vector<int> particle_indices;
    WasmVec3<T> momentum;
    T mass;

    void reset() {
        particle_indices.clear();
        momentum = WasmVec3<T>();
        mass = 0;
    }
};

// Simplified physics engine for WebAssembly
template<typename T>
class WasmPhysicsEngine {
private:
    std::vector<WasmParticle<T>> particles_;
    std::vector<WasmMaterial<T>> materials_;
    std::vector<WasmGridCell<T>> grid_;

    WasmVec3<T> gravity_;
    T timestep_;
    T grid_spacing_;
    int grid_resolution_;
    WasmVec3<T> domain_min_;
    WasmVec3<T> domain_max_;

    size_t max_particles_;
    bool use_simd_;

    // Performance metrics
    size_t frame_count_;
    double total_time_;
    double last_fps_;

public:
    WasmPhysicsEngine(size_t max_particles = 10000) :
        gravity_(0, -9.81, 0),
        timestep_(0.001),
        grid_spacing_(0.1),
        grid_resolution_(50),
        domain_min_(-5, -5, -5),
        domain_max_(5, 5, 5),
        max_particles_(max_particles),
        use_simd_(false),
        frame_count_(0),
        total_time_(0),
        last_fps_(0) {

        particles_.reserve(max_particles);
        materials_.push_back(WasmMaterial<T>());

        size_t grid_size = grid_resolution_ * grid_resolution_ * grid_resolution_;
        grid_.resize(grid_size);
    }

    // Add a particle to the simulation
    int addParticle(const WasmVec3<T>& position, const WasmVec3<T>& velocity) {
        if (particles_.size() >= max_particles_) {
            return -1;
        }

        WasmParticle<T> p;
        p.position = position;
        p.velocity = velocity;
        p.force = WasmVec3<T>();
        particles_.push_back(p);

        return static_cast<int>(particles_.size() - 1);
    }

    // Add multiple particles at once
    void addParticleBlock(const WasmVec3<T>& corner, const WasmVec3<T>& dimensions,
                         int nx, int ny, int nz, int material_id = 0) {
        T dx = dimensions.x / nx;
        T dy = dimensions.y / ny;
        T dz = dimensions.z / nz;

        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    WasmVec3<T> pos(
                        corner.x + (i + 0.5) * dx,
                        corner.y + (j + 0.5) * dy,
                        corner.z + (k + 0.5) * dz
                    );

                    if (particles_.size() < max_particles_) {
                        WasmParticle<T> p;
                        p.position = pos;
                        p.velocity = WasmVec3<T>();
                        p.material_id = material_id;
                        particles_.push_back(p);
                    }
                }
            }
        }
    }

    // Main simulation step
    void step() {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Clear grid
        for (auto& cell : grid_) {
            cell.reset();
        }

        // Particle to grid (P2G)
        particleToGrid();

        // Update grid forces
        updateGrid();

        // Grid to particle (G2P)
        gridToParticle();

        // Apply boundary conditions
        applyBoundaries();

        // Update performance metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        total_time_ += diff.count();
        frame_count_++;

        if (frame_count_ % 60 == 0) {
            last_fps_ = 60.0 / total_time_;
            total_time_ = 0;
        }
    }

private:
    // Convert position to grid index
    int positionToGridIndex(const WasmVec3<T>& pos) {
        int ix = static_cast<int>((pos.x - domain_min_.x) / grid_spacing_);
        int iy = static_cast<int>((pos.y - domain_min_.y) / grid_spacing_);
        int iz = static_cast<int>((pos.z - domain_min_.z) / grid_spacing_);

        ix = std::max(0, std::min(grid_resolution_ - 1, ix));
        iy = std::max(0, std::min(grid_resolution_ - 1, iy));
        iz = std::max(0, std::min(grid_resolution_ - 1, iz));

        return ix + iy * grid_resolution_ + iz * grid_resolution_ * grid_resolution_;
    }

    // Particle to grid transfer
    void particleToGrid() {
        for (size_t i = 0; i < particles_.size(); ++i) {
            if (!particles_[i].active) continue;

            const auto& p = particles_[i];
            int grid_idx = positionToGridIndex(p.position);

            if (grid_idx >= 0 && static_cast<size_t>(grid_idx) < grid_.size()) {
                grid_[grid_idx].particle_indices.push_back(i);
                grid_[grid_idx].mass += p.mass;
                grid_[grid_idx].momentum = grid_[grid_idx].momentum +
                                          p.velocity * p.mass;
            }
        }
    }

    // Update grid with forces
    void updateGrid() {
        for (auto& cell : grid_) {
            if (cell.mass > 0) {
                // Apply gravity
                cell.momentum = cell.momentum + gravity_ * (cell.mass * timestep_);

                // Apply material constitutive model (simplified)
                // This would be expanded for full MPM
            }
        }
    }

    // Grid to particle transfer
    void gridToParticle() {
        for (size_t i = 0; i < particles_.size(); ++i) {
            if (!particles_[i].active) continue;

            auto& p = particles_[i];
            int grid_idx = positionToGridIndex(p.position);

            if (grid_idx >= 0 && static_cast<size_t>(grid_idx) < grid_.size() && grid_[grid_idx].mass > 0) {
                // Update velocity from grid momentum
                WasmVec3<T> grid_velocity = grid_[grid_idx].momentum * (1.0 / grid_[grid_idx].mass);

                // Blend with particle velocity (FLIP/PIC mixing)
                T flip_ratio = 0.95;
                p.velocity = p.velocity * flip_ratio + grid_velocity * (1.0 - flip_ratio);

                // Update position
                p.position = p.position + p.velocity * timestep_;
            }
        }
    }

    // Apply boundary conditions
    void applyBoundaries() {
        for (auto& p : particles_) {
            if (!p.active) continue;

            // Box boundaries with restitution
            T restitution = 0.5;

            if (p.position.x < domain_min_.x) {
                p.position.x = domain_min_.x;
                p.velocity.x *= -restitution;
            }
            if (p.position.x > domain_max_.x) {
                p.position.x = domain_max_.x;
                p.velocity.x *= -restitution;
            }

            if (p.position.y < domain_min_.y) {
                p.position.y = domain_min_.y;
                p.velocity.y *= -restitution;
            }
            if (p.position.y > domain_max_.y) {
                p.position.y = domain_max_.y;
                p.velocity.y *= -restitution;
            }

            if (p.position.z < domain_min_.z) {
                p.position.z = domain_min_.z;
                p.velocity.z *= -restitution;
            }
            if (p.position.z > domain_max_.z) {
                p.position.z = domain_max_.z;
                p.velocity.z *= -restitution;
            }
        }
    }

public:
    // Getters for JavaScript access
    std::vector<T> getParticlePositions() {
        std::vector<T> positions;
        positions.reserve(particles_.size() * 3);

        for (const auto& p : particles_) {
            if (p.active) {
                positions.push_back(p.position.x);
                positions.push_back(p.position.y);
                positions.push_back(p.position.z);
            }
        }

        return positions;
    }

    std::vector<T> getParticleVelocities() {
        std::vector<T> velocities;
        velocities.reserve(particles_.size() * 3);

        for (const auto& p : particles_) {
            if (p.active) {
                velocities.push_back(p.velocity.x);
                velocities.push_back(p.velocity.y);
                velocities.push_back(p.velocity.z);
            }
        }

        return velocities;
    }

    size_t getParticleCount() const { return particles_.size(); }
    T getTimestep() const { return timestep_; }
    void setTimestep(T dt) { timestep_ = dt; }

    void setGravity(T gx, T gy, T gz) {
        gravity_ = WasmVec3<T>(gx, gy, gz);
    }

    double getFPS() const { return last_fps_; }

    void reset() {
        particles_.clear();
        for (auto& cell : grid_) {
            cell.reset();
        }
        frame_count_ = 0;
        total_time_ = 0;
        last_fps_ = 0;
    }

    // Enable SIMD optimizations if available
    void enableSIMD(bool enable) {
        use_simd_ = enable;
        #ifdef __EMSCRIPTEN_SIMD__
        if (enable) {
            // SIMD optimizations would be implemented here
        }
        #endif
    }
};

// JavaScript interface wrapper
class WasmInterface {
private:
    std::unique_ptr<WasmPhysicsEngine<float>> engine_;
    bool running_;

public:
    WasmInterface() : running_(false) {
        engine_ = std::make_unique<WasmPhysicsEngine<float>>(50000);
    }

    void initialize(int max_particles) {
        engine_ = std::make_unique<WasmPhysicsEngine<float>>(max_particles);
    }

    void addParticle(float x, float y, float z, float vx, float vy, float vz) {
        engine_->addParticle(WasmVec3<float>(x, y, z), WasmVec3<float>(vx, vy, vz));
    }

    void addBlock(float x, float y, float z,
                 float w, float h, float d,
                 int nx, int ny, int nz) {
        engine_->addParticleBlock(
            WasmVec3<float>(x, y, z),
            WasmVec3<float>(w, h, d),
            nx, ny, nz
        );
    }

    void step() {
        if (engine_) {
            engine_->step();
        }
    }

    void start() { running_ = true; }
    void stop() { running_ = false; }
    bool isRunning() const { return running_; }

    std::vector<float> getPositions() {
        return engine_->getParticlePositions();
    }

    std::vector<float> getVelocities() {
        return engine_->getParticleVelocities();
    }

    int getParticleCount() {
        return static_cast<int>(engine_->getParticleCount());
    }

    void setGravity(float x, float y, float z) {
        engine_->setGravity(x, y, z);
    }

    void setTimestep(float dt) {
        engine_->setTimestep(dt);
    }

    float getFPS() {
        return static_cast<float>(engine_->getFPS());
    }

    void reset() {
        engine_->reset();
    }

    void enableSIMD(bool enable) {
        engine_->enableSIMD(enable);
    }
};

// Memory management utilities
class WasmMemoryManager {
private:
    static size_t allocated_bytes_;
    static size_t peak_bytes_;
    static std::unordered_map<void*, size_t> allocations_;

public:
    static void* allocate(size_t size) {
        void* ptr = malloc(size);
        if (ptr) {
            allocations_[ptr] = size;
            allocated_bytes_ += size;
            peak_bytes_ = std::max(peak_bytes_, allocated_bytes_);
        }
        return ptr;
    }

    static void deallocate(void* ptr) {
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            allocated_bytes_ -= it->second;
            allocations_.erase(it);
        }
        free(ptr);
    }

    static size_t getAllocatedBytes() { return allocated_bytes_; }
    static size_t getPeakBytes() { return peak_bytes_; }
    static size_t getAllocationCount() { return allocations_.size(); }

    static void reset() {
        allocated_bytes_ = 0;
        peak_bytes_ = 0;
        allocations_.clear();
    }
};

// Static member initialization
size_t WasmMemoryManager::allocated_bytes_ = 0;
size_t WasmMemoryManager::peak_bytes_ = 0;
std::unordered_map<void*, size_t> WasmMemoryManager::allocations_;

} // namespace wasm
} // namespace physgrad

// Emscripten bindings for JavaScript
#ifdef __EMSCRIPTEN__

using namespace physgrad::wasm;

EMSCRIPTEN_BINDINGS(physgrad_wasm) {
    // Vector type
    emscripten::value_object<WasmVec3<float>>("Vec3")
        .field("x", &WasmVec3<float>::x)
        .field("y", &WasmVec3<float>::y)
        .field("z", &WasmVec3<float>::z);

    // Main interface
    emscripten::class_<WasmInterface>("PhysicsEngine")
        .constructor<>()
        .function("initialize", &WasmInterface::initialize)
        .function("addParticle", &WasmInterface::addParticle)
        .function("addBlock", &WasmInterface::addBlock)
        .function("step", &WasmInterface::step)
        .function("start", &WasmInterface::start)
        .function("stop", &WasmInterface::stop)
        .function("isRunning", &WasmInterface::isRunning)
        .function("getPositions", &WasmInterface::getPositions)
        .function("getVelocities", &WasmInterface::getVelocities)
        .function("getParticleCount", &WasmInterface::getParticleCount)
        .function("setGravity", &WasmInterface::setGravity)
        .function("setTimestep", &WasmInterface::setTimestep)
        .function("getFPS", &WasmInterface::getFPS)
        .function("reset", &WasmInterface::reset)
        .function("enableSIMD", &WasmInterface::enableSIMD);

    // Memory management
    emscripten::class_<WasmMemoryManager>("MemoryManager")
        .class_function("getAllocatedBytes", &WasmMemoryManager::getAllocatedBytes)
        .class_function("getPeakBytes", &WasmMemoryManager::getPeakBytes)
        .class_function("getAllocationCount", &WasmMemoryManager::getAllocationCount)
        .class_function("reset", &WasmMemoryManager::reset);

    // Register vector types
    emscripten::register_vector<float>("FloatVector");
    emscripten::register_vector<int>("IntVector");
}

#endif