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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <atomic>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace physgrad {
namespace digitaltwin {

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
struct SensorReading {
    std::string sensor_id;
    T timestamp;
    Vec3<T> position;
    Vec3<T> velocity;
    Vec3<T> acceleration;
    T temperature;
    T pressure;
    std::unordered_map<std::string, T> custom_data;

    SensorReading() : timestamp(0), temperature(0), pressure(0) {}

    SensorReading(const std::string& id, T time, const Vec3<T>& pos)
        : sensor_id(id), timestamp(time), position(pos),
          velocity(Vec3<T>()), acceleration(Vec3<T>()), temperature(0), pressure(0) {}
};

template<typename T>
class SensorModel {
private:
    std::string sensor_type_;
    T noise_stddev_;
    T sampling_rate_;
    T latency_;
    bool is_active_;

public:
    SensorModel(const std::string& type, T noise = 0.01, T rate = 100.0, T latency = 0.001)
        : sensor_type_(type), noise_stddev_(noise), sampling_rate_(rate),
          latency_(latency), is_active_(true) {}

    SensorReading<T> generateReading(const std::string& id, T timestamp,
                                   const Vec3<T>& true_position,
                                   const Vec3<T>& true_velocity = Vec3<T>(),
                                   T true_temperature = 20.0) {
        SensorReading<T> reading(id, timestamp, true_position);

        if (is_active_) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> noise(0.0, noise_stddev_);

            // Add noise to measurements
            reading.position.x += noise(gen);
            reading.position.y += noise(gen);
            reading.position.z += noise(gen);

            reading.velocity = true_velocity;
            reading.velocity.x += noise(gen) * 0.1;
            reading.velocity.y += noise(gen) * 0.1;
            reading.velocity.z += noise(gen) * 0.1;

            reading.temperature = true_temperature + noise(gen);
            reading.pressure = 101325.0 + noise(gen) * 100.0; // Standard atmospheric pressure with noise

            // Add sensor-specific data
            if (sensor_type_ == "IMU") {
                reading.custom_data["angular_velocity_x"] = noise(gen);
                reading.custom_data["angular_velocity_y"] = noise(gen);
                reading.custom_data["angular_velocity_z"] = noise(gen);
            } else if (sensor_type_ == "GPS") {
                reading.custom_data["altitude"] = true_position.z + noise(gen);
                reading.custom_data["satellites"] = 8 + static_cast<int>(noise(gen) * 2);
            }
        }

        return reading;
    }

    T getSamplingRate() const { return sampling_rate_; }
    T getLatency() const { return latency_; }
    bool isActive() const { return is_active_; }
    void setActive(bool active) { is_active_ = active; }

    std::string getSensorType() const { return sensor_type_; }
    T getNoiseLevel() const { return noise_stddev_; }
};

template<typename T>
struct StateVector {
    Vec3<T> position;
    Vec3<T> velocity;
    Vec3<T> acceleration;
    T timestamp;
    std::unordered_map<std::string, T> parameters;

    StateVector() : timestamp(0) {}

    StateVector(const Vec3<T>& pos, const Vec3<T>& vel, T time)
        : position(pos), velocity(vel), acceleration(Vec3<T>()), timestamp(time) {}

    T distance(const StateVector& other) const {
        return position.distance(other.position) +
               velocity.distance(other.velocity) * 0.1; // Weight velocity difference less
    }
};

template<typename T>
class StateEstimator {
private:
    StateVector<T> current_state_;
    StateVector<T> predicted_state_;
    std::vector<StateVector<T>> state_history_;
    size_t max_history_size_;
    T process_noise_;
    T measurement_noise_;

public:
    StateEstimator(size_t history_size = 100, T proc_noise = 0.01, T meas_noise = 0.1)
        : max_history_size_(history_size), process_noise_(proc_noise), measurement_noise_(meas_noise) {}

    void predict(T dt) {
        // Simple kinematic prediction
        predicted_state_ = current_state_;
        predicted_state_.position = predicted_state_.position + predicted_state_.velocity * dt;
        predicted_state_.velocity = predicted_state_.velocity + predicted_state_.acceleration * dt;
        predicted_state_.timestamp = current_state_.timestamp + dt;

        // Add process noise
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> noise(0.0, process_noise_);

        predicted_state_.position.x += noise(gen);
        predicted_state_.position.y += noise(gen);
        predicted_state_.position.z += noise(gen);
    }

    void update(const SensorReading<T>& measurement) {
        // Kalman filter-like update (simplified)
        T innovation_weight = measurement_noise_ / (measurement_noise_ + process_noise_);

        current_state_.position.x = predicted_state_.position.x +
                                   innovation_weight * (measurement.position.x - predicted_state_.position.x);
        current_state_.position.y = predicted_state_.position.y +
                                   innovation_weight * (measurement.position.y - predicted_state_.position.y);
        current_state_.position.z = predicted_state_.position.z +
                                   innovation_weight * (measurement.position.z - predicted_state_.position.z);

        current_state_.velocity = measurement.velocity;
        current_state_.acceleration = measurement.acceleration;
        current_state_.timestamp = measurement.timestamp;

        // Store in history
        state_history_.push_back(current_state_);
        if (state_history_.size() > max_history_size_) {
            state_history_.erase(state_history_.begin());
        }
    }

    const StateVector<T>& getCurrentState() const { return current_state_; }
    const StateVector<T>& getPredictedState() const { return predicted_state_; }

    std::vector<StateVector<T>> getStateHistory() const { return state_history_; }

    T getEstimationError(const StateVector<T>& ground_truth) const {
        return current_state_.distance(ground_truth);
    }

    void reset() {
        current_state_ = StateVector<T>();
        predicted_state_ = StateVector<T>();
        state_history_.clear();
    }
};

template<typename T>
struct CalibrationParameter {
    std::string name;
    T value;
    T min_value;
    T max_value;
    T uncertainty;
    bool is_calibrating;

    CalibrationParameter(const std::string& n, T val, T min_val, T max_val)
        : name(n), value(val), min_value(min_val), max_value(max_val),
          uncertainty(std::abs(max_val - min_val) * 0.1), is_calibrating(true) {}
};

template<typename T>
class ParameterCalibrator {
private:
    std::vector<CalibrationParameter<T>> parameters_;
    std::vector<SensorReading<T>> calibration_data_;
    T learning_rate_;
    size_t max_iterations_;
    T convergence_threshold_;

public:
    ParameterCalibrator(T lr = 0.01, size_t max_iter = 100, T threshold = 1e-6)
        : learning_rate_(lr), max_iterations_(max_iter), convergence_threshold_(threshold) {}

    void addParameter(const CalibrationParameter<T>& param) {
        parameters_.push_back(param);
    }

    void addCalibrationData(const SensorReading<T>& data) {
        calibration_data_.push_back(data);
    }

    T computeObjective(const std::vector<SensorReading<T>>& predicted_data) const {
        if (predicted_data.size() != calibration_data_.size() || calibration_data_.empty()) {
            return std::numeric_limits<T>::infinity();
        }

        T total_error = 0;
        for (size_t i = 0; i < calibration_data_.size(); ++i) {
            T pos_error = calibration_data_[i].position.distance(predicted_data[i].position);
            T vel_error = calibration_data_[i].velocity.distance(predicted_data[i].velocity);
            T temp_error = std::abs(calibration_data_[i].temperature - predicted_data[i].temperature);

            // Check for invalid values
            if (std::isnan(pos_error) || std::isnan(vel_error) || std::isnan(temp_error) ||
                std::isinf(pos_error) || std::isinf(vel_error) || std::isinf(temp_error)) {
                return std::numeric_limits<T>::infinity();
            }

            total_error += pos_error + vel_error * 0.1 + temp_error * 0.01;
        }

        T result = total_error / calibration_data_.size();
        return std::isfinite(result) ? result : std::numeric_limits<T>::infinity();
    }

    std::vector<T> computeGradient(const std::vector<SensorReading<T>>& predicted_data) const {
        std::vector<T> gradients(parameters_.size(), 0);
        T eps = 1e-5;

        T base_objective = computeObjective(predicted_data);

        for (size_t i = 0; i < parameters_.size(); ++i) {
            if (!parameters_[i].is_calibrating) continue;

            // Finite difference approximation
            std::vector<SensorReading<T>> perturbed_data = predicted_data;

            // This is a simplified gradient computation
            // In practice, this would require running the simulation with perturbed parameters
            for (auto& reading : perturbed_data) {
                if (parameters_[i].name == "mass") {
                    reading.acceleration = reading.acceleration * (1.0 + eps);
                } else if (parameters_[i].name == "damping") {
                    reading.velocity = reading.velocity * (1.0 - eps);
                } else if (parameters_[i].name == "spring_constant") {
                    reading.position = reading.position * (1.0 + eps * 0.1);
                }
            }

            T perturbed_objective = computeObjective(perturbed_data);
            gradients[i] = (perturbed_objective - base_objective) / eps;
        }

        return gradients;
    }

    bool calibrate(std::function<std::vector<SensorReading<T>>(const std::vector<CalibrationParameter<T>>&)> simulate) {
        if (calibration_data_.empty()) {
            std::cout << "No calibration data available" << std::endl;
            return false;
        }

        T prev_objective = std::numeric_limits<T>::infinity();

        for (size_t iter = 0; iter < max_iterations_; ++iter) {
            // Run simulation with current parameters
            auto predicted_data = simulate(parameters_);

            T current_objective = computeObjective(predicted_data);
            auto gradients = computeGradient(predicted_data);

            // Update parameters using gradient descent
            for (size_t i = 0; i < parameters_.size(); ++i) {
                if (!parameters_[i].is_calibrating) continue;

                T update = -learning_rate_ * gradients[i];
                parameters_[i].value += update;

                // Clamp to bounds
                parameters_[i].value = std::clamp(parameters_[i].value,
                                                parameters_[i].min_value,
                                                parameters_[i].max_value);

                // Update uncertainty
                parameters_[i].uncertainty = std::abs(update);
            }

            if (iter % 10 == 0) {
                std::cout << "Calibration iteration " << iter
                          << ": Objective = " << current_objective << std::endl;
            }

            // Check convergence
            if (std::abs(prev_objective - current_objective) < convergence_threshold_) {
                std::cout << "Calibration converged after " << iter << " iterations" << std::endl;
                return true;
            }

            prev_objective = current_objective;
        }

        std::cout << "Calibration reached maximum iterations" << std::endl;
        return false;
    }

    const std::vector<CalibrationParameter<T>>& getParameters() const { return parameters_; }

    void setParameter(const std::string& name, T value) {
        for (auto& param : parameters_) {
            if (param.name == name) {
                param.value = value;
                break;
            }
        }
    }

    T getParameter(const std::string& name) const {
        for (const auto& param : parameters_) {
            if (param.name == name) {
                return param.value;
            }
        }
        return 0;
    }

    void clearCalibrationData() {
        calibration_data_.clear();
    }
};

template<typename T>
class RealTimeSimulator {
private:
    std::atomic<bool> is_running_;
    std::atomic<T> simulation_time_;
    std::atomic<T> time_step_;
    std::thread simulation_thread_;
    mutable std::mutex state_mutex_;
    StateVector<T> current_state_;
    std::queue<SensorReading<T>> sensor_buffer_;
    std::function<StateVector<T>(const StateVector<T>&, T, const std::vector<CalibrationParameter<T>>&)> physics_step_;
    std::vector<CalibrationParameter<T>> parameters_;
    size_t max_buffer_size_;

public:
    RealTimeSimulator(T dt = 0.001, size_t buffer_size = 1000)
        : is_running_(false), simulation_time_(0), time_step_(dt), max_buffer_size_(buffer_size) {}

    ~RealTimeSimulator() {
        stop();
    }

    void setPhysicsStep(std::function<StateVector<T>(const StateVector<T>&, T, const std::vector<CalibrationParameter<T>>&)> step) {
        physics_step_ = step;
    }

    void setParameters(const std::vector<CalibrationParameter<T>>& params) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        parameters_ = params;
    }

    void start(const StateVector<T>& initial_state) {
        if (is_running_.load()) {
            return;
        }

        if (!physics_step_) {
            throw std::runtime_error("Physics step function not set");
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_state_ = initial_state;
            simulation_time_ = initial_state.timestamp;
        }

        is_running_ = true;
        simulation_thread_ = std::thread(&RealTimeSimulator::simulationLoop, this);
    }

    void stop() {
        is_running_ = false;
        if (simulation_thread_.joinable()) {
            simulation_thread_.join();
        }
    }

    StateVector<T> getCurrentState() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return current_state_;
    }

    T getSimulationTime() const {
        return simulation_time_.load();
    }

    void addSensorReading(const SensorReading<T>& reading) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        sensor_buffer_.push(reading);

        if (sensor_buffer_.size() > max_buffer_size_) {
            sensor_buffer_.pop();
        }
    }

    std::vector<SensorReading<T>> getSensorBuffer() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        std::vector<SensorReading<T>> readings;
        auto buffer_copy = sensor_buffer_;

        while (!buffer_copy.empty()) {
            readings.push_back(buffer_copy.front());
            buffer_copy.pop();
        }

        return readings;
    }

    bool isRunning() const {
        return is_running_.load();
    }

    void setTimeStep(T dt) {
        time_step_ = dt;
    }

    T getTimeStep() const {
        return time_step_.load();
    }

private:
    void simulationLoop() {
        auto last_time = std::chrono::high_resolution_clock::now();

        while (is_running_.load()) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_time);
            T real_dt = elapsed.count() / 1e6;

            T dt = time_step_.load();

            {
                std::lock_guard<std::mutex> lock(state_mutex_);

                // Process any pending sensor readings
                if (!sensor_buffer_.empty()) {
                    // Simple sensor fusion - use most recent reading to correct state
                    auto latest_reading = sensor_buffer_.back();

                    // Apply correction based on sensor data
                    T correction_weight = 0.1;
                    current_state_.position = current_state_.position * (1.0 - correction_weight) +
                                            latest_reading.position * correction_weight;
                }

                // Advance physics simulation
                current_state_ = physics_step_(current_state_, dt, parameters_);
                simulation_time_.store(simulation_time_.load() + dt);
            }

            last_time = current_time;

            // Control simulation rate
            std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(dt * 1e6)));
        }
    }
};

template<typename T>
class DigitalTwinFramework {
private:
    std::unique_ptr<RealTimeSimulator<T>> simulator_;
    std::unique_ptr<StateEstimator<T>> estimator_;
    std::unique_ptr<ParameterCalibrator<T>> calibrator_;
    std::vector<std::unique_ptr<SensorModel<T>>> sensors_;
    std::string twin_id_;
    bool is_calibrated_;

public:
    DigitalTwinFramework(const std::string& id, T dt = 0.001)
        : twin_id_(id), is_calibrated_(false) {
        simulator_ = std::make_unique<RealTimeSimulator<T>>(dt);
        estimator_ = std::make_unique<StateEstimator<T>>();
        calibrator_ = std::make_unique<ParameterCalibrator<T>>();
    }

    void addSensor(std::unique_ptr<SensorModel<T>> sensor) {
        sensors_.push_back(std::move(sensor));
    }

    void setPhysicsModel(std::function<StateVector<T>(const StateVector<T>&, T, const std::vector<CalibrationParameter<T>>&)> physics) {
        simulator_->setPhysicsStep(physics);
    }

    void addCalibrationParameter(const CalibrationParameter<T>& param) {
        calibrator_->addParameter(param);
    }

    bool calibrate(const std::vector<SensorReading<T>>& real_data) {
        std::cout << "Starting digital twin calibration for " << twin_id_ << std::endl;

        // Add calibration data
        for (const auto& reading : real_data) {
            calibrator_->addCalibrationData(reading);
        }

        // Define simulation function for calibration
        auto simulate_for_calibration = [this](const std::vector<CalibrationParameter<T>>& params) -> std::vector<SensorReading<T>> {
            std::vector<SensorReading<T>> simulated_data;

            // Simple simulation for calibration
            StateVector<T> state;
            state.position = Vec3<T>(0, 0, 0);
            state.velocity = Vec3<T>(1, 0, 0);
            state.timestamp = 0;

            T dt = 0.01;
            for (int step = 0; step < 100; ++step) {
                // Simple physics with parameters
                T mass = 1.0;
                T damping = 0.1;
                T spring_k = 10.0;

                for (const auto& param : params) {
                    if (param.name == "mass") mass = param.value;
                    else if (param.name == "damping") damping = param.value;
                    else if (param.name == "spring_constant") spring_k = param.value;
                }

                // Simple spring-damper dynamics
                Vec3<T> force = state.position * (-spring_k) + state.velocity * (-damping);
                state.acceleration = force * (1.0 / mass);
                state.velocity = state.velocity + state.acceleration * dt;
                state.position = state.position + state.velocity * dt;
                state.timestamp += dt;

                // Generate sensor reading
                if (step % 10 == 0) {
                    SensorReading<T> reading("sim", state.timestamp, state.position);
                    reading.velocity = state.velocity;
                    reading.acceleration = state.acceleration;
                    reading.temperature = 20.0 + state.position.norm() * 0.1;
                    simulated_data.push_back(reading);
                }
            }

            return simulated_data;
        };

        bool calibration_success = calibrator_->calibrate(simulate_for_calibration);

        if (calibration_success) {
            // Update simulator with calibrated parameters
            simulator_->setParameters(calibrator_->getParameters());
            is_calibrated_ = true;
            std::cout << "Digital twin calibration completed successfully" << std::endl;
        } else {
            std::cout << "Digital twin calibration failed" << std::endl;
        }

        return calibration_success;
    }

    void start(const StateVector<T>& initial_state) {
        if (!is_calibrated_) {
            std::cout << "Warning: Starting uncalibrated digital twin" << std::endl;
        }
        simulator_->start(initial_state);
    }

    void stop() {
        simulator_->stop();
    }

    void processRealTimeData(const SensorReading<T>& real_data) {
        // Add to simulator for sensor fusion
        simulator_->addSensorReading(real_data);

        // Update state estimator
        estimator_->predict(0.001); // Small time step for prediction
        estimator_->update(real_data);
    }

    StateVector<T> getCurrentState() const {
        return simulator_->getCurrentState();
    }

    StateVector<T> getEstimatedState() const {
        return estimator_->getCurrentState();
    }

    std::vector<SensorReading<T>> generateVirtualSensorData(T timestamp) const {
        std::vector<SensorReading<T>> virtual_data;

        StateVector<T> current_state = simulator_->getCurrentState();

        for (const auto& sensor : sensors_) {
            std::string sensor_id = twin_id_ + "_" + sensor->getSensorType();
            auto reading = sensor->generateReading(sensor_id, timestamp, current_state.position,
                                                 current_state.velocity, 20.0);
            virtual_data.push_back(reading);
        }

        return virtual_data;
    }

    T getPredictionAccuracy(const std::vector<SensorReading<T>>& real_data) const {
        if (real_data.empty()) return 0.0;

        T total_error = 0.0;
        for (const auto& real_reading : real_data) {
            StateVector<T> sim_state = simulator_->getCurrentState();
            StateVector<T> real_state(real_reading.position, real_reading.velocity, real_reading.timestamp);
            total_error += sim_state.distance(real_state);
        }

        return total_error / real_data.size();
    }

    void saveConfiguration(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }

        file << "# Digital Twin Configuration: " << twin_id_ << std::endl;
        file << "calibrated: " << (is_calibrated_ ? "true" : "false") << std::endl;
        file << "sensors: " << sensors_.size() << std::endl;

        const auto& params = calibrator_->getParameters();
        file << "parameters: " << params.size() << std::endl;
        for (const auto& param : params) {
            file << param.name << " " << param.value << " "
                 << param.uncertainty << " " << (param.is_calibrating ? "true" : "false") << std::endl;
        }

        file.close();
    }

    bool isCalibrated() const { return is_calibrated_; }
    bool isRunning() const { return simulator_->isRunning(); }
    std::string getTwinId() const { return twin_id_; }

    void resetCalibration() {
        calibrator_->clearCalibrationData();
        estimator_->reset();
        is_calibrated_ = false;
    }

    const std::vector<CalibrationParameter<T>>& getCalibrationParameters() const {
        return calibrator_->getParameters();
    }

    size_t getSensorCount() const { return sensors_.size(); }

    T getSimulationTime() const {
        return simulator_->getSimulationTime();
    }
};

template<typename T>
class DigitalTwinManager {
private:
    std::unordered_map<std::string, std::unique_ptr<DigitalTwinFramework<T>>> twins_;
    mutable std::mutex twins_mutex_;

public:
    void addTwin(const std::string& id, std::unique_ptr<DigitalTwinFramework<T>> twin) {
        std::lock_guard<std::mutex> lock(twins_mutex_);
        twins_[id] = std::move(twin);
    }

    DigitalTwinFramework<T>* getTwin(const std::string& id) {
        std::lock_guard<std::mutex> lock(twins_mutex_);
        auto it = twins_.find(id);
        return (it != twins_.end()) ? it->second.get() : nullptr;
    }

    bool removeTwin(const std::string& id) {
        std::lock_guard<std::mutex> lock(twins_mutex_);
        auto it = twins_.find(id);
        if (it != twins_.end()) {
            it->second->stop();
            twins_.erase(it);
            return true;
        }
        return false;
    }

    std::vector<std::string> getTwinIds() const {
        std::lock_guard<std::mutex> lock(twins_mutex_);
        std::vector<std::string> ids;
        for (const auto& pair : twins_) {
            ids.push_back(pair.first);
        }
        return ids;
    }

    void startAllTwins() {
        std::lock_guard<std::mutex> lock(twins_mutex_);
        for (auto& pair : twins_) {
            if (!pair.second->isRunning()) {
                StateVector<T> initial_state;
                pair.second->start(initial_state);
            }
        }
    }

    void stopAllTwins() {
        std::lock_guard<std::mutex> lock(twins_mutex_);
        for (auto& pair : twins_) {
            pair.second->stop();
        }
    }

    size_t getTwinCount() const {
        std::lock_guard<std::mutex> lock(twins_mutex_);
        return twins_.size();
    }

    void broadcastSensorData(const SensorReading<T>& data) {
        std::lock_guard<std::mutex> lock(twins_mutex_);
        for (auto& pair : twins_) {
            pair.second->processRealTimeData(data);
        }
    }

    std::vector<std::pair<std::string, T>> getAccuracyReport(const std::vector<SensorReading<T>>& real_data) const {
        std::lock_guard<std::mutex> lock(twins_mutex_);
        std::vector<std::pair<std::string, T>> report;

        for (const auto& pair : twins_) {
            T accuracy = pair.second->getPredictionAccuracy(real_data);
            report.emplace_back(pair.first, accuracy);
        }

        return report;
    }
};

} // namespace digitaltwin
} // namespace physgrad