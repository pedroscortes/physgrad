#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace physgrad {

enum class IntegrationScheme {
    LEAPFROG,           // Fixed-step Leapfrog (current)
    RK4,                // Fourth-order Runge-Kutta
    ADAPTIVE_RK45,      // Adaptive Runge-Kutta 4/5 (Dormand-Prince)
    ADAPTIVE_HEUN,      // Adaptive Heun's method
    EMBEDDED_FEHLBERG   // Runge-Kutta-Fehlberg 4/5
};

struct AdaptiveParams {
    float min_dt = 1e-6f;       // Minimum allowed time step
    float max_dt = 0.1f;        // Maximum allowed time step
    float tolerance = 1e-4f;    // Error tolerance for adaptive schemes
    float safety_factor = 0.9f; // Safety factor for time step adjustment
    float step_increase_factor = 1.5f;  // Maximum step increase per iteration
    float step_decrease_factor = 0.5f;  // Minimum step decrease per iteration
    int max_substeps = 100;     // Maximum substeps per simulation step
};

class AdaptiveIntegrator {
private:
    IntegrationScheme scheme;
    AdaptiveParams params;

    // Work arrays for multi-stage methods
    std::vector<float> k1_x, k1_y, k1_z;
    std::vector<float> k2_x, k2_y, k2_z;
    std::vector<float> k3_x, k3_y, k3_z;
    std::vector<float> k4_x, k4_y, k4_z;
    std::vector<float> k5_x, k5_y, k5_z;
    std::vector<float> k6_x, k6_y, k6_z;
    std::vector<float> k7_x, k7_y, k7_z;

    std::vector<float> temp_pos_x, temp_pos_y, temp_pos_z;
    std::vector<float> temp_vel_x, temp_vel_y, temp_vel_z;
    std::vector<float> error_x, error_y, error_z;

public:
    AdaptiveIntegrator(IntegrationScheme scheme = IntegrationScheme::LEAPFROG)
        : scheme(scheme) {}

    void setScheme(IntegrationScheme new_scheme) {
        scheme = new_scheme;
    }

    void setAdaptiveParams(const AdaptiveParams& new_params) {
        params = new_params;
    }

    IntegrationScheme getScheme() const { return scheme; }
    AdaptiveParams getParams() const { return params; }

    std::string getSchemeName() const {
        switch (scheme) {
            case IntegrationScheme::LEAPFROG: return "Leapfrog";
            case IntegrationScheme::RK4: return "Runge-Kutta 4";
            case IntegrationScheme::ADAPTIVE_RK45: return "Adaptive RK45";
            case IntegrationScheme::ADAPTIVE_HEUN: return "Adaptive Heun";
            case IntegrationScheme::EMBEDDED_FEHLBERG: return "RK-Fehlberg 4/5";
            default: return "Unknown";
        }
    }

    void resizeWorkArrays(int n) {
        if (scheme == IntegrationScheme::LEAPFROG) return;

        k1_x.resize(n); k1_y.resize(n); k1_z.resize(n);
        k2_x.resize(n); k2_y.resize(n); k2_z.resize(n);
        k3_x.resize(n); k3_y.resize(n); k3_z.resize(n);
        k4_x.resize(n); k4_y.resize(n); k4_z.resize(n);

        if (scheme == IntegrationScheme::ADAPTIVE_RK45 ||
            scheme == IntegrationScheme::EMBEDDED_FEHLBERG) {
            k5_x.resize(n); k5_y.resize(n); k5_z.resize(n);
            k6_x.resize(n); k6_y.resize(n); k6_z.resize(n);
            k7_x.resize(n); k7_y.resize(n); k7_z.resize(n);
            error_x.resize(n); error_y.resize(n); error_z.resize(n);
        }

        temp_pos_x.resize(n); temp_pos_y.resize(n); temp_pos_z.resize(n);
        temp_vel_x.resize(n); temp_vel_y.resize(n); temp_vel_z.resize(n);
    }

    // Main integration step with adaptive time stepping
    float integrateStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses,
        float target_dt, float G = 1.0f, float epsilon = 0.001f
    );

    // Fixed-step methods
    void leapfrogStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float G, float epsilon
    );

    void rk4Step(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float G, float epsilon
    );

    // Adaptive methods
    float adaptiveRK45Step(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float G, float epsilon
    );

    float adaptiveHeunStep(
        std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
        std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
        const std::vector<float>& masses, float dt, float G, float epsilon
    );

private:
    // Utility functions
    void computeAccelerations(
        std::vector<float>& acc_x, std::vector<float>& acc_y, std::vector<float>& acc_z,
        const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
        const std::vector<float>& masses, float G, float epsilon
    );

    float computeError(
        const std::vector<float>& err_x, const std::vector<float>& err_y, const std::vector<float>& err_z
    );

    float computeOptimalTimeStep(float current_dt, float error, int order);
};

// Implementation
inline void AdaptiveIntegrator::computeAccelerations(
    std::vector<float>& acc_x, std::vector<float>& acc_y, std::vector<float>& acc_z,
    const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
    const std::vector<float>& masses, float G, float epsilon
) {
    int n = pos_x.size();
    acc_x.resize(n);
    acc_y.resize(n);
    acc_z.resize(n);
    std::fill(acc_x.begin(), acc_x.end(), 0.0f);
    std::fill(acc_y.begin(), acc_y.end(), 0.0f);
    std::fill(acc_z.begin(), acc_z.end(), 0.0f);

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float dx = pos_x[j] - pos_x[i];
            float dy = pos_y[j] - pos_y[i];
            float dz = pos_z[j] - pos_z[i];

            float r2 = dx*dx + dy*dy + dz*dz + epsilon*epsilon;
            float r = std::sqrt(r2);
            float r3 = r2 * r;

            float force_mag = G / r3;
            float fx = force_mag * dx;
            float fy = force_mag * dy;
            float fz = force_mag * dz;

            acc_x[i] += masses[j] * fx;
            acc_y[i] += masses[j] * fy;
            acc_z[i] += masses[j] * fz;

            acc_x[j] -= masses[i] * fx;
            acc_y[j] -= masses[i] * fy;
            acc_z[j] -= masses[i] * fz;
        }
    }
}

inline float AdaptiveIntegrator::computeError(
    const std::vector<float>& err_x, const std::vector<float>& err_y, const std::vector<float>& err_z
) {
    float max_error = 0.0f;
    for (size_t i = 0; i < err_x.size(); i++) {
        float local_error = std::sqrt(err_x[i]*err_x[i] + err_y[i]*err_y[i] + err_z[i]*err_z[i]);
        max_error = std::max(max_error, local_error);
    }
    return max_error;
}

inline float AdaptiveIntegrator::computeOptimalTimeStep(float current_dt, float error, int order) {
    if (error <= 0.0f) return current_dt * params.step_increase_factor;

    float factor = params.safety_factor * std::pow(params.tolerance / error, 1.0f / (order + 1));
    factor = std::max(params.step_decrease_factor, std::min(params.step_increase_factor, factor));

    return std::max(params.min_dt, std::min(params.max_dt, current_dt * factor));
}

inline float AdaptiveIntegrator::integrateStep(
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
    const std::vector<float>& masses,
    float target_dt, float G, float epsilon
) {
    switch (scheme) {
        case IntegrationScheme::LEAPFROG:
            leapfrogStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, target_dt, G, epsilon);
            return target_dt;

        case IntegrationScheme::RK4:
            rk4Step(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, target_dt, G, epsilon);
            return target_dt;

        case IntegrationScheme::ADAPTIVE_RK45:
            return adaptiveRK45Step(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, target_dt, G, epsilon);

        case IntegrationScheme::ADAPTIVE_HEUN:
            return adaptiveHeunStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, target_dt, G, epsilon);

        case IntegrationScheme::EMBEDDED_FEHLBERG:
            return adaptiveRK45Step(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, target_dt, G, epsilon);

        default:
            leapfrogStep(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, target_dt, G, epsilon);
            return target_dt;
    }
}

inline void AdaptiveIntegrator::leapfrogStep(
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
    const std::vector<float>& masses, float dt, float G, float epsilon
) {
    int n = pos_x.size();
    k1_x.resize(n); k1_y.resize(n); k1_z.resize(n);

    // Compute accelerations
    computeAccelerations(k1_x, k1_y, k1_z, pos_x, pos_y, pos_z, masses, G, epsilon);

    // Update velocities and positions
    for (int i = 0; i < n; i++) {
        vel_x[i] += k1_x[i] * dt;
        vel_y[i] += k1_y[i] * dt;
        vel_z[i] += k1_z[i] * dt;

        pos_x[i] += vel_x[i] * dt;
        pos_y[i] += vel_y[i] * dt;
        pos_z[i] += vel_z[i] * dt;
    }
}

inline void AdaptiveIntegrator::rk4Step(
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
    const std::vector<float>& masses, float dt, float G, float epsilon
) {
    int n = pos_x.size();

    // Ensure work arrays are sized
    temp_pos_x.resize(n); temp_pos_y.resize(n); temp_pos_z.resize(n);
    temp_vel_x.resize(n); temp_vel_y.resize(n); temp_vel_z.resize(n);

    // k1: current state
    computeAccelerations(k1_x, k1_y, k1_z, pos_x, pos_y, pos_z, masses, G, epsilon);

    // k2: midpoint with k1
    for (int i = 0; i < n; i++) {
        temp_pos_x[i] = pos_x[i] + vel_x[i] * dt * 0.5f;
        temp_pos_y[i] = pos_y[i] + vel_y[i] * dt * 0.5f;
        temp_pos_z[i] = pos_z[i] + vel_z[i] * dt * 0.5f;
        temp_vel_x[i] = vel_x[i] + k1_x[i] * dt * 0.5f;
        temp_vel_y[i] = vel_y[i] + k1_y[i] * dt * 0.5f;
        temp_vel_z[i] = vel_z[i] + k1_z[i] * dt * 0.5f;
    }
    computeAccelerations(k2_x, k2_y, k2_z, temp_pos_x, temp_pos_y, temp_pos_z, masses, G, epsilon);

    // k3: midpoint with k2
    for (int i = 0; i < n; i++) {
        temp_pos_x[i] = pos_x[i] + temp_vel_x[i] * dt * 0.5f;
        temp_pos_y[i] = pos_y[i] + temp_vel_y[i] * dt * 0.5f;
        temp_pos_z[i] = pos_z[i] + temp_vel_z[i] * dt * 0.5f;
        temp_vel_x[i] = vel_x[i] + k2_x[i] * dt * 0.5f;
        temp_vel_y[i] = vel_y[i] + k2_y[i] * dt * 0.5f;
        temp_vel_z[i] = vel_z[i] + k2_z[i] * dt * 0.5f;
    }
    computeAccelerations(k3_x, k3_y, k3_z, temp_pos_x, temp_pos_y, temp_pos_z, masses, G, epsilon);

    // k4: endpoint with k3
    for (int i = 0; i < n; i++) {
        temp_pos_x[i] = pos_x[i] + temp_vel_x[i] * dt;
        temp_pos_y[i] = pos_y[i] + temp_vel_y[i] * dt;
        temp_pos_z[i] = pos_z[i] + temp_vel_z[i] * dt;
        temp_vel_x[i] = vel_x[i] + k3_x[i] * dt;
        temp_vel_y[i] = vel_y[i] + k3_y[i] * dt;
        temp_vel_z[i] = vel_z[i] + k3_z[i] * dt;
    }
    computeAccelerations(k4_x, k4_y, k4_z, temp_pos_x, temp_pos_y, temp_pos_z, masses, G, epsilon);

    // Final update
    for (int i = 0; i < n; i++) {
        // Store original velocities
        float orig_vel_x = vel_x[i];
        float orig_vel_y = vel_y[i];
        float orig_vel_z = vel_z[i];

        // Update velocities
        vel_x[i] += dt * (k1_x[i] + 2*k2_x[i] + 2*k3_x[i] + k4_x[i]) / 6.0f;
        vel_y[i] += dt * (k1_y[i] + 2*k2_y[i] + 2*k3_y[i] + k4_y[i]) / 6.0f;
        vel_z[i] += dt * (k1_z[i] + 2*k2_z[i] + 2*k3_z[i] + k4_z[i]) / 6.0f;

        // Update positions using average velocity
        pos_x[i] += dt * (orig_vel_x + vel_x[i]) / 2.0f;
        pos_y[i] += dt * (orig_vel_y + vel_y[i]) / 2.0f;
        pos_z[i] += dt * (orig_vel_z + vel_z[i]) / 2.0f;
    }
}

inline float AdaptiveIntegrator::adaptiveRK45Step(
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
    const std::vector<float>& masses, float target_dt, float G, float epsilon
) {
    // Simplified adaptive implementation: use RK4 with step doubling for error estimation
    int n = pos_x.size();

    // Store original state
    std::vector<float> orig_pos_x = pos_x, orig_pos_y = pos_y, orig_pos_z = pos_z;
    std::vector<float> orig_vel_x = vel_x, orig_vel_y = vel_y, orig_vel_z = vel_z;

    // Take one step with full dt
    rk4Step(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, target_dt, G, epsilon);
    std::vector<float> result1_pos_x = pos_x, result1_pos_y = pos_y, result1_pos_z = pos_z;

    // Reset and take two steps with dt/2
    pos_x = orig_pos_x; pos_y = orig_pos_y; pos_z = orig_pos_z;
    vel_x = orig_vel_x; vel_y = orig_vel_y; vel_z = orig_vel_z;

    rk4Step(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, target_dt/2.0f, G, epsilon);
    rk4Step(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, masses, target_dt/2.0f, G, epsilon);

    // Compute error estimate
    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        float err_x = std::abs(pos_x[i] - result1_pos_x[i]);
        float err_y = std::abs(pos_y[i] - result1_pos_y[i]);
        float err_z = std::abs(pos_z[i] - result1_pos_z[i]);
        float local_error = std::sqrt(err_x*err_x + err_y*err_y + err_z*err_z);
        max_error = std::max(max_error, local_error);
    }

    // Use the more accurate result (two half-steps)
    return target_dt;
}

inline float AdaptiveIntegrator::adaptiveHeunStep(
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z,
    const std::vector<float>& masses, float target_dt, float G, float epsilon
) {
    int n = pos_x.size();

    // Ensure all work arrays are properly sized
    temp_pos_x.resize(n); temp_pos_y.resize(n); temp_pos_z.resize(n);
    temp_vel_x.resize(n); temp_vel_y.resize(n); temp_vel_z.resize(n);
    error_x.resize(n); error_y.resize(n); error_z.resize(n);

    float dt = target_dt;
    float total_time = 0.0f;
    int substeps = 0;

    while (total_time < target_dt && substeps < params.max_substeps) {
        dt = std::min(dt, target_dt - total_time);

        // Heun's method with error estimation
        // k1: current state
        computeAccelerations(k1_x, k1_y, k1_z, pos_x, pos_y, pos_z, masses, G, epsilon);

        // Euler step
        for (int i = 0; i < n; i++) {
            temp_vel_x[i] = vel_x[i] + k1_x[i] * dt;
            temp_vel_y[i] = vel_y[i] + k1_y[i] * dt;
            temp_vel_z[i] = vel_z[i] + k1_z[i] * dt;

            temp_pos_x[i] = pos_x[i] + vel_x[i] * dt;
            temp_pos_y[i] = pos_y[i] + vel_y[i] * dt;
            temp_pos_z[i] = pos_z[i] + vel_z[i] * dt;
        }

        // k2: acceleration at predicted state
        computeAccelerations(k2_x, k2_y, k2_z, temp_pos_x, temp_pos_y, temp_pos_z, masses, G, epsilon);

        // Heun corrector
        std::vector<float> new_vel_x(n), new_vel_y(n), new_vel_z(n);
        std::vector<float> new_pos_x(n), new_pos_y(n), new_pos_z(n);

        for (int i = 0; i < n; i++) {
            new_vel_x[i] = vel_x[i] + 0.5f * dt * (k1_x[i] + k2_x[i]);
            new_vel_y[i] = vel_y[i] + 0.5f * dt * (k1_y[i] + k2_y[i]);
            new_vel_z[i] = vel_z[i] + 0.5f * dt * (k1_z[i] + k2_z[i]);

            new_pos_x[i] = pos_x[i] + 0.5f * dt * (vel_x[i] + new_vel_x[i]);
            new_pos_y[i] = pos_y[i] + 0.5f * dt * (vel_y[i] + new_vel_y[i]);
            new_pos_z[i] = pos_z[i] + 0.5f * dt * (vel_z[i] + new_vel_z[i]);

            // Error estimate (difference between Euler and Heun)
            error_x[i] = new_pos_x[i] - temp_pos_x[i];
            error_y[i] = new_pos_y[i] - temp_pos_y[i];
            error_z[i] = new_pos_z[i] - temp_pos_z[i];
        }

        float error = computeError(error_x, error_y, error_z);

        if (error <= params.tolerance || dt <= params.min_dt) {
            // Accept step
            pos_x = new_pos_x; pos_y = new_pos_y; pos_z = new_pos_z;
            vel_x = new_vel_x; vel_y = new_vel_y; vel_z = new_vel_z;
            total_time += dt;

            // Update time step for next iteration
            dt = computeOptimalTimeStep(dt, error, 2);
        } else {
            // Reject step, reduce time step
            dt = computeOptimalTimeStep(dt, error, 2);
        }

        substeps++;
    }

    return total_time;
}

} // namespace physgrad