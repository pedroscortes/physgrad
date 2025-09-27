#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

namespace physgrad {

// Base optimizer interface
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(std::vector<float>& params, const std::vector<float>& gradients) = 0;
    virtual void reset() = 0;
    virtual std::string getName() const = 0;
    virtual float getLearningRate() const = 0;
    virtual void setLearningRate(float lr) = 0;
};

// Simple momentum-based optimizer (current implementation)
class MomentumOptimizer : public Optimizer {
private:
    float learning_rate;
    float momentum_decay;
    std::vector<float> momentum;

public:
    MomentumOptimizer(float lr = 0.01f, float decay = 0.9f)
        : learning_rate(lr), momentum_decay(decay) {}

    void step(std::vector<float>& params, const std::vector<float>& gradients) override {
        if (momentum.size() != params.size()) {
            momentum.resize(params.size(), 0.0f);
        }

        for (size_t i = 0; i < params.size(); i++) {
            momentum[i] = momentum_decay * momentum[i] + learning_rate * gradients[i];
            params[i] -= momentum[i];
        }
    }

    void reset() override {
        std::fill(momentum.begin(), momentum.end(), 0.0f);
    }

    std::string getName() const override { return "Momentum"; }

    void setLearningRate(float lr) { learning_rate = lr; }
    float getLearningRate() const { return learning_rate; }
};

// Adaptive Moment Estimation (Adam) optimizer
class AdamOptimizer : public Optimizer {
private:
    float learning_rate;
    float beta1;          // Exponential decay rate for first moment estimates
    float beta2;          // Exponential decay rate for second moment estimates
    float epsilon;        // Small constant to prevent division by zero

    std::vector<float> m; // First moment vector (momentum)
    std::vector<float> v; // Second moment vector (RMSprop)
    int t;               // Time step (for bias correction)

public:
    AdamOptimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    void step(std::vector<float>& params, const std::vector<float>& gradients) override {
        if (m.size() != params.size()) {
            m.resize(params.size(), 0.0f);
            v.resize(params.size(), 0.0f);
        }

        t++; // Increment time step

        for (size_t i = 0; i < params.size(); i++) {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0f - beta1) * gradients[i];

            // Update biased second raw moment estimate
            v[i] = beta2 * v[i] + (1.0f - beta2) * gradients[i] * gradients[i];

            // Compute bias-corrected first moment estimate
            float m_hat = m[i] / (1.0f - std::pow(beta1, t));

            // Compute bias-corrected second raw moment estimate
            float v_hat = v[i] / (1.0f - std::pow(beta2, t));

            // Update parameters
            params[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }

    void reset() override {
        std::fill(m.begin(), m.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        t = 0;
    }

    std::string getName() const override { return "Adam"; }

    void setLearningRate(float lr) { learning_rate = lr; }
    float getLearningRate() const { return learning_rate; }

    // Getters for debugging
    float getBeta1() const { return beta1; }
    float getBeta2() const { return beta2; }
    int getTimeStep() const { return t; }
};

// AdamW optimizer (Adam with weight decay)
class AdamWOptimizer : public Optimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;

    std::vector<float> m;
    std::vector<float> v;
    int t;

public:
    AdamWOptimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f,
                   float eps = 1e-8f, float wd = 0.01f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0) {}

    void step(std::vector<float>& params, const std::vector<float>& gradients) override {
        if (m.size() != params.size()) {
            m.resize(params.size(), 0.0f);
            v.resize(params.size(), 0.0f);
        }

        t++;

        for (size_t i = 0; i < params.size(); i++) {
            // Apply weight decay directly to parameters (AdamW style)
            params[i] -= learning_rate * weight_decay * params[i];

            // Standard Adam updates
            m[i] = beta1 * m[i] + (1.0f - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * gradients[i] * gradients[i];

            float m_hat = m[i] / (1.0f - std::pow(beta1, t));
            float v_hat = v[i] / (1.0f - std::pow(beta2, t));

            params[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }

    void reset() override {
        std::fill(m.begin(), m.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        t = 0;
    }

    std::string getName() const override { return "AdamW"; }

    void setLearningRate(float lr) { learning_rate = lr; }
    float getLearningRate() const { return learning_rate; }
    void setWeightDecay(float wd) { weight_decay = wd; }
};

// L-BFGS optimizer (simplified implementation)
class LBFGSOptimizer : public Optimizer {
private:
    float learning_rate;
    int history_size;
    std::vector<std::vector<float>> s_history; // Parameter differences
    std::vector<std::vector<float>> y_history; // Gradient differences
    std::vector<float> rho_history;           // 1 / (y^T s)
    std::vector<float> prev_params;
    std::vector<float> prev_gradients;
    bool first_step;

public:
    LBFGSOptimizer(float lr = 1.0f, int history = 10)
        : learning_rate(lr), history_size(history), first_step(true) {}

    void step(std::vector<float>& params, const std::vector<float>& gradients) override {
        if (first_step) {
            // First step: simple gradient descent
            for (size_t i = 0; i < params.size(); i++) {
                params[i] -= learning_rate * gradients[i];
            }
            prev_params = params;
            prev_gradients = gradients;
            first_step = false;
            return;
        }

        // Compute s_k = x_{k+1} - x_k and y_k = g_{k+1} - g_k
        std::vector<float> s(params.size()), y(params.size());
        for (size_t i = 0; i < params.size(); i++) {
            s[i] = params[i] - prev_params[i];
            y[i] = gradients[i] - prev_gradients[i];
        }

        // Compute rho_k = 1 / (y_k^T s_k)
        float y_dot_s = 0.0f;
        for (size_t i = 0; i < params.size(); i++) {
            y_dot_s += y[i] * s[i];
        }

        if (std::abs(y_dot_s) > 1e-10f) {
            float rho = 1.0f / y_dot_s;

            // Store in history (with circular buffer)
            if (s_history.size() >= static_cast<size_t>(history_size)) {
                s_history.erase(s_history.begin());
                y_history.erase(y_history.begin());
                rho_history.erase(rho_history.begin());
            }

            s_history.push_back(s);
            y_history.push_back(y);
            rho_history.push_back(rho);
        }

        // Two-loop recursion to compute search direction
        std::vector<float> q = gradients; // Start with current gradient
        std::vector<float> alpha(s_history.size());

        // First loop (backward)
        for (int i = static_cast<int>(s_history.size()) - 1; i >= 0; i--) {
            float s_dot_q = 0.0f;
            for (size_t j = 0; j < params.size(); j++) {
                s_dot_q += s_history[i][j] * q[j];
            }
            alpha[i] = rho_history[i] * s_dot_q;

            for (size_t j = 0; j < params.size(); j++) {
                q[j] -= alpha[i] * y_history[i][j];
            }
        }

        // Apply initial Hessian approximation (identity scaled)
        std::vector<float> r = q;
        if (!s_history.empty()) {
            float gamma = 1.0f / rho_history.back();
            for (size_t i = 0; i < params.size(); i++) {
                r[i] *= gamma;
            }
        }

        // Second loop (forward)
        for (size_t i = 0; i < s_history.size(); i++) {
            float y_dot_r = 0.0f;
            for (size_t j = 0; j < params.size(); j++) {
                y_dot_r += y_history[i][j] * r[j];
            }
            float beta = rho_history[i] * y_dot_r;

            for (size_t j = 0; j < params.size(); j++) {
                r[j] += s_history[i][j] * (alpha[i] - beta);
            }
        }

        // Update parameters
        prev_params = params;
        for (size_t i = 0; i < params.size(); i++) {
            params[i] -= learning_rate * r[i];
        }
        prev_gradients = gradients;
    }

    void reset() override {
        s_history.clear();
        y_history.clear();
        rho_history.clear();
        prev_params.clear();
        prev_gradients.clear();
        first_step = true;
    }

    std::string getName() const override { return "L-BFGS"; }

    void setLearningRate(float lr) { learning_rate = lr; }
    float getLearningRate() const { return learning_rate; }
};

} // namespace physgrad