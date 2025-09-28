#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "stability_improvements.h"

namespace physgrad {

struct SimParams {
    int num_bodies = 1024;
    float time_step = 0.001f;
    float epsilon = 0.001f;
    float theta = 0.5f;
    float G = 1.0f;
    float cluster_scale = 1.0f;
    float velocity_scale = 0.5f;
    float max_force = 100.0f;
    StabilityParams stability;
};

struct BodySystem {
    float* d_pos_x = nullptr;
    float* d_pos_y = nullptr;
    float* d_pos_z = nullptr;
    float* d_vel_x = nullptr;
    float* d_vel_y = nullptr;
    float* d_vel_z = nullptr;
    float* d_acc_x = nullptr;
    float* d_acc_y = nullptr;
    float* d_acc_z = nullptr;
    float* d_mass = nullptr;

    float* d_grad_pos_x = nullptr;
    float* d_grad_pos_y = nullptr;
    float* d_grad_pos_z = nullptr;
    float* d_grad_vel_x = nullptr;
    float* d_grad_vel_y = nullptr;
    float* d_grad_vel_z = nullptr;

    float* d_grad_mass = nullptr;
    float* d_grad_G = nullptr;
    float* d_grad_epsilon = nullptr;
    float* d_grad_dt = nullptr;

    int n;

    BodySystem(int num_bodies);
    ~BodySystem();
    BodySystem(const BodySystem&) = delete;
    BodySystem& operator=(const BodySystem&) = delete;

    void initializeCluster(const SimParams& params);
    void getPositions(std::vector<float>& pos_x,
                     std::vector<float>& pos_y,
                     std::vector<float>& pos_z) const;
    float computeEnergy(const SimParams& params) const;

    // Optimized batch transfer methods
    void setStateFromHost(const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
                         const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
                         const std::vector<float>& masses, cudaStream_t stream = 0);
    void setStateFromHostAsync(const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
                              const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
                              const std::vector<float>& masses, cudaStream_t pos_stream, cudaStream_t vel_stream);
    void getStateToHost(std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
                       std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z) const;

    // Gradient methods
    void allocateGradients();
    void zeroGradients();
    void setGradientFromEnergy(float grad_energy = 1.0f);
    void getGradients(std::vector<float>& grad_pos_x,
                     std::vector<float>& grad_pos_y,
                     std::vector<float>& grad_pos_z) const;

    // Parameter gradient methods
    void allocateParameterGradients();
    void zeroParameterGradients();
    void getParameterGradients(std::vector<float>& grad_mass,
                              float& grad_G,
                              float& grad_epsilon) const;
    void getParameterGradientsWithTime(std::vector<float>& grad_mass,
                                      float& grad_G,
                                      float& grad_epsilon,
                                      float& grad_dt) const;
};

struct SimulationState {
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> vel_x, vel_y, vel_z;
};

class DifferentiableTape {
public:
    void recordState(const BodySystem& bodies);
    void recordStateAsync(const BodySystem& bodies, cudaStream_t stream);
    void clear();
    size_t size() const { return states.size(); }
    const SimulationState& getState(size_t index) const { return states[index]; }

private:
    std::vector<SimulationState> states;
};

class Simulation {
public:
    Simulation(const SimParams& params);
    ~Simulation() = default;

    void step();
    BodySystem* getBodies() { return bodies.get(); }
    const BodySystem* getBodies() const { return bodies.get(); }
    float* getPackedPositions();
    float getLastStepTime() const { return last_step_ms; }
    float getGFLOPS() const;

    // Differentiable programming
    void enableGradients();
    void disableGradients();
    void clearTape();
    void resetState();  // Reset simulation state for reuse instead of object recreation
    float computeGradients(const std::vector<float>& target_pos_x,
                          const std::vector<float>& target_pos_y,
                          const std::vector<float>& target_pos_z);

    // Parameter differentiation
    void enableParameterGradients(bool enable = true);
    float computeParameterGradients(const std::vector<float>& target_pos_x,
                                   const std::vector<float>& target_pos_y,
                                   const std::vector<float>& target_pos_z,
                                   std::vector<float>& grad_mass,
                                   float& grad_G,
                                   float& grad_epsilon);

    float computeParameterGradientsWithTime(const std::vector<float>& target_pos_x,
                                           const std::vector<float>& target_pos_y,
                                           const std::vector<float>& target_pos_z,
                                           std::vector<float>& grad_mass,
                                           float& grad_G,
                                           float& grad_epsilon,
                                           float& grad_dt);

    // Numerical stability
    void enableStableForces(bool enable = true);
    void stabilizeGradients();

private:
    SimParams params;
    std::unique_ptr<BodySystem> bodies;
    float last_step_ms = 0.0f;
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    float* d_packed_positions = nullptr;

    // Differentiable state
    bool gradients_enabled = false;
    bool parameter_gradients_enabled = false;
    DifferentiableTape tape;

    // Stability state
    bool stable_forces_enabled = false;
};

void launchComputeForces(float* d_acc_x, float* d_acc_y, float* d_acc_z,
                        const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
                        const float* d_mass, int n, float epsilon, float G,
                        cudaStream_t stream = 0);

void launchComputeForcesStable(float* d_acc_x, float* d_acc_y, float* d_acc_z,
                              const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
                              const float* d_mass, int n, float epsilon, float G,
                              float max_force, cudaStream_t stream = 0);

void launchIntegrate(float* d_pos_x, float* d_pos_y, float* d_pos_z,
                    float* d_vel_x, float* d_vel_y, float* d_vel_z,
                    const float* d_acc_x, const float* d_acc_y, const float* d_acc_z,
                    int n, float dt, cudaStream_t stream = 0);

// Adjoint (gradient) kernels for backpropagation
void launchComputeForcesAdjoint(
    float* d_grad_pos_x, float* d_grad_pos_y, float* d_grad_pos_z,
    const float* d_grad_acc_x, const float* d_grad_acc_y, const float* d_grad_acc_z,
    const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
    const float* d_mass, int n, float epsilon, float G,
    cudaStream_t stream = 0);

// Parameter gradient kernels
void launchComputeForcesParameterAdjoint(
    float* d_grad_mass, float* d_grad_G, float* d_grad_epsilon,
    const float* d_grad_acc_x, const float* d_grad_acc_y, const float* d_grad_acc_z,
    const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
    const float* d_mass, int n, float epsilon, float G,
    cudaStream_t stream = 0);

void launchIntegrateAdjoint(
    float* d_grad_pos_x, float* d_grad_pos_y, float* d_grad_pos_z,
    float* d_grad_vel_x, float* d_grad_vel_y, float* d_grad_vel_z,
    float* d_grad_acc_x, float* d_grad_acc_y, float* d_grad_acc_z,
    const float* d_grad_pos_next_x, const float* d_grad_pos_next_y, const float* d_grad_pos_next_z,
    const float* d_grad_vel_next_x, const float* d_grad_vel_next_y, const float* d_grad_vel_next_z,
    int n, float dt, cudaStream_t stream = 0);

void checkCudaError(cudaError_t error, const char* file, int line);
#define CUDA_CHECK(error) checkCudaError(error, __FILE__, __LINE__)

} // namespace physgrad