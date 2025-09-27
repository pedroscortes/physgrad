#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace physgrad {

// Configuration parameters
struct SimParams {
    int num_bodies = 1024;
    float time_step = 0.001f;
    float epsilon = 0.001f;
    float theta = 0.5f;
    float G = 1.0f;
    float cluster_scale = 1.0f;
    float velocity_scale = 0.5f;
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

private:
    SimParams params;
    std::unique_ptr<BodySystem> bodies;
    float last_step_ms = 0.0f;
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    float* d_packed_positions = nullptr;
};

void launchComputeForces(float* d_acc_x, float* d_acc_y, float* d_acc_z,
                        const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
                        const float* d_mass, int n, float epsilon, float G,
                        cudaStream_t stream = 0);

void launchIntegrate(float* d_pos_x, float* d_pos_y, float* d_pos_z,
                    float* d_vel_x, float* d_vel_y, float* d_vel_z,
                    const float* d_acc_x, const float* d_acc_y, const float* d_acc_z,
                    int n, float dt, cudaStream_t stream = 0);

void checkCudaError(cudaError_t error, const char* file, int line);
#define CUDA_CHECK(error) checkCudaError(error, __FILE__, __LINE__)

} // namespace physgrad