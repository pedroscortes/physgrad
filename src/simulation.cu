#include "simulation.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <cmath>
#include <iostream>
#include <chrono>

namespace physgrad {

// Error checking utility
void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line
                  << " - " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

__global__ void computeForces(
    float* acc_x, float* acc_y, float* acc_z,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* mass, int n, float epsilon_sq, float G)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = pos_x[i];
    float yi = pos_y[i];
    float zi = pos_z[i];

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    #pragma unroll 4
    for (int j = 0; j < n; j++) {
        if (i == j) continue;

        float dx = pos_x[j] - xi;
        float dy = pos_y[j] - yi;
        float dz = pos_z[j] - zi;

        float dist_sq = dx*dx + dy*dy + dz*dz + epsilon_sq;
        float inv_dist = rsqrtf(dist_sq);
        float inv_dist_cube = inv_dist * inv_dist * inv_dist;

        float force_mag = G * mass[j] * inv_dist_cube;

        ax += force_mag * dx;
        ay += force_mag * dy;
        az += force_mag * dz;
    }

    acc_x[i] = ax;
    acc_y[i] = ay;
    acc_z[i] = az;
}

__global__ void integrateLeapfrog(
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    const float* acc_x, const float* acc_y, const float* acc_z,
    int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Leapfrog-Verlet: v(t+dt/2) = v(t-dt/2) + a(t)*dt
    vel_x[i] += acc_x[i] * dt;
    vel_y[i] += acc_y[i] * dt;
    vel_z[i] += acc_z[i] * dt;

    // x(t+dt) = x(t) + v(t+dt/2)*dt
    pos_x[i] += vel_x[i] * dt;
    pos_y[i] += vel_y[i] * dt;
    pos_z[i] += vel_z[i] * dt;
}

__global__ void initializeRandom(
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    float* mass, int n,
    float cluster_scale, float velocity_scale,
    unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    curandState_t state;
    curand_init(seed, i, 0, &state);

    pos_x[i] = curand_normal(&state) * cluster_scale;
    pos_y[i] = curand_normal(&state) * cluster_scale;
    pos_z[i] = curand_normal(&state) * cluster_scale;

    vel_x[i] = curand_normal(&state) * velocity_scale;
    vel_y[i] = curand_normal(&state) * velocity_scale;
    vel_z[i] = curand_normal(&state) * velocity_scale;

    mass[i] = 1.0f / n;
}

__global__ void packPositions(
    float* packed,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* mass, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    packed[i * 4 + 0] = pos_x[i];
    packed[i * 4 + 1] = pos_y[i];
    packed[i * 4 + 2] = pos_z[i];
    packed[i * 4 + 3] = mass[i];
}

BodySystem::BodySystem(int num_bodies) : n(num_bodies) {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    size_t size = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_pos_x, size));
    CUDA_CHECK(cudaMalloc(&d_pos_y, size));
    CUDA_CHECK(cudaMalloc(&d_pos_z, size));

    CUDA_CHECK(cudaMalloc(&d_vel_x, size));
    CUDA_CHECK(cudaMalloc(&d_vel_y, size));
    CUDA_CHECK(cudaMalloc(&d_vel_z, size));

    CUDA_CHECK(cudaMalloc(&d_acc_x, size));
    CUDA_CHECK(cudaMalloc(&d_acc_y, size));
    CUDA_CHECK(cudaMalloc(&d_acc_z, size));

    CUDA_CHECK(cudaMalloc(&d_mass, size));

    CUDA_CHECK(cudaMemset(d_acc_x, 0, size));
    CUDA_CHECK(cudaMemset(d_acc_y, 0, size));
    CUDA_CHECK(cudaMemset(d_acc_z, 0, size));
}

BodySystem::~BodySystem() {
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_pos_z);
    cudaFree(d_vel_x);
    cudaFree(d_vel_y);
    cudaFree(d_vel_z);
    cudaFree(d_acc_x);
    cudaFree(d_acc_y);
    cudaFree(d_acc_z);
    cudaFree(d_mass);
}

void BodySystem::initializeCluster(const SimParams& params) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();

    initializeRandom<<<blocks, threads>>>(
        d_pos_x, d_pos_y, d_pos_z,
        d_vel_x, d_vel_y, d_vel_z,
        d_mass, n,
        params.cluster_scale, params.velocity_scale,
        seed
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void BodySystem::getPositions(std::vector<float>& pos_x,
                              std::vector<float>& pos_y,
                              std::vector<float>& pos_z) const {
    pos_x.resize(n);
    pos_y.resize(n);
    pos_z.resize(n);

    size_t size = n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(pos_x.data(), d_pos_x, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pos_y.data(), d_pos_y, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pos_z.data(), d_pos_z, size, cudaMemcpyDeviceToHost));
}

float BodySystem::computeEnergy(const SimParams& params) const {
    // For now, just compute kinetic energy as a simple validation
    // We'll add potential energy calculation later

    std::vector<float> vel_x(n), vel_y(n), vel_z(n), mass(n);

    size_t size = n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(vel_x.data(), d_vel_x, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel_y.data(), d_vel_y, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel_z.data(), d_vel_z, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mass.data(), d_mass, size, cudaMemcpyDeviceToHost));

    float kinetic_energy = 0.0f;
    for (int i = 0; i < n; i++) {
        float v2 = vel_x[i]*vel_x[i] + vel_y[i]*vel_y[i] + vel_z[i]*vel_z[i];
        kinetic_energy += 0.5f * mass[i] * v2;
    }

    return kinetic_energy;
}

// ============================================================================
// Simulation Implementation
// ============================================================================

Simulation::Simulation(const SimParams& params)
    : params(params), bodies(std::make_unique<BodySystem>(params.num_bodies))
{
    // Create CUDA streams
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&transfer_stream));

    // Allocate packed position buffer for OpenGL
    CUDA_CHECK(cudaMalloc(&d_packed_positions, params.num_bodies * 4 * sizeof(float)));

    // Initialize the particle system
    bodies->initializeCluster(params);

    std::cout << "Simulation initialized with " << params.num_bodies << " bodies\n";
    std::cout << "Initial energy: " << bodies->computeEnergy(params) << "\n";
}

void Simulation::step() {
    auto start = std::chrono::high_resolution_clock::now();

    // Compute forces
    launchComputeForces(
        bodies->d_acc_x, bodies->d_acc_y, bodies->d_acc_z,
        bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
        bodies->d_mass, bodies->n,
        params.epsilon * params.epsilon,  // epsilon squared
        params.G,
        compute_stream
    );

    // Integrate positions and velocities
    launchIntegrate(
        bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
        bodies->d_vel_x, bodies->d_vel_y, bodies->d_vel_z,
        bodies->d_acc_x, bodies->d_acc_y, bodies->d_acc_z,
        bodies->n, params.time_step,
        compute_stream
    );

    // Wait for computation to finish
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    auto end = std::chrono::high_resolution_clock::now();
    last_step_ms = std::chrono::duration<float, std::milli>(end - start).count();
}

float* Simulation::getPackedPositions() {
    int threads = 256;
    int blocks = (bodies->n + threads - 1) / threads;

    packPositions<<<blocks, threads, 0, transfer_stream>>>(
        d_packed_positions,
        bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
        bodies->d_mass, bodies->n
    );

    CUDA_CHECK(cudaStreamSynchronize(transfer_stream));
    return d_packed_positions;
}

float Simulation::getGFLOPS() const {
    // Each force pair: ~20 FLOPs
    // Total: N*(N-1) pairs
    float flops = 20.0f * bodies->n * (bodies->n - 1);
    return (flops / last_step_ms) / 1e6f;  // GFLOPS
}

// ============================================================================
// Kernel Launchers
// ============================================================================

void launchComputeForces(float* d_acc_x, float* d_acc_y, float* d_acc_z,
                        const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
                        const float* d_mass, int n, float epsilon, float G,
                        cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    computeForces<<<blocks, threads, 0, stream>>>(
        d_acc_x, d_acc_y, d_acc_z,
        d_pos_x, d_pos_y, d_pos_z,
        d_mass, n, epsilon, G
    );

    CUDA_CHECK(cudaGetLastError());
}

void launchIntegrate(float* d_pos_x, float* d_pos_y, float* d_pos_z,
                    float* d_vel_x, float* d_vel_y, float* d_vel_z,
                    const float* d_acc_x, const float* d_acc_y, const float* d_acc_z,
                    int n, float dt, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    integrateLeapfrog<<<blocks, threads, 0, stream>>>(
        d_pos_x, d_pos_y, d_pos_z,
        d_vel_x, d_vel_y, d_vel_z,
        d_acc_x, d_acc_y, d_acc_z,
        n, dt
    );

    CUDA_CHECK(cudaGetLastError());
}

} // namespace physgrad