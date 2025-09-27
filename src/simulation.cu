#include "simulation.h"
#include "stability_improvements.h"
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

__global__ void computeForcesParameterAdjoint(
    float* grad_mass, float* grad_G, float* grad_epsilon,
    const float* grad_acc_x, const float* grad_acc_y, const float* grad_acc_z,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* mass, int n, float epsilon_sq, float G)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;  // Simplified: just return if out of bounds

    float xi = pos_x[i];
    float yi = pos_y[i];
    float zi = pos_z[i];

    float grad_acc_xi = grad_acc_x[i];
    float grad_acc_yi = grad_acc_y[i];
    float grad_acc_zi = grad_acc_z[i];

    float local_grad_G = 0.0f;
    float local_grad_epsilon = 0.0f;

    // Compute gradients for this particle
    for (int j = 0; j < n; j++) {
        if (i == j) continue;

        float dx = pos_x[j] - xi;
        float dy = pos_y[j] - yi;
        float dz = pos_z[j] - zi;

        float dist_sq = dx*dx + dy*dy + dz*dz;
        float regularized_dist_sq = dist_sq + epsilon_sq;
        float inv_dist = rsqrtf(regularized_dist_sq);
        float inv_dist_cube = inv_dist * inv_dist * inv_dist;

        // Gradient contribution from this force
        float grad_force_contribution = grad_acc_xi * dx + grad_acc_yi * dy + grad_acc_zi * dz;

        // Gradient w.r.t. mass[j]
        atomicAdd(&grad_mass[j], grad_force_contribution * G * inv_dist_cube);

        // Gradient w.r.t. G (accumulate locally)
        local_grad_G += grad_force_contribution * mass[j] * inv_dist_cube;

        // Gradient w.r.t. epsilon (simplified)
        if (epsilon_sq > 0) {
            float inv_dist_five = inv_dist_cube * inv_dist * inv_dist;
            float grad_inv_dist_cube = grad_force_contribution * G * mass[j];
            float grad_regularized_dist_sq = grad_inv_dist_cube * (-1.5f) * inv_dist_five;
            local_grad_epsilon += grad_regularized_dist_sq * 2.0f * sqrtf(epsilon_sq);
        }
    }

    // Accumulate scalar gradients with atomic operations (simpler than shared memory)
    atomicAdd(grad_G, local_grad_G);
    atomicAdd(grad_epsilon, local_grad_epsilon);
}

__global__ void computeForcesAdjoint(
    float* grad_pos_x, float* grad_pos_y, float* grad_pos_z,
    const float* grad_acc_x, const float* grad_acc_y, const float* grad_acc_z,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* mass, int n, float epsilon_sq, float G)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = pos_x[i];
    float yi = pos_y[i];
    float zi = pos_z[i];

    float grad_xi = 0.0f;
    float grad_yi = 0.0f;
    float grad_zi = 0.0f;

    float grad_acc_xi = grad_acc_x[i];
    float grad_acc_yi = grad_acc_y[i];
    float grad_acc_zi = grad_acc_z[i];

    #pragma unroll 4
    for (int j = 0; j < n; j++) {
        if (i == j) continue;

        float dx = pos_x[j] - xi;
        float dy = pos_y[j] - yi;
        float dz = pos_z[j] - zi;

        float dist_sq = dx*dx + dy*dy + dz*dz + epsilon_sq;
        float inv_dist = rsqrtf(dist_sq);
        float inv_dist_cube = inv_dist * inv_dist * inv_dist;
        float inv_dist_five = inv_dist_cube * inv_dist * inv_dist;

        float force_mag = G * mass[j] * inv_dist_cube;

        // Gradient from force magnitude term
        float grad_force_mag_x = grad_acc_xi * dx;
        float grad_force_mag_y = grad_acc_yi * dy;
        float grad_force_mag_z = grad_acc_zi * dz;

        // Gradient from direction terms
        float grad_dx = grad_acc_xi * force_mag;
        float grad_dy = grad_acc_yi * force_mag;
        float grad_dz = grad_acc_zi * force_mag;

        // Gradient through inverse distance cubed
        float grad_inv_dist_cube = (grad_force_mag_x * dx + grad_force_mag_y * dy + grad_force_mag_z * dz) * G * mass[j];

        // Gradient through distance squared
        float grad_dist_sq = grad_inv_dist_cube * (-1.5f) * inv_dist_five;

        // Accumulate gradients w.r.t. position differences
        float grad_dx_total = grad_dx + grad_dist_sq * 2.0f * dx;
        float grad_dy_total = grad_dy + grad_dist_sq * 2.0f * dy;
        float grad_dz_total = grad_dz + grad_dist_sq * 2.0f * dz;

        // Gradient w.r.t. xi (negative because dx = pos_x[j] - xi)
        grad_xi -= grad_dx_total;
        grad_yi -= grad_dy_total;
        grad_zi -= grad_dz_total;

        // Add contribution to grad_pos_x[j] (positive because dx = pos_x[j] - xi)
        atomicAdd(&grad_pos_x[j], grad_dx_total);
        atomicAdd(&grad_pos_y[j], grad_dy_total);
        atomicAdd(&grad_pos_z[j], grad_dz_total);
    }

    grad_pos_x[i] += grad_xi;
    grad_pos_y[i] += grad_yi;
    grad_pos_z[i] += grad_zi;
}

__global__ void integrateLeapfrog(
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    const float* acc_x, const float* acc_y, const float* acc_z,
    int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vel_x[i] += acc_x[i] * dt;
    vel_y[i] += acc_y[i] * dt;
    vel_z[i] += acc_z[i] * dt;

    pos_x[i] += vel_x[i] * dt;
    pos_y[i] += vel_y[i] * dt;
    pos_z[i] += vel_z[i] * dt;
}

__global__ void integrateLeapfrogAdjoint(
    float* grad_pos_x, float* grad_pos_y, float* grad_pos_z,
    float* grad_vel_x, float* grad_vel_y, float* grad_vel_z,
    float* grad_acc_x, float* grad_acc_y, float* grad_acc_z,
    const float* grad_pos_next_x, const float* grad_pos_next_y, const float* grad_pos_next_z,
    const float* grad_vel_next_x, const float* grad_vel_next_y, const float* grad_vel_next_z,
    int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Forward: vel_new = vel_old + acc * dt
    // Forward: pos_new = pos_old + vel_new * dt

    // Total gradient flowing into vel_new
    float total_grad_vel_x = grad_vel_next_x[i] + grad_pos_next_x[i] * dt;
    float total_grad_vel_y = grad_vel_next_y[i] + grad_pos_next_y[i] * dt;
    float total_grad_vel_z = grad_vel_next_z[i] + grad_pos_next_z[i] * dt;

    // Backpropagate through position update: pos_new = pos_old + vel_new * dt
    grad_pos_x[i] = grad_pos_next_x[i];
    grad_pos_y[i] = grad_pos_next_y[i];
    grad_pos_z[i] = grad_pos_next_z[i];

    // Backpropagate through velocity update: vel_new = vel_old + acc * dt
    grad_vel_x[i] = total_grad_vel_x;
    grad_vel_y[i] = total_grad_vel_y;
    grad_vel_z[i] = total_grad_vel_z;

    grad_acc_x[i] = total_grad_vel_x * dt;
    grad_acc_y[i] = total_grad_vel_y * dt;
    grad_acc_z[i] = total_grad_vel_z * dt;
}

__global__ void updateVelocitiesKernel(
    float* vel_x, float* vel_y, float* vel_z,
    const float* acc_x, const float* acc_y, const float* acc_z,
    int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vel_x[i] += acc_x[i] * dt;
    vel_y[i] += acc_y[i] * dt;
    vel_z[i] += acc_z[i] * dt;
}

__global__ void integrateLeapfrogTimeAdjoint(
    float* grad_dt,
    const float* grad_pos_next_x, const float* grad_pos_next_y, const float* grad_pos_next_z,
    const float* grad_vel_next_x, const float* grad_vel_next_y, const float* grad_vel_next_z,
    const float* vel_x, const float* vel_y, const float* vel_z,
    const float* acc_x, const float* acc_y, const float* acc_z,
    int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Forward equations:
    // vel_new = vel_old + acc * dt
    // pos_new = pos_old + vel_new * dt = pos_old + (vel_old + acc * dt) * dt
    //         = pos_old + vel_old * dt + acc * dt^2

    // Gradients w.r.t. dt:
    // ∂loss/∂dt from velocity update: grad_vel_next * acc
    float local_grad_dt_vel = grad_vel_next_x[i] * acc_x[i] +
                              grad_vel_next_y[i] * acc_y[i] +
                              grad_vel_next_z[i] * acc_z[i];

    // ∂loss/∂dt from position update: grad_pos_next * (vel_new + acc * dt)
    // The position update is: pos_new = pos_old + vel_new * dt
    // Since vel_new = vel_old + acc * dt, we have:
    // pos_new = pos_old + (vel_old + acc * dt) * dt = pos_old + vel_old * dt + acc * dt^2
    // So ∂pos_new/∂dt = vel_old + 2 * acc * dt
    // But since vel_new = vel_old + acc * dt, we get: vel_old = vel_new - acc * dt
    // Therefore: ∂pos_new/∂dt = (vel_new - acc * dt) + 2 * acc * dt = vel_new + acc * dt
    float local_grad_dt_pos = grad_pos_next_x[i] * (vel_x[i] + acc_x[i] * dt) +
                              grad_pos_next_y[i] * (vel_y[i] + acc_y[i] * dt) +
                              grad_pos_next_z[i] * (vel_z[i] + acc_z[i] * dt);

    // Accumulate to global gradient
    atomicAdd(grad_dt, local_grad_dt_vel + local_grad_dt_pos);
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

    cudaFree(d_grad_pos_x);
    cudaFree(d_grad_pos_y);
    cudaFree(d_grad_pos_z);
    cudaFree(d_grad_vel_x);
    cudaFree(d_grad_vel_y);
    cudaFree(d_grad_vel_z);

    cudaFree(d_grad_mass);
    cudaFree(d_grad_G);
    cudaFree(d_grad_epsilon);
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

void BodySystem::allocateGradients() {
    size_t size = n * sizeof(float);

    if (!d_grad_pos_x) {
        CUDA_CHECK(cudaMalloc(&d_grad_pos_x, size));
        CUDA_CHECK(cudaMalloc(&d_grad_pos_y, size));
        CUDA_CHECK(cudaMalloc(&d_grad_pos_z, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_x, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_y, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_z, size));
    }
}

void BodySystem::zeroGradients() {
    if (!d_grad_pos_x) return;

    size_t size = n * sizeof(float);
    CUDA_CHECK(cudaMemset(d_grad_pos_x, 0, size));
    CUDA_CHECK(cudaMemset(d_grad_pos_y, 0, size));
    CUDA_CHECK(cudaMemset(d_grad_pos_z, 0, size));
    CUDA_CHECK(cudaMemset(d_grad_vel_x, 0, size));
    CUDA_CHECK(cudaMemset(d_grad_vel_y, 0, size));
    CUDA_CHECK(cudaMemset(d_grad_vel_z, 0, size));
}

void BodySystem::setGradientFromEnergy(float grad_energy) {
    allocateGradients();
    zeroGradients();

    float grad_value = grad_energy / n;
    std::vector<float> temp_grad(n, grad_value);

    size_t size = n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_grad_pos_x, temp_grad.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_pos_y, temp_grad.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_pos_z, temp_grad.data(), size, cudaMemcpyHostToDevice));
}

void BodySystem::getGradients(std::vector<float>& grad_pos_x,
                             std::vector<float>& grad_pos_y,
                             std::vector<float>& grad_pos_z) const {
    if (!d_grad_pos_x) return;

    grad_pos_x.resize(n);
    grad_pos_y.resize(n);
    grad_pos_z.resize(n);

    size_t size = n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(grad_pos_x.data(), d_grad_pos_x, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad_pos_y.data(), d_grad_pos_y, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad_pos_z.data(), d_grad_pos_z, size, cudaMemcpyDeviceToHost));
}

// Optimized batch transfer methods
void BodySystem::setStateFromHost(const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
                                 const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
                                 const std::vector<float>& masses, cudaStream_t stream) {
    size_t size = n * sizeof(float);

    // Batch position transfers
    if (stream != 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice, stream));

        // Batch velocity and mass transfers
        CUDA_CHECK(cudaMemcpyAsync(d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_mass, masses.data(), size, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mass, masses.data(), size, cudaMemcpyHostToDevice));
    }
}

void BodySystem::setStateFromHostAsync(const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
                                      const std::vector<float>& vel_x, const std::vector<float>& vel_y, const std::vector<float>& vel_z,
                                      const std::vector<float>& masses, cudaStream_t pos_stream, cudaStream_t vel_stream) {
    size_t size = n * sizeof(float);

    // Stream 1: Position transfers
    CUDA_CHECK(cudaMemcpyAsync(d_pos_x, pos_x.data(), size, cudaMemcpyHostToDevice, pos_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pos_y, pos_y.data(), size, cudaMemcpyHostToDevice, pos_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pos_z, pos_z.data(), size, cudaMemcpyHostToDevice, pos_stream));

    // Stream 2: Velocity and mass transfers
    CUDA_CHECK(cudaMemcpyAsync(d_vel_x, vel_x.data(), size, cudaMemcpyHostToDevice, vel_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vel_y, vel_y.data(), size, cudaMemcpyHostToDevice, vel_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vel_z, vel_z.data(), size, cudaMemcpyHostToDevice, vel_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_mass, masses.data(), size, cudaMemcpyHostToDevice, vel_stream));
}

void BodySystem::getStateToHost(std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
                               std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z) const {
    pos_x.resize(n); pos_y.resize(n); pos_z.resize(n);
    vel_x.resize(n); vel_y.resize(n); vel_z.resize(n);

    size_t size = n * sizeof(float);

    // Batch position transfers
    CUDA_CHECK(cudaMemcpy(pos_x.data(), d_pos_x, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pos_y.data(), d_pos_y, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pos_z.data(), d_pos_z, size, cudaMemcpyDeviceToHost));

    // Batch velocity transfers
    CUDA_CHECK(cudaMemcpy(vel_x.data(), d_vel_x, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel_y.data(), d_vel_y, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel_z.data(), d_vel_z, size, cudaMemcpyDeviceToHost));
}

void BodySystem::allocateParameterGradients() {
    size_t size = n * sizeof(float);

    if (!d_grad_mass) {
        CUDA_CHECK(cudaMalloc(&d_grad_mass, size));
        CUDA_CHECK(cudaMalloc(&d_grad_G, sizeof(float)));        // Scalar gradient
        CUDA_CHECK(cudaMalloc(&d_grad_epsilon, sizeof(float)));  // Scalar gradient
        CUDA_CHECK(cudaMalloc(&d_grad_dt, sizeof(float)));       // Scalar gradient
    }
}

void BodySystem::zeroParameterGradients() {
    if (!d_grad_mass) return;

    size_t size = n * sizeof(float);
    CUDA_CHECK(cudaMemset(d_grad_mass, 0, size));
    CUDA_CHECK(cudaMemset(d_grad_G, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_epsilon, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_dt, 0, sizeof(float)));
}

void BodySystem::getParameterGradients(std::vector<float>& grad_mass,
                                      float& grad_G,
                                      float& grad_epsilon) const {
    if (!d_grad_mass) return;

    grad_mass.resize(n);
    size_t size = n * sizeof(float);

    CUDA_CHECK(cudaMemcpy(grad_mass.data(), d_grad_mass, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&grad_G, d_grad_G, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&grad_epsilon, d_grad_epsilon, sizeof(float), cudaMemcpyDeviceToHost));
}

void BodySystem::getParameterGradientsWithTime(std::vector<float>& grad_mass,
                                              float& grad_G,
                                              float& grad_epsilon,
                                              float& grad_dt) const {
    if (!d_grad_mass) return;

    grad_mass.resize(n);
    size_t size = n * sizeof(float);

    CUDA_CHECK(cudaMemcpy(grad_mass.data(), d_grad_mass, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&grad_G, d_grad_G, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&grad_epsilon, d_grad_epsilon, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&grad_dt, d_grad_dt, sizeof(float), cudaMemcpyDeviceToHost));
}


Simulation::Simulation(const SimParams& params)
    : params(params), bodies(std::make_unique<BodySystem>(params.num_bodies))
{
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&transfer_stream));
    CUDA_CHECK(cudaMalloc(&d_packed_positions, params.num_bodies * 4 * sizeof(float)));
    bodies->initializeCluster(params);
}

void DifferentiableTape::recordState(const BodySystem& bodies) {
    SimulationState state;
    state.pos_x.resize(bodies.n);
    state.pos_y.resize(bodies.n);
    state.pos_z.resize(bodies.n);
    state.vel_x.resize(bodies.n);
    state.vel_y.resize(bodies.n);
    state.vel_z.resize(bodies.n);

    size_t size = bodies.n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(state.pos_x.data(), bodies.d_pos_x, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.pos_y.data(), bodies.d_pos_y, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.pos_z.data(), bodies.d_pos_z, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.vel_x.data(), bodies.d_vel_x, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.vel_y.data(), bodies.d_vel_y, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.vel_z.data(), bodies.d_vel_z, size, cudaMemcpyDeviceToHost));

    states.push_back(std::move(state));
}

void DifferentiableTape::recordStateAsync(const BodySystem& bodies, cudaStream_t stream) {
    SimulationState state;
    state.pos_x.resize(bodies.n);
    state.pos_y.resize(bodies.n);
    state.pos_z.resize(bodies.n);
    state.vel_x.resize(bodies.n);
    state.vel_y.resize(bodies.n);
    state.vel_z.resize(bodies.n);

    size_t size = bodies.n * sizeof(float);

    // Batch position transfers
    CUDA_CHECK(cudaMemcpyAsync(state.pos_x.data(), bodies.d_pos_x, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(state.pos_y.data(), bodies.d_pos_y, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(state.pos_z.data(), bodies.d_pos_z, size, cudaMemcpyDeviceToHost, stream));

    // Batch velocity transfers
    CUDA_CHECK(cudaMemcpyAsync(state.vel_x.data(), bodies.d_vel_x, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(state.vel_y.data(), bodies.d_vel_y, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(state.vel_z.data(), bodies.d_vel_z, size, cudaMemcpyDeviceToHost, stream));

    states.push_back(std::move(state));
}

void DifferentiableTape::clear() {
    states.clear();
}

void Simulation::step() {
    auto start = std::chrono::high_resolution_clock::now();

    if (stable_forces_enabled) {
        launchComputeForcesStable(
            bodies->d_acc_x, bodies->d_acc_y, bodies->d_acc_z,
            bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
            bodies->d_mass, bodies->n,
            params.epsilon * params.epsilon,
            params.G, params.max_force,
            compute_stream
        );
    } else {
        launchComputeForces(
            bodies->d_acc_x, bodies->d_acc_y, bodies->d_acc_z,
            bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
            bodies->d_mass, bodies->n,
            params.epsilon * params.epsilon,
            params.G,
            compute_stream
        );
    }

    // Record state AFTER force computation but BEFORE integration
    // This ensures adjoint computation uses the same positions as forward pass
    if (gradients_enabled) {
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));  // Ensure forces are computed
        tape.recordState(*bodies);
    }

    launchIntegrate(
        bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
        bodies->d_vel_x, bodies->d_vel_y, bodies->d_vel_z,
        bodies->d_acc_x, bodies->d_acc_y, bodies->d_acc_z,
        bodies->n, params.time_step,
        compute_stream
    );

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
    float flops = 20.0f * bodies->n * (bodies->n - 1);
    return (flops / last_step_ms) / 1e6f;
}

void Simulation::enableGradients() {
    gradients_enabled = true;
    bodies->allocateGradients();
}

void Simulation::disableGradients() {
    gradients_enabled = false;
}

void Simulation::clearTape() {
    tape.clear();
}

void Simulation::resetState() {
    // Clear gradient tape
    tape.clear();

    // Reset timing info
    last_step_ms = 0.0f;

    // Zero all gradient arrays if allocated
    if (bodies->d_grad_pos_x) {
        bodies->zeroGradients();
    }
    if (bodies->d_grad_mass) {
        bodies->zeroParameterGradients();
    }

    // Note: We keep all GPU memory allocations intact for reuse
    // Note: We keep simulation parameters (num_bodies, time_step, etc.) unchanged
    // This allows reuse of the same Simulation object with different initial conditions
}

float Simulation::computeGradients(const std::vector<float>& target_pos_x,
                                  const std::vector<float>& target_pos_y,
                                  const std::vector<float>& target_pos_z) {
    if (!gradients_enabled || tape.size() == 0) {
        return 0.0f;
    }

    bodies->allocateGradients();
    bodies->zeroGradients();

    // Compute loss (MSE between final positions and target)
    std::vector<float> pos_x, pos_y, pos_z;
    bodies->getPositions(pos_x, pos_y, pos_z);

    float loss = 0.0f;
    for (int i = 0; i < bodies->n; i++) {
        float dx = pos_x[i] - target_pos_x[i];
        float dy = pos_y[i] - target_pos_y[i];
        float dz = pos_z[i] - target_pos_z[i];
        loss += dx*dx + dy*dy + dz*dz;
    }
    loss /= (2.0f * bodies->n);  // Match finite difference normalization

    // Set initial gradients (gradient of loss w.r.t. final positions)
    std::vector<float> grad_pos_x(bodies->n), grad_pos_y(bodies->n), grad_pos_z(bodies->n);
    for (int i = 0; i < bodies->n; i++) {
        grad_pos_x[i] = (pos_x[i] - target_pos_x[i]) / bodies->n;  // Gradient of MSE/2
        grad_pos_y[i] = (pos_y[i] - target_pos_y[i]) / bodies->n;
        grad_pos_z[i] = (pos_z[i] - target_pos_z[i]) / bodies->n;
    }

    size_t size = bodies->n * sizeof(float);
    CUDA_CHECK(cudaMemcpy(bodies->d_grad_pos_x, grad_pos_x.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bodies->d_grad_pos_y, grad_pos_y.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bodies->d_grad_pos_z, grad_pos_z.data(), size, cudaMemcpyHostToDevice));

    // Backpropagate through time (reverse order)
    for (int step = tape.size() - 1; step >= 0; step--) {
        const auto& state = tape.getState(step);

        // Restore state for adjoint computation
        CUDA_CHECK(cudaMemcpy(bodies->d_pos_x, state.pos_x.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_pos_y, state.pos_y.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_pos_z, state.pos_z.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_vel_x, state.vel_x.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_vel_y, state.vel_y.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_vel_z, state.vel_z.data(), size, cudaMemcpyHostToDevice));

        // Allocate temporary gradient arrays
        float *d_grad_acc_x, *d_grad_acc_y, *d_grad_acc_z;
        float *d_grad_pos_next_x, *d_grad_pos_next_y, *d_grad_pos_next_z;
        float *d_grad_vel_next_x, *d_grad_vel_next_y, *d_grad_vel_next_z;

        CUDA_CHECK(cudaMalloc(&d_grad_acc_x, size));
        CUDA_CHECK(cudaMalloc(&d_grad_acc_y, size));
        CUDA_CHECK(cudaMalloc(&d_grad_acc_z, size));
        CUDA_CHECK(cudaMalloc(&d_grad_pos_next_x, size));
        CUDA_CHECK(cudaMalloc(&d_grad_pos_next_y, size));
        CUDA_CHECK(cudaMalloc(&d_grad_pos_next_z, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_next_x, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_next_y, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_next_z, size));

        CUDA_CHECK(cudaMemset(d_grad_acc_x, 0, size));
        CUDA_CHECK(cudaMemset(d_grad_acc_y, 0, size));
        CUDA_CHECK(cudaMemset(d_grad_acc_z, 0, size));
        CUDA_CHECK(cudaMemcpy(d_grad_pos_next_x, bodies->d_grad_pos_x, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_pos_next_y, bodies->d_grad_pos_y, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_pos_next_z, bodies->d_grad_pos_z, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_vel_next_x, bodies->d_grad_vel_x, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_vel_next_y, bodies->d_grad_vel_y, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_vel_next_z, bodies->d_grad_vel_z, size, cudaMemcpyDeviceToDevice));

        // Adjoint of integration step
        launchIntegrateAdjoint(
            bodies->d_grad_pos_x, bodies->d_grad_pos_y, bodies->d_grad_pos_z,
            bodies->d_grad_vel_x, bodies->d_grad_vel_y, bodies->d_grad_vel_z,
            d_grad_acc_x, d_grad_acc_y, d_grad_acc_z,
            d_grad_pos_next_x, d_grad_pos_next_y, d_grad_pos_next_z,
            d_grad_vel_next_x, d_grad_vel_next_y, d_grad_vel_next_z,
            bodies->n, params.time_step, compute_stream
        );

        // Adjoint of force computation
        launchComputeForcesAdjoint(
            bodies->d_grad_pos_x, bodies->d_grad_pos_y, bodies->d_grad_pos_z,
            d_grad_acc_x, d_grad_acc_y, d_grad_acc_z,
            bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
            bodies->d_mass, bodies->n,
            params.epsilon * params.epsilon, params.G,
            compute_stream
        );

        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // Clean up temporary arrays
        cudaFree(d_grad_acc_x);
        cudaFree(d_grad_acc_y);
        cudaFree(d_grad_acc_z);
        cudaFree(d_grad_pos_next_x);
        cudaFree(d_grad_pos_next_y);
        cudaFree(d_grad_pos_next_z);
        cudaFree(d_grad_vel_next_x);
        cudaFree(d_grad_vel_next_y);
        cudaFree(d_grad_vel_next_z);
    }

    return loss;
}

void Simulation::enableParameterGradients(bool enable) {
    parameter_gradients_enabled = enable;
    if (enable) {
        bodies->allocateParameterGradients();
    }
}

float Simulation::computeParameterGradients(const std::vector<float>& target_pos_x,
                                           const std::vector<float>& target_pos_y,
                                           const std::vector<float>& target_pos_z,
                                           std::vector<float>& grad_mass,
                                           float& grad_G,
                                           float& grad_epsilon) {

    if (!parameter_gradients_enabled) {
        std::cerr << "Parameter gradients not enabled!\n";
        return 0.0f;
    }

    // First compute the regular loss and gradients
    float loss = computeGradients(target_pos_x, target_pos_y, target_pos_z);

    // Zero parameter gradients
    bodies->zeroParameterGradients();

    // Backpropagate through time, accumulating parameter gradients
    size_t size = bodies->n * sizeof(float);

    for (int step = tape.size() - 1; step >= 0; step--) {
        const auto& state = tape.getState(step);

        // Restore state for adjoint computation
        CUDA_CHECK(cudaMemcpy(bodies->d_pos_x, state.pos_x.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_pos_y, state.pos_y.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_pos_z, state.pos_z.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_vel_x, state.vel_x.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_vel_y, state.vel_y.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_vel_z, state.vel_z.data(), size, cudaMemcpyHostToDevice));

        // Allocate temporary gradient arrays for acceleration
        float *d_grad_acc_x, *d_grad_acc_y, *d_grad_acc_z;
        float *d_grad_pos_next_x, *d_grad_pos_next_y, *d_grad_pos_next_z;
        float *d_grad_vel_next_x, *d_grad_vel_next_y, *d_grad_vel_next_z;

        CUDA_CHECK(cudaMalloc(&d_grad_acc_x, size));
        CUDA_CHECK(cudaMalloc(&d_grad_acc_y, size));
        CUDA_CHECK(cudaMalloc(&d_grad_acc_z, size));
        CUDA_CHECK(cudaMalloc(&d_grad_pos_next_x, size));
        CUDA_CHECK(cudaMalloc(&d_grad_pos_next_y, size));
        CUDA_CHECK(cudaMalloc(&d_grad_pos_next_z, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_next_x, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_next_y, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_next_z, size));

        CUDA_CHECK(cudaMemset(d_grad_acc_x, 0, size));
        CUDA_CHECK(cudaMemset(d_grad_acc_y, 0, size));
        CUDA_CHECK(cudaMemset(d_grad_acc_z, 0, size));
        CUDA_CHECK(cudaMemcpy(d_grad_pos_next_x, bodies->d_grad_pos_x, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_pos_next_y, bodies->d_grad_pos_y, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_pos_next_z, bodies->d_grad_pos_z, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_vel_next_x, bodies->d_grad_vel_x, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_vel_next_y, bodies->d_grad_vel_y, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_vel_next_z, bodies->d_grad_vel_z, size, cudaMemcpyDeviceToDevice));

        // Adjoint of integration step
        launchIntegrateAdjoint(
            bodies->d_grad_pos_x, bodies->d_grad_pos_y, bodies->d_grad_pos_z,
            bodies->d_grad_vel_x, bodies->d_grad_vel_y, bodies->d_grad_vel_z,
            d_grad_acc_x, d_grad_acc_y, d_grad_acc_z,
            d_grad_pos_next_x, d_grad_pos_next_y, d_grad_pos_next_z,
            d_grad_vel_next_x, d_grad_vel_next_y, d_grad_vel_next_z,
            bodies->n, params.time_step, compute_stream
        );

        // Adjoint of force computation for position gradients
        launchComputeForcesAdjoint(
            bodies->d_grad_pos_x, bodies->d_grad_pos_y, bodies->d_grad_pos_z,
            d_grad_acc_x, d_grad_acc_y, d_grad_acc_z,
            bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
            bodies->d_mass, bodies->n,
            params.epsilon * params.epsilon, params.G,
            compute_stream
        );

        // Parameter adjoint for G, epsilon, and masses
        launchComputeForcesParameterAdjoint(
            bodies->d_grad_mass, bodies->d_grad_G, bodies->d_grad_epsilon,
            d_grad_acc_x, d_grad_acc_y, d_grad_acc_z,
            bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
            bodies->d_mass, bodies->n,
            params.epsilon * params.epsilon, params.G,
            compute_stream
        );

        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // Clean up temporary arrays
        cudaFree(d_grad_acc_x);
        cudaFree(d_grad_acc_y);
        cudaFree(d_grad_acc_z);
        cudaFree(d_grad_pos_next_x);
        cudaFree(d_grad_pos_next_y);
        cudaFree(d_grad_pos_next_z);
        cudaFree(d_grad_vel_next_x);
        cudaFree(d_grad_vel_next_y);
        cudaFree(d_grad_vel_next_z);
    }

    // Retrieve parameter gradients
    bodies->getParameterGradients(grad_mass, grad_G, grad_epsilon);

    return loss;
}

float Simulation::computeParameterGradientsWithTime(const std::vector<float>& target_pos_x,
                                                   const std::vector<float>& target_pos_y,
                                                   const std::vector<float>& target_pos_z,
                                                   std::vector<float>& grad_mass,
                                                   float& grad_G,
                                                   float& grad_epsilon,
                                                   float& grad_dt) {

    if (!parameter_gradients_enabled) {
        std::cerr << "Parameter gradients not enabled!\n";
        return 0.0f;
    }

    // First compute the regular loss and gradients
    float loss = computeGradients(target_pos_x, target_pos_y, target_pos_z);

    // Zero parameter gradients (including time step)
    bodies->zeroParameterGradients();

    // Backpropagate through time, accumulating parameter gradients
    size_t size = bodies->n * sizeof(float);

    for (int step = tape.size() - 1; step >= 0; step--) {
        const auto& state = tape.getState(step);

        // Restore state for adjoint computation
        CUDA_CHECK(cudaMemcpy(bodies->d_pos_x, state.pos_x.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_pos_y, state.pos_y.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_pos_z, state.pos_z.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_vel_x, state.vel_x.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_vel_y, state.vel_y.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bodies->d_vel_z, state.vel_z.data(), size, cudaMemcpyHostToDevice));

        // CRITICAL: Recompute forces for this restored state
        // The time step gradient kernel needs the correct acceleration values
        launchComputeForces(
            bodies->d_acc_x, bodies->d_acc_y, bodies->d_acc_z,
            bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
            bodies->d_mass, bodies->n,
            params.epsilon * params.epsilon, params.G,
            compute_stream
        );

        // CRITICAL: Also need to compute the velocity AFTER the update for time step gradients
        // The tape stores velocities BEFORE the step, but we need velocities AFTER
        // We'll store the updated velocities in temporary arrays
        float *d_vel_after_x, *d_vel_after_y, *d_vel_after_z;
        CUDA_CHECK(cudaMalloc(&d_vel_after_x, size));
        CUDA_CHECK(cudaMalloc(&d_vel_after_y, size));
        CUDA_CHECK(cudaMalloc(&d_vel_after_z, size));

        // Copy current (before) velocities to temporary
        CUDA_CHECK(cudaMemcpy(d_vel_after_x, bodies->d_vel_x, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel_after_y, bodies->d_vel_y, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel_after_z, bodies->d_vel_z, size, cudaMemcpyDeviceToDevice));

        // Apply velocity update: vel_new = vel_old + acc * dt
        int threads = 256;
        int blocks = (bodies->n + threads - 1) / threads;

        updateVelocitiesKernel<<<blocks, threads, 0, compute_stream>>>(
            d_vel_after_x, d_vel_after_y, d_vel_after_z,
            bodies->d_acc_x, bodies->d_acc_y, bodies->d_acc_z,
            bodies->n, params.time_step
        );

        // Allocate temporary gradient arrays for acceleration
        float *d_grad_acc_x, *d_grad_acc_y, *d_grad_acc_z;
        float *d_grad_pos_next_x, *d_grad_pos_next_y, *d_grad_pos_next_z;
        float *d_grad_vel_next_x, *d_grad_vel_next_y, *d_grad_vel_next_z;

        CUDA_CHECK(cudaMalloc(&d_grad_acc_x, size));
        CUDA_CHECK(cudaMalloc(&d_grad_acc_y, size));
        CUDA_CHECK(cudaMalloc(&d_grad_acc_z, size));
        CUDA_CHECK(cudaMalloc(&d_grad_pos_next_x, size));
        CUDA_CHECK(cudaMalloc(&d_grad_pos_next_y, size));
        CUDA_CHECK(cudaMalloc(&d_grad_pos_next_z, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_next_x, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_next_y, size));
        CUDA_CHECK(cudaMalloc(&d_grad_vel_next_z, size));

        CUDA_CHECK(cudaMemset(d_grad_acc_x, 0, size));
        CUDA_CHECK(cudaMemset(d_grad_acc_y, 0, size));
        CUDA_CHECK(cudaMemset(d_grad_acc_z, 0, size));
        CUDA_CHECK(cudaMemcpy(d_grad_pos_next_x, bodies->d_grad_pos_x, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_pos_next_y, bodies->d_grad_pos_y, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_pos_next_z, bodies->d_grad_pos_z, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_vel_next_x, bodies->d_grad_vel_x, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_vel_next_y, bodies->d_grad_vel_y, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_vel_next_z, bodies->d_grad_vel_z, size, cudaMemcpyDeviceToDevice));

        // Time step gradient computation (before integration adjoint)
        // Use the UPDATED velocities (after the step) not the original ones
        integrateLeapfrogTimeAdjoint<<<blocks, threads, 0, compute_stream>>>(
            bodies->d_grad_dt,
            d_grad_pos_next_x, d_grad_pos_next_y, d_grad_pos_next_z,
            d_grad_vel_next_x, d_grad_vel_next_y, d_grad_vel_next_z,
            d_vel_after_x, d_vel_after_y, d_vel_after_z,  // Use updated velocities!
            bodies->d_acc_x, bodies->d_acc_y, bodies->d_acc_z,
            bodies->n, params.time_step
        );

        // Adjoint of integration step
        launchIntegrateAdjoint(
            bodies->d_grad_pos_x, bodies->d_grad_pos_y, bodies->d_grad_pos_z,
            bodies->d_grad_vel_x, bodies->d_grad_vel_y, bodies->d_grad_vel_z,
            d_grad_acc_x, d_grad_acc_y, d_grad_acc_z,
            d_grad_pos_next_x, d_grad_pos_next_y, d_grad_pos_next_z,
            d_grad_vel_next_x, d_grad_vel_next_y, d_grad_vel_next_z,
            bodies->n, params.time_step, compute_stream
        );

        // Adjoint of force computation for position gradients
        launchComputeForcesAdjoint(
            bodies->d_grad_pos_x, bodies->d_grad_pos_y, bodies->d_grad_pos_z,
            d_grad_acc_x, d_grad_acc_y, d_grad_acc_z,
            bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
            bodies->d_mass, bodies->n,
            params.epsilon * params.epsilon, params.G,
            compute_stream
        );

        // Parameter adjoint for G, epsilon, and masses
        launchComputeForcesParameterAdjoint(
            bodies->d_grad_mass, bodies->d_grad_G, bodies->d_grad_epsilon,
            d_grad_acc_x, d_grad_acc_y, d_grad_acc_z,
            bodies->d_pos_x, bodies->d_pos_y, bodies->d_pos_z,
            bodies->d_mass, bodies->n,
            params.epsilon * params.epsilon, params.G,
            compute_stream
        );

        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // Clean up temporary arrays
        cudaFree(d_grad_acc_x);
        cudaFree(d_grad_acc_y);
        cudaFree(d_grad_acc_z);
        cudaFree(d_grad_pos_next_x);
        cudaFree(d_grad_pos_next_y);
        cudaFree(d_grad_pos_next_z);
        cudaFree(d_grad_vel_next_x);
        cudaFree(d_grad_vel_next_y);
        cudaFree(d_grad_vel_next_z);
        cudaFree(d_vel_after_x);
        cudaFree(d_vel_after_y);
        cudaFree(d_vel_after_z);
    }

    // Retrieve parameter gradients including time step
    bodies->getParameterGradientsWithTime(grad_mass, grad_G, grad_epsilon, grad_dt);

    return loss;
}


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

void launchComputeForcesAdjoint(
    float* d_grad_pos_x, float* d_grad_pos_y, float* d_grad_pos_z,
    const float* d_grad_acc_x, const float* d_grad_acc_y, const float* d_grad_acc_z,
    const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
    const float* d_mass, int n, float epsilon, float G,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    computeForcesAdjoint<<<blocks, threads, 0, stream>>>(
        d_grad_pos_x, d_grad_pos_y, d_grad_pos_z,
        d_grad_acc_x, d_grad_acc_y, d_grad_acc_z,
        d_pos_x, d_pos_y, d_pos_z,
        d_mass, n, epsilon, G
    );

    CUDA_CHECK(cudaGetLastError());
}

void launchComputeForcesParameterAdjoint(
    float* d_grad_mass, float* d_grad_G, float* d_grad_epsilon,
    const float* d_grad_acc_x, const float* d_grad_acc_y, const float* d_grad_acc_z,
    const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
    const float* d_mass, int n, float epsilon, float G,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    computeForcesParameterAdjoint<<<blocks, threads, 0, stream>>>(
        d_grad_mass, d_grad_G, d_grad_epsilon,
        d_grad_acc_x, d_grad_acc_y, d_grad_acc_z,
        d_pos_x, d_pos_y, d_pos_z,
        d_mass, n, epsilon, G
    );

    CUDA_CHECK(cudaGetLastError());
}

void launchIntegrateAdjoint(
    float* d_grad_pos_x, float* d_grad_pos_y, float* d_grad_pos_z,
    float* d_grad_vel_x, float* d_grad_vel_y, float* d_grad_vel_z,
    float* d_grad_acc_x, float* d_grad_acc_y, float* d_grad_acc_z,
    const float* d_grad_pos_next_x, const float* d_grad_pos_next_y, const float* d_grad_pos_next_z,
    const float* d_grad_vel_next_x, const float* d_grad_vel_next_y, const float* d_grad_vel_next_z,
    int n, float dt, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    integrateLeapfrogAdjoint<<<blocks, threads, 0, stream>>>(
        d_grad_pos_x, d_grad_pos_y, d_grad_pos_z,
        d_grad_vel_x, d_grad_vel_y, d_grad_vel_z,
        d_grad_acc_x, d_grad_acc_y, d_grad_acc_z,
        d_grad_pos_next_x, d_grad_pos_next_y, d_grad_pos_next_z,
        d_grad_vel_next_x, d_grad_vel_next_y, d_grad_vel_next_z,
        n, dt
    );

    CUDA_CHECK(cudaGetLastError());
}

void Simulation::enableStableForces(bool enable) {
    stable_forces_enabled = enable;
}

void Simulation::stabilizeGradients() {
    if (bodies->d_grad_pos_x && bodies->d_grad_pos_y && bodies->d_grad_pos_z) {
        physgrad::stabilizeGradients(bodies->d_grad_pos_x, bodies->d_grad_pos_y, bodies->d_grad_pos_z,
                                    bodies->n, params.stability);
    }
}

} // namespace physgrad