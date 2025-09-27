#include "stability_improvements.h"
#include "simulation.h"
#include <algorithm>
#include <cmath>

namespace physgrad {

__global__ void clipGradients(
    float* grad_x, float* grad_y, float* grad_z,
    int n, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float gx = grad_x[i];
    float gy = grad_y[i];
    float gz = grad_z[i];

    float norm = sqrtf(gx*gx + gy*gy + gz*gz);

    if (norm > threshold) {
        float scale = threshold / norm;
        grad_x[i] = gx * scale;
        grad_y[i] = gy * scale;
        grad_z[i] = gz * scale;
    }
}

__global__ void normalizeGradients(
    float* grad_x, float* grad_y, float* grad_z,
    int n, float target_norm)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Compute total norm in shared memory reduction
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    float local_norm_sq = 0.0f;
    if (i < n) {
        float gx = grad_x[i];
        float gy = grad_y[i];
        float gz = grad_z[i];
        local_norm_sq = gx*gx + gy*gy + gz*gz;
    }

    sdata[tid] = local_norm_sq;
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Store partial sum for this block
    __shared__ float total_norm_sq;
    if (tid == 0) {
        total_norm_sq = sdata[0];
    }
    __syncthreads();

    // Note: This is a simplified version. For proper normalization across all blocks,
    // we'd need a two-pass approach or use cooperative groups.
    // For now, we'll approximate by using the block's contribution.

    if (i < n && total_norm_sq > 0.0f) {
        float current_norm = sqrtf(total_norm_sq);
        float scale = target_norm / current_norm;

        grad_x[i] *= scale;
        grad_y[i] *= scale;
        grad_z[i] *= scale;
    }
}

__global__ void adaptiveEpsilonKernel(
    float* d_acc_x, float* d_acc_y, float* d_acc_z,
    const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
    const float* d_mass, int n, float base_epsilon, float adaptive_factor, float G)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = d_pos_x[i];
    float yi = d_pos_y[i];
    float zi = d_pos_z[i];

    float acc_x = 0.0f;
    float acc_y = 0.0f;
    float acc_z = 0.0f;

    #pragma unroll 4
    for (int j = 0; j < n; j++) {
        if (i == j) continue;

        float dx = d_pos_x[j] - xi;
        float dy = d_pos_y[j] - yi;
        float dz = d_pos_z[j] - zi;

        float dist_sq = dx*dx + dy*dy + dz*dz;

        // Adaptive epsilon based on minimum distance
        float min_dist = sqrtf(dist_sq);
        float adaptive_eps = base_epsilon + adaptive_factor * expf(-min_dist * 10.0f);
        float epsilon_sq = adaptive_eps * adaptive_eps;

        float inv_dist = rsqrtf(dist_sq + epsilon_sq);
        float inv_dist_cube = inv_dist * inv_dist * inv_dist;

        float force_mag = G * d_mass[j] * inv_dist_cube;

        acc_x += force_mag * dx;
        acc_y += force_mag * dy;
        acc_z += force_mag * dz;
    }

    d_acc_x[i] = acc_x;
    d_acc_y[i] = acc_y;
    d_acc_z[i] = acc_z;
}

float computeGradientNorm(const float* grad_x, const float* grad_y, const float* grad_z, int n) {
    // Copy gradients to host for norm computation
    std::vector<float> host_grad_x(n), host_grad_y(n), host_grad_z(n);

    cudaMemcpy(host_grad_x.data(), grad_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_grad_y.data(), grad_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_grad_z.data(), grad_z, n * sizeof(float), cudaMemcpyDeviceToHost);

    float norm_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        norm_sq += host_grad_x[i] * host_grad_x[i];
        norm_sq += host_grad_y[i] * host_grad_y[i];
        norm_sq += host_grad_z[i] * host_grad_z[i];
    }

    return std::sqrt(norm_sq);
}

void stabilizeGradients(float* grad_x, float* grad_y, float* grad_z, int n, const StabilityParams& params) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    if (params.use_gradient_clipping) {
        // Check if clipping is needed
        float norm = computeGradientNorm(grad_x, grad_y, grad_z, n);

        if (norm > params.gradient_clipping_threshold) {
            clipGradients<<<blocks, threads>>>(
                grad_x, grad_y, grad_z, n, params.gradient_clipping_threshold);
            cudaDeviceSynchronize();
        }
    }
}

// Enhanced force computation with numerical stability
__global__ void computeForcesStable(
    float* d_acc_x, float* d_acc_y, float* d_acc_z,
    const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
    const float* d_mass, int n, float epsilon, float G, float max_force)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = d_pos_x[i];
    float yi = d_pos_y[i];
    float zi = d_pos_z[i];

    float acc_x = 0.0f;
    float acc_y = 0.0f;
    float acc_z = 0.0f;

    #pragma unroll 4
    for (int j = 0; j < n; j++) {
        if (i == j) continue;

        float dx = d_pos_x[j] - xi;
        float dy = d_pos_y[j] - yi;
        float dz = d_pos_z[j] - zi;

        float dist_sq = dx*dx + dy*dy + dz*dz;
        float epsilon_sq = epsilon * epsilon;

        // Numerical stability: avoid division by very small numbers
        float regularized_dist_sq = fmaxf(dist_sq + epsilon_sq, epsilon_sq);

        float inv_dist = rsqrtf(regularized_dist_sq);
        float inv_dist_cube = inv_dist * inv_dist * inv_dist;

        float force_mag = G * d_mass[j] * inv_dist_cube;

        // Clamp force magnitude to prevent numerical explosion
        force_mag = fminf(force_mag, max_force);

        acc_x += force_mag * dx;
        acc_y += force_mag * dy;
        acc_z += force_mag * dz;
    }

    d_acc_x[i] = acc_x;
    d_acc_y[i] = acc_y;
    d_acc_z[i] = acc_z;
}

void launchComputeForcesStable(float* d_acc_x, float* d_acc_y, float* d_acc_z,
                              const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
                              const float* d_mass, int n, float epsilon, float G,
                              float max_force, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    computeForcesStable<<<blocks, threads, 0, stream>>>(
        d_acc_x, d_acc_y, d_acc_z,
        d_pos_x, d_pos_y, d_pos_z,
        d_mass, n, epsilon, G, max_force);
}

} // namespace physgrad