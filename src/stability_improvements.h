#pragma once

#include <cuda_runtime.h>

namespace physgrad {

// Numerical stability improvements
struct StabilityParams {
    float gradient_clipping_threshold = 10.0f;  // Clip gradients above this magnitude
    float loss_regularization = 1e-6f;          // Regularization term for loss
    bool use_gradient_clipping = true;          // Enable gradient clipping
    bool use_loss_regularization = true;       // Enable loss regularization
    float epsilon_adaptive = 1e-4f;            // Adaptive epsilon for conditioning
    bool use_adaptive_epsilon = true;          // Enable adaptive epsilon
};

// CUDA kernels for numerical stability
__global__ void clipGradients(
    float* grad_x, float* grad_y, float* grad_z,
    int n, float threshold);

__global__ void normalizeGradients(
    float* grad_x, float* grad_y, float* grad_z,
    int n, float target_norm);

__global__ void adaptiveEpsilon(
    float* d_acc_x, float* d_acc_y, float* d_acc_z,
    const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
    const float* d_mass, int n, float base_epsilon, float adaptive_factor, float G);

// Helper functions
float computeGradientNorm(const float* grad_x, const float* grad_y, const float* grad_z, int n);
void stabilizeGradients(float* grad_x, float* grad_y, float* grad_z, int n, const StabilityParams& params);

} // namespace physgrad