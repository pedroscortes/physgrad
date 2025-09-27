#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA devices" << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err == cudaSuccess) {
            std::cout << "Device " << i << ": " << prop.name << std::endl;
            std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        }
    }

    // Test simple kernel
    float *d_test;
    err = cudaMalloc(&d_test, 1024 * sizeof(float));
    if (err == cudaSuccess) {
        std::cout << "CUDA malloc test: SUCCESS" << std::endl;
        cudaFree(d_test);
    } else {
        std::cout << "CUDA malloc test: FAILED - " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}