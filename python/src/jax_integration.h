#pragma once

#ifdef WITH_JAX

#include <string>
#include <vector>
#include <memory>

// JAX XLA headers
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/c_api.h"
#include "xla/service/custom_call_target_registry.h"

// CUDA headers
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace jax_integration {

// XLA FFI function signatures for physics operations
XLA_FFI_DECLARE_HANDLER_SYMBOL(PhysicsStepHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(ForceComputationHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(ConstraintHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CollisionDetectionHandler);

// JAX primitive registration
void registerJAXPrimitives();

// Custom call implementations
void* simulationStepCustomCall(cudaStream_t stream, void** buffers,
                              const char* opaque, size_t opaque_len);

void* forceComputationCustomCall(cudaStream_t stream, void** buffers,
                                const char* opaque, size_t opaque_len);

void* constraintCustomCall(cudaStream_t stream, void** buffers,
                          const char* opaque, size_t opaque_len);

void* collisionDetectionCustomCall(cudaStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len);

// JAX array interface
struct JAXArray {
    void* data;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int element_size;

    JAXArray(void* ptr, const std::vector<int64_t>& dims, int elem_size)
        : data(ptr), shape(dims), element_size(elem_size) {
        calculateStrides();
    }

private:
    void calculateStrides();
};

// High-level JAX functions
JAXArray simulationStepJAX(const JAXArray& positions, const JAXArray& velocities,
                          const JAXArray& masses, const JAXArray& forces, float dt);

JAXArray computeForcesJAX(const JAXArray& positions, const JAXArray& velocities,
                         const JAXArray& masses);

std::pair<JAXArray, JAXArray> applyConstraintsJAX(const JAXArray& positions,
                                                  const JAXArray& velocities,
                                                  const JAXArray& masses,
                                                  const JAXArray& constraint_params);

// Utility functions
std::string serializePhysicsParams(float dt, int num_particles);
void deserializePhysicsParams(const std::string& serialized, float& dt, int& num_particles);

// Memory management for JAX arrays
class JAXMemoryPool {
private:
    std::vector<std::unique_ptr<char[]>> allocated_buffers;
    size_t total_allocated = 0;

public:
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void clear();
    size_t getTotalAllocated() const { return total_allocated; }
};

// Global memory pool
extern JAXMemoryPool g_jax_memory_pool;

// Error handling
class JAXIntegrationError : public std::exception {
private:
    std::string message;

public:
    JAXIntegrationError(const std::string& msg) : message(msg) {}
    const char* what() const noexcept override { return message.c_str(); }
};

// Registration helpers
void registerPhysicsStepPrimitive();
void registerForceComputationPrimitive();
void registerConstraintPrimitive();
void registerCollisionDetectionPrimitive();

} // namespace jax_integration

#endif // WITH_JAX