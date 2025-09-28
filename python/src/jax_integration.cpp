#include "jax_integration.h"

#ifdef WITH_JAX

#include <sstream>
#include <cstring>
#include <algorithm>

// PhysGrad headers
extern "C" {
    void launch_physics_step_kernel(
        float* pos_x, float* pos_y, float* pos_z,
        float* vel_x, float* vel_y, float* vel_z,
        const float* forces_x, const float* forces_y, const float* forces_z,
        const float* masses, int num_particles, float dt,
        cudaStream_t stream
    );

    void launch_force_computation_kernel(
        const float* pos_x, const float* pos_y, const float* pos_z,
        const float* vel_x, const float* vel_y, const float* vel_z,
        float* forces_x, float* forces_y, float* forces_z,
        const float* masses, int num_particles,
        cudaStream_t stream
    );

    void launch_constraint_kernel(
        float* pos_x, float* pos_y, float* pos_z,
        float* vel_x, float* vel_y, float* vel_z,
        const float* masses, int num_particles,
        const float* constraint_params, int num_constraints,
        float dt, cudaStream_t stream
    );
}

namespace jax_integration {

// Global memory pool instance
JAXMemoryPool g_jax_memory_pool;

// JAXArray implementation
void JAXArray::calculateStrides() {
    strides.resize(shape.size());
    if (shape.empty()) return;

    strides.back() = element_size;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// Memory pool implementation
void* JAXMemoryPool::allocate(size_t size) {
    auto buffer = std::make_unique<char[]>(size);
    void* ptr = buffer.get();
    allocated_buffers.push_back(std::move(buffer));
    total_allocated += size;
    return ptr;
}

void JAXMemoryPool::deallocate(void* ptr) {
    // Find and remove the buffer
    allocated_buffers.erase(
        std::remove_if(allocated_buffers.begin(), allocated_buffers.end(),
                      [ptr](const std::unique_ptr<char[]>& buffer) {
                          return buffer.get() == ptr;
                      }),
        allocated_buffers.end()
    );
}

void JAXMemoryPool::clear() {
    allocated_buffers.clear();
    total_allocated = 0;
}

// Serialization utilities
std::string serializePhysicsParams(float dt, int num_particles) {
    std::ostringstream oss;
    oss.write(reinterpret_cast<const char*>(&dt), sizeof(float));
    oss.write(reinterpret_cast<const char*>(&num_particles), sizeof(int));
    return oss.str();
}

void deserializePhysicsParams(const std::string& serialized, float& dt, int& num_particles) {
    std::istringstream iss(serialized);
    iss.read(reinterpret_cast<char*>(&dt), sizeof(float));
    iss.read(reinterpret_cast<char*>(&num_particles), sizeof(int));
}

// XLA FFI Handlers
XLA_FFI_DEFINE_HANDLER(PhysicsStepHandler, PhysicsStepHandlerImpl,
                      XLA_FFI_HANDLER_TRAITS(ffi::Ffi::DataTypes<float>()));

XLA_FFI_Error* PhysicsStepHandlerImpl(XLA_FFI_CallFrame* call_frame) {
    auto call = xla::ffi::Call::Create(call_frame);
    if (!call.ok()) {
        return new XLA_FFI_Error{XLA_FFI_Error_Code_INTERNAL,
                                call.status().message().data()};
    }

    // Extract input buffers
    auto positions = call->arg(0).flat<float>();
    auto velocities = call->arg(1).flat<float>();
    auto masses = call->arg(2).flat<float>();
    auto forces = call->arg(3).flat<float>();

    // Extract output buffers
    auto new_positions = call->ret(0).flat<float>();
    auto new_velocities = call->ret(1).flat<float>();

    // Extract attributes
    auto dt_attr = call->attr<float>("dt");
    auto num_particles_attr = call->attr<int64_t>("num_particles");

    if (!dt_attr.ok() || !num_particles_attr.ok()) {
        return new XLA_FFI_Error{XLA_FFI_Error_Code_INVALID_ARGUMENT,
                                "Missing required attributes"};
    }

    float dt = dt_attr.value();
    int num_particles = static_cast<int>(num_particles_attr.value());

    // Get CUDA stream
    auto stream = call->execution_context()->gpu_stream;

    // Copy input to output first
    cudaMemcpyAsync(new_positions.data(), positions.data(),
                   num_particles * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(new_velocities.data(), velocities.data(),
                   num_particles * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);

    // Launch physics step kernel
    launch_physics_step_kernel(
        new_positions.data(),                     // pos_x
        new_positions.data() + num_particles,     // pos_y
        new_positions.data() + 2 * num_particles, // pos_z
        new_velocities.data(),                    // vel_x
        new_velocities.data() + num_particles,    // vel_y
        new_velocities.data() + 2 * num_particles, // vel_z
        forces.data(),                            // force_x
        forces.data() + num_particles,            // force_y
        forces.data() + 2 * num_particles,        // force_z
        masses.data(),
        num_particles,
        dt,
        stream
    );

    return nullptr; // Success
}

XLA_FFI_DEFINE_HANDLER(ForceComputationHandler, ForceComputationHandlerImpl,
                      XLA_FFI_HANDLER_TRAITS(ffi::Ffi::DataTypes<float>()));

XLA_FFI_Error* ForceComputationHandlerImpl(XLA_FFI_CallFrame* call_frame) {
    auto call = xla::ffi::Call::Create(call_frame);
    if (!call.ok()) {
        return new XLA_FFI_Error{XLA_FFI_Error_Code_INTERNAL,
                                call.status().message().data()};
    }

    // Extract input buffers
    auto positions = call->arg(0).flat<float>();
    auto velocities = call->arg(1).flat<float>();
    auto masses = call->arg(2).flat<float>();

    // Extract output buffer
    auto forces = call->ret(0).flat<float>();

    // Extract attributes
    auto num_particles_attr = call->attr<int64_t>("num_particles");
    if (!num_particles_attr.ok()) {
        return new XLA_FFI_Error{XLA_FFI_Error_Code_INVALID_ARGUMENT,
                                "Missing num_particles attribute"};
    }

    int num_particles = static_cast<int>(num_particles_attr.value());
    auto stream = call->execution_context()->gpu_stream;

    // Launch force computation kernel
    launch_force_computation_kernel(
        positions.data(),                         // pos_x
        positions.data() + num_particles,         // pos_y
        positions.data() + 2 * num_particles,     // pos_z
        velocities.data(),                        // vel_x
        velocities.data() + num_particles,        // vel_y
        velocities.data() + 2 * num_particles,    // vel_z
        forces.data(),                            // force_x
        forces.data() + num_particles,            // force_y
        forces.data() + 2 * num_particles,        // force_z
        masses.data(),
        num_particles,
        stream
    );

    return nullptr;
}

XLA_FFI_DEFINE_HANDLER(ConstraintHandler, ConstraintHandlerImpl,
                      XLA_FFI_HANDLER_TRAITS(ffi::Ffi::DataTypes<float>()));

XLA_FFI_Error* ConstraintHandlerImpl(XLA_FFI_CallFrame* call_frame) {
    auto call = xla::ffi::Call::Create(call_frame);
    if (!call.ok()) {
        return new XLA_FFI_Error{XLA_FFI_Error_Code_INTERNAL,
                                call.status().message().data()};
    }

    // Extract input buffers
    auto positions = call->arg(0).flat<float>();
    auto velocities = call->arg(1).flat<float>();
    auto masses = call->arg(2).flat<float>();
    auto constraint_params = call->arg(3).flat<float>();

    // Extract output buffers
    auto new_positions = call->ret(0).flat<float>();
    auto new_velocities = call->ret(1).flat<float>();

    // Extract attributes
    auto dt_attr = call->attr<float>("dt");
    auto num_particles_attr = call->attr<int64_t>("num_particles");
    auto num_constraints_attr = call->attr<int64_t>("num_constraints");

    if (!dt_attr.ok() || !num_particles_attr.ok() || !num_constraints_attr.ok()) {
        return new XLA_FFI_Error{XLA_FFI_Error_Code_INVALID_ARGUMENT,
                                "Missing required attributes"};
    }

    float dt = dt_attr.value();
    int num_particles = static_cast<int>(num_particles_attr.value());
    int num_constraints = static_cast<int>(num_constraints_attr.value());

    auto stream = call->execution_context()->gpu_stream;

    // Copy input to output
    cudaMemcpyAsync(new_positions.data(), positions.data(),
                   num_particles * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(new_velocities.data(), velocities.data(),
                   num_particles * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);

    // Launch constraint kernel
    launch_constraint_kernel(
        new_positions.data(),                     // pos_x
        new_positions.data() + num_particles,     // pos_y
        new_positions.data() + 2 * num_particles, // pos_z
        new_velocities.data(),                    // vel_x
        new_velocities.data() + num_particles,    // vel_y
        new_velocities.data() + 2 * num_particles, // vel_z
        masses.data(),
        num_particles,
        constraint_params.data(),
        num_constraints,
        dt,
        stream
    );

    return nullptr;
}

// Custom call implementations (legacy interface)
void* simulationStepCustomCall(cudaStream_t stream, void** buffers,
                              const char* opaque, size_t opaque_len) {
    // Parse opaque data
    float dt;
    int num_particles;
    std::string opaque_str(opaque, opaque_len);
    deserializePhysicsParams(opaque_str, dt, num_particles);

    // Extract buffer pointers
    float* positions = static_cast<float*>(buffers[0]);
    float* velocities = static_cast<float*>(buffers[1]);
    float* masses = static_cast<float*>(buffers[2]);
    float* forces = static_cast<float*>(buffers[3]);
    float* new_positions = static_cast<float*>(buffers[4]);
    float* new_velocities = static_cast<float*>(buffers[5]);

    // Copy input to output
    cudaMemcpyAsync(new_positions, positions, num_particles * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(new_velocities, velocities, num_particles * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);

    // Launch kernel
    launch_physics_step_kernel(
        new_positions,                           // pos_x
        new_positions + num_particles,           // pos_y
        new_positions + 2 * num_particles,       // pos_z
        new_velocities,                          // vel_x
        new_velocities + num_particles,          // vel_y
        new_velocities + 2 * num_particles,      // vel_z
        forces,                                  // force_x
        forces + num_particles,                  // force_y
        forces + 2 * num_particles,              // force_z
        masses,
        num_particles,
        dt,
        stream
    );

    return nullptr;
}

void* forceComputationCustomCall(cudaStream_t stream, void** buffers,
                                const char* opaque, size_t opaque_len) {
    int num_particles = *reinterpret_cast<const int*>(opaque);

    float* positions = static_cast<float*>(buffers[0]);
    float* velocities = static_cast<float*>(buffers[1]);
    float* masses = static_cast<float*>(buffers[2]);
    float* forces = static_cast<float*>(buffers[3]);

    launch_force_computation_kernel(
        positions,                               // pos_x
        positions + num_particles,               // pos_y
        positions + 2 * num_particles,           // pos_z
        velocities,                              // vel_x
        velocities + num_particles,              // vel_y
        velocities + 2 * num_particles,          // vel_z
        forces,                                  // force_x
        forces + num_particles,                  // force_y
        forces + 2 * num_particles,              // force_z
        masses,
        num_particles,
        stream
    );

    return nullptr;
}

void* constraintCustomCall(cudaStream_t stream, void** buffers,
                          const char* opaque, size_t opaque_len) {
    // Parse opaque data: dt, num_particles, num_constraints
    std::istringstream iss(std::string(opaque, opaque_len));
    float dt;
    int num_particles, num_constraints;
    iss.read(reinterpret_cast<char*>(&dt), sizeof(float));
    iss.read(reinterpret_cast<char*>(&num_particles), sizeof(int));
    iss.read(reinterpret_cast<char*>(&num_constraints), sizeof(int));

    float* positions = static_cast<float*>(buffers[0]);
    float* velocities = static_cast<float*>(buffers[1]);
    float* masses = static_cast<float*>(buffers[2]);
    float* constraint_params = static_cast<float*>(buffers[3]);
    float* new_positions = static_cast<float*>(buffers[4]);
    float* new_velocities = static_cast<float*>(buffers[5]);

    // Copy input to output
    cudaMemcpyAsync(new_positions, positions, num_particles * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(new_velocities, velocities, num_particles * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);

    launch_constraint_kernel(
        new_positions,                           // pos_x
        new_positions + num_particles,           // pos_y
        new_positions + 2 * num_particles,       // pos_z
        new_velocities,                          // vel_x
        new_velocities + num_particles,          // vel_y
        new_velocities + 2 * num_particles,      // vel_z
        masses,
        num_particles,
        constraint_params,
        num_constraints,
        dt,
        stream
    );

    return nullptr;
}

void* collisionDetectionCustomCall(cudaStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len) {
    // Placeholder for collision detection
    // This would need the actual collision detection kernel
    return nullptr;
}

// High-level JAX functions
JAXArray simulationStepJAX(const JAXArray& positions, const JAXArray& velocities,
                          const JAXArray& masses, const JAXArray& forces, float dt) {
    if (positions.shape.size() != 2 || positions.shape[1] != 3) {
        throw JAXIntegrationError("Positions must have shape [N, 3]");
    }

    int num_particles = positions.shape[0];

    // Allocate output arrays
    size_t pos_size = num_particles * 3 * sizeof(float);
    void* new_pos_data = g_jax_memory_pool.allocate(pos_size);
    void* new_vel_data = g_jax_memory_pool.allocate(pos_size);

    JAXArray new_positions(new_pos_data, positions.shape, sizeof(float));
    JAXArray new_velocities(new_vel_data, velocities.shape, sizeof(float));

    // Get current CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Copy input to output
    cudaMemcpyAsync(new_positions.data, positions.data, pos_size,
                   cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(new_velocities.data, velocities.data, pos_size,
                   cudaMemcpyDeviceToDevice, stream);

    // Launch kernel
    launch_physics_step_kernel(
        static_cast<float*>(new_positions.data),                     // pos_x
        static_cast<float*>(new_positions.data) + num_particles,     // pos_y
        static_cast<float*>(new_positions.data) + 2 * num_particles, // pos_z
        static_cast<float*>(new_velocities.data),                    // vel_x
        static_cast<float*>(new_velocities.data) + num_particles,    // vel_y
        static_cast<float*>(new_velocities.data) + 2 * num_particles, // vel_z
        static_cast<const float*>(forces.data),                      // force_x
        static_cast<const float*>(forces.data) + num_particles,      // force_y
        static_cast<const float*>(forces.data) + 2 * num_particles,  // force_z
        static_cast<const float*>(masses.data),
        num_particles,
        dt,
        stream
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return new_positions; // For simplicity, return only positions
}

JAXArray computeForcesJAX(const JAXArray& positions, const JAXArray& velocities,
                         const JAXArray& masses) {
    int num_particles = positions.shape[0];
    size_t force_size = num_particles * 3 * sizeof(float);

    void* force_data = g_jax_memory_pool.allocate(force_size);
    JAXArray forces(force_data, positions.shape, sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    launch_force_computation_kernel(
        static_cast<const float*>(positions.data),                   // pos_x
        static_cast<const float*>(positions.data) + num_particles,   // pos_y
        static_cast<const float*>(positions.data) + 2 * num_particles, // pos_z
        static_cast<const float*>(velocities.data),                  // vel_x
        static_cast<const float*>(velocities.data) + num_particles,  // vel_y
        static_cast<const float*>(velocities.data) + 2 * num_particles, // vel_z
        static_cast<float*>(forces.data),                            // force_x
        static_cast<float*>(forces.data) + num_particles,            // force_y
        static_cast<float*>(forces.data) + 2 * num_particles,        // force_z
        static_cast<const float*>(masses.data),
        num_particles,
        stream
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return forces;
}

std::pair<JAXArray, JAXArray> applyConstraintsJAX(const JAXArray& positions,
                                                  const JAXArray& velocities,
                                                  const JAXArray& masses,
                                                  const JAXArray& constraint_params) {
    int num_particles = positions.shape[0];
    int num_constraints = constraint_params.shape[0];

    size_t pos_size = num_particles * 3 * sizeof(float);
    void* new_pos_data = g_jax_memory_pool.allocate(pos_size);
    void* new_vel_data = g_jax_memory_pool.allocate(pos_size);

    JAXArray new_positions(new_pos_data, positions.shape, sizeof(float));
    JAXArray new_velocities(new_vel_data, velocities.shape, sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Copy input to output
    cudaMemcpyAsync(new_positions.data, positions.data, pos_size,
                   cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(new_velocities.data, velocities.data, pos_size,
                   cudaMemcpyDeviceToDevice, stream);

    // Launch constraint kernel
    launch_constraint_kernel(
        static_cast<float*>(new_positions.data),                     // pos_x
        static_cast<float*>(new_positions.data) + num_particles,     // pos_y
        static_cast<float*>(new_positions.data) + 2 * num_particles, // pos_z
        static_cast<float*>(new_velocities.data),                    // vel_x
        static_cast<float*>(new_velocities.data) + num_particles,    // vel_y
        static_cast<float*>(new_velocities.data) + 2 * num_particles, // vel_z
        static_cast<const float*>(masses.data),
        num_particles,
        static_cast<const float*>(constraint_params.data),
        num_constraints,
        0.01f, // Fixed dt for now
        stream
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return std::make_pair(new_positions, new_velocities);
}

// Registration functions
void registerPhysicsStepPrimitive() {
    xla::CustomCallTargetRegistry::Global()->Register(
        "physgrad_physics_step", simulationStepCustomCall, "gpu");

    XLA_FFI_REGISTER_HANDLER("physgrad_physics_step_ffi", "GPU",
                            PhysicsStepHandler);
}

void registerForceComputationPrimitive() {
    xla::CustomCallTargetRegistry::Global()->Register(
        "physgrad_force_computation", forceComputationCustomCall, "gpu");

    XLA_FFI_REGISTER_HANDLER("physgrad_force_computation_ffi", "GPU",
                            ForceComputationHandler);
}

void registerConstraintPrimitive() {
    xla::CustomCallTargetRegistry::Global()->Register(
        "physgrad_constraints", constraintCustomCall, "gpu");

    XLA_FFI_REGISTER_HANDLER("physgrad_constraints_ffi", "GPU",
                            ConstraintHandler);
}

void registerCollisionDetectionPrimitive() {
    xla::CustomCallTargetRegistry::Global()->Register(
        "physgrad_collision_detection", collisionDetectionCustomCall, "gpu");
}

void registerJAXPrimitives() {
    registerPhysicsStepPrimitive();
    registerForceComputationPrimitive();
    registerConstraintPrimitive();
    registerCollisionDetectionPrimitive();
}

} // namespace jax_integration

#endif // WITH_JAX