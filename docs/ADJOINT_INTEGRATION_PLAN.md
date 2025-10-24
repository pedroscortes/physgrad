# Adjoint Integrator Integration Plan

## Goal
Merge `adjoint_integrators_standalone.h` improvements into main `adjoint_integrators.h` to create a production-ready, unified API.

## Key Features to Integrate

### From Standalone (adjoint_integrators_standalone.h)
1. ✅ **Bug Fixes**
   - Steps 3&5 disabled in backward pass (prevents 4× gradient error)
   - Analytical loss gradients for velocity-dependent losses
   - Proper epsilon scaling for numerical stability

2. ✅ **Parameter Gradients**
   - `ParameterGradients` struct for spring constants and rest lengths
   - `computeForceParameterGradients()` method
   - `computeAllGradients()` comprehensive API

3. ✅ **Analytical Loss Gradients**
   - Auto-detection for velocity-independent losses
   - Auto-detection for kinetic energy losses
   - User-provided custom analytical gradients

4. ✅ **SimpleForceEngine**
   - Self-contained force engine for testing
   - Spring force computations with proper Jacobians

### From Original (adjoint_integrators.h)
1. ✅ **Clean Architecture**
   - `AdjointStateManager` for state management
   - Separate checkpoint management
   - Better separation of concerns

2. ✅ **Concepts Support**
   - PhysicsScalar concept integration
   - Compile-time type checking

3. ✅ **DifferentiableForceEngine Interface**
   - Integration with main force engine abstraction
   - Works with production physics pipeline

4. ✅ **Leapfrog Integrator**
   - Alternative integration scheme
   - More memory efficient for some cases

## Integration Strategy

### Phase 1: Core Merge ✅
1. Keep clean architecture from original
2. Add bug fixes from standalone to backward pass
3. Add ParameterGradients support
4. Add analytical loss gradient support

### Phase 2: API Unification ✅
1. Create `AllGradients` struct
2. Implement `computeAllGradients()` method
3. Add auto-detection for loss gradients
4. Maintain backward compatibility

### Phase 3: Force Engine Abstraction ✅
1. Define `ForceEngineInterface` concept/trait
2. Make integrators work with any force engine
3. Keep SimpleForceEngine for testing
4. Integrate with DifferentiableForceEngine

### Phase 4: Documentation & Testing
1. Add comprehensive API documentation
2. Update existing tests
3. Create integration tests
4. Performance benchmarks

## File Structure

```
src/
├── adjoint_integrators.h          # NEW: Unified production version
├── adjoint_integrators_standalone.h  # KEEP: Self-contained for testing
├── adjoint_integrators_old.h      # BACKUP: Original version
├── loss_gradients.h               # KEEP: Analytical gradient library
└── adjoint_kernels.cu/h           # KEEP: CUDA implementations
```

## API Design

### Unified Gradient Computation API

```cpp
// Simple API: Just position and velocity gradients
auto [pos_grads, vel_grads] = simulation.computeGradients(
    initial_pos, initial_vel, masses, dt, num_steps,
    loss_function,
    analytical_loss_gradient  // Optional
);

// Comprehensive API: ALL gradients including parameters
auto all_grads = simulation.computeAllGradients(
    initial_pos, initial_vel, masses, dt, num_steps,
    loss_function,
    analytical_loss_gradient  // Optional
);

// Access results
auto& pos_grads = all_grads.position_grads;
auto& vel_grads = all_grads.velocity_grads;
auto& spring_k_grads = all_grads.parameter_grads.spring_constant_grads;
auto& spring_r0_grads = all_grads.parameter_grads.rest_length_grads;
```

### Force Engine Interface

```cpp
template<typename T>
concept ForceEngineInterface = requires(E engine, std::vector<ConceptVector3D<T>> pos) {
    { engine.computeForcesAndGradients(pos) } ->
        std::pair<std::vector<ConceptVector3D<T>>, ForceJacobian>;
    { engine.computeForceParameterGradients(pos, adjoints) } ->
        ParameterGradients;
};
```

## Testing Strategy

1. **Unit Tests**
   - Test each integrator independently
   - Validate gradient accuracy vs finite differences
   - Test parameter gradient computation

2. **Integration Tests**
   - Test with SimpleForceEngine
   - Test with DifferentiableForceEngine
   - Test PyTorch integration

3. **Performance Tests**
   - Benchmark gradient computation time
   - Memory usage profiling
   - Scaling tests

## Migration Path

### For Existing Code
```cpp
// Old API (still works)
auto grads = simulation.computeGradients(...);

// New API (recommended)
auto all_grads = simulation.computeAllGradients(...);
```

### Deprecation Timeline
1. **v1.0**: Release unified version, mark old API as deprecated
2. **v1.1**: Add deprecation warnings
3. **v2.0**: Remove old API, standalone becomes legacy

## Success Criteria

- ✅ All gradient tests passing (<1% error)
- ✅ Parameter gradients working
- ✅ Analytical loss gradients supported
- ✅ PyTorch integration maintained
- ✅ No performance regression
- ✅ Comprehensive documentation
- ✅ Clean, maintainable code

## Timeline

- **Day 1**: Core merge and API design ← WE ARE HERE
- **Day 2**: Testing and validation
- **Day 3**: Documentation and examples
- **Day 4**: Integration with physics engine
- **Day 5**: Polish and release
