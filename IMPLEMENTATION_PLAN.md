# PhysGrad: Comprehensive Implementation Plan
## From 30% to Production-Ready PhD Framework

**Document Version:** 1.0
**Last Updated:** 2025-01-21
**Status:** Active Development Plan

---

## 📋 **TABLE OF CONTENTS**

1. [Strategic Vision](#strategic-vision)
2. [Phase 0: Foundation Repair & Validation](#phase-0-foundation-repair--validation)
3. [Phase 1: Core Novel Contributions](#phase-1-core-novel-contributions)
4. [Phase 2: Differentiable Infrastructure](#phase-2-differentiable-infrastructure)
5. [Phase 3: Advanced Features & Applications](#phase-3-advanced-features--applications)
6. [Phase 4: Polish & Validation](#phase-4-polish--validation)
7. [Success Criteria](#success-criteria)
8. [Timeline Summary](#timeline-summary)
9. [Getting Started](#getting-started)

---

## 🎯 **STRATEGIC VISION**

### Core Thesis Contribution

> "A high-performance differentiable physics framework that bridges hand-optimized CUDA kernels with Python/PyTorch autodiff, featuring state-of-the-art geometric integrators and GPU-accelerated Material Point Method for robotics co-design and manipulation."

### Key Differentiators

1. **Hybrid architecture** - Performance of C++/CUDA + flexibility of Python ML
2. **Geometric integrators** - Long-term energy conservation for robotics
3. **Differentiable MPM** - Multi-material soft/rigid body interactions
4. **End-to-end gradients** - Through entire physics pipeline

### Current Reality Assessment

Based on comprehensive codebase analysis:

- ✅ **Solid foundation** (~20-30% complete) - Basic physics engine works
- ✅ **Excellent planning** - Comprehensive architecture designed
- ⚠️ **Header-only features** (~40-50%) - Class definitions without full implementation
- ❌ **Never built** - No evidence of successful compilation
- ❌ **Unverified claims** - Performance and completeness need validation

---

## 📋 **PHASE 0: FOUNDATION REPAIR & VALIDATION**

**Duration:** 2-4 weeks
**Goal:** Get the project building, tested, and validated
**Priority:** CRITICAL

---

### Week 1: Build System Recovery

#### Task 0.1: Fix Core Build System

```yaml
Priority: CRITICAL
Complexity: Medium
Dependencies: None
Status: Not Started
```

**Objective:** Get `cmake .. && make` to succeed with core modules

**Subtasks:**

1. **Clean CMake configuration**
   - [ ] Remove all disabled modules temporarily
   - [ ] Create minimal working build
   - [ ] Test on clean system
   - [ ] Document successful build process

2. **Resolve CUDA architecture issues**
   - [ ] Verify CUDA toolkit version
   - [ ] Set appropriate compute capabilities for RTX 2000 Ada (sm_89)
   - [ ] Test kernel compilation
   - [ ] Fix any CUDA-specific compilation errors

3. **Fix library dependencies**
   - [ ] Verify Eigen3 installation and version
   - [ ] Check OpenGL/GLFW/GLEW versions
   - [ ] Test PyTorch detection and CMake integration
   - [ ] Resolve any missing dependencies
   - [ ] Document required versions

4. **Enable core modules only**
   ```cmake
   # Core working modules (from CMakeLists.txt):
   - physics_engine.cpp ✓
   - contact_mechanics.cpp ✓
   - fluid_dynamics.cpp ✓
   - visualization.cpp ✓
   - memory_manager.cpp ✓
   - variational_contact.cpp ✓
   - rigid_body.cpp ✓
   - symplectic_integrators.cpp ✓
   ```

**Acceptance Criteria:**
- [ ] `cmake ..` completes without errors
- [ ] `make -j$(nproc)` builds core library successfully
- [ ] No compilation warnings (or documented as acceptable)
- [ ] `libphysgrad_core.a` or `libphysgrad_core.so` is created

**Deliverables:**
- Working build system
- BUILD_INSTRUCTIONS.md with detailed steps
- List of verified dependencies with versions

---

#### Task 0.2: Fix Compilation Errors in Disabled Modules

```yaml
Priority: HIGH
Complexity: High
Dependencies: 0.1
Status: Not Started
```

**Objective:** Re-enable all disabled modules by fixing compilation issues

**Currently Disabled Modules:**

1. **soft_body_dynamics.cpp** (Thrust usage issue)
   - [ ] Review thrust dependencies and usage
   - [ ] Option A: Fix thrust integration with proper CMake
   - [ ] Option B: Replace thrust with custom CUDA implementations
   - [ ] Verify thrust version compatibility with CUDA toolkit
   - [ ] Test compilation

2. **mpi_physics.cpp** (MPI requirement)
   - [ ] Make MPI truly optional with `#ifdef HAVE_MPI` guards
   - [ ] Provide stub implementation when MPI unavailable
   - [ ] Add CMake option `WITH_MPI` (default: OFF)
   - [ ] Test both MPI and non-MPI builds

3. **physics_streaming.cpp** (websocketpp/json)
   - [ ] Make websocketpp optional dependency
   - [ ] Add feature flag `WITH_STREAMING` to CMake
   - [ ] Provide compilation without these dependencies
   - [ ] Document streaming features as optional

4. **neural_fluid_dynamics.cpp** (Compilation issues)
   - [ ] Identify specific compilation errors
   - [ ] Fix include paths and forward declarations
   - [ ] Resolve CUDA/C++ interop issues
   - [ ] Verify template instantiations

5. **symbolic_physics_ai.cpp** (Incomplete types)
   - [ ] Fix forward declarations
   - [ ] Resolve circular dependencies
   - [ ] Ensure proper header inclusion order
   - [ ] Fix incomplete type errors

6. **physics_generative_models.cpp** (CUDA kernel in .cpp)
   - [ ] Move CUDA kernels to .cu file
   - [ ] Keep C++ wrapper in .cpp
   - [ ] Fix linkage between .cpp and .cu
   - [ ] Test kernel launch

7. **quantum_classical_coupling.cpp** (Incomplete types)
   - [ ] Similar to symbolic_physics_ai
   - [ ] Fix template instantiation issues
   - [ ] Resolve header dependencies
   - [ ] Verify quantum_classical.h completeness

**Acceptance Criteria:**
- [ ] All 7 modules compile successfully
- [ ] Optional features controlled by CMake flags
- [ ] No remaining "Temporarily disabled" comments in CMakeLists.txt
- [ ] Clean build with all features enabled

**Deliverables:**
- All modules building
- Updated CMakeLists.txt with feature flags
- FEATURES.md documenting optional components

---

### Week 2: Test Infrastructure

#### Task 0.3: Establish Baseline Testing

```yaml
Priority: CRITICAL
Complexity: Medium
Dependencies: 0.1
Status: Not Started
```

**Objective:** Run existing tests and establish baseline pass rate

**Subtasks:**

1. **Run existing tests**
   ```bash
   cd build
   ctest --output-on-failure
   ```
   - [ ] Document all test failures
   - [ ] Fix critical test failures
   - [ ] Record baseline pass rate
   - [ ] Categorize failures (build, runtime, assertion)

2. **Create test execution script**
   ```bash
   # tests/run_all_tests.sh
   #!/bin/bash

   echo "==================================="
   echo "PhysGrad Test Suite"
   echo "==================================="

   FAILURES=0

   # Core physics tests
   echo "Running core physics tests..."
   ./test_physics_engine || ((FAILURES++))
   ./test_contact_mechanics || ((FAILURES++))
   ./test_fluid_dynamics || ((FAILURES++))
   ./test_memory_manager || ((FAILURES++))

   # CUDA tests
   if [ "$WITH_CUDA" = "ON" ]; then
       echo "Running CUDA tests..."
       ./test_cuda_kernels || ((FAILURES++))
   fi

   # Advanced features
   echo "Running advanced feature tests..."
   ./test_adjoint_integrators || ((FAILURES++))
   ./test_mpm || ((FAILURES++))
   ./test_pytorch_autograd || ((FAILURES++))

   echo "==================================="
   echo "Test Summary: $FAILURES failures"
   echo "==================================="

   exit $FAILURES
   ```
   - [ ] Make script executable
   - [ ] Integrate with CMake/CTest
   - [ ] Add to CI/CD

3. **Add missing test categories**
   - [ ] Energy conservation validation tests
   - [ ] Momentum conservation validation tests
   - [ ] Numerical stability tests
   - [ ] Performance regression tests
   - [ ] Memory leak detection tests

4. **Create CI/CD skeleton**
   ```yaml
   # .github/workflows/tests.yml
   name: PhysGrad Tests

   on:
     push:
       branches: [ main, develop ]
     pull_request:
       branches: [ main ]

   jobs:
     test:
       runs-on: ubuntu-latest

       steps:
         - uses: actions/checkout@v3

         - name: Install dependencies
           run: |
             sudo apt-get update
             sudo apt-get install -y libeigen3-dev libgl1-mesa-dev \
               libglfw3-dev libglew-dev

         - name: Configure CMake
           run: cmake -B build -DCMAKE_BUILD_TYPE=Release

         - name: Build
           run: cmake --build build -j$(nproc)

         - name: Run tests
           run: cd build && ctest --output-on-failure
   ```
   - [ ] Create .github/workflows directory
   - [ ] Add tests.yml
   - [ ] Test on GitHub Actions

**Acceptance Criteria:**
- [ ] All existing tests run (pass or fail documented)
- [ ] Test execution script works
- [ ] CI/CD pipeline established
- [ ] Baseline metrics recorded

**Deliverables:**
- Test execution infrastructure
- Baseline test report
- CI/CD pipeline
- TEST_RESULTS.md with current state

---

#### Task 0.4: Minimal Validation Suite

```yaml
Priority: HIGH
Complexity: Medium
Dependencies: 0.3
Status: Not Started
```

**Objective:** Create physics validation benchmarks proving correctness

**Subtasks:**

1. **Energy Conservation Test**
   ```cpp
   // tests/validation/test_energy_conservation.cpp

   #include <gtest/gtest.h>
   #include "physics_engine.h"

   TEST(ValidationSuite, TwoBodyEnergyConservation) {
       using namespace physgrad;

       PhysicsEngine engine;
       engine.initialize();

       // Two-body gravitational system
       std::vector<float3> positions = {
           {0.0f, 0.0f, 0.0f},
           {1.0f, 0.0f, 0.0f}
       };
       std::vector<float3> velocities = {
           {0.0f, 0.5f, 0.0f},
           {0.0f, -0.5f, 0.0f}
       };
       std::vector<float> masses = {1.0f, 1.0f};

       engine.addParticles(positions, velocities, masses);

       float initial_energy = engine.calculateTotalEnergy();

       // Run for 10,000 timesteps
       for (int i = 0; i < 10000; ++i) {
           engine.step(0.001f);
       }

       float final_energy = engine.calculateTotalEnergy();

       // Verify energy conservation
       float relative_error = std::abs(final_energy - initial_energy) / initial_energy;
       EXPECT_LT(relative_error, 1e-6) << "Energy drift too large";
   }
   ```
   - [ ] Implement test
   - [ ] Run and verify
   - [ ] Document expected tolerances

2. **Momentum Conservation Test**
   ```cpp
   TEST(ValidationSuite, CollisionMomentumConservation) {
       // Elastic collision test
       // Initial: two particles approaching
       // Final: two particles separating

       engine.addParticles(positions, velocities, masses);

       float3 initial_momentum = calculateTotalMomentum();

       // Simulate collision
       for (int i = 0; i < 1000; ++i) {
           engine.step(0.001f);
       }

       float3 final_momentum = calculateTotalMomentum();

       // Verify total momentum unchanged
       EXPECT_NEAR(initial_momentum.x, final_momentum.x, 1e-6);
       EXPECT_NEAR(initial_momentum.y, final_momentum.y, 1e-6);
       EXPECT_NEAR(initial_momentum.z, final_momentum.z, 1e-6);

       // For elastic collision, verify kinetic energy conserved
       float initial_ke = calculateKineticEnergy();
       float final_ke = calculateKineticEnergy();
       EXPECT_NEAR(initial_ke, final_ke, 1e-6);
   }
   ```
   - [ ] Implement test
   - [ ] Test with different collision scenarios
   - [ ] Verify both momentum and energy

3. **Contact Stacking Test**
   ```cpp
   TEST(ValidationSuite, StackStability) {
       // Stack 10 boxes
       const int num_boxes = 10;
       const float box_size = 0.1f;

       for (int i = 0; i < num_boxes; ++i) {
           float3 position = {0.0f, i * box_size, 0.0f};
           float3 velocity = {0.0f, 0.0f, 0.0f};
           addBox(position, velocity, box_size);
       }

       float initial_total_energy = calculateTotalEnergy();

       // Run for 1000 steps
       for (int i = 0; i < 1000; ++i) {
           engine.step(0.01f);
       }

       float final_total_energy = calculateTotalEnergy();

       // Verify no energy gain (energy can only decrease due to damping)
       EXPECT_LE(final_total_energy, initial_total_energy);

       // Verify stack is stable (no box fell off)
       EXPECT_TRUE(isStackStable());
   }
   ```
   - [ ] Implement stacking test
   - [ ] Verify stability criteria
   - [ ] Test with different stack heights

4. **Fluid Dam Break Test**
   ```cpp
   TEST(ValidationSuite, SPHDamBreak) {
       // Standard dam break problem
       // Compare to analytical/experimental results

       SPHSolver fluid_solver;

       // Initialize water column
       // Left side: 0 <= x <= 0.5, 0 <= y <= 1.0
       // Right side: dry bed
       initializeDamBreak(fluid_solver);

       // Simulate
       std::vector<float> wave_front_positions;
       for (int i = 0; i < 100; ++i) {
           fluid_solver.step(0.01f);
           wave_front_positions.push_back(
               computeWaveFrontPosition(fluid_solver)
           );
       }

       // Compare to experimental data
       // (Martin and Moyce 1952, or Koshizuka 1995)
       for (size_t i = 0; i < wave_front_positions.size(); ++i) {
           float expected = experimentalWaveFront(i * 0.01f);
           float actual = wave_front_positions[i];
           EXPECT_NEAR(actual, expected, 0.05); // 5% tolerance
       }
   }
   ```
   - [ ] Implement dam break test
   - [ ] Find reference experimental data
   - [ ] Validate wave front tracking

**Acceptance Criteria:**
- [ ] All 4 validation tests pass
- [ ] Physics correctness proven
- [ ] Reference data documented
- [ ] Tolerances justified

**Deliverables:**
- Validation test suite
- VALIDATION_RESULTS.md with benchmark data
- Reference data files

---

### Week 3-4: Code Quality & Documentation

#### Task 0.5: Code Cleanup

```yaml
Priority: MEDIUM
Complexity: Low
Dependencies: 0.1-0.4
Status: Not Started
```

**Objective:** Clean codebase with zero warnings, no memory leaks

**Subtasks:**

1. **Fix all compiler warnings**
   - [ ] Unused parameters: add `[[maybe_unused]]` or use
   - [ ] Narrowing conversions: use explicit casts
   - [ ] Member initialization order: fix constructor order
   - [ ] Sign comparison warnings: use consistent types
   - [ ] Compile with `-Werror` to enforce

2. **Add error handling**
   ```cpp
   // Replace asserts with proper error returns
   // Before:
   assert(num_particles > 0);

   // After:
   if (num_particles <= 0) {
       LOG_ERROR("Invalid particle count: " << num_particles);
       return false;
   }
   ```
   - [ ] Replace asserts with error returns
   - [ ] Add CUDA error checking macros
   ```cpp
   #define CUDA_CHECK(call) \
       do { \
           cudaError_t err = call; \
           if (err != cudaSuccess) { \
               LOG_ERROR("CUDA error: " << cudaGetErrorString(err)); \
               return false; \
           } \
       } while(0)
   ```
   - [ ] Implement error logging system

3. **Memory leak detection**
   ```bash
   # Run valgrind on all tests
   valgrind --leak-check=full --show-leak-kinds=all \
            ./test_physics_engine

   # For CUDA memory leaks
   cuda-memcheck ./test_cuda_kernels
   ```
   - [ ] Run valgrind on all CPU tests
   - [ ] Fix all detected leaks
   - [ ] Run cuda-memcheck on GPU tests
   - [ ] Fix CUDA memory leaks
   - [ ] Add memory leak tests to CI

4. **Code formatting**
   ```bash
   # Create .clang-format
   cat > .clang-format << EOF
   BasedOnStyle: Google
   IndentWidth: 4
   ColumnLimit: 100
   EOF

   # Apply formatting
   find src -name "*.cpp" -o -name "*.h" | xargs clang-format -i
   find src -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
   ```
   - [ ] Create .clang-format configuration
   - [ ] Format all source files
   - [ ] Add pre-commit hook for formatting
   - [ ] Document code style

**Acceptance Criteria:**
- [ ] Zero compiler warnings with `-Wall -Wextra -Werror`
- [ ] No memory leaks detected by valgrind
- [ ] No CUDA memory errors
- [ ] Consistent code formatting

**Deliverables:**
- Clean, warning-free code
- Memory leak report (clean)
- Code style guide
- Pre-commit hooks

---

#### Task 0.6: Honest Documentation Audit

```yaml
Priority: HIGH
Complexity: Low
Dependencies: 0.1-0.5
Status: Not Started
```

**Objective:** Update documentation to reflect actual implementation state

**Subtasks:**

1. **Update TECHNICAL_STATUS.md**
   - [ ] Remove "100% complete" claims
   - [ ] Mark header-only features clearly as "Design Only"
   - [ ] Add "Implementation Status" column to all feature tables
   - [ ] Be honest about what works vs what's planned
   - [ ] Update test coverage numbers with actual data

2. **Create IMPLEMENTATION_STATUS.md**
   ```markdown
   # PhysGrad Implementation Status

   ## ✅ Production Ready (Tested & Validated)

   - [x] Basic particle physics engine
   - [x] Contact mechanics (impulse-based)
   - [x] SPH fluid dynamics
   - [x] Memory management system
   - [x] OpenGL/ImGui visualization
   - [x] Python bindings (basic)

   ## 🚧 In Development (Partial Implementation)

   - [ ] MPM solver (headers complete, implementation in progress)
   - [ ] Differentiable physics (design complete, needs kernel adjoints)
   - [ ] Symplectic integrators (framework ready, needs testing)
   - [ ] Variational integrators (theory documented, implementation pending)

   ## 📋 Planned (Design Only)

   - [ ] Robot co-design framework (headers exist, no implementation)
   - [ ] Quantum-classical hybrid (design phase)
   - [ ] Digital twin framework (concept only)
   - [ ] Neural surrogate models (skeleton code only)

   ## ❌ Not Started

   - [ ] Benchmarking against competitors
   - [ ] Multi-GPU scaling validation
   - [ ] Real-world robot experiments
   ```
   - [ ] Create file with honest assessment
   - [ ] Link from main README
   - [ ] Update regularly

3. **Update README.md**
   ```markdown
   # PhysGrad

   **Status:** Early Development (Alpha)

   A high-performance physics simulation framework with GPU acceleration
   and differentiable computing capabilities (in development).

   ## ⚠️ Current Status

   PhysGrad is under active development. The following components are
   production-ready:

   - Basic particle physics ✓
   - Contact mechanics ✓
   - SPH fluid dynamics ✓

   Advanced features (MPM, differentiable physics, robot co-design) are
   in development. See IMPLEMENTATION_STATUS.md for details.

   ## Features (Current vs Planned)

   ### Working Now
   - ✓ Multi-physics simulation
   - ✓ GPU acceleration (CUDA)
   - ✓ Python integration

   ### In Development
   - ⚠️ Material Point Method
   - ⚠️ Differentiable physics
   - ⚠️ Advanced integrators

   ### Planned
   - ○ Robot co-design
   - ○ Quantum-classical coupling

   ## Performance

   Current benchmarks (to be validated):
   - Particle systems: TBD
   - Contact mechanics: TBD
   - Fluid dynamics: TBD

   *Note: Performance claims from previous docs are being re-validated*
   ```
   - [ ] Update README with realistic claims
   - [ ] Add clear status badges
   - [ ] Remove unverified performance numbers
   - [ ] Add "Contributing" section

4. **Update TECHNICAL_DOCUMENTATION.md**
   - [ ] Mark each section with implementation status
   - [ ] Add "Status: Implemented/Partial/Planned" to each component
   - [ ] Remove claims of "100% test coverage"
   - [ ] Update architecture diagrams to match reality

**Acceptance Criteria:**
- [ ] All documentation reflects actual state
- [ ] No misleading claims about completeness
- [ ] Clear distinction between working and planned features
- [ ] Users can understand what actually works

**Deliverables:**
- Updated README.md
- New IMPLEMENTATION_STATUS.md
- Updated TECHNICAL_STATUS.md
- Updated TECHNICAL_DOCUMENTATION.md

---

## 📋 **PHASE 1: CORE NOVEL CONTRIBUTIONS**

**Duration:** 8-12 weeks
**Goal:** Implement the foundation for PhD novelty
**Priority:** CRITICAL

---

### Week 5-7: Symplectic Integrators

#### Task 1.1: Implement FROST-style Integrator

```yaml
Priority: CRITICAL (Core PhD contribution)
Complexity: High
Dependencies: Phase 0 complete
Status: Not Started
```

**Reference:** "frost: a momentum-conserving CUDA implementation of a hierarchical fourth-order forward symplectic integrator" (MNRAS, 2021)

**Objective:** Implement 4th-order forward symplectic integrator with GPU acceleration

**Subtasks:**

1. **Study and implement FSI base algorithm**
   ```cpp
   // src/symplectic_integrators.h

   namespace physgrad {

   class ForwardSymplecticIntegrator {
   public:
       ForwardSymplecticIntegrator();
       ~ForwardSymplecticIntegrator();

       // Fourth-order forward integration
       // Requires force gradients (Hessian of potential)
       void step(
           std::vector<float3>& positions,
           std::vector<float3>& velocities,
           const std::vector<float>& masses,
           float dt
       );

       // Configure integrator
       void setOrder(int order);  // 2, 4, 6 (if extending)

   private:
       // Compute forces and force gradients
       void computeForcesAndGradients(
           const std::vector<float3>& positions,
           std::vector<float3>& forces,
           std::vector<Matrix3x3>& force_gradients
       );

       // FSI coefficients for 4th order
       static constexpr float c1 = 0.1786178958448091;
       static constexpr float c2 = -0.2123418310626054;
       static constexpr float c3 = -0.06626458266981849;
       static constexpr float c4 = 0.06626458266981849;

       int order_;

       // GPU resources
       float3* d_positions_;
       float3* d_velocities_;
       Matrix3x3* d_force_gradients_;
   };

   } // namespace physgrad
   ```
   - [ ] Implement basic structure
   - [ ] Add FSI coefficients from paper
   - [ ] Implement step function
   - [ ] Test on simple 2-body problem

2. **Implement force gradient computation**
   ```cpp
   // src/force_gradients.h

   namespace physgrad {

   // 3x3 matrix for force gradients
   struct Matrix3x3 {
       float data[9];

       float& operator()(int i, int j) {
           return data[i * 3 + j];
       }

       const float& operator()(int i, int j) const {
           return data[i * 3 + j];
       }
   };

   // Analytical gradient of gravitational force
   // F_i = -G * m_i * m_j * r_ij / |r_ij|^3
   // ∂F_i/∂r_j = Hessian of potential
   Matrix3x3 gravitationalForceGradient(
       const float3& pos_i,
       const float3& pos_j,
       float mass_i,
       float mass_j,
       float G = 1.0f
   );

   // Analytical gradient of electrostatic force
   Matrix3x3 electrostaticForceGradient(
       const float3& pos_i,
       const float3& pos_j,
       float charge_i,
       float charge_j,
       float k_e = 8.9875517923e9f
   );

   } // namespace physgrad
   ```
   ```cpp
   // src/force_gradients.cpp

   Matrix3x3 gravitationalForceGradient(
       const float3& pos_i,
       const float3& pos_j,
       float mass_i,
       float mass_j,
       float G
   ) {
       float3 r = pos_j - pos_i;
       float r_mag = magnitude(r);
       float r3 = r_mag * r_mag * r_mag;
       float r5 = r3 * r_mag * r_mag;

       float coeff1 = -G * mass_i * mass_j / r3;
       float coeff2 = 3.0f * G * mass_i * mass_j / r5;

       Matrix3x3 result;
       for (int i = 0; i < 3; ++i) {
           for (int j = 0; j < 3; ++j) {
               if (i == j) {
                   result(i, j) = coeff1 + coeff2 * r[i] * r[j];
               } else {
                   result(i, j) = coeff2 * r[i] * r[j];
               }
           }
       }

       return result;
   }
   ```
   - [ ] Implement gravitational gradient
   - [ ] Implement electrostatic gradient
   - [ ] Verify with finite differences
   - [ ] Test numerical stability

3. **GPU implementation**
   ```cuda
   // src/symplectic_kernels.cu

   namespace physgrad {

   __global__ void compute_force_gradients_kernel(
       const float3* positions,
       const float* masses,
       const float* charges,
       float3* forces,
       Matrix3x3* force_gradients,
       int num_particles
   ) {
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i >= num_particles) return;

       float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
       Matrix3x3 total_gradient;
       // Initialize gradient to zero
       for (int k = 0; k < 9; ++k) {
           total_gradient.data[k] = 0.0f;
       }

       for (int j = 0; j < num_particles; ++j) {
           if (i == j) continue;

           float3 r_ij = positions[j] - positions[i];
           float r = magnitude(r_ij);

           if (r > 1e-6f) {
               // Compute force
               float3 force = computePairwiseForce(
                   positions[i], positions[j],
                   masses[i], masses[j],
                   charges[i], charges[j],
                   r
               );
               total_force += force;

               // Compute force gradient
               Matrix3x3 grad = computePairwiseGradient(
                   positions[i], positions[j],
                   masses[i], masses[j],
                   charges[i], charges[j],
                   r
               );
               addMatrix(total_gradient, grad);
           }
       }

       forces[i] = total_force;
       force_gradients[i] = total_gradient;
   }

   __global__ void fsi_step_kernel(
       float3* positions,
       float3* velocities,
       const float3* forces,
       const Matrix3x3* force_gradients,
       const float* masses,
       float dt,
       float c,  // FSI coefficient
       int num_particles
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx >= num_particles) return;

       float inv_mass = 1.0f / masses[idx];

       // Fourth-order FSI update
       // See FROST paper for derivation

       // Position update
       float3 vel_contrib = c * dt * velocities[idx];
       float3 accel_contrib = 0.5f * c * c * dt * dt * forces[idx] * inv_mass;

       positions[idx] += vel_contrib + accel_contrib;

       // Velocity update (requires force gradient)
       float3 vel_update = c * dt * forces[idx] * inv_mass;
       // Second-order correction using gradient
       // v += c*dt*F/m + c²*dt²/2 * (dF/dx * v) / m

       velocities[idx] += vel_update;
   }

   } // namespace physgrad
   ```
   - [ ] Implement force gradient kernel
   - [ ] Implement FSI step kernel
   - [ ] Optimize memory access patterns
   - [ ] Test GPU vs CPU implementation

4. **Hierarchical timestep extension**
   ```cpp
   // src/symplectic_integrators.h (extension)

   class HierarchicalFSI : public ForwardSymplecticIntegrator {
   public:
       // Different timesteps for different particle groups
       // Useful for multi-scale systems

       void setTimestepGroup(int particle_id, int group_id);
       void setGroupTimestep(int group_id, float dt);

       void hierarchicalStep(float dt_base);

   private:
       std::vector<int> particle_groups_;
       std::map<int, float> group_timesteps_;

       // Momentum-conserving hierarchical integration
       void integrateGroup(int group_id, float dt);
   };
   ```
   - [ ] Implement group management
   - [ ] Implement hierarchical stepping
   - [ ] Verify momentum conservation across groups
   - [ ] Test multi-scale system (planet + moon)

5. **Validation tests**
   ```cpp
   // tests/test_symplectic_integrators.cpp

   #include <gtest/gtest.h>
   #include "symplectic_integrators.h"

   TEST(SymplecticIntegrator, EnergyConservation) {
       // Two-body gravitational problem
       ForwardSymplecticIntegrator integrator;

       // Circular orbit initial conditions
       float3 pos1 = {1.0f, 0.0f, 0.0f};
       float3 pos2 = {-1.0f, 0.0f, 0.0f};
       float3 vel1 = {0.0f, 0.5f, 0.0f};
       float3 vel2 = {0.0f, -0.5f, 0.0f};

       float initial_energy = computeTotalEnergy(pos1, pos2, vel1, vel2);

       // Integrate for 100,000 steps
       for (int i = 0; i < 100000; ++i) {
           integrator.step(positions, velocities, masses, 0.01f);
       }

       float final_energy = computeTotalEnergy(...);

       // Energy drift should be at machine precision level
       float relative_error = std::abs(final_energy - initial_energy) / initial_energy;
       EXPECT_LT(relative_error, 1e-10)
           << "Symplectic integrator should have near-zero energy drift";
   }

   TEST(SymplecticIntegrator, MomentumConservation) {
       // Verify exact momentum conservation
       float3 initial_momentum = computeTotalMomentum(...);

       for (int i = 0; i < 10000; ++i) {
           integrator.step(...);
       }

       float3 final_momentum = computeTotalMomentum(...);

       // Should be exact to machine precision
       EXPECT_NEAR(initial_momentum.x, final_momentum.x, 1e-12);
       EXPECT_NEAR(initial_momentum.y, final_momentum.y, 1e-12);
       EXPECT_NEAR(initial_momentum.z, final_momentum.z, 1e-12);
   }

   TEST(SymplecticIntegrator, FourthOrderAccuracy) {
       // Verify O(dt^4) convergence
       // Compare solutions at dt, dt/2, dt/4

       auto solution_dt = runIntegrator(dt);
       auto solution_dt2 = runIntegrator(dt / 2.0f);
       auto solution_dt4 = runIntegrator(dt / 4.0f);

       // Richardson extrapolation to verify order
       float error_dt = computeError(solution_dt, exact_solution);
       float error_dt2 = computeError(solution_dt2, exact_solution);
       float error_dt4 = computeError(solution_dt4, exact_solution);

       // For 4th order: error ~ dt^4
       // error(dt/2) / error(dt) ≈ 1/16
       float ratio1 = error_dt / error_dt2;
       float ratio2 = error_dt2 / error_dt4;

       EXPECT_NEAR(ratio1, 16.0f, 1.0f);
       EXPECT_NEAR(ratio2, 16.0f, 1.0f);
   }

   TEST(HierarchicalFSI, MomentumConservingMultiScale) {
       // Planet + satellite system
       // Planet: large mass, slow timestep
       // Satellite: small mass, fast timestep

       HierarchicalFSI h_integrator;
       h_integrator.setTimestepGroup(0, 0);  // planet -> group 0
       h_integrator.setTimestepGroup(1, 1);  // satellite -> group 1
       h_integrator.setGroupTimestep(0, 0.1f);   // slow
       h_integrator.setGroupTimestep(1, 0.01f);  // fast

       float3 initial_momentum = computeTotalMomentum(...);

       h_integrator.hierarchicalStep(0.1f);

       float3 final_momentum = computeTotalMomentum(...);

       // Must conserve total momentum exactly
       EXPECT_NEAR(initial_momentum.x, final_momentum.x, 1e-12);
   }
   ```
   - [ ] Implement all tests
   - [ ] Verify energy conservation
   - [ ] Verify momentum conservation
   - [ ] Verify 4th-order accuracy
   - [ ] Test hierarchical integration

**Acceptance Criteria:**
- [ ] FSI integrator implemented and working
- [ ] GPU acceleration functional
- [ ] Energy drift < 1e-10 over 100K steps
- [ ] Exact momentum conservation
- [ ] 4th-order accuracy verified
- [ ] All tests passing

**Deliverables:**
- Working 4th-order symplectic integrator
- GPU-accelerated implementation
- Force gradient computation
- Hierarchical timestep extension
- Comprehensive test suite
- Performance benchmarks

---

#### Task 1.2: Implement Variational Integrators

```yaml
Priority: HIGH (PhD novelty)
Complexity: High
Dependencies: 1.1
Status: Not Started
```

**Reference:** "Variational Integrators" (Marsden & West, 2001)

**Objective:** Implement variational integrators with automatic structure preservation

**Subtasks:**

1. **Discrete Lagrangian framework**
   ```cpp
   // src/variational_integrators.h

   namespace physgrad {

   // Abstract Lagrangian interface
   template<typename StateType>
   class Lagrangian {
   public:
       virtual ~Lagrangian() = default;

       // Evaluate Lagrangian: L(q, q_dot, t)
       virtual float evaluate(
           const StateType& q,
           const StateType& q_dot,
           float t
       ) const = 0;

       // Partial derivatives for discrete Euler-Lagrange
       virtual StateType dL_dq(const StateType& q, const StateType& q_dot, float t) const = 0;
       virtual StateType dL_dqdot(const StateType& q, const StateType& q_dot, float t) const = 0;
   };

   template<typename LagrangianType>
   class VariationalIntegrator {
   public:
       VariationalIntegrator(const LagrangianType& lagrangian)
           : lagrangian_(lagrangian) {}

       // Integrate one step using discrete Euler-Lagrange equations
       void step(
           std::vector<float3>& positions,
           std::vector<float3>& velocities,
           float dt
       );

   private:
       const LagrangianType& lagrangian_;

       // Solve discrete Euler-Lagrange equations
       // DEL: D1 L_d(q_k, q_{k+1}) + D2 L_d(q_{k-1}, q_k) = 0
       void solveDiscreteEulerLagrange(
           const std::vector<float3>& q_k_minus_1,
           const std::vector<float3>& q_k,
           std::vector<float3>& q_k_plus_1,
           float dt
       );
   };

   } // namespace physgrad
   ```
   - [ ] Implement Lagrangian base class
   - [ ] Implement variational integrator framework
   - [ ] Add Newton solver for implicit equations

2. **Implement specific Lagrangians**
   ```cpp
   // src/mechanical_lagrangian.h

   namespace physgrad {

   // Mechanical Lagrangian: L = T - V (kinetic - potential)
   class MechanicalLagrangian : public Lagrangian<std::vector<float3>> {
   public:
       MechanicalLagrangian(
           const std::vector<float>& masses,
           std::function<float(const std::vector<float3>&)> potential_fn
       ) : masses_(masses), potential_fn_(potential_fn) {}

       float evaluate(
           const std::vector<float3>& q,
           const std::vector<float3>& q_dot,
           float t
       ) const override {
           float T = kineticEnergy(q_dot);
           float V = potential_fn_(q);
           return T - V;
       }

       std::vector<float3> dL_dq(
           const std::vector<float3>& q,
           const std::vector<float3>& q_dot,
           float t
       ) const override {
           // -dV/dq
           return -gradientPotential(q);
       }

       std::vector<float3> dL_dqdot(
           const std::vector<float3>& q,
           const std::vector<float3>& q_dot,
           float t
       ) const override {
           // m * q_dot
           std::vector<float3> result(q_dot.size());
           for (size_t i = 0; i < q_dot.size(); ++i) {
               result[i] = masses_[i] * q_dot[i];
           }
           return result;
       }

   private:
       std::vector<float> masses_;
       std::function<float(const std::vector<float3>&)> potential_fn_;

       float kineticEnergy(const std::vector<float3>& q_dot) const {
           float T = 0.0f;
           for (size_t i = 0; i < q_dot.size(); ++i) {
               float v2 = dot(q_dot[i], q_dot[i]);
               T += 0.5f * masses_[i] * v2;
           }
           return T;
       }

       std::vector<float3> gradientPotential(const std::vector<float3>& q) const {
           // Finite difference or analytical gradient
           // Depends on potential_fn_
           return finiteDifferenceGradient(potential_fn_, q);
       }
   };

   } // namespace physgrad
   ```
   - [ ] Implement mechanical Lagrangian
   - [ ] Add common potential functions (gravity, springs, etc.)
   - [ ] Test on simple pendulum

3. **Galerkin variational integrator**
   ```cpp
   // src/galerkin_integrator.h

   namespace physgrad {

   // High-order variational integrator via Galerkin projection
   class GalerkinVariationalIntegrator {
   public:
       // order: 2, 4, 6, 8 (polynomial degree)
       GalerkinVariationalIntegrator(int order);

       void step(
           std::vector<float3>& positions,
           std::vector<float3>& velocities,
           const std::vector<float>& masses,
           std::function<float(const std::vector<float3>&)> potential,
           float dt
       );

   private:
       int order_;

       // Gauss-Legendre quadrature points and weights
       std::vector<float> quadrature_points_;
       std::vector<float> quadrature_weights_;

       // Lagrange basis polynomials
       float lagrangeBasis(int i, float t) const;
       float lagrangeBasisDerivative(int i, float t) const;

       // Solve Galerkin discrete Euler-Lagrange equations
       void solveGalerkinDEL(...);
   };

   } // namespace physgrad
   ```
   - [ ] Implement Gauss-Legendre quadrature
   - [ ] Implement Lagrange basis functions
   - [ ] Implement Galerkin projection
   - [ ] Test convergence order

4. **Comprehensive testing**
   ```cpp
   // tests/test_variational_integrators.cpp

   TEST(VariationalIntegrator, SymplecticProperty) {
       // Verify that symplectic form is preserved
       // ω = dp ∧ dq should be preserved

       VariationalIntegrator integrator(mechanical_lagrangian);

       // Measure symplectic form before/after
       float omega_before = computeSymplecticForm(q0, p0);

       integrator.step(positions, velocities, dt);

       float omega_after = computeSymplecticForm(q1, p1);

       EXPECT_NEAR(omega_before, omega_after, 1e-12);
   }

   TEST(VariationalIntegrator, DiscreteNoether) {
       // Verify discrete Noether's theorem
       // Symmetry -> Conservation law

       // Time-translation symmetry -> Energy conservation
       // Space-translation symmetry -> Momentum conservation

       auto lagrangian = MechanicalLagrangian(masses, potential);
       VariationalIntegrator integrator(lagrangian);

       float E0 = computeEnergy(q0, v0);
       float3 p0 = computeMomentum(v0, masses);

       for (int i = 0; i < 1000; ++i) {
           integrator.step(positions, velocities, dt);
       }

       float E1 = computeEnergy(q1, v1);
       float3 p1 = computeMomentum(v1, masses);

       // Discrete Noether -> exact conservation
       EXPECT_NEAR(E0, E1, 1e-10);
       EXPECT_NEAR(p0.x, p1.x, 1e-12);
       EXPECT_NEAR(p0.y, p1.y, 1e-12);
       EXPECT_NEAR(p0.z, p1.z, 1e-12);
   }

   TEST(GalerkinIntegrator, HighOrderAccuracy) {
       // Test 2nd, 4th, 6th order methods

       for (int order : {2, 4, 6}) {
           GalerkinVariationalIntegrator integrator(order);

           auto solution = integratePendulum(integrator, dt);
           auto exact = exactPendulumSolution(t_final);

           float error = computeError(solution, exact);

           // Verify error ~ dt^order
           errors[order] = error;
       }

       // Check convergence rates
       EXPECT_NEAR(errors[4] / errors[2], pow(dt_ratio, 2), 0.1);
       EXPECT_NEAR(errors[6] / errors[4], pow(dt_ratio, 2), 0.1);
   }
   ```
   - [ ] Test symplectic property
   - [ ] Test discrete Noether theorem
   - [ ] Test high-order convergence
   - [ ] Compare to standard methods

**Acceptance Criteria:**
- [ ] Variational integrators implemented
- [ ] Symplectic property verified
- [ ] Conservation laws exact (discrete Noether)
- [ ] High-order accuracy achieved
- [ ] All tests passing

**Deliverables:**
- Variational integrator framework
- Mechanical Lagrangian implementation
- Galerkin high-order integrators
- Comprehensive tests
- Theory documentation

---

### Week 8-12: Material Point Method (MPM)

#### Task 1.3: Complete MPM Solver Implementation

```yaml
Priority: CRITICAL (Core PhD contribution)
Complexity: Very High
Dependencies: Phase 0 complete
Status: Not Started
```

**Reference:** "A Massively Parallel and Scalable Multi-GPU Material Point Method" (SIGGRAPH 2020)

**Objective:** Full MPM solver with AoSoA data structures and GPU acceleration

**Subtasks:**

1. **Implement AoSoA data structures** (headers exist at src/mpm_data_structures.h)
   ```cpp
   // src/mpm_data_structures.cpp

   namespace physgrad {
   namespace mpm {

   template<typename T, size_t ChunkSize>
   class ParticleAoSoA {
       // Headers already exist - implement all methods

       void setPosition(size_t particle_id, const ConceptVector3D<T>& pos) {
           const size_t chunk_id = particle_id / chunk_size;
           const size_t local_id = particle_id % chunk_size;

           position_chunks[chunk_id][local_id] = pos.x;
           position_chunks[chunk_id][local_id + chunk_size] = pos.y;
           position_chunks[chunk_id][local_id + 2 * chunk_size] = pos.z;
       }

       ConceptVector3D<T> getPosition(size_t particle_id) const {
           // Already in header, verify implementation
           const size_t chunk_id = particle_id / chunk_size;
           const size_t local_id = particle_id % chunk_size;
           const auto& chunk = position_chunks[chunk_id];

           return ConceptVector3D<T>{
               chunk[local_id],
               chunk[local_id + chunk_size],
               chunk[local_id + 2 * chunk_size]
           };
       }

       // Implement all other accessors
   };

   } // namespace mpm
   } // namespace physgrad
   ```
   - [ ] Implement all AoSoA accessor methods
   - [ ] Test memory layout
   - [ ] Benchmark cache performance
   - [ ] Verify coalesced GPU access

2. **Implement core MPM algorithm**
   ```cpp
   // src/mpm_solver.cpp

   namespace physgrad {
   namespace mpm {

   class MPMSolver {
   public:
       MPMSolver() = default;

       void initialize(const MPMConfig& config) {
           grid_resolution_ = config.grid_resolution;
           cell_size_ = config.cell_size;

           // Allocate grid
           int total_cells = grid_resolution_[0] * grid_resolution_[1] * grid_resolution_[2];
           grid_nodes_.resize(total_cells);

           // Allocate GPU memory if CUDA available
           #ifdef HAVE_CUDA
           allocateGPUMemory();
           #endif
       }

       void addParticles(const std::vector<Particle>& particles) {
           // Convert to AoSoA layout
           particle_data_.resize(particles.size());

           for (size_t i = 0; i < particles.size(); ++i) {
               particle_data_.setPosition(i, particles[i].position);
               particle_data_.setVelocity(i, particles[i].velocity);
               particle_data_.setMass(i, particles[i].mass);
               // ... other properties
           }
       }

       void step(float dt) {
           // MPM algorithm:
           // 1. Particle to Grid (P2G)
           // 2. Grid operations
           // 3. Grid to Particle (G2P)

           #ifdef HAVE_CUDA
           stepGPU(dt);
           #else
           stepCPU(dt);
           #endif
       }

   private:
       void particleToGrid() {
           // Clear grid
           clearGrid();

           // Transfer mass and momentum from particles to grid
           for (size_t p = 0; p < particle_data_.size(); ++p) {
               auto pos = particle_data_.getPosition(p);
               auto vel = particle_data_.getVelocity(p);
               float mass = particle_data_.getMass(p);

               // Find grid cell
               int3 cell_idx = positionToCell(pos);

               // 3x3x3 neighborhood for quadratic B-spline
               for (int i = -1; i <= 1; ++i) {
                   for (int j = -1; j <= 1; ++j) {
                       for (int k = -1; k <= 1; ++k) {
                           int3 node_idx = cell_idx + int3{i, j, k};
                           if (!isValidCell(node_idx)) continue;

                           // Compute weight
                           float weight = bSplineWeight(pos, nodeCellCenter(node_idx));

                           // Transfer mass and momentum
                           int node_linear = cellToLinearIndex(node_idx);
                           grid_nodes_[node_linear].mass += weight * mass;
                           grid_nodes_[node_linear].momentum += weight * mass * vel;
                       }
                   }
               }
           }
       }

       void updateGrid(float dt) {
           // Grid momentum update
           for (auto& node : grid_nodes_) {
               if (node.mass < 1e-6f) continue;

               // Velocity = momentum / mass
               float3 velocity = node.momentum / node.mass;

               // Apply forces (gravity, external forces)
               float3 acceleration = gravity_ + node.force / node.mass;

               // Update momentum
               node.momentum += node.mass * acceleration * dt;

               // Apply boundary conditions
               applyGridBoundaryConditions(node);
           }
       }

       void gridToParticle() {
           for (size_t p = 0; p < particle_data_.size(); ++p) {
               auto pos = particle_data_.getPosition(p);
               float3 new_vel = {0.0f, 0.0f, 0.0f};
               float3 affine_term = {0.0f, 0.0f, 0.0f};

               int3 cell_idx = positionToCell(pos);

               // Gather from grid
               for (int i = -1; i <= 1; ++i) {
                   for (int j = -1; j <= 1; ++j) {
                       for (int k = -1; k <= 1; ++k) {
                           int3 node_idx = cell_idx + int3{i, j, k};
                           if (!isValidCell(node_idx)) continue;

                           float weight = bSplineWeight(pos, nodeCellCenter(node_idx));
                           int node_linear = cellToLinearIndex(node_idx);

                           float3 grid_vel = grid_nodes_[node_linear].momentum /
                                            grid_nodes_[node_linear].mass;

                           new_vel += weight * grid_vel;

                           // Affine term for APIC
                           float3 dist = nodeCellCenter(node_idx) - pos;
                           affine_term += weight * grid_vel * dist;
                       }
                   }
               }

               // Update particle velocity and position
               particle_data_.setVelocity(p, new_vel);
               particle_data_.setPosition(p, pos + new_vel * dt);

               // Update deformation gradient (elastic/plastic)
               updateDeformationGradient(p, affine_term, dt);
           }
       }

       ParticleAoSoA<float> particle_data_;
       std::vector<GridNode> grid_nodes_;
       int3 grid_resolution_;
       float cell_size_;
       float3 gravity_;
   };

   } // namespace mpm
   } // namespace physgrad
   ```
   - [ ] Implement P2G transfer
   - [ ] Implement grid update
   - [ ] Implement G2P transfer
   - [ ] Implement B-spline weights
   - [ ] Test on simple elastic cube

3. **GPU kernels for MPM**
   ```cuda
   // src/mpm_kernels.cu (already exists, complete it)

   namespace physgrad {
   namespace mpm {

   __global__ void p2g_kernel(
       const ParticleAoSoA<float>* particles,
       GridNode* grid,
       int num_particles,
       int3 grid_resolution,
       float cell_size
   ) {
       int p = blockIdx.x * blockDim.x + threadIdx.x;
       if (p >= num_particles) return;

       // Get particle data
       float3 pos = particles->getPosition(p);
       float3 vel = particles->getVelocity(p);
       float mass = particles->getMass(p);

       // Find base cell
       int3 cell = make_int3(
           (int)(pos.x / cell_size),
           (int)(pos.y / cell_size),
           (int)(pos.z / cell_size)
       );

       // Transfer to 3x3x3 neighborhood
       for (int i = -1; i <= 1; ++i) {
           for (int j = -1; j <= 1; ++j) {
               for (int k = -1; k <= 1; ++k) {
                   int3 node_idx = cell + make_int3(i, j, k);

                   if (isValidNode(node_idx, grid_resolution)) {
                       float weight = computeBSplineWeight(pos, node_idx, cell_size);
                       int node_linear = nodeToLinearIndex(node_idx, grid_resolution);

                       // Atomic add for mass and momentum
                       atomicAdd(&grid[node_linear].mass, weight * mass);
                       atomicAdd(&grid[node_linear].momentum.x, weight * mass * vel.x);
                       atomicAdd(&grid[node_linear].momentum.y, weight * mass * vel.y);
                       atomicAdd(&grid[node_linear].momentum.z, weight * mass * vel.z);
                   }
               }
           }
       }
   }

   __global__ void grid_update_kernel(
       GridNode* grid,
       float dt,
       float3 gravity,
       int num_nodes
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx >= num_nodes) return;

       GridNode& node = grid[idx];

       if (node.mass < 1e-6f) return;

       // Compute velocity
       float3 velocity = node.momentum / node.mass;

       // Apply gravity
       velocity += gravity * dt;

       // Update momentum
       node.momentum = node.mass * velocity;

       // Boundary conditions (simple box)
       // TODO: Make configurable
   }

   __global__ void g2p_kernel(
       ParticleAoSoA<float>* particles,
       const GridNode* grid,
       float dt,
       int num_particles,
       int3 grid_resolution,
       float cell_size
   ) {
       int p = blockIdx.x * blockDim.x + threadIdx.x;
       if (p >= num_particles) return;

       float3 pos = particles->getPosition(p);
       float3 new_vel = make_float3(0.0f, 0.0f, 0.0f);

       int3 cell = make_int3(
           (int)(pos.x / cell_size),
           (int)(pos.y / cell_size),
           (int)(pos.z / cell_size)
       );

       // Gather from grid
       for (int i = -1; i <= 1; ++i) {
           for (int j = -1; j <= 1; ++j) {
               for (int k = -1; k <= 1; ++k) {
                   int3 node_idx = cell + make_int3(i, j, k);

                   if (isValidNode(node_idx, grid_resolution)) {
                       float weight = computeBSplineWeight(pos, node_idx, cell_size);
                       int node_linear = nodeToLinearIndex(node_idx, grid_resolution);

                       float3 grid_vel = grid[node_linear].momentum / grid[node_linear].mass;
                       new_vel += weight * grid_vel;
                   }
               }
           }
       }

       // Update particle
       particles->setVelocity(p, new_vel);
       particles->setPosition(p, pos + new_vel * dt);
   }

   } // namespace mpm
   } // namespace physgrad
   ```
   - [ ] Implement P2G kernel
   - [ ] Implement grid update kernel
   - [ ] Implement G2P kernel
   - [ ] Optimize atomic operations (use shared memory)
   - [ ] Profile and optimize

4. **G2P2G kernel fusion** (Performance optimization from paper)
   ```cuda
   // src/mpm_g2p2g_kernels.cu (already exists, complete it)

   __global__ void g2p2g_fused_kernel(
       ParticleAoSoA<float>* particles,
       GridNode* grid_current,
       GridNode* grid_next,
       float dt,
       int num_particles,
       int3 grid_resolution,
       float cell_size
   ) {
       // Fuse G2P of timestep n with P2G of timestep n+1
       // Reduces global memory traffic significantly

       int p = blockIdx.x * blockDim.x + threadIdx.x;
       if (p >= num_particles) return;

       // G2P: Read from grid_current, update particle
       float3 pos = particles->getPosition(p);
       float3 new_vel = make_float3(0.0f, 0.0f, 0.0f);

       // ... G2P logic ...

       particles->setVelocity(p, new_vel);
       particles->setPosition(p, pos + new_vel * dt);

       // P2G: Write to grid_next
       float3 updated_pos = particles->getPosition(p);
       float3 updated_vel = particles->getVelocity(p);
       float mass = particles->getMass(p);

       // ... P2G logic ...
   }
   ```
   - [ ] Implement fused kernel
   - [ ] Benchmark vs separate kernels
   - [ ] Measure memory bandwidth improvement

5. **Material models**
   ```cpp
   // src/mpm_materials.h

   namespace physgrad {
   namespace mpm {

   class MaterialModel {
   public:
       virtual ~MaterialModel() = default;

       // Compute Cauchy stress from deformation gradient
       virtual Matrix3x3 computeStress(
           const Matrix3x3& deformation_gradient,
           const MaterialProperties& props
       ) const = 0;

       // Update deformation gradient
       virtual Matrix3x3 updateDeformationGradient(
           const Matrix3x3& F_old,
           const Matrix3x3& velocity_gradient,
           float dt
       ) const = 0;
   };

   class NeoHookeanElastic : public MaterialModel {
   public:
       Matrix3x3 computeStress(
           const Matrix3x3& F,
           const MaterialProperties& props
       ) const override {
           // Neo-Hookean model
           // W = μ/2 (I₁ - 3) - μ log(J) + λ/2 log²(J)
           // P = ∂W/∂F

           float J = determinant(F);
           float mu = props.young_modulus / (2.0f * (1.0f + props.poisson_ratio));
           float lambda = props.young_modulus * props.poisson_ratio /
                         ((1.0f + props.poisson_ratio) * (1.0f - 2.0f * props.poisson_ratio));

           Matrix3x3 F_inv_T = transpose(inverse(F));

           Matrix3x3 P = mu * (F - F_inv_T) + lambda * log(J) * F_inv_T;

           return P;
       }
   };

   class DruckerPrager : public MaterialModel {
       // Elastoplastic model for granular materials (sand, soil)
       // Yield function: f = ||τ|| + α p - k
       // where τ is deviatoric stress, p is pressure
   };

   class SnowPlasticity : public MaterialModel {
       // From "A material point method for snow simulation" (Stomakhin et al., 2013)
       // Elastoplastic model with hardening
   };

   } // namespace mpm
   } // namespace physgrad
   ```
   - [ ] Implement Neo-Hookean elastic
   - [ ] Implement Drucker-Prager plasticity
   - [ ] Implement snow plasticity
   - [ ] Test each material model

6. **MPM validation tests**
   ```cpp
   // tests/test_mpm_complete.cpp

   TEST(MPM, ElasticBeamBending) {
       // Cantilever beam under gravity
       // Compare to analytical beam theory

       MPMSolver solver;
       solver.initialize(config);

       // Create beam (10cm x 1cm x 1cm)
       addElasticBeam(solver, length=0.1, width=0.01, height=0.01);

       // Fix left end
       solver.addBoundaryCondition(FixedBC(x < 0.01));

       // Simulate
       for (int i = 0; i < 1000; ++i) {
           solver.step(0.001f);
       }

       // Measure tip deflection
       float tip_deflection = solver.getTipPosition().y - initial_tip_y;

       // Compare to Euler-Bernoulli beam theory
       // δ = (w L⁴) / (8 E I)
       float theoretical_deflection = computeBeamDeflection(L, E, I, w);

       EXPECT_NEAR(tip_deflection, theoretical_deflection, 0.01);
   }

   TEST(MPM, PlasticDeformation) {
       // Test yield stress and plastic flow

       MPMSolver solver;

       // Create material with known yield stress
       DruckerPrager material(E, nu, yield_stress);

       // Apply increasing load
       // Verify elastic regime
       // Verify plastic flow after yield
   }

   TEST(MPM, MultiMaterialCollision) {
       // Elastic sphere + plastic cube

       MPMSolver solver;

       addElasticSphere(solver, pos={0, 1, 0}, radius=0.05, velocity={0, -1, 0});
       addPlasticCube(solver, pos={0, 0, 0}, size=0.1);

       // Simulate collision
       for (int i = 0; i < 5000; ++i) {
           solver.step(0.0001f);
       }

       // Verify material interface handling
       // Verify energy conservation (with plastic dissipation)
   }

   TEST(MPM, PerformanceScaling) {
       // Test scaling to 1M+ particles

       for (int num_particles : {1000, 10000, 100000, 1000000}) {
           MPMSolver solver;
           addRandomParticles(solver, num_particles);

           auto start = std::chrono::high_resolution_clock::now();

           for (int i = 0; i < 100; ++i) {
               solver.step(0.01f);
           }

           auto end = std::chrono::high_resolution_clock::now();
           auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

           std::cout << num_particles << " particles: "
                     << (100.0 / (duration.count() / 1000.0)) << " FPS\n";
       }
   }
   ```
   - [ ] Implement elastic beam test
   - [ ] Implement plastic deformation test
   - [ ] Implement multi-material test
   - [ ] Benchmark scaling performance

**Acceptance Criteria:**
- [ ] Full MPM solver working (P2G, grid update, G2P)
- [ ] AoSoA data structures implemented
- [ ] GPU acceleration functional
- [ ] 3+ material models working
- [ ] Validates against analytical solutions
- [ ] Scales to 1M+ particles
- [ ] All tests passing

**Deliverables:**
- Complete MPM solver
- GPU-accelerated kernels
- AoSoA implementation
- Multiple material models
- Validation tests
- Performance benchmarks
- Documentation

---

#### Task 1.4: Multi-Material MPM

```yaml
Priority: HIGH (Novelty factor)
Complexity: High
Dependencies: 1.3
Status: Not Started
```

**Objective:** Enable MPM to handle multiple materials simultaneously

**Subtasks:**

1. **Material interface handling**
   ```cpp
   // src/multi_material_mpm.h

   namespace physgrad {
   namespace mpm {

   enum class MaterialType {
       NEO_HOOKEAN_ELASTIC,
       DRUCKER_PRAGER_PLASTIC,
       SNOW_PLASTIC,
       FLUID_WEAKLY_COMPRESSIBLE
   };

   class MultiMaterialMPM : public MPMSolver {
   public:
       void setParticleMaterial(int particle_id, MaterialType type);

       void addMaterialModel(MaterialType type, std::unique_ptr<MaterialModel> model);

   private:
       // Different constitutive models per particle
       std::vector<MaterialType> particle_materials_;
       std::unordered_map<MaterialType, std::unique_ptr<MaterialModel>> materials_;

       // Override stress computation to use per-particle material
       void computeStress(int particle_id, Matrix3x3& stress) override;
   };

   } // namespace mpm
   } // namespace physgrad
   ```
   - [ ] Implement material assignment
   - [ ] Implement per-particle material lookup
   - [ ] Test material interface behavior

2. **Coupling with rigid bodies**
   ```cpp
   // src/mpm_rigid_coupling.h

   namespace physgrad {
   namespace mpm {

   class MPMRigidCoupling {
   public:
       MPMRigidCoupling(MultiMaterialMPM* mpm_solver);

       // Add rigid obstacles
       void addRigidBody(const RigidBody& body);

       // Resolve contact between MPM particles and rigid bodies
       void resolveMPMRigidContact();

       // Step coupled system
       void step(float dt);

   private:
       MultiMaterialMPM* mpm_solver_;
       std::vector<RigidBody> rigid_bodies_;

       // Detect MPM particles near rigid surface
       std::vector<int> detectParticlesNearRigid(const RigidBody& body);

       // Apply rigid boundary conditions on grid
       void applyRigidBoundaryConditions(const RigidBody& body);
   };

   } // namespace mpm
   } // namespace physgrad
   ```
   - [ ] Implement rigid body integration
   - [ ] Implement contact detection
   - [ ] Implement boundary conditions
   - [ ] Test MPM-rigid coupling

3. **Validation demos**
   ```cpp
   // examples/mpm_jello_cube.cpp

   int main() {
       // Elastic jello cube falling onto rigid ground

       MultiMaterialMPM mpm;
       mpm.initialize(config);

       // Add jello (soft elastic)
       addJelloCube(mpm, pos={0, 0.5, 0}, size=0.1);

       // Add rigid ground
       MPMRigidCoupling coupling(&mpm);
       coupling.addRigidBody(createGroundPlane());

       // Visualize
       Visualizer viz;
       while (viz.isOpen()) {
           coupling.step(0.001f);
           viz.render(mpm, coupling);
       }

       return 0;
   }
   ```
   ```cpp
   // examples/mpm_sand_pile.cpp
   // Granular material (sand) pouring

   // examples/mpm_snow_ball.cpp
   // Snow plasticity under compression
   ```
   - [ ] Create jello demo
   - [ ] Create sand pile demo
   - [ ] Create snow demo
   - [ ] Record videos

**Acceptance Criteria:**
- [ ] Multi-material MPM working
- [ ] Rigid coupling functional
- [ ] 3 demos running
- [ ] Videos/screenshots captured

**Deliverables:**
- Multi-material MPM
- MPM-rigid coupling
- 3 working demos
- Demo videos
- Documentation

---

### Week 12: Integration & Testing

#### Task 1.5: Integrate Geometric Integrators with Physics Engine

```yaml
Priority: HIGH
Complexity: Medium
Dependencies: 1.1, 1.2
Status: Not Started
```

**Objective:** Make all integrators accessible via unified physics engine API

**Subtasks:**

1. **Extend IntegrationMethod enum**
   ```cpp
   // src/common_types.h

   enum class IntegrationMethod {
       // Basic methods (already exist)
       EULER,
       VERLET,
       RUNGE_KUTTA_4,
       LEAPFROG,

       // Symplectic integrators (new)
       SYMPLECTIC_FSI_2,          // 2nd order
       SYMPLECTIC_FSI_4,          // 4th order (FROST)
       HIERARCHICAL_FSI,          // Multi-timestep

       // Variational integrators (new)
       VARIATIONAL_MIDPOINT,      // 2nd order
       VARIATIONAL_GALERKIN_4,    // 4th order
       VARIATIONAL_GALERKIN_6,    // 6th order
       VARIATIONAL_GALERKIN_8     // 8th order
   };
   ```
   - [ ] Add new enum values
   - [ ] Update documentation

2. **Update physics engine**
   ```cpp
   // src/physics_engine.h

   class PhysicsEngine {
   public:
       // ... existing methods ...

       void setIntegrationMethod(IntegrationMethod method);

   private:
       // Integrator instances
       std::unique_ptr<ForwardSymplecticIntegrator> symplectic_integrator_;
       std::unique_ptr<HierarchicalFSI> hierarchical_integrator_;
       std::unique_ptr<VariationalIntegrator> variational_integrator_;
       std::unique_ptr<GalerkinVariationalIntegrator> galerkin_integrator_;

       IntegrationMethod integration_method_;
   };
   ```
   ```cpp
   // src/physics_engine.cpp

   void PhysicsEngine::step(float dt) {
       // Update forces first
       updateForces();

       // Apply appropriate integrator
       switch (integration_method_) {
           case IntegrationMethod::EULER:
               stepEuler(dt);
               break;

           case IntegrationMethod::VERLET:
               stepVerlet(dt);
               break;

           case IntegrationMethod::SYMPLECTIC_FSI_4:
               if (!symplectic_integrator_) {
                   symplectic_integrator_ = std::make_unique<ForwardSymplecticIntegrator>();
               }
               symplectic_integrator_->step(positions_, velocities_, masses_, dt);
               break;

           case IntegrationMethod::HIERARCHICAL_FSI:
               if (!hierarchical_integrator_) {
                   hierarchical_integrator_ = std::make_unique<HierarchicalFSI>();
               }
               hierarchical_integrator_->hierarchicalStep(dt);
               break;

           case IntegrationMethod::VARIATIONAL_GALERKIN_4:
               if (!galerkin_integrator_) {
                   galerkin_integrator_ = std::make_unique<GalerkinVariationalIntegrator>(4);
               }
               galerkin_integrator_->step(positions_, velocities_, masses_,
                                         potentialFunction, dt);
               break;

           // ... other cases ...
       }

       // Apply boundary conditions
       applyBoundaryConditions();
   }
   ```
   - [ ] Implement integrator switching
   - [ ] Initialize integrators lazily
   - [ ] Handle state transfer between integrators

3. **Performance comparison tests**
   ```cpp
   // tests/test_integrator_comparison.cpp

   TEST(Integrators, PerformanceComparison) {
       // Same physical system (e.g., solar system)
       // Run with all integrators

       struct IntegratorResult {
           std::string name;
           double energy_drift;
           double momentum_error;
           double computation_time_ms;
           double max_timestep;
       };

       std::vector<IntegratorResult> results;

       for (auto method : {
           IntegrationMethod::EULER,
           IntegrationMethod::VERLET,
           IntegrationMethod::RUNGE_KUTTA_4,
           IntegrationMethod::SYMPLECTIC_FSI_4,
           IntegrationMethod::VARIATIONAL_GALERKIN_4
       }) {
           PhysicsEngine engine;
           engine.initialize();
           engine.setIntegrationMethod(method);

           // Setup test system
           setupSolarSystem(engine);

           float initial_energy = engine.calculateTotalEnergy();
           float3 initial_momentum = engine.calculateTotalMomentum();

           auto start = std::chrono::high_resolution_clock::now();

           // Integrate for fixed time
           for (int i = 0; i < 10000; ++i) {
               engine.step(0.01f);
           }

           auto end = std::chrono::high_resolution_clock::now();

           float final_energy = engine.calculateTotalEnergy();
           float3 final_momentum = engine.calculateTotalMomentum();

           IntegratorResult result;
           result.name = integratorName(method);
           result.energy_drift = std::abs(final_energy - initial_energy) / initial_energy;
           result.momentum_error = magnitude(final_momentum - initial_momentum);
           result.computation_time_ms =
               std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

           results.push_back(result);
       }

       // Print comparison table
       printResultsTable(results);

       // Verify symplectic integrators have best energy conservation
       EXPECT_LT(results[FSI_4].energy_drift, results[RK4].energy_drift);
   }
   ```
   - [ ] Implement comparison framework
   - [ ] Test on multiple physical systems
   - [ ] Generate performance report
   - [ ] Create comparison table/graph

**Acceptance Criteria:**
- [ ] All integrators accessible via setIntegrationMethod()
- [ ] Integrators can be switched dynamically
- [ ] Performance comparison documented
- [ ] Best integrator for each use case identified

**Deliverables:**
- Unified integrator API
- Performance comparison tests
- INTEGRATOR_COMPARISON.md report
- Recommendation guide

---

## 📋 **PHASE 2: DIFFERENTIABLE INFRASTRUCTURE**

**Duration:** 8-12 weeks
**Goal:** Build end-to-end differentiable pipeline
**Priority:** CRITICAL

*(This section continues with Tasks 2.1-2.4 covering adjoint methods, differentiable contact, PyTorch integration, and gradient verification - total ~50 more pages)*

---

## 📋 **PHASE 3: ADVANCED FEATURES & APPLICATIONS**

**Duration:** 12-16 weeks
**Goal:** Implement robotics applications and advanced physics
**Priority:** HIGH

*(This section covers Tasks 3.1-3.4: FSI coupling, robot co-design, manipulation demos, neural surrogates)*

---

## 📋 **PHASE 4: POLISH & VALIDATION**

**Duration:** 4-6 weeks
**Goal:** Production-ready framework
**Priority:** HIGH

*(This section covers Tasks 4.1-4.3: Documentation, performance validation, integration testing)*

---

## 🎯 **SUCCESS CRITERIA**

### Minimum Viable PhD (Must Have)

- ✅ Geometric integrators (symplectic + variational) implemented and validated
- ✅ GPU-accelerated MPM solver with multi-material support
- ✅ Fully differentiable physics pipeline (verified gradients)
- ✅ PyTorch integration with functional API
- ✅ Differentiable contact mechanics
- ✅ Robot co-design framework working
- ✅ 3+ manipulation demos with results

### Excellent PhD (Should Have)

- ✅ All "Must Have" items
- ✅ Fluid-structure interaction
- ✅ Soft object manipulation
- ✅ Performance validation (actual benchmarks)
- ✅ Multi-GPU scaling demonstration
- ✅ Comprehensive test suite (>80% coverage)

### Outstanding PhD (Nice to Have)

- ✅ All "Should Have" items
- ✅ Quantum-classical hybrid (if relevant to thesis)
- ✅ Neural surrogates for acceleration
- ✅ Real-world robot validation
- ✅ Open-source release with community adoption

---

## 📊 **TIMELINE SUMMARY**

| Phase | Duration | Weeks | Key Deliverables |
|-------|----------|-------|------------------|
| **Phase 0: Foundation** | 2-4 weeks | 1-4 | Building, testing, honest docs |
| **Phase 1: Core Contributions** | 8-12 weeks | 5-16 | Geometric integrators + MPM |
| **Phase 2: Differentiability** | 8-12 weeks | 17-28 | Adjoint methods + PyTorch API |
| **Phase 3: Applications** | 12-16 weeks | 29-44 | FSI + Robot co-design + Demos |
| **Phase 4: Polish** | 4-6 weeks | 45-50 | Documentation + Validation |
| **TOTAL** | **34-50 weeks** | **~8-12 months** | **PhD-worthy framework** |

---

## 🚀 **GETTING STARTED**

### Immediate Next Steps (This Week)

#### Day 1-2: Environment Setup

1. **Verify development environment**
   ```bash
   # Check CUDA
   nvcc --version
   nvidia-smi

   # Check compilers
   g++ --version  # Should be 9.0+
   cmake --version  # Should be 3.18+

   # Check Python
   python3 --version  # Should be 3.8+
   python3 -c "import torch; print(torch.__version__)"
   ```

2. **Create working branch**
   ```bash
   cd /home/tuso/Downloads/mestrado/cuda/physgrad
   git checkout -b phase0-foundation-repair
   git push -u origin phase0-foundation-repair
   ```

3. **Set up project tracking**
   - [ ] Create GitHub project board
   - [ ] Create issues for each task
   - [ ] Set up weekly progress log

#### Day 3-5: Initial Build Attempt

1. **Fix the build** (Task 0.1)
   ```bash
   mkdir -p build
   cd build
   cmake .. 2>&1 | tee cmake_output.log
   make -j$(nproc) 2>&1 | tee make_output.log
   ```

2. **Document all failures**
   - Create BUILD_ERRORS.md
   - List every compilation error
   - Categorize by type (missing dep, syntax, linking, etc.)

3. **Fix critical path**
   - Start with core modules only
   - Get libphysgrad_core building first
   - Document what works

#### Week 2: First Success

- [ ] Core library builds successfully
- [ ] At least one test runs
- [ ] Documented build process
- [ ] Updated README with actual build instructions

---

### Project Structure

```
physgrad/
├── IMPLEMENTATION_PLAN.md         # This file
├── IMPLEMENTATION_STATUS.md       # Current status (to be created)
├── BUILD_INSTRUCTIONS.md          # Detailed build guide (to be created)
├── TODO.md                        # Active task tracking
├── docs/
│   ├── theory/                    # Mathematical foundations
│   ├── implementation/            # Implementation details
│   └── tutorials/                 # Usage tutorials
├── progress/
│   └── weekly/                    # Weekly progress reports
│       ├── week01.md
│       ├── week02.md
│       └── ...
└── ... (existing structure)
```

---

### Tracking Progress

Create TODO.md for active tasks:

```markdown
# PhysGrad Active TODO List

## This Week (Week N)

### In Progress
- [ ] Task 0.1: Fix core build system
  - [x] Clean CMake configuration
  - [ ] Resolve CUDA architecture issues
  - [ ] Fix library dependencies

### Blocked
- [ ] Task 0.2: Fix disabled modules
  - Blocked by: Need Task 0.1 complete first

### Completed This Week
- [x] Created IMPLEMENTATION_PLAN.md
- [x] Set up project tracking
```

---

### Weekly Review Template

```markdown
# Week N Progress Report

## Date: YYYY-MM-DD

### Completed
- Task X.Y: Description
  - Subtask 1
  - Subtask 2

### In Progress
- Task X.Z: Description
  - Status: 60% complete
  - Blockers: None

### Next Week Goals
1. Complete Task X.Z
2. Start Task X.A

### Issues Encountered
- Issue 1: Description and resolution

### Metrics
- Lines of code: +XXX / -XXX
- Tests passing: XX/YY
- Build time: XX seconds
```

---

## 📝 **NOTES**

### Implementation Philosophy

1. **Correctness First, Performance Second**
   - Get physics right before optimizing
   - Validate against analytical solutions
   - Write tests before implementation

2. **Incremental Development**
   - Build on working foundation
   - One feature at a time
   - Continuous integration

3. **Documentation as You Go**
   - Document design decisions immediately
   - Keep implementation notes
   - Update docs with code

4. **Honesty Above All**
   - No unverified claims
   - Document known issues
   - Be realistic about timelines

### Key References

1. **Symplectic Integrators**
   - FROST paper (MNRAS 2021)
   - Hairer et al., "Geometric Numerical Integration" (2006)

2. **Variational Integrators**
   - Marsden & West (2001)
   - Lew et al. (2004)

3. **MPM**
   - SIGGRAPH 2020 multi-GPU MPM paper
   - Stomakhin et al. snow paper (2013)

4. **Differentiable Physics**
   - "Single-Level Differentiable Contact" (2022)
   - Taichi Lang papers

---

## 📞 **SUPPORT & RESOURCES**

### When Stuck

1. **Check documentation**
   - Read paper referenced in task
   - Look at similar implementations
   - Review test cases

2. **Break down further**
   - Simplify the problem
   - Test on minimal case
   - Build up gradually

3. **Ask for help**
   - Stack Overflow for specific issues
   - GitHub issues in related projects
   - Academic mailing lists

4. **Document the problem**
   - Write down what's not working
   - Often clarifies the solution
   - Helps future debugging

---

## ✅ **CHECKLIST SUMMARY**

### Phase 0 (Weeks 1-4)
- [ ] Build system working
- [ ] All modules compiling
- [ ] Tests running
- [ ] Documentation honest

### Phase 1 (Weeks 5-16)
- [ ] Symplectic integrators
- [ ] Variational integrators
- [ ] MPM solver
- [ ] Multi-material MPM

### Phase 2 (Weeks 17-28)
- [ ] Adjoint kernels
- [ ] Differentiable contact
- [ ] PyTorch integration
- [ ] Gradient verification

### Phase 3 (Weeks 29-44)
- [ ] FSI coupling
- [ ] Robot co-design
- [ ] Manipulation demos
- [ ] Results documented

### Phase 4 (Weeks 45-50)
- [ ] Documentation complete
- [ ] Performance validated
- [ ] Integration tested
- [ ] Framework ready

---

**Let's build this PhD framework!** 🚀

---

*End of Implementation Plan*
