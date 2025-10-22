# PhysGrad Performance Analysis & Optimization

## Profiling Results

### FSI Demo Profiling (10 runs, 800 fluid + 32 solid particles, 50 timesteps)

| Component | Time (ms) | Percentage | Calls |
|-----------|-----------|------------|-------|
| **FSI Coupling** | 1053.8 | 98.6% | 500 |
| Force Computation | 14.4 | 1.3% | 500 |
| Particle Update | <0.1 | <0.1% | 500 |
| **Total Simulation** | 1069.2 | 100% | 10 |

**Key Finding:** FSI coupling dominates runtime at 98.6%, processing at 2.1 ms per coupling step.

### Performance Metrics
- **Simulations/second:** 9.35
- **Timesteps/second:** 468
- **Memory usage:** 474 MB peak

## Identified Bottlenecks

### 1. **FSI Coupling: Vector Allocations** (HIGH PRIORITY)
**Problem:** `getFluidNeighbors()` allocates a new `std::vector` for every solid particle query.

**Current Code:**
```cpp
// Called 32 times per timestep (once per solid particle)
std::vector<uint32_t> getFluidNeighbors(...) const {
    auto all_neighbors = spatial_hash_->getNeighbors(...);  // Allocation 1
    std::vector<uint32_t> fluid_neighbors;                    // Allocation 2
    for (uint32_t idx : all_neighbors) {
        if (idx < fluid_particles.size()) {
            fluid_neighbors.push_back(idx);                   // Reallocations
        }
    }
    return fluid_neighbors;                                    // Return by value
}
```

**Impact:** 32 × 2 = 64 allocations per timestep, 32,000 allocations over optimization run

**Solution:**
- Reuse pre-allocated neighbor buffer
- Reserve capacity based on average neighbor count
- Return by reference or use output parameter

**Expected Speedup:** 10-20%

### 2. **Redundant Distance Calculations** (MEDIUM PRIORITY)
**Problem:** Distance computed twice - in neighbor search AND in force computation.

**Current Code:**
```cpp
auto fluid_neighbors = getFluidNeighbors(sx, sy, sz, ...);  // Distance check 1
for (uint32_t j : fluid_neighbors) {
    T distance = std::sqrt(dx*dx + dy*dy + dz*dz);          // Distance check 2
    if (distance < support_radius_) {  // Already guaranteed by getNeighbors!
        ...
    }
}
```

**Impact:** 800 × 32 = 25,600 redundant distance calculations per timestep

**Solution:**
- Remove redundant `if (distance < support_radius_)` check
- Store squared distances in neighbor query
- Reuse distance from spatial hash

**Expected Speedup:** 5-10%

### 3. **Particle Data Access Pattern** (LOW-MEDIUM PRIORITY)
**Problem:** Scattered memory access with frequent `getPosition`/`getVelocity` calls.

**Current Code:**
```cpp
for (uint32_t j : fluid_neighbors) {
    T fx, fy, fz, fvx, fvy, fvz;
    fluid_particles.getPosition(j, fx, fy, fz);    // 3 loads
    fluid_particles.getVelocity(j, fvx, fvy, fvz); // 3 loads
    T fluid_mass = fluid_particles.getMass(j);     // 1 load
    ...
}
```

**Impact:** Cache misses due to AoSoA layout and non-sequential access

**Solution:**
- Prefetch particle data
- Consider SoA (Structure of Arrays) for tight loops
- Batch process neighbors

**Expected Speedup:** 5-15% (depends on cache hierarchy)

### 4. **Delta Function Computation** (LOW PRIORITY)
**Problem:** Expensive function calls in tight loop.

**Current Code:**
```cpp
T delta_weight = deltaFunction(distance);  // Potentially expensive
```

**Solution:**
- Tabulate delta function for common distances
- Use polynomial approximation
- Inline small functions

**Expected Speedup:** 2-5%

## Optimization Strategy

### Phase 1: Quick Wins (Target: 20-30% speedup)
1. ✅ Add profiling instrumentation
2. ⏳ Eliminate vector allocations in neighbor queries
3. ⏳ Remove redundant distance checks
4. ⏳ Pre-allocate working buffers

### Phase 2: Algorithmic Improvements (Target: additional 15-25%)
1. ⏳ Improve spatial hash cell size tuning
2. ⏳ Batch particle processing
3. ⏳ Optimize memory layout for cache efficiency

### Phase 3: Advanced Optimizations (Target: 2-5x speedup)
1. ⏳ SIMD vectorization (AVX2/AVX-512)
2. ⏳ Parallel processing (OpenMP/TBB)
3. ⏳ GPU acceleration (CUDA kernels for FSI)

## Recommended Immediate Actions

### 1. Neighbor Query Optimization
```cpp
// Before
std::vector<uint32_t> getFluidNeighbors(...) const { ... }

// After
void getFluidNeighbors(T x, T y, T z,
                       std::vector<uint32_t>& neighbors_out) const {
    neighbors_out.clear();
    neighbors_out.reserve(expected_neighbors);
    spatial_hash_->getNeighbors(x, y, z, support_radius_, neighbors_out);
}
```

### 2. Eliminate Redundant Checks
```cpp
// Before
if (distance < support_radius_) {
    T delta_weight = deltaFunction(distance);
    ...
}

// After  (remove check - already guaranteed by neighbor search)
T delta_weight = deltaFunction(distance);
...
```

### 3. Batch Processing
```cpp
// Process all solid particles in batches for better cache utilization
const size_t batch_size = 8;
for (size_t batch_start = 0; batch_start < num_solid; batch_start += batch_size) {
    size_t batch_end = std::min(batch_start + batch_size, num_solid);
    // Process batch[batch_start:batch_end] together
}
```

## Long-Term Optimization Roadmap

### GPU Acceleration Potential
**Target Components:**
- FSI coupling (98.6% of runtime)
- Neighbor search (spatial hashing)
- Force distribution

**Estimated Speedup:** 10-50x for large particle counts (>10,000 particles)

**Implementation Effort:** Medium (3-5 days)
- CUDA kernels for neighbor search
- Parallel force computation
- Efficient memory transfer strategies

### Multi-Threading
**Current State:** Single-threaded
**Opportunity:** Particle-level parallelism

**Estimated Speedup:** 3-6x on modern CPUs (8-12 cores)
**Implementation Effort:** Low-Medium (1-2 days)

## Performance Targets

| Optimization Level | Expected Performance | Implementation Time |
|-------------------|---------------------|---------------------|
| **Current** | 468 timesteps/sec | - |
| **Phase 1** (Quick wins) | 600-700 timesteps/sec | 0.5 days |
| **Phase 2** (Algorithmic) | 800-1000 timesteps/sec | 1-2 days |
| **Phase 3** (Parallel) | 2000-4000 timesteps/sec | 3-5 days |
| **GPU** (Future) | 5000-20000 timesteps/sec | 5-10 days |

## Testing & Validation

**Performance Regression Tests:**
- Run profiler before/after each optimization
- Validate results match (bit-wise or within tolerance)
- Track speedup metrics in CI/CD

**Benchmarks:**
- Small (100 particles): Desktop responsiveness
- Medium (1000 particles): Research simulations
- Large (10000+ particles): Production workloads

## Conclusion

**Immediate ROI:** Phase 1 optimizations (20-30% speedup) with <1 day effort
**Best Long-term Investment:** GPU acceleration (10-50x) for scaling to production

**Critical Path:** FSI coupling optimization → Multi-threading → GPU kernels
