# PhysGrad Manipulation Demos

Contact-rich robotics manipulation demonstrations using differentiable physics.

## Overview

This directory contains manipulation demos showcasing PhysGrad's differentiable physics for robotics applications. Each demo demonstrates gradient-based optimization through contact-rich interactions.

## Demos

### Demo 1: Gradient-Based Pushing ✅
**Status:** Initial implementation complete
**File:** `demo_pushing_optimization.cpp`
**Task:** Push a box to a goal position using gradient descent on trajectory

**Features:**
- Differentiable contact mechanics
- Gradient-based trajectory optimization
- End-to-end physics simulation
- Friction modeling

**How to Run:**
```bash
cd build
./examples/demo_pushing_optimization
```

**Current Status:**
- ✅ Framework implemented
- ✅ Compiles and runs
- ⚠️ Physics tuning needed (contact forces need adjustment)
- 🔄 Gradient flow validation in progress

**Next Steps:**
- Tune contact solver parameters for better force transfer
- Implement proper tangential friction forces
- Add visualization output
- Validate gradient accuracy

---

### Demo 2: Grasping Optimization
**Status:** Planned
**Task:** Optimize gripper configuration for stable grasp

---

### Demo 3: Object Stacking
**Status:** Planned
**Task:** Stack blocks using learned placement policy

---

### Demo 4: Tool Use
**Status:** Stretch goal
**Task:** Use a stick to push a distant object

---

### Demo 5: Multi-Object Assembly
**Status:** Stretch goal
**Task:** Assemble parts into target configuration

---

## Building

All demos are built automatically when `BUILD_EXAMPLES=ON`:

```bash
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
make demo_pushing_optimization
```

## Dependencies

- PhysGrad core library
- Differentiable contact mechanics (Phase 2)
- C++20 compiler
- (Optional) PyTorch for ML integration

## Demo Structure

Each demo follows this pattern:

1. **Task Configuration** - Define task parameters
2. **Physics Simulation** - Forward dynamics with differentiable contact
3. **Loss Function** - Task-specific objective
4. **Gradient Computation** - Adjoint method or finite differences
5. **Optimization Loop** - Gradient descent on control parameters
6. **Visualization** - (Optional) Real-time rendering

## Key Concepts

### Differentiable Contact
- Contact detection via sphere approximations
- Impulse-based contact resolution
- Friction modeling with Coulomb cone
- Gradients through contact solver

### Trajectory Optimization
- Parametric trajectory representation
- Gradient-based optimization (Adam, SGD)
- Constraint handling (joint limits, collisions)
- Multi-objective loss functions

### End-to-End Learning
- Full gradient flow from task loss to control parameters
- Integration with PyTorch autograd
- Physics-informed learning

## Performance Notes

- Finite difference gradients: O(n) forward simulations per iteration
- Adjoint method gradients: 1 forward + 1 backward per iteration
- Contact detection: O(n²) naive, O(n log n) with spatial hashing
- Contact solving: Iterative, typically converges in 10-50 iterations

## Troubleshooting

**Box doesn't move:**
- Check contact solver parameters (`contact_stiffness`, `relaxation`)
- Verify friction coefficient is non-zero
- Increase pusher mass or decrease box mass
- Check that contacts are being detected

**Optimization doesn't converge:**
- Reduce learning rate
- Increase number of iterations
- Check gradient magnitudes (should be non-zero)
- Verify loss function is differentiable

**Gradients are zero:**
- Check that contact is actually occurring
- Verify solver convergence
- Increase finite difference epsilon
- Check for numerical instability

## Citation

If you use these demos in your research, please cite:

```bibtex
@software{physgrad2024,
  title={PhysGrad: Differentiable Physics for Robotics},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/physgrad}
}
```

## License

[Your License Here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
