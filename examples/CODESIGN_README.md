# PhysGrad Co-Design Demo: Hopping Vehicle Optimization

End-to-end robot co-design demonstration with gradient-based optimization.

## Overview

This demo showcases PhysGrad's ability to simultaneously optimize both **morphology** (robot design) and **control** (behavior) through differentiable physics. It demonstrates:

- **Morphology optimization** (body mass, spring stiffness)
- **Control optimization** (hop frequency, thrust angle)
- **Contact-based physics** with ground friction
- **End-to-end differentiable co-design**

## The Problem

**Task:** Design a hopping vehicle (morphology + control) that maximizes forward locomotion distance.

**Morphology parameters to optimize:**
- Body mass (0.5kg - 2.0kg)
- Spring stiffness (10 N/m - 100 N/m)

**Control parameters to optimize:**
- Hop frequency (0.5Hz - 5.0Hz)
- Thrust angle (20° - 70° from vertical)

**Physics:**
- 2-second simulation (200 timesteps)
- Gravity: 9.81 m/s²
- Ground friction: 0.8
- Contact-based ground interaction

## How to Run

```bash
cd build
./examples/demo_robot_codesign
```

## Expected Results

**Baseline (initial parameters):**
- Mass: 1.0 kg
- Spring K: 50 N/m
- Frequency: 2.0 Hz
- Thrust angle: 45°
- Distance: ~7.9 m

**After optimization:**
- Parameters adapt based on gradient descent
- Demonstrates sensitivity to co-design
- Shows coupling between morphology and control

## Technical Details

### Physics Model

**Hopping Vehicle:**
- Single rigid body with mass
- Pulsed thrust actuation (frequency controlled)
- Angled thrust creates horizontal + vertical components
- Ground contact with friction enables forward motion

**Contact Resolution:**
- Baumgarte stabilization
- Coulomb friction model (tangent impulses)
- Differentiable contact solver
- Timestep: 0.01s

### Optimization Approach

**Gradient computation:**
- Finite differences (central difference)
- 4 parameters × 2 evaluations each = 8 forward sims per iteration
- Epsilon: 1e-4

**Optimization:**
- Method: Gradient ascent (maximize distance)
- Morphology learning rate: 0.01
- Control learning rate: 0.05
- Iterations: 30
- Constraints: All parameters bounded to reasonable ranges

### Co-Design Insights

1. **Mass affects momentum:** Lighter bodies accelerate faster but have less momentum
2. **Frequency affects efficiency:** Too fast wastes energy, too slow misses opportunities
3. **Thrust angle balances:** More horizontal = more forward thrust, more vertical = more hop height
4. **Coupling is critical:** Optimal morphology depends on control strategy and vice versa

## Key Results

**Distance traveled:** ~7+ meters in 2 seconds (successful locomotion)

**Optimization behavior:**
- Mass tends to optimize based on thrust-to-weight ratio
- Frequency adjusts to match natural hopping dynamics
- Angle balances forward progress vs. ground contact time
- Spring stiffness (less sensitive in this simplified model)

**Gradient flow:** End-to-end gradients flow from final distance → through contact physics → to all 4 design parameters

## Applications

This co-design pattern applies to:
- **Robot design:** Optimize morphology and control together
- **Soft robotics:** Material properties + actuation patterns
- **Locomotion:** Legged robots, swimming, flying
- **Manipulation:** Gripper design + grasping strategy
- **Vehicles:** Shape optimization + trajectory planning

## Limitations

**Current implementation:**
- Simplified 2D point-mass model
- No rotational dynamics
- Basic thrust model (no actuator limits)
- Finite difference gradients (not adjoint method)

**Future improvements:**
- Full rigid body dynamics (3D, rotation)
- Realistic actuator models
- Multi-link articulated robots
- Adjoint gradients for efficiency
- More complex morphology (variable geometry)

## Performance

**Single evaluation:**
- ~200 timesteps at 0.01s each
- Contact solver: ~10-50 iterations per contact
- **Time: ~0.02 seconds per simulation**

**Full optimization:**
- 30 iterations × 9 simulations (1 + 8 for gradients)
- **Total time: ~5-6 seconds**

**Scaling:**
- Linear in parameters (finite diff)
- Linear in timesteps
- Could use adjoint method for O(1) gradient computation

## Code Structure

```
demo_robot_codesign.cpp
├── RobotCoDesignConfig         # Configuration parameters
├── HoppingVehicleSimulation    # Physics simulation
│   └── simulate()              # Forward dynamics + contacts
└── RobotCoDesignOptimizer      # Co-design optimization
    └── optimize()              # Gradient-based co-design loop
```

## Related Examples

- `manipulation/demo_pushing_optimization.cpp` - Single-objective manipulation
- `manipulation/demo_grasping_optimization.cpp` - Multi-contact optimization
- `demo_fsi_drag_optimization.cpp` - FSI shape optimization

## Theoretical Background

**Co-Design Problem:**
```
max_{morphology, control} J(morphology, control)
s.t. dynamics(morphology, control) = physics
```

**Key Challenge:** Morphology and control are coupled through physics
**Solution:** Differentiable physics enables joint gradient-based optimization

**Why differentiable physics matters:**
- Traditional approach: Optimize morphology, then control (or vice versa)
- Co-design approach: Optimize both simultaneously via gradients
- Result: Better solutions due to considering coupling

## Citation

If you use this co-design demo in your research:

```bibtex
@software{physgrad_codesign_2024,
  title={PhysGrad Co-Design: Differentiable Robot Design and Control},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/physgrad}
}
```

## License

[Your License Here]
