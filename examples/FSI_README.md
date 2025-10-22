# PhysGrad FSI Demo: Cylinder Drag Optimization

Fluid-Structure Interaction demonstration with gradient-based shape optimization.

## Overview

This demo showcases PhysGrad's FSI capabilities by optimizing a cylinder's shape to minimize drag force in fluid flow. It demonstrates:

- **Fluid-structure coupling** via immersed boundary method
- **Drag force computation** from FSI interactions
- **Shape parameterization** (radius, aspect ratio)
- **Gradient-based optimization** through differentiable physics

## The Problem

**Task:** Design a cylinder shape that minimizes drag in a 2 m/s fluid flow.

**Parameters to optimize:**
- Cylinder radius (0.1m - 0.5m)
- Aspect ratio (height/width: 0.5 - 2.0)

**Physics:**
- Fluid domain: 4.0m × 2.0m grid
- Inlet velocity: 2.0 m/s
- Fluid density: 1000 kg/m³
- Fluid viscosity: 0.001 Pa·s
- Immersed boundary coupling

## How to Run

```bash
cd build
./examples/demo_fsi_drag_optimization
```

## Expected Results

**Baseline (circular cylinder):**
- Radius: 0.200m
- Aspect ratio: 1.0
- Drag force: ~1.57 N

**After optimization:**
- Radius: ~0.166m (17% smaller)
- Aspect ratio: ~0.994 (nearly circular)
- Drag force: ~1.47 N
- **Drag reduction: 6.2%**

## Technical Details

### FSI Coupling Method

Uses **Immersed Boundary Method (IBM)**:
- Support radius: 0.15m
- Fluid particles: 800 (40×20 grid)
- Cylinder boundary: 32 points
- Time step: 0.001s
- Simulation steps: 50 per evaluation

### Optimization Approach

**Gradient computation:**
- Finite differences (central difference)
- Epsilon: 1e-4

**Optimization:**
- Method: Gradient descent
- Learning rate: 0.005
- Iterations: 20
- Constraints: Radius [0.1, 0.5], Aspect [0.5, 2.0]

### Drag Force Computation

Drag computed from FSI contact forces:
1. FSI coupling computes forces on cylinder boundary
2. Sum x-direction forces (flow direction)
3. Average over steady-state timesteps (last 50% of simulation)
4. Absolute value for magnitude

## Key Insights

1. **Smaller radius reduces drag** - Less frontal area = less resistance
2. **Circular cross-section is near-optimal** - Aspect ratio stays ~1.0
3. **FSI captures complex interactions** - Pressure and viscous forces
4. **Differentiable physics enables optimization** - Gradients flow through FSI coupling

## Applications

This demo pattern applies to:
- **Aerodynamic design**: Wing shapes, car bodies, turbine blades
- **Hydrodynamic optimization**: Boat hulls, submarine shapes
- **Microfluidics**: Channel design, mixer optimization
- **Biomedical**: Blood vessel shapes, stent design

## Limitations

**Current implementation:**
- 2D simulation only
- Simple immersed boundary coupling
- No turbulence model (laminar flow assumed)
- No structural deformation (rigid cylinder)

**Future improvements:**
- 3D simulations
- Turbulence models (LES, RANS)
- Fluid-structure deformation coupling
- Multi-objective optimization (drag + lift)
- Unsteady flow optimization

## Performance

**Single evaluation:**
- ~0.5-1.0 seconds per simulation
- ~40 evaluations per optimization (20 iters × 2 params)
- **Total time: ~40 seconds**

**Scaling:**
- O(n²) for naive contact detection
- O(n) for FSI force computation
- Can benefit from spatial hashing and GPU acceleration

## Code Structure

```
demo_fsi_drag_optimization.cpp
├── FSIDragConfig          # Configuration parameters
├── FSIDragSimulation      # FSI physics simulation
│   ├── simulate()         # Run FSI + compute drag
│   └── coupling_method_   # IBM coupling
└── FSIDragOptimizer       # Gradient-based optimizer
    └── optimize()         # Main optimization loop
```

## Related Examples

- `manipulation/demo_pushing_optimization.cpp` - Contact-rich manipulation
- `manipulation/demo_grasping_optimization.cpp` - Multi-contact optimization
- FSI tests in `test_fsi_*.cpp` - FSI infrastructure validation

## Citation

If you use this FSI demo in your research:

```bibtex
@software{physgrad_fsi_2024,
  title={PhysGrad FSI: Differentiable Fluid-Structure Interaction},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/physgrad}
}
```

## License

[Your License Here]
