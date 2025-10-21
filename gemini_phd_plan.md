

# **PhysGrad: A Strategic Analysis and Research Roadmap for Novel Contributions in Differentiable, GPU-Accelerated Computational Physics**

## **Part I: Competitive Landscape and Foundational Analysis**

The objective of this initial analysis is to rigorously situate the PhysGrad framework within the contemporary landscape of high-performance scientific computing. By conducting a thorough survey of state-of-the-art simulation engines and proposing a comprehensive benchmarking protocol, this section establishes the necessary context to identify and pursue avenues for novel scientific contribution. This foundational work is critical for defining a unique research identity for PhysGrad and ensuring its development is benchmarked against the highest standards of performance, accuracy, and stability.

### **Chapter 1: A Survey of Modern Physics Simulation Engines**

The field of computational physics has undergone a significant transformation, driven by the parallel processing power of Graphics Processing Units (GPUs) and the paradigm-shifting integration of machine learning. A successful PhD thesis built upon PhysGrad must not only demonstrate technical excellence but also a keen awareness of its position within this dynamic ecosystem. This chapter provides a comparative analysis of PhysGrad against leading platforms, categorizing them by their core architectural philosophies and strategic goals.

#### **1.1 The Paradigm of GPU-Native Simulation**

A dominant trend in high-performance simulation, particularly in robotics and reinforcement learning (RL), is the move towards fully GPU-resident pipelines. This architectural choice is motivated by the need to eliminate the performance bottleneck of data transfers between the CPU and GPU, which can dominate the runtime in highly parallelized, data-intensive applications.  
The NVIDIA Isaac ecosystem serves as the primary exemplar of this paradigm. The initial prototype, **Isaac Gym**, pioneered what it termed an "end-to-end GPU RL pipeline".1 Unlike traditional simulators that use the GPU as a co-processor for specific calculations before returning results to the CPU, Isaac Gym was designed to keep the entire simulation loop—state evolution, observation rendering, and reward calculation—on the GPU.2 Data is stored and manipulated as GPU tensors (e.g., PyTorch tensors), allowing for the simulation of tens of thousands of parallel environments on a single GPU, a scale previously requiring a data center.1 This design philosophy represents a significant departure from  
PhysGrad's current API, where methods like getPositions() and getVelocities() imply a data copy from the device (GPU) to the host (CPU) for every query, a pattern that would severely limit scalability in large-scale learning applications.  
The evolution of Isaac Gym into **Isaac Lab** signifies a maturation of this concept into a unified, open-source framework for robotics research.3 Built upon the Omniverse platform and Isaac Sim, Isaac Lab extends the GPU-native simulation core with photorealistic, RTX-based rendering, a wider array of simulated sensors (LIDAR, contact sensors), and support for diverse robotics workflows, including imitation learning and motion planning.5 The underlying physics engine for this ecosystem is  
**NVIDIA PhysX 5**, which itself provides direct GPU acceleration for a comprehensive feature set, including rigid bodies, articulations, FEM-based soft bodies, and particle systems.8 The strategic direction of the NVIDIA ecosystem is clear: to provide a holistic, high-fidelity platform where simulation is deeply integrated with advanced rendering and robotics-specific tooling, all accelerated by a GPU-native architecture.

#### **1.2 The Rise of High-Level, Differentiable Frameworks**

Concurrent with the push for raw performance is a movement towards frameworks that prioritize developer productivity, research flexibility, and, most critically, differentiability. Differentiable simulation, where the entire simulation process is a differentiable function, allows for the use of powerful gradient-based optimization techniques to solve inverse problems, such as control, system identification, and design optimization.  
**JAX-MD** is a prime example of a framework designed from the ground up for this purpose.9 Written entirely in Python and built on Google's JAX library, it adopts a functional programming paradigm with immutable data structures.9 It forgoes traditional object-oriented classes in favor of transforming arrays of data through composable functions. This design allows it to seamlessly leverage JAX's core transformations: just-in-time (  
jit) compilation for performance, automatic differentiation (grad) for computing gradients, and automatic vectorization (vmap) for running ensembles of simulations in parallel.9 While its performance may not match that of hand-tuned CUDA kernels for specific tasks, its strength lies in its extraordinary flexibility for research, enabling rapid prototyping and the direct integration of neural networks into the simulation loop.11 This functional, Python-native approach stands in stark contrast to  
PhysGrad's imperative, C++/OOP architecture.  
**Taichi Lang** offers a compelling middle ground.13 It is a domain-specific language (DSL) embedded within Python, designed for high-performance parallel programming.14 Taichi code, which shares Python's syntax, is just-in-time compiled into highly optimized machine code for multi-core CPUs and various GPU backends (CUDA, Vulkan, Metal).14 Its most profound innovation is the decoupling of computation logic from the underlying data structure layout via its  
SNode system.16 This allows a researcher to experiment with different memory arrangements (e.g., dense arrays, sparse grids, hash tables) to maximize performance for a given task without rewriting the core simulation code.13 Taichi also features a robust automatic differentiation system, making it a powerful tool for differentiable physics.13 This focus on memory layout optimization is highly relevant to  
PhysGrad's advanced memory management system and suggests a path for providing greater user control over performance tuning.

#### **1.3 Established Engines and Specialized Libraries**

The simulation landscape is also populated by mature, highly-regarded engines that serve as de facto benchmarks for accuracy and stability, as well as specialized libraries that provide deep functionality in specific domains.  
**MuJoCo** (Multi-Joint dynamics with Contact) has long been a standard in robotics and biomechanics research.17 Its key innovation was to combine the efficiency and accuracy of recursive algorithms in generalized (joint) coordinates with modern, optimization-based contact dynamics.19 This approach avoids the instabilities of Cartesian representations used in many gaming engines and the limitations of older spring-damper contact models.19 Since its acquisition and open-sourcing by DeepMind, MuJoCo's development has accelerated, with a clear trajectory towards the trends identified above. The introduction of  
**MuJoCo-Warp** demonstrates a commitment to GPU acceleration via NVIDIA's Warp kernel library 21, while  
**MJX** provides a fully JAX-native, differentiable version of the engine, reinforcing the critical importance of these features for future research.22  
For PhysGrad's fluid dynamics module, **SPlisHSPlasH** serves as an essential reference and benchmark.23 It is an open-source library dedicated to Smoothed Particle Hydrodynamics (SPH) and implements a comprehensive suite of state-of-the-art pressure solvers for simulating incompressibility, including WCSPH, PCISPH, PBF, IISPH, DFSPH, and PF.25 It also features advanced models for viscosity, surface tension, and vorticity, along with rigid-fluid coupling and GPU-accelerated neighborhood search.25 The breadth and depth of its SPH-specific features provide a clear roadmap for expanding  
PhysGrad's fluid simulation capabilities beyond its current implementation.

#### **1.4 Strategic Positioning for PhysGrad**

This survey reveals a complex landscape where different frameworks make explicit trade-offs between raw performance, developer productivity, and the availability of gradients for optimization. PhysGrad, with its C++/CUDA architecture, is currently positioned firmly in the high-performance camp, similar in spirit to the underlying PhysX engine but without the extensive ecosystem of Isaac Lab or the differentiability of modern frameworks.  
The path to a novel PhD thesis lies not in replicating an existing tool, but in carving out a unique and valuable position within this landscape. A central tension exists between the raw performance and hardware control offered by low-level C++/CUDA development and the flexibility and gradient-based power offered by high-level, differentiable Python frameworks. NVIDIA's tools provide immense power but within a large, prescriptive ecosystem. JAX-MD offers ultimate research flexibility but cedes some performance by relying on a general-purpose compiler (XLA) rather than hand-tuned kernels. PhysGrad has the opportunity to bridge this gap.  
The most compelling strategic direction is to evolve PhysGrad into a framework that **combines the performance of hand-optimized CUDA kernels with the flexibility of a fully differentiable, high-level Python API**. This would create a unique offering that does not currently exist: a system where researchers can leverage the Python ML ecosystem for rapid prototyping and gradient-based optimization, while still being able to drop down to C++/CUDA to implement custom, performance-critical components without sacrificing differentiability. This hybrid architecture would offer a novel solution to the "Productivity-Performance-Differentiability Trilemma," providing a powerful and unique contribution to the field.

| Feature | PhysGrad (Current) | NVIDIA Isaac Lab | JAX-MD | Taichi Lang | MuJoCo |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Core Architecture** | C++17 / CUDA | C++ / Python (Built on Isaac Sim/Omniverse) | Pure Python (JAX) | Python DSL (JIT to CPU/GPU) | C/C++ Core |
| **API Style** | Object-Oriented (C++/Python bindings) | Object-Oriented, Scene-Graph (USD) | Functional | Imperative, Pythonic | Procedural C API, OOP Python Bindings |
| **Differentiability** | None | Partial (via Newton/Warp) 21 | Full (Automatic Differentiation via JAX) 9 | Full (Source-to-source AD) 13 | Partial (via MJX/JAX) 22 |
| **Primary Application** | General Computational Physics | Robotics, RL, Sim-to-Real 6 | Molecular Dynamics, Soft Matter 27 | Graphics, Simulation, Vision 13 | Robotics, Biomechanics, RL 19 |
| **Contact Model** | Impulse-based | Optimization-based (PhysX) 8 | Potential-based (e.g., soft-sphere) 9 | User-defined (MPM, etc.) | Optimization-based (convex) 19 |
| **GPU Residency** | Kernels on GPU, control on CPU | End-to-end GPU pipeline 1 | Full JIT compilation to GPU/TPU 11 | Full JIT compilation to GPU 13 | CPU-default, GPU via MJX/Warp 21 |
| **Extensibility** | High (Modular C++ design) | High (Python extensions, USD) | High (Pure Python, composable) | High (Python ecosystem) | High (Callbacks, plugins) |

### **Chapter 2: Benchmarking for Excellence**

For PhysGrad to be accepted as a serious contribution to computational science, its claims of being "high-performance" and "production-ready" must be substantiated through rigorous, reproducible benchmarking against established standards. This chapter outlines a proposed suite of benchmarks designed to validate PhysGrad's core capabilities and guide its future optimization. The creation and publication of a challenging new benchmark suite could, in itself, constitute a valuable contribution to the research community.28

#### **2.1 Defining the Metrics of Success**

A robust evaluation framework must be built on a clear set of quantitative metrics. The following metrics should be systematically tracked for all benchmark scenarios:

* **Performance:**  
  * *Particle/Body Throughput:* The number of simulated entities (particles, rigid bodies) processed per second. This is a primary measure of raw computational speed.  
  * *Simulation Time per Step:* The wall-clock time required to advance the simulation by a single timestep (Δt), broken down by sub-steps (e.g., neighbor search, force computation, integration, collision response).  
  * *GPU Occupancy and Utilization:* Measured using tools like NVIDIA's Nsight Systems, this metric indicates how effectively the implemented CUDA kernels are utilizing the GPU's streaming multiprocessors. High occupancy is crucial for maximizing performance.29  
* **Accuracy:**  
  * *Energy Conservation:* For conservative systems (e.g., N-body gravity, ideal elastic collisions), the total energy of the system should remain constant. The relative energy error, ∣ΔE/E0​∣, should be tracked over long simulation times to detect numerical drift.  
  * *Momentum Conservation:* Linear and angular momentum should be conserved in the absence of external forces or torques. This is a critical validation for the correctness of force calculations and integration schemes.  
  * *Ground Truth Comparison:* Where analytical solutions exist (e.g., the oscillating drop problem 30), the simulation results should be compared directly to the ground truth to quantify error.  
* **Stability:**  
  * *Maximum Stable Timestep:* The largest timestep (Δt) at which the simulation remains numerically stable without catastrophic failure (e.g., "exploding" velocities).  
  * *Robustness to Complex Scenarios:* The ability of the simulator to handle challenging configurations, such as large stacks of objects or high-energy collisions, without failure or physically implausible artifacts.

#### **2.2 Rigid Body and Contact Dynamics Benchmarks**

The contact mechanics system is often the most complex and performance-critical component of a rigid body simulator. The benchmark suite must therefore include scenarios designed to stress-test this system. The literature indicates a need for more diverse and challenging benchmarks beyond simple, single-body interactions.28

* **Proposed Scenarios:**  
  1. **Stacking Stability:** A classic test involving stacking multiple objects (e.g., cubes, cylinders) into a tall tower. The benchmark measures the number of objects that can be stacked before the structure becomes unstable and collapses. This tests the solver's ability to handle multiple persistent contacts and prevent energy creep.31  
  2. **Friction and Rolling:** Simulating a sphere or cylinder rolling down an inclined plane with varying coefficients of friction. The simulation should correctly reproduce the transition from static friction (rolling) to kinetic friction (sliding), and the final linear and angular velocities should be compared against analytical solutions.  
  3. **Complex Contact Geometry:** Scenarios involving many non-convex objects with tight clearances, such as a pile of interlinked chain segments or the meshing of complex gears. These tests evaluate the robustness and performance of both the broad-phase and narrow-phase collision detection algorithms.32  
  4. **High-Impact Collision:** A scenario involving a "bowling alley" setup where a high-velocity sphere collides with a structured arrangement of objects. This tests the impulse-based resolution, energy dissipation, and stability of the contact solver under high-energy conditions.  
  5. **Granular Flow:** Simulating the flow of thousands of small rigid bodies (e.g., spheres or capsules) through a funnel (the "hourglass" test). This benchmark measures the simulator's ability to handle a massive number of simultaneous, transient contacts and serves as a key performance indicator.

#### **2.3 Fluid Dynamics (SPH) Benchmarks**

The SPH implementation in PhysGrad must be validated against standard computational fluid dynamics (CFD) benchmarks to demonstrate its accuracy and physical fidelity.

* **Proposed Scenarios:**  
  1. **The Dam Break Problem:** This is the canonical benchmark for free-surface flows.33 The simulation involves a column of water that is suddenly released. Key validation metrics include the position of the wave front over time and the water height at specific probe locations, which can be compared directly with extensive experimental data available in the literature.30 Variations should include both a dry downstream bed and a wet bed to test different flow regimes.  
  2. **Oscillating Drop:** This test case involves a 2D circular or elliptical drop of fluid oscillating under a central force field.30 It is an excellent benchmark for validating the implementation of surface tension forces and for assessing the conservation of energy, as an analytical solution for the oscillation period and shape deformation exists.30  
  3. **Sloshing Tank:** A partially filled, sealed container is subjected to prescribed harmonic motion. The simulation must accurately predict the resulting wave patterns and the impact forces on the container walls. This is a critical benchmark for applications in the automotive and aerospace industries.  
  4. **Fluid-Structure Interaction (FSI):** A dam break flow impacting a flexible, elastic plate. This advanced benchmark, for which experimental data is available 37, would test the coupling between  
     PhysGrad's fluid and solid mechanics modules. Success in this benchmark would demonstrate a significant and novel capability.

By systematically executing this suite of benchmarks, PhysGrad can generate the quantitative data necessary to validate its performance claims, identify areas for improvement, and solidly position its contributions within the scientific literature.

## **Part II: Core Architectural and Algorithmic Advancements**

To transition PhysGrad from a capable engineering tool to a platform for novel scientific research, its core algorithms must be extended beyond standard implementations. This part of the report details specific, high-impact advancements in numerical integration and physical modeling. These enhancements are not merely incremental improvements; they are foundational prerequisites for achieving long-term physical accuracy and for enabling the more advanced differentiable capabilities that will form the core of the proposed thesis.

### **Chapter 3: Preserving Structure \- The Case for Geometric Integration**

Numerical integrators are the heart of any physics simulation, dictating how the system's state evolves over time. The choice of integrator has profound implications for the long-term accuracy and physical plausibility of the simulation.

#### **3.1 Beyond Euler and Verlet: The Need for Long-Term Stability**

The current implementation of PhysGrad includes several standard numerical integrators: Euler, Verlet, Runge-Kutta 4 (RK4), and Leapfrog. While these methods are widely used and sufficient for many visual or short-term applications, they have a fundamental limitation for scientific computing: they do not, in general, preserve the underlying geometric structure of the physical system.  
For Hamiltonian systems—which describe a vast class of conservative physical phenomena from planetary motion to molecular dynamics—the dynamics are governed by a geometric structure known as a symplectic manifold. Standard integrators like RK4, while offering high-order accuracy for a single step, do not respect this structure. Over many thousands or millions of timesteps, this leads to the accumulation of numerical errors that manifest as a secular drift in the total energy of the system.38 A simulation of a planetary system using RK4, for example, might show the planet slowly spiraling away from or into its star, an unphysical artifact of the numerical method. For a simulation to be scientifically credible, particularly for long-term predictions, it must employ methods that mitigate or eliminate such drifts.

#### **3.2 Symplectic Integrators**

Symplectic integrators are a class of numerical methods specifically designed for Hamiltonian systems. Instead of just approximating the trajectory, they are constructed to exactly preserve the symplectic structure of the phase space.38 This property ensures that while the computed energy may oscillate around the true value, it will not exhibit secular drift over exponentially long times, making them the gold standard for long-term simulations in celestial mechanics and molecular dynamics.38  
A state-of-the-art example suitable for implementation in PhysGrad is the **fourth-order forward symplectic integrator (FSI)**, as implemented in the GPU-accelerated N-body code **FROST**.38 This method achieves fourth-order accuracy using only positive (forward) timesteps, which avoids stability issues present in some other high-order methods that require negative time substeps.40 Crucially, this is achieved at the cost of computing an additional term: the gradient of the force (or equivalently, the Hessian of the potential energy).40  
The implementation of such an integrator in PhysGrad would be a significant scientific enhancement. The requirement to compute force gradients provides a direct and powerful link to the goal of building a differentiable simulator. The algorithmic machinery developed to compute these gradients for the integrator can be repurposed and extended for the full backward pass required by automatic differentiation. Thus, pursuing a state-of-the-art geometric integrator is not a detour, but a strategic and foundational step toward the primary thesis goal. Furthermore, FROST's hierarchical formulation (HHS-FSI), which uses different timesteps for different parts of the system, is manifestly momentum-conserving and scales well on multi-GPU systems, providing a blueprint for advanced, scalable integrators in PhysGrad.38

#### **3.3 Variational Integrators**

Variational integrators represent a more fundamental and systematic approach to constructing structure-preserving algorithms.44 Instead of discretizing the differential equations of motion (the Euler-Lagrange equations), this methodology starts one level higher, at the principle of least action. The continuous action integral is first approximated by a discrete sum, and then a discrete version of the variational principle is applied to derive the discrete equations of motion.39  
This "discretize-then-variate" approach has profound consequences. The resulting numerical schemes are **automatically symplectic and momentum-preserving** whenever the discrete Lagrangian exhibits the corresponding symmetries (time-invariance and spatial-invariance, respectively).44 The Störmer-Verlet method, already present in  
PhysGrad, is the simplest example of a variational integrator, derived from a simple midpoint or trapezoidal quadrature rule for the action.44  
To advance PhysGrad, the implementation of higher-order variational integrators, such as those based on Galerkin methods, is recommended. In a Galerkin variational integrator, the trajectory between two timesteps is approximated by a polynomial from a finite-dimensional function space, and the coefficients of this polynomial are chosen to satisfy a weak form of the Euler-Lagrange equations. This provides a systematic way to construct integrators of any desired order of accuracy that retain the excellent conservation properties of the variational framework. The development and GPU-acceleration of a high-order Galerkin variational integrator within PhysGrad would represent a novel contribution to the field of geometric numerical integration.

### **Chapter 4: The Nuances of Interaction \- Advanced Contact and Material Models**

While the integrator governs the evolution of state over time, the physical models dictate the forces and interactions that drive this evolution. To achieve novelty, PhysGrad must expand its physical modeling capabilities beyond simple rigid bodies and basic SPH to encompass the complex behaviors of deformable solids and the intricate, non-smooth dynamics of contact.

#### **4.1 From Rigid Bodies to Deformable Continua**

The world is not exclusively composed of rigid objects and simple fluids. The ability to simulate soft, deformable bodies is a critical feature of modern, general-purpose physics engines like NVIDIA's PhysX 5 and Isaac Lab.5 Applications in robotics (soft grippers), visual effects (organic materials), and engineering (material failure) all depend on high-fidelity simulation of deformable continua.

#### **4.2 The Material Point Method (MPM)**

While Finite Element Methods (FEM) are a standard for simulating deformable solids, the **Material Point Method (MPM)** has emerged as an exceptionally powerful and versatile alternative, particularly in computer graphics and for problems involving extreme deformation.47 MPM is a hybrid Eulerian-Lagrangian method that uses a collection of material points (particles) to carry physical properties like mass, velocity, and deformation, while computations are performed on a background Cartesian grid. This unique formulation allows it to naturally handle complex phenomena like self-collision, fracture, and topological changes that are challenging for traditional mesh-based methods.48  
MPM is an ideal candidate for implementation in PhysGrad due to its high degree of parallelism and its suitability for GPU acceleration. Cutting-edge research presented at SIGGRAPH provides a clear blueprint for a high-performance, multi-GPU implementation.47 Key algorithmic and data structure innovations to consider include:

* **Array-of-Structs-of-Array (AoSoA) Data Structure:** A particle data structure designed to promote coalesced memory access on the GPU, maximizing memory bandwidth and eliminating the need for costly atomic operations during the particle-to-grid transfer step.47 This aligns perfectly with  
  PhysGrad's existing focus on memory efficiency.  
* **G2P2G Kernel Fusion:** A technique that reformulates the MPM pipeline by fusing the Grid-to-Particle (G2P) step of one timestep with the Particle-to-Grid (P2G) step of the next. This fusion significantly reduces the number of GPU kernel launches and global memory traffic, leading to substantial performance gains.47

Moreover, the versatility of MPM allows it to model a wide range of materials—including elastic solids, elastoplastic materials (like sand or snow), and even fluids—within a single, unified computational framework.47 This opens the door to simulating complex, multi-material interactions, a significant research frontier.

#### **4.3 Differentiable Contact Mechanics**

Contact dynamics are arguably the most challenging aspect of physical simulation to make differentiable. The underlying physics is inherently non-smooth and discontinuous. When an object makes contact, the governing equations change abruptly. The impulse-based contact model currently in PhysGrad is a prime example of such a non-smooth model, which poses a significant barrier to gradient-based optimization.  
A core research task for the thesis must be the development and implementation of a **differentiable contact model**. This is an active area of research in both robotics and computer graphics. Promising approaches that move beyond non-differentiable formulations include:

* **Penalty-Based Methods:** Contact is modeled as a stiff spring force that penalizes interpenetration. While simple and differentiable, these methods can require very small timesteps for stability and can introduce unwanted oscillations.  
* **Relaxation of Complementarity Constraints:** Traditional optimization-based contact models are often formulated as a Linear Complementarity Problem (LCP), which is non-smooth. These can be made differentiable by relaxing the complementarity condition (a≥0,b≥0,a⋅b=0) into a smooth approximation, such as a⋅b=ϵ for some small ϵ.  
* **Single-Level Optimization:** Recent research has proposed unifying the traditionally separate steps of collision detection and contact resolution into a single, unified optimization problem.51 This approach avoids ambiguities in defining contact points and normals that plague bilevel formulations and can be formulated to be fully differentiable. The paper "Single-Level Differentiable Contact Simulation" provides a direct and powerful blueprint for a novel implementation in  
  PhysGrad that handles a variety of convex primitives and their compositions.51

The integration of a differentiable MPM solver with a differentiable contact model would position PhysGrad at the absolute forefront of research. It would create a unified, differentiable framework for simulating complex, multi-material interactions involving both soft and rigid bodies—a capability that is highly sought after for robotics and co-design applications but is not readily available in existing open-source frameworks. This unified differentiable engine for multi-material physics would form a specific, novel, and highly impactful core for a PhD thesis.

## **Part III: The Differentiable Frontier: A Thesis Breakthrough**

This part of the report outlines the central research thrust proposed for the PhysGrad project: its transformation into a fully differentiable physics engine. This capability is not an incremental feature; it is a paradigm shift that unlocks the use of powerful, gradient-based optimization methods, forming a direct bridge between high-performance simulation and modern artificial intelligence. The successful development of this differentiable architecture and its application to challenging problems in robotics and AI will constitute the primary novel contribution of the doctoral thesis.

### **Chapter 5: The Gradient Revolution \- Architecting a Fully Differentiable PhysGrad**

The ability to compute the derivative of a simulation's outcome with respect to its initial conditions and parameters is the key that unlocks a new class of scientific inquiry. While a forward simulation answers the question, "Given these inputs, what is the outcome?", a differentiable simulation can answer the inverse question: "To achieve this desired outcome, what should the inputs be?".53 This is the fundamental question in control, planning, system identification, and design optimization.

#### **5.1 The "Why": Inverse Problems and Gradient-Based Optimization**

Solving inverse problems with traditional, non-differentiable ("black-box") simulators requires gradient-free optimization methods, such as evolutionary algorithms or reinforcement learning techniques based on random sampling. These methods are notoriously sample-inefficient, often requiring millions of forward simulation runs to solve even moderately complex problems.53  
By providing analytical gradients, a differentiable simulator allows for the use of far more efficient gradient-based optimization algorithms (e.g., L-BFGS, Adam). These methods can converge to a solution orders of magnitude faster, making it feasible to solve high-dimensional optimization problems that are intractable with gradient-free approaches.53 This efficiency is the primary motivation for building a differentiable  
PhysGrad.

#### **5.2 Architectural Blueprint for Differentiability**

Transforming PhysGrad into a differentiable engine requires a systematic approach where the derivative of every computational step is explicitly defined. This is typically achieved by implementing an **adjoint method**, which computes the gradient of a final scalar loss function with respect to all prior states and parameters by propagating derivatives backward through the simulation's computation graph (a technique also known as backpropagation through time).  
A concrete architectural plan for PhysGrad involves the following key components:

1. **Differentiating Custom CUDA Kernels:** This is where PhysGrad's C++/CUDA foundation becomes a unique and powerful asset. Unlike frameworks that rely on general-purpose compilers like JAX or Taichi, PhysGrad allows for hand-optimization of its core computational kernels. To make these differentiable, the **adjoint (backward pass) for each custom kernel must be manually derived and implemented**. For example, for the classical\_force\_kernel, a corresponding classical\_force\_kernel\_backward must be written. This backward kernel would take the gradient of the loss with respect to the output forces (∂L/∂F) and compute the gradients with respect to the input positions and charges (∂L/∂r and ∂L/∂q). While requiring more implementation effort, this approach allows for the same level of performance optimization on the backward pass as on the forward pass, a feature not readily available in higher-level systems.  
2. **Differentiating the Integrator:** The numerical integration scheme itself is a function that maps the state at time t to the state at time t+dt, and it must be part of the differentiable chain. The backward pass for the chosen geometric integrator (from Chapter 3\) must be derived. For a simple Verlet step, this is straightforward; for a higher-order symplectic or variational integrator, this is a more involved but well-defined mathematical task.  
3. **Differentiating the Contact Model:** The non-smooth nature of contact is the primary challenge. By implementing one of the differentiable contact models discussed in Chapter 4 (e.g., a single-level optimization formulation 51), its analytical gradient can be computed and integrated into the overall backward pass of the simulation step.

#### **5.3 API Design for a Differentiable Simulator**

The public-facing API, particularly the Python interface, must be redesigned to support gradient-based workflows and integrate seamlessly with the dominant deep learning ecosystems. The current imperative, stateful API (engine.add\_particles(), engine.step()) is ill-suited for this.  
A new, functional API inspired by JAX and PyTorch is proposed. The core simulation step would be exposed as a pure function:

Python

\# Proposed Functional API  
def step(state, action, params):  
    \# This function calls the underlying C++/CUDA kernels  
    \# and returns the new state. It is stateless.  
    next\_state \= physgrad.functional\_step(state, action, params)  
    return next\_state

Here, state would be a data structure (e.g., a dictionary of NumPy arrays or PyTorch tensors) containing all particle positions, velocities, etc. params would contain physical parameters like masses, charges, or friction coefficients.  
This functional design allows machine learning frameworks to trace the computation. A full simulation trajectory becomes a composition of these step functions. When a user calls .backward() on a loss computed from the final state in PyTorch, the framework's autograd engine will propagate gradients back through time. At each step function, a custom "autograd Function" would be invoked, which knows how to call PhysGrad's highly optimized C++/CUDA backward pass to compute the necessary gradients before passing them further down the chain.  
This hybrid C++/Python architecture represents the core novelty of the proposed thesis. It does not seek to build another pure-Python differentiable simulator, nor another monolithic C++ engine. Instead, it proposes a new framework and methodology for integrating hand-optimized, high-performance CUDA code into the modern, Python-based, differentiable computing ecosystem. This contribution is more general and impactful than the simulator itself; it provides a blueprint for performance-centric scientific machine learning.

### **Chapter 6: The Confluence of Physics and AI**

With a differentiable PhysGrad in place, the framework becomes a powerful tool for research at the intersection of physical simulation and artificial intelligence. This chapter explores several key research directions that leverage this unique capability.

#### **6.1 Physics-Informed Neural Networks (PINNs)**

Physics-Informed Neural Networks have emerged as a powerful paradigm for solving differential equations and modeling physical systems.57 PINNs are neural networks whose loss function includes a term that penalizes violations of known physical laws, typically expressed as a partial differential equation (PDE).60

* **PhysGrad as a Discretization Engine for PINNs:** Instead of relying on automatic differentiation within a neural network to approximate spatial derivatives (which can be slow and struggle with high-frequency components), PhysGrad's highly optimized operators can be used to compute the PDE residual. For example, the SPH formulation of the Navier-Stokes equations provides a particle-based discretization of the differential operators. This residual can be computed efficiently by PhysGrad's CUDA kernels and then used in the loss function of a PINN. This approach is formalized in recent work on **SK-PINNs (Smoothing Kernel-Informed Neural Networks)**, which explicitly use SPH kernels to accelerate PINN training, representing a direct and promising research avenue.62  
* **Hybrid Physics-ML Models:** A more advanced and novel application is the creation of hybrid models. In many complex systems, some physical processes are well-understood and can be modeled with first principles, while others (e.g., turbulence closure, complex material constitutive laws, friction models) are difficult to model analytically. A hybrid model would use PhysGrad's trusted solvers for the well-understood parts of the system, while a neural network learns the difficult, data-driven component. Because the entire PhysGrad pipeline is differentiable, this hybrid system can be trained end-to-end, allowing the neural network to learn a model that is consistent with both the observational data and the known physics simulated by the rest of the engine.

#### **6.2 Neural Surrogate Modeling**

While PhysGrad is designed for high-fidelity simulation, many applications like interactive design or digital twins require real-time predictions that are faster than a full simulation allows.63  
PhysGrad can be used as a data generation engine to create massive datasets of simulation trajectories under a wide range of initial conditions and parameters. A neural network (e.g., a Graph Neural Network 65 or a Transformer) can then be trained on this data to act as a fast  
**neural surrogate model**. This surrogate learns to approximate the results of the high-fidelity simulation but with a much lower computational cost at inference time. NVIDIA's PhysicsNeMo framework is an example of a platform designed for building such AI surrogates.63

#### **6.3 Differentiable Simulation for Inverse Problems**

This is the most direct and powerful application of the differentiable PhysGrad framework. It enables the efficient solution of inverse problems that are fundamental to science and engineering.

* **System Identification:** This involves inferring the unknown parameters of a physical model from observed data. For instance, given a video of a deformable object being manipulated, the differentiable simulator can be used with gradient descent to find the material properties (e.g., Young's modulus, Poisson's ratio, yield stress) that cause the simulation to best match the observed behavior.69 This is crucial for closing the "sim-to-real" gap in robotics, ensuring that simulations accurately reflect reality.  
* **Trajectory Optimization and Control:** This is the problem of finding a sequence of actions or control inputs that causes a system to reach a desired goal state. With a differentiable simulator, the entire trajectory of the system over time can be treated as one large, differentiable function. One can define a loss based on the difference between the final simulated state and the goal state, and then backpropagate through the entire simulation to compute the gradient of the loss with respect to the control inputs at every timestep. This allows for highly efficient optimization of complex, contact-rich manipulation plans.69

### **Chapter 7: Premier Applications \- Robotics, Control, and Co-Design**

To ground the theoretical contributions of the thesis in high-impact applications, this chapter proposes a series of case studies that leverage the full power of a differentiable, multi-material PhysGrad. These applications, particularly in robotics, represent grand challenges where progress is currently limited by the capabilities of existing simulation tools.

#### **7.1 Dexterous Manipulation and Contact-Rich Tasks**

Robotic manipulation, especially tasks involving fine motor skills and complex contact interactions, remains a significant open problem. Differentiable simulation offers a path to creating controllers for such tasks with unprecedented efficiency.

* **Benchmark Tasks:** The validation of PhysGrad should be performed on a suite of challenging manipulation benchmark tasks drawn from the robotics literature, such as those found in RLBench or the Functional Manipulation Benchmark.73 Proposed tasks include:  
  * **Peg-in-Hole Insertion:** A classic precision task requiring fine-grained reasoning about contact forces.  
  * **In-Hand Reorientation:** Manipulating an object within a robotic hand to change its pose, a task that involves a continuous sequence of making and breaking contacts.75  
  * **Tool Use:** Using a rigid tool to manipulate a soft or deformable object, a scenario that would showcase PhysGrad's unique multi-material capabilities.  
* **Skill Transfer:** A key research area is policy or skill transfer, where a control policy learned for a source task is efficiently adapted to a new, related target task. The **Diff-Transfer** framework, for example, uses gradients from a differentiable simulator to find a path of intermediate tasks that smoothly bridges the source and target, adapting the action sequence at each step.69 Implementing and extending such a framework using  
  PhysGrad would be a powerful demonstration of its capabilities.

#### **7.2 Robot Co-Design**

A truly frontier research direction, enabled by differentiable physics, is the **co-design** or **co-optimization** of a robot's physical form (morphology) and its control policy.76 Traditional robot design is a sequential process: engineers first design the hardware, and then control engineers develop a controller for that fixed design. This separation is suboptimal, as the body and "brain" are deeply intertwined.  
With a differentiable PhysGrad, physical parameters of the robot—such as link lengths, gear ratios, actuator strengths, and even the material stiffness of soft components—can be treated as trainable variables alongside the parameters of a neural network controller.56 By defining a task-based performance metric (e.g., locomotion speed, manipulation accuracy) as the loss function, gradient descent can be used to simultaneously optimize both the robot's body and its controller. This can lead to the discovery of novel, highly specialized, and high-performing robot designs that a human engineer might never conceive. A thesis that demonstrates the successful co-design of a complex, multi-material soft robot for a contact-rich task would be a landmark achievement.

#### **7.3 Digital Twins**

The concept of a digital twin—a high-fidelity, real-time virtual replica of a physical asset or system—is gaining significant traction across industries.78 Digital twins are used for monitoring, prediction, and what-if analysis.  
PhysGrad can serve as the core physics engine for a predictive digital twin. A case study could involve modeling a piece of industrial machinery or a manufacturing process.81 The differentiable capabilities of  
PhysGrad would be crucial for the twin's "self-calibration" loop. Real-world sensor data from the physical asset would be continuously streamed to the digital twin. The difference between the simulated state and the real-world state would form a loss signal. By backpropagating through the simulation, PhysGrad could continuously update its internal physical parameters to ensure the twin remains synchronized with and predictive of its real-world counterpart, a process known as online system identification.

## **Part IV: Synthesis and Strategic Roadmap**

This final part consolidates the preceding analysis into a concrete, actionable plan for a doctoral research program. It outlines a multi-year timeline with clear milestones and publication goals, and culminates in a concise articulation of the unique scientific contribution that a fully realized PhysGrad would represent.

### **Chapter 8: A Doctoral Research Roadmap**

A successful PhD requires not only a compelling research vision but also a structured plan for execution. The following three-year roadmap is proposed to guide the development of PhysGrad from its current state into a thesis-worthy scientific contribution.

* **Year 1: Foundational Enhancements and Benchmarking**  
  * **Objective:** To elevate PhysGrad to the state-of-the-art in terms of physical fidelity and numerical robustness, and to rigorously validate its performance.  
  * **Key Activities:**  
    1. **Implement Geometric Integrators:** Replace or augment the existing integrators with a high-order, momentum-preserving hierarchical symplectic integrator, drawing inspiration from the FROST architecture.38 This includes the development of kernels to compute force gradients.  
    2. **Integrate Advanced Material Models:** Implement a high-performance, GPU-accelerated Material Point Method (MPM) solver capable of simulating elastic and elastoplastic materials, leveraging advanced data structures (AoSoA) and kernel fusion (G2P2G).47  
    3. **Establish Benchmarking Suite:** Implement the comprehensive benchmark suite detailed in Chapter 2 for rigid body, contact, and fluid dynamics. Conduct a thorough performance and accuracy comparison of PhysGrad against at least two leading open-source engines (e.g., MuJoCo, SPlisHSPlasH).  
  * **Publication Target 1:** A conference paper (e.g., ACM SIGGRAPH, Robotics: Science and Systems) presenting the novel, high-performance GPU implementation of the hierarchical geometric integrator or the multi-material MPM solver, validated against the new benchmark suite.  
* **Year 2: The Differentiable Core**  
  * **Objective:** To architect and implement the end-to-end differentiable simulation pipeline, which forms the central technical innovation of the thesis.  
  * **Key Activities:**  
    1. **Develop Differentiable Kernels:** Manually derive and implement the adjoint (backward pass) for all core CUDA kernels, including force computation, SPH operators, and MPM transfer steps.  
    2. **Implement Differentiable Contact:** Replace the existing contact solver with a differentiable model, such as the single-level optimization approach.51 Implement its backward pass.  
    3. **Build the Hybrid API:** Develop the functional Python API and the custom pybind11 bindings that connect the C++/CUDA forward and backward passes to the PyTorch or JAX autograd system, as detailed in Chapter 5\.  
    4. **Validation on Inverse Problems:** Demonstrate the correctness and performance of the full differentiable pipeline on foundational inverse problems, such as system identification for simple systems.  
  * **Publication Target 2:** A top-tier machine learning or robotics conference paper (e.g., NeurIPS, ICML, ICLR, CoRL) detailing the novel hybrid C++/Python differentiable simulation framework. This paper would highlight the unique ability to combine hand-optimized CUDA performance with the flexibility of modern ML ecosystems.  
* **Year 3: High-Impact Applications and Thesis Compilation**  
  * **Objective:** To apply the fully differentiable PhysGrad framework to a grand-challenge problem that showcases its unique capabilities, and to synthesize all research into the final dissertation.  
  * **Key Activities:**  
    1. **Select and Execute a Premier Application:** Focus on one of the high-impact applications from Chapter 7, with robot co-design being a particularly novel and ambitious choice. Use the differentiable framework to co-optimize the morphology and control of a multi-material soft robot for a complex manipulation task.  
    2. **Analyze and Document Results:** Thoroughly analyze the results of the application, demonstrating how differentiability enabled the discovery of novel and high-performing solutions that would be infeasible with other methods.  
    3. **Thesis Writing:** Consolidate the research from all three years—the foundational algorithms, the differentiable architecture, and the capstone application—into a cohesive PhD dissertation.  
  * **Publication Target 3:** A high-impact journal article (e.g., Science Robotics, ACM Transactions on Graphics) or another top-tier conference paper presenting the results of the robot co-design or advanced manipulation experiments.

### **Chapter 9: Conclusion \- Defining a Unique Scientific Contribution**

The PhysGrad project, while currently a robust and well-engineered simulation framework, stands at a critical juncture. By pursuing the strategic roadmap outlined in this report, it can be transformed from a high-quality side project into a significant and novel scientific contribution suitable for a doctoral dissertation. The landscape of computational physics is rapidly evolving, driven by the dual forces of massively parallel GPU hardware and the gradient-based optimization paradigm of modern machine learning. The most impactful research emerges at the confluence of these trends.  
The proposed research program is designed to position PhysGrad precisely at this intersection. It moves beyond simply creating "another" physics engine by focusing on a core, unsolved challenge: the seamless and performant integration of low-level, hardware-specific code with high-level, differentiable programming environments. The final contribution is not just a piece of software, but a new methodology and a powerful demonstration of its potential.  
The thesis statement that emerges from this roadmap is as follows:  
**This thesis introduces PhysGrad, a novel computational framework that bridges the gap between high-performance, low-level simulation and high-level, gradient-based learning. By developing a new methodology for creating differentiable bindings for hand-optimized CUDA kernels, PhysGrad enables end-to-end optimization of complex, multi-material physical systems. This framework is built upon a foundation of state-of-the-art geometric integrators for long-term stability and advanced material point methods for simulating deformable continua. The power of this fully differentiable system is demonstrated by solving challenging inverse problems in robotics, culminating in the co-design of a robot's physical morphology and its neural network controller for a contact-rich manipulation task. This work establishes a new paradigm for performance-centric differentiable physics, enabling scientific discovery and engineering design at a scale and efficiency previously unattainable.**

#### **Referências citadas**

1. Introducing NVIDIA Isaac Gym: End-to-End Reinforcement Learning ..., acessado em setembro 29, 2025, [https://developer.nvidia.com/blog/introducing-isaac-gym-rl-for-robotics/](https://developer.nvidia.com/blog/introducing-isaac-gym-rl-for-robotics/)  
2. About Isaac Gym, acessado em setembro 29, 2025, [https://docs.robotsfan.com/isaacgym/about\_gym.html](https://docs.robotsfan.com/isaacgym/about_gym.html)  
3. Fast-Track Robot Learning in Simulation Using NVIDIA Isaac Lab, acessado em setembro 29, 2025, [https://developer.nvidia.com/blog/fast-track-robot-learning-in-simulation-using-nvidia-isaac-lab/](https://developer.nvidia.com/blog/fast-track-robot-learning-in-simulation-using-nvidia-isaac-lab/)  
4. Isaac Gym \- Preview Release \- NVIDIA Developer, acessado em setembro 29, 2025, [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)  
5. Unified framework for robot learning built on NVIDIA Isaac Sim \- GitHub, acessado em setembro 29, 2025, [https://github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)  
6. Welcome to Isaac Lab\!, acessado em setembro 29, 2025, [https://isaac-sim.github.io/IsaacLab/](https://isaac-sim.github.io/IsaacLab/)  
7. Isaac Lab: A GPU Accelerated Simulation Framework For Multi-Modal Robot Learning, acessado em setembro 29, 2025, [https://research.nvidia.com/publication/2025-09\_isaac-lab-gpu-accelerated-simulation-framework-multi-modal-robot-learning](https://research.nvidia.com/publication/2025-09_isaac-lab-gpu-accelerated-simulation-framework-multi-modal-robot-learning)  
8. GPU Simulation — physx 5.4.1 documentation, acessado em setembro 29, 2025, [https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html)  
9. JAX, M.D. \- NIPS, acessado em setembro 29, 2025, [https://papers.nips.cc/paper/2020/file/83d3d4b6c9579515e1679aca8cbc8033-Paper.pdf](https://papers.nips.cc/paper/2020/file/83d3d4b6c9579515e1679aca8cbc8033-Paper.pdf)  
10. \[1912.04232\] JAX, M.D.: A Framework for Differentiable Physics \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/abs/1912.04232](https://arxiv.org/abs/1912.04232)  
11. JAX, M.D. \- Machine Learning and the Physical Sciences, acessado em setembro 29, 2025, [https://ml4physicalsciences.github.io/2019/files/NeurIPS\_ML4PS\_2019\_86.pdf](https://ml4physicalsciences.github.io/2019/files/NeurIPS_ML4PS_2019_86.pdf)  
12. JAX, M.D. Cookbook.ipynb \- Colab \- Google, acessado em setembro 29, 2025, [https://colab.research.google.com/github/google/jax-md/blob/master/notebooks/jax\_md\_cookbook.ipynb](https://colab.research.google.com/github/google/jax-md/blob/master/notebooks/jax_md_cookbook.ipynb)  
13. Taichi Lang: High-performance Parallel Programming in Python, acessado em setembro 29, 2025, [https://www.taichi-lang.org/](https://www.taichi-lang.org/)  
14. Head First Taichi: A Beginner's Guide to High Performance Computing in Python \- Medium, acessado em setembro 29, 2025, [https://medium.com/parallel-programming-in-python/head-first-taichi-a-beginners-guide-to-high-performance-computing-in-python-be6afc5db93e](https://medium.com/parallel-programming-in-python/head-first-taichi-a-beginners-guide-to-high-performance-computing-in-python-be6afc5db93e)  
15. Taichi Lang: A high-performance parallel programming language embedded in Python : r/ProgrammingLanguages \- Reddit, acessado em setembro 29, 2025, [https://www.reddit.com/r/ProgrammingLanguages/comments/z7pejv/taichi\_lang\_a\_highperformance\_parallel/](https://www.reddit.com/r/ProgrammingLanguages/comments/z7pejv/taichi_lang_a_highperformance_parallel/)  
16. Taichi: A Language for High-Performance Computation on Spatially Sparse Data Structures \- Yuanming Hu, acessado em setembro 29, 2025, [https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf)  
17. MuJoCo \- Wikipedia, acessado em setembro 29, 2025, [https://en.wikipedia.org/wiki/MuJoCo](https://en.wikipedia.org/wiki/MuJoCo)  
18. MuJoCo — Advanced Physics Simulation, acessado em setembro 29, 2025, [https://mujoco.org/](https://mujoco.org/)  
19. MuJoCo Documentation: Overview, acessado em setembro 29, 2025, [https://mujoco.readthedocs.io/](https://mujoco.readthedocs.io/)  
20. MuJoCo Documentation: Overview, acessado em setembro 29, 2025, [https://mujoco.readthedocs.io/en/latest/overview.html](https://mujoco.readthedocs.io/en/latest/overview.html)  
21. Announcing Newton, an Open-Source Physics Engine for Robotics Simulation | NVIDIA Technical Blog, acessado em setembro 29, 2025, [https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/)  
22. google-deepmind/mujoco: Multi-Joint dynamics with Contact. A general purpose physics simulator. \- GitHub, acessado em setembro 29, 2025, [https://github.com/google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)  
23. SPlisHSPlasH \- Computer Animation, acessado em setembro 29, 2025, [https://animation.rwth-aachen.de/software/splishsplash/](https://animation.rwth-aachen.de/software/splishsplash/)  
24. SPlisHSPlasH \- SPH simulation of fluids and solids, acessado em setembro 29, 2025, [https://splishsplash.physics-simulation.org/](https://splishsplash.physics-simulation.org/)  
25. About SPlisHSPlasH — SPlisHSPlasH 2.15.1 documentation, acessado em setembro 29, 2025, [https://splishsplash.readthedocs.io/en/latest/about.html](https://splishsplash.readthedocs.io/en/latest/about.html)  
26. SPlisHSPlasH is an open-source library for the physically-based simulation of fluids. \- GitHub, acessado em setembro 29, 2025, [https://github.com/InteractiveComputerGraphics/SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)  
27. JAX MD: A Framework for Differentiable Physics \- College of Science & Engineering \- University of Minnesota, acessado em setembro 29, 2025, [https://cse.umn.edu/ctc/events/jax-md-framework-differentiable-physics](https://cse.umn.edu/ctc/events/jax-md-framework-differentiable-physics)  
28. Benchmarking Rigid Body Contact Models \- Proceedings of Machine Learning Research, acessado em setembro 29, 2025, [https://proceedings.mlr.press/v211/guo23b/guo23b.pdf](https://proceedings.mlr.press/v211/guo23b/guo23b.pdf)  
29. GPU-Accelerated Simulation, acessado em setembro 29, 2025, [https://phys-sim-book.github.io/lec4.6-gpu\_accel.html](https://phys-sim-book.github.io/lec4.6-gpu_accel.html)  
30. Test 01 | SPHERIC, acessado em setembro 29, 2025, [https://www.spheric-sph.org/tests/test-01](https://www.spheric-sph.org/tests/test-01)  
31. SimBenchmark | Physics engine benchmark for robotics applications: RaiSim vs. Bullet vs. ODE vs. MuJoCo vs. DartSim \- GitHub Pages, acessado em setembro 29, 2025, [https://leggedrobotics.github.io/SimBenchmark/](https://leggedrobotics.github.io/SimBenchmark/)  
32. Intersection-free Rigid Body Dynamics, acessado em setembro 29, 2025, [https://web.uvic.ca/\~teseo/publications/rigid-ipc/downloads/rigid\_ipc\_paper\_350ppi.pdf](https://web.uvic.ca/~teseo/publications/rigid-ipc/downloads/rigid_ipc_paper_350ppi.pdf)  
33. SPH study of the evolution of water–water interfaces in dam break flows \- White Rose Research Online, acessado em setembro 29, 2025, [https://eprints.whiterose.ac.uk/id/eprint/86310/3/WRRO\_86310.pdf](https://eprints.whiterose.ac.uk/id/eprint/86310/3/WRRO_86310.pdf)  
34. A New Experimental Study and SPH Comparison for the Sequential Dam-Break Problem, acessado em setembro 29, 2025, [https://www.researchgate.net/publication/346851765\_A\_New\_Experimental\_Study\_and\_SPH\_Comparison\_for\_the\_Sequential\_Dam-Break\_Problem](https://www.researchgate.net/publication/346851765_A_New_Experimental_Study_and_SPH_Comparison_for_the_Sequential_Dam-Break_Problem)  
35. A New Experimental Study and SPH Comparison for the Sequential Dam-Break Problem, acessado em setembro 29, 2025, [https://www.mdpi.com/2077-1312/8/11/905](https://www.mdpi.com/2077-1312/8/11/905)  
36. Validation of Dam-Break Problem over Dry Bed using SPH \- ResearchGate, acessado em setembro 29, 2025, [https://www.researchgate.net/profile/Selahattin-Kocaman/publication/322217709\_Validation\_of\_Dam-Break\_Problem\_over\_Dry\_Bed\_using\_SPH/links/5a4f71c5a6fdcc7b3cdb4e11/Validation-of-Dam-Break-Problem-over-Dry-Bed-using-SPH.pdf](https://www.researchgate.net/profile/Selahattin-Kocaman/publication/322217709_Validation_of_Dam-Break_Problem_over_Dry_Bed_using_SPH/links/5a4f71c5a6fdcc7b3cdb4e11/Validation-of-Dam-Break-Problem-over-Dry-Bed-using-SPH.pdf)  
37. Test 19 | SPHERIC, acessado em setembro 29, 2025, [https://www.spheric-sph.org/tests/test-19](https://www.spheric-sph.org/tests/test-19)  
38. frost: a momentum-conserving CUDA implementation of a hierarchical fourth-order forward symplectic integrator | Monthly Notices of the Royal Astronomical Society | Oxford Academic, acessado em setembro 29, 2025, [https://academic.oup.com/mnras/article/502/4/5546/6081060](https://academic.oup.com/mnras/article/502/4/5546/6081060)  
39. Variational Integrators \- Matthew West, acessado em setembro 29, 2025, [https://lagrange.mechse.illinois.edu/var\_int/](https://lagrange.mechse.illinois.edu/var_int/)  
40. a momentum-conserving CUDA implementation of a \- hierarchical fourth-order forward symplectic integrator \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/pdf/2011.14984](https://arxiv.org/pdf/2011.14984)  
41. \[2011.14984\] FROST: a momentum-conserving CUDA implementation of a hierarchical fourth-order forward symplectic integrator \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/abs/2011.14984](https://arxiv.org/abs/2011.14984)  
42. a momentum-conserving CUDA implementation ... \- Oxford Academic, acessado em setembro 29, 2025, [https://academic.oup.com/mnras/article-pdf/502/4/5546/49542947/stab057.pdf](https://academic.oup.com/mnras/article-pdf/502/4/5546/49542947/stab057.pdf)  
43. acessado em dezembro 31, 1969, [https://academic.oup.com/mnras/article-pdf/502/4/5546/36445337/stab311.pdf](https://academic.oup.com/mnras/article-pdf/502/4/5546/36445337/stab311.pdf)  
44. Variational Integrators, acessado em setembro 29, 2025, [https://mathweb.ucsd.edu/\~mleok/pdf/Le2015\_EACM\_variational\_integrators.pdf](https://mathweb.ucsd.edu/~mleok/pdf/Le2015_EACM_variational_integrators.pdf)  
45. AN OVERVIEW OF VARIATIONAL INTEGRATORS \- Graduate Degree in Control \+ Dynamical Systems, acessado em setembro 29, 2025, [https://www.cds.caltech.edu/\~marsden/bib/2004/02-LeMaOrWe2004a/LeMaOrWe2004a.pdf](https://www.cds.caltech.edu/~marsden/bib/2004/02-LeMaOrWe2004a/LeMaOrWe2004a.pdf)  
46. Variational integrators for non-autonomous systems with applications to stabilization of multi-agent formations \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/pdf/2202.01471](https://arxiv.org/pdf/2202.01471)  
47. \[SIGGRAPH20\] A Massively Parallel and Scalable Multi-GPU ..., acessado em setembro 29, 2025, [https://pku.ai/publication/mpmgpu2020siggraph/](https://pku.ai/publication/mpmgpu2020siggraph/)  
48. GPU Optimization of Material Point Methods \- University of Wisconsin–Madison, acessado em setembro 29, 2025, [https://pages.cs.wisc.edu/\~sifakis/papers/GPU\_MPM.pdf](https://pages.cs.wisc.edu/~sifakis/papers/GPU_MPM.pdf)  
49. SIGGRAPH2020-MultiGPU \- Google Sites, acessado em setembro 29, 2025, [https://sites.google.com/view/siggraph2020-multigpu](https://sites.google.com/view/siggraph2020-multigpu)  
50. A Massively Parallel and Scalable Multi-GPU Material ... \- Yixin Zhu, acessado em setembro 29, 2025, [https://yzhu.io/publication/mpmgpu2020siggraph/paper.pdf](https://yzhu.io/publication/mpmgpu2020siggraph/paper.pdf)  
51. \[2212.06764\] Single-Level Differentiable Contact Simulation \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/abs/2212.06764](https://arxiv.org/abs/2212.06764)  
52. Single-Level Differentiable Contact Simulation, acessado em setembro 29, 2025, [https://arxiv.org/pdf/2212.06764](https://arxiv.org/pdf/2212.06764)  
53. A Differentiable Physics Engine for Deep Learning in Robotics \- PMC, acessado em setembro 29, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6416213/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6416213/)  
54. Differentiable Physics and Stable Modes for Tool-Use and Manipulation Planning \- Robotics, acessado em setembro 29, 2025, [https://www.roboticsproceedings.org/rss14/p44.pdf](https://www.roboticsproceedings.org/rss14/p44.pdf)  
55. Fast and Feature-Complete Differentiable Physics for Articulated Rigid Bodies with Contact \- NSF PAR, acessado em setembro 29, 2025, [https://par.nsf.gov/servlets/purl/10298121](https://par.nsf.gov/servlets/purl/10298121)  
56. A Differentiable Physics Engine for Deep Learning in Robotics \- Frontiers, acessado em setembro 29, 2025, [https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2019.00006/full](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2019.00006/full)  
57. Physics-informed neural networks for physiological signal processing and modeling: a narrative review \- PMC \- PubMed Central, acessado em setembro 29, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12308510/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12308510/)  
58. Understanding Physics-Informed Neural Networks: Techniques, Applications, Trends, and Challenges \- MDPI, acessado em setembro 29, 2025, [https://www.mdpi.com/2673-2688/5/3/74](https://www.mdpi.com/2673-2688/5/3/74)  
59. Physics-Informed Neural Networks: A Review of Methodological Evolution, Theoretical Foundations, and Interdisciplinary Frontiers Toward Next-Generation Scientific Computing \- MDPI, acessado em setembro 29, 2025, [https://www.mdpi.com/2076-3417/15/14/8092](https://www.mdpi.com/2076-3417/15/14/8092)  
60. Essential Review Papers on Physics-Informed Neural Networks: A Curated Guide for Practitioners | Towards Data Science, acessado em setembro 29, 2025, [https://towardsdatascience.com/essential-review-papers-on-physics-informed-neural-networks-a-curated-guide-for-practitioners/](https://towardsdatascience.com/essential-review-papers-on-physics-informed-neural-networks-a-curated-guide-for-practitioners/)  
61. \[2408.16806\] Physics-Informed Neural Networks and Extensions \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/abs/2408.16806](https://arxiv.org/abs/2408.16806)  
62. \[2411.02411\] SK-PINN: Accelerated physics-informed deep learning by smoothing kernel gradients \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/abs/2411.02411](https://arxiv.org/abs/2411.02411)  
63. NVIDIA PhysicsNeMo, acessado em setembro 29, 2025, [https://developer.nvidia.com/physicsnemo](https://developer.nvidia.com/physicsnemo)  
64. AI Physics Powered By NVIDIA \- Rescale, acessado em setembro 29, 2025, [https://rescale.com/platform/ai-physics/](https://rescale.com/platform/ai-physics/)  
65. Efficient n-body simulations using physics informed graph neural networks \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/abs/2504.01169](https://arxiv.org/abs/2504.01169)  
66. Efficient n-body simulations using physics informed graph neural networks \- OpenReview, acessado em setembro 29, 2025, [https://openreview.net/pdf?id=kpVTdAlmyL](https://openreview.net/pdf?id=kpVTdAlmyL)  
67. Efficient n-body simulations using physics informed graph neural networks \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/html/2504.01169v1](https://arxiv.org/html/2504.01169v1)  
68. NVIDIA AI-Physics Framework for Accelerating Computational Engineering with Emulation of AI \- YouTube, acessado em setembro 29, 2025, [https://www.youtube.com/watch?v=pZm-Hs2Wstc](https://www.youtube.com/watch?v=pZm-Hs2Wstc)  
69. Differentiable Physics-based System Identification for Robotic Manipulation of Elastoplastic Materials \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/html/2411.00554v1](https://arxiv.org/html/2411.00554v1)  
70. Diff-Transfer: Model-based Robotic Manipulation Skill Transfer via Differentiable Physics Simulation | OpenReview, acessado em setembro 29, 2025, [https://openreview.net/forum?id=EODzbQ2Gy4¬eId=mmtfG61xd3](https://openreview.net/forum?id=EODzbQ2Gy4&noteId=mmtfG61xd3)  
71. \[2310.04930\] Diff-Transfer: Model-based Robotic Manipulation Skill Transfer via Differentiable Physics Simulation \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/abs/2310.04930](https://arxiv.org/abs/2310.04930)  
72. acessado em dezembro 31, 1969, [https://openreview.net/pdf?id=EODzbQ2Gy4](https://openreview.net/pdf?id=EODzbQ2Gy4)  
73. FMB: A Functional Manipulation Benchmark for Generalizable Robotic Learning, acessado em setembro 29, 2025, [https://functional-manipulation-benchmark.github.io/](https://functional-manipulation-benchmark.github.io/)  
74. PerAct 2 : Benchmarking and Learning for Robotic Bimanual Manipulation Tasks \- arXiv, acessado em setembro 29, 2025, [https://arxiv.org/html/2407.00278v1](https://arxiv.org/html/2407.00278v1)  
75. Benchmarking In-Hand Manipulation \- LL4MA lab, acessado em setembro 29, 2025, [https://robot-learning.cs.utah.edu/project/benchmarking\_in\_hand\_manipulation](https://robot-learning.cs.utah.edu/project/benchmarking_in_hand_manipulation)  
76. Differentiable Simulation of Soft Multi-body Systems, acessado em setembro 29, 2025, [https://proceedings.neurips.cc/paper/2021/file/8e296a067a37563370ded05f5a3bf3ec-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/8e296a067a37563370ded05f5a3bf3ec-Paper.pdf)  
77. Neural Approaches to Robot Co-Optimization \- TTIC, acessado em setembro 29, 2025, [https://ttic.edu/ripl/research/02\_research\_joint\_optimization.html](https://ttic.edu/ripl/research/02_research_joint_optimization.html)  
78. Digital twins at work: 9 examples | SAP, acessado em setembro 29, 2025, [https://www.sap.com/blogs/digital-twins-at-work](https://www.sap.com/blogs/digital-twins-at-work)  
79. Evaluation of Digital Twin Modeling and Simulation \- Sandia National Laboratories, acessado em setembro 29, 2025, [https://www.sandia.gov/app/uploads/sites/273/2024/11/SAND\_Digital\_Twins\_Final.pdf](https://www.sandia.gov/app/uploads/sites/273/2024/11/SAND_Digital_Twins_Final.pdf)  
80. Digital Twin for Flexible Manufacturing Systems and Optimization Through Simulation: A Case Study \- MDPI, acessado em setembro 29, 2025, [https://www.mdpi.com/2075-1702/12/11/785](https://www.mdpi.com/2075-1702/12/11/785)  
81. A case-study in the introduction of a digital twin in a large-scale smart manufacturing facility, acessado em setembro 29, 2025, [https://www.researchgate.net/publication/347072976\_A\_case-study\_in\_the\_introduction\_of\_a\_digital\_twin\_in\_a\_large-scale\_smart\_manufacturing\_facility](https://www.researchgate.net/publication/347072976_A_case-study_in_the_introduction_of_a_digital_twin_in_a_large-scale_smart_manufacturing_facility)