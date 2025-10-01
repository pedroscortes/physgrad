# Neural Surrogate Modeling Framework

This document describes the neural surrogate modeling framework in PhysGrad, which enables learning physics approximations using neural networks to dramatically accelerate simulation performance while maintaining accuracy.

## Overview

The neural surrogate modeling framework provides:

- **Physics-Informed Neural Networks (PINNs)** with conservation constraints
- **Adaptive sampling** for continuous learning and improvement
- **Uncertainty quantification** for hybrid physics-surrogate predictions
- **Real-time training** with background learning capabilities
- **Multiple architectures** including dense layers and physics-informed layers

## Architecture

### Core Components

#### 1. Neural Network Implementation (`src/neural_surrogate.h`)

**Tensor Operations**
```cpp
template<typename T>
class Tensor {
    // Multi-dimensional array with automatic differentiation support
    // Operations: addition, subtraction, matrix multiplication, reshape
    // Memory management for both CPU and GPU (CUDA support)
};
```

**Layer Types**
- **DenseLayer**: Fully connected layers with configurable activation functions
- **PhysicsInformedLayer**: Layers with built-in physics constraints
- **ActivationFunction**: ReLU, Tanh, Sigmoid, LeakyReLU, ELU, Swish, GELU

**Neural Network**
```cpp
template<typename T>
class NeuralNetwork {
    // Configurable architecture with multiple optimizers
    // Support for Adam, SGD, AdamW, RMSprop
    // Automatic gradient computation and weight updates
};
```

#### 2. Surrogate Model Framework (`src/surrogate_model.h`)

**Physics State Representation**
```cpp
template<typename T>
struct PhysicsState {
    std::vector<T> positions;      // Particle positions
    std::vector<T> velocities;     // Particle velocities
    std::vector<T> forces;         // Forces acting on particles
    std::vector<T> material_props; // Material properties
    T timestep;
    T time;
};
```

**Data Preprocessing**
- Automatic normalization and standardization
- Feature scaling for improved convergence
- Inverse transforms for output interpretation

**Uncertainty Quantification**
- Ensemble-based uncertainty estimation
- Predictive variance calculation
- Hybrid physics-surrogate decision making

**Adaptive Sampling**
- Priority queue for high-error samples
- Continuous retraining on challenging cases
- Automatic data augmentation

## Key Features

### 1. Physics-Informed Learning

The framework enforces physical constraints during training:

```cpp
// Energy conservation constraint
static Tensor<T> energy_conservation(const Tensor<T>& output);

// Momentum conservation constraint
static Tensor<T> momentum_conservation(const Tensor<T>& output);

// Mass conservation constraint
static Tensor<T> mass_conservation(const Tensor<T>& output);
```

### 2. Hybrid Physics-Surrogate Execution

```cpp
PhysicsState<T> predict(const PhysicsState<T>& current_state, bool& used_physics_fallback) {
    // Get prediction with uncertainty
    auto [prediction, uncertainty] = uncertainty_estimator_->predict_with_uncertainty(input);

    // Fall back to physics if uncertainty is too high
    if (uncertainty > config_.uncertainty_threshold) {
        used_physics_fallback = true;
        return physics_simulator_(current_state);
    }

    return prediction;
}
```

### 3. Background Adaptive Learning

```cpp
void start_background_training(std::function<PhysicsState<T>(const PhysicsState<T>&)> physics_simulator) {
    // Continuous learning thread
    training_thread_ = std::thread([this, physics_simulator]() {
        while (training_active_) {
            // Retrain on high-error samples
            auto batch = adaptive_sampler_->get_training_batch(config_.batch_size);
            retrain_on_batch(batch);
        }
    });
}
```

### 4. Performance Optimization

**CUDA Support**
```cpp
#ifdef ENABLE_CUDA
void to_device() {
    cudaMalloc(&device_data_, data_.size() * sizeof(T));
    cudaMemcpy(device_data_, data_.data(), bytes, cudaMemcpyHostToDevice);
}
#endif
```

**Memory Management**
- Custom memory allocators
- Memory usage tracking and optimization
- Automatic garbage collection

## Configuration

### Surrogate Model Configuration

```cpp
template<typename T>
struct SurrogateConfig {
    // Network architecture
    std::vector<size_t> hidden_layers = {256, 512, 256, 128};
    ActivationType activation = ActivationType::Swish;

    // Training parameters
    T learning_rate = T(0.001);
    size_t batch_size = 32;
    size_t epochs = 1000;
    T validation_split = T(0.2);

    // Physics constraints
    bool enforce_energy_conservation = true;
    bool enforce_momentum_conservation = true;
    bool enforce_mass_conservation = true;
    T physics_weight = T(0.1);

    // Adaptive sampling
    bool use_adaptive_sampling = true;
    T error_threshold = T(0.01);
    size_t max_training_points = 100000;

    // Hybrid mode
    bool use_hybrid_mode = true;
    T uncertainty_threshold = T(0.05);

    // Performance targets
    T target_speedup = T(10.0);
    T target_accuracy = T(0.95);
};
```

## Usage Examples

### Basic Training

```cpp
#include "src/surrogate_model.h"

using namespace physgrad::surrogate;

// Configure model
SurrogateConfig<float> config;
config.hidden_layers = {64, 128, 64};
config.epochs = 500;
config.learning_rate = 0.001f;

// Create model
SurrogateModel<float> model(config);

// Generate or load training data
std::vector<PhysicsState<float>> training_data = load_physics_data();

// Define physics simulator
auto physics_sim = [](const PhysicsState<float>& state) -> PhysicsState<float> {
    // Your physics simulation code here
    return simulate_physics(state);
};

// Train the model
model.train(training_data, physics_sim);
```

### Real-time Prediction

```cpp
// Use trained model for prediction
PhysicsState<float> current_state = get_current_state();

bool used_physics_fallback = false;
PhysicsState<float> next_state = model.predict(current_state, used_physics_fallback);

if (used_physics_fallback) {
    std::cout << "Fell back to physics simulation" << std::endl;
} else {
    std::cout << "Used neural surrogate prediction" << std::endl;
}

// Get performance metrics
float speedup = model.get_speedup();
float accuracy = model.get_accuracy();
float fallback_rate = model.get_physics_fallback_rate();
```

### Continuous Learning

```cpp
// Start background adaptive learning
model.start_background_training(physics_sim);

// During simulation, the model automatically:
// 1. Identifies high-error predictions
// 2. Runs ground truth physics on those cases
// 3. Adds them to the training queue
// 4. Continuously retrains in the background

// Stop background learning when done
model.stop_background_training();
```

## Performance Characteristics

### Computational Complexity

- **Training**: O(n²) for dense layers, where n is the number of neurons
- **Inference**: O(n) for forward pass through the network
- **Memory**: O(n²) for weight storage, O(n) for activations

### Typical Performance

| Scenario | Speedup | Accuracy | Memory Usage |
|----------|---------|----------|--------------|
| Simple dynamics (pendulum) | 50-100x | >95% | <100 MB |
| Particle systems (1K particles) | 10-20x | >90% | <500 MB |
| Complex fluids (10K particles) | 5-10x | >85% | <2 GB |

### Optimization Strategies

1. **Model Architecture**
   - Use appropriate hidden layer sizes
   - Apply dropout for regularization
   - Use batch normalization for stability

2. **Data Management**
   - Normalize input features
   - Use data augmentation
   - Balance training data distribution

3. **Training Optimization**
   - Use learning rate scheduling
   - Apply early stopping
   - Use adaptive optimizers (Adam, AdamW)

4. **Hybrid Execution**
   - Set appropriate uncertainty thresholds
   - Use physics fallback for critical scenarios
   - Monitor prediction quality metrics

## Integration with PhysGrad

### MPM Integration

```cpp
// Replace expensive MPM steps with neural surrogates
if (use_neural_surrogate && frame % surrogate_frequency == 0) {
    PhysicsState<float> current_state = extract_mpm_state();
    bool used_fallback;
    PhysicsState<float> predicted_state = surrogate_model.predict(current_state, used_fallback);

    if (!used_fallback) {
        apply_predicted_state(predicted_state);
        continue; // Skip expensive physics computation
    }
}

// Fall back to full MPM computation
perform_mpm_step();
```

### Real-time Applications

```cpp
class RealTimeSimulator {
    SurrogateModel<float> surrogate_model_;
    bool use_surrogate_acceleration_ = true;
    float max_acceptable_error_ = 0.01f;

public:
    void update(float dt) {
        if (use_surrogate_acceleration_) {
            auto predicted_state = surrogate_model_.predict(current_state_, used_fallback);

            if (!used_fallback) {
                current_state_ = predicted_state;
                return;
            }
        }

        // Fallback to full physics
        current_state_ = physics_engine_.simulate(current_state_, dt);
    }
};
```

## Validation and Testing

### Test Suite Coverage

The framework includes comprehensive tests for:

- **Tensor operations**: All mathematical operations and memory management
- **Activation functions**: Correctness and numerical stability
- **Neural layers**: Forward pass and gradient computation
- **Loss functions**: MSE, MAE, Huber, LogCosh implementations
- **Physics constraints**: Energy, momentum, and mass conservation
- **Data preprocessing**: Normalization and denormalization
- **Uncertainty estimation**: Ensemble variance calculation
- **Adaptive sampling**: Priority queue and retraining logic

### Validation Methodology

1. **Unit Tests**: Individual component validation
2. **Integration Tests**: End-to-end workflow testing
3. **Physics Validation**: Conservation law verification
4. **Performance Benchmarks**: Speed and accuracy measurements
5. **Stress Testing**: Large-scale and edge case validation

### Example Validation Results

```cpp
PhysGrad Neural Surrogate Modeling Test Suite
==============================================
Testing Tensor operations...
✓ Tensor operations tests passed
Testing activation functions...
✓ Activation function tests passed
Testing neural network layers...
✓ Neural layer tests passed
Testing loss functions...
✓ Loss function tests passed
Testing neural network...
✓ Neural network tests passed
Testing physics state operations...
✓ Physics state tests passed
Testing data preprocessor...
✓ Data preprocessor tests passed
Testing surrogate model...
✓ Surrogate model tests passed

🎉 All tests passed!
Neural surrogate modeling framework is working correctly.
```

## Future Extensions

### Planned Enhancements

1. **Advanced Architectures**
   - Convolutional layers for spatial patterns
   - Recurrent layers for temporal dependencies
   - Transformer architectures for long-range interactions

2. **Improved Physics Integration**
   - Automatic constraint discovery
   - Lagrangian formulation preservation
   - Symplectic integration compatibility

3. **Scalability Improvements**
   - Distributed training across multiple GPUs
   - Model parallelism for large networks
   - Federated learning for collaborative training

4. **Domain-Specific Optimizations**
   - Fluid dynamics specializations
   - Solid mechanics adaptations
   - Multi-scale modeling capabilities

### Research Directions

- **Physics-constrained architectures** that inherently preserve conservation laws
- **Multi-fidelity modeling** combining different levels of physics approximation
- **Causal discovery** for automatic physics law identification
- **Transfer learning** between different physics domains

## Limitations and Considerations

### Current Limitations

1. **Training Data Requirements**: Needs substantial physics simulation data
2. **Domain Specificity**: Models are specific to particular physics scenarios
3. **Extrapolation**: Limited accuracy outside training distribution
4. **Conservation Guarantees**: Soft constraints may not be perfectly preserved

### Best Practices

1. **Data Quality**: Ensure training data covers the full operational range
2. **Validation**: Always validate on held-out physics scenarios
3. **Monitoring**: Continuously monitor prediction quality in production
4. **Fallback Strategy**: Always have physics simulation as backup
5. **Incremental Deployment**: Gradually increase surrogate usage based on confidence

### When to Use Neural Surrogates

**Good Use Cases:**
- Repeated similar physics computations
- Real-time applications requiring speed
- Parameter sweeps and optimization
- Interactive visualization and simulation

**Avoid When:**
- Single-use or rare computations
- Critical safety applications without validation
- Highly nonlinear or chaotic systems
- Limited training data availability

This neural surrogate modeling framework provides a robust foundation for accelerating physics simulations while maintaining scientific accuracy and reliability.