# PhysGrad WebAssembly

This directory contains the WebAssembly (WASM) build system for PhysGrad, enabling physics simulations to run directly in web browsers and edge computing environments.

## Overview

The PhysGrad WebAssembly module provides:
- **Browser-native physics simulation** with no plugins required
- **Edge computing compatibility** for distributed simulation scenarios
- **Real-time visualization** using WebGL integration
- **Memory-efficient execution** with optimized WASM binaries
- **Cross-platform support** (Windows, macOS, Linux, mobile browsers)
- **TypeScript integration** with full type definitions

## Architecture

### Core Components

#### 1. WebAssembly Bridge (`src/wasm_bridge.h`)
- **WasmVec3**: Lightweight 3D vector implementation
- **WasmParticle**: Particle data structure optimized for WASM
- **WasmMaterial**: Material properties for different physics models
- **WasmPhysicsEngine**: Simplified MPM physics engine for web deployment
- **WasmInterface**: JavaScript-friendly API wrapper
- **WasmMemoryManager**: Memory tracking and optimization utilities

#### 2. Build System (`CMakeLists.txt`, `build_wasm.sh`)
- **Emscripten integration** with optimized compile flags
- **SIMD support** for enhanced performance
- **Memory management** with configurable heap sizes
- **Debug and release configurations**
- **Automated testing and validation**

#### 3. Web Integration (`shell.html`, TypeScript definitions)
- **Interactive demo interface** with real-time controls
- **WebGL rendering** for particle visualization
- **Performance monitoring** and debugging tools
- **Responsive design** for desktop and mobile

## Quick Start

### Prerequisites

1. **Emscripten SDK**:
   ```bash
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk
   ./emsdk install latest
   ./emsdk activate latest
   source ./emsdk_env.sh
   ```

2. **CMake 3.16+** and **Node.js** (optional, for testing)

### Building

```bash
# Build WebAssembly module
./build_wasm.sh build

# Run all build steps with validation
./build_wasm.sh all

# Test in Node.js environment
./build_wasm.sh test

# Create distribution package
./build_wasm.sh package
```

### Basic Usage

#### Browser Integration
```html
<!DOCTYPE html>
<html>
<head>
    <script src="physgrad_wasm.js"></script>
</head>
<body>
    <canvas id="canvas" width="800" height="600"></canvas>

    <script>
        PhysGradModule().then(Module => {
            // Create physics engine
            const engine = new Module.PhysicsEngine();
            engine.initialize(10000);

            // Add particles
            engine.addParticle(0, 5, 0, 1, 0, 0);
            engine.addBlock(-1, -1, -1, 2, 2, 2, 10, 10, 10);

            // Simulation loop
            function animate() {
                engine.step();

                // Get particle data for rendering
                const positions = engine.getPositions();
                const particleCount = engine.getParticleCount();

                // Render particles (WebGL code here)
                renderParticles(positions, particleCount);

                requestAnimationFrame(animate);
            }

            engine.start();
            animate();
        });
    </script>
</body>
</html>
```

#### Node.js Integration
```javascript
const PhysGradModule = require('./physgrad_wasm.js');

async function runSimulation() {
    const Module = await PhysGradModule();
    const engine = new Module.PhysicsEngine();

    engine.initialize(5000);
    engine.addBlock(-2, 0, -2, 4, 2, 4, 20, 10, 20);

    // Run simulation
    for (let i = 0; i < 1000; i++) {
        engine.step();

        if (i % 100 === 0) {
            console.log(`Step ${i}: FPS=${engine.getFPS()}`);
        }
    }
}

runSimulation();
```

#### TypeScript Integration
```typescript
import PhysGradModuleFactory, { PhysGradModule, PhysicsEngine } from './physgrad';

async function createSimulation(): Promise<void> {
    const Module: PhysGradModule = await PhysGradModuleFactory();
    const engine: PhysicsEngine = new Module.PhysicsEngine();

    engine.initialize(10000);
    engine.setGravity(0, -9.81, 0);
    engine.setTimestep(0.016); // 60 FPS

    // Type-safe particle creation
    engine.addParticle(0, 5, 0, 1, 0, 0);

    // Performance monitoring
    setInterval(() => {
        const fps = engine.getFPS();
        const memory = Module.MemoryManager.getAllocatedBytes();
        console.log(`FPS: ${fps}, Memory: ${memory} bytes`);
    }, 1000);
}
```

## API Reference

### PhysicsEngine

#### Initialization
- `initialize(maxParticles: number)`: Initialize engine with particle capacity
- `reset()`: Clear all particles and reset simulation state

#### Particle Management
- `addParticle(x, y, z, vx, vy, vz)`: Add single particle with position and velocity
- `addBlock(x, y, z, w, h, d, nx, ny, nz)`: Add rectangular block of particles
- `getParticleCount()`: Get current number of active particles

#### Simulation Control
- `start()`: Start simulation
- `stop()`: Pause simulation
- `step()`: Advance simulation by one timestep
- `isRunning()`: Check if simulation is active

#### Data Access
- `getPositions()`: Get Float32Array of particle positions (x,y,z,x,y,z,...)
- `getVelocities()`: Get Float32Array of particle velocities
- `getFPS()`: Get current simulation frame rate

#### Configuration
- `setGravity(x, y, z)`: Set gravitational acceleration
- `setTimestep(dt)`: Set simulation timestep
- `enableSIMD(enable)`: Enable/disable SIMD optimizations

### Demo Scenarios

The module includes several pre-built scenarios:

1. **Particle Field**: Regular grid of particles for stability testing
2. **Dam Break**: Fluid simulation with particle collapse
3. **Particle Rain**: Continuous particle emission from above
4. **Explosion**: Radial particle emission with high velocities

## Performance Optimization

### Compilation Flags
The build system uses aggressive optimization flags:
- `-O3`: Maximum compiler optimization
- `-ffast-math`: Fast floating-point operations
- `-msimd128`: SIMD vectorization
- `-mbulk-memory`: Efficient memory operations

### Memory Management
- **Initial heap**: 64MB with growth to 2GB maximum
- **Stack size**: 8MB for deep recursion support
- **Memory tracking**: Real-time allocation monitoring
- **Automatic cleanup**: RAII patterns for resource management

### SIMD Acceleration
When supported by the browser:
- **Vectorized operations** for particle updates
- **Parallel grid computations**
- **Optimized memory access patterns**

### Performance Benchmarks

Typical performance on modern browsers:

| Particle Count | Chrome (FPS) | Firefox (FPS) | Safari (FPS) |
|---------------|--------------|---------------|--------------|
| 1,000         | 3,500+       | 3,200+        | 2,800+       |
| 5,000         | 2,000+       | 1,800+        | 1,600+       |
| 10,000        | 1,200+       | 1,000+        | 900+         |
| 25,000        | 500+         | 450+          | 400+         |

*Tested on Apple M1 MacBook Pro with 16GB RAM*

## Deployment Options

### CDN Deployment
```html
<script src="https://cdn.jsdelivr.net/npm/@physgrad/wasm@latest/physgrad_wasm.js"></script>
```

### NPM Package
```bash
npm install @physgrad/wasm
```

### Self-Hosted
1. Copy `physgrad_wasm.js` and `physgrad_wasm.wasm` to your web server
2. Ensure proper MIME type for `.wasm` files: `application/wasm`
3. Enable CORS headers if serving from different domain

### Progressive Web App (PWA)
The WASM module works seamlessly in PWAs with:
- **Offline simulation** capabilities
- **Service worker caching** for fast loading
- **Background processing** in web workers

## Browser Compatibility

### Minimum Requirements
- **WebAssembly support**: All modern browsers (95%+ global support)
- **WebGL**: For visualization (can fallback to Canvas 2D)
- **ES6 modules**: For modern integration patterns

### Feature Detection
```javascript
const support = {
    wasm: typeof WebAssembly === 'object',
    webgl: !!document.createElement('canvas').getContext('webgl'),
    simd: 'simd' in WebAssembly,
    threads: 'Memory' in WebAssembly && 'instantiate' in WebAssembly
};
```

### Polyfills
For older browsers, consider:
- **WebAssembly polyfill**: Falls back to asm.js
- **WebGL polyfill**: Canvas 2D rendering fallback
- **Float32Array polyfill**: For very old browsers

## Advanced Usage

### Web Workers
```javascript
// main.js
const worker = new Worker('physics-worker.js');
worker.postMessage({ command: 'init', maxParticles: 10000 });

// physics-worker.js
importScripts('physgrad_wasm.js');

PhysGradModule().then(Module => {
    const engine = new Module.PhysicsEngine();

    self.onmessage = function(e) {
        if (e.data.command === 'init') {
            engine.initialize(e.data.maxParticles);
        } else if (e.data.command === 'step') {
            engine.step();
            const positions = engine.getPositions();
            self.postMessage({ positions: positions });
        }
    };
});
```

### WebGL Integration
```javascript
function setupWebGL(canvas) {
    const gl = canvas.getContext('webgl2');

    // Vertex shader for particles
    const vertexShader = `
        attribute vec3 position;
        uniform mat4 projectionMatrix;
        uniform mat4 viewMatrix;

        void main() {
            gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
            gl_PointSize = 2.0;
        }
    `;

    // Fragment shader
    const fragmentShader = `
        precision mediump float;
        uniform vec3 color;

        void main() {
            gl_FragColor = vec4(color, 1.0);
        }
    `;

    // Create and compile shaders
    const program = createShaderProgram(gl, vertexShader, fragmentShader);

    return { gl, program };
}

function renderParticles(gl, program, positions) {
    gl.useProgram(program);

    // Upload particle positions to GPU
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);

    // Set vertex attributes
    const positionLoc = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);

    // Draw particles
    gl.drawArrays(gl.POINTS, 0, positions.length / 3);
}
```

### Real-time Networking
```javascript
// WebSocket integration for multiplayer physics
const socket = new WebSocket('ws://physics-server.com');

function broadcastPhysicsState(engine) {
    const state = {
        positions: Array.from(engine.getPositions()),
        velocities: Array.from(engine.getVelocities()),
        timestamp: Date.now()
    };

    socket.send(JSON.stringify(state));
}

socket.onmessage = function(event) {
    const remoteState = JSON.parse(event.data);
    // Synchronize with remote physics state
    synchronizePhysics(remoteState);
};
```

## Troubleshooting

### Common Issues

#### 1. WASM Module Loading Fails
```javascript
PhysGradModule().catch(error => {
    console.error('Failed to load WASM module:', error);
    // Fallback to JavaScript implementation
});
```

#### 2. Out of Memory Errors
```javascript
// Monitor memory usage
setInterval(() => {
    const memory = Module.MemoryManager.getAllocatedBytes();
    if (memory > 1024 * 1024 * 100) { // 100MB threshold
        console.warn('High memory usage:', memory);
        engine.reset(); // Clear particles if needed
    }
}, 1000);
```

#### 3. Performance Issues
```javascript
// Adaptive quality based on performance
function adaptiveQuality(engine) {
    const fps = engine.getFPS();
    const particleCount = engine.getParticleCount();

    if (fps < 30 && particleCount > 1000) {
        // Reduce particle count or timestep
        engine.setTimestep(engine.getTimestep() * 1.5);
    }
}
```

### Debug Mode
Build with debug flags for development:
```bash
BUILD_TYPE=Debug ./build_wasm.sh build
```

Debug features include:
- **Assertion checking**
- **Memory bounds validation**
- **Stack overflow detection**
- **Detailed error messages**

## Contributing

### Development Setup
1. Clone repository and install Emscripten
2. Run native tests: `./test_wasm_bridge`
3. Build WASM module: `./build_wasm.sh build`
4. Test in browser: Open `shell.html` in browser

### Testing
```bash
# Run all tests
./build_wasm.sh test

# Performance benchmarks
./build_wasm.sh benchmark

# Memory leak detection
valgrind ./test_wasm_bridge
```

### Code Guidelines
- Use const/constexpr for compile-time constants
- Prefer RAII for resource management
- Add comprehensive test coverage for new features
- Document public API changes in TypeScript definitions

## License

PhysGrad WebAssembly module is released under the MIT License. See the main repository for full license terms.