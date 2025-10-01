#!/bin/bash
set -euo pipefail

# PhysGrad WebAssembly Build Script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
BUILD_TYPE="${BUILD_TYPE:-Release}"
EMSCRIPTEN_VERSION="${EMSCRIPTEN_VERSION:-latest}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/build}"
INSTALL_DIR="${INSTALL_DIR:-${SCRIPT_DIR}/dist}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking WebAssembly build prerequisites..."

    # Check for Emscripten
    if ! command -v emcc &> /dev/null; then
        log_error "Emscripten not found. Please install Emscripten SDK:"
        echo "  git clone https://github.com/emscripten-core/emsdk.git"
        echo "  cd emsdk"
        echo "  ./emsdk install latest"
        echo "  ./emsdk activate latest"
        echo "  source ./emsdk_env.sh"
        exit 1
    fi

    local emcc_version=$(emcc --version | head -n1)
    log_success "Found Emscripten: $emcc_version"

    # Check for CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Please install CMake 3.16 or later."
        exit 1
    fi

    local cmake_version=$(cmake --version | head -n1)
    log_success "Found CMake: $cmake_version"

    # Check Node.js (for testing)
    if command -v node &> /dev/null; then
        local node_version=$(node --version)
        log_success "Found Node.js: $node_version"
    else
        log_warning "Node.js not found - WASM module testing will be limited"
    fi
}

# Setup build environment
setup_build_env() {
    log_info "Setting up WebAssembly build environment..."

    # Create build directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$INSTALL_DIR"

    # Set Emscripten environment
    if [[ -n "${EMSDK:-}" ]]; then
        source "$EMSDK/emsdk_env.sh"
        log_success "Emscripten environment activated"
    fi

    # Export build variables
    export CMAKE_TOOLCHAIN_FILE="$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"
    export EMSCRIPTEN_ROOT_PATH="$EMSDK/upstream/emscripten"

    log_success "Build environment configured"
}

# Configure CMake for WebAssembly
configure_cmake() {
    log_info "Configuring CMake for WebAssembly build..."

    cd "$OUTPUT_DIR"

    # Configure with Emscripten toolchain
    cmake "$SCRIPT_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DEMSCRIPTEN=ON \
        -DENABLE_WASM_SIMD=ON \
        -DENABLE_WASM_THREADS=OFF \
        -DENABLE_WASM_BULK_MEMORY=ON

    log_success "CMake configuration completed"
}

# Build WebAssembly module
build_wasm() {
    log_info "Building WebAssembly module..."

    cd "$OUTPUT_DIR"

    # Build with parallel jobs
    local num_jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    make -j"$num_jobs"

    log_success "WebAssembly build completed"
}

# Validate WebAssembly output
validate_wasm() {
    log_info "Validating WebAssembly output..."

    local wasm_file="$OUTPUT_DIR/physgrad_wasm.wasm"
    local js_file="$OUTPUT_DIR/physgrad_wasm.js"

    # Check if files exist
    if [[ ! -f "$wasm_file" ]]; then
        log_error "WASM file not found: $wasm_file"
        return 1
    fi

    if [[ ! -f "$js_file" ]]; then
        log_error "JavaScript file not found: $js_file"
        return 1
    fi

    # Check file sizes
    local wasm_size=$(stat -f%z "$wasm_file" 2>/dev/null || stat -c%s "$wasm_file" 2>/dev/null)
    local js_size=$(stat -f%z "$js_file" 2>/dev/null || stat -c%s "$js_file" 2>/dev/null)

    log_success "WASM file: $wasm_file ($(($wasm_size / 1024)) KB)"
    log_success "JS file: $js_file ($(($js_size / 1024)) KB)"

    # Validate WASM with wasm-validate if available
    if command -v wasm-validate &> /dev/null; then
        if wasm-validate "$wasm_file"; then
            log_success "WASM validation passed"
        else
            log_error "WASM validation failed"
            return 1
        fi
    else
        log_warning "wasm-validate not found - skipping WASM validation"
    fi

    return 0
}

# Test WebAssembly module
test_wasm() {
    log_info "Testing WebAssembly module..."

    if ! command -v node &> /dev/null; then
        log_warning "Node.js not available - skipping WASM tests"
        return 0
    fi

    # Create simple test script
    cat > "$OUTPUT_DIR/test_wasm.js" << 'EOF'
const fs = require('fs');
const path = require('path');

// Load the WebAssembly module
const wasmPath = path.join(__dirname, 'physgrad_wasm.wasm');
const jsPath = path.join(__dirname, 'physgrad_wasm.js');

if (!fs.existsSync(wasmPath) || !fs.existsSync(jsPath)) {
    console.error('WASM files not found');
    process.exit(1);
}

// Simple module loading test
try {
    const Module = require(jsPath);

    Module().then(instance => {
        console.log('✓ WASM module loaded successfully');

        // Basic functionality test
        if (instance.PhysicsEngine) {
            const engine = new instance.PhysicsEngine();
            console.log('✓ PhysicsEngine created');

            engine.initialize(1000);
            console.log('✓ Engine initialized');

            engine.addParticle(0, 0, 0, 1, 0, 0);
            const count = engine.getParticleCount();
            console.log(`✓ Particle added, count: ${count}`);

            engine.step();
            console.log('✓ Simulation step completed');

            const positions = engine.getPositions();
            console.log(`✓ Retrieved ${positions.length / 3} particle positions`);

            console.log('All tests passed!');
        } else {
            console.error('✗ PhysicsEngine not found in module');
            process.exit(1);
        }
    }).catch(err => {
        console.error('✗ Failed to load WASM module:', err);
        process.exit(1);
    });
} catch (err) {
    console.error('✗ Error testing WASM module:', err);
    process.exit(1);
}
EOF

    # Run the test
    cd "$OUTPUT_DIR"
    if node test_wasm.js; then
        log_success "WebAssembly tests passed"
    else
        log_error "WebAssembly tests failed"
        return 1
    fi

    return 0
}

# Optimize WebAssembly output
optimize_wasm() {
    log_info "Optimizing WebAssembly output..."

    local wasm_file="$OUTPUT_DIR/physgrad_wasm.wasm"
    local optimized_file="$OUTPUT_DIR/physgrad_wasm_optimized.wasm"

    # Use wasm-opt if available
    if command -v wasm-opt &> /dev/null; then
        log_info "Running wasm-opt optimization..."

        wasm-opt -Oz --enable-simd --enable-bulk-memory \
                 -o "$optimized_file" "$wasm_file"

        # Compare sizes
        local original_size=$(stat -f%z "$wasm_file" 2>/dev/null || stat -c%s "$wasm_file")
        local optimized_size=$(stat -f%z "$optimized_file" 2>/dev/null || stat -c%s "$optimized_file")
        local reduction=$((($original_size - $optimized_size) * 100 / $original_size))

        log_success "Optimization completed: ${reduction}% size reduction"

        # Replace original with optimized
        mv "$optimized_file" "$wasm_file"
    else
        log_warning "wasm-opt not found - skipping optimization"
    fi
}

# Package for distribution
package_wasm() {
    log_info "Packaging WebAssembly distribution..."

    cd "$OUTPUT_DIR"

    # Install files
    make install

    # Copy additional files
    cp "$SCRIPT_DIR/physgrad.d.ts" "$INSTALL_DIR/"

    # Generate package.json from template
    if [[ -f "$SCRIPT_DIR/package.json.in" ]]; then
        sed "s/@PROJECT_VERSION@/1.0.0/g" "$SCRIPT_DIR/package.json.in" > "$INSTALL_DIR/package.json"
    fi

    # Create README for distribution
    cat > "$INSTALL_DIR/README.md" << 'EOF'
# PhysGrad WebAssembly

This package contains the WebAssembly build of PhysGrad physics simulation engine.

## Usage

### Browser
```html
<script src="physgrad_wasm.js"></script>
<script>
PhysGradModule().then(Module => {
    const engine = new Module.PhysicsEngine();
    engine.initialize(10000);
    // ... use the engine
});
</script>
```

### Node.js
```javascript
const PhysGradModule = require('./physgrad_wasm.js');

PhysGradModule().then(Module => {
    const engine = new Module.PhysicsEngine();
    engine.initialize(10000);
    // ... use the engine
});
```

## Files
- `physgrad_wasm.wasm` - WebAssembly binary
- `physgrad_wasm.js` - JavaScript wrapper
- `physgrad.d.ts` - TypeScript definitions
- `package.json` - NPM package metadata

## API
See `physgrad.d.ts` for complete TypeScript definitions.
EOF

    # Create tarball
    tar -czf "physgrad-wasm-$(date +%Y%m%d).tar.gz" -C "$INSTALL_DIR" .

    log_success "Distribution package created: $INSTALL_DIR"
}

# Benchmark WebAssembly performance
benchmark_wasm() {
    log_info "Running WebAssembly performance benchmarks..."

    if ! command -v node &> /dev/null; then
        log_warning "Node.js not available - skipping benchmarks"
        return 0
    fi

    # Create benchmark script
    cat > "$OUTPUT_DIR/benchmark_wasm.js" << 'EOF'
const Module = require('./physgrad_wasm.js');

async function runBenchmarks() {
    const instance = await Module();
    const engine = new instance.PhysicsEngine();

    const particleCounts = [100, 500, 1000, 2000, 5000];

    console.log('WebAssembly Performance Benchmarks');
    console.log('===================================');

    for (const count of particleCounts) {
        console.log(`\nTesting ${count} particles:`);

        engine.reset();
        engine.initialize(count);

        // Create particle field
        const n = Math.cbrt(count);
        const spacing = 0.2;
        const offset = -n * spacing * 0.5;

        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                for (let k = 0; k < n; k++) {
                    const x = offset + i * spacing;
                    const y = offset + j * spacing;
                    const z = offset + k * spacing;
                    engine.addParticle(x, y, z, 0, 0, 0);
                }
            }
        }

        // Warm up
        for (let i = 0; i < 10; i++) {
            engine.step();
        }

        // Benchmark
        const frames = 100;
        const startTime = performance.now();

        for (let i = 0; i < frames; i++) {
            engine.step();
        }

        const endTime = performance.now();
        const totalTime = endTime - startTime;
        const avgFrameTime = totalTime / frames;
        const fps = 1000 / avgFrameTime;

        console.log(`  Average frame time: ${avgFrameTime.toFixed(2)} ms`);
        console.log(`  Average FPS: ${fps.toFixed(1)}`);
        console.log(`  Particles per second: ${(count * fps).toFixed(0)}`);
    }
}

runBenchmarks().catch(console.error);
EOF

    # Run benchmarks
    cd "$OUTPUT_DIR"
    if node benchmark_wasm.js; then
        log_success "Benchmarks completed successfully"
    else
        log_warning "Benchmarks failed or incomplete"
    fi
}

# Main build function
main() {
    local command="${1:-build}"

    case "$command" in
        check)
            check_prerequisites
            ;;
        configure)
            check_prerequisites
            setup_build_env
            configure_cmake
            ;;
        build)
            check_prerequisites
            setup_build_env
            configure_cmake
            build_wasm
            validate_wasm
            ;;
        test)
            test_wasm
            ;;
        optimize)
            optimize_wasm
            ;;
        package)
            package_wasm
            ;;
        benchmark)
            benchmark_wasm
            ;;
        clean)
            log_info "Cleaning build directory..."
            rm -rf "$OUTPUT_DIR"
            rm -rf "$INSTALL_DIR"
            log_success "Clean completed"
            ;;
        all)
            check_prerequisites
            setup_build_env
            configure_cmake
            build_wasm
            validate_wasm
            test_wasm
            optimize_wasm
            package_wasm
            benchmark_wasm
            ;;
        *)
            echo "Usage: $0 {check|configure|build|test|optimize|package|benchmark|clean|all}"
            echo ""
            echo "Commands:"
            echo "  check      - Check prerequisites"
            echo "  configure  - Configure build environment"
            echo "  build      - Build WebAssembly module"
            echo "  test       - Test WebAssembly module"
            echo "  optimize   - Optimize WebAssembly binary"
            echo "  package    - Create distribution package"
            echo "  benchmark  - Run performance benchmarks"
            echo "  clean      - Clean build artifacts"
            echo "  all        - Run all steps"
            echo ""
            echo "Environment variables:"
            echo "  BUILD_TYPE         - Debug or Release (default: Release)"
            echo "  EMSCRIPTEN_VERSION - Emscripten version (default: latest)"
            echo "  OUTPUT_DIR         - Build output directory"
            echo "  INSTALL_DIR        - Installation directory"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"