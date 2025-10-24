#include "src/wasm_bridge.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace physgrad::wasm;

// Test utility functions
void assertTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << std::endl;
        exit(1);
    }
}

void assertNear(float a, float b, float tolerance, const std::string& message) {
    if (std::abs(a - b) > tolerance) {
        std::cerr << "FAILED: " << message << " (expected " << a << ", got " << b << ")" << std::endl;
        exit(1);
    }
}

// Test Vec3 functionality
void testVec3() {
    std::cout << "Testing WasmVec3..." << std::endl;

    WasmVec3<float> v1(1.0f, 2.0f, 3.0f);
    WasmVec3<float> v2(4.0f, 5.0f, 6.0f);

    // Test addition
    auto v3 = v1 + v2;
    assertTrue(v3.x == 5.0f && v3.y == 7.0f && v3.z == 9.0f, "Vec3 addition");

    // Test subtraction
    auto v4 = v2 - v1;
    assertTrue(v4.x == 3.0f && v4.y == 3.0f && v4.z == 3.0f, "Vec3 subtraction");

    // Test scalar multiplication
    auto v5 = v1 * 2.0f;
    assertTrue(v5.x == 2.0f && v5.y == 4.0f && v5.z == 6.0f, "Vec3 scalar multiplication");

    // Test dot product
    float dot = v1.dot(v2);
    assertNear(dot, 32.0f, 1e-6f, "Vec3 dot product");

    // Test norm
    WasmVec3<float> unit(3.0f, 4.0f, 0.0f);
    assertNear(unit.norm(), 5.0f, 1e-6f, "Vec3 norm");

    // Test normalization
    auto normalized = unit.normalized();
    assertNear(normalized.norm(), 1.0f, 1e-6f, "Vec3 normalization");

    std::cout << "✓ WasmVec3 tests passed" << std::endl;
}

// Test particle functionality
void testParticle() {
    std::cout << "Testing WasmParticle..." << std::endl;

    WasmParticle<float> particle;
    particle.position = WasmVec3<float>(1.0f, 2.0f, 3.0f);
    particle.velocity = WasmVec3<float>(0.1f, 0.2f, 0.3f);
    particle.mass = 1.5f;
    particle.radius = 0.05f;
    particle.material_id = 1;

    assertTrue(particle.active, "Particle should be active by default");
    assertNear(particle.mass, 1.5f, 1e-6f, "Particle mass");
    assertNear(particle.radius, 0.05f, 1e-6f, "Particle radius");
    assertTrue(particle.material_id == 1, "Particle material ID");

    std::cout << "✓ WasmParticle tests passed" << std::endl;
}

// Test material functionality
void testMaterial() {
    std::cout << "Testing WasmMaterial..." << std::endl;

    WasmMaterial<float> material;
    material.density = 1200.0f;
    material.youngs_modulus = 2e6f;
    material.poisson_ratio = 0.35f;
    material.material_type = 1; // plastic

    assertNear(material.density, 1200.0f, 1e-6f, "Material density");
    assertNear(material.youngs_modulus, 2e6f, 1e-6f, "Material Young's modulus");
    assertNear(material.poisson_ratio, 0.35f, 1e-6f, "Material Poisson ratio");
    assertTrue(material.material_type == 1, "Material type");

    std::cout << "✓ WasmMaterial tests passed" << std::endl;
}

// Test physics engine functionality
void testPhysicsEngine() {
    std::cout << "Testing WasmPhysicsEngine..." << std::endl;

    WasmPhysicsEngine<float> engine(1000);

    // Test initial state
    assertTrue(engine.getParticleCount() == 0, "Initial particle count should be zero");
    assertNear(engine.getTimestep(), 0.001f, 1e-6f, "Default timestep");

    // Test adding particles
    int id1 = engine.addParticle(WasmVec3<float>(0, 0, 0), WasmVec3<float>(1, 0, 0));
    int id2 = engine.addParticle(WasmVec3<float>(1, 1, 1), WasmVec3<float>(0, 1, 0));

    assertTrue(id1 == 0, "First particle ID");
    assertTrue(id2 == 1, "Second particle ID");
    assertTrue(engine.getParticleCount() == 2, "Particle count after adding");

    // Test particle block creation
    engine.addParticleBlock(WasmVec3<float>(-1, -1, -1), WasmVec3<float>(0.5f, 0.5f, 0.5f), 2, 2, 2);
    assertTrue(engine.getParticleCount() == 10, "Particle count after adding block"); // 2 + 2*2*2 = 10

    // Test getting positions
    auto positions = engine.getParticlePositions();
    assertTrue(positions.size() == 30, "Position array size should be 3 * particle count"); // 10 particles * 3 components

    // Test getting velocities
    auto velocities = engine.getParticleVelocities();
    assertTrue(velocities.size() == 30, "Velocity array size should be 3 * particle count");

    // Test simulation step
    engine.step();
    auto positions_after = engine.getParticlePositions();
    assertTrue(positions_after.size() == 30, "Position array size after step");

    // Test gravity setting
    engine.setGravity(0, -10.0f, 0);
    engine.step(); // Should apply new gravity

    // Test timestep setting
    engine.setTimestep(0.01f);
    assertNear(engine.getTimestep(), 0.01f, 1e-6f, "Updated timestep");

    // Test reset
    engine.reset();
    assertTrue(engine.getParticleCount() == 0, "Particle count after reset");

    std::cout << "✓ WasmPhysicsEngine tests passed" << std::endl;
}

// Test grid functionality
void testGrid() {
    std::cout << "Testing grid functionality..." << std::endl;

    WasmPhysicsEngine<float> engine(100);

    // Add particles in a known pattern
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            engine.addParticle(
                WasmVec3<float>(i * 0.1f, j * 0.1f, 0),
                WasmVec3<float>(0, 0, 0)
            );
        }
    }

    assertTrue(engine.getParticleCount() == 25, "Grid particle count");

    // Run a few simulation steps
    for (int step = 0; step < 10; ++step) {
        engine.step();
    }

    auto positions = engine.getParticlePositions();
    assertTrue(positions.size() == 75, "Grid position array size"); // 25 particles * 3 components

    // Verify particles are still within reasonable bounds
    bool particles_in_bounds = true;
    for (size_t i = 0; i < positions.size(); i += 3) {
        float x = positions[i];
        float y = positions[i + 1];
        float z = positions[i + 2];

        if (std::abs(x) > 10.0f || std::abs(y) > 10.0f || std::abs(z) > 10.0f) {
            particles_in_bounds = false;
            break;
        }
    }
    assertTrue(particles_in_bounds, "Particles should remain within bounds");

    std::cout << "✓ Grid functionality tests passed" << std::endl;
}

// Test memory management
void testMemoryManagement() {
    std::cout << "Testing memory management..." << std::endl;

    WasmMemoryManager::reset();
    size_t initial_allocated = WasmMemoryManager::getAllocatedBytes();
    assertTrue(initial_allocated == 0, "Initial allocated memory should be zero");

    // Test allocation
    void* ptr = WasmMemoryManager::allocate(1024);
    assertTrue(ptr != nullptr, "Memory allocation should succeed");
    assertTrue(WasmMemoryManager::getAllocatedBytes() >= 1024, "Allocated memory should be tracked");
    assertTrue(WasmMemoryManager::getAllocationCount() == 1, "Allocation count should be tracked");

    // Test deallocation
    WasmMemoryManager::deallocate(ptr);
    assertTrue(WasmMemoryManager::getAllocatedBytes() == 0, "Memory should be deallocated");
    assertTrue(WasmMemoryManager::getAllocationCount() == 0, "Allocation count should be zero");

    std::cout << "✓ Memory management tests passed" << std::endl;
}

// Test interface functionality
void testInterface() {
    std::cout << "Testing WasmInterface..." << std::endl;

    WasmInterface interface;

    // Test initialization
    interface.initialize(500);
    assertTrue(interface.getParticleCount() == 0, "Initial interface particle count");

    // Test adding particles through interface
    interface.addParticle(0, 0, 0, 1, 0, 0);
    interface.addParticle(1, 1, 1, 0, 1, 0);
    assertTrue(interface.getParticleCount() == 2, "Interface particle count after adding");

    // Test adding block through interface
    interface.addBlock(-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 2, 2, 2);
    assertTrue(interface.getParticleCount() == 10, "Interface particle count after adding block");

    // Test simulation control
    assertTrue(!interface.isRunning(), "Interface should not be running initially");
    interface.start();
    assertTrue(interface.isRunning(), "Interface should be running after start");

    interface.step();
    auto positions = interface.getPositions();
    assertTrue(positions.size() == 30, "Interface position array size");

    interface.stop();
    assertTrue(!interface.isRunning(), "Interface should not be running after stop");

    // Test parameter setting
    interface.setGravity(0, -5.0f, 0);
    interface.setTimestep(0.005f);
    interface.enableSIMD(true);

    // Test reset
    interface.reset();
    assertTrue(interface.getParticleCount() == 0, "Interface particle count after reset");

    std::cout << "✓ WasmInterface tests passed" << std::endl;
}

// Performance test
void testPerformance() {
    std::cout << "Testing performance..." << std::endl;

    WasmInterface interface;
    interface.initialize(5000);

    // Create a moderate particle system
    interface.addBlock(-1.0f, -1.0f, -1.0f, 2.0f, 2.0f, 2.0f, 10, 10, 10);
    std::cout << "Created " << interface.getParticleCount() << " particles for performance test" << std::endl;

    interface.setTimestep(0.01f);
    interface.start();

    auto start_time = std::chrono::high_resolution_clock::now();

    // Run 100 simulation steps
    for (int i = 0; i < 100; ++i) {
        interface.step();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end_time - start_time;

    double avg_frame_time = diff.count() / 100.0;
    double fps = 1000.0 / avg_frame_time;

    std::cout << "Performance results:" << std::endl;
    std::cout << "  Average frame time: " << avg_frame_time << " ms" << std::endl;
    std::cout << "  Average FPS: " << fps << std::endl;
    std::cout << "  Particles per second: " << (interface.getParticleCount() * fps) << std::endl;

    // Performance should be reasonable
    assertTrue(fps > 100, "FPS should be greater than 100 for this particle count");

    interface.stop();

    std::cout << "✓ Performance test passed" << std::endl;
}

int main() {
    std::cout << "PhysGrad WebAssembly Bridge Test Suite" << std::endl;
    std::cout << "======================================" << std::endl;

    try {
        testVec3();
        testParticle();
        testMaterial();
        testPhysicsEngine();
        testGrid();
        testMemoryManagement();
        testInterface();
        testPerformance();

        std::cout << std::endl;
        std::cout << "🎉 All tests passed!" << std::endl;
        std::cout << "WebAssembly bridge implementation is working correctly." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}