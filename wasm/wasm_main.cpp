#include "../src/wasm_bridge.h"
#include <iostream>
#include <chrono>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

using namespace physgrad::wasm;

// Global instance for callback access
WasmInterface* g_engine = nullptr;

// Main simulation loop for Emscripten
#ifdef __EMSCRIPTEN__
void emscripten_main_loop() {
    if (g_engine && g_engine->isRunning()) {
        g_engine->step();
    }
}
#endif

// Initialize the WebAssembly module
extern "C" {
    void wasm_initialize() {
        std::cout << "PhysGrad WebAssembly module initialized" << std::endl;

        if (!g_engine) {
            g_engine = new WasmInterface();
        }

        #ifdef __EMSCRIPTEN__
        // Set up main loop for Emscripten
        emscripten_set_main_loop(emscripten_main_loop, 60, 1);
        #endif
    }

    void wasm_cleanup() {
        if (g_engine) {
            delete g_engine;
            g_engine = nullptr;
        }
    }
}

// Demo scenarios for testing
class WasmDemo {
public:
    static void createParticleField(WasmInterface& engine, int count) {
        engine.reset();

        // Create a grid of particles
        int n = static_cast<int>(std::cbrt(count));
        float spacing = 0.2f;
        float offset = -n * spacing * 0.5f;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    float x = offset + i * spacing;
                    float y = offset + j * spacing;
                    float z = offset + k * spacing;

                    // Small random velocity
                    float vx = (rand() / float(RAND_MAX) - 0.5f) * 0.1f;
                    float vy = (rand() / float(RAND_MAX) - 0.5f) * 0.1f;
                    float vz = (rand() / float(RAND_MAX) - 0.5f) * 0.1f;

                    engine.addParticle(x, y, z, vx, vy, vz);
                }
            }
        }

        std::cout << "Created particle field with " << engine.getParticleCount()
                  << " particles" << std::endl;
    }

    static void createDamBreak(WasmInterface& engine) {
        engine.reset();

        // Create a block of particles (dam)
        engine.addBlock(-2.0f, -2.0f, -1.0f,  // corner
                       1.5f, 2.0f, 2.0f,      // dimensions
                       15, 20, 20);           // resolution

        std::cout << "Created dam break scenario with "
                  << engine.getParticleCount() << " particles" << std::endl;
    }

    static void createParticleRain(WasmInterface& engine, int count) {
        engine.reset();

        for (int i = 0; i < count; ++i) {
            // Random position above the domain
            float x = (rand() / float(RAND_MAX) - 0.5f) * 8.0f;
            float y = 4.0f + (rand() / float(RAND_MAX)) * 2.0f;
            float z = (rand() / float(RAND_MAX) - 0.5f) * 8.0f;

            // Small random horizontal velocity
            float vx = (rand() / float(RAND_MAX) - 0.5f) * 2.0f;
            float vy = 0.0f;
            float vz = (rand() / float(RAND_MAX) - 0.5f) * 2.0f;

            engine.addParticle(x, y, z, vx, vy, vz);
        }

        std::cout << "Created particle rain with " << engine.getParticleCount()
                  << " particles" << std::endl;
    }

    static void createExplosion(WasmInterface& engine, int count) {
        engine.reset();

        WasmVec3<float> center(0, 0, 0);
        float explosion_force = 10.0f;

        for (int i = 0; i < count; ++i) {
            // Random position near center
            float r = (rand() / float(RAND_MAX)) * 0.5f;
            float theta = (rand() / float(RAND_MAX)) * 2 * M_PI;
            float phi = (rand() / float(RAND_MAX)) * M_PI;

            float x = center.x + r * std::sin(phi) * std::cos(theta);
            float y = center.y + r * std::sin(phi) * std::sin(theta);
            float z = center.z + r * std::cos(phi);

            // Radial velocity
            WasmVec3<float> pos(x, y, z);
            WasmVec3<float> dir = (pos - center).normalized();
            WasmVec3<float> vel = dir * explosion_force;

            engine.addParticle(x, y, z, vel.x, vel.y, vel.z);
        }

        std::cout << "Created explosion with " << engine.getParticleCount()
                  << " particles" << std::endl;
    }
};

// Benchmarking utilities
class WasmBenchmark {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::vector<double> frame_times_;
    size_t frame_count_;

public:
    WasmBenchmark() : frame_count_(0) {
        frame_times_.reserve(1000);
    }

    void startFrame() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void endFrame() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end_time - start_time_;
        frame_times_.push_back(diff.count());
        frame_count_++;
    }

    void printStats() {
        if (frame_times_.empty()) return;

        double sum = 0;
        double min_time = frame_times_[0];
        double max_time = frame_times_[0];

        for (double time : frame_times_) {
            sum += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }

        double avg_time = sum / frame_times_.size();
        double avg_fps = 1000.0 / avg_time;

        std::cout << "Benchmark Results:" << std::endl;
        std::cout << "  Frames: " << frame_times_.size() << std::endl;
        std::cout << "  Avg Time: " << avg_time << " ms" << std::endl;
        std::cout << "  Min Time: " << min_time << " ms" << std::endl;
        std::cout << "  Max Time: " << max_time << " ms" << std::endl;
        std::cout << "  Avg FPS: " << avg_fps << std::endl;
    }

    void reset() {
        frame_times_.clear();
        frame_count_ = 0;
    }
};

// Performance testing
void runPerformanceTest() {
    std::cout << "Running WebAssembly performance tests..." << std::endl;

    WasmInterface engine;
    WasmBenchmark benchmark;

    // Test different particle counts
    std::vector<int> particle_counts = {100, 500, 1000, 2000, 5000};

    for (int count : particle_counts) {
        std::cout << "\nTesting with " << count << " particles:" << std::endl;

        WasmDemo::createParticleField(engine, count);
        engine.setTimestep(0.016f); // 60 FPS target

        benchmark.reset();

        // Run simulation for 60 frames
        for (int frame = 0; frame < 60; ++frame) {
            benchmark.startFrame();
            engine.step();
            benchmark.endFrame();
        }

        benchmark.printStats();

        // Memory usage
        std::cout << "  Memory allocated: "
                  << WasmMemoryManager::getAllocatedBytes() / 1024
                  << " KB" << std::endl;
    }
}

// Main function
int main() {
    std::cout << "PhysGrad WebAssembly Engine" << std::endl;
    std::cout << "============================" << std::endl;

    #ifdef __EMSCRIPTEN__
    std::cout << "Running in WebAssembly mode" << std::endl;
    wasm_initialize();
    #else
    std::cout << "Running in native mode for testing" << std::endl;

    // Run performance tests in native mode
    runPerformanceTest();

    // Demo scenarios
    WasmInterface engine;

    std::cout << "\nTesting dam break scenario:" << std::endl;
    WasmDemo::createDamBreak(engine);

    for (int i = 0; i < 100; ++i) {
        engine.step();
        if (i % 20 == 0) {
            std::cout << "Step " << i << ", FPS: " << engine.getFPS() << std::endl;
        }
    }

    std::cout << "\nTesting explosion scenario:" << std::endl;
    WasmDemo::createExplosion(engine, 1000);

    for (int i = 0; i < 100; ++i) {
        engine.step();
        if (i % 20 == 0) {
            std::cout << "Step " << i << ", FPS: " << engine.getFPS() << std::endl;
        }
    }
    #endif

    return 0;
}