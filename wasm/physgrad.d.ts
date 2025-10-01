// TypeScript definitions for PhysGrad WebAssembly module

export interface Vec3 {
    x: number;
    y: number;
    z: number;
}

export interface PhysicsEngine {
    // Initialization
    initialize(maxParticles: number): void;

    // Particle management
    addParticle(x: number, y: number, z: number, vx: number, vy: number, vz: number): void;
    addBlock(x: number, y: number, z: number, w: number, h: number, d: number,
             nx: number, ny: number, nz: number): void;

    // Simulation control
    step(): void;
    start(): void;
    stop(): void;
    isRunning(): boolean;
    reset(): void;

    // Data access
    getPositions(): Float32Array;
    getVelocities(): Float32Array;
    getParticleCount(): number;

    // Configuration
    setGravity(x: number, y: number, z: number): void;
    setTimestep(dt: number): void;
    enableSIMD(enable: boolean): void;

    // Performance monitoring
    getFPS(): number;
}

export interface MemoryManager {
    getAllocatedBytes(): number;
    getPeakBytes(): number;
    getAllocationCount(): number;
    reset(): void;
}

export interface PhysGradModule extends EmscriptenModule {
    PhysicsEngine: {
        new(): PhysicsEngine;
    };
    MemoryManager: MemoryManager;
}

// Demo and utility functions
export interface WasmDemo {
    createParticleField(engine: PhysicsEngine, count: number): void;
    createDamBreak(engine: PhysicsEngine): void;
    createParticleRain(engine: PhysicsEngine, count: number): void;
    createExplosion(engine: PhysicsEngine, count: number): void;
}

export interface WasmBenchmark {
    startFrame(): void;
    endFrame(): void;
    printStats(): void;
    reset(): void;
}

// Module loading
export default function PhysGradModuleFactory(options?: Partial<EmscriptenModule>): Promise<PhysGradModule>;

// Performance monitoring types
export interface PerformanceStats {
    averageFrameTime: number;
    minFrameTime: number;
    maxFrameTime: number;
    averageFPS: number;
    memoryUsage: number;
    peakMemoryUsage: number;
}

// Configuration options
export interface EngineConfig {
    maxParticles?: number;
    timestep?: number;
    gravity?: Vec3;
    enableSIMD?: boolean;
    domainMin?: Vec3;
    domainMax?: Vec3;
    gridResolution?: number;
}

// Simulation scenarios
export enum SimulationScenario {
    PARTICLE_FIELD = 'particle_field',
    DAM_BREAK = 'dam_break',
    PARTICLE_RAIN = 'particle_rain',
    EXPLOSION = 'explosion',
    CUSTOM = 'custom'
}

// Material types
export enum MaterialType {
    ELASTIC = 0,
    PLASTIC = 1,
    FLUID = 2
}

// Advanced particle data
export interface ParticleData {
    positions: Float32Array;
    velocities: Float32Array;
    forces?: Float32Array;
    masses?: Float32Array;
    materials?: Int32Array;
    active?: boolean[];
}

// Rendering helpers
export interface RenderingUtils {
    createParticleBuffer(positions: Float32Array): WebGLBuffer;
    updateParticleBuffer(buffer: WebGLBuffer, positions: Float32Array): void;
    createShaderProgram(gl: WebGLRenderingContext, vertexSource: string, fragmentSource: string): WebGLProgram;
}

// Event system
export interface SimulationEvents {
    onStep?: (frameData: ParticleData) => void;
    onReset?: () => void;
    onParticleAdded?: (count: number) => void;
    onPerformanceUpdate?: (stats: PerformanceStats) => void;
}

// Integration with web technologies
export interface WebIntegration {
    // WebGL rendering
    setupWebGLRenderer(canvas: HTMLCanvasElement): WebGLRenderingContext;
    renderParticles(gl: WebGLRenderingContext, positions: Float32Array): void;

    // Web Workers
    createWorker(): Worker;
    runInWorker(engine: PhysicsEngine): void;

    // Real-time communication
    setupWebSocket(url: string): WebSocket;
    broadcastData(ws: WebSocket, data: ParticleData): void;
}

// Browser compatibility
export interface BrowserSupport {
    hasWebAssembly(): boolean;
    hasWebGL(): boolean;
    hasWebGL2(): boolean;
    hasSIMD(): boolean;
    hasSharedArrayBuffer(): boolean;
    hasWebWorkers(): boolean;
}

// Error handling
export class PhysGradWasmError extends Error {
    constructor(message: string, public code?: string) {
        super(message);
        this.name = 'PhysGradWasmError';
    }
}

// Utility functions
export namespace Utils {
    export function createVec3(x: number, y: number, z: number): Vec3;
    export function vec3Distance(a: Vec3, b: Vec3): number;
    export function vec3Normalize(v: Vec3): Vec3;
    export function vec3Dot(a: Vec3, b: Vec3): number;
    export function checkBrowserSupport(): BrowserSupport;
    export function downloadArrayAsFile(data: Float32Array, filename: string): void;
    export function loadArrayFromFile(file: File): Promise<Float32Array>;
}

// Constants
export const WASM_CONSTANTS = {
    MAX_PARTICLES: 100000,
    DEFAULT_TIMESTEP: 0.016, // 60 FPS
    DEFAULT_GRAVITY: { x: 0, y: -9.81, z: 0 },
    GRID_RESOLUTION: 64,
    MEMORY_GROWTH_FACTOR: 1.5
} as const;