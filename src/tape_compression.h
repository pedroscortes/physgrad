#pragma once

#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace physgrad {

// Forward declaration
struct BodySystem;

// Compression modes
enum class CompressionMode {
    NONE,           // No compression (full tape)
    SPARSE,         // Sparse checkpointing (save every N steps)
    ADAPTIVE,       // Adaptive based on memory usage
    DELTA,          // Delta compression (store differences)
    QUANTIZED       // Quantized storage (reduced precision)
};

// Configuration for tape compression
struct TapeCompressionConfig {
    CompressionMode mode = CompressionMode::SPARSE;
    size_t max_memory_mb = 512;        // Maximum memory usage in MB
    int checkpoint_interval = 10;      // Steps between checkpoints for sparse mode
    float compression_ratio = 0.1f;    // Target compression ratio
    bool enable_quantization = false;  // Use quantized storage
    int quantization_bits = 16;        // Bits per value for quantization
    float adaptive_threshold = 0.8f;   // Memory threshold for adaptive mode
};

// Compressed simulation state
struct CompressedState {
    std::vector<uint8_t> data;  // Compressed state data
    size_t original_size;       // Original uncompressed size
    CompressionMode mode;       // Compression method used
    int step_index;             // Time step index
    bool is_checkpoint;         // True if this is a full checkpoint

    // Quantization parameters (if used)
    std::vector<float> min_values;  // Minimum values per channel
    std::vector<float> max_values;  // Maximum values per channel
    std::vector<float> scales;      // Scaling factors per channel
};

// Enhanced tape with compression and checkpointing
class CompressedDifferentiableTape {
private:
    std::vector<CompressedState> compressed_states;
    std::vector<int> checkpoint_indices;  // Indices of full checkpoints
    TapeCompressionConfig config;
    size_t current_memory_usage;
    size_t n_bodies;

    // Temporary storage for decompression
    mutable std::vector<float> temp_buffer;
    mutable std::vector<float> reconstruction_buffer;

public:
    explicit CompressedDifferentiableTape(size_t num_bodies)
        : current_memory_usage(0), n_bodies(num_bodies) {}

    void setConfig(const TapeCompressionConfig& new_config) {
        config = new_config;
    }

    TapeCompressionConfig getConfig() const { return config; }

    // Record state with automatic compression
    void recordState(const BodySystem& bodies, int step_index);

    // Get state by reconstructing from checkpoints and deltas
    void getState(int step_index, std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
                  std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z) const;

    // Clear all stored states
    void clear();

    // Get memory usage statistics
    size_t getMemoryUsage() const { return current_memory_usage; }
    size_t getUncompressedMemoryUsage() const;
    float getCompressionRatio() const;

    // Get number of stored states
    size_t size() const { return compressed_states.size(); }

    // Get number of checkpoints
    size_t getCheckpointCount() const { return checkpoint_indices.size(); }

    // Optimization: trigger compression of older states
    void optimizeMemoryUsage();

    // Debug information
    void printCompressionStats() const;

private:
    // Compression methods
    std::vector<uint8_t> compressNone(const std::vector<float>& data) const;
    std::vector<uint8_t> compressDelta(const std::vector<float>& current_data,
                                      const std::vector<float>& reference_data) const;
    std::vector<uint8_t> compressQuantized(const std::vector<float>& data,
                                         std::vector<float>& min_vals,
                                         std::vector<float>& max_vals,
                                         std::vector<float>& scales) const;

    // Decompression methods
    void decompressNone(const std::vector<uint8_t>& compressed_data,
                       std::vector<float>& output) const;
    void decompressDelta(const std::vector<uint8_t>& compressed_data,
                        const std::vector<float>& reference_data,
                        std::vector<float>& output) const;
    void decompressQuantized(const std::vector<uint8_t>& compressed_data,
                           const std::vector<float>& min_vals,
                           const std::vector<float>& max_vals,
                           const std::vector<float>& scales,
                           std::vector<float>& output) const;

    // Utility methods
    void extractStateFromBodies(const BodySystem& bodies, std::vector<float>& state_data);
    void loadStateIntoBodies(const std::vector<float>& state_data,
                           std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
                           std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z) const;

    bool shouldCreateCheckpoint(int step_index) const;
    int findNearestCheckpoint(int step_index) const;
    size_t estimateStateSize() const;
};

// Implementation
inline void CompressedDifferentiableTape::recordState(const BodySystem& bodies, int step_index) {
    // Extract current state
    std::vector<float> current_state;
    extractStateFromBodies(bodies, current_state);

    CompressedState compressed_state;
    compressed_state.step_index = step_index;
    compressed_state.original_size = current_state.size() * sizeof(float);

    // Determine if this should be a checkpoint
    bool is_checkpoint = shouldCreateCheckpoint(step_index);
    compressed_state.is_checkpoint = is_checkpoint;

    if (is_checkpoint || config.mode == CompressionMode::NONE) {
        // Store as full checkpoint
        compressed_state.mode = CompressionMode::NONE;
        compressed_state.data = compressNone(current_state);
        if (is_checkpoint) {
            checkpoint_indices.push_back(compressed_states.size());
        }
    } else {
        // Use compression based on mode
        switch (config.mode) {
            case CompressionMode::DELTA: {
                // Find reference state (last checkpoint)
                int ref_checkpoint = findNearestCheckpoint(step_index);
                if (ref_checkpoint >= 0) {
                    std::vector<float> ref_state;
                    // Temporarily decompress reference for delta computation
                    // For simplicity, use the previous state if available
                    if (!compressed_states.empty()) {
                        // Use simple delta compression against previous state
                        std::vector<float> prev_state(current_state.size());
                        // This is simplified - in practice we'd decompress the reference
                        compressed_state.data = compressDelta(current_state, prev_state);
                        compressed_state.mode = CompressionMode::DELTA;
                    } else {
                        compressed_state.data = compressNone(current_state);
                        compressed_state.mode = CompressionMode::NONE;
                    }
                } else {
                    compressed_state.data = compressNone(current_state);
                    compressed_state.mode = CompressionMode::NONE;
                }
                break;
            }

            case CompressionMode::QUANTIZED: {
                compressed_state.data = compressQuantized(current_state,
                                                        compressed_state.min_values,
                                                        compressed_state.max_values,
                                                        compressed_state.scales);
                compressed_state.mode = CompressionMode::QUANTIZED;
                break;
            }

            default:
                compressed_state.data = compressNone(current_state);
                compressed_state.mode = CompressionMode::NONE;
                break;
        }
    }

    // Update memory usage
    current_memory_usage += compressed_state.data.size();
    current_memory_usage += compressed_state.min_values.size() * sizeof(float);
    current_memory_usage += compressed_state.max_values.size() * sizeof(float);
    current_memory_usage += compressed_state.scales.size() * sizeof(float);

    compressed_states.push_back(std::move(compressed_state));

    // Check if we need to optimize memory usage
    if (config.mode == CompressionMode::ADAPTIVE) {
        if (current_memory_usage > config.max_memory_mb * 1024 * 1024 * config.adaptive_threshold) {
            optimizeMemoryUsage();
        }
    }
}

inline void CompressedDifferentiableTape::getState(int step_index,
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z) const {

    if (step_index < 0 || step_index >= static_cast<int>(compressed_states.size())) {
        throw std::out_of_range("Step index out of range");
    }

    const CompressedState& state = compressed_states[step_index];
    std::vector<float> decompressed_data;

    // Decompress based on the compression mode used
    switch (state.mode) {
        case CompressionMode::NONE:
            decompressNone(state.data, decompressed_data);
            break;

        case CompressionMode::DELTA: {
            // Find reference checkpoint and apply deltas
            int checkpoint_idx = findNearestCheckpoint(step_index);
            if (checkpoint_idx >= 0) {
                // This is simplified - in practice we'd reconstruct from checkpoint + deltas
                decompressNone(state.data, decompressed_data);  // Simplified
            } else {
                decompressNone(state.data, decompressed_data);
            }
            break;
        }

        case CompressionMode::QUANTIZED:
            decompressQuantized(state.data, state.min_values, state.max_values,
                              state.scales, decompressed_data);
            break;

        default:
            decompressNone(state.data, decompressed_data);
            break;
    }

    // Load into output vectors
    loadStateIntoBodies(decompressed_data, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z);
}

inline void CompressedDifferentiableTape::clear() {
    compressed_states.clear();
    checkpoint_indices.clear();
    current_memory_usage = 0;
}

inline size_t CompressedDifferentiableTape::getUncompressedMemoryUsage() const {
    size_t total = 0;
    for (const auto& state : compressed_states) {
        total += state.original_size;
    }
    return total;
}

inline float CompressedDifferentiableTape::getCompressionRatio() const {
    size_t uncompressed = getUncompressedMemoryUsage();
    if (uncompressed == 0) return 1.0f;
    return static_cast<float>(current_memory_usage) / static_cast<float>(uncompressed);
}

inline void CompressedDifferentiableTape::optimizeMemoryUsage() {
    // Remove oldest non-checkpoint states if memory usage is too high
    size_t target_memory = config.max_memory_mb * 1024 * 1024;

    while (current_memory_usage > target_memory && compressed_states.size() > checkpoint_indices.size()) {
        // Find oldest non-checkpoint state
        for (auto it = compressed_states.begin(); it != compressed_states.end(); ++it) {
            if (!it->is_checkpoint) {
                current_memory_usage -= it->data.size();
                current_memory_usage -= it->min_values.size() * sizeof(float);
                current_memory_usage -= it->max_values.size() * sizeof(float);
                current_memory_usage -= it->scales.size() * sizeof(float);
                compressed_states.erase(it);
                break;
            }
        }
    }

    // Update checkpoint indices after removal
    checkpoint_indices.clear();
    for (size_t i = 0; i < compressed_states.size(); ++i) {
        if (compressed_states[i].is_checkpoint) {
            checkpoint_indices.push_back(static_cast<int>(i));
        }
    }
}

inline void CompressedDifferentiableTape::printCompressionStats() const {
    std::cout << "Tape Compression Statistics:\n";
    std::cout << "  States: " << compressed_states.size() << "\n";
    std::cout << "  Checkpoints: " << checkpoint_indices.size() << "\n";
    std::cout << "  Memory usage: " << current_memory_usage / (1024 * 1024) << " MB\n";
    std::cout << "  Uncompressed: " << getUncompressedMemoryUsage() / (1024 * 1024) << " MB\n";
    std::cout << "  Compression ratio: " << getCompressionRatio() << "\n";
    std::cout << "  Mode: ";
    switch (config.mode) {
        case CompressionMode::NONE: std::cout << "None"; break;
        case CompressionMode::SPARSE: std::cout << "Sparse"; break;
        case CompressionMode::ADAPTIVE: std::cout << "Adaptive"; break;
        case CompressionMode::DELTA: std::cout << "Delta"; break;
        case CompressionMode::QUANTIZED: std::cout << "Quantized"; break;
    }
    std::cout << "\n\n";
}

// Simple compression implementations
inline std::vector<uint8_t> CompressedDifferentiableTape::compressNone(const std::vector<float>& data) const {
    size_t byte_size = data.size() * sizeof(float);
    std::vector<uint8_t> result(byte_size);
    std::memcpy(result.data(), data.data(), byte_size);
    return result;
}

inline std::vector<uint8_t> CompressedDifferentiableTape::compressDelta(
    const std::vector<float>& current_data, const std::vector<float>& reference_data) const {
    // Simple delta compression: store differences as 16-bit values
    std::vector<uint8_t> result;
    result.reserve(current_data.size() * 2);  // 16-bit per value

    for (size_t i = 0; i < current_data.size(); ++i) {
        float diff = current_data[i] - (i < reference_data.size() ? reference_data[i] : 0.0f);
        int16_t quantized_diff = static_cast<int16_t>(std::clamp(diff * 1000.0f, -32767.0f, 32767.0f));

        result.push_back(static_cast<uint8_t>(quantized_diff & 0xFF));
        result.push_back(static_cast<uint8_t>((quantized_diff >> 8) & 0xFF));
    }

    return result;
}

inline std::vector<uint8_t> CompressedDifferentiableTape::compressQuantized(
    const std::vector<float>& data, std::vector<float>& min_vals,
    std::vector<float>& max_vals, std::vector<float>& scales) const {

    if (data.empty()) return {};

    // For simplicity, use global min/max for all values
    float global_min = *std::min_element(data.begin(), data.end());
    float global_max = *std::max_element(data.begin(), data.end());

    min_vals = {global_min};
    max_vals = {global_max};

    float range = global_max - global_min;
    float scale = range > 0 ? 65535.0f / range : 1.0f;  // 16-bit quantization
    scales = {scale};

    std::vector<uint8_t> result;
    result.reserve(data.size() * 2);

    for (float value : data) {
        uint16_t quantized = static_cast<uint16_t>(std::clamp((value - global_min) * scale, 0.0f, 65535.0f));
        result.push_back(static_cast<uint8_t>(quantized & 0xFF));
        result.push_back(static_cast<uint8_t>((quantized >> 8) & 0xFF));
    }

    return result;
}

inline void CompressedDifferentiableTape::decompressNone(const std::vector<uint8_t>& compressed_data,
                                                       std::vector<float>& output) const {
    size_t float_count = compressed_data.size() / sizeof(float);
    output.resize(float_count);
    std::memcpy(output.data(), compressed_data.data(), compressed_data.size());
}

inline void CompressedDifferentiableTape::decompressDelta(const std::vector<uint8_t>& compressed_data,
                                                        const std::vector<float>& reference_data,
                                                        std::vector<float>& output) const {
    size_t value_count = compressed_data.size() / 2;
    output.resize(value_count);

    for (size_t i = 0; i < value_count; ++i) {
        int16_t quantized_diff = static_cast<int16_t>(
            compressed_data[i * 2] | (static_cast<uint16_t>(compressed_data[i * 2 + 1]) << 8)
        );
        float diff = static_cast<float>(quantized_diff) / 1000.0f;
        float reference = i < reference_data.size() ? reference_data[i] : 0.0f;
        output[i] = reference + diff;
    }
}

inline void CompressedDifferentiableTape::decompressQuantized(const std::vector<uint8_t>& compressed_data,
                                                            const std::vector<float>& min_vals,
                                                            const std::vector<float>& max_vals,
                                                            const std::vector<float>& scales,
                                                            std::vector<float>& output) const {
    if (min_vals.empty() || max_vals.empty() || scales.empty()) return;

    size_t value_count = compressed_data.size() / 2;
    output.resize(value_count);

    float global_min = min_vals[0];
    float scale = scales[0];

    for (size_t i = 0; i < value_count; ++i) {
        uint16_t quantized = compressed_data[i * 2] | (static_cast<uint16_t>(compressed_data[i * 2 + 1]) << 8);
        output[i] = global_min + static_cast<float>(quantized) / scale;
    }
}

inline void CompressedDifferentiableTape::extractStateFromBodies(const BodySystem& bodies,
                                                                std::vector<float>& state_data) {
    // Reserve space for positions and velocities
    state_data.clear();
    state_data.reserve(n_bodies * 6);  // 3 pos + 3 vel per body

    // This would need to be implemented to extract data from BodySystem
    // For now, create placeholder implementation
    state_data.resize(n_bodies * 6, 0.0f);
}

inline void CompressedDifferentiableTape::loadStateIntoBodies(const std::vector<float>& state_data,
    std::vector<float>& pos_x, std::vector<float>& pos_y, std::vector<float>& pos_z,
    std::vector<float>& vel_x, std::vector<float>& vel_y, std::vector<float>& vel_z) const {

    pos_x.resize(n_bodies);
    pos_y.resize(n_bodies);
    pos_z.resize(n_bodies);
    vel_x.resize(n_bodies);
    vel_y.resize(n_bodies);
    vel_z.resize(n_bodies);

    for (size_t i = 0; i < n_bodies && i * 6 + 5 < state_data.size(); ++i) {
        pos_x[i] = state_data[i * 6 + 0];
        pos_y[i] = state_data[i * 6 + 1];
        pos_z[i] = state_data[i * 6 + 2];
        vel_x[i] = state_data[i * 6 + 3];
        vel_y[i] = state_data[i * 6 + 4];
        vel_z[i] = state_data[i * 6 + 5];
    }
}

inline bool CompressedDifferentiableTape::shouldCreateCheckpoint(int step_index) const {
    switch (config.mode) {
        case CompressionMode::SPARSE:
            return step_index % config.checkpoint_interval == 0;

        case CompressionMode::ADAPTIVE: {
            // Create checkpoint if we haven't had one in a while
            if (checkpoint_indices.empty()) return true;
            int last_checkpoint = checkpoint_indices.back();
            return (step_index - compressed_states[last_checkpoint].step_index) >= config.checkpoint_interval;
        }

        case CompressionMode::NONE:
            return false;  // Every state is effectively a checkpoint

        case CompressionMode::DELTA:
            return step_index % config.checkpoint_interval == 0;

        case CompressionMode::QUANTIZED:
            return step_index % config.checkpoint_interval == 0;

        default:
            return step_index % config.checkpoint_interval == 0;
    }
}

inline int CompressedDifferentiableTape::findNearestCheckpoint(int step_index) const {
    int nearest = -1;
    for (int checkpoint_idx : checkpoint_indices) {
        if (compressed_states[checkpoint_idx].step_index <= step_index) {
            nearest = checkpoint_idx;
        } else {
            break;
        }
    }
    return nearest;
}

inline size_t CompressedDifferentiableTape::estimateStateSize() const {
    return n_bodies * 6 * sizeof(float);  // 6 floats per body (pos + vel)
}

} // namespace physgrad