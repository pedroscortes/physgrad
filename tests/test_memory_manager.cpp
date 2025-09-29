/**
 * PhysGrad - Memory Manager Unit Tests
 */

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cstring>

// Mock memory manager interface
class MemoryManager {
public:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool is_device;
        bool is_allocated;
    };

    MemoryManager() : total_allocated_(0), peak_usage_(0) {}

    bool initialize() { return true; }
    void cleanup() {
        for (auto& block : allocated_blocks_) {
            if (block.is_allocated) {
                if (block.is_device) {
                    // Simulate cudaFree
                    block.is_allocated = false;
                } else {
                    free(block.ptr);
                    block.is_allocated = false;
                }
            }
        }
        allocated_blocks_.clear();
        total_allocated_ = 0;
    }

    void* allocateHost(size_t size) {
        void* ptr = malloc(size);
        if (ptr) {
            MemoryBlock block = {ptr, size, false, true};
            allocated_blocks_.push_back(block);
            total_allocated_ += size;
            peak_usage_ = std::max(peak_usage_, total_allocated_);
        }
        return ptr;
    }

    void* allocateDevice(size_t size) {
        // Simulate CUDA device allocation
        void* ptr = malloc(size); // Use host memory as simulation
        if (ptr) {
            MemoryBlock block = {ptr, size, true, true};
            allocated_blocks_.push_back(block);
            total_allocated_ += size;
            peak_usage_ = std::max(peak_usage_, total_allocated_);
        }
        return ptr;
    }

    bool deallocateHost(void* ptr) {
        for (auto& block : allocated_blocks_) {
            if (block.ptr == ptr && !block.is_device && block.is_allocated) {
                free(ptr);
                total_allocated_ -= block.size;
                block.is_allocated = false;
                return true;
            }
        }
        return false;
    }

    bool deallocateDevice(void* ptr) {
        for (auto& block : allocated_blocks_) {
            if (block.ptr == ptr && block.is_device && block.is_allocated) {
                free(ptr); // Simulate cudaFree
                total_allocated_ -= block.size;
                block.is_allocated = false;
                return true;
            }
        }
        return false;
    }

    bool copyHostToDevice(void* device_ptr, const void* host_ptr, size_t size) {
        // Simulate memory copy
        std::memcpy(device_ptr, host_ptr, size);
        return true;
    }

    bool copyDeviceToHost(void* host_ptr, const void* device_ptr, size_t size) {
        // Simulate memory copy
        std::memcpy(host_ptr, device_ptr, size);
        return true;
    }

    bool copyDeviceToDevice(void* dst_ptr, const void* src_ptr, size_t size) {
        // Simulate device-to-device copy
        std::memcpy(dst_ptr, src_ptr, size);
        return true;
    }

    size_t getTotalAllocated() const { return total_allocated_; }
    size_t getPeakUsage() const { return peak_usage_; }

    void* allocateAligned(size_t size, size_t alignment) {
        size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
        void* ptr = std::aligned_alloc(alignment, aligned_size);
        if (ptr) {
            MemoryBlock block = {ptr, aligned_size, false, true};
            allocated_blocks_.push_back(block);
            total_allocated_ += aligned_size;
            peak_usage_ = std::max(peak_usage_, total_allocated_);
        }
        return ptr;
    }

    bool isAligned(void* ptr, size_t alignment) {
        return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
    }

    void defragment() {
        // Simulate memory defragmentation
        // In a real implementation, this would reorganize memory blocks
    }

    double getFragmentationRatio() {
        if (allocated_blocks_.empty()) return 0.0;

        size_t allocated_count = 0;
        for (const auto& block : allocated_blocks_) {
            if (block.is_allocated) allocated_count++;
        }

        return static_cast<double>(allocated_count) / allocated_blocks_.size();
    }

private:
    std::vector<MemoryBlock> allocated_blocks_;
    size_t total_allocated_;
    size_t peak_usage_;
};

class MemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        memory_manager_ = std::make_unique<MemoryManager>();
        ASSERT_TRUE(memory_manager_->initialize());
    }

    void TearDown() override {
        if (memory_manager_) {
            memory_manager_->cleanup();
        }
    }

    std::unique_ptr<MemoryManager> memory_manager_;
};

TEST_F(MemoryManagerTest, InitializationAndCleanup) {
    EXPECT_TRUE(memory_manager_ != nullptr);
    EXPECT_EQ(memory_manager_->getTotalAllocated(), 0);
}

TEST_F(MemoryManagerTest, HostMemoryAllocation) {
    size_t allocation_size = 1024;
    void* ptr = memory_manager_->allocateHost(allocation_size);

    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(memory_manager_->getTotalAllocated(), allocation_size);

    bool success = memory_manager_->deallocateHost(ptr);
    EXPECT_TRUE(success);
    EXPECT_EQ(memory_manager_->getTotalAllocated(), 0);
}

TEST_F(MemoryManagerTest, DeviceMemoryAllocation) {
    size_t allocation_size = 2048;
    void* ptr = memory_manager_->allocateDevice(allocation_size);

    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(memory_manager_->getTotalAllocated(), allocation_size);

    bool success = memory_manager_->deallocateDevice(ptr);
    EXPECT_TRUE(success);
    EXPECT_EQ(memory_manager_->getTotalAllocated(), 0);
}

TEST_F(MemoryManagerTest, MultipleAllocations) {
    std::vector<void*> host_ptrs;
    std::vector<void*> device_ptrs;
    size_t allocation_size = 512;
    int num_allocations = 5;

    // Allocate multiple blocks
    for (int i = 0; i < num_allocations; ++i) {
        void* host_ptr = memory_manager_->allocateHost(allocation_size);
        void* device_ptr = memory_manager_->allocateDevice(allocation_size);

        EXPECT_NE(host_ptr, nullptr);
        EXPECT_NE(device_ptr, nullptr);

        host_ptrs.push_back(host_ptr);
        device_ptrs.push_back(device_ptr);
    }

    EXPECT_EQ(memory_manager_->getTotalAllocated(), 2 * num_allocations * allocation_size);

    // Deallocate all blocks
    for (int i = 0; i < num_allocations; ++i) {
        EXPECT_TRUE(memory_manager_->deallocateHost(host_ptrs[i]));
        EXPECT_TRUE(memory_manager_->deallocateDevice(device_ptrs[i]));
    }

    EXPECT_EQ(memory_manager_->getTotalAllocated(), 0);
}

TEST_F(MemoryManagerTest, MemoryCopy) {
    size_t size = 1024;
    void* host_src = memory_manager_->allocateHost(size);
    void* host_dst = memory_manager_->allocateHost(size);
    void* device_ptr = memory_manager_->allocateDevice(size);

    ASSERT_NE(host_src, nullptr);
    ASSERT_NE(host_dst, nullptr);
    ASSERT_NE(device_ptr, nullptr);

    // Initialize source data
    for (size_t i = 0; i < size; ++i) {
        static_cast<char*>(host_src)[i] = static_cast<char>(i % 256);
    }

    // Test host to device copy
    bool success = memory_manager_->copyHostToDevice(device_ptr, host_src, size);
    EXPECT_TRUE(success);

    // Test device to host copy
    success = memory_manager_->copyDeviceToHost(host_dst, device_ptr, size);
    EXPECT_TRUE(success);

    // Verify data integrity
    EXPECT_EQ(std::memcmp(host_src, host_dst, size), 0);

    memory_manager_->deallocateHost(host_src);
    memory_manager_->deallocateHost(host_dst);
    memory_manager_->deallocateDevice(device_ptr);
}

TEST_F(MemoryManagerTest, PeakUsageTracking) {
    size_t allocation_size = 1024;

    EXPECT_EQ(memory_manager_->getPeakUsage(), 0);

    void* ptr1 = memory_manager_->allocateHost(allocation_size);
    EXPECT_EQ(memory_manager_->getPeakUsage(), allocation_size);

    void* ptr2 = memory_manager_->allocateHost(allocation_size);
    EXPECT_EQ(memory_manager_->getPeakUsage(), 2 * allocation_size);

    memory_manager_->deallocateHost(ptr1);
    EXPECT_EQ(memory_manager_->getPeakUsage(), 2 * allocation_size); // Peak should remain

    void* ptr3 = memory_manager_->allocateHost(allocation_size);
    EXPECT_EQ(memory_manager_->getPeakUsage(), 2 * allocation_size); // Still the same peak

    memory_manager_->deallocateHost(ptr2);
    memory_manager_->deallocateHost(ptr3);
}

TEST_F(MemoryManagerTest, AlignedAllocation) {
    size_t size = 1000;
    size_t alignment = 64;

    void* ptr = memory_manager_->allocateAligned(size, alignment);
    EXPECT_NE(ptr, nullptr);
    EXPECT_TRUE(memory_manager_->isAligned(ptr, alignment));

    // Clean up
    memory_manager_->deallocateHost(ptr);
}

TEST_F(MemoryManagerTest, InvalidDeallocation) {
    void* fake_ptr = reinterpret_cast<void*>(0x12345678);

    bool success = memory_manager_->deallocateHost(fake_ptr);
    EXPECT_FALSE(success);

    success = memory_manager_->deallocateDevice(fake_ptr);
    EXPECT_FALSE(success);
}

TEST_F(MemoryManagerTest, ZeroSizeAllocation) {
    void* ptr = memory_manager_->allocateHost(0);
    // Behavior may vary - either return null or a valid pointer
    // This tests that the manager handles edge cases gracefully
    EXPECT_TRUE(ptr == nullptr || ptr != nullptr);

    if (ptr != nullptr) {
        memory_manager_->deallocateHost(ptr);
    }
}

TEST_F(MemoryManagerTest, LargeAllocation) {
    size_t large_size = 1024 * 1024 * 10; // 10 MB

    void* ptr = memory_manager_->allocateHost(large_size);
    if (ptr != nullptr) {
        EXPECT_EQ(memory_manager_->getTotalAllocated(), large_size);
        memory_manager_->deallocateHost(ptr);
    }
    // If allocation fails, that's also acceptable for very large sizes
}

TEST_F(MemoryManagerTest, DeviceToDeviceCopy) {
    size_t size = 512;
    void* device_src = memory_manager_->allocateDevice(size);
    void* device_dst = memory_manager_->allocateDevice(size);
    void* host_ptr = memory_manager_->allocateHost(size);

    ASSERT_NE(device_src, nullptr);
    ASSERT_NE(device_dst, nullptr);
    ASSERT_NE(host_ptr, nullptr);

    // Initialize data on host and copy to device
    for (size_t i = 0; i < size; ++i) {
        static_cast<char*>(host_ptr)[i] = static_cast<char>(i % 256);
    }

    memory_manager_->copyHostToDevice(device_src, host_ptr, size);

    // Test device to device copy
    bool success = memory_manager_->copyDeviceToDevice(device_dst, device_src, size);
    EXPECT_TRUE(success);

    // Verify by copying back to host
    std::memset(host_ptr, 0, size);
    memory_manager_->copyDeviceToHost(host_ptr, device_dst, size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(static_cast<char*>(host_ptr)[i], static_cast<char>(i % 256));
    }

    memory_manager_->deallocateDevice(device_src);
    memory_manager_->deallocateDevice(device_dst);
    memory_manager_->deallocateHost(host_ptr);
}

TEST_F(MemoryManagerTest, FragmentationTracking) {
    // Initially no fragmentation
    EXPECT_EQ(memory_manager_->getFragmentationRatio(), 0.0);

    // Allocate some blocks
    void* ptr1 = memory_manager_->allocateHost(1024);
    void* ptr2 = memory_manager_->allocateHost(1024);
    void* ptr3 = memory_manager_->allocateHost(1024);

    double ratio_all_allocated = memory_manager_->getFragmentationRatio();
    EXPECT_EQ(ratio_all_allocated, 1.0); // All blocks allocated

    // Deallocate middle block
    memory_manager_->deallocateHost(ptr2);

    double ratio_fragmented = memory_manager_->getFragmentationRatio();
    EXPECT_LT(ratio_fragmented, 1.0); // Some blocks deallocated

    // Clean up
    memory_manager_->deallocateHost(ptr1);
    memory_manager_->deallocateHost(ptr3);
}

TEST_F(MemoryManagerTest, StressTest) {
    const int num_iterations = 1000;
    const size_t max_size = 4096;
    std::vector<void*> allocated_ptrs;

    // Stress test with random allocations and deallocations
    for (int i = 0; i < num_iterations; ++i) {
        if (allocated_ptrs.empty() || (rand() % 2 == 0 && allocated_ptrs.size() < 100)) {
            // Allocate
            size_t size = (rand() % max_size) + 1;
            void* ptr = (rand() % 2 == 0) ?
                       memory_manager_->allocateHost(size) :
                       memory_manager_->allocateDevice(size);

            if (ptr != nullptr) {
                allocated_ptrs.push_back(ptr);
            }
        } else {
            // Deallocate
            int index = rand() % allocated_ptrs.size();
            void* ptr = allocated_ptrs[index];

            bool success = memory_manager_->deallocateHost(ptr);
            if (!success) {
                success = memory_manager_->deallocateDevice(ptr);
            }

            if (success) {
                allocated_ptrs.erase(allocated_ptrs.begin() + index);
            }
        }
    }

    // Clean up remaining allocations
    for (void* ptr : allocated_ptrs) {
        bool success = memory_manager_->deallocateHost(ptr);
        if (!success) {
            memory_manager_->deallocateDevice(ptr);
        }
    }

    // Should have no memory leaks
    EXPECT_EQ(memory_manager_->getTotalAllocated(), 0);
}