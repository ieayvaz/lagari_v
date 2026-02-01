#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "lagari/core/ring_buffer.hpp"
#include "lagari/core/config.hpp"

#include <thread>
#include <vector>
#include <atomic>
#include <memory>

using namespace lagari;

class RingBufferTest : public ::testing::Test {
protected:
    static constexpr size_t kCapacity = 4;
    SPMCRingBuffer<int, kCapacity> buffer;
};

TEST_F(RingBufferTest, InitiallyEmpty) {
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.capacity(), kCapacity);
}

TEST_F(RingBufferTest, PushAndPop) {
    EXPECT_TRUE(buffer.push(42));
    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.size(), 1);
    
    auto val = buffer.try_pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42);
    EXPECT_TRUE(buffer.empty());
}

TEST_F(RingBufferTest, PeekLatest) {
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    
    auto latest = buffer.peek_latest();
    ASSERT_TRUE(latest.has_value());
    EXPECT_EQ(*latest, 3);
    
    // Peek doesn't remove
    EXPECT_EQ(buffer.size(), 3);
}

TEST_F(RingBufferTest, OverwriteOnFull) {
    // Fill buffer
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    buffer.push(4);
    EXPECT_EQ(buffer.size(), kCapacity);
    
    // Push one more - should overwrite oldest
    bool no_overwrite = buffer.push(5);
    EXPECT_FALSE(no_overwrite);  // Overwrite occurred
    
    // Pop should give 2 (oldest after overwrite)
    auto val = buffer.try_pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 2);
}

TEST_F(RingBufferTest, PopFromEmpty) {
    auto val = buffer.try_pop();
    EXPECT_FALSE(val.has_value());
}

TEST_F(RingBufferTest, MultipleProducerConsumer) {
    constexpr int kNumItems = 1000;
    std::atomic<int> sum{0};
    std::atomic<bool> producer_done{false};
    
    // Producer thread
    std::thread producer([this, &producer_done]() {
        for (int i = 1; i <= kNumItems; ++i) {
            buffer.push(i);
            if (i % 100 == 0) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });
    
    // Consumer threads
    std::vector<std::thread> consumers;
    for (int c = 0; c < 2; ++c) {
        consumers.emplace_back([this, &sum, &producer_done]() {
            while (!producer_done.load(std::memory_order_acquire) || !buffer.empty()) {
                if (auto val = buffer.try_pop()) {
                    sum.fetch_add(*val, std::memory_order_relaxed);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }
    
    producer.join();
    for (auto& t : consumers) {
        t.join();
    }
    
    // Due to overwrites, sum might be less than expected
    // But it should be positive
    EXPECT_GT(sum.load(), 0);
}

// Test with shared_ptr (common use case for frames)
TEST(RingBufferSharedPtrTest, SharedPtrSupport) {
    SPMCRingBuffer<std::shared_ptr<int>, 4> buffer;
    
    auto ptr1 = std::make_shared<int>(100);
    auto ptr2 = std::make_shared<int>(200);
    
    buffer.push(ptr1);
    buffer.push(ptr2);
    
    EXPECT_EQ(ptr1.use_count(), 2);  // Original + buffer
    
    auto popped = buffer.try_pop();
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(**popped, 100);
    EXPECT_EQ(ptr1.use_count(), 2);  // Original + popped
    
    auto latest = buffer.peek_latest();
    ASSERT_TRUE(latest.has_value());
    EXPECT_EQ(**latest, 200);
}
