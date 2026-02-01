#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "lagari/core/spsc_queue.hpp"

#include <lagari/core/config.hpp>

#include <thread>
#include <vector>
#include <numeric>

using namespace lagari;

class SPSCQueueTest : public ::testing::Test {
protected:
    static constexpr size_t kCapacity = 8;
    SPSCQueue<int, kCapacity> queue;
};

TEST_F(SPSCQueueTest, InitiallyEmpty) {
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
    EXPECT_EQ(queue.capacity(), kCapacity - 1);  // One slot always empty
}

TEST_F(SPSCQueueTest, PushAndPop) {
    EXPECT_TRUE(queue.try_push(42));
    EXPECT_FALSE(queue.empty());
    EXPECT_EQ(queue.size(), 1);
    
    auto val = queue.try_pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42);
    EXPECT_TRUE(queue.empty());
}

TEST_F(SPSCQueueTest, FIFO) {
    queue.try_push(1);
    queue.try_push(2);
    queue.try_push(3);
    
    EXPECT_EQ(*queue.try_pop(), 1);
    EXPECT_EQ(*queue.try_pop(), 2);
    EXPECT_EQ(*queue.try_pop(), 3);
}

TEST_F(SPSCQueueTest, FullQueue) {
    // Fill to capacity
    for (size_t i = 0; i < queue.capacity(); ++i) {
        EXPECT_TRUE(queue.try_push(static_cast<int>(i)));
    }
    
    // Should be full now
    EXPECT_FALSE(queue.try_push(999));
    EXPECT_EQ(queue.size(), queue.capacity());
}

TEST_F(SPSCQueueTest, PushOverwrite) {
    // Fill queue
    for (size_t i = 0; i < queue.capacity(); ++i) {
        queue.try_push(static_cast<int>(i));
    }
    
    // Overwrite should work but return false
    EXPECT_FALSE(queue.push_overwrite(100));
    
    // First pop should not be 0 anymore (it was overwritten)
    auto val = queue.try_pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_NE(*val, 0);
}

TEST_F(SPSCQueueTest, Peek) {
    queue.try_push(42);
    
    const int* peeked = queue.peek();
    ASSERT_NE(peeked, nullptr);
    EXPECT_EQ(*peeked, 42);
    
    // Peek doesn't remove
    EXPECT_EQ(queue.size(), 1);
}

TEST_F(SPSCQueueTest, PopAll) {
    queue.try_push(1);
    queue.try_push(2);
    queue.try_push(3);
    
    std::vector<int> results;
    size_t count = queue.pop_all(std::back_inserter(results));
    
    EXPECT_EQ(count, 3);
    EXPECT_THAT(results, ::testing::ElementsAre(1, 2, 3));
    EXPECT_TRUE(queue.empty());
}

TEST_F(SPSCQueueTest, ProducerConsumer) {
    constexpr int kNumItems = 10000;
    std::atomic<bool> done{false};
    std::vector<int> received;
    received.reserve(kNumItems);
    
    // Producer
    std::thread producer([this, &done]() {
        for (int i = 0; i < kNumItems; ++i) {
            while (!queue.try_push(i)) {
                std::this_thread::yield();
            }
        }
        done.store(true, std::memory_order_release);
    });
    
    // Consumer
    std::thread consumer([this, &done, &received]() {
        while (!done.load(std::memory_order_acquire) || !queue.empty()) {
            if (auto val = queue.try_pop()) {
                received.push_back(*val);
            } else {
                std::this_thread::yield();
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    // Verify all items received in order
    EXPECT_EQ(received.size(), kNumItems);
    for (int i = 0; i < kNumItems; ++i) {
        EXPECT_EQ(received[i], i);
    }
}

TEST_F(SPSCQueueTest, MoveSemantics) {
    struct MoveOnly {
        int value;
        MoveOnly() = default;
        MoveOnly(int v) : value(v) {}
        MoveOnly(const MoveOnly&) = delete;
        MoveOnly& operator=(const MoveOnly&) = delete;
        MoveOnly(MoveOnly&& other) noexcept : value(other.value) { other.value = -1; }
        MoveOnly& operator=(MoveOnly&& other) noexcept {
            value = other.value;
            other.value = -1;
            return *this;
        }
    };
    
    SPSCQueue<MoveOnly, 4> mo_queue;
    
    mo_queue.try_push(MoveOnly{42});
    
    auto popped = mo_queue.try_pop();
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(popped->value, 42);
}
