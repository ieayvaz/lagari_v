#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <new>
#include <type_traits>

namespace lagari {

/**
 * @brief Single-Producer Single-Consumer Lock-Free Queue
 * 
 * Designed for lightweight message passing between exactly two threads:
 * - Detection results from detector to guidance
 * - QR results from decoder to telemetry
 * - Commands from guidance to MAVLink
 * 
 * Uses Lamport's algorithm with cache-line padding to prevent false sharing.
 * 
 * @tparam T Type of elements (should be trivially copyable or moveable)
 * @tparam Capacity Queue capacity (must be power of 2)
 */
template<typename T, size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity >= 2, "Capacity must be at least 2");

public:
    SPSCQueue() = default;

    // Non-copyable, non-moveable
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;
    SPSCQueue(SPSCQueue&&) = delete;
    SPSCQueue& operator=(SPSCQueue&&) = delete;

    /**
     * @brief Try to push an item (producer only)
     * 
     * @param item Item to push
     * @return true if successful, false if queue is full
     */
    bool try_push(const T& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (head + 1) & (Capacity - 1);
        
        // Check if queue is full
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        
        buffer_[head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    /**
     * @brief Try to push an item with move semantics (producer only)
     * 
     * @param item Item to push
     * @return true if successful, false if queue is full
     */
    bool try_push(T&& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (head + 1) & (Capacity - 1);
        
        // Check if queue is full
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        
        buffer_[head] = std::move(item);
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    /**
     * @brief Push item, overwriting oldest if full (producer only)
     * 
     * @param item Item to push
     * @return true if pushed without overwrite, false if overwrite occurred
     */
    bool push_overwrite(T item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (head + 1) & (Capacity - 1);
        
        bool overwrite = (next_head == tail_.load(std::memory_order_acquire));
        
        if (overwrite) {
            // Advance tail to make room
            size_t new_tail = (next_head + 1) & (Capacity - 1);
            tail_.store(new_tail, std::memory_order_release);
        }
        
        buffer_[head] = std::move(item);
        head_.store(next_head, std::memory_order_release);
        return !overwrite;
    }

    /**
     * @brief Try to pop an item (consumer only)
     * 
     * @return Item if available, nullopt if queue is empty
     */
    std::optional<T> try_pop() {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        
        // Check if queue is empty
        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        
        T item = std::move(buffer_[tail]);
        tail_.store((tail + 1) & (Capacity - 1), std::memory_order_release);
        return item;
    }

    /**
     * @brief Try to peek at front item without removing (consumer only)
     * 
     * @return Pointer to front item if available, nullptr if empty
     */
    const T* peek() const {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        
        if (tail == head_.load(std::memory_order_acquire)) {
            return nullptr;
        }
        
        return &buffer_[tail];
    }

    /**
     * @brief Pop all available items (consumer only)
     * 
     * @tparam OutputIt Output iterator type
     * @param out Output iterator to write items to
     * @return Number of items popped
     */
    template<typename OutputIt>
    size_t pop_all(OutputIt out) {
        size_t count = 0;
        while (auto item = try_pop()) {
            *out++ = std::move(*item);
            ++count;
        }
        return count;
    }

    /**
     * @brief Check if queue is empty
     */
    bool empty() const {
        return head_.load(std::memory_order_acquire) == 
               tail_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get approximate number of items in queue
     */
    size_t size() const {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail) & (Capacity - 1);
    }

    /**
     * @brief Get queue capacity
     */
    constexpr size_t capacity() const {
        return Capacity - 1;  // One slot always empty
    }

    /**
     * @brief Clear all items (not thread-safe, use only during init/shutdown)
     */
    void clear() {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

private:
    // Cache line alignment to prevent false sharing between producer and consumer
    static constexpr size_t kCacheLineSize = 64;

    alignas(kCacheLineSize) std::atomic<size_t> head_{0};
    alignas(kCacheLineSize) std::atomic<size_t> tail_{0};
    
    alignas(kCacheLineSize) std::array<T, Capacity> buffer_;
};

/**
 * @brief Type alias for common queue sizes
 */
template<typename T>
using DetectionQueue = SPSCQueue<T, 16>;

template<typename T>
using CommandQueue = SPSCQueue<T, 32>;

template<typename T>
using QRQueue = SPSCQueue<T, 8>;

}  // namespace lagari
