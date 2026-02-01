#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace lagari {

/**
 * @brief Single-Producer Multi-Consumer Lock-Free Ring Buffer
 * 
 * Designed for high-throughput frame passing where:
 * - One producer (camera thread) pushes frames
 * - Multiple consumers (detection, recording, telemetry) read frames
 * - Old frames can be overwritten if buffer is full (latency > completeness)
 * 
 * Uses sequence-based design for lock-free operation.
 * 
 * @tparam T Type of elements (should be moveable, typically std::shared_ptr<Frame>)
 * @tparam Capacity Buffer capacity (must be power of 2)
 */
template<typename T, size_t Capacity>
class SPMCRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity >= 2, "Capacity must be at least 2");

public:
    SPMCRingBuffer() {
        for (size_t i = 0; i < Capacity; ++i) {
            sequence_[i].store(i, std::memory_order_relaxed);
        }
    }

    // Non-copyable, non-moveable
    SPMCRingBuffer(const SPMCRingBuffer&) = delete;
    SPMCRingBuffer& operator=(const SPMCRingBuffer&) = delete;
    SPMCRingBuffer(SPMCRingBuffer&&) = delete;
    SPMCRingBuffer& operator=(SPMCRingBuffer&&) = delete;

    /**
     * @brief Push item to buffer (producer only)
     * 
     * If buffer is full, overwrites oldest item.
     * 
     * @param item Item to push
     * @return true if pushed successfully, false if overwrite occurred
     */
    bool push(T item) {
        const size_t pos = head_.load(std::memory_order_relaxed);
        const size_t index = pos & (Capacity - 1);
        
        size_t seq = sequence_[index].load(std::memory_order_acquire);
        
        // Check if slot is ready for writing
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
        
        bool overwrite = (diff < 0);
        
        // Store the item
        buffer_[index] = std::move(item);
        
        // Update sequence to mark as written
        sequence_[index].store(pos + 1, std::memory_order_release);
        
        // Advance head
        head_.store(pos + 1, std::memory_order_release);
        
        // Advance tail if we overwrote
        if (overwrite) {
            size_t expected = pos - Capacity + 1;
            tail_.compare_exchange_strong(expected, pos - Capacity + 2,
                                         std::memory_order_release,
                                         std::memory_order_relaxed);
        }
        
        return !overwrite;
    }

    /**
     * @brief Try to pop oldest item from buffer (consumer)
     * 
     * @return Item if available, nullopt otherwise
     */
    std::optional<T> try_pop() {
        size_t pos = tail_.load(std::memory_order_relaxed);
        
        while (true) {
            const size_t index = pos & (Capacity - 1);
            size_t seq = sequence_[index].load(std::memory_order_acquire);
            
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            
            if (diff < 0) {
                // Buffer is empty
                return std::nullopt;
            }
            
            if (diff == 0) {
                // Try to claim this slot
                if (tail_.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_release,
                                               std::memory_order_relaxed)) {
                    T item = std::move(buffer_[index]);
                    sequence_[index].store(pos + Capacity, std::memory_order_release);
                    return item;
                }
            } else {
                // Someone else took it, retry with new tail
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
    }

    /**
     * @brief Get the latest item without removing it
     * 
     * Useful for consumers that want the most recent frame
     * regardless of what they've seen before.
     * 
     * @return Latest item if available, nullopt if empty
     */
    std::optional<T> peek_latest() const {
        size_t head = head_.load(std::memory_order_acquire);
        if (head == 0) return std::nullopt;
        
        const size_t index = (head - 1) & (Capacity - 1);
        size_t seq = sequence_[index].load(std::memory_order_acquire);
        
        // Check if this slot has valid data
        if (seq == head) {
            return buffer_[index];
        }
        
        return std::nullopt;
    }

    /**
     * @brief Get approximate number of items in buffer
     */
    size_t size() const {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_relaxed);
        return (head >= tail) ? (head - tail) : 0;
    }

    /**
     * @brief Check if buffer is empty
     */
    bool empty() const {
        return size() == 0;
    }

    /**
     * @brief Get buffer capacity
     */
    constexpr size_t capacity() const {
        return Capacity;
    }

private:
    // Cache line alignment to prevent false sharing
    static constexpr size_t kCacheLineSize = 64;

    alignas(kCacheLineSize) std::atomic<size_t> head_{0};
    alignas(kCacheLineSize) std::atomic<size_t> tail_{0};
    
    std::array<T, Capacity> buffer_;
    std::array<std::atomic<size_t>, Capacity> sequence_;
};

}  // namespace lagari
