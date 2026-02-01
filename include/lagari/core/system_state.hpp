#pragma once

#include "lagari/core/types.hpp"
#include <atomic>
#include <functional>
#include <mutex>
#include <vector>

namespace lagari {

/**
 * @brief System state machine
 * 
 * Manages state transitions and notifies subscribers of state changes.
 * Thread-safe for concurrent access.
 */
class SystemStateMachine {
public:
    using StateCallback = std::function<void(SystemState old_state, SystemState new_state)>;

    SystemStateMachine() : state_(SystemState::INIT) {}

    /**
     * @brief Get current state
     */
    SystemState state() const {
        return state_.load(std::memory_order_acquire);
    }

    /**
     * @brief Set state directly
     * 
     * @param new_state New state
     * @return true if state changed
     */
    bool set_state(SystemState new_state) {
        SystemState old_state = state_.exchange(new_state, std::memory_order_acq_rel);
        
        if (old_state != new_state) {
            notify_state_change(old_state, new_state);
            return true;
        }
        return false;
    }

    /**
     * @brief Transition to new state if current state matches expected
     * 
     * @param expected Expected current state
     * @param new_state State to transition to
     * @return true if transition occurred
     */
    bool transition(SystemState expected, SystemState new_state) {
        if (state_.compare_exchange_strong(expected, new_state,
                                           std::memory_order_acq_rel,
                                           std::memory_order_acquire)) {
            notify_state_change(expected, new_state);
            return true;
        }
        return false;
    }

    /**
     * @brief Check if in specific state
     */
    bool is_state(SystemState s) const {
        return state() == s;
    }

    /**
     * @brief Check if system is in an active processing state
     */
    bool is_active() const {
        SystemState s = state();
        return s == SystemState::SEARCHING || 
               s == SystemState::DETECTED || 
               s == SystemState::TRACKING ||
               s == SystemState::LANDING;
    }

    /**
     * @brief Check if system can be stopped
     */
    bool is_stoppable() const {
        SystemState s = state();
        return s != SystemState::INIT && s != SystemState::SHUTDOWN;
    }

    /**
     * @brief Register callback for state changes
     * 
     * @param callback Function to call on state change
     */
    void on_state_change(StateCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        callbacks_.push_back(std::move(callback));
    }

    /**
     * @brief Get state as string
     */
    const char* state_string() const {
        return to_string(state());
    }

private:
    std::atomic<SystemState> state_;
    std::mutex callback_mutex_;
    std::vector<StateCallback> callbacks_;

    void notify_state_change(SystemState old_state, SystemState new_state) {
        std::vector<StateCallback> callbacks_copy;
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            callbacks_copy = callbacks_;
        }
        
        for (const auto& cb : callbacks_copy) {
            cb(old_state, new_state);
        }
    }
};

/**
 * @brief Global state machine instance
 */
SystemStateMachine& global_state();

}  // namespace lagari
