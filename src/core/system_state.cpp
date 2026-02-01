#include "lagari/core/system_state.hpp"

namespace lagari {

SystemStateMachine& global_state() {
    static SystemStateMachine instance;
    return instance;
}

}  // namespace lagari
