// Video recorder
// TODO: Implement video recording with hardware acceleration

#include "lagari/recording/recorder.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

namespace lagari {

std::unique_ptr<IRecorder> create_recorder(const Config& config) {
    (void)config;
    LOG_INFO("Recorder not yet implemented");
    return nullptr;
}

std::unique_ptr<IRecorder> create_recorder(const RecordingConfig& config) {
    (void)config;
    return nullptr;
}

}  // namespace lagari
