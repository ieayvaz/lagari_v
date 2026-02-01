// QR Decoder using ZBar
// TODO: Implement full QR decoder

#include "lagari/qr/qr_decoder.hpp"
#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

namespace lagari {

std::unique_ptr<IQRDecoder> create_qr_decoder(const Config& config) {
    (void)config;
    LOG_INFO("QR decoder not yet implemented");
    return nullptr;
}

}  // namespace lagari
