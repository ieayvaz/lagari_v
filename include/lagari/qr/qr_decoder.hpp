#pragma once

#include "lagari/core/module.hpp"
#include "lagari/core/types.hpp"
#include <memory>
#include <string>

namespace lagari {

class Config;

/**
 * @brief QR decoder interface
 * 
 * Decodes QR codes from frames or detection ROIs using ZBar.
 */
class IQRDecoder : public IModule {
public:
    virtual ~IQRDecoder() = default;

    /**
     * @brief Decode QR code from detection ROI
     * 
     * @param frame Source frame
     * @param detection Detection with QR code bounding box
     * @return Decoded QR result
     */
    virtual QRResult decode(const Frame& frame, const Detection& detection) = 0;

    /**
     * @brief Decode QR code from full frame
     * 
     * Scans entire frame for QR codes.
     * 
     * @param frame Source frame
     * @return Decoded QR result (first found)
     */
    virtual QRResult decode_full_frame(const Frame& frame) = 0;

    /**
     * @brief Get cached result for frame
     * 
     * Returns cached result if frame was recently decoded.
     * 
     * @param frame_id Frame ID to check
     * @return Cached result or empty result
     */
    virtual QRResult get_cached(uint64_t frame_id) = 0;

    /**
     * @brief Clear decode cache
     */
    virtual void clear_cache() = 0;

    /**
     * @brief Get decode statistics
     */
    struct QRStats {
        uint64_t attempts = 0;
        uint64_t successes = 0;
        uint64_t cache_hits = 0;
        Duration average_decode_time{0};
    };
    virtual QRStats get_stats() const = 0;
};

/**
 * @brief Create QR decoder
 */
std::unique_ptr<IQRDecoder> create_qr_decoder(const Config& config);

}  // namespace lagari
