#pragma once

#include "vramz/types.hpp"

#include <cstdint>

namespace vramz {

struct TierUsage final {
    ByteSize committed{};
    ByteSize reserved{};
    ByteSize staging{};
    ByteSize workspace{};
    ByteSize cleanup_debt{};
    ByteSize peak_charged{};
};

struct AsyncErrorStats final {
    std::uint32_t capacity{};
    std::uint32_t retained{};
    std::uint64_t produced{};
    std::uint64_t coalesced{};
    std::uint64_t dropped{};
    std::uint64_t diagnostic_text_failures{};
};

struct CompressionStats final {
    ByteSize logical_bytes_represented{};
    ByteSize stored_payload_bytes{};
    ByteSize physical_storage_bytes{};
    std::uint64_t successful_compressions{};
    std::uint64_t successful_decompressions{};
    std::uint64_t integrity_failures{};
    ByteSize bytes_compressed{};
    ByteSize bytes_decompressed{};

    [[nodiscard]] double physical_compression_ratio() const noexcept {
        return physical_storage_bytes.value() == 0U
                   ? 0.0
                   : static_cast<double>(logical_bytes_represented.value()) /
                         static_cast<double>(physical_storage_bytes.value());
    }
};

} // namespace vramz
