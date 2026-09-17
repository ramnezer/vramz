#pragma once

#include "vramz/result.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace vramz::detail {

enum class CodecId : std::uint8_t { none, lz4_block };

class CompressionCodec {
  public:
    virtual ~CompressionCodec() = default;

    [[nodiscard]] virtual CodecId id() const noexcept = 0;
    [[nodiscard]] virtual bool available() const noexcept = 0;
    [[nodiscard]] virtual ByteSize maximum_input_size() const noexcept = 0;
    [[nodiscard]] virtual Result<ByteSize>
    maximum_compressed_size(ByteSize input_size) const noexcept = 0;
    [[nodiscard]] virtual Result<ByteSize> compress(std::span<const std::byte> input,
                                                    std::span<std::byte> output) noexcept = 0;
    [[nodiscard]] virtual Result<ByteSize> decompress(std::span<const std::byte> input,
                                                      std::span<std::byte> output,
                                                      ByteSize expected_size) noexcept = 0;
};

[[nodiscard]] CompressionCodec& cpu_compression_codec() noexcept;

} // namespace vramz::detail
