#include "vramz/detail/compression.hpp"

#include <lz4.h>

#include <cstdint>
#include <limits>

namespace vramz::detail {
namespace {

[[nodiscard]] Error codec_error(ErrorCode code, OperationId operation,
                                std::uint64_t detail = 0U) noexcept {
    Error error = make_error(code, operation, 0U, detail);
    error.backend = BackendId{2U};
    error.native_domain = NativeErrorDomain::compression_backend;
    return error;
}

[[nodiscard]] Result<int> checked_codec_size(std::size_t value, OperationId operation) noexcept {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return codec_error(ErrorCode::arithmetic_overflow, operation,
                           static_cast<std::uint64_t>(value));
    }
    return static_cast<int>(value);
}

class Lz4BlockCodec final : public CompressionCodec {
  public:
    [[nodiscard]] CodecId id() const noexcept override { return CodecId::lz4_block; }
    [[nodiscard]] bool available() const noexcept override { return true; }
    [[nodiscard]] ByteSize maximum_input_size() const noexcept override {
        return ByteSize{static_cast<std::uint64_t>(LZ4_MAX_INPUT_SIZE)};
    }

    [[nodiscard]] Result<ByteSize>
    maximum_compressed_size(ByteSize input_size) const noexcept override {
        if (input_size > maximum_input_size()) {
            return codec_error(ErrorCode::invalid_argument, OperationId::compress,
                               input_size.value());
        }
        const auto input =
            checked_codec_size(static_cast<std::size_t>(input_size.value()), OperationId::compress);
        if (!input) {
            return input.error();
        }
        if (input.value() == 0) {
            return ByteSize{};
        }
        const int bound = LZ4_compressBound(input.value());
        if (bound <= 0) {
            return codec_error(ErrorCode::backend_failure, OperationId::compress,
                               input_size.value());
        }
        return ByteSize{static_cast<std::uint64_t>(bound)};
    }

    [[nodiscard]] Result<ByteSize> compress(std::span<const std::byte> input,
                                            std::span<std::byte> output) noexcept override {
        if (input.empty()) {
            return ByteSize{};
        }
        if (input.size() > static_cast<std::size_t>(maximum_input_size().value())) {
            return codec_error(ErrorCode::invalid_argument, OperationId::compress,
                               static_cast<std::uint64_t>(input.size()));
        }
        const auto input_size = checked_codec_size(input.size(), OperationId::compress);
        const auto output_size = checked_codec_size(output.size(), OperationId::compress);
        if (!input_size) {
            return input_size.error();
        }
        if (!output_size) {
            return output_size.error();
        }
        const int written = LZ4_compress_default(reinterpret_cast<const char*>(input.data()),
                                                 reinterpret_cast<char*>(output.data()),
                                                 input_size.value(), output_size.value());
        if (written <= 0) {
            return codec_error(ErrorCode::backend_failure, OperationId::compress,
                               static_cast<std::uint64_t>(input.size()));
        }
        return ByteSize{static_cast<std::uint64_t>(written)};
    }

    [[nodiscard]] Result<ByteSize> decompress(std::span<const std::byte> input,
                                              std::span<std::byte> output,
                                              ByteSize expected_size) noexcept override {
        if (expected_size.value() == 0U) {
            return input.empty()
                       ? Result<ByteSize>{ByteSize{}}
                       : Result<ByteSize>{codec_error(ErrorCode::integrity_failure,
                                                      OperationId::decompress,
                                                      static_cast<std::uint64_t>(input.size()))};
        }
        if (expected_size.value() > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            return codec_error(ErrorCode::arithmetic_overflow, OperationId::decompress,
                               expected_size.value());
        }
        if (output.size() < static_cast<std::size_t>(expected_size.value())) {
            return codec_error(ErrorCode::invalid_range, OperationId::decompress,
                               expected_size.value());
        }
        const auto input_size = checked_codec_size(input.size(), OperationId::decompress);
        const auto output_size = checked_codec_size(static_cast<std::size_t>(expected_size.value()),
                                                    OperationId::decompress);
        if (!input_size) {
            return input_size.error();
        }
        if (!output_size) {
            return output_size.error();
        }
        const int written = LZ4_decompress_safe(reinterpret_cast<const char*>(input.data()),
                                                reinterpret_cast<char*>(output.data()),
                                                input_size.value(), output_size.value());
        if (written < 0 || written != output_size.value()) {
            return codec_error(ErrorCode::integrity_failure, OperationId::decompress,
                               static_cast<std::uint64_t>(input.size()));
        }
        return ByteSize{static_cast<std::uint64_t>(written)};
    }
};

} // namespace

CompressionCodec& cpu_compression_codec() noexcept {
    static Lz4BlockCodec codec{};
    return codec;
}

} // namespace vramz::detail
