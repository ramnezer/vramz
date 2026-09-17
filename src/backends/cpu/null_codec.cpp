#include "vramz/detail/compression.hpp"

namespace vramz::detail {
namespace {

class NullCodec final : public CompressionCodec {
  public:
    [[nodiscard]] CodecId id() const noexcept override { return CodecId::none; }
    [[nodiscard]] bool available() const noexcept override { return false; }
    [[nodiscard]] ByteSize maximum_input_size() const noexcept override { return {}; }

    [[nodiscard]] Result<ByteSize> maximum_compressed_size(ByteSize) const noexcept override {
        return make_error(ErrorCode::unsupported, OperationId::compress);
    }

    [[nodiscard]] Result<ByteSize> compress(std::span<const std::byte>,
                                            std::span<std::byte>) noexcept override {
        return make_error(ErrorCode::unsupported, OperationId::compress);
    }

    [[nodiscard]] Result<ByteSize> decompress(std::span<const std::byte>, std::span<std::byte>,
                                              ByteSize) noexcept override {
        return make_error(ErrorCode::unsupported, OperationId::decompress);
    }
};

} // namespace

CompressionCodec& cpu_compression_codec() noexcept {
    static NullCodec codec{};
    return codec;
}

} // namespace vramz::detail
