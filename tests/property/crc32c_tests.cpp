#include "../test_support.hpp"
#include "vramz/detail/crc32c.hpp"

#include <array>
#include <span>

using namespace vramz;
namespace {
std::uint32_t bit_reference(std::span<const std::byte> bytes) {
    std::uint32_t value = 0xFFFFFFFFU;
    for (const auto byte : bytes) {
        value ^= std::to_integer<std::uint32_t>(byte);
        for (std::uint32_t bit = 0U; bit < 8U; ++bit) {
            value = (value >> 1U) ^ ((value & 1U) != 0U ? 0x82F63B78U : 0U);
        }
    }
    return ~value;
}
} // namespace

int main() {
    test::Runner runner;
    runner.begin("accelerated and portable CRC preserve the independent bitwise oracle");
    std::array<std::byte, 8208U> storage{};
    std::uint64_t random{0x564d5aU};
    for (std::uint32_t round = 0U; round < 2048U; ++round) {
        for (auto& byte : storage) {
            random ^= random >> 12U;
            random ^= random << 25U;
            random ^= random >> 27U;
            byte = static_cast<std::byte>(random & 0xFFU);
        }
        const auto offset = static_cast<std::size_t>(round % 16U);
        const auto length = static_cast<std::size_t>(round < 32U ? round : random % 8193U);
        const auto bytes = std::span<const std::byte>{storage}.subspan(offset, length);
        const auto reference = bit_reference(bytes);
        VRAMZ_CHECK(runner, crc32c(bytes) == reference);
        VRAMZ_CHECK(runner, detail::crc32c_portable(bytes) == reference);
    }
    VRAMZ_CHECK(runner, crc32c({}) == 0U && detail::crc32c_portable({}) == 0U);
    return runner.finish();
}
