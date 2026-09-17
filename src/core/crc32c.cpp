#include "vramz/crc32c.hpp"
#include "vramz/detail/crc32c.hpp"

#include <array>
#include <cstring>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <nmmintrin.h>
#endif

namespace vramz {
namespace {
constexpr auto make_table() {
    std::array<std::uint32_t, 256U> values{};
    for (std::uint32_t i = 0U; i < values.size(); ++i) {
        auto value = i;
        for (std::uint32_t bit = 0U; bit < 8U; ++bit) {
            value = (value >> 1U) ^ (0x82F63B78U & (0U - (value & 1U)));
        }
        values[i] = value;
    }
    return values;
}
constexpr auto table = make_table();

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// Dispatch guards this function; the installed binary does not require SSE4.2 globally.
__attribute__((target("sse4.2"))) std::uint32_t
hardware_crc(std::span<const std::byte> bytes) noexcept {
    unsigned long long state = 0xFFFFFFFFU;
    std::size_t offset{};
    while (bytes.size() - offset >= sizeof(std::uint64_t)) {
        std::uint64_t value{};
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        state = _mm_crc32_u64(state, value);
        offset += sizeof(value);
    }
    auto tail = static_cast<std::uint32_t>(state);
    for (; offset < bytes.size(); ++offset) {
        tail = _mm_crc32_u8(tail, std::to_integer<std::uint8_t>(bytes[offset]));
    }
    return ~tail;
}
#endif
} // namespace

std::uint32_t detail::crc32c_portable(std::span<const std::byte> bytes) noexcept {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const auto value : bytes) {
        crc = (crc >> 8U) ^ table[(crc ^ std::to_integer<std::uint32_t>(value)) & 0xFFU];
    }
    return ~crc;
}

std::uint32_t crc32c(std::span<const std::byte> bytes) noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const auto implementation =
        __builtin_cpu_supports("sse4.2") != 0 ? hardware_crc : detail::crc32c_portable;
    return implementation(bytes);
#else
    return detail::crc32c_portable(bytes);
#endif
}

} // namespace vramz
