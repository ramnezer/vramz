#pragma once

#include <cstdint>
#include <limits>

namespace vramz {

[[nodiscard]] constexpr std::uint64_t saturating_add(std::uint64_t left,
                                                     std::uint64_t right) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left
               ? std::numeric_limits<std::uint64_t>::max()
               : left + right;
}

constexpr void saturating_increment(std::uint64_t& value) noexcept {
    value = saturating_add(value, 1U);
}

} // namespace vramz
