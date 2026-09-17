#pragma once

#include "vramz/crc32c.hpp"

namespace vramz::detail {
// Portable implementation retained for unsupported CPUs and differential validation.
[[nodiscard]] std::uint32_t crc32c_portable(std::span<const std::byte> bytes) noexcept;
} // namespace vramz::detail
