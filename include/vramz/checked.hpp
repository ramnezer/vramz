#pragma once

#include "vramz/result.hpp"

#include <cstdint>
#include <limits>

namespace vramz {

[[nodiscard]] constexpr Result<std::uint64_t> checked_add(std::uint64_t left, std::uint64_t right,
                                                          OperationId operation) noexcept {
    if (right > (std::numeric_limits<std::uint64_t>::max() - left)) {
        return make_error(ErrorCode::arithmetic_overflow, operation, left, right);
    }
    return left + right;
}

[[nodiscard]] constexpr Result<std::uint64_t> checked_sub(std::uint64_t left, std::uint64_t right,
                                                          OperationId operation) noexcept {
    if (right > left) {
        return make_error(ErrorCode::arithmetic_overflow, operation, left, right);
    }
    return left - right;
}

[[nodiscard]] constexpr Result<std::uint64_t> checked_mul(std::uint64_t left, std::uint64_t right,
                                                          OperationId operation) noexcept {
    if ((left != 0U) && (right > (std::numeric_limits<std::uint64_t>::max() / left))) {
        return make_error(ErrorCode::arithmetic_overflow, operation, left, right);
    }
    return left * right;
}

[[nodiscard]] constexpr Result<std::uint64_t>
checked_align_up(std::uint64_t value, std::uint64_t alignment, OperationId operation) noexcept {
    if (alignment == 0U) {
        return make_error(ErrorCode::invalid_alignment, operation, value, alignment);
    }
    const auto remainder = value % alignment;
    if (remainder == 0U) {
        return value;
    }
    return checked_add(value, alignment - remainder, operation);
}

[[nodiscard]] constexpr Result<ByteOffset> checked_range_end(MemoryRange range,
                                                             OperationId operation) noexcept {
    const auto end = checked_add(range.offset.value(), range.length.value(), operation);
    if (!end) {
        return end.error();
    }
    return ByteOffset{end.value()};
}

} // namespace vramz
