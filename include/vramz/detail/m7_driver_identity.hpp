#pragma once

#include "vramz/checked.hpp"

#include <cstdint>
#include <sys/stat.h>
#include <utility>

namespace vramz::detail {

struct DriverFileIdentity final {
    std::uint64_t device{};
    std::uint64_t inode{};
    std::uint64_t size{};
    std::uint64_t mtime_ns{};
    std::uint64_t ctime_ns{};
};

[[nodiscard]] inline Result<std::uint64_t> driver_timestamp_ns(const timespec& value) noexcept {
    if (value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000L ||
        !std::in_range<std::uint64_t>(value.tv_sec)) {
        return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    }
    const auto seconds = checked_mul(static_cast<std::uint64_t>(value.tv_sec), 1000000000U,
                                     OperationId::runtime_create);
    if (!seconds) {
        return seconds.error();
    }
    return checked_add(seconds.value(), static_cast<std::uint64_t>(value.tv_nsec),
                       OperationId::runtime_create);
}

// Pure metadata comparison: does not open a file, load a library, or call a driver.
[[nodiscard]] inline bool matches_driver_file_identity(const struct stat& value,
                                                       DriverFileIdentity expected) noexcept {
    if (!S_ISREG(value.st_mode) || !std::in_range<std::uint64_t>(value.st_dev) ||
        !std::in_range<std::uint64_t>(value.st_ino) ||
        !std::in_range<std::uint64_t>(value.st_size)) {
        return false;
    }
    const auto mtime = driver_timestamp_ns(value.st_mtim);
    const auto ctime = driver_timestamp_ns(value.st_ctim);
    return mtime && ctime && static_cast<std::uint64_t>(value.st_dev) == expected.device &&
           static_cast<std::uint64_t>(value.st_ino) == expected.inode &&
           static_cast<std::uint64_t>(value.st_size) == expected.size &&
           mtime.value() == expected.mtime_ns && ctime.value() == expected.ctime_ns;
}

} // namespace vramz::detail
