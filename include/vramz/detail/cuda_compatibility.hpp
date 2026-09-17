#pragma once

#include "vramz/error.hpp"
#include "vramz/result.hpp"

#include <array>
#include <charconv>
#include <string_view>
#include <system_error>

namespace vramz::detail {

struct NvidiaDriverRelease final {
    std::uint32_t branch{};
    std::uint32_t minor{};
    std::uint32_t patch{};
    auto operator<=>(const NvidiaDriverRelease&) const = default;
};

// Parse the kernel module's bounded release text, not a CUDA API version integer.
[[nodiscard]] inline Result<NvidiaDriverRelease>
parse_nvidia_driver_release(std::string_view text) noexcept {
    if (text.ends_with('\n')) {
        text.remove_suffix(1U);
    }
    std::array<std::uint32_t, 3U> parts{};
    std::size_t count{};
    while (!text.empty() && count < parts.size()) {
        const auto end = text.find('.');
        const auto part = text.substr(0U, end);
        const auto converted =
            std::from_chars(part.data(), part.data() + part.size(), parts[count]);
        if (part.empty() || converted.ec != std::errc{} ||
            converted.ptr != part.data() + part.size()) {
            return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
        }
        ++count;
        if (end == std::string_view::npos) {
            text = {};
            break;
        }
        text.remove_prefix(end + 1U);
        if (text.empty()) {
            return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
        }
    }
    if (!text.empty() || count < 2U || parts[0U] == 0U) {
        return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    }
    return NvidiaDriverRelease{parts[0U], parts[1U], parts[2U]};
}

enum class CudaCompatibility : std::uint8_t {
    not_evaluated,
    same_family_candidate,
    minor_version_candidate,
    unsupported_family,
    driver_below_minimum
};

// CUDA 13 minor compatibility has a Driver release floor, not an API minor-order floor.
// A candidate does not prove feature, kernel, stream or nvCOMP behavior on hardware.
[[nodiscard]] constexpr CudaCompatibility
compression_runtime_candidate(std::int32_t driver_api, std::int32_t runtime_api,
                              NvidiaDriverRelease installed) noexcept {
    if (driver_api / 1000 != 13 || runtime_api / 1000 != 13) {
        return CudaCompatibility::unsupported_family;
    }
    if (installed.branch < 580U) {
        return CudaCompatibility::driver_below_minimum;
    }
    return driver_api < runtime_api ? CudaCompatibility::minor_version_candidate
                                    : CudaCompatibility::same_family_candidate;
}
[[nodiscard]] constexpr bool is_candidate(CudaCompatibility value) noexcept {
    return value == CudaCompatibility::same_family_candidate ||
           value == CudaCompatibility::minor_version_candidate;
}
[[nodiscard]] constexpr std::string_view compatibility_name(CudaCompatibility value) noexcept {
    switch (value) {
    case CudaCompatibility::not_evaluated:
        return "NOT_EVALUATED";
    case CudaCompatibility::same_family_candidate:
        return "SAME_FAMILY_CANDIDATE";
    case CudaCompatibility::minor_version_candidate:
        return "MINOR_COMPATIBILITY_CANDIDATE";
    case CudaCompatibility::unsupported_family:
        return "UNSUPPORTED_CUDA_FAMILY";
    case CudaCompatibility::driver_below_minimum:
        return "DRIVER_BELOW_CUDA13_MINIMUM";
    }
    return "NOT_EVALUATED";
}

// Native values are checked against NVIDIA headers in the real adapter compilation.
// Preserve ambiguity: an error code never proves that queued work can be destroyed.
[[nodiscard]] constexpr Error cuda_runtime_error(std::int64_t native, OperationId operation,
                                                 bool ambiguous) noexcept {
    return {ambiguous ? ErrorCode::ambiguous_backend_state : ErrorCode::backend_failure,
            operation,
            BackendId{3U},
            NativeErrorDomain::cuda_runtime,
            native,
            0U,
            0U};
}
[[nodiscard]] constexpr std::string_view cuda_compatibility_failure(Error error) noexcept {
    if (error.native_domain == NativeErrorDomain::cuda_runtime ||
        error.native_domain == NativeErrorDomain::cuda_driver) {
        switch (error.native_code) {
        case 35:
            return error.native_domain == NativeErrorDomain::cuda_runtime ? "INSUFFICIENT_DRIVER"
                                                                          : "NOT_IDENTIFIED";
        case 36:
            return "CALL_REQUIRES_NEWER_DRIVER";
        case 222:
            return "UNSUPPORTED_PTX_VERSION";
        case 801:
            return "UNSUPPORTED_FEATURE";
        case 803:
            return "SYSTEM_DRIVER_MISMATCH";
        case 804:
            return "COMPATIBILITY_NOT_SUPPORTED_ON_DEVICE";
        default:
            break;
        }
    }
    return error.code == ErrorCode::unsupported ? "UNSUPPORTED_FEATURE" : "NOT_IDENTIFIED";
}

} // namespace vramz::detail
