#pragma once

#include "vramz/detail/cuda_vmm_backend.hpp"

#include <array>
#include <span>
#include <string_view>

namespace vramz::detail {

inline constexpr ByteSize m7_physical_cap{std::uint64_t{16U} * 1024U * 1024U};
inline constexpr std::size_t m7_payload_cap = std::size_t{64U} * 1024U;
enum class RawSmokeResult : std::uint8_t {
    planned,
    passed,
    failed,
    unsupported,
    unsupported_granularity,
    preflight_rejected
};
struct RawSmokeOptions final {
    bool acknowledged{};
    std::int32_t device{};
    ByteSize maximum_physical{m7_physical_cap};
};
struct RawSmokeReport final {
    RawSmokeResult result{RawSmokeResult::planned};
    std::int32_t device_ordinal{};
    ByteSize maximum_physical{m7_physical_cap};
    CudaProbeInfo probe{};
    ByteSize logical_payload{};
    ByteSize physical_charge{};
    std::uint32_t crc_expected{};
    std::uint32_t crc_actual{};
    bool h2d_success{};
    bool d2h_success{};
    bool size_equal{};
    bool byte_compare{};
    bool stable_va_reserved{};
    bool mapping_verified{};
    bool access_verified{};
    bool cleanup_unmap{};
    bool cleanup_physical_release{};
    bool cleanup_va_free{};
    bool cleanup_context_release{};
    bool final_counts_known{};
    std::uint64_t final_backend_resources{};
    std::uint64_t final_va_reservations{};
    ByteSize final_budget_charge{};
    bool has_error{};
    Error error{};
};

[[nodiscard]] Result<RawSmokeOptions>
parse_raw_smoke_arguments(std::span<const std::string_view> arguments) noexcept;
// The caller authenticates the executable/driver locally before supplying a real adapter.
// No acknowledgement means no initialize, including when a fake adapter is supplied.
void run_raw_smoke(std::unique_ptr<CudaDriverApi> driver, RawSmokeOptions options,
                   RawSmokeReport& report) noexcept;
// Fixed-size, allocation-free JSON, without raw resource/address/error-detail fields.
[[nodiscard]] FormatResult format_raw_smoke_report(const RawSmokeReport& report,
                                                   std::string_view source_sha256,
                                                   bool physical_execution,
                                                   std::span<char> output) noexcept;

} // namespace vramz::detail
