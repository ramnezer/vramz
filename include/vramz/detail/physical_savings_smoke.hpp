#pragma once

#include "vramz/detail/compression_smoke.hpp"

namespace vramz::detail {
inline constexpr ByteSize m9_logical_payload{8ULL * 1024U * 1024U};
inline constexpr ByteSize m9_physical_cap{96ULL * 1024U * 1024U};
// Size and cap are fixed, never adjustable by arguments or a retry.
struct PhysicalSavingsSmokeOptions final {
    bool acknowledged{};
    std::int32_t device{};
    NvidiaDriverRelease installed_driver{};
    std::int32_t runtime_api{13030};
    std::int32_t prior_driver_api{13020};
};
[[nodiscard]] constexpr bool saves_physical_unit(ByteSize raw, ByteSize compressed,
                                                 ByteSize minimum) noexcept {
    // Subtraction follows strict ordering; no sum or difference can overflow.
    return minimum != ByteSize{} && compressed != ByteSize{} && compressed < raw &&
           raw.value() % minimum.value() == 0U && compressed.value() % minimum.value() == 0U &&
           raw.value() - compressed.value() >= minimum.value();
}
[[nodiscard]] constexpr std::byte physical_savings_payload_byte(std::size_t index) noexcept {
    // M8's repeating/varying record formula shifted from 1..251 to 0..250.
    // This bijective byte remapping preserves structure and introduces real zero bytes.
    return static_cast<std::byte>(((index % 4096U) * 37U + index % 17U) % 251U);
}
[[nodiscard]] Result<PhysicalSavingsSmokeOptions>
parse_physical_savings_smoke_arguments(std::span<const std::string_view> arguments) noexcept;
void run_physical_savings_smoke(std::unique_ptr<CudaDriverApi> driver,
                                std::unique_ptr<GpuCompressionApi> codec,
                                PhysicalSavingsSmokeOptions options,
                                CompressionSmokeReport& report) noexcept;
[[nodiscard]] FormatResult
format_physical_savings_smoke_report(const CompressionSmokeReport& report,
                                     std::string_view source_sha256, bool physical_execution,
                                     std::span<char> output) noexcept;
} // namespace vramz::detail
