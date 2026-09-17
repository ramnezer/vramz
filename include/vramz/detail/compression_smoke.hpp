#pragma once

#include "vramz/detail/cuda_compatibility.hpp"
#include "vramz/detail/cuda_vmm_backend.hpp"

#include <array>
#include <span>
#include <string_view>

namespace vramz::detail {

inline constexpr ByteSize m8_physical_cap{64ULL * 1024U * 1024U};
inline constexpr ByteSize m8_logical_cap{4ULL * 1024U * 1024U};
inline constexpr ByteSize m8_default_payload{2ULL * 1024U * 1024U};
inline constexpr std::string_view m8_device_name{"NVIDIA GeForce RTX 3060"};
enum class CompressionSmokeResult : std::uint8_t {
    planned,
    passed,
    failed,
    unsupported,
    resource_cap,
    preflight_rejected,
    no_physical_savings,
    compression_not_beneficial
};
struct CompressionSmokeOptions final {
    bool acknowledged{};
    std::int32_t device{};
    ByteSize logical_size{m8_default_payload};
    ByteSize physical_cap{m8_physical_cap};
    NvidiaDriverRelease installed_driver{};
    std::int32_t runtime_api{13030};
    std::int32_t prior_driver_api{13020};
};
struct CompressionSmokeReport final {
    CompressionSmokeResult result{CompressionSmokeResult::planned};
    std::int32_t device_ordinal{};
    NvidiaDriverRelease installed_driver{};
    std::int32_t runtime_api{};
    std::int32_t prior_driver_api{};
    CudaCompatibility compatibility{CudaCompatibility::not_evaluated};
    CudaProbeInfo probe{};
    CudaCompressionAdmission admission{};
    GpuCodecAudit codec{};
    // Measured after M2 adoption and exact BudgetLedger materialization, not an estimate.
    ByteSize raw_charge{};
    bool charges_corroborated{};
    ByteSize stored_bytes{};
    ByteSize compressed_charge{};
    std::uint32_t logical_crc_expected{};
    std::uint32_t logical_crc_actual{};
    bool source_verified{};
    bool size_equal{};
    bool byte_equal{};
    bool compaction_exercised{};
    bool cleanup_workspace{};
    bool cleanup_bound_output{};
    bool cleanup_compressed{};
    bool cleanup_raw{};
    bool cleanup_va{};
    bool cleanup_stream{};
    bool cleanup_context{};
    bool final_counts_known{};
    std::uint64_t final_resources{};
    std::uint64_t final_workspace{};
    std::uint64_t final_metadata{};
    std::uint64_t final_bound_output{};
    std::uint64_t final_compaction{};
    std::uint64_t final_raw{};
    std::uint64_t final_compressed{};
    std::uint64_t final_va{};
    ByteSize final_budget{};
    bool has_error{};
    Error error{};
};

[[nodiscard]] Result<CompressionSmokeOptions>
parse_compression_smoke_arguments(std::span<const std::string_view> arguments) noexcept;
// Preserve a backend fatal error for bounded reporting without locks or cleanup retries.
void finalize_compression_smoke_failure(CompressionSmokeReport& report) noexcept;
// Private smoke allocations never publish or replace Runtime/Buffer authority. The codec
// borrows the supplied driver's lifetime; report must outlive the backend and codec.
void run_compression_smoke(std::unique_ptr<CudaDriverApi> driver,
                           std::unique_ptr<GpuCompressionApi> codec,
                           CompressionSmokeOptions options,
                           CompressionSmokeReport& report) noexcept;
[[nodiscard]] FormatResult format_compression_smoke_report(const CompressionSmokeReport& report,
                                                           std::string_view source_sha256,
                                                           bool physical_execution,
                                                           std::span<char> output) noexcept;

} // namespace vramz::detail
