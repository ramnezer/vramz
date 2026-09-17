#pragma once

#include "vramz/detail/physical_savings_smoke.hpp"

namespace vramz::detail {

inline constexpr std::size_t m10_chunk_count{4U};
inline constexpr ByteSize m10_logical_payload{8ULL * 1024U * 1024U};
inline constexpr ByteSize m10_physical_cap{128ULL * 1024U * 1024U};

// Neither the fixed chunk count, logical size nor cap is a runtime option.
struct MultiChunkResidencySmokeOptions final {
    bool acknowledged{};
    std::int32_t device{};
    NvidiaDriverRelease installed_driver{};
    std::int32_t runtime_api{13030};
    std::int32_t prior_driver_api{13020};
};

[[nodiscard]] constexpr std::byte multi_chunk_residency_payload_byte(std::size_t chunk,
                                                                     std::size_t index) noexcept {
    // A fixed byte permutation of M9's record pattern: all variants retain zeros,
    // nonzero bytes and variation, while their deterministic chunk identities differ.
    return static_cast<std::byte>(((index % 4096U) * 37U + index % 17U + (chunk % 4U) * 53U) %
                                  251U);
}

struct MultiChunkResidencyChunkReport final {
    std::uint32_t chunk_index{};
    ByteSize logical_bytes{m10_logical_payload};
    ByteSize raw_charge{};
    ByteSize stored_bytes{};
    ByteSize compressed_charge{};
    ByteSize physical_bytes_saved{};
    std::uint32_t logical_crc_expected{};
    std::uint32_t logical_crc_actual{};
    std::uint32_t stored_crc_expected{};
    std::uint32_t stored_crc_actual{};
    bool source_verified{};
    bool charges_corroborated{};
    bool minimum_unit_saved{};
    bool compression_enqueue{};
    bool compression_completion{};
    bool decompression_enqueue{};
    bool decompression_completion{};
    bool compaction_exercised{};
    bool compressed_at_snapshot{};
    bool size_equal{};
    bool byte_equal{};
    bool round_trip_verified{};
    bool cleanup_complete{};
};

struct MultiChunkResidencyTotals final {
    ByteSize logical_bytes{};
    ByteSize raw_baseline_charge{};
    ByteSize compressed_physical_charge{};
    ByteSize physical_bytes_saved{};
};

// Arithmetic helpers consume measured charges; they do not allocate or invoke a backend.
[[nodiscard]] Result<MultiChunkResidencyTotals>
calculate_multi_chunk_residency_totals(std::span<const MultiChunkResidencyChunkReport> chunks,
                                       ByteSize minimum_granularity) noexcept;
[[nodiscard]] Result<ByteSize> admit_multi_chunk_residency_stage(ByteSize retained_charge,
                                                                 ByteSize operation_peak) noexcept;

enum class MultiChunkResidencyEventKind : std::uint8_t {
    compression_start,
    compressed_retained,
    aggregate_snapshot,
    restoration_start,
    restoration_verified,
    cleanup_complete,
    pass_published
};
struct MultiChunkResidencyEvent final {
    MultiChunkResidencyEventKind kind{};
    std::uint32_t chunk_index{};
    std::uint32_t compressed_mask{};
};

struct MultiChunkResidencySmokeReport final {
    CompressionSmokeResult result{CompressionSmokeResult::planned};
    std::int32_t device_ordinal{};
    NvidiaDriverRelease installed_driver{};
    std::int32_t runtime_api{};
    std::int32_t prior_driver_api{};
    CudaCompatibility compatibility{CudaCompatibility::not_evaluated};
    CudaProbeInfo probe{};
    CudaCompressionAdmission admission{};
    GpuCodecAudit codec{};
    std::array<MultiChunkResidencyChunkReport, m10_chunk_count> chunks{
        {{.chunk_index = 0U}, {.chunk_index = 1U}, {.chunk_index = 2U}, {.chunk_index = 3U}}};
    MultiChunkResidencyTotals totals{};
    ByteSize peak_admitted_gpu_bytes{};
    ByteSize maximum_physical_bytes{m10_physical_cap};
    bool simultaneous_compressed_residency_proven{};
    // Passive bounded observations, recorded only after the corresponding ownership proof.
    // These never control ordering, ownership or synchronization.
    std::array<MultiChunkResidencyEvent, 19U> events{};
    std::size_t event_count{};
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
    ByteSize final_unmaterialized_reservations{};
    bool has_error{};
    Error error{};
};

[[nodiscard]] Result<MultiChunkResidencySmokeOptions>
parse_multi_chunk_residency_smoke_arguments(std::span<const std::string_view> arguments) noexcept;
void finalize_multi_chunk_residency_smoke_failure(MultiChunkResidencySmokeReport& report) noexcept;
void run_multi_chunk_residency_smoke(std::unique_ptr<CudaDriverApi> driver,
                                     std::unique_ptr<GpuCompressionApi> codec,
                                     MultiChunkResidencySmokeOptions options,
                                     MultiChunkResidencySmokeReport& report) noexcept;
[[nodiscard]] FormatResult
format_multi_chunk_residency_smoke_report(const MultiChunkResidencySmokeReport& report,
                                          std::string_view source_sha256, bool physical_execution,
                                          std::span<char> output) noexcept;

} // namespace vramz::detail
