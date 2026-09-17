#pragma once

#include "vramz/detail/policy_pressure_smoke.hpp"
#include "vramz/policy.hpp"
#include "vramz/runtime.hpp"
#include "vramz/transaction.hpp"

namespace vramz::detail {

inline constexpr std::uint32_t m13_maximum_cycles{8U};
inline constexpr ByteSize m13_physical_cap{128ULL * 1024U * 1024U};
inline constexpr ByteSize m13_migration_reserve{64ULL * 1024U * 1024U};
inline constexpr ByteSize m13_settled_target{32ULL * 1024U * 1024U};
inline constexpr ByteSize m13_high_threshold{58720256U};

inline constexpr std::size_t m13_chunk_count{8U};
inline constexpr ByteSize m13_logical_payload{8ULL * 1024U * 1024U};
inline constexpr ByteSize m13_aggregate_logical{64ULL * 1024U * 1024U};
inline constexpr ByteSize m13_normal_limit{64ULL * 1024U * 1024U};

// Same M10/M11/M12 record structure and first six variants; extend to indices 6 and 7.
[[nodiscard]] constexpr std::byte controlled_capacity_2x_payload_byte(std::size_t chunk,
                                                                      std::size_t index) noexcept {
    return static_cast<std::byte>(((index % 4096U) * 37U + index % 17U + (chunk % 8U) * 53U) %
                                  251U);
}

// The physical product is checked before any capacity claim; floating ratios are telemetry only.
[[nodiscard]] Result<bool> controlled_capacity_2_0x(ByteSize logical, ByteSize physical) noexcept;

// These are the existing production budget/pressure semantics, with unchanged tuning.
// Only the bounded number of transitions per requested maintenance cycle is reduced.
[[nodiscard]] RuntimeConfig controlled_capacity_2x_configuration() noexcept;
[[nodiscard]] Result<ByteSize> admit_controlled_capacity_2x_stage(ByteSize live,
                                                                  ByteSize destination,
                                                                  ByteSize workspace,
                                                                  ByteSize compaction) noexcept;

enum class ControlledCapacity2xEventKind : std::uint8_t {
    initial_snapshot,
    policy_cycle_start,
    policy_settled_snapshot,
    integrity_start,
    cleanup_complete,
    pass_published
};
struct ControlledCapacity2xEvent final {
    ControlledCapacity2xEventKind kind{};
    std::uint32_t index{};
};

// Optional non-owning test observer. The production harness never supplies one.
// It sees phase boundaries only, cannot supply victims, actions, or replacement results.
class ControlledCapacity2xObserver {
  public:
    virtual ~ControlledCapacity2xObserver() = default;
    virtual void observe(ControlledCapacity2xEvent event) noexcept = 0;
};

struct ControlledCapacity2xSmokeReport final {
    CompressionSmokeResult result{CompressionSmokeResult::planned};
    std::int32_t device_ordinal{};
    NvidiaDriverRelease installed_driver{};
    std::int32_t runtime_api{};
    std::int32_t prior_driver_api{};
    CudaCompatibility compatibility{CudaCompatibility::not_evaluated};
    struct Provider final {
        std::string_view role{};
        std::string_view path{};
        std::string_view sha256{};
    };
    std::array<Provider, 3U> approved_providers{};
    CudaProbeInfo probe{};
    CudaCompressionAdmission admission{};
    GpuCodecAudit codec{};
    std::array<PolicyPressureChunkReport, m13_chunk_count> chunks{
        {{.integrity = {.chunk_index = 0U}},
         {.integrity = {.chunk_index = 1U}},
         {.integrity = {.chunk_index = 2U}},
         {.integrity = {.chunk_index = 3U}},
         {.integrity = {.chunk_index = 4U}},
         {.integrity = {.chunk_index = 5U}},
         {.integrity = {.chunk_index = 6U}},
         {.integrity = {.chunk_index = 7U}}}};
    std::array<PolicyPressureCycleReport, m13_maximum_cycles> cycles{};
    std::uint32_t cycle_count{};
    std::uint32_t automatic_policy_actions{};
    TierUsage initial_usage{};
    TierUsage settled_usage{};
    TierUsage final_usage{};
    ByteSize aggregate_logical_bytes{};
    ByteSize aggregate_raw_charge{};
    ByteSize aggregate_settled_charge{};
    ByteSize reclaimed{};
    ByteSize peak_admitted_gpu_bytes{};
    Pressure initial_pressure{Pressure::normal};
    Pressure settled_pressure{Pressure::normal};
    bool initial_snapshot_proven{};
    bool hot_chunk_preserved_raw{};
    bool warm_chunk_preserved_raw{};
    bool pressure_target_reached{};
    bool controlled_capacity_snapshot_proven{};
    bool controlled_capacity_2_0x_proven{};
    bool cleanup_raw{};
    bool cleanup_compressed{};
    bool cleanup_workspace{};
    bool cleanup_va{};
    bool cleanup_stream{};
    bool cleanup_context{};
    bool final_counts_known{};
    std::uint64_t final_resources{};
    std::uint64_t final_va{};
    std::uint64_t final_workspace{};
    std::uint64_t final_raw{};
    std::uint64_t final_compressed{};
    bool has_error{};
    Error error{};
};

using ControlledCapacity2xSmokeOptions = MultiChunkResidencySmokeOptions;
[[nodiscard]] Result<ControlledCapacity2xSmokeOptions>
parse_controlled_capacity_2x_smoke_arguments(std::span<const std::string_view> arguments) noexcept;
void finalize_controlled_capacity_2x_smoke_failure(
    ControlledCapacity2xSmokeReport& report) noexcept;
void run_controlled_capacity_2x_smoke(std::unique_ptr<CudaDriverApi> driver,
                                      std::unique_ptr<GpuCompressionApi> codec,
                                      ControlledCapacity2xSmokeOptions options,
                                      ControlledCapacity2xSmokeReport& report,
                                      ControlledCapacity2xObserver* observer = nullptr) noexcept;
[[nodiscard]] FormatResult
format_controlled_capacity_2x_smoke_report(const ControlledCapacity2xSmokeReport& report,
                                           std::string_view source_sha256, bool physical_execution,
                                           std::span<char> output) noexcept;

} // namespace vramz::detail
