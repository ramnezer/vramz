#pragma once

#include "vramz/detail/policy_pressure_smoke.hpp"
#include "vramz/policy.hpp"
#include "vramz/runtime.hpp"
#include "vramz/transaction.hpp"

namespace vramz::detail {

inline constexpr std::uint32_t m12_maximum_cycles{6U};
inline constexpr ByteSize m12_physical_cap{128ULL * 1024U * 1024U};
inline constexpr ByteSize m12_migration_reserve{80ULL * 1024U * 1024U};
inline constexpr ByteSize m12_settled_target{32ULL * 1024U * 1024U};
inline constexpr ByteSize m12_high_threshold{46137344U};

inline constexpr std::size_t m12_chunk_count{6U};
inline constexpr ByteSize m12_logical_payload{8ULL * 1024U * 1024U};
inline constexpr ByteSize m12_aggregate_logical{48ULL * 1024U * 1024U};
inline constexpr ByteSize m12_normal_limit{48ULL * 1024U * 1024U};

// Same M10/M11 record structure and first four variants; extend the index domain to six.
[[nodiscard]] constexpr std::byte controlled_capacity_payload_byte(std::size_t chunk,
                                                                   std::size_t index) noexcept {
    return static_cast<std::byte>(((index % 4096U) * 37U + index % 17U + (chunk % 6U) * 53U) %
                                  251U);
}

// Both products are checked before any capacity claim; floating ratios are telemetry only.
[[nodiscard]] Result<bool> controlled_capacity_1_5x(ByteSize logical, ByteSize physical) noexcept;

// These are the existing production budget/pressure semantics, with unchanged tuning.
// Only the bounded number of transitions per requested maintenance cycle is reduced.
[[nodiscard]] RuntimeConfig controlled_capacity_configuration() noexcept;
[[nodiscard]] Result<ByteSize> admit_controlled_capacity_stage(ByteSize live, ByteSize destination,
                                                               ByteSize workspace,
                                                               ByteSize compaction) noexcept;

enum class ControlledCapacityEventKind : std::uint8_t {
    initial_snapshot,
    policy_cycle_start,
    policy_settled_snapshot,
    integrity_start,
    cleanup_complete,
    pass_published
};
struct ControlledCapacityEvent final {
    ControlledCapacityEventKind kind{};
    std::uint32_t index{};
};

// Optional non-owning test observer. The production harness never supplies one.
// It sees phase boundaries only, cannot supply victims, actions, or replacement results.
class ControlledCapacityObserver {
  public:
    virtual ~ControlledCapacityObserver() = default;
    virtual void observe(ControlledCapacityEvent event) noexcept = 0;
};

struct ControlledCapacitySmokeReport final {
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
    std::array<PolicyPressureChunkReport, m12_chunk_count> chunks{
        {{.integrity = {.chunk_index = 0U}},
         {.integrity = {.chunk_index = 1U}},
         {.integrity = {.chunk_index = 2U}},
         {.integrity = {.chunk_index = 3U}},
         {.integrity = {.chunk_index = 4U}},
         {.integrity = {.chunk_index = 5U}}}};
    std::array<PolicyPressureCycleReport, m12_maximum_cycles> cycles{};
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
    bool pressure_target_reached{};
    bool controlled_capacity_snapshot_proven{};
    bool controlled_capacity_1_5x_proven{};
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

using ControlledCapacitySmokeOptions = MultiChunkResidencySmokeOptions;
[[nodiscard]] Result<ControlledCapacitySmokeOptions>
parse_controlled_capacity_smoke_arguments(std::span<const std::string_view> arguments) noexcept;
void finalize_controlled_capacity_smoke_failure(ControlledCapacitySmokeReport& report) noexcept;
void run_controlled_capacity_smoke(std::unique_ptr<CudaDriverApi> driver,
                                   std::unique_ptr<GpuCompressionApi> codec,
                                   ControlledCapacitySmokeOptions options,
                                   ControlledCapacitySmokeReport& report,
                                   ControlledCapacityObserver* observer = nullptr) noexcept;
[[nodiscard]] FormatResult
format_controlled_capacity_smoke_report(const ControlledCapacitySmokeReport& report,
                                        std::string_view source_sha256, bool physical_execution,
                                        std::span<char> output) noexcept;

} // namespace vramz::detail
