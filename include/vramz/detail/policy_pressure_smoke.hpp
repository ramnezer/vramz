#pragma once

#include "vramz/detail/multi_chunk_residency_smoke.hpp"
#include "vramz/policy.hpp"
#include "vramz/runtime.hpp"
#include "vramz/transaction.hpp"

namespace vramz::detail {

inline constexpr std::uint32_t m11_maximum_cycles{4U};
inline constexpr ByteSize m11_physical_cap{128ULL * 1024U * 1024U};
inline constexpr ByteSize m11_migration_reserve{93ULL * 1024U * 1024U};
inline constexpr ByteSize m11_settled_target{22ULL * 1024U * 1024U};
inline constexpr ByteSize m11_high_threshold{33292288U};

// These are the existing production budget/pressure semantics, with unchanged tuning.
// Only the bounded number of transitions per requested maintenance cycle is reduced.
[[nodiscard]] RuntimeConfig policy_pressure_configuration() noexcept;
[[nodiscard]] Result<ByteSize> policy_pressure_charge(const TierUsage& usage) noexcept;
[[nodiscard]] Result<ByteSize> admit_policy_pressure_stage(ByteSize live, ByteSize destination,
                                                           ByteSize workspace,
                                                           ByteSize compaction) noexcept;

struct PolicyPressureChunkReport final {
    MultiChunkResidencyChunkReport integrity{};
    policy::Metadata access{};
    std::uint64_t access_revision{};
    std::uint32_t effective_frequency{};
    Temperature temperature{Temperature::hot};
    RepresentationState initial{RepresentationState::gpu_raw};
    RepresentationState settled{RepresentationState::gpu_raw};
    std::uint32_t selected_count{};
    bool integrity_verified{};
};

struct PolicyPressureStageReport final {
    TransactionPhase phase{};
    TierUsage usage{};
    ByteSize owned_physical_charge{};
    std::uint64_t backend_resources{};
};
struct PolicyPressureCycleReport final {
    std::uint64_t cycle_id{};
    std::uint64_t selected_revision{};
    ByteSize charge_before{};
    ByteSize charge_after{};
    ByteSize reclaimed{};
    std::uint64_t retry_after_epoch{};
    Error error{};
    TierUsage before{};
    TierUsage after{};
    policy::Proposal proposal{};
    std::array<PolicyPressureStageReport, 16U> stages{};
    std::uint32_t selected_chunk{4U};
    std::uint32_t stage_count{};
    std::uint32_t workspace_materializations{};
    Pressure pressure_before{Pressure::normal};
    Pressure pressure_after{Pressure::normal};
    RepresentationState representation_before{RepresentationState::gpu_raw};
    bool selected{};
    bool completed{};
    bool transition_succeeded{};
    bool stale_rejection{};
    Compressibility compressibility_after{Compressibility::unknown};
    bool has_error{};
    bool charge_reconciled{};
};

enum class PolicyPressureEventKind : std::uint8_t {
    initial_snapshot,
    policy_cycle_start,
    policy_settled_snapshot,
    integrity_start,
    cleanup_complete,
    pass_published
};
struct PolicyPressureEvent final {
    PolicyPressureEventKind kind{};
    std::uint32_t index{};
};

// Optional non-owning test observer. The production harness never supplies one.
// It sees phase boundaries only, cannot supply victims, actions, or replacement results.
class PolicyPressureObserver {
  public:
    virtual ~PolicyPressureObserver() = default;
    virtual void observe(PolicyPressureEvent event) noexcept = 0;
};

struct PolicyPressureSmokeReport final {
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
    std::array<PolicyPressureChunkReport, m10_chunk_count> chunks{
        {{.integrity = {.chunk_index = 0U}},
         {.integrity = {.chunk_index = 1U}},
         {.integrity = {.chunk_index = 2U}},
         {.integrity = {.chunk_index = 3U}}}};
    std::array<PolicyPressureCycleReport, m11_maximum_cycles> cycles{};
    std::uint32_t cycle_count{};
    std::uint32_t automatic_policy_actions{};
    TierUsage initial_usage{};
    TierUsage settled_usage{};
    TierUsage final_usage{};
    ByteSize aggregate_raw_charge{};
    ByteSize aggregate_settled_charge{};
    ByteSize reclaimed{};
    ByteSize peak_admitted_gpu_bytes{};
    Pressure initial_pressure{Pressure::normal};
    Pressure settled_pressure{Pressure::normal};
    bool initial_snapshot_proven{};
    bool hot_chunk_preserved_raw{};
    bool pressure_target_reached{};
    bool policy_settled_snapshot_proven{};
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

using PolicyPressureSmokeOptions = MultiChunkResidencySmokeOptions;
[[nodiscard]] Result<PolicyPressureSmokeOptions>
parse_policy_pressure_smoke_arguments(std::span<const std::string_view> arguments) noexcept;
void finalize_policy_pressure_smoke_failure(PolicyPressureSmokeReport& report) noexcept;
void run_policy_pressure_smoke(std::unique_ptr<CudaDriverApi> driver,
                               std::unique_ptr<GpuCompressionApi> codec,
                               PolicyPressureSmokeOptions options,
                               PolicyPressureSmokeReport& report,
                               PolicyPressureObserver* observer = nullptr) noexcept;
[[nodiscard]] FormatResult
format_policy_pressure_smoke_report(const PolicyPressureSmokeReport& report,
                                    std::string_view source_sha256, bool physical_execution,
                                    std::span<char> output) noexcept;

} // namespace vramz::detail
