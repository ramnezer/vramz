#pragma once

#include "vramz/result.hpp"

#include <cstdint>

namespace vramz {

enum class PolicyMode : std::uint8_t { disabled, gpu_resident, gpu_resident_with_host_fallback };
enum class PolicyStrategy : std::uint8_t { adaptive, oldest_unpinned };
enum class Temperature : std::uint8_t { hot, warm, cold };
enum class Compressibility : std::uint8_t {
    unknown,
    compressible,
    poorly_compressible,
    incompressible
};
enum class Pressure : std::uint8_t { normal, soft, hard, critical };
enum class PolicyAction : std::uint8_t { keep, compress_gpu, restore_gpu_raw, host_fallback };

// Optional research controls; ordinary callers need only select the mode.
struct PolicyTuning final {
    ByteSize minimum_savings_bytes{64U};
    std::uint32_t minimum_savings_basis_points{1250U};
    std::uint64_t hot_epochs{2U};
    std::uint64_t warm_epochs{12U};
    std::uint64_t frequency_window{8U};
    std::uint32_t hot_frequency{4U};
    std::uint64_t minimum_raw_epochs{3U};
    std::uint64_t transition_cooldown{3U};
    std::uint64_t reevaluate_epochs{64U};
    std::uint64_t thrash_window{16U};
    std::uint32_t thrash_transition_count{4U};
    std::uint64_t thrash_hold_epochs{16U};
    std::uint32_t maximum_candidates{4096U};
    std::uint32_t maximum_transitions{32U};
    std::uint64_t compression_cost_per_block{3U};
    std::uint64_t restore_cost_per_block{2U};
    std::uint64_t host_cost_per_block{12U};
    PolicyStrategy strategy{PolicyStrategy::adaptive};
};

struct PolicyConfig final {
    PolicyMode mode{PolicyMode::gpu_resident};
    PolicyTuning tuning{};
};

struct PolicyStats final {
    ByteSize logical_gpu_resident_bytes{};
    ByteSize logical_host_bytes{};
    ByteSize gpu_raw_charge{};
    ByteSize gpu_compressed_charge{};
    ByteSize host_charge{};
    ByteSize saved_gpu_bytes{};
    std::uint64_t access_epoch{};
    std::uint64_t policy_cycles{};
    std::uint64_t candidates_inspected{};
    std::uint64_t policy_transitions{};
    std::uint64_t compression_attempts{};
    std::uint64_t compression_successes{};
    std::uint64_t compression_failures{};
    std::uint64_t compression_rejected{};
    std::uint64_t compression_skipped_incompressible{};
    std::uint64_t stale_proposal_rejections{};
    std::uint64_t restore_count{};
    std::uint64_t host_fallback_count{};
    std::uint64_t pcie_transition_count{};
    std::uint64_t thrash_events{};
    std::uint64_t simulated_compression_cost{};
    std::uint64_t simulated_restore_cost{};
    std::uint64_t simulated_host_cost{};
    std::uint64_t bounded_stops{};
    std::uint64_t decision_digest{};
    BufferId last_victim_buffer{};
    ChunkId last_victim_chunk{};
    PolicyAction last_action{PolicyAction::keep};
    Pressure current_pressure{Pressure::normal};

    [[nodiscard]] double gpu_effective_ratio() const noexcept;
};

} // namespace vramz
