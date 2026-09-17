#pragma once

#include "vramz/policy_types.hpp"
#include "vramz/state.hpp"

#include <cstdint>
#include <optional>

namespace vramz {

namespace policy {

// Advisory, fixed-size, one record per chunk. Never owns resources or determines validity.
struct Metadata final {
    ContentTag content{};
    ByteSize raw_charge{};
    ByteSize compressed_charge{};
    std::uint64_t last_access_epoch{};
    std::uint64_t access_count{};
    std::uint64_t last_transition_epoch{};
    std::uint64_t raw_since{};
    std::uint64_t retry_after{};
    std::uint64_t last_attempt_cycle{};
    std::uint64_t last_fallback_cycle{};
    std::uint64_t stale_rejection_cycle{};
    std::uint64_t transition_window_start{};
    std::uint64_t thrash_until{};
    std::uint64_t compression_attempts{};
    std::uint64_t compression_failures{};
    std::uint64_t thrash_events{};
    std::uint32_t recent_frequency{};
    std::uint32_t window_transitions{};
    RepresentationState observed_state{RepresentationState::gpu_raw};
    Compressibility compressibility{Compressibility::unknown};
    PolicyAction last_action{PolicyAction::keep};
    bool initialized{};
    bool last_attempt_failed{};
};

struct Proposal final {
    PolicyAction action{PolicyAction::keep};
    ByteSize expected_saved_bytes{};
    std::uint64_t simulated_restore_cost{};
    std::uint32_t frequency{};
    Temperature temperature{Temperature::hot};
    std::uint64_t last_access{};
    BufferId buffer{};
    ChunkId chunk{};
};

[[nodiscard]] Result<void> validate(const PolicyConfig& config) noexcept;
[[nodiscard]] std::uint64_t age(std::uint64_t now, std::uint64_t then) noexcept;
[[nodiscard]] std::uint32_t frequency(const Metadata& metadata, std::uint64_t now,
                                      const PolicyTuning& tuning) noexcept;
[[nodiscard]] Temperature temperature(const Metadata& metadata, std::uint64_t now,
                                      const PolicyTuning& tuning) noexcept;
[[nodiscard]] Pressure pressure(ByteSize charged, ByteSize soft, ByteSize hard, ByteSize reserve,
                                ByteSize required = {}) noexcept;
[[nodiscard]] std::optional<ByteSize> compression_limit(ByteSize raw,
                                                        const PolicyTuning& tuning) noexcept;
[[nodiscard]] std::uint64_t cost(ByteSize bytes, std::uint64_t per_block) noexcept;
void observe(Metadata& metadata, const ChunkSnapshot& chunk, ByteSize raw_charge,
             std::uint64_t now) noexcept;
void access(Metadata& metadata, std::uint64_t now, const PolicyTuning& tuning) noexcept;
void transition(Metadata& metadata, RepresentationState destination, std::uint64_t now,
                const PolicyTuning& tuning) noexcept;
void compression_result(Metadata& metadata, ByteSize compressed, std::uint64_t now,
                        const PolicyTuning& tuning) noexcept;
[[nodiscard]] Proposal propose(const Metadata& metadata, const ChunkSnapshot& chunk,
                               BufferId buffer, Pressure pressure, std::uint64_t now,
                               const PolicyConfig& config, bool fallback = false) noexcept;
[[nodiscard]] bool preferred(const Proposal& candidate, const Proposal& current,
                             PolicyStrategy strategy) noexcept;

} // namespace policy
} // namespace vramz
