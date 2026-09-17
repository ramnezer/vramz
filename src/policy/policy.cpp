#include "vramz/policy.hpp"

#include "vramz/checked.hpp"
#include "vramz/saturating.hpp"

#include <algorithm>
#include <limits>
#include <tuple>

namespace vramz {

double PolicyStats::gpu_effective_ratio() const noexcept {
    const auto physical = checked_add(gpu_raw_charge.value(), gpu_compressed_charge.value(),
                                      OperationId::budget_transfer);
    return !physical || physical.value() == 0U
               ? 0.0
               : static_cast<double>(logical_gpu_resident_bytes.value()) /
                     static_cast<double>(physical.value());
}

namespace policy {

Result<void> validate(const PolicyConfig& config) noexcept {
    const auto& t = config.tuning;
    if ((config.mode != PolicyMode::disabled && config.mode != PolicyMode::gpu_resident &&
         config.mode != PolicyMode::gpu_resident_with_host_fallback) ||
        (t.strategy != PolicyStrategy::adaptive && t.strategy != PolicyStrategy::oldest_unpinned) ||
        t.minimum_savings_basis_points > 10000U || t.hot_epochs >= t.warm_epochs ||
        t.frequency_window == 0U || t.hot_frequency == 0U || t.hot_frequency > 16U ||
        t.reevaluate_epochs == 0U || t.thrash_window == 0U || t.thrash_transition_count < 2U ||
        t.maximum_candidates == 0U || t.maximum_candidates > 65536U ||
        t.maximum_transitions == 0U || t.maximum_transitions > 1024U) {
        return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    }
    return {};
}

std::uint64_t age(std::uint64_t now, std::uint64_t then) noexcept {
    return now >= then ? now - then : 0U;
}

std::uint32_t frequency(const Metadata& metadata, std::uint64_t now,
                        const PolicyTuning& tuning) noexcept {
    const auto elapsed_windows = age(now, metadata.last_access_epoch) / tuning.frequency_window;
    return elapsed_windows >= 16U ? 0U : metadata.recent_frequency >> elapsed_windows;
}

Temperature temperature(const Metadata& metadata, std::uint64_t now,
                        const PolicyTuning& tuning) noexcept {
    const auto elapsed = age(now, metadata.last_access_epoch);
    if (elapsed <= tuning.hot_epochs ||
        (elapsed <= tuning.warm_epochs &&
         frequency(metadata, now, tuning) >= tuning.hot_frequency)) {
        return Temperature::hot;
    }
    return elapsed <= tuning.warm_epochs ? Temperature::warm : Temperature::cold;
}

Pressure pressure(ByteSize charged, ByteSize soft, ByteSize hard, ByteSize reserve,
                  ByteSize required) noexcept {
    if (reserve > hard || charged >= hard) {
        return Pressure::critical;
    }
    const auto normal = hard.value() - reserve.value();
    if (charged.value() > normal || required.value() > normal - charged.value()) {
        return Pressure::critical;
    }
    const auto soft_value = std::min(soft.value(), normal);
    const auto high = normal - (normal - soft_value) / 4U;
    if (charged.value() >= high) {
        return Pressure::hard;
    }
    return charged.value() > soft_value ? Pressure::soft : Pressure::normal;
}

std::optional<ByteSize> compression_limit(ByteSize raw, const PolicyTuning& tuning) noexcept {
    // Quotient/remainder arithmetic avoids raw * basis_points overflow, including UINT64_MAX.
    const auto points = static_cast<std::uint64_t>(tuning.minimum_savings_basis_points);
    if (points > 10000U) {
        return std::nullopt;
    }
    const auto ratio_saving =
        (raw.value() / 10000U) * points + ((raw.value() % 10000U) * points + 9999U) / 10000U;
    const auto saving =
        std::max({std::uint64_t{1U}, tuning.minimum_savings_bytes.value(), ratio_saving});
    if (saving >= raw.value()) {
        return std::nullopt;
    }
    return ByteSize{raw.value() - saving};
}

std::uint64_t cost(ByteSize bytes, std::uint64_t per_block) noexcept {
    const auto blocks = bytes.value() / 4096U + (bytes.value() % 4096U == 0U ? 0U : 1U);
    const auto units = checked_mul(blocks, per_block, OperationId::migrate);
    return units ? units.value() : std::numeric_limits<std::uint64_t>::max();
}

void observe(Metadata& metadata, const ChunkSnapshot& chunk, ByteSize raw_charge,
             std::uint64_t now) noexcept {
    if (!metadata.initialized) {
        metadata.initialized = true;
        metadata.raw_since = now;
        metadata.last_transition_epoch = now;
        metadata.observed_state = chunk.authoritative.state;
    }
    if (metadata.content != chunk.authoritative.content) {
        metadata.content = chunk.authoritative.content;
        metadata.compressibility = Compressibility::unknown;
        metadata.compressed_charge = ByteSize{};
        metadata.retry_after = 0U;
        metadata.last_attempt_failed = false;
    }
    metadata.raw_charge = raw_charge;
    if (metadata.observed_state != chunk.authoritative.state) {
        metadata.observed_state = chunk.authoritative.state;
        metadata.last_transition_epoch = now;
        if (chunk.authoritative.state == RepresentationState::gpu_raw) {
            metadata.raw_since = now;
        }
    }
}

void access(Metadata& metadata, std::uint64_t now, const PolicyTuning& tuning) noexcept {
    metadata.recent_frequency = std::min(16U, frequency(metadata, now, tuning) + 1U);
    metadata.last_access_epoch = now;
    saturating_increment(metadata.access_count);
}

void transition(Metadata& metadata, RepresentationState destination, std::uint64_t now,
                const PolicyTuning& tuning) noexcept {
    if (age(now, metadata.transition_window_start) > tuning.thrash_window) {
        metadata.transition_window_start = now;
        metadata.window_transitions = 0U;
    }
    if (metadata.window_transitions < std::numeric_limits<std::uint32_t>::max()) {
        ++metadata.window_transitions;
    }
    if (metadata.window_transitions >= tuning.thrash_transition_count) {
        metadata.thrash_until = saturating_add(now, tuning.thrash_hold_epochs);
        metadata.window_transitions = 0U;
        metadata.transition_window_start = now;
        saturating_increment(metadata.thrash_events);
    }
    metadata.last_transition_epoch = now;
    metadata.observed_state = destination;
    if (destination == RepresentationState::gpu_raw) {
        metadata.raw_since = now;
    }
}

void compression_result(Metadata& metadata, ByteSize compressed, std::uint64_t now,
                        const PolicyTuning& tuning) noexcept {
    metadata.compressed_charge = compressed;
    metadata.last_attempt_failed = false;
    const auto limit = compression_limit(metadata.raw_charge, tuning);
    metadata.compressibility =
        compressed >= metadata.raw_charge
            ? Compressibility::incompressible
            : (!limit || compressed > *limit ? Compressibility::poorly_compressible
                                             : Compressibility::compressible);
    metadata.retry_after = saturating_add(now, tuning.reevaluate_epochs);
}

Proposal propose(const Metadata& metadata, const ChunkSnapshot& chunk, BufferId buffer,
                 Pressure current_pressure, std::uint64_t now, const PolicyConfig& config,
                 bool fallback) noexcept {
    Proposal proposal{};
    if (config.mode == PolicyMode::disabled || current_pressure == Pressure::normal ||
        chunk.lifecycle != LifecycleState::live || chunk.transition_active ||
        chunk.read_pins != 0U || chunk.write_pin || chunk.lease_intents != 0U ||
        chunk.cleanup_resource_count != 0U ||
        tier_of(chunk.authoritative.state) != PhysicalTier::gpu) {
        return proposal;
    }
    const auto& t = config.tuning;
    const bool naive = t.strategy == PolicyStrategy::oldest_unpinned;
    const auto heat = temperature(metadata, now, t);
    if (!naive && (now < metadata.thrash_until ||
                   (metadata.last_attempt_failed && now < metadata.retry_after) ||
                   age(now, metadata.last_transition_epoch) < t.transition_cooldown ||
                   (heat == Temperature::hot && current_pressure == Pressure::soft))) {
        return proposal;
    }
    if (fallback) {
        if (config.mode != PolicyMode::gpu_resident_with_host_fallback) {
            return proposal;
        }
        proposal.action = PolicyAction::host_fallback;
        proposal.expected_saved_bytes = chunk.authoritative.charge;
    } else {
        if (chunk.authoritative.state != RepresentationState::gpu_raw ||
            !compression_limit(chunk.authoritative.charge, t) ||
            (!naive && (age(now, metadata.raw_since) < t.minimum_raw_epochs ||
                        ((metadata.compressibility == Compressibility::incompressible ||
                          metadata.compressibility == Compressibility::poorly_compressible) &&
                         now < metadata.retry_after)))) {
            return proposal;
        }
        proposal.action = PolicyAction::compress_gpu;
        proposal.expected_saved_bytes =
            ByteSize{metadata.compressed_charge != ByteSize{} &&
                             metadata.compressed_charge < metadata.raw_charge
                         ? metadata.raw_charge.value() - metadata.compressed_charge.value()
                         : metadata.raw_charge.value() / 2U};
    }
    proposal.buffer = buffer;
    proposal.chunk = chunk.id;
    proposal.temperature = heat;
    proposal.frequency = frequency(metadata, now, t);
    proposal.last_access = metadata.last_access_epoch;
    proposal.simulated_restore_cost = cost(metadata.raw_charge, t.restore_cost_per_block);
    return proposal;
}

bool preferred(const Proposal& candidate, const Proposal& current,
               PolicyStrategy strategy) noexcept {
    if (candidate.action == PolicyAction::keep) {
        return false;
    }
    if (current.action == PolicyAction::keep) {
        return true;
    }
    if (strategy == PolicyStrategy::oldest_unpinned) {
        return std::tuple{candidate.last_access, candidate.buffer, candidate.chunk} <
               std::tuple{current.last_access, current.buffer, current.chunk};
    }
    if (candidate.temperature != current.temperature) {
        return candidate.temperature > current.temperature;
    }
    if (candidate.frequency != current.frequency) {
        return candidate.frequency < current.frequency;
    }
    if (candidate.expected_saved_bytes != current.expected_saved_bytes) {
        return candidate.expected_saved_bytes > current.expected_saved_bytes;
    }
    return std::tuple{candidate.simulated_restore_cost, candidate.last_access, candidate.buffer,
                      candidate.chunk} < std::tuple{current.simulated_restore_cost,
                                                    current.last_access, current.buffer,
                                                    current.chunk};
}

} // namespace policy
} // namespace vramz
