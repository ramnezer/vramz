#include "vramz/runtime.hpp"

#include "vramz/checked.hpp"
#include "vramz/detail/allocation.hpp"
#include "vramz/detail/compression.hpp"
#include "vramz/detail/cuda_vmm_backend.hpp"
#include "vramz/mock_backend.hpp"
#include "vramz/policy.hpp"
#include "vramz/saturating.hpp"
#include "vramz/testing.hpp"
#include "vramz/timing.hpp"
#include "vramz/transaction.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace vramz {
namespace {

[[nodiscard]] bool tier_disabled(const TierBudget& budget) noexcept {
    return budget.hard_limit.value() == 0U && budget.soft_target.value() == 0U &&
           budget.migration_reserve.value() == 0U;
}

[[nodiscard]] Result<void> validate_tier(const TierBudget& budget, bool allow_disabled) noexcept {
    if (tier_disabled(budget)) {
        return allow_disabled ? Result<void>{}
                              : Result<void>{make_error(ErrorCode::invalid_argument,
                                                        OperationId::runtime_create)};
    }
    if (budget.hard_limit.value() == 0U || budget.soft_target.value() > budget.hard_limit.value() ||
        budget.migration_reserve.value() > budget.hard_limit.value()) {
        return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    }
    return {};
}

[[nodiscard]] Result<std::uint64_t> greatest_common_divisor(std::uint64_t left,
                                                            std::uint64_t right) noexcept {
    if (left == 0U || right == 0U) {
        return make_error(ErrorCode::invalid_alignment, OperationId::runtime_create);
    }
    while (right != 0U) {
        const auto remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

[[nodiscard]] Result<std::uint64_t> least_common_multiple(std::uint64_t left,
                                                          std::uint64_t right) noexcept {
    const auto divisor = greatest_common_divisor(left, right);
    if (!divisor) {
        return divisor.error();
    }
    return checked_mul(left / divisor.value(), right, OperationId::runtime_create);
}

[[nodiscard]] Result<ByteSize> choose_chunk_size(const RuntimeConfig& config,
                                                 const BackendCapabilities& capabilities) noexcept {
    if (config.preferred_chunk_size.has_value()) {
        return *config.preferred_chunk_size;
    }
    auto granularity = least_common_multiple(capabilities.reservation_granularity.value(),
                                             capabilities.mapping_granularity.value());
    if (!granularity) {
        return granularity.error();
    }
    granularity =
        least_common_multiple(granularity.value(), capabilities.allocation_granularity.value());
    if (!granularity) {
        return granularity.error();
    }
    return ByteSize{granularity.value()};
}

} // namespace

Result<void> validate_config(const RuntimeConfig& config,
                             const BackendCapabilities& capabilities) noexcept {
    const auto policy_valid = policy::validate(config.policy);
    if (!policy_valid) {
        return policy_valid.error();
    }
    if (config.policy.mode == PolicyMode::gpu_resident_with_host_fallback &&
        (tier_disabled(config.budgets.host) || !capabilities.host_tier)) {
        return make_error(ErrorCode::unsupported, OperationId::runtime_create);
    }
    const auto gpu = validate_tier(config.budgets.gpu, false);
    if (!gpu) {
        return gpu.error();
    }
    const auto host = validate_tier(config.budgets.host, true);
    if (!host) {
        return host.error();
    }
    if (!tier_disabled(config.budgets.host) && !capabilities.host_tier) {
        return make_error(ErrorCode::unsupported, OperationId::runtime_create);
    }
    if (config.required_capabilities.host_tier &&
        (tier_disabled(config.budgets.host) || !capabilities.host_tier)) {
        return make_error(ErrorCode::unsupported, OperationId::runtime_create);
    }
    if (config.required_capabilities.stable_device_address && !capabilities.stable_device_address) {
        return make_error(ErrorCode::unsupported, OperationId::runtime_create);
    }
    if (config.required_capabilities.asynchronous_operations &&
        !capabilities.asynchronous_operations) {
        return make_error(ErrorCode::unsupported, OperationId::runtime_create);
    }
    if (config.maximum_pending_completions == 0U ||
        config.maximum_pending_completions > max_pending_completions ||
        config.async_errors.max_retained_errors == 0U ||
        config.async_errors.max_retained_errors > max_async_error_capacity) {
        return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    }
    if (capabilities.reservation_granularity.value() == 0U ||
        capabilities.mapping_granularity.value() == 0U ||
        capabilities.allocation_granularity.value() == 0U) {
        return make_error(ErrorCode::invalid_alignment, OperationId::runtime_create);
    }
    const auto chunk = choose_chunk_size(config, capabilities);
    if (!chunk || chunk.value().value() == 0U) {
        return chunk ? make_error(ErrorCode::invalid_argument, OperationId::runtime_create)
                     : chunk.error();
    }
    const auto size = chunk.value().value();
    if (capabilities.maximum_chunk_size != ByteSize{} &&
        chunk.value() > capabilities.maximum_chunk_size) {
        return make_error(ErrorCode::unsupported, OperationId::runtime_create, 0U, size);
    }
    if ((size % capabilities.reservation_granularity.value()) != 0U ||
        (size % capabilities.mapping_granularity.value()) != 0U ||
        (size % capabilities.allocation_granularity.value()) != 0U) {
        return make_error(ErrorCode::invalid_alignment, OperationId::runtime_create, 0U, size);
    }
    return {};
}

namespace detail {

struct DeferredLeaseEntry final {
    CompletionToken token{};
    std::shared_ptr<BufferState> buffer{};
    MemoryRange range{};
    AccessMode access{AccessMode::read_only};
    std::unique_ptr<std::uint64_t[]> chunk_indices{};
    std::unique_ptr<ContentTag[]> initial_contents{};
    std::uint64_t chunk_count{};
    bool occupied{};
    bool processing{};
    bool completion_proven{};
};

struct RuntimeState final {
    RuntimeState(RuntimeConfig runtime_config, std::unique_ptr<StorageBackend> storage,
                 RuntimeId identity)
        : config(runtime_config), backend(std::move(storage)), ledger(config.budgets),
          async_errors(config.async_errors.max_retained_errors),
          coordinator(ledger, *backend, async_errors), id(identity) {}
    ~RuntimeState() noexcept {
        if (std::ranges::any_of(deferred, [](const auto& entry) { return entry.occupied; })) {
            async_errors.push(
                make_error(ErrorCode::ambiguous_backend_state, OperationId::shutdown));
            std::terminate();
        }
    }

    RuntimeConfig config;
    std::unique_ptr<StorageBackend> backend;
    // Non-owning test-control bridge, set only by the default Mock factory.
    MockBackend* mock_controls{};
    BudgetLedger ledger;
    AsyncErrorChannel async_errors;
    TransactionCoordinator coordinator;
    RuntimeId id{};
    std::array<DeferredLeaseEntry, max_pending_completions> deferred{};
    CompletionStats completion_stats{};
    PolicyStats policy_stats{};
    PerformanceStats performance{};
    std::uint64_t policy_cycle{};
    std::mutex allocation_mutex{};
    std::mutex mutex{};
    std::array<std::weak_ptr<BufferState>, max_runtime_buffers> buffers{};
    std::array<std::shared_ptr<BufferState>, max_runtime_buffers> quarantine{};
    std::uint64_t next_buffer_id{1U};
    std::uint64_t operations_in_flight{};
    std::array<CleanupResource, 4U> orphan_cleanup{};
    std::uint32_t orphan_cleanup_count{};
    bool shutting_down{};
    bool shutdown_in_progress{};
    bool shutdown_complete{};
};

struct BufferState final {
    RuntimeState* runtime{};
    BufferId id{};
    std::uint32_t registry_slot{max_runtime_buffers};
    ByteSize logical_size{};
    ByteSize chunk_size{};
    DeviceAddress base_address{};
    AddressReservation reservation{};
    std::vector<std::unique_ptr<Chunk>> chunks{};
    std::vector<policy::Metadata> policy_metadata{};
    mutable std::mutex close_mutex{};
    bool closing{};
    bool dead{};
    bool cleanup_in_progress{};
};

} // namespace detail

namespace {

[[noreturn]] void completion_fatal(detail::RuntimeState& state, OperationId operation) noexcept {
    state.async_errors.push(make_error(ErrorCode::backend_contract_violation, operation));
    std::terminate();
}

void check_completion_error(detail::RuntimeState& state, Error error) noexcept {
    if (error.code == ErrorCode::ambiguous_backend_state ||
        error.code == ErrorCode::backend_contract_violation) {
        completion_fatal(state, error.operation);
    }
}

[[nodiscard]] Result<RuntimeId> next_runtime_identity() noexcept {
    static std::mutex identity_mutex;
    static std::uint64_t next_id{1U};
    const std::scoped_lock lock{identity_mutex};
    if (next_id == std::numeric_limits<std::uint64_t>::max()) {
        return make_error(ErrorCode::arithmetic_overflow, OperationId::runtime_create);
    }
    return RuntimeId{next_id++};
}

[[nodiscard]] bool policy_enabled(const detail::RuntimeState& state) noexcept {
    return state.config.policy.mode != PolicyMode::disabled;
}

// All advisory metadata/counters are protected by lifecycle -> buffer locks. Execution uses
// the existing outer allocation gate; no lifecycle/buffer lock is held across a transaction.
[[nodiscard]] std::uint64_t policy_tick(detail::RuntimeState& state) noexcept {
    const std::scoped_lock lock{state.mutex};
    saturating_increment(state.policy_stats.access_epoch);
    return state.policy_stats.access_epoch;
}

[[nodiscard]] ByteSize raw_charge(const detail::RuntimeState& state,
                                  const ChunkSnapshot& snapshot) noexcept {
    if (is_raw(snapshot.authoritative.state)) {
        return snapshot.authoritative.charge;
    }
    const auto aligned = checked_align_up(
        snapshot.authoritative.metadata.logical_size.value(),
        state.backend->capabilities().allocation_granularity.value(), OperationId::migrate);
    return aligned ? ByteSize{aligned.value()} : ByteSize{};
}

struct PolicyCandidate final {
    std::shared_ptr<detail::BufferState> buffer{};
    std::size_t index{};
    ChunkSnapshot snapshot{};
    policy::Proposal proposal{};
    bool scan_complete{true};
};

[[nodiscard]] PolicyCandidate select_candidate(detail::RuntimeState& state, Pressure pressure,
                                               std::uint64_t cycle, std::uint32_t& inspected,
                                               bool fallback) noexcept {
    PolicyCandidate selected{};
    const std::scoped_lock lock{state.mutex};
    const PerformanceSample timing{state.config.collect_performance,
                                   state.performance.policy_selection};
    const auto now = state.policy_stats.access_epoch;
    const auto& config = state.config.policy;
    for (const auto& weak : state.buffers) {
        const auto buffer = weak.lock();
        if (!buffer) {
            continue;
        }
        const std::scoped_lock close_lock{buffer->close_mutex};
        if (buffer->closing || buffer->dead) {
            continue;
        }
        for (std::size_t index = 0U; index < buffer->chunks.size(); ++index) {
            if (inspected == config.tuning.maximum_candidates) {
                selected.scan_complete = false;
                return selected;
            }
            ++inspected;
            saturating_increment(state.policy_stats.candidates_inspected);
            const auto snapshot = buffer->chunks[index]->snapshot();
            auto& metadata = buffer->policy_metadata[index];
            policy::observe(metadata, snapshot, raw_charge(state, snapshot), now);
            if (metadata.stale_rejection_cycle == cycle ||
                (fallback ? metadata.last_fallback_cycle : metadata.last_attempt_cycle) == cycle) {
                continue;
            }
            const auto proposal =
                policy::propose(metadata, snapshot, buffer->id, pressure, now, config, fallback);
            if (!fallback && snapshot.authoritative.state == RepresentationState::gpu_raw &&
                proposal.action == PolicyAction::keep && now < metadata.retry_after &&
                (metadata.compressibility == Compressibility::incompressible ||
                 metadata.compressibility == Compressibility::poorly_compressible)) {
                saturating_increment(state.policy_stats.compression_skipped_incompressible);
            }
            if (policy::preferred(proposal, selected.proposal, config.tuning.strategy)) {
                selected.buffer = buffer;
                selected.index = index;
                selected.snapshot = snapshot;
                selected.proposal = proposal;
            }
        }
    }
    return selected;
}

void record_policy_transition(detail::RuntimeState& state, detail::BufferState& buffer,
                              std::size_t index, RepresentationState destination,
                              PolicyAction action) noexcept {
    const std::scoped_lock lock{state.mutex};
    const std::scoped_lock close_lock{buffer.close_mutex};
    auto& metadata = buffer.policy_metadata[index];
    const auto old_thrash = metadata.thrash_events;
    policy::transition(metadata, destination, state.policy_stats.access_epoch,
                       state.config.policy.tuning);
    state.policy_stats.thrash_events =
        saturating_add(state.policy_stats.thrash_events, metadata.thrash_events - old_thrash);
    metadata.last_action = action;
    saturating_increment(state.policy_stats.policy_transitions);
    if (action == PolicyAction::restore_gpu_raw) {
        saturating_increment(state.policy_stats.restore_count);
        state.policy_stats.simulated_restore_cost = saturating_add(
            state.policy_stats.simulated_restore_cost,
            policy::cost(metadata.raw_charge, state.config.policy.tuning.restore_cost_per_block));
    }
}

class PolicyTransactionTrace final : public TransactionObserver {
  public:
    explicit PolicyTransactionTrace(testing::PolicySelectionObserver* observer) noexcept
        : observer_(observer) {}
    void on_phase(TransactionPhase phase, const ChunkSnapshot& snapshot) noexcept override {
        if (observer_ != nullptr) {
            observer_->on_transaction_phase(phase, snapshot);
        }
    }

  private:
    testing::PolicySelectionObserver* observer_{};
};

// Caller owns allocation_mutex. Inspection and attempts are bounded across the entire cycle,
// including fallback. An incomplete GPU scan never authorizes a host fallback decision.
[[nodiscard]] Result<void>
reclaim_locked(detail::RuntimeState& state, ByteSize target, bool admission_pressure,
               testing::PolicySelectionObserver* observer = nullptr) noexcept {
    if (!policy_enabled(state)) {
        return {};
    }
    std::uint64_t cycle = 0U;
    {
        const std::scoped_lock lock{state.mutex};
        if (state.shutting_down) {
            return make_error(ErrorCode::shutting_down, OperationId::migrate);
        }
        if (state.policy_cycle == std::numeric_limits<std::uint64_t>::max()) {
            return make_error(ErrorCode::arithmetic_overflow, OperationId::migrate);
        }
        cycle = ++state.policy_cycle;
        saturating_increment(state.policy_stats.policy_cycles);
    }
    const auto& budget = state.config.budgets.gpu;
    std::uint32_t inspected = 0U;
    std::uint32_t attempted = 0U;
    bool fallback = false;
    while (state.ledger.charged(PhysicalTier::gpu) > target &&
           attempted < state.config.policy.tuning.maximum_transitions) {
        auto pressure =
            policy::pressure(state.ledger.charged(PhysicalTier::gpu), budget.soft_target,
                             budget.hard_limit, budget.migration_reserve);
        pressure = admission_pressure ? Pressure::critical : std::max(pressure, Pressure::soft);
        {
            const std::scoped_lock lock{state.mutex};
            if (state.shutting_down) {
                return make_error(ErrorCode::shutting_down, OperationId::migrate);
            }
            state.policy_stats.current_pressure = pressure;
        }
        auto selected = select_candidate(state, pressure, cycle, inspected, fallback);
        if (!selected.buffer) {
            if (!fallback && selected.scan_complete && admission_pressure &&
                state.config.policy.mode == PolicyMode::gpu_resident_with_host_fallback) {
                fallback = true;
                continue;
            }
            break;
        }
        ++attempted;
        if (observer != nullptr) {
            observer->on_selected(selected.snapshot, selected.proposal);
        }
        const auto source = selected.snapshot.authoritative;
        const auto destination = fallback
                                     ? (is_raw(source.state) ? RepresentationState::host_raw
                                                             : RepresentationState::host_compressed)
                                     : RepresentationState::gpu_compressed;
        MigrationConstraints constraints{source.resource, source.content, {}};
        constraints.expected_access_revision = selected.snapshot.access_revision;
        if (!fallback) {
            constraints.maximum_destination_charge =
                policy::compression_limit(source.charge, state.config.policy.tuning);
        }
        {
            const std::scoped_lock lock{state.mutex};
            const std::scoped_lock close_lock{selected.buffer->close_mutex};
            auto& metadata = selected.buffer->policy_metadata[selected.index];
            auto& stats = state.policy_stats;
            (fallback ? metadata.last_fallback_cycle : metadata.last_attempt_cycle) = cycle;
            stats.last_victim_buffer = selected.buffer->id;
            stats.last_victim_chunk = selected.snapshot.id;
            stats.last_action = selected.proposal.action;
            metadata.last_action = selected.proposal.action;
            // Intentional modulo-2^64 decision hashing, not byte/accounting arithmetic.
            stats.decision_digest =
                (stats.decision_digest ^ selected.buffer->id.value()) * 1099511628211ULL;
            stats.decision_digest =
                (stats.decision_digest ^ selected.snapshot.id.value()) * 1099511628211ULL;
            stats.decision_digest =
                (stats.decision_digest ^ static_cast<std::uint64_t>(selected.proposal.action)) *
                1099511628211ULL;
        }
        PolicyTransactionTrace trace{observer};
        TimingCounter transition_time{};
        PerformanceSample transition_timer{state.config.collect_performance, transition_time};
        const auto result =
            state.coordinator.migrate(*selected.buffer->chunks[selected.index], destination, false,
                                      observer != nullptr ? &trace : nullptr, &constraints);
        transition_timer.finish();
        const auto after = selected.buffer->chunks[selected.index]->snapshot();
        if (observer != nullptr) {
            observer->on_completed(after, result);
        }
        const bool stale = !result && result.error().code == ErrorCode::conflict;
        const bool committed = !stale && after.authoritative.resource != source.resource &&
                               after.authoritative.state == destination;
        if (committed) {
            record_policy_transition(state, *selected.buffer, selected.index, destination,
                                     selected.proposal.action);
        }
        {
            const std::scoped_lock lock{state.mutex};
            const std::scoped_lock close_lock{selected.buffer->close_mutex};
            auto& metadata = selected.buffer->policy_metadata[selected.index];
            auto& stats = state.policy_stats;
            auto& measured = state.performance.policy_transition;
            const auto elapsed =
                checked_add(measured.nanoseconds, transition_time.nanoseconds, OperationId::verify);
            const auto samples =
                checked_add(measured.samples, transition_time.samples, OperationId::verify);
            if (!elapsed || !samples || transition_time.overflow) {
                measured.overflow = true;
            } else {
                measured.nanoseconds = elapsed.value();
                measured.samples = samples.value();
            }
            if (stale) {
                // A stale victim is excluded across all actions in this cycle. A genuine
                // nonbeneficial result still permits the separate emergency fallback pass.
                metadata.stale_rejection_cycle = cycle;
                saturating_increment(stats.stale_proposal_rejections);
            } else if (!fallback) {
                saturating_increment(metadata.compression_attempts);
                saturating_increment(stats.compression_attempts);
                stats.simulated_compression_cost = saturating_add(
                    stats.simulated_compression_cost,
                    policy::cost(source.metadata.logical_size,
                                 state.config.policy.tuning.compression_cost_per_block));
                if (committed ||
                    (!result && result.error().code == ErrorCode::compression_not_beneficial)) {
                    policy::compression_result(metadata,
                                               committed ? after.authoritative.charge
                                                         : ByteSize{result.error().detail},
                                               stats.access_epoch, state.config.policy.tuning);
                    if (committed) {
                        saturating_increment(stats.compression_successes);
                    } else {
                        saturating_increment(stats.compression_rejected);
                    }
                } else if (!result) {
                    saturating_increment(stats.compression_failures);
                    saturating_increment(metadata.compression_failures);
                    metadata.last_attempt_failed = true;
                    metadata.retry_after = saturating_add(
                        stats.access_epoch, state.config.policy.tuning.transition_cooldown);
                }
            } else if (committed) {
                saturating_increment(stats.host_fallback_count);
                saturating_increment(stats.pcie_transition_count);
                stats.simulated_host_cost =
                    saturating_add(stats.simulated_host_cost,
                                   policy::cost(source.metadata.logical_size,
                                                state.config.policy.tuning.host_cost_per_block));
            }
        }
        if (after.lifecycle == LifecycleState::poisoned || after.cleanup_resource_count != 0U) {
            return after.first_error.value_or(
                result ? make_error(ErrorCode::busy, OperationId::migrate) : result.error());
        }
        // Never infer the next admission from the proposal's predicted savings.
    }
    if (state.ledger.charged(PhysicalTier::gpu) > target) {
        const std::scoped_lock lock{state.mutex};
        saturating_increment(state.policy_stats.bounded_stops);
        return make_error(ErrorCode::out_of_gpu_memory, OperationId::migrate, 0U, target.value());
    }
    return {};
}

[[nodiscard]] Result<void> restore_with_policy(detail::RuntimeState& state,
                                               detail::BufferState& buffer,
                                               std::size_t index) noexcept {
    auto& chunk = *buffer.chunks[index];
    if (!policy_enabled(state)) {
        return state.coordinator.migrate(chunk, RepresentationState::gpu_raw, true);
    }
    const std::scoped_lock gate{state.allocation_mutex};
    const auto before = chunk.snapshot();
    const auto raw =
        state.backend->allocation_bound(RepresentationState::gpu_raw, chunk.logical_size());
    const auto workspace = state.backend->workspace_bound(
        before.authoritative.state, RepresentationState::gpu_raw, chunk.logical_size());
    if (!raw || !workspace) {
        return !raw ? raw.error() : workspace.error();
    }
    const auto temporary =
        checked_add(raw.value().value(), workspace.value().value(), OperationId::acquire);
    const auto& budget = state.config.budgets.gpu;
    const auto normal = budget.hard_limit.value() - budget.migration_reserve.value();
    const auto credit = tier_of(before.authoritative.state) == PhysicalTier::gpu
                            ? before.authoritative.charge.value()
                            : 0U;
    const auto growth = raw.value().value() > credit ? raw.value().value() - credit : 0U;
    if (!temporary || temporary.value() > budget.hard_limit.value() || growth > normal) {
        return make_error(ErrorCode::out_of_gpu_memory, OperationId::acquire);
    }
    const ByteSize target{std::min(normal - growth, budget.hard_limit.value() - temporary.value())};
    const auto reclaimed = reclaim_locked(state, target, true);
    if (!reclaimed) {
        return reclaimed.error();
    }
    const auto result = state.coordinator.migrate(chunk, RepresentationState::gpu_raw, true);
    const auto after = chunk.snapshot();
    if (after.authoritative.resource != before.authoritative.resource &&
        after.authoritative.state == RepresentationState::gpu_raw) {
        record_policy_transition(state, buffer, index, RepresentationState::gpu_raw,
                                 PolicyAction::restore_gpu_raw);
        if (tier_of(before.authoritative.state) == PhysicalTier::host) {
            const std::scoped_lock lock{state.mutex};
            saturating_increment(state.policy_stats.pcie_transition_count);
            state.policy_stats.simulated_host_cost = saturating_add(
                state.policy_stats.simulated_host_cost,
                policy::cost(chunk.logical_size(), state.config.policy.tuning.host_cost_per_block));
        }
    }
    return result;
}

void record_access(detail::RuntimeState& state, detail::BufferState& buffer,
                   std::span<const std::uint64_t> indices) noexcept {
    if (!policy_enabled(state)) {
        return;
    }
    const std::scoped_lock lock{state.mutex};
    const std::scoped_lock close_lock{buffer.close_mutex};
    saturating_increment(state.policy_stats.access_epoch);
    for (const auto index : indices) {
        const auto position = static_cast<std::size_t>(index);
        const auto snapshot = buffer.chunks[position]->snapshot();
        auto& metadata = buffer.policy_metadata[position];
        policy::observe(metadata, snapshot, raw_charge(state, snapshot),
                        state.policy_stats.access_epoch);
        policy::access(metadata, state.policy_stats.access_epoch, state.config.policy.tuning);
    }
}

[[nodiscard]] Result<void> retain_failed_buffer(detail::RuntimeState& state,
                                                const std::shared_ptr<detail::BufferState>& buffer,
                                                const Error error) noexcept {
    Error retention_error{};
    bool retention_failed = false;
    {
        const std::scoped_lock lock{state.mutex};
        state.shutting_down = true;
        const std::scoped_lock buffer_lock{buffer->close_mutex};
        if (buffer->runtime != &state || buffer->registry_slot >= state.quarantine.size() ||
            state.buffers[buffer->registry_slot].lock() != buffer) {
            retention_error = make_error(ErrorCode::internal_invariant_violation,
                                         OperationId::close_buffer, buffer->id.value());
            retention_failed = true;
        } else {
            auto& slot = state.quarantine[buffer->registry_slot];
            if (!slot) {
                slot = buffer;
            } else if (slot != buffer) {
                retention_error =
                    make_error(ErrorCode::internal_invariant_violation, OperationId::close_buffer,
                               buffer->id.value(), buffer->registry_slot);
                retention_failed = true;
            }
        }
    }
    state.async_errors.push(error);
    if (retention_failed) {
        state.async_errors.push(retention_error);
        return retention_error;
    }
    return {};
}

[[nodiscard]] Result<void>
cleanup_private_buffer(const std::shared_ptr<detail::BufferState>& buffer) noexcept {
    const auto result = detail::close_buffer_state(*buffer);
    if (!result && result.error().code != ErrorCode::stale_handle) {
        static_cast<void>(retain_failed_buffer(*buffer->runtime, buffer, result.error()));
    }
    return result;
}

class RuntimeOperationGuard final {
  public:
    explicit RuntimeOperationGuard(detail::RuntimeState& state) noexcept : state_(state) {}
    RuntimeOperationGuard(const RuntimeOperationGuard&) = delete;
    RuntimeOperationGuard& operator=(const RuntimeOperationGuard&) = delete;
    ~RuntimeOperationGuard() noexcept {
        const std::scoped_lock lock{state_.mutex};
        if (state_.operations_in_flight != 0U) {
            --state_.operations_in_flight;
        }
    }

  private:
    detail::RuntimeState& state_;
};

[[nodiscard]] Result<void>
reclaim_request(detail::RuntimeState& state, ByteSize target, bool admission_pressure,
                testing::PolicySelectionObserver* observer = nullptr) noexcept {
    if (target > state.config.budgets.gpu.hard_limit) {
        return make_error(ErrorCode::invalid_argument, OperationId::migrate);
    }
    if (!policy_enabled(state)) {
        return {};
    }
    {
        const std::scoped_lock lock{state.mutex};
        if (state.shutting_down) {
            return make_error(ErrorCode::shutting_down, OperationId::migrate);
        }
        if (state.operations_in_flight == std::numeric_limits<std::uint64_t>::max()) {
            return make_error(ErrorCode::arithmetic_overflow, OperationId::migrate);
        }
        ++state.operations_in_flight;
    }
    RuntimeOperationGuard operation_guard{state};
    const std::scoped_lock gate{state.allocation_mutex};
    static_cast<void>(policy_tick(state));
    return reclaim_locked(state, target, admission_pressure, observer);
}

[[nodiscard]] Result<void> remember_runtime_cleanup(detail::RuntimeState& state,
                                                    CleanupResource resource) noexcept {
    const std::scoped_lock lock{state.mutex};
    if (state.orphan_cleanup_count >= state.orphan_cleanup.size()) {
        const auto error = make_error(ErrorCode::internal_invariant_violation, OperationId::release,
                                      resource.resource.value());
        state.async_errors.push(error);
        return error;
    }
    state.orphan_cleanup[state.orphan_cleanup_count] = resource;
    ++state.orphan_cleanup_count;
    return {};
}

} // namespace

namespace detail {

Result<void> close_buffer_state(BufferState& buffer) noexcept {
    std::unique_lock close_lock{buffer.close_mutex};
    if (buffer.dead) {
        return make_error(ErrorCode::stale_handle, OperationId::close_buffer, buffer.id.value());
    }
    if (buffer.cleanup_in_progress) {
        return make_error(ErrorCode::busy, OperationId::close_buffer, buffer.id.value());
    }
    buffer.closing = true;
    buffer.cleanup_in_progress = true;
    close_lock.unlock();
    // Closing freezes layout. A single cleanup owner may call the backend without holding
    // core lifecycle/chunk locks; concurrent close observes busy and retains its ownership.
    struct CleanupGuard final {
        BufferState& state;
        explicit CleanupGuard(BufferState& buffer_state) noexcept : state(buffer_state) {}
        CleanupGuard(const CleanupGuard&) = delete;
        CleanupGuard& operator=(const CleanupGuard&) = delete;
        ~CleanupGuard() noexcept {
            const std::scoped_lock lock{state.close_mutex};
            state.cleanup_in_progress = false;
        }
    } cleanup_guard{buffer};
    for (const auto& chunk : buffer.chunks) {
        const std::scoped_lock chunk_lock{chunk->mutex_};
        if (chunk->read_pins_ != 0U || chunk->write_pin_ || chunk->transition_active_ ||
            chunk->lease_intents_ != 0U) {
            chunk->lifecycle_ = LifecycleState::closing;
            return make_error(ErrorCode::busy, OperationId::close_buffer, chunk->id_.value());
        }
        if (chunk->lifecycle_ == LifecycleState::live) {
            chunk->lifecycle_ = LifecycleState::closing;
        }
    }

    for (const auto& chunk : buffer.chunks) {
        for (;;) {
            CleanupResource cleanup{};
            {
                const std::scoped_lock chunk_lock{chunk->mutex_};
                if (chunk->cleanup_resource_count_ == 0U) {
                    break;
                }
                cleanup = chunk->cleanup_resources_[0];
            }
            const auto cleanup_release =
                buffer.runtime->backend->release(cleanup.resource, ReleasePhase::close);
            if (!cleanup_release || buffer.runtime->backend->owns(cleanup.resource)) {
                const Error error = cleanup_release
                                        ? make_error(ErrorCode::backend_contract_violation,
                                                     OperationId::release, cleanup.resource.value())
                                        : cleanup_release.error();
                if (cleanup_release) {
                    const std::scoped_lock chunk_lock{chunk->mutex_};
                    chunk->lifecycle_ = LifecycleState::poisoned;
                    if (!chunk->first_error_.has_value()) {
                        chunk->first_error_ = error;
                    }
                }
                buffer.runtime->async_errors.push(error);
                return error;
            }
            const auto cleanup_uncharged =
                buffer.runtime->ledger.release_cleanup_debt(cleanup.tier, cleanup.charge);
            if (!cleanup_uncharged) {
                const std::scoped_lock chunk_lock{chunk->mutex_};
                chunk->lifecycle_ = LifecycleState::poisoned;
                if (!chunk->first_error_.has_value()) {
                    chunk->first_error_ = cleanup_uncharged.error();
                }
                return cleanup_uncharged.error();
            }
            const std::scoped_lock chunk_lock{chunk->mutex_};
            if (chunk->cleanup_resource_count_ == 0U ||
                chunk->cleanup_resources_[0].resource != cleanup.resource) {
                const auto error = make_error(ErrorCode::internal_invariant_violation,
                                              OperationId::release, cleanup.resource.value());
                chunk->lifecycle_ = LifecycleState::poisoned;
                if (!chunk->first_error_.has_value()) {
                    chunk->first_error_ = error;
                }
                return error;
            }
            --chunk->cleanup_resource_count_;
            chunk->cleanup_resources_[0] =
                chunk->cleanup_resources_[chunk->cleanup_resource_count_];
            chunk->cleanup_resources_[chunk->cleanup_resource_count_] = CleanupResource{};
        }

        Representation representation{};
        bool retire_authority = false;
        {
            const std::scoped_lock chunk_lock{chunk->mutex_};
            if (chunk->lifecycle_ == LifecycleState::dead) {
                continue;
            }
            representation = chunk->authoritative_;
            retire_authority = !chunk->authority_retired_to_debt_;
        }
        if (retire_authority) {
            const auto retired = buffer.runtime->ledger.retire_committed(
                tier_of(representation.state), representation.charge);
            if (!retired) {
                const std::scoped_lock chunk_lock{chunk->mutex_};
                chunk->lifecycle_ = LifecycleState::poisoned;
                if (!chunk->first_error_.has_value()) {
                    chunk->first_error_ = retired.error();
                }
                return retired.error();
            }
            const std::scoped_lock chunk_lock{chunk->mutex_};
            chunk->authority_retired_to_debt_ = true;
        }
        const auto released =
            buffer.runtime->backend->release(representation.resource, ReleasePhase::close);
        if (!released || buffer.runtime->backend->owns(representation.resource)) {
            const Error error =
                released ? make_error(ErrorCode::backend_contract_violation, OperationId::release,
                                      representation.resource.value())
                         : released.error();
            if (released) {
                const std::scoped_lock chunk_lock{chunk->mutex_};
                chunk->lifecycle_ = LifecycleState::poisoned;
                if (!chunk->first_error_.has_value()) {
                    chunk->first_error_ = error;
                }
            }
            buffer.runtime->async_errors.push(error);
            return error;
        }
        const auto uncharged = buffer.runtime->ledger.release_cleanup_debt(
            tier_of(representation.state), representation.charge);
        if (!uncharged) {
            const std::scoped_lock chunk_lock{chunk->mutex_};
            chunk->lifecycle_ = LifecycleState::poisoned;
            if (!chunk->first_error_.has_value()) {
                chunk->first_error_ = uncharged.error();
            }
            return uncharged.error();
        }
        const std::scoped_lock chunk_lock{chunk->mutex_};
        chunk->authority_retired_to_debt_ = false;
        chunk->lifecycle_ = LifecycleState::dead;
    }
    if (buffer.reservation.id != AddressReservationId{}) {
        const auto released = buffer.runtime->backend->release_address_space(buffer.reservation);
        if (!released || buffer.runtime->backend->owns_address_space(buffer.reservation.id)) {
            const auto error = released
                                   ? make_error(ErrorCode::backend_contract_violation,
                                                OperationId::release, buffer.reservation.id.value())
                                   : released.error();
            buffer.runtime->async_errors.push(error);
            if (released) {
                std::terminate();
            }
            return error;
        }
        const std::scoped_lock lock{buffer.close_mutex};
        buffer.reservation = {};
    }
    {
        const std::scoped_lock lock{buffer.close_mutex};
        buffer.dead = true;
    }
    return {};
}

} // namespace detail

namespace {

[[nodiscard]] Result<std::pair<std::uint64_t, std::uint64_t>>
chunk_span(const detail::BufferState& buffer, MemoryRange range) noexcept {
    if (range.length.value() == 0U) {
        return make_error(ErrorCode::invalid_range, OperationId::acquire);
    }
    const auto end = checked_range_end(range, OperationId::acquire);
    if (!end || end.value().value() > buffer.logical_size.value()) {
        return end ? make_error(ErrorCode::invalid_range, OperationId::acquire) : end.error();
    }
    const auto first = range.offset.value() / buffer.chunk_size.value();
    const auto last = (end.value().value() - 1U) / buffer.chunk_size.value();
    return std::pair{first, last};
}

} // namespace

Runtime::Runtime(std::shared_ptr<detail::RuntimeState> state) noexcept : state_(std::move(state)) {}

Runtime::~Runtime() noexcept {
    if (state_) {
        const auto result = shutdown();
        if (!result && result.error().code != ErrorCode::stale_handle) {
            state_->async_errors.push(result.error());
        }
        bool pending = false;
        {
            const std::scoped_lock lock{state_->mutex};
            pending = std::ranges::any_of(state_->deferred,
                                          [](const auto& entry) { return entry.occupied; });
        }
        if (pending) {
            completion_fatal(*state_, OperationId::shutdown);
        }
    }
}

Result<Runtime> Runtime::create(RuntimeConfig config) noexcept {
    try {
        auto backend = std::make_unique<MockBackend>(MockBackendConfig{});
        auto* controls = backend.get();
        auto created = create_backend(config, std::move(backend));
        if (created) {
            created.value().state_->mock_controls = controls;
        }
        return created;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::runtime_create);
    }
}

Result<Buffer> Runtime::allocate(ByteSize logical_size) noexcept {
    if (!state_ || logical_size.value() == 0U) {
        return make_error(ErrorCode::invalid_argument, OperationId::allocate);
    }
    const auto selected_chunk = choose_chunk_size(state_->config, state_->backend->capabilities());
    if (!selected_chunk) {
        return selected_chunk.error();
    }
    const auto numerator = checked_add(logical_size.value(), selected_chunk.value().value() - 1U,
                                       OperationId::allocate);
    if (!numerator) {
        return numerator.error();
    }
    const auto chunk_count = numerator.value() / selected_chunk.value().value();
    if (chunk_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return make_error(ErrorCode::arithmetic_overflow, OperationId::allocate);
    }
    const auto address_span = checked_align_up(logical_size.value(), selected_chunk.value().value(),
                                               OperationId::allocate);
    if (!address_span) {
        return address_span.error();
    }

    BufferId buffer_id{};
    {
        const std::scoped_lock lock{state_->mutex};
        if (state_->shutting_down) {
            return make_error(ErrorCode::shutting_down, OperationId::allocate);
        }
        if (state_->next_buffer_id == std::numeric_limits<std::uint64_t>::max()) {
            return make_error(ErrorCode::arithmetic_overflow, OperationId::allocate);
        }
        if (state_->operations_in_flight == std::numeric_limits<std::uint64_t>::max()) {
            return make_error(ErrorCode::arithmetic_overflow, OperationId::allocate);
        }
        buffer_id = BufferId{state_->next_buffer_id};
        ++state_->next_buffer_id;
        ++state_->operations_in_flight;
    }
    RuntimeOperationGuard operation_guard{*state_};
    const std::unique_lock allocation_lock{state_->allocation_mutex};
    {
        const std::scoped_lock lock{state_->mutex};
        if (state_->shutting_down) {
            return make_error(ErrorCode::shutting_down, OperationId::allocate);
        }
    }

    std::shared_ptr<detail::BufferState> buffer;
    try {
        buffer = std::make_shared<detail::BufferState>();
        buffer->runtime = state_.get();
        buffer->id = buffer_id;
        buffer->logical_size = logical_size;
        buffer->chunk_size = selected_chunk.value();
        if (chunk_count > static_cast<std::uint64_t>(buffer->chunks.max_size())) {
            return make_error(ErrorCode::arithmetic_overflow, OperationId::allocate);
        }
        buffer->chunks.reserve(static_cast<std::size_t>(chunk_count));
        if (policy_enabled(*state_)) {
            if (chunk_count > static_cast<std::uint64_t>(buffer->policy_metadata.max_size())) {
                return make_error(ErrorCode::arithmetic_overflow, OperationId::allocate);
            }
            buffer->policy_metadata.reserve(static_cast<std::size_t>(chunk_count));
        }
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::allocate);
    }

    bool registered = false;
    {
        const std::scoped_lock lock{state_->mutex};
        for (std::uint32_t index = 0U; index < state_->buffers.size(); ++index) {
            if (state_->buffers[index].expired() && !state_->quarantine[index]) {
                state_->buffers[index] = buffer;
                buffer->registry_slot = index;
                registered = true;
                break;
            }
        }
    }
    if (!registered) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::allocate, 0U,
                          max_runtime_buffers);
    }

    // Chunk sizes may be non-power-of-two multiples of the backend granularities. VA
    // alignment depends on those granularities, not on an optional preferred chunk size.
    const auto reservation_alignment =
        choose_chunk_size(RuntimeConfig{}, state_->backend->capabilities());
    if (!reservation_alignment) {
        static_cast<void>(cleanup_private_buffer(buffer));
        return reservation_alignment.error();
    }
    const auto reservation = state_->backend->reserve_address_space(ByteSize{address_span.value()},
                                                                    reservation_alignment.value());
    if (!reservation) {
        static_cast<void>(cleanup_private_buffer(buffer));
        return reservation.error();
    }
    const auto va = reservation.value();
    if (va.id == AddressReservationId{} || va.base == DeviceAddress{} ||
        va.size != ByteSize{address_span.value()} ||
        va.alignment != reservation_alignment.value() ||
        va.base.value() % va.alignment.value() != 0U ||
        !checked_add(va.base.value(), va.size.value(), OperationId::allocate) ||
        !state_->backend->owns_address_space(va.id)) {
        state_->async_errors.push(
            make_error(ErrorCode::backend_contract_violation, OperationId::allocate));
        std::terminate();
    }
    {
        const std::scoped_lock close_lock{buffer->close_mutex};
        buffer->reservation = va;
        buffer->base_address = va.base;
    }

    std::uint32_t policy_retries = 0U;
    for (std::uint64_t index = 0U; index < chunk_count; ++index) {
        const auto offset_value =
            checked_mul(index, selected_chunk.value().value(), OperationId::allocate);
        if (!offset_value) {
            static_cast<void>(cleanup_private_buffer(buffer));
            return offset_value.error();
        }
        const auto remaining = logical_size.value() - offset_value.value();
        const ByteSize length{std::min(remaining, selected_chunk.value().value())};
        const auto address_value =
            checked_add(buffer->base_address.value(), offset_value.value(), OperationId::allocate);
        if (!address_value) {
            static_cast<void>(cleanup_private_buffer(buffer));
            return address_value.error();
        }
        const auto tag = initial_content_tag(buffer_id, ChunkId{index});
        const DeviceAddress address{address_value.value()};
        std::unique_ptr<Chunk> prepared_chunk;
        try {
            prepared_chunk =
                std::make_unique<Chunk>(ChunkId{index}, ByteOffset{offset_value.value()}, length,
                                        Representation{RepresentationState::gpu_raw, ResourceId{},
                                                       ByteSize{}, tag, address},
                                        address);
        } catch (const std::bad_alloc&) {
            static_cast<void>(cleanup_private_buffer(buffer));
            return make_error(ErrorCode::out_of_host_memory, OperationId::allocate);
        }
        const auto bound = state_->backend->allocation_bound(RepresentationState::gpu_raw, length);
        if (!bound) {
            static_cast<void>(cleanup_private_buffer(buffer));
            return bound.error();
        }
        ReservationRequest request{PhysicalTier::gpu, ChargeBucket::staging, bound.value(),
                                   AdmissionKind::normal};
        ReservationToken token{};
        auto reserved = state_->ledger.reserve(TransactionId{buffer_id.value()},
                                               std::span<const ReservationRequest>{&request, 1U},
                                               std::span<ReservationToken>{&token, 1U});
        if (!reserved && reserved.error().code == ErrorCode::out_of_gpu_memory &&
            policy_enabled(*state_) && policy_retries < 32U) {
            ++policy_retries;
            static_cast<void>(policy_tick(*state_));
            const auto& budget = state_->config.budgets.gpu;
            const auto normal = budget.hard_limit.value() - budget.migration_reserve.value();
            if (bound.value().value() <= normal) {
                const auto reclaimed =
                    reclaim_locked(*state_, ByteSize{normal - bound.value().value()}, true);
                if (!reclaimed) {
                    static_cast<void>(cleanup_private_buffer(buffer));
                    return reclaimed.error();
                }
                reserved = state_->ledger.reserve(TransactionId{buffer_id.value()},
                                                  std::span<const ReservationRequest>{&request, 1U},
                                                  std::span<ReservationToken>{&token, 1U});
            }
        }
        if (!reserved) {
            static_cast<void>(cleanup_private_buffer(buffer));
            return reserved.error();
        }
        const auto storage_granularity = state_->backend->capabilities().allocation_granularity;
        const auto provisional_allocation = state_->backend->allocate_representation(
            RepresentationAllocationRequest{RepresentationState::gpu_raw, length, tag, address});
        if (!provisional_allocation) {
            static_cast<void>(state_->ledger.release_reservation(token));
            static_cast<void>(cleanup_private_buffer(buffer));
            return provisional_allocation.error();
        }
        for (const auto& previous : buffer->chunks) {
            if (previous->snapshot().authoritative.resource == provisional_allocation.value().id) {
                detail::allocation_contract_failure(state_->async_errors,
                                                    provisional_allocation.value());
            }
        }
        const auto allocation = detail::corroborate_allocation(
            *state_->backend, state_->async_errors, provisional_allocation.value(),
            detail::AllocationExpectation{PhysicalTier::gpu, ResourceKind::representation,
                                          bound.value(), storage_granularity, length,
                                          RepresentationState::gpu_raw, address});
        const auto materialized = state_->ledger.materialize(token, allocation.charge);
        if (!materialized) {
            if (!state_->ledger.record_contract_debt(token, allocation.charge) ||
                !remember_runtime_cleanup(
                    *state_, CleanupResource{allocation.id, allocation.tier, allocation.charge})) {
                detail::allocation_contract_failure(state_->async_errors, allocation);
            }
            {
                const std::scoped_lock lock{state_->mutex};
                state_->shutting_down = true;
            }
            state_->async_errors.push(materialized.error());
            static_cast<void>(cleanup_private_buffer(buffer));
            return materialized.error();
        }
        static_cast<void>(state_->ledger.release_reservation(token));
        const auto committed = state_->ledger.commit_initial(PhysicalTier::gpu, allocation.charge);
        if (!committed) {
            const auto released = state_->backend->release(allocation.id, ReleasePhase::rollback);
            if (released && !state_->backend->owns(allocation.id)) {
                static_cast<void>(state_->ledger.release_materialized(
                    PhysicalTier::gpu, ChargeBucket::staging, allocation.charge));
            } else {
                static_cast<void>(state_->ledger.move_materialized_to_debt(
                    PhysicalTier::gpu, ChargeBucket::staging, allocation.charge));
                static_cast<void>(remember_runtime_cleanup(
                    *state_, CleanupResource{allocation.id, allocation.tier, allocation.charge}));
            }
            {
                const std::scoped_lock lock{state_->mutex};
                state_->shutting_down = true;
            }
            state_->async_errors.push(committed.error());
            static_cast<void>(cleanup_private_buffer(buffer));
            return committed.error();
        }
        prepared_chunk->authoritative_ = Representation{
            RepresentationState::gpu_raw, allocation.id, allocation.charge, tag, address,
            allocation.metadata};
        const auto epoch = policy_enabled(*state_) ? policy_tick(*state_) : 0U;
        {
            const std::scoped_lock close_lock{buffer->close_mutex};
            if (policy_enabled(*state_)) {
                policy::Metadata metadata{};
                policy::observe(metadata, prepared_chunk->snapshot(), allocation.charge, epoch);
                metadata.last_access_epoch = epoch;
                buffer->policy_metadata.push_back(metadata);
            }
            buffer->chunks.push_back(std::move(prepared_chunk));
        }
    }

    if (policy_enabled(*state_) &&
        state_->ledger.charged(PhysicalTier::gpu) > state_->config.budgets.gpu.soft_target) {
        const auto soft_reclaim =
            reclaim_locked(*state_, state_->config.budgets.gpu.soft_target, false);
        if (!soft_reclaim && soft_reclaim.error().code != ErrorCode::out_of_gpu_memory) {
            static_cast<void>(cleanup_private_buffer(buffer));
            return soft_reclaim.error();
        }
    }
    bool publish_allowed = false;
    {
        const std::scoped_lock lock{state_->mutex};
        if (!state_->shutting_down) {
            publish_allowed = true;
        }
    }
    if (!publish_allowed) {
        static_cast<void>(cleanup_private_buffer(buffer));
        return make_error(ErrorCode::shutting_down, OperationId::allocate, buffer->id.value());
    }
    return Buffer{state_, std::move(buffer)};
}

Result<void> Runtime::reclaim_to_target(ByteSize target_gpu_charge) noexcept {
    if (!state_) {
        return make_error(ErrorCode::invalid_argument, OperationId::migrate);
    }
    return reclaim_request(*state_, target_gpu_charge, false);
}

RuntimeStats Runtime::stats() const noexcept {
    if (!state_) {
        return {};
    }
    std::uint64_t live = 0U;
    std::uint64_t quarantined = 0U;
    bool shutting_down = false;
    auto compression = state_->backend->compression_stats();
    PolicyStats policy_stats{};
    CompletionStats completion_stats{};
    {
        const std::scoped_lock lock{state_->mutex};
        policy_stats = state_->policy_stats;
        completion_stats = state_->completion_stats;
        shutting_down = state_->shutting_down;
        for (std::size_t index = 0U; index < state_->buffers.size(); ++index) {
            const auto& weak = state_->buffers[index];
            const auto buffer = weak.lock();
            if (buffer) {
                const std::scoped_lock buffer_lock{buffer->close_mutex};
                if (!buffer->dead) {
                    saturating_increment(live);
                    for (const auto& chunk : buffer->chunks) {
                        const auto snapshot = chunk->snapshot();
                        if (snapshot.lifecycle == LifecycleState::live && !buffer->closing) {
                            auto& logical =
                                tier_of(snapshot.authoritative.state) == PhysicalTier::gpu
                                    ? policy_stats.logical_gpu_resident_bytes
                                    : policy_stats.logical_host_bytes;
                            logical = ByteSize{
                                saturating_add(logical.value(), chunk->logical_size().value())};
                            auto& charge =
                                tier_of(snapshot.authoritative.state) == PhysicalTier::host
                                    ? policy_stats.host_charge
                                    : (is_raw(snapshot.authoritative.state)
                                           ? policy_stats.gpu_raw_charge
                                           : policy_stats.gpu_compressed_charge);
                            charge = ByteSize{saturating_add(
                                charge.value(), snapshot.authoritative.charge.value())};
                            if (snapshot.authoritative.state ==
                                RepresentationState::gpu_compressed) {
                                const auto raw = raw_charge(*state_, snapshot);
                                const auto saved =
                                    raw > snapshot.authoritative.charge
                                        ? raw.value() - snapshot.authoritative.charge.value()
                                        : 0U;
                                policy_stats.saved_gpu_bytes = ByteSize{
                                    saturating_add(policy_stats.saved_gpu_bytes.value(), saved)};
                            }
                        }
                        if (!is_raw(snapshot.authoritative.state)) {
                            compression.logical_bytes_represented = ByteSize{saturating_add(
                                compression.logical_bytes_represented.value(),
                                snapshot.authoritative.metadata.logical_size.value())};
                            compression.stored_payload_bytes = ByteSize{saturating_add(
                                compression.stored_payload_bytes.value(),
                                snapshot.authoritative.metadata.stored_size.value())};
                            compression.physical_storage_bytes =
                                ByteSize{saturating_add(compression.physical_storage_bytes.value(),
                                                        snapshot.authoritative.charge.value())};
                        }
                    }
                }
            }
            if (state_->quarantine[index]) {
                saturating_increment(quarantined);
            }
        }
    }
    policy_stats.current_pressure = policy::pressure(
        state_->ledger.charged(PhysicalTier::gpu), state_->config.budgets.gpu.soft_target,
        state_->config.budgets.gpu.hard_limit, state_->config.budgets.gpu.migration_reserve);
    return RuntimeStats{state_->ledger.usage(PhysicalTier::gpu),
                        state_->ledger.usage(PhysicalTier::host),
                        state_->async_errors.stats(),
                        compression,
                        live,
                        quarantined,
                        shutting_down,
                        policy_stats,
                        completion_stats};
}

BufferId Buffer::id() const noexcept { return state_ ? state_->id : BufferId{}; }

Result<ChunkInfo> Buffer::inspect_chunk(std::uint64_t index) const noexcept {
    if (!runtime_ || !state_) {
        return make_error(ErrorCode::stale_handle, OperationId::verify);
    }
    const std::scoped_lock runtime_lock{runtime_->mutex};
    const std::scoped_lock buffer_lock{state_->close_mutex};
    if (state_->dead || index >= state_->chunks.size()) {
        return make_error(ErrorCode::invalid_range, OperationId::verify);
    }
    const auto offset = static_cast<std::size_t>(index);
    const auto snap = state_->chunks[offset]->snapshot();
    ChunkInfo info{};
    info.index = index;
    info.logical_bytes = snap.authoritative.metadata.logical_size;
    info.stored_bytes = snap.authoritative.metadata.stored_size;
    info.physical_charge = snap.authoritative.charge;
    info.logical_crc = snap.authoritative.metadata.crc32c;
    info.stored_crc = snap.authoritative.metadata.stored_crc32c;
    info.residency = static_cast<Residency>(snap.authoritative.state);
    info.access_revision = snap.access_revision;
    if (policy_enabled(*runtime_)) {
        if (offset >= state_->policy_metadata.size()) {
            return make_error(ErrorCode::backend_contract_violation, OperationId::verify);
        }
        const auto& metadata = state_->policy_metadata[offset];
        info.policy_metadata_available = true;
        info.access_epoch = metadata.last_access_epoch;
        info.access_count = metadata.access_count;
        info.retry_after = metadata.retry_after;
        info.compression_attempts = metadata.compression_attempts;
        info.effective_frequency = policy::frequency(metadata, runtime_->policy_stats.access_epoch,
                                                     runtime_->config.policy.tuning);
        info.temperature = policy::temperature(metadata, runtime_->policy_stats.access_epoch,
                                               runtime_->config.policy.tuning);
        info.compressibility = metadata.compressibility;
    }
    info.pending = snap.transition_active || snap.lease_intents != 0U || snap.read_pins != 0U ||
                   snap.write_pin || snap.cleanup_resource_count != 0U;
    info.poisoned = snap.lifecycle == LifecycleState::poisoned;
    return info;
}

Result<ResourceStats> Runtime::resources() const noexcept {
    if (!state_) {
        return make_error(ErrorCode::stale_handle, OperationId::verify);
    }
    // Quiescent accounting observation; callers must join users before final zero checks.
    const std::scoped_lock gate{state_->allocation_mutex};
    ResourceStats result{};
    result.owned_gpu_charge = state_->backend->owned_charge(PhysicalTier::gpu);
    result.ledger_gpu_charge = state_->ledger.charged(PhysicalTier::gpu);
    result.unmaterialized_reservations =
        state_->ledger.unmaterialized_reservations(PhysicalTier::gpu);
    result.backend_resources = state_->backend->owned_resource_count();
    const auto total = checked_add(result.owned_gpu_charge.value(),
                                   result.unmaterialized_reservations.value(), OperationId::verify);
    result.accounting_conserved = total && total.value() == result.ledger_gpu_charge.value();
    if (const auto* cuda = dynamic_cast<detail::CudaVmmBackend*>(state_->backend.get())) {
        const auto counts = cuda->resource_counts();
        result.raw_resources = counts.raw;
        result.compressed_resources = counts.compressed;
        result.workspace_resources = counts.workspace;
        result.va_reservations = cuda->owned_address_count();
        result.retained_context = counts.retained_context;
        result.stream_owned = counts.stream_owned;
    }
    return result;
}

PerformanceStats Runtime::performance() const noexcept {
    if (!state_) {
        return {};
    }
    PerformanceStats result{};
    {
        const std::scoped_lock lock{state_->mutex};
        result = state_->performance;
        result.enabled = state_->config.collect_performance;
    }
    if (const auto* cuda = dynamic_cast<detail::CudaVmmBackend*>(state_->backend.get())) {
        const auto codec = cuda->performance();
        result.compression = codec.compression;
        result.decompression = codec.decompression;
    }
    return result;
}

RuntimeCapabilities Runtime::capabilities() const noexcept {
    if (!state_) {
        return {};
    }
    const auto& capabilities = state_->backend->capabilities();
    const auto compression = state_->backend->compression_available();
    return {capabilities.backend_id,
            true,
            compression,
            capabilities.host_tier,
            capabilities.host_tier && compression,
            capabilities.stable_device_address,
            compression,
            capabilities.supports_external_async_completion,
            capabilities.allocation_granularity,
            capabilities.recommended_allocation_granularity,
            capabilities.maximum_chunk_size,
            HardwareValidationState::not_tested};
}

std::optional<AsyncErrorRecord> Runtime::poll_async_error() noexcept {
    return state_ ? state_->async_errors.poll() : std::nullopt;
}

std::optional<AsyncErrorRecord> Runtime::first_async_error() const noexcept {
    return state_ ? state_->async_errors.first() : std::nullopt;
}

Result<void> Runtime::shutdown() noexcept {
    if (!state_) {
        return make_error(ErrorCode::stale_handle, OperationId::shutdown);
    }
    {
        const std::scoped_lock lock{state_->mutex};
        state_->shutting_down = true;
    }
    const auto drained = poll_ready_completions();
    if (!drained) {
        return drained.error();
    }
    {
        const std::scoped_lock lock{state_->mutex};
        if (state_->shutdown_complete) {
            return make_error(ErrorCode::stale_handle, OperationId::shutdown);
        }
        state_->shutting_down = true;
        if (state_->shutdown_in_progress || state_->operations_in_flight != 0U) {
            return make_error(ErrorCode::busy, OperationId::shutdown);
        }
        if (std::ranges::any_of(state_->deferred,
                                [](const auto& entry) { return entry.occupied; })) {
            return make_error(ErrorCode::busy, OperationId::shutdown);
        }
        for (std::size_t index = 0U; index < state_->buffers.size(); ++index) {
            const auto buffer = state_->buffers[index].lock();
            if (buffer) {
                const std::scoped_lock buffer_lock{buffer->close_mutex};
                if (!buffer->dead && state_->quarantine[index] != buffer) {
                    return make_error(ErrorCode::busy, OperationId::shutdown, buffer->id.value());
                }
            }
        }
        state_->shutdown_in_progress = true;
    }

    for (;;) {
        CleanupResource cleanup{};
        {
            const std::scoped_lock lock{state_->mutex};
            if (state_->orphan_cleanup_count == 0U) {
                break;
            }
            cleanup = state_->orphan_cleanup[0];
        }
        const auto released = state_->backend->release(cleanup.resource, ReleasePhase::close);
        if (!released || state_->backend->owns(cleanup.resource)) {
            const auto error = released ? make_error(ErrorCode::backend_contract_violation,
                                                     OperationId::release, cleanup.resource.value())
                                        : released.error();
            {
                const std::scoped_lock lock{state_->mutex};
                state_->shutdown_in_progress = false;
            }
            state_->async_errors.push(error);
            return error;
        }
        const auto uncharged = state_->ledger.release_cleanup_debt(cleanup.tier, cleanup.charge);
        if (!uncharged) {
            {
                const std::scoped_lock lock{state_->mutex};
                state_->shutdown_in_progress = false;
            }
            state_->async_errors.push(uncharged.error());
            return uncharged.error();
        }
        const std::scoped_lock lock{state_->mutex};
        if (state_->orphan_cleanup_count == 0U ||
            state_->orphan_cleanup[0].resource != cleanup.resource) {
            state_->shutdown_in_progress = false;
            const auto error = make_error(ErrorCode::internal_invariant_violation,
                                          OperationId::release, cleanup.resource.value());
            state_->async_errors.push(error);
            return error;
        }
        --state_->orphan_cleanup_count;
        state_->orphan_cleanup[0] = state_->orphan_cleanup[state_->orphan_cleanup_count];
        state_->orphan_cleanup[state_->orphan_cleanup_count] = CleanupResource{};
    }

    for (;;) {
        std::shared_ptr<detail::BufferState> quarantined;
        std::size_t quarantine_slot = state_->quarantine.size();
        {
            const std::scoped_lock lock{state_->mutex};
            for (std::size_t index = 0U; index < state_->quarantine.size(); ++index) {
                if (state_->quarantine[index]) {
                    quarantined = state_->quarantine[index];
                    quarantine_slot = index;
                    break;
                }
            }
        }
        if (!quarantined) {
            break;
        }

        const auto closed = detail::close_buffer_state(*quarantined);
        if (!closed && closed.error().code != ErrorCode::stale_handle) {
            {
                const std::scoped_lock lock{state_->mutex};
                state_->shutdown_in_progress = false;
            }
            state_->async_errors.push(closed.error());
            return closed.error();
        }
        {
            const std::scoped_lock lock{state_->mutex};
            const std::scoped_lock buffer_lock{quarantined->close_mutex};
            if (quarantine_slot >= state_->quarantine.size() ||
                state_->quarantine[quarantine_slot] != quarantined) {
                state_->shutdown_in_progress = false;
                const auto error = make_error(ErrorCode::internal_invariant_violation,
                                              OperationId::shutdown, quarantined->id.value());
                state_->async_errors.push(error);
                return error;
            }
            state_->quarantine[quarantine_slot].reset();
        }
    }

    const auto finalized = state_->backend->shutdown();
    {
        const std::scoped_lock lock{state_->mutex};
        state_->shutdown_in_progress = false;
        state_->shutdown_complete = static_cast<bool>(finalized);
    }
    if (!finalized) {
        state_->async_errors.push(finalized.error());
    }
    return finalized;
}

Buffer::Buffer(std::shared_ptr<detail::RuntimeState> runtime,
               std::shared_ptr<detail::BufferState> state) noexcept
    : runtime_(std::move(runtime)), state_(std::move(state)) {}

Buffer::~Buffer() noexcept {
    if (state_) {
        const auto result = detail::close_buffer_state(*state_);
        if (!result && result.error().code != ErrorCode::stale_handle &&
            result.error().code != ErrorCode::busy) {
            static_cast<void>(retain_failed_buffer(*runtime_, state_, result.error()));
        }
    }
}

ByteSize Buffer::size() const noexcept { return state_ ? state_->logical_size : ByteSize{}; }

Result<Lease> Buffer::acquire(MemoryRange range, AcquireOptions options) noexcept {
    if (!state_) {
        return make_error(ErrorCode::stale_handle, OperationId::acquire);
    }
    if (options.access != AccessMode::read_only && options.access != AccessMode::read_write) {
        return make_error(ErrorCode::invalid_argument, OperationId::acquire);
    }
    {
        const std::scoped_lock lock{runtime_->mutex};
        if (runtime_->shutting_down) {
            return make_error(ErrorCode::shutting_down, OperationId::acquire);
        }
    }
    {
        const std::scoped_lock lock{state_->close_mutex};
        if (state_->dead) {
            return make_error(ErrorCode::stale_handle, OperationId::acquire, state_->id.value());
        }
        if (state_->closing) {
            return make_error(ErrorCode::shutting_down, OperationId::acquire, state_->id.value());
        }
    }
    const auto span = chunk_span(*state_, range);
    if (!span) {
        return span.error();
    }
    const auto count = span.value().second - span.value().first + 1U;
    std::unique_ptr<std::uint64_t[]> indices;
    std::unique_ptr<ContentTag[]> initial_contents;
    try {
        indices = std::make_unique<std::uint64_t[]>(static_cast<std::size_t>(count));
        initial_contents = std::make_unique<ContentTag[]>(static_cast<std::size_t>(count));
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::acquire);
    }
    for (std::uint64_t index = 0U; index < count; ++index) {
        const auto chunk_index = checked_add(span.value().first, index, OperationId::acquire);
        if (!chunk_index) {
            return chunk_index.error();
        }
        indices[static_cast<std::size_t>(index)] = chunk_index.value();
    }

    std::vector<std::unique_lock<std::mutex>> intent_locks;
    if (count > static_cast<std::uint64_t>(intent_locks.max_size())) {
        return make_error(ErrorCode::arithmetic_overflow, OperationId::acquire);
    }
    try {
        intent_locks.reserve(static_cast<std::size_t>(count));
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::acquire);
    }
    for (std::uint64_t index = 0U; index < count; ++index) {
        auto& chunk =
            *state_->chunks[static_cast<std::size_t>(indices[static_cast<std::size_t>(index)])];
        intent_locks.emplace_back(chunk.mutex_);
    }
    for (std::uint64_t index = 0U; index < count; ++index) {
        auto& chunk =
            *state_->chunks[static_cast<std::size_t>(indices[static_cast<std::size_t>(index)])];
        const bool incompatible =
            chunk.lifecycle_ != LifecycleState::live || chunk.transition_active_ ||
            chunk.write_lease_intent_ ||
            (options.access == AccessMode::read_write && chunk.lease_intents_ != 0U) ||
            chunk.write_pin_ ||
            (options.access == AccessMode::read_write && chunk.read_pins_ != 0U);
        if (incompatible) {
            return make_error(chunk.lifecycle_ == LifecycleState::poisoned ? ErrorCode::poisoned
                                                                           : ErrorCode::conflict,
                              OperationId::acquire, chunk.id_.value());
        }
        if (chunk.access_revision_ == std::numeric_limits<std::uint64_t>::max() ||
            chunk.lease_intents_ == std::numeric_limits<std::uint64_t>::max() ||
            (options.access == AccessMode::read_only &&
             chunk.read_pins_ == std::numeric_limits<std::uint64_t>::max())) {
            return make_error(ErrorCode::arithmetic_overflow, OperationId::acquire,
                              chunk.id_.value());
        }
    }
    for (std::uint64_t index = 0U; index < count; ++index) {
        auto& chunk =
            *state_->chunks[static_cast<std::size_t>(indices[static_cast<std::size_t>(index)])];
        ++chunk.access_revision_;
        ++chunk.lease_intents_;
        if (options.access == AccessMode::read_write) {
            chunk.write_lease_intent_ = true;
        }
    }
    intent_locks.clear();
    // Count valid access attempts, including restoration OOM. Otherwise a thrash/cooldown
    // hold could freeze its own clock by preventing every subsequent successful access.
    record_access(*runtime_, *state_,
                  std::span<const std::uint64_t>{indices.get(), static_cast<std::size_t>(count)});

    for (std::uint64_t index = 0U; index < count; ++index) {
        auto& chunk =
            *state_->chunks[static_cast<std::size_t>(indices[static_cast<std::size_t>(index)])];
        if (chunk.snapshot().authoritative.state != RepresentationState::gpu_raw) {
            const auto migrated = restore_with_policy(
                *runtime_, *state_,
                static_cast<std::size_t>(indices[static_cast<std::size_t>(index)]));
            if (!migrated) {
                for (std::uint64_t cleanup = 0U; cleanup < count; ++cleanup) {
                    auto& cleanup_chunk = *state_->chunks[static_cast<std::size_t>(
                        indices[static_cast<std::size_t>(cleanup)])];
                    const std::scoped_lock cleanup_lock{cleanup_chunk.mutex_};
                    if (cleanup_chunk.lease_intents_ != 0U) {
                        --cleanup_chunk.lease_intents_;
                    }
                    if (options.access == AccessMode::read_write) {
                        cleanup_chunk.write_lease_intent_ = false;
                    }
                }
                return migrated.error();
            }
        }
    }

    std::vector<std::unique_lock<std::mutex>> pin_locks;
    if (count > static_cast<std::uint64_t>(pin_locks.max_size())) {
        for (std::uint64_t cleanup = 0U; cleanup < count; ++cleanup) {
            auto& cleanup_chunk =
                *state_
                     ->chunks[static_cast<std::size_t>(indices[static_cast<std::size_t>(cleanup)])];
            const std::scoped_lock cleanup_lock{cleanup_chunk.mutex_};
            --cleanup_chunk.lease_intents_;
            if (options.access == AccessMode::read_write) {
                cleanup_chunk.write_lease_intent_ = false;
            }
        }
        return make_error(ErrorCode::arithmetic_overflow, OperationId::acquire);
    }
    try {
        pin_locks.reserve(static_cast<std::size_t>(count));
    } catch (const std::bad_alloc&) {
        for (std::uint64_t cleanup = 0U; cleanup < count; ++cleanup) {
            auto& cleanup_chunk =
                *state_
                     ->chunks[static_cast<std::size_t>(indices[static_cast<std::size_t>(cleanup)])];
            const std::scoped_lock cleanup_lock{cleanup_chunk.mutex_};
            --cleanup_chunk.lease_intents_;
            if (options.access == AccessMode::read_write) {
                cleanup_chunk.write_lease_intent_ = false;
            }
        }
        return make_error(ErrorCode::out_of_host_memory, OperationId::acquire);
    }
    for (std::uint64_t index = 0U; index < count; ++index) {
        auto& chunk =
            *state_->chunks[static_cast<std::size_t>(indices[static_cast<std::size_t>(index)])];
        pin_locks.emplace_back(chunk.mutex_);
    }
    for (std::uint64_t index = 0U; index < count; ++index) {
        auto& chunk =
            *state_->chunks[static_cast<std::size_t>(indices[static_cast<std::size_t>(index)])];
        if (chunk.lifecycle_ != LifecycleState::live || chunk.transition_active_ ||
            chunk.authoritative_.state != RepresentationState::gpu_raw ||
            chunk.authoritative_.address.value() == 0U) {
            for (std::uint64_t cleanup = 0U; cleanup < count; ++cleanup) {
                auto& cleanup_chunk = *state_->chunks[static_cast<std::size_t>(
                    indices[static_cast<std::size_t>(cleanup)])];
                --cleanup_chunk.lease_intents_;
                if (options.access == AccessMode::read_write) {
                    cleanup_chunk.write_lease_intent_ = false;
                }
            }
            const auto code = chunk.lifecycle_ == LifecycleState::poisoned
                                  ? ErrorCode::poisoned
                                  : (chunk.lifecycle_ != LifecycleState::live
                                         ? ErrorCode::shutting_down
                                         : ErrorCode::internal_invariant_violation);
            return make_error(code, OperationId::acquire, chunk.id_.value());
        }
        initial_contents[static_cast<std::size_t>(index)] = chunk.authoritative_.content;
    }
    for (std::uint64_t index = 0U; index < count; ++index) {
        auto& chunk =
            *state_->chunks[static_cast<std::size_t>(indices[static_cast<std::size_t>(index)])];
        --chunk.lease_intents_;
        if (options.access == AccessMode::read_write) {
            chunk.write_lease_intent_ = false;
        }
        if (options.access == AccessMode::read_only) {
            ++chunk.read_pins_;
        } else {
            chunk.write_pin_ = true;
        }
    }
    pin_locks.clear();
    return Lease{
        runtime_, state_, range, options.access, std::move(indices), std::move(initial_contents),
        count};
}

Result<PendingLease> Buffer::acquire_async(MemoryRange range, AcquireOptions options) noexcept {
    auto acquired = acquire(range, options);
    if (!acquired) {
        return acquired.error();
    }
    return PendingLease{std::move(acquired).value()};
}

Result<Operation> Buffer::prefetch(MemoryRange range) noexcept {
    auto acquired = acquire(range, AcquireOptions{AccessMode::read_only});
    if (!acquired) {
        return acquired.error();
    }
    const auto closed = acquired.value().close();
    return Operation{closed};
}

BufferStats Buffer::stats() const noexcept {
    if (!state_) {
        return {};
    }
    const std::scoped_lock close_lock{state_->close_mutex};
    BufferStats result{state_->logical_size, static_cast<std::uint64_t>(state_->chunks.size())};
    result.closing = state_->closing;
    for (const auto& chunk : state_->chunks) {
        const auto snapshot = chunk->snapshot();
        result.active_read_leases = saturating_add(result.active_read_leases, snapshot.read_pins);
        result.active_write_leases =
            saturating_add(result.active_write_leases, snapshot.write_pin ? 1U : 0U);
        result.poisoned = result.poisoned || snapshot.lifecycle == LifecycleState::poisoned;
    }
    return result;
}

Result<void> Buffer::close() noexcept {
    if (!state_) {
        return make_error(ErrorCode::stale_handle, OperationId::close_buffer);
    }
    return detail::close_buffer_state(*state_);
}

Lease::Lease(std::shared_ptr<detail::RuntimeState> runtime,
             std::shared_ptr<detail::BufferState> state, MemoryRange range, AccessMode access,
             std::unique_ptr<std::uint64_t[]> chunk_indices,
             std::unique_ptr<ContentTag[]> initial_contents, std::uint64_t chunk_count) noexcept
    : runtime_(std::move(runtime)), state_(std::move(state)), range_(range), access_(access),
      chunk_indices_(std::move(chunk_indices)), initial_contents_(std::move(initial_contents)),
      chunk_count_(chunk_count), active_(true) {}

Lease::Lease(Lease&& other) noexcept
    : runtime_(std::move(other.runtime_)), state_(std::move(other.state_)), range_(other.range_),
      access_(other.access_), chunk_indices_(std::move(other.chunk_indices_)),
      initial_contents_(std::move(other.initial_contents_)), chunk_count_(other.chunk_count_),
      active_(other.active_) {
    other.chunk_count_ = 0U;
    other.active_ = false;
}

Lease::~Lease() noexcept {
    if (active_) {
        static_cast<void>(close());
    }
}

bool Lease::valid() const noexcept { return active_ && state_ != nullptr; }
MemoryRange Lease::range() const noexcept { return range_; }
AccessMode Lease::access_mode() const noexcept { return access_; }

Result<DeviceSpan> Lease::device_span() const noexcept {
    if (!valid()) {
        return make_error(ErrorCode::stale_handle, OperationId::acquire);
    }
    const auto address =
        checked_add(state_->base_address.value(), range_.offset.value(), OperationId::acquire);
    if (!address) {
        return address.error();
    }
    return DeviceSpan{DeviceAddress{address.value()}, range_.length};
}

Result<void> Lease::close() noexcept {
    if (!valid()) {
        return make_error(ErrorCode::stale_handle, OperationId::close_lease);
    }
    Error first_error{};
    bool failed = false;
    const auto remember_error = [&first_error, &failed](Error error) noexcept {
        if (!failed) {
            first_error = error;
        }
        failed = true;
    };
    for (std::uint64_t index = 0U; index < chunk_count_; ++index) {
        auto& chunk = *state_->chunks[static_cast<std::size_t>(
            chunk_indices_[static_cast<std::size_t>(index)])];
        if (access_ == AccessMode::read_only) {
            const std::scoped_lock lock{chunk.mutex_};
            if (chunk.read_pins_ == 0U) {
                remember_error(make_error(ErrorCode::internal_invariant_violation,
                                          OperationId::close_lease, chunk.id_.value()));
                continue;
            }
            --chunk.read_pins_;
            continue;
        }

        ResourceId resource{};
        Representation original{};
        ContentTag updated{};
        {
            const std::scoped_lock lock{chunk.mutex_};
            if (!chunk.write_pin_) {
                remember_error(make_error(ErrorCode::internal_invariant_violation,
                                          OperationId::close_lease, chunk.id_.value()));
                continue;
            }
            if (chunk.authoritative_.content.generation ==
                std::numeric_limits<std::uint64_t>::max()) {
                const auto error = make_error(ErrorCode::arithmetic_overflow,
                                              OperationId::close_lease, chunk.id_.value());
                chunk.lifecycle_ = LifecycleState::poisoned;
                if (!chunk.first_error_.has_value()) {
                    chunk.first_error_ = error;
                }
                chunk.write_pin_ = false;
                remember_error(error);
                continue;
            }
            if (chunk.authoritative_.content !=
                initial_contents_[static_cast<std::size_t>(index)]) {
                chunk.write_pin_ = false;
                continue;
            }
            resource = chunk.authoritative_.resource;
            original = chunk.authoritative_;
            updated = next_content_tag(chunk.authoritative_.content);
        }

        const auto backend_update = runtime_->backend->write_content(resource, updated);
        const auto descriptor = backend_update ? runtime_->backend->allocation(resource)
                                               : Result<BackendAllocation>{backend_update.error()};
        if (descriptor && (descriptor.value().id != original.resource ||
                           descriptor.value().tier != PhysicalTier::gpu ||
                           descriptor.value().kind != ResourceKind::representation ||
                           descriptor.value().charge != original.charge ||
                           descriptor.value().address != original.address)) {
            runtime_->async_errors.push(
                make_error(ErrorCode::backend_contract_violation, OperationId::close_lease));
            std::terminate();
        }
        {
            const std::scoped_lock lock{chunk.mutex_};
            if (!descriptor || descriptor.value().metadata.encoding != Encoding::raw ||
                descriptor.value().metadata.logical_size != chunk.logical_size_ ||
                descriptor.value().metadata.stored_size != chunk.logical_size_) {
                const auto error = descriptor ? make_error(ErrorCode::backend_contract_violation,
                                                           OperationId::close_lease)
                                              : descriptor.error();
                chunk.lifecycle_ = LifecycleState::poisoned;
                if (!chunk.first_error_.has_value()) {
                    chunk.first_error_ = error;
                }
                chunk.write_pin_ = false;
                remember_error(error);
                continue;
            }
            if (!chunk.write_pin_ || chunk.authoritative_.resource != resource) {
                const auto error = make_error(ErrorCode::internal_invariant_violation,
                                              OperationId::close_lease, chunk.id_.value());
                chunk.lifecycle_ = LifecycleState::poisoned;
                if (!chunk.first_error_.has_value()) {
                    chunk.first_error_ = error;
                }
                chunk.write_pin_ = false;
                remember_error(error);
                continue;
            }
            chunk.authoritative_.content = updated;
            chunk.authoritative_.metadata = descriptor.value().metadata;
            chunk.write_pin_ = false;
        }
    }
    active_ = false;
    const auto runtime = runtime_;
    const auto buffer = state_;
    runtime_.reset();
    state_.reset();
    chunk_indices_.reset();
    initial_contents_.reset();
    chunk_count_ = 0U;
    bool closing = false;
    {
        const std::scoped_lock close_lock{buffer->close_mutex};
        closing = buffer->closing;
    }
    if (closing) {
        const auto closed = detail::close_buffer_state(*buffer);
        if (!closed && closed.error().code != ErrorCode::stale_handle &&
            closed.error().code != ErrorCode::busy) {
            static_cast<void>(retain_failed_buffer(*runtime, buffer, closed.error()));
        }
    }
    if (failed) {
        runtime->async_errors.push(first_error);
        return Result<void>{first_error};
    }
    return {};
}

Result<void> Lease::close_and_wait() noexcept { return close(); }

Result<PendingLeaseRelease> Lease::defer() noexcept {
    if (!valid()) {
        return make_error(ErrorCode::stale_handle, OperationId::defer_lease);
    }
    auto runtime = runtime_;
    if (!runtime->backend->capabilities().supports_external_async_completion) {
        return make_error(ErrorCode::unsupported, OperationId::defer_lease);
    }
    detail::DeferredLeaseEntry* entry = nullptr;
    {
        const std::scoped_lock lock{runtime->mutex};
        const std::scoped_lock buffer_lock{state_->close_mutex};
        if (runtime->shutting_down || state_->closing) {
            return make_error(ErrorCode::shutting_down, OperationId::defer_lease);
        }
        if (runtime->operations_in_flight == std::numeric_limits<std::uint64_t>::max()) {
            return make_error(ErrorCode::arithmetic_overflow, OperationId::defer_lease);
        }
        for (std::uint32_t index = 0U; index < runtime->config.maximum_pending_completions;
             ++index) {
            if (!runtime->deferred[index].occupied) {
                entry = &runtime->deferred[index];
                break;
            }
        }
        if (entry == nullptr) {
            return make_error(ErrorCode::busy, OperationId::defer_lease);
        }
        entry->occupied = true;
        entry->processing = true;
        ++runtime->operations_in_flight;
    }
    const RuntimeOperationGuard operation{*runtime};
    const auto created = runtime->backend->create_completion();
    if (!created) {
        check_completion_error(*runtime, created.error());
        const std::scoped_lock lock{runtime->mutex};
        *entry = {};
        saturating_increment(runtime->completion_stats.completion_failures);
        return created.error();
    }
    const auto backend_id = runtime->backend->capabilities().backend_id;
    const CompletionToken token{created.value(), backend_id, runtime->id};
    {
        const std::scoped_lock lock{runtime->mutex};
        if (token.id == CompletionTokenId{} || backend_id == BackendId{} ||
            std::ranges::any_of(runtime->deferred, [token](const auto& candidate) {
                return candidate.token == token;
            })) {
            completion_fatal(*runtime, OperationId::defer_lease);
        }
        entry->token = token;
        entry->buffer = std::move(state_);
        entry->range = range_;
        entry->access = access_;
        entry->chunk_indices = std::move(chunk_indices_);
        entry->initial_contents = std::move(initial_contents_);
        entry->chunk_count = chunk_count_;
        if (entry->access == AccessMode::read_write) {
            // A test/reference write may already have advanced generation synchronously.
            // Establish a new baseline so deferred external writes always refresh integrity
            // at completion rather than taking the immediate-close "already finalized" path.
            const std::scoped_lock buffer_lock{entry->buffer->close_mutex};
            for (std::uint64_t index = 0U; index < entry->chunk_count; ++index) {
                entry->initial_contents[index] =
                    entry->buffer->chunks[static_cast<std::size_t>(entry->chunk_indices[index])]
                        ->snapshot()
                        .authoritative.content;
            }
        }
        entry->processing = false;
        ++runtime->completion_stats.pending_completion_count;
        active_ = false;
        chunk_count_ = 0U;
        runtime_.reset();
    }
    return PendingLeaseRelease{std::move(runtime), token};
}

Result<bool> Runtime::poll_completion(CompletionToken token) noexcept {
    if (!state_) {
        return make_error(ErrorCode::stale_handle, OperationId::query_completion);
    }
    detail::DeferredLeaseEntry* entry = nullptr;
    {
        const std::scoped_lock lock{state_->mutex};
        if (token.runtime != state_->id ||
            token.backend != state_->backend->capabilities().backend_id ||
            token.id == CompletionTokenId{}) {
            return make_error(ErrorCode::invalid_argument, OperationId::query_completion);
        }
        for (auto& candidate : state_->deferred) {
            if (candidate.occupied && candidate.token == token) {
                entry = &candidate;
                break;
            }
        }
        if (entry == nullptr) {
            return make_error(ErrorCode::stale_handle, OperationId::query_completion,
                              token.id.value());
        }
        if (entry->processing || state_->shutdown_in_progress) {
            return make_error(ErrorCode::busy, OperationId::query_completion);
        }
        if (state_->operations_in_flight == std::numeric_limits<std::uint64_t>::max()) {
            return make_error(ErrorCode::arithmetic_overflow, OperationId::query_completion);
        }
        entry->processing = true;
        ++state_->operations_in_flight;
    }
    const RuntimeOperationGuard operation{*state_};
    const auto retain_error = [this, entry](Error error) -> Result<bool> {
        check_completion_error(*state_, error);
        const std::scoped_lock lock{state_->mutex};
        entry->processing = false;
        saturating_increment(state_->completion_stats.completion_failures);
        state_->async_errors.push(error);
        return error;
    };
    if (!entry->completion_proven) {
        const auto queried = state_->backend->query_completion(token.id);
        if (!queried) {
            return retain_error(queried.error());
        }
        if (queried.value() == CompletionState::pending) {
            const std::scoped_lock lock{state_->mutex};
            entry->processing = false;
            saturating_increment(state_->completion_stats.completion_not_ready_count);
            return false;
        }
        if (queried.value() != CompletionState::complete) {
            completion_fatal(*state_, OperationId::query_completion);
        }
        entry->completion_proven = true;
    }
    // Release the proved-complete fence first. A definite cleanup failure retains both
    // the registry entry and the original pins, permitting a bounded retry without reuse.
    const auto released = state_->backend->release_completion(token.id);
    if (!released) {
        return retain_error(released.error());
    }
    Lease completed{state_,
                    entry->buffer,
                    entry->range,
                    entry->access,
                    std::move(entry->chunk_indices),
                    std::move(entry->initial_contents),
                    entry->chunk_count};
    const auto closed = completed.close();
    {
        const std::scoped_lock lock{state_->mutex};
        *entry = {};
        --state_->completion_stats.pending_completion_count;
        saturating_increment(state_->completion_stats.completed_deferred_releases);
        if (!closed) {
            saturating_increment(state_->completion_stats.completion_failures);
        }
    }
    return closed ? Result<bool>{true} : Result<bool>{closed.error()};
}

Result<std::uint32_t> Runtime::poll_ready_completions() noexcept {
    if (!state_) {
        return make_error(ErrorCode::stale_handle, OperationId::query_completion);
    }
    std::array<CompletionToken, max_pending_completions> tokens{};
    std::size_t count = 0U;
    {
        const std::scoped_lock lock{state_->mutex};
        for (const auto& entry : state_->deferred) {
            if (entry.occupied && !entry.processing) {
                tokens[count++] = entry.token;
            }
        }
    }
    std::uint32_t completed = 0U;
    for (std::size_t index = 0U; index < count; ++index) {
        const auto result = poll_completion(tokens[index]);
        if (!result) {
            // Another bounded poller may have consumed the snapshot's token.
            if (result.error().code == ErrorCode::busy ||
                result.error().code == ErrorCode::stale_handle) {
                continue;
            }
            return result.error();
        }
        if (result.value()) {
            ++completed;
        }
    }
    return completed;
}

PendingLease::PendingLease(Lease lease) noexcept : lease_(std::move(lease)), has_lease_(true) {}

PendingLease::PendingLease(PendingLease&& other) noexcept
    : lease_(std::move(other.lease_)), has_lease_(other.has_lease_), consumed_(other.consumed_) {
    other.has_lease_ = false;
    other.consumed_ = true;
}

OperationStatus PendingLease::status() const noexcept {
    return consumed_ ? OperationStatus::cancelled : OperationStatus::succeeded;
}

Result<std::optional<Lease>> PendingLease::poll() noexcept {
    if (consumed_ || !has_lease_) {
        return make_error(ErrorCode::stale_handle, OperationId::acquire);
    }
    consumed_ = true;
    std::optional<Lease> result{std::move(lease_)};
    has_lease_ = false;
    return result;
}

Result<Lease> PendingLease::wait() noexcept {
    if (consumed_ || !has_lease_) {
        return make_error(ErrorCode::stale_handle, OperationId::acquire);
    }
    consumed_ = true;
    has_lease_ = false;
    return std::move(lease_);
}

Result<void> PendingLease::cancel() noexcept {
    if (consumed_) {
        return make_error(ErrorCode::stale_handle, OperationId::acquire);
    }
    consumed_ = true;
    if (has_lease_) {
        has_lease_ = false;
        return lease_.close();
    }
    return {};
}

Operation::Operation(Result<void> result) noexcept {
    if (!result) {
        status_ = OperationStatus::failed;
        error_ = result.error();
        has_error_ = true;
    }
}

Result<void> Operation::wait() const noexcept {
    return has_error_ ? Result<void>{error_} : Result<void>{};
}

Result<void> Operation::cancel() noexcept {
    if (status_ == OperationStatus::succeeded) {
        return make_error(ErrorCode::stale_handle, OperationId::migrate);
    }
    status_ = OperationStatus::cancelled;
    return {};
}

Result<void> Lease::read(ByteOffset relative_offset, std::span<std::byte> output) const noexcept {
    const auto& lease = *this;
    if (!lease.valid() || !lease.runtime_ || !lease.state_) {
        return make_error(ErrorCode::stale_handle, OperationId::acquire);
    }
    const auto relative_end = checked_add(
        relative_offset.value(), static_cast<std::uint64_t>(output.size()), OperationId::acquire);
    if (!relative_end || relative_end.value() > lease.range_.length.value()) {
        return relative_end ? make_error(ErrorCode::invalid_range, OperationId::acquire)
                            : relative_end.error();
    }
    const auto absolute_begin =
        checked_add(lease.range_.offset.value(), relative_offset.value(), OperationId::acquire);
    if (!absolute_begin) {
        return absolute_begin.error();
    }
    const auto absolute_end = checked_add(
        absolute_begin.value(), static_cast<std::uint64_t>(output.size()), OperationId::acquire);
    if (!absolute_end) {
        return absolute_end.error();
    }
    for (std::uint64_t index = 0U; index < lease.chunk_count_; ++index) {
        auto& chunk = *lease.state_->chunks[static_cast<std::size_t>(
            lease.chunk_indices_[static_cast<std::size_t>(index)])];
        const auto chunk_end =
            checked_add(chunk.offset_.value(), chunk.logical_size_.value(), OperationId::acquire);
        if (!chunk_end) {
            return chunk_end.error();
        }
        const auto begin = std::max(absolute_begin.value(), chunk.offset_.value());
        const auto end = std::min(absolute_end.value(), chunk_end.value());
        if (begin >= end) {
            continue;
        }
        std::unique_lock lock{chunk.mutex_};
        const bool pinned =
            lease.access_ == AccessMode::read_only ? chunk.read_pins_ != 0U : chunk.write_pin_;
        if (!pinned || chunk.authoritative_.state != RepresentationState::gpu_raw) {
            return make_error(ErrorCode::stale_handle, OperationId::acquire, chunk.id_.value());
        }
        const auto destination_offset = static_cast<std::size_t>(begin - absolute_begin.value());
        const auto count = static_cast<std::size_t>(end - begin);
        const auto resource_id = chunk.authoritative_.resource;
        lock.unlock();
        const auto read = lease.runtime_->backend->read_bytes(
            resource_id, ByteOffset{begin - chunk.offset_.value()},
            output.subspan(destination_offset, count));
        if (!read) {
            return read.error();
        }
    }
    return {};
}

Result<void> Lease::write(ByteOffset relative_offset, std::span<const std::byte> input) noexcept {
    auto& lease = *this;
    if (!lease.valid() || !lease.runtime_ || !lease.state_) {
        return make_error(ErrorCode::stale_handle, OperationId::close_lease);
    }
    if (lease.access_ != AccessMode::read_write) {
        return make_error(ErrorCode::conflict, OperationId::close_lease);
    }
    const auto relative_end =
        checked_add(relative_offset.value(), static_cast<std::uint64_t>(input.size()),
                    OperationId::close_lease);
    if (!relative_end || relative_end.value() > lease.range_.length.value()) {
        return relative_end ? make_error(ErrorCode::invalid_range, OperationId::close_lease)
                            : relative_end.error();
    }
    const auto absolute_begin =
        checked_add(lease.range_.offset.value(), relative_offset.value(), OperationId::close_lease);
    if (!absolute_begin) {
        return absolute_begin.error();
    }
    const auto absolute_end = checked_add(
        absolute_begin.value(), static_cast<std::uint64_t>(input.size()), OperationId::close_lease);
    if (!absolute_end) {
        return absolute_end.error();
    }
    for (std::uint64_t index = 0U; index < lease.chunk_count_; ++index) {
        auto& chunk = *lease.state_->chunks[static_cast<std::size_t>(
            lease.chunk_indices_[static_cast<std::size_t>(index)])];
        const auto chunk_end = checked_add(chunk.offset_.value(), chunk.logical_size_.value(),
                                           OperationId::close_lease);
        if (!chunk_end) {
            return chunk_end.error();
        }
        const auto begin = std::max(absolute_begin.value(), chunk.offset_.value());
        const auto end = std::min(absolute_end.value(), chunk_end.value());
        if (begin >= end) {
            continue;
        }
        std::unique_lock lock{chunk.mutex_};
        if (!chunk.write_pin_ || chunk.authoritative_.state != RepresentationState::gpu_raw) {
            return make_error(ErrorCode::stale_handle, OperationId::close_lease, chunk.id_.value());
        }
        const auto source_offset = static_cast<std::size_t>(begin - absolute_begin.value());
        const auto count = static_cast<std::size_t>(end - begin);
        const auto original = chunk.authoritative_;
        lock.unlock();
        const auto updated = lease.runtime_->backend->write_bytes(
            original.resource, ByteOffset{begin - chunk.offset_.value()},
            input.subspan(source_offset, count), original.content);
        const auto resource = updated ? lease.runtime_->backend->allocation(original.resource)
                                      : Result<BackendAllocation>{updated.error()};
        lock.lock();
        if (!chunk.write_pin_ || chunk.authoritative_.resource != original.resource ||
            chunk.authoritative_.content != original.content) {
            lease.runtime_->async_errors.push(
                make_error(ErrorCode::internal_invariant_violation, OperationId::close_lease));
            std::terminate();
        }
        if (!updated) {
            chunk.lifecycle_ = LifecycleState::poisoned;
            if (!chunk.first_error_.has_value()) {
                chunk.first_error_ = updated.error();
            }
            return updated.error();
        }
        if (resource && (resource.value().id != original.resource ||
                         resource.value().tier != PhysicalTier::gpu ||
                         resource.value().kind != ResourceKind::representation ||
                         resource.value().address != original.address ||
                         resource.value().charge != original.charge)) {
            lease.runtime_->async_errors.push(
                make_error(ErrorCode::backend_contract_violation, OperationId::close_lease));
            std::terminate();
        }
        if (!resource || resource.value().metadata.encoding != Encoding::raw ||
            resource.value().metadata.stored_size != chunk.logical_size_ ||
            resource.value().metadata.logical_size != chunk.logical_size_) {
            const Error error = resource ? make_error(ErrorCode::backend_contract_violation,
                                                      OperationId::close_lease, chunk.id_.value())
                                         : resource.error();
            chunk.lifecycle_ = LifecycleState::poisoned;
            if (!chunk.first_error_.has_value()) {
                chunk.first_error_ = error;
            }
            return error;
        }
        chunk.authoritative_.content = updated.value();
        chunk.authoritative_.metadata = resource.value().metadata;
    }
    return {};
}

Result<Runtime> Runtime::create_backend(RuntimeConfig config,
                                        std::unique_ptr<StorageBackend> backend) noexcept {
    if (!backend) {
        return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    }
    const auto valid = validate_config(config, backend->capabilities());
    if (!valid) {
        return valid.error();
    }
    if (config.policy.mode != PolicyMode::disabled && !backend->compression_available()) {
        return make_error(ErrorCode::unsupported, OperationId::runtime_create);
    }
    if (config.collect_performance) {
        if (auto* cuda = dynamic_cast<detail::CudaVmmBackend*>(backend.get())) {
            const auto enabled = cuda->enable_performance();
            if (!enabled) {
                return enabled.error();
            }
        }
    }
    const auto identity = next_runtime_identity();
    if (!identity) {
        return identity.error();
    }
    try {
        return Runtime{
            std::make_shared<detail::RuntimeState>(config, std::move(backend), identity.value())};
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::runtime_create);
    }
}

namespace testing {

BufferId RuntimeAccess::buffer_identity(const Buffer& buffer) noexcept {
    return buffer.state_ ? buffer.state_->id : BufferId{};
}

Result<Runtime>
RuntimeAccess::create_with_backend(RuntimeConfig config,
                                   std::unique_ptr<StorageBackend> backend) noexcept {
    return Runtime::create_backend(config, std::move(backend));
}

Result<void> RuntimeAccess::reclaim(Runtime& runtime, ByteSize target, bool admission_pressure,
                                    PolicySelectionObserver* observer) noexcept {
    return runtime.state_
               ? reclaim_request(*runtime.state_, target, admission_pressure, observer)
               : Result<void>{make_error(ErrorCode::invalid_argument, OperationId::migrate)};
}

Result<void> RuntimeAccess::advance_access_revision(Buffer& buffer, ChunkId chunk_id,
                                                    std::uint64_t amount) noexcept {
    const auto chunk_index = chunk_id.value();
    if (!buffer.state_) {
        return make_error(ErrorCode::stale_handle, OperationId::acquire);
    }
    const std::scoped_lock close_lock{buffer.state_->close_mutex};
    if (chunk_index >= static_cast<std::uint64_t>(buffer.state_->chunks.size())) {
        return make_error(ErrorCode::invalid_range, OperationId::acquire, chunk_index);
    }
    auto& chunk = *buffer.state_->chunks[static_cast<std::size_t>(chunk_index)];
    const std::scoped_lock lock{chunk.mutex_};
    if (chunk.lifecycle_ != LifecycleState::live || chunk.transition_active_ ||
        chunk.read_pins_ != 0U || chunk.write_pin_ || chunk.lease_intents_ != 0U) {
        return make_error(ErrorCode::busy, OperationId::acquire, chunk_index);
    }
    const auto next = checked_add(chunk.access_revision_, amount, OperationId::acquire);
    if (!next) {
        return next.error();
    }
    chunk.access_revision_ = next.value();
    return {};
}

Result<policy::Metadata> RuntimeAccess::policy_metadata(const Buffer& buffer,
                                                        std::uint64_t chunk_index) noexcept {
    if (!buffer.runtime_ || !buffer.state_ || !policy_enabled(*buffer.runtime_)) {
        return make_error(ErrorCode::unsupported, OperationId::migrate);
    }
    const std::scoped_lock lock{buffer.runtime_->mutex};
    const std::scoped_lock close_lock{buffer.state_->close_mutex};
    if (chunk_index >= static_cast<std::uint64_t>(buffer.state_->chunks.size())) {
        return make_error(ErrorCode::invalid_range, OperationId::migrate);
    }
    const auto index = static_cast<std::size_t>(chunk_index);
    auto& metadata = buffer.state_->policy_metadata[index];
    const auto snapshot = buffer.state_->chunks[index]->snapshot();
    policy::observe(metadata, snapshot, raw_charge(*buffer.runtime_, snapshot),
                    buffer.runtime_->policy_stats.access_epoch);
    return metadata;
}

void RuntimeAccess::advance_policy_epoch(Runtime& runtime, std::uint64_t epochs) noexcept {
    if (runtime.state_ && policy_enabled(*runtime.state_)) {
        const std::scoped_lock lock{runtime.state_->mutex};
        runtime.state_->policy_stats.access_epoch =
            saturating_add(runtime.state_->policy_stats.access_epoch, epochs);
    }
}

Result<void> RuntimeAccess::advance_policy_cycle(Runtime& runtime, std::uint64_t cycles) noexcept {
    if (!runtime.state_ || !policy_enabled(*runtime.state_)) {
        return make_error(ErrorCode::invalid_argument, OperationId::migrate);
    }
    const std::scoped_lock gate{runtime.state_->allocation_mutex};
    const std::scoped_lock lock{runtime.state_->mutex};
    const auto advanced = checked_add(runtime.state_->policy_cycle, cycles, OperationId::migrate);
    if (!advanced) {
        return advanced.error();
    }
    runtime.state_->policy_cycle = advanced.value();
    return {};
}

MockBackend& RuntimeAccess::backend(Runtime& runtime) noexcept {
    if (!runtime.state_ || runtime.state_->mock_controls == nullptr) {
        std::terminate();
    }
    return *runtime.state_->mock_controls;
}

void RuntimeAccess::inject_materialize_failure(Runtime& runtime,
                                               std::uint64_t invocation) noexcept {
    if (runtime.state_) {
        inject_materialize_failure(runtime.state_->ledger, invocation);
    }
}

void RuntimeAccess::inject_failure(Runtime& runtime, FaultPoint point,
                                   std::uint64_t invocation) noexcept {
    if (runtime.state_) {
        backend(runtime).inject_failure(point, invocation);
    }
}

Result<void> RuntimeAccess::migrate(Buffer& buffer, std::uint64_t chunk_index,
                                    RepresentationState destination, TransactionObserver* observer,
                                    const MigrationConstraints* constraints) noexcept {
    if (!buffer.state_ || chunk_index >= static_cast<std::uint64_t>(buffer.state_->chunks.size())) {
        return make_error(ErrorCode::invalid_argument, OperationId::migrate, chunk_index);
    }
    return buffer.runtime_->coordinator.migrate(
        *buffer.state_->chunks[static_cast<std::size_t>(chunk_index)], destination, false, observer,
        constraints);
}

Result<ChunkSnapshot> RuntimeAccess::chunk_snapshot(const Buffer& buffer,
                                                    std::uint64_t chunk_index) noexcept {
    if (!buffer.state_ || chunk_index >= static_cast<std::uint64_t>(buffer.state_->chunks.size())) {
        return make_error(ErrorCode::invalid_argument, OperationId::migrate, chunk_index);
    }
    return buffer.state_->chunks[static_cast<std::size_t>(chunk_index)]->snapshot();
}

Result<CleanupResource> RuntimeAccess::cleanup_resource(const Buffer& buffer,
                                                        std::uint64_t chunk_index,
                                                        std::uint32_t index) noexcept {
    if (!buffer.state_ || chunk_index >= static_cast<std::uint64_t>(buffer.state_->chunks.size())) {
        return make_error(ErrorCode::invalid_argument, OperationId::release, chunk_index);
    }
    return cleanup_resource(*buffer.state_->chunks[static_cast<std::size_t>(chunk_index)], index);
}

bool RuntimeAccess::accounting_conserved(const Runtime& runtime, PhysicalTier tier) noexcept {
    if (!runtime.state_) {
        return false;
    }
    const auto expected =
        checked_add(runtime.state_->backend->owned_charge(tier).value(),
                    runtime.state_->ledger.unmaterialized_reservations(tier).value(),
                    OperationId::budget_transfer);
    return expected && runtime.state_->ledger.charged(tier).value() == expected.value();
}

Result<void> RuntimeAccess::quarantine(Buffer& buffer) noexcept {
    if (!buffer.runtime_ || !buffer.state_) {
        return make_error(ErrorCode::stale_handle, OperationId::close_buffer);
    }
    return retain_failed_buffer(*buffer.runtime_, buffer.state_,
                                make_error(ErrorCode::backend_failure, OperationId::close_buffer,
                                           buffer.state_->id.value()));
}

std::uint64_t RuntimeAccess::owned_resource_count(const Runtime& runtime) noexcept {
    return runtime.state_ ? runtime.state_->backend->owned_resource_count() : 0U;
}

std::weak_ptr<const void> RuntimeAccess::runtime_lifetime(const Runtime& runtime) noexcept {
    return std::weak_ptr<const void>{runtime.state_};
}

std::weak_ptr<const void> RuntimeAccess::buffer_lifetime(const Buffer& buffer) noexcept {
    return std::weak_ptr<const void>{buffer.state_};
}

Result<void> RuntimeAccess::read_bytes(const Lease& lease, ByteOffset offset,
                                       std::span<std::byte> output) noexcept {
    return lease.read(offset, output);
}
Result<void> RuntimeAccess::write_bytes(Lease& lease, ByteOffset offset,
                                        std::span<const std::byte> input) noexcept {
    return lease.write(offset, input);
}

Result<void> RuntimeAccess::corrupt(Buffer& buffer, std::uint64_t chunk_index,
                                    ResourceCorruption corruption) noexcept {
    if (!buffer.state_ || chunk_index >= static_cast<std::uint64_t>(buffer.state_->chunks.size())) {
        return make_error(ErrorCode::invalid_argument, OperationId::verify, chunk_index);
    }
    auto& chunk = *buffer.state_->chunks[static_cast<std::size_t>(chunk_index)];
    const std::scoped_lock lock{chunk.mutex_};
    if (chunk.transition_active_ || chunk.read_pins_ != 0U || chunk.write_pin_) {
        return make_error(ErrorCode::busy, OperationId::verify, chunk.id_.value());
    }
    if (buffer.runtime_->mock_controls == nullptr) {
        return make_error(ErrorCode::unsupported, OperationId::verify);
    }
    return buffer.runtime_->mock_controls->corrupt(chunk.authoritative_.resource, corruption);
}

Result<RepresentationMetadata>
RuntimeAccess::representation_metadata(const Buffer& buffer, std::uint64_t chunk_index) noexcept {
    const auto snapshot = chunk_snapshot(buffer, chunk_index);
    return snapshot ? Result<RepresentationMetadata>{snapshot.value().authoritative.metadata}
                    : Result<RepresentationMetadata>{snapshot.error()};
}

bool RuntimeAccess::cpu_lz4_available(const Runtime& runtime) noexcept {
    return runtime.state_ && runtime.state_->backend->compression_available();
}

Result<ByteSize> RuntimeAccess::codec_maximum_compressed_size(ByteSize input) noexcept {
    return detail::cpu_compression_codec().maximum_compressed_size(input);
}

ByteSize RuntimeAccess::codec_maximum_input_size() noexcept {
    return detail::cpu_compression_codec().maximum_input_size();
}

Result<void> RuntimeAccess::codec_decompress(std::span<const std::byte> input,
                                             std::span<std::byte> output) noexcept {
    const auto decoded = detail::cpu_compression_codec().decompress(
        input, output, ByteSize{static_cast<std::uint64_t>(output.size())});
    if (!decoded) {
        return decoded.error();
    }
    return decoded.value().value() == output.size()
               ? Result<void>{}
               : Result<void>{make_error(ErrorCode::integrity_failure, OperationId::decompress)};
}

Result<void> RuntimeAccess::codec_round_trip(std::span<const std::byte> input,
                                             std::span<std::byte> output) noexcept {
    if (output.size() != input.size()) {
        return make_error(ErrorCode::invalid_range, OperationId::decompress);
    }
    try {
        auto& codec = detail::cpu_compression_codec();
        const auto bound = codec.maximum_compressed_size(ByteSize{input.size()});
        if (!bound) {
            return bound.error();
        }
        std::vector<std::byte> encoded(static_cast<std::size_t>(bound.value().value()));
        const auto stored = codec.compress(input, encoded);
        if (!stored) {
            return stored.error();
        }
        const auto restored = codec.decompress(
            std::span<const std::byte>{encoded.data(),
                                       static_cast<std::size_t>(stored.value().value())},
            output, ByteSize{input.size()});
        if (!restored || restored.value().value() != input.size()) {
            return restored ? make_error(ErrorCode::integrity_failure, OperationId::decompress)
                            : restored.error();
        }
        return {};
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::compress);
    }
}

} // namespace testing

} // namespace vramz
