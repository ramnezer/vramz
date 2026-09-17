#include "../test_support.hpp"
#include "vramz/detail/simulation.hpp"

#include <array>
#include <limits>
#include <optional>

using namespace vramz;

namespace {

[[nodiscard]] ByteSize measured_charge(test::Runner& runner, simulation::Dataset dataset) {
    auto config = test::test_runtime_config();
    config.budgets.gpu = TierBudget{ByteSize{131072U}, ByteSize{98304U}, ByteSize{12288U}};
    config.preferred_chunk_size = ByteSize{4096U};
    auto runtime = std::move(Runtime::create(config)).value();
    auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
    std::array<std::byte, 4096U> bytes{};
    simulation::fill(dataset, 0U, 0U, bytes);
    auto lease = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}},
                                AcquireOptions{AccessMode::read_write});
    VRAMZ_CHECK(runner, lease && testing::write_bytes(lease.value(), ByteOffset{}, bytes));
    VRAMZ_CHECK(runner, lease.value().close());
    VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
    const auto charge = testing::chunk_snapshot(buffer, 0U).value().authoritative.charge;
    auto restored = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
    std::array<std::byte, 4096U> decoded{};
    VRAMZ_CHECK(runner, restored && testing::read_bytes(restored.value(), ByteOffset{}, decoded));
    VRAMZ_CHECK(runner, decoded == bytes && restored.value().close());
    VRAMZ_CHECK(runner, buffer.close());
    VRAMZ_CHECK(runner, runtime.shutdown());
    return charge;
}

// Independent ledger/decision reference model. Charges come from real verified M2 probes above.
// It allocates no payload and makes no byte-integrity or backend-corroboration claim by itself.
class ScaledModel final {
  public:
    ScaledModel(test::Runner& runner, std::uint64_t scale, ByteSize measured)
        : runner_(runner), scale_(scale),
          ledger_(MemoryBudgets{TierBudget{scaled(131072U), scaled(98304U), scaled(12288U)},
                                TierBudget{scaled(131072U), scaled(98304U), scaled(12288U)}}),
          compressed_(scaled(measured.value())) {
        config_.tuning.minimum_savings_bytes = scaled(64U);
    }

    [[nodiscard]] std::uint32_t populate(std::uint32_t count) {
        for (std::uint32_t index = 0U; index < count; ++index) {
            std::uint32_t attempts = 0U;
            auto reserved = reserve_raw();
            while (!reserved && attempts < 32U) {
                std::optional<std::uint32_t> victim{};
                policy::Proposal best{};
                for (std::uint32_t item = 0U; item < index; ++item) {
                    const auto proposal =
                        policy::propose(metadata_[item], chunks_[item], BufferId{1U},
                                        Pressure::critical, 100U + index, config_);
                    if (policy::preferred(proposal, best, config_.tuning.strategy)) {
                        best = proposal;
                        victim = item;
                    }
                }
                if (!victim) {
                    break;
                }
                ++attempts;
                compress(*victim, 100U + index);
                reserved = reserve_raw();
            }
            if (!reserved) {
                return index;
            }
            add_owned(scaled(4096U));
            VRAMZ_CHECK(runner_, ledger_.materialize(raw_token_, scaled(4096U)));
            VRAMZ_CHECK(runner_, ledger_.release_reservation(raw_token_));
            VRAMZ_CHECK(runner_, ledger_.commit_initial(PhysicalTier::gpu, scaled(4096U)));
            chunks_[index] = ChunkSnapshot{};
            chunks_[index].id = ChunkId{index};
            chunks_[index].lifecycle = LifecycleState::live;
            chunks_[index].authoritative.state = RepresentationState::gpu_raw;
            chunks_[index].authoritative.charge = scaled(4096U);
            chunks_[index].authoritative.content = ContentTag{1U, index};
            policy::observe(metadata_[index], chunks_[index], scaled(4096U), 1U);
            check();
        }
        return count;
    }

    [[nodiscard]] ByteSize committed() const noexcept {
        return ledger_.usage(PhysicalTier::gpu).committed;
    }
    [[nodiscard]] std::uint64_t transitions() const noexcept { return transitions_; }

    void close() {
        for (auto& chunk : chunks_) {
            if (chunk.lifecycle != LifecycleState::live) {
                continue;
            }
            const auto charge = chunk.authoritative.charge;
            VRAMZ_CHECK(runner_, ledger_.retire_committed(PhysicalTier::gpu, charge));
            check();
            remove_owned(charge);
            VRAMZ_CHECK(runner_, ledger_.release_cleanup_debt(PhysicalTier::gpu, charge));
            chunk.lifecycle = LifecycleState::dead;
            check();
        }
        VRAMZ_CHECK(runner_, owned_ == 0U && ledger_.charged(PhysicalTier::gpu) == ByteSize{});
    }

  private:
    void add_owned(ByteSize charge) {
        const auto added = checked_add(owned_, charge.value(), OperationId::budget_transfer);
        VRAMZ_CHECK(runner_, added);
        owned_ = added.value();
    }

    void remove_owned(ByteSize charge) {
        const auto removed = checked_sub(owned_, charge.value(), OperationId::budget_transfer);
        VRAMZ_CHECK(runner_, removed);
        owned_ = removed.value();
    }

    [[nodiscard]] ByteSize scaled(std::uint64_t value) const {
        const auto result = checked_mul(value, scale_, OperationId::budget_transfer);
        VRAMZ_CHECK(runner_, result);
        return ByteSize{result.value()};
    }

    [[nodiscard]] Result<void> reserve_raw() {
        ReservationRequest request{PhysicalTier::gpu, ChargeBucket::staging, scaled(4096U),
                                   AdmissionKind::normal};
        const auto result =
            ledger_.reserve(TransactionId{1U}, std::span{&request, 1U}, std::span{&raw_token_, 1U});
        check();
        return result;
    }

    void compress(std::uint32_t index, std::uint64_t now) {
        const auto raw = chunks_[index].authoritative.charge;
        const auto limit = policy::compression_limit(raw, config_.tuning);
        VRAMZ_CHECK(runner_, limit);
        if (!limit.has_value()) {
            return;
        }
        VRAMZ_CHECK(runner_, ledger_.validate_migration_projection(PhysicalTier::gpu, *limit,
                                                                   PhysicalTier::gpu, raw));
        std::array<ReservationRequest, 2U> requests{
            ReservationRequest{PhysicalTier::gpu, ChargeBucket::staging, scaled(4160U),
                               AdmissionKind::migration_temporary},
            ReservationRequest{PhysicalTier::gpu, ChargeBucket::workspace, scaled(4160U),
                               AdmissionKind::migration_temporary}};
        std::array<ReservationToken, 2U> tokens{};
        VRAMZ_CHECK(runner_, ledger_.reserve(TransactionId{2U}, requests, tokens));
        check();
        for (std::size_t item = 0U; item < requests.size(); ++item) {
            add_owned(requests[item].amount);
            VRAMZ_CHECK(runner_, ledger_.materialize(tokens[item], requests[item].amount));
            VRAMZ_CHECK(runner_, ledger_.release_reservation(tokens[item]));
            check();
        }
        const auto saving =
            checked_sub(scaled(4160U).value(), compressed_.value(), OperationId::budget_transfer);
        VRAMZ_CHECK(runner_, saving);
        remove_owned(ByteSize{saving.value()});
        VRAMZ_CHECK(runner_, ledger_.shrink_materialized(PhysicalTier::gpu, ChargeBucket::staging,
                                                         scaled(4160U), compressed_));
        check();
        remove_owned(scaled(4160U));
        VRAMZ_CHECK(runner_, ledger_.release_materialized(PhysicalTier::gpu,
                                                          ChargeBucket::workspace, scaled(4160U)));
        check();
        if (compressed_ <= *limit) {
            VRAMZ_CHECK(runner_, ledger_.commit_destination_and_retire_source(
                                     PhysicalTier::gpu, compressed_, PhysicalTier::gpu, raw));
            check();
            remove_owned(raw);
            VRAMZ_CHECK(runner_, ledger_.release_cleanup_debt(PhysicalTier::gpu, raw));
            chunks_[index].authoritative.state = RepresentationState::gpu_compressed;
            chunks_[index].authoritative.charge = compressed_;
            ++transitions_;
        } else {
            remove_owned(compressed_);
            VRAMZ_CHECK(runner_, ledger_.release_materialized(PhysicalTier::gpu,
                                                              ChargeBucket::staging, compressed_));
        }
        policy::compression_result(metadata_[index], compressed_, now, config_.tuning);
        check();
    }

    void check() {
        const auto expected =
            checked_add(owned_, ledger_.unmaterialized_reservations(PhysicalTier::gpu).value(),
                        OperationId::budget_transfer);
        VRAMZ_CHECK(runner_,
                    expected && ledger_.charged(PhysicalTier::gpu).value() == expected.value());
        VRAMZ_CHECK(runner_, ledger_.charged(PhysicalTier::host) == ByteSize{});
        VRAMZ_CHECK(runner_, ledger_.usage(PhysicalTier::gpu).peak_charged <= scaled(131072U));
    }

    test::Runner& runner_;
    std::uint64_t scale_{};
    BudgetLedger ledger_;
    ByteSize compressed_{};
    PolicyConfig config_{};
    std::array<ChunkSnapshot, 96U> chunks_{};
    std::array<policy::Metadata, 96U> metadata_{};
    ReservationToken raw_token_{};
    std::uint64_t owned_{};
    std::uint64_t transitions_{};
};

void healthy(test::Runner& runner, const Result<simulation::ScenarioResult>& result,
             const simulation::Scenario& scenario) {
    if (!result || !result.value().integrity_ok || !result.value().accounting_ok ||
        !result.value().cleanup_ok) {
        std::cerr << "policy seed=" << scenario.seed
                  << " trace=" << simulation::name(scenario.trace)
                  << " dataset=" << simulation::name(scenario.dataset)
                  << " target=" << scenario.target_percent << '\n';
    }
    VRAMZ_CHECK(runner, result && result.value().integrity_ok && result.value().accounting_ok &&
                            result.value().cleanup_ok);
}

class PhaseCount final : public TransactionObserver {
  public:
    void on_phase(TransactionPhase, const ChunkSnapshot&) noexcept override { ++count; }
    std::uint32_t count{};
};

class CrossPhaseTrace final : public testing::PolicySelectionObserver {
  public:
    CrossPhaseTrace(Runtime& runtime, Buffer& buffer, bool invalidate) noexcept
        : runtime_(runtime), buffer_(buffer), invalidate_(invalidate) {}

    void on_selected(const ChunkSnapshot& snapshot,
                     const policy::Proposal& proposal) noexcept override {
        if (count == actions.size()) {
            valid = false;
            return;
        }
        actions[count] = proposal.action;
        chunks[count] = snapshot.id;
        ++count;
        valid = valid && testing::accounting_conserved(runtime_, PhysicalTier::gpu) &&
                testing::accounting_conserved(runtime_, PhysicalTier::host);
        if (!invalidate_ || read_completed || snapshot.id != ChunkId{} ||
            proposal.action != PolicyAction::compress_gpu) {
            return;
        }
        auto lease = buffer_.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
        if (!lease) {
            valid = false;
            return;
        }
        std::array<std::byte, 4096U> bytes{};
        valid = valid && testing::read_bytes(lease.value(), ByteOffset{}, bytes) &&
                bytes == std::array<std::byte, 4096U>{};
        const auto closed = lease.value().close();
        const auto after = testing::chunk_snapshot(buffer_, 0U);
        const auto metadata = testing::RuntimeAccess::policy_metadata(buffer_, 0U);
        valid = valid && closed && after && metadata;
        if (after && metadata) {
            valid = valid && proposal.temperature == Temperature::cold &&
                    after.value().access_revision == snapshot.access_revision + 1U &&
                    after.value().authoritative.resource == snapshot.authoritative.resource &&
                    after.value().authoritative.content == snapshot.authoritative.content &&
                    after.value().read_pins == 0U && after.value().lease_intents == 0U &&
                    policy::temperature(metadata.value(), runtime_.stats().policy.access_epoch,
                                        PolicyTuning{}) == Temperature::hot;
        }
        valid = valid && testing::accounting_conserved(runtime_, PhysicalTier::gpu) &&
                testing::accounting_conserved(runtime_, PhysicalTier::host);
        read_completed = true;
    }

    std::array<PolicyAction, 8U> actions{};
    std::array<ChunkId, 8U> chunks{};
    std::size_t count{};
    bool valid{true};
    bool read_completed{};

  private:
    Runtime& runtime_;
    Buffer& buffer_;
    bool invalidate_{};
};

void cycle_conserved(test::Runner& runner, Runtime& runtime, Buffer& buffer, std::uint64_t chunks) {
    VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == chunks);
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu) &&
                            testing::accounting_conserved(runtime, PhysicalTier::host));
    for (const auto usage : {runtime.stats().gpu, runtime.stats().host}) {
        VRAMZ_CHECK(runner, usage.cleanup_debt == ByteSize{} && usage.reserved == ByteSize{} &&
                                usage.staging == ByteSize{} && usage.workspace == ByteSize{});
    }
    for (std::uint64_t index = 0U; index < chunks; ++index) {
        const auto snapshot = testing::chunk_snapshot(buffer, index).value();
        VRAMZ_CHECK(runner, snapshot.lifecycle == LifecycleState::live &&
                                !snapshot.transition_active &&
                                snapshot.cleanup_resource_count == 0U && snapshot.read_pins == 0U &&
                                snapshot.lease_intents == 0U && !snapshot.write_pin &&
                                snapshot.authoritative.resource != ResourceId{} &&
                                snapshot.authoritative.charge != ByteSize{});
    }
}

enum class CrossPhaseSequence : std::uint8_t { stale_only, stale_with_other, rejected_compression };

void cross_phase_sequence(test::Runner& runner, CrossPhaseSequence sequence,
                          std::uint64_t generation) {
    auto config = test::test_runtime_config();
    config.budgets =
        MemoryBudgets{TierBudget{ByteSize{32768U}, ByteSize{20480U}, ByteSize{12288U}},
                      TierBudget{ByteSize{32768U}, ByteSize{20480U}, ByteSize{12288U}}};
    config.preferred_chunk_size = ByteSize{4096U};
    config.policy.mode = PolicyMode::gpu_resident_with_host_fallback;
    config.policy.tuning.maximum_candidates = 32U;
    config.policy.tuning.maximum_transitions = 8U;
    const ByteSize logical{sequence == CrossPhaseSequence::stale_with_other ? 8192U : 4096U};
    const std::uint64_t chunks = sequence == CrossPhaseSequence::stale_with_other ? 2U : 1U;
    auto runtime = std::move(Runtime::create(config)).value();
    auto buffer = std::move(runtime.allocate(logical)).value();
    std::array<std::byte, 4096U> random{};
    simulation::fill(simulation::Dataset::random, 0U, generation, random);
    if (sequence != CrossPhaseSequence::stale_only) {
        auto lease = buffer.acquire(MemoryRange{ByteOffset{}, logical},
                                    AcquireOptions{AccessMode::read_write});
        const ByteOffset random_offset{sequence == CrossPhaseSequence::stale_with_other ? 4096U
                                                                                        : 0U};
        VRAMZ_CHECK(runner, lease && testing::write_bytes(lease.value(), random_offset, random));
        VRAMZ_CHECK(runner, lease.value().close());
    }
    cycle_conserved(runner, runtime, buffer, chunks);
    testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
    const auto original = testing::chunk_snapshot(buffer, 0U).value();
    const auto original_other = sequence == CrossPhaseSequence::stale_with_other
                                    ? testing::chunk_snapshot(buffer, 1U).value()
                                    : original;
    const auto before = runtime.stats().policy;
    CrossPhaseTrace trace{runtime, buffer, sequence != CrossPhaseSequence::rejected_compression};
    const ByteSize target{sequence == CrossPhaseSequence::stale_with_other ? 4096U : 2048U};
    const auto result = testing::RuntimeAccess::reclaim(runtime, target, true, &trace);
    const auto after = runtime.stats().policy;
    const auto current = testing::chunk_snapshot(buffer, 0U).value();
    const auto metadata = testing::RuntimeAccess::policy_metadata(buffer, 0U).value();
    VRAMZ_CHECK(runner, trace.valid && trace.actions[0U] == PolicyAction::compress_gpu &&
                            trace.chunks[0U] == ChunkId{} &&
                            trace.count <= config.policy.tuning.maximum_transitions &&
                            after.policy_cycles == before.policy_cycles + 1U &&
                            after.candidates_inspected - before.candidates_inspected <=
                                config.policy.tuning.maximum_candidates);
    VRAMZ_CHECK(runner, after.compression_failures == 0U && !metadata.last_attempt_failed &&
                            current.authoritative.content == original.authoritative.content);
    cycle_conserved(runner, runtime, buffer, chunks);
    if (sequence != CrossPhaseSequence::rejected_compression) {
        VRAMZ_CHECK(runner, trace.read_completed && after.stale_proposal_rejections == 1U &&
                                current.access_revision == original.access_revision + 1U &&
                                current.authoritative.resource == original.authoritative.resource &&
                                current.authoritative.state == RepresentationState::gpu_raw &&
                                metadata.stale_rejection_cycle == after.policy_cycles &&
                                metadata.last_fallback_cycle != after.policy_cycles &&
                                metadata.compression_attempts == 0U);
    }
    if (sequence == CrossPhaseSequence::stale_only) {
        VRAMZ_CHECK(runner, !result && result.error().code == ErrorCode::out_of_gpu_memory &&
                                trace.count == 1U && after.host_fallback_count == 0U &&
                                after.policy_transitions == 0U && after.compression_attempts == 0U);
        // A new cycle re-evaluates current metadata; the stale marker is not a blacklist.
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        CrossPhaseTrace fresh{runtime, buffer, false};
        VRAMZ_CHECK(runner, testing::RuntimeAccess::reclaim(runtime, target, true, &fresh));
        const auto next = runtime.stats().policy;
        const auto reevaluated = testing::RuntimeAccess::policy_metadata(buffer, 0U).value();
        VRAMZ_CHECK(runner, fresh.valid && fresh.count == 1U &&
                                fresh.actions[0U] == PolicyAction::compress_gpu &&
                                next.policy_cycles == after.policy_cycles + 1U &&
                                next.stale_proposal_rejections == 1U &&
                                next.host_fallback_count == 0U &&
                                next.compression_successes == 1U &&
                                next.candidates_inspected - after.candidates_inspected <=
                                    config.policy.tuning.maximum_candidates &&
                                reevaluated.stale_rejection_cycle == after.policy_cycles &&
                                reevaluated.last_attempt_cycle == next.policy_cycles &&
                                testing::chunk_snapshot(buffer, 0U).value().authoritative.state ==
                                    RepresentationState::gpu_compressed);
        cycle_conserved(runner, runtime, buffer, chunks);
    } else if (sequence == CrossPhaseSequence::stale_with_other) {
        const auto other = testing::chunk_snapshot(buffer, 1U).value();
        const auto other_metadata = testing::RuntimeAccess::policy_metadata(buffer, 1U).value();
        VRAMZ_CHECK(runner,
                    result && trace.count == 3U &&
                        trace.actions[1U] == PolicyAction::compress_gpu &&
                        trace.actions[2U] == PolicyAction::host_fallback &&
                        trace.chunks[1U] == ChunkId{1U} && trace.chunks[2U] == ChunkId{1U} &&
                        other.authoritative.state == RepresentationState::host_raw &&
                        other.authoritative.content == original_other.authoritative.content &&
                        other_metadata.stale_rejection_cycle == 0U &&
                        other_metadata.last_fallback_cycle == after.policy_cycles &&
                        after.compression_rejected == 1U && after.compression_attempts == 1U &&
                        after.host_fallback_count == 1U);
    } else {
        VRAMZ_CHECK(runner,
                    result && trace.count == 2U && !trace.read_completed &&
                        trace.actions[1U] == PolicyAction::host_fallback &&
                        trace.chunks[1U] == ChunkId{} && metadata.stale_rejection_cycle == 0U &&
                        metadata.last_attempt_cycle == after.policy_cycles &&
                        metadata.last_fallback_cycle == after.policy_cycles &&
                        current.authoritative.state == RepresentationState::host_raw &&
                        after.stale_proposal_rejections == 0U && after.compression_rejected == 1U &&
                        after.compression_attempts == 1U && after.host_fallback_count == 1U);
    }
    // Read the actual bytes back through restoration without introducing another reclaim cycle.
    auto restored = buffer.acquire(MemoryRange{ByteOffset{}, logical});
    VRAMZ_CHECK(runner, restored);
    for (std::uint64_t index = 0U; index < chunks; ++index) {
        std::array<std::byte, 4096U> actual{};
        VRAMZ_CHECK(runner,
                    testing::read_bytes(restored.value(), ByteOffset{index * 4096U}, actual));
        VRAMZ_CHECK(runner,
                    actual == ((sequence == CrossPhaseSequence::rejected_compression || index == 1U)
                                   ? random
                                   : std::array<std::byte, 4096U>{}));
    }
    VRAMZ_CHECK(runner, restored.value().close());
    cycle_conserved(runner, runtime, buffer, chunks);
    VRAMZ_CHECK(runner, buffer.close());
    VRAMZ_CHECK(runner, runtime.shutdown());
    VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U &&
                            testing::accounting_conserved(runtime, PhysicalTier::gpu) &&
                            testing::accounting_conserved(runtime, PhysicalTier::host));
    for (const auto usage : {runtime.stats().gpu, runtime.stats().host}) {
        VRAMZ_CHECK(runner, usage.committed == ByteSize{} && usage.cleanup_debt == ByteSize{} &&
                                usage.reserved == ByteSize{} && usage.staging == ByteSize{} &&
                                usage.workspace == ByteSize{});
    }
}

void access_sequence(test::Runner& runner, std::uint32_t sequence) {
    auto config = test::test_runtime_config();
    config.budgets.gpu = TierBudget{ByteSize{131072U}, ByteSize{98304U}, ByteSize{12288U}};
    config.preferred_chunk_size = ByteSize{4096U};
    config.policy.mode = PolicyMode::gpu_resident_with_host_fallback;
    auto runtime = std::move(Runtime::create(config)).value();
    auto buffer = std::move(runtime.allocate(ByteSize{8192U})).value();
    testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
    const auto selected = testing::chunk_snapshot(buffer, 0U).value();
    const auto metadata = testing::RuntimeAccess::policy_metadata(buffer, 0U).value();
    const auto proposal =
        policy::propose(metadata, selected, BufferId{1U}, Pressure::soft,
                        runtime.stats().policy.access_epoch, config.policy, sequence == 3U);
    VRAMZ_CHECK(runner, proposal.temperature == Temperature::cold &&
                            proposal.action == (sequence == 3U ? PolicyAction::host_fallback
                                                               : PolicyAction::compress_gpu));
    MigrationConstraints constraints{selected.authoritative.resource,
                                     selected.authoritative.content,
                                     {},
                                     selected.access_revision};
    const ByteOffset accessed_offset{sequence == 1U ? 4096U : 0U};
    std::array<std::byte, 4096U> bytes{};
    auto lease = buffer.acquire(
        MemoryRange{accessed_offset, ByteSize{4096U}},
        AcquireOptions{sequence == 2U ? AccessMode::read_write : AccessMode::read_only});
    VRAMZ_CHECK(runner, lease);
    if (sequence == 2U) {
        bytes.fill(std::byte{0x7A});
        VRAMZ_CHECK(runner, testing::write_bytes(lease.value(), ByteOffset{}, bytes));
    } else {
        VRAMZ_CHECK(runner, testing::read_bytes(lease.value(), ByteOffset{}, bytes));
        VRAMZ_CHECK(runner, (bytes == std::array<std::byte, 4096U>{}));
    }
    VRAMZ_CHECK(runner, lease.value().close());
    const auto after_access = testing::chunk_snapshot(buffer, 0U).value();
    VRAMZ_CHECK(runner, after_access.access_revision ==
                            selected.access_revision + (sequence == 1U ? 0U : 1U));
    const auto destination =
        sequence == 3U ? RepresentationState::host_raw : RepresentationState::gpu_compressed;
    const auto owned = testing::owned_resource_count(runtime);
    const auto charged = runtime.stats();
    const auto unchanged = [&]() {
        const auto current = testing::chunk_snapshot(buffer, 0U).value();
        VRAMZ_CHECK(runner,
                    current.authoritative.resource == after_access.authoritative.resource &&
                        current.authoritative.content == after_access.authoritative.content &&
                        current.cleanup_resource_count == 0U && !current.transition_active);
        VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == owned);
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu) &&
                                testing::accounting_conserved(runtime, PhysicalTier::host));
        const auto stats = runtime.stats();
        VRAMZ_CHECK(runner, stats.gpu.cleanup_debt == ByteSize{} &&
                                stats.host.cleanup_debt == ByteSize{} &&
                                stats.gpu.peak_charged == charged.gpu.peak_charged &&
                                stats.host.peak_charged == charged.host.peak_charged);
    };
    PhaseCount phases;
    const auto result =
        testing::RuntimeAccess::migrate(buffer, 0U, destination, &phases, &constraints);
    const auto after = testing::chunk_snapshot(buffer, 0U).value();
    if (sequence == 1U) {
        VRAMZ_CHECK(runner,
                    result && phases.count > 0U && after.authoritative.state == destination);
    } else {
        VRAMZ_CHECK(runner,
                    !result && result.error().code == ErrorCode::conflict && phases.count == 0U);
        unchanged();
        VRAMZ_CHECK(runner, after.authoritative.resource == after_access.authoritative.resource &&
                                after.authoritative.content == after_access.authoritative.content &&
                                after.authoritative.state == RepresentationState::gpu_raw &&
                                !after.transition_active &&
                                after.transition_epoch == selected.transition_epoch);
        VRAMZ_CHECK(runner, runtime.stats().gpu.committed == charged.gpu.committed &&
                                runtime.stats().gpu.peak_charged == charged.gpu.peak_charged &&
                                runtime.stats().host.peak_charged == charged.host.peak_charged);
        if (sequence == 2U) {
            VRAMZ_CHECK(runner,
                        after_access.authoritative.content != selected.authoritative.content);
            auto read_only_token = constraints;
            read_only_token.expected_content = after_access.authoritative.content;
            const auto stale_access =
                testing::RuntimeAccess::migrate(buffer, 0U, destination, &phases, &read_only_token);
            VRAMZ_CHECK(runner, !stale_access && stale_access.error().code == ErrorCode::conflict);
            unchanged();
            auto content_only_token = constraints;
            content_only_token.expected_access_revision = after_access.access_revision;
            const auto stale_content = testing::RuntimeAccess::migrate(
                buffer, 0U, destination, &phases, &content_only_token);
            VRAMZ_CHECK(runner,
                        !stale_content && stale_content.error().code == ErrorCode::conflict);
            unchanged();
            VRAMZ_CHECK(runner, phases.count == 0U);
        }
        // Explicit M2 transitions intentionally need no policy access token.
        MigrationConstraints explicit_request{
            after.authoritative.resource, after.authoritative.content, {}};
        VRAMZ_CHECK(runner, testing::RuntimeAccess::migrate(buffer, 0U, destination, nullptr,
                                                            &explicit_request));
    }
    VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == owned);
    for (const auto tier : {PhysicalTier::gpu, PhysicalTier::host}) {
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, tier));
    }
    VRAMZ_CHECK(runner,
                runtime.stats().gpu.cleanup_debt == ByteSize{} &&
                    runtime.stats().host.cleanup_debt == ByteSize{} &&
                    testing::chunk_snapshot(buffer, 0U).value().cleanup_resource_count == 0U);
    // The caller's new revision must not invalidate its own acquire-owned-intent restoration.
    const auto before_restore = testing::chunk_snapshot(buffer, 0U).value();
    auto restored = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
    std::array<std::byte, 4096U> output{};
    VRAMZ_CHECK(runner, restored && testing::read_bytes(restored.value(), ByteOffset{}, output));
    VRAMZ_CHECK(runner, output == bytes && restored.value().close());
    VRAMZ_CHECK(runner, testing::chunk_snapshot(buffer, 0U).value().access_revision ==
                            before_restore.access_revision + 1U);
    VRAMZ_CHECK(runner, buffer.close());
    VRAMZ_CHECK(runner, runtime.shutdown());
    VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U);
    VRAMZ_CHECK(runner, runtime.stats().gpu.committed == ByteSize{} &&
                            runtime.stats().host.committed == ByteSize{});
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu) &&
                            testing::accounting_conserved(runtime, PhysicalTier::host));
}

} // namespace

int main() {
    test::Runner runner;
    runner.begin("real M2 byte probes provide exact compressed charges for scaled model");
    const auto compressed = measured_charge(runner, simulation::Dataset::repeated);
    const auto incompressible = measured_charge(runner, simulation::Dataset::random);
    for (const auto percent : simulation::expansion_percent) {
        runner.begin("scaled 8 GiB budget conserves every phase and matches byte-unit policy at "
                     "target ratio");
        ScaledModel small{runner, 1U, compressed};
        ScaledModel large{runner, 65536U, compressed};
        const auto count = 32U * percent / 100U;
        VRAMZ_CHECK(runner, small.populate(count) == count);
        VRAMZ_CHECK(runner, large.populate(count) == count);
        VRAMZ_CHECK(runner, small.transitions() == large.transitions());
        VRAMZ_CHECK(runner, large.committed().value() == small.committed().value() * 65536U);
        small.close();
        large.close();
    }
    runner.begin("incompressible scaled 24 GiB request honestly stops at normal raw admission");
    ScaledModel poor{runner, 65536U, incompressible};
    VRAMZ_CHECK(runner, poor.populate(96U) == 29U);
    VRAMZ_CHECK(runner, poor.transitions() == 0U);
    poor.close();

    runner.begin("checked scale rejects zero and overflow rather than inventing larger capacity");
    simulation::Scenario scenario{};
    simulation::ScenarioResult result{};
    result.requested_logical = ByteSize{393216U};
    scenario.scale_factor = 0U;
    VRAMZ_CHECK(runner, !simulation::scale(scenario, result));
    scenario.scale_factor = std::numeric_limits<std::uint64_t>::max();
    VRAMZ_CHECK(runner, !simulation::scale(scenario, result));

    runner.begin(
        "deterministic scenario repeats final counters bytes transitions and pressure decisions");
    scenario = simulation::Scenario{};
    scenario.steps = 32U;
    const auto first = simulation::run(scenario);
    const auto second = simulation::run(scenario);
    healthy(runner, first, scenario);
    healthy(runner, second, scenario);
    VRAMZ_CHECK(runner, first.value().capacity_pass && second.value().capacity_pass);
    VRAMZ_CHECK(runner, first.value().stats.gpu.committed == second.value().stats.gpu.committed);
    VRAMZ_CHECK(runner, first.value().stats.policy.policy_transitions ==
                            second.value().stats.policy.policy_transitions);
    VRAMZ_CHECK(runner, first.value().stats.policy.last_victim_buffer ==
                            second.value().stats.policy.last_victim_buffer);
    VRAMZ_CHECK(runner, first.value().stats.policy.decision_digest ==
                            second.value().stats.policy.decision_digest);
    VRAMZ_CHECK(runner, first.value().stats.policy.simulated_compression_cost ==
                            second.value().stats.policy.simulated_compression_cost);
    VRAMZ_CHECK(runner, first.value().stats.policy.logical_host_bytes == ByteSize{});
    VRAMZ_CHECK(runner, first.value().scaled.gpu_budget == ByteSize{8589934592ULL});
    VRAMZ_CHECK(runner, first.value().scaled.requested_logical == ByteSize{25769803776ULL});

    for (const auto percent : {150U, 200U, 250U, 300U}) {
        runner.begin("mixed-data progressive capacity failures are reported without corruption or "
                     "hidden host bytes");
        scenario.target_percent = percent;
        scenario.dataset = simulation::Dataset::mixed;
        scenario.trace = simulation::Trace::streaming;
        scenario.steps = 16U;
        const auto mixed = simulation::run(scenario);
        healthy(runner, mixed, scenario);
        VRAMZ_CHECK(runner, mixed.value().stats.policy.logical_host_bytes == ByteSize{});
        VRAMZ_CHECK(runner, mixed.value().capacity_pass == (percent <= 200U));
    }

    runner.begin("host fallback is explicit and cannot turn its larger workload into GPU-only "
                 "capacity PASS");
    scenario = simulation::Scenario{};
    scenario.target_percent = 200U;
    scenario.steps = 16U;
    scenario.dataset = simulation::Dataset::random;
    const auto gpu_only = simulation::run(scenario);
    healthy(runner, gpu_only, scenario);
    VRAMZ_CHECK(runner, !gpu_only.value().allocation_success &&
                            gpu_only.value().stats.policy.host_fallback_count == 0U);
    scenario.mode = PolicyMode::gpu_resident_with_host_fallback;
    const auto fallback = simulation::run(scenario);
    healthy(runner, fallback, scenario);
    VRAMZ_CHECK(runner,
                fallback.value().allocation_success && fallback.value().failed_accesses == 0U);
    VRAMZ_CHECK(runner, fallback.value().stats.policy.host_fallback_count > 0U &&
                            !fallback.value().capacity_pass);
    VRAMZ_CHECK(runner, fallback.value().stats.policy.logical_gpu_resident_bytes <
                            fallback.value().allocated_logical);

    runner.begin(
        "all seven generic traces run real payloads with accounting and cleanup validation");
    scenario = simulation::Scenario{};
    scenario.target_percent = 150U;
    scenario.steps = 32U;
    for (const auto trace : simulation::traces) {
        scenario.trace = trace;
        const auto trace_result = simulation::run(scenario);
        healthy(runner, trace_result, scenario);
        VRAMZ_CHECK(runner, trace_result.value().capacity_pass);
    }

    runner.begin("long deterministic graphics trace interleaves thousands of reads writes pressure "
                 "and migrations");
    scenario = simulation::Scenario{};
    scenario.target_percent = 150U;
    scenario.steps = 2048U;
    scenario.trace = simulation::Trace::graphics;
    const auto long_run = simulation::run(scenario);
    healthy(runner, long_run, scenario);
    VRAMZ_CHECK(runner, long_run.value().capacity_pass && long_run.value().operations > 2048U);

    runner.begin(
        "repeated allocation release and pressure mix remains bounded with exact ownership");
    auto config = test::test_runtime_config();
    config.budgets.gpu = TierBudget{ByteSize{32768U}, ByteSize{16384U}, ByteSize{12288U}};
    config.preferred_chunk_size = ByteSize{4096U};
    config.policy.mode = PolicyMode::gpu_resident;
    auto runtime = std::move(Runtime::create(config)).value();
    auto stable = std::move(runtime.allocate(ByteSize{4096U})).value();
    for (std::uint64_t epoch = 0U; epoch < 1000U; ++epoch) {
        auto temporary = runtime.allocate(ByteSize{4096U});
        VRAMZ_CHECK(runner, temporary);
        auto access = stable.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
        VRAMZ_CHECK(runner, access && access.value().close());
        if (epoch % 8U == 7U) {
            static_cast<void>(runtime.reclaim_to_target(ByteSize{6000U}));
        }
        VRAMZ_CHECK(runner, temporary.value().close());
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
        VRAMZ_CHECK(runner, runtime.stats().policy.host_fallback_count == 0U);
    }
    VRAMZ_CHECK(runner, stable.close());
    VRAMZ_CHECK(runner, runtime.shutdown());
    VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U);
    constexpr std::array sequences{
        "selected victim then same-chunk read conflicts with no mutation",
        "selected victim then unrelated read preserves per-chunk freshness",
        "selected victim then write independently invalidates access and ContentTag tokens",
        "selected host-fallback victim then same-chunk read conflicts before backend work"};
    for (std::uint32_t sequence = 0U; sequence < sequences.size(); ++sequence) {
        runner.begin(sequences[sequence]);
        for (std::uint32_t iteration = 0U; iteration < 32U; ++iteration) {
            access_sequence(runner, sequence);
        }
    }
    constexpr std::array cross_phase_sequences{
        "stale compression excludes every fallback action then permits fresh next-cycle evaluation",
        "stale chunk A leaves unrelated chunk B eligible for verified compression and host "
        "fallback",
        "genuine compression-not-beneficial permits same-cycle fallback without stale exclusion"};
    for (std::uint32_t sequence = 0U; sequence < cross_phase_sequences.size(); ++sequence) {
        runner.begin(cross_phase_sequences[sequence]);
        for (std::uint64_t generation = 0U; generation < 32U; ++generation) {
            cross_phase_sequence(runner, static_cast<CrossPhaseSequence>(sequence), generation);
        }
    }
    return runner.finish();
}
