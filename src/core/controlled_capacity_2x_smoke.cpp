#include "vramz/detail/controlled_capacity_2x_smoke.hpp"

#include "vramz/checked.hpp"
#include "vramz/crc32c.hpp"
#include "vramz/detail/allocation.hpp"
#include "vramz/testing.hpp"

#include <algorithm>
#include <exception>
#include <new>
#include <optional>
#include <utility>
#include <vector>

namespace vramz::detail {

RuntimeConfig controlled_capacity_2x_configuration() noexcept {
    RuntimeConfig config{};
    config.required_capabilities.host_tier = false;
    config.preferred_chunk_size = m13_logical_payload;
    config.budgets.gpu = {m13_physical_cap, m13_settled_target, m13_migration_reserve};
    config.policy.mode = PolicyMode::gpu_resident;
    config.policy.tuning.maximum_transitions = 1U;
    return config;
}

Result<bool> controlled_capacity_2_0x(ByteSize logical, ByteSize physical) noexcept {
    const auto right = checked_mul(physical.value(), 2U, OperationId::verify);
    if (!right) {
        return right.error();
    }
    return logical == m13_aggregate_logical && physical != ByteSize{} &&
           physical <= m13_settled_target && logical.value() >= right.value();
}

Result<ByteSize> admit_controlled_capacity_2x_stage(ByteSize live, ByteSize destination,
                                                    ByteSize workspace,
                                                    ByteSize compaction) noexcept {
    TierUsage usage{};
    usage.committed = live;
    usage.staging = destination;
    usage.workspace = workspace;
    usage.reserved = compaction;
    const auto peak = policy_pressure_charge(usage);
    if (!peak) {
        return peak.error();
    }
    if (peak.value() > m13_physical_cap) {
        return make_error(ErrorCode::unsupported, OperationId::budget_reserve);
    }
    return peak;
}

namespace {
void fail(ControlledCapacity2xSmokeReport& report, Error error) noexcept {
    report.has_error = true;
    report.error = error;
    report.result = CompressionSmokeResult::failed;
}

[[nodiscard]] Pressure pressure(const TierUsage& usage) noexcept {
    const auto charged = policy_pressure_charge(usage);
    return charged ? policy::pressure(charged.value(), m13_settled_target, m13_physical_cap,
                                      m13_migration_reserve)
                   : Pressure::critical;
}

// Buffer/Chunk ownership and all mutations remain inside Runtime and its coordinator.
// The existing internal test factory supplies the backend without changing Runtime::create.
// Physical construction is reachable only from the separate default-OFF M13 executable.
class CapacitySession final : public testing::PolicySelectionObserver {
  public:
    CapacitySession(Runtime runtime, CudaVmmBackend& backend,
                    ControlledCapacity2xSmokeReport& report,
                    ControlledCapacity2xObserver* observer) noexcept
        : runtime_(std::move(runtime)), backend_(backend), report_(report), observer_(observer) {}
    CapacitySession(const CapacitySession&) = delete;
    CapacitySession& operator=(const CapacitySession&) = delete;
    ~CapacitySession() noexcept override { finish(); }

    void run() {
        std::vector<std::byte> initial(static_cast<std::size_t>(m13_logical_payload.value()));
        std::vector<std::byte> observed(initial.size());
        // Actual leases protect partially initialized buffers from allocation maintenance.
        // They are all closed before the initial snapshot and requested policy cycles.
        std::array<std::optional<Lease>, m13_chunk_count> initializing{};
        for (std::size_t index = 0U; index < buffers_.size(); ++index) {
            auto& record = report_.chunks[index].integrity;
            record.chunk_index = static_cast<std::uint32_t>(index);
            fill(index, initial);
            record.logical_crc_expected = crc32c(initial);
            if (!admit(report_.admission.raw_charge, {}, {})) {
                return;
            }
            auto allocated = runtime_.allocate(m13_logical_payload);
            if (!accept(allocated)) {
                return;
            }
            buffers_[index].emplace(std::move(allocated).value());
            identities_[index] = testing::RuntimeAccess::buffer_identity(owned_buffer(index));
            auto acquired = owned_buffer(index).acquire(full_range(), {AccessMode::read_write});
            if (!accept(acquired)) {
                return;
            }
            auto& initialized = initializing[index].emplace(std::move(acquired).value());
            if (!accept(testing::write_bytes(initialized, {}, initial)) ||
                !accept(testing::read_bytes(initialized, {}, observed))) {
                return;
            }
            if (initial != observed || crc32c(observed) != record.logical_crc_expected) {
                fail(report_, make_error(ErrorCode::integrity_failure, OperationId::verify));
                return;
            }
            record.source_verified = true;
        }
        for (auto& lease : initializing) {
            if (!lease.has_value()) {
                contract();
            }
            if (!accept(lease->close())) {
                stop(report_.error);
            }
            lease.reset();
        }
        // The precise M10 bytes are unchanged. Only ordinary read access builds heat.
        for (std::size_t index = 1U; index <= 6U; ++index) {
            if (!access(index)) {
                return;
            }
        }
        for (std::uint32_t count = 0U; count < 16U; ++count) {
            if (!access(0U)) {
                return;
            }
        }
        if (!access(7U)) {
            return;
        }
        for (std::uint32_t count = 0U; count < 3U; ++count) {
            if (!access(0U)) {
                return;
            }
        }
        if (!initial_snapshot()) {
            return;
        }
        event(ControlledCapacity2xEventKind::initial_snapshot, m13_chunk_count);
        for (std::uint32_t cycle = 0U; cycle < m13_maximum_cycles; ++cycle) {
            if (policy_pressure_charge(runtime_.stats().gpu).value() <= m13_settled_target) {
                break;
            }
            if (!run_cycle(cycle)) {
                return;
            }
        }
        if (!settled_snapshot()) {
            return;
        }
        event(ControlledCapacity2xEventKind::policy_settled_snapshot, m13_chunk_count);
        for (std::size_t index = 0U; index < buffers_.size(); ++index) {
            fill(index, initial);
            if (!verify_chunk(index, initial, observed)) {
                return;
            }
        }
        verified_ = true;
    }

    void on_selected(const ChunkSnapshot& snapshot,
                     const policy::Proposal& proposal) noexcept override {
        auto& cycle = report_.cycles[report_.cycle_count - 1U];
        if (cycle.selected || proposal.action != PolicyAction::compress_gpu ||
            snapshot.authoritative.state != RepresentationState::gpu_raw) {
            contract();
        }
        const auto found = std::ranges::find(identities_, proposal.buffer);
        if (found == identities_.end() || proposal.chunk != ChunkId{} ||
            snapshot.id != proposal.chunk) {
            contract();
        }
        cycle.selected_chunk = static_cast<std::uint32_t>(found - identities_.begin());
        cycle.cycle_id = runtime_.stats().policy.policy_cycles;
        cycle.proposal = proposal;
        cycle.selected_revision = snapshot.access_revision;
        cycle.representation_before = snapshot.authoritative.state;
        cycle.charge_before = snapshot.authoritative.charge;
        cycle.selected = true;
        ++report_.chunks[cycle.selected_chunk].selected_count;
    }

    void on_completed(const ChunkSnapshot& snapshot, const Result<void>& result) noexcept override {
        auto& cycle = report_.cycles[report_.cycle_count - 1U];
        if (!cycle.selected || cycle.completed) {
            contract();
        }
        cycle.completed = true;
        cycle.transition_succeeded = static_cast<bool>(result);
        cycle.charge_after = snapshot.authoritative.charge;
        cycle.after = runtime_.stats().gpu;
        cycle.pressure_after = pressure(cycle.after);
        if (!result) {
            cycle.has_error = true;
            cycle.error = result.error();
            cycle.stale_rejection = result.error().code == ErrorCode::conflict;
        }
        if (snapshot.cleanup_resource_count != 0U ||
            snapshot.lifecycle == LifecycleState::poisoned) {
            stop(snapshot.first_error.value_or(
                result ? make_error(ErrorCode::poisoned, OperationId::migrate) : result.error()));
        }
    }

    void on_transaction_phase(TransactionPhase phase, const ChunkSnapshot&) noexcept override {
        auto& cycle = report_.cycles[report_.cycle_count - 1U];
        if (cycle.stage_count >= cycle.stages.size()) {
            contract();
        }
        const auto usage = runtime_.stats().gpu;
        const auto charged = policy_pressure_charge(usage);
        const auto physical = backend_.owned_charge(PhysicalTier::gpu);
        const auto total =
            checked_add(physical.value(), usage.reserved.value(), OperationId::verify);
        if (!charged || !total || charged.value().value() != total.value() ||
            charged.value() > report_.peak_admitted_gpu_bytes ||
            !testing::accounting_conserved(runtime_, PhysicalTier::gpu)) {
            contract();
        }
        cycle.stages[cycle.stage_count++] = {phase, usage, physical,
                                             backend_.owned_resource_count()};
        cycle.workspace_materializations +=
            phase == TransactionPhase::workspace_materialized ? 1U : 0U;
        cycle.charge_reconciled =
            cycle.charge_reconciled || phase == TransactionPhase::charge_reconciled;
    }

  private:
    Buffer& owned_buffer(std::size_t index) noexcept {
        if (index >= buffers_.size()) {
            contract();
        }
        auto& buffer = buffers_[index];
        if (!buffer.has_value()) {
            contract();
        }
        return *buffer;
    }
    static MemoryRange full_range() noexcept { return {{}, m13_logical_payload}; }
    static void fill(std::size_t chunk, std::span<std::byte> bytes) noexcept {
        for (std::size_t index = 0U; index < bytes.size(); ++index) {
            bytes[index] = controlled_capacity_2x_payload_byte(chunk, index);
        }
    }
    template <typename T> bool accept(const Result<T>& result) noexcept {
        if (!result) {
            if (result.error().code == ErrorCode::ambiguous_backend_state ||
                result.error().code == ErrorCode::backend_contract_violation) {
                stop(result.error());
            }
            fail(report_, result.error());
            return false;
        }
        return true;
    }
    void event(ControlledCapacity2xEventKind kind, std::size_t index) noexcept {
        if (observer_ != nullptr) {
            observer_->observe({kind, static_cast<std::uint32_t>(index)});
        }
    }
    [[noreturn]] void contract() noexcept {
        stop(make_error(ErrorCode::backend_contract_violation, OperationId::verify));
    }
    bool access(std::size_t index) noexcept {
        auto acquired = owned_buffer(index).acquire(full_range());
        if (!accept(acquired)) {
            return false;
        }
        const auto closed = acquired.value().close();
        if (!closed) {
            stop(closed.error());
        }
        return true;
    }
    bool corroborate(std::size_t index, ChunkSnapshot& snapshot) noexcept {
        const auto inspected = testing::chunk_snapshot(owned_buffer(index), 0U);
        if (!accept(inspected)) {
            return false;
        }
        snapshot = inspected.value();
        const auto& authoritative = snapshot.authoritative;
        const auto allocation = backend_.allocation(authoritative.resource);
        if (!accept(allocation)) {
            return false;
        }
        if (allocation.value().id != authoritative.resource ||
            allocation.value().tier != PhysicalTier::gpu ||
            allocation.value().kind != ResourceKind::representation ||
            allocation.value().charge != authoritative.charge ||
            !same_metadata(allocation.value().metadata, authoritative.metadata) ||
            allocation.value().address != authoritative.address ||
            authoritative.metadata.logical_size != m13_logical_payload ||
            authoritative.charge == ByteSize{} ||
            authoritative.charge.value() % report_.probe.minimum.value() != 0U ||
            snapshot.read_pins != 0U || snapshot.write_pin || snapshot.lease_intents != 0U ||
            snapshot.transition_active || snapshot.cleanup_resource_count != 0U ||
            snapshot.lifecycle != LifecycleState::live) {
            contract();
        }
        return true;
    }
    ByteSize corroborate_accounting() noexcept {
        const auto stats = runtime_.stats();
        const auto charged = policy_pressure_charge(stats.gpu);
        const auto host = policy_pressure_charge(stats.host);
        if (!charged || !host || host.value() != ByteSize{} ||
            !testing::accounting_conserved(runtime_, PhysicalTier::gpu) ||
            !testing::accounting_conserved(runtime_, PhysicalTier::host) ||
            stats.gpu.reserved != ByteSize{} || stats.gpu.staging != ByteSize{} ||
            stats.gpu.workspace != ByteSize{} || stats.gpu.cleanup_debt != ByteSize{} ||
            stats.completions.pending_completion_count != 0U ||
            stats.policy.host_fallback_count != 0U ||
            charged.value() != backend_.owned_charge(PhysicalTier::gpu) ||
            charged.value() > m13_physical_cap) {
            contract();
        }
        return charged.value();
    }
    bool admit(ByteSize destination, ByteSize workspace, ByteSize compaction) noexcept {
        const auto live = corroborate_accounting();
        const auto peak =
            admit_controlled_capacity_2x_stage(live, destination, workspace, compaction);
        if (!accept(peak)) {
            report_.result = CompressionSmokeResult::resource_cap;
            return false;
        }
        report_.peak_admitted_gpu_bytes = std::max(report_.peak_admitted_gpu_bytes, peak.value());
        return true;
    }
    bool initial_snapshot() noexcept {
        const auto charged = corroborate_accounting();
        report_.initial_usage = runtime_.stats().gpu;
        report_.initial_pressure = pressure(report_.initial_usage);
        for (std::size_t index = 0U; index < buffers_.size(); ++index) {
            ChunkSnapshot snapshot{};
            if (!corroborate(index, snapshot)) {
                return false;
            }
            auto& record = report_.chunks[index];
            if (snapshot.authoritative.state != RepresentationState::gpu_raw ||
                snapshot.authoritative.metadata.crc32c != record.integrity.logical_crc_expected ||
                owned_buffer(index).stats().chunk_count != 1U) {
                contract();
            }
            const auto metadata = testing::RuntimeAccess::policy_metadata(owned_buffer(index), 0U);
            if (!accept(metadata)) {
                return false;
            }
            record.access = metadata.value();
            record.access_revision = snapshot.access_revision;
            record.effective_frequency =
                policy::frequency(record.access, runtime_.stats().policy.access_epoch,
                                  controlled_capacity_2x_configuration().policy.tuning);
            record.temperature =
                policy::temperature(record.access, runtime_.stats().policy.access_epoch,
                                    controlled_capacity_2x_configuration().policy.tuning);
            record.integrity.raw_charge = snapshot.authoritative.charge;
            const auto total =
                checked_add(report_.aggregate_raw_charge.value(),
                            snapshot.authoritative.charge.value(), OperationId::verify);
            if (!total) {
                contract();
            }
            report_.aggregate_raw_charge = ByteSize{total.value()};
        }
        if (report_.aggregate_raw_charge != charged || report_.initial_pressure != Pressure::hard ||
            charged <= m13_high_threshold || runtime_.stats().policy.compression_attempts != 0U ||
            report_.chunks[0].temperature != Temperature::hot ||
            report_.chunks[1].temperature != Temperature::cold ||
            report_.chunks[2].temperature != Temperature::cold ||
            report_.chunks[3].temperature != Temperature::cold ||
            report_.chunks[4].temperature != Temperature::cold ||
            report_.chunks[5].temperature != Temperature::cold ||
            report_.chunks[6].temperature != Temperature::cold ||
            report_.chunks[7].temperature != Temperature::warm) {
            fail(report_, make_error(ErrorCode::unsupported, OperationId::verify));
            return false;
        }
        report_.initial_snapshot_proven = true;
        return true;
    }
    bool run_cycle(std::uint32_t index) noexcept {
        if (!admit(report_.admission.compression.output_charge,
                   report_.admission.compression.workspace_charge,
                   report_.admission.compression.output_charge)) {
            return false;
        }
        auto& cycle = report_.cycles[index];
        cycle.selected_chunk = static_cast<std::uint32_t>(m13_chunk_count);
        report_.cycle_count = index + 1U;
        cycle.before = runtime_.stats().gpu;
        cycle.pressure_before = pressure(cycle.before);
        report_.codec.compression_enqueue = false;
        report_.codec.compression_completion = false;
        event(ControlledCapacity2xEventKind::policy_cycle_start, index);
        // This call carries only a budget target. The production scan supplies the victim.
        const auto result =
            testing::RuntimeAccess::reclaim(runtime_, m13_settled_target, false, this);
        cycle.cycle_id = runtime_.stats().policy.policy_cycles;
        cycle.after = runtime_.stats().gpu;
        cycle.pressure_after = pressure(cycle.after);
        if (!cycle.selected || !cycle.completed) {
            fail(report_, result ? make_error(ErrorCode::unsupported, OperationId::migrate)
                                 : result.error());
            return false;
        }
        const auto metadata =
            testing::RuntimeAccess::policy_metadata(owned_buffer(cycle.selected_chunk), 0U);
        if (!accept(metadata)) {
            return false;
        }
        cycle.compressibility_after = metadata.value().compressibility;
        cycle.retry_after_epoch = metadata.value().retry_after;
        if (!cycle.transition_succeeded) {
            // A nonbeneficial result is handled by the existing policy metadata. A new
            // bounded cycle may choose another eligible victim; no failed operation is retried.
            if (cycle.error.code == ErrorCode::compression_not_beneficial) {
                static_cast<void>(corroborate_accounting());
                return true;
            }
            fail(report_, cycle.error);
            return false;
        }
        if (!result && result.error().code != ErrorCode::out_of_gpu_memory) {
            fail(report_, result.error());
            return false;
        }
        ChunkSnapshot snapshot{};
        if (!corroborate(cycle.selected_chunk, snapshot)) {
            return false;
        }
        if (snapshot.authoritative.state != RepresentationState::gpu_compressed ||
            cycle.charge_before != report_.chunks[cycle.selected_chunk].integrity.raw_charge ||
            cycle.charge_after != snapshot.authoritative.charge) {
            contract();
        }
        auto& record = report_.chunks[cycle.selected_chunk].integrity;
        record.compressed_charge = snapshot.authoritative.charge;
        record.stored_bytes = snapshot.authoritative.metadata.stored_size;
        record.stored_crc_expected = snapshot.authoritative.metadata.stored_crc32c;
        record.stored_crc_actual = report_.codec.stored_crc_actual;
        record.compression_enqueue = report_.codec.compression_enqueue;
        record.compression_completion = report_.codec.compression_completion;
        record.charges_corroborated = true;
        record.compaction_exercised =
            cycle.workspace_materializations == 2U && cycle.charge_reconciled;
        record.minimum_unit_saved =
            saves_physical_unit(record.raw_charge, record.compressed_charge, report_.probe.minimum);
        if (!record.minimum_unit_saved) {
            fail(report_, make_error(ErrorCode::unsupported, OperationId::verify));
            report_.result = CompressionSmokeResult::no_physical_savings;
            return false;
        }
        record.physical_bytes_saved =
            ByteSize{record.raw_charge.value() - record.compressed_charge.value()};
        cycle.reclaimed = record.physical_bytes_saved;
        ++report_.automatic_policy_actions;
        static_cast<void>(corroborate_accounting());
        if (!record.compression_completion || !record.compression_enqueue ||
            !record.compaction_exercised ||
            record.stored_crc_actual != record.stored_crc_expected ||
            (cycle.selected_chunk == 0U || cycle.selected_chunk == 7U)) {
            fail(report_, make_error(ErrorCode::integrity_failure, OperationId::verify));
            return false;
        }
        return true;
    }
    bool settled_snapshot() noexcept {
        report_.aggregate_settled_charge = corroborate_accounting();
        report_.settled_usage = runtime_.stats().gpu;
        report_.settled_pressure = pressure(report_.settled_usage);
        report_.pressure_target_reached = report_.aggregate_settled_charge <= m13_settled_target;
        ByteSize sum{};
        ByteSize logical{};
        for (std::size_t index = 0U; index < buffers_.size(); ++index) {
            ChunkSnapshot snapshot{};
            if (!corroborate(index, snapshot)) {
                return false;
            }
            report_.chunks[index].settled = snapshot.authoritative.state;
            const auto next = checked_add(sum.value(), snapshot.authoritative.charge.value(),
                                          OperationId::verify);
            if (!next) {
                contract();
            }
            sum = ByteSize{next.value()};
            const auto next_logical =
                checked_add(logical.value(), snapshot.authoritative.metadata.logical_size.value(),
                            OperationId::verify);
            if (!next_logical) {
                contract();
            }
            logical = ByteSize{next_logical.value()};
        }
        report_.aggregate_logical_bytes = logical;
        const auto capacity = controlled_capacity_2_0x(logical, sum);
        report_.hot_chunk_preserved_raw =
            report_.chunks[0].settled == RepresentationState::gpu_raw &&
            report_.chunks[0].selected_count == 0U;
        report_.warm_chunk_preserved_raw =
            report_.chunks[7].settled == RepresentationState::gpu_raw &&
            report_.chunks[7].selected_count == 0U;
        const auto saved =
            checked_sub(report_.aggregate_raw_charge.value(), sum.value(), OperationId::verify);
        if (!capacity || !capacity.value() || !saved || sum != report_.aggregate_settled_charge ||
            !report_.pressure_target_reached || !report_.hot_chunk_preserved_raw ||
            !report_.warm_chunk_preserved_raw || report_.automatic_policy_actions != 6U ||
            report_.cycle_count != 6U || report_.cycles[0].selected_chunk != 1U ||
            report_.cycles[1].selected_chunk != 2U || report_.cycles[2].selected_chunk != 3U ||
            report_.cycles[3].selected_chunk != 4U || report_.cycles[4].selected_chunk != 5U ||
            report_.cycles[5].selected_chunk != 6U ||
            report_.chunks[1].settled != RepresentationState::gpu_compressed ||
            report_.chunks[2].settled != RepresentationState::gpu_compressed ||
            report_.chunks[3].settled != RepresentationState::gpu_compressed ||
            report_.chunks[4].settled != RepresentationState::gpu_compressed ||
            report_.chunks[5].settled != RepresentationState::gpu_compressed ||
            report_.chunks[6].settled != RepresentationState::gpu_compressed ||
            report_.chunks[7].settled != RepresentationState::gpu_raw ||
            report_.settled_pressure != Pressure::normal) {
            fail(report_, make_error(ErrorCode::unsupported, OperationId::verify));
            return false;
        }
        report_.reclaimed = ByteSize{saved.value()};
        report_.controlled_capacity_snapshot_proven = true;
        return true;
    }
    bool verify_chunk(std::size_t index, std::span<const std::byte> expected,
                      std::span<std::byte> observed) noexcept {
        if (!report_.controlled_capacity_snapshot_proven) {
            contract();
        }
        auto& record = report_.chunks[index];
        if (record.settled == RepresentationState::gpu_compressed &&
            !admit(report_.admission.raw_charge, report_.admission.decompression.workspace_charge,
                   {})) {
            return false;
        }
        event(ControlledCapacity2xEventKind::integrity_start, index);
        const auto actions = runtime_.stats().policy.compression_attempts;
        report_.codec.decompression_enqueue = false;
        report_.codec.decompression_completion = false;
        auto acquired = owned_buffer(index).acquire(full_range());
        if (!accept(acquired)) {
            return false;
        }
        const auto read = testing::read_bytes(acquired.value(), {}, observed);
        const bool read_ok = accept(read);
        const auto close = acquired.value().close();
        if (!close) {
            stop(close.error());
        }
        if (!read_ok) {
            return false;
        }
        ChunkSnapshot snapshot{};
        if (!corroborate(index, snapshot)) {
            return false;
        }
        record.integrity.logical_crc_actual = crc32c(observed);
        record.integrity.size_equal =
            snapshot.authoritative.metadata.logical_size == m13_logical_payload;
        record.integrity.byte_equal = std::ranges::equal(expected, observed);
        if (record.settled == RepresentationState::gpu_compressed) {
            record.integrity.decompression_enqueue = report_.codec.decompression_enqueue;
            record.integrity.decompression_completion = report_.codec.decompression_completion;
            if (!record.integrity.decompression_enqueue ||
                !record.integrity.decompression_completion ||
                report_.codec.stored_crc_expected != report_.codec.stored_crc_actual) {
                fail(report_, make_error(ErrorCode::integrity_failure, OperationId::verify));
                return false;
            }
        }
        if (snapshot.authoritative.state != RepresentationState::gpu_raw ||
            record.integrity.logical_crc_actual != record.integrity.logical_crc_expected ||
            !record.integrity.byte_equal || !record.integrity.size_equal ||
            runtime_.stats().policy.compression_attempts != actions) {
            fail(report_, make_error(ErrorCode::integrity_failure, OperationId::verify));
            return false;
        }
        record.integrity_verified = true;
        record.integrity.round_trip_verified = true;
        // Closing each verified buffer keeps restoration below the settled pressure target.
        const auto closed = owned_buffer(index).close();
        if (!closed) {
            stop(closed.error());
        }
        record.integrity.cleanup_complete = true;
        buffers_[index].reset();
        static_cast<void>(corroborate_accounting());
        return true;
    }
    void snapshot_final() noexcept {
        report_.final_usage = runtime_.stats().gpu;
        report_.final_resources = backend_.owned_resource_count();
        report_.final_va = backend_.owned_address_count();
        const auto counts = backend_.resource_counts();
        report_.final_raw = counts.raw;
        report_.final_compressed = counts.compressed;
        report_.final_workspace = counts.workspace;
        report_.final_counts_known = testing::accounting_conserved(runtime_, PhysicalTier::gpu) &&
                                     testing::accounting_conserved(runtime_, PhysicalTier::host);
    }
    [[noreturn]] void stop(Error error) noexcept {
        fail(report_, error);
        if (error.code != ErrorCode::ambiguous_backend_state &&
            error.code != ErrorCode::backend_contract_violation) {
            snapshot_final();
        } else {
            report_.final_counts_known = false;
        }
        // No destructors/retries after uncertain completion or any definite cleanup failure.
        std::terminate();
    }
    void finish() noexcept {
        if (finished_) {
            return;
        }
        finished_ = true;
        if (report_.codec.has_fatal_error) {
            stop(report_.codec.fatal_error);
        }
        for (std::size_t index = 0U; index < buffers_.size(); ++index) {
            if (buffers_[index]) {
                const auto snapshot = testing::chunk_snapshot(owned_buffer(index), 0U);
                if (snapshot && (snapshot.value().cleanup_resource_count != 0U ||
                                 snapshot.value().lifecycle == LifecycleState::poisoned)) {
                    stop(snapshot.value().first_error.value_or(
                        make_error(ErrorCode::poisoned, OperationId::release)));
                }
                const auto closed = owned_buffer(index).close();
                if (!closed) {
                    stop(closed.error());
                }
                report_.chunks[index].integrity.cleanup_complete = true;
                buffers_[index].reset();
            }
        }
        const auto shutdown = runtime_.shutdown();
        if (!shutdown) {
            stop(shutdown.error());
        }
        snapshot_final();
        const auto charged = policy_pressure_charge(report_.final_usage);
        if (!charged || charged.value() != ByteSize{} || report_.final_resources != 0U ||
            report_.final_va != 0U || report_.codec.stream_owned || report_.codec.context_owned ||
            !report_.final_counts_known) {
            contract();
        }
        report_.cleanup_raw = true;
        report_.cleanup_compressed = true;
        report_.cleanup_workspace = true;
        report_.cleanup_va = true;
        report_.cleanup_stream = true;
        report_.cleanup_context = true;
        event(ControlledCapacity2xEventKind::cleanup_complete, m13_chunk_count);
        if (verified_ && !report_.has_error) {
            report_.controlled_capacity_2_0x_proven = true;
            report_.result = CompressionSmokeResult::passed;
            event(ControlledCapacity2xEventKind::pass_published, m13_chunk_count);
        }
    }

    Runtime runtime_;
    CudaVmmBackend& backend_;
    ControlledCapacity2xSmokeReport& report_;
    ControlledCapacity2xObserver* observer_{};
    std::array<std::optional<Buffer>, m13_chunk_count> buffers_{};
    std::array<BufferId, m13_chunk_count> identities_{};
    bool verified_{};
    bool finished_{};
};
} // namespace

void finalize_controlled_capacity_2x_smoke_failure(
    ControlledCapacity2xSmokeReport& report) noexcept {
    if (report.codec.has_fatal_error) {
        fail(report, report.codec.fatal_error);
        report.final_counts_known = false;
    } else if (!report.has_error) {
        fail(report, make_error(ErrorCode::ambiguous_backend_state, OperationId::migrate));
        report.final_counts_known = false;
    }
}

void run_controlled_capacity_2x_smoke(std::unique_ptr<CudaDriverApi> driver,
                                      std::unique_ptr<GpuCompressionApi> codec,
                                      ControlledCapacity2xSmokeOptions options,
                                      ControlledCapacity2xSmokeReport& report,
                                      ControlledCapacity2xObserver* observer) noexcept {
    report = {};
    report.device_ordinal = options.device;
    report.installed_driver = options.installed_driver;
    report.runtime_api = options.runtime_api;
    report.prior_driver_api = options.prior_driver_api;
    if (!options.acknowledged && options.device == 0 && driver && codec) {
        report.final_counts_known = true;
        return;
    }
    report.compatibility = compression_runtime_candidate(
        options.prior_driver_api, options.runtime_api, options.installed_driver);
    if (!options.acknowledged || options.device != 0 || !driver || !codec ||
        !is_candidate(report.compatibility)) {
        fail(report, make_error(ErrorCode::unsupported, OperationId::runtime_create));
        report.result = CompressionSmokeResult::preflight_rejected;
        report.final_counts_known = true;
        return;
    }
    report.admission.logical_size = m13_logical_payload;
    report.admission.physical_cap = m13_physical_cap;
    report.admission.minimum_driver_version = 13000;
    report.admission.required_driver_family = 13;
    report.admission.required_device_name = "NVIDIA GeForce RTX 3060";
    auto created = CudaVmmBackend::create(std::move(driver), 0, std::move(codec), &report.probe,
                                          &report.admission, &report.codec);
    if (!created) {
        fail(report, created.error());
        if (created.error().code == ErrorCode::ambiguous_backend_state ||
            created.error().code == ErrorCode::backend_contract_violation) {
            std::terminate();
        }
        if (report.admission.peak > m13_physical_cap ||
            created.error().code == ErrorCode::arithmetic_overflow) {
            report.result = CompressionSmokeResult::resource_cap;
        }
        report.final_counts_known = !report.codec.stream_owned && !report.codec.context_owned;
        return;
    }
    auto* backend = created.value().get();
    auto runtime = testing::RuntimeAccess::create_with_backend(
        controlled_capacity_2x_configuration(), std::move(created).value());
    if (!runtime) {
        fail(report, runtime.error());
        return;
    }
    try {
        CapacitySession session{std::move(runtime).value(), *backend, report, observer};
        session.run();
    } catch (const std::bad_alloc&) {
        fail(report, make_error(ErrorCode::out_of_host_memory, OperationId::allocate));
    }
}

} // namespace vramz::detail
