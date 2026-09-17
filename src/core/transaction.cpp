#include "vramz/transaction.hpp"

#include "vramz/detail/allocation.hpp"

#include <array>
#include <exception>
#include <limits>

namespace vramz {
namespace {
// Only the optional compaction reservation uses this guard. Existing reservation ownership
// is unchanged. Destruction cannot silently lose protected headroom on an early return.
class CompactionReservation final {
  public:
    CompactionReservation(BudgetLedger& ledger, ReservationToken& token, bool active) noexcept
        : ledger_(ledger), token_(token), active_(active) {}
    ~CompactionReservation() { release(); }
    CompactionReservation(const CompactionReservation&) = delete;
    CompactionReservation& operator=(const CompactionReservation&) = delete;
    void consumed() noexcept { active_ = false; }
    void release() noexcept {
        if (active_) {
            if (!ledger_.release_reservation(token_)) {
                std::terminate();
            }
            active_ = false;
        }
    }

  private:
    BudgetLedger& ledger_;
    ReservationToken& token_;
    bool active_{};
};
} // namespace

Chunk::Chunk(ChunkId id, ByteOffset offset, ByteSize logical_size, Representation authoritative,
             DeviceAddress stable_address) noexcept
    : id_(id), offset_(offset), logical_size_(logical_size), lifecycle_(LifecycleState::live),
      authoritative_(authoritative),
      stable_address_(stable_address.value() == 0U ? authoritative.address : stable_address) {}

ChunkSnapshot Chunk::snapshot() const noexcept {
    const std::scoped_lock lock{mutex_};
    return ChunkSnapshot{id_,          lifecycle_,     authoritative_,     transition_epoch_,
                         read_pins_,   write_pin_,     transition_active_, cleanup_resource_count_,
                         first_error_, lease_intents_, access_revision_};
}

Result<TransitionPlan> make_transition_plan(RepresentationState source,
                                            RepresentationState destination) noexcept {
    if (!is_valid(source) || !is_valid(destination) || source == destination) {
        return make_error(ErrorCode::invalid_argument, OperationId::migrate);
    }
    const bool transform = is_raw(source) != is_raw(destination);
    const bool workspace = !is_raw(source) || !is_raw(destination);
    const bool crosses_tier = tier_of(source) != tier_of(destination);
    return TransitionPlan{source,    destination, tier_of(destination), tier_of(destination),
                          transform, workspace,   crosses_tier};
}

TransactionCoordinator::TransactionCoordinator(BudgetLedger& ledger, StorageBackend& backend,
                                               AsyncErrorChannel& async_errors) noexcept
    : ledger_(ledger), backend_(backend), async_errors_(async_errors) {}

void TransactionCoordinator::notify(TransactionObserver* observer, TransactionPhase phase,
                                    const Chunk& chunk) const noexcept {
    if (observer != nullptr) {
        observer->on_phase(phase, chunk.snapshot());
    }
}

void TransactionCoordinator::clear_transition(Chunk& chunk, TransactionId transaction) noexcept {
    const std::scoped_lock lock{chunk.mutex_};
    if (chunk.transaction_ == transaction) {
        chunk.transition_active_ = false;
        chunk.transaction_ = TransactionId{};
    }
}

void TransactionCoordinator::poison(Chunk& chunk, TransactionId transaction, Error error) noexcept {
    {
        const std::scoped_lock lock{chunk.mutex_};
        if (!chunk.first_error_.has_value()) {
            chunk.first_error_ = error;
        }
        chunk.lifecycle_ = LifecycleState::poisoned;
        if (chunk.transaction_ == transaction) {
            chunk.transition_active_ = false;
            chunk.transaction_ = TransactionId{};
        }
    }
    async_errors_.push(error);
}

void TransactionCoordinator::rollback_resource(const BackendAllocation& allocation,
                                               ChargeBucket bucket, Error causal_error,
                                               Chunk& chunk, TransactionId transaction,
                                               bool poison_on_contract) noexcept {
    const auto released = backend_.release(allocation.id, ReleasePhase::rollback);
    if (released && !backend_.owns(allocation.id)) {
        const auto uncharged =
            ledger_.release_materialized(allocation.tier, bucket, allocation.charge);
        if (!uncharged) {
            poison(chunk, transaction, uncharged.error());
        }
        return;
    }
    const auto debt = ledger_.move_materialized_to_debt(allocation.tier, bucket, allocation.charge);
    if (!debt) {
        poison(chunk, transaction, debt.error());
        return;
    }
    if (!remember_cleanup(chunk, CleanupResource{allocation.id, allocation.tier, allocation.charge},
                          transaction)) {
        return;
    }
    if (released && backend_.owns(allocation.id)) {
        causal_error = make_error(ErrorCode::backend_contract_violation, OperationId::release,
                                  allocation.id.value());
        poison_on_contract = true;
    }
    async_errors_.push(released ? causal_error : released.error());
    if (poison_on_contract) {
        poison(chunk, transaction, causal_error);
    }
}

bool TransactionCoordinator::remember_cleanup(Chunk& chunk, CleanupResource resource,
                                              TransactionId transaction) noexcept {
    const std::scoped_lock lock{chunk.mutex_};
    if (chunk.cleanup_resource_count_ >= chunk.cleanup_resources_.size()) {
        const Error error =
            make_error(ErrorCode::internal_invariant_violation, OperationId::release,
                       chunk.id_.value(), resource.resource.value());
        if (!chunk.first_error_.has_value()) {
            chunk.first_error_ = error;
        }
        chunk.lifecycle_ = LifecycleState::poisoned;
        if (chunk.transaction_ == transaction) {
            chunk.transition_active_ = false;
            chunk.transaction_ = TransactionId{};
        }
        async_errors_.push(error);
        return false;
    }
    chunk.cleanup_resources_[chunk.cleanup_resource_count_] = resource;
    ++chunk.cleanup_resource_count_;
    return true;
}

void TransactionCoordinator::forget_cleanup(Chunk& chunk, ResourceId resource) noexcept {
    const std::scoped_lock lock{chunk.mutex_};
    for (std::uint32_t index = 0U; index < chunk.cleanup_resource_count_; ++index) {
        if (chunk.cleanup_resources_[index].resource == resource) {
            --chunk.cleanup_resource_count_;
            chunk.cleanup_resources_[index] =
                chunk.cleanup_resources_[chunk.cleanup_resource_count_];
            chunk.cleanup_resources_[chunk.cleanup_resource_count_] = CleanupResource{};
            return;
        }
    }
}

Result<void> TransactionCoordinator::migrate(Chunk& chunk, RepresentationState destination,
                                             bool acquire_owned_intent,
                                             TransactionObserver* observer,
                                             const MigrationConstraints* constraints) noexcept {
    auto transaction_value = next_transaction_.load(std::memory_order_relaxed);
    do {
        if (transaction_value == std::numeric_limits<std::uint64_t>::max()) {
            return make_error(ErrorCode::arithmetic_overflow, OperationId::migrate);
        }
    } while (!next_transaction_.compare_exchange_weak(transaction_value, transaction_value + 1U,
                                                      std::memory_order_relaxed,
                                                      std::memory_order_relaxed));
    const TransactionId transaction{transaction_value};
    Representation source{};
    {
        const std::scoped_lock lock{chunk.mutex_};
        if (chunk.lifecycle_ == LifecycleState::poisoned) {
            return make_error(ErrorCode::poisoned, OperationId::migrate, chunk.id_.value());
        }
        if (chunk.lifecycle_ != LifecycleState::live) {
            return make_error(ErrorCode::shutting_down, OperationId::migrate, chunk.id_.value());
        }
        if (constraints != nullptr &&
            (chunk.authoritative_.resource != constraints->expected_resource ||
             chunk.authoritative_.content != constraints->expected_content ||
             (constraints->expected_access_revision.has_value() &&
              chunk.access_revision_ != *constraints->expected_access_revision))) {
            return make_error(ErrorCode::conflict, OperationId::migrate, chunk.id_.value());
        }
        if (chunk.transition_active_ || chunk.read_pins_ != 0U || chunk.write_pin_) {
            return make_error(ErrorCode::busy, OperationId::migrate, chunk.id_.value());
        }
        if (chunk.cleanup_resource_count_ != 0U) {
            return make_error(ErrorCode::busy, OperationId::migrate, chunk.id_.value(),
                              chunk.cleanup_resource_count_);
        }
        if (chunk.transition_epoch_ == std::numeric_limits<std::uint64_t>::max()) {
            return make_error(ErrorCode::arithmetic_overflow, OperationId::migrate,
                              chunk.id_.value());
        }
        if (chunk.lease_intents_ != 0U &&
            (!acquire_owned_intent || destination != RepresentationState::gpu_raw)) {
            return make_error(ErrorCode::busy, OperationId::migrate, chunk.id_.value());
        }
        if (chunk.authoritative_.state == destination) {
            return {};
        }
        source = chunk.authoritative_;
        chunk.transition_active_ = true;
        chunk.transaction_ = transaction;
    }
    const auto source_allocation = backend_.allocation(source.resource);
    const bool invalid_source_metadata =
        !source_allocation || source_allocation.value().id != source.resource ||
        source_allocation.value().tier != tier_of(source.state) ||
        source_allocation.value().charge != source.charge ||
        source_allocation.value().kind != ResourceKind::representation ||
        source_allocation.value().address != source.address ||
        source_allocation.value().metadata.logical_size != chunk.logical_size_ ||
        source_allocation.value().metadata.logical_size != source.metadata.logical_size ||
        source_allocation.value().metadata.stored_size != source.metadata.stored_size ||
        source_allocation.value().metadata.crc32c != source.metadata.crc32c ||
        source_allocation.value().metadata.stored_crc32c != source.metadata.stored_crc32c ||
        source_allocation.value().metadata.stored_size.value() > source.charge.value() ||
        (is_raw(source.state) &&
         (source_allocation.value().metadata.encoding != Encoding::raw ||
          source_allocation.value().metadata.stored_size != chunk.logical_size_)) ||
        (!is_raw(source.state) &&
         (source_allocation.value().metadata.encoding != Encoding::lz4_block ||
          source_allocation.value().metadata.stored_size.value() == 0U));
    if (invalid_source_metadata) {
        const Error error = source_allocation
                                ? make_error(ErrorCode::integrity_failure, OperationId::verify,
                                             source.resource.value())
                                : source_allocation.error();
        poison(chunk, transaction, error);
        notify(observer, TransactionPhase::poisoned, chunk);
        return error;
    }
    notify(observer, TransactionPhase::source_validated, chunk);

    const auto plan = make_transition_plan(source.state, destination);
    if (!plan) {
        clear_transition(chunk, transaction);
        return plan.error();
    }
    const auto destination_bound = backend_.allocation_bound(destination, chunk.logical_size_);
    if (!destination_bound) {
        clear_transition(chunk, transaction);
        return destination_bound.error();
    }
    ByteSize workspace_bound{};
    if (plan.value().requires_workspace) {
        const auto bound = backend_.workspace_bound(source.state, destination, chunk.logical_size_);
        if (!bound) {
            clear_transition(chunk, transaction);
            return bound.error();
        }
        workspace_bound = bound.value();
    }
    // A verified acceptance ceiling can bound committed growth independently of temporary
    // worst-case storage. Actual destination admission still reserves the full backend bound.
    const auto committed_bound =
        constraints != nullptr && constraints->maximum_destination_charge &&
                *constraints->maximum_destination_charge < destination_bound.value()
            ? *constraints->maximum_destination_charge
            : destination_bound.value();
    const auto projection = ledger_.validate_migration_projection(
        plan.value().destination_tier, committed_bound, tier_of(source.state), source.charge);
    if (!projection) {
        clear_transition(chunk, transaction);
        return projection.error();
    }

    const auto compaction_bound = backend_.compaction_bound(destination, chunk.logical_size_);
    if (!compaction_bound) {
        clear_transition(chunk, transaction);
        return compaction_bound.error();
    }
    const bool needs_compaction = compaction_bound.value() != ByteSize{};
    std::array<ReservationRequest, 3U> requests{};
    std::array<ReservationToken, 3U> tokens{};
    requests[0] = ReservationRequest{plan.value().destination_tier, ChargeBucket::staging,
                                     destination_bound.value(), AdmissionKind::migration_temporary};
    std::size_t request_count = 1U;
    if (plan.value().requires_workspace) {
        requests[1] = ReservationRequest{plan.value().workspace_tier, ChargeBucket::workspace,
                                         workspace_bound, AdmissionKind::migration_temporary};
        request_count = 2U;
    }
    if (needs_compaction && request_count != 2U) {
        detail::allocation_contract_failure(async_errors_, {});
    }
    requests[2] = ReservationRequest{plan.value().destination_tier, ChargeBucket::workspace,
                                     compaction_bound.value(), AdmissionKind::migration_temporary};
    const auto total_requests = request_count + (needs_compaction ? 1U : 0U);
    const auto reserved = ledger_.reserve(
        transaction, std::span<const ReservationRequest>{requests.data(), total_requests},
        std::span<ReservationToken>{tokens.data(), total_requests});
    if (!reserved) {
        clear_transition(chunk, transaction);
        return reserved.error();
    }
    CompactionReservation compaction_reservation{ledger_, tokens[2], needs_compaction};
    notify(observer, TransactionPhase::reserved, chunk);

    const auto storage_granularity = backend_.capabilities().allocation_granularity;
    const auto provisional_allocation =
        backend_.allocate_representation(RepresentationAllocationRequest{
            destination, chunk.logical_size_, source.content,
            destination == RepresentationState::gpu_raw ? chunk.stable_address_ : DeviceAddress{}});
    if (!provisional_allocation) {
        for (std::size_t index = 0U; index < request_count; ++index) {
            static_cast<void>(ledger_.release_reservation(tokens[index]));
        }
        clear_transition(chunk, transaction);
        notify(observer, TransactionPhase::rolled_back, chunk);
        return provisional_allocation.error();
    }
    const auto allocation = detail::corroborate_allocation(
        backend_, async_errors_, provisional_allocation.value(),
        detail::AllocationExpectation{
            plan.value().destination_tier, ResourceKind::representation, destination_bound.value(),
            storage_granularity, chunk.logical_size_, destination,
            destination == RepresentationState::gpu_raw ? chunk.stable_address_ : DeviceAddress{}},
        source.resource);
    const auto materialized = ledger_.materialize(tokens[0], allocation.charge);
    if (!materialized) {
        if (!ledger_.record_contract_debt(tokens[0], allocation.charge) ||
            !remember_cleanup(chunk,
                              CleanupResource{allocation.id, allocation.tier, allocation.charge},
                              transaction)) {
            detail::allocation_contract_failure(async_errors_, allocation);
        }
        if (request_count == 2U) {
            static_cast<void>(ledger_.release_reservation(tokens[1]));
        }
        poison(chunk, transaction, materialized.error());
        return materialized.error();
    }
    static_cast<void>(ledger_.release_reservation(tokens[0]));
    notify(observer, TransactionPhase::destination_materialized, chunk);

    BackendAllocation workspace{};
    bool has_workspace = false;
    if (request_count == 2U) {
        const auto workspace_allocation =
            backend_.allocate_workspace(plan.value().workspace_tier, workspace_bound);
        if (!workspace_allocation) {
            static_cast<void>(ledger_.release_reservation(tokens[1]));
            rollback_resource(allocation, ChargeBucket::staging, workspace_allocation.error(),
                              chunk, transaction, false);
            clear_transition(chunk, transaction);
            notify(observer, TransactionPhase::rolled_back, chunk);
            return workspace_allocation.error();
        }
        workspace = detail::corroborate_allocation(
            backend_, async_errors_, workspace_allocation.value(),
            detail::AllocationExpectation{plan.value().workspace_tier,
                                          ResourceKind::workspace,
                                          workspace_bound,
                                          storage_granularity,
                                          {},
                                          {}},
            source.resource, allocation.id);
        has_workspace = true;
        const auto workspace_materialized = ledger_.materialize(tokens[1], workspace.charge);
        if (!workspace_materialized) {
            if (!ledger_.record_contract_debt(tokens[1], workspace.charge) ||
                !remember_cleanup(chunk,
                                  CleanupResource{workspace.id, workspace.tier, workspace.charge},
                                  transaction)) {
                detail::allocation_contract_failure(async_errors_, workspace);
            }
            rollback_resource(allocation, ChargeBucket::staging, workspace_materialized.error(),
                              chunk, transaction, true);
            poison(chunk, transaction, workspace_materialized.error());
            return workspace_materialized.error();
        }
        static_cast<void>(ledger_.release_reservation(tokens[1]));
        notify(observer, TransactionPhase::workspace_materialized, chunk);
    }

    const ResourceId workspace_id = has_workspace ? workspace.id : ResourceId{};
    const auto source_verified =
        backend_.verify_authoritative(source.resource, source.content, workspace_id);
    if (!source_verified) {
        if (has_workspace) {
            rollback_resource(workspace, ChargeBucket::workspace, source_verified.error(), chunk,
                              transaction, false);
        }
        rollback_resource(allocation, ChargeBucket::staging, source_verified.error(), chunk,
                          transaction, false);
        if (source_verified.error().code == ErrorCode::integrity_failure ||
            source_verified.error().code == ErrorCode::backend_contract_violation) {
            poison(chunk, transaction, source_verified.error());
            notify(observer, TransactionPhase::poisoned, chunk);
        } else {
            clear_transition(chunk, transaction);
            notify(observer, TransactionPhase::rolled_back, chunk);
        }
        return source_verified.error();
    }

    const auto rollback_prepared = [&](Error error) noexcept -> Result<void> {
        compaction_reservation.release();
        if (has_workspace) {
            rollback_resource(workspace, ChargeBucket::workspace, error, chunk, transaction, false);
        }
        rollback_resource(allocation, ChargeBucket::staging, error, chunk, transaction, false);
        clear_transition(chunk, transaction);
        notify(observer, TransactionPhase::rolled_back, chunk);
        return error;
    };
    const auto prepared = backend_.prepare_transfer(source.resource, allocation.id, workspace_id);
    if (!prepared) {
        if (prepared.error().code == ErrorCode::ambiguous_backend_state) {
            detail::allocation_contract_failure(async_errors_, allocation);
        }
        return rollback_prepared(prepared.error());
    }
    BackendAllocation compaction{};
    if (needs_compaction) {
        if (prepared.value() == ByteSize{} || prepared.value() > compaction_bound.value() ||
            prepared.value() > allocation.charge ||
            prepared.value().value() % storage_granularity.value() != 0U) {
            detail::allocation_contract_failure(async_errors_, allocation);
        }
        const auto provisional = backend_.allocate_workspace(allocation.tier, prepared.value());
        if (!provisional) {
            return rollback_prepared(provisional.error());
        }
        if (provisional.value().id == workspace_id) {
            detail::allocation_contract_failure(async_errors_, provisional.value());
        }
        compaction =
            detail::corroborate_allocation(backend_, async_errors_, provisional.value(),
                                           detail::AllocationExpectation{allocation.tier,
                                                                         ResourceKind::workspace,
                                                                         prepared.value(),
                                                                         storage_granularity,
                                                                         {},
                                                                         {}},
                                           source.resource, allocation.id);
        // The exchange contract needs the exact prepared padded capacity, not a smaller target.
        if (compaction.charge != prepared.value()) {
            detail::allocation_contract_failure(async_errors_, compaction);
        }
        const auto materialized_compaction = ledger_.materialize(tokens[2], compaction.charge);
        if (!materialized_compaction) {
            if (!ledger_.record_contract_debt(tokens[2], compaction.charge) ||
                !remember_cleanup(chunk, {compaction.id, compaction.tier, compaction.charge},
                                  transaction)) {
                detail::allocation_contract_failure(async_errors_, compaction);
            }
            // record_contract_debt consumes the complete token, including its remainder.
            compaction_reservation.consumed();
            rollback_resource(workspace, ChargeBucket::workspace, materialized_compaction.error(),
                              chunk, transaction, false);
            rollback_resource(allocation, ChargeBucket::staging, materialized_compaction.error(),
                              chunk, transaction, true);
            poison(chunk, transaction, materialized_compaction.error());
            return materialized_compaction.error();
        }
        compaction_reservation.release();
        notify(observer, TransactionPhase::workspace_materialized, chunk);
    } else if (prepared.value() != ByteSize{}) {
        detail::allocation_contract_failure(async_errors_, allocation);
    }
    const auto allocation_granularity = backend_.capabilities().allocation_granularity;
    const auto transferred = backend_.transfer(source.resource, allocation.id, workspace_id,
                                               CompactionTarget{compaction.id});
    if (!transferred) {
        if (needs_compaction) {
            rollback_resource(compaction, ChargeBucket::workspace, transferred.error(), chunk,
                              transaction, false);
        }
        if (has_workspace) {
            rollback_resource(workspace, ChargeBucket::workspace, transferred.error(), chunk,
                              transaction, false);
        }
        rollback_resource(allocation, ChargeBucket::staging, transferred.error(), chunk,
                          transaction,
                          transferred.error().code == ErrorCode::ambiguous_backend_state);
        if (transferred.error().code == ErrorCode::ambiguous_backend_state) {
            poison(chunk, transaction, transferred.error());
            notify(observer, TransactionPhase::poisoned, chunk);
        } else {
            clear_transition(chunk, transaction);
            notify(observer, TransactionPhase::rolled_back, chunk);
        }
        return transferred.error();
    }
    const auto receipt = transferred.value();
    // An invalid accounting oracle cannot be recovered by guessing a charge or reporting poison
    // with a fictitious ledger. This is a fatal internal contract defect, not an operation error.
    if (receipt.charge == ByteSize{} || receipt.charge > allocation.charge ||
        allocation_granularity == ByteSize{} ||
        receipt.charge.value() % allocation_granularity.value() != 0U) {
        async_errors_.push(make_error(ErrorCode::backend_contract_violation, OperationId::migrate,
                                      allocation.id.value(), receipt.charge.value()));
        std::terminate();
    }
    const bool invalid_metadata =
        receipt.metadata.logical_size != chunk.logical_size_ ||
        receipt.metadata.stored_size == ByteSize{} ||
        receipt.metadata.stored_size > receipt.charge ||
        receipt.metadata.crc32c != source.metadata.crc32c ||
        (is_raw(destination) && (receipt.metadata.encoding != Encoding::raw ||
                                 receipt.metadata.stored_size != chunk.logical_size_ ||
                                 receipt.metadata.stored_crc32c != receipt.metadata.crc32c)) ||
        (!is_raw(destination) && receipt.metadata.encoding != Encoding::lz4_block);
    // Provisional reconciliation is not permission to recover using an uncorroborated charge.
    // No backend call or observer may intervene before this update in the original tier.
    if (needs_compaction && receipt.charge != compaction.charge) {
        detail::allocation_contract_failure(async_errors_, allocation);
    }
    if (receipt.charge != allocation.charge || needs_compaction) {
        const auto shrunk =
            needs_compaction
                ? ledger_.exchange_compaction(allocation.tier, allocation.charge, receipt.charge)
                : ledger_.shrink_materialized(allocation.tier, ChargeBucket::staging,
                                              allocation.charge, receipt.charge);
        if (!shrunk) {
            // Preconditions were checked and this transaction owns the staging charge.
            // Ledger corruption cannot be disguised as a recoverable rollback.
            async_errors_.push(shrunk.error());
            std::terminate();
        }
    }
    if (needs_compaction) {
        compaction.charge = allocation.charge;
        const auto corroborated_compaction = backend_.allocation(compaction.id);
        if (!corroborated_compaction || corroborated_compaction.value().id != compaction.id ||
            corroborated_compaction.value().tier != compaction.tier ||
            corroborated_compaction.value().kind != compaction.kind ||
            corroborated_compaction.value().address != compaction.address ||
            corroborated_compaction.value().charge != compaction.charge) {
            detail::allocation_contract_failure(async_errors_, compaction);
        }
    }
    // Corroboration is mandatory even when local receipt metadata is invalid. Until it succeeds,
    // every contract anomaly is fail-stop: neither cleanup nor a normal observer may run.
    const auto checked_final = backend_.allocation(allocation.id);
    if (!checked_final) {
        async_errors_.push(make_error(ErrorCode::backend_contract_violation, OperationId::migrate,
                                      allocation.id.value(),
                                      static_cast<std::uint64_t>(checked_final.error().code)));
        std::terminate();
    }
    const auto queried = checked_final.value();
    if (queried.id != allocation.id || queried.tier != allocation.tier ||
        queried.kind != allocation.kind || queried.address != allocation.address) {
        // Original identity is trusted, but a malformed query cannot corroborate mutable charge.
        async_errors_.push(make_error(ErrorCode::backend_contract_violation, OperationId::migrate,
                                      allocation.id.value()));
        std::terminate();
    }
    if (queried.charge != receipt.charge) {
        // The same resource now has contradictory exact charge oracles. Neither value can be
        // chosen for rollback/debt. Record a fixed-size error and stop without cleanup or
        // callbacks.
        async_errors_.push(make_error(ErrorCode::backend_contract_violation, OperationId::migrate,
                                      allocation.id.value(), queried.charge.value()));
        std::terminate();
    }
    // Exact-charge corroboration barrier: all recoverable post-transfer paths are below here.
    // Stable accounting identity still comes only from the adopted destination.
    const BackendAllocation final_allocation{allocation.id,   allocation.tier,  receipt.charge,
                                             allocation.kind, receipt.metadata, allocation.address};
    notify(observer, TransactionPhase::charge_reconciled, chunk);
    if (needs_compaction) {
        const auto released = backend_.release(compaction.id, ReleasePhase::rollback);
        if (!released || backend_.owns(compaction.id)) {
            const auto error = released ? make_error(ErrorCode::backend_contract_violation,
                                                     OperationId::release, compaction.id.value())
                                        : released.error();
            if (!ledger_.move_materialized_to_debt(compaction.tier, ChargeBucket::workspace,
                                                   compaction.charge) ||
                !remember_cleanup(chunk, {compaction.id, compaction.tier, compaction.charge},
                                  transaction)) {
                detail::allocation_contract_failure(async_errors_, compaction);
            }
            rollback_resource(workspace, ChargeBucket::workspace, error, chunk, transaction, false);
            rollback_resource(final_allocation, ChargeBucket::staging, error, chunk, transaction,
                              released.has_value());
            if (released) {
                poison(chunk, transaction, error);
            } else {
                clear_transition(chunk, transaction);
            }
            notify(observer, released ? TransactionPhase::poisoned : TransactionPhase::rolled_back,
                   chunk);
            return error;
        }
        if (!ledger_.release_materialized(compaction.tier, ChargeBucket::workspace,
                                          compaction.charge)) {
            detail::allocation_contract_failure(async_errors_, compaction);
        }
    }
    const auto rollback_contract = [&](Error contract) noexcept -> Result<void> {
        if (has_workspace) {
            rollback_resource(workspace, ChargeBucket::workspace, contract, chunk, transaction,
                              false);
        }
        rollback_resource(final_allocation, ChargeBucket::staging, contract, chunk, transaction,
                          true);
        poison(chunk, transaction, contract);
        notify(observer, TransactionPhase::poisoned, chunk);
        return contract;
    };
    if (invalid_metadata ||
        queried.metadata.logical_size != final_allocation.metadata.logical_size ||
        queried.metadata.stored_size != final_allocation.metadata.stored_size ||
        queried.metadata.encoding != final_allocation.metadata.encoding ||
        queried.metadata.crc32c != final_allocation.metadata.crc32c ||
        queried.metadata.stored_crc32c != final_allocation.metadata.stored_crc32c) {
        // Metadata disagreement cannot change the now unambiguous physical charge.
        return rollback_contract(make_error(ErrorCode::backend_contract_violation,
                                            OperationId::allocate, final_allocation.id.value(),
                                            final_allocation.metadata.stored_size.value()));
    }
    notify(observer, TransactionPhase::transferred, chunk);

    const auto verified = backend_.verify_transfer(source.resource, final_allocation.id,
                                                   source.content, workspace_id);
    if (!verified) {
        if (has_workspace) {
            rollback_resource(workspace, ChargeBucket::workspace, verified.error(), chunk,
                              transaction, false);
        }
        rollback_resource(final_allocation, ChargeBucket::staging, verified.error(), chunk,
                          transaction, false);
        if (verified.error().code == ErrorCode::backend_contract_violation ||
            verified.error().code == ErrorCode::ambiguous_backend_state) {
            poison(chunk, transaction, verified.error());
            notify(observer, TransactionPhase::poisoned, chunk);
        } else {
            clear_transition(chunk, transaction);
            notify(observer, TransactionPhase::rolled_back, chunk);
        }
        return verified.error();
    }
    notify(observer, TransactionPhase::verified, chunk);

    if (constraints != nullptr && constraints->maximum_destination_charge &&
        final_allocation.charge > *constraints->maximum_destination_charge) {
        const auto rejected =
            make_error(ErrorCode::compression_not_beneficial, OperationId::compress,
                       chunk.id_.value(), final_allocation.charge.value());
        if (has_workspace) {
            rollback_resource(workspace, ChargeBucket::workspace, rejected, chunk, transaction,
                              false);
        }
        rollback_resource(final_allocation, ChargeBucket::staging, rejected, chunk, transaction,
                          false);
        clear_transition(chunk, transaction);
        notify(observer,
               chunk.snapshot().lifecycle == LifecycleState::poisoned
                   ? TransactionPhase::poisoned
                   : TransactionPhase::rolled_back,
               chunk);
        return rejected;
    }

    if (has_workspace) {
        const auto released = backend_.release(workspace.id, ReleasePhase::rollback);
        if (!released || backend_.owns(workspace.id)) {
            const Error cleanup_error = released
                                            ? make_error(ErrorCode::backend_contract_violation,
                                                         OperationId::release, workspace.id.value())
                                            : released.error();
            static_cast<void>(ledger_.move_materialized_to_debt(
                workspace.tier, ChargeBucket::workspace, workspace.charge));
            static_cast<void>(remember_cleanup(
                chunk, CleanupResource{workspace.id, workspace.tier, workspace.charge},
                transaction));
            rollback_resource(final_allocation, ChargeBucket::staging, cleanup_error, chunk,
                              transaction, released.has_value());
            if (released) {
                poison(chunk, transaction, cleanup_error);
            } else {
                clear_transition(chunk, transaction);
            }
            notify(observer, released ? TransactionPhase::poisoned : TransactionPhase::rolled_back,
                   chunk);
            return cleanup_error;
        }
        const auto removed =
            ledger_.release_materialized(workspace.tier, ChargeBucket::workspace, workspace.charge);
        if (!removed) {
            poison(chunk, transaction, removed.error());
            return removed.error();
        }
    }
    notify(observer, TransactionPhase::commit_ready, chunk);

    Error commit_error{};
    bool commit_failed = false;
    {
        const std::scoped_lock lock{chunk.mutex_};
        if (!chunk.transition_active_ || chunk.transaction_ != transaction ||
            (chunk.lifecycle_ != LifecycleState::live &&
             chunk.lifecycle_ != LifecycleState::closing) ||
            chunk.authoritative_.resource != source.resource ||
            chunk.authoritative_.content != source.content) {
            commit_error = make_error(ErrorCode::internal_invariant_violation, OperationId::migrate,
                                      chunk.id_.value());
            commit_failed = true;
        } else {
            const auto committed = ledger_.commit_destination_and_retire_source(
                final_allocation.tier, final_allocation.charge, tier_of(source.state),
                source.charge);
            if (!committed) {
                commit_error = committed.error();
                commit_failed = true;
            } else {
                chunk.cleanup_resources_[0] =
                    CleanupResource{source.resource, tier_of(source.state), source.charge};
                chunk.cleanup_resource_count_ = 1U;
                chunk.authoritative_ = Representation{destination,
                                                      final_allocation.id,
                                                      final_allocation.charge,
                                                      source.content,
                                                      destination == RepresentationState::gpu_raw
                                                          ? chunk.stable_address_
                                                          : DeviceAddress{},
                                                      final_allocation.metadata};
                ++chunk.transition_epoch_;
            }
        }
    }
    if (commit_failed) {
        rollback_resource(final_allocation, ChargeBucket::staging, commit_error, chunk, transaction,
                          true);
        poison(chunk, transaction, commit_error);
        notify(observer, TransactionPhase::poisoned, chunk);
        return commit_error;
    }
    notify(observer, TransactionPhase::committed, chunk);

    const auto released = backend_.release(source.resource, ReleasePhase::post_commit);
    if (!released || backend_.owns(source.resource)) {
        const Error cleanup_error = released
                                        ? make_error(ErrorCode::backend_contract_violation,
                                                     OperationId::release, source.resource.value())
                                        : released.error();
        async_errors_.push(cleanup_error);
        if (released) {
            poison(chunk, transaction, cleanup_error);
        } else {
            clear_transition(chunk, transaction);
        }
        return cleanup_error;
    }
    const auto removed = ledger_.release_cleanup_debt(tier_of(source.state), source.charge);
    if (!removed) {
        poison(chunk, transaction, removed.error());
        return removed.error();
    }
    forget_cleanup(chunk, source.resource);
    clear_transition(chunk, transaction);
    notify(observer, TransactionPhase::cleanup_complete, chunk);
    return {};
}

} // namespace vramz
