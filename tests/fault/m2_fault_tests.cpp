#include "../test_support.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>
#include <utility>

using namespace vramz;

namespace {

struct MigrationCase final {
    RepresentationState source;
    RepresentationState destination;
};

static_assert(std::is_trivially_copyable_v<BackendAllocation>);
static_assert(std::is_nothrow_copy_constructible_v<BackendAllocation>);
static_assert(std::is_nothrow_constructible_v<Result<BackendAllocation>, BackendAllocation>);
template <class T>
concept SuppliesIdentity = requires(T value) { value.id; };
template <class T>
concept SuppliesTier = requires(T value) { value.tier; };
template <class T>
concept SuppliesKind = requires(T value) { value.kind; };
static_assert(!SuppliesIdentity<TransferReceipt>);
static_assert(!SuppliesTier<TransferReceipt>);
static_assert(!SuppliesKind<TransferReceipt>);
static_assert(std::is_aggregate_v<TransferReceipt>);
static_assert(std::is_standard_layout_v<TransferReceipt>);
static_assert(std::is_trivially_copyable_v<TransferReceipt>);
static_assert(std::is_nothrow_default_constructible_v<TransferReceipt>);
static_assert(std::is_nothrow_copy_constructible_v<TransferReceipt>);
static_assert(std::is_trivially_copyable_v<Result<TransferReceipt>>);
static_assert(std::is_nothrow_constructible_v<Result<TransferReceipt>, TransferReceipt>);
static_assert(std::is_same_v<decltype(std::declval<StorageBackend&>().transfer(
                                 ResourceId{}, ResourceId{}, ResourceId{})),
                             Result<TransferReceipt>>);

void check_transfer_receipt(test::Runner& runner, MigrationCase migration,
                            FaultPoint fault = FaultPoint::count) {
    // Exercise the backend contract directly, without coordinator reconciliation or discovery.
    test::CoreFixture fixture{migration.source};
    VRAMZ_CHECK(runner, fixture.ready());
    const auto source = fixture.chunk->snapshot().authoritative;
    const auto provisional_destination =
        fixture.backend
            .allocate_representation(RepresentationAllocationRequest{
                migration.destination, fixture.logical_size, source.content,
                migration.destination == RepresentationState::gpu_raw ? fixture.address_space.base
                                                                      : DeviceAddress{}})
            .value();
    const auto adopted_destination = fixture.backend.adopt_allocation(provisional_destination);
    VRAMZ_CHECK(runner, adopted_destination);
    const auto destination = adopted_destination.value();
    const auto workspace_bound =
        fixture.backend
            .workspace_bound(migration.source, migration.destination, fixture.logical_size)
            .value();
    BackendAllocation workspace{};
    if (workspace_bound != ByteSize{}) {
        workspace =
            fixture.backend.allocate_workspace(tier_of(migration.destination), workspace_bound)
                .value();
        const auto adopted_workspace = fixture.backend.adopt_allocation(workspace);
        VRAMZ_CHECK(runner, adopted_workspace);
        workspace = adopted_workspace.value();
    }
    if (fault != FaultPoint::count) {
        fixture.backend.inject_failure(fault);
    }
    // Even an immediately failing discovery query cannot affect transfer's accounting receipt.
    fixture.backend.inject_failure(FaultPoint::stored_size_contract);
    const auto transferred =
        fixture.backend.transfer(source.resource, destination.id, workspace.id);
    VRAMZ_CHECK(runner, transferred.has_value() == (fault == FaultPoint::count));
    const auto actual =
        transferred
            ? BackendAllocation{destination.id, destination.tier, transferred.value().charge,
                                destination.kind, transferred.value().metadata}
            : destination;
    VRAMZ_CHECK(runner, actual.id == destination.id && actual.tier == destination.tier &&
                            actual.kind == destination.kind);
    if (transferred) {
        VRAMZ_CHECK(runner, actual.charge ==
                                (is_raw(migration.destination) ? ByteSize{256U} : ByteSize{128U}));
        VRAMZ_CHECK(runner, actual.charge <= destination.charge);
    }
    for (const auto tier : {PhysicalTier::gpu, PhysicalTier::host}) {
        const auto source_charge = tier_of(migration.source) == tier ? source.charge.value() : 0U;
        const auto destination_charge = actual.tier == tier ? actual.charge.value() : 0U;
        const auto workspace_charge = workspace.tier == tier ? workspace.charge.value() : 0U;
        VRAMZ_CHECK(runner, fixture.backend.owned_charge(tier).value() ==
                                source_charge + destination_charge + workspace_charge);
    }
    VRAMZ_CHECK(runner, !fixture.backend.allocation(destination.id));
    const auto discovered = fixture.backend.allocation(destination.id);
    VRAMZ_CHECK(runner, discovered && discovered.value().charge == actual.charge);
    VRAMZ_CHECK(runner,
                fixture.backend.allocation(source.resource).value().charge == source.charge);
    if (workspace.id != ResourceId{}) {
        VRAMZ_CHECK(runner,
                    fixture.backend.allocation(workspace.id).value().charge == workspace.charge);
        VRAMZ_CHECK(runner, fixture.backend.release(workspace.id, ReleasePhase::rollback));
    }
    VRAMZ_CHECK(runner, fixture.backend.release(actual.id, ReleasePhase::rollback));
    VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::gpu));
    VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::host));
}

void expect_compound_rollback(test::Runner& runner, FaultPoint point, bool include_raw = false,
                              bool cleanup_fails = true) {
    for (const auto initial : test::all_states) {
        for (const auto destination : test::all_states) {
            if (initial == destination || (!include_raw && is_raw(destination))) {
                continue;
            }
            test::CoreFixture fixture{initial};
            VRAMZ_CHECK(runner, fixture.ready());
            const auto before = fixture.chunk->snapshot();
            const auto source = before.authoritative;
            const auto tier = tier_of(destination);
            test::CoreAccountingObserver observer{fixture};
            const bool query_fault = point == FaultPoint::allocation_logical_size_mismatch ||
                                     point == FaultPoint::allocation_stored_size_mismatch ||
                                     point == FaultPoint::allocation_encoding_mismatch ||
                                     point == FaultPoint::allocation_crc_mismatch;
            fixture.backend.inject_failure(point, query_fault ? 2U : 1U);
            if (cleanup_fails) {
                // A workspace, if needed, is released before the destination.
                const bool has_workspace = !is_raw(initial) || !is_raw(destination);
                fixture.backend.inject_failure(FaultPoint::rollback_cleanup,
                                               has_workspace ? 2U : 1U);
            }
            const auto migrated =
                fixture.coordinator.migrate(*fixture.chunk, destination, false, &observer);
            const bool contract = point != FaultPoint::verification;
            VRAMZ_CHECK(runner, !migrated && migrated.error().code ==
                                                 (contract ? ErrorCode::backend_contract_violation
                                                           : ErrorCode::integrity_failure));
            const auto after = fixture.chunk->snapshot();
            VRAMZ_CHECK(runner, after.lifecycle ==
                                    (contract ? LifecycleState::poisoned : LifecycleState::live));
            VRAMZ_CHECK(runner, after.authoritative.resource == source.resource);
            VRAMZ_CHECK(runner, after.authoritative.charge == source.charge);
            VRAMZ_CHECK(runner, after.authoritative.content == source.content);
            VRAMZ_CHECK(runner, after.authoritative.metadata.crc32c == source.metadata.crc32c);
            VRAMZ_CHECK(runner, after.transition_epoch == before.transition_epoch);
            VRAMZ_CHECK(runner, !after.transition_active &&
                                    after.cleanup_resource_count == (cleanup_fails ? 1U : 0U));
            VRAMZ_CHECK(runner,
                        fixture.backend.owned_resource_count() == (cleanup_fails ? 2U : 1U));
            const auto original_destination = fixture.backend.last_transfer_destination();
            const ByteSize actual_charge = is_raw(destination) ? ByteSize{256U} : ByteSize{128U};
            VRAMZ_CHECK(runner, original_destination != ResourceId{} &&
                                    original_destination != source.resource);
            VRAMZ_CHECK(runner, fixture.backend.last_rollback_resource() == original_destination);
            VRAMZ_CHECK(runner, fixture.backend.owns(original_destination) == cleanup_fails);
            if (cleanup_fails) {
                const auto cleanup = testing::RuntimeAccess::cleanup_resource(*fixture.chunk, 0U);
                VRAMZ_CHECK(runner, cleanup && cleanup.value().resource == original_destination);
                VRAMZ_CHECK(runner, cleanup && cleanup.value().tier == tier);
                VRAMZ_CHECK(runner, cleanup && cleanup.value().charge == actual_charge);
                if (cleanup) {
                    const auto retained = fixture.backend.allocation(original_destination);
                    VRAMZ_CHECK(runner, retained && retained.value().id == original_destination &&
                                            retained.value().tier == tier &&
                                            retained.value().kind == ResourceKind::representation &&
                                            retained.value().charge == cleanup.value().charge);
                }
            }
            VRAMZ_CHECK(runner, observer.saw(TransactionPhase::charge_reconciled));
            VRAMZ_CHECK(runner,
                        observer.at(TransactionPhase::destination_materialized, tier).staging ==
                            (is_raw(destination) ? ByteSize{256U} : ByteSize{320U}));
            VRAMZ_CHECK(runner, observer.at(TransactionPhase::charge_reconciled, tier).staging ==
                                    actual_charge);
            VRAMZ_CHECK(runner, observer.saw(contract ? TransactionPhase::poisoned
                                                      : TransactionPhase::rolled_back));
            VRAMZ_CHECK(runner, !observer.saw(TransactionPhase::committed));
            VRAMZ_CHECK(runner, observer.conserved());
            for (const auto checked_tier : {PhysicalTier::gpu, PhysicalTier::host}) {
                const auto usage = fixture.ledger.usage(checked_tier);
                VRAMZ_CHECK(runner, usage.reserved == ByteSize{} && usage.staging == ByteSize{} &&
                                        usage.workspace == ByteSize{});
                VRAMZ_CHECK(runner, usage.cleanup_debt == (checked_tier == tier && cleanup_fails
                                                               ? actual_charge
                                                               : ByteSize{}));
                VRAMZ_CHECK(runner, fixture.conserved(checked_tier));
            }
            if (is_raw(initial)) {
                std::array<std::byte, 256U> bytes{};
                bytes.fill(std::byte{0xFFU});
                VRAMZ_CHECK(runner,
                            fixture.backend.read_bytes(source.resource, ByteOffset{}, bytes));
                VRAMZ_CHECK(runner, std::all_of(bytes.begin(), bytes.end(), [](std::byte byte) {
                                return byte == std::byte{};
                            }));
            }
            // Poison or outstanding debt must prevent a new transition from hiding the failure.
            VRAMZ_CHECK(runner, !fixture.coordinator.migrate(*fixture.chunk, destination));
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::gpu));
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::host));
        }
    }
}

void expect_rollback(test::Runner& runner, MigrationCase migration, FaultPoint point) {
    test::CoreFixture fixture{migration.source};
    VRAMZ_CHECK(runner, fixture.ready());
    const auto before = fixture.chunk->snapshot();
    const auto owned_before = fixture.backend.owned_resource_count();
    fixture.backend.inject_failure(point);
    const auto result = fixture.coordinator.migrate(*fixture.chunk, migration.destination);
    const auto after = fixture.chunk->snapshot();
    VRAMZ_CHECK(runner, !result);
    VRAMZ_CHECK(runner, after.lifecycle == LifecycleState::live);
    VRAMZ_CHECK(runner, after.authoritative.resource == before.authoritative.resource);
    VRAMZ_CHECK(runner, after.authoritative.content == before.authoritative.content);
    VRAMZ_CHECK(runner,
                after.authoritative.metadata.crc32c == before.authoritative.metadata.crc32c);
    VRAMZ_CHECK(runner, fixture.backend.owned_resource_count() == owned_before);
    VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::gpu));
    VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::host));
}

void expect_fatal_charge_contract(test::Runner& runner, FaultPoint point) {
    for (const auto destination :
         {RepresentationState::gpu_compressed, RepresentationState::host_compressed}) {
        const auto child = ::fork();
        VRAMZ_CHECK(runner, child >= 0);
        if (child < 0) {
            return;
        }
        if (child == 0) {
            // An isolated death test; no intentionally broken accounting oracle runs in parent.
            static_cast<void>(::alarm(5U));
            test::CoreFixture fixture{RepresentationState::gpu_raw};
            if (!fixture.ready()) {
                std::_Exit(1);
            }
            static test::CoreFixture* child_fixture{};
            static PhysicalTier destination_tier{};
            child_fixture = &fixture;
            destination_tier = tier_of(destination);
            std::set_terminate([]() noexcept {
                const auto usage = child_fixture->ledger.usage(destination_tier);
                // Detection must precede ledger mutation, cleanup, or an observable poisoned state.
                const bool untouched =
                    usage.staging == ByteSize{320U} && usage.workspace == ByteSize{320U} &&
                    usage.cleanup_debt == ByteSize{} &&
                    child_fixture->chunk->snapshot().lifecycle == LifecycleState::live &&
                    child_fixture->backend.owned_resource_count() == 3U &&
                    !child_fixture->backend.allocation(
                        child_fixture->backend.last_transfer_destination());
                std::_Exit(untouched ? 86 : 1);
            });
            fixture.backend.inject_failure(point);
            // Local impossible-charge checks must leave the corroboration query unreached.
            fixture.backend.inject_failure(FaultPoint::stored_size_contract, 2U);
            static_cast<void>(fixture.coordinator.migrate(*fixture.chunk, destination));
            std::_Exit(1);
        }
        int status = 0;
        pid_t waited = 0;
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        VRAMZ_CHECK(runner, waited == child);
        VRAMZ_CHECK(runner, WIFEXITED(status) && WEXITSTATUS(status) == 86);
    }
}

[[nodiscard]] constexpr bool query_fault(FaultPoint point) noexcept {
    return point == FaultPoint::stored_size_contract ||
           (point >= FaultPoint::allocation_descriptor_mismatch &&
            point <= FaultPoint::allocation_crc_mismatch);
}

void arm_post_transfer_fault(MockBackend& backend, FaultPoint point) noexcept {
    if (point != FaultPoint::count) {
        // Source inspection is the first query/CRC check; inject only at destination validation.
        backend.inject_failure(point,
                               query_fault(point) || point == FaultPoint::crc_comparison ? 2U : 1U);
    }
}

void expect_fatal_post_transfer(test::Runner& runner, MigrationCase migration, FaultPoint first,
                                FaultPoint second = FaultPoint::count, bool cleanup_fails = false) {
    const auto child = ::fork();
    VRAMZ_CHECK(runner, child >= 0);
    if (child < 0) {
        return;
    }
    if (child == 0) {
        static_cast<void>(::alarm(5U));
        test::CoreFixture fixture{migration.source};
        if (!fixture.ready()) {
            std::_Exit(1);
        }
        test::RecordingObserver observer{migration.source};
        static test::CoreFixture* child_fixture{};
        static test::RecordingObserver* child_observer{};
        static ChunkSnapshot before{};
        static ResourceId previous_cleanup{};
        static PhysicalTier destination_tier{};
        static ByteSize receipt_charge{};
        static ByteSize actual_charge{};
        static ByteSize workspace_charge{};
        static std::uint64_t expected_error_detail{};
        child_fixture = &fixture;
        child_observer = &observer;
        before = fixture.chunk->snapshot();
        previous_cleanup = fixture.backend.last_rollback_resource();
        destination_tier = tier_of(migration.destination);
        actual_charge = is_raw(migration.destination) ? ByteSize{256U} : ByteSize{128U};
        const bool false_receipt = first == FaultPoint::receipt_plausible_charge_mismatch ||
                                   second == FaultPoint::receipt_plausible_charge_mismatch;
        receipt_charge = false_receipt ? ByteSize{192U} : actual_charge;
        workspace_charge =
            is_raw(migration.source) && is_raw(migration.destination) ? ByteSize{} : ByteSize{320U};
        const auto wrong_identity = [](FaultPoint point) noexcept {
            return point == FaultPoint::allocation_identity_mismatch ||
                   point == FaultPoint::allocation_tier_mismatch ||
                   point == FaultPoint::allocation_kind_mismatch;
        };
        expected_error_detail = actual_charge.value();
        if (first == FaultPoint::stored_size_contract ||
            second == FaultPoint::stored_size_contract) {
            expected_error_detail =
                static_cast<std::uint64_t>(ErrorCode::backend_contract_violation);
        } else if (wrong_identity(first) || wrong_identity(second)) {
            expected_error_detail = 0U;
        } else if (first == FaultPoint::allocation_descriptor_mismatch ||
                   second == FaultPoint::allocation_descriptor_mismatch) {
            expected_error_detail =
                checked_add(actual_charge.value(), 64U, OperationId::migrate).value();
        }
        std::set_terminate([]() noexcept {
            const auto destination_id = child_fixture->backend.last_transfer_destination();
            // Clear any unreached second fault only for read-only inspection in the dying child.
            child_fixture->backend.clear_failures();
            const auto actual = child_fixture->backend.allocation(destination_id);
            const auto usage = child_fixture->ledger.usage(destination_tier);
            const auto snapshot = child_fixture->chunk->snapshot();
            const auto error = child_fixture->errors.first();
            const bool fatal_boundary =
                actual && actual.value().id == destination_id &&
                actual.value().tier == destination_tier &&
                actual.value().kind == ResourceKind::representation &&
                actual.value().charge == actual_charge && usage.staging == receipt_charge &&
                usage.workspace == workspace_charge && usage.reserved == ByteSize{} &&
                child_fixture->ledger.usage(PhysicalTier::gpu).cleanup_debt == ByteSize{} &&
                child_fixture->ledger.usage(PhysicalTier::host).cleanup_debt == ByteSize{} &&
                snapshot.lifecycle == LifecycleState::live && snapshot.transition_active &&
                snapshot.cleanup_resource_count == 0U &&
                snapshot.authoritative.resource == before.authoritative.resource &&
                snapshot.authoritative.state == before.authoritative.state &&
                snapshot.authoritative.charge == before.authoritative.charge &&
                snapshot.authoritative.content == before.authoritative.content &&
                snapshot.authoritative.metadata.crc32c == before.authoritative.metadata.crc32c &&
                snapshot.transition_epoch == before.transition_epoch &&
                child_fixture->backend.owned_resource_count() ==
                    (workspace_charge == ByteSize{} ? 2U : 3U) &&
                child_fixture->backend.last_rollback_resource() == previous_cleanup &&
                !child_observer->saw(TransactionPhase::charge_reconciled) &&
                !child_observer->saw(TransactionPhase::transferred) &&
                !child_observer->saw(TransactionPhase::verified) &&
                !child_observer->saw(TransactionPhase::rolled_back) &&
                !child_observer->saw(TransactionPhase::poisoned) &&
                !child_observer->saw(TransactionPhase::committed) && error &&
                error->error.code == ErrorCode::backend_contract_violation &&
                error->error.operation == OperationId::migrate &&
                error->error.object_id == destination_id.value() &&
                error->error.detail == expected_error_detail;
            // No conservation assertion: the uncorroborated charge is precisely why we stop.
            std::_Exit(fatal_boundary ? 86 : 1);
        });
        arm_post_transfer_fault(fixture.backend, first);
        arm_post_transfer_fault(fixture.backend, second);
        if (cleanup_fails) {
            fixture.backend.inject_failure(FaultPoint::rollback_workspace_cleanup);
            fixture.backend.inject_failure(FaultPoint::rollback_destination_cleanup);
        }
        static_cast<void>(
            fixture.coordinator.migrate(*fixture.chunk, migration.destination, false, &observer));
        std::_Exit(1);
    }
    int status = 0;
    pid_t waited = 0;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    VRAMZ_CHECK(runner, waited == child);
    VRAMZ_CHECK(runner, WIFEXITED(status) && WEXITSTATUS(status) == 86);
}

void expect_fatal_matrix(test::Runner& runner, FaultPoint first,
                         FaultPoint second = FaultPoint::count, bool all_pairs = false) {
    for (const auto source : test::all_states) {
        for (const auto destination : test::all_states) {
            if (source == destination ||
                (!all_pairs && (source != RepresentationState::gpu_raw || is_raw(destination)))) {
                continue;
            }
            for (const bool cleanup_fails : {false, true}) {
                expect_fatal_post_transfer(runner, MigrationCase{source, destination}, first,
                                           second, cleanup_fails);
            }
        }
    }
}

[[nodiscard]] RuntimeConfig fault_config() noexcept {
    auto config = test::test_runtime_config();
    config.budgets =
        MemoryBudgets{TierBudget{ByteSize{65536U}, ByteSize{49152U}, ByteSize{16384U}},
                      TierBudget{ByteSize{65536U}, ByteSize{49152U}, ByteSize{16384U}}};
    config.preferred_chunk_size = ByteSize{256U};
    return config;
}

[[nodiscard]] constexpr bool prevents_corroboration(FaultPoint point) noexcept {
    return point == FaultPoint::receipt_plausible_charge_mismatch ||
           point == FaultPoint::stored_size_contract ||
           point == FaultPoint::allocation_descriptor_mismatch ||
           point == FaultPoint::allocation_identity_mismatch ||
           point == FaultPoint::allocation_tier_mismatch ||
           point == FaultPoint::allocation_kind_mismatch;
}

[[nodiscard]] constexpr bool metadata_or_contract_fault(FaultPoint point) noexcept {
    return point == FaultPoint::receipt_logical_size_mismatch ||
           point == FaultPoint::receipt_stored_size_mismatch ||
           point == FaultPoint::receipt_encoding_mismatch ||
           (point >= FaultPoint::allocation_logical_size_mismatch &&
            point <= FaultPoint::allocation_crc_mismatch) ||
           point == FaultPoint::verification_contract;
}

void expect_recoverable_pair(test::Runner& runner, RepresentationState destination,
                             FaultPoint first, FaultPoint second) {
    test::CoreFixture fixture{RepresentationState::gpu_raw};
    VRAMZ_CHECK(runner, fixture.ready());
    const auto before = fixture.chunk->snapshot();
    test::CoreAccountingObserver observer{fixture};
    arm_post_transfer_fault(fixture.backend, first);
    arm_post_transfer_fault(fixture.backend, second);
    const auto migrated =
        fixture.coordinator.migrate(*fixture.chunk, destination, false, &observer);
    const auto after = fixture.chunk->snapshot();
    const bool poisoned = metadata_or_contract_fault(first) || metadata_or_contract_fault(second);
    VRAMZ_CHECK(runner, !migrated);
    VRAMZ_CHECK(runner,
                after.lifecycle == (poisoned ? LifecycleState::poisoned : LifecycleState::live));
    VRAMZ_CHECK(runner,
                after.authoritative.resource == before.authoritative.resource &&
                    after.authoritative.content == before.authoritative.content &&
                    after.authoritative.charge == before.authoritative.charge &&
                    after.authoritative.metadata.crc32c == before.authoritative.metadata.crc32c &&
                    after.transition_epoch == before.transition_epoch && !after.transition_active);
    VRAMZ_CHECK(runner, observer.saw(TransactionPhase::charge_reconciled));
    VRAMZ_CHECK(runner,
                observer.at(TransactionPhase::charge_reconciled, tier_of(destination)).staging ==
                    ByteSize{128U});
    VRAMZ_CHECK(runner, observer.saw(poisoned ? TransactionPhase::poisoned
                                              : TransactionPhase::rolled_back));
    VRAMZ_CHECK(runner, !observer.saw(TransactionPhase::committed) && observer.conserved());
    const auto expected_debts =
        static_cast<std::uint32_t>(first == FaultPoint::rollback_workspace_cleanup ||
                                   second == FaultPoint::rollback_workspace_cleanup) +
        static_cast<std::uint32_t>(first == FaultPoint::rollback_destination_cleanup ||
                                   second == FaultPoint::rollback_destination_cleanup);
    VRAMZ_CHECK(runner, after.cleanup_resource_count == expected_debts);
    VRAMZ_CHECK(runner, fixture.backend.owned_resource_count() == 1U + expected_debts);
    ByteSize debt{};
    ResourceId previous{};
    fixture.backend.clear_failures();
    for (std::uint32_t index = 0U; index < after.cleanup_resource_count; ++index) {
        const auto record = testing::RuntimeAccess::cleanup_resource(*fixture.chunk, index);
        VRAMZ_CHECK(runner, record);
        if (!record) {
            continue;
        }
        const auto actual = fixture.backend.allocation(record.value().resource);
        VRAMZ_CHECK(runner, actual && actual.value().id == record.value().resource &&
                                actual.value().tier == record.value().tier &&
                                actual.value().charge == record.value().charge);
        VRAMZ_CHECK(runner, record.value().resource != before.authoritative.resource &&
                                record.value().resource != previous &&
                                record.value().tier == tier_of(destination));
        const bool is_destination =
            record.value().resource == fixture.backend.last_transfer_destination();
        VRAMZ_CHECK(runner,
                    actual && actual.value().kind == (is_destination ? ResourceKind::representation
                                                                     : ResourceKind::workspace));
        VRAMZ_CHECK(runner,
                    record.value().charge == (is_destination ? ByteSize{128U} : ByteSize{320U}));
        const auto sum =
            checked_add(debt.value(), record.value().charge.value(), OperationId::budget_transfer);
        VRAMZ_CHECK(runner, sum);
        if (sum) {
            debt = ByteSize{sum.value()};
        }
        previous = record.value().resource;
    }
    for (const auto tier : {PhysicalTier::gpu, PhysicalTier::host}) {
        const auto usage = fixture.ledger.usage(tier);
        VRAMZ_CHECK(runner, fixture.conserved(tier));
        VRAMZ_CHECK(runner,
                    usage.reserved == ByteSize{} && usage.staging == ByteSize{} &&
                        usage.workspace == ByteSize{} &&
                        usage.cleanup_debt == (tier == tier_of(destination) ? debt : ByteSize{}));
    }
    std::array<std::byte, 256U> bytes{};
    bytes.fill(std::byte{0xFFU});
    VRAMZ_CHECK(runner,
                fixture.backend.read_bytes(before.authoritative.resource, ByteOffset{}, bytes));
    VRAMZ_CHECK(runner, std::all_of(bytes.begin(), bytes.end(),
                                    [](std::byte value) { return value == std::byte{}; }));
}

void test_post_transfer_pairs(test::Runner& runner) {
    // Only post-transfer/pre-commit fault classes, not unrelated global allocation failures.
    constexpr std::array faults{FaultPoint::receipt_plausible_charge_mismatch,
                                FaultPoint::receipt_logical_size_mismatch,
                                FaultPoint::receipt_stored_size_mismatch,
                                FaultPoint::receipt_encoding_mismatch,
                                FaultPoint::stored_size_contract,
                                FaultPoint::allocation_identity_mismatch,
                                FaultPoint::allocation_tier_mismatch,
                                FaultPoint::allocation_kind_mismatch,
                                FaultPoint::allocation_descriptor_mismatch,
                                FaultPoint::allocation_logical_size_mismatch,
                                FaultPoint::allocation_stored_size_mismatch,
                                FaultPoint::allocation_encoding_mismatch,
                                FaultPoint::allocation_crc_mismatch,
                                FaultPoint::verification,
                                FaultPoint::byte_comparison,
                                FaultPoint::crc_comparison,
                                FaultPoint::verification_contract,
                                FaultPoint::rollback_workspace_cleanup,
                                FaultPoint::rollback_destination_cleanup};
    std::uint32_t fatal_runs = 0U;
    std::uint32_t recoverable_runs = 0U;
    std::uint32_t excluded = 0U;
    for (std::size_t left = 0U; left < faults.size(); ++left) {
        for (std::size_t right = left + 1U; right < faults.size(); ++right) {
            // Two matching false oracles (both 192 for actual 128) cannot be detected by this
            // barrier. Explicitly exclude that colluding-backend contract breach, never call it
            // a conserved recovery. All other unordered pairs in this bounded model are tested.
            if (faults[left] == FaultPoint::receipt_plausible_charge_mismatch &&
                faults[right] == FaultPoint::allocation_descriptor_mismatch) {
                ++excluded;
                continue;
            }
            for (const auto destination :
                 {RepresentationState::gpu_compressed, RepresentationState::host_compressed}) {
                if (prevents_corroboration(faults[left]) || prevents_corroboration(faults[right])) {
                    expect_fatal_post_transfer(
                        runner, MigrationCase{RepresentationState::gpu_raw, destination},
                        faults[left], faults[right]);
                    ++fatal_runs;
                } else {
                    expect_recoverable_pair(runner, destination, faults[left], faults[right]);
                    ++recoverable_runs;
                }
            }
        }
    }
    VRAMZ_CHECK(runner, excluded == 1U && fatal_runs == 184U && recoverable_runs == 156U);
}

} // namespace

int main() {
    test::Runner runner;

    runner.begin("false plausible receipt plus local metadata defects cannot bypass corroboration");
    for (const auto fault :
         {FaultPoint::receipt_logical_size_mismatch, FaultPoint::receipt_stored_size_mismatch,
          FaultPoint::receipt_encoding_mismatch}) {
        expect_fatal_matrix(runner, FaultPoint::receipt_plausible_charge_mismatch, fault);
    }
    runner.begin("false plausible receipt plus query anomalies cannot bypass corroboration");
    for (const auto fault :
         {FaultPoint::stored_size_contract, FaultPoint::allocation_identity_mismatch,
          FaultPoint::allocation_tier_mismatch, FaultPoint::allocation_kind_mismatch}) {
        expect_fatal_matrix(runner, FaultPoint::receipt_plausible_charge_mismatch, fault);
    }
    runner.begin(
        "bounded post-transfer pairwise model preserves authority ownership and accounting");
    test_post_transfer_pairs(runner);

    runner.begin("all twelve transfers deliver exact allocation receipts without discovery");
    for (const auto source : test::all_states) {
        for (const auto destination : test::all_states) {
            if (source != destination) {
                check_transfer_receipt(runner, MigrationCase{source, destination});
            }
        }
    }

    runner.begin("every failing compaction or transfer preserves admitted charges");
    for (const auto source : test::all_states) {
        for (const auto destination :
             {RepresentationState::gpu_compressed, RepresentationState::host_compressed}) {
            if (source == destination) {
                continue;
            }
            for (const auto fault : {FaultPoint::transfer, FaultPoint::ambiguous_transfer,
                                     FaultPoint::final_storage_allocation}) {
                check_transfer_receipt(runner, MigrationCase{source, destination}, fault);
            }
            check_transfer_receipt(runner, MigrationCase{source, destination},
                                   is_raw(source) ? FaultPoint::compression
                                                  : FaultPoint::compressed_copy);
        }
    }

    runner.begin("descriptor identity disagreement prevents corroboration and is fatal");
    expect_fatal_matrix(runner, FaultPoint::allocation_identity_mismatch, FaultPoint::count, true);

    runner.begin("descriptor tier disagreement cannot redirect staging or cleanup debt");
    expect_fatal_matrix(runner, FaultPoint::allocation_tier_mismatch, FaultPoint::count, true);

    runner.begin("descriptor kind disagreement cannot replace adopted resource ownership");
    expect_fatal_matrix(runner, FaultPoint::allocation_kind_mismatch, FaultPoint::count, true);

    runner.begin("malformed query identity and charge cannot corroborate accounting");
    expect_fatal_matrix(runner, FaultPoint::allocation_identity_mismatch,
                        FaultPoint::allocation_descriptor_mismatch, true);
    runner.begin("malformed query tier and charge cannot corroborate accounting");
    expect_fatal_matrix(runner, FaultPoint::allocation_tier_mismatch,
                        FaultPoint::allocation_descriptor_mismatch, true);
    runner.begin("malformed query kind and charge cannot corroborate accounting");
    expect_fatal_matrix(runner, FaultPoint::allocation_kind_mismatch,
                        FaultPoint::allocation_descriptor_mismatch, true);

    runner.begin("query logical size mismatch with matching charge permits exact rollback");
    for (const bool cleanup_fails : {false, true}) {
        expect_compound_rollback(runner, FaultPoint::allocation_logical_size_mismatch, true,
                                 cleanup_fails);
    }
    runner.begin("query stored size mismatch with matching charge permits exact rollback");
    for (const bool cleanup_fails : {false, true}) {
        expect_compound_rollback(runner, FaultPoint::allocation_stored_size_mismatch, true,
                                 cleanup_fails);
    }
    runner.begin("query encoding mismatch with matching charge permits exact rollback");
    for (const bool cleanup_fails : {false, true}) {
        expect_compound_rollback(runner, FaultPoint::allocation_encoding_mismatch, true,
                                 cleanup_fails);
    }
    runner.begin("query CRC mismatch with matching charge permits exact rollback");
    for (const bool cleanup_fails : {false, true}) {
        expect_compound_rollback(runner, FaultPoint::allocation_crc_mismatch, true, cleanup_fails);
    }

    runner.begin("plausible false receipt charge contradicting the same resource is fatal");
    expect_fatal_matrix(runner, FaultPoint::receipt_plausible_charge_mismatch);

    runner.begin("invalid receipt logical size rolls back only after charge corroboration");
    for (const bool cleanup_fails : {false, true}) {
        expect_compound_rollback(runner, FaultPoint::receipt_logical_size_mismatch, true,
                                 cleanup_fails);
    }

    runner.begin("invalid receipt stored size rolls back only after charge corroboration");
    for (const bool cleanup_fails : {false, true}) {
        expect_compound_rollback(runner, FaultPoint::receipt_stored_size_mismatch, true,
                                 cleanup_fails);
    }

    runner.begin("invalid receipt encoding rolls back only after charge corroboration");
    for (const bool cleanup_fails : {false, true}) {
        expect_compound_rollback(runner, FaultPoint::receipt_encoding_mismatch, true,
                                 cleanup_fails);
    }

    runner.begin("charge beyond admission is a fatal contract error before ledger mutation");
    expect_fatal_charge_contract(runner, FaultPoint::receipt_charge_exceeds_admission);
    runner.begin("zero charge is a fatal contract error before ledger mutation");
    expect_fatal_charge_contract(runner, FaultPoint::receipt_zero_charge);
    runner.begin("unaligned charge is a fatal contract error before ledger mutation");
    expect_fatal_charge_contract(runner, FaultPoint::receipt_unaligned_charge);

    runner.begin("verification workspace OOM rolls back before source mutation");
    expect_rollback(
        runner, MigrationCase{RepresentationState::gpu_raw, RepresentationState::gpu_compressed},
        FaultPoint::verification_allocation);

    runner.begin("final compressed storage OOM rolls back private destination");
    expect_rollback(
        runner, MigrationCase{RepresentationState::gpu_raw, RepresentationState::host_compressed},
        FaultPoint::final_storage_allocation);

    runner.begin("compression failure rolls back private destination and workspace");
    expect_rollback(
        runner, MigrationCase{RepresentationState::gpu_raw, RepresentationState::gpu_compressed},
        FaultPoint::compression);

    runner.begin("decompression failure preserves compressed source");
    expect_rollback(
        runner, MigrationCase{RepresentationState::gpu_compressed, RepresentationState::host_raw},
        FaultPoint::decompression);

    runner.begin("unexpected successful decompression length fails verification and rolls back");
    expect_rollback(
        runner, MigrationCase{RepresentationState::gpu_compressed, RepresentationState::host_raw},
        FaultPoint::decompression_size_mismatch);

    runner.begin("raw copy failure preserves raw source");
    expect_rollback(runner,
                    MigrationCase{RepresentationState::gpu_raw, RepresentationState::host_raw},
                    FaultPoint::raw_copy);

    runner.begin("compressed copy failure preserves compressed source");
    expect_rollback(
        runner,
        MigrationCase{RepresentationState::gpu_compressed, RepresentationState::host_compressed},
        FaultPoint::compressed_copy);

    runner.begin("private destination byte corruption fails verification and rolls back");
    expect_rollback(
        runner, MigrationCase{RepresentationState::gpu_raw, RepresentationState::gpu_compressed},
        FaultPoint::corrupt_content);

    runner.begin("byte comparison mismatch cannot publish an otherwise valid destination");
    expect_rollback(runner,
                    MigrationCase{RepresentationState::gpu_raw, RepresentationState::host_raw},
                    FaultPoint::byte_comparison);

    runner.begin("destination verification fault cannot publish staging");
    expect_rollback(runner,
                    MigrationCase{RepresentationState::gpu_raw, RepresentationState::host_raw},
                    FaultPoint::verification);

    runner.begin("post-transfer query error prevents corroboration and fails stop");
    expect_fatal_matrix(runner, FaultPoint::stored_size_contract);

    runner.begin("post-compaction query error cannot enter cleanup in any state pair");
    expect_fatal_matrix(runner, FaultPoint::stored_size_contract, FaultPoint::count, true);

    runner.begin("post-compaction verification and cleanup failure conserve exact charge");
    expect_compound_rollback(runner, FaultPoint::verification);

    runner.begin("plausible false queried charge contradicting the same resource is fatal");
    expect_fatal_matrix(runner, FaultPoint::allocation_descriptor_mismatch);

    runner.begin("post-compaction verifier contract and cleanup failure poison with exact debt");
    expect_compound_rollback(runner, FaultPoint::verification_contract);

    runner.begin("runtime retries exact compacted cleanup debt once even after repeated failure");
    for (const auto destination :
         {RepresentationState::gpu_compressed, RepresentationState::host_compressed}) {
        auto runtime = Runtime::create(fault_config()).value();
        auto buffer = runtime.allocate(ByteSize{256U}).value();
        const auto source = testing::chunk_snapshot(buffer, 0U).value().authoritative;
        // A metadata contract fault permits exact debt only after charge corroboration.
        testing::inject_failure(runtime, FaultPoint::allocation_stored_size_mismatch, 2U);
        testing::inject_failure(runtime, FaultPoint::rollback_cleanup, 2U);
        const auto migrated = testing::migrate(buffer, 0U, destination);
        VRAMZ_CHECK(runner, !migrated);
        VRAMZ_CHECK(runner, testing::chunk_snapshot(buffer, 0U).value().authoritative.resource ==
                                source.resource);
        const auto retained = testing::RuntimeAccess::cleanup_resource(buffer, 0U, 0U).value();
        VRAMZ_CHECK(runner, retained.charge == ByteSize{64U});
        VRAMZ_CHECK(runner, !buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{256U}}));
        for (std::uint32_t attempt = 0U; attempt < 3U; ++attempt) {
            testing::inject_failure(runtime, FaultPoint::close_cleanup);
            VRAMZ_CHECK(runner, !buffer.close());
            const auto record = testing::RuntimeAccess::cleanup_resource(buffer, 0U, 0U).value();
            VRAMZ_CHECK(runner,
                        record.resource == retained.resource && record.charge == retained.charge);
            const auto stats = runtime.stats();
            const auto& usage = tier_of(destination) == PhysicalTier::gpu ? stats.gpu : stats.host;
            VRAMZ_CHECK(runner, usage.cleanup_debt == retained.charge);
            VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 2U);
            VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
            VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
        }
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, !buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U);
        const auto stats = runtime.stats();
        for (const auto& usage : {stats.gpu, stats.host}) {
            VRAMZ_CHECK(runner, usage.committed == ByteSize{} && usage.reserved == ByteSize{} &&
                                    usage.staging == ByteSize{} && usage.workspace == ByteSize{} &&
                                    usage.cleanup_debt == ByteSize{});
        }
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
    }

    runner.begin("authoritative raw corruption variants poison before migration commit");
    constexpr std::array<ResourceCorruption, 6U> raw_corruptions{
        ResourceCorruption::payload_flip, ResourceCorruption::logical_size,
        ResourceCorruption::stored_size,  ResourceCorruption::crc32c,
        ResourceCorruption::encoding,     ResourceCorruption::content_tag};
    for (const auto corruption : raw_corruptions) {
        auto runtime_result = Runtime::create(fault_config());
        auto runtime = std::move(runtime_result).value();
        auto buffer_result = runtime.allocate(ByteSize{256U});
        auto buffer = std::move(buffer_result).value();
        const auto before = testing::chunk_snapshot(buffer, 0U).value();
        VRAMZ_CHECK(runner, testing::corrupt(buffer, 0U, corruption));
        const auto migration = testing::migrate(buffer, 0U, RepresentationState::host_raw);
        const auto after = testing::chunk_snapshot(buffer, 0U).value();
        VRAMZ_CHECK(runner, !migration && migration.error().code == ErrorCode::integrity_failure);
        VRAMZ_CHECK(runner, after.lifecycle == LifecycleState::poisoned);
        VRAMZ_CHECK(runner, after.authoritative.resource == before.authoritative.resource);
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U);
    }

    runner.begin("authoritative compressed corruption variants poison safely");
    constexpr std::array<ResourceCorruption, 8U> compressed_corruptions{
        ResourceCorruption::payload_flip,     ResourceCorruption::payload_multiple_bits,
        ResourceCorruption::truncate_payload, ResourceCorruption::append_payload,
        ResourceCorruption::logical_size,     ResourceCorruption::stored_size,
        ResourceCorruption::crc32c,           ResourceCorruption::encoding};
    for (const auto corruption : compressed_corruptions) {
        auto runtime_result = Runtime::create(fault_config());
        auto runtime = std::move(runtime_result).value();
        auto buffer_result = runtime.allocate(ByteSize{256U});
        auto buffer = std::move(buffer_result).value();
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        const auto before = testing::chunk_snapshot(buffer, 0U).value();
        VRAMZ_CHECK(runner, testing::corrupt(buffer, 0U, corruption));
        const auto migration = testing::migrate(buffer, 0U, RepresentationState::host_raw);
        const auto after = testing::chunk_snapshot(buffer, 0U).value();
        VRAMZ_CHECK(runner, !migration);
        VRAMZ_CHECK(runner, after.lifecycle == LifecycleState::poisoned);
        VRAMZ_CHECK(runner, after.authoritative.resource == before.authoritative.resource);
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U);
    }

    runner.begin("failed migration never advances content generation");
    test::CoreFixture generation_fixture{RepresentationState::gpu_raw};
    const auto generation_before =
        generation_fixture.chunk->snapshot().authoritative.content.generation;
    generation_fixture.backend.inject_failure(FaultPoint::compression);
    VRAMZ_CHECK(runner, !generation_fixture.coordinator.migrate(
                            *generation_fixture.chunk, RepresentationState::host_compressed));
    VRAMZ_CHECK(runner, generation_fixture.chunk->snapshot().authoritative.content.generation ==
                            generation_before);
    VRAMZ_CHECK(runner, generation_fixture.ledger.usage(PhysicalTier::gpu).reserved == ByteSize{});
    VRAMZ_CHECK(runner, generation_fixture.ledger.usage(PhysicalTier::gpu).staging == ByteSize{});
    VRAMZ_CHECK(runner, generation_fixture.ledger.usage(PhysicalTier::gpu).workspace == ByteSize{});
    VRAMZ_CHECK(runner, generation_fixture.ledger.usage(PhysicalTier::host).reserved == ByteSize{});
    VRAMZ_CHECK(runner, generation_fixture.ledger.usage(PhysicalTier::host).staging == ByteSize{});

    return runner.finish();
}
