#include "../test_support.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <limits>
#include <optional>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace vramz;

namespace {

struct StatePair final {
    RepresentationState source;
    RepresentationState destination;
};

void expect_rollback(test::Runner& runner, FaultPoint fault, StatePair states) {
    test::CoreFixture fixture{states.source};
    const auto before = fixture.chunk->snapshot();
    fixture.backend.inject_failure(fault);
    const auto result = fixture.coordinator.migrate(*fixture.chunk, states.destination);
    VRAMZ_CHECK(runner, !result);
    const auto after = fixture.chunk->snapshot();
    VRAMZ_CHECK(runner, after.authoritative.resource == before.authoritative.resource);
    VRAMZ_CHECK(runner, after.authoritative.content == before.authoritative.content);
    VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::gpu));
    VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::host));
}

void expect_allocation_contract_stop(test::Runner& runner, FaultPoint point, bool initial = false,
                                     std::uint64_t invocation = 1U) {
    const auto child = ::fork();
    VRAMZ_CHECK(runner, child >= 0);
    if (child < 0) {
        return;
    }
    if (child == 0) {
        static_cast<void>(::alarm(5U));
        if (initial) {
            auto created = Runtime::create(test::test_runtime_config());
            if (!created) {
                std::_Exit(1);
            }
            auto runtime = std::move(created).value();
            static Runtime* child_runtime{};
            static ByteSize previous_committed{};
            child_runtime = &runtime;
            previous_committed = invocation == 2U ? ByteSize{64U} : ByteSize{};
            std::set_terminate([]() noexcept {
                const auto stats = child_runtime->stats();
                const auto error = child_runtime->first_async_error();
                const auto& backend = testing::RuntimeAccess::backend(*child_runtime);
                const bool stopped = stats.gpu.committed == previous_committed &&
                                     stats.host.committed == ByteSize{} &&
                                     stats.gpu.cleanup_debt == ByteSize{} &&
                                     stats.host.cleanup_debt == ByteSize{} &&
                                     backend.last_release_resource() == ResourceId{} && error &&
                                     error->error.code == ErrorCode::backend_contract_violation;
                std::_Exit(stopped ? 86 : 1);
            });
            testing::inject_failure(runtime, point, invocation);
            static_cast<void>(runtime.allocate(ByteSize{invocation == 2U ? 128U : 64U}));
        } else {
            test::CoreFixture fixture{RepresentationState::gpu_raw};
            if (!fixture.ready()) {
                std::_Exit(1);
            }
            test::RecordingObserver observer{RepresentationState::gpu_raw};
            static test::CoreFixture* child_fixture{};
            static test::RecordingObserver* child_observer{};
            static ChunkSnapshot before{};
            child_fixture = &fixture;
            child_observer = &observer;
            before = fixture.chunk->snapshot();
            std::set_terminate([]() noexcept {
                const auto after = child_fixture->chunk->snapshot();
                const auto error = child_fixture->errors.first();
                const bool stopped =
                    after.lifecycle == LifecycleState::live && after.transition_active &&
                    after.authoritative.resource == before.authoritative.resource &&
                    after.authoritative.charge == before.authoritative.charge &&
                    after.authoritative.content == before.authoritative.content &&
                    after.cleanup_resource_count == 0U &&
                    child_fixture->ledger.usage(PhysicalTier::gpu).cleanup_debt == ByteSize{} &&
                    child_fixture->ledger.usage(PhysicalTier::host).cleanup_debt == ByteSize{} &&
                    child_fixture->backend.last_release_resource() == ResourceId{} &&
                    !child_observer->saw(TransactionPhase::destination_materialized) &&
                    !child_observer->saw(TransactionPhase::rolled_back) &&
                    !child_observer->saw(TransactionPhase::poisoned) &&
                    !child_observer->saw(TransactionPhase::committed) && error &&
                    error->error.code == ErrorCode::backend_contract_violation;
                std::_Exit(stopped ? 86 : 1);
            });
            fixture.backend.inject_failure(point, invocation);
            static_cast<void>(fixture.coordinator.migrate(
                *fixture.chunk, RepresentationState::host_raw, false, &observer));
        }
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

} // namespace

int main() {
    test::Runner runner;

    runner.begin("GPU destination allocation failure rolls back");
    expect_rollback(runner, FaultPoint::gpu_allocation,
                    StatePair{RepresentationState::host_raw, RepresentationState::gpu_raw});

    runner.begin("host destination allocation failure rolls back");
    expect_rollback(runner, FaultPoint::host_allocation,
                    StatePair{RepresentationState::gpu_raw, RepresentationState::host_raw});

    runner.begin("workspace allocation failure rolls back");
    expect_rollback(runner, FaultPoint::workspace_allocation,
                    StatePair{RepresentationState::gpu_raw, RepresentationState::gpu_compressed});

    runner.begin("transfer failure rolls back");
    expect_rollback(runner, FaultPoint::transfer,
                    StatePair{RepresentationState::gpu_raw, RepresentationState::host_raw});

    runner.begin("verification failure rolls back");
    expect_rollback(runner, FaultPoint::verification,
                    StatePair{RepresentationState::gpu_raw, RepresentationState::host_raw});

    runner.begin("corrupt content is detected and rolled back");
    expect_rollback(runner, FaultPoint::corrupt_content,
                    StatePair{RepresentationState::gpu_raw, RepresentationState::host_raw});

    runner.begin("rollback cleanup failure becomes cleanup debt");
    test::CoreFixture rollback_cleanup{RepresentationState::gpu_raw};
    rollback_cleanup.backend.inject_failure(FaultPoint::transfer);
    rollback_cleanup.backend.inject_failure(FaultPoint::rollback_cleanup);
    const auto rollback_result = rollback_cleanup.coordinator.migrate(
        *rollback_cleanup.chunk, RepresentationState::gpu_compressed);
    VRAMZ_CHECK(runner, !rollback_result);
    VRAMZ_CHECK(runner, rollback_cleanup.chunk->snapshot().lifecycle == LifecycleState::live);
    VRAMZ_CHECK(runner, rollback_cleanup.ledger.usage(PhysicalTier::gpu).cleanup_debt.value() > 0U);
    VRAMZ_CHECK(runner, rollback_cleanup.chunk->snapshot().cleanup_resource_count > 0U);
    VRAMZ_CHECK(runner, rollback_cleanup.conserved(PhysicalTier::gpu));

    runner.begin("verified workspace cleanup failure retains a retryable handle");
    auto workspace_cleanup_runtime_result = Runtime::create(test::test_runtime_config());
    auto workspace_cleanup_runtime = std::move(workspace_cleanup_runtime_result).value();
    auto workspace_cleanup_buffer_result = workspace_cleanup_runtime.allocate(ByteSize{64U});
    auto workspace_cleanup_buffer = std::move(workspace_cleanup_buffer_result).value();
    testing::inject_failure(workspace_cleanup_runtime, FaultPoint::rollback_cleanup);
    const auto workspace_cleanup_migration =
        testing::migrate(workspace_cleanup_buffer, 0U, RepresentationState::gpu_compressed);
    VRAMZ_CHECK(runner, !workspace_cleanup_migration);
    const auto workspace_cleanup_snapshot = testing::chunk_snapshot(workspace_cleanup_buffer, 0U);
    VRAMZ_CHECK(runner, workspace_cleanup_snapshot &&
                            workspace_cleanup_snapshot.value().authoritative.state ==
                                RepresentationState::gpu_raw &&
                            workspace_cleanup_snapshot.value().cleanup_resource_count == 1U);
    VRAMZ_CHECK(runner, workspace_cleanup_runtime.stats().gpu.cleanup_debt.value() > 0U);
    VRAMZ_CHECK(runner, workspace_cleanup_buffer.close());
    VRAMZ_CHECK(runner, workspace_cleanup_runtime.stats().gpu.cleanup_debt == ByteSize{});
    VRAMZ_CHECK(runner, workspace_cleanup_runtime.shutdown());

    runner.begin("ambiguous backend outcome poisons and fails closed");
    test::CoreFixture ambiguous{RepresentationState::gpu_raw};
    ambiguous.backend.inject_failure(FaultPoint::ambiguous_transfer);
    const auto ambiguous_result =
        ambiguous.coordinator.migrate(*ambiguous.chunk, RepresentationState::host_raw);
    VRAMZ_CHECK(runner, !ambiguous_result);
    VRAMZ_CHECK(runner, ambiguous.chunk->snapshot().lifecycle == LifecycleState::poisoned);
    const auto poisoned_retry =
        ambiguous.coordinator.migrate(*ambiguous.chunk, RepresentationState::gpu_compressed);
    VRAMZ_CHECK(runner, !poisoned_retry && poisoned_retry.error().code == ErrorCode::poisoned);
    VRAMZ_CHECK(runner, ambiguous.conserved(PhysicalTier::gpu));
    VRAMZ_CHECK(runner, ambiguous.conserved(PhysicalTier::host));

    runner.begin("advertised bound smaller than actual stops before guessed migration debt");
    expect_allocation_contract_stop(runner, FaultPoint::advertised_bound_too_small);

    runner.begin("initial allocation bound violation stops before guessed runtime debt");
    expect_allocation_contract_stop(runner, FaultPoint::advertised_bound_too_small, true);

    runner.begin("failed private allocation cleanup is quarantined until shutdown retry");
    auto quarantine_runtime_result = Runtime::create(test::test_runtime_config());
    auto quarantine_runtime = std::move(quarantine_runtime_result).value();
    testing::inject_failure(quarantine_runtime, FaultPoint::gpu_allocation, 2U);
    testing::inject_failure(quarantine_runtime, FaultPoint::close_cleanup);
    const auto quarantine_buffer = quarantine_runtime.allocate(ByteSize{128U});
    VRAMZ_CHECK(runner, !quarantine_buffer &&
                            quarantine_buffer.error().code == ErrorCode::out_of_gpu_memory);
    VRAMZ_CHECK(runner, quarantine_runtime.stats().shutting_down);
    VRAMZ_CHECK(runner, quarantine_runtime.stats().gpu.committed == ByteSize{});
    VRAMZ_CHECK(runner, quarantine_runtime.stats().gpu.cleanup_debt == ByteSize{64U});
    VRAMZ_CHECK(runner, testing::accounting_conserved(quarantine_runtime, PhysicalTier::gpu));
    VRAMZ_CHECK(runner, quarantine_runtime.shutdown());
    VRAMZ_CHECK(runner, quarantine_runtime.stats().gpu.cleanup_debt == ByteSize{});

    runner.begin("one failed destructor cleanup is retained and removed after retry");
    auto one_quarantine_runtime_result = Runtime::create(test::test_runtime_config());
    auto one_quarantine_runtime = std::move(one_quarantine_runtime_result).value();
    {
        auto one_buffer_result = one_quarantine_runtime.allocate(ByteSize{64U});
        auto one_buffer = std::move(one_buffer_result).value();
        testing::inject_failure(one_quarantine_runtime, FaultPoint::close_cleanup);
    }
    VRAMZ_CHECK(runner, one_quarantine_runtime.stats().quarantined_buffers == 1U);
    VRAMZ_CHECK(runner, one_quarantine_runtime.stats().gpu.committed == ByteSize{});
    VRAMZ_CHECK(runner, one_quarantine_runtime.stats().gpu.cleanup_debt == ByteSize{64U});
    VRAMZ_CHECK(runner, testing::owned_resource_count(one_quarantine_runtime) == 1U);
    VRAMZ_CHECK(runner, one_quarantine_runtime.shutdown());
    VRAMZ_CHECK(runner, one_quarantine_runtime.stats().quarantined_buffers == 0U);
    VRAMZ_CHECK(runner, one_quarantine_runtime.stats().gpu.cleanup_debt == ByteSize{});
    VRAMZ_CHECK(runner, testing::owned_resource_count(one_quarantine_runtime) == 0U);

    runner.begin("several failed destructor cleanups retain independent runtime slots");
    auto several_runtime_result = Runtime::create(test::test_runtime_config());
    auto several_runtime = std::move(several_runtime_result).value();
    std::array<std::optional<Buffer>, 3U> several_buffers{};
    for (auto& slot : several_buffers) {
        auto allocated = several_runtime.allocate(ByteSize{64U});
        VRAMZ_CHECK(runner, allocated);
        slot.emplace(std::move(allocated).value());
    }
    for (auto& slot : several_buffers) {
        testing::inject_failure(several_runtime, FaultPoint::close_cleanup);
        slot.reset();
    }
    VRAMZ_CHECK(runner, several_runtime.stats().quarantined_buffers == 3U);
    VRAMZ_CHECK(runner, several_runtime.stats().gpu.cleanup_debt == ByteSize{192U});
    VRAMZ_CHECK(runner, testing::owned_resource_count(several_runtime) == 3U);
    VRAMZ_CHECK(runner, testing::accounting_conserved(several_runtime, PhysicalTier::gpu));
    VRAMZ_CHECK(runner, several_runtime.shutdown());
    VRAMZ_CHECK(runner, several_runtime.stats().quarantined_buffers == 0U);
    VRAMZ_CHECK(runner, testing::owned_resource_count(several_runtime) == 0U);

    runner.begin("duplicate quarantine insertion is idempotent");
    auto duplicate_quarantine_runtime_result = Runtime::create(test::test_runtime_config());
    auto duplicate_quarantine_runtime = std::move(duplicate_quarantine_runtime_result).value();
    auto duplicate_quarantine_buffer_result = duplicate_quarantine_runtime.allocate(ByteSize{64U});
    auto duplicate_quarantine_buffer = std::move(duplicate_quarantine_buffer_result).value();
    VRAMZ_CHECK(runner, testing::quarantine(duplicate_quarantine_buffer));
    VRAMZ_CHECK(runner, testing::quarantine(duplicate_quarantine_buffer));
    VRAMZ_CHECK(runner, duplicate_quarantine_runtime.stats().quarantined_buffers == 1U);
    VRAMZ_CHECK(runner, duplicate_quarantine_runtime.shutdown());
    VRAMZ_CHECK(runner, duplicate_quarantine_runtime.stats().quarantined_buffers == 0U);
    VRAMZ_CHECK(runner, testing::owned_resource_count(duplicate_quarantine_runtime) == 0U);

    runner.begin("persistent quarantine retry failure stays owned and accounted");
    auto persistent_runtime_result = Runtime::create(test::test_runtime_config());
    auto persistent_runtime = std::move(persistent_runtime_result).value();
    {
        auto persistent_buffer_result = persistent_runtime.allocate(ByteSize{64U});
        auto persistent_buffer = std::move(persistent_buffer_result).value();
        testing::inject_failure(persistent_runtime, FaultPoint::close_cleanup);
    }
    testing::inject_failure(persistent_runtime, FaultPoint::close_cleanup);
    const auto persistent_retry = persistent_runtime.shutdown();
    VRAMZ_CHECK(runner,
                !persistent_retry && persistent_retry.error().code == ErrorCode::backend_failure);
    VRAMZ_CHECK(runner, persistent_runtime.stats().quarantined_buffers == 1U);
    VRAMZ_CHECK(runner, persistent_runtime.stats().gpu.cleanup_debt == ByteSize{64U});
    VRAMZ_CHECK(runner, testing::owned_resource_count(persistent_runtime) == 1U);
    VRAMZ_CHECK(runner, testing::accounting_conserved(persistent_runtime, PhysicalTier::gpu));
    VRAMZ_CHECK(runner, persistent_runtime.shutdown());
    VRAMZ_CHECK(runner, testing::owned_resource_count(persistent_runtime) == 0U);

    runner.begin("runtime destruction breaks quarantine ownership without a shared pointer cycle");
    std::weak_ptr<const void> runtime_lifetime{};
    std::weak_ptr<const void> buffer_lifetime{};
    {
        auto cycle_runtime_result = Runtime::create(test::test_runtime_config());
        auto cycle_runtime = std::move(cycle_runtime_result).value();
        runtime_lifetime = testing::runtime_lifetime(cycle_runtime);
        {
            auto cycle_buffer_result = cycle_runtime.allocate(ByteSize{64U});
            auto cycle_buffer = std::move(cycle_buffer_result).value();
            buffer_lifetime = testing::buffer_lifetime(cycle_buffer);
            testing::inject_failure(cycle_runtime, FaultPoint::close_cleanup);
        }
        VRAMZ_CHECK(runner, !runtime_lifetime.expired() && !buffer_lifetime.expired());
        VRAMZ_CHECK(runner, cycle_runtime.stats().quarantined_buffers == 1U);
        VRAMZ_CHECK(runner, cycle_runtime.stats().gpu.cleanup_debt == ByteSize{64U});
        VRAMZ_CHECK(runner, testing::accounting_conserved(cycle_runtime, PhysicalTier::gpu));
        testing::inject_failure(cycle_runtime, FaultPoint::close_cleanup);
    }
    VRAMZ_CHECK(runner, runtime_lifetime.expired());
    VRAMZ_CHECK(runner, buffer_lifetime.expired());

    runner.begin("quarantine capacity is reserved before backend resources are created");
    auto capacity_runtime_result = Runtime::create(test::test_runtime_config());
    auto capacity_runtime = std::move(capacity_runtime_result).value();
    std::vector<Buffer> capacity_buffers{};
    capacity_buffers.reserve(max_runtime_buffers);
    for (std::uint32_t index = 0U; index < max_runtime_buffers; ++index) {
        auto allocated = capacity_runtime.allocate(ByteSize{64U});
        VRAMZ_CHECK(runner, allocated);
        capacity_buffers.emplace_back(std::move(allocated).value());
    }
    const auto before_capacity_failure = capacity_runtime.stats();
    const auto capacity_failure = capacity_runtime.allocate(ByteSize{64U});
    VRAMZ_CHECK(runner, !capacity_failure &&
                            capacity_failure.error().code == ErrorCode::out_of_host_memory);
    VRAMZ_CHECK(runner,
                capacity_runtime.stats().gpu.committed == before_capacity_failure.gpu.committed);
    VRAMZ_CHECK(runner, capacity_runtime.stats().gpu.cleanup_debt == ByteSize{});
    VRAMZ_CHECK(runner, testing::owned_resource_count(capacity_runtime) == max_runtime_buffers);
    for (auto& buffer : capacity_buffers) {
        VRAMZ_CHECK(runner, buffer.close());
    }
    capacity_buffers.clear();
    VRAMZ_CHECK(runner, capacity_runtime.shutdown());
    VRAMZ_CHECK(runner, testing::owned_resource_count(capacity_runtime) == 0U);

    runner.begin("invalid ownership identity is fatal without inventing a resource");
    expect_allocation_contract_stop(runner, FaultPoint::invalid_resource_identity);

    runner.begin("duplicate resource identity is rejected");
    expect_allocation_contract_stop(runner, FaultPoint::duplicate_resource_identity);

    runner.begin("duplicate identity from another chunk fails the runtime closed");
    expect_allocation_contract_stop(runner, FaultPoint::duplicate_resource_identity, true, 2U);

    runner.begin("backend rejects an allocation charge that violates granularity");
    auto invalid_charge_config = test::test_backend_config();
    invalid_charge_config.exact_state_charges[0] = ByteSize{65U};
    MockBackend invalid_charge_backend{invalid_charge_config};
    const auto invalid_charge =
        invalid_charge_backend.allocation_bound(RepresentationState::gpu_raw, ByteSize{256U});
    VRAMZ_CHECK(runner, !invalid_charge &&
                            invalid_charge.error().code == ErrorCode::backend_contract_violation);

    runner.begin("backend double release is a contract error");
    MockBackend release_backend{test::test_backend_config()};
    const auto release_address =
        release_backend.reserve_address_space(ByteSize{256U}, ByteSize{64U});
    VRAMZ_CHECK(runner, release_address);
    const auto release_allocation = release_backend.allocate_representation(
        RepresentationAllocationRequest{RepresentationState::gpu_raw, ByteSize{256U},
                                        ContentTag{1U, 2U}, release_address.value().base});
    VRAMZ_CHECK(runner, release_allocation);
    const auto release_adopted = release_backend.adopt_allocation(release_allocation.value());
    VRAMZ_CHECK(runner, release_adopted);
    VRAMZ_CHECK(runner,
                release_backend.release(release_adopted.value().id, ReleasePhase::rollback));
    const auto second_release =
        release_backend.release(release_adopted.value().id, ReleasePhase::rollback);
    VRAMZ_CHECK(runner, !second_release &&
                            second_release.error().code == ErrorCode::backend_contract_violation);

    runner.begin("false release success is detected after commit");
    test::CoreFixture false_release{RepresentationState::gpu_raw};
    false_release.backend.inject_failure(FaultPoint::false_release_success);
    const auto false_release_result =
        false_release.coordinator.migrate(*false_release.chunk, RepresentationState::host_raw);
    VRAMZ_CHECK(runner, !false_release_result && false_release_result.error().code ==
                                                     ErrorCode::backend_contract_violation);
    VRAMZ_CHECK(runner, false_release.chunk->snapshot().lifecycle == LifecycleState::poisoned);
    VRAMZ_CHECK(runner, false_release.conserved(PhysicalTier::gpu));
    VRAMZ_CHECK(runner, false_release.conserved(PhysicalTier::host));

    runner.begin("poisoned runtime buffer rejects acquire prefetch and migration");
    auto poisoned_runtime_result = Runtime::create(test::test_runtime_config());
    auto poisoned_runtime = std::move(poisoned_runtime_result).value();
    auto poisoned_buffer_result = poisoned_runtime.allocate(ByteSize{64U});
    auto poisoned_buffer = std::move(poisoned_buffer_result).value();
    testing::inject_failure(poisoned_runtime, FaultPoint::ambiguous_transfer);
    VRAMZ_CHECK(runner, !testing::migrate(poisoned_buffer, 0U, RepresentationState::host_raw));
    const auto poisoned_read = poisoned_buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{64U}});
    const auto poisoned_write = poisoned_buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{64U}},
                                                        AcquireOptions{AccessMode::read_write});
    const auto poisoned_prefetch =
        poisoned_buffer.prefetch(MemoryRange{ByteOffset{}, ByteSize{64U}});
    VRAMZ_CHECK(runner, !poisoned_read && poisoned_read.error().code == ErrorCode::poisoned);
    VRAMZ_CHECK(runner, !poisoned_write && poisoned_write.error().code == ErrorCode::poisoned);
    VRAMZ_CHECK(runner,
                !poisoned_prefetch && poisoned_prefetch.error().code == ErrorCode::poisoned);
    const auto poisoned_migration =
        testing::migrate(poisoned_buffer, 0U, RepresentationState::gpu_compressed);
    VRAMZ_CHECK(runner,
                !poisoned_migration && poisoned_migration.error().code == ErrorCode::poisoned);
    VRAMZ_CHECK(runner, poisoned_buffer.close());
    VRAMZ_CHECK(runner, poisoned_runtime.shutdown());

    runner.begin("write completion failure poisons but releases its pin");
    auto write_failure_runtime_result = Runtime::create(test::test_runtime_config());
    auto write_failure_runtime = std::move(write_failure_runtime_result).value();
    auto write_failure_buffer_result = write_failure_runtime.allocate(ByteSize{64U});
    auto write_failure_buffer = std::move(write_failure_buffer_result).value();
    auto write_failure_lease = write_failure_buffer.acquire(
        MemoryRange{ByteOffset{}, ByteSize{64U}}, AcquireOptions{AccessMode::read_write});
    VRAMZ_CHECK(runner, write_failure_lease);
    testing::inject_failure(write_failure_runtime, FaultPoint::write_content);
    VRAMZ_CHECK(runner, !write_failure_lease.value().close());
    const auto write_failure_snapshot = testing::chunk_snapshot(write_failure_buffer, 0U);
    VRAMZ_CHECK(runner, write_failure_snapshot &&
                            write_failure_snapshot.value().lifecycle == LifecycleState::poisoned &&
                            !write_failure_snapshot.value().write_pin);
    VRAMZ_CHECK(runner, write_failure_buffer.close());
    VRAMZ_CHECK(runner, write_failure_runtime.shutdown());

    runner.begin("active lease prevents migration and stale release cannot double unpin");
    auto runtime_result = Runtime::create(test::test_runtime_config());
    auto runtime = std::move(runtime_result).value();
    auto buffer_result = runtime.allocate(ByteSize{64U});
    auto buffer = std::move(buffer_result).value();
    auto lease_result = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{64U}});
    VRAMZ_CHECK(runner, lease_result);
    const auto blocked = testing::migrate(buffer, 0U, RepresentationState::host_raw);
    VRAMZ_CHECK(runner, !blocked && blocked.error().code == ErrorCode::busy);
    VRAMZ_CHECK(runner, lease_result.value().close());
    const auto stale = lease_result.value().close();
    VRAMZ_CHECK(runner, !stale && stale.error().code == ErrorCode::stale_handle);
    VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::host_raw));
    VRAMZ_CHECK(runner, buffer.close());
    VRAMZ_CHECK(runner, runtime.shutdown());

    runner.begin("shutdown observes live buffer and close cleanup failure remains charged");
    auto shutdown_runtime_result = Runtime::create(test::test_runtime_config());
    auto shutdown_runtime = std::move(shutdown_runtime_result).value();
    auto shutdown_buffer_result = shutdown_runtime.allocate(ByteSize{64U});
    auto shutdown_buffer = std::move(shutdown_buffer_result).value();
    VRAMZ_CHECK(runner, !shutdown_runtime.shutdown());
    testing::inject_failure(shutdown_runtime, FaultPoint::close_cleanup);
    const auto close_failure = shutdown_buffer.close();
    VRAMZ_CHECK(runner, !close_failure && close_failure.error().code == ErrorCode::backend_failure);
    VRAMZ_CHECK(runner, shutdown_runtime.stats().gpu.cleanup_debt.value() > 0U);
    VRAMZ_CHECK(runner, shutdown_buffer.close());
    VRAMZ_CHECK(runner, shutdown_runtime.stats().gpu.cleanup_debt == ByteSize{});
    VRAMZ_CHECK(runner, shutdown_runtime.shutdown());

    runner.begin("invalid and overflowing ranges fail before resource mutation");
    auto range_runtime_result = Runtime::create(test::test_runtime_config());
    auto range_runtime = std::move(range_runtime_result).value();
    auto range_buffer_result = range_runtime.allocate(ByteSize{64U});
    auto range_buffer = std::move(range_buffer_result).value();
    const auto before_range = range_runtime.stats().gpu.committed;
    VRAMZ_CHECK(runner, !range_buffer.acquire(MemoryRange{ByteOffset{64U}, ByteSize{1U}}));
    VRAMZ_CHECK(runner, !range_buffer.acquire(MemoryRange{
                            ByteOffset{std::numeric_limits<std::uint64_t>::max()}, ByteSize{2U}}));
    VRAMZ_CHECK(runner, range_runtime.stats().gpu.committed == before_range);
    VRAMZ_CHECK(runner, range_buffer.close());
    VRAMZ_CHECK(runner, range_runtime.shutdown());

    runner.begin("async channel saturation cannot alter transaction correctness");
    AsyncErrorChannel channel{1U};
    for (std::uint64_t index = 0U; index < 1000U; ++index) {
        channel.push(make_error(index == 0U ? ErrorCode::backend_failure : ErrorCode::timeout,
                                OperationId::release, index));
    }
    VRAMZ_CHECK(runner, channel.stats().retained == 1U);
    VRAMZ_CHECK(runner, channel.stats().dropped == 999U);
    const auto sticky_error = channel.first().value_or(AsyncErrorRecord{});
    VRAMZ_CHECK(runner, sticky_error.error.code == ErrorCode::backend_failure);
    test::CoreFixture after_saturation{RepresentationState::gpu_raw};
    VRAMZ_CHECK(runner, after_saturation.coordinator.migrate(*after_saturation.chunk,
                                                             RepresentationState::host_raw));
    VRAMZ_CHECK(runner, after_saturation.conserved(PhysicalTier::gpu));
    VRAMZ_CHECK(runner, after_saturation.conserved(PhysicalTier::host));

    return runner.finish();
}
