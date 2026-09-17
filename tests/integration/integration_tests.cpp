#include "../test_support.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <future>
#include <latch>
#include <mutex>
#include <optional>

using namespace vramz;

namespace {

class BlockingObserver final : public TransactionObserver {
  public:
    void on_phase(TransactionPhase phase, const ChunkSnapshot&) noexcept override {
        if (phase != TransactionPhase::reserved) {
            return;
        }
        std::unique_lock lock{mutex_};
        reached_ = true;
        condition_.notify_all();
        condition_.wait(lock, [this] { return released_; });
    }

    [[nodiscard]] bool wait_until_reached() {
        std::unique_lock lock{mutex_};
        return condition_.wait_for(lock, std::chrono::seconds{5}, [this] { return reached_; });
    }

    void release() {
        const std::scoped_lock lock{mutex_};
        released_ = true;
        condition_.notify_all();
    }

  private:
    std::mutex mutex_{};
    std::condition_variable condition_{};
    bool reached_{};
    bool released_{};
};

} // namespace

int main() {
    test::Runner runner;

    runner.begin("runtime shutdown succeeds without buffers");
    auto empty_runtime_result = Runtime::create(test::test_runtime_config());
    VRAMZ_CHECK(runner, empty_runtime_result);
    auto empty_runtime = std::move(empty_runtime_result).value();
    VRAMZ_CHECK(runner, empty_runtime.shutdown());
    const auto second_empty_shutdown = empty_runtime.shutdown();
    VRAMZ_CHECK(runner, !second_empty_shutdown &&
                            second_empty_shutdown.error().code == ErrorCode::stale_handle);

    runner.begin("all twelve semantic transitions share transaction machinery");
    std::uint64_t transition_count = 0U;
    for (const auto source : test::all_states) {
        for (const auto destination : test::all_states) {
            if (source == destination) {
                continue;
            }
            test::CoreFixture fixture{source};
            VRAMZ_CHECK(runner, fixture.ready());
            const auto before = fixture.chunk->snapshot();
            test::RecordingObserver observer{source};
            const auto migrated =
                fixture.coordinator.migrate(*fixture.chunk, destination, false, &observer);
            VRAMZ_CHECK(runner, migrated);
            const auto after = fixture.chunk->snapshot();
            VRAMZ_CHECK(runner, after.lifecycle == LifecycleState::live);
            VRAMZ_CHECK(runner, after.authoritative.state == destination);
            VRAMZ_CHECK(runner, after.authoritative.content == before.authoritative.content);
            VRAMZ_CHECK(runner, !observer.source_changed_before_commit());
            VRAMZ_CHECK(runner, observer.saw(TransactionPhase::commit_ready));
            VRAMZ_CHECK(runner, observer.saw(TransactionPhase::committed));
            VRAMZ_CHECK(runner, observer.saw(TransactionPhase::cleanup_complete));
            VRAMZ_CHECK(runner, !fixture.backend.owns(before.authoritative.resource));
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::gpu));
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::host));
            ++transition_count;
        }
    }
    VRAMZ_CHECK(runner, transition_count == 12U);

    runner.begin("all twelve transitions roll back before commit");
    transition_count = 0U;
    for (const auto source : test::all_states) {
        for (const auto destination : test::all_states) {
            if (source == destination) {
                continue;
            }
            test::CoreFixture fixture{source};
            const auto before = fixture.chunk->snapshot();
            fixture.backend.inject_failure(FaultPoint::transfer);
            test::RecordingObserver observer{source};
            const auto migrated =
                fixture.coordinator.migrate(*fixture.chunk, destination, false, &observer);
            VRAMZ_CHECK(runner, !migrated);
            const auto after = fixture.chunk->snapshot();
            VRAMZ_CHECK(runner, after.lifecycle == LifecycleState::live);
            VRAMZ_CHECK(runner, after.authoritative.resource == before.authoritative.resource);
            VRAMZ_CHECK(runner, after.authoritative.content == before.authoritative.content);
            VRAMZ_CHECK(runner, observer.saw(TransactionPhase::rolled_back));
            VRAMZ_CHECK(runner, !observer.saw(TransactionPhase::committed));
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::gpu));
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::host));
            if (source != RepresentationState::gpu_raw) {
                VRAMZ_CHECK(runner, fixture.coordinator.migrate(*fixture.chunk,
                                                                RepresentationState::gpu_raw));
            }
            std::array<std::byte, 256U> restored{};
            restored.fill(std::byte{0xFFU});
            const std::array<std::byte, 256U> expected{};
            const auto restored_snapshot = fixture.chunk->snapshot();
            VRAMZ_CHECK(runner, fixture.backend.read_bytes(restored_snapshot.authoritative.resource,
                                                           ByteOffset{}, restored));
            VRAMZ_CHECK(runner, restored == expected);
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::gpu));
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::host));
            ++transition_count;
        }
    }
    VRAMZ_CHECK(runner, transition_count == 12U);

    runner.begin("post commit cleanup failure keeps new authority and cleanup debt");
    for (const auto source : test::all_states) {
        for (const auto destination : test::all_states) {
            if (source == destination) {
                continue;
            }
            test::CoreFixture fixture{source};
            const auto before = fixture.chunk->snapshot();
            fixture.backend.inject_failure(FaultPoint::post_commit_cleanup);
            const auto migrated = fixture.coordinator.migrate(*fixture.chunk, destination);
            VRAMZ_CHECK(runner, !migrated);
            const auto after = fixture.chunk->snapshot();
            VRAMZ_CHECK(runner, after.authoritative.state == destination);
            VRAMZ_CHECK(runner, after.authoritative.content == before.authoritative.content);
            VRAMZ_CHECK(runner, after.authoritative.metadata.crc32c ==
                                    before.authoritative.metadata.crc32c);
            VRAMZ_CHECK(runner, fixture.backend.owns(after.authoritative.resource));
            VRAMZ_CHECK(runner, fixture.ledger.usage(tier_of(source)).cleanup_debt.value() >=
                                    before.authoritative.charge.value());
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::gpu));
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::host));
        }
    }

    runner.begin("runtime buffer and lease lifetime");
    auto runtime_result = Runtime::create(test::test_runtime_config());
    VRAMZ_CHECK(runner, runtime_result);
    auto runtime = std::move(runtime_result).value();
    auto buffer_result = runtime.allocate(ByteSize{192U});
    VRAMZ_CHECK(runner, buffer_result);
    auto buffer = std::move(buffer_result).value();
    const MemoryRange whole{ByteOffset{}, ByteSize{192U}};
    auto read_one = buffer.acquire(whole);
    auto read_two = buffer.acquire(whole);
    VRAMZ_CHECK(runner, read_one && read_two);
    auto write_conflict = buffer.acquire(whole, AcquireOptions{AccessMode::read_write});
    VRAMZ_CHECK(runner, !write_conflict && write_conflict.error().code == ErrorCode::conflict);
    VRAMZ_CHECK(runner, !buffer.close() && buffer.stats().closing);
    VRAMZ_CHECK(runner, read_one.value().device_span());
    VRAMZ_CHECK(runner, read_one.value().close());
    const auto stale_read_close = read_one.value().close();
    VRAMZ_CHECK(runner,
                !stale_read_close && stale_read_close.error().code == ErrorCode::stale_handle);
    VRAMZ_CHECK(runner, read_two.value().close());
    VRAMZ_CHECK(runner, buffer.close().error().code == ErrorCode::stale_handle);
    VRAMZ_CHECK(runner, runtime.shutdown());

    runner.begin("buffer close rejects new acquires across untouched chunks");
    auto closing_runtime_result = Runtime::create(test::test_runtime_config());
    auto closing_runtime = std::move(closing_runtime_result).value();
    auto closing_buffer_result = closing_runtime.allocate(ByteSize{128U});
    auto closing_buffer = std::move(closing_buffer_result).value();
    auto first_chunk_lease = closing_buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{64U}});
    VRAMZ_CHECK(runner, first_chunk_lease);
    VRAMZ_CHECK(runner, !closing_buffer.close());
    const auto untouched_acquire =
        closing_buffer.acquire(MemoryRange{ByteOffset{64U}, ByteSize{64U}});
    VRAMZ_CHECK(runner,
                !untouched_acquire && untouched_acquire.error().code == ErrorCode::shutting_down);
    VRAMZ_CHECK(runner, first_chunk_lease.value().close());
    VRAMZ_CHECK(runner, closing_runtime.shutdown());

    runner.begin("lease destruction releases pins exactly once");
    auto destructor_runtime_result = Runtime::create(test::test_runtime_config());
    auto destructor_runtime = std::move(destructor_runtime_result).value();
    auto destructor_buffer_result = destructor_runtime.allocate(ByteSize{64U});
    auto destructor_buffer = std::move(destructor_buffer_result).value();
    {
        auto scoped_lease = destructor_buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{64U}});
        VRAMZ_CHECK(runner, scoped_lease);
        VRAMZ_CHECK(runner, destructor_buffer.stats().active_read_leases == 1U);
    }
    VRAMZ_CHECK(runner, destructor_buffer.stats().active_read_leases == 0U);
    VRAMZ_CHECK(runner, destructor_buffer.close());
    VRAMZ_CHECK(runner, destructor_runtime.shutdown());

    runner.begin("writable lease changes deterministic content generation");
    auto write_runtime_result = Runtime::create(test::test_runtime_config());
    auto write_runtime = std::move(write_runtime_result).value();
    auto write_buffer_result = write_runtime.allocate(ByteSize{64U});
    auto write_buffer = std::move(write_buffer_result).value();
    const auto before_write = testing::chunk_snapshot(write_buffer, 0U);
    auto write_lease = write_buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{64U}},
                                            AcquireOptions{AccessMode::read_write});
    VRAMZ_CHECK(runner, write_lease);
    VRAMZ_CHECK(runner, write_lease.value().close());
    const auto after_write = testing::chunk_snapshot(write_buffer, 0U);
    VRAMZ_CHECK(runner, after_write.value().authoritative.content.generation ==
                            before_write.value().authoritative.content.generation + 1U);
    VRAMZ_CHECK(runner, write_buffer.close());
    VRAMZ_CHECK(runner, write_runtime.shutdown());

    runner.begin("every migration out of GPU raw is blocked by an active lease");
    for (const auto destination : test::all_states) {
        if (destination == RepresentationState::gpu_raw) {
            continue;
        }
        auto lease_runtime_result = Runtime::create(test::test_runtime_config());
        auto lease_runtime = std::move(lease_runtime_result).value();
        auto lease_buffer_result = lease_runtime.allocate(ByteSize{64U});
        auto lease_buffer = std::move(lease_buffer_result).value();
        auto pinned = lease_buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{64U}});
        VRAMZ_CHECK(runner, pinned);
        const auto blocked = testing::migrate(lease_buffer, 0U, destination);
        VRAMZ_CHECK(runner, !blocked && blocked.error().code == ErrorCode::busy);
        VRAMZ_CHECK(runner, pinned.value().close());
        VRAMZ_CHECK(runner, lease_buffer.close());
        VRAMZ_CHECK(runner, lease_runtime.shutdown());
    }

    runner.begin("simulated asynchronous APIs are immediately ready");
    auto async_runtime_result = Runtime::create(test::test_runtime_config());
    auto async_runtime = std::move(async_runtime_result).value();
    auto async_buffer_result = async_runtime.allocate(ByteSize{64U});
    auto async_buffer = std::move(async_buffer_result).value();
    std::optional<PendingLease> moved_pending{};
    {
        auto pending = async_buffer.acquire_async(MemoryRange{ByteOffset{}, ByteSize{64U}});
        VRAMZ_CHECK(runner, pending && pending.value().status() == OperationStatus::succeeded);
        moved_pending.emplace(std::move(pending).value());
    }
    auto ready_lease = moved_pending->wait();
    VRAMZ_CHECK(runner, ready_lease && ready_lease.value().close());
    auto operation = async_buffer.prefetch(MemoryRange{ByteOffset{}, ByteSize{64U}});
    VRAMZ_CHECK(runner, operation && operation.value().wait());
    VRAMZ_CHECK(runner, async_buffer.close());
    VRAMZ_CHECK(runner, async_runtime.shutdown());

    runner.begin("failed cleanup retention and runtime stats obey one lock order");
    for (std::uint64_t iteration = 0U; iteration < 16U; ++iteration) {
        auto lock_runtime_result = Runtime::create(test::test_runtime_config());
        auto lock_runtime = std::move(lock_runtime_result).value();
        auto lock_buffer_result = lock_runtime.allocate(ByteSize{64U});
        std::optional<Buffer> owner{};
        owner.emplace(std::move(lock_buffer_result).value());
        testing::inject_failure(lock_runtime, FaultPoint::close_cleanup);
        std::latch ready{2};
        std::latch start{1};
        auto destroyer =
            std::async(std::launch::async, [owned = std::move(owner), &ready, &start]() mutable {
                ready.count_down();
                start.wait();
                owned.reset();
            });
        auto observer = std::async(std::launch::async, [&lock_runtime, &ready, &start] {
            ready.count_down();
            start.wait();
            for (std::uint64_t sample = 0U; sample < 256U; ++sample) {
                static_cast<void>(lock_runtime.stats());
            }
        });
        ready.wait();
        start.count_down();
        VRAMZ_CHECK(runner,
                    destroyer.wait_for(std::chrono::seconds{2}) == std::future_status::ready);
        VRAMZ_CHECK(runner,
                    observer.wait_for(std::chrono::seconds{2}) == std::future_status::ready);
        destroyer.get();
        observer.get();
        VRAMZ_CHECK(runner, lock_runtime.stats().quarantined_buffers == 1U);
        VRAMZ_CHECK(runner, lock_runtime.shutdown());
    }

    runner.begin("failed cleanup retention can race runtime shutdown without AB BA deadlock");
    for (std::uint64_t iteration = 0U; iteration < 16U; ++iteration) {
        auto lock_runtime_result = Runtime::create(test::test_runtime_config());
        auto lock_runtime = std::move(lock_runtime_result).value();
        auto lock_buffer_result = lock_runtime.allocate(ByteSize{64U});
        std::optional<Buffer> owner{};
        owner.emplace(std::move(lock_buffer_result).value());
        testing::inject_failure(lock_runtime, FaultPoint::close_cleanup);
        std::latch ready{2};
        std::latch start{1};
        auto destroyer =
            std::async(std::launch::async, [owned = std::move(owner), &ready, &start]() mutable {
                ready.count_down();
                start.wait();
                owned.reset();
            });
        auto shutdown = std::async(std::launch::async, [&lock_runtime, &ready, &start] {
            ready.count_down();
            start.wait();
            return lock_runtime.shutdown();
        });
        ready.wait();
        start.count_down();
        VRAMZ_CHECK(runner,
                    destroyer.wait_for(std::chrono::seconds{2}) == std::future_status::ready);
        VRAMZ_CHECK(runner,
                    shutdown.wait_for(std::chrono::seconds{2}) == std::future_status::ready);
        destroyer.get();
        const auto shutdown_result = shutdown.get();
        if (!shutdown_result) {
            VRAMZ_CHECK(runner, shutdown_result.error().code == ErrorCode::busy);
            VRAMZ_CHECK(runner, lock_runtime.shutdown());
        }
        VRAMZ_CHECK(runner, testing::owned_resource_count(lock_runtime) == 0U);
    }

    runner.begin("buffer destruction cleanup can race shutdown without ownership loss");
    for (std::uint64_t iteration = 0U; iteration < 16U; ++iteration) {
        auto lock_runtime_result = Runtime::create(test::test_runtime_config());
        auto lock_runtime = std::move(lock_runtime_result).value();
        auto lock_buffer_result = lock_runtime.allocate(ByteSize{64U});
        std::optional<Buffer> owner{};
        owner.emplace(std::move(lock_buffer_result).value());
        std::latch ready{2};
        std::latch start{1};
        auto destroyer =
            std::async(std::launch::async, [owned = std::move(owner), &ready, &start]() mutable {
                ready.count_down();
                start.wait();
                owned.reset();
            });
        auto shutdown = std::async(std::launch::async, [&lock_runtime, &ready, &start] {
            ready.count_down();
            start.wait();
            return lock_runtime.shutdown();
        });
        ready.wait();
        start.count_down();
        VRAMZ_CHECK(runner,
                    destroyer.wait_for(std::chrono::seconds{2}) == std::future_status::ready);
        VRAMZ_CHECK(runner,
                    shutdown.wait_for(std::chrono::seconds{2}) == std::future_status::ready);
        destroyer.get();
        const auto shutdown_result = shutdown.get();
        if (!shutdown_result) {
            VRAMZ_CHECK(runner, shutdown_result.error().code == ErrorCode::busy);
            VRAMZ_CHECK(runner, lock_runtime.shutdown());
        }
        VRAMZ_CHECK(runner, testing::owned_resource_count(lock_runtime) == 0U);
    }

    runner.begin("concurrent multi chunk readers complete without lock inversion");
    auto concurrent_runtime_result = Runtime::create(test::test_runtime_config());
    auto concurrent_runtime = std::move(concurrent_runtime_result).value();
    auto concurrent_buffer_result = concurrent_runtime.allocate(ByteSize{256U});
    auto concurrent_buffer = std::move(concurrent_buffer_result).value();
    auto worker = [&concurrent_buffer]() {
        for (std::uint64_t iteration = 0U; iteration < 100U; ++iteration) {
            auto lease = concurrent_buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{256U}});
            if (!lease || !lease.value().close()) {
                return false;
            }
        }
        return true;
    };
    auto first_worker = std::async(std::launch::async, worker);
    auto second_worker = std::async(std::launch::async, worker);
    VRAMZ_CHECK(runner,
                first_worker.wait_for(std::chrono::seconds{5}) == std::future_status::ready);
    VRAMZ_CHECK(runner,
                second_worker.wait_for(std::chrono::seconds{5}) == std::future_status::ready);
    VRAMZ_CHECK(runner, first_worker.get());
    VRAMZ_CHECK(runner, second_worker.get());
    VRAMZ_CHECK(runner, concurrent_buffer.close());
    VRAMZ_CHECK(runner, concurrent_runtime.shutdown());

    runner.begin("concurrent acquire and shutdown cannot cross an active transaction");
    auto race_runtime_result = Runtime::create(test::test_runtime_config());
    auto race_runtime = std::move(race_runtime_result).value();
    auto race_buffer_result = race_runtime.allocate(ByteSize{64U});
    auto race_buffer = std::move(race_buffer_result).value();
    BlockingObserver blocking_observer{};
    auto migration = std::async(std::launch::async, [&race_buffer, &blocking_observer] {
        return testing::migrate(race_buffer, 0U, RepresentationState::host_raw, &blocking_observer);
    });
    VRAMZ_CHECK(runner, blocking_observer.wait_until_reached());
    const auto acquire_during_migration =
        race_buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{64U}});
    VRAMZ_CHECK(runner, !acquire_during_migration &&
                            acquire_during_migration.error().code == ErrorCode::conflict);
    const auto shutdown_during_migration = race_runtime.shutdown();
    VRAMZ_CHECK(runner, !shutdown_during_migration &&
                            shutdown_during_migration.error().code == ErrorCode::busy);
    blocking_observer.release();
    VRAMZ_CHECK(runner, migration.wait_for(std::chrono::seconds{5}) == std::future_status::ready);
    VRAMZ_CHECK(runner, migration.get());
    VRAMZ_CHECK(runner, race_buffer.close());
    VRAMZ_CHECK(runner, race_runtime.shutdown());

    return runner.finish();
}
