#include "../support/fake_cuda_driver.hpp"
#include "../support/fake_nvcomp.hpp"
#include "../test_support.hpp"
#include "vramz/detail/cuda_vmm_backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <limits>
#include <sys/wait.h>
#include <thread>
#include <type_traits>
#include <unistd.h>

using namespace vramz;

namespace {
static_assert(std::is_trivially_copyable_v<CompletionToken>);
static_assert(std::is_nothrow_constructible_v<Result<CompletionToken>, CompletionToken>);
static_assert(std::is_nothrow_move_constructible_v<PendingLeaseRelease>);
static_assert(!std::is_move_assignable_v<PendingLeaseRelease>);
static_assert(!std::is_copy_constructible_v<PendingLeaseRelease>);
static_assert(!std::is_move_assignable_v<Lease>);
static_assert(!std::is_move_assignable_v<Buffer>);
static_assert(!std::is_move_assignable_v<Runtime>);

struct Fixture final {
    std::shared_ptr<test::FakeCudaState> cuda{std::make_shared<test::FakeCudaState>()};
    std::shared_ptr<test::FakeGpuCodecState> codec{std::make_shared<test::FakeGpuCodecState>()};
    test::FakeCudaDriverApi* driver{};
    detail::CudaVmmBackend* storage{};
    std::unique_ptr<detail::CudaVmmBackend> backend{};
    Fixture() {
        auto driver_owner = std::make_unique<test::FakeCudaDriverApi>(cuda);
        driver = driver_owner.get();
        auto codec_owner = std::make_unique<test::FakeNvcompLz4Api>(*driver, codec);
        backend = std::move(detail::CudaVmmBackend::create(std::move(driver_owner), 0,
                                                           std::move(codec_owner)))
                      .value();
        storage = backend.get();
    }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
    Runtime runtime(std::uint32_t capacity = 64U, bool policy_enabled = false) {
        auto config = test::test_runtime_config();
        config.required_capabilities.host_tier = false;
        config.budgets.host = {};
        config.budgets.gpu = {ByteSize{1024ULL * 1024ULL}, ByteSize{512ULL * 1024ULL},
                              ByteSize{512ULL * 1024ULL}};
        config.preferred_chunk_size = ByteSize{4096U};
        config.maximum_pending_completions = capacity;
        if (policy_enabled) {
            config.policy.mode = PolicyMode::gpu_resident;
        }
        return std::move(testing::RuntimeAccess::create_with_backend(config, std::move(backend)))
            .value();
    }
};

constexpr MemoryRange whole{ByteOffset{}, ByteSize{4096U}};
void conserved(test::Runner& runner, Runtime& runtime) {
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
}
void finish(test::Runner& runner, Runtime& runtime, Buffer& buffer, Fixture& fixture) {
    const auto closed = buffer.close();
    VRAMZ_CHECK(runner, closed || closed.error().code == ErrorCode::stale_handle);
    VRAMZ_CHECK(runner, runtime.shutdown());
    conserved(runner, runtime);
    VRAMZ_CHECK(runner, fixture.driver->empty());
    VRAMZ_CHECK(runner, runtime.stats().completions.pending_completion_count == 0U);
}

void read_lifetime(test::Runner& runner) {
    runner.begin("deferred read pins survive handle destruction, NOT_READY and buffer close");
    Fixture fixture;
    auto runtime = fixture.runtime();
    auto buffer = std::move(runtime.allocate(whole.length)).value();
    const auto before = testing::chunk_snapshot(buffer, 0U).value();
    CompletionToken token{};
    {
        auto lease = std::move(buffer.acquire(whole)).value();
        auto pending = std::move(lease.defer()).value();
        token = pending.token();
        VRAMZ_CHECK(runner, !lease.valid());
        VRAMZ_CHECK(runner, !lease.device_span());
        VRAMZ_CHECK(runner, !lease.close());
        auto moved = std::move(pending);
        // The API explicitly defines empty() after move; this regression must inspect that state.
        // NOLINTNEXTLINE(bugprone-use-after-move): valid moved-from query, not resource access.
        VRAMZ_CHECK(runner, pending.empty());
        VRAMZ_CHECK(runner, moved.token() == token);
    }
    const auto blocked = testing::migrate(buffer, 0U, RepresentationState::gpu_compressed);
    VRAMZ_CHECK(runner, !blocked && blocked.error().code == ErrorCode::busy);
    const auto not_ready = runtime.poll_completion(token);
    VRAMZ_CHECK(runner, not_ready && !not_ready.value());
    VRAMZ_CHECK(runner, buffer.stats().active_read_leases == 1U);
    VRAMZ_CHECK(runner, testing::chunk_snapshot(buffer, 0U).value().authoritative.content ==
                            before.authoritative.content);
    const auto unmaps = fixture.driver->count(test::CudaCall::unmap);
    VRAMZ_CHECK(runner, !buffer.close());
    const auto shutdown = runtime.shutdown();
    VRAMZ_CHECK(runner, !shutdown && shutdown.error().code == ErrorCode::busy);
    VRAMZ_CHECK(runner, fixture.driver->count(test::CudaCall::unmap) == unmaps);
    VRAMZ_CHECK(runner, fixture.driver->mark_completion_ready(token.id));
    VRAMZ_CHECK(runner, runtime.poll_completion(token).value());
    const auto duplicate = runtime.poll_completion(token);
    VRAMZ_CHECK(runner, !duplicate && duplicate.error().code == ErrorCode::stale_handle);
    finish(runner, runtime, buffer, fixture);
}

void write_lifetime(test::Runner& runner, bool fail_finalization) {
    runner.begin(fail_finalization
                     ? "completed writer readback failure poisons without stale valid CRC"
                     : "pending writer finalizes generation and CRC only after completion");
    Fixture fixture;
    auto runtime = fixture.runtime();
    auto buffer = std::move(runtime.allocate(whole.length)).value();
    auto lease = std::move(buffer.acquire(whole, {AccessMode::read_write})).value();
    const auto address = lease.device_span().value().address;
    std::array<std::byte, 4096U> bytes{};
    // Preserve a prior immediate reference write: deferral still must finalize later writes.
    VRAMZ_CHECK(runner, testing::write_bytes(lease, {}, bytes));
    const auto before = testing::chunk_snapshot(buffer, 0U).value();
    auto pending = std::move(lease.defer()).value();
    bytes.fill(std::byte{0x3a});
    VRAMZ_CHECK(runner, fixture.driver->push_context(detail::CudaContext{1U}));
    VRAMZ_CHECK(runner, fixture.driver->copy_to_device(address, bytes));
    VRAMZ_CHECK(runner, fixture.driver->pop_context());
    VRAMZ_CHECK(runner, !runtime.poll_completion(pending.token()).value());
    const auto during = testing::chunk_snapshot(buffer, 0U).value();
    VRAMZ_CHECK(runner,
                during.write_pin && during.authoritative.content == before.authoritative.content);
    VRAMZ_CHECK(runner,
                during.authoritative.metadata.crc32c == before.authoritative.metadata.crc32c);
    VRAMZ_CHECK(runner, !buffer.acquire(whole));
    VRAMZ_CHECK(runner, fixture.driver->mark_completion_ready(pending.token().id));
    if (fail_finalization) {
        fixture.driver->inject(test::CudaCall::device_to_host);
    }
    const auto completed = runtime.poll_completion(pending.token());
    const auto after = testing::chunk_snapshot(buffer, 0U).value();
    VRAMZ_CHECK(runner, !after.write_pin);
    if (fail_finalization) {
        VRAMZ_CHECK(runner, !completed && after.lifecycle == LifecycleState::poisoned);
        VRAMZ_CHECK(runner, !buffer.acquire(whole));
    } else {
        VRAMZ_CHECK(runner, completed && completed.value());
        VRAMZ_CHECK(runner, after.authoritative.content.generation ==
                                before.authoritative.content.generation + 1U);
        auto read = std::move(buffer.acquire(whole)).value();
        std::array<std::byte, 4096U> output{};
        VRAMZ_CHECK(runner, testing::read_bytes(read, {}, output));
        VRAMZ_CHECK(runner, output == bytes);
        VRAMZ_CHECK(runner, read.close());
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_raw));
    }
    finish(runner, runtime, buffer, fixture);
}

void capacity_and_identity(test::Runner& runner) {
    runner.begin(
        "bounded capacity, foreign token rejection and definite creation failure retain Lease");
    Fixture fixture;
    auto runtime = fixture.runtime(1U);
    auto buffer = std::move(runtime.allocate(whole.length)).value();
    auto first = std::move(buffer.acquire(whole)).value();
    auto second = std::move(buffer.acquire(whole)).value();
    fixture.driver->inject(test::CudaCall::completion_create);
    VRAMZ_CHECK(runner, !first.defer() && first.valid());
    auto pending = std::move(first.defer()).value();
    const auto creates = fixture.driver->count(test::CudaCall::completion_create);
    VRAMZ_CHECK(runner, !second.defer() && second.valid());
    VRAMZ_CHECK(runner, fixture.driver->count(test::CudaCall::completion_create) == creates);
    auto foreign = pending.token();
    foreign.runtime = RuntimeId{foreign.runtime.value() + 1U};
    VRAMZ_CHECK(runner, !runtime.poll_completion(foreign));
    foreign = pending.token();
    foreign.backend = BackendId{99U};
    VRAMZ_CHECK(runner, !runtime.poll_completion(foreign));
    VRAMZ_CHECK(runner, second.close());
    VRAMZ_CHECK(runner, !buffer.close());
    const auto blocked_shutdown = runtime.shutdown();
    VRAMZ_CHECK(runner, !blocked_shutdown && blocked_shutdown.error().code == ErrorCode::busy);
    VRAMZ_CHECK(runner, fixture.driver->mark_completion_ready(pending.token().id));
    VRAMZ_CHECK(runner, runtime.poll_ready_completions().value() == 1U);
    finish(runner, runtime, buffer, fixture);
}

void recoverable_pairs(test::Runner& runner) {
    for (const bool writer : {false, true}) {
        for (const auto fault :
             {test::CudaCall::completion_query, test::CudaCall::completion_release}) {
            runner.begin("completion failure plus buffer close retains exact ownership for retry");
            Fixture fixture;
            auto runtime = fixture.runtime();
            auto buffer = std::move(runtime.allocate(whole.length)).value();
            auto lease = std::move(buffer.acquire(whole, {writer ? AccessMode::read_write
                                                                 : AccessMode::read_only}))
                             .value();
            auto pending = std::move(lease.defer()).value();
            VRAMZ_CHECK(runner, fixture.driver->mark_completion_ready(pending.token().id));
            fixture.driver->inject(fault);
            const auto failed = runtime.poll_completion(pending.token());
            VRAMZ_CHECK(runner, !failed && failed.error().code == ErrorCode::backend_failure);
            VRAMZ_CHECK(runner,
                        failed.error().operation == (fault == test::CudaCall::completion_query
                                                         ? OperationId::query_completion
                                                         : OperationId::release_completion));
            VRAMZ_CHECK(runner, !buffer.close());
            const auto snapshot = testing::chunk_snapshot(buffer, 0U).value();
            VRAMZ_CHECK(runner, snapshot.write_pin || snapshot.read_pins == 1U);
            VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == ByteSize{});
            conserved(runner, runtime);
            VRAMZ_CHECK(runner, runtime.poll_completion(pending.token()).value());
            finish(runner, runtime, buffer, fixture);
        }
    }
}

void registry_boundaries(test::Runner& runner) {
    runner.begin("maximum registry capacity and exhausted IDs never abandon a live Lease");
    Fixture fixture;
    auto runtime = fixture.runtime();
    auto buffer = std::move(runtime.allocate(whole.length)).value();
    std::array<std::optional<PendingLeaseRelease>, 64U> pending{};
    for (auto& slot : pending) {
        auto lease = std::move(buffer.acquire(whole)).value();
        slot.emplace(std::move(lease.defer()).value());
    }
    auto excess = std::move(buffer.acquire(whole)).value();
    VRAMZ_CHECK(runner, !excess.defer() && excess.valid());
    VRAMZ_CHECK(runner, buffer.stats().active_read_leases == 65U);
    for (const auto& slot : pending) {
        VRAMZ_CHECK(runner, fixture.driver->mark_completion_ready(slot->token().id));
    }
    VRAMZ_CHECK(runner, runtime.poll_ready_completions().value() == 64U);
    {
        const std::scoped_lock lock{fixture.cuda->mutex};
        fixture.cuda->next_completion_id = std::numeric_limits<std::uint64_t>::max();
    }
    const auto overflow = excess.defer();
    VRAMZ_CHECK(runner, !overflow && overflow.error().code == ErrorCode::arithmetic_overflow &&
                            excess.valid());
    VRAMZ_CHECK(runner, excess.close());
    finish(runner, runtime, buffer, fixture);
    for (const auto capacity : {0U, 65U}) {
        auto invalid = test::test_runtime_config();
        invalid.maximum_pending_completions = capacity;
        const auto rejected = Runtime::create(invalid);
        VRAMZ_CHECK(runner, !rejected && rejected.error().code == ErrorCode::invalid_argument);
    }
}

void integrated_model(test::Runner& runner) {
    runner.begin(
        "M6 deterministic 2048-cycle mixed completion/compression/restore model seed 12648430");
    Fixture fixture;
    auto runtime = fixture.runtime(64U, true);
    auto buffer = std::move(runtime.allocate(whole.length)).value();
    const auto stable = testing::chunk_snapshot(buffer, 0U).value().authoritative.address;
    constexpr std::uint64_t seed = 12648430U;
    std::uint64_t random = seed;
    for (std::uint32_t step = 0U; step < 2048U; ++step) {
        // Deterministic bit mixing is test scheduling, never size/accounting arithmetic.
        random ^= random << 13U;
        random ^= random >> 7U;
        random ^= random << 17U;
        const bool writable = random % 3U == 0U;
        runner.trace(seed, step);
        auto lease = std::move(buffer.acquire(whole, {writable ? AccessMode::read_write
                                                               : AccessMode::read_only}))
                         .value();
        const auto before = testing::chunk_snapshot(buffer, 0U).value().authoritative.content;
        auto pending = std::move(lease.defer()).value();
        VRAMZ_CHECK(runner, !runtime.poll_completion(pending.token()).value());
        VRAMZ_CHECK(runner, !testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        if (random % 7U == 0U) {
            fixture.driver->inject(test::CudaCall::completion_query);
            VRAMZ_CHECK(runner, !runtime.poll_completion(pending.token()));
        }
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, fixture.driver->mark_completion_ready(pending.token().id));
        VRAMZ_CHECK(runner, runtime.poll_ready_completions().value() == 1U);
        const auto after = testing::chunk_snapshot(buffer, 0U).value();
        VRAMZ_CHECK(runner, after.authoritative.address == stable);
        VRAMZ_CHECK(runner, after.authoritative.content.generation ==
                                before.generation + (writable ? 1U : 0U));
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_raw));
        conserved(runner, runtime);
        if (step % 64U == 0U) {
            auto extra = std::move(runtime.allocate(whole.length)).value();
            std::array<std::byte, 4096U> expected{};
            expected.fill(static_cast<std::byte>(random % 256U));
            auto writer = std::move(extra.acquire(whole, {AccessMode::read_write})).value();
            VRAMZ_CHECK(runner, testing::write_bytes(writer, {}, expected));
            VRAMZ_CHECK(runner, writer.close());
            conserved(runner, runtime);
            testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
            const auto pressure = testing::RuntimeAccess::reclaim(runtime, ByteSize{128U}, true);
            VRAMZ_CHECK(runner, pressure || pressure.error().code == ErrorCode::out_of_gpu_memory);
            conserved(runner, runtime);
            auto reader = std::move(extra.acquire(whole)).value();
            std::array<std::byte, 4096U> actual{};
            VRAMZ_CHECK(runner, testing::read_bytes(reader, {}, actual));
            VRAMZ_CHECK(runner, actual == expected && reader.close());
            conserved(runner, runtime);
            fixture.driver->inject(test::CudaCall::unmap);
            VRAMZ_CHECK(runner, !extra.close());
            VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == whole.length);
            conserved(runner, runtime);
            VRAMZ_CHECK(runner, extra.close());
            VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == ByteSize{});
            conserved(runner, runtime);
        }
    }
    VRAMZ_CHECK(runner, runtime.stats().completions.completed_deferred_releases == 2048U);
    finish(runner, runtime, buffer, fixture);
}

void concurrency(test::Runner& runner) {
    runner.begin("bounded concurrent completion pollers and statistics preserve single release");
    Fixture fixture;
    auto runtime = fixture.runtime();
    auto buffer = std::move(runtime.allocate(whole.length)).value();
    for (std::uint32_t iteration = 0U; iteration < 128U; ++iteration) {
        auto lease = std::move(buffer.acquire(whole)).value();
        auto pending = std::move(lease.defer()).value();
        VRAMZ_CHECK(runner, fixture.driver->mark_completion_ready(pending.token().id));
        std::atomic<std::uint32_t> completed{};
        const auto poll = [&] {
            const auto result = runtime.poll_completion(pending.token());
            if (result && result.value()) {
                completed.fetch_add(1U);
            }
            static_cast<void>(runtime.stats());
        };
        std::thread first{poll};
        std::thread second{poll};
        first.join();
        second.join();
        VRAMZ_CHECK(runner, completed.load() == 1U);
        conserved(runner, runtime);
    }
    finish(runner, runtime, buffer, fixture);
}

class DeferredAccessObserver final : public testing::PolicySelectionObserver {
  public:
    explicit DeferredAccessObserver(Buffer& buffer) noexcept : buffer_(buffer) {}
    void on_selected(const ChunkSnapshot& snapshot,
                     const policy::Proposal& proposal) noexcept override {
        if (pending || proposal.action != PolicyAction::compress_gpu) {
            return;
        }
        auto lease = buffer_.acquire(whole);
        if (!lease) {
            std::terminate();
        }
        auto deferred = lease.value().defer();
        if (!deferred) {
            std::terminate();
        }
        pending.emplace(std::move(deferred).value());
        revision = snapshot.access_revision;
    }
    std::optional<PendingLeaseRelease> pending{};
    std::uint64_t revision{};

  private:
    Buffer& buffer_;
};

void stale_policy(test::Runner& runner) {
    runner.begin("selected policy victim invalidated by deferred reader until a fresh cycle");
    Fixture fixture;
    auto runtime = fixture.runtime(64U, true);
    auto buffer = std::move(runtime.allocate(whole.length)).value();
    testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
    DeferredAccessObserver observer{buffer};
    const auto reclaimed = testing::RuntimeAccess::reclaim(runtime, {}, true, &observer);
    VRAMZ_CHECK(runner, !reclaimed && observer.pending.has_value());
    if (observer.pending) {
        const auto snapshot = testing::chunk_snapshot(buffer, 0U).value();
        VRAMZ_CHECK(runner, snapshot.access_revision == observer.revision + 1U);
        VRAMZ_CHECK(runner, snapshot.read_pins == 1U &&
                                snapshot.authoritative.state == RepresentationState::gpu_raw);
        VRAMZ_CHECK(runner, runtime.stats().policy.stale_proposal_rejections == 1U);
        VRAMZ_CHECK(runner, !testing::RuntimeAccess::reclaim(runtime, {}, true));
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, fixture.driver->mark_completion_ready(observer.pending->token().id));
        VRAMZ_CHECK(runner, runtime.poll_ready_completions().value() == 1U);
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        static_cast<void>(testing::RuntimeAccess::reclaim(runtime, ByteSize{512U}, true));
        VRAMZ_CHECK(runner, testing::chunk_snapshot(buffer, 0U).value().authoritative.state ==
                                RepresentationState::gpu_compressed);
    }
    finish(runner, runtime, buffer, fixture);
}

void mixed_shutdown(test::Runner& runner) {
    runner.begin(
        "full-stack shutdown preserves mixed RAW/compressed/debt/VA/fence/stream ownership");
    Fixture fixture;
    auto runtime = fixture.runtime(64U, true);
    auto raw = std::move(runtime.allocate(whole.length)).value();
    auto compressed = std::move(runtime.allocate(whole.length)).value();
    auto debt = std::move(runtime.allocate(whole.length)).value();
    VRAMZ_CHECK(runner, testing::migrate(compressed, 0U, RepresentationState::gpu_compressed));
    fixture.driver->inject(test::CudaCall::unmap);
    VRAMZ_CHECK(runner, !debt.close());
    VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == whole.length);
    auto lease = std::move(raw.acquire(whole)).value();
    auto pending = std::move(lease.defer()).value();
    VRAMZ_CHECK(runner, !raw.close());
    const auto unmaps = fixture.driver->count(test::CudaCall::unmap);
    const auto releases = fixture.driver->count(test::CudaCall::release);
    const auto frees = fixture.driver->count(test::CudaCall::free_address);
    const auto contexts = fixture.driver->count(test::CudaCall::release_primary);
    VRAMZ_CHECK(runner, !runtime.shutdown());
    VRAMZ_CHECK(runner, fixture.driver->count(test::CudaCall::unmap) == unmaps);
    VRAMZ_CHECK(runner, fixture.driver->count(test::CudaCall::release) == releases);
    VRAMZ_CHECK(runner, fixture.driver->count(test::CudaCall::free_address) == frees);
    VRAMZ_CHECK(runner, fixture.driver->count(test::CudaCall::release_primary) == contexts);
    conserved(runner, runtime);
    VRAMZ_CHECK(runner, fixture.driver->mark_completion_ready(pending.token().id));
    VRAMZ_CHECK(runner, runtime.poll_ready_completions().value() == 1U);
    VRAMZ_CHECK(runner, debt.close());
    VRAMZ_CHECK(runner, compressed.close());
    finish(runner, runtime, raw, fixture);
    VRAMZ_CHECK(runner, fixture.codec->observation.streams == 0U);
}

void capabilities(test::Runner& runner) {
    runner.begin(
        "build, Mock, fake RAW and fake compressed capabilities do not imply hardware validation");
    const auto build = build_capabilities();
    VRAMZ_CHECK(runner, !build.real_gpu_execution_enabled &&
                            build.hardware == HardwareValidationState::not_tested);
    auto mock = std::move(Runtime::create(test::test_runtime_config())).value();
    VRAMZ_CHECK(runner, mock.capabilities().gpu_raw && mock.capabilities().gpu_compressed &&
                            mock.capabilities().host_raw);
    VRAMZ_CHECK(runner, !mock.capabilities().external_async_completion);
    VRAMZ_CHECK(runner, mock.shutdown());
    for (const bool codec : {false, true}) {
        Fixture fixture;
        if (!codec) {
            fixture.backend.reset();
            auto driver = std::make_unique<test::FakeCudaDriverApi>(fixture.cuda);
            fixture.driver = driver.get();
            fixture.backend = std::move(detail::CudaVmmBackend::create(std::move(driver))).value();
        }
        auto runtime = fixture.runtime();
        const auto actual = runtime.capabilities();
        VRAMZ_CHECK(runner, actual.gpu_raw && actual.gpu_compressed == codec && !actual.host_raw);
        VRAMZ_CHECK(runner, actual.external_async_completion && actual.stable_virtual_address);
        VRAMZ_CHECK(runner, actual.hardware == HardwareValidationState::not_tested);
        VRAMZ_CHECK(runner, actual.minimum_allocation_granularity == ByteSize{64U});
        VRAMZ_CHECK(runner, actual.recommended_allocation_granularity == ByteSize{256U});
        VRAMZ_CHECK(runner, runtime.shutdown());
        VRAMZ_CHECK(runner, fixture.driver->empty());
    }
}

void death_cases(test::Runner& runner) {
    for (const auto fault : {test::CudaCall::completion_create, test::CudaCall::completion_query,
                             test::CudaCall::completion_release, test::CudaCall::count}) {
        runner.begin("bounded fail-stop on ambiguous completion or final Runtime destruction");
        const auto child = ::fork();
        VRAMZ_CHECK(runner, child >= 0);
        if (child == 0) {
            static_cast<void>(::alarm(5U));
            Fixture fixture;
            auto runtime = fixture.runtime();
            auto buffer = std::move(runtime.allocate(whole.length)).value();
            auto lease = std::move(buffer.acquire(whole)).value();
            // Child-local audit bindings cannot escape: both outcomes terminate via _Exit.
            static test::FakeCudaDriverApi* dying_driver{};
            static Runtime* dying_runtime{};
            static Buffer* dying_buffer{};
            static std::uint64_t expected_unmaps{};
            dying_driver = fixture.driver;
            dying_runtime = &runtime;
            dying_buffer = &buffer;
            expected_unmaps = fixture.driver->count(test::CudaCall::unmap);
            std::set_terminate([]() noexcept {
                const auto snapshot = testing::chunk_snapshot(*dying_buffer, 0U);
                const bool valid =
                    snapshot && (snapshot.value().read_pins != 0U || snapshot.value().write_pin) &&
                    snapshot.value().lifecycle == LifecycleState::live &&
                    dying_driver->count(test::CudaCall::unmap) == expected_unmaps &&
                    dying_runtime->stats().gpu.cleanup_debt == ByteSize{};
                std::_Exit(valid ? 86 : 87);
            });
            if (fault == test::CudaCall::completion_create) {
                fixture.driver->inject(fault, test::CudaFaultMode::after);
                static_cast<void>(lease.defer());
            } else {
                auto pending = std::move(lease.defer()).value();
                if (fault == test::CudaCall::count) {
                    // Explicitly invoke the destructor in this process, which must never return.
                    runtime.~Runtime();
                } else {
                    VRAMZ_CHECK(runner, fixture.driver->mark_completion_ready(pending.token().id));
                    fixture.driver->inject(fault, test::CudaFaultMode::after);
                    static_cast<void>(runtime.poll_completion(pending.token()));
                }
            }
            std::_Exit(88);
        }
        if (child > 0) {
            int status{};
            pid_t waited{};
            do {
                waited = ::waitpid(child, &status, 0);
            } while (waited < 0 && errno == EINTR);
            VRAMZ_CHECK(runner, waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 86);
        }
    }
}
} // namespace

int main() {
    test::Runner runner;
    read_lifetime(runner);
    write_lifetime(runner, false);
    write_lifetime(runner, true);
    capacity_and_identity(runner);
    recoverable_pairs(runner);
    registry_boundaries(runner);
    integrated_model(runner);
    concurrency(runner);
    stale_policy(runner);
    mixed_shutdown(runner);
    capabilities(runner);
    death_cases(runner);
    return runner.finish();
}
