#include "../test_support.hpp"
#include "vramz/detail/simulation.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <limits>
#include <thread>
#include <vector>

using namespace vramz;

namespace {

[[nodiscard]] RuntimeConfig policy_config(PolicyMode mode = PolicyMode::gpu_resident) noexcept {
    auto config = test::test_runtime_config();
    config.budgets =
        MemoryBudgets{TierBudget{ByteSize{32768U}, ByteSize{20480U}, ByteSize{12288U}},
                      TierBudget{ByteSize{32768U}, ByteSize{20480U}, ByteSize{12288U}}};
    config.preferred_chunk_size = ByteSize{4096U};
    config.policy.mode = mode;
    return config;
}

void conserved(test::Runner& runner, Runtime& runtime) {
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
}

void empty(test::Runner& runner, Runtime& runtime) {
    conserved(runner, runtime);
    VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U);
    for (const auto usage : {runtime.stats().gpu, runtime.stats().host}) {
        VRAMZ_CHECK(runner, usage.committed == ByteSize{} && usage.reserved == ByteSize{} &&
                                usage.staging == ByteSize{} && usage.workspace == ByteSize{} &&
                                usage.cleanup_debt == ByteSize{});
    }
}

void write(test::Runner& runner, Buffer& buffer, simulation::Dataset dataset) {
    std::array<std::byte, 4096U> data{};
    simulation::fill(dataset, 0U, 0U, data);
    auto lease = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}},
                                AcquireOptions{AccessMode::read_write});
    VRAMZ_CHECK(runner, lease);
    if (lease) {
        VRAMZ_CHECK(runner, testing::write_bytes(lease.value(), ByteOffset{}, data));
        VRAMZ_CHECK(runner, lease.value().close());
    }
}

void verify(test::Runner& runner, Buffer& buffer, simulation::Dataset dataset) {
    std::array<std::byte, 4096U> expected{};
    std::array<std::byte, 4096U> actual{};
    simulation::fill(dataset, 0U, 0U, expected);
    auto lease = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
    VRAMZ_CHECK(runner, lease);
    if (lease) {
        VRAMZ_CHECK(runner, testing::read_bytes(lease.value(), ByteOffset{}, actual));
        VRAMZ_CHECK(runner, actual == expected);
        VRAMZ_CHECK(runner, lease.value().close());
    }
}

void round_trip(test::Runner& runner, simulation::Dataset dataset) {
    auto runtime = std::move(Runtime::create(policy_config())).value();
    auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
    write(runner, buffer, dataset);
    testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
    VRAMZ_CHECK(runner, runtime.reclaim_to_target(ByteSize{2048U}));
    const auto compressed = testing::chunk_snapshot(buffer, 0U).value();
    VRAMZ_CHECK(runner, compressed.authoritative.state == RepresentationState::gpu_compressed);
    conserved(runner, runtime);
    verify(runner, buffer, dataset);
    const auto raw = testing::chunk_snapshot(buffer, 0U).value();
    VRAMZ_CHECK(runner, raw.authoritative.content == compressed.authoritative.content);
    VRAMZ_CHECK(runner,
                raw.authoritative.metadata.crc32c == compressed.authoritative.metadata.crc32c);
    VRAMZ_CHECK(runner, runtime.stats().policy.restore_count == 1U);
    VRAMZ_CHECK(runner, runtime.stats().policy.logical_host_bytes == ByteSize{});
    conserved(runner, runtime);
    VRAMZ_CHECK(runner, buffer.close());
    VRAMZ_CHECK(runner, runtime.shutdown());
    empty(runner, runtime);
}

class SelectionPause final : public testing::PolicySelectionObserver {
  public:
    void on_selected(const ChunkSnapshot& observed,
                     const policy::Proposal& proposal) noexcept override {
        snapshot = observed;
        selected_proposal = proposal;
        selected.arrive_and_wait();
        accessed.arrive_and_wait();
    }

    std::barrier<> selected{2};
    std::barrier<> accessed{2};
    ChunkSnapshot snapshot{};
    policy::Proposal selected_proposal{};
};

class ReadFallback final : public testing::PolicySelectionObserver {
  public:
    explicit ReadFallback(Buffer& buffer) noexcept : buffer_(buffer) {}
    void on_selected(const ChunkSnapshot& snapshot,
                     const policy::Proposal& proposal) noexcept override {
        if (proposal.action != PolicyAction::host_fallback) {
            return;
        }
        ++calls;
        original = snapshot;
        auto lease = buffer_.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
        if (!lease) {
            valid = false;
            return;
        }
        std::array<std::byte, 4096U> expected{};
        std::array<std::byte, 4096U> actual{};
        simulation::fill(simulation::Dataset::random, 0U, 0U, expected);
        valid =
            valid && testing::read_bytes(lease.value(), ByteOffset{}, actual) && actual == expected;
        const auto closed = lease.value().close();
        valid = valid && closed;
    }

    std::uint32_t calls{};
    bool valid{true};
    ChunkSnapshot original{};

  private:
    Buffer& buffer_;
};

class SelectionTrace final : public testing::PolicySelectionObserver {
  public:
    explicit SelectionTrace(Runtime& runtime, Buffer* read_victim = nullptr) noexcept
        : runtime_(runtime), read_victim_(read_victim) {}

    void on_selected(const ChunkSnapshot& snapshot,
                     const policy::Proposal& proposal) noexcept override {
        if (calls == snapshots.size()) {
            valid = false;
            return;
        }
        snapshots[calls] = snapshot;
        proposals[calls] = proposal;
        ++calls;
        if (read_victim_ == nullptr || calls != 1U) {
            return;
        }
        if (proposal.action != PolicyAction::compress_gpu || snapshot.id != ChunkId{}) {
            valid = false;
            return;
        }
        auto lease = read_victim_->acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
        if (!lease) {
            valid = false;
            return;
        }
        std::array<std::byte, 4096U> bytes{};
        const auto read = testing::read_bytes(lease.value(), ByteOffset{}, bytes);
        const auto closed = lease.value().close();
        read_succeeded = read && closed && std::all_of(bytes.begin(), bytes.end(), [](std::byte b) {
                             return b == std::byte{};
                         });
        const auto after = testing::chunk_snapshot(*read_victim_, 0U);
        const auto metadata = testing::RuntimeAccess::policy_metadata(*read_victim_, 0U);
        if (!after || !metadata) {
            valid = false;
            return;
        }
        after_read = after.value();
        temperature_after_read = policy::temperature(
            metadata.value(), runtime_.stats().policy.access_epoch, PolicyTuning{});
    }

    std::array<ChunkSnapshot, 8U> snapshots{};
    std::array<policy::Proposal, 8U> proposals{};
    std::size_t calls{};
    bool valid{true};
    bool read_succeeded{};
    ChunkSnapshot after_read{};
    Temperature temperature_after_read{Temperature::cold};

  private:
    Runtime& runtime_;
    Buffer* read_victim_;
};

} // namespace

int main() {
    test::Runner runner;
    runner.begin(
        "DISABLED preserves M2 admission and never automatically compresses restores or offloads");
    {
        auto runtime = std::move(Runtime::create(policy_config(PolicyMode::disabled))).value();
        auto buffer = std::move(runtime.allocate(ByteSize{16384U})).value();
        VRAMZ_CHECK(runner, runtime.reclaim_to_target(ByteSize{}));
        const auto allocation = runtime.allocate(ByteSize{8192U});
        VRAMZ_CHECK(runner, !allocation && allocation.error().code == ErrorCode::out_of_gpu_memory);
        VRAMZ_CHECK(runner, runtime.stats().gpu.committed == ByteSize{16384U});
        VRAMZ_CHECK(runner, runtime.stats().policy.policy_transitions == 0U &&
                                runtime.stats().policy.policy_cycles == 0U);
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        auto lease = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
        VRAMZ_CHECK(runner, lease && lease.value().close());
        VRAMZ_CHECK(runner, runtime.stats().policy.restore_count == 0U);
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin(
        "automatic real-byte round trips preserve CRC and generation for five useful datasets");
    for (const auto dataset :
         {simulation::Dataset::zeros, simulation::Dataset::repeated, simulation::Dataset::sparse,
          simulation::Dataset::integers, simulation::Dataset::fp_like}) {
        round_trip(runner, dataset);
    }

    runner.begin(
        "unhelpful compression rolls back verified staging before commit and preserves RAW bytes");
    {
        auto runtime = std::move(Runtime::create(policy_config())).value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        write(runner, buffer, simulation::Dataset::random);
        const auto before = testing::chunk_snapshot(buffer, 0U).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        VRAMZ_CHECK(runner, !runtime.reclaim_to_target(ByteSize{2048U}));
        const auto after = testing::chunk_snapshot(buffer, 0U).value();
        VRAMZ_CHECK(runner, before.authoritative.resource == after.authoritative.resource &&
                                after.lifecycle == LifecycleState::live &&
                                after.cleanup_resource_count == 0U);
        const auto metadata = testing::RuntimeAccess::policy_metadata(buffer, 0U).value();
        VRAMZ_CHECK(runner, metadata.compressibility == Compressibility::incompressible);
        const auto attempts = runtime.stats().policy.compression_attempts;
        for (std::uint32_t index = 0U; index < 8U; ++index) {
            VRAMZ_CHECK(runner, !runtime.reclaim_to_target(ByteSize{2048U}));
            conserved(runner, runtime);
        }
        VRAMZ_CHECK(runner, runtime.stats().policy.compression_attempts == attempts);
        verify(runner, buffer, simulation::Dataset::random);
        write(runner, buffer, simulation::Dataset::repeated);
        VRAMZ_CHECK(runner,
                    testing::RuntimeAccess::policy_metadata(buffer, 0U).value().compressibility ==
                        Compressibility::unknown);
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        VRAMZ_CHECK(runner, runtime.reclaim_to_target(ByteSize{2048U}));
        verify(runner, buffer, simulation::Dataset::repeated);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("pin exclusion and deterministic tie break protect the active lease");
    {
        auto runtime = std::move(Runtime::create(policy_config())).value();
        auto a = std::move(runtime.allocate(ByteSize{4096U})).value();
        auto b = std::move(runtime.allocate(ByteSize{4096U})).value();
        auto pinned = a.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
        VRAMZ_CHECK(runner, pinned);
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        VRAMZ_CHECK(runner, runtime.reclaim_to_target(ByteSize{5000U}));
        VRAMZ_CHECK(runner, testing::chunk_snapshot(a, 0U).value().authoritative.state ==
                                RepresentationState::gpu_raw);
        VRAMZ_CHECK(runner, testing::chunk_snapshot(b, 0U).value().authoritative.state ==
                                RepresentationState::gpu_compressed);
        VRAMZ_CHECK(runner, pinned.value().close());
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, a.close());
        VRAMZ_CHECK(runner, b.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("compression at full normal admission retains full staging reserve and never "
                 "commits above normal limit");
    {
        auto runtime = std::move(Runtime::create(policy_config())).value();
        auto buffer = std::move(runtime.allocate(ByteSize{20480U})).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        auto added = runtime.allocate(ByteSize{4096U});
        VRAMZ_CHECK(runner, added);
        VRAMZ_CHECK(runner, runtime.stats().gpu.committed <= ByteSize{20480U});
        VRAMZ_CHECK(runner, runtime.stats().gpu.peak_charged <= ByteSize{32768U});
        VRAMZ_CHECK(runner, runtime.stats().policy.compression_successes > 0U);
        conserved(runner, runtime);
        if (added) {
            VRAMZ_CHECK(runner, added.value().close());
        }
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("bounded candidate scanning cannot falsely authorize host fallback");
    {
        auto config = policy_config(PolicyMode::gpu_resident_with_host_fallback);
        config.policy.tuning.maximum_candidates = 1U;
        auto runtime = std::move(Runtime::create(config)).value();
        auto buffer = std::move(runtime.allocate(ByteSize{8192U})).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        VRAMZ_CHECK(runner, !runtime.reclaim_to_target(ByteSize{1024U}));
        VRAMZ_CHECK(runner, runtime.stats().policy.candidates_inspected == 1U);
        VRAMZ_CHECK(runner, runtime.stats().policy.host_fallback_count == 0U);
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("recoverable compression failure tries another candidate within bounded policy");
    {
        auto runtime = std::move(Runtime::create(policy_config())).value();
        auto buffer = std::move(runtime.allocate(ByteSize{12288U})).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        testing::inject_failure(runtime, FaultPoint::compression);
        VRAMZ_CHECK(runner, runtime.reclaim_to_target(ByteSize{9000U}));
        VRAMZ_CHECK(runner, runtime.stats().policy.compression_failures == 1U);
        VRAMZ_CHECK(runner, runtime.stats().policy.compression_successes >= 1U);
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("threshold rollback plus failed cleanup retains exact corroborated charge");
    {
        auto runtime = std::move(Runtime::create(policy_config())).value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        write(runner, buffer, simulation::Dataset::random);
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        testing::inject_failure(runtime, FaultPoint::rollback_destination_cleanup);
        VRAMZ_CHECK(runner, !runtime.reclaim_to_target(ByteSize{2048U}));
        const auto snapshot = testing::chunk_snapshot(buffer, 0U).value();
        VRAMZ_CHECK(runner, snapshot.cleanup_resource_count == 1U &&
                                snapshot.authoritative.state == RepresentationState::gpu_raw);
        const auto cleanup = testing::RuntimeAccess::cleanup_resource(buffer, 0U, 0U).value();
        const auto exact =
            testing::RuntimeAccess::backend(runtime).allocation(cleanup.resource).value();
        VRAMZ_CHECK(runner, cleanup.charge == exact.charge &&
                                runtime.stats().gpu.cleanup_debt == exact.charge);
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("corroborated metadata contract failure remains POISONED and excluded by policy");
    {
        auto runtime = std::move(Runtime::create(policy_config())).value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        testing::inject_failure(runtime, FaultPoint::receipt_logical_size_mismatch);
        VRAMZ_CHECK(runner, !runtime.reclaim_to_target(ByteSize{2048U}));
        VRAMZ_CHECK(runner, testing::chunk_snapshot(buffer, 0U).value().lifecycle ==
                                LifecycleState::poisoned);
        const auto attempts = runtime.stats().policy.compression_attempts;
        VRAMZ_CHECK(runner, !runtime.reclaim_to_target(ByteSize{2048U}));
        VRAMZ_CHECK(runner, runtime.stats().policy.compression_attempts == attempts);
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin(
        "actual pressure-driven thrashing is detected and temporarily defers recompression");
    {
        auto config = policy_config();
        config.policy.tuning.minimum_raw_epochs = 0U;
        config.policy.tuning.transition_cooldown = 0U;
        auto runtime = std::move(Runtime::create(config)).value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 4U);
        for (std::uint32_t iteration = 0U; iteration < 2U; ++iteration) {
            VRAMZ_CHECK(runner, runtime.reclaim_to_target(ByteSize{2048U}));
            verify(runner, buffer, simulation::Dataset::zeros);
            testing::RuntimeAccess::advance_policy_epoch(runtime, 3U);
        }
        VRAMZ_CHECK(runner, runtime.stats().policy.thrash_events > 0U);
        VRAMZ_CHECK(runner, !runtime.reclaim_to_target(ByteSize{2048U}));
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin(
        "failed restoration attempts advance the clock so residency cooldown cannot freeze itself");
    {
        auto config = policy_config();
        config.policy.tuning.minimum_raw_epochs = 100U;
        auto runtime = std::move(Runtime::create(config)).value();
        auto raw = std::move(runtime.allocate(ByteSize{12288U})).value();
        auto first = std::move(runtime.allocate(ByteSize{4096U})).value();
        VRAMZ_CHECK(runner, testing::migrate(first, 0U, RepresentationState::gpu_compressed));
        auto second = std::move(runtime.allocate(ByteSize{4096U})).value();
        VRAMZ_CHECK(runner, testing::migrate(second, 0U, RepresentationState::gpu_compressed));
        auto extra = std::move(runtime.allocate(ByteSize{4096U})).value();
        std::uint32_t failures = 0U;
        bool restored = false;
        for (std::uint32_t attempt = 0U; attempt < 120U; ++attempt) {
            const auto before = runtime.stats().policy.access_epoch;
            const auto revision = testing::chunk_snapshot(first, 0U).value().access_revision;
            auto acquired = first.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
            VRAMZ_CHECK(runner, runtime.stats().policy.access_epoch > before);
            const auto after_access = testing::chunk_snapshot(first, 0U).value();
            VRAMZ_CHECK(runner, after_access.access_revision == revision + 1U);
            VRAMZ_CHECK(runner, after_access.lease_intents == 0U);
            conserved(runner, runtime);
            if (acquired) {
                VRAMZ_CHECK(runner, acquired.value().close());
                restored = true;
                break;
            }
            VRAMZ_CHECK(runner, acquired.error().code == ErrorCode::out_of_gpu_memory);
            ++failures;
        }
        VRAMZ_CHECK(runner, restored && failures > 0U);
        VRAMZ_CHECK(runner, raw.close());
        VRAMZ_CHECK(runner, first.close());
        VRAMZ_CHECK(runner, second.close());
        VRAMZ_CHECK(runner, extra.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("soft target failure cannot use host as an ordinary storage tier");
    {
        auto config = policy_config(PolicyMode::gpu_resident_with_host_fallback);
        config.budgets.gpu.soft_target = ByteSize{4096U};
        auto runtime = std::move(Runtime::create(config)).value();
        auto buffer = std::move(runtime.allocate(ByteSize{8192U})).value();
        VRAMZ_CHECK(runner, runtime.stats().policy.host_fallback_count == 0U);
        VRAMZ_CHECK(runner, runtime.stats().policy.logical_host_bytes == ByteSize{});
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("concurrent policy acquire and statistics respect lock order and lease lifetime");
    {
        auto runtime = std::move(Runtime::create(policy_config())).value();
        auto buffer = std::move(runtime.allocate(ByteSize{16384U})).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        std::barrier start{2};
        std::atomic<bool> valid{true};
        std::jthread reader{[&]() {
            start.arrive_and_wait();
            for (std::uint64_t index = 0U; index < 100U; ++index) {
                auto lease =
                    buffer.acquire(MemoryRange{ByteOffset{(index % 4U) * 4096U}, ByteSize{4096U}});
                if (lease) {
                    std::array<std::byte, 4096U> data{};
                    const auto read = testing::read_bytes(lease.value(), ByteOffset{}, data);
                    const auto closed = lease.value().close();
                    if (!read || !closed || !std::all_of(data.begin(), data.end(), [](std::byte b) {
                            return b == std::byte{};
                        })) {
                        valid.store(false);
                    }
                } else if (lease.error().code != ErrorCode::busy &&
                           lease.error().code != ErrorCode::conflict &&
                           lease.error().code != ErrorCode::out_of_gpu_memory) {
                    valid.store(false);
                }
                static_cast<void>(runtime.stats());
            }
        }};
        std::jthread pressure{[&]() {
            start.arrive_and_wait();
            for (std::uint32_t index = 0U; index < 100U; ++index) {
                const auto reclaimed = runtime.reclaim_to_target(ByteSize{8192U});
                if (!reclaimed && reclaimed.error().code != ErrorCode::out_of_gpu_memory &&
                    reclaimed.error().code != ErrorCode::busy) {
                    valid.store(false);
                }
            }
        }};
        reader.join();
        pressure.join();
        VRAMZ_CHECK(runner, valid.load());
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }
    runner.begin("completed read invalidates selected COLD proposal before runtime migration in "
                 "100 deterministic two-thread schedules");
    {
        auto config = policy_config();
        config.budgets.gpu.soft_target = ByteSize{2048U};
        auto runtime = std::move(Runtime::create(config)).value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        write(runner, buffer, simulation::Dataset::repeated);
        SelectionPause pause;
        std::atomic<bool> valid{true};
        std::jthread reader{[&]() {
            for (std::uint32_t iteration = 0U; iteration < 100U; ++iteration) {
                pause.selected.arrive_and_wait();
                auto acquired = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
                if (!acquired) {
                    valid.store(false);
                } else {
                    std::array<std::byte, 4096U> expected{};
                    std::array<std::byte, 4096U> actual{};
                    simulation::fill(simulation::Dataset::repeated, 0U, 0U, expected);
                    const auto read = testing::read_bytes(acquired.value(), ByteOffset{}, actual);
                    const auto pinned = testing::chunk_snapshot(buffer, 0U).value();
                    const auto active_move =
                        testing::migrate(buffer, 0U, RepresentationState::gpu_compressed);
                    const auto closed = acquired.value().close();
                    const auto after = testing::chunk_snapshot(buffer, 0U).value();
                    const auto metadata =
                        testing::RuntimeAccess::policy_metadata(buffer, 0U).value();
                    const auto now = runtime.stats().policy.access_epoch;
                    const auto fresh = policy::propose(metadata, after, BufferId{1U},
                                                       Pressure::soft, now, config.policy);
                    if (!read || actual != expected || !closed || pinned.read_pins != 1U ||
                        active_move || active_move.error().code != ErrorCode::busy ||
                        after.read_pins != 0U || after.lease_intents != 0U ||
                        after.access_revision != pause.snapshot.access_revision + 1U ||
                        after.authoritative.resource != pause.snapshot.authoritative.resource ||
                        after.authoritative.content != pause.snapshot.authoritative.content ||
                        policy::temperature(metadata, now, config.policy.tuning) !=
                            Temperature::hot ||
                        fresh.action != PolicyAction::keep) {
                        valid.store(false);
                    }
                }
                pause.accessed.arrive_and_wait();
            }
        }};
        for (std::uint32_t iteration = 0U; iteration < 100U; ++iteration) {
            testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
            const auto result =
                testing::RuntimeAccess::reclaim(runtime, ByteSize{2048U}, false, &pause);
            VRAMZ_CHECK(runner, !result && result.error().code == ErrorCode::out_of_gpu_memory);
            VRAMZ_CHECK(runner, pause.selected_proposal.temperature == Temperature::cold &&
                                    pause.selected_proposal.action == PolicyAction::compress_gpu);
            const auto after = testing::chunk_snapshot(buffer, 0U).value();
            VRAMZ_CHECK(runner,
                        after.authoritative.resource == pause.snapshot.authoritative.resource &&
                            after.authoritative.state == RepresentationState::gpu_raw &&
                            !after.transition_active && after.transition_epoch == 0U &&
                            after.cleanup_resource_count == 0U);
            const auto stats = runtime.stats();
            VRAMZ_CHECK(runner, stats.policy.stale_proposal_rejections == iteration + 1U &&
                                    stats.policy.compression_failures == 0U &&
                                    stats.policy.compression_attempts == 0U &&
                                    stats.policy.simulated_compression_cost == 0U &&
                                    stats.policy.policy_transitions == 0U);
            const auto metadata = testing::RuntimeAccess::policy_metadata(buffer, 0U).value();
            VRAMZ_CHECK(runner, !metadata.last_attempt_failed && metadata.retry_after == 0U &&
                                    metadata.compression_failures == 0U);
            conserved(runner, runtime);
            VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 1U &&
                                    stats.gpu.cleanup_debt == ByteSize{} &&
                                    stats.host.cleanup_debt == ByteSize{});
        }
        reader.join();
        VRAMZ_CHECK(runner, valid.load());
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("actual fallback selection carries access revision and rejects a completed read");
    {
        auto runtime =
            std::move(Runtime::create(policy_config(PolicyMode::gpu_resident_with_host_fallback)))
                .value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        write(runner, buffer, simulation::Dataset::random);
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        ReadFallback observer{buffer};
        const auto result =
            testing::RuntimeAccess::reclaim(runtime, ByteSize{2048U}, true, &observer);
        VRAMZ_CHECK(runner, !result && result.error().code == ErrorCode::out_of_gpu_memory);
        const auto after = testing::chunk_snapshot(buffer, 0U).value();
        VRAMZ_CHECK(runner,
                    observer.valid && observer.calls == 1U &&
                        after.access_revision == observer.original.access_revision + 1U &&
                        after.authoritative.resource == observer.original.authoritative.resource &&
                        after.authoritative.state == RepresentationState::gpu_raw);
        const auto stats = runtime.stats().policy;
        VRAMZ_CHECK(runner,
                    stats.stale_proposal_rejections == 1U && stats.host_fallback_count == 0U &&
                        stats.compression_failures == 0U && stats.compression_rejected == 1U &&
                        stats.compression_attempts == 1U);
        VRAMZ_CHECK(
            runner,
            !testing::RuntimeAccess::policy_metadata(buffer, 0U).value().last_attempt_failed);
        const auto metadata = testing::RuntimeAccess::policy_metadata(buffer, 0U).value();
        VRAMZ_CHECK(runner, metadata.stale_rejection_cycle == stats.policy_cycles &&
                                metadata.last_fallback_cycle == stats.policy_cycles &&
                                metadata.last_attempt_cycle == stats.policy_cycles);
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("stale compression excludes the newly HOT victim from same-cycle host fallback");
    {
        auto runtime =
            std::move(Runtime::create(policy_config(PolicyMode::gpu_resident_with_host_fallback)))
                .value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        const auto before = testing::chunk_snapshot(buffer, 0U).value();
        const auto cycles = runtime.stats().policy.policy_cycles;
        SelectionTrace observer{runtime, &buffer};
        const auto result =
            testing::RuntimeAccess::reclaim(runtime, ByteSize{2048U}, true, &observer);
        VRAMZ_CHECK(runner, !result && result.error().code == ErrorCode::out_of_gpu_memory);
        VRAMZ_CHECK(runner, observer.valid && observer.calls == 1U && observer.read_succeeded &&
                                observer.proposals[0U].action == PolicyAction::compress_gpu &&
                                observer.proposals[0U].temperature == Temperature::cold &&
                                observer.temperature_after_read == Temperature::hot &&
                                observer.after_read.access_revision == before.access_revision + 1U);
        const auto after = testing::chunk_snapshot(buffer, 0U).value();
        const auto stats = runtime.stats();
        const auto metadata = testing::RuntimeAccess::policy_metadata(buffer, 0U).value();
        VRAMZ_CHECK(runner, after.authoritative.resource == before.authoritative.resource &&
                                after.authoritative.content == before.authoritative.content &&
                                after.authoritative.state == RepresentationState::gpu_raw &&
                                after.lifecycle == LifecycleState::live &&
                                after.cleanup_resource_count == 0U && !after.transition_active);
        VRAMZ_CHECK(runner, stats.policy.policy_cycles == cycles + 1U &&
                                stats.policy.stale_proposal_rejections == 1U &&
                                stats.policy.host_fallback_count == 0U &&
                                stats.policy.policy_transitions == 0U &&
                                stats.policy.compression_attempts == 0U &&
                                stats.policy.compression_failures == 0U &&
                                stats.policy.compression_rejected == 0U);
        VRAMZ_CHECK(runner, metadata.stale_rejection_cycle == stats.policy.policy_cycles &&
                                metadata.last_attempt_cycle == stats.policy.policy_cycles &&
                                metadata.last_fallback_cycle != stats.policy.policy_cycles &&
                                !metadata.last_attempt_failed && metadata.retry_after == 0U);
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, stats.gpu.cleanup_debt == ByteSize{} &&
                                stats.host.cleanup_debt == ByteSize{} &&
                                testing::owned_resource_count(runtime) == 1U);
        verify(runner, buffer, simulation::Dataset::zeros);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("cycle-wide stale exclusion expires and a later cycle evaluates a fresh snapshot");
    {
        auto runtime =
            std::move(Runtime::create(policy_config(PolicyMode::gpu_resident_with_host_fallback)))
                .value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        SelectionTrace stale{runtime, &buffer};
        const auto rejected =
            testing::RuntimeAccess::reclaim(runtime, ByteSize{2048U}, true, &stale);
        VRAMZ_CHECK(runner, !rejected && rejected.error().code == ErrorCode::out_of_gpu_memory &&
                                stale.valid && stale.read_succeeded && stale.calls == 1U);
        const auto rejected_cycle = runtime.stats().policy.policy_cycles;
        const auto accessed = testing::chunk_snapshot(buffer, 0U).value();
        const auto metadata = testing::RuntimeAccess::policy_metadata(buffer, 0U).value();
        VRAMZ_CHECK(runner, metadata.stale_rejection_cycle == rejected_cycle);
        conserved(runner, runtime);
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        SelectionTrace fresh{runtime};
        VRAMZ_CHECK(runner,
                    testing::RuntimeAccess::reclaim(runtime, ByteSize{2048U}, true, &fresh));
        VRAMZ_CHECK(runner, fresh.valid && fresh.calls == 1U &&
                                fresh.proposals[0U].action == PolicyAction::compress_gpu &&
                                fresh.proposals[0U].temperature == Temperature::cold &&
                                fresh.snapshots[0U].access_revision == accessed.access_revision &&
                                fresh.proposals[0U].last_access == metadata.last_access_epoch);
        const auto stats = runtime.stats();
        VRAMZ_CHECK(runner, stats.policy.policy_cycles == rejected_cycle + 1U &&
                                stats.policy.stale_proposal_rejections == 1U &&
                                stats.policy.compression_successes == 1U &&
                                stats.policy.compression_failures == 0U &&
                                stats.policy.host_fallback_count == 0U);
        VRAMZ_CHECK(runner, testing::chunk_snapshot(buffer, 0U).value().authoritative.state ==
                                RepresentationState::gpu_compressed);
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, stats.gpu.cleanup_debt == ByteSize{} &&
                                stats.host.cleanup_debt == ByteSize{} &&
                                testing::owned_resource_count(runtime) == 1U);
        verify(runner, buffer, simulation::Dataset::zeros);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("stale Chunk A does not exclude a useful Chunk B in the same reclaim cycle");
    {
        auto runtime =
            std::move(Runtime::create(policy_config(PolicyMode::gpu_resident_with_host_fallback)))
                .value();
        auto buffer = std::move(runtime.allocate(ByteSize{8192U})).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        const auto a = testing::chunk_snapshot(buffer, 0U).value();
        const auto b = testing::chunk_snapshot(buffer, 1U).value();
        SelectionTrace observer{runtime, &buffer};
        VRAMZ_CHECK(runner,
                    testing::RuntimeAccess::reclaim(runtime, ByteSize{5000U}, true, &observer));
        VRAMZ_CHECK(runner, observer.valid && observer.read_succeeded && observer.calls == 2U &&
                                observer.snapshots[0U].id == ChunkId{0U} &&
                                observer.snapshots[1U].id == ChunkId{1U} &&
                                observer.proposals[0U].action == PolicyAction::compress_gpu &&
                                observer.proposals[1U].action == PolicyAction::compress_gpu);
        const auto a_after = testing::chunk_snapshot(buffer, 0U).value();
        const auto b_after = testing::chunk_snapshot(buffer, 1U).value();
        const auto stats = runtime.stats();
        VRAMZ_CHECK(runner,
                    a_after.authoritative.resource == a.authoritative.resource &&
                        a_after.authoritative.state == RepresentationState::gpu_raw &&
                        a_after.cleanup_resource_count == 0U &&
                        b_after.authoritative.resource != b.authoritative.resource &&
                        b_after.authoritative.content == b.authoritative.content &&
                        b_after.authoritative.state == RepresentationState::gpu_compressed &&
                        b_after.cleanup_resource_count == 0U);
        VRAMZ_CHECK(runner, stats.policy.stale_proposal_rejections == 1U &&
                                stats.policy.compression_successes == 1U &&
                                stats.policy.compression_attempts == 1U &&
                                stats.policy.compression_failures == 0U &&
                                stats.policy.host_fallback_count == 0U &&
                                stats.policy.policy_transitions == 1U);
        VRAMZ_CHECK(
            runner,
            testing::RuntimeAccess::policy_metadata(buffer, 0U).value().stale_rejection_cycle ==
                    stats.policy.policy_cycles &&
                testing::RuntimeAccess::policy_metadata(buffer, 1U).value().stale_rejection_cycle !=
                    stats.policy.policy_cycles);
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, stats.gpu.cleanup_debt == ByteSize{} &&
                                stats.host.cleanup_debt == ByteSize{} &&
                                testing::owned_resource_count(runtime) == 2U);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("verified nonbeneficial compression still permits same-cycle same-chunk fallback");
    {
        auto runtime =
            std::move(Runtime::create(policy_config(PolicyMode::gpu_resident_with_host_fallback)))
                .value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        write(runner, buffer, simulation::Dataset::random);
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        const auto before = testing::chunk_snapshot(buffer, 0U).value();
        SelectionTrace observer{runtime};
        VRAMZ_CHECK(runner,
                    testing::RuntimeAccess::reclaim(runtime, ByteSize{2048U}, true, &observer));
        VRAMZ_CHECK(runner, observer.valid && observer.calls == 2U &&
                                observer.proposals[0U].action == PolicyAction::compress_gpu &&
                                observer.proposals[1U].action == PolicyAction::host_fallback &&
                                observer.snapshots[0U].authoritative.resource ==
                                    before.authoritative.resource &&
                                observer.snapshots[1U].authoritative.resource ==
                                    before.authoritative.resource &&
                                observer.snapshots[1U].access_revision == before.access_revision);
        const auto after = testing::chunk_snapshot(buffer, 0U).value();
        const auto stats = runtime.stats();
        const auto metadata = testing::RuntimeAccess::policy_metadata(buffer, 0U).value();
        VRAMZ_CHECK(runner, after.authoritative.state == RepresentationState::host_raw &&
                                after.authoritative.content == before.authoritative.content &&
                                after.cleanup_resource_count == 0U);
        VRAMZ_CHECK(runner, stats.policy.compression_attempts == 1U &&
                                stats.policy.compression_rejected == 1U &&
                                stats.policy.compression_failures == 0U &&
                                stats.policy.host_fallback_count == 1U &&
                                stats.policy.stale_proposal_rejections == 0U &&
                                metadata.stale_rejection_cycle == 0U &&
                                metadata.last_attempt_cycle == stats.policy.policy_cycles &&
                                metadata.last_fallback_cycle == stats.policy.policy_cycles);
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, stats.gpu.cleanup_debt == ByteSize{} &&
                                stats.host.cleanup_debt == ByteSize{} &&
                                testing::owned_resource_count(runtime) == 1U);
        verify(runner, buffer, simulation::Dataset::random);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin("access revision reaches UINT64_MAX once and then rejects without wrapping");
    for (const auto mode : {PolicyMode::disabled, PolicyMode::gpu_resident}) {
        auto runtime = std::move(Runtime::create(policy_config(mode))).value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        VRAMZ_CHECK(runner,
                    testing::RuntimeAccess::advance_access_revision(
                        buffer, ChunkId{0U}, std::numeric_limits<std::uint64_t>::max() - 1U));
        auto lease = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}});
        VRAMZ_CHECK(runner, lease && lease.value().close());
        const auto before = testing::chunk_snapshot(buffer, 0U).value();
        const auto epoch = runtime.stats().policy.access_epoch;
        auto exhausted = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{4096U}},
                                        AcquireOptions{AccessMode::read_write});
        VRAMZ_CHECK(runner, !exhausted && exhausted.error().code == ErrorCode::arithmetic_overflow);
        const auto after = testing::chunk_snapshot(buffer, 0U).value();
        VRAMZ_CHECK(runner, after.access_revision == std::numeric_limits<std::uint64_t>::max() &&
                                after.lease_intents == 0U && after.read_pins == 0U &&
                                !after.write_pin &&
                                after.authoritative.content == before.authoritative.content &&
                                runtime.stats().policy.access_epoch == epoch);
        conserved(runner, runtime);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        empty(runner, runtime);
    }

    runner.begin(
        "multi-chunk overflow validates the whole range before any revision or intent advances");
    for (const auto access : {AccessMode::read_only, AccessMode::read_write}) {
        for (std::uint64_t exhausted = 0U; exhausted < 3U; ++exhausted) {
            auto runtime = std::move(Runtime::create(policy_config())).value();
            auto buffer = std::move(runtime.allocate(ByteSize{12288U})).value();
            auto initial = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{12288U}});
            VRAMZ_CHECK(runner, initial);
            for (std::uint64_t index = 0U; index < 3U; ++index) {
                const auto snapshot = testing::chunk_snapshot(buffer, index).value();
                VRAMZ_CHECK(runner, snapshot.access_revision == 1U && snapshot.read_pins == 1U &&
                                        snapshot.lease_intents == 0U);
            }
            VRAMZ_CHECK(runner, initial.value().close());
            VRAMZ_CHECK(runner, testing::RuntimeAccess::advance_access_revision(
                                    buffer, ChunkId{exhausted},
                                    std::numeric_limits<std::uint64_t>::max() - 1U));
            const auto epoch = runtime.stats().policy.access_epoch;
            auto result =
                buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{12288U}}, AcquireOptions{access});
            VRAMZ_CHECK(runner, !result && result.error().code == ErrorCode::arithmetic_overflow);
            for (std::uint64_t index = 0U; index < 3U; ++index) {
                const auto snapshot = testing::chunk_snapshot(buffer, index).value();
                VRAMZ_CHECK(
                    runner,
                    snapshot.access_revision ==
                        (index == exhausted ? std::numeric_limits<std::uint64_t>::max() : 1U));
                VRAMZ_CHECK(runner, snapshot.lease_intents == 0U && snapshot.read_pins == 0U &&
                                        !snapshot.write_pin && !snapshot.transition_active &&
                                        snapshot.authoritative.content.generation == 0U);
            }
            VRAMZ_CHECK(runner, runtime.stats().policy.access_epoch == epoch);
            const auto unaffected = (exhausted + 1U) % 3U;
            auto live = buffer.acquire(MemoryRange{ByteOffset{unaffected * 4096U}, ByteSize{4096U}},
                                       AcquireOptions{AccessMode::read_write});
            VRAMZ_CHECK(runner, live && live.value().close());
            conserved(runner, runtime);
            VRAMZ_CHECK(runner, buffer.close());
            VRAMZ_CHECK(runner, runtime.shutdown());
            empty(runner, runtime);
        }
    }
    return runner.finish();
}
