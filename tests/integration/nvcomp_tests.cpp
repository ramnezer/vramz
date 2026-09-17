#include "../support/fake_cuda_driver.hpp"
#include "../support/fake_nvcomp.hpp"
#include "../test_support.hpp"
#include "vramz/detail/cuda_vmm_backend.hpp"
#include "vramz/detail/simulation.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <limits>
#include <optional>
#include <sys/wait.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

using namespace vramz;
using detail::CudaVmmBackend;
using detail::GpuCodecDirection;
using test::CudaCall;
using test::CudaFaultMode;
using test::GpuCodecCall;
using test::GpuCodecFault;

namespace {
static_assert(std::is_trivially_copyable_v<TransferReceipt>);
static_assert(std::is_nothrow_constructible_v<Result<TransferReceipt>, TransferReceipt>);
static_assert(!std::is_move_assignable_v<test::FakeNvcompLz4Api>);
static_assert(!std::is_copy_constructible_v<test::FakeNvcompLz4Api>);

struct Fixture final {
    std::shared_ptr<test::FakeCudaState> cuda{std::make_shared<test::FakeCudaState>()};
    std::shared_ptr<test::FakeGpuCodecState> codec{std::make_shared<test::FakeGpuCodecState>()};
    test::FakeCudaDriverApi* driver{};
    test::FakeNvcompLz4Api* api{};
    CudaVmmBackend* storage{};
    std::unique_ptr<CudaVmmBackend> backend{};
    explicit Fixture(ByteSize granularity = ByteSize{64U}, std::optional<ByteSize> host_page = {}) {
        cuda->config.minimum = granularity;
        cuda->config.host_page = host_page.value_or(granularity);
        cuda->config.physical_limit = ByteSize{64ULL * 1024ULL * 1024ULL};
        cuda->config.maximum_resource = ByteSize{32ULL * 1024ULL * 1024ULL};
        auto driver_owner = std::make_unique<test::FakeCudaDriverApi>(cuda);
        driver = driver_owner.get();
        auto codec_owner = std::make_unique<test::FakeNvcompLz4Api>(*driver, codec);
        api = codec_owner.get();
        auto result = CudaVmmBackend::create(std::move(driver_owner), 0, std::move(codec_owner));
        if (!result) {
            std::terminate();
        }
        backend = std::move(result).value();
        storage = backend.get();
    }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
};

[[nodiscard]] RuntimeConfig config(ByteSize chunk = ByteSize{4096U}) noexcept {
    auto value = test::test_runtime_config();
    value.required_capabilities.host_tier = false;
    value.preferred_chunk_size = chunk;
    value.budgets.host = {};
    value.budgets.gpu = {ByteSize{32ULL * 1024ULL * 1024ULL}, ByteSize{16ULL * 1024ULL * 1024ULL},
                         ByteSize{16ULL * 1024ULL * 1024ULL}};
    return value;
}
[[nodiscard]] Runtime runtime(Fixture& fixture, RuntimeConfig configuration = config()) {
    auto created =
        testing::RuntimeAccess::create_with_backend(configuration, std::move(fixture.backend));
    if (!created) {
        std::terminate();
    }
    return std::move(created).value();
}
void conserved(test::Runner& runner, const Runtime& runtime, const Fixture& fixture) {
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
    const auto usage = runtime.stats().gpu;
    const std::scoped_lock lock{fixture.cuda->mutex};
    VRAMZ_CHECK(runner, usage.committed.value() + usage.staging.value() + usage.workspace.value() +
                                usage.cleanup_debt.value() ==
                            fixture.cuda->owned_bytes);
}
void clean(test::Runner& runner, Runtime& runtime, Fixture& fixture) {
    VRAMZ_CHECK(runner, runtime.shutdown());
    conserved(runner, runtime, fixture);
    const auto usage = runtime.stats().gpu;
    VRAMZ_CHECK(runner, usage.committed == ByteSize{} && usage.reserved == ByteSize{} &&
                            usage.staging == ByteSize{} && usage.workspace == ByteSize{} &&
                            usage.cleanup_debt == ByteSize{});
    VRAMZ_CHECK(runner, fixture.driver->empty());
    VRAMZ_CHECK(runner, fixture.api->snapshot().streams == 0U);
}
void write(test::Runner& runner, Buffer& buffer, std::span<const std::byte> bytes) {
    auto lease = buffer.acquire({ByteOffset{}, ByteSize{bytes.size()}}, {AccessMode::read_write});
    VRAMZ_CHECK(runner, lease);
    if (lease) {
        VRAMZ_CHECK(runner, testing::write_bytes(lease.value(), {}, bytes));
        VRAMZ_CHECK(runner, lease.value().close());
    }
}
void read(test::Runner& runner, Buffer& buffer, std::span<const std::byte> expected) {
    std::vector<std::byte> bytes(expected.size());
    auto lease = buffer.acquire({ByteOffset{}, ByteSize{expected.size()}});
    VRAMZ_CHECK(runner, lease);
    if (lease) {
        VRAMZ_CHECK(runner, testing::read_bytes(lease.value(), {}, bytes));
        VRAMZ_CHECK(runner, std::ranges::equal(bytes, expected));
        VRAMZ_CHECK(runner, lease.value().close());
    }
}

class Observer final : public TransactionObserver {
  public:
    Observer(Runtime& runtime, Fixture& fixture) noexcept : runtime_(runtime), fixture_(fixture) {}
    void on_phase(TransactionPhase phase, const ChunkSnapshot& snapshot) noexcept override {
        phases[static_cast<std::size_t>(phase)] = true;
        valid = valid && testing::accounting_conserved(runtime_, PhysicalTier::gpu) &&
                testing::accounting_conserved(runtime_, PhysicalTier::host);
        const auto usage = runtime_.stats().gpu;
        const auto total = usage.committed.value() + usage.staging.value() +
                           usage.workspace.value() + usage.cleanup_debt.value();
        {
            const std::scoped_lock lock{fixture_.cuda->mutex};
            valid = valid && total == fixture_.cuda->owned_bytes;
        }
        if (phase == TransactionPhase::charge_reconciled) {
            staging = usage.staging;
            workspace = usage.workspace;
        }
        if (phase == TransactionPhase::committed) {
            committed = snapshot.authoritative;
        }
    }
    std::array<bool, 12U> phases{};
    bool valid{true};
    ByteSize staging{};
    ByteSize workspace{};
    Representation committed{};

  private:
    Runtime& runtime_;
    Fixture& fixture_;
};

void round_trips(test::Runner& runner) {
    for (const auto dataset :
         {simulation::Dataset::zeros, simulation::Dataset::repeated, simulation::Dataset::random}) {
        runner.begin(
            "fake nvCOMP byte round trip, exact compaction, stable VA and accounting phases");
        Fixture fixture;
        auto owner = runtime(fixture);
        auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
        std::array<std::byte, 4096U> bytes{};
        simulation::fill(dataset, 0U, 7U, bytes);
        write(runner, buffer, bytes);
        const auto before = testing::chunk_snapshot(buffer, 0U).value().authoritative;
        Observer observer{owner, fixture};
        const auto compressed =
            testing::migrate(buffer, 0U, RepresentationState::gpu_compressed, &observer);
        VRAMZ_CHECK(runner, compressed);
        if (compressed) {
            const auto after = testing::chunk_snapshot(buffer, 0U).value().authoritative;
            VRAMZ_CHECK(runner,
                        after.address == DeviceAddress{} && after.content == before.content);
            VRAMZ_CHECK(runner, after.metadata.crc32c == before.metadata.crc32c);
            const auto padded =
                checked_align_up(after.metadata.stored_size.value(), 64U, OperationId::compress)
                    .value();
            VRAMZ_CHECK(runner, after.charge == ByteSize{padded});
            VRAMZ_CHECK(runner,
                        observer.staging == after.charge && observer.workspace >= ByteSize{4160U});
            VRAMZ_CHECK(
                runner,
                observer.valid &&
                    observer.phases[static_cast<std::size_t>(TransactionPhase::cleanup_complete)]);
            VRAMZ_CHECK(runner, fixture.api->snapshot().completions == 1U);
            // The stable public mapping is absent; only private GPU backing remains.
            {
                const std::scoped_lock lock{fixture.cuda->mutex};
                VRAMZ_CHECK(runner,
                            std::ranges::none_of(fixture.cuda->mappings, [&](const auto& entry) {
                                return entry.address == before.address;
                            }));
            }
            read(runner, buffer, bytes);
            const auto restored = testing::chunk_snapshot(buffer, 0U).value().authoritative;
            VRAMZ_CHECK(runner,
                        restored.address == before.address && restored.content == before.content);
            VRAMZ_CHECK(runner, fixture.api->snapshot().decompressions == 1U);
        }
        conserved(runner, owner, fixture);
        VRAMZ_CHECK(runner, buffer.close());
        clean(runner, owner, fixture);
    }
}

void planning_tests(test::Runner& runner) {
    Fixture fixture;
    for (const auto size :
         {detail::gpu_lz4_max_chunk.value() - 1U, detail::gpu_lz4_max_chunk.value(),
          detail::gpu_lz4_max_chunk.value() + 1U}) {
        runner.begin("LZ4 16 MiB boundary is checked by planning before any allocation or launch");
        const auto creates = fixture.driver->count(CudaCall::create);
        const auto plan =
            fixture.backend->allocation_bound(RepresentationState::gpu_compressed, ByteSize{size});
        VRAMZ_CHECK(runner, plan.has_value() == (size <= detail::gpu_lz4_max_chunk.value()));
        VRAMZ_CHECK(runner, fixture.driver->count(CudaCall::create) == creates);
        VRAMZ_CHECK(runner, fixture.api->snapshot().compressions == 0U);
    }
    runner.begin("empty planning and allocation preserve zero-length semantics without codec work");
    const auto empty = fixture.api->requirements(GpuCodecDirection::compress, {});
    VRAMZ_CHECK(runner, empty);
    if (empty) {
        const auto plan = detail::make_gpu_compression_plan(empty.value(), {{}, ByteSize{64U}});
        VRAMZ_CHECK(runner, plan && plan.value().workspace_charge == ByteSize{});
    }
    auto owner = runtime(fixture);
    const auto calls = fixture.driver->count(CudaCall::create);
    VRAMZ_CHECK(runner, !owner.allocate(ByteSize{}));
    VRAMZ_CHECK(runner, fixture.driver->count(CudaCall::create) == calls);
    clean(runner, owner, fixture);

    for (const auto fault : {GpuCodecFault::plan_bound_overflow, GpuCodecFault::plan_alignment_zero,
                             GpuCodecFault::plan_temp_overflow}) {
        runner.begin(
            "malformed queried requirements reject before mutation and preserve authority");
        Fixture active;
        auto runtime_owner = runtime(active);
        auto buffer = std::move(runtime_owner.allocate(ByteSize{4096U})).value();
        const auto source = testing::chunk_snapshot(buffer, 0U).value().authoritative;
        const auto created = active.driver->count(CudaCall::create);
        active.api->inject(fault);
        VRAMZ_CHECK(runner, !testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        VRAMZ_CHECK(runner, active.driver->count(CudaCall::create) == created);
        VRAMZ_CHECK(runner, testing::chunk_snapshot(buffer, 0U).value().authoritative.resource ==
                                source.resource);
        conserved(runner, runtime_owner, active);
        VRAMZ_CHECK(runner, buffer.close());
        clean(runner, runtime_owner, active);
    }
    runner.begin(
        "unsupported HOST policy and oversized configured chunk reject before Runtime mutation");
    for (const bool oversized : {false, true}) {
        Fixture rejected;
        auto configuration = config();
        if (oversized) {
            configuration.preferred_chunk_size = ByteSize{detail::gpu_lz4_max_chunk.value() + 64U};
        } else {
            configuration.policy.mode = PolicyMode::gpu_resident_with_host_fallback;
        }
        VRAMZ_CHECK(runner, !testing::RuntimeAccess::create_with_backend(
                                configuration, std::move(rejected.backend)));
        VRAMZ_CHECK(runner,
                    rejected.cuda->owned_bytes == 0U && rejected.codec->observation.streams == 0U);
    }
    runner.begin("alignment LCM, packed metadata padding and arithmetic overflow are explicit");
    const detail::GpuCodecRequirements requirements{ByteSize{513U}, ByteSize{17U}, ByteSize{256U},
                                                    ByteSize{128U}, ByteSize{512U}};
    const auto plan =
        detail::make_gpu_compression_plan(requirements, {ByteSize{512U}, ByteSize{64U}});
    VRAMZ_CHECK(runner, plan && plan.value().address_alignment == ByteSize{512U} &&
                            plan.value().temporary_offset == ByteSize{512U} &&
                            plan.value().workspace_charge == ByteSize{576U} &&
                            plan.value().output_charge == ByteSize{576U});
    VRAMZ_CHECK(runner,
                !detail::make_gpu_compression_plan(requirements, {ByteSize{512U}, ByteSize{}}));
}

void policy_tests(test::Runner& runner) {
    runner.begin("M3 GPU-resident policy uses fake nvCOMP, exact CRC, revision and stable restore");
    Fixture fixture;
    auto configuration = config();
    configuration.policy.mode = PolicyMode::gpu_resident;
    configuration.budgets.gpu = {ByteSize{32768U}, ByteSize{8192U}, ByteSize{16384U}};
    auto owner = runtime(fixture, configuration);
    auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
    std::array<std::byte, 4096U> bytes{};
    simulation::fill(simulation::Dataset::repeated, 1U, 3U, bytes);
    write(runner, buffer, bytes);
    const auto source = testing::chunk_snapshot(buffer, 0U).value();
    testing::RuntimeAccess::advance_policy_epoch(owner, 20U);
    VRAMZ_CHECK(runner, owner.reclaim_to_target(ByteSize{2048U}));
    const auto compressed = testing::chunk_snapshot(buffer, 0U).value();
    VRAMZ_CHECK(runner, compressed.authoritative.state == RepresentationState::gpu_compressed);
    VRAMZ_CHECK(runner, compressed.authoritative.content == source.authoritative.content &&
                            compressed.access_revision == source.access_revision);
    read(runner, buffer, bytes);
    const auto restored = testing::chunk_snapshot(buffer, 0U).value();
    VRAMZ_CHECK(runner, restored.access_revision == source.access_revision + 1U &&
                            restored.authoritative.address == source.authoritative.address);
    VRAMZ_CHECK(runner, owner.stats().policy.restore_count == 1U &&
                            owner.stats().policy.host_fallback_count == 0U);
    conserved(runner, owner, fixture);
    VRAMZ_CHECK(runner, buffer.close());
    clean(runner, owner, fixture);

    runner.begin("scaled 3x fake GPU-resident working set restores every real byte without HOST");
    Fixture scaled;
    auto scaled_owner = runtime(scaled, configuration);
    std::vector<Buffer> buffers;
    buffers.reserve(24U);
    for (std::uint64_t index = 0U; index < 24U; ++index) {
        auto allocated = scaled_owner.allocate(ByteSize{4096U});
        VRAMZ_CHECK(runner, allocated);
        if (!allocated) {
            break;
        }
        buffers.push_back(std::move(allocated).value());
        simulation::fill(simulation::Dataset::repeated, index, 0U, bytes);
        write(runner, buffers.back(), bytes);
        testing::RuntimeAccess::advance_policy_epoch(scaled_owner, 20U);
        conserved(runner, scaled_owner, scaled);
    }
    VRAMZ_CHECK(runner, buffers.size() == 24U);
    const auto stats = scaled_owner.stats();
    VRAMZ_CHECK(runner, stats.policy.logical_gpu_resident_bytes == ByteSize{98304U});
    VRAMZ_CHECK(runner, stats.policy.logical_host_bytes == ByteSize{} &&
                            stats.gpu.peak_charged <= configuration.budgets.gpu.hard_limit);
    std::cout << "M5_SCENARIO {\"name\":\"scaled_3x\",\"fake_only\":true,\"budget\":32768,"
              << "\"logical\":98304,\"raw_charge\":" << stats.policy.gpu_raw_charge.value()
              << ",\"stored_bytes\":" << stats.compression.stored_payload_bytes.value()
              << ",\"padded_compressed\":" << stats.policy.gpu_compressed_charge.value()
              << ",\"total_charge\":" << stats.gpu.committed.value()
              << ",\"peak_charge\":" << stats.gpu.peak_charged.value()
              << ",\"effective_ratio\":3.0}\n";
    for (std::size_t index = 0U; index < buffers.size(); ++index) {
        simulation::fill(simulation::Dataset::repeated, index, 0U, bytes);
        read(runner, buffers[index], bytes);
        testing::RuntimeAccess::advance_policy_epoch(scaled_owner, 20U);
        conserved(runner, scaled_owner, scaled);
    }
    for (auto& item : buffers) {
        VRAMZ_CHECK(runner, item.close());
    }
    clean(runner, scaled_owner, scaled);

    for (const auto dataset : {simulation::Dataset::random, simulation::Dataset::encoded_like}) {
        runner.begin(
            "incompressible physical savings reject and policy avoids repeated useless codec work");
        Fixture random;
        auto random_owner = runtime(random, configuration);
        auto random_buffer = std::move(random_owner.allocate(ByteSize{4096U})).value();
        simulation::fill(dataset, 2U, 0U, bytes);
        write(runner, random_buffer, bytes);
        testing::RuntimeAccess::advance_policy_epoch(random_owner, 20U);
        VRAMZ_CHECK(runner, !random_owner.reclaim_to_target(ByteSize{2048U}));
        const auto attempts = random.api->snapshot().compressions;
        VRAMZ_CHECK(runner,
                    attempts == 1U && random_owner.stats().policy.compression_rejected == 1U);
        for (unsigned int attempt = 0U; attempt < 4U; ++attempt) {
            VRAMZ_CHECK(runner, !random_owner.reclaim_to_target(ByteSize{2048U}));
        }
        VRAMZ_CHECK(runner, random.api->snapshot().compressions == attempts);
        VRAMZ_CHECK(runner,
                    testing::chunk_snapshot(random_buffer, 0U).value().authoritative.state ==
                        RepresentationState::gpu_raw);
        read(runner, random_buffer, bytes);
        conserved(runner, random_owner, random);
        VRAMZ_CHECK(runner, random_buffer.close());
        clean(runner, random_owner, random);
    }
}

void sensitivity_tests(test::Runner& runner) {
    for (const bool larger_host_page : {false, true}) {
        runner.begin(
            "private VA reservation page size and physical VMM charge are independently rounded");
        const ByteSize granularity{larger_host_page ? 64U : 4096U};
        const ByteSize host_page{larger_host_page ? 4096U : 64U};
        Fixture fixture{granularity, host_page};
        auto owner = runtime(fixture);
        auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        VRAMZ_CHECK(runner, testing::chunk_snapshot(buffer, 0U).value().authoritative.charge ==
                                granularity);
        {
            const std::scoped_lock lock{fixture.cuda->mutex};
            for (const auto& reservation : fixture.cuda->reservations) {
                if (reservation.address != DeviceAddress{}) {
                    VRAMZ_CHECK(runner, reservation.size.value() % host_page.value() == 0U);
                }
            }
        }
        conserved(runner, owner, fixture);
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_raw));
        VRAMZ_CHECK(runner, buffer.close());
        clean(runner, owner, fixture);
    }
    for (const auto granularity : {4096U, 65536U, 2097152U}) {
        runner.begin("queried 4 KiB / 64 KiB / 2 MiB VMM padding controls actual physical saving");
        Fixture fixture{ByteSize{granularity}};
        const ByteSize logical{2097152U};
        auto owner = runtime(fixture, config(logical));
        auto buffer = std::move(owner.allocate(logical)).value();
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        const auto snapshot = testing::chunk_snapshot(buffer, 0U).value();
        const auto padded = checked_align_up(snapshot.authoritative.metadata.stored_size.value(),
                                             granularity, OperationId::compress)
                                .value();
        VRAMZ_CHECK(runner, snapshot.authoritative.charge == ByteSize{padded});
        std::cout << "M5_SCENARIO {\"name\":\"granularity\",\"fake_only\":true,\"granularity\":"
                  << granularity << ",\"logical\":" << logical.value()
                  << ",\"stored_bytes\":" << snapshot.authoritative.metadata.stored_size.value()
                  << ",\"padded_compressed\":" << padded
                  << ",\"saved_bytes\":" << logical.value() - padded << "}\n";
        VRAMZ_CHECK(runner, granularity != 2097152U || snapshot.authoritative.charge == logical);
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_raw));
        conserved(runner, owner, fixture);
        VRAMZ_CHECK(runner, buffer.close());
        clean(runner, owner, fixture);
    }
    runner.begin(
        "final compressed bytes fit but bound plus workspace plus compaction headroom does not");
    Fixture fixture;
    auto configuration = config();
    configuration.budgets.gpu = {ByteSize{8192U}, ByteSize{4096U}, ByteSize{4096U}};
    auto owner = runtime(fixture, configuration);
    auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
    const auto created = fixture.driver->count(CudaCall::create);
    const auto result = testing::migrate(buffer, 0U, RepresentationState::gpu_compressed);
    VRAMZ_CHECK(runner, !result && result.error().code == ErrorCode::out_of_gpu_memory);
    VRAMZ_CHECK(runner, fixture.driver->count(CudaCall::create) == created &&
                            fixture.api->snapshot().compressions == 0U);
    conserved(runner, owner, fixture);
    VRAMZ_CHECK(runner, buffer.close());
    clean(runner, owner, fixture);
}

void recoverable_faults(test::Runner& runner) {
    constexpr std::array faults{GpuCodecFault::metadata_submission_failure,
                                GpuCodecFault::launch_failure,
                                GpuCodecFault::status_read_failure,
                                GpuCodecFault::size_read_failure,
                                GpuCodecFault::status_failure,
                                GpuCodecFault::size_zero,
                                GpuCodecFault::size_oversized,
                                GpuCodecFault::size_mismatch,
                                GpuCodecFault::output_corruption};
    for (const bool restore : {false, true}) {
        for (const auto fault : faults) {
            for (const bool cleanup_failure : {false, true}) {
                runner.begin("completed codec failure plus optional rollback failure retains exact "
                             "owned debt");
                Fixture fixture;
                auto owner = runtime(fixture);
                auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
                if (restore) {
                    VRAMZ_CHECK(runner,
                                testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
                }
                const auto before = testing::chunk_snapshot(buffer, 0U).value().authoritative;
                const auto codec_before = fixture.api->snapshot();
                fixture.api->inject(fault);
                if (cleanup_failure) {
                    fixture.driver->inject(CudaCall::unmap);
                }
                Observer observer{owner, fixture};
                const auto migrated = testing::migrate(
                    buffer, 0U,
                    restore ? RepresentationState::gpu_raw : RepresentationState::gpu_compressed,
                    &observer);
                VRAMZ_CHECK(runner, !migrated);
                if (fault == GpuCodecFault::metadata_submission_failure) {
                    const auto codec_after = fixture.api->snapshot();
                    VRAMZ_CHECK(runner, !codec_after.pending &&
                                            !codec_after.metadata_publication_pending &&
                                            codec_after.log_size == codec_before.log_size &&
                                            codec_after.metadata_consumptions ==
                                                codec_before.metadata_consumptions);
                }
                const auto after = testing::chunk_snapshot(buffer, 0U).value();
                VRAMZ_CHECK(runner, after.authoritative.resource == before.resource &&
                                        after.authoritative.content == before.content &&
                                        observer.valid);
                VRAMZ_CHECK(
                    runner,
                    !observer.phases[static_cast<std::size_t>(TransactionPhase::committed)]);
                if (cleanup_failure) {
                    const auto debt = testing::RuntimeAccess::cleanup_resource(buffer, 0U, 0U);
                    VRAMZ_CHECK(runner,
                                debt && owner.stats().gpu.cleanup_debt == debt.value().charge);
                    if (debt) {
                        const auto actual = fixture.storage->owned_charge(PhysicalTier::gpu);
                        VRAMZ_CHECK(runner, debt.value().charge.value() + before.charge.value() ==
                                                actual.value());
                    }
                } else {
                    VRAMZ_CHECK(runner, owner.stats().gpu.cleanup_debt == ByteSize{});
                }
                conserved(runner, owner, fixture);
                VRAMZ_CHECK(runner, buffer.close());
                clean(runner, owner, fixture);
            }
        }
    }
    for (const auto fault : {CudaCall::create, CudaCall::device_to_device}) {
        for (const bool cleanup_failure : {false, true}) {
            runner.begin("compaction target allocation or copy failure preserves source and "
                         "accounts exact overlap");
            Fixture fixture;
            auto owner = runtime(fixture);
            auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
            const auto before = testing::chunk_snapshot(buffer, 0U).value().authoritative;
            fixture.driver->inject(fault, CudaFaultMode::before,
                                   {fault == CudaCall::create ? 3U : 1U, 0U});
            if (cleanup_failure) {
                fixture.driver->inject(CudaCall::unmap);
            }
            Observer observer{owner, fixture};
            VRAMZ_CHECK(runner, !testing::migrate(buffer, 0U, RepresentationState::gpu_compressed,
                                                  &observer));
            VRAMZ_CHECK(runner,
                        testing::chunk_snapshot(buffer, 0U).value().authoritative.resource ==
                            before.resource);
            VRAMZ_CHECK(runner, observer.valid && fixture.api->snapshot().compressions == 1U);
            conserved(runner, owner, fixture);
            VRAMZ_CHECK(runner, buffer.close());
            clean(runner, owner, fixture);
        }
    }
    for (const auto occurrence : {1U, 2U, 3U}) {
        runner.begin("old bound output, packed codec workspace and post-commit source cleanup "
                     "failure are exact");
        Fixture fixture;
        auto owner = runtime(fixture);
        auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
        fixture.driver->inject(CudaCall::unmap, CudaFaultMode::before, {occurrence, 0U});
        Observer observer{owner, fixture};
        VRAMZ_CHECK(runner,
                    !testing::migrate(buffer, 0U, RepresentationState::gpu_compressed, &observer));
        const auto usage = owner.stats().gpu;
        const auto expected = occurrence == 1U ? 4160U : occurrence == 2U ? 192U : 4096U;
        VRAMZ_CHECK(runner, usage.cleanup_debt == ByteSize{expected} && observer.valid);
        conserved(runner, owner, fixture);
        VRAMZ_CHECK(runner, buffer.close());
        clean(runner, owner, fixture);
    }
    runner.begin(
        "compaction target post-adoption materialization failure is exact debt, not fail-stop");
    {
        Fixture fixture;
        auto owner = runtime(fixture);
        auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
        testing::RuntimeAccess::inject_materialize_failure(owner, 3U);
        VRAMZ_CHECK(runner, !testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        VRAMZ_CHECK(runner, owner.stats().gpu.cleanup_debt == ByteSize{64U});
        conserved(runner, owner, fixture);
        VRAMZ_CHECK(runner, buffer.close());
        clean(runner, owner, fixture);
    }
    runner.begin(
        "VA-only cleanup failure never becomes physical debt and shutdown retries the retained VA");
    {
        Fixture fixture;
        auto owner = runtime(fixture);
        auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
        fixture.driver->inject(CudaCall::free_address);
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        VRAMZ_CHECK(runner, owner.stats().gpu.cleanup_debt == ByteSize{});
        conserved(runner, owner, fixture);
        VRAMZ_CHECK(runner, buffer.close());
        clean(runner, owner, fixture);
    }
}
enum class FatalKind : std::uint8_t {
    metadata_submission,
    partial_metadata_publication,
    launch,
    completion,
    early_status,
    early_output,
    compaction_copy,
    allocation_properties,
    compaction_adoption
};
void fatal_child(test::Runner& runner, FatalKind kind, bool restore, bool cleanup_fault) {
    const auto child = ::fork();
    VRAMZ_CHECK(runner, child >= 0);
    if (child < 0) {
        return;
    }
    if (child == 0) {
        static_cast<void>(::alarm(5U));
        Fixture fixture;
        auto owner = runtime(fixture);
        auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
        if (restore && !testing::migrate(buffer, 0U, RepresentationState::gpu_compressed)) {
            std::_Exit(1);
        }
        const auto before = testing::chunk_snapshot(buffer, 0U).value().authoritative;
        Observer observer{owner, fixture};
        static Runtime* observed_runtime{};
        static Buffer* observed_buffer{};
        static CudaVmmBackend* observed_backend{};
        static test::FakeCudaState* audit{};
        static Observer* phases{};
        static Representation original{};
        static std::uint64_t unmaps{};
        static test::FakeGpuCodecState* codec_audit{};
        static test::FakeGpuCodecSnapshot codec_before{};
        static bool metadata_fault{};
        static bool metadata_submission_fault{};
        observed_runtime = &owner;
        observed_buffer = &buffer;
        observed_backend = fixture.storage;
        audit = fixture.cuda.get();
        phases = &observer;
        original = before;
        unmaps = fixture.driver->count(CudaCall::unmap);
        codec_audit = fixture.codec.get();
        codec_before = fixture.api->snapshot();
        metadata_fault = kind == FatalKind::metadata_submission ||
                         kind == FatalKind::partial_metadata_publication;
        metadata_submission_fault = kind == FatalKind::metadata_submission;
        std::set_terminate([]() noexcept {
            // The registry mutex is held at the boundary. Inspect only independent audit,
            // leaf error/telemetry channels and core state; never reacquire that mutex.
            const auto error = observed_backend->first_error();
            const auto core_error = observed_runtime->first_async_error();
            const auto snapshot = testing::chunk_snapshot(*observed_buffer, 0U);
            const auto stats = observed_runtime->stats();
            const auto& codec = codec_audit->observation;
            const bool publication_stopped =
                !metadata_fault ||
                (codec.pending && codec.metadata_publication_pending &&
                 codec.metadata_publications == codec_before.metadata_publications &&
                 codec.metadata_consumptions == codec_before.metadata_consumptions &&
                 codec.result_reads == codec_before.result_reads &&
                 (!metadata_submission_fault ||
                  (codec.compressions == codec_before.compressions &&
                   codec.decompressions == codec_before.decompressions)));
            const bool stopped =
                publication_stopped && snapshot &&
                snapshot.value().authoritative.resource == original.resource &&
                snapshot.value().authoritative.content == original.content &&
                snapshot.value().lifecycle != LifecycleState::poisoned &&
                stats.gpu.cleanup_debt == ByteSize{} && stats.gpu.committed == original.charge &&
                audit->counts[static_cast<std::size_t>(CudaCall::unmap)] == unmaps &&
                audit->owned_bytes >= original.charge.value() &&
                !phases->phases[static_cast<std::size_t>(TransactionPhase::committed)] &&
                !phases->phases[static_cast<std::size_t>(TransactionPhase::poisoned)] &&
                !phases->phases[static_cast<std::size_t>(TransactionPhase::rolled_back)] &&
                ((error && error->error.code == ErrorCode::backend_contract_violation) ||
                 (core_error && core_error->error.code == ErrorCode::backend_contract_violation));
            std::_Exit(stopped ? 86 : 1);
        });
        switch (kind) {
        case FatalKind::metadata_submission:
            fixture.api->inject(GpuCodecFault::metadata_submission_ambiguous);
            break;
        case FatalKind::partial_metadata_publication:
            fixture.api->inject(GpuCodecFault::metadata_publication_partial);
            break;
        case FatalKind::launch:
            fixture.api->inject(GpuCodecFault::launch_ambiguous);
            break;
        case FatalKind::completion:
            fixture.api->inject(GpuCodecFault::completion_failure);
            break;
        case FatalKind::early_status:
            fixture.api->inject(GpuCodecFault::premature_status);
            fixture.api->inject(GpuCodecFault::completion_failure);
            break;
        case FatalKind::early_output:
            fixture.api->inject(GpuCodecFault::premature_output);
            fixture.api->inject(GpuCodecFault::completion_failure);
            break;
        case FatalKind::compaction_copy:
            fixture.driver->inject(CudaCall::device_to_device, CudaFaultMode::after);
            break;
        case FatalKind::allocation_properties:
            fixture.driver->inject(CudaCall::properties, CudaFaultMode::malformed, {2U, 0U});
            break;
        case FatalKind::compaction_adoption:
            // Source query, destination create/adopt, workspace create/adopt, source verify,
            // target create/adopt. Source verify also queries physical properties.
            fixture.driver->inject(CudaCall::properties, CudaFaultMode::malformed, {8U, 0U});
            break;
        }
        if (cleanup_fault) {
            fixture.driver->inject(CudaCall::unmap);
        }
        static_cast<void>(testing::migrate(buffer, 0U,
                                           restore ? RepresentationState::gpu_raw
                                                   : RepresentationState::gpu_compressed,
                                           &observer));
        std::_Exit(1);
    }
    int status{};
    pid_t waited{};
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    VRAMZ_CHECK(runner, waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 86);
}
void death_tests(test::Runner& runner) {
    for (const bool restore : {false, true}) {
        for (const auto kind :
             {FatalKind::metadata_submission, FatalKind::partial_metadata_publication,
              FatalKind::launch, FatalKind::completion, FatalKind::early_status,
              FatalKind::early_output, FatalKind::allocation_properties}) {
            for (const bool cleanup_fault : {false, true}) {
                runner.begin("bounded fatal child: unproved completion/adoption cannot unmap, "
                             "commit or create debt");
                fatal_child(runner, kind, restore, cleanup_fault);
            }
        }
    }
    for (const auto kind : {FatalKind::compaction_copy, FatalKind::compaction_adoption}) {
        for (const bool cleanup_fault : {false, true}) {
            runner.begin("ambiguous compaction copy or target corroboration is fail-stop before "
                         "any cleanup");
            fatal_child(runner, kind, false, cleanup_fault);
        }
    }
}

void integrity_and_lifetime_tests(test::Runner& runner) {
    for (const auto fault :
         {GpuCodecFault::count, GpuCodecFault::premature_status, GpuCodecFault::premature_output}) {
        runner.begin("pending batch never exposes a result even when partial device output/status "
                     "is visible");
        auto audit = std::make_shared<test::FakeCudaState>();
        test::FakeCudaDriverApi driver{audit};
        VRAMZ_CHECK(runner, driver.initialize());
        const auto context = driver.retain_primary(0).value();
        VRAMZ_CHECK(runner, driver.push_context(context));
        auto codec_audit = std::make_shared<test::FakeGpuCodecState>();
        test::FakeNvcompLz4Api api{driver, codec_audit};
        VRAMZ_CHECK(runner, api.initialize(context));
        const std::array sizes{ByteSize{64U}, ByteSize{64U}, ByteSize{256U}};
        std::array<DeviceAddress, 3U> addresses{};
        std::array<detail::CudaPhysicalHandle, 3U> handles{};
        for (std::size_t index = 0U; index < sizes.size(); ++index) {
            addresses[index] = driver.reserve({sizes[index], ByteSize{64U}}).value();
            handles[index] = driver.create(sizes[index], 0).value();
            VRAMZ_CHECK(runner, driver.map(addresses[index], sizes[index], handles[index]));
            VRAMZ_CHECK(runner, driver.set_access(addresses[index], sizes[index], 0));
        }
        detail::GpuBatchMetadata metadata{
            addresses[0].value(), 32U, addresses[1].value(), 64U, 0U, -1, 0U};
        const auto expected_metadata = metadata;
        if (fault != GpuCodecFault::count) {
            api.inject(fault);
        }
        VRAMZ_CHECK(runner, api.launch({GpuCodecDirection::compress, ByteSize{32U}, addresses[2],
                                        DeviceAddress{addresses[2].value() + 48U}, ByteSize{128U},
                                        metadata}));
        VRAMZ_CHECK(runner, api.snapshot().pending && !api.result());
        const auto pending = api.snapshot();
        VRAMZ_CHECK(runner, pending.metadata_publication_pending &&
                                pending.metadata_publications == 0U &&
                                pending.metadata_consumptions == 0U && pending.log_size == 2U &&
                                pending.log[0U] == GpuCodecCall::metadata_copy_enqueue &&
                                pending.log[1U] == GpuCodecCall::nvcomp_launch);
        detail::GpuBatchMetadata unpublished{};
        VRAMZ_CHECK(runner, driver.copy_from_device(
                                std::as_writable_bytes(std::span{&unpublished, 1U}), addresses[2]));
        VRAMZ_CHECK(runner, unpublished.input == 0U && unpublished.input_size == 0U);
        // Changing or attempting to resubmit the caller's metadata cannot overwrite the copy
        // retained by the one-pending-operation adapter.
        metadata = {};
        VRAMZ_CHECK(runner, !api.launch({GpuCodecDirection::compress, ByteSize{32U}, addresses[2],
                                         DeviceAddress{addresses[2].value() + 48U}, ByteSize{128U},
                                         metadata}));
        VRAMZ_CHECK(runner, api.synchronize());
        const auto result = api.result();
        VRAMZ_CHECK(runner, result && result.value().status == 0 &&
                                result.value().actual_size != ByteSize{});
        const auto completed = api.snapshot();
        constexpr std::array expected_order{
            GpuCodecCall::metadata_copy_enqueue, GpuCodecCall::nvcomp_launch,
            GpuCodecCall::stream_synchronize, GpuCodecCall::metadata_status_read};
        VRAMZ_CHECK(runner, !completed.pending && !completed.metadata_publication_pending &&
                                completed.rejected_early_reads == 1U &&
                                completed.metadata_publications == 1U &&
                                completed.metadata_consumptions == 1U);
        VRAMZ_CHECK(runner,
                    std::ranges::equal(std::as_bytes(std::span{&completed.consumed_metadata, 1U}),
                                       std::as_bytes(std::span{&expected_metadata, 1U})));
        VRAMZ_CHECK(runner,
                    completed.log_size == expected_order.size() &&
                        std::ranges::equal(std::span{completed.log}.first(completed.log_size),
                                           expected_order) &&
                        completed.dropped_calls == 0U);
        VRAMZ_CHECK(runner, api.shutdown());
        for (std::size_t index = 0U; index < sizes.size(); ++index) {
            VRAMZ_CHECK(runner, driver.unmap(addresses[index], sizes[index]));
            VRAMZ_CHECK(runner, driver.release(handles[index]));
            VRAMZ_CHECK(runner, driver.free_address(addresses[index], sizes[index]));
        }
        VRAMZ_CHECK(runner, driver.pop_context().value() == context);
        VRAMZ_CHECK(runner, driver.release_primary(0));
        VRAMZ_CHECK(runner, driver.empty());
    }
    for (const bool padding : {false, true}) {
        runner.begin("stored CRC protects compressed input before decompressor; physical padding "
                     "is excluded");
        Fixture fixture;
        auto owner = runtime(fixture);
        auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        const auto before = testing::chunk_snapshot(buffer, 0U).value().authoritative;
        const auto calls = fixture.api->snapshot().decompressions;
        {
            const std::scoped_lock lock{fixture.cuda->mutex};
            auto physical = std::ranges::find_if(fixture.cuda->physical, [](const auto& entry) {
                return entry.handle != detail::CudaPhysicalHandle{};
            });
            VRAMZ_CHECK(runner, physical != fixture.cuda->physical.end());
            if (physical != fixture.cuda->physical.end()) {
                const auto offset = padding ? before.metadata.stored_size.value() : 0U;
                VRAMZ_CHECK(runner, offset < physical->bytes.size());
                if (offset < physical->bytes.size()) {
                    physical->bytes[static_cast<std::size_t>(offset)] ^= std::byte{1U};
                }
            }
        }
        const auto result = testing::migrate(buffer, 0U, RepresentationState::gpu_raw);
        if (padding) {
            VRAMZ_CHECK(runner, result && fixture.api->snapshot().decompressions == calls + 1U);
        } else {
            VRAMZ_CHECK(runner, !result && result.error().code == ErrorCode::integrity_failure);
            const auto after = testing::chunk_snapshot(buffer, 0U).value();
            VRAMZ_CHECK(runner, after.lifecycle == LifecycleState::poisoned &&
                                    after.authoritative.resource == before.resource);
            VRAMZ_CHECK(runner, fixture.api->snapshot().decompressions == calls);
        }
        conserved(runner, owner, fixture);
        VRAMZ_CHECK(runner, buffer.close());
        clean(runner, owner, fixture);
    }
    runner.begin(
        "result before completion is rejected and stream shutdown precedes primary release");
    {
        Fixture fixture;
        VRAMZ_CHECK(runner, !fixture.api->result());
        auto owner = runtime(fixture);
        auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        VRAMZ_CHECK(runner, fixture.api->snapshot().rejected_early_reads == 1U);
        VRAMZ_CHECK(runner, buffer.close());
        fixture.api->inject(GpuCodecFault::shutdown_failure);
        VRAMZ_CHECK(runner, !owner.shutdown());
        VRAMZ_CHECK(runner, fixture.cuda->primary_references == 1U &&
                                fixture.api->snapshot().streams == 1U);
        clean(runner, owner, fixture);
    }
    runner.begin("active read lease excludes GPU compression before launch");
    {
        Fixture fixture;
        auto owner = runtime(fixture);
        auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
        auto lease = std::move(buffer.acquire({ByteOffset{}, ByteSize{4096U}})).value();
        VRAMZ_CHECK(runner, !testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        VRAMZ_CHECK(runner, fixture.api->snapshot().compressions == 0U);
        VRAMZ_CHECK(runner, lease.close());
        VRAMZ_CHECK(runner, buffer.close());
        clean(runner, owner, fixture);
    }
}

void stream_ordering_tests(test::Runner& runner) {
    constexpr std::array expected_order{
        GpuCodecCall::metadata_copy_enqueue, GpuCodecCall::nvcomp_launch,
        GpuCodecCall::stream_synchronize, GpuCodecCall::metadata_status_read};
    for (const bool restore : {false, true}) {
        runner.begin("Runtime codec submission publishes exact metadata in explicit stream order");
        Fixture fixture;
        auto owner = runtime(fixture);
        auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
        std::array<std::byte, 4096U> payload{};
        simulation::fill(simulation::Dataset::repeated, 0U, 17U, payload);
        write(runner, buffer, payload);
        if (restore) {
            VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_compressed));
        }
        const auto source = testing::chunk_snapshot(buffer, 0U).value().authoritative;
        const auto before = fixture.api->snapshot();
        Observer observer{owner, fixture};
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U,
                                             restore ? RepresentationState::gpu_raw
                                                     : RepresentationState::gpu_compressed,
                                             &observer));
        const auto after = fixture.api->snapshot();
        VRAMZ_CHECK(runner, after.log_size == before.log_size + expected_order.size());
        VRAMZ_CHECK(runner, std::ranges::equal(std::span{after.log}.subspan(before.log_size,
                                                                            expected_order.size()),
                                               expected_order));
        VRAMZ_CHECK(runner,
                    !after.pending && !after.metadata_publication_pending &&
                        after.metadata_publications == before.metadata_publications + 1U &&
                        after.metadata_consumptions == before.metadata_consumptions + 1U &&
                        after.result_reads == before.result_reads + 1U &&
                        after.consumed_metadata.input_size == source.metadata.stored_size.value() &&
                        after.consumed_metadata.input != 0U &&
                        after.consumed_metadata.output != 0U &&
                        after.consumed_metadata.output_size == 0U &&
                        after.consumed_metadata.status == -1 && observer.valid);
        conserved(runner, owner, fixture);
        read(runner, buffer, payload);
        VRAMZ_CHECK(runner, buffer.close());
        clean(runner, owner, fixture);
    }
    runner.begin("bounded stream operation log saturation cannot affect publication or accounting");
    Fixture fixture;
    auto owner = runtime(fixture);
    auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
    for (unsigned int index = 0U; index < 20U; ++index) {
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U,
                                             index % 2U == 0U ? RepresentationState::gpu_compressed
                                                              : RepresentationState::gpu_raw));
        conserved(runner, owner, fixture);
    }
    const auto audit = fixture.api->snapshot();
    VRAMZ_CHECK(runner, audit.log_size == audit.log.size() && audit.dropped_calls == 16U &&
                            audit.metadata_publications == 20U &&
                            audit.metadata_consumptions == 20U && audit.result_reads == 20U);
    std::array<std::byte, 4096U> zeros{};
    read(runner, buffer, zeros);
    VRAMZ_CHECK(runner, buffer.close());
    clean(runner, owner, fixture);
}

class ReadVictim final : public testing::PolicySelectionObserver {
  public:
    explicit ReadVictim(Buffer& buffer) noexcept : buffer_(buffer) {}
    void on_selected(const ChunkSnapshot&, const policy::Proposal&) noexcept override {
        ++calls;
        auto lease = buffer_.acquire({ByteOffset{}, ByteSize{4096U}});
        valid = valid && lease.has_value();
        if (lease) {
            valid = valid && lease.value().close().has_value();
        }
    }
    bool valid{true};
    unsigned int calls{};

  private:
    Buffer& buffer_;
};
void model_and_concurrency_tests(test::Runner& runner) {
    runner.begin(
        "fake GPU stale policy selection preserves M3 access revision and cycle-wide exclusion");
    Fixture fixture;
    auto configuration = config();
    configuration.policy.mode = PolicyMode::gpu_resident;
    auto owner = runtime(fixture, configuration);
    auto buffer = std::move(owner.allocate(ByteSize{4096U})).value();
    for (unsigned int iteration = 0U; iteration < 16U; ++iteration) {
        testing::RuntimeAccess::advance_policy_epoch(owner, 20U);
        ReadVictim read_victim{buffer};
        VRAMZ_CHECK(runner,
                    !testing::RuntimeAccess::reclaim(owner, ByteSize{64U}, true, &read_victim));
        VRAMZ_CHECK(runner, read_victim.valid && read_victim.calls == 1U);
        VRAMZ_CHECK(runner, fixture.api->snapshot().compressions == 0U);
        conserved(runner, owner, fixture);
    }
    VRAMZ_CHECK(runner, owner.stats().policy.stale_proposal_rejections == 16U);
    VRAMZ_CHECK(runner, buffer.close());
    clean(runner, owner, fixture);
    runner.begin(
        "bounded threaded fake GPU pressure versus reads conserves authority and ownership");
    Fixture concurrent;
    auto concurrent_owner = runtime(concurrent, configuration);
    auto shared = std::move(concurrent_owner.allocate(ByteSize{4096U})).value();
    std::atomic<bool> valid{true};
    std::thread reader{[&]() {
        for (unsigned int index = 0U; index < 64U; ++index) {
            auto lease = shared.acquire({ByteOffset{}, ByteSize{4096U}});
            if (lease) {
                std::array<std::byte, 4096U> output{};
                if (!testing::read_bytes(lease.value(), {}, output) ||
                    !std::ranges::all_of(output,
                                         [](std::byte value) { return value == std::byte{}; }) ||
                    !lease.value().close()) {
                    valid.store(false);
                }
            } else if (lease.error().code != ErrorCode::conflict &&
                       lease.error().code != ErrorCode::busy) {
                valid.store(false);
            }
        }
    }};
    std::thread pressure{[&]() {
        for (unsigned int index = 0U; index < 64U; ++index) {
            testing::RuntimeAccess::advance_policy_epoch(concurrent_owner, 20U);
            const auto reclaimed =
                testing::RuntimeAccess::reclaim(concurrent_owner, ByteSize{64U}, true);
            if (!reclaimed && reclaimed.error().code != ErrorCode::out_of_gpu_memory &&
                reclaimed.error().code != ErrorCode::conflict &&
                reclaimed.error().code != ErrorCode::busy) {
                valid.store(false);
            }
        }
    }};
    reader.join();
    pressure.join();
    VRAMZ_CHECK(runner, valid.load());
    conserved(runner, concurrent_owner, concurrent);
    VRAMZ_CHECK(runner, shared.close());
    clean(runner, concurrent_owner, concurrent);
    runner.begin(
        "deterministic model mixes restore, compaction, access and recoverable status errors");
    Fixture modeled;
    auto modeled_owner = runtime(modeled);
    auto item = std::move(modeled_owner.allocate(ByteSize{4096U})).value();
    for (unsigned int iteration = 0U; iteration < 24U; ++iteration) {
        if (iteration % 3U == 0U) {
            modeled.api->inject(GpuCodecFault::status_read_failure);
        }
        const auto previous = testing::chunk_snapshot(item, 0U).value().authoritative;
        Observer observer{modeled_owner, modeled};
        const auto result = testing::migrate(item, 0U,
                                             previous.state == RepresentationState::gpu_raw
                                                 ? RepresentationState::gpu_compressed
                                                 : RepresentationState::gpu_raw,
                                             &observer);
        const auto current = testing::chunk_snapshot(item, 0U).value().authoritative;
        VRAMZ_CHECK(runner, observer.valid && current.content == previous.content);
        if (!result) {
            VRAMZ_CHECK(runner, current.resource == previous.resource);
        }
        VRAMZ_CHECK(runner, modeled_owner.stats().gpu.cleanup_debt == ByteSize{});
        conserved(runner, modeled_owner, modeled);
    }
    VRAMZ_CHECK(runner, item.close());
    clean(runner, modeled_owner, modeled);
}
} // namespace

int main() {
    test::Runner runner;
    round_trips(runner);
    planning_tests(runner);
    policy_tests(runner);
    sensitivity_tests(runner);
    recoverable_faults(runner);
    death_tests(runner);
    integrity_and_lifetime_tests(runner);
    stream_ordering_tests(runner);
    model_and_concurrency_tests(runner);
    return runner.finish();
}
