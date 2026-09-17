#include "../test_support.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <optional>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>
#include <utility>

using namespace vramz;

namespace {

static_assert(std::is_same_v<decltype(std::declval<StorageBackend&>().adopt_allocation(
                                 std::declval<const BackendAllocation&>())),
                             Result<BackendAllocation>>);
static_assert(std::is_trivially_copyable_v<BackendAllocation>);
static_assert(std::is_nothrow_copy_constructible_v<BackendAllocation>);
static_assert(noexcept(Result<BackendAllocation>{BackendAllocation{}}));

enum class Site : std::uint8_t {
    initial,
    gpu_destination,
    host_destination,
    gpu_workspace,
    host_workspace
};

enum class Companion : std::uint8_t {
    none,
    adoption,
    materialization,
    transfer,
    destination_cleanup,
    workspace_cleanup,
    close_cleanup
};

inline constexpr std::array sites{Site::initial, Site::gpu_destination, Site::host_destination,
                                  Site::gpu_workspace, Site::host_workspace};

[[nodiscard]] constexpr bool workspace_site(Site site) noexcept {
    return site == Site::gpu_workspace || site == Site::host_workspace;
}

[[nodiscard]] constexpr RepresentationState source_state(Site site) noexcept {
    return site == Site::gpu_destination ? RepresentationState::host_raw
                                         : RepresentationState::gpu_raw;
}

[[nodiscard]] constexpr RepresentationState destination_state(Site site) noexcept {
    switch (site) {
    case Site::initial:
    case Site::gpu_destination:
        return RepresentationState::gpu_raw;
    case Site::host_destination:
        return RepresentationState::host_raw;
    case Site::gpu_workspace:
        return RepresentationState::gpu_compressed;
    case Site::host_workspace:
        return RepresentationState::host_compressed;
    }
    return RepresentationState::gpu_raw;
}

void arm_descriptor(MockBackend& backend, Site site, FaultPoint point) noexcept {
    if (point == FaultPoint::count) {
        return;
    }
    if (point == FaultPoint::provisional_charge_mismatch && !workspace_site(site)) {
        // Admission headroom makes this false charge locally plausible; the real charge stays 256.
        backend.inject_failure(FaultPoint::allocation_bound_padding);
    }
    backend.inject_failure(point, workspace_site(site) ? 2U : 1U);
}

void arm_companion(MockBackend& backend, Site site, Companion companion) noexcept {
    switch (companion) {
    case Companion::adoption:
        backend.inject_failure(FaultPoint::adoption_failure, workspace_site(site) ? 2U : 1U);
        break;
    case Companion::transfer:
        backend.inject_failure(FaultPoint::transfer);
        break;
    case Companion::destination_cleanup:
        backend.inject_failure(FaultPoint::rollback_destination_cleanup);
        break;
    case Companion::workspace_cleanup:
        backend.inject_failure(FaultPoint::rollback_workspace_cleanup);
        break;
    case Companion::close_cleanup:
        backend.inject_failure(FaultPoint::close_cleanup);
        break;
    case Companion::none:
    case Companion::materialization:
        break;
    }
}

[[nodiscard]] bool same_authority(const ChunkSnapshot& before,
                                  const ChunkSnapshot& after) noexcept {
    return before.authoritative.resource == after.authoritative.resource &&
           before.authoritative.state == after.authoritative.state &&
           before.authoritative.charge == after.authoritative.charge &&
           before.authoritative.content == after.authoritative.content &&
           before.authoritative.metadata.logical_size ==
               after.authoritative.metadata.logical_size &&
           before.authoritative.metadata.stored_size == after.authoritative.metadata.stored_size &&
           before.authoritative.metadata.encoding == after.authoritative.metadata.encoding &&
           before.authoritative.metadata.crc32c == after.authoritative.metadata.crc32c &&
           before.transition_epoch == after.transition_epoch;
}

void expect_fatal(test::Runner& runner, Site site, FaultPoint point,
                  Companion companion = Companion::none,
                  FaultPoint second_descriptor = FaultPoint::count) {
    const auto child = ::fork();
    VRAMZ_CHECK(runner, child >= 0);
    if (child < 0) {
        return;
    }
    if (child == 0) {
        static_cast<void>(::alarm(5U));
        if (site == Site::initial) {
            auto config = test::test_runtime_config();
            config.preferred_chunk_size = ByteSize{256U};
            auto created = Runtime::create(config);
            if (!created) {
                std::_Exit(1);
            }
            auto runtime = std::move(created).value();
            static Runtime* child_runtime{};
            static bool adoption_expected{};
            static bool false_charge{};
            child_runtime = &runtime;
            false_charge = point == FaultPoint::provisional_charge_mismatch;
            adoption_expected = point == FaultPoint::provisional_charge_mismatch ||
                                point == FaultPoint::provisional_metadata_mismatch ||
                                point == FaultPoint::adoption_failure;
            std::set_terminate([]() noexcept {
                const auto stats = child_runtime->stats();
                const auto error = child_runtime->first_async_error();
                const auto& backend = testing::RuntimeAccess::backend(*child_runtime);
                bool charge_witness = true;
                if (false_charge) {
                    const auto actual = backend.allocation(backend.last_allocated_resource());
                    charge_witness = actual && actual.value().charge == ByteSize{256U} && error &&
                                     error->error.detail == 320U &&
                                     error->error.object_id == actual.value().id.value();
                }
                const bool stopped =
                    stats.gpu.committed == ByteSize{} && stats.host.committed == ByteSize{} &&
                    stats.gpu.staging == ByteSize{} && stats.host.staging == ByteSize{} &&
                    stats.gpu.workspace == ByteSize{} && stats.host.workspace == ByteSize{} &&
                    stats.gpu.cleanup_debt == ByteSize{} && stats.host.cleanup_debt == ByteSize{} &&
                    backend.last_release_resource() == ResourceId{} &&
                    backend.adoption_attempt_count() == (adoption_expected ? 1U : 0U) &&
                    charge_witness && error &&
                    error->error.code == ErrorCode::backend_contract_violation;
                // No conservation assertion is valid for the uncertain resource in a dying process.
                std::_Exit(stopped ? 86 : 1);
            });
            auto& backend = testing::RuntimeAccess::backend(runtime);
            arm_descriptor(backend, site, point);
            arm_descriptor(backend, site, second_descriptor);
            arm_companion(backend, site, companion);
            if (companion == Companion::materialization) {
                testing::RuntimeAccess::inject_materialize_failure(runtime);
            }
            static_cast<void>(runtime.allocate(ByteSize{256U}));
        } else {
            test::CoreFixture fixture{source_state(site)};
            if (!fixture.ready()) {
                std::_Exit(1);
            }
            test::RecordingObserver observer{source_state(site)};
            static test::CoreFixture* child_fixture{};
            static test::RecordingObserver* child_observer{};
            static ChunkSnapshot before{};
            static ResourceId previous_release{};
            static std::uint64_t expected_adoptions{};
            static bool workspace{};
            static bool false_charge{};
            child_fixture = &fixture;
            child_observer = &observer;
            before = fixture.chunk->snapshot();
            previous_release = fixture.backend.last_release_resource();
            workspace = workspace_site(site);
            false_charge = point == FaultPoint::provisional_charge_mismatch;
            const bool adoption_expected = point == FaultPoint::provisional_charge_mismatch ||
                                           point == FaultPoint::provisional_metadata_mismatch ||
                                           point == FaultPoint::adoption_failure;
            expected_adoptions = fixture.backend.adoption_attempt_count() + (workspace ? 1U : 0U) +
                                 (adoption_expected ? 1U : 0U);
            std::set_terminate([]() noexcept {
                const auto after = child_fixture->chunk->snapshot();
                const auto gpu = child_fixture->ledger.usage(PhysicalTier::gpu);
                const auto host = child_fixture->ledger.usage(PhysicalTier::host);
                const auto error = child_fixture->errors.first();
                bool charge_witness = true;
                if (false_charge) {
                    const auto actual = child_fixture->backend.allocation(
                        child_fixture->backend.last_allocated_resource());
                    charge_witness = actual &&
                                     actual.value().charge == ByteSize{workspace ? 320U : 256U} &&
                                     error && error->error.detail == (workspace ? 256U : 320U) &&
                                     error->error.object_id == actual.value().id.value();
                }
                const bool stopped =
                    after.lifecycle == LifecycleState::live && after.transition_active &&
                    same_authority(before, after) && after.cleanup_resource_count == 0U &&
                    gpu.cleanup_debt == ByteSize{} && host.cleanup_debt == ByteSize{} &&
                    gpu.workspace == ByteSize{} && host.workspace == ByteSize{} &&
                    child_fixture->backend.last_release_resource() == previous_release &&
                    child_fixture->backend.adoption_attempt_count() == expected_adoptions &&
                    child_observer->saw(TransactionPhase::destination_materialized) == workspace &&
                    !child_observer->saw(TransactionPhase::workspace_materialized) &&
                    !child_observer->saw(TransactionPhase::charge_reconciled) &&
                    !child_observer->saw(TransactionPhase::transferred) &&
                    !child_observer->saw(TransactionPhase::rolled_back) &&
                    !child_observer->saw(TransactionPhase::poisoned) &&
                    !child_observer->saw(TransactionPhase::committed) && charge_witness && error &&
                    error->error.code == ErrorCode::backend_contract_violation;
                std::_Exit(stopped ? 86 : 1);
            });
            arm_descriptor(fixture.backend, site, point);
            arm_descriptor(fixture.backend, site, second_descriptor);
            arm_companion(fixture.backend, site, companion);
            if (companion == Companion::materialization) {
                testing::RuntimeAccess::inject_materialize_failure(fixture.ledger,
                                                                   workspace ? 2U : 1U);
            }
            static_cast<void>(fixture.coordinator.migrate(*fixture.chunk, destination_state(site),
                                                          false, &observer));
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

void check_empty(test::Runner& runner, Runtime& runtime) {
    const auto stats = runtime.stats();
    for (const auto& usage : {stats.gpu, stats.host}) {
        VRAMZ_CHECK(runner, usage.committed == ByteSize{} && usage.reserved == ByteSize{} &&
                                usage.staging == ByteSize{} && usage.workspace == ByteSize{} &&
                                usage.cleanup_debt == ByteSize{});
    }
    VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U);
}

void check_conservation(test::Runner& runner, Runtime& runtime) {
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
}

class RuntimeAccountingObserver final : public TransactionObserver {
  public:
    RuntimeAccountingObserver(const Runtime& runtime, ChunkSnapshot before) noexcept
        : runtime_(runtime), before_(before) {}

    void on_phase(TransactionPhase phase, const ChunkSnapshot& snapshot) noexcept override {
        conserved_ = conserved_ && testing::accounting_conserved(runtime_, PhysicalTier::gpu) &&
                     testing::accounting_conserved(runtime_, PhysicalTier::host);
        authority_unchanged_ = authority_unchanged_ && same_authority(before_, snapshot);
        committed_ = committed_ || phase == TransactionPhase::committed;
    }

    [[nodiscard]] bool valid() const noexcept {
        return conserved_ && authority_unchanged_ && !committed_;
    }

  private:
    const Runtime& runtime_;
    const ChunkSnapshot before_;
    bool conserved_{true};
    bool authority_unchanged_{true};
    bool committed_{};
};

void expect_recoverable(test::Runner& runner, Site site, Companion first, Companion second) {
    auto config = test::test_runtime_config();
    config.preferred_chunk_size = ByteSize{256U};
    auto created = Runtime::create(config);
    VRAMZ_CHECK(runner, created);
    if (!created) {
        return;
    }
    auto runtime = std::move(created).value();
    auto& backend = testing::RuntimeAccess::backend(runtime);
    std::optional<Buffer> buffer{};
    ChunkSnapshot before{};
    if (site != Site::initial) {
        auto allocated = runtime.allocate(ByteSize{256U});
        VRAMZ_CHECK(runner, allocated);
        if (!allocated) {
            return;
        }
        buffer.emplace(std::move(allocated).value());
        if (source_state(site) != RepresentationState::gpu_raw) {
            VRAMZ_CHECK(runner, testing::migrate(*buffer, 0U, source_state(site)));
        }
        const auto snapshot = testing::chunk_snapshot(*buffer, 0U);
        VRAMZ_CHECK(runner, snapshot);
        if (!snapshot) {
            return;
        }
        before = snapshot.value();
    }
    const auto previous_adoptions = backend.adoption_attempt_count();
    arm_companion(backend, site, first);
    arm_companion(backend, site, second);
    const bool materialize_fails =
        first == Companion::materialization || second == Companion::materialization;
    const bool close_fails =
        first == Companion::close_cleanup || second == Companion::close_cleanup;
    if (materialize_fails) {
        testing::RuntimeAccess::inject_materialize_failure(runtime, workspace_site(site) ? 2U : 1U);
    }
    if (site == Site::initial) {
        const auto allocated = runtime.allocate(ByteSize{256U});
        VRAMZ_CHECK(runner, !allocated && materialize_fails);
        VRAMZ_CHECK(runner, runtime.stats().gpu.committed == ByteSize{});
        VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == ByteSize{256U});
        const auto exact = backend.allocation(backend.last_allocated_resource());
        VRAMZ_CHECK(runner, exact && exact.value().charge == ByteSize{256U} &&
                                exact.value().tier == PhysicalTier::gpu &&
                                exact.value().kind == ResourceKind::representation);
        VRAMZ_CHECK(runner, backend.adoption_attempt_count() == previous_adoptions + 1U);
        VRAMZ_CHECK(runner, !backend.is_unadopted(backend.last_allocated_resource()));
    } else {
        RuntimeAccountingObserver observer{runtime, before};
        const auto migrated = testing::migrate(*buffer, 0U, destination_state(site), &observer);
        VRAMZ_CHECK(runner, !migrated);
        const auto after = testing::chunk_snapshot(*buffer, 0U);
        VRAMZ_CHECK(runner, after);
        if (!after) {
            return;
        }
        VRAMZ_CHECK(runner, same_authority(before, after.value()));
        VRAMZ_CHECK(runner, !after.value().transition_active && observer.valid());
        const auto adoptions = workspace_site(site) ? 2U : 1U;
        VRAMZ_CHECK(runner, backend.adoption_attempt_count() == previous_adoptions + adoptions);
        std::array<ByteSize, 2U> debt{};
        for (std::uint32_t index = 0U; index < after.value().cleanup_resource_count; ++index) {
            const auto cleanup = testing::RuntimeAccess::cleanup_resource(*buffer, 0U, index);
            VRAMZ_CHECK(runner, cleanup);
            if (!cleanup) {
                continue;
            }
            const auto exact = backend.allocation(cleanup.value().resource);
            VRAMZ_CHECK(runner, exact && exact.value().id == cleanup.value().resource &&
                                    exact.value().charge == cleanup.value().charge &&
                                    exact.value().tier == cleanup.value().tier &&
                                    !backend.is_unadopted(cleanup.value().resource));
            const auto tier_index = cleanup.value().tier == PhysicalTier::gpu ? 0U : 1U;
            const auto summed =
                checked_add(debt[tier_index].value(), cleanup.value().charge.value(),
                            OperationId::budget_transfer);
            VRAMZ_CHECK(runner, summed);
            if (summed) {
                debt[tier_index] = ByteSize{summed.value()};
            }
        }
        const auto stats = runtime.stats();
        VRAMZ_CHECK(runner,
                    stats.gpu.cleanup_debt == debt[0] && stats.host.cleanup_debt == debt[1]);
        VRAMZ_CHECK(runner,
                    backend.owned_resource_count() == after.value().cleanup_resource_count + 1U);
        std::array<std::byte, 256U> bytes{};
        VRAMZ_CHECK(runner, backend.read_bytes(before.authoritative.resource, ByteOffset{}, bytes));
        VRAMZ_CHECK(runner, std::all_of(bytes.begin(), bytes.end(),
                                        [](std::byte value) { return value == std::byte{}; }));
    }
    check_conservation(runner, runtime);
    const auto stats = runtime.stats();
    for (const auto& usage : {stats.gpu, stats.host}) {
        VRAMZ_CHECK(runner, usage.reserved == ByteSize{} && usage.staging == ByteSize{} &&
                                usage.workspace == ByteSize{});
    }
    if (close_fails) {
        const auto cleanup = buffer ? buffer->close() : runtime.shutdown();
        VRAMZ_CHECK(runner, !cleanup);
        check_conservation(runner, runtime);
        VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == stats.gpu.cleanup_debt &&
                                runtime.stats().host.cleanup_debt == stats.host.cleanup_debt);
    }
    if (buffer) {
        VRAMZ_CHECK(runner, buffer->close());
        check_conservation(runner, runtime);
    }
    VRAMZ_CHECK(runner, runtime.shutdown());
    check_conservation(runner, runtime);
    check_empty(runner, runtime);
}

void check_adoption_fields(test::Runner& runner) {
    for (const auto state : test::all_states) {
        for (std::uint32_t field = 0U; field < 8U; ++field) {
            MockBackend backend{test::test_backend_config()};
            const auto address = backend.reserve_address_space(ByteSize{256U}, ByteSize{64U});
            VRAMZ_CHECK(runner, address);
            const auto provisional =
                backend.allocate_representation(RepresentationAllocationRequest{
                    state, ByteSize{256U}, ContentTag{1U, 2U},
                    state == RepresentationState::gpu_raw ? address.value().base
                                                          : DeviceAddress{}});
            VRAMZ_CHECK(runner, provisional);
            if (!provisional) {
                continue;
            }
            auto changed = provisional.value();
            switch (field) {
            case 0U:
                changed.id = ResourceId{};
                break;
            case 1U:
                changed.tier =
                    changed.tier == PhysicalTier::gpu ? PhysicalTier::host : PhysicalTier::gpu;
                break;
            case 2U:
                changed.charge = ByteSize{};
                break;
            case 3U:
                changed.kind = ResourceKind::workspace;
                break;
            case 4U:
                changed.metadata.encoding = changed.metadata.encoding == Encoding::raw
                                                ? Encoding::lz4_block
                                                : Encoding::raw;
                break;
            case 5U:
                changed.metadata.logical_size = ByteSize{};
                break;
            case 6U:
                changed.metadata.stored_size = ByteSize{123U};
                break;
            case 7U:
                changed.metadata.crc32c ^= 1U;
                break;
            default:
                std::terminate();
            }
            const auto rejected = backend.adopt_allocation(changed);
            VRAMZ_CHECK(runner, !rejected &&
                                    rejected.error().code == ErrorCode::backend_contract_violation);
            VRAMZ_CHECK(runner, backend.is_unadopted(provisional.value().id));
            const auto adopted = backend.adopt_allocation(provisional.value());
            VRAMZ_CHECK(runner,
                        adopted && adopted.value().id == provisional.value().id &&
                            adopted.value().tier == provisional.value().tier &&
                            adopted.value().charge == provisional.value().charge &&
                            adopted.value().kind == provisional.value().kind &&
                            adopted.value().metadata.crc32c == provisional.value().metadata.crc32c);
            if (adopted) {
                VRAMZ_CHECK(runner, backend.release(adopted.value().id, ReleasePhase::close));
            }
            VRAMZ_CHECK(runner, backend.owned_resource_count() == 0U);
        }
    }
}

void pairwise_allocation_model(test::Runner& runner) {
    // Only applicable allocation-region classes are combined; no unrelated global fault product.
    struct Choice final {
        FaultPoint descriptor{FaultPoint::count};
        Companion companion{Companion::none};
    };
    constexpr Choice charge{FaultPoint::provisional_charge_mismatch, Companion::none};
    constexpr Choice metadata{FaultPoint::provisional_metadata_mismatch, Companion::none};
    constexpr Choice adoption{FaultPoint::count, Companion::adoption};
    constexpr Choice materialization{FaultPoint::count, Companion::materialization};
    constexpr Choice transfer{FaultPoint::count, Companion::transfer};
    constexpr Choice destination_cleanup{FaultPoint::count, Companion::destination_cleanup};
    constexpr Choice workspace_cleanup{FaultPoint::count, Companion::workspace_cleanup};
    constexpr Choice close_cleanup{FaultPoint::count, Companion::close_cleanup};
    std::uint32_t fatal_count = 0U;
    std::uint32_t recoverable_count = 0U;
    for (const auto site : sites) {
        const std::array choices =
            site == Site::initial
                ? std::array{charge, metadata, adoption, materialization, close_cleanup, Choice{}}
                : (workspace_site(site) ? std::array{charge, adoption, materialization, transfer,
                                                     destination_cleanup, workspace_cleanup}
                                        : std::array{charge, metadata, adoption, materialization,
                                                     transfer, destination_cleanup});
        const std::size_t count = site == Site::initial ? 5U : 6U;
        for (std::size_t left = 0U; left < count; ++left) {
            for (std::size_t right = left + 1U; right < count; ++right) {
                const auto a = choices[left];
                const auto b = choices[right];
                if (a.descriptor != FaultPoint::count || b.descriptor != FaultPoint::count ||
                    a.companion == Companion::adoption || b.companion == Companion::adoption) {
                    if (a.descriptor != FaultPoint::count && b.descriptor != FaultPoint::count) {
                        expect_fatal(runner, site, a.descriptor, Companion::none, b.descriptor);
                    } else if (a.descriptor != FaultPoint::count ||
                               b.descriptor != FaultPoint::count) {
                        const auto descriptor =
                            a.descriptor != FaultPoint::count ? a.descriptor : b.descriptor;
                        const auto companion =
                            a.descriptor != FaultPoint::count ? b.companion : a.companion;
                        expect_fatal(runner, site, descriptor, companion);
                    } else {
                        const auto companion =
                            a.companion == Companion::adoption ? b.companion : a.companion;
                        expect_fatal(runner, site, FaultPoint::adoption_failure, companion);
                    }
                    ++fatal_count;
                } else {
                    expect_recoverable(runner, site, a.companion, b.companion);
                    ++recoverable_count;
                }
            }
        }
    }
    VRAMZ_CHECK(runner, fatal_count == 51U && recoverable_count == 19U);
}

} // namespace

int main() {
    test::Runner runner;

    runner.begin("adoption corroborates all eight representation fields for all four states");
    check_adoption_fields(runner);

    runner.begin("workspace adoption ignores irrelevant payload metadata and returns the retained "
                 "descriptor");
    for (const auto tier : {PhysicalTier::gpu, PhysicalTier::host}) {
        MockBackend backend{test::test_backend_config()};
        const auto allocation = backend.allocate_workspace(tier, ByteSize{320U});
        VRAMZ_CHECK(runner, allocation);
        if (!allocation) {
            continue;
        }
        auto provisional = allocation.value();
        provisional.metadata =
            RepresentationMetadata{Encoding::lz4_block, ByteSize{999U}, ByteSize{777U}, 42U};
        const auto adopted = backend.adopt_allocation(provisional);
        VRAMZ_CHECK(runner,
                    adopted && adopted.value().id == allocation.value().id &&
                        adopted.value().tier == tier && adopted.value().charge == ByteSize{320U} &&
                        adopted.value().kind == ResourceKind::workspace &&
                        adopted.value().metadata.logical_size ==
                            allocation.value().metadata.logical_size &&
                        adopted.value().metadata.crc32c == allocation.value().metadata.crc32c);
        if (adopted) {
            VRAMZ_CHECK(runner, backend.release(adopted.value().id, ReleasePhase::close));
        }
    }

    runner.begin("plausible false initial charge stops at adoption without publishing a buffer");
    expect_fatal(runner, Site::initial, FaultPoint::provisional_charge_mismatch);

    runner.begin("plausible false destination charge stops at adoption in both tiers");
    for (const auto site : {Site::gpu_destination, Site::host_destination}) {
        expect_fatal(runner, site, FaultPoint::provisional_charge_mismatch);
    }

    runner.begin("plausible false workspace charge stops at adoption in both tiers");
    for (const auto site : {Site::gpu_workspace, Site::host_workspace}) {
        expect_fatal(runner, site, FaultPoint::provisional_charge_mismatch);
    }

    runner.begin(
        "initial allocation cannot publish a plausible false CRC as authoritative metadata");
    expect_fatal(runner, Site::initial, FaultPoint::provisional_metadata_mismatch);

    runner.begin("representation provisional structural faults fail before adoption or debt");
    for (const auto point :
         {FaultPoint::provisional_tier_mismatch, FaultPoint::provisional_kind_mismatch,
          FaultPoint::provisional_logical_size_mismatch,
          FaultPoint::provisional_stored_size_mismatch, FaultPoint::provisional_encoding_mismatch,
          FaultPoint::provisional_zero_charge, FaultPoint::provisional_unaligned_charge,
          FaultPoint::invalid_resource_identity}) {
        for (const auto site : {Site::initial, Site::gpu_destination, Site::host_destination}) {
            expect_fatal(runner, site, point);
        }
    }

    runner.begin("workspace provisional structural and alias faults are fatal in both tiers");
    for (const auto point :
         {FaultPoint::provisional_tier_mismatch, FaultPoint::provisional_kind_mismatch,
          FaultPoint::provisional_zero_charge, FaultPoint::provisional_unaligned_charge,
          FaultPoint::invalid_resource_identity, FaultPoint::duplicate_resource_identity}) {
        for (const auto site : {Site::gpu_workspace, Site::host_workspace}) {
            expect_fatal(runner, site, point);
        }
    }

    runner.begin("explicit adoption refusal stops all five allocation sites before accounting");
    for (const auto site : sites) {
        expect_fatal(runner, site, FaultPoint::adoption_failure);
    }

    runner.begin("uncorroborated allocation never reaches armed cleanup failures");
    for (const auto site : sites) {
        expect_fatal(runner, site, FaultPoint::provisional_charge_mismatch,
                     site == Site::initial ? Companion::close_cleanup
                                           : Companion::destination_cleanup);
    }

    runner.begin("post-adoption materialization failures retain exact debt and retry cleanup once");
    for (const auto site : sites) {
        expect_recoverable(runner, site, Companion::materialization, Companion::none);
        expect_recoverable(runner, site, Companion::materialization, Companion::close_cleanup);
    }

    runner.begin("post-adoption transfer and cleanup failures preserve exact resource ownership");
    for (const auto site : {Site::gpu_destination, Site::host_destination, Site::gpu_workspace,
                            Site::host_workspace}) {
        expect_recoverable(runner, site, Companion::transfer, Companion::destination_cleanup);
        if (workspace_site(site)) {
            expect_recoverable(runner, site, Companion::transfer, Companion::workspace_cleanup);
        }
    }

    runner.begin("bounded allocation pairwise model preserves barriers ownership and conservation");
    pairwise_allocation_model(runner);

    return runner.finish();
}
