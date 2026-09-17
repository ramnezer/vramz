#include "../test_support.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <limits>
#include <sys/wait.h>
#include <unistd.h>

using namespace vramz;

namespace {

[[nodiscard]] RuntimeConfig config() noexcept {
    auto settings = test::test_runtime_config();
    settings.budgets.gpu = TierBudget{ByteSize{32768U}, ByteSize{20480U}, ByteSize{12288U}};
    settings.preferred_chunk_size = ByteSize{4096U};
    settings.policy.mode = PolicyMode::gpu_resident;
    return settings;
}

void death(test::Runner& runner, FaultPoint point, std::uint64_t invocation = 1U,
           FaultPoint companion = FaultPoint::count) {
    const auto child = ::fork();
    VRAMZ_CHECK(runner, child >= 0);
    if (child < 0) {
        return;
    }
    if (child == 0) {
        static_cast<void>(::alarm(5U));
        auto created = Runtime::create(config());
        if (!created) {
            std::_Exit(1);
        }
        auto runtime = std::move(created).value();
        auto allocated = runtime.allocate(ByteSize{4096U});
        if (!allocated) {
            std::_Exit(1);
        }
        auto buffer = std::move(allocated).value();
        static Runtime* observed_runtime{};
        static Buffer* observed_buffer{};
        static ResourceId authority{};
        observed_runtime = &runtime;
        observed_buffer = &buffer;
        authority = testing::chunk_snapshot(buffer, 0U).value().authoritative.resource;
        std::set_terminate([]() noexcept {
            const auto stats = observed_runtime->stats();
            const auto snapshot = testing::chunk_snapshot(*observed_buffer, 0U);
            const auto error = observed_runtime->first_async_error();
            const auto& backend = testing::RuntimeAccess::backend(*observed_runtime);
            const bool valid = snapshot && snapshot.value().authoritative.resource == authority &&
                               snapshot.value().lifecycle == LifecycleState::live &&
                               snapshot.value().transition_active &&
                               snapshot.value().cleanup_resource_count == 0U &&
                               stats.gpu.cleanup_debt == ByteSize{} &&
                               stats.host.cleanup_debt == ByteSize{} &&
                               stats.policy.policy_transitions == 0U &&
                               backend.last_release_resource() == ResourceId{} && error &&
                               error->error.code == ErrorCode::backend_contract_violation;
            std::_Exit(valid ? 86 : 1);
        });
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        testing::inject_failure(runtime, point, invocation);
        if (point == FaultPoint::provisional_charge_mismatch && invocation == 1U) {
            testing::inject_failure(runtime, FaultPoint::allocation_bound_padding);
        }
        if (companion != FaultPoint::count) {
            testing::inject_failure(runtime, companion);
        }
        testing::inject_failure(runtime, FaultPoint::rollback_destination_cleanup);
        static_cast<void>(runtime.reclaim_to_target(ByteSize{2048U}));
        std::_Exit(1);
    }
    int status = 0;
    pid_t waited = 0;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    VRAMZ_CHECK(runner, waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 86);
}

} // namespace

int main() {
    test::Runner runner;
    runner.begin("automatic policy preserves allocation corroboration fatal boundary for "
                 "destinations and workspace");
    for (const auto point :
         {FaultPoint::provisional_charge_mismatch, FaultPoint::adoption_failure,
          FaultPoint::provisional_tier_mismatch, FaultPoint::provisional_kind_mismatch}) {
        death(runner, point);
        death(runner, point, 2U);
    }

    runner.begin(
        "automatic policy preserves transfer corroboration before threshold or cleanup decisions");
    for (const auto point :
         {FaultPoint::receipt_zero_charge, FaultPoint::receipt_plausible_charge_mismatch,
          FaultPoint::allocation_identity_mismatch, FaultPoint::allocation_tier_mismatch,
          FaultPoint::allocation_kind_mismatch}) {
        const bool query = point == FaultPoint::allocation_identity_mismatch ||
                           point == FaultPoint::allocation_tier_mismatch ||
                           point == FaultPoint::allocation_kind_mismatch;
        death(runner, point, query ? 2U : 1U);
    }
    death(runner, FaultPoint::receipt_plausible_charge_mismatch, 1U,
          FaultPoint::receipt_logical_size_mismatch);

    runner.begin("stale policy proposal cannot migrate a different resource generation");
    test::CoreFixture fixture{RepresentationState::gpu_raw};
    const auto source = fixture.chunk->snapshot().authoritative;
    MigrationConstraints stale{source.resource, next_content_tag(source.content), ByteSize{64U}};
    const auto rejected = fixture.coordinator.migrate(
        *fixture.chunk, RepresentationState::gpu_compressed, false, nullptr, &stale);
    VRAMZ_CHECK(runner, !rejected && rejected.error().code == ErrorCode::conflict);
    VRAMZ_CHECK(runner, fixture.chunk->snapshot().authoritative.resource == source.resource);
    VRAMZ_CHECK(runner, !fixture.chunk->snapshot().transition_active);

    runner.begin("verified gain rejection is observed only after both M2 trust barriers and "
                 "preserves exact debt");
    for (const auto cleanup : {false, true}) {
        test::CoreFixture trial{RepresentationState::gpu_raw};
        const auto before = trial.chunk->snapshot().authoritative;
        MigrationConstraints limit{before.resource, before.content, ByteSize{1U}};
        test::CoreAccountingObserver observer{trial};
        if (cleanup) {
            trial.backend.inject_failure(FaultPoint::rollback_destination_cleanup);
        }
        const auto result = trial.coordinator.migrate(
            *trial.chunk, RepresentationState::gpu_compressed, false, &observer, &limit);
        VRAMZ_CHECK(runner,
                    !result && result.error().code == ErrorCode::compression_not_beneficial);
        VRAMZ_CHECK(runner, observer.conserved() && observer.saw(TransactionPhase::verified) &&
                                observer.saw(TransactionPhase::charge_reconciled) &&
                                !observer.saw(TransactionPhase::committed));
        const auto snapshot = trial.chunk->snapshot();
        VRAMZ_CHECK(runner, snapshot.authoritative.resource == before.resource &&
                                snapshot.lifecycle == LifecycleState::live);
        VRAMZ_CHECK(runner, snapshot.cleanup_resource_count == (cleanup ? 1U : 0U));
        if (cleanup) {
            const auto resource =
                testing::RuntimeAccess::cleanup_resource(*trial.chunk, 0U).value();
            VRAMZ_CHECK(runner,
                        trial.ledger.usage(PhysicalTier::gpu).cleanup_debt == resource.charge);
            VRAMZ_CHECK(runner, trial.backend.allocation(resource.resource).value().charge ==
                                    resource.charge);
        }
    }

    runner.begin("post-adoption materialization failure remains exact under automatic policy");
    {
        auto runtime = std::move(Runtime::create(config())).value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        testing::RuntimeAccess::inject_materialize_failure(runtime);
        VRAMZ_CHECK(runner, !runtime.reclaim_to_target(ByteSize{2048U}));
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
        const auto cleanup = testing::RuntimeAccess::cleanup_resource(buffer, 0U, 0U).value();
        VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == cleanup.charge);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U);
    }
    runner.begin("ordinary compression and allocation failures are not stale victim conflicts");
    for (const auto point : {FaultPoint::compression, FaultPoint::gpu_allocation,
                             FaultPoint::workspace_allocation, FaultPoint::transfer}) {
        auto settings = config();
        settings.policy.mode = PolicyMode::gpu_resident_with_host_fallback;
        auto runtime = std::move(Runtime::create(settings)).value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        const auto before = testing::chunk_snapshot(buffer, 0U).value();
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        testing::inject_failure(runtime, point);
        const auto result = testing::RuntimeAccess::reclaim(runtime, ByteSize{2048U}, true);
        VRAMZ_CHECK(runner, !result && result.error().code == ErrorCode::out_of_gpu_memory);
        const auto metadata = testing::RuntimeAccess::policy_metadata(buffer, 0U).value();
        const auto stats = runtime.stats().policy;
        VRAMZ_CHECK(runner, metadata.stale_rejection_cycle == 0U && metadata.last_attempt_failed);
        VRAMZ_CHECK(runner,
                    stats.stale_proposal_rejections == 0U && stats.compression_attempts == 1U &&
                        stats.compression_failures == 1U && stats.compression_rejected == 0U &&
                        stats.host_fallback_count == 0U);
        VRAMZ_CHECK(runner, testing::chunk_snapshot(buffer, 0U).value().authoritative.resource ==
                                before.authoritative.resource);
        for (const auto tier : {PhysicalTier::gpu, PhysicalTier::host}) {
            VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, tier));
        }
        VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == ByteSize{} &&
                                runtime.stats().host.cleanup_debt == ByteSize{} &&
                                testing::owned_resource_count(runtime) == 1U);
        testing::RuntimeAccess::advance_policy_epoch(runtime, 4U);
        VRAMZ_CHECK(runner, testing::RuntimeAccess::reclaim(runtime, ByteSize{2048U}, true));
        VRAMZ_CHECK(runner, runtime.stats().policy.compression_successes == 1U &&
                                runtime.stats().policy.stale_proposal_rejections == 0U);
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        for (const auto tier : {PhysicalTier::gpu, PhysicalTier::host}) {
            VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, tier));
        }
        VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U);
    }

    runner.begin(
        "policy cycle identity is independent of statistics and never reused at exhaustion");
    {
        auto runtime = std::move(Runtime::create(config())).value();
        auto buffer = std::move(runtime.allocate(ByteSize{4096U})).value();
        constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
        VRAMZ_CHECK(runner, testing::RuntimeAccess::advance_policy_cycle(runtime, maximum - 2U));
        VRAMZ_CHECK(runner, runtime.stats().policy.policy_cycles == 0U);
        testing::RuntimeAccess::advance_policy_epoch(runtime, 20U);
        testing::inject_failure(runtime, FaultPoint::compression);
        VRAMZ_CHECK(runner, !runtime.reclaim_to_target(ByteSize{2048U}));
        VRAMZ_CHECK(
            runner,
            testing::RuntimeAccess::policy_metadata(buffer, 0U).value().last_attempt_cycle ==
                maximum - 1U);
        testing::RuntimeAccess::advance_policy_epoch(runtime, 4U);
        VRAMZ_CHECK(runner, runtime.reclaim_to_target(ByteSize{2048U}));
        VRAMZ_CHECK(
            runner,
            testing::RuntimeAccess::policy_metadata(buffer, 0U).value().last_attempt_cycle ==
                maximum);
        const auto before = testing::chunk_snapshot(buffer, 0U).value();
        const auto stats = runtime.stats().policy;
        const auto last_allocation =
            testing::RuntimeAccess::backend(runtime).last_allocated_resource();
        VRAMZ_CHECK(runner, stats.policy_cycles == 2U);
        for (std::uint32_t attempt = 0U; attempt < 3U; ++attempt) {
            const auto result = runtime.reclaim_to_target(ByteSize{});
            VRAMZ_CHECK(runner, !result && result.error().code == ErrorCode::arithmetic_overflow);
            const auto after = testing::chunk_snapshot(buffer, 0U).value();
            VRAMZ_CHECK(runner, after.authoritative.resource == before.authoritative.resource &&
                                    after.authoritative.content == before.authoritative.content &&
                                    !after.transition_active && after.cleanup_resource_count == 0U);
            VRAMZ_CHECK(runner,
                        runtime.stats().policy.policy_cycles == stats.policy_cycles &&
                            runtime.stats().policy.candidates_inspected ==
                                stats.candidates_inspected &&
                            runtime.stats().policy.policy_transitions == stats.policy_transitions);
            VRAMZ_CHECK(runner,
                        testing::RuntimeAccess::backend(runtime).last_allocated_resource() ==
                            last_allocation);
            for (const auto tier : {PhysicalTier::gpu, PhysicalTier::host}) {
                VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, tier));
            }
            VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == ByteSize{} &&
                                    runtime.stats().host.cleanup_debt == ByteSize{} &&
                                    testing::owned_resource_count(runtime) == 1U);
        }
        const auto overflow = testing::RuntimeAccess::advance_policy_cycle(runtime, 1U);
        VRAMZ_CHECK(runner, !overflow && overflow.error().code == ErrorCode::arithmetic_overflow);
        VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, RepresentationState::gpu_raw));
        VRAMZ_CHECK(runner, buffer.close());
        VRAMZ_CHECK(runner, runtime.shutdown());
        VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U);
    }
    return runner.finish();
}
