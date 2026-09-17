#include "../test_support.hpp"

#include <limits>
#include <type_traits>

using namespace vramz;

namespace {

[[nodiscard]] ChunkSnapshot raw_snapshot() noexcept {
    ChunkSnapshot snapshot{};
    snapshot.lifecycle = LifecycleState::live;
    snapshot.id = ChunkId{2U};
    snapshot.authoritative =
        Representation{RepresentationState::gpu_raw,
                       ResourceId{1U},
                       ByteSize{4096U},
                       ContentTag{1U, 7U},
                       DeviceAddress{64U},
                       RepresentationMetadata{Encoding::raw, ByteSize{4096U}, ByteSize{4096U}, 1U}};
    return snapshot;
}

} // namespace

int main() {
    test::Runner runner;
    PolicyConfig config{};
    const auto snapshot = raw_snapshot();
    policy::Metadata metadata{};
    policy::observe(metadata, snapshot, snapshot.authoritative.charge, 1U);
    policy::access(metadata, 1U, config.tuning);
    static_assert(std::is_trivially_copyable_v<policy::Metadata>);
    static_assert(std::is_nothrow_copy_constructible_v<policy::Proposal>);

    runner.begin("HOT WARM COLD follow deterministic recency without representation changes");
    VRAMZ_CHECK(runner, policy::temperature(metadata, 2U, config.tuning) == Temperature::hot);
    VRAMZ_CHECK(runner, policy::temperature(metadata, 5U, config.tuning) == Temperature::warm);
    VRAMZ_CHECK(runner, policy::temperature(metadata, 20U, config.tuning) == Temperature::cold);
    VRAMZ_CHECK(runner, metadata.observed_state == RepresentationState::gpu_raw);

    runner.begin("frequency raises heat then deterministically decays");
    for (std::uint64_t epoch = 2U; epoch <= 8U; ++epoch) {
        policy::access(metadata, epoch, config.tuning);
    }
    VRAMZ_CHECK(runner, policy::temperature(metadata, 12U, config.tuning) == Temperature::hot);
    VRAMZ_CHECK(runner,
                policy::frequency(metadata, 24U, config.tuning) < metadata.recent_frequency);
    VRAMZ_CHECK(runner, policy::frequency(metadata, 1000U, config.tuning) == 0U);

    runner.begin("normal soft hard critical pressure includes protected reserve and demand");
    const auto pressure = [](std::uint64_t amount, std::uint64_t required = 0U) {
        return policy::pressure(ByteSize{amount}, ByteSize{600U}, ByteSize{1000U}, ByteSize{200U},
                                ByteSize{required});
    };
    VRAMZ_CHECK(runner, pressure(500U) == Pressure::normal);
    VRAMZ_CHECK(runner, pressure(650U) == Pressure::soft);
    VRAMZ_CHECK(runner, pressure(760U) == Pressure::hard);
    VRAMZ_CHECK(runner, pressure(700U, 101U) == Pressure::critical);
    VRAMZ_CHECK(runner, pressure(1000U) == Pressure::critical);

    runner.begin("useful gain must pass both bytes and integer percent thresholds");
    auto tuning = config.tuning;
    tuning.minimum_savings_bytes = ByteSize{1024U};
    tuning.minimum_savings_basis_points = 1000U;
    VRAMZ_CHECK(runner, policy::compression_limit(ByteSize{4096U}, tuning) == ByteSize{3072U});
    tuning.minimum_savings_basis_points = 5000U;
    VRAMZ_CHECK(runner, policy::compression_limit(ByteSize{4096U}, tuning) == ByteSize{2048U});
    VRAMZ_CHECK(runner, !policy::compression_limit(ByteSize{64U}, tuning));

    runner.begin("threshold sensitivity uses actual physical charge not payload or compressBound");
    auto measured = metadata;
    policy::compression_result(measured, ByteSize{3600U}, 20U, config.tuning);
    VRAMZ_CHECK(runner, measured.compressibility == Compressibility::poorly_compressible);
    tuning.minimum_savings_bytes = ByteSize{64U};
    tuning.minimum_savings_basis_points = 1000U;
    policy::compression_result(measured, ByteSize{3600U}, 20U, tuning);
    VRAMZ_CHECK(runner, measured.compressibility == Compressibility::compressible);
    policy::compression_result(measured, ByteSize{4160U}, 20U, tuning);
    VRAMZ_CHECK(runner, measured.compressibility == Compressibility::incompressible);

    runner.begin("incompressible generation is not retried before cooldown");
    const auto skipped =
        policy::propose(measured, snapshot, BufferId{1U}, Pressure::critical, 30U, config);
    VRAMZ_CHECK(runner, skipped.action == PolicyAction::keep);
    const auto reevaluate =
        policy::propose(measured, snapshot, BufferId{1U}, Pressure::critical, 100U, config);
    VRAMZ_CHECK(runner, reevaluate.action == PolicyAction::compress_gpu);

    runner.begin("new writable generation invalidates compressibility without changing authority");
    auto changed = snapshot;
    changed.authoritative.content = next_content_tag(snapshot.authoritative.content);
    policy::observe(measured, changed, changed.authoritative.charge, 31U);
    VRAMZ_CHECK(runner, measured.compressibility == Compressibility::unknown);
    VRAMZ_CHECK(runner, measured.compressed_charge == ByteSize{} && measured.retry_after == 0U);

    runner.begin("minimum raw residency prevents immediate recompression at critical pressure");
    auto recent = metadata;
    recent.raw_since = 29U;
    VRAMZ_CHECK(
        runner,
        policy::propose(recent, snapshot, BufferId{1U}, Pressure::critical, 30U, config).action ==
            PolicyAction::keep);

    runner.begin("transition cooldown applies independently of temperature");
    auto cooling = metadata;
    cooling.last_transition_epoch = 29U;
    VRAMZ_CHECK(
        runner,
        policy::propose(cooling, snapshot, BufferId{1U}, Pressure::critical, 30U, config).action ==
            PolicyAction::keep);

    runner.begin("thrashing produces bounded hold and eventually expires");
    auto thrashing = metadata;
    for (std::uint64_t epoch = 10U; epoch < 14U; ++epoch) {
        policy::transition(thrashing,
                           epoch % 2U == 0U ? RepresentationState::gpu_compressed
                                            : RepresentationState::gpu_raw,
                           epoch, config.tuning);
    }
    VRAMZ_CHECK(runner, thrashing.thrash_events == 1U && thrashing.thrash_until == 29U);
    VRAMZ_CHECK(runner,
                policy::propose(thrashing, snapshot, BufferId{1U}, Pressure::critical, 20U, config)
                        .action == PolicyAction::keep);
    VRAMZ_CHECK(runner,
                policy::propose(thrashing, snapshot, BufferId{1U}, Pressure::critical, 40U, config)
                        .action == PolicyAction::compress_gpu);

    runner.begin("leases intents transition cleanup and POISONED always exclude candidates");
    for (std::uint32_t exclusion = 0U; exclusion < 6U; ++exclusion) {
        auto excluded = snapshot;
        switch (exclusion) {
        case 0U:
            excluded.read_pins = 1U;
            break;
        case 1U:
            excluded.write_pin = true;
            break;
        case 2U:
            excluded.lease_intents = 1U;
            break;
        case 3U:
            excluded.transition_active = true;
            break;
        case 4U:
            excluded.cleanup_resource_count = 1U;
            break;
        case 5U:
            excluded.lifecycle = LifecycleState::poisoned;
            break;
        default:
            std::terminate();
        }
        VRAMZ_CHECK(runner, policy::propose(metadata, excluded, BufferId{1U}, Pressure::critical,
                                            40U, config)
                                    .action == PolicyAction::keep);
    }

    runner.begin("GPU mode never proposes host and fallback mode needs explicit fallback phase");
    VRAMZ_CHECK(runner, policy::propose(metadata, snapshot, BufferId{1U}, Pressure::critical, 40U,
                                        config, true)
                                .action == PolicyAction::keep);
    auto fallback = config;
    fallback.mode = PolicyMode::gpu_resident_with_host_fallback;
    VRAMZ_CHECK(runner,
                policy::propose(metadata, snapshot, BufferId{1U}, Pressure::critical, 40U, fallback)
                        .action == PolicyAction::compress_gpu);
    VRAMZ_CHECK(runner, policy::propose(metadata, snapshot, BufferId{1U}, Pressure::critical, 40U,
                                        fallback, true)
                                .action == PolicyAction::host_fallback);

    runner.begin("deterministic score prioritizes useful bytes and stable ID tie breaks");
    auto candidate =
        policy::propose(metadata, snapshot, BufferId{2U}, Pressure::critical, 40U, config);
    auto other = candidate;
    other.expected_saved_bytes = ByteSize{1U};
    VRAMZ_CHECK(runner, policy::preferred(candidate, other, PolicyStrategy::adaptive));
    other = candidate;
    other.buffer = BufferId{3U};
    VRAMZ_CHECK(runner, policy::preferred(candidate, other, PolicyStrategy::adaptive));
    other = candidate;
    other.chunk = ChunkId{1U};
    VRAMZ_CHECK(runner, policy::preferred(other, candidate, PolicyStrategy::adaptive));

    runner.begin("naive baseline prefers oldest unpinned and ignores performance hysteresis only");
    auto naive = config;
    naive.tuning.strategy = PolicyStrategy::oldest_unpinned;
    VRAMZ_CHECK(
        runner,
        policy::propose(thrashing, snapshot, BufferId{1U}, Pressure::critical, 20U, naive).action ==
            PolicyAction::compress_gpu);
    auto pinned = snapshot;
    pinned.read_pins = 1U;
    VRAMZ_CHECK(
        runner,
        policy::propose(thrashing, pinned, BufferId{1U}, Pressure::critical, 20U, naive).action ==
            PolicyAction::keep);

    runner.begin("disabled policy always keeps the approved baseline");
    auto disabled = config;
    disabled.mode = PolicyMode::disabled;
    VRAMZ_CHECK(runner, policy::propose(metadata, snapshot, BufferId{1U}, Pressure::critical, 100U,
                                        disabled)
                                .action == PolicyAction::keep);

    runner.begin("hot chunks spared at soft pressure but eligible at critical after cooldown");
    auto hot = metadata;
    policy::access(hot, 40U, config.tuning);
    VRAMZ_CHECK(runner,
                policy::propose(hot, snapshot, BufferId{1U}, Pressure::soft, 40U, config).action ==
                    PolicyAction::keep);
    VRAMZ_CHECK(
        runner,
        policy::propose(hot, snapshot, BufferId{1U}, Pressure::critical, 40U, config).action ==
            PolicyAction::compress_gpu);

    runner.begin("maximum-sized ratio and simulated cost arithmetic cannot wrap");
    const ByteSize maximum{std::numeric_limits<std::uint64_t>::max()};
    VRAMZ_CHECK(runner, policy::compression_limit(maximum, config.tuning));
    VRAMZ_CHECK(runner, policy::cost(maximum, maximum.value()) == maximum.value());
    PolicyStats empty{};
    VRAMZ_CHECK(runner, empty.gpu_effective_ratio() == 0.0);
    empty.gpu_raw_charge = maximum;
    empty.gpu_compressed_charge = ByteSize{1U};
    VRAMZ_CHECK(runner, empty.gpu_effective_ratio() == 0.0);

    runner.begin("policy configuration bounds reject invalid fields before resource mutation");
    for (std::uint32_t field = 0U; field < 7U; ++field) {
        auto invalid = config;
        switch (field) {
        case 0U:
            invalid.tuning.maximum_candidates = 0U;
            break;
        case 1U:
            invalid.tuning.maximum_transitions = 1025U;
            break;
        case 2U:
            invalid.tuning.minimum_savings_basis_points = 10001U;
            break;
        case 3U:
            invalid.tuning.frequency_window = 0U;
            break;
        case 4U:
            invalid.tuning.hot_epochs = invalid.tuning.warm_epochs;
            break;
        case 5U:
            invalid.tuning.thrash_transition_count = 1U;
            break;
        case 6U:
            invalid.mode = static_cast<PolicyMode>(255U);
            break;
        default:
            std::terminate();
        }
        VRAMZ_CHECK(runner, !policy::validate(invalid));
    }
    return runner.finish();
}
