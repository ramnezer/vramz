#include "../support/fake_cuda_driver.hpp"
#include "../support/fake_nvcomp.hpp"
#include "../test_support.hpp"
#include "vramz/detail/policy_pressure_smoke.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <optional>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace vramz;
namespace {
constexpr std::uint64_t mib{1024ULL * 1024U};
struct Fixture final {
    std::shared_ptr<test::FakeCudaState> cuda{std::make_shared<test::FakeCudaState>()};
    std::shared_ptr<test::FakeGpuCodecState> codec{std::make_shared<test::FakeGpuCodecState>()};
    std::unique_ptr<test::FakeCudaDriverApi> driver{
        std::make_unique<test::FakeCudaDriverApi>(cuda)};
    std::unique_ptr<test::FakeNvcompLz4Api> api{
        std::make_unique<test::FakeNvcompLz4Api>(*driver, codec)};
    detail::PolicyPressureSmokeReport report{};
    Fixture() {
        cuda->config.minimum = ByteSize{2U * mib};
        cuda->config.recommended = cuda->config.minimum;
        cuda->config.version = 13020;
        cuda->config.physical_limit = detail::m11_physical_cap;
        cuda->config.maximum_resource = detail::m11_physical_cap;
        cuda->config.device_info.name = std::array<char, 256U>{"NVIDIA GeForce RTX 3060"};
        codec->cuda_audit = cuda;
    }
    void run(detail::PolicyPressureObserver* observer = nullptr) {
        detail::run_policy_pressure_smoke(std::move(driver), std::move(api),
                                          {true, 0, {595U, 84U, 0U}}, report, observer);
    }
    void empty(test::Runner& runner) const {
        VRAMZ_CHECK(runner, report.final_counts_known && report.final_resources == 0U &&
                                report.final_va == 0U);
        VRAMZ_CHECK(runner,
                    detail::policy_pressure_charge(report.final_usage).value() == ByteSize{});
        VRAMZ_CHECK(runner, cuda->owned_bytes == 0U && cuda->primary_references == 0U);
        VRAMZ_CHECK(runner, codec->observation.streams == 0U && !codec->observation.pending);
        VRAMZ_CHECK(runner, cuda->dropped_calls == 0U && codec->observation.dropped_calls == 0U);
    }
};
class Audit final : public detail::PolicyPressureObserver {
  public:
    void observe(detail::PolicyPressureEvent event) noexcept override {
        if (count < events.size()) {
            events[count++] = event;
        } else {
            overflow = true;
        }
    }
    std::array<detail::PolicyPressureEvent, 16U> events{};
    std::size_t count{};
    bool overflow{};
};
void baseline(test::Runner& runner) {
    runner.begin("M11 four exact M10 chunks, automatic production policy victims, HOT protection, "
                 "settled snapshot and zero cleanup");
    Fixture fixture;
    Audit audit;
    fixture.run(&audit);
    const auto& r = fixture.report;
    if (r.has_error) {
        std::cerr << "M11 error=" << static_cast<unsigned>(r.error.code)
                  << " operation=" << static_cast<unsigned>(r.error.operation) << '\n';
    }
    VRAMZ_CHECK(runner, r.result == detail::CompressionSmokeResult::passed);
    VRAMZ_CHECK(runner, r.initial_snapshot_proven && r.policy_settled_snapshot_proven);
    VRAMZ_CHECK(runner, r.aggregate_raw_charge == ByteSize{32U * mib});
    VRAMZ_CHECK(runner, r.aggregate_settled_charge == ByteSize{20U * mib} &&
                            r.reclaimed == ByteSize{12U * mib});
    VRAMZ_CHECK(runner,
                r.initial_pressure == Pressure::hard && r.settled_pressure == Pressure::normal);
    VRAMZ_CHECK(runner, r.hot_chunk_preserved_raw && r.pressure_target_reached);
    VRAMZ_CHECK(runner, r.cycle_count == 2U && r.automatic_policy_actions == 2U);
    VRAMZ_CHECK(runner, r.cycles[0].selected_chunk == 1U && r.cycles[1].selected_chunk == 2U);
    const std::array crcs{383231952U, 846517591U, 3954088913U, 243520057U};
    const std::array epochs{30U, 9U, 10U, 27U};
    const std::array revisions{20U, 2U, 2U, 2U};
    for (std::size_t i = 0U; i < r.chunks.size(); ++i) {
        const auto& c = r.chunks[i];
        VRAMZ_CHECK(runner, c.integrity.logical_crc_expected == crcs[i] &&
                                c.integrity.logical_crc_actual == crcs[i]);
        VRAMZ_CHECK(runner,
                    c.access.last_access_epoch == epochs[i] && c.access_revision == revisions[i]);
        VRAMZ_CHECK(runner,
                    c.integrity_verified && c.integrity.byte_equal && c.integrity.cleanup_complete);
        if (i == 1U || i == 2U) {
            VRAMZ_CHECK(runner, c.integrity.charges_corroborated && c.integrity.minimum_unit_saved);
            VRAMZ_CHECK(runner, c.integrity.compressed_charge == ByteSize{2U * mib});
            VRAMZ_CHECK(runner,
                        c.integrity.compression_completion && c.integrity.decompression_completion);
            VRAMZ_CHECK(runner, c.integrity.stored_crc_expected == c.integrity.stored_crc_actual);
        } else {
            VRAMZ_CHECK(runner,
                        c.selected_count == 0U && c.settled == RepresentationState::gpu_raw);
        }
    }
    VRAMZ_CHECK(runner, !audit.overflow && audit.count == 10U);
    VRAMZ_CHECK(runner,
                audit.events[3].kind == detail::PolicyPressureEventKind::policy_settled_snapshot);
    VRAMZ_CHECK(runner, audit.events[4].kind == detail::PolicyPressureEventKind::integrity_start);
    VRAMZ_CHECK(runner, audit.events[8].kind == detail::PolicyPressureEventKind::cleanup_complete);
    VRAMZ_CHECK(runner, audit.events[9].kind == detail::PolicyPressureEventKind::pass_published);
    VRAMZ_CHECK(runner, fixture.codec->observation.compressions == 2U &&
                            fixture.codec->observation.decompressions == 2U);
    const std::array expected_order{
        test::GpuCodecCall::metadata_copy_enqueue, test::GpuCodecCall::nvcomp_launch,
        test::GpuCodecCall::stream_synchronize, test::GpuCodecCall::metadata_status_read};
    VRAMZ_CHECK(runner, fixture.codec->observation.log_size == 16U);
    for (std::size_t operation = 0U; operation < 4U; ++operation) {
        VRAMZ_CHECK(runner, std::ranges::equal(std::span{fixture.codec->observation.log}.subspan(
                                                   operation * 4U, 4U),
                                               expected_order));
    }
    VRAMZ_CHECK(runner, fixture.codec->observation.metadata_publications == 4U &&
                            fixture.codec->observation.metadata_consumptions == 4U);
    for (const auto& cycle : std::span{r.cycles}.first(r.cycle_count)) {
        VRAMZ_CHECK(runner, cycle.workspace_materializations == 2U && cycle.charge_reconciled);
        VRAMZ_CHECK(runner, cycle.stage_count > 8U && cycle.stage_count <= cycle.stages.size());
        for (const auto& stage : std::span{cycle.stages}.first(cycle.stage_count)) {
            const auto charge = detail::policy_pressure_charge(stage.usage);
            VRAMZ_CHECK(runner,
                        charge && charge.value().value() == stage.owned_physical_charge.value() +
                                                                stage.usage.reserved.value());
            VRAMZ_CHECK(runner, charge.value() <= r.peak_admitted_gpu_bytes);
        }
    }
    fixture.empty(runner);
    std::array<char, 32768U> json{};
    const auto formatted = detail::format_policy_pressure_smoke_report(r, "", false, json);
    VRAMZ_CHECK(runner, !formatted.truncated);
    std::cout << "M11_FAKE_RESULT " << std::string_view{json.data(), formatted.written} << '\n';
}
class Inject final : public detail::PolicyPressureObserver {
  public:
    Inject(Fixture& fixture, detail::PolicyPressureEventKind when, std::uint32_t index,
           test::CudaCall call, test::CudaFaultMode mode, std::uint64_t occurrence)
        : fixture_(fixture), driver_(*fixture.driver), codec_(*fixture.api), when_(when),
          index_(index), call_(call), mode_(mode), occurrence_(occurrence) {}
    void observe(detail::PolicyPressureEvent event) noexcept override {
        if (event.kind == when_ && event.index == index_) {
            driver_.inject(call_, mode_, {occurrence_, 0U});
            armed = true;
        }
        if (event.kind == detail::PolicyPressureEventKind::policy_cycle_start && nonbeneficial) {
            codec_.inject(test::GpuCodecFault::literal_only_output);
        }
        if (event.kind == detail::PolicyPressureEventKind::initial_snapshot) {
            creates_before =
                fixture_.cuda->counts[static_cast<std::size_t>(test::CudaCall::create)];
        }
    }
    bool armed{};
    bool nonbeneficial{};
    std::uint64_t creates_before{};

  private:
    Fixture& fixture_;
    test::FakeCudaDriverApi& driver_;
    test::FakeNvcompLz4Api& codec_;
    detail::PolicyPressureEventKind when_;
    std::uint32_t index_{};
    test::CudaCall call_{};
    test::CudaFaultMode mode_{};
    std::uint64_t occurrence_{};
};
void faults(test::Runner& runner) {
    for (const auto occurrence : {1U, 2U, 3U}) {
        runner.begin("M11 second policy transition preparation/workspace/compaction allocation "
                     "failure cleans all retained ownership");
        Fixture f;
        Inject injection{f,
                         detail::PolicyPressureEventKind::policy_cycle_start,
                         1U,
                         test::CudaCall::create,
                         test::CudaFaultMode::before,
                         occurrence};
        f.run(&injection);
        VRAMZ_CHECK(runner,
                    injection.armed && f.report.result != detail::CompressionSmokeResult::passed);
        VRAMZ_CHECK(runner, f.report.cycle_count == 2U && f.report.automatic_policy_actions == 1U);
        VRAMZ_CHECK(runner, !f.report.policy_settled_snapshot_proven &&
                                f.codec->observation.decompressions == 0U);
        f.empty(runner);
    }
    for (const auto occurrence : {1U, 2U, 3U}) {
        runner.begin(
            "M11 failed first/second automatic compression or restoration stops without retry");
        Fixture f;
        f.api->inject(test::GpuCodecFault::launch_failure, occurrence);
        f.run();
        VRAMZ_CHECK(runner, f.report.result != detail::CompressionSmokeResult::passed &&
                                f.report.has_error);
        VRAMZ_CHECK(runner,
                    f.codec->observation.compressions + f.codec->observation.decompressions ==
                        occurrence - 1U);
        f.empty(runner);
    }
    for (const auto fault : {test::GpuCodecFault::output_corruption,
                             test::GpuCodecFault::crc_preserving_output_corruption}) {
        runner.begin("M11 logical CRC or CRC-neutral byte mismatch during production restoration "
                     "forbids PASS");
        Fixture f;
        f.api->inject(fault, 3U);
        f.run();
        VRAMZ_CHECK(runner, f.report.result != detail::CompressionSmokeResult::passed);
        VRAMZ_CHECK(runner, f.report.policy_settled_snapshot_proven && f.report.has_error);
        f.empty(runner);
    }
    runner.begin(
        "M11 stored compressed CRC detects corrupted compaction copy before policy success");
    {
        Fixture f;
        Inject injection{f,
                         detail::PolicyPressureEventKind::policy_cycle_start,
                         1U,
                         test::CudaCall::device_to_device,
                         test::CudaFaultMode::malformed,
                         1U};
        f.run(&injection);
        VRAMZ_CHECK(runner, f.report.result != detail::CompressionSmokeResult::passed);
        VRAMZ_CHECK(runner,
                    f.report.cycle_count == 2U && f.codec->observation.decompressions == 0U);
        f.empty(runner);
    }
    runner.begin("M11 nonbeneficial policy result backs off that victim and selects a different "
                 "eligible chunk without HOST");
    {
        Fixture f;
        f.api->inject(test::GpuCodecFault::literal_only_output);
        f.run();
        VRAMZ_CHECK(runner, f.report.result != detail::CompressionSmokeResult::passed);
        VRAMZ_CHECK(runner, f.report.cycles[0].error.code == ErrorCode::compression_not_beneficial);
        VRAMZ_CHECK(runner,
                    f.report.cycles[0].compressibility_after == Compressibility::incompressible &&
                        f.report.cycles[0].retry_after_epoch >
                            f.report.chunks[1].access.last_access_epoch);
        VRAMZ_CHECK(runner, f.report.cycles[0].selected_chunk == 1U &&
                                f.report.cycles[1].selected_chunk == 2U);
        VRAMZ_CHECK(runner, f.report.chunks[1].selected_count == 1U &&
                                f.report.automatic_policy_actions == 2U);
        f.empty(runner);
    }
    runner.begin(
        "M11 smaller stored bytes without physical-unit savings cannot pass policy acceptance");
    {
        Fixture f;
        f.codec->config.literal_prefix_bytes = ByteSize{7U * mib};
        f.api->inject(test::GpuCodecFault::literal_prefix_output);
        f.run();
        VRAMZ_CHECK(runner, f.report.result != detail::CompressionSmokeResult::passed);
        VRAMZ_CHECK(runner, f.report.cycles[0].error.code == ErrorCode::compression_not_beneficial);
        VRAMZ_CHECK(runner,
                    f.report.cycles[0].compressibility_after == Compressibility::incompressible);
        VRAMZ_CHECK(runner, f.report.chunks[1].selected_count == 1U &&
                                f.report.chunks[1].settled == RepresentationState::gpu_raw);
        f.empty(runner);
    }
    runner.begin("M11 four nonbeneficial victims exhaust the fixed cycle bound without policy "
                 "convergence or retry");
    {
        Fixture f;
        Inject injection{f,
                         detail::PolicyPressureEventKind::pass_published,
                         0U,
                         test::CudaCall::count,
                         test::CudaFaultMode::before,
                         1U};
        injection.nonbeneficial = true;
        f.run(&injection);
        VRAMZ_CHECK(runner, f.report.result != detail::CompressionSmokeResult::passed);
        VRAMZ_CHECK(runner, f.report.cycle_count == detail::m11_maximum_cycles);
        VRAMZ_CHECK(runner,
                    !f.report.pressure_target_reached && f.report.automatic_policy_actions == 0U);
        VRAMZ_CHECK(runner, f.codec->observation.compressions == 4U &&
                                f.codec->observation.decompressions == 0U);
        for (const auto& chunk : f.report.chunks) {
            VRAMZ_CHECK(runner, chunk.selected_count == 1U);
        }
        f.empty(runner);
    }
    runner.begin("M11 full retained RAW plus bound/workspace/compaction peak rejects above 128 MiB "
                 "before policy mutation");
    {
        Fixture f;
        f.codec->config.compression_bound_floor = ByteSize{30U * mib};
        f.codec->config.compression_temp = ByteSize{38U * mib - sizeof(detail::GpuBatchMetadata)};
        f.run();
        VRAMZ_CHECK(runner, f.report.result == detail::CompressionSmokeResult::resource_cap);
        VRAMZ_CHECK(runner,
                    f.report.initial_snapshot_proven && f.codec->observation.compressions == 0U);
        VRAMZ_CHECK(runner, f.cuda->counts[static_cast<std::size_t>(test::CudaCall::create)] == 4U);
        f.empty(runner);
    }
    runner.begin(
        "M11 impossible preflight workspace cap rejects before retained context or backing");
    {
        Fixture f;
        f.codec->config.compression_temp = detail::m11_physical_cap;
        f.run();
        VRAMZ_CHECK(runner, f.report.result == detail::CompressionSmokeResult::resource_cap);
        VRAMZ_CHECK(runner,
                    f.cuda->counts[static_cast<std::size_t>(test::CudaCall::retain_primary)] == 0U);
        f.empty(runner);
    }
}

void fatal_case(test::Runner& runner, test::GpuCodecFault codec_fault, test::CudaCall call,
                test::CudaFaultMode mode,
                detail::PolicyPressureEvent event = {
                    detail::PolicyPressureEventKind::policy_cycle_start, 0U}) {
    const auto child = ::fork();
    VRAMZ_CHECK(runner, child >= 0);
    if (child == 0) {
        static_cast<void>(::alarm(900U));
        Fixture fixture;
        static const Fixture* observed{};
        static bool definite{};
        observed = &fixture;
        definite = call != test::CudaCall::count && mode == test::CudaFaultMode::before;
        std::set_terminate([]() noexcept {
            const auto charge = detail::policy_pressure_charge(observed->report.final_usage);
            const bool ok = observed->report.result != detail::CompressionSmokeResult::passed &&
                            observed->cuda->destroyed_drivers == 0U &&
                            (!definite || (observed->report.final_counts_known && charge &&
                                           charge.value().value() ==
                                               observed->cuda->owned_bytes +
                                                   observed->report.final_usage.reserved.value()));
            std::_Exit(ok ? 86 : 1);
        });
        if (codec_fault != test::GpuCodecFault::count) {
            fixture.api->inject(codec_fault);
        }
        if (call != test::CudaCall::count) {
            Inject injection{fixture, event.kind, event.index, call, mode, 1U};
            fixture.run(&injection);
        } else {
            fixture.run();
        }
        std::_Exit(1);
    }
    if (child < 0) {
        return;
    }
    int status{};
    pid_t waited{};
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    VRAMZ_CHECK(runner, waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 86);
}
void fatal_faults(test::Runner& runner) {
    for (const auto fault :
         {test::GpuCodecFault::launch_ambiguous, test::GpuCodecFault::completion_failure}) {
        runner.begin("M11 ambiguous automatic transition completion is fail-stop with no "
                     "destructor cleanup");
        fatal_case(runner, fault, test::CudaCall::count, test::CudaFaultMode::before);
    }
    for (const auto mode : {test::CudaFaultMode::before, test::CudaFaultMode::after}) {
        runner.begin("M11 definite or ambiguous compaction cleanup failure retains accounting and "
                     "never retries");
        fatal_case(runner, test::GpuCodecFault::count, test::CudaCall::unmap, mode);
    }
    for (const auto index : {0U, 1U, 2U, 3U}) {
        runner.begin("M11 each settled RAW/COMPRESSED owner retains exact debt on cleanup failure, "
                     "with no retry");
        fatal_case(runner, test::GpuCodecFault::count, test::CudaCall::unmap,
                   test::CudaFaultMode::before,
                   {detail::PolicyPressureEventKind::integrity_start, index});
    }
}

struct PolicyModel final {
    Fixture fixture{};
    std::optional<Runtime> owner{};
    std::array<std::optional<Buffer>, 2U> buffers{};
    PolicyModel() {
        fixture.cuda->config.minimum = ByteSize{64U};
        fixture.cuda->config.recommended = ByteSize{64U};
        auto backend =
            detail::CudaVmmBackend::create(std::move(fixture.driver), 0, std::move(fixture.api));
        if (!backend) {
            std::terminate();
        }
        auto config = detail::policy_pressure_configuration();
        config.preferred_chunk_size = ByteSize{4096U};
        config.budgets.gpu = {ByteSize{32768U}, ByteSize{8192U}, ByteSize{16384U}};
        config.policy.tuning.maximum_transitions = 4U;
        auto created =
            testing::RuntimeAccess::create_with_backend(config, std::move(backend).value());
        if (!created) {
            std::terminate();
        }
        owner.emplace(std::move(created).value());
        for (std::size_t index = 0U; index < buffers.size(); ++index) {
            auto allocated = runtime().allocate(ByteSize{4096U});
            if (!allocated) {
                std::terminate();
            }
            buffers[index].emplace(std::move(allocated).value());
            auto lease =
                owned_buffer(index).acquire({{}, ByteSize{4096U}}, {AccessMode::read_write});
            if (!lease) {
                std::terminate();
            }
            std::array<std::byte, 4096U> data{};
            for (std::size_t i = 0U; i < data.size(); ++i) {
                data[i] = detail::multi_chunk_residency_payload_byte(index, i);
            }
            if (!testing::write_bytes(lease.value(), {}, data) || !lease.value().close()) {
                std::terminate();
            }
        }
    }
    Runtime& runtime() {
        if (!owner.has_value()) {
            std::terminate();
        }
        return owner.value();
    }
    Buffer& owned_buffer(std::size_t index) {
        auto& buffer = buffers.at(index);
        if (!buffer.has_value()) {
            std::terminate();
        }
        return buffer.value();
    }
    void age() {
        for (std::uint32_t i = 0U; i < 16U; ++i) {
            auto lease = owned_buffer(1U).acquire({{}, ByteSize{4096U}});
            if (!lease || !lease.value().close()) {
                std::terminate();
            }
        }
    }
    void clean(test::Runner& runner) {
        for (std::size_t index = 0U; index < buffers.size(); ++index) {
            VRAMZ_CHECK(runner, owned_buffer(index).close());
        }
        VRAMZ_CHECK(runner, runtime().shutdown());
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime(), PhysicalTier::gpu));
        VRAMZ_CHECK(runner,
                    detail::policy_pressure_charge(runtime().stats().gpu).value() == ByteSize{});
        VRAMZ_CHECK(runner,
                    fixture.cuda->owned_bytes == 0U && fixture.codec->observation.streams == 0U);
    }
};
class AccessAfterProposal final : public testing::PolicySelectionObserver {
  public:
    explicit AccessAfterProposal(Buffer& buffer) : buffer_(buffer) {}
    void on_selected(const ChunkSnapshot& snapshot,
                     const policy::Proposal& proposal) noexcept override {
        ++selections;
        old_revision = snapshot.access_revision;
        valid = valid && proposal.temperature == Temperature::cold;
        auto lease = buffer_.acquire({{}, ByteSize{4096U}});
        valid = valid && static_cast<bool>(lease);
        if (lease) {
            valid = valid && static_cast<bool>(lease.value().close());
        }
    }
    void on_completed(const ChunkSnapshot& snapshot, const Result<void>& result) noexcept override {
        ++completions;
        rejected = !result && result.error().code == ErrorCode::conflict;
        valid = valid && snapshot.access_revision == old_revision + 1U &&
                snapshot.authoritative.state == RepresentationState::gpu_raw;
    }
    bool valid{true};
    bool rejected{};
    std::uint32_t selections{};
    std::uint32_t completions{};
    std::uint64_t old_revision{};

  private:
    Buffer& buffer_;
};
void policy_models(test::Runner& runner) {
    runner.begin("M11 no-pressure production cycle performs no action or codec launch");
    {
        PolicyModel model;
        VRAMZ_CHECK(runner, model.runtime().reclaim_to_target(ByteSize{8192U}));
        VRAMZ_CHECK(runner, model.runtime().stats().policy.compression_attempts == 0U);
        VRAMZ_CHECK(runner, model.fixture.codec->observation.compressions == 0U);
        model.clean(runner);
    }
    runner.begin("M11 real access invalidates revision, rejects atomically, excludes same-cycle "
                 "victim and permits later policy reevaluation");
    {
        PolicyModel model;
        model.age();
        const auto before = testing::chunk_snapshot(model.owned_buffer(0U), 0U).value();
        AccessAfterProposal observer{model.owned_buffer(0U)};
        const auto result =
            testing::RuntimeAccess::reclaim(model.runtime(), ByteSize{6144U}, false, &observer);
        VRAMZ_CHECK(runner, !result && observer.valid && observer.rejected);
        VRAMZ_CHECK(runner, observer.selections == 1U && observer.completions == 1U);
        const auto after = testing::chunk_snapshot(model.owned_buffer(0U), 0U).value();
        VRAMZ_CHECK(runner, after.authoritative.resource == before.authoritative.resource &&
                                after.authoritative.charge == before.authoritative.charge);
        const auto metadata =
            testing::RuntimeAccess::policy_metadata(model.owned_buffer(0U), 0U).value();
        VRAMZ_CHECK(runner,
                    metadata.stale_rejection_cycle == model.runtime().stats().policy.policy_cycles);
        VRAMZ_CHECK(runner, model.fixture.codec->observation.compressions == 0U);
        model.age();
        VRAMZ_CHECK(runner, model.runtime().reclaim_to_target(ByteSize{6144U}));
        VRAMZ_CHECK(runner, model.runtime().stats().policy.compression_successes == 1U);
        VRAMZ_CHECK(
            runner,
            testing::chunk_snapshot(model.owned_buffer(1U), 0U).value().authoritative.state ==
                RepresentationState::gpu_raw);
        model.clean(runner);
    }
    runner.begin("M11 no useful action while all candidates are HOT gives a bounded non-PASS");
    {
        PolicyModel model;
        // Both initial leases have just passed through the normal access path.
        auto first = model.owned_buffer(0U).acquire({{}, ByteSize{4096U}});
        VRAMZ_CHECK(runner, first && first.value().close());
        VRAMZ_CHECK(runner, !model.runtime().reclaim_to_target(ByteSize{6144U}));
        VRAMZ_CHECK(runner, model.fixture.codec->observation.compressions == 0U);
        model.clean(runner);
    }
    runner.begin("M11 migration reserve is inside hard limit and cannot fund unaccounted emergency "
                 "headroom");
    {
        const auto config = detail::policy_pressure_configuration();
        BudgetLedger ledger{config.budgets};
        ReservationRequest initial{PhysicalTier::gpu, ChargeBucket::staging, ByteSize{32U * mib},
                                   AdmissionKind::normal};
        ReservationToken token{};
        VRAMZ_CHECK(runner, ledger.reserve(TransactionId{1U}, std::span{&initial, 1U},
                                           std::span{&token, 1U}));
        VRAMZ_CHECK(runner, ledger.materialize(token, initial.amount));
        VRAMZ_CHECK(runner, ledger.release_reservation(token));
        VRAMZ_CHECK(runner, ledger.commit_initial(PhysicalTier::gpu, initial.amount));
        ReservationRequest normal{PhysicalTier::gpu, ChargeBucket::staging, ByteSize{4U * mib},
                                  AdmissionKind::normal};
        VRAMZ_CHECK(runner, !ledger.reserve(TransactionId{2U}, std::span{&normal, 1U},
                                            std::span{&token, 1U}));
        ReservationRequest migration{PhysicalTier::gpu, ChargeBucket::workspace,
                                     ByteSize{96U * mib}, AdmissionKind::migration_temporary};
        VRAMZ_CHECK(runner, ledger.reserve(TransactionId{3U}, std::span{&migration, 1U},
                                           std::span{&token, 1U}));
        ReservationRequest excess{PhysicalTier::gpu, ChargeBucket::workspace, ByteSize{1U},
                                  AdmissionKind::migration_temporary};
        ReservationToken rejected{};
        VRAMZ_CHECK(runner, !ledger.reserve(TransactionId{4U}, std::span{&excess, 1U},
                                            std::span{&rejected, 1U}));
        VRAMZ_CHECK(runner, ledger.charged(PhysicalTier::gpu) == detail::m11_physical_cap);
        VRAMZ_CHECK(runner, ledger.release_reservation(token));
        VRAMZ_CHECK(runner, ledger.retire_committed(PhysicalTier::gpu, initial.amount));
        VRAMZ_CHECK(runner, ledger.release_cleanup_debt(PhysicalTier::gpu, initial.amount));
        VRAMZ_CHECK(runner, ledger.charged(PhysicalTier::gpu) == ByteSize{});
    }
    runner.begin("M11 ledger materialization contradiction preserves adopted ownership and blocks "
                 "successful policy transition");
    {
        PolicyModel model;
        model.age();
        testing::RuntimeAccess::inject_materialize_failure(model.runtime());
        VRAMZ_CHECK(runner, !model.runtime().reclaim_to_target(ByteSize{6144U}));
        VRAMZ_CHECK(runner, model.runtime().stats().policy.compression_successes == 0U);
        VRAMZ_CHECK(runner, testing::accounting_conserved(model.runtime(), PhysicalTier::gpu));
        VRAMZ_CHECK(runner, model.fixture.codec->observation.compressions == 0U);
        model.clean(runner);
    }
}

} // namespace
int main() {
    test::Runner runner;
    baseline(runner);
    policy_models(runner);
    faults(runner);
    fatal_faults(runner);
    return runner.finish();
}
