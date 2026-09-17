#include "../support/fake_cuda_driver.hpp"
#include "../support/fake_nvcomp.hpp"
#include "../test_support.hpp"
#include "vramz/crc32c.hpp"
#include "vramz/detail/physical_savings_smoke.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <sys/wait.h>
#include <unistd.h>

using namespace vramz;
using detail::CompressionSmokeReport;
using detail::CompressionSmokeResult;
using test::CudaCall;
using test::CudaFaultMode;
using test::GpuCodecFault;

namespace {
constexpr ByteSize unit{2ULL * 1024U * 1024U};
struct Fixture final {
    std::shared_ptr<test::FakeCudaState> cuda{std::make_shared<test::FakeCudaState>()};
    std::shared_ptr<test::FakeGpuCodecState> codec{std::make_shared<test::FakeGpuCodecState>()};
    std::unique_ptr<test::FakeCudaDriverApi> driver{
        std::make_unique<test::FakeCudaDriverApi>(cuda)};
    std::unique_ptr<test::FakeNvcompLz4Api> nvcomp{
        std::make_unique<test::FakeNvcompLz4Api>(*driver, codec)};
    CompressionSmokeReport report{};
    Fixture() {
        cuda->config.version = 13020;
        cuda->config.minimum = unit;
        cuda->config.recommended = unit;
        cuda->config.maximum_resource = detail::m9_physical_cap;
        cuda->config.physical_limit = detail::m9_physical_cap;
        cuda->config.device_info.name = std::array<char, 256U>{"NVIDIA GeForce RTX 3060"};
    }
    void run(detail::PhysicalSavingsSmokeOptions options = {true, 0, {595U, 84U, 0U}}) {
        detail::run_physical_savings_smoke(std::move(driver), std::move(nvcomp), options, report);
    }
    [[nodiscard]] std::uint64_t calls(CudaCall call) const {
        return cuda->counts[static_cast<std::size_t>(call)];
    }
    void empty(test::Runner& runner) const {
        VRAMZ_CHECK(runner, report.final_counts_known && report.final_resources == 0U);
        VRAMZ_CHECK(runner, report.final_workspace == 0U && report.final_metadata == 0U);
        VRAMZ_CHECK(runner, report.final_bound_output == 0U && report.final_compaction == 0U);
        VRAMZ_CHECK(runner, report.final_raw == 0U && report.final_compressed == 0U);
        VRAMZ_CHECK(runner, report.final_va == 0U && report.final_budget == ByteSize{});
        VRAMZ_CHECK(runner, cuda->owned_bytes == 0U && cuda->primary_references == 0U);
        VRAMZ_CHECK(runner, codec->observation.streams == 0U && !codec->observation.pending);
        VRAMZ_CHECK(runner, cuda->dropped_calls == 0U && codec->observation.dropped_calls == 0U);
    }
};
void results(test::Runner& runner) {
    for (const auto granularity : {unit, ByteSize{4194304U}, detail::m9_logical_payload}) {
        runner.begin("M9 real CPU LZ4 bytes cross three units, one unit, or no units without fake "
                     "size overrides");
        Fixture f;
        f.cuda->config.minimum = granularity;
        f.cuda->config.recommended = granularity;
        f.run();
        const bool savings = granularity < detail::m9_logical_payload;
        VRAMZ_CHECK(runner,
                    f.report.result == (savings ? CompressionSmokeResult::passed
                                                : CompressionSmokeResult::no_physical_savings));
        VRAMZ_CHECK(runner, !f.report.has_error && f.report.charges_corroborated);
        VRAMZ_CHECK(runner, f.report.raw_charge == detail::m9_logical_payload);
        VRAMZ_CHECK(runner, f.report.compressed_charge == granularity);
        VRAMZ_CHECK(runner, f.report.stored_bytes < granularity);
        VRAMZ_CHECK(runner,
                    detail::saves_physical_unit(f.report.raw_charge, f.report.compressed_charge,
                                                granularity) == savings);
        VRAMZ_CHECK(runner, f.report.source_verified && f.report.size_equal &&
                                f.report.byte_equal && f.report.compaction_exercised);
        VRAMZ_CHECK(runner, f.report.logical_crc_expected == f.report.logical_crc_actual);
        VRAMZ_CHECK(runner, f.report.codec.stored_crc_expected == f.report.codec.stored_crc_actual);
        VRAMZ_CHECK(runner, f.report.cleanup_workspace && f.report.cleanup_bound_output &&
                                f.report.cleanup_compressed && f.report.cleanup_raw &&
                                f.report.cleanup_va && f.report.cleanup_stream &&
                                f.report.cleanup_context);
        const auto& c = f.codec->observation;
        VRAMZ_CHECK(runner, c.compressions == 1U && c.decompressions == 1U && c.completions == 2U);
        VRAMZ_CHECK(runner, c.metadata_publications == 2U && c.metadata_consumptions == 2U &&
                                c.result_reads == 2U);
        constexpr std::array ordered{
            test::GpuCodecCall::metadata_copy_enqueue, test::GpuCodecCall::nvcomp_launch,
            test::GpuCodecCall::stream_synchronize, test::GpuCodecCall::metadata_status_read};
        VRAMZ_CHECK(runner, c.log_size == 8U);
        for (std::size_t i = 0U; i < c.log_size; ++i) {
            VRAMZ_CHECK(runner, c.log[i] == ordered[i % ordered.size()]);
        }
        VRAMZ_CHECK(runner,
                    f.calls(CudaCall::device) == 1U && f.calls(CudaCall::device_to_device) == 1U);
        if (granularity == unit) {
            VRAMZ_CHECK(runner, f.report.admission.peak == ByteSize{40ULL * 1024U * 1024U});
            std::array<char, 8192U> buffer{};
            const auto formatted =
                detail::format_physical_savings_smoke_report(f.report, "", false, buffer);
            VRAMZ_CHECK(runner, !formatted.truncated);
            std::puts(buffer.data());
        }
        f.empty(runner);
    }
    runner.begin("M9 valid literal-only LZ4 output is non-beneficial, fully verified and cleaned, "
                 "never PASS");
    Fixture f;
    f.nvcomp->inject(GpuCodecFault::literal_only_output);
    f.run();
    VRAMZ_CHECK(runner, f.report.result == CompressionSmokeResult::compression_not_beneficial);
    VRAMZ_CHECK(runner, f.report.stored_bytes > detail::m9_logical_payload && f.report.byte_equal);
    VRAMZ_CHECK(runner, f.report.compressed_charge > f.report.raw_charge);
    f.empty(runner);
}
void preflight(test::Runner& runner) {
    for (const detail::PhysicalSavingsSmokeOptions options : {detail::PhysicalSavingsSmokeOptions{},
                                                              {true, 1, {595U, 84U, 0U}},
                                                              {true, 0, {579U, 99U, 0U}}}) {
        runner.begin("M9 acknowledgment device and compatibility reject before any fake CUDA call");
        Fixture f;
        f.run(options);
        VRAMZ_CHECK(runner,
                    f.report.result != CompressionSmokeResult::passed && f.cuda->log_size == 0U);
    }
    for (const bool excess : {false, true}) {
        runner.begin(
            "M9 exact 96 MiB admission accepts; one additional VMM unit rejects before mutation");
        Fixture f;
        f.codec->config.compression_temp =
            ByteSize{30ULL * 1024U * 1024U - sizeof(detail::GpuBatchMetadata)};
        f.codec->config.decompression_temp =
            ByteSize{(excess ? 32ULL : 30ULL) * 1024U * 1024U - sizeof(detail::GpuBatchMetadata)};
        f.run();
        VRAMZ_CHECK(runner, f.report.result == (excess ? CompressionSmokeResult::resource_cap
                                                       : CompressionSmokeResult::passed));
        VRAMZ_CHECK(runner, f.report.admission.proven != excess);
        VRAMZ_CHECK(runner,
                    f.report.admission.peak == ByteSize{(excess ? 98ULL : 96ULL) * 1024U * 1024U});
        if (excess) {
            VRAMZ_CHECK(runner, f.calls(CudaCall::retain_primary) == 0U &&
                                    f.calls(CudaCall::reserve) == 0U &&
                                    f.calls(CudaCall::create) == 0U);
        }
        f.empty(runner);
    }
    for (const bool overflow : {false, true}) {
        runner.begin("M9 full 96 MiB admission rejects excessive workspace or overflow before "
                     "ownership mutation");
        Fixture f;
        if (overflow) {
            f.nvcomp->inject(GpuCodecFault::plan_temp_overflow);
        } else {
            f.codec->config.decompression_temp = detail::m9_physical_cap;
        }
        f.run();
        VRAMZ_CHECK(runner, f.report.result == CompressionSmokeResult::resource_cap);
        VRAMZ_CHECK(runner, !f.report.admission.proven);
        VRAMZ_CHECK(runner, f.calls(CudaCall::retain_primary) == 0U &&
                                f.calls(CudaCall::reserve) == 0U &&
                                f.calls(CudaCall::create) == 0U);
        VRAMZ_CHECK(runner, f.codec->observation.initialization_attempts == 0U);
        f.empty(runner);
    }
}
void faults(test::Runner& runner) {
    for (const auto fault : {GpuCodecFault::premature_status, GpuCodecFault::premature_output}) {
        runner.begin("M9 partial visibility never bypasses synchronization");
        Fixture f;
        f.nvcomp->inject(fault);
        f.run();
        VRAMZ_CHECK(runner, f.report.result == CompressionSmokeResult::passed);
        f.empty(runner);
    }
    for (const auto fault :
         {GpuCodecFault::metadata_submission_failure, GpuCodecFault::launch_failure,
          GpuCodecFault::status_failure, GpuCodecFault::size_zero, GpuCodecFault::size_oversized,
          GpuCodecFault::size_mismatch, GpuCodecFault::output_corruption,
          GpuCodecFault::crc_preserving_output_corruption}) {
        for (const std::uint64_t occurrence : {1U, 2U}) {
            runner.begin("M9 encode/decode faults, logical CRC corruption and CRC-neutral byte "
                         "mismatch stop without PASS");
            Fixture f;
            f.nvcomp->inject(fault, occurrence);
            f.run();
            VRAMZ_CHECK(runner,
                        f.report.result == CompressionSmokeResult::failed && f.report.has_error);
            VRAMZ_CHECK(runner, f.codec->observation.compressions <= 1U &&
                                    f.codec->observation.decompressions <= 1U);
            f.empty(runner);
        }
    }
    for (const std::uint64_t occurrence : {1U, 2U, 3U, 4U, 5U, 6U}) {
        runner.begin("M9 RAW bound-output encode-workspace compaction restored-RAW "
                     "decode-workspace OOM stops once");
        Fixture f;
        f.driver->inject(CudaCall::create, CudaFaultMode::before, {occurrence, 0U});
        f.run();
        VRAMZ_CHECK(runner, f.report.result != CompressionSmokeResult::passed &&
                                f.calls(CudaCall::create) == occurrence);
        f.empty(runner);
    }
    for (const auto mode : {CudaFaultMode::before, CudaFaultMode::malformed}) {
        runner.begin("M9 compaction copy failure or stored CRC corruption forbids decompression");
        Fixture f;
        f.driver->inject(CudaCall::device_to_device, mode);
        f.run();
        VRAMZ_CHECK(runner, f.report.result == CompressionSmokeResult::failed &&
                                f.codec->observation.decompressions == 0U);
        f.empty(runner);
    }
}
void fatal(test::Runner& runner, GpuCodecFault fault, CudaCall call, CudaFaultMode mode,
           std::uint64_t occurrence = 1U) {
    const auto child = ::fork();
    VRAMZ_CHECK(runner, child >= 0);
    if (child == 0) {
        static_cast<void>(::alarm(90U));
        Fixture f;
        if (fault != GpuCodecFault::count) {
            f.nvcomp->inject(fault);
        }
        if (call != CudaCall::count) {
            f.driver->inject(call, mode, {occurrence, 0U});
        }
        static const Fixture* observed{};
        static CudaCall expected_call{};
        static std::uint64_t expected_occurrence{};
        static bool definite{};
        observed = &f;
        expected_call = call;
        expected_occurrence = occurrence;
        definite = call != CudaCall::count && mode == CudaFaultMode::before;
        std::set_terminate([]() noexcept {
            const bool ok =
                observed->report.result != CompressionSmokeResult::passed &&
                observed->cuda->destroyed_drivers == 0U &&
                (expected_call == CudaCall::count ||
                 observed->calls(expected_call) == expected_occurrence) &&
                (!definite ||
                 (observed->report.has_error && observed->report.final_counts_known &&
                  observed->report.final_budget.value() == observed->cuda->owned_bytes));
            std::_Exit(ok ? 86 : 1);
        });
        f.run();
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
void fatal_tests(test::Runner& runner) {
    for (const auto fault :
         {GpuCodecFault::metadata_submission_ambiguous, GpuCodecFault::metadata_publication_partial,
          GpuCodecFault::launch_ambiguous, GpuCodecFault::completion_failure,
          GpuCodecFault::shutdown_failure, GpuCodecFault::shutdown_ambiguous}) {
        runner.begin(
            "M9 ambiguous submission/completion or stream cleanup fails stop without retry");
        fatal(runner, fault, CudaCall::count, CudaFaultMode::before);
    }
    Fixture baseline;
    baseline.run();
    std::uint64_t release = 1U;
    for (const auto& entry : std::span{baseline.cuda->log}.first(baseline.cuda->log_size)) {
        if (entry.call == CudaCall::unmap) {
            break;
        }
        if (entry.call == CudaCall::release) {
            ++release;
        }
    }
    for (const auto call :
         {CudaCall::release, CudaCall::unmap, CudaCall::free_address, CudaCall::release_primary}) {
        for (const auto mode : {CudaFaultMode::before, CudaFaultMode::after}) {
            runner.begin(
                "M9 definite cleanup retains exact accounting; ambiguous cleanup never retries");
            fatal(runner, GpuCodecFault::count, call, mode,
                  call == CudaCall::release ? release : 1U);
        }
    }
}
} // namespace
int main() {
    test::Runner runner;
    results(runner);
    preflight(runner);
    faults(runner);
    fatal_tests(runner);
    return runner.finish();
}
