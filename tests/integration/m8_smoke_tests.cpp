#include "../support/fake_cuda_driver.hpp"
#include "../support/fake_nvcomp.hpp"
#include "../test_support.hpp"
#include "vramz/detail/compression_smoke.hpp"

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
        cuda->config.minimum = detail::m8_default_payload;
        cuda->config.recommended = detail::m8_default_payload;
        cuda->config.maximum_resource = detail::m8_physical_cap;
        cuda->config.physical_limit = detail::m8_physical_cap;
        cuda->config.device_info.name = std::array<char, 256U>{"NVIDIA GeForce RTX 3060"};
    }
    void run(detail::CompressionSmokeOptions options = {
                 true, 0, detail::m8_default_payload, detail::m8_physical_cap, {595U, 84U, 0U}}) {
        options.prior_driver_api = cuda->config.version;
        detail::run_compression_smoke(std::move(driver), std::move(nvcomp), options, report);
    }
    void small_run() {
        cuda->config.minimum = ByteSize{4096U};
        cuda->config.recommended = ByteSize{4096U};
        run({true, 0, ByteSize{32768U}, detail::m8_physical_cap, {595U, 84U, 0U}});
    }
    [[nodiscard]] std::uint64_t calls(CudaCall call) const {
        return cuda->counts[static_cast<std::size_t>(call)];
    }
    void empty(test::Runner& runner) const {
        VRAMZ_CHECK(runner, report.final_counts_known && report.final_resources == 0U);
        VRAMZ_CHECK(runner, report.final_va == 0U && report.final_budget == ByteSize{});
        VRAMZ_CHECK(runner, cuda->owned_bytes == 0U && cuda->primary_references == 0U);
        VRAMZ_CHECK(runner, codec->observation.streams == 0U && !codec->observation.pending);
        VRAMZ_CHECK(runner, cuda->dropped_calls == 0U && codec->observation.dropped_calls == 0U);
    }
};

void success(test::Runner& runner) {
    runner.begin(
        "M8 production fake flow compresses and decompresses exactly once with exact cleanup");
    Fixture f;
    f.run();
    VRAMZ_CHECK(runner, f.report.result == CompressionSmokeResult::passed && !f.report.has_error);
    VRAMZ_CHECK(runner,
                f.report.compatibility == detail::CudaCompatibility::minor_version_candidate);
    VRAMZ_CHECK(runner, f.report.source_verified && f.report.byte_equal && f.report.size_equal);
    VRAMZ_CHECK(runner, f.report.compaction_exercised);
    VRAMZ_CHECK(runner, f.report.stored_bytes < detail::m8_default_payload);
    VRAMZ_CHECK(runner, f.report.compressed_charge == detail::m8_default_payload);
    VRAMZ_CHECK(runner, f.report.admission.peak == ByteSize{16ULL * 1024U * 1024U});
    VRAMZ_CHECK(runner,
                f.report.codec.compression_enqueue && f.report.codec.compression_completion);
    VRAMZ_CHECK(runner,
                f.report.codec.decompression_enqueue && f.report.codec.decompression_completion);
    VRAMZ_CHECK(runner, f.report.codec.stored_crc_expected == f.report.codec.stored_crc_actual);
    VRAMZ_CHECK(runner, f.report.logical_crc_expected == f.report.logical_crc_actual);
    VRAMZ_CHECK(runner, f.report.cleanup_workspace && f.report.cleanup_bound_output &&
                            f.report.cleanup_compressed);
    VRAMZ_CHECK(runner, f.report.cleanup_raw && f.report.cleanup_va && f.report.cleanup_context &&
                            f.report.cleanup_stream);
    const auto& c = f.codec->observation;
    VRAMZ_CHECK(runner, c.compressions == 1U && c.decompressions == 1U && c.completions == 2U);
    VRAMZ_CHECK(runner, c.metadata_publications == 2U && c.metadata_consumptions == 2U);
    constexpr std::array ordered{
        test::GpuCodecCall::metadata_copy_enqueue, test::GpuCodecCall::nvcomp_launch,
        test::GpuCodecCall::stream_synchronize, test::GpuCodecCall::metadata_status_read};
    VRAMZ_CHECK(runner, c.log_size == 8U);
    for (std::size_t i = 0U; i < c.log_size; ++i) {
        VRAMZ_CHECK(runner, c.log[i] == ordered[i % ordered.size()]);
    }
    VRAMZ_CHECK(runner,
                f.calls(CudaCall::device) == 1U && f.calls(CudaCall::device_to_device) == 1U);
    f.empty(runner);
    std::array<char, 8192U> buffer{};
    const auto result = detail::format_compression_smoke_report(
        f.report, "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", false,
        buffer);
    VRAMZ_CHECK(runner, !result.truncated);
    const std::string_view json{buffer.data(), result.written};
    VRAMZ_CHECK(runner,
                json.find("\"hardware_validation\":\"NOT_TESTED\"") != std::string_view::npos);
    VRAMZ_CHECK(runner,
                json.find("\"compatibility_candidate\":\"MINOR_COMPATIBILITY_CANDIDATE\"") !=
                    std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("0x") == std::string_view::npos);
    std::puts(buffer.data());
}

void compatibility(test::Runner& runner) {
    for (unsigned int scenario = 0U; scenario < 3U; ++scenario) {
        runner.begin(
            "M8 same minor and CUDA 13 minor compatibility candidates exercise the fake flow");
        Fixture f;
        f.cuda->config.minimum = ByteSize{4096U};
        f.cuda->config.recommended = ByteSize{4096U};
        detail::CompressionSmokeOptions options{
            true, 0, ByteSize{32768U}, detail::m8_physical_cap, {595U, 84U, 0U}};
        if (scenario == 0U) {
            f.cuda->config.version = 13030;
        }
        if (scenario == 2U) {
            options.installed_driver = {580U, 0U, 0U};
        }
        f.run(options);
        VRAMZ_CHECK(runner, f.report.result == CompressionSmokeResult::passed);
        VRAMZ_CHECK(runner,
                    f.report.compatibility ==
                        (scenario == 0U ? detail::CudaCompatibility::same_family_candidate
                                        : detail::CudaCompatibility::minor_version_candidate));
        VRAMZ_CHECK(runner,
                    f.calls(CudaCall::initialize) == 1U && f.calls(CudaCall::version) == 1U);
        f.empty(runner);
    }
    for (unsigned int scenario = 0U; scenario < 4U; ++scenario) {
        runner.begin("M8 release floor and CUDA family rejection precede all CUDA calls");
        Fixture f;
        detail::CompressionSmokeOptions options{
            true, 0, detail::m8_default_payload, detail::m8_physical_cap, {595U, 84U, 0U}};
        if (scenario == 0U) {
            options.installed_driver = {579U, 999U, 999U};
        }
        if (scenario == 1U) {
            f.cuda->config.version = 12090;
        }
        if (scenario == 2U) {
            options.runtime_api = 12090;
        }
        if (scenario == 3U) {
            options.runtime_api = 14000;
        }
        f.run(options);
        VRAMZ_CHECK(runner, f.report.result == CompressionSmokeResult::unsupported);
        VRAMZ_CHECK(runner, !detail::is_candidate(f.report.compatibility));
        VRAMZ_CHECK(runner, f.cuda->log_size == 0U && !f.cuda->initialized);
        VRAMZ_CHECK(runner, f.codec->observation.plans == 0U &&
                                f.codec->observation.initialization_attempts == 0U);
        f.empty(runner);
    }
    runner.begin(
        "M8 observed CUDA family drift stops after initialization before context or allocation");
    Fixture drift;
    drift.cuda->config.version = 12090;
    const detail::CompressionSmokeOptions options{
        true, 0, detail::m8_default_payload, detail::m8_physical_cap, {595U, 84U, 0U}};
    detail::run_compression_smoke(std::move(drift.driver), std::move(drift.nvcomp), options,
                                  drift.report);
    VRAMZ_CHECK(runner, drift.report.result == CompressionSmokeResult::unsupported);
    VRAMZ_CHECK(runner,
                drift.calls(CudaCall::initialize) == 1U && drift.calls(CudaCall::version) == 1U);
    VRAMZ_CHECK(runner, drift.calls(CudaCall::retain_primary) == 0U &&
                            drift.calls(CudaCall::reserve) == 0U &&
                            drift.calls(CudaCall::create) == 0U);
    drift.empty(runner);
    for (const auto native : {36, 803}) {
        runner.begin(
            "M8 definite Driver initialization incompatibility stops with known zero ownership");
        Fixture f;
        f.cuda->config.initialize_native_error = native;
        f.run();
        VRAMZ_CHECK(runner,
                    f.report.result == CompressionSmokeResult::failed && f.report.has_error);
        VRAMZ_CHECK(runner, f.report.error.native_domain == NativeErrorDomain::cuda_driver &&
                                f.report.error.native_code == native);
        VRAMZ_CHECK(runner, f.calls(CudaCall::initialize) == 1U &&
                                f.calls(CudaCall::version) == 0U && !f.cuda->initialized);
        VRAMZ_CHECK(runner, f.calls(CudaCall::retain_primary) == 0U &&
                                f.calls(CudaCall::reserve) == 0U &&
                                f.calls(CudaCall::create) == 0U);
        VRAMZ_CHECK(runner, f.codec->observation.initialization_attempts == 0U);
        f.empty(runner);
    }
}

void preflight(test::Runner& runner) {
    for (const detail::CompressionSmokeOptions options :
         {detail::CompressionSmokeOptions{},
          {true, 1},
          {true, 0, ByteSize{4194305U}},
          {true, 0, detail::m8_default_payload, ByteSize{67108865U}}}) {
        runner.begin("M8 local bounds and acknowledgment precede all driver calls");
        Fixture f;
        f.run(options);
        VRAMZ_CHECK(runner, f.report.result != CompressionSmokeResult::passed);
        VRAMZ_CHECK(runner, f.cuda->log_size == 0U && f.codec->observation.plans == 0U);
    }
    for (unsigned int scenario = 0U; scenario < 5U; ++scenario) {
        runner.begin("M8 unsupported capabilities or full peak cap reject before context stream VA "
                     "or physical allocation");
        Fixture f;
        if (scenario == 0U) {
            f.cuda->config.version = 12090;
        }
        if (scenario == 1U) {
            f.cuda->config.unified_addressing = false;
        }
        if (scenario == 2U) {
            f.cuda->config.vmm = false;
        }
        if (scenario == 3U) {
            f.cuda->config.device_info.name[0U] = 'X';
        }
        if (scenario == 4U) {
            f.codec->config.compression_temp = detail::m8_physical_cap;
        }
        f.run();
        VRAMZ_CHECK(runner,
                    f.report.result == (scenario == 4U ? CompressionSmokeResult::resource_cap
                                                       : CompressionSmokeResult::unsupported));
        VRAMZ_CHECK(runner, f.calls(CudaCall::retain_primary) == 0U &&
                                f.calls(CudaCall::reserve) == 0U &&
                                f.calls(CudaCall::create) == 0U);
        f.empty(runner);
    }
}

void faults(test::Runner& runner) {
    for (const auto fault : {GpuCodecFault::premature_status, GpuCodecFault::premature_output}) {
        runner.begin("M8 partial visibility is consumed only after explicit synchronization");
        Fixture f;
        f.nvcomp->inject(fault);
        f.small_run();
        VRAMZ_CHECK(runner, f.report.result == CompressionSmokeResult::passed);
        VRAMZ_CHECK(runner, f.codec->observation.completions == 2U &&
                                f.codec->observation.result_reads == 2U);
        f.empty(runner);
    }
    for (const auto fault :
         {GpuCodecFault::metadata_submission_failure, GpuCodecFault::launch_failure,
          GpuCodecFault::status_read_failure, GpuCodecFault::size_read_failure,
          GpuCodecFault::status_failure, GpuCodecFault::size_zero, GpuCodecFault::size_oversized,
          GpuCodecFault::size_mismatch, GpuCodecFault::output_corruption}) {
        for (const std::uint64_t occurrence : {1U, 2U}) {
            runner.begin("M8 codec failures and corruption never publish PASS and perform complete "
                         "definite cleanup");
            Fixture f;
            f.nvcomp->inject(fault, occurrence);
            f.small_run();
            VRAMZ_CHECK(runner,
                        f.report.result != CompressionSmokeResult::passed && f.report.has_error);
            f.empty(runner);
        }
    }
    for (const std::uint64_t occurrence : {1U, 3U, 4U, 6U}) {
        runner.begin("M8 OOM workspace and compaction allocation failures stop without retry");
        Fixture f;
        f.driver->inject(CudaCall::create, CudaFaultMode::before, {occurrence, 0U});
        f.small_run();
        VRAMZ_CHECK(runner, f.report.result != CompressionSmokeResult::passed);
        VRAMZ_CHECK(runner, f.calls(CudaCall::create) == occurrence);
        f.empty(runner);
    }
    runner.begin("M8 stored CRC rejects corrupted compaction backing before decompression");
    Fixture f;
    f.driver->inject(CudaCall::device_to_device, CudaFaultMode::malformed);
    f.small_run();
    VRAMZ_CHECK(runner, f.report.result != CompressionSmokeResult::passed);
    VRAMZ_CHECK(runner, f.codec->observation.decompressions == 0U);
    f.empty(runner);
}

void fatal(test::Runner& runner, GpuCodecFault fault, CudaCall call, CudaFaultMode mode,
           std::uint64_t occurrence = 1U) {
    const auto child = ::fork();
    VRAMZ_CHECK(runner, child >= 0);
    if (child == 0) {
        static_cast<void>(::alarm(30U));
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
        static bool definite_cleanup{};
        observed = &f;
        expected_call = call;
        expected_occurrence = occurrence;
        definite_cleanup = call != CudaCall::count && mode == CudaFaultMode::before;
        std::set_terminate([]() noexcept {
            const bool ok =
                observed->report.result != CompressionSmokeResult::passed &&
                observed->cuda->destroyed_drivers == 0U &&
                (expected_call == CudaCall::count ||
                 observed->calls(expected_call) == expected_occurrence) &&
                (!definite_cleanup ||
                 (observed->report.has_error && observed->report.final_counts_known &&
                  observed->report.final_budget.value() == observed->cuda->owned_bytes));
            std::_Exit(ok ? 86 : 1);
        });
        f.small_run();
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
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 86) {
        static_cast<void>(
            std::fprintf(stderr, "M8 fatal case codec=%u driver=%u mode=%u status=%d\n",
                         static_cast<unsigned int>(fault), static_cast<unsigned int>(call),
                         static_cast<unsigned int>(mode), status));
    }
    VRAMZ_CHECK(runner, waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 86);
}
void fatal_tests(test::Runner& runner) {
    for (const auto fault :
         {GpuCodecFault::metadata_submission_ambiguous, GpuCodecFault::metadata_publication_partial,
          GpuCodecFault::launch_ambiguous, GpuCodecFault::completion_failure,
          GpuCodecFault::shutdown_failure, GpuCodecFault::shutdown_ambiguous}) {
        runner.begin("M8 ambiguous codec work or failed stream destruction fails stop without "
                     "cleanup retry");
        fatal(runner, fault, CudaCall::count, CudaFaultMode::before);
    }
    Fixture baseline;
    baseline.small_run();
    std::uint64_t first_physical_release = 1U;
    for (const auto& entry : std::span{baseline.cuda->log}.first(baseline.cuda->log_size)) {
        if (entry.call == CudaCall::unmap) {
            break;
        }
        if (entry.call == CudaCall::release) {
            ++first_physical_release;
        }
    }
    for (const auto mode : {CudaFaultMode::before, CudaFaultMode::after}) {
        runner.begin("M8 physical release failure retains exact debt or fails stop on ambiguity");
        fatal(runner, GpuCodecFault::count, CudaCall::release, mode, first_physical_release);
    }
    for (const auto call : {CudaCall::unmap, CudaCall::free_address, CudaCall::release_primary}) {
        for (const auto mode : {CudaFaultMode::before, CudaFaultMode::after}) {
            runner.begin(
                "M8 definite or ambiguous cleanup preserves ownership and stops after one attempt");
            fatal(runner, GpuCodecFault::count, call, mode);
        }
    }
}

void runtime_failure(test::Runner& runner, test::GpuCodecRuntimeFault point, std::int64_t native) {
    runner.begin("M8 actual runtime incompatibility preserves native failure and stops without "
                 "cleanup retry");
    const auto child = ::fork();
    VRAMZ_CHECK(runner, child >= 0);
    if (child == 0) {
        static_cast<void>(::alarm(30U));
        Fixture f;
        f.codec->config.runtime_fault = point;
        f.codec->config.runtime_native_error = native;
        static Fixture* observed{};
        static std::int64_t expected_native{};
        observed = &f;
        expected_native = native;
        std::set_terminate([]() noexcept {
            detail::finalize_compression_smoke_failure(observed->report);
            const auto& report = observed->report;
            std::array<char, 8192U> buffer{};
            const auto formatted = detail::format_compression_smoke_report(
                report, "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", false,
                buffer);
            const std::string_view json{buffer.data(), formatted.written};
            const std::string_view expected =
                expected_native == 36
                    ? "\"runtime_compatibility_failure\":\"CALL_REQUIRES_NEWER_DRIVER\""
                    : "\"runtime_compatibility_failure\":\"SYSTEM_DRIVER_MISMATCH\"";
            const bool ok =
                report.result == CompressionSmokeResult::failed && report.has_error &&
                report.error.code == ErrorCode::ambiguous_backend_state &&
                report.error.native_domain == NativeErrorDomain::cuda_runtime &&
                report.error.native_code == expected_native && !report.final_counts_known &&
                observed->cuda->destroyed_drivers == 0U &&
                observed->codec->observation.initialization_attempts == 1U &&
                observed->codec->observation.shutdown_attempts == 0U &&
                observed->calls(CudaCall::unmap) == 0U &&
                observed->calls(CudaCall::free_address) == 0U &&
                observed->calls(CudaCall::release_primary) == 0U && !formatted.truncated &&
                json.find(expected) != std::string_view::npos &&
                json.find("\"final_budget_charge\":null") != std::string_view::npos &&
                json.find("\"hardware_validation\":\"NOT_TESTED\"") != std::string_view::npos;
            std::_Exit(ok ? 86 : 1);
        });
        f.small_run();
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
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 86) {
        static_cast<void>(std::fprintf(stderr, "M8 runtime fatal point=%u native=%lld status=%d\n",
                                       static_cast<unsigned int>(point),
                                       static_cast<long long>(native), status));
    }
    VRAMZ_CHECK(runner, waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 86);
}
} // namespace
int main() {
    test::Runner runner;
    success(runner);
    compatibility(runner);
    preflight(runner);
    faults(runner);
    fatal_tests(runner);
    for (const auto point :
         {test::GpuCodecRuntimeFault::initialize, test::GpuCodecRuntimeFault::launch,
          test::GpuCodecRuntimeFault::synchronize}) {
        for (const auto native : {36, 803}) {
            runtime_failure(runner, point, native);
        }
    }
    return runner.finish();
}
