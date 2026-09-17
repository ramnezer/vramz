#include "../../tools/common/workload.hpp"
#include "../support/fake_cuda_driver.hpp"
#include "../support/fake_nvcomp.hpp"
#include "../test_support.hpp"
#include "vramz/detail/cuda_vmm_backend.hpp"
#include "vramz/timing.hpp"

#include <iostream>

using namespace vramz;
namespace {
void workload(test::Runner& runner, example::Profile profile, bool concurrent = false,
              bool multi_chunk = false) {
    runner.begin("public Runtime/Lease path with physical-charge Fake and actual codec decisions");
    auto cuda = std::make_shared<test::FakeCudaState>();
    cuda->config.minimum = ByteSize{2097152U};
    cuda->config.recommended = cuda->config.minimum;
    cuda->config.physical_limit = ByteSize{134217728U};
    cuda->config.maximum_resource = ByteSize{134217728U};
    auto codec = std::make_shared<test::FakeGpuCodecState>();
    auto driver = std::make_unique<test::FakeCudaDriverApi>(cuda);
    auto api = std::make_unique<test::FakeNvcompLz4Api>(*driver, codec);
    auto backend = detail::CudaVmmBackend::create(std::move(driver), 0, std::move(api));
    VRAMZ_CHECK(runner, backend);
    if (!backend) {
        return;
    }
    example::WorkloadConfig config{};
    config.profile = profile;
    config.concurrent_reads = concurrent;
    config.measure = concurrent;
    if (multi_chunk) {
        config.chunk_size = ByteSize{4194304U};
    }
    const auto settings = example::runtime_config(config);
    auto runtime =
        testing::RuntimeAccess::create_with_backend(settings.value(), std::move(backend).value());
    VRAMZ_CHECK(runner, runtime);
    if (!runtime) {
        return;
    }
    example::WorkloadReport report{};
    example::run_workload(runtime.value(), config, report);
    example::print_report(std::cout, report, false);
    VRAMZ_CHECK(runner,
                report.passed && report.cleanup && report.integrity && report.snapshot_proven);
    VRAMZ_CHECK(runner, report.initial_charge == ByteSize{67108864U});
    VRAMZ_CHECK(runner, profile == example::Profile::p25 || report.hot_preserved);
    VRAMZ_CHECK(runner, !concurrent || report.concurrent_reads_verified);
    if (multi_chunk) {
        VRAMZ_CHECK(runner, report.chunk_count == 16U);
        VRAMZ_CHECK(runner, report.settled_charge == ByteSize{37748736U});
    }
    if (profile == example::Profile::p25) {
        VRAMZ_CHECK(runner, !report.target_reached && report.settled_charge == ByteSize{54525952U});
        // Approved M3 may use HOT only at HARD pressure after all preferred candidates.
        VRAMZ_CHECK(runner, !report.hot_preserved);
    }
    VRAMZ_CHECK(runner, cuda->owned_bytes == 0U && cuda->primary_references == 0U);
    VRAMZ_CHECK(runner, codec->observation.streams == 0U && !codec->observation.pending);
    if (profile == example::Profile::high && !multi_chunk) {
        VRAMZ_CHECK(runner, report.target_reached && report.settled_charge == ByteSize{29360128U});
        VRAMZ_CHECK(runner, report.cycle_count == 6U && report.warm_preserved);
    } else if (profile == example::Profile::mixed) {
        VRAMZ_CHECK(runner, report.cycles[0].after.compression_rejected == 1U);
        VRAMZ_CHECK(runner, report.chunks[1].settled.residency == Residency::gpu_raw);
        VRAMZ_CHECK(runner, report.chunks[1].settled.compression_attempts == 1U);
        VRAMZ_CHECK(runner,
                    report.chunks[1].settled.retry_after > report.chunks[1].settled.access_epoch);
        VRAMZ_CHECK(runner, report.settled_charge < report.initial_charge);
    } else if (profile == example::Profile::random) {
        VRAMZ_CHECK(runner,
                    !report.target_reached && report.settled_charge == report.initial_charge);
        for (std::uint32_t i = 1U; i < report.chunk_count; ++i) {
            VRAMZ_CHECK(runner, report.chunks[i].settled.compression_attempts <= 1U);
        }
    }
}
void repeated(test::Runner& runner, bool raw) {
    runner.begin("measured steady-state cycles retain only the context and stream between rounds");
    auto cuda = std::make_shared<test::FakeCudaState>();
    cuda->config.minimum = ByteSize{2097152U};
    cuda->config.recommended = cuda->config.minimum;
    cuda->config.physical_limit = ByteSize{134217728U};
    cuda->config.maximum_resource = ByteSize{134217728U};
    auto codec = std::make_shared<test::FakeGpuCodecState>();
    auto driver = std::make_unique<test::FakeCudaDriverApi>(cuda);
    auto api = std::make_unique<test::FakeNvcompLz4Api>(*driver, codec);
    auto backend = detail::CudaVmmBackend::create(std::move(driver), 0, std::move(api));
    VRAMZ_CHECK(runner, backend);
    if (!backend) {
        return;
    }
    example::WorkloadConfig config{};
    config.profile = example::Profile::high;
    config.measure = true;
    config.keep_runtime = true;
    config.raw_baseline = raw;
    const auto settings = example::runtime_config(config);
    auto runtime =
        testing::RuntimeAccess::create_with_backend(settings.value(), std::move(backend).value());
    VRAMZ_CHECK(runner, runtime);
    if (!runtime) {
        return;
    }
    for (std::uint32_t i = 0U; i < 3U; ++i) {
        example::WorkloadReport report{};
        example::run_workload(runtime.value(), config, report);
        VRAMZ_CHECK(runner, report.passed && report.cleanup && report.integrity);
        VRAMZ_CHECK(runner,
                    report.final_resources.retained_context && report.final_resources.stream_owned);
        VRAMZ_CHECK(runner, report.elapsed.samples == 1U && !report.elapsed.overflow);
        const auto samples = report.performance_after.compression.samples -
                             report.performance_before.compression.samples;
        VRAMZ_CHECK(runner, samples == (raw ? 0U : 6U));
        const auto restores = report.performance_after.decompression.samples -
                              report.performance_before.decompression.samples;
        VRAMZ_CHECK(runner, restores == samples);
        VRAMZ_CHECK(runner, report.performance_after.policy_transition.samples -
                                    report.performance_before.policy_transition.samples ==
                                samples);
        VRAMZ_CHECK(runner, cuda->owned_bytes == 0U && codec->observation.streams == 1U);
    }
    VRAMZ_CHECK(runner, runtime.value().shutdown());
    VRAMZ_CHECK(runner, cuda->primary_references == 0U && codec->observation.streams == 0U);
    runner.begin("timing overflow is explicit and disabled measurement leaves counters unchanged");
    TimingCounter overflow{UINT64_MAX, 0U, false};
    PerformanceSample measured{true, overflow};
    measured.finish();
    VRAMZ_CHECK(runner, overflow.overflow);
    TimingCounter disabled{};
    PerformanceSample off{false, disabled};
    off.finish();
    VRAMZ_CHECK(runner, disabled.samples == 0U && disabled.nanoseconds == 0U && !disabled.overflow);
}
void boundaries(test::Runner& runner) {
    runner.begin("public workload bounds reject unsafe sizes before backend mutation");
    example::WorkloadConfig config{};
    config.buffers = 33U;
    VRAMZ_CHECK(runner, !example::runtime_config(config));
    config.buffers = 8U;
    config.hard_limit = ByteSize{536870913U};
    VRAMZ_CHECK(runner, !example::runtime_config(config));
    config.hard_limit = ByteSize{134217728U};
    config.bytes_per_buffer = ByteSize{UINT64_MAX};
    VRAMZ_CHECK(runner, !example::runtime_config(config));
    auto runtime = Runtime::create(test::test_runtime_config());
    auto buffer = runtime.value().allocate(ByteSize{64U});
    auto lease = buffer.value().acquire({{}, ByteSize{64U}}, {AccessMode::read_write});
    std::array<std::byte, 64U> data{};
    data[3] = std::byte{29U};
    VRAMZ_CHECK(runner, lease.value().write({}, data));
    VRAMZ_CHECK(runner, !lease.value().write(ByteOffset{1U}, data));
    std::array<std::byte, 64U> observed{};
    VRAMZ_CHECK(runner, lease.value().read({}, observed) && data == observed);
    VRAMZ_CHECK(runner, lease.value().close());
    VRAMZ_CHECK(runner, !lease.value().read({}, observed));
    const auto raw_snapshot = buffer.value().inspect_chunk(0U);
    VRAMZ_CHECK(runner, raw_snapshot && !raw_snapshot.value().policy_metadata_available &&
                            raw_snapshot.value().logical_bytes == ByteSize{64U});
    VRAMZ_CHECK(runner, !buffer.value().inspect_chunk(1U));
    VRAMZ_CHECK(runner, buffer.value().close());
    VRAMZ_CHECK(runner, runtime.value().shutdown());
}
} // namespace
int main(int argc, char* argv[]) {
    test::Runner runner;
    if (argc == 2 && std::string_view{argv[1]} == "--measurement-only") {
        boundaries(runner);
        repeated(runner, false);
        repeated(runner, true);
        return runner.finish();
    }
    if (argc != 1) {
        return 2;
    }
    boundaries(runner);
    for (const auto profile :
         {example::Profile::high, example::Profile::mixed, example::Profile::random}) {
        workload(runner, profile);
    }
    workload(runner, example::Profile::high, true);
    workload(runner, example::Profile::high, false, true);
    workload(runner, example::Profile::p25);
    repeated(runner, false);
    repeated(runner, true);
    return runner.finish();
}
