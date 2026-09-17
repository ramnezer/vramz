#include "../support/fake_cuda_driver.hpp"
#include "../support/fake_nvcomp.hpp"
#include "../test_support.hpp"
#include "vramz/crc32c.hpp"
#include "vramz/detail/multi_chunk_residency_smoke.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace vramz;
using detail::CompressionSmokeResult;
using detail::MultiChunkResidencyChunkReport;
using detail::MultiChunkResidencyEventKind;
using detail::MultiChunkResidencySmokeReport;
using test::CudaCall;
using test::CudaFaultMode;
using test::GpuCodecFault;

namespace {
constexpr ByteSize unit{2ULL * 1024U * 1024U};
constexpr std::uint64_t mib{1024ULL * 1024U};
struct Fixture final {
    std::shared_ptr<test::FakeCudaState> cuda{std::make_shared<test::FakeCudaState>()};
    std::shared_ptr<test::FakeGpuCodecState> codec{std::make_shared<test::FakeGpuCodecState>()};
    std::unique_ptr<test::FakeCudaDriverApi> driver{
        std::make_unique<test::FakeCudaDriverApi>(cuda)};
    std::unique_ptr<test::FakeNvcompLz4Api> nvcomp{
        std::make_unique<test::FakeNvcompLz4Api>(*driver, codec)};
    MultiChunkResidencySmokeReport report{};
    Fixture() {
        cuda->config.version = 13020;
        cuda->config.minimum = unit;
        cuda->config.recommended = unit;
        cuda->config.maximum_resource = detail::m10_physical_cap;
        cuda->config.physical_limit = detail::m10_physical_cap;
        cuda->config.device_info.name = std::array<char, 256U>{"NVIDIA GeForce RTX 3060"};
        codec->cuda_audit = cuda;
    }
    void run(detail::MultiChunkResidencySmokeOptions options = {true, 0, {595U, 84U, 0U}}) {
        detail::run_multi_chunk_residency_smoke(std::move(driver), std::move(nvcomp), options,
                                                report);
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
        VRAMZ_CHECK(runner, report.final_unmaterialized_reservations == ByteSize{});
        VRAMZ_CHECK(runner, cuda->owned_bytes == 0U && cuda->primary_references == 0U);
        VRAMZ_CHECK(runner, codec->observation.streams == 0U && !codec->observation.pending);
        VRAMZ_CHECK(runner, cuda->dropped_calls == 0U && codec->observation.dropped_calls == 0U);
        VRAMZ_CHECK(runner, std::ranges::all_of(cuda->physical, [](const auto& allocation) {
                        return allocation.handle == detail::CudaPhysicalHandle{};
                    }));
        VRAMZ_CHECK(runner, std::ranges::all_of(cuda->reservations, [](const auto& reservation) {
                        return reservation.address == DeviceAddress{};
                    }));
    }
};
[[nodiscard]] const test::FakeGpuCodecState::LiveAllocation*
find_live(const test::FakeGpuCodecState::LaunchAudit& audit, std::uint64_t address) {
    const auto allocations = std::span{audit.allocations}.first(audit.allocation_count);
    const auto found = std::ranges::find_if(allocations, [address](const auto& allocation) {
        return allocation.address.value() == address;
    });
    return found == allocations.end() ? nullptr : &*found;
}
void ordering(test::Runner& runner, const Fixture& f) {
    runner.begin(
        "M10 real fake ownership proves retained chunks, four-way residency and sequential "
        "restoration before final PASS");
    VRAMZ_CHECK(runner, f.codec->launch_count == 8U);
    if (f.codec->launch_count != 8U) {
        return;
    }
    const auto& launches = f.codec->launches;
    std::array<std::uint64_t, 4U> compressed{};
    for (std::size_t i = 0U; i < compressed.size(); ++i) {
        compressed[i] = launches[4U + i].metadata.input;
        VRAMZ_CHECK(runner, launches[i].direction == detail::GpuCodecDirection::compress);
        VRAMZ_CHECK(runner, launches[4U + i].direction == detail::GpuCodecDirection::decompress);
        const auto* raw = find_live(launches[i], launches[i].metadata.input);
        VRAMZ_CHECK(runner, raw != nullptr && raw->charge == f.report.chunks[i].raw_charge);
        const auto* exact = find_live(launches[4U], compressed[i]);
        VRAMZ_CHECK(runner,
                    exact != nullptr && exact->charge == f.report.chunks[i].compressed_charge);
        for (std::size_t prior = 0U; prior < i; ++prior) {
            VRAMZ_CHECK(runner, compressed[prior] != compressed[i]);
        }
    }
    for (std::size_t operation = 0U; operation < 4U; ++operation) {
        for (std::size_t retained = 0U; retained < operation; ++retained) {
            const auto* live = find_live(launches[operation], compressed[retained]);
            const auto* snapshot = find_live(launches[4U], compressed[retained]);
            VRAMZ_CHECK(runner, live != nullptr && snapshot != nullptr &&
                                    live->handle == snapshot->handle &&
                                    live->charge == snapshot->charge);
        }
        for (std::size_t restored = 0U; restored < operation; ++restored) {
            VRAMZ_CHECK(runner,
                        find_live(launches[4U + operation], compressed[restored]) == nullptr);
            VRAMZ_CHECK(runner, find_live(launches[4U + operation],
                                          launches[4U + restored].metadata.output) == nullptr);
        }
    }
    std::size_t last_compaction{};
    for (std::size_t i = 0U; i < launches[4U].cuda_log_position; ++i) {
        const auto& entry = f.cuda->log[i];
        if (entry.call == CudaCall::device_to_device) {
            last_compaction = i;
        }
        if (entry.call == CudaCall::unmap) {
            VRAMZ_CHECK(runner, std::ranges::find(compressed, entry.identity) == compressed.end());
        }
    }
    for (const auto address : compressed) {
        bool corroborated{};
        for (std::size_t i = last_compaction + 1U; i < launches[4U].cuda_log_position; ++i) {
            const auto& entry = f.cuda->log[i];
            corroborated = corroborated ||
                           (entry.call == CudaCall::retain_mapping && entry.identity == address);
        }
        VRAMZ_CHECK(runner, corroborated);
    }
    VRAMZ_CHECK(runner, f.report.event_count == 19U);
    if (f.report.event_count == 19U) {
        for (std::size_t i = 0U; i < 4U; ++i) {
            const auto& start = f.report.events[2U * i];
            const auto& retained = f.report.events[2U * i + 1U];
            VRAMZ_CHECK(runner, start.kind == MultiChunkResidencyEventKind::compression_start &&
                                    start.chunk_index == i &&
                                    start.compressed_mask == ((1U << i) - 1U));
            VRAMZ_CHECK(runner,
                        retained.kind == MultiChunkResidencyEventKind::compressed_retained &&
                            retained.chunk_index == i &&
                            retained.compressed_mask == ((1U << (i + 1U)) - 1U));
            const auto& restore = f.report.events[9U + 2U * i];
            const auto& verified = f.report.events[10U + 2U * i];
            VRAMZ_CHECK(runner, restore.kind == MultiChunkResidencyEventKind::restoration_start &&
                                    restore.chunk_index == i &&
                                    restore.compressed_mask == (15U & ~((1U << i) - 1U)));
            VRAMZ_CHECK(runner,
                        verified.kind == MultiChunkResidencyEventKind::restoration_verified &&
                            verified.chunk_index == i);
        }
        VRAMZ_CHECK(runner,
                    f.report.events[8U].kind == MultiChunkResidencyEventKind::aggregate_snapshot &&
                        f.report.events[8U].compressed_mask == 15U);
        VRAMZ_CHECK(runner,
                    f.report.events[17U].kind == MultiChunkResidencyEventKind::cleanup_complete &&
                        f.report.events[17U].compressed_mask == 0U);
        VRAMZ_CHECK(runner,
                    f.report.events[18U].kind == MultiChunkResidencyEventKind::pass_published &&
                        f.report.events[18U].compressed_mask == 0U);
    }
    constexpr std::array order{
        test::GpuCodecCall::metadata_copy_enqueue, test::GpuCodecCall::nvcomp_launch,
        test::GpuCodecCall::stream_synchronize, test::GpuCodecCall::metadata_status_read};
    const auto& codec = f.codec->observation;
    VRAMZ_CHECK(runner, codec.log_size == 32U);
    for (std::size_t i = 0U; i < codec.log_size; ++i) {
        VRAMZ_CHECK(runner, codec.log[i] == order[i % order.size()]);
    }
    VRAMZ_CHECK(runner, codec.metadata_publications == 8U && codec.metadata_consumptions == 8U &&
                            codec.result_reads == 8U && codec.completions == 8U);
}
void result_tests(test::Runner& runner, const Fixture& baseline) {
    runner.begin(
        "M10 four distinct 8 MiB chunks retain measured 32 MiB RAW baseline and live 8 MiB "
        "compressed backing with complete round trips");
    VRAMZ_CHECK(runner, baseline.report.result == CompressionSmokeResult::passed &&
                            !baseline.report.has_error);
    VRAMZ_CHECK(runner, baseline.report.simultaneous_compressed_residency_proven);
    VRAMZ_CHECK(runner, baseline.report.totals.logical_bytes == ByteSize{32U * mib});
    VRAMZ_CHECK(runner, baseline.report.totals.raw_baseline_charge == ByteSize{32U * mib});
    VRAMZ_CHECK(runner, baseline.report.totals.compressed_physical_charge == ByteSize{8U * mib});
    VRAMZ_CHECK(runner, baseline.report.totals.physical_bytes_saved == ByteSize{24U * mib});
    VRAMZ_CHECK(runner, baseline.report.peak_admitted_gpu_bytes <= detail::m10_physical_cap);
    std::array<std::uint32_t, 4U> crcs{};
    std::vector<std::byte> payload(static_cast<std::size_t>(detail::m10_logical_payload.value()));
    for (std::size_t chunk = 0U; chunk < 4U; ++chunk) {
        bool zero{};
        bool nonzero{};
        bool variation{};
        for (std::size_t i = 0U; i < payload.size(); ++i) {
            payload[i] = detail::multi_chunk_residency_payload_byte(chunk, i);
            zero = zero || payload[i] == std::byte{};
            nonzero = nonzero || payload[i] != std::byte{};
            variation = variation || payload[i] != payload.front();
        }
        VRAMZ_CHECK(runner, zero && nonzero && variation);
        crcs[chunk] = crc32c(payload);
        for (std::size_t prior = 0U; prior < chunk; ++prior) {
            VRAMZ_CHECK(runner, crcs[chunk] != crcs[prior]);
        }
        const auto& item = baseline.report.chunks[chunk];
        VRAMZ_CHECK(runner,
                    item.chunk_index == chunk && item.logical_bytes == detail::m10_logical_payload);
        VRAMZ_CHECK(runner, item.raw_charge == detail::m10_logical_payload &&
                                item.compressed_charge == unit &&
                                item.physical_bytes_saved == ByteSize{6U * mib});
        VRAMZ_CHECK(runner, item.logical_crc_expected == crcs[chunk] &&
                                item.logical_crc_actual == crcs[chunk]);
        VRAMZ_CHECK(runner, item.stored_crc_expected == item.stored_crc_actual);
        VRAMZ_CHECK(runner, item.charges_corroborated && item.minimum_unit_saved &&
                                item.source_verified && item.compaction_exercised &&
                                item.compressed_at_snapshot && item.compression_enqueue &&
                                item.compression_completion && item.decompression_enqueue &&
                                item.decompression_completion && item.size_equal &&
                                item.byte_equal && item.round_trip_verified &&
                                item.cleanup_complete);
    }
    baseline.empty(runner);
    ordering(runner, baseline);
    std::array<char, 16384U> json{};
    const auto formatted =
        detail::format_multi_chunk_residency_smoke_report(baseline.report, "", false, json);
    VRAMZ_CHECK(runner, !formatted.truncated);
    std::puts(json.data());

    runner.begin("M10 valid literal-prefix codec output leaves one selected chunk saving exactly "
                 "one VMM unit; all four independently restore");
    Fixture exact;
    exact.codec->config.literal_prefix_bytes = ByteSize{5U * mib};
    exact.nvcomp->inject(GpuCodecFault::literal_prefix_output, 3U);
    exact.run();
    VRAMZ_CHECK(runner, exact.report.result == CompressionSmokeResult::passed);
    VRAMZ_CHECK(runner, exact.report.chunks[2U].compressed_charge == ByteSize{6U * mib} &&
                            exact.report.chunks[2U].physical_bytes_saved == unit);
    VRAMZ_CHECK(runner, exact.report.totals.compressed_physical_charge == ByteSize{12U * mib});
    exact.empty(runner);
}
void arithmetic_tests(test::Runner& runner) {
    runner.begin("M10 aggregate arithmetic sums measured padded RAW charges rather than logical "
                 "bytes");
    std::array<MultiChunkResidencyChunkReport, 4U> chunks{};
    for (std::size_t i = 0U; i < chunks.size(); ++i) {
        auto& chunk = chunks[i];
        chunk.chunk_index = static_cast<std::uint32_t>(i);
        chunk.raw_charge = ByteSize{(10U + 2U * i) * mib};
        chunk.compressed_charge = ByteSize{(2U + 2U * i) * mib};
        chunk.stored_bytes = ByteSize{1024U};
        chunk.charges_corroborated = true;
        chunk.minimum_unit_saved = true;
        chunk.compressed_at_snapshot = true;
    }
    const auto measured = detail::calculate_multi_chunk_residency_totals(chunks, unit);
    VRAMZ_CHECK(runner, measured && measured.value().raw_baseline_charge == ByteSize{52U * mib} &&
                            measured.value().compressed_physical_charge == ByteSize{20U * mib} &&
                            measured.value().logical_bytes == ByteSize{32U * mib});
    for (const std::size_t missing : {0U, 1U, 2U, 3U}) {
        runner.begin("M10 aggregate refuses one missing independently corroborated representation");
        auto broken = chunks;
        broken[missing].charges_corroborated = false;
        VRAMZ_CHECK(runner, !detail::calculate_multi_chunk_residency_totals(broken, unit));
    }
    for (const std::size_t field : {0U, 1U, 2U}) {
        runner.begin(
            "M10 aggregate overflow is rejected without wrapping logical RAW or compressed "
            "totals");
        auto overflow = chunks;
        const ByteSize large{std::numeric_limits<std::uint64_t>::max() - 1U};
        for (auto& chunk : overflow) {
            if (field == 0U) {
                chunk.logical_bytes = large;
            } else {
                chunk.raw_charge = large;
                chunk.compressed_charge = field == 1U ? ByteSize{2U} : ByteSize{large.value() - 2U};
            }
        }
        VRAMZ_CHECK(runner,
                    !detail::calculate_multi_chunk_residency_totals(overflow, ByteSize{2U}));
    }
    runner.begin("M10 exact 128 MiB stage boundary accepted; one extra byte and integer overflow "
                 "rejected");
    VRAMZ_CHECK(runner, detail::admit_multi_chunk_residency_stage(ByteSize{8U * mib},
                                                                  ByteSize{120U * mib}));
    VRAMZ_CHECK(runner, !detail::admit_multi_chunk_residency_stage(ByteSize{8U * mib},
                                                                   ByteSize{120U * mib + 1U}));
    VRAMZ_CHECK(runner, !detail::admit_multi_chunk_residency_stage(
                            ByteSize{std::numeric_limits<std::uint64_t>::max()}, ByteSize{1U}));
}
void preflight_tests(test::Runner& runner) {
    for (const detail::MultiChunkResidencySmokeOptions options :
         {detail::MultiChunkResidencySmokeOptions{},
          {true, 1, {595U, 84U, 0U}},
          {true, 0, {579U, 99U, 0U}},
          {true, 0, {595U, 84U, 0U}, 12030, 13020}}) {
        runner.begin("M10 acknowledgment device Driver minimum and CUDA major gate precede GPU "
                     "calls");
        Fixture f;
        f.run(options);
        VRAMZ_CHECK(runner,
                    f.report.result != CompressionSmokeResult::passed && f.cuda->log_size == 0U);
    }
    for (const bool overflow : {false, true}) {
        runner.begin("M10 excessive operation admission or overflow stops before primary-context "
                     "retention and physical mutation");
        Fixture f;
        if (overflow) {
            f.nvcomp->inject(GpuCodecFault::plan_temp_overflow);
        } else {
            f.codec->config.decompression_temp = detail::m10_physical_cap;
        }
        f.run();
        VRAMZ_CHECK(runner, f.report.result == CompressionSmokeResult::resource_cap);
        VRAMZ_CHECK(runner, f.calls(CudaCall::retain_primary) == 0U &&
                                f.calls(CudaCall::reserve) == 0U &&
                                f.calls(CudaCall::create) == 0U);
        f.empty(runner);
    }
    for (const bool exceeds : {false, true}) {
        runner.begin("M10 retained live compressed charge is added at every stage, accepting the "
                     "128 MiB boundary or stopping before chunk three mutation");
        Fixture f;
        f.codec->config.compression_bound_floor = ByteSize{22U * mib};
        f.codec->config.compression_temp = ByteSize{32U * mib - sizeof(detail::GpuBatchMetadata)};
        f.codec->config.decompression_temp =
            ByteSize{(exceeds ? 32U : 30U) * mib - sizeof(detail::GpuBatchMetadata)};
        f.run();
        VRAMZ_CHECK(runner, f.report.result == (exceeds ? CompressionSmokeResult::resource_cap
                                                        : CompressionSmokeResult::passed));
        VRAMZ_CHECK(runner, f.report.admission.peak == ByteSize{(exceeds ? 124U : 122U) * mib});
        if (exceeds) {
            VRAMZ_CHECK(runner, f.codec->observation.compressions == 3U &&
                                    f.codec->observation.decompressions == 0U &&
                                    f.calls(CudaCall::create) == 12U);
            VRAMZ_CHECK(runner, !f.report.simultaneous_compressed_residency_proven);
        } else {
            VRAMZ_CHECK(runner, f.report.peak_admitted_gpu_bytes == detail::m10_physical_cap);
        }
        f.empty(runner);
    }
}
void savings_faults(test::Runner& runner) {
    constexpr std::uint64_t occurrence{3U};
    for (const bool beneficial : {false, true}) {
        runner.begin("M10 one non-saving or non-beneficial chunk stops without aggregate "
                     "compensation and releases all retained chunks");
        Fixture f;
        f.codec->config.literal_prefix_bytes = ByteSize{7U * mib};
        f.nvcomp->inject(beneficial ? GpuCodecFault::literal_prefix_output
                                    : GpuCodecFault::literal_only_output,
                         occurrence);
        f.run();
        VRAMZ_CHECK(runner, f.report.result ==
                                (beneficial ? CompressionSmokeResult::no_physical_savings
                                            : CompressionSmokeResult::compression_not_beneficial));
        VRAMZ_CHECK(runner, f.codec->observation.compressions == occurrence &&
                                f.codec->observation.decompressions == 0U);
        VRAMZ_CHECK(runner, !f.report.simultaneous_compressed_residency_proven);
        const auto& chunk = f.report.chunks[static_cast<std::size_t>(occurrence - 1U)];
        VRAMZ_CHECK(runner, beneficial ? chunk.stored_bytes < chunk.logical_bytes
                                       : chunk.stored_bytes > chunk.logical_bytes);
        VRAMZ_CHECK(runner, chunk.compressed_charge >= chunk.raw_charge);
        f.empty(runner);
    }
}
void codec_faults(test::Runner& runner) {
    for (const std::uint64_t occurrence : {1U, 2U, 3U, 4U, 6U}) {
        runner.begin("M10 failed compression on each chunk and failed sequential restore stop once "
                     "and clean every previously retained representation");
        Fixture f;
        f.nvcomp->inject(GpuCodecFault::launch_failure, occurrence);
        f.run();
        VRAMZ_CHECK(runner,
                    f.report.result == CompressionSmokeResult::failed && f.report.has_error);
        VRAMZ_CHECK(runner,
                    f.codec->observation.compressions + f.codec->observation.decompressions ==
                        occurrence - 1U);
        VRAMZ_CHECK(runner, f.report.simultaneous_compressed_residency_proven == (occurrence > 4U));
        f.empty(runner);
    }
    for (const auto fault :
         {GpuCodecFault::status_failure, GpuCodecFault::size_zero, GpuCodecFault::size_oversized,
          GpuCodecFault::size_mismatch, GpuCodecFault::output_corruption,
          GpuCodecFault::crc_preserving_output_corruption}) {
        runner.begin("M10 malformed size, stored/logical CRC and CRC-neutral byte faults never "
                     "publish PASS");
        Fixture f;
        f.nvcomp->inject(fault, 6U);
        f.run();
        VRAMZ_CHECK(runner,
                    f.report.result == CompressionSmokeResult::failed && f.report.has_error);
        f.empty(runner);
    }
    for (const auto fault : {GpuCodecFault::premature_status, GpuCodecFault::premature_output}) {
        runner.begin("M10 partially visible results remain pending until same-stream completion");
        Fixture f;
        f.nvcomp->inject(fault, 3U);
        f.run();
        VRAMZ_CHECK(runner, f.report.result == CompressionSmokeResult::passed);
        f.empty(runner);
    }
}
[[nodiscard]] std::uint64_t count_before(const Fixture& baseline, CudaCall call,
                                         std::size_t position) {
    std::uint64_t count{};
    for (std::size_t i = 0U; i < position; ++i) {
        if (baseline.cuda->log[i].call == call) {
            ++count;
        }
    }
    return count;
}
void allocation_faults(test::Runner& runner, const Fixture& baseline) {
    constexpr std::size_t chunk{2U};
    const auto position = baseline.codec->launches[chunk].cuda_log_position;
    const auto workspace = count_before(baseline, CudaCall::create, position);
    for (const auto occurrence : {workspace - 1U, workspace, workspace + 1U}) {
        runner.begin("M10 bound-output, workspace and compaction allocation failures with prior "
                     "chunks live clean all retained compressed resources without retry");
        Fixture f;
        f.driver->inject(CudaCall::create, CudaFaultMode::before, {occurrence, 0U});
        f.run();
        VRAMZ_CHECK(runner, f.report.result != CompressionSmokeResult::passed &&
                                f.calls(CudaCall::create) == occurrence);
        VRAMZ_CHECK(runner, f.codec->observation.decompressions == 0U);
        f.empty(runner);
    }
    for (const auto mode : {CudaFaultMode::before, CudaFaultMode::malformed}) {
        runner.begin("M10 exact compaction copy failure or stored-byte CRC corruption releases "
                     "all earlier live chunks and forbids restore");
        Fixture f;
        f.driver->inject(CudaCall::device_to_device, mode, {chunk + 1U, 0U});
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
        static_cast<void>(::alarm(600U));
        Fixture f;
        if (fault != GpuCodecFault::count) {
            f.nvcomp->inject(fault, occurrence);
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
            const auto conserved = checked_add(
                observed->cuda->owned_bytes,
                observed->report.final_unmaterialized_reservations.value(), OperationId::verify);
            // The first private VA free follows encode-workspace release, before the
            // unused RAW restore and decode reservations end. Compaction excess was returned.
            const auto restore_reservations =
                checked_add(observed->report.admission.raw_charge.value(),
                            observed->report.admission.decompression.workspace_charge.value(),
                            OperationId::verify);
            const bool early_va =
                expected_call == CudaCall::free_address && expected_occurrence == 1U;
            const bool ok =
                observed->report.result != CompressionSmokeResult::passed &&
                observed->cuda->destroyed_drivers == 0U &&
                (expected_call == CudaCall::count ||
                 observed->calls(expected_call) == expected_occurrence) &&
                (!definite ||
                 (observed->report.has_error && observed->report.final_counts_known && conserved &&
                  observed->report.final_budget.value() == conserved.value() &&
                  restore_reservations &&
                  observed->report.final_unmaterialized_reservations.value() ==
                      (early_va ? restore_reservations.value() : 0U)));
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
void fatal_tests(test::Runner& runner, const Fixture& baseline) {
    for (const auto fault :
         {GpuCodecFault::metadata_submission_ambiguous, GpuCodecFault::metadata_publication_partial,
          GpuCodecFault::launch_ambiguous, GpuCodecFault::completion_failure}) {
        const std::uint64_t occurrence = fault == GpuCodecFault::completion_failure ? 6U : 3U;
        runner.begin("M10 ambiguous submission or completion with retained chunks fails stop "
                     "without destructive retry");
        fatal(runner, fault, CudaCall::count, CudaFaultMode::before, occurrence);
    }
    for (const auto fault : {GpuCodecFault::shutdown_failure, GpuCodecFault::shutdown_ambiguous}) {
        runner.begin("M10 stream shutdown failure cannot publish PASS or trigger cleanup retry");
        fatal(runner, fault, CudaCall::count, CudaFaultMode::before);
    }
    for (std::size_t chunk = 0U; chunk < 4U; ++chunk) {
        const auto address = baseline.codec->launches[4U + chunk].metadata.input;
        std::size_t position{};
        for (; position < baseline.cuda->log_size; ++position) {
            if (baseline.cuda->log[position].call == CudaCall::unmap &&
                baseline.cuda->log[position].identity == address) {
                break;
            }
        }
        VRAMZ_CHECK(runner, position < baseline.cuda->log_size);
        if (position == baseline.cuda->log_size) {
            continue;
        }
        const auto occurrence = count_before(baseline, CudaCall::unmap, position) + 1U;
        for (const auto mode : {CudaFaultMode::before, CudaFaultMode::after}) {
            if (mode == CudaFaultMode::after && chunk != 2U) {
                continue;
            }
            runner.begin("M10 cleanup failure on each independently owned compressed chunk keeps "
                         "known ownership or fails stop on ambiguity without retry");
            fatal(runner, GpuCodecFault::count, CudaCall::unmap, mode, occurrence);
        }
        if (chunk == 2U) {
            std::size_t release_position = position + 1U;
            while (release_position < baseline.cuda->log_size &&
                   baseline.cuda->log[release_position].call != CudaCall::release) {
                ++release_position;
            }
            VRAMZ_CHECK(runner, release_position < baseline.cuda->log_size);
            const auto release_occurrence =
                count_before(baseline, CudaCall::release, release_position) + 1U;
            for (const auto mode : {CudaFaultMode::before, CudaFaultMode::after}) {
                runner.begin("M10 physical release after unmapping retains exact ownership on "
                             "definite failure and stops on ambiguous cleanup");
                fatal(runner, GpuCodecFault::count, CudaCall::release, mode, release_occurrence);
            }
        }
    }
    for (const auto call : {CudaCall::free_address, CudaCall::release_primary}) {
        for (const auto mode : {CudaFaultMode::before, CudaFaultMode::after}) {
            runner.begin("M10 VA or context cleanup failure blocks PASS and never guesses release");
            fatal(runner, GpuCodecFault::count, call, mode);
        }
    }
}
} // namespace
int main() {
    test::Runner runner;
    Fixture baseline;
    baseline.run();
    runner.begin("M10 baseline flow must pass before replaying recorded fault positions");
    VRAMZ_CHECK(runner, baseline.report.result == CompressionSmokeResult::passed);
    if (baseline.report.result != CompressionSmokeResult::passed) {
        return runner.finish();
    }
    result_tests(runner, baseline);
    arithmetic_tests(runner);
    preflight_tests(runner);
    savings_faults(runner);
    codec_faults(runner);
    allocation_faults(runner, baseline);
    fatal_tests(runner, baseline);
    return runner.finish();
}
