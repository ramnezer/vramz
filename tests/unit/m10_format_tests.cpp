#include "../test_support.hpp"
#include "vramz/detail/multi_chunk_residency_smoke.hpp"
#include "vramz/detail/smoke_json.hpp"

#include <algorithm>
#include <array>

namespace {
constexpr std::string_view source{
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"};
}

int main() {
    using namespace vramz;
    test::Runner runner;
    runner.begin(
        "M10 arguments require acknowledgement and explicit device zero with fixed limits");
    const std::array<std::string_view, 3U> accepted{"--run-multi-chunk-residency-smoke", "--device",
                                                    "0"};
    const auto parsed = detail::parse_multi_chunk_residency_smoke_arguments(accepted);
    VRAMZ_CHECK(runner, parsed && parsed.value().acknowledged && parsed.value().device == 0);
    for (const auto args :
         {std::array<std::string_view, 2U>{"--device", "1"},
          {"--logical-bytes", "8388608"},
          {"--max-physical-bytes", "134217728"},
          {"--chunk-count", "4"},
          {"--policy", "off"},
          {"--pressure", "off"},
          {"--host-fallback", "off"},
          {"--device", "00"},
          {"--run-multi-chunk-residency-smoke", "--run-multi-chunk-residency-smoke"}}) {
        VRAMZ_CHECK(runner, !detail::parse_multi_chunk_residency_smoke_arguments(args));
    }
    const std::array<std::string_view, 1U> missing_device{"--run-multi-chunk-residency-smoke"};
    VRAMZ_CHECK(runner, !detail::parse_multi_chunk_residency_smoke_arguments(missing_device));
    const std::array<std::string_view, 1U> help{"--help"};
    const auto help_result = detail::parse_multi_chunk_residency_smoke_arguments(help);
    VRAMZ_CHECK(runner, help_result && !help_result.value().acknowledged);
    const std::array<std::string_view, 4U> excessive{"--device", "0", "--device", "0"};
    VRAMZ_CHECK(runner, !detail::parse_multi_chunk_residency_smoke_arguments(excessive));
    VRAMZ_CHECK(runner, detail::parse_multi_chunk_residency_smoke_arguments({}));

    runner.begin(
        "M10 fixed payload variants preserve zeros variation periodicity and distinctness");
    std::array<std::array<std::byte, 8192U>, detail::m10_chunk_count> samples{};
    for (std::size_t chunk = 0U; chunk < samples.size(); ++chunk) {
        for (std::size_t index = 0U; index < samples[chunk].size(); ++index) {
            samples[chunk][index] = detail::multi_chunk_residency_payload_byte(chunk, index);
            VRAMZ_CHECK(runner,
                        samples[chunk][index] == detail::multi_chunk_residency_payload_byte(
                                                     chunk, index + std::size_t{4096U} * 17U));
        }
        VRAMZ_CHECK(runner,
                    std::ranges::any_of(samples[chunk], [](auto b) { return b == std::byte{}; }));
        VRAMZ_CHECK(runner,
                    std::ranges::any_of(samples[chunk], [](auto b) { return b != std::byte{}; }));
        VRAMZ_CHECK(runner, samples[chunk][1U] != samples[chunk][2U]);
        for (std::size_t prior = 0U; prior < chunk; ++prior) {
            VRAMZ_CHECK(runner, samples[chunk] != samples[prior]);
        }
    }

    runner.begin(
        "M10 bounded JSON publishes four distinct chunk objects and no unproven aggregate");
    detail::MultiChunkResidencySmokeReport report{};
    std::array<char, 16384U> buffer{};
    for (std::size_t i = 0U; i < report.chunks.size(); ++i) {
        report.chunks[i].chunk_index = static_cast<std::uint32_t>(i);
    }
    const auto format =
        detail::format_multi_chunk_residency_smoke_report(report, "invalid", false, buffer);
    const std::string_view json{buffer.data(), format.written};
    VRAMZ_CHECK(runner, !format.truncated && json.starts_with("{\"m10_version\":1,"));
    VRAMZ_CHECK(runner, json.find("\"chunk_count\":4") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"logical_bytes_per_chunk\":8388608") != std::string_view::npos);
    VRAMZ_CHECK(runner,
                json.find("\"maximum_physical_bytes\":134217728") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"source_sha256\":null") != std::string_view::npos);
    VRAMZ_CHECK(runner,
                json.find("\"aggregate_raw_baseline_charge\":null") != std::string_view::npos);
    VRAMZ_CHECK(runner,
                json.find("\"aggregate_physical_charge_ratio\":null") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"final_budget_charge\":null") != std::string_view::npos);
    VRAMZ_CHECK(runner,
                json.find("\"final_unmaterialized_reservations\":null") != std::string_view::npos);
    VRAMZ_CHECK(runner,
                json.find("\"hardware_validation\":\"NOT_TESTED\"") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"gate_g\":\"NOT_RUN\"") != std::string_view::npos);
    for (const auto index :
         {"\"chunk_index\":0", "\"chunk_index\":1", "\"chunk_index\":2", "\"chunk_index\":3"}) {
        VRAMZ_CHECK(runner, json.find(index) != std::string_view::npos);
    }

    runner.begin(
        "M10 snapshot reports measured baseline and live charge without hardware claims in Fake");
    report.result = detail::CompressionSmokeResult::passed;
    report.simultaneous_compressed_residency_proven = true;
    report.totals = {ByteSize{33554432U}, ByteSize{33554432U}, ByteSize{8388608U},
                     ByteSize{25165824U}};
    report.final_counts_known = true;
    const auto snapshot_format =
        detail::format_multi_chunk_residency_smoke_report(report, source, false, buffer);
    const std::string_view snapshot{buffer.data(), snapshot_format.written};
    VRAMZ_CHECK(runner, !snapshot_format.truncated);
    VRAMZ_CHECK(runner, snapshot.find("\"aggregate_raw_baseline_charge\":33554432") !=
                            std::string_view::npos);
    VRAMZ_CHECK(runner, snapshot.find("\"aggregate_compressed_physical_charge\":8388608") !=
                            std::string_view::npos);
    VRAMZ_CHECK(runner, snapshot.find("\"aggregate_physical_bytes_saved\":25165824") !=
                            std::string_view::npos);
    VRAMZ_CHECK(runner,
                snapshot.find("\"aggregate_physical_charge_ratio\":4") != std::string_view::npos);
    VRAMZ_CHECK(runner, snapshot.find("\"final_budget_charge\":0") != std::string_view::npos);
    VRAMZ_CHECK(runner,
                snapshot.find("\"final_unmaterialized_reservations\":0") != std::string_view::npos);
    VRAMZ_CHECK(runner,
                snapshot.find("\"hardware_validation\":\"NOT_TESTED\"") != std::string_view::npos);
    report.simultaneous_compressed_residency_proven = false;
    const auto absent_format =
        detail::format_multi_chunk_residency_smoke_report(report, source, true, buffer);
    VRAMZ_CHECK(runner,
                (std::string_view{buffer.data(), absent_format.written})
                        .find("\"hardware_validation\":\"NOT_TESTED\"") != std::string_view::npos);

    runner.begin("M10 failure and diagnostic JSON does not discard exact known failed ownership");
    report.result = detail::CompressionSmokeResult::no_physical_savings;
    report.chunks[1U].charges_corroborated = true;
    report.chunks[1U].raw_charge = ByteSize{8388608U};
    report.chunks[1U].compressed_charge = ByteSize{8388608U};
    report.final_compressed = 2U;
    report.final_budget = ByteSize{6291456U};
    report.final_unmaterialized_reservations = ByteSize{2097152U};
    report.has_error = true;
    report.error = make_error(ErrorCode::unsupported, OperationId::runtime_create);
    const auto failure_format =
        detail::format_multi_chunk_residency_smoke_report(report, source, false, buffer);
    const std::string_view failure{buffer.data(), failure_format.written};
    VRAMZ_CHECK(runner,
                failure.find("\"result\":\"NO_PHYSICAL_SAVINGS\"") != std::string_view::npos);
    VRAMZ_CHECK(runner, failure.find("\"final_compressed_backing\":2") != std::string_view::npos);
    VRAMZ_CHECK(runner, failure.find("\"final_budget_charge\":6291456") != std::string_view::npos);
    VRAMZ_CHECK(runner, failure.find("\"final_unmaterialized_reservations\":2097152") !=
                            std::string_view::npos);
    VRAMZ_CHECK(runner,
                failure.find("\"runtime_compatibility_failure\":") != std::string_view::npos);

    runner.begin("SmokeJsonWriter object arrays escape strings and restore parent separators");
    const std::array<std::string_view, 2U> entries{"quote\" slash\\", "line\n"};
    detail::SmokeJsonWriter writer{buffer};
    writer.number("before", 1U);
    writer.object_array("items", entries, [](auto& output, auto entry) noexcept {
        output.text({"text", entry});
        output.boolean("valid", true);
    });
    writer.number("after", 2U);
    const auto array_format = writer.finish();
    constexpr std::string_view expected{
        "{\"before\":1,\"items\":[{\"text\":\"quote\\\" slash\\\\\",\"valid\":true},"
        "{\"text\":\"line\\u000a\",\"valid\":true}],\"after\":2}"};
    VRAMZ_CHECK(runner, !array_format.truncated &&
                            (std::string_view{buffer.data(), array_format.written}) == expected);

    runner.begin("SmokeJsonWriter empty arrays and bounded output never overrun sentinels");
    const std::array<std::uint32_t, 0U> empty{};
    detail::SmokeJsonWriter empty_writer{buffer};
    empty_writer.object_array(
        "items", empty, [](auto& output, auto entry) noexcept { output.number("value", entry); });
    const auto empty_format = empty_writer.finish();
    VRAMZ_CHECK(runner,
                (std::string_view{buffer.data(), empty_format.written}) == "{\"items\":[]}");
    for (std::size_t size = 0U; size < 64U; ++size) {
        std::array<char, 66U> guarded{};
        guarded.fill('X');
        const auto truncated = detail::format_multi_chunk_residency_smoke_report(
            report, source, false, std::span{guarded}.subspan(1U, size));
        VRAMZ_CHECK(runner,
                    truncated.truncated && guarded.front() == 'X' && guarded[size + 1U] == 'X');
    }
    return runner.finish();
}
