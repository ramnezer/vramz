#include "../test_support.hpp"
#include "vramz/crc32c.hpp"
#include "vramz/detail/physical_savings_smoke.hpp"
#include "vramz/detail/smoke_json.hpp"

#include <algorithm>
#include <array>
#include <limits>

int main() {
    using namespace vramz;
    test::Runner runner;
    runner.begin(
        "M9 arguments require explicit device zero and reject adjustable sizes caps and modes");
    const std::array<std::string_view, 3U> accepted{"--run-physical-savings-smoke", "--device",
                                                    "0"};
    VRAMZ_CHECK(runner, detail::parse_physical_savings_smoke_arguments(accepted));
    for (const auto args : {std::array<std::string_view, 2U>{"--device", "1"},
                            {"--logical-bytes", "8388608"},
                            {"--max-physical-bytes", "100663296"},
                            {"--policy", "off"},
                            {"--pressure", "off"},
                            {"--device", "00"},
                            {"--run-physical-savings-smoke", "--run-physical-savings-smoke"}}) {
        VRAMZ_CHECK(runner, !detail::parse_physical_savings_smoke_arguments(args));
    }
    const std::array<std::string_view, 1U> missing_device{"--run-physical-savings-smoke"};
    VRAMZ_CHECK(runner, !detail::parse_physical_savings_smoke_arguments(missing_device));
    runner.begin(
        "M9 savings arithmetic rejects zero unaligned unchanged or larger charge without overflow");
    constexpr auto limit = std::numeric_limits<std::uint64_t>::max();
    VRAMZ_CHECK(runner, detail::saves_physical_unit(ByteSize{8U}, ByteSize{2U}, ByteSize{2U}));
    VRAMZ_CHECK(runner, detail::saves_physical_unit(ByteSize{8U}, ByteSize{6U}, ByteSize{2U}));
    VRAMZ_CHECK(runner,
                detail::saves_physical_unit(ByteSize{limit}, ByteSize{limit - 1U}, ByteSize{1U}));
    for (const auto triple : {std::array<std::uint64_t, 3U>{8U, 8U, 2U},
                              {8U, 10U, 2U},
                              {8U, 0U, 2U},
                              {8U, 2U, 0U},
                              {8U, 7U, 2U},
                              {7U, 2U, 2U},
                              {limit, limit, 2U}}) {
        VRAMZ_CHECK(runner, !detail::saves_physical_unit(ByteSize{triple[0U]}, ByteSize{triple[1U]},
                                                         ByteSize{triple[2U]}));
    }
    runner.begin(
        "M9 deterministic pattern contains zeros nonzero variation and repeating structure");
    std::array<std::byte, 8192U> pattern{};
    for (std::size_t i = 0U; i < pattern.size(); ++i) {
        pattern[i] = detail::physical_savings_payload_byte(i);
    }
    VRAMZ_CHECK(runner, std::ranges::any_of(pattern, [](auto b) { return b == std::byte{}; }));
    VRAMZ_CHECK(runner, std::ranges::any_of(pattern, [](auto b) { return b != std::byte{}; }));
    VRAMZ_CHECK(runner, pattern[1U] != pattern[2U]);
    for (std::size_t i = 0U; i < pattern.size(); ++i) {
        VRAMZ_CHECK(runner, pattern[i] == detail::physical_savings_payload_byte(
                                              i + std::size_t{4096U} * 17U));
    }
    const auto expected_crc = crc32c(pattern);
    constexpr std::array difference{std::byte{1U}, std::byte{3U}, std::byte{0x83U},
                                    std::byte{0x6bU}, std::byte{0xf2U}};
    for (std::size_t i = 0U; i < difference.size(); ++i) {
        pattern[i] ^= difference[i];
    }
    VRAMZ_CHECK(runner, crc32c(pattern) == expected_crc &&
                            pattern.front() != detail::physical_savings_payload_byte(0U));
    runner.begin(
        "M9 bounded JSON distinguishes unknown ratios negative savings fake results and Gate G");
    detail::CompressionSmokeReport report{};
    std::array<char, 8192U> buffer{};
    const auto result =
        detail::format_physical_savings_smoke_report(report, "invalid", false, buffer);
    const std::string_view json{buffer.data(), result.written};
    VRAMZ_CHECK(runner,
                !result.truncated && json.find("\"m9_version\":1") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"source_sha256\":null") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"physical_charge_ratio\":null") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"physical_bytes_saved\":null") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"gate_g\":\"NOT_RUN\"") != std::string_view::npos);
    std::array<char, 8U> small{};
    VRAMZ_CHECK(runner,
                detail::format_physical_savings_smoke_report(report, "", false, small).truncated);
    report.charges_corroborated = true;
    report.raw_charge = ByteSize{8U};
    report.compressed_charge = ByteSize{10U};
    report.stored_bytes = ByteSize{9U};
    report.admission.logical_size = ByteSize{8U};
    report.result = detail::CompressionSmokeResult::compression_not_beneficial;
    const auto negative = detail::format_physical_savings_smoke_report(report, "", false, buffer);
    const std::string_view negative_json{buffer.data(), negative.written};
    VRAMZ_CHECK(runner,
                negative_json.find("\"physical_bytes_saved\":-2") != std::string_view::npos);
    VRAMZ_CHECK(runner,
                negative_json.find("\"physical_charge_ratio\":0.8") != std::string_view::npos);
    VRAMZ_CHECK(runner, negative_json.find("\"hardware_validation\":\"NOT_TESTED\"") !=
                            std::string_view::npos);
    runner.begin(
        "M9 diagnostic ratio and difference support unsigned extremes and zero denominators");
    detail::SmokeJsonWriter writer{buffer};
    writer.ratio("ratio", limit, 1U);
    writer.ratio("unknown", 1U, 0U);
    writer.difference("negative", 0U, limit, true);
    const auto extremes = writer.finish();
    VRAMZ_CHECK(runner, !extremes.truncated);
    VRAMZ_CHECK(runner, std::string_view{buffer.data()}.find("-18446744073709551615") !=
                            std::string_view::npos);
    return runner.finish();
}
