#include "../test_support.hpp"

#include "vramz/detail/raw_smoke.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <string_view>
#include <type_traits>

using namespace vramz;
using namespace std::string_view_literals;
using detail::format_raw_smoke_report;
using detail::parse_raw_smoke_arguments;
using detail::RawSmokeReport;
using detail::RawSmokeResult;

namespace {

constexpr std::string_view identity =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
static_assert(std::is_trivially_copyable_v<detail::RawSmokeOptions>);
static_assert(noexcept(parse_raw_smoke_arguments({})));
static_assert(noexcept(format_raw_smoke_report(RawSmokeReport{}, {}, false, {})));

[[nodiscard]] RawSmokeReport successful_report() noexcept {
    RawSmokeReport report{};
    report.result = RawSmokeResult::passed;
    report.device_ordinal = 2;
    report.probe.device.name = std::array<char, 256U>{"Fake device"};
    report.probe.driver_version = 13030;
    report.probe.uva = true;
    report.probe.vmm = true;
    report.probe.minimum = ByteSize{65536U};
    report.probe.recommended = ByteSize{2097152U};
    report.logical_payload = ByteSize{65536U};
    report.physical_charge = ByteSize{65536U};
    report.crc_expected = 1234567U;
    report.crc_actual = report.crc_expected;
    report.h2d_success = true;
    report.d2h_success = true;
    report.size_equal = true;
    report.byte_compare = true;
    report.stable_va_reserved = true;
    report.mapping_verified = true;
    report.access_verified = true;
    report.cleanup_unmap = true;
    report.cleanup_physical_release = true;
    report.cleanup_va_free = true;
    report.cleanup_context_release = true;
    report.final_counts_known = true;
    return report;
}

void parser_tests(test::Runner& runner) {
    runner.begin("M7 empty CLI plans device zero without execution acknowledgement");
    const auto defaults = parse_raw_smoke_arguments({});
    VRAMZ_CHECK(runner, defaults);
    VRAMZ_CHECK(runner, !defaults.value().acknowledged && defaults.value().device == 0);
    VRAMZ_CHECK(runner, defaults.value().maximum_physical == detail::m7_physical_cap);

    runner.begin("M7 acknowledgement must be an exact explicit CLI flag");
    const std::array acknowledge{"--run-raw-smoke"sv};
    const auto acknowledged = parse_raw_smoke_arguments(acknowledge);
    VRAMZ_CHECK(runner, acknowledged && acknowledged.value().acknowledged);
    const std::array invalid_ack{"--run-raw-smoke=true"sv};
    VRAMZ_CHECK(runner, !parse_raw_smoke_arguments(invalid_ack));

    runner.begin("M7 explicitly selected device and lower cap parse in either option order");
    const std::array arguments{"--device"sv, "3"sv, "--run-raw-smoke"sv, "--max-physical-bytes"sv,
                               "65536"sv};
    const auto configured = parse_raw_smoke_arguments(arguments);
    VRAMZ_CHECK(runner, configured && configured.value().acknowledged);
    VRAMZ_CHECK(runner, configured.value().device == 3);
    VRAMZ_CHECK(runner, configured.value().maximum_physical == ByteSize{65536U});
    const std::array reverse{"--max-physical-bytes"sv, "64"sv, "--device"sv, "1"sv};
    const auto planned = parse_raw_smoke_arguments(reverse);
    VRAMZ_CHECK(runner, planned && !planned.value().acknowledged);
    VRAMZ_CHECK(runner, planned.value().device == 1);
    VRAMZ_CHECK(runner, planned.value().maximum_physical == ByteSize{64U});

    runner.begin("M7 help is an isolated nonexecuting plan and cannot override acknowledgement");
    const std::array help{"--help"sv};
    const auto planned_help = parse_raw_smoke_arguments(help);
    VRAMZ_CHECK(runner, planned_help && !planned_help.value().acknowledged);
    const std::array ambiguous{"--run-raw-smoke"sv, "--help"sv};
    VRAMZ_CHECK(runner, !parse_raw_smoke_arguments(ambiguous));

    runner.begin("M7 duplicate authorization device and allocation-limit flags are rejected");
    const std::array duplicate_ack{"--run-raw-smoke"sv, "--run-raw-smoke"sv};
    const std::array duplicate_device{"--device"sv, "0"sv, "--device"sv, "1"sv};
    const std::array duplicate_cap{"--max-physical-bytes"sv, "64"sv, "--max-physical-bytes"sv,
                                   "128"sv};
    VRAMZ_CHECK(runner, !parse_raw_smoke_arguments(duplicate_ack));
    VRAMZ_CHECK(runner, !parse_raw_smoke_arguments(duplicate_device));
    VRAMZ_CHECK(runner, !parse_raw_smoke_arguments(duplicate_cap));

    runner.begin("M7 unknown missing negative whitespace and malformed numeric options reject");
    for (const auto value : {""sv, "-1"sv, "+1"sv, " 1"sv, "1 "sv, "1x"sv, "0x1"sv}) {
        const std::array arguments_bad{"--device"sv, value};
        const auto parsed = parse_raw_smoke_arguments(arguments_bad);
        VRAMZ_CHECK(runner, !parsed && parsed.error().code == ErrorCode::invalid_argument);
    }
    const std::array unknown{"--compress"sv};
    const std::array missing_device{"--device"sv};
    const std::array missing_cap{"--max-physical-bytes"sv};
    VRAMZ_CHECK(runner, !parse_raw_smoke_arguments(unknown));
    VRAMZ_CHECK(runner, !parse_raw_smoke_arguments(missing_device));
    VRAMZ_CHECK(runner, !parse_raw_smoke_arguments(missing_cap));

    runner.begin("M7 numeric overflow is explicit before any partial option result is returned");
    for (const auto value : {"2147483648"sv, "18446744073709551616"sv}) {
        const std::array overflow{"--run-raw-smoke"sv, "--device"sv, value};
        const auto parsed = parse_raw_smoke_arguments(overflow);
        VRAMZ_CHECK(runner, !parsed && parsed.error().code == ErrorCode::arithmetic_overflow);
    }
    const std::array overflow_cap{"--max-physical-bytes"sv, "18446744073709551616"sv};
    const auto parsed = parse_raw_smoke_arguments(overflow_cap);
    VRAMZ_CHECK(runner, !parsed && parsed.error().code == ErrorCode::arithmetic_overflow);

    runner.begin("M7 maximum allocation cap and device narrowing boundaries are checked");
    const std::array maximum{"--device"sv, "2147483647"sv, "--max-physical-bytes"sv, "16777216"sv};
    const auto max = parse_raw_smoke_arguments(maximum);
    VRAMZ_CHECK(runner, max && max.value().device == std::numeric_limits<std::int32_t>::max());
    VRAMZ_CHECK(runner, max.value().maximum_physical == detail::m7_physical_cap);
    const std::array minimum{"--max-physical-bytes"sv, "1"sv};
    VRAMZ_CHECK(runner, parse_raw_smoke_arguments(minimum));
    for (const auto value : {"0"sv, "16777217"sv}) {
        const std::array outside{"--max-physical-bytes"sv, value};
        VRAMZ_CHECK(runner, !parse_raw_smoke_arguments(outside));
    }

    runner.begin("M7 parser has a fixed maximum argument count");
    const std::array excess{"--device"sv,        "0"sv,     "--max-physical-bytes"sv, "64"sv,
                            "--run-raw-smoke"sv, "--help"sv};
    VRAMZ_CHECK(runner, !parse_raw_smoke_arguments(excess));
}

void formatter_tests(test::Runner& runner) {
    std::array<char, 4096U> buffer{};
    const auto report = successful_report();
    runner.begin("M7 JSON includes every lifecycle byte and accounting field without addresses");
    const auto result = format_raw_smoke_report(report, identity, true, buffer);
    VRAMZ_CHECK(runner, !result.truncated && result.written > 0U);
    VRAMZ_CHECK(runner, result.written < buffer.size() && buffer[result.written] == '\0');
    const std::string_view json{buffer.data(), result.written};
    for (const auto field : {"\"m7_version\":1"sv,
                             "\"result\":\"PASS\""sv,
                             "\"execution_kind\":\"physical\""sv,
                             "\"source_identity_valid\":true"sv,
                             "\"device_ordinal\":2"sv,
                             "\"device_name\":\"Fake device\""sv,
                             "\"driver_api_version\":13030"sv,
                             "\"compute_mode\":0"sv,
                             "\"uva_supported\":true"sv,
                             "\"vmm_supported\":true"sv,
                             "\"minimum_granularity\":65536"sv,
                             "\"recommended_granularity\":2097152"sv,
                             "\"logical_payload_bytes\":65536"sv,
                             "\"physical_charge_bytes\":65536"sv,
                             "\"h2d_success\":true"sv,
                             "\"d2h_success\":true"sv,
                             "\"crc_expected\":1234567"sv,
                             "\"crc_actual\":1234567"sv,
                             "\"size_equal\":true"sv,
                             "\"byte_compare\":true"sv,
                             "\"stable_va_reserved\":true"sv,
                             "\"mapping_verified\":true"sv,
                             "\"access_verified\":true"sv,
                             "\"cleanup_unmap\":true"sv,
                             "\"cleanup_physical_release\":true"sv,
                             "\"cleanup_va_free\":true"sv,
                             "\"cleanup_context_release\":true"sv,
                             "\"final_counts_known\":true"sv,
                             "\"final_backend_resources\":0"sv,
                             "\"final_va_reservations\":0"sv,
                             "\"final_budget_charge\":0"sv,
                             "\"error\":null"sv,
                             "\"hardware_validation\":\"SMOKE_VALIDATED\""sv}) {
        VRAMZ_CHECK(runner, json.find(field) != std::string_view::npos);
    }
    VRAMZ_CHECK(runner, json.front() == '{' && json.back() == '}');
    VRAMZ_CHECK(runner, json.find(identity) != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("address") == std::string_view::npos);

    runner.begin("M7 fake success never claims physical hardware validation");
    const auto fake = format_raw_smoke_report(report, identity, false, buffer);
    const std::string_view fake_json{buffer.data(), fake.written};
    VRAMZ_CHECK(runner, !fake.truncated);
    VRAMZ_CHECK(runner, fake_json.find("\"execution_kind\":\"fake\"") != std::string_view::npos);
    VRAMZ_CHECK(runner,
                fake_json.find("\"hardware_validation\":\"NOT_TESTED\"") != std::string_view::npos);
    VRAMZ_CHECK(runner, fake_json.find("SMOKE_VALIDATED") == std::string_view::npos);

    runner.begin("M7 unsuccessful and planned reports cannot promote hardware validation");
    constexpr std::array states{
        RawSmokeResult::planned, RawSmokeResult::failed, RawSmokeResult::unsupported,
        RawSmokeResult::unsupported_granularity, RawSmokeResult::preflight_rejected};
    constexpr std::array labels{"PLANNED"sv, "FAIL"sv, "UNSUPPORTED"sv,
                                "M7_UNSUPPORTED_GRANULARITY"sv, "PREFLIGHT_REJECTED"sv};
    for (std::size_t index = 0U; index < states.size(); ++index) {
        auto changed = report;
        changed.result = states[index];
        const auto formatted = format_raw_smoke_report(changed, identity, true, buffer);
        const std::string_view output{buffer.data(), formatted.written};
        VRAMZ_CHECK(runner, !formatted.truncated);
        VRAMZ_CHECK(runner, output.find(labels[index]) != std::string_view::npos);
        VRAMZ_CHECK(runner, output.find("SMOKE_VALIDATED") == std::string_view::npos);
    }

    runner.begin("M7 source identity rejects nonhex wrong lengths and injection text");
    for (const auto digest :
         {""sv, "123"sv, "\"},\"injected\":true,{\""sv,
          "g123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"sv}) {
        const auto formatted = format_raw_smoke_report(report, digest, false, buffer);
        const std::string_view output{buffer.data(), formatted.written};
        VRAMZ_CHECK(runner, !formatted.truncated);
        VRAMZ_CHECK(runner, output.find("\"source_sha256\":null") != std::string_view::npos);
        VRAMZ_CHECK(runner,
                    output.find("\"source_identity_valid\":false") != std::string_view::npos);
        VRAMZ_CHECK(runner, output.find("injected") == std::string_view::npos);
    }

    runner.begin("M7 device name escapes quotes slashes controls and non-ASCII bytes");
    auto escaped = report;
    escaped.probe.device.name = {};
    escaped.probe.device.name[0U] = '"';
    escaped.probe.device.name[1U] = '\\';
    escaped.probe.device.name[2U] = '\n';
    escaped.probe.device.name[3U] = static_cast<char>(0x80);
    const auto escaped_result = format_raw_smoke_report(escaped, identity, false, buffer);
    const std::string_view escaped_json{buffer.data(), escaped_result.written};
    VRAMZ_CHECK(runner, !escaped_result.truncated);
    VRAMZ_CHECK(runner, escaped_json.find("\"device_name\":\"\\\"\\\\\\u000a\\u0080\"") !=
                            std::string_view::npos);
    VRAMZ_CHECK(runner, escaped_json.find('\n') == std::string_view::npos);

    runner.begin("M7 maximum unterminated device names remain bounded JSON strings");
    auto full_name = report;
    full_name.probe.device.name.fill(static_cast<char>(0xff));
    const auto full_result = format_raw_smoke_report(full_name, identity, false, buffer);
    VRAMZ_CHECK(runner, !full_result.truncated && full_result.written < buffer.size());
    VRAMZ_CHECK(runner, buffer[full_result.written] == '\0');

    runner.begin("M7 diagnostics omit object identity and detail while preserving numeric errors");
    auto failed = report;
    failed.result = RawSmokeResult::failed;
    failed.has_error = true;
    failed.error = make_error(ErrorCode::backend_failure, OperationId::verify, 123456789012345U,
                              234567890123456U);
    failed.error.native_code = std::numeric_limits<std::int64_t>::min();
    const auto failed_result = format_raw_smoke_report(failed, identity, true, buffer);
    const std::string_view failed_json{buffer.data(), failed_result.written};
    VRAMZ_CHECK(runner, !failed_result.truncated);
    VRAMZ_CHECK(runner, failed_json.find("\"error\":{\"code\":11,\"operation\":7,") !=
                            std::string_view::npos);
    VRAMZ_CHECK(runner,
                failed_json.find("\"native_code\":-9223372036854775808") != std::string_view::npos);
    for (const auto forbidden :
         {"object"sv, "detail"sv, "123456789012345"sv, "234567890123456"sv}) {
        VRAMZ_CHECK(runner, failed_json.find(forbidden) == std::string_view::npos);
    }

    runner.begin("M7 unknown resource totals are null rather than falsely reported zero");
    const auto unknown = format_raw_smoke_report(RawSmokeReport{}, {}, false, buffer);
    const std::string_view unknown_json{buffer.data(), unknown.written};
    VRAMZ_CHECK(runner, !unknown.truncated);
    for (const auto field :
         {"\"final_backend_resources\":null"sv, "\"final_va_reservations\":null"sv,
          "\"final_budget_charge\":null"sv}) {
        VRAMZ_CHECK(runner, unknown_json.find(field) != std::string_view::npos);
    }

    runner.begin("M7 formatter truncation always terminates and respects every output bound");
    const auto complete = format_raw_smoke_report(report, identity, false, buffer);
    std::array<char, 4096U> limited{};
    for (std::size_t size = 0U; size <= complete.written + 1U; ++size) {
        limited.fill('#');
        const auto formatted =
            format_raw_smoke_report(report, identity, false, std::span{limited}.first(size));
        VRAMZ_CHECK(runner, formatted.truncated == (size <= complete.written));
        VRAMZ_CHECK(runner, formatted.written == (size == 0U ? 0U : size - 1U));
        if (size != 0U) {
            VRAMZ_CHECK(runner, limited[formatted.written] == '\0');
        }
        VRAMZ_CHECK(runner, limited[size] == '#');
    }

    runner.begin("M7 numeric telemetry handles full uint64 range without format allocation");
    auto large = report;
    large.final_backend_resources = std::numeric_limits<std::uint64_t>::max();
    large.final_budget_charge = ByteSize{std::numeric_limits<std::uint64_t>::max()};
    const auto large_result = format_raw_smoke_report(large, identity, false, buffer);
    const std::string_view large_json{buffer.data(), large_result.written};
    VRAMZ_CHECK(runner, !large_result.truncated);
    VRAMZ_CHECK(runner, large_json.find("\"final_backend_resources\":18446744073709551615") !=
                            std::string_view::npos);
    VRAMZ_CHECK(runner, large_json.find("\"final_budget_charge\":18446744073709551615") !=
                            std::string_view::npos);
}

} // namespace

int main() {
    test::Runner runner;
    parser_tests(runner);
    formatter_tests(runner);
    return runner.finish();
}
