#include "vramz/detail/raw_smoke.hpp"
#include "vramz/detail/smoke_json.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <system_error>

namespace vramz::detail {
namespace {

[[nodiscard]] constexpr Error
argument_error(ErrorCode code = ErrorCode::invalid_argument) noexcept {
    return make_error(code, OperationId::runtime_create);
}

[[nodiscard]] Result<std::uint64_t> parse_unsigned(std::string_view text) noexcept {
    if (text.empty()) {
        return argument_error();
    }
    std::uint64_t value{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec == std::errc::result_out_of_range) {
        return argument_error(ErrorCode::arithmetic_overflow);
    }
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        return argument_error();
    }
    return value;
}

[[nodiscard]] constexpr std::string_view result_name(RawSmokeResult result) noexcept {
    switch (result) {
    case RawSmokeResult::planned:
        return "PLANNED";
    case RawSmokeResult::passed:
        return "PASS";
    case RawSmokeResult::failed:
        return "FAIL";
    case RawSmokeResult::unsupported:
        return "UNSUPPORTED";
    case RawSmokeResult::unsupported_granularity:
        return "M7_UNSUPPORTED_GRANULARITY";
    case RawSmokeResult::preflight_rejected:
        return "PREFLIGHT_REJECTED";
    }
    return "FAIL";
}

[[nodiscard]] bool valid_sha256(std::string_view digest) noexcept {
    return digest.size() == 64U && std::ranges::all_of(digest, [](char value) {
               return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') ||
                      (value >= 'A' && value <= 'F');
           });
}

} // namespace

Result<RawSmokeOptions>
parse_raw_smoke_arguments(std::span<const std::string_view> arguments) noexcept {
    RawSmokeOptions options{};
    if (arguments.size() == 1U && arguments.front() == "--help") {
        return options;
    }
    if (arguments.size() > 5U) {
        return argument_error();
    }
    bool saw_device{};
    bool saw_cap{};
    for (std::size_t index = 0U; index < arguments.size(); ++index) {
        const auto argument = arguments[index];
        if (argument == "--run-raw-smoke") {
            if (options.acknowledged) {
                return argument_error();
            }
            options.acknowledged = true;
            continue;
        }
        const bool device = argument == "--device";
        const bool cap = argument == "--max-physical-bytes";
        if ((!device && !cap) || (device && saw_device) || (cap && saw_cap) ||
            index + 1U >= arguments.size()) {
            return argument_error();
        }
        const auto parsed = parse_unsigned(arguments[++index]);
        if (!parsed) {
            return parsed.error();
        }
        if (device) {
            if (parsed.value() >
                static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
                return argument_error(ErrorCode::arithmetic_overflow);
            }
            options.device = static_cast<std::int32_t>(parsed.value());
            saw_device = true;
        } else {
            if (parsed.value() == 0U || parsed.value() > m7_physical_cap.value()) {
                return argument_error();
            }
            options.maximum_physical = ByteSize{parsed.value()};
            saw_cap = true;
        }
    }
    return options;
}

FormatResult format_raw_smoke_report(const RawSmokeReport& report, std::string_view source_sha256,
                                     bool physical_execution, std::span<char> output) noexcept {
    SmokeJsonWriter writer{output};
    writer.number("m7_version", 1U);
    writer.text({.key = "result", .value = result_name(report.result)});
    writer.text({.key = "execution_kind", .value = physical_execution ? "physical" : "fake"});
    const bool identity_valid = valid_sha256(source_sha256);
    writer.boolean("source_identity_valid", identity_valid);
    if (identity_valid) {
        writer.text({.key = "source_sha256", .value = source_sha256});
    } else {
        writer.null("source_sha256");
    }
    writer.signed_number("device_ordinal", report.device_ordinal);
    writer.number("maximum_physical_bytes", report.maximum_physical.value());
    if (report.result == RawSmokeResult::planned) {
        writer.text({.key = "planned_operations",
                     .value = "one selected device: initialize, probe, reserve one minimum unit, "
                              "create GPU_RAW, map, RW access, corroborate, one HtoD, one DtoH, "
                              "verify bytes and CRC32C, whole unmap, release handle, free VA, "
                              "release primary context, stop; no compression or policy"});
    }
    const auto end = std::ranges::find(report.probe.device.name, '\0');
    writer.text({.key = "device_name",
                 .value = std::string_view{
                     report.probe.device.name.data(),
                     static_cast<std::size_t>(end - report.probe.device.name.begin())}});
    writer.signed_number("driver_api_version", report.probe.driver_version);
    writer.signed_number("compute_mode", report.probe.device.compute_mode);
    writer.boolean("uva_supported", report.probe.uva);
    writer.boolean("vmm_supported", report.probe.vmm);
    writer.number("minimum_granularity", report.probe.minimum.value());
    writer.number("recommended_granularity", report.probe.recommended.value());
    writer.number("logical_payload_bytes", report.logical_payload.value());
    writer.number("physical_charge_bytes", report.physical_charge.value());
    writer.boolean("h2d_success", report.h2d_success);
    writer.boolean("d2h_success", report.d2h_success);
    writer.number("crc_expected", report.crc_expected);
    writer.number("crc_actual", report.crc_actual);
    writer.boolean("size_equal", report.size_equal);
    writer.boolean("byte_compare", report.byte_compare);
    writer.boolean("stable_va_reserved", report.stable_va_reserved);
    writer.boolean("mapping_verified", report.mapping_verified);
    writer.boolean("access_verified", report.access_verified);
    writer.boolean("cleanup_unmap", report.cleanup_unmap);
    writer.boolean("cleanup_physical_release", report.cleanup_physical_release);
    writer.boolean("cleanup_va_free", report.cleanup_va_free);
    writer.boolean("cleanup_context_release", report.cleanup_context_release);
    writer.boolean("final_counts_known", report.final_counts_known);
    writer.nullable_number("final_backend_resources", report.final_backend_resources,
                           report.final_counts_known);
    writer.nullable_number("final_va_reservations", report.final_va_reservations,
                           report.final_counts_known);
    writer.nullable_number("final_budget_charge", report.final_budget_charge.value(),
                           report.final_counts_known);
    if (report.has_error) {
        writer.error(report.error);
    } else {
        writer.null("error");
    }
    writer.text({.key = "hardware_validation",
                 .value = physical_execution && report.result == RawSmokeResult::passed
                              ? "SMOKE_VALIDATED"
                              : "NOT_TESTED"});
    return writer.finish();
}

} // namespace vramz::detail
