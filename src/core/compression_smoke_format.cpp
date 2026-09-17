#include "vramz/detail/physical_savings_smoke.hpp"
#include "vramz/detail/smoke_json.hpp"

#include <algorithm>
#include <charconv>
#include <system_error>

namespace vramz::detail {
namespace {
[[nodiscard]] constexpr std::string_view name(CompressionSmokeResult result) noexcept {
    switch (result) {
    case CompressionSmokeResult::planned:
        return "PLANNED";
    case CompressionSmokeResult::passed:
        return "PASS";
    case CompressionSmokeResult::failed:
        return "FAIL";
    case CompressionSmokeResult::unsupported:
        return "UNSUPPORTED";
    case CompressionSmokeResult::resource_cap:
        return "RESOURCE_CAP";
    case CompressionSmokeResult::no_physical_savings:
        return "NO_PHYSICAL_SAVINGS";
    case CompressionSmokeResult::compression_not_beneficial:
        return "COMPRESSION_NOT_BENEFICIAL";
    case CompressionSmokeResult::preflight_rejected:
        return "PREFLIGHT_REJECTED";
    }
    return "FAIL";
}
} // namespace

Result<CompressionSmokeOptions>
parse_compression_smoke_arguments(std::span<const std::string_view> arguments) noexcept {
    CompressionSmokeOptions options{};
    const auto invalid = make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    if (arguments.size() == 1U && arguments.front() == "--help") {
        return options;
    }
    if (arguments.size() > 7U) {
        return invalid;
    }
    bool saw_device{};
    bool saw_size{};
    bool saw_cap{};
    for (std::size_t index = 0U; index < arguments.size(); ++index) {
        const auto argument = arguments[index];
        if (argument == "--run-compression-smoke") {
            if (options.acknowledged) {
                return invalid;
            }
            options.acknowledged = true;
            continue;
        }
        const bool device = argument == "--device";
        const bool size = argument == "--logical-bytes";
        const bool cap = argument == "--max-physical-bytes";
        if ((!device && !size && !cap) || (device && saw_device) || (size && saw_size) ||
            (cap && saw_cap) || index + 1U >= arguments.size()) {
            return invalid;
        }
        const auto value = arguments[++index];
        std::uint64_t number{};
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
        if (value.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != value.data() + value.size()) {
            return invalid;
        }
        if (device) {
            if (number != 0U) {
                return invalid;
            }
            saw_device = true;
        } else if (size) {
            if (number == 0U || number > m8_logical_cap.value()) {
                return invalid;
            }
            options.logical_size = ByteSize{number};
            saw_size = true;
        } else {
            if (number == 0U || number > m8_physical_cap.value()) {
                return invalid;
            }
            options.physical_cap = ByteSize{number};
            saw_cap = true;
        }
    }
    return options;
}

namespace {
FormatResult format_report(const CompressionSmokeReport& report, std::string_view source_sha256,
                           bool physical_execution, std::span<char> output, bool savings) noexcept {
    SmokeJsonWriter writer{output};
    writer.number(savings ? "m9_version" : "m8_version", 1U);
    writer.text({"result", name(report.result)});
    writer.text({"execution_kind", physical_execution ? "physical" : "fake"});
    const bool valid_source = source_sha256.size() == 64U &&
                              std::ranges::all_of(source_sha256,
                                                  [](char value) {
                                                      return (value >= '0' && value <= '9') ||
                                                             (value >= 'a' && value <= 'f');
                                                  }) &&
                              source_sha256.find_first_not_of('0') != std::string_view::npos;
    if (valid_source) {
        writer.text({"source_sha256", source_sha256});
    } else {
        writer.null("source_sha256");
    }
    writer.signed_number("device_ordinal", report.device_ordinal);
    const auto end = std::ranges::find(report.probe.device.name, '\0');
    writer.text({"device_name",
                 {report.probe.device.name.data(),
                  static_cast<std::size_t>(end - report.probe.device.name.begin())}});
    writer.signed_number("driver_api_version", report.probe.driver_version);
    writer.signed_number("prior_m7_driver_api_version", report.prior_driver_api);
    writer.signed_number("cuda_runtime_api_version", report.runtime_api);
    writer.number("nvidia_driver_branch", report.installed_driver.branch);
    writer.number("nvidia_driver_minor", report.installed_driver.minor);
    writer.number("nvidia_driver_patch", report.installed_driver.patch);
    writer.text({"compatibility_candidate", compatibility_name(report.compatibility)});
    writer.text({"cuda_development_version", "13.3.1"});
    writer.text({"nvcomp_version", "5.3.0"});
    writer.text({"nvcomp_build", "5.3.0.16"});
    writer.boolean("uva_supported", report.probe.uva);
    writer.boolean("vmm_supported", report.probe.vmm);
    writer.number("minimum_granularity", report.probe.minimum.value());
    writer.number("recommended_granularity", report.probe.recommended.value());
    writer.number("raw_logical_bytes", report.admission.logical_size.value());
    writer.number("raw_physical_charge",
                  savings ? report.raw_charge.value() : report.admission.raw_charge.value());
    if (savings) {
        writer.boolean("charges_corroborated", report.charges_corroborated);
        const bool known = report.charges_corroborated && report.compressed_charge != ByteSize{};
        writer.difference("physical_bytes_saved", report.raw_charge.value(),
                          report.compressed_charge.value(), known);
        writer.ratio("stored_compression_ratio", report.admission.logical_size.value(),
                     known ? report.stored_bytes.value() : 0U);
        writer.ratio("physical_charge_ratio", report.raw_charge.value(),
                     known ? report.compressed_charge.value() : 0U);
        writer.boolean("minimum_unit_saved",
                       known && saves_physical_unit(report.raw_charge, report.compressed_charge,
                                                    report.probe.minimum));
    }
    writer.number("compression_output_bound",
                  report.admission.compression.requirements.output_bound.value());
    writer.number("bound_output_physical_charge",
                  report.admission.compression.output_charge.value());
    writer.number("compaction_admission_bound", report.admission.compression.output_charge.value());
    writer.number("stored_compressed_bytes", report.stored_bytes.value());
    writer.number("compressed_physical_charge", report.compressed_charge.value());
    writer.number("compression_workspace_bytes",
                  report.admission.compression.workspace_charge.value());
    writer.number("decompression_workspace_bytes",
                  report.admission.decompression.workspace_charge.value());
    writer.number("metadata_bytes_per_workspace", sizeof(GpuBatchMetadata));
    writer.number("peak_admitted_gpu_bytes", report.admission.peak.value());
    writer.number("maximum_physical_bytes", report.admission.physical_cap.value());
    writer.boolean("admission_proven", report.admission.proven);
    writer.boolean("source_verified", report.source_verified);
    writer.boolean("compression_enqueue", report.codec.compression_enqueue);
    writer.boolean("compression_completion", report.codec.compression_completion);
    writer.number("stored_crc_expected", report.codec.stored_crc_expected);
    writer.number("stored_crc_actual", report.codec.stored_crc_actual);
    writer.boolean("decompression_enqueue", report.codec.decompression_enqueue);
    writer.boolean("decompression_completion", report.codec.decompression_completion);
    writer.number("logical_crc_expected", report.logical_crc_expected);
    writer.number("logical_crc_actual", report.logical_crc_actual);
    writer.boolean("size_equal", report.size_equal);
    writer.boolean("byte_equal", report.byte_equal);
    writer.boolean("compaction_exercised", report.compaction_exercised);
    writer.boolean("cleanup_workspace_and_metadata", report.cleanup_workspace);
    writer.boolean("cleanup_bound_output_and_compaction", report.cleanup_bound_output);
    writer.boolean("cleanup_compressed", report.cleanup_compressed);
    writer.boolean("cleanup_raw", report.cleanup_raw);
    writer.boolean("cleanup_va", report.cleanup_va);
    writer.boolean("cleanup_stream", report.cleanup_stream);
    writer.boolean("cleanup_context", report.cleanup_context);
    writer.boolean("final_counts_known", report.final_counts_known);
    writer.nullable_number("final_backend_resources", report.final_resources,
                           report.final_counts_known);
    writer.nullable_number("final_codec_workspace_allocations", report.final_workspace,
                           report.final_counts_known);
    writer.nullable_number("final_metadata_allocations", report.final_metadata,
                           report.final_counts_known);
    writer.nullable_number("final_bound_output_backing", report.final_bound_output,
                           report.final_counts_known);
    writer.nullable_number("final_compaction_staging", report.final_compaction,
                           report.final_counts_known);
    writer.nullable_number("final_raw_resources", report.final_raw, report.final_counts_known);
    writer.nullable_number("final_compressed_backing", report.final_compressed,
                           report.final_counts_known);
    writer.nullable_number("final_va_reservations", report.final_va, report.final_counts_known);
    writer.nullable_number("final_budget_charge", report.final_budget.value(),
                           report.final_counts_known);
    writer.nullable_number("final_stream_resources", report.codec.stream_owned ? 1U : 0U,
                           report.final_counts_known);
    writer.nullable_number("final_retained_contexts", report.codec.context_owned ? 1U : 0U,
                           report.final_counts_known);
    if (report.has_error) {
        writer.error(report.error);
        writer.text({"runtime_compatibility_failure", cuda_compatibility_failure(report.error)});
        writer.number("native_error_domain",
                      static_cast<std::uint64_t>(report.error.native_domain));
    } else {
        writer.null("error");
        writer.null("runtime_compatibility_failure");
    }
    writer.text(
        {"hardware_validation",
         physical_execution && valid_source && report.result == CompressionSmokeResult::passed
             ? (savings ? "PHYSICAL_SAVINGS_VALIDATED" : "COMPRESSION_SMOKE_VALIDATED")
             : "NOT_TESTED"});
    writer.text({"gate_g", "NOT_RUN"});
    return writer.finish();
}
} // namespace

FormatResult format_compression_smoke_report(const CompressionSmokeReport& report,
                                             std::string_view source_sha256,
                                             bool physical_execution,
                                             std::span<char> output) noexcept {
    return format_report(report, source_sha256, physical_execution, output, false);
}
FormatResult format_physical_savings_smoke_report(const CompressionSmokeReport& report,
                                                  std::string_view source_sha256,
                                                  bool physical_execution,
                                                  std::span<char> output) noexcept {
    return format_report(report, source_sha256, physical_execution, output, true);
}
Result<PhysicalSavingsSmokeOptions>
parse_physical_savings_smoke_arguments(std::span<const std::string_view> arguments) noexcept {
    PhysicalSavingsSmokeOptions options{};
    const auto invalid = make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    if (arguments.size() == 1U && arguments.front() == "--help") {
        return options;
    }
    bool device{};
    for (std::size_t i = 0U; i < arguments.size(); ++i) {
        if (arguments[i] == "--run-physical-savings-smoke" && !options.acknowledged) {
            options.acknowledged = true;
        } else if (arguments[i] == "--device" && !device && i + 1U < arguments.size() &&
                   arguments[i + 1U] == "0") {
            device = true;
            ++i;
        } else {
            return invalid;
        }
    }
    // A future physical run explicitly selects the single reviewed device.
    if (options.acknowledged && !device) {
        return invalid;
    }
    return options;
}
} // namespace vramz::detail
