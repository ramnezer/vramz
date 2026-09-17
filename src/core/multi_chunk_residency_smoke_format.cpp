#include "vramz/detail/multi_chunk_residency_smoke.hpp"
#include "vramz/detail/smoke_json.hpp"

#include <algorithm>

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
void format_chunk(SmokeJsonWriter& writer, const MultiChunkResidencyChunkReport& chunk) noexcept {
    writer.number("chunk_index", chunk.chunk_index);
    writer.number("logical_bytes", chunk.logical_bytes.value());
    writer.number("logical_crc_expected", chunk.logical_crc_expected);
    writer.number("raw_physical_charge", chunk.raw_charge.value());
    writer.number("stored_compressed_bytes", chunk.stored_bytes.value());
    writer.number("compressed_physical_charge", chunk.compressed_charge.value());
    writer.difference("physical_bytes_saved", chunk.raw_charge.value(),
                      chunk.compressed_charge.value(), chunk.charges_corroborated);
    writer.ratio("stored_compression_ratio", chunk.logical_bytes.value(),
                 chunk.charges_corroborated ? chunk.stored_bytes.value() : 0U);
    writer.boolean("source_verified", chunk.source_verified);
    writer.boolean("charges_corroborated", chunk.charges_corroborated);
    writer.boolean("minimum_unit_saved", chunk.minimum_unit_saved);
    writer.number("stored_crc_expected", chunk.stored_crc_expected);
    writer.number("stored_crc_actual", chunk.stored_crc_actual);
    writer.boolean("compression_enqueue", chunk.compression_enqueue);
    writer.boolean("compression_completion", chunk.compression_completion);
    writer.boolean("decompression_enqueue", chunk.decompression_enqueue);
    writer.boolean("decompression_completion", chunk.decompression_completion);
    writer.number("restore_crc", chunk.logical_crc_actual);
    writer.boolean("size_equal", chunk.size_equal);
    writer.boolean("byte_equal", chunk.byte_equal);
    writer.boolean("compaction_exercised", chunk.compaction_exercised);
    writer.boolean("compressed_at_snapshot", chunk.compressed_at_snapshot);
    writer.boolean("round_trip_verified", chunk.round_trip_verified);
    writer.boolean("cleanup_complete", chunk.cleanup_complete);
}
} // namespace

Result<MultiChunkResidencySmokeOptions>
parse_multi_chunk_residency_smoke_arguments(std::span<const std::string_view> arguments) noexcept {
    MultiChunkResidencySmokeOptions options{};
    const auto invalid = make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    if (arguments.size() == 1U && arguments.front() == "--help") {
        return options;
    }
    if (arguments.size() > 3U) {
        return invalid;
    }
    bool device{};
    for (std::size_t i = 0U; i < arguments.size(); ++i) {
        if (arguments[i] == "--run-multi-chunk-residency-smoke" && !options.acknowledged) {
            options.acknowledged = true;
        } else if (arguments[i] == "--device" && !device && i + 1U < arguments.size() &&
                   arguments[i + 1U] == "0") {
            device = true;
            ++i;
        } else {
            return invalid;
        }
    }
    if (options.acknowledged && !device) {
        return invalid;
    }
    return options;
}

FormatResult format_multi_chunk_residency_smoke_report(const MultiChunkResidencySmokeReport& report,
                                                       std::string_view source_sha256,
                                                       bool physical_execution,
                                                       std::span<char> output) noexcept {
    SmokeJsonWriter writer{output};
    writer.number("m10_version", 1U);
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
    writer.number("chunk_count", m10_chunk_count);
    writer.number("logical_bytes_per_chunk", m10_logical_payload.value());
    writer.number("aggregate_logical_bytes", report.totals.logical_bytes.value());
    writer.object_array("chunks", report.chunks, format_chunk);
    writer.boolean("simultaneous_compressed_residency_proven",
                   report.simultaneous_compressed_residency_proven);
    const bool snapshot = report.simultaneous_compressed_residency_proven;
    writer.nullable_number("aggregate_raw_baseline_charge",
                           report.totals.raw_baseline_charge.value(), snapshot);
    writer.nullable_number("aggregate_compressed_physical_charge",
                           report.totals.compressed_physical_charge.value(), snapshot);
    writer.nullable_number("aggregate_physical_bytes_saved",
                           report.totals.physical_bytes_saved.value(), snapshot);
    writer.ratio("aggregate_physical_charge_ratio", report.totals.raw_baseline_charge.value(),
                 snapshot ? report.totals.compressed_physical_charge.value() : 0U);
    writer.number("peak_admitted_gpu_bytes", report.peak_admitted_gpu_bytes.value());
    writer.number("maximum_physical_bytes", report.maximum_physical_bytes.value());
    writer.boolean("admission_proven", report.admission.proven);
    writer.number("compression_output_bound",
                  report.admission.compression.requirements.output_bound.value());
    writer.number("compression_workspace_bytes",
                  report.admission.compression.workspace_charge.value());
    writer.number("decompression_workspace_bytes",
                  report.admission.decompression.workspace_charge.value());
    writer.number("metadata_bytes_per_workspace", sizeof(GpuBatchMetadata));
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
    writer.nullable_number("final_unmaterialized_reservations",
                           report.final_unmaterialized_reservations.value(),
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
    writer.text({"hardware_validation", physical_execution && valid_source && snapshot &&
                                                report.result == CompressionSmokeResult::passed
                                            ? "MULTI_CHUNK_RESIDENCY_VALIDATED"
                                            : "NOT_TESTED"});
    writer.text({"gate_g", "NOT_RUN"});
    return writer.finish();
}
} // namespace vramz::detail
