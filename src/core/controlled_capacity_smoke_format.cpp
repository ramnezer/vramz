#include "vramz/detail/controlled_capacity_smoke.hpp"
#include "vramz/detail/smoke_json.hpp"

#include <algorithm>

namespace vramz::detail {
namespace {
constexpr std::string_view name(CompressionSmokeResult result) noexcept {
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
constexpr std::string_view name(Temperature value) noexcept {
    switch (value) {
    case Temperature::hot:
        return "HOT";
    case Temperature::warm:
        return "WARM";
    case Temperature::cold:
        return "COLD";
    }
    return "UNKNOWN";
}
constexpr std::string_view name(Pressure value) noexcept {
    switch (value) {
    case Pressure::normal:
        return "NORMAL";
    case Pressure::soft:
        return "SOFT";
    case Pressure::hard:
        return "HARD";
    case Pressure::critical:
        return "CRITICAL";
    }
    return "UNKNOWN";
}
constexpr std::string_view name(RepresentationState value) noexcept {
    switch (value) {
    case RepresentationState::gpu_raw:
        return "GPU_RAW";
    case RepresentationState::gpu_compressed:
        return "GPU_COMPRESSED";
    case RepresentationState::host_raw:
        return "HOST_RAW";
    case RepresentationState::host_compressed:
        return "HOST_COMPRESSED";
    }
    return "UNKNOWN";
}
void usage(SmokeJsonWriter& writer, const TierUsage& value) noexcept {
    const auto charged = policy_pressure_charge(value);
    writer.nullable_number("charged", charged ? charged.value().value() : 0U,
                           static_cast<bool>(charged));
    writer.number("committed", value.committed.value());
    writer.number("reserved", value.reserved.value());
    writer.number("staging", value.staging.value());
    writer.number("workspace", value.workspace.value());
    writer.number("cleanup_debt", value.cleanup_debt.value());
    writer.number("peak_charged", value.peak_charged.value());
}
void chunk(SmokeJsonWriter& writer, const PolicyPressureChunkReport& record) noexcept {
    const auto& c = record.integrity;
    writer.number("chunk_index", c.chunk_index);
    writer.number("logical_crc", c.logical_crc_expected);
    writer.number("access_epoch", record.access.last_access_epoch);
    writer.number("access_count", record.access.access_count);
    writer.number("recent_frequency", record.access.recent_frequency);
    writer.number("access_frequency", record.effective_frequency);
    writer.number("access_revision", record.access_revision);
    writer.text({"temperature", name(record.temperature)});
    writer.text({"initial_representation", name(record.initial)});
    writer.number("policy_selected_count", record.selected_count);
    writer.number("raw_physical_charge", c.raw_charge.value());
    writer.number("stored_compressed_bytes", c.stored_bytes.value());
    writer.number("compressed_physical_charge", c.compressed_charge.value());
    writer.number("physical_bytes_saved", c.physical_bytes_saved.value());
    writer.boolean("charges_corroborated", c.charges_corroborated);
    writer.boolean("minimum_unit_saved", c.minimum_unit_saved);
    writer.number("stored_crc_expected", c.stored_crc_expected);
    writer.number("stored_crc_actual", c.stored_crc_actual);
    writer.boolean("compression_enqueue", c.compression_enqueue);
    writer.boolean("compression_completion", c.compression_completion);
    writer.boolean("decompression_enqueue", c.decompression_enqueue);
    writer.boolean("decompression_completion", c.decompression_completion);
    writer.number("restore_crc", c.logical_crc_actual);
    writer.boolean("source_verified", c.source_verified);
    writer.boolean("size_equal", c.size_equal);
    writer.boolean("byte_equal", c.byte_equal);
    writer.boolean("compaction_exercised", c.compaction_exercised);
    writer.text({"policy_settled_representation", name(record.settled)});
    writer.boolean("integrity_verified", record.integrity_verified);
    writer.boolean("cleanup_complete", c.cleanup_complete);
}
void stage(SmokeJsonWriter& writer, const PolicyPressureStageReport& value) noexcept {
    writer.number("transaction_phase", static_cast<std::uint64_t>(value.phase));
    usage(writer, value.usage);
    writer.number("owned_physical_charge", value.owned_physical_charge.value());
    writer.number("backend_resources", value.backend_resources);
}
void cycle(SmokeJsonWriter& writer, const PolicyPressureCycleReport& c) noexcept {
    writer.number("cycle_id", c.cycle_id);
    writer.text({"pressure_before", name(c.pressure_before)});
    writer.text({"pressure_after", name(c.pressure_after)});
    writer.object_array("ledger_before", std::array{c.before}, usage);
    writer.object_array("ledger_after", std::array{c.after}, usage);
    writer.nullable_number("selected_chunk", c.selected_chunk, c.selected);
    writer.nullable_number("selected_buffer_id", c.proposal.buffer.value(), c.selected);
    writer.nullable_number("selected_access_revision", c.selected_revision, c.selected);
    writer.text({"temperature", name(c.proposal.temperature)});
    writer.text({"representation_before", name(c.representation_before)});
    writer.text({"action", c.selected && c.proposal.action == PolicyAction::compress_gpu
                               ? "COMPRESS_GPU"
                               : "KEEP"});
    writer.boolean("stale_rejection", c.stale_rejection);
    writer.number("compressibility_after", static_cast<std::uint64_t>(c.compressibility_after));
    writer.number("retry_after_epoch", c.retry_after_epoch);
    writer.text({"transition_result", !c.completed
                                          ? "NOT_ATTEMPTED"
                                          : (c.transition_succeeded ? "SUCCEEDED" : "FAILED")});
    writer.number("exact_charge_before", c.charge_before.value());
    writer.number("exact_charge_after", c.charge_after.value());
    writer.number("physical_bytes_reclaimed", c.reclaimed.value());
    writer.object_array(
        "transaction_stages",
        std::span{c.stages}.first(std::min<std::size_t>(c.stage_count, c.stages.size())), stage);
    if (c.has_error) {
        writer.error(c.error);
    } else {
        writer.null("error");
    }
}
} // namespace

Result<ControlledCapacitySmokeOptions>
parse_controlled_capacity_smoke_arguments(std::span<const std::string_view> arguments) noexcept {
    ControlledCapacitySmokeOptions options{};
    const auto invalid = make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    if (arguments.size() == 1U && arguments.front() == "--help") {
        return options;
    }
    if (arguments.size() > 3U) {
        return invalid;
    }
    bool device{};
    for (std::size_t i = 0U; i < arguments.size(); ++i) {
        if (arguments[i] == "--run-controlled-capacity-smoke" && !options.acknowledged) {
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

FormatResult format_controlled_capacity_smoke_report(const ControlledCapacitySmokeReport& report,
                                                     std::string_view source_sha256,
                                                     bool physical_execution,
                                                     std::span<char> output) noexcept {
    SmokeJsonWriter writer{output};
    writer.number("m12_version", 1U);
    writer.text({"result", name(report.result)});
    writer.text({"execution_kind", physical_execution ? "physical" : "fake"});
    const bool valid_source = source_sha256.size() == 64U &&
                              source_sha256.find_first_not_of('0') != std::string_view::npos &&
                              std::ranges::all_of(source_sha256, [](char c) {
                                  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                              });
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
    writer.object_array(
        "approved_providers", report.approved_providers,
        [](SmokeJsonWriter& out, const ControlledCapacitySmokeReport::Provider& provider) noexcept {
            out.text({"role", provider.role});
            out.text({"path", provider.path});
            out.text({"sha256", provider.sha256});
        });
    writer.number("minimum_granularity", report.probe.minimum.value());
    writer.number("recommended_granularity", report.probe.recommended.value());
    writer.number("maximum_physical_bytes", m12_physical_cap.value());
    writer.number("budget_hard_limit", m12_physical_cap.value());
    writer.number("migration_reserve", m12_migration_reserve.value());
    writer.number("normal_admission_limit", m12_normal_limit.value());
    writer.number("maximum_transitions_per_cycle", 1U);
    writer.number("host_budget", 0U);
    writer.number("capacity_reference_bytes", m12_settled_target.value());
    writer.text({"policy_mode", "GPU_RESIDENT"});
    writer.number("pressure_high_threshold", m12_high_threshold.value());
    writer.number("pressure_settled_threshold", m12_settled_target.value());
    writer.number("maximum_policy_cycles", m12_maximum_cycles);
    writer.number("chunk_count", m12_chunk_count);
    writer.number("logical_bytes_per_chunk", m12_logical_payload.value());
    writer.number("configured_aggregate_logical_bytes", m12_aggregate_logical.value());
    writer.number("aggregate_logical_bytes", report.aggregate_logical_bytes.value());
    writer.number("settled_physical_charge", report.aggregate_settled_charge.value());
    writer.ratio("controlled_capacity_ratio", report.aggregate_logical_bytes.value(),
                 report.aggregate_settled_charge.value());
    writer.boolean("controlled_capacity_1_5x_proven", report.controlled_capacity_1_5x_proven);
    writer.number("aggregate_raw_physical_charge", report.aggregate_raw_charge.value());
    writer.number("aggregate_settled_physical_charge", report.aggregate_settled_charge.value());
    writer.number("physical_bytes_reclaimed", report.reclaimed.value());
    writer.object_array("initial_ledger", std::array{report.initial_usage}, usage);
    writer.object_array("settled_ledger", std::array{report.settled_usage}, usage);
    writer.text({"initial_pressure", name(report.initial_pressure)});
    writer.text({"settled_pressure", name(report.settled_pressure)});
    writer.boolean("initial_snapshot_proven", report.initial_snapshot_proven);
    writer.object_array("chunks", report.chunks, chunk);
    writer.object_array("policy_cycles",
                        std::span{report.cycles}.first(
                            std::min<std::size_t>(report.cycle_count, report.cycles.size())),
                        cycle);
    writer.boolean("manual_victim_selection", false);
    writer.boolean("manual_compression_transition", false);
    writer.boolean("hot_chunk_preserved_raw", report.hot_chunk_preserved_raw);
    writer.number("automatic_policy_actions", report.automatic_policy_actions);
    writer.boolean("pressure_target_reached", report.pressure_target_reached);
    writer.boolean("controlled_capacity_snapshot_proven",
                   report.controlled_capacity_snapshot_proven);
    writer.number("peak_admitted_gpu_bytes", report.peak_admitted_gpu_bytes.value());
    writer.number("compression_output_bound", report.admission.compression.output_charge.value());
    writer.number("compression_workspace_bytes",
                  report.admission.compression.workspace_charge.value());
    writer.number("decompression_workspace_bytes",
                  report.admission.decompression.workspace_charge.value());
    writer.boolean("cleanup_raw", report.cleanup_raw);
    writer.boolean("cleanup_compressed", report.cleanup_compressed);
    writer.boolean("cleanup_workspace_and_metadata", report.cleanup_workspace);
    writer.boolean("cleanup_bound_output_and_compaction", report.cleanup_workspace);
    writer.boolean("cleanup_va", report.cleanup_va);
    writer.boolean("cleanup_stream", report.cleanup_stream);
    writer.boolean("cleanup_context", report.cleanup_context);
    writer.boolean("final_counts_known", report.final_counts_known);
    writer.nullable_number("final_backend_resources", report.final_resources,
                           report.final_counts_known);
    writer.nullable_number("final_raw_resources", report.final_raw, report.final_counts_known);
    writer.nullable_number("final_compressed_resources", report.final_compressed,
                           report.final_counts_known);
    writer.nullable_number("final_va_reservations", report.final_va, report.final_counts_known);
    writer.nullable_number("final_codec_workspace", report.final_workspace,
                           report.final_counts_known);
    writer.nullable_number("final_metadata_allocations", 0U,
                           report.final_counts_known && report.final_resources == 0U);
    writer.nullable_number("final_bound_compaction_resources", 0U,
                           report.final_counts_known && report.final_resources == 0U);
    writer.object_array("final_ledger", std::array{report.final_usage}, usage);
    const auto charge = policy_pressure_charge(report.final_usage);
    writer.nullable_number("final_budget_charge", charge ? charge.value().value() : 0U,
                           report.final_counts_known && static_cast<bool>(charge));
    writer.nullable_number("final_stream_resources", report.codec.stream_owned ? 1U : 0U,
                           report.final_counts_known);
    writer.nullable_number("final_retained_contexts", report.codec.context_owned ? 1U : 0U,
                           report.final_counts_known);
    if (report.has_error) {
        writer.error(report.error);
        writer.text({"runtime_compatibility_failure", cuda_compatibility_failure(report.error)});
    } else {
        writer.null("error");
        writer.null("runtime_compatibility_failure");
    }
    writer.text({"hardware_validation", physical_execution && valid_source &&
                                                report.result == CompressionSmokeResult::passed &&
                                                report.controlled_capacity_snapshot_proven &&
                                                report.controlled_capacity_1_5x_proven &&
                                                report.final_counts_known
                                            ? "CONTROLLED_CAPACITY_1_5X_VALIDATED"
                                            : "NOT_TESTED"});
    writer.text({"gate_g", "NOT_RUN"});
    return writer.finish();
}
} // namespace vramz::detail
