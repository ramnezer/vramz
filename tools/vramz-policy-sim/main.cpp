#include "vramz/detail/simulation.hpp"

#include <charconv>
#include <cstdint>
#include <iostream>
#include <span>
#include <string_view>

using namespace vramz;

namespace {

void print(const simulation::Scenario& scenario, const simulation::ScenarioResult& result) {
    const auto& policy = result.stats.policy;
    const auto charge = policy.gpu_raw_charge.value() + policy.gpu_compressed_charge.value();
    const auto logical =
        policy.logical_gpu_resident_bytes.value() + policy.logical_host_bytes.value();
    const auto ratio = [budget = scenario.gpu_budget.value()](std::uint64_t bytes) {
        return budget == 0U ? 0.0 : static_cast<double>(bytes) / static_cast<double>(budget);
    };
    std::cout
        << "{\"dataset\":\"" << simulation::name(scenario.dataset) << "\",\"trace\":\""
        << simulation::name(scenario.trace) << "\",\"mode\":\""
        << (scenario.mode == PolicyMode::gpu_resident ? "gpu"
            : scenario.mode == PolicyMode::disabled   ? "disabled"
                                                      : "fallback")
        << "\",\"strategy\":\""
        << (scenario.strategy == PolicyStrategy::adaptive ? "adaptive" : "naive")
        << "\",\"seed\":" << scenario.seed << ",\"steps\":" << scenario.steps
        << ",\"target_percent\":" << scenario.target_percent
        << ",\"gpu_hard_budget\":" << scenario.gpu_budget.value()
        << ",\"requested_logical_bytes\":" << result.requested_logical.value()
        << ",\"allocated_real_payload_bytes\":" << result.allocated_logical.value()
        << ",\"gpu_raw_charge\":" << policy.gpu_raw_charge.value()
        << ",\"gpu_compressed_charge\":" << policy.gpu_compressed_charge.value()
        << ",\"gpu_committed_charge\":" << charge
        << ",\"gpu_peak_charged_including_transients\":" << result.stats.gpu.peak_charged.value()
        << ",\"gpu_cleanup_debt\":" << result.stats.gpu.cleanup_debt.value()
        << ",\"gpu_resident_logical_bytes\":" << policy.logical_gpu_resident_bytes.value()
        << ",\"host_logical_bytes\":" << policy.logical_host_bytes.value()
        << ",\"host_charge\":" << policy.host_charge.value()
        << ",\"gpu_effective_ratio_actual_committed\":" << policy.gpu_effective_ratio()
        << ",\"gpu_budget_working_set_ratio\":" << ratio(policy.logical_gpu_resident_bytes.value())
        << ",\"total_working_set_ratio_including_host\":" << ratio(logical)
        << ",\"all_tier_physical_compression_ratio\":"
        << result.stats.compression.physical_compression_ratio()
        << ",\"allocation_success\":" << result.allocation_success
        << ",\"failed_accesses\":" << result.failed_accesses
        << ",\"trace_accesses\":" << result.trace_accesses
        << ",\"failed_pressure_targets\":" << result.failed_pressure_targets
        << ",\"integrity_ok\":" << result.integrity_ok
        << ",\"accounting_ok\":" << result.accounting_ok << ",\"cleanup_ok\":" << result.cleanup_ok
        << ",\"gpu_resident_capacity_pass\":" << result.capacity_pass
        << ",\"policy_cycles\":" << policy.policy_cycles
        << ",\"policy_transitions\":" << policy.policy_transitions
        << ",\"decision_digest\":" << policy.decision_digest
        << ",\"compression_attempts\":" << policy.compression_attempts
        << ",\"compression_successes\":" << policy.compression_successes
        << ",\"compression_rejected\":" << policy.compression_rejected
        << ",\"compression_skipped_incompressible\":" << policy.compression_skipped_incompressible
        << ",\"stale_proposal_rejections\":" << policy.stale_proposal_rejections
        << ",\"restore_count\":" << policy.restore_count
        << ",\"thrash_events\":" << policy.thrash_events
        << ",\"host_fallback_events\":" << policy.host_fallback_count
        << ",\"pcie_transition_count_simulated\":" << policy.pcie_transition_count
        << ",\"simulated_compression_cost\":" << policy.simulated_compression_cost
        << ",\"simulated_restore_cost\":" << policy.simulated_restore_cost
        << ",\"simulated_host_cost\":" << policy.simulated_host_cost
        << ",\"hot_raw_observations\":" << result.hot_raw_observations
        << ",\"hot_observations\":" << result.hot_observations
        << ",\"scale_factor\":" << scenario.scale_factor
        << ",\"scaled_gpu_hard_budget\":" << result.scaled.gpu_budget.value()
        << ",\"scaled_requested_logical_bytes\":" << result.scaled.requested_logical.value()
        << ",\"scaled_gpu_resident_logical_bytes\":" << result.scaled.gpu_resident_logical.value()
        << ",\"scaled_gpu_committed\":" << result.scaled.gpu_committed.value()
        << ",\"scaled_gpu_peak_charged\":" << result.scaled.gpu_peak_charged.value()
        << ",\"hardware_performance_target\":\"NOT VALIDATED\"}\n";
}

[[nodiscard]] bool run_and_print(const simulation::Scenario& scenario) {
    const auto result = simulation::run(scenario);
    if (!result) {
        std::cerr << "scenario_error code=" << static_cast<unsigned int>(result.error().code)
                  << '\n';
        return false;
    }
    print(scenario, result.value());
    // Honest capacity failure is a research result, not a broken simulator invocation.
    return result.value().integrity_ok && result.value().accounting_ok && result.value().cleanup_ok;
}

[[nodiscard]] bool matrix(simulation::Scenario base) {
    bool healthy = true;
    for (const auto dataset : simulation::datasets) {
        for (const auto ratio : simulation::expansion_percent) {
            base.dataset = dataset;
            base.target_percent = ratio;
            base.trace = simulation::Trace::streaming;
            healthy = run_and_print(base) && healthy;
        }
    }
    // Fixed generic trace/naive/fallback comparisons, including honest failed targets.
    base.target_percent = 200U;
    for (const auto dataset : {simulation::Dataset::repeated, simulation::Dataset::mixed}) {
        base.dataset = dataset;
        for (const auto trace : simulation::traces) {
            base.trace = trace;
            for (const auto strategy :
                 {PolicyStrategy::adaptive, PolicyStrategy::oldest_unpinned}) {
                base.strategy = strategy;
                healthy = run_and_print(base) && healthy;
            }
        }
    }
    // A 1.0x logical cyclic workload is slightly larger than normal RAW admission,
    // since the protected migration reserve remains inside the physical budget.
    base.target_percent = 100U;
    base.trace = simulation::Trace::cyclic;
    for (const auto dataset : {simulation::Dataset::repeated, simulation::Dataset::mixed}) {
        base.dataset = dataset;
        for (const auto strategy : {PolicyStrategy::adaptive, PolicyStrategy::oldest_unpinned}) {
            base.strategy = strategy;
            healthy = run_and_print(base) && healthy;
        }
    }
    base.target_percent = 200U;
    base.strategy = PolicyStrategy::adaptive;
    base.dataset = simulation::Dataset::random;
    base.trace = simulation::Trace::streaming;
    for (const auto mode :
         {PolicyMode::gpu_resident, PolicyMode::gpu_resident_with_host_fallback}) {
        base.mode = mode;
        healthy = run_and_print(base) && healthy;
    }
    return healthy;
}

} // namespace

int main(int argc, char* argv[]) {
    simulation::Scenario scenario{};
    bool full_matrix = false;
    const std::span<char*> arguments{argv, static_cast<std::size_t>(argc)};
    for (std::size_t index = 1U; index < arguments.size(); ++index) {
        const std::string_view option{arguments[index]};
        if (option == "--matrix") {
            full_matrix = true;
            continue;
        }
        if (option == "--help") {
            std::cout << "vramz-policy-sim [--matrix] [--dataset "
                         "zeros|repeated|sparse|integers|fp-like|mixed|random|encoded-like]\n"
                         " [--trace hot-cold|streaming|cyclic|random|bursty|graphics|layers] "
                         "[--target-percent 100..300]\n"
                         " [--mode gpu|fallback|disabled] [--strategy adaptive|naive] [--steps "
                         "0..10000]\n"
                         " [--gpu-budget bytes] [--scale-factor positive_integer]\n"
                         "JSONL: real-byte CPU/mock execution plus explicitly scaled accounting, "
                         "no GPU performance measurement.\n";
            return 0;
        }
        if (index + 1U == arguments.size()) {
            std::cerr << "Missing option value\n";
            return 2;
        }
        ++index;
        const std::string_view value{arguments[index]};
        bool recognized = false;
        if (option == "--dataset") {
            for (const auto dataset : simulation::datasets) {
                if (value == simulation::name(dataset)) {
                    scenario.dataset = dataset;
                    recognized = true;
                }
            }
        } else if (option == "--trace") {
            for (const auto trace : simulation::traces) {
                if (value == simulation::name(trace)) {
                    scenario.trace = trace;
                    recognized = true;
                }
            }
        } else if (option == "--mode") {
            recognized = value == "gpu" || value == "fallback" || value == "disabled";
            scenario.mode = value == "gpu"        ? PolicyMode::gpu_resident
                            : value == "disabled" ? PolicyMode::disabled
                                                  : PolicyMode::gpu_resident_with_host_fallback;
        } else if (option == "--strategy") {
            recognized = value == "adaptive" || value == "naive";
            scenario.strategy =
                value == "adaptive" ? PolicyStrategy::adaptive : PolicyStrategy::oldest_unpinned;
        } else {
            std::uint64_t number = 0U;
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
            if (parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size()) {
                if (option == "--target-percent" && number >= 100U && number <= 300U) {
                    scenario.target_percent = static_cast<std::uint32_t>(number);
                    recognized = true;
                } else if (option == "--steps" && number <= 10000U) {
                    scenario.steps = static_cast<std::uint32_t>(number);
                    recognized = true;
                } else if (option == "--gpu-budget") {
                    scenario.gpu_budget = ByteSize{number};
                    recognized = true;
                } else if (option == "--scale-factor" && number != 0U) {
                    scenario.scale_factor = number;
                    recognized = true;
                }
            }
        }
        if (!recognized) {
            std::cerr << "Invalid option/value: " << option << '\n';
            return 2;
        }
    }
    return (full_matrix ? matrix(scenario) : run_and_print(scenario)) ? 0 : 1;
}
