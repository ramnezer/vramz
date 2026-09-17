#include "../common/workload.hpp"
#include "vramz/cuda_runtime.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string_view>
#include <system_error>
#include <unistd.h>

#ifndef VRAMZ_V1_SOURCE_SHA256
#define VRAMZ_V1_SOURCE_SHA256 "unfrozen-development"
#endif

namespace {
using namespace vramz;
constinit example::WorkloadReport report{};
[[noreturn]] void fatal() noexcept {
    constexpr std::string_view text{"{\"result\":\"FAIL_STOP\",\"final_counts_known\":false,"
                                    "\"error\":\"ambiguous_or_fatal_runtime_state\"}\n"};
    const auto written = ::write(STDOUT_FILENO, text.data(), text.size());
    std::_Exit(written == static_cast<ssize_t>(text.size()) ? 86 : 87);
}
[[nodiscard]] bool number(std::string_view text, std::uint64_t& value) noexcept {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}
int error(Error value) {
    std::cout << "{\"result\":\"FAIL\",\"error_code\":" << static_cast<unsigned>(value.code)
              << ",\"operation\":" << static_cast<unsigned>(value.operation)
              << ",\"native_code\":" << value.native_code << "}\n";
    return 1;
}
} // namespace

int main(int argc, char* argv[]) {
    std::set_terminate(fatal);
    vramz::CudaRuntimeOptions options{};
    vramz::example::WorkloadConfig config{};
    std::string_view command;
    bool device_ack{};
    std::string_view profile;
    std::uint32_t iterations{7U};
    std::uint32_t warmups{2U};
    bool raw_baseline{};
    std::array<std::string_view, 16U> seen{};
    std::size_t seen_count{};
    for (int i = 1; i < argc; ++i) {
        const std::string_view key{argv[i]};
        if (key == "--help") {
            std::cout << "VRAMZ experimental GPU-resident runtime\n"
                         "Usage: vramz-run --device 0 --driver FILE --cudart FILE --nvcomp FILE\n"
                         "  --run info|self-test|demo|high|moderate|random|mixed|p75|p50|p25\n"
                         "  or --run benchmark|soak --profile high|moderate|random|mixed\n"
                         "  [--iterations 1..128] [--warmups 0..8] [--raw-baseline 0|1]\n"
                         "  [--buffers 4..32] [--buffer-mib N] [--chunk-mib 2..16] [--hard-mib N]\n"
                         "All execution is explicit; maximum physical/staging charge is 512 MiB.\n";
            return 0;
        }
        if (seen_count == seen.size() || std::ranges::find(seen, key) != seen.end()) {
            return 2;
        }
        seen[seen_count++] = key;
        if (++i >= argc) {
            return 2;
        }
        const std::string_view value{argv[i]};
        if (key == "--device") {
            if (value != "0" || device_ack) {
                return 2;
            }
            device_ack = true;
        } else if (key == "--driver") {
            options.driver_path = value;
        } else if (key == "--cudart") {
            options.cudart_path = value;
        } else if (key == "--nvcomp") {
            options.nvcomp_path = value;
        } else if (key == "--profile") {
            profile = value;
        } else if (key == "--run") {
            if (!command.empty()) {
                return 2;
            }
            command = value;
        } else {
            std::uint64_t count{};
            if (!number(value, count) || count > 512U) {
                return 2;
            }
            if (key == "--iterations" && count >= 1U && count <= 128U) {
                iterations = static_cast<std::uint32_t>(count);
            } else if (key == "--warmups" && count <= 8U) {
                warmups = static_cast<std::uint32_t>(count);
            } else if (key == "--raw-baseline" && count <= 1U) {
                raw_baseline = count == 1U;
            } else if (key == "--buffers") {
                config.buffers = static_cast<std::uint32_t>(count);
            } else if (key == "--buffer-mib") {
                config.bytes_per_buffer = vramz::ByteSize{count * 1048576U};
            } else if (key == "--chunk-mib") {
                config.chunk_size = vramz::ByteSize{count * 1048576U};
            } else if (key == "--hard-mib") {
                config.hard_limit = vramz::ByteSize{count * 1048576U};
            } else {
                return 2;
            }
        }
    }
    if (!device_ack || command.empty() || options.driver_path.empty() ||
        options.cudart_path.empty() || options.nvcomp_path.empty()) {
        std::cerr << "Explicit --run, --device 0 and provider paths are required. See --help.\n";
        return 2;
    }
    const bool benchmark = command == "benchmark";
    const bool soak = command == "soak";
    if ((!benchmark && !soak) && (!profile.empty() || raw_baseline ||
                                  std::ranges::find(seen, "--iterations") != seen.end() ||
                                  std::ranges::find(seen, "--warmups") != seen.end())) {
        return 2;
    }
    if ((benchmark || soak) && profile.empty()) {
        return 2;
    }
    config.measure = benchmark;
    config.keep_runtime = benchmark || soak;
    config.raw_baseline = raw_baseline;
    const auto selected = benchmark || soak ? profile : command;
    using Profile = vramz::example::Profile;
    if (selected == "high") {
        config.profile = Profile::high;
    } else if (selected == "moderate") {
        config.profile = Profile::moderate;
    } else if (selected == "random") {
        config.profile = Profile::random;
    } else if (selected == "p75") {
        config.profile = Profile::p75;
    } else if (selected == "p50") {
        config.profile = Profile::p50;
    } else if (selected == "p25") {
        config.profile = Profile::p25;
    } else if (selected == "concurrency") {
        config.profile = Profile::high;
        config.concurrent_reads = true;
    } else if (selected != "mixed" && selected != "self-test" && selected != "demo" &&
               selected != "info") {
        return 2;
    }
    const auto settings = vramz::example::runtime_config(config);
    if (!settings) {
        return error(settings.error());
    }
    constexpr std::string_view source{VRAMZ_V1_SOURCE_SHA256};
    if (source.size() != 64U || source.find_first_not_of('0') == std::string_view::npos ||
        !std::ranges::all_of(
            source, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); })) {
        std::cerr << "Physical tools require a frozen source SHA256 at build time.\n";
        return 2;
    }
    options.allow_execution = true;
    vramz::CudaRuntimeInfo info{};
    auto runtime = vramz::Runtime::create_cuda(settings.value(), options, info);
    if (!runtime) {
        return error(runtime.error());
    }
    std::cout << std::boolalpha << "{\"kind\":\"capabilities\",\"source_sha256\":\""
              << VRAMZ_V1_SOURCE_SHA256 << "\",\"device_ordinal\":0,\"device_name\":\""
              << info.device_name.data() << "\",\"driver_release\":\"" << info.driver_release.data()
              << "\",\"driver_api\":" << info.driver_api << ",\"runtime_api\":" << info.runtime_api
              << ",\"nvcomp_version\":\"" << info.nvcomp_version << "\",\"uva\":" << info.uva
              << ",\"vmm\":" << info.vmm
              << ",\"minimum_granularity\":" << info.minimum_granularity.value()
              << ",\"recommended_granularity\":" << info.recommended_granularity.value()
              << ",\"gpu_resident_compression\":true,\"policy\":true,\"host_fallback\":false"
              << ",\"capability_scope\":\"initialized_reference_provider_and_stream\""
              << ",\"universal_gpu_validation\":false,\"physical_savings_requires_workload_"
                 "validation\":true"
              << ",\"external_async_completion\":"
              << runtime.value().capabilities().external_async_completion
              << ",\"compatibility\":\"MINOR_COMPATIBILITY_CANDIDATE\"}\n";
    if (command == "info") {
        const auto closed = runtime.value().shutdown();
        if (!closed) {
            return error(closed.error());
        }
        const auto counts = runtime.value().resources();
        if (!counts || !counts.value().accounting_conserved ||
            counts.value().backend_resources != 0U || counts.value().va_reservations != 0U ||
            counts.value().workspace_resources != 0U ||
            counts.value().ledger_gpu_charge != vramz::ByteSize{} ||
            counts.value().owned_gpu_charge != vramz::ByteSize{} ||
            counts.value().unmaterialized_reservations != vramz::ByteSize{} ||
            counts.value().retained_context || counts.value().stream_owned) {
            return 1;
        }
        std::cout
            << "{\"kind\":\"info_summary\",\"result\":\"PASS\",\"cleanup\":true,\"final_counts_"
               "known\":true,"
               "\"final_backend_resources\":0,\"final_va_reservations\":0,\"final_workspace\":0,"
               "\"final_budget_charge\":0,\"final_owned_charge\":0,\"final_unmaterialized_"
               "reservations\":0,"
               "\"final_context\":false,\"final_stream\":false,\"physical_workload_validation\":"
               "\"NOT_RUN_BY_INFO\"}\n";
        return 0;
    }
    if (!config.keep_runtime) {
        vramz::example::run_workload(runtime.value(), config, report);
        vramz::example::print_report(std::cout, report, true);
        return report.passed ? 0 : 1;
    }
    std::uint64_t successes{};
    std::uint64_t attempts{};
    for (std::uint32_t i = 0U; i < iterations + warmups; ++i) {
        auto current = config;
        if (soak) {
            current.seed += i;
        }
        vramz::example::run_workload(runtime.value(), current, report);
        report.iteration = i;
        report.warmup = i < warmups;
        vramz::example::print_report(std::cout, report, true);
        std::cout.flush();
        if (!report.passed) {
            return 1;
        }
        for (std::uint32_t j = 0U; j < report.cycle_count; ++j) {
            successes += report.cycles[j].after.compression_successes -
                         report.cycles[j].before.compression_successes;
            attempts += report.cycles[j].after.compression_attempts -
                        report.cycles[j].before.compression_attempts;
        }
        std::uint64_t total_pages{}, resident_pages{};
        std::ifstream memory{"/proc/self/statm"};
        const auto page = ::sysconf(_SC_PAGESIZE);
        if (!(memory >> total_pages >> resident_pages) || page <= 0 ||
            resident_pages > UINT64_MAX / static_cast<std::uint64_t>(page)) {
            return 1;
        }
        std::cout
            << "{\"kind\":\"checkpoint\",\"iteration\":" << i
            << ",\"rss_bytes\":" << resident_pages * static_cast<std::uint64_t>(page)
            << ",\"live_gpu_allocations\":0,\"ledger_charge\":0,\"contexts\":1,\"streams\":1}\n";
    }
    const auto stopped = runtime.value().shutdown();
    if (!stopped) {
        return error(stopped.error());
    }
    const auto resources = runtime.value().resources();
    if (!resources) {
        return error(resources.error());
    }
    const auto& counts = resources.value();
    const bool clean = counts.accounting_conserved && counts.backend_resources == 0U &&
                       counts.va_reservations == 0U && counts.workspace_resources == 0U &&
                       counts.ledger_gpu_charge == vramz::ByteSize{} &&
                       counts.owned_gpu_charge == vramz::ByteSize{} &&
                       counts.unmaterialized_reservations == vramz::ByteSize{} &&
                       !counts.retained_context && !counts.stream_owned;
    std::cout << "{\"kind\":\"series_summary\",\"result\":\"" << (clean ? "PASS" : "FAIL")
              << "\",\"cleanup\":" << clean
              << ",\"final_counts_known\":true,\"iterations\":" << iterations
              << ",\"warmups\":" << warmups << ",\"compressions\":" << successes
              << ",\"attempts\":" << attempts
              << ",\"aggregate_logical_bytes\":" << report.logical.value()
              << ",\"settled_physical_charge\":" << report.settled_charge.value()
              << ",\"peak_admitted_gpu_bytes\":" << runtime.value().stats().gpu.peak_charged.value()
              << ",\"maximum_physical_bytes\":" << config.hard_limit.value()
              << ",\"final_backend_resources\":" << counts.backend_resources
              << ",\"final_raw_resources\":" << counts.raw_resources
              << ",\"final_compressed_resources\":" << counts.compressed_resources
              << ",\"final_va_reservations\":" << counts.va_reservations
              << ",\"final_budget_charge\":" << counts.ledger_gpu_charge.value()
              << ",\"final_owned_charge\":" << counts.owned_gpu_charge.value()
              << ",\"final_unmaterialized_reservations\":"
              << counts.unmaterialized_reservations.value()
              << ",\"final_workspace\":" << counts.workspace_resources
              << ",\"final_context\":" << counts.retained_context
              << ",\"final_stream\":" << counts.stream_owned << "}\n";
    return clean ? 0 : 1;
}
