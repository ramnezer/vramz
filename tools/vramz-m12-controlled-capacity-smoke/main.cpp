#include "m12_build_config.hpp"
#include "vramz/capabilities.hpp"
#include "vramz/detail/controlled_capacity_smoke.hpp"
#include "vramz/detail/m7_driver_identity.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvcomp/lz4.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <fcntl.h>
#include <link.h>
#include <span>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace {

static_assert(CUDA_VERSION == 13030 && CUDART_VERSION == 13030);

// Fixed process-lifetime report storage also survives an allocation-free fail-stop path.
// Constant initialization does not construct an adapter or invoke any CUDA function.
constinit vramz::detail::ControlledCapacitySmokeReport report{};
constinit bool physical_execution{};

[[nodiscard]] bool write_output(std::span<const char> text) noexcept {
    std::size_t offset{};
    // Bound interrupted/partial diagnostic writes; output progress cannot control cleanup.
    for (std::size_t attempt = 0U; attempt < 8U && offset < text.size(); ++attempt) {
        const auto written = ::write(STDOUT_FILENO, text.data() + offset, text.size() - offset);
        if (written < 0) {
            if (errno != EINTR) {
                return false;
            }
        } else if (written == 0) {
            return false;
        } else {
            const auto count = static_cast<std::size_t>(written);
            if (count > text.size() - offset) {
                return false;
            }
            offset += count;
        }
    }
    return offset == text.size();
}

[[nodiscard]] bool emit_report() noexcept {
    report.approved_providers = {
        {{"driver", vramz::detail::m12_driver.path, vramz::detail::m12_driver.sha256},
         {"nvcomp", vramz::detail::m12_nvcomp.path, vramz::detail::m12_nvcomp.sha256},
         {"cudart", vramz::detail::m12_cudart.path, vramz::detail::m12_cudart.sha256}}};
    std::array<char, 65536U> output{};
    const auto formatted = vramz::detail::format_controlled_capacity_smoke_report(
        report, vramz::detail::m12_source_sha256, physical_execution,
        std::span{output}.first(output.size() - 1U));
    if (formatted.truncated || formatted.written >= output.size()) {
        constexpr std::string_view fallback{
            "{\"m12_version\":1,\"result\":\"FAIL\",\"error\":\"report_format_failure\","
            "\"hardware_validation\":\"NOT_TESTED\"}\n"};
        static_cast<void>(write_output(std::span{fallback.data(), fallback.size()}));
        return false;
    }
    std::size_t written = formatted.written;
    if (written == 0U || output[written - 1U] != '\n') {
        output[written] = '\n';
        ++written;
    }
    return write_output(std::span{output}.first(written));
}

[[noreturn]] void fatal_termination() noexcept {
    vramz::detail::finalize_controlled_capacity_smoke_failure(report);
    // A precise definite-cleanup failure retains its exact report. No destructor, retry,
    // guessed accounting repair or in-flight resource cleanup runs across this boundary.
    static_cast<void>(emit_report());
    std::_Exit(86);
}

[[nodiscard]] bool
loaded_file_is_approved(const void* symbol,
                        const vramz::detail::M12LibraryIdentity& approved) noexcept {
    Dl_info loaded{};
    if (::dladdr(symbol, &loaded) == 0 || loaded.dli_fname == nullptr || approved.path.empty()) {
        return false;
    }
    std::array<char, PATH_MAX> path{};
    if (::realpath(loaded.dli_fname, path.data()) == nullptr ||
        std::string_view{path.data()} != approved.path ||
        approved.path.find("/stubs/") != std::string_view::npos) {
        return false;
    }
    struct stat observed {};
    return ::stat(path.data(), &observed) == 0 &&
           vramz::detail::matches_driver_file_identity(observed, approved.identity);
}

[[nodiscard]] bool build_and_loader_approved() noexcept {
#if defined(VRAMZ_M12_CONTROLLED_CAPACITY_ONLY) && VRAMZ_M12_CONTROLLED_CAPACITY_ONLY == 1
    constexpr bool enabled = true;
#else
    constexpr bool enabled = false;
#endif
    const auto source = vramz::detail::m12_source_sha256;
    if (source.size() != 64U || source.find_first_not_of('0') == std::string_view::npos ||
        !std::ranges::all_of(source, [](char value) {
            return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
        })) {
        return false;
    }
    const auto capabilities = vramz::build_capabilities();
    // Exact command-scoped paths are provided by the future reviewed launcher. A
    // inherited preload/audit mechanism cannot silently replace a verified provider.
    const auto* path = std::getenv("LD_LIBRARY_PATH");
    if (!capabilities.cuda_adapter_compiled || !capabilities.nvcomp_adapter_compiled ||
        std::getenv("LD_PRELOAD") != nullptr || std::getenv("LD_AUDIT") != nullptr ||
        path == nullptr || std::string_view{path} != vramz::detail::m12_loader_path) {
        return false;
    }
    const auto inspect_loaded = [](dl_phdr_info* entry, std::size_t, void*) noexcept -> int {
        if (entry == nullptr || entry->dlpi_name == nullptr) {
            return 1;
        }
        const std::string_view name{entry->dlpi_name};
        const auto filename = name.substr(
            name.find_last_of('/') == std::string_view::npos ? 0U : name.find_last_of('/') + 1U);
        if (!filename.starts_with("libcuda") && !filename.starts_with("libnvcomp") &&
            !filename.starts_with("libnvidia")) {
            return 0;
        }
        std::array<char, PATH_MAX> resolved{};
        if (::realpath(entry->dlpi_name, resolved.data()) == nullptr) {
            return 1;
        }
        const std::string_view canonical{resolved.data()};
        for (const auto& allowed :
             {vramz::detail::m12_driver, vramz::detail::m12_nvcomp, vramz::detail::m12_cudart}) {
            if (canonical == allowed.path) {
                return 0;
            }
        }
        return 1;
    };
    return enabled && ::dl_iterate_phdr(inspect_loaded, nullptr) == 0 &&
           loaded_file_is_approved(reinterpret_cast<const void*>(&cuInit),
                                   vramz::detail::m12_driver) &&
           loaded_file_is_approved(reinterpret_cast<const void*>(&cudaRuntimeGetVersion),
                                   vramz::detail::m12_cudart) &&
           loaded_file_is_approved(reinterpret_cast<const void*>(&nvcompBatchedLZ4CompressAsync),
                                   vramz::detail::m12_nvcomp);
}

[[nodiscard]] vramz::Result<vramz::detail::NvidiaDriverRelease>
installed_driver_release() noexcept {
    const auto invalid =
        vramz::make_error(vramz::ErrorCode::unsupported, vramz::OperationId::runtime_create);
    // Kernel module metadata only: no NVML, device-node access, CUDA query or probe.
    const int descriptor = ::open("/sys/module/nvidia/version", O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        return invalid;
    }
    std::array<char, 64U> buffer{};
    const auto count = ::read(descriptor, buffer.data(), buffer.size());
    char extra{};
    const auto trailing = ::read(descriptor, &extra, 1U);
    const auto closed = ::close(descriptor);
    if (count <= 0 || static_cast<std::size_t>(count) > buffer.size() || trailing != 0 ||
        closed != 0) {
        return invalid;
    }
    const auto parsed = vramz::detail::parse_nvidia_driver_release(
        {buffer.data(), static_cast<std::size_t>(count)});
    if (!parsed || parsed.value() != vramz::detail::m12_expected_driver_release) {
        return invalid;
    }
    return parsed.value();
}

[[nodiscard]] int reject(vramz::Error error) noexcept {
    report.result = vramz::detail::CompressionSmokeResult::preflight_rejected;
    report.error = error;
    report.has_error = true;
    report.final_counts_known = true;
    static_cast<void>(emit_report());
    return 2;
}

} // namespace

int main(int argc, char* argv[]) {
    std::set_terminate(fatal_termination);
    std::array<std::string_view, 16U> arguments{};
    if (argc < 1 || static_cast<std::size_t>(argc - 1) > arguments.size()) {
        return reject(vramz::make_error(vramz::ErrorCode::invalid_argument,
                                        vramz::OperationId::runtime_create));
    }
    const auto count = static_cast<std::size_t>(argc - 1);
    for (std::size_t index = 0U; index < count; ++index) {
        if (argv[index + 1U] == nullptr) {
            return reject(vramz::make_error(vramz::ErrorCode::invalid_argument,
                                            vramz::OperationId::runtime_create));
        }
        arguments[index] = argv[index + 1U];
    }
    const auto parsed =
        vramz::detail::parse_controlled_capacity_smoke_arguments(std::span{arguments}.first(count));
    if (!parsed) {
        return reject(parsed.error());
    }
    auto options = parsed.value();
    report.device_ordinal = options.device;
    if (!options.acknowledged) {
        report.final_counts_known = true;
        return emit_report() ? 0 : 1;
    }
    if (!build_and_loader_approved()) {
        return reject(
            vramz::make_error(vramz::ErrorCode::unsupported, vramz::OperationId::runtime_create));
    }
    auto driver = vramz::detail::make_real_cuda_driver();
    if (!driver) {
        return reject(driver.error());
    }
    const auto release = installed_driver_release();
    if (!release) {
        return reject(release.error());
    }
    options.installed_driver = release.value();
    options.runtime_api = CUDART_VERSION;
    options.prior_driver_api = vramz::detail::m12_prior_driver_api;
    // Shared local preflight uses the pinned prior M7 observation. It calls no CUDA
    // API. After future authorization, the backend observes the live API after cuInit.
    auto codec = vramz::detail::make_real_nvcomp_lz4_api(*driver.value());
    if (!codec) {
        return reject(codec.error());
    }
    physical_execution = true;
    vramz::detail::run_controlled_capacity_smoke(std::move(driver).value(),
                                                 std::move(codec).value(), options, report);
    if (!emit_report()) {
        return 1;
    }
    return report.result == vramz::detail::CompressionSmokeResult::passed ? 0 : 1;
}
