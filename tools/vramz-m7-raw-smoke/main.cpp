#include "m7_build_config.hpp"
#include "vramz/capabilities.hpp"
#include "vramz/detail/m7_driver_identity.hpp"
#include "vramz/detail/raw_smoke.hpp"

#include <cuda.h>

#include <array>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <span>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace {

// Fixed process-lifetime report storage also survives an allocation-free fail-stop path.
// Constant initialization does not construct an adapter or invoke any CUDA function.
constinit vramz::detail::RawSmokeReport report{};
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
    std::array<char, 8192U> output{};
    const auto formatted = vramz::detail::format_raw_smoke_report(
        report, vramz::detail::m7_source_sha256, physical_execution,
        std::span{output}.first(output.size() - 1U));
    if (formatted.truncated || formatted.written >= output.size()) {
        constexpr std::string_view fallback{
            "{\"m7_version\":1,\"result\":\"FAIL\",\"error\":\"report_format_failure\","
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
    report.result = vramz::detail::RawSmokeResult::failed;
    if (!report.has_error) {
        report.error = vramz::make_error(vramz::ErrorCode::backend_contract_violation,
                                         vramz::OperationId::unknown);
        report.has_error = true;
        report.final_counts_known = false;
    }
    // A precise definite-cleanup failure retains its exact report. No destructor, retry,
    // guessed accounting repair or in-flight resource cleanup runs across this boundary.
    static_cast<void>(emit_report());
    std::_Exit(86);
}

[[nodiscard]] bool approved_identity(const struct stat& value) noexcept {
    return vramz::detail::matches_driver_file_identity(
        value, {vramz::detail::m7_driver_device, vramz::detail::m7_driver_inode,
                vramz::detail::m7_driver_size, vramz::detail::m7_driver_mtime_ns,
                vramz::detail::m7_driver_ctime_ns});
}

[[nodiscard]] bool loaded_driver_is_approved() noexcept {
    Dl_info loaded{};
    // POSIX dladdr accepts a code address as void*. This Linux x86_64 ABI conversion
    // observes the loaded cuInit symbol only; it neither calls cuInit nor loads a library.
    if (::dladdr(reinterpret_cast<const void*>(&cuInit), &loaded) == 0 ||
        loaded.dli_fname == nullptr) {
        return false;
    }
    const auto approved = vramz::detail::m7_approved_driver_path;
    if (approved.empty() || approved.find("/stubs/") != std::string_view::npos ||
        std::string_view{loaded.dli_fname}.find("/stubs/") != std::string_view::npos) {
        return false;
    }
    std::array<char, PATH_MAX> loaded_path{};
    std::array<char, PATH_MAX> approved_path{};
    if (::realpath(loaded.dli_fname, loaded_path.data()) == nullptr ||
        ::realpath(approved.data(), approved_path.data()) == nullptr) {
        return false;
    }
    if (std::string_view{loaded_path.data()} != approved ||
        std::string_view{approved_path.data()} != approved) {
        return false;
    }
    struct stat loaded_stat {};
    struct stat approved_stat {};
    return ::stat(loaded_path.data(), &loaded_stat) == 0 &&
           ::stat(approved_path.data(), &approved_stat) == 0 && approved_identity(loaded_stat) &&
           approved_identity(approved_stat);
}

[[nodiscard]] bool build_allows_raw_smoke() noexcept {
#if defined(VRAMZ_M7_RAW_ONLY) && VRAMZ_M7_RAW_ONLY == 1
    constexpr bool raw_only = true;
#else
    constexpr bool raw_only = false;
#endif
    const auto capabilities = vramz::build_capabilities();
    const auto source = vramz::detail::m7_source_sha256;
    bool nonzero{};
    for (const auto value : source) {
        if ((value < '0' || value > '9') && (value < 'a' || value > 'f')) {
            return false;
        }
        nonzero = nonzero || value != '0';
    }
    // The public Runtime still reports real execution disabled. Only this separately
    // gated executable authorizes RAW flow; it never constructs Runtime/policy/nvCOMP.
    if (!capabilities.cuda_adapter_compiled || capabilities.nvcomp_adapter_compiled ||
        source.size() != 64U || !nonzero || vramz::detail::m7_approved_driver_sha256.empty()) {
        return false;
    }
    return raw_only;
}

[[nodiscard]] int reject(vramz::Error error) noexcept {
    report.result = vramz::detail::RawSmokeResult::preflight_rejected;
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
    const auto parsed = vramz::detail::parse_raw_smoke_arguments(std::span{arguments}.first(count));
    if (!parsed) {
        return reject(parsed.error());
    }
    const auto options = parsed.value();
    report.device_ordinal = options.device;
    report.maximum_physical = options.maximum_physical;
    if (!options.acknowledged) {
        report.final_counts_known = true;
        return emit_report() ? 0 : 1;
    }
    if (!build_allows_raw_smoke() || !loaded_driver_is_approved()) {
        return reject(
            vramz::make_error(vramz::ErrorCode::unsupported, vramz::OperationId::runtime_create));
    }
    auto driver = vramz::detail::make_real_cuda_driver();
    if (!driver) {
        return reject(driver.error());
    }
    physical_execution = true;
    vramz::detail::run_raw_smoke(std::move(driver).value(), options, report);
    if (!emit_report()) {
        return 1;
    }
    return report.result == vramz::detail::RawSmokeResult::passed ? 0 : 1;
}
