#include "../test_support.hpp"
#include "vramz/detail/compression_smoke.hpp"
#include <array>
#include <string_view>

namespace {
void compatibility(vramz::test::Runner& runner) {
    using namespace vramz;
    using detail::CudaCompatibility;
    runner.begin("M8 CUDA family and NVIDIA release select a candidate without proving features");
    constexpr detail::NvidiaDriverRelease installed{595U, 84U, 0U};
    VRAMZ_CHECK(runner, detail::compression_runtime_candidate(13030, 13030, installed) ==
                            CudaCompatibility::same_family_candidate);
    VRAMZ_CHECK(runner, detail::compression_runtime_candidate(13020, 13030, installed) ==
                            CudaCompatibility::minor_version_candidate);
    VRAMZ_CHECK(runner, detail::compression_runtime_candidate(13020, 13030, {580U, 0U, 0U}) ==
                            CudaCompatibility::minor_version_candidate);
    VRAMZ_CHECK(runner, detail::compression_runtime_candidate(13020, 13030, {579U, 999U, 999U}) ==
                            CudaCompatibility::driver_below_minimum);
    for (const auto versions : {std::array{12090, 13030},
                                {13020, 12090},
                                {14000, 13030},
                                {13020, 14000},
                                {-13020, 13030},
                                {0, 13030}}) {
        VRAMZ_CHECK(runner,
                    detail::compression_runtime_candidate(versions[0U], versions[1U], installed) ==
                        CudaCompatibility::unsupported_family);
    }
    VRAMZ_CHECK(runner, !detail::is_candidate(CudaCompatibility::not_evaluated));
    VRAMZ_CHECK(runner, !detail::is_candidate(CudaCompatibility::unsupported_family));
    VRAMZ_CHECK(runner, !detail::is_candidate(CudaCompatibility::driver_below_minimum));
    runner.begin("M8 NVIDIA release parser accepts exact release text and rejects malformed input");
    for (const auto text : {"595.84", "595.84\n", "595.84.0", "595.84.0\n"}) {
        const auto parsed = detail::parse_nvidia_driver_release(text);
        VRAMZ_CHECK(runner, parsed && parsed.value() == installed);
    }
    const auto patched = detail::parse_nvidia_driver_release("580.12.3");
    VRAMZ_CHECK(runner, patched && patched.value().branch == 580U && patched.value().minor == 12U &&
                            patched.value().patch == 3U);
    for (const auto text :
         {"", "595", "595.", ".84", "595..84", "595.84.", "595.84.0.1", " 595.84", "595.84 ",
          "+595.84", "-595.84", "595.-84", "0.84", "595.84\n\n", "595.84suffix", "595.84\r\n",
          "4294967296.84", "595.4294967296", "595.84.4294967296"}) {
        VRAMZ_CHECK(runner, !detail::parse_nvidia_driver_release(text));
    }
    runner.begin("M8 native compatibility failures preserve domain and completion ambiguity");
    for (const auto native : {36, 803}) {
        const auto definite =
            detail::cuda_runtime_error(native, OperationId::runtime_create, false);
        const auto ambiguous = detail::cuda_runtime_error(native, OperationId::compress, true);
        const auto expected =
            native == 36 ? "CALL_REQUIRES_NEWER_DRIVER" : "SYSTEM_DRIVER_MISMATCH";
        VRAMZ_CHECK(runner, definite.code == ErrorCode::backend_failure &&
                                definite.native_domain == NativeErrorDomain::cuda_runtime &&
                                definite.native_code == native);
        VRAMZ_CHECK(runner, ambiguous.code == ErrorCode::ambiguous_backend_state &&
                                ambiguous.native_domain == NativeErrorDomain::cuda_runtime &&
                                ambiguous.native_code == native);
        VRAMZ_CHECK(runner, detail::cuda_compatibility_failure(definite) == expected);
        VRAMZ_CHECK(runner, detail::cuda_compatibility_failure(ambiguous) == expected);
        auto nvcomp_status = definite;
        nvcomp_status.native_domain = NativeErrorDomain::compression_backend;
        VRAMZ_CHECK(runner, detail::cuda_compatibility_failure(nvcomp_status) == "NOT_IDENTIFIED");
        auto driver_status = definite;
        driver_status.native_domain = NativeErrorDomain::cuda_driver;
        VRAMZ_CHECK(runner, detail::cuda_compatibility_failure(driver_status) == expected);
    }
    constexpr std::array<std::int64_t, 4U> other_codes{35, 222, 801, 804};
    constexpr std::array<std::string_view, 4U> other_names{
        "INSUFFICIENT_DRIVER", "UNSUPPORTED_PTX_VERSION", "UNSUPPORTED_FEATURE",
        "COMPATIBILITY_NOT_SUPPORTED_ON_DEVICE"};
    for (std::size_t index = 0U; index < other_codes.size(); ++index) {
        auto status = detail::cuda_runtime_error(other_codes[index], OperationId::compress, true);
        VRAMZ_CHECK(runner, detail::cuda_compatibility_failure(status) == other_names[index]);
        status.native_domain = NativeErrorDomain::cuda_driver;
        VRAMZ_CHECK(runner, detail::cuda_compatibility_failure(status) ==
                                (index == 0U ? "NOT_IDENTIFIED" : other_names[index]));
    }
}
} // namespace

int main() {
    using namespace vramz;
    test::Runner runner;
    compatibility(runner);
    runner.begin("M8 local arguments require explicit acknowledgment and strict caps");
    const std::array<std::string_view, 1U> acknowledged{"--run-compression-smoke"};
    const auto accepted = detail::parse_compression_smoke_arguments(acknowledged);
    VRAMZ_CHECK(runner, accepted && accepted.value().acknowledged);
    for (const auto arguments : {std::array<std::string_view, 2U>{"--device", "1"},
                                 {"--device", "-1"},
                                 {"--logical-bytes", "4194305"},
                                 {"--logical-bytes", "0"},
                                 {"--max-physical-bytes", "67108865"},
                                 {"--max-physical-bytes", "18446744073709551616"},
                                 {"--run-compression-smoke", "--run-compression-smoke"},
                                 {"--device", "0x0"},
                                 {"--unknown", "0"}}) {
        VRAMZ_CHECK(runner, !detail::parse_compression_smoke_arguments(arguments));
    }
    runner.begin("M8 JSON bounds unknown accounting and rejects invalid source identity");
    detail::CompressionSmokeReport report{};
    std::array<char, 8192U> buffer{};
    const auto result = detail::format_compression_smoke_report(report, "invalid", false, buffer);
    const std::string_view json{buffer.data(), result.written};
    VRAMZ_CHECK(runner, !result.truncated);
    VRAMZ_CHECK(runner, json.find("\"source_sha256\":null") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"final_budget_charge\":null") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"gate_g\":\"NOT_RUN\"") != std::string_view::npos);
    std::array<char, 8U> short_buffer{};
    VRAMZ_CHECK(runner,
                detail::format_compression_smoke_report(report, "", false, short_buffer).truncated);
    runner.begin("M8 compatibility candidate alone never reports hardware validation");
    report.compatibility = detail::CudaCompatibility::minor_version_candidate;
    report.installed_driver = {595U, 84U, 0U};
    report.runtime_api = 13030;
    const auto planned = detail::format_compression_smoke_report(
        report, "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", true, buffer);
    const std::string_view planned_json{buffer.data(), planned.written};
    VRAMZ_CHECK(runner, !planned.truncated);
    VRAMZ_CHECK(runner,
                planned_json.find("MINOR_COMPATIBILITY_CANDIDATE") != std::string_view::npos);
    VRAMZ_CHECK(runner, planned_json.find("\"hardware_validation\":\"NOT_TESTED\"") !=
                            std::string_view::npos);
    return runner.finish();
}
