#include "../support/fake_cuda_driver.hpp"
#include "../test_support.hpp"

#include "vramz/crc32c.hpp"
#include "vramz/detail/m7_driver_identity.hpp"
#include "vramz/detail/raw_smoke.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string_view>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>

using namespace vramz;
using detail::RawSmokeOptions;
using detail::RawSmokeReport;
using detail::RawSmokeResult;
using test::CudaCall;
using test::CudaFaultMode;

namespace {

static_assert(std::is_trivially_copyable_v<RawSmokeReport>);
static_assert(std::is_nothrow_default_constructible_v<RawSmokeReport>);
static_assert(std::is_trivially_copyable_v<detail::DriverFileIdentity>);

[[nodiscard]] std::uint64_t calls(const test::FakeCudaState& audit, CudaCall call) noexcept {
    return audit.counts[static_cast<std::size_t>(call)];
}

[[nodiscard]] bool empty(const test::FakeCudaState& audit) noexcept {
    return audit.owned_bytes == 0U && audit.primary_references == 0U &&
           std::ranges::none_of(audit.physical,
                                [](const auto& resource) {
                                    return resource.handle != detail::CudaPhysicalHandle{};
                                }) &&
           std::ranges::none_of(
               audit.reservations,
               [](const auto& reservation) { return reservation.address != DeviceAddress{}; }) &&
           std::ranges::none_of(
               audit.mappings,
               [](const auto& mapping) { return mapping.address != DeviceAddress{}; }) &&
           std::ranges::none_of(audit.contexts,
                                [](const auto& context) { return context.depth != 0U; });
}

[[nodiscard]] std::shared_ptr<test::FakeCudaState> state(ByteSize minimum = ByteSize{64U}) {
    auto audit = std::make_shared<test::FakeCudaState>();
    audit->config.minimum = minimum;
    audit->config.recommended = ByteSize{32ULL * 1024U * 1024U};
    audit->config.physical_limit = detail::m7_physical_cap;
    audit->config.maximum_resource = detail::m7_physical_cap;
    return audit;
}

void run(const std::shared_ptr<test::FakeCudaState>& audit, RawSmokeReport& report,
         RawSmokeOptions options = {true, 0, detail::m7_physical_cap}) {
    detail::run_raw_smoke(std::make_unique<test::FakeCudaDriverApi>(audit), options, report);
}

void final_empty(test::Runner& runner, const test::FakeCudaState& audit,
                 const RawSmokeReport& report) {
    VRAMZ_CHECK(runner, report.final_counts_known);
    VRAMZ_CHECK(runner, report.final_backend_resources == 0U);
    VRAMZ_CHECK(runner, report.final_va_reservations == 0U);
    VRAMZ_CHECK(runner, report.final_budget_charge == ByteSize{});
    VRAMZ_CHECK(runner, empty(audit));
    VRAMZ_CHECK(runner, audit.destroyed_drivers == 1U);
    VRAMZ_CHECK(runner, audit.dropped_calls == 0U);
}

void check_success(test::Runner& runner, const test::FakeCudaState& audit,
                   const RawSmokeReport& report, ByteSize minimum) {
    VRAMZ_CHECK(runner, report.result == RawSmokeResult::passed && !report.has_error);
    VRAMZ_CHECK(runner, report.probe.uva && report.probe.vmm);
    VRAMZ_CHECK(runner, report.probe.minimum == minimum);
    VRAMZ_CHECK(runner, report.physical_charge == minimum);
    const ByteSize payload{
        std::min(minimum.value(), static_cast<std::uint64_t>(detail::m7_payload_cap))};
    VRAMZ_CHECK(runner, report.logical_payload == payload);
    VRAMZ_CHECK(runner, report.h2d_success && report.d2h_success && report.size_equal);
    VRAMZ_CHECK(runner, report.byte_compare && report.crc_expected == report.crc_actual);
    VRAMZ_CHECK(runner, report.crc_expected != 0U);
    VRAMZ_CHECK(runner,
                report.stable_va_reserved && report.mapping_verified && report.access_verified);
    VRAMZ_CHECK(runner, report.cleanup_unmap && report.cleanup_physical_release &&
                            report.cleanup_va_free && report.cleanup_context_release);
    for (const auto call :
         {CudaCall::initialize, CudaCall::device, CudaCall::retain_primary, CudaCall::reserve,
          CudaCall::create, CudaCall::map, CudaCall::set_access, CudaCall::host_to_device,
          CudaCall::device_to_host, CudaCall::unmap, CudaCall::free_address,
          CudaCall::release_primary}) {
        VRAMZ_CHECK(runner, calls(audit, call) == 1U);
    }
    VRAMZ_CHECK(runner, calls(audit, CudaCall::properties) == 2U);
    VRAMZ_CHECK(runner, calls(audit, CudaCall::retain_mapping) == 2U);
    VRAMZ_CHECK(runner, calls(audit, CudaCall::get_access) == 2U);
    VRAMZ_CHECK(runner, calls(audit, CudaCall::release) == 3U);
    VRAMZ_CHECK(runner, calls(audit, CudaCall::device_to_device) == 0U);
    VRAMZ_CHECK(runner, calls(audit, CudaCall::completion_create) == 0U);
    std::array<CudaCall, 5U> ordered{};
    std::size_t count{};
    bool after_unmap{};
    for (const auto& entry : std::span{audit.log}.first(audit.log_size)) {
        if (entry.call == CudaCall::reserve || entry.call == CudaCall::create ||
            entry.call == CudaCall::map || entry.call == CudaCall::set_access ||
            entry.call == CudaCall::unmap || entry.call == CudaCall::free_address) {
            VRAMZ_CHECK(runner, entry.size == minimum);
        }
        if (entry.call == CudaCall::host_to_device || entry.call == CudaCall::device_to_host) {
            VRAMZ_CHECK(runner,
                        entry.size == payload && entry.size.value() <= detail::m7_payload_cap);
        }
        if (entry.call == CudaCall::unmap) {
            after_unmap = true;
        }
        if (after_unmap &&
            (entry.call == CudaCall::unmap || entry.call == CudaCall::release ||
             entry.call == CudaCall::free_address || entry.call == CudaCall::release_primary)) {
            VRAMZ_CHECK(runner, count < ordered.size());
            if (count < ordered.size()) {
                ordered[count++] = entry.call;
            }
        }
    }
    const std::array expected{CudaCall::unmap, CudaCall::release, CudaCall::free_address,
                              CudaCall::release_primary};
    VRAMZ_CHECK(runner, count == expected.size());
    VRAMZ_CHECK(runner, std::ranges::equal(std::span{ordered}.first(count), expected));
    final_empty(runner, audit, report);
}

void success_tests(test::Runner& runner) {
    for (const ByteSize minimum :
         {ByteSize{64U}, ByteSize{65536U}, ByteSize{2097152U}, detail::m7_physical_cap}) {
        runner.begin("M7 one minimum-granularity RAW allocation and one bounded round trip");
        auto audit = state(minimum);
        RawSmokeReport report{};
        run(audit, report);
        check_success(runner, *audit, report, minimum);
    }
    runner.begin("M7 seeded bounded granularity model conserves ownership and charge");
    constexpr std::uint64_t seed = 12648430U;
    std::uint64_t sequence = seed;
    for (std::uint64_t step = 0U; step < 64U; ++step) {
        runner.trace(seed, step);
        sequence ^= sequence << 13U;
        sequence ^= sequence >> 7U;
        sequence ^= sequence << 17U;
        const ByteSize minimum{64ULL << (sequence % 13U)};
        auto audit = state(minimum);
        RawSmokeReport report{};
        run(audit, report);
        check_success(runner, *audit, report, minimum);
    }
}

void preflight_tests(test::Runner& runner) {
    runner.begin("M7 acknowledgement and argument validation precede all driver initialization");
    for (const RawSmokeOptions options :
         {RawSmokeOptions{}, RawSmokeOptions{true, -1, detail::m7_physical_cap},
          RawSmokeOptions{true, 0, ByteSize{}},
          RawSmokeOptions{true, 0, ByteSize{detail::m7_physical_cap.value() + 1U}}}) {
        auto audit = state();
        RawSmokeReport report{};
        run(audit, report, options);
        VRAMZ_CHECK(runner,
                    report.result == (options.acknowledged ? RawSmokeResult::preflight_rejected
                                                           : RawSmokeResult::planned));
        VRAMZ_CHECK(runner, calls(*audit, CudaCall::initialize) == 0U);
        VRAMZ_CHECK(runner, empty(*audit) && audit->destroyed_drivers == 1U);
    }
    runner.begin("M7 strict bounded argument parser accepts only explicit RAW options");
    const std::array good{std::string_view{"--run-raw-smoke"}, std::string_view{"--device"},
                          std::string_view{"2"}, std::string_view{"--max-physical-bytes"},
                          std::string_view{"65536"}};
    const auto options = detail::parse_raw_smoke_arguments(good);
    VRAMZ_CHECK(runner, options && options.value().acknowledged && options.value().device == 2 &&
                            options.value().maximum_physical == ByteSize{65536U});
    const std::array help{std::string_view{"--help"}};
    const auto planned = detail::parse_raw_smoke_arguments(help);
    VRAMZ_CHECK(runner, planned && !planned.value().acknowledged);
    for (const std::string_view bad :
         {"-1", "+1", "1x", "", "2147483648", "18446744073709551616"}) {
        const std::array invalid{std::string_view{"--device"}, bad};
        VRAMZ_CHECK(runner, !detail::parse_raw_smoke_arguments(invalid));
    }
    for (const std::array invalid :
         {std::array{std::string_view{"--run-raw-smoke"}, std::string_view{"--run-raw-smoke"}},
          std::array{std::string_view{"--max-physical-bytes"}, std::string_view{"0"}},
          std::array{std::string_view{"--max-physical-bytes"}, std::string_view{"16777217"}},
          std::array{std::string_view{"--device"}, std::string_view{"--run-raw-smoke"}}}) {
        VRAMZ_CHECK(runner, !detail::parse_raw_smoke_arguments(invalid));
    }
    for (const std::string_view bad : {"--compress", "--device", "--max-physical-bytes"}) {
        const std::array invalid{bad};
        VRAMZ_CHECK(runner, !detail::parse_raw_smoke_arguments(invalid));
    }
    for (const bool lower_cap : {false, true}) {
        runner.begin("M7 unsupported allocation granularity never reserves or creates GPU storage");
        auto audit = state(lower_cap ? ByteSize{65536U} : ByteSize{33554432U});
        RawSmokeReport report{};
        run(audit, report, {true, 0, lower_cap ? ByteSize{4096U} : detail::m7_physical_cap});
        VRAMZ_CHECK(runner, report.result == RawSmokeResult::unsupported_granularity);
        VRAMZ_CHECK(runner, calls(*audit, CudaCall::reserve) == 0U);
        VRAMZ_CHECK(runner, calls(*audit, CudaCall::create) == 0U);
        VRAMZ_CHECK(runner, report.physical_charge == ByteSize{});
        final_empty(runner, *audit, report);
    }
    for (std::uint32_t variant = 0U; variant < 3U; ++variant) {
        runner.begin("M7 unsupported UVA VMM or prohibited compute mode never allocates");
        auto audit = state();
        audit->config.unified_addressing = variant != 0U;
        audit->config.vmm = variant != 1U;
        audit->config.device_info.compute_mode = variant == 2U ? 2 : 0;
        RawSmokeReport report{};
        run(audit, report);
        VRAMZ_CHECK(runner, report.result == RawSmokeResult::unsupported);
        VRAMZ_CHECK(runner, calls(*audit, CudaCall::retain_primary) == 0U);
        VRAMZ_CHECK(runner, calls(*audit, CudaCall::reserve) == 0U);
        VRAMZ_CHECK(runner, calls(*audit, CudaCall::create) == 0U);
        final_empty(runner, *audit, report);
    }
    runner.begin("M7 requested invalid device stops without iterating other devices");
    auto audit = state();
    RawSmokeReport report{};
    run(audit, report, {true, 1, detail::m7_physical_cap});
    VRAMZ_CHECK(runner, report.result != RawSmokeResult::passed);
    VRAMZ_CHECK(runner, calls(*audit, CudaCall::device) == 1U);
    VRAMZ_CHECK(runner, calls(*audit, CudaCall::create) == 0U);
    final_empty(runner, *audit, report);
}

void operational_fault_tests(test::Runner& runner) {
    for (const auto point : {CudaCall::create, CudaCall::map, CudaCall::set_access,
                             CudaCall::host_to_device, CudaCall::device_to_host}) {
        runner.begin(
            "M7 definite allocation or copy failure cleans exactly without retry or fallback");
        auto audit = state();
        auto driver = std::make_unique<test::FakeCudaDriverApi>(audit);
        driver->inject(point);
        RawSmokeReport report{};
        detail::run_raw_smoke(std::move(driver), {true, 0, detail::m7_physical_cap}, report);
        VRAMZ_CHECK(runner, report.result == RawSmokeResult::failed && report.has_error);
        VRAMZ_CHECK(runner, calls(*audit, point) == 1U);
        VRAMZ_CHECK(runner, calls(*audit, CudaCall::create) == 1U);
        VRAMZ_CHECK(runner, calls(*audit, CudaCall::reserve) == 1U);
        VRAMZ_CHECK(runner, calls(*audit, CudaCall::device_to_device) == 0U);
        VRAMZ_CHECK(runner, report.physical_charge == ByteSize{});
        if (point == CudaCall::create) {
            VRAMZ_CHECK(runner, report.error.code == ErrorCode::out_of_gpu_memory);
            VRAMZ_CHECK(runner, calls(*audit, CudaCall::map) == 0U);
        }
        final_empty(runner, *audit, report);
    }
    runner.begin("M7 genuine physical OOM does not reclaim retry or allocate another tier");
    auto limited = state();
    limited->config.physical_limit = ByteSize{32U};
    RawSmokeReport report{};
    run(limited, report);
    VRAMZ_CHECK(runner, report.result == RawSmokeResult::failed && report.has_error);
    VRAMZ_CHECK(runner, report.error.code == ErrorCode::out_of_gpu_memory);
    VRAMZ_CHECK(runner, calls(*limited, CudaCall::create) == 1U);
    VRAMZ_CHECK(runner, calls(*limited, CudaCall::map) == 0U);
    final_empty(runner, *limited, report);
    for (const auto point : {CudaCall::host_to_device, CudaCall::device_to_host}) {
        runner.begin("M7 corrupted round-trip bytes fail integrity and release exact storage");
        auto audit = state();
        auto driver = std::make_unique<test::FakeCudaDriverApi>(audit);
        driver->inject(point, CudaFaultMode::malformed);
        RawSmokeReport corrupted{};
        detail::run_raw_smoke(std::move(driver), {true, 0, detail::m7_physical_cap}, corrupted);
        VRAMZ_CHECK(runner, corrupted.result == RawSmokeResult::failed && corrupted.has_error);
        VRAMZ_CHECK(runner, corrupted.error.code == ErrorCode::integrity_failure);
        VRAMZ_CHECK(runner, calls(*audit, CudaCall::unmap) == 1U);
        VRAMZ_CHECK(runner, calls(*audit, CudaCall::free_address) == 1U);
        final_empty(runner, *audit, corrupted);
    }
    runner.begin("M7 failed DtoH after host mutation is recoverable without accepting output");
    auto audit = state();
    auto driver = std::make_unique<test::FakeCudaDriverApi>(audit);
    driver->inject(CudaCall::device_to_host, CudaFaultMode::after);
    RawSmokeReport partial{};
    detail::run_raw_smoke(std::move(driver), {true, 0, detail::m7_physical_cap}, partial);
    VRAMZ_CHECK(runner, partial.result == RawSmokeResult::failed);
    VRAMZ_CHECK(runner, !partial.byte_compare && !partial.d2h_success);
    final_empty(runner, *audit, partial);
}

struct DeathCase final {
    CudaCall point{};
    CudaFaultMode mode{};
    std::uint64_t occurrence{1U};
    std::uint64_t detail{};
    std::array<std::uint64_t, 4U> cleanup_counts{};
    bool session_failure{};
};

void fatal_case(test::Runner& runner, DeathCase scenario) {
    const auto child = ::fork();
    VRAMZ_CHECK(runner, child >= 0);
    if (child < 0) {
        return;
    }
    if (child == 0) {
        static_cast<void>(::alarm(5U));
        auto audit = state();
        auto driver = std::make_unique<test::FakeCudaDriverApi>(audit);
        driver->inject(scenario.point, scenario.mode, {scenario.occurrence, scenario.detail});
        RawSmokeReport report{};
        static const test::FakeCudaState* child_audit{};
        static const RawSmokeReport* child_report{};
        static DeathCase expected{};
        child_audit = audit.get();
        child_report = &report;
        expected = scenario;
        std::set_terminate([]() noexcept {
            const auto& observed = *child_audit;
            const std::array actual{calls(observed, CudaCall::unmap),
                                    calls(observed, CudaCall::release),
                                    calls(observed, CudaCall::free_address),
                                    calls(observed, CudaCall::release_primary)};
            const bool stopped =
                actual == expected.cleanup_counts &&
                calls(observed, expected.point) == expected.occurrence &&
                observed.destroyed_drivers == 0U &&
                child_report->result != RawSmokeResult::passed &&
                (!expected.session_failure ||
                 (child_report->has_error && child_report->final_counts_known &&
                  child_report->final_budget_charge.value() == observed.owned_bytes));
            // No backend locks or cleanup at the fatal boundary. Exact ambiguous ownership
            // is intentionally not inferred from telemetry in a process that cannot continue.
            std::_Exit(stopped ? 86 : 1);
        });
        detail::run_raw_smoke(std::move(driver), {true, 0, detail::m7_physical_cap}, report);
        std::_Exit(1);
    }
    int status{};
    pid_t waited{};
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    VRAMZ_CHECK(runner, waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 86);
}

void fatal_tests(test::Runner& runner) {
    const std::array scenarios{
        DeathCase{CudaCall::map, CudaFaultMode::after, 1U, 0U, {0U, 0U, 0U, 0U}},
        DeathCase{CudaCall::host_to_device, CudaFaultMode::after, 1U, 0U, {0U, 1U, 0U, 0U}},
        DeathCase{CudaCall::properties, CudaFaultMode::malformed, 1U, 0U, {0U, 0U, 0U, 0U}},
        DeathCase{CudaCall::properties, CudaFaultMode::malformed, 2U, 0U, {0U, 1U, 0U, 0U}},
        DeathCase{CudaCall::get_access, CudaFaultMode::malformed, 1U, 0U, {0U, 1U, 0U, 0U}},
        DeathCase{CudaCall::retain_mapping, CudaFaultMode::malformed, 1U, 1U, {0U, 0U, 0U, 0U}}};
    for (const auto scenario : scenarios) {
        runner.begin("M7 ambiguous copy map or corroboration stops without destructive cleanup");
        fatal_case(runner, scenario);
    }
    for (const auto mode : {CudaFaultMode::before, CudaFaultMode::after}) {
        const bool definite = mode == CudaFaultMode::before;
        for (const DeathCase scenario :
             {DeathCase{CudaCall::unmap, mode, 1U, 0U, {1U, 2U, 0U, 0U}, definite},
              DeathCase{CudaCall::release, mode, 3U, 0U, {1U, 3U, 0U, 0U}, definite},
              DeathCase{CudaCall::free_address, mode, 1U, 0U, {1U, 3U, 1U, 0U}, definite},
              DeathCase{CudaCall::release_primary, mode, 1U, 0U, {1U, 3U, 1U, 1U}, definite}}) {
            runner.begin("M7 cleanup failure stops with no retry or dependent destruction");
            fatal_case(runner, scenario);
        }
    }
}

void initialization_proof_tests(test::Runner& runner) {
    runner.begin("M7 private verified initialization blocks writes until exact adoption");
    auto audit = state();
    auto created = detail::CudaVmmBackend::create(std::make_unique<test::FakeCudaDriverApi>(audit));
    VRAMZ_CHECK(runner, created);
    if (!created) {
        return;
    }
    auto backend = std::move(created).value();
    const auto address = backend->reserve_address_space(ByteSize{64U}, ByteSize{64U});
    VRAMZ_CHECK(runner, address);
    if (!address) {
        return;
    }
    std::array<std::byte, 64U> input{};
    std::array<std::byte, 64U> output{};
    for (std::size_t index = 0U; index < input.size(); ++index) {
        input[index] = static_cast<std::byte>((index * 17U) & 255U);
    }
    const ContentTag tag{1U, 1U};
    const RepresentationAllocationRequest request{RepresentationState::gpu_raw, ByteSize{64U}, tag,
                                                  address.value().base};
    const auto provisional = backend->allocate_initialized_raw(request, input, output);
    VRAMZ_CHECK(runner, provisional);
    if (!provisional) {
        return;
    }
    VRAMZ_CHECK(runner, input == output);
    VRAMZ_CHECK(runner, provisional.value().metadata.crc32c == crc32c(input));
    VRAMZ_CHECK(runner, !backend->write_bytes(provisional.value().id, ByteOffset{}, input, tag));
    VRAMZ_CHECK(runner, !backend->write_content(provisional.value().id, next_content_tag(tag)));
    VRAMZ_CHECK(runner, calls(*audit, CudaCall::host_to_device) == 1U);
    VRAMZ_CHECK(runner, calls(*audit, CudaCall::device_to_host) == 1U);
    runner.begin("M7 private proof never accepts a substituted representation CRC");
    auto malformed = provisional.value();
    malformed.metadata.crc32c ^= 1U;
    malformed.metadata.stored_crc32c = malformed.metadata.crc32c;
    VRAMZ_CHECK(runner, !backend->adopt_allocation(malformed));
    VRAMZ_CHECK(runner, backend->is_unadopted(provisional.value().id));
    runner.begin("M7 valid one-use adoption corroborates mapping without a second byte readback");
    const auto adopted = backend->adopt_allocation(provisional.value());
    VRAMZ_CHECK(runner, adopted);
    VRAMZ_CHECK(runner, !backend->is_unadopted(provisional.value().id));
    VRAMZ_CHECK(runner, calls(*audit, CudaCall::device_to_host) == 1U);
    VRAMZ_CHECK(runner, calls(*audit, CudaCall::properties) == 2U);
    VRAMZ_CHECK(runner, !backend->adopt_allocation(provisional.value()));
    VRAMZ_CHECK(runner, backend->release(provisional.value().id, ReleasePhase::close));
    VRAMZ_CHECK(runner, backend->release_address_space(address.value()));
    VRAMZ_CHECK(runner, backend->shutdown());
    backend.reset();
    VRAMZ_CHECK(runner, empty(*audit));

    runner.begin("M7 initialized RAW rejects overlapping or inconsistent host spans before create");
    auto invalid_audit = state();
    auto invalid_backend =
        detail::CudaVmmBackend::create(std::make_unique<test::FakeCudaDriverApi>(invalid_audit));
    VRAMZ_CHECK(runner, invalid_backend);
    if (!invalid_backend) {
        return;
    }
    std::array<std::byte, 128U> bytes{};
    const RepresentationAllocationRequest invalid_request{RepresentationState::gpu_raw,
                                                          ByteSize{64U}, tag, DeviceAddress{64U}};
    auto& instance = *invalid_backend.value();
    const auto first = std::span{bytes}.first(64U);
    const auto overlapping = std::span{bytes}.subspan(32U, 64U);
    const auto separate = std::span{bytes}.last(64U);
    VRAMZ_CHECK(runner, !instance.allocate_initialized_raw(invalid_request, first, first));
    VRAMZ_CHECK(runner, !instance.allocate_initialized_raw(invalid_request, first, overlapping));
    VRAMZ_CHECK(runner, !instance.allocate_initialized_raw(invalid_request, overlapping, first));
    VRAMZ_CHECK(runner,
                !instance.allocate_initialized_raw(invalid_request, first, separate.first(63U)));
    VRAMZ_CHECK(runner, !instance.allocate_initialized_raw(invalid_request, {}, {}));
    VRAMZ_CHECK(runner, calls(*invalid_audit, CudaCall::create) == 0U);
    VRAMZ_CHECK(runner, instance.shutdown());
}

void identity_and_json_tests(test::Runner& runner) {
    runner.begin(
        "M7 real-driver identity compares every immutable stat field without driver calls");
    struct stat information {};
    information.st_mode = S_IFREG | S_IRUSR;
    information.st_dev = 2;
    information.st_ino = 3;
    information.st_size = 4096;
    information.st_mtim = {7, 11};
    information.st_ctim = {9, 13};
    const detail::DriverFileIdentity expected{2U, 3U, 4096U, 7000000011ULL, 9000000013ULL};
    VRAMZ_CHECK(runner, detail::matches_driver_file_identity(information, expected));
    for (std::uint32_t field = 0U; field < 10U; ++field) {
        auto changed = information;
        switch (field) {
        case 0U:
            changed.st_mode = S_IFDIR;
            break;
        case 1U:
            ++changed.st_dev;
            break;
        case 2U:
            ++changed.st_ino;
            break;
        case 3U:
            ++changed.st_size;
            break;
        case 4U:
            ++changed.st_mtim.tv_nsec;
            break;
        case 5U:
            ++changed.st_ctim.tv_nsec;
            break;
        case 6U:
            changed.st_mtim.tv_sec = -1;
            break;
        case 7U:
            changed.st_ctim.tv_nsec = 1000000000L;
            break;
        case 8U:
            changed.st_size = -1;
            break;
        default:
            changed.st_mtim.tv_sec = std::numeric_limits<time_t>::max();
            break;
        }
        VRAMZ_CHECK(runner, !detail::matches_driver_file_identity(changed, expected));
    }
    const timespec negative_nanoseconds{0, -1};
    VRAMZ_CHECK(runner, !detail::driver_timestamp_ns(negative_nanoseconds));

    runner.begin("M7 fake JSON never claims hardware validation or emits raw address details");
    auto audit = state();
    RawSmokeReport report{};
    run(audit, report);
    report.has_error = true;
    report.error = make_error(ErrorCode::backend_failure, OperationId::verify, 123456789012345ULL,
                              987654321098765ULL);
    constexpr std::string_view digest =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    std::array<char, 4096U> output{};
    const auto formatted = detail::format_raw_smoke_report(report, digest, false, output);
    VRAMZ_CHECK(runner, !formatted.truncated);
    const std::string_view json{output.data(), formatted.written};
    VRAMZ_CHECK(runner, json.starts_with('{') && json.ends_with('}'));
    VRAMZ_CHECK(runner,
                json.find("\"hardware_validation\":\"NOT_TESTED\"") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"execution_kind\":\"fake\"") != std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("123456789012345") == std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("987654321098765") == std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("0x") == std::string_view::npos);
    VRAMZ_CHECK(runner, json.find("\"final_budget_charge\":0") != std::string_view::npos);
    runner.begin("M7 JSON reports bounded truncation and rejects malformed source identity");
    std::array<char, 8U> small{};
    const auto truncated = detail::format_raw_smoke_report(report, digest, false, small);
    VRAMZ_CHECK(runner, truncated.truncated && small.back() == '\0');
    const auto invalid = detail::format_raw_smoke_report(report, "not-a-digest", false, output);
    const std::string_view invalid_json{output.data(), invalid.written};
    VRAMZ_CHECK(runner,
                invalid_json.find("\"source_identity_valid\":false") != std::string_view::npos);
    VRAMZ_CHECK(runner, invalid_json.find("\"source_sha256\":null") != std::string_view::npos);
}

} // namespace

int main() {
    test::Runner runner;
    success_tests(runner);
    preflight_tests(runner);
    operational_fault_tests(runner);
    fatal_tests(runner);
    initialization_proof_tests(runner);
    identity_and_json_tests(runner);
    return runner.finish();
}
