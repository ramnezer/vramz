#include "../support/fake_cuda_driver.hpp"
#include "../test_support.hpp"

#include "vramz/crc32c.hpp"
#include "vramz/detail/allocation.hpp"
#include "vramz/detail/cuda_vmm_backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <limits>
#include <sys/wait.h>
#include <thread>
#include <type_traits>
#include <unistd.h>

using namespace vramz;
using detail::CudaVmmBackend;
using test::CudaCall;
using test::CudaFaultMode;

namespace {

static_assert(!std::is_copy_constructible_v<CudaVmmBackend>);
static_assert(!std::is_move_assignable_v<CudaVmmBackend>);
static_assert(!std::is_move_assignable_v<test::FakeCudaDriverApi>);
static_assert(std::is_trivially_copyable_v<AddressReservation>);
static_assert(std::is_trivially_copyable_v<BackendAllocation>);
static_assert(std::is_trivially_copyable_v<detail::CudaAllocationProperties>);
static_assert(std::is_trivially_copyable_v<detail::CudaDeviceInfo>);
static_assert(std::is_nothrow_default_constructible_v<detail::CudaDeviceInfo>);
static_assert(noexcept(Result<detail::CudaDeviceInfo>{detail::CudaDeviceInfo{}}));
static_assert(noexcept(Result<detail::CudaPhysicalHandle>{detail::CudaPhysicalHandle{}}));

struct Fixture final {
    std::shared_ptr<test::FakeCudaState> audit{std::make_shared<test::FakeCudaState>()};
    test::FakeCudaDriverApi* driver{};
    std::unique_ptr<CudaVmmBackend> backend{};
    explicit Fixture(const test::FakeCudaConfig& config = {}) {
        audit->config = config;
        auto api = std::make_unique<test::FakeCudaDriverApi>(audit);
        driver = api.get();
        auto created = CudaVmmBackend::create(std::move(api));
        if (created) {
            backend = std::move(created).value();
        }
    }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
};

[[nodiscard]] RuntimeConfig runtime_config() noexcept {
    auto config = test::test_runtime_config();
    config.required_capabilities.host_tier = false;
    config.budgets.host = {};
    return config;
}
[[nodiscard]] std::array<std::byte, 256U> payload() noexcept {
    std::array<std::byte, 256U> bytes{};
    for (std::size_t index = 0U; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::byte>((index * 17U + 23U) & 255U);
    }
    return bytes;
}
void conserved(test::Runner& runner, const Runtime& runtime, const Fixture& fixture) {
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
    VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
    const auto usage = runtime.stats().gpu;
    const auto sum = usage.committed.value() + usage.staging.value() + usage.workspace.value() +
                     usage.cleanup_debt.value();
    const std::scoped_lock lock{fixture.audit->mutex};
    VRAMZ_CHECK(runner, sum == fixture.audit->owned_bytes);
}
void clean(test::Runner& runner, Runtime& runtime, Fixture& fixture) {
    VRAMZ_CHECK(runner, runtime.shutdown());
    conserved(runner, runtime, fixture);
    const auto usage = runtime.stats().gpu;
    VRAMZ_CHECK(runner, usage.committed == ByteSize{} && usage.reserved == ByteSize{} &&
                            usage.staging == ByteSize{} && usage.workspace == ByteSize{} &&
                            usage.cleanup_debt == ByteSize{});
    VRAMZ_CHECK(runner, fixture.driver->empty());
}

void capability_tests(test::Runner& runner) {
    runner.begin(
        "fake initialization is explicit idempotent and required before device/context mutation");
    {
        auto audit = std::make_shared<test::FakeCudaState>();
        test::FakeCudaDriverApi driver{audit};
        VRAMZ_CHECK(runner, !audit->initialized);
        VRAMZ_CHECK(runner, !driver.device(0));
        VRAMZ_CHECK(runner, !driver.retain_primary(0));
        driver.inject(CudaCall::initialize);
        VRAMZ_CHECK(runner, !driver.initialize() && !audit->initialized);
        VRAMZ_CHECK(runner, driver.initialize());
        VRAMZ_CHECK(runner, driver.initialize());
        VRAMZ_CHECK(runner, audit->initialized && driver.device(0));
        VRAMZ_CHECK(runner, driver.empty());
    }
    runner.begin("device telemetry is explicit bounded and read-only after initialization");
    {
        auto audit = std::make_shared<test::FakeCudaState>();
        test::FakeCudaDriverApi driver{audit};
        VRAMZ_CHECK(runner, !driver.device_info(0));
        VRAMZ_CHECK(runner, !audit->initialized);
        VRAMZ_CHECK(runner, driver.initialize());
        const auto info = driver.device_info(0);
        VRAMZ_CHECK(runner, info);
        if (info) {
            VRAMZ_CHECK(runner, info.value().name == audit->config.device_info.name);
            VRAMZ_CHECK(runner, info.value().compute_mode == 0);
        }
        VRAMZ_CHECK(runner, !driver.device_info(1));
        VRAMZ_CHECK(runner, driver.count(CudaCall::initialize) == 1U);
        VRAMZ_CHECK(runner, driver.count(CudaCall::retain_primary) == 0U);
        VRAMZ_CHECK(runner, driver.count(CudaCall::create) == 0U);
        VRAMZ_CHECK(runner, driver.empty());
    }
    runner.begin("device telemetry failure neither retains a context nor mutates compute mode");
    {
        auto audit = std::make_shared<test::FakeCudaState>();
        audit->config.device_info.compute_mode = 2;
        test::FakeCudaDriverApi driver{audit};
        VRAMZ_CHECK(runner, driver.initialize());
        driver.inject(CudaCall::device_info);
        const auto failed = driver.device_info(0);
        VRAMZ_CHECK(runner, !failed);
        if (!failed) {
            VRAMZ_CHECK(runner, failed.error().code == ErrorCode::backend_failure);
        }
        const auto info = driver.device_info(0);
        VRAMZ_CHECK(runner, info && info.value().compute_mode == 2);
        VRAMZ_CHECK(runner, audit->config.device_info.compute_mode == 2);
        VRAMZ_CHECK(runner, driver.count(CudaCall::retain_primary) == 0U);
        VRAMZ_CHECK(runner, driver.empty());
    }
    runner.begin("device name telemetry is fixed-size terminated without modifying source state");
    {
        auto audit = std::make_shared<test::FakeCudaState>();
        audit->config.device_info.name.fill('X');
        audit->config.device_info.compute_mode = 3;
        test::FakeCudaDriverApi driver{audit};
        VRAMZ_CHECK(runner, driver.initialize());
        const auto info = driver.device_info(0);
        VRAMZ_CHECK(runner, info);
        if (info) {
            VRAMZ_CHECK(runner, info.value().name.front() == 'X');
            VRAMZ_CHECK(runner, info.value().name.back() == '\0');
            VRAMZ_CHECK(runner, info.value().compute_mode == 3);
        }
        VRAMZ_CHECK(runner, audit->config.device_info.name.back() == 'X');
        VRAMZ_CHECK(runner, driver.empty());
    }
    runner.begin("CUDA-neutral capabilities and queried minimum versus recommended");
    Fixture fixture;
    VRAMZ_CHECK(runner, fixture.backend);
    if (!fixture.backend) {
        return;
    }
    const auto caps = fixture.backend->capabilities();
    VRAMZ_CHECK(runner, caps.allocation_granularity == ByteSize{64U});
    VRAMZ_CHECK(runner, fixture.backend->recommended_granularity() == ByteSize{256U});
    VRAMZ_CHECK(runner,
                caps.stable_device_address && !caps.host_tier && !caps.asynchronous_operations);
    VRAMZ_CHECK(runner, !fixture.backend->compression_available());
    VRAMZ_CHECK(
        runner,
        fixture.backend->allocation_bound(RepresentationState::gpu_raw, ByteSize{65U}).value() ==
            ByteSize{128U});
    VRAMZ_CHECK(runner, !fixture.backend->allocation_bound(
                            RepresentationState::gpu_raw,
                            ByteSize{std::numeric_limits<std::uint64_t>::max()}));
    VRAMZ_CHECK(runner,
                !fixture.backend->allocation_bound(RepresentationState::host_raw, ByteSize{64U}));
    VRAMZ_CHECK(runner, !fixture.backend->allocate_workspace(PhysicalTier::gpu, ByteSize{64U}));
    VRAMZ_CHECK(runner, fixture.backend->shutdown());
    VRAMZ_CHECK(runner, fixture.driver->empty());

    for (std::uint32_t variant = 0U; variant < 6U; ++variant) {
        runner.begin("unsupported probe rejects before context/resource mutation");
        test::FakeCudaConfig config{};
        switch (variant) {
        case 0U:
            config.unified_addressing = false;
            break;
        case 1U:
            config.vmm = false;
            break;
        case 2U:
            config.minimum = ByteSize{};
            break;
        case 3U:
            config.minimum = ByteSize{96U};
            break;
        case 4U:
            config.host_page = ByteSize{};
            break;
        default:
            config.version = 11010;
            break;
        }
        Fixture rejected{config};
        VRAMZ_CHECK(runner, !rejected.backend);
        VRAMZ_CHECK(runner, rejected.audit->primary_references == 0U);
        VRAMZ_CHECK(runner,
                    rejected.audit->counts[static_cast<std::size_t>(CudaCall::create)] == 0U);
    }
    runner.begin("minimum 64 KiB does not allocate recommended 2 MiB");
    test::FakeCudaConfig larger{};
    larger.minimum = ByteSize{65536U};
    larger.recommended = ByteSize{2097152U};
    larger.host_page = ByteSize{4096U};
    Fixture large{larger};
    VRAMZ_CHECK(runner, large.backend);
    if (large.backend) {
        VRAMZ_CHECK(runner,
                    large.backend->allocation_bound(RepresentationState::gpu_raw, ByteSize{4097U})
                            .value() == ByteSize{65536U});
    }
}

void reservation_tests(test::Runner& runner) {
    runner.begin("bounded VA ownership, invalid ranges and retryable free");
    Fixture fixture;
    auto& backend = *fixture.backend;
    VRAMZ_CHECK(runner, !backend.reserve_address_space(ByteSize{65U}, ByteSize{64U}));
    VRAMZ_CHECK(runner, !backend.reserve_address_space(ByteSize{64U}, ByteSize{96U}));
    fixture.driver->inject(CudaCall::reserve);
    VRAMZ_CHECK(runner, !backend.reserve_address_space(ByteSize{64U}, ByteSize{64U}));
    const auto address = backend.reserve_address_space(ByteSize{256U}, ByteSize{64U});
    VRAMZ_CHECK(runner, address);
    if (!address) {
        return;
    }
    VRAMZ_CHECK(runner, backend.owned_charge(PhysicalTier::gpu) == ByteSize{});
    fixture.driver->inject(CudaCall::free_address);
    VRAMZ_CHECK(runner, !backend.release_address_space(address.value()));
    VRAMZ_CHECK(runner, backend.owns_address_space(address.value().id));
    VRAMZ_CHECK(runner, backend.release_address_space(address.value()));
    VRAMZ_CHECK(runner, !backend.release_address_space(address.value()));
    VRAMZ_CHECK(runner, backend.shutdown());
    VRAMZ_CHECK(runner, fixture.driver->empty());

    runner.begin("address registry exhausts before driver mutation");
    Fixture bounded;
    std::array<AddressReservation, 64U> addresses{};
    for (auto& entry : addresses) {
        const auto reserved = bounded.backend->reserve_address_space(ByteSize{64U}, ByteSize{64U});
        VRAMZ_CHECK(runner, reserved);
        if (reserved) {
            entry = reserved.value();
        }
    }
    const auto calls = bounded.driver->count(CudaCall::reserve);
    VRAMZ_CHECK(runner, !bounded.backend->reserve_address_space(ByteSize{64U}, ByteSize{64U}));
    VRAMZ_CHECK(runner, bounded.driver->count(CudaCall::reserve) == calls);
    for (const auto& entry : addresses) {
        VRAMZ_CHECK(runner, bounded.backend->release_address_space(entry));
    }
}

void remap_oracle(test::Runner& runner) {
    runner.begin(
        "Buffer VA reservation survives physical replacement with identical bytes CRC and address");
    Fixture fixture;
    auto& backend = *fixture.backend;
    const auto reserved = backend.reserve_address_space(ByteSize{256U}, ByteSize{64U});
    VRAMZ_CHECK(runner, reserved);
    if (!reserved) {
        return;
    }
    const auto address = reserved.value();
    const auto bytes = payload();
    std::array<std::byte, 256U> read{};
    ResourceId previous{};
    std::uint32_t previous_crc{};
    AsyncErrorChannel errors{4U};
    for (std::uint32_t iteration = 0U; iteration < 2U; ++iteration) {
        const auto provisional = backend.allocate_representation(
            {RepresentationState::gpu_raw, ByteSize{256U}, ContentTag{}, address.base});
        VRAMZ_CHECK(runner, provisional);
        if (!provisional) {
            return;
        }
        const auto exact = detail::corroborate_allocation(
            backend, errors, provisional.value(),
            {PhysicalTier::gpu, ResourceKind::representation, ByteSize{256U}, ByteSize{64U},
             ByteSize{256U}, RepresentationState::gpu_raw, address.base});
        VRAMZ_CHECK(runner, exact.id != previous && exact.address == address.base);
        VRAMZ_CHECK(runner, !backend.release_address_space(address));
        const auto written = backend.write_bytes(exact.id, ByteOffset{}, bytes, ContentTag{});
        VRAMZ_CHECK(runner, written);
        if (!written) {
            return;
        }
        VRAMZ_CHECK(runner, backend.verify(exact.id, written.value()));
        VRAMZ_CHECK(runner, backend.read_bytes(exact.id, ByteOffset{}, read));
        VRAMZ_CHECK(runner, bytes == read && crc32c(read) == crc32c(bytes));
        const auto descriptor = backend.allocation(exact.id);
        VRAMZ_CHECK(runner, descriptor && descriptor.value().metadata.crc32c == crc32c(bytes));
        if (iteration != 0U) {
            VRAMZ_CHECK(runner, descriptor.value().metadata.crc32c == previous_crc);
        }
        previous_crc = descriptor.value().metadata.crc32c;
        previous = exact.id;
        VRAMZ_CHECK(runner, backend.release(exact.id, ReleasePhase::close));
        VRAMZ_CHECK(runner, backend.owned_charge(PhysicalTier::gpu) == ByteSize{});
        VRAMZ_CHECK(runner, backend.owns_address_space(address.id));
    }
    VRAMZ_CHECK(runner, fixture.driver->count(CudaCall::set_access) == 2U);
    VRAMZ_CHECK(runner, fixture.driver->count(CudaCall::unmap) == 2U);
    VRAMZ_CHECK(runner, backend.release_address_space(address));
    VRAMZ_CHECK(runner, backend.shutdown());
    VRAMZ_CHECK(runner, fixture.driver->empty());
    constexpr std::array ordered{
        CudaCall::reserve,        CudaCall::create,  CudaCall::map,     CudaCall::set_access,
        CudaCall::unmap,          CudaCall::release, CudaCall::create,  CudaCall::map,
        CudaCall::set_access,     CudaCall::unmap,   CudaCall::release, CudaCall::free_address,
        CudaCall::release_primary};
    std::size_t matched{};
    const std::scoped_lock audit_lock{fixture.audit->mutex};
    for (std::size_t index = 0U; index < fixture.audit->log_size && matched < ordered.size();
         ++index) {
        if (fixture.audit->log[index].call == ordered[matched]) {
            ++matched;
        }
    }
    VRAMZ_CHECK(runner, matched == ordered.size());
}

void runtime_oracle(test::Runner& runner) {
    runner.begin(
        "real Runtime through fake VMM, padded tail, contiguous lease bytes and complete cleanup");
    Fixture fixture;
    auto created =
        testing::RuntimeAccess::create_with_backend(runtime_config(), std::move(fixture.backend));
    VRAMZ_CHECK(runner, created);
    if (!created) {
        return;
    }
    auto runtime = std::move(created).value();
    auto allocated = runtime.allocate(ByteSize{200U});
    VRAMZ_CHECK(runner, allocated);
    if (!allocated) {
        return;
    }
    auto buffer = std::move(allocated).value();
    conserved(runner, runtime, fixture);
    VRAMZ_CHECK(runner, runtime.stats().gpu.committed == ByteSize{256U});
    const auto first = testing::chunk_snapshot(buffer, 0U).value();
    const auto last = testing::chunk_snapshot(buffer, 3U).value();
    VRAMZ_CHECK(runner,
                last.authoritative.address.value() == first.authoritative.address.value() + 192U);
    auto acquired = buffer.acquire({ByteOffset{}, ByteSize{200U}}, {AccessMode::read_write});
    VRAMZ_CHECK(runner, acquired);
    if (!acquired) {
        return;
    }
    auto lease = std::move(acquired).value();
    const auto bytes = payload();
    std::array<std::byte, 200U> read{};
    VRAMZ_CHECK(runner, lease.device_span().value().address == first.authoritative.address);
    VRAMZ_CHECK(runner, testing::write_bytes(lease, ByteOffset{}, std::span{bytes}.first(200U)));
    VRAMZ_CHECK(runner, testing::read_bytes(lease, ByteOffset{}, read));
    VRAMZ_CHECK(runner, std::ranges::equal(read, std::span{bytes}.first(200U)));
    VRAMZ_CHECK(runner, !buffer.close());
    VRAMZ_CHECK(runner, lease.close());
    const auto closed = buffer.close();
    VRAMZ_CHECK(runner, !closed && closed.error().code == ErrorCode::stale_handle);
    VRAMZ_CHECK(runner, runtime.stats().gpu.committed == ByteSize{});
    clean(runner, runtime, fixture);
}

void writable_close_oracle(test::Runner& runner) {
    runner.begin(
        "writable lease close reads back CRC and advances content without host mirror authority");
    Fixture fixture;
    auto created =
        testing::RuntimeAccess::create_with_backend(runtime_config(), std::move(fixture.backend));
    VRAMZ_CHECK(runner, created);
    if (!created) {
        return;
    }
    auto runtime = std::move(created).value();
    auto allocated = runtime.allocate(ByteSize{64U});
    VRAMZ_CHECK(runner, allocated);
    if (!allocated) {
        return;
    }
    auto buffer = std::move(allocated).value();
    const auto before = testing::chunk_snapshot(buffer, 0U).value();
    auto acquired = buffer.acquire({ByteOffset{}, ByteSize{64U}}, {AccessMode::read_write});
    VRAMZ_CHECK(runner, acquired);
    if (!acquired) {
        return;
    }
    auto lease = std::move(acquired).value();
    const auto bytes = payload();
    // Only the fake API is invoked: simulate application writes inside its owned lease.
    VRAMZ_CHECK(runner, fixture.driver->push_context(detail::CudaContext{1U}));
    VRAMZ_CHECK(runner, fixture.driver->copy_to_device(lease.device_span().value().address,
                                                       std::span{bytes}.first(64U)));
    VRAMZ_CHECK(runner, fixture.driver->pop_context());
    VRAMZ_CHECK(runner, lease.close());
    const auto after = testing::chunk_snapshot(buffer, 0U).value();
    VRAMZ_CHECK(runner, after.authoritative.metadata.crc32c == crc32c(std::span{bytes}.first(64U)));
    VRAMZ_CHECK(runner,
                after.authoritative.content == next_content_tag(before.authoritative.content));
    VRAMZ_CHECK(runner, buffer.close());
    clean(runner, runtime, fixture);
}

void cleanup_tests(test::Runner& runner) {
    for (const auto point : {CudaCall::unmap, CudaCall::release, CudaCall::free_address}) {
        runner.begin("Runtime quarantine retains exact physical or VA ownership on definite "
                     "cleanup failure");
        Fixture fixture;
        auto created = testing::RuntimeAccess::create_with_backend(runtime_config(),
                                                                   std::move(fixture.backend));
        VRAMZ_CHECK(runner, created);
        if (!created) {
            return;
        }
        auto runtime = std::move(created).value();
        auto allocated = runtime.allocate(ByteSize{64U});
        VRAMZ_CHECK(runner, allocated);
        if (!allocated) {
            return;
        }
        auto buffer = std::move(allocated).value();
        fixture.driver->inject(point);
        VRAMZ_CHECK(runner, !buffer.close());
        conserved(runner, runtime, fixture);
        VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt ==
                                ByteSize{point == CudaCall::free_address ? 0U : 64U});
        if (point == CudaCall::release) {
            VRAMZ_CHECK(runner, fixture.driver->count(CudaCall::unmap) == 1U);
        }
        VRAMZ_CHECK(runner, testing::quarantine(buffer));
        clean(runner, runtime, fixture);
        if (point == CudaCall::release) {
            VRAMZ_CHECK(runner, fixture.driver->count(CudaCall::unmap) == 1U);
        }
    }
    runner.begin(
        "corroborated allocation materialize error preserves exact orphan ownership until retry");
    Fixture fixture;
    auto created =
        testing::RuntimeAccess::create_with_backend(runtime_config(), std::move(fixture.backend));
    auto runtime = std::move(created).value();
    testing::RuntimeAccess::inject_materialize_failure(runtime);
    fixture.driver->inject(CudaCall::unmap);
    VRAMZ_CHECK(runner, !runtime.allocate(ByteSize{64U}));
    conserved(runner, runtime, fixture);
    VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == ByteSize{64U});
    // Contract-debt orphans are first retried by shutdown, before their Buffer VA owner.
    const auto first_shutdown = runtime.shutdown();
    VRAMZ_CHECK(runner,
                !first_shutdown && first_shutdown.error().code == ErrorCode::backend_failure);
    conserved(runner, runtime, fixture);
    VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == ByteSize{64U});
    clean(runner, runtime, fixture);
}

void context_tests(test::Runner& runner) {
    runner.begin(
        "nested context restores caller context and retries definite primary release failure");
    Fixture fixture;
    VRAMZ_CHECK(runner, fixture.driver->push_context(detail::CudaContext{99U}));
    const auto reserved = fixture.backend->reserve_address_space(ByteSize{64U}, ByteSize{64U});
    VRAMZ_CHECK(runner, reserved);
    VRAMZ_CHECK(runner, fixture.driver->current_context() == detail::CudaContext{99U});
    VRAMZ_CHECK(runner, fixture.backend->release_address_space(reserved.value()));
    VRAMZ_CHECK(runner, fixture.driver->pop_context().value() == detail::CudaContext{99U});
    fixture.driver->inject(CudaCall::release_primary);
    VRAMZ_CHECK(runner, !fixture.backend->shutdown());
    VRAMZ_CHECK(runner, fixture.audit->primary_references == 1U);
    VRAMZ_CHECK(runner, fixture.backend->shutdown());
    VRAMZ_CHECK(runner, fixture.driver->empty());
    fixture.backend.reset();
    VRAMZ_CHECK(runner, fixture.audit->destroyed_drivers == 1U);

    runner.begin(
        "direct backend destruction releases owned mapping handle VA and context without cycles");
    Fixture owned;
    const auto address = owned.backend->reserve_address_space(ByteSize{64U}, ByteSize{64U}).value();
    VRAMZ_CHECK(runner, owned.backend->allocate_representation(
                            {RepresentationState::gpu_raw, ByteSize{64U}, {}, address.base}));
    owned.backend.reset();
    VRAMZ_CHECK(runner, owned.audit->owned_bytes == 0U && owned.audit->primary_references == 0U);
    VRAMZ_CHECK(runner, owned.audit->destroyed_drivers == 1U);
}

// A child-specific terminate handler observes only independent fixed-size audit/error state.
// It never reacquires the backend registry mutex held at a fatal boundary.
struct DeathOptions final {
    std::uint64_t detail{};
    std::uint64_t after_calls{1U};
    std::uint64_t companion_after{1U};
};
void death(test::Runner& runner, CudaCall point, CudaFaultMode fault_mode,
           CudaCall companion = CudaCall::count, DeathOptions options = {}) {
    const auto child = ::fork();
    VRAMZ_CHECK(runner, child >= 0);
    if (child < 0) {
        return;
    }
    if (child == 0) {
        static_cast<void>(::alarm(5U));
        Fixture fixture;
        if (!fixture.backend) {
            std::_Exit(1);
        }
        static CudaVmmBackend* child_backend{};
        static test::FakeCudaState* audit{};
        static Runtime* child_runtime{};
        child_backend = fixture.backend.get();
        audit = fixture.audit.get();
        auto created = testing::RuntimeAccess::create_with_backend(runtime_config(),
                                                                   std::move(fixture.backend));
        if (!created) {
            std::_Exit(1);
        }
        auto runtime = std::move(created).value();
        child_runtime = &runtime;
        std::set_terminate([]() noexcept {
            const auto stats = child_runtime->stats();
            const auto backend_error = child_backend->first_error();
            const auto core_error = child_runtime->first_async_error();
            const bool stopped =
                stats.gpu.committed == ByteSize{} && stats.gpu.cleanup_debt == ByteSize{} &&
                stats.gpu.staging == ByteSize{} && stats.gpu.workspace == ByteSize{} &&
                audit->counts[static_cast<std::size_t>(CudaCall::free_address)] == 0U &&
                ((backend_error &&
                  backend_error->error.code == ErrorCode::backend_contract_violation) ||
                 (core_error && core_error->error.code == ErrorCode::backend_contract_violation));
            std::_Exit(stopped ? 86 : 1);
        });
        fixture.driver->inject(point, fault_mode, {options.after_calls, options.detail});
        if (companion != CudaCall::count) {
            fixture.driver->inject(companion, CudaFaultMode::before, {options.companion_after, 0U});
        }
        static_cast<void>(runtime.allocate(ByteSize{64U}));
        std::_Exit(1);
    }
    int status{};
    pid_t waited{};
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    VRAMZ_CHECK(runner, waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 86);
}

void fault_tests(test::Runner& runner) {
    for (const auto point :
         {CudaCall::reserve, CudaCall::create, CudaCall::map, CudaCall::set_access,
          CudaCall::retain_mapping, CudaCall::push, CudaCall::pop}) {
        runner.begin("ambiguous mutation fails stop before provisional allocation publication");
        death(runner, point, CudaFaultMode::after);
    }
    for (std::uint64_t variant = 0U; variant < 5U; ++variant) {
        runner.begin("allocation properties must corroborate device type export and compression");
        death(runner, CudaCall::properties, CudaFaultMode::malformed, CudaCall::count,
              {.detail = variant});
    }
    for (const auto point : {CudaCall::reserve, CudaCall::create, CudaCall::get_access,
                             CudaCall::retain_mapping, CudaCall::pop, CudaCall::set_access}) {
        runner.begin("malformed address handle access or context output cannot publish authority");
        death(runner, point, CudaFaultMode::malformed);
    }
    runner.begin("adoption corroborates mapping again after provisional success");
    death(runner, CudaCall::properties, CudaFaultMode::malformed, CudaCall::count,
          {.after_calls = 2U});
    runner.begin("contradictory nonzero retained handle is never used as a cleanup target");
    death(runner, CudaCall::retain_mapping, CudaFaultMode::malformed, CudaCall::count,
          {.detail = 1U});
    runner.begin(
        "plausible false VA and contradictory map rejection cannot retain guessed ownership");
    death(runner, CudaCall::reserve, CudaFaultMode::malformed, CudaCall::count, {.detail = 1U});

    runner.begin(
        "bounded pairwise creation faults and cleanup retain either exact accounting or fail stop");
    for (const auto primary : {CudaCall::map, CudaCall::set_access, CudaCall::host_to_device,
                               CudaCall::device_to_host}) {
        for (const auto cleanup : {CudaCall::unmap, CudaCall::release}) {
            // A failed map never established a mapping, so unmap is not attempted.
            if (primary == CudaCall::map && cleanup == CudaCall::unmap) {
                Fixture fixture;
                auto created = testing::RuntimeAccess::create_with_backend(
                    runtime_config(), std::move(fixture.backend));
                VRAMZ_CHECK(runner, created);
                if (!created) {
                    return;
                }
                auto runtime = std::move(created).value();
                fixture.driver->inject(primary);
                fixture.driver->inject(cleanup);
                VRAMZ_CHECK(runner, !runtime.allocate(ByteSize{64U}));
                VRAMZ_CHECK(runner, fixture.driver->count(CudaCall::unmap) == 0U);
                conserved(runner, runtime, fixture);
                clean(runner, runtime, fixture);
                continue;
            }
            // A validation reference is released before byte copies; target final cleanup.
            const auto release_after =
                primary == CudaCall::host_to_device || primary == CudaCall::device_to_host ? 2U
                                                                                           : 1U;
            death(runner, primary, CudaFaultMode::before, cleanup,
                  {.companion_after = cleanup == CudaCall::release ? release_after : 1U});
        }
    }
    for (const auto point : {CudaCall::create, CudaCall::map, CudaCall::set_access,
                             CudaCall::host_to_device, CudaCall::device_to_host}) {
        runner.begin(
            "definite pre-mutation allocation failure cleans private resources without debt");
        Fixture fixture;
        auto created = testing::RuntimeAccess::create_with_backend(runtime_config(),
                                                                   std::move(fixture.backend));
        auto runtime = std::move(created).value();
        fixture.driver->inject(point);
        VRAMZ_CHECK(runner, !runtime.allocate(ByteSize{64U}));
        conserved(runner, runtime, fixture);
        VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == ByteSize{});
        clean(runner, runtime, fixture);
    }
}

void cleanup_death_tests(test::Runner& runner) {
    for (const auto point :
         {CudaCall::unmap, CudaCall::release, CudaCall::free_address, CudaCall::release_primary}) {
        runner.begin("ambiguous cleanup mutation cannot return or publish successful shutdown");
        const auto child = ::fork();
        VRAMZ_CHECK(runner, child >= 0);
        if (child < 0) {
            return;
        }
        if (child == 0) {
            static_cast<void>(::alarm(5U));
            Fixture fixture;
            static CudaVmmBackend* backend{};
            static test::FakeCudaState* audit{};
            static CudaCall fault_point{};
            backend = fixture.backend.get();
            audit = fixture.audit.get();
            fault_point = point;
            auto created = testing::RuntimeAccess::create_with_backend(runtime_config(),
                                                                       std::move(fixture.backend));
            if (!created) {
                std::_Exit(1);
            }
            auto runtime = std::move(created).value();
            auto allocated = runtime.allocate(ByteSize{64U});
            if (!allocated) {
                std::_Exit(1);
            }
            auto buffer = std::move(allocated).value();
            std::set_terminate([]() noexcept {
                const auto error = backend->first_error();
                const auto frees = audit->counts[static_cast<std::size_t>(CudaCall::free_address)];
                const bool stopped = error &&
                                     error->error.code == ErrorCode::backend_contract_violation &&
                                     (fault_point == CudaCall::free_address ||
                                              fault_point == CudaCall::release_primary
                                          ? frees == 1U
                                          : frees == 0U);
                std::_Exit(stopped ? 86 : 1);
            });
            fixture.driver->inject(point, CudaFaultMode::after);
            static_cast<void>(buffer.close());
            static_cast<void>(runtime.shutdown());
            std::_Exit(1);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        VRAMZ_CHECK(runner, waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 86);
    }
}

void mapping_rules(test::Runner& runner) {
    runner.begin("full-map access and unmap boundaries are enforced and remapping clears access");
    Fixture fixture;
    auto& driver = *fixture.driver;
    VRAMZ_CHECK(runner, driver.push_context(detail::CudaContext{1U}));
    const auto address = driver.reserve({ByteSize{128U}, ByteSize{64U}}).value();
    const auto handle = driver.create(ByteSize{128U}, 0).value();
    VRAMZ_CHECK(runner, driver.map(address, ByteSize{128U}, handle));
    std::array<std::byte, 64U> bytes{};
    VRAMZ_CHECK(runner, !driver.copy_from_device(bytes, address));
    VRAMZ_CHECK(runner, !driver.set_access(address, ByteSize{64U}, 0));
    VRAMZ_CHECK(runner, driver.set_access(address, ByteSize{128U}, 0));
    VRAMZ_CHECK(runner, !driver.unmap(address, ByteSize{64U}));
    VRAMZ_CHECK(runner, !driver.release(handle));
    VRAMZ_CHECK(runner, !driver.free_address(address, ByteSize{128U}));
    VRAMZ_CHECK(runner, driver.unmap(address, ByteSize{128U}));
    VRAMZ_CHECK(runner, driver.map(address, ByteSize{128U}, handle));
    VRAMZ_CHECK(runner, !driver.has_read_write_access(address, 0).value());
    VRAMZ_CHECK(runner, !driver.copy_from_device(bytes, address));
    VRAMZ_CHECK(runner, driver.set_access(address, ByteSize{128U}, 0));
    VRAMZ_CHECK(runner, driver.copy_from_device(bytes, address));
    VRAMZ_CHECK(runner, driver.unmap(address, ByteSize{128U}));
    VRAMZ_CHECK(runner, driver.release(handle));
    VRAMZ_CHECK(runner, driver.free_address(address, ByteSize{128U}));
    VRAMZ_CHECK(runner, driver.pop_context());
    VRAMZ_CHECK(runner, fixture.backend->shutdown());
    VRAMZ_CHECK(runner, driver.empty());
}

void runtime_configuration_tests(test::Runner& runner) {
    runner.begin(
        "preferred chunks preserve general multiples instead of imposing power-of-two sizes");
    Fixture fixture;
    auto config = runtime_config();
    config.preferred_chunk_size = ByteSize{192U};
    auto created = testing::RuntimeAccess::create_with_backend(config, std::move(fixture.backend));
    VRAMZ_CHECK(runner, created);
    if (!created) {
        return;
    }
    auto runtime = std::move(created).value();
    auto allocated = runtime.allocate(ByteSize{193U});
    VRAMZ_CHECK(runner, allocated);
    if (!allocated) {
        return;
    }
    auto buffer = std::move(allocated).value();
    VRAMZ_CHECK(runner, runtime.stats().gpu.committed == ByteSize{256U});
    VRAMZ_CHECK(runner,
                testing::chunk_snapshot(buffer, 1U).value().authoritative.charge == ByteSize{64U});
    VRAMZ_CHECK(runner, buffer.close());
    clean(runner, runtime, fixture);

    for (const auto mode :
         {PolicyMode::gpu_resident, PolicyMode::gpu_resident_with_host_fallback}) {
        runner.begin(
            "unsupported automatic compression policy rejected before resource/VA mutation");
        Fixture rejected;
        auto options = runtime_config();
        options.policy.mode = mode;
        const auto result =
            testing::RuntimeAccess::create_with_backend(options, std::move(rejected.backend));
        VRAMZ_CHECK(runner, !result && result.error().code == ErrorCode::unsupported);
        VRAMZ_CHECK(runner,
                    rejected.audit->counts[static_cast<std::size_t>(CudaCall::reserve)] == 0U);
        VRAMZ_CHECK(runner,
                    rejected.audit->owned_bytes == 0U && rejected.audit->primary_references == 0U);
    }
}

void runtime_model(test::Runner& runner) {
    runner.begin(
        "bounded deterministic Runtime allocation access cleanup model conserves each phase");
    Fixture fixture;
    auto created =
        testing::RuntimeAccess::create_with_backend(runtime_config(), std::move(fixture.backend));
    VRAMZ_CHECK(runner, created);
    if (!created) {
        return;
    }
    auto runtime = std::move(created).value();
    std::array<std::optional<Buffer>, 8U> buffers{};
    std::uint32_t random = 0x4D34564DU;
    for (std::uint32_t iteration = 0U; iteration < 128U; ++iteration) {
        // Intentional modulo mixing is a deterministic generator, not size arithmetic.
        random = random * 1664525U + 1013904223U;
        auto& slot = buffers[random % buffers.size()];
        if (!slot) {
            const ByteSize size{64U + (random % 137U)};
            auto allocated = runtime.allocate(size);
            VRAMZ_CHECK(runner, allocated);
            if (allocated) {
                slot.emplace(std::move(allocated).value());
            }
        } else if ((random & 32U) != 0U) {
            VRAMZ_CHECK(runner, slot->close());
            slot.reset();
        } else {
            auto acquired = slot->acquire({ByteOffset{}, slot->size()}, {AccessMode::read_write});
            VRAMZ_CHECK(runner, acquired);
            if (acquired) {
                auto lease = std::move(acquired).value();
                const auto bytes = payload();
                std::array<std::byte, 256U> read{};
                const auto size = static_cast<std::size_t>(slot->size().value());
                VRAMZ_CHECK(runner, testing::write_bytes(lease, ByteOffset{},
                                                         std::span{bytes}.first(size)));
                VRAMZ_CHECK(runner,
                            testing::read_bytes(lease, ByteOffset{}, std::span{read}.first(size)));
                VRAMZ_CHECK(runner, std::ranges::equal(std::span{bytes}.first(size),
                                                       std::span{read}.first(size)));
                VRAMZ_CHECK(runner, lease.close());
            }
        }
        conserved(runner, runtime, fixture);
        VRAMZ_CHECK(runner, runtime.stats().gpu.cleanup_debt == ByteSize{});
    }
    for (auto& slot : buffers) {
        if (slot) {
            VRAMZ_CHECK(runner, slot->close());
            slot.reset();
        }
    }
    clean(runner, runtime, fixture);
}

void concurrency_test(test::Runner& runner) {
    runner.begin(
        "bounded Runtime allocate/read/close with concurrent statistics and caller contexts");
    Fixture fixture;
    auto created =
        testing::RuntimeAccess::create_with_backend(runtime_config(), std::move(fixture.backend));
    auto runtime = std::move(created).value();
    std::atomic<bool> valid{true};
    std::array<std::thread, 3U> workers;
    for (auto& worker : workers) {
        worker = std::thread{[&]() {
            for (std::uint32_t iteration = 0U; iteration < 32U; ++iteration) {
                auto allocated = runtime.allocate(ByteSize{64U});
                if (!allocated) {
                    valid = false;
                    continue;
                }
                auto buffer = std::move(allocated).value();
                auto acquired = buffer.acquire({ByteOffset{}, ByteSize{64U}});
                if (!acquired) {
                    valid = false;
                    continue;
                }
                auto lease = std::move(acquired).value();
                std::array<std::byte, 64U> bytes{};
                if (!testing::read_bytes(lease, ByteOffset{}, bytes) || !lease.close() ||
                    !buffer.close()) {
                    valid = false;
                }
                if (fixture.driver->current_context() != detail::CudaContext{}) {
                    valid = false;
                }
                static_cast<void>(runtime.stats());
            }
        }};
    }
    for (auto& worker : workers) {
        worker.join();
    }
    VRAMZ_CHECK(runner, valid.load());
    clean(runner, runtime, fixture);
}

} // namespace

int main() {
    test::Runner runner;
    capability_tests(runner);
    reservation_tests(runner);
    remap_oracle(runner);
    runtime_oracle(runner);
    writable_close_oracle(runner);
    cleanup_tests(runner);
    context_tests(runner);
    fault_tests(runner);
    cleanup_death_tests(runner);
    mapping_rules(runner);
    runtime_configuration_tests(runner);
    runtime_model(runner);
    concurrency_test(runner);
    return runner.finish();
}
