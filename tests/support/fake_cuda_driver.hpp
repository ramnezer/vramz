#pragma once

#include "vramz/config.hpp"
#include "vramz/detail/cuda_driver.hpp"

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace vramz::test {

enum class CudaCall : std::uint8_t {
    initialize,
    version,
    device,
    uva,
    vmm,
    host_page,
    minimum,
    recommended,
    retain_primary,
    release_primary,
    push,
    pop,
    reserve,
    free_address,
    create,
    properties,
    map,
    set_access,
    get_access,
    retain_mapping,
    unmap,
    release,
    host_to_device,
    device_to_host,
    device_to_device,
    completion_create,
    completion_query,
    completion_release,
    device_info,
    count
};
enum class CudaFaultMode : std::uint8_t { before, after, malformed };
struct CudaFaultOptions final {
    std::uint64_t after_calls{1U};
    std::uint64_t detail{};
};
struct CudaFault final {
    CudaCall call{CudaCall::count};
    std::uint64_t occurrence{};
    CudaFaultMode mode{};
    std::uint64_t detail{};
};
struct CudaCallRecord final {
    CudaCall call{};
    std::uint64_t identity{};
    ByteSize size{};
};
struct FakeCudaConfig final {
    std::int32_t version{13030};
    bool unified_addressing{true};
    bool vmm{true};
    bool external_completions{true};
    ByteSize minimum{64U};
    ByteSize recommended{256U};
    ByteSize host_page{64U};
    ByteSize physical_limit{8U * 1024U * 1024U};
    ByteSize maximum_resource{2U * 1024U * 1024U};
    detail::CudaDeviceInfo device_info{std::array<char, 256U>{"VRAMZ Fake NVIDIA device"}, 0};
    std::int64_t initialize_native_error{};
};

// External audit ownership survives destruction of the driver. It never owns the backend.
// All state, faults, calls and context stacks are bounded; byte payloads have a hard limit.
struct FakeCudaState final {
    struct Reservation final {
        DeviceAddress address{};
        ByteSize size{};
    };
    struct Physical final {
        detail::CudaPhysicalHandle handle{};
        std::vector<std::byte> bytes{};
        std::uint32_t references{};
        detail::CudaAllocationProperties properties{};
    };
    struct Mapping final {
        DeviceAddress address{};
        ByteSize size{};
        detail::CudaPhysicalHandle handle{};
        bool read_write{};
    };
    struct ThreadContext final {
        std::thread::id thread{};
        std::array<detail::CudaContext, 8U> stack{};
        std::size_t depth{};
    };
    mutable std::mutex mutex{};
    struct Completion final {
        CompletionTokenId id{};
        CompletionState state{CompletionState::pending};
    };
    std::array<Completion, max_pending_completions> completions{};
    std::uint64_t next_completion_id{1U};
    bool initialized{};
    FakeCudaConfig config{};
    std::array<Reservation, 64U> reservations{};
    std::array<Physical, 64U> physical{};
    std::array<Mapping, 64U> mappings{};
    std::array<ThreadContext, 32U> contexts{};
    std::array<CudaFault, 16U> faults{};
    std::array<std::uint64_t, static_cast<std::size_t>(CudaCall::count)> counts{};
    std::array<CudaCallRecord, 16384U> log{};
    std::size_t log_size{};
    std::uint64_t dropped_calls{};
    std::uint64_t next_address{0x100000000ULL};
    std::uint64_t next_handle{1U};
    std::uint64_t owned_bytes{};
    std::uint32_t primary_references{};
    std::uint32_t destroyed_drivers{};
};

class FakeCudaDriverApi final : public detail::CudaDriverApi {
  public:
    explicit FakeCudaDriverApi(std::shared_ptr<FakeCudaState> state) noexcept;
    ~FakeCudaDriverApi() override;
    FakeCudaDriverApi(const FakeCudaDriverApi&) = delete;
    FakeCudaDriverApi& operator=(const FakeCudaDriverApi&) = delete;
    FakeCudaDriverApi(FakeCudaDriverApi&&) = delete;
    FakeCudaDriverApi& operator=(FakeCudaDriverApi&&) = delete;

    void inject(CudaCall call, CudaFaultMode mode = CudaFaultMode::before,
                CudaFaultOptions options = {}) noexcept;
    [[nodiscard]] std::uint64_t count(CudaCall call) const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] bool supports_external_completions() const noexcept override;
    [[nodiscard]] Result<CompletionTokenId> create_completion() noexcept override;
    [[nodiscard]] Result<CompletionState> query_completion(CompletionTokenId id) noexcept override;
    [[nodiscard]] Result<void> release_completion(CompletionTokenId id) noexcept override;
    [[nodiscard]] Result<void> mark_completion_ready(CompletionTokenId id) noexcept;
    [[nodiscard]] detail::CudaContext current_context() const noexcept;
    [[nodiscard]] Result<void> initialize() noexcept override;
    [[nodiscard]] Result<std::int32_t> version() noexcept override;
    [[nodiscard]] Result<detail::CudaDevice> device(std::int32_t ordinal) noexcept override;
    [[nodiscard]] Result<detail::CudaDeviceInfo>
    device_info(detail::CudaDevice device) noexcept override;
    [[nodiscard]] Result<bool> attribute(detail::CudaDevice device,
                                         detail::CudaAttribute attribute) noexcept override;
    [[nodiscard]] Result<ByteSize> host_page_size() noexcept override;
    [[nodiscard]] Result<ByteSize> granularity(detail::CudaDevice device,
                                               bool recommended) noexcept override;
    [[nodiscard]] Result<detail::CudaContext>
    retain_primary(detail::CudaDevice device) noexcept override;
    [[nodiscard]] Result<void> release_primary(detail::CudaDevice device) noexcept override;
    [[nodiscard]] Result<void> push_context(detail::CudaContext context) noexcept override;
    [[nodiscard]] Result<detail::CudaContext> pop_context() noexcept override;
    [[nodiscard]] Result<bool> context_is_current(detail::CudaContext context) noexcept override;
    [[nodiscard]] Result<DeviceAddress>
    reserve(detail::CudaAddressRequest request) noexcept override;
    [[nodiscard]] Result<void> free_address(DeviceAddress address, ByteSize size) noexcept override;
    [[nodiscard]] Result<detail::CudaPhysicalHandle>
    create(ByteSize size, detail::CudaDevice device) noexcept override;
    [[nodiscard]] Result<detail::CudaAllocationProperties>
    properties(detail::CudaPhysicalHandle handle) noexcept override;
    [[nodiscard]] Result<void> map(DeviceAddress address, ByteSize size,
                                   detail::CudaPhysicalHandle handle) noexcept override;
    [[nodiscard]] Result<void> set_access(DeviceAddress address, ByteSize size,
                                          detail::CudaDevice device) noexcept override;
    [[nodiscard]] Result<bool> has_read_write_access(DeviceAddress address,
                                                     detail::CudaDevice device) noexcept override;
    [[nodiscard]] Result<detail::CudaPhysicalHandle>
    retain_mapping(DeviceAddress address) noexcept override;
    [[nodiscard]] Result<void> unmap(DeviceAddress address, ByteSize size) noexcept override;
    [[nodiscard]] Result<void> release(detail::CudaPhysicalHandle handle) noexcept override;
    [[nodiscard]] Result<void> copy_to_device(DeviceAddress destination,
                                              std::span<const std::byte> source) noexcept override;
    [[nodiscard]] Result<void> copy_from_device(std::span<std::byte> destination,
                                                DeviceAddress source) noexcept override;
    [[nodiscard]] Result<void> copy_device_to_device(DeviceAddress destination,
                                                     DeviceAddress source,
                                                     ByteSize size) noexcept override;

  private:
    [[nodiscard]] std::optional<CudaFault> call(CudaCall point, std::uint64_t identity = 0U,
                                                ByteSize size = {}) noexcept;
    [[nodiscard]] FakeCudaState::Physical* physical(detail::CudaPhysicalHandle handle) noexcept;
    [[nodiscard]] FakeCudaState::Mapping* mapping(DeviceAddress address) noexcept;
    [[nodiscard]] FakeCudaState::ThreadContext* thread_context() noexcept;
    [[nodiscard]] bool valid_context() noexcept;
    [[nodiscard]] Result<void> copy(DeviceAddress address, std::span<std::byte> destination,
                                    std::span<const std::byte> source, bool to_device) noexcept;
    std::shared_ptr<FakeCudaState> state_;
};

} // namespace vramz::test
