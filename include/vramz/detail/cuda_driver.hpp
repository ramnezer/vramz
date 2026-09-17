#pragma once

#include "vramz/completion.hpp"
#include "vramz/result.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace vramz::detail {

struct CudaContextTag;
struct CudaPhysicalHandleTag;
using CudaContext = StrongValue<CudaContextTag, std::uintptr_t>;
using CudaPhysicalHandle = StrongValue<CudaPhysicalHandleTag, std::uint64_t>;
using CudaDevice = std::int32_t;

enum class CudaAttribute : std::uint8_t { unified_addressing, virtual_memory_management };

struct CudaDeviceInfo final {
    std::array<char, 256U> name{};
    // Read-only CUDA compute mode: 0 = default, 2 = prohibited, 3 = exclusive process.
    std::int32_t compute_mode{};
};

struct CudaAddressRequest final {
    ByteSize size{};
    ByteSize alignment{};
};

struct CudaAllocationProperties final {
    CudaDevice device{};
    bool device_local{true};
    bool pinned{true};
    bool no_export{true};
    bool compression_disabled{true};
    auto operator<=>(const CudaAllocationProperties&) const = default;
};

// No NVIDIA types cross this boundary. A mutation returning an ordinary Error proves no
// mutation occurred. If that cannot be proved, it returns ambiguous_backend_state instead.
// Output handles/addresses are delivered without allocating after successful mutation.
class CudaDriverApi {
  public:
    virtual ~CudaDriverApi() = default;
    // Internal fake-event contract, not arbitrary application-stream support.
    // The real Driver adapter intentionally keeps these defaults in M6.
    [[nodiscard]] virtual bool supports_external_completions() const noexcept { return false; }
    [[nodiscard]] virtual Result<CompletionTokenId> create_completion() noexcept {
        return make_error(ErrorCode::unsupported, OperationId::defer_lease);
    }
    [[nodiscard]] virtual Result<CompletionState> query_completion(CompletionTokenId) noexcept {
        return make_error(ErrorCode::unsupported, OperationId::query_completion);
    }
    [[nodiscard]] virtual Result<void> release_completion(CompletionTokenId) noexcept {
        return make_error(ErrorCode::unsupported, OperationId::release_completion);
    }
    [[nodiscard]] virtual Result<void> initialize() noexcept = 0;
    [[nodiscard]] virtual Result<std::int32_t> version() noexcept = 0;
    [[nodiscard]] virtual Result<CudaDevice> device(std::int32_t ordinal) noexcept = 0;
    // Explicit read-only telemetry after initialize/device selection; never initializes CUDA.
    [[nodiscard]] virtual Result<CudaDeviceInfo> device_info(CudaDevice device) noexcept = 0;
    [[nodiscard]] virtual Result<bool> attribute(CudaDevice device,
                                                 CudaAttribute attribute) noexcept = 0;
    [[nodiscard]] virtual Result<ByteSize> host_page_size() noexcept = 0;
    [[nodiscard]] virtual Result<ByteSize> granularity(CudaDevice device,
                                                       bool recommended) noexcept = 0;
    [[nodiscard]] virtual Result<CudaContext> retain_primary(CudaDevice device) noexcept = 0;
    [[nodiscard]] virtual Result<void> release_primary(CudaDevice device) noexcept = 0;
    [[nodiscard]] virtual Result<void> push_context(CudaContext context) noexcept = 0;
    [[nodiscard]] virtual Result<CudaContext> pop_context() noexcept = 0;
    [[nodiscard]] virtual Result<bool> context_is_current(CudaContext context) noexcept = 0;
    [[nodiscard]] virtual Result<DeviceAddress> reserve(CudaAddressRequest request) noexcept = 0;
    [[nodiscard]] virtual Result<void> free_address(DeviceAddress address,
                                                    ByteSize size) noexcept = 0;
    [[nodiscard]] virtual Result<CudaPhysicalHandle> create(ByteSize size,
                                                            CudaDevice device) noexcept = 0;
    [[nodiscard]] virtual Result<CudaAllocationProperties>
    properties(CudaPhysicalHandle handle) noexcept = 0;
    [[nodiscard]] virtual Result<void> map(DeviceAddress address, ByteSize size,
                                           CudaPhysicalHandle handle) noexcept = 0;
    [[nodiscard]] virtual Result<void> set_access(DeviceAddress address, ByteSize size,
                                                  CudaDevice device) noexcept = 0;
    [[nodiscard]] virtual Result<bool> has_read_write_access(DeviceAddress address,
                                                             CudaDevice device) noexcept = 0;
    [[nodiscard]] virtual Result<CudaPhysicalHandle>
    retain_mapping(DeviceAddress address) noexcept = 0;
    [[nodiscard]] virtual Result<void> unmap(DeviceAddress address, ByteSize size) noexcept = 0;
    [[nodiscard]] virtual Result<void> release(CudaPhysicalHandle handle) noexcept = 0;
    [[nodiscard]] virtual Result<void>
    copy_to_device(DeviceAddress destination, std::span<const std::byte> source) noexcept = 0;
    [[nodiscard]] virtual Result<void> copy_from_device(std::span<std::byte> destination,
                                                        DeviceAddress source) noexcept = 0;
    [[nodiscard]] virtual Result<void> copy_device_to_device(DeviceAddress destination,
                                                             DeviceAddress source,
                                                             ByteSize size) noexcept = 0;
};

// This factory only constructs an adapter. It never initializes or probes a CUDA device.
// Defined exclusively by the opt-in compile/link artifact, never linked into normal tests.
[[nodiscard]] Result<std::unique_ptr<CudaDriverApi>> make_real_cuda_driver() noexcept;

} // namespace vramz::detail
