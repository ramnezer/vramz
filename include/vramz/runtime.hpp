#pragma once

#include "vramz/capabilities.hpp"
#include "vramz/completion.hpp"
#include "vramz/config.hpp"
#include "vramz/performance.hpp"
#include "vramz/stats.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace vramz {

class StorageBackend;
struct CudaRuntimeOptions;
struct CudaRuntimeInfo;

namespace testing {
class RuntimeAccess;
} // namespace testing

namespace detail {
struct RuntimeState;
struct BufferState;
} // namespace detail

enum class AccessMode : std::uint8_t { read_only, read_write };
enum class OperationStatus : std::uint8_t { succeeded, failed, cancelled };

struct AcquireOptions final {
    AccessMode access{AccessMode::read_only};
};

struct BufferStats final {
    ByteSize logical_size{};
    std::uint64_t chunk_count{};
    std::uint64_t active_read_leases{};
    std::uint64_t active_write_leases{};
    bool closing{};
    bool poisoned{};
};

struct RuntimeStats final {
    TierUsage gpu{};
    TierUsage host{};
    AsyncErrorStats async_errors{};
    CompressionStats compression{};
    std::uint64_t live_buffers{};
    std::uint64_t quarantined_buffers{};
    bool shutting_down{};
    PolicyStats policy{};
    CompletionStats completions{};
};

enum class Residency : std::uint8_t { gpu_raw, gpu_compressed, host_raw, host_compressed };

struct ChunkInfo final {
    std::uint64_t index{};
    ByteSize logical_bytes{};
    ByteSize stored_bytes{};
    ByteSize physical_charge{};
    std::uint32_t logical_crc{};
    std::uint32_t stored_crc{};
    Residency residency{};
    std::uint64_t access_revision{};
    std::uint64_t access_epoch{};
    std::uint64_t access_count{};
    std::uint64_t retry_after{};
    std::uint64_t compression_attempts{};
    std::uint32_t effective_frequency{};
    Temperature temperature{};
    Compressibility compressibility{};
    bool policy_metadata_available{};
    bool pending{};
    bool poisoned{};
};

struct ResourceStats final {
    ByteSize owned_gpu_charge{};
    ByteSize ledger_gpu_charge{};
    ByteSize unmaterialized_reservations{};
    std::uint64_t backend_resources{};
    std::uint64_t va_reservations{};
    std::uint64_t raw_resources{};
    std::uint64_t compressed_resources{};
    std::uint64_t workspace_resources{};
    bool retained_context{};
    bool stream_owned{};
    bool accounting_conserved{};
};

class PendingLeaseRelease;

class Lease final {
  public:
    Lease() noexcept = default;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) = delete;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease() noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] MemoryRange range() const noexcept;
    [[nodiscard]] AccessMode access_mode() const noexcept;
    [[nodiscard]] Result<DeviceSpan> device_span() const noexcept;
    // Synchronous, range-checked copies through the authoritative leased resource.
    [[nodiscard]] Result<void> read(ByteOffset offset, std::span<std::byte> output) const noexcept;
    [[nodiscard]] Result<void> write(ByteOffset offset, std::span<const std::byte> input) noexcept;
    [[nodiscard]] Result<void> close() noexcept;
    [[nodiscard]] Result<void> close_and_wait() noexcept;
    // Reserve tracking BEFORE external work may be submitted. On failure this Lease is intact.
    [[nodiscard]] Result<PendingLeaseRelease> defer() noexcept;

  private:
    friend class Buffer;
    friend class Runtime;
    friend class testing::RuntimeAccess;
    Lease(std::shared_ptr<detail::RuntimeState> runtime, std::shared_ptr<detail::BufferState> state,
          MemoryRange range, AccessMode access, std::unique_ptr<std::uint64_t[]> chunk_indices,
          std::unique_ptr<ContentTag[]> initial_contents, std::uint64_t chunk_count) noexcept;

    std::shared_ptr<detail::RuntimeState> runtime_{};
    std::shared_ptr<detail::BufferState> state_{};
    MemoryRange range_{};
    AccessMode access_{AccessMode::read_only};
    std::unique_ptr<std::uint64_t[]> chunk_indices_{};
    std::unique_ptr<ContentTag[]> initial_contents_{};
    std::uint64_t chunk_count_{};
    bool active_{};
};

// Move-only observer of a Runtime-owned pin/fence obligation, not a self-owning Lease.
// Destroying this handle never unpins. Explicit Runtime polling drives bounded completion.
class PendingLeaseRelease final {
  public:
    PendingLeaseRelease(PendingLeaseRelease&&) noexcept = default;
    PendingLeaseRelease& operator=(PendingLeaseRelease&&) = delete;
    PendingLeaseRelease(const PendingLeaseRelease&) = delete;
    PendingLeaseRelease& operator=(const PendingLeaseRelease&) = delete;
    ~PendingLeaseRelease() noexcept = default;
    [[nodiscard]] bool empty() const noexcept { return !runtime_; }
    [[nodiscard]] CompletionToken token() const noexcept {
        return runtime_ ? token_ : CompletionToken{};
    }

  private:
    friend class Lease;
    PendingLeaseRelease(std::shared_ptr<detail::RuntimeState> runtime,
                        CompletionToken token) noexcept
        : runtime_(std::move(runtime)), token_(token) {}
    std::shared_ptr<detail::RuntimeState> runtime_{};
    CompletionToken token_{};
};

class PendingLease final {
  public:
    PendingLease(PendingLease&& other) noexcept;
    PendingLease& operator=(PendingLease&&) = delete;
    PendingLease(const PendingLease&) = delete;
    PendingLease& operator=(const PendingLease&) = delete;
    ~PendingLease() noexcept = default;

    [[nodiscard]] OperationStatus status() const noexcept;
    [[nodiscard]] Result<std::optional<Lease>> poll() noexcept;
    [[nodiscard]] Result<Lease> wait() noexcept;
    [[nodiscard]] Result<void> cancel() noexcept;

  private:
    friend class Buffer;
    explicit PendingLease(Lease lease) noexcept;
    Lease lease_{};
    bool has_lease_{};
    bool consumed_{};
};

class Operation final {
  public:
    Operation() noexcept = default;
    Operation(const Operation&) = delete;
    Operation& operator=(const Operation&) = delete;
    Operation(Operation&&) noexcept = default;
    Operation& operator=(Operation&&) noexcept = default;
    ~Operation() noexcept = default;
    [[nodiscard]] OperationStatus status() const noexcept { return status_; }
    [[nodiscard]] Result<void> wait() const noexcept;
    [[nodiscard]] Result<void> cancel() noexcept;

  private:
    friend class Buffer;
    explicit Operation(Result<void> result) noexcept;
    OperationStatus status_{OperationStatus::succeeded};
    Error error_{};
    bool has_error_{};
};

class Buffer final {
  public:
    Buffer() noexcept = default;
    Buffer(Buffer&&) noexcept = default;
    Buffer& operator=(Buffer&&) = delete;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() noexcept;

    [[nodiscard]] ByteSize size() const noexcept;
    [[nodiscard]] BufferId id() const noexcept;
    [[nodiscard]] Result<Lease> acquire(MemoryRange range, AcquireOptions options = {}) noexcept;
    [[nodiscard]] Result<PendingLease> acquire_async(MemoryRange range,
                                                     AcquireOptions options = {}) noexcept;
    [[nodiscard]] Result<Operation> prefetch(MemoryRange range) noexcept;
    [[nodiscard]] BufferStats stats() const noexcept;
    [[nodiscard]] Result<ChunkInfo> inspect_chunk(std::uint64_t index) const noexcept;
    [[nodiscard]] Result<void> close() noexcept;

  private:
    friend class Runtime;
    friend class testing::RuntimeAccess;
    Buffer(std::shared_ptr<detail::RuntimeState> runtime,
           std::shared_ptr<detail::BufferState> state) noexcept;
    std::shared_ptr<detail::RuntimeState> runtime_{};
    std::shared_ptr<detail::BufferState> state_{};
};

class Runtime final {
  public:
    Runtime() noexcept = default;
    Runtime(Runtime&&) noexcept = default;
    Runtime& operator=(Runtime&&) = delete;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    ~Runtime() noexcept;

    [[nodiscard]] static Result<Runtime> create(RuntimeConfig config) noexcept;
    // Defined by the separately opt-in VRAMZ::cuda library. No implicit device selection.
    [[nodiscard]] static Result<Runtime> create_cuda(RuntimeConfig config,
                                                     const CudaRuntimeOptions& options,
                                                     CudaRuntimeInfo& info) noexcept;
    [[nodiscard]] Result<ResourceStats> resources() const noexcept;
    [[nodiscard]] PerformanceStats performance() const noexcept;
    [[nodiscard]] Result<Buffer> allocate(ByteSize logical_size) noexcept;
    [[nodiscard]] Result<void> reclaim_to_target(ByteSize target_gpu_charge) noexcept;
    [[nodiscard]] RuntimeStats stats() const noexcept;
    [[nodiscard]] RuntimeCapabilities capabilities() const noexcept;
    [[nodiscard]] std::optional<AsyncErrorRecord> poll_async_error() noexcept;
    [[nodiscard]] std::optional<AsyncErrorRecord> first_async_error() const noexcept;
    [[nodiscard]] Result<void> shutdown() noexcept;
    // One bounded pass, no waiting. A pending token returns false, never releases a pin.
    [[nodiscard]] Result<bool> poll_completion(CompletionToken token) noexcept;
    [[nodiscard]] Result<std::uint32_t> poll_ready_completions() noexcept;

  private:
    friend class testing::RuntimeAccess;
    [[nodiscard]] static Result<Runtime>
    create_backend(RuntimeConfig config, std::unique_ptr<StorageBackend> backend) noexcept;
    explicit Runtime(std::shared_ptr<detail::RuntimeState> state) noexcept;
    std::shared_ptr<detail::RuntimeState> state_{};
};

} // namespace vramz
