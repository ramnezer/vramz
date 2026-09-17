#pragma once

#include "vramz/async_error_channel.hpp"
#include "vramz/backend.hpp"
#include "vramz/detail/cuda_driver.hpp"
#include "vramz/detail/gpu_compression.hpp"

#include <array>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

namespace vramz::detail {

struct CudaProbeInfo final {
    CudaDeviceInfo device{};
    std::int32_t driver_version{};
    bool uva{};
    bool vmm{};
    ByteSize minimum{};
    ByteSize recommended{};
};

// Optional bounded smoke admission. Planning precedes primary-context/stream ownership.
// Workspace charges include device metadata, codec scratch, alignment and VMM padding.
struct CudaCompressionAdmission final {
    ByteSize logical_size{};
    ByteSize physical_cap{};
    std::int32_t minimum_driver_version{};
    std::int32_t required_driver_family{};
    std::string_view required_device_name{};
    ByteSize raw_charge{};
    GpuCompressionPlan compression{};
    GpuCompressionPlan decompression{};
    ByteSize peak{};
    bool proven{};
};

// Caller-owned, fixed telemetry storage; never a synchronization or ownership oracle.
struct GpuCodecAudit final {
    Error fatal_error{};
    bool has_fatal_error{};
    bool compression_enqueue{};
    bool compression_completion{};
    bool decompression_enqueue{};
    bool decompression_completion{};
    bool stream_owned{};
    bool context_owned{};
    std::uint32_t stored_crc_expected{};
    std::uint32_t stored_crc_actual{};
};

// Passive registry counts; this performs no Driver/API call and transfers no ownership.
struct CudaResourceCounts final {
    std::uint64_t raw{};
    std::uint64_t compressed{};
    std::uint64_t workspace{};
    bool retained_context{};
    bool stream_owned{};
};

// Internal foundation, not a public Runtime backend selector. Tests inject a fake driver.
class CudaVmmBackend final : public StorageBackend {
  public:
    class ConstructionKey final {
        friend class CudaVmmBackend;
        ConstructionKey() noexcept = default;
    };
    explicit CudaVmmBackend(ConstructionKey, std::unique_ptr<CudaDriverApi> driver,
                            std::unique_ptr<GpuCompressionApi> compression) noexcept;
    [[nodiscard]] static Result<std::unique_ptr<CudaVmmBackend>>
    create(std::unique_ptr<CudaDriverApi> driver, std::int32_t ordinal = 0,
           std::unique_ptr<GpuCompressionApi> compression = {}, CudaProbeInfo* probe_info = nullptr,
           CudaCompressionAdmission* admission = nullptr, GpuCodecAudit* audit = nullptr) noexcept;
    ~CudaVmmBackend() override;
    CudaVmmBackend(const CudaVmmBackend&) = delete;
    CudaVmmBackend& operator=(const CudaVmmBackend&) = delete;
    CudaVmmBackend(CudaVmmBackend&&) = delete;
    CudaVmmBackend& operator=(CudaVmmBackend&&) = delete;

    [[nodiscard]] const BackendCapabilities& capabilities() const noexcept override;
    [[nodiscard]] ByteSize recommended_granularity() const noexcept;
    [[nodiscard]] std::int32_t driver_version() const noexcept;
    [[nodiscard]] std::optional<AsyncErrorRecord> first_error() const noexcept;
    [[nodiscard]] Result<AddressReservation>
    reserve_address_space(ByteSize size, ByteSize alignment) noexcept override;
    [[nodiscard]] Result<void>
    release_address_space(AddressReservation reservation) noexcept override;
    [[nodiscard]] bool owns_address_space(AddressReservationId id) const noexcept override;
    [[nodiscard]] CompressionStats compression_stats() const noexcept override;
    [[nodiscard]] bool compression_available() const noexcept override;
    [[nodiscard]] ByteSize owned_charge(PhysicalTier tier) const noexcept override;
    [[nodiscard]] std::uint64_t owned_resource_count() const noexcept override;
    [[nodiscard]] CudaResourceCounts resource_counts() const noexcept;
    [[nodiscard]] Result<void> enable_performance() noexcept;
    [[nodiscard]] PerformanceStats performance() const noexcept;
    [[nodiscard]] std::uint64_t owned_address_count() const noexcept;
    [[nodiscard]] Result<void> shutdown() noexcept override;
    [[nodiscard]] Result<CompletionTokenId> create_completion() noexcept override;
    [[nodiscard]] Result<CompletionState>
    query_completion(CompletionTokenId token) noexcept override;
    [[nodiscard]] Result<void> release_completion(CompletionTokenId token) noexcept override;
    [[nodiscard]] Result<ByteSize> allocation_bound(RepresentationState state,
                                                    ByteSize logical_size) noexcept override;
    [[nodiscard]] Result<ByteSize> workspace_bound(RepresentationState source,
                                                   RepresentationState destination,
                                                   ByteSize logical_size) noexcept override;
    [[nodiscard]] Result<ByteSize> compaction_bound(RepresentationState destination,
                                                    ByteSize logical_size) noexcept override;
    [[nodiscard]] Result<ByteSize> prepare_transfer(ResourceId source, ResourceId destination,
                                                    ResourceId workspace) noexcept override;
    [[nodiscard]] Result<BackendAllocation>
    allocate_representation(RepresentationAllocationRequest request) noexcept override;
    // One bounded initial round trip. Adoption consumes the private byte proof and still
    // corroborates every VMM identity/property/access field. No external writer is allowed.
    [[nodiscard]] Result<BackendAllocation>
    allocate_initialized_raw(RepresentationAllocationRequest request,
                             std::span<const std::byte> initial,
                             std::span<std::byte> observed) noexcept;
    [[nodiscard]] Result<BackendAllocation> allocate_workspace(PhysicalTier tier,
                                                               ByteSize charge) noexcept override;
    [[nodiscard]] Result<BackendAllocation>
    adopt_allocation(const BackendAllocation& provisional) noexcept override;
    [[nodiscard]] bool is_unadopted(ResourceId resource) const noexcept override;
    [[nodiscard]] Result<TransferReceipt>
    transfer(ResourceId source, ResourceId destination, ResourceId workspace,
             CompactionTarget compaction = {}) noexcept override;
    [[nodiscard]] Result<void> verify(ResourceId resource, ContentTag expected,
                                      ResourceId workspace = {}) noexcept override;
    [[nodiscard]] Result<void> verify_authoritative(ResourceId resource, ContentTag expected,
                                                    ResourceId workspace = {}) noexcept override;
    [[nodiscard]] Result<void> verify_transfer(ResourceId source, ResourceId destination,
                                               ContentTag expected,
                                               ResourceId workspace) noexcept override;
    [[nodiscard]] Result<BackendAllocation> allocation(ResourceId resource) const noexcept override;
    [[nodiscard]] Result<void> read_bytes(ResourceId resource, ByteOffset offset,
                                          std::span<std::byte> output) const noexcept override;
    [[nodiscard]] Result<ContentTag> write_bytes(ResourceId resource, ByteOffset offset,
                                                 std::span<const std::byte> input,
                                                 ContentTag previous) noexcept override;
    [[nodiscard]] Result<void> write_content(ResourceId resource,
                                             ContentTag content) noexcept override;
    [[nodiscard]] Result<void> release(ResourceId resource, ReleasePhase phase) noexcept override;
    [[nodiscard]] bool owns(ResourceId resource) const noexcept override;

  private:
    class ContextScope;
    enum class MappingState : std::uint8_t { empty, physical_created, mapped_no_access, mapped_rw };
    struct Resource final {
        BackendAllocation allocation{};
        ContentTag content{};
        CudaPhysicalHandle handle{};
        MappingState state{MappingState::empty};
        bool adopted{};
        DeviceAddress private_address{};
        bool prepared{};
        bool private_initialization_verified{};
    };
    [[nodiscard]] Result<void> probe(std::int32_t ordinal, CudaProbeInfo* info,
                                     CudaCompressionAdmission* admission) noexcept;
    [[nodiscard]] Result<void> admit_compression(CudaCompressionAdmission& admission) noexcept;
    [[noreturn]] void fatal(Error error) const noexcept;
    void reject_ambiguity(const Error& error) const noexcept;
    [[nodiscard]] Resource* find(ResourceId id) noexcept;
    [[nodiscard]] const Resource* find(ResourceId id) const noexcept;
    [[nodiscard]] Result<void> corroborate(const Resource& resource) const noexcept;
    [[nodiscard]] Result<void> release_locked(Resource& resource) noexcept;
    [[nodiscard]] Result<std::vector<std::byte>> read_all(const Resource& resource) const noexcept;
    [[nodiscard]] Result<void> verify_locked(const Resource& resource,
                                             ContentTag expected) const noexcept;
    [[nodiscard]] bool contains(DeviceAddress address, ByteSize size) const noexcept;
    [[nodiscard]] static DeviceAddress mapped_address(const Resource& resource) noexcept;
    [[nodiscard]] Result<GpuCompressionPlan> compression_plan(GpuCodecDirection direction,
                                                              ByteSize logical_size) noexcept;
    [[nodiscard]] Result<BackendAllocation>
    allocate_storage(BackendAllocation descriptor, ContentTag content,
                     std::span<const std::byte> initial = {},
                     std::span<std::byte> observed = {}) noexcept;
    [[nodiscard]] Result<ByteSize> execute_codec(Resource& source, Resource& destination,
                                                 Resource& workspace,
                                                 GpuCodecDirection direction) noexcept;
    [[nodiscard]] Result<void> free_private_address(DeviceAddress address) noexcept;

    std::unique_ptr<CudaDriverApi> driver_;
    std::unique_ptr<GpuCompressionApi> compression_;
    mutable std::mutex mutex_{};
    mutable AsyncErrorChannel errors_{16U};
    BackendCapabilities capabilities_{};
    ByteSize recommended_{};
    std::int32_t driver_version_{};
    CudaDevice device_{};
    CudaContext context_{};
    std::array<AddressReservation, max_runtime_buffers> addresses_{};
    std::array<Resource, 256U> resources_{};
    struct PrivateAddress final {
        DeviceAddress base{};
        ByteSize size{};
    };
    std::array<PrivateAddress, 256U> private_addresses_{};
    CompressionStats compression_stats_{};
    mutable std::mutex stats_mutex_{}; // Leaf telemetry lock, never a driver/registry dependency.
    ByteSize codec_alignment_{};
    ByteSize physical_cap_{};
    ByteSize planned_logical_{};
    std::array<GpuCompressionPlan, 2U> admitted_plans_{};
    GpuCodecAudit* codec_audit_{};
    bool stream_initialized_{};
    std::uint64_t next_address_id_{1U};
    std::uint64_t next_resource_id_{1U};
    std::array<CompletionTokenId, max_pending_completions> completions_{};
    CompletionTokenId last_completion_id_{};
};

} // namespace vramz::detail
