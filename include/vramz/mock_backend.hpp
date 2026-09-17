#pragma once

#include "vramz/backend.hpp"
#include "vramz/stats.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <span>
#include <vector>

namespace vramz::detail {
class CompressionCodec;
}

namespace vramz {

enum class FaultPoint : std::uint8_t {
    gpu_allocation,
    host_allocation,
    workspace_allocation,
    transfer,
    verification,
    corrupt_content,
    rollback_cleanup,
    rollback_workspace_cleanup,
    rollback_destination_cleanup,
    post_commit_cleanup,
    close_cleanup,
    ambiguous_transfer,
    advertised_bound_too_small,
    invalid_resource_identity,
    duplicate_resource_identity,
    false_release_success,
    write_content,
    verification_allocation,
    final_storage_allocation,
    compression,
    decompression,
    decompression_size_mismatch,
    crc_comparison,
    byte_comparison,
    raw_copy,
    compressed_copy,
    stored_size_contract,
    allocation_descriptor_mismatch,
    allocation_identity_mismatch,
    allocation_tier_mismatch,
    allocation_kind_mismatch,
    allocation_logical_size_mismatch,
    allocation_stored_size_mismatch,
    allocation_encoding_mismatch,
    allocation_crc_mismatch,
    receipt_logical_size_mismatch,
    receipt_stored_size_mismatch,
    receipt_encoding_mismatch,
    receipt_charge_exceeds_admission,
    receipt_zero_charge,
    receipt_unaligned_charge,
    receipt_plausible_charge_mismatch,
    verification_contract,
    allocation_bound_padding,
    provisional_charge_mismatch,
    provisional_tier_mismatch,
    provisional_kind_mismatch,
    provisional_metadata_mismatch,
    provisional_logical_size_mismatch,
    provisional_stored_size_mismatch,
    provisional_encoding_mismatch,
    provisional_zero_charge,
    provisional_unaligned_charge,
    adoption_failure,
    address_reserve,
    address_free,
    count
};

enum class ResourceCorruption : std::uint8_t {
    payload_flip,
    payload_multiple_bits,
    truncate_payload,
    append_payload,
    logical_size,
    stored_size,
    crc32c,
    encoding,
    content_tag
};

struct MockBackendConfig final {
    BackendCapabilities capabilities{ByteSize{64U}, ByteSize{64U}, ByteSize{64U},
                                     true,          true,          false};
    std::array<ByteSize, 4U> exact_state_charges{};
    ByteSize workspace_charge{64U};
};

class MockBackend final : public StorageBackend {
  public:
    explicit MockBackend(const MockBackendConfig& config);
    ~MockBackend() override;

    [[nodiscard]] const BackendCapabilities& capabilities() const noexcept override;
    [[nodiscard]] Result<AddressReservation>
    reserve_address_space(ByteSize size, ByteSize alignment) noexcept override;
    [[nodiscard]] Result<void>
    release_address_space(AddressReservation reservation) noexcept override;
    [[nodiscard]] bool owns_address_space(AddressReservationId id) const noexcept override;
    void inject_failure(FaultPoint point, std::uint64_t invocation = 1U) noexcept;
    void clear_failures() noexcept;

    [[nodiscard]] Result<ByteSize> allocation_bound(RepresentationState state,
                                                    ByteSize logical_size) noexcept override;
    [[nodiscard]] Result<ByteSize> workspace_bound(RepresentationState source,
                                                   RepresentationState destination,
                                                   ByteSize logical_size) noexcept override;
    [[nodiscard]] Result<BackendAllocation>
    allocate_representation(RepresentationAllocationRequest request) noexcept override;
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
    [[nodiscard]] ByteSize owned_charge(PhysicalTier tier) const noexcept override;
    [[nodiscard]] std::uint64_t owned_resource_count() const noexcept override;
    [[nodiscard]] Result<ContentTag> content(ResourceId resource) const noexcept;
    [[nodiscard]] Result<void> corrupt(ResourceId resource, ResourceCorruption corruption) noexcept;
    [[nodiscard]] CompressionStats compression_stats() const noexcept override;
    [[nodiscard]] bool compression_available() const noexcept override;
    [[nodiscard]] ResourceId last_transfer_destination() const noexcept;
    [[nodiscard]] ResourceId last_rollback_resource() const noexcept;
    [[nodiscard]] ResourceId last_allocated_resource() const noexcept;
    [[nodiscard]] ResourceId last_release_resource() const noexcept;
    [[nodiscard]] std::uint64_t adoption_attempt_count() const noexcept;

  private:
    struct Resource final {
        ResourceId id{};
        PhysicalTier tier{PhysicalTier::gpu};
        ByteSize charge{};
        ResourceKind kind{ResourceKind::representation};
        RepresentationState state{RepresentationState::gpu_raw};
        ContentTag content{};
        RepresentationMetadata metadata{};
        DeviceAddress address{};
        std::vector<std::byte> bytes{};
        bool awaiting_adoption{};
    };

    [[nodiscard]] bool should_fail(FaultPoint point) const noexcept;
    [[nodiscard]] Result<ByteSize> computed_charge(RepresentationState state,
                                                   ByteSize logical_size) const noexcept;
    [[nodiscard]] Result<BackendAllocation> allocate(Resource resource, FaultPoint point) noexcept;
    [[nodiscard]] Result<BackendAllocation>
    provisional_descriptor(const Resource& resource) const noexcept;
    [[nodiscard]] Result<void> verify_locked(const Resource& resource, ContentTag expected,
                                             Resource* workspace) noexcept;
    [[nodiscard]] Result<TransferReceipt>
    compact_compressed(Resource& resource, ByteSize stored_size, Resource* workspace) noexcept;
    [[nodiscard]] Result<std::span<const std::byte>> logical_bytes(const Resource& resource,
                                                                   Resource* workspace) noexcept;

    MockBackendConfig config_{};
    detail::CompressionCodec& codec_;
    mutable std::mutex mutex_{};
    std::map<std::uint64_t, Resource> resources_{};
    std::array<AddressReservation, max_runtime_buffers> reservations_{};
    std::uint64_t next_address_id_{1U};
    std::uint64_t next_device_address_{0x100000000ULL};
    mutable std::array<std::uint64_t, static_cast<std::size_t>(FaultPoint::count)> fail_on_{};
    mutable std::array<std::uint64_t, static_cast<std::size_t>(FaultPoint::count)> calls_{};
    std::uint64_t next_resource_id_{1U};
    ResourceId last_transfer_destination_{};
    ResourceId last_rollback_resource_{};
    ResourceId last_allocated_resource_{};
    ResourceId last_release_resource_{};
    std::uint64_t adoption_attempt_count_{};
    std::uint64_t successful_compressions_{};
    std::uint64_t successful_decompressions_{};
    std::uint64_t integrity_failures_{};
    std::uint64_t bytes_compressed_{};
    std::uint64_t bytes_decompressed_{};
};

} // namespace vramz
