#pragma once

#include "vramz/completion.hpp"
#include "vramz/config.hpp"
#include "vramz/state.hpp"
#include "vramz/stats.hpp"

#include <cstdint>
#include <span>

namespace vramz {

enum class ResourceKind : std::uint8_t { representation, workspace };
enum class ReleasePhase : std::uint8_t { rollback, post_commit, close };

struct AddressReservation final {
    AddressReservationId id{};
    DeviceAddress base{};
    ByteSize size{};
    ByteSize alignment{};
    auto operator<=>(const AddressReservation&) const = default;
};

struct RepresentationAllocationRequest final {
    RepresentationState state{};
    ByteSize logical_size{};
    ContentTag content{};
    DeviceAddress stable_address{};
};

struct BackendAllocation final {
    ResourceId id{};
    PhysicalTier tier{PhysicalTier::gpu};
    ByteSize charge{};
    ResourceKind kind{ResourceKind::representation};
    RepresentationMetadata metadata{};
    DeviceAddress address{};
};

// Transfer cannot replace the adopted allocation's identity, tier, or resource kind.
struct TransferReceipt final {
    ByteSize charge{};
    RepresentationMetadata metadata{};
};
struct CompactionTarget final {
    ResourceId resource{};
};

class StorageBackend {
  public:
    virtual ~StorageBackend() = default;

    [[nodiscard]] virtual const BackendCapabilities& capabilities() const noexcept = 0;
    [[nodiscard]] virtual Result<AddressReservation>
    reserve_address_space(ByteSize size, ByteSize alignment) noexcept = 0;
    [[nodiscard]] virtual Result<void>
    release_address_space(AddressReservation reservation) noexcept = 0;
    [[nodiscard]] virtual bool owns_address_space(AddressReservationId id) const noexcept = 0;
    [[nodiscard]] virtual CompressionStats compression_stats() const noexcept = 0;
    [[nodiscard]] virtual bool compression_available() const noexcept = 0;
    // Observational ownership totals; never an allocation/transfer charge oracle.
    [[nodiscard]] virtual ByteSize owned_charge(PhysicalTier tier) const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t owned_resource_count() const noexcept = 0;
    // Idempotent backend-specific finalization, called after all Buffer/resource cleanup.
    [[nodiscard]] virtual Result<void> shutdown() noexcept { return {}; }
    // Reserve a never-reused, initially pending fence BEFORE external work is accepted.
    // M6 exposes creation only through deferred-lease registration; real adapters disable it.
    [[nodiscard]] virtual Result<CompletionTokenId> create_completion() noexcept {
        return make_error(ErrorCode::unsupported, OperationId::defer_lease);
    }
    [[nodiscard]] virtual Result<CompletionState> query_completion(CompletionTokenId) noexcept {
        return make_error(ErrorCode::unsupported, OperationId::query_completion);
    }
    // Only a proven-complete fence may be released. Ordinary Error proves no mutation.
    [[nodiscard]] virtual Result<void> release_completion(CompletionTokenId) noexcept {
        return make_error(ErrorCode::unsupported, OperationId::release_completion);
    }
    [[nodiscard]] virtual Result<ByteSize> allocation_bound(RepresentationState state,
                                                            ByteSize logical_size) noexcept = 0;
    [[nodiscard]] virtual Result<ByteSize> workspace_bound(RepresentationState source,
                                                           RepresentationState destination,
                                                           ByteSize logical_size) noexcept = 0;
    // Optional private compaction target, reserved independently of codec workspace.
    [[nodiscard]] virtual Result<ByteSize> compaction_bound(RepresentationState,
                                                            ByteSize) noexcept {
        return ByteSize{};
    }
    // Preparation may populate private bytes, but cannot change any owned charge.
    // A nonzero result requests an exact compaction target within the admitted bound.
    [[nodiscard]] virtual Result<ByteSize> prepare_transfer(ResourceId, ResourceId,
                                                            ResourceId) noexcept {
        return ByteSize{};
    }
    [[nodiscard]] virtual Result<BackendAllocation>
    allocate_representation(RepresentationAllocationRequest request) noexcept = 0;
    [[nodiscard]] virtual Result<BackendAllocation>
    allocate_workspace(PhysicalTier tier, ByteSize charge) noexcept = 0;
    // Successful allocation returns a provisional descriptor, not a trusted ledger charge.
    // Adoption corroborates identity, tier, exact charge and kind against the retained resource,
    // plus all representation metadata for representations (workspace metadata is irrelevant).
    // Success delivers the retained exact descriptor without allocation or further fallible work.
    // Failure after allocation success is a fatal contract boundary: core cannot safely clean up
    // or publish debt using an uncorroborated descriptor.
    [[nodiscard]] virtual Result<BackendAllocation>
    adopt_allocation(const BackendAllocation& provisional) noexcept = 0;
    [[nodiscard]] virtual bool is_unadopted(ResourceId resource) const noexcept = 0;
    // Success delivers exact destination charge/metadata in the operation that changes them.
    // Failure leaves all charges unchanged. Adopted identity, tier and kind never change;
    // destination charge may only shrink. Source/ordinary workspace charges remain unchanged.
    // An optional adopted compaction target exchanges backing with destination: destination
    // takes its exact charge, target takes the old destination charge. Both IDs stay fixed.
    // Core atomically rebuckets this exchange and corroborates BOTH retained descriptors.
    // No allocation, callback or fallible discovery is allowed between mutation and delivery.
    [[nodiscard]] virtual Result<TransferReceipt>
    transfer(ResourceId source, ResourceId destination, ResourceId workspace,
             CompactionTarget compaction = {}) noexcept = 0;
    [[nodiscard]] virtual Result<void> verify(ResourceId resource, ContentTag expected,
                                              ResourceId workspace = {}) noexcept = 0;
    [[nodiscard]] virtual Result<void> verify_authoritative(ResourceId resource,
                                                            ContentTag expected,
                                                            ResourceId workspace = {}) noexcept = 0;
    [[nodiscard]] virtual Result<void> verify_transfer(ResourceId source, ResourceId destination,
                                                       ContentTag expected,
                                                       ResourceId workspace) noexcept = 0;
    [[nodiscard]] virtual Result<BackendAllocation>
    allocation(ResourceId resource) const noexcept = 0;
    [[nodiscard]] virtual Result<void> read_bytes(ResourceId resource, ByteOffset offset,
                                                  std::span<std::byte> output) const noexcept = 0;
    [[nodiscard]] virtual Result<ContentTag> write_bytes(ResourceId resource, ByteOffset offset,
                                                         std::span<const std::byte> input,
                                                         ContentTag previous) noexcept = 0;
    [[nodiscard]] virtual Result<void> write_content(ResourceId resource,
                                                     ContentTag content) noexcept = 0;
    [[nodiscard]] virtual Result<void> release(ResourceId resource,
                                               ReleasePhase phase) noexcept = 0;
    [[nodiscard]] virtual bool owns(ResourceId resource) const noexcept = 0;
};

} // namespace vramz
