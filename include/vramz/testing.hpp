#pragma once

#include "vramz/mock_backend.hpp"
#include "vramz/policy.hpp"
#include "vramz/runtime.hpp"
#include "vramz/transaction.hpp"

#include <cstddef>
#include <span>

namespace vramz::testing {

// Synchronous, non-owning test seam. Called outside lifecycle/chunk locks, while the
// allocation gate is held. Tests may coordinate RAW reads, never re-enter a gated operation.
class PolicySelectionObserver {
  public:
    virtual ~PolicySelectionObserver() = default;
    virtual void on_selected(const ChunkSnapshot& snapshot,
                             const policy::Proposal& proposal) noexcept = 0;
    // Passive result telemetry, outside lifecycle/chunk locks. It cannot replace the result.
    virtual void on_completed(const ChunkSnapshot&, const Result<void>&) noexcept {}
    virtual void on_transaction_phase(TransactionPhase, const ChunkSnapshot&) noexcept {}
};

class RuntimeAccess final {
  public:
    [[nodiscard]] static Result<Runtime>
    create_with_backend(RuntimeConfig config, std::unique_ptr<StorageBackend> backend) noexcept;
    [[nodiscard]] static Result<void> reclaim(Runtime& runtime, ByteSize target,
                                              bool admission_pressure,
                                              PolicySelectionObserver* observer = nullptr) noexcept;
    [[nodiscard]] static Result<void> advance_access_revision(Buffer& buffer, ChunkId chunk_id,
                                                              std::uint64_t amount) noexcept;
    [[nodiscard]] static Result<policy::Metadata>
    policy_metadata(const Buffer& buffer, std::uint64_t chunk_index) noexcept;
    static void advance_policy_epoch(Runtime& runtime, std::uint64_t epochs) noexcept;
    [[nodiscard]] static Result<void> advance_policy_cycle(Runtime& runtime,
                                                           std::uint64_t cycles) noexcept;
    [[nodiscard]] static MockBackend& backend(Runtime& runtime) noexcept;
    static void inject_materialize_failure(Runtime& runtime,
                                           std::uint64_t invocation = 1U) noexcept;
    static void inject_materialize_failure(BudgetLedger& ledger,
                                           std::uint64_t invocation = 1U) noexcept {
        const std::scoped_lock lock{ledger.mutex_};
        ledger.materialize_failure_countdown_ = invocation;
    }
    static void inject_failure(Runtime& runtime, FaultPoint point,
                               std::uint64_t invocation) noexcept;
    [[nodiscard]] static Result<void>
    migrate(Buffer& buffer, std::uint64_t chunk_index, RepresentationState destination,
            TransactionObserver* observer,
            const MigrationConstraints* constraints = nullptr) noexcept;
    [[nodiscard]] static Result<ChunkSnapshot> chunk_snapshot(const Buffer& buffer,
                                                              std::uint64_t chunk_index) noexcept;
    [[nodiscard]] static BufferId buffer_identity(const Buffer& buffer) noexcept;
    [[nodiscard]] static Result<CleanupResource> cleanup_resource(const Chunk& chunk,
                                                                  std::uint32_t index) noexcept {
        const std::scoped_lock lock{chunk.mutex_};
        if (index >= chunk.cleanup_resource_count_) {
            return make_error(ErrorCode::invalid_argument, OperationId::release, index);
        }
        return chunk.cleanup_resources_[index];
    }
    [[nodiscard]] static Result<CleanupResource>
    cleanup_resource(const Buffer& buffer, std::uint64_t chunk_index, std::uint32_t index) noexcept;
    [[nodiscard]] static bool accounting_conserved(const Runtime& runtime,
                                                   PhysicalTier tier) noexcept;
    [[nodiscard]] static Result<void> quarantine(Buffer& buffer) noexcept;
    [[nodiscard]] static std::uint64_t owned_resource_count(const Runtime& runtime) noexcept;
    [[nodiscard]] static std::weak_ptr<const void>
    runtime_lifetime(const Runtime& runtime) noexcept;
    [[nodiscard]] static std::weak_ptr<const void> buffer_lifetime(const Buffer& buffer) noexcept;
    [[nodiscard]] static Result<void> read_bytes(const Lease& lease, ByteOffset relative_offset,
                                                 std::span<std::byte> output) noexcept;
    [[nodiscard]] static Result<void> write_bytes(Lease& lease, ByteOffset relative_offset,
                                                  std::span<const std::byte> input) noexcept;
    [[nodiscard]] static Result<void> corrupt(Buffer& buffer, std::uint64_t chunk_index,
                                              ResourceCorruption corruption) noexcept;
    [[nodiscard]] static Result<RepresentationMetadata>
    representation_metadata(const Buffer& buffer, std::uint64_t chunk_index) noexcept;
    [[nodiscard]] static bool cpu_lz4_available(const Runtime& runtime) noexcept;
    [[nodiscard]] static Result<ByteSize> codec_maximum_compressed_size(ByteSize input) noexcept;
    [[nodiscard]] static ByteSize codec_maximum_input_size() noexcept;
    [[nodiscard]] static Result<void> codec_round_trip(std::span<const std::byte> input,
                                                       std::span<std::byte> output) noexcept;
    [[nodiscard]] static Result<void> codec_decompress(std::span<const std::byte> input,
                                                       std::span<std::byte> output) noexcept;
};

inline void inject_failure(Runtime& runtime, FaultPoint point,
                           std::uint64_t invocation = 1U) noexcept {
    RuntimeAccess::inject_failure(runtime, point, invocation);
}

[[nodiscard]] inline Result<void> migrate(Buffer& buffer, std::uint64_t chunk_index,
                                          RepresentationState destination,
                                          TransactionObserver* observer = nullptr) noexcept {
    return RuntimeAccess::migrate(buffer, chunk_index, destination, observer);
}

[[nodiscard]] inline Result<ChunkSnapshot> chunk_snapshot(const Buffer& buffer,
                                                          std::uint64_t chunk_index) noexcept {
    return RuntimeAccess::chunk_snapshot(buffer, chunk_index);
}

[[nodiscard]] inline bool accounting_conserved(const Runtime& runtime, PhysicalTier tier) noexcept {
    return RuntimeAccess::accounting_conserved(runtime, tier);
}

[[nodiscard]] inline Result<void> quarantine(Buffer& buffer) noexcept {
    return RuntimeAccess::quarantine(buffer);
}

[[nodiscard]] inline std::uint64_t owned_resource_count(const Runtime& runtime) noexcept {
    return RuntimeAccess::owned_resource_count(runtime);
}

[[nodiscard]] inline std::weak_ptr<const void> runtime_lifetime(const Runtime& runtime) noexcept {
    return RuntimeAccess::runtime_lifetime(runtime);
}

[[nodiscard]] inline std::weak_ptr<const void> buffer_lifetime(const Buffer& buffer) noexcept {
    return RuntimeAccess::buffer_lifetime(buffer);
}

[[nodiscard]] inline Result<void> read_bytes(const Lease& lease, ByteOffset relative_offset,
                                             std::span<std::byte> output) noexcept {
    return RuntimeAccess::read_bytes(lease, relative_offset, output);
}

[[nodiscard]] inline Result<void> write_bytes(Lease& lease, ByteOffset relative_offset,
                                              std::span<const std::byte> input) noexcept {
    return RuntimeAccess::write_bytes(lease, relative_offset, input);
}

[[nodiscard]] inline Result<void> corrupt(Buffer& buffer, std::uint64_t chunk_index,
                                          ResourceCorruption corruption) noexcept {
    return RuntimeAccess::corrupt(buffer, chunk_index, corruption);
}

[[nodiscard]] inline Result<RepresentationMetadata>
representation_metadata(const Buffer& buffer, std::uint64_t chunk_index) noexcept {
    return RuntimeAccess::representation_metadata(buffer, chunk_index);
}

[[nodiscard]] inline bool cpu_lz4_available(const Runtime& runtime) noexcept {
    return RuntimeAccess::cpu_lz4_available(runtime);
}

[[nodiscard]] inline Result<ByteSize> codec_maximum_compressed_size(ByteSize input) noexcept {
    return RuntimeAccess::codec_maximum_compressed_size(input);
}

[[nodiscard]] inline ByteSize codec_maximum_input_size() noexcept {
    return RuntimeAccess::codec_maximum_input_size();
}

[[nodiscard]] inline Result<void> codec_round_trip(std::span<const std::byte> input,
                                                   std::span<std::byte> output) noexcept {
    return RuntimeAccess::codec_round_trip(input, output);
}

[[nodiscard]] inline Result<void> codec_decompress(std::span<const std::byte> input,
                                                   std::span<std::byte> output) noexcept {
    return RuntimeAccess::codec_decompress(input, output);
}

} // namespace vramz::testing
