#pragma once

#include <compare>
#include <cstdint>

namespace vramz {

// Immutable data-generation identity used by the Lease implementation, not a freshness token.
struct ContentTag final {
    std::uint64_t generation{};
    std::uint64_t fingerprint{};
    auto operator<=>(const ContentTag&) const = default;
};

template <typename Tag, typename Storage> class StrongValue final {
  public:
    constexpr StrongValue() noexcept = default;
    explicit constexpr StrongValue(Storage value) noexcept : value_(value) {}

    [[nodiscard]] constexpr Storage value() const noexcept { return value_; }
    auto operator<=>(const StrongValue&) const = default;

  private:
    Storage value_{};
};

struct ByteSizeTag;
struct ByteOffsetTag;
struct ChunkIdTag;
struct BufferIdTag;
struct TransactionIdTag;
struct ResourceIdTag;
struct BackendIdTag;
struct DeviceAddressTag;
struct AddressReservationIdTag;
struct CompletionTokenIdTag;
struct RuntimeIdTag;

using ByteSize = StrongValue<ByteSizeTag, std::uint64_t>;
using ByteOffset = StrongValue<ByteOffsetTag, std::uint64_t>;
using ChunkId = StrongValue<ChunkIdTag, std::uint64_t>;
using BufferId = StrongValue<BufferIdTag, std::uint64_t>;
using TransactionId = StrongValue<TransactionIdTag, std::uint64_t>;
using ResourceId = StrongValue<ResourceIdTag, std::uint64_t>;
using BackendId = StrongValue<BackendIdTag, std::uint32_t>;
using DeviceAddress = StrongValue<DeviceAddressTag, std::uint64_t>;
using AddressReservationId = StrongValue<AddressReservationIdTag, std::uint64_t>;
using CompletionTokenId = StrongValue<CompletionTokenIdTag, std::uint64_t>;
using RuntimeId = StrongValue<RuntimeIdTag, std::uint64_t>;

struct MemoryRange final {
    ByteOffset offset{};
    ByteSize length{};
};

struct DeviceSpan final {
    DeviceAddress address{};
    ByteSize length{};
};

enum class PhysicalTier : std::uint8_t { gpu, host };

[[nodiscard]] constexpr bool is_valid(PhysicalTier tier) noexcept {
    return tier == PhysicalTier::gpu || tier == PhysicalTier::host;
}

} // namespace vramz
