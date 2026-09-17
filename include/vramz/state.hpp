#pragma once

#include "vramz/result.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>

namespace vramz {

namespace detail {
struct BufferState;
[[nodiscard]] Result<void> close_buffer_state(BufferState& buffer) noexcept;
} // namespace detail

namespace testing {
class RuntimeAccess;
}

enum class RepresentationState : std::uint8_t {
    gpu_raw,
    gpu_compressed,
    host_raw,
    host_compressed
};

enum class LifecycleState : std::uint8_t { creating, live, closing, poisoned, dead };

enum class Encoding : std::uint8_t { raw, lz4_block };

struct RepresentationMetadata final {
    Encoding encoding{Encoding::raw};
    ByteSize logical_size{};
    ByteSize stored_size{};
    std::uint32_t crc32c{};
    // CRC of stored bytes only, excluding physical padding. crc32c identifies logical bytes.
    std::uint32_t stored_crc32c{};
};

[[nodiscard]] constexpr bool is_valid(RepresentationState state) noexcept {
    return state == RepresentationState::gpu_raw || state == RepresentationState::gpu_compressed ||
           state == RepresentationState::host_raw || state == RepresentationState::host_compressed;
}

[[nodiscard]] constexpr PhysicalTier tier_of(RepresentationState state) noexcept {
    return (state == RepresentationState::gpu_raw || state == RepresentationState::gpu_compressed)
               ? PhysicalTier::gpu
               : PhysicalTier::host;
}

[[nodiscard]] constexpr bool is_raw(RepresentationState state) noexcept {
    return state == RepresentationState::gpu_raw || state == RepresentationState::host_raw;
}

[[nodiscard]] constexpr ContentTag initial_content_tag(BufferId buffer, ChunkId chunk) noexcept {
    // Fingerprints use intentional modulo-2^64 mixing; they are not size arithmetic.
    return ContentTag{0U, (buffer.value() * 0x9E3779B185EBCA87ULL) ^
                              (chunk.value() + 0xD1B54A32D192ED03ULL)};
}

[[nodiscard]] constexpr ContentTag next_content_tag(ContentTag current) noexcept {
    // Fingerprints use intentional modulo-2^64 mixing; generation overflow is checked by callers.
    return ContentTag{current.generation + 1U,
                      (current.fingerprint ^ 0xA0761D6478BD642FULL) + current.generation + 1U};
}

struct Representation final {
    RepresentationState state{RepresentationState::gpu_raw};
    ResourceId resource{};
    ByteSize charge{};
    ContentTag content{};
    DeviceAddress address{};
    RepresentationMetadata metadata{};
};

struct CleanupResource final {
    ResourceId resource{};
    PhysicalTier tier{PhysicalTier::gpu};
    ByteSize charge{};
};

struct ChunkSnapshot final {
    ChunkId id{};
    LifecycleState lifecycle{LifecycleState::creating};
    Representation authoritative{};
    std::uint64_t transition_epoch{};
    std::uint64_t read_pins{};
    bool write_pin{};
    bool transition_active{};
    std::uint32_t cleanup_resource_count{};
    std::optional<Error> first_error{};
    std::uint64_t lease_intents{};
    std::uint64_t access_revision{};
};

class TransactionCoordinator;
class Buffer;
class Lease;

class Chunk final {
  public:
    Chunk(ChunkId id, ByteOffset offset, ByteSize logical_size, Representation authoritative,
          DeviceAddress stable_address = {}) noexcept;

    Chunk(const Chunk&) = delete;
    Chunk& operator=(const Chunk&) = delete;

    [[nodiscard]] ChunkSnapshot snapshot() const noexcept;
    [[nodiscard]] ByteOffset offset() const noexcept { return offset_; }
    [[nodiscard]] ByteSize logical_size() const noexcept { return logical_size_; }

  private:
    friend class TransactionCoordinator;
    friend class Runtime;
    friend class Buffer;
    friend class Lease;
    friend class testing::RuntimeAccess;
    friend Result<void> detail::close_buffer_state(detail::BufferState& buffer) noexcept;

    mutable std::mutex mutex_{};
    ChunkId id_{};
    ByteOffset offset_{};
    ByteSize logical_size_{};
    LifecycleState lifecycle_{LifecycleState::creating};
    Representation authoritative_{};
    DeviceAddress stable_address_{};
    std::uint64_t transition_epoch_{};
    std::uint64_t read_pins_{};
    bool write_pin_{};
    bool transition_active_{};
    TransactionId transaction_{};
    std::uint64_t lease_intents_{};
    // Optimistic policy freshness, advanced atomically with valid acquire intents.
    std::uint64_t access_revision_{};
    bool write_lease_intent_{};
    std::optional<Error> first_error_{};
    std::array<CleanupResource, 4U> cleanup_resources_{};
    std::uint32_t cleanup_resource_count_{};
    bool authority_retired_to_debt_{};
};

} // namespace vramz
