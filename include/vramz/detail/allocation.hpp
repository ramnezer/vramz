#pragma once

#include "vramz/async_error_channel.hpp"
#include "vramz/backend.hpp"

#include <exception>

namespace vramz::detail {

struct AllocationExpectation final {
    PhysicalTier tier{};
    ResourceKind kind{};
    ByteSize bound{};
    ByteSize granularity{};
    ByteSize logical_size{};
    RepresentationState state{};
    DeviceAddress address{};
};

[[noreturn]] inline void
allocation_contract_failure(AsyncErrorChannel& errors,
                            const BackendAllocation& provisional) noexcept {
    errors.push(make_error(ErrorCode::backend_contract_violation, OperationId::allocate,
                           provisional.id.value(), provisional.charge.value()));
    std::terminate();
}

[[nodiscard]] inline bool same_metadata(const RepresentationMetadata& left,
                                        const RepresentationMetadata& right) noexcept {
    return left.encoding == right.encoding && left.logical_size == right.logical_size &&
           left.stored_size == right.stored_size && left.crc32c == right.crc32c &&
           left.stored_crc32c == right.stored_crc32c;
}

// This function returns only a corroborated descriptor. No recoverable error, cleanup, or
// observable retained-resource accounting is possible while the allocation remains provisional.
[[nodiscard]] inline BackendAllocation
corroborate_allocation(StorageBackend& backend, AsyncErrorChannel& errors,
                       const BackendAllocation& provisional, AllocationExpectation expected,
                       ResourceId forbidden_first = {}, ResourceId forbidden_second = {}) noexcept {
    if (provisional.id == ResourceId{} || provisional.id == forbidden_first ||
        provisional.id == forbidden_second || provisional.tier != expected.tier ||
        provisional.kind != expected.kind || provisional.charge == ByteSize{} ||
        provisional.address != expected.address || provisional.charge > expected.bound ||
        expected.granularity == ByteSize{} ||
        provisional.charge.value() % expected.granularity.value() != 0U) {
        allocation_contract_failure(errors, provisional);
    }
    if (expected.kind == ResourceKind::representation) {
        const auto& metadata = provisional.metadata;
        const bool raw = is_raw(expected.state);
        if ((expected.state == RepresentationState::gpu_raw) !=
            (expected.address != DeviceAddress{})) {
            allocation_contract_failure(errors, provisional);
        }
        if (expected.logical_size == ByteSize{} || !is_valid(expected.state) ||
            tier_of(expected.state) != expected.tier ||
            metadata.logical_size != expected.logical_size ||
            metadata.encoding != (raw ? Encoding::raw : Encoding::lz4_block) ||
            (raw && (metadata.stored_size != expected.logical_size ||
                     metadata.stored_size > provisional.charge)) ||
            (raw && metadata.stored_crc32c != metadata.crc32c) ||
            (!raw && (metadata.stored_size != ByteSize{} || metadata.crc32c != 0U ||
                      metadata.stored_crc32c != 0U))) {
            allocation_contract_failure(errors, provisional);
        }
    }
    const auto adopted = backend.adopt_allocation(provisional);
    if (!adopted) {
        allocation_contract_failure(errors, provisional);
    }
    const auto exact = adopted.value();
    // Adoption cannot silently substitute a different identity, charge, or representation.
    if (exact.id != provisional.id || exact.tier != provisional.tier ||
        exact.kind != provisional.kind || exact.charge != provisional.charge ||
        exact.address != provisional.address ||
        (expected.kind == ResourceKind::representation &&
         !same_metadata(exact.metadata, provisional.metadata))) {
        allocation_contract_failure(errors, provisional);
    }
    return exact;
}

} // namespace vramz::detail
