#include "vramz/mock_backend.hpp"

#include "vramz/checked.hpp"
#include "vramz/crc32c.hpp"
#include "vramz/detail/compression.hpp"
#include "vramz/saturating.hpp"

#include <algorithm>
#include <limits>
#include <new>

namespace vramz {
namespace {

[[nodiscard]] constexpr std::size_t state_index(RepresentationState state) noexcept {
    return static_cast<std::size_t>(state);
}

[[nodiscard]] Error backend_error(ErrorCode code, OperationId operation, std::uint64_t object = 0U,
                                  std::uint64_t detail = 0U) noexcept {
    Error error = make_error(code, operation, object, detail);
    error.backend = BackendId{1U};
    error.native_domain = NativeErrorDomain::internal_backend;
    return error;
}

[[nodiscard]] constexpr Encoding encoding_for(RepresentationState state) noexcept {
    return is_raw(state) ? Encoding::raw : Encoding::lz4_block;
}

[[nodiscard]] Result<std::size_t> checked_storage_size(std::uint64_t value,
                                                       OperationId operation) noexcept {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return backend_error(ErrorCode::arithmetic_overflow, operation, 0U, value);
    }
    return static_cast<std::size_t>(value);
}

} // namespace

MockBackend::MockBackend(const MockBackendConfig& config)
    : config_(config), codec_(detail::cpu_compression_codec()) {}

MockBackend::~MockBackend() = default;

Result<AddressReservation> MockBackend::reserve_address_space(ByteSize size,
                                                              ByteSize alignment) noexcept {
    const std::scoped_lock lock{mutex_};
    const auto granularity = config_.capabilities.reservation_granularity;
    if (size == ByteSize{} || alignment == ByteSize{} || granularity == ByteSize{} ||
        size.value() % granularity.value() != 0U || alignment.value() % granularity.value() != 0U) {
        return backend_error(ErrorCode::invalid_alignment, OperationId::allocate);
    }
    if (should_fail(FaultPoint::address_reserve)) {
        return backend_error(ErrorCode::out_of_host_memory, OperationId::allocate);
    }
    const auto base =
        checked_align_up(next_device_address_, alignment.value(), OperationId::allocate);
    if (!base) {
        return base.error();
    }
    const auto end = checked_add(base.value(), size.value(), OperationId::allocate);
    if (!end || next_address_id_ == std::numeric_limits<std::uint64_t>::max()) {
        return backend_error(ErrorCode::arithmetic_overflow, OperationId::allocate);
    }
    for (auto& slot : reservations_) {
        if (slot.id == AddressReservationId{}) {
            slot = AddressReservation{AddressReservationId{next_address_id_},
                                      DeviceAddress{base.value()}, size, alignment};
            ++next_address_id_;
            next_device_address_ = end.value();
            return slot;
        }
    }
    return backend_error(ErrorCode::out_of_host_memory, OperationId::allocate);
}

Result<void> MockBackend::release_address_space(AddressReservation reservation) noexcept {
    const std::scoped_lock lock{mutex_};
    for (auto& slot : reservations_) {
        if (slot.id != AddressReservationId{} && slot == reservation) {
            for (const auto& [id, resource] : resources_) {
                static_cast<void>(id);
                if (resource.address >= slot.base &&
                    resource.address.value() - slot.base.value() < slot.size.value()) {
                    return backend_error(ErrorCode::busy, OperationId::release,
                                         reservation.id.value());
                }
            }
            if (should_fail(FaultPoint::address_free)) {
                return backend_error(ErrorCode::backend_failure, OperationId::release,
                                     reservation.id.value());
            }
            slot = {};
            return {};
        }
    }
    return backend_error(ErrorCode::stale_handle, OperationId::release, reservation.id.value());
}

bool MockBackend::owns_address_space(AddressReservationId id) const noexcept {
    const std::scoped_lock lock{mutex_};
    return id != AddressReservationId{} && std::any_of(reservations_.begin(), reservations_.end(),
                                                       [id](const AddressReservation& reservation) {
                                                           return reservation.id == id;
                                                       });
}

const BackendCapabilities& MockBackend::capabilities() const noexcept {
    return config_.capabilities;
}

void MockBackend::inject_failure(FaultPoint point, std::uint64_t invocation) noexcept {
    const std::scoped_lock lock{mutex_};
    const auto index = static_cast<std::size_t>(point);
    if (index >= fail_on_.size()) {
        return;
    }
    fail_on_[index] = invocation;
    calls_[index] = 0U;
}

void MockBackend::clear_failures() noexcept {
    const std::scoped_lock lock{mutex_};
    fail_on_.fill(0U);
    calls_.fill(0U);
}

bool MockBackend::should_fail(FaultPoint point) const noexcept {
    const auto index = static_cast<std::size_t>(point);
    if (calls_[index] != std::numeric_limits<std::uint64_t>::max()) {
        ++calls_[index];
    }
    return fail_on_[index] != 0U && calls_[index] == fail_on_[index];
}

Result<ByteSize> MockBackend::computed_charge(RepresentationState state,
                                              ByteSize logical_size) const noexcept {
    if (!is_valid(state) || logical_size.value() == 0U) {
        return backend_error(ErrorCode::invalid_argument, OperationId::allocate);
    }
    const auto granularity = config_.capabilities.allocation_granularity.value();
    if (granularity == 0U) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::allocate);
    }
    std::uint64_t requested = logical_size.value();
    if (!is_raw(state)) {
        if (!codec_.available()) {
            return backend_error(ErrorCode::unsupported, OperationId::compress);
        }
        const auto bound = codec_.maximum_compressed_size(logical_size);
        if (!bound) {
            return bound.error();
        }
        requested = bound.value().value();
    }
    const auto configured = config_.exact_state_charges[state_index(state)].value();
    if (configured != 0U && (configured % granularity) != 0U) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::allocate,
                             configured);
    }
    requested = std::max(requested, configured);
    const auto aligned = checked_align_up(requested, granularity, OperationId::allocate);
    if (!aligned) {
        return aligned.error();
    }
    return ByteSize{aligned.value()};
}

Result<ByteSize> MockBackend::allocation_bound(RepresentationState state,
                                               ByteSize logical_size) noexcept {
    const std::scoped_lock lock{mutex_};
    auto charge = computed_charge(state, logical_size);
    if (!charge) {
        return charge.error();
    }
    if (should_fail(FaultPoint::advertised_bound_too_small) && charge.value().value() > 1U) {
        return ByteSize{charge.value().value() - 1U};
    }
    if (should_fail(FaultPoint::allocation_bound_padding)) {
        const auto padded =
            checked_add(charge.value().value(), config_.capabilities.allocation_granularity.value(),
                        OperationId::allocate);
        if (!padded) {
            return padded.error();
        }
        return ByteSize{padded.value()};
    }
    return charge;
}

Result<ByteSize> MockBackend::workspace_bound(RepresentationState source,
                                              RepresentationState destination,
                                              ByteSize logical_size) noexcept {
    const std::scoped_lock lock{mutex_};
    if (!is_valid(source) || !is_valid(destination) || logical_size.value() == 0U) {
        return backend_error(ErrorCode::invalid_argument, OperationId::allocate);
    }
    if (is_raw(source) && is_raw(destination)) {
        return ByteSize{};
    }
    if (!codec_.available()) {
        return backend_error(ErrorCode::unsupported, OperationId::compress);
    }
    const auto compressed_bound = codec_.maximum_compressed_size(logical_size);
    if (!compressed_bound) {
        return compressed_bound.error();
    }
    const auto requested = std::max(
        {logical_size.value(), compressed_bound.value().value(), config_.workspace_charge.value()});
    const auto aligned = checked_align_up(
        requested, config_.capabilities.allocation_granularity.value(), OperationId::allocate);
    if (!aligned) {
        return aligned.error();
    }
    return ByteSize{aligned.value()};
}

Result<BackendAllocation>
MockBackend::provisional_descriptor(const Resource& resource) const noexcept {
    BackendAllocation descriptor{resource.id,   resource.tier,     resource.charge,
                                 resource.kind, resource.metadata, resource.address};
    if (should_fail(FaultPoint::provisional_charge_mismatch)) {
        const auto granularity = config_.capabilities.allocation_granularity.value();
        if (resource.kind == ResourceKind::workspace && resource.charge.value() > granularity) {
            descriptor.charge = ByteSize{resource.charge.value() - granularity};
        } else {
            const auto increased =
                checked_add(resource.charge.value(), granularity, OperationId::allocate);
            if (!increased) {
                return increased.error();
            }
            descriptor.charge = ByteSize{increased.value()};
        }
    }
    if (should_fail(FaultPoint::provisional_tier_mismatch)) {
        descriptor.tier =
            resource.tier == PhysicalTier::gpu ? PhysicalTier::host : PhysicalTier::gpu;
    }
    if (should_fail(FaultPoint::provisional_kind_mismatch)) {
        descriptor.kind = resource.kind == ResourceKind::representation
                              ? ResourceKind::workspace
                              : ResourceKind::representation;
    }
    if (should_fail(FaultPoint::provisional_metadata_mismatch)) {
        descriptor.metadata.crc32c ^= 1U;
        if (descriptor.metadata.encoding == Encoding::raw) {
            // Keep this fault structurally plausible: adoption, not local consistency,
            // must disprove the reported RAW integrity metadata against actual storage.
            descriptor.metadata.stored_crc32c = descriptor.metadata.crc32c;
        }
    }
    if (should_fail(FaultPoint::provisional_logical_size_mismatch)) {
        descriptor.metadata.logical_size = ByteSize{};
    }
    if (should_fail(FaultPoint::provisional_stored_size_mismatch)) {
        descriptor.metadata.stored_size = ByteSize{std::numeric_limits<std::uint64_t>::max()};
    }
    if (should_fail(FaultPoint::provisional_encoding_mismatch)) {
        descriptor.metadata.encoding =
            resource.metadata.encoding == Encoding::raw ? Encoding::lz4_block : Encoding::raw;
    }
    if (should_fail(FaultPoint::provisional_zero_charge)) {
        descriptor.charge = ByteSize{};
    }
    if (should_fail(FaultPoint::provisional_unaligned_charge)) {
        descriptor.charge = ByteSize{1U};
    }
    return descriptor;
}

Result<BackendAllocation> MockBackend::allocate(Resource resource, FaultPoint point) noexcept {
    const std::scoped_lock lock{mutex_};
    if (resource.kind == ResourceKind::representation &&
        resource.state == RepresentationState::gpu_raw) {
        const bool contained =
            resource.address != DeviceAddress{} &&
            std::any_of(reservations_.begin(), reservations_.end(),
                        [&](const AddressReservation& reservation) {
                            return reservation.id != AddressReservationId{} &&
                                   resource.address >= reservation.base &&
                                   resource.address.value() - reservation.base.value() <=
                                       reservation.size.value() &&
                                   resource.charge.value() <=
                                       reservation.size.value() -
                                           (resource.address.value() - reservation.base.value());
                        });
        if (!contained || config_.capabilities.mapping_granularity.value() == 0U ||
            resource.address.value() % config_.capabilities.mapping_granularity.value() != 0U) {
            return backend_error(ErrorCode::invalid_range, OperationId::allocate);
        }
        for (const auto& [id, owned] : resources_) {
            static_cast<void>(id);
            if (owned.address != DeviceAddress{} &&
                (owned.address <= resource.address
                     ? resource.address.value() - owned.address.value() < owned.charge.value()
                     : owned.address.value() - resource.address.value() <
                           resource.charge.value())) {
                return backend_error(ErrorCode::busy, OperationId::allocate);
            }
        }
    } else if (resource.address != DeviceAddress{}) {
        return backend_error(ErrorCode::invalid_argument, OperationId::allocate);
    }
    if (should_fail(point) || (point == FaultPoint::workspace_allocation &&
                               should_fail(FaultPoint::verification_allocation))) {
        const auto code =
            point == FaultPoint::gpu_allocation
                ? ErrorCode::out_of_gpu_memory
                : (point == FaultPoint::host_allocation ? ErrorCode::out_of_host_memory
                                                        : ErrorCode::backend_failure);
        return backend_error(code, OperationId::allocate);
    }
    if (should_fail(FaultPoint::invalid_resource_identity)) {
        return BackendAllocation{ResourceId{},  resource.tier,     resource.charge,
                                 resource.kind, resource.metadata, resource.address};
    }
    if (should_fail(FaultPoint::duplicate_resource_identity) && !resources_.empty()) {
        const auto& existing = resources_.begin()->second;
        return BackendAllocation{existing.id,   resource.tier,     resource.charge,
                                 resource.kind, resource.metadata, resource.address};
    }
    if (resources_.size() == resources_.max_size() ||
        next_resource_id_ == std::numeric_limits<std::uint64_t>::max()) {
        return backend_error(ErrorCode::out_of_host_memory, OperationId::allocate);
    }
    resource.id = ResourceId{next_resource_id_};
    ++next_resource_id_;
    resource.awaiting_adoption = true;
    const auto id = resource.id;
    // Fault preparation is fallible only before the resource becomes backend-owned.
    const auto descriptor = provisional_descriptor(resource);
    if (!descriptor) {
        return descriptor.error();
    }
    try {
        const auto [iterator, inserted] = resources_.try_emplace(id.value(), std::move(resource));
        if (!inserted) {
            return backend_error(ErrorCode::backend_contract_violation, OperationId::allocate,
                                 iterator->first);
        }
        last_allocated_resource_ = iterator->second.id;
        return descriptor.value();
    } catch (const std::bad_alloc&) {
        return backend_error(ErrorCode::out_of_host_memory, OperationId::allocate);
    }
}

Result<BackendAllocation>
MockBackend::allocate_representation(RepresentationAllocationRequest request) noexcept {
    const auto state = request.state;
    const auto logical_size = request.logical_size;
    const auto content = request.content;
    const auto charge = computed_charge(state, logical_size);
    if (!charge) {
        return charge.error();
    }
    Resource resource{};
    resource.tier = tier_of(state);
    resource.charge = charge.value();
    resource.kind = ResourceKind::representation;
    resource.state = state;
    resource.content = content;
    resource.address = request.stable_address;
    resource.metadata = RepresentationMetadata{encoding_for(state), logical_size,
                                               is_raw(state) ? logical_size : ByteSize{}, 0U};
    const auto storage_size = checked_storage_size(charge.value().value(), OperationId::allocate);
    const auto data_size = checked_storage_size(logical_size.value(), OperationId::allocate);
    if (!storage_size || !data_size) {
        return storage_size ? data_size.error() : storage_size.error();
    }
    try {
        resource.bytes.resize(storage_size.value(), std::byte{});
    } catch (const std::bad_alloc&) {
        return backend_error(resource.tier == PhysicalTier::gpu ? ErrorCode::out_of_gpu_memory
                                                                : ErrorCode::out_of_host_memory,
                             OperationId::allocate);
    }
    if (is_raw(state)) {
        resource.metadata.crc32c =
            crc32c(std::span<const std::byte>{resource.bytes.data(), data_size.value()});
        resource.metadata.stored_crc32c = resource.metadata.crc32c;
    }
    const auto point = resource.tier == PhysicalTier::gpu ? FaultPoint::gpu_allocation
                                                          : FaultPoint::host_allocation;
    return allocate(std::move(resource), point);
}

Result<BackendAllocation> MockBackend::allocate_workspace(PhysicalTier tier,
                                                          ByteSize charge) noexcept {
    if (!is_valid(tier) || charge.value() == 0U) {
        return backend_error(ErrorCode::invalid_argument, OperationId::allocate);
    }
    Resource resource{};
    resource.tier = tier;
    resource.charge = charge;
    resource.kind = ResourceKind::workspace;
    const auto storage_size = checked_storage_size(charge.value(), OperationId::allocate);
    if (!storage_size) {
        return storage_size.error();
    }
    try {
        resource.bytes.resize(storage_size.value(), std::byte{});
    } catch (const std::bad_alloc&) {
        return backend_error(tier == PhysicalTier::gpu ? ErrorCode::out_of_gpu_memory
                                                       : ErrorCode::out_of_host_memory,
                             OperationId::allocate);
    }
    return allocate(std::move(resource), FaultPoint::workspace_allocation);
}

Result<BackendAllocation>
MockBackend::adopt_allocation(const BackendAllocation& provisional) noexcept {
    const std::scoped_lock lock{mutex_};
    saturating_increment(adoption_attempt_count_);
    if (should_fail(FaultPoint::adoption_failure)) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::allocate,
                             provisional.id.value());
    }
    const auto iterator = resources_.find(provisional.id.value());
    if (iterator == resources_.end() || !iterator->second.awaiting_adoption ||
        iterator->second.id != provisional.id || iterator->second.tier != provisional.tier ||
        iterator->second.charge != provisional.charge ||
        iterator->second.kind != provisional.kind ||
        iterator->second.address != provisional.address) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::allocate,
                             provisional.id.value());
    }
    auto& stored = iterator->second;
    if (stored.kind == ResourceKind::representation &&
        (stored.metadata.encoding != provisional.metadata.encoding ||
         stored.metadata.logical_size != provisional.metadata.logical_size ||
         stored.metadata.stored_size != provisional.metadata.stored_size ||
         stored.metadata.crc32c != provisional.metadata.crc32c ||
         stored.metadata.stored_crc32c != provisional.metadata.stored_crc32c)) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::allocate,
                             provisional.id.value());
    }
    stored.awaiting_adoption = false;
    return BackendAllocation{stored.id,   stored.tier,     stored.charge,
                             stored.kind, stored.metadata, stored.address};
}

bool MockBackend::is_unadopted(ResourceId resource) const noexcept {
    const std::scoped_lock lock{mutex_};
    const auto iterator = resources_.find(resource.value());
    return iterator != resources_.end() && iterator->second.awaiting_adoption;
}

Result<std::span<const std::byte>> MockBackend::logical_bytes(const Resource& resource,
                                                              Resource* workspace) noexcept {
    const auto logical = static_cast<std::size_t>(resource.metadata.logical_size.value());
    if (resource.metadata.stored_size.value() > resource.bytes.size() ||
        crc32c(std::span<const std::byte>{
            resource.bytes.data(),
            static_cast<std::size_t>(resource.metadata.stored_size.value())}) !=
            resource.metadata.stored_crc32c) {
        return backend_error(ErrorCode::integrity_failure, OperationId::verify,
                             resource.id.value());
    }
    if (is_raw(resource.state)) {
        if (resource.metadata.encoding != Encoding::raw ||
            resource.metadata.stored_size != resource.metadata.logical_size ||
            resource.bytes.size() < logical) {
            return backend_error(ErrorCode::integrity_failure, OperationId::verify,
                                 resource.id.value());
        }
        return std::span<const std::byte>{resource.bytes.data(), logical};
    }
    if (resource.metadata.encoding != Encoding::lz4_block || workspace == nullptr ||
        resource.metadata.stored_size.value() > resource.charge.value() ||
        resource.metadata.stored_size.value() > resource.bytes.size() ||
        workspace->bytes.size() < logical) {
        return backend_error(ErrorCode::integrity_failure, OperationId::verify,
                             resource.id.value());
    }
    if (should_fail(FaultPoint::decompression)) {
        return backend_error(ErrorCode::backend_failure, OperationId::decompress,
                             resource.id.value());
    }
    const auto decoded = codec_.decompress(
        std::span<const std::byte>{resource.bytes.data(),
                                   static_cast<std::size_t>(resource.metadata.stored_size.value())},
        std::span<std::byte>{workspace->bytes.data(), logical}, resource.metadata.logical_size);
    if (!decoded || decoded.value() != resource.metadata.logical_size) {
        return decoded ? backend_error(ErrorCode::integrity_failure, OperationId::decompress,
                                       resource.id.value(), decoded.value().value())
                       : decoded.error();
    }
    saturating_increment(successful_decompressions_);
    bytes_decompressed_ =
        saturating_add(bytes_decompressed_, resource.metadata.logical_size.value());
    return std::span<const std::byte>{workspace->bytes.data(), logical};
}

Result<void> MockBackend::verify_locked(const Resource& resource, ContentTag expected,
                                        Resource* workspace) noexcept {
    if (resource.kind != ResourceKind::representation || resource.content != expected ||
        resource.metadata.logical_size.value() == 0U ||
        resource.metadata.encoding != encoding_for(resource.state)) {
        saturating_increment(integrity_failures_);
        return backend_error(ErrorCode::integrity_failure, OperationId::verify,
                             resource.id.value());
    }
    const auto bytes = logical_bytes(resource, workspace);
    if (!bytes) {
        saturating_increment(integrity_failures_);
        return bytes.error();
    }
    const auto actual_crc = crc32c(bytes.value());
    if (should_fail(FaultPoint::crc_comparison) || actual_crc != resource.metadata.crc32c) {
        saturating_increment(integrity_failures_);
        return backend_error(ErrorCode::integrity_failure, OperationId::verify, resource.id.value(),
                             actual_crc);
    }
    return {};
}

Result<TransferReceipt> MockBackend::compact_compressed(Resource& resource, ByteSize stored_size,
                                                        Resource* workspace) noexcept {
    const auto configured = config_.exact_state_charges[state_index(resource.state)].value();
    const auto requested = std::max(stored_size.value(), configured);
    const auto aligned = checked_align_up(
        requested, config_.capabilities.allocation_granularity.value(), OperationId::compress);
    if (!aligned || workspace == nullptr || workspace->kind != ResourceKind::workspace ||
        aligned.value() > resource.bytes.size() || aligned.value() > workspace->bytes.size()) {
        return aligned ? backend_error(ErrorCode::backend_contract_violation, OperationId::compress,
                                       resource.id.value(), aligned.value())
                       : aligned.error();
    }
    if (should_fail(FaultPoint::final_storage_allocation)) {
        return backend_error(resource.tier == PhysicalTier::gpu ? ErrorCode::out_of_gpu_memory
                                                                : ErrorCode::out_of_host_memory,
                             OperationId::compress, resource.id.value());
    }
    const auto storage_size = checked_storage_size(aligned.value(), OperationId::compress);
    if (!storage_size) {
        return storage_size.error();
    }
    workspace->bytes.resize(storage_size.value());
    std::copy_n(resource.bytes.begin(), static_cast<std::size_t>(stored_size.value()),
                workspace->bytes.begin());
    resource.bytes.swap(workspace->bytes);
    resource.charge = ByteSize{aligned.value()};
    resource.metadata.stored_size = stored_size;
    return TransferReceipt{resource.charge, resource.metadata};
}

Result<TransferReceipt> MockBackend::transfer(ResourceId source_id, ResourceId destination_id,
                                              ResourceId workspace_id,
                                              CompactionTarget compaction) noexcept {
    if (compaction.resource != ResourceId{}) {
        return backend_error(ErrorCode::unsupported, OperationId::migrate);
    }
    const std::scoped_lock lock{mutex_};
    last_transfer_destination_ = destination_id;
    if (should_fail(FaultPoint::ambiguous_transfer)) {
        return backend_error(ErrorCode::ambiguous_backend_state, OperationId::migrate,
                             destination_id.value());
    }
    if (should_fail(FaultPoint::transfer)) {
        return backend_error(ErrorCode::backend_failure, OperationId::migrate,
                             destination_id.value());
    }
    const auto source_iterator = resources_.find(source_id.value());
    const auto destination_iterator = resources_.find(destination_id.value());
    if (source_iterator == resources_.end() || destination_iterator == resources_.end() ||
        source_iterator->second.kind != ResourceKind::representation ||
        destination_iterator->second.kind != ResourceKind::representation) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::migrate);
    }
    auto& source = source_iterator->second;
    auto& destination = destination_iterator->second;
    const auto admitted_charge = destination.charge;
    Resource* workspace = nullptr;
    if (workspace_id.value() != 0U) {
        const auto workspace_iterator = resources_.find(workspace_id.value());
        if (workspace_iterator == resources_.end() ||
            workspace_iterator->second.kind != ResourceKind::workspace) {
            return backend_error(ErrorCode::backend_contract_violation, OperationId::migrate,
                                 workspace_id.value());
        }
        workspace = &workspace_iterator->second;
    }
    destination.content = source.content;
    destination.metadata.logical_size = source.metadata.logical_size;
    destination.metadata.crc32c = source.metadata.crc32c;

    if (is_raw(source.state) && is_raw(destination.state)) {
        if (should_fail(FaultPoint::raw_copy) ||
            destination.bytes.size() <
                static_cast<std::size_t>(source.metadata.logical_size.value())) {
            return backend_error(ErrorCode::backend_failure, OperationId::migrate,
                                 destination.id.value());
        }
        std::copy_n(source.bytes.begin(),
                    static_cast<std::size_t>(source.metadata.logical_size.value()),
                    destination.bytes.begin());
        destination.metadata.encoding = Encoding::raw;
        destination.metadata.stored_size = source.metadata.logical_size;
    } else if (is_raw(source.state) && !is_raw(destination.state)) {
        if (should_fail(FaultPoint::compression)) {
            return backend_error(ErrorCode::backend_failure, OperationId::compress,
                                 destination.id.value());
        }
        const auto written =
            codec_.compress(std::span<const std::byte>{source.bytes.data(),
                                                       static_cast<std::size_t>(
                                                           source.metadata.logical_size.value())},
                            destination.bytes);
        if (!written) {
            return written.error();
        }
        destination.metadata.encoding = Encoding::lz4_block;
        const auto compacted = compact_compressed(destination, written.value(), workspace);
        if (!compacted) {
            return compacted.error();
        }
        saturating_increment(successful_compressions_);
        bytes_compressed_ = saturating_add(bytes_compressed_, source.metadata.logical_size.value());
    } else if (!is_raw(source.state) && is_raw(destination.state)) {
        if (workspace == nullptr || should_fail(FaultPoint::decompression)) {
            return backend_error(ErrorCode::backend_failure, OperationId::decompress,
                                 source.id.value());
        }
        const auto decoded = codec_.decompress(
            std::span<const std::byte>{
                source.bytes.data(), static_cast<std::size_t>(source.metadata.stored_size.value())},
            destination.bytes, source.metadata.logical_size);
        if (!decoded || decoded.value() != source.metadata.logical_size ||
            should_fail(FaultPoint::decompression_size_mismatch)) {
            return decoded ? backend_error(ErrorCode::integrity_failure, OperationId::decompress,
                                           source.id.value(), decoded.value().value())
                           : decoded.error();
        }
        destination.metadata.encoding = Encoding::raw;
        destination.metadata.stored_size = source.metadata.logical_size;
        saturating_increment(successful_decompressions_);
        bytes_decompressed_ =
            saturating_add(bytes_decompressed_, source.metadata.logical_size.value());
    } else {
        if (should_fail(FaultPoint::compressed_copy) ||
            source.metadata.stored_size.value() > destination.bytes.size()) {
            return backend_error(ErrorCode::backend_failure, OperationId::migrate,
                                 destination.id.value());
        }
        std::copy_n(source.bytes.begin(),
                    static_cast<std::size_t>(source.metadata.stored_size.value()),
                    destination.bytes.begin());
        destination.metadata.encoding = Encoding::lz4_block;
        const auto compacted =
            compact_compressed(destination, source.metadata.stored_size, workspace);
        if (!compacted) {
            return compacted.error();
        }
    }
    destination.metadata.stored_crc32c = crc32c(std::span<const std::byte>{
        destination.bytes.data(),
        static_cast<std::size_t>(destination.metadata.stored_size.value())});
    if (should_fail(FaultPoint::corrupt_content) && !destination.bytes.empty()) {
        destination.bytes[0] ^= std::byte{1U};
    }
    // All error returns precede compaction. This fixed-size receipt cannot fail to allocate.
    TransferReceipt receipt{destination.charge, destination.metadata};
    if (should_fail(FaultPoint::receipt_logical_size_mismatch)) {
        receipt.metadata.logical_size = ByteSize{};
    }
    if (should_fail(FaultPoint::receipt_stored_size_mismatch)) {
        receipt.metadata.stored_size = ByteSize{std::numeric_limits<std::uint64_t>::max()};
    }
    if (should_fail(FaultPoint::receipt_encoding_mismatch)) {
        receipt.metadata.encoding = is_raw(destination.state) ? Encoding::lz4_block : Encoding::raw;
    }
    if (should_fail(FaultPoint::receipt_charge_exceeds_admission)) {
        receipt.charge = ByteSize{std::numeric_limits<std::uint64_t>::max()};
    }
    if (should_fail(FaultPoint::receipt_zero_charge)) {
        receipt.charge = ByteSize{};
    }
    if (should_fail(FaultPoint::receipt_unaligned_charge)) {
        receipt.charge = ByteSize{1U};
    }
    // This fault needs at least one granule of compaction slack. It deliberately lies without
    // changing the resource and still passes all local receipt checks. Delivery cannot fail.
    const auto granularity = config_.capabilities.allocation_granularity.value();
    if (should_fail(FaultPoint::receipt_plausible_charge_mismatch) && granularity != 0U &&
        destination.charge <= admitted_charge &&
        granularity <= admitted_charge.value() - destination.charge.value()) {
        receipt.charge = ByteSize{destination.charge.value() + granularity};
    }
    return receipt;
}

Result<void> MockBackend::verify(ResourceId resource_id, ContentTag expected,
                                 ResourceId workspace_id) noexcept {
    const std::scoped_lock lock{mutex_};
    if (should_fail(FaultPoint::verification)) {
        saturating_increment(integrity_failures_);
        return backend_error(ErrorCode::integrity_failure, OperationId::verify,
                             resource_id.value());
    }
    const auto iterator = resources_.find(resource_id.value());
    if (iterator == resources_.end()) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::verify,
                             resource_id.value());
    }
    Resource* workspace = nullptr;
    if (workspace_id.value() != 0U) {
        const auto workspace_iterator = resources_.find(workspace_id.value());
        if (workspace_iterator == resources_.end()) {
            return backend_error(ErrorCode::backend_contract_violation, OperationId::verify,
                                 workspace_id.value());
        }
        workspace = &workspace_iterator->second;
    }
    return verify_locked(iterator->second, expected, workspace);
}

Result<void> MockBackend::verify_authoritative(ResourceId resource_id, ContentTag expected,
                                               ResourceId workspace_id) noexcept {
    const std::scoped_lock lock{mutex_};
    const auto iterator = resources_.find(resource_id.value());
    if (iterator == resources_.end()) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::verify,
                             resource_id.value());
    }
    Resource* workspace = nullptr;
    if (workspace_id.value() != 0U) {
        const auto workspace_iterator = resources_.find(workspace_id.value());
        if (workspace_iterator == resources_.end()) {
            return backend_error(ErrorCode::backend_contract_violation, OperationId::verify,
                                 workspace_id.value());
        }
        workspace = &workspace_iterator->second;
    }
    return verify_locked(iterator->second, expected, workspace);
}

Result<void> MockBackend::verify_transfer(ResourceId source_id, ResourceId destination_id,
                                          ContentTag expected, ResourceId workspace_id) noexcept {
    const std::scoped_lock lock{mutex_};
    if (should_fail(FaultPoint::verification_contract)) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::verify,
                             destination_id.value());
    }
    if (should_fail(FaultPoint::verification)) {
        saturating_increment(integrity_failures_);
        return backend_error(ErrorCode::integrity_failure, OperationId::verify,
                             destination_id.value());
    }
    const auto source_iterator = resources_.find(source_id.value());
    const auto destination_iterator = resources_.find(destination_id.value());
    if (source_iterator == resources_.end() || destination_iterator == resources_.end()) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::verify);
    }
    Resource* workspace = nullptr;
    if (workspace_id.value() != 0U) {
        const auto workspace_iterator = resources_.find(workspace_id.value());
        if (workspace_iterator == resources_.end()) {
            return backend_error(ErrorCode::backend_contract_violation, OperationId::verify,
                                 workspace_id.value());
        }
        workspace = &workspace_iterator->second;
    }
    const auto& source = source_iterator->second;
    const auto& destination = destination_iterator->second;
    const auto destination_valid = verify_locked(destination, expected, workspace);
    if (!destination_valid) {
        return destination_valid.error();
    }
    if (!is_raw(source.state) && !is_raw(destination.state)) {
        const bool same_metadata =
            source.metadata.encoding == destination.metadata.encoding &&
            source.metadata.logical_size == destination.metadata.logical_size &&
            source.metadata.stored_size == destination.metadata.stored_size &&
            source.metadata.crc32c == destination.metadata.crc32c;
        const auto payload_end =
            source.bytes.begin() + static_cast<std::ptrdiff_t>(source.metadata.stored_size.value());
        const bool same_payload = same_metadata && std::equal(source.bytes.begin(), payload_end,
                                                              destination.bytes.begin());
        if (!same_payload || should_fail(FaultPoint::byte_comparison)) {
            saturating_increment(integrity_failures_);
            return backend_error(ErrorCode::integrity_failure, OperationId::verify,
                                 destination.id.value());
        }
        return {};
    }
    if (is_raw(source.state) && is_raw(destination.state)) {
        const bool same_metadata =
            source.metadata.logical_size == destination.metadata.logical_size &&
            source.metadata.stored_size == destination.metadata.stored_size &&
            source.metadata.crc32c == destination.metadata.crc32c;
        const auto source_end = source.bytes.begin() +
                                static_cast<std::ptrdiff_t>(source.metadata.logical_size.value());
        if (!same_metadata || should_fail(FaultPoint::byte_comparison) ||
            !std::equal(source.bytes.begin(), source_end, destination.bytes.begin())) {
            saturating_increment(integrity_failures_);
            return backend_error(ErrorCode::integrity_failure, OperationId::verify,
                                 destination.id.value());
        }
        return {};
    }
    const Resource& raw = is_raw(source.state) ? source : destination;
    const Resource& encoded = is_raw(source.state) ? destination : source;
    if (workspace == nullptr || workspace->bytes.size() < encoded.metadata.logical_size.value()) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::verify,
                             workspace_id.value());
    }
    if (should_fail(FaultPoint::decompression)) {
        return backend_error(ErrorCode::backend_failure, OperationId::decompress,
                             encoded.id.value());
    }
    const auto decoded = codec_.decompress(
        std::span<const std::byte>{encoded.bytes.data(),
                                   static_cast<std::size_t>(encoded.metadata.stored_size.value())},
        std::span<std::byte>{workspace->bytes.data(),
                             static_cast<std::size_t>(encoded.metadata.logical_size.value())},
        encoded.metadata.logical_size);
    if (!decoded || decoded.value() != raw.metadata.logical_size ||
        should_fail(FaultPoint::decompression_size_mismatch)) {
        return decoded ? backend_error(ErrorCode::integrity_failure, OperationId::decompress,
                                       encoded.id.value())
                       : decoded.error();
    }
    saturating_increment(successful_decompressions_);
    bytes_decompressed_ =
        saturating_add(bytes_decompressed_, encoded.metadata.logical_size.value());
    const auto raw_end =
        raw.bytes.begin() + static_cast<std::ptrdiff_t>(raw.metadata.logical_size.value());
    const bool same = std::equal(raw.bytes.begin(), raw_end, workspace->bytes.begin());
    const auto verified_crc = crc32c(std::span<const std::byte>{
        workspace->bytes.data(), static_cast<std::size_t>(encoded.metadata.logical_size.value())});
    if (!same || should_fail(FaultPoint::byte_comparison) ||
        verified_crc != encoded.metadata.crc32c) {
        saturating_increment(integrity_failures_);
        return backend_error(ErrorCode::integrity_failure, OperationId::verify,
                             destination.id.value());
    }
    return {};
}

Result<BackendAllocation> MockBackend::allocation(ResourceId resource) const noexcept {
    const std::scoped_lock lock{mutex_};
    const auto iterator = resources_.find(resource.value());
    if (iterator == resources_.end()) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::allocate,
                             resource.value());
    }
    if (should_fail(FaultPoint::stored_size_contract)) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::allocate,
                             resource.value(), iterator->second.charge.value());
    }
    auto reported_charge = iterator->second.charge;
    if (should_fail(FaultPoint::allocation_descriptor_mismatch)) {
        const auto granularity = config_.capabilities.allocation_granularity.value();
        if (granularity <= std::numeric_limits<std::uint64_t>::max() - reported_charge.value()) {
            reported_charge = ByteSize{reported_charge.value() + granularity};
        } else {
            reported_charge = ByteSize{};
        }
    }
    const auto reported_id =
        should_fail(FaultPoint::allocation_identity_mismatch) ? ResourceId{} : iterator->second.id;
    const auto reported_tier =
        should_fail(FaultPoint::allocation_tier_mismatch)
            ? (iterator->second.tier == PhysicalTier::gpu ? PhysicalTier::host : PhysicalTier::gpu)
            : iterator->second.tier;
    const auto reported_kind = should_fail(FaultPoint::allocation_kind_mismatch)
                                   ? ResourceKind::workspace
                                   : iterator->second.kind;
    auto reported_metadata = iterator->second.metadata;
    if (should_fail(FaultPoint::allocation_logical_size_mismatch)) {
        reported_metadata.logical_size = ByteSize{};
    }
    if (should_fail(FaultPoint::allocation_stored_size_mismatch)) {
        reported_metadata.stored_size = ByteSize{};
    }
    if (should_fail(FaultPoint::allocation_encoding_mismatch)) {
        reported_metadata.encoding =
            reported_metadata.encoding == Encoding::raw ? Encoding::lz4_block : Encoding::raw;
    }
    if (should_fail(FaultPoint::allocation_crc_mismatch)) {
        reported_metadata.crc32c ^= 1U;
    }
    return BackendAllocation{reported_id,   reported_tier,     reported_charge,
                             reported_kind, reported_metadata, iterator->second.address};
}

Result<void> MockBackend::read_bytes(ResourceId resource, ByteOffset offset,
                                     std::span<std::byte> output) const noexcept {
    const std::scoped_lock lock{mutex_};
    const auto iterator = resources_.find(resource.value());
    if (iterator == resources_.end() || !is_raw(iterator->second.state)) {
        return backend_error(ErrorCode::stale_handle, OperationId::acquire, resource.value());
    }
    const auto end = checked_add(offset.value(), static_cast<std::uint64_t>(output.size()),
                                 OperationId::acquire);
    if (!end || end.value() > iterator->second.metadata.logical_size.value()) {
        return end ? backend_error(ErrorCode::invalid_range, OperationId::acquire, resource.value(),
                                   end.value())
                   : end.error();
    }
    std::copy_n(iterator->second.bytes.begin() + static_cast<std::ptrdiff_t>(offset.value()),
                output.size(), output.begin());
    return {};
}

Result<ContentTag> MockBackend::write_bytes(ResourceId resource, ByteOffset offset,
                                            std::span<const std::byte> input,
                                            ContentTag previous) noexcept {
    const std::scoped_lock lock{mutex_};
    if (should_fail(FaultPoint::write_content)) {
        return backend_error(ErrorCode::backend_failure, OperationId::close_lease,
                             resource.value());
    }
    const auto iterator = resources_.find(resource.value());
    if (iterator == resources_.end() || !is_raw(iterator->second.state) ||
        iterator->second.content != previous) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::close_lease,
                             resource.value());
    }
    if (previous.generation == std::numeric_limits<std::uint64_t>::max()) {
        return backend_error(ErrorCode::arithmetic_overflow, OperationId::close_lease,
                             resource.value());
    }
    const auto end = checked_add(offset.value(), static_cast<std::uint64_t>(input.size()),
                                 OperationId::close_lease);
    if (!end || end.value() > iterator->second.metadata.logical_size.value()) {
        return end ? backend_error(ErrorCode::invalid_range, OperationId::close_lease,
                                   resource.value(), end.value())
                   : end.error();
    }
    auto& stored = iterator->second;
    std::copy(input.begin(), input.end(),
              stored.bytes.begin() + static_cast<std::ptrdiff_t>(offset.value()));
    stored.metadata.crc32c = crc32c(std::span<const std::byte>{
        stored.bytes.data(), static_cast<std::size_t>(stored.metadata.logical_size.value())});
    stored.metadata.stored_crc32c = stored.metadata.crc32c;
    stored.content = next_content_tag(previous);
    return stored.content;
}

Result<void> MockBackend::write_content(ResourceId resource, ContentTag content) noexcept {
    const std::scoped_lock lock{mutex_};
    if (should_fail(FaultPoint::write_content)) {
        return backend_error(ErrorCode::backend_failure, OperationId::close_lease,
                             resource.value());
    }
    const auto iterator = resources_.find(resource.value());
    if (iterator == resources_.end()) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::close_lease,
                             resource.value());
    }
    iterator->second.content = content;
    return {};
}

Result<void> MockBackend::release(ResourceId resource, ReleasePhase phase) noexcept {
    const std::scoped_lock lock{mutex_};
    last_release_resource_ = resource;
    if (phase == ReleasePhase::rollback) {
        last_rollback_resource_ = resource;
    }
    const auto point = phase == ReleasePhase::rollback
                           ? FaultPoint::rollback_cleanup
                           : (phase == ReleasePhase::post_commit ? FaultPoint::post_commit_cleanup
                                                                 : FaultPoint::close_cleanup);
    if (should_fail(point)) {
        return backend_error(ErrorCode::backend_failure, OperationId::release, resource.value());
    }
    const auto iterator = resources_.find(resource.value());
    if (iterator == resources_.end()) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::release,
                             resource.value());
    }
    if (phase == ReleasePhase::rollback &&
        should_fail(iterator->second.kind == ResourceKind::workspace
                        ? FaultPoint::rollback_workspace_cleanup
                        : FaultPoint::rollback_destination_cleanup)) {
        return backend_error(ErrorCode::backend_failure, OperationId::release, resource.value());
    }
    if (should_fail(FaultPoint::false_release_success)) {
        return {};
    }
    resources_.erase(iterator);
    return {};
}

bool MockBackend::owns(ResourceId resource) const noexcept {
    const std::scoped_lock lock{mutex_};
    return resources_.contains(resource.value());
}

ResourceId MockBackend::last_transfer_destination() const noexcept {
    const std::scoped_lock lock{mutex_};
    return last_transfer_destination_;
}

ResourceId MockBackend::last_rollback_resource() const noexcept {
    const std::scoped_lock lock{mutex_};
    return last_rollback_resource_;
}

ResourceId MockBackend::last_allocated_resource() const noexcept {
    const std::scoped_lock lock{mutex_};
    return last_allocated_resource_;
}

ResourceId MockBackend::last_release_resource() const noexcept {
    const std::scoped_lock lock{mutex_};
    return last_release_resource_;
}

std::uint64_t MockBackend::adoption_attempt_count() const noexcept {
    const std::scoped_lock lock{mutex_};
    return adoption_attempt_count_;
}

ByteSize MockBackend::owned_charge(PhysicalTier tier) const noexcept {
    const std::scoped_lock lock{mutex_};
    std::uint64_t total_charge = 0U;
    for (const auto& [id, resource] : resources_) {
        static_cast<void>(id);
        if (resource.tier == tier) {
            total_charge = saturating_add(total_charge, resource.charge.value());
        }
    }
    return ByteSize{total_charge};
}

std::uint64_t MockBackend::owned_resource_count() const noexcept {
    const std::scoped_lock lock{mutex_};
    return static_cast<std::uint64_t>(resources_.size());
}

Result<ContentTag> MockBackend::content(ResourceId resource) const noexcept {
    const std::scoped_lock lock{mutex_};
    const auto iterator = resources_.find(resource.value());
    if (iterator == resources_.end()) {
        return backend_error(ErrorCode::backend_contract_violation, OperationId::verify,
                             resource.value());
    }
    return iterator->second.content;
}

Result<void> MockBackend::corrupt(ResourceId resource, ResourceCorruption corruption) noexcept {
    const std::scoped_lock lock{mutex_};
    const auto iterator = resources_.find(resource.value());
    if (iterator == resources_.end() || iterator->second.kind != ResourceKind::representation) {
        return backend_error(ErrorCode::stale_handle, OperationId::verify, resource.value());
    }
    auto& value = iterator->second;
    try {
        switch (corruption) {
        case ResourceCorruption::payload_flip:
            if (value.metadata.stored_size.value() == 0U) {
                return backend_error(ErrorCode::invalid_argument, OperationId::verify,
                                     resource.value());
            }
            value.bytes[0] ^= std::byte{1U};
            break;
        case ResourceCorruption::payload_multiple_bits:
            if (value.metadata.stored_size.value() == 0U) {
                return backend_error(ErrorCode::invalid_argument, OperationId::verify,
                                     resource.value());
            }
            value.bytes[0] ^= std::byte{0xA5U};
            if (value.metadata.stored_size.value() > 1U) {
                value.bytes[1] ^= std::byte{0x5AU};
            }
            break;
        case ResourceCorruption::truncate_payload:
            if (value.metadata.stored_size.value() == 0U) {
                return backend_error(ErrorCode::invalid_argument, OperationId::verify,
                                     resource.value());
            }
            value.metadata.stored_size = ByteSize{value.metadata.stored_size.value() - 1U};
            break;
        case ResourceCorruption::append_payload: {
            if (value.metadata.stored_size.value() < value.bytes.size()) {
                value.bytes[static_cast<std::size_t>(value.metadata.stored_size.value())] =
                    std::byte{0x5AU};
            }
            const auto appended =
                checked_add(value.metadata.stored_size.value(), 1U, OperationId::verify);
            if (!appended) {
                return appended.error();
            }
            value.metadata.stored_size = ByteSize{appended.value()};
            break;
        }
        case ResourceCorruption::logical_size: {
            const auto invalid_size =
                checked_add(value.metadata.logical_size.value(), 1U, OperationId::verify);
            if (!invalid_size) {
                return invalid_size.error();
            }
            value.metadata.logical_size = ByteSize{invalid_size.value()};
            break;
        }
        case ResourceCorruption::stored_size: {
            const auto invalid_size = checked_add(value.charge.value(), 1U, OperationId::verify);
            if (!invalid_size) {
                return invalid_size.error();
            }
            value.metadata.stored_size = ByteSize{invalid_size.value()};
            break;
        }
        case ResourceCorruption::crc32c:
            value.metadata.crc32c ^= 1U;
            break;
        case ResourceCorruption::encoding:
            value.metadata.encoding =
                value.metadata.encoding == Encoding::raw ? Encoding::lz4_block : Encoding::raw;
            break;
        case ResourceCorruption::content_tag:
            value.content.fingerprint ^= 1U;
            break;
        }
    } catch (const std::bad_alloc&) {
        return backend_error(ErrorCode::out_of_host_memory, OperationId::verify, resource.value());
    }
    return {};
}

CompressionStats MockBackend::compression_stats() const noexcept {
    const std::scoped_lock lock{mutex_};
    CompressionStats stats{};
    stats.successful_compressions = successful_compressions_;
    stats.successful_decompressions = successful_decompressions_;
    stats.integrity_failures = integrity_failures_;
    stats.bytes_compressed = ByteSize{bytes_compressed_};
    stats.bytes_decompressed = ByteSize{bytes_decompressed_};
    return stats;
}

bool MockBackend::compression_available() const noexcept {
    const std::scoped_lock lock{mutex_};
    return codec_.available();
}

} // namespace vramz
