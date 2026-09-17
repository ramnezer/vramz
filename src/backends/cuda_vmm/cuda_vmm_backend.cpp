#include "vramz/detail/cuda_vmm_backend.hpp"

#include "vramz/checked.hpp"
#include "vramz/crc32c.hpp"
#include "vramz/detail/compression.hpp"
#include "vramz/saturating.hpp"

#include <algorithm>
#include <bit>
#include <exception>
#include <functional>
#include <limits>
#include <new>
#include <utility>

namespace vramz::detail {
namespace {

[[nodiscard]] Error unsupported(OperationId operation) noexcept {
    return make_error(ErrorCode::unsupported, operation);
}
[[nodiscard]] Error contract(ResourceId id = {}) noexcept {
    return make_error(ErrorCode::backend_contract_violation, OperationId::verify, id.value());
}
[[nodiscard]] bool metadata_equal(RepresentationMetadata left,
                                  RepresentationMetadata right) noexcept {
    return left.encoding == right.encoding && left.logical_size == right.logical_size &&
           left.stored_size == right.stored_size && left.crc32c == right.crc32c &&
           left.stored_crc32c == right.stored_crc32c;
}
[[nodiscard]] bool allocation_equal(const BackendAllocation& left,
                                    const BackendAllocation& right) noexcept {
    return left.id == right.id && left.tier == right.tier && left.charge == right.charge &&
           left.kind == right.kind && left.address == right.address &&
           (left.kind == ResourceKind::workspace || metadata_equal(left.metadata, right.metadata));
}
[[nodiscard]] Result<std::vector<std::byte>> byte_buffer(ByteSize size) noexcept {
    if (size.value() > std::numeric_limits<std::size_t>::max()) {
        return make_error(ErrorCode::arithmetic_overflow, OperationId::allocate);
    }
    try {
        std::vector<std::byte> bytes;
        if (size.value() > bytes.max_size()) {
            return make_error(ErrorCode::out_of_host_memory, OperationId::allocate);
        }
        bytes.resize(static_cast<std::size_t>(size.value()));
        return bytes;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::allocate);
    }
}

} // namespace

class CudaVmmBackend::ContextScope final {
  public:
    explicit ContextScope(const CudaVmmBackend& backend) noexcept
        : backend_(backend),
          entered_(backend.context_ == CudaContext{}
                       ? Result<void>{make_error(ErrorCode::shutting_down, OperationId::acquire)}
                       : backend.driver_->push_context(backend.context_)) {
        if (!entered_) {
            backend_.reject_ambiguity(entered_.error());
        }
    }
    ~ContextScope() noexcept {
        if (entered_) {
            const auto popped = backend_.driver_->pop_context();
            if (!popped) {
                backend_.fatal(popped.error());
            }
            if (popped.value() != backend_.context_) {
                backend_.fatal(contract());
            }
        }
    }
    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;
    [[nodiscard]] Result<void> status() const noexcept { return entered_; }

  private:
    const CudaVmmBackend& backend_;
    Result<void> entered_;
};

CudaVmmBackend::CudaVmmBackend(ConstructionKey, std::unique_ptr<CudaDriverApi> driver,
                               std::unique_ptr<GpuCompressionApi> compression) noexcept
    : driver_(std::move(driver)), compression_(std::move(compression)) {}

Result<std::unique_ptr<CudaVmmBackend>>
CudaVmmBackend::create(std::unique_ptr<CudaDriverApi> driver, std::int32_t ordinal,
                       std::unique_ptr<GpuCompressionApi> compression, CudaProbeInfo* probe_info,
                       CudaCompressionAdmission* admission, GpuCodecAudit* audit) noexcept {
    if (!driver || ordinal < 0) {
        return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    }
    try {
        // Unique ownership is established before any driver resource is acquired.
        auto backend = std::make_unique<CudaVmmBackend>(ConstructionKey{}, std::move(driver),
                                                        std::move(compression));
        backend->codec_audit_ = audit;
        const auto ready = backend->probe(ordinal, probe_info, admission);
        if (!ready) {
            return ready.error();
        }
        return backend;
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::runtime_create);
    }
}

Result<void> CudaVmmBackend::probe(std::int32_t ordinal, CudaProbeInfo* info,
                                   CudaCompressionAdmission* admission) noexcept {
    if (admission != nullptr &&
        (!compression_ || info == nullptr || admission->logical_size == ByteSize{} ||
         admission->physical_cap == ByteSize{})) {
        return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    }
    const auto initialized = driver_->initialize();
    if (!initialized) {
        return initialized.error();
    }
    const auto version = driver_->version();
    if (!version) {
        return version.error();
    }
    driver_version_ = version.value();
    if (info != nullptr) {
        info->driver_version = driver_version_;
    }
    if (driver_version_ < 11020 ||
        (admission != nullptr && (driver_version_ < admission->minimum_driver_version ||
                                  (admission->required_driver_family != 0 &&
                                   driver_version_ / 1000 != admission->required_driver_family)))) {
        return unsupported(OperationId::runtime_create);
    }
    const auto selected = driver_->device(ordinal);
    if (!selected) {
        return selected.error();
    }
    device_ = selected.value();
    if (info != nullptr) {
        const auto device_info = driver_->device_info(device_);
        if (!device_info) {
            reject_ambiguity(device_info.error());
            return device_info.error();
        }
        info->device = device_info.value();
        const auto name_end = std::ranges::find(info->device.name, '\0');
        const std::string_view name{info->device.name.data(),
                                    static_cast<std::size_t>(name_end - info->device.name.begin())};
        if (admission != nullptr && !admission->required_device_name.empty() &&
            name != admission->required_device_name) {
            return unsupported(OperationId::runtime_create);
        }
        if (info->device.compute_mode == 2 || info->device.compute_mode < 0 ||
            info->device.compute_mode > 3) {
            return unsupported(OperationId::runtime_create);
        }
    }
    const auto uva = driver_->attribute(device_, CudaAttribute::unified_addressing);
    if (!uva) {
        return uva.error();
    }
    const auto vmm = driver_->attribute(device_, CudaAttribute::virtual_memory_management);
    if (!vmm) {
        return vmm.error();
    }
    if (info != nullptr) {
        info->uva = uva.value();
        info->vmm = vmm.value();
    }
    if (!uva.value() || !vmm.value()) {
        return unsupported(OperationId::runtime_create);
    }
    const auto minimum = driver_->granularity(device_, false);
    const auto page = driver_->host_page_size();
    if (!minimum) {
        return minimum.error();
    }
    if (!page) {
        return page.error();
    }
    if (info != nullptr) {
        info->minimum = minimum.value();
    }
    if (!std::has_single_bit(minimum.value().value()) ||
        !std::has_single_bit(page.value().value())) {
        return make_error(ErrorCode::invalid_alignment, OperationId::runtime_create);
    }
    capabilities_ =
        BackendCapabilities{page.value(), minimum.value(), minimum.value(), true, false, false};
    capabilities_.backend_id = BackendId{2U};
    capabilities_.supports_external_async_completion = driver_->supports_external_completions();
    if (compression_) {
        if (!cpu_compression_codec().available()) {
            return unsupported(OperationId::runtime_create);
        }
        capabilities_.maximum_chunk_size = gpu_lz4_max_chunk;
        for (const auto direction : {GpuCodecDirection::compress, GpuCodecDirection::decompress}) {
            const auto plan = compression_plan(
                direction, admission != nullptr ? admission->logical_size : gpu_lz4_max_chunk);
            if (!plan) {
                return plan.error();
            }
            codec_alignment_ = std::max(codec_alignment_, plan.value().address_alignment);
        }
        capabilities_.mapping_granularity = std::max(minimum.value(), codec_alignment_);
        if (admission != nullptr) {
            const auto admitted = admit_compression(*admission);
            if (!admitted) {
                return admitted.error();
            }
        }
    }
    const auto recommended = driver_->granularity(device_, true);
    if (recommended) {
        recommended_ = recommended.value();
        capabilities_.recommended_allocation_granularity = recommended_;
        if (info != nullptr) {
            info->recommended = recommended_;
        }
    } else {
        if (info != nullptr) {
            reject_ambiguity(recommended.error());
            return recommended.error();
        }
        errors_.push(recommended.error());
    }
    const auto context = driver_->retain_primary(device_);
    if (!context) {
        reject_ambiguity(context.error());
        return context.error();
    }
    if (context.value() == CudaContext{}) {
        fatal(contract());
    }
    context_ = context.value();
    if (codec_audit_ != nullptr) {
        codec_audit_->context_owned = true;
    }
    if (compression_) {
        const ContextScope scope{*this};
        if (!scope.status()) {
            return scope.status().error();
        }
        const auto ready = compression_->initialize(context_);
        if (!ready) {
            reject_ambiguity(ready.error());
            return ready.error();
        }
        stream_initialized_ = true;
        if (codec_audit_ != nullptr) {
            codec_audit_->stream_owned = true;
        }
    }
    return {};
}

Result<void> CudaVmmBackend::admit_compression(CudaCompressionAdmission& admission) noexcept {
    const auto raw = allocation_bound(RepresentationState::gpu_raw, admission.logical_size);
    const auto compressed = compression_plan(GpuCodecDirection::compress, admission.logical_size);
    const auto restored = compression_plan(GpuCodecDirection::decompress, admission.logical_size);
    if (!raw || !compressed || !restored) {
        return !raw ? raw.error() : (!compressed ? compressed.error() : restored.error());
    }
    admission.raw_charge = raw.value();
    admission.compression = compressed.value();
    admission.decompression = restored.value();
    std::uint64_t peak{};
    // Admit simultaneous ownership, including a full-bound compaction reservation and
    // both workspaces. Actual exact compaction can only reduce this conservative peak.
    for (const auto charge : {raw.value(), raw.value(), compressed.value().output_charge,
                              compressed.value().output_charge, compressed.value().workspace_charge,
                              restored.value().workspace_charge}) {
        const auto sum = checked_add(peak, charge.value(), OperationId::allocate);
        if (!sum) {
            return sum.error();
        }
        peak = sum.value();
    }
    admission.peak = ByteSize{peak};
    if (admission.peak > admission.physical_cap) {
        return unsupported(OperationId::allocate);
    }
    physical_cap_ = admission.physical_cap;
    planned_logical_ = admission.logical_size;
    admitted_plans_ = {compressed.value(), restored.value()};
    admission.proven = true;
    return {};
}

CudaVmmBackend::~CudaVmmBackend() {
    if (std::ranges::any_of(completions_, [](auto id) { return id != CompletionTokenId{}; })) {
        fatal(make_error(ErrorCode::ambiguous_backend_state, OperationId::release_completion));
    }
    // Explicit shutdown normally emptied these owners. A direct internal owner still gets
    // bounded RAII cleanup; persistent or ambiguous cleanup never silently abandons ownership.
    for (const auto& resource : resources_) {
        if (resource.state != MappingState::empty) {
            const auto released = release(resource.allocation.id, ReleasePhase::close);
            if (!released) {
                fatal(released.error());
            }
        }
    }
    for (const auto& address : addresses_) {
        if (address.id != AddressReservationId{}) {
            const auto released = release_address_space(address);
            if (!released) {
                fatal(released.error());
            }
        }
    }
    const auto finalized = shutdown();
    if (!finalized) {
        fatal(finalized.error());
    }
}

[[noreturn]] void CudaVmmBackend::fatal(Error error) const noexcept {
    if (codec_audit_ != nullptr && !codec_audit_->has_fatal_error) {
        codec_audit_->fatal_error = error;
        codec_audit_->has_fatal_error = true;
    }
    error.code = ErrorCode::backend_contract_violation;
    errors_.push(error);
    std::terminate();
}
void CudaVmmBackend::reject_ambiguity(const Error& error) const noexcept {
    if (error.code == ErrorCode::ambiguous_backend_state ||
        error.code == ErrorCode::backend_contract_violation ||
        error.code == ErrorCode::invalid_argument || error.code == ErrorCode::invalid_range ||
        error.code == ErrorCode::invalid_alignment || error.code == ErrorCode::stale_handle) {
        // Driver calls receive locally validated arguments and owned identities. Rejecting
        // those as invalid/stale contradicts prior successful observations, not an ordinary
        // OOM or a certified pre-mutation operational failure. Never guess cleanup ownership.
        fatal(error);
    }
}
const BackendCapabilities& CudaVmmBackend::capabilities() const noexcept { return capabilities_; }
ByteSize CudaVmmBackend::recommended_granularity() const noexcept { return recommended_; }
std::int32_t CudaVmmBackend::driver_version() const noexcept { return driver_version_; }
std::optional<AsyncErrorRecord> CudaVmmBackend::first_error() const noexcept {
    return errors_.first();
}
CompressionStats CudaVmmBackend::compression_stats() const noexcept {
    const std::scoped_lock lock{stats_mutex_};
    return compression_stats_;
}
bool CudaVmmBackend::compression_available() const noexcept { return compression_ != nullptr; }

DeviceAddress CudaVmmBackend::mapped_address(const Resource& resource) noexcept {
    return resource.private_address == DeviceAddress{} ? resource.allocation.address
                                                       : resource.private_address;
}

bool CudaVmmBackend::contains(DeviceAddress address, ByteSize size) const noexcept {
    return std::ranges::any_of(addresses_, [address, size](const AddressReservation& reservation) {
        return reservation.id != AddressReservationId{} && address >= reservation.base &&
               address.value() - reservation.base.value() <= reservation.size.value() &&
               size.value() <=
                   reservation.size.value() - (address.value() - reservation.base.value());
    });
}

Result<AddressReservation> CudaVmmBackend::reserve_address_space(ByteSize size,
                                                                 ByteSize alignment) noexcept {
    const std::scoped_lock lock{mutex_};
    if (size.value() == 0U || capabilities_.reservation_granularity.value() == 0U ||
        size.value() % capabilities_.reservation_granularity.value() != 0U ||
        !std::has_single_bit(alignment.value()) ||
        alignment < capabilities_.reservation_granularity) {
        return make_error(ErrorCode::invalid_alignment, OperationId::allocate);
    }
    auto slot = std::ranges::find_if(
        addresses_, [](const auto& entry) { return entry.id == AddressReservationId{}; });
    if (slot == addresses_.end() || next_address_id_ == std::numeric_limits<std::uint64_t>::max()) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::allocate);
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    const auto reserved = driver_->reserve(CudaAddressRequest{size, alignment});
    if (!reserved) {
        reject_ambiguity(reserved.error());
        return reserved.error();
    }
    const auto end = checked_add(reserved.value().value(), size.value(), OperationId::allocate);
    if (reserved.value() == DeviceAddress{} || reserved.value().value() % alignment.value() != 0U ||
        !end) {
        fatal(contract());
    }
    for (const auto& existing : addresses_) {
        if (existing.id != AddressReservationId{} &&
            (reserved.value() >= existing.base
                 ? reserved.value().value() - existing.base.value() < existing.size.value()
                 : existing.base.value() - reserved.value().value() < size.value())) {
            fatal(contract());
        }
    }
    for (const auto& existing : private_addresses_) {
        if (existing.base != DeviceAddress{} &&
            (reserved.value() >= existing.base
                 ? reserved.value().value() - existing.base.value() < existing.size.value()
                 : existing.base.value() - reserved.value().value() < size.value())) {
            fatal(contract());
        }
    }
    *slot = AddressReservation{AddressReservationId{next_address_id_}, reserved.value(), size,
                               alignment};
    ++next_address_id_;
    return *slot;
}

Result<void> CudaVmmBackend::release_address_space(AddressReservation reservation) noexcept {
    const std::scoped_lock lock{mutex_};
    auto slot = std::ranges::find(addresses_, reservation);
    if (reservation.id == AddressReservationId{} || slot == addresses_.end()) {
        return make_error(ErrorCode::stale_handle, OperationId::release);
    }
    for (const auto& resource : resources_) {
        if (resource.state != MappingState::empty &&
            resource.allocation.address >= reservation.base &&
            resource.allocation.address.value() - reservation.base.value() <
                reservation.size.value()) {
            return make_error(ErrorCode::busy, OperationId::release);
        }
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    const auto freed = driver_->free_address(reservation.base, reservation.size);
    if (!freed) {
        reject_ambiguity(freed.error());
        return freed.error();
    }
    *slot = {};
    return {};
}
bool CudaVmmBackend::owns_address_space(AddressReservationId id) const noexcept {
    const std::scoped_lock lock{mutex_};
    return id != AddressReservationId{} &&
           std::ranges::any_of(addresses_, [id](const auto& entry) { return entry.id == id; });
}

Result<ByteSize> CudaVmmBackend::allocation_bound(RepresentationState state,
                                                  ByteSize logical_size) noexcept {
    if (state == RepresentationState::gpu_compressed && compression_) {
        const auto plan = compression_plan(GpuCodecDirection::compress, logical_size);
        return plan ? Result<ByteSize>{plan.value().output_charge} : Result<ByteSize>{plan.error()};
    }
    if (state != RepresentationState::gpu_raw ||
        (compression_ && logical_size > gpu_lz4_max_chunk)) {
        return unsupported(OperationId::allocate);
    }
    if (logical_size.value() == 0U) {
        return make_error(ErrorCode::invalid_argument, OperationId::allocate);
    }
    const auto padded = checked_align_up(
        logical_size.value(), capabilities_.allocation_granularity.value(), OperationId::allocate);
    if (!padded) {
        return padded.error();
    }
    if (padded.value() > std::numeric_limits<std::size_t>::max()) {
        return make_error(ErrorCode::arithmetic_overflow, OperationId::allocate);
    }
    return ByteSize{padded.value()};
}
Result<GpuCompressionPlan> CudaVmmBackend::compression_plan(GpuCodecDirection direction,
                                                            ByteSize logical_size) noexcept {
    if (!compression_ || logical_size > gpu_lz4_max_chunk) {
        return unsupported(OperationId::allocate);
    }
    if (planned_logical_ != ByteSize{}) {
        if (logical_size != planned_logical_) {
            return unsupported(OperationId::allocate);
        }
        return admitted_plans_[direction == GpuCodecDirection::compress ? 0U : 1U];
    }
    const auto requirements = compression_->requirements(direction, logical_size);
    if (!requirements) {
        return requirements.error();
    }
    const auto plan = make_gpu_compression_plan(
        requirements.value(),
        GpuPlanDimensions{logical_size, capabilities_.allocation_granularity});
    if (plan && context_ != CudaContext{} && codec_alignment_ != ByteSize{} &&
        plan.value().address_alignment > codec_alignment_) {
        return make_error(ErrorCode::invalid_alignment, OperationId::allocate);
    }
    return plan;
}
Result<ByteSize> CudaVmmBackend::workspace_bound(RepresentationState source,
                                                 RepresentationState destination,
                                                 ByteSize logical_size) noexcept {
    if (tier_of(source) != PhysicalTier::gpu || tier_of(destination) != PhysicalTier::gpu ||
        source == destination) {
        return unsupported(OperationId::allocate);
    }
    const auto plan = compression_plan(is_raw(destination) ? GpuCodecDirection::decompress
                                                           : GpuCodecDirection::compress,
                                       logical_size);
    return plan ? Result<ByteSize>{plan.value().workspace_charge} : Result<ByteSize>{plan.error()};
}
Result<ByteSize> CudaVmmBackend::compaction_bound(RepresentationState destination,
                                                  ByteSize logical_size) noexcept {
    return destination == RepresentationState::gpu_compressed && compression_
               ? allocation_bound(destination, logical_size)
               : Result<ByteSize>{ByteSize{}};
}
Result<BackendAllocation> CudaVmmBackend::allocate_workspace(PhysicalTier tier,
                                                             ByteSize charge) noexcept {
    if (!compression_ || tier != PhysicalTier::gpu || charge == ByteSize{} ||
        charge.value() % capabilities_.allocation_granularity.value() != 0U) {
        return unsupported(OperationId::allocate);
    }
    return allocate_storage(BackendAllocation{{}, tier, charge, ResourceKind::workspace, {}, {}},
                            {});
}

Result<ByteSize> CudaVmmBackend::execute_codec(Resource& source, Resource& destination,
                                               Resource& workspace,
                                               GpuCodecDirection direction) noexcept {
    const auto logical_size = source.allocation.metadata.logical_size;
    const auto plan = compression_plan(direction, logical_size);
    if (!plan) {
        return plan.error();
    }
    if (workspace.allocation.charge < plan.value().workspace_charge ||
        destination.allocation.charge < plan.value().output_charge ||
        mapped_address(source).value() % plan.value().requirements.input_alignment.value() != 0U ||
        mapped_address(destination).value() % plan.value().requirements.output_alignment.value() !=
            0U) {
        return contract(destination.allocation.id);
    }
    // This also gates committed compressed input with its stored-payload CRC before launch.
    const auto expected = read_all(source);
    if (!expected) {
        return expected.error();
    }
    if (crc32c(expected.value()) != source.allocation.metadata.crc32c) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify);
    }
    const GpuBatchMetadata metadata{mapped_address(source).value(),
                                    source.allocation.metadata.stored_size.value(),
                                    mapped_address(destination).value(),
                                    plan.value().requirements.output_bound.value(),
                                    0U,
                                    -1,
                                    0U};
    const auto temporary =
        checked_add(mapped_address(workspace).value(), plan.value().temporary_offset.value(),
                    OperationId::compress);
    if (!temporary) {
        return temporary.error();
    }
    const GpuCodecJob job{direction,
                          logical_size,
                          mapped_address(workspace),
                          DeviceAddress{temporary.value()},
                          plan.value().requirements.temporary_bytes,
                          metadata};
    // The codec owns metadata publication and its stream dependency. A generic pageable
    // Driver HtoD here would not establish ordering with the non-blocking codec stream.
    const auto launched = compression_->launch(job);
    if (!launched) {
        reject_ambiguity(launched.error());
        return launched.error();
    }
    if (codec_audit_ != nullptr) {
        (direction == GpuCodecDirection::compress ? codec_audit_->compression_enqueue
                                                  : codec_audit_->decompression_enqueue) = true;
    }
    const auto completed = compression_->synchronize();
    if (!completed) {
        // Successful submission with no proof of completion forbids cleanup of any operand.
        fatal(completed.error());
    }
    if (codec_audit_ != nullptr) {
        (direction == GpuCodecDirection::compress ? codec_audit_->compression_completion
                                                  : codec_audit_->decompression_completion) = true;
    }
    const auto result = compression_->result();
    if (!result) {
        reject_ambiguity(result.error());
        return result.error();
    }
    if (result.value().status != 0 || result.value().actual_size == ByteSize{} ||
        result.value().actual_size > plan.value().requirements.output_bound ||
        result.value().actual_size > destination.allocation.charge ||
        (direction == GpuCodecDirection::decompress &&
         result.value().actual_size != logical_size)) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify,
                          destination.allocation.id.value(), result.value().actual_size.value());
    }
    auto stored = byte_buffer(result.value().actual_size);
    if (!stored) {
        return stored.error();
    }
    const auto read = driver_->copy_from_device(stored.value(), mapped_address(destination));
    if (!read) {
        reject_ambiguity(read.error());
        return read.error();
    }
    RepresentationMetadata final_metadata{
        direction == GpuCodecDirection::compress ? Encoding::lz4_block : Encoding::raw,
        logical_size, result.value().actual_size, source.allocation.metadata.crc32c,
        crc32c(stored.value())};
    // Decode through the bounded reference oracle and compare every logical byte. Metadata
    // and ContentTag are published to this private resource only after successful verification.
    Resource candidate = destination;
    candidate.allocation.metadata = final_metadata;
    const auto observed = read_all(candidate);
    if (!observed) {
        return observed.error();
    }
    if (observed.value() != expected.value()) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify);
    }
    destination.allocation.metadata = final_metadata;
    destination.content = source.content;
    {
        const std::scoped_lock stats_lock{stats_mutex_};
        if (direction == GpuCodecDirection::compress) {
            saturating_increment(compression_stats_.successful_compressions);
            compression_stats_.bytes_compressed = ByteSize{
                saturating_add(compression_stats_.bytes_compressed.value(), logical_size.value())};
        } else {
            saturating_increment(compression_stats_.successful_decompressions);
            compression_stats_.bytes_decompressed = ByteSize{saturating_add(
                compression_stats_.bytes_decompressed.value(), logical_size.value())};
        }
    }
    const auto padded =
        checked_align_up(result.value().actual_size.value(),
                         capabilities_.allocation_granularity.value(), OperationId::compress);
    return padded ? Result<ByteSize>{ByteSize{padded.value()}} : Result<ByteSize>{padded.error()};
}

Result<ByteSize> CudaVmmBackend::prepare_transfer(ResourceId source_id, ResourceId destination_id,
                                                  ResourceId workspace_id) noexcept {
    if (source_id == destination_id || source_id == workspace_id ||
        destination_id == workspace_id) {
        return contract(destination_id);
    }
    const std::scoped_lock lock{mutex_};
    auto* source = find(source_id);
    auto* destination = find(destination_id);
    auto* workspace = find(workspace_id);
    if (!compression_ || source == nullptr || destination == nullptr || workspace == nullptr ||
        source == destination || source == workspace || destination == workspace ||
        !source->adopted || !destination->adopted || !workspace->adopted ||
        source->allocation.kind != ResourceKind::representation ||
        destination->allocation.kind != ResourceKind::representation ||
        workspace->allocation.kind != ResourceKind::workspace) {
        return unsupported(OperationId::migrate);
    }
    if (destination->allocation.metadata.encoding == Encoding::raw) {
        return ByteSize{};
    }
    if (source->allocation.metadata.encoding != Encoding::raw || destination->prepared) {
        return contract(destination_id);
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    const auto encoded =
        execute_codec(*source, *destination, *workspace, GpuCodecDirection::compress);
    if (!encoded) {
        if (encoded.error().code == ErrorCode::integrity_failure) {
            const std::scoped_lock stats_lock{stats_mutex_};
            saturating_increment(compression_stats_.integrity_failures);
        }
        return encoded.error();
    }
    destination->prepared = true;
    return encoded.value();
}

Result<TransferReceipt> CudaVmmBackend::transfer(ResourceId source_id, ResourceId destination_id,
                                                 ResourceId workspace_id,
                                                 CompactionTarget target) noexcept {
    const auto compaction_id = target.resource;
    if (source_id == destination_id || source_id == workspace_id ||
        destination_id == workspace_id) {
        return contract(destination_id);
    }
    const std::scoped_lock lock{mutex_};
    auto* source = find(source_id);
    auto* destination = find(destination_id);
    auto* workspace = find(workspace_id);
    if (!compression_ || source == nullptr || destination == nullptr || workspace == nullptr ||
        source == destination || source == workspace || destination == workspace ||
        !source->adopted || !destination->adopted || !workspace->adopted ||
        source->allocation.kind != ResourceKind::representation ||
        destination->allocation.kind != ResourceKind::representation ||
        workspace->allocation.kind != ResourceKind::workspace) {
        return unsupported(OperationId::migrate);
    }
    if (destination->allocation.metadata.encoding == Encoding::raw) {
        const ContextScope context{*this};
        if (!context.status()) {
            return context.status().error();
        }
        if (compaction_id != ResourceId{} ||
            source->allocation.metadata.encoding != Encoding::lz4_block) {
            return contract(destination_id);
        }
        const auto decoded =
            execute_codec(*source, *destination, *workspace, GpuCodecDirection::decompress);
        if (!decoded) {
            return decoded.error();
        }
        return TransferReceipt{destination->allocation.charge, destination->allocation.metadata};
    }
    auto* compaction = find(compaction_id);
    if (!destination->prepared || compaction == nullptr || !compaction->adopted ||
        compaction == source || compaction == destination || compaction == workspace ||
        compaction->allocation.kind != ResourceKind::workspace ||
        compaction->allocation.charge > destination->allocation.charge ||
        destination->allocation.metadata.stored_size > compaction->allocation.charge) {
        return contract(destination_id);
    }
    {
        const ContextScope context{*this};
        if (!context.status()) {
            return context.status().error();
        }
        const auto source_valid = verify_locked(*source, source->content);
        if (!source_valid) {
            return source_valid.error();
        }
        const auto proven = corroborate(*compaction);
        if (!proven) {
            reject_ambiguity(proven.error());
            return proven.error();
        }
        const auto copied = driver_->copy_device_to_device(
            mapped_address(*compaction), mapped_address(*destination),
            destination->allocation.metadata.stored_size);
        if (!copied) {
            reject_ambiguity(copied.error());
            return copied.error();
        }
        Resource candidate = *compaction;
        candidate.allocation.kind = ResourceKind::representation;
        candidate.allocation.metadata = destination->allocation.metadata;
        const auto observed = read_all(candidate);
        const auto expected = read_all(*source);
        if (!observed) {
            return observed.error();
        }
        if (!expected) {
            return expected.error();
        }
        if (observed.value() != expected.value()) {
            return make_error(ErrorCode::integrity_failure, OperationId::verify);
        }
    }
    // The sole charge mutation is a non-failing backing exchange. Neither identity changes.
    // Both physical allocations were adopted and charged before this point. No fallible call,
    // allocation, cleanup, observer or driver operation occurs between exchange and receipt.
    std::swap(destination->handle, compaction->handle);
    std::swap(destination->private_address, compaction->private_address);
    std::swap(destination->state, compaction->state);
    std::swap(destination->allocation.charge, compaction->allocation.charge);
    destination->prepared = false;
    return TransferReceipt{destination->allocation.charge, destination->allocation.metadata};
}

Result<void> CudaVmmBackend::verify_transfer(ResourceId source_id, ResourceId destination_id,
                                             ContentTag expected, ResourceId) noexcept {
    if (source_id == destination_id) {
        return contract(destination_id);
    }
    const std::scoped_lock lock{mutex_};
    const auto* source = find(source_id);
    const auto* destination = find(destination_id);
    if (source == nullptr || destination == nullptr || source->content != expected ||
        destination->content != expected) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify);
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    const auto left = read_all(*source);
    const auto right = read_all(*destination);
    if (!left) {
        return left.error();
    }
    if (!right) {
        return right.error();
    }
    if (left.value() != right.value() ||
        crc32c(right.value()) != destination->allocation.metadata.crc32c) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify);
    }
    return {};
}

CudaVmmBackend::Resource* CudaVmmBackend::find(ResourceId id) noexcept {
    return const_cast<Resource*>(std::as_const(*this).find(id));
}
const CudaVmmBackend::Resource* CudaVmmBackend::find(ResourceId id) const noexcept {
    const auto found = std::ranges::find_if(resources_, [id](const auto& resource) {
        return id != ResourceId{} && resource.state != MappingState::empty &&
               resource.allocation.id == id;
    });
    return found == resources_.end() ? nullptr : &*found;
}

Result<void> CudaVmmBackend::corroborate(const Resource& resource) const noexcept {
    if (resource.state != MappingState::mapped_rw) {
        return contract(resource.allocation.id);
    }
    const auto properties = driver_->properties(resource.handle);
    if (!properties) {
        reject_ambiguity(properties.error());
        return properties.error();
    }
    if (properties.value() != CudaAllocationProperties{device_, true, true, true, true}) {
        return contract(resource.allocation.id);
    }
    // cuMemRetainAllocationHandle guarantees the same handle used by cuMemMap. Inspect each
    // minimum-granularity page, not only the beginning of a potentially multi-page mapping.
    for (std::uint64_t offset = 0U; offset < resource.allocation.charge.value();
         offset += capabilities_.allocation_granularity.value()) {
        const DeviceAddress address{mapped_address(resource).value() + offset};
        const auto retained = driver_->retain_mapping(address);
        if (!retained) {
            reject_ambiguity(retained.error());
            return retained.error();
        }
        // A contradictory handle is not a safe cleanup target. Stop before another driver
        // call rather than releasing an unrelated or uncorroborated reference.
        if (retained.value() != resource.handle) {
            fatal(contract(resource.allocation.id));
        }
        const auto access = driver_->has_read_write_access(address, device_);
        const auto released = driver_->release(retained.value());
        if (!released) {
            fatal(released.error());
        }
        if (!access) {
            reject_ambiguity(access.error());
            return access.error();
        }
        if (!access.value()) {
            return contract(resource.allocation.id);
        }
    }
    return {};
}

Result<void> CudaVmmBackend::release_locked(Resource& resource) noexcept {
    if (resource.state == MappingState::mapped_rw ||
        resource.state == MappingState::mapped_no_access) {
        const auto unmapped = driver_->unmap(mapped_address(resource), resource.allocation.charge);
        if (!unmapped) {
            reject_ambiguity(unmapped.error());
            return unmapped.error();
        }
        resource.state = MappingState::physical_created;
    }
    const auto released = driver_->release(resource.handle);
    if (!released) {
        reject_ambiguity(released.error());
        return released.error();
    }
    const auto private_address = resource.private_address;
    resource = {};
    if (private_address != DeviceAddress{}) {
        // Physical ownership ended. A known VA-free failure is retained separately, never
        // represented as physical cleanup debt. shutdown retries the preallocated VA owner.
        const auto freed = free_private_address(private_address);
        if (!freed) {
            errors_.push(freed.error());
        }
    }
    return {};
}

Result<BackendAllocation>
CudaVmmBackend::allocate_representation(RepresentationAllocationRequest request) noexcept {
    const auto bound = allocation_bound(request.state, request.logical_size);
    if (!bound) {
        return bound.error();
    }
    const auto encoding = is_raw(request.state) ? Encoding::raw : Encoding::lz4_block;
    if (encoding == Encoding::lz4_block && request.stable_address != DeviceAddress{}) {
        return make_error(ErrorCode::invalid_argument, OperationId::allocate);
    }
    return allocate_storage(
        BackendAllocation{{},
                          PhysicalTier::gpu,
                          bound.value(),
                          ResourceKind::representation,
                          {encoding, request.logical_size,
                           encoding == Encoding::raw ? request.logical_size : ByteSize{}, 0U, 0U},
                          request.stable_address},
        request.content);
}

Result<void> CudaVmmBackend::free_private_address(DeviceAddress address) noexcept {
    auto slot = std::ranges::find_if(
        private_addresses_, [address](const auto& entry) { return entry.base == address; });
    if (slot == private_addresses_.end()) {
        fatal(contract());
    }
    const auto freed = driver_->free_address(slot->base, slot->size);
    if (!freed) {
        reject_ambiguity(freed.error());
        return freed.error();
    }
    *slot = {};
    return {};
}

Result<BackendAllocation>
CudaVmmBackend::allocate_initialized_raw(RepresentationAllocationRequest request,
                                         std::span<const std::byte> initial,
                                         std::span<std::byte> observed) noexcept {
    const std::less<const std::byte*> before{};
    if (compression_ || request.state != RepresentationState::gpu_raw || initial.empty() ||
        initial.size() > std::size_t{64U} * 1024U || initial.size() != observed.size() ||
        request.logical_size.value() != initial.size() ||
        (before(initial.data(), observed.data() + observed.size()) &&
         before(observed.data(), initial.data() + initial.size()))) {
        return make_error(ErrorCode::invalid_argument, OperationId::allocate);
    }
    const auto bound = allocation_bound(request.state, request.logical_size);
    if (!bound) {
        return bound.error();
    }
    return allocate_storage(
        BackendAllocation{{},
                          PhysicalTier::gpu,
                          bound.value(),
                          ResourceKind::representation,
                          {Encoding::raw, request.logical_size, request.logical_size, 0U, 0U},
                          request.stable_address},
        request.content, initial, observed);
}

Result<BackendAllocation> CudaVmmBackend::allocate_storage(BackendAllocation descriptor,
                                                           ContentTag content,
                                                           std::span<const std::byte> provided,
                                                           std::span<std::byte> readback) noexcept {
    const bool raw = descriptor.kind == ResourceKind::representation &&
                     descriptor.metadata.encoding == Encoding::raw;
    // All reference buffers precede the first physical mutation.
    auto initial =
        byte_buffer(raw && provided.empty() ? descriptor.metadata.logical_size : ByteSize{});
    auto observed =
        byte_buffer(raw && provided.empty() ? descriptor.metadata.logical_size : ByteSize{});
    if (!initial) {
        return initial.error();
    }
    if (!observed) {
        return observed.error();
    }
    const auto initial_bytes =
        provided.empty() ? std::span<const std::byte>{initial.value()} : provided;
    const auto observed_bytes =
        provided.empty() ? std::span<std::byte>{observed.value()} : readback;
    const std::scoped_lock lock{mutex_};
    if (raw && (descriptor.address == DeviceAddress{} ||
                !contains(descriptor.address, descriptor.charge) ||
                descriptor.address.value() % capabilities_.mapping_granularity.value() != 0U)) {
        return make_error(ErrorCode::invalid_alignment, OperationId::allocate);
    }
    std::uint64_t total = descriptor.charge.value();
    for (const auto& resource : resources_) {
        if (resource.state == MappingState::empty) {
            continue;
        }
        const auto sum =
            checked_add(total, resource.allocation.charge.value(), OperationId::allocate);
        if (!sum) {
            return sum.error();
        }
        total = sum.value();
        const auto base = mapped_address(resource);
        if (raw &&
            (descriptor.address >= base
                 ? descriptor.address.value() - base.value() < resource.allocation.charge.value()
                 : base.value() - descriptor.address.value() < descriptor.charge.value())) {
            return make_error(ErrorCode::conflict, OperationId::allocate);
        }
    }
    if (physical_cap_ != ByteSize{} && total > physical_cap_.value()) {
        return unsupported(OperationId::allocate);
    }
    auto slot = std::ranges::find_if(
        resources_, [](const auto& resource) { return resource.state == MappingState::empty; });
    if (slot == resources_.end() ||
        next_resource_id_ == std::numeric_limits<std::uint64_t>::max()) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::allocate);
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    DeviceAddress address = descriptor.address;
    if (!raw) {
        const auto padded_va =
            checked_align_up(descriptor.charge.value(),
                             capabilities_.reservation_granularity.value(), OperationId::allocate);
        if (!padded_va) {
            return padded_va.error();
        }
        const ByteSize private_va_size{padded_va.value()};
        auto private_slot = std::ranges::find_if(
            private_addresses_, [](const auto& entry) { return entry.base == DeviceAddress{}; });
        if (private_slot == private_addresses_.end()) {
            return make_error(ErrorCode::out_of_host_memory, OperationId::allocate);
        }
        const auto reserved = driver_->reserve(CudaAddressRequest{
            private_va_size, std::max(capabilities_.reservation_granularity, codec_alignment_)});
        if (!reserved) {
            reject_ambiguity(reserved.error());
            return reserved.error();
        }
        address = reserved.value();
        if (address == DeviceAddress{} ||
            address.value() %
                    std::max(capabilities_.reservation_granularity, codec_alignment_).value() !=
                0U ||
            !checked_add(address.value(), private_va_size.value(), OperationId::allocate) ||
            std::ranges::any_of(
                addresses_,
                [address, private_va_size](const auto& entry) {
                    return entry.id != AddressReservationId{} &&
                           (address >= entry.base
                                ? address.value() - entry.base.value() < entry.size.value()
                                : entry.base.value() - address.value() < private_va_size.value());
                }) ||
            std::ranges::any_of(private_addresses_, [address, private_va_size](const auto& entry) {
                return entry.base != DeviceAddress{} &&
                       (address >= entry.base
                            ? address.value() - entry.base.value() < entry.size.value()
                            : entry.base.value() - address.value() < private_va_size.value());
            })) {
            fatal(contract());
        }
        *private_slot = {address, private_va_size};
    }
    const auto created = driver_->create(descriptor.charge, device_);
    if (!created) {
        reject_ambiguity(created.error());
        if (!raw) {
            const auto freed = free_private_address(address);
            if (!freed) {
                errors_.push(freed.error());
            }
        }
        return created.error();
    }
    if (created.value() == CudaPhysicalHandle{} ||
        std::ranges::any_of(resources_, [&](const auto& resource) {
            return resource.state != MappingState::empty && resource.handle == created.value();
        })) {
        fatal(contract());
    }
    descriptor.id = ResourceId{next_resource_id_};
    if (raw) {
        descriptor.metadata.crc32c = crc32c(initial_bytes);
        descriptor.metadata.stored_crc32c = descriptor.metadata.crc32c;
    }
    *slot = Resource{descriptor,
                     content,
                     created.value(),
                     MappingState::physical_created,
                     false,
                     raw ? DeviceAddress{} : address,
                     false};
    ++next_resource_id_;
    const auto cleanup_failure = [&](Error error) -> Result<BackendAllocation> {
        reject_ambiguity(error);
        const auto cleaned = release_locked(*slot);
        if (!cleaned) {
            fatal(cleaned.error());
        }
        return error;
    };
    const auto mapped = driver_->map(address, descriptor.charge, created.value());
    if (!mapped) {
        return cleanup_failure(mapped.error());
    }
    slot->state = MappingState::mapped_no_access;
    const auto access = driver_->set_access(address, descriptor.charge, device_);
    if (!access) {
        return cleanup_failure(access.error());
    }
    slot->state = MappingState::mapped_rw;
    const auto proven = corroborate(*slot);
    if (!proven) {
        fatal(proven.error());
    }
    if (!raw) {
        return slot->allocation;
    }
    const auto written = driver_->copy_to_device(address, initial_bytes);
    if (!written) {
        return cleanup_failure(written.error());
    }
    const auto read = driver_->copy_from_device(observed_bytes, address);
    if (!read) {
        return cleanup_failure(read.error());
    }
    if (!std::ranges::equal(observed_bytes, initial_bytes) ||
        crc32c(observed_bytes) != descriptor.metadata.crc32c) {
        return cleanup_failure(make_error(ErrorCode::integrity_failure, OperationId::verify));
    }
    slot->private_initialization_verified = !provided.empty();
    return slot->allocation;
}

Result<BackendAllocation>
CudaVmmBackend::adopt_allocation(const BackendAllocation& provisional) noexcept {
    const std::scoped_lock lock{mutex_};
    auto* resource = find(provisional.id);
    if (resource == nullptr || resource->adopted ||
        !allocation_equal(resource->allocation, provisional)) {
        return contract(provisional.id);
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    const auto proven = corroborate(*resource);
    if (!proven) {
        return proven.error();
    }
    if (resource->allocation.kind == ResourceKind::representation &&
        resource->allocation.metadata.encoding == Encoding::raw &&
        !resource->private_initialization_verified) {
        const auto valid = verify_locked(*resource, resource->content);
        if (!valid) {
            return valid.error();
        }
    }
    resource->private_initialization_verified = false;
    resource->adopted = true;
    return resource->allocation;
}
bool CudaVmmBackend::is_unadopted(ResourceId id) const noexcept {
    const std::scoped_lock lock{mutex_};
    const auto* resource = find(id);
    return resource != nullptr && !resource->adopted;
}

Result<BackendAllocation> CudaVmmBackend::allocation(ResourceId id) const noexcept {
    const std::scoped_lock lock{mutex_};
    const auto* resource = find(id);
    if (resource == nullptr) {
        return make_error(ErrorCode::stale_handle, OperationId::verify);
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    const auto proven = corroborate(*resource);
    if (!proven) {
        return proven.error();
    }
    return resource->allocation;
}

Result<std::vector<std::byte>> CudaVmmBackend::read_all(const Resource& resource) const noexcept {
    const auto& metadata = resource.allocation.metadata;
    if (resource.allocation.kind != ResourceKind::representation ||
        metadata.stored_size == ByteSize{} || metadata.stored_size > resource.allocation.charge ||
        (compression_ && metadata.logical_size > gpu_lz4_max_chunk)) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify);
    }
    auto bytes = byte_buffer(metadata.stored_size);
    if (!bytes) {
        return bytes.error();
    }
    const auto read = driver_->copy_from_device(bytes.value(), mapped_address(resource));
    if (!read) {
        return read.error();
    }
    // CRC covers only the stored representation, never VMM padding. This gate precedes
    // every decompressor call, including the reference verifier and GPU restore.
    const auto stored_crc = crc32c(bytes.value());
    if (codec_audit_ != nullptr && metadata.encoding == Encoding::lz4_block) {
        codec_audit_->stored_crc_expected = metadata.stored_crc32c;
        codec_audit_->stored_crc_actual = stored_crc;
    }
    if (stored_crc != metadata.stored_crc32c) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify);
    }
    if (metadata.encoding == Encoding::raw) {
        if (metadata.stored_size != metadata.logical_size) {
            return make_error(ErrorCode::integrity_failure, OperationId::verify);
        }
        return bytes;
    }
    if (metadata.encoding != Encoding::lz4_block || !compression_) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify);
    }
    auto logical = byte_buffer(metadata.logical_size);
    if (!logical) {
        return logical.error();
    }
    const auto decoded =
        cpu_compression_codec().decompress(bytes.value(), logical.value(), metadata.logical_size);
    if (!decoded || decoded.value() != metadata.logical_size) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify);
    }
    return logical;
}
Result<void> CudaVmmBackend::verify_locked(const Resource& resource,
                                           ContentTag expected) const noexcept {
    if (resource.content != expected) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify);
    }
    const auto bytes = read_all(resource);
    if (!bytes) {
        return bytes.error();
    }
    if (crc32c(bytes.value()) != resource.allocation.metadata.crc32c) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify,
                          resource.allocation.id.value());
    }
    return {};
}
Result<void> CudaVmmBackend::verify(ResourceId id, ContentTag expected,
                                    ResourceId workspace) noexcept {
    if (workspace != ResourceId{} && !compression_) {
        return unsupported(OperationId::verify);
    }
    const std::scoped_lock lock{mutex_};
    const auto* resource = find(id);
    if (resource == nullptr) {
        return make_error(ErrorCode::stale_handle, OperationId::verify);
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    const auto proven = corroborate(*resource);
    if (!proven) {
        return proven.error();
    }
    return verify_locked(*resource, expected);
}
Result<void> CudaVmmBackend::verify_authoritative(ResourceId id, ContentTag expected,
                                                  ResourceId workspace) noexcept {
    return verify(id, expected, workspace);
}

Result<void> CudaVmmBackend::read_bytes(ResourceId id, ByteOffset offset,
                                        std::span<std::byte> output) const noexcept {
    const std::scoped_lock lock{mutex_};
    const auto* resource = find(id);
    if (resource == nullptr || resource->state != MappingState::mapped_rw ||
        resource->allocation.kind != ResourceKind::representation ||
        resource->allocation.metadata.encoding != Encoding::raw) {
        return make_error(ErrorCode::stale_handle, OperationId::acquire);
    }
    const auto end = checked_add(offset.value(), output.size(), OperationId::acquire);
    if (!end || end.value() > resource->allocation.metadata.logical_size.value()) {
        return make_error(ErrorCode::invalid_range, OperationId::acquire);
    }
    if (output.empty()) {
        return {};
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    return driver_->copy_from_device(
        output, DeviceAddress{resource->allocation.address.value() + offset.value()});
}

Result<ContentTag> CudaVmmBackend::write_bytes(ResourceId id, ByteOffset offset,
                                               std::span<const std::byte> input,
                                               ContentTag previous) noexcept {
    const std::scoped_lock lock{mutex_};
    auto* resource = find(id);
    if (resource != nullptr && resource->private_initialization_verified && !resource->adopted) {
        return make_error(ErrorCode::busy, OperationId::close_lease);
    }
    if (resource == nullptr || resource->state != MappingState::mapped_rw ||
        resource->allocation.kind != ResourceKind::representation ||
        resource->allocation.metadata.encoding != Encoding::raw || resource->content != previous) {
        return make_error(ErrorCode::stale_handle, OperationId::close_lease);
    }
    const auto end = checked_add(offset.value(), input.size(), OperationId::close_lease);
    if (!end || end.value() > resource->allocation.metadata.logical_size.value()) {
        return make_error(ErrorCode::invalid_range, OperationId::close_lease);
    }
    if (previous.generation == std::numeric_limits<std::uint64_t>::max()) {
        return make_error(ErrorCode::arithmetic_overflow, OperationId::close_lease);
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    auto bytes = read_all(*resource);
    auto observed = byte_buffer(resource->allocation.metadata.logical_size);
    if (!bytes) {
        return bytes.error();
    }
    if (!observed) {
        return observed.error();
    }
    if (crc32c(bytes.value()) != resource->allocation.metadata.crc32c) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify);
    }
    auto destination =
        std::span{bytes.value()}.subspan(static_cast<std::size_t>(offset.value()), input.size());
    std::ranges::copy(input, destination.begin());
    const auto written = driver_->copy_to_device(resource->allocation.address, bytes.value());
    if (!written) {
        return written.error();
    }
    const auto read = driver_->copy_from_device(observed.value(), resource->allocation.address);
    if (!read) {
        return read.error();
    }
    if (observed.value() != bytes.value()) {
        return make_error(ErrorCode::integrity_failure, OperationId::verify);
    }
    resource->allocation.metadata.crc32c = crc32c(observed.value());
    resource->allocation.metadata.stored_crc32c = resource->allocation.metadata.crc32c;
    resource->content = next_content_tag(previous);
    return resource->content;
}

Result<void> CudaVmmBackend::write_content(ResourceId id, ContentTag content) noexcept {
    const std::scoped_lock lock{mutex_};
    auto* resource = find(id);
    if (resource != nullptr && resource->private_initialization_verified && !resource->adopted) {
        return make_error(ErrorCode::busy, OperationId::close_lease);
    }
    if (resource == nullptr || resource->state != MappingState::mapped_rw ||
        resource->allocation.kind != ResourceKind::representation ||
        resource->allocation.metadata.encoding != Encoding::raw) {
        return make_error(ErrorCode::stale_handle, OperationId::close_lease);
    }
    if (resource->content.generation == std::numeric_limits<std::uint64_t>::max() ||
        content != next_content_tag(resource->content)) {
        return contract(id);
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    // A writable lease authorizes new raw bytes; its previous CRC is intentionally stale.
    // This exclusive close operation establishes both CRCs for the new generation.
    auto bytes = byte_buffer(resource->allocation.metadata.logical_size);
    if (!bytes) {
        return bytes.error();
    }
    const auto read = driver_->copy_from_device(bytes.value(), mapped_address(*resource));
    if (!read) {
        return read.error();
    }
    resource->allocation.metadata.crc32c = crc32c(bytes.value());
    resource->allocation.metadata.stored_crc32c = resource->allocation.metadata.crc32c;
    resource->content = content;
    return {};
}

Result<void> CudaVmmBackend::release(ResourceId id, ReleasePhase) noexcept {
    const std::scoped_lock lock{mutex_};
    auto* resource = find(id);
    if (resource == nullptr) {
        return make_error(ErrorCode::stale_handle, OperationId::release);
    }
    const ContextScope context{*this};
    if (!context.status()) {
        return context.status().error();
    }
    return release_locked(*resource);
}
bool CudaVmmBackend::owns(ResourceId id) const noexcept {
    const std::scoped_lock lock{mutex_};
    return find(id) != nullptr;
}
ByteSize CudaVmmBackend::owned_charge(PhysicalTier tier) const noexcept {
    const std::scoped_lock lock{mutex_};
    if (tier != PhysicalTier::gpu) {
        return {};
    }
    std::uint64_t total{};
    for (const auto& resource : resources_) {
        if (resource.state != MappingState::empty) {
            total += resource.allocation.charge.value();
        }
    }
    return ByteSize{total};
}
Result<void> CudaVmmBackend::enable_performance() noexcept {
    const std::scoped_lock lock{mutex_};
    if (!compression_ || compression_->performance().enabled) {
        return {};
    }
    auto timed = make_timed_codec(compression_);
    if (!timed) {
        return timed.error();
    }
    compression_ = std::move(timed).value();
    return {};
}

PerformanceStats CudaVmmBackend::performance() const noexcept {
    const std::scoped_lock lock{mutex_};
    return compression_ ? compression_->performance() : PerformanceStats{};
}

CudaResourceCounts CudaVmmBackend::resource_counts() const noexcept {
    const std::scoped_lock lock{mutex_};
    CudaResourceCounts counts{};
    for (const auto& resource : resources_) {
        if (resource.allocation.id == ResourceId{}) {
            continue;
        }
        if (resource.allocation.kind == ResourceKind::workspace) {
            ++counts.workspace;
        } else if (resource.allocation.metadata.encoding == Encoding::raw) {
            ++counts.raw;
        } else {
            ++counts.compressed;
        }
    }
    counts.retained_context = context_ != CudaContext{};
    counts.stream_owned = stream_initialized_;
    return counts;
}

std::uint64_t CudaVmmBackend::owned_resource_count() const noexcept {
    const std::scoped_lock lock{mutex_};
    return static_cast<std::uint64_t>(std::ranges::count_if(
        resources_, [](const auto& resource) { return resource.state != MappingState::empty; }));
}
std::uint64_t CudaVmmBackend::owned_address_count() const noexcept {
    const std::scoped_lock lock{mutex_};
    return static_cast<std::uint64_t>(std::ranges::count_if(
               addresses_, [](const auto& entry) { return entry.id != AddressReservationId{}; })) +
           static_cast<std::uint64_t>(
               std::ranges::count_if(private_addresses_, [](const auto& entry) {
                   return entry.base != DeviceAddress{};
               }));
}

Result<void> CudaVmmBackend::shutdown() noexcept {
    const std::scoped_lock lock{mutex_};
    if (std::ranges::any_of(completions_, [](auto id) { return id != CompletionTokenId{}; })) {
        return make_error(ErrorCode::busy, OperationId::shutdown);
    }
    if (std::ranges::any_of(
            resources_,
            [](const auto& resource) { return resource.state != MappingState::empty; }) ||
        std::ranges::any_of(
            addresses_, [](const auto& address) { return address.id != AddressReservationId{}; })) {
        return make_error(ErrorCode::busy, OperationId::shutdown);
    }
    if (context_ == CudaContext{}) {
        return {};
    }
    {
        const ContextScope scope{*this};
        if (!scope.status()) {
            return scope.status().error();
        }
        for (const auto& address : private_addresses_) {
            if (address.base != DeviceAddress{}) {
                const auto freed = free_private_address(address.base);
                if (!freed) {
                    return freed.error();
                }
            }
        }
        if (compression_) {
            const auto closed = compression_->shutdown();
            if (!closed) {
                reject_ambiguity(closed.error());
                return closed.error();
            }
            stream_initialized_ = false;
            if (codec_audit_ != nullptr) {
                codec_audit_->stream_owned = false;
            }
        }
    }
    const auto released = driver_->release_primary(device_);
    if (!released) {
        reject_ambiguity(released.error());
        return released.error();
    }
    context_ = {};
    if (codec_audit_ != nullptr) {
        codec_audit_->context_owned = false;
    }
    return {};
}

Result<CompletionTokenId> CudaVmmBackend::create_completion() noexcept {
    const std::scoped_lock lock{mutex_};
    if (!capabilities_.supports_external_async_completion) {
        return unsupported(OperationId::defer_lease);
    }
    const auto slot = std::ranges::find(completions_, CompletionTokenId{});
    if (slot == completions_.end()) {
        return make_error(ErrorCode::busy, OperationId::defer_lease);
    }
    const ContextScope scope{*this};
    if (!scope.status()) {
        return scope.status().error();
    }
    const auto created = driver_->create_completion();
    if (!created) {
        reject_ambiguity(created.error());
        return created.error();
    }
    if (created.value() <= last_completion_id_) {
        fatal(make_error(ErrorCode::backend_contract_violation, OperationId::defer_lease));
    }
    *slot = created.value();
    last_completion_id_ = created.value();
    return created;
}

Result<CompletionState> CudaVmmBackend::query_completion(CompletionTokenId id) noexcept {
    const std::scoped_lock lock{mutex_};
    if (id == CompletionTokenId{} || std::ranges::find(completions_, id) == completions_.end()) {
        return make_error(ErrorCode::stale_handle, OperationId::query_completion, id.value());
    }
    const ContextScope scope{*this};
    if (!scope.status()) {
        return scope.status().error();
    }
    const auto queried = driver_->query_completion(id);
    if (!queried) {
        reject_ambiguity(queried.error());
        return queried.error();
    }
    if (queried.value() != CompletionState::pending &&
        queried.value() != CompletionState::complete) {
        fatal(make_error(ErrorCode::ambiguous_backend_state, OperationId::query_completion,
                         id.value()));
    }
    return queried;
}

Result<void> CudaVmmBackend::release_completion(CompletionTokenId id) noexcept {
    const std::scoped_lock lock{mutex_};
    const auto slot = std::ranges::find(completions_, id);
    if (id == CompletionTokenId{} || slot == completions_.end()) {
        return make_error(ErrorCode::stale_handle, OperationId::release_completion, id.value());
    }
    const ContextScope scope{*this};
    if (!scope.status()) {
        return scope.status().error();
    }
    const auto released = driver_->release_completion(id);
    if (!released) {
        reject_ambiguity(released.error());
        return released.error();
    }
    *slot = {};
    return {};
}

} // namespace vramz::detail
