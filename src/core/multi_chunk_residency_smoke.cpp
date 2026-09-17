#include "vramz/detail/multi_chunk_residency_smoke.hpp"

#include "vramz/budget_ledger.hpp"
#include "vramz/checked.hpp"
#include "vramz/crc32c.hpp"
#include "vramz/detail/allocation.hpp"

#include <algorithm>
#include <exception>
#include <new>
#include <utility>
#include <vector>

namespace vramz::detail {

Result<MultiChunkResidencyTotals>
calculate_multi_chunk_residency_totals(std::span<const MultiChunkResidencyChunkReport> chunks,
                                       ByteSize minimum_granularity) noexcept {
    if (chunks.size() != m10_chunk_count || minimum_granularity == ByteSize{}) {
        return make_error(ErrorCode::invalid_argument, OperationId::verify);
    }
    MultiChunkResidencyTotals total{};
    for (const auto& chunk : chunks) {
        if (!chunk.charges_corroborated) {
            return make_error(ErrorCode::invalid_argument, OperationId::verify);
        }
        const auto logical = checked_add(total.logical_bytes.value(), chunk.logical_bytes.value(),
                                         OperationId::verify);
        const auto raw = checked_add(total.raw_baseline_charge.value(), chunk.raw_charge.value(),
                                     OperationId::verify);
        const auto compressed = checked_add(total.compressed_physical_charge.value(),
                                            chunk.compressed_charge.value(), OperationId::verify);
        if (!logical || !raw || !compressed) {
            return !logical ? logical.error() : (!raw ? raw.error() : compressed.error());
        }
        total.logical_bytes = ByteSize{logical.value()};
        total.raw_baseline_charge = ByteSize{raw.value()};
        total.compressed_physical_charge = ByteSize{compressed.value()};
    }
    const auto saved = checked_sub(total.raw_baseline_charge.value(),
                                   total.compressed_physical_charge.value(), OperationId::verify);
    const auto required =
        checked_mul(m10_chunk_count, minimum_granularity.value(), OperationId::verify);
    if (!saved || !required) {
        return !saved ? saved.error() : required.error();
    }
    if (saved.value() < required.value() ||
        !std::ranges::all_of(chunks, [minimum_granularity](const auto& chunk) {
            return saves_physical_unit(chunk.raw_charge, chunk.compressed_charge,
                                       minimum_granularity);
        })) {
        return make_error(ErrorCode::unsupported, OperationId::verify);
    }
    total.physical_bytes_saved = ByteSize{saved.value()};
    return total;
}

Result<ByteSize> admit_multi_chunk_residency_stage(ByteSize retained_charge,
                                                   ByteSize operation_peak) noexcept {
    const auto peak =
        checked_add(retained_charge.value(), operation_peak.value(), OperationId::allocate);
    if (!peak) {
        return peak.error();
    }
    if (peak.value() > m10_physical_cap.value()) {
        return make_error(ErrorCode::unsupported, OperationId::allocate);
    }
    return ByteSize{peak.value()};
}

namespace {
enum class ResourceSlot : std::uint8_t {
    raw,
    compressed,
    encode_workspace,
    compaction,
    restored,
    decode_workspace
};
constexpr auto raw = ResourceSlot::raw;
constexpr auto compressed = ResourceSlot::compressed;
constexpr auto encode_workspace = ResourceSlot::encode_workspace;
constexpr auto compaction = ResourceSlot::compaction;
constexpr auto restored = ResourceSlot::restored;
constexpr auto decode_workspace = ResourceSlot::decode_workspace;
[[nodiscard]] constexpr std::size_t slot_index(ResourceSlot slot) noexcept {
    return static_cast<std::size_t>(slot);
}
constexpr std::array buckets{ChargeBucket::staging,   ChargeBucket::staging,
                             ChargeBucket::workspace, ChargeBucket::workspace,
                             ChargeBucket::staging,   ChargeBucket::workspace};
enum class Ownership : std::uint8_t { staging, committed, cleanup_debt };
enum class Authority : std::uint8_t { none, gpu_raw, gpu_compressed, restored_raw };

void failure(MultiChunkResidencySmokeReport& report, Error error) noexcept {
    report.has_error = true;
    report.error = error;
    report.result = error.code == ErrorCode::unsupported ? CompressionSmokeResult::unsupported
                                                         : CompressionSmokeResult::failed;
}

struct ChunkOwner final {
    std::array<BackendAllocation, 6U> allocations{};
    std::array<ReservationToken, 6U> tokens{};
    std::array<bool, 6U> active{};
    std::array<Ownership, 6U> ownership{};
    std::array<AddressReservation, 2U> addresses{};
    ContentTag content{};
    Authority authority{Authority::none};
};

// Four independent private owners exercise the production backend and exact ledger
// transitions. No Runtime policy, pressure trigger, packing or public Buffer is involved.
class ResidencySession final {
  public:
    ResidencySession(std::unique_ptr<CudaVmmBackend> backend,
                     MultiChunkResidencySmokeReport& report) noexcept
        : backend_(std::move(backend)), report_(report),
          ledger_({{m10_physical_cap, m10_physical_cap, {}}, {}}) {}
    ResidencySession(const ResidencySession&) = delete;
    ResidencySession& operator=(const ResidencySession&) = delete;
    ~ResidencySession() noexcept { finish(); }

    void run() {
        std::vector<std::byte> initial(static_cast<std::size_t>(m10_logical_payload.value()));
        std::vector<std::byte> observed(initial.size());
        for (std::size_t index = 0U; index < owners_.size(); ++index) {
            make_payload(index, initial);
            if (!compress_chunk(index, initial, observed)) {
                return;
            }
        }
        if (!residency_snapshot()) {
            return;
        }
        // This loop is reachable only after all four independently owned compressed
        // allocations have passed the aggregate snapshot. Exactly one restore is live.
        for (std::size_t index = 0U; index < owners_.size(); ++index) {
            make_payload(index, initial);
            if (!restore_chunk(index, initial, observed)) {
                return;
            }
        }
        const auto stats = backend_->compression_stats();
        if (stats.successful_compressions != m10_chunk_count ||
            stats.successful_decompressions != m10_chunk_count ||
            !std::ranges::all_of(report_.chunks, [](const auto& chunk) {
                return chunk.round_trip_verified && chunk.minimum_unit_saved &&
                       chunk.compressed_at_snapshot;
            })) {
            contract();
        }
        verified_ = true;
    }

  private:
    static void make_payload(std::size_t chunk, std::span<std::byte> bytes) noexcept {
        for (std::size_t index = 0U; index < bytes.size(); ++index) {
            bytes[index] = multi_chunk_residency_payload_byte(chunk, index);
        }
    }
    [[noreturn]] void contract() noexcept {
        failure(report_, make_error(ErrorCode::backend_contract_violation, OperationId::migrate));
        report_.final_counts_known = false;
        std::terminate();
    }
    void require(Result<void> result) noexcept {
        if (!result) {
            stop(result.error());
        }
    }
    bool accept(Result<void> result) noexcept {
        if (!result) {
            failure(report_, result.error());
            return false;
        }
        return true;
    }
    void corroborate_accounting() noexcept {
        const auto owned = backend_->owned_charge(PhysicalTier::gpu).value();
        const auto charged = ledger_.charged(PhysicalTier::gpu).value();
        const auto reserved = ledger_.unmaterialized_reservations(PhysicalTier::gpu).value();
        if (charged < reserved || charged - reserved != owned ||
            charged > m10_physical_cap.value()) {
            contract();
        }
    }
    [[nodiscard]] std::uint32_t compressed_mask() const noexcept {
        std::uint32_t mask{};
        for (std::size_t index = 0U; index < owners_.size(); ++index) {
            if (owners_[index].authority == Authority::gpu_compressed) {
                mask |= std::uint32_t{1U} << index;
            }
        }
        return mask;
    }
    void event(MultiChunkResidencyEventKind kind, std::size_t index) noexcept {
        if (report_.event_count == report_.events.size()) {
            contract();
        }
        report_.events[report_.event_count++] = {kind, static_cast<std::uint32_t>(index),
                                                 compressed_mask()};
    }
    void capture_codec(MultiChunkResidencyChunkReport& chunk) noexcept {
        chunk.compression_enqueue = report_.codec.compression_enqueue;
        chunk.compression_completion = report_.codec.compression_completion;
        chunk.decompression_enqueue = report_.codec.decompression_enqueue;
        chunk.decompression_completion = report_.codec.decompression_completion;
        chunk.stored_crc_expected = report_.codec.stored_crc_expected;
        chunk.stored_crc_actual = report_.codec.stored_crc_actual;
    }
    void capture_stored_crc(std::size_t chunk) noexcept {
        report_.chunks[chunk].stored_crc_expected = report_.codec.stored_crc_expected;
        report_.chunks[chunk].stored_crc_actual = report_.codec.stored_crc_actual;
    }
    void begin_compression_audit() noexcept {
        report_.codec.compression_enqueue = false;
        report_.codec.compression_completion = false;
        report_.codec.decompression_enqueue = false;
        report_.codec.decompression_completion = false;
        report_.codec.stored_crc_expected = 0U;
        report_.codec.stored_crc_actual = 0U;
    }
    void begin_restoration_audit(const MultiChunkResidencyChunkReport& chunk) noexcept {
        report_.codec.compression_enqueue = chunk.compression_enqueue;
        report_.codec.compression_completion = chunk.compression_completion;
        report_.codec.decompression_enqueue = false;
        report_.codec.decompression_completion = false;
        report_.codec.stored_crc_expected = chunk.stored_crc_expected;
        report_.codec.stored_crc_actual = chunk.stored_crc_actual;
    }
    bool corroborate_retained() noexcept {
        for (std::size_t index = 0U; index < owners_.size(); ++index) {
            const auto& owner = owners_[index];
            if (owner.authority != Authority::gpu_compressed) {
                continue;
            }
            corroborate(index, compressed);
            const auto& allocation = owner.allocations[slot_index(compressed)];
            if (owner.ownership[slot_index(compressed)] != Ownership::committed ||
                allocation.kind != ResourceKind::representation ||
                allocation.metadata.encoding != Encoding::lz4_block ||
                allocation.metadata.logical_size != m10_logical_payload ||
                allocation.charge != report_.chunks[index].compressed_charge ||
                allocation.metadata.stored_size != report_.chunks[index].stored_bytes) {
                contract();
            }
            // A context or corroboration error may precede the CRC read. Preserve this
            // chunk's last observation rather than borrowing another chunk's audit values.
            report_.codec.stored_crc_expected = report_.chunks[index].stored_crc_expected;
            report_.codec.stored_crc_actual = report_.chunks[index].stored_crc_actual;
            const auto verified = backend_->verify_authoritative(allocation.id, owner.content);
            capture_stored_crc(index);
            if (!accept(verified)) {
                return false;
            }
        }
        corroborate_accounting();
        return true;
    }
    bool admit_stage(ByteSize peak) noexcept {
        if (!corroborate_retained()) {
            return false;
        }
        const auto admitted =
            admit_multi_chunk_residency_stage(backend_->owned_charge(PhysicalTier::gpu), peak);
        if (!admitted) {
            failure(report_, admitted.error());
            report_.result = CompressionSmokeResult::resource_cap;
            return false;
        }
        stage_peak_ = admitted.value();
        report_.peak_admitted_gpu_bytes = std::max(report_.peak_admitted_gpu_bytes, stage_peak_);
        return true;
    }
    bool admit_allocation(ByteSize charge) noexcept {
        corroborate_accounting();
        const auto admitted =
            admit_multi_chunk_residency_stage(backend_->owned_charge(PhysicalTier::gpu), charge);
        if (!admitted) {
            failure(report_, admitted.error());
            report_.result = CompressionSmokeResult::resource_cap;
            return false;
        }
        if (admitted.value() > stage_peak_) {
            contract();
        }
        return true;
    }
    bool reserve_compression(std::size_t chunk) noexcept {
        const auto& plan = report_.admission;
        if (!admit_stage(plan.peak)) {
            return false;
        }
        // The reviewed single-operation bound includes both RAW allocations and both
        // workspaces. Unused restore reservations are released before retaining the chunk.
        const std::array charges{plan.raw_charge,
                                 plan.compression.output_charge,
                                 plan.compression.workspace_charge,
                                 plan.compression.output_charge,
                                 plan.raw_charge,
                                 plan.decompression.workspace_charge};
        std::array<ReservationRequest, 6U> requests{};
        for (std::size_t index = 0U; index < requests.size(); ++index) {
            requests[index] = {PhysicalTier::gpu, buckets[index], charges[index],
                               AdmissionKind::normal};
        }
        auto& owner = owners_[chunk];
        if (!accept(ledger_.reserve(TransactionId{chunk + 1U}, requests, owner.tokens))) {
            return false;
        }
        owner.active.fill(true);
        return true;
    }
    bool reserve_restoration(std::size_t chunk) noexcept {
        const auto& plan = report_.admission;
        const auto peak =
            checked_add(plan.raw_charge.value(), plan.decompression.workspace_charge.value(),
                        OperationId::allocate);
        if (!peak) {
            failure(report_, peak.error());
            report_.result = CompressionSmokeResult::resource_cap;
            return false;
        }
        if (!admit_stage(ByteSize{peak.value()})) {
            return false;
        }
        const std::array requests{
            ReservationRequest{PhysicalTier::gpu, buckets[slot_index(restored)], plan.raw_charge,
                               AdmissionKind::normal},
            ReservationRequest{PhysicalTier::gpu, buckets[slot_index(decode_workspace)],
                               plan.decompression.workspace_charge, AdmissionKind::normal}};
        auto& owner = owners_[chunk];
        if (!accept(ledger_.reserve(TransactionId{chunk + m10_chunk_count + 1U}, requests,
                                    std::span{owner.tokens}.subspan(slot_index(restored), 2U)))) {
            return false;
        }
        owner.active[slot_index(restored)] = true;
        owner.active[slot_index(decode_workspace)] = true;
        return true;
    }
    bool reserve_address(std::size_t chunk, std::size_t slot) noexcept {
        const auto& caps = backend_->capabilities();
        const auto address = backend_->reserve_address_space(
            report_.admission.raw_charge,
            std::max(caps.mapping_granularity, caps.reservation_granularity));
        if (!address) {
            failure(report_, address.error());
            return false;
        }
        owners_[chunk].addresses[slot] = address.value();
        return true;
    }
    bool adopt(std::size_t chunk, ResourceSlot slot, Result<BackendAllocation> result,
               AllocationExpectation expectation) noexcept {
        if (!result) {
            failure(report_, result.error());
            return false;
        }
        for (const auto& owner : owners_) {
            for (const auto& allocation : owner.allocations) {
                if (allocation.id != ResourceId{} && allocation.id == result.value().id) {
                    contract();
                }
            }
        }
        auto& owner = owners_[chunk];
        owner.allocations[slot_index(slot)] =
            corroborate_allocation(*backend_, errors_, result.value(), expectation);
        require(ledger_.materialize(owner.tokens[slot_index(slot)],
                                    owner.allocations[slot_index(slot)].charge));
        require(ledger_.release_reservation(owner.tokens[slot_index(slot)]));
        owner.active[slot_index(slot)] = false;
        owner.ownership[slot_index(slot)] = Ownership::staging;
        corroborate_accounting();
        return true;
    }
    bool representation(std::size_t chunk, ResourceSlot slot, DeviceAddress address,
                        RepresentationState state) noexcept {
        const auto& plan = report_.admission;
        const auto bound = state == RepresentationState::gpu_raw ? plan.raw_charge
                                                                 : plan.compression.output_charge;
        if (!admit_allocation(bound)) {
            return false;
        }
        return adopt(chunk, slot,
                     backend_->allocate_representation(
                         {state, m10_logical_payload, owners_[chunk].content, address}),
                     {PhysicalTier::gpu, ResourceKind::representation, bound,
                      backend_->capabilities().allocation_granularity, m10_logical_payload, state,
                      address});
    }
    bool workspace(std::size_t chunk, ResourceSlot slot, ByteSize charge) noexcept {
        if (!admit_allocation(charge)) {
            return false;
        }
        return adopt(chunk, slot, backend_->allocate_workspace(PhysicalTier::gpu, charge),
                     {PhysicalTier::gpu,
                      ResourceKind::workspace,
                      charge,
                      backend_->capabilities().allocation_granularity,
                      {},
                      {},
                      {}});
    }
    void corroborate(std::size_t chunk, ResourceSlot slot) noexcept {
        const auto& expected = owners_[chunk].allocations[slot_index(slot)];
        const auto observed = backend_->allocation(expected.id);
        if (!observed || observed.value().id != expected.id ||
            observed.value().tier != expected.tier || observed.value().kind != expected.kind ||
            observed.value().address != expected.address ||
            observed.value().charge != expected.charge ||
            (expected.kind == ResourceKind::representation &&
             !same_metadata(observed.value().metadata, expected.metadata))) {
            contract();
        }
    }
    void reconcile(std::size_t chunk, TransferReceipt receipt, ResourceSlot destination,
                   ResourceSlot source, ResourceSlot workspace_slot, bool compacted) noexcept {
        auto& allocations = owners_[chunk].allocations;
        auto& target = allocations[slot_index(destination)];
        const auto& metadata = receipt.metadata;
        if (receipt.charge == ByteSize{} || receipt.charge > target.charge ||
            receipt.charge.value() % backend_->capabilities().allocation_granularity.value() !=
                0U ||
            metadata.logical_size != m10_logical_payload || metadata.stored_size == ByteSize{} ||
            metadata.stored_size > receipt.charge ||
            metadata.crc32c != report_.chunks[chunk].logical_crc_expected ||
            metadata.encoding != (compacted ? Encoding::lz4_block : Encoding::raw) ||
            (!compacted &&
             (receipt.charge != target.charge || metadata.stored_size != metadata.logical_size ||
              metadata.stored_crc32c != metadata.crc32c))) {
            contract();
        }
        if (compacted) {
            if (receipt.charge != allocations[slot_index(compaction)].charge) {
                contract();
            }
            require(ledger_.exchange_compaction(PhysicalTier::gpu, target.charge, receipt.charge));
            allocations[slot_index(compaction)].charge = target.charge;
        }
        target.charge = receipt.charge;
        target.metadata = metadata;
        if (compacted) {
            corroborate(chunk, compaction);
        }
        corroborate(chunk, destination);
        corroborate(chunk, source);
        corroborate(chunk, workspace_slot);
        corroborate_accounting();
    }
    bool initialize_raw(std::size_t chunk, std::span<const std::byte> initial,
                        std::span<std::byte> observed) noexcept {
        auto& owner = owners_[chunk];
        auto& record = report_.chunks[chunk];
        owner.content = {1U, chunk + 1U};
        record.logical_crc_expected = crc32c(initial);
        if (!reserve_address(chunk, 0U) ||
            !representation(chunk, raw, owner.addresses[0U].base, RepresentationState::gpu_raw)) {
            return false;
        }
        require(
            ledger_.commit_initial(PhysicalTier::gpu, owner.allocations[slot_index(raw)].charge));
        owner.ownership[slot_index(raw)] = Ownership::committed;
        owner.authority = Authority::gpu_raw;
        const auto written = backend_->write_bytes(owner.allocations[slot_index(raw)].id, {},
                                                   initial, owner.content);
        if (!written) {
            failure(report_, written.error());
            return false;
        }
        owner.content = written.value();
        const auto source = backend_->allocation(owner.allocations[slot_index(raw)].id);
        if (!source || source.value().id != owner.allocations[slot_index(raw)].id ||
            source.value().charge != owner.allocations[slot_index(raw)].charge ||
            source.value().address != owner.allocations[slot_index(raw)].address ||
            source.value().kind != ResourceKind::representation ||
            source.value().tier != PhysicalTier::gpu ||
            source.value().metadata.encoding != Encoding::raw ||
            source.value().metadata.logical_size != m10_logical_payload ||
            source.value().metadata.stored_size != m10_logical_payload ||
            source.value().metadata.crc32c != record.logical_crc_expected ||
            source.value().metadata.stored_crc32c != record.logical_crc_expected) {
            contract();
        }
        owner.allocations[slot_index(raw)] = source.value();
        if (!accept(backend_->verify_authoritative(source.value().id, owner.content)) ||
            !accept(backend_->read_bytes(source.value().id, {}, observed))) {
            return false;
        }
        if (!std::ranges::equal(initial, observed) ||
            crc32c(observed) != record.logical_crc_expected) {
            failure(report_, make_error(ErrorCode::integrity_failure, OperationId::verify));
            return false;
        }
        record.source_verified = true;
        // The actual adopted allocation, corroborated after source integrity, is the
        // immutable RAW baseline. Neither logical length nor an admission bound replaces it.
        record.raw_charge = source.value().charge;
        corroborate_accounting();
        return true;
    }
    bool compress_chunk(std::size_t chunk, std::span<const std::byte> initial,
                        std::span<std::byte> observed) noexcept {
        if (!reserve_compression(chunk)) {
            return false;
        }
        event(MultiChunkResidencyEventKind::compression_start, chunk);
        begin_compression_audit();
        if (!initialize_raw(chunk, initial, observed) ||
            !representation(chunk, compressed, {}, RepresentationState::gpu_compressed) ||
            !workspace(chunk, encode_workspace, report_.admission.compression.workspace_charge)) {
            return false;
        }
        auto& owner = owners_[chunk];
        auto& allocations = owner.allocations;
        auto& record = report_.chunks[chunk];
        const auto exact = backend_->prepare_transfer(allocations[slot_index(raw)].id,
                                                      allocations[slot_index(compressed)].id,
                                                      allocations[slot_index(encode_workspace)].id);
        capture_codec(record);
        if (!exact) {
            failure(report_, exact.error());
            return false;
        }
        if (exact.value() == ByteSize{} ||
            exact.value() > report_.admission.compression.output_charge ||
            exact.value().value() % report_.probe.minimum.value() != 0U) {
            contract();
        }
        if (!workspace(chunk, compaction, exact.value())) {
            return false;
        }
        const auto encoded = backend_->transfer(
            allocations[slot_index(raw)].id, allocations[slot_index(compressed)].id,
            allocations[slot_index(encode_workspace)].id,
            CompactionTarget{allocations[slot_index(compaction)].id});
        capture_codec(record);
        if (!encoded) {
            failure(report_, encoded.error());
            return false;
        }
        reconcile(chunk, encoded.value(), compressed, raw, encode_workspace, true);
        record.compaction_exercised = true;
        record.stored_bytes = allocations[slot_index(compressed)].metadata.stored_size;
        record.compressed_charge = allocations[slot_index(compressed)].charge;
        const auto transfer_verified = backend_->verify_transfer(
            allocations[slot_index(raw)].id, allocations[slot_index(compressed)].id, owner.content,
            allocations[slot_index(encode_workspace)].id);
        capture_stored_crc(chunk);
        if (!accept(transfer_verified)) {
            return false;
        }
        const auto authoritative =
            backend_->verify_authoritative(allocations[slot_index(compressed)].id, owner.content);
        capture_stored_crc(chunk);
        if (!accept(authoritative)) {
            return false;
        }
        if (!record.compression_enqueue || !record.compression_completion ||
            record.stored_crc_expected != record.stored_crc_actual ||
            record.stored_crc_expected !=
                allocations[slot_index(compressed)].metadata.stored_crc32c) {
            contract();
        }
        record.charges_corroborated = true;
        record.minimum_unit_saved =
            saves_physical_unit(record.raw_charge, record.compressed_charge, report_.probe.minimum);
        if (record.stored_bytes >= m10_logical_payload) {
            report_.result = CompressionSmokeResult::compression_not_beneficial;
            return false;
        }
        if (!record.minimum_unit_saved) {
            report_.result = CompressionSmokeResult::no_physical_savings;
            return false;
        }
        record.physical_bytes_saved =
            ByteSize{record.raw_charge.value() - record.compressed_charge.value()};
        commit_transition(chunk, compressed, raw, Authority::gpu_compressed);
        release(chunk, encode_workspace);
        release(chunk, compaction);
        release(chunk, raw);
        release_reservations(chunk);
        release_address(chunk, 0U);
        if (!corroborate_retained()) {
            return false;
        }
        event(MultiChunkResidencyEventKind::compressed_retained, chunk);
        return true;
    }
    void commit_transition(std::size_t chunk, ResourceSlot destination, ResourceSlot source,
                           Authority authority) noexcept {
        auto& owner = owners_[chunk];
        if (owner.ownership[slot_index(destination)] != Ownership::staging ||
            owner.ownership[slot_index(source)] != Ownership::committed) {
            contract();
        }
        require(ledger_.commit_destination_and_retire_source(
            PhysicalTier::gpu, owner.allocations[slot_index(destination)].charge, PhysicalTier::gpu,
            owner.allocations[slot_index(source)].charge));
        owner.ownership[slot_index(destination)] = Ownership::committed;
        owner.ownership[slot_index(source)] = Ownership::cleanup_debt;
        owner.authority = authority;
        corroborate_accounting();
    }
    bool residency_snapshot() noexcept {
        if (compressed_mask() != 15U || !corroborate_retained()) {
            if (!report_.has_error) {
                contract();
            }
            return false;
        }
        for (std::size_t chunk = 0U; chunk < owners_.size(); ++chunk) {
            const auto& owner = owners_[chunk];
            for (std::size_t slot = 0U; slot < owner.allocations.size(); ++slot) {
                if (slot != slot_index(compressed) && owner.allocations[slot].id != ResourceId{}) {
                    contract();
                }
            }
            for (std::size_t other = 0U; other < chunk; ++other) {
                if (owners_[other].allocations[slot_index(compressed)].id ==
                    owner.allocations[slot_index(compressed)].id) {
                    contract();
                }
            }
        }
        const auto totals =
            calculate_multi_chunk_residency_totals(report_.chunks, report_.probe.minimum);
        if (!totals) {
            failure(report_, totals.error());
            return false;
        }
        if (totals.value().logical_bytes != ByteSize{4U * m10_logical_payload.value()} ||
            totals.value().compressed_physical_charge !=
                backend_->owned_charge(PhysicalTier::gpu) ||
            totals.value().compressed_physical_charge != ledger_.charged(PhysicalTier::gpu) ||
            totals.value().compressed_physical_charge !=
                ledger_.usage(PhysicalTier::gpu).committed ||
            backend_->owned_resource_count() != m10_chunk_count ||
            ledger_.unmaterialized_reservations(PhysicalTier::gpu) != ByteSize{}) {
            contract();
        }
        report_.totals = totals.value();
        for (auto& chunk : report_.chunks) {
            chunk.compressed_at_snapshot = true;
        }
        report_.simultaneous_compressed_residency_proven = true;
        event(MultiChunkResidencyEventKind::aggregate_snapshot, m10_chunk_count);
        return true;
    }
    bool restore_chunk(std::size_t chunk, std::span<const std::byte> initial,
                       std::span<std::byte> observed) noexcept {
        if (!report_.simultaneous_compressed_residency_proven) {
            contract();
        }
        for (const auto& owner : owners_) {
            if (owner.allocations[slot_index(restored)].id != ResourceId{} ||
                owner.allocations[slot_index(decode_workspace)].id != ResourceId{}) {
                contract();
            }
        }
        if (!reserve_restoration(chunk)) {
            return false;
        }
        event(MultiChunkResidencyEventKind::restoration_start, chunk);
        auto& owner = owners_[chunk];
        auto& allocations = owner.allocations;
        auto& record = report_.chunks[chunk];
        begin_restoration_audit(record);
        if (crc32c(initial) != record.logical_crc_expected) {
            contract();
        }
        if (!reserve_address(chunk, 1U) ||
            !representation(chunk, restored, owner.addresses[1U].base,
                            RepresentationState::gpu_raw) ||
            !workspace(chunk, decode_workspace, report_.admission.decompression.workspace_charge)) {
            return false;
        }
        const auto decoded = backend_->transfer(allocations[slot_index(compressed)].id,
                                                allocations[slot_index(restored)].id,
                                                allocations[slot_index(decode_workspace)].id);
        capture_codec(record);
        if (!decoded) {
            failure(report_, decoded.error());
            return false;
        }
        reconcile(chunk, decoded.value(), restored, compressed, decode_workspace, false);
        const auto transfer_verified = backend_->verify_transfer(
            allocations[slot_index(compressed)].id, allocations[slot_index(restored)].id,
            owner.content, allocations[slot_index(decode_workspace)].id);
        capture_stored_crc(chunk);
        if (!accept(transfer_verified) ||
            !accept(backend_->read_bytes(allocations[slot_index(restored)].id, {}, observed))) {
            return false;
        }
        record.logical_crc_actual = crc32c(observed);
        record.size_equal =
            allocations[slot_index(restored)].metadata.logical_size == m10_logical_payload &&
            allocations[slot_index(restored)].metadata.stored_size == m10_logical_payload;
        record.byte_equal = std::ranges::equal(initial, observed);
        if (!record.size_equal || !record.byte_equal ||
            record.logical_crc_actual != record.logical_crc_expected ||
            record.stored_crc_actual != record.stored_crc_expected ||
            !record.decompression_enqueue || !record.decompression_completion) {
            failure(report_, make_error(ErrorCode::integrity_failure, OperationId::verify));
            return false;
        }
        record.round_trip_verified = true;
        commit_transition(chunk, restored, compressed, Authority::restored_raw);
        release(chunk, decode_workspace);
        release(chunk, compressed);
        release(chunk, restored);
        owner.authority = Authority::none;
        release_reservations(chunk);
        release_address(chunk, 1U);
        record.cleanup_complete = true;
        if (!corroborate_retained()) {
            return false;
        }
        event(MultiChunkResidencyEventKind::restoration_verified, chunk);
        return true;
    }
    void snapshot_final() noexcept {
        report_.final_resources = backend_->owned_resource_count();
        report_.final_va = backend_->owned_address_count();
        report_.final_budget = ledger_.charged(PhysicalTier::gpu);
        report_.final_unmaterialized_reservations =
            ledger_.unmaterialized_reservations(PhysicalTier::gpu);
        report_.final_workspace = 0U;
        report_.final_compaction = 0U;
        report_.final_bound_output = 0U;
        report_.final_raw = 0U;
        report_.final_compressed = 0U;
        for (std::size_t chunk = 0U; chunk < owners_.size(); ++chunk) {
            const auto present = [&](ResourceSlot slot) -> std::uint64_t {
                return owners_[chunk].allocations[slot_index(slot)].id != ResourceId{} ? 1U : 0U;
            };
            report_.final_workspace += present(encode_workspace) + present(decode_workspace);
            report_.final_compaction += present(compaction);
            report_.final_bound_output +=
                present(report_.chunks[chunk].compaction_exercised ? compaction : compressed);
            report_.final_raw += present(raw) + present(restored);
            report_.final_compressed += present(compressed);
        }
        report_.final_metadata = report_.final_workspace;
        const auto total =
            checked_add(backend_->owned_charge(PhysicalTier::gpu).value(),
                        report_.final_unmaterialized_reservations.value(), OperationId::verify);
        report_.final_counts_known = total && total.value() == report_.final_budget.value();
    }
    [[noreturn]] void stop(Error error) noexcept {
        failure(report_, error);
        if (error.code == ErrorCode::ambiguous_backend_state ||
            error.code == ErrorCode::backend_contract_violation) {
            report_.final_counts_known = false;
        } else {
            snapshot_final();
        }
        // Preserve exact known ownership on definite cleanup failure; never retry a
        // destructive operation or unwind operands after uncertain completion/cleanup.
        std::terminate();
    }
    void check_backend_error() noexcept {
        if (const auto error = backend_->first_error()) {
            stop(error->error);
        }
    }
    void release(std::size_t chunk, ResourceSlot slot) noexcept {
        auto& owner = owners_[chunk];
        auto& allocation = owner.allocations[slot_index(slot)];
        if (allocation.id == ResourceId{}) {
            return;
        }
        switch (owner.ownership[slot_index(slot)]) {
        case Ownership::committed:
            require(ledger_.retire_committed(PhysicalTier::gpu, allocation.charge));
            break;
        case Ownership::staging:
            require(ledger_.move_materialized_to_debt(PhysicalTier::gpu, buckets[slot_index(slot)],
                                                      allocation.charge));
            break;
        case Ownership::cleanup_debt:
            break;
        }
        owner.ownership[slot_index(slot)] = Ownership::cleanup_debt;
        require(backend_->release(allocation.id, ReleasePhase::close));
        if (backend_->owns(allocation.id)) {
            contract();
        }
        require(ledger_.release_cleanup_debt(PhysicalTier::gpu, allocation.charge));
        allocation = {};
        check_backend_error();
        corroborate_accounting();
    }
    void release_reservations(std::size_t chunk) noexcept {
        auto& owner = owners_[chunk];
        for (std::size_t slot = 0U; slot < owner.tokens.size(); ++slot) {
            if (owner.active[slot]) {
                require(ledger_.release_reservation(owner.tokens[slot]));
                owner.active[slot] = false;
            }
        }
    }
    void release_address(std::size_t chunk, std::size_t slot) noexcept {
        auto& address = owners_[chunk].addresses[slot];
        if (address.id != AddressReservationId{}) {
            require(backend_->release_address_space(address));
            address = {};
        }
    }
    void finish() noexcept {
        if (finished_) {
            return;
        }
        finished_ = true;
        if (report_.has_error && (report_.error.code == ErrorCode::ambiguous_backend_state ||
                                  report_.error.code == ErrorCode::backend_contract_violation)) {
            stop(report_.error);
        }
        check_backend_error();
        for (std::size_t chunk = 0U; chunk < owners_.size(); ++chunk) {
            release(chunk, encode_workspace);
            release(chunk, decode_workspace);
            release(chunk, compaction);
            release(chunk, compressed);
            release(chunk, restored);
            release(chunk, raw);
            release_reservations(chunk);
            release_address(chunk, 0U);
            release_address(chunk, 1U);
            owners_[chunk].authority = Authority::none;
            report_.chunks[chunk].cleanup_complete = true;
        }
        report_.cleanup_workspace = true;
        report_.cleanup_bound_output = true;
        report_.cleanup_compressed = true;
        report_.cleanup_raw = true;
        report_.cleanup_va = true;
        require(backend_->shutdown());
        report_.cleanup_stream = !report_.codec.stream_owned;
        report_.cleanup_context = !report_.codec.context_owned;
        snapshot_final();
        if (!report_.final_counts_known || report_.final_resources != 0U ||
            report_.final_va != 0U || report_.final_budget != ByteSize{} ||
            !report_.cleanup_stream || !report_.cleanup_context) {
            contract();
        }
        event(MultiChunkResidencyEventKind::cleanup_complete, m10_chunk_count);
        if (verified_) {
            if (!report_.simultaneous_compressed_residency_proven || report_.has_error) {
                contract();
            }
            report_.result = CompressionSmokeResult::passed;
            event(MultiChunkResidencyEventKind::pass_published, m10_chunk_count);
        }
    }

    std::unique_ptr<CudaVmmBackend> backend_;
    MultiChunkResidencySmokeReport& report_;
    BudgetLedger ledger_;
    AsyncErrorChannel errors_{8U};
    std::array<ChunkOwner, m10_chunk_count> owners_{};
    ByteSize stage_peak_{};
    bool verified_{};
    bool finished_{};
};
} // namespace

void finalize_multi_chunk_residency_smoke_failure(MultiChunkResidencySmokeReport& report) noexcept {
    if (report.codec.has_fatal_error) {
        report.error = report.codec.fatal_error;
        report.has_error = true;
        report.final_counts_known = false;
    } else if (!report.has_error) {
        report.error = make_error(ErrorCode::backend_contract_violation, OperationId::unknown);
        report.has_error = true;
        report.final_counts_known = false;
    }
    report.result = CompressionSmokeResult::failed;
}

void run_multi_chunk_residency_smoke(std::unique_ptr<CudaDriverApi> driver,
                                     std::unique_ptr<GpuCompressionApi> codec,
                                     MultiChunkResidencySmokeOptions options,
                                     MultiChunkResidencySmokeReport& report) noexcept {
    report = {};
    report.device_ordinal = options.device;
    report.installed_driver = options.installed_driver;
    report.runtime_api = options.runtime_api;
    report.prior_driver_api = options.prior_driver_api;
    report.admission.logical_size = m10_logical_payload;
    report.admission.physical_cap = m10_physical_cap;
    report.admission.minimum_driver_version = 13000;
    report.admission.required_driver_family = 13;
    report.admission.required_device_name = m8_device_name;
    if (options.device != 0 || !driver || !codec) {
        failure(report, make_error(ErrorCode::invalid_argument, OperationId::runtime_create));
        report.result = CompressionSmokeResult::preflight_rejected;
        return;
    }
    if (!options.acknowledged) {
        report.final_counts_known = true;
        return;
    }
    report.compatibility = compression_runtime_candidate(
        options.prior_driver_api, options.runtime_api, options.installed_driver);
    if (!is_candidate(report.compatibility)) {
        failure(report, make_error(ErrorCode::unsupported, OperationId::runtime_create));
        report.final_counts_known = true;
        return;
    }
    auto created = CudaVmmBackend::create(std::move(driver), 0, std::move(codec), &report.probe,
                                          &report.admission, &report.codec);
    if (!created) {
        failure(report, created.error());
        if (created.error().code == ErrorCode::ambiguous_backend_state ||
            created.error().code == ErrorCode::backend_contract_violation) {
            std::terminate();
        }
        if (report.admission.peak > m10_physical_cap ||
            created.error().code == ErrorCode::arithmetic_overflow) {
            report.result = CompressionSmokeResult::resource_cap;
        }
        report.final_counts_known = true;
        return;
    }
    ResidencySession session{std::move(created).value(), report};
    try {
        session.run();
    } catch (const std::bad_alloc&) {
        failure(report, make_error(ErrorCode::out_of_host_memory, OperationId::allocate));
    }
}
} // namespace vramz::detail
