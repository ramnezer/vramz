#include "vramz/detail/physical_savings_smoke.hpp"

#include "vramz/budget_ledger.hpp"
#include "vramz/crc32c.hpp"
#include "vramz/detail/allocation.hpp"

#include <algorithm>
#include <exception>
#include <new>
#include <utility>
#include <vector>

namespace vramz::detail {
namespace {

constexpr std::size_t raw = 0U;
constexpr std::size_t compressed = 1U;
constexpr std::size_t encode_workspace = 2U;
constexpr std::size_t compaction = 3U;
constexpr std::size_t restored = 4U;
constexpr std::size_t decode_workspace = 5U;
constexpr std::array buckets{ChargeBucket::staging,   ChargeBucket::staging,
                             ChargeBucket::workspace, ChargeBucket::workspace,
                             ChargeBucket::staging,   ChargeBucket::workspace};

void failure(CompressionSmokeReport& report, Error error) noexcept {
    report.has_error = true;
    report.error = error;
    report.result = error.code == ErrorCode::unsupported ? CompressionSmokeResult::unsupported
                                                         : CompressionSmokeResult::failed;
}

// One private batch-one owner. Production backend methods perform all codec operations;
// no Runtime policy/eviction/reclaim or public representation publication occurs here.
class CompressionSession final {
  public:
    CompressionSession(std::unique_ptr<CudaVmmBackend> backend, CompressionSmokeReport& report,
                       bool savings) noexcept
        : backend_(std::move(backend)), report_(report),
          ledger_({{report.admission.physical_cap, report.admission.physical_cap, {}}, {}}),
          savings_(savings) {}
    CompressionSession(const CompressionSession&) = delete;
    CompressionSession& operator=(const CompressionSession&) = delete;
    ~CompressionSession() noexcept { finish(); }

    void run() {
        const auto& plan = report_.admission;
        const auto logical = plan.logical_size;
        std::vector<std::byte> initial(static_cast<std::size_t>(logical.value()));
        std::vector<std::byte> observed(initial.size());
        // Repeated 4096-byte records with varying, nonzero fields; never an all-zero fixture.
        for (std::size_t index = 0U; index < initial.size(); ++index) {
            initial[index] =
                savings_
                    ? physical_savings_payload_byte(index)
                    : static_cast<std::byte>(1U + ((index % 4096U) * 37U + index % 17U) % 251U);
        }
        report_.logical_crc_expected = crc32c(initial);
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
        if (!accept(ledger_.reserve(TransactionId{1U}, requests, tokens_))) {
            return;
        }
        active_.fill(true);
        if (!reserve_address(0U) || !reserve_address(1U)) {
            return;
        }
        if (!representation(raw, addresses_[0U].base, RepresentationState::gpu_raw) ||
            !representation(compressed, {}, RepresentationState::gpu_compressed) ||
            !workspace(encode_workspace, plan.compression.workspace_charge)) {
            return;
        }
        require(ledger_.commit_initial(PhysicalTier::gpu, allocations_[raw].charge));
        raw_committed_ = true;
        const auto written = backend_->write_bytes(allocations_[raw].id, {}, initial, content_);
        if (!written) {
            failure(report_, written.error());
            return;
        }
        content_ = written.value();
        const auto source = backend_->allocation(allocations_[raw].id);
        if (!source || source.value().id != allocations_[raw].id ||
            source.value().charge != allocations_[raw].charge ||
            source.value().address != allocations_[raw].address ||
            source.value().kind != ResourceKind::representation ||
            source.value().tier != PhysicalTier::gpu ||
            source.value().metadata.encoding != Encoding::raw ||
            source.value().metadata.logical_size != logical ||
            source.value().metadata.stored_size != logical ||
            source.value().metadata.crc32c != report_.logical_crc_expected ||
            source.value().metadata.stored_crc32c != report_.logical_crc_expected) {
            contract();
        }
        allocations_[raw] = source.value();
        if (!accept(backend_->verify_authoritative(allocations_[raw].id, content_)) ||
            !accept(backend_->read_bytes(allocations_[raw].id, {}, observed))) {
            return;
        }
        if (!std::ranges::equal(initial, observed) ||
            crc32c(observed) != report_.logical_crc_expected) {
            failure(report_, make_error(ErrorCode::integrity_failure, OperationId::verify));
            return;
        }
        report_.source_verified = true;
        report_.raw_charge = allocations_[raw].charge;
        corroborate_accounting();
        const auto exact = backend_->prepare_transfer(
            allocations_[raw].id, allocations_[compressed].id, allocations_[encode_workspace].id);
        if (!exact) {
            failure(report_, exact.error());
            return;
        }
        if (exact.value() == ByteSize{} || exact.value() > plan.compression.output_charge ||
            exact.value().value() % backend_->capabilities().allocation_granularity.value() != 0U) {
            contract();
        }
        if (!workspace(compaction, exact.value())) {
            return;
        }
        const auto encoded = backend_->transfer(allocations_[raw].id, allocations_[compressed].id,
                                                allocations_[encode_workspace].id,
                                                CompactionTarget{allocations_[compaction].id});
        if (!encoded) {
            failure(report_, encoded.error());
            return;
        }
        reconcile(encoded.value(), compressed, raw, encode_workspace, true);
        report_.compaction_exercised = true;
        report_.stored_bytes = allocations_[compressed].metadata.stored_size;
        report_.compressed_charge = allocations_[compressed].charge;
        corroborate_accounting();
        report_.charges_corroborated = true;
        if (!accept(backend_->verify_transfer(allocations_[raw].id, allocations_[compressed].id,
                                              content_, allocations_[encode_workspace].id)) ||
            !accept(backend_->verify_authoritative(allocations_[compressed].id, content_))) {
            return;
        }
        if (!representation(restored, addresses_[1U].base, RepresentationState::gpu_raw) ||
            !workspace(decode_workspace, plan.decompression.workspace_charge)) {
            return;
        }
        const auto decoded =
            backend_->transfer(allocations_[compressed].id, allocations_[restored].id,
                               allocations_[decode_workspace].id);
        if (!decoded) {
            failure(report_, decoded.error());
            return;
        }
        reconcile(decoded.value(), restored, compressed, decode_workspace, false);
        if (!accept(backend_->verify_transfer(allocations_[compressed].id,
                                              allocations_[restored].id, content_,
                                              allocations_[decode_workspace].id)) ||
            !accept(backend_->read_bytes(allocations_[restored].id, {}, observed))) {
            return;
        }
        report_.logical_crc_actual = crc32c(observed);
        report_.size_equal = allocations_[restored].metadata.stored_size == logical &&
                             allocations_[restored].metadata.logical_size == logical;
        report_.byte_equal = std::ranges::equal(initial, observed);
        if (!report_.size_equal || !report_.byte_equal ||
            report_.logical_crc_actual != report_.logical_crc_expected ||
            report_.codec.stored_crc_expected != report_.codec.stored_crc_actual) {
            failure(report_, make_error(ErrorCode::integrity_failure, OperationId::verify));
            return;
        }
        const auto statistics = backend_->compression_stats();
        if (statistics.successful_compressions != 1U ||
            statistics.successful_decompressions != 1U || !report_.codec.compression_enqueue ||
            !report_.codec.compression_completion || !report_.codec.decompression_enqueue ||
            !report_.codec.decompression_completion) {
            contract();
        }
        verified_ = true;
    }

  private:
    [[noreturn]] void contract() noexcept {
        failure(report_, make_error(ErrorCode::backend_contract_violation, OperationId::migrate));
        report_.final_counts_known = false;
        std::terminate();
    }
    void require(Result<void> value) noexcept {
        if (!value) {
            stop(value.error());
        }
    }
    bool accept(Result<void> value) noexcept {
        if (!value) {
            failure(report_, value.error());
            return false;
        }
        return true;
    }
    void corroborate_accounting() noexcept {
        const auto owned = backend_->owned_charge(PhysicalTier::gpu).value();
        const auto charged = ledger_.charged(PhysicalTier::gpu).value();
        const auto reserved = ledger_.unmaterialized_reservations(PhysicalTier::gpu).value();
        if (charged < reserved || charged - reserved != owned) {
            contract();
        }
    }
    void snapshot() noexcept {
        report_.final_resources = backend_->owned_resource_count();
        report_.final_va = backend_->owned_address_count();
        report_.final_budget = ledger_.charged(PhysicalTier::gpu);
        const auto present = [&](std::size_t index) -> std::uint64_t {
            return allocations_[index].id != ResourceId{} ? 1U : 0U;
        };
        report_.final_workspace = present(encode_workspace) + present(decode_workspace);
        report_.final_metadata = report_.final_workspace; // Packed into those same allocations.
        report_.final_compaction = present(compaction);
        report_.final_bound_output =
            present(report_.compaction_exercised ? compaction : compressed);
        report_.final_raw = present(raw) + present(restored);
        report_.final_compressed = present(compressed);
        report_.final_counts_known =
            report_.final_budget.value() ==
            backend_->owned_charge(PhysicalTier::gpu).value() +
                ledger_.unmaterialized_reservations(PhysicalTier::gpu).value();
    }
    [[noreturn]] void stop(Error error) noexcept {
        failure(report_, error);
        if (error.code == ErrorCode::ambiguous_backend_state ||
            error.code == ErrorCode::backend_contract_violation) {
            report_.final_counts_known = false;
        } else {
            snapshot();
        }
        // One cleanup pass only. No destructor or retry follows uncertain/failed cleanup.
        std::terminate();
    }
    void check_backend_error() noexcept {
        if (const auto error = backend_->first_error()) {
            stop(error->error);
        }
    }
    bool reserve_address(std::size_t index) noexcept {
        const auto& caps = backend_->capabilities();
        const auto address = backend_->reserve_address_space(
            report_.admission.raw_charge,
            std::max(caps.mapping_granularity, caps.reservation_granularity));
        if (!address) {
            failure(report_, address.error());
            return false;
        }
        addresses_[index] = address.value();
        return true;
    }
    bool adopt(std::size_t index, Result<BackendAllocation> result,
               AllocationExpectation expectation) noexcept {
        if (!result) {
            failure(report_, result.error());
            return false;
        }
        for (const auto& other : allocations_) {
            if (other.id != ResourceId{} && other.id == result.value().id) {
                contract();
            }
        }
        allocations_[index] =
            corroborate_allocation(*backend_, errors_, result.value(), expectation);
        require(ledger_.materialize(tokens_[index], allocations_[index].charge));
        require(ledger_.release_reservation(tokens_[index]));
        active_[index] = false;
        return true;
    }
    bool representation(std::size_t index, DeviceAddress address,
                        RepresentationState state) noexcept {
        const auto& p = report_.admission;
        const auto bound =
            state == RepresentationState::gpu_raw ? p.raw_charge : p.compression.output_charge;
        return adopt(
            index, backend_->allocate_representation({state, p.logical_size, content_, address}),
            {PhysicalTier::gpu, ResourceKind::representation, bound,
             backend_->capabilities().allocation_granularity, p.logical_size, state, address});
    }
    bool workspace(std::size_t index, ByteSize charge) noexcept {
        return adopt(index, backend_->allocate_workspace(PhysicalTier::gpu, charge),
                     {PhysicalTier::gpu,
                      ResourceKind::workspace,
                      charge,
                      backend_->capabilities().allocation_granularity,
                      {},
                      {},
                      {}});
    }
    void corroborate(std::size_t index) noexcept {
        const auto& expected = allocations_[index];
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
    void reconcile(TransferReceipt receipt, std::size_t destination, std::size_t source,
                   std::size_t workspace_index, bool compacted) noexcept {
        auto& target = allocations_[destination];
        const auto& metadata = receipt.metadata;
        if (receipt.charge == ByteSize{} || receipt.charge > target.charge ||
            receipt.charge.value() % backend_->capabilities().allocation_granularity.value() !=
                0U ||
            metadata.logical_size != report_.admission.logical_size ||
            metadata.stored_size == ByteSize{} || metadata.stored_size > receipt.charge ||
            metadata.crc32c != report_.logical_crc_expected ||
            metadata.encoding != (compacted ? Encoding::lz4_block : Encoding::raw) ||
            (!compacted &&
             (receipt.charge != target.charge || metadata.stored_size != metadata.logical_size ||
              metadata.stored_crc32c != metadata.crc32c))) {
            contract();
        }
        if (compacted) {
            if (receipt.charge != allocations_[compaction].charge) {
                contract();
            }
            // Rebucket the proven backing exchange before any fallible backend query.
            require(ledger_.exchange_compaction(PhysicalTier::gpu, target.charge, receipt.charge));
            allocations_[compaction].charge = target.charge;
        }
        target.charge = receipt.charge;
        target.metadata = metadata;
        if (compacted) {
            corroborate(compaction);
        }
        corroborate(destination);
        corroborate(source);
        corroborate(workspace_index);
    }
    void release(std::size_t index) noexcept {
        auto& allocation = allocations_[index];
        if (allocation.id == ResourceId{}) {
            return;
        }
        if (index == raw && raw_committed_) {
            require(ledger_.retire_committed(PhysicalTier::gpu, allocation.charge));
        } else {
            require(ledger_.move_materialized_to_debt(PhysicalTier::gpu, buckets[index],
                                                      allocation.charge));
        }
        require(backend_->release(allocation.id, ReleasePhase::close));
        if (backend_->owns(allocation.id)) {
            contract();
        }
        require(ledger_.release_cleanup_debt(PhysicalTier::gpu, allocation.charge));
        allocation = {};
        // A definite private-VA free failure is separate, zero-physical-charge ownership.
        // Stop after exact ledger release; do not reach backend shutdown's normal VA retry.
        check_backend_error();
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
        release(encode_workspace);
        release(decode_workspace);
        report_.cleanup_workspace = true;
        release(compaction);
        report_.cleanup_bound_output = true;
        release(compressed);
        report_.cleanup_compressed = true;
        release(restored);
        release(raw);
        report_.cleanup_raw = true;
        for (std::size_t index = 0U; index < tokens_.size(); ++index) {
            if (active_[index]) {
                require(ledger_.release_reservation(tokens_[index]));
                active_[index] = false;
            }
        }
        for (auto& address : addresses_) {
            if (address.id != AddressReservationId{}) {
                require(backend_->release_address_space(address));
                address = {};
            }
        }
        report_.cleanup_va = true;
        require(backend_->shutdown());
        report_.cleanup_stream = !report_.codec.stream_owned;
        report_.cleanup_context = !report_.codec.context_owned;
        snapshot();
        if (!report_.final_counts_known || report_.final_resources != 0U ||
            report_.final_va != 0U || report_.final_budget != ByteSize{} ||
            !report_.cleanup_stream || !report_.cleanup_context) {
            contract();
        }
        if (verified_) {
            report_.result = CompressionSmokeResult::passed;
            if (savings_) {
                if (!report_.charges_corroborated) {
                    contract();
                }
                if (report_.stored_bytes >= report_.admission.logical_size) {
                    report_.result = CompressionSmokeResult::compression_not_beneficial;
                } else if (!saves_physical_unit(report_.raw_charge, report_.compressed_charge,
                                                report_.probe.minimum)) {
                    report_.result = CompressionSmokeResult::no_physical_savings;
                }
            }
        }
    }

    std::unique_ptr<CudaVmmBackend> backend_;
    CompressionSmokeReport& report_;
    BudgetLedger ledger_;
    AsyncErrorChannel errors_{8U};
    std::array<BackendAllocation, 6U> allocations_{};
    std::array<ReservationToken, 6U> tokens_{};
    std::array<bool, 6U> active_{};
    std::array<AddressReservation, 2U> addresses_{};
    ContentTag content_{1U, 1U};
    const bool savings_;
    bool raw_committed_{};
    bool verified_{};
    bool finished_{};
};
} // namespace

void finalize_compression_smoke_failure(CompressionSmokeReport& report) noexcept {
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

namespace {
void run_smoke(std::unique_ptr<CudaDriverApi> driver, std::unique_ptr<GpuCompressionApi> codec,
               CompressionSmokeOptions options, CompressionSmokeReport& report,
               bool savings) noexcept {
    report = {};
    report.device_ordinal = options.device;
    report.installed_driver = options.installed_driver;
    report.runtime_api = options.runtime_api;
    report.prior_driver_api = options.prior_driver_api;
    report.admission.logical_size = options.logical_size;
    report.admission.physical_cap = options.physical_cap;
    report.admission.minimum_driver_version = 13000;
    report.admission.required_driver_family = 13;
    report.admission.required_device_name = m8_device_name;
    const bool valid_bounds =
        savings
            ? options.logical_size == m9_logical_payload && options.physical_cap == m9_physical_cap
            : options.logical_size != ByteSize{} && options.logical_size <= m8_logical_cap &&
                  options.physical_cap != ByteSize{} && options.physical_cap <= m8_physical_cap;
    if (options.device != 0 || !valid_bounds || !driver || !codec) {
        failure(report, make_error(ErrorCode::invalid_argument, OperationId::runtime_create));
        report.result = CompressionSmokeResult::preflight_rejected;
        return;
    }
    if (!options.acknowledged) {
        report.final_counts_known = true;
        return;
    }
    // The pinned prior M7 observation permits a candidate only. No CUDA call occurs
    // in this preflight. The backend corroborates the live API family after cuInit.
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
        if (report.admission.peak > report.admission.physical_cap ||
            created.error().code == ErrorCode::arithmetic_overflow) {
            report.result = CompressionSmokeResult::resource_cap;
        }
        report.final_counts_known = true;
        return;
    }
    CompressionSession session{std::move(created).value(), report, savings};
    try {
        session.run();
    } catch (const std::bad_alloc&) {
        failure(report, make_error(ErrorCode::out_of_host_memory, OperationId::allocate));
    }
}
} // namespace

void run_compression_smoke(std::unique_ptr<CudaDriverApi> driver,
                           std::unique_ptr<GpuCompressionApi> codec,
                           CompressionSmokeOptions options,
                           CompressionSmokeReport& report) noexcept {
    run_smoke(std::move(driver), std::move(codec), options, report, false);
}
void run_physical_savings_smoke(std::unique_ptr<CudaDriverApi> driver,
                                std::unique_ptr<GpuCompressionApi> codec,
                                PhysicalSavingsSmokeOptions options,
                                CompressionSmokeReport& report) noexcept {
    const CompressionSmokeOptions fixed{
        options.acknowledged,     options.device,      m9_logical_payload,      m9_physical_cap,
        options.installed_driver, options.runtime_api, options.prior_driver_api};
    run_smoke(std::move(driver), std::move(codec), fixed, report, true);
}
} // namespace vramz::detail
