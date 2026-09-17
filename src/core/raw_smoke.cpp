#include "vramz/detail/raw_smoke.hpp"

#include "vramz/budget_ledger.hpp"
#include "vramz/crc32c.hpp"
#include "vramz/detail/allocation.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <utility>

namespace vramz::detail {
namespace {

void record_failure(RawSmokeReport& report, Error error) noexcept {
    report.has_error = true;
    report.error = error;
    report.result =
        error.code == ErrorCode::unsupported ? RawSmokeResult::unsupported : RawSmokeResult::failed;
}

// Single scoped owner for core admission/commit and the production backend. There is no
// Runtime selector, policy, lease, transition, compression adapter or external writer here.
class SmokeSession final {
  public:
    SmokeSession(std::unique_ptr<CudaVmmBackend> backend, RawSmokeReport& report,
                 ByteSize cap) noexcept
        : backend_(std::move(backend)), report_(report), ledger_({{cap, cap, {}}, {}}), cap_(cap) {}
    SmokeSession(const SmokeSession&) = delete;
    SmokeSession& operator=(const SmokeSession&) = delete;
    ~SmokeSession() noexcept { finish(); }

    void run() noexcept {
        const auto& caps = backend_->capabilities();
        const auto physical = caps.allocation_granularity;
        if (backend_->compression_available() || caps.host_tier || !caps.stable_device_address ||
            physical == ByteSize{} || physical > m7_physical_cap || physical > cap_ ||
            caps.reservation_granularity == ByteSize{} ||
            physical.value() % caps.reservation_granularity.value() != 0U) {
            report_.result = RawSmokeResult::unsupported_granularity;
            return;
        }
        const auto size = static_cast<std::size_t>(
            std::min(physical.value(), static_cast<std::uint64_t>(m7_payload_cap)));
        const ByteSize logical{static_cast<std::uint64_t>(size)};
        std::array<std::byte, m7_payload_cap> initial{};
        std::array<std::byte, m7_payload_cap> observed{};
        for (std::size_t index = 0U; index < size; ++index) {
            initial[index] = static_cast<std::byte>(
                index % 17U == 0U ? 0U : (index * 37U + index / 251U) & 255U);
        }
        const auto input = std::span<const std::byte>{initial}.first(size);
        const auto output = std::span<std::byte>{observed}.first(size);
        report_.logical_payload = logical;
        report_.crc_expected = crc32c(input);
        const auto bound = backend_->allocation_bound(RepresentationState::gpu_raw, logical);
        if (!bound || bound.value() != physical) {
            stop(bound ? make_error(ErrorCode::backend_contract_violation, OperationId::allocate)
                       : bound.error());
        }
        const std::array requests{ReservationRequest{PhysicalTier::gpu, ChargeBucket::staging,
                                                     physical, AdmissionKind::normal}};
        const auto reserved = ledger_.reserve(TransactionId{1U}, requests, std::span{&token_, 1U});
        if (!reserved) {
            record_failure(report_, reserved.error());
            return;
        }
        token_active_ = true;
        const auto address = backend_->reserve_address_space(
            physical, std::max(caps.mapping_granularity, caps.reservation_granularity));
        if (!address) {
            record_failure(report_, address.error());
            return;
        }
        address_ = address.value();
        report_.stable_va_reserved = true;
        const auto allocated = backend_->allocate_initialized_raw(
            {RepresentationState::gpu_raw, logical, ContentTag{1U, 1U}, address_.base}, input,
            output);
        if (!allocated) {
            record_failure(report_, allocated.error());
            return;
        }
        // The same M2 barrier as Runtime and TransactionCoordinator: no provisional debt.
        allocation_ = corroborate_allocation(*backend_, errors_, allocated.value(),
                                             {PhysicalTier::gpu, ResourceKind::representation,
                                              physical, caps.allocation_granularity, logical,
                                              RepresentationState::gpu_raw, address_.base});
        require(ledger_.materialize(token_, allocation_.charge));
        require(ledger_.release_reservation(token_));
        token_active_ = false;
        require(ledger_.commit_initial(PhysicalTier::gpu, allocation_.charge));
        report_.physical_charge = allocation_.charge;
        report_.h2d_success = true;
        report_.d2h_success = true;
        report_.size_equal = input.size() == output.size();
        report_.crc_actual = crc32c(output);
        report_.byte_compare = std::ranges::equal(input, output);
        report_.mapping_verified = true;
        report_.access_verified = true;
        if (allocation_.metadata.crc32c != report_.crc_expected || !report_.size_equal ||
            !report_.byte_compare || report_.crc_expected != report_.crc_actual) {
            record_failure(report_, make_error(ErrorCode::integrity_failure, OperationId::verify));
            return;
        }
        round_trip_verified_ = true;
    }

  private:
    void require(Result<void> result) noexcept {
        if (!result) {
            stop(result.error());
        }
    }
    [[noreturn]] void stop(Error error) noexcept {
        record_failure(report_, error);
        errors_.push(error);
        snapshot();
        // Deliberately no retry or destructor cleanup after a failed cleanup operation.
        // Definite failures retain exact ownership; ambiguous backend calls stop earlier.
        std::terminate();
    }
    void snapshot() noexcept {
        report_.final_backend_resources = backend_->owned_resource_count();
        report_.final_va_reservations = backend_->owned_address_count();
        report_.final_budget_charge = ledger_.charged(PhysicalTier::gpu);
        report_.final_counts_known =
            report_.final_budget_charge.value() ==
            backend_->owned_charge(PhysicalTier::gpu).value() +
                ledger_.unmaterialized_reservations(PhysicalTier::gpu).value();
    }
    void finish() noexcept {
        if (allocation_.id != ResourceId{}) {
            require(ledger_.retire_committed(PhysicalTier::gpu, allocation_.charge));
            // release() performs whole unmap then physical release, never the opposite.
            require(backend_->release(allocation_.id, ReleasePhase::close));
            require(ledger_.release_cleanup_debt(PhysicalTier::gpu, allocation_.charge));
            allocation_ = {};
            report_.cleanup_unmap = true;
            report_.cleanup_physical_release = true;
        }
        if (token_active_) {
            require(ledger_.release_reservation(token_));
            token_active_ = false;
        }
        if (address_.id != AddressReservationId{}) {
            require(backend_->release_address_space(address_));
            address_ = {};
            report_.cleanup_va_free = true;
        }
        require(backend_->shutdown());
        report_.cleanup_context_release = true;
        snapshot();
        if (report_.final_backend_resources != 0U || report_.final_va_reservations != 0U ||
            report_.final_budget_charge != ByteSize{} ||
            backend_->owned_charge(PhysicalTier::gpu) != ByteSize{}) {
            stop(make_error(ErrorCode::internal_invariant_violation, OperationId::shutdown));
        }
        if (round_trip_verified_) {
            report_.result = RawSmokeResult::passed;
        }
    }

    std::unique_ptr<CudaVmmBackend> backend_;
    RawSmokeReport& report_;
    BudgetLedger ledger_;
    const ByteSize cap_;
    AsyncErrorChannel errors_{8U};
    ReservationToken token_{};
    bool token_active_{};
    bool round_trip_verified_{};
    AddressReservation address_{};
    BackendAllocation allocation_{};
};

} // namespace

void run_raw_smoke(std::unique_ptr<CudaDriverApi> driver, RawSmokeOptions options,
                   RawSmokeReport& report) noexcept {
    report = {};
    report.device_ordinal = options.device;
    report.maximum_physical = options.maximum_physical;
    if (options.device < 0 || options.maximum_physical == ByteSize{} ||
        options.maximum_physical > m7_physical_cap) {
        report.result = RawSmokeResult::preflight_rejected;
        report.has_error = true;
        report.error = make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
        return;
    }
    if (!options.acknowledged) {
        return;
    }
    auto created = CudaVmmBackend::create(std::move(driver), options.device, {}, &report.probe);
    if (!created) {
        record_failure(report, created.error());
        if (created.error().code == ErrorCode::ambiguous_backend_state ||
            created.error().code == ErrorCode::backend_contract_violation) {
            std::terminate();
        }
        // A failed factory returned only after its private RAII owner was fully released.
        report.final_counts_known = true;
        return;
    }
    SmokeSession session{std::move(created).value(), report, options.maximum_physical};
    session.run();
}

} // namespace vramz::detail
