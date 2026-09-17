#include "vramz/budget_ledger.hpp"

#include "vramz/checked.hpp"

#include <array>
#include <limits>

namespace vramz {
namespace {

[[nodiscard]] Error ledger_error(ErrorCode code, std::uint64_t detail = 0U) noexcept {
    return make_error(code, OperationId::budget_transfer, 0U, detail);
}

[[nodiscard]] Error oom_error(PhysicalTier tier, std::uint64_t amount) noexcept {
    return make_error(tier == PhysicalTier::gpu ? ErrorCode::out_of_gpu_memory
                                                : ErrorCode::out_of_host_memory,
                      OperationId::budget_reserve, 0U, amount);
}

} // namespace

ReservationToken::ReservationToken(ReservationToken&& other) noexcept
    : owner(other.owner), transaction(other.transaction), token_id(other.token_id),
      tier(other.tier), bucket(other.bucket), remaining(other.remaining), active(other.active) {
    other.owner = nullptr;
    other.token_id = 0U;
    other.remaining = ByteSize{};
    other.active = false;
}

BudgetLedger::BudgetLedger(MemoryBudgets budgets) noexcept : budgets_(budgets) {}

BudgetLedger::Counters& BudgetLedger::counters(PhysicalTier tier) noexcept {
    return tier == PhysicalTier::gpu ? gpu_ : host_;
}

const BudgetLedger::Counters& BudgetLedger::counters(PhysicalTier tier) const noexcept {
    return tier == PhysicalTier::gpu ? gpu_ : host_;
}

const TierBudget& BudgetLedger::budget(PhysicalTier tier) const noexcept {
    return tier == PhysicalTier::gpu ? budgets_.gpu : budgets_.host;
}

Result<std::uint64_t> BudgetLedger::total(const Counters& value) noexcept {
    auto sum = checked_add(value.committed, value.reserved, OperationId::budget_transfer);
    if (!sum) {
        return sum.error();
    }
    sum = checked_add(sum.value(), value.staging, OperationId::budget_transfer);
    if (!sum) {
        return sum.error();
    }
    sum = checked_add(sum.value(), value.workspace, OperationId::budget_transfer);
    if (!sum) {
        return sum.error();
    }
    return checked_add(sum.value(), value.cleanup_debt, OperationId::budget_transfer);
}

Result<void> BudgetLedger::add_to(std::uint64_t& value, std::uint64_t amount) noexcept {
    const auto result = checked_add(value, amount, OperationId::budget_transfer);
    if (!result) {
        return result.error();
    }
    value = result.value();
    return {};
}

Result<void> BudgetLedger::subtract_from(std::uint64_t& value, std::uint64_t amount) noexcept {
    const auto result = checked_sub(value, amount, OperationId::budget_transfer);
    if (!result) {
        return ledger_error(ErrorCode::internal_invariant_violation, amount);
    }
    value = result.value();
    return {};
}

void BudgetLedger::update_peak(Counters& value) noexcept {
    const auto current = total(value);
    if (current && current.value() > value.peak_charged) {
        value.peak_charged = current.value();
    }
}

Result<void> BudgetLedger::reserve(TransactionId transaction,
                                   std::span<const ReservationRequest> requests,
                                   std::span<ReservationToken> output) noexcept {
    if (transaction.value() == 0U || requests.empty() || output.size() < requests.size()) {
        return make_error(ErrorCode::invalid_argument, OperationId::budget_reserve);
    }
    for (std::size_t index = 0U; index < requests.size(); ++index) {
        if (output[index].active) {
            return make_error(ErrorCode::busy, OperationId::budget_reserve, output[index].token_id);
        }
    }

    std::array<std::uint64_t, 2U> requested{};
    std::array<std::uint64_t, 2U> normal_requested{};
    std::array<bool, 2U> temporary{};
    for (const auto& request : requests) {
        if (!is_valid(request.tier) ||
            (request.bucket != ChargeBucket::staging &&
             request.bucket != ChargeBucket::workspace) ||
            (request.admission != AdmissionKind::normal &&
             request.admission != AdmissionKind::migration_temporary) ||
            request.amount.value() == 0U) {
            return make_error(ErrorCode::invalid_argument, OperationId::budget_reserve);
        }
        const auto index = request.tier == PhysicalTier::gpu ? 0U : 1U;
        const auto sum =
            checked_add(requested[index], request.amount.value(), OperationId::budget_reserve);
        if (!sum) {
            return sum.error();
        }
        requested[index] = sum.value();
        if (request.admission == AdmissionKind::normal) {
            const auto normal_sum = checked_add(normal_requested[index], request.amount.value(),
                                                OperationId::budget_reserve);
            if (!normal_sum) {
                return normal_sum.error();
            }
            normal_requested[index] = normal_sum.value();
        }
        temporary[index] =
            temporary[index] || request.admission == AdmissionKind::migration_temporary;
    }

    const std::scoped_lock lock{mutex_};
    for (std::size_t index = 0U; index < requested.size(); ++index) {
        if (requested[index] == 0U) {
            continue;
        }
        const auto tier = index == 0U ? PhysicalTier::gpu : PhysicalTier::host;
        const auto& current = counters(tier);
        if (current.admission_blocked) {
            return ledger_error(ErrorCode::backend_contract_violation);
        }
        const auto charged_now = total(current);
        if (!charged_now) {
            return charged_now.error();
        }
        const auto projected =
            checked_add(charged_now.value(), requested[index], OperationId::budget_reserve);
        if (!projected) {
            return projected.error();
        }
        const auto& tier_budget = budget(tier);
        const auto normal_limit =
            checked_sub(tier_budget.hard_limit.value(), tier_budget.migration_reserve.value(),
                        OperationId::budget_reserve);
        if (!normal_limit) {
            return normal_limit.error();
        }
        const auto allowed =
            temporary[index] ? tier_budget.hard_limit.value() : normal_limit.value();
        if (projected.value() > allowed) {
            return oom_error(tier, requested[index]);
        }
        if (normal_requested[index] != 0U) {
            const auto projected_normal = checked_add(charged_now.value(), normal_requested[index],
                                                      OperationId::budget_reserve);
            if (!projected_normal) {
                return projected_normal.error();
            }
            if (projected_normal.value() > normal_limit.value()) {
                return oom_error(tier, normal_requested[index]);
            }
        }
    }
    if (next_token_id_ >
        std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(requests.size())) {
        return make_error(ErrorCode::arithmetic_overflow, OperationId::budget_reserve);
    }

    for (std::size_t index = 0U; index < requests.size(); ++index) {
        auto& current = counters(requests[index].tier);
        const auto added = add_to(current.reserved, requests[index].amount.value());
        if (!added) {
            return added.error();
        }
        output[index].owner = this;
        output[index].transaction = transaction;
        output[index].token_id = next_token_id_;
        ++next_token_id_;
        output[index].tier = requests[index].tier;
        output[index].bucket = requests[index].bucket;
        output[index].remaining = requests[index].amount;
        output[index].active = true;
        update_peak(current);
    }
    return {};
}

Result<void> BudgetLedger::materialize(ReservationToken& token, ByteSize exact_charge) noexcept {
    const std::scoped_lock lock{mutex_};
    if (token.owner != this || !token.active || token.token_id == 0U ||
        exact_charge.value() == 0U || exact_charge.value() > token.remaining.value()) {
        return ledger_error(ErrorCode::backend_contract_violation, exact_charge.value());
    }
    if (materialize_failure_countdown_ != 0U) {
        --materialize_failure_countdown_;
        if (materialize_failure_countdown_ == 0U) {
            return ledger_error(ErrorCode::backend_failure, exact_charge.value());
        }
    }
    auto& current = counters(token.tier);
    const auto* destination =
        token.bucket == ChargeBucket::staging ? &current.staging : &current.workspace;
    const auto sum = checked_add(*destination, exact_charge.value(), OperationId::budget_transfer);
    if (!sum || current.reserved < exact_charge.value()) {
        return ledger_error(ErrorCode::internal_invariant_violation, exact_charge.value());
    }
    current.reserved -= exact_charge.value();
    if (token.bucket == ChargeBucket::staging) {
        current.staging = sum.value();
    } else {
        current.workspace = sum.value();
    }
    token.remaining = ByteSize{token.remaining.value() - exact_charge.value()};
    update_peak(current);
    return {};
}

Result<void> BudgetLedger::release_reservation(ReservationToken& token) noexcept {
    const std::scoped_lock lock{mutex_};
    if (token.owner != this || !token.active) {
        return ledger_error(ErrorCode::stale_handle, token.token_id);
    }
    auto& current = counters(token.tier);
    if (current.reserved < token.remaining.value()) {
        return ledger_error(ErrorCode::internal_invariant_violation, token.remaining.value());
    }
    current.reserved -= token.remaining.value();
    token.remaining = ByteSize{};
    token.active = false;
    token.owner = nullptr;
    return {};
}

Result<void> BudgetLedger::commit_initial(PhysicalTier tier, ByteSize amount) noexcept {
    const std::scoped_lock lock{mutex_};
    auto& current = counters(tier);
    if (current.staging < amount.value()) {
        return ledger_error(ErrorCode::internal_invariant_violation, amount.value());
    }
    const auto sum = checked_add(current.committed, amount.value(), OperationId::budget_transfer);
    if (!sum) {
        return sum.error();
    }
    current.staging -= amount.value();
    current.committed = sum.value();
    return {};
}

Result<void> BudgetLedger::commit_destination_and_retire_source(PhysicalTier destination_tier,
                                                                ByteSize destination_amount,
                                                                PhysicalTier source_tier,
                                                                ByteSize source_amount) noexcept {
    const std::scoped_lock lock{mutex_};
    auto& destination = counters(destination_tier);
    auto& source = counters(source_tier);
    if (destination.staging < destination_amount.value() ||
        source.committed < source_amount.value()) {
        return ledger_error(ErrorCode::internal_invariant_violation);
    }
    const auto committed_sum = checked_add(destination.committed, destination_amount.value(),
                                           OperationId::budget_transfer);
    const auto debt_sum =
        checked_add(source.cleanup_debt, source_amount.value(), OperationId::budget_transfer);
    if (!committed_sum) {
        return committed_sum.error();
    }
    if (!debt_sum) {
        return debt_sum.error();
    }
    destination.staging -= destination_amount.value();
    destination.committed = committed_sum.value();
    source.committed -= source_amount.value();
    source.cleanup_debt = debt_sum.value();
    return {};
}

Result<void> BudgetLedger::release_materialized(PhysicalTier tier, ChargeBucket bucket,
                                                ByteSize amount) noexcept {
    const std::scoped_lock lock{mutex_};
    auto& current = counters(tier);
    auto& value = bucket == ChargeBucket::staging ? current.staging : current.workspace;
    return subtract_from(value, amount.value());
}

Result<void> BudgetLedger::shrink_materialized(PhysicalTier tier, ChargeBucket bucket,
                                               ByteSize old_amount, ByteSize new_amount) noexcept {
    if (new_amount.value() > old_amount.value()) {
        return ledger_error(ErrorCode::backend_contract_violation, new_amount.value());
    }
    const std::scoped_lock lock{mutex_};
    auto& current = counters(tier);
    auto& value = bucket == ChargeBucket::staging ? current.staging : current.workspace;
    if (value < old_amount.value()) {
        return ledger_error(ErrorCode::internal_invariant_violation, old_amount.value());
    }
    value -= old_amount.value() - new_amount.value();
    return {};
}

Result<void> BudgetLedger::exchange_compaction(PhysicalTier tier, ByteSize old_destination,
                                               ByteSize exact_destination) noexcept {
    const std::scoped_lock lock{mutex_};
    auto& current = counters(tier);
    if (exact_destination == ByteSize{} || exact_destination > old_destination ||
        current.staging < old_destination.value() ||
        current.workspace < exact_destination.value()) {
        return ledger_error(ErrorCode::internal_invariant_violation);
    }
    const auto difference = old_destination.value() - exact_destination.value();
    const auto sum = checked_add(current.workspace, difference, OperationId::budget_transfer);
    if (!sum) {
        return sum.error();
    }
    current.staging -= difference;
    current.workspace = sum.value();
    return {};
}

Result<void> BudgetLedger::move_materialized_to_debt(PhysicalTier tier, ChargeBucket bucket,
                                                     ByteSize amount) noexcept {
    const std::scoped_lock lock{mutex_};
    auto& current = counters(tier);
    auto& source = bucket == ChargeBucket::staging ? current.staging : current.workspace;
    if (source < amount.value()) {
        return ledger_error(ErrorCode::internal_invariant_violation, amount.value());
    }
    const auto sum =
        checked_add(current.cleanup_debt, amount.value(), OperationId::budget_transfer);
    if (!sum) {
        return sum.error();
    }
    source -= amount.value();
    current.cleanup_debt = sum.value();
    return {};
}

Result<void> BudgetLedger::retire_committed(PhysicalTier tier, ByteSize amount) noexcept {
    const std::scoped_lock lock{mutex_};
    auto& current = counters(tier);
    if (current.committed < amount.value()) {
        return ledger_error(ErrorCode::internal_invariant_violation, amount.value());
    }
    const auto sum =
        checked_add(current.cleanup_debt, amount.value(), OperationId::budget_transfer);
    if (!sum) {
        return sum.error();
    }
    current.committed -= amount.value();
    current.cleanup_debt = sum.value();
    return {};
}

Result<void> BudgetLedger::release_cleanup_debt(PhysicalTier tier, ByteSize amount) noexcept {
    const std::scoped_lock lock{mutex_};
    return subtract_from(counters(tier).cleanup_debt, amount.value());
}

Result<void> BudgetLedger::record_contract_debt(ReservationToken& token,
                                                ByteSize actual_charge) noexcept {
    const std::scoped_lock lock{mutex_};
    if (token.owner != this || !token.active) {
        return ledger_error(ErrorCode::stale_handle, token.token_id);
    }
    auto& current = counters(token.tier);
    if (current.reserved < token.remaining.value()) {
        return ledger_error(ErrorCode::internal_invariant_violation);
    }
    const auto sum =
        checked_add(current.cleanup_debt, actual_charge.value(), OperationId::budget_transfer);
    if (!sum) {
        return sum.error();
    }
    current.reserved -= token.remaining.value();
    current.cleanup_debt = sum.value();
    current.admission_blocked = true;
    token.remaining = ByteSize{};
    token.active = false;
    token.owner = nullptr;
    update_peak(current);
    return {};
}

Result<void> BudgetLedger::validate_migration_projection(PhysicalTier destination_tier,
                                                         ByteSize destination_amount,
                                                         PhysicalTier source_tier,
                                                         ByteSize source_amount) const noexcept {
    const std::scoped_lock lock{mutex_};
    std::array<std::uint64_t, 2U> projected{};
    for (std::size_t index = 0U; index < projected.size(); ++index) {
        const auto tier = index == 0U ? PhysicalTier::gpu : PhysicalTier::host;
        const auto current = total(counters(tier));
        if (!current) {
            return current.error();
        }
        projected[index] = current.value();
    }
    const auto destination_index = destination_tier == PhysicalTier::gpu ? 0U : 1U;
    const auto source_index = source_tier == PhysicalTier::gpu ? 0U : 1U;
    if (projected[source_index] < source_amount.value()) {
        return ledger_error(ErrorCode::internal_invariant_violation, source_amount.value());
    }
    projected[source_index] -= source_amount.value();
    const auto sum = checked_add(projected[destination_index], destination_amount.value(),
                                 OperationId::budget_reserve);
    if (!sum) {
        return sum.error();
    }
    projected[destination_index] = sum.value();

    for (std::size_t index = 0U; index < projected.size(); ++index) {
        const auto tier = index == 0U ? PhysicalTier::gpu : PhysicalTier::host;
        const auto& tier_budget = budget(tier);
        if (tier_budget.hard_limit.value() == 0U) {
            if (projected[index] != 0U) {
                return oom_error(tier, projected[index]);
            }
            continue;
        }
        const auto normal =
            checked_sub(tier_budget.hard_limit.value(), tier_budget.migration_reserve.value(),
                        OperationId::budget_reserve);
        if (!normal) {
            return normal.error();
        }
        if (projected[index] > normal.value()) {
            return oom_error(tier, projected[index]);
        }
    }
    return {};
}

TierUsage BudgetLedger::usage(PhysicalTier tier) const noexcept {
    const std::scoped_lock lock{mutex_};
    const auto& current = counters(tier);
    return TierUsage{ByteSize{current.committed},    ByteSize{current.reserved},
                     ByteSize{current.staging},      ByteSize{current.workspace},
                     ByteSize{current.cleanup_debt}, ByteSize{current.peak_charged}};
}

ByteSize BudgetLedger::charged(PhysicalTier tier) const noexcept {
    const std::scoped_lock lock{mutex_};
    const auto sum = total(counters(tier));
    return ByteSize{sum ? sum.value() : std::numeric_limits<std::uint64_t>::max()};
}

ByteSize BudgetLedger::unmaterialized_reservations(PhysicalTier tier) const noexcept {
    const std::scoped_lock lock{mutex_};
    return ByteSize{counters(tier).reserved};
}

bool BudgetLedger::admission_blocked(PhysicalTier tier) const noexcept {
    const std::scoped_lock lock{mutex_};
    return counters(tier).admission_blocked;
}

} // namespace vramz
