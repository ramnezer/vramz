#pragma once

#include "vramz/config.hpp"
#include "vramz/stats.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>

namespace vramz {

namespace testing {
class RuntimeAccess;
}

class BudgetLedger;

enum class ChargeBucket : std::uint8_t { staging, workspace };
enum class AdmissionKind : std::uint8_t { normal, migration_temporary };

struct ReservationRequest final {
    PhysicalTier tier{PhysicalTier::gpu};
    ChargeBucket bucket{ChargeBucket::staging};
    ByteSize amount{};
    AdmissionKind admission{AdmissionKind::normal};
};

struct ReservationToken final {
    ReservationToken() noexcept = default;
    ReservationToken(const ReservationToken&) = delete;
    ReservationToken& operator=(const ReservationToken&) = delete;
    ReservationToken(ReservationToken&& other) noexcept;
    ReservationToken& operator=(ReservationToken&& other) noexcept = delete;

  private:
    friend class BudgetLedger;
    const BudgetLedger* owner{};
    TransactionId transaction{};
    std::uint64_t token_id{};
    PhysicalTier tier{PhysicalTier::gpu};
    ChargeBucket bucket{ChargeBucket::staging};
    ByteSize remaining{};
    bool active{};
};

class BudgetLedger final {
  public:
    explicit BudgetLedger(MemoryBudgets budgets) noexcept;

    [[nodiscard]] Result<void> reserve(TransactionId transaction,
                                       std::span<const ReservationRequest> requests,
                                       std::span<ReservationToken> output) noexcept;
    [[nodiscard]] Result<void> materialize(ReservationToken& token, ByteSize exact_charge) noexcept;
    [[nodiscard]] Result<void> release_reservation(ReservationToken& token) noexcept;
    [[nodiscard]] Result<void> commit_initial(PhysicalTier tier, ByteSize amount) noexcept;
    [[nodiscard]] Result<void>
    commit_destination_and_retire_source(PhysicalTier destination_tier, ByteSize destination_amount,
                                         PhysicalTier source_tier, ByteSize source_amount) noexcept;
    [[nodiscard]] Result<void> release_materialized(PhysicalTier tier, ChargeBucket bucket,
                                                    ByteSize amount) noexcept;
    [[nodiscard]] Result<void> shrink_materialized(PhysicalTier tier, ChargeBucket bucket,
                                                   ByteSize old_amount,
                                                   ByteSize new_amount) noexcept;
    [[nodiscard]] Result<void> exchange_compaction(PhysicalTier tier, ByteSize old_destination,
                                                   ByteSize exact_destination) noexcept;
    [[nodiscard]] Result<void> move_materialized_to_debt(PhysicalTier tier, ChargeBucket bucket,
                                                         ByteSize amount) noexcept;
    [[nodiscard]] Result<void> retire_committed(PhysicalTier tier, ByteSize amount) noexcept;
    [[nodiscard]] Result<void> release_cleanup_debt(PhysicalTier tier, ByteSize amount) noexcept;
    // Requires an adopted, corroborated exact resource charge, never an admission bound.
    [[nodiscard]] Result<void> record_contract_debt(ReservationToken& token,
                                                    ByteSize actual_charge) noexcept;
    [[nodiscard]] Result<void> validate_migration_projection(PhysicalTier destination_tier,
                                                             ByteSize destination_amount,
                                                             PhysicalTier source_tier,
                                                             ByteSize source_amount) const noexcept;

    [[nodiscard]] TierUsage usage(PhysicalTier tier) const noexcept;
    [[nodiscard]] ByteSize charged(PhysicalTier tier) const noexcept;
    [[nodiscard]] ByteSize unmaterialized_reservations(PhysicalTier tier) const noexcept;
    [[nodiscard]] bool admission_blocked(PhysicalTier tier) const noexcept;

  private:
    friend class testing::RuntimeAccess;
    struct Counters final {
        std::uint64_t committed{};
        std::uint64_t reserved{};
        std::uint64_t staging{};
        std::uint64_t workspace{};
        std::uint64_t cleanup_debt{};
        std::uint64_t peak_charged{};
        bool admission_blocked{};
    };

    [[nodiscard]] Counters& counters(PhysicalTier tier) noexcept;
    [[nodiscard]] const Counters& counters(PhysicalTier tier) const noexcept;
    [[nodiscard]] const TierBudget& budget(PhysicalTier tier) const noexcept;
    [[nodiscard]] static Result<std::uint64_t> total(const Counters& counters) noexcept;
    [[nodiscard]] static Result<void> add_to(std::uint64_t& value, std::uint64_t amount) noexcept;
    [[nodiscard]] static Result<void> subtract_from(std::uint64_t& value,
                                                    std::uint64_t amount) noexcept;
    static void update_peak(Counters& counters) noexcept;

    MemoryBudgets budgets_{};
    mutable std::mutex mutex_{};
    Counters gpu_{};
    Counters host_{};
    std::uint64_t next_token_id_{1U};
    // Deterministic pre-mutation fault injection, accessible only through testing::RuntimeAccess.
    std::uint64_t materialize_failure_countdown_{};
};

} // namespace vramz
