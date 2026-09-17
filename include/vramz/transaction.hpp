#pragma once

#include "vramz/async_error_channel.hpp"
#include "vramz/backend.hpp"
#include "vramz/budget_ledger.hpp"

#include <atomic>
#include <cstdint>

namespace vramz {

enum class TransactionPhase : std::uint8_t {
    source_validated,
    reserved,
    destination_materialized,
    workspace_materialized,
    charge_reconciled, // Emitted only after original identity and exact charge are corroborated.
    transferred,
    verified,
    commit_ready,
    committed,
    cleanup_complete,
    rolled_back,
    poisoned
};

struct TransitionPlan final {
    RepresentationState source{RepresentationState::gpu_raw};
    RepresentationState destination{RepresentationState::gpu_raw};
    PhysicalTier destination_tier{PhysicalTier::gpu};
    PhysicalTier workspace_tier{PhysicalTier::gpu};
    bool requires_transform{};
    bool requires_workspace{};
    bool crosses_tier{};
};

class TransactionObserver {
  public:
    virtual ~TransactionObserver() = default;
    virtual void on_phase(TransactionPhase phase, const ChunkSnapshot& snapshot) noexcept = 0;
};

// Optional advisory request constraints, enforced by the coordinator, never by policy mutation.
struct MigrationConstraints final {
    ResourceId expected_resource{};
    ContentTag expected_content{};
    std::optional<ByteSize> maximum_destination_charge{};
    std::optional<std::uint64_t> expected_access_revision{};
};

[[nodiscard]] Result<TransitionPlan> make_transition_plan(RepresentationState source,
                                                          RepresentationState destination) noexcept;

class TransactionCoordinator final {
  public:
    TransactionCoordinator(BudgetLedger& ledger, StorageBackend& backend,
                           AsyncErrorChannel& async_errors) noexcept;

    [[nodiscard]] Result<void> migrate(Chunk& chunk, RepresentationState destination,
                                       bool acquire_owned_intent = false,
                                       TransactionObserver* observer = nullptr,
                                       const MigrationConstraints* constraints = nullptr) noexcept;

  private:
    void notify(TransactionObserver* observer, TransactionPhase phase,
                const Chunk& chunk) const noexcept;
    void clear_transition(Chunk& chunk, TransactionId transaction) noexcept;
    void poison(Chunk& chunk, TransactionId transaction, Error error) noexcept;
    void rollback_resource(const BackendAllocation& allocation, ChargeBucket bucket,
                           Error causal_error, Chunk& chunk, TransactionId transaction,
                           bool poison_on_contract) noexcept;
    [[nodiscard]] bool remember_cleanup(Chunk& chunk, CleanupResource resource,
                                        TransactionId transaction) noexcept;
    void forget_cleanup(Chunk& chunk, ResourceId resource) noexcept;

    BudgetLedger& ledger_;
    StorageBackend& backend_;
    AsyncErrorChannel& async_errors_;
    std::atomic<std::uint64_t> next_transaction_{1U};
};

} // namespace vramz
