#pragma once

#include "vramz/checked.hpp"
#include "vramz/testing.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>

namespace vramz::test {

class Runner final {
  public:
    void begin(std::string_view name) {
        ++cases_;
        current_ = name;
        trace_active_ = false;
    }

    void check(bool condition, std::string_view expression, std::string_view file, int line) {
        if (!condition) {
            ++failures_;
            std::cerr << file << ':' << line << ": " << current_ << ": CHECK(" << expression
                      << ") failed\n";
            if (trace_active_) {
                std::cerr << "replay seed=" << trace_seed_ << " operation=" << trace_step_ << '\n';
            }
        }
    }

    void trace(std::uint64_t seed, std::uint64_t step) noexcept {
        trace_active_ = true;
        trace_seed_ = seed;
        trace_step_ = step;
    }

    [[nodiscard]] int finish() const {
        if (failures_ == 0U) {
            std::cout << cases_ << " deterministic cases passed\n";
            return 0;
        }
        std::cerr << failures_ << " checks failed across " << cases_ << " cases\n";
        return 1;
    }

  private:
    std::string_view current_{};
    std::uint64_t cases_{};
    std::uint64_t failures_{};
    bool trace_active_{};
    std::uint64_t trace_seed_{};
    std::uint64_t trace_step_{};
};

#define VRAMZ_CHECK(runner, expression)                                                            \
    (runner).check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

[[nodiscard]] inline MemoryBudgets test_budgets() noexcept {
    return MemoryBudgets{TierBudget{ByteSize{8192U}, ByteSize{6144U}, ByteSize{2048U}},
                         TierBudget{ByteSize{8192U}, ByteSize{6144U}, ByteSize{2048U}}};
}

[[nodiscard]] inline RuntimeConfig test_runtime_config() noexcept {
    RuntimeConfig config{};
    config.budgets = test_budgets();
    config.preferred_chunk_size = ByteSize{64U};
    config.async_errors.max_retained_errors = 8U;
    // Preserve the approved M1/M2 sequence, faults, and allocation behavior exactly.
    config.policy.mode = PolicyMode::disabled;
    return config;
}

[[nodiscard]] inline MockBackendConfig test_backend_config() noexcept {
    MockBackendConfig config{};
    config.exact_state_charges =
        std::array<ByteSize, 4U>{ByteSize{256U}, ByteSize{128U}, ByteSize{256U}, ByteSize{128U}};
    config.workspace_charge = ByteSize{64U};
    return config;
}

class CoreFixture final {
  public:
    explicit CoreFixture(RepresentationState initial)
        : backend(test_backend_config()), ledger(test_budgets()), errors(8U),
          coordinator(ledger, backend, errors) {
        constexpr auto initial_state = RepresentationState::gpu_raw;
        const auto reserved_address = backend.reserve_address_space(ByteSize{256U}, ByteSize{64U});
        if (!reserved_address) {
            setup_error = reserved_address.error();
            return;
        }
        address_space = reserved_address.value();
        ReservationRequest request{tier_of(initial_state), ChargeBucket::staging, ByteSize{256U},
                                   AdmissionKind::normal};
        const auto bound = backend.allocation_bound(initial_state, logical_size);
        if (!bound) {
            setup_error = bound.error();
            return;
        }
        request.amount = bound.value();
        const auto reserved =
            ledger.reserve(TransactionId{1U}, std::span<const ReservationRequest>{&request, 1U},
                           std::span<ReservationToken>{&token, 1U});
        if (!reserved) {
            setup_error = reserved.error();
            return;
        }
        const auto tag = initial_content_tag(BufferId{1U}, ChunkId{0U});
        const auto allocation = backend.allocate_representation(
            RepresentationAllocationRequest{initial_state, logical_size, tag, address_space.base});
        if (!allocation) {
            setup_error = allocation.error();
            return;
        }
        const auto adopted = backend.adopt_allocation(allocation.value());
        if (!adopted) {
            setup_error = adopted.error();
            return;
        }
        const auto exact = adopted.value();
        const auto materialized = ledger.materialize(token, exact.charge);
        if (!materialized) {
            setup_error = materialized.error();
            return;
        }
        const auto released = ledger.release_reservation(token);
        if (!released) {
            setup_error = released.error();
            return;
        }
        const auto committed = ledger.commit_initial(tier_of(initial_state), exact.charge);
        if (!committed) {
            setup_error = committed.error();
            return;
        }
        const DeviceAddress address = exact.address;
        Representation representation{initial_state, exact.id, exact.charge,
                                      tag,           address,  exact.metadata};
        chunk = std::make_unique<Chunk>(ChunkId{0U}, ByteOffset{}, logical_size, representation,
                                        address);
        if (initial != initial_state) {
            const auto migrated = coordinator.migrate(*chunk, initial);
            if (!migrated) {
                setup_error = migrated.error();
                chunk.reset();
            }
        }
    }

    [[nodiscard]] bool ready() const noexcept { return chunk != nullptr; }

    [[nodiscard]] bool conserved(PhysicalTier tier) const noexcept {
        const auto expected = checked_add(backend.owned_charge(tier).value(),
                                          ledger.unmaterialized_reservations(tier).value(),
                                          OperationId::budget_transfer);
        return expected && ledger.charged(tier).value() == expected.value();
    }

    MockBackend backend;
    BudgetLedger ledger;
    AsyncErrorChannel errors;
    TransactionCoordinator coordinator;
    std::unique_ptr<Chunk> chunk{};
    ReservationToken token{};
    AddressReservation address_space{};
    ByteSize logical_size{256U};
    Error setup_error{};
};

class CoreAccountingObserver final : public TransactionObserver {
  public:
    explicit CoreAccountingObserver(const CoreFixture& fixture) noexcept : fixture_(fixture) {}

    void on_phase(TransactionPhase phase, const ChunkSnapshot&) noexcept override {
        const auto index = static_cast<std::size_t>(phase);
        seen_[index] = true;
        gpu_[index] = fixture_.ledger.usage(PhysicalTier::gpu);
        host_[index] = fixture_.ledger.usage(PhysicalTier::host);
        conserved_ = conserved_ && fixture_.conserved(PhysicalTier::gpu) &&
                     fixture_.conserved(PhysicalTier::host);
    }

    [[nodiscard]] bool conserved() const noexcept { return conserved_; }
    [[nodiscard]] bool saw(TransactionPhase phase) const noexcept {
        return seen_[static_cast<std::size_t>(phase)];
    }
    [[nodiscard]] const TierUsage& at(TransactionPhase phase, PhysicalTier tier) const noexcept {
        const auto index = static_cast<std::size_t>(phase);
        return tier == PhysicalTier::gpu ? gpu_[index] : host_[index];
    }

  private:
    static constexpr std::size_t phase_count =
        static_cast<std::size_t>(TransactionPhase::poisoned) + 1U;
    const CoreFixture& fixture_;
    std::array<TierUsage, phase_count> gpu_{};
    std::array<TierUsage, phase_count> host_{};
    std::array<bool, phase_count> seen_{};
    bool conserved_{true};
};

class RecordingObserver final : public TransactionObserver {
  public:
    explicit RecordingObserver(RepresentationState expected_source) noexcept
        : expected_source_(expected_source) {}

    void on_phase(TransactionPhase phase, const ChunkSnapshot& snapshot) noexcept override {
        if (count_ < phases_.size()) {
            phases_[count_] = phase;
            ++count_;
        }
        if (phase < TransactionPhase::committed && phase != TransactionPhase::rolled_back &&
            phase != TransactionPhase::poisoned &&
            snapshot.authoritative.state != expected_source_) {
            source_changed_before_commit_ = true;
        }
    }

    [[nodiscard]] bool saw(TransactionPhase wanted) const noexcept {
        for (std::size_t index = 0U; index < count_; ++index) {
            if (phases_[index] == wanted) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool source_changed_before_commit() const noexcept {
        return source_changed_before_commit_;
    }

  private:
    RepresentationState expected_source_;
    std::array<TransactionPhase, 16U> phases_{};
    std::size_t count_{};
    bool source_changed_before_commit_{};
};

inline constexpr std::array<RepresentationState, 4U> all_states{
    RepresentationState::gpu_raw, RepresentationState::gpu_compressed,
    RepresentationState::host_raw, RepresentationState::host_compressed};

} // namespace vramz::test
