#include "../test_support.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>

using namespace vramz;

namespace {

class DeterministicGenerator final {
  public:
    explicit DeterministicGenerator(std::uint64_t seed) noexcept : state_(seed) {}

    [[nodiscard]] std::uint64_t next() noexcept {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return state_;
    }

  private:
    std::uint64_t state_;
};

} // namespace

int main() {
    test::Runner runner;
    constexpr std::array<std::uint64_t, 4U> seeds{1U, 0xC0FFEEU, 0x123456789ABCDEF0ULL,
                                                  0xFFFFFFFFFFFFFFC5ULL};

    runner.begin("deterministic migration and failure sequences preserve global invariants");
    for (const auto seed : seeds) {
        std::cout << "seed=" << seed << '\n';
        DeterministicGenerator generator{seed};
        test::CoreFixture fixture{
            test::all_states[static_cast<std::size_t>(generator.next() % 4U)]};
        VRAMZ_CHECK(runner, fixture.ready());
        ContentTag expected = fixture.chunk->snapshot().authoritative.content;
        for (std::uint64_t operation = 0U; operation < 500U; ++operation) {
            const auto current = fixture.chunk->snapshot().authoritative.state;
            auto destination = test::all_states[static_cast<std::size_t>(generator.next() % 4U)];
            if (destination == current) {
                destination = test::all_states[(static_cast<std::size_t>(destination) + 1U) % 4U];
            }
            const bool inject = (generator.next() % 13U) == 0U;
            if (inject) {
                fixture.backend.inject_failure((generator.next() & 1U) == 0U
                                                   ? FaultPoint::transfer
                                                   : FaultPoint::verification);
            }
            test::CoreAccountingObserver observer{fixture};
            const auto migrated =
                fixture.coordinator.migrate(*fixture.chunk, destination, false, &observer);
            VRAMZ_CHECK(runner, observer.conserved());
            if (migrated) {
                VRAMZ_CHECK(runner, observer.saw(TransactionPhase::charge_reconciled));
            }
            fixture.backend.clear_failures();
            const auto snapshot = fixture.chunk->snapshot();
            VRAMZ_CHECK(runner, snapshot.lifecycle == LifecycleState::live);
            VRAMZ_CHECK(runner, snapshot.authoritative.content == expected);
            if (migrated) {
                VRAMZ_CHECK(runner, snapshot.authoritative.state == destination);
            } else {
                VRAMZ_CHECK(runner, snapshot.authoritative.state == current);
            }
            VRAMZ_CHECK(runner, fixture.backend.owns(snapshot.authoritative.resource));
            VRAMZ_CHECK(runner, fixture.ledger.usage(PhysicalTier::gpu).reserved == ByteSize{});
            VRAMZ_CHECK(runner, fixture.ledger.usage(PhysicalTier::host).reserved == ByteSize{});
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::gpu));
            VRAMZ_CHECK(runner, fixture.conserved(PhysicalTier::host));
        }
    }

    runner.begin("independent ledger model conserves reservations and resources");
    for (const auto seed : seeds) {
        DeterministicGenerator generator{seed};
        BudgetLedger ledger{test::test_budgets()};
        std::array<std::uint64_t, 2U> modeled_resources{};
        std::array<std::uint64_t, 2U> modeled_reservations{};
        for (std::uint64_t operation = 0U; operation < 500U; ++operation) {
            const auto tier =
                (generator.next() & 1U) == 0U ? PhysicalTier::gpu : PhysicalTier::host;
            const auto tier_index = tier == PhysicalTier::gpu ? 0U : 1U;
            const ByteSize requested{1U + generator.next() % 32U};
            ReservationRequest request{tier, ChargeBucket::staging, requested,
                                       AdmissionKind::migration_temporary};
            ReservationToken token{};
            const auto reserved = ledger.reserve(TransactionId{operation + 1U},
                                                 std::span{&request, 1U}, std::span{&token, 1U});
            VRAMZ_CHECK(runner, reserved);
            modeled_reservations[tier_index] += requested.value();
            VRAMZ_CHECK(runner, ledger.unmaterialized_reservations(tier).value() ==
                                    modeled_reservations[tier_index]);

            const ByteSize materialized{1U + generator.next() % requested.value()};
            VRAMZ_CHECK(runner, ledger.materialize(token, materialized));
            modeled_reservations[tier_index] -= materialized.value();
            modeled_resources[tier_index] += materialized.value();
            VRAMZ_CHECK(runner, ledger.release_reservation(token));
            modeled_reservations[tier_index] = 0U;

            if ((generator.next() & 1U) == 0U) {
                VRAMZ_CHECK(runner,
                            ledger.release_materialized(tier, ChargeBucket::staging, materialized));
            } else {
                VRAMZ_CHECK(runner, ledger.move_materialized_to_debt(tier, ChargeBucket::staging,
                                                                     materialized));
                VRAMZ_CHECK(runner, ledger.release_cleanup_debt(tier, materialized));
            }
            modeled_resources[tier_index] -= materialized.value();
            for (std::size_t index = 0U; index < modeled_resources.size(); ++index) {
                const auto checked_tier = index == 0U ? PhysicalTier::gpu : PhysicalTier::host;
                VRAMZ_CHECK(runner, ledger.charged(checked_tier).value() ==
                                        modeled_resources[index] + modeled_reservations[index]);
            }
        }
    }

    runner.begin("generated runtime lifecycles conserve ownership after every operation");
    for (const auto seed : seeds) {
        DeterministicGenerator generator{seed};
        auto runtime_result = Runtime::create(test::test_runtime_config());
        VRAMZ_CHECK(runner, runtime_result);
        auto runtime = std::move(runtime_result).value();
        auto buffer_result = runtime.allocate(ByteSize{64U});
        VRAMZ_CHECK(runner, buffer_result);
        std::optional<Buffer> buffer{};
        buffer.emplace(std::move(buffer_result).value());

        for (std::uint64_t operation = 0U; operation < 250U; ++operation) {
            if ((operation % 47U) == 46U) {
                const auto before = testing::chunk_snapshot(*buffer, 0U);
                VRAMZ_CHECK(runner, before);
                auto destination = test::all_states
                    [(static_cast<std::size_t>(before.value().authoritative.state) + 1U) % 4U];
                testing::inject_failure(runtime, FaultPoint::post_commit_cleanup);
                const auto migrated = testing::migrate(*buffer, 0U, destination);
                VRAMZ_CHECK(runner, !migrated);
                const auto after = testing::chunk_snapshot(*buffer, 0U);
                VRAMZ_CHECK(runner, after && after.value().authoritative.state == destination);
                VRAMZ_CHECK(runner, buffer->close());
                buffer.reset();
                auto replacement = runtime.allocate(ByteSize{64U});
                VRAMZ_CHECK(runner, replacement);
                buffer.emplace(std::move(replacement).value());
            } else if ((operation % 11U) == 0U) {
                const auto before = testing::chunk_snapshot(*buffer, 0U);
                auto lease = buffer->acquire(MemoryRange{ByteOffset{}, ByteSize{64U}},
                                             AcquireOptions{AccessMode::read_write});
                VRAMZ_CHECK(runner, before && lease);
                if (lease) {
                    VRAMZ_CHECK(runner, lease.value().close());
                    const auto after = testing::chunk_snapshot(*buffer, 0U);
                    VRAMZ_CHECK(runner,
                                after && after.value().authoritative.content.generation ==
                                             before.value().authoritative.content.generation + 1U);
                }
            } else {
                const auto before = testing::chunk_snapshot(*buffer, 0U);
                VRAMZ_CHECK(runner, before);
                auto destination =
                    test::all_states[static_cast<std::size_t>(generator.next() % 4U)];
                if (destination == before.value().authoritative.state) {
                    destination =
                        test::all_states[(static_cast<std::size_t>(destination) + 1U) % 4U];
                }
                const bool fail_transfer = (operation % 17U) == 0U;
                if (fail_transfer) {
                    testing::inject_failure(runtime, FaultPoint::transfer);
                }
                const auto migrated = testing::migrate(*buffer, 0U, destination);
                const auto after = testing::chunk_snapshot(*buffer, 0U);
                VRAMZ_CHECK(runner, after);
                if (fail_transfer) {
                    VRAMZ_CHECK(runner, !migrated && after.value().authoritative.resource ==
                                                         before.value().authoritative.resource);
                } else {
                    VRAMZ_CHECK(runner,
                                migrated && after.value().authoritative.state == destination);
                }
            }

            const auto stats = runtime.stats();
            VRAMZ_CHECK(runner, stats.gpu.reserved == ByteSize{} &&
                                    stats.gpu.staging == ByteSize{} &&
                                    stats.gpu.workspace == ByteSize{});
            VRAMZ_CHECK(runner, stats.host.reserved == ByteSize{} &&
                                    stats.host.staging == ByteSize{} &&
                                    stats.host.workspace == ByteSize{});
            VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
            VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
        }
        VRAMZ_CHECK(runner, buffer->close());
        buffer.reset();
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
        VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
        VRAMZ_CHECK(runner, runtime.shutdown());
    }

    runner.begin("adversarial arithmetic boundaries never wrap");
    constexpr std::array<std::uint64_t, 5U> values{0U, 1U, 2U,
                                                   std::numeric_limits<std::uint64_t>::max() - 1U,
                                                   std::numeric_limits<std::uint64_t>::max()};
    for (const auto left : values) {
        for (const auto right : values) {
            const auto sum = checked_add(left, right, OperationId::allocate);
            const bool representable = right <= std::numeric_limits<std::uint64_t>::max() - left;
            VRAMZ_CHECK(runner, static_cast<bool>(sum) == representable);
            const auto product = checked_mul(left, right, OperationId::allocate);
            const bool product_representable =
                left == 0U || right <= std::numeric_limits<std::uint64_t>::max() / left;
            VRAMZ_CHECK(runner, static_cast<bool>(product) == product_representable);
        }
    }

    return runner.finish();
}
