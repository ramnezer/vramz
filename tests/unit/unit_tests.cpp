#include "../test_support.hpp"
#include "vramz/saturating.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>

using namespace vramz;

namespace {
std::atomic_bool fail_allocations{false};
}

void* operator new(std::size_t size) {
    if (fail_allocations.load(std::memory_order_relaxed)) {
        throw std::bad_alloc{};
    }
    void* const memory = std::malloc(size == 0U ? 1U : size);
    if (memory == nullptr) {
        throw std::bad_alloc{};
    }
    return memory;
}

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

int main() {
    test::Runner runner;

    runner.begin("strong types are distinct and lightweight");
    static_assert(!std::is_same_v<ByteSize, ByteOffset>);
    static_assert(sizeof(ByteSize) == sizeof(std::uint64_t));
    VRAMZ_CHECK(runner, ByteSize{3U} < ByteSize{4U});

    runner.begin("RAII owners are movable only when replacement cannot discard ownership");
    static_assert(std::is_move_constructible_v<Runtime>);
    static_assert(!std::is_move_assignable_v<Runtime>);
    static_assert(std::is_move_constructible_v<Buffer>);
    static_assert(!std::is_move_assignable_v<Buffer>);
    static_assert(std::is_move_constructible_v<Lease>);
    static_assert(!std::is_move_assignable_v<Lease>);
    static_assert(std::is_move_constructible_v<PendingLease>);
    static_assert(!std::is_move_assignable_v<PendingLease>);
    static_assert(std::is_move_constructible_v<ReservationToken>);
    static_assert(!std::is_move_assignable_v<ReservationToken>);
    static_assert(!std::is_copy_constructible_v<Runtime>);
    static_assert(!std::is_copy_assignable_v<Runtime>);
    static_assert(!std::is_copy_constructible_v<Buffer>);
    static_assert(!std::is_copy_assignable_v<Buffer>);
    static_assert(!std::is_copy_constructible_v<Lease>);
    static_assert(!std::is_copy_assignable_v<Lease>);
    static_assert(!std::is_copy_constructible_v<PendingLease>);
    static_assert(!std::is_copy_assignable_v<PendingLease>);
    static_assert(!std::is_copy_constructible_v<ReservationToken>);
    static_assert(!std::is_copy_assignable_v<ReservationToken>);
    static_assert(!std::is_move_constructible_v<Chunk>);
    static_assert(!std::is_move_constructible_v<BudgetLedger>);
    static_assert(!std::is_move_constructible_v<MockBackend>);
    static_assert(!std::is_move_constructible_v<AsyncErrorChannel>);
    static_assert(!std::is_move_constructible_v<TransactionCoordinator>);
    static_assert(!std::is_copy_constructible_v<Operation>);
    static_assert(!std::is_copy_assignable_v<Operation>);
    static_assert(std::is_move_assignable_v<Operation>);
    static_assert(std::is_trivially_destructible_v<Operation>);
    VRAMZ_CHECK(runner, true);

    runner.begin("invalid enum values and illegal no-op transitions are rejected");
    const auto invalid_state = std::bit_cast<RepresentationState>(static_cast<std::uint8_t>(255U));
    VRAMZ_CHECK(runner,
                !make_transition_plan(RepresentationState::gpu_raw, RepresentationState::gpu_raw));
    VRAMZ_CHECK(runner, !make_transition_plan(RepresentationState::gpu_raw, invalid_state));
    MockBackend enum_backend{test::test_backend_config()};
    VRAMZ_CHECK(runner, !enum_backend.allocation_bound(invalid_state, ByteSize{64U}));
    BudgetLedger enum_ledger{test::test_budgets()};
    ReservationRequest enum_request{PhysicalTier::gpu, ChargeBucket::staging, ByteSize{64U},
                                    AdmissionKind::migration_temporary};
    ReservationToken enum_token{};
    VRAMZ_CHECK(runner, !enum_ledger.reserve(TransactionId{}, std::span{&enum_request, 1U},
                                             std::span{&enum_token, 1U}));

    runner.begin("checked add sub and multiply boundaries");
    VRAMZ_CHECK(runner, checked_add(2U, 3U, OperationId::allocate).value() == 5U);
    VRAMZ_CHECK(runner,
                !checked_add(std::numeric_limits<std::uint64_t>::max(), 1U, OperationId::allocate));
    VRAMZ_CHECK(runner, !checked_sub(0U, 1U, OperationId::allocate));
    VRAMZ_CHECK(runner,
                !checked_mul(std::numeric_limits<std::uint64_t>::max(), 2U, OperationId::allocate));
    VRAMZ_CHECK(
        runner,
        checked_mul(0U, std::numeric_limits<std::uint64_t>::max(), OperationId::allocate).value() ==
            0U);

    runner.begin("checked alignment and range end");
    VRAMZ_CHECK(runner, checked_align_up(65U, 48U, OperationId::allocate).value() == 96U);
    VRAMZ_CHECK(runner, !checked_align_up(1U, 0U, OperationId::allocate));
    VRAMZ_CHECK(runner, !checked_align_up(std::numeric_limits<std::uint64_t>::max(), 2U,
                                          OperationId::allocate));
    VRAMZ_CHECK(runner,
                checked_range_end(MemoryRange{ByteOffset{7U}, ByteSize{5U}}, OperationId::acquire)
                        .value() == ByteOffset{12U});
    VRAMZ_CHECK(runner, !checked_range_end(
                            MemoryRange{ByteOffset{std::numeric_limits<std::uint64_t>::max()},
                                        ByteSize{1U}},
                            OperationId::acquire));

    runner.begin("allocation safe Error and Result");
    static_assert(std::is_trivially_copyable_v<Error>);
    static_assert(std::is_trivially_copyable_v<Result<void>>);
    const Error original = make_error(ErrorCode::out_of_host_memory, OperationId::allocate, 9U);
    const Result<void> void_result{original};
    const Result<std::uint64_t> value_result{original};
    VRAMZ_CHECK(runner, !void_result && void_result.error() == original);
    VRAMZ_CHECK(runner, !value_result && value_result.error() == original);

    runner.begin("nested OOM reporting performs no secondary allocation");
    fail_allocations.store(true, std::memory_order_relaxed);
    const Error nested_error =
        make_error(ErrorCode::out_of_host_memory, OperationId::runtime_create, 17U);
    const Result<void> nested_void{nested_error};
    const Result<std::uint64_t> nested_value{nested_error};
    std::array<char, 64U> nested_text{};
    const auto nested_format = format_error(nested_error, nested_text);
    const auto runtime_oom = Runtime::create(test::test_runtime_config());
    const std::array<std::byte, 4U> codec_input{std::byte{1U}, std::byte{2U}, std::byte{3U},
                                                std::byte{4U}};
    std::array<std::byte, 4U> codec_output{};
    const auto codec_oom = testing::codec_round_trip(codec_input, codec_output);
    fail_allocations.store(false, std::memory_order_relaxed);
    VRAMZ_CHECK(runner, !nested_void && nested_void.error() == nested_error);
    VRAMZ_CHECK(runner, !nested_value && nested_value.error() == nested_error);
    VRAMZ_CHECK(runner, nested_format.written > 0U);
    VRAMZ_CHECK(runner, !runtime_oom && runtime_oom.error().code == ErrorCode::out_of_host_memory);
    VRAMZ_CHECK(runner, !codec_oom && codec_oom.error().code == ErrorCode::out_of_host_memory);

    runner.begin("bounded error formatting");
    std::array<char, 128U> full{};
    std::array<char, 4U> tiny{};
    std::span<char> empty{};
    const auto full_result = format_error(original, full);
    const auto tiny_result = format_error(original, tiny);
    const auto empty_result = format_error(original, empty);
    VRAMZ_CHECK(runner, full_result.written > 0U && !full_result.truncated);
    VRAMZ_CHECK(runner, tiny_result.truncated);
    VRAMZ_CHECK(runner, empty_result.truncated);
    VRAMZ_CHECK(runner, original.code == ErrorCode::out_of_host_memory);

    runner.begin("configuration validation");
    auto config = test::test_runtime_config();
    const auto capabilities = test::test_backend_config().capabilities;
    VRAMZ_CHECK(runner, validate_config(config, capabilities));
    config.budgets.gpu.soft_target = ByteSize{9000U};
    VRAMZ_CHECK(runner, !validate_config(config, capabilities));
    config = test::test_runtime_config();
    config.budgets.host.migration_reserve = ByteSize{9000U};
    VRAMZ_CHECK(runner, !validate_config(config, capabilities));
    config = test::test_runtime_config();
    config.preferred_chunk_size = ByteSize{};
    VRAMZ_CHECK(runner, !validate_config(config, capabilities));
    config = test::test_runtime_config();
    config.async_errors.max_retained_errors = 0U;
    VRAMZ_CHECK(runner, !validate_config(config, capabilities));
    config.async_errors.max_retained_errors = max_async_error_capacity + 1U;
    VRAMZ_CHECK(runner, !validate_config(config, capabilities));
    config = test::test_runtime_config();
    config.preferred_chunk_size = ByteSize{65U};
    VRAMZ_CHECK(runner, !validate_config(config, capabilities));
    config = test::test_runtime_config();
    config.budgets.gpu.hard_limit = ByteSize{};
    config.budgets.gpu.soft_target = ByteSize{};
    config.budgets.gpu.migration_reserve = ByteSize{};
    VRAMZ_CHECK(runner, !validate_config(config, capabilities));
    config = test::test_runtime_config();
    auto invalid_capabilities = capabilities;
    invalid_capabilities.mapping_granularity = ByteSize{};
    VRAMZ_CHECK(runner, !validate_config(config, invalid_capabilities));
    invalid_capabilities = capabilities;
    invalid_capabilities.host_tier = false;
    VRAMZ_CHECK(runner, !validate_config(config, invalid_capabilities));
    invalid_capabilities = capabilities;
    invalid_capabilities.stable_device_address = false;
    VRAMZ_CHECK(runner, !validate_config(config, invalid_capabilities));
    config = test::test_runtime_config();
    config.required_capabilities.asynchronous_operations = true;
    VRAMZ_CHECK(runner, !validate_config(config, capabilities));
    config = test::test_runtime_config();
    config.preferred_chunk_size.reset();
    invalid_capabilities = capabilities;
    invalid_capabilities.reservation_granularity =
        ByteSize{std::numeric_limits<std::uint64_t>::max()};
    invalid_capabilities.mapping_granularity =
        ByteSize{std::numeric_limits<std::uint64_t>::max() - 1U};
    VRAMZ_CHECK(runner, !validate_config(config, invalid_capabilities));

    runner.begin("reservation materialization transfers without double charge");
    BudgetLedger ledger{test::test_budgets()};
    std::array<ReservationRequest, 2U> requests{
        ReservationRequest{PhysicalTier::gpu, ChargeBucket::staging, ByteSize{300U},
                           AdmissionKind::migration_temporary},
        ReservationRequest{PhysicalTier::gpu, ChargeBucket::workspace, ByteSize{100U},
                           AdmissionKind::migration_temporary}};
    std::array<ReservationToken, 2U> tokens{};
    VRAMZ_CHECK(runner, ledger.reserve(TransactionId{1U}, requests, tokens));
    VRAMZ_CHECK(runner, ledger.charged(PhysicalTier::gpu) == ByteSize{400U});
    VRAMZ_CHECK(runner, ledger.materialize(tokens[0], ByteSize{240U}));
    auto usage = ledger.usage(PhysicalTier::gpu);
    VRAMZ_CHECK(runner, usage.reserved == ByteSize{160U});
    VRAMZ_CHECK(runner, usage.staging == ByteSize{240U});
    VRAMZ_CHECK(runner, ledger.charged(PhysicalTier::gpu) == ByteSize{400U});
    VRAMZ_CHECK(runner, ledger.materialize(tokens[1], ByteSize{80U}));
    usage = ledger.usage(PhysicalTier::gpu);
    VRAMZ_CHECK(runner, usage.reserved == ByteSize{80U});
    VRAMZ_CHECK(runner, usage.workspace == ByteSize{80U});
    VRAMZ_CHECK(runner, ledger.release_reservation(tokens[0]));
    VRAMZ_CHECK(runner, ledger.release_reservation(tokens[1]));
    VRAMZ_CHECK(runner, ledger.commit_initial(PhysicalTier::gpu, ByteSize{240U}));
    VRAMZ_CHECK(runner, ledger.move_materialized_to_debt(PhysicalTier::gpu, ChargeBucket::workspace,
                                                         ByteSize{80U}));
    usage = ledger.usage(PhysicalTier::gpu);
    VRAMZ_CHECK(runner, usage.committed == ByteSize{240U});
    VRAMZ_CHECK(runner, usage.cleanup_debt == ByteSize{80U});

    runner.begin("cross tier reservation is atomic");
    MemoryBudgets small{TierBudget{ByteSize{100U}, ByteSize{100U}, ByteSize{}},
                        TierBudget{ByteSize{50U}, ByteSize{50U}, ByteSize{}}};
    BudgetLedger atomic_ledger{small};
    std::array<ReservationRequest, 2U> cross{
        ReservationRequest{PhysicalTier::gpu, ChargeBucket::staging, ByteSize{40U},
                           AdmissionKind::migration_temporary},
        ReservationRequest{PhysicalTier::host, ChargeBucket::workspace, ByteSize{60U},
                           AdmissionKind::migration_temporary}};
    std::array<ReservationToken, 2U> cross_tokens{};
    VRAMZ_CHECK(runner, !atomic_ledger.reserve(TransactionId{2U}, cross, cross_tokens));
    VRAMZ_CHECK(runner, atomic_ledger.charged(PhysicalTier::gpu) == ByteSize{});
    VRAMZ_CHECK(runner, atomic_ledger.charged(PhysicalTier::host) == ByteSize{});

    runner.begin("migration reserve protects normal admission");
    MemoryBudgets reserve_budget{TierBudget{ByteSize{100U}, ByteSize{80U}, ByteSize{20U}},
                                 TierBudget{ByteSize{100U}, ByteSize{100U}, ByteSize{}}};
    BudgetLedger reserve_ledger{reserve_budget};
    ReservationRequest normal{PhysicalTier::gpu, ChargeBucket::staging, ByteSize{81U},
                              AdmissionKind::normal};
    ReservationToken normal_token{};
    VRAMZ_CHECK(runner, !reserve_ledger.reserve(TransactionId{3U}, std::span{&normal, 1U},
                                                std::span{&normal_token, 1U}));
    normal.admission = AdmissionKind::migration_temporary;
    VRAMZ_CHECK(runner, reserve_ledger.reserve(TransactionId{4U}, std::span{&normal, 1U},
                                               std::span{&normal_token, 1U}));
    VRAMZ_CHECK(runner, reserve_ledger.release_reservation(normal_token));

    runner.begin("migration reserve exhaustion fails without a partial charge");
    ReservationRequest exhausted{PhysicalTier::gpu, ChargeBucket::staging, ByteSize{101U},
                                 AdmissionKind::migration_temporary};
    ReservationToken exhausted_token{};
    VRAMZ_CHECK(runner, !reserve_ledger.reserve(TransactionId{5U}, std::span{&exhausted, 1U},
                                                std::span{&exhausted_token, 1U}));
    VRAMZ_CHECK(runner, reserve_ledger.charged(PhysicalTier::gpu) == ByteSize{});

    runner.begin("zero migration reserve permits normal admission through hard limit");
    MemoryBudgets zero_reserve{TierBudget{ByteSize{100U}, ByteSize{90U}, ByteSize{}},
                               TierBudget{ByteSize{100U}, ByteSize{90U}, ByteSize{}}};
    BudgetLedger zero_reserve_ledger{zero_reserve};
    ReservationRequest full_request{PhysicalTier::gpu, ChargeBucket::staging, ByteSize{100U},
                                    AdmissionKind::normal};
    ReservationToken full_token{};
    VRAMZ_CHECK(runner, zero_reserve_ledger.reserve(TransactionId{5U}, std::span{&full_request, 1U},
                                                    std::span{&full_token, 1U}));
    VRAMZ_CHECK(runner, zero_reserve_ledger.release_reservation(full_token));
    VRAMZ_CHECK(runner, !zero_reserve_ledger.release_reservation(full_token));

    runner.begin("reservation tokens cannot be overwritten or used by another ledger");
    BudgetLedger token_ledger{test::test_budgets()};
    BudgetLedger foreign_ledger{test::test_budgets()};
    ReservationRequest token_request{PhysicalTier::gpu, ChargeBucket::staging, ByteSize{64U},
                                     AdmissionKind::migration_temporary};
    ReservationToken guarded_token{};
    VRAMZ_CHECK(runner, token_ledger.reserve(TransactionId{6U}, std::span{&token_request, 1U},
                                             std::span{&guarded_token, 1U}));
    VRAMZ_CHECK(runner, !token_ledger.reserve(TransactionId{7U}, std::span{&token_request, 1U},
                                              std::span{&guarded_token, 1U}));
    VRAMZ_CHECK(runner, token_ledger.charged(PhysicalTier::gpu) == ByteSize{64U});
    VRAMZ_CHECK(runner, !foreign_ledger.materialize(guarded_token, ByteSize{64U}));
    VRAMZ_CHECK(runner, foreign_ledger.charged(PhysicalTier::gpu) == ByteSize{});
    VRAMZ_CHECK(runner, token_ledger.release_reservation(guarded_token));

    runner.begin("bounded async channel preserves first and deterministic overflow");
    AsyncErrorChannel channel{2U};
    const auto first = make_error(ErrorCode::backend_failure, OperationId::release, 1U);
    const auto second = make_error(ErrorCode::integrity_failure, OperationId::verify, 2U);
    const auto third = make_error(ErrorCode::timeout, OperationId::migrate, 3U);
    channel.push(first);
    channel.push(first);
    channel.push(second);
    channel.push(third);
    channel.record_diagnostic_failure();
    const auto async_stats = channel.stats();
    VRAMZ_CHECK(runner, async_stats.capacity == 2U && async_stats.retained == 2U);
    VRAMZ_CHECK(runner, async_stats.produced == 4U && async_stats.coalesced == 1U);
    VRAMZ_CHECK(runner, async_stats.dropped == 1U);
    VRAMZ_CHECK(runner, async_stats.diagnostic_text_failures == 1U);
    const auto sticky_error = channel.first().value_or(AsyncErrorRecord{});
    const auto polled_error = channel.poll().value_or(AsyncErrorRecord{});
    VRAMZ_CHECK(runner, sticky_error.error == first);
    VRAMZ_CHECK(runner, polled_error.repeat_count == 2U);
    std::uint64_t saturated = std::numeric_limits<std::uint64_t>::max();
    saturating_increment(saturated);
    VRAMZ_CHECK(runner, saturated == std::numeric_limits<std::uint64_t>::max());
    VRAMZ_CHECK(runner, saturating_add(std::numeric_limits<std::uint64_t>::max() - 1U, 2U) ==
                            std::numeric_limits<std::uint64_t>::max());

    return runner.finish();
}
