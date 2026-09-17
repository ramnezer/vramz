#include "vramz/detail/simulation.hpp"

#include "vramz/checked.hpp"
#include "vramz/saturating.hpp"
#include "vramz/testing.hpp"

#include <algorithm>
#include <array>
#include <new>
#include <vector>

namespace vramz::simulation {
namespace {

constexpr std::size_t payload_size = 8192U;

[[nodiscard]] std::uint64_t next_random(std::uint64_t& state) noexcept {
    // Intentional unsigned PRNG mixing, never size or ownership arithmetic.
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return state;
}

[[nodiscard]] std::size_t trace_index(Trace trace, std::uint64_t step, std::size_t count,
                                      std::uint64_t& seed) noexcept {
    const auto hot_count = std::min(std::size_t{4U}, count);
    switch (trace) {
    case Trace::hot_cold:
        return static_cast<std::size_t>(step % 8U == 7U ? step % count : step % hot_count);
    case Trace::streaming:
    case Trace::cyclic:
    case Trace::layers:
        return static_cast<std::size_t>(step % count);
    case Trace::random:
        return static_cast<std::size_t>(next_random(seed) % count);
    case Trace::bursty:
        return static_cast<std::size_t>((step / 16U + step % 2U) % count);
    case Trace::graphics:
        // Rotating scene, nearby reusable assets, and a streaming background population.
        return static_cast<std::size_t>(
            (step / 32U * 4U + (step % 5U == 4U ? 8U + step / 5U : step % 4U)) % count);
    }
    return 0U;
}

void inspect(Runtime& runtime, ScenarioResult& result, ByteSize hard_limit) noexcept {
    result.accounting_ok = result.accounting_ok &&
                           testing::accounting_conserved(runtime, PhysicalTier::gpu) &&
                           testing::accounting_conserved(runtime, PhysicalTier::host);
    const auto stats = runtime.stats();
    result.accounting_ok = result.accounting_ok && stats.gpu.peak_charged <= hard_limit &&
                           stats.gpu.committed <= ByteSize{hard_limit.value() - 12288U};
    ++result.operations;
}

void access_bytes(Runtime& runtime, Buffer& buffer, std::uint64_t index, std::uint64_t& generation,
                  Dataset dataset, bool write, ScenarioResult& result) noexcept {
    std::array<std::byte, payload_size> expected{};
    std::array<std::byte, payload_size> actual{};
    fill(dataset, index, generation, expected);
    auto acquired =
        buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{payload_size}},
                       AcquireOptions{write ? AccessMode::read_write : AccessMode::read_only});
    if (!acquired) {
        ++result.failed_accesses;
        return;
    }
    auto lease = std::move(acquired).value();
    const auto read = testing::read_bytes(lease, ByteOffset{}, actual);
    result.integrity_ok = result.integrity_ok && read && actual == expected;
    if (write) {
        ++generation;
        fill(dataset, index, generation, expected);
        const auto written = testing::write_bytes(lease, ByteOffset{}, expected);
        result.integrity_ok = result.integrity_ok && written;
    }
    const auto closed = lease.close();
    result.integrity_ok = result.integrity_ok && closed;
    static_cast<void>(runtime);
}

[[nodiscard]] Result<ScenarioResult> run_checked(const Scenario& scenario) {
    const auto requested =
        checked_mul(scenario.gpu_budget.value(), scenario.target_percent, OperationId::allocate);
    if (!requested || requested.value() % 100U != 0U) {
        return make_error(ErrorCode::invalid_argument, OperationId::allocate);
    }
    const auto logical = requested.value() / 100U;
    if (logical % payload_size != 0U || logical / payload_size > max_runtime_buffers ||
        logical == 0U) {
        return make_error(ErrorCode::invalid_argument, OperationId::allocate);
    }
    RuntimeConfig config{};
    config.budgets =
        MemoryBudgets{TierBudget{scenario.gpu_budget,
                                 ByteSize{scenario.gpu_budget.value() * 3U / 4U}, ByteSize{12288U}},
                      TierBudget{ByteSize{1048576U}, ByteSize{786432U}, ByteSize{12288U}}};
    config.preferred_chunk_size = ByteSize{4096U};
    config.policy.mode = scenario.mode;
    config.policy.tuning.strategy = scenario.strategy;
    auto created = Runtime::create(config);
    if (!created) {
        return created.error();
    }
    auto runtime = std::move(created).value();
    std::vector<Buffer> buffers;
    buffers.reserve(static_cast<std::size_t>(logical / payload_size));
    std::vector<std::uint64_t> generations(static_cast<std::size_t>(logical / payload_size), 0U);
    ScenarioResult result{};
    result.requested_logical = ByteSize{logical};
    std::array<std::byte, payload_size> input{};
    for (std::uint64_t index = 0U; index < logical / payload_size; ++index) {
        auto allocated = runtime.allocate(ByteSize{payload_size});
        if (!allocated) {
            result.allocation_success = false;
            inspect(runtime, result, scenario.gpu_budget);
            break;
        }
        buffers.push_back(std::move(allocated).value());
        result.allocated_logical = ByteSize{result.allocated_logical.value() + payload_size};
        auto acquired = buffers.back().acquire(MemoryRange{ByteOffset{}, ByteSize{payload_size}},
                                               AcquireOptions{AccessMode::read_write});
        if (!acquired) {
            ++result.failed_accesses;
            result.allocation_success = false;
            const auto closed = buffers.back().close();
            result.cleanup_ok = result.cleanup_ok && closed;
            buffers.pop_back();
            result.allocated_logical = ByteSize{result.allocated_logical.value() - payload_size};
            inspect(runtime, result, scenario.gpu_budget);
            break;
        }
        auto lease = std::move(acquired).value();
        fill(scenario.dataset, index, 0U, input);
        const auto written = testing::write_bytes(lease, ByteOffset{}, input);
        const auto closed = lease.close();
        result.integrity_ok = result.integrity_ok && written && closed;
        inspect(runtime, result, scenario.gpu_budget);
    }
    auto seed = scenario.seed;
    const auto trace_steps = scenario.trace == Trace::streaming
                                 ? std::min(static_cast<std::uint64_t>(scenario.steps),
                                            static_cast<std::uint64_t>(buffers.size()))
                                 : static_cast<std::uint64_t>(scenario.steps);
    for (std::uint64_t step = 0U; step < trace_steps && !buffers.empty(); ++step) {
        const auto index = trace_index(scenario.trace, step, buffers.size(), seed);
        ++result.trace_accesses;
        access_bytes(runtime, buffers[index], index, generations[index], scenario.dataset,
                     step % 31U == 30U, result);
        if (scenario.trace == Trace::bursty && step % 16U == 15U) {
            testing::RuntimeAccess::advance_policy_epoch(runtime, 12U);
        }
        if (step % 16U == 15U) {
            const auto reclaimed = runtime.reclaim_to_target(config.budgets.gpu.soft_target);
            if (!reclaimed) {
                ++result.failed_pressure_targets;
            }
        }
        for (std::size_t hot = 0U; hot < std::min(std::size_t{4U}, buffers.size()); ++hot) {
            const auto snapshot = testing::chunk_snapshot(buffers[hot], 0U);
            if (snapshot) {
                ++result.hot_observations;
                if (snapshot.value().authoritative.state == RepresentationState::gpu_raw) {
                    ++result.hot_raw_observations;
                }
            }
        }
        inspect(runtime, result, scenario.gpu_budget);
    }
    // Even a failed-capacity row verifies its admitted prefix, not a reduced advertised workload.
    for (std::size_t index = 0U; index < buffers.size(); ++index) {
        access_bytes(runtime, buffers[index], index, generations[index], scenario.dataset, false,
                     result);
        inspect(runtime, result, scenario.gpu_budget);
    }
    result.stats = runtime.stats();
    result.capacity_pass =
        result.allocation_success && result.allocated_logical == result.requested_logical &&
        result.failed_accesses == 0U && result.integrity_ok && result.accounting_ok &&
        result.stats.policy.logical_host_bytes == ByteSize{} &&
        result.stats.policy.host_fallback_count == 0U &&
        result.stats.policy.logical_gpu_resident_bytes == result.requested_logical;
    const auto scaled = scale(scenario, result);
    if (!scaled) {
        return scaled.error();
    }
    result.scaled = scaled.value();
    for (auto& buffer : buffers) {
        const auto closed = buffer.close();
        result.cleanup_ok = result.cleanup_ok && closed;
        inspect(runtime, result, scenario.gpu_budget);
    }
    const auto stopped = runtime.shutdown();
    result.cleanup_ok =
        result.cleanup_ok && stopped && testing::owned_resource_count(runtime) == 0U;
    const auto final = runtime.stats();
    for (const auto& tier : {final.gpu, final.host}) {
        result.cleanup_ok = result.cleanup_ok && tier.committed == ByteSize{} &&
                            tier.reserved == ByteSize{} && tier.staging == ByteSize{} &&
                            tier.workspace == ByteSize{} && tier.cleanup_debt == ByteSize{};
    }
    result.capacity_pass = result.capacity_pass && result.cleanup_ok && result.accounting_ok;
    return result;
}

} // namespace

std::string_view name(Dataset dataset) noexcept {
    switch (dataset) {
    case Dataset::zeros:
        return "zeros";
    case Dataset::repeated:
        return "repeated";
    case Dataset::sparse:
        return "sparse";
    case Dataset::integers:
        return "integers";
    case Dataset::fp_like:
        return "fp-like";
    case Dataset::mixed:
        return "mixed";
    case Dataset::random:
        return "random";
    case Dataset::encoded_like:
        return "encoded-like";
    }
    return "invalid";
}

std::string_view name(Trace trace) noexcept {
    switch (trace) {
    case Trace::hot_cold:
        return "hot-cold";
    case Trace::streaming:
        return "streaming";
    case Trace::cyclic:
        return "cyclic";
    case Trace::random:
        return "random";
    case Trace::bursty:
        return "bursty";
    case Trace::graphics:
        return "graphics";
    case Trace::layers:
        return "layers";
    }
    return "invalid";
}

void fill(Dataset dataset, std::uint64_t resource_index, std::uint64_t generation,
          std::span<std::byte> output) noexcept {
    // Intentional PRNG seed mixing; no derived allocation size uses modulo arithmetic.
    std::uint64_t random =
        0x123456789ABCDEFULL ^ (resource_index * 0x9E3779B185EBCA87ULL) ^ generation;
    for (std::size_t offset = 0U; offset < output.size(); ++offset) {
        const auto local_dataset =
            dataset == Dataset::mixed
                ? ((resource_index * 2U + offset / 4096U) % 3U == 0U ? Dataset::random
                                                                     : Dataset::integers)
                : dataset;
        const auto noise = next_random(random);
        std::uint64_t value = 0U;
        switch (local_dataset) {
        case Dataset::zeros:
            break;
        case Dataset::repeated:
            value = (offset % 32U + resource_index + generation) & 255U;
            break;
        case Dataset::sparse:
            value = offset % 64U == 0U ? noise & 255U : 0U;
            break;
        case Dataset::integers:
            value = offset % 4U == 0U ? (offset / 4U + generation) % 64U : 0U;
            break;
        case Dataset::fp_like: {
            const std::uint64_t bits = 0x3F800000U | ((offset / 4U % 128U) << 8U);
            value = (bits >> ((offset % 4U) * 8U)) & 255U;
            break;
        }
        case Dataset::mixed:
            break;
        case Dataset::random:
            value = noise & 255U;
            break;
        case Dataset::encoded_like:
            value = (noise >> 8U) & 255U;
            break;
        }
        output[offset] = static_cast<std::byte>(value);
    }
}

Result<ScaledMetrics> scale(const Scenario& scenario, const ScenarioResult& result) noexcept {
    if (scenario.scale_factor == 0U) {
        return make_error(ErrorCode::invalid_argument, OperationId::budget_transfer);
    }
    const std::array input{scenario.gpu_budget,
                           result.requested_logical,
                           result.stats.policy.logical_gpu_resident_bytes,
                           result.stats.policy.logical_host_bytes,
                           result.stats.gpu.committed,
                           result.stats.gpu.peak_charged};
    std::array<ByteSize, 6U> output{};
    for (std::size_t index = 0U; index < input.size(); ++index) {
        const auto multiplied =
            checked_mul(input[index].value(), scenario.scale_factor, OperationId::budget_transfer);
        if (!multiplied) {
            return multiplied.error();
        }
        output[index] = ByteSize{multiplied.value()};
    }
    return ScaledMetrics{output[0], output[1], output[2], output[3], output[4], output[5]};
}

Result<ScenarioResult> run(const Scenario& scenario) noexcept {
    if (scenario.gpu_budget < ByteSize{32768U} || scenario.gpu_budget > ByteSize{524288U} ||
        scenario.target_percent < 100U || scenario.target_percent > 300U ||
        scenario.steps > 10000U ||
        std::find(datasets.begin(), datasets.end(), scenario.dataset) == datasets.end() ||
        std::find(traces.begin(), traces.end(), scenario.trace) == traces.end()) {
        return make_error(ErrorCode::invalid_argument, OperationId::allocate);
    }
    try {
        return run_checked(scenario);
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::allocate);
    }
}

} // namespace vramz::simulation
