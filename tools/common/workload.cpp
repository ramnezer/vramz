#include "workload.hpp"

#include "vramz/checked.hpp"
#include "vramz/crc32c.hpp"
#include "vramz/timing.hpp"

#include <algorithm>
#include <exception>
#include <latch>
#include <new>
#include <optional>
#include <ostream>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace vramz::example {
namespace {
constexpr std::uint64_t mib = std::uint64_t{1024U} * 1024U;
[[nodiscard]] ByteSize charge(const TierUsage& usage) {
    std::uint64_t total{};
    for (const auto part :
         {usage.committed, usage.reserved, usage.staging, usage.workspace, usage.cleanup_debt}) {
        const auto sum = checked_add(total, part.value(), OperationId::verify);
        if (!sum) {
            std::terminate();
        }
        total = sum.value();
    }
    return ByteSize{total};
}
void fail(WorkloadReport& report, Error error) noexcept {
    report.has_error = true;
    report.error = error;
    if (error.code == ErrorCode::ambiguous_backend_state ||
        error.code == ErrorCode::backend_contract_violation) {
        std::terminate();
    }
}
template <class T> bool accept(const Result<T>& result, WorkloadReport& report) noexcept {
    if (!result) {
        fail(report, result.error());
    }
    return static_cast<bool>(result);
}
void contract(WorkloadReport& report) noexcept {
    fail(report, make_error(ErrorCode::integrity_failure, OperationId::verify));
}
[[nodiscard]] bool temporary_empty(const TierUsage& usage) noexcept {
    return usage.reserved == ByteSize{} && usage.staging == ByteSize{} &&
           usage.workspace == ByteSize{} && usage.cleanup_debt == ByteSize{};
}
[[nodiscard]] const char* residency(Residency state) noexcept {
    switch (state) {
    case Residency::gpu_raw:
        return "GPU_RAW";
    case Residency::gpu_compressed:
        return "GPU_COMPRESSED";
    case Residency::host_raw:
        return "HOST_RAW";
    case Residency::host_compressed:
        return "HOST_COMPRESSED";
    }
    return "INVALID";
}
class ConcurrentReaders final {
  public:
    ConcurrentReaders() = default;
    ConcurrentReaders(const ConcurrentReaders&) = delete;
    ConcurrentReaders& operator=(const ConcurrentReaders&) = delete;
    ~ConcurrentReaders() { finish(); }

    [[nodiscard]] bool start(std::array<std::optional<Buffer>, 32U>& buffers,
                             const WorkloadConfig& config) noexcept {
        try {
            for (std::size_t n = 0U; n < threads_.size(); ++n) {
                threads_[n] = std::jthread([&, n] {
                    const auto index = n < 2U ? 0U : config.buffers - 1U;
                    std::array<std::byte, 4096U> expected{};
                    std::array<std::byte, 4096U> observed{};
                    fill_payload(expected, data_class(config.profile, index), index, config.seed);
                    auto lease = buffers[index]->acquire({{}, ByteSize{expected.size()}});
                    ready_.count_down();
                    bool valid = static_cast<bool>(lease);
                    if (lease) {
                        for (std::uint32_t i = 0U; i < 16U && valid; ++i) {
                            valid = static_cast<bool>(lease.value().read({}, observed)) &&
                                    observed == expected;
                        }
                    }
                    release_.wait();
                    if (lease) {
                        valid = static_cast<bool>(lease.value().close()) && valid;
                    }
                    verified_[n] = valid;
                });
            }
            ready_.wait();
            return true;
        } catch (const std::system_error&) {
            finish();
            return false;
        } catch (const std::bad_alloc&) {
            finish();
            return false;
        }
    }
    void finish() noexcept {
        if (!released_) {
            release_.count_down();
            released_ = true;
        }
        for (auto& worker : threads_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    [[nodiscard]] bool verified() const noexcept {
        return std::ranges::all_of(verified_, [](bool value) { return value; });
    }

  private:
    std::latch ready_{3};
    std::latch release_{1};
    std::array<std::jthread, 3U> threads_{};
    std::array<bool, 3U> verified_{};
    bool released_{};
};
} // namespace

Result<RuntimeConfig> runtime_config(WorkloadConfig config) noexcept {
    const auto total =
        checked_mul(config.buffers, config.bytes_per_buffer.value(), OperationId::runtime_create);
    if (!total || config.buffers < 4U || config.buffers > 32U ||
        config.chunk_size < ByteSize{2U * mib} || config.chunk_size > ByteSize{16U * mib} ||
        config.bytes_per_buffer == ByteSize{} ||
        config.bytes_per_buffer.value() % config.chunk_size.value() != 0U ||
        total.value() > 256U * mib || total.value() / config.chunk_size.value() > 64U ||
        config.hard_limit > ByteSize{512U * mib} || config.hard_limit.value() <= total.value() ||
        static_cast<unsigned>(config.profile) > static_cast<unsigned>(Profile::p25)) {
        return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    }
    RuntimeConfig result{};
    result.required_capabilities.host_tier = false;
    result.preferred_chunk_size = config.chunk_size;
    result.budgets.gpu = {config.hard_limit, ByteSize{total.value() / 2U},
                          ByteSize{config.hard_limit.value() - total.value()}};
    result.policy.mode = config.raw_baseline ? PolicyMode::disabled : PolicyMode::gpu_resident;
    result.collect_performance = config.measure;
    result.policy.tuning.maximum_transitions = 1U;
    return result;
}

DataClass data_class(Profile profile, std::uint32_t index) noexcept {
    switch (profile) {
    case Profile::high:
        return DataClass::high;
    case Profile::moderate:
        return DataClass::moderate;
    case Profile::random:
        return DataClass::random;
    case Profile::mixed:
        return index % 8U == 1U ? DataClass::random
                                : (index % 8U == 2U ? DataClass::moderate : DataClass::high);
    case Profile::p75:
        return index % 4U == 1U ? DataClass::random : DataClass::high;
    case Profile::p50:
        return index % 4U < 2U ? DataClass::random : DataClass::high;
    case Profile::p25:
        return index % 4U == 0U ? DataClass::high : DataClass::random;
    }
    return DataClass::random;
}

void fill_payload(std::span<std::byte> bytes, DataClass kind, std::uint32_t index,
                  std::uint64_t seed) noexcept {
    // Intentional modulo-2^64 PRNG state; never used for size/accounting arithmetic.
    std::uint64_t state =
        (seed ^ (static_cast<std::uint64_t>(index) + 1U) * 0x9e3779b97f4a7c15ULL) | 1U;
    for (std::size_t offset = 0U; offset < bytes.size(); ++offset) {
        if (offset % 8U == 0U) {
            state ^= state >> 12U;
            state ^= state << 25U;
            state ^= state >> 27U;
        }
        if (kind == DataClass::high ||
            (kind == DataClass::moderate && (offset / 4096U) % 2U == 0U)) {
            bytes[offset] = static_cast<std::byte>(((offset % 4096U) * 37U + offset % 17U +
                                                    static_cast<std::size_t>(index % 8U) * 53U) %
                                                   251U);
        } else {
            const auto random = state * 0x2545f4914f6cdd1dULL;
            bytes[offset] = static_cast<std::byte>((random >> ((offset % 8U) * 8U)) & 0xffU);
        }
    }
}

void run_workload(Runtime& runtime, WorkloadConfig config, WorkloadReport& report) noexcept {
    report = {};
    report.config = config;
    PerformanceSample total_timer{config.measure, report.elapsed};
    report.performance_before = runtime.performance();
    const auto configuration = runtime_config(config);
    if (!accept(configuration, report)) {
        return;
    }
    report.target = configuration.value().budgets.gpu.soft_target;
    std::array<std::optional<Buffer>, 32U> buffers{};
    std::array<std::optional<Lease>, 32U> initializing{};
    const MemoryRange range{{}, config.bytes_per_buffer};
    try {
        const auto execute = [&]() {
            std::vector<std::byte> expected(
                static_cast<std::size_t>(config.bytes_per_buffer.value()));
            std::vector<std::byte> observed(expected.size());
            for (std::uint32_t i = 0U; i < config.buffers; ++i) {
                PerformanceSample allocate_timer{config.measure, report.allocation_times[i]};
                auto allocated = runtime.allocate(config.bytes_per_buffer);
                allocate_timer.finish();
                if (!accept(allocated, report)) {
                    return;
                }
                buffers[i].emplace(std::move(allocated).value());
                auto lease = buffers[i]->acquire(range, {AccessMode::read_write});
                if (!accept(lease, report)) {
                    return;
                }
                initializing[i].emplace(std::move(lease).value());
                fill_payload(expected, data_class(config.profile, i), i, config.seed);
                if (!accept(initializing[i]->write({}, expected), report) ||
                    !accept(initializing[i]->read({}, observed), report)) {
                    return;
                }
                if (expected != observed || crc32c(expected) != crc32c(observed)) {
                    contract(report);
                    return;
                }
            }
            for (std::uint32_t i = 0U; i < config.buffers; ++i) {
                if (!accept(initializing[i]->close(), report)) {
                    return;
                }
                initializing[i].reset();
            }
            const auto access = [&](std::uint32_t index) {
                auto lease = buffers[index]->acquire(range);
                return accept(lease, report) && accept(lease.value().close(), report);
            };
            for (std::uint32_t i = 1U; i + 1U < config.buffers; ++i) {
                if (!access(i)) {
                    return;
                }
            }
            for (std::uint32_t i = 0U; i < 16U; ++i) {
                if (!access(0U)) {
                    return;
                }
            }
            if (!access(config.buffers - 1U)) {
                return;
            }
            for (std::uint32_t i = 0U; i < 3U; ++i) {
                if (!access(0U)) {
                    return;
                }
            }
            for (std::uint32_t i = 0U; i < config.buffers; ++i) {
                for (std::uint64_t j = 0U; j < buffers[i]->stats().chunk_count; ++j) {
                    const auto chunk = buffers[i]->inspect_chunk(j);
                    if (!accept(chunk, report)) {
                        return;
                    }
                    if (report.chunk_count >= report.chunks.size() || chunk.value().pending ||
                        chunk.value().poisoned || chunk.value().residency != Residency::gpu_raw) {
                        contract(report);
                        return;
                    }
                    auto& record = report.chunks[report.chunk_count++];
                    record.buffer = i;
                    record.owner = buffers[i]->id();
                    record.data_class = data_class(config.profile, i);
                    record.initial = chunk.value();
                    const auto raw =
                        checked_add(report.initial_charge.value(),
                                    chunk.value().physical_charge.value(), OperationId::verify);
                    const auto logical =
                        checked_add(report.logical.value(), chunk.value().logical_bytes.value(),
                                    OperationId::verify);
                    if (!accept(raw, report) || !accept(logical, report)) {
                        return;
                    }
                    report.initial_charge = ByteSize{raw.value()};
                    report.logical = ByteSize{logical.value()};
                }
            }
            const auto initial = runtime.stats();
            if (charge(initial.gpu) != report.initial_charge || !temporary_empty(initial.gpu) ||
                charge(initial.host) != ByteSize{}) {
                contract(report);
                return;
            }
            ConcurrentReaders readers;
            if (config.concurrent_reads && !readers.start(buffers, config)) {
                fail(report, make_error(ErrorCode::out_of_host_memory, OperationId::acquire));
                return;
            }
            for (std::uint32_t cycle = 0U; cycle < report.chunk_count && !config.raw_baseline;
                 ++cycle) {
                // The active working buffer stays HOT through ordinary accesses, not metadata
                // edits. In mixed profiles HARD pressure may still select it after colder choices
                // fail.
                if ((config.profile == Profile::high || config.profile == Profile::mixed ||
                     config.concurrent_reads) &&
                    !access(0U)) {
                    return;
                }
                const auto before = runtime.stats();
                if (charge(before.gpu) <= report.target) {
                    break;
                }
                auto& item = report.cycles[report.cycle_count++];
                item.before = before.policy;
                item.charged_before = charge(before.gpu);
                // Only a target enters production policy. No buffer/chunk/resource is supplied.
                item.timing_before = runtime.performance();
                PerformanceSample reclaim_timer{config.measure, item.elapsed};
                const auto reclaimed = runtime.reclaim_to_target(report.target);
                reclaim_timer.finish();
                item.timing_after = runtime.performance();
                const auto after = runtime.stats();
                item.after = after.policy;
                item.charged_after = charge(after.gpu);
                item.incomplete = !reclaimed;
                if (!reclaimed) {
                    item.error = reclaimed.error();
                }
                if ((!reclaimed && reclaimed.error().code != ErrorCode::out_of_gpu_memory) ||
                    after.policy.compression_failures != before.policy.compression_failures) {
                    fail(report, reclaimed
                                     ? make_error(ErrorCode::backend_failure, OperationId::migrate)
                                     : reclaimed.error());
                    return;
                }
                if (after.policy.host_fallback_count != 0U || charge(after.host) != ByteSize{} ||
                    !temporary_empty(after.gpu)) {
                    contract(report);
                    return;
                }
                if (after.policy.compression_attempts == before.policy.compression_attempts) {
                    break;
                }
            }
            readers.finish();
            report.concurrent_reads_verified = config.concurrent_reads && readers.verified();
            if (config.concurrent_reads && !report.concurrent_reads_verified) {
                contract(report);
                return;
            }
            const auto settled = runtime.stats();
            const auto resources = runtime.resources();
            if (!accept(resources, report)) {
                return;
            }
            report.settled_resources = resources.value();
            std::uint64_t live_logical{};
            const auto granularity = runtime.capabilities().minimum_allocation_granularity.value();
            for (std::uint32_t i = 0U; i < report.chunk_count; ++i) {
                auto& record = report.chunks[i];
                const auto chunk = buffers[record.buffer]->inspect_chunk(record.initial.index);
                if (!accept(chunk, report)) {
                    return;
                }
                record.settled = chunk.value();
                if (chunk.value().pending || chunk.value().poisoned ||
                    (chunk.value().residency != Residency::gpu_raw &&
                     chunk.value().residency != Residency::gpu_compressed)) {
                    contract(report);
                    return;
                }
                const auto measured_logical = checked_add(
                    live_logical, chunk.value().logical_bytes.value(), OperationId::verify);
                if (!accept(measured_logical, report)) {
                    return;
                }
                live_logical = measured_logical.value();
                if (chunk.value().logical_bytes != record.initial.logical_bytes ||
                    chunk.value().logical_crc != record.initial.logical_crc || granularity == 0U ||
                    chunk.value().physical_charge == ByteSize{} ||
                    chunk.value().physical_charge.value() % granularity != 0U) {
                    contract(report);
                    return;
                }
                if (chunk.value().residency == Residency::gpu_compressed &&
                    (chunk.value().physical_charge >= record.initial.physical_charge ||
                     record.initial.physical_charge.value() -
                             chunk.value().physical_charge.value() <
                         granularity)) {
                    contract(report);
                    return;
                }
                const auto sum =
                    checked_add(report.settled_charge.value(),
                                chunk.value().physical_charge.value(), OperationId::verify);
                if (!accept(sum, report)) {
                    return;
                }
                report.settled_charge = ByteSize{sum.value()};
            }
            report.hot_preserved = true;
            report.warm_preserved = true;
            for (std::uint32_t i = 0U; i < report.chunk_count; ++i) {
                const auto& record = report.chunks[i];
                if (record.buffer == 0U && record.settled.residency != Residency::gpu_raw) {
                    report.hot_preserved = false;
                }
                if (record.buffer == config.buffers - 1U &&
                    record.settled.residency != Residency::gpu_raw) {
                    report.warm_preserved = false;
                }
            }
            report.target_reached = report.settled_charge <= report.target;
            report.snapshot_proven =
                settled.live_buffers == config.buffers && live_logical == report.logical.value() &&
                temporary_empty(settled.gpu) && charge(settled.host) == ByteSize{} &&
                report.settled_charge == charge(settled.gpu) &&
                report.settled_charge == resources.value().owned_gpu_charge &&
                resources.value().accounting_conserved;
            if (!report.snapshot_proven ||
                ((config.profile == Profile::high || config.profile == Profile::mixed ||
                  config.concurrent_reads) &&
                 !report.hot_preserved)) {
                contract(report);
                return;
            }
            // Restoration starts only after the live authoritative snapshot above.
            for (std::uint32_t i = 0U; i < config.buffers; ++i) {
                fill_payload(expected, data_class(config.profile, i), i, config.seed);
                PerformanceSample restore_timer{config.measure, report.restore_times[i]};
                auto lease = buffers[i]->acquire(range);
                restore_timer.finish();
                if (!accept(lease, report) || !accept(lease.value().read({}, observed), report)) {
                    return;
                }
                if (expected != observed || crc32c(expected) != crc32c(observed)) {
                    contract(report);
                    return;
                }
                if (!accept(lease.value().close(), report)) {
                    return;
                }
                for (std::uint32_t j = 0U; j < report.chunk_count; ++j) {
                    if (report.chunks[j].buffer == i) {
                        report.chunks[j].integrity = true;
                    }
                }
                if (!accept(buffers[i]->close(), report)) {
                    return;
                }
                buffers[i].reset();
            }
            report.integrity = true;
        };
        execute();
    } catch (const std::bad_alloc&) {
        fail(report, make_error(ErrorCode::out_of_host_memory, OperationId::allocate));
    }
    for (auto& lease : initializing) {
        if (lease && lease->valid() && !accept(lease->close(), report)) {
            return;
        }
        lease.reset();
    }
    for (auto& buffer : buffers) {
        if (buffer && !accept(buffer->close(), report)) {
            return;
        }
        buffer.reset();
    }
    report.peak = runtime.stats().gpu.peak_charged;
    if (!config.keep_runtime && !accept(runtime.shutdown(), report)) {
        return;
    }
    const auto resources = runtime.resources();
    if (!accept(resources, report)) {
        return;
    }
    report.final_resources = resources.value();
    report.final_counts_known = true;
    const auto stats = runtime.stats();
    report.cleanup =
        resources.value().accounting_conserved && resources.value().backend_resources == 0U &&
        resources.value().va_reservations == 0U &&
        resources.value().owned_gpu_charge == ByteSize{} &&
        resources.value().ledger_gpu_charge == ByteSize{} &&
        resources.value().unmaterialized_reservations == ByteSize{} &&
        (config.keep_runtime
             ? (resources.value().retained_context && resources.value().stream_owned)
             : (!resources.value().retained_context && !resources.value().stream_owned)) &&
        stats.live_buffers == 0U && stats.quarantined_buffers == 0U &&
        charge(stats.gpu) == ByteSize{} && charge(stats.host) == ByteSize{};
    total_timer.finish();
    report.performance_after = runtime.performance();
    if (report.elapsed.overflow || report.performance_after.compression.overflow ||
        report.performance_after.decompression.overflow ||
        report.performance_after.policy_selection.overflow ||
        report.performance_after.policy_transition.overflow) {
        contract(report);
    }
    report.passed = !report.has_error && report.snapshot_proven && report.integrity &&
                    report.cleanup && report.peak <= config.hard_limit;
}

namespace {
void timing(std::ostream& out, const char* name, TimingCounter value) {
    out << ",\"" << name << "\":{\"samples\":" << value.samples
        << ",\"nanoseconds\":" << value.nanoseconds << ",\"overflow\":" << value.overflow << '}';
}
TimingCounter difference(TimingCounter after, TimingCounter before) {
    if (after.samples < before.samples || after.nanoseconds < before.nanoseconds) {
        return {0U, 0U, true};
    }
    return {after.samples - before.samples, after.nanoseconds - before.nanoseconds,
            after.overflow || before.overflow};
}
void timings(std::ostream& out, const PerformanceStats& after, const PerformanceStats& before) {
    timing(out, "policy_selection", difference(after.policy_selection, before.policy_selection));
    timing(out, "policy_transition", difference(after.policy_transition, before.policy_transition));
    timing(out, "compression", difference(after.compression, before.compression));
    timing(out, "decompression", difference(after.decompression, before.decompression));
}
} // namespace
void print_report(std::ostream& out, const WorkloadReport& r, bool physical) {
    out << std::boolalpha << "{\"v1_workload_version\":1,\"result\":\""
        << (r.passed ? "PASS" : "FAIL") << "\",\"execution_kind\":\""
        << (physical ? "physical" : "fake")
        << "\",\"profile\":" << static_cast<unsigned>(r.config.profile)
        << ",\"iteration\":" << r.iteration << ",\"warmup\":" << r.warmup
        << ",\"measurement_enabled\":" << r.config.measure
        << ",\"raw_baseline\":" << r.config.raw_baseline
        << ",\"runtime_retained\":" << r.config.keep_runtime << ",\"seed\":" << r.config.seed
        << ",\"buffer_count\":" << r.config.buffers << ",\"chunk_count\":" << r.chunk_count
        << ",\"aggregate_logical_bytes\":" << r.logical.value()
        << ",\"raw_physical_charge\":" << r.initial_charge.value()
        << ",\"settled_physical_charge\":" << r.settled_charge.value()
        << ",\"target\":" << r.target.value() << ",\"target_reached\":" << r.target_reached
        << ",\"peak_admitted_gpu_bytes\":" << r.peak.value()
        << ",\"maximum_physical_bytes\":" << r.config.hard_limit.value()
        << ",\"manual_victim_selection\":false,\"manual_compression_transition\":false"
        << ",\"snapshot_proven\":" << r.snapshot_proven << ",\"hot_preserved\":" << r.hot_preserved
        << ",\"concurrent_reads_enabled\":" << r.config.concurrent_reads
        << ",\"concurrent_reads_verified\":" << r.concurrent_reads_verified
        << ",\"warm_preserved\":" << r.warm_preserved << ",\"integrity\":" << r.integrity
        << ",\"cleanup\":" << r.cleanup << ",\"cycles\":[";
    for (std::uint32_t i = 0U; i < r.cycle_count; ++i) {
        const auto& c = r.cycles[i];
        if (i != 0U) {
            out << ',';
        }
        out << "{\"id\":" << c.after.policy_cycles
            << ",\"selected_buffer\":" << c.after.last_victim_buffer.value()
            << ",\"selected_chunk\":" << c.after.last_victim_chunk.value()
            << ",\"before\":" << c.charged_before.value()
            << ",\"after\":" << c.charged_after.value()
            << ",\"attempts\":" << c.after.compression_attempts - c.before.compression_attempts
            << ",\"rejected\":" << c.after.compression_rejected - c.before.compression_rejected
            << ",\"successes\":" << c.after.compression_successes - c.before.compression_successes;
        timing(out, "elapsed", c.elapsed);
        timings(out, c.timing_after, c.timing_before);
        out << '}';
    }
    out << "],\"chunks\":[";
    for (std::uint32_t i = 0U; i < r.chunk_count; ++i) {
        const auto& c = r.chunks[i];
        if (i != 0U) {
            out << ',';
        }
        out << "{\"buffer\":" << c.buffer << ",\"owner\":" << c.owner.value()
            << ",\"chunk\":" << c.initial.index
            << ",\"class\":" << static_cast<unsigned>(c.data_class)
            << ",\"raw_charge\":" << c.initial.physical_charge.value()
            << ",\"stored_bytes\":" << c.settled.stored_bytes.value()
            << ",\"physical_charge\":" << c.settled.physical_charge.value()
            << ",\"representation\":\"" << residency(c.settled.residency)
            << "\",\"compressibility\":" << static_cast<unsigned>(c.settled.compressibility)
            << ",\"policy_metadata_available\":" << c.settled.policy_metadata_available
            << ",\"retry_after\":" << c.settled.retry_after
            << ",\"attempts\":" << c.settled.compression_attempts
            << ",\"crc32c\":" << c.initial.logical_crc
            << ",\"stored_crc32c\":" << c.settled.stored_crc << ",\"integrity\":" << c.integrity
            << '}';
    }
    out << "],\"timing\":{\"clock\":\"host_steady_clock\"";
    timing(out, "elapsed", r.elapsed);
    timings(out, r.performance_after, r.performance_before);
    out << ",\"buffers\":[";
    for (std::uint32_t i = 0U; i < r.config.buffers; ++i) {
        if (i != 0U) {
            out << ',';
        }
        out << "{\"index\":" << i;
        timing(out, "allocation", r.allocation_times[i]);
        timing(out, "acquire_restore", r.restore_times[i]);
        out << '}';
    }
    out << "]},\"final_counts_known\":" << r.final_counts_known;
    if (r.final_counts_known) {
        out << ",\"final_backend_resources\":" << r.final_resources.backend_resources
            << ",\"final_raw_resources\":" << r.final_resources.raw_resources
            << ",\"final_compressed_resources\":" << r.final_resources.compressed_resources
            << ",\"final_va_reservations\":" << r.final_resources.va_reservations
            << ",\"final_budget_charge\":" << r.final_resources.ledger_gpu_charge.value()
            << ",\"final_unmaterialized_reservations\":"
            << r.final_resources.unmaterialized_reservations.value()
            << ",\"final_workspace\":" << r.final_resources.workspace_resources
            << ",\"final_context\":" << r.final_resources.retained_context
            << ",\"final_stream\":" << r.final_resources.stream_owned;
    }
    out << ",\"error\":";
    if (r.has_error) {
        out << "{\"code\":" << static_cast<unsigned>(r.error.code)
            << ",\"operation\":" << static_cast<unsigned>(r.error.operation)
            << ",\"native\":" << r.error.native_code << '}';
    } else {
        out << "null";
    }
    out << "}\n";
}

} // namespace vramz::example
