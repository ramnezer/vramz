#pragma once

#include "vramz/runtime.hpp"

#include <array>
#include <iosfwd>
#include <span>

namespace vramz::example {

enum class Profile : std::uint8_t { high, moderate, random, mixed, p75, p50, p25 };
enum class DataClass : std::uint8_t { high, moderate, random };
struct WorkloadConfig final {
    std::uint32_t buffers{8U};
    ByteSize bytes_per_buffer{std::uint64_t{8U} * 1024U * 1024U};
    ByteSize chunk_size{std::uint64_t{8U} * 1024U * 1024U};
    ByteSize hard_limit{std::uint64_t{128U} * 1024U * 1024U};
    Profile profile{Profile::mixed};
    std::uint64_t seed{0x564d5aU};
    bool concurrent_reads{};
    bool measure{};
    bool raw_baseline{};
    bool keep_runtime{};
};
struct ChunkRecord final {
    std::uint32_t buffer{};
    BufferId owner{};
    DataClass data_class{};
    ChunkInfo initial{};
    ChunkInfo settled{};
    bool integrity{};
};
struct CycleRecord final {
    PolicyStats before{};
    PolicyStats after{};
    ByteSize charged_before{};
    ByteSize charged_after{};
    Error error{};
    bool incomplete{};
    TimingCounter elapsed{};
    PerformanceStats timing_before{};
    PerformanceStats timing_after{};
};
struct WorkloadReport final {
    WorkloadConfig config{};
    ByteSize logical{};
    ByteSize initial_charge{};
    ByteSize settled_charge{};
    ByteSize target{};
    ByteSize peak{};
    std::array<ChunkRecord, 64U> chunks{};
    std::uint32_t chunk_count{};
    std::array<CycleRecord, 64U> cycles{};
    std::uint32_t cycle_count{};
    ResourceStats settled_resources{};
    ResourceStats final_resources{};
    bool snapshot_proven{};
    bool hot_preserved{};
    bool warm_preserved{};
    bool target_reached{};
    bool integrity{};
    bool cleanup{};
    bool final_counts_known{};
    bool concurrent_reads_verified{};
    std::uint32_t iteration{};
    bool warmup{};
    TimingCounter elapsed{};
    std::array<TimingCounter, 32U> allocation_times{};
    std::array<TimingCounter, 32U> restore_times{};
    PerformanceStats performance_before{};
    PerformanceStats performance_after{};
    bool passed{};
    bool has_error{};
    Error error{};
};

[[nodiscard]] Result<RuntimeConfig> runtime_config(WorkloadConfig config) noexcept;
[[nodiscard]] DataClass data_class(Profile profile, std::uint32_t index) noexcept;
void fill_payload(std::span<std::byte> bytes, DataClass kind, std::uint32_t index,
                  std::uint64_t seed) noexcept;
void run_workload(Runtime& runtime, WorkloadConfig config, WorkloadReport& report) noexcept;
void print_report(std::ostream& output, const WorkloadReport& report, bool physical);

} // namespace vramz::example
