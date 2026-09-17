#pragma once

#include "vramz/runtime.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <string_view>

namespace vramz::simulation {

enum class Dataset : std::uint8_t {
    zeros,
    repeated,
    sparse,
    integers,
    fp_like,
    mixed,
    random,
    encoded_like
};
enum class Trace : std::uint8_t { hot_cold, streaming, cyclic, random, bursty, graphics, layers };

inline constexpr std::array datasets{Dataset::zeros,    Dataset::repeated,    Dataset::sparse,
                                     Dataset::integers, Dataset::fp_like,     Dataset::mixed,
                                     Dataset::random,   Dataset::encoded_like};
inline constexpr std::array traces{Trace::hot_cold, Trace::streaming, Trace::cyclic, Trace::random,
                                   Trace::bursty,   Trace::graphics,  Trace::layers};
inline constexpr std::array expansion_percent{100U, 125U, 150U, 200U, 250U, 300U};

struct Scenario final {
    Dataset dataset{Dataset::repeated};
    Trace trace{Trace::hot_cold};
    PolicyMode mode{PolicyMode::gpu_resident};
    PolicyStrategy strategy{PolicyStrategy::adaptive};
    ByteSize gpu_budget{131072U};
    std::uint32_t target_percent{300U};
    std::uint32_t steps{128U};
    std::uint64_t seed{0xC0DEC0DEU};
    std::uint64_t scale_factor{65536U};
};

struct ScaledMetrics final {
    ByteSize gpu_budget{};
    ByteSize requested_logical{};
    ByteSize gpu_resident_logical{};
    ByteSize host_logical{};
    ByteSize gpu_committed{};
    ByteSize gpu_peak_charged{};
};

struct ScenarioResult final {
    RuntimeStats stats{};
    ByteSize requested_logical{};
    ByteSize allocated_logical{};
    ScaledMetrics scaled{};
    std::uint64_t operations{};
    std::uint64_t trace_accesses{};
    std::uint64_t failed_accesses{};
    std::uint64_t failed_pressure_targets{};
    std::uint64_t hot_raw_observations{};
    std::uint64_t hot_observations{};
    bool allocation_success{true};
    bool integrity_ok{true};
    bool accounting_ok{true};
    bool cleanup_ok{true};
    bool capacity_pass{};
};

[[nodiscard]] std::string_view name(Dataset dataset) noexcept;
[[nodiscard]] std::string_view name(Trace trace) noexcept;
void fill(Dataset dataset, std::uint64_t resource_index, std::uint64_t generation,
          std::span<std::byte> output) noexcept;
[[nodiscard]] Result<ScaledMetrics> scale(const Scenario& scenario,
                                          const ScenarioResult& result) noexcept;
[[nodiscard]] Result<ScenarioResult> run(const Scenario& scenario) noexcept;

} // namespace vramz::simulation
