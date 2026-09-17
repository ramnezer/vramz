#pragma once

#include "vramz/policy_types.hpp"
#include "vramz/result.hpp"

#include <cstdint>
#include <optional>

namespace vramz {

struct TierBudget final {
    ByteSize hard_limit{};
    ByteSize soft_target{};
    ByteSize migration_reserve{};
};

struct MemoryBudgets final {
    TierBudget gpu{};
    TierBudget host{};
};

struct AsyncErrorConfig final {
    std::uint32_t max_retained_errors{64U};
};

struct BackendRequirements final {
    bool stable_device_address{true};
    bool host_tier{true};
    bool asynchronous_operations{false};
};

struct BackendCapabilities final {
    ByteSize reservation_granularity{};
    ByteSize mapping_granularity{};
    ByteSize allocation_granularity{};
    bool stable_device_address{true};
    bool host_tier{true};
    bool asynchronous_operations{false};
    ByteSize maximum_chunk_size{}; // Zero means no additional backend-specific upper bound.
    bool supports_external_async_completion{false};
    BackendId backend_id{1U};
    ByteSize recommended_allocation_granularity{};
};

struct RuntimeConfig final {
    MemoryBudgets budgets{};
    std::optional<ByteSize> preferred_chunk_size{};
    BackendRequirements required_capabilities{};
    AsyncErrorConfig async_errors{};
    PolicyConfig policy{};
    std::uint32_t maximum_pending_completions{64U};
    bool collect_performance{};
};

inline constexpr std::uint32_t max_async_error_capacity = 256U;
inline constexpr std::uint32_t max_runtime_buffers = 64U;
inline constexpr std::uint32_t max_pending_completions = 64U;

[[nodiscard]] Result<void> validate_config(const RuntimeConfig& config,
                                           const BackendCapabilities& capabilities) noexcept;

} // namespace vramz
