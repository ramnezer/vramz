#pragma once

#include <cstdint>

namespace vramz {

// Optional host elapsed time, including synchronization. Never used in policy decisions.
struct TimingCounter final {
    std::uint64_t samples{};
    std::uint64_t nanoseconds{};
    bool overflow{};
};
struct PerformanceStats final {
    bool enabled{};
    TimingCounter policy_selection{};
    TimingCounter policy_transition{};
    TimingCounter compression{};
    TimingCounter decompression{};
};

} // namespace vramz
