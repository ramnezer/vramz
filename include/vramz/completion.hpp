#pragma once

#include "vramz/types.hpp"

#include <cstdint>
#include <type_traits>

namespace vramz {

enum class CompletionState : std::uint8_t { pending, complete, ambiguous, released };

// Non-owning proof identity. Copying this value never copies a fence or pin obligation.
// The bounded Runtime registry is the unique cleanup owner; IDs are never reused.
struct CompletionToken final {
    CompletionTokenId id{};
    BackendId backend{};
    RuntimeId runtime{};
    auto operator<=>(const CompletionToken&) const = default;
};
static_assert(std::is_trivially_copyable_v<CompletionToken>);

struct CompletionStats final {
    std::uint64_t pending_completion_count{};
    std::uint64_t completed_deferred_releases{};
    std::uint64_t completion_not_ready_count{};
    std::uint64_t completion_failures{};
};

} // namespace vramz
