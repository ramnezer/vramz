#pragma once

#include "vramz/config.hpp"
#include "vramz/stats.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>

namespace vramz {

class AsyncErrorChannel final {
  public:
    explicit AsyncErrorChannel(std::uint32_t capacity) noexcept;

    void push(Error error) noexcept;
    [[nodiscard]] std::optional<AsyncErrorRecord> poll() noexcept;
    [[nodiscard]] std::optional<AsyncErrorRecord> first() const noexcept;
    [[nodiscard]] AsyncErrorStats stats() const noexcept;
    void record_diagnostic_failure() noexcept;

  private:
    mutable std::mutex mutex_{};
    std::array<AsyncErrorRecord, max_async_error_capacity> records_{};
    std::optional<AsyncErrorRecord> first_{};
    std::uint32_t capacity_{};
    std::uint32_t head_{};
    std::uint32_t count_{};
    std::uint64_t next_sequence_{1U};
    std::uint64_t produced_{};
    std::uint64_t coalesced_{};
    std::uint64_t dropped_{};
    std::uint64_t diagnostic_text_failures_{};
};

} // namespace vramz
