#include "vramz/async_error_channel.hpp"

#include "vramz/saturating.hpp"

#include <algorithm>

namespace vramz {

AsyncErrorChannel::AsyncErrorChannel(std::uint32_t capacity) noexcept
    : capacity_(std::clamp(capacity, 1U, max_async_error_capacity)) {}

void AsyncErrorChannel::push(Error error) noexcept {
    const std::scoped_lock lock{mutex_};
    saturating_increment(produced_);
    const AsyncErrorRecord candidate{error, next_sequence_, 1U};
    saturating_increment(next_sequence_);
    if (!first_.has_value()) {
        first_ = candidate;
    }

    for (std::uint32_t index = 0U; index < count_; ++index) {
        const auto slot = static_cast<std::uint32_t>((head_ + index) % capacity_);
        if (records_[slot].error == error) {
            saturating_increment(records_[slot].repeat_count);
            saturating_increment(coalesced_);
            return;
        }
    }

    if (count_ == capacity_) {
        saturating_increment(dropped_);
        return;
    }
    const auto slot = static_cast<std::uint32_t>((head_ + count_) % capacity_);
    records_[slot] = candidate;
    ++count_;
}

std::optional<AsyncErrorRecord> AsyncErrorChannel::poll() noexcept {
    const std::scoped_lock lock{mutex_};
    if (count_ == 0U) {
        return std::nullopt;
    }
    const auto result = records_[head_];
    head_ = static_cast<std::uint32_t>((head_ + 1U) % capacity_);
    --count_;
    return result;
}

std::optional<AsyncErrorRecord> AsyncErrorChannel::first() const noexcept {
    const std::scoped_lock lock{mutex_};
    return first_;
}

AsyncErrorStats AsyncErrorChannel::stats() const noexcept {
    const std::scoped_lock lock{mutex_};
    return AsyncErrorStats{capacity_,  count_,   produced_,
                           coalesced_, dropped_, diagnostic_text_failures_};
}

void AsyncErrorChannel::record_diagnostic_failure() noexcept {
    const std::scoped_lock lock{mutex_};
    saturating_increment(diagnostic_text_failures_);
}

} // namespace vramz
