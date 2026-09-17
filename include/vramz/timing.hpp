#pragma once

#include "vramz/checked.hpp"
#include "vramz/performance.hpp"

#include <chrono>

namespace vramz {

class PerformanceSample final {
  public:
    PerformanceSample(bool enabled, TimingCounter& counter) noexcept
        : counter_(counter), enabled_(enabled),
          start_(enabled ? Clock::now() : Clock::time_point{}) {}
    PerformanceSample(const PerformanceSample&) = delete;
    PerformanceSample& operator=(const PerformanceSample&) = delete;
    ~PerformanceSample() { finish(); }
    void finish() noexcept {
        if (!enabled_) {
            return;
        }
        enabled_ = false;
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count();
        if (elapsed < 0) {
            counter_.overflow = true;
            return;
        }
        const auto total = checked_add(counter_.nanoseconds, static_cast<std::uint64_t>(elapsed),
                                       OperationId::verify);
        const auto samples = checked_add(counter_.samples, 1U, OperationId::verify);
        if (!total || !samples) {
            counter_.overflow = true;
            return;
        }
        counter_.nanoseconds = total.value();
        counter_.samples = samples.value();
    }

  private:
    using Clock = std::chrono::steady_clock;
    TimingCounter& counter_;
    bool enabled_{};
    Clock::time_point start_{};
};

} // namespace vramz
