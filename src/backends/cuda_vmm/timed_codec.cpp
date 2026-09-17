#include "vramz/detail/gpu_compression.hpp"
#include "vramz/timing.hpp"

#include <new>
#include <optional>
#include <utility>

namespace vramz::detail {
namespace {
// Decorates the production stream contract without adding a GPU operation or changing ordering.
class TimedCodec final : public GpuCompressionApi {
  public:
    explicit TimedCodec(std::unique_ptr<GpuCompressionApi> inner) noexcept
        : inner_(std::move(inner)) {}
    Result<GpuCodecRequirements> requirements(GpuCodecDirection direction,
                                              ByteSize bytes) noexcept override {
        return inner_->requirements(direction, bytes);
    }
    Result<void> initialize(CudaContext context) noexcept override {
        return inner_->initialize(context);
    }
    Result<void> launch(GpuCodecJob job) noexcept override {
        sample_.emplace(true, job.direction == GpuCodecDirection::compress ? stats_.compression
                                                                           : stats_.decompression);
        const auto result = inner_->launch(job);
        if (!result) {
            sample_.reset();
        }
        return result;
    }
    Result<void> synchronize() noexcept override {
        const auto result = inner_->synchronize();
        if (!result) {
            sample_.reset();
        }
        return result;
    }
    Result<GpuCodecResult> result() noexcept override {
        const auto result = inner_->result();
        sample_.reset();
        return result;
    }
    Result<void> shutdown() noexcept override { return inner_->shutdown(); }
    PerformanceStats performance() const noexcept override { return stats_; }

  private:
    std::unique_ptr<GpuCompressionApi> inner_;
    PerformanceStats stats_{true, {}, {}, {}};
    std::optional<PerformanceSample> sample_{};
};
} // namespace

Result<std::unique_ptr<GpuCompressionApi>>
make_timed_codec(std::unique_ptr<GpuCompressionApi>& inner) noexcept {
    if (!inner) {
        return make_error(ErrorCode::invalid_argument, OperationId::runtime_create);
    }
    try {
        return std::unique_ptr<GpuCompressionApi>{std::make_unique<TimedCodec>(std::move(inner))};
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::runtime_create);
    }
}
} // namespace vramz::detail
