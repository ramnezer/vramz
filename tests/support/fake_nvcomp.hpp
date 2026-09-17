#pragma once

#include "fake_cuda_driver.hpp"
#include "vramz/detail/gpu_compression.hpp"

#include <array>
#include <memory>
#include <mutex>

namespace vramz::test {

enum class GpuCodecFault : std::uint8_t {
    plan_bound_overflow,
    plan_alignment_zero,
    plan_temp_overflow,
    metadata_submission_failure,
    metadata_submission_ambiguous,
    metadata_publication_partial,
    launch_failure,
    launch_ambiguous,
    completion_failure,
    premature_status,
    premature_output,
    status_read_failure,
    size_read_failure,
    status_failure,
    size_zero,
    size_oversized,
    size_mismatch,
    output_corruption,
    crc_preserving_output_corruption,
    literal_only_output,
    shutdown_failure,
    shutdown_ambiguous,
    literal_prefix_output,
    count
};
enum class GpuCodecRuntimeFault : std::uint8_t { none, initialize, launch, synchronize };
struct FakeGpuCodecConfig final {
    ByteSize alignment{4U};
    ByteSize compression_temp{128U};
    ByteSize decompression_temp{64U};
    GpuCodecRuntimeFault runtime_fault{GpuCodecRuntimeFault::none};
    std::int64_t runtime_native_error{};
    ByteSize literal_prefix_bytes{};
    // A conservative admission bound may exceed the CPU codec's minimum required bound.
    ByteSize compression_bound_floor{};
};
enum class GpuCodecCall : std::uint8_t {
    metadata_copy_enqueue,
    nvcomp_launch,
    stream_synchronize,
    metadata_status_read
};
struct FakeGpuCodecSnapshot final {
    std::uint64_t plans{};
    std::uint64_t compressions{};
    std::uint64_t decompressions{};
    std::uint64_t completions{};
    std::uint64_t result_reads{};
    std::uint64_t rejected_early_reads{};
    std::uint64_t streams{};
    std::uint64_t initialization_attempts{};
    std::uint64_t shutdown_attempts{};
    bool pending{};
    bool metadata_publication_pending{};
    std::uint64_t metadata_publications{};
    std::uint64_t metadata_consumptions{};
    detail::GpuBatchMetadata consumed_metadata{};
    std::array<GpuCodecCall, 64U> log{};
    std::size_t log_size{};
    std::uint64_t dropped_calls{};
};
struct FakeGpuCodecState final {
    struct LiveAllocation final {
        detail::CudaPhysicalHandle handle{};
        DeviceAddress address{};
        ByteSize charge{};
    };
    struct LaunchAudit final {
        detail::GpuCodecDirection direction{};
        detail::GpuBatchMetadata metadata{};
        std::array<LiveAllocation, 64U> allocations{};
        std::size_t allocation_count{};
        std::size_t cuda_log_position{};
        ByteSize owned_bytes{};
    };
    mutable std::mutex mutex{};
    FakeGpuCodecConfig config{};
    FakeGpuCodecSnapshot observation{};
    std::array<std::uint64_t, static_cast<std::size_t>(GpuCodecFault::count)> countdowns{};
    // Optional passive observations never alter the fake's allocation or codec contract.
    std::weak_ptr<FakeCudaState> cuda_audit{};
    std::array<LaunchAudit, 16U> launches{};
    std::size_t launch_count{};
};

class FakeNvcompLz4Api final : public detail::GpuCompressionApi {
  public:
    FakeNvcompLz4Api(detail::CudaDriverApi& driver,
                     std::shared_ptr<FakeGpuCodecState> state) noexcept;
    ~FakeNvcompLz4Api() override;
    FakeNvcompLz4Api(const FakeNvcompLz4Api&) = delete;
    FakeNvcompLz4Api& operator=(const FakeNvcompLz4Api&) = delete;
    FakeNvcompLz4Api(FakeNvcompLz4Api&&) = delete;
    FakeNvcompLz4Api& operator=(FakeNvcompLz4Api&&) = delete;
    void inject(GpuCodecFault fault, std::uint64_t occurrence = 1U) noexcept;
    [[nodiscard]] FakeGpuCodecSnapshot snapshot() const noexcept;
    [[nodiscard]] Result<detail::GpuCodecRequirements>
    requirements(detail::GpuCodecDirection direction, ByteSize logical_size) noexcept override;
    [[nodiscard]] Result<void> initialize(detail::CudaContext context) noexcept override;
    [[nodiscard]] Result<void> launch(detail::GpuCodecJob job) noexcept override;
    [[nodiscard]] Result<void> synchronize() noexcept override;
    [[nodiscard]] Result<detail::GpuCodecResult> result() noexcept override;
    [[nodiscard]] Result<void> shutdown() noexcept override;

  private:
    [[nodiscard]] bool fault(GpuCodecFault point) noexcept;
    [[nodiscard]] bool correct_context() noexcept;
    void record(GpuCodecCall call) noexcept;
    void audit_launch(const detail::GpuCodecJob& job) noexcept;
    detail::CudaDriverApi& driver_;
    std::shared_ptr<FakeGpuCodecState> state_;
    detail::CudaContext context_{};
    detail::GpuCodecJob job_{};
    bool completed_{};
};

} // namespace vramz::test
