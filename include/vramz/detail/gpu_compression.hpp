#pragma once

#include "vramz/detail/cuda_driver.hpp"
#include "vramz/performance.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace vramz::detail {

inline constexpr ByteSize gpu_lz4_max_chunk{16ULL * 1024ULL * 1024ULL};
enum class GpuCodecDirection : std::uint8_t { compress, decompress };

struct GpuCodecRequirements final {
    ByteSize output_bound{};
    ByteSize temporary_bytes{};
    ByteSize input_alignment{};
    ByteSize output_alignment{};
    ByteSize temporary_alignment{};
};

// One batched LLIF element. Scalars/arrays share one charged device workspace.
// This is an internal Linux x86_64 wire layout, not an NVIDIA type or public API.
struct GpuBatchMetadata final {
    std::uint64_t input{};
    std::uint64_t input_size{};
    std::uint64_t output{};
    std::uint64_t output_capacity{};
    std::uint64_t output_size{};
    std::int32_t status{-1};
    std::uint32_t reserved{};
};
static_assert(sizeof(GpuBatchMetadata) == 48U);
static_assert(offsetof(GpuBatchMetadata, output_size) == 32U);
static_assert(offsetof(GpuBatchMetadata, status) == 40U);
static_assert(std::is_trivially_copyable_v<GpuBatchMetadata>);

struct GpuCompressionPlan final {
    GpuCodecRequirements requirements{};
    ByteSize output_charge{};
    ByteSize temporary_offset{};
    ByteSize workspace_charge{};
    ByteSize address_alignment{};
};
struct GpuPlanDimensions final {
    ByteSize logical_size{};
    ByteSize physical_granularity{};
};

struct GpuCodecJob final {
    GpuCodecDirection direction{};
    ByteSize logical_size{};
    DeviceAddress metadata{};
    DeviceAddress temporary{};
    ByteSize temporary_size{};
    GpuBatchMetadata host_metadata{};
};
static_assert(std::is_trivially_copyable_v<GpuCodecJob>);
static_assert(std::is_nothrow_copy_assignable_v<GpuCodecJob>);
struct GpuCodecResult final {
    ByteSize actual_size{};
    std::int32_t status{};
};
static_assert(std::is_trivially_copyable_v<GpuCodecResult>);
static_assert(std::is_nothrow_constructible_v<Result<GpuCodecResult>, GpuCodecResult>);

// All execution happens within the owning backend's current context. The API borrows its
// driver; it owns only a non-default internal stream, never payload or workspace resources.
// launch retains a fixed-size copy of host_metadata until completion. It enqueues metadata
// publication to job.metadata BEFORE the codec on the same dedicated stream, never relying
// on default-stream ordering. Successful launch is pending; synchronize proves completion;
// result requires that proof. An uncertain enqueue failure is ambiguous, not safe rollback.
class GpuCompressionApi {
  public:
    virtual ~GpuCompressionApi() = default;
    [[nodiscard]] virtual Result<GpuCodecRequirements>
    requirements(GpuCodecDirection direction, ByteSize logical_size) noexcept = 0;
    [[nodiscard]] virtual Result<void> initialize(CudaContext context) noexcept = 0;
    [[nodiscard]] virtual Result<void> launch(GpuCodecJob job) noexcept = 0;
    [[nodiscard]] virtual Result<void> synchronize() noexcept = 0;
    [[nodiscard]] virtual Result<GpuCodecResult> result() noexcept = 0;
    [[nodiscard]] virtual Result<void> shutdown() noexcept = 0;
    [[nodiscard]] virtual PerformanceStats performance() const noexcept { return {}; }
};

[[nodiscard]] Result<GpuCompressionPlan>
make_gpu_compression_plan(GpuCodecRequirements requirements, GpuPlanDimensions dimensions) noexcept;

// Construction does not initialize CUDA or call nvCOMP. Compile/link artifact only.
[[nodiscard]] Result<std::unique_ptr<GpuCompressionApi>>
make_real_nvcomp_lz4_api(CudaDriverApi& driver) noexcept;

[[nodiscard]] Result<std::unique_ptr<GpuCompressionApi>>
make_timed_codec(std::unique_ptr<GpuCompressionApi>& inner) noexcept;

} // namespace vramz::detail
