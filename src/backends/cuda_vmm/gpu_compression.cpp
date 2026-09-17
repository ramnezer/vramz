#include "vramz/detail/gpu_compression.hpp"

#include "vramz/checked.hpp"

#include <bit>
#include <numeric>

namespace vramz::detail {

Result<GpuCompressionPlan> make_gpu_compression_plan(GpuCodecRequirements requirements,
                                                     GpuPlanDimensions dimensions) noexcept {
    const auto logical_size = dimensions.logical_size;
    const auto physical_granularity = dimensions.physical_granularity;
    if (logical_size > gpu_lz4_max_chunk) {
        return make_error(ErrorCode::unsupported, OperationId::compress);
    }
    const auto minimum = physical_granularity.value();
    std::uint64_t alignment = minimum;
    for (const auto required :
         {requirements.input_alignment, requirements.output_alignment,
          requirements.temporary_alignment, ByteSize{alignof(GpuBatchMetadata)}}) {
        if (!std::has_single_bit(required.value()) || !std::has_single_bit(minimum)) {
            return make_error(ErrorCode::invalid_alignment, OperationId::compress);
        }
        const auto common = checked_mul(alignment / std::gcd(alignment, required.value()),
                                        required.value(), OperationId::compress);
        if (!common) {
            return common.error();
        }
        alignment = common.value();
    }
    if (logical_size == ByteSize{}) {
        return GpuCompressionPlan{requirements, {}, {}, {}, ByteSize{alignment}};
    }
    if (requirements.output_bound == ByteSize{}) {
        return make_error(ErrorCode::backend_contract_violation, OperationId::compress);
    }
    const auto output =
        checked_align_up(requirements.output_bound.value(), minimum, OperationId::compress);
    const auto offset = checked_align_up(
        sizeof(GpuBatchMetadata), requirements.temporary_alignment.value(), OperationId::compress);
    if (!output || !offset) {
        return output ? offset.error() : output.error();
    }
    const auto total =
        checked_add(offset.value(), requirements.temporary_bytes.value(), OperationId::compress);
    if (!total) {
        return total.error();
    }
    const auto workspace = checked_align_up(total.value(), minimum, OperationId::compress);
    if (!workspace) {
        return workspace.error();
    }
    return GpuCompressionPlan{requirements, ByteSize{output.value()}, ByteSize{offset.value()},
                              ByteSize{workspace.value()}, ByteSize{alignment}};
}

} // namespace vramz::detail
