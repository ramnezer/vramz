#include "fake_nvcomp.hpp"

#include "vramz/checked.hpp"
#include "vramz/detail/compression.hpp"
#include "vramz/detail/cuda_compatibility.hpp"
#include "vramz/saturating.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <new>
#include <vector>

namespace vramz::test {
using detail::GpuBatchMetadata;
using detail::GpuCodecDirection;
using detail::GpuCodecJob;
using detail::GpuCodecRequirements;
using detail::GpuCodecResult;
namespace {
[[nodiscard]] Result<ByteSize> literal_prefix_block(std::span<const std::byte> input,
                                                    std::span<std::byte> output, ByteSize prefix) {
    // Prepending real source literals to the first sequence preserves every suffix match
    // offset. This is a valid LZ4 block, never padding, a fake stored size or fake charge.
    if (prefix.value() >= input.size()) {
        return make_error(ErrorCode::unsupported, OperationId::compress);
    }
    const auto prefix_size = static_cast<std::size_t>(prefix.value());
    const auto suffix = input.subspan(prefix_size);
    std::vector<std::byte> encoded(output.size());
    const auto compressed = detail::cpu_compression_codec().compress(suffix, encoded);
    if (!compressed || compressed.value() == ByteSize{}) {
        return compressed;
    }
    const auto encoded_size = static_cast<std::size_t>(compressed.value().value());
    const auto token = std::to_integer<unsigned int>(encoded.front());
    std::size_t literals = token >> 4U;
    std::size_t header = 1U;
    if (literals == 15U) {
        unsigned int extension{};
        do {
            if (header >= encoded_size) {
                return make_error(ErrorCode::backend_contract_violation, OperationId::compress);
            }
            extension = std::to_integer<unsigned int>(encoded[header++]);
            literals += extension;
        } while (extension == 255U);
    }
    if (literals > suffix.size() || literals > encoded_size - header) {
        return make_error(ErrorCode::backend_contract_violation, OperationId::compress);
    }
    const auto total_literals = prefix_size + literals;
    const auto length_bytes = total_literals < 15U ? 0U : 1U + (total_literals - 15U) / 255U;
    const auto remainder = encoded_size - header;
    if (1U + length_bytes + prefix_size > output.size() ||
        remainder > output.size() - (1U + length_bytes + prefix_size)) {
        return make_error(ErrorCode::unsupported, OperationId::compress);
    }
    std::size_t offset{};
    output[offset++] =
        static_cast<std::byte>((std::min(total_literals, std::size_t{15U}) << 4U) | (token & 15U));
    if (total_literals >= 15U) {
        auto remaining = total_literals - 15U;
        while (remaining >= 255U) {
            output[offset++] = std::byte{255U};
            remaining -= 255U;
        }
        output[offset++] = static_cast<std::byte>(remaining);
    }
    std::ranges::copy(input.first(prefix_size),
                      output.begin() + static_cast<std::ptrdiff_t>(offset));
    offset += prefix_size;
    std::ranges::copy(std::span{encoded}.subspan(header, remainder),
                      output.begin() + static_cast<std::ptrdiff_t>(offset));
    return ByteSize{offset + remainder};
}
[[nodiscard]] Result<ByteSize> literal_block(std::span<const std::byte> input,
                                             std::span<std::byte> output) noexcept {
    // A legal worst-case LZ4 block: one final literal sequence, no matches. This
    // models a non-beneficial codec result without fabricating sizes or ownership.
    if (input.size() < 15U || output.size() < input.size() ||
        output.size() - input.size() < 2U + (input.size() - 15U) / 255U) {
        return make_error(ErrorCode::unsupported, OperationId::compress);
    }
    std::size_t offset{};
    output[offset++] = std::byte{0xf0U};
    auto remaining = input.size() - 15U;
    while (remaining >= 255U) {
        output[offset++] = std::byte{0xffU};
        remaining -= 255U;
    }
    output[offset++] = static_cast<std::byte>(remaining);
    std::ranges::copy(input, output.begin() + static_cast<std::ptrdiff_t>(offset));
    return ByteSize{offset + input.size()};
}
[[nodiscard]] Error failed(OperationId operation, bool ambiguous = false) noexcept {
    return Error{ambiguous ? ErrorCode::ambiguous_backend_state : ErrorCode::backend_failure,
                 operation,
                 BackendId{3U},
                 NativeErrorDomain::compression_backend,
                 1,
                 0U,
                 0U};
}
} // namespace

FakeNvcompLz4Api::FakeNvcompLz4Api(detail::CudaDriverApi& driver,
                                   std::shared_ptr<FakeGpuCodecState> state) noexcept
    : driver_(driver), state_(std::move(state)) {
    if (!state_) {
        std::terminate();
    }
}
FakeNvcompLz4Api::~FakeNvcompLz4Api() {
    const std::scoped_lock lock{state_->mutex};
    if (state_->observation.streams != 0U) {
        std::terminate();
    }
}
void FakeNvcompLz4Api::inject(GpuCodecFault point, std::uint64_t occurrence) noexcept {
    const std::scoped_lock lock{state_->mutex};
    if (point == GpuCodecFault::count || occurrence == 0U) {
        std::terminate();
    }
    state_->countdowns[static_cast<std::size_t>(point)] = occurrence;
}
bool FakeNvcompLz4Api::fault(GpuCodecFault point) noexcept {
    auto& remaining = state_->countdowns[static_cast<std::size_t>(point)];
    if (remaining == 0U) {
        return false;
    }
    --remaining;
    return remaining == 0U;
}
FakeGpuCodecSnapshot FakeNvcompLz4Api::snapshot() const noexcept {
    const std::scoped_lock lock{state_->mutex};
    return state_->observation;
}
bool FakeNvcompLz4Api::correct_context() noexcept {
    const auto current = driver_.context_is_current(context_);
    return current && current.value();
}
void FakeNvcompLz4Api::record(GpuCodecCall call) noexcept {
    auto& observation = state_->observation;
    if (observation.log_size < observation.log.size()) {
        observation.log[observation.log_size] = call;
        ++observation.log_size;
    } else {
        saturating_increment(observation.dropped_calls);
    }
}
void FakeNvcompLz4Api::audit_launch(const GpuCodecJob& job) noexcept {
    const auto cuda = state_->cuda_audit.lock();
    if (!cuda) {
        return;
    }
    if (state_->launch_count == state_->launches.size()) {
        std::terminate();
    }
    const std::scoped_lock lock{cuda->mutex};
    auto& audit = state_->launches[state_->launch_count++];
    audit.direction = job.direction;
    audit.metadata = job.host_metadata;
    audit.cuda_log_position = cuda->log_size;
    audit.owned_bytes = ByteSize{cuda->owned_bytes};
    for (const auto& physical : cuda->physical) {
        if (physical.handle == detail::CudaPhysicalHandle{}) {
            continue;
        }
        const auto mapping = std::ranges::find_if(
            cuda->mappings, [&](const auto& item) { return item.handle == physical.handle; });
        audit.allocations[audit.allocation_count++] = FakeGpuCodecState::LiveAllocation{
            physical.handle, mapping == cuda->mappings.end() ? DeviceAddress{} : mapping->address,
            ByteSize{physical.bytes.size()}};
    }
}
Result<GpuCodecRequirements> FakeNvcompLz4Api::requirements(GpuCodecDirection direction,
                                                            ByteSize logical) noexcept {
    const std::scoped_lock lock{state_->mutex};
    saturating_increment(state_->observation.plans);
    if (logical > detail::gpu_lz4_max_chunk) {
        return make_error(ErrorCode::unsupported, OperationId::compress);
    }
    auto bound = direction == GpuCodecDirection::compress
                     ? detail::cpu_compression_codec().maximum_compressed_size(logical)
                     : Result<ByteSize>{logical};
    if (!bound) {
        return bound.error();
    }
    if (direction == GpuCodecDirection::compress) {
        bound = std::max(bound.value(), state_->config.compression_bound_floor);
    }
    auto alignment = state_->config.alignment;
    auto temporary = direction == GpuCodecDirection::compress ? state_->config.compression_temp
                                                              : state_->config.decompression_temp;
    if (fault(GpuCodecFault::plan_bound_overflow)) {
        bound = ByteSize{std::numeric_limits<std::uint64_t>::max()};
    }
    if (fault(GpuCodecFault::plan_alignment_zero)) {
        alignment = {};
    }
    if (fault(GpuCodecFault::plan_temp_overflow)) {
        temporary = ByteSize{std::numeric_limits<std::uint64_t>::max()};
    }
    return GpuCodecRequirements{bound.value(), temporary, alignment, alignment, alignment};
}
Result<void> FakeNvcompLz4Api::initialize(detail::CudaContext context) noexcept {
    const std::scoped_lock lock{state_->mutex};
    saturating_increment(state_->observation.initialization_attempts);
    const auto current = driver_.context_is_current(context);
    if (state_->observation.streams != 0U || !current || !current.value()) {
        return make_error(ErrorCode::backend_contract_violation, OperationId::runtime_create);
    }
    context_ = context;
    state_->observation.streams = 1U;
    if (state_->config.runtime_fault == GpuCodecRuntimeFault::initialize) {
        return detail::cuda_runtime_error(state_->config.runtime_native_error,
                                          OperationId::runtime_create, true);
    }
    return {};
}
Result<void> FakeNvcompLz4Api::launch(GpuCodecJob job) noexcept {
    const std::scoped_lock lock{state_->mutex};
    if (!correct_context() || state_->observation.streams != 1U || state_->observation.pending ||
        job.logical_size == ByteSize{} || job.logical_size > detail::gpu_lz4_max_chunk) {
        return make_error(ErrorCode::backend_contract_violation, OperationId::compress);
    }
    if (fault(GpuCodecFault::metadata_submission_failure) || fault(GpuCodecFault::launch_failure)) {
        return failed(OperationId::compress);
    }
    // Retain all host values by value. The caller's job/metadata may die immediately.
    audit_launch(job);
    job_ = job;
    completed_ = false;
    state_->observation.pending = true;
    state_->observation.metadata_publication_pending = true;
    record(GpuCodecCall::metadata_copy_enqueue);
    if (fault(GpuCodecFault::metadata_submission_ambiguous)) {
        return failed(OperationId::compress, true);
    }
    record(GpuCodecCall::nvcomp_launch);
    auto& count = job.direction == GpuCodecDirection::compress ? state_->observation.compressions
                                                               : state_->observation.decompressions;
    saturating_increment(count);
    if (fault(GpuCodecFault::launch_ambiguous)) {
        return failed(OperationId::compress, true);
    }
    if (state_->config.runtime_fault == GpuCodecRuntimeFault::launch) {
        return detail::cuda_runtime_error(state_->config.runtime_native_error,
                                          OperationId::compress, true);
    }
    const bool early_status = fault(GpuCodecFault::premature_status);
    const bool early_output = fault(GpuCodecFault::premature_output);
    if (early_status || early_output) {
        // Partial device visibility is not completion. result() must still reject early reads.
        auto metadata = job_.host_metadata;
        metadata.output_size = 1U;
        metadata.status = 0;
        const auto result_bytes = std::as_bytes(std::span{&metadata, 1U})
                                      .subspan(offsetof(GpuBatchMetadata, output_size));
        const auto result_address = checked_add(
            job.metadata.value(), offsetof(GpuBatchMetadata, output_size), OperationId::compress);
        if (!result_address ||
            (early_status &&
             !driver_.copy_to_device(DeviceAddress{result_address.value()}, result_bytes))) {
            return failed(OperationId::compress, true);
        }
        const std::array<std::byte, 1U> partial{std::byte{42U}};
        if (early_output && !driver_.copy_to_device(DeviceAddress{metadata.output}, partial)) {
            return failed(OperationId::compress, true);
        }
    }
    // Except for deliberate partial-visibility faults, output is produced only at completion.
    return {};
}
Result<void> FakeNvcompLz4Api::synchronize() noexcept {
    const std::scoped_lock lock{state_->mutex};
    if (!correct_context() || !state_->observation.pending) {
        return failed(OperationId::migrate, true);
    }
    record(GpuCodecCall::stream_synchronize);
    if (state_->config.runtime_fault == GpuCodecRuntimeFault::synchronize) {
        return detail::cuda_runtime_error(state_->config.runtime_native_error, OperationId::migrate,
                                          true);
    }
    if (fault(GpuCodecFault::completion_failure)) {
        return failed(OperationId::migrate, true);
    }
    const auto host_bytes = std::as_bytes(std::span{&job_.host_metadata, 1U});
    if (fault(GpuCodecFault::metadata_publication_partial)) {
        // A submitted copy can be partly visible without authorizing codec consumption.
        static_cast<void>(driver_.copy_to_device(job_.metadata, host_bytes.first(24U)));
        return failed(OperationId::migrate, true);
    }
    if (!driver_.copy_to_device(job_.metadata, host_bytes)) {
        return failed(OperationId::migrate, true);
    }
    state_->observation.metadata_publication_pending = false;
    saturating_increment(state_->observation.metadata_publications);
    GpuBatchMetadata metadata{};
    const auto read =
        driver_.copy_from_device(std::as_writable_bytes(std::span{&metadata, 1U}), job_.metadata);
    // Test memory is independently bounded, even when a malformed device size is injected.
    if (!read || metadata.input_size > 32ULL * 1024ULL * 1024ULL ||
        metadata.output_capacity > 32ULL * 1024ULL * 1024ULL) {
        return failed(OperationId::migrate, true);
    }
    state_->observation.consumed_metadata = metadata;
    saturating_increment(state_->observation.metadata_consumptions);
    try {
        std::vector<std::byte> input(static_cast<std::size_t>(metadata.input_size));
        std::vector<std::byte> output(static_cast<std::size_t>(metadata.output_capacity));
        if (!driver_.copy_from_device(input, DeviceAddress{metadata.input})) {
            return failed(OperationId::migrate, true);
        }
        auto converted =
            job_.direction == GpuCodecDirection::compress
                ? detail::cpu_compression_codec().compress(input, output)
                : detail::cpu_compression_codec().decompress(input, output, job_.logical_size);
        if (job_.direction == GpuCodecDirection::compress &&
            fault(GpuCodecFault::literal_only_output)) {
            converted = literal_block(input, output);
        }
        if (job_.direction == GpuCodecDirection::compress &&
            fault(GpuCodecFault::literal_prefix_output)) {
            converted = literal_prefix_block(input, output, state_->config.literal_prefix_bytes);
        }
        metadata.status = converted ? 0 : 1;
        metadata.output_size = converted ? converted.value().value() : 0U;
        if (fault(GpuCodecFault::status_failure)) {
            metadata.status = 1;
        }
        if (fault(GpuCodecFault::output_corruption) && !output.empty()) {
            output.front() ^= std::byte{1U};
        }
        if (fault(GpuCodecFault::crc_preserving_output_corruption) && output.size() >= 5U) {
            // This CRC32C-neutral difference isolates exact byte verification from CRC checks.
            constexpr std::array difference{std::byte{0x01U}, std::byte{0x03U}, std::byte{0x83U},
                                            std::byte{0x6bU}, std::byte{0xf2U}};
            for (std::size_t index = 0U; index < difference.size(); ++index) {
                output[index] ^= difference[index];
            }
        }
        if (!driver_.copy_to_device(DeviceAddress{metadata.output}, output)) {
            return failed(OperationId::migrate, true);
        }
        if (fault(GpuCodecFault::size_zero)) {
            metadata.output_size = 0U;
        }
        if (fault(GpuCodecFault::size_oversized)) {
            metadata.output_size = std::numeric_limits<std::uint64_t>::max();
        }
        if (fault(GpuCodecFault::size_mismatch)) {
            metadata.output_size = metadata.output_size > 1U ? metadata.output_size - 1U : 2U;
        }
        if (!driver_.copy_to_device(job_.metadata, std::as_bytes(std::span{&metadata, 1U}))) {
            return failed(OperationId::migrate, true);
        }
    } catch (const std::bad_alloc&) {
        return failed(OperationId::migrate, true);
    }
    state_->observation.pending = false;
    completed_ = true;
    saturating_increment(state_->observation.completions);
    return {};
}
Result<GpuCodecResult> FakeNvcompLz4Api::result() noexcept {
    const std::scoped_lock lock{state_->mutex};
    if (!correct_context() || !completed_ || state_->observation.pending) {
        saturating_increment(state_->observation.rejected_early_reads);
        return make_error(ErrorCode::busy, OperationId::verify);
    }
    record(GpuCodecCall::metadata_status_read);
    saturating_increment(state_->observation.result_reads);
    if (fault(GpuCodecFault::status_read_failure) || fault(GpuCodecFault::size_read_failure)) {
        return failed(OperationId::verify);
    }
    GpuBatchMetadata metadata{};
    const auto read =
        driver_.copy_from_device(std::as_writable_bytes(std::span{&metadata, 1U}), job_.metadata);
    if (!read) {
        return read.error();
    }
    return GpuCodecResult{ByteSize{metadata.output_size}, metadata.status};
}
Result<void> FakeNvcompLz4Api::shutdown() noexcept {
    const std::scoped_lock lock{state_->mutex};
    saturating_increment(state_->observation.shutdown_attempts);
    if (state_->observation.streams == 0U) {
        return {};
    }
    if (!correct_context() || state_->observation.pending ||
        fault(GpuCodecFault::shutdown_ambiguous)) {
        return failed(OperationId::shutdown, true);
    }
    if (fault(GpuCodecFault::shutdown_failure)) {
        return failed(OperationId::shutdown);
    }
    state_->observation.streams = 0U;
    context_ = {};
    completed_ = false;
    return {};
}

} // namespace vramz::test
