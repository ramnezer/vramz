#include "vramz/detail/cuda_compatibility.hpp"
#include "vramz/detail/gpu_compression.hpp"

#include <cuda_runtime_api.h>
#include <nvcomp/lz4.h>
#include <nvcomp/version.h>

#include <exception>
#include <limits>
#include <new>

namespace vramz::detail {
namespace {

static_assert(NVCOMP_VER_MAJOR == 5 && NVCOMP_VER_MINOR == 3 && NVCOMP_VER_PATCH == 0 &&
              NVCOMP_VER_BUILD == 16);
static_assert(cudaErrorInsufficientDriver == 35 && cudaErrorCallRequiresNewerDriver == 36 &&
              cudaErrorUnsupportedPtxVersion == 222 && cudaErrorNotSupported == 801 &&
              cudaErrorSystemDriverMismatch == 803 && cudaErrorCompatNotSupportedOnDevice == 804);
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));
static_assert(sizeof(void*) == sizeof(std::uint64_t));
static_assert(sizeof(nvcompStatus_t) == sizeof(std::int32_t));
static_assert(nvcompLZ4CompressionMaxAllowedChunkSize == gpu_lz4_max_chunk.value());

[[nodiscard]] constexpr nvcompBatchedLZ4CompressOpts_t compression_options() noexcept {
    nvcompBatchedLZ4CompressOpts_t options{};
    options.data_type = NVCOMP_TYPE_CHAR;
    options.bitshuffle_mode = NVCOMP_BITSHUFFLE_NONE;
    return options;
}
[[nodiscard]] constexpr nvcompBatchedLZ4DecompressOpts_t decompression_options() noexcept {
    nvcompBatchedLZ4DecompressOpts_t options{};
    options.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    options.data_type = NVCOMP_TYPE_CHAR;
    options.bitshuffle_mode = NVCOMP_BITSHUFFLE_NONE;
    return options;
}
static_assert(compression_options().data_type == NVCOMP_TYPE_CHAR);
static_assert(compression_options().bitshuffle_mode == NVCOMP_BITSHUFFLE_NONE);
static_assert(decompression_options().backend == NVCOMP_DECOMPRESS_BACKEND_CUDA);

[[nodiscard]] Error api_error(std::int64_t native, OperationId operation,
                              bool ambiguous = false) noexcept {
    return Error{ambiguous ? ErrorCode::ambiguous_backend_state : ErrorCode::backend_failure,
                 operation,
                 BackendId{3U},
                 NativeErrorDomain::compression_backend,
                 native,
                 0U,
                 0U};
}
template <class Pointer> [[nodiscard]] Pointer device_pointer(std::uint64_t address) noexcept {
    // LLIF requires typed device-pointer arrays as pointer arguments. These opaque GPU addresses
    // are never dereferenced by host C++; the conversion is the required NVIDIA ABI boundary.
    return reinterpret_cast<Pointer>(address); // NOLINT(performance-no-int-to-ptr)
}

class RealNvcompLz4Api final : public GpuCompressionApi {
  public:
    explicit RealNvcompLz4Api(CudaDriverApi& driver) noexcept : driver_(driver) {}
    ~RealNvcompLz4Api() override {
        // The backend must close the stream in its context before releasing that context.
        if (stream_ != nullptr) {
            std::terminate();
        }
    }
    RealNvcompLz4Api(const RealNvcompLz4Api&) = delete;
    RealNvcompLz4Api& operator=(const RealNvcompLz4Api&) = delete;
    RealNvcompLz4Api(RealNvcompLz4Api&&) = delete;
    RealNvcompLz4Api& operator=(RealNvcompLz4Api&&) = delete;

    Result<GpuCodecRequirements> requirements(GpuCodecDirection direction,
                                              ByteSize logical) noexcept override {
        const auto operation = direction == GpuCodecDirection::compress ? OperationId::compress
                                                                        : OperationId::decompress;
        if (logical > gpu_lz4_max_chunk) {
            return make_error(ErrorCode::unsupported, operation);
        }
        if (logical == ByteSize{}) {
            return GpuCodecRequirements{{}, {}, ByteSize{1U}, ByteSize{1U}, ByteSize{1U}};
        }
        const auto bytes = static_cast<std::size_t>(logical.value());
        nvcompAlignmentRequirements_t alignments{};
        std::size_t temporary{};
        std::size_t bound = bytes;
        nvcompStatus_t queried{};
        if (direction == GpuCodecDirection::compress) {
            queried =
                nvcompBatchedLZ4CompressGetRequiredAlignments(compression_options(), &alignments);
            if (queried == nvcompSuccess) {
                queried = nvcompBatchedLZ4CompressGetTempSizeAsync(1U, bytes, compression_options(),
                                                                   &temporary, bytes);
            }
            if (queried == nvcompSuccess) {
                queried = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
                    bytes, compression_options(), &bound);
            }
        } else {
            queried = nvcompBatchedLZ4DecompressGetRequiredAlignments(decompression_options(),
                                                                      &alignments);
            if (queried == nvcompSuccess) {
                queried = nvcompBatchedLZ4DecompressGetTempSizeAsync(
                    1U, bytes, decompression_options(), &temporary, bytes);
            }
        }
        if (queried != nvcompSuccess) {
            return api_error(queried, operation);
        }
        return GpuCodecRequirements{ByteSize{bound}, ByteSize{temporary},
                                    ByteSize{alignments.input}, ByteSize{alignments.output},
                                    ByteSize{alignments.temp}};
    }

    Result<void> initialize(CudaContext context) noexcept override {
        if (stream_ != nullptr) {
            return make_error(ErrorCode::busy, OperationId::runtime_create);
        }
        const auto current = driver_.context_is_current(context);
        if (!current || !current.value()) {
            return make_error(ErrorCode::backend_contract_violation, OperationId::runtime_create);
        }
        const auto created = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (created != cudaSuccess) {
            return cuda_runtime_error(created, OperationId::runtime_create, true);
        }
        if (stream_ == nullptr) {
            return make_error(ErrorCode::ambiguous_backend_state, OperationId::runtime_create);
        }
        context_ = context;
        return {};
    }
    Result<void> launch(GpuCodecJob job) noexcept override {
        const auto operation = job.direction == GpuCodecDirection::compress
                                   ? OperationId::compress
                                   : OperationId::decompress;
        const auto current = driver_.context_is_current(context_);
        if (!current || !current.value() || stream_ == nullptr || pending_ ||
            job.logical_size == ByteSize{} || job.logical_size > gpu_lz4_max_chunk ||
            job.metadata == DeviceAddress{} ||
            job.metadata.value() % alignof(GpuBatchMetadata) != 0U ||
            job.metadata.value() >
                std::numeric_limits<std::uint64_t>::max() - sizeof(GpuBatchMetadata) ||
            job.temporary_size.value() > std::numeric_limits<std::size_t>::max()) {
            return make_error(ErrorCode::backend_contract_violation, operation);
        }
        const auto base = job.metadata.value();
        const auto inputs =
            device_pointer<const void* const*>(base + offsetof(GpuBatchMetadata, input));
        const auto input_sizes =
            device_pointer<const std::size_t*>(base + offsetof(GpuBatchMetadata, input_size));
        const auto outputs =
            device_pointer<void* const*>(base + offsetof(GpuBatchMetadata, output));
        const auto sizes =
            device_pointer<std::size_t*>(base + offsetof(GpuBatchMetadata, output_size));
        const auto statuses =
            device_pointer<nvcompStatus_t*>(base + offsetof(GpuBatchMetadata, status));
        const auto temporary = device_pointer<void*>(job.temporary.value());
        const auto temporary_size = static_cast<std::size_t>(job.temporary_size.value());
        // This non-movable adapter owns the submitted host values until stream completion.
        // Set pending before the first enqueue: an error cannot prove that nothing is in flight.
        job_ = job;
        pending_ = true;
        complete_ = false;
        const auto published =
            cudaMemcpyAsync(device_pointer<void*>(base), &job_.host_metadata,
                            sizeof(GpuBatchMetadata), cudaMemcpyHostToDevice, stream_);
        if (published != cudaSuccess) {
            // Do not launch LLIF or reuse the retained host buffer after uncertain submission.
            return cuda_runtime_error(published, operation, true);
        }
        // FIFO ordering on this exact stream makes all pointer/size/status fields visible
        // before LLIF consumes them; no legacy/default-stream or device-wide sync is assumed.
        const auto launched =
            job.direction == GpuCodecDirection::compress
                ? nvcompBatchedLZ4CompressAsync(inputs, input_sizes,
                                                static_cast<std::size_t>(job.logical_size.value()),
                                                1U, temporary, temporary_size, outputs, sizes,
                                                compression_options(), statuses, stream_)
                : nvcompBatchedLZ4DecompressAsync(
                      inputs, input_sizes,
                      device_pointer<const std::size_t*>(
                          base + offsetof(GpuBatchMetadata, output_capacity)),
                      sizes, 1U, temporary, temporary_size, outputs, decompression_options(),
                      statuses, stream_);
        if (launched != nvcompSuccess) {
            // LLIF errors do not generally certify that no stream work was enqueued.
            return api_error(launched, operation, true);
        }
        return {};
    }
    Result<void> synchronize() noexcept override {
        const auto current = driver_.context_is_current(context_);
        if (!current || !current.value() || !pending_) {
            return make_error(ErrorCode::ambiguous_backend_state, OperationId::migrate);
        }
        const auto completed = cudaStreamSynchronize(stream_);
        if (completed != cudaSuccess) {
            return cuda_runtime_error(completed, OperationId::migrate, true);
        }
        pending_ = false;
        complete_ = true;
        return {};
    }
    Result<GpuCodecResult> result() noexcept override {
        const auto current = driver_.context_is_current(context_);
        if (!current || !current.value()) {
            return make_error(ErrorCode::backend_contract_violation, OperationId::verify);
        }
        if (!complete_ || pending_) {
            return make_error(ErrorCode::busy, OperationId::verify);
        }
        GpuBatchMetadata metadata{};
        const auto copied = driver_.copy_from_device(
            std::as_writable_bytes(std::span{&metadata, 1U}), job_.metadata);
        if (!copied) {
            return copied.error();
        }
        return GpuCodecResult{ByteSize{metadata.output_size}, metadata.status};
    }
    Result<void> shutdown() noexcept override {
        if (stream_ == nullptr) {
            return {};
        }
        const auto current = driver_.context_is_current(context_);
        if (!current || !current.value() || pending_) {
            return make_error(ErrorCode::ambiguous_backend_state, OperationId::shutdown);
        }
        const auto destroyed = cudaStreamDestroy(stream_);
        if (destroyed != cudaSuccess) {
            return cuda_runtime_error(destroyed, OperationId::shutdown, true);
        }
        stream_ = nullptr;
        context_ = {};
        complete_ = false;
        return {};
    }

  private:
    CudaDriverApi& driver_;
    cudaStream_t stream_{};
    CudaContext context_{};
    GpuCodecJob job_{};
    bool pending_{};
    bool complete_{};
};

} // namespace

Result<std::unique_ptr<GpuCompressionApi>>
make_real_nvcomp_lz4_api(CudaDriverApi& driver) noexcept {
    std::unique_ptr<GpuCompressionApi> api;
    try {
        api = std::make_unique<RealNvcompLz4Api>(driver);
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::runtime_create);
    }
    return api;
}

} // namespace vramz::detail
