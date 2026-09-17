#include "vramz/detail/cuda_driver.hpp"

#include <cuda.h>

#include <limits>
#include <new>
#include <unistd.h>

namespace vramz::detail {
namespace {

[[nodiscard]] Error driver_error(CUresult result, OperationId operation, bool mutation) noexcept {
    return Error{mutation ? ErrorCode::ambiguous_backend_state : ErrorCode::backend_failure,
                 operation,
                 BackendId{2U},
                 NativeErrorDomain::cuda_driver,
                 static_cast<std::int64_t>(result),
                 0U,
                 0U};
}

[[nodiscard]] Result<void> status(CUresult result, OperationId operation, bool mutation) noexcept {
    if (result != CUDA_SUCCESS) {
        return driver_error(result, operation, mutation);
    }
    return {};
}

[[nodiscard]] Result<std::size_t> native_size(ByteSize size) noexcept {
    if (size.value() > std::numeric_limits<std::size_t>::max()) {
        return make_error(ErrorCode::arithmetic_overflow, OperationId::allocate);
    }
    return static_cast<std::size_t>(size.value());
}

[[nodiscard]] CUmemAllocationProp allocation_properties(CudaDevice device) noexcept {
    CUmemAllocationProp properties{};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    properties.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
    return properties;
}

class RealCudaDriverApi final : public CudaDriverApi {
  public:
    RealCudaDriverApi() noexcept = default;
    RealCudaDriverApi(const RealCudaDriverApi&) = delete;
    RealCudaDriverApi& operator=(const RealCudaDriverApi&) = delete;
    RealCudaDriverApi(RealCudaDriverApi&&) = delete;
    RealCudaDriverApi& operator=(RealCudaDriverApi&&) = delete;
    Result<void> initialize() noexcept override {
        return status(cuInit(0U), OperationId::runtime_create, false);
    }
    Result<std::int32_t> version() noexcept override {
        int value{};
        const auto result = cuDriverGetVersion(&value);
        if (result != CUDA_SUCCESS) {
            return driver_error(result, OperationId::runtime_create, false);
        }
        return value;
    }
    Result<CudaDevice> device(std::int32_t ordinal) noexcept override {
        CUdevice value{};
        const auto result = cuDeviceGet(&value, ordinal);
        if (result != CUDA_SUCCESS) {
            return driver_error(result, OperationId::runtime_create, false);
        }
        return value;
    }
    Result<CudaDeviceInfo> device_info(CudaDevice selected) noexcept override {
        CudaDeviceInfo info{};
        static_assert(info.name.size() <=
                      static_cast<std::size_t>(std::numeric_limits<int>::max()));
        const auto named =
            cuDeviceGetName(info.name.data(), static_cast<int>(info.name.size()), selected);
        if (named != CUDA_SUCCESS) {
            return driver_error(named, OperationId::runtime_create, false);
        }
        info.name.back() = '\0';
        int compute_mode{};
        const auto mode =
            cuDeviceGetAttribute(&compute_mode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, selected);
        if (mode != CUDA_SUCCESS) {
            return driver_error(mode, OperationId::runtime_create, false);
        }
        info.compute_mode = compute_mode;
        return info;
    }
    Result<bool> attribute(CudaDevice selected, CudaAttribute attribute_id) noexcept override {
        int value{};
        const auto native = attribute_id == CudaAttribute::unified_addressing
                                ? CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
                                : CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED;
        const auto result = cuDeviceGetAttribute(&value, native, selected);
        if (result != CUDA_SUCCESS) {
            return driver_error(result, OperationId::runtime_create, false);
        }
        return value != 0;
    }
    Result<ByteSize> host_page_size() noexcept override {
        const long value = sysconf(_SC_PAGESIZE);
        if (value <= 0) {
            return make_error(ErrorCode::unsupported, OperationId::runtime_create);
        }
        return ByteSize{static_cast<std::uint64_t>(value)};
    }
    Result<ByteSize> granularity(CudaDevice selected, bool recommended) noexcept override {
        const auto properties = allocation_properties(selected);
        std::size_t value{};
        const auto result = cuMemGetAllocationGranularity(
            &value, &properties,
            recommended ? CU_MEM_ALLOC_GRANULARITY_RECOMMENDED : CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (result != CUDA_SUCCESS) {
            return driver_error(result, OperationId::runtime_create, false);
        }
        return ByteSize{value};
    }
    Result<CudaContext> retain_primary(CudaDevice selected) noexcept override {
        if (context_ != nullptr) {
            return make_error(ErrorCode::conflict, OperationId::runtime_create);
        }
        CUcontext value{};
        const auto result = cuDevicePrimaryCtxRetain(&value, selected);
        if (result != CUDA_SUCCESS) {
            return driver_error(result, OperationId::runtime_create, true);
        }
        if (value == nullptr) {
            return make_error(ErrorCode::ambiguous_backend_state, OperationId::runtime_create);
        }
        context_ = value;
        context_device_ = selected;
        return CudaContext{1U};
    }
    Result<void> release_primary(CudaDevice selected) noexcept override {
        if (context_ == nullptr || selected != context_device_) {
            return make_error(ErrorCode::stale_handle, OperationId::shutdown);
        }
        const auto released =
            status(cuDevicePrimaryCtxRelease(selected), OperationId::shutdown, true);
        if (released) {
            context_ = nullptr;
        }
        return released;
    }
    Result<void> push_context(CudaContext context) noexcept override {
        if (context != CudaContext{1U} || context_ == nullptr) {
            return make_error(ErrorCode::stale_handle, OperationId::acquire);
        }
        return status(cuCtxPushCurrent(context_), OperationId::acquire, true);
    }
    Result<CudaContext> pop_context() noexcept override {
        CUcontext value{};
        const auto result = cuCtxPopCurrent(&value);
        if (result != CUDA_SUCCESS) {
            return driver_error(result, OperationId::release, true);
        }
        if (context_ == nullptr || value != context_) {
            return make_error(ErrorCode::ambiguous_backend_state, OperationId::release);
        }
        return CudaContext{1U};
    }
    Result<bool> context_is_current(CudaContext context) noexcept override {
        CUcontext current{};
        const auto result = cuCtxGetCurrent(&current);
        if (result != CUDA_SUCCESS) {
            return driver_error(result, OperationId::verify, false);
        }
        return context == CudaContext{1U} && context_ != nullptr && current == context_;
    }
    Result<DeviceAddress> reserve(CudaAddressRequest request) noexcept override {
        const auto bytes = native_size(request.size);
        const auto aligned = native_size(request.alignment);
        if (!bytes || !aligned) {
            return make_error(ErrorCode::arithmetic_overflow, OperationId::allocate);
        }
        CUdeviceptr value{};
        const auto result = cuMemAddressReserve(&value, bytes.value(), aligned.value(), 0U, 0U);
        if (result != CUDA_SUCCESS) {
            auto error = driver_error(result, OperationId::allocate, true);
            if (result == CUDA_ERROR_OUT_OF_MEMORY && value == 0U) {
                error.code = ErrorCode::out_of_gpu_memory;
            }
            return error;
        }
        return DeviceAddress{value};
    }
    Result<void> free_address(DeviceAddress address, ByteSize size) noexcept override {
        const auto bytes = native_size(size);
        if (!bytes) {
            return bytes.error();
        }
        return status(cuMemAddressFree(address.value(), bytes.value()), OperationId::release, true);
    }
    Result<CudaPhysicalHandle> create(ByteSize size, CudaDevice selected) noexcept override {
        const auto bytes = native_size(size);
        if (!bytes) {
            return bytes.error();
        }
        const auto properties = allocation_properties(selected);
        CUmemGenericAllocationHandle handle{};
        const auto result = cuMemCreate(&handle, bytes.value(), &properties, 0U);
        if (result != CUDA_SUCCESS) {
            auto error = driver_error(result, OperationId::allocate, true);
            if (result == CUDA_ERROR_OUT_OF_MEMORY && handle == 0U) {
                error.code = ErrorCode::out_of_gpu_memory;
            }
            return error;
        }
        return CudaPhysicalHandle{handle};
    }
    Result<CudaAllocationProperties> properties(CudaPhysicalHandle handle) noexcept override {
        CUmemAllocationProp properties{};
        const auto result = cuMemGetAllocationPropertiesFromHandle(&properties, handle.value());
        if (result != CUDA_SUCCESS) {
            return driver_error(result, OperationId::verify, false);
        }
        return CudaAllocationProperties{
            properties.location.id, properties.location.type == CU_MEM_LOCATION_TYPE_DEVICE,
            properties.type == CU_MEM_ALLOCATION_TYPE_PINNED,
            properties.requestedHandleTypes == CU_MEM_HANDLE_TYPE_NONE,
            properties.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_NONE};
    }
    Result<void> map(DeviceAddress address, ByteSize size,
                     CudaPhysicalHandle handle) noexcept override {
        const auto bytes = native_size(size);
        if (!bytes) {
            return bytes.error();
        }
        return status(cuMemMap(address.value(), bytes.value(), 0U, handle.value(), 0U),
                      OperationId::allocate, true);
    }
    Result<void> set_access(DeviceAddress address, ByteSize size,
                            CudaDevice selected) noexcept override {
        const auto bytes = native_size(size);
        if (!bytes) {
            return bytes.error();
        }
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = selected;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        return status(cuMemSetAccess(address.value(), bytes.value(), &access, 1U),
                      OperationId::allocate, true);
    }
    Result<bool> has_read_write_access(DeviceAddress address,
                                       CudaDevice selected) noexcept override {
        CUmemLocation location{};
        location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        location.id = selected;
        unsigned long long flags{};
        const auto result = cuMemGetAccess(&flags, &location, address.value());
        if (result != CUDA_SUCCESS) {
            return driver_error(result, OperationId::verify, false);
        }
        return flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    Result<CudaPhysicalHandle> retain_mapping(DeviceAddress address) noexcept override {
        static_assert(sizeof(std::uintptr_t) >= sizeof(CUdeviceptr));
        CUmemGenericAllocationHandle handle{};
        // NVIDIA requires a void* for a CUdeviceptr here, not a host-dereferenceable pointer.
        // This documented Driver ABI conversion is unavoidable and never dereferenced by VRAMZ.
        const auto numeric = static_cast<std::uintptr_t>(address.value());
        void* const ptr = reinterpret_cast<void*>(numeric); // NOLINT(performance-no-int-to-ptr)
        const auto result = cuMemRetainAllocationHandle(&handle, ptr);
        if (result != CUDA_SUCCESS) {
            return driver_error(result, OperationId::verify, true);
        }
        return CudaPhysicalHandle{handle};
    }
    Result<void> unmap(DeviceAddress address, ByteSize size) noexcept override {
        const auto bytes = native_size(size);
        if (!bytes) {
            return bytes.error();
        }
        return status(cuMemUnmap(address.value(), bytes.value()), OperationId::release, true);
    }
    Result<void> release(CudaPhysicalHandle handle) noexcept override {
        return status(cuMemRelease(handle.value()), OperationId::release, true);
    }
    Result<void> copy_to_device(DeviceAddress destination,
                                std::span<const std::byte> source) noexcept override {
        return status(cuMemcpyHtoD(destination.value(), source.data(), source.size()),
                      OperationId::acquire, true);
    }
    Result<void> copy_from_device(std::span<std::byte> destination,
                                  DeviceAddress source) noexcept override {
        return status(cuMemcpyDtoH(destination.data(), source.value(), destination.size()),
                      OperationId::verify, false);
    }

    Result<void> copy_device_to_device(DeviceAddress destination, DeviceAddress source,
                                       ByteSize size) noexcept override {
        const auto bytes = native_size(size);
        if (!bytes) {
            return bytes.error();
        }
        return status(cuMemcpyDtoD(destination.value(), source.value(), bytes.value()),
                      OperationId::migrate, true);
    }

  private:
    // Native token lookup only; the retained reference belongs to CudaVmmBackend.
    // Its context guard serializes use and releases it before destroying the adapter.
    CUcontext context_{};
    CudaDevice context_device_{};
};

} // namespace

Result<std::unique_ptr<CudaDriverApi>> make_real_cuda_driver() noexcept {
    try {
        return std::unique_ptr<CudaDriverApi>{std::make_unique<RealCudaDriverApi>()};
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::runtime_create);
    }
}

} // namespace vramz::detail
