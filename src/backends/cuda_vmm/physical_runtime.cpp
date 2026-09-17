#include "vramz/cuda_runtime.hpp"

#include "vramz/detail/cuda_compatibility.hpp"
#include "vramz/detail/cuda_vmm_backend.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvcomp/lz4.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace vramz {
namespace {

[[nodiscard]] bool loaded_provider(const void* symbol, std::string_view expected,
                                   std::string_view filename) noexcept {
    if (expected.empty() || expected.size() >= PATH_MAX || expected.front() != '/' ||
        expected.find('\0') != std::string_view::npos ||
        expected.find("/stubs/") != std::string_view::npos || !expected.ends_with(filename)) {
        return false;
    }
    Dl_info loaded{};
    std::array<char, PATH_MAX> canonical{};
    if (::dladdr(symbol, &loaded) == 0 || loaded.dli_fname == nullptr ||
        ::realpath(loaded.dli_fname, canonical.data()) == nullptr ||
        std::string_view{canonical.data()} != expected) {
        return false;
    }
    struct stat file {};
    return ::stat(canonical.data(), &file) == 0 && S_ISREG(file.st_mode) &&
           (file.st_mode & S_IWOTH) == 0;
}

[[nodiscard]] Result<detail::NvidiaDriverRelease>
driver_release(std::array<char, 64U>& text) noexcept {
    const auto error = make_error(ErrorCode::unsupported, OperationId::runtime_create);
    const int fd = ::open("/sys/module/nvidia/version", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return error;
    }
    const auto count = ::read(fd, text.data(), text.size() - 1U);
    char extra{};
    const auto trailing = ::read(fd, &extra, 1U);
    const auto closed = ::close(fd);
    if (count <= 0 || trailing != 0 || closed != 0) {
        return error;
    }
    const auto size = static_cast<std::size_t>(count);
    if (text[size - 1U] == '\n') {
        text[size - 1U] = '\0';
    }
    return detail::parse_nvidia_driver_release(std::string_view{text.data()});
}

Result<std::unique_ptr<StorageBackend>> create_cuda_backend(RuntimeConfig& config,
                                                            const CudaRuntimeOptions& options,
                                                            CudaRuntimeInfo& info) noexcept {
    info = {};
    const auto unsupported = make_error(ErrorCode::unsupported, OperationId::runtime_create);
    const auto chunk =
        config.preferred_chunk_size.value_or(ByteSize{std::uint64_t{8U} * 1024U * 1024U});
    // v1's reference platform and bounded GPU-only execution are explicit, never inferred.
    if (!options.allow_execution || options.device_ordinal != 0 ||
        config.budgets.gpu.hard_limit == ByteSize{} ||
        config.budgets.gpu.hard_limit > physical_runtime_maximum ||
        config.budgets.gpu.migration_reserve > config.budgets.gpu.hard_limit ||
        config.budgets.gpu.soft_target > config.budgets.gpu.hard_limit ||
        config.budgets.host.hard_limit != ByteSize{} ||
        config.budgets.host.soft_target != ByteSize{} ||
        config.budgets.host.migration_reserve != ByteSize{} ||
        config.required_capabilities.host_tier ||
        config.policy.mode == PolicyMode::gpu_resident_with_host_fallback || chunk == ByteSize{} ||
        chunk > detail::gpu_lz4_max_chunk || std::getenv("LD_PRELOAD") != nullptr ||
        std::getenv("LD_AUDIT") != nullptr) {
        return unsupported;
    }
    const auto release = driver_release(info.driver_release);
    const auto filename = options.driver_path.substr(options.driver_path.find_last_of('/') + 1U);
    if (!release || release.value().branch < 580U || !filename.starts_with("libcuda.so.") ||
        filename.substr(std::string_view{"libcuda.so."}.size()) !=
            std::string_view{info.driver_release.data()} ||
        !options.driver_path.ends_with(std::string_view{info.driver_release.data()}) ||
        !loaded_provider(reinterpret_cast<const void*>(&cuInit), options.driver_path,
                         std::string_view{info.driver_release.data()}) ||
        !loaded_provider(reinterpret_cast<const void*>(&cudaRuntimeGetVersion), options.cudart_path,
                         "/libcudart.so.13.3.29") ||
        !loaded_provider(reinterpret_cast<const void*>(&nvcompBatchedLZ4CompressAsync),
                         options.nvcomp_path, "/libnvcomp.so.5.3.0.16")) {
        return unsupported;
    }
    config.preferred_chunk_size = chunk;
    auto driver = detail::make_real_cuda_driver();
    if (!driver) {
        return driver.error();
    }
    auto codec = detail::make_real_nvcomp_lz4_api(*driver.value());
    if (!codec) {
        return codec.error();
    }
    detail::CudaProbeInfo probe{};
    detail::CudaCompressionAdmission admission{};
    admission.logical_size = chunk;
    admission.physical_cap = config.budgets.gpu.hard_limit;
    admission.minimum_driver_version = 13000;
    admission.required_driver_family = 13;
    admission.required_device_name = "NVIDIA GeForce RTX 3060";
    auto backend = detail::CudaVmmBackend::create(std::move(driver).value(), 0,
                                                  std::move(codec).value(), &probe, &admission);
    if (!backend) {
        return backend.error();
    }
    int runtime_api{};
    const auto runtime_version = cudaRuntimeGetVersion(&runtime_api);
    if (runtime_version != cudaSuccess) {
        return detail::cuda_runtime_error(runtime_version, OperationId::runtime_create, false);
    }
    if (!detail::is_candidate(detail::compression_runtime_candidate(
            probe.driver_version, runtime_api, release.value()))) {
        return unsupported;
    }
    info.device_name = probe.device.name;
    info.driver_api = probe.driver_version;
    info.runtime_api = runtime_api;
    info.uva = probe.uva;
    info.vmm = probe.vmm;
    info.minimum_granularity = probe.minimum;
    info.recommended_granularity = probe.recommended;
    info.single_operation_admission = admission.peak;
    return std::unique_ptr<StorageBackend>{std::move(backend).value()};
}

} // namespace

Result<Runtime> Runtime::create_cuda(RuntimeConfig config, const CudaRuntimeOptions& options,
                                     CudaRuntimeInfo& info) noexcept {
    auto backend = create_cuda_backend(config, options, info);
    if (!backend) {
        return backend.error();
    }
    return create_backend(config, std::move(backend).value());
}

} // namespace vramz
