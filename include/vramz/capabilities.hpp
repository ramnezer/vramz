#pragma once

#include "vramz/types.hpp"

#include <cstdint>
#include <string_view>

namespace vramz {

enum class HardwareValidationState : std::uint8_t {
    not_tested,
    smoke_validated,
    integration_validated,
    performance_validated
};

struct BuildCapabilities final {
    std::string_view version{};
    std::string_view cpu_lz4_version{};
    std::string_view cuda_development_version{};
    std::string_view nvcomp_development_version{};
    bool cpu_lz4_compiled{};
    bool cuda_adapter_compiled{};
    bool nvcomp_adapter_compiled{};
    bool real_gpu_execution_enabled{};
    HardwareValidationState hardware{HardwareValidationState::not_tested};
};

struct RuntimeCapabilities final {
    BackendId backend{};
    bool gpu_raw{};
    bool gpu_compressed{};
    bool host_raw{};
    bool host_compressed{};
    bool stable_virtual_address{};
    bool compression_available{};
    bool external_async_completion{};
    ByteSize minimum_allocation_granularity{};
    ByteSize recommended_allocation_granularity{};
    ByteSize maximum_chunk_size{};
    HardwareValidationState hardware{HardwareValidationState::not_tested};
};

// Build metadata only: never loads a GPU library, probes a device, or initializes a driver.
[[nodiscard]] BuildCapabilities build_capabilities() noexcept;

} // namespace vramz
