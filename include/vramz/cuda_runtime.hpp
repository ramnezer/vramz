#pragma once

#include "vramz/runtime.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace vramz {

// Paths identify externally installed providers. Normal runtime never needs root.
// The application is responsible for trusting these files and arranging its local loader.
struct CudaRuntimeOptions final {
    bool allow_execution{};
    std::int32_t device_ordinal{};
    std::string_view driver_path{};
    std::string_view cudart_path{};
    std::string_view nvcomp_path{};
};

struct CudaRuntimeInfo final {
    std::array<char, 256U> device_name{};
    std::array<char, 64U> driver_release{};
    std::int32_t device_ordinal{};
    std::int32_t driver_api{};
    std::int32_t runtime_api{};
    std::string_view nvcomp_version{"5.3.0.16"};
    bool uva{};
    bool vmm{};
    ByteSize minimum_granularity{};
    ByteSize recommended_granularity{};
    ByteSize single_operation_admission{};
};

inline constexpr ByteSize physical_runtime_maximum{std::uint64_t{512U} * 1024U * 1024U};

} // namespace vramz
