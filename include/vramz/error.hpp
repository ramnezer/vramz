#pragma once

#include "vramz/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace vramz {

enum class ErrorCode : std::uint16_t {
    invalid_argument,
    invalid_range,
    arithmetic_overflow,
    invalid_alignment,
    unsupported,
    out_of_gpu_memory,
    out_of_host_memory,
    conflict,
    busy,
    cancelled,
    timeout,
    backend_failure,
    backend_contract_violation,
    integrity_failure,
    ambiguous_backend_state,
    poisoned,
    shutting_down,
    stale_handle,
    internal_invariant_violation,
    compression_not_beneficial
};

enum class OperationId : std::uint16_t {
    unknown,
    runtime_create,
    allocate,
    acquire,
    migrate,
    compress,
    decompress,
    verify,
    release,
    close_lease,
    close_buffer,
    shutdown,
    budget_reserve,
    budget_transfer,
    format_diagnostic,
    defer_lease,
    query_completion,
    release_completion
};

enum class NativeErrorDomain : std::uint8_t {
    none,
    operating_system,
    compression_backend,
    internal_backend,
    cuda_driver,
    cuda_runtime
};

struct Error final {
    ErrorCode code{ErrorCode::internal_invariant_violation};
    OperationId operation{OperationId::unknown};
    BackendId backend{};
    NativeErrorDomain native_domain{NativeErrorDomain::none};
    std::int64_t native_code{};
    std::uint64_t object_id{};
    std::uint64_t detail{};

    auto operator<=>(const Error&) const = default;
};

static_assert(std::is_trivially_copyable_v<Error>);
static_assert(std::is_trivially_destructible_v<Error>);

struct AsyncErrorRecord final {
    Error error{};
    std::uint64_t sequence{};
    std::uint64_t repeat_count{};
};

struct FormatResult final {
    std::size_t written{};
    bool truncated{};
};

[[nodiscard]] FormatResult format_error(const Error& error, std::span<char> destination) noexcept;

[[nodiscard]] constexpr Error make_error(ErrorCode code, OperationId operation,
                                         std::uint64_t object_id = 0U,
                                         std::uint64_t detail = 0U) noexcept {
    return Error{code, operation, BackendId{}, NativeErrorDomain::none, 0, object_id, detail};
}

} // namespace vramz
