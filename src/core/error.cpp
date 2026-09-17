#include "vramz/error.hpp"

#include <charconv>
#include <string_view>

namespace vramz {
namespace {

class BoundedWriter final {
  public:
    explicit BoundedWriter(std::span<char> output) noexcept : output_(output) {}

    void append(std::string_view text) noexcept {
        for (const char character : text) {
            if (written_ < output_.size()) {
                output_[written_] = character;
                ++written_;
            } else {
                truncated_ = true;
            }
        }
    }

    void append_number(std::uint64_t value) noexcept {
        char buffer[32]{};
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
        if (result.ec == std::errc{}) {
            append(std::string_view{buffer, static_cast<std::size_t>(result.ptr - buffer)});
        } else {
            truncated_ = true;
        }
    }

    [[nodiscard]] FormatResult finish() noexcept {
        if (!output_.empty()) {
            const auto terminator = (written_ < output_.size()) ? written_ : output_.size() - 1U;
            output_[terminator] = '\0';
            if (terminator != written_) {
                truncated_ = true;
            }
        } else if (written_ != 0U) {
            truncated_ = true;
        }
        return FormatResult{written_, truncated_};
    }

  private:
    std::span<char> output_;
    std::size_t written_{};
    bool truncated_{};
};

[[nodiscard]] constexpr std::string_view code_name(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::invalid_argument:
        return "invalid_argument";
    case ErrorCode::invalid_range:
        return "invalid_range";
    case ErrorCode::arithmetic_overflow:
        return "arithmetic_overflow";
    case ErrorCode::invalid_alignment:
        return "invalid_alignment";
    case ErrorCode::unsupported:
        return "unsupported";
    case ErrorCode::out_of_gpu_memory:
        return "out_of_gpu_memory";
    case ErrorCode::out_of_host_memory:
        return "out_of_host_memory";
    case ErrorCode::conflict:
        return "conflict";
    case ErrorCode::busy:
        return "busy";
    case ErrorCode::cancelled:
        return "cancelled";
    case ErrorCode::timeout:
        return "timeout";
    case ErrorCode::backend_failure:
        return "backend_failure";
    case ErrorCode::backend_contract_violation:
        return "backend_contract_violation";
    case ErrorCode::integrity_failure:
        return "integrity_failure";
    case ErrorCode::ambiguous_backend_state:
        return "ambiguous_backend_state";
    case ErrorCode::poisoned:
        return "poisoned";
    case ErrorCode::shutting_down:
        return "shutting_down";
    case ErrorCode::stale_handle:
        return "stale_handle";
    case ErrorCode::internal_invariant_violation:
        return "internal_invariant_violation";
    case ErrorCode::compression_not_beneficial:
        return "compression_not_beneficial";
    }
    return "unknown";
}

} // namespace

FormatResult format_error(const Error& error, std::span<char> destination) noexcept {
    BoundedWriter writer{destination};
    writer.append(code_name(error.code));
    writer.append(" operation=");
    writer.append_number(static_cast<std::uint64_t>(error.operation));
    writer.append(" object=");
    writer.append_number(error.object_id);
    writer.append(" detail=");
    writer.append_number(error.detail);
    return writer.finish();
}

} // namespace vramz
