#pragma once

#include "vramz/error.hpp"

#include <array>
#include <charconv>
#include <span>
#include <string_view>
#include <system_error>

namespace vramz::detail {

class SmokeJsonWriter final {
  public:
    struct TextField final {
        std::string_view key;
        std::string_view value;
    };
    explicit SmokeJsonWriter(std::span<char> output) noexcept : output_(output) { append("{"); }

    void text(const TextField& entry) noexcept {
        field(entry.key);
        quoted(entry.value);
    }
    void boolean(std::string_view key, bool value) noexcept {
        field(key);
        append(value ? "true" : "false");
    }
    void number(std::string_view key, std::uint64_t value) noexcept {
        field(key);
        integer(value);
    }
    void difference(std::string_view key, std::uint64_t left, std::uint64_t right,
                    bool known) noexcept {
        field(key);
        if (!known) {
            append("null");
        } else if (left >= right) {
            integer(left - right);
        } else {
            character('-');
            integer(right - left);
        }
    }
    // Diagnostic ratios only. Integer ownership/savings decisions never use floating point.
    void ratio(std::string_view key, std::uint64_t numerator, std::uint64_t denominator) noexcept {
        field(key);
        if (denominator == 0U) {
            append("null");
            return;
        }
        std::array<char, 32U> buffer{};
        const auto converted =
            std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                          static_cast<double>(numerator) / static_cast<double>(denominator),
                          std::chars_format::general, 9);
        if (converted.ec != std::errc{}) {
            truncated_ = true;
            return;
        }
        append({buffer.data(), static_cast<std::size_t>(converted.ptr - buffer.data())});
    }
    void signed_number(std::string_view key, std::int64_t value) noexcept {
        field(key);
        integer(value);
    }
    void nullable_number(std::string_view key, std::uint64_t value, bool known) noexcept {
        field(key);
        if (known) {
            integer(value);
        } else {
            append("null");
        }
    }
    void null(std::string_view key) noexcept {
        field(key);
        append("null");
    }
    void error(const Error& value) noexcept {
        field("error");
        append("{\"code\":");
        integer(static_cast<std::uint64_t>(value.code));
        append(",\"operation\":");
        integer(static_cast<std::uint64_t>(value.operation));
        append(",\"native_code\":");
        integer(value.native_code);
        append("}");
    }
    // Each entry is formatted in the same bounded buffer; no raw JSON or allocation.
    template <typename Entries, typename Formatter>
    void object_array(std::string_view key, const Entries& entries, Formatter formatter) noexcept {
        field(key);
        append("[");
        bool first_entry = true;
        for (const auto& entry : entries) {
            if (!first_entry) {
                character(',');
            }
            first_entry = false;
            character('{');
            first_ = true;
            formatter(*this, entry);
            character('}');
        }
        character(']');
        first_ = false;
    }
    [[nodiscard]] FormatResult finish() noexcept {
        append("}");
        if (!output_.empty()) {
            output_[written_] = '\0';
        }
        return FormatResult{written_, truncated_};
    }

  private:
    void character(char value) noexcept {
        if (!output_.empty() && written_ < output_.size() - 1U) {
            output_[written_++] = value;
        } else {
            truncated_ = true;
        }
    }
    void append(std::string_view text) noexcept {
        for (const char value : text) {
            character(value);
        }
    }
    void quoted(std::string_view text) noexcept {
        constexpr std::string_view hex = "0123456789abcdef";
        character('"');
        for (const char value : text) {
            const auto byte = static_cast<unsigned char>(value);
            if (value == '"' || value == '\\') {
                character('\\');
                character(value);
            } else if (byte < 0x20U || byte >= 0x7fU) {
                append("\\u00");
                character(hex[byte >> 4U]);
                character(hex[byte & 0x0fU]);
            } else {
                character(value);
            }
        }
        character('"');
    }
    template <typename Integer> void integer(Integer value) noexcept {
        std::array<char, 32U> buffer{};
        const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
        if (converted.ec != std::errc{}) {
            truncated_ = true;
            return;
        }
        append(std::string_view{buffer.data(),
                                static_cast<std::size_t>(converted.ptr - buffer.data())});
    }
    void field(std::string_view key) noexcept {
        if (!first_) {
            character(',');
        }
        first_ = false;
        quoted(key);
        character(':');
    }

    std::span<char> output_;
    std::size_t written_{};
    bool truncated_{};
    bool first_{true};
};

} // namespace vramz::detail
