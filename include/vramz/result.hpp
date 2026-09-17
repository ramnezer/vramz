#pragma once

#include "vramz/error.hpp"

#include <cassert>
#include <type_traits>
#include <utility>
#include <variant>

namespace vramz {

template <typename T> class [[nodiscard]] Result final {
  public:
    constexpr Result(T value) noexcept(std::is_nothrow_move_constructible_v<T>)
        : storage_(std::in_place_type<T>, std::move(value)) {}
    constexpr Result(Error error) noexcept : storage_(std::in_place_type<Error>, error) {}

    [[nodiscard]] constexpr bool has_value() const noexcept {
        return std::holds_alternative<T>(storage_);
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return has_value(); }

    [[nodiscard]] constexpr T& value() & noexcept {
        auto* const value = std::get_if<T>(&storage_);
        assert(value != nullptr);
        return *value;
    }
    [[nodiscard]] constexpr const T& value() const& noexcept {
        const auto* const value = std::get_if<T>(&storage_);
        assert(value != nullptr);
        return *value;
    }
    [[nodiscard]] constexpr T&& value() && noexcept {
        auto* const value = std::get_if<T>(&storage_);
        assert(value != nullptr);
        return std::move(*value);
    }
    [[nodiscard]] constexpr Error error() const noexcept {
        const auto* const error = std::get_if<Error>(&storage_);
        assert(error != nullptr);
        return *error;
    }

  private:
    std::variant<T, Error> storage_;
};

template <> class [[nodiscard]] Result<void> final {
  public:
    constexpr Result() noexcept = default;
    constexpr Result(Error error) noexcept : error_(error), has_value_(false) {}

    [[nodiscard]] constexpr bool has_value() const noexcept { return has_value_; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return has_value_; }
    [[nodiscard]] constexpr Error error() const noexcept {
        assert(!has_value_);
        return error_;
    }

  private:
    Error error_{};
    bool has_value_{true};
};

static_assert(std::is_trivially_copyable_v<Result<void>>);

} // namespace vramz
