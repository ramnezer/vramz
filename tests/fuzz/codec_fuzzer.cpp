#include "vramz/testing.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <span>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    constexpr std::size_t maximum_input = static_cast<std::size_t>(1024U) * 1024U;
    if (size < 3U || size > maximum_input) {
        return 0;
    }
    const auto expected_size =
        ((static_cast<std::size_t>(data[0]) << 16U) | (static_cast<std::size_t>(data[1]) << 8U) |
         static_cast<std::size_t>(data[2])) %
        (maximum_input + 1U);
    try {
        std::vector<std::byte> output(expected_size);
        const auto compressed =
            std::span<const std::byte>{reinterpret_cast<const std::byte*>(data + 3U), size - 3U};
        static_cast<void>(vramz::testing::codec_decompress(compressed, output));
    } catch (const std::bad_alloc&) {
        return 0;
    } catch (...) {
        std::abort();
    }
    return 0;
}
