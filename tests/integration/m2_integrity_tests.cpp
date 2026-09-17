#include "../test_support.hpp"

#include "vramz/crc32c.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

using namespace vramz;

namespace {

enum class DataPattern : std::uint32_t {
    zeros,
    ones,
    alternating,
    repeated_long,
    ramp,
    sparse,
    structured_u32,
    structured_fp32,
    embedded_zeros,
    random,
    count
};

[[nodiscard]] RuntimeConfig data_config(ByteSize chunk_size = ByteSize{64U}) noexcept {
    RuntimeConfig config{};
    constexpr std::uint64_t limit = 64ULL * 1024ULL * 1024ULL;
    config.budgets = MemoryBudgets{TierBudget{ByteSize{limit}, ByteSize{48ULL * 1024ULL * 1024ULL},
                                              ByteSize{16ULL * 1024ULL * 1024ULL}},
                                   TierBudget{ByteSize{limit}, ByteSize{48ULL * 1024ULL * 1024ULL},
                                              ByteSize{16ULL * 1024ULL * 1024ULL}}};
    config.preferred_chunk_size = chunk_size;
    config.async_errors.max_retained_errors = 16U;
    return config;
}

[[nodiscard]] std::vector<std::byte> make_data(std::size_t size, DataPattern pattern) {
    std::vector<std::byte> result(size);
    std::uint32_t state = 0xC001D00DU ^ static_cast<std::uint32_t>(pattern);
    for (std::size_t index = 0U; index < size; ++index) {
        switch (pattern) {
        case DataPattern::zeros:
            result[index] = std::byte{};
            break;
        case DataPattern::ones:
            result[index] = std::byte{0xFFU};
            break;
        case DataPattern::alternating:
            result[index] = (index & 1U) == 0U ? std::byte{0xAAU} : std::byte{0x55U};
            break;
        case DataPattern::repeated_long:
            result[index] = static_cast<std::byte>((index % 64U) * 3U);
            break;
        case DataPattern::ramp:
            result[index] = static_cast<std::byte>(index & 0xFFU);
            break;
        case DataPattern::sparse:
            result[index] = (index % 97U) == 0U ? std::byte{0x7FU} : std::byte{};
            break;
        case DataPattern::structured_u32: {
            const auto word = static_cast<std::uint32_t>((index / 4U) % 17U);
            const auto shift = static_cast<std::uint32_t>((index % 4U) * 8U);
            result[index] = static_cast<std::byte>((word >> shift) & 0xFFU);
            break;
        }
        case DataPattern::structured_fp32: {
            constexpr std::array<std::uint32_t, 6U> fp32_bits{
                0x00000000U, 0x3F800000U, 0xBF800000U, 0x3F000000U, 0x40000000U, 0x40490FDBU};
            const auto bits = fp32_bits[(index / 4U) % fp32_bits.size()];
            const auto shift = static_cast<std::uint32_t>((index % 4U) * 8U);
            result[index] = static_cast<std::byte>((bits >> shift) & 0xFFU);
            break;
        }
        case DataPattern::embedded_zeros:
            result[index] =
                (index % 5U) == 0U ? std::byte{} : static_cast<std::byte>((index * 29U) & 0xFFU);
            break;
        case DataPattern::random:
            state = state * 1664525U + 1013904223U;
            result[index] = static_cast<std::byte>((state >> 24U) & 0xFFU);
            break;
        case DataPattern::count:
            result[index] = std::byte{};
            break;
        }
    }
    return result;
}

[[nodiscard]] bool write_buffer(Buffer& buffer, std::span<const std::byte> bytes) {
    auto lease = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{bytes.size()}},
                                AcquireOptions{AccessMode::read_write});
    if (!lease) {
        return false;
    }
    const auto written = testing::write_bytes(lease.value(), ByteOffset{}, bytes);
    const auto closed = lease.value().close();
    return written.has_value() && closed.has_value();
}

[[nodiscard]] bool read_buffer(Buffer& buffer, std::span<std::byte> output) {
    auto lease = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{output.size()}});
    if (!lease) {
        return false;
    }
    const auto read = testing::read_bytes(lease.value(), ByteOffset{}, output);
    const auto closed = lease.value().close();
    return read.has_value() && closed.has_value();
}

[[nodiscard]] bool clean(const RuntimeStats& stats) noexcept {
    const auto empty = [](const TierUsage& usage) noexcept {
        return usage.committed == ByteSize{} && usage.reserved == ByteSize{} &&
               usage.staging == ByteSize{} && usage.workspace == ByteSize{} &&
               usage.cleanup_debt == ByteSize{};
    };
    return empty(stats.gpu) && empty(stats.host);
}

class AccountingObserver final : public TransactionObserver {
  public:
    explicit AccountingObserver(const Runtime& runtime) noexcept : runtime_(runtime) {}

    void on_phase(TransactionPhase phase, const ChunkSnapshot&) noexcept override {
        const auto index = static_cast<std::size_t>(phase);
        if (index < samples_.size()) {
            samples_[index] = runtime_.stats();
            seen_[index] = true;
        }
        conserved_ = conserved_ && testing::accounting_conserved(runtime_, PhysicalTier::gpu) &&
                     testing::accounting_conserved(runtime_, PhysicalTier::host);
    }

    [[nodiscard]] bool conserved() const noexcept { return conserved_; }
    [[nodiscard]] bool saw(TransactionPhase phase) const noexcept {
        return seen_[static_cast<std::size_t>(phase)];
    }
    [[nodiscard]] const RuntimeStats& at(TransactionPhase phase) const noexcept {
        return samples_[static_cast<std::size_t>(phase)];
    }

  private:
    static constexpr std::size_t phase_count =
        static_cast<std::size_t>(TransactionPhase::poisoned) + 1U;
    const Runtime& runtime_;
    std::array<RuntimeStats, phase_count> samples_{};
    std::array<bool, phase_count> seen_{};
    bool conserved_{true};
};

} // namespace

int main() {
    test::Runner runner;

    runner.begin("CRC32C Castagnoli known vector and edge cases");
    constexpr std::array<std::byte, 9U> check{std::byte{'1'}, std::byte{'2'}, std::byte{'3'},
                                              std::byte{'4'}, std::byte{'5'}, std::byte{'6'},
                                              std::byte{'7'}, std::byte{'8'}, std::byte{'9'}};
    const std::array<std::byte, 1U> one{std::byte{1U}};
    const std::array<std::byte, 4U> repeated{std::byte{0xA5U}, std::byte{0xA5U}, std::byte{0xA5U},
                                             std::byte{0xA5U}};
    const std::array<std::byte, 4U> binary{std::byte{0U}, std::byte{0xFFU}, std::byte{0x80U},
                                           std::byte{0x7FU}};
    VRAMZ_CHECK(runner, crc32c(check) == 0xE3069283U);
    VRAMZ_CHECK(runner, crc32c({}) == 0U);
    VRAMZ_CHECK(runner, crc32c(one) != 0U);
    VRAMZ_CHECK(runner, crc32c(repeated) != crc32c(binary));

    runner.begin("LZ4 Block codec handles zero and checked maximum input");
    const std::span<const std::byte> empty{};
    const std::span<std::byte> empty_output{};
    VRAMZ_CHECK(runner, testing::codec_round_trip(empty, empty_output));
    const auto maximum_input = testing::codec_maximum_input_size();
    VRAMZ_CHECK(runner, maximum_input.value() > 0U);
    VRAMZ_CHECK(runner, testing::codec_maximum_compressed_size(maximum_input));
    const auto oversized =
        testing::codec_maximum_compressed_size(ByteSize{maximum_input.value() + 1U});
    VRAMZ_CHECK(runner, !oversized && oversized.error().code == ErrorCode::invalid_argument);

    runner.begin("new reference allocations are zero filled");
    auto zero_runtime_result = Runtime::create(data_config());
    auto zero_runtime = std::move(zero_runtime_result).value();
    auto zero_buffer_result = zero_runtime.allocate(ByteSize{127U});
    auto zero_buffer = std::move(zero_buffer_result).value();
    std::vector<std::byte> zero_output(127U, std::byte{0xFFU});
    VRAMZ_CHECK(runner, read_buffer(zero_buffer, zero_output));
    VRAMZ_CHECK(runner, std::all_of(zero_output.begin(), zero_output.end(),
                                    [](std::byte value) noexcept { return value == std::byte{}; }));
    VRAMZ_CHECK(runner, zero_buffer.close());
    VRAMZ_CHECK(runner, zero_runtime.shutdown());
    VRAMZ_CHECK(runner, clean(zero_runtime.stats()));

    runner.begin("deterministic corpus survives raw compressed host round trips");
    constexpr std::array<std::size_t, 20U> sizes{1U,
                                                 2U,
                                                 3U,
                                                 4U,
                                                 15U,
                                                 16U,
                                                 31U,
                                                 32U,
                                                 63U,
                                                 64U,
                                                 65U,
                                                 255U,
                                                 256U,
                                                 257U,
                                                 4095U,
                                                 4096U,
                                                 4097U,
                                                 static_cast<std::size_t>(64U) * 1024U,
                                                 static_cast<std::size_t>(256U) * 1024U,
                                                 static_cast<std::size_t>(1024U) * 1024U};
    for (const auto size : sizes) {
        const auto chunk_value = ((static_cast<std::uint64_t>(size) + 63U) / 64U) * 64U;
        for (std::uint32_t pattern = 0U; pattern < static_cast<std::uint32_t>(DataPattern::count);
             ++pattern) {
            const auto input = make_data(size, static_cast<DataPattern>(pattern));
            std::vector<std::byte> codec_output(size);
            const auto codec_result = testing::codec_round_trip(input, codec_output);
            VRAMZ_CHECK(runner, codec_result && codec_output == input);
            auto runtime_result = Runtime::create(data_config(ByteSize{chunk_value}));
            VRAMZ_CHECK(runner, runtime_result);
            auto runtime = std::move(runtime_result).value();
            VRAMZ_CHECK(runner, testing::cpu_lz4_available(runtime));
            auto buffer_result = runtime.allocate(ByteSize{size});
            VRAMZ_CHECK(runner, buffer_result);
            auto buffer = std::move(buffer_result).value();
            VRAMZ_CHECK(runner, write_buffer(buffer, input));
            const auto written = testing::chunk_snapshot(buffer, 0U);
            VRAMZ_CHECK(runner,
                        written && written.value().authoritative.metadata.crc32c == crc32c(input));
            constexpr std::array<RepresentationState, 6U> path{
                RepresentationState::gpu_compressed, RepresentationState::host_compressed,
                RepresentationState::host_raw,       RepresentationState::gpu_raw,
                RepresentationState::host_raw,       RepresentationState::gpu_raw};
            for (const auto destination : path) {
                const auto migrated = testing::migrate(buffer, 0U, destination);
                if (!migrated) {
                    std::cerr << "size=" << size << " pattern=" << pattern
                              << " destination=" << static_cast<unsigned>(destination)
                              << " error=" << static_cast<unsigned>(migrated.error().code)
                              << " operation=" << static_cast<unsigned>(migrated.error().operation)
                              << '\n';
                }
                VRAMZ_CHECK(runner, migrated);
                VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
                VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
            }
            std::vector<std::byte> output(size);
            VRAMZ_CHECK(runner, read_buffer(buffer, output));
            VRAMZ_CHECK(runner, output == input);
            VRAMZ_CHECK(runner, buffer.close());
            VRAMZ_CHECK(runner, runtime.shutdown());
            VRAMZ_CHECK(runner, clean(runtime.stats()));
            VRAMZ_CHECK(runner, testing::owned_resource_count(runtime) == 0U);
        }
    }

    runner.begin("all twelve semantic transitions preserve actual bytes");
    const auto transition_data = make_data(4096U, DataPattern::random);
    for (const auto source : test::all_states) {
        for (const auto destination : test::all_states) {
            if (source == destination) {
                continue;
            }
            auto runtime_result = Runtime::create(data_config(ByteSize{4096U}));
            auto runtime = std::move(runtime_result).value();
            auto buffer_result = runtime.allocate(ByteSize{transition_data.size()});
            auto buffer = std::move(buffer_result).value();
            VRAMZ_CHECK(runner, write_buffer(buffer, transition_data));
            const auto expected_content =
                testing::chunk_snapshot(buffer, 0U).value().authoritative.content;
            if (source != RepresentationState::gpu_raw) {
                VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, source));
            }
            const auto source_snapshot = testing::chunk_snapshot(buffer, 0U).value();
            VRAMZ_CHECK(runner, testing::migrate(buffer, 0U, destination));
            const auto destination_snapshot = testing::chunk_snapshot(buffer, 0U).value();
            VRAMZ_CHECK(runner, destination_snapshot.authoritative.state == destination);
            VRAMZ_CHECK(runner, destination_snapshot.authoritative.resource !=
                                    source_snapshot.authoritative.resource);
            VRAMZ_CHECK(runner, destination_snapshot.authoritative.content == expected_content);
            VRAMZ_CHECK(runner, destination_snapshot.authoritative.metadata.logical_size ==
                                    ByteSize{transition_data.size()});
            VRAMZ_CHECK(runner, destination_snapshot.authoritative.metadata.crc32c ==
                                    crc32c(transition_data));
            VRAMZ_CHECK(runner,
                        is_raw(destination)
                            ? destination_snapshot.authoritative.metadata.stored_size ==
                                  ByteSize{transition_data.size()}
                            : destination_snapshot.authoritative.metadata.stored_size.value() > 0U);
            VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::gpu));
            VRAMZ_CHECK(runner, testing::accounting_conserved(runtime, PhysicalTier::host));
            std::vector<std::byte> output(transition_data.size());
            VRAMZ_CHECK(runner, read_buffer(buffer, output));
            VRAMZ_CHECK(runner, output == transition_data);
            VRAMZ_CHECK(runner, buffer.close());
            VRAMZ_CHECK(runner, runtime.shutdown());
            VRAMZ_CHECK(runner, clean(runtime.stats()));
        }
    }

    runner.begin("partial writes across chunk boundaries update generations and CRC");
    auto partial_runtime_result = Runtime::create(data_config(ByteSize{64U}));
    auto partial_runtime = std::move(partial_runtime_result).value();
    auto partial_buffer_result = partial_runtime.allocate(ByteSize{160U});
    auto partial_buffer = std::move(partial_buffer_result).value();
    auto partial_lease = partial_buffer.acquire(MemoryRange{ByteOffset{48U}, ByteSize{80U}},
                                                AcquireOptions{AccessMode::read_write});
    const auto patch_bytes = make_data(80U, DataPattern::ramp);
    VRAMZ_CHECK(runner, partial_lease);
    VRAMZ_CHECK(runner, testing::write_bytes(partial_lease.value(), ByteOffset{}, patch_bytes));
    VRAMZ_CHECK(runner, partial_lease.value().close());
    for (std::uint64_t index = 0U; index < 2U; ++index) {
        const auto snapshot = testing::chunk_snapshot(partial_buffer, index);
        VRAMZ_CHECK(runner, snapshot && snapshot.value().authoritative.content.generation == 1U);
    }
    auto second_lease = partial_buffer.acquire(MemoryRange{ByteOffset{64U}, ByteSize{16U}},
                                               AcquireOptions{AccessMode::read_write});
    const std::array<std::byte, 4U> second_patch{std::byte{0xDEU}, std::byte{0xADU},
                                                 std::byte{0xBEU}, std::byte{0xEFU}};
    VRAMZ_CHECK(runner, second_lease);
    VRAMZ_CHECK(runner, testing::write_bytes(second_lease.value(), ByteOffset{4U}, second_patch));
    VRAMZ_CHECK(runner, second_lease.value().close());
    VRAMZ_CHECK(
        runner,
        testing::chunk_snapshot(partial_buffer, 1U).value().authoritative.content.generation == 2U);
    std::vector<std::byte> partial_output(160U);
    std::vector<std::byte> partial_expected(160U);
    std::copy(patch_bytes.begin(), patch_bytes.end(), partial_expected.begin() + 48);
    std::copy(second_patch.begin(), second_patch.end(), partial_expected.begin() + 68);
    VRAMZ_CHECK(runner, read_buffer(partial_buffer, partial_output));
    VRAMZ_CHECK(runner, partial_output == partial_expected);
    for (std::uint64_t index = 0U; index < 3U; ++index) {
        const auto snapshot = testing::chunk_snapshot(partial_buffer, index).value();
        const auto offset = static_cast<std::size_t>(index * 64U);
        const auto length = std::min<std::size_t>(64U, partial_expected.size() - offset);
        VRAMZ_CHECK(runner, snapshot.authoritative.metadata.crc32c ==
                                crc32c(std::span<const std::byte>{partial_expected}.subspan(
                                    offset, length)));
    }
    VRAMZ_CHECK(runner, partial_buffer.close());
    VRAMZ_CHECK(runner, partial_runtime.shutdown());

    runner.begin("read-only and stale leases cannot mutate storage");
    auto lease_runtime_result = Runtime::create(data_config());
    auto lease_runtime = std::move(lease_runtime_result).value();
    auto lease_buffer_result = lease_runtime.allocate(ByteSize{64U});
    auto lease_buffer = std::move(lease_buffer_result).value();
    auto read_lease = lease_buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{64U}});
    const std::array<std::byte, 1U> mutation{std::byte{1U}};
    VRAMZ_CHECK(runner, read_lease);
    const auto denied = testing::write_bytes(read_lease.value(), ByteOffset{}, mutation);
    VRAMZ_CHECK(runner, !denied && denied.error().code == ErrorCode::conflict);
    VRAMZ_CHECK(runner, read_lease.value().close());
    const auto stale_read =
        testing::read_bytes(read_lease.value(), ByteOffset{}, std::span<std::byte>{});
    VRAMZ_CHECK(runner, !stale_read && stale_read.error().code == ErrorCode::stale_handle);
    VRAMZ_CHECK(runner, lease_buffer.close());
    VRAMZ_CHECK(runner, lease_runtime.shutdown());

    runner.begin("incompressible payload and compression statistics are reported honestly");
    const auto random_data = make_data(65536U, DataPattern::random);
    auto stats_runtime_result = Runtime::create(data_config(ByteSize{65536U}));
    auto stats_runtime = std::move(stats_runtime_result).value();
    auto stats_buffer_result = stats_runtime.allocate(ByteSize{random_data.size()});
    auto stats_buffer = std::move(stats_buffer_result).value();
    VRAMZ_CHECK(runner, write_buffer(stats_buffer, random_data));
    VRAMZ_CHECK(runner, testing::migrate(stats_buffer, 0U, RepresentationState::gpu_compressed));
    const auto compressed_metadata = testing::representation_metadata(stats_buffer, 0U);
    const auto compression_stats = stats_runtime.stats().compression;
    VRAMZ_CHECK(runner,
                compressed_metadata && compressed_metadata.value().encoding == Encoding::lz4_block);
    VRAMZ_CHECK(runner,
                compression_stats.logical_bytes_represented == ByteSize{random_data.size()});
    VRAMZ_CHECK(runner, compression_stats.stored_payload_bytes.value() >= random_data.size());
    VRAMZ_CHECK(runner, compression_stats.physical_compression_ratio() <= 1.0);
    VRAMZ_CHECK(runner, compression_stats.successful_compressions >= 1U);
    VRAMZ_CHECK(runner, stats_buffer.close());
    VRAMZ_CHECK(runner, stats_runtime.shutdown());

    runner.begin("real compression conserves every ledger phase for compressible and random data");
    constexpr std::array<DataPattern, 2U> accounting_patterns{DataPattern::zeros,
                                                              DataPattern::random};
    for (const auto pattern : accounting_patterns) {
        const auto bytes = make_data(65536U, pattern);
        auto accounting_runtime_result = Runtime::create(data_config(ByteSize{65536U}));
        auto accounting_runtime = std::move(accounting_runtime_result).value();
        auto accounting_buffer_result = accounting_runtime.allocate(ByteSize{bytes.size()});
        auto accounting_buffer = std::move(accounting_buffer_result).value();
        VRAMZ_CHECK(runner, write_buffer(accounting_buffer, bytes));
        AccountingObserver observer{accounting_runtime};
        VRAMZ_CHECK(runner, testing::migrate(accounting_buffer, 0U,
                                             RepresentationState::gpu_compressed, &observer));
        VRAMZ_CHECK(runner, observer.conserved());
        VRAMZ_CHECK(runner, observer.saw(TransactionPhase::reserved));
        VRAMZ_CHECK(runner, observer.saw(TransactionPhase::destination_materialized));
        VRAMZ_CHECK(runner, observer.saw(TransactionPhase::charge_reconciled));
        VRAMZ_CHECK(runner, observer.at(TransactionPhase::charge_reconciled).gpu.staging ==
                                observer.at(TransactionPhase::transferred).gpu.staging);
        VRAMZ_CHECK(runner, observer.saw(TransactionPhase::workspace_materialized));
        VRAMZ_CHECK(runner, observer.saw(TransactionPhase::commit_ready));
        VRAMZ_CHECK(runner, observer.saw(TransactionPhase::committed));
        VRAMZ_CHECK(runner, observer.saw(TransactionPhase::cleanup_complete));
        VRAMZ_CHECK(runner, observer.at(TransactionPhase::reserved).gpu.reserved.value() > 0U);
        VRAMZ_CHECK(runner,
                    observer.at(TransactionPhase::destination_materialized).gpu.staging.value() >
                        0U);
        VRAMZ_CHECK(runner,
                    observer.at(TransactionPhase::workspace_materialized).gpu.workspace.value() >
                        0U);
        if (pattern == DataPattern::zeros) {
            VRAMZ_CHECK(runner,
                        observer.at(TransactionPhase::transferred).gpu.staging <
                            observer.at(TransactionPhase::destination_materialized).gpu.staging);
        }
        const auto commit_ready = observer.at(TransactionPhase::commit_ready).gpu;
        VRAMZ_CHECK(runner, commit_ready.staging.value() > 0U &&
                                commit_ready.workspace == ByteSize{} &&
                                commit_ready.reserved == ByteSize{});
        VRAMZ_CHECK(runner, observer.at(TransactionPhase::committed).gpu.cleanup_debt.value() > 0U);
        const auto cleanup_complete = observer.at(TransactionPhase::cleanup_complete).gpu;
        VRAMZ_CHECK(runner, cleanup_complete.staging == ByteSize{} &&
                                cleanup_complete.workspace == ByteSize{} &&
                                cleanup_complete.cleanup_debt == ByteSize{});
        VRAMZ_CHECK(runner, accounting_buffer.close());
        VRAMZ_CHECK(runner, accounting_runtime.shutdown());
    }

    runner.begin("multiple writes replace generation checksum and bytes without stale metadata");
    const auto generation_a = make_data(4096U, DataPattern::structured_fp32);
    const auto generation_b = make_data(4096U, DataPattern::embedded_zeros);
    auto generations_runtime_result = Runtime::create(data_config(ByteSize{4096U}));
    auto generations_runtime = std::move(generations_runtime_result).value();
    auto generations_buffer_result = generations_runtime.allocate(ByteSize{generation_a.size()});
    auto generations_buffer = std::move(generations_buffer_result).value();
    VRAMZ_CHECK(runner, write_buffer(generations_buffer, generation_a));
    const auto tag_a =
        testing::chunk_snapshot(generations_buffer, 0U).value().authoritative.content;
    VRAMZ_CHECK(runner,
                testing::migrate(generations_buffer, 0U, RepresentationState::gpu_compressed));
    VRAMZ_CHECK(runner, testing::migrate(generations_buffer, 0U, RepresentationState::gpu_raw));
    std::vector<std::byte> restored_a(generation_a.size());
    VRAMZ_CHECK(runner, read_buffer(generations_buffer, restored_a));
    VRAMZ_CHECK(runner, restored_a == generation_a);
    VRAMZ_CHECK(runner, write_buffer(generations_buffer, generation_b));
    const auto metadata_b = testing::chunk_snapshot(generations_buffer, 0U).value().authoritative;
    VRAMZ_CHECK(runner, metadata_b.content.generation == tag_a.generation + 1U);
    VRAMZ_CHECK(runner, metadata_b.content != tag_a);
    VRAMZ_CHECK(runner, metadata_b.metadata.crc32c == crc32c(generation_b));
    VRAMZ_CHECK(runner, metadata_b.metadata.crc32c != crc32c(generation_a));
    VRAMZ_CHECK(runner,
                testing::migrate(generations_buffer, 0U, RepresentationState::host_compressed));
    VRAMZ_CHECK(runner, testing::migrate(generations_buffer, 0U, RepresentationState::host_raw));
    VRAMZ_CHECK(runner, testing::migrate(generations_buffer, 0U, RepresentationState::gpu_raw));
    std::vector<std::byte> restored_b(generation_b.size());
    VRAMZ_CHECK(runner, read_buffer(generations_buffer, restored_b));
    VRAMZ_CHECK(runner, restored_b == generation_b);
    VRAMZ_CHECK(runner, generations_buffer.close());
    VRAMZ_CHECK(runner, generations_runtime.shutdown());

    runner.begin("one thousand migrations preserve bytes generations and accounting");
    const auto chain_data = make_data(1024U, DataPattern::structured_u32);
    auto chain_runtime_result = Runtime::create(data_config(ByteSize{1024U}));
    auto chain_runtime = std::move(chain_runtime_result).value();
    auto chain_buffer_result = chain_runtime.allocate(ByteSize{chain_data.size()});
    auto chain_buffer = std::move(chain_buffer_result).value();
    VRAMZ_CHECK(runner, write_buffer(chain_buffer, chain_data));
    const auto generation_before =
        testing::chunk_snapshot(chain_buffer, 0U).value().authoritative.content.generation;
    for (std::uint64_t index = 0U; index < 1000U; ++index) {
        const auto current = testing::chunk_snapshot(chain_buffer, 0U).value().authoritative.state;
        auto destination = test::all_states[static_cast<std::size_t>((index + 1U) % 4U)];
        if (destination == current) {
            destination = test::all_states[static_cast<std::size_t>((index + 2U) % 4U)];
        }
        VRAMZ_CHECK(runner, testing::migrate(chain_buffer, 0U, destination));
        VRAMZ_CHECK(runner, testing::accounting_conserved(chain_runtime, PhysicalTier::gpu));
        VRAMZ_CHECK(runner, testing::accounting_conserved(chain_runtime, PhysicalTier::host));
        if (destination == RepresentationState::gpu_raw) {
            std::vector<std::byte> intermediate(chain_data.size());
            VRAMZ_CHECK(runner, read_buffer(chain_buffer, intermediate));
            VRAMZ_CHECK(runner, intermediate == chain_data);
        }
    }
    std::vector<std::byte> chain_output(chain_data.size());
    VRAMZ_CHECK(runner, read_buffer(chain_buffer, chain_output));
    VRAMZ_CHECK(runner, chain_output == chain_data);
    VRAMZ_CHECK(
        runner,
        testing::chunk_snapshot(chain_buffer, 0U).value().authoritative.content.generation ==
            generation_before);
    VRAMZ_CHECK(runner, chain_buffer.close());
    VRAMZ_CHECK(runner, chain_runtime.shutdown());
    VRAMZ_CHECK(runner, clean(chain_runtime.stats()));

    return runner.finish();
}
