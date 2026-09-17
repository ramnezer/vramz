#include "../test_support.hpp"
#include "vramz/detail/controlled_capacity_smoke.hpp"
#include <array>
#include <limits>
int main() {
    using namespace vramz;
    test::Runner runner;
    runner.begin("M12 fixed argument gates reject device/count/size/cap/mode overrides");
    const std::array<std::string_view, 3U> args{"--run-controlled-capacity-smoke", "--device", "0"};
    VRAMZ_CHECK(runner, detail::parse_controlled_capacity_smoke_arguments(args));
    for (const auto bad : {std::array<std::string_view, 2U>{"--device", "1"},
                           {"--chunk-count", "5"},
                           {"--logical-bytes", "16"},
                           {"--max-physical-bytes", "1"},
                           {"--host-fallback", "on"},
                           {"--policy", "off"}}) {
        VRAMZ_CHECK(runner, !detail::parse_controlled_capacity_smoke_arguments(bad));
    }
    runner.begin("M12 budget configuration uses the unchanged production pressure formula");
    const auto c = detail::controlled_capacity_configuration();
    VRAMZ_CHECK(runner, c.policy.mode == PolicyMode::gpu_resident);
    VRAMZ_CHECK(runner, c.budgets.gpu.hard_limit == ByteSize{134217728U});
    VRAMZ_CHECK(runner, policy::pressure(ByteSize{50331648U}, c.budgets.gpu.soft_target,
                                         c.budgets.gpu.hard_limit,
                                         c.budgets.gpu.migration_reserve) == Pressure::hard);
    VRAMZ_CHECK(runner, policy::pressure(ByteSize{37748736U}, c.budgets.gpu.soft_target,
                                         c.budgets.gpu.hard_limit,
                                         c.budgets.gpu.migration_reserve) == Pressure::soft);
    VRAMZ_CHECK(runner, policy::pressure(ByteSize{31457280U}, c.budgets.gpu.soft_target,
                                         c.budgets.gpu.hard_limit,
                                         c.budgets.gpu.migration_reserve) == Pressure::normal);
    VRAMZ_CHECK(runner, c.budgets.gpu.migration_reserve == ByteSize{83886080U});
    VRAMZ_CHECK(runner, c.budgets.gpu.soft_target == ByteSize{33554432U});
    VRAMZ_CHECK(runner, c.policy.tuning.maximum_transitions == 1U);
    VRAMZ_CHECK(runner, c.budgets.host.hard_limit == ByteSize{});
    for (const auto value : {46137343U, 46137344U, 46137345U}) {
        const auto observed =
            policy::pressure(ByteSize{value}, c.budgets.gpu.soft_target, c.budgets.gpu.hard_limit,
                             c.budgets.gpu.migration_reserve);
        VRAMZ_CHECK(runner, observed == (value < 46137344U ? Pressure::soft : Pressure::hard));
    }
    runner.begin("M12 exact integer capacity boundary, lower charge, above target and overflow");
    for (const auto value : {33554432U, 31457280U}) {
        const auto accepted =
            detail::controlled_capacity_1_5x(detail::m12_aggregate_logical, ByteSize{value});
        VRAMZ_CHECK(runner, accepted && accepted.value());
    }
    for (const auto value : {0U, 33554433U, 35651584U}) {
        const auto rejected =
            detail::controlled_capacity_1_5x(detail::m12_aggregate_logical, ByteSize{value});
        VRAMZ_CHECK(runner, rejected && !rejected.value());
    }
    const auto wrong_logical =
        detail::controlled_capacity_1_5x(ByteSize{50331647U}, ByteSize{33554432U});
    VRAMZ_CHECK(runner, wrong_logical && !wrong_logical.value());
    VRAMZ_CHECK(runner, !detail::controlled_capacity_1_5x(
                            ByteSize{std::numeric_limits<std::uint64_t>::max()}, ByteSize{1U}));
    VRAMZ_CHECK(runner, !detail::controlled_capacity_1_5x(
                            detail::m12_aggregate_logical,
                            ByteSize{std::numeric_limits<std::uint64_t>::max()}));
    runner.begin("M12 six distinct M10-family variants preserve the first four definitions");
    std::array<std::array<std::byte, 8192U>, 6U> patterns{};
    for (std::size_t chunk = 0U; chunk < patterns.size(); ++chunk) {
        bool zero{}, nonzero{}, variation{};
        for (std::size_t index = 0U; index < patterns[chunk].size(); ++index) {
            auto& value = patterns[chunk][index];
            value = detail::controlled_capacity_payload_byte(chunk, index);
            zero = zero || value == std::byte{};
            nonzero = nonzero || value != std::byte{};
            variation = variation || value != patterns[chunk][0];
            if (chunk < 4U) {
                VRAMZ_CHECK(runner,
                            value == detail::multi_chunk_residency_payload_byte(chunk, index));
            }
        }
        VRAMZ_CHECK(runner, zero && nonzero && variation);
        for (std::size_t prior = 0U; prior < chunk; ++prior) {
            VRAMZ_CHECK(runner, patterns[prior] != patterns[chunk]);
        }
    }
    runner.begin("M12 checked full stage admission accepts exact cap and rejects overflow");
    VRAMZ_CHECK(runner, detail::admit_controlled_capacity_stage(ByteSize{134217727U}, ByteSize{1U},
                                                                {}, {}));
    VRAMZ_CHECK(runner, !detail::admit_controlled_capacity_stage(ByteSize{134217728U}, ByteSize{1U},
                                                                 {}, {}));
    VRAMZ_CHECK(runner,
                !detail::admit_controlled_capacity_stage(
                    ByteSize{std::numeric_limits<std::uint64_t>::max()}, ByteSize{1U}, {}, {}));
    runner.begin("M12 bounded JSON keeps Fake hardware and Gate G unvalidated");
    detail::ControlledCapacitySmokeReport report{};
    std::array<char, 65536U> output{};
    const auto formatted =
        detail::format_controlled_capacity_smoke_report(report, "", false, output);
    VRAMZ_CHECK(runner, !formatted.truncated);
    const std::string_view json{output.data(), formatted.written};
    for (const auto field :
         {"\"m12_version\":1", "\"manual_victim_selection\":false",
          "\"manual_compression_transition\":false", "\"gate_g\":\"NOT_RUN\"",
          "\"hardware_validation\":\"NOT_TESTED\"", "\"final_budget_charge\":null"}) {
        VRAMZ_CHECK(runner, json.find(field) != std::string_view::npos);
    }
    runner.begin("M12 full six-cycle report remains bounded even with maximal counters");
    report.cycle_count = 6U;
    for (auto& cycle : report.cycles) {
        cycle.selected = true;
        cycle.stage_count = 16U;
        cycle.cycle_id = std::numeric_limits<std::uint64_t>::max();
        for (auto& stage : cycle.stages) {
            stage.usage.committed = ByteSize{std::numeric_limits<std::uint64_t>::max()};
            stage.owned_physical_charge = stage.usage.committed;
        }
    }
    VRAMZ_CHECK(
        runner,
        !detail::format_controlled_capacity_smoke_report(report, "", false, output).truncated);
    std::array<char, 32U> tiny{};
    VRAMZ_CHECK(runner,
                detail::format_controlled_capacity_smoke_report(report, "", false, tiny).truncated);
    return runner.finish();
}
