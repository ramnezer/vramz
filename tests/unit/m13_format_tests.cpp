#include "../test_support.hpp"
#include "vramz/detail/controlled_capacity_2x_smoke.hpp"
#include "vramz/detail/controlled_capacity_smoke.hpp"
#include <array>
#include <limits>
int main() {
    using namespace vramz;
    test::Runner runner;
    runner.begin("M13 fixed argument gates reject device/count/size/cap/mode overrides");
    const std::array<std::string_view, 3U> args{"--run-controlled-capacity-2x-smoke", "--device",
                                                "0"};
    VRAMZ_CHECK(runner, detail::parse_controlled_capacity_2x_smoke_arguments(args));
    for (const auto bad : {std::array<std::string_view, 2U>{"--device", "1"},
                           {"--chunk-count", "5"},
                           {"--logical-bytes", "16"},
                           {"--max-physical-bytes", "1"},
                           {"--host-fallback", "on"},
                           {"--policy", "off"}}) {
        VRAMZ_CHECK(runner, !detail::parse_controlled_capacity_2x_smoke_arguments(bad));
    }
    runner.begin("M13 budget configuration uses the unchanged production pressure formula");
    const auto c = detail::controlled_capacity_2x_configuration();
    VRAMZ_CHECK(runner, c.policy.mode == PolicyMode::gpu_resident);
    VRAMZ_CHECK(runner, c.budgets.gpu.hard_limit == ByteSize{134217728U});
    VRAMZ_CHECK(runner, policy::pressure(ByteSize{67108864U}, c.budgets.gpu.soft_target,
                                         c.budgets.gpu.hard_limit,
                                         c.budgets.gpu.migration_reserve) == Pressure::hard);
    VRAMZ_CHECK(runner, policy::pressure(ByteSize{37748736U}, c.budgets.gpu.soft_target,
                                         c.budgets.gpu.hard_limit,
                                         c.budgets.gpu.migration_reserve) == Pressure::soft);
    VRAMZ_CHECK(runner, policy::pressure(ByteSize{31457280U}, c.budgets.gpu.soft_target,
                                         c.budgets.gpu.hard_limit,
                                         c.budgets.gpu.migration_reserve) == Pressure::normal);
    VRAMZ_CHECK(runner, c.budgets.gpu.migration_reserve == ByteSize{67108864U});
    VRAMZ_CHECK(runner, c.budgets.gpu.soft_target == ByteSize{33554432U});
    VRAMZ_CHECK(runner, c.policy.tuning.maximum_transitions == 1U);
    VRAMZ_CHECK(runner, c.budgets.host.hard_limit == ByteSize{});
    VRAMZ_CHECK(runner,
                c.budgets.gpu.hard_limit.value() - c.budgets.gpu.migration_reserve.value() ==
                    detail::m13_normal_limit.value());
    for (const auto value : {58720255U, 58720256U, 58720257U}) {
        const auto observed =
            policy::pressure(ByteSize{value}, c.budgets.gpu.soft_target, c.budgets.gpu.hard_limit,
                             c.budgets.gpu.migration_reserve);
        VRAMZ_CHECK(runner, observed == (value < 58720256U ? Pressure::soft : Pressure::hard));
    }
    runner.begin("M13 exact integer capacity boundary, lower charge, above target and overflow");
    for (const auto value : {33554432U, 31457280U}) {
        const auto accepted =
            detail::controlled_capacity_2_0x(detail::m13_aggregate_logical, ByteSize{value});
        VRAMZ_CHECK(runner, accepted && accepted.value());
    }
    for (const auto value : {0U, 33554433U, 35651584U}) {
        const auto rejected =
            detail::controlled_capacity_2_0x(detail::m13_aggregate_logical, ByteSize{value});
        VRAMZ_CHECK(runner, rejected && !rejected.value());
    }
    const auto wrong_logical =
        detail::controlled_capacity_2_0x(ByteSize{67108863U}, ByteSize{33554432U});
    VRAMZ_CHECK(runner, wrong_logical && !wrong_logical.value());
    const auto excessive_logical = detail::controlled_capacity_2_0x(
        ByteSize{std::numeric_limits<std::uint64_t>::max()}, ByteSize{1U});
    VRAMZ_CHECK(runner, excessive_logical && !excessive_logical.value());
    VRAMZ_CHECK(runner, !detail::controlled_capacity_2_0x(
                            detail::m13_aggregate_logical,
                            ByteSize{std::numeric_limits<std::uint64_t>::max()}));
    runner.begin("M13 eight distinct M10-family variants preserve all six M12 definitions");
    std::array<std::array<std::byte, 69632U>, 8U> patterns{};
    for (std::size_t chunk = 0U; chunk < patterns.size(); ++chunk) {
        bool zero{}, nonzero{}, variation{};
        for (std::size_t index = 0U; index < patterns[chunk].size(); ++index) {
            auto& value = patterns[chunk][index];
            value = detail::controlled_capacity_2x_payload_byte(chunk, index);
            zero = zero || value == std::byte{};
            nonzero = nonzero || value != std::byte{};
            variation = variation || value != patterns[chunk][0];
            if (chunk < 6U) {
                VRAMZ_CHECK(runner,
                            value == detail::controlled_capacity_payload_byte(chunk, index));
            }
        }
        VRAMZ_CHECK(runner, zero && nonzero && variation);
        for (std::size_t prior = 0U; prior < chunk; ++prior) {
            VRAMZ_CHECK(runner, patterns[prior] != patterns[chunk]);
        }
    }
    runner.begin("M13 checked full stage admission accepts exact cap and rejects overflow");
    VRAMZ_CHECK(runner, detail::admit_controlled_capacity_2x_stage(ByteSize{134217727U},
                                                                   ByteSize{1U}, {}, {}));
    VRAMZ_CHECK(runner, !detail::admit_controlled_capacity_2x_stage(ByteSize{134217728U},
                                                                    ByteSize{1U}, {}, {}));
    VRAMZ_CHECK(runner,
                !detail::admit_controlled_capacity_2x_stage(
                    ByteSize{std::numeric_limits<std::uint64_t>::max()}, ByteSize{1U}, {}, {}));
    runner.begin("M13 bounded JSON keeps Fake hardware and Gate G unvalidated");
    detail::ControlledCapacity2xSmokeReport report{};
    std::array<char, 65536U> output{};
    const auto formatted =
        detail::format_controlled_capacity_2x_smoke_report(report, "", false, output);
    VRAMZ_CHECK(runner, !formatted.truncated);
    const std::string_view json{output.data(), formatted.written};
    for (const auto field :
         {"\"m13_version\":1", "\"manual_victim_selection\":false",
          "\"manual_compression_transition\":false", "\"gate_g\":\"NOT_RUN\"",
          "\"hardware_validation\":\"NOT_TESTED\"", "\"final_budget_charge\":null"}) {
        VRAMZ_CHECK(runner, json.find(field) != std::string_view::npos);
    }
    runner.begin("M13 full eight-cycle report remains bounded even with maximal counters");
    report.cycle_count = 8U;
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
        !detail::format_controlled_capacity_2x_smoke_report(report, "", false, output).truncated);
    std::array<char, 32U> tiny{};
    VRAMZ_CHECK(
        runner,
        detail::format_controlled_capacity_2x_smoke_report(report, "", false, tiny).truncated);
    return runner.finish();
}
