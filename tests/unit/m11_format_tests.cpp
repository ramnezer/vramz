#include "../test_support.hpp"
#include "vramz/detail/policy_pressure_smoke.hpp"
#include <array>
#include <limits>
int main() {
    using namespace vramz;
    test::Runner runner;
    runner.begin("M11 fixed argument gates reject device/count/size/cap/mode overrides");
    const std::array<std::string_view, 3U> args{"--run-policy-pressure-smoke", "--device", "0"};
    VRAMZ_CHECK(runner, detail::parse_policy_pressure_smoke_arguments(args));
    for (const auto bad : {std::array<std::string_view, 2U>{"--device", "1"},
                           {"--chunk-count", "5"},
                           {"--logical-bytes", "16"},
                           {"--max-physical-bytes", "1"},
                           {"--host-fallback", "on"},
                           {"--policy", "off"}}) {
        VRAMZ_CHECK(runner, !detail::parse_policy_pressure_smoke_arguments(bad));
    }
    runner.begin("M11 budget configuration uses the unchanged production pressure formula");
    const auto c = detail::policy_pressure_configuration();
    VRAMZ_CHECK(runner, c.policy.mode == PolicyMode::gpu_resident);
    VRAMZ_CHECK(runner, c.budgets.gpu.hard_limit == ByteSize{134217728U});
    VRAMZ_CHECK(runner, policy::pressure(ByteSize{33554432U}, c.budgets.gpu.soft_target,
                                         c.budgets.gpu.hard_limit,
                                         c.budgets.gpu.migration_reserve) == Pressure::hard);
    VRAMZ_CHECK(runner, policy::pressure(ByteSize{27262976U}, c.budgets.gpu.soft_target,
                                         c.budgets.gpu.hard_limit,
                                         c.budgets.gpu.migration_reserve) == Pressure::soft);
    VRAMZ_CHECK(runner, policy::pressure(ByteSize{20971520U}, c.budgets.gpu.soft_target,
                                         c.budgets.gpu.hard_limit,
                                         c.budgets.gpu.migration_reserve) == Pressure::normal);
    runner.begin("M11 checked full stage admission accepts exact cap and rejects overflow");
    VRAMZ_CHECK(runner,
                detail::admit_policy_pressure_stage(ByteSize{134217727U}, ByteSize{1U}, {}, {}));
    VRAMZ_CHECK(runner,
                !detail::admit_policy_pressure_stage(ByteSize{134217728U}, ByteSize{1U}, {}, {}));
    VRAMZ_CHECK(runner,
                !detail::admit_policy_pressure_stage(
                    ByteSize{std::numeric_limits<std::uint64_t>::max()}, ByteSize{1U}, {}, {}));
    runner.begin("M11 bounded JSON keeps Fake hardware and Gate G unvalidated");
    detail::PolicyPressureSmokeReport report{};
    std::array<char, 32768U> output{};
    const auto formatted = detail::format_policy_pressure_smoke_report(report, "", false, output);
    VRAMZ_CHECK(runner, !formatted.truncated);
    const std::string_view json{output.data(), formatted.written};
    for (const auto field :
         {"\"m11_version\":1", "\"manual_victim_selection\":false",
          "\"manual_compression_transition\":false", "\"gate_g\":\"NOT_RUN\"",
          "\"hardware_validation\":\"NOT_TESTED\"", "\"final_budget_charge\":null"}) {
        VRAMZ_CHECK(runner, json.find(field) != std::string_view::npos);
    }
    std::array<char, 32U> tiny{};
    VRAMZ_CHECK(runner,
                detail::format_policy_pressure_smoke_report(report, "", false, tiny).truncated);
    return runner.finish();
}
