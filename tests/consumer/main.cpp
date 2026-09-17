#include <vramz/runtime.hpp>

#include <type_traits>

static_assert(!std::is_move_assignable_v<vramz::PendingLeaseRelease>);
static_assert(!std::is_move_assignable_v<vramz::Runtime>);

int main() {
    vramz::RuntimeConfig config{};
    config.budgets.gpu = {vramz::ByteSize{8192U}, vramz::ByteSize{6144U}, vramz::ByteSize{2048U}};
    config.required_capabilities.host_tier = false;
    config.preferred_chunk_size = vramz::ByteSize{64U};
    config.policy.mode = vramz::PolicyMode::disabled;
    auto runtime = vramz::Runtime::create(config);
    if (!runtime) {
        return 1;
    }
    auto buffer = runtime.value().allocate(vramz::ByteSize{64U});
    if (!buffer || !buffer.value().close() || !runtime.value().shutdown()) {
        return 2;
    }
    return vramz::build_capabilities().real_gpu_execution_enabled ? 3 : 0;
}
