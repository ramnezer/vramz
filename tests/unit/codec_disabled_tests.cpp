#include "../test_support.hpp"

using namespace vramz;

int main() {
    test::Runner runner;
    runner.begin("CPU LZ4 disabled keeps raw core and rejects compressed requests");
    auto runtime_result = Runtime::create(test::test_runtime_config());
    VRAMZ_CHECK(runner, runtime_result);
    auto runtime = std::move(runtime_result).value();
    VRAMZ_CHECK(runner, !testing::cpu_lz4_available(runtime));
    auto buffer_result = runtime.allocate(ByteSize{64U});
    VRAMZ_CHECK(runner, buffer_result);
    auto buffer = std::move(buffer_result).value();
    const auto compressed = testing::migrate(buffer, 0U, RepresentationState::gpu_compressed);
    VRAMZ_CHECK(runner, !compressed && compressed.error().code == ErrorCode::unsupported);
    auto lease = buffer.acquire(MemoryRange{ByteOffset{}, ByteSize{64U}});
    VRAMZ_CHECK(runner, lease);
    VRAMZ_CHECK(runner, lease.value().close());
    VRAMZ_CHECK(runner, buffer.close());
    VRAMZ_CHECK(runner, runtime.shutdown());
    runner.begin("automatic policy rejects a missing codec before runtime allocation");
    auto enabled = test::test_runtime_config();
    enabled.policy.mode = PolicyMode::gpu_resident;
    const auto unsupported = Runtime::create(enabled);
    VRAMZ_CHECK(runner, !unsupported && unsupported.error().code == ErrorCode::unsupported);
    return runner.finish();
}
