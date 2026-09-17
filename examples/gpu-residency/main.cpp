#include "vramz/cuda_runtime.hpp"
#include "workload.hpp"

#include <iostream>
#include <string_view>

int main(int argc, char* argv[]) {
    if (argc != 5 || std::string_view{argv[1]} != "--run-device-0-demo") {
        std::cerr << "Usage: vramz-public-demo --run-device-0-demo DRIVER CUDART NVCOMP\n";
        return 2;
    }
    vramz::example::WorkloadConfig workload{};
    const auto config = vramz::example::runtime_config(workload);
    if (!config) {
        return 2;
    }
    const vramz::CudaRuntimeOptions options{true, 0, argv[2], argv[3], argv[4]};
    vramz::CudaRuntimeInfo info{};
    auto runtime = vramz::Runtime::create_cuda(config.value(), options, info);
    if (!runtime) {
        std::cerr << "CUDA runtime creation failed: " << static_cast<unsigned>(runtime.error().code)
                  << '\n';
        return 1;
    }
    vramz::example::WorkloadReport report{};
    vramz::example::run_workload(runtime.value(), workload, report);
    vramz::example::print_report(std::cout, report, true);
    return report.passed ? 0 : 1;
}
