#include "vramz/capabilities.hpp"

#include <iostream>

int main() {
    const auto build = vramz::build_capabilities();
    std::cout << std::boolalpha << "{\"version\":\"" << build.version
              << "\",\"status\":\"PRE_HARDWARE_RC\",\"default_backend\":\"mock\""
              << ",\"cpu_lz4_compiled\":" << build.cpu_lz4_compiled << ",\"cpu_lz4_version\":\""
              << build.cpu_lz4_version
              << "\",\"cuda_adapter_compiled\":" << build.cuda_adapter_compiled
              << ",\"cuda_development_version\":\"" << build.cuda_development_version
              << "\",\"nvcomp_adapter_compiled\":" << build.nvcomp_adapter_compiled
              << ",\"nvcomp_development_version\":\"" << build.nvcomp_development_version
              << "\",\"real_gpu_execution_enabled\":" << build.real_gpu_execution_enabled
              << ",\"hardware_validation\":\"NOT_TESTED\",\"hardware_gates\":\"NOT RUN\"}\n";
    return std::cout ? 0 : 1;
}
