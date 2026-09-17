cmake_minimum_required(VERSION 3.25)

project(VRAMZ VERSION 1.0.0 LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VRAMZ_ENABLE_CUDA "Compile/link the M4 Driver adapter against a development stub only" OFF)
option(VRAMZ_ENABLE_NVCOMP "Compile/link the isolated M5 nvCOMP LZ4 adapter; never execute it" OFF)
option(VRAMZ_ENABLE_CPU_LZ4 "Enable the M2 CPU LZ4 Block reference codec" ON)
option(VRAMZ_ENABLE_FUZZING "Build the optional bounded Clang/libFuzzer codec target" OFF)
option(VRAMZ_BUILD_TESTS "Build deterministic tests" ON)
option(VRAMZ_ALLOW_REAL_GPU_EXECUTION "Permit a future separately reviewed hardware execution path" OFF)
set(VRAMZ_SANITIZER "none" CACHE STRING "Sanitizer: none, address, leak, undefined, or thread")
set_property(CACHE VRAMZ_SANITIZER PROPERTY STRINGS none address leak undefined thread)
include(cmake/PhysicalRuntime.cmake)
include(cmake/M7RawSmoke.cmake)
option(VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE "Build the separately reviewed M9 physical savings smoke" OFF)
option(VRAMZ_BUILD_M10_MULTI_CHUNK_RESIDENCY_SMOKE "Build the separately reviewed M10 multi-chunk residency smoke" OFF)
option(VRAMZ_BUILD_M12_CONTROLLED_CAPACITY_SMOKE "Build the separately reviewed M12 controlled capacity smoke" OFF)
option(VRAMZ_BUILD_M11_POLICY_PRESSURE_SMOKE "Build the separately reviewed M11 policy pressure smoke" OFF)
option(VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE "Build the separately reviewed M13 controlled 2x capacity smoke" OFF)
set(VRAMZ_REVIEWED_LZ4_PREFIX "" CACHE PATH "Optional reviewed static LZ4 1.9.4 prefix for historical physical-smoke reproduction")
include(cmake/M8CompressionSmoke.cmake)
include(cmake/M9PhysicalSavingsSmoke.cmake)
include(cmake/M10MultiChunkResidencySmoke.cmake)
include(cmake/M11PolicyPressureSmoke.cmake)
include(cmake/M12ControlledCapacitySmoke.cmake)
include(cmake/M13ControlledCapacity2xSmoke.cmake)

if(VRAMZ_ENABLE_NVCOMP AND NOT VRAMZ_ENABLE_CUDA)
    message(FATAL_ERROR "VRAMZ_ENABLE_NVCOMP=ON requires VRAMZ_ENABLE_CUDA=ON")
endif()

if(VRAMZ_ENABLE_CUDA)
    # M4 never enables the CUDA language or executes a driver probe. An explicit isolated
    # Toolkit is mandatory so discovery cannot silently choose a system driver.
    if(NOT CUDAToolkit_ROOT)
        message(FATAL_ERROR "M4 CUDA compile validation requires explicit CUDAToolkit_ROOT")
    endif()
    find_package(CUDAToolkit 11.2 REQUIRED)
    find_library(VRAMZ_CUDA_DRIVER_STUB NAMES cuda
        PATHS "${CUDAToolkit_ROOT}/lib/stubs" "${CUDAToolkit_ROOT}/lib64/stubs"
              "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib/stubs"
        NO_DEFAULT_PATH NO_CACHE REQUIRED)
    file(REAL_PATH "${CUDAToolkit_ROOT}" VRAMZ_CUDA_ROOT_REAL)
    file(REAL_PATH "${VRAMZ_CUDA_DRIVER_STUB}" VRAMZ_CUDA_STUB_REAL)
    cmake_path(IS_PREFIX VRAMZ_CUDA_ROOT_REAL "${VRAMZ_CUDA_STUB_REAL}" NORMALIZE VRAMZ_STUB_IS_LOCAL)
    if(NOT VRAMZ_STUB_IS_LOCAL OR NOT VRAMZ_CUDA_STUB_REAL MATCHES "/stubs/libcuda\\.so$")
        message(FATAL_ERROR "M4 requires libcuda.so inside the isolated Toolkit stubs directory")
    endif()
    add_library(VRAMZ_CUDA_DRIVER_STUB_TARGET SHARED IMPORTED)
    set_target_properties(VRAMZ_CUDA_DRIVER_STUB_TARGET PROPERTIES
        IMPORTED_LOCATION "${VRAMZ_CUDA_STUB_REAL}")
    message(STATUS "M4 compile/link ONLY: CUDA ${CUDAToolkit_VERSION}; stub=${VRAMZ_CUDA_STUB_REAL}")
endif()

if(NOT VRAMZ_ENABLE_PHYSICAL_RUNTIME AND NOT VRAMZ_BUILD_M8_COMPRESSION_SMOKE AND NOT VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE AND
   NOT VRAMZ_BUILD_M10_MULTI_CHUNK_RESIDENCY_SMOKE AND NOT VRAMZ_BUILD_M11_POLICY_PRESSURE_SMOKE AND NOT VRAMZ_BUILD_M12_CONTROLLED_CAPACITY_SMOKE AND NOT VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE)
    vramz_validate_m7_configuration()
endif()

if(VRAMZ_ENABLE_CPU_LZ4)
    find_path(VRAMZ_LZ4_INCLUDE_DIR NAMES lz4.h)
    find_library(VRAMZ_LZ4_LIBRARY NAMES lz4)
    if(NOT VRAMZ_LZ4_INCLUDE_DIR OR NOT VRAMZ_LZ4_LIBRARY)
        message(FATAL_ERROR
            "VRAMZ_ENABLE_CPU_LZ4=ON requires system liblz4 development files >= 1.9.4 "
            "(Debian/Ubuntu package: liblz4-dev). VRAMZ does not download dependencies.")
    endif()
    file(STRINGS "${VRAMZ_LZ4_INCLUDE_DIR}/lz4.h" VRAMZ_LZ4_MAJOR_LINE
         REGEX "^#define LZ4_VERSION_MAJOR +[0-9]+")
    file(STRINGS "${VRAMZ_LZ4_INCLUDE_DIR}/lz4.h" VRAMZ_LZ4_MINOR_LINE
         REGEX "^#define LZ4_VERSION_MINOR +[0-9]+")
    file(STRINGS "${VRAMZ_LZ4_INCLUDE_DIR}/lz4.h" VRAMZ_LZ4_RELEASE_LINE
         REGEX "^#define LZ4_VERSION_RELEASE +[0-9]+")
    if(NOT VRAMZ_LZ4_MAJOR_LINE OR NOT VRAMZ_LZ4_MINOR_LINE OR NOT VRAMZ_LZ4_RELEASE_LINE)
        message(FATAL_ERROR "Unable to determine the system liblz4 version from lz4.h")
    endif()
    string(REGEX REPLACE ".*LZ4_VERSION_MAJOR +([0-9]+).*" "\\1" VRAMZ_LZ4_MAJOR
                         "${VRAMZ_LZ4_MAJOR_LINE}")
    string(REGEX REPLACE ".*LZ4_VERSION_MINOR +([0-9]+).*" "\\1" VRAMZ_LZ4_MINOR
                         "${VRAMZ_LZ4_MINOR_LINE}")
    string(REGEX REPLACE ".*LZ4_VERSION_RELEASE +([0-9]+).*" "\\1" VRAMZ_LZ4_RELEASE
                         "${VRAMZ_LZ4_RELEASE_LINE}")
    set(VRAMZ_LZ4_VERSION "${VRAMZ_LZ4_MAJOR}.${VRAMZ_LZ4_MINOR}.${VRAMZ_LZ4_RELEASE}")
    if(VRAMZ_LZ4_VERSION VERSION_LESS 1.9.4)
        message(FATAL_ERROR "VRAMZ M2 requires system LZ4 >= 1.9.4; found ${VRAMZ_LZ4_VERSION}")
    endif()
    add_library(VRAMZ_LZ4 UNKNOWN IMPORTED)
    set_target_properties(VRAMZ_LZ4 PROPERTIES
        IMPORTED_LOCATION "${VRAMZ_LZ4_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${VRAMZ_LZ4_INCLUDE_DIR}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${VRAMZ_LZ4_INCLUDE_DIR}")
    message(STATUS "VRAMZ CPU LZ4 Block backend: ${VRAMZ_LZ4_VERSION}")
endif()

include(cmake/Warnings.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/Quality.cmake)

if(VRAMZ_ENABLE_NVCOMP)
    if(NOT CUDAToolkit_VERSION VERSION_EQUAL 13.3.1)
        message(FATAL_ERROR "M5 compile validation is pinned to the isolated CUDA 13.3.1 Toolkit")
    endif()
    get_target_property(VRAMZ_CUDART_LIBRARY CUDA::cudart IMPORTED_LOCATION)
    file(REAL_PATH "${VRAMZ_CUDART_LIBRARY}" VRAMZ_CUDART_LIBRARY_REAL)
    cmake_path(IS_PREFIX VRAMZ_CUDA_ROOT_REAL "${VRAMZ_CUDART_LIBRARY_REAL}" NORMALIZE VRAMZ_CUDART_LOCAL)
    if(NOT VRAMZ_CUDART_LOCAL)
        message(FATAL_ERROR "M5 requires libcudart inside the explicit isolated CUDA Toolkit")
    endif()
    if(NOT VRAMZ_NVCOMP_ROOT)
        message(FATAL_ERROR "M5 requires an explicit isolated VRAMZ_NVCOMP_ROOT (nvCOMP 5.3.0 CUDA 13)")
    endif()
    find_package(nvcomp 5.3.0.16 EXACT CONFIG REQUIRED COMPONENTS nvcomp
        PATHS "${VRAMZ_NVCOMP_ROOT}/lib/cmake/nvcomp" NO_DEFAULT_PATH)
    get_target_property(VRAMZ_NVCOMP_LIBRARY nvcomp::nvcomp IMPORTED_LOCATION_RELEASE)
    file(REAL_PATH "${VRAMZ_NVCOMP_ROOT}" VRAMZ_NVCOMP_ROOT_REAL)
    file(REAL_PATH "${VRAMZ_NVCOMP_LIBRARY}" VRAMZ_NVCOMP_LIBRARY_REAL)
    cmake_path(IS_PREFIX VRAMZ_NVCOMP_ROOT_REAL "${VRAMZ_NVCOMP_LIBRARY_REAL}" NORMALIZE VRAMZ_NVCOMP_LOCAL)
    if(NOT VRAMZ_NVCOMP_LOCAL OR NOT VRAMZ_NVCOMP_LIBRARY_REAL MATCHES "/libnvcomp\\.so\\.5\\.3\\.0\\.16$")
        message(FATAL_ERROR "M5 requires the isolated nvCOMP 5.3.0.16 CUDA 13 shared library")
    endif()
    # Derived from the official SHA256-verified CUDA 13 archive recorded in the M5 checkpoint.
    # Pinning this compile-only artifact rejects a differently built/versioned library even
    # when its filename or package metadata claims to match. No runtime probe is performed.
    file(SHA256 "${VRAMZ_NVCOMP_LIBRARY_REAL}" VRAMZ_NVCOMP_LIBRARY_SHA256)
    if(NOT VRAMZ_NVCOMP_LIBRARY_SHA256 STREQUAL "42582e1075df6fe93e1045430fadf5774218e25f990f8fb248bc9352dc806011")
        message(FATAL_ERROR "M5 nvCOMP library does not match the verified CUDA 13 build")
    endif()
    add_library(vramz_nvcomp_lz4 SHARED src/backends/nvcomp/nvcomp_lz4_api.cpp)
    target_compile_features(vramz_nvcomp_lz4 PRIVATE cxx_std_20)
    target_include_directories(vramz_nvcomp_lz4 PRIVATE include)
    target_include_directories(vramz_nvcomp_lz4 SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${nvcomp_INCLUDE_DIR})
    target_link_libraries(vramz_nvcomp_lz4 PRIVATE nvcomp::nvcomp CUDA::cudart)
    target_link_options(vramz_nvcomp_lz4 PRIVATE -Wl,--no-undefined)
    vramz_set_project_warnings(vramz_nvcomp_lz4)
    vramz_enable_clang_tidy(vramz_nvcomp_lz4)
    message(STATUS "M5 compile/link ONLY: nvCOMP ${nvcomp_VERSION}; library=${VRAMZ_NVCOMP_LIBRARY_REAL}")
endif()

if(VRAMZ_ENABLE_CUDA)
    # A linked artifact, not an execution target. No test/tool links this library.
    add_library(vramz_cuda_driver SHARED src/backends/cuda_vmm/cuda_driver.cpp)
    target_compile_features(vramz_cuda_driver PRIVATE cxx_std_20)
    target_include_directories(vramz_cuda_driver PRIVATE include)
    target_include_directories(vramz_cuda_driver SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(vramz_cuda_driver PRIVATE VRAMZ_CUDA_DRIVER_STUB_TARGET)
    target_link_options(vramz_cuda_driver PRIVATE -Wl,--no-undefined)
    vramz_set_project_warnings(vramz_cuda_driver)
    vramz_enable_clang_tidy(vramz_cuda_driver)
endif()

set(VRAMZ_SOURCES
    src/core/async_error_channel.cpp
    src/core/budget_ledger.cpp
    src/core/crc32c.cpp
    src/core/compression_smoke.cpp
    src/core/compression_smoke_format.cpp
    src/core/controlled_capacity_smoke.cpp
    src/core/controlled_capacity_smoke_format.cpp
    src/core/controlled_capacity_2x_smoke.cpp
    src/core/controlled_capacity_2x_smoke_format.cpp
    src/core/policy_pressure_smoke.cpp
    src/core/policy_pressure_smoke_format.cpp
    src/core/multi_chunk_residency_smoke.cpp
    src/core/multi_chunk_residency_smoke_format.cpp
    src/core/error.cpp
    src/core/raw_smoke.cpp
    src/core/raw_smoke_format.cpp
    src/core/runtime.cpp
    src/core/transaction.cpp
    src/policy/policy.cpp
    src/simulation/scenario.cpp
    src/backends/cuda_vmm/cuda_vmm_backend.cpp
    src/backends/cuda_vmm/gpu_compression.cpp
    src/backends/cuda_vmm/timed_codec.cpp
    src/backends/mock/mock_backend.cpp)
if(VRAMZ_ENABLE_CPU_LZ4)
    list(APPEND VRAMZ_SOURCES src/backends/cpu/lz4_codec.cpp)
else()
    list(APPEND VRAMZ_SOURCES src/backends/cpu/null_codec.cpp)
endif()
add_library(vramz ${VRAMZ_SOURCES})
add_library(VRAMZ::vramz ALIAS vramz)

foreach(feature CPU_LZ4 CUDA NVCOMP)
    if(VRAMZ_ENABLE_${feature})
        set(VRAMZ_BUILD_${feature} true)
    else()
        set(VRAMZ_BUILD_${feature} false)
    endif()
endforeach()
set(VRAMZ_BUILD_PHYSICAL false)
if(VRAMZ_ENABLE_PHYSICAL_RUNTIME)
    set(VRAMZ_BUILD_PHYSICAL true)
endif()
configure_file(cmake/build_capabilities.cpp.in generated/build_capabilities.cpp @ONLY)
target_sources(vramz PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/build_capabilities.cpp")
# Installed objects must not retain operational dependencies on the checkout/build roots.
target_compile_options(vramz PRIVATE
    "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=."
    "-ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=build")

target_compile_features(vramz PUBLIC cxx_std_20)
target_include_directories(vramz PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
vramz_set_project_warnings(vramz)
vramz_enable_sanitizer(vramz)
vramz_enable_clang_tidy(vramz)
if(VRAMZ_ENABLE_CPU_LZ4)
    target_link_libraries(vramz PRIVATE VRAMZ_LZ4)
    target_compile_definitions(vramz PUBLIC VRAMZ_CPU_LZ4_ENABLED=1)
else()
    target_compile_definitions(vramz PUBLIC VRAMZ_CPU_LZ4_ENABLED=0)
endif()

find_package(Threads REQUIRED)
add_library(vramz_workload STATIC tools/common/workload.cpp)
target_link_libraries(vramz_workload PUBLIC VRAMZ::vramz Threads::Threads)
vramz_set_project_warnings(vramz_workload)
vramz_enable_sanitizer(vramz_workload)
vramz_enable_clang_tidy(vramz_workload)

vramz_add_physical_runtime()

vramz_add_m7_raw_smoke(vramz-m7-raw-smoke tools/vramz-m7-raw-smoke/main.cpp)
vramz_add_m7_compile_validation(vramz_m7_main_compile_check tools/vramz-m7-raw-smoke/main.cpp)
vramz_add_m8_smoke()
vramz_add_m9_smoke()
vramz_add_m10_smoke()
vramz_add_m11_smoke()
vramz_add_m12_smoke()
vramz_add_m13_smoke()

file(GLOB_RECURSE VRAMZ_HEADERS CONFIGURE_DEPENDS include/vramz/*.hpp)
file(GLOB_RECURSE VRAMZ_TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp tests/*.hpp tools/*.cpp)
vramz_add_quality_targets(${VRAMZ_SOURCES} ${VRAMZ_HEADERS} ${VRAMZ_TEST_SOURCES}
    src/backends/cuda_vmm/cuda_driver.cpp src/backends/cuda_vmm/physical_runtime.cpp
    src/backends/nvcomp/nvcomp_lz4_api.cpp)

add_executable(vramz-policy-sim tools/vramz-policy-sim/main.cpp)
target_link_libraries(vramz-policy-sim PRIVATE VRAMZ::vramz)
vramz_set_project_warnings(vramz-policy-sim)
vramz_enable_sanitizer(vramz-policy-sim)
vramz_enable_clang_tidy(vramz-policy-sim)

add_executable(vramz-info tools/vramz-info/main.cpp)
target_link_libraries(vramz-info PRIVATE VRAMZ::vramz)
vramz_set_project_warnings(vramz-info)
vramz_enable_sanitizer(vramz-info)
vramz_enable_clang_tidy(vramz-info)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/VRAMZConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/VRAMZConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VRAMZ")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/VRAMZConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(TARGETS vramz vramz-info vramz-policy-sim EXPORT VRAMZTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(EXPORT VRAMZTargets NAMESPACE VRAMZ:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VRAMZ")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/VRAMZConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/VRAMZConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VRAMZ")
install(FILES include/vramz/runtime.hpp include/vramz/config.hpp include/vramz/policy_types.hpp
    include/vramz/completion.hpp include/vramz/capabilities.hpp include/vramz/types.hpp
    include/vramz/error.hpp include/vramz/result.hpp include/vramz/stats.hpp
    include/vramz/checked.hpp include/vramz/crc32c.hpp
    include/vramz/performance.hpp include/vramz/timing.hpp
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/vramz")
install(FILES README.md LICENSE NOTICE.md CITATION.cff
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/vramz")
if(VRAMZ_ENABLE_PHYSICAL_RUNTIME)
    include(cmake/Packaging.cmake)
endif()
# Install public documentation only; regression fixtures stay in the source tree.
install(DIRECTORY docs/ DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/vramz/docs"
    FILES_MATCHING PATTERN "*.md" PATTERN "THIRD_PARTY*.txt")
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    set(VRAMZ_PREHARDWARE_INSTALL "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "Local package evidence root; never a system install")
    add_custom_target(vramz-prehardware-check
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/vramz-prehardware-check.py"
            --source "${CMAKE_CURRENT_SOURCE_DIR}" --build "${CMAKE_CURRENT_BINARY_DIR}"
            --install "${VRAMZ_PREHARDWARE_INSTALL}"
        VERBATIM)
endif()

if(VRAMZ_BUILD_TESTS)
    enable_testing()
    find_package(Threads REQUIRED)

    function(vramz_add_test target source label)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE VRAMZ::vramz Threads::Threads)
        vramz_set_project_warnings(${target})
        vramz_enable_sanitizer(${target})
        vramz_enable_clang_tidy(${target})
        if(VRAMZ_SANITIZER STREQUAL "thread" AND UNIX)
            find_program(VRAMZ_SETARCH_EXECUTABLE setarch REQUIRED)
            add_test(NAME ${target}
                     COMMAND ${VRAMZ_SETARCH_EXECUTABLE} ${CMAKE_SYSTEM_PROCESSOR} -R
                             $<TARGET_FILE:${target}>)
        else()
            add_test(NAME ${target} COMMAND ${target})
        endif()
        if(VRAMZ_SANITIZER STREQUAL "none")
            set(VRAMZ_TEST_TIMEOUT 60)
        else()
            set(VRAMZ_TEST_TIMEOUT 120)
        endif()
        set_tests_properties(${target} PROPERTIES LABELS "${label}" TIMEOUT ${VRAMZ_TEST_TIMEOUT})
    endfunction()

    add_library(vramz_fake_cuda STATIC tests/support/fake_cuda_driver.cpp tests/support/fake_nvcomp.cpp)
    target_link_libraries(vramz_fake_cuda PRIVATE VRAMZ::vramz Threads::Threads)
    vramz_set_project_warnings(vramz_fake_cuda)
    vramz_enable_sanitizer(vramz_fake_cuda)
    vramz_enable_clang_tidy(vramz_fake_cuda)
    vramz_add_test(vramz_cuda_vmm_tests tests/integration/cuda_vmm_tests.cpp "unit;integration;property;fault")
    target_link_libraries(vramz_cuda_vmm_tests PRIVATE vramz_fake_cuda)
    vramz_add_test(vramz_m7_smoke_tests tests/integration/m7_smoke_tests.cpp "unit;integration;property;fault")
    target_link_libraries(vramz_m7_smoke_tests PRIVATE vramz_fake_cuda)
    vramz_add_test(vramz_m7_format_tests tests/unit/m7_format_tests.cpp "unit")
    vramz_add_test(vramz_m8_format_tests tests/unit/m8_format_tests.cpp "unit")
    vramz_add_test(vramz_m9_format_tests tests/unit/m9_format_tests.cpp "unit")
    vramz_add_test(vramz_m10_format_tests tests/unit/m10_format_tests.cpp "unit")
    vramz_add_test(vramz_m13_format_tests tests/unit/m13_format_tests.cpp "unit")
    vramz_add_test(vramz_m12_format_tests tests/unit/m12_format_tests.cpp "unit")
    vramz_add_test(vramz_m11_format_tests tests/unit/m11_format_tests.cpp "unit")

    vramz_add_test(vramz_crc32c_tests tests/property/crc32c_tests.cpp "unit;property")
    if(VRAMZ_ENABLE_CPU_LZ4)
        vramz_add_test(vramz_v1_runtime_tests tests/integration/v1_runtime_tests.cpp "unit;integration;property")
        target_link_libraries(vramz_v1_runtime_tests PRIVATE vramz_workload vramz_fake_cuda)
        set_tests_properties(vramz_v1_runtime_tests PROPERTIES TIMEOUT 1800)
        vramz_add_test(vramz_m13_smoke_tests tests/integration/m13_smoke_tests.cpp "unit;integration;property;fault")
        target_link_libraries(vramz_m13_smoke_tests PRIVATE vramz_fake_cuda)
        set_tests_properties(vramz_m13_smoke_tests PROPERTIES TIMEOUT 14400)
        vramz_add_test(vramz_m12_smoke_tests tests/integration/m12_smoke_tests.cpp "unit;integration;property;fault")
        target_link_libraries(vramz_m12_smoke_tests PRIVATE vramz_fake_cuda)
        set_tests_properties(vramz_m12_smoke_tests PROPERTIES TIMEOUT 10800)
        vramz_add_test(vramz_m11_smoke_tests tests/integration/m11_smoke_tests.cpp "unit;integration;property;fault")
        target_link_libraries(vramz_m11_smoke_tests PRIVATE vramz_fake_cuda)
        set_tests_properties(vramz_m11_smoke_tests PROPERTIES TIMEOUT 7200)
        vramz_add_test(vramz_m10_smoke_tests tests/integration/m10_smoke_tests.cpp "unit;integration;property;fault")
        target_link_libraries(vramz_m10_smoke_tests PRIVATE vramz_fake_cuda)
        set_tests_properties(vramz_m10_smoke_tests PROPERTIES TIMEOUT 7200)
        vramz_add_test(vramz_m9_smoke_tests tests/integration/m9_smoke_tests.cpp "unit;integration;property;fault")
        target_link_libraries(vramz_m9_smoke_tests PRIVATE vramz_fake_cuda)
        # Every M9 fault uses the fixed 8 MiB payload; instrumented CPU CRC work is larger.
        set_tests_properties(vramz_m9_smoke_tests PROPERTIES TIMEOUT 1800)
        vramz_add_test(vramz_m8_smoke_tests tests/integration/m8_smoke_tests.cpp "unit;integration;property;fault")
        target_link_libraries(vramz_m8_smoke_tests PRIVATE vramz_fake_cuda)
        vramz_add_test(vramz_nvcomp_tests tests/integration/nvcomp_tests.cpp "unit;integration;property;fault")
        target_link_libraries(vramz_nvcomp_tests PRIVATE vramz_fake_cuda)
        vramz_add_test(vramz_m6_completion_tests tests/integration/m6_completion_tests.cpp "unit;integration;property;fault")
        target_link_libraries(vramz_m6_completion_tests PRIVATE vramz_fake_cuda)
        vramz_add_test(vramz_unit_tests tests/unit/unit_tests.cpp unit)
        vramz_add_test(vramz_property_tests tests/property/property_tests.cpp property)
        vramz_add_test(vramz_integration_tests tests/integration/integration_tests.cpp integration)
        vramz_add_test(vramz_fault_tests tests/fault/fault_tests.cpp fault)
        vramz_add_test(vramz_m2_integrity_tests tests/integration/m2_integrity_tests.cpp integration)
        vramz_add_test(vramz_m2_fault_tests tests/fault/m2_fault_tests.cpp fault)
        vramz_add_test(vramz_allocation_fault_tests tests/fault/allocation_fault_tests.cpp fault)
        vramz_add_test(vramz_policy_unit_tests tests/unit/policy_tests.cpp unit)
        vramz_add_test(vramz_policy_integration_tests tests/integration/policy_tests.cpp integration)
        vramz_add_test(vramz_policy_property_tests tests/property/policy_tests.cpp property)
        vramz_add_test(vramz_policy_fault_tests tests/fault/policy_tests.cpp fault)
    else()
        vramz_add_test(vramz_codec_disabled_tests tests/unit/codec_disabled_tests.cpp unit)
    endif()
endif()

if(VRAMZ_ENABLE_FUZZING)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "VRAMZ_ENABLE_FUZZING=ON requires Clang with libFuzzer")
    endif()
    if(NOT VRAMZ_ENABLE_CPU_LZ4)
        message(FATAL_ERROR "VRAMZ_ENABLE_FUZZING=ON requires VRAMZ_ENABLE_CPU_LZ4=ON")
    endif()
    target_compile_options(vramz PRIVATE -fsanitize=fuzzer-no-link,address,undefined)
    # Every consumer of the instrumented static library needs its sanitizer runtimes.
    target_link_options(vramz INTERFACE -fsanitize=address,undefined)
    add_executable(vramz_codec_fuzzer tests/fuzz/codec_fuzzer.cpp)
    target_link_libraries(vramz_codec_fuzzer PRIVATE VRAMZ::vramz)
    target_compile_options(vramz_codec_fuzzer PRIVATE -fsanitize=fuzzer,address,undefined)
    target_link_options(vramz_codec_fuzzer PRIVATE -fsanitize=fuzzer,address,undefined)
    vramz_set_project_warnings(vramz_codec_fuzzer)
endif()
