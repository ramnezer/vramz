# M8A compiles an object entry point. Only separately reviewed M8B may build a runnable target.
option(VRAMZ_BUILD_M8_COMPRESSION_SMOKE "Build the explicitly reviewed M8 compression smoke" OFF)
option(VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION "Permit only a separately gated reviewed compression smoke" OFF)
set(VRAMZ_M8_SOURCE_SHA256 "" CACHE STRING "Independent reviewed M8 source archive SHA256")

if(VRAMZ_BUILD_M8_COMPRESSION_SMOKE AND VRAMZ_BUILD_M7_RAW_SMOKE)
    message(FATAL_ERROR "M7 RAW and M8 compression smoke gates are mutually exclusive")
endif()
if(VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE AND (VRAMZ_BUILD_M7_RAW_SMOKE OR VRAMZ_BUILD_M8_COMPRESSION_SMOKE))
    message(FATAL_ERROR "M7, M8 and M9 physical smoke gates are mutually exclusive")
endif()
if(VRAMZ_BUILD_M10_MULTI_CHUNK_RESIDENCY_SMOKE AND
   (VRAMZ_BUILD_M7_RAW_SMOKE OR VRAMZ_BUILD_M8_COMPRESSION_SMOKE OR VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE))
    message(FATAL_ERROR "M7, M8, M9 and M10 physical smoke gates are mutually exclusive")
endif()
if(VRAMZ_BUILD_M11_POLICY_PRESSURE_SMOKE AND
   (VRAMZ_BUILD_M7_RAW_SMOKE OR VRAMZ_BUILD_M8_COMPRESSION_SMOKE OR VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE OR VRAMZ_BUILD_M10_MULTI_CHUNK_RESIDENCY_SMOKE))
    message(FATAL_ERROR "M7, M8, M9, M10 and M11 physical smoke gates are mutually exclusive")
endif()
if(VRAMZ_BUILD_M12_CONTROLLED_CAPACITY_SMOKE AND
   (VRAMZ_BUILD_M7_RAW_SMOKE OR VRAMZ_BUILD_M8_COMPRESSION_SMOKE OR VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE OR VRAMZ_BUILD_M10_MULTI_CHUNK_RESIDENCY_SMOKE OR VRAMZ_BUILD_M11_POLICY_PRESSURE_SMOKE))
    message(FATAL_ERROR "M7 through M12 physical smoke gates are mutually exclusive")
endif()
if(VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE AND
   (VRAMZ_BUILD_M7_RAW_SMOKE OR VRAMZ_BUILD_M8_COMPRESSION_SMOKE OR VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE OR VRAMZ_BUILD_M10_MULTI_CHUNK_RESIDENCY_SMOKE OR VRAMZ_BUILD_M11_POLICY_PRESSURE_SMOKE OR VRAMZ_BUILD_M12_CONTROLLED_CAPACITY_SMOKE))
    message(FATAL_ERROR "M7 through M13 physical smoke gates are mutually exclusive")
endif()
if(VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION AND NOT VRAMZ_ENABLE_PHYSICAL_RUNTIME AND NOT VRAMZ_BUILD_M8_COMPRESSION_SMOKE AND
   NOT VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE AND NOT VRAMZ_BUILD_M10_MULTI_CHUNK_RESIDENCY_SMOKE AND
   NOT VRAMZ_BUILD_M11_POLICY_PRESSURE_SMOKE AND NOT VRAMZ_BUILD_M12_CONTROLLED_CAPACITY_SMOKE AND NOT VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE)
    message(FATAL_ERROR "Real nvCOMP execution requires the explicit M8, M9, M10, M11, M12 or M13 target gate")
endif()

function(vramz_add_m8_smoke)
    if(VRAMZ_BUILD_M8_COMPRESSION_SMOKE)
        if(NOT VRAMZ_ALLOW_REAL_GPU_EXECUTION OR NOT VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION OR
           NOT VRAMZ_ENABLE_CUDA OR NOT VRAMZ_ENABLE_NVCOMP OR NOT VRAMZ_ENABLE_CPU_LZ4 OR
           VRAMZ_BUILD_TESTS OR VRAMZ_ENABLE_FUZZING OR NOT VRAMZ_SANITIZER STREQUAL "none")
            message(FATAL_ERROR "M8 requires CUDA/nvCOMP/CPU-LZ4 and both execution gates ON; tests/fuzzing/sanitizers OFF")
        endif()
        string(LENGTH "${VRAMZ_M8_SOURCE_SHA256}" length)
        if(NOT length EQUAL 64 OR NOT VRAMZ_M8_SOURCE_SHA256 MATCHES "^[0-9a-f]+$" OR
           VRAMZ_M8_SOURCE_SHA256 MATCHES "^0+$" OR
           VRAMZ_M8_SOURCE_SHA256 STREQUAL "c67dc98206470c74045aa0c7ed1ae69ae596b2f7687a966cab6c6b7e3dec93bd" OR
           VRAMZ_M8_SOURCE_SHA256 STREQUAL "56b3a392464febf9b5ec8f49653b68a2de5bd50c2bd814a1d38ac7b133334c76")
            message(FATAL_ERROR "M8 requires its own reviewed source archive SHA256, never the M7 identity")
        endif()
        find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
        execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/ci/m8-elf-check.py"
            --driver "${VRAMZ_REAL_CUDA_DRIVER_LIBRARY}" --cuda-root "${CUDAToolkit_ROOT}"
            --nvcomp-root "${VRAMZ_NVCOMP_ROOT}" --identity-header "${CMAKE_BINARY_DIR}/m8-identities.cmake"
            RESULT_VARIABLE status OUTPUT_VARIABLE evidence ERROR_VARIABLE diagnostic TIMEOUT 30)
        if(NOT status EQUAL 0)
            message(FATAL_ERROR "M8 static identity/closure failure: ${evidence}${diagnostic}")
        endif()
        include("${CMAKE_BINARY_DIR}/m8-identities.cmake")
        file(WRITE "${CMAKE_BINARY_DIR}/m8-sdk-elf.json" "${evidence}")
        set(target vramz-m8-compression-smoke)
        add_library(VRAMZ_M8_REAL_DRIVER SHARED IMPORTED)
        set_target_properties(VRAMZ_M8_REAL_DRIVER PROPERTIES IMPORTED_LOCATION "${VRAMZ_M8_DRIVER_PATH}")
        add_executable(${target} tools/vramz-m8-compression-smoke/main.cpp
            src/backends/cuda_vmm/cuda_driver.cpp src/backends/nvcomp/nvcomp_lz4_api.cpp)
        target_compile_definitions(${target} PRIVATE VRAMZ_M8_COMPRESSION_ONLY=1)
        target_link_libraries(${target} PRIVATE VRAMZ::vramz VRAMZ_M8_REAL_DRIVER nvcomp::nvcomp CUDA::cudart ${CMAKE_DL_LIBS})
        target_link_options(${target} PRIVATE -Wl,--no-undefined)
        set_target_properties(${target} PROPERTIES SKIP_BUILD_RPATH TRUE BUILD_RPATH "" INSTALL_RPATH "")
        add_custom_target(${target}-elf-check
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/ci/m8-elf-check.py"
                --driver "${VRAMZ_REAL_CUDA_DRIVER_LIBRARY}" --cuda-root "${CUDAToolkit_ROOT}"
                --nvcomp-root "${VRAMZ_NVCOMP_ROOT}" --binary "$<TARGET_FILE:${target}>"
            DEPENDS ${target} VERBATIM)
    elseif(VRAMZ_ENABLE_CUDA AND VRAMZ_ENABLE_NVCOMP)
        set(target vramz_m8_main_compile_check)
        add_library(${target} OBJECT tools/vramz-m8-compression-smoke/main.cpp)
        set(VRAMZ_M8_SOURCE_SHA256 "")
        set(VRAMZ_M8_LOADER_PATH "")
        set(VRAMZ_M8_DRIVER_RELEASE "0U, 0U, 0U")
        set(VRAMZ_M8_PRIOR_DRIVER_API 0)
        foreach(kind DRIVER NVCOMP CUDART)
            set(VRAMZ_M8_${kind}_IDENTITY "\"\", \"\", {}")
        endforeach()
    else()
        return()
    endif()
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/m8_build_config.hpp.in"
        "${CMAKE_BINARY_DIR}/generated/${target}/m8_build_config.hpp" @ONLY)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE -fno-plt)
    target_include_directories(${target} PRIVATE include "${CMAKE_BINARY_DIR}/generated/${target}")
    target_include_directories(${target} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${nvcomp_INCLUDE_DIR})
    vramz_set_project_warnings(${target})
    vramz_enable_clang_tidy(${target})
endfunction()
