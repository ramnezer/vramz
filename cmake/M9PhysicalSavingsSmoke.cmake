# M9A compiles main as an OBJECT only; a future reviewed M9B build is separate.
set(VRAMZ_M9_SOURCE_SHA256 "" CACHE STRING "Independent reviewed M9 source archive SHA256")

function(vramz_add_m9_smoke)
    if(VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE)
        if(NOT VRAMZ_ALLOW_REAL_GPU_EXECUTION OR NOT VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION OR
           NOT VRAMZ_ENABLE_CUDA OR NOT VRAMZ_ENABLE_NVCOMP OR NOT VRAMZ_ENABLE_CPU_LZ4 OR
           VRAMZ_BUILD_TESTS OR VRAMZ_ENABLE_FUZZING OR NOT VRAMZ_SANITIZER STREQUAL "none")
            message(FATAL_ERROR "M9 requires CUDA/nvCOMP/CPU-LZ4 and both execution gates ON; tests/fuzzing/sanitizers OFF")
        endif()
        string(LENGTH "${VRAMZ_M9_SOURCE_SHA256}" length)
        if(NOT length EQUAL 64 OR NOT VRAMZ_M9_SOURCE_SHA256 MATCHES "^[0-9a-f]+$" OR
           VRAMZ_M9_SOURCE_SHA256 MATCHES "^0+$" OR
           VRAMZ_M9_SOURCE_SHA256 STREQUAL "c67dc98206470c74045aa0c7ed1ae69ae596b2f7687a966cab6c6b7e3dec93bd" OR
           VRAMZ_M9_SOURCE_SHA256 STREQUAL "56b3a392464febf9b5ec8f49653b68a2de5bd50c2bd814a1d38ac7b133334c76" OR
           VRAMZ_M9_SOURCE_SHA256 STREQUAL "ed54613a38f4532f4f12b80020bcec191dc11045225541995c0cc2132e2d4b94")
            message(FATAL_ERROR "M9 requires its own reviewed source archive SHA256, never an M7/M8 identity")
        endif()
        # The future physical artifact retains the exact reviewed static CPU oracle.
        if(NOT VRAMZ_REVIEWED_LZ4_PREFIX)
            message(FATAL_ERROR "Historical physical smoke requires explicit VRAMZ_REVIEWED_LZ4_PREFIX for the reviewed static LZ4 1.9.4 tree")
        endif()
        set(lz4_prefix "${VRAMZ_REVIEWED_LZ4_PREFIX}")
        if(NOT VRAMZ_LZ4_LIBRARY STREQUAL "${lz4_prefix}/lib/x86_64-linux-gnu/liblz4.a" OR
           NOT VRAMZ_LZ4_INCLUDE_DIR STREQUAL "${lz4_prefix}/include" OR
           NOT VRAMZ_LZ4_VERSION VERSION_EQUAL 1.9.4)
            message(FATAL_ERROR "M9 requires the reviewed isolated static LZ4 1.9.4 prefix")
        endif()
        file(SHA256 "${VRAMZ_LZ4_LIBRARY}" lz4_hash)
        file(SHA256 "${VRAMZ_LZ4_INCLUDE_DIR}/lz4.h" lz4_header_hash)
        if(NOT lz4_hash STREQUAL "c598b9b376be47052e5c2dd93a6bfbd34329d48b384c05eab9618edb3caf93fb" OR
           NOT lz4_header_hash STREQUAL "c1614ecf7ada7b0be1acb560d4239595f96fbb7aa6a79a7c40cb358753830be6")
            message(FATAL_ERROR "M9 reviewed LZ4 identity mismatch")
        endif()
        find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
        execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/ci/m8-elf-check.py"
            --driver "${VRAMZ_REAL_CUDA_DRIVER_LIBRARY}" --cuda-root "${CUDAToolkit_ROOT}"
            --nvcomp-root "${VRAMZ_NVCOMP_ROOT}" --identity-header "${CMAKE_BINARY_DIR}/m9-identities.cmake"
            RESULT_VARIABLE status OUTPUT_VARIABLE evidence ERROR_VARIABLE diagnostic TIMEOUT 30)
        if(NOT status EQUAL 0)
            message(FATAL_ERROR "M9 static identity/closure failure: ${evidence}${diagnostic}")
        endif()
        include("${CMAKE_BINARY_DIR}/m9-identities.cmake")
        foreach(field DRIVER_PATH DRIVER_IDENTITY NVCOMP_IDENTITY CUDART_IDENTITY LOADER_PATH DRIVER_RELEASE PRIOR_DRIVER_API)
            set(VRAMZ_M9_${field} "${VRAMZ_M8_${field}}")
        endforeach()
        file(WRITE "${CMAKE_BINARY_DIR}/m9-sdk-elf.json" "${evidence}")
        set(target vramz-m9-physical-savings-smoke)
        add_library(VRAMZ_M9_REAL_DRIVER SHARED IMPORTED)
        set_target_properties(VRAMZ_M9_REAL_DRIVER PROPERTIES IMPORTED_LOCATION "${VRAMZ_M9_DRIVER_PATH}")
        add_executable(${target} tools/vramz-m9-physical-savings-smoke/main.cpp
            src/backends/cuda_vmm/cuda_driver.cpp src/backends/nvcomp/nvcomp_lz4_api.cpp)
        target_compile_definitions(${target} PRIVATE VRAMZ_M9_PHYSICAL_SAVINGS_ONLY=1)
        target_link_libraries(${target} PRIVATE VRAMZ::vramz VRAMZ_M9_REAL_DRIVER nvcomp::nvcomp CUDA::cudart ${CMAKE_DL_LIBS})
        target_link_options(${target} PRIVATE -Wl,--no-undefined)
        set_target_properties(${target} PROPERTIES SKIP_BUILD_RPATH TRUE BUILD_RPATH "" INSTALL_RPATH "")
        add_custom_target(${target}-elf-check
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/ci/m8-elf-check.py"
                --driver "${VRAMZ_REAL_CUDA_DRIVER_LIBRARY}" --cuda-root "${CUDAToolkit_ROOT}"
                --nvcomp-root "${VRAMZ_NVCOMP_ROOT}" --binary "$<TARGET_FILE:${target}>"
            DEPENDS ${target} VERBATIM)
    elseif(VRAMZ_ENABLE_CUDA AND VRAMZ_ENABLE_NVCOMP)
        set(target vramz_m9_main_compile_check)
        add_library(${target} OBJECT tools/vramz-m9-physical-savings-smoke/main.cpp)
        set(VRAMZ_M9_SOURCE_SHA256 "")
        set(VRAMZ_M9_LOADER_PATH "")
        set(VRAMZ_M9_DRIVER_RELEASE "0U, 0U, 0U")
        set(VRAMZ_M9_PRIOR_DRIVER_API 0)
        foreach(kind DRIVER NVCOMP CUDART)
            set(VRAMZ_M9_${kind}_IDENTITY "\"\", \"\", {}")
        endforeach()
    else()
        return()
    endif()
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/m9_build_config.hpp.in"
        "${CMAKE_BINARY_DIR}/generated/${target}/m9_build_config.hpp" @ONLY)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE -fno-plt)
    target_include_directories(${target} PRIVATE include "${CMAKE_BINARY_DIR}/generated/${target}")
    target_include_directories(${target} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${nvcomp_INCLUDE_DIR})
    vramz_set_project_warnings(${target})
    vramz_enable_clang_tidy(${target})
endfunction()
