option(VRAMZ_ENABLE_CLANG_TIDY "Run clang-tidy while compiling VRAMZ" OFF)

function(vramz_enable_clang_tidy target)
    if(NOT VRAMZ_ENABLE_CLANG_TIDY)
        return()
    endif()
    find_program(VRAMZ_CLANG_TIDY_EXECUTABLE clang-tidy REQUIRED)
    set_property(TARGET ${target} PROPERTY CXX_CLANG_TIDY
                 "${VRAMZ_CLANG_TIDY_EXECUTABLE};--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy")
endfunction()

function(vramz_add_quality_targets)
    find_program(VRAMZ_CLANG_FORMAT_EXECUTABLE clang-format)
    if(VRAMZ_CLANG_FORMAT_EXECUTABLE)
        add_custom_target(vramz-format-check
            COMMAND ${VRAMZ_CLANG_FORMAT_EXECUTABLE} --dry-run --Werror ${ARGN}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            VERBATIM)
    endif()

    find_program(VRAMZ_CPPCHECK_EXECUTABLE cppcheck)
    if(VRAMZ_CPPCHECK_EXECUTABLE)
        set(VRAMZ_ANALYSIS_SOURCES ${ARGN})
        set(VRAMZ_ANALYSIS_INCLUDES)
        set(VRAMZ_ANALYSIS_EXTERNAL_EXCEPTIONS)
        if(VRAMZ_ENABLE_CUDA)
            if(VRAMZ_BUILD_M7_RAW_SMOKE)
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz-m7-raw-smoke")
            else()
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz_m7_main_compile_check")
            endif()
            foreach(directory IN LISTS CUDAToolkit_INCLUDE_DIRS)
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${directory}")
                if(CUDAToolkit_VERSION VERSION_EQUAL 13.3.1)
                    # Reviewed NVIDIA 13.3.1 cuda.h enum CU_COREDUMP_LOG_ONLY = (1 << 31).
                    # Cppcheck 2.13 flags its signed-shift portability. This unused vendor
                    # declaration is outside owned-source policy; retain analysis everywhere
                    # else and never edit the SDK. Exact rule/file/line, not a broad exclusion.
                    list(APPEND VRAMZ_ANALYSIS_EXTERNAL_EXCEPTIONS
                        "--suppress=shiftTooManyBitsSigned:${directory}/cuda.h:25952")
                endif()
            endforeach()
        else()
            # The only NVIDIA-header consumer is opt-in; all VMM/core/fake logic
            # remains analyzed with CUDA OFF and without any Toolkit dependency.
            list(FILTER VRAMZ_ANALYSIS_SOURCES EXCLUDE REGEX "cuda_driver\\.cpp$")
            list(FILTER VRAMZ_ANALYSIS_SOURCES EXCLUDE REGEX "physical_runtime\\.cpp$")
            list(FILTER VRAMZ_ANALYSIS_SOURCES EXCLUDE REGEX "tools/vramz-m7-raw-smoke/main\\.cpp$")
        endif()
        if(VRAMZ_ENABLE_NVCOMP)
            if(VRAMZ_BUILD_M8_COMPRESSION_SMOKE)
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz-m8-compression-smoke")
            else()
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz_m8_main_compile_check")
            endif()
            if(VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE)
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz-m9-physical-savings-smoke")
            else()
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz_m9_main_compile_check")
            endif()
            if(VRAMZ_BUILD_M10_MULTI_CHUNK_RESIDENCY_SMOKE)
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz-m10-multi-chunk-residency-smoke")
            else()
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz_m10_main_compile_check")
            endif()
            if(VRAMZ_BUILD_M11_POLICY_PRESSURE_SMOKE)
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz-m11-policy-pressure-smoke")
            else()
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz_m11_main_compile_check")
            endif()
            if(VRAMZ_BUILD_M12_CONTROLLED_CAPACITY_SMOKE)
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz-m12-controlled-capacity-smoke")
            else()
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz_m12_main_compile_check")
            endif()
            if(VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE)
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz-m13-controlled-capacity-2x-smoke")
            else()
                list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${CMAKE_BINARY_DIR}/generated/vramz_m13_main_compile_check")
            endif()
            list(APPEND VRAMZ_ANALYSIS_INCLUDES -I "${nvcomp_INCLUDE_DIR}")
            # M5 is host C++ only, never NVCC/device compilation. Model the actual
            # preprocessor contract instead of impossible vendor device branches.
            list(APPEND VRAMZ_ANALYSIS_INCLUDES -U__CUDACC__ -U__CUDACC_RTC__ -U__CUDA_ARCH__)
        else()
            list(FILTER VRAMZ_ANALYSIS_SOURCES EXCLUDE REGEX "nvcomp_lz4_api\\.cpp$")
            list(FILTER VRAMZ_ANALYSIS_SOURCES EXCLUDE REGEX "tools/vramz-m8-compression-smoke/main\\.cpp$")
            list(FILTER VRAMZ_ANALYSIS_SOURCES EXCLUDE REGEX "tools/vramz-m9-physical-savings-smoke/main\\.cpp$")
            list(FILTER VRAMZ_ANALYSIS_SOURCES EXCLUDE REGEX "tools/vramz-m10-multi-chunk-residency-smoke/main\\.cpp$")
            list(FILTER VRAMZ_ANALYSIS_SOURCES EXCLUDE REGEX "tools/vramz-m11-policy-pressure-smoke/main\\.cpp$")
            list(FILTER VRAMZ_ANALYSIS_SOURCES EXCLUDE REGEX "tools/vramz-m12-controlled-capacity-smoke/main\\.cpp$")
            list(FILTER VRAMZ_ANALYSIS_SOURCES EXCLUDE REGEX "tools/vramz-m13-controlled-capacity-2x-smoke/main\\.cpp$")
        endif()
        add_custom_target(vramz-cppcheck
            COMMAND ${VRAMZ_CPPCHECK_EXECUTABLE}
                    --enable=warning,performance,portability
                    --error-exitcode=2
                    --std=c++20
                    -I ${CMAKE_SOURCE_DIR}/include
                    ${VRAMZ_ANALYSIS_INCLUDES}
                    ${VRAMZ_ANALYSIS_EXTERNAL_EXCEPTIONS}
                    ${VRAMZ_ANALYSIS_SOURCES}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            VERBATIM)
    endif()
endfunction()
