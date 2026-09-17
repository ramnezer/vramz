# The default build remains pre-hardware. This module never executes a GPU artifact.
option(VRAMZ_BUILD_M7_RAW_SMOKE "Build the explicit user-run M7 RAW-only host smoke" OFF)
set(VRAMZ_REAL_CUDA_DRIVER_LIBRARY "" CACHE FILEPATH "Explicit host NVIDIA libcuda.so.1; never a Toolkit stub")
set(VRAMZ_M7_SOURCE_SHA256 "" CACHE STRING "SHA256 of the reviewed deterministic M7 source archive")

function(vramz_validate_m7_configuration)
    if(NOT VRAMZ_ALLOW_REAL_GPU_EXECUTION AND NOT VRAMZ_BUILD_M7_RAW_SMOKE)
        return()
    endif()
    if(NOT VRAMZ_BUILD_M7_RAW_SMOKE)
        if(VRAMZ_CUDA_STUB_REAL MATCHES "/stubs/")
            message(FATAL_ERROR "Executable hardware builds cannot use the selected CUDA Driver stub: ${VRAMZ_CUDA_STUB_REAL}; M7 also requires VRAMZ_BUILD_M7_RAW_SMOKE=ON")
        endif()
        message(FATAL_ERROR "real GPU execution is unavailable without VRAMZ_BUILD_M7_RAW_SMOKE=ON")
    endif()
    if(NOT VRAMZ_ALLOW_REAL_GPU_EXECUTION)
        message(FATAL_ERROR "M7 RAW smoke requires explicit VRAMZ_ALLOW_REAL_GPU_EXECUTION=ON")
    endif()
    if(NOT VRAMZ_ENABLE_CUDA)
        message(FATAL_ERROR "M7 RAW smoke requires VRAMZ_ENABLE_CUDA=ON")
    endif()
    if(VRAMZ_ENABLE_NVCOMP)
        message(FATAL_ERROR "M7 RAW smoke requires VRAMZ_ENABLE_NVCOMP=OFF")
    endif()
    string(LENGTH "${VRAMZ_M7_SOURCE_SHA256}" source_hash_length)
    if(NOT source_hash_length EQUAL 64 OR NOT VRAMZ_M7_SOURCE_SHA256 MATCHES "^[0-9a-f]+$" OR
       VRAMZ_M7_SOURCE_SHA256 MATCHES "^0+$")
        message(FATAL_ERROR "M7 requires VRAMZ_M7_SOURCE_SHA256 with the reviewed archive's 64 lowercase hexadecimal SHA256")
    endif()
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        message(FATAL_ERROR "M7 RAW smoke is limited to Linux x86_64")
    endif()
    if(NOT IS_ABSOLUTE "${VRAMZ_REAL_CUDA_DRIVER_LIBRARY}" OR
       NOT EXISTS "${VRAMZ_REAL_CUDA_DRIVER_LIBRARY}" OR
       IS_DIRECTORY "${VRAMZ_REAL_CUDA_DRIVER_LIBRARY}")
        message(FATAL_ERROR "M7 requires explicit existing absolute VRAMZ_REAL_CUDA_DRIVER_LIBRARY")
    endif()
    find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
    find_program(VRAMZ_READELF_EXECUTABLE readelf REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/ci/m7-elf-check.py"
            --driver "${VRAMZ_REAL_CUDA_DRIVER_LIBRARY}" --toolkit-root "${CUDAToolkit_ROOT}"
        RESULT_VARIABLE result OUTPUT_VARIABLE evidence ERROR_VARIABLE diagnostic
        TIMEOUT 30)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "M7 real driver ELF validation failed: ${evidence}${diagnostic}")
    endif()
    file(REAL_PATH "${VRAMZ_REAL_CUDA_DRIVER_LIBRARY}" driver_real)
    string(JSON driver_sha256 GET "${evidence}" driver_sha256)
    add_library(VRAMZ_M7_REAL_CUDA_DRIVER SHARED IMPORTED GLOBAL)
    set_target_properties(VRAMZ_M7_REAL_CUDA_DRIVER PROPERTIES
        IMPORTED_LOCATION "${driver_real}" IMPORTED_SONAME "libcuda.so.1")
    set(VRAMZ_M7_DRIVER_REAL "${driver_real}" PARENT_SCOPE)
    set(VRAMZ_M7_DRIVER_SHA256 "${driver_sha256}" PARENT_SCOPE)
    foreach(field device inode size mtime_ns ctime_ns)
        string(JSON identity GET "${evidence}" "driver_${field}")
        set(VRAMZ_M7_DRIVER_${field} "${identity}" PARENT_SCOPE)
    endforeach()
    set(VRAMZ_M7_PYTHON "${Python3_EXECUTABLE}" PARENT_SCOPE)
    file(WRITE "${CMAKE_BINARY_DIR}/m7-driver-elf.json" "${evidence}")
    message(STATUS "M7 RAW host driver validated statically: ${driver_real}; SHA256=${driver_sha256}")
endfunction()

function(vramz_add_m7_raw_smoke target)
    if(NOT VRAMZ_BUILD_M7_RAW_SMOKE)
        return()
    endif()
    if(NOT TARGET VRAMZ_M7_REAL_CUDA_DRIVER)
        message(FATAL_ERROR "M7 target creation requires successful driver configuration validation")
    endif()
    add_executable(${target} ${ARGN} src/backends/cuda_vmm/cuda_driver.cpp)
    vramz_m7_configure_identity(${target} "${VRAMZ_M7_DRIVER_REAL}" "${VRAMZ_M7_DRIVER_SHA256}")
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_definitions(${target} PRIVATE VRAMZ_M7_RAW_ONLY=1)
    # dladdr(&cuInit) must observe the resolved driver symbol, never an executable PLT stub.
    target_compile_options(${target} PRIVATE -fno-plt)
    target_include_directories(${target} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE VRAMZ::vramz VRAMZ_M7_REAL_CUDA_DRIVER ${CMAKE_DL_LIBS})
    target_link_options(${target} PRIVATE -Wl,--no-undefined)
    set_target_properties(${target} PROPERTIES
        SKIP_BUILD_RPATH TRUE BUILD_RPATH "" INSTALL_RPATH "")
    vramz_set_project_warnings(${target})
    vramz_enable_clang_tidy(${target})
    # Static inspection is a build gate, not execution of the smoke binary. In particular,
    # the smoke is never a CTest, POST_BUILD execution, install hook or package hook.
    add_custom_target(${target}-elf-check ALL
        COMMAND "${VRAMZ_M7_PYTHON}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/ci/m7-elf-check.py"
            --driver "${VRAMZ_M7_DRIVER_REAL}" --toolkit-root "${CUDAToolkit_ROOT}"
            --binary "$<TARGET_FILE:${target}>"
        DEPENDS ${target} VERBATIM)
endfunction()

function(vramz_m7_configure_identity target driver_path driver_sha256)
    # Escape C++ string-literal syntax without embedding any Toolkit path in the artifact.
    string(REPLACE "\\" "\\\\" VRAMZ_M7_DRIVER_PATH_ESCAPED "${driver_path}")
    string(REPLACE "\"" "\\\"" VRAMZ_M7_DRIVER_PATH_ESCAPED "${VRAMZ_M7_DRIVER_PATH_ESCAPED}")
    if(VRAMZ_M7_DRIVER_PATH_ESCAPED MATCHES "[\r\n]")
        message(FATAL_ERROR "M7 driver path must not contain line breaks")
    endif()
    set(VRAMZ_M7_DRIVER_SHA256_VALUE "${driver_sha256}")
    if(driver_path STREQUAL "")
        set(VRAMZ_M7_SOURCE_SHA256 "0000000000000000000000000000000000000000000000000000000000000000")
        foreach(field device inode size mtime_ns ctime_ns)
            set(VRAMZ_M7_DRIVER_${field} 0)
        endforeach()
    endif()
    set(identity_directory "${CMAKE_BINARY_DIR}/generated/${target}")
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/m7_build_config.hpp.in"
        "${identity_directory}/m7_build_config.hpp" @ONLY)
    target_include_directories(${target} PRIVATE "${identity_directory}")
endfunction()

function(vramz_add_m7_compile_validation target)
    if(NOT VRAMZ_ENABLE_CUDA OR VRAMZ_BUILD_M7_RAW_SMOKE)
        return()
    endif()
    # Compile main in the VM, but produce neither an executable nor a loadable adapter.
    # Empty approved identity and the absent RAW_ONLY marker also forbid runtime approval.
    add_library(${target} OBJECT ${ARGN})
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE -fno-plt)
    target_include_directories(${target} PRIVATE include)
    target_include_directories(${target} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    vramz_m7_configure_identity(${target} "" "")
    vramz_set_project_warnings(${target})
    vramz_enable_clang_tidy(${target})
endfunction()
