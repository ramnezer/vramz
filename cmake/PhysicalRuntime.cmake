option(VRAMZ_ENABLE_PHYSICAL_RUNTIME "Build opt-in public CUDA runtime and tools" OFF)

function(vramz_add_physical_runtime)
    if(NOT VRAMZ_ENABLE_PHYSICAL_RUNTIME)
        if(VRAMZ_ENABLE_CUDA AND VRAMZ_ENABLE_NVCOMP)
            # Compile public physical entry points for SDK quality; never link or execute them.
            add_library(vramz_public_cuda_compile_check OBJECT
                src/backends/cuda_vmm/physical_runtime.cpp tools/vramz-run/main.cpp)
            # This non-runnable OBJECT fixture permits analysis of the complete CLI.
            # Its syntactically valid identity is never linked or installed.
            target_compile_definitions(vramz_public_cuda_compile_check PRIVATE
                VRAMZ_V1_SOURCE_SHA256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            target_link_libraries(vramz_public_cuda_compile_check PRIVATE VRAMZ::vramz)
            target_include_directories(vramz_public_cuda_compile_check SYSTEM PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS} "${VRAMZ_NVCOMP_ROOT}/include")
            vramz_set_project_warnings(vramz_public_cuda_compile_check)
            vramz_enable_sanitizer(vramz_public_cuda_compile_check)
            vramz_enable_clang_tidy(vramz_public_cuda_compile_check)
        endif()
        return()
    endif()
    if(NOT VRAMZ_ENABLE_CUDA OR NOT VRAMZ_ENABLE_NVCOMP OR NOT VRAMZ_ENABLE_CPU_LZ4 OR
       NOT VRAMZ_ALLOW_REAL_GPU_EXECUTION OR NOT VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION)
        message(FATAL_ERROR "Public physical runtime requires CUDA, nvCOMP, LZ4 and both execution build gates")
    endif()
    foreach(gate M7_RAW M8_COMPRESSION M9_PHYSICAL_SAVINGS M10_MULTI_CHUNK_RESIDENCY
                 M11_POLICY_PRESSURE M12_CONTROLLED_CAPACITY M13_CONTROLLED_CAPACITY_2X)
        if(VRAMZ_BUILD_${gate}_SMOKE)
            message(FATAL_ERROR "Public physical runtime build must not enable historical smoke targets")
        endif()
    endforeach()
    if(NOT VRAMZ_LZ4_LIBRARY MATCHES "liblz4\\.a$")
        message(FATAL_ERROR "Physical product requires the reviewed static LZ4 archive")
    endif()
    file(SHA256 "${VRAMZ_LZ4_LIBRARY}" lz4_archive_sha)
    file(SHA256 "${VRAMZ_LZ4_INCLUDE_DIR}/lz4.h" lz4_header_sha)
    if(NOT lz4_archive_sha STREQUAL "c598b9b376be47052e5c2dd93a6bfbd34329d48b384c05eab9618edb3caf93fb" OR
       NOT lz4_header_sha STREQUAL "c1614ecf7ada7b0be1acb560d4239595f96fbb7aa6a79a7c40cb358753830be6")
        message(FATAL_ERROR "Physical LZ4 archive/header differs from reviewed providers")
    endif()
    find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/ci/m13-elf-check.py"
        --driver "${VRAMZ_REAL_CUDA_DRIVER_LIBRARY}" --cuda-root "${CUDAToolkit_ROOT}"
        --nvcomp-root "${VRAMZ_NVCOMP_ROOT}"
        RESULT_VARIABLE status OUTPUT_VARIABLE evidence ERROR_VARIABLE diagnostic TIMEOUT 30)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "Physical provider verification failed: ${evidence}${diagnostic}")
    endif()
    file(WRITE "${CMAKE_BINARY_DIR}/v1-provider-preflight.json" "${evidence}")
    foreach(pair DRIVER CUDART NVCOMP)
        add_library(VRAMZ_V1_${pair} SHARED IMPORTED GLOBAL)
    endforeach()
    set_target_properties(VRAMZ_V1_DRIVER PROPERTIES IMPORTED_LOCATION "${VRAMZ_REAL_CUDA_DRIVER_LIBRARY}")
    set_target_properties(VRAMZ_V1_CUDART PROPERTIES IMPORTED_LOCATION "${VRAMZ_CUDART_LIBRARY_REAL}")
    set_target_properties(VRAMZ_V1_NVCOMP PROPERTIES IMPORTED_LOCATION "${VRAMZ_NVCOMP_LIBRARY_REAL}")
    add_library(vramz_cuda STATIC src/backends/cuda_vmm/physical_runtime.cpp
        src/backends/cuda_vmm/cuda_driver.cpp src/backends/nvcomp/nvcomp_lz4_api.cpp)
    add_library(VRAMZ::cuda ALIAS vramz_cuda)
    set_target_properties(vramz_cuda PROPERTIES EXPORT_NAME cuda POSITION_INDEPENDENT_CODE ON)
    target_compile_features(vramz_cuda PUBLIC cxx_std_20)
    target_include_directories(vramz_cuda SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS} "${VRAMZ_NVCOMP_ROOT}/include")
    target_compile_options(vramz_cuda PRIVATE -fno-plt
        "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=." "-ffile-prefix-map=${CMAKE_BINARY_DIR}=build")
    target_link_libraries(vramz_cuda PUBLIC VRAMZ::vramz PRIVATE
        VRAMZ_V1_DRIVER VRAMZ_V1_NVCOMP VRAMZ_V1_CUDART ${CMAKE_DL_LIBS})
    vramz_set_project_warnings(vramz_cuda)
    vramz_enable_sanitizer(vramz_cuda)
    vramz_enable_clang_tidy(vramz_cuda)
    set(VRAMZ_V1_SOURCE_SHA256 "unfrozen-development" CACHE STRING "Frozen source identity for physical evidence")
    add_executable(vramz-run tools/vramz-run/main.cpp)
    target_link_libraries(vramz-run PRIVATE VRAMZ::cuda vramz_workload)
    target_compile_definitions(vramz-run PRIVATE VRAMZ_V1_SOURCE_SHA256="${VRAMZ_V1_SOURCE_SHA256}")
    target_link_options(vramz-run PRIVATE -Wl,--no-undefined)
    set_target_properties(vramz-run PROPERTIES SKIP_BUILD_RPATH TRUE BUILD_RPATH "" INSTALL_RPATH "")
    vramz_set_project_warnings(vramz-run)
    vramz_enable_sanitizer(vramz-run)
    vramz_enable_clang_tidy(vramz-run)
    install(TARGETS vramz-run RUNTIME DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/vramz")
    install(PROGRAMS tools/vramz/benchmark_report.py DESTINATION "${CMAKE_INSTALL_BINDIR}" RENAME vramz-benchmark-report)
    install(PROGRAMS tools/vramz/launcher.py DESTINATION "${CMAKE_INSTALL_BINDIR}" RENAME vramz)
    install(FILES tools/ci/m13-elf-check.py tools/ci/m7-elf-check.py DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/vramz")
    install(TARGETS vramz_cuda EXPORT VRAMZTargets ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    install(FILES include/vramz/cuda_runtime.hpp DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/vramz")
    install(FILES tools/common/workload.cpp tools/common/workload.hpp examples/gpu-residency/main.cpp
        examples/gpu-residency/CMakeLists.txt examples/gpu-residency/README.md
        DESTINATION "${CMAKE_INSTALL_DATADIR}/vramz/examples/gpu-residency")
endfunction()
