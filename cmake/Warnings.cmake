function(vramz_set_project_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -fstack-protector-strong
            "$<$<CONFIG:Release>:-D_FORTIFY_SOURCE=3>"
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wconversion
            -Wsign-conversion
            -Wshadow
            -Wformat=2
            -Wundef
            -Wdouble-promotion
            -Wcast-align
            -Wcast-qual
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Woverloaded-virtual
            -Wimplicit-fallthrough)
        target_link_options(${target} PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)
    endif()
endfunction()
