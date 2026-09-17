#!/usr/bin/env bash
# Source-controlled M6 gates. Real adapters are compiled/linked, never executed.
set -euo pipefail

if [[ $# != 2 ]]; then
    printf 'Usage: bash %s {debug|release|matrix|analysis|sanitizers|fuzz|no-lz4} NEW_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi
group="$1"
case "$group" in
    debug|release|matrix|analysis|sanitizers|fuzz|no-lz4) ;;
    *) printf 'Unknown gate group: %s\n' "$group" >&2; exit 2 ;;
esac
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
for tool in cmake ctest make rg; do
    command -v "$tool" > /dev/null
done
jobs="${VRAMZ_BUILD_JOBS:-2}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'VRAMZ_BUILD_JOBS must be a positive integer\n' >&2
    exit 2
fi
gcc="${VRAMZ_GXX:-g++}"
clangxx="${VRAMZ_CLANGXX:-clang++}"
sdk=()
case "$group" in
    release|matrix|analysis)
        : "${CUDAToolkit_ROOT:?set the isolated CUDA development root}"
        : "${VRAMZ_NVCOMP_ROOT:?set the isolated nvCOMP development root}"
        sdk=(-DVRAMZ_ENABLE_CUDA=ON -DVRAMZ_ENABLE_NVCOMP=ON
             "-DCUDAToolkit_ROOT=$CUDAToolkit_ROOT" "-DVRAMZ_NVCOMP_ROOT=$VRAMZ_NVCOMP_ROOT")
        ;;
esac

# Refuse old output, including old caches. Never remove or reuse user build state.
mkdir -p -- "$(dirname -- "$2")"
mkdir -- "$2"
output="$(cd -- "$2" && pwd -P)"
trap 'printf "Gate failed at line %s; retained evidence: %s\n" "$LINENO" "$output" >&2' ERR
printf 'M6 source: %s\nM6 output: %s\n' "$source_root" "$output"
# Isolated development prefixes must not become operational ELF loader paths.
# Compile-only SDK adapters are never executed; runtime dependencies use the caller's
# explicit validation environment or the host's ordinary loader configuration.
common=(-G 'Unix Makefiles' "-DCMAKE_PREFIX_PATH=${VRAMZ_DEPENDENCY_PREFIX:-}"
        -DCMAKE_SKIP_BUILD_RPATH=ON
        -DVRAMZ_ALLOW_REAL_GPU_EXECUTION=OFF -DVRAMZ_ENABLE_CUDA=OFF -DVRAMZ_ENABLE_NVCOMP=OFF
        -DCUDAToolkit_ROOT= -DVRAMZ_NVCOMP_ROOT= -DVRAMZ_ENABLE_CPU_LZ4=ON
        -DVRAMZ_BUILD_TESTS=ON -DVRAMZ_ENABLE_FUZZING=OFF -DVRAMZ_ENABLE_CLANG_TIDY=OFF
        -DVRAMZ_SANITIZER=none -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
        -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF)

check_warnings() {
    if rg -n 'warning:' "$output/$1-configure.log" "$output/$1-build.log"; then
        printf 'Unexpected compiler warning in %s\n' "$1" >&2
        return 1
    fi
}
run_build() {
    local name="$1"
    shift
    cmake -S "$source_root" -B "$output/$name" "${common[@]}" "$@" > "$output/$name-configure.log" 2>&1
    cmake --build "$output/$name" -j "$jobs" > "$output/$name-build.log" 2>&1
    check_warnings "$name"
}
run_tests() {
    ctest --test-dir "$output/$1" -V --no-tests=error --output-junit m6-results.xml > "$output/$1-test.log" 2>&1
    printf '%s PASS\n' "$1"
}

case "$group" in
debug)
    run_build m6-gpp-Debug-OFF "-DCMAKE_CXX_COMPILER=$gcc" -DCMAKE_BUILD_TYPE=Debug
    run_tests m6-gpp-Debug-OFF
    ;;
release|matrix)
    compilers=(gcc)
    types=(Release)
    if [[ "$group" == matrix ]]; then compilers=(gcc clang); types=(Debug Release); fi
    for compiler in "${compilers[@]}"; do
        compiler_path="$gcc"
        label=gpp
        if [[ "$compiler" == clang ]]; then compiler_path="$clangxx"; label=clangpp; fi
        for type in "${types[@]}"; do
            for mode in OFF ON; do
                name="m6-$label-$type-$mode"
                extra=()
                if [[ "$mode" == ON ]]; then extra=("${sdk[@]}"); fi
                run_build "$name" "-DCMAKE_CXX_COMPILER=$compiler_path" "-DCMAKE_BUILD_TYPE=$type" "${extra[@]}"
                run_tests "$name"
            done
        done
    done
    ;;
sanitizers)
    export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
    export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
    export LSAN_OPTIONS=exitcode=23
    export TSAN_OPTIONS=halt_on_error=1
    for sanitizer in address undefined leak thread; do
        name="m6-$sanitizer"
        run_build "$name" "-DCMAKE_CXX_COMPILER=$gcc" -DCMAKE_BUILD_TYPE=Debug "-DVRAMZ_SANITIZER=$sanitizer"
        run_tests "$name"
    done
    ;;
analysis)
    analyzer="$(command -v "${VRAMZ_CLANG:-clang}")"
    for mode in ON OFF; do
        extra=(-DVRAMZ_ENABLE_CPU_LZ4=OFF)
        if [[ "$mode" == ON ]]; then extra=("${sdk[@]}"); fi
        name="m6-tidy-$mode"
        run_build "$name" "-DCMAKE_CXX_COMPILER=$clangxx" -DCMAKE_BUILD_TYPE=Debug -DVRAMZ_ENABLE_CLANG_TIDY=ON "${extra[@]}"
        run_tests "$name"
        cmake --build "$output/$name" --target vramz-format-check vramz-cppcheck > "$output/$name-quality.log" 2>&1
        printf '%s format/cppcheck PASS\n' "$name"
        name="m6-fanalyzer-$mode"
        run_build "$name" "-DCMAKE_CXX_COMPILER=$gcc" -DCMAKE_BUILD_TYPE=Debug -DVRAMZ_BUILD_TESTS=OFF -DCMAKE_CXX_FLAGS=-fanalyzer "${extra[@]}"
        printf '%s PASS\n' "$name"
        name="m6-scan-$mode"
        scan-build --use-analyzer="$analyzer" --status-bugs -o "$output/$name-reports" \
            cmake -S "$source_root" -B "$output/$name" "${common[@]}" -DCMAKE_BUILD_TYPE=Debug "${extra[@]}" > "$output/$name-configure.log" 2>&1
        scan-build --use-analyzer="$analyzer" --status-bugs -o "$output/$name-reports" \
            cmake --build "$output/$name" -j "$jobs" > "$output/$name-build.log" 2>&1
        check_warnings "$name"
        run_tests "$name"
    done
    ;;
fuzz)
    export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
    export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
    run_build m6-fuzz "-DCMAKE_CXX_COMPILER=$clangxx" -DCMAKE_BUILD_TYPE=Debug -DVRAMZ_BUILD_TESTS=OFF -DVRAMZ_ENABLE_FUZZING=ON
    "$output/m6-fuzz/vramz_codec_fuzzer" -runs=20000 -seed=12648430 -max_len=1048576 -rss_limit_mb=512 -timeout=5 > "$output/m6-fuzz-run.log" 2>&1
    printf 'm6-fuzz PASS\n'
    ;;
no-lz4)
    run_build m6-no-lz4 "-DCMAKE_CXX_COMPILER=$gcc" -DCMAKE_BUILD_TYPE=Debug -DVRAMZ_ENABLE_CPU_LZ4=OFF
    run_tests m6-no-lz4
    ;;
esac
