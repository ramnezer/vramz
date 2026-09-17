#!/usr/bin/env bash
# Independently build and test both package variants; never require a previous gate run.
set -euo pipefail

if [[ $# != 1 ]]; then
    printf 'Usage: bash %s NEW_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi
: "${CUDAToolkit_ROOT:?set the isolated CUDA development root}"
: "${VRAMZ_NVCOMP_ROOT:?set the isolated nvCOMP development root}"
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
for tool in cmake ctest make rg python3 readelf cmp sha256sum sed; do
    command -v "$tool" > /dev/null
done
jobs="${VRAMZ_BUILD_JOBS:-2}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'VRAMZ_BUILD_JOBS must be a positive integer\n' >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$1")"
mkdir -- "$1"
output="$(cd -- "$1" && pwd -P)"
trap 'printf "Package gate failed at line %s; retained evidence: %s\n" "$LINENO" "$output" >&2' ERR
printf 'M6 package output: %s\n' "$output"
bash "$source_root/tools/ci/m6-run-gates.sh" release "$output/builds"

for mode in OFF ON; do
    directory="$output/builds/m6-gpp-Release-$mode"
    cmake --install "$directory" --prefix "$directory/install" > "$output/m6-install-$mode.log" 2>&1
    cmake -S "$source_root/tests/consumer" -B "$output/m6-consumer-$mode" -G 'Unix Makefiles' \
        "-DCMAKE_CXX_COMPILER=${VRAMZ_GXX:-g++}" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SKIP_BUILD_RPATH=ON \
        -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF \
        "-DCMAKE_PREFIX_PATH=$directory/install;${VRAMZ_DEPENDENCY_PREFIX:-}" > "$output/m6-consumer-$mode.log" 2>&1
    cmake --build "$output/m6-consumer-$mode" -j "$jobs" >> "$output/m6-consumer-$mode.log" 2>&1
    "$output/m6-consumer-$mode/vramz-public-consumer" >> "$output/m6-consumer-$mode.log" 2>&1
    "$directory/install/bin/vramz-info" > "$output/m6-info-$mode.json"
    python3 "$source_root/tools/vramz-prehardware-check.py" --source "$source_root" --build "$directory" \
        --install "$directory/install" > "$output/m6-prehardware-$mode.json"
    printf 'install/consumer/prehardware %s PASS\n' "$mode"
done

reject() {
    local name="$1" pattern="$2"
    shift 2
    if cmake -S "$source_root" -B "$output/m6-reject-$name" -G 'Unix Makefiles' \
        "-DCMAKE_CXX_COMPILER=${VRAMZ_GXX:-g++}" "-DCMAKE_PREFIX_PATH=${VRAMZ_DEPENDENCY_PREFIX:-}" \
        -DVRAMZ_ENABLE_CUDA=OFF -DVRAMZ_ENABLE_NVCOMP=OFF -DVRAMZ_ALLOW_REAL_GPU_EXECUTION=OFF \
        -DCUDAToolkit_ROOT= -DVRAMZ_NVCOMP_ROOT= -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF \
        -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF "$@" > "$output/m6-reject-$name.log" 2>&1; then
        printf 'ERROR: expected rejection %s\n' "$name" >&2
        return 1
    fi
    rg -q "$pattern" "$output/m6-reject-$name.log"
    printf 'expected rejection %s PASS\n' "$name"
}
reject cuda-root 'requires explicit CUDAToolkit_ROOT' -DVRAMZ_ENABLE_CUDA=ON
reject nvcomp-no-cuda 'requires VRAMZ_ENABLE_CUDA=ON' -DVRAMZ_ENABLE_NVCOMP=ON
reject nvcomp-root 'explicit isolated VRAMZ_NVCOMP_ROOT' -DVRAMZ_ENABLE_CUDA=ON -DVRAMZ_ENABLE_NVCOMP=ON \
    "-DCUDAToolkit_ROOT=$CUDAToolkit_ROOT"
reject nvcomp-missing-package 'Could not find a package configuration file|Could not find a configuration file' \
    -DVRAMZ_ENABLE_CUDA=ON -DVRAMZ_ENABLE_NVCOMP=ON "-DCUDAToolkit_ROOT=$CUDAToolkit_ROOT" \
    "-DVRAMZ_NVCOMP_ROOT=$output/nonexistent-package"
reject nvcomp-wrong-version 'version: 5.2.0.0|not compatible' -DVRAMZ_ENABLE_CUDA=ON -DVRAMZ_ENABLE_NVCOMP=ON \
    "-DCUDAToolkit_ROOT=$CUDAToolkit_ROOT" "-DVRAMZ_NVCOMP_ROOT=$source_root/tools/ci/fixtures/wrong-nvcomp"
reject execution 'real GPU execution is unavailable' -DVRAMZ_ALLOW_REAL_GPU_EXECUTION=ON
reject execution-stub 'cannot use the selected CUDA Driver stub' -DVRAMZ_ALLOW_REAL_GPU_EXECUTION=ON \
    -DVRAMZ_ENABLE_CUDA=ON "-DCUDAToolkit_ROOT=$CUDAToolkit_ROOT"
mkdir -- "$output/empty-dependencies"
reject missing-lz4 'requires system liblz4 development files' \
    "-DCMAKE_FIND_ROOT_PATH=$output/empty-dependencies" \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY

cpu_build="$output/builds/m6-gpp-Release-OFF"
"$cpu_build/vramz-policy-sim" --matrix > "$output/m6-scenarios-first.jsonl"
"$cpu_build/vramz-policy-sim" --matrix > "$output/m6-scenarios-second.jsonl"
cmp "$output/m6-scenarios-first.jsonl" "$output/m6-scenarios-second.jsonl"
cmp "$output/m6-scenarios-first.jsonl" "$source_root/tests/fixtures/oracles/M3_SCENARIO_RESULTS.jsonl"
"$cpu_build/vramz_nvcomp_tests" | sed -n 's/^M5_SCENARIO //p' > "$output/m6-fake-first.jsonl"
"$cpu_build/vramz_nvcomp_tests" | sed -n 's/^M5_SCENARIO //p' > "$output/m6-fake-second.jsonl"
cmp "$output/m6-fake-first.jsonl" "$output/m6-fake-second.jsonl"
cmp "$output/m6-fake-first.jsonl" "$source_root/tests/fixtures/oracles/M5_FAKE_GPU_COMPRESSION_RESULTS.jsonl"
sha256sum "$output/m6-scenarios-first.jsonl" "$output/m6-fake-first.jsonl" > "$output/m6-oracle-hashes.log"
printf 'M3/M5 oracles PASS\n'
