# Tests and reproducibility checks

The source release retains all RC1 test implementation files. There are no
bundled build outputs, coverage dumps, core files, raw host logs or fuzz corpora.
The source file `tests/README.md` describes the layout.

## Full CPU/Fake suite

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DVRAMZ_ENABLE_CUDA=OFF -DVRAMZ_ENABLE_NVCOMP=OFF \
  -DVRAMZ_ENABLE_PHYSICAL_RUNTIME=OFF
cmake --build build-cpu --parallel 2
ctest --test-dir build-cpu --output-on-failure --parallel 2
```

This runs every enabled CTest suite. The LZ4-enabled evaluated RC1 configuration
contains 30 suites / 772 named deterministic cases. The CPU-LZ4-OFF configuration
has a different applicable suite, not an equivalent compression test.

Some capacity fault suites replay many complete byte-storage and child-process
failure scenarios. Their configured watchdogs are intentionally long (up to
four hours); they are not quick startup checks. Watchdog limits are not promises
about how long a run takes. Do not disable a valid test to report a full PASS.

## Focused development checks

List enabled tests with `ctest --test-dir build-cpu -N`. A focused selection is
useful while editing, but must be described as partial validation:

```sh
ctest --test-dir build-cpu --output-on-failure -R '^vramz_(unit|policy_.*|crc32c)_tests$'
```

Python build/loader/report tests are separate from the CTest case count:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s tests/tooling -p '*_test.py'
```

## Compiler and analyzer scope

`CMakePresets.json` retains GCC/Clang Debug/Release and ASan/UBSan/LSan/TSan
configurations. CMake also retains strict warnings, format, tidy and analyzer
support. Do not apply environment-specific sanitizer workarounds silently.
Historical configured scopes and limitations are in [research/QUALITY.md](research/QUALITY.md).
A CPU sanitizer run does not execute the real NVIDIA adapters.

## Reference fixtures

The source directory `tests/fixtures` (with its own README) holds the preserved policy and
Fake-compression oracles. They are simulator expectations, not GPU evidence.
No checksum changed when they moved from documentation into `tests/fixtures`.

The simulator oracle can be reproduced independently:

```sh
./build-cpu/vramz-policy-sim --matrix > build-cpu/policy-results.jsonl
cmp build-cpu/policy-results.jsonl tests/fixtures/oracles/M3_SCENARIO_RESULTS.jsonl
```

The existing `tools/ci/m6-package-gates.sh` also compares both oracles. Its phase
name is retained for compatibility with the regression tooling.

## CPU package/prehardware check

This is a static/CPU-only package check, not a hardware probe. It requires a
complete successful CTest JUnit record and a staged installation:

```sh
ctest --test-dir build-cpu --output-on-failure --parallel 2 \
  --output-junit "$PWD/build-cpu/m6-results.xml"
cmake --install build-cpu --prefix "$PWD/stage-cpu"
python3 -B tools/vramz-prehardware-check.py \
  --source "$PWD" --build "$PWD/build-cpu" --install "$PWD/stage-cpu"
```

The legacy JSON gate names in this checker refer to the CPU-only configuration;
they do not revoke prior physical RC1 observations. Physical product validation
requires the separately opted-in path described in [BUILD.md](BUILD.md).

## What a publication check does not prove

Keeping runtime bytes unchanged and rerunning CPU/Fake tests checks the cleanup
operation. It is not a new GPU benchmark, hardware compatibility certification,
full security audit or re-execution of all prior sanitizer/analyzer matrices.
The external publication validation report records exactly which checks ran.
