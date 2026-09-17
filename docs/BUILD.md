# Build VRAMZ

Run commands from the repository root. Build directories are separate from source.
Do not run compilers, tests or the runtime as root.

## CPU/Fake: the default path

Requirements: Linux x86-64, C++20, CMake >=3.25 and development files for LZ4 >=1.9.4.
Python >=3.11 and Binutils are used by the tooling/provider checks. The evaluated
RC1 toolchains were GCC 13 and Clang 18; that is a tested reference, not a claim
that all other compiler versions fail.

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DVRAMZ_ENABLE_CUDA=OFF -DVRAMZ_ENABLE_NVCOMP=OFF \
  -DVRAMZ_ENABLE_PHYSICAL_RUNTIME=OFF
cmake --build build-cpu --parallel 2
ctest --test-dir build-cpu --output-on-failure --parallel 2
```

CMake discovers system LZ4 for this CPU-only path. A custom development prefix can
be selected with `VRAMZ_LZ4_INCLUDE_DIR` and `VRAMZ_LZ4_LIBRARY`. No LZ4 source or
NVIDIA SDK is downloaded automatically.

Without CPU LZ4, compression support is disabled and only the applicable tests run:

```sh
cmake -S . -B build-no-lz4 -DCMAKE_BUILD_TYPE=Release \
  -DVRAMZ_ENABLE_CPU_LZ4=OFF -DVRAMZ_ENABLE_CUDA=OFF \
  -DVRAMZ_ENABLE_NVCOMP=OFF -DVRAMZ_ENABLE_PHYSICAL_RUNTIME=OFF
cmake --build build-no-lz4 --parallel 2
ctest --test-dir build-no-lz4 --output-on-failure
```

The GCC/Clang Debug/Release and sanitizer presets are in
`CMakePresets.json` in the source archive. See [TESTING.md](TESTING.md) for full scope.

## Physical build: deliberate opt-in

Read [LIMITATIONS.md](LIMITATIONS.md) and [SECURITY_MODEL.md](SECURITY_MODEL.md)
first. A physical build produces a program capable of GPU execution; CPU test
success is not permission to assume an untested hardware/provider combination works.

The evaluated reference uses RTX 3060 device 0, Driver 595.91.07, CUDA 13.3.1,
nvCOMP 5.3.0.16 and the exact reviewed static LZ4 1.9.4 archive/header.
The provider checkers and `PhysicalRuntime.cmake` pin binary identities, not merely
version strings. Do not bypass a mismatch, replace a Driver, or install a Toolkit
stub as a runtime library to get the build through.

Set the following to existing trusted external installations; these example paths
are not created or downloaded by VRAMZ:

```sh
CUDA_ROOT=/opt/cuda-13.3.1
NVCOMP_ROOT=/opt/nvcomp-5.3.0
LZ4_PREFIX=/opt/lz4-1.9.4
DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so.595.91.07
```

The static LZ4 path below assumes the reviewed prefix layout. It must identify the
approved archive, not an arbitrary newly rebuilt copy with the same version.
NVIDIA libraries are not bundled in this repository or installed by its package.

Verify the source archive against its **external** checksum before extracting it.
Set `SOURCE_SHA256` to the hash of the exact publication archive being built, not
the earlier frozen RC1 hash. Keep that source unchanged through the build:

```sh
# Set this to the verified 64-character archive hash from the external checksum.
SOURCE_SHA256=REPLACE_WITH_VERIFIED_PUBLICATION_ARCHIVE_SHA256

cmake -S . -B build-physical -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DBUILD_SHARED_LIBS=OFF \
  -DVRAMZ_BUILD_TESTS=OFF -DVRAMZ_ENABLE_FUZZING=OFF -DVRAMZ_SANITIZER=none \
  -DVRAMZ_ENABLE_CPU_LZ4=ON -DVRAMZ_ENABLE_CUDA=ON -DVRAMZ_ENABLE_NVCOMP=ON \
  -DVRAMZ_ENABLE_PHYSICAL_RUNTIME=ON \
  -DVRAMZ_ALLOW_REAL_GPU_EXECUTION=ON -DVRAMZ_ALLOW_REAL_NVCOMP_EXECUTION=ON \
  -DCMAKE_SKIP_RPATH=ON -DCMAKE_SKIP_BUILD_RPATH=ON \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCUDAToolkit_SENTINEL_FILE="$CUDA_ROOT/version.json" \
  -DCUDAToolkit_NVCC_EXECUTABLE= \
  -DVRAMZ_NVCOMP_ROOT="$NVCOMP_ROOT" \
  -DVRAMZ_REAL_CUDA_DRIVER_LIBRARY="$DRIVER" \
  -DVRAMZ_LZ4_INCLUDE_DIR="$LZ4_PREFIX/include" \
  -DVRAMZ_LZ4_LIBRARY="$LZ4_PREFIX/lib/x86_64-linux-gnu/liblz4.a" \
  -DVRAMZ_V1_SOURCE_SHA256="$SOURCE_SHA256" \
  -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib
cmake --build build-physical --parallel 2
```

All historical `VRAMZ_BUILD_M*_SMOKE` options remain OFF. They are not the public
product interface and must not be enabled alongside the physical runtime.
Building or installing does not automatically run a GPU workload.

Continue with [staging and installation](INSTALL.md). Inspect the resulting ELF
and provider closure before execution. The evaluated physical result belongs to
the original RC1 source; a new publication build has its own identity and has not
inherited a new hardware test simply by compiling successfully.
