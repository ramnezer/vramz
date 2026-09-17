# Reproducibility

## Evaluated source versus publication source

The frozen evaluated RC1 source archive has SHA256:

```text
e0165afa1fa8e06a9f6ef9fcd99fc6182873a908ecfc4810c44ea4f0fcb3a395
```

This publication has a separate external archive hash. Runtime/API, existing
C++ examples and test implementation bytes are retained; documentation and fixture
locations were reorganized. See [provenance](PROVENANCE.md).

## Reported physical reference

Linux x86-64 / Ubuntu 24.04-family userspace; RTX 3060 device 0;
Driver 595.91.07; physically observed Driver API 13020; CUDA 13.3.1/runtime API
13030; nvCOMP 5.3.0.16; static LZ4 1.9.4; GCC 13 / Clang 18.

These are evaluated identities, not broad support promises. Physical provider
checks pin exact files. Matching a version label alone is insufficient; CUDA,
nvCOMP and Driver binaries are external and are not redistributed here.

## Reproduce software checks

Follow [BUILD.md](../BUILD.md) and [TESTING.md](../TESTING.md). The CPU/Fake tests
and oracle fixtures can be inspected without NVIDIA execution. The user-facing
`examples/gpu-residency/README.md` in the source archive and documented product path
are preferable to recreating old milestone runs.

## Reproduce an experiment

Record the exact source archive hash, compiler/build flags, binary hash, provider
identities, device, data profile, chunk sizes, hard cap, command, exit code and
unaltered result JSON. Preserve failed and nonbeneficial runs rather than removing
them from the record. Keep raw results outside the source tree.

For performance, use identical profile/size/settings for policy and RAW baselines,
retain warmups separately and include synchronization, verification and cleanup.
Do not generalize host-clock workload timings into GPU-kernel or LLM throughput.

## Evidence availability

This source release contains the reported physical/performance summaries and small
CPU/Fake oracle fixtures. It does not bundle the complete private raw physical,
benchmark, installation or sanitizer campaign evidence. A reader can inspect and
rerun the implementation, but cannot reconstruct every historical measurement
from these summaries alone. No new hardware run is implied by publication.
