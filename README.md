# VRAMZ

### GPU-resident compressed memory · Research preview

**VRAMZ is an experimental C++20 userspace runtime for NVIDIA GPUs.** It explores
keeping colder application data compressed in GPU memory, restoring it on demand,
and reducing the physical CUDA VMM backing needed at a settled residency snapshot.

**Status:** frozen experimental research preview, not production-ready.

> **Scope:** applications must explicitly use the VRAMZ API. Installing VRAMZ does
> not increase the physical memory reported by NVIDIA tools, transparently expand
> arbitrary CUDA applications, or make a 12 GiB GPU a general-purpose 24 GiB GPU.
>
> **Performance:** the evaluated verified workflows incurred **209–707% overhead**
> versus matching RAW-only workflows. The 15% target was not achieved.

## Research result

On the evaluated RTX 3060 reference platform, the automatic policy retained eight
logical buffers totalling **64 MiB** with **28 MiB of settled physical VMM backing**:
approximately **2.286× logical/physical capacity at that snapshot**. All buffers
were subsequently verified by size, CRC and exact byte comparison.

| Reference profile | Logical data | Settled VMM backing | Logical / physical |
|---|---:|---:|---:|
| Highly compressible | 64 MiB | 28 MiB | 2.286× |
| Mixed high/moderate/random | 64 MiB | 32 MiB | 2.000× |
| Effectively incompressible | 64 MiB | 64 MiB | 1.000× |

These are workload-specific measurements, not guaranteed capacity multipliers.
The **28 MiB settled value is not the peak memory requirement**: the corresponding
64 MiB test reports an 86 MiB peak explicit allocation/reservation charge.
Driver/runtime internal allocations and other applications are outside that accounting.
No HOST-tier backing contributed to the reported GPU capacity; host-side setup,
integrity copies and CPU work still occur.

Read the [capacity results](docs/research/CAPACITY_RESULTS.md),
[physical validation](docs/research/PHYSICAL_VALIDATION.md) and
[performance methodology and results](docs/research/PERFORMANCE_RESULTS.md).
These pages summarize the frozen RC1 evaluation; this publication preparation
is not a new GPU experiment.

## How it works

```text
Application: Runtime / Buffer / Lease
                 |
       production residency policy
                 |
       transaction coordinator
                 |
      GPU_RAW <--> GPU_COMPRESSED
                 |
    CUDA VMM + nvCOMP + exact accounting
```

The policy proposes reclaim candidates. The transaction coordinator verifies
ownership, access freshness, integrity and accounting before publishing a new
representation. Compressed storage is not a RAW device pointer: applications
must acquire a lease and restore the required bytes before ordinary compute use.

## Build and test without a GPU

A CPU/Fake build uses CMake 3.25 or newer, a C++20 compiler and LZ4 development
files (1.9.4 or newer). CUDA and physical execution are disabled here.
Run these commands from the extracted repository root:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DVRAMZ_ENABLE_CUDA=OFF -DVRAMZ_ENABLE_NVCOMP=OFF \
  -DVRAMZ_ENABLE_PHYSICAL_RUNTIME=OFF
cmake --build build-cpu --parallel 2
ctest --test-dir build-cpu --output-on-failure --parallel 2
```

This is the full CPU/Fake suite, including the larger fault-injection tests.
See [testing](docs/TESTING.md) for scope, tooling checks and fixture provenance.
CPU/Fake success does not establish hardware compatibility.

## Use the physical runtime

The evaluated configuration was Linux x86-64 / Ubuntu 24.04-family userspace,
RTX 3060 device 0, Driver **595.91.07**, CUDA **13.3.1**, nvCOMP **5.3.0.16** and
static LZ4 **1.9.4**. Physical builds intentionally verify exact reference provider
hashes. An arbitrary newer Driver or same-version library rebuild is not silently
accepted as the evaluated installation.

NVIDIA libraries are external; this source archive does not bundle them.
Normal runtime use is unprivileged. Hardware self-tests, demos and benchmarks
require an explicit opt-in and do real GPU work.

Start with [build instructions](docs/BUILD.md), then
[installation and CLI usage](docs/INSTALL.md) and the
[public C++ API guide](docs/API.md).

## Repository guide

| Path | Contents |
|---|---|
| `include/`, `src/` | Frozen RC1 runtime, API and backend implementations |
| `cmake/` | Build configuration, execution gates and packaging |
| `tools/` | Product CLI, reporting, simulator and validation utilities |
| `examples/` | Public-API integration example |
| `tests/` | Unit, integration, property, fault and tooling tests; reference fixtures |
| `docs/` | Public technical guides and evaluated research summaries |

The full [documentation index](docs/README.md) separates usage, design and results.
Internal handoffs, milestone diaries, raw campaign logs, SDKs, model weights and
build outputs are not part of this publication tree. Regression names such as
`m10` remain in code so the original test identities and implementation stay intact.

## Limits and research status

This preview has synchronous public copies, no supported external asynchronous
CUDA-stream integration, a 16 MiB chunk limit, a 256 MiB CLI logical-workload limit,
and a 512 MiB explicit physical-resource ceiling. It is not a full-card-scale,
production LLM, gaming, multi-GPU, OOM-recovery or long-duration reliability result.
Read [limitations](docs/LIMITATIONS.md) before use.

The research snapshot is frozen. Later AI-performance experiments are excluded.
See [future research](docs/research/FUTURE_WORK.md) for unresolved questions and
[provenance](docs/research/PROVENANCE.md) for the distinction between the evaluated
RC1 archive and this documentation/packaging overlay.

## License, credit and reporting

Copyright © 2026 Rami Nezer.

VRAMZ is open-source software licensed under the
[Apache License 2.0](LICENSE). The license permits use, modification and
distribution, including commercial use, subject to its terms. Third-party
notices are retained in [NOTICE.md](NOTICE.md).

Use [CITATION.cff](CITATION.cff) to cite the work.
