# Reported RC1 quality and review scope

These are the quality results reported with the frozen evaluated RC1 source,
not a claim that every check was repeated during publication cleanup. The external
publication validation report lists the new checks separately. Raw private
campaign logs are not included in this source-only distribution.

| Check | Result and scope |
|---|---|
| GCC Debug/Release, SDK OFF/ON | Four full configurations PASS; 30 CTest suites / 772 named deterministic cases each; zero compiler warnings. |
| Clang Debug/Release, SDK OFF/ON | Four full configurations PASS; same complete regression and zero compiler warnings. |
| clang-format | PASS, all 122 owned C++/header files. |
| clang-tidy | PASS, SDK OFF and ON configured owned source. |
| Clang Static Analyzer | PASS, SDK OFF and ON. |
| Cppcheck | PASS, SDK OFF and ON; one exact pre-existing NVIDIA-header signed-shift exception, no broad suppression. |
| GCC fanalyzer | PASS, SDK OFF and ON, production/tool code with tests OFF. |
| Python | compileall PASS; nine build-gate/product/parser files, 113 tests PASS. |
| ASan + LSan | Full 30 suites / 772 cases PASS. |
| UBSan | Full 30 suites / 772 cases PASS. |
| Standalone LSan | Full 30 suites / 772 cases PASS. |
| TSan | Full 30 suites / 772 cases PASS on identical binaries in two disjoint shards. |
| Bounded libFuzzer | 20,000 runs PASS; seed 12648430; max input 1 MiB; RSS limit 512 MiB; ASan/UBSan enabled. |
| CPU LZ4 OFF | 11 applicable suites / 161 named cases PASS. |
| Benchmark parser | All 12 retained physical series accepted by the final stricter parser; no measurement edited. |
| M3/M5 oracles | Two independent repetitions each, byte-identical to approved hashes. |


## Scope and limitations

SDK-ON checks compile real adapters and entry-point objects; executed CPU test
ELF closures exclude NVIDIA libraries. GCC fanalyzer covers production/tools with
tests OFF. Cppcheck retains its configured preprocessor-configuration bound and
the exact vendor `cuda.h:25952` signed-shift exception. No broad exception is
introduced by this publication cleanup.

The reported TSan coverage combined two disjoint shards of identical binaries
covering all 30 suites. The interrupted scheduler command itself remains a
non-success record in the original evidence; it was not relabelled as one complete
successful run. The combined result and the individual command status are different facts.

## Reviewed contracts

The source review covered checked ranges and reservations, authoritative
representations, lease pins and lifecycle, concurrent readers, policy freshness,
CRC/byte integrity, same-stream nvCOMP completion, provider identity and loader
trust, bounded report parsing and install/uninstall behavior.

Important limits remain: HOT is advisory rather than a pin; external asynchronous
CUDA completion is unsupported by the physical adapter; root and same-UID trust
are outside the provider-check security boundary; opaque Driver allocations are
not exact VRAMZ backing. See [architecture](../ARCHITECTURE.md),
[security model](../SECURITY_MODEL.md) and [limitations](../LIMITATIONS.md).

The reported engineering outcome is:

**zero unresolved findings in the currently configured checks**

It is not a claim of zero vulnerabilities or proof for all possible schedules,
inputs, tool configurations or NVIDIA hardware.
