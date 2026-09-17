# Capacity results

## Controlled 64 MiB profile matrix

| Data profile | Logical MiB | Settled VMM MiB | Logical / physical |
|---|---:|---:|---:|
| Highly compressible | 64 | 28 | 2.285714x |
| Mixed high/moderate/random | 64 | 32 | 2.0x |
| 75/25 high/random | 64 | 34 | 1.882353x |
| 50/50 high/random | 64 | 40 | 1.6x |
| 25/75 high/random | 64 | 52 | 1.230769x |
| Random / effectively incompressible | 64 | 64 | 1.0x |

These ratios use measured VMM physical backing, not stored compressed-byte counts.

## Representative scale

The evaluated campaign included 8, 32, 64, 128 and 256 MiB logical workloads. The 256 MiB representative workload settled at 116 MiB of explicit VMM backing with a 294 MiB measured explicit peak.

At the 2 MiB VMM granularity observed on the reference GPU, very small chunks can fail to save a whole physical allocation unit even when the stored compressed representation is smaller. VRAMZ therefore treats physical VMM charge — not codec output size — as the capacity authority.

See [PHYSICAL_VALIDATION.md](PHYSICAL_VALIDATION.md) for complete interpretation and safety boundaries.

## Settled backing is not peak memory

The 64 MiB highly compressible case reports 28 MiB at the settled snapshot and
86 MiB peak explicit allocation/reservation charge. Initial RAW storage, migration
staging and restore require additional headroom. The result does not prove that
the complete workload executes under a 28 MiB or 32 MiB physical hard limit.
