# Physical validation of the experimental RC

## Reference platform and safety

The measured platform is NVIDIA GeForce RTX 3060, device 0, Driver 595.91.07,
CUDA development toolkit 13.3.1 / runtime API 13030, nvCOMP 5.3.0.16 and static
LZ4 1.9.4. The Driver API observed physically is 13020. Compatibility remains
`MINOR_COMPATIBILITY_CANDIDATE`; completed primitive/workload tests establish the
specific observations below, not support for every CUDA 13 feature or NVIDIA GPU.

Every new campaign physical run used a frozen source identity, an executable
hash, static recursive ELF/provider checks and a command-scoped loader path.
No Driver stubs or dynamic LZ4 fallback were accepted. Original NDJSON and exit
codes are retained separately from explanations. Historical physical evidence was preserved. Device 1 was not used.

The absolute explicit-resource ceiling is 512 MiB. Each run uses a smaller
configured hard cap where possible, including migration reserve, codec buffers,
workspace and metadata. No free-VRAM query was used to size a workload, no
intentional Driver OOM occurred, and no global Driver/loader setting was changed.
Opaque Driver/runtime internal allocations are outside the explicitly accounted
VRAMZ backing; the cap is not a claim about total GPU memory used by the process.

## Mixed and profile results

All rows use eight independent 8 MiB logical buffers: 64 MiB logical total.
Production policy chooses victims; the application supplies only a target.

| Deterministic profile | Measured settled VMM MiB | Logical/physical ratio | Outcome |
|---|---:|---:|---|
| Highly compressible | 28 | 2.285714 | target reached; HOT and WARM retained RAW |
| 75/25 high/random | 34 | 1.882353 | smaller benefit; target not reached |
| 50/50 high/random | 40 | 1.6 | smaller benefit; target not reached |
| 25/75 high/random | 52 | 1.230769 | smaller benefit; target not reached |
| Random | 64 | 1.0 | safe nonbeneficial rejection; no physical savings |
| Mixed high/moderate/random | 32 | 2.0 | first COLD random candidate rejected; useful alternatives selected |

PASS in these profile tests means correct decisions, integrity and accounting,
not a forced capacity factor. The 25/75 profile demonstrates unchanged M3 behavior:
under HARD pressure, HOT can be considered after preferable candidates fail.
An active lease protects access; the HOT label alone is not an unconditional pin.
No HOST capacity was used. Rejected candidates receive the existing backoff;
there is no repeated futile transition for the same candidate within one cycle.

## Representative scale

| Logical MiB | Buffers / buffer MiB / chunk MiB | Settled VMM MiB | Peak explicit MiB |
|---:|---|---:|---:|
| 8 | 4 / 2 / 2 | 8 | 18 |
| 32 | 4 / 8 / 8 | 14 | 54 |
| 64 | 8 / 8 / 8 | 28 | 86 |
| 128 | 8 / 16 / 16 | 58 | 166 |
| 256 | 8 / 32 / 16 | 116 | 294 |

At 2 MiB VMM granularity, a 2 MiB chunk cannot save a whole physical unit; that
case correctly retained 8 MiB aggregate backing. The 256 MiB workload kept each
chunk inside the unchanged 16 MiB codec limit. Backend charges, not stored byte
counts, determine every physical result.

## Concurrency and integrity

Physical concurrency used three public-API reader threads: two independently
leased readers of the active buffer and one reader of another buffer, overlapping
production reclaim. Leases remained pinned until readers joined. All bytes and
CRCs matched; final resources, VA reservations and ledger charge returned to zero.
Fake and sanitizer tests additionally cover stale revisions, deferred completion,
shutdown/destruction races and bounded registries. External asynchronous CUDA
completion is not advertised by the physical public adapter.

Every successful workload verifies actual authoritative source data, stored
compressed integrity and actual restored size/CRC/bytes. The settled snapshot is
captured before restoration. Cleanup includes codec workspace, metadata, bound
output, compaction, RAW/compressed allocations, reservations, VA, stream and context.
An ambiguous completion or ownership result fails stop; it does not report guessed
zero counters or retry destructively.

## Repetition and evidence

Bounded soak validation is recorded under external campaign evidence `04-soak/`.
Two planned 64-round series (high and mixed) passed: 768 real compressions,
768 verified restores and 64 additional nonbeneficial attempts. Each of 128
checkpoints proved zero GPU allocations/VA/workspace/reservations/ledger charge,
with one deliberately retained context and stream; final shutdown released both.
Peak explicit charge was 90,177,536 bytes under a 128 MiB cap. Recorded seeds were
5,655,898 through 5,655,961. High-profile RSS plateaued; mixed RSS varied within
a bounded observed range. No new NVIDIA kernel fault or resource drift occurred.
This is bounded repeated-transition evidence, not long-duration stability proof.
Performance measurements are described separately in [PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md).
Original per-run source/binary identities, commands, exact results, checksums,
provider closure and kernel-health logs were retained in the private development evidence.
This public release provides the evaluated summaries rather than the complete internal log set.

These are controlled deterministic userspace results. They do not prove universal
2x VRAM, actual exhaustion/OOM recovery, HOST fallback, gaming/graphics/LLM behavior
or long-duration production stability. Historical Gate G remains NOT RUN.
