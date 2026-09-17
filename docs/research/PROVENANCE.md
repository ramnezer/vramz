# Publication provenance

## Three separate identities

| Snapshot | Archive SHA256 | Meaning |
|---|---|---|
| Frozen evaluated RC1 | `e0165afa1fa8e06a9f6ef9fcd99fc6182873a908ecfc4810c44ea4f0fcb3a395` | Original evaluated implementation and source documentation |
| Earlier research overlay | `b586c394d3b8a6ad1e7f16b8aba7565f2390cbdf8ce6711d95dbfb5d12678f26` | Added initial licensing, citation and publication material |
| Clean publication tree | External archive checksum | This documentation/fixture organization pass |

The final publication archive hash is external to avoid a self-referential hash.
A version label alone is not evidence of validation or physical execution.

## What changed in this pass

Duplicate `V1_*` pages and large historical design/test diaries were consolidated
into the public guides. Internal checkpoints, publication-preparation checklists
and the copied old private README were removed from the distribution. The source
archive remains available as the immutable record; those deletions do not alter it.

The M3/M5 JSONL oracles and prehardware dependency manifest were moved byte-for-byte
into `tests/fixtures`. Two verification scripts were updated only for those paths.
CMake installs public documentation, not the test fixtures. The source-archive
helper now explicitly accepts `LICENSE`, `CITATION.cff` and `.gitattributes`, so
it can package the public tree without dropping its licensing/citation metadata.
A new tooling regression checks this publication layout and archive behavior.
Five existing Python gate fixtures were brought into alignment with the explicit
`VRAMZ_REVIEWED_LZ4_PREFIX` introduced by the earlier overlay: invalid-library
checks reach their intended guard again, and missing-prefix rejection has its
own test. No production guard or assertion was removed to obtain a PASS.
The CPU installed-consumer fixture now requests the actual `1.0` CMake package
series instead of its obsolete `0.6` requirement. The public GPU example already
requested `1.0`; no library ABI or runtime code changed.
Existing C++ tests,
runtime/API code, product implementation and physical execution barriers were not
rewritten for publication. Names of historical test targets remain intact.

The public tree is licensed under the Apache License 2.0. Copyright attribution
and the third-party LZ4 notice are retained. This publication cleanup does not
change the provenance of the frozen evaluated runtime or its recorded results.

## Claims and historical scope

The reported physical results refer to the original evaluated RC1 and its recorded
provider configurations. This cleanup does not claim new GPU execution or attach
historical measurements to a newly built binary. The 2.286× example is a settled
snapshot result, not a peak-memory bound or a universal capacity multiplier.

Historical `Gate G = NOT RUN` and the separate evaluated
`VRAMZ_V1_RC_READY = PASS` remain distinct. The former includes broader research
capacity/performance criteria; publication does not redefine it.

The inherited quality summary contains an isolated `rc2` label despite being
shipped inside the frozen RC1 archive. The public quality page identifies the
reported matrix by its frozen source provenance, rather than asserting that this
publication includes the later RC2 AI experiments.

## Local publication validation

A separate validation report and machine-readable manifest accompany this delivery.
They record original/final hashes, preserved code and tests, file moves/removals,
commands actually run and limitations. Build logs and audit manifests are not
installed as public product documentation or mixed into hardware evidence.
