# Performance results and methodology

## Scope and method

These measurements characterize the explicitly integrated reference workload on RTX 3060
with Driver 595.91.07, live Driver API 13020, CUDA runtime API 13030 and nvCOMP 5.3.0.16.
Each series has two warmups and seven measured rounds. Eight buffers remain logically live.
All data is verified before policy work and after restore; every round checks zero allocations
and ledger charge. A context and stream are retained between rounds and released at series end.
No concurrent quality builds ran during the measured series. Physical execution was device 0 only.

Timing uses the host steady clock. Codec intervals include same-stream metadata publication,
launch, synchronization and result consumption. They include host work and synchronization;
they are not CUDA-event kernel times. Per-cycle codec/coordinator counters are call averages.
Their p95 is a distribution of those means, not a fabricated distribution of individual kernels.

Total workload time includes deterministic host-data construction, initialization, verification,
accesses, reclaim, actual restoration, byte/CRC checks and cleanup. The RAW baseline performs the
same verified workload with policy disabled. Reported overhead is this complete application
comparison, not gaming/LLM slowdown or an isolated codec throughput claim.

## Optimized Release measurements

| Profile | Logical MiB | Settled VMM MiB | Peak explicit MiB | Policy median ms | RAW median ms | Measured overhead |
|---|---:|---:|---:|---:|---:|---:|
| high | 64 | 28 | 86 | 2419.273 | 667.369 | 262.51% |
| moderate | 64 | 50 | 86 | 3526.076 | 641.722 | 449.47% |
| random | 64 | 64 | 86 | 4035.684 | 550.809 | 632.68% |
| high | 128 | 58 | 166 | 4015.300 | 1298.951 | 209.12% |
| moderate | 128 | 86 | 166 | 6988.156 | 1259.122 | 455.00% |
| random | 128 | 128 | 166 | 8855.878 | 1096.882 | 707.37% |

Every series passed integrity and final zero cleanup. Random data produced no physical
savings and incurred the cost of bounded nonbeneficial attempts; those costs are retained.
The long-term 15% overhead target was not met. This is an experimental capacity/correctness RC,
not a performance recommendation for latency-sensitive applications.

## Operation timings: 64 MiB highly compressible profile

| Observed interval | Samples | Median ms | p95 ms | Min ms | Max ms | Population stddev ms |
|---|---:|---:|---:|---:|---:|---:|
| workload | 7 | 2419.272902 | 2434.158118 | 2411.336062 | 2434.158118 | 6.648958 |
| allocation | 56 | 23.668035 | 25.232455 | 23.150420 | 28.400747 | 0.802816 |
| raw_acquire | 14 | 0.003798 | 0.005320 | 0.003286 | 0.005320 | 0.000575 |
| restore_acquire | 42 | 82.776719 | 86.004474 | 80.640955 | 86.800643 | 1.564347 |
| policy_call_average | 42 | 0.000832 | 0.001222 | 0.000661 | 0.001382 | 0.000157 |
| compression_call_average | 42 | 152.470779 | 165.973308 | 142.768995 | 167.854062 | 6.323394 |
| decompression_call_average | 7 | 25.775455 | 26.160159 | 25.611414 | 26.160159 | 0.210058 |
| transition_call_average | 42 | 206.396111 | 219.921614 | 197.281153 | 223.820912 | 6.557084 |
| reclaim_cycle | 42 | 206.399251 | 219.924370 | 197.284910 | 223.824248 | 6.557036 |

The complete machine-readable summary contains these intervals for every profile/size,
raw timing samples, warmup counts, verified logical bytes/second, nonbeneficial attempts,
physical ratios and peaks. Run `vramz-benchmark-report` on original NDJSON to reproduce it.

## Optimization performed

The original per-byte bitwise CRC32C was an obvious CPU cost. A runtime-dispatched x86 SSE4.2
implementation with a portable table fallback now computes the identical Castagnoli CRC.
Independent bitwise differential tests cover 2,048 randomized/unaligned spans, empty and short
inputs. Integrity verification was not removed or sampled.

The initial high-profile median was 15.761 seconds, compared with 2.419 seconds after CRC
acceleration (about 84.7% lower elapsed time). The RAW baseline also improved substantially,
so the relative policy overhead did not improve. Do not confuse absolute speedup with reduced
relative overhead. Remaining costs include synchronous host integrity copies, VMM transitions,
bounded workspace/metadata handling and failed compression attempts.

## Evidence provenance

Pre-optimization frozen source: `4000ebba81a92edf6c3ad72c355181716f8ae4546b0fe91c92afdfded16c4fb8`.
Optimized frozen source: `b2df47a7b1e6546e72d592504cc62dc71770ede5ad685d85391c7d7b1217a538`.
Original output, binary hashes, exact commands, static closure and kernel-health observations
are retained in the external campaign evidence under `03-performance/`. Generated summary:
`optimized-summary-r6.json`. No measurement was hand-edited. Historical Gate G remains NOT RUN.
