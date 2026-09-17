# Install and use

Build first with [BUILD.md](BUILD.md). This source-only publication does not include
a newly evaluated GPU binary package. CPU/Fake installation contains the CPU library
and tools; the opt-in physical build additionally provides the `vramz` launcher,
`vramz-run`, benchmark reporting and the public CUDA library.

## Stage before installing

For a CPU-only build, install into a user-owned test prefix:

```sh
cmake --install build-cpu --prefix "$PWD/stage-cpu"
```

For the physical build configured with `/usr` as its prefix, stage and package
without writing to the system:

```sh
DESTDIR="$PWD/stage-physical" cmake --install build-physical
cpack --config build-physical/CPackConfig.cmake -G DEB
```

Inspect package contents, dependencies, checksums and ELF before installing.
Tests, oracle fixtures, internal backend/testing headers and NVIDIA SDKs do not
belong in the installed package. Public guides and third-party notices do.
The packaging metadata remains `1.0.0~rc1`; do not confuse a rebuilt research
package with the earlier evaluated binary. Record the actual source and package hashes.

## Install the verified local package

Only package installation/removal requires privilege. Use the matching package
filename actually produced by CPack:

```sh
sudo dpkg -i ./vramz_1.0.0~rc1_amd64.deb
```

Set `DRIVER`, `CUDA_ROOT` and `NVCOMP_ROOT` as in the build guide. Do not export a
global loader path. The launcher verifies the providers and constructs a scoped
child environment. Normal runtime use must be non-root.

## Inspect, self-test and demo

```sh
vramz --driver "$DRIVER" --cuda-root "$CUDA_ROOT" --nvcomp-root "$NVCOMP_ROOT" --check-only
vramz --driver "$DRIVER" --cuda-root "$CUDA_ROOT" --nvcomp-root "$NVCOMP_ROOT" -- --device 0 --run info
vramz --driver "$DRIVER" --cuda-root "$CUDA_ROOT" --nvcomp-root "$NVCOMP_ROOT" -- --device 0 --run self-test
vramz --human --driver "$DRIVER" --cuda-root "$CUDA_ROOT" --nvcomp-root "$NVCOMP_ROOT" -- --device 0 --run demo
```

**Only `--check-only` is static.** `info` probes device capabilities and the
context/stream lifecycle. Self-test and demo allocate, compress, restore and verify
GPU data. Do not use them on an irreplaceable live workload. Preserve any first
failure output and exit status; do not retry ambiguous completion or cleanup.

## Benchmark

```sh
vramz --driver "$DRIVER" --cuda-root "$CUDA_ROOT" --nvcomp-root "$NVCOMP_ROOT" -- \
  --device 0 --run benchmark --profile high --iterations 7 --warmups 2 > high.jsonl
vramz-benchmark-report high.jsonl > high-summary.json
```

Available profiles include `high`, `moderate`, `random`, `mixed`, `p75`, `p50` and
`p25`. `--raw-baseline 1` disables policy for the matching verified workflow.
Use identical data/size settings for comparisons. Summaries exclude warmups but
retain their original samples. Read [performance methodology](research/PERFORMANCE_RESULTS.md)
before interpreting timings.

The default workload has eight 8 MiB buffers and a 128 MiB explicit-resource cap.
`--buffers`, `--buffer-mib`, `--chunk-mib` and `--hard-mib` are bounded; the RC1
limits are 16 MiB per chunk, 256 MiB logical CLI data and 512 MiB explicit peak.
A controlled cap is not a guarantee of available memory or total process GPU usage.

## Public C++ example

Copy the installed `/usr/share/vramz/examples/gpu-residency` directory into a
user-owned development directory. Follow its README to link `VRAMZ::cuda` with
trusted providers. The source-tree `examples/gpu-residency/README.md`
and [API guide](API.md) describe the same interface. No internal test API is required.

## Uninstall

```sh
sudo dpkg --remove vramz
```

The package has no service or persistent runtime state. User-created result JSON
and external NVIDIA/LZ4 installations are not removed. A verified package may be
installed again with the same `dpkg -i` command.

## Troubleshooting

- **Provider/preflight rejected:** check exact binary hashes, canonical paths,
  ancestor permissions and the loaded Driver version. Never substitute a stub
  or disable the check to accept an unreviewed installation.
- **No physical savings:** inspect physical charge and data profile. Smaller stored
  bytes may still occupy one complete VMM allocation unit. Random data can be
  correctly nonbeneficial.
- **Busy or conflict:** close/join legitimate users; never free unknown in-flight
  work. A HOT label is not a lease.
- **Integrity, completion or cleanup failure:** preserve diagnostics and stop;
  guessed-zero accounting is not recovery.
- **Driver Xid/reset or display instability:** stop GPU work and investigate the
  host separately. VRAMZ does not repair or replace Drivers.
- **Slow execution:** the measured 209–707% verified-workflow overhead is a known
  research limitation, not an installation defect.
