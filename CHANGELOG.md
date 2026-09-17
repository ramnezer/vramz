# Release notes

## v1.0.0-rc1-research

Prepared from the frozen evaluated `v1.0.0-rc1` as an open-source research
preview. This is publication cleanup, not a new runtime/performance milestone.

- Retains the RC1 Runtime/Buffer/Lease implementation, CUDA VMM/nvCOMP path,
  production policy, accounting and existing regression tests.
- Presents one public documentation set for build, installation, API, design,
  testing, measured results and limitations.
- Moves required reference fixtures next to tests without changing their bytes.
- Omits private development diaries, duplicate documentation and raw campaign logs.
- Releases the publication tree under the Apache License 2.0 while retaining
  author attribution and third-party notices.

Known limits include substantial measured workflow overhead, explicit integration,
reference-provider restrictions, bounded experimental sizes and no supported
external asynchronous-stream integration. See [limitations](docs/LIMITATIONS.md).

Publication source has a new external checksum. Prior physical results retain
their original provenance; see [PROVENANCE.md](docs/research/PROVENANCE.md).
