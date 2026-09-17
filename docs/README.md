# Documentation

VRAMZ v1.0.0-rc1 research preview. This is public usage/design documentation,
not a development diary. Start from the [project overview](../README.md).

## Build and use

- [Build](BUILD.md): CPU/Fake and opt-in physical builds; reference dependency boundary.
- [Install and run](INSTALL.md): staging, packaging, safe launcher, self-test and uninstall.
- [Public API](API.md): Runtime/Buffer/Lease ownership, copies, reclaim and inspection.
- [Tests](TESTING.md): full regression suite, tooling checks and retained fixtures.

## Design and safety

- [Architecture](ARCHITECTURE.md): representations, transactions and physical accounting.
- [Limitations](LIMITATIONS.md): supported scope and important non-goals.
- [Security model](SECURITY_MODEL.md): provider trust, privileges and failure behavior.
- [Third-party LZ4 notice](THIRD_PARTY_LZ4_COPYRIGHT.txt).

## Research results

- [Physical validation](research/PHYSICAL_VALIDATION.md).
- [Capacity results](research/CAPACITY_RESULTS.md).
- [Performance results and methodology](research/PERFORMANCE_RESULTS.md).
- [Reproducibility](research/REPRODUCIBILITY.md).
- [Reported quality and review scope](research/QUALITY.md).
- [Publication provenance](research/PROVENANCE.md).
- [Future work and transparent-integration boundary](research/FUTURE_WORK.md).

Physical measurements are the reported frozen RC1 evaluation. Raw private campaign
logs are not bundled, and publication checks do not substitute for new GPU measurements.
