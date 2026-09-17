# Regression tests

This directory is source code, not a collection of development logs. All existing RC1 C++ tests are retained unchanged. Five Python build-gate
fixtures now pass the explicit reviewed-LZ4-prefix input introduced by the
earlier publication overlay, and separately test missing-prefix rejection.
See [the testing guide](../docs/TESTING.md) for build and execution commands.

| Directory | Purpose |
|---|---|
| `unit/` | Values, argument parsing, bounds and report-format checks |
| `integration/` | Runtime/backend flows, compression, residency and lifecycle |
| `property/` | Deterministic model sequences and CRC differential checks |
| `fault/` | Injected failure, rollback, ownership and accounting checks |
| `support/` | Fake CUDA/nvCOMP support with real CPU byte payloads |
| `tooling/` | Build gates, loader checks and benchmark-report tests |
| `fuzz/` | Bounded CPU codec fuzz target |
| `consumer/` | Installed public-header consumer |
| `fixtures/` | Small reference metadata and byte-identical oracle data |

Names such as `m7` through `m13` are retained regression identifiers. They are not
instructions to repeat historical hardware runs. Physical smoke targets remain
separate, default-OFF and outside ordinary CTest/install execution.
