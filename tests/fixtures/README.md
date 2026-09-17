# Retained reference fixtures

These files are required inputs to the CPU prehardware/package verification tools.
They were moved out of `docs/` without changing any bytes.

| File | Role |
|---|---|
| `oracles/M3_SCENARIO_RESULTS.jsonl` | 82 deterministic simulator scenarios |
| `oracles/M5_FAKE_GPU_COMPRESSION_RESULTS.jsonl` | Four Fake compression/granularity rows |
| `PREHARDWARE_DEPENDENCIES.json` | Historical development-only dependency profile |

SHA256:

```text
8094c716c4dad5e494eeaa7261b5aa3b8ff73670ee0b47c881e0049fe5a73948  oracles/M3_SCENARIO_RESULTS.jsonl
fb39d58d3f7196031c04d262b4ec9822bab324abbac3288b9fef7be24aeda7ee  oracles/M5_FAKE_GPU_COMPRESSION_RESULTS.jsonl
```

The prehardware manifest's old release identifier and `NOT_TESTED` state describe
that compile/CPU-only fixture. They do not describe the active RC1 public release
or overwrite its reported hardware results. These files are not installed in
binary packages and must not be presented as new physical measurements.
