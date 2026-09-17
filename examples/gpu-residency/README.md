# Installed public-API demo

This consumer links only `VRAMZ::cuda` and uses public Runtime, Buffer, Lease,
chunk inspection and accounting interfaces. The installed example includes
`workload.cpp` and `workload.hpp`, the same deterministic application logic used
by the user-facing demo. It supplies only a target to production policy.

Configure `CMAKE_PREFIX_PATH` to the VRAMZ installation and explicitly set
`VRAMZ_DRIVER_LIBRARY`, `VRAMZ_CUDART_LIBRARY`, `VRAMZ_NVCOMP_LIBRARY`,
`VRAMZ_LZ4_LIBRARY` (static 1.9.4) and `VRAMZ_LZ4_INCLUDE_DIR` to external
reviewed providers. No development-tree path is embedded in the package config.
Inspect the consumer ELF and verify providers before execution. Arrange a
command-scoped loader path and invoke `--run-device-0-demo DRIVER CUDART NVCOMP`.
The application uses device 0 only and a 128 MiB explicit-resource ceiling.
