# VRAMZ release notice

Copyright © 2026 Rami Nezer.

This public open-source tree is a publication overlay built from the evaluated frozen VRAMZ v1.0.0-rc1 source archive and is distributed under the Apache License 2.0.

Frozen evaluated RC1 archive SHA256:

`e0165afa1fa8e06a9f6ef9fcd99fc6182873a908ecfc4810c44ea4f0fcb3a395`

The public overlay adds publication metadata/documentation and removes internal milestone/authorization logs. Core runtime implementation files under `include/`, `src/`, public examples and product tools are intended to remain byte-identical to the frozen RC1 source as checked by the accompanying publication validation manifest.

NVIDIA CUDA/nvCOMP/Driver libraries are external dependencies and are not distributed by VRAMZ. See [publication provenance](docs/research/PROVENANCE.md) for the documentation/fixture cleanup.

LZ4 attribution is retained in `docs/THIRD_PARTY_LZ4_COPYRIGHT.txt`.
