# Security policy

VRAMZ v1.0.0-rc1 is an experimental research runtime, not a production security boundary.

## Reporting a vulnerability

Please report security-sensitive issues privately to the project owner rather than publishing exploit details in a public issue before coordination is possible. If no private contact channel is listed in the repository, open a minimal public issue requesting a private security contact without including exploit details.

Useful reports include:

- affected version and platform;
- reproducible steps;
- expected and observed behavior;
- whether GPU state, ownership, accounting, integrity, loader/provider validation or privilege boundaries are involved.

## Scope

The security model is described in [docs/SECURITY_MODEL.md](docs/SECURITY_MODEL.md). Normal runtime use is unprivileged. The project does not install a daemon, setuid helper, kernel module, Driver replacement, or global loader modification.

CRC32C is used for accidental-corruption detection; it is not cryptographic authentication. Same-UID compromise and administrator/root compromise are outside the trust boundary.
