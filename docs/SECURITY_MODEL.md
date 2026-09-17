# Security and operational model

Normal execution is unprivileged. There is no daemon, service, network listener,
setuid binary, kernel module or global loader modification. Public examples and
workloads require explicit device-0 acknowledgement.

Use the installed `vramz` launcher for the reference provider preflight. It checks
ELF dependency closure and pinned NVIDIA hashes before starting the executable,
rejects stubs, RPATH/RUNPATH, dynamic LZ4 fallback, loader shadowing and untrusted
file ownership/permissions, then sets a command-scoped loader environment. It
drops inherited loader/audit/preload variables instead of extending them. CUDA
and nvCOMP must be supplied externally; they are not bundled into the package.

Providers must be read-only mounted or writable only by root/the invoking user.
Writable parent directories may additionally use a private group containing only
that user/root; other group/world writers are rejected. That user and root are
trusted. The checks are not a security boundary
against a malicious same-UID process or administrator replacing files between
verification and loading. Do not update providers concurrently with a run. Hash
verification after a campaign run records drift; it cannot retroactively make an
in-flight replacement safe. The C++ factory additionally compares loaded symbol
providers with the explicit canonical paths and the loaded kernel release.
Applications using the C++ library directly must arrange equally trusted loader
paths before the process starts; a check in `main` cannot undo a malicious shared
library constructor.

Sizes and offsets use checked arithmetic. Bounded pools and registries limit
metadata growth. CRC32C detects accidental corruption; it is not a cryptographic
message-authentication mechanism. The integrity design assumes the host process
and its memory are trusted, not adversarial data authentication.

On ambiguous device ownership/completion, the runtime fails stop. Preserve the
first diagnostic and do not destructively retry. Driver Xid/reset or display
instability requires operator investigation. Do not use this experimental RC for
irreplaceable live workloads. Back up input data and test integration separately.

The evidence states checks actually executed. It does not claim absence of all
vulnerabilities, universal GPU compatibility, OOM recovery or production SLA.
