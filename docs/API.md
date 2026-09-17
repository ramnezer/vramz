# Public C++ API

The authoritative declarations are `include/vramz/runtime.hpp` and
`include/vramz/cuda_runtime.hpp` in the source archive. This guide describes
the frozen RC1 interface, not proposed asynchronous or transparent integrations.

## Runtime selection

`Runtime::create(RuntimeConfig)` constructs the CPU model. Real CUDA is explicit:
link the separately enabled `VRAMZ::cuda` target and use
`Runtime::create_cuda(RuntimeConfig, const CudaRuntimeOptions&, CudaRuntimeInfo&)`.

`CudaRuntimeOptions` carries an execution acknowledgement, device ordinal, and
explicit Driver/cudart/nvCOMP paths. This RC's physical factory supports the
reviewed device-0 configuration and caps explicit GPU allocation/reservation
charge at 512 MiB. The caller is responsible for a trusted loader setup before
process startup; the C++ factory is not a substitute for the launcher's full
preflight. See [security model](SECURITY_MODEL.md).

## Lifetime and access

| Interface | Responsibility |
|---|---|
| `Runtime::allocate(ByteSize)` | Create a logically owned Buffer within admission rules. |
| `Buffer::acquire(MemoryRange, AcquireOptions)` | Obtain pinned RAW access; restore compressed data when necessary. |
| `Lease::read(ByteOffset, span<byte>)` | Complete a synchronous verified read. |
| `Lease::write(ByteOffset, span<const byte>)` | Write through an exclusive writable lease. |
| `Lease::close()` | Release access and finalize writable integrity metadata. |
| `Runtime::reclaim_to_target(ByteSize)` | Request production-policy reclaim by budget target, not victim identity. |
| `Buffer::inspect_chunk(uint64_t)` | Inspect authoritative representation/charge and available policy metadata. |
| `Runtime::stats()`, `resources()`, `performance()` | Read accounting, quiescent resource counts and optional timings. |
| `Buffer::close()`, `Runtime::shutdown()` | Explicitly close ownership and inspect cleanup results. |

Runtime, Buffer and Lease ownership is non-copyable. Their exact special-member
rules are in the headers. Do not concurrently move, close or destroy the same
handle object while another thread uses it. Use independently acquired leases
for independent readers, and join users before teardown.

A lease is the hard pinning boundary. HOT/WARM/COLD is policy metadata and does
not make an allocation permanently immovable. Acquiring compressed data restores
a RAW representation before it can be exposed for ordinary use.

## Error handling

Operations return `Result<T>` or `Result<void>`. Check results from allocation,
acquire, copies, close, reclaim and shutdown; do not discard cleanup failures.
Error values use fixed-size structured fields. Runtime also exposes bounded
asynchronous error polling and a first-error record.

Known failures retain or clean exactly known ownership. Ambiguous GPU completion
or ownership is fail-stop, not a signal to retry destruction. This release has
no safe forced-free shortcut for unknown in-flight work.

## Completion boundary

The public API contains `acquire_async`, `PendingLease`, completion tokens and
deferred lease release, with CPU/Fake coverage. **The physical RC1 adapter does
not support arbitrary external asynchronous CUDA-stream completion.** Use the
validated synchronous public copies and runtime-owned codec work. Do not enqueue
unknown external kernels, close a lease and assume VRAMZ tracked their lifetime.

## Accounting

Keep three values separate: logical size, stored compressed length and physical
VMM allocation charge. Use authoritative chunk inspection and quiescent resource
counts; never derive a physical-capacity claim solely from codec output size.
Migration reserve is inside the hard limit and is not free extra capacity.
The complete accounting contract is summarized in [ARCHITECTURE.md](ARCHITECTURE.md).

## Complete examples

The source example `examples/gpu-residency/main.cpp` uses only public
interfaces and the shared deterministic workload implementation. Its installed
CMake project links `VRAMZ::cuda`. See `examples/gpu-residency/README.md` in the source archive
for provider arguments and explicit execution.

The source `tests/consumer/main.cpp` is a minimal installed-header check
using `Runtime::create`, allocation, close and shutdown without NVIDIA execution.
Public consumer builds are separate from private backend/fault-injection tests.
