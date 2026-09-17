# Architecture and accounting

VRAMZ is an explicitly integrated C++20 userspace runtime. Applications own a
`Runtime`, allocate move-only `Buffer` objects, and acquire move-only `Lease`
objects before reading or writing bytes. Closing a writable lease finalizes
logical integrity metadata. Closing a buffer and shutting down the runtime are
observable operations: callers must inspect their results.

`Runtime::create()` creates the CPU model. Physical CUDA is a separate opt-in
`VRAMZ::cuda` library and `Runtime::create_cuda()` factory. Device 0, explicit
provider paths, execution acknowledgement and a bounded GPU-only configuration
are required. The reference platform is RTX 3060, NVIDIA 595.91.07, CUDA 13.3.1
userspace and nvCOMP 5.3.0.16. Other combinations are not validated by this RC.

## Representation and ownership

Each logical chunk has one authoritative representation. RAW storage uses the
buffer's reserved virtual address. Compressed storage is a separately owned VMM
resource and is never exposed as a RAW device span. The source stays authoritative
until TransactionCoordinator commits a fully corroborated destination.

The CPU never selects a test-specific victim. Production policy proposes a
candidate based on pressure, temperature, access revision, usefulness/backoff and
eligibility. TransactionCoordinator validates the proposal and performs the
transition. A stale access revision rejects the proposal atomically and excludes
that chunk for the remaining policy cycle. HARD pressure may consider a HOT
candidate after preferable candidates are exhausted; HOT is not an unconditional
pin. Hold a lease when an allocation must remain accessible.

## Physical accounting

Logical bytes, stored compressed bytes and physical VMM backing are distinct.
For example, 319,079 stored bytes can still require a 2 MiB physical allocation.
A smaller compressed byte count alone does not prove physical savings.

BudgetLedger separates committed allocations, unmaterialized reservations,
staging, workspace and cleanup debt. Migration reserve is inside the hard limit.
Checked bounds cover destination, bound output, workspace, metadata and exact
compaction. Actual materialized charge replaces an admission bound only after
backend corroboration. At quiescence, owned physical charge plus unmaterialized
reservations must equal Ledger charge. No host tier contributes to the v1
GPU-resident capacity results.

The physical factory enforces at most 512 MiB of explicit owned/reserved GPU
charge. This ceiling does not include opaque Driver/context/SDK internal overhead
or other applications' allocations, and it is not a free-memory or OOM guarantee.
The CLI uses explicit workloads of at most 256 MiB logical data, with an unchanged
16 MiB per-chunk codec limit. It never sizes a workload using free VRAM.

## Integrity and completion

The nvCOMP adapter publishes metadata using asynchronous copies on its dedicated
stream, launches nvCOMP on that same stream, synchronizes, and only then consumes
results. Both stored and logical CRC32C and exact byte comparisons remain enabled.
Bound output is compacted into exact VMM backing with checked ownership transfer.

Known failures retain or clean exactly known ownership. Ambiguous enqueue,
completion or ownership fails stop; there is no destructive retry or guessed-zero
accounting. The CLI prints `final_counts_known=false` on fatal ambiguity.

## Concurrency and capability boundary

Runtime lifecycle, buffer and chunk locks preserve the reviewed lock ordering.
The allocation gate serializes migration/admission. Independent and same-buffer
read leases may coexist; policy cannot migrate leased chunks. Resource snapshots
are intended for quiescent verification after users have joined. Move-only handle objects
are not themselves a synchronization boundary: do not concurrently move, close or destroy
the same Runtime/Buffer/Lease handle object while another thread uses that object. Use
separately acquired read leases for concurrent readers and join users before handle teardown.

The public deferred-completion API has extensive CPU/Fake tests. The physical
adapter currently reports external asynchronous completion unsupported. Do not
submit arbitrary external asynchronous CUDA work and release a lease as though
VRAMZ had tracked its completion. v1's tested interface uses synchronous
`Lease::read` and `Lease::write` and runtime-managed codec work.

## Measurement

`RuntimeConfig::collect_performance` is false by default. When enabled, host
steady-clock counters observe selection, coordinator transitions and completed
codec calls. They do not feed policy decisions. Codec timings include metadata
publication, launch, synchronization and result consumption; they are not
kernel-only CUDA-event timings. The benchmark also records allocation, acquire /
restore and complete verified-workload intervals.
