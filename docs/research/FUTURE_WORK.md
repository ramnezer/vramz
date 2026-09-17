# Future research

## Transparent integration boundary

The current runtime owns only allocations made through its public Buffer/Lease
API. It cannot change a third-party `cudaMalloc` allocation, observe arbitrary
kernel lifetimes, safely replace an application's pointers, or change the
physical memory that NVIDIA tools report. Compressed backing is not directly
readable by an application's kernel. A RAW lease must restore the resource first.

Therefore arbitrary unmodified CUDA, gaming, graphics and LLM programs do not
receive extra addressable VRAM from installing v1. This is an architecture
boundary inferred from the actual ownership and completion interfaces, not a
missing desktop switch.

Potential future integrations:

- Explicit application/framework allocators: strongest ownership and completion
  contract, but each integration must implement stream-fence and pinning rules.
- Compiler/task-graph integration: can expose lifetimes and hot sets, but requires
  application/toolchain participation and workload-specific validation.
- Driver/UVM cooperation: could coordinate faults and migration more broadly, but
  requires a different trust boundary, kernel/Driver support and much larger
  correctness/security validation.
- API interception: cannot infer arbitrary device-pointer aliasing, graphics
  ownership or work on unknown streams. LD_PRELOAD interception is not an
  implementation or supported deployment mode in this RC.

No kernel/UVM patches, Driver replacement, synthetic nvidia-smi capacity, or
system-wide interception are part of this research release. A transparent
architecture requires a separate ownership design and validation effort.

## Other unresolved questions

- Reduce synchronous verification/copy and VMM/codec transition costs without
  removing integrity or exact accounting.
- Design explicit external stream/fence integration with bounded lifetime tracking.
- Evaluate batching or packing only with a corresponding ownership/accounting model.
- Study broader data distributions, temporal access patterns and device/provider combinations.
- Validate long-duration operation, real OOM recovery and HOST fallback separately;
  they are not implied by the existing bounded GPU-resident tests.
- Measure useful application/framework integrations before making AI, graphics or
  gaming claims. RC2 AI experiments are not included in this frozen RC1 publication.

These are research directions, not a commitment to implement them or a guarantee
that every workload can benefit from lossless GPU compression.
