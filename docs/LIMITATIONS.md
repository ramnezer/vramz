# Exact limitations

- Experimental explicitly integrated C++ runtime; no transparent system-wide VRAM.
- Only RTX 3060 device 0 with the pinned NVIDIA 595.91.07 / CUDA 13.3.1 /
  nvCOMP 5.3.0.16 providers is physically validated in this campaign.
- Compression benefit depends on actual data and live VMM granularity. Stored
  compression ratio is not physical capacity. Incompressible profiles save nothing.
- Policy uses deterministic access metadata, not a learned workload predictor.
  HARD pressure may select HOT after less-preferred candidates fail; a live lease
  is the ownership mechanism for pinning data, not a temperature label.
- The physical adapter supports synchronous public copies and its own dedicated
  codec stream. External asynchronous CUDA stream completion is not supported.
- CPU CRC/reference integrity validation and host round trips impose substantial
  overhead. Performance measurements describe the verified workflow and include
  synchronization. No <=15% universal overhead guarantee is made.
- The physical API limits explicit tracked GPU ownership/reservations to 512 MiB,
  chunk size to 16 MiB and CLI logical workload to 256 MiB. These are campaign/RC
  limits, not measured hardware capacity or available-memory guarantees.
- Driver/context/nvCOMP internal allocations are opaque and are not claimed to be
  part of exact explicit VMM accounting. Other applications' memory is outside the
  runtime's ownership.
- GPU-only capacity tests use HOST budget zero. HOST fallback, real GPU exhaustion,
  Driver OOM recovery, arbitrary application integration, gaming/graphics/LLM
  behavior, multi-GPU and long-duration stability are not validated.
- Bounded soak evidence is not a production stability SLA. Ambiguous GPU ownership
  or completion is fail-stop. No destructive retries or guessed cleanup values.
- Installed applications must use compatible compilers/standard libraries and
  explicit trusted external providers. There is no stable cross-version C++ ABI
  promise for this RC.
- This is an open-source prerelease licensed under the Apache License 2.0.
  It remains experimental and is not presented as production-validated software.

Historical Gate G remains NOT RUN against its original complete capacity and
performance acceptance criteria. The campaign's bounded RC acceptance is tracked
separately as `VRAMZ_V1_RC_READY`; broader research requirements are not silently
weakened to obtain a release label.

The public distribution contains evaluated summaries, not the complete private
raw measurement archive. The recorded capacity ratios describe settled snapshots;
they do not reduce initialization, temporary migration or restore peaks to the
settled backing size. See [reproducibility](research/REPRODUCIBILITY.md).
