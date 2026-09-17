#!/usr/bin/python3
"""Summarize VRAMZ benchmark NDJSON; warmups stay in raw evidence, not statistics."""
import argparse
import json
import math
import re
from pathlib import Path
import statistics
import sys


def summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"count": len(values), "median_ns": statistics.median(values),
            "p95_ns_nearest_rank": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "minimum_ns": ordered[0], "maximum_ns": ordered[-1],
            "population_stddev_ns": statistics.pstdev(values), "samples_ns": values}


def unsigned(value, minimum=0, maximum=(1 << 64) - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("Invalid bounded unsigned integer")
    return value


def zero_cleanup(row, retained=False):
    if row["cleanup"] is not True or row["final_counts_known"] is not True:
        raise ValueError("Final accounting is not proven")
    for key in ("backend_resources", "raw_resources", "compressed_resources",
                "va_reservations", "budget_charge", "unmaterialized_reservations", "workspace"):
        if unsigned(row["final_" + key]) != 0:
            raise ValueError("Nonzero final resources")
    for key in ("final_context", "final_stream"):
        if row[key] is not retained:
            raise ValueError("Unexpected context/stream ownership")
    if "final_owned_charge" in row and unsigned(row["final_owned_charge"]) != 0:
        raise ValueError("Nonzero final owned charge")


def counter(item):
    if item["overflow"] is not False:
        raise ValueError("Timing counter overflow or unknown overflow state")
    n, elapsed = unsigned(item["samples"]), unsigned(item["nanoseconds"])
    if not n and elapsed:
        raise ValueError("Elapsed time without samples")
    return elapsed / n if n else None


def analyze(text):
    if len(text) > 16 * 1024 * 1024:
        raise ValueError("Benchmark input exceeds the 16 MiB bound")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or rows[-1].get("kind") != "series_summary" or rows[-1].get("result") != "PASS":
        raise ValueError("A completed, successful series with final cleanup is required")
    end = rows[-1]
    zero_cleanup(end)
    iterations = unsigned(end["iterations"], 1, 128)
    warmups = unsigned(end["warmups"], 0, 8)
    source = rows[0]["source_sha256"]
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{64}", source) or source == "0" * 64:
        raise ValueError("Missing frozen source identity")
    workloads = [r for r in rows if "v1_workload_version" in r]
    if len(workloads) != iterations + warmups:
        raise ValueError("Incomplete samples")
    keys = ("aggregate_logical_bytes", "profile", "raw_baseline", "chunk_count",
            "buffer_count", "maximum_physical_bytes")
    for index, row in enumerate(workloads):
        if (row["result"] != "PASS" or row["integrity"] is not True or
                row["measurement_enabled"] is not True or row["snapshot_proven"] is not True):
            raise ValueError("Failed or unmeasured workload")
        if (unsigned(row["iteration"], 0, 135) != index or
                row["warmup"] is not (index < warmups) or
                row["runtime_retained"] is not True or type(row["raw_baseline"]) is not bool):
            raise ValueError("Invalid series ordering or runtime ownership")
        zero_cleanup(row, retained=True)
        unsigned(row["aggregate_logical_bytes"], 1, 256 * 1024 * 1024)
        cap = unsigned(row["maximum_physical_bytes"], 1, 512 * 1024 * 1024)
        unsigned(row["settled_physical_charge"], 1, cap)
        unsigned(row["peak_admitted_gpu_bytes"], row["settled_physical_charge"], cap)
        unsigned(row["chunk_count"], 1, 64)
        unsigned(row["buffer_count"], 4, 32)
        if any(row[k] != workloads[0][k] for k in keys):
            raise ValueError("Inconsistent workload configuration")
        if counter(row["timing"]["elapsed"]) is None:
            raise ValueError("Missing elapsed sample")
    measured = workloads[warmups:]
    first = measured[0]
    samples = {name: [] for name in ["workload", "allocation", "raw_acquire", "restore_acquire", "policy_call_average", "compression_call_average", "decompression_call_average", "transition_call_average", "reclaim_cycle", "nonbeneficial_reclaim_cycle"]}

    def add(name, item):
        value = counter(item)
        if value is not None:
            samples[name].append(value)

    for r in measured:
        add("workload", r["timing"]["elapsed"])
        add("decompression_call_average", r["timing"]["decompression"])
        compressed = {c["buffer"] for c in r["chunks"] if c["representation"] == "GPU_COMPRESSED"}
        for buffer in r["timing"]["buffers"]:
            add("allocation", buffer["allocation"])
            add("restore_acquire" if buffer["index"] in compressed else "raw_acquire", buffer["acquire_restore"])
        for c in r["cycles"]:
            add("reclaim_cycle", c["elapsed"])
            add("policy_call_average", c["policy_selection"])
            add("transition_call_average", c["policy_transition"])
            add("compression_call_average", c["compression"])
            if c["rejected"]:
                add("nonbeneficial_reclaim_cycle", c["elapsed"])
    elapsed = statistics.median(samples["workload"])
    return {"schema": 1, "source_sha256": rows[0]["source_sha256"],
            **{key: first[key] for key in keys}, "warmups": end["warmups"],
            "iterations": end["iterations"], "clock": "host_steady_clock",
            "codec_scope": "metadata publication through launch, synchronize and result; includes host overhead",
            "sample_scope": "call_average metrics summarize per-cycle or per-workload means, not individual GPU kernel latency",
            "statistics": {name: summary(values) for name, values in samples.items()},
            "verified_logical_bytes_per_second": first["aggregate_logical_bytes"] * 1e9 / elapsed if elapsed else None,
            "settled_physical_bytes": [r["settled_physical_charge"] for r in measured],
            "logical_physical_ratios": [r["aggregate_logical_bytes"] / r["settled_physical_charge"] for r in measured],
            "peak_explicit_bytes": max(r["peak_admitted_gpu_bytes"] for r in measured),
            "successful_policy_transitions": sum(c["successes"] for r in measured for c in r["cycles"]),
            "nonbeneficial_attempts": sum(c["rejected"] for r in measured for c in r["cycles"]),
            "final_cleanup": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        results = []
        for path in args.inputs:
            with path.open() as stream:
                text = stream.read(16 * 1024 * 1024 + 1)
            results.append({"input": str(path), **analyze(text)})
        print(json.dumps(results, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print("Benchmark rejected: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
