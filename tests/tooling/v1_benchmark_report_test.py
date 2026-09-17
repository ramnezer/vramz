#!/usr/bin/env python3
"""CPU-only validation of measured-data parsing and summary boundaries."""
import importlib.util
import copy
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("benchmark_report", ROOT / "tools/vramz/benchmark_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class BenchmarkReportTests(unittest.TestCase):
    def test_nearest_rank_and_raw_samples(self):
        result = REPORT.summary([10, 30, 20])
        self.assertEqual(result["median_ns"], 20)
        self.assertEqual(result["p95_ns_nearest_rank"], 30)
        self.assertEqual(result["samples_ns"], [10, 30, 20])
        self.assertIsNone(REPORT.summary([]))

    def test_counter_is_not_a_fabricated_kernel_sample(self):
        self.assertEqual(REPORT.counter({"samples": 2, "nanoseconds": 100, "overflow": False}), 50)
        self.assertIsNone(REPORT.counter({"samples": 0, "nanoseconds": 0, "overflow": False}))
        for value in [{"samples": 1, "nanoseconds": 10, "overflow": True},
                      {"samples": -1, "nanoseconds": 10, "overflow": False},
                      {"samples": 1, "nanoseconds": float("nan"), "overflow": False}]:
            with self.assertRaises(ValueError):
                REPORT.counter(value)

    def test_failure_and_partial_series_rejected(self):
        for value in [[], [{"kind": "series_summary", "result": "FAIL"}],
                      [{"kind": "series_summary", "result": "PASS", "cleanup": False, "final_counts_known": False}],
                      [{"kind": "series_summary", "result": "PASS", "cleanup": True, "final_counts_known": True, "final_backend_resources": 1}]]:
            with self.assertRaises(ValueError):
                REPORT.analyze("\n".join(json.dumps(row) for row in value))

    @staticmethod
    def valid_series():
        zero = {"final_" + key: 0 for key in
                ("backend_resources", "raw_resources", "compressed_resources", "va_reservations",
                 "budget_charge", "unmaterialized_reservations", "workspace")}
        timing = {"samples": 1, "nanoseconds": 100, "overflow": False}
        workload = {"v1_workload_version": 1, "result": "PASS", "integrity": True,
                    "cleanup": True, "final_counts_known": True, "final_context": True,
                    "final_stream": True, **zero, "measurement_enabled": True,
                    "snapshot_proven": True, "iteration": 0, "warmup": False,
                    "runtime_retained": True, "raw_baseline": False, "profile": 0,
                    "aggregate_logical_bytes": 67108864, "chunk_count": 8, "buffer_count": 8,
                    "settled_physical_charge": 29360128, "peak_admitted_gpu_bytes": 90177536,
                    "maximum_physical_bytes": 134217728, "cycles": [], "chunks": [],
                    "timing": {"elapsed": timing, "decompression": timing, "buffers": []}}
        end = {"kind": "series_summary", "result": "PASS", "cleanup": True,
               "final_counts_known": True, "iterations": 1, "warmups": 0,
               "final_context": False, "final_stream": False, "final_owned_charge": 0, **zero}
        return [{"source_sha256": "a" * 64}, workload, end]

    def test_complete_series(self):
        rows = self.valid_series()
        result = REPORT.analyze("\n".join(map(json.dumps, rows)))
        self.assertTrue(result["final_cleanup"])
        self.assertEqual(result["settled_physical_bytes"], [29360128])
        self.assertEqual(result["statistics"]["workload"]["samples_ns"], [100])

    def test_forged_flags_missing_accounting_and_invalid_bounds(self):
        changes = [(2, "cleanup", "true"), (2, "final_counts_known", 1),
                   (2, "final_context", 0), (2, "iterations", 0), (2, "iterations", 129),
                   (2, "warmups", 9), (2, "final_workspace", -1),
                   (1, "integrity", "true"), (1, "warmup", 0), (1, "iteration", 1),
                   (1, "settled_physical_charge", 0), (1, "peak_admitted_gpu_bytes", 536870913),
                   (1, "runtime_retained", False), (0, "source_sha256", "unfrozen")]
        for row, key, value in changes:
            with self.subTest(row=row, key=key, value=value):
                rows = copy.deepcopy(self.valid_series())
                rows[row][key] = value
                with self.assertRaises(ValueError):
                    REPORT.analyze("\n".join(map(json.dumps, rows)))
        rows = self.valid_series()
        del rows[2]["final_budget_charge"]
        with self.assertRaises(KeyError):
            REPORT.analyze("\n".join(map(json.dumps, rows)))

    def test_timing_type_and_overflow_boundaries(self):
        for item in [{"samples": True, "nanoseconds": 1, "overflow": False},
                     {"samples": 0, "nanoseconds": 1, "overflow": False},
                     {"samples": 1, "nanoseconds": 1 << 64, "overflow": False},
                     {"samples": 1, "nanoseconds": 1, "overflow": 0}]:
            with self.assertRaises(ValueError):
                REPORT.counter(item)


if __name__ == "__main__":
    unittest.main()
