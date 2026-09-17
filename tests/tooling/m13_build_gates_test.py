#!/usr/bin/env python3
"""M13 configuration and static ELF rejection tests; no inspected executable is run."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from unittest import mock

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[2]


class Gates(unittest.TestCase):
    def configure(self, values):
        with tempfile.TemporaryDirectory(prefix="vramz-m13-gates-") as name:
            root = Path(name)
            (root / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(M13Gates NONE)\n'
                f'include("{SOURCE}/cmake/M7RawSmoke.cmake")\n'
                f'include("{SOURCE}/cmake/M8CompressionSmoke.cmake")\n'
                f'include("{SOURCE}/cmake/M13ControlledCapacity2xSmoke.cmake")\n'
                'vramz_add_m13_smoke()\n'
                'if(TARGET vramz-m13-controlled-capacity-2x-smoke)\n'
                'message(FATAL_ERROR "unexpected physical target")\nendif()\n')
            result = subprocess.run(["cmake", "-S", name, "-B", str(root / "build"),
                *[f"-D{k}={v}" for k, v in values.items()]],
                capture_output=True, text=True, timeout=30)
            return result.returncode, result.stdout + result.stderr

    def test_default_has_no_physical_target(self):
        self.assertEqual(self.configure({})[0], 0)

    def test_each_gate_is_required(self):
        for key in ("VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION", "VRAMZ_ALLOW_REAL_GPU_EXECUTION",
                    "VRAMZ_ENABLE_CUDA", "VRAMZ_ENABLE_NVCOMP", "VRAMZ_ENABLE_CPU_LZ4"):
            with self.subTest(key=key):
                status, text = self.configure({**self.physical_values(), key: "OFF",
                                              "VRAMZ_M13_SOURCE_SHA256": "a" * 64})
                self.assertNotEqual(status, 0)
                self.assertIn("both execution gates ON", text)
        self.assertNotEqual(self.configure({"VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION": "ON"})[0], 0)

    def test_m7_and_m13_are_exclusive(self):
        status, text = self.configure({"VRAMZ_BUILD_M7_RAW_SMOKE": "ON", "VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE": "ON"})
        self.assertNotEqual(status, 0)
        self.assertIn("mutually exclusive", text)

    def physical_values(self):
        return {**{key: "ON" for key in ("VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE", "VRAMZ_ENABLE_CUDA",
            "VRAMZ_ENABLE_NVCOMP", "VRAMZ_ENABLE_CPU_LZ4", "VRAMZ_ALLOW_REAL_GPU_EXECUTION",
            "VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION")}, "VRAMZ_SANITIZER": "none", "VRAMZ_BUILD_TESTS": "OFF"}

    def test_tests_and_sanitizers_cannot_enable_execution(self):
        for key, value in (("VRAMZ_BUILD_TESTS", "ON"), ("VRAMZ_ENABLE_FUZZING", "ON"), ("VRAMZ_SANITIZER", "address"), ("BUILD_SHARED_LIBS", "ON")):
            with self.subTest(key=key):
                status, text = self.configure({**self.physical_values(), key: value})
                self.assertNotEqual(status, 0)
                self.assertIn("tests/fuzzing/sanitizers OFF", text)

    def test_m7_obsolete_or_invalid_hash_never_identifies_m13(self):
        for digest in ("", "a" * 63, "0" * 64, "A" * 64,
                       "2596306a767e9f2c5796cc7543dbbe0bf5f994de073f0135ee32ce16a11e2073",
                       "56b3a392464febf9b5ec8f49653b68a2de5bd50c2bd814a1d38ac7b133334c76",
                       "ed54613a38f4532f4f12b80020bcec191dc11045225541995c0cc2132e2d4b94",
                       "929433a01d724d7c8d8bf9ac7e8e251c5718a17f844f02f9da434ef84fdc698d",
                       "dd63e36734e1ef1bb3a69055223b45e785e47f8b288723beb3b1132598dbdab2",
                       "40748e087d303d06cd1a8dc41676bdf0100bced11e924797d2d4fed7e2459407",
                       "c67dc98206470c74045aa0c7ed1ae69ae596b2f7687a966cab6c6b7e3dec93bd"):
            with self.subTest(digest=digest):
                status, text = self.configure({**self.physical_values(), "VRAMZ_M13_SOURCE_SHA256": digest})
                self.assertNotEqual(status, 0)
                self.assertIn("own reviewed source", text)

    def test_missing_reviewed_lz4_prefix_is_rejected(self):
        status, text = self.configure({**self.physical_values(),
            "VRAMZ_M13_SOURCE_SHA256": "a" * 64})
        self.assertNotEqual(status, 0)
        self.assertIn("VRAMZ_REVIEWED_LZ4_PREFIX", text)

    def test_unreviewed_or_shared_lz4_is_rejected_before_dependency_setup(self):
        for library in ("", "/usr/lib/x86_64-linux-gnu/liblz4.so", "/tmp/liblz4.a"):
            with self.subTest(library=library):
                status, text = self.configure({**self.physical_values(),
                    "VRAMZ_M13_SOURCE_SHA256": "a" * 64, "VRAMZ_LZ4_LIBRARY": library,
                    "VRAMZ_LZ4_INCLUDE_DIR": "/usr/include", "VRAMZ_LZ4_VERSION": "1.9.4",
                    "VRAMZ_REVIEWED_LZ4_PREFIX": "/test-fixtures/reviewed-lz4"})
                self.assertNotEqual(status, 0)
                self.assertIn("reviewed isolated static LZ4", text)

    def test_gate_excludes_m8(self):
        status, text = self.configure({"VRAMZ_BUILD_M8_COMPRESSION_SMOKE": "ON", "VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE": "ON"})
        self.assertNotEqual(status, 0)
        self.assertIn("mutually exclusive", text)

    def test_gate_excludes_m9(self):
        status, text = self.configure({"VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE": "ON", "VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE": "ON"})
        self.assertNotEqual(status, 0)
        self.assertIn("mutually exclusive", text)

    def test_gate_excludes_m10(self):
        status, text = self.configure({"VRAMZ_BUILD_M10_MULTI_CHUNK_RESIDENCY_SMOKE": "ON", "VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE": "ON"})
        self.assertNotEqual(status, 0)
        self.assertIn("mutually exclusive", text)

    def test_gate_excludes_m11(self):
        status, text = self.configure({"VRAMZ_BUILD_M11_POLICY_PRESSURE_SMOKE": "ON", "VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE": "ON"})
        self.assertNotEqual(status, 0)
        self.assertIn("mutually exclusive", text)

    def test_gate_excludes_m12(self):
        status, text = self.configure({"VRAMZ_BUILD_M12_CONTROLLED_CAPACITY_SMOKE": "ON", "VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE": "ON"})
        self.assertNotEqual(status, 0)
        self.assertIn("mutually exclusive", text)

    def test_physical_main_is_object_only_in_quality_builds(self):
        module = (SOURCE / "cmake/M13ControlledCapacity2xSmoke.cmake").read_text()
        fallback = module.split("elseif(VRAMZ_ENABLE_CUDA AND VRAMZ_ENABLE_NVCOMP)")[1]
        self.assertIn("add_library(${target} OBJECT", fallback)
        self.assertNotIn("add_executable", fallback)
        self.assertNotIn("POST_BUILD", module)
        self.assertNotIn("add_test", module)
        self.assertNotIn("install(", module)
        self.assertIn("--binary \"$<TARGET_FILE:${target}>\"", module)
        self.assertIn("m13-elf-check.py", module)
        main = (SOURCE / "tools/vramz-m13-controlled-capacity-2x-smoke/main.cpp").read_text()
        self.assertIn("VRAMZ_M13_CONTROLLED_CAPACITY_2X_ONLY", main)
        self.assertIn("constexpr bool enabled = false", main)

    def test_harness_uses_production_policy_without_victim_or_transition_override(self):
        harness = (SOURCE / "src/core/controlled_capacity_2x_smoke.cpp").read_text()
        self.assertIn("RuntimeAccess::reclaim(runtime_, m13_settled_target, false, this)", harness)
        for forbidden in ("testing::migrate(", "coordinator.migrate(", "backend_.transfer(", "backend_.prepare_transfer(", "advance_policy_epoch(", "advance_access_revision(", "policy::propose(", "policy::preferred("):
            self.assertNotIn(forbidden, harness)

    def test_physical_main_has_no_test_observer_or_runtime_selection_override(self):
        main = (SOURCE / "tools/vramz-m13-controlled-capacity-2x-smoke/main.cpp").read_text()
        import re
        calls = re.findall(r"run_controlled_capacity_2x_smoke\((.*?)\);", main, re.S)
        self.assertEqual(len(calls), 1)
        self.assertRegex(calls[0], r"options,\s*report$")
        self.assertNotIn("Observer", main)
        runtime = (SOURCE / "src/core/runtime.cpp").read_text()
        self.assertIn("constraints.expected_access_revision = selected.snapshot.access_revision", runtime)

    def test_prehardware_explicitly_rejects_m13(self):
        script = (SOURCE / "tools/vramz-prehardware-check.py").read_text()
        self.assertIn("VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE", script)
        self.assertIn("vramz-m13-controlled-capacity-2x-smoke", script)


class ProviderProvenance(unittest.TestCase):
    @staticmethod
    def load(name):
        spec = importlib.util.spec_from_file_location(name, SOURCE / f"tools/ci/{name}-elf-check.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_historical_and_active_profiles_are_distinct(self):
        old, current = self.load("m8"), self.load("m13")
        self.assertEqual(old.DRIVER_RELEASE, (595, 84, 0))
        self.assertEqual(current.DRIVER_RELEASE, (595, 91, 7))
        self.assertEqual(old.PINNED["libcuda.so.1"], (
            "/usr/lib/x86_64-linux-gnu/libcuda.so.595.84",
            "c147185c80a0270d635db626537fc9223b2ad1764f86cb12fd116d26e08b175d"))
        self.assertEqual(current.PINNED["libcuda.so.1"], (
            "/usr/lib/x86_64-linux-gnu/libcuda.so.595.91.07",
            "4839b5da17cd8f58a8e9c57e97c9b61f3c41be7934479f48afc983416b2492c3"))
        for soname in ("libnvcomp.so.5", "libcudart.so.13"):
            self.assertEqual(old.PINNED[soname], current.PINNED[soname])
        self.assertEqual(old.PRIOR_M7_DRIVER_API, current.PRIOR_M7_DRIVER_API)
        self.assertEqual(current.PRIOR_M7_DRIVER_API, 13020)

    def test_new_kernel_profile_rejects_temporary_mismatch(self):
        elf = self.load("m13")
        with mock.patch.object(Path, "open", mock.mock_open(read_data="595.91.07\n")):
            self.assertEqual(elf.installed_driver_release(), (595, 91, 7))
        for text in ("595.84\n", "595.91.08\n", "579.99\n", "9" * 65):
            with self.subTest(text=text), mock.patch.object(Path, "open", mock.mock_open(read_data=text)):
                with self.assertRaises(ValueError):
                    elf.installed_driver_release()

    def test_m13_identity_header_does_not_overwrite_historical_namespace(self):
        elf = self.load("m13")
        item = {"path": "/verified/library.so", "sha256": "a" * 64, "identity": [1] * 5}
        report = {"libraries": {name: item for name in elf.PINNED},
                  "loader_path": "/verified", "nvidia_driver_release": [595, 91, 7]}
        text = elf.identity_config(report)
        self.assertIn("set(VRAMZ_M13_DRIVER_RELEASE [=[595U, 91U, 7U]=])", text)
        self.assertIn("set(VRAMZ_M13_PRIOR_DRIVER_API 13020)", text)
        self.assertNotIn("VRAMZ_M8_", text)
        self.assertNotIn("13030", text)

    def test_m13_closure_still_rejects_unsafe_loader_entries(self):
        elf = self.load("m13")
        for tag in ("RPATH", "RUNPATH"):
            with mock.patch.object(elf.m7, "readelf", return_value=f"({tag}) Library: [/tmp/lib]"):
                with self.assertRaisesRegex(ValueError, "RPATH/RUNPATH"):
                    elf.inspect(Path(sys.executable))
        with mock.patch.object(elf.m7, "readelf", return_value="(NEEDED) Shared library: [/tmp/libcuda.so]"):
            with self.assertRaisesRegex(ValueError, "path in NEEDED"):
                elf.inspect(Path(sys.executable))
        for soname in ("libnvcomp.so.5", "libcudart.so.13", "libcuda.so.1", "libnvidia-ptxjitcompiler.so.1"):
            self.assertNotIn(soname, elf.SYSTEM)

if __name__ == "__main__":
    unittest.main()
