#!/usr/bin/env python3
"""M8 configuration and static ELF rejection tests; no inspected executable is run."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("m8_elf", SOURCE / "tools/ci/m8-elf-check.py")
ELF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ELF)


class Gates(unittest.TestCase):
    def configure(self, values):
        with tempfile.TemporaryDirectory(prefix="vramz-m8-gates-") as name:
            root = Path(name)
            (root / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(M8Gates NONE)\n'
                f'include("{SOURCE}/cmake/M7RawSmoke.cmake")\n'
                f'include("{SOURCE}/cmake/M8CompressionSmoke.cmake")\n'
                'vramz_add_m8_smoke()\n')
            result = subprocess.run(["cmake", "-S", name, "-B", str(root / "build"),
                *[f"-D{k}={v}" for k, v in values.items()]],
                capture_output=True, text=True, timeout=30)
            return result.returncode, result.stdout + result.stderr

    def test_default_has_no_physical_target(self):
        self.assertEqual(self.configure({})[0], 0)

    def test_each_gate_is_required(self):
        for values in ({"VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION": "ON"},
                       {"VRAMZ_BUILD_M8_COMPRESSION_SMOKE": "ON"},
                       {"VRAMZ_BUILD_M8_COMPRESSION_SMOKE": "ON", "VRAMZ_ALLOW_REAL_GPU_EXECUTION": "ON"}):
            with self.subTest(values=values):
                self.assertNotEqual(self.configure(values)[0], 0)

    def test_m7_and_m8_are_exclusive(self):
        status, text = self.configure({"VRAMZ_BUILD_M7_RAW_SMOKE": "ON", "VRAMZ_BUILD_M8_COMPRESSION_SMOKE": "ON"})
        self.assertNotEqual(status, 0)
        self.assertIn("mutually exclusive", text)

    def physical_values(self):
        return {**{key: "ON" for key in ("VRAMZ_BUILD_M8_COMPRESSION_SMOKE", "VRAMZ_ENABLE_CUDA",
            "VRAMZ_ENABLE_NVCOMP", "VRAMZ_ENABLE_CPU_LZ4", "VRAMZ_ALLOW_REAL_GPU_EXECUTION",
            "VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION")}, "VRAMZ_SANITIZER": "none", "VRAMZ_BUILD_TESTS": "OFF"}

    def test_tests_and_sanitizers_cannot_enable_execution(self):
        for key, value in (("VRAMZ_BUILD_TESTS", "ON"), ("VRAMZ_ENABLE_FUZZING", "ON"), ("VRAMZ_SANITIZER", "address")):
            with self.subTest(key=key):
                status, text = self.configure({**self.physical_values(), key: value})
                self.assertNotEqual(status, 0)
                self.assertIn("tests/fuzzing/sanitizers OFF", text)

    def test_m7_obsolete_or_invalid_hash_never_identifies_m8(self):
        for digest in ("", "a" * 63, "0" * 64, "A" * 64,
                       "56b3a392464febf9b5ec8f49653b68a2de5bd50c2bd814a1d38ac7b133334c76",
                       "c67dc98206470c74045aa0c7ed1ae69ae596b2f7687a966cab6c6b7e3dec93bd"):
            with self.subTest(digest=digest):
                status, text = self.configure({**self.physical_values(), "VRAMZ_M8_SOURCE_SHA256": digest})
                self.assertNotEqual(status, 0)
                self.assertIn("own reviewed source", text)

    def test_runtime_loader_paths_rejected(self):
        for tag in ("RPATH", "RUNPATH"):
            with mock.patch.object(ELF.m7, "readelf", return_value=f"({tag}) Library: [/tmp/lib]"):
                with self.assertRaisesRegex(ValueError, "RPATH/RUNPATH"):
                    ELF.inspect(Path(sys.executable))

    def test_absolute_needed_is_rejected(self):
        with mock.patch.object(ELF.m7, "readelf", return_value="(NEEDED) Shared library: [/tmp/libcuda.so]"):
            with self.assertRaisesRegex(ValueError, "path in NEEDED"):
                ELF.inspect(Path(sys.executable))

    def test_identity_path_injection_is_rejected(self):
        item = {"path": "/tmp/bad\npath", "sha256": "a" * 64, "identity": [0] * 5}
        with self.assertRaisesRegex(ValueError, "syntax"):
            ELF.identity_config({"libraries": {"libcuda.so.1": item}})

    def test_system_nvcomp_is_not_a_fallback(self):
        self.assertNotIn("libnvcomp.so.5", ELF.SYSTEM)
        self.assertNotIn("libcudart.so.13", ELF.SYSTEM)
        self.assertNotIn("libcuda.so.1", ELF.SYSTEM)
        self.assertNotIn("libnvidia-ptxjitcompiler.so.1", ELF.SYSTEM)

    def test_cuda13_driver_release_floor(self):
        for text in ("580.0", "580.65.06\n", "595.84\n"):
            with self.subTest(text=text):
                self.assertGreaterEqual(ELF.driver_release(text)[0], 580)
        with self.assertRaisesRegex(ValueError, "minimum branch 580"):
            ELF.driver_release("579.999.999")

    def test_malformed_or_overflowed_release_is_rejected(self):
        for text in ("", "595", "595.84.extra", "595.84\n\n", " 595.84", "595.84.",
                     "595.84.0.1", "4294967296.84", "595.4294967296"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                ELF.driver_release(text)

    def test_kernel_release_is_pinned_without_cuda_query(self):
        with mock.patch.object(Path, "open", mock.mock_open(read_data="595.84\n")):
            self.assertEqual(ELF.installed_driver_release(), (595, 84, 0))
        for text in ("580.65.6\n", "595.85\n", "579.99\n", "9" * 65):
            with self.subTest(text=text), mock.patch.object(Path, "open", mock.mock_open(read_data=text)):
                with self.assertRaises(ValueError):
                    ELF.installed_driver_release()
        with mock.patch.object(Path, "open", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                ELF.installed_driver_release()

    def test_identity_config_keeps_prior_observation_separate_from_runtime(self):
        item = {"path": "/verified/library.so", "sha256": "a" * 64, "identity": [1] * 5}
        report = {"libraries": {name: item for name in ELF.PINNED},
                  "loader_path": "/verified", "nvidia_driver_release": [595, 84, 0]}
        output = ELF.identity_config(report)
        self.assertIn("set(VRAMZ_M8_DRIVER_RELEASE [=[595U, 84U, 0U]=])", output)
        self.assertIn("set(VRAMZ_M8_PRIOR_DRIVER_API 13020)", output)
        self.assertNotIn("13030", output)


if __name__ == "__main__":
    unittest.main()
