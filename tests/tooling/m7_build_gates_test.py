#!/usr/bin/env python3
"""Bounded M7 configuration/ELF tests using inert synthetic ELF files, never a driver."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("m7_elf_check", SOURCE / "tools/ci/m7-elf-check.py")
ELF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ELF)


class M7BuildGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="vramz-m7-build-gates-")
        cls.root = Path(cls.temporary.name)
        cls.toolkit = cls.root / "toolkit"
        cls.toolkit.mkdir()
        cls.driver = cls.root / "synthetic-libcuda.so.1"
        cls.fixture = cls.root / "inert_symbols.cpp"
        symbols = ("cuInit", "cuDriverGetVersion", "cuMemCreate", "cuMemMap", "cuMemUnmap",
                   "cuMemSetAccess", "cuMemGetAllocationPropertiesFromHandle")
        cls.fixture.write_text("\n".join('extern "C" void ' + symbol + "() {}"
                                         for symbol in symbols), encoding="utf-8")
        subprocess.run([os.environ.get("CXX", "g++"), "-shared", "-fPIC", "-Wl,-soname,libcuda.so.1",
                        str(cls.fixture), "-o", str(cls.driver)], check=True,
                       capture_output=True, timeout=30)
        cls.cmake_source = cls.root / "cmake-source"
        cls.cmake_source.mkdir()
        cls.cmake_source.joinpath("CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.25)\nproject(M7GateFixture NONE)\n"
            f'include("{SOURCE / "cmake/M7RawSmoke.cmake"}")\n'
            "vramz_validate_m7_configuration()\n"
            'message(STATUS "M7_FIXTURE_MTIME_NS=${VRAMZ_M7_DRIVER_mtime_ns}")\n',
            encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def configure(self, expected_success, **overrides):
        values = {
            "VRAMZ_ENABLE_CUDA": "ON", "VRAMZ_ENABLE_NVCOMP": "OFF",
            "VRAMZ_ALLOW_REAL_GPU_EXECUTION": "ON", "VRAMZ_BUILD_M7_RAW_SMOKE": "ON",
            "CUDAToolkit_ROOT": str(self.toolkit),
            "VRAMZ_REAL_CUDA_DRIVER_LIBRARY": str(self.driver),
            "VRAMZ_M7_SOURCE_SHA256": "a" * 64,
        }
        values.update(overrides)
        result = subprocess.run(["cmake", "-S", str(self.cmake_source),
                                 "-B", str(self.root / self.id().split(".")[-1]),
                                 *[f"-D{key}={value}" for key, value in values.items()]],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode == 0, expected_success,
                         result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_default_remains_off(self):
        self.configure(True, VRAMZ_ALLOW_REAL_GPU_EXECUTION="OFF", VRAMZ_BUILD_M7_RAW_SMOKE="OFF",
                       VRAMZ_REAL_CUDA_DRIVER_LIBRARY="", VRAMZ_M7_SOURCE_SHA256="")

    def test_both_explicit_execution_gates_required(self):
        self.assertIn("requires explicit", self.configure(False, VRAMZ_ALLOW_REAL_GPU_EXECUTION="OFF"))

    def test_execution_without_raw_target_rejected(self):
        self.assertIn("real GPU execution is unavailable",
                      self.configure(False, VRAMZ_BUILD_M7_RAW_SMOKE="OFF"))

    def test_cuda_required(self):
        self.assertIn("requires VRAMZ_ENABLE_CUDA=ON", self.configure(False, VRAMZ_ENABLE_CUDA="OFF"))

    def test_nvcomp_forbidden(self):
        self.assertIn("requires VRAMZ_ENABLE_NVCOMP=OFF", self.configure(False, VRAMZ_ENABLE_NVCOMP="ON"))

    def test_archive_identity_required(self):
        self.assertIn("VRAMZ_M7_SOURCE_SHA256", self.configure(False, VRAMZ_M7_SOURCE_SHA256=""))

    def test_archive_identity_malformed_rejected(self):
        self.assertIn("VRAMZ_M7_SOURCE_SHA256", self.configure(False, VRAMZ_M7_SOURCE_SHA256="A" * 64))

    def test_compile_only_archive_identity_rejected(self):
        self.assertIn("VRAMZ_M7_SOURCE_SHA256", self.configure(False, VRAMZ_M7_SOURCE_SHA256="0" * 64))

    def test_explicit_absolute_driver_required(self):
        self.assertIn("explicit existing absolute", self.configure(False, VRAMZ_REAL_CUDA_DRIVER_LIBRARY="libcuda.so.1"))

    def test_nonexistent_driver_rejected(self):
        self.assertIn("explicit existing absolute", self.configure(False, VRAMZ_REAL_CUDA_DRIVER_LIBRARY=str(self.root / "absent.so")))

    def test_structural_fixture_accepted_without_execution(self):
        output = self.configure(True)
        self.assertIn("validated statically", output)
        self.assertIn(f"M7_FIXTURE_MTIME_NS={self.driver.stat().st_mtime_ns}", output)

    def test_stub_path_rejected(self):
        stub = self.root / "stubs/libcuda.so"
        stub.parent.mkdir(exist_ok=True)
        shutil.copyfile(self.driver, stub)
        self.assertIn("stub path is forbidden", self.configure(False, VRAMZ_REAL_CUDA_DRIVER_LIBRARY=str(stub)))

    def test_symlink_to_stub_rejected(self):
        stub = self.root / "symlink-fixture/stubs/libcuda.so"
        stub.parent.mkdir(parents=True)
        shutil.copyfile(self.driver, stub)
        link = self.root / "looks-like-driver.so.1"
        link.symlink_to(stub)
        self.assertIn("stub path is forbidden", self.configure(False, VRAMZ_REAL_CUDA_DRIVER_LIBRARY=str(link)))

    def test_toolkit_library_rejected(self):
        candidate = self.toolkit / "development-libcuda.so.1"
        shutil.copyfile(self.driver, candidate)
        with self.assertRaisesRegex(ValueError, "outside the isolated CUDA"):
            ELF.inspect_driver(candidate, self.toolkit)

    def test_copied_stub_rejected(self):
        toolkit = self.root / "copied-stub-toolkit"
        stub = toolkit / "lib/stubs/libcuda.so"
        stub.parent.mkdir(parents=True)
        shutil.copyfile(self.driver, stub)
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            ELF.inspect_driver(self.driver, toolkit)

    def test_non_elf_rejected(self):
        candidate = self.root / "text-libcuda.so.1"
        candidate.write_text("not a driver\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not an ELF"):
            ELF.inspect_driver(candidate, self.toolkit)

    def test_wrong_soname_rejected(self):
        candidate = self.root / "wrong-soname.so.1"
        subprocess.run([os.environ.get("CXX", "g++"), "-shared", "-fPIC", "-Wl,-soname,libwrong.so.1",
                        str(self.fixture), "-o", str(candidate)], check=True,
                       capture_output=True, timeout=30)
        with self.assertRaisesRegex(ValueError, "SONAME"):
            ELF.inspect_driver(candidate, self.toolkit)

    def test_binary_rpath_is_rejected(self):
        dynamic = "(NEEDED) Shared library: [libcuda.so.1]\n(RUNPATH) Library runpath: [/tmp/stubs/]"
        with mock.patch.object(ELF, "readelf", return_value=dynamic):
            with self.assertRaisesRegex(ValueError, "RPATH/RUNPATH"):
                ELF.inspect_binary(self.driver)

    def test_binary_nvcomp_is_rejected(self):
        dynamic = "(NEEDED) Shared library: [libcuda.so.1]\n(NEEDED) Shared library: [libnvcomp.so.5]"
        with mock.patch.object(ELF, "readelf", return_value=dynamic):
            with self.assertRaisesRegex(ValueError, "nvCOMP or CUDA Runtime"):
                ELF.inspect_binary(self.driver)

    def test_binary_cudart_is_rejected(self):
        dynamic = "(NEEDED) Shared library: [libcuda.so.1]\n(NEEDED) Shared library: [libcudart.so.13]"
        with mock.patch.object(ELF, "readelf", return_value=dynamic):
            with self.assertRaisesRegex(ValueError, "nvCOMP or CUDA Runtime"):
                ELF.inspect_binary(self.driver)

    def test_binary_driver_dependency_required(self):
        with mock.patch.object(ELF, "readelf", return_value="(NEEDED) Shared library: [libc.so.6]"):
            with self.assertRaisesRegex(ValueError, "exactly one libcuda"):
                ELF.inspect_binary(self.driver)

    def test_binary_absolute_dependency_rejected(self):
        dynamic = "(NEEDED) Shared library: [libcuda.so.1]\n(NEEDED) Shared library: [/tmp/library.so]"
        with mock.patch.object(ELF, "readelf", return_value=dynamic):
            with self.assertRaisesRegex(ValueError, "absolute dependency"):
                ELF.inspect_binary(self.driver)


if __name__ == "__main__":
    unittest.main()
