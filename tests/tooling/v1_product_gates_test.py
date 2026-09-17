#!/usr/bin/env python3
"""CPU-only product execution barriers, loader isolation and permission checks."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("v1_launcher", SOURCE / "tools/vramz/launcher.py")
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


class BuildGates(unittest.TestCase):
    def configure(self, values):
        with tempfile.TemporaryDirectory(prefix="vramz-v1-gates-") as name:
            root = Path(name)
            (root / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(V1Gates NONE)\n'
                f'include("{SOURCE}/cmake/PhysicalRuntime.cmake")\n'
                'vramz_add_physical_runtime()\n'
                'if(TARGET vramz-run OR TARGET vramz_cuda)\n'
                'message(FATAL_ERROR "unexpected physical target")\nendif()\n')
            result = subprocess.run(["cmake", "-S", name, "-B", str(root / "build"),
                *[f"-D{k}={v}" for k, v in values.items()]],
                capture_output=True, text=True, timeout=30)
            return result.returncode, result.stdout + result.stderr

    @staticmethod
    def enabled():
        return {key: "ON" for key in ("VRAMZ_ENABLE_PHYSICAL_RUNTIME", "VRAMZ_ENABLE_CUDA",
            "VRAMZ_ENABLE_NVCOMP", "VRAMZ_ENABLE_CPU_LZ4", "VRAMZ_ALLOW_REAL_GPU_EXECUTION",
            "VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION")}

    def test_default_has_no_cuda_product(self):
        self.assertEqual(self.configure({})[0], 0)

    def test_independent_required_build_permissions(self):
        for key in self.enabled():
            if key == "VRAMZ_ENABLE_PHYSICAL_RUNTIME":
                continue
            with self.subTest(key=key):
                status, output = self.configure({**self.enabled(), key: "OFF"})
                self.assertNotEqual(status, 0)
                self.assertIn("both execution build gates", " ".join(output.split()))

    def test_historical_smokes_are_excluded(self):
        for gate in ("M7_RAW", "M8_COMPRESSION", "M9_PHYSICAL_SAVINGS", "M10_MULTI_CHUNK_RESIDENCY",
                     "M11_POLICY_PRESSURE", "M12_CONTROLLED_CAPACITY", "M13_CONTROLLED_CAPACITY_2X"):
            with self.subTest(gate=gate):
                status, output = self.configure({**self.enabled(), f"VRAMZ_BUILD_{gate}_SMOKE": "ON"})
                self.assertNotEqual(status, 0)
                self.assertIn("must not enable historical smoke targets", output)

    def test_dynamic_lz4_rejected_before_provider_setup(self):
        status, output = self.configure({**self.enabled(), "VRAMZ_LZ4_LIBRARY": "/tmp/liblz4.so"})
        self.assertNotEqual(status, 0)
        self.assertIn("reviewed static LZ4 archive", output)

    def test_wrong_static_lz4_identity_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="vramz-wrong-lz4-") as name:
            root = Path(name)
            (root / "liblz4.a").write_bytes(b"unreviewed archive")
            (root / "lz4.h").write_text("unreviewed header")
            status, output = self.configure({**self.enabled(), "VRAMZ_LZ4_LIBRARY": str(root / "liblz4.a"),
                                            "VRAMZ_LZ4_INCLUDE_DIR": str(root)})
            self.assertNotEqual(status, 0)
            self.assertIn("differs from reviewed providers", output)


class Launcher(unittest.TestCase):
    def setUp(self):
        # A private fixture stays under a trusted source parent, never /tmp's shared namespace.
        self.temporary = tempfile.TemporaryDirectory(prefix=".v1-loader-test-", dir=SOURCE.parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        support = self.root / "libexec/vramz"
        support.mkdir(parents=True)
        for name in ("vramz-run", "m13-elf-check.py", "m7-elf-check.py"):
            (support / name).write_text("CPU-only fixture, never executed")
            (support / name).chmod(0o644)
        libraries = self.root / "libraries"
        libraries.mkdir()
        self.report = {"loader_path": str(libraries), "libraries": {}}
        for name in ("libcuda.so.1", "libnvcomp.so.5", "libcudart.so.13"):
            path = libraries / name
            path.write_text("CPU-only provider identity fixture")
            path.chmod(0o644)
            self.report["libraries"][name] = {"path": str(path)}

    def invoke(self, extra):
        loader = types.SimpleNamespace(closure=mock.Mock(return_value=self.report))
        arguments = ["vramz", "--driver", str(self.root / "libraries/libcuda.so.1"),
                     "--cuda-root", str(self.root), "--nvcomp-root", str(self.root), *extra]
        with mock.patch.object(LAUNCHER, "__file__", str(self.root / "bin/vramz")), \
             mock.patch.object(LAUNCHER.os, "geteuid", return_value=1000), \
             mock.patch.object(LAUNCHER.importlib.util, "module_from_spec", return_value=loader), \
             mock.patch.object(LAUNCHER.importlib.util, "spec_from_file_location", return_value=types.SimpleNamespace(loader=mock.Mock())), \
             mock.patch.object(LAUNCHER.sys, "argv", arguments), \
             mock.patch.object(LAUNCHER.os, "execve") as execute, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = LAUNCHER.main()
        return status, execute, loader

    def test_check_only_cannot_execute(self):
        status, execute, loader = self.invoke(["--check-only"])
        self.assertEqual(status, 0)
        execute.assert_not_called()
        loader.closure.assert_called_once()

    def test_scoped_loader_discards_inherited_injection(self):
        with mock.patch.dict(os.environ, {"LD_PRELOAD": "/malicious.so", "LD_AUDIT": "/audit.so",
                                          "LD_LIBRARY_PATH": "/untrusted", "CUDA_VISIBLE_DEVICES": "1"}):
            _, execute, _ = self.invoke(["--", "--device", "0", "--run", "self-test"])
        execute.assert_called_once()
        environment = execute.call_args.args[2]
        self.assertEqual(environment["LD_LIBRARY_PATH"], self.report["loader_path"])
        self.assertEqual(set(environment), {"PATH", "LANG", "LC_ALL", "HOME", "LD_LIBRARY_PATH"})

    def test_shadowed_provider_rejected_without_loading(self):
        self.report["libraries"]["libcuda.so.1"]["path"] = str(self.root / "other-driver")
        (self.root / "other-driver").write_text("wrong provider")
        (self.root / "other-driver").chmod(0o644)
        status, execute, _ = self.invoke(["--check-only"])
        self.assertEqual(status, 2)
        execute.assert_not_called()

    def test_dynamic_lz4_is_rejected(self):
        self.report["libraries"]["liblz4.so.1"] = {"path": "/not-used"}
        status, execute, _ = self.invoke(["--check-only"])
        self.assertEqual(status, 2)
        execute.assert_not_called()

    def test_writable_provider_and_parent_rejected(self):
        path = self.root / "provider"
        path.write_text("identity")
        path.chmod(0o600)
        LAUNCHER.trusted_file(path)
        path.chmod(0o666)
        with self.assertRaises(ValueError):
            LAUNCHER.trusted_file(path)
        path.chmod(0o600)
        self.root.chmod(0o777)
        with self.assertRaises(ValueError):
            LAUNCHER.trusted_file(path)
        self.root.chmod(0o700)


if __name__ == "__main__":
    unittest.main()
