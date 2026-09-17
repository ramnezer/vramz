#!/usr/bin/env python3
"""Read-only, deterministic pre-hardware evidence checks. Never probes or loads CUDA."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


M3_HASH = "8094c716c4dad5e494eeaa7261b5aa3b8ff73670ee0b47c881e0049fe5a73948"
M5_HASH = "fb39d58d3f7196031c04d262b4ec9822bab324abbac3288b9fef7be24aeda7ee"
COMPILE_ONLY = {"libvramz_cuda_driver.so", "libvramz_nvcomp_lz4.so"}
NVIDIA = re.compile(r"NEEDED.*(?:libcuda\b|libcudart\b|libnvcomp\b)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_elf(path):
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            return False
    result = subprocess.run(["readelf", "-d", str(path)], check=True,
                            capture_output=True, text=True, timeout=10)
    require(not NVIDIA.search(result.stdout), f"NVIDIA runtime dependency: {path.name}")
    for line in result.stdout.splitlines():
        if "RPATH" in line or "RUNPATH" in line:
            require(not re.search(r"/stubs/|/home/|/tmp/", line),
                    f"nonportable or stub loader path: {path.name}")
    return True


def check(source, build, install):
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    require("VRAMZ_ALLOW_REAL_GPU_EXECUTION:BOOL=OFF" in cache,
            "real execution gate must explicitly be OFF")
    for gate in ("VRAMZ_BUILD_M8_COMPRESSION_SMOKE", "VRAMZ_ALLOW_REAL_NVCOMP_EXECUTION",
                 "VRAMZ_BUILD_M9_PHYSICAL_SAVINGS_SMOKE",
                 "VRAMZ_BUILD_M10_MULTI_CHUNK_RESIDENCY_SMOKE", "VRAMZ_BUILD_M11_POLICY_PRESSURE_SMOKE", "VRAMZ_BUILD_M12_CONTROLLED_CAPACITY_SMOKE", "VRAMZ_BUILD_M13_CONTROLLED_CAPACITY_2X_SMOKE"):
        require(f"{gate}:BOOL=OFF" in cache, f"{gate} must be OFF")
    for directory in (build, install / "bin"):
        require(not (directory / "vramz-m8-compression-smoke").exists(),
                "physical M8 target present in ordinary build/install")
        require(not (directory / "vramz-m9-physical-savings-smoke").exists(),
                "physical M9 target present in ordinary build/install")
        require(not (directory / "vramz-m10-multi-chunk-residency-smoke").exists(),
                "physical M10 target present in ordinary build/install")
        require(not (directory / "vramz-m11-policy-pressure-smoke").exists(),
                "physical M11 target present in ordinary build/install")
        require(not (directory / "vramz-m12-controlled-capacity-smoke").exists(),
                "physical M12 target present in ordinary build/install")
        require(not (directory / "vramz-m13-controlled-capacity-2x-smoke").exists(),
                "physical M13 target present in ordinary build/install")
    manifest = json.loads((source / "tests/fixtures/PREHARDWARE_DEPENDENCIES.json").read_text())
    require(manifest["real_hardware_validation"] == "NOT_TESTED", "hardware claim is forbidden")
    require(manifest["cuda"]["development_release"] == "13.3.1", "CUDA identity mismatch")
    require(manifest["nvcomp"]["build"] == "5.3.0.16", "nvCOMP identity mismatch")
    for name, expected, rows in [
        ("M3_SCENARIO_RESULTS.jsonl", M3_HASH, 82),
        ("M5_FAKE_GPU_COMPRESSION_RESULTS.jsonl", M5_HASH, 4),
    ]:
        payload = (source / "tests/fixtures/oracles" / name).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == expected, f"oracle mismatch: {name}")
        require(len(payload.splitlines()) == rows, f"row count mismatch: {name}")
    results = ET.parse(build / "m6-results.xml").getroot()
    cases = list(results.iter("testcase"))
    require(len(cases) >= (14 if "VRAMZ_ENABLE_CPU_LZ4:BOOL=ON" in cache else 2),
            "incomplete CTest evidence")
    for case in cases:
        require(case.find("failure") is None and case.find("error") is None and
                case.find("skipped") is None, "failed or skipped CTest case")
        require(case.attrib.get("status", "run") == "run", "CTest did not run")
    checked_build = 0
    for path in sorted(build.iterdir()):
        if path.is_file() and path.name not in COMPILE_ONLY:
            checked_build += int(inspect_elf(path))
    require(checked_build >= len(cases), "CTest executables missing from evidence")
    checked_install = 0
    require((install / "include/vramz/runtime.hpp").is_file(), "public install missing")
    for path in sorted(install.rglob("*")):
        if not path.is_file():
            continue
        require(path.name not in COMPILE_ONLY, "compile-only adapter installed")
        require(path.name not in {"cuda.h", "cuda_runtime.h", "nvcomp.h", "testing.hpp", "backend.hpp"},
                "SDK or internal test/backend header installed")
        checked_install += int(inspect_elf(path))
        if path.suffix == ".cmake":
            require(not re.search(r"/home/|/tmp/|/stubs/", path.read_text()),
                    "nonportable installed package configuration")
    require(checked_install >= 2, "installed CPU tools missing")
    return {"schema_version": 1, "checks": "PASS", "ctest_suites": len(cases),
            "build_elf_checked": checked_build, "installed_elf_checked": checked_install,
            "real_execution_gate": "OFF", "hardware_validation": "NOT_TESTED",
            "gates_E_F_G": "NOT RUN", "m3_sha256": M3_HASH, "m5_sha256": M5_HASH,
            "scope": "configuration, retained results, ELF and packaging; full quality matrix is separate evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--install", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = check(args.source.resolve(), args.build.resolve(), args.install.resolve())
    except (OSError, ValueError, KeyError, ET.ParseError, subprocess.SubprocessError) as error:
        print(json.dumps({"checks": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
