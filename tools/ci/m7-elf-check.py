#!/usr/bin/env python3
"""Inspect M7 driver/binary ELF data without loading a library or executing CUDA."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def readelf(path, option):
    with path.open("rb") as stream:
        require(stream.read(4) == b"\x7fELF", f"not an ELF file: {path}")
    return subprocess.run(["readelf", "--wide", option, str(path)], check=True,
                          capture_output=True, text=True, timeout=10,
                          env={**os.environ, "LC_ALL": "C"}).stdout


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dynamic_entries(dynamic, tag):
    return re.findall(r"\((?:" + tag + r")\).*?\[([^\]]+)\]", dynamic)


def inspect_driver(driver, toolkit_root):
    require(driver.is_absolute(), "real driver path must be explicit and absolute")
    require("stubs" not in driver.parts, "Driver stub path is forbidden")
    driver = driver.resolve(strict=True)
    toolkit_root = toolkit_root.resolve(strict=True)
    require(driver.is_file(), "real driver must be a regular file")
    require("stubs" not in driver.parts, "resolved Driver stub path is forbidden")
    require(not driver.is_relative_to(toolkit_root),
            "real driver must be outside the isolated CUDA development root")
    initial_identity = driver.stat()
    header = readelf(driver, "--file-header")
    require(re.search(r"Class:\s+ELF64\b", header), "M7 requires an ELF64 driver")
    require(re.search(r"Type:\s+DYN\b", header), "driver must be a shared ELF object")
    require(re.search(r"Machine:\s+Advanced Micro Devices X86-64\b", header),
            "M7 requires the Linux x86_64 driver")
    dynamic = readelf(driver, "--dynamic")
    require(dynamic_entries(dynamic, "SONAME") == ["libcuda.so.1"],
            "driver ELF SONAME must be libcuda.so.1")
    dependencies = dynamic_entries(dynamic, "NEEDED")
    require(not any(re.search(r"lib(?:nvcomp|cudart)", name) for name in dependencies),
            "RAW driver must not depend on nvCOMP or CUDA Runtime")
    for value in dynamic_entries(dynamic, "RPATH|RUNPATH"):
        require("/stubs/" not in value and str(toolkit_root) not in value,
                "driver contains a development loader path")
    symbols = readelf(driver, "--dyn-syms")
    for symbol in ("cuInit", "cuDriverGetVersion", "cuMemCreate", "cuMemMap",
                   "cuMemUnmap", "cuMemSetAccess", "cuMemGetAllocationPropertiesFromHandle"):
        require(any(re.search(r"\b" + symbol + r"(?:@@?\S+)?$", line) and
                    not re.search(r"\bUND\b", line) for line in symbols.splitlines()),
                f"driver does not export required symbol: {symbol}")
    digest = sha256(driver)
    for relative in ("lib/stubs/libcuda.so", "lib64/stubs/libcuda.so",
                     "targets/x86_64-linux/lib/stubs/libcuda.so"):
        stub = toolkit_root / relative
        if stub.is_file():
            require(sha256(stub) != digest,
                    "driver is byte-identical to the development stub (possibly copied/renamed)")
    identity = driver.stat()
    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
        require(getattr(identity, field) == getattr(initial_identity, field),
                "driver changed during static inspection; rebuild only after manual review")
    return {"driver_path": str(driver), "driver_sha256": digest,
            "driver_soname": "libcuda.so.1", "driver_needed": dependencies,
            "driver_device": identity.st_dev, "driver_inode": identity.st_ino,
            "driver_size": identity.st_size, "driver_mtime_ns": identity.st_mtime_ns,
            "driver_ctime_ns": identity.st_ctime_ns}


def inspect_binary(binary):
    binary = binary.resolve(strict=True)
    dynamic = readelf(binary, "--dynamic")
    dependencies = dynamic_entries(dynamic, "NEEDED")
    require(dependencies.count("libcuda.so.1") == 1,
            "M7 binary must directly depend on exactly one libcuda.so.1")
    require(not any("/" in name for name in dependencies),
            "M7 ELF must not embed an absolute dependency path")
    require(not any(re.search(r"lib(?:nvcomp|cudart)", name) for name in dependencies),
            "M7 RAW binary must not depend on nvCOMP or CUDA Runtime")
    require(not dynamic_entries(dynamic, "RPATH|RUNPATH"),
            "M7 RAW binary must have no RPATH/RUNPATH; use the host driver loader configuration")
    return {"binary_path": str(binary), "binary_sha256": sha256(binary),
            "binary_needed": dependencies, "binary_rpath": "NONE"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    try:
        report = inspect_driver(args.driver, args.toolkit_root)
        if args.binary is not None:
            report.update(inspect_binary(args.binary))
        report.update({"checks": "PASS", "cuda_executed": False,
                       "scope": "static ELF inspection only; loaded driver still requires host corroboration"})
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"checks": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
