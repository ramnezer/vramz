#!/usr/bin/env python3
"""Read M13 ELF closure and hashes; never load/execute inspected ELF files."""

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

spec = importlib.util.spec_from_file_location("m7_elf", Path(__file__).with_name("m7-elf-check.py"))
m7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m7)

# M13+ active host profile. M7-M12 historical profiles and evidence are immutable.
# The prior M7 API below is historical provenance, not a new Driver API observation.
PINNED = {
    "libcuda.so.1": ("/usr/lib/x86_64-linux-gnu/libcuda.so.595.91.07",
                     "4839b5da17cd8f58a8e9c57e97c9b61f3c41be7934479f48afc983416b2492c3"),
    "libnvcomp.so.5": ("lib/libnvcomp.so.5.3.0.16",
                      "42582e1075df6fe93e1045430fadf5774218e25f990f8fb248bc9352dc806011"),
    "libcudart.so.13": ("lib/libcudart.so.13.3.29",
                       "9f92e4c3f7e14d0fed69ef876a45d4255a2dc88cb552a673a90ce881c26282fe"),
}
SYSTEM = {"librt.so.1", "libdl.so.2", "libpthread.so.0", "libstdc++.so.6", "libm.so.6",
          "libgcc_s.so.1", "libc.so.6", "ld-linux-x86-64.so.2", "liblz4.so.1"}
SYSTEM_DIR = Path("/usr/lib/x86_64-linux-gnu")
DRIVER_RELEASE = (595, 91, 7)
PRIOR_M7_DRIVER_API = 13020


def driver_release(text):
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)(?:\.([0-9]+))?\n?", text)
    m7.require(match is not None, "invalid NVIDIA Driver release metadata")
    release = tuple(int(value or 0) for value in match.groups())
    m7.require(all(value <= 4294967295 for value in release), "Driver release overflow")
    m7.require(release[0] >= 580, "NVIDIA Driver below CUDA-13 minimum branch 580")
    return release


def installed_driver_release():
    # sysfs module version text is CPU metadata; never query NVML or a device node.
    with Path("/sys/module/nvidia/version").open("r", encoding="ascii") as stream:
        text = stream.read(65)
    m7.require(len(text) <= 64, "unbounded NVIDIA Driver release metadata")
    release = driver_release(text)
    m7.require(release == DRIVER_RELEASE, "kernel Driver release differs from pinned real library")
    return release


def inspect(path):
    path = path.resolve(strict=True)
    m7.require("stubs" not in path.parts, "stub in M13 runtime closure")
    before = path.stat()
    dynamic = m7.readelf(path, "--dynamic")
    m7.require(not m7.dynamic_entries(dynamic, "RPATH|RUNPATH"), f"RPATH/RUNPATH: {path}")
    needed = m7.dynamic_entries(dynamic, "NEEDED")
    m7.require(all("/" not in name for name in needed), f"path in NEEDED: {path}")
    digest = m7.sha256(path)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    m7.require(all(getattr(before, f) == getattr(after, f) for f in fields), "file changed during inspection")
    return {"path": str(path), "sha256": digest, "needed": needed,
            "soname": m7.dynamic_entries(dynamic, "SONAME"),
            "identity": [getattr(after, f) for f in fields], "rpath": None, "runpath": None}


def closure(driver, cuda, nvcomp, binary=None):
    m7.inspect_driver(driver, cuda)
    release = installed_driver_release()
    roots = [cuda.resolve(strict=True), nvcomp.resolve(strict=True)]
    m7.require(all(r.is_absolute() and not str(r).startswith("/usr/local/") for r in roots),
               "M13 requires isolated development roots")
    chosen = {"libcuda.so.1": driver.resolve(strict=True),
              "libnvcomp.so.5": roots[1] / PINNED["libnvcomp.so.5"][0],
              "libcudart.so.13": roots[0] / PINNED["libcudart.so.13"][0]}
    for name, root in (("libnvcomp.so.5", roots[1]), ("libcudart.so.13", roots[0])):
        m7.require(chosen[name].resolve(strict=True).is_relative_to(root),
                   f"NVIDIA runtime escapes its verified isolated root: {name}")
    m7.require(str(chosen["libcuda.so.1"]) == PINNED["libcuda.so.1"][0], "unexpected real Driver")
    libraries = {}
    for name, path in chosen.items():
        data = inspect(path)
        m7.require(data["sha256"] == PINNED[name][1], f"unreviewed NVIDIA library: {name}")
        m7.require(data["soname"] == [name], f"unexpected SONAME: {name}")
        libraries[name] = data
    pending = [name for item in libraries.values() for name in item["needed"]]
    executable = None
    if binary is not None:
        executable = inspect(binary)
        m7.require(set(PINNED).issubset(executable["needed"]), "M13 binary missing a direct NVIDIA dependency")
        pending.extend(executable["needed"])
    while pending:
        name = pending.pop()
        if name in libraries:
            continue
        m7.require(name in SYSTEM, f"unexpected dependency: {name}")
        item = inspect(SYSTEM_DIR / name)
        libraries[name] = item
        pending.extend(item["needed"])
    for name, path in chosen.items():
        alias = (roots[1] / "lib" / name if name == "libnvcomp.so.5" else
                 roots[0] / "lib" / name if name == "libcudart.so.13" else SYSTEM_DIR / name)
        m7.require(alias.resolve(strict=True) == path.resolve(strict=True), f"incorrect loader alias: {alias}")
    loader_path = ":".join(str(p) for p in (roots[1] / "lib", roots[0] / "lib", SYSTEM_DIR))
    return {"checks": "PASS", "scope": "static ELF NEEDED closure; no dynamic loading or GPU calls",
            "libraries": libraries, "binary": executable, "loader_path": loader_path,
            "provider_baseline": "M13_PLUS_595_91_07",
            "prior_driver_api_observation_kind": "historical_M7_595_84",
            "prior_m7_driver_api": PRIOR_M7_DRIVER_API, "cuda_runtime_api": 13030,
            "nvidia_driver_release": list(release), "cuda13_minimum_driver_branch": 580,
            "compatibility_candidate": "MINOR_COMPATIBILITY_CANDIDATE",
            "physical_compatibility": "NOT_TESTED",
            "cuda_executed": False, "nvcomp_executed": False}


def identity_config(report):
    lines = []
    for kind, name in (("DRIVER", "libcuda.so.1"), ("NVCOMP", "libnvcomp.so.5"), ("CUDART", "libcudart.so.13")):
        item = report["libraries"][name]
        m7.require(re.fullmatch(r"/[A-Za-z0-9_./+-]+", item["path"]), "unsupported identity path syntax")
        value = json.dumps(item["path"]) + ", " + json.dumps(item["sha256"]) + ", {" + ", ".join(str(x) + "ULL" for x in item["identity"]) + "}"
        lines.append(f"set(VRAMZ_M13_{kind}_IDENTITY [=[{value}]=])")
        lines.append(f"set(VRAMZ_M13_{kind}_PATH [=[{item['path']}]=])")
    lines.append(f"set(VRAMZ_M13_LOADER_PATH [=[{report['loader_path']}]=])")
    lines.append("set(VRAMZ_M13_DRIVER_RELEASE [=[" + ", ".join(str(x) + "U" for x in report["nvidia_driver_release"]) + "]=])")
    lines.append(f"set(VRAMZ_M13_PRIOR_DRIVER_API {PRIOR_M7_DRIVER_API})")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--nvcomp-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--identity-header", type=Path)
    args = parser.parse_args()
    try:
        report = closure(args.driver, args.cuda_root, args.nvcomp_root, args.binary)
        if args.identity_header is not None:
            args.identity_header.write_text(identity_config(report))
        print(json.dumps(report, sort_keys=True, indent=2))
    except (OSError, ValueError, m7.subprocess.SubprocessError) as error:
        print(json.dumps({"checks": "FAIL", "error": str(error)}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
