#!/usr/bin/python3
"""Validate the reference providers statically, then run one explicit device-0 command."""
import argparse
import grp
import pwd
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

sys.dont_write_bytecode = True


@lru_cache(maxsize=64)
def private_group(gid):
    group = grp.getgrgid(gid)
    allowed = {0, os.getuid()}
    return all(pwd.getpwnam(name).pw_uid in allowed for name in group.gr_mem) and all(
        user.pw_uid in allowed for user in pwd.getpwall() if user.pw_gid == gid)


def trusted_file(path):
    path = path.resolve(strict=True)
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Provider or executable is not a regular file: " + str(path))
    # Read-only system mounts can expose mapped ownership in restricted environments.
    if not os.statvfs(path).f_flag & os.ST_RDONLY:
        if value.st_mode & 0o022 or value.st_uid not in (0, os.getuid()):
            raise ValueError("Untrusted provider or executable permissions: " + str(path))
    for directory in path.parents:
        if os.statvfs(directory).f_flag & os.ST_RDONLY:
            continue
        parent = directory.stat()
        if (parent.st_uid not in (0, os.getuid()) or parent.st_mode & 0o002 or
                (parent.st_mode & 0o020 and not private_group(parent.st_gid))):
            raise ValueError("Untrusted parent directory: " + str(directory))


def human_report(rows):
    info = next((row for row in rows if row.get("kind") == "capabilities"), {})
    if info:
        print(f"VRAMZ experimental runtime | {info['device_name']} | device 0")
        print(f"Driver {info['driver_release']} API {info['driver_api']} | CUDA runtime API {info['runtime_api']} | nvCOMP {info['nvcomp_version']}")
    for row in rows:
        if "v1_workload_version" not in row:
            continue
        logical = row["aggregate_logical_bytes"]
        physical = row["settled_physical_charge"]
        compressed = sum(chunk["representation"] == "GPU_COMPRESSED" for chunk in row["chunks"])
        rejected = sum(cycle["rejected"] for cycle in row["cycles"])
        print(f"Logical working set: {logical:,} bytes")
        print(f"Measured RAW baseline: {row['raw_physical_charge']:,} bytes")
        print(f"Settled VMM backing: {physical:,} bytes | saved {row['raw_physical_charge'] - physical:,} bytes")
        print(f"Controlled logical/physical ratio: {logical / physical:.4f} (this data only)")
        print(f"Compressed chunks: {compressed}/{row['chunk_count']} | nonbeneficial attempts: {rejected}")
        print(f"Peak explicit GPU charge: {row['peak_admitted_gpu_bytes']:,} bytes")
        print(f"Integrity: {row['integrity']} | policy target reached: {row['target_reached']}")
    if rows:
        last = rows[-1]
        print(f"Result: {last.get('result', 'UNKNOWN')} | complete cleanup: {last.get('cleanup', False)}")
    print("Explicitly integrated GPU-resident storage; no universal VRAM expansion claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True, help="Canonical libcuda.so provider")
    parser.add_argument("--cuda-root", type=Path, required=True, help="Externally installed reviewed CUDA root")
    parser.add_argument("--nvcomp-root", type=Path, required=True, help="Externally installed reviewed nvCOMP root")
    parser.add_argument("--human", action="store_true", help="Render a readable summary; omit to retain raw JSON")
    parser.add_argument("--check-only", action="store_true", help="Inspect ELF/hashes without loading CUDA")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="-- --device 0 --run self-test (or demo, info, profiles, concurrency)")
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error("Run VRAMZ as an ordinary user, never as root")
    support = Path(__file__).resolve().parent.parent / "libexec/vramz"
    try:
        executable = (support / "vramz-run").resolve(strict=True)
        trusted_file(executable)
        for module in ("m13-elf-check.py", "m7-elf-check.py"):
            trusted_file(support / module)
        roots = [args.driver, args.cuda_root, args.nvcomp_root]
        if any(not path.is_absolute() or ":" in str(path) for path in roots):
            raise ValueError("Provider paths must be absolute and contain no colon")
        spec = importlib.util.spec_from_file_location("vramz_provider_check", support / "m13-elf-check.py")
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        report = check.closure(args.driver, args.cuda_root, args.nvcomp_root, executable)
        if "liblz4.so.1" in report["libraries"]:
            raise ValueError("Unexpected dynamic LZ4 fallback")
        directories = [Path(p) for p in report["loader_path"].split(":")]
        for name, item in report["libraries"].items():
            path = Path(item["path"])
            trusted_file(path)
            resolved = next((directory / name for directory in directories if (directory / name).is_file()), None)
            if resolved is None or resolved.resolve() != path:
                raise ValueError("Loader shadowing of " + name)
        trusted_file(executable)
        if args.check_only:
            print(json.dumps(report, indent=2))
            return 0
        arguments = args.arguments
        if arguments and arguments[0] == "--":
            arguments = arguments[1:]
        if not arguments:
            parser.error("Explicit --device 0 --run COMMAND is required")
        forbidden = {"--driver", "--cudart", "--nvcomp"}
        if any(a in forbidden for a in arguments):
            parser.error("Provider options cannot be overridden in the workload arguments")
        environment = {"PATH":"/usr/bin:/bin", "LANG":"C", "LC_ALL":"C",
                       "HOME":str(Path.home()), "LD_LIBRARY_PATH":report["loader_path"]}
        command = [str(executable), "--driver", report["libraries"]["libcuda.so.1"]["path"],
                   "--cudart", report["libraries"]["libcudart.so.13"]["path"],
                   "--nvcomp", report["libraries"]["libnvcomp.so.5"]["path"], *arguments]
        if args.human:
            # The child CLI bounds workload/output size. Preserve nonzero status and diagnostics.
            completed = subprocess.run(command, env=environment, capture_output=True, text=True)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            if completed.returncode:
                print(completed.stdout, end="")
                return completed.returncode if completed.returncode > 0 else 1
            rows = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
            if not rows or rows[-1].get("result") != "PASS":
                raise ValueError("Child did not report completed PASS")
            human_report(rows)
            return 0
        os.execve(str(executable), command, environment)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError) as error:
        print(json.dumps({"result":"PREFLIGHT_REJECTED", "error":str(error)}), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
