#!/usr/bin/env python3
"""Create a deterministic source-only review archive with an external SHA256 file."""
import argparse
import gzip
import hashlib
import io
from pathlib import Path
import tarfile

SUFFIXES = {".cpp", ".hpp", ".md", ".cmake", ".py", ".in", ".txt", ".json", ".jsonl", ".sh"}
DOTFILES = {".gitignore", ".gitattributes", ".clang-format", ".clang-tidy"}
PUBLIC_METADATA = {"LICENSE", "CITATION.cff"}
EXCLUDED = {".git", ".agents", ".codex", ".cache", "__pycache__", ".venv", "node_modules",
            "evidence", "downloads", "sdk"}


def source_files(source):
    files = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(p in EXCLUDED or p == "build" or p.startswith(("build-", "cmake-build-"))
               for p in relative.parts[:-1]):
            continue
        if path.is_symlink():
            raise ValueError(f"source symlink requires explicit review: {relative}")
        if not path.is_file():
            continue
        if (path.suffix not in SUFFIXES and path.name not in DOTFILES
                and path.name not in PUBLIC_METADATA):
            raise ValueError(f"unexpected source file: {relative}")
        payload = path.read_bytes()
        if payload.startswith((b"\x7fELF", b"!<arch>")) or b"\0" in payload:
            raise ValueError(f"binary is not source: {relative}")
        files.append((relative.as_posix(), payload, 0o755 if path.stat().st_mode & 0o111 else 0o644))
    return files


def write_archive(source, destination):
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.resolve().is_relative_to(source):
        raise ValueError("archive must be outside the source tree")
    digest_path = destination.with_name(destination.name + ".sha256")
    if destination.exists() or digest_path.exists():
        raise ValueError("review archives and hash files are never overwritten")
    files = source_files(source)
    if not files:
        raise ValueError("empty source tree")
    with destination.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, mtime=0, fileobj=raw) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, payload, mode in files:
                    entry = tarfile.TarInfo(name)
                    entry.size = len(payload)
                    entry.mode = mode
                    entry.mtime = entry.uid = entry.gid = 0
                    entry.uname = entry.gname = ""
                    archive.addfile(entry, io.BytesIO(payload))
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with digest_path.open("x") as stream:
        stream.write(f"{digest}  {destination.name}\n")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(write_archive(args.source, args.output))


if __name__ == "__main__":
    main()
