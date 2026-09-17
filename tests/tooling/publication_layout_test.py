#!/usr/bin/env python3
"""CPU-only checks for the cleaned public source layout and archive metadata."""
import hashlib
import importlib.util
from pathlib import Path
import re
import tarfile
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

SOURCE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "source_archiver", SOURCE / "tools/ci/m8-source-archive.py")
ARCHIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVER)


class PublicationLayout(unittest.TestCase):
    def test_public_metadata_is_archivable(self):
        names = {name for name, _, _ in ARCHIVER.source_files(SOURCE)}
        self.assertTrue({"LICENSE", "CITATION.cff", ".gitattributes", "README.md"} <= names)

    def test_installed_consumer_requests_current_package_series(self):
        cmake = (SOURCE / "CMakeLists.txt").read_text()
        version = re.search(r"project\(VRAMZ VERSION (\d+\.\d+)\.\d+", cmake)
        self.assertIsNotNone(version)
        consumer = (SOURCE / "tests/consumer/CMakeLists.txt").read_text()
        self.assertIn(f"find_package(VRAMZ {version.group(1)} CONFIG REQUIRED)", consumer)

    def test_required_oracles_keep_exact_bytes(self):
        fixtures = SOURCE / "tests/fixtures/oracles"
        expected = {
            "M3_SCENARIO_RESULTS.jsonl":
                ("8094c716c4dad5e494eeaa7261b5aa3b8ff73670ee0b47c881e0049fe5a73948", 82),
            "M5_FAKE_GPU_COMPRESSION_RESULTS.jsonl":
                ("fb39d58d3f7196031c04d262b4ec9822bab324abbac3288b9fef7be24aeda7ee", 4),
        }
        for name, (digest, rows) in expected.items():
            with self.subTest(name=name):
                payload = (fixtures / name).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
                self.assertEqual(len(payload.splitlines()), rows)

    def test_no_checkpoint_diary_directory(self):
        self.assertFalse((SOURCE / "docs/checkpoints").exists())
        self.assertTrue((SOURCE / "tests/fixtures/PREHARDWARE_DEPENDENCIES.json").is_file())

    def test_public_markdown_relative_links_exist(self):
        for path in SOURCE.rglob("*.md"):
            # Generated build trees are not documentation inputs.
            if any(part.startswith(("build", "cmake-build"))
                   for part in path.relative_to(SOURCE).parts[:-1]):
                continue
            text = re.sub(r"(?ms)^```.*?^```[^\n]*", "", path.read_text())
            for target in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", text):
                target = target.strip().split(' "', 1)[0].strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith(("#", "//")):
                    continue
                with self.subTest(document=str(path.relative_to(SOURCE)), link=target):
                    resolved = (path.parent / unquote(parsed.path)).resolve()
                    self.assertTrue(resolved.is_relative_to(SOURCE.resolve()))
                    self.assertTrue(resolved.exists())

    def test_archive_preserves_metadata_and_is_deterministic(self):
        with tempfile.TemporaryDirectory(prefix="vramz-source-package-") as name:
            directory = Path(name)
            source = directory / "source"
            source.mkdir()
            for entry in ("LICENSE", "CITATION.cff", ".gitattributes", "README.md"):
                (source / entry).write_text("test metadata\n")
            one, two = directory / "one.tar.gz", directory / "two.tar.gz"
            self.assertEqual(ARCHIVER.write_archive(source, one),
                             ARCHIVER.write_archive(source, two))
            self.assertEqual(one.read_bytes(), two.read_bytes())
            with tarfile.open(one) as archive:
                self.assertEqual(set(archive.getnames()),
                                 {"LICENSE", "CITATION.cff", ".gitattributes", "README.md"})
            with self.assertRaises(ValueError):
                ARCHIVER.write_archive(source, one)

    def test_archive_rejects_source_output_symlinks_and_binary(self):
        with tempfile.TemporaryDirectory(prefix="vramz-source-negative-") as name:
            directory = Path(name)
            source = directory / "source"
            source.mkdir()
            (source / "README.md").write_text("text\n")
            with self.assertRaises(ValueError):
                ARCHIVER.write_archive(source, source / "inside.tar.gz")
            (source / "link.md").symlink_to(source / "README.md")
            with self.assertRaises(ValueError):
                ARCHIVER.source_files(source)
            (source / "link.md").unlink()
            (source / "bad.cpp").write_bytes(b"\x7fELF\0binary")
            with self.assertRaises(ValueError):
                ARCHIVER.source_files(source)


if __name__ == "__main__":
    unittest.main()
