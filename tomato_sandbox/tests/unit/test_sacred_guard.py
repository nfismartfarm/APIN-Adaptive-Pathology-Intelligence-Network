"""
Unit tests for tomato_sandbox.utils.sacred_guard.verify_manifest().

All tests use temporary directories and a fixture manifest — they NEVER touch
the real ``.claude/sacred_manifest.json`` or any sacred file.

# spec: section 2.6 + .claude/sacred_manifest.json
#        directory_hash_algorithm_canonical + DEC-019 log_exclusions extension

Coverage targets (100% per S26 utility-module requirement):
  - _sha256_file
  - _sha256_directory (with and without exclusions)
  - _check_entry (PASS / FAIL / MISSING for file and directory entries)
  - verify_manifest (PASS dict, FAIL on drift, MISSING on absent, manifest
    missing, bad JSON, empty entries)
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tomato_sandbox.utils.sacred_guard import (
    _sha256_directory,
    _sha256_file,
    verify_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_manifest(entries: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal manifest dict compatible with verify_manifest()."""
    return {
        "version": 1,
        "entries": entries,
        "directory_hash_algorithm_canonical": {},
    }


def _dump_manifest(tmp_path: Path, entries: dict[str, Any]) -> Path:
    """Write a fixture manifest JSON to tmp_path/.claude/sacred_manifest.json
    and return the path."""
    manifest_dir = tmp_path / ".claude"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "sacred_manifest.json"
    manifest_path.write_text(
        json.dumps(_make_manifest(entries), indent=2), encoding="utf-8"
    )
    return manifest_path


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_dir_manual(dir_path: Path, exclusions: list[str] | None = None) -> str:
    """Reference implementation used in tests to independently compute expected hashes."""
    import fnmatch
    excl = exclusions or []
    file_hashes: dict[str, str] = {}
    for dirpath, _dirs, filenames in os.walk(str(dir_path)):
        for fname in filenames:
            if any(fnmatch.fnmatch(os.path.basename(fname), pat) for pat in excl):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, str(dir_path)).replace(os.sep, "/")
            with open(full, "rb") as fh:
                file_hashes[rel] = hashlib.sha256(fh.read()).hexdigest()
    canonical = json.dumps(file_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# _sha256_file
# ---------------------------------------------------------------------------

class TestSha256File:
    def test_known_bytes(self, tmp_path: Path) -> None:
        data = b"hello sacred guard"
        f = tmp_path / "sample.bin"
        _write(f, data)
        assert _sha256_file(f) == _hash_bytes(data)

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        _write(f, b"")
        assert _sha256_file(f) == _hash_bytes(b"")

    def test_binary_content(self, tmp_path: Path) -> None:
        data = bytes(range(256))
        f = tmp_path / "binary.bin"
        _write(f, data)
        assert _sha256_file(f) == _hash_bytes(data)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        _write(f1, b"abc")
        _write(f2, b"abd")
        assert _sha256_file(f1) != _sha256_file(f2)


# ---------------------------------------------------------------------------
# _sha256_directory
# ---------------------------------------------------------------------------

class TestSha256Directory:
    def test_empty_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "empty_dir"
        d.mkdir()
        result = _sha256_directory(d, [])
        # Empty dir → empty dict → canonical is "{}" → sha256 of that
        expected = hashlib.sha256(
            json.dumps({}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert result == expected

    def test_single_file(self, tmp_path: Path) -> None:
        d = tmp_path / "mydir"
        d.mkdir()
        _write(d / "hello.txt", b"world")
        result = _sha256_directory(d, [])
        expected = _hash_dir_manual(d)
        assert result == expected

    def test_multiple_files_sorted(self, tmp_path: Path) -> None:
        """Result must be identical regardless of os.walk traversal order."""
        d = tmp_path / "multi"
        d.mkdir()
        _write(d / "b.txt", b"bbb")
        _write(d / "a.txt", b"aaa")
        result = _sha256_directory(d, [])
        expected = _hash_dir_manual(d)
        assert result == expected

    def test_subdirectory_files(self, tmp_path: Path) -> None:
        d = tmp_path / "nested"
        d.mkdir()
        _write(d / "sub" / "deep.txt", b"deep")
        _write(d / "top.txt", b"top")
        result = _sha256_directory(d, [])
        expected = _hash_dir_manual(d)
        assert result == expected

    def test_forward_slash_paths_in_hash(self, tmp_path: Path) -> None:
        """rel_path inside the JSON must use forward slashes on all platforms."""
        d = tmp_path / "slashtest"
        d.mkdir()
        _write(d / "sub" / "file.py", b"code")
        result = _sha256_directory(d, [])
        # Manually build expected with forward slashes to verify normalisation
        file_hashes = {"sub/file.py": _hash_bytes(b"code")}
        canonical = json.dumps(file_hashes, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert result == expected

    def test_exclusions_applied(self, tmp_path: Path) -> None:
        """Files matching *.log should be excluded from the hash."""
        d = tmp_path / "withlog"
        d.mkdir()
        _write(d / "source.py", b"print('hi')")
        _write(d / "runtime.log", b"lots of logs")
        _write(d / "server.log.1", b"rotated")
        # Without exclusion the hash includes the log files
        hash_no_excl = _sha256_directory(d, [])
        # With exclusion the hash omits them
        hash_excl = _sha256_directory(d, ["*.log", "*.log.*"])
        assert hash_no_excl != hash_excl
        # The excluded hash should equal a dir that only has source.py
        expected = _hash_dir_manual(d, exclusions=["*.log", "*.log.*"])
        assert hash_excl == expected

    def test_log_exclusion_basename_only(self, tmp_path: Path) -> None:
        """fnmatch is applied to basename, not full path, so a dir named
        'logs' is traversed and its non-log files are included."""
        d = tmp_path / "basenames"
        d.mkdir()
        # File inside a directory named 'logs' but not itself a .log file
        _write(d / "logs" / "important.py", b"crucial")
        # File that IS a .log
        _write(d / "server.log", b"noisy")
        result = _sha256_directory(d, ["*.log"])
        # Expect only important.py
        file_hashes = {"logs/important.py": _hash_bytes(b"crucial")}
        canonical = json.dumps(file_hashes, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert result == expected

    def test_compact_json_separators(self, tmp_path: Path) -> None:
        """Verify that the compact separators (',', ':') are used (not default
        which adds spaces).  The two produce different hashes."""
        d = tmp_path / "sep_test"
        d.mkdir()
        _write(d / "x.txt", b"x")
        # Compute what the DEFAULT separators would produce
        file_hashes = {"x.txt": _hash_bytes(b"x")}
        canonical_default = json.dumps(file_hashes, sort_keys=True)
        canonical_compact = json.dumps(
            file_hashes, sort_keys=True, separators=(",", ":")
        )
        # They differ (unless there's exactly one key with no separator ambiguity)
        # With one k/v pair: default={"x.txt": "..."} vs compact={"x.txt":"..."}
        assert canonical_default != canonical_compact
        # Our function must match compact
        result = _sha256_directory(d, [])
        expected = hashlib.sha256(canonical_compact.encode("utf-8")).hexdigest()
        assert result == expected


# ---------------------------------------------------------------------------
# verify_manifest — fixture-based (no real manifest touched)
# ---------------------------------------------------------------------------

class TestVerifyManifestFileEntries:
    def test_single_file_pass(self, tmp_path: Path) -> None:
        content = b"sacred content"
        _write(tmp_path / "models" / "model.pt", content)
        entries = {
            "models/model.pt": {
                "type": "file",
                "sha256": _hash_bytes(content),
            }
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        # verify_manifest uses CWD as project root when manifest_path is given
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert result == {"models/model.pt": "PASS"}

    def test_single_file_fail_on_drift(self, tmp_path: Path) -> None:
        content = b"original"
        _write(tmp_path / "models" / "model.pt", content)
        entries = {
            "models/model.pt": {
                "type": "file",
                "sha256": _hash_bytes(b"different content"),
            }
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert result == {"models/model.pt": "FAIL"}

    def test_single_file_missing(self, tmp_path: Path) -> None:
        entries = {
            "models/missing.pt": {
                "type": "file",
                "sha256": "aabbcc",
            }
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert result == {"models/missing.pt": "MISSING"}

    def test_multiple_files_mixed(self, tmp_path: Path) -> None:
        content_a = b"file a data"
        content_b = b"file b data"
        _write(tmp_path / "a.pt", content_a)
        _write(tmp_path / "b.pt", content_b)
        entries = {
            "a.pt": {"type": "file", "sha256": _hash_bytes(content_a)},
            "b.pt": {"type": "file", "sha256": _hash_bytes(b"wrong")},
            "c.pt": {"type": "file", "sha256": "doesntmatter"},
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert result["a.pt"] == "PASS"
        assert result["b.pt"] == "FAIL"
        assert result["c.pt"] == "MISSING"


class TestVerifyManifestDirectoryEntries:
    def test_directory_pass(self, tmp_path: Path) -> None:
        d = tmp_path / "scripts" / "apin"
        d.mkdir(parents=True)
        _write(d / "main.py", b"code")
        expected_hash = _hash_dir_manual(d)
        entries = {
            "scripts/apin/": {
                "type": "directory",
                "sha256": expected_hash,
            }
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert result == {"scripts/apin/": "PASS"}

    def test_directory_fail_on_added_file(self, tmp_path: Path) -> None:
        d = tmp_path / "scripts" / "apin"
        d.mkdir(parents=True)
        _write(d / "main.py", b"code")
        expected_hash = _hash_dir_manual(d)
        # Now add another file (simulating drift)
        _write(d / "new_file.py", b"intruder")
        entries = {
            "scripts/apin/": {
                "type": "directory",
                "sha256": expected_hash,
            }
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert result == {"scripts/apin/": "FAIL"}

    def test_directory_missing(self, tmp_path: Path) -> None:
        entries = {
            "scripts/nonexistent/": {
                "type": "directory",
                "sha256": "aabbcc",
            }
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert result == {"scripts/nonexistent/": "MISSING"}

    def test_directory_log_exclusion_pass(self, tmp_path: Path) -> None:
        """A log file added after baselining must NOT cause FAIL if excluded."""
        d = tmp_path / "scripts" / "apin"
        d.mkdir(parents=True)
        _write(d / "server.py", b"apin code")
        # Baseline WITHOUT the log file
        expected_hash = _hash_dir_manual(d, exclusions=["*.log"])
        # Now simulate a runtime log being written
        _write(d / "runtime.log", b"lots of noise")
        entries = {
            "scripts/apin/": {
                "type": "directory",
                "sha256": expected_hash,
                "log_exclusions": ["*.log"],
            }
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        # PASS because the log is excluded from the hash
        assert result == {"scripts/apin/": "PASS"}

    def test_directory_log_exclusion_still_fails_on_source_drift(
        self, tmp_path: Path
    ) -> None:
        """Exclusion only covers *.log files.  A .py file change still FAILS."""
        d = tmp_path / "scripts" / "apin"
        d.mkdir(parents=True)
        _write(d / "server.py", b"original")
        expected_hash = _hash_dir_manual(d, exclusions=["*.log"])
        # Modify a .py file (real drift)
        _write(d / "server.py", b"tampered")
        entries = {
            "scripts/apin/": {
                "type": "directory",
                "sha256": expected_hash,
                "log_exclusions": ["*.log"],
            }
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert result == {"scripts/apin/": "FAIL"}

    def test_directory_trailing_slash_resolved(self, tmp_path: Path) -> None:
        """Manifest keys with trailing '/' must resolve correctly."""
        d = tmp_path / "data" / "mydir"
        d.mkdir(parents=True)
        _write(d / "record.csv", b"col1,col2\n1,2\n")
        expected_hash = _hash_dir_manual(d)
        entries = {
            "data/mydir/": {
                "type": "directory",
                "sha256": expected_hash,
            }
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert result["data/mydir/"] == "PASS"


class TestVerifyManifestEdgeCases:
    def test_manifest_file_not_found_returns_empty_dict(self, tmp_path: Path) -> None:
        missing_manifest = tmp_path / "nonexistent_manifest.json"
        result = verify_manifest(missing_manifest)
        assert result == {}

    def test_manifest_invalid_json_returns_empty_dict(self, tmp_path: Path) -> None:
        bad_manifest = tmp_path / "bad.json"
        bad_manifest.write_text("{this is not json", encoding="utf-8")
        result = verify_manifest(bad_manifest)
        assert result == {}

    def test_empty_entries_returns_empty_dict(self, tmp_path: Path) -> None:
        manifest_path = _dump_manifest(tmp_path, {})
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert result == {}

    def test_return_type_is_dict_str_str(self, tmp_path: Path) -> None:
        content = b"data"
        _write(tmp_path / "file.pt", content)
        entries = {"file.pt": {"type": "file", "sha256": _hash_bytes(content)}}
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
            assert v in {"PASS", "FAIL", "MISSING"}

    def test_all_pass_scenario(self, tmp_path: Path) -> None:
        """When everything matches, every value is 'PASS'."""
        content1 = b"model weights"
        content2 = b"config data"
        _write(tmp_path / "models" / "m.pt", content1)
        _write(tmp_path / "app" / "config.py", content2)
        d = tmp_path / "scripts" / "src"
        d.mkdir(parents=True)
        _write(d / "a.py", b"code a")
        dir_hash = _hash_dir_manual(d)
        entries = {
            "models/m.pt": {"type": "file", "sha256": _hash_bytes(content1)},
            "app/config.py": {"type": "file", "sha256": _hash_bytes(content2)},
            "scripts/src/": {"type": "directory", "sha256": dir_hash},
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert all(v == "PASS" for v in result.values())
        assert set(result.keys()) == {"models/m.pt", "app/config.py", "scripts/src/"}

    def test_only_one_entry_fails_when_only_one_drifts(self, tmp_path: Path) -> None:
        """Changing one file must not affect status of other entries."""
        content_a = b"unchanged"
        content_b = b"original"
        _write(tmp_path / "stable.pt", content_a)
        _write(tmp_path / "drifted.pt", content_b)
        entries = {
            "stable.pt": {"type": "file", "sha256": _hash_bytes(content_a)},
            "drifted.pt": {"type": "file", "sha256": _hash_bytes(b"expected")},
        }
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        assert result["stable.pt"] == "PASS"
        assert result["drifted.pt"] == "FAIL"

    def test_no_print_calls(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
        """No output should come from print() — only from the logger (stdout JSON).
        We can't easily distinguish, but at minimum the function should not crash."""
        manifest_path = _dump_manifest(tmp_path, {})
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            verify_manifest(manifest_path)
        finally:
            os.chdir(old_cwd)
        # If we reach here without exception, the function ran cleanly.
        # The logger may emit to stdout; that's expected and not a print() call.

    def test_manifest_loaded_each_call(self, tmp_path: Path) -> None:
        """verify_manifest must reload the manifest on each call so rebaselines
        are picked up without a process restart."""
        content = b"initial"
        _write(tmp_path / "f.pt", content)
        entries = {"f.pt": {"type": "file", "sha256": _hash_bytes(content)}}
        manifest_path = _dump_manifest(tmp_path, entries)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            r1 = verify_manifest(manifest_path)
            assert r1["f.pt"] == "PASS"

            # Update file and rebaseline the manifest on disk
            new_content = b"updated"
            _write(tmp_path / "f.pt", new_content)
            new_entries = {"f.pt": {"type": "file", "sha256": _hash_bytes(new_content)}}
            manifest_path.write_text(
                json.dumps(_make_manifest(new_entries)), encoding="utf-8"
            )

            r2 = verify_manifest(manifest_path)
            assert r2["f.pt"] == "PASS"
        finally:
            os.chdir(old_cwd)
