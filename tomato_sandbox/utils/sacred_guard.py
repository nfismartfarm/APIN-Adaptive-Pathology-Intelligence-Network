"""
Sacred-file integrity guard for the Tomato 3-Signal sandbox.

Spec section: 2 (Sacred files), specifically spec Section 2.6 and
`.claude/sacred_manifest.json` field ``directory_hash_algorithm_canonical``
with DEC-019 ``log_exclusions`` extension.

Public API
----------
verify_manifest(manifest_path=None) -> dict[str, str]
    Verify every entry in ``.claude/sacred_manifest.json`` against its stored
    SHA-256 hash.  Returns a dict mapping each entry path string to one of:
    - ``"PASS"``    – hash matches
    - ``"FAIL"``    – hash mismatch (drift detected)
    - ``"MISSING"`` – path does not exist on disk

# spec: section 2.6 + .claude/sacred_manifest.json
#        directory_hash_algorithm_canonical + DEC-019 log_exclusions extension
#
# DEC-019 note: scripts/apin/ has log_exclusions: ["*.log", "*.log.*"] because
# the legacy APIN server writes runtime logs into its own sacred directory.
# Excluding *.log patterns prevents false drift events on every server run.
# Other directory entries omit log_exclusions (default = []).
# See rebaseline_history in the manifest entry for the audit trail.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tomato_sandbox.utils.logging import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# This file lives at tomato_sandbox/utils/sacred_guard.py.
# parents: [utils/, tomato_sandbox/, <project_root>]
#           0        1                2
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
_DEFAULT_MANIFEST_PATH = _PROJECT_ROOT / ".claude" / "sacred_manifest.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file's raw bytes.

    # spec: sacred_manifest.json directory_hash_algorithm_canonical — file entries
    # use simple sha256(file_bytes).hexdigest(); no JSON wrapping.
    """
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _sha256_directory(dir_path: Path, exclusions: list[str]) -> str:
    """Return the canonical directory hash for a directory tree.

    Algorithm is implemented verbatim from
    ``.claude/sacred_manifest.json`` ``directory_hash_algorithm_canonical``:

    1. file_hashes = {}
    2. exclusions = entry.get('log_exclusions', [])   # per DEC-019
    3. For each file under directory (os.walk):
       basename = os.path.basename(full_path)
       if any(fnmatch.fnmatch(basename, pat) for pat in exclusions): continue
       rel_path = relpath(full, dir).replace(os.sep, "/")
       file_hashes[rel_path] = sha256(file_contents).hexdigest()
    4. canonical = json.dumps(file_hashes, sort_keys=True, separators=(",", ":"))
       # IMPORTANT: separators=(",", ":") produces compact JSON without spaces.
       # Default json.dumps uses separators=(", ", ": ") which gives DIFFERENT bytes.
    5. return sha256(canonical.encode("utf-8")).hexdigest()

    # spec: sacred_manifest.json directory_hash_algorithm_canonical pseudocode steps 1-5
    # DEC-019: exclusions matched via fnmatch against file basename
    """
    file_hashes: dict[str, str] = {}

    for dirpath, _dirs, filenames in os.walk(str(dir_path)):
        for fname in filenames:
            # Step 3a: apply exclusions via fnmatch on basename
            basename = os.path.basename(fname)
            if any(fnmatch.fnmatch(basename, pat) for pat in exclusions):
                continue

            full_path = os.path.join(dirpath, fname)

            # Step 3b: relative path with forward slashes
            rel_path = os.path.relpath(full_path, str(dir_path)).replace(
                os.sep, "/"
            )

            # Step 3c: SHA-256 of file contents
            with open(full_path, "rb") as fh:
                file_hashes[rel_path] = hashlib.sha256(fh.read()).hexdigest()

    # Step 4: compact JSON, keys sorted
    # separators=(",", ":") is mandatory — default adds spaces and produces
    # a different byte sequence and therefore a different hash.
    canonical = json.dumps(file_hashes, sort_keys=True, separators=(",", ":"))

    # Step 5: SHA-256 of the UTF-8 encoded canonical string
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_entry(
    entry_key: str,
    entry: dict[str, Any],
    project_root: Path,
) -> str:
    """Verify a single manifest entry.  Returns "PASS", "FAIL", or "MISSING".

    # spec: section 2.6; sacred_manifest.json entries schema
    """
    entry_type: str = entry.get("type", "file")
    expected_hash: str = entry["sha256"]

    # Resolve the path relative to project root.
    # Manifest keys may end with "/" for directories; strip that.
    clean_key = entry_key.rstrip("/")
    target = project_root / clean_key

    if not target.exists():
        return "MISSING"

    try:
        if entry_type == "directory":
            exclusions: list[str] = entry.get("log_exclusions", [])
            actual_hash = _sha256_directory(target, exclusions)
        else:
            actual_hash = _sha256_file(target)
    except OSError as exc:
        _logger.error(
            "sacred_guard_io_error",
            entry=entry_key,
            error=str(exc),
        )
        return "FAIL"

    return "PASS" if actual_hash == expected_hash else "FAIL"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_manifest(manifest_path: Path | None = None) -> dict[str, str]:
    """Verify every entry in ``.claude/sacred_manifest.json``.

    The manifest is loaded from disk on each call so that rebaselines are
    picked up without restarting the process (important during CI runs that
    update the manifest before running verification).

    Args:
        manifest_path: Path to the JSON manifest file.  Defaults to
            ``<project_root>/.claude/sacred_manifest.json``.  Pass a
            different path in unit tests to use a fixture manifest without
            touching the real one.

    Returns:
        A ``dict[str, str]`` with one key per manifest entry (the entry's
        path string as written in the manifest, e.g. ``"scripts/apin/"``).
        Each value is one of:

        - ``"PASS"``    – SHA-256 matches the manifest record.
        - ``"FAIL"``    – SHA-256 mismatch: the file or directory has drifted.
        - ``"MISSING"`` – The path does not exist on disk.

    # spec: section 2.6 + .claude/sacred_manifest.json
    #        directory_hash_algorithm_canonical + DEC-019 log_exclusions extension
    """
    resolved_manifest = manifest_path if manifest_path is not None else _DEFAULT_MANIFEST_PATH

    # Determine project root from the manifest location.
    # When a custom manifest_path is supplied (unit tests), the project root
    # is the parent of the manifest file's parent directory unless the test
    # arranges it differently.  For the default path the root is _PROJECT_ROOT.
    if manifest_path is None:
        project_root = _PROJECT_ROOT
    else:
        # For test fixtures the project_root defaults to the CWD so that
        # relative paths in the fixture manifest resolve against the test's
        # temporary directory tree.
        project_root = Path.cwd()

    try:
        with open(resolved_manifest, "r", encoding="utf-8") as fh:
            manifest_data: dict[str, Any] = json.load(fh)
    except FileNotFoundError:
        _logger.error(
            "sacred_guard_manifest_missing",
            manifest_path=str(resolved_manifest),
        )
        return {}
    except json.JSONDecodeError as exc:
        _logger.error(
            "sacred_guard_manifest_invalid_json",
            manifest_path=str(resolved_manifest),
            error=str(exc),
        )
        return {}

    entries: dict[str, Any] = manifest_data.get("entries", {})
    results: dict[str, str] = {}

    for entry_key, entry in entries.items():
        status = _check_entry(entry_key, entry, project_root)
        results[entry_key] = status
        _logger.debug(
            "sacred_guard_entry_checked",
            entry=entry_key,
            status=status,
        )

    passed = sum(1 for s in results.values() if s == "PASS")
    failed = sum(1 for s in results.values() if s == "FAIL")
    missing = sum(1 for s in results.values() if s == "MISSING")

    _logger.info(
        "sacred_guard_summary",
        total=len(results),
        passed=passed,
        failed=failed,
        missing=missing,
        all_pass=(failed == 0 and missing == 0),
    )

    return results
