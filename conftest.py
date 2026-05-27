"""Pytest root conftest — ensures Phase 0 mandatory env vars are set before
ANY test module imports `scripts.apin_v2.*`.

Background (PDA-P0-R1-F04):
    `scripts/apin_v2/apin_server.py` raises `RuntimeError` at module import
    time when `IP_HASH_SALT` is absent or shorter than 16 chars. This was
    introduced to remove a hardcoded salt fallback that weakened the privacy
    contract (REV-R2-I10 + §22.3 row 6). The side effect: any test that
    imports the server module without first setting the env var crashes at
    collection time, before pytest can even report a useful traceback.

    This file runs BEFORE pytest collects test modules (`conftest.py` at the
    rootdir is loaded eagerly). It sets a deterministic, test-only salt so
    every test starts from the same hash space. Real deployments use the
    operator-supplied value from `.env` — this default ONLY applies when no
    operator value is present in the process environment.

Determinism: the test salt is constant across runs so hash comparisons in
fixtures don't flake. It is intentionally LONGER than 16 chars to also pass
the minimum-length check.

This file does NOT load real `.env` values — pytest tests should be hermetic
against the developer's local environment.
"""
import os

# Test-only salt. Never used in production — apin_server.py reads from env
# at import time; this just ensures `os.environ` has a value before that
# import happens.
_TEST_SALT = "test-ip-hash-salt-deterministic-32chars"

os.environ.setdefault("IP_HASH_SALT", _TEST_SALT)
os.environ.setdefault("APIN_TRUSTED_PROXY_HOPS", "1")
