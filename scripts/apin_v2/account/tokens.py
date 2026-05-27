"""API Console token generation, hashing, and display helpers.

Spec contract: spec_v7.md §4 (Token Format, Lifecycle, One-Time View)

Public surface (for callers OUTSIDE the apin_v2.account package):
    ALPHABET             — 62-char base62 alphabet (digits, upper, lower)
    BASE                 — 62
    SECRET_LEN_NEW       — 43 (chars of base62 secret in new tokens)
    SECRET_LEN_LEGACY    — 32 (chars of hex secret in legacy tokens)
    TOKEN_LEN_NEW        — 53 (full new-token length: "apin_live_" + 43)
    TOKEN_LEN_LEGACY     — 37 (full legacy-token length: "apin_" + 32)
    VALID_TOKEN_REGEX    — compiled regex matching new OR legacy format
    WEBHOOK_SECRET_LEN   — 32 (base62 chars after `whsec_`)
    WEBHOOK_SECRET_PREFIX — "whsec_"
    gen_token            — env -> "apin_{env}_<43>" (53 chars total)
    gen_webhook_secret   — () -> "whsec_<32 base62>"
    token_hash           — token -> sha256 hex (64 chars)
    is_valid_token_format — regex-only format check (no DB lookup)
    last_four            — token -> last 4 chars of the secret portion
    redact_token         — token -> display-safe form (PDA-R2-F49)
    redact_webhook_secret — secret -> display-safe form

Module-private helpers (PDA-P2.1-R1-F08 — leading underscore is intentional):
    _b62_fixed_N         — bytes -> N-char base62 (PDA-R2-F54); only
                           called internally by _b62_fixed_43. The
                           leading underscore signals "implementation
                           detail of this module; downstream consumers
                           should call gen_token / gen_webhook_secret"
    _b62_fixed_43        — bytes -> 43-char base62 (called by gen_token)

Why ASCII `*` for redaction (PDA-R2-F49): renders reliably across terminals,
monospace fonts, CSV exports, JSON dumps, and assistive tech. The Unicode
bullet `•` (U+2022) sometimes breaks in CSV/log pipelines.

Why fast SHA-256 not bcrypt (PDA-F55): 256 bits of entropy in the token
itself makes brute force infeasible regardless of hash speed. The high
entropy is what makes this safe — not the hash's slowness. Fast hash also
gives O(1) auth-path lookup, which matters at scale.
"""
from __future__ import annotations

import hashlib
import re
import secrets

# ── Constants ──────────────────────────────────────────────────────────────

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE = len(ALPHABET)   # 62
assert BASE == 62, "ALPHABET must be 62 chars (base62)"

# Length of the secret-portion of each token shape.
SECRET_LEN_NEW = 43       # new tokens: "apin_live_" + 43 base62 chars
SECRET_LEN_LEGACY = 32    # legacy tokens: "apin_" + 32 hex chars

# Full token lengths for callers that want to size buffers / validate:
# new   = len("apin_live_") + 43 = 10 + 43 = 53   OR   len("apin_test_") + 43 = 53
# legacy = len("apin_")      + 32 = 5  + 32 = 37
TOKEN_LEN_NEW = len("apin_live_") + SECRET_LEN_NEW     # 53
TOKEN_LEN_LEGACY = len("apin_") + SECRET_LEN_LEGACY    # 37

# Webhook signing secret format: "whsec_<32 base62>" (PDA-R2-F54).
WEBHOOK_SECRET_PREFIX = "whsec_"
WEBHOOK_SECRET_LEN = 32     # 32 base62 chars = ~190 bits entropy
WEBHOOK_SECRET_BYTES = 24   # 24 random bytes -> 192 bits

# Token grammar (spec §4.1, R3-F12 corrected).
#   \A(apin_(?:live|test)_[0-9A-Za-z]{43} | apin_[0-9a-f]{32})\Z
# Single \A/\Z (NOT ^/$), non-capturing inner alternation.
#
# PDA-P2.1-R1-F01 fix: Python's `$` matches end-of-string OR before a final
# newline, so `re.match(r"^foo$", "foo\n")` returns True. That tolerance is
# wrong here — a Bearer header ending in `\n` should not authenticate, and
# letting the `\n` survive through to `redact_token` would also breach
# PDA-R2-F49's CSV/JSON-safety guarantee for redacted output. `\Z` matches
# only at absolute end-of-string, so this regex now refuses any trailing
# whitespace including newlines. `\A` similarly defends the left edge
# (technically `^` is safe inside `re.match` without MULTILINE, but the
# symmetry keeps the contract obvious).
VALID_TOKEN_REGEX = re.compile(
    r"\A(apin_(?:live|test)_[0-9A-Za-z]{43}|apin_[0-9a-f]{32})\Z"
)

# Display-redaction character (PDA-R2-F49). DO NOT change to U+2022 — see
# module docstring for rationale.
REDACT_CHAR = "*"


# ── Token generation ───────────────────────────────────────────────────────

def _b62_fixed_N(raw: bytes, n: int) -> str:
    """General helper: convert `raw` bytes to a fixed-length n-char base62
    string. Pads with FRESH random base62 characters (NOT the zero-alphabet
    char) to preserve entropy on the ~1/256 leading-zero-byte case.

    Spec §4.2 (PDA-R2-F54): introduced as the general form. `_b62_fixed_43`
    becomes a thin wrapper. Webhook secrets use `_b62_fixed_N(..., 32)`.

    Args:
        raw: input bytes. Must have enough entropy to produce `n` chars.
        n:   target output length. Must be > 0.

    Returns:
        n-char string from ALPHABET. Guaranteed length == n.
    """
    if n <= 0:
        raise ValueError(f"_b62_fixed_N requires n > 0, got {n}")
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError(
            f"_b62_fixed_N requires bytes input, got {type(raw).__name__}"
        )

    # Pure-base62 conversion: treat `raw` as a big-endian unsigned integer.
    num = int.from_bytes(raw, "big")
    out: list[str] = []
    while num > 0:
        num, rem = divmod(num, BASE)
        out.append(ALPHABET[rem])
    # Pad with FRESH random base62 chars to reach length n. Zero-alphabet
    # padding ("000…") would leak structural info about the input on the
    # ~1/256 leading-zero-byte case; random padding looks identical to the
    # body of the secret, so no information leaks.
    while len(out) < n:
        out.append(ALPHABET[secrets.randbelow(BASE)])
    # If somehow longer than n (huge input bytes), truncate from the right
    # (least-significant digits dropped). This matches the spec reference
    # in §4.2 line 1555 which uses `[:43]` on the reversed list (MSB-keep).
    # VER-P2.1-F02 fix: previous form `s[-n:]` kept the LSB digits, which
    # is mathematically safe but diverged from the spec; aligning the
    # truncation direction removes a real spec-implementation gap.
    #
    # Note: with the inputs we currently use (gen_token: 32 bytes -> max
    # 43 base62 chars), this branch is unreachable. gen_webhook_secret no
    # longer uses _b62_fixed_N at all (PDA-P2.1-R1-F04), so the only
    # remaining caller path is _b62_fixed_43 -> 32 bytes -> 43 chars.
    s = "".join(reversed(out))
    return s[:n] if len(s) > n else s


def _b62_fixed_43(raw: bytes) -> str:
    """Convenience alias: convert 32 random bytes to 43-char base62 string.

    Equivalent to `_b62_fixed_N(raw, 43)`. Kept as a named helper because
    spec §4.2 originally specified it before §4.2 was generalised.
    """
    return _b62_fixed_N(raw, SECRET_LEN_NEW)


def gen_token(env: str) -> str:
    """Generate a new-format token: "apin_{env}_{43 base62 chars}".

    Args:
        env: must be "live" or "test". Asserts on any other value — this is
             a programming mistake, not a runtime input.

    Returns:
        53-char ASCII string. Plaintext shown to user once; never stored.

    Implementation notes (PDA-F03 corrected):
      - The retry loop (`for _ in range(8)`) defends against the theoretical
        case where `_b62_fixed_43` returns a string of length != 43.
        With the current implementation that's unreachable, but the retry
        keeps the contract loud-on-failure rather than silent.
      - 256 bits of entropy. Brute force is infeasible regardless of hash
        algorithm; see module docstring.
    """
    if env not in ("live", "test"):
        raise ValueError(f"gen_token requires env in ('live','test'), got {env!r}")

    secret = ""
    for _ in range(8):
        raw = secrets.token_bytes(32)
        secret = _b62_fixed_43(raw)
        if len(secret) == SECRET_LEN_NEW:
            break
    # Defensive contract assertion — protects against future regression
    # in _b62_fixed_N changing the length invariant.
    if len(secret) != SECRET_LEN_NEW:
        raise RuntimeError(
            f"_b62_fixed_43 failed to produce {SECRET_LEN_NEW} chars after "
            f"8 attempts (got {len(secret)} chars). This is a programming "
            f"bug in tokens.py."
        )
    return f"apin_{env}_{secret}"


def gen_webhook_secret() -> str:
    """Generate a webhook signing secret: "whsec_{32 base62 chars}".

    Spec §6.5 + PDA-R2-F54: 32 chars of base62 = 32 * log2(62) ~= 190.5 bits
    of entropy, well above the 128-bit comfort threshold.

    PDA-P2.1-R1-F04 fix: implementation switched from
    `_b62_fixed_N(secrets.token_bytes(24), 32)` to char-by-char
    `secrets.choice(ALPHABET)`. The byte-conversion approach was subtly
    non-uniform because 62^32 < 2^192: 24-byte inputs occasionally
    produced 33 base62 chars, triggering the truncation branch ~36% of
    the time. Char-by-char sampling is the textbook uniform method:
    each char is drawn independently from `ALPHABET` via `secrets.choice`
    (CSPRNG-backed, no modular bias).

    PDA-P2.1-R1-F09 fix: a length-invariant assertion guards the output
    (matches the defensive posture of `gen_token`). The assertion is
    unreachable in practice — `secrets.choice` always returns exactly
    one char per call — but the contract is now consistent across the
    two generation functions.

    The plaintext secret is shown to the user once; the server stores
    `secret_hash` for display and `secret_encrypted` for HMAC signing.
    See spec §18.11 for the encryption-at-rest contract.
    """
    body = "".join(
        secrets.choice(ALPHABET) for _ in range(WEBHOOK_SECRET_LEN)
    )
    # Defensive length-invariant guard (consistency with gen_token).
    if len(body) != WEBHOOK_SECRET_LEN:
        raise RuntimeError(
            f"gen_webhook_secret failed to produce {WEBHOOK_SECRET_LEN} "
            f"chars (got {len(body)} — programming bug in tokens.py)"
        )
    return WEBHOOK_SECRET_PREFIX + body


# ── Hashing + lookup ───────────────────────────────────────────────────────

def token_hash(token: str) -> str:
    """SHA-256 hex digest of the token's ASCII bytes (64 hex chars).

    Used as the storage key (`api_keys.token_hash UNIQUE`) and the lookup
    key during request authentication. ASCII-only because every legal token
    contains only `0-9A-Za-z_`.

    PDA-P2.1-R1-F02 fix: non-ASCII input now raises `ValueError` with a
    clear message rather than letting `UnicodeEncodeError` propagate. This
    matters because a Bearer header containing Unicode-confusable chars
    (Cyrillic 'а' that looks like Latin 'a', etc.) would otherwise hit
    an obscure UnicodeEncodeError mid-auth-path. The expected upstream
    contract is that `TokenFormatMiddleware` rejects malformed Bearer
    headers BEFORE this function is called — but defence-in-depth makes
    the error message comprehensible if the validation chain is bypassed.
    """
    if not isinstance(token, str):
        raise TypeError(f"token_hash requires str, got {type(token).__name__}")
    try:
        encoded = token.encode("ascii")
    except UnicodeEncodeError as e:
        # Identify the offending byte position for diagnostic clarity.
        raise ValueError(
            f"token_hash requires ASCII-only input (token grammar §4.1 "
            f"allows only [0-9A-Za-z_]); got non-ASCII char at position "
            f"{e.start}. Did TokenFormatMiddleware bypass validation?"
        ) from None
    return hashlib.sha256(encoded).hexdigest()


def is_valid_token_format(token: str) -> bool:
    """Cheap regex-only check (no DB lookup) that `token` could be a real
    API key. Used by TokenFormatMiddleware to reject malformed Bearer
    headers EARLY, before they reach the auth code path."""
    if not isinstance(token, str):
        return False
    return VALID_TOKEN_REGEX.match(token) is not None


# ── Display + redaction (§4.6, PDA-R2-F49) ─────────────────────────────────

def last_four(token: str) -> str:
    """Return the last 4 chars of the secret portion of `token`.

    For new tokens ("apin_live_<43>" / "apin_test_<43>"): last 4 of the 43
    base62 chars (i.e. the literal last 4 chars of `token`).
    For legacy tokens ("apin_<32hex>"): last 4 of the 32 hex chars (also
    the literal last 4 chars of `token`).

    Returns the empty string if the token is too short to safely index.
    """
    if not isinstance(token, str) or len(token) < 4:
        return ""
    return token[-4:]


def redact_token(token: str) -> str:
    """Display-safe redaction (§4.6 / PDA-R2-F49).

    Output formats:
      "apin_live_******_<last4>"   for new live tokens
      "apin_test_******_<last4>"   for new test tokens
      "apin_******_<last4>"        for legacy tokens
      "<invalid token>"            for anything that doesn't match grammar

    Never reveals more than 4 chars of secret material. Always uses ASCII
    `*` (NOT Unicode `•`) — see module docstring (PDA-R2-F49).
    """
    if not is_valid_token_format(token):
        return "<invalid token>"
    suffix = last_four(token)
    if token.startswith("apin_live_"):
        return f"apin_live_{REDACT_CHAR * 6}_{suffix}"
    if token.startswith("apin_test_"):
        return f"apin_test_{REDACT_CHAR * 6}_{suffix}"
    # Legacy form "apin_<32hex>"
    return f"apin_{REDACT_CHAR * 6}_{suffix}"


def redact_webhook_secret(secret: str) -> str:
    """Display-safe redaction for webhook signing secrets.

    Output: "whsec_******_<last4>". Returns "<invalid webhook secret>" on
    anything that doesn't match the expected shape (PDA-R2-F49 + §15.12).

    PDA-P2.1-R1-F03 fix: now validates the BODY charset too. The previous
    form accepted `whsec_` + 32 chars of ANY character (including newlines,
    control chars, Unicode) because it only checked prefix + length. A
    malformed secret containing a newline could break the CSV/JSON output
    contract the redaction is designed to protect. Body now must be 32
    chars all from ALPHABET (base62).
    """
    if (not isinstance(secret, str)
            or not secret.startswith(WEBHOOK_SECRET_PREFIX)
            or len(secret) != len(WEBHOOK_SECRET_PREFIX) + WEBHOOK_SECRET_LEN):
        return "<invalid webhook secret>"
    body = secret[len(WEBHOOK_SECRET_PREFIX):]
    # PDA-P2.1-R1-F03: every body char must be in ALPHABET (base62).
    if not all(c in ALPHABET for c in body):
        return "<invalid webhook secret>"
    suffix = secret[-4:]
    return f"{WEBHOOK_SECRET_PREFIX}{REDACT_CHAR * 6}_{suffix}"
