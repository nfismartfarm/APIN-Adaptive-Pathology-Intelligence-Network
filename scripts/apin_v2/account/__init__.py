"""APIN API Console subpackage.

All Console-only code lives here so the existing inference surface
(scripts/apin_v2/apin_server.py, /predict, /feedback, /warmup) stays
untouched. The Console mounts under /api/account/* and uses session-cookie
auth — never Bearer / X-API-Key.

Modules:
    tokens         — token generation, format validation, SHA-256 hashing
    auth_decorator — @require_scope for endpoint protection (Phase 2.2)
    middlewares    — TokenRedaction, TokenFormat, Sudo (Phase 2.3)
    sandbox        — deterministic synthetic responses (Phase 5)
    webhook_*      — webhook delivery worker, signing, URL validator (Phase 4)
    audit_integrity — daily hash-chain check (Phase 4+)
    sse            — alerts + per-key event streams (Phase 4)
"""
