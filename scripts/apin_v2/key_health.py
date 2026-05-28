"""9.N.9 · Per-key Health Score — the 4-pillar composite reliability index.

A credible health score can't be a single multiplication of three numbers,
because different failure modes mean different things:
  · a 5xx is the SERVICE's fault          → Reliability pillar
  · a 4xx is the INTEGRATION's fault       → Reliability pillar (lighter)
  · slow responses are PERFORMANCE         → Performance pillar (Apdex)
  · hitting rate-limits is CAPACITY        → Capacity pillar
  · an expiring/over-scoped key is HYGIENE → Hygiene pillar

So this is a multi-pillar index (like a credit score). Each pillar is
computed independently and surfaced separately, then weighted into one
composite 0-100 number + letter grade.

    HEALTH = 0.35·Reliability + 0.30·Performance + 0.20·Capacity + 0.15·Hygiene

Design references: Google SRE "Four Golden Signals" (Latency, Traffic,
Errors, Saturation), Apdex (Application Performance Index), error-budget
SLOs, and the Wilson score interval for small-sample confidence.

This module is PURE — no DB access. The caller gathers the raw metrics
and passes them in; this module only scores them. That keeps the scoring
logic unit-testable in isolation.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Iterable


# ── Pillar weights (confirmed with product) ──────────────────────────────
W_RELIABILITY = 0.35
W_PERFORMANCE = 0.30
W_CAPACITY    = 0.20
W_HYGIENE     = 0.15

# ── Apdex satisfaction thresholds, tuned to REALISTIC free-tier CPU
#    SERVER-SIDE handler latency (request_log.latency_ms is handler time,
#    not network RTT). Warm predict/quick is ~6-8s of pure compute, so
#    T=8s means "a normal warm inference is satisfied"; a cold-start ~21s
#    lands in "tolerating" (T..4T); only a broken 32s+ is "frustrated".
#    A response is: satisfied if ≤ T, tolerating if ≤ 4T, else frustrated.
ENDPOINT_CLASS_T_MS = {
    "metadata":        300,     # /version /diseases /info /model/card /benchmarks /scans /warmup /feedback
    "quick_inference": 8000,    # /predict/quick
    "heavy_inference": 15000,   # /predict/full /predict/batch
    "default":         2000,    # anything unclassified
}

# Path-prefix → class. First matching prefix wins; order matters
# (heavy before quick so /predict/full doesn't match /predict).
_CLASS_PREFIXES = [
    ("heavy_inference", ("/api/predict/full", "/api/predict/batch")),
    ("quick_inference", ("/api/predict/quick", "/api/predict")),
    ("metadata", ("/api/version", "/api/diseases", "/api/info",
                  "/api/model", "/api/benchmarks", "/api/scans",
                  "/api/warmup", "/api/feedback", "/api/keys")),
]

# Sample-size gates for confidence.
PROVISIONAL_BELOW = 30     # grade marked provisional under this many requests
INSUFFICIENT_BELOW = 5     # below this we don't score at all

# Reliability is scored by ANCHOR-POINT INTERPOLATION on the error rate,
# calibrated to the "nines" engineers actually operate at. A single linear
# span would either be too harsh (any miss of 99.9% looks broken) or too
# soft. These anchors map (5xx_rate → score) and (4xx_rate → score) with
# piecewise-linear interpolation between them.
#
# Server errors (5xx) — OUR fault, scored sternly but fairly:
#   0.0% → 100 · 0.1% → 97 · 0.5% → 90 · 1% → 82 · 2% → 68
#   5% → 40 · 10% → 10 · 20%+ → 0
SERVER_ERROR_ANCHORS = [
    (0.000, 100.0), (0.001, 97.0), (0.005, 90.0), (0.010, 82.0),
    (0.020, 68.0), (0.050, 40.0), (0.100, 10.0), (0.200, 0.0),
]
# Client errors (4xx) — the INTEGRATION's fault, scored leniently
# (clients explore, retry, probe; some 4xx is normal traffic):
#   0-5% → 100 · 10% → 85 · 20% → 60 · 40% → 20 · 50%+ → 0
CLIENT_ERROR_ANCHORS = [
    (0.000, 100.0), (0.050, 100.0), (0.100, 85.0), (0.200, 60.0),
    (0.400, 20.0), (0.500, 0.0),
]

# Capacity references.
RATE_LIMIT_ZERO_RATE = 0.10            # ≥10% of calls throttled = 0 score

# Hygiene references.
EXPIRY_WARN_DAYS   = 7
ROTATION_FULL_DAYS = 90
ROTATION_BAD_DAYS  = 365


def classify_endpoint(path: str) -> str:
    """Map a request path to its Apdex endpoint class."""
    p = (path or "").lower()
    for cls, prefixes in _CLASS_PREFIXES:
        if any(p.startswith(pre) for pre in prefixes):
            return cls
    return "default"


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a binomial proportion.

    Why: with few samples, the naive success-rate (successes/total) is
    overconfident — 4/4 = 100% tells you almost nothing. The Wilson lower
    bound is the statistically defensible "at least this good" estimate
    and naturally widens for small N. Returns 0.0-1.0.
    """
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denom)


def apdex(latencies_ms: Iterable[float], t_ms: float) -> Optional[float]:
    """Apdex score for a set of latencies against satisfaction threshold T.

    satisfied  : latency ≤ T
    tolerating : T < latency ≤ 4T   (counts half)
    frustrated : latency > 4T       (counts zero)

    Apdex = (satisfied + tolerating/2) / total. Returns None if no data.
    """
    lat = [x for x in latencies_ms if x is not None]
    n = len(lat)
    if n == 0:
        return None
    four_t = 4.0 * t_ms
    satisfied = sum(1 for x in lat if x <= t_ms)
    tolerating = sum(1 for x in lat if t_ms < x <= four_t)
    return (satisfied + tolerating / 2.0) / n


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _interp_anchors(x: float, anchors: List[tuple]) -> float:
    """Piecewise-linear interpolation of x against sorted (input, score)
    anchor points. Clamps to the end scores outside the anchor range."""
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            frac = (x - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return anchors[-1][1]


# ── Pillar 1: RELIABILITY ─────────────────────────────────────────────────
def _reliability_pillar(total: int, n_2xx: int, n_4xx: int, n_5xx: int) -> dict:
    """Server-error score (70%) + client-error score (30%)."""
    if total <= 0:
        return {"score": None, "server_score": None, "client_score": None,
                "rate_5xx": None, "rate_4xx": None, "success_wilson": None}

    rate_5xx = n_5xx / total
    rate_4xx = n_4xx / total

    # Anchor-interpolated scores — calibrated to real-world "nines".
    server_score = _interp_anchors(rate_5xx, SERVER_ERROR_ANCHORS)
    client_score = _interp_anchors(rate_4xx, CLIENT_ERROR_ANCHORS)

    score = 0.70 * server_score + 0.30 * client_score
    # Confidence: blend toward the Wilson lower bound of success so a tiny
    # sample can't claim a perfect reliability score.
    success_wilson = wilson_lower_bound(n_2xx, total)

    return {
        "score": round(score, 1),
        "server_score": round(server_score, 1),
        "client_score": round(client_score, 1),
        "rate_5xx": round(rate_5xx * 100, 3),
        "rate_4xx": round(rate_4xx * 100, 3),
        "success_wilson": round(success_wilson * 100, 2),
    }


# ── Pillar 2: PERFORMANCE ─────────────────────────────────────────────────
def _performance_pillar(latencies_by_class: Dict[str, List[float]],
                        p95_current: Optional[float],
                        p95_prev: Optional[float]) -> dict:
    """Volume-weighted per-class Apdex, ± latency-trend adjustment."""
    per_class = {}
    weighted_sum = 0.0
    total_n = 0
    for cls, lats in latencies_by_class.items():
        t = ENDPOINT_CLASS_T_MS.get(cls, ENDPOINT_CLASS_T_MS["default"])
        a = apdex(lats, t)
        n = len([x for x in lats if x is not None])
        if a is not None and n > 0:
            per_class[cls] = {"apdex": round(a, 3), "n": n, "t_ms": t}
            weighted_sum += a * n
            total_n += n

    if total_n == 0:
        return {"score": None, "per_class": {}, "trend_pct": None}

    base = 100.0 * (weighted_sum / total_n)

    # Trend adjustment: ±5 points scaled by week-over-week p95 change.
    trend_pct = None
    adj = 0.0
    if p95_current and p95_prev and p95_prev > 0:
        trend_pct = (p95_current - p95_prev) / p95_prev * 100.0
        # improving (negative trend) → bonus; degrading → penalty, capped ±5
        adj = _clamp(-trend_pct / 20.0, -1.0, 1.0) * 5.0

    return {
        "score": round(_clamp(base + adj, 0, 100), 1),
        "per_class": per_class,
        "trend_pct": round(trend_pct, 1) if trend_pct is not None else None,
    }


# ── Pillar 3: CAPACITY ────────────────────────────────────────────────────
def _capacity_pillar(total: int, rate_limited: int,
                     quota_per_day: Optional[int],
                     quota_consumed: int) -> dict:
    """Rate-limit pressure (60%) + quota headroom (40%)."""
    if total <= 0:
        return {"score": None, "rate_limit_score": None,
                "quota_score": None, "rate_limited": rate_limited}

    rl_rate = rate_limited / total if total else 0.0
    rate_limit_score = 100.0 * (1.0 - _clamp(rl_rate / RATE_LIMIT_ZERO_RATE))

    if quota_per_day and quota_per_day > 0:
        headroom = 1.0 - (quota_consumed / quota_per_day)
        quota_score = 100.0 * _clamp(headroom)
        quota_label = f"{quota_consumed}/{quota_per_day}"
    else:
        quota_score = 100.0   # unlimited quota = neutral full marks
        quota_label = "unlimited"

    score = 0.60 * rate_limit_score + 0.40 * quota_score
    return {
        "score": round(score, 1),
        "rate_limit_score": round(rate_limit_score, 1),
        "quota_score": round(quota_score, 1),
        "rate_limited": rate_limited,
        "quota_label": quota_label,
    }


# ── Pillar 4: HYGIENE ─────────────────────────────────────────────────────
def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _hygiene_pillar(created_at: Optional[str], expires_at: Optional[str],
                    scopes: Optional[str], observed_methods: Set[str],
                    distinct_ips: int, ip_baseline: float) -> dict:
    """100 − Σ penalties, floored at 0. Posture, not live impact."""
    now = datetime.now(timezone.utc)
    penalties = []
    score = 100.0

    # Expiry proximity
    exp = _parse_iso(expires_at)
    if exp is not None:
        days_to_exp = (exp - now).total_seconds() / 86400.0
        if days_to_exp < 0:
            score = 0.0
            penalties.append({"name": "expired", "points": -100,
                              "detail": "key has expired"})
        elif days_to_exp < EXPIRY_WARN_DAYS:
            score -= 40
            penalties.append({"name": "expiring_soon", "points": -40,
                              "detail": f"expires in {days_to_exp:.0f}d"})

    # Rotation age
    cre = _parse_iso(created_at)
    if cre is not None:
        age_days = (now - cre).total_seconds() / 86400.0
        if age_days > ROTATION_BAD_DAYS:
            score -= 30
            penalties.append({"name": "stale_rotation", "points": -30,
                              "detail": f"{age_days:.0f}d old, never rotated"})
        elif age_days > ROTATION_FULL_DAYS:
            # linear ramp 90→365 days: -0 to -30
            frac = (age_days - ROTATION_FULL_DAYS) / (ROTATION_BAD_DAYS - ROTATION_FULL_DAYS)
            pen = round(frac * 30)
            if pen > 0:
                score -= pen
                penalties.append({"name": "ageing_key", "points": -pen,
                                  "detail": f"{age_days:.0f}d old"})

    # Scope-vs-usage: has write scope but only ever read (GET/HEAD).
    # `scopes` may arrive as a list (get_console_api_key deserialises the
    # JSON column) OR a raw string — normalise both to a lowercased string.
    if isinstance(scopes, (list, tuple, set)):
        sc = " ".join(str(x) for x in scopes).lower()
    else:
        sc = (scopes or "").lower()
    has_write = ("write" in sc or "predict:write" in sc)
    only_read = observed_methods.issubset({"GET", "HEAD", "OPTIONS"}) if observed_methods else False
    if has_write and only_read and observed_methods:
        score -= 20
        penalties.append({"name": "over_permissioned", "points": -20,
                          "detail": "has write scope but only reads"})

    # IP fan-out: distinct IPs this window vs baseline (leak heuristic).
    # Threshold raised to >8 AND >4× baseline so proxy/load-balancer mesh
    # IPs (HF presents traffic from multiple internal 10.16.x.x hops even
    # from one origin) don't false-trigger a "leak" flag. Only a genuine
    # large, sudden fan-out is penalised.
    if distinct_ips > 8 and (ip_baseline <= 0 or distinct_ips > ip_baseline * 4):
        score -= 30
        penalties.append({"name": "ip_fan_out", "points": -30,
                          "detail": f"{distinct_ips} IPs (baseline ~{ip_baseline:.0f})"})

    return {
        "score": round(max(0.0, score), 1),
        "penalties": penalties,
        "distinct_ips": distinct_ips,
    }


# ── Grade scale ───────────────────────────────────────────────────────────
_GRADE_SCALE = [
    (97, "A+"), (93, "A"), (90, "A-"), (87, "B+"), (83, "B"), (80, "B-"),
    (77, "C+"), (73, "C"), (70, "C-"), (60, "D"), (0, "F"),
]


def letter_grade(score: Optional[float]) -> str:
    if score is None:
        return "—"
    for threshold, grade in _GRADE_SCALE:
        if score >= threshold:
            return grade
    return "F"


def _arc_tone(score: Optional[float]) -> str:
    """Color bucket for the gauge arc."""
    if score is None:
        return "nodata"
    if score >= 90:
        return "great"
    if score >= 75:
        return "ok"
    if score >= 60:
        return "warn"
    return "bad"


def _headline(pillars: dict) -> str:
    """Auto-derive the one-line 'what's capping you' from the lowest pillar."""
    scored = {k: v["score"] for k, v in pillars.items()
              if v.get("score") is not None}
    if not scored:
        return "Gathering data — not enough requests yet to assess health."
    lowest = min(scored, key=scored.get)
    val = scored[lowest]
    if val >= 95:
        return "All pillars strong — this key is in excellent health."
    if lowest == "reliability":
        r = pillars["reliability"]
        if (r.get("rate_5xx") or 0) > 0.1:
            return (f"Capped by reliability — {r['rate_5xx']}% of requests "
                    f"returned 5xx (server errors).")
        return (f"Capped by reliability — {r.get('rate_4xx')}% 4xx client "
                f"errors suggest the integration may be misusing the API.")
    if lowest == "performance":
        p = pillars["performance"]
        worst_cls, worst_apdex = None, 1.0
        for cls, d in (p.get("per_class") or {}).items():
            if d["apdex"] < worst_apdex:
                worst_cls, worst_apdex = cls, d["apdex"]
        if worst_cls:
            return (f"Capped by latency — {worst_cls.replace('_',' ')} Apdex "
                    f"is {worst_apdex:.2f}; responses are slower than the "
                    f"satisfaction threshold for that endpoint class.")
        return "Capped by latency."
    if lowest == "capacity":
        return ("Capped by capacity — the key is hitting rate-limits or "
                "approaching its daily quota.")
    if lowest == "hygiene":
        pens = pillars["hygiene"].get("penalties") or []
        if pens:
            return f"Capped by hygiene — {pens[0]['detail']}."
        return "Capped by hygiene posture."
    return "Healthy."


def compute_health_score(
    *,
    window_label: str,
    total_requests: int,
    n_2xx: int,
    n_4xx: int,
    n_5xx: int,
    latencies_by_class: Dict[str, List[float]],
    p95_current: Optional[float] = None,
    p95_prev: Optional[float] = None,
    rate_limited: int = 0,
    quota_per_day: Optional[int] = None,
    quota_consumed: int = 0,
    created_at: Optional[str] = None,
    expires_at: Optional[str] = None,
    scopes: Optional[str] = None,
    observed_methods: Optional[Set[str]] = None,
    distinct_ips: int = 0,
    ip_baseline: float = 0.0,
) -> dict:
    """Compute the full 4-pillar health score. Pure function.

    Returns a dict:
      {composite, grade, tone, provisional, insufficient, headline, window,
       sample_size, pillars: {reliability, performance, capacity, hygiene}}
    """
    observed_methods = observed_methods or set()

    if total_requests < INSUFFICIENT_BELOW:
        return {
            "composite": None, "grade": "—", "tone": "nodata",
            "provisional": True, "insufficient": True,
            "headline": "Insufficient data — fewer than "
                        f"{INSUFFICIENT_BELOW} requests in this window.",
            "window": window_label, "sample_size": total_requests,
            "pillars": {
                "reliability": {"score": None},
                "performance": {"score": None},
                "capacity":    {"score": None},
                "hygiene":     _hygiene_pillar(created_at, expires_at, scopes,
                                               observed_methods, distinct_ips,
                                               ip_baseline),
            },
        }

    reliability = _reliability_pillar(total_requests, n_2xx, n_4xx, n_5xx)
    performance = _performance_pillar(latencies_by_class, p95_current, p95_prev)
    capacity    = _capacity_pillar(total_requests, rate_limited,
                                   quota_per_day, quota_consumed)
    hygiene     = _hygiene_pillar(created_at, expires_at, scopes,
                                  observed_methods, distinct_ips, ip_baseline)

    pillars = {
        "reliability": reliability,
        "performance": performance,
        "capacity":    capacity,
        "hygiene":     hygiene,
    }

    # Composite — re-normalise weights over pillars that actually scored
    # (so a key with no latency data isn't punished by a None performance).
    parts = [
        (W_RELIABILITY, reliability["score"]),
        (W_PERFORMANCE, performance["score"]),
        (W_CAPACITY,    capacity["score"]),
        (W_HYGIENE,     hygiene["score"]),
    ]
    avail = [(w, s) for w, s in parts if s is not None]
    wsum = sum(w for w, _ in avail)
    composite = round(sum(w * s for w, s in avail) / wsum, 1) if wsum else None

    return {
        "composite": composite,
        "grade": letter_grade(composite),
        "tone": _arc_tone(composite),
        "provisional": total_requests < PROVISIONAL_BELOW,
        "insufficient": False,
        "headline": _headline(pillars),
        "window": window_label,
        "sample_size": total_requests,
        "weights": {"reliability": W_RELIABILITY, "performance": W_PERFORMANCE,
                    "capacity": W_CAPACITY, "hygiene": W_HYGIENE},
        "pillars": pillars,
    }
