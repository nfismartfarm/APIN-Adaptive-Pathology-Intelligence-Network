"""Daily Field Brief generator.

Produces a 2-3 sentence narrative paragraph summarising the user's recent
fieldwork. The goal is for the brief to *feel* unique each time the page
is loaded (and across days) without invoking a real LLM — useful while
the model isn't wired up.

Design:
    * Three independent slot grammars (opener / middle / closer).
    * Each slot has 6-10 templates; templates reference data signals via
      {placeholders} that the renderer fills.
    * The (template selection, placeholder values) tuple is salted with a
      hash of (date + user_id + signature), so the SAME data on the SAME
      day produces the SAME brief (idempotent for a given visitor) but
      different days / different data produce different briefs.
    * If a real LLM gets wired up later, swap the `generate_brief()`
      function — its inputs and output shape stay identical.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional


# ───────────────────────── Template library ────────────────────────────────
#
# Each template uses Python str.format named placeholders. Available keys:
#   {n_today}         int   — predictions logged today
#   {n_week}          int   — predictions logged in last 7 days
#   {n_last_week}     int   — predictions logged in the previous 7-day window
#   {top_disease}     str   — most-common disease this week, human-readable
#   {top_count}       int   — how many times {top_disease} occurred
#   {top_delta}       str   — "↑ from X last week" / "↓ from X" / "first sightings"
#   {avg_conf}        float — average confidence (0..1)
#   {tier_descr}      str   — "field-grade" / "lab-grade" / "mostly borderline"
#   {streak_days}     int   — current daily streak
#   {streak_phrase}   str   — "1st day"/"3-day streak"/"steady twelve-day run"
#   {weekday}         str   — "Monday" .. "Sunday"
#   {date_phrase}     str   — "early in the morning" / "midday" / etc.
#   {dominant_crop}   str   — "tomato" / "okra" / etc.

_OPENERS = [
    "Logged {n_today} specimen{s_today} today.",
    "A {weekday} of fieldwork — {n_today} new entr{ies_today} in the journal.",
    "Quietly added {n_today} page{s_today} to the journal this {weekday}.",
    "Today's harvest: {n_today} specimen{s_today} catalogued.",
    "The {weekday} {date_phrase} brought {n_today} new entr{ies_today}.",
    "{n_today_word} fresh specimen{s_today} pressed into the journal today.",
    "Another {weekday}; {n_today} new leaves preserved.",
]

_OPENERS_EMPTY = [
    "No new specimens today — yet.",
    "A quiet {weekday} so far; the journal awaits.",
    "Nothing pressed into the journal today.",
    "{weekday} stands empty in the ledger.",
]

_MIDDLES_TREND_UP = [
    "{top_disease} continues to dominate the ledger ({top_count} entr{ies_top} this week, {top_delta}).",
    "{top_disease} keeps appearing — {top_count} this week ({top_delta}).",
    "The week's recurring guest is {top_disease}: {top_count} entr{ies_top}, {top_delta}.",
]

_MIDDLES_TREND_DOWN = [
    "{top_disease} has eased off — {top_count} this week ({top_delta}).",
    "Fewer {top_disease} sightings this week ({top_count}, {top_delta}).",
    "{top_disease} appears to be retreating ({top_count} entr{ies_top}, {top_delta}).",
]

_MIDDLES_TREND_FLAT = [
    "The ledger is led by {top_disease} ({top_count} entr{ies_top} this week).",
    "{top_disease} stays at the top of the chart — {top_count} entr{ies_top} this week.",
    "Most photographed disease this week: {top_disease} ({top_count}).",
]

_MIDDLES_FIRST = [
    "First time seeing {top_disease} this season — worth a closer look.",
    "{top_disease} arrived in the journal this week ({top_count} entr{ies_top}, new).",
]

_MIDDLES_NEUTRAL = [
    "{n_week} entries this week, against {n_last_week} last week.",
    "{n_week} pages added this week ({n_last_week} the week before).",
    "Weekly tally: {n_week} specimens — last week {n_last_week}.",
]

_CLOSERS = [
    "Average confidence held at {avg_conf} — {tier_descr} calls.",
    "Confidence sits at {avg_conf}; {tier_descr} territory.",
    "Calls averaged {avg_conf} — {tier_descr}.",
    "The model held {avg_conf} on average, {tier_descr}.",
]

_CLOSERS_STREAK = [
    "{streak_phrase}: keep the rhythm.",
    "A {streak_phrase} so far — quietly steady.",
    "Fieldwork now into a {streak_phrase}.",
]


# ───────────────────────── Salt / picker ───────────────────────────────────

def _pick(items: list, salt: str, key: str) -> Any:
    """Deterministic choice from `items` given a salt string."""
    h = hashlib.sha256((salt + "|" + key).encode("utf-8")).hexdigest()
    return items[int(h[:8], 16) % len(items)]


def _pluralise(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


# ───────────────────────── Helpers ─────────────────────────────────────────

_TIER_BY_AVG = [
    (0.85, "field-grade"),
    (0.70, "lab-grade"),
    (0.50, "borderline"),
    (0.00, "low-confidence"),
]


def _tier_descr(avg_conf: float) -> str:
    for thresh, label in _TIER_BY_AVG:
        if avg_conf >= thresh:
            return label
    return "low-confidence"


def _streak_phrase(n: int) -> str:
    if n <= 0:
        return ""
    if n == 1:
        return "first day of a fresh streak"
    if n < 4:
        return f"{n}-day streak"
    if n < 8:
        return f"{n}-day run"
    return f"steady {n}-day stretch"


_WORDS_LOW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def _to_word(n: int) -> str:
    if 0 <= n < len(_WORDS_LOW):
        return _WORDS_LOW[n].capitalize()
    return str(n)


def _date_phrase(now: datetime) -> str:
    h = now.hour
    if   h < 5:  return "well before dawn"
    elif h < 11: return "in the morning"
    elif h < 14: return "around midday"
    elif h < 17: return "in the afternoon"
    elif h < 20: return "into the evening"
    else:        return "late into the night"


def _humanise_disease(class_name: Optional[str]) -> str:
    if not class_name:
        return "an unidentified leaf"
    # tomato_early_blight → "tomato early blight"
    return class_name.replace("_", " ")


# ───────────────────────── Public API ──────────────────────────────────────

def generate_brief(
    *,
    user_id: int,
    n_today: int,
    n_week: int,
    n_last_week: int,
    top_disease: Optional[str],
    top_count: int,
    top_last_week_count: int,
    avg_confidence: float,
    streak_days: int,
    dominant_crop: Optional[str],
    now: Optional[datetime] = None,
) -> dict:
    """Generate a 2-3 sentence Daily Field Brief.

    Returns a dict {"text": str, "salt": str, "tokens": dict} so the frontend
    can attribute the brief to a deterministic salt (useful for debugging
    "why does my brief read like this?").

    The brief is composed of three slots:
        1. Opener  — references today's count + weekday + time-of-day
        2. Middle  — references trend in top disease or week-over-week
        3. Closer  — references confidence level + (optionally) streak

    A real LLM can later replace this function; the inputs and the
    output `text` field are stable.
    """
    now = now or datetime.now(timezone.utc)
    weekday = now.strftime("%A")

    # Compose the deterministic salt: same user + same day + same key facts
    # → same brief. Keys deliberately chosen so a brief refreshes when ANY
    # underlying number changes.
    salt = "|".join([
        now.strftime("%Y-%m-%d"),
        str(user_id),
        f"t={n_today}",
        f"w={n_week}",
        f"lw={n_last_week}",
        f"top={top_disease or '-'}",
        f"tc={top_count}",
        f"conf={round(avg_confidence, 2)}",
        f"streak={streak_days}",
    ])

    # ── Pick slot 1 (opener) ────────────────────────────────────────────────
    if n_today == 0:
        opener_tpl = _pick(_OPENERS_EMPTY, salt, "opener")
    else:
        opener_tpl = _pick(_OPENERS, salt, "opener")

    # ── Pick slot 2 (middle) ────────────────────────────────────────────────
    has_top = bool(top_disease) and top_count > 0
    delta = top_count - top_last_week_count
    if not has_top:
        middle_tpl = _pick(_MIDDLES_NEUTRAL, salt, "middle")
        top_delta = ""
    elif top_last_week_count == 0:
        middle_tpl = _pick(_MIDDLES_FIRST, salt, "middle")
        top_delta = "new this week"
    elif delta > 0:
        middle_tpl = _pick(_MIDDLES_TREND_UP, salt, "middle")
        top_delta = f"up from {top_last_week_count} last week"
    elif delta < 0:
        middle_tpl = _pick(_MIDDLES_TREND_DOWN, salt, "middle")
        top_delta = f"down from {top_last_week_count} last week"
    else:
        middle_tpl = _pick(_MIDDLES_TREND_FLAT, salt, "middle")
        top_delta = f"steady at {top_count}"

    # ── Pick slot 3 (closer) ────────────────────────────────────────────────
    streak_phrase = _streak_phrase(streak_days)
    if streak_days >= 3 and (int(hashlib.sha256(salt.encode()).hexdigest()[:2], 16) % 2 == 0):
        closer_tpl = _pick(_CLOSERS_STREAK, salt, "closer")
    else:
        closer_tpl = _pick(_CLOSERS, salt, "closer")

    # ── Render with placeholder substitution ────────────────────────────────
    tokens = {
        "n_today":           n_today,
        "n_today_word":      _to_word(n_today),
        "s_today":           _pluralise(n_today, "", "s"),
        "ies_today":         _pluralise(n_today, "y", "ies"),
        "n_week":            n_week,
        "n_last_week":       n_last_week,
        "top_disease":       _humanise_disease(top_disease),
        "top_count":         top_count,
        "ies_top":           _pluralise(top_count, "y", "ies"),
        "top_delta":         top_delta,
        "avg_conf":          f"{avg_confidence:.2f}",
        "tier_descr":        _tier_descr(avg_confidence),
        "streak_days":       streak_days,
        "streak_phrase":     streak_phrase,
        "weekday":           weekday,
        "date_phrase":       _date_phrase(now),
        "dominant_crop":     dominant_crop or "—",
    }

    sentences = []
    for tpl in (opener_tpl, middle_tpl, closer_tpl):
        try:
            sentences.append(tpl.format(**tokens))
        except KeyError:
            # If a template references an unknown token, skip rather than crash.
            continue

    text = " ".join(sentences).strip()

    return {
        "text":   text,
        "salt":   salt[:16],   # short version for debugging only
        "tokens": tokens,
    }


def build_brief_for_user(user_id: int, db_module, *,
                         dashboard_data: dict | None = None) -> dict:
    """Convenience wrapper: pulls the stats it needs from `db_module`
    (typically `auth_db`) and returns a brief dict.

    Caller may pass `dashboard_data=auth_db.get_dashboard_data(user_id)` if
    they have it already — this avoids re-running 6 aggregation queries
    when the brief is built inside the /dashboard/data route handler that
    just computed it. If not supplied, we fetch it ourselves.

    Query budget when dashboard_data is supplied:
        - count_predictions × 3 (today / this week / last week)
        - aggregate_by_disease × 1 with date_from filter (top disease this week)
        - count_predictions × 1 (top disease last week, only if a top was found)
        Total: 4-5 queries. (Was 11 when we recomputed dashboard_data.)

    Semantic fix (code-review finding #8): the previous version filtered the
    full all-time disease aggregate by `last_seen >= week_start`, which
    counts ALL-TIME occurrences of any disease whose most recent sighting
    happens to fall this week. That over-credited diseases with a single
    recent appearance and a long historical tail. The correct filter is to
    count occurrences IN this week, which the new `date_from` parameter on
    aggregate_by_disease does in SQL."""
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    week_start = (now.date() - timedelta(days=6)).isoformat()
    last_week_start = (now.date() - timedelta(days=13)).isoformat()
    last_week_end = (now.date() - timedelta(days=7)).isoformat()

    # ── Date-window counts ────────────────────────────────────────────────
    n_today = db_module.count_predictions(
        user_id, date_from=today, date_to=today,
    )
    n_week = db_module.count_predictions(
        user_id, date_from=week_start, date_to=today,
    )
    n_last_week = db_module.count_predictions(
        user_id, date_from=last_week_start, date_to=last_week_end,
    )

    # ── Top disease *this week* — counted strictly within the week ───────
    # The aggregate now does the date filtering in SQL via date_from/date_to.
    top_disease = None
    top_count = 0
    top_last_week_count = 0
    week_agg = db_module.aggregate_by_disease(
        user_id, date_from=week_start, date_to=today, limit=10,
    )
    if week_agg:
        top = week_agg[0]
        top_disease = top["class"]
        top_count = top["count"]  # already the week-only count
        top_last_week_count = db_module.count_predictions(
            user_id, disease=top_disease,
            date_from=last_week_start, date_to=last_week_end,
        )

    # ── Reuse caller-supplied dashboard payload to avoid 6 extra queries ─
    if dashboard_data is None:
        dashboard_data = db_module.get_dashboard_data(user_id)
    avg_confidence = float(dashboard_data.get("hero", {}).get("avg_confidence", 0.0))
    streak_days = int(dashboard_data.get("hero", {}).get("streak_days", 0))
    dominant_crop = dashboard_data.get("dominant_crop")

    return generate_brief(
        user_id=user_id,
        n_today=n_today,
        n_week=n_week,
        n_last_week=n_last_week,
        top_disease=top_disease,
        top_count=top_count,
        top_last_week_count=top_last_week_count,
        avg_confidence=avg_confidence,
        streak_days=streak_days,
        dominant_crop=dominant_crop,
        now=now,
    )
