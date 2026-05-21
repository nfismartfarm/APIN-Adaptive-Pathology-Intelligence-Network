"""report_text.py — the phrase engine for the APIN weekly report.

The report's narrative is composed, not template-stamped. Each sentence is
assembled from independent fragment banks, so the variety is multiplicative
and the same data reads differently from one week to the next. Word choice
is magnitude-aware: a small rise and a steep one pick different verbs.

Voice: an agronomist's field notes. Short, plain, concrete. Deliberately
avoids the marks of generated prose:
  - no em dashes (commas, full stops and brackets instead)
  - no "it is not just X, it is Y"
  - no buzzwords (leverage, robust, seamless, underscore, delve, vital)
  - no "Moreover", "Furthermore", "Notably", "It is worth noting"
  - not every sentence the same length and shape

Determinism: a PhraseEngine is seeded by the week, so a given week always
renders identically (a report must be reproducible), while different weeks
diverge. Within one report a fragment bank does not repeat until exhausted.
"""
from __future__ import annotations

import random


# ── disease name prettifier ────────────────────────────────────────────────
_ACRONYMS = {"yvmv": "YVMV"}
_CROP_WORD = {"okra": "okra", "brassica": "brassica",
              "tomato": "tomato", "chilli": "chilli"}


def pretty_class(cls: str) -> str:
    """okra_yvmv -> 'Okra YVMV', tomato_early_blight -> 'Tomato Early Blight'."""
    if not cls:
        return "Unknown"
    parts = str(cls).split("_")
    out = []
    for p in parts:
        out.append(_ACRONYMS.get(p.lower(), p.capitalize()))
    return " ".join(out)


def disease_only(cls: str) -> str:
    """The disease part without the crop prefix: okra_yvmv -> 'YVMV'."""
    if not cls:
        return "unknown"
    parts = str(cls).split("_")
    if len(parts) > 1 and parts[0].lower() in _CROP_WORD:
        parts = parts[1:]
    return " ".join(_ACRONYMS.get(p.lower(), p) for p in parts)


def _sentence_case(s: str) -> str:
    """Capitalise the first alphabetic character of a sentence and leave the
    rest untouched, so an acronym placed mid-sentence (YVMV) is preserved.
    Used where a fragment that may start with a lower-case disease name
    (disease_only -> 'late blight') lands at the start of a sentence."""
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
        if not ch.isspace():
            break
    return s


# ── magnitude-aware word choice ────────────────────────────────────────────
# Picked by the size of a week-over-week move (ratio of curr to prev).
_RISE = {
    "slight": ["edged up", "ticked up", "crept up", "nudged higher",
               "rose a little", "inched up"],
    "clear":  ["rose", "climbed", "went up", "picked up", "moved up",
               "grew"],
    "sharp":  ["rose sharply", "jumped", "climbed steeply", "shot up",
               "spiked", "rose hard"],
}
_FALL = {
    "slight": ["edged down", "eased a little", "slipped slightly",
               "ticked down", "came off a touch", "softened"],
    "clear":  ["fell", "dropped", "came down", "eased back", "declined",
               "pulled back"],
    "sharp":  ["fell sharply", "dropped steeply", "fell away", "tumbled",
               "fell hard", "dropped right off"],
}
_FLAT = ["held steady", "stayed put", "barely moved", "was unchanged",
         "sat where it was", "held its ground"]


def _move_words(prev: float, curr: float, salt: int = 0) -> str:
    """A magnitude-aware verb phrase for a count moving prev -> curr.
    `salt` lets the caller vary the pick when two diseases happen to make
    the identical move (e.g. both 2 -> 5) so the prose does not repeat."""
    rng = random.Random(int(prev * 1000) + int(curr * 1000) + 7 + salt * 101)
    if curr == prev:
        return rng.choice(_FLAT)
    rising = curr > prev
    hi, lo = max(curr, prev), min(curr, prev)
    ratio = (hi / lo) if lo > 0 else (hi + 1)
    if ratio >= 2.4 or (lo == 0 and hi >= 3):
        band = "sharp"
    elif ratio >= 1.4:
        band = "clear"
    else:
        band = "slight"
    return rng.choice((_RISE if rising else _FALL)[band])


# ── fragment banks ─────────────────────────────────────────────────────────
# Each key is a slot. Sentences are composed from these by PhraseEngine.
_BANKS = {
    "epigraph": [
        "A leaf shows you the trouble a week before the plant does.",
        "The field keeps its own diary. This is a fair copy.",
        "Read the leaves while the reading is cheap.",
        "Small spots are easier to answer than large ones.",
        "Every plot has a quiet week and a loud one.",
        "What the eye misses, the week's tally remembers.",
        "A clean leaf is also a finding worth writing down.",
        "Catch it on the leaf, not on the harvest.",
    ],
    # headline: [observation] optionally + [tail]
    "headline_obs": [
        "{lead_disease} was the week's main story, on {lead_count} {leaf_w}",
        "Most of the week's attention went to {lead_disease}",
        "{lead_disease} ran through the journal this week, on {lead_count} {leaf_w}",
        "The week turned on {lead_disease}, found on {lead_count} {leaf_w}",
        "{lead_disease} kept coming back this week, {lead_count} {leaf_w} in all",
        "Of everything scanned, {lead_disease} stood out, {lead_count} {leaf_w}",
    ],
    "headline_tail": [
        ", and {new_count} of them had not been recorded before.",
        ", a clear step up on the week before.",
        ", with the okra plots taking most of it.",
        ". The rest of the field stayed quiet.",
        ", enough to keep an eye on.",
        ".",
    ],
    "headline_quiet": [
        "A quiet week. {total} {leaf_w} scanned, nothing pressing.",
        "Not much to report. {total} {leaf_w} this week, the field held steady.",
        "A calm week across the plots, {total} {leaf_w} in the journal.",
        "Little stirred this week. {total} {leaf_w}, and no alarms.",
    ],
    "headline_healthy": [
        "A healthy week. Most of the {total} {leaf_w} scanned came back clean.",
        "Good news this week. The plots read mostly healthy across {total} {leaf_w}.",
        "A clean run. Of {total} {leaf_w}, the large share showed no disease.",
    ],
    # week opener
    "opener_volume_up": [
        "A busier week than last, {total} {leaf_w} against {prev_total}.",
        "More came in this week, {total} {leaf_w} where last week had {prev_total}.",
        "The journal filled faster this week, {total} {leaf_w} to last week's {prev_total}.",
    ],
    "opener_volume_down": [
        "A lighter week, {total} {leaf_w} against last week's {prev_total}.",
        "Fewer specimens this week, {total} against {prev_total} before.",
        "Things slowed a little, {total} {leaf_w} where last week had {prev_total}.",
    ],
    "opener_volume_same": [
        "About the same weight of work as last week, {total} {leaf_w}.",
        "A steady week, {total} {leaf_w}, much like the last.",
        "{total} {leaf_w} this week, in step with the week before.",
    ],
    "opener_focus": [
        "The {focus_crop} plots are where the attention is needed.",
        "Most of what needs doing sits in the {focus_crop} rows.",
        "If anything is asking for a look, it is the {focus_crop}.",
        "The {focus_crop} side of the field carried the week.",
    ],
    # new disease
    "new_disease": [
        "{disease} turned up for the first time this week, on {count} {leaf_w}.",
        "{count} {leaf_w} came back as {disease}, a disease not recorded before.",
        "A first for the journal this week, {disease}, on {count} {leaf_w}.",
        "{disease} is new to these pages. It showed on {count} {leaf_w}.",
        "The week brought one disease the plots had not shown before, {disease}.",
        "{disease} appears here for the first time, {count} {leaf_w} in all.",
        "Worth noting a newcomer, {disease}, on {count} {leaf_w} this week.",
    ],
    # trend line aside
    "trend_aside": [
        "",
        " Worth a second look next week.",
        " Nothing alarming yet.",
        " The move is small but real.",
        " One to watch.",
        "",
        "",
    ],
    # urgent callout
    # Phrased to read correctly for any count, singular or plural: the
    # verbs do not sit directly against {count} {case_w}.
    "urgent_lead": [
        "Treatment should not wait on {count} {case_w} this week.",
        "Act now on {count} {case_w} the week flagged as urgent.",
        "The week threw up {count} {case_w} for acting on today.",
        "Time is short on {count} {case_w} this week.",
    ],
    "urgent_detail": [
        "{disease} on {crop} spreads fast once it takes. Treat within the day and keep affected plants apart from the rest.",
        "{disease} will not wait. Act within twenty four hours and isolate the plants you have flagged.",
        "{disease} moves through a plot quickly. The sooner it is treated and the sick plants are pulled apart, the better.",
        "Left alone, {disease} carries to neighbouring plants. Treat now, separate what is affected.",
    ],
    # re-scan callout
    "rescan": [
        "{count} {leaf_w} came back without a confident reading. A sharper photo in better light would settle the call.",
        "The model was unsure on {count} {leaf_w}. Worth a repeat scan, closer and in good light.",
        "A second photo would help on {count} {leaf_w}. The first was too unclear to call.",
        "Confidence was low on {count} {leaf_w}. A clearer image would help the model decide.",
    ],
    # watch next week
    "watch": [
        "Keep an eye on {topic} in the week ahead.",
        "Worth watching next week, {topic}.",
        "For next week, {topic} is the thing to track.",
        "Carry forward one note, {topic}.",
    ],
    # bridges between sections
    "bridge_charts": [
        "The numbers above point somewhere. The next page reads them.",
        "What the charts show, the next page explains.",
        "Those are the shapes of the week. Here is what they mean.",
    ],
    "bridge_findings": [
        "The specimens behind these findings follow.",
        "The leaves themselves are on the next page.",
        "Here are the specimens worth seeing up close.",
    ],
    "bridge_gallery": [
        "The full record of the week is set out below.",
        "Every specimen, in order, follows.",
        "The complete ledger comes next.",
    ],
    # quiet / empty week
    "quiet_week": [
        "A quiet week in the journal. Too little came in to draw the usual charts, so this is a short note rather than a full report.",
        "Not much to record this week. With only a handful of specimens, the charts would say little, so the report keeps to the essentials.",
        "A light week. The journal stayed mostly closed, and a brief note serves better than a thin set of charts.",
    ],
    "empty_week": [
        "No specimens were scanned this week. Nothing to report, which is its own kind of good news.",
        "The journal stayed shut this week. No scans, no findings.",
        "A blank week. Nothing came in, so there is nothing to set down.",
    ],
}

# chart annotations are keyed by the chart module name.
_CHART_NOTES = {
    "daily_activity": [
        "the busiest day by some way",
        "most of the week's scanning fell here",
        "a clear peak, worth a thought on why",
        "the day the journal filled up",
    ],
    "disease_breakdown": [
        "one disease carried most of the week",
        "the top two account for the bulk of it",
        "a spread across several, none dominant",
        "the long tail is mostly single cases",
    ],
    "crop_mix": [
        "the okra plots did most of the talking",
        "fairly even across the crops",
        "one crop took nearly all the attention",
    ],
    "confidence": [
        "tight and high, a confident week",
        "a wide spread, several uncertain calls",
        "mostly confident, a few soft readings",
        "the low tail is the re-scan list",
    ],
    "tier_split": [
        "most calls were field grade",
        "a fair share landed in review",
        "more abstentions than usual this week",
    ],
    "severity": [
        "mostly mild, caught early",
        "a cluster of severe cases worth noting",
        "the severe end is small but real",
    ],
    "week_over_week": [
        "the jump is the headline of the week",
        "a clear move on the week before",
        "one line moved, the rest held",
    ],
}


class PhraseEngine:
    """Composes report narrative. Seed it with the week so a report is
    reproducible; different weeks diverge."""

    def __init__(self, seed):
        try:
            seed_int = int(seed)
        except (TypeError, ValueError):
            seed_int = abs(hash(str(seed))) % (2 ** 31)
        self._rng = random.Random(seed_int)
        self._used: dict[str, set] = {}
        self._trend_n = 0
        self._last_trend_verb = None

    # ── core picker ────────────────────────────────────────────────────────
    def _pick(self, bank_key: str) -> str:
        bank = _BANKS.get(bank_key) or [""]
        used = self._used.setdefault(bank_key, set())
        avail = [i for i in range(len(bank)) if i not in used]
        if not avail:                       # bank exhausted, allow reuse
            used.clear()
            avail = list(range(len(bank)))
        idx = self._rng.choice(avail)
        used.add(idx)
        return bank[idx]

    @staticmethod
    def _leaf_word(n: int) -> str:
        return "leaf" if n == 1 else "leaves"

    @staticmethod
    def _case_word(n: int) -> str:
        return "case" if n == 1 else "cases"

    def _fill(self, template: str, **kw) -> str:
        out = template
        for k, v in kw.items():
            out = out.replace("{" + k + "}", str(v))
        return out

    # ── public narrative methods ───────────────────────────────────────────
    def epigraph(self) -> str:
        return self._pick("epigraph")

    def headline(self, stats: dict) -> str:
        total = stats.get("total", 0)
        if total == 0:
            return self._pick("empty_week")
        healthy_share = stats.get("healthy_share", 0.0)
        lead = stats.get("lead_disease")        # raw class or None
        if lead is None:
            return self._fill(self._pick("headline_quiet"),
                              total=total, leaf_w=self._leaf_word(total))
        if healthy_share >= 0.8:
            return self._fill(self._pick("headline_healthy"),
                              total=total, leaf_w=self._leaf_word(total))
        lc = stats.get("lead_count", 0)
        obs = self._fill(self._pick("headline_obs"),
                         lead_disease=pretty_class(lead), lead_count=lc,
                         leaf_w=self._leaf_word(lc))
        new_count = stats.get("new_count", 0)
        tails = [t for t in _BANKS["headline_tail"]]
        # only offer the "new" tail when there really were new diseases
        tail = self._pick("headline_tail")
        if "{new_count}" in tail and new_count <= 0:
            tail = "."
        tail = self._fill(tail, new_count=new_count)
        return obs + tail

    def week_opener(self, stats: dict) -> str:
        total = stats.get("total", 0)
        prev = stats.get("prev_total", 0)
        if total > prev * 1.15 and prev > 0:
            vol = self._fill(self._pick("opener_volume_up"),
                             total=total, prev_total=prev,
                             leaf_w=self._leaf_word(total))
        elif prev > 0 and total < prev * 0.85:
            vol = self._fill(self._pick("opener_volume_down"),
                             total=total, prev_total=prev,
                             leaf_w=self._leaf_word(total))
        else:
            vol = self._fill(self._pick("opener_volume_same"),
                             total=total, leaf_w=self._leaf_word(total))
        focus = stats.get("focus_crop")
        if focus:
            vol += " " + self._fill(self._pick("opener_focus"),
                                    focus_crop=_CROP_WORD.get(focus, focus))
        return vol

    def new_disease(self, cls: str, count: int) -> str:
        return self._fill(self._pick("new_disease"),
                          disease=pretty_class(cls), count=count,
                          leaf_w=self._leaf_word(count))

    def trend_line(self, cls: str, prev: int, curr: int) -> str:
        # Vary the verb so two diseases that made the same move do not read
        # identically, and so no two adjacent trend lines share a verb.
        n = self._trend_n
        self._trend_n += 1
        verb = _move_words(prev, curr, salt=n)
        tries = 1
        while verb == self._last_trend_verb and tries < 8:
            verb = _move_words(prev, curr, salt=n + tries * 1000)
            tries += 1
        self._last_trend_verb = verb
        aside = self._pick("trend_aside")
        return f"{pretty_class(cls)} {verb}, {prev} to {curr}.{aside}"

    def urgent_text(self, cases: list[dict]) -> str:
        n = len(cases)
        lead = self._fill(self._pick("urgent_lead"),
                          count=n, case_w=self._case_word(n))
        first = cases[0] if cases else {}
        detail = self._fill(
            self._pick("urgent_detail"),
            disease=disease_only(first.get("predicted_class", "")),
            crop=_CROP_WORD.get(first.get("crop", ""), "the crop"))
        # Several urgent_detail templates open with {disease}; disease_only
        # is lower-case, so fix the sentence start.
        return lead + " " + _sentence_case(detail)

    def rescan_text(self, count: int) -> str:
        return self._fill(self._pick("rescan"),
                          count=count, leaf_w=self._leaf_word(count))

    def watch_text(self, topic: str) -> str:
        return self._fill(self._pick("watch"), topic=topic)

    def chart_note(self, chart_key: str) -> str:
        notes = _CHART_NOTES.get(chart_key)
        if not notes:
            return ""
        used = self._used.setdefault("chart:" + chart_key, set())
        avail = [i for i in range(len(notes)) if i not in used]
        if not avail:
            used.clear()
            avail = list(range(len(notes)))
        idx = self._rng.choice(avail)
        used.add(idx)
        return notes[idx]

    def bridge(self, after_section: str) -> str:
        key = {"charts": "bridge_charts",
               "findings": "bridge_findings",
               "gallery": "bridge_gallery"}.get(after_section)
        return self._pick(key) if key else ""

    def quiet_week(self) -> str:
        return self._pick("quiet_week")
