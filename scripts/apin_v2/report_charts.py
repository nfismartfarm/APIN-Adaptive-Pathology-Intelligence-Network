"""report_charts.py — adaptive chart catalog for the APIN weekly report.

There is no fixed set of charts. Each chart module decides for itself
whether it is worth showing this week (a relevance test) and how much it
matters (a salience score). select_charts() runs the catalog and returns
the most salient charts for the week's data, so the report changes shape
with the season.

Charts render in the field-notebook palette (cream paper, the APIN greens)
as PNG bytes. The squiggly annotation lines are drawn later by report_pdf;
each chart only reports a `note_key` so the phrase engine can caption it.
"""
from __future__ import annotations

import io
import json
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── field-notebook palette ──────────────────────────────────────────────────
PAPER   = "#f4efe2"
INK     = "#1f2a22"
PRIMARY = "#2d6a4f"
LEAF    = "#5a9367"
TAN     = "#c8b88a"
AMBER   = "#c9803b"
CRIMSON = "#a8201f"
GRID    = "#ddd4bd"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": "#a89f86",
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 150,
})

_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _frame(ax, grid_axis="y"):
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def _is_healthy(cls: str) -> bool:
    return "healthy" in (cls or "").lower()


def _parse_severity(resp_json: str):
    """Best-effort severity from a stored response_json. Returns one of
    mild / moderate / severe, or None when the response carries none."""
    if not resp_json:
        return None
    try:
        d = json.loads(resp_json)
    except Exception:
        return None
    for key in ("severity", "severity_label"):
        v = d.get(key)
        if isinstance(v, str) and v.lower() in ("mild", "moderate", "severe"):
            return v.lower()
        if isinstance(v, dict):
            lab = v.get("label") or v.get("level")
            if isinstance(lab, str) and lab.lower() in ("mild", "moderate", "severe"):
                return lab.lower()
    return None


# ── stats ───────────────────────────────────────────────────────────────────
def compute_stats(predictions: list[dict],
                  prev_predictions: list[dict] | None = None) -> dict:
    """Derive every figure the report needs from the week's predictions."""
    prev_predictions = prev_predictions or []
    total = len(predictions)
    diseased = [p for p in predictions
                if not _is_healthy(p.get("predicted_class"))]
    healthy = total - len(diseased)

    by_disease = Counter(p.get("predicted_class") for p in diseased)
    by_crop = Counter((p.get("crop") or "unknown") for p in predictions)
    by_tier = Counter(str(p.get("tier") or "untiered") for p in predictions)
    confs = [float(p["confidence"]) for p in predictions
             if p.get("confidence") is not None]

    by_day = Counter()
    for p in predictions:
        ts = p.get("created_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            by_day[dt.weekday()] += 1
        except Exception:
            pass

    severities = Counter()
    for p in predictions:
        s = _parse_severity(p.get("response_json"))
        if s:
            severities[s] += 1

    prev_by_disease = Counter(p.get("predicted_class") for p in prev_predictions
                              if not _is_healthy(p.get("predicted_class")))

    lead_disease, lead_count = (None, 0)
    if by_disease:
        lead_disease, lead_count = by_disease.most_common(1)[0]

    new_diseases = [d for d in by_disease if d not in prev_by_disease]
    crop_focus = None
    if by_crop:
        crop_focus = by_crop.most_common(1)[0][0]
        if crop_focus == "unknown":
            crop_focus = None

    return {
        "total": total,
        "healthy": healthy,
        "healthy_share": (healthy / total) if total else 0.0,
        "diseased": len(diseased),
        "by_disease": dict(by_disease),
        "by_crop": dict(by_crop),
        "by_tier": dict(by_tier),
        "by_day": dict(by_day),
        "confidences": confs,
        "severities": dict(severities),
        "prev_total": len(prev_predictions),
        "prev_by_disease": dict(prev_by_disease),
        "lead_disease": lead_disease,
        "lead_count": lead_count,
        "new_diseases": new_diseases,
        "new_count": len(new_diseases),
        "focus_crop": crop_focus,
        "n_crops": len([k for k in by_crop if k != "unknown"]),
        "n_diseases": len(by_disease),
    }


def _pretty(cls: str) -> str:
    from scripts.apin_v2.report_text import pretty_class
    return pretty_class(cls)


# ── chart modules ───────────────────────────────────────────────────────────
# Each returns a dict {png, salience, note_key, title} or None if not relevant.

def _chart_daily(stats):
    by_day = stats["by_day"]
    active_days = len([d for d, n in by_day.items() if n > 0])
    if stats["total"] < 3 or active_days < 2:
        return None
    vals = [by_day.get(i, 0) for i in range(7)]
    spread = (max(vals) - min(vals))
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    fig.patch.set_facecolor(PAPER)
    ax.bar(range(7), vals, color=PRIMARY, edgecolor=INK, linewidth=0.7,
           width=0.66)
    ax.set_xticks(range(7)); ax.set_xticklabels(_DAYS, fontsize=9)
    ax.set_ylabel("specimens", fontsize=9)
    _frame(ax)
    sal = 0.45 + min(0.4, spread * 0.12)
    return {"png": _fig_to_png(fig), "salience": sal,
            "note_key": "daily_activity", "title": "Daily scan activity"}


def _chart_disease(stats):
    bd = stats["by_disease"]
    if len(bd) < 1:
        return None
    items = sorted(bd.items(), key=lambda kv: kv[1])
    labels = [_pretty(k) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.4, max(2.4, 0.5 * len(items) + 1.1)))
    fig.patch.set_facecolor(PAPER)
    ax.barh(range(len(items)), vals, color=LEAF, edgecolor=INK,
            linewidth=0.7, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.02, i, str(v), va="center", fontsize=9)
    ax.set_yticks(range(len(items))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("specimens this week", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.16)
    _frame(ax, grid_axis="x")
    sal = 0.7 + min(0.2, len(bd) * 0.03)
    return {"png": _fig_to_png(fig), "salience": sal,
            "note_key": "disease_breakdown",
            "title": "What the week's disease was"}


def _chart_crop(stats):
    bc = {k: v for k, v in stats["by_crop"].items() if k != "unknown"}
    if len(bc) < 2:
        return None
    items = sorted(bc.items(), key=lambda kv: -kv[1])
    labels = [k.capitalize() for k, _ in items]
    vals = [v for _, v in items]
    colors = [PRIMARY, LEAF, TAN, AMBER][:len(items)]
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    fig.patch.set_facecolor(PAPER)
    ax.pie(vals, labels=labels, colors=colors,
           autopct=lambda p: f"{p:.0f}%", startangle=90,
           textprops={"fontsize": 9, "color": INK},
           wedgeprops={"edgecolor": INK, "linewidth": 0.7})
    ax.set_aspect("equal")
    top = vals[0] / sum(vals)
    sal = 0.4 + (0.25 if 0.35 < top < 0.7 else 0.1)
    return {"png": _fig_to_png(fig), "salience": sal,
            "note_key": "crop_mix", "title": "Crop mix"}


def _chart_confidence(stats):
    confs = stats["confidences"]
    if len(confs) < 6:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    fig.patch.set_facecolor(PAPER)
    ax.hist(confs, bins=10, range=(0, 1), color=PRIMARY,
            edgecolor=INK, linewidth=0.7)
    ax.set_xlabel("calibrated confidence", fontsize=9)
    ax.set_ylabel("specimens", fontsize=9)
    _frame(ax)
    low_share = len([c for c in confs if c < 0.6]) / len(confs)
    sal = 0.4 + min(0.4, low_share * 1.5)
    return {"png": _fig_to_png(fig), "salience": sal,
            "note_key": "confidence", "title": "Confidence spread"}


def _chart_tier(stats):
    bt = stats["by_tier"]
    if len(bt) < 2:
        return None
    items = sorted(bt.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    fig.patch.set_facecolor(PAPER)
    ax.bar(range(len(items)), vals, color=TAN, edgecolor=INK,
           linewidth=0.7, width=0.6)
    ax.set_xticks(range(len(items))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("specimens", fontsize=9)
    _frame(ax)
    return {"png": _fig_to_png(fig), "salience": 0.42,
            "note_key": "tier_split", "title": "Decision tier"}


def _chart_severity(stats):
    sev = stats["severities"]
    if not sev:
        return None
    order = ["mild", "moderate", "severe"]
    labels = [s for s in order if sev.get(s)]
    vals = [sev[s] for s in labels]
    cmap = {"mild": LEAF, "moderate": AMBER, "severe": CRIMSON}
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    fig.patch.set_facecolor(PAPER)
    ax.bar(range(len(labels)), vals, color=[cmap[s] for s in labels],
           edgecolor=INK, linewidth=0.7, width=0.58)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([s.capitalize() for s in labels], fontsize=9)
    ax.set_ylabel("cases", fontsize=9)
    _frame(ax)
    sal = 0.5 + min(0.35, sev.get("severe", 0) * 0.12)
    return {"png": _fig_to_png(fig), "salience": sal,
            "note_key": "severity", "title": "Severity profile"}


def _chart_wow(stats):
    bd, pd_ = stats["by_disease"], stats["prev_by_disease"]
    if not pd_ or not bd:
        return None
    keys = sorted(set(bd) | set(pd_),
                  key=lambda k: -(bd.get(k, 0) + pd_.get(k, 0)))[:6]
    if not keys:
        return None
    moved = max((abs(bd.get(k, 0) - pd_.get(k, 0)) for k in keys), default=0)
    if moved < 2:
        return None
    labels = [_pretty(k) for k in keys]
    cur = [bd.get(k, 0) for k in keys]
    prev = [pd_.get(k, 0) for k in keys]
    y = range(len(keys)); h = 0.36
    fig, ax = plt.subplots(figsize=(7.4, max(2.4, 0.62 * len(keys) + 1)))
    fig.patch.set_facecolor(PAPER)
    ax.barh([i + h/2 for i in y], prev, h, color=TAN, edgecolor=INK,
            linewidth=0.7, label="last week")
    ax.barh([i - h/2 for i in y], cur, h, color=PRIMARY, edgecolor=INK,
            linewidth=0.7, label="this week")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("specimens", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="best")
    _frame(ax, grid_axis="x")
    sal = 0.65 + min(0.3, moved * 0.06)
    return {"png": _fig_to_png(fig), "salience": sal,
            "note_key": "week_over_week",
            "title": "This week against last"}


_CATALOG = [_chart_disease, _chart_wow, _chart_daily, _chart_severity,
            _chart_confidence, _chart_crop, _chart_tier]


def select_charts(stats: dict, max_n: int = 5) -> list[dict]:
    """Run the catalog, keep the relevant charts, return up to max_n
    ordered by salience (most important first)."""
    picked = []
    for module in _CATALOG:
        try:
            result = module(stats)
        except Exception:
            result = None
        finally:
            # Close any figure a module left open after raising mid-draw.
            # _fig_to_png already closes figures on the success path; this
            # catches the leak when a module raises between subplots() and
            # _fig_to_png(). plt.close on a stale number is a harmless no-op.
            for _num in plt.get_fignums():
                plt.close(_num)
        if result:
            picked.append(result)
    picked.sort(key=lambda c: c["salience"], reverse=True)
    return picked[:max_n]
