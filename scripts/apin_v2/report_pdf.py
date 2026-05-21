"""Weekly PDF report generator.

Produces a multi-page PDF summarising a user's last 7 days of fieldwork.
Uses reportlab so the output is a real vector PDF (not an image dump), and
keeps the field-notebook journal aesthetic — cream-paper background colour,
Times Roman serif (closest built-in to Fraunces), monospace numerics where
they fit the lab-notebook tone.

Page layout:
    Page 1 — Cover: collector name, date range, headline counts
    Page 2 — Weekly summary: per-disease counts, avg confidence, streak
    Page 3+ — Per-prediction details (one row per prediction, up to 50)

This generator is intentionally self-contained so we can later swap reportlab
for weasyprint (HTML-to-PDF) without changing the calling code.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone, timedelta
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as _canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ── Visual tokens (mirror the dashboard's CSS palette) ────────────────────
INK         = colors.HexColor("#1a1612")
INK_SOFT    = colors.HexColor("#5a5246")
PAPER       = colors.HexColor("#fbf9f3")
PAPER_DEEP  = colors.HexColor("#e9e2d1")
PAPER_EDGE  = colors.HexColor("#c7bca9")
ACCENT_GREEN = colors.HexColor("#2f6f3e")
ACCENT_OCHRE = colors.HexColor("#b87d1e")


def _draw_paper_background(c, page_w, page_h):
    """Soft cream paper fill on every page."""
    c.setFillColor(PAPER)
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)


def _draw_washi_tape(c, x, y, w=60*mm, h=4*mm, rotate_deg=-2):
    """Decorative washi tape rectangle."""
    c.saveState()
    c.translate(x, y)
    c.rotate(rotate_deg)
    c.setFillColor(PAPER_DEEP)
    c.setStrokeColor(PAPER_EDGE)
    c.setLineWidth(0.4)
    c.rect(0, 0, w, h, fill=1, stroke=1)
    c.restoreState()


def _draw_footer(c, page_w, page_num, total_pages, collector):
    """Page footer: collector name + page number, small grey serif italic."""
    c.setFillColor(INK_SOFT)
    c.setFont("Times-Italic", 8)
    c.drawString(15*mm, 10*mm,
                 f"{collector} — Pathology Journal — Weekly Report")
    c.drawRightString(page_w - 15*mm, 10*mm,
                      f"Page {page_num} of {total_pages}")


def _pretty_class(s: str) -> str:
    if not s:
        return "—"
    return s.replace("_", " ").title()


def _tier_label(t: Optional[str]) -> str:
    if not t:
        return "—"
    return t.replace("_", " ").title()


def _date_human(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        d = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return d.strftime("%d %b · %H:%M")
    except (TypeError, ValueError):
        return iso[:16]


def generate_weekly_pdf(
    *,
    user: dict,
    predictions: list[dict],
    week_start: str,
    week_end: str,
) -> bytes:
    """Render the weekly report for ONE user and return the PDF bytes.

    Args:
        user:          {id, username, display_name, ...}
        predictions:   list of slim prediction rows (id, crop, predicted_class,
                       confidence, tier, created_at) for the week. Newest first.
        week_start:    YYYY-MM-DD
        week_end:      YYYY-MM-DD
    """
    buf = io.BytesIO()
    c = _canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4

    # Compute total pages: cover + summary + ceil(n/25) detail pages
    n_pred = len(predictions)
    detail_pages = max(1, (n_pred + 24) // 25)
    total_pages = 2 + detail_pages
    collector = user.get("display_name") or user.get("username") or "Collector"

    # ─── Page 1: cover ────────────────────────────────────────────────────
    _draw_paper_background(c, page_w, page_h)
    _draw_washi_tape(c, page_w/2 - 30*mm, page_h - 28*mm, w=60*mm, h=5*mm, rotate_deg=-2)

    # Title
    c.setFillColor(INK)
    c.setFont("Times-Bold", 36)
    c.drawCentredString(page_w/2, page_h - 70*mm, "Pathology Journal")

    c.setFont("Times-Italic", 14)
    c.setFillColor(INK_SOFT)
    c.drawCentredString(page_w/2, page_h - 80*mm, "Weekly Field Report")

    # Date range
    c.setFont("Times-Roman", 14)
    c.setFillColor(INK)
    c.drawCentredString(page_w/2, page_h - 100*mm,
                        f"{week_start}  to  {week_end}")

    # Collector
    c.setFont("Times-Italic", 12)
    c.setFillColor(INK_SOFT)
    c.drawCentredString(page_w/2, page_h - 110*mm, f"Collector: {collector}")

    # Headline counts (computed from predictions)
    total = len(predictions)
    healthy = sum(1 for p in predictions
                  if "healthy" in (p.get("predicted_class") or "").lower())
    avg_conf = (sum(p.get("confidence") or 0.0 for p in predictions) / total) if total else 0.0
    field_grade = sum(1 for p in predictions
                      if (p.get("tier") or "").upper() == "FIELD_GRADE")

    # Big stats grid
    stats_y = page_h - 170*mm
    stat_w = 35*mm
    cx_center = page_w / 2
    stats = [
        ("samples", str(total)),
        ("healthy",  str(healthy)),
        ("field-grade", str(field_grade)),
        ("avg conf", f"{avg_conf:.2f}" if total else "—"),
    ]
    for i, (label, val) in enumerate(stats):
        x = cx_center - 2*stat_w + i*stat_w
        c.setFillColor(PAPER_DEEP)
        c.setStrokeColor(PAPER_EDGE)
        c.setLineWidth(0.5)
        c.rect(x, stats_y - 24*mm, stat_w - 2*mm, 24*mm, fill=1, stroke=1)
        c.setFillColor(INK)
        c.setFont("Courier-Bold", 18)
        c.drawCentredString(x + (stat_w-2*mm)/2, stats_y - 13*mm, val)
        c.setFillColor(INK_SOFT)
        c.setFont("Times-Italic", 9)
        c.drawCentredString(x + (stat_w-2*mm)/2, stats_y - 21*mm, label)

    # Footer note on cover
    c.setFillColor(INK_SOFT)
    c.setFont("Times-Italic", 10)
    c.drawCentredString(page_w/2, 28*mm,
                        f"Generated {datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')}")
    _draw_footer(c, page_w, 1, total_pages, collector)
    c.showPage()

    # ─── Page 2: summary ──────────────────────────────────────────────────
    _draw_paper_background(c, page_w, page_h)
    _draw_washi_tape(c, 30*mm, page_h - 20*mm, w=50*mm, h=4*mm, rotate_deg=2)

    c.setFillColor(INK)
    c.setFont("Times-Bold", 22)
    c.drawString(20*mm, page_h - 28*mm, "Weekly Summary")

    c.setFont("Times-Italic", 11)
    c.setFillColor(INK_SOFT)
    c.drawString(20*mm, page_h - 36*mm,
                 f"{week_start} — {week_end}    ·    {total} specimen{'' if total==1 else 's'} catalogued")

    # Per-disease counts
    from collections import Counter
    by_class = Counter(p.get("predicted_class") for p in predictions
                       if p.get("predicted_class"))
    by_crop  = Counter(p.get("crop")  for p in predictions if p.get("crop"))

    # Section A — by disease
    y = page_h - 56*mm
    c.setFillColor(ACCENT_GREEN)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(20*mm, y, "BY DISEASE")
    y -= 4*mm
    c.setStrokeColor(PAPER_EDGE)
    c.setLineWidth(0.4)
    c.line(20*mm, y, page_w - 20*mm, y)
    y -= 6*mm

    c.setFillColor(INK)
    c.setFont("Times-Roman", 11)
    for cls, n in by_class.most_common(15):
        if y < 80*mm:
            break  # leave room for the crops section
        # disease name (left)
        c.drawString(20*mm, y, _pretty_class(cls))
        # count (right, monospace)
        c.setFont("Courier", 11)
        c.drawRightString(page_w - 20*mm, y, str(n))
        c.setFont("Times-Roman", 11)
        # dotted leader
        # (skip the visual dotted line to keep the layout clean for now)
        y -= 6*mm

    # Section B — by crop
    y = max(y, 50*mm)
    c.setFillColor(ACCENT_GREEN)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(20*mm, y, "BY CROP")
    y -= 4*mm
    c.setStrokeColor(PAPER_EDGE)
    c.line(20*mm, y, page_w - 20*mm, y)
    y -= 6*mm
    c.setFillColor(INK)
    c.setFont("Times-Roman", 11)
    # Code-review fix #5: floor-guard mirrors the disease section above.
    # In normal use _ALLOWED_CROPS has 4 entries so overflow is unlikely,
    # but legacy data or future expansion could push y below the 10mm
    # footer area and cause text-on-footer overlap.
    for crop, n in by_crop.most_common():
        if y < 25*mm:
            break
        c.drawString(20*mm, y, (crop or "—").title())
        c.setFont("Courier", 11)
        c.drawRightString(page_w - 20*mm, y, str(n))
        c.setFont("Times-Roman", 11)
        y -= 6*mm

    _draw_footer(c, page_w, 2, total_pages, collector)
    c.showPage()

    # ─── Pages 3+: details (25 rows per page) ────────────────────────────
    PER_PAGE = 25
    for page_idx in range(detail_pages):
        _draw_paper_background(c, page_w, page_h)
        _draw_washi_tape(c, page_w - 80*mm, page_h - 18*mm, w=50*mm, h=4*mm, rotate_deg=-3)

        c.setFillColor(INK)
        c.setFont("Times-Bold", 18)
        # Code-review fix #4: when there are 0 predictions the math
        # rendered "Specimens (1–0 of 0)" which is gibberish. Special-case it.
        if n_pred == 0:
            header_text = "Specimens — none this week"
        else:
            header_text = (
                f"Specimens ({page_idx*PER_PAGE + 1}–"
                f"{min((page_idx+1)*PER_PAGE, n_pred)} of {n_pred})"
            )
        c.drawString(20*mm, page_h - 28*mm, header_text)

        # Table header
        y = page_h - 42*mm
        c.setFillColor(ACCENT_GREEN)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(20*mm, y, "ID")
        c.drawString(35*mm, y, "DISEASE")
        c.drawString(105*mm, y, "CROP")
        c.drawString(130*mm, y, "TIER")
        c.drawString(155*mm, y, "CONF")
        c.drawRightString(page_w - 20*mm, y, "WHEN")
        y -= 2*mm
        c.setStrokeColor(PAPER_EDGE)
        c.setLineWidth(0.6)
        c.line(20*mm, y, page_w - 20*mm, y)
        y -= 5*mm

        # Rows
        start = page_idx * PER_PAGE
        end = min(start + PER_PAGE, n_pred)
        for p in predictions[start:end]:
            c.setFillColor(INK)
            c.setFont("Courier", 9)
            c.drawString(20*mm, y, f"#{p.get('id', '—'):04d}" if p.get('id') is not None else "#----")
            c.setFont("Times-Roman", 9.5)
            c.drawString(35*mm, y, _pretty_class(p.get("predicted_class"))[:35])
            c.drawString(105*mm, y, (p.get("crop") or "—").title()[:10])
            c.setFont("Helvetica", 8)
            c.setFillColor(INK_SOFT)
            c.drawString(130*mm, y, _tier_label(p.get("tier"))[:10])
            c.setFont("Courier", 9.5)
            c.setFillColor(INK)
            conf = p.get("confidence")
            c.drawString(155*mm, y, f"{conf:.3f}" if conf is not None else "—")
            c.setFont("Times-Italic", 8.5)
            c.setFillColor(INK_SOFT)
            c.drawRightString(page_w - 20*mm, y, _date_human(p.get("created_at")))
            y -= 5.5*mm
            if y < 25*mm:
                break

        _draw_footer(c, page_w, 3 + page_idx, total_pages, collector)
        c.showPage()

    c.save()
    return buf.getvalue()
