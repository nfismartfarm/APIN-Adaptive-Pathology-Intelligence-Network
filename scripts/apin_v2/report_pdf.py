"""report_pdf.py — the APIN weekly field report renderer.

Renders the adaptive weekly PDF in the field-notebook visual language:
a six-section arc, hand-drawn vector icons (no emojis), washi-taped and
titled specimen images, the nanofarm logo on the cover, notebook charts
chosen by report_charts, narrative from report_text's phrase engine, and
a graceful partial report for quiet weeks.

Public entry point: generate_weekly_pdf(...) -> (pdf_bytes, summary_dict).
"""
from __future__ import annotations

import io
import math
import os
from datetime import datetime, timezone

from reportlab.pdfgen import canvas as _canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader

from scripts.apin_v2.report_text import (
    PhraseEngine, pretty_class, disease_only)
from scripts.apin_v2.report_charts import compute_stats, select_charts

# ── palette ─────────────────────────────────────────────────────────────────
PAPER      = HexColor("#f4efe2")
PAPER_SOFT = HexColor("#fbf9f0")
PAPER_DEEP = HexColor("#e9e1cd")
EDGE       = HexColor("#cabfa9")
RULE       = HexColor("#b9ac8e")
INK        = HexColor("#1f2a22")
INK_SOFT   = HexColor("#5d5446")
INK_FAINT  = HexColor("#8c8369")
GREEN      = HexColor("#2d6a4f")
LEAFC      = HexColor("#5a9367")
AMBER      = HexColor("#c9803b")
CRIMSON    = HexColor("#a8201f")
TAPE       = HexColor("#e0cea0")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

SERIF, SERIF_B, SERIF_I = "Times-Roman", "Times-Bold", "Times-Italic"
MONO = "Courier"

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOGO_PATH = os.path.join(_ROOT, "logo.png")


def _clean(text):
    """Strip em / en dashes from reference content so the report keeps the
    plain-voice rule even when the source knowledge base uses them."""
    if not text:
        return text
    return (str(text).replace(" — ", ", ").replace(" – ", ", ")
            .replace("—", ", ").replace("–", ", "))


def _tier_grade(t):
    t = str(t or "").upper()
    if t in ("1A", "1B", "5"):
        return "field grade"
    if t in ("4A", "4B"):
        return "abstained"
    if t in ("3A", "3B"):
        return "review"
    return t.lower() or "untiered"


# ── hand-drawn vector icons (stroked, slight wobble) ────────────────────────
def _wob(rng, v):
    return v + (rng - 0.5) * v * 0.06 if rng else v


def icon_star(c, x, y, s):
    c.saveState(); c.setStrokeColor(INK); c.setLineWidth(1.1)
    for i in range(6):
        a = i * math.pi / 3
        c.line(x, y, x + math.cos(a) * s, y + math.sin(a) * s)
    c.restoreState()


def icon_sprout(c, x, y, s):
    c.saveState(); c.setStrokeColor(GREEN); c.setLineWidth(1.2)
    c.line(x, y - s, x, y + s * 0.4)
    p = c.beginPath(); p.moveTo(x, y); p.curveTo(x, y + s, x - s, y + s, x - s, y)
    c.drawPath(p)
    p = c.beginPath(); p.moveTo(x, y); p.curveTo(x, y + s, x + s, y + s, x + s, y)
    c.drawPath(p)
    c.restoreState()


def icon_flag(c, x, y, s):
    c.saveState(); c.setStrokeColor(CRIMSON); c.setLineWidth(1.2)
    c.line(x - s * 0.6, y - s, x - s * 0.6, y + s)
    p = c.beginPath()
    p.moveTo(x - s * 0.6, y + s)
    p.lineTo(x + s * 0.7, y + s * 0.45)
    p.lineTo(x - s * 0.6, y - s * 0.1)
    c.setFillColor(CRIMSON); c.drawPath(p, stroke=1, fill=1)
    c.restoreState()


def icon_lens(c, x, y, s):
    c.saveState(); c.setStrokeColor(AMBER); c.setLineWidth(1.3)
    c.circle(x - s * 0.2, y + s * 0.2, s * 0.7, stroke=1, fill=0)
    c.line(x + s * 0.32, y - s * 0.32, x + s, y - s)
    c.restoreState()


def icon_eye(c, x, y, s):
    c.saveState(); c.setStrokeColor(GREEN); c.setLineWidth(1.1)
    p = c.beginPath()
    p.moveTo(x - s, y); p.curveTo(x - s * 0.3, y + s, x + s * 0.3, y + s, x + s, y)
    p.curveTo(x + s * 0.3, y - s, x - s * 0.3, y - s, x - s, y)
    c.drawPath(p)
    c.circle(x, y, s * 0.34, stroke=1, fill=0)
    c.restoreState()


def icon_spray(c, x, y, s):
    c.saveState(); c.setStrokeColor(INK_SOFT); c.setLineWidth(1.1)
    c.rect(x - s * 0.45, y - s, s * 0.9, s * 1.5, stroke=1, fill=0)
    c.line(x - s * 0.2, y + s * 0.5, x - s * 0.2, y + s)
    c.line(x - s * 0.2, y + s, x + s * 0.55, y + s)
    for dx in (0.8, 1.0, 1.2):
        c.line(x + s * 0.6, y + s, x + s * dx, y + s + s * 0.25 * (dx - 1) * 4)
    c.restoreState()


def icon_book(c, x, y, s):
    c.saveState(); c.setStrokeColor(GREEN); c.setLineWidth(1.1)
    c.line(x, y - s, x, y + s)
    p = c.beginPath(); p.moveTo(x, y + s)
    p.curveTo(x - s, y + s * 0.7, x - s, y - s, x - s, y - s)
    p.lineTo(x, y - s); c.drawPath(p)
    p = c.beginPath(); p.moveTo(x, y + s)
    p.curveTo(x + s, y + s * 0.7, x + s, y - s, x + s, y - s)
    p.lineTo(x, y - s); c.drawPath(p)
    c.restoreState()


def _squiggle(c, x1, y1, x2, y2, color=INK_FAINT):
    """A hand-drawn wavy connector from a margin note to a chart feature."""
    c.saveState(); c.setStrokeColor(color); c.setLineWidth(0.8)
    dx, dy = x2 - x1, y2 - y1
    seg = 3
    p = c.beginPath(); p.moveTo(x1, y1)
    for i in range(1, seg + 1):
        t = i / seg
        mx, my = x1 + dx * t, y1 + dy * t
        off = 4 if i % 2 else -4
        cx1 = x1 + dx * (t - 0.66 / seg)
        cy1 = y1 + dy * (t - 0.66 / seg) + off
        cx2 = x1 + dx * (t - 0.33 / seg)
        cy2 = y1 + dy * (t - 0.33 / seg) + off
        p.curveTo(cx1, cy1, cx2, cy2, mx, my)
    c.drawPath(p)
    c.restoreState()


# ── builder ─────────────────────────────────────────────────────────────────
class _Builder:
    def __init__(self):
        self.buf = io.BytesIO()
        self.c = _canvas.Canvas(self.buf, pagesize=A4)
        self.y = PAGE_H - MARGIN
        self.page = 0
        self._start_page(running=False)

    # page management ------------------------------------------------------
    def _paper(self):
        self.c.setFillColor(PAPER)
        self.c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        # faint horizontal rule lines, ledger-paper feel
        self.c.setStrokeColor(HexColor("#e7dec8"))
        self.c.setLineWidth(0.4)
        yy = PAGE_H - 34 * mm
        while yy > 22 * mm:
            self.c.line(MARGIN, yy, PAGE_W - MARGIN, yy)
            yy -= 8.6 * mm

    def washi(self, x, y, w=42 * mm, rot=-2):
        c = self.c
        c.saveState(); c.translate(x, y); c.rotate(rot)
        c.setFillColor(TAPE); c.setStrokeColor(HexColor("#d2bd86"))
        c.setLineWidth(0.4)
        c.rect(0, 0, w, 4.6 * mm, fill=1, stroke=1)
        c.setStrokeColor(HexColor("#d2bd86")); c.setLineWidth(0.5)
        for i in range(int(w / (3 * mm))):
            c.line(i * 3 * mm, 0, i * 3 * mm + 4.6 * mm, 4.6 * mm)
        c.restoreState()

    def _start_page(self, running=True):
        self.page += 1
        self._paper()
        if running:
            self.c.setFont(SERIF_I, 8.5)
            self.c.setFillColor(INK_FAINT)
            self.c.drawString(MARGIN, PAGE_H - 12 * mm,
                              "APIN  Weekly Field Report")
        self.y = PAGE_H - MARGIN - (8 * mm if running else 0)

    def _footer(self, collector):
        self.c.setStrokeColor(RULE); self.c.setLineWidth(0.6)
        self.c.line(MARGIN, 15 * mm, PAGE_W - MARGIN, 15 * mm)
        self.c.setFont(MONO, 7.5); self.c.setFillColor(INK_FAINT)
        self.c.drawString(MARGIN, 11 * mm, f"APIN  .  {collector}")
        self.c.drawRightString(PAGE_W - MARGIN, 11 * mm, f"page {self.page}")

    def page_break(self, collector):
        self._footer(collector)
        self.c.showPage()
        self._start_page(running=True)

    def need(self, h, collector):
        """Ensure h points of vertical room remain, else break the page."""
        if self.y - h < 22 * mm:
            self.page_break(collector)

    # text helpers ---------------------------------------------------------
    def wrap(self, text, font, size, max_w):
        words, lines, cur = str(text).split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if self.c.stringWidth(t, font, size) <= max_w:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    def section_head(self, text, icon=None):
        c = self.c
        if icon:
            icon(c, MARGIN + 3, self.y - 3, 4)
        c.setFont(SERIF_B, 15); c.setFillColor(INK)
        c.drawString(MARGIN + (14 if icon else 0), self.y - 7, text)
        c.setStrokeColor(RULE); c.setLineWidth(0.7)
        c.line(MARGIN, self.y - 12, PAGE_W - MARGIN, self.y - 12)
        self.y -= 22

    def paragraph(self, text, font=SERIF, size=10.5, color=INK,
                  lead=14, indent=0, max_w=None):
        max_w = max_w or (CONTENT_W - indent)
        for ln in self.wrap(text, font, size, max_w):
            self.c.setFont(font, size); self.c.setFillColor(color)
            self.c.drawString(MARGIN + indent, self.y - size, ln)
            self.y -= lead
        self.y -= 3

    def note_block(self, text):
        """A 'Note.' aside set off by a dotted vertical rule."""
        inner = CONTENT_W - 14
        lines = self.wrap("Note.  " + text, SERIF_I, 9.8, inner)
        h = len(lines) * 12.5 + 8
        c = self.c
        c.saveState()
        c.setStrokeColor(INK_FAINT); c.setLineWidth(1)
        c.setDash(1, 2)
        c.line(MARGIN + 2, self.y, MARGIN + 2, self.y - h)
        c.setDash()
        c.restoreState()
        yy = self.y - 4
        for ln in lines:
            c.setFont(SERIF_I, 9.8); c.setFillColor(INK_SOFT)
            c.drawString(MARGIN + 12, yy - 9, ln)
            yy -= 12.5
        self.y -= h + 6

    def pull_quote(self, text):
        lines = self.wrap(text, SERIF_I, 14, CONTENT_W - 40)
        c = self.c
        c.setFont(SERIF_B, 22); c.setFillColor(PAPER_DEEP)
        c.drawString(MARGIN + 6, self.y - 14, "“")
        for ln in lines:
            c.setFont(SERIF_I, 14); c.setFillColor(INK)
            c.drawCentredString(PAGE_W / 2, self.y - 14, ln)
            self.y -= 19
        self.y -= 8

    def callout(self, title, body, accent, icon):
        inner = CONTENT_W - 26
        lines = self.wrap(body, SERIF, 10, inner)
        h = 16 + len(lines) * 13 + 8
        c = self.c
        c.setFillColor(PAPER_SOFT); c.setStrokeColor(EDGE); c.setLineWidth(0.6)
        c.rect(MARGIN, self.y - h, CONTENT_W, h, fill=1, stroke=1)
        c.setFillColor(accent)
        c.rect(MARGIN, self.y - h, 3.2, h, fill=1, stroke=0)
        if icon:
            icon(c, MARGIN + 15, self.y - 11, 4.4)
        c.setFont(SERIF_B, 10.5); c.setFillColor(accent)
        c.drawString(MARGIN + 26, self.y - 14, title.upper())
        yy = self.y - 28
        for ln in lines:
            c.setFont(SERIF, 10); c.setFillColor(INK)
            c.drawString(MARGIN + 14, yy, ln)
            yy -= 13
        self.y -= h + 9

    def kpi_row(self, tiles):
        n = len(tiles)
        gap = 4 * mm
        tw = (CONTENT_W - 2 * mm - gap * (n - 1)) / n
        c = self.c
        for i, (val, label, hot) in enumerate(tiles):
            x = MARGIN + i * (tw + gap)
            c.setFillColor(PAPER_SOFT)
            c.setStrokeColor(CRIMSON if hot else EDGE)
            c.setLineWidth(1.0 if hot else 0.6)
            c.rect(x, self.y - 22 * mm, tw, 22 * mm, fill=1, stroke=1)
            c.setFont(MONO, 17)
            c.setFillColor(CRIMSON if hot else INK)
            c.drawCentredString(x + tw / 2, self.y - 12 * mm, str(val))
            c.setFont(SERIF_I, 8.5); c.setFillColor(INK_SOFT)
            c.drawCentredString(x + tw / 2, self.y - 18.5 * mm, label)
        self.y -= 22 * mm + 6

    def image_tile(self, img_bytes, tile_w, tile_h, caption, fallback):
        """One tape-topped, captioned image tile. Draws at the current x via
        the caller positioning; returns nothing, caller manages layout."""
        # placeholder for callers that draw tiles in a grid; see gallery code

    def finish(self, collector):
        self._footer(collector)
        self.c.showPage()
        self.c.save()
        return self.buf.getvalue()


# ── helpers for image tiles in a 2-up grid ──────────────────────────────────
def _draw_tile(b, x, top_y, w, h, img_bytes, caption):
    c = b.c
    # washi tape above
    b.washi(x + w / 2 - 16 * mm, top_y + 1.5 * mm, w=32 * mm,
            rot=(-3 if int(x) % 2 else 3))
    fy = top_y - h
    c.setFillColor(PAPER_DEEP); c.setStrokeColor(EDGE); c.setLineWidth(0.7)
    c.rect(x, fy, w, h, fill=1, stroke=1)
    drawn = False
    if img_bytes:
        try:
            c.drawImage(ImageReader(io.BytesIO(img_bytes)), x + 1.5, fy + 1.5,
                        w - 3, h - 3, preserveAspectRatio=True,
                        anchor='c', mask='auto')
            drawn = True
        except Exception:
            drawn = False
    if not drawn:
        c.setFont(SERIF_I, 8.5); c.setFillColor(INK_FAINT)
        c.drawCentredString(x + w / 2, fy + h / 2, "image not captured")
    c.setFont(MONO, 7); c.setFillColor(INK_FAINT)
    c.drawCentredString(x + w / 2, fy - 9, caption.upper())
    return fy


# ── main entry point ────────────────────────────────────────────────────────
def generate_weekly_pdf(*, user, predictions, prev_predictions,
                        week_start, week_end,
                        fetch_image=None, fetch_heatmap=None,
                        treatments=None, diagnosis_lookup=None):
    """Render the weekly report. Returns (pdf_bytes, summary_dict)."""
    treatments = treatments or []
    diagnosis_lookup = diagnosis_lookup or {}
    collector = (user or {}).get("display_name") or \
        (user or {}).get("username") or "Collector"

    stats = compute_stats(predictions, prev_predictions)
    total = stats["total"]
    seed = int(str(week_start).replace("-", "") or "0")
    ph = PhraseEngine(seed)

    b = _Builder()
    c = b.c

    # ── COVER ────────────────────────────────────────────────────────────
    b.washi(PAGE_W / 2 - 24 * mm, PAGE_H - 24 * mm, w=48 * mm, rot=-2)
    logo_drawn = False
    if os.path.exists(_LOGO_PATH):
        try:
            c.drawImage(ImageReader(_LOGO_PATH), PAGE_W / 2 - 17 * mm,
                        PAGE_H - 96 * mm, 34 * mm, 34 * mm,
                        preserveAspectRatio=True, mask='auto')
            logo_drawn = True
        except Exception:
            logo_drawn = False
    ty = PAGE_H - (110 if logo_drawn else 86) * mm
    c.setFont(SERIF_B, 30); c.setFillColor(INK)
    c.drawCentredString(PAGE_W / 2, ty, "APIN")
    c.setFont(SERIF_I, 13); c.setFillColor(INK_SOFT)
    c.drawCentredString(PAGE_W / 2, ty - 9 * mm, "Weekly Field Report")
    c.setFont(SERIF, 12); c.setFillColor(INK)
    c.drawCentredString(PAGE_W / 2, ty - 18 * mm,
                        f"{week_start}  to  {week_end}")
    c.setFont(SERIF_I, 10.5); c.setFillColor(INK_SOFT)
    c.drawCentredString(PAGE_W / 2, ty - 25 * mm, f"Collector  .  {collector}")

    b.y = ty - 38 * mm
    # headline callout
    headline = ph.headline({
        "total": total, "lead_disease": stats["lead_disease"],
        "lead_count": stats["lead_count"], "new_count": stats["new_count"],
        "healthy_share": stats["healthy_share"],
        "prev_total": stats["prev_total"]})
    b.callout("this week, in a line", headline, GREEN, icon_star)

    # KPI tiles
    urgent_n = _count_urgent(predictions)
    b.kpi_row([
        (total, "specimens", False),
        (stats["n_diseases"], "diseases", False),
        (f"{int(round(stats['healthy_share'] * 100))}%", "healthy", False),
        (_mean_conf(predictions), "mean conf", False),
        (urgent_n, "urgent", urgent_n > 0),
    ])
    c.setFont(SERIF_I, 9); c.setFillColor(INK_FAINT)
    c.drawCentredString(PAGE_W / 2, 46 * mm,
                        ph.epigraph())
    c.drawCentredString(
        PAGE_W / 2, 40 * mm,
        "Generated " + datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC"))
    b.page_break(collector)

    # ── quiet-week short-circuit ─────────────────────────────────────────
    if total == 0:
        b.section_head("A quiet week")
        b.paragraph(ph.headline({"total": 0}))
        b.paragraph(ph.quiet_week(), font=SERIF_I, color=INK_SOFT)
        return b.finish(collector), _summary(stats, 0)

    sparse = total < 5

    # ── CHARTS or SPOTLIGHT ──────────────────────────────────────────────
    if not sparse:
        charts = select_charts(stats, max_n=5)
        b.section_head("The shape of the week")
        for ch in charts:
            iw = CONTENT_W * 0.62
            ih = iw * _img_ratio(ch["png"])
            ih = min(ih, 62 * mm)
            b.need(ih + 26, collector)
            c.setFont(SERIF_B, 10.5); c.setFillColor(INK)
            c.drawString(MARGIN, b.y - 9, ch["title"])
            b.y -= 14
            try:
                c.drawImage(ImageReader(io.BytesIO(ch["png"])), MARGIN,
                            b.y - ih, iw, ih, preserveAspectRatio=True,
                            anchor='nw', mask='auto')
            except Exception:
                pass
            note = ph.chart_note(ch["note_key"])
            if note:
                nx = MARGIN + iw + 8
                ny = b.y - ih * 0.4
                _squiggle(c, MARGIN + iw - 4, b.y - ih * 0.55, nx, ny)
                c.setFont(SERIF_I, 9); c.setFillColor(INK_SOFT)
                for ln in b.wrap(note, SERIF_I, 9,
                                 PAGE_W - MARGIN - nx - 2):
                    c.drawString(nx, ny, ln)
                    ny -= 11.5
            b.y -= ih + 14
        b.paragraph(ph.bridge("charts"), font=SERIF_I, color=INK_FAINT)
        b.page_break(collector)

    # ── FINDINGS ─────────────────────────────────────────────────────────
    b.section_head("What it means")
    b.paragraph(ph.week_opener({
        "total": total, "prev_total": stats["prev_total"],
        "focus_crop": stats["focus_crop"]}))

    for d in stats["new_diseases"][:3]:
        b.need(60, collector)
        b.callout("new this week",
                  ph.new_disease(d, stats["by_disease"].get(d, 1)),
                  LEAFC, icon_sprout)

    trend = _trend_pairs(stats)
    if trend:
        body = "  ".join(ph.trend_line(d, p, cu) for d, p, cu in trend[:4])
        b.need(70, collector)
        b.callout("trending against last week", body, GREEN, icon_star)

    urgent_cases = _urgent_cases(predictions)
    if urgent_cases:
        b.need(70, collector)
        b.callout("needs action now", ph.urgent_text(urgent_cases),
                  CRIMSON, icon_flag)

    rescan = [p for p in predictions
              if (p.get("confidence") or 1) < 0.6]
    if rescan:
        shown = rescan[:8]
        ids = ", ".join("#%s" % p.get("id") for p in shown)
        extra = len(rescan) - len(shown)
        if extra > 0:
            ids += ", and %d more" % extra
        b.need(64, collector)
        b.callout("worth a second look",
                  ph.rescan_text(len(rescan)) + "  " + ids,
                  AMBER, icon_lens)

    if sparse:
        b.note_block(ph.quiet_week())
    b.paragraph(ph.bridge("findings"), font=SERIF_I, color=INK_FAINT)
    b.page_break(collector)

    # ── SPECIMEN GALLERY ─────────────────────────────────────────────────
    notable = _notable(predictions)
    if notable and (fetch_image or fetch_heatmap):
        b.section_head("Specimens of note", icon=icon_eye)
        for p in notable[:4]:
            tile_w = (CONTENT_W - 10 * mm) / 2
            tile_h = 40 * mm
            b.need(tile_h + 40 * mm, collector)
            b.y -= 6 * mm                       # room for the tape above
            img = fetch_image(p["id"]) if fetch_image else None
            cam = fetch_heatmap(p["id"]) if fetch_heatmap else None
            _draw_tile(b, MARGIN, b.y, tile_w, tile_h, img,
                       "the leaf as found")
            _draw_tile(b, MARGIN + tile_w + 10 * mm, b.y, tile_w, tile_h,
                       cam, "where the model looked")
            b.y -= tile_h + 17                  # tile + caption + a gap
            conf = p.get("confidence")
            meta = "#%s    %s    %s    %s" % (
                p.get("id"), pretty_class(p.get("predicted_class")),
                ("%.0f%%" % (conf * 100)) if conf is not None else "no score",
                _tier_grade(p.get("tier")))
            c.setFont(SERIF_B, 10); c.setFillColor(INK)
            c.drawString(MARGIN, b.y, meta)
            b.y -= 11
            c.setStrokeColor(RULE); c.setLineWidth(0.5)
            c.line(MARGIN, b.y, PAGE_W - MARGIN, b.y)
            b.y -= 15
        b.paragraph(ph.bridge("gallery"), font=SERIF_I, color=INK_FAINT)
        b.page_break(collector)

    # ── LEDGER ───────────────────────────────────────────────────────────
    b.section_head("The full record")
    _ledger(b, c, predictions, collector)
    b.y -= 6
    c.setFont(SERIF_I, 9); c.setFillColor(INK_FAINT)
    c.drawRightString(PAGE_W - MARGIN, b.y,
                      "%d specimen%s in all" % (total, "" if total == 1 else "s"))
    b.y -= 16

    # ── TREATMENTS + REFERENCE ───────────────────────────────────────────
    b.need(90, collector)
    if treatments:
        b.section_head("Treatments you logged", icon=icon_spray)
        for t in treatments[:8]:
            b.need(16, collector)
            c.setFont(MONO, 8.5); c.setFillColor(INK_SOFT)
            c.drawString(MARGIN, b.y, str(t.get("applied_date", ""))[:10])
            c.setFont(SERIF, 10); c.setFillColor(INK)
            line = "%s   %s" % (t.get("treatment", ""),
                                pretty_class(t.get("disease")) if t.get("disease") else "")
            c.drawString(MARGIN + 24 * mm, b.y, line[:70])
            b.y -= 14
        b.y -= 6

    seen = list(stats["by_disease"].keys())
    if seen and diagnosis_lookup:
        b.need(60, collector)
        b.section_head("Field reference", icon=icon_book)
        for d in seen[:5]:
            info = diagnosis_lookup.get(d)
            if not info:
                continue
            b.need(54, collector)
            c.setFont(SERIF_B, 10.5); c.setFillColor(GREEN)
            c.drawString(MARGIN, b.y, pretty_class(d))
            b.y -= 13
            for lbl, key in (("Cause", "cause"), ("Symptoms", "symptoms"),
                             ("Prevent", "prevention")):
                val = info.get(key)
                if isinstance(val, dict):
                    val = val.get("moderate") or next(iter(val.values()), "")
                if not val:
                    continue
                c.setFont(SERIF_I, 8.5); c.setFillColor(INK_FAINT)
                c.drawString(MARGIN + 4, b.y, lbl)
                lines = b.wrap(_clean(str(val)), SERIF, 9, CONTENT_W - 24 * mm)
                c.setFont(SERIF, 9); c.setFillColor(INK)
                yy = b.y
                for ln in lines[:3]:
                    c.drawString(MARGIN + 22 * mm, yy, ln)
                    yy -= 11
                b.y = yy - 2
            b.y -= 8

    b.need(40, collector)
    watch = _watch_topic(stats)
    b.callout("watch next week", ph.watch_text(watch), GREEN, icon_eye)

    return b.finish(collector), _summary(stats, total)


# ── small computation helpers ───────────────────────────────────────────────
def _img_ratio(png):
    try:
        ir = ImageReader(io.BytesIO(png))
        w, h = ir.getSize()
        return h / w
    except Exception:
        return 0.5


def _mean_conf(preds):
    cs = [p["confidence"] for p in preds if p.get("confidence") is not None]
    return ("%.2f" % (sum(cs) / len(cs))) if cs else "n/a"


def _is_urgent(p):
    cls = (p.get("predicted_class") or "").lower()
    if "healthy" in cls:
        return False
    hot = ("black_rot", "yvmv", "enation", "late_blight", "clubroot",
           "anthracnose", "yellow_leaf_curl")
    conf = p.get("confidence") or 0
    return any(h in cls for h in hot) and conf >= 0.6


def _count_urgent(preds):
    return sum(1 for p in preds if _is_urgent(p))


def _urgent_cases(preds):
    return [p for p in preds if _is_urgent(p)]


def _trend_pairs(stats):
    bd, pd_ = stats["by_disease"], stats["prev_by_disease"]
    out = []
    for d in set(bd) | set(pd_):
        prev, cur = pd_.get(d, 0), bd.get(d, 0)
        if prev == 0:                       # new disease, handled elsewhere
            continue
        if prev != cur:
            out.append((d, prev, cur))
    out.sort(key=lambda x: -abs(x[2] - x[1]))
    return out


def _notable(preds):
    """The most report-worthy specimens: urgent first, then most confident
    diseased, dedup, capped."""
    urg = _urgent_cases(preds)
    dis = sorted([p for p in preds
                  if "healthy" not in (p.get("predicted_class") or "").lower()],
                 key=lambda p: -(p.get("confidence") or 0))
    seen, out = set(), []
    for p in urg + dis:
        if p.get("id") in seen:
            continue
        seen.add(p.get("id"))
        out.append(p)
    return out


def _watch_topic(stats):
    if stats["lead_disease"]:
        return "%s on the %s plots" % (
            disease_only(stats["lead_disease"]),
            stats.get("focus_crop") or "affected")
    return "the plots that were quiet this week"


def _ledger(b, c, preds, collector):
    cols = [(0, "#"), (16 * mm, "DATE"), (38 * mm, "CROP"),
            (62 * mm, "DISEASE"), (118 * mm, "TIER"), (140 * mm, "CONF")]

    def header():
        c.setFont(MONO, 7.5); c.setFillColor(INK_FAINT)
        for dx, name in cols:
            c.drawString(MARGIN + dx, b.y, name)
        b.y -= 4
        c.setStrokeColor(RULE); c.setLineWidth(0.6)
        c.line(MARGIN, b.y, PAGE_W - MARGIN, b.y)
        b.y -= 12

    header()
    for i, p in enumerate(preds):
        # Single pagination mechanism: break at the top of the row so the
        # column header is always redrawn on the new page. 30mm leaves room
        # for the 12pt row plus the bottom margin.
        if b.y < 30 * mm:
            b.page_break(collector)
            header()
        urgent = _is_urgent(p)
        if i % 2 == 0:
            c.setFillColor(PAPER_SOFT)
            c.rect(MARGIN, b.y - 3, CONTENT_W, 12, fill=1, stroke=0)
        if urgent:
            c.setFillColor(HexColor("#f6e3df"))
            c.rect(MARGIN, b.y - 3, CONTENT_W, 12, fill=1, stroke=0)
            c.setFillColor(CRIMSON)
            c.rect(MARGIN, b.y - 3, 2, 12, fill=1, stroke=0)
        conf = p.get("confidence")
        vals = [
            "%s" % (p.get("id") or "?"),
            str(p.get("created_at", ""))[:10],
            (p.get("crop") or "?").capitalize()[:9],
            pretty_class(p.get("predicted_class"))[:34],
            _tier_grade(p.get("tier"))[:12],
            ("%.0f%%" % (conf * 100)) if conf is not None else "n/a",
        ]
        c.setFont(SERIF, 8.6)
        c.setFillColor(CRIMSON if urgent else INK)
        for (dx, _), v in zip(cols, vals):
            c.drawString(MARGIN + dx + (3 if dx == 0 and urgent else 0),
                         b.y, v)
        b.y -= 12


def _summary(stats, total):
    return {
        "specimens": total,
        "diseases": stats["n_diseases"],
        "healthy": stats["healthy"],
        "new": stats["new_count"],
    }
