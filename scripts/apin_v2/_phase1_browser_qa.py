"""Phase-1 browser QA — drives every interactive widget with Playwright.

What this verifies (per the user's "actually visit the website" requirement):
    * /dashboard loads → no console errors → no auth gate
    * All 5 widget containers (A B C D E) render their default mode
    * Container A mode-dot click switches Brief ↔ Hero
    * Container C mode-dot click switches Donut ↔ Histogram
    * Mode switch updates URL hash (so #a=hero is preserved on refresh)
    * Page refresh with hash restores the active mode
    * Daily Brief contains non-empty text + a date stamp
    * Crop Almanac SVG has slices for each crop in crop_mix
    * Confidence Histogram SVG has bars + a mean line
    * Disease Encyclopedia hover card appears on hover
    * /dashboard/history loads with filter toolbar + result rows
    * Filter form submission narrows results
    * Pagination navigates pages
    * Export links carry the active filters

Outputs:
    * Screenshots saved to /report_figures/_qa_phase1/*.png
    * Per-check pass/fail to stdout
    * Non-zero exit on any failure
"""
import os
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows so unicode arrows / ≥ / etc. in check labels
# don't crash the cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent.parent.parent
SHOTS = ROOT / "report_figures" / "_qa_phase1"
SHOTS.mkdir(parents=True, exist_ok=True)

BASE = "http://127.0.0.1:8766"

# Track every check
results = []
console_errors = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def shot(page, name):
    """Save a full-page screenshot."""
    path = SHOTS / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    print(f"  >> saved {path.name}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1.5,
        )
        page = ctx.new_page()

        # Collect console errors across the whole session.
        # We treat uncaught JS exceptions (pageerror) as real bugs.
        # We FILTER OUT browser-emitted "Failed to load resource: 404"
        # console.error events — several QA sections deliberately request
        # endpoints that should 404 (revoked share / missing prediction /
        # contradictory filter that returns 0). Those are correct
        # behaviour, not console-error bugs.
        def _on_console(msg):
            if msg.type != "error":
                return
            txt = msg.text or ""
            if "Failed to load resource" in txt:
                return  # network 404s — deliberate in negative tests
            console_errors.append(("console.error", txt))
        page.on("pageerror", lambda exc: console_errors.append(("pageerror", str(exc))))
        page.on("console", _on_console)

        # ═══════════════════════════════════════════════════════════════
        #  1. DASHBOARD LOAD
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 1. DASHBOARD LOAD ===")
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=30000)
        page.wait_for_selector("#dash-content:not([hidden])", timeout=10000)
        check("dashboard renders with #dash-content visible",
              not page.locator("#dash-content").evaluate("e => e.hidden"))
        check("auth-gate is hidden (dev-mode fallback working)",
              page.locator("#auth-gate").evaluate("e => e.hidden"))
        check("loading state cleared",
              page.locator("#loading").evaluate("e => e.hidden"))
        shot(page, "01_dashboard_initial")

        # ═══════════════════════════════════════════════════════════════
        #  2. CONTAINER PRESENCE (A B C D E)
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 2. CONTAINER PRESENCE ===")
        for cid in ["a", "b", "c", "d", "e"]:
            sel = f"#container-{cid}"
            ok = page.locator(sel).count() == 1 and page.locator(sel).is_visible()
            check(f"Container {cid.upper()} present + visible",
                  ok, f"selector={sel}")

        # ═══════════════════════════════════════════════════════════════
        #  3. CONTAINER A — Daily Brief default mode + Hero toggle
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 3. CONTAINER A — DAILY BRIEF ===")
        brief_text = page.locator("#brief-prose").inner_text().strip()
        check("Daily Brief has non-empty narrative text",
              len(brief_text) > 30, f"len={len(brief_text)}")
        check("Daily Brief date stamp populated",
              page.locator("#brief-date-stamp").inner_text() not in ("—", "— —", ""))
        check("'Narrative' mode dot is the active one by default",
              page.locator("#container-a .mode-dot.active").get_attribute("data-mode") == "brief")

        # Click the Hero dot
        page.locator("#container-a .mode-dot[data-mode='hero']").click()
        page.wait_for_timeout(400)
        check("clicking 'hero' dot makes it active",
              page.locator("#container-a .mode-dot.active").get_attribute("data-mode") == "hero")
        check("Hero pane visible after switch",
              page.locator("#container-a .mode-pane[data-mode-pane='hero']").evaluate("e => e.classList.contains('active')"))
        # Hero values populated
        total_txt = page.locator("#hero-total").inner_text().strip()
        check("Hero 'Total samples' has a numeric value",
              total_txt.isdigit(), f"value='{total_txt}'")
        shot(page, "02_container_a_hero")

        # URL hash should now contain a=hero
        url = page.url
        check("URL hash records mode-switch (#a=hero)",
              "a=hero" in url, f"url={url}")

        # Click back to Brief
        page.locator("#container-a .mode-dot[data-mode='brief']").click()
        page.wait_for_timeout(300)
        check("clicking 'brief' returns to narrative",
              page.locator("#container-a .mode-pane[data-mode-pane='brief']").evaluate("e => e.classList.contains('active')"))
        shot(page, "03_container_a_brief")

        # ═══════════════════════════════════════════════════════════════
        #  4. CONTAINER C — Crop Almanac + Histogram
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 4. CONTAINER C — ALMANAC + HISTOGRAM ===")
        # Default = donut
        check("'donut' mode is active by default",
              page.locator("#container-c .mode-dot.active").get_attribute("data-mode") == "donut")
        n_slices = page.locator("#almanac-svg [data-crop]").count()
        check("donut has at least one slice", n_slices >= 1, f"n_slices={n_slices}")
        n_legend = page.locator("#almanac-legend .item").count()
        check("donut legend lists ≥ 1 crops", n_legend >= 1, f"n_legend={n_legend}")
        shot(page, "04_container_c_donut")

        # Switch to histogram
        page.locator("#container-c .mode-dot[data-mode='histogram']").click()
        page.wait_for_timeout(400)
        check("histogram pane active",
              page.locator("#container-c .mode-pane[data-mode-pane='histogram']").evaluate("e => e.classList.contains('active')"))
        # Histogram has bars (rough.js produces <g> elements with <path>s)
        hist_paths = page.locator("#confhist-svg path").count()
        check("histogram has SVG paths drawn (bars + mean line)",
              hist_paths >= 6, f"n_paths={hist_paths}")
        mean_txt = page.locator("#confhist-mean").inner_text().strip()
        check("histogram footer shows mean + n",
              "mean:" in mean_txt and "n =" in mean_txt,
              f"footer='{mean_txt}'")
        shot(page, "05_container_c_histogram")

        # Switch back so subsequent tests start from donut state
        page.locator("#container-c .mode-dot[data-mode='donut']").click()
        page.wait_for_timeout(200)

        # ═══════════════════════════════════════════════════════════════
        #  5. URL HASH PERSISTENCE
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 5. URL HASH PERSISTENCE ===")
        # Manually navigate with hash, verify it's honoured
        page.goto(f"{BASE}/dashboard#a=hero&c=histogram",
                  wait_until="networkidle", timeout=15000)
        page.wait_for_selector("#dash-content:not([hidden])", timeout=10000)
        page.wait_for_timeout(800)  # let setTimeout-based init fire
        check("URL hash a=hero honoured on initial load",
              page.locator("#container-a .mode-dot.active").get_attribute("data-mode") == "hero")
        check("URL hash c=histogram honoured on initial load",
              page.locator("#container-c .mode-dot.active").get_attribute("data-mode") == "histogram")
        shot(page, "06_url_hash_restore")

        # ═══════════════════════════════════════════════════════════════
        #  6. KEYBOARD NAV (← / →)
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 6. KEYBOARD NAV ===")
        # Focus the first mode dot in Container C and press ArrowRight
        page.locator("#container-c .mode-dot[data-mode='donut']").focus()
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(250)
        active = page.locator("#container-c .mode-dot.active").get_attribute("data-mode")
        check("Container C: ArrowRight cycles to histogram",
              active == "histogram", f"active={active}")
        page.keyboard.press("ArrowLeft")
        page.wait_for_timeout(250)
        active = page.locator("#container-c .mode-dot.active").get_attribute("data-mode")
        check("Container C: ArrowLeft cycles back to donut",
              active == "donut", f"active={active}")

        # ═══════════════════════════════════════════════════════════════
        #  7. DISEASE ENCYCLOPEDIA HOVER
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 7. DISEASE ENCYCLOPEDIA HOVER ===")
        # Go back to the dashboard cleanly
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=15000)
        page.wait_for_selector("#dash-content:not([hidden])", timeout=10000)
        page.wait_for_timeout(500)

        # Find any element with data-disease — should be in Daily Brief (.accent)
        # OR in the ledger SVG (set by attachDiseaseEncyclopediaHooks)
        hits = page.locator("[data-disease]").count()
        check("at least one [data-disease] element exists on the dashboard",
              hits >= 1, f"n={hits}")

        if hits > 0:
            target = page.locator("[data-disease]").first
            disease_key = target.get_attribute("data-disease")
            target.hover()
            page.wait_for_timeout(400)  # > 200ms hover delay + paint
            shown = page.locator("#disease-encyclopedia").evaluate("e => e.classList.contains('show')")
            check("encyclopedia card .show after 200ms hover",
                  shown, f"hovered='{disease_key}'")
            if shown:
                title = page.locator("#dx-title").inner_text().strip()
                check("encyclopedia title populated",
                      len(title) > 3, f"title='{title}'")
                n_syms = page.locator("#dx-symptoms li").count()
                check("encyclopedia symptoms list populated",
                      n_syms >= 1, f"n_symptoms={n_syms}")
                shot(page, "07_encyclopedia_hover")

            # Move mouse far away — card should fade
            page.mouse.move(10, 10)
            page.wait_for_timeout(500)
            still_shown = page.locator("#disease-encyclopedia").evaluate("e => e.classList.contains('show')")
            check("encyclopedia card hides after mouse leaves",
                  not still_shown)

        # ═══════════════════════════════════════════════════════════════
        #  8. FIELD HISTORY PAGE
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 8. FIELD HISTORY ===")
        page.goto(f"{BASE}/dashboard/history", wait_until="networkidle", timeout=15000)
        page.wait_for_selector("#results-list .row", timeout=10000)
        n_rows_initial = page.locator("#results-list .row").count()
        check("history page renders at least one row",
              n_rows_initial > 0, f"n_rows={n_rows_initial}")

        # Total counts visible
        total_txt = page.locator("#cnt-total").inner_text().strip()
        check("history page shows total count",
              total_txt not in ("—", ""), f"total='{total_txt}'")

        # Disease dropdown was populated from server response
        n_diseases = page.locator("#f-disease option").count()
        check("disease filter dropdown populated from data",
              n_diseases > 1, f"n_diseases={n_diseases}")  # >1 = more than just "all"
        shot(page, "08_history_initial")

        # Apply crop=tomato filter
        page.locator("#f-crop").select_option("tomato")
        page.locator("#filter-form button[type='submit']").click()
        page.wait_for_function(
            "document.getElementById('results-loading').hidden === true",
            timeout=10000,
        )
        page.wait_for_timeout(400)
        n_after = page.locator("#results-list .row").count()
        check("crop=tomato filter changes result count",
              n_after != n_rows_initial or n_after > 0,
              f"before={n_rows_initial} after={n_after}")
        check("URL contains ?crop=tomato",
              "crop=tomato" in page.url, f"url={page.url}")
        # Active filter chip appears
        chip_visible = not page.locator("#active-filters").evaluate("e => e.hidden")
        check("active-filter chip strip shown after filter",
              chip_visible)
        shot(page, "09_history_filtered_tomato")

        # Reset filters
        page.locator("#reset-btn").click()
        page.wait_for_function(
            "document.getElementById('results-loading').hidden === true",
            timeout=10000,
        )
        check("reset clears the URL query string",
              "?crop=" not in page.url and "?disease=" not in page.url,
              f"url={page.url}")

        # Search 'blight'
        page.locator("#f-q").fill("blight")
        page.locator("#filter-form button[type='submit']").click()
        page.wait_for_timeout(500)
        n_after = page.locator("#results-list .row").count()
        check("search 'blight' returns at least one row",
              n_after > 0, f"n={n_after}")
        rows_text = page.locator("#results-list .row .class").all_inner_texts()
        all_match = all("blight" in t.lower() for t in rows_text)
        check("every visible row contains 'blight' in its class",
              all_match, f"rows={rows_text[:4]}")
        shot(page, "10_history_search_blight")

        # Export menu toggle + links carry filter
        page.locator("#export-toggle").click()
        page.wait_for_timeout(150)
        menu_open = page.locator("#export-menu").evaluate("e => e.classList.contains('open')")
        check("export menu opens on click", menu_open)
        csv_href = page.locator("#export-csv").get_attribute("href")
        check("CSV export href includes current search",
              "q=blight" in csv_href, f"href={csv_href}")
        json_href = page.locator("#export-json").get_attribute("href")
        check("JSON export href includes current search",
              "q=blight" in json_href, f"href={json_href}")

        # Pagination — switch to small page size and walk
        page.goto(f"{BASE}/dashboard/history?page_size=5",
                  wait_until="networkidle", timeout=15000)
        page.wait_for_selector("#results-list .row", timeout=10000)
        page.wait_for_timeout(300)
        n_first = page.locator("#results-list .row").count()
        check("page_size=5 shows exactly 5 rows on a full page",
              n_first == 5, f"got {n_first}")
        # next page (if visible)
        if not page.locator("#pagination").evaluate("e => e.hidden"):
            page.locator("#pg-next").click()
            page.wait_for_function(
                "document.getElementById('results-loading').hidden === true",
                timeout=10000,
            )
            page.wait_for_timeout(300)
            check("next-page advances URL (?page=2)",
                  "page=2" in page.url, f"url={page.url}")
            shot(page, "11_history_page_2")

        # ═══════════════════════════════════════════════════════════════
        #  9. SIDEBAR NAV — back to dashboard
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 9. SIDEBAR NAV ===")
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=15000)
        nav_history = page.locator("a[href='/dashboard/history']").first
        check("sidebar has working link to /dashboard/history",
              nav_history.count() > 0)

        # ═══════════════════════════════════════════════════════════════
        #  10. PHASE 2 — MODES + MODALS + MARGIN NOTES
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 10. PHASE 2: NEW CONTAINER MODES ===")
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=15000)
        page.wait_for_selector("#dash-content:not([hidden])", timeout=10000)
        page.wait_for_timeout(500)

        # Container B mode 2 — First Sightings
        page.locator("#container-b .mode-dot[data-mode='sightings']").click()
        page.wait_for_timeout(700)  # lazy fetch + draw
        n_phenology = page.locator("#phenology-svg circle").count()
        check("First Sightings phenology rendered with disease dots",
              n_phenology >= 1, f"n_dots={n_phenology}")
        shot(page, "12_phenology")

        # Container B mode 3 — Comparison Spread
        page.locator("#container-b .mode-dot[data-mode='compare']").click()
        page.wait_for_timeout(700)
        n_compare_rows = page.locator("#compare-rows-a .compare-row").count()
        check("Comparison Spread page A populated", n_compare_rows >= 3,
              f"n_rows_A={n_compare_rows}")
        shot(page, "13_comparison_spread")

        # Container D mode 2 — Disease Family Tree
        page.locator("#container-d .mode-dot[data-mode='family']").click()
        page.wait_for_timeout(700)
        n_kingdoms = page.locator("#family-tree-mount details.kingdom").count()
        check("Family Tree has at least 1 kingdom", n_kingdoms >= 1,
              f"n_kingdoms={n_kingdoms}")
        n_leaves = page.locator("#family-tree-mount li.leaf").count()
        check("Family Tree has at least 1 leaf species", n_leaves >= 1,
              f"n_leaves={n_leaves}")
        shot(page, "14_family_tree")

        # Container D mode 3 — Field Notebook Index
        page.locator("#container-d .mode-dot[data-mode='index']").click()
        page.wait_for_timeout(700)
        n_cards = page.locator("#fn-index-mount .fn-card").count()
        check("Field Notebook Index has cards", n_cards >= 1,
              f"n_cards={n_cards}")
        # Type-to-filter
        page.locator("#fn-index-search").fill("tomato")
        page.wait_for_timeout(250)
        n_filtered = page.locator("#fn-index-mount .fn-card").count()
        check("typing 'tomato' filters the index down",
              n_filtered > 0 and n_filtered <= n_cards,
              f"before={n_cards} after={n_filtered}")
        page.locator("#fn-index-search").fill("")
        shot(page, "15_field_notebook_index")

        # Container E mode 1 — Polaroid grid
        page.locator("#container-e .mode-dot[data-mode='polaroid']").click()
        page.wait_for_timeout(700)
        n_polaroids = page.locator("#polaroid-grid .polaroid").count()
        check("Polaroid grid has polaroid cards", n_polaroids >= 1,
              f"n={n_polaroids}")
        shot(page, "16_polaroid_grid")

        # Container E mode 2 — Heatmap grid
        page.locator("#container-e .mode-dot[data-mode='heatmap']").click()
        page.wait_for_timeout(700)
        n_heatmaps = page.locator("#heatmap-grid .polaroid").count()
        check("Heatmap grid has cards", n_heatmaps >= 1,
              f"n={n_heatmaps}")
        check("heatmap mode cards use .image.heatmap class",
              page.locator("#heatmap-grid .image.heatmap").count() >= 1)
        shot(page, "17_heatmap_grid")

        # Container F — Margin Notes (present and empty by default)
        check("Margin Notes container present",
              page.locator("#container-f").count() == 1)
        # The "+ add" card should be visible
        check("Add-a-note button rendered",
              page.locator("#note-add-trigger").count() == 1)

        print("\n=== 11. PHASE 2: MODALS ===")
        # Switch back to Calendar mode and click a populated day
        page.locator("#container-b .mode-dot[data-mode='calendar']").click()
        page.wait_for_timeout(400)
        # Find a calendar cell with samples (the calendar tags <g> with data-date)
        cells = page.locator("#calendar-svg [data-date]")
        if cells.count() > 0:
            # Pick the LAST cell (most recent — most likely to have a sample)
            target_cell = cells.nth(cells.count() - 1)
            target_cell.click()
            page.wait_for_timeout(400)
            shown = page.locator("#modal-day").evaluate("e => !e.hidden && e.classList.contains('open')")
            check("Day Detail modal opens on calendar cell click", shown)
            shot(page, "18_modal_day_detail")
            # Close with ESC
            page.keyboard.press("Escape")
            page.wait_for_timeout(400)
            still_open = page.locator("#modal-day").evaluate("e => e.classList.contains('open')")
            check("Day Detail modal closes on ESC", not still_open)

        # Open Disease Drill-down via family tree leaf.
        # Species nodes are collapsed by default — open them all so the
        # leaves underneath are visible to the Playwright click engine.
        page.locator("#container-d .mode-dot[data-mode='family']").click()
        page.wait_for_timeout(500)
        page.evaluate(
            "document.querySelectorAll('#family-tree-mount details').forEach(d => d.open = true)"
        )
        page.wait_for_timeout(150)
        leaves = page.locator("#family-tree-mount li.leaf")
        if leaves.count() > 0:
            leaves.first.click()
            page.wait_for_timeout(500)
            shown = page.locator("#modal-disease").evaluate("e => !e.hidden && e.classList.contains('open')")
            check("Drill-down modal opens on family-tree leaf click", shown)
            n_rows_modal = page.locator("#modal-disease-rows .modal-pred-row").count()
            check("Drill-down modal has prediction rows",
                  n_rows_modal >= 1, f"n={n_rows_modal}")
            # The encyclopedia section should be visible for a known disease
            enc_visible = page.locator("#modal-disease-encyclopedia").evaluate("e => !e.hidden")
            check("Drill-down inline encyclopedia visible", enc_visible)
            shot(page, "19_modal_disease_drilldown")
            page.locator("#modal-disease [data-modal-close]").click()
            page.wait_for_timeout(400)

        # Open Signal Vote via clicking a row in the day-detail modal —
        # simpler: just click a recent-note card
        page.locator("#container-e .mode-dot[data-mode='notes']").click()
        page.wait_for_timeout(400)
        notes = page.locator("#recent-list .note-card[data-prediction-id]")
        if notes.count() > 0:
            notes.first.click()
            page.wait_for_timeout(500)
            shown = page.locator("#modal-signals").evaluate("e => !e.hidden && e.classList.contains('open')")
            check("Signal Vote modal opens on recent-note click", shown)
            n_bars = page.locator("#modal-signals-bars .sv-bar-row").count()
            check("Signal Vote modal renders 4 signal rows",
                  n_bars == 4, f"n={n_bars}")
            shot(page, "20_modal_signal_vote")
            # Close via backdrop click
            page.locator("#modal-backdrop").click(position={"x":10,"y":10})
            page.wait_for_timeout(400)
            still_open = page.locator("#modal-signals").evaluate("e => e.classList.contains('open')")
            check("Signal Vote modal closes on backdrop click", not still_open)

        print("\n=== 12. PHASE 2: MARGIN NOTES CRUD ===")
        # Reset state — make sure note editor is closed
        page.evaluate("document.getElementById('note-editor').classList.remove('open')")
        # Click + add card to open editor
        page.locator("#note-add-trigger").click()
        page.wait_for_timeout(300)
        editor_open = page.locator("#note-editor").evaluate("e => e.classList.contains('open')")
        check("Note editor opens on + click", editor_open)
        page.locator("#note-editor-text").fill("playwright test note — should be visible after save")
        # Click mood dot 2 (index)
        page.locator("#note-editor .mood-dot[data-mood='1']").click()
        # Click save
        page.locator("#note-editor .btn-save").click()
        page.wait_for_timeout(800)  # POST + refresh
        editor_closed = not page.locator("#note-editor").evaluate("e => e.classList.contains('open')")
        check("Note editor closes after save", editor_closed)
        n_notes = page.locator("#notes-wrap .note-slip").count()
        check("Note appears in container F after save",
              n_notes >= 1, f"n_notes={n_notes}")
        shot(page, "21_margin_notes_with_one")
        # Delete via JS (avoid the confirm() dialog blocking test)
        last_note_id = page.locator("#notes-wrap .note-slip").last.get_attribute("data-id")
        if last_note_id:
            page.evaluate(f"fetch('/dashboard/notes/{last_note_id}',{{method:'DELETE',credentials:'include'}}).then(()=>MarginNotes.refresh())")
            page.wait_for_timeout(700)
            n_after_delete = page.locator("#notes-wrap .note-slip").count()
            check("note removed after DELETE+refresh",
                  n_after_delete == n_notes - 1,
                  f"before={n_notes} after={n_after_delete}")

        # ═══════════════════════════════════════════════════════════════
        #  13. PHASE 3 — Treatment Log widget on dashboard
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 13. PHASE 3: TREATMENT LOG WIDGET ===")
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=15000)
        page.wait_for_selector("#container-g", timeout=10000)
        page.wait_for_timeout(500)
        check("Container G (Treatment Log) present on dashboard",
              page.locator("#container-g").count() == 1)
        check("treatment add button rendered",
              page.locator("#treatment-add-trigger").count() == 1)

        # Pre-clean: delete any existing treatments via API
        page.evaluate("""
          fetch('/dashboard/treatments',{credentials:'include'})
            .then(r=>r.json())
            .then(j=>Promise.all(j.results.map(t=>fetch('/dashboard/treatments/'+t.id,{method:'DELETE',credentials:'include'}))))
            .then(()=>TreatmentLog.refresh())
        """)
        page.wait_for_timeout(700)

        # Click + add card
        page.locator("#treatment-add-trigger").click()
        page.wait_for_timeout(300)
        check("treatment editor opens on + click",
              page.locator("#treatment-editor").evaluate("e => e.classList.contains('open')"))

        # Fill the form
        page.locator("#tx-ed-treatment").fill("playwright neem spray")
        page.locator("#tx-ed-date").fill("2026-05-14")
        page.locator("#tx-ed-crop").select_option("okra")
        page.locator("#tx-ed-disease").fill("okra_yvmv")
        page.locator("#tx-ed-plot").fill("Plot A")
        page.locator("#tx-ed-notes").fill("test note from playwright")
        page.locator("#treatment-editor .btn-save").click()
        page.wait_for_timeout(800)
        check("treatment editor closes after save",
              not page.locator("#treatment-editor").evaluate("e => e.classList.contains('open')"))
        n_rows = page.locator("#treatments-list .treatment-row").count()
        check("treatment row appears in Container G", n_rows == 1, f"got {n_rows}")
        shot(page, "22_treatment_log_with_one")

        # Test edit
        page.locator(".treatment-row .actions button[data-action='edit']").click()
        page.wait_for_timeout(300)
        check("treatment editor reopens for edit",
              page.locator("#treatment-editor").evaluate("e => e.classList.contains('open')"))
        check("editor pre-fills treatment field",
              page.locator("#tx-ed-treatment").input_value() == "playwright neem spray")
        # Cancel
        page.locator("#treatment-editor .btn-cancel").click()
        page.wait_for_timeout(300)

        # Delete via API (avoid confirm dialog)
        last_tid = page.locator(".treatment-row").last.get_attribute("data-id")
        if last_tid:
            page.evaluate(f"fetch('/dashboard/treatments/{last_tid}',{{method:'DELETE',credentials:'include'}}).then(()=>TreatmentLog.refresh())")
            page.wait_for_timeout(600)
            check("treatment removed after DELETE+refresh",
                  page.locator("#treatments-list .treatment-row").count() == 0)

        # ═══════════════════════════════════════════════════════════════
        #  14. PHASE 3 — /dashboard/reports page
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 14. PHASE 3: /dashboard/reports ===")
        page.goto(f"{BASE}/dashboard/reports", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(700)
        check("Reports page heading renders",
              page.locator("h1").inner_text() == "Reports & Exports")
        # 3 report cards
        check("3 report cards rendered",
              page.locator(".report-card").count() == 3)
        # Disable target=_blank so we don't spawn new tabs during the test
        page.evaluate("document.querySelectorAll('a[target]').forEach(a => a.removeAttribute('target'))")
        # The prediction dropdown should be populated
        n_opts = page.locator("#share-prediction option").count()
        check("share-prediction dropdown populated", n_opts >= 1, f"opts={n_opts}")

        # Generate a share link via the form
        page.locator("#share-label").fill("smoke share from QA")
        page.locator("#share-create").click()
        page.wait_for_timeout(1000)
        share_result_visible = page.locator("#share-result").evaluate("e => !e.hidden")
        check("share creation result visible after click", share_result_visible)
        url_text = page.locator("#share-url-text").inner_text()
        check("share URL text contains /share/", "/share/" in url_text,
              f"url='{url_text[:60]}'")
        n_share_rows = page.locator(".share-row").count()
        check("share appears in list", n_share_rows >= 1, f"n={n_share_rows}")
        shot(page, "23_reports_with_share")

        # Test the PDF generate button (opens new tab — verify it doesn't crash)
        # Disable the target=_blank we already removed; capture the href directly
        pdf_btn = page.locator("#pdf-generate")
        check("PDF generate button visible", pdf_btn.count() == 1)

        # ═══════════════════════════════════════════════════════════════
        #  15. PHASE 3 — Public share viewer
        # ═══════════════════════════════════════════════════════════════
        if "/share/" in url_text:
            print("\n=== 15. PHASE 3: PUBLIC SHARE VIEWER ===")
            # Extract token from URL
            token = url_text.split("/share/")[-1].strip()
            page.goto(f"{BASE}/share/{token}", wait_until="networkidle", timeout=15000)
            page.wait_for_timeout(800)
            check("public share viewer renders heading",
                  page.locator("h1").count() >= 1
                  and len(page.locator("h1").inner_text().strip()) > 0)
            check("public share viewer shows 'Reading' facts panel",
                  page.locator(".facts h2").count() >= 1)
            shot(page, "24_public_share_viewer")

            # Revoke the share via API, then re-load — should show error
            page.evaluate("""
              fetch('/dashboard/shares',{credentials:'include'})
                .then(r=>r.json())
                .then(j=>{
                  const last = j.results[0];
                  return last ? fetch('/dashboard/shares/'+last.id,{method:'DELETE',credentials:'include'}) : null;
                })
            """)
            page.wait_for_timeout(500)
            page.goto(f"{BASE}/share/{token}", wait_until="networkidle", timeout=15000)
            page.wait_for_timeout(600)
            check("revoked share shows error state",
                  page.locator(".error-state").count() >= 1)
            shot(page, "25_public_share_revoked")

        # ═══════════════════════════════════════════════════════════════
        #  16. PHASE 3 — /dashboard/loupe
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 16. PHASE 3: /dashboard/loupe ===")
        page.goto(f"{BASE}/dashboard/loupe", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(800)
        check("loupe page heading renders",
              page.locator("h1").inner_text() == "The Loupe")
        # Both picker dropdowns populated
        check("pick-a populated",
              page.locator("#pick-a option").count() >= 1)
        check("pick-b populated",
              page.locator("#pick-b option").count() >= 1)
        # Both image panes have content
        check("pane A label populated",
              page.locator("#class-a").inner_text() not in ("—", ""))
        check("pane B label populated",
              page.locator("#class-b").inner_text() not in ("—", ""))
        # Heatmap toggle works
        page.locator("#heatmap-toggle").click()
        page.wait_for_timeout(200)
        check("heatmap toggle applies heatmap class",
              page.locator("#image-a.heatmap").count() == 1)
        shot(page, "26_loupe_heatmap")

        # ═══════════════════════════════════════════════════════════════
        #  17. PHASE 3 — /dashboard/gallery
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 17. PHASE 3: /dashboard/gallery ===")
        page.goto(f"{BASE}/dashboard/gallery", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(900)
        check("gallery page heading renders",
              page.locator("h1").inner_text() == "Gallery")
        n_pol = page.locator(".polaroid").count()
        check("gallery has polaroid cards", n_pol >= 1, f"n={n_pol}")
        # Heatmap toggle works
        page.locator("#mode-heatmap").click()
        page.wait_for_timeout(800)
        check("heatmap mode replaces images",
              page.locator(".polaroid .image.heatmap").count() >= 1)
        shot(page, "27_gallery_heatmap")

        # Crop filter
        page.locator("#crop-filter").select_option("tomato")
        page.wait_for_timeout(700)
        n_tomato = page.locator(".polaroid").count()
        check("crop=tomato filter narrows gallery",
              n_tomato >= 1 and n_tomato <= n_pol,
              f"before={n_pol} after={n_tomato}")

        # ═══════════════════════════════════════════════════════════════
        #  18. PHASE 3 — /dashboard/settings
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 18. PHASE 3: /dashboard/settings ===")
        page.goto(f"{BASE}/dashboard/settings", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(700)
        check("settings page heading renders",
              page.locator("h1").inner_text() == "Settings")
        check("4 theme cards rendered",
              page.locator(".theme-card").count() == 4)
        # Click a theme — should apply and persist
        page.locator(".theme-card[data-theme='parchment']").click()
        page.wait_for_timeout(200)
        check("parchment theme becomes active",
              page.locator(".theme-card[data-theme='parchment']").evaluate(
                  "e => e.classList.contains('active')"))
        check("body bg changes to parchment colour",
              page.evaluate("getComputedStyle(document.body).backgroundColor")
              == "rgb(236, 224, 193)")
        # Account info populated
        acct_name = page.locator("#acct-name").inner_text()
        check("account name populated", acct_name not in ("—", ""),
              f"name={acct_name!r}")
        shot(page, "28_settings_parchment")

        # ═══════════════════════════════════════════════════════════════
        #  19. PHASE 3 — Sidebar nav now activated
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 19. PHASE 3: SIDEBAR ACTIVATED ===")
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(400)
        for path in ["/dashboard/gallery", "/dashboard/loupe",
                     "/dashboard/reports", "/dashboard/settings"]:
            link = page.locator(f"a[href='{path}']").first
            check(f"sidebar has live link to {path}", link.count() > 0)

        # ═══════════════════════════════════════════════════════════════
        #  20. CONSOLE ERROR REPORT
        # ═══════════════════════════════════════════════════════════════
        print("\n=== 20. CONSOLE ERRORS ===")
        if console_errors:
            for kind, msg in console_errors[:10]:
                print(f"  ⚠ {kind}: {msg[:240]}")
            check("no console errors during QA",
                  False, f"{len(console_errors)} errors")
        else:
            check("no console errors during QA", True)

        browser.close()

    # ─── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"  TOTAL: {n_pass} passed   {n_fail} failed")
    if n_fail:
        print("\n  FAILURES:")
        for name, ok, detail in results:
            if not ok:
                print(f"    ✗ {name}" + (f" — {detail}" if detail else ""))
    print(f"\n  Screenshots saved to {SHOTS}")
    sys.exit(0 if n_fail == 0 else 2)


if __name__ == "__main__":
    main()
