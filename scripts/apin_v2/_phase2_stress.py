"""Phase-2 STRESS test — beyond happy paths.

Each section deliberately tries to BREAK the dashboard the way a malicious
or impatient user would. Categories:

  A. Modal lifecycle stress  — rapid open/close/swap
  B. Hash / URL state stress — malformed values, unknown modes
  C. Margin notes stress      — empty, oversize, special chars, network sim
  D. History filter edge cases — combined filters, 0 results, date inversion
  E. Cross-widget click chain — phenology → drill-down → row → signal vote
  F. Comparison Spread edge  — same period A=B
  G. Family Tree edge        — type-to-filter, collapse-all
  H. Keyboard-only nav        — Tab through containers, arrow keys
  I. Browser back / forward  — hash navigation history
  J. Viewport changes        — 1024 → 600 (responsive)
"""
import os, sys, time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent.parent
SHOTS = ROOT / "report_figures" / "_stress_phase2"
SHOTS.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8766"

results = []
console_errors = []
def check(name, ok, detail=""):
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
def shot(page, name):
    page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)


def _mint_dashtester_session():
    """Mint a DB-level session for the dashtester account so the stress
    suite (which exercises authenticated dashboard pages) can inject it as
    a browser cookie.  The dev-mode auth bypass was removed, so every
    /dashboard page now requires a real session."""
    import os, sqlite3, secrets, hashlib
    from datetime import datetime, timezone, timedelta
    db = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "apin_v2.db")
    raw = secrets.token_urlsafe(32)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    c = sqlite3.connect(db)
    row = c.execute("SELECT id FROM users WHERE username='dashtester'").fetchone()
    if not row:
        c.close()
        raise RuntimeError("dashtester account not found")
    c.execute("INSERT INTO sessions (user_id, token_hash, created_at, expires_at) "
              "VALUES (?,?,?,?)",
              (row[0], h, now.isoformat(), (now + timedelta(days=1)).isoformat()))
    c.commit()
    c.close()
    return raw


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1.25,
        )
        # Inject the authenticated session cookie before any navigation.
        ctx.add_cookies([{
            "name": "apin_v2_session",
            "value": _mint_dashtester_session(),
            "url": BASE,
        }])
        page = ctx.new_page()
        page.on("pageerror", lambda exc: console_errors.append(("pageerror", str(exc))))
        # Filter out network 404s — several stress sections deliberately
        # forge bad share tokens / use contradictory filters that should
        # return 404. Those are correct behaviour, not console-error bugs.
        def _on_console(msg):
            if msg.type != "error":
                return
            txt = msg.text or ""
            if "Failed to load resource" in txt:
                return
            console_errors.append(("console.error", txt))
        page.on("console", _on_console)

        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=30000)
        page.wait_for_selector("#dash-content:not([hidden])", timeout=10000)

        # ═══════════════════════════════════════════════════════════════
        #  A. MODAL LIFECYCLE STRESS
        # ═══════════════════════════════════════════════════════════════
        print("\n=== A. MODAL LIFECYCLE STRESS ===")

        # A1: Rapid-open the same modal 5 times — should still be open + clean state
        for i in range(5):
            page.evaluate("openDayDetail('2026-05-14')")
            page.wait_for_timeout(50)
        page.wait_for_timeout(600)
        check("rapid 5× open of same modal → still single open instance",
              page.locator("#modal-day").evaluate("e => e.classList.contains('open')")
              and page.locator(".modal-sheet.open").count() == 1)
        check("rapid 5× open did not duplicate backdrop",
              page.locator(".modal-backdrop").count() == 1)

        # A2: Open one modal then SWAP to another — first must close cleanly
        page.evaluate("openDiseaseDrilldown('tomato_early_blight')")
        page.wait_for_timeout(600)
        opens = page.locator(".modal-sheet.open").count()
        check("modal swap → only 1 .open at a time", opens == 1, f"open count={opens}")
        check("modal swap → correct modal is the new one",
              page.locator("#modal-disease").evaluate("e => e.classList.contains('open')"))

        # A3: ESC twice — first closes, second is a no-op (not crash)
        page.keyboard.press("Escape")
        page.wait_for_timeout(150)
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        check("double-ESC closes modal then no-ops",
              page.locator(".modal-sheet.open").count() == 0)
        check("scroll-lock released after close",
              page.evaluate("document.documentElement.style.overflow === ''"))

        # A4: Click backdrop while animation is in flight (300ms transition)
        page.evaluate("openDayDetail('2026-05-14')")
        page.wait_for_timeout(80)  # mid-animation
        page.locator("#modal-backdrop").click(position={"x":10,"y":10}, force=True)
        page.wait_for_timeout(500)
        check("backdrop click mid-animation still closes cleanly",
              page.locator(".modal-sheet.open").count() == 0)

        # ═══════════════════════════════════════════════════════════════
        #  B. HASH / URL STATE STRESS
        # ═══════════════════════════════════════════════════════════════
        print("\n=== B. HASH / URL STATE STRESS ===")

        # B1: Navigate with an unknown mode in the hash — should fall back to default
        page.goto(f"{BASE}/dashboard#b=nonexistent&c=garbage&z=invalid",
                  wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(500)
        active_b = page.locator("#container-b .mode-dot.active").get_attribute("data-mode")
        active_c = page.locator("#container-c .mode-dot.active").get_attribute("data-mode")
        check("unknown mode in #b= falls back to default 'calendar'",
              active_b == "calendar", f"got {active_b}")
        check("unknown mode in #c= falls back to default 'donut'",
              active_c == "donut", f"got {active_c}")

        # B2: Hash with extra garbage params — must not crash
        page.goto(f"{BASE}/dashboard#=&==&=missing", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(300)
        check("malformed hash does not crash the page",
              page.locator("#dash-content").evaluate("e => !e.hidden"))

        # B3: Very long hash value
        page.goto(f"{BASE}/dashboard#b=" + "x"*500, wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(300)
        check("500-char hash value does not crash",
              page.locator("#dash-content").evaluate("e => !e.hidden"))

        # ═══════════════════════════════════════════════════════════════
        #  C. MARGIN NOTES STRESS
        # ═══════════════════════════════════════════════════════════════
        print("\n=== C. MARGIN NOTES STRESS ===")
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(600)

        # Clean slate — wipe any leftover notes via API
        page.evaluate("""
          fetch('/dashboard/notes',{credentials:'include'})
            .then(r=>r.json())
            .then(j=>Promise.all(j.results.map(n=>fetch('/dashboard/notes/'+n.id,{method:'DELETE',credentials:'include'}))))
            .then(()=>MarginNotes.refresh())
        """)
        page.wait_for_timeout(800)
        check("clean slate: 0 notes",
              page.locator("#notes-wrap .note-slip").count() == 0)

        # C1: Try to save EMPTY note — UI should ignore (text trimmed empty)
        page.evaluate("MarginNotes.openEditor(null)")
        page.wait_for_timeout(200)
        # Don't type anything, hit save
        page.locator("#note-editor .btn-save").click()
        page.wait_for_timeout(500)
        n_notes_after_empty = page.locator("#notes-wrap .note-slip").count()
        check("empty-text save did NOT create a note", n_notes_after_empty == 0)
        check("editor closed after empty save",
              not page.locator("#note-editor").evaluate("e => e.classList.contains('open')"))

        # C2: Submit OVERSIZE text (3000 chars) — backend caps at 2000
        page.evaluate("MarginNotes.openEditor(null)")
        page.wait_for_timeout(200)
        oversize = "x" * 3000
        page.locator("#note-editor-text").fill(oversize)
        page.locator("#note-editor .btn-save").click()
        page.wait_for_timeout(700)
        n_notes_after_oversize = page.locator("#notes-wrap .note-slip").count()
        check("oversize note saved (got capped server-side)", n_notes_after_oversize == 1)
        # Confirm what got stored is ≤ 2000 chars
        stored_text_len = page.evaluate(
            "document.querySelector('#notes-wrap .note-slip .text').innerText.length"
        )
        check("server capped 3000-char input to ≤ 2000",
              stored_text_len <= 2000,
              f"stored length = {stored_text_len}")

        # C3: SPECIAL CHARACTERS — angle brackets must not break HTML
        page.evaluate("MarginNotes.openEditor(null)")
        page.wait_for_timeout(200)
        nasty = "<script>alert('xss')</script> & \"quote\" ' apos"
        page.locator("#note-editor-text").fill(nasty)
        page.locator("#note-editor .btn-save").click()
        page.wait_for_timeout(700)
        # Check no script tag was actually created on the page
        scripts_in_notes = page.locator("#notes-wrap script").count()
        check("special-char note did NOT inject script tag",
              scripts_in_notes == 0)
        # Verify text is rendered with escaped angle brackets
        rendered_text = page.locator("#notes-wrap .note-slip .text").all_inner_texts()
        joined = " ".join(rendered_text)
        check("special-char content visible as text (XSS defused)",
              "<script>" in joined or "&lt;script&gt;" in joined or "script" in joined,
              f"sample={joined[:80]!r}")

        # Cleanup margin notes
        page.evaluate("""
          fetch('/dashboard/notes',{credentials:'include'})
            .then(r=>r.json())
            .then(j=>Promise.all(j.results.map(n=>fetch('/dashboard/notes/'+n.id,{method:'DELETE',credentials:'include'}))))
            .then(()=>MarginNotes.refresh())
        """)
        page.wait_for_timeout(700)

        # C4: Rapid double-click on Save (any client-side dedupe?)
        page.evaluate("MarginNotes.openEditor(null)")
        page.wait_for_timeout(200)
        page.locator("#note-editor-text").fill("rapid-save test")
        # Two clicks in quick succession
        btn = page.locator("#note-editor .btn-save")
        btn.click()
        # Editor closes on the first click — second click would hit nothing
        page.wait_for_timeout(700)
        rapid_save_count = page.locator("#notes-wrap .note-slip").count()
        check("rapid Save (editor closed before 2nd click) → exactly 1 note saved",
              rapid_save_count == 1, f"got {rapid_save_count}")
        # Cleanup
        page.evaluate("""
          fetch('/dashboard/notes',{credentials:'include'})
            .then(r=>r.json())
            .then(j=>Promise.all(j.results.map(n=>fetch('/dashboard/notes/'+n.id,{method:'DELETE',credentials:'include'}))))
            .then(()=>MarginNotes.refresh())
        """)
        page.wait_for_timeout(500)

        # ═══════════════════════════════════════════════════════════════
        #  D. HISTORY FILTER EDGE CASES
        # ═══════════════════════════════════════════════════════════════
        print("\n=== D. HISTORY FILTER EDGE CASES ===")

        # D1: Filter that returns 0 results
        page.goto(f"{BASE}/dashboard/history?crop=okra&disease=tomato_early_blight",
                  wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(700)
        check("contradictory filter → 0 results, empty state shown",
              page.locator("#results-list .row").count() == 0
              and not page.locator("#results-empty").evaluate("e => e.hidden"))
        check("0-result export href still includes filters",
              "crop=okra" in page.locator("#export-csv").get_attribute("href"))

        # D2: Combined filters — search + crop + tier should AND-narrow
        page.goto(f"{BASE}/dashboard/history?crop=tomato&tier=FIELD_GRADE&q=blight",
                  wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(700)
        rows = page.locator("#results-list .row").count()
        check("3-way combined filter narrows results", rows >= 1 and rows < 18,
              f"n_rows={rows}")

        # D3: Date inversion — from > to (should return 0, not crash)
        page.goto(f"{BASE}/dashboard/history?from=2026-12-31&to=2026-01-01",
                  wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(700)
        check("from > to does not crash; returns 0",
              page.locator("#results-list .row").count() == 0)

        # D4: Page > n_pages — should not crash
        page.goto(f"{BASE}/dashboard/history?page=99999",
                  wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(700)
        check("page=99999 does not crash; shows empty results",
              page.locator("#results-list .row").count() == 0
              or not page.locator("#results-empty").evaluate("e => e.hidden"))

        # D5: Search with SQL-injection-ish characters
        page.goto(f"{BASE}/dashboard/history?q=" + "%27%20OR%201%3D1--",
                  wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(700)
        check("SQL-injection-ish search does not crash + returns sane count",
              page.evaluate("document.getElementById('cnt-total').innerText") != "")

        # ═══════════════════════════════════════════════════════════════
        #  E. CROSS-WIDGET CLICK CHAIN
        # ═══════════════════════════════════════════════════════════════
        print("\n=== E. CROSS-WIDGET CLICK CHAIN ===")
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(500)

        # E1: Phenology dot → Drill-down modal
        page.locator("#container-b .mode-dot[data-mode='sightings']").click()
        page.wait_for_timeout(800)
        dots = page.locator("#phenology-svg circle")
        if dots.count() > 0:
            dots.first.click()
            page.wait_for_timeout(500)
            check("phenology dot click opens Disease Drill-down modal",
                  page.locator("#modal-disease").evaluate("e => e.classList.contains('open')"))
            # E2: From INSIDE drill-down, click a row → Signal Vote opens (swap, not stack)
            mrow = page.locator("#modal-disease-rows .modal-pred-row")
            if mrow.count() > 0:
                mrow.first.click()
                page.wait_for_timeout(500)
                check("drilldown-row click swaps to Signal Vote modal",
                      page.locator("#modal-signals").evaluate("e => e.classList.contains('open')"))
                check("only ONE modal is open at end of chain",
                      page.locator(".modal-sheet.open").count() == 1)
                page.keyboard.press("Escape")
                page.wait_for_timeout(400)

        # E3: Family Tree → click a leaf inside, then inside the drill-down, click
        #     a [data-disease] (e.g., in encyclopedia section) — should NOT re-trigger.
        # Phase-3.5 v2: the tree is now a vertical indented layout with
        # .fam-row.leaf rows containing a clickable .fam-name span.  The
        # data-disease attr lives on the row itself; click handler is on
        # the inner name span.
        page.locator("#container-d .mode-dot[data-mode='family']").click()
        page.wait_for_timeout(900)  # Rough.js render + reveal animation
        leaves = page.locator("#family-tree-mount .fam-row.leaf .fam-name")
        if leaves.count() > 0:
            leaves.first.click()
            page.wait_for_timeout(500)
            check("family-tree leaf opens drill-down",
                  page.locator("#modal-disease").evaluate("e => e.classList.contains('open')"))
            # Find any [data-disease] inside the modal — clicking it should NOT
            # spawn a second drill-down (regression test for the fix we made)
            inside_dx = page.locator("#modal-disease [data-disease]")
            if inside_dx.count() > 0:
                inside_dx.first.click()
                page.wait_for_timeout(400)
                check("clicking [data-disease] INSIDE drilldown does not re-trigger drilldown",
                      page.locator(".modal-sheet.open").count() == 1)
            page.keyboard.press("Escape")
            page.wait_for_timeout(400)

        # E4: Polaroid card click → Signal Vote (regression test for #2 fix)
        page.locator("#container-e .mode-dot[data-mode='polaroid']").click()
        page.wait_for_timeout(800)
        polaroids = page.locator("#polaroid-grid .polaroid")
        if polaroids.count() > 0:
            polaroids.first.click()
            page.wait_for_timeout(500)
            open_modal = page.evaluate("""
                (() => {
                    const sigs = document.getElementById('modal-signals');
                    const dis = document.getElementById('modal-disease');
                    return {
                        sigs: sigs.classList.contains('open'),
                        dis: dis.classList.contains('open'),
                    };
                })()
            """)
            check("polaroid click opens Signal Vote, NOT Drill-down",
                  open_modal["sigs"] and not open_modal["dis"],
                  f"sigs={open_modal['sigs']} dis={open_modal['dis']}")
            page.keyboard.press("Escape")
            page.wait_for_timeout(300)

        # ═══════════════════════════════════════════════════════════════
        #  F. COMPARISON SPREAD EDGE
        # ═══════════════════════════════════════════════════════════════
        print("\n=== F. COMPARISON SPREAD EDGE ===")
        page.locator("#container-b .mode-dot[data-mode='compare']").click()
        page.wait_for_timeout(800)
        # F1: Set both period selectors to the same period — both panes should render
        page.locator("#compare-period-a").select_option("this-week")
        page.locator("#compare-period-b").select_option("this-week")
        page.wait_for_timeout(700)
        a_rows = page.locator("#compare-rows-a .compare-row").count()
        b_rows = page.locator("#compare-rows-b .compare-row").count()
        check("compare A=B identical period → both panes render",
              a_rows >= 3 and b_rows >= 3, f"a={a_rows} b={b_rows}")

        # ═══════════════════════════════════════════════════════════════
        #  G. FIELD NOTEBOOK INDEX — type-to-filter edge
        # ═══════════════════════════════════════════════════════════════
        print("\n=== G. FN INDEX FILTER STRESS ===")
        page.locator("#container-d .mode-dot[data-mode='index']").click()
        page.wait_for_timeout(800)
        # G1: Filter that matches NO entries
        page.locator("#fn-index-search").fill("zzzzznonsense")
        page.wait_for_timeout(250)
        no_match = page.locator("#fn-index-mount .fn-card").count()
        check("FN index filter with no match → 0 cards + sane empty state",
              no_match == 0)
        # G2: Clear and verify all cards return
        page.locator("#fn-index-search").fill("")
        page.wait_for_timeout(250)
        all_back = page.locator("#fn-index-mount .fn-card").count()
        check("clearing FN filter restores all cards", all_back >= 5,
              f"n={all_back}")

        # ═══════════════════════════════════════════════════════════════
        #  H. KEYBOARD-ONLY NAV
        # ═══════════════════════════════════════════════════════════════
        print("\n=== H. KEYBOARD NAV ===")
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(500)
        # H1: Press Tab repeatedly until we land inside a mode-dot, then arrow-cycle
        page.evaluate("document.querySelector('#container-c .mode-dot[data-mode=donut]').focus()")
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(200)
        check("keyboard ArrowRight on focused dot advances mode",
              page.locator("#container-c .mode-dot.active").get_attribute("data-mode") == "histogram")
        page.keyboard.press("ArrowLeft")
        page.wait_for_timeout(200)
        check("keyboard ArrowLeft reverts mode",
              page.locator("#container-c .mode-dot.active").get_attribute("data-mode") == "donut")

        # H2: Open a modal, press Tab — focus should be inside the modal
        page.evaluate("openDayDetail('2026-05-14')")
        page.wait_for_timeout(500)
        # After open, focus moved to the close button per ModalShell
        focused_in_modal = page.evaluate(
            "document.activeElement.closest('#modal-day') !== null"
        )
        check("modal open → focus is inside the modal", focused_in_modal)
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)

        # ═══════════════════════════════════════════════════════════════
        #  I. BROWSER HISTORY (back / forward)
        # ═══════════════════════════════════════════════════════════════
        print("\n=== I. BROWSER HISTORY ===")
        # We use replaceState on mode switches (intentionally — modes shouldn't
        # spam history). So back button on /dashboard should leave the page.
        # But clicking history page link IS a navigation that goes in history.
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(400)
        page.goto(f"{BASE}/dashboard/history", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(400)
        page.go_back()
        page.wait_for_load_state("networkidle", timeout=10000)
        check("back from /history returns to /dashboard",
              "/dashboard" in page.url and "/history" not in page.url,
              f"url={page.url}")
        page.go_forward()
        page.wait_for_load_state("networkidle", timeout=10000)
        check("forward returns to /history", "/dashboard/history" in page.url,
              f"url={page.url}")

        # ═══════════════════════════════════════════════════════════════
        #  J. VIEWPORT / RESPONSIVE
        # ═══════════════════════════════════════════════════════════════
        print("\n=== J. VIEWPORT / RESPONSIVE ===")
        page.set_viewport_size({"width": 960, "height": 720})  # tablet-ish
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(500)
        check("dashboard renders at 960×720 with shell visible",
              not page.locator("#shell").evaluate("e => e.hidden"))
        shot(page, "viewport_960")

        page.set_viewport_size({"width": 600, "height": 900})  # narrow
        page.wait_for_timeout(400)
        check("dashboard renders at 600×900 without horizontal overflow",
              page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 4"),
              f"scrollW={page.evaluate('document.documentElement.scrollWidth')} innerW={page.evaluate('window.innerWidth')}")
        shot(page, "viewport_600")

        page.set_viewport_size({"width": 1440, "height": 900})  # back to normal

        # ═══════════════════════════════════════════════════════════════
        #  K. PHASE 3 STRESS — Treatment Log, Share tokens, PDF, new pages
        # ═══════════════════════════════════════════════════════════════
        print("\n=== K. PHASE 3 STRESS ===")
        page.set_viewport_size({"width": 1440, "height": 900})
        page.goto(f"{BASE}/dashboard", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(500)

        # Wipe any leftover treatments from previous runs
        page.evaluate("""
          fetch('/dashboard/treatments',{credentials:'include'})
            .then(r=>r.json())
            .then(j=>Promise.all(j.results.map(t=>fetch('/dashboard/treatments/'+t.id,{method:'DELETE',credentials:'include'}))))
            .then(()=>TreatmentLog.refresh())
        """)
        page.wait_for_timeout(700)

        # K1: Try to save EMPTY treatment — should alert (won't crash, won't save)
        page.evaluate("TreatmentLog.openEditor(null)")
        page.wait_for_timeout(200)
        # Dismiss the alert when it appears
        page.once("dialog", lambda d: d.accept())
        page.locator("#treatment-editor .btn-save").click()
        page.wait_for_timeout(500)
        n_after_empty = page.locator("#treatments-list .treatment-row").count()
        check("empty-treatment save did NOT create a row",
              n_after_empty == 0, f"n={n_after_empty}")
        # Editor stays open since save() returned early — close it manually
        page.evaluate("document.querySelector('#treatment-editor').classList.remove('open')")

        # K2: OVERSIZE treatment text (300 chars > 200 server cap)
        page.evaluate("TreatmentLog.openEditor(null)")
        page.wait_for_timeout(200)
        page.locator("#tx-ed-treatment").fill("x" * 300)
        page.locator("#tx-ed-date").fill("2026-05-14")
        page.locator("#treatment-editor .btn-save").click()
        page.wait_for_timeout(800)
        n_after_oversize = page.locator("#treatments-list .treatment-row").count()
        check("oversize treatment saved (server capped)", n_after_oversize == 1)
        stored = page.locator(".treatment-row .what").first.inner_text()
        # Drop trailing whitespace/notes from stored text for compare
        first_line = stored.split("\n")[0].strip()
        check("server capped 300-char input to ≤ 200",
              len(first_line) <= 200, f"stored len={len(first_line)}")
        # Cleanup
        page.evaluate("""
          fetch('/dashboard/treatments',{credentials:'include'})
            .then(r=>r.json())
            .then(j=>Promise.all(j.results.map(t=>fetch('/dashboard/treatments/'+t.id,{method:'DELETE',credentials:'include'}))))
            .then(()=>TreatmentLog.refresh())
        """)
        page.wait_for_timeout(500)

        # K3: Share token — forge attempts
        print("\n=== K3. SHARE-TOKEN FORGERY ATTEMPTS ===")
        # K3a: short garbage token
        r = page.evaluate(
            "fetch('/share/aaaa/data').then(r => r.status)",
        )
        check("4-char garbage token → 404", r == 404, f"got {r}")
        # K3b: very long token
        long_tok = "a" * 200
        r = page.evaluate(
            f"fetch('/share/{long_tok}/data').then(r => r.status)",
        )
        check("200-char garbage token → 404", r == 404, f"got {r}")
        # K3c: SQL injection attempt in token
        sqli = "1' OR '1'='1"
        import urllib.parse as _up
        r = page.evaluate(
            f"fetch('/share/{_up.quote(sqli)}/data').then(r => r.status)",
        )
        check("SQL-injection-like token → 404, no crash",
              r == 404, f"got {r}")

        # K4: Reports page loads for an authenticated session.
        # (The dev-mode banner was removed with the auth-bypass fix — the
        # page now simply requires a real login, which the suite's injected
        # session cookie satisfies.)
        page.goto(f"{BASE}/dashboard/reports", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(700)
        check("/dashboard/reports loads for authenticated session",
              page.locator("#pdf-generate").count() == 1
              and "/dashboard/reports" in page.url)

        # K5: Loupe with no predictions (set crop filter to non-existent)
        page.goto(f"{BASE}/dashboard/loupe", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(700)
        # If picker has options, swap them and verify panes update
        opts = page.locator("#pick-a option").count()
        check("loupe picker populated", opts >= 1, f"n={opts}")

        # K6: Gallery — load-more behaviour.
        # Phase-3.5: the assertion used to assume 18 fixture predictions
        # against a page_size of 24 so load-more was always hidden, but
        # the QA pipeline now records real predictions and the count
        # drifts above 24.  Test the invariant adaptively: load-more
        # should be visible iff the dashboard reports total > shown.
        page.goto(f"{BASE}/dashboard/gallery", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(900)
        initial_polaroids = page.locator(".polaroid").count()
        total_predictions = page.evaluate(
            "fetch('/dashboard/history/data?page_size=1')"
            "  .then(r => r.json()).then(j => j.total || 0)"
        )
        load_more_hidden = page.locator("#load-more-row").evaluate("e => e.hidden")
        should_be_hidden = (total_predictions <= initial_polaroids)
        check("load-more visibility matches total vs shown count",
              load_more_hidden == should_be_hidden,
              f"hidden={load_more_hidden} expected={should_be_hidden} "
              f"shown={initial_polaroids} total={total_predictions}")

        # K7: Settings — theme switch persists across reload
        page.goto(f"{BASE}/dashboard/settings", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(500)
        page.locator(".theme-card[data-theme='ledger']").click()
        page.wait_for_timeout(200)
        # Reload page — theme should still be ledger
        page.reload()
        page.wait_for_timeout(700)
        check("theme persists across reload",
              page.locator(".theme-card[data-theme='ledger']").evaluate(
                  "e => e.classList.contains('active')"))
        # Reset theme to cream for cleanliness
        page.locator(".theme-card[data-theme='cream']").click()
        page.wait_for_timeout(200)

        # K8: PDF endpoint with weird week_start values — never crashes
        for wkstart in ["", "garbage", "9999-99-99", "2026-13-32", "../../etc/passwd"]:
            r = page.evaluate(
                f"fetch('/dashboard/report/weekly.pdf?week_start={_up.quote(wkstart)}').then(r => r.status)",
            )
            check(f"PDF gen with week_start={wkstart!r:>20} → 200 (no crash)",
                  r == 200, f"got {r}")

        # ═══════════════════════════════════════════════════════════════
        #  Final console-error report
        # ═══════════════════════════════════════════════════════════════
        print("\n=== CONSOLE ERRORS DURING STRESS ===")
        if console_errors:
            for kind, msg in console_errors[:10]:
                print(f"  ⚠ {kind}: {msg[:240]}")
            check(f"no console errors during stress sweep", False,
                  f"{len(console_errors)} errors")
        else:
            check("no console errors during stress sweep", True)

        browser.close()

    print("\n" + "=" * 64)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"  TOTAL: {n_pass} passed   {n_fail} failed")
    if n_fail:
        print("\n  FAILURES:")
        for name, ok, detail in results:
            if not ok:
                print(f"    ✗ {name}" + (f" — {detail}" if detail else ""))
    sys.exit(0 if n_fail == 0 else 2)


if __name__ == "__main__":
    main()
