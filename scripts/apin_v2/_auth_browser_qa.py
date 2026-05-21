"""Live-browser QA for the authentication gate (Playwright/Chromium).

Exercises the real visitor flows against a running server on :8766 —
anonymous upload gate, the guest path with its quota, signup, login,
session persistence, and the dashboard login wall.
"""
import io
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8766"
checks = []


def check(name, ok, detail=""):
    checks.append((name, ok))
    mark = "\x1b[32mPASS\x1b[0m" if ok else "\x1b[31mFAIL\x1b[0m"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def _leaf_jpeg():
    """A textured green JPEG so the validator's blur check passes."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(7)
    h = w = 360
    base = np.zeros((h, w, 3), dtype=np.uint8)
    y, x = np.indices((h, w))
    base[:, :, 1] = 110 + ((x // 4 + y // 5) % 70).astype(np.uint8)
    noise = rng.integers(-28, 28, (h, w, 3), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def main():
    jpeg = _leaf_jpeg()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # ── 1. ANONYMOUS visitor lands on the inference page ──────────────
        ctx = browser.new_context(viewport={"width": 1380, "height": 900})
        page = ctx.new_page()
        page.goto(BASE + "/", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1500)

        check("inference page is the landing page",
              "Plant Disease Detection" in page.title()
              or page.locator("#upload-area").count() == 1)
        check("account area shows 'Sign in' for anonymous visitor",
              "sign in" in (page.locator("#acct-area").inner_text() or "").lower())
        check("auth modal is hidden on first load",
              page.locator("#auth-backdrop").get_attribute("hidden") is not None)

        # ── 2. Anonymous upload → modal appears (blocking) ────────────────
        page.set_input_files("#file-input", {"name": "leaf.jpg",
                             "mimeType": "image/jpeg", "buffer": jpeg})
        page.wait_for_timeout(900)
        modal_open = page.locator("#auth-backdrop").get_attribute("hidden") is None
        check("upload by anonymous visitor opens the auth modal", modal_open)
        check("modal has Sign In + Create Account tabs",
              page.locator(".auth-tab").count() == 2)
        check("modal has 'Continue as guest' option",
              page.locator("#auth-guest-btn").is_visible())
        # Blocking: backdrop click must NOT dismiss it
        page.mouse.click(20, 20)
        page.wait_for_timeout(300)
        check("upload-gate modal is blocking (backdrop click does not close)",
              page.locator("#auth-backdrop").get_attribute("hidden") is None)

        # ── 3. Continue as guest → modal closes, inference proceeds ───────
        page.locator("#auth-guest-btn").click()
        page.wait_for_timeout(2500)
        check("guest path closes the modal",
              page.locator("#auth-backdrop").get_attribute("hidden") is not None)
        acct = (page.locator("#acct-area").inner_text() or "").lower()
        check("account area shows guest counter after guest start",
              "guest" in acct and "left" in acct)
        # the held file should have been resubmitted — pending or result shows
        page.wait_for_timeout(8000)
        ran = (page.locator("#result").is_visible()
               or page.locator("#pending").is_visible()
               or page.locator("#error-box").is_visible())
        check("held specimen is submitted after guest auth", ran)
        ctx.close()

        # ── 4. SIGNUP a fresh account ─────────────────────────────────────
        ctx = browser.new_context(viewport={"width": 1380, "height": 900})
        page = ctx.new_page()
        page.goto(BASE + "/", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1200)
        page.set_input_files("#file-input", {"name": "leaf.jpg",
                             "mimeType": "image/jpeg", "buffer": jpeg})
        page.wait_for_timeout(800)
        page.locator('.auth-tab[data-tab="signup"]').click()
        page.wait_for_timeout(300)
        check("Create Account tab reveals the signup form",
              page.locator("#auth-form-signup").is_visible())
        # Username AND display name must both be unique (the users table
        # enforces a unique index on each) — derive both from the timestamp.
        sfx = str(int(time.time()))[-6:]
        uname = "browserqa" + sfx
        page.fill("#signup-username", uname)
        page.fill("#signup-display", "Browser QA " + sfx)
        page.fill("#signup-email", uname + "@example.com")
        page.fill("#signup-mobile", "9876501234")
        page.fill("#signup-password", "BrowserQa!2026")
        page.wait_for_timeout(300)
        met = page.locator("#signup-pwrules li.met").count()
        check("password rule checklist lights up (all 6 met)", met == 6,
              f"met={met}/6")
        page.locator("#signup-submit").click()
        # signup = argon2id hash (deliberately slow) + /auth/state refresh +
        # 240ms close transition — give it generous headroom.
        page.wait_for_timeout(5000)
        check("signup closes the modal",
              page.locator("#auth-backdrop").get_attribute("hidden") is not None)
        check("account chip shows the new user's name",
              "browser qa" in (page.locator("#acct-area").inner_text() or "").lower())

        # ── 5. Session persistence — reload, still signed in ─────────────
        page.goto(BASE + "/", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1500)
        check("returning user is remembered after reload (no modal)",
              page.locator("#auth-backdrop").get_attribute("hidden") is not None
              and "browser qa" in (page.locator("#acct-area").inner_text() or "").lower())

        # ── 6. Signed-in user can open the dashboard ─────────────────────
        page.goto(BASE + "/dashboard", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1500)
        check("signed-in user reaches /dashboard (not redirected)",
              "/dashboard" in page.url)
        ctx.close()

        # ── 7. Dashboard wall — anonymous /dashboard redirects to / ──────
        ctx = browser.new_context(viewport={"width": 1380, "height": 900})
        page = ctx.new_page()
        page.goto(BASE + "/dashboard", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(800)
        check("anonymous /dashboard redirects to the inference page",
              page.url.rstrip("/") == BASE.rstrip("/"))
        ctx.close()

        browser.close()

    npass = sum(1 for _, ok in checks if ok)
    nfail = sum(1 for _, ok in checks if not ok)
    print(f"\n{'='*56}")
    print(f"  AUTH BROWSER QA: {npass} passed, {nfail} failed")
    sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
