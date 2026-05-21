"""Phase D test: render the weekly PDF across data volumes.

Generates reports for many / medium / few / zero specimens, checks each is
a valid PDF, and writes the 'many' sample to deploy/_sample_report.pdf for
visual inspection. Run: python scripts/apin_v2/_report_pdf_test.py
"""
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.apin_v2.report_pdf import generate_weekly_pdf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIAG = {}
try:
    with open(os.path.join(ROOT, "diagnosis", "diagnosis_lookup.json"),
              encoding="utf-8") as f:
        DIAG = json.load(f)
except Exception as e:
    print("diagnosis lookup not loaded:", e)

DISEASES = ["okra_yvmv", "okra_powdery_mildew", "okra_cercospora",
            "okra_enation", "okra_healthy", "brassica_black_rot",
            "brassica_downy_mildew", "brassica_alternaria", "brassica_healthy"]
TIERS = ["1A", "2", "3A", "4A", "5"]
SEVS = ["mild", "moderate", "severe"]


def _png(seed):
    """A tiny valid PNG so the gallery/image path is exercised."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 2))
    rng = random.Random(seed)
    ax.imshow([[rng.random() for _ in range(8)] for _ in range(8)],
              cmap="YlGn")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def mkpreds(n, seed=1):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        cls = rng.choice(DISEASES)
        out.append({
            "id": 1000 + i,
            "crop": "okra" if cls.startswith("okra") else "brassica",
            "predicted_class": cls,
            "confidence": round(rng.uniform(0.45, 0.999), 3),
            "tier": rng.choice(TIERS),
            "created_at": f"2026-05-{18 + i % 7:02d}T{8 + i % 11:02d}:30:00+00:00",
            "response_json": json.dumps({"severity": rng.choice(SEVS)}),
            "has_image": True, "has_heatmap": True,
        })
    return out


checks = []


def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
          (f" — {detail}" if detail else ""))


user = {"display_name": "Devkrishna S", "username": "dev"}
img_cache = {i: _png(i) for i in range(1100, 1106)}


def fetch_image(pid):
    return img_cache.get(pid) or _png(pid)


def fetch_heatmap(pid):
    return _png(pid + 7)


treatments = [
    {"applied_date": "2026-05-20", "treatment": "Copper spray",
     "disease": "brassica_black_rot"},
    {"applied_date": "2026-05-18", "treatment": "Neem oil",
     "disease": "okra_yvmv"},
]

print("\n=== weekly PDF across data volumes ===")
for label, n in [("MANY", 42), ("MEDIUM", 13), ("FEW", 3), ("ZERO", 0)]:
    preds = mkpreds(n, seed=n + 1)
    prev = mkpreds(max(0, n // 2), seed=n + 99)
    try:
        pdf, summary = generate_weekly_pdf(
            user=user, predictions=preds, prev_predictions=prev,
            week_start="2026-05-18", week_end="2026-05-24",
            fetch_image=fetch_image, fetch_heatmap=fetch_heatmap,
            treatments=treatments, diagnosis_lookup=DIAG)
        ok = isinstance(pdf, bytes) and pdf[:5] == b"%PDF-"
        pages = pdf.count(b"/Type /Page")
        check(f"{label:7s} (n={n}) renders a valid PDF", ok,
              f"{len(pdf) // 1024} KB, ~{pages} pages, summary={summary}")
        if label == "MANY":
            outp = os.path.join(ROOT, "deploy", "_sample_report.pdf")
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            with open(outp, "wb") as f:
                f.write(pdf)
            print(f"          sample written to {outp}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(f"{label} renders without error", False, str(e))

print("\n=== edge cases ===")
try:
    # all healthy
    preds = [dict(p, predicted_class="okra_healthy") for p in mkpreds(10)]
    pdf, _ = generate_weekly_pdf(
        user=user, predictions=preds, prev_predictions=[],
        week_start="2026-05-18", week_end="2026-05-24",
        diagnosis_lookup=DIAG)
    check("all-healthy week renders", pdf[:5] == b"%PDF-")
except Exception as e:
    check("all-healthy week renders", False, str(e))
try:
    # no images, no prev, no treatments, no lookup
    pdf, _ = generate_weekly_pdf(
        user={}, predictions=mkpreds(8), prev_predictions=[],
        week_start="2026-05-18", week_end="2026-05-24")
    check("minimal inputs (no images/lookup/treatments) render",
          pdf[:5] == b"%PDF-")
except Exception as e:
    check("minimal inputs render", False, str(e))
try:
    # one specimen
    pdf, _ = generate_weekly_pdf(
        user=user, predictions=mkpreds(1), prev_predictions=[],
        week_start="2026-05-18", week_end="2026-05-24",
        fetch_image=fetch_image, fetch_heatmap=fetch_heatmap,
        diagnosis_lookup=DIAG)
    check("single-specimen week renders", pdf[:5] == b"%PDF-")
except Exception as e:
    check("single-specimen week renders", False, str(e))

print("\n" + "=" * 56)
npass = sum(1 for ok in checks if ok)
nfail = sum(1 for ok in checks if not ok)
print(f"  PHASE D PDF TEST: {npass} passed   {nfail} failed")
sys.exit(0 if nfail == 0 else 1)
