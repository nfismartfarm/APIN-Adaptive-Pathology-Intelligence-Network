"""Render the README benchmark charts in the APIN field-notebook style.

Outputs (into assets/):
  bench_ensemble.png   APIN ensemble vs the best single model, both crops
  bench_perclass.png   per-class F1 across the 9 okra / brassica classes
  bench_reliability.png  calibration + ranking quality at a glance

All numbers are the real measured results:
  okra / brassica  -> scripts/apin/results/apin_comprehensive_metrics.json
  tomato           -> a4_field_val_three_way_report.md (field_val, n=203)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

# ── Field-notebook palette ────────────────────────────────────────────────
PAPER   = "#f4efe2"   # cream paper
INK     = "#1b4332"   # deep green ink (text)
PRIMARY = "#2d6a4f"   # APIN green
LEAF    = "#5a9367"   # mid green
TAN     = "#c8b88a"   # muted tan (baselines)
AMBER   = "#c9803b"   # warm accent
GRID    = "#d8cfb8"   # soft grid line

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#a89f86",
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.8,
})


def _frame(ax):
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x" if ax.get_xlim()[1] <= 1.01 else "y",
            color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


# ── 1. APIN ensemble vs best single model ────────────────────────────────
def chart_ensemble():
    crops   = ["Okra & Brassica\n(9 classes, n=1350)", "Tomato\n(6 classes, n=203)"]
    single  = [0.9428, 0.9144]      # best single model
    ensemble= [0.9445, 0.9384]      # APIN ensemble
    x = range(len(crops)); w = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    fig.patch.set_facecolor(PAPER)
    b1 = ax.bar([i - w/2 for i in x], single,   w, color=TAN,
                edgecolor=INK, linewidth=0.8, label="Best single model")
    b2 = ax.bar([i + w/2 for i in x], ensemble, w, color=PRIMARY,
                edgecolor=INK, linewidth=0.8, label="APIN ensemble")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.004,
                    f"{r.get_height():.3f}", ha="center", va="bottom",
                    fontsize=10.5, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(crops, fontsize=10.5)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("macro F1  (higher is better)", fontsize=10.5)
    ax.set_title("Ensembling beats the strongest single model on both crops",
                 fontsize=13, color=INK, pad=12, weight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper left",
              bbox_to_anchor=(0.01, 0.99), ncol=2)
    _frame(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "bench_ensemble.png"), dpi=150,
                facecolor=PAPER)
    plt.close(fig)


# ── 2. Per-class F1 across the 9 okra / brassica classes ─────────────────
def chart_perclass():
    classes = ["Okra YVMV", "Okra Powdery Mildew", "Okra Cercospora",
               "Okra Enation", "Okra Healthy", "Brassica Black Rot",
               "Brassica Downy Mildew", "Brassica Alternaria",
               "Brassica Healthy"]
    f1 = [0.9669, 0.9143, 0.9307, 0.9176, 0.9764,
          0.9605, 0.9293, 0.9333, 0.9712]
    colors = [LEAF]*5 + [PRIMARY]*4
    macro = 0.9445
    order = sorted(range(len(f1)), key=lambda i: f1[i])
    classes = [classes[i] for i in order]
    f1      = [f1[i] for i in order]
    colors  = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.patch.set_facecolor(PAPER)
    y = range(len(classes))
    ax.barh(list(y), f1, color=colors, edgecolor=INK, linewidth=0.8, height=0.66)
    for i, v in enumerate(f1):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=10, color=INK)
    ax.axvline(macro, color=AMBER, linewidth=1.6, linestyle=(0, (4, 2)))
    ax.text(macro, len(classes) - 0.3, f"  macro F1 {macro:.3f}",
            color=AMBER, fontsize=9.5, va="center")
    ax.set_yticks(list(y)); ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlim(0.80, 1.02)
    ax.set_xlabel("F1  (validation set, n=1350)", fontsize=10.5)
    ax.set_title("APIN holds 0.91+ F1 on every okra and brassica class",
                 fontsize=13, color=INK, pad=12, weight="bold")
    _frame(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "bench_perclass.png"), dpi=150,
                facecolor=PAPER)
    plt.close(fig)


# ── 3. Reliability / ranking quality scoreboard ──────────────────────────
def chart_reliability():
    labels = ["Accuracy", "Weighted F1", "Top-3 accuracy",
              "Mean AUROC", "Calibration\n(1 - ECE)"]
    vals   = [0.9593, 0.9592, 0.9770, 0.9916, 1 - 0.0766]

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    fig.patch.set_facecolor(PAPER)
    y = range(len(labels))
    ax.barh(list(y), vals, color=PRIMARY, edgecolor=INK, linewidth=0.8,
            height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=10.5, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.08)
    ax.set_xlabel("score  (okra & brassica, validation, n=1350)", fontsize=10.5)
    ax.set_title("APIN scoreboard: accurate, well-ranked, well-calibrated",
                 fontsize=13, color=INK, pad=12, weight="bold")
    _frame(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "bench_reliability.png"), dpi=150,
                facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    chart_ensemble()
    chart_perclass()
    chart_reliability()
    print("README charts written to assets/:")
    for f in sorted(os.listdir(ASSETS)):
        print(f"  assets/{f}  ({os.path.getsize(os.path.join(ASSETS, f))//1024} KB)")
