"""PlantDoc per-class threshold optimisation. Print-only — no file writes."""
import numpy as np, torch, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DEVICE, ROOT, BEST_MODEL, CLASS_NAMES
from app.model import load_model_for_inference
from app.inference import preprocess_for_inference
from PIL import Image
from sklearn.metrics import f1_score

print("Loading Swin-Tiny model...")
model = load_model_for_inference(os.path.join(ROOT, BEST_MODEL), DEVICE)
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
print()

df = pd.read_csv("data/metadata/source_map.csv")
pd_eval = df[df["source_dataset"].str.contains("plantdoc_eval", case=False, na=False)]
present_classes = sorted(pd_eval["class_name"].unique())
present_indices = [CLASS_NAMES.index(c) for c in present_classes]

print(f"Running inference on {len(pd_eval)} PlantDoc images...")
print()

all_probs, all_labels = [], []
failed = 0

for i, (_, row) in enumerate(pd_eval.iterrows()):
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(pd_eval)} images processed...")

    img_path = row["image_path"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(ROOT, img_path)

    try:
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        tensor = preprocess_for_inference(img).to(DEVICE)
        with torch.no_grad():
            _, d, _ = model(tensor)
        probs = torch.sigmoid(d).cpu().numpy()[0]
        label = np.zeros(len(CLASS_NAMES))
        label[CLASS_NAMES.index(row["class_name"])] = 1
        all_probs.append(probs)
        all_labels.append(label)
    except Exception as e:
        failed += 1

print(f"  Done: {len(all_probs)} succeeded, {failed} failed")
print()

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# ============================================================
# SECTION 1: Global threshold sweep
# ============================================================
print("=" * 65)
print("SECTION 1: GLOBAL THRESHOLD SWEEP (all 7 classes same threshold)")
print("=" * 65)
header = f"{'Threshold':<12} {'Macro F1':<12}"
for cls in present_classes:
    short = cls.replace("tomato_", "t_")[:7]
    header += f" {short:>7}"
print(header)
print("-" * 80)

for thresh in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]:
    preds = (all_probs > thresh).astype(int)
    per_class = f1_score(
        all_labels[:, present_indices],
        preds[:, present_indices],
        average=None, zero_division=0
    )
    macro = per_class.mean()
    marker = " <-- CURRENT" if thresh == 0.50 else ""
    line = f"{thresh:<12.2f} {macro:<12.4f}"
    for v in per_class:
        line += f" {v:>7.3f}"
    print(line + marker)

# ============================================================
# SECTION 2: Per-class optimal threshold search
# ============================================================
print()
print("=" * 65)
print("SECTION 2: PER-CLASS OPTIMAL THRESHOLD SEARCH")
print("=" * 65)
print()

best_thresholds = {}
best_f1s = {}

thresholds_to_try = np.arange(0.05, 0.65, 0.025)

for i, cls_name in enumerate(present_classes):
    cls_idx = present_indices[i]
    cls_labels = all_labels[:, cls_idx].astype(int)
    support = int(cls_labels.sum())

    results = []
    for t in thresholds_to_try:
        preds_cls = (all_probs[:, cls_idx] > t).astype(int)
        tp = int((preds_cls * cls_labels).sum())
        fp = int((preds_cls * (1 - cls_labels)).sum())
        fn = int(((1 - preds_cls) * cls_labels).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        results.append((t, f1, precision, recall, tp, fp, fn))

    best = max(results, key=lambda x: x[1])
    best_t, best_f1, best_p, best_r, tp, fp, fn = best
    current_result = next(r for r in results if abs(r[0] - 0.5) < 0.02)

    best_thresholds[cls_name] = float(best_t)
    best_f1s[cls_name] = float(best_f1)

    print(f"Class: {cls_name}")
    print(f"  Support (PlantDoc eval):  {support} images")
    print(f"  At threshold 0.50:        F1={current_result[1]:.3f}  P={current_result[2]:.3f}  R={current_result[3]:.3f}")
    print(f"  Optimal threshold:        {best_t:.3f}")
    print(f"  At optimal threshold:     F1={best_f1:.3f}  P={best_p:.3f}  R={best_r:.3f}")
    print(f"  F1 gain from tuning:      +{best_f1 - current_result[1]:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print()

# ============================================================
# SECTION 3: Summary comparison
# ============================================================
print("=" * 65)
print("SECTION 3: SUMMARY -- BEFORE vs AFTER THRESHOLD TUNING")
print("=" * 65)
print()

preds_before = (all_probs > 0.5).astype(int)
f1_before_per_class = f1_score(
    all_labels[:, present_indices],
    preds_before[:, present_indices],
    average=None, zero_division=0
)
macro_before = f1_before_per_class.mean()

preds_after = np.zeros_like(all_probs)
for i, cls_name in enumerate(present_classes):
    cls_idx = present_indices[i]
    preds_after[:, cls_idx] = (all_probs[:, cls_idx] > best_thresholds[cls_name]).astype(float)

f1_after_per_class = f1_score(
    all_labels[:, present_indices],
    preds_after[:, present_indices].astype(int),
    average=None, zero_division=0
)
macro_after = f1_after_per_class.mean()

print(f"{'Class':<42} {'Before':>8} {'After':>8} {'Gain':>8} {'Threshold':>10}")
print("-" * 80)
for i, cls_name in enumerate(present_classes):
    before = f1_before_per_class[i]
    after = f1_after_per_class[i]
    gain = after - before
    t = best_thresholds[cls_name]
    gain_str = f"+{gain:.3f}" if gain >= 0 else f"{gain:.3f}"
    print(f"{cls_name:<42} {before:>8.3f} {after:>8.3f} {gain_str:>8} {t:>10.3f}")

print("-" * 80)
gain_macro = macro_after - macro_before
print(f"{'MACRO F1 (7 PlantDoc classes)':<42} {macro_before:>8.4f} {macro_after:>8.4f} {gain_macro:>+8.4f}")
print()
print(f"PlantDoc macro F1 improvement: {macro_before:.4f} -> {macro_after:.4f} (+{macro_after-macro_before:.4f})")
print()

# ============================================================
# SECTION 4: Proposed config.py changes
# ============================================================
print("=" * 65)
print("SECTION 4: PROPOSED DISEASE_THRESHOLDS CHANGES")
print("=" * 65)
print()
print("The following changes would update DISEASE_THRESHOLDS in app/config.py.")
print("Only the 7 PlantDoc tomato classes would be updated.")
print("All other classes remain at 0.50.")
print()
print("Proposed new threshold values:")
for cls_name in present_classes:
    t = best_thresholds[cls_name]
    idx = present_classes.index(cls_name)
    gain = best_f1s[cls_name] - f1_before_per_class[idx]
    print(f"  {cls_name}: {t:.3f}  (was 0.500, F1 gain: +{gain:.3f})")

print()
print("DO NOT UPDATE CONFIG YET.")
print("Report these results and wait for confirmation before making any changes.")

# ============================================================
# SECTION 5: Confidence distribution analysis
# ============================================================
print()
print("=" * 65)
print("SECTION 5: CONFIDENCE DISTRIBUTION ON PLANTDOC IMAGES")
print("(Why thresholds need to be lower for wild-condition images)")
print("=" * 65)
print()
print("For each class: model confidence on TRUE POSITIVE PlantDoc images")
print()

for i, cls_name in enumerate(present_classes):
    cls_idx = present_indices[i]
    cls_labels = all_labels[:, cls_idx].astype(bool)
    if cls_labels.sum() == 0:
        continue
    true_pos_probs = all_probs[cls_labels, cls_idx]
    mean_conf = true_pos_probs.mean()
    median_conf = np.median(true_pos_probs)
    pct_above_05 = (true_pos_probs > 0.5).mean() * 100
    pct_above_03 = (true_pos_probs > 0.3).mean() * 100
    pct_above_opt = (true_pos_probs > best_thresholds[cls_name]).mean() * 100
    print(f"{cls_name}:")
    print(f"  Mean confidence on true images:    {mean_conf:.3f}")
    print(f"  Median confidence on true images:  {median_conf:.3f}")
    print(f"  Fraction above 0.50 threshold:     {pct_above_05:.1f}%  (detected at default)")
    print(f"  Fraction above 0.30 threshold:     {pct_above_03:.1f}%  (detected at 0.30)")
    print(f"  Fraction above optimal {best_thresholds[cls_name]:.3f}:    {pct_above_opt:.1f}%  (detected at optimal)")
    print()
