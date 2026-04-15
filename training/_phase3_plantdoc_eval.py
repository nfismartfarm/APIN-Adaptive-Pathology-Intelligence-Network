"""Phase 3 PlantDoc evaluation with second pass + ensemble."""
import numpy as np, torch, pandas as pd, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (DEVICE, ROOT, BEST_MODEL, CLASS_NAMES, DISEASE_THRESHOLDS)
from app.model import load_model_for_inference
from app.inference import run_inference
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

print("=== PHASE 3 PLANTDOC EVALUATION ===")
print()

model = load_model_for_inference(os.path.join(ROOT, BEST_MODEL), DEVICE)
model.eval()

df = pd.read_csv("data/metadata/source_map.csv")
pd_eval = df[df["source_dataset"].str.contains("plantdoc_eval", case=False, na=False)]
present_classes = sorted(pd_eval["class_name"].unique())
present_indices = [CLASS_NAMES.index(c) for c in present_classes]

print(f"Evaluating {len(pd_eval)} PlantDoc images...")
print()

all_probs = []
all_labels = []
all_uncertainties = []
all_ood_flags = []
second_pass_count = 0
ensemble_count = 0
failed = 0
total_time = 0

for i, (_, row) in enumerate(pd_eval.iterrows()):
    if i % 50 == 0:
        avg_t = total_time / max(i, 1)
        remain = (len(pd_eval) - i) * avg_t
        print(f"  Progress: {i}/{len(pd_eval)} | avg {avg_t:.1f}s/img | ~{remain/60:.0f}min remaining",
              flush=True)

    img_path = row["image_path"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(ROOT, img_path)

    try:
        img_pil = Image.open(img_path).convert("RGB")
        # Resize large images to prevent SAM OOM
        if max(img_pil.size) > 800:
            ratio = 800 / max(img_pil.size)
            img_pil = img_pil.resize((int(img_pil.size[0]*ratio),
                                       int(img_pil.size[1]*ratio)))
        img = np.array(img_pil)

        t0 = time.time()
        result = run_inference(model, img, return_raw_probs=True)
        t1 = time.time()
        total_time += (t1 - t0)

        # Collect raw probs for evaluation
        if "raw_disease_probs" in result:
            probs = result["raw_disease_probs"]
        else:
            probs = np.zeros(len(CLASS_NAMES))

        label = np.zeros(len(CLASS_NAMES))
        label[CLASS_NAMES.index(row["class_name"])] = 1

        all_probs.append(probs)
        all_labels.append(label)
        all_uncertainties.append(result.get("uncertainty", 0))
        all_ood_flags.append(result.get("ood_flagged", False))
        if result.get("second_pass_applied", False):
            second_pass_count += 1
        if result.get("ensemble_applied", False):
            ensemble_count += 1

        # Clear CUDA cache periodically
        if i % 100 == 0:
            torch.cuda.empty_cache()

    except Exception as e:
        failed += 1

print(f"  Complete: {len(all_probs)} succeeded, {failed} failed")
print(f"  Total time: {total_time:.0f}s ({total_time/max(len(all_probs),1):.1f}s avg)")
print(f"  Second pass triggered: {second_pass_count} of {len(all_probs)}")
print(f"  Ensemble triggered: {ensemble_count} of {len(all_probs)}")
print()

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Apply per-class thresholds
thresholds = np.array([DISEASE_THRESHOLDS.get(cls, 0.5) for cls in CLASS_NAMES])
binary_preds = (all_probs > thresholds).astype(int)

labels_7 = all_labels[:, present_indices]
preds_7 = binary_preds[:, present_indices]

macro_f1 = f1_score(labels_7, preds_7, average="macro", zero_division=0)
weighted_f1 = f1_score(labels_7, preds_7, average="weighted", zero_division=0)
per_class_f1 = f1_score(labels_7, preds_7, average=None, zero_division=0)
per_class_p = precision_score(labels_7, preds_7, average=None, zero_division=0)
per_class_r = recall_score(labels_7, preds_7, average=None, zero_division=0)

print("=" * 70)
print("PHASE 3 PLANTDOC RESULTS -- FULL BREAKDOWN")
print("=" * 70)
print()
print(f"PlantDoc macro F1 (7 classes):    {macro_f1:.4f}")
print(f"PlantDoc weighted F1:              {weighted_f1:.4f}")
print()

print("Per-class breakdown:")
print(f"{'Class':<42} {'F1':>6} {'P':>6} {'R':>6} {'Support':>8}")
print("-" * 70)
for i, cls_name in enumerate(present_classes):
    support = int(all_labels[:, present_indices[i]].sum())
    print(f"{cls_name:<42} {per_class_f1[i]:>6.3f} {per_class_p[i]:>6.3f} {per_class_r[i]:>6.3f} {support:>8}")
print("-" * 70)
print(f"{'MACRO':<42} {macro_f1:>6.4f}")
print()

print("=" * 70)
print("COMPARISON: BEFORE PHASE 3 vs AFTER PHASE 3")
print("=" * 70)
before_values = {
    "tomato_bacterial_spot": 0.311,
    "tomato_early_blight": 0.505,
    "tomato_late_blight": 0.779,
    "tomato_leaf_mold": 0.522,
    "tomato_mosaic_virus": 0.425,
    "tomato_septoria_leaf_spot": 0.675,
    "tomato_yellow_leaf_curl_virus": 0.724,
}
before_macro = 0.5630

print(f"{'Class':<42} {'Before':>8} {'After':>8} {'Change':>8}")
print("-" * 70)
for i, cls_name in enumerate(present_classes):
    before = before_values.get(cls_name, 0.0)
    after = per_class_f1[i]
    change = after - before
    cs = f"+{change:.3f}" if change >= 0 else f"{change:.3f}"
    print(f"{cls_name:<42} {before:>8.3f} {after:>8.3f} {cs:>8}")
print("-" * 70)
cm = macro_f1 - before_macro
print(f"{'MACRO F1':<42} {before_macro:>8.4f} {macro_f1:>8.4f} {cm:>+8.4f}")
print()

print("=" * 70)
print("INFERENCE STATISTICS")
print("=" * 70)
ood_count = sum(all_ood_flags)
high_unc = sum(1 for u in all_uncertainties if u > 0.30)
mean_unc = np.mean(all_uncertainties)
print(f"OOD flagged:                       {ood_count}/{len(all_probs)} ({100*ood_count/max(len(all_probs),1):.1f}%)")
print(f"High uncertainty (>0.30):          {high_unc}/{len(all_probs)} ({100*high_unc/max(len(all_probs),1):.1f}%)")
print(f"Mean uncertainty:                  {mean_unc:.3f}")
print(f"Second pass triggered:             {second_pass_count}/{len(all_probs)}")
print(f"Ensemble triggered:                {ensemble_count}/{len(all_probs)}")
print(f"Average inference time:            {total_time/max(len(all_probs),1):.1f}s/image")
print()

print("=" * 70)
print("FINAL PHASE 3 SUMMARY")
print("=" * 70)
print(f"Controlled test F1 (from Phase 2): 0.9410")
print(f"PlantDoc F1 before Phase 3:        0.5630")
print(f"PlantDoc F1 after Phase 3:         {macro_f1:.4f}")
print(f"Improvement from Phase 3:          {macro_f1-before_macro:+.4f}")
print()
if macro_f1 >= 0.62:
    print("TARGET REACHED: PlantDoc F1 >= 0.62")
elif macro_f1 >= 0.55:
    print(f"BELOW TARGET: {macro_f1:.4f} vs 0.62 target (gap: {0.62-macro_f1:.4f})")
else:
    print(f"SIGNIFICANTLY BELOW TARGET: {macro_f1:.4f} vs 0.62 target")
