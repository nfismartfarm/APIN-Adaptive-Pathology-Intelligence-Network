"""Phase 3 Option 1: PlantDoc eval at strict coverage threshold."""
import numpy as np, torch, pandas as pd, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DEVICE, ROOT, BEST_MODEL, CLASS_NAMES, DISEASE_THRESHOLDS
from app.model import load_model_for_inference
from app.inference import run_inference
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

THRESHOLD_BEING_TESTED = 0.10

print("=" * 70, flush=True)
print(f"PLANTDOC EVALUATION -- COVERAGE THRESHOLD {THRESHOLD_BEING_TESTED}", flush=True)
print("=" * 70, flush=True)
print(flush=True)

model = load_model_for_inference(os.path.join(ROOT, BEST_MODEL), DEVICE)
model.eval()

df = pd.read_csv("data/metadata/source_map.csv")
pd_eval = df[df["source_dataset"].str.contains("plantdoc_eval", case=False, na=False)]
present_classes = sorted(pd_eval["class_name"].unique())
present_indices = [CLASS_NAMES.index(c) for c in present_classes]

print(f"Evaluating {len(pd_eval)} PlantDoc images...", flush=True)
print(f"Coverage threshold: {THRESHOLD_BEING_TESTED}", flush=True)
print(flush=True)

all_probs, all_labels = [], []
second_pass_count = 0
failed = 0
total_time = 0

for i, (_, row) in enumerate(pd_eval.iterrows()):
    if i > 0 and i % 50 == 0:
        avg = total_time / i
        remaining = (len(pd_eval) - i) * avg
        sp_pct = 100 * second_pass_count / i
        print(f"  [{i}/{len(pd_eval)}] avg={avg:.1f}s "
              f"2ndpass={sp_pct:.0f}% ~{remaining/60:.0f}min left", flush=True)

    img_path = row["image_path"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(ROOT, img_path)

    try:
        img_pil = Image.open(img_path).convert("RGB")
        if max(img_pil.size) > 800:
            ratio = 800 / max(img_pil.size)
            img_pil = img_pil.resize((int(img_pil.size[0]*ratio),
                                       int(img_pil.size[1]*ratio)))
        img = np.array(img_pil)

        t0 = time.time()
        result = run_inference(model, img, return_raw_probs=True)
        total_time += time.time() - t0

        if result.get("second_pass_applied", False):
            second_pass_count += 1

        probs = result.get("raw_disease_probs", np.zeros(len(CLASS_NAMES)))
        label = np.zeros(len(CLASS_NAMES))
        label[CLASS_NAMES.index(row["class_name"])] = 1
        all_probs.append(probs)
        all_labels.append(label)

        if i % 100 == 0:
            torch.cuda.empty_cache()
    except Exception:
        failed += 1

print(f"  Done: {len(all_probs)} succeeded, {failed} failed", flush=True)
print(f"  Second pass triggered: {second_pass_count}/{len(all_probs)} "
      f"({100*second_pass_count/max(len(all_probs),1):.1f}%)", flush=True)
print(f"  Avg inference time: {total_time/max(len(all_probs),1):.1f}s", flush=True)
print(flush=True)

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

thresholds = np.array([DISEASE_THRESHOLDS.get(c, 0.5) for c in CLASS_NAMES])
binary_preds = (all_probs > thresholds).astype(int)

labels_7 = all_labels[:, present_indices]
preds_7 = binary_preds[:, present_indices]

macro_f1 = f1_score(labels_7, preds_7, average="macro", zero_division=0)
per_class_f1 = f1_score(labels_7, preds_7, average=None, zero_division=0)
per_class_p = precision_score(labels_7, preds_7, average=None, zero_division=0)
per_class_r = recall_score(labels_7, preds_7, average=None, zero_division=0)

PHASE2_BASELINE = {
    "tomato_bacterial_spot": 0.311, "tomato_early_blight": 0.505,
    "tomato_late_blight": 0.779, "tomato_leaf_mold": 0.522,
    "tomato_mosaic_virus": 0.425, "tomato_septoria_leaf_spot": 0.675,
    "tomato_yellow_leaf_curl_virus": 0.724,
}
PHASE2_MACRO = 0.5630

print("=" * 70, flush=True)
print(f"RESULTS: COVERAGE THRESHOLD {THRESHOLD_BEING_TESTED}", flush=True)
print("=" * 70, flush=True)
print(flush=True)
print(f"PlantDoc macro F1:  {macro_f1:.4f}", flush=True)
print(f"Phase 2 baseline:   {PHASE2_MACRO:.4f}", flush=True)
change = macro_f1 - PHASE2_MACRO
print(f"Change:             {change:+.4f}", flush=True)
print(flush=True)

print(f"Per-class breakdown:", flush=True)
print(f"  {'Class':<42} {'Phase2':>7} {'Now':>7} {'Change':>8} {'P':>6} {'R':>6} {'Support':>8}",
      flush=True)
print("  " + "-" * 85, flush=True)

regressions = []
improvements = []
for i, cls_name in enumerate(present_classes):
    baseline = PHASE2_BASELINE.get(cls_name, 0.0)
    current = per_class_f1[i]
    diff = current - baseline
    diff_str = f"+{diff:.3f}" if diff >= 0 else f"{diff:.3f}"
    support = int(all_labels[:, present_indices[i]].sum())
    flag = ""
    if diff < -0.05:
        flag = " REGRESSION"
        regressions.append((cls_name, diff))
    elif diff > 0.01:
        flag = " improved"
        improvements.append((cls_name, diff))
    print(f"  {cls_name:<42} {baseline:>7.3f} {current:>7.3f} "
          f"{diff_str:>8} {per_class_p[i]:>6.3f} {per_class_r[i]:>6.3f} "
          f"{support:>8}{flag}", flush=True)

print("  " + "-" * 85, flush=True)
macro_str = f"{change:+.4f}"
print(f"  {'MACRO F1':<42} {PHASE2_MACRO:>7.4f} {macro_f1:>7.4f} {macro_str:>8}", flush=True)
print(flush=True)

print("=" * 70, flush=True)
print("VERDICT", flush=True)
print("=" * 70, flush=True)
if regressions:
    print("Classes with regression > 0.05:", flush=True)
    for cls, diff in regressions:
        print(f"  {cls}: {diff:.3f}", flush=True)
else:
    print("No class regressed by more than 0.05 F1", flush=True)

if improvements:
    print("Classes with improvement:", flush=True)
    for cls, diff in improvements:
        print(f"  {cls}: +{diff:.3f}", flush=True)

print(flush=True)
if macro_f1 > PHASE2_MACRO and not regressions:
    print(f"THRESHOLD {THRESHOLD_BEING_TESTED} IS ACCEPTABLE", flush=True)
elif macro_f1 > PHASE2_MACRO and regressions:
    print(f"THRESHOLD {THRESHOLD_BEING_TESTED} HAS REGRESSIONS", flush=True)
else:
    print(f"THRESHOLD {THRESHOLD_BEING_TESTED} IS NOT ACCEPTABLE", flush=True)
    print(f"PlantDoc F1 did not improve vs Phase 2 baseline", flush=True)

print(flush=True)
print(f"Second pass trigger rate: {100*second_pass_count/max(len(all_probs),1):.1f}%", flush=True)
print(f"Avg inference time: {total_time/max(len(all_probs),1):.1f}s/image", flush=True)
