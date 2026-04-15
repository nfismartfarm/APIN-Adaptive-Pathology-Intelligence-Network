"""Phase 4: Complete evaluation — Groups 4.1 through 4.4 in one script."""
import numpy as np, torch, pandas as pd, os, sys, time, json
import torch.nn.functional as F_torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (DEVICE, ROOT, BEST_MODEL, CLASS_NAMES, DISEASE_THRESHOLDS,
                          NUM_CLASSES, CROP_NAMES, CROP_TO_DISEASE_INDICES, FPN_OUT_CH,
                          NUM_CROPS, CLASS_TO_IDX, CROP_FROM_IDX)
from app.model import load_model_for_inference
from app.inference import preprocess_single, preprocess_for_inference
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

print("Loading model...", flush=True)
model = load_model_for_inference(os.path.join(ROOT, BEST_MODEL), DEVICE)
model.eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)
print(flush=True)

df = pd.read_csv(os.path.join(ROOT, "data", "metadata", "source_map.csv"))
thresholds_arr = np.array([DISEASE_THRESHOLDS.get(c, 0.5) for c in CLASS_NAMES])

plantdoc_classes = sorted([
    "tomato_bacterial_spot", "tomato_early_blight", "tomato_late_blight",
    "tomato_leaf_mold", "tomato_mosaic_virus", "tomato_septoria_leaf_spot",
    "tomato_yellow_leaf_curl_virus",
])
plantdoc_indices = [CLASS_NAMES.index(c) for c in plantdoc_classes]


def get_probs(img_np):
    """Run single image through model, return disease probs."""
    tensor = preprocess_single(img_np).to(DEVICE)
    with torch.no_grad():
        _, d, _ = model(tensor)
    return torch.sigmoid(d).cpu().numpy()[0]


def load_and_resize(path, max_dim=800):
    img_pil = Image.open(path).convert("RGB")
    if max(img_pil.size) > max_dim:
        ratio = max_dim / max(img_pil.size)
        img_pil = img_pil.resize((int(img_pil.size[0]*ratio), int(img_pil.size[1]*ratio)))
    return np.array(img_pil)


# ======================================================================
# GROUP 4.1: CROPPED PLANTDOC F1
# ======================================================================
print("=" * 70, flush=True)
print("GROUP 4.1: CROPPED PLANTDOC F1 (center-crop 50%)", flush=True)
print("=" * 70, flush=True)

pd_eval = df[df["source_dataset"].str.contains("plantdoc_eval", case=False, na=False)]

probs_full_list, probs_crop_list, labels_list = [], [], []
for i, (_, row) in enumerate(pd_eval.iterrows()):
    if i % 100 == 0:
        print(f"  [{i}/{len(pd_eval)}]...", flush=True)
    img_path = row["image_path"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(ROOT, img_path)
    try:
        img = load_and_resize(img_path)
        H, W = img.shape[:2]

        probs_full_list.append(get_probs(img))

        # Center crop 50%
        y1 = int(H * 0.25); y2 = int(H * 0.75)
        x1 = int(W * 0.25); x2 = int(W * 0.75)
        crop = img[y1:y2, x1:x2]
        probs_crop_list.append(get_probs(crop))

        label = np.zeros(NUM_CLASSES)
        label[CLASS_NAMES.index(row["class_name"])] = 1
        labels_list.append(label)
    except:
        pass

probs_full = np.array(probs_full_list)
probs_crop = np.array(probs_crop_list)
labels_all = np.array(labels_list)

preds_full = (probs_full > thresholds_arr).astype(int)
preds_crop = (probs_crop > thresholds_arr).astype(int)

l7 = labels_all[:, plantdoc_indices]
pf7 = preds_full[:, plantdoc_indices]
pc7 = preds_crop[:, plantdoc_indices]

macro_full = f1_score(l7, pf7, average="macro", zero_division=0)
macro_crop = f1_score(l7, pc7, average="macro", zero_division=0)
pcf_full = f1_score(l7, pf7, average=None, zero_division=0)
pcf_crop = f1_score(l7, pc7, average=None, zero_division=0)

print(f"\nFull image macro F1:  {macro_full:.4f}", flush=True)
print(f"Cropped macro F1:     {macro_crop:.4f}", flush=True)
print(f"Change:               {macro_crop - macro_full:+.4f}", flush=True)
print(flush=True)

print(f"  {'Class':<42} {'Full':>7} {'Crop':>7} {'Change':>8}", flush=True)
print("  " + "-" * 65, flush=True)
for i, cls in enumerate(plantdoc_classes):
    d = pcf_crop[i] - pcf_full[i]
    ds = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
    print(f"  {cls:<42} {pcf_full[i]:>7.3f} {pcf_crop[i]:>7.3f} {ds:>8}", flush=True)
print("  " + "-" * 65, flush=True)
mc = macro_crop - macro_full
print(f"  {'MACRO':<42} {macro_full:>7.4f} {macro_crop:>7.4f} {mc:>+8.4f}", flush=True)

if macro_crop < macro_full - 0.03:
    crop_interpretation = "Cropping HURTS — model depends on global context"
elif macro_crop > macro_full + 0.05:
    crop_interpretation = "Background is a significant failure cause"
elif macro_crop > macro_full:
    crop_interpretation = "Background causes minor interference"
else:
    crop_interpretation = "Minimal effect — disease features are the bottleneck"
print(f"\nInterpretation: {crop_interpretation}", flush=True)
print(flush=True)

# Store for report
g41_results = {
    "macro_full": macro_full, "macro_crop": macro_crop,
    "per_class_full": pcf_full, "per_class_crop": pcf_crop,
    "interpretation": crop_interpretation,
}

# ======================================================================
# GROUP 4.2: PER-CLASS DOMAIN GAP TABLE
# ======================================================================
print("=" * 70, flush=True)
print("GROUP 4.2: PER-CLASS DOMAIN GAP TABLE", flush=True)
print("=" * 70, flush=True)

# Controlled test set (tomato classes)
test_df = df[df["split"] == "test"]
test_tomato = test_df[test_df["class_name"].isin(plantdoc_classes)]

print(f"Evaluating controlled test (tomato): {len(test_tomato)} images...", flush=True)
probs_test, labels_test = [], []
for i, (_, row) in enumerate(test_tomato.iterrows()):
    if i % 200 == 0 and i > 0:
        print(f"  [{i}/{len(test_tomato)}]...", flush=True)
    img_path = row["image_path"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(ROOT, img_path)
    try:
        img = load_and_resize(img_path)
        probs_test.append(get_probs(img))
        label = np.zeros(NUM_CLASSES)
        label[CLASS_NAMES.index(row["class_name"])] = 1
        labels_test.append(label)
    except:
        pass

probs_test = np.array(probs_test)
labels_test = np.array(labels_test)
preds_test = (probs_test > thresholds_arr).astype(int)

print(f"  Done: {len(probs_test)} images", flush=True)
print(flush=True)

# PlantDoc already computed above
gap_results = []
print(f"  {'Class':<42} {'Ctrl F1':>8} {'PD F1':>8} {'Gap':>7} {'Failure mode':<20}", flush=True)
print("  " + "-" * 90, flush=True)

for i, cls in enumerate(plantdoc_classes):
    cidx = CLASS_NAMES.index(cls)

    # Controlled
    ct_l = labels_test[:, cidx]
    ct_p = preds_test[:, cidx]
    sup_ct = int(ct_l.sum())
    f1_ct = f1_score(ct_l, ct_p, zero_division=0) if sup_ct > 0 else 0.0
    p_ct = precision_score(ct_l, ct_p, zero_division=0) if sup_ct > 0 else 0.0
    r_ct = recall_score(ct_l, ct_p, zero_division=0) if sup_ct > 0 else 0.0

    # PlantDoc
    pd_l = labels_all[:, cidx]
    pd_p = preds_full[:, cidx]
    sup_pd = int(pd_l.sum())
    f1_pd = f1_score(pd_l, pd_p, zero_division=0) if sup_pd > 0 else 0.0
    p_pd = precision_score(pd_l, pd_p, zero_division=0) if sup_pd > 0 else 0.0
    r_pd = recall_score(pd_l, pd_p, zero_division=0) if sup_pd > 0 else 0.0

    gap = f1_ct - f1_pd
    if r_pd < 0.30 and p_pd > 0.40:
        failure = "Low recall (misses)"
    elif p_pd < 0.30 and r_pd > 0.40:
        failure = "Low precision (FPs)"
    elif f1_pd > 0.70:
        failure = "Transfers well"
    else:
        failure = "Mixed"

    gap_results.append({
        "class": cls, "f1_ct": f1_ct, "p_ct": p_ct, "r_ct": r_ct, "sup_ct": sup_ct,
        "f1_pd": f1_pd, "p_pd": p_pd, "r_pd": r_pd, "sup_pd": sup_pd,
        "gap": gap, "failure": failure,
    })
    print(f"  {cls:<42} {f1_ct:>8.3f} {f1_pd:>8.3f} {gap:>+7.3f} {failure:<20}", flush=True)

gap_results.sort(key=lambda x: x["gap"], reverse=True)
avg_ct = np.mean([r["f1_ct"] for r in gap_results])
avg_pd = np.mean([r["f1_pd"] for r in gap_results])
print("  " + "-" * 90, flush=True)
print(f"  {'MACRO':<42} {avg_ct:>8.3f} {avg_pd:>8.3f} {avg_ct-avg_pd:>+7.3f}", flush=True)
print(flush=True)

print(f"Largest gap:  {gap_results[0]['class']} ({gap_results[0]['gap']:+.3f})", flush=True)
print(f"Smallest gap: {gap_results[-1]['class']} ({gap_results[-1]['gap']:+.3f})", flush=True)
print(flush=True)

# Detailed P/R table
print(f"  {'Class':<42} {'Ct P':>6} {'Ct R':>6} {'PD P':>6} {'PD R':>6} {'Ct N':>6} {'PD N':>6}", flush=True)
print("  " + "-" * 80, flush=True)
for r in gap_results:
    print(f"  {r['class']:<42} {r['p_ct']:>6.3f} {r['r_ct']:>6.3f} "
          f"{r['p_pd']:>6.3f} {r['r_pd']:>6.3f} {r['sup_ct']:>6d} {r['sup_pd']:>6d}", flush=True)
print(flush=True)

# ======================================================================
# GROUP 4.3: COMPONENT ABLATION STUDY
# ======================================================================
print("=" * 70, flush=True)
print("GROUP 4.3: COMPONENT ABLATION STUDY", flush=True)
print("=" * 70, flush=True)

# Use 500 val images for speed
val_df = df[df["split"] == "val"].reset_index(drop=True)
if len(val_df) > 500:
    val_sample = val_df.sample(500, random_state=42).reset_index(drop=True)
else:
    val_sample = val_df

print(f"Ablation on {len(val_sample)} val images", flush=True)
print(flush=True)


def eval_model_on(sample_df, label):
    all_p, all_l = [], []
    for _, row in sample_df.iterrows():
        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(ROOT, img_path)
        try:
            img = load_and_resize(img_path)
            all_p.append(get_probs(img))
            lbl = np.zeros(NUM_CLASSES)
            lbl[CLASS_NAMES.index(row["class_name"])] = 1
            all_l.append(lbl)
        except:
            pass
    ap = np.array(all_p)
    al = np.array(all_l)
    preds = (ap > thresholds_arr).astype(int)
    f1 = f1_score(al, preds, average="macro", zero_division=0)
    print(f"  {label}: macro F1={f1:.4f} ({len(ap)} images)", flush=True)
    return f1


# BASELINE
print("BASELINE...", flush=True)
baseline_f1 = eval_model_on(val_sample, "Full model")
print(flush=True)

# A: Remove Attention Pooling
print("ABLATION A: GAP instead of AttPool...", flush=True)
orig_att = model.att_pool.forward
model.att_pool.forward = lambda x: F_torch.adaptive_avg_pool2d(x, 1).flatten(1)
abl_a = eval_model_on(val_sample, "Without AttPool")
model.att_pool.forward = orig_att
print(f"  Contribution: {baseline_f1 - abl_a:+.4f}", flush=True)
print(flush=True)

# B: Remove CLN
print("ABLATION B: Identity instead of CLN...", flush=True)
orig_cln = model.cln.forward
model.cln.forward = lambda x, cp: x
abl_b = eval_model_on(val_sample, "Without CLN")
model.cln.forward = orig_cln
print(f"  Contribution: {baseline_f1 - abl_b:+.4f}", flush=True)
print(flush=True)

# C: Uniform MoE routing
print("ABLATION C: Uniform expert weights...", flush=True)
orig_moe = model.disease_head.forward
def uniform_moe(x, cp):
    return orig_moe(x, torch.ones_like(cp) * 0.25)
model.disease_head.forward = uniform_moe
abl_c = eval_model_on(val_sample, "Uniform MoE")
model.disease_head.forward = orig_moe
print(f"  Contribution: {baseline_f1 - abl_c:+.4f}", flush=True)
print(flush=True)

# D: No cross-crop masking
print("ABLATION D: No cross-crop masking...", flush=True)
orig_mask = model.disease_head.crop_mask.clone()
model.disease_head.crop_mask = torch.ones_like(orig_mask)
abl_d = eval_model_on(val_sample, "No masking")
model.disease_head.crop_mask = orig_mask
print(f"  Contribution: {baseline_f1 - abl_d:+.4f}", flush=True)
print(flush=True)

print("ABLATION SUMMARY:", flush=True)
print(f"  {'Configuration':<45} {'F1':>8} {'Drop':>8}", flush=True)
print("  " + "-" * 65, flush=True)
print(f"  {'Full model (baseline)':<45} {baseline_f1:>8.4f} {'---':>8}", flush=True)
ablations = [
    ("- Attention Pooling (use GAP)", abl_a),
    ("- Conditional Layer Norm (identity)", abl_b),
    ("- MoE routing (uniform weights)", abl_c),
    ("- Cross-crop masking (all classes)", abl_d),
]
for name, f1 in ablations:
    print(f"  {name:<45} {f1:>8.4f} {baseline_f1-f1:>+8.4f}", flush=True)
print(flush=True)

components = sorted(ablations, key=lambda x: baseline_f1 - x[1], reverse=True)
print("Ranked by contribution:", flush=True)
for name, f1 in components:
    c = baseline_f1 - f1
    note = "significant" if c > 0.01 else "minor" if c > 0 else "negligible/negative"
    print(f"  {name}: {c:+.4f} ({note})", flush=True)
print(flush=True)

# ======================================================================
# GROUP 4.4: CROP CONFUSION MATRIX
# ======================================================================
print("=" * 70, flush=True)
print("GROUP 4.4: CROP CONFUSION MATRIX ON REAL-WORLD IMAGES", flush=True)
print("=" * 70, flush=True)

# Use iNaturalist images from source_map
inat_df = df[df["source_dataset"].str.startswith("inaturalist", na=False)]
print(f"iNaturalist images: {len(inat_df)}", flush=True)

crop_name_to_idx = {"okra": 0, "brassica": 1, "tomato": 2, "chilli": 3}
confusion = np.zeros((4, 4), dtype=int)
total_inat = 0

for _, row in inat_df.iterrows():
    img_path = row["image_path"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(ROOT, img_path)
    if not os.path.exists(img_path):
        continue

    cls_name = str(row.get("class_name", ""))
    true_crop = -1
    for cn, ci in crop_name_to_idx.items():
        if cls_name.startswith(cn):
            true_crop = ci
            break
    if true_crop == -1:
        continue

    try:
        img = load_and_resize(img_path)
        tensor = preprocess_single(img).to(DEVICE)
        with torch.no_grad():
            c_log, _, _ = model(tensor)
        pred_crop = int(torch.softmax(c_log, dim=-1)[0].argmax())
        confusion[true_crop, pred_crop] += 1
        total_inat += 1
    except:
        pass

print(f"Processed: {total_inat} images", flush=True)
print(flush=True)

if total_inat > 0:
    CROP_DISPLAY = ["okra", "brassica", "tomato", "chilli"]
    print("CONFUSION MATRIX (rows=true, cols=predicted):", flush=True)
    header = f"  {'':>10}" + "".join(f"{c:>10}" for c in CROP_DISPLAY)
    print(header, flush=True)
    print("  " + "-" * 50, flush=True)
    for i, tc in enumerate(CROP_DISPLAY):
        rt = confusion[i].sum()
        row_str = f"  {tc:>10}"
        for j in range(4):
            cnt = confusion[i, j]
            pct = 100 * cnt / max(rt, 1)
            row_str += f" {cnt:>3}({pct:>3.0f}%)"
        row_str += f"  n={rt}"
        print(row_str, flush=True)
    print(flush=True)

    # Per-crop accuracy
    print("Per-crop accuracy:", flush=True)
    overall_correct = 0
    for i, cn in enumerate(CROP_DISPLAY):
        rt = confusion[i].sum()
        if rt > 0:
            acc = 100 * confusion[i, i] / rt
            overall_correct += confusion[i, i]
            status = "PASS" if acc >= 80 else "BELOW" if acc >= 60 else "FAIL"
            print(f"  {cn}: {confusion[i,i]}/{rt} = {acc:.1f}% [{status}]", flush=True)
    overall_acc = 100 * overall_correct / max(total_inat, 1)
    print(f"  Overall: {overall_correct}/{total_inat} = {overall_acc:.1f}%", flush=True)
    print(flush=True)

    # Chilli specific
    chi = 3
    if confusion[chi].sum() > 0:
        chi_acc = 100 * confusion[chi, chi] / confusion[chi].sum()
        chi_tom = 100 * confusion[chi, 2] / confusion[chi].sum()
        print(f"CHILLI: accuracy={chi_acc:.1f}%, confused as tomato={chi_tom:.1f}%", flush=True)
else:
    print("No iNaturalist images processed", flush=True)

print(flush=True)

# ======================================================================
# SAVE ALL RESULTS FOR REPORT GENERATION
# ======================================================================
results = {
    "g41": g41_results,
    "g42": gap_results,
    "g43": {"baseline": baseline_f1, "ablations": ablations, "components": components},
    "g44": {"confusion": confusion.tolist(), "total": total_inat},
}
results_path = os.path.join(ROOT, "reports", "_phase4_results.json")
with open(results_path, "w") as f:
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.integer): return int(obj)
            return super().default(obj)
    json.dump(results, f, indent=2, cls=NpEncoder)
print(f"Results saved to {results_path}", flush=True)

print(flush=True)
print("=" * 70, flush=True)
print("ALL PHASE 4 GROUPS COMPLETE", flush=True)
print("=" * 70, flush=True)
