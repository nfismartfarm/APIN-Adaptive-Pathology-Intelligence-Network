"""
Router Post-Training Diagnostic — Full analysis per MASTER_PLAN Section 4.
Runs: confusion matrix, per-source F1, shortcut detection, conformal thresholds.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config_router import (
    BACKBONE_NAME, DINOV2_IMG_SIZE, DINOV2_EMBED_DIM,
    NUM_CLASSES, CLASS_NAMES, CLASS_TO_IDX,
)

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    import timm
    backbone = timm.create_model(BACKBONE_NAME, pretrained=True,
                                  num_classes=0, img_size=DINOV2_IMG_SIZE)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    head = nn.Linear(DINOV2_EMBED_DIM, NUM_CLASSES).to(device)
    ema_head = nn.Linear(DINOV2_EMBED_DIM, NUM_CLASSES).to(device)

    ckpt = torch.load(ROOT / 'models' / 'router' / 'router_best.pt',
                       map_location=device, weights_only=False)
    head.load_state_dict(ckpt['head_state_dict'])
    ema_head.load_state_dict(ckpt['ema_head_state_dict'])
    head.eval()
    ema_head.eval()
    print(f"Loaded: epoch={ckpt['epoch']}, best_f1={ckpt['best_f1']:.4f}")
    return backbone, head, ema_head


def run_diagnostic():
    backbone, head, ema_head = load_model()

    # Load cached val features
    val_cache = torch.load(ROOT / 'cache' / 'router' / 'val_features.pt', weights_only=False)
    val_feats = val_cache['features'].to(device)
    val_labels = val_cache['labels'].numpy()

    # Load CSV for source info
    df = pd.read_csv(ROOT / 'data' / 'specialist' / 'router' / 'router_unified_source_map.csv')
    with open(ROOT / 'data' / 'specialist' / 'router' / 'split_indices.json') as f:
        splits = json.load(f)
    val_df = df.iloc[splits['val']].reset_index(drop=True)
    is_field = val_df['is_field_photo'].astype(str).str.lower().isin(['true']).values
    sources = val_df['source_dataset'].values

    # ── Inference ─────────────────────────────────────────────────────────
    # [FIX] EMA head is broken (decay=0.9999 too slow for 11-epoch training).
    # Use REGULAR head (F1=0.9862) for all diagnostics.
    with torch.no_grad():
        reg_logits = head(val_feats)
        reg_probs = torch.softmax(reg_logits.float(), dim=1).cpu().numpy()
        reg_preds = reg_probs.argmax(axis=1)

        ema_logits = ema_head(val_feats)
        reg_probs = torch.softmax(ema_logits.float(), dim=1).cpu().numpy()
        reg_preds = reg_probs.argmax(axis=1)

    # ══════════════════════════════════════════════════════════════════════
    # 1. CONFUSION MATRIX
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("1. CONFUSION MATRIX (EMA Head)")
    print("=" * 70)

    ema_f1 = f1_score(val_labels, reg_preds, average='macro')
    reg_f1 = f1_score(val_labels, reg_preds, average='macro')
    print(f"Regular head F1: {reg_f1:.4f}")
    print(f"EMA head F1:     {ema_f1:.4f}")

    cm = confusion_matrix(val_labels, reg_preds)
    print(f"\nPredicted ->  {'  '.join(f'{c:>8s}' for c in CLASS_NAMES)}")
    for i, row in enumerate(cm):
        pcts = [f"{x/row.sum()*100:7.1f}%" for x in row]
        print(f"  {CLASS_NAMES[i]:>10s}  {'  '.join(pcts)}   (n={row.sum()})")

    print(f"\nFull classification report (EMA):")
    print(classification_report(val_labels, reg_preds, target_names=CLASS_NAMES, digits=4))

    # Dangerous off-diagonal: highlight any >1% confusion
    print("Cross-crop confusion analysis:")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i][j] > 0:
                pct = cm[i][j] / cm[i].sum() * 100
                danger = "DANGEROUS" if pct > 2 else "minor" if pct > 0.5 else "negligible"
                # Check if same specialist
                spec_i = 'M2' if CLASS_NAMES[i] in ('okra','brassica') else 'M3'
                spec_j = 'M2' if CLASS_NAMES[j] in ('okra','brassica') else 'M3'
                routing_impact = "WRONG SPECIALIST" if spec_i != spec_j else "same specialist"
                print(f"  {CLASS_NAMES[i]:>10s} -> {CLASS_NAMES[j]:<10s}: "
                      f"{cm[i][j]:3d} ({pct:.1f}%) [{danger}] [{routing_impact}]")

    # ══════════════════════════════════════════════════════════════════════
    # 2. PER-SOURCE BREAKDOWN (shortcut detection)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("2. PER-SOURCE F1 BREAKDOWN (shortcut detection)")
    print("=" * 70)

    for crop_idx, crop in enumerate(CLASS_NAMES):
        crop_mask = val_labels == crop_idx
        n_crop = crop_mask.sum()
        if n_crop == 0:
            continue

        crop_acc = (reg_preds[crop_mask] == crop_idx).mean() * 100

        field_mask = crop_mask & is_field
        lab_mask = crop_mask & ~is_field
        field_acc = (reg_preds[field_mask] == crop_idx).mean() * 100 if field_mask.any() else 0
        lab_acc = (reg_preds[lab_mask] == crop_idx).mean() * 100 if lab_mask.any() else 0
        field_n = field_mask.sum()
        lab_n = lab_mask.sum()
        gap = abs(field_acc - lab_acc)

        flag = "SHORTCUT RISK" if gap > 10 else "MONITOR" if gap > 5 else "OK"
        print(f"\n{crop.upper()} (n={n_crop}, overall={crop_acc:.1f}%):")
        print(f"  Field: {field_acc:5.1f}%  (n={field_n}, {field_n/n_crop*100:.0f}% of {crop})")
        print(f"  Lab:   {lab_acc:5.1f}%  (n={lab_n}, {lab_n/n_crop*100:.0f}% of {crop})")
        print(f"  Gap:   {gap:.1f}%  [{flag}]")

        # Per-source
        crop_sources = sources[crop_mask]
        crop_preds_sub = reg_preds[crop_mask]
        for src in sorted(set(crop_sources)):
            src_mask = crop_sources == src
            src_acc = (crop_preds_sub[src_mask] == crop_idx).mean() * 100
            print(f"    {src:35s}: {src_acc:5.1f}%  (n={src_mask.sum()})")

    # ══════════════════════════════════════════════════════════════════════
    # 3. CONFIDENCE ANALYSIS + WRONG PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3. CONFIDENCE ANALYSIS")
    print("=" * 70)

    max_confs = reg_probs.max(axis=1)
    correct = reg_preds == val_labels
    wrong = ~correct

    print(f"\nCorrect (n={correct.sum()}):")
    print(f"  Mean conf: {max_confs[correct].mean():.4f}")
    print(f"  Min conf:  {max_confs[correct].min():.4f}")
    print(f"  P10 conf:  {np.percentile(max_confs[correct], 10):.4f}")

    if wrong.any():
        print(f"\nWrong (n={wrong.sum()}):")
        print(f"  Mean conf: {max_confs[wrong].mean():.4f}")
        print(f"  Max conf:  {max_confs[wrong].max():.4f} <-- most dangerous")
        print(f"\n  Detailed wrong predictions:")
        for i in np.where(wrong)[0]:
            true_cls = CLASS_NAMES[val_labels[i]]
            pred_cls = CLASS_NAMES[reg_preds[i]]
            conf = max_confs[i]
            src = sources[i] if i < len(sources) else '?'
            fld = 'field' if is_field[i] else 'lab'
            print(f"    True={true_cls:10s} Pred={pred_cls:10s} "
                  f"Conf={conf:.3f} [{fld:5s}] src={src}")
    else:
        print("\nNo wrong predictions on validation set!")

    # ══════════════════════════════════════════════════════════════════════
    # 4. CONFORMAL ROUTING THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("4. CONFORMAL ROUTING THRESHOLDS")
    print("=" * 70)

    if 'conformal' in splits:
        conf_idx = splits['conformal']
        conf_df = df.iloc[conf_idx].reset_index(drop=True)
        print(f"Conformal split: {len(conf_idx)} images")

        # Compute features for conformal split
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        transform = A.Compose([
            A.Resize(DINOV2_IMG_SIZE, DINOV2_IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        class SimpleDS(Dataset):
            def __init__(self, paths, labels, tfm):
                self.paths, self.labels, self.tfm = paths, labels, tfm
            def __len__(self): return len(self.paths)
            def __getitem__(self, idx):
                p = self.paths[idx]
                clahe = p.replace(os.sep+'cleaned'+os.sep, os.sep+'cleaned_clahe'+os.sep)
                if os.path.exists(clahe): p = clahe
                try: img = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)
                except: img = np.zeros((224,224,3), dtype=np.uint8)
                return self.tfm(image=img)['image'], self.labels[idx]

        conf_labels = [CLASS_TO_IDX[c] for c in conf_df['crop']]
        conf_ds = SimpleDS(conf_df['image_path'].tolist(), conf_labels, transform)
        conf_loader = DataLoader(conf_ds, batch_size=128, shuffle=False, num_workers=0)

        print("Computing features on conformal split...", flush=True)
        conf_probs_list = []
        conf_labels_list = []
        with torch.no_grad():
            for imgs, labs in conf_loader:
                feats = backbone(imgs.to(device))
                logits = head(feats)  # Use regular head, not broken EMA
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                conf_probs_list.append(probs)
                conf_labels_list.extend(labs.numpy())

        conf_probs = np.vstack(conf_probs_list)
        conf_labels_arr = np.array(conf_labels_list)
        conf_preds = conf_probs.argmax(axis=1)

        conf_f1 = f1_score(conf_labels_arr, conf_preds, average='macro')
        print(f"Conformal split F1: {conf_f1:.4f}")

        # Per-crop thresholds at alpha=0.05
        alpha = 0.05
        thresholds = {}
        for crop_idx, crop in enumerate(CLASS_NAMES):
            crop_mask = conf_labels_arr == crop_idx
            if not crop_mask.any():
                thresholds[crop] = 0.5
                continue
            correct = (conf_preds == crop_idx) & crop_mask
            if not correct.any():
                thresholds[crop] = 0.9
                continue
            correct_confs = conf_probs[correct, crop_idx]
            threshold = float(np.quantile(correct_confs, alpha))
            thresholds[crop] = round(threshold, 4)
            n_below = (conf_probs[crop_mask, crop_idx] < threshold).sum()
            print(f"  {crop:10s}: threshold={threshold:.4f} "
                  f"(abstain {n_below}/{crop_mask.sum()} = {n_below/crop_mask.sum()*100:.1f}%)")

        # Save
        thresh_path = ROOT / 'data' / 'specialist' / 'router' / 'conformal_thresholds.json'
        with open(thresh_path, 'w') as f:
            json.dump(thresholds, f, indent=2)
        print(f"\nSaved thresholds: {thresh_path}")

        # Simulated abstention
        max_conf = conf_probs.max(axis=1)
        pred_crops = [CLASS_NAMES[p] for p in conf_preds]
        abstain = sum(1 for i, c in enumerate(pred_crops) if max_conf[i] < thresholds[c])
        rate = abstain / len(conf_labels_arr) * 100
        print(f"Simulated abstention: {abstain}/{len(conf_labels_arr)} = {rate:.1f}% (target: 2-5%)")
    else:
        print("WARNING: No conformal split found!")

    # ══════════════════════════════════════════════════════════════════════
    # 5. SAVE PRODUCTION ROUTER
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("5. PRODUCTION ROUTER SAVE")
    print("=" * 70)

    # Save the full model state (backbone + EMA head) as production checkpoint
    prod_path = ROOT / 'models' / 'router' / 'router_production.pt'
    torch.save({
        'backbone_name': BACKBONE_NAME,
        'embed_dim': DINOV2_EMBED_DIM,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'head_state_dict': head.state_dict(),  # Regular head (EMA broken)
        'val_f1': reg_f1,
        'img_size': DINOV2_IMG_SIZE,
    }, prod_path)
    print(f"Production router saved: {prod_path}")
    print(f"  Using: Regular head (F1={reg_f1:.4f})")
    print(f"  Size: {prod_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 70)
    print("ROUTER DIAGNOSTIC COMPLETE")
    print("=" * 70, flush=True)


if __name__ == '__main__':
    run_diagnostic()
