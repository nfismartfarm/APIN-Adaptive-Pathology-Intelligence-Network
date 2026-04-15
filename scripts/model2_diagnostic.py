"""
Model 2 Post-Training Diagnostic
Runs: confusion matrix, per-source shortcut detection, conformal thresholds, production save.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config_model2 import (
    NUM_CLASSES, CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS,
    DINOV3_BACKBONE, BACKBONE_EMBED_DIM,
)
from scripts.models import Model2ConvNeXt

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleDS(Dataset):
    def __init__(self, paths, labels, img_size=384):
        self.paths, self.labels = paths, labels
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        c = p.replace(os.sep+'cleaned'+os.sep, os.sep+'cleaned_clahe'+os.sep)
        if os.path.exists(c): p = c
        try: img = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)
        except: img = np.zeros((384, 384, 3), dtype=np.uint8)
        return self.transform(image=img)['image'], self.labels[idx]


@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labs in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits = model(imgs.to(device))
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.extend(labs.numpy())
    return np.vstack(all_probs), np.array(all_labels)


def run_diagnostic():
    # Load model
    model = Model2ConvNeXt(num_classes=NUM_CLASSES, pretrained=False).to(device)
    ckpt_path = ROOT / 'models' / 'model2_specialist' / 'model2_best.pt'
    if not ckpt_path.exists():
        ckpt_path = ROOT / 'models' / 'model2_specialist' / 'model2_stage1_best.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded: {ckpt_path.name}, F1={ckpt.get('best_f1', '?'):.4f}", flush=True)

    # Load data
    df = pd.read_csv(ROOT / 'data' / 'specialist' / 'model2' / 'model2_unified_source_map.csv')
    with open(ROOT / 'data' / 'specialist' / 'model2' / 'split_indices.json') as f:
        splits = json.load(f)

    # ======================================================================
    # 1. CONFUSION MATRIX on val set
    # ======================================================================
    print("\n" + "=" * 70)
    print("1. CONFUSION MATRIX (val set)")
    print("=" * 70, flush=True)

    val_idx = splits.get('val_and_soup', splits.get('val'))
    val_df = df.iloc[val_idx].reset_index(drop=True)
    val_labels = [CLASS_TO_IDX[c] for c in val_df['class_name']]
    val_ds = SimpleDS(val_df['image_path'].tolist(), val_labels, img_size=384)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    probs, labels = predict_all(model, val_loader)
    preds = probs.argmax(axis=1)
    macro_f1 = f1_score(labels, preds, average='macro',
                         labels=list(range(NUM_CLASSES)), zero_division=0)

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    print(f"\nMacro F1: {macro_f1:.4f}")
    print(f"\nPredicted -> {' '.join(f'{c[:8]:>9s}' for c in CLASS_NAMES)}")
    for i, row in enumerate(cm):
        total = max(row.sum(), 1)
        pcts = [f"{x/total*100:8.1f}%" for x in row]
        print(f"  {CLASS_NAMES[i][:12]:>12s} {' '.join(pcts)}  (n={row.sum()})")

    print(f"\n{classification_report(labels, preds, target_names=CLASS_NAMES, digits=4, labels=list(range(NUM_CLASSES)))}")

    # Cross-class confusions > 1%
    print("Significant confusions (>1%):")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i][j] > 0:
                pct = cm[i][j] / max(cm[i].sum(), 1) * 100
                if pct > 1.0:
                    crop_i = 'okra' if i < 5 else 'brassica'
                    crop_j = 'okra' if j < 5 else 'brassica'
                    cross = 'CROSS-CROP' if crop_i != crop_j else 'same crop'
                    print(f"  {CLASS_NAMES[i]:>25s} -> {CLASS_NAMES[j]:<25s}: "
                          f"{cm[i][j]:3d} ({pct:.1f}%) [{cross}]")

    # ======================================================================
    # 2. PER-SOURCE SHORTCUT DETECTION
    # ======================================================================
    print("\n" + "=" * 70)
    print("2. PER-SOURCE SHORTCUT DETECTION")
    print("=" * 70, flush=True)

    is_field = val_df['is_field_photo'].astype(str).str.lower().isin(['true']).values
    sources = val_df['source_dataset'].values

    for ci, cls in enumerate(CLASS_NAMES):
        mask = labels == ci
        if not mask.any(): continue
        n = mask.sum()
        acc = (preds[mask] == ci).mean() * 100
        fld = mask & is_field
        lab = mask & ~is_field
        fld_acc = (preds[fld] == ci).mean() * 100 if fld.any() else 0
        lab_acc = (preds[lab] == ci).mean() * 100 if lab.any() else 0
        gap = abs(fld_acc - lab_acc)
        flag = "SHORTCUT" if gap > 10 else "MONITOR" if gap > 5 else "OK"
        print(f"{cls:>25s} (n={n:4d}): field={fld_acc:5.1f}% lab={lab_acc:5.1f}% gap={gap:.1f}% [{flag}]")

    # ======================================================================
    # 3. WRONG PREDICTIONS ANALYSIS
    # ======================================================================
    print("\n" + "=" * 70)
    print("3. WRONG PREDICTIONS")
    print("=" * 70, flush=True)

    wrong = preds != labels
    max_confs = probs.max(axis=1)
    n_wrong = wrong.sum()
    print(f"Total wrong: {n_wrong}/{len(labels)} ({n_wrong/len(labels)*100:.2f}%)")
    if n_wrong > 0:
        print(f"Wrong mean conf: {max_confs[wrong].mean():.4f}")
        print(f"Wrong max conf:  {max_confs[wrong].max():.4f}")
        print(f"Correct mean conf: {max_confs[~wrong].mean():.4f}")
        # Show high-confidence errors
        high_conf_wrong = np.where(wrong & (max_confs > 0.8))[0]
        if len(high_conf_wrong) > 0:
            print(f"\nHigh-confidence errors (conf > 0.8): {len(high_conf_wrong)}")
            for i in high_conf_wrong[:15]:
                print(f"  True={CLASS_NAMES[labels[i]]:>25s} Pred={CLASS_NAMES[preds[i]]:<25s} "
                      f"Conf={max_confs[i]:.3f} [{'field' if is_field[i] else 'lab':5s}] {sources[i]}")

    # ======================================================================
    # 4. CONFORMAL THRESHOLDS (APS)
    # ======================================================================
    print("\n" + "=" * 70)
    print("4. CONFORMAL THRESHOLDS (APS)")
    print("=" * 70, flush=True)

    if 'conformal' in splits:
        conf_idx = splits['conformal']
        conf_df = df.iloc[conf_idx].reset_index(drop=True)
        conf_labels = [CLASS_TO_IDX[c] for c in conf_df['class_name']]
        print(f"Conformal split: {len(conf_idx)} images", flush=True)

        conf_ds = SimpleDS(conf_df['image_path'].tolist(), conf_labels, img_size=384)
        conf_loader = DataLoader(conf_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

        conf_probs, conf_labels_arr = predict_all(model, conf_loader)
        conf_preds = conf_probs.argmax(axis=1)
        conf_f1 = f1_score(conf_labels_arr, conf_preds, average='macro',
                           labels=list(range(NUM_CLASSES)), zero_division=0)
        print(f"Conformal split F1: {conf_f1:.4f}")

        # Per-class APS thresholds
        alpha = 0.05
        thresholds = {}
        for ci, cls in enumerate(CLASS_NAMES):
            cmask = conf_labels_arr == ci
            if not cmask.any():
                thresholds[cls] = 0.5; continue
            correct = (conf_preds == ci) & cmask
            if not correct.any():
                thresholds[cls] = 0.9; continue
            correct_confs = conf_probs[correct, ci]
            threshold = float(np.quantile(correct_confs, alpha))
            thresholds[cls] = round(threshold, 4)
            n_below = (conf_probs[cmask, ci] < threshold).sum()
            print(f"  {cls:>25s}: threshold={threshold:.4f} "
                  f"(abstain {n_below}/{cmask.sum()} = {n_below/cmask.sum()*100:.1f}%)")

        thresh_path = ROOT / 'data' / 'specialist' / 'model2' / 'conformal_thresholds.json'
        with open(thresh_path, 'w') as f:
            json.dump(thresholds, f, indent=2)
        print(f"\nSaved: {thresh_path}")

        # Abstention rate
        abstain = sum(1 for i in range(len(conf_preds))
                      if conf_probs[i, conf_preds[i]] < thresholds[CLASS_NAMES[conf_preds[i]]])
        print(f"Abstention rate: {abstain}/{len(conf_labels_arr)} = {abstain/len(conf_labels_arr)*100:.1f}% (target: 2-5%)")
    else:
        print("WARNING: No conformal split!")

    # ======================================================================
    # 5. SAVE PRODUCTION CHECKPOINT
    # ======================================================================
    print("\n" + "=" * 70)
    print("5. PRODUCTION CHECKPOINT")
    print("=" * 70, flush=True)

    prod_path = ROOT / 'models' / 'model2_specialist' / 'model2_production.pt'
    torch.save({
        'backbone_name': DINOV3_BACKBONE,
        'embed_dim': BACKBONE_EMBED_DIM,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'model_state_dict': model.state_dict(),
        'val_f1': macro_f1,
        'conformal_thresholds': thresholds if 'conformal' in splits else {},
        'img_size': 384,
        'training_recipe': 'Stage1(25ep progressive 128->224->384 ASAM+SupCon+LLRD) + Stage2(7ep CutMix+Focal)',
    }, prod_path)
    print(f"Saved: {prod_path} ({prod_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Val F1: {macro_f1:.4f}")

    print("\n" + "=" * 70)
    print("MODEL 2 DIAGNOSTIC COMPLETE")
    print("=" * 70, flush=True)


if __name__ == '__main__':
    run_diagnostic()
