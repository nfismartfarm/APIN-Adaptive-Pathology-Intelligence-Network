"""Final evaluation on final_val split — TWICE (uncertainty + ground-truth crop labels).

Loads model3_production.pt (the soup), runs dual-stream eval on the locked
final_val split (3,224 images, never used during training or model selection),
and writes the results JSON to logs/full_final_evaluation.json.

Run via: python -m scripts.model3_training.run_final_evaluation
(Must be run as a module so __main__ guard works; uses num_workers=0 to
avoid Windows DataLoader spawn issues on inline scripts.)
"""
from __future__ import annotations

import sys
import json
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import argparse
    import torch
    from torch.utils.data import DataLoader

    from scripts.model3_training.architecture.model3_full import Model3
    from scripts.model3_training.data.model3_dataset import M3ValDataset, load_and_split_csv
    from scripts.model3_training.evaluation.dual_stream_eval import dual_stream_evaluate
    from scripts.model3_training.training.trainer import Trainer
    from scripts.model3_training.model3_config import (
        CLASS_NAMES, NUM_CLASSES, BATCH_SIZE, CHECKPOINT_DIR, LOG_DIR,
        ROOT as CFG_ROOT,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path relative to checkpoint dir; default model3_production.pt')
    parser.add_argument('--out-suffix', type=str, default='',
                        help='Suffix for output JSON filename (to avoid overwrite)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_name = args.checkpoint or 'model3_production.pt'
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    print(f'Loading: {ckpt_path.name}  ({ckpt_path.stat().st_size/1e6:.1f} MB)', flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f'  ingredient_epochs: {ckpt.get("ingredient_epochs")}', flush=True)
    print(f'  ingredient_stops:  {ckpt.get("ingredient_stops")}', flush=True)
    soup_f1 = ckpt.get("soup_selection_field_f1")
    if soup_f1 is not None:
        print(f'  soup_selection field_f1: {soup_f1:.4f}', flush=True)
    else:
        print(f'  (single checkpoint — no soup metadata)  '
              f'epoch={ckpt.get("epoch")}  '
              f'val_field_f1={ckpt.get("val_stats", {}).get("field_f1", "?")}', flush=True)

    model = Model3(n_classes=NUM_CLASSES, pretrained=False, use_lora=True).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    print('  loaded with strict=True', flush=True)

    print('\nLoading splits...', flush=True)
    splits = load_and_split_csv()
    final_val = splits['final_val']
    field_n = sum(1 for r in final_val if r['is_field_photo'])
    lab_n = len(final_val) - field_n
    print(f'  final_val: {len(final_val)} (field={field_n}, lab={lab_n})', flush=True)

    # num_workers=0 — single-threaded loading, no Windows spawn issues
    NW = 0

    # ── Eval 1: GROUND-TRUTH crop labels (ceiling) ──────────────────────
    print('\n=== EVAL 1: GROUND-TRUTH crop labels (ceiling estimate) ===', flush=True)
    ds_gt = M3ValDataset(items=final_val)
    dl_gt = DataLoader(ds_gt, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NW, pin_memory=True,
                       persistent_workers=False,
                       collate_fn=Trainer._collate)
    gt_result = dual_stream_evaluate(model, dl_gt, device, CLASS_NAMES)
    print(f'  overall_f1 = {gt_result["overall_f1"]:.4f}', flush=True)
    print(f'  field_f1   = {gt_result["field_f1"]:.4f}  (n={gt_result["n_field_val"]})', flush=True)
    print(f'  lab_f1     = {gt_result["lab_f1"]:.4f}  (n={gt_result["n_lab_val"]})', flush=True)

    # ── Eval 2: UNCERTAINTY mode ────────────────────────────────────────
    print('\n=== EVAL 2: UNCERTAINTY mode (conservative deployment estimate) ===', flush=True)
    class M3ValUncertain(M3ValDataset):
        def __getitem__(self, idx):
            d = super().__getitem__(idx)
            d['crop_mode_groundtruth'] = 2  # uncertainty for all
            d['crop_mode'] = 2
            return d
    ds_unc = M3ValUncertain(items=final_val)
    dl_unc = DataLoader(ds_unc, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NW, pin_memory=True,
                        persistent_workers=False,
                        collate_fn=Trainer._collate)
    unc_result = dual_stream_evaluate(model, dl_unc, device, CLASS_NAMES)
    print(f'  overall_f1 = {unc_result["overall_f1"]:.4f}', flush=True)
    print(f'  field_f1   = {unc_result["field_f1"]:.4f}  (n={unc_result["n_field_val"]})', flush=True)
    print(f'  lab_f1     = {unc_result["lab_f1"]:.4f}  (n={unc_result["n_lab_val"]})', flush=True)

    # ── Per-class table ────────────────────────────────────────────────
    print('\n=== Per-class FIELD F1 (Conservative | Ceiling) ===', flush=True)
    print(f'  {"Class":<35} {"Conserv":>8} {"Ceiling":>8}', flush=True)
    for c in CLASS_NAMES:
        cv = unc_result['per_class_field'].get(c, 0.0)
        gv = gt_result['per_class_field'].get(c, 0.0)
        print(f'  {c:<35} {cv:>8.4f} {gv:>8.4f}', flush=True)

    # ── Save ───────────────────────────────────────────────────────────
    out_path = LOG_DIR / f'full_final_evaluation{args.out_suffix}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'checkpoint': str(ckpt_path.relative_to(CFG_ROOT)),
            'ingredient_epochs': ckpt.get('ingredient_epochs'),
            'final_val_n': len(final_val),
            'final_val_field_n': field_n,
            'final_val_lab_n': lab_n,
            'ceiling_groundtruth_crop': gt_result,
            'conservative_uncertainty_mode': unc_result,
        }, f, indent=2)
    print(f'\nSaved: {out_path.relative_to(CFG_ROOT)}', flush=True)


if __name__ == '__main__':
    main()
