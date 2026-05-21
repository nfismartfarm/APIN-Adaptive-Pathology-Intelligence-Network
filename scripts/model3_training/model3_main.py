"""Model 3 training CLI.

Usage:
  python -m scripts.model3_training.model3_main --mode verify
  python -m scripts.model3_training.model3_main --mode full
  python -m scripts.model3_training.model3_main --mode evaluate --checkpoint <path>

Modes:
  verify   - 3-epoch LoRA verification (spec Part 8 gate)
  full     - full training run (max 20 epochs, early stopping patience=3)
  evaluate - load checkpoint, report dual-stream metrics on val + final_val

This is the orchestration entry point. The actual training loop lives in
training/trainer.py; this script chooses the mode, sets command-line-exposed
overrides, and dispatches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['verify', 'full', 'evaluate'], required=True)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for --mode evaluate')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override max_epochs (default: config values)')
    parser.add_argument('--no-mixstyle', action='store_true',
                        help='Disable MixStyle (ablation)')
    parser.add_argument('--no-cutmix', action='store_true',
                        help='Disable CutMix (ablation)')
    args = parser.parse_args()

    from scripts.model3_training.model3_config import MAX_EPOCHS, LORA_VERIFY_EPOCHS
    from scripts.model3_training.training.trainer import Trainer

    if args.mode == 'verify':
        max_e = args.max_epochs or LORA_VERIFY_EPOCHS
        trainer = Trainer(
            run_name='lora_verify',
            max_epochs=max_e,
            use_lora=True,
            cutmix_enabled=not args.no_cutmix,
            mixstyle_enabled=not args.no_mixstyle,
        )
        result = trainer.run()
        print(f"\n[verify] best_epoch={result['best_epoch']}, "
              f"best_stop={result['best_stopping_metric']:.4f}, "
              f"halt={result['halt_reason']}")

    elif args.mode == 'full':
        max_e = args.max_epochs or MAX_EPOCHS
        # Round B retrain: run_name='full_v3'. Per-epoch checkpoints become
        # full_v3_epoch{NN}_stop{X.XXXX}.pt (no conflict with existing
        # full_v2_*.pt or full_*.pt from prior runs). v1 + v2 production
        # checkpoints are preserved side-by-side for 3-way comparison.
        trainer = Trainer(
            run_name='full_v3',
            max_epochs=max_e,
            use_lora=True,
            cutmix_enabled=not args.no_cutmix,
            mixstyle_enabled=not args.no_mixstyle,
        )
        result = trainer.run()
        print(f"\n[full_v3] training best_epoch={result['best_epoch']}, "
              f"best_stop={result['best_stopping_metric']:.4f}, "
              f"halt={result['halt_reason']}")

        # PVA R3 Check 59: guard soup against diagnostic halts. If training
        # aborted due to CLASS_COLLAPSE or SEPTORIA_DIAGNOSTIC_HALT, we only
        # have a few epoch checkpoints that are all near-degenerate — souping
        # them would produce a garbage production model. Skip soup entirely
        # and surface the halt reason loudly.
        halt = result.get('halt_reason')
        allow_soup = (halt is None) or (isinstance(halt, str) and halt.startswith('EARLY_STOP'))
        if not allow_soup:
            print(f"\n[full_v3] SKIPPING soup — training halted for diagnostic reason: "
                  f"{halt}")
            print(f"[full_v3] No production checkpoint written. Investigate the halt "
                  f"before running soup manually.")
            return

        # Post-training greedy soup — part of the official --mode full pipeline.
        # Writes checkpoints/model3_production_v3.pt + logs/full_v3_soup.json.
        # v1 (model3_production.pt) and v2 (model3_production_v2.pt safety net)
        # are preserved for the 3-way comparison after training.
        import torch as _torch
        from scripts.model3_training.training.soup import greedy_soup
        from scripts.model3_training.model3_config import PRODUCTION_V3_CHECKPOINT_NAME
        device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
        soup_result = greedy_soup(run_name='full_v3', device=device,
                                  production_name=PRODUCTION_V3_CHECKPOINT_NAME)
        print(f"\n[full] soup ingredients: {soup_result['final_ingredients']}  "
              f"field_f1={soup_result['final_field_f1']:.4f}  "
              f"overall_f1={soup_result['final_overall_f1']:.4f}")
        print(f"\n[full] best_epoch={result['best_epoch']}, "
              f"best_stop={result['best_stopping_metric']:.4f}, "
              f"halt={result['halt_reason']}")

    elif args.mode == 'evaluate':
        import torch
        from scripts.model3_training.architecture.model3_full import Model3
        from scripts.model3_training.data.model3_dataset import (
            M3ValDataset, load_and_split_csv,
        )
        from scripts.model3_training.evaluation.dual_stream_eval import dual_stream_evaluate
        from scripts.model3_training.model3_config import (
            CLASS_NAMES, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
        )
        from torch.utils.data import DataLoader

        if not args.checkpoint:
            print("--checkpoint <path> required for --mode evaluate")
            sys.exit(1)
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model3(n_classes=NUM_CLASSES, pretrained=False, use_lora=True).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        model.eval()

        splits = load_and_split_csv()
        results = {}
        for split_name in ('val', 'final_val'):
            if split_name not in splits:
                continue
            ds = M3ValDataset(items=splits[split_name])
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=(NUM_WORKERS > 0),
                            prefetch_factor=2 if NUM_WORKERS > 0 else None,
                            collate_fn=Trainer._collate)
            out = dual_stream_evaluate(model, dl, device, CLASS_NAMES)
            results[split_name] = out
            print(f"\n{split_name.upper()}:")
            print(f"  overall_f1 = {out['overall_f1']:.4f}")
            print(f"  field_f1   = {out['field_f1']:.4f} (n={out['n_field_val']})")
            print(f"  lab_f1     = {out['lab_f1']:.4f} (n={out['n_lab_val']})")
            print(f"  per-class field F1:")
            for c in CLASS_NAMES:
                print(f"    {c:<36} {out['per_class_field'][c]:.4f}")

        # Save
        out_path = ROOT / 'scripts' / 'model3_training' / 'logs' / \
                   f"evaluate_{ckpt_path.stem}.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
