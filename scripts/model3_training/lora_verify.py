"""3-epoch LoRA verification test (spec Part 8).

Non-negotiable gate before the full training run. Decision tree after 3 epochs:
  septoria field F1 > 0.70  -> PROCEED to full run
  0.60 <= f1 <= 0.70        -> DISCUSS with project lead (halt, report)
  f1 < 0.55                 -> STOP (LoRA rank=4 insufficient)

Same architecture, same data pipeline, same optimizer as the full run — only
difference is max_epochs=3 and run_name='lora_verify'.

Reads best septoria field F1 across all 3 epochs (not just epoch 3), since the
sep-target class may peak mid-run.

Writes decision to logs/lora_verify_decision.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.model3_training.model3_config import (
    LOG_DIR, LORA_VERIFY_EPOCHS,
    LORA_VERIFY_SEPTORIA_PROCEED,
    LORA_VERIFY_SEPTORIA_DISCUSS_LOW,
    LORA_VERIFY_SEPTORIA_STOP,
)
from scripts.model3_training.training.trainer import Trainer


SEPTORIA = 'tomato_septoria_leaf_spot'


def main():
    trainer = Trainer(
        run_name='lora_verify',
        max_epochs=LORA_VERIFY_EPOCHS,
        use_lora=True,
        cutmix_enabled=False,   # CutMix starts at epoch 3, won't fire in 3-epoch verify
        mixstyle_enabled=True,
    )
    result = trainer.run()
    epoch_logs = result['epoch_logs']

    # Extract septoria field F1 per epoch + best
    septoria_per_epoch = []
    for rec in epoch_logs:
        f = rec['per_class_field'].get(SEPTORIA, 0.0)
        septoria_per_epoch.append({
            'epoch': rec['epoch'],
            'septoria_field_f1': f,
            'overall_f1': rec['overall_f1'],
            'field_f1': rec['field_f1'],
        })
    best_septoria = max((e['septoria_field_f1'] for e in septoria_per_epoch), default=0.0)
    best_epoch = max(septoria_per_epoch, key=lambda e: e['septoria_field_f1'])['epoch'] \
        if septoria_per_epoch else 0

    if best_septoria > LORA_VERIFY_SEPTORIA_PROCEED:
        decision = 'PROCEED_TO_FULL_RUN'
        message = (f"LoRA rank=4 hypothesis CONFIRMED. "
                   f"septoria field F1 = {best_septoria:.4f} (best at epoch {best_epoch}) "
                   f"> {LORA_VERIFY_SEPTORIA_PROCEED}.")
    elif best_septoria >= LORA_VERIFY_SEPTORIA_DISCUSS_LOW:
        decision = 'DISCUSS_MARGINAL'
        message = (f"Marginal improvement. "
                   f"septoria field F1 = {best_septoria:.4f} in "
                   f"[{LORA_VERIFY_SEPTORIA_DISCUSS_LOW}, {LORA_VERIFY_SEPTORIA_PROCEED}]. "
                   f"Halt for project-lead decision before full run.")
    elif best_septoria < LORA_VERIFY_SEPTORIA_STOP:
        decision = 'STOP_FULL_RUN'
        message = (f"LoRA rank=4 insufficient. "
                   f"septoria field F1 = {best_septoria:.4f} < {LORA_VERIFY_SEPTORIA_STOP}. "
                   f"Full training must not start.")
    else:
        # 0.55 <= f1 < 0.60 — ambiguous zone, treat as DISCUSS
        decision = 'DISCUSS_MARGINAL'
        message = (f"Ambiguous. septoria field F1 = {best_septoria:.4f}. "
                   f"Halt for decision.")

    out = {
        'verdict': decision,
        'message': message,
        'best_septoria_field_f1': best_septoria,
        'best_epoch': best_epoch,
        'septoria_per_epoch': septoria_per_epoch,
        'halt_reason': result.get('halt_reason'),
        'best_stopping_metric': result.get('best_stopping_metric'),
        'full_log_path': result.get('log_path'),
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / 'lora_verify_decision.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n{'=' * 72}")
    print(f"LoRA VERIFICATION DECISION: {decision}")
    print(f"{'=' * 72}")
    print(message)
    print(f"\nSaved: {out_path.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
