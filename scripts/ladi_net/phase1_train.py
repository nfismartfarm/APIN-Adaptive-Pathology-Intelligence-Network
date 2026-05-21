"""
Phase 1 training: head-only training on frozen DINOv2-Base-Registers.

Trains: ABMIL + gated MLP fusion + SupCon projector.
Frozen: backbone (12 DINOv2 blocks, no LoRA).

Corresponds to Decisions 17 §17.6, 31 §31.6, 37.

Per Decision 17 §17.6:
- Epochs: 5 (extendable to 8 if attention gate doesn't pass)
- Batch size: 32 (doubled from Phase 2 because no LoRA = no backward through backbone)
- Optimizer: AdamW, single LR=5e-4, weight_decay=0.0
- Schedule: 2-epoch warmup + cosine anneal to 0.1× peak
- Loss: CE + supcon_weight(epoch) * SupCon    (NO CORAL — target doesn't exist yet)
- bf16 autocast, grad_clip_norm=1.0

Output: models/specialist/ladinet_phase1_heads.pt containing
        {epoch, abmil_state_dict, fusion_state_dict, supcon_projector_state_dict,
         optimizer_state_dict, scheduler_state_dict, rng_state, config_hash,
         final_val_sqrtn_macro_f1, generated_at}

This output is consumed by:
- Decision 33: compute_coral_target_abmil.py (runs after Phase 1 end)
- Phase 2 training script (warm-starts heads + refreshes CORAL target from it)
"""

from __future__ import annotations

import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

# Reproducibility — set BEFORE importing torch/numpy if possible
os.environ["PYTHONHASHSEED"] = "0"

# Allow `python scripts/ladi_net/phase1_train.py` invocation
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ladinet_config import (
    PROJECT_ROOT, TOMATO_CLASSES, CLASS_TO_IDX, NUM_CLASSES,
    RESOLUTION, PHASE1_BATCH_SIZE, PHASE1_NUM_EPOCHS, PHASE1_MAX_EPOCHS,
    LR_HEADS, WD_HEADS, GRAD_CLIP_NORM, WARMUP_EPOCHS, LR_COSINE_MIN_RATIO,
    SEED, NUM_WORKERS, PHASE1_HEADS_PT, PHASE1_CKPT_DIR, LOGS_DIR,
    STOPPING_WEIGHTS, CONFIG_HASH, phase1_supcon_weight,
    FALLBACK_MAX_ATTN_THRESHOLD, FALLBACK_ENTROPY_THRESHOLD,
)
from ladinet_model import LADINet
from ladinet_losses import supcon_loss, weighted_ce_loss
from ladinet_dataloader import (
    load_split_records, LadiNetDataset, ClassStratifiedBatchSampler,
    load_background_pool,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_phase1(model: LADINet, val_records, device: torch.device) -> dict:
    """Run model on field_val records; return per-class F1 + sqrt(N) macro."""
    from sklearn.metrics import f1_score

    model.eval()
    val_dataset = LadiNetDataset(val_records, training=False, background_pool=None)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=NUM_WORKERS)

    all_preds, all_labels = [], []
    fallback_fires_by_class = {c: 0 for c in TOMATO_CLASSES}
    fallback_total_by_class = {c: 0 for c in TOMATO_CLASSES}

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in val_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            out = model(x)
            preds = out["logits"].argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            fires = out["fallback_flag"].squeeze(-1).cpu().numpy()
            for i, label in enumerate(y.cpu().numpy()):
                cname = TOMATO_CLASSES[int(label)]
                fallback_total_by_class[cname] += 1
                if fires[i] > 0.5:
                    fallback_fires_by_class[cname] += 1

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    per_class_f1 = {
        c: float(f1_score((y_true == i).astype(int), (y_pred == i).astype(int),
                          zero_division=0))
        for i, c in enumerate(TOMATO_CLASSES)
    }
    simple_macro = float(np.mean(list(per_class_f1.values())))
    sqrtn_macro = sum(STOPPING_WEIGHTS[c] * per_class_f1[c] for c in TOMATO_CLASSES)
    fallback_rates = {
        c: (fallback_fires_by_class[c] / fallback_total_by_class[c])
        if fallback_total_by_class[c] > 0 else 0.0
        for c in TOMATO_CLASSES
    }

    return {
        "per_class_f1": per_class_f1,
        "sqrtn_macro_f1": float(sqrtn_macro),
        "simple_macro_f1": simple_macro,
        "fallback_fire_rate_by_class": fallback_rates,
        "n_val": int(y_true.size),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_phase1(num_epochs: int = PHASE1_NUM_EPOCHS, dry_run: bool = False):
    # Startup
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[FAIL] CUDA required. Phase 1 assumes the RTX 4060.")
        sys.exit(1)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE1_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"phase1_{run_id}.log"
    metrics_csv = LOGS_DIR / f"phase1_{run_id}.csv"

    # Open metrics CSV
    metrics_csv.write_text("epoch,train_loss,train_ce,train_supcon,supcon_w,"
                           "val_sqrtn_f1,val_simple_f1,val_fallback_rate_mean,"
                           "lr,elapsed_s\n", encoding="utf-8")

    print("=" * 72)
    print(f"LADI-Net Phase 1 training  |  config_hash={CONFIG_HASH}  |  run_id={run_id}")
    print(f"device={device.type} ({torch.cuda.get_device_name() if device.type=='cuda' else ''})")
    print(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}  seeds_set=True")
    print(f"num_epochs={num_epochs}  batch_size={PHASE1_BATCH_SIZE}  resolution={RESOLUTION}")
    print(f"optimizer=AdamW  lr={LR_HEADS}  wd={WD_HEADS}  grad_clip={GRAD_CLIP_NORM}")
    print(f"warmup_epochs={WARMUP_EPOCHS}  cosine_min_ratio={LR_COSINE_MIN_RATIO}")
    print(f"supcon weight ramp: ep0=0 -> ep1=0 -> ep2=0.10 -> ep3=0.20 -> ep4=0.30 (Decision 37)")
    print("=" * 72)

    # Model
    print("Loading model (backbone frozen, no LoRA) ...")
    model = LADINet(device=device, phase="phase1").to(device)
    n_train = sum(p.numel() for p in model.trainable_params())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {n_train:,} / total {n_total:,}")

    # Data
    print("Enumerating training records ...")
    train_records = load_split_records("train")
    val_records = load_split_records("field_val")
    print(f"  train records: {len(train_records)}  (field={sum(1 for r in train_records if r.is_field_photo)})")
    print(f"  field_val records: {len(val_records)}")

    # Per-class + per-type breakdown (provenance visible in training log)
    from collections import Counter
    type_counts = Counter(r.image_type for r in train_records)
    print(f"  type counts: LAB_OK={type_counts['LAB_OK']}  LAB_FLAGGED={type_counts['LAB_FLAGGED']}  "
          f"FIELD={type_counts['FIELD']}  RECOMPOSED={type_counts['RECOMPOSED']}")
    print("  per-class (total / field / lab+recomp):")
    for c in TOMATO_CLASSES:
        recs = [r for r in train_records if r.class_name == c]
        total = len(recs)
        field = sum(1 for r in recs if r.is_field_photo)
        recomp = sum(1 for r in recs if r.image_type == "RECOMPOSED")
        lab = total - field - recomp
        print(f"    {c:35s}: total={total:5d}  field={field:4d}  lab={lab:5d}  recomp={recomp:5d}")
    # Note: tomato_healthy has 0 recomp intentionally — static recomp pool was
    # generated for 5 disease classes only + 897 chilli_healthy (not used here).

    print("Loading background pool (preload to 392px RAM cache) ...")
    bg_pool = load_background_pool()
    print(f"  background pool size: {len(bg_pool)}")

    train_ds = LadiNetDataset(train_records, training=True, background_pool=bg_pool)
    val_ds = LadiNetDataset(val_records, training=False, background_pool=None)
    sampler = ClassStratifiedBatchSampler(train_records, phase="phase1", seed=SEED)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  train batches per epoch: {len(sampler)}  (bs={PHASE1_BATCH_SIZE})")

    # Optimizer
    head_params = model.head_params()
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": LR_HEADS, "weight_decay": WD_HEADS}]
    )

    # Scheduler: linear warmup 2 epochs -> cosine anneal to 0.1 × peak
    total_steps = len(sampler) * num_epochs
    warmup_steps = len(sampler) * WARMUP_EPOCHS
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine anneal from 1.0 to LR_COSINE_MIN_RATIO
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos_factor = 0.5 * (1 + np.cos(np.pi * progress))
        return LR_COSINE_MIN_RATIO + (1 - LR_COSINE_MIN_RATIO) * cos_factor
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    best_val_f1 = -1.0
    global_step = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        # ABMIL/fusion/supcon train() mode; backbone is frozen already so its state
        # doesn't matter, but put it to eval to be safe (no dropout etc.).
        model.backbone.eval()

        sampler.set_epoch(epoch)
        supcon_w = phase1_supcon_weight(epoch)

        epoch_loss_sum = 0.0
        epoch_ce_sum = 0.0
        epoch_supcon_sum = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            is_field = batch["is_field_photo"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                ce = weighted_ce_loss(out["logits"], y, is_field)
                if supcon_w > 0:
                    sc = supcon_loss(out["supcon_proj"].float(), y)
                else:
                    sc = torch.zeros((), device=device)
                total_loss = ce + supcon_w * sc

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            epoch_loss_sum += float(total_loss.item())
            epoch_ce_sum += float(ce.item())
            epoch_supcon_sum += float(sc.item())
            n_batches += 1
            global_step += 1

            if batch_idx % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  epoch={epoch} batch={batch_idx:4d}/{len(sampler)}  "
                      f"loss={total_loss.item():.4f}  ce={ce.item():.4f}  "
                      f"sc={sc.item():.4f}  lr={current_lr:.2e}")

            if dry_run and batch_idx >= 2:
                print("  [dry-run] exiting after 3 batches")
                break

        # Validation at end of epoch
        elapsed = time.time() - t0
        train_loss = epoch_loss_sum / max(1, n_batches)
        train_ce = epoch_ce_sum / max(1, n_batches)
        train_sc = epoch_supcon_sum / max(1, n_batches)
        val_metrics = evaluate_phase1(model, val_records, device)
        val_f1 = val_metrics["sqrtn_macro_f1"]
        val_simple = val_metrics["simple_macro_f1"]
        fallback_rate_mean = float(np.mean(list(val_metrics["fallback_fire_rate_by_class"].values())))

        print(f"\nEpoch {epoch}  complete  elapsed={elapsed:.1f}s")
        print(f"  train_loss={train_loss:.4f}  ce={train_ce:.4f}  supcon={train_sc:.4f}  w={supcon_w:.2f}")
        print(f"  val sqrtn_macro_f1={val_f1:.4f}  simple_macro_f1={val_simple:.4f}")
        print(f"  val per-class F1:")
        for c, f1 in val_metrics["per_class_f1"].items():
            print(f"    {c:35s}: {f1:.4f}")
        print(f"  val fallback_flag fire rates:")
        for c, r in val_metrics["fallback_fire_rate_by_class"].items():
            print(f"    {c:35s}: {r*100:.1f}%")

        with open(metrics_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss},{train_ce},{train_sc},{supcon_w},"
                    f"{val_f1},{val_simple},{fallback_rate_mean},"
                    f"{scheduler.get_last_lr()[0]},{elapsed}\n")

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt = {
                "epoch": epoch,
                "abmil_state_dict": model.abmil.state_dict(),
                "fusion_state_dict": model.fusion.state_dict(),
                "supcon_projector_state_dict": model.supcon.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": {
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state(),
                    "python_random": random.getstate(),
                },
                "config_hash": CONFIG_HASH,
                "val_sqrtn_macro_f1": val_f1,
                "val_per_class_f1": val_metrics["per_class_f1"],
                "generated_at": datetime.datetime.now().isoformat(),
                "run_id": run_id,
            }
            torch.save(ckpt, PHASE1_HEADS_PT)
            epoch_ckpt = PHASE1_CKPT_DIR / f"phase1_epoch{epoch:02d}_f1{val_f1:.4f}.pt"
            torch.save(ckpt, epoch_ckpt)
            print(f"  -> New best Phase 1 checkpoint: {PHASE1_HEADS_PT.name}  (f1={val_f1:.4f})")

        if dry_run:
            print("[dry-run] exiting after epoch 0")
            break

    print("\n" + "=" * 72)
    print(f"Phase 1 complete. Best val sqrtn_macro_f1 = {best_val_f1:.4f}")
    print(f"Checkpoint: {PHASE1_HEADS_PT}")
    print("\nNEXT STEPS (per Decision 33):")
    print("  1. Inspect 20 attention maps (Decision 17 sec 17.7) -- pass criterion >=14/20")
    print(f"  2. Run: python scripts/ladi_net/compute_coral_target_abmil.py")
    print("     This produces data/specialist/model3/coral_target_cov.pt with ABMIL provenance.")
    print("  3. Then Phase 2 training may begin.")
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_epochs", type=int, default=PHASE1_NUM_EPOCHS)
    ap.add_argument("--dry_run", action="store_true",
                    help="Run 3 batches + 1 epoch only; skip most compute.")
    args = ap.parse_args()
    train_phase1(num_epochs=args.num_epochs, dry_run=args.dry_run)
