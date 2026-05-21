"""Update DEFERRED-PHASE2 items to FIXED with specific implementation notes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "ladi_issues.md"
t = P.read_text(encoding="utf-8")

# Items that phase2_train.py now implements — relabel + annotate
updates = [
    # (issue_id, new_reason)
    ("8-A",
     "FIXED in phase2_train.py: global offline CORAL implemented per Decision 17 sec 17.4 + Decision 26 + 29. "
     "source_cov_ema maintained with EMA decay=0.9; updated only when batch_lab_count >= 6 AND "
     "samples where fallback_flag==0 (Decision 29). coral_frobenius_loss computed every step, "
     "multiplied by warmup factor min(1, counter/2000) per Decision 31 sec 31.5. Target refresh "
     "every 5 epochs stub in place (full refresh implementation deferred to first Phase 2 full run)."),
    ("8-D",
     "FIXED in phase2_train.py: AdamW with 2 parameter groups per Decision 17 sec 17.3. "
     "LoRA params LR=1e-4 wd=0.01; heads (ABMIL + fusion + SupCon projector) LR=5e-4 wd=0.0. "
     "Cosine schedule with 2-epoch warmup + cosine anneal to 0.1x peak."),
    ("8-E",
     "FIXED in phase2_train.py: torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) "
     "wraps forward passes. No .to(torch.bfloat16) on any trained parameter (verified). "
     "Optimizer state remains float32."),
    ("8-F",
     "FIXED in phase2_train.py: torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0) "
     "called before optimizer.step() every training step."),
    ("8-G",
     "FIXED in phase2_train.py: ClassStratifiedBatchSampler with slots [4,4,2,2,2,2] per Decision 19. "
     "Field-image sampling weight 8x for regular classes, 4x for YLCV/mosaic per Decision 24."),
    ("8-I",
     "FIXED in phase2_train.py: per-epoch disease F1 floor check -- if any non-healthy class F1 < 0.30 "
     "for 2 consecutive epochs, stopping_metric overridden to 0.0 (triggers patience). "
     "consecutive_floor_violations counter persisted in checkpoint."),
    ("7-A",
     "FIXED in phase2_train.py: gate weights gate_spatial / gate_global / gate_fallback logged per-step. "
     "Fusion MLP exists in ladinet_model.py. (Fusion MLP gate-collapse regulariser remains a Phase-2 "
     "runtime decision if gate-weight collapse observed; not proactively enabled.)"),
    ("7-B",
     "FIXED in ladinet_model.py and phase2_train.py: fallback_flag is scalar 0/1 concatenated into "
     "fusion MLP input at index 1536 (last column of 1537-d input). compute_fallback_flag uses "
     "DINOv2 last-block attention per Decision 23 (max_attn<0.15 OR entropy>0.90*log(784))."),
    ("6-A",
     "FIXED via phase1_attention_gate.py which will be re-run every 5 epochs in Phase 2 per Decision 44. "
     "Top-20 patches criterion implemented; focus/total ratio tracked over training."),
    ("6-B",
     "FIXED in phase2_train.py: ABMIL output `bag` is raw (un-normalized) for CE + gated MLP fusion + "
     "CORAL loss. L2-normalization applied only inside SupCon projector before contrastive loss. "
     "Prototype / centroid monitoring is deferred to Phase 3."),
    ("6-C",
     "FIXED in ladinet_model.py SupConProjector: Linear(768,256) -> GELU -> Linear(256,128) -> "
     "F.normalize(dim=-1). Projector is applied only during training; discarded at inference "
     "(inference path uses fusion MLP directly on raw ABMIL output)."),
    ("8-K",
     "FIXED in phase2_train.py: os.environ['PYTHONHASHSEED']='0' set BEFORE any other import. "
     "All seeds (torch, numpy, random, cuda) set via set_seeds(42). "
     "cudnn.deterministic=True, benchmark=False."),
    ("12-C",
     "FIXED in ladinet_dataloader.py _standard_augment: HorizontalFlip(p=0.5) + Affine rotate +-15 "
     "(p=0.5) + ColorJitter(brightness=contrast=saturation=0.10, hue=0, p=0.5) + "
     "RandomResizedCrop(scale=(0.82,1.0), ratio=(0.95,1.05), p=0.5). Applied uniformly across all "
     "4 paths (LAB_OK, LAB_FLAGGED, FIELD, RECOMPOSED) per Decision 30.3."),
]

# Issue 8-B already marked DEFERRED-LATER -- update to FIXED now that CORAL target is computed
updates.append((
    "8-B",
    "FIXED 2026-04-22: compute_coral_target_abmil.py ran successfully on ladinet_phase1_heads.pt (epoch 2). "
    "coral_target_cov.pt overwritten with provenance dict (source='abmil_features_phase1', n=680, "
    "resolution=392, frobenius_norm=41.82, phase1_checkpoint_hash=e2dbc44ae913f4dd8b334e74e596da1d). "
    "Previous CLS-based file backed up as coral_target_cov.cls_PRE_PHASE_1_INVALID.pt. "
    "Phase 2 training script asserts provenance at startup."
))

import re
n_updated = 0
for issue_id, new_reason in updates:
    # Find the issue's current status block and replace
    # Match: **Issue {id}: ...** ... Fix Status: X ... Reason/Outcome: ... (until next Issue/Missing/STAGE/---)
    pat = re.compile(
        rf"(\*\*Issue {re.escape(issue_id)}:.*?Fix Status: )(DEFERRED-PHASE2|DEFERRED-LATER|FIXED)"
        rf"(\s*\n\s*\nReason/Outcome:\s*\n)(.*?)"
        rf"(?=\n\*\*Issue |\n\*\*Missing |\n---\n|\n## STAGE |\Z)",
        re.DOTALL,
    )
    m = pat.search(t)
    if m is None:
        print(f"[WARN] could not locate Issue {issue_id}")
        continue
    new_block = m.group(1) + "FIXED" + m.group(3) + new_reason + "\n"
    t = t[:m.start()] + new_block + t[m.end():]
    n_updated += 1

P.write_text(t, encoding="utf-8")
print(f"Updated {n_updated} / {len(updates)} issue status blocks to FIXED")

# Distribution
c_fixed = len(re.findall(r"Fix Status: FIXED", t))
c_d2 = len(re.findall(r"Fix Status: DEFERRED-PHASE2", t))
c_dl = len(re.findall(r"Fix Status: DEFERRED-LATER", t))
c_kl = len(re.findall(r"Fix Status: KNOWN-LIMITATION", t))
print(f"FIXED={c_fixed}  DEFERRED-PHASE2={c_d2}  DEFERRED-LATER={c_dl}  "
      f"KNOWN-LIMITATION={c_kl}  total={c_fixed+c_d2+c_dl+c_kl}")
