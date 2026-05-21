"""
Round-1-PV fix: relabel 6 inflated FIXED items to DEFERRED-PHASE2.
Each has a documentation-only fix awaiting Phase 2 code.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
ISSUES = ROOT / "ladi_issues.md"

# (anchor_text, new_status, new_reason) — we match on a UNIQUE start-of-reason phrase
# from the current block, so we can safely string-replace.
RELABELS = [
    # Issue 4-B — bbox clamping
    (
        "Fix Status: FIXED\n\nReason/Outcome:\nTrivial one-line fix documented in Decision 17 (pre-phase-1 code patterns). The bounding-box clamp code pattern is: `x1 = max(0, x1 - pad); y1 = max(0, y1 - pad); x2 = min(H-1, x2 + pad); y2 = min(W-1, y2 + pad)` where H=W=392 for our resolution. The Phase 2 training script and the inference pipeline both use this pattern verbatim. No runnable code change needed until the Phase 2 training script is written, at which point the pattern is directly inserted.",
        "Fix Status: DEFERRED-PHASE2",
        "Reason/Outcome:\nDocumented in Decision 17 (attention-extraction code pattern): `x1=max(0, x1-pad); y1=max(0, y1-pad); x2=min(H-1, x2+pad); y2=min(W-1, y2+pad)` with H=W=392. This is a code pattern specification, not an applied code fix — the Phase 2 training script that contains the attention extraction logic does not yet exist. Phase 2 script implementation must use this pattern verbatim."
    ),

    # Issue 8-K — PYTHONHASHSEED + step counter
    (
        "Fix Status: FIXED\n\nReason/Outcome:\nDecision 17 documents: Phase 2 training entrypoint script MUST set `os.environ['PYTHONHASHSEED'] = '0'` BEFORE any `import` statement (has no effect once the interpreter has started). Additionally, the recomposer's RNG seed computation is enhanced with a step counter: `key = (self.seed * 1_000_003 + epoch * 31 + step_within_epoch * 17 + hash(image_path)) & 0xFFFFFFFF`. This change is queued for the Phase 2 recomposer call site, not the recomposer class itself (which stays stable).",
        "Fix Status: DEFERRED-PHASE2",
        "Reason/Outcome:\nSpecified in Decision 17 but NOT yet applied. The Phase 2 training entrypoint script MUST set `os.environ['PYTHONHASHSEED'] = '0'` BEFORE any `import`. The recomposer call site in the Phase 2 DataLoader passes a step counter so the RNG seed formula becomes `(self.seed * 1_000_003 + epoch * 31 + step_within_epoch * 17 + hash(image_path)) & 0xFFFFFFFF`. Neither of these changes is in any current script; both are pending Phase 2 implementation."
    ),

    # Issue 12-C — standard augmentation
    (
        "Fix Status: FIXED\n\nReason/Outcome:\nDecision 17 documents the standard augmentation chain for Phase 2 (applied AFTER recomposition, AFTER LAB-CLAHE, AFTER resize to 392px — all augmentations operate on the final 392×392 image): `A.HorizontalFlip(p=0.5)` + `A.Affine(rotate=(-15,15), p=0.5)` (no shear, no scale — preserves disease pattern structure) + `A.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.0, p=0.5)` (conservative; hue=0 prevents disease colour-signature shifts) + `A.RandomResizedCrop(392, 392, scale=(0.82, 1.0), ratio=(0.95, 1.05), p=0.5)` (mild scale jitter, adds scale invariance). The Phase 2 training DataLoader chains these via albumentations in the stated order.",
        "Fix Status: DEFERRED-PHASE2",
        "Reason/Outcome:\nSpecification in Decision 17 (pipeline order, after recomposition+CLAHE+resize-to-392px): `HorizontalFlip(p=0.5)` → `Affine(rotate=±15°, p=0.5)` → `ColorJitter(b/c/s=±10%, hue=0, p=0.5)` → `RandomResizedCrop(392, scale=(0.82,1.0), ratio=(0.95,1.05), p=0.5)`. This is a pipeline spec only; no DataLoader exists yet. Phase 2 training-script DataLoader must chain these albumentations transforms in the stated order."
    ),

    # Cross-check weight decay
    (
        "Fix Status: FIXED\n\nReason/Outcome:\nDocumented in Decision 17. AdamW weight_decay=0.01 for LoRA parameters only (no weight decay on ABMIL/gated MLP/SupCon projector — they're trained from scratch and weight decay on small new parameter sets tends to over-regularise). Bias and LayerNorm parameters excluded via parameter-group filtering.",
        "Fix Status: DEFERRED-PHASE2",
        "Reason/Outcome:\nSpecified in Decision 17 §17.3 (optimizer + scheduler + precision). AdamW weight_decay=0.01 on LoRA parameter group; weight_decay=0.0 on from-scratch heads; bias/LayerNorm parameters excluded via parameter-group filtering. Specification only — the Phase 2 training script that constructs these parameter groups does not yet exist."
    ),

    # Cross-check cosine schedule warmup
    (
        "Fix Status: FIXED\n\nReason/Outcome:\nDocumented in Decision 17. 2-epoch linear warmup from 0 → peak LR, then cosine anneal for the remaining epochs to 10% of peak LR. For Phase 2 (25 epochs × ~2000 steps/epoch = 50,000 steps): warmup = 4,000 steps ≈ 2 epochs. Implemented via torch.optim.lr_scheduler.SequentialLR chaining LinearLR + CosineAnnealingLR.",
        "Fix Status: DEFERRED-PHASE2",
        "Reason/Outcome:\nSpecified in Decision 17 §17.3. 2-epoch linear warmup from 0 → peak LR, then cosine anneal for the remaining epochs down to 10% of peak. For Phase 2 at ~2000 steps/epoch × 25 epochs = 50,000 steps: warmup = 4,000 steps ≈ 2 epochs. Implemented via `torch.optim.lr_scheduler.SequentialLR` chaining `LinearLR` + `CosineAnnealingLR`. Specification only — scheduler is constructed in the as-yet-unwritten Phase 2 training script."
    ),

    # Cross-check k-means random_state
    (
        "Fix Status: FIXED\n\nReason/Outcome:\nDocumented in Decision 17. All k-means calls in the prototype-bank construction (Phase 3) and any other clustering operation use `random_state=42`. Phase 3 script enforces this via a shared config constant K_MEANS_SEED=42.",
        "Fix Status: DEFERRED-PHASE2",
        "Reason/Outcome:\nSpecified in Decision 17: all k-means calls in the prototype-bank construction (Phase 3) and any other clustering operation use `random_state=42` via a shared config constant `K_MEANS_SEED=42`. Specification only — the Phase 3 prototype-bank construction script that enforces this constant does not yet exist."
    ),
]


def main():
    text = ISSUES.read_text(encoding="utf-8")
    n = 0
    for old_block, new_status, new_reason in RELABELS:
        # Full replacement: status line + blank line + reason line
        full_old = old_block
        full_new = new_status + "\n\n" + new_reason
        if full_old in text:
            text = text.replace(full_old, full_new, 1)
            n += 1
        else:
            print(f"[WARN] block not found: {new_status}")
    ISSUES.write_text(text, encoding="utf-8")
    print(f"Relabeled {n} / {len(RELABELS)} items")
    # Post-relabel distribution
    import re
    c_fixed = len(re.findall(r"Fix Status: FIXED", text))
    c_def2 = len(re.findall(r"Fix Status: DEFERRED-PHASE2", text))
    c_defl = len(re.findall(r"Fix Status: DEFERRED-LATER", text))
    c_kl = len(re.findall(r"Fix Status: KNOWN-LIMITATION", text))
    print(f"FIXED={c_fixed}  DEFERRED-PHASE2={c_def2}  DEFERRED-LATER={c_defl}  KNOWN-LIMITATION={c_kl}  total={c_fixed+c_def2+c_defl+c_kl}")


if __name__ == "__main__":
    main()
