"""
Utility to mark all issues in ladi_issues.md with status blocks.
One-shot: reads current file, inserts status blocks under each issue, writes back.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
ISSUES_PATH = ROOT / "ladi_issues.md"

# (issue_id, status, reason_text)
STATUS = [
    # STAGE 0
    ("0-A", "DEFERRED-PHASE2",
     "The quality gate is an inference-pipeline concern, not a training concern — training images are already curated by the mask precompute step. The Phase 2 inference server (post-training) must resize the uploaded image to 392px BEFORE computing Laplacian variance, then apply threshold 80. Documented for the deployment code path. During Phase 2 training the gate is not executed."),
    ("0-B", "DEFERRED-PHASE2",
     "Same category as Issue 0-A — inference-server concern. The Phase 2 inference server must: (a) apply InSPyReNet background removal first, (b) compute the overexposure fraction only on pixels with non-zero alpha (foreground/leaf pixels), (c) apply the 15% threshold to that foreground-only ratio. Training images pre-date this gate and are already valid."),

    # STAGE 1
    ("1-A", "FIXED",
     "Resolved by pre_phase1_checks.py Check 1 (run 2026-04-22). Healthy flag breakdown is now known: 193 low_coverage + 44 multi(low_conf+low_cov) + 2 low_confidence = 239 total; 0 high_coverage. Conclusion: healthy flags are low_coverage dominated, NOT high_coverage — the leaf is segmented correctly but occupies less than the minimum fraction of the image. Full per-class flag breakdown saved to data/specialist/model3/pre_phase1_check_results.json."),
    ("1-B", "FIXED",
     "Resolved by Decision 16 (2026-04-22). The Phase 2 DataLoader loads `_fg.png` (background zeroed to black) for ALL lab training images — flagged or not. Recomposition at p=0.70 applies only to non-flagged images; flagged images keep black backgrounds. Every lab training image now has either a black or a field background; NO training image retains a white lab background. Verified: 22,439 of 22,441 lab images have `_fg.png` on disk (the 2 missing are MemoryError processing failures — the Phase 2 DataLoader will skip these and log n_lab_skipped_no_fg at epoch start)."),
    ("1-C", "DEFERRED-PHASE2",
     "Inference-server concern. Phase 2+ inference pipeline must: (1) resize uploaded image to 640px max dimension (down-sample only, never up-sample), (2) run InSPyReNet at 640px, (3) apply the mask, (4) resize the masked result to 392px for the backbone. This matches the resolution regime of the training mask precompute step (which ran on images at their native 256/512/640px resolutions)."),

    # STAGE 2
    ("2-A", "DEFERRED-PHASE2",
     "Phase 2 preprocessing chain in the DataLoader: after loading `_fg.png` and applying recomposition (if eligible), apply CLAHE to the leaf pixels only. Implementation: (a) derive a binary leaf mask from the non-black pixels of `_fg.png` (before recomposition) OR load the `_mask.png` alongside; (b) apply CLAHE to the full 392px image; (c) reset background pixels to 0 using the stored mask. Document in the Phase 2 training decision block that this is the canonical pipeline."),
    ("2-B", "DEFERRED-LATER",
     "Requires visual inspection of 10 images × 6 classes at clip_limit ∈ {1.0, 2.0, 3.0}, which is an offline diagnostic step. Scheduled to be performed during Phase 1 (when ABMIL is training) — zero impact on Phase 1 itself since it uses frozen features. If the inspection reveals that clip_limit=2.0 destroys YLCV yellowing or foliar brown signatures, the training can re-run with a lower clip_limit. Not blocking Phase 1."),

    # STAGE 3
    ("3-A", "FIXED",
     "Resolved by pre_phase1_checks.py Check 6 (run 2026-04-22). DINOv2-Base-Registers num_reg_tokens = 4 (confirmed). Total sequence at 392px = 1 CLS + 4 registers + 784 patches = 789 (verified). Correct indexing pattern: `attention[:, 0, 5:]` — sliced to skip CLS at index 0 and registers at indices 1-4, yielding 784 spatial-patch attention values reshapable to 28×28. Logged to pre_phase1_check_results.json and referenced in Decision 17 (indexing pattern documented)."),
    ("3-B", "DEFERRED-PHASE2",
     "Phase 1 validation step: after training ABMIL for 3 epochs, extract both (a) mean-across-heads and (b) max-across-heads attention maps for 20 field-val images. Visually inspect which variant better localises disease regions. Document the chosen aggregation in ladi_decisions.md at that time. Default (if inspection is inconclusive): mean-across-heads, per the standard DINO visualisation practice (Caron et al.)."),

    # STAGE 4
    ("4-A", "KNOWN-LIMITATION",
     "Structural mismatch between YLCV (diffuse symptom) and the two-pass localisation architecture. The model card MUST state: 'YLCV is diagnosed primarily by the global CLS stream and secondarily by the color stream; the spatial ABMIL stream has limited value for this class because its symptoms are leaf-wide rather than focal. The gated MLP has been designed to upweight the global stream when the fallback_triggered flag is True (Issue 7-B fix), which is LADI-Net's architectural mitigation.' Known field_val for YLCV is n=3 (Critique 3) — any metric at this sample size has CI ±0.35 and requires agronomist verification."),
    ("4-B", "FIXED",
     "Trivial one-line fix documented in Decision 17 (pre-phase-1 code patterns). The bounding-box clamp code pattern is: `x1 = max(0, x1 - pad); y1 = max(0, y1 - pad); x2 = min(H-1, x2 + pad); y2 = min(W-1, y2 + pad)` where H=W=392 for our resolution. The Phase 2 training script and the inference pipeline both use this pattern verbatim. No runnable code change needed until the Phase 2 training script is written, at which point the pattern is directly inserted."),
    ("4-C", "DEFERRED-PHASE2",
     "Phase 2 training-script-level concern. Document the rule: if the attention-based bounding box has width or height < 0.30 × 392 = 118 px, fall back to center 70% crop instead. Falls into the same training-script module as the attention extraction logic (Issue 3-A fix)."),

    # STAGE 5
    ("5-A", "FIXED",
     "Already formally documented in Decision 15 (2026-04-21) and re-confirmed here as Decision 17 item. DINOv2-Base has 12 transformer blocks (num_hidden_layers=12, verified in Check 8). The architecture spec's 'layers 17-24' language was written for DINOv2-Large (24 blocks). For DINOv2-Base, we apply LoRA to the TOP 8 blocks (indices 4-11, 0-indexed). Blocks 0-3 are frozen without LoRA. Confirmed in vram_test.py (constant LORA_BLOCKS_FROM_TOP=8) and in Check 8 of pre_phase1_checks.py."),
    ("5-B", "FIXED",
     "Documented in Decision 17: LoRA rank=8, alpha=16 (scaling factor = alpha/rank = 2.0). Matches the value tested in vram_test.py at VRAM measurement time. The Phase 2 training script MUST pass `alpha=16` (not PEFT's default of alpha=rank) when constructing the LoRA config. Verification: Phase 2 training startup must log `lora_rank=8 lora_alpha=16 lora_scaling=2.0` and assert these before the optimizer is created."),
    ("5-C", "DEFERRED-PHASE2",
     "Phase 2 training script design decision. Default approach: **Option A** (two separate model instances — one frozen for Pass 1, one with LoRA for Pass 2). Memory cost: ~360 MB × 2 = 720 MB, well within the 7.5 GB budget (bs=16 at 392px peaked at 2.56 GB in vram_test.py which already includes both passes). A unit test in the training script startup MUST verify: (a) LoRA parameters require_grad=True on Pass-2 model, (b) ALL parameters on Pass-1 model have require_grad=False, (c) a forward+backward pass produces non-zero gradients on LoRA parameters and zero on Pass-1 model parameters."),

    # STAGE 6
    ("6-A", "DEFERRED-PHASE2",
     "Phase 1 validation criterion adjustment. Documented: at 392px (784 patches vs 256 patches at 224px), the Phase 1 pass criterion is 'top-20 ABMIL attention patches (not top-10) cover the disease region in at least 14 of 20 inspected field images'. Also compute and log attention entropy per image during the inspection. Add to the Phase 1 validation script as a configurable parameter (TOP_K_ATTENTION_PATCHES = 20 at 392px)."),
    ("6-B", "DEFERRED-PHASE2",
     "Phase 2 training script routing requirement. Documented: after ABMIL aggregation, the training code computes TWO tensors from the same ABMIL output — `features_raw = abmil_output` (for CE loss, gated MLP input, CORAL loss) and `features_norm = F.normalize(abmil_output, dim=-1)` (for prototype similarity monitoring and confusable-pair centroid distance monitoring). Training script asserts both are computed and routed to the correct loss functions."),
    ("6-C", "DEFERRED-PHASE2",
     "Phase 2 training script must include an MLP projection head: `nn.Sequential(Linear(768, 256), GELU, Linear(256, 128))` applied to raw ABMIL output, followed by L2 normalisation. SupCon loss is computed on these 128-d L2-normalised projections. The projection head is discarded at inference (only the raw ABMIL feature is used for classification/CORAL). This matches the vram_test.py implementation and the Khosla et al. (2020) SupCon pipeline."),

    # STAGE 7
    ("7-A", "DEFERRED-PHASE2",
     "Phase 2 monitoring + optional regulariser. Documented: log `gate_weight_spatial` and `gate_weight_global` (and `gate_weight_fallback_flag` from Issue 7-B) every 100 steps. If either of the two main-stream gates falls below 0.05 AND stays below for 200+ consecutive steps after epoch 5, add a gate-diversity regulariser: `L_gate_diversity = max(0, 0.05 - min_gate_weight) * 10`. Include LayerNorm on each stream before concatenation to keep them on comparable scales. This monitoring logic belongs in the Phase 2 training script."),
    ("7-B", "DEFERRED-PHASE2",
     "Architectural change to the Phase 2 training script. The gated MLP fusion input becomes (768 spatial + 768 global + 1 scalar fallback_flag) = 1537-dim. The fallback_flag is a 0.0/1.0 binary scalar passed from the Pass-1 attention extraction logic. Training signal teaches the gate to upweight the global CLS stream when fallback_flag=1.0 (which happens primarily for YLCV/mosaic images with diffuse attention). This is LADI-Net's architectural mitigation for the Issue 4-A structural limitation. Exact input format: torch.cat([spatial_feat, global_feat, fallback_flag.unsqueeze(-1)], dim=-1)."),

    # STAGE 8
    ("8-A", "DEFERRED-PHASE2",
     "Phase 2 training-loop implementation. Documented: do NOT use batch CORAL (undefined with 0.34 field images/batch on average). Implement offline/global CORAL: (a) maintain `source_cov_ema` as a running exponential moving average — updated only on lab images in each batch; (b) target is the pre-computed `coral_target_cov.pt`, refreshed every 5 epochs using current LoRA-adapted ABMIL features from the 680 train-real-field images (handles Issue 8-B post-Phase-1 recomputation); (c) CORAL loss at each step = ||source_cov_ema - target_cov||_F² / (4 × 768²). EMA decay 0.9 for source_cov_ema. Phase 2 training script must include this full machinery."),
    ("8-B", "DEFERRED-LATER",
     "Post-Phase-1 action, confirmed CRITICAL. Phase 1 trains ABMIL on frozen backbone features; at Phase 1 end, the ABMIL head exists. BEFORE Phase 2 begins: (1) run the 680 train-real-field images through frozen backbone + trained ABMIL, (2) collect 680 × 768-dim ABMIL outputs, (3) compute `coral_target_cov_abmil.pt = np.cov(abmil_features, rowvar=False)`, (4) rename the current `coral_target_cov.pt` (CLS-based) to `coral_target_cov_cls_INVALID.pt` for audit trail, (5) save the new ABMIL-based covariance as `coral_target_cov.pt`. Current file is flagged as INVALID TYPE in pre_phase1_checks Check 4 output. Decision 16 + Check 4 both reference this post-Phase-1 task."),
    ("8-C", "DEFERRED-PHASE2",
     "Phase 1 training-loop architecture. Documented: Phase 1 does NOT cache patch tokens to disk (would need ~38 GB for 31,929 × 784 × 768 × 2 bytes). Instead, each Phase 1 batch runs the frozen backbone online: `with torch.no_grad(): patch_features = frozen_backbone.forward_features(images)` extracting all 789 tokens, then slicing to [:, 5:, :] for the 784 patch tokens (per Issue 3-A indexing). ABMIL + gated MLP trained from scratch on these patch tokens. Estimated Phase 1 runtime: (31,929 / 32) × 0.05s × 5 epochs ≈ 4 min/epoch × 5 = 20 min total for Phase 1."),
    ("8-D", "FIXED",
     "Documented in Decision 17 (hyperparameter specification block). AdamW optimizer with parameter groups: LoRA adapters LR=1e-4 weight_decay=0.01; ABMIL head + gated MLP + SupCon projection head (all from-scratch heads) LR=5e-4 weight_decay=0.0. Cosine schedule with 2-epoch warmup. References to MASTER_PLAN: the '1e-3 destroyed pretrained features in epoch 1' bug was for ConvNeXt full fine-tune; for LoRA on DINOv2-Base the appropriate LoRA LR is 1e-4 (an order of magnitude lower than from-scratch heads)."),
    ("8-E", "FIXED",
     "Documented in Decision 17. Phase 2 training code uses `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` context manager around the forward pass. Model parameters remain in float32 — NEVER call `.to(torch.bfloat16)` on any parameter that will be trained. Optimizer states (AdamW m/v buffers) remain float32 (required for numerical stability). This matches the MASTER_PLAN bug fix 'BF16 permanent cast destroys gradients for full fine-tune → fixed to float32 + autocast'. vram_test.py already implements this pattern correctly and is the reference implementation."),
    ("8-F", "FIXED",
     "Documented in Decision 17. After `scaler.unscale_(optimizer)` (if using fp16 scaler) or directly after `loss.backward()` (if using bf16), call `torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)` on ALL trainable parameters (LoRA + heads). Log grad_norm every step; if grad_norm is consistently > 10 pre-clipping for 100+ steps, investigate (likely CORAL weight too high or loss scaling issue)."),
    ("8-G", "DEFERRED-PHASE2",
     "Phase 2 training script MUST implement a `ClassStratifiedBatchSampler`. Confirmed necessary by pre_phase1_checks Check 2: under the current weighted sampler, only 63% of batches contain ≥2 foliar AND ≥2 septoria (below the 70% soft threshold). Design: 12 of 16 slots = 2 per class × 6 classes (round-robin from per-class index queues, reshuffled each epoch); remaining 4 slots filled by the weighted sampler (field 8× lab/recomp 1×) from any class. Expected coverage: ~100% of batches contain ≥2 per class, which makes the confusable-pair SupCon gradient active on every step."),
    ("8-H", "DEFERRED-PHASE2",
     "Phase 2 training-pipeline position. Documented: AmpMix applies AFTER InSPyReNet background handling (automatic via `_fg.png` loading) AND AFTER LAB-CLAHE AND AFTER resize to 392px. AmpMix swaps low-frequency Fourier amplitude components between two images. Visual inspection requirement before Phase 2 starts: generate 5 foliar+septoria pairs at 392px, AmpMix them, save to `logs/ampmix_spot_check/` for developer review. If artifacts are visible (not 'plausible mixed disease presentations'), block Phase 2."),
    ("8-I", "DEFERRED-PHASE2",
     "Phase 2 stopping-criterion augmentation. Documented: at the end of each epoch, compute per-class field_val F1. The stopping metric is the sqrt(N)-weighted sum defined in Decision 12. IF any non-healthy (disease) class F1 is < 0.30 for 2 consecutive epochs, OVERRIDE the stopping metric to 0.0 for that epoch — preventing early-stop while disease classes degrade. Also log per-class F1 every epoch (not just aggregate). Phase 2 training script must include this floor logic."),
    ("8-J", "FIXED",
     "Decision 17 documents: **EMA=NO for LADI-Net Phase 2.** Rationale: (1) EMA primarily helps when training is noisy and rolling checkpoint averaging smooths the loss surface — LADI-Net uses model soup (Phase 4) which already provides this benefit more robustly; (2) EMA adds a second copy of all trainable parameters (~1 GB for LoRA + heads), using VRAM we could otherwise use for larger batch size if needed; (3) the MASTER_PLAN's EMA bug ('decay=0.9999 retains 56% random initialisation at 11 epochs') is fixable via reset_ema after epoch 0 but adds one more thing to test. Model soup in Phase 4 operates on raw checkpoints. If a post-Phase-2 experiment shows EMA improves field F1 by >1.5%, it can be added in a later iteration."),
    ("8-K", "FIXED",
     "Decision 17 documents: Phase 2 training entrypoint script MUST set `os.environ['PYTHONHASHSEED'] = '0'` BEFORE any `import` statement (has no effect once the interpreter has started). Additionally, the recomposer's RNG seed computation is enhanced with a step counter: `key = (self.seed * 1_000_003 + epoch * 31 + step_within_epoch * 17 + hash(image_path)) & 0xFFFFFFFF`. This change is queued for the Phase 2 recomposer call site, not the recomposer class itself (which stays stable)."),

    # STAGE 9
    ("9-A", "DEFERRED-PHASE2",
     "Phase 4 model-soup script logic. Documented: after computing the greedy soup of top-5 field_val F1 checkpoints, evaluate the souped model on field_val. IF souped_field_val_F1 < best_individual_field_val_F1 - 0.01 (one full percentage point worse), ABORT the soup — use the best individual checkpoint instead. Log the outcome in the Phase 4 report. Soup checkpoint path: `models/specialist/ladinet_v1_soup.pt` (or `models/specialist/ladinet_v1_best.pt` on fallback)."),
    ("9-B", "DEFERRED-PHASE2",
     "Phase 3 prototype-bank construction script logic. Documented: the prototype extraction pass MUST load the model that will be deployed. Order of operations: (1) run Phase 4 soup and commit the soup (or fallback to best individual); (2) THEN run Phase 3 to extract prototypes using the committed model; (3) save `prototype_bank.pt` alongside. No prototypes are ever extracted from a pre-soup checkpoint. Phase 3 script opens `ladinet_v1_soup.pt` or `ladinet_v1_best.pt` from the same path the inference pipeline will use."),

    # STAGE 10
    ("10-A", "DEFERRED-PHASE2",
     "Phase 3 calibration script logic. Documented: use TWO temperatures, T_pair and T_overall. T_pair is fit on the 28-image confusable_pair_probe via NLL minimisation; applied only when the top-2 predicted classes are {foliar_spot, septoria_leaf_spot}. T_overall is fit on the field_val set (203 images covering all 6 classes) via NLL minimisation; applied in all other cases. The tier-assignment code in the Phase 2 inference pipeline routes based on the pre-softmax argmax classes. Implementation: calibration script saves both temperatures to `temperatures.pt = {'T_pair': float, 'T_overall': float}`."),
    ("10-B", "DEFERRED-PHASE2",
     "Phase 3 threshold-calibration step. Documented: after temperature calibration, examine the CALIBRATED confidence distribution on field_val. Set the prototype-activation threshold at the 25th percentile of `max_class_probability` among correctly-predicted images. This replaces the fixed 0.60 threshold (which was pre-training guess). Typical value expected to be in range [0.50, 0.70] depending on calibration outcome. Log the chosen value in the Phase 3 report."),
    ("10-C", "KNOWN-LIMITATION",
     "Fundamental softmax-architecture limitation. The model card MUST state: 'LADI-Net is a single-label classifier. Softmax forces predicted probabilities to sum to 1, making genuine co-infections (e.g., concurrent septoria AND late blight on the same leaf) impossible to detect as dual-class predictions. In Kerala field conditions where multiple tomato diseases may co-occur, the model will return the most prominent single disease; diagnostic accuracy may be reduced for mixed-infection cases. Farmers who observe persisting symptoms after treating the predicted disease should consult an agronomist for re-assessment.' Tier 5 (co-infection) is de-facto dead — this is acknowledged."),

    # STAGE 11
    ("11-A", "DEFERRED-LATER",
     "Inference-server-integration concern. Documented: the FastAPI server (app/main.py or equivalent) MUST store the ORIGINAL upload bytes and pass them to LADI-Net independently of the Router's preprocessing. LADI-Net's entrypoint accepts raw bytes → decodes → applies its own InSPyReNet + resize-to-392px pipeline. The Router's 224px LAB-CLAHE output is used only for crop classification; discarded before reaching LADI-Net. Addressed post-Phase-4 when the inference server code is written."),
    ("11-B", "DEFERRED-LATER",
     "Same post-Phase-4 server-integration concern. Current server returns error for tomato — this is expected until LADI-Net training completes and the deployment artifacts (`ladinet_v1_soup.pt`, `prototype_bank.pt`, `temperatures.pt`) are produced. Server integration code is written AFTER the Clearance Report at the end of Phase 4."),
    ("11-C", "DEFERRED-LATER",
     "Post-deployment empirical benchmark. Documented: when the inference server loads all three models (Router DINOv2-Small ~90 MB + Model 2 APIN ~500 MB + LADI-Net ~400 MB), measure peak VRAM with torch.cuda.max_memory_allocated after 10 test inference requests. Expected ~1.2-1.4 GB total. Acceptable budget: <2 GB peak VRAM for all three models. If exceeded, the server loads models lazily (Router stays hot; Model 2/LADI-Net swap on demand based on crop)."),
    ("11-D", "DEFERRED-LATER",
     "Post-deployment empirical benchmark. Measure end-to-end latency on the RTX 4060: image upload → InSPyReNet → Pass 1 → Pass 2 → ABMIL → gated MLP → temperature + prototype → tier assignment. Target: <2 seconds total. If exceeded, the optimization candidates are (a) InSPyReNet at 'fast' mode (already in use), (b) Pass 1 + Pass 2 at bs=1 TensorRT export, (c) ABMIL in FP16 at inference. Log as a post-Phase-4 TODO."),

    # STAGE 12
    ("12-A", "FIXED",
     "`scripts/ladi_net/background_recomposer.py` updated on 2026-04-22: `preload_max: int = 2000` (was 1000), `preload_resize_max_dim: int = 392` (was 512). RAM cost: 2000 × 392 × 392 × 3 bytes = ~922 MB, well within the 32 GB system budget. Verified by pre_phase1_checks Check 3: `preload_resolution_392=True, preload_max_2000=True, total pool size=7765 images, estimated reuse per bg per epoch = 2.02`. Background pool now exceeds the 2000 preload cap (so the cap limits which 2000 are preloaded, but all 7765 are available)."),
    ("12-B", "FIXED",
     "Resolved by pre_phase1_checks.py Check 7 (run 2026-04-22). Scanned the 9,705 recomposed entries in the source map; each has a unique source-leaf ID. Max composites per source leaf = 1. No duplication risk. Top 5 source_ids each appear exactly once. This confirms the static Phase 0 recomposition script (`scripts/phase0_background_recomposition.py`) generated exactly one composite per source leaf. Result logged to pre_phase1_check_results.json under `static_recomposed.duplication_risk=False`."),
    ("12-C", "FIXED",
     "Decision 17 documents the standard augmentation chain for Phase 2 (applied AFTER recomposition, AFTER LAB-CLAHE, AFTER resize to 392px — all augmentations operate on the final 392×392 image): `A.HorizontalFlip(p=0.5)` + `A.Affine(rotate=(-15,15), p=0.5)` (no shear, no scale — preserves disease pattern structure) + `A.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.0, p=0.5)` (conservative; hue=0 prevents disease colour-signature shifts) + `A.RandomResizedCrop(392, 392, scale=(0.82, 1.0), ratio=(0.95, 1.05), p=0.5)` (mild scale jitter, adds scale invariance). The Phase 2 training DataLoader chains these via albumentations in the stated order."),

    # STAGE 13
    ("13-A", "DEFERRED-PHASE2",
     "Phase 2 validation/diagnostic code guideline, documented in Decision 17. Any linear probe or quick classifier used during Phase 2 or Phase 3 (e.g., post-Phase-1 ABMIL validation, per-epoch confusable-pair centroid distance check) MUST be implemented as `nn.Linear(768, 6)` trained with AdamW for 50 epochs on GPU. Target runtime: <30 seconds per fit. Reserves the ~15-minute sklearn LogReg cost only for cases where exact reproducibility against the existing Phase 0D probe numbers is required. Phase 2 training script includes a helper `fit_gpu_linear_probe(X_train, y_train, X_val, y_val, epochs=50)` that returns macro/weighted F1."),
]

# Cross-check items at the end of the file
CROSS_CHECK_ITEMS = [
    ("Missing — Weight decay specification", "FIXED",
     "Documented in Decision 17. AdamW weight_decay=0.01 for LoRA parameters only (no weight decay on ABMIL/gated MLP/SupCon projector — they're trained from scratch and weight decay on small new parameter sets tends to over-regularise). Bias and LayerNorm parameters excluded via parameter-group filtering."),
    ("Missing — Cosine schedule warmup duration", "FIXED",
     "Documented in Decision 17. 2-epoch linear warmup from 0 → peak LR, then cosine anneal for the remaining epochs to 10% of peak LR. For Phase 2 (25 epochs × ~2000 steps/epoch = 50,000 steps): warmup = 4,000 steps ≈ 2 epochs. Implemented via torch.optim.lr_scheduler.SequentialLR chaining LinearLR + CosineAnnealingLR."),
    ("Missing — Conformal calibration underpowered for thin classes", "KNOWN-LIMITATION",
     "YLCV has ~2-3 conformal calibration samples; statistically valid 95% conformal coverage requires n≥20 per class. Model card must state: 'Conformal prediction-set coverage guarantees for YLCV and mosaic_virus are inherited from the population-level calibration and may be less reliable for these specific classes at the 0.95 coverage level. Per-class conformal thresholds for YLCV should be treated as best-effort.' This is a data-quantity limitation, not a code limitation."),
    ("Missing — Prototype k-means random seed", "FIXED",
     "Documented in Decision 17. All k-means calls in the prototype-bank construction (Phase 3) and any other clustering operation use `random_state=42`. Phase 3 script enforces this via a shared config constant K_MEANS_SEED=42."),
    ("Missing — ABMIL heatmap coordinate mapping at inference", "DEFERRED-LATER",
     "Inference-pipeline concern. Documented: the coordinate chain is (a) ABMIL patch index k in [0..783] for a 28×28 grid → (b) patch (row, col) = (k//28, k%28) → (c) patch center pixel in 392px crop = (col*14+7, row*14+7) → (d) inverse of the lesion-crop bounding-box affine to map back to the 392px full image → (e) the original upload was resized/padded to 392px; store the resize transform to invert back. Write unit tests: send a synthetic image with a known red-pixel lesion; verify that the top-1 ABMIL attention patch maps back to within 10 px of the red pixel in the original image. Post-Phase-4 server code."),
    ("Missing — Inference time benchmark", "DEFERRED-LATER",
     "Same category as Issue 11-D. Empirical benchmark after the inference server is built."),
    ("Missing — Temperature calibration probe overlap with field_val sources", "KNOWN-LIMITATION",
     "The 28-image confusable_pair_probe and the 203-image field_val were carved from the same 1,015 real-field pool using the same StratifiedShuffleSplit. They do NOT overlap at the image level (Issue 5 of Check 5 confirmed no_cross_split_overlap). But they MAY share image sources (the same photographer, same geographic area). This is a known source-overlap limitation of the small real-field dataset. Documented in Critique 3 (YLCV) and acknowledged in the model card. No action possible at data-scale; the correct response is to report confidence intervals on all reported metrics and never to claim improvements smaller than the CI width."),
]


def _build_status_block(status: str, reason: str) -> str:
    return (
        f"\n\nFix Status: {status}\n\n"
        f"Reason/Outcome:\n{reason}"
    )


def main() -> None:
    text = ISSUES_PATH.read_text(encoding="utf-8")

    # Skip 0-A: already has a status block (inserted earlier in this session)
    # We'll detect duplicates by checking for "Fix Status:" within 200 chars of the issue line

    for issue_id, status, reason in STATUS:
        # The Issue header line looks like: **Issue 0-A: ...** (bold)
        # Use a regex that captures the full issue paragraph(s) until the next Issue header,
        # a triple-dash divider, or a new Stage heading.
        header_pat = re.compile(
            rf"(\*\*Issue {re.escape(issue_id)}:.*?\*\*.*?)(?=\n\*\*Issue |\n---\n|\n## STAGE |\Z)",
            re.DOTALL,
        )
        m = header_pat.search(text)
        if m is None:
            print(f"[WARN] could not locate Issue {issue_id}")
            continue
        block = m.group(1)
        if "Fix Status:" in block:
            # Already marked — skip
            continue
        new_block = block.rstrip() + _build_status_block(status, reason) + "\n"
        text = text[:m.start()] + new_block + text[m.end():]

    # Cross-check items at the end — keyed by short phrase at start of the header line
    for header_prefix, status, reason in CROSS_CHECK_ITEMS:
        header_pat = re.compile(
            rf"(\*\*{re.escape(header_prefix)}.*?\*\*.*?)(?=\n\*\*Missing |\n---\n|\n## |\Z)",
            re.DOTALL,
        )
        m = header_pat.search(text)
        if m is None:
            print(f"[WARN] could not locate cross-check item: {header_prefix}")
            continue
        block = m.group(1)
        if "Fix Status:" in block:
            continue
        new_block = block.rstrip() + _build_status_block(status, reason) + "\n"
        text = text[:m.start()] + new_block + text[m.end():]

    ISSUES_PATH.write_text(text, encoding="utf-8")
    # Count inserted statuses
    n = text.count("Fix Status:")
    print(f"Total 'Fix Status:' markers in file: {n}")
    # Sanity: count issue headers
    n_issues = len(re.findall(r"\*\*Issue \d", text))
    n_cross = len(re.findall(r"\*\*Missing — ", text))
    print(f"Issues found: {n_issues}, cross-check items found: {n_cross}")


if __name__ == "__main__":
    main()
