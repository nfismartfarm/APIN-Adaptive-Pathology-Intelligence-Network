# decisions.md — Plant Disease Detection: Kerala
# Authoritative log of every architectural, data, and infrastructure decision.
# Pre-populated with all decisions made during project design.
# Claude Code appends new entries during implementation.
# NEVER delete an entry. NEVER change a past entry without adding a superseding entry.
# Last updated: [DATE]

---

## HOW TO USE THIS FILE

Before changing ANY constant, threshold, architecture component, or data strategy,
search this file for that thing. If a decision entry exists, read it fully before
acting. If the entry says "tried X, resulted in Y, reverted to Z" — do not try X
again without a concrete reason why this time will be different.

When you make a new decision during implementation, append it at the bottom of the
relevant section immediately. Do not wait until later. The value of this file
depends entirely on it being written in the moment, not reconstructed from memory.

Entry format:
```
DECISION   : [What was decided]
DATE       : [When it was decided — "design phase" for pre-populated entries]
CONTEXT    : [Why this decision needed to be made at all]
CONSIDERED : [Alternatives that were evaluated]
CHOSEN     : [What was selected and why — specific reasons, not vague]
EVIDENCE   : [Any metric, experiment, or reasoning that supports the choice]
REVISIT IF : [Conditions under which this decision should be reconsidered]
STATUS     : [ACTIVE / SUPERSEDED BY decision-XX]
```

---

## SECTION 1: BACKBONE ARCHITECTURE

---

**DECISION-01**
```
DECISION   : Use EfficientNetV2-S as the backbone, not MobileNetV2
DATE       : Design phase
CONTEXT    : The colleague's system uses two separate MobileNetV2 models (one for
             tomato, one for chilli). We evaluated whether to extend that approach
             or redesign from scratch.
CONSIDERED : MobileNetV2 (colleague's approach), EfficientNetV2-S, EfficientNetV2-M,
             ConvNeXtV2-T, DINOv2-S, ResNet50
CHOSEN     : EfficientNetV2-S
EVIDENCE   : MobileNetV2 was designed for mobile edge inference at 224×224 with
             strict VRAM and compute budgets. It achieves good single-label
             classification but its feature maps are shallow — the top feature map
             is 7×7 at 224px input, which makes spatial disease localisation (needed
             for Grad-CAM) coarse. EfficientNetV2-S produces richer multi-scale
             features at stages 2 (28×28), 3 (14×14), and 4 (7×7). This is the
             exact multi-scale structure that FPN requires. ConvNeXtV2-T and
             DINOv2-S would give better features but cost 3-4× the VRAM and
             2-3× the training time on the RTX 4060. EfficientNetV2-M was
             considered but at 54M parameters it strains the 8GB VRAM budget
             at batch 32 with mixed precision. EfficientNetV2-S at 22M parameters
             fits comfortably.
REVISIT IF : VRAM is upgraded to 16GB+, or inference time > 15 seconds is
             acceptable, or tier-2 macro F1 < 0.40 after full training suggests
             the backbone is the bottleneck.
STATUS     : ACTIVE
```

---

**DECISION-02**
```
DECISION   : Use timm features_only=True with out_indices=(2,3,4) for backbone
DATE       : Design phase
CONTEXT    : timm offers two ways to use EfficientNetV2-S: as a full classifier
             (include_top=True) or as a feature extractor (features_only=True).
             The FPN requires intermediate feature maps at multiple scales, not
             just the final pooled vector.
CONSIDERED : features_only=True out_indices=(2,3,4) vs full model with hooks vs
             manually slicing the model
CHOSEN     : features_only=True, out_indices=(2,3,4)
EVIDENCE   : timm's features_only interface is purpose-built for this use case.
             It returns a list of feature maps at the specified stages without
             any code to maintain. Hooks require careful registration and
             teardown — they interact badly with torch.compile and pytorch_grad_cam.
             Manual slicing requires re-implementing timm's internal model
             structure. Stage indices (2,3,4) correspond to spatial sizes
             (28,14,7) at 224px input, which matches the FPN's expected inputs.
REVISIT IF : timm version changes and out_indices semantics change. Always verify
             backbone shapes with verify_backbone_shapes() before any training.
             If actual shapes do not match (48,28,28),(160,14,14),(256,7,7), update
             FPN_IN_CH in app/config.py to match actual values.
STATUS     : ACTIVE
```

---

**DECISION-03**
```
DECISION   : FPN output channels = 256, not 128
DATE       : Design phase
CONTEXT    : FPN projects all three backbone stages to a uniform channel count.
             Higher channels = more expressive features but more VRAM and compute.
CONSIDERED : 64, 128, 256, 512
CHOSEN     : 256
EVIDENCE   : 128 channels produces a pooled feature vector of 128 dimensions for
             the classification heads. Preliminary reasoning: the disease head must
             distinguish 10 classes with co-infection possibility, and the crop
             classifier must be reliable enough to gate disease predictions. 128
             dimensions is tight for 10 fine-grained disease classes. 256 is the
             standard for FPN implementations in detection literature (Feature
             Pyramid Networks, Lin et al. 2017). 512 would double the VRAM cost
             with diminishing returns at this scale. 256 fits comfortably in the
             VRAM budget at batch 32.
REVISIT IF : VRAM is consistently above 7.5 GB during Phase 2 — reduce to 128
             as first relief measure. OR if macro F1 plateaus and a capacity
             increase is warranted — try 512.
STATUS     : ACTIVE
```

---

**DECISION-04**
```
DECISION   : Use FPN top-down pathway with output at P3 (28×28), not P5 (7×7)
DATE       : Design phase
CONTEXT    : FPN produces feature maps at multiple scales. The final pooled feature
             vector for the classification heads comes from one of these scales.
             We also need a meaningful feature map for Grad-CAM visualisation.
CONSIDERED : P3 (28×28), P4 (14×14), P5 (7×7)
CHOSEN     : P3 (28×28) — fused features via top-down pathway, output at P3
EVIDENCE   : P5 (7×7) is the coarsest resolution. A 7×7 Grad-CAM heatmap at
             224×224 input is a 32-pixel grid — too coarse to meaningfully
             localise disease regions on a leaf. P3 (28×28) gives an 8-pixel
             grid, which is standard for CAM-based visualisation. The FPN
             top-down pathway fuses P5 semantics into P3 spatial resolution,
             giving the best of both: rich semantic content from deep layers
             propagated to higher spatial resolution.
REVISIT IF : Never. P3 as FPN output is a fundamental design choice that the
             entire inference pipeline (Grad-CAM hooks, heatmap generation) depends
             on. Changing this would require updating target_layer in generate_heatmap
             and retraining. The correct target is always model.fpn.out_p3.
STATUS     : ACTIVE — NEVER CHANGE model.fpn.out_p3 without updating all Grad-CAM code
```

---

## SECTION 2: CLASSIFICATION HEAD ARCHITECTURE

---

**DECISION-05**
```
DECISION   : Use sigmoid + BCEWithLogitsLoss for disease head (multi-label),
             not softmax + CrossEntropyLoss (single-label)
DATE       : Design phase
CONTEXT    : The colleague's system uses softmax, which assumes exactly one disease
             per image. The system being built must handle co-infection: okra_yvmv
             and okra_cercospora can appear simultaneously on the same leaf.
CONSIDERED : Softmax + CrossEntropyLoss (single-label, one disease per image),
             Sigmoid + BCEWithLogitsLoss (multi-label, any combination per image)
CHOSEN     : Sigmoid + BCEWithLogitsLoss (multi-label)
EVIDENCE   : Softmax outputs sum to 1.0 — if okra_yvmv gets 0.7, all other classes
             share the remaining 0.3 regardless of whether they are also present.
             This is mathematically incompatible with co-infection. Sigmoid treats
             each class independently: okra_yvmv=0.82, okra_cercospora=0.67 is a
             valid output meaning both are present. This matches the biological
             reality where a plant can have multiple concurrent diseases.
             BCEWithLogitsLoss with pos_weight handles the class imbalance that
             would otherwise suppress predictions for thin classes.
REVISIT IF : Never for the disease head. The multi-label requirement is fundamental
             to the project specification. The crop head correctly uses softmax
             (a leaf is either okra or brassica, never both).
STATUS     : ACTIVE
```

---

**DECISION-06**
```
DECISION   : Use FiLM conditioning from crop classifier into disease head
DATE       : Design phase
CONTEXT    : The 10 disease classes are split across two crops: okra_* and brassica_*.
             Without conditioning, the disease head must learn to suppress cross-crop
             predictions from scratch. With FiLM, the crop embedding explicitly
             modulates the disease features to bias toward crop-relevant diseases.
CONSIDERED : No conditioning (disease head predicts all 10 classes independently),
             Hard gating (zero out cross-crop logits during inference only),
             FiLM conditioning (learned soft modulation during training and inference),
             Separate disease heads per crop
CHOSEN     : FiLM conditioning (scale/shift from crop embedding)
EVIDENCE   : Hard gating at inference only means the model never learns to use the
             crop identity during training — it must learn 10-class prediction with
             cross-crop examples confusing the gradients. Separate heads double the
             parameter count and require the pipeline to route to the right head.
             FiLM is a learned modulation: gamma (scale) and beta (shift) from the
             64-dim crop embedding are applied to the 256-dim pooled features before
             the disease MLP. This is end-to-end differentiable and teaches the model
             that crop identity is a relevant signal for disease prediction.
             Hard gating is still applied at inference as a post-processing step
             to prevent any cross-crop leakage above DISEASE_THRESH.
REVISIT IF : Crop classifier accuracy < 0.85 (FiLM conditioning on a poor crop
             embedding would hurt disease prediction). In that case, disable FiLM
             and use hard gating only.
STATUS     : ACTIVE
```

---

**DECISION-07**
```
DECISION   : Separate severity head (3-class softmax), not integrated into disease head
DATE       : Design phase
CONTEXT    : Severity (mild/moderate/severe) needs to be output per inference.
             Two options: predict severity jointly with disease (one head), or
             as a separate head from the same pooled features.
CONSIDERED : Joint disease+severity prediction (one multi-task head), Separate
             severity head, No severity head (compute from Grad-CAM coverage only)
CHOSEN     : Separate severity head with CrossEntropyLoss
EVIDENCE   : Joint prediction forces the model to share capacity between disease
             localisation and severity regression — they have different loss scales
             and gradients can interfere. Severity from Grad-CAM coverage alone
             (the proxy label approach) is used to generate training labels but
             is not accurate enough for production use — it works as a training
             signal but not as a direct output. A separate 3-class head trained
             on proxy severity labels gives a direct, calibrated severity output
             that is presented to the farmer alongside treatment advice.
REVISIT IF : Severity head accuracy < 60% on validation set after Phase 2 — in
             that case, fall back to computing severity from Grad-CAM coverage
             directly at inference time and remove the severity head from the
             production output.
STATUS     : ACTIVE
```

---

**DECISION-08**
```
DECISION   : DROPOUT_P = 0.3, not 0.5
DATE       : Design phase
CONTEXT    : Dropout rate affects both training regularisation and MC Dropout
             uncertainty estimation. Higher dropout = more regularisation but also
             wider uncertainty intervals.
CONSIDERED : 0.1, 0.2, 0.3, 0.5
CHOSEN     : 0.3
EVIDENCE   : 0.5 is aggressive for a model where the heads are relatively thin
             (256 → 256 → N_CLASSES). With only one hidden layer, 50% dropout
             means the effective capacity is halved every forward pass, which
             may be too aggressive for learning fine-grained disease features.
             0.1-0.2 is too low to provide meaningful uncertainty signal for
             MC Dropout — the variance between passes would be negligible.
             0.3 is the standard moderate dropout rate and gives meaningful
             variance in MC passes without destroying head capacity.
REVISIT IF : MC Dropout uncertainty is consistently too low (all predictions
             near 0.0 uncertainty even for clearly ambiguous images) — increase
             to 0.4. OR if training loss fails to converge — decrease to 0.2.
STATUS     : ACTIVE
```

---

## SECTION 3: TRAINING STRATEGY

---

**DECISION-09**
```
DECISION   : Two-phase training with feature caching for Phase 1
DATE       : Design phase
CONTEXT    : The RTX 4060 has 8GB VRAM. Training the full model end-to-end from
             epoch 1 is possible but slow. An alternative is to freeze the backbone
             and train only the heads first (Phase 1), then unfreeze for fine-tuning
             (Phase 2).
CONSIDERED : Single-phase full training (no freeze), Two-phase with feature caching,
             Two-phase without caching (backbone runs frozen but still in GPU memory)
CHOSEN     : Two-phase with feature caching
EVIDENCE   : Phase 1 with caching is the key insight: if the backbone is frozen,
             its output features for each image are identical every epoch. There is
             no point running the backbone forward pass N_IMAGES × N_EPOCHS times
             when we can run it once and cache the results. Caching converts Phase 1
             from a 3.5-hour operation to a 30-minute operation. Phase 2 (fine-tuning)
             cannot use caching because unfreezing the backbone means its outputs
             change every epoch. The two phases teach the heads first (Phase 1),
             then jointly refine backbone and heads (Phase 2). This is standard
             transfer learning practice and avoids the early epochs of Phase 2
             destroying the heads' learned representations.
REVISIT IF : Never change the two-phase structure. Feature caching is critical for
             the training time target of ~4 hours total. Removing caching would
             triple Phase 1 time.
STATUS     : ACTIVE
```

---

**DECISION-10**
```
DECISION   : Cache features using get_eval_transform(), NOT get_train_transform()
DATE       : Design phase
CONTEXT    : When caching backbone features for Phase 1, we must decide which
             transform to apply to images before passing through the backbone.
CONSIDERED : get_train_transform() (with augmentation), get_eval_transform() (deterministic)
CHOSEN     : get_eval_transform() — deterministic resize + CLAHE + normalize only
EVIDENCE   : If features are cached with get_train_transform(), each image gets a
             random augmentation applied before the backbone. That augmented feature
             vector is then fixed for all Phase 1 epochs. The result: Phase 1 trains
             on a set of fixed, randomly-augmented features with no diversity
             benefit — the augmentation ran once and is then frozen. This gives
             worse Phase 1 generalisation than just using deterministic features.
             The correct approach: cache clean eval features (deterministic), let
             Phase 2 apply augmentation during actual full-model training.
REVISIT IF : Never. This is a correctness requirement, not a tuning decision.
             Using train_transform for caching is always wrong.
STATUS     : ACTIVE
```

---

**DECISION-11**
```
DECISION   : LLRD (Layer-wise Learning Rate Decay) for Phase 2 optimizer
DATE       : Design phase
CONTEXT    : Phase 2 unfreezes the top 1/3 of backbone blocks. Different layers
             need different learning rates: heads need a higher rate to adapt
             quickly, shallow backbone layers need a very low rate to preserve
             ImageNet pretraining.
CONSIDERED : Uniform LR for all parameters, LLRD, Frozen backbone throughout (no Phase 2)
CHOSEN     : LLRD via get_llrd_optimizer() in training/helpers.py
EVIDENCE   : Shallow backbone layers (stem, early blocks) encode low-level features
             like edges, textures, and colour gradients that are already well-calibrated
             from ImageNet pretraining. Applying full PHASE2_BASE_LR to these layers
             would overwrite useful learned features. Deep backbone blocks encode
             task-specific features that need updating for the disease domain. LLRD
             decays LR by decay=0.85 per block moving from output (deepest) toward
             input (shallowest). With 10 blocks and decay=0.85, the ratio from
             deepest to shallowest is 0.85^10 ≈ 0.20 — deepest blocks get 5× the
             LR of the shallowest. Heads and FPN get full PHASE2_BASE_LR = 1e-4.
REVISIT IF : Phase 2 training is unstable (gradient norm > 10 consistently) —
             reduce PHASE2_BASE_LR to 5e-5. OR if Phase 2 shows no improvement over
             Phase 1 — increase PHASE2_BASE_LR to 2e-4.
STATUS     : ACTIVE
```

---

**DECISION-12**
```
DECISION   : OneCycleLR scheduler with pct_start=ONE_CYCLE_PCT (0.1), not manual warmup
DATE       : Design phase
CONTEXT    : Phase 2 fine-tuning requires a warmup period to prevent large gradient
             updates at the start from destroying Phase 1 head weights.
CONSIDERED : Linear warmup (500 steps) + cosine decay, OneCycleLR, ReduceLROnPlateau,
             Constant LR, Cosine annealing without warmup
CHOSEN     : OneCycleLR with pct_start=0.1 (10% of total steps as warmup)
EVIDENCE   : OneCycleLR has warmup built in via pct_start. Adding a separate manual
             warmup on top creates two conflicting LR schedules for the first 10% of
             training — OneCycleLR ramps up while manual warmup also ramps up, and the
             actual LR becomes undefined. This is a documented failure mode in the v6
             spec. OneCycleLR with pct_start=0.1, div_factor=10, final_div_factor=1000
             gives: starting LR = max_lr/10, warm up to max_lr over 10% of steps, then
             cosine anneal to max_lr/1000. This is a well-tested schedule for fine-tuning.
REVISIT IF : Never add a separate manual warmup on top of OneCycleLR. The three
             constants (ONE_CYCLE_PCT, ONE_CYCLE_DIV, ONE_CYCLE_FDIV) are defined in
             app/config.py and must be imported — never hardcode 0.1, 10, 1000.
STATUS     : ACTIVE
```

---

**DECISION-13**
```
DECISION   : Unfreeze top 1/3 of backbone blocks for Phase 2, not all blocks
DATE       : Design phase
CONTEXT    : How many backbone layers to unfreeze for Phase 2 fine-tuning.
             More unfrozen layers = more adaptation but more risk of catastrophic
             forgetting of ImageNet features.
CONSIDERED : Unfreeze all layers, Unfreeze top 50%, Unfreeze top 33%, Unfreeze top 20%,
             Keep everything frozen (Phase 1 only)
CHOSEN     : Top 33% (fraction=0.33 in unfreeze_top_fraction())
EVIDENCE   : Unfreezing all layers on a dataset of ~10,000 images is high risk —
             with limited data, the backbone can overfit and forget ImageNet features
             that generalise well to leaf texture. Top 33% gives EfficientNetV2-S
             approximately the last 3-4 of its 10 stage blocks. These encode the
             most task-specific features (disease textures, colour patterns) while
             the earlier blocks (edge detectors, colour gradients) remain frozen.
             Top 50% was considered but the additional VRAM cost (more parameters
             computing gradients) tightens the 8GB budget.
REVISIT IF : Phase 2 macro F1 improves less than 0.05 over Phase 1 — try increasing
             to 50%. OR if Phase 2 shows catastrophic forgetting (val F1 collapses
             on early-learning classes) — reduce to 20%.
STATUS     : ACTIVE
```

---

**DECISION-14**
```
DECISION   : BATCH_SIZE=32 with GRAD_ACCUM_STEPS=1 as default; fallback to 16+2
DATE       : Design phase
CONTEXT    : RTX 4060 has 8GB VRAM. Batch size affects training stability and speed.
CONSIDERED : Batch 8, 16, 32, 64
CHOSEN     : 32 as default; 16 with GRAD_ACCUM_STEPS=2 as OOM fallback
EVIDENCE   : VRAM budget analysis: backbone (top 33% unfrozen) ~2.2GB, FPN+heads
             ~0.4GB, batch 32 activations at 224×224 ~3.0GB, mixed precision overhead
             ~0.5GB = total ~6.1GB. This leaves ~1.9GB headroom in 8GB — sufficient
             for gradient buffers and CUDA overhead. Batch 64 would require ~5GB for
             activations alone, exceeding the budget. Batch 32 + GRAD_ACCUM_STEPS=2
             is mathematically equivalent to batch 64 but uses half the VRAM by
             accumulating gradients over two smaller batches before the optimizer step.
             If OOM occurs at batch 32: reduce to 16, set GRAD_ACCUM_STEPS=2.
REVISIT IF : OOM occurs during Phase 2 — immediately reduce to BATCH_SIZE=16 and
             GRAD_ACCUM_STEPS=2 in app/config.py.
STATUS     : ACTIVE
```

---

**DECISION-15**
```
DECISION   : pos_weight for BCEWithLogitsLoss computed as n_neg/n_pos from binary
             label matrix, NOT sklearn compute_class_weight
DATE       : Design phase
CONTEXT    : The disease classes are imbalanced. brassica_clubroot has ~150 images
             while okra_yvmv has ~2000. Without weighting, the model ignores thin classes.
CONSIDERED : sklearn.compute_class_weight('balanced'), Manual n_neg/n_pos formula,
             No weighting, Oversampling only
CHOSEN     : n_neg/n_pos per class from binary label matrix
EVIDENCE   : sklearn.compute_class_weight('balanced') is designed for single-label
             classification — it takes a 1D array of integer class labels. For
             multi-label, each image contributes to multiple classes simultaneously.
             Using sklearn here is mathematically wrong: it would count the total
             number of positive label occurrences as n_total, making n_neg ≈ 0 for
             all classes and producing near-zero pos_weights. The correct formula for
             BCEWithLogitsLoss pos_weight is: build a binary matrix [N, NUM_CLASSES]
             where each row has a 1 for the true class, then pos_weight[j] = (N -
             n_pos[j]) / n_pos[j] where n_pos[j] is the column sum for class j.
             This is specified in the PyTorch BCEWithLogitsLoss documentation.
REVISIT IF : Never change this formula without re-reading the PyTorch docs. The
             n_neg/n_pos formula is correct for multi-label. sklearn is incorrect.
STATUS     : ACTIVE
```

---

**DECISION-16**
```
DECISION   : MAX_POS_WEIGHT = 10.0 cap on pos_weight values
DATE       : Design phase
CONTEXT    : pos_weight can become very large if a class has very few positives
             (e.g. 10 positives out of 10,000 images → pos_weight = 999).
             Extreme pos_weights destabilise training.
CONSIDERED : No cap, cap at 5, cap at 10, cap at 20
CHOSEN     : Cap at 10.0
EVIDENCE   : A pos_weight of 999 means the model is penalised 999× more for
             false negatives on that class. With SGD this creates enormous gradient
             updates that overwhelm the gradients from all other classes, causing
             training instability. Cap at 10 means the thinnest class is penalised
             at most 10× relative to negatives. Combined with CLUBROOT_OVERSAMPLE=2.0
             (the WeightedRandomSampler doubling clubroot's appearance frequency),
             the effective treatment of thin classes is adequate without destabilising
             the loss.
REVISIT IF : brassica_clubroot F1 consistently < 0.30 after Phase 2 — increase cap
             to 15 or increase CLUBROOT_OVERSAMPLE to 3.0.
STATUS     : ACTIVE
```

---

**DECISION-17**
```
DECISION   : WeightedRandomSampler oversamples brassica_clubroot by CLUBROOT_OVERSAMPLE=2.0
DATE       : Design phase
CONTEXT    : brassica_clubroot is the thinnest class (~150 images). Above-ground
             symptoms (wilting, yellowing) are non-specific — the model needs to
             see many examples to learn to associate these symptoms with clubroot
             rather than other causes of wilting.
CONSIDERED : No oversampling, Oversampling by 2×, Oversampling by 3×, Synthetic
             image generation only
CHOSEN     : 2× oversampling via WeightedRandomSampler, plus synthetic generation
             if < 150 images
EVIDENCE   : 2× oversampling means clubroot appears twice as often in each training
             epoch. Combined with the pos_weight in the loss, clubroot receives
             additional learning signal without dominating the batch. 3× was
             considered but risks over-fitting to the limited clubroot examples —
             the model might memorise the training clubroot images rather than
             generalising. Synthetic generation handles the case where the dataset
             genuinely has < 150 clubroot images.
REVISIT IF : Clubroot F1 > 0.70 after Phase 2 — reduce to 1.5× (no need for
             aggressive oversampling if the class is well-represented already).
             OR if clubroot F1 < 0.30 — increase to 3× and add more synthetic images.
STATUS     : ACTIVE
```

---

**DECISION-18**
```
DECISION   : drop_last=False in all DataLoaders
DATE       : Design phase
CONTEXT    : PyTorch DataLoader has a drop_last parameter. When True, the last
             incomplete batch is discarded.
CONSIDERED : drop_last=True (clean batches, no batch-norm issues), drop_last=False
CHOSEN     : drop_last=False
EVIDENCE   : brassica_clubroot has approximately 150 images in training. With
             batch_size=32 and drop_last=True, the final batch of 22 images is
             discarded every epoch. Over 7 Phase 2 epochs, the model sees only
             (150 - 22) * 7 = 896 clubroot images instead of 150 * 7 = 1050.
             That is 14.7% of the already-thin clubroot class lost permanently.
             This is unacceptable for the thinnest class. BatchNorm with a batch
             of 22 is slightly noisier but not a problem when mixed precision and
             the frozen BatchNorm in eval mode (for inference) are used correctly.
REVISIT IF : Never. drop_last=False is a correctness requirement for thin classes.
STATUS     : ACTIVE
```

---

**DECISION-19**
```
DECISION   : persistent_workers=False, num_workers=0 on Windows
DATE       : Design phase
CONTEXT    : PyTorch DataLoader workers enable parallel data loading but have
             significant caveats on Windows.
CONSIDERED : num_workers=4 (parallel loading), num_workers=2, num_workers=0
             (main process only), persistent_workers=True
CHOSEN     : num_workers=0 on Windows, persistent_workers=False always
EVIDENCE   : Windows uses the "spawn" multiprocessing method instead of "fork".
             With num_workers > 0 and spawn, each worker process re-imports the
             entire training module. If any module-level code runs on import (e.g.
             model construction, dataset scanning), it runs in every worker process,
             creating exponential process spawning and an immediate crash. The fix is
             if __name__ == '__main__' guard around all training scripts, which is
             already required. However, even with the guard, debugging DataLoader
             workers on Windows is significantly harder. num_workers=0 runs data
             loading in the main process and is recommended for Windows development.
             The training time difference on this dataset size is small — the backbone
             forward pass dominates total time, not data loading.
             persistent_workers=True is always False because interrupted training
             (Ctrl+C, OOM, crash) leaves zombie worker processes holding file locks
             on data directories, causing PermissionError on the next run.
REVISIT IF : Training on Linux/WSL2 — num_workers=2 is then safe and beneficial.
             Detect platform at runtime: n_workers = 0 if sys.platform.startswith('win')
             else 2.
STATUS     : ACTIVE
```

---

**DECISION-20**
```
DECISION   : Early stopping monitors val/macro_f1, patience=EARLY_STOP_PAT (5 epochs)
DATE       : Design phase
CONTEXT    : We need to stop training when validation performance stops improving
             to avoid overfitting.
CONSIDERED : Monitor val_loss, monitor val/macro_f1, patience 3, 5, 10
CHOSEN     : val/macro_f1, patience=5
EVIDENCE   : val_loss is a proxy — it can decrease while per-class F1 for thin
             classes (brassica_clubroot, okra_enation) is still improving due to
             the weighted loss. val/macro_f1 directly measures what matters: average
             per-class performance across all 10 classes, giving thin classes equal
             weight. Patience=3 is too aggressive — the LR schedule creates natural
             dips in F1 that recover within 1-2 epochs. Patience=5 allows the model
             to recover from LR dips before stopping. Patience=10 would allow too
             much overfitting in a dataset of this size.
REVISIT IF : Training consistently hits the epoch limit without triggering early
             stopping — reduce patience to 3. OR if training is stopping too early
             (F1 still trending upward) — increase patience to 7.
STATUS     : ACTIVE
```

---

## SECTION 4: DATA STRATEGY

---

**DECISION-21**
```
DECISION   : Three-tier evaluation strategy (val / PlantDoc / Kerala)
DATE       : Design phase
CONTEXT    : A single train/val split cannot measure whether the model works in
             the real deployment environment (Kerala field conditions).
CONSIDERED : Single val split only, Train/val/test split (single domain),
             Three-tier (val + PlantDoc wild test + Kerala field test)
CHOSEN     : Three-tier
EVIDENCE   : Training data comes from Bangladesh (sabbir, iubat, faruk, kareem,
             misrak, ghose datasets), the US (PlantVillage lineage), and Europe.
             Kerala has different conditions: monsoon overcast lighting (blue-shifted
             colour temperature), local phone cameras (variable quality), and local
             crop varieties that may show different symptom presentation. A model
             that scores 0.85 on val (same distribution as training) may score 0.60
             on PlantDoc (independent collection, different conditions) and 0.55 on
             real Kerala photos. The three-tier strategy explicitly measures each
             level of generalisation. Tier-1 (val) measures training convergence.
             Tier-2 (PlantDoc) measures domain generalisation. Tier-3 (Kerala)
             measures deployment readiness.
REVISIT IF : Never change this structure. The three-tier strategy is the only
             honest way to know whether the model works for actual Kerala farmers.
STATUS     : ACTIVE
```

---

**DECISION-22**
```
DECISION   : PlantDoc is reserved entirely as tier-2 test — NEVER used for training
DATE       : Design phase
CONTEXT    : PlantDoc is the only freely available wild-collected brassica disease
             dataset with independent collection methodology.
CONSIDERED : Use PlantDoc for training (more data), Use PlantDoc for validation,
             Reserve entirely for tier-2 test (chosen)
CHOSEN     : Tier-2 test only — zero PlantDoc images in training pool
EVIDENCE   : If PlantDoc images appear in the training pool, the model optimises
             for PlantDoc's specific collection style (lighting, angles, backgrounds).
             The tier-2 evaluation then measures memorisation of that style rather
             than genuine domain generalisation. PlantDoc's value is precisely that
             it is independent. Using it for training destroys that independence
             permanently.
REVISIT IF : Never. This is a data integrity decision. Once tier-2 evaluation
             runs with PlantDoc as test, no PlantDoc images can ever enter training
             without invalidating all tier-2 results.
STATUS     : ACTIVE — LOCKED
```

---

**DECISION-23**
```
DECISION   : StratifiedGroupKFold split (stratified by class, grouped by source)
DATE       : Design phase
CONTEXT    : How to split the training pool into train/val/test.
CONSIDERED : Random split (train_test_split), Stratified split (StratifiedKFold),
             Stratified + grouped by source (StratifiedGroupKFold)
CHOSEN     : StratifiedGroupKFold grouping by source_dataset
EVIDENCE   : Random split allows images from the same Kaggle dataset to appear
             in both train and val/test. Kaggle datasets often contain images
             collected in the same location, with the same camera, in the same
             conditions. Near-duplicate images appearing on both sides of the
             train/test boundary give artificially inflated test scores — the model
             recognises specific image backgrounds and lighting setups, not the
             disease itself. Grouping by source ensures all images from, say,
             sabbir_okra are in one split — no source contributes to both train
             and test.
REVISIT IF : A new source has very few images (< 50) and putting it entirely in
             one split leaves val/test very thin. In that case: merge small sources
             into a 'misc' group before splitting.
STATUS     : ACTIVE
```

---

**DECISION-24**
```
DECISION   : image_path in source_map.csv is RELATIVE to project ROOT
DATE       : Design phase (gap fix)
CONTEXT    : source_map.csv is the master data registry. Paths must be either
             absolute or relative.
CONSIDERED : Absolute paths (C:\Users\seena\project\data\raw\...), Relative paths
             (data/raw/sabbir_okra/...)
CHOSEN     : Relative paths from ROOT
EVIDENCE   : Absolute paths are machine-specific. If the project is moved to a
             different directory, renamed, or run on a different machine, every
             path in source_map.csv breaks. Relative paths require only ROOT to be
             correct. ROOT is computed at runtime in app/config.py:
             ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).
             PlantDiseaseDataset resolves them with os.path.join(ROOT, rel_path).
             source_map.csv is in .gitignore (too large, machine-specific content
             even with relative paths since data/ is not committed). It must be
             regenerated by running 01_prepare_data.py on each machine.
REVISIT IF : Never use absolute paths in source_map.csv.
STATUS     : ACTIVE
```

---

**DECISION-25**
```
DECISION   : Images stay in data/raw/ — data/processed/ is created but left empty
DATE       : Design phase (gap fix)
CONTEXT    : Some pipelines copy images from raw to processed directories (applying
             transforms). Others read from raw and apply transforms on the fly.
CONSIDERED : Copy-and-transform to data/processed/ at pipeline setup,
             Read from data/raw/ and transform on the fly (chosen)
CHOSEN     : Read from data/raw/, transform on the fly in PlantDiseaseDataset
EVIDENCE   : Copying images to processed adds ~5-15 GB of disk usage with no benefit
             since albumentations transforms are fast (< 5ms per image). Copying also
             creates synchronisation problems: if source_map.csv changes (new images
             added, labels corrected), the processed directory must also be updated.
             Reading from raw with on-the-fly transforms means the only source of
             truth is data/raw/ + source_map.csv. data/processed/ directories exist
             to satisfy any external tooling that checks for the directory structure,
             but are never populated.
REVISIT IF : Data loading becomes the bottleneck (> 50% of batch time on data loading).
             In that case, consider pre-caching processed images. Monitor with
             DataLoader's worker timing before making this change.
STATUS     : ACTIVE
```

---

**DECISION-26**
```
DECISION   : CLAHE applied as first transform in BOTH training and inference
DATE       : Design phase
CONTEXT    : CLAHE (Contrast Limited Adaptive Histogram Equalisation) corrects
             for Kerala's monsoon blue-shift and indoor fluorescent yellow-shift.
CONSIDERED : CLAHE at inference only (post-processing), CLAHE at training only,
             CLAHE at both training and inference (chosen), No CLAHE
CHOSEN     : CLAHE at both training and inference — always first step
EVIDENCE   : If CLAHE is applied at inference but not training, the model trains
             on raw pixels and is tested on CLAHE-normalised pixels. This is a
             distribution mismatch — the model sees a fundamentally different
             representation at test time than it trained on. The correct approach:
             apply CLAHE at training (in get_train_transform) AND at inference
             (in preprocess_for_inference). Both functions apply CLAHE as the
             first step, before any augmentation or normalisation. This extends
             to _SevProxyDataset for severity label generation — severity saliency
             maps must also be computed on CLAHE images to match the training
             distribution.
REVISIT IF : Never remove CLAHE from inference without also removing from training.
             The two must always match.
STATUS     : ACTIVE
```

---

**DECISION-27**
```
DECISION   : Synthetic images go to data/raw/synthetic/{class_name}/, not data/processed/
DATE       : Design phase (gap fix)
CONTEXT    : Stable Diffusion generates synthetic images for thin classes. Where
             to store them.
CONSIDERED : data/processed/train/{class_name}/, data/raw/synthetic/{class_name}/,
             data/synthetic/{class_name}/
CHOSEN     : data/raw/synthetic/{class_name}/
EVIDENCE   : Keeping all images in data/raw/ (including synthetic) means
             01_prepare_data.py has one consistent scanning strategy. If synthetic
             images were in data/processed/, the script would need separate logic
             to scan a different directory. data/raw/synthetic/ fits naturally into
             the source scanning loop: synthetic is treated as another source_dataset
             with source_id='synthetic' and split always='train'.
REVISIT IF : Never. Changing the synthetic image path would require updating the
             01_prepare_data.py scanner, source_map.csv, and all path references.
STATUS     : ACTIVE
```

---

## SECTION 5: INFERENCE PIPELINE

---

**DECISION-28**
```
DECISION   : MC Dropout with 5 passes (MC_PASSES=5) for uncertainty estimation
DATE       : Design phase
CONTEXT    : The system must output an uncertainty score alongside its predictions
             so farmers can judge whether to trust the result.
CONSIDERED : No uncertainty (single forward pass), MC Dropout 3 passes, 5 passes,
             10 passes, 20 passes, Deep ensembles
CHOSEN     : MC Dropout with 5 passes
EVIDENCE   : MC Dropout enables uncertainty estimation from any model with Dropout
             layers by running multiple stochastic forward passes. 5 passes gives
             a reasonable uncertainty estimate with acceptable latency: at ~1 second
             per pass on the RTX 4060, 5 passes = ~5 seconds total, well within
             the 10-second target inference time. 10 passes would be more accurate
             but 10 seconds approaches the limit of acceptable web UX. 3 passes
             gives too little variance information. Deep ensembles (train multiple
             models) would be more principled but require 3-5× the training time
             and storage. MC Dropout is the practical choice.
             CRITICAL: MC Dropout requires model.eval() with ONLY Dropout layers
             set to train mode — NOT model.train() which puts BatchNorm in training
             mode and corrupts single-image batch statistics.
REVISIT IF : Inference time consistently > 10 seconds per image — reduce to 3 passes.
             OR if uncertainty scores are uninformative (all near 0 or all near 0.4)
             — increase to 10 passes and check DROPOUT_P.
STATUS     : ACTIVE
```

---

**DECISION-29**
```
DECISION   : Three separate temperature scalars (T_disease, T_crop, T_severity)
             fitted independently on val set
DATE       : Design phase
CONTEXT    : Model logits are not calibrated (probabilities do not match frequencies).
             Temperature scaling is the simplest post-hoc calibration method.
CONSIDERED : Single temperature T for all heads, Separate T per head (chosen),
             Platt scaling, Isotonic regression, No calibration
CHOSEN     : Separate T per head
EVIDENCE   : A single T fitted on disease logits does not correctly calibrate the
             crop or severity distributions — they have different output scales and
             loss functions (BCE vs CrossEntropy vs CrossEntropy). The disease head
             uses sigmoid (each class independent), the crop head uses softmax
             (sum to 1), the severity head uses softmax. A T fitted to minimise
             BCE on disease logits would systematically miscalibrate the softmax
             heads. Three separate LBFGS optimisations, each fitting one T for one
             head, takes under 60 seconds total and gives correctly calibrated
             probabilities from each head independently.
REVISIT IF : ECE after calibration is >= 0.10 — T fitting may have failed to
             converge. Check LBFGS iterations and TEMP_INIT value.
STATUS     : ACTIVE
```

---

**DECISION-30**
```
DECISION   : OOD detection returns HTTP 200 with ood_flagged=True, not HTTP 422/400
DATE       : Design phase
CONTEXT    : When the model is uncertain (low confidence or high MC Dropout variance),
             the response should signal this to the frontend without being an error.
CONSIDERED : HTTP 400 (bad request), HTTP 422 (Pydantic validation error), HTTP 200
             with ood_flagged=True in body (chosen)
CHOSEN     : HTTP 200 with ood_flagged=True
EVIDENCE   : OOD detection is a valid successful inference outcome — the model ran
             correctly and chose to flag uncertainty. HTTP 400 implies the request
             was malformed (it was not). HTTP 422 is FastAPI's Pydantic validation
             error code — using it for OOD would confuse clients that check error
             codes programmatically and might interpret it as a request body error.
             HTTP 200 with ood_flagged=True allows the JavaScript frontend to display
             a "Low confidence" badge without treating the response as an error.
             The frontend checks !response.ok (which is false for 200) and renders
             the result normally, with the OOD badge visible.
REVISIT IF : Never. HTTP 422 for OOD is explicitly wrong.
STATUS     : ACTIVE
```

---

**DECISION-31**
```
DECISION   : apply_clahe defined inline in app/inference.py, NOT imported from
             training/transforms.py
DATE       : Design phase (gap fix)
CONTEXT    : apply_clahe uses only cv2 and numpy. It exists in both
             training/transforms.py (for training) and app/inference.py (for inference).
CONSIDERED : Single definition in training/transforms.py, imported everywhere;
             Single definition in app/utils.py; Inline copy in app/inference.py (chosen)
CHOSEN     : Inline copy in app/inference.py
EVIDENCE   : app/inference.py is a production server module. Importing from
             training/transforms.py would pull in the entire training package as a
             dependency of the production server: albumentations, sklearn, and the
             full training module tree would be loaded every time the server starts.
             This is a cross-layer import that violates the app/training boundary.
             apply_clahe is 6 lines using only cv2 and numpy — both already in
             requirements.txt for production. Duplicating 6 lines is better than
             creating a training package dependency in production.
REVISIT IF : apply_clahe grows significantly more complex (> 30 lines) — at that
             point, move it to app/utils.py and import from there in both places.
STATUS     : ACTIVE
```

---

## SECTION 6: INFRASTRUCTURE

---

**DECISION-32**
```
DECISION   : Local-only training — no Vast.ai, no cloud
DATE       : Design phase
CONTEXT    : An earlier version of the spec included Vast.ai as a cloud training
             fallback option.
CONSIDERED : Vast.ai cloud instances (RTX 5070 Ti at ~$0.11/hr),
             Google Colab / Kaggle free tier, Local RTX 4060 only (chosen)
CHOSEN     : Local RTX 4060 only
EVIDENCE   : The RTX 4060 with all optimisations (feature caching, mixed precision,
             OneCycleLR, torch.compile if available) trains Phase 1 in 30 minutes
             and Phase 2 in 3-3.5 hours. Total training time ≈ 4 hours. Vast.ai
             adds cloud billing complexity, potential data privacy concerns (uploading
             10,000+ disease images to a remote server), and network dependency.
             The local hardware is sufficient. Cloud is never the answer for this
             project.
REVISIT IF : Phase 2 training time consistently exceeds 8 hours due to unexpected
             data scale. Then and only then consider Kaggle free tier (not Vast.ai).
STATUS     : ACTIVE — LOCKED
```

---

**DECISION-33**
```
DECISION   : Virtual environment required — no system Python installs
DATE       : Design phase (gap fix)
CONTEXT    : Python packages for this project (torch, timm, albumentations) conflict
             with common system package versions and require specific version pinning.
CONSIDERED : System Python install, conda environment, venv (chosen)
CHOSEN     : venv — python -m venv venv
EVIDENCE   : Installing torch 2.2.0, albumentations 1.4.4, and their dependencies
             system-wide risks breaking other Python projects on the same machine.
             Windows without admin rights cannot install to some system Python
             locations. venv is built into Python 3.10+, requires no additional
             installation, and creates an isolated environment at venv/ in the
             project root. setup_project.py verifies that a virtual environment
             is active and exits with instructions if not.
REVISIT IF : Never. Always use a virtual environment.
STATUS     : ACTIVE
```

---

**DECISION-34**
```
DECISION   : training/helpers.py is the ONLY location for EarlyStopping,
             save_checkpoint, load_checkpoint, cleanup_old_checkpoints, get_llrd_optimizer
DATE       : Design phase (gap fix)
CONTEXT    : Earlier spec versions had these utilities defined in both
             04_train_phase1.py AND in training/helpers.py, causing duplication
             and potential divergence.
CONSIDERED : Define in each training script, Define in helpers only (chosen),
             Define in 04_train_phase1.py and import from there in 05_train_phase2.py
CHOSEN     : training/helpers.py only — import at module level in both scripts
EVIDENCE   : Duplicating function definitions in two files means any bug fix or
             improvement must be made in two places. Importing 04_train_phase1.py
             from 05_train_phase2.py creates a circular import path and pulls in
             Phase 1's model construction code as a dependency of Phase 2.
             training/helpers.py is a shared utility module with no knowledge of
             either training phase — it only contains stateless utilities and the
             LLRD optimizer factory. Both training scripts import from it at MODULE
             LEVEL (top of file, not inside __main__). If imported inside __main__,
             the functions are unavailable when called from train_phase1() or
             train_phase2() during pipeline execution.
REVISIT IF : Never. helpers.py is the single source of truth for shared training
             utilities.
STATUS     : ACTIVE
```

---

**DECISION-35**
```
DECISION   : Phase 2 resumes from latest phase2_epoch*.pt checkpoint if it exists
DATE       : Design phase (gap fix)
CONTEXT    : Phase 2 takes 3-3.5 hours. A machine restart or power failure partway
             through loses all progress and requires restarting from Phase 1 weights.
CONSIDERED : Always restart Phase 2 from Phase 1 weights, Resume from latest
             phase2 checkpoint if available (chosen)
CHOSEN     : Resume from latest phase2_epoch*.pt
EVIDENCE   : A machine restart mid-Phase 2 without resume costs 1.5-3 hours of
             redundant computation. The checkpoint saves full training state: model
             weights, optimizer state, scheduler state, and scaler state. Resuming
             from this gives identical training continuation — the LR schedule
             continues from where it left off, not from the start. The resume logic
             scans for phase2_epoch*.pt files in CKPT_DIR and loads the most recent.
             If none exist, it starts fresh from phase1_best.pt as normal.
REVISIT IF : Never remove the resume logic. Phase 2 resume is a safety feature.
STATUS     : ACTIVE
```

---

## SECTION 7: DECISIONS MADE DURING IMPLEMENTATION

[Claude Code appends entries here as it makes decisions during the project.
Use the same format as above. Never modify existing entries — only add new ones.]

**DECISION-36 (template for Claude Code to fill)**
```
DECISION   :
DATE       :
CONTEXT    :
CONSIDERED :
CHOSEN     :
EVIDENCE   :
REVISIT IF :
STATUS     :
```
