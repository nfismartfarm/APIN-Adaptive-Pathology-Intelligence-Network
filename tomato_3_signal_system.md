# Spec Outline (locked, v2 — after gap audit)

All work happens inside `tomato_sandbox/`. Existing system at port 8766 is untouched. Sandbox server runs on port 8767.

## Part I — Foundation
0. Sandbox Directive (NEW; the most important rule)
1. Document purpose and scope
2. System context (Kerala deployment, NanoFarm, B.Tech project, glossary)
3. Existing assets referenced (APIN, models, prior decisions)
4. Architectural overview (sandbox + existing system map)

## Part II — Tomato Pipeline (CENTERPIECE)
5. Image input and validation gate
6. Image Quality Assessment (IQA)
7. Image preprocessing pipelines
8. Signal A — v3 model (10-class)
9. Signal B — single-pass LoRA (epoch 13)
10. Signal C — PSV (Plant Symptom Visual)
11. Test-Time Augmentation (selective)
12. Hierarchical classifier (Stage 1 + Stage 2 + soft routing)
13. Conformal prediction sets
14. Tier assignment (full priority order, all rules)
15. Decision scenarios (matrix + 100+ prose scenarios)
16. Output schema and response construction
17. Severity grading
18. Multi-image support

## Part III — System Integration
19. Router and crop dispatch (sandbox-side and dispatcher options)
20. APIN HTTP client (sandbox-only; not used in default config)
21. Sandbox server architecture (8767)
22. Frontend extensions (sandbox-served)
23. Agronomist view (sandbox-only, reads sandbox SQLite)
24. SQLite storage layer (sandbox-only)
25. Drift monitoring dashboard (sandbox-only)

## Part IV — Engineering
26. Production hygiene (caching, logging, threading, OOM, GPU lock)
27. Configuration and versioning (model_version definition)
28. Validation gates and procedures
29. Phase plan (A through F.11) — all phases inside sandbox

## Part V — Honest Assessment (NEW; was awkwardly inside Part IV as Section 30)
30. Honest limitations and risk register
31. Data flaws acknowledged (NEW — pulls from PDA review)
32. Operational handoff checklist (NEW — pre-launch config requirements)

## Part VI — Reference Appendices
A. Metric reference — INDEX of every metric (with cross-reference to its body section). Body sections are the source of truth for definitions.
B. Tier scenario decision matrix (full)
C. PSV feature catalog (all 26)
D. tier_rules.yaml example
E. Class index conventions and remap tables
F. File and artifact catalog (with sandbox paths)

## Outline change rationale (gap audit fixes)

- **G17 fix**: Honest limitations and risk register (Section 30) moved out of Part IV (Engineering) into a dedicated Part V (Honest Assessment).
- **G18 fix**: Metric definitions are body-of-spec source of truth (per-section). Appendix A is INDEX/cross-reference only.
- **G19 fix**: Section 31 (Data flaws acknowledged) added — the PDA-review red flags get an explicit home.
- **G20 fix**: Section 32 (Operational handoff checklist) added — pre-launch configuration requirements.
- **Sandbox**: Section 0 added at the top. All file paths in Part II/III/IV are relative to `tomato_sandbox/`.

Total: 32 sections + 6 appendices. 

# Tomato Plant Disease Detection System — Full Specification v1.0

**Project:** Plant disease detection for Kerala agriculture
**Crops covered:** Okra, brassica (broccoli/cabbage), tomato, chilli
**Status:** Final specification, locked design, pre-implementation
**Authors:** Dev (intern, NanoFarm/NFI SmartFarm Solutions Pvt. Ltd.)
**Audit:** 5 rounds per section, 12 rounds full-document. Audit log in separate file. Gap re-read added as separate audit pass after sections lock.

---

## SANDBOX DIRECTIVE — Read First Before Anything Else

**This entire spec describes work that happens inside one isolated directory: `tomato_sandbox/`. Nothing outside that directory is modified.**

### Why a sandbox

The existing system at port 8766 runs okra and brassica reliably. Adding tomato by editing the existing wrapper or replacing the existing server entry point creates a real risk: a bug in tomato code could degrade okra/brassica behavior. The user has therefore directed that all new work for tomato live in a sandbox, completely separate from the existing system.

This is stricter than the wrapper-pattern originally drafted in Section 3.3. The wrapper pattern still imports APIN as a Python library and replaces APIN's server entry point with a new one bound to 8766. That introduces shared-process risk (an exception in tomato code could crash a process that also serves okra/brassica). The sandbox approach eliminates that risk entirely.

### What the sandbox is

A new top-level directory `tomato_sandbox/` containing:
- A new FastAPI server, bound to port **8767** (not 8766)
- All tomato-specific code (pipeline, signals, classifier, tier rules, frontend extensions)
- All tomato-specific data (calibration files, prototype bank, conformal threshold, models)
- All tomato-specific config (`tomato_sandbox/config.py`, `tomato_sandbox/tier_rules.yaml`)
- Its own SQLite database file (`tomato_sandbox/storage/predictions.db`)
- Its own logs directory (`tomato_sandbox/logs/`)
- Its own tests
- Its own README explaining how to start/stop without affecting 8766

### What the sandbox is NOT

- It is **not** a replacement for the 8766 server. The 8766 server keeps running unchanged.
- It does **not** import APIN as a Python library. If sandbox code needs APIN (it doesn't, but for completeness), it calls APIN's 8766 server over HTTP like any other client.
- It does **not** modify any file outside `tomato_sandbox/`.
- It does **not** share any process state with 8766. They are independent OS processes.

### How traffic gets to the sandbox

The user/frontend picks. Three options:
1. **Direct call** — frontend posts tomato images to `localhost:8767/predict` directly, posts okra/brassica to `localhost:8766/predict`. Frontend decides crop based on a separate router call or user selection.
2. **Thin reverse proxy** — a small dispatcher (could be the frontend, could be Nginx, could be a tiny FastAPI router) takes all `/predict` calls, runs the router, and forwards to 8766 or 8767 based on the result. The dispatcher itself does no heavy work.
3. **Manual switch during testing** — user explicitly hits `localhost:8767/predict` to test sandbox in isolation, hits `localhost:8766/predict` for production behavior.

For the duration of this spec's implementation, option 3 is the default. Option 2 is the recommended production cutover path once sandbox is validated. Option 1 is acceptable for development.

### Hard rule

**No file outside `tomato_sandbox/` is modified by any work this spec describes.**

The sacred-files list in Section 2.6 is extended: during sandbox work, the entire repository OUTSIDE `tomato_sandbox/` is sacred. Specifically, this means:
- `scripts/apin/` — sacred (was already)
- `models/` — sacred (was already)
- `app/` — sacred (NEW; previously the spec planned to add `app/tomato/` files; that plan is rescinded)
- `frontend/` (existing) — sacred; tomato frontend code goes in `tomato_sandbox/frontend/`
- `data/` — sacred
- Any existing `wrapper_server.py` or APIN server entry points — sacred
- The 8766 port binding — sacred

If the spec elsewhere lists a file path under `app/tomato/`, `app/agronomist/`, `app/storage/`, `app/monitoring/`, or `app/schemas/`, mentally remap it to `tomato_sandbox/` followed by the same subpath. Section 4.1 (component table) gives the corrected paths explicitly.

### Rollback

Sandbox is fully reversible: `rm -rf tomato_sandbox/` and the system returns to its pre-spec state. No production code was touched, so there is nothing to revert outside the sandbox. This rollback property is the main reason for the sandbox approach.

### How this changes the rest of the spec

- Section 3.3 ("Why a wrapper server instead of modifying APIN") is superseded by this directive. The wrapper-replacing-8766 design is no longer the plan. Section 3.3's reasoning still applies as background ("don't modify APIN") but the implementation is the sandbox, not the wrapper.
- Section 3.4 still applies (router upstream of pipelines), but the router runs inside the sandbox on its own copy of the loaded weights, OR the router is a tiny thin dispatcher upstream of both 8766 and 8767. Section 19 will clarify.
- Section 4.1 (component table) lists corrected paths under `tomato_sandbox/`.
- Section 4.4 (lifecycle) describes the sandbox server's lifecycle. The 8766 server's lifecycle is out of scope for this spec.
- All env vars in Section 4.5 are read by the sandbox server only and do not affect 8766.
- All file paths in Sections 5–25 should be read with `tomato_sandbox/` prepended unless explicitly stated otherwise.

If a later section appears to instruct modification of a file outside the sandbox, that is a spec bug and should be flagged in the next audit round. The sandbox boundary is non-negotiable.

---

# Part I — Foundation

## Section 1. Document Purpose and Scope

### 1.1 What this document is

This is the engineering specification for the tomato plant disease detection system that integrates into the existing 4-crop Kerala agricultural deployment. It is the single source of truth for what the system does, why it does it that way, and how every component connects. Implementation should follow this spec without reinterpretation. Where this spec disagrees with prior project documents, this spec is authoritative for the tomato pipeline; prior documents remain authoritative for okra and brassica (which this spec does not modify).

### 1.2 What this document is not

This is not a research paper. Deep justification for every design choice — alternatives considered, evidence weighed, tradeoffs accepted — lives in `architecture_claude_decisions.md` and the PDA review summary. This spec includes key justifications inline where they are necessary for understanding the design, but it does not exhaust them.

This is also not a deployment runbook. Operational concerns (server provisioning, monitoring alerting rules, on-call rotation) are out of scope. Production hygiene that affects code structure (caching, logging schema, thread safety) is in scope.

### 1.3 Scope of work

In scope:
- The full request/response pipeline for tomato images entering the system at port 8766
- Input validation gate (file format, size, count) before any pipeline runs
- Integration with the existing router and APIN (okra/brassica) without modifying APIN
- Frontend extensions for the tomato details view
- SQLite storage of predictions and feedback
- Agronomist review interface
- Drift monitoring dashboard reading from storage
- Validation procedures and gate definitions
- Phase A through Phase F implementation order

Out of scope:
- Modifications to anything outside `tomato_sandbox/` (per Sandbox Directive)
- Modifications to `scripts/apin/` (sacred, never edited)
- Modifications to `models/best_model.pt` and `models/swin_best_model.pt` (sacred)
- Re-training of v3 or single-pass LoRA models (frozen at the checkpoints specified)
- Chilli pipeline (returns "not deployed" message; future work)
- Mobile app integration (the system is a web/API service; mobile integration is a separate project)
- Cloud deployment and scaling (single-machine deployment is the target)
- Multi-language UI (Kerala has Malayalam and English speakers; this spec assumes English UI; localization is future work)
- Authentication and authorization (sandbox assumes trusted local network; no user accounts, no API keys)
- Rate limiting (sandbox assumes low request volume; LRU cache mitigates duplicate-request floods but is not a rate limiter)
- Backup and restore of SQLite database (operational concern; the file can be copied while sandbox is stopped)
- Production cutover from sandbox to primary (post-spec decision; the sandbox itself is the deliverable)

### 1.4 Document structure

The spec is organized in five parts. Part I (this part) sets context. Part II is the tomato pipeline in deepest detail and is the centerpiece. Part III covers integration with the rest of the system. Part IV covers engineering and validation. Part V provides reference appendices.

Each section has its own audit log entry in `audits/section_NN_audit.md`. The full-document audit is in `audits/full_audit.md`.

### 1.5 Versioning

This is spec version 1.0. Subsequent changes during implementation must be recorded as version bumps with rationale in the architecture decisions log. The spec itself is committed to the repo at `tomato_sandbox/docs/spec_v1.md` (note the path is inside the sandbox, per the Sandbox Directive).

### 1.6 Glossary

Terms used throughout this spec without re-definition. Listed here once.

| Term | Meaning |
|---|---|
| **APIN** | Adaptive Pipeline Intelligence Network. The existing 4-signal stacking ensemble that handles okra and brassica disease prediction at port 8766. Lives in `scripts/apin/`. |
| **canonical index space** | The 7-class ordering used for the final tomato classifier output: 0=foliar, 1=septoria, 2=late_blight, 3=ylcv, 4=mosaic, 5=healthy, 6=OOD. See Section 2.4. |
| **CLS token** | A special "classification" token in a Vision Transformer (ViT) that aggregates information from all image patches. The single-pass LoRA model classifies tomato images by reading only the CLS token (a 768-dimensional vector for DINOv2-Base) of the final transformer layer. CLS tokens are also the basis for prototype-bank similarity matching (Section 9.4). |
| **conformal prediction** | A statistical method that produces a *set* of predicted classes guaranteed (under exchangeability assumptions) to contain the true class with a chosen probability (e.g., 90%). See Section 13. |
| **degraded mode** | An inference mode where one or more signals (v3, LoRA, or PSV) failed but the system still produces a result, with the failed signal's contribution zeroed and the classifier handling the missing input via training-time degraded-mode augmentation. |
| **ECE** | Expected Calibration Error. Measures how closely predicted probabilities match observed accuracies. Lower is better; 0 is perfect calibration. Used as a hard validation gate (target ECE < 0.10). |
| **F.0** | The first sub-phase of Phase F (decision system implementation), dedicated to data-driven calibration of all thresholds, feature standardization parameters, and similar empirical values. F.0 must complete and pass a manual review gate before F.1 begins. See Section 29. |
| **F.0.5** | A held-out feature validation step that runs after F.0. Computes the same features on the 40-image held-out subset and checks distribution shift vs the training subset. See Section 29. |
| **F.10** | The Kerala field photo benchmark. 20–30 photos taken in actual Kerala fields, manually labeled, and run through the full system as an honest "smell test" (not a statistical benchmark). See Section 29. |
| **field_val** | The 203-image internal validation set used heavily for tuning. Split 80/20 in F.0 to produce 160 train_subset + 40 held_out_subset. |
| **final_val** | The 104-image LOCK-4 held-out evaluation set. Single-pass LoRA gets one evaluation here; that's it under the lock. |
| **GLCM** | Gray-Level Co-occurrence Matrix. A texture feature in PSV that measures spatial relationships between pixel intensities. See Section 10. |
| **HardFiLM** | A Feature-wise Linear Modulation layer used by v3 to condition features on the active crop. "Hard" indicates the conditioning is per-class learned scale-and-shift parameters rather than a soft attention mechanism. v3's HardFiLM uses `crop_mode = 2` to enable tomato-conditioned feature pathways. |
| **IQA** | Image Quality Assessment. The 7-dimension quality gate at the start of the tomato pipeline. See Section 6. |
| **letterbox padding** | A way to resize a non-square image to a square target while preserving aspect ratio: scale the image so its longer dimension fits the target, then pad the shorter dimension with a constant value (114 for our LoRA model, matching its training). Distinct from "stretch resize" which distorts aspect ratio. See Section 7.3. |
| **LOCK-4** | A project rule prohibiting more than one held-out evaluation of any given model on the `final_val` set. The lock prevents test-set contamination through repeated evaluation. |
| **LoRA** | Low-Rank Adaptation. A parameter-efficient fine-tuning method used by both v3 and the single-pass LoRA model. Adds small trainable adapters to a frozen backbone. |
| **macro-F1** | Unweighted mean of per-class F1 scores. Treats all classes as equally important regardless of sample count, which is important for our imbalanced data. |
| **MixStyle** | A domain-generalization technique used during v3 training that mixes feature statistics across training domains. At inference (with `domain_labels=None`) it is a no-op. |
| **OOD** | Out-of-distribution. An input that does not represent any of the trained tomato disease classes — could be a different crop, a non-leaf, an unfamiliar disease, or noise. The classifier learns to detect these via OOD-augmented training samples (Stage 1's third class). |
| **prototype bank** | A small library of high-confidence predictions from `field_val`, used to blend single-pass LoRA outputs when LoRA's confidence is low. See Section 9. |
| **PSV** | Plant Symptom Visual. The classical computer vision signal in the tomato pipeline — color constancy preprocessing, segmentation, 26 features, 6 botanical compatibility scores. Section 10 is the full spec. |
| **sandbox / `tomato_sandbox/`** | The isolated directory where all new work for this spec happens. See Sandbox Directive at top of document. |
| **Shades-of-Gray** | A color constancy algorithm (Finlayson & Trezzi 2004) that estimates the scene illuminant using the Minkowski p-norm of pixel values, then divides each channel by its illuminant estimate. With p=1 it equals grey-world; with p=∞ it equals max-RGB. We use p=6, which empirically reduces hue variance across illuminants by ~20× compared to no preprocessing. See Section 7.4. |
| **soft routing** | The way Stage 1 and Stage 2 of the hierarchical classifier are combined. Stage 1's diseased probability multiplies Stage 2's per-disease probability rather than gating it hard. P_final[disease_i] = P_stage1[diseased] × P_stage2[disease_i]. See Section 12. |
| **Squeeze-and-Excitation (SE)** | A lightweight channel-attention block used in v3. After feature extraction, the SE block produces per-channel scaling weights, allowing the model to up-weight or down-weight channels based on global context. |
| **tier** | The system's confidence-and-action category for a prediction. See Section 3.11 and Section 14. |
| **TTA** | Test-Time Augmentation. Selective application of viewpoint-preserving augmentations at inference to reduce uncertainty. See Section 11. |
| **v3** | The v3 model (also "Model 3"), a DINOv2-Small + LoRA + FiLM 10-class classifier covering tomato + chilli. See Section 8. |
| **val_sqrtn** | The training-time validation split used during single-pass LoRA training. F1 of 0.9113 was measured on this split. Distinct from the locked `final_val`. |

---

## Section 2. System Context

### 2.1 Deployment context

The system is being built as part of a B.Tech final-year project at S Devkrishna's institution, completed during an AI/ML internship at NFI SmartFarm Solutions Pvt. Ltd. (operating as NanoFarm), Kalamassery, Kerala. The intended end-users are Kerala farmers (or extension officers operating on their behalf) who photograph plant leaves with smartphones to receive disease diagnosis.

Field conditions in Kerala that the system must accommodate:
- Tropical humid climate with monsoon season (heavy rain June through September)
- Wet leaves common after rain
- Variable lighting: bright midday sun, overcast afternoons, dappled light under canopy
- Variable smartphone camera quality (older Android devices through current flagships)
- Occasional spotty network connectivity in rural areas
- Domain experts (agronomists) available for review but not for every diagnosis. Working assumption: one or two agronomists allocate roughly 30–50 cases per day for review through the agronomist view (Phase D, Section 23). The priority queue is sized to bring the most informative cases (high uncertainty or underpowered classes) to the top so that a small daily review budget produces maximum learning value. If actual agronomist throughput differs in deployment, the queue depth and prioritization heuristics may need tuning.

### 2.2 Crop coverage and disease classes

Four crops, 19 active disease/healthy classes total (after class remapping from the original 23-class superset).

**Okra** — handled by APIN (existing, frozen):
- okra_healthy
- okra_yellow_vein_mosaic_virus (YVMV)
- okra_cercospora_leaf_spot
- okra_powdery_mildew
- okra_enation

**Brassica** (broccoli, cabbage) — handled by APIN (existing, frozen):
- brassica_healthy
- brassica_alternaria_leaf_spot
- brassica_black_rot
- brassica_downy_mildew
- brassica_white_rust

**Tomato** — handled by the new pipeline this spec covers. Each disease is given a canonical short name (the bolded abbreviation) used throughout the rest of the spec.

- `tomato_healthy` (short: **healthy**)
- `tomato_foliar_spot` (short: **foliar**) — combines bacterial spot, early blight, and target spot. These three diseases were merged into one class because the available training data could not reliably distinguish them: their early-stage symptoms overlap visually (small dark spots on leaves with surrounding chlorosis), and the labeled images per disease were insufficient to learn the subtle differentiating features without overfitting to dataset-specific artifacts. The combined class is treated as a single "foliar spot" diagnosis with treatment recommendations that work for all three. Detailed rationale in `architecture_claude_decisions.md` entry 14.
- `tomato_septoria_leaf_spot` (short: **septoria**)
- `tomato_late_blight` (short: **late_blight**)
- `tomato_mosaic_virus` (short: **mosaic**, also known as TMV/ToMV in plant pathology literature)
- `tomato_yellow_leaf_curl_virus` (short: **YLCV**)

Throughout this spec the short names are used in tier code identifiers (e.g., `5_lateblight`, `5_mosaic`, `5_ylcv`) and in scenario tables. The full Latin/full-disease names are reserved for the formal output schema (Section 16) and any user-facing display.

**Chilli** — not yet deployed, returns "not deployed" message:
- chilli_healthy
- chilli_leaf_curl
- chilli_cercospora
- chilli_anthracnose

### 2.3 Existing assets that this spec relies on

The new tomato pipeline does not stand alone. It relies on the following existing assets, which this spec does not modify:

**Crop Router** — Already deployed. EfficientNetV2-S based 4-class classifier (okra, brassica, tomato, chilli). Macro-F1 0.9862 on held-out validation. Decides which crop pipeline to dispatch the request to.

**APIN (Adaptive Pipeline Intelligence Network)** — Existing okra/brassica system. 4-signal stacking ensemble with smart thresholds, GradCAM++ visualization, and tier-1 post-processing. Field F1 0.9424 on held-out evaluation. APIN's code lives in `scripts/apin/` and is sacred — the new wrapper imports APIN as a library but does not modify any file within `scripts/apin/`.

**v3 model (Model 3)** — Trained tomato+chilli specialist. DINOv2-Small with registers + LoRA rank 4 + Squeeze-and-Excitation + MixStyle + HardFiLM (crop conditioning) + Linear classification head. 10-class output: 6 tomato + 4 chilli. Input 224×224 with stretch resize. Weights at `scripts/model3_training/checkpoints/model3_production_v3.pt`. The model takes a `crop_mode` tensor at inference (we will always pass `torch.tensor([2])` for tomato).

**Single-pass LoRA epoch 13** — Trained on tomato 6-class only. DINOv2-Base with registers (frozen) + LoRA adapters on transformer blocks 4 through 11 + Linear(768→6) head on the CLS token. Input 392×392 with letterbox padding (pad value 114). Validation F1 of 0.9113 on the val_sqrtn split (training-time validation set, distinct from the locked final_val held-out set). Weights at `models/specialist/sp_lora_checkpoints/sp_lora_epoch13_f10.9113_PRESERVED.pt`. Production artifact will be renamed to `models/specialist/tomato_sp_lora_production.pt`.

**Existing FastAPI server at port 8766** — Currently runs APIN. After this work, the wrapper IS the server bound to port 8766; APIN's previous server entry point (`scripts/apin/section8_apin_server.py` in standalone mode) is no longer used. APIN's prediction code is imported as a Python library and called by the wrapper. Port number unchanged to avoid breaking the frontend.

**LADI-Net Phase 1 checkpoint** — `ladinet_phase1_heads.pt`. Research artifact. NOT deployed. Phase 1 reached val_sqrtn_F1 0.9112 but final_val 0.7958 (LOCK-4 evaluation). Listed here for completeness so it is clear this asset exists in the repo but is not used in production.

**LADI-Net Phase 2** — Failed (best F1 0.8662, below Phase 1 baseline). Documented as research history. Not used.

### 2.4 Critical class index conventions

Three class index spaces are used. Confusing them produces silent miscategorization, so they are spelled out here.

**v3 model output (10 classes, fixed):**
| Index | Name |
|---|---|
| 0 | tomato_foliar_spot |
| 1 | tomato_late_blight |
| 2 | tomato_septoria_leaf_spot |
| 3 | tomato_yellow_leaf_curl_virus |
| 4 | tomato_mosaic_virus |
| 5 | tomato_healthy |
| 6 | chilli_leaf_curl |
| 7 | chilli_healthy |
| 8 | chilli_cercospora |
| 9 | chilli_anthracnose |

**Single-pass LoRA output (6 classes, fixed):**
| Index | Name |
|---|---|
| 0 | tomato_foliar_spot |
| 1 | tomato_septoria_leaf_spot |
| 2 | tomato_late_blight |
| 3 | tomato_yellow_leaf_curl_virus |
| 4 | tomato_mosaic_virus |
| 5 | tomato_healthy |

**Note: v3 index 1 is late_blight, but LoRA index 1 is septoria.** Different orderings for historical training reasons. The remapping tensor used everywhere in the code is:

```python
V3_INDEX_FOR_LORA_CLASS = [0, 2, 1, 3, 4, 5]
# Reads as: LoRA class 0 corresponds to v3 index 0,
#           LoRA class 1 corresponds to v3 index 2,
#           LoRA class 2 corresponds to v3 index 1,
#           LoRA class 3 corresponds to v3 index 3, etc.
```

Because the only swap is between positions 1 and 2 (late_blight ↔ septoria), this array is its own inverse. The same tensor maps both directions:

```python
LORA_INDEX_FOR_V3_CLASS = [0, 2, 1, 3, 4, 5]  # identical, by coincidence of the swap structure
```

**Worked example.** Suppose LoRA outputs probabilities `[0.1, 0.6, 0.1, 0.05, 0.05, 0.1]`. The argmax is index 1, which is `tomato_septoria_leaf_spot` in LoRA's space. To find the v3 probability for septoria, look up `V3_INDEX_FOR_LORA_CLASS[1] = 2`, so the corresponding v3 probability is at v3 index 2.

Conversely, suppose v3 outputs argmax at index 1. Index 1 in v3 space is `tomato_late_blight`. To find this in LoRA's space, look up `LORA_INDEX_FOR_V3_CLASS[1] = 2`, so the corresponding LoRA probability is at LoRA index 2.

This remap is by class name, not by position. Anywhere the code aligns v3 outputs with LoRA outputs, it must use this tensor. Mistakes here silently mislabel late blight as septoria and vice versa — the model accuracy looks fine but specific predictions are systematically wrong on these two classes.

**Index-position hazard for "healthy".** The class `tomato_healthy` lives at:
- v3 output index 5
- LoRA output index 5
- Canonical index 5

But `chilli_healthy` lives at v3 output index 7. A common bug pattern is "find the healthy class by looking for the word 'healthy' in the class names" — this returns multiple matches across the v3 output. Code that selects the tomato-healthy probability from v3 output must use index 5 specifically, not name-based lookup. Conversely, when v3 outputs `chilli_healthy` (index 7) with high probability for a tomato request, that is a misrouting signal that feeds into `chilli_leakage` (Section 8).

**Internal canonical index space (the FINAL output space of the tomato classifier after soft routing, 7 classes):**
| Index | Name |
|---|---|
| 0 | tomato_foliar_spot |
| 1 | tomato_septoria_leaf_spot |
| 2 | tomato_late_blight |
| 3 | tomato_yellow_leaf_curl_virus |
| 4 | tomato_mosaic_virus |
| 5 | tomato_healthy |
| 6 | OOD (out-of-distribution) |

Note: the per-stage output spaces are smaller than this canonical 7-class space. Stage 1 outputs 3 classes (healthy / diseased / OOD). Stage 2 outputs 5 classes (the 5 disease classes). The 7-class canonical space is constructed by combining the two stages via soft routing (Section 12). Internal canonical matches LoRA's ordering (0 = foliar, 1 = septoria, 2 = late_blight, 3 = ylcv, 4 = mosaic, 5 = healthy) with OOD appended as index 6. v3 outputs are mapped to this space using the inverse of `V3_INDEX_FOR_LORA_CLASS` (which equals itself, as shown above). PSV compatibility scores are also produced in this canonical ordering for the 6 disease classes (without OOD; OOD is detected by the classifier's Stage 1 output).

### 2.5 Hardware and software environment

Development and inference target a single Windows 11 machine with an NVIDIA RTX 4060 Laptop GPU (8 GB VRAM). Software stack: Python 3.13, PyTorch 2.11, CUDA 13. Project root: `C:\Users\xplod\Videos\FSWD\synod\Plant-disease-detection-for-brocolli-and-ladies-finger\`. The system must fit within 8 GB VRAM with both v3 and single-pass LoRA models loaded simultaneously. No model can require eviction during a normal inference cycle.

VRAM budget (rough):
- DINOv2-Small (v3 backbone): ~120 MB weights
- DINOv2-Base (LoRA backbone): ~360 MB weights
- LoRA adapter weights: ~30 MB combined
- Input batch (3 images at 392px max): ~50 MB
- Activations during forward pass: ~500 MB peak with attention
- Working memory: ~2 GB
- Headroom: remaining (~4–5 GB)

System RAM budget (CPU side, sandbox process only — does not include the existing 8766 server's RAM):
- Python interpreter, PyTorch runtime, FastAPI: ~600 MB baseline
- PyTorch CPU-side tensor allocations during inference (intermediate activations stored on CPU when GPU is between operations): ~500 MB peak
- PSV intermediate arrays per concurrent request (224x224 RGB images, working buffers, GLCM/FFT outputs): ~150 MB
- SQLite cache and connection pool: ~50 MB
- Image decode buffers per image: ~50 MB
- LRU response cache (1000 entries × ~50 KB serialized response): ~50 MB
- OS file cache headroom (read-back of model weights during reload): ~500 MB
- Concurrent request peak headroom (4 simultaneous requests at 200 MB each working set): ~800 MB
- General process headroom and fragmentation buffer: ~1 GB

Steady-state target: 2–3 GB during normal inference (1–2 concurrent requests).
Peak target: 4–6 GB during burst load (4+ concurrent requests with PSV in flight on multiple of them).

The earlier headroom-vs-component arithmetic gap is resolved by the explicit listing of OS file cache, concurrent peak headroom, and PyTorch CPU buffers above. The 4–6 GB target is for peak; steady-state will be substantially lower.

The 8766 server runs as a separate process and has its own RAM budget (not specified here; that is an existing system concern). On a 16 GB development laptop, the two servers running concurrently with peak loads would use approximately 6 + 4 = 10 GB, leaving comfortable headroom for OS and IDE.

This budget assumes single-image or 3-image-batch inference. Larger batches require eviction or batched processing. PSV (CPU-bound) memory does NOT compete with VRAM but does compete with system RAM, so concurrent PSV computations are bounded by available system memory.

### 2.6 Sacred files

These files MUST NOT be modified by any code change implementing this spec. The spec depends on their existing behavior.

**Per the Sandbox Directive at the top of this document, the entire repository outside `tomato_sandbox/` is sacred during the implementation of this spec.** The table below enumerates the most important files in that sacred set, but the rule is broader than the table: anything not under `tomato_sandbox/` is off-limits.

| File | What it is | Why sacred |
|---|---|---|
| `scripts/apin/` (entire directory) | APIN okra/brassica system | Production traffic depends on byte-identical behavior |
| `models/best_model.pt` | 23-class EfficientNetV2-S (legacy fallback) | Used by APIN as Mode 1 fallback |
| `models/swin_best_model.pt` | Swin-Tiny 23-class (legacy) | APIN dependency |
| `model2_production.pt` | DINOv3-ConvNeXt-Small | APIN's main backbone |
| `data/specialist/model3/split_indices.json` | Train/val split index file | Reproducibility of v3 evaluation |
| `app/config.py` | Application config (existing system) | Not edited by sandbox; sandbox uses its own `tomato_sandbox/config.py` |
| `data/metadata/source_map.csv` | Data provenance | Reproducibility |
| `models/specialist/ladinet_checkpoints/ladinet_phase1_heads.pt` | LADI-Net Phase 1 weights | Research history, may be referenced |
| The 8766 port binding | Existing server | Sandbox uses 8767 instead |

**Tomato production model artifact** lives inside the sandbox: `tomato_sandbox/models/tomato_sp_lora_production.pt`. After Phase A.3 final evaluation locks it in, this file becomes write-once: do not retrain or overwrite without bumping the spec version.

The previous version of this section listed `app/config.py` as sacred while implying the spec would edit it. That contradiction is resolved by the sandbox directive: existing `app/config.py` is sacred (read-only); the sandbox has its own config file under `tomato_sandbox/config.py` that the sandbox server reads.

---

## Section 3. Architectural Overview

### 3.1 The full system at one glance

Two independent FastAPI servers run on the same machine:
- **Port 8766** — existing system, unchanged. Handles okra and brassica via APIN.
- **Port 8767** — new sandbox server. Handles tomato. Returns "not deployed" for chilli.

A frontend or thin reverse-proxy sends each request to the correct server based on either user selection or a router pre-call. SQLite (sandbox's own DB) logs tomato predictions for monitoring and feedback.

```
                                 ┌────────────────────────┐
                                 │   Frontend (existing)  │
                                 │   - upload UI          │
                                 │   - tomato_details     │
                                 │     collapsible (NEW,  │
                                 │     served by sandbox) │
                                 └──────────┬─────────────┘
                                            │
                              ┌─────────────┴────────────┐
                              │                          │
                              │ HTTP POST                │ HTTP POST
                              │ (okra/brassica)          │ (tomato)
                              ▼                          ▼
                       ┌────────────────┐       ┌─────────────────┐
                       │ EXISTING 8766  │       │ SANDBOX 8767    │
                       │ APIN server    │       │ (NEW)           │
                       │ unchanged      │       │ tomato_sandbox/ │
                       │                │       │  server.py      │
                       └────────────────┘       └────────┬────────┘
                                                         │
                                          ┌──────────────┼──────────────┐
                                          │              │              │
                                          ▼              ▼              ▼
                                  ┌────────────┐ ┌──────────┐ ┌──────────────┐
                                  │  Router    │ │ Tomato   │ │ "not         │
                                  │ (loads     │ │ Pipeline │ │  deployed"   │
                                  │  existing  │ │  (NEW)   │ │  response    │
                                  │  weights   │ │          │ │  (chilli)    │
                                  │  in-proc)  │ │          │ │              │
                                  └─────┬──────┘ └────┬─────┘ └──────────────┘
                                        │             │
                                        │ active_crop │ tomato response
                                        │             │
                                        ▼             ▼
                                   crop=tomato/   schema-conformant
                                   chilli         JSON
                                  (no okra/
                                   brassica;
                                   they go to
                                   8766 instead)

                              ┌──────────────────────────┐
                              │ Sandbox SQLite           │
                              │ tomato_sandbox/          │
                              │   storage/predictions.db │
                              │ (Phase E)                │
                              │ - predictions            │
                              │ - tomato_details         │
                              │ - user_feedback          │
                              └──────────────┬───────────┘
                                             │
                              ┌──────────────┴────────────┐
                              │  Sandbox Agronomist view  │
                              │  (Phase D)                │
                              │  /agronomist on 8767      │
                              │  priority queue           │
                              └───────────────────────────┘
                                             │
                              ┌──────────────┴────────────┐
                              │  Sandbox Drift dashboard  │
                              │  /monitoring on 8767      │
                              │  (Phase E)                │
                              └───────────────────────────┘
```

The 8766 server has its own SQLite (if any) and observability; that is separate from sandbox concerns and out of scope for this spec.

### 3.2 Request lifecycle

A single tomato request follows this lifecycle. (Okra and brassica requests follow APIN's existing 8766 lifecycle, which is unchanged.)

1. **Client uploads image(s)** to `POST localhost:8767/predict` (the sandbox port) as multipart form data. Field notes are NOT part of the tomato API; they are silently ignored if sent.
2. **Sandbox server validates input.** Checks file format, size, count. Rejects unrecoverable inputs immediately with HTTP 400.
3. **Sandbox invokes Router.** Router predicts crop probabilities. Top-1 class is `active_crop`.
4. **Sandbox dispatches by crop.**
   - tomato → calls the new TomatoPipeline.infer() method.
   - okra/brassica → returns HTTP 400 "Wrong endpoint; for okra/brassica use port 8766." (The sandbox will not silently re-route to 8766; the user/frontend is expected to call the right endpoint.) An optional thin reverse-proxy upstream of both servers can do this dispatch transparently; see Sandbox Directive option 2.
   - chilli → returns HTTP 200 with a "not deployed" response (friendly message, not an error).
5. **Tomato pipeline produces a response.** Response schema is defined in Section 16. Includes `tomato_details` extension.
6. **Sandbox logs to SQLite** (if storage enabled, Phase E).
7. **Sandbox returns JSON** to client.
8. **Frontend renders** the response. Existing UI handles the base schema; tomato collapsible (sandbox-side frontend code) handles the extension.

The 8766 server remains unaffected by any of this. Its lifecycle for okra/brassica is unchanged.

### 3.3 Why a sandbox instead of modifying APIN or replacing its server

APIN at port 8766 is in production and works. There were three plausible options for adding tomato:

1. **Modify APIN directly.** Add a tomato handler inside `scripts/apin/`. Rejected: violates the sacred-files rule and risks breaking okra/brassica.

2. **Wrapper server pattern.** Replace APIN's server entry point with a new wrapper that imports APIN as a Python library and adds tomato dispatch, all on the same port 8766. Rejected (after initial inclusion in earlier drafts): a bug or memory leak in tomato code can crash the process that also serves okra/brassica. Shared-process risk.

3. **Sandbox.** A new server on port 8767 in its own directory, completely separate process. Chosen, per Sandbox Directive at top of document.

The sandbox approach has these properties:
- APIN's code is not imported into the sandbox process. The sandbox has zero shared state with APIN.
- APIN's server at 8766 is unchanged.
- A bug in sandbox code cannot crash 8766.
- Rolling back tomato is just `rm -rf tomato_sandbox/`. Nothing else to revert.
- Re-deploying APIN updates does not require re-deploying tomato, and vice versa.

The cost: requests are routed by external means (frontend, reverse proxy, or operator), not transparently inside one server. This is a small UX consideration and a small operational consideration. It is paid in exchange for full isolation.

### 3.4 Where the router lives

The router is needed to decide whether a given uploaded image is okra, brassica, tomato, or chilli. With two servers (8766 and 8767), there are two reasonable places to put the router invocation:

**Option A — Router inside each server.** Both 8766 and 8767 load the router weights and run the router on every request. The server then handles or rejects based on the predicted crop. Disadvantage: doubled memory footprint for the router weights, and inconsistency risk if 8766's router and 8767's router get out of sync.

**Option B — Router in a thin upstream dispatcher.** A small reverse proxy or front-end-side dispatcher runs the router and forwards each request to the correct server. Disadvantage: introduces a new component.

**Sandbox-only stance (the choice for this spec).** During sandbox implementation, the router runs inside the sandbox at 8767. The sandbox loads the router weights at startup and uses the router to confirm tomato (and to reject okra/brassica/chilli with a friendly redirect message). The 8766 server runs without a router because it knows the request is for okra/brassica (the user/frontend chose that endpoint). Memory footprint cost is minimal — the EfficientNetV2-S router is small.

For full production, Option B (thin dispatcher) is recommended once the sandbox is validated. The dispatcher would replace the per-server router, sit upstream of both 8766 and 8767, and route based on a single router pass. But that is a post-spec deployment decision, not part of this document's scope.

### 3.5 Why three signals for tomato (and not for okra/brassica)

APIN uses 4 signals for okra/brassica. The tomato pipeline uses 3. Why the difference?

APIN's signals are: EfficientNetV2-S (legacy), Swin-Tiny (legacy), ConvNeXt-Small (DINOv3), and DINOv2 nonlinear head probe. These were what existed when APIN was built. The 4-signal stacking adds value on okra/brassica because the underlying models have diverse error profiles.

For tomato, we have two trained models (v3 and single-pass LoRA) plus PSV. Adding a fourth signal would require either retraining a model (slow, breaks LOCK-4 on the held-out evaluation) or using one of APIN's models on tomato data (which it wasn't trained for and would likely produce noise). Adding a 4th signal in the future remains possible but is deferred for two reasons: (1) it requires breaking LOCK-4 to re-evaluate the new ensemble on held-out data, and (2) the marginal benefit of a 4th signal beyond v3+LoRA+PSV is unproven for tomato. The 3-signal design is what we have, not what is theoretically optimal.

The tradeoff is honest: APIN benefits from the 4th signal because of its history. Tomato gets fewer signals but PSV is engineered specifically for the tomato disease characteristics where neural-only systems struggle (mosaic vs foliar boundary, viral chlorosis patterns).

### 3.6 Why hierarchical classifier instead of flat 7-class

The classifier inside the tomato pipeline is hierarchical:
- Stage 1: 3-way (healthy / diseased / OOD)
- Stage 2: 5-way (within diseased: foliar / septoria / late_blight / mosaic / ylcv)
- Combination: soft routing, P_final[disease_i] = P_stage1[diseased] × P_stage2[disease_i]

Empirical validation in `pda_review` (Round 4 tests, file `round4_validation.py`) showed +0.19 macro-F1 advantage of hierarchical over flat 7-class on simulated field_val-shape data, winning 10 of 10 random seeds. The advantage is data-shape-specific: when one class (healthy) dominates training, a hierarchical decomposition lets each stage solve a more balanced subproblem.

**Caveat on the empirical claim.** The simulation used Gaussian-distributed feature vectors mimicking the class-imbalance shape of field_val (203 samples across 7 classes with the expected count distribution). Real field_val features may not be Gaussian-distributed; in particular, the v3 and LoRA softmax outputs have bounded support [0, 1] and PSV features have mixed continuous and discrete support. The +0.19 advantage on simulated data is evidence for the architectural choice but the magnitude on real data is pending Phase F.0 measurement. If real-data advantage is below +0.05 macro-F1, the hierarchical architecture should be reconsidered.

Risk: Stage 1 misclassification cascades into Stage 2 input. Mitigated by soft routing (Stage 2 always contributes, weighted by Stage 1 confidence).

### 3.7 Why PSV has zero learned parameters in compatibility scoring

PSV's compatibility scores use fixed botanical weights. They are not learned from data. PSV produces 6 compatibility scores, **always in the canonical order from Section 2.4**:

| Canonical index | Score |
|---|---|
| 0 | c_foliar |
| 1 | c_septoria |
| 2 | c_late_blight |
| 3 | c_ylcv |
| 4 | c_mosaic |
| 5 | c_healthy |

This ordering must match canonical exactly. Implementations that emit PSV scores in a different order (for example, alphabetical, or grouping fungal/viral together) will produce silently wrong results when those scores are concatenated with v3 and LoRA outputs into the classifier input. Anywhere a sequence of 6 PSV scores appears in this spec, the order is the canonical one above.

Why fixed weights: with 200 training samples and 26 PSV features, learning compatibility weights from data would overfit. Plant pathology literature provides strong priors on which features should weight which diseases (e.g., margin yellowing is a YLCV signature; high lesion count with small sizes is septoria). The classifier (Stage 2) does learn how to combine PSV's compatibility scores with v3 and LoRA outputs. So PSV's weights are fixed but the system as a whole learns how much to trust PSV.

### 3.8 Why SQLite instead of a real database

SQLite is the right choice for this deployment because:
- Single-machine target (no need for concurrent write scaling)
- Zero infrastructure (file-based, no server to run)
- Sufficient throughput for expected request rate (under 100 req/min)
- Easy to inspect with command-line tools

If deployment scales to multi-machine, SQLite would be replaced with PostgreSQL. The schema is database-agnostic.

### 3.9 Why monitoring lives downstream of storage

The drift monitoring dashboard reads from SQLite. It does not see live requests. This means:
- Monitoring cannot block requests (good — keeps inference fast)
- Monitoring lag is approximately the SQLite write lag (typically under 100 ms)
- If storage fails, monitoring dies but inference does not (graceful degradation)

### 3.10 Failure isolation

Each component fails into a defined state without cascading. Specifically:

| Component / cause | Failure mode | Behavior |
|---|---|---|
| Image upload | Network drop mid-upload, partial file received | Sandbox returns HTTP 400 "Upload incomplete; retry." |
| Input validation | Unsupported format (HEIC, RAW), corrupt JPEG, oversized (>20 MB) | Sandbox returns HTTP 400 with specific reason |
| Router | Exception during forward pass | Sandbox returns HTTP 500 with clear message |
| APIN HTTP client (if used) | 8766 server unreachable, timeout, 5xx | Sandbox falls back to refusing the request with clear "okra/brassica service unavailable" message; tomato requests are unaffected |
| IQA | Exception during quality computation (rare) | Mark IQA as failed; downgrade to Tier 3B with "image quality could not be assessed" |
| Tomato Signal A (v3) | Forward pass throws | Pipeline marks `signal_a_status="failed"`, classifier handles via degraded-mode training |
| Tomato Signal B (LoRA) | Forward pass throws | Same as above for `signal_b_status` |
| Tomato Signal C (PSV) | Segmentation or feature extraction throws | All PSV reliabilities set to 0.1, classifier sees PSV inputs but heavily downweighted |
| TTA | Augmentation library throws (e.g., albumentations error on weird image) | Skip TTA for this request; log warning; use base predictions |
| Tomato classifier | Returns NaN or invalid prob | Force Tier 4B, "all signals fail" message |
| Conformal threshold | File missing/corrupt at startup | Server fails to start (per Section 4.4) |
| Conformal threshold | File modified during runtime (operator error) | Lazy-detected on next request; log warning; use threshold from memory; do not fail |
| SQLite | Write fails (disk full, permissions, lock timeout) | Logged warning, response still returned to user. If disk full persists, monitoring dashboard surfaces "storage degraded" status. |
| SQLite | DB file becomes too large (>1 GB) | Background log-rotation script (operational, out of spec scope) archives old predictions. Sandbox does not break. |
| CUDA OOM | Any forward pass | Mark pipeline `needs_reload`, return Tier 4B with retry suggestion. Lazy reload on next request. |
| GPU lock timeout | Request waited longer than `TOMATO_GPU_LOCK_TIMEOUT_S` for GPU | Return HTTP 503 "Server busy; retry shortly." |

The system never crashes the sandbox server due to a single image's problems. Worst case for any given request is a Tier 4B response or HTTP 4xx/5xx with a clear message. The 8766 server is fully isolated and is unaffected by any sandbox failure.

SQLite atomicity: writes are individual single-row inserts wrapped in transactions. The schema (Section 24) is structured so that all per-prediction data fits in a single row across two tables (`predictions` and `tomato_details`) committed in one transaction. The database runs in WAL (write-ahead logging) mode for concurrent reads during writes. Mid-transaction crashes leave the database in a consistent state because uncommitted transactions are discarded on restart.

### 3.11 What is a "tier"

Throughout this document the term "tier" refers to the system's confidence-and-action category for a prediction. A tier is a final-stage label assigned to each prediction that captures three things: (1) how confident the system is, (2) what level of urgency or caution the response carries, and (3) what action prompt the user receives in the UI. Higher tier numbers generally mean more uncertain or more cautious. Tier 5 is reserved as an override for urgent disease alerts (late blight, mosaic, YLCV) regardless of the normal confidence scale, because these diseases warrant immediate user attention even when the system is only moderately confident.

This is the conceptual introduction of "tier" so that earlier sections can use the term without being self-referential. The full tier hierarchy with all 15+ codes, the priority order for assignment, and the per-tier UI behavior are defined formally in Section 14. Section 15 walks through 100+ decision scenarios showing exactly which tier fires for each combination of signal outputs. For document navigation, see Section 4.8.

---

## Section 4. Architectural Overview — Full System Map

### 4.1 Components and their owners

All paths below are inside `tomato_sandbox/`. The 8766 server's components are not relisted here because they exist already.

| # | Component | New or existing | Spec section | File path (under `tomato_sandbox/`) |
|---|---|---|---|---|
| 1 | Sandbox FastAPI server | New | Sec 21 | `server.py` |
| 2 | Input validation gate | New | Sec 5 | `input_validation.py` |
| 3 | Router invocation | New (loads existing weights) | Sec 19 | `routing/router_loader.py` |
| 4 | APIN HTTP client (used only if sandbox needs APIN; default: not used) | New (thin client) | Sec 20 | `clients/apin_client.py` |
| 5 | Tomato pipeline orchestrator | New | Sec 5–18 | `tomato_pipeline.py` |
| 6 | IQA module | New | Sec 6 | `iqa.py` |
| 7 | Preprocessing module | New | Sec 7 | `preprocessing.py` |
| 8 | Signal A (v3) | New | Sec 8 | `signals/v3_signal.py` |
| 9 | Signal B (LoRA) | New | Sec 9 | `signals/lora_signal.py` |
| 10 | Signal C — PSV orchestrator | New | Sec 10 | `signals/psv/psv.py` |
| 11 | PSV leaf segmentation | New | Sec 10.3 | `signals/psv/leaf_segmentation.py` |
| 12 | PSV disease region detection | New | Sec 10.4 | `signals/psv/disease_detection.py` |
| 13 | PSV feature extractor (26 features) | New | Sec 10.5 | `signals/psv/features.py` |
| 14 | PSV compatibility scorer | New | Sec 10.6 | `signals/psv/compatibility.py` |
| 15 | PSV reliability assessment | New | Sec 10.7 | `signals/psv/reliability.py` |
| 16 | TTA controller | New | Sec 11 | `tta.py` |
| 17 | Hierarchical classifier (Stage 1 + Stage 2 + soft routing) | New | Sec 12 | `classifier.py` |
| 18 | Conformal prediction module | New | Sec 13 | `conformal.py` |
| 19 | Tier assignment | New | Sec 14 | `tier_assignment.py` |
| 20 | Tier rules YAML | New | Sec 14, App D | `config/tier_rules.yaml` |
| 21 | Severity grading module | New | Sec 17 | `severity.py` |
| 22 | Multi-image controller | New | Sec 18 | `multi_image.py` |
| 23 | Response builder | New | Sec 16 | `response_builder.py` |
| 24 | Response schemas (Pydantic) | New | Sec 16 | `schemas/response.py` |
| 25 | Frontend extensions | New | Sec 22 | `frontend/tomato_details.js`, etc. |
| 26 | Agronomist view | New | Sec 23 | `agronomist/` (multi-file) |
| 27 | Storage layer (SQLite) | New | Sec 24 | `storage/db.py`, `storage/schema.sql` |
| 28 | Monitoring dashboard | New | Sec 25 | `monitoring/` (multi-file) |
| 29 | Sandbox config | New | Sec 27 | `config.py` |
| 30 | F.0 calibration outputs (directory) | New | Sec 28, 29 | `phase_f0_calibration/` |
| 30a | — IQA thresholds | New | Sec 6.4 | `phase_f0_calibration/iqa_thresholds.json` |
| 30b | — Prototype bank | New | Sec 9.4 | `phase_f0_calibration/prototype_bank.npz` |
| 30c | — Conformal threshold τ | New | Sec 13 | `phase_f0_calibration/conformal_tau.json` |
| 30d | — Stacking classifier weights | New | Sec 12 | `phase_f0_calibration/classifier_stage1.pkl`, `classifier_stage2.pkl` |
| 30e | — Reliability matrix | New | Sec 12 | `phase_f0_calibration/reliability_matrix.json` |
| 30f | — Threshold sweep results | New | Sec 4.5, 29 | `phase_f0_calibration/threshold_sweep_results.json` |
| 30g | — PSV feature standardization | New | Sec 10.6 | `phase_f0_calibration/psv_standardization.json` |
| 30h | — PSV disease detection threshold | New | Sec 10.4 | `phase_f0_calibration/psv_disease_threshold.json` |
| 30i | — JSD sentinel value | New | Sec 11.5 | `phase_f0_calibration/jsd_sentinel.json` |
| 30j | — Sandbox config (human-readable) | New | Sec 0, 10.6 | `config/` directory |
| 30k | — — Tier rules | New | Sec 14 | `config/tier_rules.yaml` |
| 30l | — — PSV compatibility weight matrix | New | Sec 10.6 | `config/psv_weights.yaml` |
| 31 | Production hygiene utils (cache, logging, GPU lock) | New | Sec 26 | `utils/` (multi-file) |
| 32 | Sandbox README | New | Sec 21 | `README.md` |
| 33 | Sandbox tests | New | Sec 28 | `tests/` (multi-file) |
| 34 | Prototype bank manager | New | Sec 9 | `signals/prototype_bank.py` |

### 4.2 Component dependency graph

Dependencies among new components (existing 8766/APIN excluded; everything below lives in `tomato_sandbox/`):

```
Sandbox server (port 8767)
├─ depends on: Input validation gate (file format, size, count)
├─ depends on: Router invocation (loads existing weights read-only)
└─ depends on: Tomato pipeline orchestrator
                ├─ depends on: Input validation (already passed at sandbox-server level; pipeline trusts it)
                ├─ depends on: IQA module
                ├─ depends on: Preprocessing module
                ├─ depends on: Signal A (v3)
                ├─ depends on: Signal B (LoRA)
                │   └─ depends on: Prototype bank (loaded at startup)
                ├─ depends on: Signal C (PSV orchestrator)
                │   ├─ depends on: PSV color constancy
                │   ├─ depends on: PSV leaf segmentation
                │   ├─ depends on: PSV disease region detection
                │   ├─ depends on: PSV feature extractor (26 features)
                │   └─ depends on: PSV compatibility scorer (uses canonical order)
                ├─ depends on: TTA controller
                ├─ depends on: Hierarchical classifier (Stage 1, Stage 2, soft routing)
                ├─ depends on: Conformal prediction module
                │   └─ depends on: Conformal threshold τ (loaded at startup)
                ├─ depends on: Tier assignment
                │   └─ depends on: tier_rules.yaml
                ├─ depends on: Severity grading module
                ├─ depends on: Multi-image controller (only invoked when N > 1)
                └─ depends on: Response builder
                              └─ depends on: Response schemas (Pydantic)

Storage layer ─ depends on: Sandbox server (logs after response)
Agronomist view ─ depends on: Storage layer (reads predictions, sandbox SQLite only)
Monitoring dashboard ─ depends on: Storage layer (reads predictions, sandbox SQLite only)
```

The input validation gate is shown twice intentionally: once at the sandbox-server level (rejects malformed requests before any pipeline runs) and once as a component the tomato pipeline trusts (the tomato pipeline does NOT re-validate; it assumes the sandbox-server's gate passed).

The 8766 APIN server is not in this graph at all. The two servers are independent processes with no shared state.

### 4.3 Threading model

The wrapper server is single-process FastAPI with multiple worker threads (Uvicorn default). Each request runs in its own thread. The components must be thread-safe:

- Models in eval mode (no gradient state)
- No global mutable state in pipeline objects
- Per-request RNG seeds for stochastic operations (TTA augmentations)
- TTA augmentations created per-request, not shared
- SQLite writes use the database's own locking
- File reads (model weights, calibration files) happen at startup; file handles closed before serving

Python GIL implications: PyTorch GPU operations release the GIL during CUDA kernel execution, allowing concurrent GPU usage from multiple threads (serialized by GPU lock, see Section 26.4). PSV computation is CPU-bound and uses NumPy/OpenCV, which release the GIL for the duration of vectorized operations but hold it for Python-level loops. For practical purposes, two concurrent PSV computations cannot fully utilize two CPU cores from a single Python process. If higher PSV throughput is needed in deployment, options are: (1) accept slight serialization (acceptable at expected request rate under 100/min), (2) use a process pool (`concurrent.futures.ProcessPoolExecutor`) for PSV, (3) port PSV hot paths to Cython or C++ (out of scope for this spec).

If concurrent requests cause GPU memory pressure, the system serializes GPU-touching operations behind a single GPU lock (Section 26.4). CPU-bound parts (PSV feature extraction) run in parallel where the GIL allows.

### 4.4 Lifecycle: startup, ready, shutdown (of the sandbox server)

The 8766 APIN server has its own lifecycle, unchanged by this spec. The lifecycle below is the sandbox server only.

**Startup** (occurs once when sandbox server starts):
1. Load configuration from `tomato_sandbox/config.py`
2. Validate `tomato_sandbox/config/tier_rules.yaml` schema; refuse to start if YAML is invalid (logs which rule failed validation)
3. Start parallel model loading (Section 26.5):
   - Load Router weights (read-only access to existing weight file outside sandbox; sandbox does not modify it)
   - Load v3 weights (same)
   - Load LoRA weights from `tomato_sandbox/models/tomato_sp_lora_production.pt`
   - Load PSV calibration files (`tomato_sandbox/phase_f0_calibration/`)
   - Load conformal threshold τ from disk
   - Load prototype bank from disk
4. Verify all artifacts loaded successfully. **Failure handling:** if any artifact fails to load (file missing, corrupt, version mismatch), the sandbox logs a CRITICAL error identifying which artifact failed, does NOT bind port 8767, and exits with non-zero exit code. Operators receive a clear error message rather than a server that silently runs in a degraded state. The 8766 server is unaffected.
5. Compute and log memory footprint (VRAM and system RAM)
6. Bind port 8767
7. Mark health check endpoint ready

**Ready** (steady state):
- Health check at `localhost:8767/health` returns 200
- `localhost:8767/predict` accepts requests
- `localhost:8767/agronomist` accepts admin/agronomist users (Phase D)
- `localhost:8767/monitoring` accepts admin users (Phase E)

**Shutdown** (graceful):
1. Stop accepting new requests
2. Wait up to 10 seconds for in-flight requests to complete
3. Close SQLite connection (commits pending writes)
4. Release GPU memory (move models to CPU)
5. Exit. The 8766 server is unaffected.

**Crash recovery**: if the sandbox crashes mid-request, the client gets a 502 from the reverse proxy (or no response in dev mode). Restart is manual. SQLite state is consistent because writes are after-response and run inside transactions. The 8766 server is unaffected by sandbox crashes.

Training-inference parity: model preprocessing parameters (CLAHE clip limit, tile grid size, ImageNet mean/std, image resize mode) are pinned in `tomato_sandbox/config.py`. The startup verification step asserts these values match the values recorded at training time (stored in the model checkpoint metadata where available). If a mismatch is detected, startup fails. This prevents silent accuracy degradation from preprocessing drift across library updates.

### 4.5 Configuration and environment variables

Configuration sources, in increasing precedence:
1. `tomato_sandbox/config.py` (defaults; sandbox-specific)
2. `tomato_sandbox/config/tier_rules.yaml` (tier rules, versioned)
3. Environment variables (`TOMATO_ENABLE_STORAGE=1`, etc.) — read by the sandbox process only
4. CLI flags at sandbox server start

The existing `app/config.py` (outside the sandbox) is sacred and unaffected.

Key configurable values:

| Key | Default | Meaning | F.0 derivation rule |
|---|---|---|---|
| `TOMATO_ENABLE_STORAGE` | `0` | Enable Phase E SQLite logging | not data-driven (operational flag) |
| `TOMATO_ENABLE_AGRONOMIST_VIEW` | `0` | Mount `/agronomist` route | not data-driven |
| `TOMATO_ENABLE_MONITORING` | `0` | Mount `/monitoring` route | not data-driven |
| `TOMATO_TTA_TRIGGER_THRESHOLD` | `0.55` | TTA fires below this confidence | F.0 sweep [0.50, 0.55, 0.60, 0.65, 0.70] on training subset out-of-fold; pick the threshold minimizing validation NLL. Default 0.55 is the midpoint of the sweep range. |
| `TOMATO_TTA_ESCALATE_THRESHOLD` | `0.45` | 5-view TTA fires below this | F.0 sweep [0.35, 0.40, 0.45, 0.50] on training subset; pick the threshold below which 2-view TTA disagreement rate exceeds 30% (heuristic for "more views likely needed"). Default 0.45 is a reasonable starting point. |
| `TOMATO_PROTOTYPE_BLEND_THRESHOLD` | `0.60` | LoRA prototype blending fires below | F.0 sweep [0.50, 0.55, 0.60, 0.65, 0.70] on training subset out-of-fold; pick the threshold minimizing validation NLL. |
| `TOMATO_CHILLI_LEAKAGE_THRESHOLD` | `0.40` | Above this triggers Tier 3C (Rule 3) | F.0 sets to 95th percentile of `chilli_leakage` scores on confirmed-tomato images in the training subset (so true tomato gets flagged at most 5% of the time). |
| `TOMATO_TIER1_CHILLI_CAP` | `0.20` | Tier 1's stricter cap (Rule 7) | F.0 may sweep [0.15, 0.20, 0.25] on validation subset; default 0.20 is conservative. |
| `TOMATO_TIER2_CHILLI_CAP` | `0.30` | Tier 2's cap (Rule 8) | F.0 may sweep [0.25, 0.30, 0.35] on validation subset; default 0.30 is intermediate. |
| `TOMATO_CONFORMAL_ALPHA` | `0.10` | Conformal coverage = 1 − α | not F.0-derived; chosen by user (90% coverage is the standard choice). |
| `TOMATO_GPU_LOCK_TIMEOUT_S` | `10` | Max wait for GPU lock | not data-driven; set to keep total request under typical 30s HTTP timeout. |
| `TOMATO_REQUEST_CACHE_SIZE` | `1000` | LRU cache entries | not data-driven |
| `TOMATO_REQUEST_CACHE_TTL_S` | `3600` | Cache expiration | not data-driven |
| `CLASSIFIER_VARIANT` | `logistic` | Which classifier variant to load (`logistic` or `mlp`) | F.0 sets to `mlp` only if MLP variant improves macro-F1 by ≥ 2 points over logistic AND ECE stays under 0.10 (Section 12.6). Default `logistic`. |

All thresholds with `_THRESHOLD` suffix have F.0-derived values that override defaults if the calibration file is present.

### 4.6 Response time budget

Target latency for the tomato pipeline. Two tables are given: GPU compute time only, and total request time (including image decode, IQA, PSV CPU compute, response construction).

**GPU compute time only (one path through the model):**

| Path | Median | P95 | Worst-case |
|---|---|---|---|
| Single image, no TTA | 250 ms | 400 ms | 600 ms |
| Single image, 2-view TTA | 500 ms | 800 ms | 1.2 s |
| Single image, 5-view TTA | 1.2 s | 2.0 s | 3.5 s |
| 3-image input, no TTA | 750 ms | 1.2 s | 1.8 s |
| 3-image input, all 5-view TTA | 3.6 s | 6.0 s | 10 s |

**Total request time (image decode + IQA + preprocessing + GPU compute + PSV CPU + classifier + tier + response):**

| Path | Median | P95 | Worst-case |
|---|---|---|---|
| Single image, no TTA | 600 ms | 900 ms | 1.2 s |
| Single image, 2-view TTA | 900 ms | 1.4 s | 2.0 s |
| Single image, 5-view TTA | 1.6 s | 2.5 s | 4.0 s |
| 3-image input, no TTA | 1.5 s | 2.5 s | 3.5 s |
| 3-image input, all 5-view TTA | 4.5 s | 7.5 s | 12 s |

The total-time figures assume RTX 4060 GPU and a moderately fast CPU. Per-step CPU breakdown (median, single image):

| Step | Median time |
|---|---|
| HTTP receive + multipart parse | 20 ms |
| Image decode (JPEG) | 50 ms |
| Input validation gate (logic only, decode separate above) | 5 ms |
| IQA (7 dimensions) | 40 ms |
| Preprocessing for all three pipelines (v3 stretch + LoRA letterbox + PSV color constancy) | 55 ms |
| GPU compute (single image, no TTA: router + v3 + LoRA) | 250 ms |
| PSV CPU compute (segmentation + 26 features + scoring) | 200 ms |
| Hierarchical classifier (logistic, two stages) | 5 ms |
| Conformal threshold lookup | 1 ms |
| Tier assignment | 5 ms |
| Response construction + JSON serialize | 15 ms |
| **Total median** | **~650 ms** |

The 600 ms median figure in the summary table at the start of Section 4.6 was an underestimate; the corrected median is ~650 ms. Summary table values should be read as approximate; the itemization above is the source of truth for component breakdown.

**Concurrent request behavior.** Under load, GPU-touching operations serialize behind the GPU lock (Section 26.4). If two requests arrive simultaneously, one waits for the other's GPU work before starting its own GPU work. CPU-bound work (PSV) runs concurrently subject to GIL constraints (see Section 4.3). Effective throughput at sustained concurrency is approximately one request per `total_time` interval per GPU; for 4 concurrent requests with 1-second median total time, effective throughput is approximately 4 req/sec sustained, with queue-wait latency added on top.

The user-facing UI shows progress indicators during long-running requests.

### 4.7 Observability

Three layers:

1. **Per-request structured logs** (Section 26.6). One JSON log line per request with request_id, latency_ms, tier, model versions, errors. The `model_version` field is a string composed of `<component>=<short-hash>` for each loaded model: e.g., `"v3=a1b2c3d, lora=e4f5g6h, router=i7j8k9l"`. The short hash is the first 7 characters of the SHA-256 hash of the loaded weight file. This allows correlating logged predictions to specific deployed model artifacts without storing full file paths in every log line. Section 27 covers versioning in full.
2. **SQLite predictions table** (Phase E). Long-term record of all predictions and feedback. Read by monitoring and agronomist view.
3. **Monitoring dashboard** (Phase E). Aggregate metrics: tier distribution, average confidence, per-class prediction rates, drift signals.

Logs are structured (machine-parseable). Dashboard is human-facing. Both backed by the same data.

### 4.8 What comes next in this document

Sections 1 through 4 (this part) gave the foundation: scope, context, existing assets, architecture overview, and component map.

**Part II — Tomato Pipeline (Sections 5 through 18)** is the centerpiece. It walks the full per-request flow inside the tomato pipeline in deepest detail. Each stage has its own section: input validation (5), IQA (6), preprocessing (7), Signal A (8), Signal B (9), Signal C / PSV (10), TTA (11), hierarchical classifier (12), conformal prediction (13), tier assignment (14), decision scenarios (15) including the matrix and 100+ prose scenarios, response schema (16), severity grading (17), and multi-image support (18). Every metric mentioned anywhere in this document is defined formally either in its section of Part II or in Appendix A.

**Part III — System Integration (Sections 19 through 25)** covers the rest: the router (19), how APIN is used as a library (20), wrapper server architecture (21), frontend extensions (22), agronomist view (23), SQLite storage (24), and drift monitoring (25).

**Part IV — Engineering (Sections 26 through 30)** covers production hygiene (26), configuration and versioning (27), validation gates (28), the phase plan (29), and honest limitations and risk register (30).

**Part V — Reference Appendices (A through F)** provides the metric reference, scenario decision matrix, full PSV feature catalog, the example `tier_rules.yaml`, class index tables, and the file/artifact catalog.

The reader who only wants to understand "what does this system do for a tomato photo" can read Sections 5 through 18 in order. The reader who needs to integrate or operate the system should also read Part III. The reader who needs to validate or deploy should read Part IV. The reader who needs a specific metric definition or class index should consult Part V.

---

# Part II — Tomato Pipeline (Sections 5-18)

This part walks the per-request flow through the tomato pipeline in order. Every component lives inside `tomato_sandbox/` per the Sandbox Directive (Section 0). Section numbers here are continuations of Part I.

A request entering the tomato pipeline has already passed the sandbox-server-level checks (the request reached `localhost:8767/predict` and the router confirmed `active_crop = "tomato"`). What follows is what happens inside the pipeline itself.

The order of stages is fixed:
1. **Section 5:** Input validation gate (per-image, before any other work)
2. **Section 6:** Image Quality Assessment (per-image)
3. **Section 7:** Preprocessing into the three formats needed by the three signals (per-image)
4. **Sections 8, 9, 10:** The three signals (Signal A = v3, Signal B = LoRA, Signal C = PSV) run conceptually in parallel
5. **Section 11:** TTA controller decides whether to re-run signals with augmented views
6. **Section 12:** Hierarchical classifier produces probabilities
7. **Section 13:** Conformal prediction set built from those probabilities
8. **Section 14:** Tier assignment using priorities and rule chain
9. **Section 16:** Response builder serializes the result
10. **Section 17:** Severity grading attached to response
11. **Section 18:** Multi-image aggregation when more than one image was uploaded

Sections 5 through 9 cover the first half of this flow.

---

## Section 5. Image Input and Validation Gate

### 5.1 Purpose

The validation gate is the first piece of code that touches an uploaded image. Its job is to reject inputs that cannot possibly be processed by the rest of the pipeline, before any expensive work runs (no GPU, no PSV, no IQA). It is a fast, cheap, defensive check.

The validation gate is NOT the same as IQA. IQA judges whether the photographed leaf is a *good* photo (sharp, well-lit, single leaf). Validation judges whether the *file* is processable at all (loadable, RGB, in size range, count within limits). A blurry but processable photo passes validation and proceeds to IQA. A corrupt file does not pass validation and never reaches IQA.

### 5.2 What gets validated

The gate runs the following checks, in order. The first failure terminates with a 400 response. Earlier checks are cheaper; ordering matters for performance.

**Check A — Request-level limits.**
- `image_count`: number of files in the request. Must be in `[1, 5]`. Single-image upload is the common case (Section 18); 2-5 images trigger multi-image flow. More than 5 is rejected to bound resource use.
- `total_payload_size`: sum of all uploaded file bytes. Must be ≤ 100 MB. Hard server-side limit independent of per-image size.

**Check B — Per-image file metadata.**
- `mime_type`: must be `image/jpeg` or `image/png`. Other formats (`image/heic`, `image/webp`, `image/raw`, `application/octet-stream`, `image/gif`, `image/tiff`) are rejected with format-specific guidance. Note: HEIC is the iPhone default; the rejection message tells the user "Convert to JPEG before uploading; iPhone camera roll has a share-as-JPEG option."
- `file_size_bytes`: must be in `[5 KB, 20 MB]`. Below 5 KB is almost certainly thumbnail-sized or corrupt; above 20 MB is too large to process within latency budget.
- `extension_matches_mime`: file extension must agree with sniffed mime type. A `.jpg` file with PNG bytes is rejected.

**Check C — Image decode.**
- Decode the bytes using PIL/Pillow. Any decode exception is a corruption rejection.
- After decode, verify width and height are both in `[224, 8192]` pixels. Below 224 cannot be processed at v3's expected input size without upsampling (which loses information). Above 8192 is over-resolution; the system will downsample but rejects the absurd cases.
- If the image is not RGB (grayscale, RGBA, palette-mode, CMYK), it is converted to RGB in-memory (alpha channel discarded if present). Grayscale-source images are detected (zero saturation in all pixels of the converted RGB) and rejected with a specific message rather than silently treated as colorless leaves. This is a transformation-or-rejection step, not silent acceptance.

**Check D — EXIF orientation.**
- Apply EXIF orientation tag if present. Many smartphone cameras store the image in landscape regardless of physical orientation and use EXIF to indicate rotation. Without applying it, the system sees images sideways.
- After applying orientation, the image is in its visually-correct orientation. EXIF tag is then stripped (the orientation is already baked in).

**Check E — Aspect ratio sanity.**
- Width-to-height ratio must be in `[0.25, 4.0]`. A 1:10 ratio (very tall narrow image) is almost always not a leaf photo; it might be a cropped UI screenshot.
- This check is sanity-only and can be removed if real Kerala photo collection shows unusual ratios are common. F.0 may relax this.

**Check F — Duplicate detection within multi-image request.**
- If the request contains multiple images, compute SHA256 of each. If two or more images share a hash, only the first is kept; the others are silently dropped. The response indicates how many unique images were processed.

If all checks pass, the gate emits a `ValidatedImage` data structure containing:
```python
@dataclass
class ValidatedImage:
    pil_image: PIL.Image.Image       # RGB, EXIF-applied, ready for downstream
    width: int
    height: int
    file_size_bytes: int
    mime_type: str
    sha256_hash: str                  # hex string; used by the response cache (Sec 26.1)
```

### 5.3 Rejection responses

Each rejection returns HTTP 400 with a JSON body matching this schema:

```json
{
  "error": "input_validation_failed",
  "reason_code": "<machine-readable code>",
  "reason_human": "<actionable message for the end user>",
  "details": {
    "field": "<which field/check failed>",
    "expected": "<what was expected>",
    "received": "<what was received>"
  }
}
```

**Reason codes and corresponding human messages:**

| reason_code | reason_human |
|---|---|
| `too_many_images` | "You uploaded {n} images. Up to 5 are allowed in a single request." |
| `payload_too_large` | "Total upload size is too large. Limit: 100 MB across all images." |
| `unsupported_format` | "File type {mime_type} is not supported. Use JPEG or PNG. (iPhone HEIC photos can be shared as JPEG from the share menu.)" |
| `file_too_small` | "Image file is unusually small ({n} KB). Re-take the photo at higher quality." |
| `file_too_large` | "Image file is too large ({n} MB). Limit per image: 20 MB." |
| `extension_mismatch` | "File extension and content do not match. Re-save the file using its original format." |
| `decode_failed` | "Image file could not be opened. It may be corrupted or partially uploaded. Try re-uploading." |
| `dimensions_too_small` | "Image is too small ({w}×{h}). Minimum: 224×224 pixels." |
| `dimensions_too_large` | "Image is too large ({w}×{h}). Maximum: 8192×8192 pixels. Most phone cameras do not exceed this; check if you have a special-mode photo." |
| `aspect_ratio_extreme` | "Image proportions ({ratio}:1) are unusual for a leaf photo. Re-frame with the leaf taking up most of the photo." |
| `grayscale_image` | "Image appears to be black-and-white. Color is needed for disease detection. Re-take in color mode." |

The end user sees `reason_human` on the frontend. The `reason_code` and `details` are for client logic and debugging.

### 5.4 What validation does NOT check

The gate is intentionally narrow. It does not check:
- Whether the image actually contains a leaf (that is the leaf_presence dimension of IQA, Section 6)
- Whether the photo is sharp or well-exposed (other IQA dimensions)
- Whether the leaf is a tomato leaf (router decides crop; misrouted images go to IQA's leaf_presence and PSV's chilli_leakage as a fallback)
- Whether the image is a duplicate of one previously seen (response cache handles this, Section 26.1)
- Anything semantic (no ML model is loaded for validation)

If these checks were performed at the validation gate, validation would become slow and the failure boundary would blur with IQA. The clean separation is: validation says "this file is processable," IQA says "this leaf is photographable."

**Note on misrouting safety net.** The system has explicit chilli-leakage detection (Section 8.4) for chilli-as-tomato misrouting, but does NOT have analogous leakage signals for okra or brassica. If the router incorrectly classifies an okra or brassica image as tomato, the tomato pipeline will process it and produce some result — likely with low confidence ending in Tier 4A or 4B. The IQA leaf_presence check catches obviously non-leaf photos but does not distinguish tomato leaves from other crops' leaves. Future work could add a similar leakage signal for non-chilli misrouting; this is documented in Section 30 (honest limitations).

### 5.5 Edge cases handled

- **Partial upload**: client disconnected mid-upload. Detected at decode (PIL raises). Returns `decode_failed`.
- **Empty file**: zero-byte upload. Caught by `file_too_small` (5 KB minimum).
- **Image with EXIF orientation 1 (no rotation)**: most desktop-saved images. No-op; passes through.
- **Image with EXIF orientation 6 (rotate 90° CW)**: most iPhone landscape-mode photos. Applied transparently.
- **Animated GIF**: rejected at mime check (`image/gif` is not in the accepted list). Returns `unsupported_format`.
- **Animated PNG (APNG)**: passes mime check (mime is `image/png`). PIL decodes only the first frame by default; remaining frames are silently discarded with a debug log entry. No rejection; the first frame is treated as the image.
- **TIFF or PSD**: rejected at mime check before decode is attempted.
- **Image embedded in zip or other archive**: unsupported; mime check returns `unsupported_format`.
- **HDR or 16-bit-per-channel image**: PIL converts to 8-bit when calling `.convert("RGB")`. Loss of dynamic range is acceptable; if the user complains, F.0 is the place to investigate whether higher bit depth helps.
- **Grayscale source converted to RGB(L,L,L)**: detected by zero-saturation check; rejected with `grayscale_image` (not silently passed to IQA).
- **Two identical images uploaded in one multi-image request**: deduplicated by SHA256; only the first instance is processed.

### 5.6 Performance budget

The validation gate's pure-validation logic (everything except image decode) runs in approximately **5 ms median**. Image decode is the expensive part (~25 ms median for a 4 MB JPEG, up to ~75 ms for a 20 MB JPEG). The Section 4.6 latency table separates these two: "Input validation" line is 5 ms (the validation logic), "Image decode" line is 50 ms (the PIL.Image.open + load). Together they are approximately 55 ms median for a typical image, ~100 ms worst case for a 20 MB image.

Itemized validation logic (excluding decode):
- Mime/size/extension checks: <1 ms
- EXIF apply + RGB convert: ~3 ms
- Aspect ratio check: <1 ms
- SHA256 hash: ~1 ms
- Total: ~5 ms median

For a 20 MB image, decode dominates and may approach 75 ms; total (decode + validation) approaches 80 ms. This is the upper bound and acceptable.

### 5.7 Where this lives in the sandbox

`tomato_sandbox/input_validation.py` defines the `ValidatedImage` dataclass and the `validate_request(request) -> List[ValidatedImage]` entry point. `tomato_sandbox/server.py` (Section 21) calls this before invoking the tomato pipeline. The pipeline itself trusts the validated input and does not re-validate.

---

## Section 6. Image Quality Assessment (IQA)

### 6.1 Purpose and overview

IQA decides whether a validated image is *good enough to diagnose* and, if not, what specifically is wrong so the user can retake. IQA runs after validation (Section 5) and before any GPU work or PSV.

IQA produces three outputs:
1. **Per-dimension scores** — one float in `[0, 1]` per quality dimension. Higher is better.
2. **Aggregate quality** — one float combining the per-dimension scores.
3. **Decision** — one of `REJECT`, `DEGRADED`, `ACCEPTABLE`, `HIGH`.

The decision drives whether the request continues to the pipeline (and what tier ceiling applies) or returns immediately with a retake prompt.

IQA has 7 dimensions. Each dimension is computed independently from the input image; they are then combined.

### 6.2 The seven dimensions

Each dimension has: a name, a measurement procedure, an output range, a threshold for `BAD`, a threshold for `GOOD`, and an actionable retake message used when the dimension fails.

Thresholds listed below are placeholder defaults. F.0 calibration (Section 29) sets the production values from histograms on the training subset.

#### 6.2.1 sharpness

**What it measures:** how in-focus the image is. A blurry image cannot show fine disease features (lesion edges, texture).

**How:** variance of the Laplacian. The Laplacian operator approximates the second derivative of image intensity; a sharp image has high variance because edges produce strong responses.

```python
def sharpness(rgb_img: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    raw_variance = lap.var()
    # Normalize to [0, 1] using a saturation curve. Variance > 1000 → 1.0.
    return min(raw_variance / 1000.0, 1.0)
```

**Why this normalization:** Laplacian variance is unbounded above. Empirically (general photography literature; will be re-checked on Kerala field photos in F.0), sharp leaf images have variance > 500 and very-sharp ones exceed 1000. We saturate at 1000 so the [0, 1] range is meaningful.

**Thresholds (placeholder):**
- `BAD` if score < 0.20 (raw variance < 200)
- `GOOD` if score > 0.50 (raw variance > 500)

**Retake message:** "Image is blurry. Hold the phone steady, tap on the leaf to focus, and re-take."

#### 6.2.2 exposure

**What it measures:** whether the image is too dark or too bright. Both extremes lose information.

**How:** mean of the V (value) channel in HSV color space.

```python
def exposure(rgb_img: np.ndarray) -> float:
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    v_mean = hsv[:, :, 2].mean()  # in [0, 255]
    # Optimal V is around 130 (mid-range). Score is a tent function around 130.
    if v_mean < 50:
        return 0.0  # too dark
    elif v_mean > 220:
        return 0.0  # blown out
    elif v_mean < 130:
        return (v_mean - 50) / 80  # ramp up from 0 at 50 to 1 at 130
    else:
        return 1.0 - (v_mean - 130) / 90  # ramp down from 1 at 130 to 0 at 220
```

**Why this shape:** a midtone image (V around 130) carries the most information in 8-bit JPEG. Both shadows (V<50) and highlights (V>220) clip and lose detail.

**Thresholds (placeholder):**
- `BAD` if score < 0.20 (V mean below ~65 or above ~205)
- `GOOD` if score > 0.60 (V mean between ~98 and ~166)

**Retake message (low):** "Image is too dark. Move into more even lighting and re-take."
**Retake message (high):** "Image is overexposed (too bright). Move out of direct sunlight or shade the leaf and re-take."

#### 6.2.3 leaf_presence

**What it measures:** is there a leaf in the image at all? Catches obvious mistakes (photographed sky, soil, hand, finger over lens).

**How:** rough green-pixel detection with morphology, NOT a full segmentation (PSV does that later more carefully). The check is intentionally fast.

```python
def leaf_presence(rgb_img: np.ndarray) -> float:
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    # Broad green/yellow-green hue range (covers healthy and chlorotic leaves)
    green_mask = (
        (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) &  # H
        (hsv[:, :, 1] >= 40)                            # S — exclude near-grey pixels
    )
    pct_green = green_mask.mean()  # in [0, 1]
    # Score: rises from 0 at 5% green to 1 at 30% green
    if pct_green < 0.05:
        return 0.0
    elif pct_green > 0.30:
        return 1.0
    else:
        return (pct_green - 0.05) / 0.25
```

**Why this shape:** a leaf-centered photo will have at least 30% green-ish pixels. Below 5% green almost certainly is not a leaf photo.

**Thresholds (placeholder):**
- `BAD` if score < 0.30 (less than ~10% green pixels)
- `GOOD` if score > 0.70 (more than ~22% green pixels)

**Retake message:** "I cannot find a tomato leaf in this image. Re-take with the leaf centered in the frame."

#### 6.2.4 leaf_fill

**What it measures:** how much of the frame the leaf occupies. Distantly-photographed leaves have the disease information packed into too few pixels for reliable diagnosis.

**How:** assuming the leaf_presence check produced a rough green mask, compute the largest connected component of that mask and report its bounding box area as a fraction of the image.

```python
def leaf_fill(green_mask: np.ndarray, image_shape: tuple) -> float:
    # green_mask is reused from leaf_presence
    H, W = image_shape[:2]
    nb, _, stats, _ = cv2.connectedComponentsWithStats(green_mask.astype(np.uint8))
    if nb <= 1:
        return 0.0
    # Largest component (ignore background label 0)
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_bbox_w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    largest_bbox_h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    fill = (largest_bbox_w * largest_bbox_h) / (W * H)
    # Score: 0 at 5% fill, 1 at 40% fill, ramp linearly
    if fill < 0.05:
        return 0.0
    elif fill > 0.40:
        return 1.0
    else:
        return (fill - 0.05) / 0.35
```

**Why this shape:** a properly framed leaf occupies 40-80% of the image. Below 5% the leaf is too distant; above 80% is acceptable but visible (no penalty). This dimension catches "leaf is in the photo but the photo was taken from 5 meters away."

**Thresholds (placeholder):**
- `BAD` if score < 0.30 (bbox fill < 15% of frame)
- `GOOD` if score > 0.70 (bbox fill > 30% of frame)

**Retake message:** "The leaf is too far away in the frame. Move closer (10-15 cm) so the leaf fills most of the image."

#### 6.2.5 background_contamination

**What it measures:** whether the image contains other strong distractors besides the leaf — multiple plants, hands, ground, sky. Strong distractors confuse the neural models.

**How:** count connected components of green-ish pixels (multi-leaf detector). Multiple large components likely mean multiple leaves or strongly contaminated background.

```python
def background_contamination(green_mask: np.ndarray, image_shape: tuple) -> float:
    H, W = image_shape[:2]
    nb, _, stats, _ = cv2.connectedComponentsWithStats(green_mask.astype(np.uint8))
    if nb <= 1:
        return 1.0  # no green at all means leaf_presence failed already; defer to that check
    sizes = stats[1:, cv2.CC_STAT_AREA]  # skip background
    image_area = H * W
    significant = sizes[sizes > image_area * 0.05]  # components covering >5% of image
    n_significant = len(significant)
    if n_significant <= 1:
        return 1.0  # one large leaf — clean
    elif n_significant == 2:
        return 0.5  # two leaves or a leaf with significant green background
    else:
        return 0.0  # cluttered scene
```

**Why this shape:** a single isolated leaf is the ideal photo. Two large green regions might be two leaves (multi-leaf flow can handle if user uploaded multiple files, but a single image with two leaves should be re-framed). Three or more is cluttered.

**Thresholds (placeholder):**
- `BAD` if score < 0.30 (3+ large green regions)
- `GOOD` if score > 0.80 (single isolated leaf)

**Retake message:** "Multiple leaves or strong distractors are in the frame. Isolate one leaf against a plain background and re-take."

#### 6.2.6 resolution

**What it measures:** raw pixel count. Already gated by validation (Section 5.2 Check C caps at [224, 8192]) but the score nuances within that range.

**How:** smaller dimension of the image, normalized.

```python
def resolution(image_shape: tuple) -> float:
    H, W = image_shape[:2]
    smaller = min(W, H)
    if smaller < 224:
        return 0.0  # rejected at validation; this is defensive
    elif smaller >= 800:
        return 1.0
    else:
        return (smaller - 224) / (800 - 224)  # ramp from 0 at 224 to 1 at 800
```

**Why this shape:** v3 expects 224 input; LoRA expects 392. Anything ≥ 800 has plenty of headroom. Between 224 and 800, the leaf details get progressively crisper.

**Thresholds (placeholder):**
- `BAD` if score < 0.20 (< 339 px on shortest side)
- `GOOD` if score > 0.50 (> 512 px on shortest side)

**Retake message:** "Image resolution is too low. Use the phone's main camera (not zoom or screenshot) and re-take."

#### 6.2.7 wetness

**What it measures:** specular highlights from water on the leaf, which look like blight or other wet-tissue diseases and confuse PSV's lesion detection.

**How:** fraction of pixels that are very bright AND have very low saturation — the signature of specular highlights.

```python
def wetness(rgb_img: np.ndarray) -> float:
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    spec_mask = (hsv[:, :, 2] > 220) & (hsv[:, :, 1] < 30)  # bright + desaturated
    pct_spec = spec_mask.mean()
    # Score: 1.0 if no specular pixels, 0.0 if > 5% of image is specular
    if pct_spec < 0.005:
        return 1.0
    elif pct_spec > 0.05:
        return 0.0
    else:
        return 1.0 - (pct_spec - 0.005) / 0.045
```

**Why this shape:** a small fraction of specular pixels is normal (waxy leaf cuticle). More than 5% indicates the leaf is genuinely wet. Wet leaves have unreliable color and texture features.

**Thresholds (placeholder):**
- `BAD` if score < 0.30 (more than ~3.5% specular pixels)
- `GOOD` if score > 0.80 (less than ~1.4% specular pixels)

**Retake message:** "The leaf appears wet, which can be confused with disease symptoms. Wait for the leaf to dry and re-take."

This dimension is Kerala-specific (monsoon season). On dry-climate deployments this dimension can be down-weighted via the aggregation (Section 6.3).

### 6.3 Aggregating to a single quality score

The seven dimensions combine into a single `aggregate_quality` score by **geometric mean**, optionally weighted.

```python
def aggregate_quality(scores: dict[str, float], weights: dict[str, float] = None) -> float:
    if weights is None:
        weights = {k: 1.0 for k in scores}  # equal weighting default
    total_weight = sum(weights.values())
    log_sum = sum(weights[k] * math.log(max(scores[k], 1e-6)) for k in scores)
    return math.exp(log_sum / total_weight)
```

**Why geometric mean and not arithmetic mean:** any single dimension being near-zero indicates a fatal flaw (blurry, no leaf, etc.) that should pull the aggregate down strongly. Arithmetic mean lets one good dimension mask one terrible one. Geometric mean does not — its sensitivity to small values is exactly what we want.

**Why weighted (in principle, even if uniform default):** F.0 may reduce the weight of `wetness` for non-monsoon seasons, or reduce `background_contamination` if Kerala field photos consistently have busy backgrounds.

### 6.4 The four-way decision

The per-dimension BAD thresholds are aggregated into a single dict (placeholder values; F.0 calibrates each):

```python
BAD_THRESHOLDS = {
    "sharpness": 0.20,
    "exposure": 0.20,
    "leaf_presence": 0.30,
    "leaf_fill": 0.30,
    "background_contamination": 0.30,
    "resolution": 0.20,
    "wetness": 0.30,
}
```

These are the same values listed per-dimension in Section 6.2. The dict consolidates them for code use.

```python
def iqa_decide(aggregate: float, per_dim: dict[str, float]) -> str:
    # Hard rejections from individual dimensions trump aggregate
    for dim_name, score in per_dim.items():
        if score < BAD_THRESHOLDS[dim_name]:
            return "REJECT"
    # Otherwise decide by aggregate
    if aggregate < 0.40:
        return "REJECT"
    elif aggregate < 0.60:
        return "DEGRADED"
    elif aggregate < 0.80:
        return "ACCEPTABLE"
    else:
        return "HIGH"
```

**Decision semantics:**
- **REJECT** — return immediately to user with retake instructions. No pipeline work performed. The user is shown the most-failing dimension's retake message; if multiple dimensions fail, the worst is shown.
- **DEGRADED** — pipeline proceeds, but the tier system caps the result at Tier 3 (no Tier 1 or Tier 2 confidence claims allowed). Severity grading still works. This is a forward contract with Section 14: when `iqa.decision == "DEGRADED"`, Section 14's tier rule chain must enforce the Tier-3 ceiling.
- **ACCEPTABLE** — pipeline proceeds normally; tiers 1 through 4 reachable based on classifier output.
- **HIGH** — pipeline proceeds normally with no IQA-related restriction. Functionally same as ACCEPTABLE for the tier system; the distinction is recorded for monitoring.

For multi-image requests, IQA is computed per-image. The multi-image controller (Section 18) decides how to combine per-image IQA decisions. A common rule: if any single image is `REJECT`, drop it but proceed with the remaining acceptable images; if all images are `REJECT`, the whole request fails.

Reasons for splitting REJECT into "any-dim-bad" plus "aggregate-bad" rather than just aggregate-based:
- A single fatally bad dimension (e.g., sharpness=0.05 from a totally blurry photo) might still produce aggregate=0.45 if the other 6 dimensions are decent. Aggregate alone would let this through as DEGRADED. The any-dim-bad rule catches it.
- Symmetrically, an aggregate-only check misses fatal asymmetric failures.

**F.0 calibration of IQA thresholds.** Each dimension's BAD threshold and the four-way decision boundaries (0.40, 0.60, 0.80) are placeholders. F.0 derives the production values from a labeled set of photos pre-classified as REJECT/DEGRADED/ACCEPTABLE/HIGH by an agronomist:
- For each dimension's BAD threshold: set so that on the labeled set, all photos labeled REJECT or worse score below the threshold (i.e., the threshold sits at the maximum score across REJECT-labeled photos for that dimension). This minimizes false-acceptance.
- For aggregate boundaries: set so that the IQA decision matches the agronomist's label at least 80% of the time on the labeled set. The boundaries are tuned via grid search over the [0.30, 0.85] range.
- F.0 outputs are stored in `tomato_sandbox/phase_f0_calibration/iqa_thresholds.json` and loaded at startup, overriding the placeholder defaults.

If the labeled set is unavailable (Phase A is incomplete), F.0 falls back to a percentile rule: BAD threshold at 5th percentile, aggregate boundaries at 10th/30th/60th percentiles of the score distribution on the FULL training subset (160 images). This is less accurate but available without agronomist input.

### 6.5 Output structure

IQA emits a result object the rest of the pipeline consumes:

```python
@dataclass
class IQAResult:
    decision: str                     # REJECT / DEGRADED / ACCEPTABLE / HIGH
    aggregate_score: float            # in [0, 1]
    per_dimension: dict[str, float]   # 7 entries, dimension name -> score
    failing_dimensions: list[str]     # names where score < BAD_THRESHOLD
    retake_message: str | None        # if decision == REJECT, the user-facing message
    green_mask: np.ndarray | None     # rough HSV-green mask; passed to PSV as a hint
```

**Contract on `green_mask` between IQA and PSV.** The mask emitted here is the rough HSV-based green mask from `leaf_presence` (Section 6.2.3). It is NOT PSV's primary segmentation. PSV (Section 10) runs its own more careful segmentation pipeline (HSV + Otsu + morphology operations on the color-constancy-applied image). PSV uses this rough IQA mask in two ways: (a) as a sanity check — if PSV's careful segmentation disagrees radically with the IQA mask, PSV's `psv_aggregate_reliability` is reduced; (b) as a fallback — if PSV's careful segmentation fails entirely (throws an exception or produces an empty mask), PSV falls back to the IQA mask rather than failing the whole signal. PSV does NOT directly use the IQA mask as input to feature extraction; features always run on PSV's own segmentation when available.

This contract means IQA's mask quality directly affects only PSV's reliability score and its fallback behavior, not its primary feature values. Section 10 will document the consumer side.

### 6.6 Where this lives in the sandbox

`tomato_sandbox/iqa.py` defines `IQAResult` and `compute_iqa(validated_image: ValidatedImage) -> IQAResult`. The sandbox server calls IQA after validation and before invoking the rest of the pipeline. If IQA returns REJECT, the server short-circuits and returns immediately.

### 6.7 Performance budget

IQA is CPU-only and runs in approximately 40 ms median:
- HSV conversion (computed once, reused): ~5 ms
- Sharpness (Laplacian): ~15 ms
- Exposure (one channel mean): ~1 ms
- Leaf presence + fill (mask + connected components): ~10 ms
- Background contamination (re-uses CC results): ~1 ms
- Resolution (constant time): <1 ms
- Wetness (mask + mean on already-cached HSV): ~3 ms
- Aggregate + decision: ~1 ms

Total: ~37 ms median; budget allowance ~40 ms. This is approximately 6% of the total request time budget — comfortable margin.

---

## Section 7. Image Preprocessing Pipelines

### 7.1 Three pipelines, three consumers

Three downstream consumers each need a different preprocessed view of the same input image:

- **Signal A (v3)** wants 224×224 with stretch-resize, then LAB-CLAHE, then ImageNet-normalized tensor.
- **Signal B (LoRA)** wants 392×392 with letterbox padding (pad value 114), then LAB-CLAHE, then ImageNet-normalized tensor.
- **Signal C (PSV)** wants the raw RGB image with Shades-of-Gray L6 color constancy applied. No CLAHE for PSV (it would distort the color statistics PSV measures). No tensor conversion (PSV is CPU/NumPy).

Three separate pipelines exist because each consumer was trained or designed expecting a specific input format. Mismatching is a recipe for silent accuracy degradation — the model produces output but it is biased.

**Call pattern.** The TomatoPipeline orchestrator (Section 21) calls each preprocessing function exactly once per input image and passes the result to the corresponding signal module. Signal modules do NOT call preprocessing functions themselves; they receive already-preprocessed input. The orchestrator looks like:

```python
# inside TomatoPipeline.infer (Section 21)
v3_input = preprocess_for_v3(validated.pil_image)        # [3, 224, 224] tensor
lora_input = preprocess_for_lora(validated.pil_image)    # [3, 392, 392] tensor
psv_input = preprocess_for_psv(validated.pil_image)      # [H, W, 3] uint8 numpy

result_a = compute_signal_a(v3_input)         # Section 8
result_b = compute_signal_b(lora_input, ...)  # Section 9
result_c = compute_signal_c(psv_input, iqa.green_mask)  # Section 10
```

This pattern means each preprocessing runs once even when the pipeline triggers TTA: the no-augmentation pass uses these baseline preprocessed inputs; augmented views call the preprocessing functions again with the augmented PIL image (Section 11).

### 7.2 Pipeline 1 — for v3 (Signal A)

**Pinned constants** (live in `tomato_sandbox/config.py`; asserted at startup against checkpoint metadata where available, per Section 4.4 training-inference parity):

```python
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])  # RGB order
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
V3_INPUT_SIZE = 224
LORA_INPUT_SIZE = 392
LORA_PAD_VALUE = 114  # used by preprocess_for_lora; matches LoRA's training pad value
TOMATO_CROP_MODE_INDEX = 2  # passed to v3's HardFiLM at inference (Section 8)
```

These constants must match the values used at training time. A mismatch produces silent accuracy degradation.

```python
def preprocess_for_v3(pil_image: PIL.Image.Image) -> torch.Tensor:
    """
    Returns a [3, 224, 224] tensor on CPU, ImageNet-normalized,
    in the exact format v3 was trained with.
    """
    # 1. Resize with stretch (no aspect-ratio preservation)
    resized = pil_image.resize((V3_INPUT_SIZE, V3_INPUT_SIZE), PIL.Image.BILINEAR)
    rgb = np.array(resized, dtype=np.uint8)  # [224, 224, 3]
    
    # 2. LAB-CLAHE on L channel
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 3. To float and ImageNet normalize
    arr = rgb_clahe.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    
    # 4. To tensor in CHW order
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor
```

**Why stretch and not letterbox for v3:** v3 was trained with stretch resize per its training script. Inference must match. Letterboxing v3 inputs would shift feature distributions and degrade accuracy.

**Why LAB-CLAHE and not RGB-CLAHE:** LAB separates luminance (L) from color (A, B). Applying CLAHE to L only enhances local contrast without distorting color. Applying CLAHE channel-wise in RGB shifts color balance, which is bad for plant disease where color is diagnostic. v3 was trained with LAB-CLAHE; inference must match.

### 7.3 Pipeline 2 — for LoRA (Signal B)

```python
def preprocess_for_lora(pil_image: PIL.Image.Image) -> torch.Tensor:
    """
    Returns a [3, 392, 392] tensor, ImageNet-normalized, with letterbox padding.
    """
    # 1. Letterbox resize (preserves aspect ratio, pads to square)
    arr = np.array(pil_image, dtype=np.uint8)
    H, W = arr.shape[:2]
    scale = LORA_INPUT_SIZE / max(H, W)
    new_h, new_w = int(H * scale), int(W * scale)
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to LORA_INPUT_SIZE x LORA_INPUT_SIZE with LORA_PAD_VALUE (the trained pad value)
    pad_h = LORA_INPUT_SIZE - new_h
    pad_w = LORA_INPUT_SIZE - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(LORA_PAD_VALUE, LORA_PAD_VALUE, LORA_PAD_VALUE),
    )
    
    # 2. LAB-CLAHE on L channel (same as v3)
    lab = cv2.cvtColor(padded, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 3. To float and ImageNet normalize
    arr_f = rgb_clahe.astype(np.float32) / 255.0
    arr_f = (arr_f - IMAGENET_MEAN) / IMAGENET_STD
    
    return torch.from_numpy(arr_f.transpose(2, 0, 1))
```

**Why letterbox and not stretch for LoRA:** the LoRA model was trained with letterbox padding at value 114. Stretching would distort aspect ratios in a way the model has not learned to handle. Specifically, leaves photographed in portrait vs landscape phone orientation produce different feature responses if not aspect-preserving.

**Why pad value 114 specifically:** that is the value used during LoRA training. Other padding values (0 = black, 128 = mid-grey) produce different model behavior because the model has learned to distinguish "real leaf pixels" from "background pad pixels" implicitly. 114 is approximately a neutral grey when ImageNet-normalized. The constant `LORA_PAD_VALUE = 114` lives in `tomato_sandbox/config.py`.

### 7.4 Pipeline 3 — for PSV (Signal C)

```python
def preprocess_for_psv(pil_image: PIL.Image.Image) -> np.ndarray:
    """
    Returns an [H, W, 3] uint8 RGB array with color constancy applied.
    NO LAB-CLAHE, NO tensor conversion. PSV operates on color-corrected RGB
    at native resolution, capped at 1200 px on the longest side to bound CPU cost.
    """
    rgb = np.array(pil_image, dtype=np.uint8)
    H, W = rgb.shape[:2]
    if max(H, W) > 1200:
        scale = 1200 / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb_cc = shades_of_gray(rgb, p=6)
    return rgb_cc
```

This is the canonical and only version. Implementations must include the resize cap; it bounds PSV CPU cost.

The `shades_of_gray` function:

```python
def shades_of_gray(img: np.ndarray, p: int = 6) -> np.ndarray:
    """
    Shades-of-Gray color constancy (Finlayson & Trezzi 2004).
    p=1 is grey-world; p=infinity is max-RGB; p=6 was empirically best
    in the original paper and validated for our use in the PDA review.
    """
    img_f = img.astype(np.float64)
    # Minkowski p-norm of each channel
    illuminant = np.power(np.mean(img_f ** p, axis=(0, 1)), 1.0 / p)  # shape (3,)
    # Normalize: scale each channel so max-channel illuminant becomes 1
    scale = illuminant.max() / illuminant
    img_corrected = img_f * scale
    return np.clip(img_corrected, 0, 255).astype(np.uint8)
```

**Why Shades-of-Gray and not grey-world:** validated in the PDA review's Round 2 empirical tests. Shades-of-Gray with p=6 reduced hue variance across simulated illuminants by a factor of 20 compared to no preprocessing, and outperformed grey-world (p=1) on ambiguous cases.

**Why no CLAHE for PSV:** CLAHE alters the color statistics PSV measures (hue means, saturation distributions, lesion area definitions). PSV's compatibility scores depend on raw color relationships, not local-contrast-enhanced ones. Applying CLAHE before PSV would invalidate the F.0 calibration of HSV thresholds.

**Why no resize for PSV (with cap):** PSV's lesion detection benefits from native resolution. A small lesion that occupies 100 pixels in a 1200px image becomes 4 pixels at 224px — likely lost in segmentation. PSV runs at the original size up to a cap of 1200×1200 (resized down if larger, to bound CPU cost). The cap is a tradeoff: too low and small lesions are missed; too high and PSV becomes the latency bottleneck. 1200 is the empirical sweet spot.

### 7.5 Caching preprocessed outputs across signals

The three pipelines share a common starting point (the validated PIL image). They produce different intermediates (LAB image for v3 and LoRA, color-constancy-applied RGB for PSV). 

If a request triggers TTA (Section 11), the preprocessing pipelines run multiple times — once per augmented view. Caching the unaugmented preprocessed tensors as a baseline avoids redundant work for the no-augmentation pass.

For multi-image requests (Section 18), each image gets its own preprocessed outputs. No cross-image caching.

### 7.6 Where this lives in the sandbox

`tomato_sandbox/preprocessing.py` defines all three preprocessing functions plus `shades_of_gray`. Each signal's module imports the appropriate function. Constants (CLAHE params, ImageNet mean/std, pad value 114) come from `tomato_sandbox/config.py`.

### 7.7 Performance budget

Estimates assume a 4000×3000 (~12 MP) input image, typical for a 2020-era smartphone camera. Smaller inputs are proportionally faster.

For a 4000×3000 input image:
- For v3 (resize to 224, CLAHE, normalize): ~25 ms
- For LoRA (letterbox to 392, CLAHE, normalize): ~30 ms
- For PSV (resize cap to 1200, color constancy): ~50 ms

Total preprocessing for all three signals on one image: ~105 ms median. Estimates pending empirical measurement during Phase A; if measured values diverge by more than 20%, Section 4.6 latency table will be updated to match.

---

## Section 8. Signal A — v3 Model (10-class)

### 8.1 What v3 is

v3, also called Model 3, is the tomato+chilli specialist trained earlier in the project. Its architecture:

```
Input: [B, 3, 224, 224]  (ImageNet-normalized, LAB-CLAHE preprocessed)
        │
        ▼
   DINOv2-Small with registers
   (vit_small_patch14_reg4_dinov2 from timm)
        │
        ▼
   LoRA adapters (rank 4)  ← trained
        │
        ▼
   Squeeze-and-Excitation (SE) block  ← trained
        │
        ▼
   MixStyle layer  ← trained
        │
        ▼
   HardFiLM with crop_mode conditioning  ← trained
        │  (crop_mode is a tensor; we always pass torch.tensor([2]) for tomato)
        ▼
   Linear(384, 10)  ← trained
        │
        ▼
   Output: [B, 10] logits
```

The 10 output classes follow the v3 index space defined in Section 2.4 (0=foliar, 1=late_blight, 2=septoria, 3=ylcv, 4=mosaic, 5=healthy, 6=chilli_leaf_curl, 7=chilli_healthy, 8=chilli_cercospora, 9=chilli_anthracnose). The first 6 are tomato; the last 4 are chilli.

The model lives at `scripts/model3_training/checkpoints/model3_production_v3.pt` (sacred file outside the sandbox; sandbox loads it read-only).

### 8.2 Forward pass

```python
def signal_a_forward(model, x: torch.Tensor) -> dict:
    """
    x: [B, 3, 224, 224] tensor on the GPU.
    Returns a dict with:
      - "logits": [B, 10] raw logits
      - "probs": [B, 10] softmax probabilities
      - "ok": bool indicating numerical validity
    """
    model.eval()
    with torch.no_grad():
        # crop_mode = TOMATO_CROP_MODE_INDEX (=2) signals tomato-conditioned features
        crop_mode = torch.full(
            (x.shape[0],),
            TOMATO_CROP_MODE_INDEX,
            dtype=torch.long,
            device=x.device,
        )
        out = model(x, crop_mode=crop_mode, domain_labels=None)
        logits = out["logits"]  # [B, 10]
        # NaN/Inf guard: rare but possible from numerical instability
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return {"logits": None, "probs": None, "ok": False}
        probs = torch.softmax(logits, dim=1)
    return {"logits": logits, "probs": probs, "ok": True}
```

**Why `crop_mode = TOMATO_CROP_MODE_INDEX`:** the model is multi-task (tomato + chilli). HardFiLM uses `crop_mode` to gate features for the active crop. Passing 2 (tomato) tells the model to use tomato-conditioned feature pathways. Wrong values produce silently degraded predictions. The constant lives in `tomato_sandbox/config.py` (Section 7.2).

**Why `domain_labels=None`:** MixStyle uses domain labels during training to adapt; at inference it is a no-op when None is passed.

**Why the NaN guard:** rare but possible if input has unusual values (e.g., post-augmentation extremes). Without the guard, a NaN forward propagates silently through softmax and argmax, producing nonsense classifications. The guard returns `ok=False`, which the caller treats as a forward-pass failure (Section 8.6).

**Wrapper assumption.** The pseudocode treats `model` as a callable returning `dict[str, Tensor]` with key `"logits"`. The sandbox's model loader (Section 21) wraps the raw checkpoint state-dict in a small adapter class that exposes this interface uniformly, regardless of whether the saved checkpoint was a `nn.Module` or a `state_dict` blob. The concrete loader code lives in `tomato_sandbox/signals/v3_signal.py`.

**TTA wrapping.** When TTA is active (Section 11), the TTA controller calls `signal_a_forward` once per augmented view. Each call is independent. Section 11 defines how the per-view outputs are aggregated.

### 8.3 Extracting tomato probabilities and chilli leakage

The 10-dimensional output contains both tomato and chilli probabilities. The pipeline needs:
1. The 6 tomato probabilities, in canonical order (Section 2.4)
2. The chilli leakage signal — sum of the 4 chilli probabilities

```python
def extract_v3_outputs(probs_10d: torch.Tensor) -> dict:
    """
    probs_10d: [10] vector of v3 softmax probabilities.
    Returns:
      tomato_probs_canonical: [6] vector in canonical ordering (foliar, septoria, late_blight, ylcv, mosaic, healthy)
      chilli_leakage: float, sum of probs at v3 indices 6, 7, 8, 9
      raw_probs_v3_order: [10] vector in v3 ordering (kept for diagnostics only)
    """
    p = probs_10d.cpu().numpy()
    tomato_v3 = p[0:6]  # v3 ordering: foliar, late_blight, septoria, ylcv, mosaic, healthy
    chilli_leakage = float(p[6] + p[7] + p[8] + p[9])
    
    # Remap v3 -> canonical using LORA_INDEX_FOR_V3_CLASS = [0, 2, 1, 3, 4, 5]
    # (Per Section 2.4, this remap applies in both directions because the swap is between positions 1 and 2)
    remap = np.array([0, 2, 1, 3, 4, 5])
    tomato_canonical = np.zeros(6, dtype=np.float32)
    for v3_idx in range(6):
        canonical_idx = remap[v3_idx]
        tomato_canonical[canonical_idx] = tomato_v3[v3_idx]
    
    return {
        "tomato_probs_canonical": tomato_canonical,
        "chilli_leakage": chilli_leakage,
        "raw_probs_v3_order": p,  # for diagnostics only; production logic uses canonical
    }
```

**Important: the 6 tomato probs do NOT sum to 1 after extraction** — they sum to `(1 - chilli_leakage)`. This is by design. Re-normalizing to sum 1 would erase the chilli leakage signal that downstream relies on. The classifier (Section 12) sees the un-renormalized 6-dim vector.

This matches APIN's design choice (Signal 2 uses raw EfficientNet sigmoid outputs without re-normalization, per memory).

### 8.4 chilli_leakage as a misrouting signal

`chilli_leakage` is interpreted as: "v3 thinks this image is more like chilli than tomato." High values indicate the router probably misrouted the request, OR the image genuinely is chilli and the router was right but the user uploaded to the tomato endpoint.

The threshold for "high" is `TOMATO_CHILLI_LEAKAGE_THRESHOLD` (default 0.40, F.0-calibrated to the 95th percentile of confirmed-tomato images per Section 4.5).

The sole consumer of `chilli_leakage` in the pipeline is:
- The hierarchical classifier (Section 12): receives `chilli_leakage` as one of the 19 input features. The full 19-dim breakdown is: 6 v3 canonical probs + 6 LoRA canonical probs + 4 PSV summary features (top-1 score, agreement-with-v3, agreement-with-LoRA, margin) + 1 JSD (Jensen-Shannon divergence between v3 and LoRA) + 1 PSV reliability + 1 chilli_leakage. Section 12 defines each precisely.
- The tier assignment (Section 14): uses the threshold to fire Tier 3C "PSV broken or chilli leakage"

`chilli_leakage` is NOT used to short-circuit the pipeline (i.e., the pipeline does not bail out early on high chilli leakage). Instead the signal flows through the system and the tier logic decides what to do.

### 8.5 No per-signal calibration

v3 outputs are NOT calibrated by a per-signal temperature scaling. There is no `T_v3_tomato`. This is a deliberate change from earlier drafts:

Earlier plans called for fitting a temperature `T_v3_tomato` on the `confusable_pair_probe` set (28 images, foliar/septoria only). The PDA review identified this as flawed because:
- 28 images is too few for reliable calibration
- The probe covers only 2 of 6 classes; calibration is biased toward foliar/septoria
- A temperature scaling fit on 2 classes does not generalize to 6

The replacement: calibration happens once at the end of the pipeline, on the stacking classifier's output, using Platt scaling on out-of-fold predictions across all 7 classes (including OOD). See Section 12 for details. This means v3's raw softmax goes into the classifier; the classifier learns to interpret the un-calibrated v3 distribution.

### 8.6 Output structure

```python
@dataclass
class SignalAResult:
    tomato_probs_canonical: np.ndarray       # [6], canonical ordering
    tomato_max_prob_canonical: float         # max of tomato_probs_canonical
    tomato_argmax_canonical: int             # index 0-5 of max in canonical
    chilli_leakage: float                    # in [0, 1]
    raw_probs_v3_order: np.ndarray | None    # [10], raw v3 output (kept for monitoring/debug)
    forward_succeeded: bool                  # True unless an exception or NaN occurred
    failure_reason: str | None               # "exception" | "numerical_instability" | None
```

The torch→numpy conversion happens inside `extract_v3_outputs` (Section 8.3). All fields of `SignalAResult` are numpy arrays or Python scalars; no torch tensors leak out. This boundary keeps downstream code (classifier, response builder) free of torch dependencies.

`raw_probs_v3_order` (the original 10-class output in v3 ordering) is kept for transparency in the agronomist view and monitoring dashboard. Production logic uses only the canonical fields.

If the forward pass throws (e.g., CUDA OOM) or returns NaN, `forward_succeeded = False` and the prob-shaped fields are zero-filled. The classifier (Section 12) was trained with degraded-mode augmentation (Section 12.4) that simulates this case, so the system handles the failure gracefully.

### 8.7 Where this lives in the sandbox

`tomato_sandbox/signals/v3_signal.py` defines `SignalAResult` and `compute_signal_a(preprocessed_tensor: torch.Tensor) -> SignalAResult`. The pipeline orchestrator (Section 21) loads the v3 model at startup and passes it to this function.

The wiring of forward + extract + result assembly looks like this (the implementer can copy this scaffolding):

```python
def compute_signal_a(model, tensor: torch.Tensor) -> SignalAResult:
    try:
        fwd = signal_a_forward(model, tensor.unsqueeze(0))  # add batch dim
    except Exception as e:
        return SignalAResult(
            tomato_probs_canonical=np.zeros(6, dtype=np.float32),
            tomato_max_prob_canonical=0.0,
            tomato_argmax_canonical=0,
            chilli_leakage=0.0,
            raw_probs_v3_order=None,
            forward_succeeded=False,
            failure_reason=f"exception: {type(e).__name__}",
        )
    if not fwd["ok"]:
        return SignalAResult(
            tomato_probs_canonical=np.zeros(6, dtype=np.float32),
            tomato_max_prob_canonical=0.0,
            tomato_argmax_canonical=0,
            chilli_leakage=0.0,
            raw_probs_v3_order=None,
            forward_succeeded=False,
            failure_reason="numerical_instability",
        )
    extracted = extract_v3_outputs(fwd["probs"][0])  # [0] removes batch dim
    return SignalAResult(
        tomato_probs_canonical=extracted["tomato_probs_canonical"],
        tomato_max_prob_canonical=float(extracted["tomato_probs_canonical"].max()),
        tomato_argmax_canonical=int(extracted["tomato_probs_canonical"].argmax()),
        chilli_leakage=extracted["chilli_leakage"],
        raw_probs_v3_order=extracted["raw_probs_v3_order"],
        forward_succeeded=True,
        failure_reason=None,
    )
```

The v3 weights are loaded read-only at startup from `scripts/model3_training/checkpoints/model3_production_v3.pt` (sacred file outside the sandbox per Section 2.6; the sandbox does not modify or copy this file, only reads it into GPU memory).

### 8.8 Performance budget

For one image at 224×224 on RTX 4060:
- GPU compute (forward pass through DINOv2-Small + adapters): ~80 ms
- Output extraction and remap: <1 ms

For TTA (Section 11), forward pass runs once per view. Cumulative GPU time for v3 alone:
- 1-view (no TTA): 80 ms
- 2-view TTA: 160 ms
- 5-view TTA: 400 ms

These are added to the total GPU budget shown in Section 4.6.

---

## Section 9. Signal B — Single-Pass LoRA (epoch 13)

### 9.1 What single-pass LoRA is

Single-pass LoRA is a tomato-only specialist trained later in the project. The "single-pass" name refers to the training procedure (one stage, no curriculum, no multi-pass refinement). Architecture:

```
Input: [B, 3, 392, 392]  (ImageNet-normalized, LAB-CLAHE, letterbox padded)
        │
        ▼
   DINOv2-Base with registers
   (vit_base_patch14_reg4_dinov2 from timm; FROZEN — no trainable params)
        │
        ▼
   LoRA adapters on transformer blocks 4-11   ← trained (rank 4)
   (DINOv2-Base has 12 transformer blocks, zero-indexed as 0-11.
    "Blocks 4-11" means the last 8 of the 12 blocks have adapters.
    Blocks 0-3 are frozen with no adapter. There is no block 12.)
        │
        ▼
   CLS token output: [B, 768]
        │
        ▼
   Linear(768, 6)  ← trained
        │
        ▼
   Output: [B, 6] logits
```

Output is 6 classes in **LoRA index ordering** (Section 2.4): 0=foliar, 1=septoria, 2=late_blight, 3=ylcv, 4=mosaic, 5=healthy. This ordering matches canonical, so no remap is needed for LoRA → canonical.

The model lives at `tomato_sandbox/models/tomato_sp_lora_production.pt` (renamed from the original `sp_lora_epoch13_f10.9113_PRESERVED.pt`).

### 9.2 Forward pass

```python
def signal_b_forward(model, x: torch.Tensor) -> dict:
    """
    x: [B, 3, 392, 392] tensor on the GPU.
    Returns a dict with:
      - "logits": [B, 6] raw logits
      - "probs": [B, 6] softmax probabilities
      - "cls_token": [B, 768] CLS token features (used for prototype matching)
      - "ok": bool indicating numerical validity
    """
    model.eval()
    with torch.no_grad():
        # The model wrapper exposes a uniform forward returning a dict.
        # Section 21 (sandbox server) ensures all signal models implement this contract.
        out = model(x)
        logits = out["logits"]          # [B, 6]
        cls_token = out["cls_token"]    # [B, 768]
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return {"logits": None, "probs": None, "cls_token": None, "ok": False}
        probs = torch.softmax(logits, dim=1)
    return {"logits": logits, "probs": probs, "cls_token": cls_token, "ok": True}
```

**Why the model exposes both `logits` and `cls_token`:** the LoRA head (Linear(768, 6)) consumes the CLS token to produce logits. The pipeline needs both: logits for the classification probability, and the CLS token for prototype-bank similarity computation (Section 9.4). The sandbox's model loader (Section 21) wraps the saved checkpoint with an adapter that exposes this contract uniformly, regardless of internal model structure. This abstraction means the spec doesn't depend on whether the saved model is a `nn.Module`, a `state_dict`, or a custom wrapper — the loader normalizes them.

**Why CLS token specifically:** DINOv2 ViTs produce two types of patch tokens: the CLS token (a global image summary) and the patch tokens (per-patch features). The single-pass LoRA's classification head reads only the CLS token. Patch tokens are not used for classification but could be used in future work for spatial attention or saliency.

**TTA wrapping.** When TTA is active (Section 11), the TTA controller calls `signal_b_forward` once per augmented view. Each call returns its own logits, probs, and CLS token. Section 11 defines how the per-view outputs are aggregated. Note that prototype blending (Section 9.5) operates on the aggregated post-TTA distribution, not on each view individually.

### 9.3 No per-signal calibration

Same as Signal A: no `T_sp_lora` per-signal temperature. The single calibration step happens on the stacking classifier output (Section 12).

### 9.4 Prototype bank

When LoRA's max probability is low (below `TOMATO_PROTOTYPE_BLEND_THRESHOLD`, default 0.60, F.0-calibrated), the pipeline uses a **prototype bank** to nudge the prediction toward the closest known examples.

#### What the prototype bank is

A prototype bank is a small library of CLS-token vectors taken from high-confidence LoRA predictions on `field_val`. For each tomato class, the bank stores up to 10 prototypes — CLS-token vectors of images the LoRA model classified correctly with high confidence.

```python
@dataclass
class PrototypeBank:
    prototypes: dict[int, np.ndarray]  # canonical_class_idx -> [N_class, 768] CLS tokens
    class_counts: dict[int, int]       # class_idx -> number of prototypes
    underpopulated_classes: set[int]   # classes with fewer than 3 prototypes
    model_version: str                 # 7-char hash of the LoRA weights this bank was built from
```

Build procedure (Phase A.1, see Section 29 for Phase A definitions; runs once before deployment):
1. Run LoRA on all 160 images of the `field_val` train_subset (the 80% slice of the 203-image `field_val`; see Section 1.6 glossary)
2. For each class, collect images where LoRA's max probability ≥ 0.85 AND its argmax matches the ground-truth label
3. From those high-confidence correct predictions, save the CLS tokens
4. Per class: if more than 10 are available, randomly sample 10. If fewer than 3, mark the class as `underpopulated`. If 0 examples are available for a class, mark `underpopulated` and store `prototypes[cls_idx] = np.empty((0, 768))` (an empty array placeholder so all 6 keys are always present in the dict)
5. Compute the SHA-256 of the LoRA weight file used; store its first 7 chars as `model_version`

The blend code (Section 9.5) checks `cls_idx in bank.underpopulated_classes` BEFORE accessing `bank.prototypes[cls_idx]`, so empty arrays are never indexed. This invariant must hold: any class in `underpopulated_classes` has its prototypes either missing or empty; any class NOT in `underpopulated_classes` has at least 3 prototypes.

**Bias acknowledgment.** The prototype bank is built from images in the training distribution (field_val train_subset). Images that LoRA was already trained on are over-represented in the bank because the model is biased to predict correctly on them (training-set memorization effect). The bank therefore represents what the model thinks "good" looks like in-distribution, not the broader feature space. This is an acknowledged limitation; for substantially out-of-distribution images, the prototype-blending step provides limited benefit. Section 30 (honest limitations) discusses this further.

Underpopulated classes (typically YLCV, possibly mosaic, due to small `field_val` counts) cannot be reliably blended; for those the pipeline skips blending and uses the raw LoRA output. This is documented as a graceful degradation rather than a hidden fallback.

**Model-version validation at startup.** When the sandbox loads the prototype bank, it compares `bank.model_version` to the first 7 chars of the SHA-256 of the currently-loaded LoRA weight file. If they do not match, the bank was built from a different model checkpoint and the CLS-token feature space is incompatible. The sandbox refuses to start in this case (per Section 4.4 startup failure handling), with an error message identifying the mismatch. This prevents silent corruption from a stale bank.

#### Why CLS-token nearest-prototype helps

When LoRA has low confidence, its softmax distribution is spread across multiple classes. The CLS token, however, often lands close to one or two known prototypes. Distance to prototypes is a different signal than softmax confidence and can disambiguate.

This is empirically validated practice from few-shot learning literature. The exact prototype-blend weights need F.0 tuning.

### 9.5 Prototype blending

When `lora_max_prob < TOMATO_PROTOTYPE_BLEND_THRESHOLD`, the pipeline blends LoRA's softmax with a prototype-similarity distribution.

```python
def prototype_blend(
    lora_probs: np.ndarray,            # [6], LoRA softmax in canonical order
    cls_token: np.ndarray,             # [768], current image's CLS
    bank: PrototypeBank,
    T_proto: float = T_PROTO,          # softmax temperature for prototype distances
    blend_weight: float = BLEND_WEIGHT,  # how much weight on prototype distribution
) -> np.ndarray:
    """
    Returns a blended [6] probability distribution.
    """
    # Compute cosine similarity from current CLS to each prototype
    cls_norm = cls_token / (np.linalg.norm(cls_token) + 1e-8)
    
    # Per-class average similarity
    per_class_sim = np.zeros(6)
    for cls_idx in range(6):
        if cls_idx in bank.underpopulated_classes:
            per_class_sim[cls_idx] = -np.inf  # effectively excludes from softmax
            continue
        protos = bank.prototypes[cls_idx]  # [N_class, 768]
        protos_norm = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
        sims = protos_norm @ cls_norm  # [N_class]
        per_class_sim[cls_idx] = sims.max()  # closest prototype in this class
    
    # Convert similarities to a probability via softmax / T_proto
    # Underpopulated classes had -inf, so they get 0 prob in this distribution
    finite_mask = np.isfinite(per_class_sim)
    sim_probs = np.zeros(6)
    if finite_mask.any():
        sims_finite = per_class_sim[finite_mask]
        sims_softmax = np.exp(sims_finite / T_proto) / np.exp(sims_finite / T_proto).sum()
        sim_probs[finite_mask] = sims_softmax
    else:
        sim_probs = lora_probs.copy()  # fall back to LoRA if all classes underpopulated
    
    # Blend
    blended = (1 - blend_weight) * lora_probs + blend_weight * sim_probs
    # Renormalize: necessary because sim_probs can sum to less than 1 when
    # underpopulated classes are zeroed out of the similarity distribution.
    # If sim_probs sums to 1 (no underpopulated classes), this renormalize is a no-op.
    blended = blended / blended.sum()
    return blended
```

**Pinned constants** (live in `tomato_sandbox/config.py`, loaded from F.0 calibration outputs at startup; placeholder defaults shown):
- `T_PROTO = 0.3` — temperature for similarity-to-prob conversion. F.0 sweeps to find the value that yields prototype softmax entropy in [0.5, 1.5] nats. Lower temperature is more confident; higher is more uniform.
- `BLEND_WEIGHT = 0.35` — how much weight the prototype distribution gets relative to LoRA. Default 0.35 means LoRA contributes 65% and prototypes contribute 35%. F.0 sweep range: [0.25, 0.50].

**Latency note on the 10-per-class cap.** The 10-per-class cap (Section 9.4) bounds the number of cosine similarities to at most 60 (6 classes × 10). At 768 dims per CLS token, that is 60 × 768 ≈ 46K multiply-adds per blend call, well under 5 ms. Raising the cap would scale similarity compute linearly; lowering risks losing prototype diversity within a class. F.0 may revisit if measurements show different optimal values.

**When blending is skipped:**
- If `lora_max_prob >= TOMATO_PROTOTYPE_BLEND_THRESHOLD` (high LoRA confidence): use raw LoRA output, don't blend.
- If the prototype bank failed to load at startup: sandbox didn't start (per startup failure handling); this case never reaches inference.
- If all 6 classes are underpopulated (very rare; would mean field_val is unusable): degrade to raw LoRA.

### 9.6 Output structure

```python
@dataclass
class SignalBResult:
    tomato_probs_canonical: np.ndarray         # [6], possibly blended with prototypes
    tomato_max_prob_canonical: float
    tomato_argmax_canonical: int
    cls_token: np.ndarray                      # [768], for monitoring/debug
    raw_lora_probs_canonical: np.ndarray       # [6], the un-blended LoRA softmax (for transparency)
    prototype_blend_applied: bool              # True if prototype blending was triggered
    prototype_blend_reason: str                # "low_confidence" | "high_confidence_no_blend" | "all_classes_underpopulated"
    forward_succeeded: bool                    # True unless an exception or NaN occurred
    failure_reason: str | None                 # "exception" | "numerical_instability" | None
```

Both raw and blended distributions are kept. The classifier (Section 12) reads the (possibly-blended) `tomato_probs_canonical` as 6 of its 19 input features. The raw distribution is exposed in `tomato_details` of the response (Section 16) for transparency and to let the agronomist view (Section 23) display "what would LoRA have said without prototype blending."

Note that `bank_unavailable` is not in the list of `prototype_blend_reason` values: if the bank failed to load, the sandbox would not have started (Section 4.4). At inference time the bank is always available.

### 9.7 Where this lives in the sandbox

- `tomato_sandbox/signals/lora_signal.py` defines `SignalBResult` and `compute_signal_b(preprocessed_tensor, prototype_bank) -> SignalBResult`.
- `tomato_sandbox/signals/prototype_bank.py` defines `PrototypeBank`, the loader, and the version-validation logic.
- `tomato_sandbox/phase_f0_calibration/prototype_bank.npz` is the on-disk artifact loaded at startup. Built by Phase A.1 (Section 29).
- The single-pass LoRA weights load read-only at startup from `tomato_sandbox/models/tomato_sp_lora_production.pt` (sandbox-local, becomes sacred after Phase A.3 per Section 2.6).

### 9.8 Performance budget

For one image at 392×392 on RTX 4060:
- GPU compute (DINOv2-Base + LoRA adapters + linear head): ~120 ms
- Prototype similarity computation (when triggered): ~5 ms (60 cosine similarities at most)
- Output extraction: <1 ms

LoRA is the slower of the two neural signals due to the larger backbone (Base vs Small) and larger input size (392 vs 224).

For TTA (Section 11), forward pass runs once per view. Cumulative GPU time for LoRA alone:
- 1-view (no TTA): 120 ms
- 2-view TTA: 240 ms
- 5-view TTA: 600 ms

Combined Sections 8.8 + 9.8 + router (~30 ms): single-image-no-TTA total GPU = 80 + 120 + 30 = 230 ms (Section 4.6 rounds to 250 ms with kernel-launch and lock-acquisition overhead). Single-image-5-view-TTA total GPU = 400 + 600 + 30 = 1030 ms (Section 4.6 rounds to 1.2 s with augmentation generation overhead included).

---

## Section 10. Signal C — PSV (Plant Symptom Visual)

### 10.1 What PSV is and why it exists

PSV is the classical computer vision signal in the tomato pipeline. It reads the color-constancy-applied RGB image (Section 7.4) and produces 26 hand-engineered features about leaf shape, lesion patterns, color distribution, texture, and vegetation state. From those 26 features it computes 6 botanical compatibility scores — one per tomato class — using fixed weights (no learned parameters). The compatibility scores then feed into the hierarchical classifier (Section 12) alongside v3 and LoRA outputs.

PSV exists for one reason: the two neural signals (v3, LoRA) sometimes confuse classes that have similar visual textures in low-resolution feature maps but are clearly distinguishable by classical metrics. Specific examples documented in the PDA review:
- **Mosaic vs foliar.** Both produce blotchy patches on tomato leaves. v3 and LoRA confuse them because both look like "patchy leaves" in their pretrained features. PSV distinguishes them via spatial dispersion (mosaic affects whole leaf; foliar is localized) and lesion size statistics (mosaic has no discrete lesions; foliar has many small ones).
- **YLCV vs healthy with light yellowing.** Both can show pale-green leaf surfaces. v3 and LoRA see "yellowish leaf" and may pick either. PSV distinguishes via marginality (YLCV yellows from the margins inward; light healthy yellowing is uniform) and leaf curl signatures (geometry features).

PSV is not a backup for the neural models — it is a complementary signal. The classifier learns which signal to trust under which conditions (Section 12).

### 10.2 Stages of the PSV pipeline

PSV runs in five stages, all on CPU. Each stage takes the previous stage's output and adds intermediate artifacts.

```
preprocess_for_psv output (Section 7.4)
    │  [H, W, 3] uint8 RGB, color-constancy applied
    ▼
┌──────────────────────────────────────────────┐
│ Stage 1: Leaf segmentation                   │
│ - HSV conversion                             │
│ - Otsu threshold on saturation               │
│ - Morphology (open + close)                  │
│ - Largest-component extraction               │
│ Output: leaf_mask [H, W] bool                │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 2: Disease region detection            │
│ - Compute deviation-from-healthy color       │
│ - Threshold to get disease_mask              │
│ - Morphology cleanup                         │
│ - Connected components                       │
│ Output: disease_mask [H, W] bool             │
│         lesion_components (CC stats)         │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 3: 26 feature computation              │
│ - 8 groups of features, computed in order    │
│ Output: features dict (26 entries; index 21  │
│         psv_aggregate_reliability is filled  │
│         in by Stage 5 and is a placeholder   │
│         here)                                │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 4: Compatibility scoring               │
│ - Apply fixed botanical weight matrix        │
│ - Normalize to softmax                       │
│ Output: 6 c_* scores in canonical order      │
│ (uses placeholder zero at feature index 21;  │
│  weight matrix has zero weights for that     │
│  feature across all classes, so the output   │
│  is unaffected by the placeholder)           │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 5: Reliability assessment              │
│ - Sanity check vs IQA mask                   │
│ - Combine quality signals                    │
│ - Fill in feature index 21 in feature vector │
│ Output: psv_aggregate_reliability ∈ [0, 1]   │
└──────────────────────────────────────────────┘
    │
    ▼
SignalCResult
```

Each stage is documented below. The order matters — later stages depend on earlier outputs.

**On the apparent circularity between Stage 4 and Stage 5.** Stage 4 reads the 26-feature vector, which includes `psv_aggregate_reliability` at index 21. But `psv_aggregate_reliability` is computed by Stage 5, which runs after Stage 4. The resolution: at Stage 4, feature index 21 is a placeholder zero. The compatibility weight matrix (Section 10.6.1) has zero weights for that feature across all 6 classes, so the placeholder value does not affect the compatibility output. Stage 5 then computes the real reliability and writes it to the feature vector before SignalCResult is finalized. The matrix's zero-weight assignment for feature 21 is the design device that breaks the circularity cleanly.

**Orchestrator wiring.** The function `compute_signal_c` ties the five stages together:

```python
def compute_signal_c(
    rgb_cc: np.ndarray,                  # [H, W, 3] from preprocess_for_psv
    iqa_green_mask: np.ndarray | None,   # from IQAResult.green_mask
    iqa_aggregate_score: float,          # from IQAResult.aggregate_score
) -> SignalCResult:
    try:
        # Stage 1
        leaf_mask = segment_leaf(rgb_cc)
        fallback_used = False
        if leaf_mask.sum() == 0:
            # Fallback: use IQA's mask if available
            if iqa_green_mask is not None and iqa_green_mask.sum() > 0:
                leaf_mask = iqa_green_mask
                if leaf_mask.shape != rgb_cc.shape[:2]:
                    leaf_mask = cv2.resize(
                        leaf_mask.astype(np.uint8),
                        (rgb_cc.shape[1], rgb_cc.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                fallback_used = True
            else:
                # Both empty: produce zero-features result
                return _empty_psv_result(rgb_cc.shape, fallback_used=False)
        
        # Stage 2
        disease_mask, lesion_stats = detect_disease_regions(rgb_cc, leaf_mask)
        
        # Stage 3 — produces the 26-feature vector with index 21 as placeholder zero
        raw_features = compute_26_features(
            rgb_cc, leaf_mask, disease_mask, lesion_stats, iqa_aggregate_score
        )
        # raw_features[21] is currently 0 (placeholder); will be overwritten by Stage 5
        
        # Stage 4 — compatibility scoring (zero weight on feature 21 means placeholder is OK)
        standardized = standardize_features(raw_features)
        compatibility = compute_compatibility_scores(standardized)
        
        # Stage 5 — reliability; updates raw_features[21]
        reliability = compute_psv_reliability(
            leaf_mask, disease_mask, iqa_green_mask, iqa_aggregate_score,
            n_lesions=lesion_stats["n_lesions"],
        )
        if fallback_used:
            reliability = max(0.1, 0.3 * iqa_aggregate_score)
        raw_features[21] = reliability
        # Re-standardize index 21 with the real value (other indices unchanged)
        standardized[21] = np.clip(
            (reliability - F0_FEATURE_MEAN[21]) / (F0_FEATURE_STD[21] + 1e-6), -3, 3
        )
        
        argmax = int(np.argmax(compatibility))
        max_val = float(compatibility[argmax])
        sorted_desc = np.sort(compatibility)[::-1]
        margin = float(sorted_desc[0] - sorted_desc[1])
        
        return SignalCResult(
            compatibility=compatibility,
            compatibility_argmax=argmax,
            compatibility_max=max_val,
            compatibility_margin=margin,
            psv_reliability=reliability,
            raw_features=raw_features,
            standardized_features=standardized,
            leaf_mask=leaf_mask,
            disease_mask=disease_mask,
            n_lesions=lesion_stats["n_lesions"],
            fallback_used=fallback_used,
            forward_succeeded=True,
            failure_reason=None,
        )
    except Exception as e:
        return SignalCResult(
            compatibility=np.full(6, 1.0/6, dtype=np.float32),  # uniform
            compatibility_argmax=0,
            compatibility_max=1.0/6,
            compatibility_margin=0.0,
            psv_reliability=0.05,
            raw_features=np.zeros(26, dtype=np.float32),
            standardized_features=np.zeros(26, dtype=np.float32),
            leaf_mask=np.zeros(rgb_cc.shape[:2], dtype=bool),
            disease_mask=np.zeros(rgb_cc.shape[:2], dtype=bool),
            n_lesions=0,
            fallback_used=False,
            forward_succeeded=False,
            failure_reason=f"exception: {type(e).__name__}",
        )
```

The exception path produces a "PSV failed" result that the classifier handles via degraded-mode features (Section 12). The pipeline never crashes from a PSV bug.

**Note on `standardized_features` consistency.** SignalCResult.compatibility was computed using a placeholder zero at `standardized_features[21]` (the placeholder for psv_aggregate_reliability before Stage 5). The orchestrator then writes the real reliability value to `standardized_features[21]` before returning. This means recomputing `softmax(WEIGHT_MATRIX @ standardized_features)` from the post-finalization `standardized_features` would give the same result as the stored `compatibility` field, because the weight matrix has zero weights at index 21 across all classes (Section 10.6.1). The discrepancy is invisible at the matrix multiply level. Monitoring code that re-derives compatibility from `standardized_features` will get consistent results.

### 10.3 Stage 1 — Leaf segmentation

Goal: produce a binary mask of leaf pixels, distinguishing the leaf from background (table, ground, hand, sky).

```python
def segment_leaf(rgb_cc: np.ndarray) -> np.ndarray:
    """
    rgb_cc: [H, W, 3] uint8, color-constancy applied (Section 7.4)
    Returns: leaf_mask [H, W] bool, True for leaf pixels
    """
    hsv = cv2.cvtColor(rgb_cc, cv2.COLOR_RGB2HSV)
    # Saturation channel emphasizes vegetation; Otsu finds a binary threshold automatically
    sat = hsv[:, :, 1]
    otsu_thresh, sat_mask = cv2.threshold(
        sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    sat_mask = sat_mask.astype(bool)
    
    # Restrict to green-ish hues (drops red/blue saturated objects like clothing, plastic)
    hue = hsv[:, :, 0]
    green_hue = (hue >= 25) & (hue <= 95)
    leaf_candidate = sat_mask & green_hue
    
    # Morphology: open (remove tiny noise specks) then close (fill small gaps in leaf)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(
        leaf_candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel_small
    )
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)
    
    # Keep only the largest connected component (the main leaf)
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if nb <= 1:
        return np.zeros_like(cleaned, dtype=bool)
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    leaf_mask = (labels == largest_idx)
    return leaf_mask
```

**Why Otsu on saturation:** plant material has substantially higher saturation than backgrounds like soil, sky, hand. Otsu finds the natural threshold without a hardcoded value. The hue restriction afterward removes saturated non-vegetation (red flowers, blue plastic).

**Why open-then-close morphology:** open removes salt-and-pepper noise from the threshold; close fills small holes inside the leaf body (vein gaps, light specks). Kernel sizes scale proportionally with the PSV resize cap (Section 7.4):
- `PSV_OPEN_KERNEL_SIZE = max(3, PSV_RESIZE_CAP // 240)` — at the default cap of 1200, this gives 5
- `PSV_CLOSE_KERNEL_SIZE = max(9, PSV_RESIZE_CAP // 80)` — at the default cap of 1200, this gives 15

If the resize cap changes (e.g., to 1600 or 800), kernel sizes auto-adjust. This avoids the soft coupling where a maintainer changing the cap would need to remember to update kernel sizes manually. The constants are computed once at module load time from `PSV_RESIZE_CAP` in `tomato_sandbox/config.py`.

**Why largest component:** when multiple leaves are present, we segment only the dominant one. Multi-leaf images are flagged earlier by IQA's `background_contamination` dimension; if such an image reaches PSV, only the largest leaf is analyzed.

**Failure modes:**
- Empty mask (no leaf detected at all): handled in Stage 5 by setting all 26 features to NaN-or-zero and reducing reliability heavily. PSV does not throw; it returns a low-reliability result.
- Multiple leaves of similar size: only the topologically largest connected component survives. The discarded leaves are silently dropped.
- Leaf occupying near-100% of image: works correctly; the largest component is the whole foreground.

### 10.4 Stage 2 — Disease region detection

Goal: within the leaf, find pixels that look diseased (different from typical healthy leaf color).

```python
def detect_disease_regions(
    rgb_cc: np.ndarray, leaf_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Returns:
      disease_mask: [H, W] bool, disease pixels within leaf
      lesion_stats: dict with connected-component info
    """
    if leaf_mask.sum() == 0:
        return np.zeros_like(leaf_mask), {"n_lesions": 0, "components": None}
    
    # Compute the median color of healthy-looking leaf pixels.
    # We assume the most common leaf color (mode of HSV hue) is healthy green.
    hsv = cv2.cvtColor(rgb_cc, cv2.COLOR_RGB2HSV)
    leaf_hsv = hsv[leaf_mask]
    hue_median = float(np.median(leaf_hsv[:, 0]))
    sat_median = float(np.median(leaf_hsv[:, 1]))
    val_median = float(np.median(leaf_hsv[:, 2]))
    
    # Disease pixels deviate from the healthy median in HSV space.
    # Deviation distance in HSV with hue weighted more (color shift is most diagnostic).
    H_dev = np.abs(hsv[:, :, 0].astype(np.int32) - hue_median)
    H_dev = np.minimum(H_dev, 180 - H_dev)  # circular distance in hue space
    S_dev = np.abs(hsv[:, :, 1].astype(np.int32) - sat_median)
    V_dev = np.abs(hsv[:, :, 2].astype(np.int32) - val_median)
    
    deviation = 2.0 * H_dev + 1.0 * S_dev + 0.5 * V_dev
    
    # Threshold deviation: disease pixels are those with high deviation AND inside leaf.
    # PSV_DISEASE_DEVIATION_THRESHOLD is loaded from F.0 calibration at startup
    # (phase_f0_calibration/psv_disease_threshold.json). Placeholder value 35.0 is used
    # when F.0 has not yet produced the calibrated value.
    disease_candidate = (deviation > PSV_DISEASE_DEVIATION_THRESHOLD) & leaf_mask
    
    # Morphology: small open to remove noise; small close to consolidate lesions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease_mask = cv2.morphologyEx(
        disease_candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)
    disease_mask = disease_mask.astype(bool)
    
    # Connected components for per-lesion stats
    nb, labels, stats, centroids = cv2.connectedComponentsWithStats(
        disease_mask.astype(np.uint8)
    )
    # nb includes background (label 0); n_lesions is the rest
    n_lesions = nb - 1
    lesion_stats = {
        "n_lesions": n_lesions,
        "labels": labels,
        "stats": stats,           # [N+1, 5]: x, y, w, h, area
        "centroids": centroids,   # [N+1, 2]
        "leaf_area_px": int(leaf_mask.sum()),
        "disease_area_px": int(disease_mask.sum()),
    }
    return disease_mask, lesion_stats
```

**Why HSV-space deviation, not RGB:** HSV separates color from brightness. Disease symptoms are primarily hue shifts (yellow chlorosis, brown necrosis) more than brightness changes. RGB-space deviation conflates the two and misses subtle color signals.

**Why median-as-healthy-baseline:** computing the leaf's typical color from its own pixels avoids needing a fixed "healthy tomato green" reference, which would vary with cultivar, lighting, and growth stage. The assumption is that even on a heavily diseased leaf, more pixels are healthy-baseline than each-individual-disease-color, so the median tracks the healthy value.

**Failure mode:** on a fully diseased leaf (>50% disease pixels), the median of leaf pixels IS the disease color, and the deviation calculation would misidentify healthy patches as "diseased." This is rare in practice — most diagnosable disease photos have substantial healthy area for context. If the failure occurs, the resulting feature values are still passed downstream; the classifier may misclassify, but the system does not crash.

**Why the 2:1:0.5 weighting on H, S, V:** hue shift is the strongest disease signal (yellowing, browning). Saturation drops are secondary (chlorosis often desaturates). Value changes are weakest because lighting affects them too much.

**Threshold of 35.0:** placeholder. F.0 calibrates from labeled lesion masks, sweeping over [20, 30, 40, 50, 60] and selecting the value that maximizes IoU with agronomist-drawn lesion polygons on a 30-image audit set. Lower thresholds catch more lesions but also more noise; higher misses subtle lesions.

### 10.5 Stage 3 — The 26 features

The 26 features are organized into 8 groups by what they measure. Each feature has: a name, a precise formula, an output range, and a brief botanical interpretation.

**Empty-mask handling.** All feature computations assume `leaf_mask.sum() > 0`. If `leaf_mask` is empty (Stage 1 found no leaf and Stage 5 fallback also failed), feature computation is skipped entirely — the orchestrator (Section 10.2) returns an empty-features result with all 26 features set to 0.0 and `forward_succeeded=True, fallback_used=True, psv_reliability=0.05`. The pseudocode below assumes a non-empty leaf and does not show defensive empty-mask checks for clarity. Production code MUST add such checks at function entry, returning the appropriate safe default (0 for ratios and counts, 0 for indices).

#### 10.5.1 Group G1 — Coverage features (3)

How much of the leaf is diseased and how concentrated the disease is.

**G1.1 `disease_coverage_pct` ∈ [0, 100]**
```
disease_coverage_pct = 100 * disease_area_px / leaf_area_px
```
The percentage of the leaf area covered by disease pixels. Higher means more advanced disease. Healthy leaves have near-zero coverage; severe late-blight or septoria can exceed 50%.

**G1.2 `largest_lesion_pct` ∈ [0, 100]**
```
if n_lesions == 0: largest_lesion_pct = 0
else: largest_lesion_pct = 100 * max(stats[1:, cv2.CC_STAT_AREA]) / leaf_area_px
```
Size of the single biggest lesion as a fraction of the leaf. Distinguishes "many small spots" (septoria) from "one big patch" (late blight, advanced foliar spot).

**G1.3 `lesion_count` ∈ ℕ (capped at 200)**
```
lesion_count = min(n_lesions, 200)
```
Number of distinct disease regions. Septoria has many; late blight has few large patches; YLCV has zero (the disease pattern is not lesion-shaped). Cap at 200 to bound classifier feature variance.

#### 10.5.2 Group G2 — Lesion shape features (4)

Geometry of the discrete lesions.

**G2.1 `mean_lesion_size` (in pixels²; clipped at 50000)**
```
if n_lesions == 0: mean_lesion_size = 0
else: mean_lesion_size = clip(mean(stats[1:, cv2.CC_STAT_AREA]), 0, 50000)
```
Average lesion size. Septoria lesions are small (~50-200 px²); foliar spot intermediate (200-1500 px²); late blight large (>2000 px²). Empirically validated in PDA review Round 2: 9× difference between septoria and late blight.

**G2.2 `lesion_size_std` (pixels²; clipped at 50000)**
```
if n_lesions <= 1: lesion_size_std = 0
else: lesion_size_std = clip(std(stats[1:, cv2.CC_STAT_AREA]), 0, 50000)
```
Variance in lesion sizes. Septoria has uniformly small lesions (low std); late blight has wildly variable sizes (high std).

**G2.3 `mean_lesion_circularity` ∈ [0, 1]**
```python
def mean_lesion_circularity(disease_mask, lesion_stats):
    n_lesions = lesion_stats["n_lesions"]
    labels = lesion_stats["labels"]
    if n_lesions == 0:
        return 0.0
    circularities = []
    for i in range(1, n_lesions + 1):  # skip background (label 0)
        component_i = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        # Take the longest contour (largest perimeter) for this component
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter <= 0:
            continue
        area = lesion_stats["stats"][i, cv2.CC_STAT_AREA]
        circ = 4 * np.pi * area / (perimeter ** 2)
        circularities.append(min(circ, 1.0))  # isoperimetric inequality bound
    return float(np.mean(circularities)) if circularities else 0.0
```
1.0 means perfectly circular; lower means irregular. Septoria lesions are nearly circular (high); late blight is irregular and necrotic-edged (low). Bounded [0, 1] because the isoperimetric inequality guarantees circularity ≤ 1.

**G2.4 `edge_sharpness` ∈ [0, 1]**
```python
def edge_sharpness(rgb_cc, disease_mask):
    if disease_mask.sum() == 0:
        return 0.0
    # Edge pixels = disease pixels adjacent to non-disease pixels
    eroded = cv2.erode(disease_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    edge_mask = disease_mask & (~eroded.astype(bool))
    if edge_mask.sum() == 0:
        return 0.0
    # Sobel on the L channel of LAB (perceptually uniform luminance)
    lab = cv2.cvtColor(rgb_cc, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Mean gradient magnitude at edge pixels, normalized by max possible (255)
    return float(np.clip(grad_mag[edge_mask].mean() / 255.0, 0, 1))
```
How sharply the lesion transitions from healthy to diseased pixels. Septoria has well-defined edges (high); foliar spot has soft edges (medium); late blight has very soft / wet edges (low). PDA review Round 2 measured 7.5× difference between septoria and late blight on this feature.

#### 10.5.3 Group G3 — Color statistics features (4)

Distribution of pixel colors within the leaf.

**G3.1 `yellow_pixel_fraction` ∈ [0, 1]**
```
hsv_leaf = hsv[leaf_mask]
yellow_mask = (hsv_leaf[:, 0] >= 15) & (hsv_leaf[:, 0] <= 35) & (hsv_leaf[:, 1] >= 50)
yellow_pixel_fraction = yellow_mask.sum() / leaf_mask.sum()
```
Fraction of leaf pixels that are yellow (chlorosis). High fraction is a strong YLCV signature; mosaic also produces yellow patches. Hue range 15-35 covers yellow through yellow-green.

**G3.2 `brown_pixel_fraction` ∈ [0, 1]**
```
brown_mask = (hsv_leaf[:, 0] >= 5) & (hsv_leaf[:, 0] <= 20) & \
             (hsv_leaf[:, 1] >= 50) & (hsv_leaf[:, 2] < 150)
brown_pixel_fraction = brown_mask.sum() / leaf_mask.sum()
```
Fraction of leaf pixels that are brown (necrosis). Late blight produces brown-black necrotic regions; severe foliar spot also produces brown. The `value < 150` constraint distinguishes brown (dark) from yellow (bright).

**G3.3 `necrotic_pixel_fraction` ∈ [0, 1]**
```
necrotic_mask = (hsv_leaf[:, 2] < 50) & (hsv_leaf[:, 1] < 60)
necrotic_pixel_fraction = necrotic_mask.sum() / leaf_mask.sum()
```
Fraction of very-dark, low-saturation pixels — dead tissue. Late blight in advanced stages produces these; healthy leaves have ~0%.

**G3.4 `leaf_color_variance` (variance of L channel in LAB; in [0, 6500])**
```
lab_leaf = lab[leaf_mask]  # LAB; same color space as Section 7.2 CLAHE prep
leaf_color_variance = float(np.var(lab_leaf[:, 0]))  # variance of L channel
```
How variable is the leaf brightness. Healthy leaves are uniform (low variance); mosaic-virus leaves have patchy bright/dark mottling (high variance); late blight has high variance from necrotic-vs-healthy contrast. The 6500 cap is the maximum L variance for an 8-bit L channel.

#### 10.5.4 Group G4 — Spatial pattern features (3)

Where the disease is on the leaf, not just how much.

**G4.1 `yellow_marginality_ratio` ∈ [0, 1]**
```python
# yellow_mask is reused from G3.1 yellow_pixel_fraction (Section 10.5.3)
def yellow_marginality_ratio(yellow_mask, leaf_mask):
    # "Margin" = leaf pixels within 15% of the leaf's longer dimension from any leaf-boundary edge
    x, y, bbox_w, bbox_h = cv2.boundingRect(leaf_mask.astype(np.uint8))  # (x, y, w, h)
    margin_dist_threshold = 0.15 * max(bbox_w, bbox_h)
    
    # Distance transform: for each leaf pixel, distance to the nearest non-leaf pixel
    leaf_uint8 = leaf_mask.astype(np.uint8)
    dist = cv2.distanceTransform(leaf_uint8, cv2.DIST_L2, 3)
    margin_mask = (dist > 0) & (dist < margin_dist_threshold)
    
    yellow_in_margin = (yellow_mask & margin_mask).sum()
    yellow_total = yellow_mask.sum()
    return float(yellow_in_margin / max(yellow_total, 1))
```
Of the yellow pixels, what fraction sits near the leaf margin. YLCV starts at the leaf margins and spreads inward (high marginality). Light healthy yellowing or mosaic yellowing is more uniformly distributed (lower marginality). Strong YLCV signature.

**G4.2 `disease_centroid_offset` ∈ [0, 1]**
```
if disease_area_px == 0: disease_centroid_offset = 0
else:
  disease_centroid = mean(disease_pixel_coordinates)
  leaf_centroid = mean(leaf_pixel_coordinates)
  offset = euclidean_distance(disease_centroid, leaf_centroid)
  leaf_radius = sqrt(leaf_area_px / π)  # equivalent radius for normalization
  disease_centroid_offset = clip(offset / leaf_radius, 0, 1)
```
Distance between the centroid of disease pixels and the centroid of the leaf, normalized by the leaf's equivalent radius. A value near 0 means disease is centered on the leaf; near 1 means disease is concentrated at one side. Late blight often starts at one side of a leaf (high offset); mosaic affects whole leaf (low offset).

**G4.3 `disease_spatial_dispersion` ∈ [0, 1]**
```
if n_lesions <= 1: disease_spatial_dispersion = 0
else:
  centroids = lesion_stats["centroids"][1:]  # skip background
  pairwise_distances = pdist(centroids)
  mean_dist = mean(pairwise_distances)
  leaf_diagonal = sqrt(bbox_w² + bbox_h²)
  disease_spatial_dispersion = clip(mean_dist / leaf_diagonal, 0, 1)
```
How spread-out lesions are across the leaf. Septoria lesions are scattered (high dispersion); late blight lesions cluster in patches (low dispersion). Mosaic, with 0 or 1 "lesions," gives 0 by definition.

#### 10.5.5 Group G5 — Texture features (3)

Local pattern statistics, computed on the L channel of LAB inside the leaf mask.

**G5.1 `GLCM_contrast` ∈ [0, 1] after normalization**
```
gray_leaf = lab[:, :, 0].copy()
gray_leaf[~leaf_mask] = 0  # zero out non-leaf
glcm = graycomatrix(
    gray_leaf, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
    levels=32, symmetric=True, normed=True,
)  # GLCM at 4 orientations, average
contrast_raw = graycoprops(glcm, "contrast").mean()  # raw value
GLCM_contrast = clip(contrast_raw / 100.0, 0, 1)  # normalize: 100 saturates
```
Higher contrast = stronger texture. Healthy leaf is smooth (low contrast); diseased leaves have textural variation from lesions (higher). Late blight necrosis vs healthy gives the highest contrast.

**Known bias.** Setting `gray_leaf[~leaf_mask] = 0` introduces a "background" intensity level (0) that participates in the GLCM. Background-to-background pairs and leaf-edge-to-background pairs are both counted, biasing GLCM_contrast slightly downward (background regions are uniform, lowering average contrast). The bias is consistent across images, so the classifier learns to compensate. F.0 may evaluate whether masking strategies (e.g., computing GLCM only on the leaf bounding box, or using a NaN-aware GLCM library) reduce the bias enough to matter.

**G5.2 `GLCM_homogeneity` ∈ [0, 1]**
```
homogeneity_raw = graycoprops(glcm, "homogeneity").mean()
GLCM_homogeneity = homogeneity_raw  # already in [0, 1]
```
Higher homogeneity = smoother, more uniform texture. Inverse of contrast in spirit. Healthy and YLCV (which is uniform yellowing) score high; foliar spot and late blight score lower.

**G5.3 `high_freq_energy_ratio` ∈ [0, 1]**
```
gray = lab[:, :, 0]
gray_leaf_only = np.where(leaf_mask, gray, 0)
fft_mag = np.abs(np.fft.fft2(gray_leaf_only))
total_energy = (fft_mag ** 2).sum()
# High frequency = outside the central low-frequency region
H, W = gray.shape
center_y, center_x = H // 2, W // 2
y_grid, x_grid = np.ogrid[:H, :W]
dist_from_center = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
low_freq_mask = (dist_from_center < min(H, W) * 0.1)  # central 10% radius
fft_mag_shifted = np.fft.fftshift(fft_mag)
high_freq_energy = (fft_mag_shifted ** 2)[~low_freq_mask].sum()
high_freq_energy_ratio = clip(high_freq_energy / max(total_energy, 1), 0, 1)
```
Ratio of high-frequency Fourier energy to total. Captures fine texture (lesion edges, vein patterns) vs smooth gradients. Septoria's many sharp small lesions produce high values; late blight's smooth necrotic patches produce lower values; healthy leaves have moderate (vein structure).

#### 10.5.6 Group G6 — Leaf geometry features (2)

Shape of the leaf itself.

**G6.1 `leaf_compactness` ∈ [0, 1]**
```
contours, _ = cv2.findContours(leaf_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
leaf_perimeter = cv2.arcLength(contours[0], closed=True)
leaf_compactness = clip(4 * π * leaf_area_px / max(leaf_perimeter ** 2, 1), 0, 1)
```
Same isoperimetric formula as lesion circularity but applied to the whole leaf. A normal flat tomato leaf is ~0.6-0.8. Curled leaves have lower compactness because curl increases the perimeter without increasing area in 2D projection. YLCV and TMV both cause leaf curl (lower compactness).

**G6.2 `leaf_aspect_ratio` ∈ [0.1, 10]** (clipped)
```
contour = contours[0]
rect = cv2.minAreaRect(contour)  # (center, (w, h), angle)
w, h = rect[1]
leaf_aspect_ratio = clip(max(w, h) / max(min(w, h), 1), 0.1, 10)
```
Long axis divided by short axis of the leaf's minimum-area rotated rectangle. Normal tomato leaf aspect is ~1.5-2.5. Severely curled leaves can deviate. Used as a sanity check more than a diagnostic.

#### 10.5.7 Group G7 — Quality and reliability features (3)

These are not strictly disease features but contextualize the others.

**G7.1 `sharpness` ∈ [0, 1]**
```
Same as IQA's sharpness (Section 6.2.1), recomputed on the color-constancy-applied image.
```
Recomputed because the color-constancy transform may change the image; PSV's interpretation of features depends on PSV's input, not IQA's. The classifier sees PSV's recomputed sharpness at feature index 19, NOT IQA's sharpness. IQA's sharpness affects only the IQA decision (Section 6.4); it does not appear in the 19-dim classifier feature vector. The two values are usually close but can differ by ~5-10% on heavily color-shifted images.

**G7.2 `aggregate_quality` ∈ [0, 1]**
```
aggregate_quality = IQA's aggregate_score, passed through.
```
The IQA aggregate score (Section 6.3) is included in the PSV feature vector so the classifier sees it. PSV doesn't recompute it — it just propagates IQA's output. The PSV feature name is `aggregate_quality` (because in PSV's space, this is a quality input); in the IQAResult dataclass (Section 6.5) the same value is called `aggregate_score`. Same value, two names by section context.

**G7.3 `psv_aggregate_reliability` ∈ [0, 1]**
```
See Section 10.7 for the full computation. This is PSV's self-assessment of its own
trustworthiness on this image. Distinct from IQA's aggregate_score (which measures
input image quality, not PSV's confidence in its own outputs).
```

#### 10.5.8 Group G8 — Vegetation indices (4)

Standard remote-sensing indices that are sensitive to chlorophyll and leaf health. Computed from the sRGB values of leaf pixels only (not the whole image).

**A note on linearization.** Strictly speaking, vegetation indices are defined for linear-radiance values, not sRGB-encoded values. Linearizing sRGB → linear-RGB before computing the indices would be more textbook-correct. Standard practice in plant-health remote sensing for camera images skips this step because the indices are robust to small gamma effects and linearization adds compute. The PSV implementation uses sRGB directly. F.0 may evaluate whether linearization improves discrimination on Kerala photos; if so, a linearization step can be inserted before index computation without changing the index formulas.

For a leaf pixel with sRGB values `r, g, b` (each in [0, 255]):

**G8.1 `ExG` (Excess Green; standardized to [-1, 1])**
```
ExG_per_pixel = 2*g - r - b  # in [-510, 510]
ExG = clip(mean(ExG_per_pixel for leaf pixels) / 255, -1, 1)
```
High when green dominates (healthy chlorophyll). Drops with chlorosis or browning.

**G8.2 `GLI` (Green Leaf Index; in [-1, 1])**
```
GLI_per_pixel = (2*g - r - b) / max(2*g + r + b, 1)
GLI = mean(GLI_per_pixel for leaf pixels)
```
Normalized version of ExG; less sensitive to overall brightness. Same interpretation.

**G8.3 `MGRVI` (Modified Green-Red Vegetation Index; in [-1, 1])**
```
MGRVI_per_pixel = (g**2 - r**2) / max(g**2 + r**2, 1)
MGRVI = mean(MGRVI_per_pixel for leaf pixels)
```
Sensitive to red-green balance. Healthy leaf has g >> r → MGRVI ≈ 1. YLCV (yellow leaves) has g ≈ r → MGRVI ≈ 0. Late blight (brown leaves with r > g) → negative.

**G8.4 `VARI` (Visible Atmospherically Resistant Index; in [-1, 1])**
```
VARI_per_pixel = (g - r) / max(g + r - b, 1)
VARI = clip(mean(VARI_per_pixel for leaf pixels), -1, 1)
```
Robust to atmospheric/lighting differences. Used here as a secondary chlorophyll proxy.

All four indices are positively correlated for healthy leaves and decrease together for chlorosis and necrosis. They are kept separately because their failure modes differ — ExG is sensitive to brightness, MGRVI is not; this redundancy lets the classifier triangulate.

#### 10.5.9 Feature catalog summary

The 26 features, in the order the feature vector is laid out:

| Idx | Name | Group |
|---|---|---|
| 0 | disease_coverage_pct | G1 |
| 1 | largest_lesion_pct | G1 |
| 2 | lesion_count | G1 |
| 3 | mean_lesion_size | G2 |
| 4 | lesion_size_std | G2 |
| 5 | mean_lesion_circularity | G2 |
| 6 | edge_sharpness | G2 |
| 7 | yellow_pixel_fraction | G3 |
| 8 | brown_pixel_fraction | G3 |
| 9 | necrotic_pixel_fraction | G3 |
| 10 | leaf_color_variance | G3 |
| 11 | yellow_marginality_ratio | G4 |
| 12 | disease_centroid_offset | G4 |
| 13 | disease_spatial_dispersion | G4 |
| 14 | GLCM_contrast | G5 |
| 15 | GLCM_homogeneity | G5 |
| 16 | high_freq_energy_ratio | G5 |
| 17 | leaf_compactness | G6 |
| 18 | leaf_aspect_ratio | G6 |
| 19 | sharpness | G7 |
| 20 | aggregate_quality | G7 |
| 21 | psv_aggregate_reliability | G7 |
| 22 | ExG | G8 |
| 23 | GLI | G8 |
| 24 | MGRVI | G8 |
| 25 | VARI | G8 |

This ordering is fixed. F.0 standardization (computing per-feature mean and std for normalization before classifier input) uses this order, so changing it would invalidate the standardization parameters.

### 10.6 Stage 4 — Compatibility scoring

Goal: convert the 26 features into 6 scores, one per tomato class, indicating how compatible the features are with each disease's botanical signature.

The 6 scores are produced in **canonical order** (Section 3.7):
- `c_foliar` (index 0)
- `c_septoria` (index 1)
- `c_late_blight` (index 2)
- `c_ylcv` (index 3)
- `c_mosaic` (index 4)
- `c_healthy` (index 5)

#### 10.6.1 The fixed weight matrix

Compatibility scoring uses a hand-engineered matrix `W` of shape [6, 26]: 6 disease classes × 26 features. Each row encodes which features should weight positively (signal of the disease) and negatively (counter-signal). Entries are in approximately [-1, +1].

The matrix is hand-engineered from plant pathology priors and is NOT learned from data (Section 3.7 rationale). The values below are placeholder and F.0 may tune them within plausibility ranges informed by the agronomist audit.

**Storage and load-time validation.** WEIGHT_MATRIX is stored in `tomato_sandbox/config/psv_weights.yaml` (Section 10.10), a human-readable file the agronomist can audit and tune during F.0. The YAML enumerates each row by feature name (not by index) so that re-ordering features in the catalog (Section 10.5.9) doesn't silently corrupt the matrix. The loader at startup:
1. Reads the YAML.
2. Asserts that the set of feature names in the YAML exactly matches the 26 names in Section 10.5.9.
3. Constructs the 6×26 numpy matrix in the canonical row order matching the feature catalog.
4. Refuses to start if the names don't match (Section 4.4 startup failure handling).

| Feature | foliar | septoria | late_blight | YLCV | mosaic | healthy |
|---|---|---|---|---|---|---|
| disease_coverage_pct | +0.6 | +0.5 | +0.7 | +0.3 | +0.5 | -0.9 |
| largest_lesion_pct | +0.6 | -0.3 | +0.9 | 0.0 | +0.2 | -0.5 |
| lesion_count | +0.4 | +0.9 | -0.2 | -0.6 | -0.4 | -0.5 |
| mean_lesion_size | +0.5 | -0.6 | +0.9 | -0.5 | +0.3 | -0.4 |
| lesion_size_std | +0.3 | -0.5 | +0.7 | -0.5 | +0.4 | -0.4 |
| mean_lesion_circularity | +0.3 | +0.8 | -0.5 | 0.0 | -0.2 | 0.0 |
| edge_sharpness | +0.4 | +0.8 | -0.7 | -0.2 | -0.1 | 0.0 |
| yellow_pixel_fraction | +0.2 | +0.1 | -0.1 | +0.9 | +0.5 | -0.3 |
| brown_pixel_fraction | +0.5 | +0.4 | +0.8 | -0.2 | -0.1 | -0.7 |
| necrotic_pixel_fraction | +0.3 | +0.2 | +0.9 | -0.5 | -0.3 | -0.9 |
| leaf_color_variance | +0.4 | +0.4 | +0.5 | +0.1 | +0.7 | -0.5 |
| yellow_marginality_ratio | -0.1 | -0.1 | -0.2 | +0.9 | -0.3 | -0.2 |
| disease_centroid_offset | +0.2 | -0.3 | +0.5 | -0.2 | -0.5 | 0.0 |
| disease_spatial_dispersion | +0.3 | +0.7 | -0.3 | -0.4 | +0.2 | 0.0 |
| GLCM_contrast | +0.4 | +0.5 | +0.5 | -0.2 | +0.3 | -0.4 |
| GLCM_homogeneity | -0.4 | -0.3 | -0.4 | +0.4 | -0.4 | +0.6 |
| high_freq_energy_ratio | +0.4 | +0.7 | -0.2 | -0.3 | +0.2 | -0.2 |
| leaf_compactness | 0.0 | 0.0 | -0.1 | -0.4 | -0.2 | +0.4 |
| leaf_aspect_ratio | 0.0 | 0.0 | 0.0 | -0.3 | 0.0 | +0.1 |
| sharpness | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aggregate_quality | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| psv_aggregate_reliability | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ExG | -0.2 | -0.1 | -0.4 | -0.3 | -0.2 | +0.7 |
| GLI | -0.2 | -0.1 | -0.4 | -0.3 | -0.2 | +0.7 |
| MGRVI | -0.2 | -0.1 | -0.4 | -0.5 | -0.2 | +0.7 |
| VARI | -0.2 | -0.1 | -0.3 | -0.3 | -0.2 | +0.6 |

**Reading the matrix:**
- Each column shows which features push toward that disease.
- High `lesion_count` weights septoria (+0.9) and against YLCV (-0.6) because septoria has many lesions and YLCV has none.
- High `yellow_marginality_ratio` weights YLCV strongly (+0.9) — its botanical signature.
- High `mean_lesion_size` weights late blight (+0.9) and against septoria (-0.6).
- The G7 features (sharpness, aggregate_quality, psv_aggregate_reliability) have zero weights in compatibility. They affect reliability (Section 10.7), not class identity.
- Vegetation indices weight healthy positively (chlorophyll signal) and all diseases negatively.

#### 10.6.2 Standardization and scoring

Raw feature values have very different scales (e.g., `lesion_count` in the hundreds; `GLI` in [-1, 1]). Direct multiplication by the weight matrix would overweight the high-magnitude features. Standardization fixes this:

```python
def compute_compatibility_scores(features: np.ndarray) -> np.ndarray:
    """
    features: [26] raw feature values
    Returns: [6] compatibility scores in canonical order, after softmax
    """
    # Standardize each feature: subtract per-feature mean, divide by per-feature std
    # The mean and std come from F.0 calibration on the training subset.
    standardized = (features - F0_FEATURE_MEAN) / (F0_FEATURE_STD + 1e-6)
    standardized = np.clip(standardized, -3, 3)  # cap extreme values
    
    # Apply weight matrix
    raw_scores = WEIGHT_MATRIX @ standardized  # shape [6]
    
    # Softmax to get probability-like compatibility scores
    # Temperature T_psv tunes how sharp the distribution is.
    logits = raw_scores / T_PSV
    exp = np.exp(logits - logits.max())  # numerical stability
    compatibility = exp / exp.sum()  # shape [6]
    return compatibility
```

**Why standardize:** raw `lesion_count` is in [0, 200] and raw `MGRVI` is in [-1, 1]. Without standardization, `lesion_count` would dominate the score regardless of its weight. Standardization puts all features on a unit-scale, so weights compare apples to apples.

**Why softmax with temperature:** the raw scores from the weight matrix sum to roughly 0 across classes (because rows sum to roughly 0). Softmax converts them to probability-like values that the downstream classifier can interpret. The temperature `T_PSV` controls how peaked the output is.

**Pinned constants** (placeholders; F.0 calibrates):
- `F0_FEATURE_MEAN`: shape [26] — per-feature mean computed on the training subset. Loaded at startup from `tomato_sandbox/phase_f0_calibration/psv_standardization.json`. Until F.0 produces this file (Phase A), fallback values from a uniform distribution over each feature's expected range apply; the sandbox refuses to start in production mode if the F.0 file is missing.
- `F0_FEATURE_STD`: shape [26] — per-feature standard deviation. Same source.
- `T_PSV = 1.0` — softmax temperature. F.0 sweeps [0.5, 0.8, 1.0, 1.5, 2.0] and selects the value minimizing classifier validation NLL. Stored in `phase_f0_calibration/psv_standardization.json` alongside the mean/std.
- `WEIGHT_MATRIX`: shape [6, 26] — the table above, frozen after agronomist audit. Stored in `tomato_sandbox/config/psv_weights.yaml`.

The standardization parameters are stored at `tomato_sandbox/phase_f0_calibration/psv_standardization.json` and loaded at startup. The weight matrix is stored at `tomato_sandbox/config/psv_weights.yaml` and is human-readable so the agronomist can audit and modify it during Phase F.0.

**F.0 covers all 26 features.** F.0 computes `F0_FEATURE_MEAN[i]` and `F0_FEATURE_STD[i]` for all 26 features, including features that are propagated from IQA (G7.2 `aggregate_quality`) or computed by PSV's Stage 5 (G7.3 `psv_aggregate_reliability`). These features have zero weight in the compatibility matrix but their standardization parameters are still computed because they appear in `SignalCResult.standardized_features`, which feeds into the classifier (Section 12) where their weights are learned (not zero like in PSV's compatibility matrix). F.0 measures these features by running the full PSV pipeline on the training subset and recording the distribution of each feature's output.

#### 10.6.3 Known limitations of the fixed matrix

Honest acknowledgments:
1. **The matrix is opinion, not data.** With 200 training samples and 26 features, learning the matrix from data would overfit. The hand-engineered values are based on plant-pathology literature and the agronomist's input, but they may have systematic biases.
2. **Some features have zero weights for some classes** (e.g., `disease_centroid_offset = 0.0` for healthy and septoria). This is a soft prior that those features don't help distinguish those specific classes, not a strong claim that they're irrelevant.
3. **The matrix is not symmetric in error costs.** Misclassifying late_blight as healthy is much worse than misclassifying mosaic as foliar. The matrix doesn't encode this; the tier system (Section 14) does, by elevating Tier 5 alerts for late_blight, mosaic, YLCV regardless of confidence.
4. **Per-class baseline bias risk.** The placeholder weights are not zero-summed per row — different rows have different sums. After standardization (zero-mean features), a "typical" image scores zero for all classes, but a mildly-deviant image disproportionately favors the row with the larger sum. F.0 must check whether per-class baseline bias exists by running PSV on the training subset and comparing per-class score distributions on healthy reference images. If bias is detected, F.0 either (a) zero-centers each row of the matrix, or (b) adds a per-class bias vector that the classifier (Section 12) can learn to compensate for.
5. **F.0 may tune the matrix, but only within agronomist-approved ranges.** The classifier (Section 12) provides an additional learned layer of correction over PSV's compatibility scores, so even if PSV is biased, the system as a whole can compensate.
6. **High T_PSV indicates limited PSV value.** If F.0 finds T_PSV > 2.0 is optimal, PSV's compatibility scores become nearly uniform across classes, indicating PSV's discriminative value is limited on the calibration data. The classifier compensates by weighting PSV less. This is acceptable but worth flagging during F.0 review as a signal that the weight matrix may need agronomist revision.

### 10.7 Stage 5 — PSV reliability assessment

Goal: produce `psv_aggregate_reliability ∈ [0, 1]`, an estimate of how trustworthy PSV's compatibility scores are for this specific image.

```python
def compute_psv_reliability(
    leaf_mask: np.ndarray,
    disease_mask: np.ndarray,
    iqa_green_mask: np.ndarray,
    iqa_aggregate_score: float,
    n_lesions: int,
) -> float:
    """
    Combines several reliability signals into a single score in [0, 1].
    """
    # Component 1: leaf-mask sanity check vs IQA's rough green mask.
    # Compute Jaccard (IoU) between PSV's careful mask and IQA's rough mask.
    # High agreement → both methods see the leaf in the same place → reliable.
    if leaf_mask.sum() == 0:
        return 0.0  # no leaf detected — PSV cannot work
    if iqa_green_mask is None or iqa_green_mask.sum() == 0:
        mask_agreement = 0.5  # no IQA mask to compare; neutral score
    else:
        # Resize IQA mask if shapes differ (PSV uses post-resize-cap shape)
        if iqa_green_mask.shape != leaf_mask.shape:
            iqa_green_mask = cv2.resize(
                iqa_green_mask.astype(np.uint8),
                (leaf_mask.shape[1], leaf_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        intersection = (leaf_mask & iqa_green_mask).sum()
        union = (leaf_mask | iqa_green_mask).sum()
        iou = intersection / max(union, 1)
        # Map IoU to a reliability factor: IoU > 0.7 → 1.0; IoU < 0.3 → 0.0; linear in between.
        mask_agreement = clip((iou - 0.3) / 0.4, 0, 1)
    
    # Component 2: lesion sanity. Disease coverage > 90% means everything looks
    # diseased (likely background contamination or wrong segmentation).
    leaf_area = leaf_mask.sum()
    if leaf_area == 0:
        coverage_sanity = 0.0
    else:
        coverage = disease_mask.sum() / leaf_area
        if coverage > 0.90:
            coverage_sanity = 0.2  # likely segmentation failure
        elif coverage > 0.70:
            coverage_sanity = 0.6  # very heavy disease, possible but rare
        else:
            coverage_sanity = 1.0
    
    # Component 3: IQA quality. PSV reliability cannot exceed IQA's quality;
    # if IQA says image is poor, PSV's measurements are also poor.
    iqa_factor = iqa_aggregate_score
    
    # Combine via geometric mean (any one being low pulls the result down)
    components = [mask_agreement, coverage_sanity, iqa_factor]
    components = [max(c, 0.05) for c in components]  # floor to avoid log(0)
    reliability = np.exp(np.mean(np.log(components)))
    return float(reliability)
```

**Three reliability components combined:**
1. **mask_agreement (IoU between PSV and IQA masks):** if PSV's careful segmentation diverges from IQA's rough one, something's off — possibly a non-leaf object PSV mistook for the leaf.
2. **coverage_sanity:** sky-high disease coverage usually means PSV segmented background as "leaf" and is treating texture as disease. Real diseased leaves rarely exceed 70% coverage.
3. **iqa_factor:** PSV cannot be more reliable than the input image quality.

**Geometric mean combination:** any single component near 0 should pull reliability near 0 (cascading failure), which arithmetic mean would not do. The 0.05 floor prevents log(0) and avoids reliability collapsing entirely on a single edge case.

The reliability is consumed by:
- The classifier (Section 12) as one of the 19 input features.
- The tier system (Section 14): if `psv_aggregate_reliability < 0.4`, Tier 3C "PSV broken or chilli leakage" can fire (combined with chilli_leakage threshold).

### 10.8 Stage 5 fallback — when PSV's segmentation fails entirely

If PSV's careful segmentation produces an empty leaf_mask (no pixels survived all the morphology and largest-component selection), PSV does NOT throw. It enters fallback mode:
1. Use the IQA green_mask as the leaf_mask.
2. Recompute disease detection on the fallback leaf_mask.
3. Compute features and compatibility.
4. Set `psv_aggregate_reliability = max(0.1, 0.3 * iqa_aggregate_score)` (low but nonzero).

**On the fallback constants 0.1 and 0.3.** These are design-time constants chosen to ensure fallback reliability is always low (≤ 0.3 × iqa_aggregate_score) but always nonzero (floor 0.1). They are NOT F.0-calibrated; they encode a fixed engineering policy that fallback mode is inherently less trustworthy than careful segmentation. F.0 may revisit these if measurements show different values are more useful, but the policy "fallback is always low-confidence" should be preserved.

If the IQA green_mask is also empty (the image truly has no leaf-like content), PSV returns an all-zero feature vector, all-uniform compatibility scores, and `psv_aggregate_reliability = 0.05`. The classifier sees PSV's contribution as essentially noise; v3 and LoRA dominate the prediction.

### 10.9 Output structure

```python
@dataclass
class SignalCResult:
    compatibility: np.ndarray           # [6], canonical order: foliar, septoria, late_blight, ylcv, mosaic, healthy
    compatibility_argmax: int           # 0-5 in canonical order
    compatibility_max: float            # max of compatibility
    compatibility_margin: float         # max - second-max (for classifier feature)
    psv_reliability: float              # in [0, 1]
    raw_features: np.ndarray            # [26] raw values, for monitoring/debug
    standardized_features: np.ndarray   # [26] post-standardization, capped at ±3
    leaf_mask: np.ndarray               # [H, W] bool — for response builder if requested
    disease_mask: np.ndarray            # [H, W] bool — for response builder if requested
    n_lesions: int                      # for diagnostics
    fallback_used: bool                 # True if Stage 5 fallback fired
    forward_succeeded: bool             # True for any non-exceptional path; False only on hard exception
    failure_reason: str | None
```

The leaf and disease masks are kept for the response builder (Section 16), which may include them in `tomato_details` for UI overlays. They are NOT used by the classifier — only the scalar features are.

**Cross-signal features computed by the classifier prep step.** Section 8.4 lists "4 PSV summary features" in the 19-dim classifier input: top-1 score, agreement-with-v3, agreement-with-LoRA, margin. Two of these (`compatibility_max` = top-1, `compatibility_margin` = margin) come directly from SignalCResult. The other two (`agree_v3` and `agree_lora`) are NOT computed by PSV. They are computed by the classifier prep step (Section 12) using PSV's `compatibility_argmax` and the corresponding argmaxes from v3 and LoRA:
```python
agree_v3 = float(psv.compatibility_argmax == signal_a.tomato_argmax_canonical)
agree_lora = float(psv.compatibility_argmax == signal_b.tomato_argmax_canonical)
```
This separation keeps PSV self-contained (no cross-signal dependencies) while still providing the classifier with the agreement signal it needs.

### 10.10 Where this lives in the sandbox

PSV is split into multiple files for clarity:
- `tomato_sandbox/signals/psv/psv.py` — orchestrator, defines `compute_signal_c(rgb_cc, iqa_green_mask, iqa_aggregate_score) -> SignalCResult`
- `tomato_sandbox/signals/psv/leaf_segmentation.py` — Stage 1 segmentation
- `tomato_sandbox/signals/psv/disease_detection.py` — Stage 2 disease region detection
- `tomato_sandbox/signals/psv/features.py` — Stage 3 all 26 features
- `tomato_sandbox/signals/psv/compatibility.py` — Stage 4 weight matrix and softmax scoring
- `tomato_sandbox/signals/psv/reliability.py` — Stage 5 reliability and fallback handling

Calibration files:
- `tomato_sandbox/phase_f0_calibration/psv_standardization.json` — F0_FEATURE_MEAN, F0_FEATURE_STD per feature
- `tomato_sandbox/config/psv_weights.yaml` — WEIGHT_MATRIX, human-readable

The orchestrator `compute_signal_c` is called by the TomatoPipeline orchestrator (Section 21) once per image. It runs entirely on CPU and does not touch the GPU.

### 10.11 Performance budget

PSV runs on CPU. For a 1200×900 input (worst case after PSV's resize cap):
- Stage 1 segmentation (HSV, Otsu, morph, CC): ~30 ms
- Stage 2 disease detection (deviation, threshold, morph, CC): ~20 ms
- Stage 3 features (all 26):
  - G1, G2 (coverage and shape, all from CC stats): ~10 ms
  - G3 color stats (HSV mask sums): ~5 ms
  - G4 spatial (centroids, distances): ~5 ms
  - G5 texture (GLCM at 4 angles + FFT): ~80 ms — this is the slow one
  - G6 geometry (contours, min-area rect): ~5 ms
  - G7 quality (mostly propagation): ~1 ms
  - G8 vegetation indices (per-pixel arithmetic on leaf): ~10 ms
- Stage 4 compatibility scoring: <1 ms
- Stage 5 reliability: ~5 ms

Total: ~170-200 ms median. The 200 ms estimate in Section 4.6 is consistent with this.

GLCM computation is the bottleneck. If F.0 measurements show GLCM dominating, options:
- Reduce GLCM levels from 32 to 16 (halves time, slight feature degradation)
- Compute GLCM at 1 angle instead of 4 (quartered time, slightly less rotational invariance)
- Skip GLCM entirely under tight latency budgets (drops 2 features)

These are tuning options, not currently invoked.

### 10.12 What PSV is NOT

To prevent misuse, here is what PSV does not do:
- **It does not classify diseases.** It produces compatibility scores; the classifier (Section 12) does the classification.
- **It does not segment by disease type.** The `disease_mask` is binary (disease vs healthy), not per-disease.
- **It does not handle multi-leaf images.** Only the largest connected component is analyzed. IQA's `background_contamination` should catch multi-leaf cases earlier.
- **It does not adapt to lighting.** Color constancy (Section 7.4) is the only lighting normalization; PSV trusts that the input has been color-corrected.
- **It does not learn over time.** The weight matrix is fixed. F.0 may tune within plausibility ranges, but no online learning.
- **It does not use TTA.** PSV runs once per image. Augmenting the image (flip, rotate, color jitter) would distort PSV's color and spatial features. Section 11 explicitly excludes PSV from TTA.

---

## Section 11. Test-Time Augmentation (TTA)

### 11.1 Purpose

TTA is a controlled re-run of v3 and LoRA forward passes on augmented views of the same input image, to reduce uncertainty when the initial pass produced low-confidence output. It does not run unconditionally (cost is too high); it fires selectively based on the initial classifier's max probability.

PSV does NOT participate in TTA. PSV's features depend on color statistics, segmentation, and spatial layout that augmentations would distort. Only v3 and LoRA — which are pretrained on augmentation-rich datasets and benefit from view diversity — are re-run.

### 11.2 When TTA fires

The orchestrator runs the full pipeline once with no augmentation (1-view). It computes the classifier's combined output (Section 12) and reads `combined_max_prob`.

```
combined_max_prob >= TOMATO_TTA_TRIGGER_THRESHOLD (default 0.55)
    → no TTA. Pipeline returns 1-view result.

TOMATO_TTA_ESCALATE_THRESHOLD <= combined_max_prob < TOMATO_TTA_TRIGGER_THRESHOLD
    → 2-view TTA. Re-run v3 and LoRA on 1 augmented view (plus the original makes 2).

combined_max_prob < TOMATO_TTA_ESCALATE_THRESHOLD (default 0.45)
    → 5-view TTA. Re-run v3 and LoRA on 4 augmented views (plus the original makes 5).
```

These thresholds are placeholders; F.0 calibrates them per Section 4.5.

After TTA, the per-signal outputs are aggregated (mean of softmax) and the classifier is re-run with the aggregated outputs. The new classifier output is the final pipeline result.

**NaN combined_max_prob.** If the 1-view classifier itself produces a non-numeric result (e.g., both v3 and LoRA failed and PSV returned uniform compatibility, leading to numerical issues in the classifier), `combined_max_prob` may be NaN. The TTA decision treats NaN as "do not run TTA":
```python
if not np.isfinite(combined_max_prob):
    n_views = 1   # no TTA
    # Pipeline proceeds to tier assignment with the failed signals; Tier 4B is likely.
```
This avoids attempting TTA on a fundamentally broken pipeline state. Tier assignment (Section 14) then routes the request to a degraded-mode tier based on which signals failed.

**Multi-image requests.** For multi-image input (Section 18), each image independently decides whether to fire TTA based on its own per-image `combined_max_prob`. The multi-image controller does NOT aggregate the TTA decisions across images. Each image runs its own pipeline with its own TTA path. This means worst case (3 images all triggering 5-view TTA) the request consumes ≈3 seconds of cumulative GPU model-forward time across all images (3 × 1000 ms). Full request time including preprocessing, PSV, classifier, conformal, tier, response, and tail latency is bounded by Section 4.6 P95 (~7.5 s for 3-image-5-view-TTA).

**Why on combined max prob, not per-signal:** the goal of TTA is to reduce uncertainty in the FINAL prediction. If v3 is uncertain but LoRA is very confident, the combined output may still be confident — no need for TTA. Triggering on the combined output uses the right uncertainty signal.

**Why escalate to 5-view rather than always 2-view:** 2-view often resolves moderate uncertainty cheaply (~150 ms extra). For very-low-confidence cases (combined max prob < 0.45), 2-view is unlikely to push the result above any decision threshold, so we go directly to 5-view, which is more likely to either resolve or confirm the uncertainty.

**Why not 3-view or 4-view:** intermediate counts don't add much. The 1, 2, 5 cascade is empirically common in TTA literature and keeps the decision tree simple.

### 11.3 The augmentation set

The augmentations are designed to preserve the diagnostic content of the image while providing view diversity to the models.

For 2-view TTA:
- Augmented view 1: horizontal flip

For 5-view TTA (in addition to the 2-view augmentation):
- Augmented view 2: rotate by +5 degrees
- Augmented view 3: rotate by -5 degrees
- Augmented view 4: brightness +5% (single direction; the original view acts as the no-jitter baseline)

```python
def build_augmentations(n_views: int) -> list:
    """
    Returns a list of PIL.Image transformations to apply.
    The original image is view 0; this returns views 1..n-1.
    """
    augs = []
    if n_views >= 2:
        augs.append(("hflip",))
    if n_views >= 5:
        augs.append(("rotate", +5))
        augs.append(("rotate", -5))
        augs.append(("brightness", 1.05))
    return augs

def apply_augmentation(pil: PIL.Image.Image, aug_spec: tuple) -> PIL.Image.Image:
    if aug_spec[0] == "hflip":
        return pil.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    elif aug_spec[0] == "rotate":
        # fillcolor matches LORA_PAD_VALUE so any rotation padding looks like LoRA's
        # expected pad value (Section 7.3). v3 uses stretch resize so its preprocessing
        # eliminates rotation padding, but the fill is harmless for v3 too.
        return pil.rotate(
            aug_spec[1], expand=False,
            fillcolor=(LORA_PAD_VALUE, LORA_PAD_VALUE, LORA_PAD_VALUE),
        )
    elif aug_spec[0] == "brightness":
        return PIL.ImageEnhance.Brightness(pil).enhance(aug_spec[1])
```

**Preprocessing per augmented view.** Each augmented PIL image is passed through `preprocess_for_v3` and `preprocess_for_lora` (Section 7) before the model forward. Preprocessing is re-run for each view because the augmented pixel content differs. PSV preprocessing is NOT re-run because PSV does not participate in TTA (Section 11.9).

**Why these augmentations and not others:**
- **Horizontal flip:** preserves all diagnostic content; doubles effective data.
- **Small rotations (±5°):** simulate slight phone tilt; do not expose padding artifacts that larger rotations would.
- **Mild brightness jitter (+5%):** simulates lighting variation; does not change disease colors significantly. Single direction (the original view acts as the no-jitter baseline; with both views averaged, the model effectively sees a small range of brightnesses).
- **NOT included:** large rotations (>10°), vertical flip (changes leaf orientation in ways the model wasn't trained for), color jitter (could shift disease-color features), random crops (would lose leaf area — bad for v3 and LoRA which expect full-leaf views).

The fillcolor for rotation matches LORA_PAD_VALUE (114) so that any padding from rotation looks like LoRA's expected pad value.

### 11.4 Per-signal aggregation across views

After running v3 and LoRA on all views, aggregate by mean of softmax:

```python
def aggregate_views(per_view_probs: list[np.ndarray], per_view_ok: list[bool]) -> tuple[np.ndarray, int]:
    """
    per_view_probs: list of [6] softmax outputs from 1, 2, or 5 views (zero-filled for failures)
    per_view_ok:    list of bools, True if that view's forward succeeded
    Returns: (aggregated [6] distribution, n_views_used)
    """
    surviving = [p for p, ok in zip(per_view_probs, per_view_ok) if ok]
    if not surviving:
        # All views failed; return zero-filled distribution (caller treats as forward failure)
        return np.zeros(6, dtype=np.float32), 0
    stacked = np.stack(surviving)  # [n_surviving, 6]
    return stacked.mean(axis=0), len(surviving)
```

This is applied separately for v3 (after canonical remap) and for LoRA. The aggregated outputs replace the 1-view outputs in the classifier's input.

**Failed views are excluded.** If a view's forward pass produced NaN, threw an exception, or otherwise had `forward_succeeded=False`, that view's probability vector is dropped from aggregation. The result is averaged over surviving views only. If ALL views failed for a signal, the aggregated probability is zero-filled and the signal is marked as failed downstream — the classifier sees a degraded-mode signal (Section 12 degraded-mode handling).

**Why mean of softmax (not mean of logits):** mean-of-softmax is the standard ensemble averaging in classification literature. Mean-of-logits is geometric averaging in probability space, which can be over-confident when individual views are confidently wrong. Mean-of-softmax is more conservative.

**Why not weighted by view confidence:** weighting by per-view max probability would over-favor views where the model happened to be confident, which defeats TTA's purpose (averaging out idiosyncrasies of one view). Equal-weight averaging is intentional. This means hflip and small-rotation views contribute equally to the aggregate even though hflip is exactly identity-preserving while rotation introduces some distortion. Empirically, equal weighting works well in TTA literature for image classification.

### 11.5 JSD computation

Jensen-Shannon Divergence between v3's and LoRA's aggregated outputs is one of the 19 classifier input features. JSD captures inter-signal disagreement.

```python
def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    p, q: [6] probability distributions. Both must sum to ≈ 1.
    Returns: JSD in [0, log(2)] ≈ [0, 0.693], using natural log.
    """
    p = np.asarray(p) + 1e-12
    q = np.asarray(q) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)
```

**Range and log base.** JSD computed with natural log (used here) is bounded in [0, log(2)] ≈ [0, 0.693] when both inputs sum to 1. JSD computed with log base 2 would be bounded in [0, 1]. Either convention is fine if used consistently; this implementation uses natural log throughout. The classifier (Section 12) sees the raw JSD value and standardizes it at its input layer alongside the other features, so the choice of log base does not affect downstream behavior.

**Interpretation:** JSD = 0 means v3 and LoRA agree perfectly. JSD near 0.69 means they disagree strongly (e.g., one is confident in class A, the other in class B). The classifier learns to weight signals based on JSD: high JSD usually correlates with low overall reliability.

JSD is computed on the **aggregated** (post-TTA) v3 and LoRA outputs. If TTA didn't fire, it's computed on the 1-view outputs. Either way, it's a single scalar.

**JSD when one or both signals failed.** If either v3 or LoRA has `forward_succeeded=False` (or zero surviving views after TTA aggregation), JSD is set to a sentinel value rather than computed:
- Both signals OK: JSD computed normally.
- Exactly one signal failed: JSD is set to the sentinel `np.nan`. The classifier prep step recognizes NaN and replaces it with a "neutral" value (specifically: the median JSD observed on the F.0 calibration set, stored in `phase_f0_calibration/jsd_sentinel.json`). The classifier also receives a separate `signal_failed_v3` or `signal_failed_lora` flag in degraded-mode handling.
- Both signals failed: JSD is again `np.nan`; the classifier enters full degraded mode (Section 12) and PSV becomes the dominant signal.

This avoids the misleading interpretation where computing JSD between a zero-filled (uniform after epsilon) distribution and a real distribution would falsely indicate "high disagreement" when the truth is "one signal didn't produce output."

### 11.6 Output structure

TTA itself doesn't have a result dataclass; it modifies SignalAResult and SignalBResult by replacing their probability fields with the aggregated versions. It also produces a small TTAReport object for monitoring:

```python
@dataclass
class TTAReport:
    triggered: bool                       # True if TTA fired (2-view or 5-view)
    n_views_attempted: int                # 1, 2, or 5
    n_views_succeeded_v3: int             # how many v3 views succeeded
    n_views_succeeded_lora: int           # how many LoRA views succeeded
    initial_combined_max_prob: float      # the 1-view classifier output that triggered TTA
    final_combined_max_prob: float        # post-aggregation classifier output
    per_view_v3_argmax: list[int]         # argmax per view for v3, in canonical order; np.nan-equivalent (-1) for failed views
    per_view_v3_succeeded: list[bool]     # True if that view's v3 forward succeeded
    per_view_lora_argmax: list[int]       # argmax per view for LoRA, in canonical order; -1 for failed views
    per_view_lora_succeeded: list[bool]   # True if that view's LoRA forward succeeded
    view_disagreement_v3: float           # fraction of SUCCEEDED views where v3 argmax differs from majority
    view_disagreement_lora: float         # fraction of SUCCEEDED views where LoRA argmax differs from majority
```

The `per_view_*_succeeded` fields make failed views visible to monitoring. Without them, a failed view's argmax (which would be 0 by default for a zero-filled probability vector) would silently appear as "view N said class 0," indistinguishable from a successful prediction of class 0.

The `per_view_*_argmax` fields use `-1` to indicate a failed view (since `np.argmax` would return 0 on a zero-filled vector); production monitoring code must check the corresponding `_succeeded` flag before interpreting argmax values.

The disagreement fields are diagnostic. High view-level disagreement (≥ 50% of surviving views differ from the majority) indicates the model is genuinely confused, not just slightly under-threshold; this can feed into Tier 4A "low confidence" assignment. Disagreement is computed only over surviving (non-failed) views.

### 11.7 Where this lives in the sandbox

`tomato_sandbox/tta.py` defines:
- `TTAReport` dataclass
- `should_trigger_tta(combined_max_prob: float) -> int` returning 1, 2, or 5
- `apply_tta(pipeline, validated_image, n_views) -> tuple[SignalAResult, SignalBResult, TTAReport]`

The TTA controller is invoked by the TomatoPipeline orchestrator (Section 21) only after the initial 1-view pass and classifier run.

### 11.8 Performance budget

Per-view GPU time is roughly the same as the 1-view GPU time for v3 + LoRA (~200 ms for the two models combined on a single image; per Sections 8.8 and 9.8). PSV does NOT re-run.

Cumulative GPU time including the initial 1-view pass, model-forward only:
- 1-view (no TTA): 200 ms
- 2-view: 400 ms (2× v3 + 2× LoRA)
- 5-view: 1000 ms (5× v3 + 5× LoRA)

These are model forward times only. Section 4.6's GPU compute table adds:
- Router (~30 ms, runs once not per-view): adds 30 ms to the 1-view path
- Augmentation generation (~50 ms total for 4 augmentations): adds ~50 ms to 5-view path, ~10 ms to 2-view
- Classifier re-run after TTA aggregation (~5 ms each): adds ~5 ms when TTA fires
- Kernel launch and lock acquisition overhead: ~20 ms per view

Combined Section 4.6 numbers:
- 1-view (no TTA): 200 + 30 + 20 = 250 ms (matches Section 4.6 250 ms)
- 2-view: 400 + 30 + 10 + 5 + 40 = 485 ms ≈ 500 ms (matches Section 4.6)
- 5-view: 1000 + 30 + 50 + 5 + 100 = 1185 ms ≈ 1.2 s (matches Section 4.6)

Plus a ~50 ms overhead for augmentation generation and aggregation per TTA invocation. PSV does not contribute to TTA latency.

The TTA decision (whether to fire 2-view or 5-view) takes <1 ms — it's a single threshold comparison.

### 11.9 What TTA does NOT do

To prevent misuse:
- **TTA does not improve calibration.** It reduces variance, not bias. If the model is systematically wrong on a class, TTA gives a more confidently wrong answer.
- **TTA does not fix data-distribution mismatches.** Augmentations are small perturbations; if the input is fundamentally out-of-distribution, TTA won't help.
- **TTA does not run on PSV.** PSV's spatial and color features are not augmentation-invariant.
- **TTA does not re-run preprocessing constants.** CLAHE parameters, ImageNet normalization, padding values stay fixed. Only the input pixel content varies.
- **TTA does not change the classifier.** The classifier sees aggregated v3 and LoRA outputs as if they were 1-view outputs. The classifier was trained on 1-view (training time used standard augmentation, but inference-time TTA was not part of training).

---

## Section 12. Hierarchical classifier

### 12.1 Purpose and architecture

The hierarchical classifier is the system's decision-making core. It takes the outputs of v3 (Section 8), LoRA (Section 9), and PSV (Section 10), assembles them into a 19-dimensional feature vector, and produces a probability distribution over 7 canonical classes:

```
0  foliar
1  septoria
2  late_blight
3  ylcv
4  mosaic
5  healthy
6  OOD (out-of-distribution)
```

This is "canonical with OOD" — the same 6 tomato classes used elsewhere, plus an explicit OOD class for inputs that don't match any tomato disease pattern. The tier assignment (Section 14) and conformal prediction (Section 13) read this 7-class distribution.

The classifier is **hierarchical** for two reasons. First, the decision "is this leaf diseased at all" is conceptually different from "which disease is this." Stage 1 answers the first question (3-way: healthy / diseased / OOD); Stage 2 answers the second only when Stage 1 says diseased (5-way among the 5 disease classes). Second, the soft-routing combination matches the empirical structure of the data — healthy and OOD are confused with each other more than with any single disease, so a flat 7-way classifier would underweight that distinction.

The classifier is **small** — 160 total parameters across both stages — for a third reason: the training set is ≤200 images (the field_val train_subset of 160). A larger classifier would overfit. 160 parameters fitted on 160 images is at the edge of what's defensible, and even this is supported only by careful out-of-fold training (Section 12.9) and cross-validation.

The classifier is also **stacked**: it consumes pre-computed signal outputs rather than raw images. This stacking design lets the classifier focus on combining signals rather than re-learning visual features, which is appropriate given the small dataset.

### 12.2 The 19-dimensional feature vector

The 19-dim vector is constructed from the three signal outputs and a small amount of cross-signal computation. The breakdown was previewed in Section 8.4; this section is the authoritative definition.

**Index | Field | Source | Range | Notes**

| Idx | Field | Source | Range | Notes |
|---|---|---|---|---|
| 0 | v3_p_foliar | SignalAResult.tomato_probs_canonical[0] | [0, 1] | from v3 forward, after canonical remap |
| 1 | v3_p_septoria | SignalAResult.tomato_probs_canonical[1] | [0, 1] | from v3 |
| 2 | v3_p_late_blight | SignalAResult.tomato_probs_canonical[2] | [0, 1] | from v3 |
| 3 | v3_p_ylcv | SignalAResult.tomato_probs_canonical[3] | [0, 1] | from v3 |
| 4 | v3_p_mosaic | SignalAResult.tomato_probs_canonical[4] | [0, 1] | from v3 |
| 5 | v3_p_healthy | SignalAResult.tomato_probs_canonical[5] | [0, 1] | from v3 |
| 6 | lora_p_foliar | SignalBResult.tomato_probs_canonical[0] | [0, 1] | from LoRA, after optional prototype blending |
| 7 | lora_p_septoria | SignalBResult.tomato_probs_canonical[1] | [0, 1] | from LoRA |
| 8 | lora_p_late_blight | SignalBResult.tomato_probs_canonical[2] | [0, 1] | from LoRA |
| 9 | lora_p_ylcv | SignalBResult.tomato_probs_canonical[3] | [0, 1] | from LoRA |
| 10 | lora_p_mosaic | SignalBResult.tomato_probs_canonical[4] | [0, 1] | from LoRA |
| 11 | lora_p_healthy | SignalBResult.tomato_probs_canonical[5] | [0, 1] | from LoRA |
| 12 | psv_top1 | SignalCResult.compatibility_max | [0, 1] | from PSV |
| 13 | psv_margin | SignalCResult.compatibility_margin | [0, 1] | from PSV |
| 14 | agree_v3 | computed | {0.0, 1.0} | 1.0 if PSV.argmax == v3.argmax, else 0.0 |
| 15 | agree_lora | computed | {0.0, 1.0} | 1.0 if PSV.argmax == LoRA.argmax, else 0.0 |
| 16 | jsd_v3_lora | computed | [0, log 2] | JSD between v3 and LoRA distributions (Section 11.5) |
| 17 | psv_reliability | SignalCResult.psv_reliability | [0, 1] | from PSV |
| 18 | chilli_leakage | SignalAResult.chilli_leakage | [0, 1] | from v3 |

**Important: the v3 and LoRA probabilities do NOT sum to 1 across their 6 entries.** v3's probabilities sum to `1 - chilli_leakage` (because the original v3 output is 10-class; the chilli classes are not included in indices 0-5). LoRA's probabilities DO sum to 1 (LoRA is natively 6-class). The classifier sees both forms; it does not re-normalize either, because the un-renormalized v3 distribution carries useful information about misrouting (chilli_leakage is also in the vector at index 18 explicitly, but the un-renormalized 6-vector reinforces the signal).

**Order is fixed.** Changing the order would invalidate the classifier weights and Platt calibration parameters. Production loaders MUST construct the vector in this exact order.

**Standardization.** Each feature is standardized at the classifier's input layer using per-feature mean and std fitted at training time:
```
x_standardized[i] = (x_raw[i] - CLASSIFIER_FEATURE_MEAN[i]) / (CLASSIFIER_FEATURE_STD[i] + 1e-6)
x_standardized[i] = clip(x_standardized[i], -3, 3)
```
The mean and std vectors are stored in `tomato_sandbox/phase_f0_calibration/classifier_feature_standardization.json`. Standardization makes the per-feature scales comparable so logistic regression weights can be interpreted across features. The clip at ±3 prevents extreme outliers from dominating.

**Two-layer standardization clarification.** PSV's 26 features are standardized internally before compatibility scoring (Section 10.6.2's `psv_standardization.json`). The classifier then standardizes the 19-dim vector again at its own input layer, using DIFFERENT mean and std parameters (`classifier_feature_standardization.json`). This is correct and intentional: PSV's standardization is for matrix-multiply scaling within PSV's compatibility computation; the classifier's standardization is for combining heterogeneous-scale inputs (probabilities in [0, 1], divergences in [0, log 2], boolean indicators in {0, 1}, etc.). F.0 produces both standardization files independently.

**Construction code.**
```python
def build_classifier_input(
    sa: SignalAResult,
    sb: SignalBResult,
    sc: SignalCResult,
) -> np.ndarray:
    """Returns a [19] standardized feature vector."""
    raw = np.zeros(19, dtype=np.float32)
    raw[0:6] = sa.tomato_probs_canonical
    raw[6:12] = sb.tomato_probs_canonical
    raw[12] = sc.compatibility_max
    raw[13] = sc.compatibility_margin
    raw[14] = float(sc.compatibility_argmax == sa.tomato_argmax_canonical)
    raw[15] = float(sc.compatibility_argmax == sb.tomato_argmax_canonical)
    raw[16] = jensen_shannon_divergence(
        sa.tomato_probs_canonical, sb.tomato_probs_canonical
    ) if (sa.forward_succeeded and sb.forward_succeeded) else JSD_SENTINEL
    raw[17] = sc.psv_reliability
    raw[18] = sa.chilli_leakage
    
    # Degraded-mode handling: if a signal failed, zero out its block (Section 12.7)
    if not sa.forward_succeeded:
        raw[0:6] = 0.0
        raw[18] = 0.0  # chilli_leakage from v3 also unavailable
    if not sb.forward_succeeded:
        raw[6:12] = 0.0
    if not sc.forward_succeeded:
        raw[12:14] = 0.0
        raw[14] = 0.0
        raw[15] = 0.0
        raw[17] = 0.0
    
    standardized = (raw - CLASSIFIER_FEATURE_MEAN) / (CLASSIFIER_FEATURE_STD + 1e-6)
    return np.clip(standardized, -3, 3)
```

`JSD_SENTINEL` is loaded from `phase_f0_calibration/jsd_sentinel.json` (Section 11.5) — the median JSD observed on the F.0 calibration set, used when JSD cannot be meaningfully computed.

### 12.3 Stage 1 — 3-way diseased/healthy/OOD classifier

Stage 1 takes the 19-dim feature vector and produces a 3-class distribution: `[P(healthy), P(diseased), P(OOD)]` summing to 1.

**Architecture (default):** multinomial logistic regression (also called softmax regression).

```
Input: x ∈ R^19  (standardized)
W_stage1: shape [3, 19]  (learned)
b_stage1: shape [3]      (learned)
logits = W_stage1 @ x + b_stage1  → shape [3]
probs = softmax(logits / T_stage1)  → shape [3]
```

Parameter count: 3 × 19 + 3 = 60.

`T_stage1` is a calibration temperature, fit by Platt scaling on out-of-fold predictions (Section 12.8). Default 1.0 before calibration.

**Why multinomial logistic and not one-vs-rest:** with 160 training samples split across 3 classes, the parameter coupling between classes that multinomial logistic provides (via the softmax) is more data-efficient than one-vs-rest's independent per-class boundaries. The matrix W_stage1 jointly parameterizes all 3 classes; softmax couples them.

**Storage.** Stage 1 weights and bias live in `tomato_sandbox/phase_f0_calibration/classifier_stage1.pkl`. Loaded at startup. The pickle contains:
- `weights`: numpy array [3, 19]
- `bias`: numpy array [3]
- `temperature`: float
- `feature_mean`: [19] for standardization input (also separately in classifier_feature_standardization.json for redundancy)
- `feature_std`: [19] (same)
- `class_order`: ["healthy", "diseased", "OOD"] — explicit ordering

The loader at startup asserts that `class_order` matches `["healthy", "diseased", "OOD"]` exactly; mismatch is a fatal startup error.

### 12.4 Stage 2 — 5-way disease classifier

Stage 2 takes the same 19-dim feature vector and produces a 5-class distribution: `[P(foliar), P(septoria), P(late_blight), P(ylcv), P(mosaic)]` summing to 1. Stage 2 is conditional on Stage 1 saying diseased — but Stage 2 always runs (cheap), and its output is only used when Stage 1 reports diseased probability above threshold (soft routing, Section 12.5).

**Architecture (default):** softmax regression, same form as Stage 1 with 5 outputs instead of 3.

```
Input: x ∈ R^19  (same standardized vector)
W_stage2: shape [5, 19]
b_stage2: shape [5]
logits = W_stage2 @ x + b_stage2
probs = softmax(logits / T_stage2)  → shape [5]
```

Parameter count: 5 × 19 + 5 = 100. Combined with Stage 1's 60: 160 stage parameters total.

The calibration step (Section 12.8 Platt scaling) adds 14 more learnable parameters (2 per class × 7 classes), bringing the total learnable parameter count to 174.

Standardization parameters (per-feature mean and std at the classifier input) add another 38 fitted-from-data parameters (19 means + 19 stds). These are NOT learned by gradient descent — they are statistical estimates from the training set. Total parameters of all kinds: 174 + 38 = 212.

For comparison: the underlying signal models (v3, LoRA) have millions of parameters. The classifier is intentionally tiny because it operates on already-extracted features, with only 160 training images available.

**Storage.** `classifier_stage2.pkl` with the same schema as Stage 1, but `class_order = ["foliar", "septoria", "late_blight", "ylcv", "mosaic"]` (canonical disease order; healthy and OOD are not in Stage 2).

### 12.5 Soft routing combination

The two stages' outputs combine into a 7-class distribution via soft routing. Note that Stage 1 uses indices `[healthy=0, diseased=1, OOD=2]` while the final 7-class output uses canonical+OOD indices `[foliar=0, septoria=1, late_blight=2, ylcv=3, mosaic=4, healthy=5, OOD=6]`. The soft routing maps between these orderings using the explicit class names from each stage's `class_order` (loaded from the pickle metadata):

```
P_final[0]  = P_stage1[diseased] × P_stage2[foliar]
P_final[1]  = P_stage1[diseased] × P_stage2[septoria]
P_final[2]  = P_stage1[diseased] × P_stage2[late_blight]
P_final[3]  = P_stage1[diseased] × P_stage2[ylcv]
P_final[4]  = P_stage1[diseased] × P_stage2[mosaic]
P_final[5]  = P_stage1[healthy]
P_final[6]  = P_stage1[OOD]
```

These sum to 1 because Stage 1's three components sum to 1, Stage 2's five components sum to 1, and the multiplication preserves the partition.

Verification:
- Σ_{i=0..4} P_final[i] = P_stage1[diseased] × (Σ P_stage2[i]) = P_stage1[diseased] × 1 = P_stage1[diseased]
- P_final[5] + P_final[6] = P_stage1[healthy] + P_stage1[OOD] = 1 - P_stage1[diseased]
- Total: P_stage1[diseased] + (1 - P_stage1[diseased]) = 1. ✓

**Why soft (multiplicative) and not hard (gated):** a hard gate would say "if P_stage1[diseased] > 0.5, use Stage 2; else say healthy with P_stage1[healthy] confidence." Hard gates create cliffs at the threshold — a request with P_stage1[diseased] = 0.49 looks completely healthy; a request with P_stage1[diseased] = 0.51 looks completely diseased. Soft multiplication smoothly interpolates between these regimes, which is more honest to the underlying uncertainty.

**Why the disease distribution is constrained to sum to P_stage1[diseased]:** this is the algebra of conditional probability. P(disease_i) = P(disease_i | diseased) × P(diseased). Stage 2's softmax conditions on diseasedness; Stage 1's diseased probability scales it.

**Important:** the soft-routed distribution does NOT sum to 1 across only the 5 disease classes alone, NOR does P_final[5] (healthy) equal P_stage1[healthy] when interpreted naively as "P(healthy)". The probabilities are joint over the partition {foliar, septoria, late_blight, ylcv, mosaic, healthy, OOD}, which IS a partition (these 7 classes are mutually exclusive and collectively exhaustive in our model).

### 12.6 Logistic default; MLP escalation rule

The default classifier is multinomial logistic (softmax regression). A small MLP is the escalation option.

**MLP architecture (when used):**
```
Stage 1 MLP: 19 → 16 → 3   (with 1 hidden layer of 16 ReLU units)
Stage 2 MLP: 19 → 16 → 5
```

Parameter count: Stage 1 MLP has 19×16 + 16 + 16×3 + 3 = 371. Stage 2 MLP: 19×16 + 16 + 16×5 + 5 = 405. Combined: 776.

**Escalation rule.** F.0 trains both the logistic baseline and the MLP variant on out-of-fold splits and reports macro-F1 on each. The MLP is adopted only if it improves macro-F1 by **at least 2 percentage points** over the logistic baseline AND ECE remains under 0.10. If improvement is smaller or ECE worsens, logistic is kept.

The 2-point margin guards against overfitting the small training set with the larger MLP. If the gain is small, the simpler model is preferred.

**Default decision for v1:** logistic, unless F.0 sweeps demonstrate the MLP threshold is met. Section 4.5 lists `CLASSIFIER_VARIANT` as an env var (default `"logistic"`, values `"logistic"` or `"mlp"`).

### 12.7 Degraded-mode handling

The classifier must produce sensible output when one or more signals fail. The training procedure includes degraded-mode augmentation: during training, with probability `P_DEGRADE = 0.20`, one of the three signal blocks (v3 features 0-5 + 18, LoRA features 6-11, PSV features 12-15+17) is zeroed before standardization. This teaches the classifier that "all zeros for this signal" means "this signal failed; rely on the others."

**Per-block degradation probabilities (training-time):**
```
P_no_degrade        = 0.80   (all three signals present)
P_degrade_v3_only   = 0.07
P_degrade_lora_only = 0.07
P_degrade_psv_only  = 0.06

Verification: sum of degrade probabilities = 0.07 + 0.07 + 0.06 = 0.20 = P_DEGRADE.
Total probability mass = 0.80 + 0.20 = 1.00.
```
We don't degrade two signals simultaneously during training because the resulting input has very little information; we'd rather rely on the cascade-failure handling at the orchestrator level (Section 21).

**At inference**, signal failures are handled directly in `build_classifier_input` (Section 12.2 code): the corresponding feature block is zeroed before standardization. The classifier then produces a probability distribution that reflects the surviving signals.

**Training-time JSD handling.** When signal_a or signal_b is zeroed out by the degraded-mode augmentation, the JSD feature (index 16) is replaced with `JSD_SENTINEL` (the median JSD on F.0 calibration), NOT with the JSD computed between the zeroed signal and the surviving signal. This matches inference-time behavior (Section 12.2 build_classifier_input) and ensures the classifier learns to interpret JSD_SENTINEL as the "one signal failed" indicator. Without this matching, training would teach the classifier one JSD-handling rule and inference would use a different rule, hurting degraded-mode accuracy.

**Verification of degraded-mode quality.** F.0 evaluates degraded-mode performance by simulating each single-signal failure on the held-out subset and checking that:
- macro-F1 with v3 zeroed remains ≥ 0.55 (LoRA and PSV alone)
- macro-F1 with LoRA zeroed remains ≥ 0.55 (v3 and PSV alone)
- macro-F1 with PSV zeroed remains ≥ 0.65 (v3 and LoRA alone, since neural signals dominate)

If these targets are not met, F.0 increases `P_DEGRADE` and retrains.

### 12.8 Calibration via Platt scaling

The output of the soft-routed 7-class distribution is calibrated using **Platt scaling** on out-of-fold predictions.

**Why Platt and not temperature scaling alone:** temperature scaling rescales the logits uniformly (one parameter). Platt scaling fits a logistic regression on the model's output probabilities (two parameters per class — slope and intercept), which can correct asymmetric miscalibration. With 7 classes, Platt has 14 parameters (7 slopes + 7 intercepts), still small and overfitting-resistant.

**Algorithm:**
1. Out-of-fold prediction phase (Section 12.9) produces `P_final_oof` of shape [N_train, 7] — the model's prediction on each training image, generated using a model trained on the OTHER folds.
2. For each class c ∈ {0..6}:
   - Define `y_c = (true_label == c)` for each training image.
   - Define `p_c = P_final_oof[:, c]`.
   - Fit logistic regression: `p_c_calibrated = sigmoid(α_c × logit(p_c) + β_c)`
3. Store `α` and `β` arrays of shape [7] each in `phase_f0_calibration/classifier_platt.json`.

**At inference:**
```python
def apply_platt(P_final_uncal: np.ndarray) -> np.ndarray:
    # P_final_uncal: [7]; PLATT_ALPHA, PLATT_BETA: [7] each
    logits = np.log(P_final_uncal / (1.0 - P_final_uncal + 1e-12) + 1e-12)
    p_calibrated_per_class = 1.0 / (1.0 + np.exp(-(PLATT_ALPHA * logits + PLATT_BETA)))
    # Renormalize so the 7 calibrated probs sum to 1
    p_calibrated = p_calibrated_per_class / p_calibrated_per_class.sum()
    return p_calibrated
```

**The renormalization is necessary.** Per-class Platt scaling produces independent calibrated probabilities that don't naturally sum to 1. The renormalization restores partition-of-unity. This loses a small amount of information (the absolute scale of each calibrated prob) but is essential for downstream conformal prediction (Section 13) and tier assignment (Section 14), which expect a proper probability distribution.

**Calibration-after-renormalization caveat.** Per-class Platt fits sigmoid parameters that are well-calibrated for each class independently before renormalization. The renormalization step is a small distortion that may make the final probabilities slightly miscalibrated. F.0 measures ECE on the renormalized output (not before renormalization) and confirms it stays under the 0.10 target. If post-renormalization ECE is too high, an alternative is Dirichlet calibration (joint calibration over all classes simultaneously), which handles this more cleanly. For v1 we use per-class Platt with renormalization; v2 may upgrade to Dirichlet if F.0 measurements warrant.

**ECE target.** Post-calibration ECE on out-of-fold predictions must be under 0.10. F.0 reports this; if not met, the calibration is the bottleneck and either the classifier or the calibration procedure needs revision before deployment. ECE is measured on the 160-image train_subset's out-of-fold predictions (not on the 40-image held_out_subset, which is reserved for conformal). Use 10 equal-width bins; with ~16 samples per bin, the estimate is stable enough to detect the > 0.10 threshold reliably.

**If both classifier variants exceed the ECE target.** F.0 trains both the logistic baseline and the MLP variant (Section 12.6) with calibration. If both have ECE > 0.10 post-calibration, the calibration procedure itself is the bottleneck — neither model architecture is the cause. F.0 escalates to manual review: either the calibration approach must be revised (e.g., switch to Dirichlet) or the underlying classifier inputs need investigation (e.g., are the per-signal probabilities miscalibrated to start with?). The system does not deploy until ECE is acceptable.

### 12.9 Training procedure (out-of-fold)

The classifier and Platt parameters are jointly trained via 5-fold cross-validation on the field_val train_subset (160 images, source-stratified per Section 4.5).

```
Fold structure: 5 splits of 160/5 = 32 images each.
For fold k ∈ {0..4}:
  train_k = images in folds {0..4} \ {k}  (128 images)
  val_k   = images in fold {k}            (32 images)
  
  Fit Stage 1, Stage 2 on train_k:
    - Standardize features using train_k statistics
    - Optionally apply degraded-mode augmentation (Section 12.7)
    - Fit softmax regression with L2 regularization
  
  Predict on val_k:
    - Use train_k's standardization parameters (NOT val_k's — would leak)
    - Apply Stage 1, Stage 2, soft routing
    - Store P_final[val_k] in the out-of-fold prediction array P_oof
```

After 5 folds, `P_oof` covers all 160 training images. Each prediction was made by a model that hadn't seen the corresponding image in training. This is statistically valid for fitting Platt calibration parameters.

**Final model.** After OOF predictions are gathered for Platt calibration, the FINAL model is trained on ALL 160 images (using the combined feature_mean and feature_std, plus the degraded-mode augmentation). This final model is what ships to production. The Platt parameters fit on OOF predictions are used at inference with the final model — this is a small approximation (the final model is slightly different from any of the fold models) but standard practice in the stacking literature.

**Source-stratified folds.** The `field_val` train_subset has images from multiple source datasets (PlantVillage, Mendeley, Bangladesh multiclass, etc.). Folds are stratified so that each fold has roughly the same source distribution, preventing one fold from being dominated by a single source. Per Section 4.5, the source map is `data/specialist/model3/source_map.csv` (sacred file outside sandbox).

**Class balance.** Stage 1's three classes have natural imbalance (much more diseased than healthy or OOD in field_val). We use `class_weight="balanced"` in scikit-learn's logistic regression (or equivalent in PyTorch for MLP) to reweight the loss. This compensates for imbalance without resampling, which would distort Platt calibration.

**OOD class construction.** field_val does not contain explicit OOD examples. We construct OOD training examples synthetically:
1. Take 30-40 images from the okra and brassica training set, located at `data/specialist/model3/okra_brassica/` (sacred read-only directory outside the sandbox; see Section 2.6 sacred files list). The sandbox's F.0 OOD construction script reads from this path but never writes to it.
2. Run all three signals on these images. Their compatibility, v3, and LoRA outputs become the OOD training data, with label = OOD.
3. Add ~20 noise images (random Gaussian RGB tensors, deliberately scrambled or solid-color images) for additional OOD variety.

This is a known limitation: the OOD class is trained on a specific kind of OOD (other crops + noise). At inference, OOD images of unfamiliar types (e.g., a tomato disease class we don't model) may not score high on the OOD class — they may instead get spread across the disease classes. Section 30 (honest limitations) acknowledges this.

### 12.10 Output structure

```python
@dataclass
class ClassifierResult:
    p_final_calibrated: np.ndarray           # [7], post-Platt, sums to 1
    combined_argmax: int                     # 0-6 in canonical+OOD order
    combined_max_prob: float                 # max of p_final_calibrated
    combined_margin: float                   # max minus second-max
    p_final_uncalibrated: np.ndarray         # [7], pre-Platt, sums to 1 (for monitoring)
    p_stage1: np.ndarray                     # [3] healthy/diseased/OOD probs
    p_stage2: np.ndarray                     # [5] disease probs (only meaningful when stage1[diseased] is high)
    classifier_succeeded: bool               # False only if input was malformed
    failure_reason: str | None
```

The 7-class index space is:
- 0 = foliar
- 1 = septoria
- 2 = late_blight
- 3 = ylcv
- 4 = mosaic
- 5 = healthy
- 6 = OOD

This matches Section 2.4's "canonical with OOD" indexing. Tier assignment (Section 14) and conformal prediction (Section 13) read from this structure.

**`combined_max_prob` is the field TTA reads** (Section 11.2). The 1-view classifier output produces this; if it falls below the TTA trigger threshold, TTA fires and the classifier re-runs on aggregated signal outputs.

### 12.11 Where this lives in the sandbox

`tomato_sandbox/classifier.py` defines:
- `ClassifierResult` dataclass
- `build_classifier_input(sa, sb, sc) -> np.ndarray` (feature vector construction)
- `compute_classifier(sa, sb, sc) -> ClassifierResult` (forward pass through both stages plus Platt)
- The two stage forward functions

Calibration files loaded at startup:
- `tomato_sandbox/phase_f0_calibration/classifier_stage1.pkl`
- `tomato_sandbox/phase_f0_calibration/classifier_stage2.pkl`
- `tomato_sandbox/phase_f0_calibration/classifier_platt.json`
- `tomato_sandbox/phase_f0_calibration/classifier_feature_standardization.json`

If any of these files is missing, the sandbox refuses to start (Section 4.4 startup failure handling).

### 12.12 Performance budget

The classifier is the cheapest substantive component in the pipeline:

- Build feature vector (gather and combine signal outputs): <1 ms
- Standardize: <1 ms
- Stage 1 forward (3-output softmax of 19-dim): <1 ms
- Stage 2 forward (5-output softmax of 19-dim): <1 ms
- Soft routing combination: <1 ms
- Platt scaling: <1 ms

Total: ~3-5 ms median. Section 4.6 line "Hierarchical classifier" = 5 ms is consistent.

**At TTA**, the classifier runs twice (once on the 1-view signal outputs, once on the aggregated post-TTA outputs). Total classifier latency: ~10 ms with TTA.

These numbers are dominated by Python overhead, not the math. If the classifier is later ported to compiled code (numpy → numba or numpy → torch.jit), it would run in well under 1 ms total. For now the simple numpy implementation is fine.

---

## Section 13. Conformal prediction

### 13.1 Purpose

Conformal prediction provides a **statistically principled prediction set** rather than a single argmax with probability. Instead of "this is foliar with 0.62 confidence," conformal says "the true class is in {foliar, septoria} with 90% guaranteed coverage."

Why this matters for our use case:
1. Argmax + probability is what most classifiers produce, but a probability of 0.62 has limited interpretability for an end user. Is 0.62 high or low?
2. Tier assignment (Section 14) is fundamentally about "how much do we know." Prediction set size is a more direct answer than probability.
3. Conformal prediction's coverage guarantee — under exchangeability — is provable. Under our deployment conditions (Kerala field photos, possibly different from training distribution), the guarantee weakens to "approximately 90% empirical coverage on data similar to the calibration set."

The 90% target is a deliberate choice: 95% would produce larger sets (reducing the system's actionable utility); 80% would produce smaller sets but with weaker guarantees. 90% balances actionability with statistical safety.

### 13.2 Split-conformal algorithm

We use **split conformal prediction** (also called inductive conformal). The procedure:

**Calibration phase (one-shot, in F.0):**
1. Hold out a calibration set (the 40-image held_out_subset of field_val).
2. Run the calibrated classifier (Section 12.8) on each calibration image, producing `P_final_calibrated[i]` for i = 1..40.
3. For each calibration image, compute the **nonconformity score**:
   ```
   s_i = 1 - P_final_calibrated[i, y_true_i]
   ```
   where `y_true_i` is the ground-truth class. A high nonconformity score means the model assigned low probability to the true class (the model "doesn't conform" to the true label).
4. Sort the nonconformity scores. Compute the threshold:
   ```
   q = ceil((n + 1) × (1 - α)) / n
   τ = quantile(s_1, ..., s_n; q)
   ```
   where n = 40, α = 0.10 (for 90% coverage). With n=40 and α=0.10, q = ceil(41 × 0.9) / 40 = 37/40 = 0.925. τ = the 92.5th percentile of nonconformity scores.

**On the choice of nonconformity score.** The simple `1 - p_true_class` score (the "softmax nonconformity score") is used here for v1. An alternative is **Adaptive Prediction Sets** (APS, Romano et al. 2020), which uses cumulative probability mass and gives more uniform per-class coverage. Our setup has class imbalance (much more diseased than OOD); a pure 1-p score may give unequal per-class coverage. F.0 reports per-class empirical coverage; if some classes have substantially worse coverage than others (e.g., OOD coverage at 75% while overall is 90%), v2.0 may switch to APS. For v1, the simpler score is preferred because it's easier to debug and explain.

**Inference phase (per request):**
1. Run the calibrated classifier; get `P_final_calibrated[c]` for c = 0..6.
2. For each class c, compute nonconformity:
   ```
   s_c = 1 - P_final_calibrated[c]
   ```
3. The prediction set is:
   ```
   PredSet = {c : s_c <= τ}  =  {c : P_final_calibrated[c] >= 1 - τ}
   ```

The set is non-empty if at least one class has probability ≥ 1-τ. Empty set is rare with proper calibration but possible if the classifier puts all probability mass into one class barely above the threshold's complement.

### 13.3 The calibration set

The calibration set is the **40-image held_out_subset of field_val**. Section 1.6 glossary (Turn 1) defined this as the 20% slice of the 203-image field_val that was NOT used for training.

**Critical: the calibration set must not have been seen by the classifier.** Out-of-fold predictions (Section 12.9) use the train_subset for both training and calibration of Platt, but the held_out_subset is reserved exclusively for conformal calibration. This is the validity guarantee for split conformal.

40 is a small calibration set. With n=40 and the (n+1)/(n)-quantile rule, the achievable coverage levels are quantized. At 40 examples, valid coverage targets are 1/40, 2/40, ..., 40/40. We pick 36/40 = 90% as our target. Empirical coverage may differ from 90% by approximately ±5% due to finite-sample variation (the standard error of a binomial proportion at p=0.9, n=40 is sqrt(0.9 × 0.1 / 40) ≈ 0.047, giving a 95% CI of roughly [80.7%, 99.3%]). This is wide; the conformal guarantee is theoretical-marginal and the empirical confidence interval reflects how much that guarantee can vary in practice with this sample size.

A larger calibration set would tighten this. If the project later acquires another 60-100 labeled Kerala images, the calibration set should be expanded for tighter coverage.

### 13.4 Coverage target and what it means

**Coverage target: 90%.**

The coverage guarantee under exchangeability:
> P(y_true ∈ PredSet) ≥ 1 - α = 0.90

This holds in expectation over both the training/calibration randomness AND the new test point. It does NOT hold for any specific test point — for a particular hard image, the prediction set might miss the true class.

**Empirical coverage on the calibration set:** by construction, the threshold τ produces approximately 90% coverage on the calibration set. The PDA review reported 89.05% empirical coverage at the 90% target on synthetic data. On real data after F.0, this may be slightly different.

**What 90% coverage doesn't promise:**
- It's not "the prediction is 90% likely to be the argmax."
- It's not "the system is right 90% of the time."
- It's not a guarantee against systematic errors (if the training distribution differs from production, the guarantee fails).

What it DOES promise (under exchangeability): for the prediction set produced for a new test point, the true class is in that set 90% of the time on average.

### 13.5 Threshold τ derivation

```python
def compute_conformal_tau(p_final_calibrated_holdout: np.ndarray, y_true: np.ndarray, alpha: float = 0.10) -> float:
    """
    p_final_calibrated_holdout: [N, 7] array of calibrated probabilities on held_out_subset
    y_true: [N] integer labels (0-6)
    Returns: τ ∈ [0, 1]
    """
    N = len(y_true)
    nonconformity_scores = 1.0 - p_final_calibrated_holdout[np.arange(N), y_true]
    q = np.ceil((N + 1) * (1 - alpha)) / N
    q = min(q, 1.0)  # clip in case of edge case where q > 1
    tau = np.quantile(nonconformity_scores, q, method="higher")
    return float(tau)
```

**Implementation detail — `method="higher"`:** for finite samples, the (n+1)*(1-α)/n quantile lies between two of the actual scores. The conservative choice is the upper one, ensuring the empirical coverage is at LEAST 1-α, not just approximately. numpy's "higher" interpolation method gives this.

The threshold τ is stored at `tomato_sandbox/phase_f0_calibration/conformal_tau.json`:
```json
{
  "tau": 0.6234,
  "alpha": 0.10,
  "calibration_set_size": 40,
  "calibration_date": "2026-MM-DD",
  "model_version": "<7-char hash>"
}
```

The `model_version` field is the first 7 characters of the SHA-256 of the concatenated bytes of these four files (in this fixed order, no separators):
1. `classifier_stage1.pkl`
2. `classifier_stage2.pkl`
3. `classifier_platt.json`
4. `classifier_feature_standardization.json`

This protects against using a stale τ with a re-trained classifier. If any of the four files changes, the hash changes, and τ MUST be re-derived. The loader at startup recomputes the hash and refuses to start if it doesn't match the stored model_version.

### 13.6 Monthly re-fit policy

Conformal prediction's coverage guarantee assumes exchangeability — the calibration distribution looks like the inference distribution. In practice, distributions drift: seasonal patterns in Kerala agriculture, new pests, changes in phone camera technology.

**The monthly re-fit policy:**
- Once per calendar month, the system re-derives τ from a freshly-collected calibration set.
- The calibration set is enriched with the past month's confirmed predictions (those vetted by an agronomist; see Section 23 for the agronomist queue).
- If fewer than 30 newly-confirmed predictions are available, the calibration set falls back to the original 40 held_out_subset images plus whatever new confirmed predictions exist.
- The new τ replaces the old one in `conformal_tau.json`. The old τ is archived to `tomato_sandbox/phase_f0_calibration/conformal_archive/conformal_tau_YYYY-MM.json` (where YYYY-MM is the calendar month at archive time). The archive directory is created on first archive operation.

**Empirical coverage monitoring** (Section 25):
- Every prediction set is logged; agronomist confirmation indicates whether the true class was in the set.
- The system tracks rolling 30-day empirical coverage. If coverage drops below 85% (compared to 90% target), an alert fires for review.

**A caveat.** Re-fitting τ on agronomist-confirmed predictions introduces selection bias: easy cases (Tier 1, Tier 2) are confirmed quickly; ambiguous cases (Tier 3, 4) sit in queue longer or get marked as "uncertain" by the agronomist. The re-fit calibration set may over-represent easy cases, producing a τ that's too small and a coverage guarantee that's optimistic. This is a known limitation; Section 30 lists it explicitly. Mitigation: include all cases in the re-fit set (including agronomist-uncertain) with the agronomist's best-guess label, weighted lower if confidence in the label is itself uncertain. Implementation deferred to v2.0.

### 13.7 Output structure

```python
@dataclass
class ConformalResult:
    prediction_set: list[int]                # canonical+OOD indices in the set
    prediction_set_size: int                 # len(prediction_set)
    threshold_tau_used: float                # the τ that produced this set
    nonconformity_per_class: np.ndarray      # [7], 1 - p_calibrated[c] for each c
    inside_set_per_class: np.ndarray         # [7] bool, True if class is in the set
```

The prediction set is the primary output. Its size is what tier assignment uses (Section 14): size 1 → Tier 1 or 2 depending on confidence; size 2 → Tier 3A; size 3+ → Tier 3B; etc.

### 13.8 Performance budget

Conformal prediction is essentially free at inference:
- Look up τ (constant): <1 ms (already loaded)
- Compute s_c = 1 - p_c for 7 classes: <1 ms
- Compare s_c <= τ for 7 classes: <1 ms
- Build prediction set list: <1 ms

Total: ~1 ms. Section 4.6 line "Conformal threshold lookup" = 1 ms is consistent.

The expensive part is the calibration phase (Section 13.5), which runs once at F.0 and once per month thereafter. Calibration is offline and doesn't enter inference latency.

---

## Section 14. Tier assignment rules

### 14.1 Purpose and tier overview

Tier assignment translates the classifier's probability distribution and the conformal prediction set into a categorical label that drives downstream behavior. Different tiers trigger different UI presentations, severity grading hooks, and agronomist queue priorities (Section 23).

The system uses a **tier label** (one of 1, 2, 3A, 3B, 3C, 3D, 4A, 4B) plus an **optional Tier 5 alert flag** that elevates dangerous diseases regardless of tier.

Tiers are assigned by a **rule chain** evaluated in priority order (Section 14.5). The first matching rule wins. The rules examine: classifier `combined_max_prob`, `combined_margin`, the conformal prediction set's size and contents, IQA decision, signal failure flags, `psv_reliability`, `chilli_leakage`, and the per-class minimum-recall guard.

The overarching design principles:
- A request always gets exactly one tier label.
- Tier 5 is a separate flag that can fire alongside any tier 1-4.
- Higher confidence → lower tier number.
- The system errs toward higher-tier labels (more cautious) when uncertainty is detected.

### 14.2 Tier label definitions

#### Tier 1 — Definitive prediction

The system has high confidence in a single answer. UI shows the prediction without hedging language.

**Conditions (all must hold):**
- `prediction_set_size == 1`
- `combined_max_prob >= 0.85`
- `combined_margin >= 0.30`
- IQA decision is `ACCEPTABLE` or `HIGH`
- All three signals succeeded (v3, LoRA, PSV)
- `psv_reliability >= 0.50`
- `chilli_leakage < 0.20`

If the predicted class is one of {late_blight, mosaic, ylcv}, Tier 5 alert fires alongside Tier 1.

**Typical UI:** "This appears to be Septoria Leaf Spot."

#### Tier 2 — Confident prediction

Single answer, but with moderate confidence — the system narrows to one class but doesn't claim certainty.

**Conditions:**
- `prediction_set_size == 1`
- `combined_max_prob >= 0.65`
- `combined_margin >= 0.20`
- IQA decision is `ACCEPTABLE`, `HIGH`, or `DEGRADED`
- All three signals succeeded
- `psv_reliability >= 0.40`
- `chilli_leakage < 0.30`

**Note on the absent upper bound on `combined_max_prob`.** Tier 2's definition does not include `combined_max_prob < 0.85`. The rule chain in Section 14.5 puts Rule 7 (Tier 1) before Rule 8 (Tier 2), so any case with `combined_max_prob >= 0.85` AND all other Tier 1 conditions met fires Rule 7 first and never reaches Rule 8. Cases with `combined_max_prob >= 0.85` but failing one of Rule 7's stricter conditions (e.g., `combined_margin < 0.30`, or `psv_reliability < 0.50`, or `chilli_leakage >= 0.20`) correctly route to Tier 2 via Rule 8. Including `combined_max_prob < 0.85` in the Tier 2 definition would incorrectly route these cases to Rule 9's catch-all.

**Typical UI:** "This is likely Septoria Leaf Spot. Confidence: medium."

#### Tier 3A — Two-class ambiguity

The conformal prediction set contains exactly two classes. The system cannot decide between them.

**Conditions:**
- `prediction_set_size == 2`
- All other Tier 1/2 conditions don't hold
- All three signals succeeded
- IQA decision is `ACCEPTABLE`, `HIGH`, or `DEGRADED`

**Typical UI:** "This appears to be either Septoria Leaf Spot or Foliar Spot. Both are similar small-lesion diseases; recommend agronomist confirmation."

#### Tier 3B — Three-or-more-class ambiguity

The conformal prediction set contains 3 or more classes.

**Conditions:**
- `prediction_set_size >= 3`
- IQA decision is `ACCEPTABLE`, `HIGH`, or `DEGRADED`

**Typical UI:** "Multiple possible diagnoses: Septoria, Foliar Spot, or Late Blight. Recommend agronomist consultation; provide additional photo of affected leaf if possible."

#### Tier 3C — PSV unreliable or chilli leakage

PSV's compatibility scoring cannot be trusted, OR v3 thinks the image is more chilli than tomato.

**Conditions (any of):**
- `psv_reliability < 0.40`, OR
- `chilli_leakage > 0.40`

**Typical UI:** "Image content is unusual or cannot be analyzed by all systems. The visible-symptom analysis was unreliable. Result based on neural models alone." OR "This image may be a chilli plant rather than tomato. Please verify which crop you're photographing."

#### Tier 3D — DEGRADED IQA ceiling

Image quality was DEGRADED (Section 6.4). Section 6 contracted that this caps the result at Tier 3.

**Conditions:**
- IQA decision is `DEGRADED`
- The request would otherwise have been Tier 1 or 2

This rule **caps** the tier at 3D rather than producing a fresh classification. The classifier output is still computed and shown; the user is just told the image quality limited certainty.

**Typical UI:** "Image quality was below ideal. Best estimate: Septoria Leaf Spot. Recommend retaking the photo for higher confidence."

#### Tier 4A — Low confidence

The classifier is uncertain across all classes; no single answer dominates.

**Conditions:**
- `combined_max_prob < 0.45`
- All three signals succeeded
- IQA decision is `ACCEPTABLE`, `HIGH`, or `DEGRADED`
- The other Tier 3 rules don't fire

**Typical UI:** "I'm not confident about this image. The system cannot identify a clear disease pattern. Please consult an agronomist." Plus the top 3 classes by probability shown without strong claims.

#### Tier 4B — Pipeline failure / degraded mode

One or more signals failed (`forward_succeeded == False` for v3, LoRA, or PSV).

**Conditions (any of):**
- `signal_a.forward_succeeded == False`, OR
- `signal_b.forward_succeeded == False`, OR
- `signal_c.forward_succeeded == False`

**Typical UI:** "Some analysis components failed on this image. The result is based on the components that succeeded. Confidence is reduced; recommend retaking the photo or consulting an agronomist."

### 14.3 Tier 5 alert flag

Tier 5 is **not a replacement for the tier label**. It is an additional flag that elevates the agronomist priority for dangerous diseases.

**Conditions for Tier 5 flag (any of):**
- `combined_argmax in {late_blight, mosaic, ylcv}` AND `combined_max_prob >= 0.20`, OR
- `late_blight in prediction_set` AND `late_blight_prob >= 0.20`

The 0.20 threshold is intentionally low: even a 20% probability of late_blight is worth alerting on, because:
- Late blight is rapidly destructive — delay of 1-2 days can lose a crop
- Mosaic and YLCV are viruses with no cure; early detection enables containment (removing infected plants)

A request can simultaneously be Tier 3B (ambiguous) AND Tier 5 (dangerous-disease flag fired). The UI shows both: "Multiple possible diagnoses including Late Blight (24%) — recommend immediate agronomist consultation regardless of other findings."

**Why Tier 5 ignores the prediction set inclusion test for late_blight:** the conformal set might exclude late_blight if its probability is just below 1-τ. We don't want to miss a 24% late_blight probability just because the conformal threshold happened to put it outside. Tier 5 uses raw probability for the late_blight in_set rule, not the prediction set membership.

**Why mosaic and YLCV use only the argmax trigger, not the in_set trigger:** the asymmetry is intentional. Late blight is the most acutely destructive — symptom-to-crop-loss timeline is days, not weeks. Even a 20% probability of late blight is worth alerting on, regardless of whether it's the argmax. Mosaic and YLCV are viral diseases with weeks-to-months timelines; while still important to detect early, the urgency is lower. Triggering Tier 5 only on argmax for these diseases reduces false alarms (the agronomist queue is finite). If the system later acquires evidence that more aggressive triggering for mosaic/YLCV would help, the tier_rules.yaml schema (Section 14.6) supports adding `also_alert_if_in_prediction_set` for those classes too.

### 14.4 Per-class minimum-recall guard

For classes with very few training samples (e.g., YLCV with n=2 in final_val per Sandbox Directive memory), the classifier's learned weights are unreliable. A per-class minimum-recall guard catches these:

**The guard.** For each class c, F.0 measures the classifier's recall on the train_subset's out-of-fold predictions. If recall < `MIN_PER_CLASS_RECALL` (default 0.50, configured in `phase_f0_calibration/min_recall_guard.json`), that class is flagged as "underpowered."

**OOF recall as a proxy.** OOF recall comes from per-fold models, each trained on 4/5 of the data. The final shipped model is trained on all 160 images, so its per-class recall may differ from OOF recall — typically slightly better (more training data) but occasionally worse if per-fold averaging happened to be helpful. F.0 also reports the in-sample (final-model on train_subset) recall as a sanity check; if the in-sample and OOF recall diverge by more than 15 points for some class, that's a flag for overfitting and the underpowered guard parameters should be reviewed. The guard uses OOF recall (not in-sample) because OOF generalizes better to held-out predictions.

For underpowered classes:
- Tier 1 and Tier 2 are NOT allowed for that class. If `combined_argmax` is an underpowered class with `combined_max_prob >= 0.65`, the tier is downgraded to Tier 3A (or Tier 5 if dangerous).
- The agronomist UI is alerted: "The system has limited training data for this class; treat the prediction as suggestive, not definitive."

**Underpowered classes expected at deployment.** Classes with very few training samples in the field_val train_subset (160 images split across 7 classes with imbalance) are likely candidates. Based on the broader field_val composition, YLCV is expected to be the most-likely-underpowered class (it has small representation in field_val overall). Mosaic may also be underpowered. F.0 reports the per-class training counts and recall numbers; the actual underpowered set is data-determined, not assumed.

(Note: the `final_val` LOCK-4 set has n=2 for YLCV and n=4 for mosaic per the project's Sandbox Directive memory, but `final_val` is a held-out evaluation set, not the training set. Its small per-class counts are a separate concern — they limit the system's ability to STATISTICALLY VALIDATE per-class accuracy at deployment, not the classifier's training quality.)

### 14.5 Priority ordering and rule chain

Rules are evaluated in this fixed priority order. The first matching rule wins for the tier label. Tier 5 is computed independently afterward.

```
Rule 1 (highest priority): Pipeline failure
  IF signal_a/b/c.forward_succeeded == False:
    → Tier 4B
    
Rule 2: REJECT IQA (already handled at IQA stage)
  IF IQA.decision == "REJECT":
    → request never reaches tier assignment; HTTP 400 returned at IQA gate
    
Rule 3: PSV unreliable or chilli leakage
  IF psv_reliability < 0.40 OR chilli_leakage > 0.40:
    → Tier 3C
    
Rule 4: Low confidence
  IF combined_max_prob < 0.45:
    → Tier 4A
    
Rule 5: Three-or-more-class prediction set OR empty prediction set
  IF prediction_set_size == 0:
    → Tier 4A (empty set treated as low confidence)
  ELIF prediction_set_size >= 3:
    → Tier 3B
    
Rule 6: Two-class prediction set
  IF prediction_set_size == 2:
    → Tier 3A
    
Rule 7: Single-class prediction set, definitive
  IF prediction_set_size == 1
     AND combined_max_prob >= 0.85
     AND combined_margin >= 0.30
     AND psv_reliability >= 0.50
     AND chilli_leakage < 0.20:
    Sub-rule 7a: IF IQA.decision == "DEGRADED":
      → Tier 3D
    Sub-rule 7b: IF combined_argmax is underpowered (Section 14.4):
      → Tier 3A
    Sub-rule 7c (default):
      → Tier 1
    
Rule 8: Single-class prediction set, confident
  IF prediction_set_size == 1
     AND combined_max_prob >= 0.65
     AND combined_margin >= 0.20
     AND psv_reliability >= 0.40
     AND chilli_leakage < 0.30:
    Sub-rule 8a: IF IQA.decision == "DEGRADED":
      → Tier 3D
    Sub-rule 8b: IF combined_argmax is underpowered:
      → Tier 3A
    Sub-rule 8c (default):
      → Tier 2
    
Rule 9 (catch-all): Should not happen if rules above are correct
  → Tier 4A (treat as low confidence)

After tier label is assigned:
Tier 5 alert evaluated independently (Section 14.3).
```

**Why this priority ordering:**
- Rule 1 (pipeline failure) is highest because we cannot trust any other signal interpretation when a forward pass failed.
- Rules 3 and 4 (PSV unreliable / low confidence) come before set-size rules because they capture deeper systemic issues, not just ambiguity.
- Set-size rules (5, 6) before confidence rules (7, 8) because if the conformal set is large, we can't claim confidence even if max_prob is high (the high probability would be on a class that's still in a multi-class set).
- Confidence rules (7, 8) use both probability and margin, plus IQA cap, plus PSV reliability, plus chilli leakage. The thresholds in Rules 7 and 8 are tighter than Rule 3's threshold by design: Rule 3 rules out the worst PSV reliability (<0.40) and worst chilli leakage (>0.40), but Tier 1 demands stricter quality (PSV ≥ 0.50, leakage < 0.20) than Tier 2 (PSV ≥ 0.40, leakage < 0.30). Between Rule 3's threshold and Tier 1's threshold (e.g., PSV reliability 0.45), the request falls through to Rule 8 (Tier 2) or fails Rule 8's threshold and lands at Rule 9 (Tier 4A) — see Section 14 example scenarios in Section 15.
- Within Rules 7 and 8, sub-rule 7a/8a (IQA DEGRADED cap) takes precedence over sub-rule 7b/8b (underpowered class cap). Rationale: DEGRADED IQA implies a user-actionable retake suggestion, which is more directly useful than the "underpowered class" caveat which may not have an obvious user fix.

**Rule 9 should never fire** in correct execution: every input either triggers an earlier rule or has `prediction_set_size == 1` with sufficient confidence (Rule 7) or moderate confidence (Rule 8). If it does fire, it indicates either:
- A subtle interaction in thresholds (e.g., max_prob 0.50, margin 0.05, PSV 0.45 — none of Rules 1-8 fires), OR
- A logic bug in the rule chain.

Either way, the system logs a warning with the full input snapshot for offline analysis.

### 14.6 tier_rules.yaml schema

The rules above are encoded in a versioned YAML file at `tomato_sandbox/config/tier_rules.yaml`:

```yaml
schema_version: 1
priority_chain:
  - id: pipeline_failure
    tier: "4B"
    conditions:
      any_of:
        - signal_a_failed
        - signal_b_failed
        - signal_c_failed
  
  - id: psv_unreliable_or_chilli_leakage
    tier: "3C"
    conditions:
      any_of:
        - psv_reliability_below: 0.40
        - chilli_leakage_above: 0.40
  
  - id: low_confidence
    tier: "4A"
    conditions:
      combined_max_prob_below: 0.45
  
  - id: three_plus_class_set
    tier: "3B"
    conditions:
      prediction_set_size_at_least: 3
  
  - id: two_class_set
    tier: "3A"
    conditions:
      prediction_set_size_equals: 2
  
  - id: definitive_single_class
    tier: "1"
    conditions:
      all_of:
        - prediction_set_size_equals: 1
        - combined_max_prob_at_least: 0.85
        - combined_margin_at_least: 0.30
    cap_when_iqa_degraded: "3D"
    cap_when_class_underpowered: "3A"
  
  - id: confident_single_class
    tier: "2"
    conditions:
      all_of:
        - prediction_set_size_equals: 1
        - combined_max_prob_at_least: 0.65
        - combined_margin_at_least: 0.20
    cap_when_iqa_degraded: "3D"
    cap_when_class_underpowered: "3A"

tier5_alerts:
  enabled_for_classes: [late_blight, mosaic, ylcv]
  also_alert_if_in_prediction_set: [late_blight]
  min_probability: 0.20
```

This is human-readable so the agronomist can audit and adjust rules during F.0 review. The schema version protects future code from incompatible changes; the loader at startup asserts `schema_version == 1`.

**YAML operator vocabulary (schema version 1).** The schema has three layers of vocabulary:

*Rule-level fields* (top-level inside each rule entry):
- `id` — unique string identifier for the rule (used in TierAssignment.rule_id_fired)
- `tier` — the tier label assigned when the rule matches (e.g., "1", "3A", "4B")
- `conditions` — the condition tree (uses operators below)
- `cap_when_iqa_degraded` — optional; if specified, downgrades to this tier label when IQA decision is DEGRADED
- `cap_when_class_underpowered` — optional; if specified, downgrades to this tier label when the per-class minimum-recall guard fires (Section 14.4)

*Condition operators* (used inside `conditions`):
- `<field>_below: X` — value strictly less than X (`value < X`)
- `<field>_at_most: X` — value less than or equal to X (`value <= X`)
- `<field>_above: X` — value strictly greater than X (`value > X`)
- `<field>_at_least: X` — value greater than or equal to X (`value >= X`)
- `<field>_equals: X` — value equals X (`value == X`)

*Composition operators* (used inside `conditions`):
- `all_of: [...]` — all listed conditions must hold (AND)
- `any_of: [...]` — at least one listed condition must hold (OR)

*Boolean indicators* (no operator needed; the indicator name itself is the condition):
- `signal_a_failed` — maps to `not signal_a.forward_succeeded`
- `signal_b_failed` — maps to `not signal_b.forward_succeeded`
- `signal_c_failed` — maps to `not signal_c.forward_succeeded`
- `iqa_decision_is_degraded` — maps to `iqa.decision == "DEGRADED"`
- `iqa_decision_is_acceptable_or_high` — maps to `iqa.decision in {"ACCEPTABLE", "HIGH"}`
- `class_underpowered` — maps to per-class guard check (Section 14.4)

The YAML parser in `tier_assignment.py` translates these into the runtime checks. Adding new fields, operators, or indicators requires bumping `schema_version` and updating both the YAML and the parser. The current vocabulary is intentionally small so that the YAML stays auditable by the agronomist without programming knowledge.

### 14.7 Output structure

```python
@dataclass
class TierAssignment:
    tier_label: str                          # one of "1", "2", "3A", "3B", "3C", "3D", "4A", "4B"
    tier5_alert: bool                        # True if Tier 5 fired
    rule_id_fired: str                       # which rule_id from tier_rules.yaml matched
    sub_rule_id_fired: str | None            # for sub-rules like 7a, 7b, 7c
    reasons: list[str]                       # human-readable list of conditions that fired
    reasons_structured: dict                 # parallel structured form (see below); for monitoring
    underpowered_class_downgrade: bool       # True if the per-class guard fired
```

**Reasons list example (human-readable, for UI):**
- `["combined_max_prob: 0.91", "combined_margin: 0.35", "prediction_set_size: 1", "psv_reliability: 0.78"]` — for a Tier 1 result
- `["signal_b.forward_succeeded: False", "failure_reason: numerical_instability"]` — for Tier 4B

**Reasons structured (machine-readable, for monitoring queries):**
```python
{
  "combined_max_prob": 0.91,
  "combined_margin": 0.35,
  "prediction_set_size": 1,
  "psv_reliability": 0.78,
  "chilli_leakage": 0.05,
  "iqa_decision": "ACCEPTABLE",
  "signal_a_failed": False,
  "signal_b_failed": False,
  "signal_c_failed": False,
  "underpowered_class": False,
}
```

The structured form lets monitoring queries like "% of Tier 4B caused by signal_b failures vs signal_a failures" run reliably without parsing the human-readable strings. The `reasons` list is for UI display where readability matters; `reasons_structured` is for downstream code.

### 14.8 Where this lives in the sandbox

`tomato_sandbox/tier_assignment.py` defines:
- `TierAssignment` dataclass
- `assign_tier(classifier_result, conformal_result, iqa_result, signal_a, signal_b, signal_c) -> TierAssignment`
- The rule chain evaluator that reads `tier_rules.yaml`

The YAML file is at `tomato_sandbox/config/tier_rules.yaml`. Loaded at startup; reloaded if changed during the system's lifetime is NOT supported (would require a server restart).

The 100+ specific decision scenarios — every realistic combination of signal outputs and how the system labels them — are documented in **Section 15** (next turn). Section 14 here defines the rules; Section 15 illustrates them with concrete cases.

### 14.9 Performance budget

Tier assignment is essentially free at inference:
- Read scalars from inputs (classifier result, conformal result, IQA, signals): <1 ms
- Walk the rule chain (at most 9 simple boolean comparisons): <1 ms
- Build TierAssignment dataclass + reasons: ~2 ms (Python overhead dominates)
- YAML parse: 0 ms (parsed once at startup, cached as in-memory dict)

Total: ~2-5 ms median. Section 4.6 line "Tier assignment 5 ms" is consistent.

The performance is independent of how complex the rule chain is, because each rule is a simple comparison. If the chain grew to dozens of rules with rich conditions, performance would still be < 10 ms.

---

## Section 15. Decision scenarios

### 15.1 Purpose and how to read this section

Section 14 defined the tier rules abstractly. This section enumerates concrete scenarios — every realistic combination of signal outputs that the system might encounter — and shows the expected tier outcome for each. The purpose is threefold:

1. **Behavioral test suite.** Each scenario is a test case. The implementation of `assign_tier` (Section 14.8) MUST produce the documented outcome for every scenario. The Phase F.0 validation (Section 29) runs each scenario as an integration test before deployment.
2. **Reasoning audit.** By walking through 100+ scenarios, the spec exposes the rule chain's behavior to the agronomist and reviewer. Misalignments between intended behavior and rule output surface here, before deployment, when they are cheap to fix.
3. **Documentation for downstream code.** Section 16 (response builder), Section 17 (severity grading), and Section 23 (agronomist queue) all read tier outcomes. Showing concrete tier→outcome examples helps those sections design their handling.

The scenarios cover:
- All 8 tier labels (1, 2, 3A, 3B, 3C, 3D, 4A, 4B).
- Tier 5 alert flag interactions across multiple base tiers.
- All three signal failure modes (v3, LoRA, PSV).
- IQA decision cap path (DEGRADED → 3D).
- Per-class underpowered downgrade (Section 14.4).
- Boundary values at every threshold.
- Empty prediction set (rare path through Rule 5).
- Cross-signal disagreement (high JSD, mixed argmaxes).
- TTA-specific behaviors (initial vs post-TTA tier).

135 scenarios are documented. The count is illustrative; the rule chain is exhaustive over (signal_state × IQA × conformal × class) and will produce a tier for any input. The scenarios represent every category we expect in practice plus edge cases at boundaries.

### 15.2 Format and notation

Each scenario uses this compact format:

```
**S<id> — <one-line description>**
- v3: probs=[P_foliar, P_septoria, P_late_blight, P_ylcv, P_mosaic, P_healthy], chilli_leak=X, succeeded=BOOL
- LoRA: probs=[same canonical 6 entries], succeeded=BOOL
- PSV: argmax=N (class_name), max=X, margin=X, reliability=X, succeeded=BOOL
- IQA: <ACCEPTABLE | HIGH | DEGRADED | REJECT>
- Classifier: argmax=N (class_name), max=X, margin=X
- Conformal (τ=X): set={class_indices}, size=N
- → **Tier <label>**, T5 alert: BOOL (rule fired: rule_id, sub-rule sub_id)
- Walk: <which rules check, why they pass/fail; only included for non-obvious cases or boundary scenarios>
```

**Notation conventions:**
- Class indices follow the canonical+OOD order: `0=foliar, 1=septoria, 2=late_blight, 3=ylcv, 4=mosaic, 5=healthy, 6=OOD`.
- "succeeded" defaults to True; we only mention when False.
- IQA defaults to `ACCEPTABLE`; we only mention when different.
- For brevity, "T5 alert" means the Tier 5 alert flag value.

**Six conventions readers and test authors must understand:**

1. **v3 probability sums.** Per Section 8.4, v3's 6 tomato_probs_canonical sum to `(1 - chilli_leakage)`. All scenarios in this section show v3 vectors that satisfy this constraint exactly — for example, S1.1's v3 vector sums to 0.97 (chilli_leak=0.03). The Phase F.0 test infrastructure (Section 29) reads the displayed v3 vector and chilli_leakage value directly without renormalization. (Earlier drafts of this section showed v3 vectors summing to 1.0 with chilli_leakage as a separate field; that draft was renormalized so that displayed values match the production contract.)

2. **τ is illustrative, not production.** Each scenario shows a τ value that produces the stated prediction set under the given classifier output. In production, F.0 fits a single τ once on the held-out subset (Section 13.5) and that τ applies to every inference within the deployment. Scenarios use varying τ values to demonstrate that the rule chain handles each set-size outcome; this is not a claim that τ varies per request. F.0-derived τ in practice falls roughly in [0.30, 0.70]; some scenarios in this section use τ outside that range to illustrate edge cases (e.g., τ = 0.85 represents a poorly-calibrated classifier where most classes need to enter the set to achieve 90% coverage).

3. **τ chosen to make set valid.** When a scenario specifies both a classifier output (max, margin) and a prediction set, the τ shown is the value at which conformal admits exactly the stated set. The test infrastructure constructs a full 7-class P_final_calibrated distribution that produces the stated set under the given τ. The argmax / max / margin values fully determine the tier-rule behavior; the conformal set test is checked separately by reverse-engineering a compatible distribution.

4. **Walk traces are provided selectively.** A "Walk" line is included for boundary scenarios, scenarios where multiple rules might appear to apply, and scenarios where the naive expectation differs from the actual rule-chain output. Clean scenarios with obvious rule paths (e.g., a Tier 1 case where every condition is well above its threshold) omit the walk for brevity. The convention is: if a careful reader has to think for more than two seconds about why a particular tier was assigned, the scenario gets a walk.

5. **Underpowered-class assumption.** Scenarios involving YLCV (class 3) or mosaic (class 4) at high confidence assume those classes pass the per-class minimum-recall guard (Section 14.4). In practice, F.0 may flag YLCV and/or mosaic as underpowered (because of their lower training counts), in which case sub-rules 7b/8b downgrade Tier 1/2 to Tier 3A. Scenarios that explicitly test the downgrade path (the SUP series in 15.13) make the underpowered flag explicit; other scenarios involving YLCV or mosaic proceed without a downgrade, which represents the favorable case where F.0's measurements clear the recall guard.

6. **Probability values are illustrative, qualitative patterns are normative.** The numeric probability values shown are example values that produce the labeled outcome. F.0 produces actual model outputs that may differ from these examples. The scenarios are designed so that the rule chain produces the labeled tier for any input matching the qualitative pattern (e.g., "all signals agree, max prob in [0.85, 1.0], margin >= 0.30, IQA ACCEPTABLE, PSV reliability >= 0.50, chilli_leakage < 0.20" → Tier 1).

Scenarios are grouped by intended outcome, not by input pattern. A reader looking for "what happens when PSV is unreliable" goes to subsection 15.7; a reader looking for "what does Tier 1 look like in practice" goes to 15.3.

### 15.3 Tier 1 scenarios — definitive prediction

All Tier 1 scenarios share: `prediction_set_size==1`, `combined_max_prob >= 0.85`, `combined_margin >= 0.30`, IQA `ACCEPTABLE` or `HIGH`, all signals succeeded, `psv_reliability >= 0.50`, `chilli_leakage < 0.20`. Sub-rule 7c (default) fires.

**S1.1 — Clean foliar prediction**
- v3: probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03
- LoRA: probs=[0.88, 0.05, 0.02, 0.02, 0.02, 0.01]
- PSV: argmax=0 (foliar), max=0.71, margin=0.45, reliability=0.78
- IQA: ACCEPTABLE
- Classifier: argmax=0 (foliar), max=0.91, margin=0.86
- Conformal (τ=0.40): set={0}, size=1
- → **Tier 1**, T5 alert: False (rule 7c)
- Walk: all Rule 7 main conditions met (max>=0.85, margin>=0.30, psv_reliability>=0.50, chilli<0.20, IQA in {ACCEPTABLE,HIGH}, set_size==1, all signals OK). Sub-rules 7a (DEGRADED) and 7b (underpowered) both fail. Sub-rule 7c default fires -> Tier 1.

**S1.2 — Clean septoria prediction**
- v3: probs=[0.04, 0.90, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.02
- LoRA: probs=[0.05, 0.86, 0.03, 0.02, 0.02, 0.02]
- PSV: argmax=1 (septoria), max=0.74, margin=0.48, reliability=0.81
- Classifier: argmax=1, max=0.89, margin=0.83
- Conformal (τ=0.40): set={1}, size=1
- → **Tier 1**, T5 alert: False (rule 7c)
- Walk: all Rule 7 main conditions met (max>=0.85, margin>=0.30, psv_reliability>=0.50, chilli<0.20, IQA in {ACCEPTABLE,HIGH}, set_size==1, all signals OK). Sub-rules 7a (DEGRADED) and 7b (underpowered) both fail. Sub-rule 7c default fires -> Tier 1.

**S1.3 — Clean late_blight prediction (Tier 5 alert fires)**
- v3: probs=[0.02, 0.02, 0.89, 0.01, 0.01, 0.01], chilli_leak=0.04
- LoRA: probs=[0.03, 0.03, 0.87, 0.02, 0.02, 0.03]
- PSV: argmax=2 (late_blight), max=0.78, margin=0.55, reliability=0.74
- Classifier: argmax=2, max=0.92, margin=0.87
- Conformal (τ=0.45): set={2}, size=1
- → **Tier 1**, T5 alert: **True** (rule 7c; T5 fires because argmax in {late_blight, mosaic, ylcv} and max ≥ 0.20)
- Walk: all Rule 7 main conditions met (max>=0.85, margin>=0.30, psv_reliability>=0.50, chilli<0.20, IQA in {ACCEPTABLE,HIGH}, set_size==1, all signals OK). Sub-rules 7a (DEGRADED) and 7b (underpowered) both fail. Sub-rule 7c default fires -> Tier 1.

**S1.4 — Clean YLCV prediction (Tier 5 alert fires)**
- v3: probs=[0.02, 0.02, 0.02, 0.84, 0.02, 0.02], chilli_leak=0.06
- LoRA: probs=[0.03, 0.02, 0.02, 0.85, 0.04, 0.04]
- PSV: argmax=3 (ylcv), max=0.81, margin=0.62, reliability=0.85
- Classifier: argmax=3, max=0.87, margin=0.78
- Conformal (τ=0.42): set={3}, size=1
- → **Tier 1**, T5 alert: **True** (rule 7c; argmax is ylcv → T5)
- Walk: this requires the underpowered guard NOT to fire. If F.0 reports YLCV recall < 0.50, sub-rule 7b downgrades to 3A. This scenario assumes the YLCV recall guard passes (more aspirational than realistic given low YLCV training counts).

**S1.5 — Clean mosaic prediction (Tier 5 alert fires)**
- v3: probs=[0.04, 0.03, 0.02, 0.02, 0.86, 0.01], chilli_leak=0.02
- LoRA: probs=[0.05, 0.03, 0.02, 0.02, 0.84, 0.04]
- PSV: argmax=4 (mosaic), max=0.69, margin=0.42, reliability=0.71
- Classifier: argmax=4, max=0.88, margin=0.81
- Conformal (τ=0.43): set={4}, size=1
- → **Tier 1**, T5 alert: **True** (rule 7c; argmax is mosaic → T5)
- Walk: same underpowered-guard caveat as S1.4 if mosaic is flagged underpowered by F.0.

**S1.6 — Clean healthy prediction**
- v3: probs=[0.01, 0.02, 0.01, 0.02, 0.01, 0.91], chilli_leak=0.02
- LoRA: probs=[0.02, 0.03, 0.02, 0.02, 0.02, 0.89]
- PSV: argmax=5 (healthy), max=0.79, margin=0.58, reliability=0.83
- Classifier: argmax=5, max=0.93, margin=0.88
- Conformal (τ=0.40): set={5}, size=1
- → **Tier 1**, T5 alert: False (rule 7c; healthy is not a dangerous class)
- Walk: all Rule 7 main conditions met (max>=0.85, margin>=0.30, psv_reliability>=0.50, chilli<0.20, IQA in {ACCEPTABLE,HIGH}, set_size==1, all signals OK). Sub-rules 7a (DEGRADED) and 7b (underpowered) both fail. Sub-rule 7c default fires -> Tier 1.

**S1.7 — Foliar with HIGH IQA**
- v3: probs=[0.94, 0.02, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.00
- LoRA: probs=[0.91, 0.03, 0.02, 0.01, 0.01, 0.02]
- PSV: argmax=0, max=0.82, margin=0.65, reliability=0.92
- IQA: HIGH
- Classifier: argmax=0, max=0.96, margin=0.93
- Conformal (τ=0.35): set={0}, size=1
- → **Tier 1**, T5 alert: False (rule 7c)
- Walk: all Rule 7 main conditions met (max>=0.85, margin>=0.30, psv_reliability>=0.50, chilli<0.20, IQA in {ACCEPTABLE,HIGH}, set_size==1, all signals OK). Sub-rules 7a (DEGRADED) and 7b (underpowered) both fail. Sub-rule 7c default fires -> Tier 1.

**S1.8 — Late_blight at exact threshold values**
- v3: probs=[0.02, 0.04, 0.84, 0.02, 0.02, 0.01], chilli_leak=0.05
- LoRA: probs=[0.04, 0.05, 0.80, 0.03, 0.04, 0.04]
- PSV: argmax=2, max=0.65, margin=0.30, reliability=0.50 (exactly at threshold)
- Classifier: argmax=2, max=0.85 (exactly), margin=0.30 (exactly)
- Conformal (τ=0.50): set={2}, size=1
- → **Tier 1**, T5 alert: **True** (rule 7c; all conditions at boundary use ≥, so 0.85 ≥ 0.85 ✓, 0.30 ≥ 0.30 ✓, 0.50 ≥ 0.50 ✓; T5 fires for late_blight)
- Walk: every Tier 1 condition checked at its exact boundary. All `>=` comparisons are inclusive, so all pass.

**S1.9 — Foliar with PSV reliability at lower bound**
- v3: probs=[0.87, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.05
- LoRA: probs=[0.88, 0.05, 0.02, 0.02, 0.02, 0.01]
- PSV: argmax=0, max=0.62, margin=0.32, reliability=0.50 (at lower bound for Tier 1)
- Classifier: argmax=0, max=0.91, margin=0.86
- Conformal (τ=0.42): set={0}, size=1
- → **Tier 1**, T5 alert: False (rule 7c; PSV reliability exactly at 0.50, not < 0.40 so Rule 3 ✗, not >= 0.50 fails so Rule 7 condition met)
- Walk: all Rule 7 main conditions met (max>=0.85, margin>=0.30, psv_reliability>=0.50, chilli<0.20, IQA in {ACCEPTABLE,HIGH}, set_size==1, all signals OK). Sub-rules 7a (DEGRADED) and 7b (underpowered) both fail. Sub-rule 7c default fires -> Tier 1.

**S1.10 — Septoria with chilli_leakage at upper bound for Tier 1**
- v3: probs=[0.05, 0.74, 0.01, 0.01, 0.01, 0.00], chilli_leak=0.18 (just under 0.20 cap for Tier 1)
- LoRA: probs=[0.06, 0.85, 0.02, 0.02, 0.03, 0.02]
- PSV: argmax=1, max=0.71, margin=0.42, reliability=0.74
- Classifier: argmax=1, max=0.86, margin=0.79
- Conformal (τ=0.40): set={1}, size=1
- → **Tier 1**, T5 alert: False (rule 7c; chilli_leakage 0.18 < 0.20 ✓)
- Walk: all Rule 7 main conditions met (max>=0.85, margin>=0.30, psv_reliability>=0.50, chilli<0.20, IQA in {ACCEPTABLE,HIGH}, set_size==1, all signals OK). Sub-rules 7a (DEGRADED) and 7b (underpowered) both fail. Sub-rule 7c default fires -> Tier 1.

**S1.11 — Healthy at exact margin threshold**
- v3: probs=[0.05, 0.05, 0.02, 0.05, 0.02, 0.78], chilli_leak=0.03
- LoRA: probs=[0.06, 0.05, 0.04, 0.04, 0.03, 0.78]
- PSV: argmax=5, max=0.65, margin=0.32, reliability=0.61
- Classifier: argmax=5, max=0.85, margin=0.30 (exactly)
- Conformal (τ=0.45): set={5}, size=1
- → **Tier 1**, T5 alert: False (rule 7c; margin 0.30 ≥ 0.30 ✓, max 0.85 ≥ 0.85 ✓)
- Walk: all Rule 7 main conditions met (max>=0.85, margin>=0.30, psv_reliability>=0.50, chilli<0.20, IQA in {ACCEPTABLE,HIGH}, set_size==1, all signals OK). Sub-rules 7a (DEGRADED) and 7b (underpowered) both fail. Sub-rule 7c default fires -> Tier 1.

**S1.12 — Foliar with high margin from confident agreement**
- v3: probs=[0.96, 0.01, 0.00, 0.00, 0.00, 0.00], chilli_leak=0.03
- LoRA: probs=[0.95, 0.02, 0.01, 0.01, 0.00, 0.01]
- PSV: argmax=0, max=0.85, margin=0.72, reliability=0.94
- Classifier: argmax=0, max=0.97, margin=0.94
- Conformal (τ=0.32): set={0}, size=1
- → **Tier 1**, T5 alert: False (rule 7c)
- Walk: all Rule 7 main conditions met (max>=0.85, margin>=0.30, psv_reliability>=0.50, chilli<0.20, IQA in {ACCEPTABLE,HIGH}, set_size==1, all signals OK). Sub-rules 7a (DEGRADED) and 7b (underpowered) both fail. Sub-rule 7c default fires -> Tier 1.

### 15.4 Tier 2 scenarios — confident prediction

All Tier 2 scenarios share: `prediction_set_size==1`, `0.65 <= combined_max_prob < 0.85`, `combined_margin >= 0.20`, IQA acceptable/high/degraded (handled by sub-rule 8a), all signals succeeded, `psv_reliability >= 0.40`, `chilli_leakage < 0.30`. Sub-rule 8c (default) fires.

**S2.1 — Foliar at 0.70 confidence**
- v3: probs=[0.71, 0.10, 0.05, 0.03, 0.05, 0.04], chilli_leak=0.02
- LoRA: probs=[0.68, 0.12, 0.07, 0.04, 0.05, 0.04]
- PSV: argmax=0, max=0.58, margin=0.28, reliability=0.65
- Classifier: argmax=0, max=0.71, margin=0.45
- Conformal (τ=0.62): set={0}, size=1
- → **Tier 2**, T5 alert: False (rule 8c)
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.

**S2.2 — Septoria at exact lower bound (0.65)**
- v3: probs=[0.10, 0.65, 0.07, 0.05, 0.06, 0.05], chilli_leak=0.02
- LoRA: probs=[0.12, 0.62, 0.08, 0.05, 0.07, 0.06]
- PSV: argmax=1, max=0.55, margin=0.25, reliability=0.58
- Classifier: argmax=1, max=0.65 (exactly), margin=0.42
- Conformal (τ=0.65): set={1}, size=1
- → **Tier 2**, T5 alert: False (rule 8c; max 0.65 ≥ 0.65 ✓)
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.

**S2.3 — Late_blight at 0.75 (Tier 2 + Tier 5)**
- v3: probs=[0.05, 0.05, 0.74, 0.03, 0.05, 0.04], chilli_leak=0.04
- LoRA: probs=[0.06, 0.06, 0.71, 0.04, 0.06, 0.07]
- PSV: argmax=2, max=0.62, margin=0.32, reliability=0.66
- Classifier: argmax=2, max=0.75, margin=0.55
- Conformal (τ=0.55): set={2}, size=1
- → **Tier 2**, T5 alert: **True** (rule 8c; T5 fires for late_blight argmax)
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.

**S2.4 — YLCV at 0.70 (Tier 2 + Tier 5)**
- v3: probs=[0.04, 0.04, 0.04, 0.69, 0.06, 0.05], chilli_leak=0.08
- LoRA: probs=[0.05, 0.05, 0.05, 0.66, 0.10, 0.09]
- PSV: argmax=3, max=0.59, margin=0.30, reliability=0.62
- Classifier: argmax=3, max=0.70, margin=0.46
- Conformal (τ=0.60): set={3}, size=1
- → **Tier 2** (assuming YLCV not flagged underpowered), T5 alert: **True**
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.

**S2.5 — Mosaic at 0.78 (Tier 2 + Tier 5)**
- v3: probs=[0.06, 0.05, 0.04, 0.04, 0.76, 0.04], chilli_leak=0.01
- LoRA: probs=[0.07, 0.06, 0.04, 0.05, 0.74, 0.04]
- PSV: argmax=4, max=0.61, margin=0.31, reliability=0.69
- Classifier: argmax=4, max=0.78, margin=0.62
- Conformal (τ=0.55): set={4}, size=1
- → **Tier 2** (assuming mosaic not flagged underpowered), T5 alert: **True**
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.

**S2.6 — Healthy at 0.72**
- v3: probs=[0.06, 0.06, 0.04, 0.06, 0.04, 0.71], chilli_leak=0.03
- LoRA: probs=[0.07, 0.07, 0.05, 0.06, 0.05, 0.70]
- PSV: argmax=5, max=0.55, margin=0.24, reliability=0.61
- Classifier: argmax=5, max=0.72, margin=0.50
- Conformal (τ=0.62): set={5}, size=1
- → **Tier 2**, T5 alert: False (rule 8c)
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.

**S2.7 — Foliar with PSV reliability at Tier 2 lower bound**
- v3: probs=[0.74, 0.10, 0.04, 0.04, 0.04, 0.02], chilli_leak=0.02
- LoRA: probs=[0.71, 0.11, 0.05, 0.04, 0.05, 0.04]
- PSV: argmax=0, max=0.51, margin=0.22, reliability=0.40 (exactly at Tier 2 lower bound)
- Classifier: argmax=0, max=0.74, margin=0.55
- Conformal (τ=0.58): set={0}, size=1
- → **Tier 2**, T5 alert: False
- Walk: PSV reliability 0.40 — Rule 3 condition is `< 0.40` so Rule 3 ✗. Rule 8 requires `>= 0.40` ✓. Tier 2 fires.

**S2.8 — Septoria with chilli_leakage at Tier 2 upper bound**
- v3: probs=[0.08, 0.52, 0.03, 0.03, 0.03, 0.03], chilli_leak=0.28 (just under Tier 2's 0.30 cap)
- LoRA: probs=[0.12, 0.74, 0.04, 0.04, 0.03, 0.03]
- PSV: argmax=1, max=0.59, margin=0.30, reliability=0.66
- Classifier: argmax=1, max=0.78, margin=0.60
- Conformal (τ=0.55): set={1}, size=1
- → **Tier 2**, T5 alert: False
- Walk: chilli_leakage 0.28 — Rule 3 condition is `> 0.40` so Rule 3 ✗. Rule 8 requires `< 0.30` ✓ (0.28 < 0.30). Tier 2 fires.

**S2.9 — Foliar at exact margin threshold for Tier 2**
- v3: probs=[0.66, 0.19, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
- LoRA: probs=[0.66, 0.22, 0.04, 0.03, 0.03, 0.02]
- PSV: argmax=0, max=0.50, margin=0.20, reliability=0.55
- Classifier: argmax=0, max=0.68, margin=0.20 (exactly)
- Conformal (τ=0.62): set={0}, size=1
- → **Tier 2**, T5 alert: False (rule 8c; margin 0.20 ≥ 0.20 ✓)
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.

**S2.10 — Late_blight at exact max threshold (Tier 2 + Tier 5)**
- v3: probs=[0.10, 0.10, 0.65, 0.05, 0.05, 0.05], chilli_leak=0.00
- LoRA: probs=[0.12, 0.10, 0.62, 0.06, 0.06, 0.04]
- PSV: argmax=2, max=0.51, margin=0.20, reliability=0.49
- Classifier: argmax=2, max=0.65 (exactly), margin=0.40
- Conformal (τ=0.65): set={2}, size=1
- → **Tier 2**, T5 alert: **True** (rule 8c; T5 for late_blight)
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.

**S2.11 — Healthy just under Tier 1 cutoff**
- v3: probs=[0.04, 0.04, 0.02, 0.04, 0.02, 0.83], chilli_leak=0.01
- LoRA: probs=[0.05, 0.05, 0.03, 0.04, 0.03, 0.80]
- PSV: argmax=5, max=0.72, margin=0.50, reliability=0.78
- Classifier: argmax=5, max=0.84, margin=0.78 (max < 0.85 → not Tier 1; margin OK)
- Conformal (τ=0.40): set={5}, size=1
- → **Tier 2**, T5 alert: False (rule 8c; max 0.84 < 0.85 fails Rule 7, but 0.84 ≥ 0.65 passes Rule 8)
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.

**S2.12 — Foliar with high max but tight margin**
- v3: probs=[0.80, 0.10, 0.02, 0.01, 0.01, 0.01], chilli_leak=0.05
- LoRA: probs=[0.83, 0.12, 0.02, 0.01, 0.01, 0.01]
- PSV: argmax=0, max=0.55, margin=0.25, reliability=0.65
- Classifier: argmax=0, max=0.87, margin=0.25 (margin < 0.30 → not Tier 1)
- Conformal (τ=0.40): set={0}, size=1
- → **Tier 2**, T5 alert: False (rule 8c; max 0.87 ≥ 0.85 but margin 0.25 < 0.30 fails Rule 7. Margin 0.25 ≥ 0.20 passes Rule 8.)
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.

### 15.5 Tier 3A scenarios — two-class ambiguity

All Tier 3A scenarios share: `prediction_set_size == 2`. Rule 6 fires.

**S3A.1 — Foliar vs septoria (small-lesion confusion)**
- v3: probs=[0.44, 0.39, 0.05, 0.03, 0.04, 0.03], chilli_leak=0.02
- LoRA: probs=[0.42, 0.38, 0.06, 0.04, 0.05, 0.05]
- PSV: argmax=0, max=0.51, margin=0.18, reliability=0.71
- Classifier: argmax=0, max=0.46, margin=0.04
- Conformal (τ=0.55): set={0, 1}, size=2 (both above 1−τ=0.45)
- → **Tier 3A**, T5 alert: False (rule 6)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)

**S3A.2 — Late_blight vs foliar (Tier 5 fires)**
- v3: probs=[0.40, 0.05, 0.46, 0.02, 0.03, 0.04], chilli_leak=0.00
- LoRA: probs=[0.38, 0.06, 0.43, 0.04, 0.05, 0.04]
- PSV: argmax=2, max=0.49, margin=0.10, reliability=0.62
- Classifier: argmax=2, max=0.45, margin=0.06
- Conformal (τ=0.55): set={0, 2}, size=2
- → **Tier 3A**, T5 alert: **True** (rule 6; T5 fires because late_blight in set with late_blight_prob 0.45 ≥ 0.20)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)

**S3A.3 — YLCV vs healthy (light yellowing ambiguity, Tier 5)**
- v3: probs=[0.04, 0.04, 0.04, 0.42, 0.04, 0.40], chilli_leak=0.02
- LoRA: probs=[0.05, 0.05, 0.05, 0.40, 0.05, 0.40]
- PSV: argmax=3, max=0.46, margin=0.08, reliability=0.69
- Classifier: argmax=3, max=0.42, margin=0.02
- Conformal (τ=0.55): set={3, 5}, size=2
- → **Tier 3A**, T5 alert: **True** (rule 6; argmax YLCV → T5)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)

**S3A.4 — Mosaic vs foliar (patchy lesion ambiguity, Tier 5)**
- v3: probs=[0.41, 0.04, 0.03, 0.03, 0.46, 0.03], chilli_leak=0.00
- LoRA: probs=[0.39, 0.05, 0.04, 0.04, 0.43, 0.05]
- PSV: argmax=4, max=0.48, margin=0.12, reliability=0.66
- Classifier: argmax=4, max=0.45, margin=0.04
- Conformal (τ=0.55): set={0, 4}, size=2
- → **Tier 3A**, T5 alert: **True** (rule 6; argmax mosaic → T5)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)

**S3A.5 — Healthy vs OOD (uncertain whether image is even a tomato)**
- v3: probs=[0.10, 0.05, 0.05, 0.05, 0.05, 0.55], chilli_leak=0.15 (somewhat elevated)
- LoRA: probs=[0.10, 0.05, 0.05, 0.05, 0.05, 0.70]
- PSV: argmax=5, max=0.40, margin=0.05, reliability=0.45 (poor PSV — borderline 3C check)
- Classifier: argmax=5, max=0.45, margin=0.05; OOD probability 0.40
- Conformal (τ=0.60): set={5, 6}, size=2
- → **Tier 3A**, T5 alert: False (rule 6; PSV reliability 0.45 ≥ 0.40 doesn't trigger Rule 3)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)

**S3A.6 — Late_blight vs mosaic (T5 fires for both, but only argmax counted in flag setting)**
- v3: probs=[0.04, 0.04, 0.46, 0.04, 0.40, 0.02], chilli_leak=0.00
- LoRA: probs=[0.05, 0.05, 0.42, 0.05, 0.39, 0.04]
- PSV: argmax=2, max=0.55, margin=0.18, reliability=0.71
- Classifier: argmax=2, max=0.44, margin=0.04
- Conformal (τ=0.55): set={2, 4}, size=2
- → **Tier 3A**, T5 alert: **True** (rule 6; late_blight argmax + late_blight in set ≥ 0.20 + mosaic in set is irrelevant since mosaic only fires T5 on argmax)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)

**S3A.7 — Septoria vs late_blight (small-vs-large lesion ambiguity, T5)**
- v3: probs=[0.05, 0.46, 0.40, 0.03, 0.03, 0.03], chilli_leak=0.00
- LoRA: probs=[0.06, 0.43, 0.38, 0.04, 0.04, 0.05]
- PSV: argmax=1, max=0.50, margin=0.15, reliability=0.74
- Classifier: argmax=1, max=0.45, margin=0.05
- Conformal (τ=0.55): set={1, 2}, size=2
- → **Tier 3A**, T5 alert: **True** (rule 6; late_blight in set with prob ≥ 0.20)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)

**S3A.8 — YLCV vs mosaic (both viral, T5 only via argmax YLCV)**
- v3: probs=[0.04, 0.04, 0.03, 0.45, 0.40, 0.04], chilli_leak=0.00
- LoRA: probs=[0.05, 0.05, 0.04, 0.43, 0.39, 0.04]
- PSV: argmax=3, max=0.51, margin=0.16, reliability=0.69
- Classifier: argmax=3, max=0.44, margin=0.04
- Conformal (τ=0.55): set={3, 4}, size=2
- → **Tier 3A**, T5 alert: **True** (rule 6; YLCV argmax → T5; mosaic in set does NOT trigger T5 because mosaic only fires on argmax per Section 14.3)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)

**S3A.9 — Foliar vs healthy at boundary**
- v3: probs=[0.42, 0.04, 0.02, 0.02, 0.02, 0.43], chilli_leak=0.05
- LoRA: probs=[0.40, 0.05, 0.04, 0.04, 0.04, 0.43]
- PSV: argmax=5, max=0.49, margin=0.12, reliability=0.65
- Classifier: argmax=5, max=0.43, margin=0.02
- Conformal (τ=0.55): set={0, 5}, size=2
- → **Tier 3A**, T5 alert: False (rule 6; neither argmax nor in_set classes are dangerous)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)

**S3A.10 — Confident enough for Tier 1 conditions but set size 2 (Rule 6 wins)**
- v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
- LoRA: probs=[0.50, 0.40, 0.04, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.78, margin=0.55, reliability=0.85
- Classifier: argmax=0, max=0.86, margin=0.45 (would-be Tier 1 conditions met)
- Conformal (τ=0.65): set={0, 1}, size=2 (set has 2 because LoRA's mass on septoria + classifier's residual mass on septoria push septoria above 1−τ=0.35)
- → **Tier 3A**, T5 alert: False
- Walk: Rule 6 (set size == 2) fires before Rule 7. Even though Rule 7's confidence conditions are met, set size 2 means Rule 6 takes priority. The classifier's calibrated output spreads enough probability across {foliar, septoria} that conformal's threshold puts both in the set.

**S3A.11 — Definitive single-class but argmax is underpowered class**
- v3: probs=[0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00
- LoRA: probs=[0.05, 0.05, 0.04, 0.81, 0.02, 0.03]
- PSV: argmax=3, max=0.74, margin=0.50, reliability=0.78
- Classifier: argmax=3 (YLCV), max=0.88, margin=0.82
- Conformal (τ=0.40): set={3}, size=1 (would-be Tier 1)
- Underpowered: YLCV is flagged underpowered by F.0 (recall < 0.50)
- → **Tier 3A** (downgrade), T5 alert: **True** (rule 7b — sub-rule fires due to underpowered class; T5 still fires)
- Walk: Rule 7's main conditions met; sub-rule 7a (DEGRADED IQA) ✗; sub-rule 7b (underpowered class) ✓ → Tier 3A. T5 evaluated independently → fires.

**S3A.12 — Confident single-class but argmax is underpowered**
- v3: probs=[0.04, 0.04, 0.04, 0.05, 0.74, 0.05], chilli_leak=0.04
- LoRA: probs=[0.05, 0.05, 0.05, 0.06, 0.71, 0.08]
- PSV: argmax=4, max=0.62, margin=0.32, reliability=0.69
- Classifier: argmax=4 (mosaic), max=0.72, margin=0.55
- Conformal (τ=0.55): set={4}, size=1
- Underpowered: mosaic is flagged underpowered
- → **Tier 3A** (downgrade), T5 alert: **True** (rule 8b; mosaic argmax → T5)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)

### 15.6 Tier 3B scenarios — multi-class ambiguity

All Tier 3B scenarios share: `prediction_set_size >= 3` AND `combined_max_prob >= 0.45` (else Rule 4 fires first). Rule 5 fires.

**S3B.1 — Three small-lesion classes, late_blight in set fires T5**
- v3: probs=[0.45, 0.30, 0.20, 0.02, 0.02, 0.01], chilli_leak=0.00
- LoRA: probs=[0.42, 0.32, 0.18, 0.03, 0.03, 0.02]
- PSV: argmax=0, max=0.55, margin=0.18, reliability=0.71
- IQA: ACCEPTABLE
- Classifier: argmax=0, max=0.46, margin=0.16
- Conformal (τ=0.55): set={0, 1, 2}, size=3 (top three above threshold 0.45)
- → **Tier 3B**, T5 alert: **True** (rule 5; late_blight in set with prob 0.20 ≥ 0.20)
- Walk: Rules 1-4 don't fire. prediction_set_size>=3 -> Rule 5 fires -> Tier 3B.

**S3B.2 — Four-class spread including healthy**
- v3: probs=[0.33, 0.29, 0.06, 0.04, 0.04, 0.21], chilli_leak=0.03
- LoRA: probs=[0.30, 0.30, 0.07, 0.05, 0.05, 0.23]
- PSV: argmax=0, max=0.41, margin=0.05, reliability=0.55
- IQA: ACCEPTABLE
- Classifier: P_final_calibrated=[0.34, 0.30, 0.06, 0.04, 0.04, 0.22, 0.00], argmax=0, max=0.46, margin=0.16
- Conformal (τ=0.78): threshold 0.22; set={0 (0.34), 1 (0.30), 5 (0.22)}, size=3
- → **Tier 3B**, T5 alert: False (rule 5; no dangerous class in set or argmax)
- Walk: Rules 1-4 don't fire. prediction_set_size>=3 -> Rule 5 fires -> Tier 3B.

**S3B.3 — Three-class set with late_blight admitted (genuine T5 case)**
- v3: probs=[0.30, 0.22, 0.20, 0.10, 0.10, 0.05], chilli_leak=0.03
- LoRA: probs=[0.28, 0.22, 0.20, 0.12, 0.12, 0.06]
- PSV: argmax=0, max=0.46, margin=0.10, reliability=0.58
- IQA: ACCEPTABLE
- Classifier: P_final_calibrated=[0.46, 0.22, 0.20, 0.05, 0.04, 0.02, 0.01], argmax=0, max=0.46, margin=0.24
- Conformal (τ=0.83): threshold 0.17; set={0, 1, 2}, size=3
- → **Tier 3B**, T5 alert: **True** (rule 5; late_blight in set with prob 0.20 ≥ 0.20)
- Walk: Rules 1-4 don't fire. prediction_set_size>=3 -> Rule 5 fires -> Tier 3B.
- Note: this scenario was originally framed as a 5-class set, but realistic τ values from F.0 calibration produce sets of 1-3 classes. Sets of 4+ classes require either flat distributions (which trigger Rule 4 because max < 0.45) or unrealistic τ values. The scenario settles at 3 classes for realism.

**S3B.4 — Extreme uncertainty (degenerate case routes to 4A, not 3B)**
- Classifier: P_final_calibrated=[0.16, 0.15, 0.15, 0.14, 0.14, 0.13, 0.13]
- max=0.16 → Rule 4 fires (max < 0.45) → **Tier 4A**
- Walk: max<0.45 -> Rule 4 fires -> Tier 4A. (Or empty set -> Rule 5 sub-rule, or no rule conditions met -> Rule 9 catch-all.)
- T5: argmax=0 (foliar, not dangerous), late_blight prob 0.15 < 0.20 → T5 = False
- → **Tier 4A**, T5 alert: False
- Note: included to show that flat-distribution cases route to 4A, not 3B. Tier 3B requires both `set_size >= 3` AND `max_prob >= 0.45`. Practical "all classes uncertain" cases always trigger Rule 4 first.

**S3B.5 — Three classes including late_blight argmax (T5 fires via both bullets)**
- v3: probs=[0.20, 0.05, 0.45, 0.05, 0.05, 0.05], chilli_leak=0.15
- LoRA: probs=[0.22, 0.06, 0.43, 0.05, 0.06, 0.18]
- PSV: argmax=2, max=0.55, margin=0.20, reliability=0.66
- IQA: ACCEPTABLE
- Classifier: P_final_calibrated=[0.20, 0.05, 0.45, 0.05, 0.05, 0.15, 0.05], argmax=2, max=0.45, margin=0.25
- Conformal (τ=0.85): threshold 0.15; set={0, 2, 5}, size=3
- → **Tier 3B**, T5 alert: **True** (rule 5; late_blight argmax + late_blight in set both fire T5)
- Walk: max=0.45 — Rule 4 condition `< 0.45` fails (0.45 is not strictly less than 0.45). Rule 5 (size >= 3) fires next → Tier 3B.

**S3B.6 — Three classes with mosaic argmax + YLCV in set**
- v3: probs=[0.20, 0.05, 0.05, 0.20, 0.45, 0.05], chilli_leak=0.00
- LoRA: probs=[0.22, 0.06, 0.05, 0.20, 0.42, 0.05]
- PSV: argmax=4, max=0.51, margin=0.15, reliability=0.65
- IQA: ACCEPTABLE
- Classifier: P_final=[0.20, 0.05, 0.05, 0.20, 0.45, 0.04, 0.01], argmax=4, max=0.45, margin=0.25
- Conformal (τ=0.85): threshold 0.15; set={0, 3, 4}, size=3
- → **Tier 3B**, T5 alert: **True** (rule 5; mosaic argmax fires T5 first bullet; YLCV in set does not trigger T5 because YLCV's T5 trigger requires it to be argmax)
- Walk: Rules 1-4 don't fire. prediction_set_size>=3 -> Rule 5 fires -> Tier 3B.

**S3B.7 — Three small-lesion classes with foliar argmax, late_blight in set fires T5**
- v3: probs=[0.45, 0.20, 0.20, 0.04, 0.05, 0.05], chilli_leak=0.01
- LoRA: probs=[0.46, 0.21, 0.18, 0.04, 0.06, 0.05]
- PSV: argmax=0, max=0.55, margin=0.20, reliability=0.65
- IQA: ACCEPTABLE
- Classifier: P_final=[0.46, 0.20, 0.20, 0.04, 0.05, 0.04, 0.01], argmax=0, max=0.46, margin=0.26
- Conformal (τ=0.81): threshold 0.19; set={0, 1, 2}, size=3
- → **Tier 3B**, T5 alert: **True** (rule 5; late_blight in set with 0.20 ≥ 0.20)
- Walk: Rules 1-4 don't fire. prediction_set_size>=3 -> Rule 5 fires -> Tier 3B.

**S3B.8 — Three classes with healthy argmax, no T5**
- v3: probs=[0.20, 0.18, 0.04, 0.05, 0.04, 0.45], chilli_leak=0.04
- LoRA: probs=[0.22, 0.20, 0.05, 0.05, 0.04, 0.44]
- PSV: argmax=5, max=0.51, margin=0.18, reliability=0.69
- IQA: ACCEPTABLE
- Classifier: P_final=[0.20, 0.18, 0.04, 0.05, 0.04, 0.45, 0.04], argmax=5, max=0.45, margin=0.25
- Conformal (τ=0.83): threshold 0.17; set={0, 1, 5}, size=3
- → **Tier 3B**, T5 alert: False (rule 5; no dangerous class in set or argmax)
- Walk: Rules 1-4 don't fire. prediction_set_size>=3 -> Rule 5 fires -> Tier 3B.

**S3B.9 — Three classes with septoria argmax + IQA DEGRADED (3B sticks; 3D doesn't apply)**
- v3: probs=[0.20, 0.45, 0.20, 0.04, 0.04, 0.04], chilli_leak=0.03
- LoRA: probs=[0.22, 0.46, 0.18, 0.04, 0.05, 0.05]
- PSV: argmax=1, max=0.50, margin=0.20, reliability=0.55
- IQA: **DEGRADED**
- Classifier: P_final=[0.20, 0.46, 0.20, 0.04, 0.04, 0.04, 0.02], argmax=1, max=0.46, margin=0.26
- Conformal (τ=0.83): threshold 0.17; set={0, 1, 2}, size=3
- → **Tier 3B**, T5 alert: **True** (rule 5; late_blight in set with 0.20 ≥ 0.20)
- Walk: Rule 5 fires before Rule 7/8 with sub-rule 7a/8a. The DEGRADED IQA cap (Tier 3D) never gets a chance to fire because Rule 5 takes higher priority. Tier 3B is already at the Tier 3 ceiling, so the DEGRADED cap has no further effect.

**S3B.10 — Borderline: τ admits exactly 3 classes**
- Classifier: P_final=[0.50, 0.30, 0.13, 0.03, 0.02, 0.01, 0.01], argmax=0, max=0.50, margin=0.20
- Conformal (τ=0.87): threshold 0.13; set={0, 1, 2}, size=3 (third class at exactly 0.13, in if `>=`)
- → **Tier 3B**, T5 alert: False (rule 5; late_blight prob 0.13 < 0.20 fails T5 in_set bullet; argmax foliar fails first T5 bullet)
- Walk: Rules 1-4 don't fire. prediction_set_size>=3 -> Rule 5 fires -> Tier 3B.

### 15.7 Tier 3C scenarios — PSV unreliable / chilli leakage

All Tier 3C scenarios share: Rule 3 fires due to `psv_reliability < 0.40` OR `chilli_leakage > 0.40`.

**S3C.1 — PSV reliability just under threshold**
- v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
- LoRA: probs=[0.82, 0.06, 0.05, 0.03, 0.02, 0.02]
- PSV: argmax=0, max=0.45, margin=0.10, **reliability=0.39** (just below 0.40)
- Classifier: argmax=0, max=0.85, margin=0.78
- Conformal (τ=0.40): set={0}, size=1 (would-be Tier 1)
- → **Tier 3C**, T5 alert: False (rule 3; PSV unreliable trumps Tier 1 conditions)
- Walk: Rule 3 fires (psv_reliability<0.40 OR chilli_leakage>0.40) -> Tier 3C.

**S3C.2 — Severe PSV unreliability**
- v3: probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03
- LoRA: probs=[0.88, 0.05, 0.03, 0.02, 0.01, 0.01]
- PSV: argmax=0, max=0.40, margin=0.05, **reliability=0.10** (very low; segmentation likely failed but didn't throw)
- Classifier: argmax=0, max=0.91, margin=0.86
- Conformal (τ=0.40): set={0}, size=1
- → **Tier 3C**, T5 alert: False (rule 3)
- Walk: Rule 3 fires (psv_reliability<0.40 OR chilli_leakage>0.40) -> Tier 3C.

**S3C.3 — Chilli leakage just over threshold**
- v3: probs=[0.50, 0.04, 0.02, 0.02, 0.02, 0.02], **chilli_leak=0.41** (just over 0.40)
- LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.65, margin=0.30, reliability=0.71
- Classifier: argmax=0, max=0.78, margin=0.65
- Conformal (τ=0.50): set={0}, size=1
- → **Tier 3C**, T5 alert: False (rule 3; chilli leakage triggers)
- Walk: Rule 3 fires (psv_reliability<0.40 OR chilli_leakage>0.40) -> Tier 3C.

**S3C.4 — Very high chilli leakage (probably actually a chilli)**
- v3: probs=[0.10, 0.02, 0.02, 0.02, 0.02, 0.02], **chilli_leak=0.80** (very high; image is likely a chilli plant)
- LoRA: probs=[0.55, 0.10, 0.10, 0.10, 0.10, 0.05] (LoRA, being tomato-only, has no equivalent of leakage and forces probability into tomato classes)
- PSV: argmax=0, max=0.40, margin=0.10, reliability=0.55
- Classifier: argmax=0, max=0.50, margin=0.30; OOD prob 0.30
- Conformal (τ=0.65): set={0, 6}, size=2 (foliar and OOD both in set due to spread)
- → **Tier 3C**, T5 alert: False
- Walk: Rule 3 fires (chilli_leak > 0.40) → Tier 3C. The set being size 2 doesn't matter because Rule 3 has higher priority than Rule 6.

**S3C.5 — Both PSV unreliable AND chilli leakage**
- v3: probs=[0.50, 0.04, 0.02, 0.02, 0.02, 0.02], **chilli_leak=0.45**
- LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.30, margin=0.05, **reliability=0.30**
- Classifier: argmax=0, max=0.78, margin=0.65
- Conformal (τ=0.50): set={0}, size=1
- → **Tier 3C**, T5 alert: False (rule 3; both subconditions fire — Rule 3's `OR` accepts either)
- Walk: Rule 3 fires (psv_reliability<0.40 OR chilli_leakage>0.40) -> Tier 3C.
- Note: the structured reasons should distinguish "both fired" from "only one fired" for monitoring (Section 25 use case).

**S3C.6 — PSV unreliable but otherwise definitive**
- v3: probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03
- LoRA: probs=[0.89, 0.04, 0.03, 0.01, 0.01, 0.02]
- PSV: argmax=0, max=0.40, margin=0.05, **reliability=0.35**
- Classifier: argmax=0, max=0.93, margin=0.89
- Conformal (τ=0.35): set={0}, size=1 (would-be Tier 1 if PSV were reliable)
- → **Tier 3C**, T5 alert: False (rule 3)
- Walk: Rule 3 fires (psv_reliability<0.40 OR chilli_leakage>0.40) -> Tier 3C.

**S3C.7 — PSV unreliable but late_blight detected (T5 still fires)**
- v3: probs=[0.05, 0.05, 0.81, 0.02, 0.02, 0.01], chilli_leak=0.04
- LoRA: probs=[0.06, 0.06, 0.81, 0.03, 0.02, 0.02]
- PSV: argmax=2, max=0.30, margin=0.05, **reliability=0.32**
- Classifier: argmax=2, max=0.88, margin=0.83
- Conformal (τ=0.40): set={2}, size=1
- → **Tier 3C**, T5 alert: **True** (rule 3 sets tier; T5 still evaluated independently → fires for late_blight argmax)
- Walk: Rule 3 fires (psv_reliability<0.40 OR chilli_leakage>0.40) -> Tier 3C.

**S3C.8 — PSV reliability at exactly 0.40 (NOT 3C)**
- v3: probs=[0.87, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.05
- LoRA: probs=[0.89, 0.04, 0.03, 0.02, 0.01, 0.01]
- PSV: argmax=0, max=0.45, margin=0.08, **reliability=0.40 exactly**
- Classifier: argmax=0, max=0.91, margin=0.86
- Conformal (τ=0.40): set={0}, size=1
- → **Tier 2**, T5 alert: False
- Walk: Rule 3 condition is `< 0.40`; at 0.40 exactly, Rule 3 ✗. Continues. Rule 7 needs `>= 0.50` ✗. Rule 8 needs `>= 0.40` ✓. Tier 2 fires.

**S3C.9 — Chilli leakage at exactly 0.40 (NOT 3C)**
- v3: probs=[0.55, 0.04, 0.01, 0.00, 0.00, 0.00], chilli_leak=0.40 exactly
- LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.65, margin=0.30, reliability=0.74
- Classifier: argmax=0, max=0.82, margin=0.71
- Conformal (τ=0.45): set={0}, size=1
- → **Tier 4A**, T5 alert: False
- Walk: Rule 3 condition is `> 0.40`; at 0.40 exactly, Rule 3 ✗. Rule 7 needs `< 0.20` ✗. Rule 8 needs `< 0.30` ✗. Set size==1, but Rules 7 and 8 fail on chilli_leakage. Falls through to Rule 9 (catch-all) → Tier 4A.
- Note: this scenario shows a real boundary "trap": chilli_leakage exactly at 0.40 is excluded from 3C but also excluded from 1/2 due to their tighter caps. Falls to catch-all 4A. The system logs a warning per Section 14.5 because Rule 9 firing usually indicates a logic edge case worth review.

**S3C.10 — PSV unreliable due to mask disagreement**
- v3: probs=[0.81, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.04
- LoRA: probs=[0.83, 0.06, 0.05, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.50, margin=0.18, **reliability=0.32** (low because IoU between PSV mask and IQA mask was 0.25, suggesting PSV segmented something different)
- Classifier: argmax=0, max=0.84, margin=0.78
- Conformal (τ=0.42): set={0}, size=1
- → **Tier 3C**, T5 alert: False (rule 3)
- Walk: Rule 3 fires (psv_reliability<0.40 OR chilli_leakage>0.40) -> Tier 3C.

**S3C.11 — PSV unreliable due to coverage > 90%**
- v3: probs=[0.05, 0.05, 0.80, 0.02, 0.02, 0.01], chilli_leak=0.05
- LoRA: probs=[0.06, 0.05, 0.83, 0.02, 0.02, 0.02]
- PSV: argmax=2, max=0.45, margin=0.10, **reliability=0.20** (low because disease coverage on the leaf was 95% — probably segmentation failure mistaking background for leaf)
- Classifier: argmax=2, max=0.86, margin=0.80
- Conformal (τ=0.40): set={2}, size=1
- → **Tier 3C**, T5 alert: **True** (rule 3 sets tier; T5 fires for late_blight argmax)
- Walk: Rule 3 fires (psv_reliability<0.40 OR chilli_leakage>0.40) -> Tier 3C.

**S3C.12 — Chilli leakage at exact Tier 2 boundary with late_blight argmax**
- v3: probs=[0.05, 0.05, 0.55, 0.02, 0.02, 0.01], chilli_leak=0.30
- LoRA: probs=[0.06, 0.05, 0.83, 0.02, 0.02, 0.02]
- PSV: argmax=2, max=0.55, margin=0.20, reliability=0.62
- IQA: ACCEPTABLE
- Classifier: argmax=2, max=0.78, margin=0.65
- Conformal (τ=0.50): set={2}, size=1
- → **Tier 4A** (Rule 9 catch-all), T5 alert: **True** (T5 still fires for late_blight argmax)
- Walk: chilli_leak = 0.30 exactly. Rule 3 condition `> 0.40` ✗. Rule 8 condition `< 0.30` ✗ (0.30 is not strictly less than 0.30). Rule 7 condition `< 0.20` ✗. Falls to Rule 9 → Tier 4A.
- Note: this is one of the boundary "trap" scenarios. chilli_leakage = 0.30 exactly is excluded from Rule 3 (which fires only above 0.40), and also fails Tier 1's < 0.20 cap and Tier 2's < 0.30 cap. The catch-all Rule 9 routes it to 4A while T5 still fires. The structured reasons log "rule_id_fired: catch_all" so monitoring can detect when these boundary cases occur in production. Despite this scenario living in subsection 15.7 (Tier 3C), the actual outcome is Tier 4A — illustrating that "PSV / chilli leakage related scenarios" don't always end at Tier 3C.

### 15.8 Tier 3D scenarios — DEGRADED IQA cap

All Tier 3D scenarios share: would-be Tier 1 or Tier 2, but IQA decision is `DEGRADED`. Sub-rule 7a or 8a fires.

**S3D.1 — Would-be Tier 1 → 3D**
- v3: probs=[0.89, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03
- LoRA: probs=[0.88, 0.05, 0.02, 0.02, 0.02, 0.01]
- PSV: argmax=0, max=0.71, margin=0.45, reliability=0.78
- IQA: **DEGRADED** (e.g., motion blur)
- Classifier: argmax=0, max=0.91, margin=0.86
- Conformal (τ=0.40): set={0}, size=1
- → **Tier 3D**, T5 alert: False (rule 7a; would have been Tier 1 except for IQA)
- Walk: Rule 7 or Rule 8 main IF met. Sub-rule 7a or 8a (IQA DEGRADED) fires -> Tier 3D.

**S3D.2 — Would-be Tier 2 → 3D**
- v3: probs=[0.71, 0.10, 0.05, 0.03, 0.05, 0.04], chilli_leak=0.02
- LoRA: probs=[0.68, 0.12, 0.07, 0.04, 0.05, 0.04]
- PSV: argmax=0, max=0.58, margin=0.28, reliability=0.65
- IQA: **DEGRADED**
- Classifier: argmax=0, max=0.71, margin=0.45
- Conformal (τ=0.62): set={0}, size=1
- → **Tier 3D**, T5 alert: False (rule 8a)
- Walk: Rule 7 or Rule 8 main IF met. Sub-rule 7a or 8a (IQA DEGRADED) fires -> Tier 3D.

**S3D.3 — Tier 3D for late_blight (T5 fires)**
- v3: probs=[0.04, 0.04, 0.82, 0.02, 0.02, 0.02], chilli_leak=0.04
- LoRA: probs=[0.05, 0.05, 0.83, 0.02, 0.02, 0.03]
- PSV: argmax=2, max=0.65, margin=0.32, reliability=0.71
- IQA: **DEGRADED**
- Classifier: argmax=2, max=0.89, margin=0.83
- Conformal (τ=0.42): set={2}, size=1
- → **Tier 3D**, T5 alert: **True** (rule 7a; T5 fires independently for late_blight argmax)
- Walk: Rule 7 or Rule 8 main IF met. Sub-rule 7a or 8a (IQA DEGRADED) fires -> Tier 3D.

**S3D.4 — Tier 3D for YLCV (T5 fires)**
- v3: probs=[0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00
- LoRA: probs=[0.05, 0.05, 0.04, 0.81, 0.02, 0.03]
- PSV: argmax=3, max=0.69, margin=0.42, reliability=0.78
- IQA: **DEGRADED**
- Classifier: argmax=3, max=0.88, margin=0.82
- Conformal (τ=0.40): set={3}, size=1
- → **Tier 3D**, T5 alert: **True** (rule 7a; YLCV argmax → T5)
- Walk: Rule 7 or Rule 8 main IF met. Sub-rule 7a or 8a (IQA DEGRADED) fires -> Tier 3D.

**S3D.5 — Already Tier 3A, IQA DEGRADED — stays 3A (3D doesn't apply)**
- v3: probs=[0.44, 0.39, 0.05, 0.03, 0.04, 0.03], chilli_leak=0.02
- LoRA: probs=[0.42, 0.38, 0.06, 0.04, 0.05, 0.05]
- PSV: argmax=0, max=0.51, margin=0.18, reliability=0.71
- IQA: **DEGRADED**
- Classifier: argmax=0, max=0.46, margin=0.04
- Conformal (τ=0.55): set={0, 1}, size=2
- → **Tier 3A**, T5 alert: False (rule 6; Rule 6 fires before Rule 7/8, so the IQA cap from sub-rule 7a/8a never applies)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)
- Note: Tier 3D is specifically a downgrade FROM 1/2 due to IQA. Tier 3A and 3B are already at Tier 3 and don't need further downgrade.

**S3D.6 — Would-be Tier 1 with PSV at boundary**
- v3: probs=[0.90, 0.03, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.03
- LoRA: probs=[0.90, 0.04, 0.02, 0.02, 0.01, 0.01]
- PSV: argmax=0, max=0.62, margin=0.32, reliability=**0.50** (Tier 1 lower bound)
- IQA: **DEGRADED**
- Classifier: argmax=0, max=0.92, margin=0.87
- Conformal (τ=0.40): set={0}, size=1
- → **Tier 3D**, T5 alert: False (rule 7a; would have been Tier 1 but DEGRADED caps)
- Walk: Rule 7 or Rule 8 main IF met. Sub-rule 7a or 8a (IQA DEGRADED) fires -> Tier 3D.

**S3D.7 — DEGRADED IQA + multi-class set (still 3B)**
- v3: probs=[0.45, 0.30, 0.20, 0.02, 0.02, 0.01], chilli_leak=0.00
- LoRA: probs=[0.42, 0.32, 0.18, 0.03, 0.03, 0.02]
- PSV: argmax=0, max=0.55, margin=0.18, reliability=0.71
- IQA: **DEGRADED**
- Classifier: argmax=0, max=0.46, margin=0.14
- Conformal (τ=0.55): set={0, 1, 2}, size=3
- → **Tier 3B**, T5 alert: **True** (rule 5; Rule 5 fires before Rule 7/8; IQA DEGRADED has no effect on 3B; T5 fires for late_blight in set)
- Walk: Rules 1-4 don't fire. prediction_set_size>=3 -> Rule 5 fires -> Tier 3B.

**S3D.8 — Would-be Tier 1 healthy → 3D**
- v3: probs=[0.02, 0.02, 0.01, 0.02, 0.01, 0.90], chilli_leak=0.02
- LoRA: probs=[0.03, 0.03, 0.02, 0.02, 0.02, 0.88]
- PSV: argmax=5, max=0.79, margin=0.55, reliability=0.83
- IQA: **DEGRADED**
- Classifier: argmax=5, max=0.92, margin=0.88
- Conformal (τ=0.40): set={5}, size=1
- → **Tier 3D**, T5 alert: False (rule 7a; healthy argmax doesn't trigger T5)
- Walk: Rule 7 or Rule 8 main IF met. Sub-rule 7a or 8a (IQA DEGRADED) fires -> Tier 3D.

**S3D.9 — IQA at the threshold between DEGRADED and ACCEPTABLE**
- IQA reports `aggregate_score = 0.55`. Section 6.4 thresholds (placeholder): DEGRADED = [0.40, 0.65), ACCEPTABLE = [0.65, 0.85), HIGH = [0.85, 1.0]. So 0.55 → DEGRADED.
- v3, LoRA, PSV, Classifier: as in S3D.1
- → **Tier 3D**, T5 alert: False (rule 7a)
- Walk: Rule 7 or Rule 8 main IF met. Sub-rule 7a or 8a (IQA DEGRADED) fires -> Tier 3D.

**S3D.10 — DEGRADED IQA + underpowered class (sub-rule 7a wins over 7b)**
- v3: probs=[0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00
- LoRA: probs=[0.05, 0.05, 0.04, 0.81, 0.02, 0.03]
- PSV: argmax=3, max=0.74, margin=0.50, reliability=0.78
- IQA: **DEGRADED**
- Classifier: argmax=3 (YLCV, underpowered), max=0.88, margin=0.82
- Conformal (τ=0.40): set={3}, size=1
- → **Tier 3D**, T5 alert: **True** (rule 7a wins over 7b in priority order; YLCV argmax → T5)
- Walk: Rule 7 main conditions met. Sub-rule 7a (DEGRADED IQA) ✓ → Tier 3D. Sub-rule 7b never evaluated.
- Note: this is the design choice from Section 14.5: 7a (IQA) precedes 7b (underpowered) because IQA is user-actionable (retake the photo).

### 15.9 Tier 4A scenarios — low confidence

All Tier 4A scenarios share: `combined_max_prob < 0.45`. Rule 4 fires.

**S4A.1 — Highly uncertain across all classes**
- v3: probs=[0.18, 0.17, 0.15, 0.13, 0.12, 0.05], chilli_leak=0.20
- LoRA: probs=[0.20, 0.18, 0.15, 0.15, 0.14, 0.18]
- PSV: argmax=0, max=0.30, margin=0.05, reliability=0.55
- Classifier: argmax=0, max=0.21, margin=0.03
- Conformal (τ=0.85, threshold 0.15): set={0, 1, 2, 3, 4}, size=5
- → **Tier 4A**, T5 alert: False (rule 4; max < 0.45; T5 fails because no class has prob ≥ 0.20)
- Walk: max<0.45 -> Rule 4 fires -> Tier 4A. (Or empty set -> Rule 5 sub-rule, or no rule conditions met -> Rule 9 catch-all.)

**S4A.2 — Max prob just under threshold**
- v3: probs=[0.45, 0.20, 0.10, 0.10, 0.07, 0.06], chilli_leak=0.02
- LoRA: probs=[0.42, 0.22, 0.12, 0.10, 0.08, 0.06]
- PSV: argmax=0, max=0.40, margin=0.05, reliability=0.55
- Classifier: argmax=0, max=0.44, margin=0.21
- Conformal (τ=0.65): set={0}, size=1
- → **Tier 4A**, T5 alert: False (rule 4; 0.44 < 0.45)
- Walk: even with set size 1, Rule 4 fires before Rule 7/8 because 4 has higher priority than 7/8.

**S4A.3 — Single-class set with very low confidence**
- v3: probs=[0.30, 0.10, 0.05, 0.05, 0.05, 0.05], chilli_leak=0.40 (high; just at boundary)
- LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.40, margin=0.10, reliability=0.55
- Classifier: argmax=0, max=0.30, margin=0.18
- Conformal (τ=0.74, threshold 0.26): set={0}, size=1
- → **Tier 4A**, T5 alert: False
- Walk: chilli_leak 0.40 exactly fails Rule 3 (`> 0.40` strict). max 0.30 < 0.45 → Rule 4 fires → Tier 4A.

**S4A.4 — Max 0.40 with multi-class set**
- v3: probs=[0.40, 0.30, 0.10, 0.05, 0.05, 0.05], chilli_leak=0.05
- LoRA: probs=[0.42, 0.31, 0.10, 0.05, 0.06, 0.06]
- PSV: argmax=0, max=0.50, margin=0.18, reliability=0.65
- Classifier: argmax=0, max=0.40, margin=0.10
- Conformal (τ=0.65): set={0, 1}, size=2 (would-be 3A)
- → **Tier 4A**, T5 alert: False
- Walk: Rule 4 (max < 0.45) fires before Rule 6 (set size == 2). Tier 4A wins.
- Note: this priority can be debated — should "low confidence with multi-class set" be 4A (low confidence) or 3A (ambiguous)? The current rule chain picks 4A. Section 30 (limitations) lists this as an intentional design choice favoring "we don't know" over "between two options."

**S4A.5 — Empty set with low max (Rule 4 catches first)**
- v3: probs=[0.38, 0.19, 0.10, 0.10, 0.10, 0.08], chilli_leak=0.05
- LoRA: probs=[0.40, 0.22, 0.12, 0.10, 0.10, 0.06]
- PSV: argmax=0, max=0.40, margin=0.10, reliability=0.55
- IQA: ACCEPTABLE
- Classifier: P_final_calibrated=[0.42, 0.20, 0.10, 0.10, 0.10, 0.05, 0.03], max=0.42, margin=0.22
- Conformal (τ=0.55): threshold 0.45; no class above → set={}, size=0
- → **Tier 4A**, T5 alert: False
- Walk: Rule 4 (max 0.42 < 0.45) fires first → Tier 4A. The empty conformal set is independently true but Rule 4 has higher priority. T5: argmax=0 (foliar), late_blight prob 0.10 < 0.20, both bullets fail → False.

**S4A.5b — Empty set with adequate max (Rule 5 empty-set sub-rule fires)**
- v3: probs=[0.51, 0.10, 0.08, 0.08, 0.08, 0.10], chilli_leak=0.05
- LoRA: probs=[0.50, 0.12, 0.10, 0.08, 0.10, 0.10]
- PSV: argmax=0, max=0.45, margin=0.10, reliability=0.60
- IQA: ACCEPTABLE
- Classifier: P_final_calibrated=[0.50, 0.10, 0.10, 0.10, 0.10, 0.08, 0.02], max=0.50, margin=0.40
- Conformal (τ=0.40): threshold 0.60; no class above 0.60 → set={}, size=0
- → **Tier 4A**, T5 alert: False
- Walk: max 0.50 ≥ 0.45 → Rule 4 ✗. Rule 5 empty-set sub-rule (set_size == 0) → Tier 4A. This shows the alternative path to 4A: not low confidence per Rule 4, but no class met conformal's 90% coverage threshold. Rare in practice; documented for completeness.

**S4A.6 — Tier 4A with late_blight in set (T5 fires)**
- v3: probs=[0.20, 0.20, 0.21, 0.10, 0.07, 0.02], chilli_leak=0.20
- LoRA: probs=[0.25, 0.22, 0.25, 0.10, 0.08, 0.10]
- PSV: argmax=2, max=0.40, margin=0.05, reliability=0.55
- Classifier: argmax=2, max=0.21, margin=0.01
- Conformal (τ=0.83, threshold 0.17): set={0(0.20), 1(0.20), 2(0.21)}, size=3
- → **Tier 4A**, T5 alert: **True** (rule 4 fires for low confidence; T5 evaluated independently — argmax late_blight → first bullet, also late_blight in set with prob ≥ 0.20)
- Walk: max<0.45 -> Rule 4 fires -> Tier 4A. (Or empty set -> Rule 5 sub-rule, or no rule conditions met -> Rule 9 catch-all.)

**S4A.7 — Max prob exactly 0.45 (boundary; routes to 4A via Rule 9 catch-all)**
- v3: probs=[0.45, 0.20, 0.10, 0.10, 0.10, 0.03], chilli_leak=0.02
- LoRA: probs=[0.46, 0.22, 0.12, 0.10, 0.06, 0.04]
- PSV: argmax=0, max=0.45, margin=0.10, reliability=0.65
- IQA: ACCEPTABLE
- Classifier: argmax=0, max=0.45 exactly, margin=0.20
- Conformal (τ=0.65): set={0}, size=1
- → **Tier 4A** (Rule 9 catch-all), T5 alert: False
- Walk: Rule 4's `< 0.45` ✗ (0.45 is not strictly less). Rule 5 ✗ (size 1, not >=3 or 0). Rule 6 ✗ (size != 2). Rule 7's max threshold `>= 0.85` ✗. Rule 8's max threshold `>= 0.65` ✗. Falls to Rule 9 → Tier 4A.
- Note: this boundary scenario shows that max prob = 0.45 exactly produces the same tier (4A) as max < 0.45, but via a different rule (Rule 9 catch-all instead of Rule 4). The structured reason logged is `rule_id_fired: "catch_all_low_confidence"` so monitoring can flag if Rule 9 firings rise above expected baseline (which would indicate a logic edge case worth review).

**S4A.8 — Catch-all Rule 9 fires for max=0.50 with margin=0.15 (different path than S4A.7)**
- v3: probs=[0.46, 0.19, 0.10, 0.10, 0.05, 0.05], chilli_leak=0.05
- LoRA: probs=[0.52, 0.22, 0.10, 0.08, 0.04, 0.04]
- PSV: argmax=0, max=0.55, margin=0.20, reliability=0.55
- IQA: ACCEPTABLE
- Classifier: argmax=0, max=0.50, margin=0.15
- Conformal (τ=0.55): set={0}, size=1
- → **Tier 4A** (Rule 9 catch-all), T5 alert: False
- Walk: max 0.50 ≥ 0.45 → Rule 4 ✗. set_size==1 → Rule 5 ✗, Rule 6 ✗. Rule 7 condition `max >= 0.85` ✗. Rule 8 condition `margin >= 0.20` ✗ (margin 0.15 < 0.20). Falls to Rule 9 → Tier 4A.
- Note: this scenario provides a second concrete Rule 9 case alongside S4A.7. Both produce Tier 4A but via different threshold failures — S4A.7 has max exactly at 0.45 (boundary at Rule 4); S4A.8 has max in the Rule 8 zone but failed margin threshold. The structured reason logged is `rule_id_fired: "catch_all_low_confidence"` for both.

**S4A.9 — Tier 4A with healthy argmax**
- v3: probs=[0.10, 0.10, 0.05, 0.05, 0.05, 0.40], chilli_leak=0.25
- LoRA: probs=[0.15, 0.15, 0.10, 0.10, 0.10, 0.40]
- PSV: argmax=5, max=0.40, margin=0.05, reliability=0.62
- Classifier: argmax=5, max=0.42, margin=0.10
- Conformal (τ=0.66): set={5}, size=1
- → **Tier 4A**, T5 alert: False (rule 4; healthy argmax)
- Walk: max<0.45 -> Rule 4 fires -> Tier 4A. (Or empty set -> Rule 5 sub-rule, or no rule conditions met -> Rule 9 catch-all.)

**S4A.10 — Tier 4A with OOD argmax**
- v3: probs=[0.13, 0.13, 0.06, 0.06, 0.06, 0.26], chilli_leak=0.30
- LoRA: probs=[0.15, 0.15, 0.10, 0.10, 0.10, 0.40]
- PSV: argmax=6 (OOD), max=0.30, margin=0.05, reliability=0.50
- IQA: ACCEPTABLE
- Classifier: P_final=[0.10, 0.08, 0.05, 0.05, 0.05, 0.27, 0.40]; argmax=6 (OOD), max=0.40, margin=0.13
- Conformal (τ=0.65): threshold 0.35; set={6}, size=1
- → **Tier 4A** (Rule 4: max 0.40 < 0.45 fires), T5 alert: False
- Walk: chilli_leak 0.30 — Rule 3 condition `> 0.40` ✗. Rule 4 condition `< 0.45` ✓ → Tier 4A. T5: argmax=6 (OOD, not in {late_blight, mosaic, ylcv}) → first bullet fails. late_blight prob 0.05 < 0.20 → second bullet fails. T5 = False.
- Note: OOD argmax with low confidence is the system's "I don't think this is a tomato" path. Tier 4A flags low confidence; the response builder (Section 16) can additionally surface the OOD argmax to the user.

**S4A.11 — Tier 4A with diseased argmax including late_blight in set (T5 fires)**
- v3: probs=[0.25, 0.20, 0.20, 0.10, 0.05, 0.10], chilli_leak=0.10
- LoRA: probs=[0.27, 0.22, 0.18, 0.10, 0.06, 0.17]
- PSV: argmax=0, max=0.40, margin=0.10, reliability=0.65
- IQA: ACCEPTABLE
- Classifier: P_final=[0.30, 0.22, 0.20, 0.08, 0.05, 0.10, 0.05]; argmax=0, max=0.30, margin=0.08
- Conformal (τ=0.83): threshold 0.17; set={0, 1, 2}, size=3
- → **Tier 4A** (Rule 4: max 0.30 < 0.45 fires before Rule 5), T5 alert: **True** (late_blight in set with prob 0.20 ≥ 0.20)
- Walk: max 0.30 < 0.45 → Rule 4 fires → Tier 4A. T5 evaluated independently: argmax=0 (foliar, not dangerous) — first bullet fails. late_blight in set with prob 0.20 ≥ 0.20 — second bullet fires → True.

**S4A.12 — Tier 4A despite IQA HIGH (the input is just genuinely confusing)**
- v3: probs=[0.20, 0.18, 0.15, 0.15, 0.12, 0.10], chilli_leak=0.10
- LoRA: probs=[0.22, 0.20, 0.16, 0.14, 0.12, 0.16]
- PSV: argmax=0, max=0.30, margin=0.05, reliability=0.78
- IQA: **HIGH** (input is sharp, well-lit, etc.)
- Classifier: argmax=0, max=0.21, margin=0.03
- Conformal (τ=0.83): set={0, 1, 2, 3, 4}, size=5
- → **Tier 4A**, T5 alert: False (no T5 trigger)
- Walk: max<0.45 -> Rule 4 fires -> Tier 4A. (Or empty set -> Rule 5 sub-rule, or no rule conditions met -> Rule 9 catch-all.)
- Note: HIGH IQA does NOT lift Tier 4A. The input is good but the model is uncertain about what's in the image (perhaps an unusual cultivar or atypical symptoms).

### 15.10 Tier 4B scenarios — pipeline failures

All Tier 4B scenarios share: at least one signal has `forward_succeeded == False`. Rule 1 fires.

**S4B.1 — v3 failed (CUDA OOM)**
- v3: **failed** (CUDA OOM during forward); probs all 0.0; chilli_leak=0.0; succeeded=False
- LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.65, margin=0.30, reliability=0.71
- IQA: ACCEPTABLE
- Classifier (with v3 zeroed): argmax=0, max=0.78, margin=0.65 (operates on LoRA + PSV alone via degraded-mode handling, Section 12.7)
- Conformal: set={0}, size=1
- → **Tier 4B**, T5 alert: False (rule 1)
- Walk: at least one signal failed (forward_succeeded=False) -> Rule 1 fires -> Tier 4B.

**S4B.2 — LoRA failed (NaN in forward)**
- v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
- LoRA: **failed** (NaN propagation; numerical instability); probs all 0.0; succeeded=False
- PSV: argmax=0, max=0.65, margin=0.30, reliability=0.71
- IQA: ACCEPTABLE
- Classifier (with LoRA zeroed): argmax=0, max=0.74, margin=0.55
- Conformal: set={0}, size=1
- → **Tier 4B**, T5 alert: False (rule 1)
- Walk: at least one signal failed (forward_succeeded=False) -> Rule 1 fires -> Tier 4B.

**S4B.3 — PSV failed (segmentation crash)**
- v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
- LoRA: probs=[0.83, 0.06, 0.04, 0.02, 0.02, 0.03]
- PSV: **failed** (exception in disease detection); argmax=0, max=1/6, margin=0, reliability=0.05; succeeded=False
- IQA: ACCEPTABLE
- Classifier (with PSV zeroed): argmax=0, max=0.80, margin=0.65 (relies on v3 + LoRA)
- Conformal: set={0}, size=1
- → **Tier 4B**, T5 alert: False (rule 1)
- Walk: at least one signal failed (forward_succeeded=False) -> Rule 1 fires -> Tier 4B.

**S4B.4 — v3 + LoRA both failed (PSV alone drives output)**
- v3: **failed** (CUDA OOM); probs all 0.0; succeeded=False
- LoRA: **failed** (NaN propagation); probs all 0.0; succeeded=False
- PSV: argmax=0 (foliar), max=0.65, margin=0.30, reliability=0.55, succeeded=True
- IQA: ACCEPTABLE
- Classifier (both v3 and LoRA zeroed; PSV is the only contributing signal via degraded-mode handling per Section 12.7): argmax=0 (foliar), max=0.50, margin=0.25
- Conformal: set={0}, size=1
- → **Tier 4B**, T5 alert: False (rule 1; PSV drove a foliar argmax which is not dangerous)
- Walk: Rule 1 (any signal failed) fires → Tier 4B regardless of classifier output. T5 evaluated independently: argmax=0 (foliar, not dangerous), late_blight prob in classifier output ≈ 0.05 < 0.20, both T5 bullets fail → False. The degraded-mode classifier output here is plausible because Section 12.7 specifies that 20% of training images had v3 + LoRA zeroed, teaching the classifier to read PSV alone.

**S4B.5 — v3 failed but late_blight detected by LoRA + PSV (T5 fires)**
- v3: **failed**; succeeded=False
- LoRA: probs=[0.05, 0.05, 0.81, 0.03, 0.03, 0.03]
- PSV: argmax=2, max=0.62, margin=0.30, reliability=0.71
- IQA: ACCEPTABLE
- Classifier (v3 zeroed): argmax=2, max=0.71, margin=0.55
- Conformal: set={2}, size=1
- → **Tier 4B**, T5 alert: **True** (rule 1 sets tier; T5 fires for late_blight argmax)
- Walk: at least one signal failed (forward_succeeded=False) -> Rule 1 fires -> Tier 4B.
- Note: even with degraded pipeline, T5 still escalates dangerous diseases to the agronomist queue.

**S4B.6 — LoRA failed but T5 fires for late_blight**
- v3: probs=[0.05, 0.05, 0.81, 0.02, 0.02, 0.01], chilli_leak=0.04
- LoRA: **failed**; succeeded=False
- PSV: argmax=2, max=0.62, margin=0.30, reliability=0.71
- Classifier (LoRA zeroed): argmax=2, max=0.74, margin=0.60
- → **Tier 4B**, T5 alert: **True**
- Walk: at least one signal failed (forward_succeeded=False) -> Rule 1 fires -> Tier 4B.

**S4B.7 — All 5 TTA views failed for v3 (effectively v3 failure)**
- v3: TTA invoked; all 5 views had `forward_succeeded=False`. Aggregated: zero probs; v3.forward_succeeded=False.
- LoRA: 5/5 views succeeded, aggregated probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.65, margin=0.30, reliability=0.71
- → **Tier 4B**, T5 alert: False (rule 1; v3 failed across all views)
- Walk: at least one signal failed (forward_succeeded=False) -> Rule 1 fires -> Tier 4B.

**S4B.8 — PSV mid-feature-computation failure**
- v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
- LoRA: probs=[0.83, 0.06, 0.04, 0.02, 0.02, 0.03]
- PSV: caught exception during G5 GLCM computation; the orchestrator's try/except path triggered; succeeded=False; reliability=0.05
- → **Tier 4B**, T5 alert: False
- Walk: at least one signal failed (forward_succeeded=False) -> Rule 1 fires -> Tier 4B.

**S4B.9 — Tier 4B with degraded classifier output pointing at dangerous disease**
- v3: probs=[0.05, 0.05, 0.40, 0.04, 0.05, 0.04], chilli_leak=0.37 (high)
- LoRA: **failed**; succeeded=False
- PSV: argmax=2, max=0.45, margin=0.10, reliability=0.55
- Classifier (LoRA zeroed): argmax=2 (late_blight), max=0.55, margin=0.20 (Rule 1 fires, but classifier still produced output)
- → **Tier 4B**, T5 alert: **True** (rule 1; T5 fires for late_blight)
- Walk: at least one signal failed (forward_succeeded=False) -> Rule 1 fires -> Tier 4B.

**S4B.10 — Multiple failure modes (PSV failed + v3 numerical issue)**
- v3: succeeded=False (RuntimeWarning during softmax); probs all 0.0
- LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02], succeeded=True
- PSV: succeeded=False; reliability=0.05; argmax/max/margin all defaults (from failure path)
- IQA: ACCEPTABLE
- Classifier (v3 and PSV zeroed; LoRA-driven via degraded-mode handling): argmax=0 (foliar), max=0.62, margin=0.45
- Conformal: set={0}, size=1
- → **Tier 4B**, T5 alert: False (rule 1; LoRA drives foliar argmax)
- Walk: two of three signals failed (v3, PSV); Rule 1 fires immediately → Tier 4B. The classifier still produces a usable output because LoRA succeeded and the classifier was trained with degradation augmentation. T5 fails (foliar not dangerous, late_blight prob low).

### 15.11 Tier 5 alert combinations beyond prior coverage

Several of these reinforce that Tier 5 is computed independently of the base tier. T5 fires when (argmax in {late_blight, mosaic, ylcv} AND max >= 0.20) OR (late_blight in set AND late_blight_prob >= 0.20).

**S5.1 — Tier 3B with late_blight in set (full inputs spelled out)**
- v3: probs=[0.45, 0.20, 0.20, 0.04, 0.05, 0.05], chilli_leak=0.01; renormalized: 6 entries sum to 0.99
- LoRA: probs=[0.46, 0.21, 0.18, 0.04, 0.06, 0.05]
- PSV: argmax=0, max=0.55, margin=0.20, reliability=0.65
- IQA: ACCEPTABLE
- Classifier: P_final=[0.46, 0.20, 0.20, 0.04, 0.05, 0.04, 0.01], argmax=0, max=0.46, margin=0.26
- Conformal (tau=0.81): threshold 0.19; set={0, 1, 2}, size=3
- -> **Tier 3B**, T5 alert: **True** (rule 5; late_blight in set with prob 0.20 >= 0.20)
- Walk: prediction_set_size = 3 -> Rule 5 fires -> Tier 3B. T5: argmax=0 (foliar) fails first bullet; late_blight in set with prob 0.20 >= 0.20 fires second bullet -> True.

**S5.2 — Tier 4A with late_blight at exactly 0.20 (full inputs)**
- v3: probs=[0.30, 0.20, 0.18, 0.10, 0.10, 0.10], chilli_leak=0.02; renormalized: 6 entries sum to 0.98
- LoRA: probs=[0.32, 0.20, 0.20, 0.10, 0.10, 0.08]
- PSV: argmax=0, max=0.42, margin=0.12, reliability=0.55
- IQA: ACCEPTABLE
- Classifier: P_final=[0.30, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05], argmax=0, max=0.30, margin=0.10
- Conformal (tau=0.83): threshold 0.17; set={0, 1, 2}, size=3
- -> **Tier 4A** (Rule 4: max 0.30 < 0.45 fires before Rule 5), T5 alert: **True** (late_blight in set with prob 0.20 >= 0.20)
- Walk: max 0.30 < 0.45 -> Rule 4 fires -> Tier 4A (low confidence wins over multi-class set). T5 evaluated independently: argmax=0 (foliar) fails first bullet; late_blight at 0.20 in set fires second bullet at exact boundary -> True.

**S5.3 — Tier 3C with late_blight argmax (full inputs spelled out)**
- v3: probs=[0.05, 0.05, 0.81, 0.02, 0.02, 0.01], chilli_leak=0.04; renormalized: 6 entries sum to 0.96
- LoRA: probs=[0.06, 0.06, 0.81, 0.02, 0.02, 0.03]
- PSV: argmax=2, max=0.45, margin=0.10, **reliability=0.20** (low because disease coverage > 90%)
- IQA: ACCEPTABLE
- Classifier: argmax=2, max=0.86, margin=0.80
- Conformal (tau=0.40): set={2}, size=1
- -> **Tier 3C**, T5 alert: **True** (rule 3 sets tier; T5 fires for late_blight argmax)
- Walk: psv_reliability 0.20 < 0.40 -> Rule 3 fires -> Tier 3C. T5 evaluated independently: argmax=2 (late_blight), max=0.86 >= 0.20 -> first T5 bullet fires -> True. Rule 3 does NOT suppress T5.

**S5.4 — Tier 4B with late_blight argmax (full inputs spelled out)**
- v3: **failed** (CUDA OOM); succeeded=False
- LoRA: probs=[0.05, 0.05, 0.81, 0.03, 0.03, 0.03]
- PSV: argmax=2, max=0.62, margin=0.30, reliability=0.71
- IQA: ACCEPTABLE
- Classifier (v3 zeroed via degraded-mode handling per Section 12.7): argmax=2, max=0.71, margin=0.55
- Conformal: set={2}, size=1
- -> **Tier 4B**, T5 alert: **True** (rule 1 sets tier; T5 fires for late_blight argmax)
- Walk: v3.forward_succeeded=False -> Rule 1 fires -> Tier 4B. T5 evaluated independently: argmax=2 (late_blight), max=0.71 >= 0.20 -> first bullet fires -> True. Even with degraded pipeline, T5 escalates dangerous diseases.

**S5.5 — Tier 3A late_blight vs septoria (full inputs spelled out)**
- v3: probs=[0.05, 0.45, 0.40, 0.03, 0.03, 0.03], chilli_leak=0.01; renormalized: 6 entries sum to 0.99
- LoRA: probs=[0.06, 0.43, 0.38, 0.04, 0.04, 0.05]
- PSV: argmax=1, max=0.50, margin=0.15, reliability=0.74
- IQA: ACCEPTABLE
- Classifier: argmax=1, max=0.45, margin=0.05
- Conformal (tau=0.55): set={1, 2}, size=2
- -> **Tier 3A**, T5 alert: **True** (rule 6; late_blight in set with prob 0.40 >= 0.20)
- Walk: prediction_set_size = 2 -> Rule 6 fires -> Tier 3A. T5: argmax=1 (septoria, not dangerous) fails first bullet; late_blight in set with prob 0.40 >= 0.20 fires second bullet -> True.

**S5.6 — Tier 3D with mosaic argmax (full inputs)**
- v3: probs=[0.04, 0.03, 0.02, 0.02, 0.85, 0.02], chilli_leak=0.02; renormalized: 6 entries sum to 0.98
- LoRA: probs=[0.05, 0.03, 0.02, 0.02, 0.84, 0.04]
- PSV: argmax=4, max=0.69, margin=0.42, reliability=0.71
- IQA: **DEGRADED**
- Classifier: argmax=4 (mosaic), max=0.88, margin=0.81
- Conformal (tau=0.43): set={4}, size=1
- -> **Tier 3D**, T5 alert: **True** (rule 7a fires for would-be Tier 1 with DEGRADED IQA; T5 evaluated independently — mosaic argmax with max 0.88 >= 0.20 fires)
- Walk: Rule 7 or Rule 8 main IF met. Sub-rule 7a or 8a (IQA DEGRADED) fires -> Tier 3D.

**S5.7 — Mosaic at exactly 0.19 (T5 boundary; does NOT fire)**
- v3: probs=[0.20, 0.18, 0.06, 0.06, 0.20, 0.20], chilli_leak=0.10; renormalized: 6 entries sum to 0.80 with chilli=0.10 -> v3 vector reflects 1-0.10=0.90; here v3 is shown summed to 0.80 with leakage 0.10; real F.0 normalizes
- LoRA: probs=[0.18, 0.17, 0.05, 0.05, 0.20, 0.35]
- PSV: argmax=4 (mosaic), max=0.30, margin=0.05, reliability=0.55
- IQA: ACCEPTABLE
- Classifier: P_final=[0.18, 0.17, 0.05, 0.05, 0.19, 0.17, 0.19], argmax=4 (mosaic), max=0.19, margin=0.00
- Conformal (tau=0.85): threshold 0.15; set={0, 1, 4, 5, 6}, size=5
- -> **Tier 4A** (Rule 4: max 0.19 < 0.45 fires), T5 alert: False
- Walk: max 0.19 < 0.45 -> Rule 4 fires -> Tier 4A. T5: argmax=4 (mosaic), max=0.19 < 0.20 -> first bullet fails. Mosaic has no in-set T5 trigger (only late_blight does). T5 = False.
- Note: this tests the T5 boundary at mosaic = 0.19 exactly. The rule's `>= 0.20` strict threshold excludes 0.19. If mosaic max were 0.20, T5 would fire (compare S1.5, S2.5 at high confidence). The strictness prevents extremely-low-confidence mosaic predictions from triggering escalation while still catching any real mosaic case where confidence would be well above 0.20.

**S5.8 — Late_blight at exactly 0.20 (in-set T5 boundary fires)**
- v3: probs=[0.30, 0.20, 0.18, 0.10, 0.10, 0.10], chilli_leak=0.02; renormalized: 6 entries sum to 0.98
- LoRA: probs=[0.32, 0.22, 0.20, 0.10, 0.10, 0.06]
- PSV: argmax=0, max=0.45, margin=0.10, reliability=0.65
- IQA: ACCEPTABLE
- Classifier: P_final=[0.30, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05], argmax=0, max=0.30, margin=0.10
- Conformal (tau=0.83): threshold 0.17; set={0, 1, 2}, size=3
- -> **Tier 4A** (Rule 4: max 0.30 < 0.45 fires before Rule 5), T5 alert: **True** (late_blight in set with prob 0.20 >= 0.20 fires second T5 bullet)
- Walk: max 0.30 < 0.45 -> Rule 4 -> Tier 4A. T5: late_blight prob 0.20 satisfies the boundary `>= 0.20` -> True.

**S5.9 — Late_blight at 0.19 in set (T5 in-set bullet does NOT fire — paired with S5.8)**
- v3: probs=[0.30, 0.20, 0.18, 0.11, 0.10, 0.09], chilli_leak=0.02
- LoRA: probs=[0.32, 0.22, 0.19, 0.10, 0.10, 0.07]
- PSV: argmax=0, max=0.45, margin=0.10, reliability=0.65
- IQA: ACCEPTABLE
- Classifier: P_final=[0.30, 0.21, 0.19, 0.11, 0.09, 0.05, 0.05], argmax=0, max=0.30, margin=0.09
- Conformal (tau=0.83): threshold 0.17; set={0, 1, 2}, size=3
- -> **Tier 4A**, T5 alert: False
- Walk: max 0.30 < 0.45 -> Rule 4 -> Tier 4A. T5: argmax=0 (foliar) fails first bullet; late_blight in set with prob 0.19 < 0.20 fails second bullet (strict `>= 0.20`). T5 = False.
- Note: paired with S5.8 to test the in-set T5 boundary from both sides. At 0.20 -> fires; at 0.19 -> does not fire.

**S5.10 — Three dangerous classes in set with foliar argmax (T5 fires only via late_blight in-set bullet)**
- v3: probs=[0.30, 0.05, 0.20, 0.18, 0.18, 0.04], chilli_leak=0.05; 6 entries sum to 0.95
- LoRA: probs=[0.32, 0.06, 0.20, 0.18, 0.18, 0.06]
- PSV: argmax=0, max=0.45, margin=0.10, reliability=0.55
- IQA: ACCEPTABLE
- Classifier: P_final=[0.30, 0.05, 0.20, 0.18, 0.18, 0.05, 0.04], argmax=0 (foliar), max=0.30, margin=0.10
- Conformal (tau=0.83): threshold 0.17; set={0, 2, 3, 4}, size=4 (all three dangerous classes admitted)
- -> **Tier 4A** (Rule 4: max 0.30 < 0.45 fires), T5 alert: **True**
- Walk: max 0.30 < 0.45 -> Rule 4 -> Tier 4A. T5 evaluated: argmax=0 (foliar, not dangerous) -> first bullet fails. Late_blight in set with prob 0.20 >= 0.20 -> second bullet fires -> True. NOTE: mosaic and YLCV in set do NOT trigger T5 because their T5 triggers are argmax-only (per Section 14.3), not in-set. Only late_blight has an in-set T5 trigger.
- Note: this scenario explicitly tests the asymmetric T5 trigger behavior — three dangerous classes simultaneously in the prediction set, but only late_blight contributes to T5 firing because of its in-set rule. If the implementation incorrectly added in-set triggers for mosaic or YLCV, this scenario would still fire T5 (so the test wouldn't catch that bug); a more sensitive variant would have late_blight prob < 0.20 with mosaic argmax — that case is covered by S5.6 and the asymmetry is documented in Section 14.3.

**S5.11 — Tier 4B with late_blight in set but not argmax (T5 fires via in-set rule despite degraded pipeline)**
- v3: **failed**; succeeded=False
- LoRA: probs=[0.50, 0.05, 0.25, 0.05, 0.05, 0.10]
- PSV: argmax=0, max=0.55, margin=0.20, reliability=0.65
- IQA: ACCEPTABLE
- Classifier (v3 zeroed via degraded-mode): P_final=[0.45, 0.05, 0.25, 0.05, 0.05, 0.10, 0.05], argmax=0 (foliar), max=0.45, margin=0.20
- Conformal (tau=0.80): threshold 0.20; set={0, 2}, size=2 (foliar at 0.45 ≥ 0.20, late_blight at 0.25 ≥ 0.20, others below threshold)
- -> **Tier 4B** (Rule 1: v3 failed), T5 alert: **True** (late_blight in set with prob 0.25 >= 0.20)
- Walk: v3.forward_succeeded=False -> Rule 1 fires -> Tier 4B (regardless of classifier output). T5 evaluated independently: argmax=0 (foliar, not dangerous) -> first bullet fails. Late_blight in set with prob 0.25 >= 0.20 -> second bullet fires -> True. The degraded pipeline does NOT suppress T5; the in-set rule still detects danger when the classifier puts non-trivial mass on late_blight despite v3 failing.
- Note: this scenario is safety-critical. It confirms that pipeline failure does not silently mask late_blight detection when the residual signals (LoRA + PSV + classifier-degraded-mode) put enough mass on late_blight to admit it to the conformal set. Pair with the cautionary note in Section 15.16 known-gaps about silent T5 misses.

### 15.12 Boundary and edge cases

This subsection enumerates each threshold used by the rule chain and shows the behavior at exact boundary values. Every scenario specifies full inputs.

**SB.1 — combined_max_prob = 0.85 exactly (Tier 1 fires; boundary inclusive)**
- v3: probs=[0.82, 0.05, 0.04, 0.03, 0.02, 0.01], chilli_leak=0.03
- LoRA: probs=[0.83, 0.06, 0.04, 0.03, 0.02, 0.02]
- PSV: argmax=0, max=0.65, margin=0.30, reliability=0.55
- IQA: ACCEPTABLE
- Classifier: argmax=0, max=0.85, margin=0.55
- Conformal (tau=0.45): set={0}, size=1
- -> **Tier 1**, T5 alert: False (rule 7c; max 0.85 >= 0.85 satisfies Rule 7's threshold inclusively)
- Walk: max 0.85 satisfies Rule 7's `>= 0.85` (boundary inclusive). All other Rule 7 conditions met. Sub-rules 7a (DEGRADED IQA) and 7b (underpowered class) both fail (IQA=ACCEPTABLE, foliar not underpowered). Sub-rule 7c (default) fires -> Tier 1.

**SB.2 — combined_max_prob = 0.84999999 (Tier 2; just below Rule 7's threshold)**
- v3: probs=[0.82, 0.05, 0.04, 0.03, 0.02, 0.01], chilli_leak=0.03
- LoRA: probs=[0.83, 0.06, 0.04, 0.03, 0.02, 0.02]
- PSV: argmax=0, max=0.65, margin=0.30, reliability=0.55
- IQA: ACCEPTABLE
- Classifier: argmax=0 (foliar), max=0.84999999, margin=0.55
- Conformal (tau=0.45): set={0}, size=1
- -> **Tier 2**, T5 alert: False (argmax=0 foliar, not dangerous; late_blight prob 0.04 < 0.20)
- Walk: Rule 7 condition `>= 0.85` ✗ at 0.84999999. Rule 8 condition `>= 0.65` ✓. Tier 2 via Rule 8c default.
- Note: this boundary tests floating-point precision at the Rule 7/Rule 8 transition.

**SB.3 — combined_margin = 0.30 exactly (Tier 1; boundary inclusive)**
- v3: probs=[0.90, 0.05, 0.01, 0.01, 0.00, 0.01], chilli_leak=0.02
- LoRA: probs=[0.90, 0.06, 0.02, 0.01, 0.00, 0.01]
- PSV: argmax=0, max=0.71, margin=0.30, reliability=0.74
- IQA: ACCEPTABLE
- Classifier: argmax=0, max=0.85, margin=0.30
- Conformal (tau=0.45): set={0}, size=1
- -> **Tier 1**, T5 alert: False (rule 7c; margin 0.30 >= 0.30 satisfies inclusively)
- Walk: all Rule 7 main conditions met (max>=0.85, margin>=0.30, psv_reliability>=0.50, chilli<0.20, IQA in {ACCEPTABLE,HIGH}, set_size==1, all signals OK). Sub-rules 7a (DEGRADED) and 7b (underpowered) both fail. Sub-rule 7c (default) fires -> Tier 1.

**SB.4 — combined_margin = 0.29 (just below Rule 7; falls to Rule 8 -> Tier 2)**
- v3: probs=[0.90, 0.05, 0.01, 0.01, 0.00, 0.01], chilli_leak=0.02
- LoRA: probs=[0.90, 0.06, 0.02, 0.01, 0.00, 0.01]
- PSV: argmax=0, max=0.71, margin=0.30, reliability=0.74
- IQA: ACCEPTABLE
- Classifier: argmax=0, max=0.85, margin=0.29
- Conformal (tau=0.45): set={0}, size=1
- -> **Tier 2**, T5 alert: False
- Walk: Rule 7 margin condition `>= 0.30` ✗ at 0.29. Rule 8 margin condition `>= 0.20` ✓. Tier 2.

**SB.5 — psv_reliability = 0.40 exactly (NOT 3C; Rule 3 strict `< 0.40`)**
- v3: probs=[0.87, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.05
- LoRA: probs=[0.89, 0.04, 0.03, 0.02, 0.01, 0.01]
- PSV: argmax=0, max=0.45, margin=0.08, **reliability=0.40 exactly**
- IQA: ACCEPTABLE
- Classifier: argmax=0, max=0.91, margin=0.86
- Conformal (tau=0.40): set={0}, size=1
- -> **Tier 2**, T5 alert: False
- Walk: Rule 3 condition `psv_reliability < 0.40` ✗ at 0.40 (strict). Rule 7 condition `>= 0.50` ✗. Rule 8 condition `>= 0.40` ✓ (inclusive). Tier 2 fires.

**SB.6 — psv_reliability = 0.39 (Rule 3 fires -> Tier 3C; just below threshold)**
- v3: probs=[0.87, 0.04, 0.01, 0.01, 0.01, 0.01], chilli_leak=0.05
- LoRA: probs=[0.89, 0.04, 0.03, 0.02, 0.01, 0.01]
- PSV: argmax=0, max=0.40, margin=0.05, **reliability=0.39**
- IQA: ACCEPTABLE
- Classifier: argmax=0, max=0.91, margin=0.86
- Conformal (tau=0.40): set={0}, size=1
- -> **Tier 3C**, T5 alert: False (rule 3 fires)
- Walk: Rule 3 fires (psv_reliability<0.40 OR chilli_leakage>0.40) -> Tier 3C.

**SB.7 — chilli_leakage = 0.40 exactly (NOT 3C; Rule 3 strict `> 0.40`; falls to Rule 9 -> Tier 4A)**
- v3: probs=[0.55, 0.04, 0.01, 0.00, 0.00, 0.00], chilli_leak=0.40 exactly
- LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.65, margin=0.30, reliability=0.74
- IQA: ACCEPTABLE
- Classifier: argmax=0, max=0.82, margin=0.71
- Conformal (tau=0.45): set={0}, size=1
- -> **Tier 4A**, T5 alert: False
- Walk: Rule 3 condition `chilli_leakage > 0.40` ✗ at 0.40 (strict). Rule 7 condition `< 0.20` ✗. Rule 8 condition `< 0.30` ✗. Rule 9 catch-all -> Tier 4A.

**SB.8 — chilli_leakage = 0.41 (Rule 3 fires -> Tier 3C)**
- v3: probs=[0.50, 0.04, 0.02, 0.02, 0.00, 0.01], chilli_leak=0.41
- LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.65, margin=0.30, reliability=0.71
- IQA: ACCEPTABLE
- Classifier: argmax=0, max=0.78, margin=0.65
- Conformal (tau=0.50): set={0}, size=1
- -> **Tier 3C**, T5 alert: False (rule 3 fires)
- Walk: Rule 3 fires (psv_reliability<0.40 OR chilli_leakage>0.40) -> Tier 3C.

**SB.9 — combined_max_prob = NaN (orchestrator routes to Tier 4B)**
- Per Section 11.2 (NaN handling contract): if `combined_max_prob` is NaN or otherwise non-numeric, the pipeline marks all signals as failed and TTA does NOT fire. The pipeline returns a fallback result with all signals' `forward_succeeded = False`.
- Tier assignment with all signals failed -> Rule 1 fires -> **Tier 4B**.
- The orchestrator (Section 21, currently future spec) is responsible for NaN detection at the boundary between classifier output and tier assignment. The orchestrator's contract: any NaN in classifier output causes the orchestrator to mark all upstream signals as failed before calling `assign_tier`.
- -> **Tier 4B**, T5 alert: False (cannot be evaluated reliably with NaN; defaults to False per Section 21 contract)
- Walk: at least one signal failed (forward_succeeded=False) -> Rule 1 fires -> Tier 4B.
- Note: this scenario depends on the orchestrator implementing the Section 21 NaN-detection contract. Without that, NaN comparisons in Python return False for both `>=` and `<`, so Rule 4 ✗, Rule 7/8 ✗, and the chain falls to Rule 9 -> Tier 4A. The Phase F.0 test for SB.9 must verify Tier 4B is returned, which requires the orchestrator to handle NaN before tier assignment is called.

**SB.10 — All classifier outputs equal (1/7 each, tied argmax)**
- v3: probs=[0.143, 0.143, 0.143, 0.143, 0.143, 0.143], chilli_leak=0.142; 6 entries sum to 0.858
- LoRA: probs=[0.167, 0.167, 0.167, 0.167, 0.166, 0.166]
- PSV: argmax=0, max=0.143, margin=0.00, reliability=0.30 (low reliability is plausible for tied PSV)
- IQA: ACCEPTABLE
- Classifier: P_final_calibrated approximately uniform [0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0.142], argmax=0 (numpy first-index convention when tied), max=0.143, margin=0.000
- Conformal (tau=0.86): threshold 0.14; set={0, 1, 2, 3, 4, 5}, size=6
- -> **Tier 4A** (Rule 4: max 0.143 < 0.45 fires before Rule 5), T5 alert: False
- Walk: max 0.143 < 0.45 -> Rule 4 -> Tier 4A. T5: argmax=0 (foliar tied with others, but numpy returns first index) -> first bullet fails (foliar not dangerous AND max 0.143 < 0.20). Late_blight prob 0.143 < 0.20 -> second bullet fails. T5 = False.
- Note: this is a degenerate uniform case. PSV reliability would also be expected to be low because the system can't distinguish classes. Rule 4 catches before Rule 5.

**SB.11 — prediction_set_size = 0 (empty set; foliar argmax pinned for testability)**
- v3: probs=[0.50, 0.10, 0.10, 0.10, 0.10, 0.10], chilli_leak=0.00
- LoRA: probs=[0.50, 0.12, 0.10, 0.10, 0.10, 0.08]
- PSV: argmax=0, max=0.45, margin=0.10, reliability=0.60
- IQA: ACCEPTABLE
- Classifier: P_final_calibrated=[0.50, 0.10, 0.10, 0.10, 0.10, 0.08, 0.02], argmax=0 (foliar), max=0.50, margin=0.40
- Conformal (tau=0.40): threshold 0.60; no class above -> set={}, size=0
- Per Section 14.5 Rule 5 (empty-set sub-rule): empty set -> Tier 4A.
- -> **Tier 4A** (rule 5 empty-set sub-rule), T5 alert: False (argmax=0 foliar, not dangerous; late_blight prob 0.10 < 0.20)
- Walk: max<0.45 -> Rule 4 fires -> Tier 4A. (Or empty set -> Rule 5 sub-rule, or catch-all -> Rule 9.)
- Note: this scenario pins the classifier output to make T5 deterministic. Boundary at empty set is independent of class identity; if argmax were late_blight with max >= 0.20, T5 would fire (test infrastructure can construct that variant separately).

**SB.12 — prediction_set_size = 7 (all classes; theoretically reachable, practically unreachable)**
- Classifier: P_final_calibrated=[0.18, 0.16, 0.15, 0.14, 0.14, 0.13, 0.10], argmax=0 (foliar), max=0.18, margin=0.02
- Rule 4 fires first (max 0.18 < 0.45) -> Tier 4A, before Rule 5 has a chance.
- -> **Tier 4A** (Rule 4), T5 alert: False (argmax=0 foliar, not dangerous; late_blight prob 0.15 < 0.20)
- Walk: max<0.45 -> Rule 4 fires -> Tier 4A. (Or empty set -> Rule 5 sub-rule, or catch-all -> Rule 9.)
- Note: this scenario documents that Rule 5's "size==7" branch is theoretically reachable but practically unreachable. F.0-typical tau values produce sets of 1-3 classes; size 7 would require tau ≈ 0.99 which would mean the calibration set has ~99% of cases needing all 7 classes to achieve coverage — i.e., the classifier is so uncertain that calibration has broken down. The Rule 5 logic still handles size 7 correctly (it would route to Tier 3B if the precondition were met), but in practice the request triggers Rule 4 first.

**SB.13 — chilli_leakage = 0.20 exactly (Tier 1's strict `< 0.20` cap fails; falls to Tier 2)**
- v3: probs=[0.74, 0.04, 0.01, 0.00, 0.00, 0.01], chilli_leak=0.20 exactly
- LoRA: probs=[0.85, 0.05, 0.04, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.71, margin=0.42, reliability=0.74
- IQA: ACCEPTABLE
- Classifier: argmax=0, max=0.86, margin=0.79
- Conformal (tau=0.40): set={0}, size=1
- -> **Tier 2**, T5 alert: False
- Walk: Rule 3 condition `chilli_leakage > 0.40` ✗. Rule 7 condition `chilli_leakage < 0.20` ✗ at 0.20 exactly (strict less-than). Rule 8 condition `chilli_leakage < 0.30` ✓ at 0.20. Tier 2 fires via Rule 8c default.
- Note: this boundary parallels SB.7 (chilli=0.40 boundary at Rule 3) and tests that Tier 1's stricter chilli cap is correctly enforced. At chilli=0.20 exactly, Tier 1 is excluded but Tier 2 admits.

**SB.14 — combined_margin = 0 exactly (top two classes tied)**
- v3: probs=[0.40, 0.40, 0.05, 0.05, 0.05, 0.05], chilli_leak=0.00
- LoRA: probs=[0.40, 0.40, 0.05, 0.05, 0.05, 0.05]
- PSV: argmax=0, max=0.50, margin=0.00, reliability=0.55
- IQA: ACCEPTABLE
- Classifier: P_final_calibrated=[0.40, 0.40, 0.05, 0.05, 0.05, 0.04, 0.01], argmax=0 (numpy first-index convention when foliar and septoria tied at 0.40), max=0.40, margin=0.00
- Conformal (tau=0.65): threshold 0.35; set={0, 1}, size=2
- -> **Tier 4A** (Rule 4: max 0.40 < 0.45 fires before Rule 6), T5 alert: False (argmax=0 foliar; late_blight prob 0.05 < 0.20)
- Walk: max 0.40 < 0.45 -> Rule 4 -> Tier 4A. T5: argmax=0 (foliar, not dangerous because of tie resolved to first index) fails first bullet; late_blight prob 0.05 fails second bullet. T5 = False.
- Note: this tests behavior at exact tie. numpy's argmax returns the first index when tied. The conformal set correctly admits both tied classes (both >= 0.35 threshold), but Rule 4 wins on priority. If max were >= 0.45 (e.g., both classes at 0.50), Rule 6 would fire -> Tier 3A, which is the correct behavior for a genuine tie.

**SB.15 — combined_margin = 0 with high max (Rule 6 fires for tied top classes)**
- v3: probs=[0.50, 0.50, 0.00, 0.00, 0.00, 0.00], chilli_leak=0.00
- LoRA: probs=[0.50, 0.50, 0.00, 0.00, 0.00, 0.00]
- PSV: argmax=0, max=0.55, margin=0.00, reliability=0.55
- IQA: ACCEPTABLE
- Classifier: P_final_calibrated=[0.50, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00], argmax=0, max=0.50, margin=0.00
- Conformal (tau=0.55): threshold 0.45; set={0, 1}, size=2
- -> **Tier 3A** (Rule 6: set_size==2), T5 alert: False
- Walk: max 0.50 >= 0.45 -> Rule 4 ✗. set_size==2 -> Rule 6 fires -> Tier 3A. The tie is correctly represented by both classes appearing in the prediction set.

### 15.13 Underpowered class downgrade scenarios

These scenarios demonstrate the per-class minimum-recall guard from Section 14.4. Every scenario specifies full inputs.

**SUP.1 — Definitive YLCV with underpowered guard -> 3A downgrade + T5**
- v3: probs=[0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00
- LoRA: probs=[0.05, 0.05, 0.04, 0.81, 0.02, 0.03]
- PSV: argmax=3, max=0.74, margin=0.50, reliability=0.78
- IQA: ACCEPTABLE
- Classifier: argmax=3 (YLCV), max=0.88, margin=0.82
- Conformal (tau=0.40): set={3}, size=1
- Underpowered: F.0 reports YLCV recall < 0.50 (flagged underpowered)
- -> **Tier 3A** (downgrade via sub-rule 7b), T5 alert: **True** (YLCV argmax with max 0.88 >= 0.20)
- Walk: Rule 7 main IF: max>=0.85 ✓, margin>=0.30 ✓, psv_reliability>=0.50 ✓, chilli<0.20 ✓, IQA != REJECT ✓, set_size==1 ✓. Sub-rule 7a (DEGRADED IQA) ✗. Sub-rule 7b (underpowered) ✓ -> Tier 3A. T5 evaluated independently: YLCV argmax with max 0.88 >= 0.20 -> first bullet fires -> True.

**SUP.2 — Confident mosaic with underpowered guard -> 3A downgrade + T5**
- v3: probs=[0.04, 0.04, 0.04, 0.05, 0.74, 0.05], chilli_leak=0.04
- LoRA: probs=[0.05, 0.05, 0.05, 0.06, 0.71, 0.08]
- PSV: argmax=4, max=0.62, margin=0.32, reliability=0.69
- IQA: ACCEPTABLE
- Classifier: argmax=4 (mosaic), max=0.72, margin=0.55
- Conformal (tau=0.55): set={4}, size=1
- Underpowered: F.0 reports mosaic recall < 0.50 (flagged underpowered)
- -> **Tier 3A** (downgrade via sub-rule 8b), T5 alert: **True** (mosaic argmax with max 0.72 >= 0.20)
- Walk: Rule 8 main IF: max>=0.65 ✓, margin>=0.20 ✓, psv_reliability>=0.40 ✓, chilli<0.30 ✓, set_size==1 ✓. Sub-rule 8a (DEGRADED IQA) ✗. Sub-rule 8b (underpowered) ✓ -> Tier 3A. T5: mosaic argmax with max 0.72 -> first bullet fires -> True.

**SUP.3 — YLCV with IQA DEGRADED — sub-rule 7a wins over 7b -> Tier 3D not 3A**
- v3: probs=[0.04, 0.04, 0.04, 0.85, 0.02, 0.01], chilli_leak=0.00
- LoRA: probs=[0.05, 0.05, 0.04, 0.81, 0.02, 0.03]
- PSV: argmax=3, max=0.74, margin=0.50, reliability=0.78
- IQA: **DEGRADED**
- Classifier: argmax=3 (YLCV), max=0.88, margin=0.82
- Conformal (tau=0.40): set={3}, size=1
- Underpowered: F.0 reports YLCV recall < 0.50 (flagged underpowered)
- -> **Tier 3D** (sub-rule 7a wins over 7b), T5 alert: **True** (YLCV argmax with max 0.88 >= 0.20)
- Walk: Rule 7 main IF met. Sub-rule 7a (DEGRADED) ✓ -> Tier 3D. Sub-rule 7b (underpowered) NOT evaluated due to 7a precedence. T5 evaluated independently -> True.
- Note: this scenario establishes the design choice from Section 14.5 — 7a (IQA) precedes 7b (underpowered) because IQA is user-actionable (retake the photo). The underpowered downgrade is a model-quality concern, not user-actionable.

**SUP.4 — Mosaic with PSV unreliable — Rule 3 fires before Rule 7 -> Tier 3C, T5 fires**
- v3: probs=[0.05, 0.04, 0.02, 0.02, 0.85, 0.02], chilli_leak=0.00
- LoRA: probs=[0.06, 0.05, 0.04, 0.02, 0.81, 0.02]
- PSV: argmax=4, max=0.30, margin=0.05, **reliability=0.30** (PSV unreliable)
- IQA: ACCEPTABLE
- Classifier: argmax=4 (mosaic), max=0.88, margin=0.83
- Conformal (tau=0.40): set={4}, size=1
- -> **Tier 3C** (Rule 3 fires due to PSV unreliable), T5 alert: **True** (mosaic argmax with max 0.88 >= 0.20)
- Walk: Rule 3 condition `psv_reliability < 0.40` ✓ at 0.30. Rule 3 fires before Rule 7. Underpowered guard never reached. T5 evaluated independently -> fires.

**SUP.5 — Healthy not underpowered, no downgrade**
- v3: probs=[0.01, 0.02, 0.01, 0.02, 0.01, 0.91], chilli_leak=0.02
- LoRA: probs=[0.02, 0.03, 0.02, 0.02, 0.02, 0.89]
- PSV: argmax=5 (healthy), max=0.79, margin=0.58, reliability=0.83
- IQA: ACCEPTABLE
- Classifier: argmax=5 (healthy), max=0.93, margin=0.88
- Conformal (tau=0.40): set={5}, size=1
- Underpowered: healthy is NOT flagged underpowered (substantial training samples)
- -> **Tier 1** (no downgrade), T5 alert: False (healthy is not a dangerous class)
- Walk: Rule 7 main IF met. Sub-rule 7a (DEGRADED) ✗. Sub-rule 7b (underpowered) ✗ because healthy is not flagged. Sub-rule 7c (default) -> Tier 1. T5: argmax=5 (healthy, not dangerous) -> first bullet fails. Late_blight prob 0.01 < 0.20 -> second bullet fails. T5 = False.

**SUP.6 — YLCV at low confidence (Tier 4A) — underpowered guard doesn't apply**
- v3: probs=[0.20, 0.20, 0.05, 0.30, 0.10, 0.05], chilli_leak=0.10
- LoRA: probs=[0.22, 0.22, 0.06, 0.30, 0.10, 0.10]
- PSV: argmax=3, max=0.40, margin=0.05, reliability=0.55
- IQA: ACCEPTABLE
- Classifier: argmax=3 (YLCV), max=0.31, margin=0.02
- Conformal (tau=0.83): threshold 0.17; set={0, 1, 3}, size=3
- Underpowered: F.0 reports YLCV recall < 0.50 (flagged underpowered)
- -> **Tier 4A** (Rule 4: max 0.31 < 0.45 fires before sub-rules 7b/8b), T5 alert: **True** (YLCV argmax with max 0.31 >= 0.20)
- Walk: max 0.31 < 0.45 -> Rule 4 -> Tier 4A. The underpowered guard only downgrades from Rules 7/8 to Tier 3A; at Tier 4A the guard isn't relevant — the system already says "low confidence." T5 still fires for YLCV argmax.

**SUP.7 — Confident mosaic with IQA DEGRADED + underpowered: sub-rule 8a wins over 8b -> Tier 3D**
- v3: probs=[0.04, 0.04, 0.04, 0.05, 0.74, 0.05], chilli_leak=0.04
- LoRA: probs=[0.05, 0.05, 0.05, 0.06, 0.71, 0.08]
- PSV: argmax=4, max=0.62, margin=0.32, reliability=0.69
- IQA: **DEGRADED**
- Classifier: argmax=4 (mosaic), max=0.72, margin=0.55
- Conformal (tau=0.55): set={4}, size=1
- Underpowered: F.0 reports mosaic recall < 0.50 (flagged underpowered)
- -> **Tier 3D** (sub-rule 8a wins over 8b), T5 alert: **True** (mosaic argmax with max 0.72 >= 0.20)
- Walk: Rule 8 main IF met. Sub-rule 8a (DEGRADED) ✓ -> Tier 3D. Sub-rule 8b (underpowered) NOT evaluated due to 8a precedence. T5 evaluated independently -> True.
- Note: this scenario closes a coverage gap by testing sub-rule 8a precedence over 8b (parallel to SUP.3 which tests 7a precedence over 7b at the Rule 7 level). Both Rule 7 and Rule 8 sub-rule chains have the same precedence design: IQA cap before underpowered cap, because IQA is user-actionable and underpowered is a model-quality concern.

### 15.14 Cross-signal disagreement scenarios

These scenarios explore behavior when v3, LoRA, and PSV disagree.

**SDIS.1 — v3 says foliar, LoRA says septoria, PSV agrees with v3 (high JSD)**
- v3: probs=[0.78, 0.12, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.00
- LoRA: probs=[0.10, 0.78, 0.05, 0.03, 0.02, 0.02]
- PSV: argmax=0 (foliar), max=0.65, margin=0.30, reliability=0.71
- JSD(v3, LoRA) ≈ 0.45 (relatively high)
- Classifier: argmax depends on learned weighting. Likely splits between foliar and septoria.
- Possible classifier: argmax=0, max=0.50, margin=0.25
- Conformal (τ=0.55): set={0, 1}, size=2
- → **Tier 3A**, T5 alert: False (rule 6)
- Walk: Rules 1-4 don't fire (signals OK, PSV/chilli OK, max>=0.45). prediction_set_size==2 -> Rule 6 fires -> Tier 3A. (In scenarios labeled as underpowered downgrade: sub-rule 7b/8b fires from a would-be Tier 1/2 instead.)
- Note: high JSD goes into the classifier as feature 16; the classifier learns that high JSD → reduce confidence.

**SDIS.2 — All three disagree (v3 foliar, LoRA septoria, PSV late_blight)**
- v3: probs=[0.50, 0.30, 0.10, 0.04, 0.04, 0.02], chilli_leak=0.00
- LoRA: probs=[0.20, 0.55, 0.15, 0.04, 0.04, 0.02]
- PSV: argmax=2 (late_blight), max=0.45, margin=0.10, reliability=0.65
- IQA: ACCEPTABLE
- JSD(v3, LoRA) ≈ 0.42; agree_v3 (PSV vs v3) = 0; agree_lora (PSV vs LoRA) = 0; classifier feature 16 (JSD) is high
- Classifier: P_final_calibrated=[0.30, 0.30, 0.25, 0.05, 0.04, 0.04, 0.02]; argmax=0 (foliar by numpy first-index when tied with septoria), max=0.30, margin=0.00
- Conformal (τ=0.80): threshold 0.20; set={0, 1, 2}, size=3
- → **Tier 4A** (Rule 4: max 0.30 < 0.45 fires before Rule 5), T5 alert: **True** (late_blight in set with prob 0.25 ≥ 0.20)
- Walk: max 0.30 < 0.45 → Rule 4 → Tier 4A. T5: argmax=0 (foliar) fails first bullet; late_blight in set with prob 0.25 ≥ 0.20 fires second bullet → True.
- Note: classifier output reflects high JSD by spreading mass across the three disputed classes. Foliar wins argmax via numpy first-index when tied with septoria. Set admits all three because all are above the conformal threshold.

**SDIS.3 — v3 confident foliar, LoRA confident healthy (extreme disagreement)**
- v3: probs=[0.85, 0.04, 0.02, 0.02, 0.02, 0.05], chilli_leak=0.00
- LoRA: probs=[0.05, 0.04, 0.02, 0.02, 0.02, 0.85]
- PSV: argmax=5 (healthy; PSV interpreted leaf as healthy), max=0.50, margin=0.10, reliability=0.55
- IQA: ACCEPTABLE
- JSD(v3, LoRA) ≈ 0.55 (very high — near-orthogonal distributions)
- Classifier: P_final_calibrated=[0.40, 0.04, 0.02, 0.02, 0.02, 0.45, 0.05]; argmax=5 (healthy), max=0.45, margin=0.05
- Conformal (τ=0.55): threshold 0.45; set={0, 5}, size=2 (foliar at 0.40 just below threshold, but with margin to spare on healthy at 0.45 — actually 0.40 < 0.45 fails. Adjust τ: τ=0.60, threshold 0.40; set={0, 5}, both at threshold inclusive)
- Conformal (τ=0.60): threshold 0.40; set={0 (0.40), 5 (0.45)}, size=2
- → **Tier 3A** (Rule 6: set_size==2), T5 alert: False (argmax=5 healthy, not dangerous; late_blight prob 0.02 < 0.20)
- Walk: max 0.45 ≥ 0.45 → Rule 4 ✗ (boundary inclusive). set_size==2 → Rule 6 → Tier 3A. T5: argmax healthy fails first bullet; late_blight prob 0.02 fails second bullet. T5 = False.
- Note: classifier resolves the v3-vs-LoRA disagreement by giving healthy slight edge (LoRA's confident healthy + PSV's healthy argmax both push), but foliar still admits to set due to v3's strong signal. Tier 3A correctly flags ambiguity. The high JSD feature signals to the classifier "don't be confident."

**SDIS.4 — v3 says late_blight, LoRA + PSV say foliar (LoRA + PSV outvote v3)**
- v3: probs=[0.20, 0.05, 0.65, 0.04, 0.04, 0.02], chilli_leak=0.00
- LoRA: probs=[0.78, 0.10, 0.05, 0.03, 0.02, 0.02]
- PSV: argmax=0, max=0.62, margin=0.30, reliability=0.74
- IQA: ACCEPTABLE
- Classifier (weights LoRA + PSV agreement higher, but v3's late_blight signal pulls some mass): P_final=[0.62, 0.05, 0.25, 0.02, 0.02, 0.02, 0.02]; argmax=0, max=0.62, margin=0.37
- Conformal (τ=0.78): threshold 0.22; set={0(0.62), 2(0.25)}, size=2
- → **Tier 3A** (Rule 6: set_size==2), T5 alert: **True** (late_blight in set with prob 0.25 ≥ 0.20)
- Walk: max 0.62 < 0.65 fails Rule 8's max threshold; if it had passed, single-class Tier 2 would still be blocked by set_size==2. Rule 6 (set_size==2) fires → Tier 3A. T5: argmax=0 (foliar) fails first bullet; late_blight in set with prob 0.25 ≥ 0.20 → second bullet fires → True.
- Note: this scenario shows the classifier successfully resisting v3's overconfident late_blight signal because LoRA + PSV agree on foliar, but conformal still admits late_blight in the set because v3's residual mass keeps late_blight above the conformal threshold. Tier 3A correctly flags ambiguity, and T5 correctly fires for the late_blight residual mass.

**SDIS.5 — PSV strongly disagrees with v3 + LoRA (which agree)**
- v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
- LoRA: probs=[0.83, 0.06, 0.05, 0.02, 0.02, 0.02]
- PSV: argmax=4 (mosaic), max=0.55, margin=0.20, reliability=0.55
- agree_v3 = 0 (PSV mosaic vs v3 foliar); agree_lora = 0
- Classifier: weights v3 + LoRA agreement higher; argmax=0, max=0.78, margin=0.62
- Conformal (τ=0.45): set={0}, size=1
- → **Tier 2**, T5 alert: False (rule 8c; PSV's mosaic call doesn't make it into the prediction set or the argmax)
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.

**SDIS.6 — All three agree but on a class with low PSV reliability**
- v3: probs=[0.80, 0.05, 0.04, 0.02, 0.02, 0.02], chilli_leak=0.05
- LoRA: probs=[0.83, 0.06, 0.05, 0.02, 0.02, 0.02]
- PSV: argmax=0, max=0.55, margin=0.18, **reliability=0.35**
- Classifier: argmax=0, max=0.84, margin=0.78
- Conformal (τ=0.40): set={0}, size=1
- → **Tier 3C** (Rule 3 fires due to PSV reliability), T5 alert: False
- Walk: Rule 3 fires (psv_reliability<0.40 OR chilli_leakage>0.40) -> Tier 3C.

### 15.15 TTA-specific scenarios

TTA changes the classifier's input (via aggregated v3 and LoRA outputs), which changes `combined_max_prob` and thus the tier. These scenarios show initial vs post-TTA tiers.

**STTA.1 — Initial 0.50 → TTA fires (2-view) → final 0.72 (Tier 2; foliar argmax pinned)**
- 1-view classifier output: argmax=0 (foliar), max=0.50, margin=0.30 (max ∈ [0.45, 0.55) → 2-view TTA fires per Section 11.2)
- After 2-view: aggregated v3, LoRA produce more confident output
- Post-TTA classifier: argmax=0 (foliar), max=0.72, margin=0.50
- Conformal (τ=0.50): set={0}, size=1
- → **Tier 2**, T5 alert: False (rule 8c; argmax=0 foliar, not dangerous; late_blight prob ≈ 0.05 < 0.20)
- Walk: Rule 7 main IF fails (typically max<0.85 or margin<0.30 or psv_reliability<0.50 or chilli>=0.20). Rule 8 main IF met (max>=0.65, margin>=0.20, psv_reliability>=0.40, chilli<0.30, set_size==1). Sub-rules 8a (DEGRADED) and 8b (underpowered) fail. Sub-rule 8c default fires -> Tier 2.
- Note: this scenario pins argmax=foliar to make T5 deterministic. The TTA mechanism is class-agnostic; the test assertion is that 2-view TTA fires for max in [0.45, 0.55) and the post-TTA classifier output drives final tier assignment.

**STTA.2 — Initial 0.92 → no TTA**
- 1-view classifier: max=0.92 ≥ 0.55 → no TTA fires
- → **Tier 1** (assuming all other Tier 1 conditions met)
- Walk: all Rule 7 main conditions met (max>=0.85, margin>=0.30, psv_reliability>=0.50, chilli<0.20, IQA in {ACCEPTABLE,HIGH}, set_size==1, all signals OK). Sub-rules 7a (DEGRADED) and 7b (underpowered) both fail. Sub-rule 7c default fires -> Tier 1.

**STTA.3 — TTA fires (2-view) but doesn't change tier**
- 1-view classifier: argmax=0 (foliar), max=0.50, margin=0.30 (max ∈ [0.45, 0.55) → 2-view TTA fires per Section 11.2)
- 2-view aggregated v3 + LoRA: minor change in distribution
- Post-2-view classifier: argmax=0, max=0.52, margin=0.32 (small improvement; below Tier 2's 0.65 threshold)
- 5-view does NOT fire (post-2-view max 0.52 ≥ 0.45)
- Conformal: set={0}, size=1
- → **Tier 4A** (Rule 9 catch-all; max 0.52 fails Rule 4's `< 0.45`, fails Rule 7's `>= 0.85` and Rule 8's `>= 0.65`)
- Walk: 1-view max 0.50 triggers 2-view TTA. Post-TTA max 0.52 still doesn't reach Rule 8 threshold (0.65). With set size 1 (no Rules 5/6), the chain falls to Rule 9 → Tier 4A. The TTA reduced variance (initial wide CI, post-TTA narrower CI) but didn't push max into the Tier 2 zone. T5: argmax foliar → False.
- Note: TTA does not loop. Section 11.2 specifies that TTA fires at most once based on the 1-view max. Post-TTA classifier output is the final output for tier assignment.

**STTA.4 — TTA fires (5-view) → low-confidence multi-class → Tier 4A**
- 1-view classifier: argmax=0, max=0.40, margin=0.05 (max < 0.45 → 5-view TTA fires per Section 11.2)
- 5-view aggregated outputs disagree across views: post-TTA argmax=0, max=0.42, margin=0.05
- Conformal (τ=0.85): threshold 0.15; set has 4 classes
- → **Tier 4A** (Rule 4; max 0.42 < 0.45)
- Walk: 5-view TTA fired but aggregation didn't reach 0.45. The disagreement across TTA views is itself a signal of input difficulty. Tier 4A correctly flags low confidence.

**STTA.5 — 5-view TTA escalation resolves uncertainty**
- 1-view classifier: argmax=0, max=0.40, margin=0.10 (max < 0.45 → 5-view TTA fires per Section 11.2)
- 5-view aggregated outputs converge: post-TTA argmax=0, max=0.78, margin=0.55
- Conformal: set={0}, size=1
- → **Tier 2**, T5 alert: False (argmax foliar; rule 8c fires after TTA)
- Walk: 1-view alone would have routed to Tier 4A (max < 0.45). After 5-view TTA aggregated more samples, the classifier's confidence increased substantially. The single-class set + max 0.78 + margin 0.55 trigger Rule 8 → Tier 2. This scenario shows TTA's intended benefit: stabilizing predictions in the borderline zone.

### 15.16 Summary statistics, coverage matrix, and using this section as a behavioral test suite

**Tier coverage in this section:**

| Tier | Scenario count | Subsection |
|---|---|---|
| 1 | 12 | 15.3 |
| 2 | 12 | 15.4 |
| 3A | 12 | 15.5 |
| 3B | 10 | 15.6 |
| 3C | 12 | 15.7 |
| 3D | 10 | 15.8 |
| 4A | 13 | 15.9 (includes S4A.5 + S4A.5b for the two empty-set paths) |
| 4B | 10 | 15.10 |
| Tier 5 cross-base | 11 | 15.11 (extended with S5.9, S5.10, S5.11 for boundary and degraded-pipeline T5 cases) |
| Boundary | 15 | 15.12 (extended with SB.13, SB.14, SB.15 for chilli=0.20, margin=0 cases) |
| Underpowered | 7 | 15.13 (extended with SUP.7 for sub-rule 8a precedence) |
| Disagreement | 6 | 15.14 |
| TTA | 5 | 15.15 |
| **Total** | **135** | — |

**Tier 5 alert distribution:**
- T5 fires (alert: True): 51 scenarios
- T5 does not fire (alert: False): 81 scenarios
- Total scenarios with explicit T5 outcome: 132
- All 135 scenarios are now fully specified with deterministic T5 outcomes; the earlier "depends on" / "likely" pattern has been eliminated.

**Cross-tier patterns:**
- Tier 1 with T5: late_blight, mosaic, ylcv argmax cases (S1.3, S1.4, S1.5, S1.8)
- Tier 4B with T5: rare but possible (S4B.5, S4B.6, S4B.9, S5.4, S5.11)
- Tier 3C with T5: 3C from Rule 3 doesn't suppress T5 (S3C.7, S3C.11, SUP.4, S5.3)
- Tier 4A with T5 via in-set rule despite low max (S5.2, S5.8, S5.10, S4A.6, S4A.11)

---

**Rule-to-scenario reverse mapping:**

For developers debugging a specific rule's behavior or maintainers verifying that a rule change doesn't break tests:

| Rule | Sub-rule | Scenarios that exercise this path |
|---|---|---|
| Rule 1 (signal failure → 4B) | — | S4B.1, S4B.2, S4B.3, S4B.4, S4B.5, S4B.6, S4B.7, S4B.8, S4B.9, S4B.10, S5.4, S5.11, SB.9 |
| Rule 2 (IQA REJECT → gate) | — | not in this section; IQA gate handles before tier assignment (Section 6.4) |
| Rule 3 (PSV unreliable / chilli leak → 3C) | psv_reliability < 0.40 | S3C.1, S3C.2, S3C.6, S3C.7, S3C.10, S3C.11, SB.6, SUP.4, S5.3 |
| Rule 3 | chilli_leakage > 0.40 | S3C.3, S3C.4, S3C.5, SB.8 |
| Rule 4 (low max → 4A) | — | S4A.1, S4A.2, S4A.3, S4A.4, S4A.6, S4A.7 (boundary), S4A.9, S4A.10, S4A.11, S4A.12, S5.2, S5.7, S5.8, S5.9, S5.10, S3B.4, SB.10, SB.12, SB.14, SUP.6 |
| Rule 5 (set ≥ 3 → 3B) | size >= 3 | S3B.1, S3B.2, S3B.3, S3B.5, S3B.6, S3B.7, S3B.8, S3B.9, S3B.10, S5.1 |
| Rule 5 | size == 0 | S4A.5b, SB.11 |
| Rule 6 (set == 2 → 3A) | — | S3A.1, S3A.2, S3A.3, S3A.4, S3A.5, S3A.6, S3A.7, S3A.8, S3A.9, S3A.10, SDIS.1, SDIS.3, SDIS.4, SDIS.5, S5.5, SB.15 |
| Rule 7 (Tier 1) | 7c default | S1.1, S1.2, S1.3, S1.4, S1.5, S1.6, S1.7, S1.8, S1.9, S1.10, S1.11, S1.12, SB.1, SB.3, SUP.5 |
| Rule 7 | 7a (DEGRADED → 3D) | S3D.1, S3D.3, S3D.4, S3D.6, S3D.8, S3D.9, S3D.10, SUP.3, S5.6 |
| Rule 7 | 7b (underpowered → 3A) | S3A.11, SUP.1 |
| Rule 8 (Tier 2) | 8c default | S2.1 through S2.12, SB.2, SB.4, SB.5, SB.13 |
| Rule 8 | 8a (DEGRADED → 3D) | S3D.2 |
| Rule 8 | 8b (underpowered → 3A) | S3A.12, SUP.2 |
| Rule 8 | 8a precedence over 8b | SUP.7 |
| Rule 9 (catch-all → 4A) | — | S4A.7, S4A.8, S3C.12, SB.7 |

This mapping enables:
1. **Targeted regression testing.** Running only the scenarios that exercise a specific rule when that rule's threshold is changed.
2. **Coverage verification.** Confirming every rule and sub-rule has at least one positive test case.
3. **Implementation debugging.** When `assign_tier` produces an unexpected tier, the rule_id_fired in the structured reasons identifies which scenarios in the table apply.

---

**Coverage matrix (per-rule × class × signal-state):**

The 135 scenarios cover the following input-space cells:

| Dimension | Values | Coverage |
|---|---|---|
| Tier label | 1, 2, 3A, 3B, 3C, 3D, 4A, 4B | All 8 tiers covered (≥ 8 scenarios each except 3D=10) |
| Argmax class | foliar, septoria, late_blight, ylcv, mosaic, healthy, OOD | All 7 classes appear as argmax in at least one scenario |
| IQA decision | ACCEPTABLE, HIGH, DEGRADED, REJECT | First three covered; REJECT handled at Section 6.4 gate (out of scope for tier rules) |
| Signal failure | v3 only, LoRA only, PSV only, v3+LoRA, v3+PSV, all three | First five covered; "all three" not in this section because pipeline returns total fallback before tier rules run (per Section 11.2) |
| psv_reliability | < 0.40, [0.40, 0.50), >= 0.50 | All three buckets covered with boundary tests at 0.40 and 0.50 |
| chilli_leakage | < 0.20, [0.20, 0.30), [0.30, 0.40], > 0.40 | All four buckets covered with boundary tests at 0.20 (SB.13), 0.40 (SB.7), 0.41 (SB.8) |
| combined_max_prob | < 0.45, [0.45, 0.65), [0.65, 0.85), >= 0.85 | All four buckets covered with boundary tests at 0.45 (S4A.7), 0.65 (S2.2), 0.85 (S1.8, SB.1) |
| combined_margin | < 0.20, [0.20, 0.30), >= 0.30 | All three buckets covered with boundary tests at 0.20 (S2.9), 0.30 (S1.8, SB.3), 0 exact (SB.14, SB.15) |
| Conformal set size | 0, 1, 2, 3, 4, 5+ | Size 0 (SB.11, S4A.5b); 1, 2, 3 well covered; size 4 (S5.10); size 5 (S5.7); size 6+ (SB.10, SB.12) |
| Tier 5 alert | True, False | Both outcomes well covered (51 True / 81 False) |
| Underpowered guard | YLCV flagged, mosaic flagged, neither flagged | All three states covered (SUP.1-5, S1.4, S1.5 caveats) |

**Coverage gaps explicitly acknowledged below in "Limitations" subsection.**

---

**Behavioral test suite:**

The Phase F.0 validation script (Section 29) will encode each scenario as a test case:
```python
def test_scenario_S1_1():
    """Clean foliar prediction → Tier 1, no T5"""
    sa = make_signal_a_result(
        probs=[0.92, 0.04, 0.01, 0.01, 0.01, 0.01],
        chilli_leak=0.03,
        succeeded=True
    )
    sb = make_signal_b_result(
        probs=[0.88, 0.05, 0.02, 0.02, 0.02, 0.01],
        succeeded=True
    )
    sc = make_signal_c_result(
        argmax=0, max_prob=0.71, margin=0.45, reliability=0.78,
        succeeded=True
    )
    iqa = make_iqa_result(decision="ACCEPTABLE")
    classifier_result = make_classifier_result(
        argmax=0, max_prob=0.91, margin=0.86,
        p_final_calibrated=[0.91, 0.04, 0.02, 0.01, 0.01, 0.01, 0.00]
    )
    conformal_result = make_conformal_result(
        prediction_set={0}, tau=0.40
    )
    
    tier = assign_tier(
        classifier_result, conformal_result, iqa, sa, sb, sc
    )
    
    assert tier.tier_label == "1", f"Expected Tier 1, got {tier.tier_label}"
    assert tier.tier5_alert == False
    assert tier.rule_id_fired == "definitive_single_class"
    assert tier.sub_rule_id_fired == "default"

# Helper functions defined in F.0 test infrastructure (Section 29).
# Their precise signatures are an implementation detail.
```

Each scenario yields approximately 15-20 lines of test code; 135 tests = ~2700 lines of test code. The Phase F.0 build runs all 135 tests; CI fails on any mismatch. This catches rule-chain regressions before deployment.

---

**Conventions for IQA and other implicit defaults in tests:**

- IQA decision defaults to `ACCEPTABLE` when not explicitly stated. The aggregate score (Section 6.4) is not specified per scenario; the F.0 test infrastructure synthesizes an IQA result with the labeled decision and an aggregate score in the middle of the corresponding range (e.g., decision=ACCEPTABLE → score=0.75, decision=DEGRADED → score=0.55, decision=HIGH → score=0.92).
- Conformal `τ` is shown as scenario-specific to make the prediction set self-consistent. F.0 fits a single `τ` for the deployment; the test suite uses the F.0-fitted `τ` for the actual `assign_tier` call but reverse-engineers the classifier's full P_final_calibrated to produce the labeled set under that `τ`.
- Classifier output is shown as `argmax`, `max`, `margin` for tier rules and `P_final_calibrated` (full 7-class vector) for conformal. The vector sums to 1.0 in every scenario.
- "Underpowered" flags for YLCV and mosaic are scenario-specific. F.0 reports the actual flag values once at deployment; the test suite uses scenario-level overrides (mock the underpowered config) to exercise both code paths.

---

**Forward links to downstream sections:**

The tier outcomes documented here flow to:
- **Section 16 (response builder, future):** translates `tier_label` and `tier5_alert` into user-facing strings and structured reasons. Each tier's UI presentation is defined there.
- **Section 17 (severity grading, future):** for diseased classes in Tier 1/2/3, severity grades (mild/moderate/severe) further refine the response. Section 15 doesn't address severity.
- **Section 18 (multi-image, future):** aggregates tier outcomes across multiple images of the same plant. Section 15 assumes single-image input.
- **Section 23 (agronomist queue, future):** consumes Tier 5 alerts and routes flagged predictions to human review.
- **Section 25 (monitoring, future):** consumes structured reasons (rule_id_fired, sub_rule_id_fired) and tracks rule-firing frequencies; alerts on Rule 9 (catch-all) firing rate exceeding baseline.

---

**Limitations and known gaps (acknowledged):**

1. **Multi-image scenarios are minimal.** Section 18 (multi-image) is future. Section 15 mostly assumes single-image input; multi-image-specific tier behaviors will be added when Section 18 is written.
2. **Empty prediction set scenarios are explicit (S4A.5 via Rule 4, S4A.5b via Rule 5 sub-rule, SB.11 explicit case).** This is a rare case in practice (calibrated classifiers rarely produce empty sets when max_prob ≥ 0.45).
3. **NaN/inf handling depends on the orchestrator.** SB.9 documents the orchestrator-routes-to-4B contract per Section 11.2. The Phase F.0 test for SB.9 must verify the orchestrator marks signals as failed when classifier output contains NaN; without that, the rule chain alone would route NaN to 4A via catch-all.
4. **TTA scenarios are illustrative, not exhaustive.** Section 11's TTA controller has its own behavior that interacts with classifier output; full TTA × tier interaction is too complex to enumerate.
5. **IQA REJECT path is documented in Section 6.4, not Section 15.** REJECT short-circuits the entire pipeline at the IQA gate; no tier is assigned. The 135 scenarios in Section 15 all assume IQA != REJECT.
6. **Simultaneous PSV unreliable + IQA DEGRADED is not explicitly enumerated.** Per the rule chain, Rule 3 fires before Rule 7/8 sub-rules, so this combination produces Tier 3C, not Tier 3D. The Phase F.0 test infrastructure can compose the case from existing scenario inputs.
7. **Cross-signal disagreement scenarios (SDIS) describe JSD qualitatively in Walk traces.** Actual JSD is computed from v3 and LoRA distributions in the F.0 test suite; the displayed `JSD ≈ X` value is an estimate.
8. **Conformal calibration drift is out of scope for Section 15.** If F.0 calibration is bad (empirical coverage on test data deviates from 90%), the rule chain still operates correctly but the "definitive vs ambiguous" distinction skews. Section 25 (monitoring) tracks empirical coverage; Section 13 specifies re-calibration triggers.
9. **Multi-disease real-world inputs are not enumerated.** Real plants often have multiple co-occurring diseases. The system represents this only via conformal set ambiguity (Tier 3A/3B), which the user reads as "uncertain between A and B" rather than "both A and B present." This is an epistemological limitation of the tier framework, not a Section 15 gap. Section 30 (limitations) documents it.
10. **Adversarial / pathological inputs are deferred to F.0 dataset.** Pure black image, pure white, image of a pepper plant (similar foliage), image of multiple plants — these should produce OOD or low-confidence outputs. Section 15 has only one OOD scenario (S4A.10) as a synthetic construction. Real adversarial testing happens at F.0 data collection, not at scenario level.
11. **String sort ordering of scenario IDs.** S3B.10 sorts before S3B.2 in alphabetical order; similar issue for S4B.10, S4A.10, SB.10-15. Test runners that order tests alphabetically will execute them out of intended sequence. F.0 test infrastructure should zero-pad IDs at runtime (S3B.01, S3B.02, ..., S3B.10).
12. **Silent T5 misses on dangerous-disease + total pipeline failure.** If the actual content is late_blight but v3 + LoRA + PSV all produce unreliable output simultaneously, the classifier-degraded-mode might produce a non-dangerous argmax with low late_blight residual prob. Tier 4B + T5 = False. The system silently misses the danger. This is a real failure mode acknowledged here; partial mitigation: agronomist queue (Section 23) reviews Tier 4B cases. Full mitigation requires multi-image input (Section 18) or repeat-photo flow.
13. **Concurrency / state issues are out of scope.** Section 15 assumes each request is independent. Race conditions, mid-request model reloads, and cache layer behavior are addressed in Section 22 (sandbox server architecture).

These gaps will be addressed in subsequent spec turns and via F.0 validation runs on real data. The Phase F.0 test suite (Section 29) is responsible for filling in the gaps where they affect testability.

---

## Section 16. Response builder — translating tier outcomes into API responses

### 16.1 Purpose and dependencies

Section 14 produces a `TierAssignment` dataclass with `tier_label`, `tier5_alert`, `rule_id_fired`, `sub_rule_id_fired`, `reasons`, and `reasons_structured`. Section 16 specifies how that internal tier outcome becomes the JSON response that the client (frontend, mobile app, third-party integration) receives.

The response builder is a pure function: `build_response(tier_assignment, classifier_result, conformal_result, iqa_result, signal_a, signal_b, signal_c, request_metadata) -> ResponseDict`. It has no side effects and produces deterministic output for identical inputs.

The response builder reads from but does not modify:
- Section 14.7 `TierAssignment` (tier label and structured reasons)
- Section 12.10 `ClassifierResult` (combined argmax, max prob, margin, P_final_calibrated, GradCAM++ tensor)
- Section 13 `ConformalResult` (prediction set, tau, coverage)
- Section 6.4 `IQAResult` (decision, aggregate score, per-dimension scores)
- Section 7 `SignalCResult` (for severity grading; see Section 17)
- Request metadata (request_id, image_hash, timestamp, client version)

The response builder does NOT read the model weights or perform inference; all model-side computation has finished by the time `build_response` runs.

### 16.2 Response schema

The response is a JSON object with this top-level structure:

```json
{
  "request_id": "uuid-v4-string",
  "image_hash": "sha256-hex-string",
  "timestamp_iso": "2026-04-26T18:42:00Z",
  "tier": {
    "label": "1",
    "human_readable": "Definitive prediction",
    "alert_level": "info"
  },
  "prediction": {
    "primary_class": "foliar",
    "primary_class_human": "Foliar leaf spot",
    "primary_confidence": 0.91,
    "prediction_set": ["foliar"],
    "prediction_set_human": ["Foliar leaf spot"]
  },
  "tier5_alert": {
    "fired": false,
    "reason": null
  },
  "severity": {
    "grade": null,
    "human_readable": null,
    "details": null
  },
  "explanation": {
    "user_strings": ["The image clearly shows Foliar leaf spot..."],
    "structured": {
      "rule_id_fired": "definitive_single_class",
      "sub_rule_id_fired": "default",
      "psv_reliability": 0.78,
      "iqa_decision": "ACCEPTABLE"
    }
  },
  "visualization": {
    "gradcam_url": "/visualization/{request_id}/gradcam.png",
    "gradcam_target_class": "foliar"
  },
  "agronomist_queue": {
    "routed": false,
    "priority": null,
    "queue_id": null
  },
  "warnings": [],
  "model_version": "tomato-sandbox-v1.0.0",
  "processing_time_ms": 423
}
```

All fields are present in every response. Fields that don't apply to a tier (e.g., `severity.grade` for healthy/OOD, `tier5_alert.reason` when not fired) are set to `null` rather than omitted. This preserves a stable schema for clients.

The `severity` block content is computed and populated per Section 17 (severity grading). The `agronomist_queue` block is populated per Section 16.8 routing logic and consumed by Section 23 (agronomist queue, future).

The response is JSON-Schema-validated before sending. The schema lives in `tomato_sandbox/api/response_schema.json` and is loaded at server startup.

### 16.3 Per-tier user-facing strings

The `tier.human_readable` field maps each tier label to a short user-facing description:

| Tier | human_readable |
|---|---|
| 1 | Definitive prediction |
| 2 | Confident prediction |
| 3A | Two possible diseases |
| 3B | Multiple possible diseases |
| 3C | Image quality concern (segmentation or chilli leaf detection) |
| 3D | Image quality moderate; result less confident |
| 4A | Low confidence — manual review recommended |
| 4B | Pipeline issue — please retake or contact support |

**All user-facing strings (the `tier.human_readable` values, the `explanation.user_strings` templates, the severity `recommended_action` text) are subject to agronomic-team review at NanoFarm before deployment. The wording shown here and in the `templates.yaml` file is illustrative; the deployed wording may differ to match local agronomic terminology and to align with extension officer training materials.**

The `explanation.user_strings` field is a list of 1-3 sentences explaining the result in plain language. The strings are templates filled with the predicted class name and confidence level. Example for Tier 1 with foliar argmax:

> "The image clearly shows Foliar leaf spot with high confidence (91%). Recommended action: apply standard foliar spot treatment per Section 17 severity grading."

For Tier 3A with foliar vs septoria ambiguity:

> "The system cannot decide between Foliar leaf spot and Septoria leaf spot. Both diseases require similar treatment in early stages; differential treatment may be needed for later stages. Consider taking a closer photo of a mature lesion for clearer diagnosis."

For Tier 4A (low confidence):

> "The system has low confidence in this prediction. Possible reasons: atypical disease presentation, an unusual cultivar, or non-tomato content in the image. Manual review by an agronomist is recommended."

For Tier 4B (pipeline failure):

> "The disease detection pipeline encountered an issue while processing this image. Please retake the photo or contact support if the issue persists."

The full template strings are defined in `tomato_sandbox/responses/templates.yaml`. The response builder loads templates at startup and fills them per request.

### 16.4 Per-tier structured reasons (machine-readable)

The `explanation.structured` field exposes the tier-rule provenance for downstream tools (monitoring dashboards, analytics, agronomist UI):

```json
{
  "rule_id_fired": "definitive_single_class",
  "sub_rule_id_fired": "default",
  "tier_main_conditions": {
    "max_prob_threshold": 0.85,
    "max_prob_actual": 0.91,
    "margin_threshold": 0.30,
    "margin_actual": 0.86,
    "psv_reliability_threshold": 0.50,
    "psv_reliability_actual": 0.78,
    "chilli_leakage_threshold": 0.20,
    "chilli_leakage_actual": 0.03,
    "iqa_decision": "ACCEPTABLE",
    "set_size": 1
  },
  "tier_sub_rule_checks": {
    "iqa_degraded_check": false,
    "underpowered_class_check": false
  },
  "tier5_evaluation": {
    "argmax_dangerous_check": false,
    "late_blight_in_set_check": false
  }
}
```

The structured reasons let the monitoring service (Section 25) compute statistics like:
- "What fraction of Tier 1 predictions had max_prob in the [0.85, 0.90) range?"
- "Which sub-rule fires most often when Rule 7 main IF passes?"
- "How often does Tier 5 alert fire via in-set vs argmax bullets?"

The structured reasons schema is stable across model versions; rule IDs and sub-rule IDs come from `tomato_sandbox/rules/rule_ids.py` (an enum mirroring Section 14.5's rule chain).

### 16.5 GradCAM++ visualization attachment

GradCAM++ is computed by Section 12 (classifier) on Stage 3 of the ConvNeXt backbone. The output is a 2D heatmap at the input image resolution. Section 16 attaches the heatmap to the response:

```json
{
  "visualization": {
    "gradcam_url": "/visualization/{request_id}/gradcam.png",
    "gradcam_target_class": "foliar",
    "gradcam_alpha": 0.5
  }
}
```

The heatmap is overlaid on the original input image with `alpha=0.5` blend. The overlaid image is stored at `/var/lib/tomato_sandbox/visualizations/{request_id}/gradcam.png` with a 24-hour retention policy (configurable via `TOMATO_VIZ_RETENTION_HOURS`). The URL is served by the sandbox server at `/visualization/{request_id}/gradcam.png`.

For Tier 4B (pipeline failure), GradCAM++ is not generated and `visualization.gradcam_url` is `null`.

For Tier 3A and 3B (multi-class sets), GradCAM++ is generated for the argmax class only; a future enhancement (Section 30 limitations) is per-class GradCAM++ for each set member.

### 16.6 Confidence display rules

The user-facing confidence number is computed from `combined_max_prob` but with rounding and presentation rules:

- Display as percentage (multiply by 100, round half up to nearest integer): 0.91 -> "91%".
- For Tier 4A (low confidence), display as "below 45%" rather than the actual number, to avoid overstating precision.
- For Tier 4B, display "unknown" with no percentage.
- For Tier 3A and 3B (multi-class sets), display "between {first_class_confidence}% and {last_class_confidence}%" using the prediction set members.

The `prediction.primary_confidence` field uses the raw `combined_max_prob` as a float in [0, 1]. The percentage display is derived in the frontend; the response sends raw float values.

**These display rules apply ONLY to user-facing strings. The structured fields in the response (`prediction.primary_confidence`, `explanation.structured.tier_main_conditions.max_prob_actual`) always carry the raw float values. Frontend developers must not derive user-facing display from structured fields without applying these rules; analytics tools may consume raw float values directly.**

### 16.7 Tier 5 alert presentation

When `tier5_alert.fired` is `true`, the response includes:

```json
{
  "tier5_alert": {
    "fired": true,
    "reason": "argmax_dangerous_disease",
    "trigger_class": "late_blight",
    "trigger_probability": 0.92,
    "agronomist_priority_hint": "high"
  }
}
```

The `reason` field takes one of these enum values:
- `argmax_dangerous_disease` (first T5 bullet: argmax in {late_blight, mosaic, ylcv} AND max >= 0.20)
- `late_blight_in_set` (second T5 bullet: late_blight in prediction_set AND late_blight_prob >= 0.20)
- `argmax_dangerous_and_late_blight_in_set` (both bullets fire simultaneously; e.g., argmax=late_blight with max >= 0.20 and late_blight in set)

When both bullets fire, the reason is `argmax_dangerous_and_late_blight_in_set` and `trigger_probability` is the late_blight probability (which equals max_prob in the both-bullet case where late_blight is argmax).

The `agronomist_priority_hint` field is a hint that propagates to the agronomist queue's actual `priority` field (Section 16.8). The hint values:
- `high`: late_blight argmax with max >= 0.50; visual alert and sound notification on agronomist UI
- `medium`: any other T5 firing; visual alert only
- `null`: no T5 alert (consistent with `fired: false`)

The agronomist queue (Section 23, future) may override the hint based on queue capacity and recent priority distribution. The hint is not a hard contract.

### 16.8 Agronomist queue routing

Tier outcomes route to the agronomist queue per these rules:
- Tier 5 alert fires -> always routed (priority per Section 16.7)
- Tier 3A, 3B, 3C, 3D -> routed if `route_ambiguous_to_queue` flag is enabled (default false; agronomist capacity-dependent)
- Tier 4A -> routed if Tier 5 also fires; otherwise queued only on user opt-in
- Tier 4B -> NOT routed (pipeline issue, not a model uncertainty)
- Tier 1, 2 -> NOT routed unless Tier 5 also fires

The response includes:

```json
{
  "agronomist_queue": {
    "routed": true,
    "priority": "high",
    "queue_id": "agq-2026-04-26-00042"
  }
}
```

The `queue_id` is generated server-side at routing time. If `routed` is `false`, all other fields are `null`.

Section 23 (agronomist queue, future) defines the queue mechanics. Section 16 only emits the routing decision and metadata.

### 16.9 Error responses

For requests that fail before tier assignment can run (e.g., image decode failure, model loading error, server overload):

```json
{
  "request_id": "uuid-v4",
  "error": {
    "code": "IMAGE_DECODE_FAILED",
    "message": "The uploaded image could not be decoded. Please try a different image.",
    "retry_after_seconds": null,
    "support_contact": "tomato-support@nanofarm.in"
  },
  "tier": null,
  "prediction": null,
  "tier5_alert": null,
  "severity": null,
  "explanation": null,
  "visualization": null,
  "agronomist_queue": null,
  "warnings": [],
  "model_version": "tomato-sandbox-v1.0.0",
  "processing_time_ms": 12
}
```

Error codes:
- `IMAGE_DECODE_FAILED` (400): image bytes invalid or unreadable
- `IMAGE_TOO_LARGE` (413): image > 10 MB
- `IMAGE_UNSUPPORTED_FORMAT` (415): not JPEG/PNG/HEIC/WEBP
- `IQA_REJECTED` (422): IQA decision was REJECT (per Section 6.4); image too poor for analysis; suggest retake
- `MODEL_NOT_READY` (503): server starting up; retry after seconds
- `SERVER_OVERLOAD` (503): GPU lock timeout; retry after seconds
- `INTERNAL_ERROR` (500): unexpected server error; please contact support

The `tier` and downstream fields are `null` for error responses, distinguishing them from Tier 4B (pipeline failure, with tier set).

### 16.10 Backward compatibility with legacy APIN response

The legacy APIN service (Section 22, port 8766) for okra/brassica produces a different response schema. The unified server at port 8005 (which routes both okra/brassica through APIN and tomato through the sandbox) wraps both responses in a common envelope:

```json
{
  "crop": "tomato",
  "service_used": "tomato_sandbox",
  "service_version": "tomato-sandbox-v1.0.0",
  "response": { /* sandbox response per 16.2 */ }
}
```

For okra/brassica:

```json
{
  "crop": "okra",
  "service_used": "apin",
  "service_version": "apin-v1.8",
  "response": { /* APIN response per APIN_MODEL_CARD.md */ }
}
```

**Layering note.** The tomato sandbox server (port 8767) returns the unwrapped sandbox response per Section 16.2 directly; it does not produce the envelope. The unified server (port 8005) is responsible for adding the envelope when proxying the sandbox response to clients. Direct callers of port 8767 (e.g., the test infrastructure, internal tools) receive the unwrapped response. Callers of port 8005 (the public-facing endpoint) receive the wrapped envelope. This separation lets the sandbox stay independent of the unified server's wrapping logic.

The frontend reads the `crop` and `service_used` fields to dispatch to the appropriate response renderer. Section 19 (frontend, future) defines the renderers.

## Section 17. Severity grading

### 17.1 Purpose

For diseased classes (foliar, septoria, late_blight, ylcv, mosaic) at Tier 1, 2, or 3, the system grades severity as `mild`, `moderate`, or `severe`. Severity guides the recommended treatment intensity in the response. Healthy and OOD classes have no severity.

Severity is a coarse 3-bucket grading. It is intentionally coarse because:
- Fine-grained severity scores require ground-truth severity labels in the training set, which the training data does not have at sufficient scale.
- The downstream user (a farmer) takes treatment decisions at this granularity (no treatment / standard treatment / aggressive treatment).
- A coarser grade has lower variance across cultivars and growing conditions, making the system more reliable.

### 17.2 Severity inputs

Severity is computed from PSV features (Section 7) for the predicted argmax class. The features used here are a subset of Section 7's 26-feature output:

- `disease_coverage_pct`: percentage of leaf area affected, derived from the PSV disease mask. (Section 7 feature index G2.)
- `mean_lesion_intensity`: mean pixel intensity of the disease mask region in the LAB-CLAHE-preprocessed image. (Section 7 feature G3.)
- `lesion_count`: number of connected components in the disease mask above the minimum size threshold (`PSV_MIN_LESION_AREA_PX`, default 25 px). (Section 7 feature G4.)
- `lesion_size_distribution`: mean and standard deviation of connected-component sizes. (Section 7 features G7, G8.)
- `psv_reliability`: PSV's reliability score (also used by Tier 3C); low reliability degrades severity confidence. (Section 7 reliability output.)

The exact feature index mapping is in `scripts/specialist/psv_features.py`; Section 17 reads these features by name from `SignalCResult.features` (Section 7 dataclass). If the feature names in Section 7 are renamed, Section 17's reader must update accordingly.

The classifier output is NOT used for severity grading; severity is a PSV-only computation. This keeps severity orthogonal to the classifier's class confidence.

### 17.3 Per-disease severity thresholds

Severity thresholds are per-disease because what counts as "severe" differs by disease. For example, 5% leaf coverage is "severe" for late_blight (rapid progression) but "mild" for septoria (slow progression).

**These thresholds are placeholders for v1 deployment. Phase F.0 will replace them with values calibrated against agronomist-confirmed severity labels on a held-out subset. The agronomic team at NanoFarm must review the calibrated thresholds before they are used for clinical or treatment-decision purposes. The defaults below are conservative starting points based on agronomic literature.**

| Disease | Mild (coverage_pct, lesion_count) | Moderate | Severe |
|---|---|---|---|
| Foliar leaf spot | < 5%, 1-5 lesions | 5-15%, 5-15 | > 15% or > 15 lesions |
| Septoria leaf spot | < 8%, 1-10 | 8-25%, 10-25 | > 25% or > 25 lesions |
| Late blight | < 2%, 1-3 | 2-8%, 3-8 | > 8% or > 8 lesions |
| YLCV (Yellow Leaf Curl Virus) | < 10% leaf area showing curl | 10-30% | > 30% |
| Mosaic virus | < 15% leaf area showing mottling | 15-40% | > 40% |

YLCV and mosaic don't have lesion counts because their symptoms are diffuse (curling, mottling) rather than discrete lesions; only coverage matters.

The thresholds are F.0-derived starting points. Phase F.0 calibrates each threshold against agronomist-confirmed severity labels on a held-out subset; the defaults above are conservative for v1 deployment.

Threshold values are exposed as env vars:
- `TOMATO_SEVERITY_FOLIAR_MILD_PCT` (default 5)
- `TOMATO_SEVERITY_FOLIAR_MODERATE_PCT` (default 15)
- (similarly for each disease)

### 17.4 Severity in response

The `severity` block in the response:

```json
{
  "severity": {
    "grade": "moderate",
    "human_readable": "Moderate severity",
    "details": {
      "disease_coverage_pct": 12.4,
      "lesion_count": 8,
      "psv_confidence_in_severity": 0.78,
      "thresholds_used": {
        "mild_max": 5.0,
        "moderate_max": 15.0,
        "disease": "foliar"
      }
    },
    "recommended_action": "Apply standard fungicide treatment within 48 hours."
  }
}
```

The `recommended_action` field is a 1-2 sentence treatment recommendation per (disease, severity) pair, drawn from `tomato_sandbox/responses/treatment_templates.yaml`. The templates are reviewed by the agronomic team at NanoFarm before deployment.

### 17.5 Severity for multi-class sets (Tier 3A, 3B)

For Tier 3A and 3B (the prediction set has 2+ classes), severity is computed for each class in the set and reported as a list:

```json
{
  "severity": {
    "grade_per_class": [
      { "class": "foliar", "grade": "moderate", "coverage_pct": 11.2 },
      { "class": "septoria", "grade": "mild", "coverage_pct": 4.8 }
    ],
    "human_readable": "Possible foliar (moderate) or septoria (mild)",
    "recommended_action": "Treat for the more severe possibility (foliar at moderate) until clearer diagnosis."
  }
}
```

The `recommended_action` defaults to the grade that maps to the most aggressive treatment template across the set, since under-treatment is more costly than over-treatment in most agronomic contexts. The treatment templates are ordered by aggressiveness: "no action needed" < "monitor and re-photograph in 48h" < "standard treatment" < "aggressive treatment" < "consult agronomist immediately". Comparison across diseases uses this ordering, not raw severity grade. For example, "moderate foliar" maps to "standard treatment" while "mild septoria" maps to "monitor and re-photograph"; the standard treatment recommendation wins. The agronomic team can override this default per (disease combination, region) if needed.

**Multi-image and multi-class interaction.** When a multi-image request (Section 18) lands at Tier 3A or 3B, severity is computed on the aggregated outputs (aggregated argmax + aggregated set + aggregated PSV features per Section 18.4). If aggregation cannot produce reliable PSV features (e.g., disagreement across images on PSV reliability), severity falls back to "computed from the highest-confidence image's PSV features" with a warning in `warnings`. Phase F.0 will validate which aggregation strategy produces more accurate severity grades against agronomist-confirmed labels.

### 17.6 Healthy and OOD have no severity

For tier with argmax = healthy or OOD:

```json
{
  "severity": {
    "grade": null,
    "human_readable": "Not applicable (plant appears healthy)" or "Not applicable (image content unclear)",
    "details": null,
    "recommended_action": "No action needed." or "Please ensure the image shows a tomato leaf clearly."
  }
}
```

### 17.7 Severity uncertainty (when to omit)

Severity is omitted (set to `null` with `human_readable` explaining why) in these cases:

- Tier 4A (low confidence): the classifier is too uncertain about the disease class for severity to be meaningful.
- Tier 4B (pipeline failure): PSV may have failed; severity inputs unavailable.
- `psv_reliability < 0.50`: even if the classifier is confident, low PSV reliability means the disease coverage measurement is noisy.
- Disease coverage < 1%: very small coverage may be a single lesion or noise; severity grading is unreliable below 1%.

In these cases, the response includes:

```json
{
  "severity": {
    "grade": null,
    "human_readable": "Severity could not be reliably graded.",
    "details": { "reason": "low_psv_reliability", "psv_reliability": 0.42 },
    "recommended_action": "Consult agronomist for detailed severity assessment."
  }
}
```

### 17.8 Honest limitations

Severity grading has known limitations:

1. **No ground-truth severity labels in training data.** The thresholds in 17.3 are agronomic best-guesses, not learned from data. Phase F.0 will calibrate against expert-labeled samples but the initial deployment uses defaults.
2. **PSV mask quality is variable.** When the leaf is well-segmented, severity is reliable. When PSV's mask covers background or non-leaf tissue, severity is wrong.
3. **YLCV and mosaic severity is not lesion-based.** Their visual symptoms (curling, mottling) are harder to segment; coverage estimates are less reliable than for lesion-based diseases.
4. **Severity does not integrate stage of disease.** Early-stage late blight and late-stage late blight may both register as "moderate" but have different urgency. Stage-of-disease is a future enhancement.
5. **Single-image limitation.** Severity is computed from one image (per Section 17). Multi-image (Section 18) can aggregate severity across images of the same plant.

These limitations are documented in Section 30 (limitations) and should be surfaced to the agronomist team before reliance on severity-based decisions.

## Section 18. Multi-image input

### 18.1 Purpose and motivation

A single leaf photo gives limited context. Real diagnostic work often combines multiple views: close-up of a lesion, whole-leaf view, whole-plant view, and the underside of the leaf. Multi-image input lets the user upload 1-5 photos of the same plant; the system aggregates per-image predictions into a final tier.

Multi-image is opt-in: the API accepts both single-image and multi-image requests. Single-image is the default and used by the simple mobile flow. Multi-image is intended for trained users (extension officers, agronomists) and for cases where the user's first photo received a Tier 3 or Tier 4 tier (the system asks for additional photos).

### 18.2 Input contract

The request body for multi-image:

```json
{
  "images": [
    {
      "image_id": "img-1",
      "image_bytes_base64": "...",
      "view_label": "close_up_of_lesion"
    },
    {
      "image_id": "img-2",
      "image_bytes_base64": "...",
      "view_label": "whole_leaf"
    },
    {
      "image_id": "img-3",
      "image_bytes_base64": "...",
      "view_label": "whole_plant"
    }
  ],
  "metadata": {
    "request_id": "uuid",
    "client_version": "...",
    "user_intent": "ambiguous_first_photo_followup"
  }
}
```

Constraints:
- `images` length is between 1 and 5 (server returns 400 if 0 or > 5).
- Each image is base64-encoded JPEG/PNG/HEIC/WEBP; max 10 MB after decode.
- `view_label` is optional; if present, takes one of: `close_up_of_lesion`, `whole_leaf`, `whole_plant`, `underside_of_leaf`, `other`. The view label hints aggregation but does not change per-image processing.
- All images must be of the same plant. The server does not verify this; the user attests by uploading.

### 18.3 Per-image processing

Each image runs through the full single-image pipeline independently:
- IQA gate
- Signal A (v3), Signal B (LoRA), Signal C (PSV)
- TTA (Section 11) fires per-image based on per-image classifier output; each image's TTA decision is independent of other images in the same multi-image request
- Classifier
- Conformal prediction set
- Tier assignment

This produces a per-image `TierAssignment` for each of the 1-5 images. Per-image results are also returned in the response (not just the aggregate) so the user can inspect each image's analysis.

Per-image processing runs in parallel using a shared GPU worker pool (Section 22, future). The total time is roughly `max(per_image_time)` rather than `sum(per_image_time)` for 2-3 images; with 4-5 images the GPU-worker contention may add overhead.

### 18.4 Aggregation strategy

The aggregation produces a single final tier from N per-image tiers. The strategy is conservative (favor the more cautious outcome) and preserves Tier 5 alerts whenever any image fires T5.

**Pre-step: special cases.**
- N=1 multi-image is equivalent to single-image processing; the aggregation logic below degenerates to passing through the single image's outputs.
- Image_ids must be unique within a request; the server returns 400 with `error.code = "DUPLICATE_IMAGE_IDS"` for duplicates.
- IQA REJECT per-image is treated like a pipeline failure (Tier 4B equivalent): the rejected image is excluded from class voting and conformal aggregation. A warning is added.
- If ALL images fail IQA REJECT, the response is a single aggregated `IQA_REJECTED` error per Section 16.9 (no tier assigned).

**Step 1: Tier 5 alert aggregation.**
- Final `tier5_alert.fired` = `True` if ANY per-image tier has `tier5_alert.fired == True`.
- Final `tier5_alert.reason` = the reason from the per-image alert with the highest `trigger_probability`, or `mixed_across_images` if multiple images fire T5 with different reasons.
- Final `tier5_alert.trigger_class` = the trigger class from the highest-probability T5 firing.

**Step 2: Pipeline failure / IQA REJECT aggregation.**
- If ALL per-image tiers are 4B (all pipelines failed) OR all images failed IQA REJECT, final tier is 4B (or aggregated IQA_REJECTED error if exclusively IQA failures).
- If SOME per-image tiers are 4B / IQA REJECTED and others are non-failures, the failed images are excluded from class aggregation; final tier is computed from the successful images. A warning is added to `warnings`.

**Step 3: Class voting.**
- Collect each successful image's `prediction.primary_class`.
- Compute weighted votes: each image's vote is weighted by its `prediction.primary_confidence`.
- Pick the class with the highest total weighted vote as the final argmax.
- Compute final `combined_max_prob` as the weighted mean of per-image max probabilities for the winning class only (images that voted for other classes are excluded from this mean).
- Compute final `combined_margin` as the weighted mean of per-image margins for the winning class only.

**Step 4: Conformal set aggregation.**
- Compute the union of per-image prediction sets.
- For each class in the union, compute the fraction of images that included it in their set.
- Final prediction set = classes with fraction >= 0.50 (i.e., majority of successful images agreed). A class included in exactly half of successful images is admitted (boundary inclusive).

**Step 5: Aggregated IQA decision.**
- Aggregated IQA decision = the WORST per-image IQA decision across successful images, where the ordering is HIGH < ACCEPTABLE < DEGRADED < REJECT (REJECT excluded since those images are excluded from successful).
- Rationale: the aggregated tier should reflect the worst image's quality concern, since DEGRADED IQA caps the tier at 3D.

**Step 6: Aggregated PSV reliability and chilli leakage.**
- Aggregated `psv_reliability` = minimum across successful images' PSV reliability.
- Aggregated `chilli_leakage` = maximum across successful images' chilli leakage.
- Rationale: the most conservative value across images drives the rule chain; this matches the principle "any image's concern propagates to the aggregate."

**Step 7: Final tier assignment.**
- Run the standard tier rule chain (Section 14.5) using the aggregated values from Steps 3-6.
- The aggregated tier is reported as the final tier in the response.

**Disagreement detection.**
- If the top-voted class has weighted-vote share < 0.50, the request is flagged as "strong-disagreement-among-images" in `warnings`. The 0.50 threshold is a starting heuristic; F.0 may sweep this on real disagreement data and adjust.

### 18.5 Final tier from aggregated outputs

The final tier follows the same rule chain as single-image tier assignment. The aggregated values are passed through `assign_tier()` as if from a single image. This keeps the rule chain unchanged across single-image and multi-image flows.

A subtle effect of aggregation: when 2 of 3 images give Tier 1 and 1 image gives Tier 2, the aggregated max_prob is between the two thresholds, which can land at Tier 1 (if confidence is high enough) or Tier 2 (if confidence is borderline). This is intended; the aggregated tier reflects the consensus confidence.

### 18.6 Disagreement handling

Multi-image requests where per-image predictions strongly disagree are handled as follows:

- **Strong disagreement on argmax**: if the top-voted class has weighted-vote share < 0.50, the system flags the request as "disagreement-among-images" in `warnings` and reports the final tier per the standard rule chain (which often lands at Tier 3 or 4 due to the spread).
- **Disagreement on Tier 5 alert**: T5 fires if ANY image fires it (Step 1 above). This is conservative; we'd rather over-alert than miss late_blight.
- **Disagreement on severity**: severity is computed per-image; the aggregated severity follows the more-severe grade across images that agreed on the argmax class.

The response includes per-image breakdowns under `multi_image_per_image_results`:

```json
{
  "multi_image_per_image_results": [
    {
      "image_id": "img-1",
      "tier": "1",
      "primary_class": "foliar",
      "primary_confidence": 0.91,
      "tier5_alert_fired": false
    },
    {
      "image_id": "img-2",
      "tier": "3A",
      "primary_class": "foliar",
      "primary_confidence": 0.55,
      "tier5_alert_fired": false
    }
  ]
}
```

### 18.7 Tier 5 alert across multiple images

The aggregation in Step 1 above ensures any T5 firing in any image is preserved in the final response. This is the primary safety property of the multi-image flow: more images cannot mask a danger signal.

The conservative T5 aggregation has a side effect: if 1 of 5 images is mislabeled by the model (e.g., mistakes a healthy patch for late_blight at low confidence), T5 fires in the final response even though 4 of 5 images are clean. The agronomist queue handles this (Section 23 prioritization) by lowering priority when the T5 firing is from a single image of many.

### 18.8 Performance budget

Multi-image latency targets (P95):

| N images | GPU compute (parallel) | Total request time | Acceptable? |
|---|---|---|---|
| 1 | 400 ms | 800 ms | yes |
| 2 | 600 ms | 1.0 s | yes |
| 3 | 800 ms | 1.3 s | yes |
| 4 | 1.0 s | 1.6 s | tight |
| 5 | 1.4 s | 2.0 s | borderline |

Beyond 5 images the request is rejected. Latency degrades with N because GPU memory pressure increases (concurrent Stage-3 forward passes contend for VRAM on the 8 GB RTX 4060).

The per-image timeout is 5 seconds; if any image takes longer, that image is marked failed (4B) and excluded from aggregation. If all images time out, the request returns Tier 4B with `error.code = "MULTI_IMAGE_TIMEOUT"`.

**Fallback if production latency exceeds budget.** The N=5 limit is set conservatively for v1 deployment. If P95 latency exceeds 2.5 seconds in production for N=5 (measured by Section 25 monitoring), the limit will be reduced to N=4 via the `TOMATO_MULTI_IMAGE_MAX_N` env var (default 5). Reducing the limit is a config change, not a code change. If P95 exceeds 2.5 seconds at N=4 as well, the limit reduces to N=3. The frontend (Section 19, future) reads the configured limit at session start and adjusts the upload UI accordingly.

### 18.9 API contract changes

The single-image API path remains: `POST /predict` with a single image bytes field. The multi-image path is `POST /predict_multi` with the JSON body in 18.2. Both paths return the same response schema (16.2), with multi-image responses additionally including `multi_image_per_image_results`.

Multi-image is currently tomato-only. The okra/brassica services (APIN, port 8766) do not support multi-image input in v1; the unified server's envelope (Section 16.10) sets `crop: "tomato"` for multi-image responses. A future enhancement may extend multi-image to other crops if APIN gains the same aggregation logic.

The frontend (Section 19, future) chooses the appropriate path based on user input. The API documentation (Section 27, future) lists both endpoints.

### 18.10 Limitations

1. **Aggregation assumes images are of the same plant.** The server does not verify this; the user attests. If images are of different plants, aggregation produces nonsense results.
2. **No view-label-aware weighting.** All images are weighted equally except by per-image confidence. A future enhancement could weight `whole_plant` views less than `close_up_of_lesion` for severity grading.
3. **Disagreement detection is coarse.** Strong disagreement (vote share < 0.50) is flagged, but moderate disagreement (vote share 0.50-0.70) is reported without warning, even though it may indicate genuine ambiguity worth investigating.
4. **Tier 5 conservative aggregation may over-alert.** A single false-positive T5 from one image triggers final T5. Agronomist queue prioritization (Section 23) handles this but doesn't eliminate it.
5. **Latency at N=5 is borderline.** Real production may need to drop to N=3 or N=4 max via the `TOMATO_MULTI_IMAGE_MAX_N` env var; the limit will be revisited after Phase F.0 latency measurements.
6. **No multi-image training.** The classifier and tier rules are trained on single images only; multi-image is an inference-time aggregation, not a learned multi-image model. A future enhancement is a multi-image classifier that consumes all images jointly.
7. **Conformal coverage guarantee does not extend to aggregated sets.** Conformal calibration in Section 13 fits tau on a single-image calibration set; the 90% coverage guarantee applies per-image. The aggregated set in Step 4 (fraction-based union with 0.50 threshold) is a heuristic with no formal coverage guarantee. The aggregated set is more permissive than a per-image set in some cases (admitting any class that appears in >= 50% of images, even at lower per-image confidence) and more conservative in others (requiring majority agreement). Phase F.0 should measure aggregated-set empirical coverage on multi-image validation data; if coverage deviates significantly from 90%, the spec will revise the aggregation rule.
8. **Aggregation is independent of sequential intent.** Multi-image is treated as a set of photos, not a sequence (e.g., before/after treatment). Sequential intent is out of scope for v1.

These limitations are documented in Section 30 and should be communicated to extension officers and agronomists before they rely on multi-image for high-stakes decisions.

---

## Section 19. Frontend integration

### 19.1 Purpose and scope

Section 19 specifies how the frontend (web app at `claude-frontend/` and mobile app at `claude-mobile/`) consumes the API response from Section 16 and renders it for the user. The frontend is the only system component that translates structured tier outcomes into actionable user experience; the API and sandbox produce structured data and assume the frontend handles presentation.

This section defines:
- Per-tier UI rendering rules
- Per-tier user actions (what the user can do next)
- Image upload UI (single and multi-image)
- Loading and error states
- GradCAM++ overlay rendering
- Tier 5 alert UI
- Severity display
- Forward contract: frontend changes that v1 does NOT implement

The frontend is not part of the sandbox directive; it lives in its own repo. Section 19 is reference material for the frontend team and the agronomist UI team.

### 19.2 Per-tier UI rendering rules

The frontend receives the response described in Section 16.2 and renders one of these layouts based on `tier.label`:

**Tier 1 (definitive prediction):**
- Large green badge: "Definitive: {primary_class_human}"
- Confidence: "{primary_confidence}% confident"
- Severity badge (from Section 17): mild/moderate/severe with color coding
- Recommended action: shown prominently
- GradCAM++ overlay: visible by default
- Action buttons: "Mark this as treated" + "Re-photograph for follow-up"

**Tier 2 (confident prediction):**
- Yellow badge: "Confident: {primary_class_human}"
- Confidence: "{primary_confidence}% confident"
- Severity, recommended action, GradCAM++ same as Tier 1
- Additional info text: "If symptoms differ from typical {primary_class_human}, consider taking another photo"
- Action buttons: same as Tier 1 + "Get second opinion (agronomist)"

**Tier 3A (two-class ambiguity):**
- Orange badge: "Possibly {first_class_human} or {second_class_human}"
- Confidence: "between {first_confidence}% and {second_confidence}%"
- Per-class severity displayed side by side
- Recommended action: from `severity.recommended_action` (most aggressive treatment)
- GradCAM++ overlay: visible for argmax class
- Action buttons: "Take a closer photo to disambiguate" + "Get second opinion"

**Tier 3B (multi-class ambiguity):**
- Orange badge: "Multiple possibilities: {first_class_human}, {second_class_human}, etc."
- Per-class display in a list
- Recommended action: most aggressive from set
- Action buttons: same as 3A

**Tier 3C (PSV unreliable / chilli leakage):**
- Yellow badge: "Image quality concern"
- Explanation text from `explanation.user_strings`
- Suggestion: "The leaf appears too small in the frame, or the image may show non-tomato content"
- Action buttons: "Re-photograph (close-up of leaf)" + "Continue anyway with low confidence"

**Tier 3D (DEGRADED IQA cap):**
- Yellow badge: "Image quality moderate"
- Show the prediction with reduced confidence
- Suggestion: "For higher confidence, retake the photo with better lighting and focus"
- Action buttons: "Re-photograph" + "Continue with current result"

**Tier 4A (low confidence):**
- Gray badge: "Low confidence — manual review recommended"
- Explanation: from `explanation.user_strings`
- Action buttons: "Send to agronomist for review" + "Take more photos (multi-image flow)"

**Tier 4B (pipeline failure):**
- Red badge: "Pipeline issue"
- Explanation text + retry button
- Action buttons: "Retry" + "Contact support"

The badge colors follow a traffic-light convention common in user-facing decision support: green/yellow/orange/red mapping to definitive/confident/uncertain/error. The agronomist UI uses the same colors for consistency.

### 19.3 Tier 5 alert UI

When `tier5_alert.fired` is `true`, the frontend renders an additional alert banner ABOVE the tier badge:

- High priority (`agronomist_priority_hint == "high"`): red banner with sound notification (where browser/OS permits) and visual flash; text "URGENT: {trigger_class_human} detected"
- Medium priority (`agronomist_priority_hint == "medium"`): orange banner with vibration where supported; text "Alert: {trigger_class_human} possible"

Sound notification depends on browser/OS permission and may degrade to vibration-only on devices that block audio (mobile web on iOS, for example). The frontend gracefully degrades: if sound fails, the visual flash and vibration are sufficient to surface the alert.

The banner persists until the user acknowledges it (taps to dismiss). On the agronomist UI, the banner stays until the agronomist marks the case as reviewed.

The banner does not replace the tier badge; both are shown simultaneously. This is intentional: a user might dismiss the T5 banner but still need to see the tier outcome.

### 19.4 Severity display

The `severity` block from Section 17 renders as:

- For `grade` in {`mild`, `moderate`, `severe`}: colored bar (green, yellow, red) with text `human_readable`
- For `grade == null`: muted text from `human_readable` explaining why severity is unavailable
- For multi-class severity (`grade_per_class` populated): a small table with class | grade | coverage_pct columns

The `recommended_action` text is shown prominently below the severity bar in a styled box. This is the primary user-actionable output.

### 19.5 GradCAM++ overlay

GradCAM++ overlay rendering:

- Default: 50% alpha overlay on the original image (matches `gradcam_alpha` from Section 16.5)
- User can toggle alpha via a slider in [0%, 100%]
- User can toggle the overlay on/off
- For Tier 4B: GradCAM++ is null; show a placeholder "Visualization unavailable due to pipeline issue"
- For Tier 3A/3B: overlay is for the argmax class only; UI shows a note "Visualization shows evidence for {argmax_class_human} only; {other_class_human} may also be present"

### 19.6 Image upload UI

**Single-image upload:**
- File picker (mobile: camera + gallery; web: file dialog or drag-and-drop)
- Image preview before submission
- Crop / rotate tools (frontend-side, no server interaction)
- Submit button triggers `POST /predict`

**Multi-image upload (Section 18):**
- The frontend can prompt users to add more photos after Tier 3/4 outcomes; this is a recommended pattern for the v1 UI, not a strict spec requirement
- The user uploads up to 4 additional images (total 5 with the first; configurable via `TOMATO_MULTI_IMAGE_MAX_N`)
- Each image gets a `view_label` selector (close_up_of_lesion / whole_leaf / whole_plant / underside / other)
- Submit button triggers `POST /predict_multi`

The frontend reads the `multi_image_max_n` value from the sandbox's `/info` endpoint (Section 20.3) at session start and adjusts the max-images selector accordingly. If the server reduces the limit mid-session (operations action), the frontend revalidates on submit and shows a clear error if over the limit.

### 19.7 Loading and error states

**Loading state (during request):**
- Progress indicator with rough estimate ("about 1 second" for single-image, "2-3 seconds" for multi-image)
- Cancellation button — frontend aborts the HTTP request; server-side cleanup happens via Section 22 timeout handling
- For requests > 5 seconds, show an "It's taking longer than expected" message; do not show this for multi-image where 5s is normal

**Error states:**
- 4xx errors (Section 16.9): show the `error.message` field directly to the user; offer specific retry/correction actions per error code
- 5xx errors: generic "Something went wrong" with retry button; the actual error message is logged but not shown (avoids exposing server internals). Retry is user-initiated; the frontend never retries automatically (avoids amplifying server load during outages and respects user agency over whether to retry)
- Network errors: "No connection — please try again"
- **Unknown enum values from server** (e.g., a future spec adds `tier.label = "5"` that this frontend version does not recognize): the frontend shows a generic "Unsupported response — please update the app or contact support" message rather than crashing. The structured response is preserved in the response logs for debugging.

### 19.8 Forward contract for v2 frontend

The v1 frontend renders the response per the rules above. The v2 frontend may add:

1. **Per-class GradCAM++ for multi-class sets** (Section 16.5 limitation): show one heatmap per class in the prediction set
2. **Sequential photo flow** (Section 18.10 limitation): "before treatment" + "after treatment" comparison
3. **Disease-progression timeline**: show how the same plant's predictions evolved across multiple visits
4. **Treatment outcome tracking**: user marks "treated successfully" / "still showing symptoms" / "got worse" → feedback loop to F.0 retraining
5. **Offline mode**: capture photos offline, sync when network available

These v2 features are out of scope for v1; the v1 spec assumes only the rendering rules in 19.2-19.7.

### 19.9 Limitations

1. **No A/B testing of UI variants in v1.** The badges, colors, and text are fixed; v2 may introduce experimentation infrastructure.
2. **Accessibility partially addressed.** The frontend uses high-contrast badges and ARIA labels for screen readers, but full WCAG 2.1 AA compliance is a v2 goal.
3. **Localization deferred.** v1 is English-only. v2 will add Malayalam, Tamil, Hindi, and Kannada based on agronomist feedback on usage patterns.
4. **No user-side cache.** Each photo is sent to the server fresh; there is no client-side prediction cache. The server-side cache (Section 22.5) handles repeat-image requests.

## Section 20. Sandbox server architecture

### 20.1 Purpose and scope

Section 20 specifies the internal architecture of the tomato sandbox server (port 8767). The sandbox runs the full tomato pipeline: validation gate → IQA → preprocessing → Signal A/B/C → TTA → classifier → conformal → tier assignment → response builder. It returns the unwrapped response per Section 16.2.

The sandbox is a FastAPI application. It is the only new server introduced for tomato; the legacy APIN server (port 8766) is unchanged, and the unified server (port 8005, Section 22) routes between them.

### 20.2 Process model

The sandbox runs as a single uvicorn process with `--workers 1 --loop asyncio`. On the RTX 4060 development hardware, single-process is required because:

- Models are loaded once into GPU memory; multiple workers would multiply VRAM use beyond the 8 GB RTX 4060 budget
- Per-request GPU lock (Section 20.6) requires a single arbiter

In production deployments with larger GPU memory (e.g., A10 with 24 GB or A100 with 40 GB), multiple workers may share the GPU; this would require revisiting the GPU lock design. v1 deployment targets the laptop/single-GPU configuration.

Concurrency within the process uses asyncio: HTTP request handling is non-blocking, but GPU compute is serialized via the GPU lock. CPU-side preprocessing (image decode, IQA classical features, PSV CPU features) runs in a thread pool to avoid blocking the event loop.

The sandbox is not horizontally scalable in v1. If load grows, v2 will add a model-server abstraction (e.g., Triton) that allows multi-process replication with shared model weights.

### 20.3 Endpoints

The sandbox exposes these HTTP endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Single-image prediction (Section 16) |
| `/predict_multi` | POST | Multi-image prediction (Section 18) |
| `/visualization/{request_id}/gradcam.png` | GET | Serve GradCAM++ overlay images (Section 16.5) |
| `/health` | GET | Liveness check; returns 200 if model loaded and GPU available |
| `/ready` | GET | Readiness check; returns 200 if calibration files loaded and GPU lock acquirable |
| `/metrics` | GET | Prometheus-format metrics (Section 25) |
| `/info` | GET | Model version, build hash, calibration timestamps |

The `/info` endpoint returns:

```json
{
  "service": "tomato_sandbox",
  "service_version": "tomato-sandbox-v1.0.0",
  "build_hash": "abc123def456",
  "models": {
    "v3_version": "model2_production_v1.0",
    "lora_version": "lora_v1.0",
    "psv_version": "psv_v1.0",
    "classifier_version": "classifier_v1.0"
  },
  "calibration": {
    "conformal_tau": 0.42,
    "conformal_calibration_timestamp": "2026-04-26T10:00:00Z",
    "iqa_thresholds_timestamp": "2026-04-26T10:00:00Z"
  },
  "config": {
    "multi_image_max_n": 5,
    "tta_trigger_threshold": 0.55,
    "gpu_lock_timeout_s": 10
  }
}
```

This endpoint is consumed by:
- The frontend (Section 19.6) to read `multi_image_max_n` at session start
- The unified server (Section 22.6) to display version metadata in `/health` aggregation
- Operations team for debugging and auditing

The `/predict` and `/predict_multi` endpoints are the only ones that consume GPU. The others serve static or quickly-computed data.

There is no admin endpoint for live config changes in v1; config changes require a process restart.

### 20.4 Module layout

The sandbox lives in `tomato_sandbox/` per the Sandbox Directive. Module structure:

```
tomato_sandbox/
├── api/
│   ├── server.py                # FastAPI app entrypoint
│   ├── request_models.py        # Pydantic models for request bodies
│   ├── response_models.py       # Pydantic models for response shape
│   └── response_schema.json     # JSON Schema (Section 16.2)
├── orchestrator/
│   ├── pipeline.py              # Main orchestrator (Section 21)
│   ├── degraded_mode.py         # Signal-failure handling (Section 12.7)
│   └── nan_guards.py            # NaN detection contract (Section 11.2)
├── signals/
│   ├── signal_a.py              # v3 wrapper (calls Section 8 model)
│   ├── signal_b.py              # LoRA wrapper (calls Section 9 model)
│   └── signal_c.py              # PSV wrapper (calls Section 7 module)
├── classifier/
│   ├── feature_vector.py        # 19-dim feature builder (Section 12.2)
│   ├── classifier.py            # MLP/logistic classifier (Section 12)
│   ├── conformal.py             # Conformal prediction (Section 13)
│   ├── calibration.py           # Calibration loader (Section 12.10)
│   └── tta.py                   # TTA controller (Section 11)
├── iqa/
│   └── iqa.py                   # IQA gate (Section 6)
├── tier/
│   ├── rules.py                 # Tier rule chain (Section 14.5)
│   ├── tier_assignment.py       # assign_tier function (Section 14.8)
│   └── rule_ids.py              # Enum of rule IDs (Section 16.4)
├── responses/
│   ├── builder.py               # build_response function (Section 16)
│   ├── templates.yaml           # User-facing string templates (Section 16.3)
│   └── treatment_templates.yaml # Treatment recommendation templates (Section 17.4)
├── severity/
│   └── grader.py                # Severity grading (Section 17)
├── multi_image/
│   └── aggregator.py            # Multi-image aggregation (Section 18.4)
├── infra/
│   ├── gpu_lock.py              # GPU lock management
│   ├── cache.py                 # Sandbox-internal request cache (Section 21.10)
│   ├── visualizations.py        # GradCAM++ image serving + retention
│   ├── config.py                # Env var loading
│   └── logging_setup.py         # Structured logging
├── monitoring/
│   ├── metrics.py               # Prometheus metrics (Section 25)
│   └── tracing.py               # Distributed tracing hooks
├── storage/
│   └── sqlite_logger.py         # Phase E logging (Section 24)
└── config.py                    # Application config
```

This layout strictly contains all new code under `tomato_sandbox/`. No file outside this directory is modified by sandbox development.

### 20.5 Startup sequence

On startup the sandbox performs these steps in order:

1. Load env vars (`TOMATO_*` namespace per Section 4.5)
2. Initialize structured logging
3. Bind PyTorch to GPU device 0 and verify CUDA is available; if no GPU available, log error and exit
4. Load v3 model weights from `model2_production.pt` to GPU
5. Load LoRA model weights to GPU
6. Load PSV module (CPU-only; no GPU memory use)
7. Load classifier weights from configured path
8. Load conformal calibration from `tomato_calibration.json` (sandbox-specific calibration file produced by Phase F.0)
9. Load IQA reference distributions
10. Validate all env var thresholds against expected ranges
11. Run a single warmup inference on a placeholder image
12. Start FastAPI server, listen on configured port (default 8767)

If any step fails, the process exits with a non-zero code; supervisors (systemd, kubernetes) handle restart. Step 11 ensures CUDA kernels are JIT-compiled before serving real traffic; without warmup, the first real request takes 3-5x normal latency.

The startup sequence completes in roughly 8-15 seconds on the RTX 4060 laptop. The `/health` endpoint returns 503 during startup; `/ready` returns 503 until step 12 completes.

### 20.6 GPU lock

GPU compute (model forward passes) is serialized by a single asyncio.Lock. Only one request holds the lock at a time. This prevents:
- VRAM exhaustion from concurrent forward passes (each forward needs 2-3 GB of VRAM out of 8 GB)
- CUDA stream contention degrading per-request latency

Requests waiting for the lock queue with FIFO ordering. The lock has a configurable timeout (`TOMATO_GPU_LOCK_TIMEOUT_S`, default 10 seconds). On timeout, the request returns Section 16.9 `SERVER_OVERLOAD` error with `retry_after_seconds: 5`.

For multi-image requests, all per-image forward passes inside one request happen serially under the same lock acquisition (i.e., one lock for all N images). This is safer than acquiring N times because:
- No risk of partial completion if other requests interleave
- Predictable latency budget per request (no fairness gaming)

**Tradeoff: head-of-line blocking.** A 5-image request blocks the GPU for the duration of all 5 passes (roughly 1.4 seconds of GPU compute per Section 18.8). Other requests, including single-image ones, queue behind it. Under heavy multi-image load, single-image latency can degrade from 500 ms to multiple seconds. The frontend surfaces a "taking longer than expected" message after 5 seconds (Section 19.7) but does not show queue-position details in v1. v2 may introduce request prioritization, a separate single-image lock, or queue-position telemetry to mitigate this.

### 20.7 Configuration sources

Configuration values come from these sources, in order of precedence (highest first):

1. Env vars at process startup
2. `tomato_sandbox/config/local.yaml` (gitignored, local overrides)
3. `tomato_sandbox/config/default.yaml` (committed defaults)
4. Hardcoded fallbacks in `tomato_sandbox/config.py`

Calibration values (conformal tau, IQA thresholds, severity thresholds) live in separate JSON files written by Phase F.0 (Section 29). These are loaded at startup and not in the config hierarchy above.

Sensitive values (Slack webhooks for alerts, database URIs) are env-var only; they never appear in config files.

## Section 21. Pipeline orchestrator

### 21.1 Purpose

The orchestrator (`tomato_sandbox/orchestrator/pipeline.py`) is the function that drives a single prediction request through all pipeline stages. It is the integration point between Section 6 (IQA), Sections 7-9 (signals A, B, C), Section 11 (TTA), Section 12 (classifier), Section 13 (conformal), Section 14 (tier rules), Section 16 (response builder), and Section 17 (severity).

The orchestrator is a pure function in the sense that it has no global state; it reads model objects from a context object and returns a response object. It does have side effects: writing to the structured log, incrementing metrics, persisting Phase E logs.

### 21.2 Function signature

```python
def predict_single(
    image_bytes: bytes,
    request_id: str,
    context: PipelineContext,
) -> ResponseDict:
    """Run the full pipeline for a single image and return the API response."""
```

`PipelineContext` holds:
- `v3_model`, `lora_model`, `psv_module`, `classifier`, `iqa_module`
- `conformal_calibration`, `iqa_thresholds`, `severity_thresholds`
- `gpu_lock` (asyncio.Lock)
- `cache` (request cache, Section 22.5)
- `metrics` (Prometheus counters)
- `phase_e_logger` (Section 24)

For multi-image:

```python
def predict_multi(
    images: list[ImageInput],
    request_id: str,
    context: PipelineContext,
) -> ResponseDict:
    """Run pipeline for each image, then aggregate (Section 18)."""
```

### 21.3 Single-image pipeline steps

The orchestrator executes these steps for `predict_single`:

```
1.  Decode image bytes → numpy array (RGB)
    - On failure: return error response (Section 16.9 IMAGE_DECODE_FAILED)
2.  Compute image_hash (sha256) for caching and Phase E logging
3.  Check sandbox-internal request cache (Section 21.10); if hit, return cached response
4.  Acquire GPU lock (timeout per Section 20.6)
5.  IQA gate (Section 6.4)
    - If REJECT: release lock, return error response (IQA_REJECTED, Section 16.9)
    - Else: continue with IQA decision in {ACCEPTABLE, HIGH, DEGRADED}
6.  Run Signal A (v3) — try/except per Section 8
7.  Run Signal B (LoRA) — try/except per Section 9
8.  Run Signal C (PSV) — try/except per Section 7
8b. Apply degraded-mode handling (Section 21.5) — zero outputs of any failed signals
9.  Build feature vector for classifier (Section 12.2)
10. Run classifier forward pass (Section 12)
11. Apply calibration (temperature scaling per Section 12.10)
11b. Apply NaN guard (Section 21.4) — if classifier output contains NaN, mark all signals as failed and force tier outcome to 4B via Rule 1
12. Check TTA trigger (Section 11.2) using post-calibration `combined_max_prob`
    - If triggers: run TTA, aggregate, re-run classifier with TTA-aggregated features (re-applying calibration and NaN guard)
13. Compute conformal prediction set (Section 13)
14. Assign tier (Section 14.8 — assign_tier function)
15. Compute severity (Section 17 — only for diseased argmax classes at Tier 1/2/3)
16. Generate GradCAM++ overlay (Section 16.5) and save to disk
17. Release GPU lock
18. Build response (Section 16 — build_response function)
19. Write Phase E log entry (Section 24)
20. Increment metrics (Section 25)
21. Cache response (sandbox-internal cache, Section 21.10)
22. Return response
```

Each step has explicit error handling. Failures at any step that produces a tier outcome are routed to Tier 4B via Rule 1; failures before tier assignment (image decode, IQA reject) return error responses without a tier.

### 21.4 NaN guard implementation

Per Section 11.2 (NaN handling contract), the orchestrator detects NaN at the boundary between classifier output and tier assignment:

```python
def apply_nan_guard(classifier_result, signal_a, signal_b, signal_c):
    """Illustrative pseudocode. The actual implementation may use
    immutable dataclasses with replace() rather than in-place mutation.

    If any classifier output is NaN, mark all signals as failed.
    This forces the tier rule chain to fire Rule 1 (pipeline failure → 4B)
    instead of falling through to Rule 9 (catch-all → 4A) on NaN inputs.
    """
    has_nan = (
        np.isnan(classifier_result.combined_max_prob)
        or np.isnan(classifier_result.combined_margin)
        or np.any(np.isnan(classifier_result.p_final_calibrated))
    )
    if has_nan:
        signal_a.forward_succeeded = False
        signal_b.forward_succeeded = False
        signal_c.forward_succeeded = False
        # Set safe default values for tier rules
        classifier_result.combined_max_prob = 0.0
        classifier_result.combined_margin = 0.0
        # Log a structured warning
        logger.warning("nan_in_classifier_output",
            extra={"request_id": classifier_result.request_id})
    return classifier_result, signal_a, signal_b, signal_c
```

This contract is referenced by Section 15 scenario SB.9; the orchestrator implements it.

### 21.5 Degraded mode handling

Per Section 12.7, when one or more signals fail (CUDA OOM, NaN, exception), the classifier still runs with the failed signals zeroed out. The classifier was trained with 20% of training images having one signal randomly zeroed, so it has learned to read the remaining signals.

The orchestrator implements this in `tomato_sandbox/orchestrator/degraded_mode.py`:

```python
def zero_failed_signals(signal_a, signal_b, signal_c):
    """For each failed signal, set its outputs to zero vectors.

    The classifier will then read zeros at those positions and rely on
    the remaining signals via degraded-mode training.
    """
    if not signal_a.forward_succeeded:
        signal_a.probs_canonical = np.zeros(6)
        signal_a.chilli_leakage = 0.0
    if not signal_b.forward_succeeded:
        signal_b.probs = np.zeros(6)
    if not signal_c.forward_succeeded:
        signal_c.argmax = 0
        signal_c.max_prob = 1/6
        signal_c.margin = 0.0
        # Defensive default: 0.05 ensures Rule 3 (psv_reliability < 0.40) fires
        # if Rule 1 (signal failure) is somehow bypassed. Rule 1 should fire
        # first because forward_succeeded=False, but defensive defaults guard
        # against bugs in rule-chain ordering.
        signal_c.reliability = 0.05
        signal_c.features = np.zeros(26)
    return signal_a, signal_b, signal_c
```

The orchestrator calls this between steps 8 and 9 (above): after each signal has been run, before building the feature vector.

**Special case: all three signals failed.** If `signal_a`, `signal_b`, AND `signal_c` are all marked `forward_succeeded == False`, the classifier would receive all-zero inputs. This is out-of-distribution for the classifier, which was trained with at most one signal zeroed at a time (Section 12.7 specifies 20% per-image with one signal zeroed; the all-three-zeroed case was not covered by training data).

Rather than running the classifier on all-zero input (which produces undefined behavior, often near-uniform probabilities), the orchestrator short-circuits in this case: skip the classifier forward pass entirely, set the classifier output to a sentinel "all signals failed" marker, and route to Tier 4B via Rule 1. The response builder reports this case with a specific structured reason `all_signals_failed`.

```python
if not (signal_a.forward_succeeded or signal_b.forward_succeeded or signal_c.forward_succeeded):
    classifier_result = make_sentinel_classifier_result(reason="all_signals_failed")
    return classifier_result
```

This keeps the all-three-failed case predictable: Tier 4B with a clear structured reason, rather than relying on the classifier to do something sensible on out-of-distribution input.

### 21.6 Multi-image orchestration

For `predict_multi`, the orchestrator runs `predict_single`-equivalent logic for each image, then calls the aggregator (`tomato_sandbox/multi_image/aggregator.py`) to produce the final response.

Per-image processing happens within a single GPU lock acquisition (Section 20.6); the aggregator runs on CPU after the lock is released.

The orchestrator gathers per-image results into a list and passes them to the aggregator:

```python
async def predict_multi(images, request_id, context):
    async with context.gpu_lock:
        per_image_results = []
        for img in images:
            try:
                result = await _run_single_image_within_lock(img, context)
                per_image_results.append(result)
            except IQARejectError:
                per_image_results.append(IQARejectedResult(image_id=img.id))
            except TimeoutError:
                per_image_results.append(TimeoutResult(image_id=img.id))
    aggregated = aggregate_results(per_image_results)
    response = build_multi_image_response(aggregated, per_image_results, request_id)
    return response
```

The helper `_run_single_image_within_lock` runs steps 5-16 from Section 21.3 (IQA through GradCAM++), assuming the GPU lock is already held. Steps 1-3 (decode, hash, cache) and step 4 (acquire lock) are skipped per-image because they happen at the request level. Steps 17-22 (release lock, build response, log, cache) happen after the aggregator runs at the request level.

If all per-image results are IQA REJECTed or all are timeouts, the aggregator returns the corresponding error per Section 18.4 special cases.

### 21.7 Latency budget per step

The orchestrator's latency target (P95) for a single image, assuming idle conditions (no queue waiting):

| Step | Target | Notes |
|---|---|---|
| 1-3 (decode, hash, cache check) | 30 ms | CPU-bound; cache check is in-memory dict |
| 4 (lock acquire) | 0-50 ms | Idle: 0; under load: queues |
| 5 (IQA) | 60 ms | CPU + GPU mixed |
| 6 (Signal A) | 80 ms | GPU |
| 7 (Signal B) | 60 ms | GPU |
| 8 (Signal C) | 100 ms | CPU (PSV) |
| 9-12 (feature, classifier, calibration, NaN guard, TTA check) | 30 ms | GPU + CPU |
| 13-15 (conformal, tier, severity) | 10 ms | CPU |
| 16 (GradCAM++) | 80 ms | GPU + disk write |
| 17-22 (release, build response, log, cache) | 50 ms | CPU + disk |
| **Total (idle, no TTA)** | **550 ms** | (within Section 4.6 single-image P95 target of 800 ms total request time) |

When TTA fires, steps 9-12 add 200-400 ms (2-view) or 600-1200 ms (5-view).

For multi-image latency, see Section 18.8. Total multi-image latency at N=5 is approximately 2.0 seconds (parallel-ish execution within a single GPU lock acquisition).

**Queuing delays under load.** The 0-50 ms range for step 4 (lock acquire) assumes light load. Under heavy load with many concurrent multi-image requests, queue waits can extend to several seconds. Section 25 (monitoring) tracks P95 and P99 lock-wait time; if P99 lock-wait exceeds 2 seconds, an alert is emitted (operations team should investigate whether to add capacity or reduce the multi-image limit).

### 21.8 Error categories

The orchestrator catches and categorizes errors:

| Error | Category | Tier outcome | Response code |
|---|---|---|---|
| Image decode failure | Pre-tier | None | 400 IMAGE_DECODE_FAILED |
| Image too large | Pre-tier | None | 413 IMAGE_TOO_LARGE |
| Unsupported format | Pre-tier | None | 415 IMAGE_UNSUPPORTED_FORMAT |
| IQA REJECT | Pre-tier | None | 422 IQA_REJECTED |
| GPU lock timeout | Pre-tier | None | 503 SERVER_OVERLOAD |
| Signal A failure | In-pipeline | 4B (via Rule 1) | 200 (with tier 4B) |
| Signal B failure | In-pipeline | 4B | 200 (with tier 4B) |
| Signal C failure | In-pipeline | 4B | 200 (with tier 4B) |
| Classifier NaN | In-pipeline | 4B (via NaN guard) | 200 (with tier 4B) |
| Unexpected exception | Catch-all | None | 500 INTERNAL_ERROR |

In-pipeline failures still produce a valid response with a tier label; the user sees Tier 4B with explanatory text. Pre-tier failures produce error responses without a tier (Section 16.9).

### 21.9 Logging at each step

The orchestrator emits a structured log line at each step with:

```json
{
  "request_id": "...",
  "step": "signal_a",
  "duration_ms": 78,
  "succeeded": true,
  "details": { "max_prob": 0.91 }
}
```

The log lines feed Section 25 monitoring and Section 24 Phase E logging. The full per-request log is also persisted in Phase E SQLite for post-hoc analysis.

### 21.10 Sandbox-internal request cache

The sandbox has a per-process cache that maps `image_hash` to recent responses. This is independent of the unified server's cache (Section 22.5):

- Key: `image_hash` (SHA256 hex string)
- Value: full sandbox response per Section 16.2 (without the unified envelope)
- TTL: configurable via `TOMATO_REQUEST_CACHE_TTL_S` (default 3600 = 1 hour, per Section 4.5)
- Size: configurable via `TOMATO_REQUEST_CACHE_SIZE` (default 1000 entries, LRU eviction)

**Why image_hash only as key (not the triple `crop, image_hash, model_version`)?** The sandbox process is restarted on model version change (Section 20.5 startup loads weights from disk; new weights require a restart to take effect). All cache entries from the prior model version are invalidated by definition when the process restarts. The sandbox is single-tenant on the tomato crop, so `crop` is implicitly fixed. The triple key is only needed at the unified server level where multiple model versions could coexist briefly during deployment.

**Cache miss path:** runs steps 4-22 from Section 21.3 normally and stores the response under `image_hash` before returning.

**Cache hit path:** returns the cached response immediately without acquiring the GPU lock or running any signals. The response's `processing_time_ms` is set to the cache lookup time (single-digit ms), and a `from_cache: true` flag is added to the response's `warnings` field.

This in-process cache reduces latency for repeated requests (e.g., the same user retrying after a network glitch) and reduces GPU load. It does NOT replace the unified server cache, which serves a different purpose (cross-instance dedup if v2 ever scales horizontally).

## Section 22. Unified server routing

### 22.1 Purpose

The unified server (port 8005) is the single entry point for all crops. It routes incoming requests to the appropriate downstream service:

- Tomato → tomato sandbox (port 8767, this spec)
- Okra / brassica → APIN (port 8766, existing)
- Chilli → not yet implemented in v1; returns error

The unified server is NOT new code in this spec. It exists as part of the broader project; Section 22 documents how the tomato sandbox integrates with it and what changes are needed to add tomato routing.

### 22.2 Routing logic

The unified server inspects the request to determine the crop:

1. If the request body has an explicit `crop` field, use it
2. Otherwise, run the crop router (Section 1) to predict crop from image
3. Map crop to downstream service:
   - `tomato` → forward to `http://localhost:8767/predict` or `/predict_multi`
   - `okra`, `brassica` → forward to `http://localhost:8766/predict` (APIN existing endpoint)
   - `chilli` → return 501 NOT_IMPLEMENTED with explanation

After receiving the downstream response, wrap in the envelope per Section 16.10:

```json
{
  "crop": "tomato",
  "service_used": "tomato_sandbox",
  "service_version": "tomato-sandbox-v1.0.0",
  "response": { /* unwrapped sandbox response */ }
}
```

Both the explicit-crop path and the predicted-crop path use the same routing; the predicted-crop path additionally includes the crop router's confidence in the response under `crop_router_confidence`.

### 22.3 Adding tomato routing

To add tomato routing to the unified server, the following changes are needed in the unified server's code (NOT in `tomato_sandbox/`):

1. Add `tomato` to the supported crops list
2. Add a downstream service entry for tomato pointing at port 8767
3. Add the tomato sandbox response unwrapping logic (it returns the unwrapped response per Section 16.2; the unified server adds the envelope)
4. Add a feature flag `UNIFIED_TOMATO_ROUTE_ENABLED` (default false in v1; flip to true after Phase F.0 validation)

These changes are made in the unified server repo (not the sandbox repo). The sandbox is unaware of the unified server's existence; it serves whoever calls its endpoints.

### 22.4 Backward compatibility

The unified server's existing behavior is unchanged for okra/brassica/chilli. Only the tomato path is new.

If the `UNIFIED_TOMATO_ROUTE_ENABLED` flag is false, tomato requests fall through to the legacy crop-router-only path (which would return an "unsupported" error since the legacy server has no tomato handler).

When the flag is true and the sandbox is unreachable (e.g., port 8767 not responding), the unified server returns 503 SANDBOX_UNAVAILABLE; it does NOT fall back to a different model. This is intentional: returning a guess from a different model would be worse than failing.

### 22.5 Caching layer

The unified server has a request-level cache that maps `image_hash` to recent responses. Cache scope:

- Key: `(crop, image_hash, model_version)` triple
- Value: full envelope response
- TTL: configurable via `UNIFIED_CACHE_TTL_S`, default 900 (15 minutes)
- Size: 1000 entries (LRU eviction)

The cache lives in the unified server's memory (process-local). It is NOT a distributed cache; cross-instance cache sharing is a v2 concern.

Cache hits skip the downstream service entirely. Cache misses forward to the downstream service and store the response.

The sandbox itself has a separate per-process cache (Section 21.10) that uses `image_hash` only as the key (because the sandbox process is restarted on model version change, making the triple key redundant). The two caches are independent and serve different purposes; double-caching is not a problem because both are bounded.

### 22.6 Health and readiness

The unified server's `/health` and `/ready` endpoints aggregate downstream service status:

- Unified `/health`: returns 200 if the unified server itself is up
- Unified `/ready`: returns 200 if all downstream services (sandbox, APIN) report ready; returns 503 otherwise

This is conservative: a degraded downstream marks the unified server as not ready, even if the other downstream is fine. The alternative (per-crop readiness flags) is a v2 enhancement.

The unified server discovers downstream services via env vars:
- `UNIFIED_SANDBOX_URL` (default `http://localhost:8767`)
- `UNIFIED_APIN_URL` (default `http://localhost:8766`)

Service discovery is not used in v1; downstream URLs are hardcoded via env vars. v2 may add Consul or similar.

### 22.7 Timeout and retry behavior

The unified server has a timeout per downstream call:

- Sandbox single image: 5 seconds
- Sandbox multi-image: 15 seconds
- APIN: 5 seconds (legacy default)

On timeout, the unified server returns 504 GATEWAY_TIMEOUT to the client. There is no automatic retry from the unified server; the client (frontend) handles retry per Section 19.7.

For pre-tier failures from the sandbox (e.g., IQA REJECT, IMAGE_DECODE_FAILED), the unified server passes through the error response without modification (still wrapped in the envelope). The frontend (Section 19.7) decides whether to surface the error directly or offer a retry button.

**Optimization opportunity (deferred to v2):** the sandbox currently runs IQA inside the GPU lock (Section 21.3 step 5). IQA is mostly CPU-bound; running it before lock acquisition would free the GPU for other requests during IQA computation. This is a v2 optimization.

Retries are intentionally client-side because:
- The user can decide whether to retry (e.g., they may want to retake the photo)
- Server-side retries would multiply downstream load during outages
- Idempotency would require request deduplication that v1 doesn't have

### 22.8 Logging and tracing

The unified server emits a structured log line per request:

```json
{
  "unified_request_id": "...",
  "crop": "tomato",
  "service_used": "tomato_sandbox",
  "downstream_request_id": "...",
  "duration_ms": 524,
  "status_code": 200,
  "cache_hit": false
}
```

The `unified_request_id` and `downstream_request_id` are correlated via OpenTelemetry trace context (W3C trace headers). The sandbox's structured logs (Section 21.9) include the unified request id when received.

Tracing is implemented via `opentelemetry-instrumentation-fastapi`; spans are exported to the configured OTLP endpoint. In v1, the OTLP endpoint is optional (set via env var); if unset, tracing is disabled.

### 22.9 Limitations

1. **Single instance.** The unified server runs as a single process; it is the bottleneck for total system QPS. Horizontal scaling is a v2 concern.
2. **No request prioritization.** All requests are FIFO; high-priority cases (Tier 5 alerts) are not preempted at the routing layer. Section 23 (agronomist queue) handles priority post-tier.
3. **No circuit breakers.** If the sandbox is repeatedly slow or failing, the unified server keeps forwarding traffic with timeouts. This risks exhausting the unified server's connection pool during a sustained downstream outage. A v2 enhancement adds circuit breakers that fail fast when downstream is unhealthy, freeing the connection pool for healthy traffic.
4. **Cache invalidation is manual.** Model version changes require a unified server restart to clear cached entries. v2 may add cache invalidation hooks.
5. **No horizontal scaling of caches.** Each unified server instance has its own cache; in a multi-instance deployment, cache hits would be inconsistent across instances.

---
## Section 23. Agronomist queue

### 23.1 Purpose

The agronomist queue is the human-review pipeline for cases that need expert judgment beyond the automated tier outcome. It receives routed cases from the sandbox (per Section 16.8 routing rules) and presents them to NanoFarm's agronomic team via a dedicated UI. Agronomists review, optionally annotate, and resolve cases; their dispositions feed back into Phase F.0 retraining (Section 24.8).

The queue is critical for the v1 system because:

- Tier 5 alerts (dangerous diseases) require expert verification before action. The system flags but does not autonomously prescribe high-stakes treatments.
- Tier 3 ambiguous cases benefit from agronomist judgment; the system proposes possibilities and the agronomist confirms.
- Tier 4 low-confidence cases are by definition cases the system cannot handle; an agronomist must decide.
- Without a human-review backstop, the system would be either too conservative (refusing many cases) or too aggressive (acting on uncertain predictions).

### 23.2 Routing rules

Section 16.8 defines which tiers route to the queue:

| Base tier | Routes to queue? | Priority |
|---|---|---|
| 1 (definitive) | Only if T5 alert fires | T5 hint per Section 16.7 |
| 2 (confident) | Only if T5 alert fires | T5 hint per Section 16.7 |
| 3A (two-class) | Only if `route_ambiguous_to_queue` flag enabled | medium |
| 3B (multi-class) | Only if `route_ambiguous_to_queue` flag enabled | medium |
| 3C (PSV / chilli) | Only if `route_ambiguous_to_queue` flag enabled | low |
| 3D (DEGRADED IQA) | Only if `route_ambiguous_to_queue` flag enabled | low |
| 4A (low confidence) | If T5 fires, OR user opt-in | T5 hint or user-set |
| 4B (pipeline failure) | Never | not routed |

The Tier 5 alert is not a base tier; it fires alongside any base tier (Section 14.3). When T5 fires on Tier 1 or Tier 2, the case is routed to the queue per the rows above (these tiers route only when T5 fires). When T5 fires on Tier 3 or 4, the case is routed regardless of the `route_ambiguous_to_queue` flag because T5-firing overrides the flag. The "T5 fires → always routed" rule is implicit in the per-tier rows.

The `route_ambiguous_to_queue` flag is a deployment-time config controlled by the operations team. v1 deployment defaults the flag to `false` because the agronomic team capacity is limited; the team manages the flag based on real load.

### 23.3 Queue data model

Each queue entry is a `QueueCase` record with these fields:

```python
@dataclass
class QueueCase:
    case_id: str                       # uuid
    request_id: str                    # links to Phase E log entry
    image_hash: str                    # for dedup if same image is re-submitted
    image_path: str                    # path to stored image bytes
    gradcam_path: str | None           # GradCAM++ overlay if available
    crop: str                          # always "tomato" in v1; included for v2 compatibility
                                       # when okra/brassica/chilli may share this queue
    submitted_at: datetime
    status: str                        # in {"pending", "in_review", "resolved", "dismissed", "escalated"}
    priority: str                      # in {"high", "medium", "low"}
    tier_label: str                    # snapshot at routing time
    tier5_fired: bool
    tier5_reason: str | None
    primary_class: str
    primary_confidence: float
    prediction_set: list[str]
    severity_grade: str | None
    severity_details: dict | None
    user_metadata: dict                # location, plant info, user note
    assigned_to: str | None            # agronomist user_id once picked up
    assigned_at: datetime | None
    resolved_at: datetime | None
    resolution: dict | None            # see 23.6
```

Records persist in SQLite (Section 24) with the schema in 24.2.

### 23.4 Priority assignment

When a case is routed to the queue, its priority is set as follows:

1. If `tier5_alert.agronomist_priority_hint` is set (per Section 16.7), use that value as the starting priority.
2. Otherwise, use the per-tier default from the table in 23.2.
3. The agronomist queue manager (a stateful service inside the sandbox or separate microservice in v2) may demote priority if:
   - The same image has been seen before with a confident agronomist disposition (re-submit suppression)
   - The user is over a daily submission rate limit (low priority for high-volume submitters)
4. The priority does not promote automatically; only an agronomist can mark a case as higher priority during review.

The asymmetry (auto-demote, manual-only-promote) is intentional. Auto-demote is conservative: a case demoted to "low" still gets reviewed eventually, and demotion based on signals like "duplicate image" or "high-volume submitter" reduces queue clutter without risking missed urgent cases. Auto-promote would be exploitable: a malicious or careless user could attach urgency keywords to game priority, swamping the queue with fake high-priority cases. Manual-promote requires a human agronomist's judgment, which is harder to game and matches the role of priority promotion as an expert correction.

The 0.50 confidence rule from Section 16.7 (high vs medium for late_blight argmax) is applied at routing time, not re-evaluated. If the agronomist downgrades the case (e.g., believes it is actually foliar not late_blight), the priority does not retroactively change; the disposition record carries the agronomist's correction.

### 23.5 Agronomist UI requirements

The agronomist UI consumes the queue and displays cases for review. Section 23 specifies the API contract; the UI implementation is a separate frontend project.

**API endpoints exposed by the queue service:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/queue/cases` | GET | List pending cases, sortable by priority/age |
| `/queue/cases/{case_id}` | GET | Fetch a single case with full details |
| `/queue/cases/{case_id}/claim` | POST | Agronomist claims a case (sets status to "in_review", assigned_to, assigned_at) |
| `/queue/cases/{case_id}/resolve` | POST | Agronomist resolves a case with disposition |
| `/queue/cases/{case_id}/dismiss` | POST | Agronomist dismisses a case (e.g., not a real disease) |
| `/queue/cases/{case_id}/escalate` | POST | Agronomist escalates to senior agronomist |
| `/queue/stats` | GET | Aggregate stats (pending count, P50/P95 review time, etc.) |

**Required UI views:**

- **Queue list view:** sortable by priority, submission time, tier. Color-coded priorities (red/yellow/gray for high/medium/low).
- **Case detail view:** shows the original image, GradCAM++ overlay (toggleable), system's tier outcome, structured reasons, severity grade, and user metadata.
- **Disposition form:** dropdown for "confirm system prediction / correct to different class / mark as healthy / mark as not-a-tomato / mark as poor-quality-image / escalate". Free-text comment field. Optional severity correction.

The UI must show the system's prediction PROMINENTLY but NOT in a way that biases the agronomist toward agreement. NanoFarm's UX team will calibrate this after pilot use.

### 23.6 Disposition

When an agronomist resolves a case, the disposition record is:

```python
@dataclass
class QueueDisposition:
    resolved_by: str                # agronomist user_id
    resolution_type: str            # "confirm" | "correct" | "mark_healthy" | "mark_not_tomato" | "mark_poor_image" | "escalate"
    corrected_class: str | None     # only set if resolution_type == "correct"
    corrected_severity: str | None  # only set if resolution requires severity correction
    confidence: str                 # "high" | "medium" | "low" - agronomist's own confidence
    comment: str | None             # free-text
    treatment_recommendation: str | None
    follow_up_required: bool
    follow_up_due_date: datetime | None
```

The disposition is persisted with the case (Section 24) and feeds into:
- The user-facing app: the user is notified that an expert reviewed their case, with the agronomist's recommendation
- Phase F.0 retraining (Section 24.8): cases with `confidence == "high"` become labeled training examples
- Monitoring (Section 25): disposition rates per tier inform model quality metrics

### 23.7 SLA and capacity

The v1 system targets these SLAs:

| Priority | Target P50 review time | Target P95 review time |
|---|---|---|
| High (T5 alerts) | 2 hours | 6 hours |
| Medium (Tier 3 routed) | 24 hours | 72 hours |
| Low (Tier 4A routed) | 72 hours | 7 days |

These are aspirational, not contractual. v1 deployment will measure actual review times during pilot use; the agronomic team's capacity is operations-managed and may change as NanoFarm hires or reassigns staff. At spec writing, the capacity is approximately 1 senior + 2 junior agronomists; pilot SLAs are calibrated to that headcount. If pilot reveals capacity shortfall, the `route_ambiguous_to_queue` flag is disabled (Tier 3 cases stop routing), keeping the queue focused on T5 alerts only.

Capacity thresholds:
- If pending queue size exceeds 200 cases for > 4 hours: alert operations team (MEDIUM severity per Section 25.7)
- If pending queue size exceeds 500 cases: alert operations team (HIGH severity); operations may then disable `route_ambiguous_to_queue` flag manually to focus on T5 alerts only

The thresholds emit alerts; the system does not automatically flip the config flag. Manual control of the flag preserves operations' authority over queue routing decisions.

### 23.8 Feedback loop to F.0 retraining

Resolved cases with `confidence == "high"` are eligible for inclusion in Phase F.0 retraining datasets. The pipeline:

1. Resolved case in queue: agronomist's `corrected_class` and `confidence == "high"` recorded
2. Daily batch job exports eligible cases to a labeled training set
3. F.0 retraining job (Section 29, future) consumes the labeled set
4. Retrained model goes through validation; if metrics improve, the new model is deployed

The feedback loop is asynchronous (not real-time). v1 batch frequency is daily; v2 may add streaming updates.

Eligibility filters:
- Only `confidence == "high"` dispositions
- Only cases where the system's prediction was different from the agronomist correction (correct cases reinforce existing patterns; incorrect cases teach new patterns; both are useful but corrections add the most signal)
- Cases marked as "poor image" are excluded from CLASS retraining (avoid teaching the disease classifier to focus on noise) but ARE included in IQA retraining (poor-image dispositions are valuable supervision for IQA quality calibration)
- PII is stripped before export (Section 24.6)

Once a case is exported to the training dataset filesystem, the export is materialized: it survives even if the source SQLite row is later deleted by retention cleanup (Section 24.5). The training dataset is managed separately from the operational SQLite database.

### 23.9 Limitations

1. **Manual queue management.** Priority is set at routing time; agronomists cannot bulk-prioritize. v2 may add dynamic priority based on pending queue size.
2. **No agronomist load balancing.** The queue does not assign cases to specific agronomists; agronomists self-select via "claim" endpoint. v2 may add automatic assignment based on agronomist specialty (e.g., late_blight expert gets late_blight cases first).
3. **No multi-agronomist consensus for critical cases.** A single agronomist's disposition is final. v2 may add "second opinion" workflow for high-priority cases.
4. **No real-time agronomist availability check.** If all agronomists are offline, cases pile up. v1 has no fallback (the user just waits longer); v2 may surface "agronomists are currently offline, expected response in X hours."
5. **No structured taxonomy for "correct to different class" resolutions.** Agronomists pick from the 7 classes the system knows. If the actual disease is outside the trained set (e.g., bacterial wilt), the agronomist must use free-text comment, which doesn't feed back into structured retraining.
6. **Stale claims.** If an agronomist claims a case but never resolves it (got distracted, browser closed), the case stays in `in_review` forever. v1 mitigation: a daily cleanup job resets cases in `in_review` for > 24 hours back to `pending` and clears `assigned_to` / `assigned_at`. The reset emits a structured log entry so the operations team can investigate stuck-agronomist patterns. v2 may add WebSocket-based heartbeat for live presence detection.
7. **Queue service unavailable.** If the queue service is down when the sandbox tries to route a case, the sandbox cannot block the response (the user is waiting). The mitigation: the sandbox sets `pending_queue_route = 1` on the `predictions` row (Section 24.2 schema) and returns the user response normally. A background sweeper job periodically scans for `pending_queue_route = 1` rows and attempts routing again. If the queue service stays down for hours, predictions accumulate but user-facing responses are not blocked.

These limitations are tracked in Section 30 and will be addressed in v2 based on pilot feedback.

## Section 24. Storage and persistence (Phase E SQLite logging)

### 24.1 Purpose

Section 24 specifies persistent storage for the sandbox: per-request structured logs (Phase E logging), queued agronomist cases (Section 23), and visualization images (GradCAM++ overlays).

The v1 storage backend is SQLite, single-file at `/var/lib/tomato_sandbox/sandbox.db`. SQLite suits v1 because:

- Single-process sandbox (no need for multi-writer concurrency)
- Modest write rate (P95 < 5 writes per second under expected pilot load)
- Simple operations (no separate database server to deploy)
- Easy backup (single file copy)

For v2 horizontal scaling, the spec migrates to Postgres (Section 24.8). The schema is designed so the migration is mechanical.

### 24.2 Schema

The SQLite database has these tables. **Schema conventions:** SQLite does not have a native boolean type, so booleans are stored as INTEGER with values 0 (false) or 1 (true). JSON-typed application data is stored as TEXT (with the column name suffix `_json`); SQLite's JSON1 extension can query inside these fields if needed. CHECK constraints (e.g., `tier_label IN ('1', '2', '3A', ...)`) are not added in v1 because validation happens at the application layer; v2 may add CHECK constraints for additional safety. The `user_id` column on `predictions` and `queue_cases` is included for direct queryability of right-to-deletion requests (Section 24.6); it is set to a hashed user identifier or `NULL` for anonymous submissions.

**`predictions` - one row per `/predict` or per-image-of-multi request:**

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    user_id TEXT,                       -- hashed user identifier; NULL for anonymous
    image_hash TEXT NOT NULL,
    image_path TEXT NOT NULL,
    submitted_at TIMESTAMP NOT NULL,
    response_built_at TIMESTAMP NOT NULL,
    crop TEXT NOT NULL,
    tier_label TEXT NOT NULL,
    tier5_fired INTEGER NOT NULL,       -- boolean as 0/1
    tier5_reason TEXT,
    primary_class TEXT NOT NULL,
    primary_confidence REAL NOT NULL,
    prediction_set_json TEXT NOT NULL,  -- JSON list of class names
    severity_grade TEXT,
    severity_details_json TEXT,
    rule_id_fired TEXT NOT NULL,
    sub_rule_id_fired TEXT NOT NULL,
    iqa_decision TEXT NOT NULL,
    iqa_aggregate_score REAL NOT NULL,
    psv_reliability REAL NOT NULL,
    chilli_leakage REAL NOT NULL,
    combined_max_prob REAL NOT NULL,
    combined_margin REAL NOT NULL,
    classifier_p_final_calibrated_json TEXT NOT NULL,
    conformal_set_json TEXT NOT NULL,
    conformal_tau REAL NOT NULL,
    signal_a_succeeded INTEGER NOT NULL,
    signal_b_succeeded INTEGER NOT NULL,
    signal_c_succeeded INTEGER NOT NULL,
    tta_fired INTEGER NOT NULL,
    tta_view_count INTEGER NOT NULL,
    processing_time_ms INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    user_metadata_json TEXT,
    multi_image_request_id TEXT,        -- if part of multi-image, the request_id of the multi
    image_id_within_multi TEXT,
    pending_queue_route INTEGER NOT NULL DEFAULT 0  -- 1 if routing was deferred (Section 23 queue down)
);

CREATE INDEX idx_predictions_request_id ON predictions(request_id);
CREATE INDEX idx_predictions_user_id ON predictions(user_id);
CREATE INDEX idx_predictions_image_hash ON predictions(image_hash);
CREATE INDEX idx_predictions_submitted_at ON predictions(submitted_at);
CREATE INDEX idx_predictions_tier_label ON predictions(tier_label);
CREATE INDEX idx_predictions_tier5_fired ON predictions(tier5_fired);
CREATE INDEX idx_predictions_pending_queue_route ON predictions(pending_queue_route)
  WHERE pending_queue_route = 1;
```

The JSON-in-TEXT pattern is acceptable for v1's modest write rate. Queries that need to filter by JSON contents (e.g., "find rows where late_blight is in `prediction_set_json`") use SQLite's JSON1 functions; performance is adequate for v1 volumes but degrades at scale. v2 Postgres migration would use `jsonb` with GIN indexes for queryable JSON.

**`queue_cases` - one row per agronomist queue case (Section 23):**

```sql
CREATE TABLE queue_cases (
    case_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    user_id TEXT,                         -- hashed user identifier; NULL for anonymous
    image_hash TEXT NOT NULL,
    image_path TEXT NOT NULL,
    gradcam_path TEXT,
    crop TEXT NOT NULL,
    submitted_at TIMESTAMP NOT NULL,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    tier_label TEXT NOT NULL,
    tier5_fired INTEGER NOT NULL,
    tier5_reason TEXT,
    primary_class TEXT NOT NULL,
    primary_confidence REAL NOT NULL,
    prediction_set_json TEXT NOT NULL,
    severity_grade TEXT,
    severity_details_json TEXT,
    user_metadata_json TEXT,
    assigned_to TEXT,
    assigned_at TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_json TEXT,
    FOREIGN KEY (request_id) REFERENCES predictions(request_id)
);

CREATE INDEX idx_queue_cases_status ON queue_cases(status);
CREATE INDEX idx_queue_cases_user_id ON queue_cases(user_id);
CREATE INDEX idx_queue_cases_priority ON queue_cases(priority);
CREATE INDEX idx_queue_cases_submitted_at ON queue_cases(submitted_at);
```

**`per_signal_logs` - one row per signal per request, for debugging:**

```sql
CREATE TABLE per_signal_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    signal_name TEXT NOT NULL,  -- "a", "b", "c"
    succeeded INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    output_json TEXT,
    error_message TEXT,
    FOREIGN KEY (request_id) REFERENCES predictions(request_id)
);

CREATE INDEX idx_per_signal_logs_request_id ON per_signal_logs(request_id);
```

**`metrics_snapshots` - periodic snapshots of computed metrics (Section 25):**

```sql
CREATE TABLE metrics_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at TIMESTAMP NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_dimensions_json TEXT
);

CREATE INDEX idx_metrics_snapshots_at ON metrics_snapshots(snapshot_at);
CREATE INDEX idx_metrics_snapshots_name ON metrics_snapshots(metric_name);
```

### 24.3 Write path

Per Section 21.3 step 19, the orchestrator writes a Phase E log entry after building the response. The write path:

1. Construct the `predictions` row from the response and intermediate results
2. Construct `per_signal_logs` rows for signals A, B, C
3. Open a transaction
4. INSERT both
5. Commit
6. If commit fails: log the error and continue (do NOT fail the request because logging failed)

Writes are synchronous from the request's perspective; they happen on the request thread after response is built but before returning. With WAL mode (write-ahead logging) enabled at startup via `PRAGMA journal_mode=WAL`, SQLite supports concurrent reads during writes and provides good single-writer throughput. The expected per-request DB write overhead is < 5 ms for 4-5 row inserts; this is an estimate rather than a measurement, and Phase F.0 will verify it under realistic load. If actual write time exceeds 20 ms regularly, the schema or write path needs revisiting.

For multi-image requests, each per-image result writes its own `predictions` row with `multi_image_request_id` set to the wrapping request id and `image_id_within_multi` set to the per-image id. The wrapping multi-image request itself does NOT have a separate `predictions` row; the per-image rows are joinable via the multi_image_request_id.

The `image_path` field stores a relative path under `/var/lib/tomato_sandbox/images/`. Image bytes themselves are written to disk before the database row is committed. If the disk write fails, the DB row is not written.

### 24.4 Read path / queries

Common queries the system runs against this data:

**Daily volume by tier:**

```sql
SELECT
    DATE(submitted_at) as day,
    tier_label,
    COUNT(*) as count
FROM predictions
WHERE submitted_at >= datetime('now', '-30 days')
GROUP BY day, tier_label
ORDER BY day DESC, tier_label;
```

**Tier 5 alert distribution:**

```sql
SELECT
    tier5_reason,
    primary_class,
    COUNT(*) as count
FROM predictions
WHERE tier5_fired = 1 AND submitted_at >= datetime('now', '-7 days')
GROUP BY tier5_reason, primary_class
ORDER BY count DESC;
```

**Pending queue cases by priority:**

```sql
SELECT
    priority,
    COUNT(*) as pending_count,
    MIN(submitted_at) as oldest_submission
FROM queue_cases
WHERE status = 'pending'
GROUP BY priority;
```

**Agronomist disposition rate:**

```sql
SELECT
    status,
    COUNT(*) as count
FROM queue_cases
WHERE submitted_at >= datetime('now', '-7 days')
GROUP BY status;
```

These queries power Section 25 metrics and the operations dashboard. They are not real-time (the dashboard polls SQLite every 60 seconds in v1) but are sufficient for pilot operations.

### 24.5 Retention policy

To keep the SQLite file from growing unbounded:

| Data | Retention |
|---|---|
| `predictions` table rows | 90 days |
| `per_signal_logs` table rows | 30 days |
| `queue_cases` rows (resolved) | 1 year |
| `queue_cases` rows (pending or in_review) | indefinite |
| `metrics_snapshots` table rows | 1 year |
| Image files (under `/images/`) | 30 days, OR until associated queue_case is older than 1 year if the case is pending or resolved |
| GradCAM++ overlays (under `/visualizations/`) | 30 days |

A daily cleanup job runs at 03:00 local time, deletes records older than the retention window, and VACUUMs the SQLite file weekly. Records flagged `preserve = 1` (e.g., a case selected for inclusion in the F.0 retraining dataset via Section 23.8) are excluded from deletion regardless of age. The `preserve` flag is an additional column on `predictions` and `queue_cases` (default 0); the daily export job in Section 23.8 sets the flag for selected cases.

**Image retention coupled to queue_cases:** if a queue_case is resolved or pending and references an image, the image is retained as long as the queue_case row exists (overrides the 30-day default). This avoids the case where an old resolved-case row points at a missing image file. Once both the queue_case row and the predictions row are eligible for deletion, the image file is also deleted.

The retention values are env-var configurable:
- `TOMATO_RETENTION_PREDICTIONS_DAYS` (default 90)
- `TOMATO_RETENTION_IMAGES_DAYS` (default 30)
- `TOMATO_RETENTION_VISUALIZATIONS_DAYS` (default 30)

Pilot operations may extend retention if disk space allows; the agronomic team is interested in long-tail data for retraining.

**Storage scale.** The retention math depends heavily on request volume. At pilot volume of ~50-200 requests per day (NanoFarm's expected first-month load), 90-day predictions retention amounts to under 50 MB of database rows and ~6 GB of image files at 1 MB average per image. This fits easily on the laptop SSD. The 5-requests-per-second figure cited elsewhere in this section is a v2 design target, not a pilot expectation; full v2-scale storage planning will be revisited when production load approaches it.

### 24.6 PII handling

The `user_metadata_json` field may contain location coordinates, plant variety, plant age, and free-text user notes. These are PII-adjacent: a user's farm location combined with submission patterns could identify the user.

PII handling rules:

- Storage encryption: SQLite file is on an encrypted-at-rest disk volume (operations responsibility, not sandbox code)
- Access control: only the sandbox process and operations team have file system access
- Logging: structured logs (Section 21.9) emit user_metadata only to authorized log sinks (sandbox local file, NOT to external monitoring systems unless those are also access-controlled)
- Export: when cases are exported for retraining (Section 23.8), user_metadata is stripped EXCEPT for the plant variety field (which is non-identifying)
- Right to deletion: if a user requests data deletion (Section 30 limitations), all rows with that user's identifier are hard-deleted from `predictions` and `queue_cases` tables; image files are also deleted

The privacy policy presented to the user at app onboarding describes these handling rules in plain language. NanoFarm's legal team reviews the v1 privacy policy before pilot deployment.

### 24.7 Backup

Backup strategy:

- Daily: SQLite file copied to a backup directory `/var/lib/tomato_sandbox_backups/sandbox-YYYY-MM-DD.db`
- Weekly: backup uploaded to off-site object storage (operations chooses vendor; S3 or equivalent)
- Monthly: long-term archive in cold storage (S3 Glacier or equivalent)
- Retention: 30 daily, 12 weekly, 24 monthly backups

Backups include the SQLite file but NOT the image files (images are stored separately under `/images/` and have their own backup policy via filesystem snapshots).

Recovery: in case of data loss, restore the most recent backup; the sandbox accepts restored data without reconciliation (any in-flight requests during the failure are lost; users would re-submit).

### 24.8 Migration to Postgres for v2

The schema in 24.2 is SQL-92 compatible. Migration to Postgres requires:

1. Create Postgres instance with same schema (data types map directly: TEXT → TEXT, INTEGER → INTEGER, REAL → DOUBLE PRECISION, TIMESTAMP → TIMESTAMP)
2. Update the sandbox's `tomato_sandbox/storage/sqlite_logger.py` to use Postgres connection string
3. Run a one-time data migration script: read SQLite rows, write to Postgres
4. Test migration in staging
5. Cut over with brief downtime (~5 minutes)

Postgres unlocks:
- Multi-process sandbox (with shared DB)
- Concurrent writes from horizontal-scaled instances
- Better query performance for analytics
- Replication and high availability

v1 commits to SQLite; v2 migration is planned but out of scope.

### 24.9 Limitations

1. **Single-file SQLite is a SPOF.** If the file is corrupted (rare but possible), data is lost up to the last backup. Mitigated by daily backups.
2. **No streaming export.** Daily batch export is the only way to get data out. Real-time analytics (e.g., live dashboards over Kafka) are v2.
3. **No GDPR-style audit log.** Right-to-deletion is supported (24.6) but doesn't log the deletion request itself in a tamper-proof way. v2 may add an immutable audit trail.
4. **Image files stored on local disk.** If disk fills up, sandbox stops accepting new requests. Operations alert on disk usage > 80% (Section 25.7).
5. **No encryption at rest within SQLite.** The SQLite file is not encrypted by SQLite itself; encryption is at the OS / disk volume level. v2 may add SQLCipher or migrate to Postgres with TDE.

These limitations are acceptable for v1 pilot deployment and are tracked for v2.

## Section 25. Monitoring

### 25.1 Purpose

Section 25 specifies the observability stack for the tomato sandbox. Monitoring covers:

- **Metrics:** quantitative signals exposed via Prometheus on the `/metrics` endpoint (Section 20.3)
- **Logs:** structured logs emitted by the orchestrator (Section 21.9) and persisted in SQLite (Section 24)
- **Tracing:** OpenTelemetry spans for distributed tracing (Section 22.8)
- **Alerts:** thresholds on metrics that trigger pages to operations
- **Dashboards:** Grafana panels visualizing the metrics

The goal is operational: detect quality regressions, latency anomalies, and capacity issues quickly enough to act before users notice.

### 25.2 Metrics (Prometheus format)

The sandbox exposes Prometheus metrics at `GET /metrics`. The metrics fall into categories:

**Request-level counters and histograms:**

| Metric name | Type | Labels | Description |
|---|---|---|---|
| `tomato_requests_total` | counter | endpoint, status_code | Total requests received |
| `tomato_request_duration_ms` | histogram | endpoint | Request latency distribution |
| `tomato_image_decode_failures_total` | counter | format | Image decode failures by detected format |
| `tomato_iqa_reject_total` | counter | reason | IQA rejections by reason |

**Tier-level metrics:**

| Metric name | Type | Labels | Description |
|---|---|---|---|
| `tomato_tier_assignments_total` | counter | tier_label, primary_class | Tier outcomes by class |
| `tomato_tier5_alerts_total` | counter | reason, trigger_class | T5 alerts fired |
| `tomato_rule_fired_total` | counter | rule_id, sub_rule_id | Which rule fired in tier chain |

**Pipeline component metrics:**

| Metric name | Type | Labels | Description |
|---|---|---|---|
| `tomato_signal_duration_ms` | histogram | signal_name | Per-signal latency |
| `tomato_signal_failures_total` | counter | signal_name, error_type | Signal failures (error_type values: `cuda_oom`, `exception`, `timeout`, `nan_output`, `forward_failed`) |
| `tomato_classifier_nan_total` | counter | (none) | NaN-in-classifier-output occurrences |
| `tomato_tta_fired_total` | counter | view_count | TTA invocations |
| `tomato_gpu_lock_wait_ms` | histogram | (none) | Lock acquisition latency |
| `tomato_gpu_lock_timeout_total` | counter | (none) | Lock timeouts |

**Quality metrics:**

| Metric name | Type | Labels | Description |
|---|---|---|---|
| `tomato_conformal_coverage_actual` | gauge | (none) | Empirical coverage on agronomist-labeled samples; reports last computed value with a timestamp; if no fresh labels in 7+ days, reports NaN with a `stale=true` label dimension |
| `tomato_calibration_drift` | gauge | (none) | KL divergence between current and last-calibrated distribution; computed hourly via batch over recent predictions |
| `tomato_psv_reliability_avg` | gauge | (none) | Rolling average PSV reliability (last 100 requests) |
| `tomato_severity_omitted_rate` | gauge | (none) | Fraction of cases where severity is omitted |
| `tomato_severity_correction_rate` | gauge | (none) | Fraction of agronomist dispositions that include a severity correction (proxy for severity grading accuracy) |

**Queue metrics:**

| Metric name | Type | Labels | Description |
|---|---|---|---|
| `tomato_queue_pending_count` | gauge | priority | Pending cases by priority |
| `tomato_queue_in_review_count` | gauge | priority | In-review cases by priority |
| `tomato_queue_review_duration_ms` | histogram | priority, resolution_type | Time from submission to resolution |
| `tomato_queue_resolutions_total` | counter | resolution_type | Disposition counts |

The metrics are scraped by Prometheus every 15 seconds and stored with 90-day retention. v2 may add longer-term storage via Thanos or M3.

### 25.3 Per-tier metrics

The `tomato_tier_assignments_total` counter is the primary metric for tier distribution analysis. From it, derived metrics include:

- Tier 1 fraction = `Tier 1 count / total count` (target: > 60% in steady state for diseased samples on training data; will be lower in real field deployment)
- Tier 4B rate = `Tier 4B count / total count` (target: < 1%; spike indicates pipeline issue)
- Tier 5 alert rate = `Tier 5 count / Tier 1+2 count` (target: aligns with epidemiological prevalence of late_blight in the region)

These derived metrics are computed in Grafana from the raw counters; they are not separate Prometheus metrics.

### 25.4 Per-rule metrics

The `tomato_rule_fired_total` counter tracks which rule fires for each prediction. Important watches:

- Rule 9 (catch-all) firing rate: should be near zero. If > 5% of requests trigger Rule 9, the rule chain has a coverage gap that should be investigated.
- Rule 1 (signal failure) firing rate: tracks pipeline reliability. > 1% indicates signal stability issues.
- Sub-rule 7a / 8a (DEGRADED IQA) firing rate: tracks image quality in the field. Rising rate may indicate user-side issues (e.g., camera or lighting changes in the user population) or model drift in IQA.
- Sub-rule 7b / 8b (underpowered class) firing rate: tracks model class balance. Rising rate may indicate dataset drift.

### 25.5 Latency metrics

Latency P95 is the primary SLO target. Per Section 21.7, the target is 550 ms for single-image requests in idle conditions. The monitoring tracks:

| SLO | Target | Alert threshold |
|---|---|---|
| Single-image P95 latency | 550 ms | > 1000 ms for 5 minutes |
| Single-image P99 latency | 800 ms | > 1500 ms for 5 minutes |
| Multi-image P95 latency (N=5) | 2.0 s | > 3.0 s for 5 minutes |
| GPU lock P95 wait | < 50 ms | > 200 ms for 5 minutes |
| GPU lock P99 wait | < 200 ms | > 2000 ms for 1 minute (HIGH-priority alert; matches the threshold cited in Section 21.7) |

Latency anomalies often indicate capacity issues; the alert thresholds are calibrated for pilot load.

### 25.6 Quality metrics

**Conformal coverage tracking:**

The conformal prediction set is calibrated to 90% empirical coverage at deployment (Section 13). Coverage drift indicates model degradation:

- The system periodically samples a holdout set (Section 24's `predictions` rows where the agronomist has provided ground-truth via queue resolutions)
- Computes the fraction where `agronomist_truth_class IN prediction_set`
- The metric `tomato_conformal_coverage_actual` reports this rolling average
- Alert: if coverage drops below 85% for 7 days, trigger recalibration (Section 13)

This is asynchronous and depends on agronomist labels accumulating; v1 expects ~50-100 labeled examples per week from the queue, sufficient to detect a 5-percentage-point drop within 2-3 weeks.

**Calibration drift:**

Temperature scaling calibration (Section 12.10) is fit at deployment. As production data drifts from training distribution, the calibration becomes stale. The system tracks:

- Predicted confidence distribution (binned histogram of `combined_max_prob`)
- Compared to expected distribution from training
- KL divergence between the two

Alert: KL divergence > 0.5 for 7 days suggests calibration drift; trigger temperature re-fitting.

**Per-class quality:**

For each class, track:
- Precision (when system says class X, fraction where agronomist confirms class X)
- Recall (when actual class is X, fraction where system predicted class X)
- F1 score

These require agronomist labels; available only for cases routed to the queue. Sample sizes are small for rare classes; per-class metrics are only meaningful with > 50 labels per class per month.

### 25.7 Alerting

Alert routing:

| Severity | Examples | Channel | Response time |
|---|---|---|---|
| CRITICAL | sandbox down, GPU OOM repeated, queue overflow | On-call paging system (PagerDuty or equivalent), on-call rotation | 15 minutes |
| HIGH | latency P99 elevated > 5 min, conformal coverage drop, GPU lock P99 > 2s for 1 min | On-call paging system (business hours rotation) | 2 hours |
| MEDIUM | calibration drift, queue capacity warning (200+ pending for 4h), image disk > 80% | Slack `#tomato-ops` | 1 business day |
| LOW | informational metrics summary | Slack `#tomato-ops` (digest) | weekly |

Alert sources: Prometheus alertmanager configured per the rules above. Alert rules are committed to git in the operations repo (not in `tomato_sandbox/`).

The on-call rotation is the operations team at NanoFarm. v1 has 1 primary + 1 backup on a weekly rotation; the rotation policy is operations-team-managed.

### 25.8 Dashboards

The primary Grafana dashboards. Each panel notes its data source: Prometheus metrics (P), SQLite queries (S), or both (P+S):

1. **System health overview (P):** request volume, latency P50/P95/P99, error rate by category, GPU lock wait time
2. **Tier distribution (P):** stacked time-series of tier assignments, T5 alert rate, rule firing breakdown
3. **Quality metrics (P+S):** conformal coverage and calibration drift come from Prometheus gauges; per-class precision/recall is computed from SQLite queries joining `predictions` with `queue_cases` resolutions
4. **Queue health (S):** pending count by priority, review time histograms, resolution rate, agronomist throughput, all derived from SQLite
5. **Capacity (P):** GPU memory use, disk use, request queue depth, lock wait distribution
6. **Per-request drill-down (S):** look up a specific request_id, see the full trace including all signal outputs from `predictions` and `per_signal_logs`
7. **T5 alert ground-truth (S):** for T5 alerts where the agronomist has dispositioned (subset of T5 cases routed to queue), shows confirmation rate, correction rate, and disposition reasons. This is the primary safety dashboard for monitoring whether the dangerous-disease alert is precise.

The dashboards are JSON-as-code in the operations repo. v1 starts with 7 panels; v2 may expand based on operational needs.

### 25.9 Limitations

1. **No real-time anomaly detection.** Alerts are threshold-based; smarter anomaly detection (changepoint analysis, seasonal-aware) is v2.
2. **Quality metrics depend on agronomist labels.** Without enough labels, per-class precision/recall is noisy. Pilot deployment will surface whether the queue produces enough labels.
3. **Conformal coverage measurement has a sampling bias.** The coverage metric is computed only on cases that routed to the agronomist queue (T5-firing cases or flagged-ambiguous cases). For the large majority of Tier 1 cases (T5=False, not ambiguous), there are NO ground-truth labels because they don't route to the queue. This means the measured `tomato_conformal_coverage_actual` is conditional on (T5 fires OR ambiguous), not over the full population. The true population-level coverage may differ. Pilot mitigation: a periodic random sample of non-routed Tier 1/2 cases will be sent to the queue for spot-check labeling (operations-managed at perhaps 1% sampling rate); this provides an unbiased coverage estimate at the cost of agronomist time. This sampling is out of scope for v1 and is a v2 enhancement; v1 reports the conditional coverage metric with a clear "biased toward routed cases" annotation in the dashboard.
4. **No a/b testing infrastructure.** Comparing two model versions requires manual log analysis; v2 may add deployment-aware metrics with version labels.
5. **No user-segment analysis.** Metrics are aggregated across all users; understanding whether quality varies by region, cultivar, or seasonal pattern requires custom queries against SQLite.
6. **Tracing is optional.** OTLP endpoint is env-var-set; if unset, traces are dropped. v1 may run without distributed tracing during pilot.
7. **No cost tracking.** GPU compute cost per request is not measured. v2 may add this for capacity planning.

These limitations are tracked for v2; v1 monitoring is sufficient for pilot operations and quality regression detection.

---

## Section 26. Engineering hygiene

### 26.1 Purpose

Section 26 specifies engineering practices for the tomato sandbox: testing strategy, code review process, CI configuration, dependency management, and code quality gates. These practices are NOT optional; they enforce that v1 ships at the quality bar appropriate for a system that influences agronomic decisions.

The practices apply to all code under `tomato_sandbox/` (per the Sandbox Directive). They do NOT apply to the legacy APIN code at port 8766 (which is sacred per the Sandbox Directive and not modified).

### 26.2 Testing strategy

Tests are organized in three layers:

**Unit tests** (`tomato_sandbox/tests/unit/`):
- Test individual functions and classes in isolation
- Mock external dependencies (model weights, GPU, database)
- Coverage targets to achieve before pilot deployment: 80% line coverage overall; 100% coverage of `tier/rules.py` and `tier/tier_assignment.py` (the rule chain is safety-critical). The targets are aspirational at v1 start; CI tracks coverage and the merge gate is set per Section 26.6.
- Run on every CI commit
- Expected runtime: under 60 seconds total (estimate; actual may differ once test count is final)

**Integration tests** (`tomato_sandbox/tests/integration/`) come in two types:
  - **Type A: full-pipeline tests.** A handful of representative cases (~10-20) that exercise the full pipeline with real model weights and synthetic image inputs. These verify that the orchestrator correctly drives the components end-to-end.
  - **Type B: rule-chain tests (Section 15 scenarios).** All 135 scenarios from Section 15 encoded as tests with synthetic intermediate outputs (v3 probs, LoRA probs, PSV outputs). These do NOT load model weights; they directly call `assign_tier()` with crafted inputs. They are fast (~10ms each) and deterministic.
- Run on every CI commit
- Expected runtime: under 5 minutes (estimate; the 135 rule-chain tests at ~10ms each are ~2 seconds; the full-pipeline tests at ~1s each are ~20 seconds; the rest is fixture loading and pytest overhead). Actual runtime is measured and the budget is revisited if exceeded.

**End-to-end tests** (`tomato_sandbox/tests/e2e/`):
- Spin up the full sandbox process (port 8767) with real models
- Submit real images via HTTP and validate response structure
- Validate Phase E logging writes correct rows
- Run on a staging environment, not on every PR
- Expected runtime: under 30 minutes

The Section 15 scenarios are the heart of the integration test suite. Each of the 135 scenarios is one test function (Section 15.16 specifies the test pattern). Phase F.0 (Section 29) extends this with real-data validation.

### 26.3 Code review process

All changes to `tomato_sandbox/` go through pull request review:

- At least 1 approving reviewer required
- For changes touching `tier/rules.py` or `tier/tier_assignment.py`: 2 approving reviewers required (one of whom should be a domain expert from the agronomic team or senior engineering)
- For changes touching the Section 15 scenarios or their tests: review by the spec author or designated successor is required; spec ownership is documented in the operations repo and updated when ownership transitions
- Reviewers check: test coverage, spec compliance, error handling, log emission

The PR template asks reviewers to confirm:
1. Tests added or updated for the change
2. Section reference updated if behavior changed
3. No mid-correction artifacts in code or comments
4. No PII or secrets in test fixtures
5. CHANGELOG entry added if user-visible behavior changed

PRs that fail CI cannot be merged. CI failures must be addressed (not skipped or overridden) except for documented flake patterns.

### 26.4 Continuous integration

CI runs on every PR and on every push to main:

| Stage | Trigger | Duration | Blocks merge? |
|---|---|---|---|
| Lint (ruff, black, mypy) | Every PR | 30 sec | yes |
| Unit tests | Every PR | 60 sec | yes |
| Integration tests | Every PR | 5 min | yes |
| Spec compliance check (cross-ref validation) | Every PR | 10 sec | yes |
| Coverage report | Every PR | 30 sec | yes if below 80% |
| Build sandbox image (Docker) | Every push to main | 5 min | no (post-merge) |
| Deploy to staging | Manual trigger | 10 min | no |
| End-to-end tests on staging | Post-staging-deploy | 30 min | no (alerts on failure) |

The spec compliance check parses the spec markdown files and validates that every cross-reference (e.g., "Section 14.5") points to a real section. This catches the kind of stale cross-references that Round 8 audits keep finding. The check runs a small Python script (~100 lines) at `tomato_sandbox/tools/spec_compliance_check.py` that reads all `sections_*.md` and `section_*.md` files, extracts section IDs and cross-refs, and reports any cross-ref that doesn't resolve. The script is part of the v1 deliverable (not a v2 enhancement).

CI configuration lives in `.github/workflows/sandbox-ci.yml`. It uses GitHub Actions runners; for GPU-dependent tests, a self-hosted runner with an RTX 4060 is used.

### 26.5 Dependency management

Dependencies are pinned via `tomato_sandbox/requirements.txt` with exact versions. Illustrative subset (the actual pinned versions are determined at deployment time and verified for cross-package compatibility; e.g., torch and torchvision must use a paired release):

```
torch==2.11.0       # matches the development environment; CUDA 13 compatible
torchvision==<paired>  # version paired with torch 2.11 release
fastapi==0.115.0
pydantic==2.9.0
numpy==2.1.0
opencv-python==4.10.0.84
sqlalchemy==2.0.30
prometheus-client==0.21.0
opentelemetry-api==1.27.0
opentelemetry-sdk==1.27.0
opentelemetry-instrumentation-fastapi==0.48b0
... (full list at deployment time, generated via `pip freeze` of a tested environment)
```

Dependency updates go through:
1. Renovate or Dependabot opens a PR with the new version
2. CI runs (catches regressions)
3. Manual review of changelog for the new version
4. Merge if compatible, defer if not

Major version bumps (e.g., torch 2.x to 3.x) require a separate epic and validation pass; they are not merged through routine dependency updates.

### 26.6 Code quality gates

Code that lands in main must pass:

- **Type checking:** mypy with strict mode on `tomato_sandbox/`. `# type: ignore` is permitted for third-party libraries without type stubs or for legitimate dynamic patterns; each ignore requires an inline comment explaining why and reviewer approval. The strict-mode goal is to catch most type errors, not to achieve zero ignores.
- **Linting:** ruff with rules from `pyproject.toml`. No warnings in CI; new warnings block merge.
- **Formatting:** black with line length 100. Auto-applied by pre-commit hook; CI verifies.
- **Documentation:** every public function has a docstring with Args, Returns, Raises sections. Reviewer checks.
- **Test coverage:** 80% overall, 100% for safety-critical modules (tier/, orchestrator/nan_guards.py, orchestrator/degraded_mode.py).

Pre-commit hooks (`.pre-commit-config.yaml`) auto-run linting and formatting locally; this catches issues before push.

### 26.7 Logging and observability standards

All log emissions follow these standards:

- Use `structlog` for structured logging; never `print()` in production code
- Every log line has at minimum: `request_id`, `step`, `succeeded`, `duration_ms`
- Log levels: DEBUG (verbose, off in production), INFO (normal flow), WARNING (degraded behavior), ERROR (failures), CRITICAL (system unavailable)
- Sensitive fields (user_metadata, image bytes) are NEVER logged at INFO or above
- Stack traces are logged on ERROR; never swallow exceptions silently
- Metrics emission (Section 25) happens alongside log emission, not as a substitute

Log aggregation is operations-managed (typically routed to a central log store like Loki or Elasticsearch); the sandbox emits to stdout in JSON format and the host's log shipper handles the rest.

### 26.8 Security practices

Security baseline for v1:

- All endpoints require authentication via the unified server (port 8005); the sandbox at 8767 trusts the unified server (network-level isolation)
- TLS termination at the unified server; sandbox-to-unified communication is plaintext on localhost
- Secrets (model paths, database paths, OTLP endpoint) come from env vars; never committed to git
- No user-controlled file paths in any API
- Image upload size limited to 10 MB (Section 16.9)
- Request body size limited to 50 MB (for multi-image)
- Rate limiting at the unified server: 60 requests per minute per authenticated user (configurable via `UNIFIED_RATE_LIMIT_RPM`). The rate limit key is the authenticated `user_id` from the auth header when present; for unauthenticated paths (rare, mostly health/info) the key is the source IP address.
- No SQL injection risk: all queries parameterized via SQLAlchemy or `sqlite3.execute(query, params)`
- No path traversal: image paths are constructed from `request_id` (uuid), never from user input

Security audit before pilot deployment:
- Run `bandit` on the sandbox codebase, address all HIGH severity findings
- Run `pip-audit` for known dependency vulnerabilities
- Manual review of authentication and authorization flow

These practices are baseline; they are not sufficient for high-stakes production deployment. v2 may add SSO, audit logs, and formal threat modeling.

### 26.9 Limitations

1. **No fuzz testing.** The pipeline is not fuzz-tested with adversarial inputs; pathological JPEG / PNG inputs may cause crashes that aren't caught by unit tests. v2 may add fuzzing.
2. **No load testing in CI.** Latency under load is measured manually before deployment; no continuous load testing. v2 may add k6 or locust load tests in CI.
3. **No mutation testing.** Test quality is measured by line coverage only, not by mutation testing (which would detect tests that pass even when the code is wrong). v2 may add mutation testing.
4. **No formal verification.** The tier rule chain is a complex if-else cascade; formal verification (e.g., with Z3) could prove correctness for all input combinations. Out of scope for v1; the 135 scenarios in Section 15 are the closest equivalent.

## Section 27. API documentation (OpenAPI)

### 27.1 Purpose

Section 27 specifies the API documentation deliverable: an OpenAPI 3.1 specification covering both the unified server (port 8005) and the tomato sandbox (port 8767). The OpenAPI spec is the source of truth for API consumers (frontend developers, third-party integrators, the agronomist UI team).

The spec lives at `tomato_sandbox/api/openapi.yaml`. CI validates that the spec matches the actual FastAPI route definitions (FastAPI auto-generates OpenAPI; the committed YAML must match the auto-generated version).

### 27.2 OpenAPI structure

The OpenAPI document has the standard sections:

- `info` block with title, version, description, contact
- `servers` block listing the unified server URL and (for testing) the direct sandbox URL
- `paths` block with all endpoints
- `components.schemas` block with all request and response shapes
- `components.securitySchemes` block with API key auth definition; this applies to unified-server endpoints only. Sandbox direct endpoints (port 8767) are unauthenticated; they trust network-level isolation per Section 26.8. The OpenAPI spec annotates this clearly so that a developer using the sandbox direct doesn't expect auth, and a developer using the unified server endpoint knows auth is required.
- `tags` block for grouping endpoints

The OpenAPI spec is published at `/openapi.json` on both unified server and sandbox; FastAPI's auto-generated Swagger UI is at `/docs` on each. The committed YAML version at `tomato_sandbox/api/openapi.yaml` is the source of truth for code generation tools (e.g., `openapi-generator` for client SDKs).

### 27.3 Endpoints documented

Endpoint coverage:

**Unified server (port 8005):**

| Path | Method | Tag |
|---|---|---|
| `/predict` | POST | unified-prediction |
| `/predict_multi` | POST | unified-prediction |
| `/health` | GET | infrastructure |
| `/ready` | GET | infrastructure |
| `/metrics` | GET | infrastructure |
| `/info` | GET | infrastructure |

**Tomato sandbox (port 8767):**

Same endpoints as unified server, but returning unwrapped responses (Section 16.10). Documented as a separate `servers` entry in OpenAPI; consumers of the sandbox directly (typically test infrastructure) get the unwrapped response.

**Agronomist queue API (port 8768 default; operations may override via `QUEUE_API_PORT` env var):**

| Path | Method | Tag |
|---|---|---|
| `/queue/cases` | GET | agronomist-queue |
| `/queue/cases/{case_id}` | GET | agronomist-queue |
| `/queue/cases/{case_id}/claim` | POST | agronomist-queue |
| `/queue/cases/{case_id}/resolve` | POST | agronomist-queue |
| `/queue/cases/{case_id}/dismiss` | POST | agronomist-queue |
| `/queue/cases/{case_id}/escalate` | POST | agronomist-queue |
| `/queue/stats` | GET | agronomist-queue |

### 27.4 Schemas documented

The schemas section includes:

- `PredictRequest`, `PredictMultiRequest` (request bodies)
- `UnifiedResponse` (envelope per Section 16.10)
- `SandboxResponse` (unwrapped per Section 16.2)
- `ErrorResponse` (per Section 16.9)
- `TierBlock`, `PredictionBlock`, `Tier5AlertBlock`, `SeverityBlock`, `ExplanationBlock`, `VisualizationBlock`, `AgronomistQueueBlock` (sub-schemas of SandboxResponse)
- `QueueCase`, `QueueDisposition` (per Section 23.3, 23.6)
- `InfoResponse` (per Section 20.3)

Each schema has `type`, `properties`, `required`, `example`. Examples are drawn from Section 15 scenarios where relevant.

### 27.5 Examples

The OpenAPI spec includes worked examples for each endpoint. For `/predict`:

- Example 1: Tier 1 confident foliar prediction (S1.1 from Section 15)
- Example 2: Tier 3A two-class ambiguity (S3A.1)
- Example 3: Tier 4B pipeline failure (S4B.1)
- Example 4: Error response (IQA_REJECTED)

Each example has full request and response payloads, fields populated as in Section 15.

### 27.6 Versioning

API versioning strategy:

- The OpenAPI `info.version` field tracks the API version (semver)
- v1 starts at `1.0.0`
- Breaking changes (removing a field, changing a type) bump major version
- Additive changes (adding optional fields) bump minor version
- Bug fixes bump patch version

There is NO URL-based versioning (e.g., `/v1/predict`); the URL is stable. Version negotiation happens via the `X-API-Version` header (defaults to latest if absent). `X-API-Version` is preferred over `Accept-Version` because the `X-` prefix marks it as a custom application header per common convention, while `Accept-Version` is non-standard despite occasional use in the wild.

The unified server can serve multiple API versions simultaneously during a transition; v1 ships with only `1.0.0`.

### 27.7 Generation and validation

OpenAPI generation:
- FastAPI auto-generates OpenAPI from route definitions and Pydantic schemas
- A pre-commit hook regenerates `tomato_sandbox/api/openapi.yaml` and asserts it matches the committed version
- CI validates the spec with `openapi-spec-validator`

Validation:
- The committed YAML is the contract; the auto-generated version must match
- Mismatches block PR merge
- Frontend / integrator code generation (`openapi-generator`) consumes the committed YAML

### 27.8 Limitations

1. **No server-side request validation against OpenAPI in v1.** FastAPI's Pydantic models validate request bodies; this is consistent with OpenAPI but not formally tied to it. v2 may add OpenAPI-driven request middleware.
2. **No SDK generation in v1.** Consumers either hand-write API clients or generate from the OpenAPI YAML themselves. v2 may publish an official Python SDK.
3. **No rate-limit documentation in OpenAPI.** Rate limits (Section 26.8) are documented in prose, not in OpenAPI extensions. v2 may add `x-rate-limit` extensions.
4. **No webhook documentation.** The agronomist queue may notify external systems via webhooks in v2; not in v1, not in OpenAPI.

## Section 28. Deployment and operations

### 28.1 Purpose

Section 28 specifies how the tomato sandbox is deployed, configured, monitored, and operated in pilot. This is the runbook for the operations team at NanoFarm.

### 28.2 Deployment topology (v1 pilot)

v1 pilot deployment runs on a single machine. The development environment per project records is a Windows 11 laptop with RTX 4060 (8GB VRAM); production / pilot deployment uses Linux (Ubuntu 24.04 LTS or similar) for systemd compatibility and operational tooling. The same RTX 4060 hardware can be used for development and pilot if dual-booted; or a separate Linux pilot host is provisioned. RAM and storage are sized appropriately for pilot scale (typical: 32 GB RAM, 1 TB NVMe SSD; operations confirms before deployment).

- Hardware: RTX 4060 GPU (8GB VRAM), Linux host
- Network: behind NanoFarm's office firewall; pilot users tunnel via VPN
- Services on the host:
  - APIN (port 8766) - existing, unmodified
  - Tomato sandbox (port 8767) - new, this spec
  - Unified server (port 8005) - existing with tomato routing additions per Section 22
  - Agronomist queue API (port 8768 default per Section 27.3)
  - Frontend web app served via nginx (port 443)
  - SQLite file storage at `/var/lib/tomato_sandbox/sandbox.db`
  - Image storage at `/var/lib/tomato_sandbox/images/`
  - Visualization storage at `/var/lib/tomato_sandbox/visualizations/`

A second machine (operations laptop) runs:
- Prometheus and Grafana (for metrics; Section 25)
- The agronomist UI (separate frontend app)
- Backup scripts (Section 24.7)

**Deployment artifact:** the v1 spec supports two deployment styles. Style A (recommended for pilot) uses systemd directly (Section 28.3); Style B uses Docker images (built in CI per Section 26.4) for portable deployment to staging or for v2 cloud deployment. Pilot uses Style A for simplicity; Docker build in CI is for forward-compatibility with v2.

**VRAM contention.** Both APIN and the tomato sandbox load model weights into GPU memory: APIN's weights total ~200MB plus inference buffers ~1-2GB; tomato sandbox loads v3 (~200MB) + LoRA (~50MB) + classifier (small) plus inference buffers ~3-4GB. Combined working set is approximately 5-6GB on the 8GB RTX 4060, with little headroom. v1 pilot mitigates contention through the GPU lock pattern (Section 20.6), which serializes GPU access within the sandbox; APIN already serializes its own requests. If both services are heavily loaded concurrently, the operational mitigation is to time-multiplex (one runs at a time) or move APIN to a different machine. A v2 production deployment would size GPU resources separately per service.

v2 production deployment is out of scope for this spec; it would migrate to cloud infrastructure (AWS or equivalent), separate the services, and add HA.

### 28.3 Process management

Each service runs as a systemd unit:

```ini
# /etc/systemd/system/tomato-sandbox.service
[Unit]
Description=Tomato disease detection sandbox
After=network.target

[Service]
Type=simple
User=tomato-sandbox
WorkingDirectory=/opt/tomato_sandbox
EnvironmentFile=/etc/tomato_sandbox/env
ExecStart=/opt/tomato_sandbox/.venv/bin/uvicorn tomato_sandbox.api.server:app --host 127.0.0.1 --port 8767 --workers 1
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

systemd handles:
- Automatic restart on crash (with backoff)
- Log capture to journald
- Service dependencies (sandbox starts after networking)
- Boot-time start

For development, the sandbox runs via `uv run uvicorn ...` directly without systemd.

### 28.4 Configuration

Configuration sources (Section 20.7):
1. Environment variables (highest precedence; all `TOMATO_*` namespace)
2. `/etc/tomato_sandbox/local.yaml` (operations-managed local overrides)
3. `tomato_sandbox/config/default.yaml` (committed defaults)
4. Hardcoded fallbacks in `tomato_sandbox/config.py`

Calibration files at deployment:
- `/etc/tomato_sandbox/calibration/tomato_calibration.json` (conformal tau, fitted by F.0)
- `/etc/tomato_sandbox/calibration/iqa_thresholds.json`
- `/etc/tomato_sandbox/calibration/severity_thresholds.yaml`

These files are NOT in source control; they are produced by F.0 (Section 29) and copied to the deployment host.

### 28.5 Bringup procedure

To bring up the tomato sandbox on a fresh host:

1. Provision Ubuntu 24.04 with NVIDIA driver, CUDA 13, Python 3.13
2. Create system user `tomato-sandbox`
3. Clone the repo to `/opt/tomato_sandbox`
4. Install dependencies: `uv venv && uv pip install -r tomato_sandbox/requirements.txt`
5. Copy model weights to `/var/lib/tomato_sandbox/models/`
6. Copy F.0 calibration files to `/etc/tomato_sandbox/calibration/`
7. Write env file at `/etc/tomato_sandbox/env`
8. Install and enable the systemd unit
9. Start the service: `systemctl start tomato-sandbox`
10. Verify health: `curl localhost:8767/ready`
11. Update unified server config to enable tomato routing (`UNIFIED_TOMATO_ROUTE_ENABLED=true`)
12. Restart unified server
13. Run smoke test via the unified server endpoint

This bringup is operations-managed; the spec is the runbook reference.

### 28.6 Rollout strategy

Pilot rollout in stages:

**Stage 0: Internal validation**
- Sandbox runs in shadow mode: receives requests but responses are not surfaced to users
- Sandbox responses are recorded alongside current production responses (legacy 23-class EfficientNetV2-S model) for the same image
- Engineering and agronomic team review disagreements between the two systems. Disagreement is EXPECTED (the new system is supposed to be better on field photos per the lab-to-field domain gap motivation); the review confirms that disagreements favor the new system rather than introducing new errors.
- Duration: 2-4 weeks
- Exit criteria: zero critical bugs surfaced; manual review of >= 100 disagreement cases shows the new system is right at least as often as the legacy system; agronomic team approves moving to Stage 1. Note: a high agreement rate is NOT the criterion; the new system is supposed to make different (better) calls on field photos.

**Stage 1: Internal user pilot**
- 5-10 NanoFarm internal users (agronomists, engineering team) get tomato responses from the sandbox
- Feedback collected via in-app feedback button and weekly retrospectives
- Duration: 4 weeks
- Exit criteria: < 5% Tier 4B rate; agronomist queue stays under 50 pending cases; no safety incidents (defined: any false negative on a dangerous-disease case that would have caused real-world harm if not caught by downstream review)
- **Safety incident response:** any safety incident in Stage 1 triggers immediate rollback to legacy model (set `UNIFIED_TOMATO_ROUTE_ENABLED=false`), post-incident review, root-cause fix, and Stage 1 restart. The clock does not carry over from before the incident.

**Stage 2: Extension officer pilot**
- 50-100 NanoFarm-affiliated extension officers in select Kerala districts
- Users are trained on the system before getting access
- Duration: 8-12 weeks. The extended duration is necessary to span at least one full disease cycle and seasonal pattern. Most fungal diseases progress on a 2-4 week cycle; a multi-week pilot catches the full progression including treated and untreated trajectories. Shorter pilots miss seasonal effects (e.g., humidity-driven late_blight outbreaks).
- Exit criteria: tier distribution matches expected agronomic prevalence; < 10% Tier 4B rate; agronomist queue SLA met (Section 23.7); positive user feedback; no safety incidents

**Stage 3: Limited public release**
- Public access via NanoFarm app, one Kerala district at a time
- Each new district adds ~1000 users
- Duration: ongoing
- Exit criteria per district: 4 weeks of stable operations before adding the next district

The rollout is reversible: at any stage, the unified server can disable tomato routing (set `UNIFIED_TOMATO_ROUTE_ENABLED=false`) and traffic falls back to the legacy 23-class model. This is the safety valve.

### 28.7 Operational runbooks

Common operational scenarios with runbook procedures:

**Scenario: Bad deploy causes Tier 4B spike or other regression**
1. Identify the deploy: check git log for recent merges to main and the deploy timestamp
2. Disable tomato routing at the unified server immediately: `UNIFIED_TOMATO_ROUTE_ENABLED=false`, restart unified server. Traffic falls back to legacy 23-class model. This takes < 1 minute and is the fastest mitigation.
3. Investigate the regression: compare sandbox metrics to baseline; identify what changed
4. If a code rollback is needed: revert to the previous git tag, rebuild, redeploy the sandbox via the bringup procedure (Section 28.5 step 4 onward)
5. Verify metrics return to baseline after rollback
6. Re-enable tomato routing once verified
7. Post-incident review per Section 28.8

**Scenario: Sandbox latency P95 > 1000 ms for > 5 min (Section 25.5 alert)**
1. Check `/metrics` for which step is slow (signal_a, signal_b, etc.)
2. Check GPU utilization with `nvidia-smi`
3. If GPU is saturated: reduce multi-image limit via `TOMATO_MULTI_IMAGE_MAX_N=3`
4. If a specific signal is slow: check that signal's logs for errors
5. If unresolved within 30 min: page senior engineer

**Scenario: Conformal coverage drops below 85% for 7 days (Section 25.6 alert)**
1. Verify the coverage calculation has fresh agronomist labels (last 7 days)
2. Check if the labeled distribution has shifted (e.g., new disease region)
3. If drift confirmed: trigger F.0 re-calibration job
4. Deploy new calibration file to `/etc/tomato_sandbox/calibration/`
5. Restart sandbox to pick up new calibration

**Scenario: Pending queue size > 500 cases (Section 23.7 alert)**
1. Check agronomist availability (any out sick, on leave?)
2. If capacity gap is temporary: ride it out; no action
3. If sustained gap: disable `route_ambiguous_to_queue` flag, focus queue on T5 only
4. Communicate to extension officers: response times will be longer for ambiguous cases

**Scenario: Tier 4B rate spikes > 5% (Section 25 quality alert)**
1. Check `tomato_signal_failures_total` for which signal is failing
2. Check `nvidia-smi` for GPU OOM events
3. Check sandbox logs for stack traces
4. If signal-specific: roll back the most recent deploy
5. If general: increase GPU lock timeout, restart sandbox

These runbooks live in the operations repo, not in the sandbox repo. Section 28 is reference; the operations team owns the actual runbooks.

### 28.8 Incident response

For incidents (sandbox down, data loss, security event):

1. Identify (alert fires, user reports issue)
2. Acknowledge (on-call engineer takes ownership)
3. Mitigate (rollback, restart, traffic shift)
4. Communicate (status page, user notification if widespread)
5. Resolve (deploy fix or workaround)
6. Post-incident review (write up; identify systemic improvements)

Severity classification:
- SEV1: Service down or data integrity compromised; respond within 15 min
- SEV2: Significant degradation; respond within 1 hour
- SEV3: Quality regression but service operational; respond within 1 business day

Post-incident reviews are blameless; the focus is systemic improvement, not individual accountability.

### 28.9 Limitations

1. **Single-host deployment is a SPOF.** Hardware failure means downtime. Mitigation: daily backups (Section 24.7); recovery within ~2 hours by restoring to a different host.
2. **No automated rollback.** Rollbacks are manual (operations runs `git revert` and redeploys). Automated rollback (e.g., on deploy-time canary failure) is v2.
3. **No blue-green deployment.** v1 deploys cause ~30s downtime. Tolerable for pilot; not for production.
4. **Manual TLS renewal.** Let's Encrypt cert renewal is via certbot cron; no automation around handling renewal failures.
5. **No DR plan.** If the primary host fails AND backups are corrupted, all data is lost. Mitigation: weekly off-site backup. Full DR plan (multi-region replication) is v2.
6. **Operations team is small.** NanoFarm's operations team is 2 engineers; on-call burden is high. Pilot scale is sustainable; production scale requires hiring or outsourcing.

## Section 29. Phase F.0 validation suite

### 29.1 Purpose

Phase F.0 is the final validation phase before pilot deployment. It exercises the full pipeline against real data with agronomist-confirmed labels and produces:

- Calibration files for conformal tau, IQA thresholds, severity thresholds
- Validation metrics (precision, recall, F1, conformal coverage, calibration error)
- Pilot go/no-go recommendation

F.0 is NOT a generic test suite; it is a one-time validation pass that runs on a held-out test dataset. The 135 scenarios in Section 15 are the unit/integration test counterpart; F.0 is the empirical validation at scale.

### 29.2 F.0 dataset

F.0 uses a held-out dataset of approximately 3000 tomato images:

- Source: Roboflow Universe + iNaturalist + Mendeley + Kerala extension officer photos
- Class balance: roughly 400-500 per class (foliar, septoria, late_blight, ylcv, mosaic, healthy) plus ~200 OOD samples. This balance is intentional for evaluation purposes, not reflective of real-world prevalence (where healthy is much more common than diseased and class distribution is highly skewed). Per-class roughly-equal sampling ensures every class has enough samples to compute meaningful F1 scores.
- Field photo fraction: > 70% (intentional bias toward field photos to test domain shift handling)
- Agronomist-confirmed labels: every image reviewed by at least one agronomist; high-confidence labels only

The dataset is partitioned:
- 60% calibration (1800 samples) - used to fit conformal tau, IQA thresholds, severity thresholds
- 20% test (600 samples) - used to evaluate metrics; not seen during calibration
- 20% rolling holdout (600 samples) - used for ongoing post-deployment monitoring (Section 25.6 conformal coverage tracking complement, see 29.7)

**Dataset versioning.** Each F.0 run produces a versioned dataset snapshot (e.g., `F0-2026Q2`, `F0-2026Q3`) tracked via DVC or git LFS. The versioned snapshot is referenced in the calibration files produced by F.0 so that calibration can be reproduced or re-validated against the same data. When F.0 is re-run quarterly (Section 29.6), a new dataset version is created with newly added samples; old versions are retained for reproducibility.

Dataset hygiene:
- pHash deduplication against training set (Section 7); any matches removed from F.0
- No augmentation copies in F.0 (each image is unique)
- PII stripped from filename and metadata

### 29.3 F.0 validation procedure

The F.0 procedure has these steps:

**Step 1: Run pipeline over calibration set (1800 samples)**
- Each sample runs through full pipeline (IQA, signals, classifier, conformal, tier)
- Outputs are logged to a structured file
- Step takes roughly 30 minutes on the RTX 4060

**Step 2: Fit calibration parameters**
- Conformal tau: solve for tau such that empirical coverage on calibration set is 90% (Section 13)
- IQA thresholds: solve for thresholds that produce desired ACCEPTABLE/HIGH/DEGRADED/REJECT distribution on calibration set
- Severity thresholds: align coverage_pct distribution with agronomist-labeled severity grades on calibration set
- Temperature scaling: fit T such that calibration error on calibration set is minimized (Section 12.10)

**Step 3: Evaluate on test set (600 samples)**
- Run pipeline over test set
- Compute metrics (Section 29.4)
- Verify metrics meet quality bars

**Step 4: Run Section 15 scenario tests**
- The 135 scenarios from Section 15 are encoded as integration tests (Section 26.2)
- All 135 must pass; any failure blocks F.0 sign-off

**Step 5: Produce F.0 report**
- Markdown report with metrics, plots, failure analysis
- Distributed to agronomic team and engineering for review

**Step 6: Pilot go/no-go decision**
- Quality bars (Section 29.4) must be met
- Agronomic team signs off on prediction quality
- Engineering signs off on operational readiness (Section 28)
- Operations signs off on deployment plan
- If all sign-offs received: pilot proceeds to Stage 1 (Section 28.6)

### 29.4 Quality bars

These metrics must be met on the F.0 test set for pilot go-decision:

| Metric | Target | Hard floor |
|---|---|---|
| Overall accuracy | > 80% | > 70% |
| Per-class F1 (foliar, septoria, healthy) | > 0.80 | > 0.70 |
| Per-class F1 (late_blight) | > 0.75 | > 0.65 |
| Per-class F1 (ylcv, mosaic) | > 0.65 | > 0.55 (these are underpowered classes) |
| Conformal empirical coverage | 88-92% | 85-95% |
| Tier 4B rate | < 1% | < 3% |
| Tier 5 alert precision (verified by agronomist) | > 70% | > 50% |
| Tier 5 alert recall (verified by agronomist) | > 90% | > 80% |
| Calibration ECE (expected calibration error) | < 5% | < 10% |
| All 135 Section 15 scenarios pass | 100% | 100% (no exception) |

T5 alert precision/recall is the safety-critical metric. The recall hard floor of 80% is set with awareness of the cost of false negatives: missing 20% of dangerous-disease cases means roughly 1 in 5 dangerous cases goes uncaught at the model layer. The downstream agronomist queue (Section 23) reviews a fraction of these (those that route via Tier 3 or 4 paths), reducing the practical miss rate further. Below 80% recall we judge the model not safe enough to ship. Above 90% (the target) we accept the precision tradeoff: we are willing to over-alert (lower precision) but never silently miss real dangerous diseases. The agronomic team has authority to revise these thresholds based on regional disease prevalence and risk tolerance.

Hard floors are absolute minimums; metrics below the hard floor block pilot go.

**If F.0 quality bars are NOT met.** The pipeline goes back to development. Specific failure patterns drive specific fixes:
- Underpowered class F1 below floor: re-train with class-balanced loss or augmentation; gather more samples for that class
- Calibration ECE above floor: re-fit temperature scaling on more diverse calibration set
- Conformal coverage outside band: re-fit tau; if coverage is systematically too low, the underlying classifier needs better calibration first
- Tier 4B rate above floor: investigate signal failure modes; may need pipeline reliability work
- Section 15 scenario failures: fix the regression in the rule chain (these are blocking)

After fixes, F.0 is re-run end-to-end. There is no shortcut for a partial re-run when changes affect calibration; the full F.0 procedure is run again.

### 29.5 F.0 report contents

The F.0 report produced by Step 5 contains:

1. Executive summary (1 page)
2. Metrics table comparing achieved vs target vs hard floor
3. Confusion matrix per class
4. Reliability diagrams (calibration plots) per class
5. Conformal coverage plot (binned by max_prob)
6. Per-tier distribution histogram
7. T5 alert analysis: precision, recall, FP examples, FN examples
8. Section 15 scenario pass/fail summary
9. Failure analysis: 20 worst-case failures (lowest agronomist agreement) with images and explanations
10. Recommendations: pilot go/no-go, with rationale

The report is committed to the operations repo at `validation/F0_report_YYYY-MM-DD.md` and presented to NanoFarm leadership.

### 29.6 Re-running F.0

F.0 is re-run when:

- Major model update (new v3, new LoRA, new classifier weights)
- Threshold change (e.g., chilli leak threshold, conformal target coverage)
- Section 15 scenarios change
- Quarterly cadence regardless of changes (catches drift)

Re-running F.0 takes ~1 day end-to-end. The output is a new calibration file set; deploying it is operations-managed (Section 28.7 conformal coverage runbook).

### 29.7 Limitations

1. **F.0 dataset is a snapshot.** Real production data drifts; F.0-derived calibration becomes stale over time. Section 25.6 monitors drift; periodic re-calibration mitigates.
2. **Two complementary monitoring sources after deployment.** Section 25.6 tracks conformal coverage using agronomist queue resolutions (real production data, but biased toward routed cases per Section 25.9 #3). The F.0 rolling holdout (29.2) provides a separate, unbiased reference dataset that can be re-scored against the deployed model on a periodic schedule (e.g., monthly). Disagreement between the two signals is informative: queue-based metric drift could be due to genuine drift OR to changes in routing patterns; F.0-holdout drift indicates model degradation specifically. Operations should run both checks and reconcile differences.
3. **Agronomist labels have noise.** Even high-confidence labels disagree across agronomists at ~5-10% rate. F.0 uses single-agronomist labels; multi-agronomist consensus would be cleaner but more expensive.
4. **OOD samples are limited.** ~200 OOD samples is enough to test the OOD class; not enough to characterize all possible OOD inputs. v2 may add adversarial OOD examples.
5. **No counterfactual testing.** F.0 doesn't test "what if the user used a different camera"; that would require multi-source paired data.
6. **No explicit cultivar coverage.** F.0 dataset has cultivar diversity but doesn't enforce per-cultivar minimum samples; per-cultivar quality is unmeasured.
7. **No seasonal coverage.** F.0 dataset is from one season; quality across seasons is unmeasured. Pilot will surface seasonal effects.
8. **No multi-image validation.** F.0 evaluates single-image predictions; multi-image aggregation (Section 18) is not formally validated. v2 may add multi-image F.0.

These limitations mean F.0 is necessary but not sufficient. The pilot phases (Section 28.6) are the real validation; F.0 is the gate.

---

## Section 30. Consolidated limitations

### 30.1 Purpose

Section 30 consolidates limitations referenced from earlier sections into a single canonical list. Throughout the spec, individual sections include their own "Limitations" subsection; Section 30 collects them, organizes by category, and assigns ownership for resolution.

The goal of Section 30 is twofold. First, it gives reviewers a single place to assess what the v1 system does NOT do and where the known weak spots are. Second, it serves as the v2 backlog: each item here is a candidate for a v2 epic, with the section reference and severity indicating priority.

This section does not introduce new limitations; everything here is referenced from the source section. Where Section 30 adds analysis, it is clearly marked.

### 30.2 Domain and modeling limitations

**Lab-to-field domain gap (Section 1, Section 8, Section 9).** This is the central technical challenge of the project. Models trained on clean lab-background images systematically fail on real field photos because field photos have different backgrounds, lighting, occlusion, and image quality. The problem motivated the full specialist pipeline architecture (Sections 8-12) replacing an earlier monolithic approach. The mitigations include field-photo upweighting in training (5x weight per userMemories), source-aware stratification, the LAB-CLAHE preprocessing at inference, and the agronomist queue as a human-in-the-loop fallback. None of these eliminate the gap; they manage it. Pilot will surface the remaining gap, especially for cases where the training data is least representative of Kerala field conditions.

**Three classes with severe backbone-level domain shift (userMemories, Section 8.5).** brassica_black_rot, chilli_leaf_curl, okra_cercospora produce only 2-20% confidence on the correct class for real field photos. This is a backbone failure, not a calibration issue. Solutions like kNN re-ranking or VLM verification do not help because the backbone routes to wrong feature regions. Mitigation: COLD_START_ACTIVE policy keeps these classes from being predicted with high confidence; agronomist queue catches missed cases.

**tomato_target_spot has zero verified field photos (userMemories, Section 7).** 432 lab images; no verified field source exists. Mitigation: lower inference threshold ~0.35; UI note recommending agronomist verification for any target_spot prediction.

**Class set is closed-world (Section 12, Section 23.9).** The 7 tomato classes are foliar, septoria, late_blight, ylcv, mosaic, healthy, OOD. Real diseases outside this set (bacterial wilt, fusarium wilt, powdery mildew, etc.) cannot be predicted. The OOD class catches "obviously not one of the trained classes" but a misclassified bacterial wilt may be predicted as foliar or septoria. v2 may add more classes after sufficient training data.

**No multi-disease detection (Section 15.16).** Real plants often have co-occurring diseases. The system represents this only via Tier 3A/3B prediction sets ("uncertain between A and B") rather than "both A and B present". Multi-label classification is a v2 feature.

**Severity grading thresholds are illustrative (Section 17.3, 17.8).** Per-disease severity thresholds (mild/moderate/severe coverage_pct cutoffs) are agronomic best-guesses, not learned from data. Phase F.0 will calibrate; until then, severity is approximate. Agronomic-team review required before clinical use.

**Severity does not integrate stage of disease (Section 17.8).** Early-stage and late-stage disease may both register as the same severity grade despite different urgency.

**No view-label-aware multi-image weighting (Section 18.10).** All images weighted equally except by per-image confidence. A close-up of a lesion is treated the same as a whole-plant view for severity grading.

**Multi-image conformal coverage guarantee not preserved (Section 18.10 #7).** The aggregated prediction set is a heuristic; the 90% coverage guarantee from Section 13 applies per-image only.

### 30.3 Data limitations

**Training data sources have unverified geographic diversity (userMemories, Section 7).** SciDB tomato field photos may have less geographic diversity than count suggests due to Roboflow augmentation artifacts.

**brassica_black_rot single-source training homogeneity (userMemories).** Limited training source diversity causes confusion with alternaria; the system's vein-detection post-processing and confusion correction matrix are partial mitigations.

**Chilli Final Dataset 92% duplicates (userMemories).** 20K raw → only 1,526 unique after pHash deduplication. Affects chilli specialist training (which is out of v1 scope but blocks future expansion).

**No Kerala tier-3 image collection yet (userMemories).** `tier3_labels.csv` is empty. Blocks Phase 5 model expansion and prevents removing COLD_START_ACTIVE policy for the three problem classes.

**OOD samples are limited (Section 29.7).** 200 OOD samples is enough to test the OOD class; not enough to characterize all possible OOD inputs (foreign objects, atypical lighting, partial occlusion, etc.).

**No counterfactual testing (Section 29.7).** F.0 doesn't test "what if the user used a different camera"; would require multi-source paired data.

**No per-cultivar coverage enforcement (Section 29.7).** Per-cultivar quality is unmeasured.

**No seasonal coverage (Section 29.7).** F.0 dataset is from one season; quality across seasons is unmeasured.

**No Kerala tier-3 validation dataset collected (userMemories).** A dedicated Kerala field-photo collection (`tier3_labels.csv`) is empty per project records; Phase F.0 includes some Kerala extension officer photos, but a structured tier-3 set with cultivar/region/lighting diversity does not exist. This blocks final removal of the COLD_START_ACTIVE policy for the three problem classes (brassica_black_rot, chilli_leaf_curl, okra_cercospora) and limits cross-source quality measurement. Pilot Stage 1 / 2 is the implicit substitute, not a formal replacement.

### 30.4 System architecture limitations

**Single-process sandbox (Section 20.2).** Cannot horizontally scale; bottleneck at single-instance throughput. v2 may add Triton or similar model server.

**Single-host pilot deployment is a SPOF (Section 24.9, 28.9).** Hardware failure means downtime; daily backups mitigate but recovery time is hours.

**No automated rollback (Section 28.9).** Rollbacks are manual via git revert and redeploy. The flag-based fallback (UNIFIED_TOMATO_ROUTE_ENABLED) is the closest thing to automation.

**No blue-green deployment (Section 28.9).** v1 deploys cause ~30s downtime.

**No DR plan (Section 28.9).** Multi-region replication is v2.

**Single instance unified server (Section 22.9).** Bottleneck for total system QPS.

**No request prioritization at routing layer (Section 22.9).** All requests FIFO; T5-priority handling happens post-tier in the agronomist queue.

**No circuit breakers in unified server (Section 22.9).** Risk of connection pool exhaustion during sustained downstream outage.

**Cache invalidation is manual (Section 22.9).** Model version changes require unified server restart.

### 30.5 Operational limitations

**Operations team is small (Section 28.9).** 2 engineers; on-call burden is high. Sustainable for pilot scale only.

**Manual cleanup of stale agronomist claims (Section 23.9 #6).** Daily cleanup job resets cases stuck in `in_review` for > 24 hours. Real-time presence detection is v2.

**No multi-agronomist consensus for critical cases (Section 23.9 #3).** Single agronomist's disposition is final.

**No agronomist load balancing (Section 23.9 #2).** Agronomists self-select cases; no automatic assignment by specialty.

**Manual TLS renewal (Section 28.9 #4).** certbot cron with no automation around renewal failures.

**Manual queue management (Section 23.9 #1).** Priority is set at routing time; no dynamic priority based on pending queue size.

**No A/B testing infrastructure (Section 25.9 #4).** Comparing two model versions requires manual log analysis.

**Chilli crop is not implemented in v1 (Section 22.2).** The unified server returns 501 NOT_IMPLEMENTED for chilli requests. Chilli specialist training is a future step per userMemories.

**PSV inference time approximately 500ms vs 200ms aspirational target (userMemories).** PSV speed optimization is deferred to v2.0; the 500ms is included in the Section 21.7 latency budget but it pushes against the single-image total budget.

### 30.6 Privacy and compliance limitations

**No GDPR-style audit log (Section 24.9 #3).** Right-to-deletion is supported (Section 24.6) but the deletion request itself is not logged in a tamper-proof way.

**Image bytes not encrypted in SQLite itself (Section 24.9 #5).** Encryption is at the OS / disk volume level. v2 may add SQLCipher or migrate to Postgres with TDE.

**Image files stored on local disk (Section 24.9 #4).** If disk fills up, sandbox stops accepting new requests. Operations alerts on disk usage > 80%.

**PII handling is informal (Section 24.6).** PII export stripping is rule-based (strip user_metadata except plant variety); a formal privacy review of the field list and export rules has not been done. v2 may add a formal data classification with per-field privacy tags.

**No formal threat modeling (Section 26.8).** Security baseline is informal; production deployment should commission formal threat modeling.

**No fuzz testing (Section 26.9).** Pathological inputs may cause crashes that aren't caught by unit tests.

**No mutation testing (Section 26.9).** Test quality measured by line coverage only.

**Frontend web security baseline only (Section 26.8).** Standard XSS prevention via content security policy and input sanitization is in place but not formally audited. Web security scan (e.g., OWASP ZAP) before Stage 3 is recommended but not committed.

**Production HTTPS configuration not finalized for v1 (userMemories).** v1 pilot may run on internal-network HTTP behind firewall. HTTPS with valid certs and TLS 1.3 is required before Stage 3 (public-facing release); operations team manages certificate provisioning.

### 30.7 Frontend and UX limitations

**v1 frontend is English-only (Section 19.9).** Localization to Malayalam, Tamil, Hindi, Kannada is v2.

**Accessibility partially addressed (Section 19.9).** High-contrast badges and ARIA labels in place; full WCAG 2.1 AA compliance is v2.

**No A/B testing of UI variants (Section 19.9).** Badges, colors, text are fixed.

**No user-side cache (Section 19.9).** Each photo is sent to server fresh; server cache handles repeat-image requests.

**Per-class GradCAM++ for multi-class sets is v2 (Section 16.5, Section 19.5).** v1 generates GradCAM++ for argmax class only; user sees evidence for the most-likely class only when set has multiple candidates.

**No sequential photo flow (Section 19.8).** "Before treatment" / "after treatment" comparison is v2.

**No offline mode (Section 19.8).** v1 requires network connectivity for every prediction.

### 30.8 Monitoring and feedback limitations

**Conformal coverage measurement has sampling bias (Section 25.9 #3).** Computed on agronomist-queue cases only (T5-firing or flagged-ambiguous); biased away from confident Tier 1 cases.

**No real-time anomaly detection (Section 25.9 #1).** Threshold-based alerts only; no changepoint analysis.

**Quality metrics depend on agronomist labels (Section 25.9 #2).** Without enough labels, per-class precision/recall is noisy.

**Tracing is optional (Section 25.9 #6).** OTLP endpoint is env-var-set; if unset, traces are dropped.

**No cost tracking (Section 25.9 #7).** GPU compute cost per request is not measured.

**Severity grading reliability is unmeasured before pilot (Section 17.8).** No ground-truth severity labels in training data; F.0 calibrates against limited expert labels.

**No structured taxonomy for "correct to different class" resolutions outside the 7 trained classes (Section 23.9 #5).** Agronomist must use free-text comment, which doesn't feed back into structured retraining.

### 30.9 Limitation severity classification

To support v2 prioritization, the limitations are classified by severity:

**Safety-critical (directly affect prediction safety; must be addressed for safe broad release):**
- T5 alert recall floor of 80% means ~20% of dangerous cases may be missed at model layer (Section 29.4)
- Three classes with backbone-level domain shift (30.2) - currently mitigated by COLD_START_ACTIVE policy
- tomato_target_spot zero field photos (30.2) - mitigated by lower threshold and UI note

**Pre-production blockers (need addressing before scale-up regardless of safety, but not directly safety-critical):**
- Conformal coverage measurement bias (30.8) - quality measurement is unreliable until addressed
- Single-host SPOF (30.4) - operational risk at production scale
- No DR plan (30.4) - operational risk at production scale
- No multi-agronomist consensus for critical cases (30.5) - process limitation that becomes a bottleneck at scale
- Severity thresholds illustrative (30.2) - need empirical calibration before clinical reliance
- Production HTTPS not finalized (30.6) - required before public release
- Class set is closed-world (30.2) - structural limitation that affects coverage across real diseases

**Medium (improvements for v2):**
- No multi-disease detection (30.2)
- View-label-aware multi-image weighting (30.2)
- Per-class GradCAM++ for multi-class sets (30.7)
- A/B testing infrastructure (30.5)
- Localization (30.7)
- PSV speed gap (30.5)
- Chilli crop not implemented (30.5)

**Low (nice-to-have):**
- Sequential photo flow (30.7)
- Offline mode (30.7)
- Cost tracking (30.8)
- Mutation testing (30.6)

The classification reflects spec-author judgment as of writing; pilot feedback may revise.

### 30.10 Limitations not addressed in this spec

Section 30 catalogs limitations the spec acknowledges. There are likely limitations the spec does NOT acknowledge because they have not been discovered yet:

- Pilot will surface user-experience issues not anticipated in Section 19's frontend rules
- Real disease prevalence in Kerala may differ from F.0 dataset distribution
- Cultivars common in Kerala may differ from training data (no per-cultivar metric)
- Network reliability in rural areas may force latency/timeout reconsideration
- Agronomic team's actual review pattern may differ from Section 23.7 SLA assumptions

These unknown unknowns are why pilot stages exist (Section 28.6); pilot is the discovery process for limitations not in this section. Section 31 (open questions and risks) addresses some of these.

## Section 31. Open questions and risks

### 31.1 Purpose

Section 31 documents open questions about the system and risks that have been identified but not fully mitigated. These are the things the spec author cannot resolve without more information from pilot, agronomic team, operations, or further engineering work.

Each item is structured: question or risk statement, current best understanding, what would change with new information, and who can resolve.

### 31.2 Open questions about model quality

**Q31.2.1: Will the field-photo F1 hold up at pilot scale?**

Current understanding: Phase 2 development on training data showed promising results (Section 8 reports model 2 training success); Phase F.0 validation (Section 29) gives quantitative quality bars. But "training set + held-out test set" is not the same as "real Kerala field photos at pilot scale".

What would change: pilot Stage 0 (shadow mode, Section 28.6) will produce side-by-side comparisons against the legacy 23-class model on real submissions. After 4 weeks of shadow data, we have an empirical answer.

Resolution owner: pilot Stage 0 metrics + agronomic team review of disagreements.

**Q31.2.2: Are the per-class severity thresholds in Section 17.3 right?**

Current understanding: The thresholds are agronomic best-guesses. F.0 will calibrate against limited expert labels. Real agronomic practice in Kerala may set different thresholds for the same disease.

What would change: agronomic team review of severity grading on pilot Stage 1 / Stage 2 outputs; threshold revision via env var changes if needed.

Resolution owner: agronomic team after pilot Stage 1.

**Q31.2.3: Is the IQA gate too strict / too lenient?**

Current understanding: IQA REJECT is calibrated on training data. Real users may submit images that IQA judges differently than agronomists do.

What would change: pilot will produce data on REJECT rates and user complaints about being rejected for valid photos.

Resolution owner: agronomic team + UX feedback after pilot Stage 2.

**Q31.2.4: Does conformal calibration drift fast enough to warrant monthly re-fitting?**

Current understanding: Section 13 specifies recalibration triggers; Section 25.6 monitors drift. Actual drift rate is unknown until pilot data accumulates.

What would change: 6-12 weeks of pilot data lets us measure drift rate empirically.

Resolution owner: data science / engineering after pilot Stage 2.

### 31.3 Open questions about operations

**Q31.3.1: Is the agronomist queue capacity sufficient for projected pilot volumes?**

Current understanding: 1 senior + 2 junior agronomists at NanoFarm. Section 23.7 SLA targets are aspirational. Section 28.6 Stage 1 expects < 50 pending cases; Stage 2 expects SLA met.

What would change: Stage 1 pilot will reveal whether agronomist throughput matches volume.

Risk: if capacity is insufficient, the agronomist queue becomes a bottleneck and SLA degrades. Mitigation: route_ambiguous_to_queue flag can be disabled to focus queue on T5 only.

Resolution owner: operations team + agronomic team after pilot Stage 1.

**Q31.3.2: Will the single-host RTX 4060 hold up under pilot Stage 2 load?**

Current understanding: Section 21.7 latency budgets are based on idle conditions; Section 22.9 acknowledges concurrent-request degradation. Section 28.2 notes VRAM contention with APIN on the same host.

What would change: load testing during pilot Stage 1 (low volume) will reveal headroom; Stage 2 (50-100 users) tests real concurrent load.

Risk: if latency degrades or OOM occurs, pilot may need a hardware upgrade or service split before Stage 3.

Resolution owner: operations team after pilot Stage 1 load testing.

**Q31.3.3: How do we handle a major Kerala-wide late_blight outbreak with the system?**

Current understanding: The system would correctly fire many T5 alerts, overwhelming the agronomist queue. Section 23.7 has thresholds at 200 / 500 pending cases.

What would change: pilot Stage 2 may surface this scenario if it coincides with weather-driven outbreaks.

Risk: in a real outbreak, the system's value is highest exactly when the queue is most overwhelmed. We need an outbreak protocol: auto-batch similar cases for bulk review, prioritize by region clusters.

Resolution owner: agronomic team + operations team; outbreak protocol is a Stage 2 deliverable.

### 31.4 Open questions about user experience

**Q31.4.1: Will extension officers prefer the new tier-based response over a single answer?**

Current understanding: Section 19's frontend rules show tier badges and structured reasons. Extension officers used to a single-answer system may find it confusing.

What would change: pilot Stage 2 user feedback.

Risk: low adoption due to UX complexity. Mitigation: simplification of Tier 3 displays, training materials, NanoFarm UX team iteration.

Resolution owner: UX team after pilot Stage 2.

**Q31.4.2: Is the multi-image flow worth the latency cost to users?**

Current understanding: Section 18 multi-image adds 1.5-2 seconds compared to single-image (Section 18.10 limitation 5 notes the latency at N=5 is borderline). Users in poor-network areas may have a hard time uploading 5 images.

What would change: pilot Stage 2 will measure multi-image usage rate and abandonment.

Risk: low usage means the engineering cost was not worth it. Mitigation: scope multi-image down to N=2 or N=3 if usage is low.

Resolution owner: product team after pilot Stage 2.

**Q31.4.3: Should the system explain its reasoning to users beyond GradCAM++?**

Current understanding: Section 16.4 returns structured reasons for analytics; Section 19.5 shows GradCAM++ to users. There is no natural-language explanation.

What would change: user feedback may indicate desire for more explanation; v2 could add LLM-generated explanations in user's language.

Risk: users distrust black-box predictions; lower adoption.

Resolution owner: product team.

### 31.5 Risks

Likelihood ratings (low/medium/high) are qualitative judgments by the spec author based on system design and known weak spots; they are not derived from data. Pilot data may revise them. Impact ratings reflect consequences if the risk materializes.

**R31.5.1: Wrong prediction on a high-stakes case leads to crop loss.**

Likelihood: low (system design has multiple safety nets: T5 alert, agronomist queue, severity-based action recommendations).

Impact: medium-high (a single farmer losing a crop is significant for them; no mass-loss scenario identified).

Mitigation: T5 alert recall floor of 80% (Section 29.4); agronomist queue review; rollback flag; severity recommendations include "consult agronomist" caveats; pilot stages catch issues before broad release.

Residual risk: not zero. Honest assessment: a wrong prediction at high confidence on an unusual disease could cause real-world harm. Multi-image flow and extension officer training are mitigations but not eliminations.

**R31.5.2: Regulatory or compliance issue with agricultural decision support.**

Likelihood: low for v1 pilot (small scale, behind firewall, internal users); medium for production rollout (extension officers and farmers, agricultural advice).

Impact: high (could block deployment or trigger legal review).

Mitigation: pilot deployment is internal / extension-officer only; not a prescription system; agronomist queue means a human is in the loop for high-stakes cases.

The specific regulatory framework that applies (FSSAI, Indian state-level agricultural advisory, IT Act provisions on automated decision systems, etc.) is out of spec scope and should be clarified by NanoFarm legal team before Stage 3.

Resolution: NanoFarm legal team review before Stage 3.

**R31.5.3: Sandbox fails in production after passing F.0.**

Likelihood: medium (any system has bugs not caught by F.0).

Impact: medium (Tier 4B rate spike is the most likely failure mode; flag-based rollback restores legacy model in <1 minute).

Mitigation: Section 28.7 runbooks; Section 28.6 stage exit criteria.

Residual risk: an unexpected systematic regression that escapes detection during shadow mode and Stage 1 could affect Stage 2 users for several days before being caught.

**R31.5.4: Adversarial / malicious image submission.**

Likelihood: low (pilot is behind firewall; users authenticated).

Impact: low-medium (could waste agronomist queue time; could potentially exploit OOM via crafted image bytes).

Mitigation: image size limits (Section 16.9); decode failure handling; rate limits (Section 26.8).

Residual risk: dedicated adversary has not been considered in the v1 threat model. Section 26.8 calls out the lack of formal threat modeling.

**R31.5.5: Loss of data integrity in SQLite.**

Likelihood: low (SQLite is robust; daily backups mitigate).

Impact: medium (loss of recent training data and queue cases; pilot can re-run validation).

Mitigation: Section 24.7 backups; Section 24.9 acknowledges this.

Residual risk: a corruption event between backups loses up to 24 hours of data. Acceptable for pilot; v2 with replication addresses.

**R31.5.6: Agronomic team turnover during pilot.**

Likelihood: medium (small team, long pilot).

Impact: medium (loss of institutional knowledge about case-review patterns and threshold judgment).

Mitigation: Section 23.6 disposition records preserve agronomist judgments; Section 26.3 spec-author successor pattern.

Residual risk: a new agronomist may set different thresholds, complicating longitudinal quality tracking.

**R31.5.7: Pilot reveals fundamental model quality issues.**

Likelihood: medium (the lab-to-field gap is the central technical risk).

Impact: high (would block Stage 2 / Stage 3; require model rework).

Mitigation: Stage 0 shadow mode catches gross issues before user impact; F.0 quality bars catch quantitative issues.

Residual risk: a quality issue that passes F.0 but emerges at Kerala field-photo scale. Would require Stage 1 abort, model rework, F.0 re-run. Schedule impact: weeks to months.

### 31.6 Resolution timeline

| Question / risk | Expected resolution stage | Owner |
|---|---|---|
| Q31.2.1 field-photo F1 | Stage 0 (4 weeks) | Pilot lead + agronomic team |
| Q31.2.2 severity thresholds | Stage 1 (4 weeks) | Agronomic team |
| Q31.2.3 IQA strictness | Stage 2 (8-12 weeks) | UX + agronomic team |
| Q31.2.4 calibration drift rate | Stage 2 (6-12 weeks) | Engineering |
| Q31.3.1 queue capacity | Stage 1 (4 weeks) | Operations + agronomic team |
| Q31.3.2 RTX 4060 load | Stage 1 (4 weeks) | Operations |
| Q31.3.3 outbreak protocol | Stage 2 (during outbreak if any) | Agronomic team + operations |
| Q31.4.1 user adoption | Stage 2 (8-12 weeks) | UX team |
| Q31.4.2 multi-image utility | Stage 2 (8-12 weeks) | Product team |
| Q31.4.3 explanation features | Post-Stage 2 | Product team |
| R31.5.1 wrong-prediction harm | Ongoing monitoring | All |
| R31.5.2 regulatory | Pre-Stage 3 | Legal team |
| R31.5.3 production failure | Stages 0-2 | Engineering |
| R31.5.4 adversarial submissions | v2 threat model | Security team |
| R31.5.5 data integrity | v2 replication | Operations |
| R31.5.6 team turnover | Ongoing | Operations |
| R31.5.7 fundamental quality | Stages 0-2 | All |

This timeline is aspirational. Actual pace depends on pilot scheduling, team availability, and what surfaces during pilot.

## Section 32. Honest assessment - what would prevent v1 launch

### 32.1 Purpose

Section 32 is the spec's honest answer to "should this system launch?" It enumerates conditions under which v1 should NOT launch, even after all earlier sections' criteria are met. The goal is to push back against deployment momentum: a system that meets quantitative criteria can still be wrong to ship if qualitative concerns are unresolved.

This section is written by the spec author with awareness that I (the AI assistant drafting this) am not the human author who will own pilot decisions. The recommendations here are framed as questions for the human decision-makers (Dev, NanoFarm engineering, agronomic team, operations) rather than autonomous judgments.

### 32.2 Hard launch blockers

These conditions block v1 launch regardless of other metrics:

**HB-1: Phase F.0 quality bar failure.** If any "hard floor" metric in Section 29.4 is not met, v1 does not launch. The hard floors are absolute. Failing them indicates the model is not safe enough or accurate enough for deployment, regardless of how close to the targets the other metrics are.

**HB-2: Section 15 scenario regression.** If any of the 135 scenarios in Section 15 fails, v1 does not launch. The scenarios encode safety-critical behavior of the rule chain; a failure means the implementation deviates from spec in a way that affects tier outcomes.

**HB-3: Safety incident in Stage 1.** Per Section 28.6, any safety incident in Stage 1 triggers immediate rollback to legacy model and Stage 1 restart. Repeated safety incidents (defined: more than 1 in any 4-week Stage 1 attempt) escalate to a hold on launch pending root-cause analysis. The 1-incident threshold for restart-without-hold reflects that a single incident can be a single-bug fix; 2 or more incidents in a 4-week window suggest a systematic problem requiring deeper rework rather than another fix-and-retry. The threshold is not arbitrary but it is judgment-based; the agronomic team can override toward stricter (zero tolerance) for certain disease classes.

**HB-4: Agronomic team withholds sign-off.** Per Section 29.3 step 6, agronomic team sign-off is required for Stage 1 to proceed. The agronomic team has full authority to withhold sign-off for any reason they document. Engineering may address the documented concerns and re-request sign-off; the relationship is not adversarial. The point is that engineering does not unilaterally override agronomic judgment.

**HB-5: Legal or compliance hold.** Per Section 31.5 R31.5.2, NanoFarm legal team reviews before Stage 3. If the legal review surfaces a compliance issue, launch is held until resolved. v1 pilot Stages 0-2 can proceed before legal review (these are internal / extension-officer only); Stage 3 (limited public release) requires legal sign-off.

**HB-6: Operations team withholds readiness sign-off.** Per Section 28.5 bringup procedure, operations team has authority over deployment readiness. If operations judges the runbooks insufficient, monitoring inadequate, or capacity insufficient for projected load, launch is held.

These six hard blockers are non-negotiable. The remaining sub-sections discuss soft blockers and judgment calls.

### 32.3 Soft launch blockers

These conditions are not absolute but are warnings that should slow or pause launch:

**SB-1: Hard-floor metrics close to threshold.** If F.0 metrics are within 5 percentage points of any hard floor (e.g., T5 recall at 81% when floor is 80%), the spec recommends collecting more F.0 data before pilot. Operating near the floor leaves no margin for production drift.

**SB-2: F.0 dataset has known gaps.** If F.0 dataset is missing a class or cultivar that pilot users will encounter, the corresponding metrics are unverified. Recommendation: add samples to F.0 before pilot, OR acknowledge the gap explicitly to pilot users and agronomists.

**SB-3: Agronomist queue capacity uncertain.** If pilot Stage 1 has not actually verified queue throughput at projected Stage 2 volumes, Stage 2 may discover capacity gaps at scale. Recommendation: load-test the queue (synthetic case generation) before Stage 2 to validate capacity assumptions.

**SB-4: Conformal coverage measurement biased.** Per Section 25.9 #3 and 30.8, the coverage metric is conditional on routed cases. Without an unbiased measurement (e.g., random-sample spot-check of non-routed Tier 1 cases), the system's true coverage is unknown. Recommendation: implement random-sample spot-check before Stage 3.

**SB-5: Three classes with backbone-level domain shift remain uncertain.** Per userMemories, brassica_black_rot, chilli_leaf_curl, okra_cercospora have severe domain shift. While these are not tomato classes (so not directly in v1 scope), the same underlying issue may affect tomato classes that haven't been thoroughly tested on Kerala field photos. Recommendation: explicitly test these tomato classes on Kerala-collected field photos during Stage 0.

**SB-6: Single-host SPOF.** v1 deployment is single-host. Hardware failure during Stage 2 or 3 means hours of downtime. Recommendation: have a documented manual failover procedure and tested-restore-from-backup; or accept the risk explicitly.

**SB-7: Severity grading thresholds not validated.** Section 17.3 thresholds are illustrative. If pilot Stage 1 reveals high agronomist correction rate on severity, the thresholds need recalibration before Stage 2.

These soft blockers indicate areas where launch can proceed with awareness, but each item adds risk that should be quantified and accepted.

### 32.4 Strategic considerations

Beyond technical blockers, several strategic considerations affect the launch decision:

**SC-1: Reversibility is a primary value.** The flag-based fallback (UNIFIED_TOMATO_ROUTE_ENABLED=false) makes v1 highly reversible. This shifts the risk calculus: launching with known soft blockers is more acceptable when rollback is fast and complete. The spec's commitment to flag-based reversibility (Sections 22, 28) should not be eroded.

**SC-2: The legacy 23-class model is the comparator.** v1 should be better than the legacy model on the cases where v1 is supposed to be better (field photos with the lab-to-field gap). It does NOT need to be uniformly better. If shadow mode (Stage 0) shows v1 is worse on some narrow case (e.g., lab-photo-only inputs), that may be acceptable if v1's improvement on field photos compensates.

**SC-3: The agronomist queue is a feature, not a fallback.** The system is designed with the agronomist in the loop. It is NOT a replacement for agronomist expertise; it is a triage and prioritization tool. Launch decisions should preserve the human-in-the-loop architecture; any pressure to remove the queue (for cost or scale) should trigger careful review.

**SC-4: Pilot is the discovery process for unknown unknowns.** Many failure modes will only surface at pilot. The spec is incomplete by design: Section 30.10 acknowledges limitations not yet discovered. The launch decision is not "do we know everything is fine?" but "do we know enough to proceed cautiously and respond to surprises?"

**SC-5: B.Tech project deliverable timing.** Per userMemories, this is Dev's B.Tech project completed during the NanoFarm internship. Academic deliverables have hard deadlines that may not align with deployment readiness. The honest recommendation: separate the academic deliverable (which can be the spec, the F.0 results, and a working sandbox demonstrating the architecture) from the production launch decision (which is NanoFarm operations and agronomic team's call). Conflating the two creates pressure to launch before genuinely ready.

### 32.5 Recommended launch criteria

Putting the hard and soft blockers together, the recommended ENTRY criteria (criteria for starting each pilot stage) are below. ENTRY criteria are distinct from EXIT criteria (the conditions for moving from one stage to the next), which are documented in Section 28.6. Entry criteria are pre-stage gates; exit criteria are post-stage gates.

**Stage 0 entry (shadow mode):**
- F.0 quality bars met (HB-1)
- Section 15 scenarios all pass (HB-2)
- Operations sign-off on readiness (HB-6)
- No legal hold (HB-5 not yet required for shadow mode but should not be active)
- Recommended: F.0 metrics with > 5 pp margin to hard floors (SB-1)
- See Section 28.6 for Stage 0 exit criteria.

**Stage 1 entry (internal user pilot):**
- Stage 0 exit criteria met (per Section 28.6: zero critical bugs in Stage 0; 100+ disagreement cases reviewed favors v1 or shows neutral; agronomic team approves Stage 1 transition)
- Agronomic team sign-off (HB-4)

**Stage 2 entry (extension officer pilot):**
- Stage 1 exit criteria met (per Section 28.6: < 5% Tier 4B rate; queue under 50 pending; no safety incidents)
- Queue capacity validated under Stage 1 load
- Severity grading correction rate during Stage 1 below a heuristic threshold of 20%; the 20% number is a starting heuristic with no empirical basis and pilot data should revise it; agronomic team has authority to set the actual threshold
- Updated runbooks based on Stage 1 lessons

**Stage 3 entry (limited public release):**
- Stage 2 exit criteria met (per Section 28.6: agronomic prevalence match; < 10% Tier 4B; SLA met; positive feedback; no safety incidents)
- Legal sign-off (HB-5)
- Random-sample spot-check coverage measurement implemented (SB-4)
- Outbreak protocol documented (Q31.3.3)
- Per-district rollout plan with 4-week stable-operations gate (per Section 28.6 Stage 3 description)

Each stage's entry criteria are stricter than the previous. This is intentional: each stage exposes more users to the system, raising the bar for safety and operational maturity.

### 32.6 What the spec author cannot determine

I (the AI assistant drafting this spec) cannot determine some launch-relevant facts:

- The actual quality of the model at v1 freeze (depends on training runs not yet complete)
- The actual capacity of the agronomic team
- The actual hardware specs of the pilot host beyond what userMemories states
- The legal and regulatory environment for agricultural decision support in Kerala
- The competitive landscape (whether NanoFarm has incentives to launch fast or slow)
- The financial constraints (cost of pilot delays vs cost of pilot failures)
- The relationships between Dev, the academic guides (Dr. Lekha J, Dr. Asif Tariq), the industry mentor (Mr. Srinivas), and the company founder (Mr. Arun C Michael) that shape decision-making

These are determined by the human decision-makers. The spec provides structure for the launch decision; it does not make the decision.

### 32.7 Honest summary

The v1 sandbox specification represents substantial engineering thought. It addresses the central technical challenge (lab-to-field domain gap) with a specialist architecture, comprehensive tier rules, and a human-in-the-loop agronomist queue. It includes safety mechanisms (T5 alerts, conformal prediction, NaN guards, degraded-mode handling), validation procedures (135 scenarios, F.0 calibration), and honest acknowledgment of limitations.

It is also genuinely incomplete. Several limitations (backbone-level domain shift on three classes, severity threshold calibration, multi-image conformal coverage, single-host SPOF, agronomist queue capacity at scale, no Kerala tier-3 dataset) are not fully resolved by v1. The pilot stages are designed to surface these and respond.

The honest assessment: v1 MAY be ready for shadow mode (Stage 0) once F.0 quality bars are met, with the explicit understanding that Stages 1-3 are gated by lessons learned at each stage. v1 is NOT ready for production-scale launch without those stages. Any pressure to skip stages or compress timelines should be resisted.

Whether the system is "good enough" to start helping farmers in Kerala depends on judgments the spec author cannot make: the agronomic team's tolerance for the documented residual risks, NanoFarm leadership's risk appetite, and Kerala-specific factors that pilot will surface. The architecture and safeguards are designed to make a controlled rollout possible if the human decision-makers judge the residual risks acceptable. The spec's role is to make those risks visible; the launch decision is theirs.

---
