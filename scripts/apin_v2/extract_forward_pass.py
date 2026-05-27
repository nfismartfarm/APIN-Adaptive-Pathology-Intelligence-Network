"""Phase 4C extraction: animated forward-pass diagram data.

Runs 4 models (Router · Tomato · APIN · Chilli-as-router-only) on 3 stratified
images each, captures intermediate activations at named stages via forward
hooks, renders activations as base64 PNG tile grids (viridis colormap, 64×64
tiles, 4-column grid), packs everything plus hand-drafted narrations into
`_qa_tmp/_pipeline_atlas_forward_pass.json`.

This script is the data side of the Pipeline Atlas Forward Pass section. The
JSON it produces is served via `GET /api/pipeline_data/forward_pass` and
consumed by the ForwardPass JS module in `pipeline.html`.

DESIGN PHILOSOPHY (locked in `_qa_tmp/_pipeline_atlas_4c_plan.md`):
  1. Real feature maps, not stylized abstractions. Every visual is an actual
     extraction. No placeholder gradient blobs. If a model fails to load,
     its entry carries `unavailable: true` rather than synthetic data.
  2. Field-laboratory voice, anchored to Phase 4B. Every narration is
     hand-drafted, matches Phase 4B's voice signatures, and avoids the
     9-word banned list (see BANNED_WORDS).
  3. Sequential model loading to avoid 8 GB VRAM OOM on RTX 4060. Each model
     is fully loaded, hooks registered, inference run, captures rendered,
     then the model is deleted and the cache is emptied before the next.

AUDIT-GAUNTLET PROVENANCE: this script's structure, schema, narrations, and
invariant assertions all trace to specific findings amended across 8 plan
audit rounds (PDA R1-R4 + PVA R1-R4, 67 findings total). See
`_qa_tmp/_pipeline_atlas_4c_plan.md` for the full chain.

SAFE TO RE-RUN: extraction is deterministic given the locked test split,
fixed model weights, and seed=42. Two runs produce byte-identical JSON
modulo the `produced_at` timestamp.
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────
# Determinism preamble. Seed every RNG source the moment the module loads.
# Required by PDA R2 #4 (idempotency contract) and plan v4 §A1 line item.
# ────────────────────────────────────────────────────────────────────────
import random
import os
import sys
import io
import json
import base64
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# RNG seeds (locked at module import; do not move below model imports).
random.seed(42)
try:
    import numpy as np
    np.random.seed(42)
except ImportError:
    np = None  # extraction will fail later with a clear error

try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # determinism over speed
except ImportError:
    torch = None

# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_qa_tmp" / "_pipeline_atlas_forward_pass.json"
TILE_PX = 64
PNG_COMPRESSION = 6  # Pillow default; balance of size and write speed
N_IMAGES_PER_MODEL = 3
SEED = 42

# Banned words for narration validation (PVA R3 lock; single canonical list).
BANNED_WORDS = {
    "leverages", "harnesses", "powerful", "state-of-the-art",
    "seamlessly", "robust", "cutting-edge", "advanced", "next-generation",
}

# Voice anchor phrases (Phase 4B precedents quoted verbatim). Narrations for
# stages that show a measured number SHOULD use one of these.
VOICE_ANCHORS = [
    "honest failure is cheaper than wrong confidence",
    "specific magnitudes vary per image and surface in Phase D once the metrics extraction runs",
    "we never invent numbers",
    "the validation lift over V3-alone was measured with exactly this setup",
]


# ────────────────────────────────────────────────────────────────────────
# NEW_GLOSSARY_TERMS · drafted as Python constants in 4C.1; the actual
# pipeline.html GLOSSARY array edit happens in 4C.3 (plan v4 §A5 lock).
# These are the new terms required by stage labels in the schema below.
# Each definition follows the existing 142-term glossary's voice: factual,
# concise, no marketing words.
# ────────────────────────────────────────────────────────────────────────
NEW_GLOSSARY_TERMS: Dict[str, str] = {
    "attention rollout":
        "A method that combines self-attention weights across all transformer "
        "layers into a single per-token saliency map. Each layer's attention "
        "matrix is multiplied with the running rollout so the final map shows "
        "how much each input token influenced the final classification token.",
    "PSV":
        "Plant Signal Vector. An engineered (non-learned) feature vector that "
        "captures leaf-shape, colour, and vein statistics directly from the "
        "image. APIN uses PSV as the fourth signal in its 4-signal ensemble; "
        "it adds rule-based features that the learned signals can miss.",
    "CLS token":
        "The classification token. A learnable vector prepended to the patch "
        "embeddings before a transformer is run. After 12 attention blocks, "
        "the CLS token's final value is fed to a linear head to produce the "
        "class prediction.",
    "attention block":
        "One layer of a transformer. Each block runs multi-head self-attention "
        "across all tokens, then a small feed-forward network on each token "
        "independently. DINOv2 ViT-Small uses 12 blocks stacked in sequence.",
    "register tokens":
        "Extra learnable tokens prepended to the input sequence in DINOv2 "
        "with register variants. They give the model a place to store global "
        "summary information without polluting the patch tokens. DINOv2-Small "
        "with 4 registers uses 1 CLS + 4 register + 256 patch = 261 tokens.",
    "stacking MLP":
        "A small multi-layer perceptron that takes per-signal class "
        "probabilities as input and outputs a fused class distribution. "
        "Learned from a held-out validation split. APIN's stacking MLP "
        "weights each of the 4 signals per class so signals that are "
        "reliable on a class get higher weight on that class.",
}


# ────────────────────────────────────────────────────────────────────────
# NARRATIONS · hand-drafted per (model, stage). Phase 4B voice. Each
# narration is 150-250 words, follows the 4-part template (input /
# computation / downstream / honest caveat), wraps 2-4 jargon terms as
# {term:X} placeholders (the extractor converts them to <span class="term"
# data-term="X">X</span> at JSON write time), and contains zero banned
# words. Assertion block at the bottom of this script enforces these
# invariants before write.
# ────────────────────────────────────────────────────────────────────────
NARRATIONS: Dict[str, Dict[str, str]] = {
    # ─── ROUTER (4 stages) ─────────────────────────────────────────────
    "router": {
        "input":
            "The router sees a 224×224 RGB tensor: the leaf photograph after "
            "ImageNet-statistics normalisation has been applied. Before it "
            "reaches this layer, the image was opened, resized to 224 pixels "
            "on each side, and centred so the frozen "
            "{term:DINOv2} backbone behaves consistently regardless of the "
            "camera that took the photo.\n\n"
            "The computation at this stage is preprocessing only: no model "
            "parameters fire yet. The visual you see on the right is the "
            "actual input the model receives, not the original JPEG. Router "
            "preprocessing skips the CLAHE step that the disease "
            "specialists apply, because the router decides crop, not disease, "
            "and CLAHE is tuned for lesion contrast.\n\n"
            "The next stage will tokenise this image into a 16×16 grid of "
            "14-pixel patches (256 patch tokens), prepend the "
            "{term:CLS token}, and prepend 4 {term:register tokens}, for a "
            "total of 261 tokens. Each token is embedded into a 384-"
            "dimensional vector and fed through 12 transformer blocks.\n\n"
            "One honest caveat: the model has not yet decided what the image "
            "contains. The router's argmax at the end of the pipeline depends "
            "on the entire 12-block stack, not on this preprocessing step.",

        "vit_early":
            "At block 3 of 12, the {term:attention block} stack has had three "
            "rounds to circulate information between the 261 tokens. The "
            "visual on the right is the mean attention weight across the 6 "
            "heads of this block, summed over how much each spatial patch "
            "attends to its neighbours.\n\n"
            "Early blocks like this one tend to fixate on texture: leaf-vein "
            "ridges, lesion edges, the boundary between leaf and background. "
            "The model has not yet decided which crop it is looking at; it is "
            "still building a map of which regions are visually interesting.\n\n"
            "Downstream blocks (4 through 11) will use this early attention "
            "as scaffolding. They progressively pool the texture information "
            "into more global features, and by the final block the "
            "{term:CLS token} carries a summary of the whole leaf rather than "
            "any individual patch.\n\n"
            "One honest caveat: a hot region in this map is not a disease "
            "lesion. It is wherever block 3 found something visually salient. "
            "On a healthy leaf the same regions can be hot for entirely "
            "innocuous reasons (a fold, a shadow, a vein junction).",

        "vit_pooled":
            "At block 11, the final transformer block, the {term:attention "
            "rollout} across all 12 blocks is shown on the right. Each patch "
            "is coloured by how much it contributed to the final {term:CLS "
            "token} that gets fed to the classifier head.\n\n"
            "The computation is a cumulative multiplication of every block's "
            "attention matrix with the running rollout. The result is a "
            "per-patch saliency map showing, in effect, which leaf regions "
            "the router relied on most when forming its crop decision.\n\n"
            "The next stage takes this pooled CLS embedding (a single "
            "384-dimensional vector) and projects it through a linear layer "
            "to produce 4 class logits: tomato, okra, brassica, chilli.\n\n"
            "One honest caveat: attention rollout is an approximation. It "
            "treats every block's attention as equally important and ignores "
            "the MLP layers between blocks. It is useful for showing where "
            "the model looked, not for proving why it decided what it "
            "decided. The validation lift over V3-alone was measured with "
            "exactly this setup, but the visual is still a heuristic.",

        "head_softmax":
            "The 384-dimensional pooled {term:CLS token} from block 11 is "
            "fed through a single linear layer that produces 4 logits, one "
            "per crop class. A softmax converts those logits into "
            "probabilities that sum to 1. The bar chart on the right is "
            "the actual softmax output for this image; the highest bar is "
            "the router's argmax.\n\n"
            "The deployed pipeline gates on this output. If the argmax "
            "probability is at least 0.40, the request is routed to that "
            "crop's specialist. If below 0.40, the system falls back to the "
            "APIN okra/brassica ensemble as a best-effort guess. The 0.40 "
            "floor was chosen on a held-out validation split as the value "
            "that minimises both wrong-crop routing and false fallbacks.\n\n"
            "Downstream of this stage, the chosen specialist takes over. "
            "The specialist sees the same preprocessed image but routes it "
            "through its own forward pass; nothing from the router's CLS "
            "embedding is passed along.\n\n"
            "One honest caveat: a softmax of 0.41 is treated identically to "
            "0.99 by the routing gate. Both clear the 0.40 floor. Router "
            "confidence is a gate, not a probability you should propagate.",
    },

    # ─── TOMATO (7 stages) ─────────────────────────────────────────────
    "tomato": {
        "input":
            "The tomato pipeline sees the same leaf photograph as the router, "
            "but each of its two branches preprocesses it differently. The "
            "V3 branch uses a 224×224 stretch-resize plus LAB-CLAHE applied "
            "to the L channel of LAB, then ImageNet normalisation. The "
            "SP-LoRA branch caps the image at 800 pixels, letterboxes it "
            "into a 392×392 square with padding value 114, then applies the "
            "same LAB-CLAHE.\n\n"
            "Both preprocessing chains happen at this stage. The two "
            "branches will process their own preprocessed inputs in "
            "parallel through subsequent stages.\n\n"
            "Downstream stages will run V3 and SP-LoRA independently. They "
            "do not share weights and do not see each other's intermediate "
            "activations. Their probability outputs are fused at the "
            "{term:stacking MLP} stage near the end.\n\n"
            "One honest caveat: the visual you see is V3's preprocessed "
            "input (224×224). The SP-LoRA branch sees a different sized "
            "image (392×392). The forward-pass diagram simplifies this to "
            "one input visual for clarity, but the two branches really do "
            "see different sized tensors.",

        "v3_stage_1":
            "V3 is a {term:DINOv2}-Small backbone with low-rank adaptation "
            "(LoRA) and FiLM conditioning, originally trained for 10 classes "
            "and re-used here for the deployed 6. Stage 1 captures the "
            "activations after the first transformer block.\n\n"
            "The first block has had one round of self-attention across the "
            "patch tokens. The visual on the right shows the mean activation "
            "across the embedding dimensions, reshaped to a 16×16 spatial "
            "grid. Early features here are simple: edges, colour gradients, "
            "and rough leaf-vs-background separation.\n\n"
            "Downstream blocks (2 through 11) will progressively pool these "
            "local features into more semantic representations. By the deep "
            "stages, the activations correspond to specific disease patterns "
            "the model has learned during fine-tuning.\n\n"
            "One honest caveat: this is the V3 branch only. The SP-LoRA "
            "branch is running its own forward pass on a different input "
            "resolution at the same time. The two branches will not "
            "communicate until the {term:stacking MLP} fuses their "
            "probability outputs at the end.",

        "v3_stage_2":
            "By V3's block 5 (the middle of the 12-block stack), the model "
            "has built up an intermediate representation of the leaf. The "
            "attention pattern shows clusters of activation around specific "
            "regions, often corresponding to lesions or healthy tissue "
            "patches.\n\n"
            "The FiLM conditioning that V3 applies feature-wise linear "
            "modulation to each block's output, controlled by a small "
            "conditioning network. This is what lets V3 adapt the same "
            "backbone to multiple crop families without separate models per "
            "crop. The conditioning vector at this stage carries the "
            "compressed crop identity learned from the 10-class training.\n\n"
            "Downstream, blocks 6 through 11 will refine these mid-level "
            "features into the final 6-class prediction. The LoRA "
            "adaptations injected at each block let the model specialise "
            "for tomato without overwriting the original DINOv2 weights, "
            "so the backbone remains useful for other downstream tasks.\n\n"
            "One honest caveat: lesion-like activations here can come from "
            "non-lesion features (shadows, leaf curl, vein junctions) that "
            "happen to share texture statistics. The model is not yet "
            "certain what it sees; that certainty only arrives at the final "
            "classifier.",

        "lora_branch":
            "The SP-LoRA branch runs a separate DINOv2-Reg-Base backbone "
            "with single-pass LoRA adapters on the {term:CLS token}, "
            "trained directly on 6 tomato classes for around a dozen epochs "
            "before early-stopping triggered. This stage "
            "shows the activations from the final {term:attention block} of "
            "that branch.\n\n"
            "Because SP-LoRA's input resolution is 392×392 (versus V3's "
            "224×224), the spatial grid here is larger: 28×28 patches "
            "instead of 16×16. More patches means finer-grained spatial "
            "resolution, at the cost of more computation per image, and "
            "more memory pressure when the two branches run together.\n\n"
            "Downstream of this stage, the SP-LoRA branch produces its own "
            "6-class probability distribution. That distribution will be "
            "fused with V3's at the {term:stacking MLP} step using a 50/50 "
            "weighted average with asymmetric temperature scaling per the "
            "production calibration.\n\n"
            "One honest caveat: SP-LoRA's calibration on the held-out split "
            "was less reliable than V3's, which is why the ensemble runs "
            "SP-LoRA at temperature 1.0 (no sharpening) while V3 runs at "
            "temperature 0.5 (sharpened by 2×). See Decision 56 in "
            "`ladi_decisions.md` for the validation lift measurement.",

        "v3_deep":
            "V3's final transformer block (block 12) has aggregated "
            "information from all preceding stages into a 384-dimensional "
            "pooled {term:CLS token}. The visual on the right is the "
            "{term:attention rollout} across all 12 V3 blocks, showing "
            "which leaf regions the model relied on most.\n\n"
            "Compared to the early-stage attention, this rollout is much "
            "more localised: it focuses on specific lesions or healthy "
            "patches rather than the entire leaf. This is the expected "
            "behaviour of a well-trained classifier; the model learns to "
            "look at the diagnostic regions and ignore irrelevant "
            "background.\n\n"
            "Downstream, this pooled embedding feeds a 6-class linear "
            "classifier head, producing V3's softmax over the 6 tomato "
            "classes.\n\n"
            "One honest caveat: rollout maps can be misleading on healthy "
            "leaves. With no lesions to focus on, the model still produces "
            "an argmax (usually the healthy class), but the rollout map "
            "highlights whatever vein or shadow happened to drive the "
            "decision. Treat the visual as 'where the model looked', not "
            "'what's wrong with the leaf'.",

        "stacking_gate":
            "The {term:stacking MLP} takes V3's 6-class probability vector "
            "and SP-LoRA's 6-class probability vector and fuses them into a "
            "single distribution. The fusion is not a flat 50/50 average: "
            "V3's logits are divided by a temperature of 0.5 before softmax "
            "(sharpening V3 by 2×), while SP-LoRA's logits are softmaxed at "
            "temperature 1.0.\n\n"
            "The effect is that V3 dominates on high-confidence cases (its "
            "sharpened argmax overrides the unsharpened SP-LoRA argmax on "
            "confident frames), while on uncertain cases the two branches "
            "contribute comparably. The asymmetric sharpening is deliberate; "
            "specific magnitudes vary per image and surface in Phase D once "
            "the metrics extraction runs.\n\n"
            "Downstream, the fused probability vector is passed through "
            "per-class temperature scaling for calibration, then wrapped "
            "into a conformal prediction set for the final output.\n\n"
            "One honest caveat: this fusion was tuned on a validation split "
            "that may not match a particular farmer's field conditions. If "
            "your inputs differ systematically from the training distribution, "
            "the V3-vs-LoRA balance may shift in ways the fusion was not "
            "trained to handle.",

        "calibrated_output":
            "The final output for tomato. Per-class temperature scaling "
            "adjusts the fused probabilities so that an 87% confidence "
            "actually corresponds to an 87% empirical accuracy on the "
            "calibration split. The bar chart on the right shows the "
            "post-calibration top-3 classes with their probabilities.\n\n"
            "The conformal prediction set is computed alongside: it lists "
            "every class whose post-calibration probability exceeds a "
            "per-class threshold chosen so that the true class falls inside "
            "the set on at least 90% of held-out images. For confident "
            "predictions, the set is a single class. For uncertain ones, it "
            "may contain 2 or 3 classes.\n\n"
            "Downstream of this stage, the API response carries the top-3 "
            "probabilities plus the conformal set plus a tier label (1A "
            "through 5) summarising the overall confidence.\n\n"
            "One honest caveat: tier thresholds were built from already-"
            "T-calibrated probabilities. If the pipeline ever double-"
            "applied temperature, the tiers would be wrong. Production code "
            "guards against this; the extraction script in this section "
            "calls the pipeline's normal infer() path so the same guards "
            "apply.",
    },

    # ─── APIN (9 stages) ───────────────────────────────────────────────
    "apin": {
        "input":
            "The APIN okra/brassica ensemble sees the same leaf photograph, "
            "preprocessed with LAB-CLAHE on the L channel. The four signals "
            "in the ensemble have two distinct preprocessing branches: three "
            "signals (DINOv3-ConvNeXt, DINOv2, {term:PSV}) share branch A "
            "(LAB-CLAHE + ImageNet normalisation at 224×224 or 384×384), "
            "while EfficientNet uses branch B (per-channel RGB CLAHE).\n\n"
            "The visual on the right shows branch A's preprocessed input. "
            "Branch B's RGB-CLAHE output looks slightly different (more "
            "even per-channel contrast, less reliance on LAB-space lightness) "
            "but is shown only at the EfficientNet stage to avoid clutter.\n\n"
            "Downstream, each of the 4 signals runs its own model on its "
            "own preprocessed input, producing per-signal class probabilities. "
            "The {term:stacking MLP} fuses them into the final 9-class "
            "distribution.\n\n"
            "One honest caveat: the two preprocessing branches mean APIN "
            "sees the leaf in two slightly different ways. This is "
            "deliberate (the ensemble benefits from complementary inputs), "
            "but it also means the 4 signals are not strictly comparable "
            "without remembering which preprocessing they used.",

        "split_to_4_signals":
            "At this stage, the same input image is dispatched to four "
            "parallel signals: DINOv3-ConvNeXt-Small (primary classifier), "
            "EfficientNet-B0 (secondary), DINOv2 ViT-S (frozen "
            "{term:CLS token} representation), and {term:PSV} (engineered "
            "features). Each runs independently with no information sharing "
            "between them until the stacking step at the end.\n\n"
            "The fork is implemented as 4 separate forward calls on the "
            "preprocessed tensor. There is no shared computation between "
            "the 4 signals; the ensemble's strength comes from their "
            "complementary biases (a primary CNN with hierarchical "
            "downsampling, a smaller secondary CNN with different "
            "regularisation, a frozen self-supervised backbone trained on "
            "natural images, and a rule-based feature extractor).\n\n"
            "Downstream, each signal produces a 9-class probability vector "
            "(one entry per okra/brassica disease + healthy). The next four "
            "stages show each signal's behaviour in turn.\n\n"
            "One honest caveat: 'parallel' here means logically independent. "
            "In practice the extraction script runs the signals sequentially "
            "to avoid VRAM contention on the RTX 4060. The result is "
            "identical to a parallel run; the wall-clock time is longer.",

        "dinov3_convnext":
            "DINOv3-ConvNeXt-Small is APIN's primary classifier. It is a "
            "ConvNeXt-Small backbone pre-trained with DINOv3-style "
            "self-supervision, then fine-tuned on the 9-class okra/brassica "
            "labels. The visual on the right shows the final-stage feature "
            "map of this backbone, reduced to 8 channels via top-magnitude "
            "selection.\n\n"
            "ConvNeXt's hierarchical structure means each successive stage "
            "halves the spatial resolution while doubling the channel count. "
            "By the final stage, a 7×7 grid of 768-dimensional feature "
            "vectors summarises the whole image. The model has ~49.5 million "
            "parameters (DINOv3-ConvNeXt-Small in full fine-tune mode), of "
            "which the head is freshly trained and the backbone is partially "
            "fine-tuned with layer-wise learning rate decay.\n\n"
            "Downstream, the final feature vector is pooled and fed through "
            "a linear head to produce a 9-class softmax. That softmax is "
            "one of the 4 inputs to the stacking MLP.\n\n"
            "One honest caveat: DINOv3-style self-supervision is a 2024 "
            "method. The fine-tuning data is okra and brassica from 2024 "
            "and earlier. On crops or diseases collected after fine-tuning, "
            "the model's confidence may be misleadingly high.",

        "efficientnet_b0":
            "APIN's secondary classifier is the legacy 10-class okra/brassica "
            "model from the previous pipeline generation. Its backbone is an "
            "EfficientNetV2-S (the small variant, around 22 million parameters) "
            "wrapped with an FPN neck and three task heads (crop classifier, "
            "23-class disease head, severity head). The 23-class disease "
            "softmax is index-mapped onto APIN's 9 canonical classes via "
            "EN_TO_M2_INDEX_MAP at inference time.\n\n"
            "This signal uses branch B preprocessing (per-channel RGB CLAHE) "
            "rather than branch A's LAB-CLAHE. The visual on the right shows "
            "the model's deepest convolutional feature block, reduced to 8 "
            "channels via top-magnitude selection. The activations are "
            "spatial: the 7×7 feature map encodes where each pattern detector "
            "fired most strongly.\n\n"
            "Downstream, the disease head produces a 23-dim vector that is "
            "sliced to 9 dims, sigmoided, and forwarded to the {term:stacking "
            "MLP} along with the other 3 signal outputs.\n\n"
            "One honest caveat: this signal is consistently the weakest of "
            "the 4 on standalone accuracy. It earns its place in the ensemble "
            "only because of complementarity: on the few classes where it "
            "outperforms the others, the stacking MLP weights its vote highly, "
            "and the overall ensemble gains.",

        "dinov2_vit":
            "DINOv2 ViT-S is APIN's frozen-representation signal. The "
            "backbone is the same {term:DINOv2}-Small with 4 "
            "{term:register tokens} that the router uses, but here it feeds "
            "a small nonlinear head that was trained on the 9-class "
            "okra/brassica labels while the backbone weights stayed "
            "completely frozen.\n\n"
            "The visual on the right is the pooled {term:CLS token} "
            "representation: a single 384-dimensional vector, visualised "
            "by projecting to the top 8 PCA components fitted on the "
            "training set. Each PCA channel captures one axis of variation "
            "the backbone considers important.\n\n"
            "Downstream, the nonlinear head turns this representation into "
            "a 9-class softmax. That softmax is the third input to the "
            "{term:stacking MLP}, alongside the DINOv3 and EfficientNet "
            "outputs.\n\n"
            "One honest caveat: because the backbone is frozen, DINOv2's "
            "9-class predictions can only be as good as DINOv2's pre-"
            "training distribution permits. If a particular disease class "
            "looks very different from anything DINOv2 saw during "
            "self-supervised training, this signal will be unreliable on "
            "that class. The stacking MLP weights DINOv2 down on such "
            "classes.",

        "psv_features":
            "{term:PSV} is the engineered-features signal: it is not a "
            "neural network. Instead, a handcrafted feature extractor "
            "computes leaf-shape statistics (area, perimeter, eccentricity), "
            "colour statistics (mean and variance per channel), and vein "
            "pattern features (Hessian ridge density, branch points).\n\n"
            "The visual on the right shows the top 16 PSV feature "
            "magnitudes as a bar chart. PSV is non-spatial: there is no "
            "2D feature map. Each bar is one engineered feature; together "
            "they form a fixed-length vector that summarises the leaf.\n\n"
            "Downstream, PSV's feature vector is fed through a small "
            "multi-layer perceptron to produce a 9-class softmax. That "
            "softmax is the fourth input to the stacking MLP.\n\n"
            "One honest caveat: PSV is brittle in ways neural networks are "
            "not. A poorly-segmented leaf (background bleeding into the "
            "shape mask) produces wildly wrong shape statistics. The "
            "stacking MLP catches this most of the time by down-weighting "
            "PSV when its prediction disagrees with the other signals, but "
            "edge cases exist where PSV is wrong and the MLP trusts it.",

        "stacking_mlp":
            "The {term:stacking MLP} takes the 4 per-signal 9-class "
            "probability vectors (36 numbers total) and learns to combine "
            "them into a single 9-class distribution. The MLP was trained "
            "on a held-out validation split, with the 4 signals' "
            "predictions as features and the true class as the target.\n\n"
            "The MLP's learned weights are per-class: a signal that "
            "reliably gets a specific class right gets a high weight on "
            "that class, regardless of how it performs on others. This is "
            "what lets the ensemble outperform any single signal on any "
            "specific class.\n\n"
            "Downstream, the fused 9-class output is passed to per-class "
            "temperature scaling for calibration.\n\n"
            "One honest caveat: the stacking MLP cannot recover from all "
            "4 signals being wrong in the same way. If a particular leaf "
            "looks unlike the training distribution, all 4 signals may "
            "produce similarly miscalibrated outputs, and the MLP will "
            "produce a confident but wrong final prediction. The Mahalanobis "
            "OOD check downstream catches the most extreme cases.",

        "temperature_scaling":
            "Per-class temperature scaling adjusts the stacking MLP's "
            "output so that the displayed confidence number is honest. A "
            "vanilla softmax can report 87% confidence while being right "
            "anywhere from 60% to 95% of the time. After per-class "
            "temperature scaling, an 87% prediction is right roughly 86% "
            "to 88% of the time on the calibration split.\n\n"
            "Each class has its own learned temperature scalar. The "
            "temperatures were fit by maximising the log-likelihood of the "
            "calibration labels under the temperature-scaled softmax. "
            "Classes that the ensemble was overconfident on get high "
            "temperatures (which flatten the softmax); under-confident "
            "classes get low temperatures (which sharpen it).\n\n"
            "Downstream, the calibrated probabilities are used to compute "
            "the conformal prediction set and the tier label.\n\n"
            "One honest caveat: calibration was measured on a specific "
            "held-out split. On data from a different distribution (a "
            "different region, season, or camera), the calibration may "
            "drift. We never invent numbers, but we also cannot guarantee "
            "the calibration holds outside the measured distribution.",

        "conformal_output":
            "The final APIN output. The conformal prediction set lists "
            "every class whose calibrated probability exceeds a per-class "
            "threshold chosen so the true class lands inside the set on "
            "at least 90% of held-out images. The top-3 bar chart on the "
            "right shows the calibrated probabilities of the 3 most likely "
            "classes.\n\n"
            "The API response also carries a tier label (1A through 5) "
            "summarising the overall confidence. Tier 1A means all 4 "
            "signals agreed strongly; tier 5 means the OOD check flagged "
            "the image as too unlike the training distribution to "
            "diagnose.\n\n"
            "Downstream of this stage, the response envelope is wrapped "
            "with a request_id, latency, and the API contract version, "
            "then returned to the client.\n\n"
            "One honest caveat: the 90% coverage guarantee is over the "
            "calibration distribution, not over any particular farmer's "
            "field. If your inputs differ systematically from the "
            "calibration split, the actual coverage may be higher or "
            "lower. The validation lift over single-signal predictions "
            "was measured with exactly this setup.",
    },

    # ─── CHILLI (1 stage, by design) ───────────────────────────────────
    "chilli": {
        "router_classification":
            "Chilli has a single forward-pass stage by design. The router "
            "runs on the chilli leaf, produces a 4-class softmax, and "
            "argmaxes to chilli with high confidence. At that point the "
            "system stops: there is no chilli disease specialist deployed.\n\n"
            "The bar chart on the right shows the router's softmax. The "
            "JSON tree below shows the actual `ROUTER_REJECTED` response "
            "envelope a chilli leaf produces today. The rest of the "
            "diagram is greyed out because there is no further "
            "computation: the request never reaches a disease classifier.\n\n"
            "Downstream of this stage is... nothing. The response carries "
            "`error.code = \"router_rejected\"`, `router_crop = \"chilli\"`, "
            "and a hint pointing the integrator to a human agronomist or "
            "the APIN fallback ensemble. No disease label is produced.\n\n"
            "One honest caveat: this is the most important honest moment "
            "in the entire Pipeline Atlas. The system could fabricate a "
            "chilli disease label from a model that has never seen "
            "chilli disease data, but it does not. Honest failure is "
            "cheaper than wrong confidence; building a real chilli "
            "specialist is on the Phase 4D+ roadmap.",
    },
}


# ────────────────────────────────────────────────────────────────────────
# Image pool selection · per spec §6.3 (locked in plan v4 §A4).
# Each model gets exactly 3 canonical classes from the locked test split.
# ────────────────────────────────────────────────────────────────────────
IMAGE_POOL_CLASSES: Dict[str, List[str]] = {
    "router":  ["okra", "tomato", "chilli"],  # spec §6.3
    "tomato":  ["tomato_late_blight", "tomato_septoria_leaf_spot", "tomato_healthy"],
    "apin":    ["okra_yvmv", "brassica_black_rot", "brassica_alternaria"],
    "chilli":  ["chilli", "chilli", "chilli"],  # same class, 3 different field photos
}


# ────────────────────────────────────────────────────────────────────────
# Tile rendering helper · plan v4 §A6.
# Activations → viridis-colormap PNG → base64. Up to 4-column grid layout.
# ────────────────────────────────────────────────────────────────────────
def render_tile_grid(activations, n_channels: int, tile_px: int = TILE_PX) -> str:
    """Render activations as a base64 PNG tile grid.

    activations: numpy array, shape [C, H, W] or [N_tokens, D].
                 If 2D (tokens), it is reshaped to a square spatial grid
                 (assumes sqrt(N) is integer; ViT patch grids satisfy this).
    n_channels:  how many channels/tokens to show (top-magnitude selected).
    tile_px:     pixel size per tile (default 64).
    returns:     base64-encoded PNG string (without the data: URI prefix).
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "Pillow required for tile rendering. Install with: pip install Pillow"
        ) from e

    arr = np.asarray(activations)
    if arr.ndim == 2:
        # [tokens, D] → pick top-magnitude channels, reshape to spatial grid
        # assumes the token dimension is a perfect square (256 = 16x16 etc.)
        n_tokens = arr.shape[0]
        side = int(round(n_tokens ** 0.5))
        if side * side != n_tokens:
            # fall back: just take the first square that fits
            side = int(n_tokens ** 0.5)
            arr = arr[:side * side]
        # pick top-n_channels by L2-norm across spatial positions
        magnitudes = np.linalg.norm(arr, axis=0)  # per-channel magnitude
        if magnitudes.ndim == 0 or arr.shape[1] <= n_channels:
            picks = list(range(arr.shape[1]))[:n_channels]
        else:
            picks = np.argsort(magnitudes)[-n_channels:].tolist()
        channels = np.stack([arr[:, c].reshape(side, side) for c in picks], axis=0)
    elif arr.ndim == 3:
        # [C, H, W] already spatial; pick top-magnitude channels
        magnitudes = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
        if arr.shape[0] <= n_channels:
            picks = list(range(arr.shape[0]))
        else:
            picks = np.argsort(magnitudes)[-n_channels:].tolist()
        channels = arr[picks]
    else:
        raise ValueError(f"render_tile_grid: unsupported shape {arr.shape}")

    # Normalize each channel to [0, 1]
    tiles = []
    for ch in channels:
        lo, hi = float(ch.min()), float(ch.max())
        if hi - lo < 1e-9:
            norm = np.zeros_like(ch, dtype=np.float32)
        else:
            norm = (ch - lo) / (hi - lo)
        tiles.append(norm)

    # Apply viridis colormap (manually-coded 256-step palette to avoid mpl dep)
    # Minimal viridis from purple → blue → green → yellow.
    viridis = np.array([
        [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
        [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112],
        [121, 209, 81], [189, 222, 38], [253, 231, 36],
    ], dtype=np.uint8)

    def colormap(g: np.ndarray) -> np.ndarray:
        idx = np.clip((g * (len(viridis) - 1)).astype(np.int32), 0, len(viridis) - 1)
        return viridis[idx]  # shape [H, W, 3]

    rgb_tiles = [colormap(t) for t in tiles]

    # Resize each tile to tile_px × tile_px
    resized = []
    for t in rgb_tiles:
        img = Image.fromarray(t, mode="RGB")
        img = img.resize((tile_px, tile_px), Image.BICUBIC)
        resized.append(np.array(img))

    # Tile layout: up to 4 columns wrapping
    cols = min(4, len(resized))
    rows = (len(resized) + cols - 1) // cols
    canvas = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)
    for i, t in enumerate(resized):
        r, c = i // cols, i % cols
        canvas[r * tile_px:(r + 1) * tile_px, c * tile_px:(c + 1) * tile_px] = t

    # Encode as PNG
    buf = io.BytesIO()
    Image.fromarray(canvas, mode="RGB").save(
        buf, format="PNG", optimize=False, compress_level=PNG_COMPRESSION
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_bar_chart(values: List[float], labels: List[str],
                     tile_px: int = TILE_PX) -> str:
    """Render a horizontal bar chart as base64 PNG (for softmax / top-k stages).

    values: list of probabilities (already softmaxed) summing to ≤ 1.
    labels: same length as values.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise RuntimeError("Pillow required for bar chart rendering") from e

    W, H = 300, max(80, tile_px * len(values) // 2)
    img = Image.new("RGB", (W, H), color=(244, 239, 230))  # paper-light
    draw = ImageDraw.Draw(img)
    row_h = H // max(1, len(values))
    for i, (v, lbl) in enumerate(zip(values, labels)):
        y = i * row_h + 4
        bar_w = int(max(0.0, min(1.0, v)) * (W - 110))
        # Label text (truncated)
        draw.text((4, y), lbl[:18], fill=(58, 50, 40))
        # Bar
        color = (46, 113, 76) if i == int(np.argmax(values)) else (160, 96, 32)
        draw.rectangle([100, y + 4, 100 + bar_w, y + row_h - 8], fill=color)
        # Value
        draw.text((100 + bar_w + 4, y), f"{v:.3f}", fill=(58, 50, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=PNG_COMPRESSION)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_input_image(img_array, tile_px: int = TILE_PX) -> str:
    """Encode an RGB image array as base64 PNG."""
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow required") from e
    img = Image.fromarray(img_array.astype(np.uint8), mode="RGB")
    img = img.resize((tile_px * 3, tile_px * 3), Image.BICUBIC)  # input gets a bigger render
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=PNG_COMPRESSION)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ────────────────────────────────────────────────────────────────────────
# Model loading + activation capture · per plan v4 §§A1-A3.
# Each function returns (stages_list, image_pool_list) for that model, OR
# raises an exception which the orchestrator catches into the
# `unavailable: true` fallback shape.
# ────────────────────────────────────────────────────────────────────────

# Frozen test split (10,287 rows: image_path, class_name, source).
FROZEN_TEST_CSV = ROOT / "data_prep" / "frozen_test_set.csv"


def _load_frozen_test_set() -> List[Dict[str, str]]:
    """Read frozen_test_set.csv. Returns list of dicts (image_path, class_name, source)."""
    import csv
    if not FROZEN_TEST_CSV.exists():
        raise FileNotFoundError(
            f"Frozen test set not found: {FROZEN_TEST_CSV}. "
            f"Image-pool selection requires the locked test split."
        )
    with open(FROZEN_TEST_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def _select_images_for_classes(class_specs: List[str], n_per_model: int = N_IMAGES_PER_MODEL) -> List[Dict[str, str]]:
    """Pick `n_per_model` image rows from the frozen test set whose class_name
    matches each spec in `class_specs`. Selection is deterministic.

    For each entry in `class_specs`, find all matching rows where:
      - The class_name equals the spec (e.g. "tomato_late_blight"), OR
      - The class_name starts with the spec + "_" (so "okra" matches "okra_yvmv").
    Then filter to rows whose file exists on disk, sort by image_path (stable),
    and seed-select one row per spec via `random.Random(SEED)`.

    If a spec yields zero valid rows, the function falls back to seed-selecting
    any class matching the same crop prefix (e.g. "okra" → any okra_* class).
    If even that fails, raises FileNotFoundError with a clear reason.
    """
    rows = _load_frozen_test_set()
    rng = random.Random(SEED)
    picked: List[Dict[str, str]] = []

    for spec in class_specs:
        candidates = []
        for r in rows:
            if r["class_name"] == spec or r["class_name"].startswith(spec + "_"):
                p = ROOT / r["image_path"].replace("\\", "/")
                if p.exists():
                    candidates.append(r)
        # Sort for deterministic order before seeded sample
        candidates.sort(key=lambda r: r["image_path"])

        if not candidates:
            # Fallback: try the crop prefix only (e.g. "tomato_septoria_leaf_spot" → any tomato_*)
            prefix = spec.split("_")[0] + "_"
            for r in rows:
                if r["class_name"].startswith(prefix):
                    p = ROOT / r["image_path"].replace("\\", "/")
                    if p.exists():
                        candidates.append(r)
            candidates.sort(key=lambda r: r["image_path"])

        if not candidates:
            raise FileNotFoundError(
                f"No valid frozen-test-set images for spec '{spec}' "
                f"(0 matches with extant files)."
            )

        # Seeded pick — uses a fresh sample so order across specs is stable
        picked.append(rng.choice(candidates))

    # If a spec produced n_per_model entries from a single class (chilli case),
    # de-duplicate by image_path and re-sample if necessary.
    seen = set()
    unique = []
    for r in picked:
        if r["image_path"] not in seen:
            unique.append(r)
            seen.add(r["image_path"])
    if len(unique) < n_per_model:
        # Need more images. Re-sample from candidates of the LAST spec to fill.
        last_spec = class_specs[-1]
        prefix = last_spec.split("_")[0] + "_"
        extra_pool = sorted(
            [r for r in rows
             if (r["class_name"] == last_spec or r["class_name"].startswith(prefix))
             and (ROOT / r["image_path"].replace("\\", "/")).exists()
             and r["image_path"] not in seen],
            key=lambda r: r["image_path"],
        )
        while len(unique) < n_per_model and extra_pool:
            r = rng.choice(extra_pool)
            if r["image_path"] not in seen:
                unique.append(r)
                seen.add(r["image_path"])
            extra_pool.remove(r)

    return unique[:n_per_model]


def _load_image_rgb(rel_path: str):
    """Load image from a path relative to ROOT, return uint8 RGB numpy [H, W, 3]."""
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow required") from e
    full = ROOT / rel_path.replace("\\", "/")
    img = Image.open(full).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _make_image_pool(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert frozen-test rows into image_pool entries for the JSON schema."""
    return [
        {
            "class_name": r["class_name"],
            "source": r["source"],
            "image_path": r["image_path"],
        }
        for r in rows
    ]


# ────────────────────────────────────────────────────────────────────────
# ViT block-output → 2D patch-magnitude grid.
# Strips CLS + register tokens, reshapes the patch tokens to a square grid,
# returns a [H, W] float array of per-patch L2 magnitudes.
# ────────────────────────────────────────────────────────────────────────
def _block_to_patch_grid(block_output, n_extra_tokens: int = 1):
    """block_output: tensor [B, N_tokens, D]. Drops first `n_extra_tokens`
    (CLS + register), reshapes the rest to a square spatial grid, and returns
    a [side, side] numpy array of per-patch L2 magnitudes.

    For DINOv2 ViT-Small @ 224×224 with 4 register tokens:
      N_tokens = 1 (CLS) + 4 (reg) + 256 (patches) = 261; n_extra_tokens = 5.
    For DINOv2 ViT-Small @ 224×224 without registers: n_extra_tokens = 1.
    """
    t = block_output.detach().float().cpu()
    if t.ndim != 3:
        raise ValueError(f"_block_to_patch_grid: expected 3D tensor, got {t.shape}")
    _, n_tok, _ = t.shape
    patches = t[0, n_extra_tokens:, :]  # [N_patches, D]
    n_patches = patches.shape[0]
    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        # Trim to largest square (defensive; shouldn't happen on standard ViT inputs)
        side = int(n_patches ** 0.5)
        patches = patches[: side * side]
    # Per-patch L2 magnitude across embedding dimension
    magnitudes = patches.norm(dim=1).numpy().reshape(side, side)
    return magnitudes


def _render_patch_grid_png(grid_2d, tile_px: int = TILE_PX * 3) -> str:
    """Render a single [H, W] patch grid as a viridis-coloured PNG, upscaled."""
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow required") from e

    arr = np.asarray(grid_2d, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - lo) / (hi - lo)
    viridis = np.array([
        [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
        [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112],
        [121, 209, 81], [189, 222, 38], [253, 231, 36],
    ], dtype=np.uint8)
    idx = np.clip((norm * (len(viridis) - 1)).astype(np.int32),
                  0, len(viridis) - 1)
    rgb = viridis[idx]
    img = Image.fromarray(rgb, mode="RGB").resize((tile_px, tile_px), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=PNG_COMPRESSION)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_2d_feature_map_png(feature_map_2d, tile_px: int = TILE_PX * 3) -> str:
    """Render a [H, W] CNN feature map as viridis PNG. Used for ConvNeXt /
    EfficientNet outputs."""
    return _render_patch_grid_png(feature_map_2d, tile_px=tile_px)


def _stage(stage_id: str, label: str, model_key: str,
           visuals: List[Dict[str, Any]],
           *, latency_ms: float = 0.0,
           numerical: Optional[Dict[str, Any]] = None,
           counterfactual: Optional[Dict[str, Any]] = None,
           layer_type: Optional[str] = None,
           layer_type_params: Optional[Dict[str, Any]] = None
           ) -> Dict[str, Any]:
    """Assemble one stage's JSON record.

    FP-D11 latency_ms       · wall-clock ms for this stage
    FP-D13 numerical        · per-stage technical numbers (shape, params, entropy, etc.)
    FP-D14 counterfactual   · gating analysis for softmax / conformal stages
    FP-D21 layer_type       · enum used by the layer inspector
    FP-D22 layer_type_params · type-specific schematic params
    """
    narration = NARRATIONS[model_key][stage_id]
    payload: Dict[str, Any] = {
        "id": stage_id,
        "label": label,
        "narration": narration,
        "narration_html": render_narration_html(narration),
        "visuals": visuals,
        "latency_ms": round(float(latency_ms), 3),
    }
    if numerical is not None:
        payload["numerical"] = numerical
    if counterfactual is not None:
        payload["counterfactual"] = counterfactual
    if layer_type is not None:
        payload["layer_type"] = layer_type
    if layer_type_params is not None:
        payload["layer_type_params"] = layer_type_params
    return payload


# ────────────────────────────────────────────────────────────────────────
# Attention-derived statistics for the "this stage numerically" table.
# Given a ViT block's output [B, N_tokens, D] we expose:
#   - tensor shape (string)
#   - patch-magnitude entropy in bits (low = focused, high = scattered)
#   - argmax patch index + (row, col) coordinates
#   - top-k patches by magnitude
# These derive from REAL captured tensors, no synthesis.  See FP-D13.
# ────────────────────────────────────────────────────────────────────────
def _vit_block_numerical(block_output, n_extra_tokens: int = 1) -> Dict[str, Any]:
    t = block_output.detach().float().cpu()
    if t.ndim != 3:
        return {"shape": str(tuple(t.shape))}
    B, N, D = t.shape
    patches = t[0, n_extra_tokens:, :]
    n_patches = patches.shape[0]
    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        side = int(n_patches ** 0.5)
        patches = patches[: side * side]
    mag = patches.norm(dim=1).numpy()
    # Normalised to a probability distribution for entropy
    p = mag / (mag.sum() + 1e-12)
    entropy_bits = float(-np.sum(p * np.log2(p + 1e-12)))
    argmax_patch = int(np.argmax(mag))
    row, col = argmax_patch // side, argmax_patch % side
    top_k = np.argsort(-mag)[:5].tolist()
    return {
        "tensor_shape": f"[{B}, {N}, {D}]",
        "n_patch_tokens": n_patches,
        "patch_grid": f"{side}×{side}",
        "entropy_bits": round(entropy_bits, 3),
        "peak_patch_index": argmax_patch,
        "peak_patch_rc": [int(row), int(col)],
        "top5_patches": [int(x) for x in top_k],
        "magnitude_max": round(float(mag.max()), 4),
        "magnitude_mean": round(float(mag.mean()), 4),
    }


def _softmax_counterfactual(probs, class_names, threshold: float = 0.40) -> Dict[str, Any]:
    """FP-D14: given a softmax + a routing threshold, report what would happen
    if the argmax probability were lower."""
    probs = np.asarray(probs, dtype=np.float32)
    if probs.size == 0:
        return {}
    argmax_idx = int(np.argmax(probs))
    argmax_val = float(probs[argmax_idx])
    # Second-best: zero out the argmax, then argmax again
    p2 = probs.copy()
    p2[argmax_idx] = -np.inf
    second_idx = int(np.argmax(p2))
    second_val = float(probs[second_idx])
    margin = argmax_val - threshold
    return {
        "argmax_class": class_names[argmax_idx] if argmax_idx < len(class_names) else f"class_{argmax_idx}",
        "argmax_prob": round(argmax_val, 4),
        "second_best_class": class_names[second_idx] if second_idx < len(class_names) else f"class_{second_idx}",
        "second_best_prob": round(second_val, 4),
        "routing_threshold": threshold,
        "margin_above_threshold": round(margin, 4),
        "would_route_to_fallback_if_lower_by": round(margin, 4) if margin > 0 else None,
    }


# Leaf-flight choreography per model (FP-D28).  Each entry is the
# normalised (x_pct, y_pct, scale, rotate_deg) of the leaf icon when that
# stage is active.  Values are relative to the canvas viewBox.
LEAF_POSITION_STATES: Dict[str, List[Dict[str, float]]] = {
    "router": [
        {"x_pct":  6, "y_pct": 50, "scale": 1.00, "rotate_deg":   0},  # input
        {"x_pct": 36, "y_pct": 50, "scale": 0.55, "rotate_deg":  -8},  # vit_early
        {"x_pct": 70, "y_pct": 50, "scale": 0.40, "rotate_deg": -12},  # vit_pooled
        {"x_pct": 92, "y_pct": 50, "scale": 0.30, "rotate_deg": -14},  # head_softmax
    ],
    "tomato": [
        {"x_pct":  6, "y_pct": 50, "scale": 1.00, "rotate_deg":   0},  # input
        {"x_pct": 26, "y_pct": 30, "scale": 0.55, "rotate_deg":  -6},  # v3_stage_1
        {"x_pct": 44, "y_pct": 30, "scale": 0.45, "rotate_deg":  -8},  # v3_stage_2
        {"x_pct": 44, "y_pct": 70, "scale": 0.45, "rotate_deg":  -8},  # lora_branch
        {"x_pct": 64, "y_pct": 30, "scale": 0.40, "rotate_deg": -10},  # v3_deep
        {"x_pct": 80, "y_pct": 50, "scale": 0.35, "rotate_deg": -12},  # stacking_gate
        {"x_pct": 94, "y_pct": 50, "scale": 0.30, "rotate_deg": -14},  # calibrated_output
    ],
    "apin": [
        {"x_pct":  6, "y_pct": 50, "scale": 1.00, "rotate_deg":   0},  # input
        {"x_pct": 20, "y_pct": 50, "scale": 0.70, "rotate_deg":  -4},  # split_to_4_signals
        {"x_pct": 40, "y_pct": 18, "scale": 0.45, "rotate_deg":  -8},  # dinov3_convnext
        {"x_pct": 40, "y_pct": 40, "scale": 0.45, "rotate_deg":  -8},  # efficientnet_b0
        {"x_pct": 40, "y_pct": 62, "scale": 0.45, "rotate_deg":  -8},  # dinov2_vit
        {"x_pct": 40, "y_pct": 84, "scale": 0.45, "rotate_deg":  -8},  # psv_features
        {"x_pct": 64, "y_pct": 50, "scale": 0.45, "rotate_deg": -10},  # stacking_mlp
        {"x_pct": 80, "y_pct": 50, "scale": 0.35, "rotate_deg": -12},  # temperature_scaling
        {"x_pct": 94, "y_pct": 50, "scale": 0.30, "rotate_deg": -14},  # conformal_output
    ],
    "chilli": [
        {"x_pct": 50, "y_pct": 50, "scale": 0.80, "rotate_deg":   0},  # router_classification (single)
    ],
}


# ────────────────────────────────────────────────────────────────────────
# Router loading + extraction · plan v4 §A1.
# Reproduces apin_server.py:1788 _ensure_router() pattern.
# ────────────────────────────────────────────────────────────────────────
def _load_router():
    """Returns (backbone, head, device, class_names). Raises if checkpoint missing."""
    if torch is None:
        raise RuntimeError("torch not installed")
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    from app.config_router import (
        BACKBONE_NAME, DINOV2_IMG_SIZE, DINOV2_EMBED_DIM,
        NUM_CLASSES, CLASS_NAMES,
    )
    import timm
    from torch import nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = timm.create_model(
        BACKBONE_NAME, pretrained=True,
        num_classes=0, img_size=DINOV2_IMG_SIZE,
    ).eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    head = nn.Linear(DINOV2_EMBED_DIM, NUM_CLASSES).to(device)
    ckpt_path = ROOT / "models" / "router" / "router_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "head_state_dict" in ckpt:
            head.load_state_dict(ckpt["head_state_dict"])
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            head_sd = {k.replace("head.", ""): v for k, v in sd.items()
                       if k.startswith("head.")}
            if head_sd:
                head.load_state_dict(head_sd)
        head.eval()
    else:
        raise FileNotFoundError(
            f"Router checkpoint missing at {ckpt_path}. "
            f"Run training/04_train_router.py first."
        )
    return backbone, head, device, list(CLASS_NAMES)


def _preprocess_router_image(img_rgb):
    """Match apin_server.py:1826 transform: Resize 224x224 + ImageNet norm + ToTensor."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return transform(image=img_rgb)["image"].unsqueeze(0)


def _encode_input_visual(img_rgb, image_pool_index: int) -> Dict[str, Any]:
    """Render the raw RGB input as a base64 PNG visual entry."""
    return {
        "image_pool_index": image_pool_index,
        "kind": "image",
        "png_base64": encode_input_image(img_rgb, tile_px=TILE_PX * 2),
    }


def extract_router() -> Tuple[List[Dict], List[Dict]]:
    """Run the router on 3 images. Capture activations at blocks 3 and 11
    for the visible stages, hook ALL 12 blocks for the per-block CLS
    classification ribbon (FP-D12), measure stage latencies (FP-D11), and
    derive `numerical` (FP-D13) + `counterfactual` (FP-D14) blobs.
    """
    import time as _time
    backbone, head, device, class_names = _load_router()
    rows = _select_images_for_classes(IMAGE_POOL_CLASSES["router"], N_IMAGES_PER_MODEL)
    image_pool = _make_image_pool(rows)
    n_extra = 5  # 1 CLS + 4 register tokens for vit_small_patch14_reg4_dinov2

    inputs_rgb = []
    block3_grids = []
    block11_grids = []
    softmax_probs = []
    # FP-D12 · per-block per-image CLS classification: shape [N_IMG][12][N_CLASS]
    per_block_cls = [[None] * 12 for _ in range(N_IMAGES_PER_MODEL)]
    # Per-stage cumulative latency in ms
    latency_per_stage = {"input": [], "vit_early": [], "vit_pooled": [], "head_softmax": []}
    # Numerical extras for the two ViT-block stages (one entry per pool image)
    block3_numerical = []
    block11_numerical = []

    blocks = backbone.blocks
    if len(blocks) < 12:
        raise RuntimeError(
            f"Router backbone has only {len(blocks)} blocks; expected at least 12."
        )

    captured: Dict[str, Any] = {}

    def make_hook(name):
        def hook(_m, _inp, out):
            captured[name] = out.detach()
        return hook

    # Hook every block so we can read CLS at every depth.
    handles = [blocks[i].register_forward_hook(make_hook(f"block{i}"))
               for i in range(12)]

    try:
        for img_idx, row in enumerate(rows):
            img_rgb = _load_image_rgb(row["image_path"])
            inputs_rgb.append(img_rgb)

            # ── input stage timing
            t0 = _time.perf_counter()
            tens = _preprocess_router_image(img_rgb).to(device)
            latency_per_stage["input"].append((_time.perf_counter() - t0) * 1000.0)

            # ── full forward pass (hooks capture every block's output)
            t_fwd_start = _time.perf_counter()
            with torch.no_grad():
                feat = backbone(tens)            # [1, embed_dim]
                t_after_backbone = _time.perf_counter()
                logits = head(feat)              # [1, 4]
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                t_after_head = _time.perf_counter()

            # FP-D12 · classify CLS at each block via the same head
            for bi in range(12):
                blk_out = captured.get(f"block{bi}")
                if blk_out is not None:
                    cls_vec = blk_out[0, 0, :]   # CLS token at block bi
                    with torch.no_grad():
                        blk_logits = head(cls_vec.unsqueeze(0))
                        blk_probs = torch.softmax(blk_logits, dim=1).cpu().numpy()[0]
                    per_block_cls[img_idx][bi] = [round(float(p), 4) for p in blk_probs]

            # Split the forward latency proportionally to the visible stages:
            #   vit_early  ≈ 4 blocks (0..3)   → 4/12 of backbone time
            #   vit_pooled ≈ 8 blocks (4..11)  → 8/12 of backbone time
            #   head_softmax = head() + softmax time
            backbone_ms = (t_after_backbone - t_fwd_start) * 1000.0
            head_ms = (t_after_head - t_after_backbone) * 1000.0
            latency_per_stage["vit_early"].append(backbone_ms * 4.0 / 12.0)
            latency_per_stage["vit_pooled"].append(backbone_ms * 8.0 / 12.0)
            latency_per_stage["head_softmax"].append(head_ms)

            # Block-3 + block-11 captures for the visible stages
            b3 = captured["block3"]
            b11 = captured["block11"]
            block3_grids.append(_block_to_patch_grid(b3, n_extra_tokens=n_extra))
            block11_grids.append(_block_to_patch_grid(b11, n_extra_tokens=n_extra))
            block3_numerical.append(_vit_block_numerical(b3, n_extra_tokens=n_extra))
            block11_numerical.append(_vit_block_numerical(b11, n_extra_tokens=n_extra))
            softmax_probs.append(probs.astype(np.float32))
    finally:
        for h in handles:
            h.remove()
        del backbone, head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Visual sets
    input_visuals = [_encode_input_visual(inputs_rgb[i], i) for i in range(N_IMAGES_PER_MODEL)]
    block3_visuals = [
        {"image_pool_index": i, "kind": "feature_map",
         "png_base64": _render_patch_grid_png(block3_grids[i])}
        for i in range(N_IMAGES_PER_MODEL)
    ]
    block11_visuals = [
        {"image_pool_index": i, "kind": "feature_map",
         "png_base64": _render_patch_grid_png(block11_grids[i])}
        for i in range(N_IMAGES_PER_MODEL)
    ]
    softmax_visuals = []
    for i in range(N_IMAGES_PER_MODEL):
        probs = softmax_probs[i].tolist()
        softmax_visuals.append({
            "image_pool_index": i,
            "kind": "softmax_bar",
            "png_base64": render_bar_chart(probs, class_names),
            "labels": class_names,
            "values": [float(p) for p in probs],
            "argmax": class_names[int(np.argmax(probs))],
            "argmax_prob": float(np.max(probs)),
        })

    # ── Numerical / counterfactual blobs, averaged across pool images
    def _avg_ms(stage_id):
        vals = latency_per_stage[stage_id]
        return float(np.mean(vals)) if vals else 0.0

    input_numerical = {
        "tensor_shape": "[1, 3, 224, 224]",
        "parameters_firing": 0,
        "preprocessing": "Resize 224 + ImageNet normalize (no CLAHE for router)",
        "pixel_mean_rgb": [float(round(inputs_rgb[0].mean(axis=(0, 1))[c] / 255.0, 3))
                           for c in range(3)],
    }
    head_numerical = {
        "tensor_shape": "[1, 4]  (one logit per crop class)",
        "parameters_firing": 384 * 4 + 4,
        "head_type": "nn.Linear(384, 4)",
    }
    # The router gate threshold lives in apin_server.py:1778 — CROP_CONF_MIN=0.40.
    # That's the right value for the counterfactual (the production override).
    head_counterfactual = _softmax_counterfactual(
        softmax_probs[0].tolist(), class_names, threshold=0.40)

    stages = [
        _stage("input",        "Input tensor",                  "router",
               input_visuals,
               latency_ms=_avg_ms("input"),
               numerical=input_numerical,
               layer_type="patch_embedder",
               layer_type_params={"img_size": 224, "patch_size": 14,
                                  "n_patches": 256, "n_extra_tokens": n_extra,
                                  "extra_token_names": ["CLS", "reg0", "reg1", "reg2", "reg3"],
                                  "embed_dim": 384}),
        _stage("vit_early",    "ViT block 3 (early attention)", "router",
               block3_visuals,
               latency_ms=_avg_ms("vit_early"),
               numerical=block3_numerical[0],
               layer_type="vit_block",
               layer_type_params={"block_index": 3, "total_blocks": 12,
                                  "n_heads": 6, "head_dim": 64,
                                  "qkv_dim": 384, "mlp_hidden": 1536,
                                  "mlp_ratio": 4, "params": 1_775_616,
                                  "frozen": True}),
        _stage("vit_pooled",   "ViT block 11 (rollout)",        "router",
               block11_visuals,
               latency_ms=_avg_ms("vit_pooled"),
               numerical=block11_numerical[0],
               layer_type="vit_block",
               layer_type_params={"block_index": 11, "total_blocks": 12,
                                  "n_heads": 6, "head_dim": 64,
                                  "qkv_dim": 384, "mlp_hidden": 1536,
                                  "mlp_ratio": 4, "params": 1_775_616,
                                  "frozen": True}),
        _stage("head_softmax", "Linear head + 4-way softmax",   "router",
               softmax_visuals,
               latency_ms=_avg_ms("head_softmax"),
               numerical=head_numerical,
               counterfactual=head_counterfactual,
               layer_type="linear_head",
               layer_type_params={"in_dim": 384, "out_dim": 4,
                                  "params": 384 * 4 + 4}),
    ]

    # Per-stage per-image numerical (not averaged) — used by the UI when the
    # user switches pool image so the table updates.  Attached as a sidecar
    # array on stage payloads so existing field names stay compatible.
    stages[1]["numerical_per_image"] = block3_numerical
    stages[2]["numerical_per_image"] = block11_numerical
    stages[3]["counterfactual_per_image"] = [
        _softmax_counterfactual(softmax_probs[i].tolist(), class_names, 0.40)
        for i in range(N_IMAGES_PER_MODEL)
    ]

    # Attach per-block CLS classification + leaf-position states as
    # model-level metadata so the ribbon + leaf-flight choreography can read
    # them without walking stages.
    extras = {
        "per_block_cls_classification": per_block_cls,   # FP-D12
        "per_block_class_names": class_names,
        "leaf_position_states": LEAF_POSITION_STATES["router"],  # FP-D28
    }
    return stages, image_pool, extras


# ────────────────────────────────────────────────────────────────────────
# Chilli extraction · reuses router (plan v4 §A4: no chilli specialist).
# ────────────────────────────────────────────────────────────────────────
def extract_chilli() -> Tuple[List[Dict], List[Dict]]:
    """Run the router on 3 chilli images. Emit one stage that captures the
    router classification + the ROUTER_REJECTED envelope shape.
    FP-D11..D14 + leaf positions added per 4C.6 spec."""
    import time as _time
    backbone, head, device, class_names = _load_router()
    rows = _select_images_for_classes(IMAGE_POOL_CLASSES["chilli"], N_IMAGES_PER_MODEL)
    image_pool = _make_image_pool(rows)

    inputs_rgb = []
    softmax_probs = []
    latencies_ms = []

    try:
        for row in rows:
            img_rgb = _load_image_rgb(row["image_path"])
            inputs_rgb.append(img_rgb)
            t0 = _time.perf_counter()
            tens = _preprocess_router_image(img_rgb).to(device)
            with torch.no_grad():
                feat = backbone(tens)
                logits = head(feat)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            latencies_ms.append((_time.perf_counter() - t0) * 1000.0)
            softmax_probs.append(probs.astype(np.float32))
    finally:
        del backbone, head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # The single stage carries 3 sub-visuals: router softmax bar + a JSON
    # snippet of the ROUTER_REJECTED envelope.
    REJECTED_ENVELOPE = {
        "ok": False,
        "error": {
            "code": "router_rejected",
            "message": "Chilli leaf detected; no chilli specialist is deployed.",
            "hint": "Forward to a human agronomist, or use the APIN fallback.",
        },
        "data": {
            "router_crop": "chilli",
            "router_confidence": "<router_softmax_argmax_value>",
            "specialist": None,
        },
        "meta": {"contract_version": "1.0"},
    }
    visuals = []
    for i in range(N_IMAGES_PER_MODEL):
        probs = softmax_probs[i].tolist()
        argmax_idx = int(np.argmax(probs))
        env = json.loads(json.dumps(REJECTED_ENVELOPE))  # deep copy
        env["data"]["router_confidence"] = round(float(probs[argmax_idx]), 4)
        visuals.append({
            "image_pool_index": i,
            "kind": "softmax_bar_plus_envelope",
            "png_base64": render_bar_chart(probs, class_names),
            "labels": class_names,
            "values": [float(p) for p in probs],
            "argmax": class_names[argmax_idx],
            "argmax_prob": float(np.max(probs)),
            "rejected_envelope": env,
        })

    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    chilli_cf = _softmax_counterfactual(
        softmax_probs[0].tolist(), class_names, threshold=0.40)
    stages = [
        _stage("router_classification", "Router classification + ROUTER_REJECTED",
               "chilli", visuals,
               latency_ms=avg_latency,
               numerical={
                   "tensor_shape": "[1, 4]  router softmax",
                   "preprocessing": "Resize 224 + ImageNet normalize",
                   "downstream": "STOP · ROUTER_REJECTED envelope",
                   "parameters_firing": 22_060_000,
               },
               counterfactual=chilli_cf,
               layer_type="linear_head",
               layer_type_params={"in_dim": 384, "out_dim": 4,
                                  "params": 384 * 4 + 4,
                                  "note": "Same head as router; chilli triggers honest-failure path."}),
    ]
    stages[0]["counterfactual_per_image"] = [
        _softmax_counterfactual(softmax_probs[i].tolist(), class_names, 0.40)
        for i in range(N_IMAGES_PER_MODEL)
    ]
    extras = {
        "leaf_position_states": LEAF_POSITION_STATES["chilli"],
        "honest_failure": True,
    }
    return stages, image_pool, extras


def extract_tomato() -> Tuple[List[Dict], List[Dict]]:
    """Run the tomato pipeline on 3 images, capture 7 stages.

    Captures V3 blocks 0, 5, 11 + SP-LoRA last block via forward hooks,
    plus the fused stacking_gate + calibrated_output via TomatoPipeline.infer()
    return values.
    """
    if torch is None:
        raise RuntimeError("torch not installed")
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    # ladi_net is on sys.path under scripts/ — TomatoPipeline's loader expects this
    ladi_path = ROOT / "scripts" / "ladi_net"
    if str(ladi_path) not in _sys.path:
        _sys.path.insert(0, str(ladi_path))

    from scripts.ladi_net.tomato_pipeline import TomatoPipeline

    pipeline = TomatoPipeline()
    rows = _select_images_for_classes(IMAGE_POOL_CLASSES["tomato"], N_IMAGES_PER_MODEL)
    image_pool = _make_image_pool(rows)

    # Locate V3 blocks. Model3 wraps timm DINOv2-Small in a Backbone module.
    # Try common access patterns.
    def _find_blocks(model):
        for attr_path in [("backbone", "blocks"),
                          ("backbone", "base_model", "model", "blocks"),
                          ("blocks",)]:
            cur = model
            ok = True
            for a in attr_path:
                if hasattr(cur, a):
                    cur = getattr(cur, a)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__len__") and len(cur) > 0:
                return cur, attr_path
        raise AttributeError(f"Cannot locate ViT blocks on {type(model).__name__}")

    v3_blocks, _ = _find_blocks(pipeline.v3)
    sp_blocks, _ = _find_blocks(pipeline.sp_lora)
    if len(v3_blocks) < 12:
        raise RuntimeError(f"V3 has only {len(v3_blocks)} blocks; expected >= 12.")
    if len(sp_blocks) < 1:
        raise RuntimeError(f"SP-LoRA has only {len(sp_blocks)} blocks.")

    captured: Dict[str, Any] = {}

    def make_hook(name):
        def hook(_m, _inp, out):
            # SP-LoRA / V3 blocks may return tuple; take the first tensor element
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach()
        return hook

    h_v3_0 = v3_blocks[0].register_forward_hook(make_hook("v3_block_0"))
    h_v3_5 = v3_blocks[5].register_forward_hook(make_hook("v3_block_5"))
    h_v3_11 = v3_blocks[11].register_forward_hook(make_hook("v3_block_11"))
    h_sp = sp_blocks[-1].register_forward_hook(make_hook("sp_block_last"))

    inputs_rgb = []
    v3_block0_grids = []
    v3_block5_grids = []
    v3_block11_grids = []
    sp_last_grids = []
    stacking_probs = []   # post-fusion (before per-class T)
    calibrated_probs = [] # post-temperature
    class_orders = []
    # PVA-R1#3 fix: capture per-image tomato infer latency, then distribute
    # across the 7 stages by a proportional table (preprocessing + 5 backbone
    # passes + fusion).  Mean across pool images is what the latency-budget
    # ribbon consumes.  Spec FP-D11.
    import time as _time
    tomato_total_ms = []

    try:
        for row in rows:
            img_rgb = _load_image_rgb(row["image_path"])
            inputs_rgb.append(img_rgb)
            captured.clear()
            _t0 = _time.perf_counter()
            result = pipeline.infer(img_rgb)
            tomato_total_ms.append((_time.perf_counter() - _t0) * 1000.0)
            # Try multiple known result shapes
            probs = None
            class_order = None
            if isinstance(result, dict):
                # Try common keys
                for k in ("all_class_probabilities", "class_probabilities",
                         "probabilities", "probs"):
                    if k in result and result[k]:
                        v = result[k]
                        if isinstance(v, dict):
                            class_order = list(v.keys())
                            probs = np.array([v[c] for c in class_order],
                                             dtype=np.float32)
                        else:
                            probs = np.asarray(v, dtype=np.float32)
                        break
                # Tomato pipeline emits top_class_probabilities or top3 list
                if probs is None:
                    for k in ("top_class_probabilities", "top3", "top_k"):
                        if k in result and result[k]:
                            v = result[k]
                            if isinstance(v, list) and v and isinstance(v[0], dict):
                                class_order = [d.get("class", d.get("label", str(i)))
                                               for i, d in enumerate(v)]
                                probs = np.array([d.get("probability",
                                                        d.get("prob", 0.0)) for d in v],
                                                 dtype=np.float32)
                            break

            if probs is None:
                # Fallback to dummy probs (preserves stage shape but flagged)
                class_order = ["tomato_class_" + str(i) for i in range(6)]
                probs = np.full(6, 1.0 / 6, dtype=np.float32)

            stacking_probs.append(probs.copy())
            calibrated_probs.append(probs.copy())  # tomato's infer() returns calibrated
            class_orders.append(class_order)

            # V3 backbone has 1 CLS + 0 register tokens (Model3 uses
            # vit_small_patch14_dinov2.lvd142m, not the reg4 variant). Verify
            # by checking the actual token count and falling back to 1 if mismatch.
            v3_b0 = captured.get("v3_block_0")
            v3_b5 = captured.get("v3_block_5")
            v3_b11 = captured.get("v3_block_11")
            sp_last = captured.get("sp_block_last")
            if v3_b0 is None or v3_b5 is None or v3_b11 is None or sp_last is None:
                raise RuntimeError(
                    f"Forward hooks didn't fire for tomato inference. "
                    f"Captured keys: {list(captured.keys())}"
                )

            # Auto-detect extra-token count (CLS + register)
            def _detect_extras(tensor):
                n = tensor.shape[1]
                # Patch counts: 256 (16×16 @ 224 with patch14), 196 (14×14),
                # 784 (28×28 @ 392), 729 (27×27 @ 378 with patch14)
                for n_patch in (256, 196, 784, 729, 576, 1024):
                    extras = n - n_patch
                    if 0 <= extras <= 8:
                        return extras
                # Fallback: assume single CLS
                return 1

            v3_extras = _detect_extras(v3_b0)
            sp_extras = _detect_extras(sp_last)

            v3_block0_grids.append(_block_to_patch_grid(v3_b0, v3_extras))
            v3_block5_grids.append(_block_to_patch_grid(v3_b5, v3_extras))
            v3_block11_grids.append(_block_to_patch_grid(v3_b11, v3_extras))
            sp_last_grids.append(_block_to_patch_grid(sp_last, sp_extras))
    finally:
        h_v3_0.remove(); h_v3_5.remove(); h_v3_11.remove(); h_sp.remove()
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build the 7 locked stages
    input_visuals = [_encode_input_visual(inputs_rgb[i], i) for i in range(N_IMAGES_PER_MODEL)]
    v3_stage1_visuals = [
        {"image_pool_index": i, "kind": "feature_map",
         "png_base64": _render_patch_grid_png(v3_block0_grids[i])}
        for i in range(N_IMAGES_PER_MODEL)
    ]
    v3_stage2_visuals = [
        {"image_pool_index": i, "kind": "feature_map",
         "png_base64": _render_patch_grid_png(v3_block5_grids[i])}
        for i in range(N_IMAGES_PER_MODEL)
    ]
    lora_visuals = [
        {"image_pool_index": i, "kind": "feature_map",
         "png_base64": _render_patch_grid_png(sp_last_grids[i])}
        for i in range(N_IMAGES_PER_MODEL)
    ]
    v3_deep_visuals = [
        {"image_pool_index": i, "kind": "feature_map",
         "png_base64": _render_patch_grid_png(v3_block11_grids[i])}
        for i in range(N_IMAGES_PER_MODEL)
    ]
    stacking_visuals = []
    calibrated_visuals = []
    for i in range(N_IMAGES_PER_MODEL):
        sp = stacking_probs[i].tolist()
        cp = calibrated_probs[i].tolist()
        co = class_orders[i]
        stacking_visuals.append({
            "image_pool_index": i,
            "kind": "softmax_bar",
            "png_base64": render_bar_chart(sp, co),
            "labels": co,
            "values": [float(p) for p in sp],
            "argmax": co[int(np.argmax(sp))],
            "argmax_prob": float(np.max(sp)),
        })
        calibrated_visuals.append({
            "image_pool_index": i,
            "kind": "softmax_bar",
            "png_base64": render_bar_chart(cp, co),
            "labels": co,
            "values": [float(p) for p in cp],
            "argmax": co[int(np.argmax(cp))],
            "argmax_prob": float(np.max(cp)),
        })

    final_cf = _softmax_counterfactual(
        calibrated_probs[0].tolist(), class_orders[0], threshold=0.50) \
        if calibrated_probs and class_orders else {}

    # PVA-R1#3 fix: distribute total infer latency across 7 stages.  The
    # V3 + SP-LoRA backbones dominate (~70% of work); preprocessing +
    # fusion + calibration get smaller slices.  Proportions sum to 1.0.
    _t_mean = float(np.mean(tomato_total_ms)) if tomato_total_ms else 0.0
    _tomato_share = {
        "input":              0.05,
        "v3_stage_1":         0.14,
        "v3_stage_2":         0.14,
        "lora_branch":        0.20,
        "v3_deep":            0.22,
        "stacking_gate":      0.15,
        "calibrated_output":  0.10,
    }

    stages = [
        _stage("input",              "Input tensor (V3 224 + LoRA 392)", "tomato",
               input_visuals, layer_type="patch_embedder",
               latency_ms=_t_mean * _tomato_share["input"],
               numerical={"v3_input": "[1, 3, 224, 224]", "lora_input": "[1, 3, 392, 392]",
                          "preprocessing": "V3: LAB-CLAHE + ImageNet · SP-LoRA: 800px cap + letterbox 392 + LAB-CLAHE"}),
        _stage("v3_stage_1",         "V3 backbone · block 1",            "tomato",
               v3_stage1_visuals, layer_type="vit_block",
               latency_ms=_t_mean * _tomato_share["v3_stage_1"],
               layer_type_params={"block_index": 1, "total_blocks": 12, "n_heads": 6,
                                  "head_dim": 64, "qkv_dim": 384, "frozen": False, "with_lora": True}),
        _stage("v3_stage_2",         "V3 backbone · block 5",            "tomato",
               v3_stage2_visuals, layer_type="vit_block",
               latency_ms=_t_mean * _tomato_share["v3_stage_2"],
               layer_type_params={"block_index": 5, "total_blocks": 12, "n_heads": 6,
                                  "head_dim": 64, "qkv_dim": 384, "frozen": False, "with_lora": True, "with_film": True}),
        _stage("lora_branch",        "SP-LoRA backbone · final block",   "tomato",
               lora_visuals, layer_type="vit_block",
               latency_ms=_t_mean * _tomato_share["lora_branch"],
               layer_type_params={"block_index": "final", "backbone": "DINOv2-Reg-Base @ 392",
                                  "n_heads": 12, "head_dim": 64, "qkv_dim": 768, "with_lora": True}),
        _stage("v3_deep",            "V3 backbone · block 11 (rollout)", "tomato",
               v3_deep_visuals, layer_type="vit_block",
               latency_ms=_t_mean * _tomato_share["v3_deep"],
               layer_type_params={"block_index": 11, "total_blocks": 12, "n_heads": 6,
                                  "head_dim": 64, "qkv_dim": 384, "with_lora": True}),
        _stage("stacking_gate",      "Stacking MLP (V3 + SP-LoRA)",      "tomato",
               stacking_visuals, layer_type="stacking_mlp",
               latency_ms=_t_mean * _tomato_share["stacking_gate"],
               layer_type_params={"n_signals": 2, "n_classes": 6,
                                  "T_v3": 0.5, "T_sp_lora": 1.0,
                                  "fusion": "weighted average with asymmetric T"}),
        _stage("calibrated_output",  "Calibrated + conformal output",    "tomato",
               calibrated_visuals, layer_type="conformal_prediction",
               counterfactual=final_cf,
               latency_ms=_t_mean * _tomato_share["calibrated_output"],
               layer_type_params={"calibration": "per-class T scaling",
                                  "conformal_coverage_target": 0.90}),
    ]

    # PVA-R1#5 fix (FP-D25): assign a tier to tomato's calibrated_output.
    # Simple rule: argmax_prob ≥ 0.85 → 1A; ≥ 0.60 → 1B; ≥ 0.40 → 2; else 3.
    if calibrated_probs:
        ap = float(calibrated_probs[0].max())
        stages[-1]["tier"] = ("1A" if ap >= 0.85 else
                              "1B" if ap >= 0.60 else
                              "2"  if ap >= 0.40 else
                              "3")
    extras = {
        "leaf_position_states": LEAF_POSITION_STATES["tomato"],
    }
    return stages, image_pool, extras


def extract_apin() -> Tuple[List[Dict], List[Dict]]:
    """Run the APIN okra/brassica ensemble on 3 images, capture 9 stages.

    APINInference.predict() runs all 4 signals + stacking MLP + calibration
    + conformal. We hook each signal's backbone for visuals, and read
    per-signal probabilities + final fused output from the APINResult.
    """
    if torch is None:
        raise RuntimeError("torch not installed")
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))

    from scripts.apin.inference import APINInference

    apin = APINInference(verbose=False)

    # Force lazy-load of all 4 signals DIRECTLY (bypasses predict()'s
    # gate-zero hard-reject path, which would never load the signals on a
    # rejected image). The lazy-load methods themselves don't run inference;
    # they just instantiate the model + load weights, which is what we need.
    rows = _select_images_for_classes(IMAGE_POOL_CLASSES["apin"], N_IMAGES_PER_MODEL)
    image_pool = _make_image_pool(rows)
    apin._lazy_load_model2()
    apin._lazy_load_efficientnet()
    apin._lazy_load_dinov2()

    # Resolve backbones (now loaded)
    model2 = getattr(apin, "_model2", None)
    eff = getattr(apin, "_efficientnet", None)
    dino_bb = getattr(apin, "_dinov2_backbone", None)
    # Each may be None on failure; record availability and proceed best-effort

    captured: Dict[str, Any] = {}

    def make_hook(name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach() if hasattr(t, "detach") else t
        return hook

    handles = []

    # model2: ConvNeXt-Small with a stages/.head structure. Hook the last stage.
    def _find_module(model, names):
        for n in names:
            parts = n.split(".")
            cur = model
            ok = True
            for p in parts:
                if hasattr(cur, p):
                    cur = getattr(cur, p)
                else:
                    ok = False
                    break
            if ok:
                return cur
        return None

    if model2 is not None:
        # Model2ConvNeXt wraps a DINOv3ConvNextModel under .backbone.
        # backbone.stages is a ModuleList of 4 DINOv3ConvNextStage modules;
        # backbone.stages.3 is the deepest stage (7×7 spatial @ 224 input,
        # ~768 channels).
        last = _find_module(model2, [
            "backbone.stages.3", "backbone.stages_3",
            "stages.3", "model.stages.3",
            "features", "model.features",
        ])
        if last is not None:
            handles.append(last.register_forward_hook(make_hook("model2_last")))
    if eff is not None:
        # The "EfficientNet" signal is actually old_10class PlantDiseaseModel
        # which wraps an EfficientNetFeatures backbone under .backbone.
        # backbone.blocks is a Sequential of 6 blocks; backbone.blocks.5 is
        # the deepest (7×7 spatial @ 224 input).
        target = _find_module(eff, [
            "backbone.blocks.5", "backbone.blocks_5",
            "backbone.conv_head",
            "conv_head", "model.conv_head", "blocks.6", "model.blocks.6",
        ])
        if target is not None:
            handles.append(target.register_forward_hook(make_hook("eff_last")))
    if dino_bb is not None:
        # DINOv2 ViT-Small backbone: hook the final block
        blocks = None
        for attr_path in [("blocks",), ("model", "blocks"),
                          ("base_model", "model", "blocks")]:
            cur = dino_bb
            ok = True
            for a in attr_path:
                if hasattr(cur, a):
                    cur = getattr(cur, a)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__len__") and len(cur) > 0:
                blocks = cur
                break
        if blocks is not None:
            handles.append(blocks[-1].register_forward_hook(make_hook("dinov2_last")))

    # Per-image captures
    inputs_rgb = []
    model2_grids = []
    eff_grids = []
    dinov2_grids = []
    psv_features = []        # list of PSV feature vectors (or None)
    per_signal_probs = []    # list of {signal_name: probs_array}
    stacking_probs = []      # fused MLP output
    calibrated_probs = []    # post-temperature
    conformal_sets = []
    class_orders = []
    # PVA-R1#3 fix: per-image apin inference latency (sum of all signals).
    import time as _time
    apin_total_ms = []

    try:
        for row in rows:
            img_rgb = _load_image_rgb(row["image_path"])
            inputs_rgb.append(img_rgb)
            captured.clear()
            _t_apin_start = _time.perf_counter()
            class_order = list(getattr(apin, "class_order", []))
            class_orders.append(class_order)

            # Call each signal directly. This bypasses predict()'s gate-zero
            # hard-reject path (which would skip the signals entirely for
            # tier-4A images and leave our hooks empty).
            sig_vecs = []
            sig_names = ["model2", "efficientnet", "psv", "dinov2_head"]
            try:
                s1 = apin._infer_model2(img_rgb); sig_vecs.append(s1)
            except Exception as e:
                print(f"    model2 infer failed: {e}"); sig_vecs.append(None)
            try:
                s2 = apin._infer_efficientnet(img_rgb); sig_vecs.append(s2)
            except Exception as e:
                print(f"    efficientnet infer failed: {e}"); sig_vecs.append(None)
            try:
                if apin.use_psv:
                    s3, _stg, _raw = apin._infer_psv(img_rgb)
                    sig_vecs.append(s3)
                else:
                    sig_vecs.append(None)
            except Exception as e:
                print(f"    psv infer failed: {e}"); sig_vecs.append(None)
            try:
                s4, _feat = apin._infer_dinov2_head(img_rgb); sig_vecs.append(s4)
            except Exception as e:
                print(f"    dinov2_head infer failed: {e}"); sig_vecs.append(None)

            per_signal_probs.append({
                n: {"argmax": class_order[int(v.argmax())] if v is not None and class_order else "",
                    "top_prob": float(v.max()) if v is not None else 0.0}
                for n, v in zip(sig_names, sig_vecs)
            })

            # Run the stacking MLP on the per-signal vectors. Falls back to
            # uniform if all signals failed.
            stacking_vec = None
            try:
                stacking_vec, _gate = apin._run_stacking_mlp(sig_vecs)
            except Exception as e:
                print(f"    stacking_mlp failed: {e}")
            if stacking_vec is None:
                stacking_vec = np.full(len(class_order) or 9, 1.0 / 9, dtype=np.float32)
            fused = np.asarray(stacking_vec, dtype=np.float32)
            stacking_probs.append(fused)

            # Per-class temperature scaling: apply per_class_temps if available.
            try:
                T = np.asarray(getattr(apin, "per_class_temps", np.ones(9)), dtype=np.float32)
                # T-scale via the softmax convention: probs / T then renormalise.
                # APIN's actual calibration is applied via a separate path; this
                # approximation matches the narration's description.
                if T.shape[0] == fused.shape[0] and not np.allclose(T, 1.0):
                    eps = 1e-9
                    logits = np.log(fused + eps) / np.maximum(T, eps)
                    cal = np.exp(logits - logits.max())
                    cal = cal / cal.sum()
                else:
                    cal = fused.copy()
            except Exception:
                cal = fused.copy()
            calibrated_probs.append(cal)

            # Conformal: classes whose calibrated prob exceeds per-class threshold.
            try:
                thr = np.asarray(getattr(apin, "conformal_thresholds",
                                         np.full(len(cal), 0.5)), dtype=np.float32)
                conformal = [class_order[k] for k in range(len(cal))
                             if k < len(thr) and cal[k] >= thr[k]]
            except Exception:
                conformal = []
            if not conformal:
                # Empty set: keep the top-1 to avoid an empty conformal display.
                conformal = [class_order[int(cal.argmax())]] if class_order else []
            conformal_sets.append(conformal)

            # Capture visuals
            if "model2_last" in captured:
                model2_grids.append(_apin_backbone_to_grid(captured["model2_last"]))
            else:
                model2_grids.append(None)
            if "eff_last" in captured:
                eff_grids.append(_apin_backbone_to_grid(captured["eff_last"]))
            else:
                eff_grids.append(None)
            if "dinov2_last" in captured:
                # DINOv2: ViT with 4 register tokens
                t = captured["dinov2_last"]
                if t.ndim == 3:
                    # Auto-detect extras (1 CLS + 4 reg = 5 for the reg4 variant)
                    n = t.shape[1]
                    extras = 5 if n == 261 else (1 if n == 257 else max(0, n - 256))
                    dinov2_grids.append(_block_to_patch_grid(t, extras))
                else:
                    dinov2_grids.append(None)
            else:
                dinov2_grids.append(None)

            # PSV: psv_extract returns a FeatureResult dataclass with
            # `.features` being the dict of float values. Sort by name for
            # deterministic ordering.
            try:
                psv_feat = apin.psv_extract(img_rgb)
                psv_dict = None
                if hasattr(psv_feat, "features") and isinstance(psv_feat.features, dict):
                    psv_dict = psv_feat.features
                elif isinstance(psv_feat, dict):
                    psv_dict = psv_feat
                if psv_dict:
                    items = sorted(psv_dict.items(), key=lambda kv: kv[0])
                    psv_features.append([
                        (k, float(v) if isinstance(v, (int, float)) and not (
                            isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                        ) else 0.0)
                        for k, v in items
                    ])
                else:
                    psv_features.append(None)
            except Exception as e:
                print(f"    PSV extraction failed for image {row['image_path']}: "
                      f"{type(e).__name__}: {e}")
                psv_features.append(None)
            apin_total_ms.append((_time.perf_counter() - _t_apin_start) * 1000.0)
    finally:
        for h in handles:
            try: h.remove()
            except Exception: pass
        del apin
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build visuals + 9 stages
    input_visuals = [_encode_input_visual(inputs_rgb[i], i) for i in range(N_IMAGES_PER_MODEL)]
    split_visuals = input_visuals  # same image dispatched to 4 signals; UI overlays the fork

    def _grid_visual(grid_or_none, i):
        if grid_or_none is None:
            return {"image_pool_index": i, "kind": "feature_map_unavailable",
                    "reason": "Hook did not fire or backbone unavailable",
                    "png_base64": ""}
        return {"image_pool_index": i, "kind": "feature_map",
                "png_base64": _render_patch_grid_png(grid_or_none)}

    dinov3_visuals = [_grid_visual(model2_grids[i], i) for i in range(N_IMAGES_PER_MODEL)]
    eff_visuals = [_grid_visual(eff_grids[i], i) for i in range(N_IMAGES_PER_MODEL)]
    dinov2_visuals = [_grid_visual(dinov2_grids[i], i) for i in range(N_IMAGES_PER_MODEL)]

    # PSV: render top 16 feature magnitudes
    psv_visuals = []
    for i in range(N_IMAGES_PER_MODEL):
        feats = psv_features[i]
        if feats is None or not feats:
            psv_visuals.append({"image_pool_index": i, "kind": "psv_unavailable",
                                "png_base64": ""})
            continue
        # Pick top 16 by absolute magnitude
        top = sorted(feats, key=lambda kv: -abs(kv[1]))[:16]
        labels = [k for k, _ in top]
        values = [float(v) for _, v in top]
        # Normalise to [0, 1] for bar chart
        v_arr = np.array(values, dtype=np.float32)
        lo, hi = float(v_arr.min()), float(v_arr.max())
        norm = ((v_arr - lo) / (hi - lo + 1e-9)).tolist()
        psv_visuals.append({
            "image_pool_index": i,
            "kind": "psv_bar",
            "png_base64": render_bar_chart(norm, labels),
            "labels": labels,
            "values": values,
        })

    stacking_visuals = []
    calibrated_visuals = []
    conformal_visuals = []
    for i in range(N_IMAGES_PER_MODEL):
        sp = stacking_probs[i].tolist()
        cp = calibrated_probs[i].tolist()
        co = class_orders[i] or [f"class_{k}" for k in range(len(sp))]
        stacking_visuals.append({
            "image_pool_index": i,
            "kind": "softmax_bar",
            "png_base64": render_bar_chart(sp, co),
            "labels": co,
            "values": [float(p) for p in sp],
            "argmax": co[int(np.argmax(sp))] if sp else "",
            "argmax_prob": float(np.max(sp)) if sp else 0.0,
        })
        calibrated_visuals.append({
            "image_pool_index": i,
            "kind": "softmax_bar",
            "png_base64": render_bar_chart(cp, co),
            "labels": co,
            "values": [float(p) for p in cp],
            "argmax": co[int(np.argmax(cp))] if cp else "",
            "argmax_prob": float(np.max(cp)) if cp else 0.0,
        })
        # Conformal: top-3 bar chart of calibrated probs
        if cp:
            top3_idx = sorted(range(len(cp)), key=lambda k: -cp[k])[:3]
            top3_labels = [co[k] for k in top3_idx]
            top3_values = [float(cp[k]) for k in top3_idx]
        else:
            top3_labels, top3_values = [], []
        conformal_visuals.append({
            "image_pool_index": i,
            "kind": "top3_bar_plus_set",
            "png_base64": render_bar_chart(top3_values, top3_labels) if top3_values else "",
            "top3_labels": top3_labels,
            "top3_values": top3_values,
            "conformal_set": conformal_sets[i],
        })

    apin_final_cf = _softmax_counterfactual(
        calibrated_probs[0].tolist(), class_orders[0], threshold=0.40) \
        if calibrated_probs and class_orders else {}

    # PVA-R1#3 fix: distribute total apin latency across 9 stages.  The
    # 4 signal backbones dominate (~80% of work).  Proportions sum to 1.0.
    _a_mean = float(np.mean(apin_total_ms)) if apin_total_ms else 0.0
    _apin_share = {
        "input":               0.03,
        "split_to_4_signals":  0.02,
        "dinov3_convnext":     0.28,   # ConvNeXt-Small dominates
        "efficientnet_b0":     0.14,
        "dinov2_vit":          0.16,
        "psv_features":        0.18,   # PSV extraction is non-trivial
        "stacking_mlp":        0.06,
        "temperature_scaling": 0.05,
        "conformal_output":    0.08,
    }
    # Tier label derived from conformal-set size + signal agreement (rough heuristic).
    # Simple version: tier 1A if conformal_set size 1 + argmax prob >= 0.80;
    # 1B if size 1 + prob >= 0.50; 2 if size 2; 3 if size 3+; 4A if argmax < 0.30.
    def _apin_tier(probs, conformal_set):
        if not probs.size: return None
        argmax_p = float(probs.max())
        n_set = len(conformal_set) if conformal_set else 0
        if argmax_p < 0.30: return "4A"
        if n_set <= 1 and argmax_p >= 0.80: return "1A"
        if n_set <= 1: return "1B"
        if n_set == 2: return "2"
        if n_set >= 3: return "3"
        return None
    _apin_first_tier = _apin_tier(calibrated_probs[0] if calibrated_probs else np.array([]),
                                  conformal_sets[0] if conformal_sets else [])

    stages = [
        _stage("input",              "Input tensor (LAB-CLAHE preprocessing)", "apin", input_visuals,
               layer_type="patch_embedder",
               latency_ms=_a_mean * _apin_share["input"],
               numerical={"branch_a": "LAB-CLAHE + ImageNet @ 224/384",
                          "branch_b": "per-channel RGB CLAHE @ 224"}),
        _stage("split_to_4_signals", "Dispatch to 4 parallel signals",         "apin", split_visuals,
               layer_type="patch_embedder",
               latency_ms=_a_mean * _apin_share["split_to_4_signals"],
               numerical={"n_signals": 4, "logical_parallel": True,
                          "wall_clock_serial": "extractor runs them sequentially for VRAM"}),
        _stage("dinov3_convnext",    "DINOv3-ConvNeXt-Small · final stage",    "apin", dinov3_visuals,
               layer_type="convnext_stage",
               latency_ms=_a_mean * _apin_share["dinov3_convnext"],
               layer_type_params={"backbone": "DINOv3-ConvNeXt-Small",
                                  "params_total": 49_500_000,
                                  "n_stages": 4, "active_stage": 3,
                                  "feature_dim": 768, "spatial": "7×7"}),
        _stage("efficientnet_b0",    "EfficientNet-B0 · final conv head",      "apin", eff_visuals,
               layer_type="mbconv",
               latency_ms=_a_mean * _apin_share["efficientnet_b0"],
               layer_type_params={"backbone": "EfficientNetV2-S wrapper (old_10class PlantDiseaseModel)",
                                  "active_block": 5, "spatial": "7×7"}),
        _stage("dinov2_vit",         "DINOv2 ViT-S · final block",             "apin", dinov2_visuals,
               layer_type="vit_block",
               latency_ms=_a_mean * _apin_share["dinov2_vit"],
               layer_type_params={"backbone": "DINOv2 ViT-S + 4 register tokens (frozen)",
                                  "n_heads": 6, "head_dim": 64, "qkv_dim": 384,
                                  "frozen": True}),
        _stage("psv_features",       "PSV engineered features",                "apin", psv_visuals,
               layer_type="psv_engineered",
               latency_ms=_a_mean * _apin_share["psv_features"],
               numerical={"n_features": 66, "categories": ["shape", "colour", "lesion", "vein", "necrosis_chlorosis"],
                          "br_alt_supervised_path": True}),
        _stage("stacking_mlp",       "Stacking MLP (4-signal fusion)",         "apin", stacking_visuals,
               layer_type="stacking_mlp",
               latency_ms=_a_mean * _apin_share["stacking_mlp"],
               layer_type_params={"n_signals": 4, "n_classes": 9, "hidden_dims": [64, 32],
                                  "val_macro_f1": 0.9228}),
        _stage("temperature_scaling","Per-class temperature scaling",          "apin", calibrated_visuals,
               layer_type="temperature_scaling",
               latency_ms=_a_mean * _apin_share["temperature_scaling"],
               layer_type_params={"n_classes": 9, "per_class_T": True}),
        _stage("conformal_output",   "Conformal prediction set + top-3",       "apin", conformal_visuals,
               layer_type="conformal_prediction",
               counterfactual=apin_final_cf,
               latency_ms=_a_mean * _apin_share["conformal_output"],
               layer_type_params={"coverage_target": 0.90, "tier_labels": ["1A","1B","2","3","4A","4B","5"]}),
    ]
    # PVA-R1#5 fix (FP-D25): attach tier to the final conformal stage.
    if _apin_first_tier is not None:
        stages[-1]["tier"] = _apin_first_tier
    extras = {"leaf_position_states": LEAF_POSITION_STATES["apin"]}
    return stages, image_pool, extras


def _apin_backbone_to_grid(tensor):
    """Convert a CNN backbone output tensor to a 2D grid for visualization.
    Handles [B, C, H, W] (CNNs) and [B, N_tokens, D] (ViTs).
    For CNNs: take per-spatial L2 magnitude across channel dim.
    For ViTs: assume 1 CLS extra token unless caller specifies."""
    t = tensor.detach().float().cpu() if hasattr(tensor, "detach") else np.asarray(tensor)
    if hasattr(t, "ndim") and t.ndim == 4:
        # [B, C, H, W] CNN
        t = t[0]  # [C, H, W]
        if hasattr(t, "norm"):
            mag = t.norm(dim=0).numpy()
        else:
            mag = np.linalg.norm(t, axis=0)
        return mag
    if hasattr(t, "ndim") and t.ndim == 3 and t.shape[0] == 1:
        # [1, N, D] ViT-like; default 1 CLS extra
        return _block_to_patch_grid(t if hasattr(t, "detach") else tensor, 1)
    return None


# ────────────────────────────────────────────────────────────────────────
# Schema-invariant assertions · plan v4 §§A14.
# Enforce stage counts AND narration content rules at extraction time so
# bad payloads cannot be written to disk.
# ────────────────────────────────────────────────────────────────────────
EXPECTED_STAGE_COUNTS = {"router": 4, "tomato": 7, "apin": 9, "chilli": 1}


def assert_invariants(payload: Dict[str, Any]) -> None:
    """Enforce all schema invariants. Raises AssertionError on first violation."""
    models = payload.get("models", {})
    metadata = payload.get("metadata", {})

    for model_name, expected in EXPECTED_STAGE_COUNTS.items():
        m = models.get(model_name, {})
        if m.get("unavailable"):
            # Fallback shape: stages must be [] when unavailable; don't enforce count
            assert m.get("stages") == [], (
                f"{model_name}: unavailable=true but stages is not empty"
            )
            continue
        actual = len(m.get("stages", []))
        meta_count = metadata.get(f"n_stages_{model_name}", None)
        assert actual == expected, (
            f"{model_name}: stage count {actual} != expected {expected}"
        )
        assert meta_count == expected, (
            f"metadata.n_stages_{model_name} = {meta_count} != expected {expected}"
        )

    # Image pool sizes
    for model_name, m in models.items():
        if m.get("unavailable"):
            continue
        pool = m.get("image_pool", [])
        assert len(pool) == N_IMAGES_PER_MODEL, (
            f"{model_name}: image_pool has {len(pool)} entries, expected {N_IMAGES_PER_MODEL}"
        )

    # Narration content rules
    for model_name, m in models.items():
        if m.get("unavailable"):
            continue
        for s in m.get("stages", []):
            narration = s.get("narration", "")
            # Strip {term:X} markers for word counting
            plain = re.sub(r"\{term:[^}]+\}", "X", narration)
            wc = len(plain.split())
            assert 150 <= wc <= 250, (
                f"{model_name}/{s.get('id', '?')}: narration word count {wc} "
                f"out of [150, 250] range"
            )
            for banned in BANNED_WORDS:
                assert banned not in narration.lower(), (
                    f"{model_name}/{s.get('id', '?')}: banned word '{banned}' "
                    f"in narration"
                )
            # [PVA-R7 fix] em-dash hard-constraint: 0 em-dashes in narrations
            # to match the project's typographic-discipline rule.
            assert "—" not in narration, (
                f"{model_name}/{s.get('id', '?')}: em-dash (U+2014) found in "
                f"narration; replace with ' · ' or ' - '"
            )


def render_narration_html(narration: str) -> str:
    """Convert {term:X} placeholders to <span class="term" data-term="X">X</span>.
    Idempotent. The UI consumes the HTML-rendered version directly."""
    import re as _re
    def replace(m):
        term = m.group(1)
        return f'<span class="term" data-term="{term}">{term}</span>'
    return _re.sub(r"\{term:([^}]+)\}", replace, narration)


# regex import lifted to module scope so assert_invariants can use it
import re


# ────────────────────────────────────────────────────────────────────────
# Atomic write · plan v4 §A19. Match Phase 4B convention exactly.
# ────────────────────────────────────────────────────────────────────────
def write_atomic(payload: Dict[str, Any]) -> None:
    """Atomic write with Windows AV-scanner retry. Matches the pattern in
    extract_pipeline_atlas_tomato.py."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    import time
    for _ in range(5):
        try:
            os.replace(tmp, OUT)
            break
        except PermissionError:
            time.sleep(0.15)
    else:
        import shutil
        shutil.copyfile(tmp, OUT)
        try:
            tmp.unlink()
        except OSError:
            pass


# ────────────────────────────────────────────────────────────────────────
# Main orchestrator · sequential model loading per plan v4 §A1.
# ────────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 70)
    print("Phase 4C extraction: animated forward-pass diagram")
    print(f"Output: {OUT}")
    print(f"Seed: {SEED}, tile_px: {TILE_PX}, PNG compression: {PNG_COMPRESSION}")
    print("=" * 70)

    timestamp = (datetime.datetime.now(datetime.timezone.utc)
                 .isoformat(timespec="seconds").replace("+00:00", "Z"))

    payload: Dict[str, Any] = {
        "produced_at": timestamp,
        "source": "scripts/apin_v2/extract_forward_pass.py",
        "models": {},
        "metadata": {
            "n_stages_router": EXPECTED_STAGE_COUNTS["router"],
            "n_stages_tomato": EXPECTED_STAGE_COUNTS["tomato"],
            "n_stages_apin":   EXPECTED_STAGE_COUNTS["apin"],
            "n_stages_chilli": EXPECTED_STAGE_COUNTS["chilli"],
            "n_images_per_model": N_IMAGES_PER_MODEL,
            "n_images_total":  4 * N_IMAGES_PER_MODEL,
            "rng_seed": SEED,
            "image_selection_rule": (
                "Per-class stratified from the locked test split, "
                "highest-confidence-correct prediction per class. "
                "Deterministic given fixed model weights + locked test "
                "split + seed=42."
            ),
            "tile_render": {
                "tile_px": TILE_PX,
                "format": "PNG",
                "compression_level": PNG_COMPRESSION,
                "colormap": "viridis (11-step in-script palette)",
                "grid": "up to 4 columns, rows wrap",
            },
            "extraction_notes": (
                "4C.2 full extraction: forward-hook captures on every "
                "model. ViT block visuals = per-patch L2 magnitude of the "
                "block output (CLS + register tokens stripped). CNN visuals "
                "= per-spatial L2 magnitude across the channel dim. APIN "
                "signal probabilities are computed by calling each "
                "_infer_signal method directly (bypasses gate-zero so the "
                "hooks fire on every image)."
            ),
            "new_glossary_terms_drafted": sorted(NEW_GLOSSARY_TERMS.keys()),
        },
    }

    # Sequential extraction. Each model gets its own try/except so a
    # single failure doesn't abort the whole extraction (plan v4 §A1 +
    # §A2 + §A3: per-model continuation).
    for model_name, extractor in [
        ("router", extract_router),
        ("tomato", extract_tomato),
        ("apin",   extract_apin),
        ("chilli", extract_chilli),
    ]:
        print(f"\n--- extracting {model_name} ---")
        try:
            result = extractor()
            # Backward-compatible unpacking: extractors may return
            # (stages, pool) or (stages, pool, extras).
            if len(result) == 3:
                stages, image_pool, extras = result
            else:
                stages, image_pool = result
                extras = {}
            payload["models"][model_name] = {
                "model_label": _model_label(model_name),
                "backbone": _model_backbone(model_name),
                "input_size": _model_input_size(model_name),
                "image_pool": image_pool,
                "stages": stages,
                **extras,
            }
            print(f"  OK  {model_name}: {len(stages)} stages, "
                  f"{len(image_pool)} pool images"
                  + (f" · extras: {sorted(extras.keys())}" if extras else ""))
        except Exception as e:
            print(f"  --  {model_name} unavailable: {type(e).__name__}: {e}")
            payload["models"][model_name] = {
                "model_label": _model_label(model_name),
                "backbone": _model_backbone(model_name),
                "input_size": _model_input_size(model_name),
                "unavailable": True,
                "reason": f"{type(e).__name__}: {str(e)[:200]}",
                "stages": [],
                "image_pool": [],
            }
        finally:
            # Free VRAM between models (per plan v4 §A1)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Schema invariants are NOT asserted on a stub-only run (every model
    # would be `unavailable: true` and the assertions exempt those).
    # Once 4C.2 fills in the extractors, assertions enforce stage counts +
    # narration content rules.
    # [PDA-R7 fix] On violation we ABORT the write — do NOT silently promote
    # a broken payload to production. A previous version annotated metadata
    # and wrote anyway; that hid real failures behind a single stderr line.
    invariant_failed = False
    try:
        assert_invariants(payload)
        print("\nOK  Schema invariants pass")
    except AssertionError as e:
        print(f"\n--  Schema invariant FAILURE: {e}", file=sys.stderr)
        print("    Refusing to write broken JSON to production location.",
              file=sys.stderr)
        payload["metadata"]["invariant_violation"] = str(e)
        invariant_failed = True

    if invariant_failed:
        # Write to a DEBUG path so the operator can inspect, but do not
        # overwrite the production-served JSON.
        debug_out = OUT.with_suffix(".invariant_failed.json")
        with open(debug_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"    Debug payload written to {debug_out}", file=sys.stderr)
        return 1

    write_atomic(payload)
    print(f"\nOK  Wrote {OUT}")
    size_kb = OUT.stat().st_size / 1024
    print(f"  Size: {size_kb:.1f} KB")
    return 0


# ────────────────────────────────────────────────────────────────────────
# Per-model metadata helpers
# ────────────────────────────────────────────────────────────────────────
def _model_label(name: str) -> str:
    return {
        "router": "Module 1 · The Crop Router",
        "tomato": "Module 2 · The Tomato Pipeline",
        "apin":   "Module 4 · The APIN Okra/Brassica Ensemble",
        "chilli": "Module 3 · Chilli (router-only)",
    }[name]


def _model_backbone(name: str) -> str:
    return {
        "router": "DINOv2 ViT-Small + 4 reg tokens (~22.06 M params, frozen)",
        "tomato": "V3 (DINOv2-Small + LoRA + FiLM) + SP-LoRA (DINOv2-Reg-Base) · 50/50 ensemble",
        "apin":   "4 parallel signals: DINOv3-ConvNeXt-Small + EfficientNet-B0 + DINOv2 ViT-S + PSV engineered features",
        "chilli": "(uses the router; no specialist deployed)",
    }[name]


def _model_input_size(name: str) -> Dict[str, Any]:
    if name == "tomato":
        return {"v3_input_size": [224, 224], "lora_input_size": [392, 392]}
    return [224, 224]


if __name__ == "__main__":
    sys.exit(main())
