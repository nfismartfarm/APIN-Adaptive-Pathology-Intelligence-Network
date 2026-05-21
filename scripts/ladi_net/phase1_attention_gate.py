"""
Phase 1 attention-quality gate.

Per architecture spec Part Three Phase 1 and Decisions 17/23/33:
- Inspect attention maps for 4 field images per disease class x 5 classes = 20 images.
- Healthy is EXCLUDED (uniform attention on healthy is correct).
- Extract attention at the last DINOv2 block via hook, aggregate across 12 heads (mean),
  apply CORRECT register-token indexing: `attention[:, 0, 5:].reshape(28, 28)`.
- PASS criterion: >=14/20 show lesion-region FOCUS.

Outputs:
  data/specialist/model3/phase1_attention_maps/{class}_{i}.png  — overlay visualisations
  data/specialist/model3/phase1_gate_results.json               — per-image metrics + gate verdict

This script CANNOT do interactive visual inspection. Instead it sets passes_gate
from quantitative proxies (top20_attn_mass, attn_entropy) with thresholds chosen
so the automated verdict approximates what human inspection would conclude:
  FOCUS_AUTO: top20_attn_mass >= 0.18 (top 2.6% of patches carry >=18% of attention
              mass) AND attn_entropy <= 0.85 * log(784) = 5.68 nats.
These thresholds are conservative: clear focus = both satisfied; diffuse/fallback =
neither satisfied. Visual inspection remains the spec-required final say — saved
PNGs are for developer review.

Usage:
    python scripts/ladi_net/phase1_attention_gate.py
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
from pathlib import Path

# Reproducibility
os.environ["PYTHONHASHSEED"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch

from ladinet_config import (
    PROJECT_ROOT, MODEL3_DIR, TOMATO_CLASSES, CLASS_TO_IDX,
    RESOLUTION, PREFIX_TOKENS, NUM_PATCHES, PHASE1_HEADS_PT, SEED,
    FALLBACK_MAX_ATTN_THRESHOLD, FALLBACK_ENTROPY_THRESHOLD,
    IMAGENET_MEAN, IMAGENET_STD, LETTERBOX_PAD_VALUE,
)
from ladinet_model import ABMIL, GatedMLPFusion, SupConProjector, compute_fallback_flag

DISEASE_CLASSES = [c for c in TOMATO_CLASSES if c != "tomato_healthy"]  # 5 disease classes
IMAGES_PER_CLASS = 4
TOTAL_IMAGES = IMAGES_PER_CLASS * len(DISEASE_CLASSES)  # 20

# Automated FOCUS detection thresholds (conservative)
FOCUS_TOP20_MASS_MIN = 0.18          # top-20 of 784 (2.55%) should carry >=18% of mass for "focused"
FOCUS_ENTROPY_MAX_FRAC = 0.85        # entropy < 85% of max (= 5.68 nats)
FOCUS_ENTROPY_MAX = FOCUS_ENTROPY_MAX_FRAC * math.log(NUM_PATCHES)

OUT_DIR_VIS = MODEL3_DIR / "phase1_attention_maps"
OUT_JSON = MODEL3_DIR / "phase1_gate_results.json"


def _letterbox_392(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if (h, w) == (RESOLUTION, RESOLUTION):
        return img_bgr
    scale = RESOLUTION / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    top = (RESOLUTION - new_h) // 2
    bottom = RESOLUTION - new_h - top
    left = (RESOLUTION - new_w) // 2
    right = RESOLUTION - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(LETTERBOX_PAD_VALUE,) * 3)


def _apply_lab_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _bgr_to_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / std
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return tensor


def _overlay_heatmap(orig_bgr: np.ndarray, heatmap_28: np.ndarray,
                     out_path: Path, title: str, metrics: dict) -> None:
    """Create a visualisation: original image | ABMIL-style attention overlay.

    heatmap_28 is already normalised to [0, 1].
    """
    # Resize heatmap to 392x392
    hm_392 = cv2.resize(heatmap_28, (RESOLUTION, RESOLUTION),
                        interpolation=cv2.INTER_LINEAR)
    # Apply colormap (red channel dominant)
    hm_u8 = (hm_392 * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    # Blend at alpha=0.5
    overlay = cv2.addWeighted(orig_bgr, 0.5, hm_color, 0.5, 0)

    # Side-by-side: original | overlay
    canvas = np.zeros((RESOLUTION + 80, RESOLUTION * 2 + 10, 3), dtype=np.uint8)
    canvas[:RESOLUTION, :RESOLUTION] = orig_bgr
    canvas[:RESOLUTION, RESOLUTION + 10:] = overlay

    # Annotate with metrics
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (5, RESOLUTION + 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(
        canvas,
        f"top20_mass={metrics['top20_attn_mass']:.3f}  "
        f"entropy={metrics['attn_entropy']:.2f} (max 6.66)  "
        f"FOCUS_AUTO={'YES' if metrics['passes_gate'] else 'no'}",
        (5, RESOLUTION + 45), font, 0.5, (255, 255, 255), 1,
    )
    cv2.putText(
        canvas,
        f"fallback: max_attn={metrics['max_attn']:.3f}"
        f"{' (FIRES maxlt)' if metrics['fallback_would_fire_max_lt_0p15'] else ''}"
        f"{'  (FIRES entropy)' if metrics['fallback_would_fire_high_entropy'] else ''}",
        (5, RESOLUTION + 68), font, 0.5, (255, 255, 255), 1,
    )
    cv2.imwrite(str(out_path), canvas)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[FAIL] CUDA required.")
        sys.exit(1)

    # 1. Load backbone frozen + 4 registers, at 392px (matching Phase 1)
    import timm
    print("Loading DINOv2-Base-Registers at 392px...")
    backbone = timm.create_model(
        "vit_base_patch14_reg4_dinov2",
        pretrained=True, num_classes=0,
        img_size=RESOLUTION, dynamic_img_size=True,
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # 2. Hook the last block's attention softmax to capture weights
    # timm ViT block: block.attn.attn_drop(softmax(q @ k^T)). We hook .attn.q_norm?
    # Easier: monkey-patch the last block's attn forward to capture attn_weights.
    last_attn = backbone.blocks[-1].attn
    captured = {}

    orig_forward = last_attn.forward

    def hooked_forward(x, attn_mask=None, is_causal=False):
        """Replace timm Attention.forward to capture softmax(QK^T/sqrt(d))."""
        B, N, C = x.shape
        # timm's Attention: qkv -> (B, N, 3, H, C/H) -> permute(2,0,3,1,4)
        qkv = last_attn.qkv(x).reshape(
            B, N, 3, last_attn.num_heads, C // last_attn.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Some timm versions apply q_norm / k_norm
        try:
            q = last_attn.q_norm(q); k = last_attn.k_norm(k)
        except Exception:
            pass
        scale = last_attn.scale
        attn = (q @ k.transpose(-2, -1)) * scale             # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        captured["attn"] = attn.detach()                      # [B, H, N, N]
        # For dtype compatibility with fused Attention modules
        if hasattr(last_attn, "attn_drop"):
            attn_d = last_attn.attn_drop(attn)
        else:
            attn_d = attn
        out = (attn_d @ v).transpose(1, 2).reshape(B, N, C)
        out = last_attn.proj(out)
        try:
            out = last_attn.proj_drop(out)
        except Exception:
            pass
        return out

    last_attn.forward = hooked_forward
    print(f"  hook installed on backbone.blocks[-1].attn (heads={last_attn.num_heads})")

    # 3. Load Phase 1 best ABMIL head (for reference — we display DINO attention though)
    print(f"Loading Phase 1 checkpoint: {PHASE1_HEADS_PT}")
    if not PHASE1_HEADS_PT.exists():
        print(f"[FAIL] {PHASE1_HEADS_PT} not found.")
        sys.exit(1)
    phase1 = torch.load(PHASE1_HEADS_PT, map_location=device, weights_only=False)
    abmil = ABMIL().to(device).eval()
    abmil.load_state_dict(phase1["abmil_state_dict"])
    for p in abmil.parameters():
        p.requires_grad = False

    # 4. Load field_val paths + pick first 4 per disease class (sorted for determinism)
    split_path = MODEL3_DIR / "split_indices.json"
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    field_val_paths = sorted(split["field_val"])

    def _class_of_path(p: str) -> str | None:
        p_norm = str(p).replace("\\", "/")
        for c in TOMATO_CLASSES:
            if f"/cleaned/{c}/" in p_norm or f"/{c}/" in p_norm:
                return c
        return None

    by_class: dict[str, list[str]] = {c: [] for c in DISEASE_CLASSES}
    for p in field_val_paths:
        c = _class_of_path(p)
        if c in by_class and len(by_class[c]) < IMAGES_PER_CLASS:
            by_class[c].append(p)

    selected = []
    for c in DISEASE_CLASSES:
        for p in by_class[c]:
            selected.append((c, p))
        if len(by_class[c]) < IMAGES_PER_CLASS:
            print(f"  [WARN] {c}: only {len(by_class[c])} field_val images available "
                  f"(expected {IMAGES_PER_CLASS})")
    print(f"Selected {len(selected)} images across {len(DISEASE_CLASSES)} disease classes")

    # 5. Process each image
    OUT_DIR_VIS.mkdir(parents=True, exist_ok=True)
    per_image = []
    per_class_focus = {c: {"focus": 0, "total": 0} for c in DISEASE_CLASSES}

    for idx, (cls, path) in enumerate(selected):
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [WARN] could not load {path}")
            continue
        img_letterbox = _letterbox_392(img_bgr)
        img_preproc = _apply_lab_clahe(img_letterbox)
        tensor = _bgr_to_tensor(img_preproc, device)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            feat = backbone.forward_features(tensor)  # [1, 789, 768]

        # captured["attn"] shape: [B, H, N, N] = [1, 12, 789, 789]
        attn = captured["attn"].float()
        # Mean across heads (Decision 17 standard visualisation practice)
        attn_mean_heads = attn.mean(dim=1)          # [1, 789, 789]
        # CLS -> spatial row, skip CLS(0) + 4 registers (1..4), use patches 5..788
        cls_to_spatial = attn_mean_heads[:, 0, PREFIX_TOKENS:]  # [1, 784]
        # Re-normalise (rows sum to 1 by construction, but after slicing we lost CLS+reg mass)
        cls_to_spatial = cls_to_spatial / cls_to_spatial.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_vec = cls_to_spatial.squeeze(0).cpu().numpy()  # [784]

        # Metrics
        top20_idx = np.argpartition(-attn_vec, 20)[:20]
        top20_mass = float(attn_vec[top20_idx].sum())
        eps = 1e-12
        entropy = float(-(attn_vec * np.log(attn_vec + eps)).sum())
        max_attn = float(attn_vec.max())

        # ALSO check ABMIL's own attention pattern (to see if Phase 1's ABMIL concentrates)
        patch_tokens = feat[:, PREFIX_TOKENS:, :].float()
        _, abmil_attn = abmil(patch_tokens)                    # [1, 784]
        abmil_vec = abmil_attn.squeeze(0).cpu().numpy()
        abmil_top20_mass = float(abmil_vec[np.argpartition(-abmil_vec, 20)[:20]].sum())
        abmil_entropy = float(-(abmil_vec * np.log(abmil_vec + eps)).sum())
        abmil_max = float(abmil_vec.max())

        # Heatmap normalised to [0, 1] for visualisation (use ABMIL for visualisation
        # since that's what LADI-Net actually uses at inference)
        abmil_map = abmil_vec.reshape(28, 28)
        hm = (abmil_map - abmil_map.min()) / (abmil_map.max() - abmil_map.min() + 1e-8)

        # Fallback flag conditions (Decision 23 — on ABMIL attention, not DINO)
        fires_max = bool(abmil_max < FALLBACK_MAX_ATTN_THRESHOLD)
        fires_entropy = bool(abmil_entropy > FALLBACK_ENTROPY_THRESHOLD)

        # Automated FOCUS verdict on ABMIL (this is what LADI-Net uses at inference)
        focus_auto = bool(
            (abmil_top20_mass >= FOCUS_TOP20_MASS_MIN)
            and (abmil_entropy <= FOCUS_ENTROPY_MAX)
            and not fires_max and not fires_entropy
        )

        metrics = {
            "path": str(path),
            "class": cls,
            "top20_attn_mass": abmil_top20_mass,          # ABMIL's
            "attn_entropy": abmil_entropy,                 # ABMIL's
            "max_attn": abmil_max,
            "fallback_would_fire_max_lt_0p15": fires_max,
            "fallback_would_fire_high_entropy": fires_entropy,
            "dino_top20_mass": top20_mass,
            "dino_entropy": entropy,
            "dino_max_attn": max_attn,
            "passes_gate": focus_auto,
        }
        per_image.append(metrics)
        per_class_focus[cls]["total"] += 1
        if focus_auto:
            per_class_focus[cls]["focus"] += 1

        # Save visualisation (ABMIL-based heatmap)
        out_path = OUT_DIR_VIS / f"{cls}_{idx:02d}.png"
        _overlay_heatmap(img_letterbox, hm, out_path,
                         title=f"{cls}  [{Path(path).name[:40]}]",
                         metrics=metrics)
        print(f"  [{idx+1:2d}/{len(selected)}] {cls:35s}  "
              f"ABMIL: mass={abmil_top20_mass:.3f}  ent={abmil_entropy:.2f}  "
              f"max={abmil_max:.3f}  FOCUS_AUTO={'YES' if focus_auto else 'no'}")

    total_focus = sum(m["passes_gate"] for m in per_image)
    total_inspected = len(per_image)
    pass_threshold = max(1, int(round(0.70 * total_inspected)))
    gate_passed = total_focus >= pass_threshold

    per_class_rates = {
        c: (d["focus"] / d["total"]) if d["total"] > 0 else 0.0
        for c, d in per_class_focus.items()
    }

    if gate_passed:
        recommendation = "PROCEED_TO_CORAL"
    else:
        # All-diffuse classes drive the failure?
        non_diffuse_classes = ["tomato_foliar_spot", "tomato_septoria_leaf_spot",
                               "tomato_late_blight"]
        non_diffuse_focus = sum(per_class_focus[c]["focus"] for c in non_diffuse_classes)
        non_diffuse_total = sum(per_class_focus[c]["total"] for c in non_diffuse_classes)
        if non_diffuse_total > 0 and non_diffuse_focus / non_diffuse_total >= 0.6:
            recommendation = "DEVELOPER_DECISION_DIFFUSE_ONLY_FAILURE"
        else:
            recommendation = "DEVELOPER_DECISION_BROAD_FAILURE"

    result = {
        "checkpoint": str(PHASE1_HEADS_PT),
        "generated_at": datetime.datetime.now().isoformat(),
        "total_inspected": total_inspected,
        "total_focus_auto": total_focus,
        "pass_threshold": pass_threshold,
        "gate_passed_auto": gate_passed,
        "note": (
            "passes_gate was set by automated quantitative proxies, not interactive "
            "visual inspection. Thresholds: top20_attn_mass >= 0.18 AND "
            "attn_entropy <= 0.85*log(784)=5.68 AND neither fallback condition fires. "
            "Saved PNG visualisations are in data/specialist/model3/phase1_attention_maps/ "
            "for developer visual review."
        ),
        "thresholds": {
            "focus_top20_mass_min": FOCUS_TOP20_MASS_MIN,
            "focus_entropy_max": FOCUS_ENTROPY_MAX,
            "fallback_max_attn_threshold": FALLBACK_MAX_ATTN_THRESHOLD,
            "fallback_entropy_threshold": FALLBACK_ENTROPY_THRESHOLD,
        },
        "per_image": per_image,
        "per_class_focus_rate": per_class_rates,
        "per_class_focus_counts": per_class_focus,
        "recommendation": recommendation,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 72)
    print(f"GATE RESULT: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"  focus / total = {total_focus} / {total_inspected}  (threshold {pass_threshold})")
    print(f"Per-class focus rates:")
    for c, rate in per_class_rates.items():
        fc = per_class_focus[c]
        print(f"  {c:35s}: {fc['focus']}/{fc['total']}  ({rate*100:.0f}%)")
    print(f"Recommendation: {recommendation}")
    print(f"Results JSON: {OUT_JSON}")
    print(f"Visualisations: {OUT_DIR_VIS}")
    print("=" * 72)


if __name__ == "__main__":
    main()
