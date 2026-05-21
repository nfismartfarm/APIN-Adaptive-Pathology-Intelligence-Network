"""
LADI-Net VRAM budget test.

Builds the full proposed LADI-Net architecture and runs exactly 5 training
steps at 392px for batch sizes {4, 2, 1} in sequence, reporting peak VRAM
via torch.cuda.max_memory_allocated() after each batch-size run.

Configuration under test:
- Backbone: DINOv2-Base-Registers (vit_base_patch14_reg4_dinov2), 12 blocks
- LoRA: rank=8, alpha=16, applied to q and v projections of the LAST 8 blocks
  (blocks[4:12] — interpreted as "layers 17-24" in the common DINOv2 numbering
  convention where blocks are 1-indexed and attention/MLP pairs count as 2
  layers per block; for a 12-block Base model this is the top 8 transformer
  blocks, i.e. the trainable portion).
- Head: ABMIL (gated attention) pool over patch tokens + CLS token concat,
  then gated MLP fusion, then 6-class linear classifier.
- SupCon projector: MLP 768 -> 256 -> 128, L2-normalised (for supervised
  contrastive loss). Skipped at bs=1 (needs >=2 samples).
- CORAL loss: Frobenius distance between batch covariance and target CORAL
  covariance loaded from data/specialist/model3/coral_target_cov.pt.

Losses summed: CE + 0.5 * SupCon + 0.1 * CORAL.
Optimizer: AdamW over LoRA params + head params only.
AMP: bfloat16 autocast (default on RTX 4060 / Ada).

The script is read-only w.r.t. sacred files. It does not touch app/config.py,
models/*.pt, data/metadata/source_map.csv, or scripts/apin/**.

Usage:
    python scripts/ladi_net/vram_test.py
    python scripts/ladi_net/vram_test.py --dtype fp16
    python scripts/ladi_net/vram_test.py --no-amp
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORAL_PATH = PROJECT_ROOT / "data" / "specialist" / "model3" / "coral_target_cov.pt"

NUM_CLASSES = 6
PATCH_SIZE = 14
EMBED_DIM = 768
LORA_RANK = 8
LORA_ALPHA = 16
LORA_BLOCKS_FROM_TOP = 8  # last 8 of 12 for Base
SUPCON_PROJ_DIM = 128
SUPCON_HIDDEN = 256
FUSION_HIDDEN = 512
SUPCON_TEMPERATURE = 0.1
LOSS_W_CE = 1.0
LOSS_W_SUPCON = 0.5
LOSS_W_CORAL = 0.1


# ---------------------------------------------------------------------------
# Manual LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank delta: y = Wx + B(Ax)*s."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero so initial delta is 0
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + delta


def _wrap_qkv_with_lora(block: nn.Module, rank: int, alpha: int) -> int:
    """Attach LoRA to q and v projections in timm ViT's fused attn.qkv (Linear).

    timm ViT blocks expose `block.attn.qkv` as a single Linear out=3*D. The
    cleanest LoRA approach is to wrap that whole Linear (trains Q, K, V deltas
    at once). Returns number of wrapped layers.
    """
    attn = getattr(block, "attn", None)
    if attn is None:
        return 0
    qkv = getattr(attn, "qkv", None)
    if qkv is None or not isinstance(qkv, nn.Linear):
        return 0
    attn.qkv = LoRALinear(qkv, rank, alpha)
    return 1


# ---------------------------------------------------------------------------
# ABMIL gated-attention pool
# ---------------------------------------------------------------------------
class GatedAttentionPool(nn.Module):
    """Ilse et al. 2018 gated-attention MIL pool. Input [B,N,D] -> [B,D]."""

    def __init__(self, dim: int, attn_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, attn_dim)
        self.U = nn.Linear(dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(self.V(tokens)) * torch.sigmoid(self.U(tokens))
        scores = self.w(a).squeeze(-1)                # [B, N]
        weights = torch.softmax(scores, dim=1)         # [B, N]
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]


# ---------------------------------------------------------------------------
# Gated MLP fusion
# ---------------------------------------------------------------------------
class GatedMLPFusion(nn.Module):
    """Fuses two D-dim vectors (CLS + MIL pool) via a gated MLP -> D."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(2 * dim, hidden)
        self.gate = nn.Linear(2 * dim, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, b], dim=-1)
        h = F.silu(self.proj(x)) * torch.sigmoid(self.gate(x))
        return self.out(h)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class LADINet(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "vit_base_patch14_reg4_dinov2",
            pretrained=True, num_classes=0,
            img_size=224, dynamic_img_size=True,
        )
        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # LoRA on the last LORA_BLOCKS_FROM_TOP blocks
        blocks = list(self.backbone.blocks)
        n_blocks = len(blocks)
        start = max(0, n_blocks - LORA_BLOCKS_FROM_TOP)
        wrapped = 0
        for blk in blocks[start:]:
            wrapped += _wrap_qkv_with_lora(blk, LORA_RANK, LORA_ALPHA)
        self._lora_layers = wrapped
        self._lora_block_range = (start, n_blocks)

        # Heads
        self.mil_pool = GatedAttentionPool(EMBED_DIM, attn_dim=128)
        self.fusion = GatedMLPFusion(EMBED_DIM, FUSION_HIDDEN)
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)
        self.supcon_proj = nn.Sequential(
            nn.Linear(EMBED_DIM, SUPCON_HIDDEN),
            nn.GELU(),
            nn.Linear(SUPCON_HIDDEN, SUPCON_PROJ_DIM),
        )

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor):
        # timm ViT forward_features returns [B, 1+num_reg+num_patches, D]
        feats = self.backbone.forward_features(x)
        # Index 0 is CLS; 1..1+num_reg are register tokens; rest are patches
        num_prefix = 1 + getattr(self.backbone, "num_reg_tokens", 4)
        cls = feats[:, 0]
        patches = feats[:, num_prefix:]
        mil = self.mil_pool(patches)
        fused = self.fusion(cls, mil)
        logits = self.classifier(fused)
        proj = F.normalize(self.supcon_proj(fused), dim=-1)
        return logits, fused, proj


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def supcon_loss(proj: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020). Requires >=2 samples."""
    B = proj.size(0)
    if B < 2:
        return proj.new_zeros(())
    sim = proj @ proj.t() / tau                          # [B, B]
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    mask_self = torch.eye(B, dtype=torch.bool, device=proj.device)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & ~mask_self
    exp_sim = torch.exp(sim) * (~mask_self)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(log_prob * pos_mask).sum(dim=1) / pos_count
    # Only average over anchors with at least one positive
    valid = pos_mask.any(dim=1)
    if not valid.any():
        return proj.new_zeros(())
    return loss[valid].mean()


def coral_loss(feats: torch.Tensor, target_cov: torch.Tensor) -> torch.Tensor:
    """||Cov(batch_feats) - target_cov||_F^2 / (4 D^2)."""
    B, D = feats.shape
    if B < 2:
        return feats.new_zeros(())
    centered = feats - feats.mean(dim=0, keepdim=True)
    batch_cov = (centered.t() @ centered) / (B - 1)
    diff = batch_cov - target_cov.to(feats.device, feats.dtype)
    return (diff * diff).sum() / (4.0 * D * D)


# ---------------------------------------------------------------------------
# One run at a given batch size
# ---------------------------------------------------------------------------
def run_one_bs(model: LADINet, target_cov: torch.Tensor, bs: int, res: int,
               steps: int, device: torch.device, amp_dtype) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scaler = None
    use_amp = amp_dtype is not None
    if use_amp and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")

    model.train()
    step_times = []
    oom = False
    oom_step = None
    err_msg = None

    try:
        for step in range(steps):
            t0 = time.time()
            x = torch.randn(bs, 3, res, res, device=device)
            y = torch.randint(0, NUM_CLASSES, (bs,), device=device)

            optim.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits, fused, proj = model(x)
                    ce = F.cross_entropy(logits, y)
                    sc = supcon_loss(proj, y, SUPCON_TEMPERATURE)
                    cl = coral_loss(fused.float(), target_cov)
                    loss = LOSS_W_CE * ce + LOSS_W_SUPCON * sc + LOSS_W_CORAL * cl
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()
            else:
                logits, fused, proj = model(x)
                ce = F.cross_entropy(logits, y)
                sc = supcon_loss(proj, y, SUPCON_TEMPERATURE)
                cl = coral_loss(fused, target_cov)
                loss = LOSS_W_CE * ce + LOSS_W_SUPCON * sc + LOSS_W_CORAL * cl
                loss.backward()
                optim.step()

            torch.cuda.synchronize(device)
            step_times.append(time.time() - t0)
            peak_now = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"    step {step + 1}/{steps}  loss={loss.item():.4f}  "
                  f"ce={ce.item():.3f} sc={float(sc):.3f} cl={float(cl):.3f}  "
                  f"peak_so_far={peak_now:.2f}GB  t={step_times[-1]:.2f}s")
    except torch.cuda.OutOfMemoryError as e:
        oom = True
        oom_step = step + 1 if step_times else 0
        err_msg = str(e).splitlines()[0]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            oom = True
            oom_step = step + 1 if step_times else 0
            err_msg = str(e).splitlines()[0]
        else:
            raise

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    return {
        "bs": bs, "res": res, "oom": oom, "oom_step": oom_step,
        "err": err_msg,
        "peak_allocated_gb": peak_gb,
        "peak_reserved_gb": reserved_gb,
        "mean_step_s": (sum(step_times) / len(step_times)) if step_times else None,
        "n_steps_completed": len(step_times),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=None,
                    help="Single resolution; overridden by --resolutions")
    ap.add_argument("--resolutions", type=int, nargs="+", default=None,
                    help="Grid of resolutions to test")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 8, 4])
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--vram-budget-gb", type=float, default=7.5,
                    help="Per-step peak allocated VRAM budget")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This test requires the RTX 4060.")
        sys.exit(1)

    # Resolve the resolution list
    if args.resolutions is None:
        resolutions = [args.res if args.res is not None else 392]
    else:
        resolutions = args.resolutions

    for r in resolutions:
        if r % PATCH_SIZE != 0:
            print(f"ERROR: resolution {r} not divisible by patch size {PATCH_SIZE}.")
            sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print("=" * 72)
    print(f"LADI-Net VRAM test  |  {gpu_name}  |  {total_vram_gb:.2f}GB total")
    print(f"resolutions={resolutions}  batch_sizes={args.batch_sizes}  "
          f"steps={args.steps}  dtype={args.dtype}  budget={args.vram_budget_gb}GB")
    print("=" * 72)

    amp_dtype = None
    if not args.no_amp:
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                     "fp32": None}[args.dtype]

    # Load CORAL target covariance (may be shape-incompatible per-res; we just
    # need a (D,D) matrix for the loss — embed_dim is 768 for Base across all res).
    if not CORAL_PATH.exists():
        print(f"ERROR: CORAL cov not found at {CORAL_PATH}")
        sys.exit(1)
    target_cov = torch.load(CORAL_PATH, weights_only=True)
    print(f"Loaded CORAL target cov: {tuple(target_cov.shape)} {target_cov.dtype}")

    # Build model once (so backbone download happens once)
    print("Building LADI-Net ...")
    model = LADINet().to(device)
    n_trainable = sum(p.numel() for p in model.trainable_params())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA wrapped {model._lora_layers} qkv layers (blocks "
          f"{model._lora_block_range[0]}..{model._lora_block_range[1] - 1})")
    print(f"  Trainable params: {n_trainable:,} / {n_total:,}")

    # Grid sweep: for each resolution, walk batch sizes in descending order.
    # If bs=X OOMs, skip larger bs at the same resolution (would also OOM).
    results = []
    # Ensure batch sizes are processed descending so early-OOM short-circuits
    bs_desc = sorted(set(args.batch_sizes), reverse=True)
    for res in resolutions:
        skip_above = None
        for bs in bs_desc:
            if skip_above is not None and bs >= skip_above:
                print(f"\n--- batch_size={bs} at {res}px : SKIPPED (larger bs OOM-ed) ---")
                results.append({
                    "bs": bs, "res": res, "oom": True, "oom_step": 0,
                    "err": "skipped — smaller bs hit OOM first",
                    "peak_allocated_gb": float("nan"),
                    "peak_reserved_gb": float("nan"),
                    "mean_step_s": None, "n_steps_completed": 0,
                    "skipped": True,
                })
                continue
            print(f"\n--- batch_size={bs} at {res}px ---")
            r = run_one_bs(model, target_cov, bs, res, args.steps, device, amp_dtype)
            r["skipped"] = False
            results.append(r)
            status = "OOM" if r["oom"] else "OK"
            print(f"  [{status}] bs={bs} res={res}  "
                  f"peak_alloc={r['peak_allocated_gb']:.2f}GB  "
                  f"peak_reserved={r['peak_reserved_gb']:.2f}GB  "
                  f"mean_step={r['mean_step_s']}")
            if r["oom"]:
                print(f"       OOM at step {r['oom_step']}: {r['err']}")
                skip_above = bs  # no point trying larger bs at this res
            torch.cuda.empty_cache()
            gc.collect()

    # ─── Summary grid ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY GRID")
    print("=" * 72)
    print(f"{'res':>5} | {'bs':>3} | {'status':>6} | {'peak_alloc_GB':>13} | "
          f"{'peak_reserved_GB':>16} | {'mean_step_s':>11}")
    for r in results:
        status = "OOM" if r["oom"] else "OK"
        ms = f"{r['mean_step_s']:.2f}" if r["mean_step_s"] is not None else "n/a"
        peak = (f"{r['peak_allocated_gb']:.2f}"
                if not math.isnan(r['peak_allocated_gb']) else "n/a")
        reserved = (f"{r['peak_reserved_gb']:.2f}"
                    if not math.isnan(r['peak_reserved_gb']) else "n/a")
        print(f"{r['res']:>5} | {r['bs']:>3} | {status:>6} | "
              f"{peak:>13} | {reserved:>16} | {ms:>11}")

    # ─── Decision: highest resolution where bs=16 fits budget ─────────────
    print("\nDECISION (bs=16 within {:.1f}GB budget):".format(args.vram_budget_gb))
    bs16_by_res = {}
    for r in results:
        if r["bs"] == 16:
            bs16_by_res[r["res"]] = r
    if not bs16_by_res:
        print("  bs=16 not in tested batch sizes. Cannot decide against primary rule.")
    else:
        viable_res = [
            res for res, r in bs16_by_res.items()
            if not r["oom"] and r["peak_allocated_gb"] <= args.vram_budget_gb
        ]
        if viable_res:
            chosen = max(viable_res)
            peak = bs16_by_res[chosen]["peak_allocated_gb"]
            print(f"  VERDICT: highest resolution with bs=16 ≤ {args.vram_budget_gb}GB  "
                  f"is {chosen}px (peak {peak:.2f}GB).")
        else:
            print(f"  VERDICT: bs=16 does not fit at any tested resolution.")
            # Report largest viable bs per resolution
            for res in resolutions:
                best_bs = None
                best_peak = None
                for r in sorted((x for x in results if x["res"] == res),
                                key=lambda z: z["bs"], reverse=True):
                    if not r["oom"] and r["peak_allocated_gb"] <= args.vram_budget_gb:
                        best_bs = r["bs"]
                        best_peak = r["peak_allocated_gb"]
                        break
                if best_bs is not None:
                    print(f"    {res}px: largest viable bs = {best_bs} "
                          f"(peak {best_peak:.2f}GB)")
                else:
                    print(f"    {res}px: no tested bs fits within budget")

    # JSON results file
    import json as _json
    out_path = PROJECT_ROOT / "logs" / "ladi_vram_grid_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump({
            "device": gpu_name, "total_vram_gb": total_vram_gb,
            "budget_gb": args.vram_budget_gb, "dtype": args.dtype,
            "resolutions": resolutions, "batch_sizes": args.batch_sizes,
            "results": [{k: (v if not isinstance(v, float) or not math.isnan(v)
                             else None) for k, v in r.items()} for r in results],
        }, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
