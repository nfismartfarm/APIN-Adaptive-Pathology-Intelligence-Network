"""Round 2 fixes: relabel 8-D/E/F and augment 8-B."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "ladi_issues.md"
t = P.read_text(encoding="utf-8")

# 8-D
old_8d = "Fix Status: FIXED\n\nReason/Outcome:\nDocumented in Decision 17 (hyperparameter specification block). AdamW optimizer with parameter groups: LoRA adapters LR=1e-4 weight_decay=0.01; ABMIL head + gated MLP + SupCon projection head (all from-scratch heads) LR=5e-4 weight_decay=0.0. Cosine schedule with 2-epoch warmup. References to MASTER_PLAN: the '1e-3 destroyed pretrained features in epoch 1' bug was for ConvNeXt full fine-tune; for LoRA on DINOv2-Base the appropriate LoRA LR is 1e-4 (an order of magnitude lower than from-scratch heads)."
new_8d = "Fix Status: DEFERRED-PHASE2\n\nReason/Outcome:\nSpecified in Decision 17 §17.3. AdamW optimizer with parameter groups: LoRA adapters LR=1e-4 weight_decay=0.01; ABMIL + gated MLP + SupCon projector (from-scratch heads) LR=5e-4 weight_decay=0.0. Cosine schedule with 2-epoch warmup. References to MASTER_PLAN: the '1e-3 destroyed pretrained features in epoch 1' bug was for ConvNeXt full fine-tune; LoRA on DINOv2-Base uses 1e-4 (order of magnitude lower than from-scratch heads). Specification only — the Phase 2 training script that constructs the optimizer and scheduler does not yet exist."
assert old_8d in t, "8-D not found"
t = t.replace(old_8d, new_8d, 1)

# 8-E
old_8e = "Fix Status: FIXED\n\nReason/Outcome:\nDocumented in Decision 17. Phase 2 training code uses `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` context manager around the forward pass. Model parameters remain in float32 — NEVER call `.to(torch.bfloat16)` on any parameter that will be trained. Optimizer states (AdamW m/v buffers) remain float32 (required for numerical stability). This matches the MASTER_PLAN bug fix 'BF16 permanent cast destroys gradients for full fine-tune → fixed to float32 + autocast'. vram_test.py already implements this pattern correctly and is the reference implementation."
new_8e = "Fix Status: DEFERRED-PHASE2\n\nReason/Outcome:\nSpecified in Decision 17 §17.3 (mixed precision line). Phase 2 training code uses `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` around the forward pass. Model parameters REMAIN in float32 — NEVER `.to(torch.bfloat16)` on any trainable parameter (MASTER_PLAN bug 'BF16 permanent cast destroys gradients for full fine-tune → fixed to float32 + autocast'). Optimizer states remain float32. `scripts/ladi_net/vram_test.py` implements this correctly as the reference. Specification only — not yet applied in any Phase 2 training script."
assert old_8e in t, "8-E not found"
t = t.replace(old_8e, new_8e, 1)

# 8-F
old_8f = "Fix Status: FIXED\n\nReason/Outcome:\nDocumented in Decision 17. After `scaler.unscale_(optimizer)` (if using fp16 scaler) or directly after `loss.backward()` (if using bf16), call `torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)` on ALL trainable parameters (LoRA + heads). Log grad_norm every step; if grad_norm is consistently > 10 pre-clipping for 100+ steps, investigate (likely CORAL weight too high or loss scaling issue)."
new_8f = "Fix Status: DEFERRED-PHASE2\n\nReason/Outcome:\nSpecified in Decision 17 §17.3. After `loss.backward()` (bf16 does not require a scaler), call `torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)` on all trainable parameters (LoRA + heads). Log grad_norm every step; if pre-clip grad_norm is consistently >10 for 100+ steps, investigate (likely CORAL weight too high or loss scaling). Specification only — pending Phase 2 training-script implementation."
assert old_8f in t, "8-F not found"
t = t.replace(old_8f, new_8f, 1)

# 8-B augmentation
old_8b_tail = "Decision 16 + Check 4 both reference this post-Phase-1 task."
new_8b_tail = ("Decision 16 + Check 4 both reference this post-Phase-1 task. "
               "**Provenance requirement (Critique 10):** when the Phase 1 script saves the "
               "new ABMIL-based CORAL target, it MUST save a dict, not a raw tensor: "
               "`torch.save({'cov': C, 'source': 'abmil_features_phase1', 'n_samples': 680, "
               "'generated_at': datetime.now().isoformat(), 'resolution': 392}, coral_target_cov.pt)`. "
               "The Phase 2 training-script loader MUST assert `data['source'] == 'abmil_features_phase1'` "
               "at startup; any other value (including a plain tensor) raises a fatal error. "
               "This prevents silent type-mismatch regression to CLS-based CORAL alignment.")
assert old_8b_tail in t, "8-B tail not found"
t = t.replace(old_8b_tail, new_8b_tail, 1)

P.write_text(t, encoding="utf-8")

import re
c_fixed = len(re.findall(r"Fix Status: FIXED", t))
c_d2 = len(re.findall(r"Fix Status: DEFERRED-PHASE2", t))
c_dl = len(re.findall(r"Fix Status: DEFERRED-LATER", t))
c_kl = len(re.findall(r"Fix Status: KNOWN-LIMITATION", t))
print(f"FIXED={c_fixed}  DEFERRED-PHASE2={c_d2}  DEFERRED-LATER={c_dl}  KNOWN-LIMITATION={c_kl}  total={c_fixed+c_d2+c_dl+c_kl}")
