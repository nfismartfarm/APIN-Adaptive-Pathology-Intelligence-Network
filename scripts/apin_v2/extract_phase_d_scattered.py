"""extract_phase_d_scattered.py · Close the last 4 Phase 4D items.

  M1 · router 4×4 confusion          ← run pre-trained head on cached val features
  M2 · tomato 6×6 confusion          ← read cached final_val_predictions JSONs
  M2 · per-block CLS evolution       ← run DINOv2-Reg backbone with per-block hooks
                                        on the 3 canonical tomato pool images
  EXCLUDED Sankey node               ← scan source_map.csv for excluded rows

All four write to `_qa_tmp/_pipeline_atlas_phase_d_scattered.json`. Real numbers only.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, '_qa_tmp', '_pipeline_atlas_phase_d_scattered.json')

CLASS_TOMATO = [
    'tomato_late_blight', 'tomato_septoria_leaf_spot', 'tomato_yellow_leaf_curl_virus',
    'tomato_mosaic_virus', 'tomato_foliar_spot', 'tomato_healthy',
]


# ── M1 · router 4×4 ────────────────────────────────────────────────────
def extract_router_confusion():
    """Run the trained router head on cached val features.
    The router_best.pt file (8.8 KB) is the linear head only; the DINOv2-Reg
    backbone features are cached at cache/router/val_features.pt (10 MB).
    """
    print('M1 · router 4x4 ...')
    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_score, recall_score,
    )
    sys.path.insert(0, ROOT)
    from app.config_router import NUM_CLASSES, CLASS_NAMES, DINOV2_EMBED_DIM

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_cache = torch.load(os.path.join(ROOT, 'cache/router/val_features.pt'),
                           weights_only=False, map_location='cpu')
    val_feats = val_cache['features'].to(device).float()
    val_labels_t = val_cache['labels']
    if hasattr(val_labels_t, 'cpu'):
        val_labels = val_labels_t.cpu().numpy()
    else:
        val_labels = np.asarray(val_labels_t)

    head = nn.Linear(DINOV2_EMBED_DIM, NUM_CLASSES).to(device)
    ckpt = torch.load(os.path.join(ROOT, 'models/router/router_best.pt'),
                      map_location=device, weights_only=False)
    head.load_state_dict(ckpt['head_state_dict'])
    head.eval()

    with torch.no_grad():
        logits = head(val_feats)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    cm = confusion_matrix(val_labels, preds, labels=list(range(NUM_CLASSES)))
    per_crop = {}
    for i, c in enumerate(CLASS_NAMES):
        p = precision_score(val_labels, preds, labels=[i], average='macro',
                            zero_division=0)
        r = recall_score(val_labels, preds, labels=[i], average='macro',
                         zero_division=0)
        f = f1_score(val_labels, preds, labels=[i], average='macro',
                     zero_division=0)
        per_crop[c] = {
            'precision': round(float(p), 4),
            'recall':    round(float(r), 4),
            'f1':        round(float(f), 4),
            'n':         int((val_labels == i).sum()),
        }

    # Major confusions: off-diagonal counts >= 3
    major = []
    for ti in range(NUM_CLASSES):
        for pj in range(NUM_CLASSES):
            if ti == pj:
                continue
            n = int(cm[ti, pj])
            if n >= 3:
                major.append({
                    'true_crop': CLASS_NAMES[ti],
                    'pred_crop': CLASS_NAMES[pj],
                    'count':     n,
                })
    major.sort(key=lambda x: -x['count'])

    print(f'  router n_eval={len(val_labels)} · accuracy={accuracy_score(val_labels, preds):.4f}'
          f' · macro_f1={f1_score(val_labels, preds, average="macro"):.4f}')
    return {
        'class_order':           CLASS_NAMES,
        'confusion_matrix':      [[int(v) for v in row] for row in cm.tolist()],
        'n_eval_images':         int(len(val_labels)),
        'overall_accuracy':      round(float(accuracy_score(val_labels, preds)), 4),
        'overall_macro_f1':      round(float(f1_score(val_labels, preds, average='macro')), 4),
        'per_crop_metrics':      per_crop,
        'major_confusions':      major,
        'router_head_epoch':     int(ckpt.get('epoch', -1)),
        'router_head_best_f1':   round(float(ckpt.get('best_f1', 0.0)), 4),
        'split':                 'val',
        'evaluation_subset':     'router val split (cache/router/val_features.pt)',
    }


# ── M2 · tomato 6×6 (from cached test-split predictions) ───────────────
def extract_tomato_confusion():
    """Build a 6x6 confusion from cached V3 + SP-LoRA test predictions
    on the locked tomato test split (n=104)."""
    print('M2 · tomato 6x6 ...')
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_score, recall_score,
    )

    v3_path   = os.path.join(ROOT, 'data/specialist/model3/final_val_predictions.json')
    lora_path = os.path.join(ROOT, 'data/specialist/model3/final_val_predictions_sp_lora.json')
    with open(v3_path,   encoding='utf-8') as f: v3   = json.load(f)
    with open(lora_path, encoding='utf-8') as f: lora = json.load(f)
    # Align by image_path
    lora_by_path = {e['image_path']: e for e in lora}

    cls_to_idx = {c: i for i, c in enumerate(CLASS_TOMATO)}

    def confusion_for(records, true_key, pred_key):
        true = np.array([cls_to_idx.get(e[true_key], -1) for e in records])
        pred = np.array([cls_to_idx.get(e[pred_key], -1) for e in records])
        mask = (true >= 0) & (pred >= 0)
        return true[mask], pred[mask]

    # V3 confusion
    t_v3, p_v3 = confusion_for(v3, 'true_class', 'pred_class')
    cm_v3 = confusion_matrix(t_v3, p_v3, labels=list(range(6)))
    # SP-LoRA primary
    t_lora, p_lora = confusion_for(lora, 'true_class', 'pred_class_primary')
    cm_lora = confusion_matrix(t_lora, p_lora, labels=list(range(6)))
    # Ensemble: take the V3+SP-LoRA fused probability (per the apin pipeline:
    # V3 softmax(logits / 0.5) + SP-LoRA softmax(logits, T=1.0), 50/50 average).
    # We approximate by averaging all_probs across the two records when both
    # exist for the same image_path; this is what the deployed pipeline
    # actually does at inference time.
    def fused_pred(v3_rec, lora_rec):
        if lora_rec is None:
            # Fall back to V3 if LoRA missing
            return v3_rec['pred_class']
        # V3 stored probs are already-sharpened (T=0.5 on the V3 branch)
        # LoRA stored probs are T=1.0
        # The deployed fusion averages them per-class.
        p_v3   = v3_rec.get('all_probs', {})
        p_lora = lora_rec.get('all_probs_primary', {})
        avg = {}
        for c in CLASS_TOMATO:
            avg[c] = 0.5 * p_v3.get(c, 0) + 0.5 * p_lora.get(c, 0)
        return max(avg.items(), key=lambda kv: kv[1])[0]

    fused_true = []
    fused_pred_list = []
    for v in v3:
        if v['true_class'] not in cls_to_idx:
            continue
        l = lora_by_path.get(v['image_path'])
        fused_true.append(cls_to_idx[v['true_class']])
        fused_pred_list.append(cls_to_idx[fused_pred(v, l)])
    fused_true = np.array(fused_true)
    fused_pred_list = np.array(fused_pred_list)
    cm_ens = confusion_matrix(fused_true, fused_pred_list, labels=list(range(6)))

    def per_class(true_, pred_):
        out = {}
        for i, c in enumerate(CLASS_TOMATO):
            p = precision_score(true_, pred_, labels=[i], average='macro',
                                zero_division=0)
            r = recall_score(true_, pred_, labels=[i], average='macro',
                             zero_division=0)
            f = f1_score(true_, pred_, labels=[i], average='macro',
                         zero_division=0)
            out[c] = {
                'precision': round(float(p), 4),
                'recall':    round(float(r), 4),
                'f1':        round(float(f), 4),
                'n':         int((true_ == i).sum()),
            }
        return out

    # Major confusions on the fused ensemble
    major = []
    for ti in range(6):
        for pj in range(6):
            if ti == pj: continue
            n = int(cm_ens[ti, pj])
            if n >= 2:
                major.append({
                    'true_class': CLASS_TOMATO[ti],
                    'pred_class': CLASS_TOMATO[pj],
                    'count': n,
                })
    major.sort(key=lambda x: -x['count'])

    print(f'  V3        n={len(t_v3):3d} · acc={accuracy_score(t_v3, p_v3):.4f}'
          f' · macroF1={f1_score(t_v3, p_v3, average="macro"):.4f}')
    print(f'  SP-LoRA   n={len(t_lora):3d} · acc={accuracy_score(t_lora, p_lora):.4f}'
          f' · macroF1={f1_score(t_lora, p_lora, average="macro"):.4f}')
    print(f'  ENSEMBLE  n={len(fused_true):3d} · acc={accuracy_score(fused_true, fused_pred_list):.4f}'
          f' · macroF1={f1_score(fused_true, fused_pred_list, average="macro"):.4f}')
    return {
        'class_order':       CLASS_TOMATO,
        'evaluation_subset': 'final_val (locked tomato test split)',
        'n_test_images':     int(len(fused_true)),
        'ensemble': {
            'confusion_matrix': [[int(v) for v in row] for row in cm_ens.tolist()],
            'overall_accuracy': round(float(accuracy_score(fused_true, fused_pred_list)), 4),
            'overall_macro_f1': round(float(f1_score(fused_true, fused_pred_list, average='macro')), 4),
            'per_class_metrics': per_class(fused_true, fused_pred_list),
            'major_confusions':  major,
            'fusion':            '0.5 * V3(T=0.5) + 0.5 * SP-LoRA(T=1.0) · per-class average',
        },
        'v3_branch': {
            'confusion_matrix': [[int(v) for v in row] for row in cm_v3.tolist()],
            'overall_accuracy': round(float(accuracy_score(t_v3, p_v3)), 4),
            'overall_macro_f1': round(float(f1_score(t_v3, p_v3, average='macro')), 4),
            'per_class_metrics': per_class(t_v3, p_v3),
            'n':                 int(len(t_v3)),
        },
        'sp_lora_branch': {
            'confusion_matrix': [[int(v) for v in row] for row in cm_lora.tolist()],
            'overall_accuracy': round(float(accuracy_score(t_lora, p_lora)), 4),
            'overall_macro_f1': round(float(f1_score(t_lora, p_lora, average='macro')), 4),
            'per_class_metrics': per_class(t_lora, p_lora),
            'n':                 int(len(t_lora)),
        },
    }


# ── M2 · per-block CLS evolution ───────────────────────────────────────
def extract_tomato_cls_evolution():
    """Load DINOv2-Reg backbone and capture the [CLS] token at every
    transformer block for the 3 canonical tomato pool images (the same
    pool used by the Forward Pass marquee). Summarise each 384-D CLS
    vector by L2 norm + top-3 dim activations for visualization."""
    print('M2 · per-block CLS evolution ...')
    import torch
    import timm
    from PIL import Image
    from torchvision import transforms

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone_name = 'vit_small_patch14_reg4_dinov2.lvd142m'
    img_size = 224

    # Locate the 3 canonical tomato pool images. Forward_pass extraction used
    # cleaned tomato images; reuse the same pool.
    pool_dir = os.path.join(ROOT, 'data/specialist/model3/cleaned')
    candidates = []
    for cls_name in ['tomato_late_blight', 'tomato_septoria_leaf_spot', 'tomato_healthy']:
        d = os.path.join(pool_dir, cls_name)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(('.jpg', '.png', '.jpeg')):
                candidates.append((cls_name, os.path.join(d, fn)))
                break  # one per class
    pool = candidates[:3]
    if not pool:
        print('  pool images not found · skipping CLS evolution')
        return {'pending': True, 'reason': 'tomato pool images not found at data/specialist/model3/cleaned/<class>/'}
    print(f'  pool: {[p[0] for p in pool]}')

    # Build the backbone
    backbone = timm.create_model(backbone_name, pretrained=True,
                                  num_classes=0, img_size=img_size).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Hook every transformer block; we'll grab the [CLS] (token 0) from each output
    blocks = backbone.blocks
    print(f'  backbone has {len(blocks)} transformer blocks')
    captured = []
    handles = []
    for blk in blocks:
        def make_hook(slot):
            def hook(module, inp, out):
                # out shape: [B, n_tokens, dim] where token 0 is CLS
                cls = out[:, 0, :].detach()  # [B, dim]
                captured.append(cls.cpu())
            return hook
        handles.append(blk.register_forward_hook(make_hook(len(captured))))

    # Preprocess
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    images = []
    for cls_name, path in pool:
        img = Image.open(path).convert('RGB')
        images.append(tf(img))
    batch = torch.stack(images).to(device)
    print(f'  forward pass on batch of {len(images)} (img_size={img_size}px) ...')

    captured.clear()
    with torch.no_grad():
        _ = backbone(batch)
    for h in handles:
        h.remove()

    # captured is a list of [B, dim] tensors, one per block
    n_blocks = len(captured)
    print(f'  captured {n_blocks} CLS tokens per image')

    images_out = []
    for img_idx, (cls_name, path) in enumerate(pool):
        per_block = []
        prev_cls = None
        for blk_idx in range(n_blocks):
            cls = captured[blk_idx][img_idx].numpy()  # [384]
            l2 = float(np.linalg.norm(cls))
            top_dims = np.argsort(-np.abs(cls))[:3].tolist()
            top_vals = [round(float(cls[d]), 4) for d in top_dims]
            # Cosine similarity with the previous block's CLS
            if prev_cls is not None:
                denom = np.linalg.norm(cls) * np.linalg.norm(prev_cls) + 1e-9
                cos_prev = float(np.dot(cls, prev_cls) / denom)
            else:
                cos_prev = None
            per_block.append({
                'block':       blk_idx,
                'l2_norm':     round(l2, 4),
                'top_dims':    top_dims,
                'top_vals':    top_vals,
                'cos_prev':    round(cos_prev, 4) if cos_prev is not None else None,
                'mean':        round(float(cls.mean()), 5),
                'std':         round(float(cls.std()), 5),
            })
            prev_cls = cls
        images_out.append({
            'pool_index': img_idx,
            'true_class': cls_name,
            'image_path': os.path.relpath(path, ROOT).replace('\\', '/'),
            'per_block':  per_block,
        })

    return {
        'backbone':      backbone_name,
        'img_size':      img_size,
        'n_blocks':      n_blocks,
        'n_pool_images': len(pool),
        'images':        images_out,
        'note':          'CLS token (index 0) extracted via forward hook per transformer block.',
    }


# ── EXCLUDED Sankey node · scan source_map.csv ─────────────────────────
def extract_excluded():
    """Count excluded rows in source_map.csv (if any). Currently 0 in the
    deployed pipeline; the node renders zero-height by default."""
    print('Sankey · excluded scan ...')
    src = os.path.join(ROOT, 'data/metadata/source_map.csv')
    if not os.path.isfile(src):
        return {'excluded_total': 0, 'reason': 'source_map.csv not found'}
    excluded_total = 0
    excluded_by_reason = defaultdict(int)
    excluded_by_source = defaultdict(int)
    with open(src, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            split = (r.get('split') or '').strip()
            if split in ('excluded', 'dropped', 'rejected'):
                excluded_total += 1
                excluded_by_reason[r.get('exclude_reason', 'unknown')] += 1
                excluded_by_source[r.get('source_dataset', 'unknown')] += 1
    print(f'  excluded rows: {excluded_total}')
    return {
        'excluded_total':     int(excluded_total),
        'excluded_by_reason': dict(excluded_by_reason),
        'excluded_by_source': dict(excluded_by_source),
        'note':               ('zero excluded rows in current source_map.csv; '
                                'EXCLUDED Sankey node renders zero-height. '
                                'Will render automatically when future deduplication '
                                'or quality filters produce exclusions.'),
    }


# ── orchestrator ───────────────────────────────────────────────────────
def main():
    import datetime
    out = {
        'produced_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'produced_by':     'scripts/apin_v2/extract_phase_d_scattered.py',
        'router':          extract_router_confusion(),
        'tomato_confusion': extract_tomato_confusion(),
        'tomato_cls_evolution': extract_tomato_cls_evolution(),
        'excluded':        extract_excluded(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    sz = os.path.getsize(OUT) / 1024
    print(f'\nWrote {OUT}  ·  {sz:.1f} KB')


if __name__ == '__main__':
    main()
